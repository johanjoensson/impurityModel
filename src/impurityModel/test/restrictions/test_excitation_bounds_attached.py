"""The thermal ground-state solve attaches the bath-excitation bounds it observed to the returned basis."""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import product_state_representation as psr
from impurityModel.ed.gs_calc import calc_gs

from .test_excitation_budget import _siam_with_baths

_VAL, _CON = (2, 3), (4, 5)


def _run(weighted_restrictions, comm=None):
    Hop, imp, bath, bs = _siam_with_baths()
    setup = dict(
        impurity_orbitals=imp,
        bath_states=bath,
        N0={0: 1},
        tau=0.01,
        dense_cutoff=1000,
        comm=comm,
        truncation_threshold=100000,
        weighted_restrictions=weighted_restrictions,
    )
    return calc_gs(Hop, setup, bs, np.eye(2, dtype=complex), verbose=False, slaterWeightMin=1e-12)


def _counts(det):
    occ = {k for k, bit in enumerate(psr.bytes2bitarray(bytes(det.to_bytearray()), 6)) if bit}
    holes, electrons = len(set(_VAL) - occ), len(occ & set(_CON))
    return holes, electrons, holes + electrons


def test_calc_gs_attaches_the_observed_bounds_and_leaves_the_callers_list_alone():
    caller = [({0: 1, 1: 1}, (0, 2))]  # an unrelated, always-satisfied entry
    before = list(caller)
    _psis, _es, basis, _rho, info = _run(caller)
    assert caller == before and basis.weighted_restrictions is not caller  # rebound, not mutated
    observed = [_counts(d) for d in basis.local_basis]
    bounds = basis.excitation_bounds
    assert (bounds.max_holes, bounds.max_electrons, bounds.max_total) == tuple(
        max(c[i] for c in observed) for i in range(3)
    )
    assert basis.weighted_restrictions[0] == caller[0]
    assert len(basis.weighted_restrictions) == 1 + 3
    # the JSON record and the result dict carry the same ints
    (rec,) = info["excitation_bounds"]
    assert (rec["max_holes"], rec["max_electrons"], rec["max_total"]) == (
        bounds.max_holes,
        bounds.max_electrons,
        bounds.max_total,
    )
    assert info["statistics"] is None or info["statistics"]["excitation_bounds"] == info["excitation_bounds"]


def test_the_attached_entries_admit_every_ground_state_determinant():
    _psis, _es, basis, _rho, _info = _run(None)
    entries = basis.weighted_restrictions
    assert len(entries) == 3
    for det in basis.local_basis:
        occ = {k for k, bit in enumerate(psr.bytes2bitarray(bytes(det.to_bytearray()), 6)) if bit}
        for weights, (q_min, q_max) in entries:
            assert q_min <= sum(w for o, w in weights.items() if o in occ) <= q_max


def test_the_excited_window_inherits_the_bounds_plus_one():
    """Through the production step (``greens_function._build_excited_restrictions``, the one every GF unit and
    the susceptibility driver use), not through a re-implementation of ``widen(floor(...))``."""
    from impurityModel.ed.greens_function import _build_excited_restrictions

    Hop, _imp, _bath, _bs = _siam_with_baths()
    psis, es, basis, _rho, _info = _run(None)
    b = basis.excitation_bounds
    _res, excited = _build_excited_restrictions(basis, Hop, psis, es, dN=1, occ_cutoff=1e-6)
    ground = basis.weighted_restrictions
    assert len(ground) == len(excited) == 3
    for (w0, (lo0, hi0), *_), (w1, (lo1, hi1), *_) in zip(ground, excited):
        assert w0 == w1 and (lo1, hi1) == (lo0 - 1, hi0 + 1)
    # In counts: one more hole, one more electron, one more in total than the ground state ever needed.
    holes, electrons, total = (b.max_holes + 1, b.max_electrons + 1, b.max_total + 1)
    for (weights, (_lo, hi)), bound in zip(excited, (holes, electrons, total)):
        n_val = len(_VAL)
        offset = n_val if sum(weights.values()) < 0 else 0  # valence entries count holes as q + n_val
        if len(weights) == len(_VAL) + len(_CON):
            offset = n_val
        assert hi + offset == bound


@pytest.mark.mpi
def test_bounds_agree_across_ranks_and_with_the_serial_run():
    _, _, serial, _, _ = _run(None)
    _, _, basis, _, _ = _run(None, comm=MPI.COMM_WORLD)
    b = basis.excitation_bounds
    assert (b.max_holes, b.max_electrons, b.max_total) == (
        serial.excitation_bounds.max_holes,
        serial.excitation_bounds.max_electrons,
        serial.excitation_bounds.max_total,
    )
    assert len(set(MPI.COMM_WORLD.allgather((b.max_holes, b.max_electrons, b.max_total)))) == 1
