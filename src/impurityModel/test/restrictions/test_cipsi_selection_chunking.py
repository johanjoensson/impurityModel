"""Exactness of the chunked CIPSI selection score (``GS_SELECTION_CHUNK``,
``cipsi_solver._score_candidates``).

The manifold-summed Epstein-Nesbet score used to be computed by materializing the whole
``(p, n_Dj)`` de2 array in one shot (``np.max(np.stack([de2_abs[g, :].sum(axis=0) for g in
groups]))``). ``_score_candidates`` computes the identical quantity in group-aligned batches over
the reference (``p``) axis, so the ``(batch_width, n_Dj)`` temporaries it needs at any one time
never span the whole ``p`` -- see ``doc/plans/dc_smo_memory.md`` for why (the crashed SrMnO3
double-counting search this exists to fix). The chunking is sound only because a batch boundary
never splits a degenerate group; this file is the direct check that it does not, at every group
layout a small system can produce and, separately, through the actual ``CIPSISolver.determine_new_Dj``
call path with ``GS_SELECTION_CHUNK`` set.
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.cipsi_solver import CIPSISolver, _degenerate_groups, _score_candidates
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant

IMPURITY_ORBITALS = {0: [[0, 1]]}
BATH_STATES = ({0: [[2, 3]]}, {0: [[4, 5]]})
N_SPIN_ORBITALS = 6
N_ELECTRONS = 3


def _reference_score(overlaps, e_ref, e_Dj, groups):
    """The pre-chunking formula, kept here verbatim as the thing the chunked version must
    reproduce (not re-derived from the same source `_score_candidates` now uses)."""
    de = np.maximum(np.abs(e_ref[:, None] - e_Dj[None, :]), 1e-12)
    de2 = np.zeros(overlaps.shape, dtype=float)
    mask = np.abs(overlaps) > 1e-12
    de2[mask] = np.square(np.abs(overlaps[mask])) / de[mask]
    de2_abs = np.abs(de2)
    return np.max(np.stack([de2_abs[g, :].sum(axis=0) for g in groups]), axis=0)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("chunk_size", [None, 1, 2, 3, 5, 1000])
def test_score_candidates_matches_the_unchunked_formula(seed, chunk_size):
    rng = np.random.default_rng(seed)
    p, n_Dj = 11, 37
    e_ref = np.sort(rng.normal(size=p))
    # force a few exact degeneracies, at different offsets per seed, so chunk boundaries are
    # exercised against manifolds of varying width and position
    e_ref[(seed + 1) % p] = e_ref[seed % p]
    e_ref[(seed + 4) % p] = e_ref[(seed + 3) % p] = e_ref[(seed + 2) % p]
    overlaps = rng.normal(size=(p, n_Dj)) + 1j * rng.normal(size=(p, n_Dj))
    e_Dj = rng.normal(size=n_Dj) * 5
    groups = _degenerate_groups(np.sort(e_ref), tol=1e-9)
    e_ref = np.sort(e_ref)  # _degenerate_groups assumes ascending input, like production callers

    expected = _reference_score(overlaps, e_ref, e_Dj, groups)
    actual = _score_candidates(overlaps, e_ref, e_Dj, groups, chunk_size=chunk_size)
    np.testing.assert_array_equal(actual, expected)


def test_score_candidates_handles_empty_candidates():
    e_ref = np.array([0.0, 1.0])
    groups = _degenerate_groups(e_ref, tol=1e-9)
    overlaps = np.zeros((2, 0), dtype=complex)
    e_Dj = np.zeros(0)
    for chunk_size in (None, 1, 5):
        scores = _score_candidates(overlaps, e_ref, e_Dj, groups, chunk_size=chunk_size)
        assert scores.shape == (0,)


def _det(occupied):
    """SlaterDeterminant with the given orbitals occupied (MSB-first bit convention)."""
    chunk = 0
    for orb in occupied:
        chunk |= 1 << (63 - orb)
    return SlaterDeterminant((chunk,))


def _make_solver(comm):
    basis = Basis(IMPURITY_ORBITALS, BATH_STATES, nominal_impurity_occ={0: 1}, comm=comm, verbose=False)
    all_dets = [_det(occ) for occ in itertools.combinations(range(N_SPIN_ORBITALS), N_ELECTRONS)]
    basis.add_states(all_dets[:6])
    return CIPSISolver(basis), all_dets[:6]


def _hamiltonian():
    # Two on-site pairs tuned equal so the two-state eigen-block is exactly degenerate --
    # `determine_new_Dj`'s manifold-sum path only actually differs from a naive max when a
    # real degeneracy reaches it.
    terms = {
        ((0, "c"), (0, "a")): 0.2,
        ((1, "c"), (1, "a")): 0.2,
        ((2, "c"), (2, "a")): 0.9,
        ((3, "c"), (3, "a")): 1.3,
        ((4, "c"), (4, "a")): 1.7,
        ((5, "c"), (5, "a")): 2.1,
    }
    for a, b in ((0, 2), (1, 3), (2, 4), (3, 5), (0, 4), (1, 5)):
        terms[((a, "c"), (b, "a"))] = 0.15 + 0.05j
        terms[((b, "c"), (a, "a"))] = 0.15 - 0.05j
    return ManyBodyOperator(terms)


def _psi_ref_and_e_ref(basis_dets):
    psi_ref = [ManyBodyState(dict.fromkeys(basis_dets[: w + 1], 1.0 + 0.3j)) for w in (1, 2, 3, 4)]
    e_ref = np.array([-0.3, -0.3, 0.5, 1.1])  # a degenerate pair, matching psi_ref's width
    return psi_ref, e_ref


@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_determine_new_dj_is_unaffected_by_gs_selection_chunk_serial(chunk_size, monkeypatch):
    """End-to-end: the actual selection path (through `_apply_block_and_redistribute` and
    `_candidate_overlaps_and_energies`, not the isolated formula above) must admit the same
    determinants regardless of GS_SELECTION_CHUNK. Both calls run inside this one test so the
    comparison never depends on parametrization order."""
    solver, basis_dets = _make_solver(None)
    H = _hamiltonian()
    psi_ref, e_ref = _psi_ref_and_e_ref(basis_dets)

    monkeypatch.delenv("GS_SELECTION_CHUNK", raising=False)
    reference = solver.determine_new_Dj(e_ref, psi_ref, H, de2_min=1e-6)
    assert len(reference) > 0, "premise: this Hamiltonian must produce admissible candidates"

    monkeypatch.setenv("GS_SELECTION_CHUNK", str(chunk_size))
    chunked = solver.determine_new_Dj(e_ref, psi_ref, H, de2_min=1e-6)
    assert chunked == reference


@pytest.mark.mpi
@pytest.mark.parametrize("chunk_size", [1, 3])
def test_determine_new_dj_is_unaffected_by_gs_selection_chunk_mpi(chunk_size, monkeypatch):
    """Same end-to-end check, distributed -- run at -n 2 and -n 3."""
    comm = MPI.COMM_WORLD
    solver, basis_dets = _make_solver(comm)
    H = _hamiltonian()
    psi_ref, e_ref = _psi_ref_and_e_ref(basis_dets)

    monkeypatch.delenv("GS_SELECTION_CHUNK", raising=False)
    reference = solver.determine_new_Dj(e_ref, psi_ref, H, de2_min=1e-6)

    monkeypatch.setenv("GS_SELECTION_CHUNK", str(chunk_size))
    chunked = solver.determine_new_Dj(e_ref, psi_ref, H, de2_min=1e-6)
    assert chunked == reference
