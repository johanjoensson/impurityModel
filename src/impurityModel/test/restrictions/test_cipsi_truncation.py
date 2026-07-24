"""Tests for CIPSI basis truncation (top-K amplitude selection + capped expansion)."""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.groundstate import calc_gs
from impurityModel.ed.manybody_basis import Basis, collective_amplitude_cutoff
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, SlaterDeterminant
from impurityModel.ed.ManyBodyUtils import applyOp as applyOp_test

IMPURITY_ORBITALS = {0: [[0, 1]]}
BATH_STATES = ({0: [[2, 3]]}, {0: [[4, 5]]})
N_SPIN_ORBITALS = 6


def _det(occupied):
    """SlaterDeterminant with the given orbitals occupied (MSB-first bit convention)."""
    chunk = 0
    for orb in occupied:
        chunk |= 1 << (63 - orb)
    return SlaterDeterminant((chunk,))


def _four_electron_dets(n):
    """The first n four-electron determinants in 6 spin-orbitals."""
    dets = [_det(occ) for occ in itertools.combinations(range(N_SPIN_ORBITALS), 4)]
    assert n <= len(dets)
    return dets[:n]


def _redistribute_single_as_width1(basis, psi):
    """Redistribute a single seed (populated on one rank, an empty placeholder on the
    others) through an explicit width-1 block, then unpack back to a flat state.

    A bare ``ManyBodyState({})`` placeholder is the width-0 polymorphic zero once the
    flat and block classes merge (Phase 7 step 3) -- an asymmetric mismatch against the
    populated (eventually width-1) seed on the owning rank that would deadlock
    redistribute_psis' collective. from_states forces an explicit width-1 block on
    every rank instead."""
    (out,) = basis.redistribute_psis(ManyBodyState.from_states([psi]))[0].to_states()
    return out


def _make_basis(comm, truncation_threshold=np.inf):
    return Basis(
        IMPURITY_ORBITALS,
        BATH_STATES,
        nominal_impurity_occ={0: 2},
        truncation_threshold=truncation_threshold,
        comm=comm,
        verbose=False,
    )


def _all_retained(basis):
    """Global sorted list of the basis determinants (identical on every rank)."""
    local = list(basis.local_basis)
    if basis.is_distributed:
        gathered = basis.comm.allgather(local)
        return sorted(det for part in gathered for det in part)
    return sorted(local)


# ---------------------------------------------------------------------------
# collective_amplitude_cutoff
# ---------------------------------------------------------------------------


def test_collective_cutoff_serial_top_k():
    scores = np.array([0.1, 0.7, 0.3, 0.9, 0.5])
    cutoff = collective_amplitude_cutoff(scores, 2, None)
    assert np.count_nonzero(scores > cutoff) == 2
    assert set(scores[scores > cutoff]) == {0.9, 0.7}


def test_collective_cutoff_underfull_keeps_all_nonzero():
    scores = np.array([0.4, 0.2, 0.0])
    cutoff = collective_amplitude_cutoff(scores, 10, None)
    assert np.count_nonzero(scores > cutoff) == 2  # exact zeros are never "above"


def test_collective_cutoff_empty_scores():
    assert collective_amplitude_cutoff(np.array([]), 3, None) == 0.0


@pytest.mark.mpi
def test_collective_cutoff_mpi_agrees_and_caps():
    comm = MPI.COMM_WORLD
    # Three distinct scores per rank so the global pool always has >= 3 values (the top-3
    # request stays well-defined even at -n 1); interleaved so the top 3 span >= 2 ranks
    # when size > 1. Values are 1 .. 3*size with no duplicates across ranks.
    scores = np.array([float(comm.rank + 1 + comm.size * i) for i in range(3)])
    cutoff = collective_amplitude_cutoff(scores, 3, comm)
    assert all(c == cutoff for c in comm.allgather(cutoff))
    n_above = comm.allreduce(int(np.count_nonzero(scores > cutoff)), op=MPI.SUM)
    assert n_above == 3


# ---------------------------------------------------------------------------
# CIPSISolver.truncate
# ---------------------------------------------------------------------------


def test_truncate_keeps_top_k_amplitudes():
    basis = _make_basis(comm=None)
    dets = _four_electron_dets(8)
    amps = np.arange(1.0, 9.0)  # importance |amp|^2 strictly increasing
    basis.clear()
    basis.add_states(dets)
    solver = CIPSISolver(basis)

    psi = ManyBodyState(dict(zip(dets, amps)))
    psis = solver.truncate([psi], target=3)

    assert basis.size == 3
    assert _all_retained(basis) == sorted(dets[-3:])
    # The returned psis are filtered to the retained set with amplitudes intact.
    kept = {det: amp for p in psis for det, amp in p.items()}
    assert kept == dict(zip(dets[-3:], amps[-3:]))


def test_truncate_importance_is_max_over_psis():
    """A determinant large in *any* eigenvector must be retained."""
    basis = _make_basis(comm=None)
    dets = _four_electron_dets(4)
    basis.clear()
    basis.add_states(dets)
    solver = CIPSISolver(basis)

    psi0 = ManyBodyState({dets[0]: 1.0, dets[1]: 0.01, dets[2]: 0.02, dets[3]: 0.03})
    psi1 = ManyBodyState({dets[0]: 0.01, dets[1]: 0.9, dets[2]: 0.02, dets[3]: 0.03})
    solver.truncate([psi0, psi1], target=2)

    assert _all_retained(basis) == sorted(dets[:2])


def test_truncate_default_target_is_threshold():
    basis = _make_basis(comm=None, truncation_threshold=5)
    dets = _four_electron_dets(9)
    basis.clear()
    basis.add_states(dets)
    solver = CIPSISolver(basis)

    psi = ManyBodyState({det: float(i + 1) for i, det in enumerate(dets)})
    solver.truncate([psi])

    assert basis.size == 5
    assert _all_retained(basis) == sorted(dets[-5:])


def test_truncate_fills_budget_not_decades_below():
    """Amplitudes spanning many decades: the ladder would overshoot, top-K must not."""
    basis = _make_basis(comm=None)
    dets = _four_electron_dets(10)
    basis.clear()
    basis.add_states(dets)
    solver = CIPSISolver(basis)

    # One dominant amplitude and nine tiny ones spread over decades: a x10 cutoff
    # ladder starting at eps would jump straight past most of them.
    amps = [1.0] + [10.0**-k for k in range(3, 12)]
    psi = ManyBodyState(dict(zip(dets, amps)))
    solver.truncate([psi], target=7)

    assert basis.size == 7
    assert _all_retained(basis) == sorted(dets[:7])


def test_truncate_drops_unsupported_determinants():
    """Basis states with zero amplitude in every psi are never retained."""
    basis = _make_basis(comm=None)
    dets = _four_electron_dets(6)
    basis.clear()
    basis.add_states(dets)
    solver = CIPSISolver(basis)

    psi = ManyBodyState({dets[0]: 0.5, dets[1]: 0.4})  # dets[2:] unsupported
    solver.truncate([psi], target=4)

    assert _all_retained(basis) == sorted(dets[:2])


@pytest.mark.mpi
def test_truncate_top_k_mpi():
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("needs >= 2 ranks")
    basis = _make_basis(comm=comm)
    dets = _four_electron_dets(12)
    basis.clear()
    basis.add_states(dets if comm.rank == 0 else [])
    solver = CIPSISolver(basis)

    # Amplitudes seeded on rank 0 only; redistribute puts each det on its owner rank.
    psi = ManyBodyState({det: float(i + 1) for i, det in enumerate(dets)} if comm.rank == 0 else {}, width=1)
    psis = [_redistribute_single_as_width1(basis, psi)]
    psis = solver.truncate(psis, target=4)

    assert basis.size == 4
    assert _all_retained(basis) == sorted(dets[-4:])
    # Each retained det lives on exactly one rank with its original amplitude.
    kept_local = {det: amp[0] for p in psis for det, amp in p.items()}
    gathered = comm.allgather(kept_local)
    kept = {}
    for part in gathered:
        assert not (kept.keys() & part.keys())
        kept.update(part)
    assert kept == {det: float(i + 1) for i, det in enumerate(dets) if i >= 8}


@pytest.mark.mpi
def test_truncate_rank_may_retain_zero_rows_mpi():
    """target=1 at >=2 ranks: most ranks own nothing retained; collectives must not hang."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("needs >= 2 ranks")
    basis = _make_basis(comm=comm)
    dets = _four_electron_dets(6)
    basis.clear()
    basis.add_states(dets if comm.rank == 0 else [])
    solver = CIPSISolver(basis)

    psi = ManyBodyState({det: float(i + 1) for i, det in enumerate(dets)} if comm.rank == 0 else {}, width=1)
    psis = [_redistribute_single_as_width1(basis, psi)]
    solver.truncate(psis, target=1)

    assert basis.size == 1
    assert _all_retained(basis) == [dets[-1]]


# ---------------------------------------------------------------------------
# Fixed-budget capped expansion
# ---------------------------------------------------------------------------


def _siam_operator():
    """Single-impurity Anderson model, 6 spin-orbitals (see test_sectorization)."""
    ed_, u, ev, ec, v = -1.0, 4.0, -3.0, 3.0, 0.5
    terms = {}
    for o in (0, 1):
        terms[((o, "c"), (o, "a"))] = ed_
    for o in (2, 3):
        terms[((o, "c"), (o, "a"))] = ev
    for o in (4, 5):
        terms[((o, "c"), (o, "a"))] = ec
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = u
    for a, b in ((0, 2), (1, 3), (0, 4), (1, 5)):
        terms[((a, "c"), (b, "a"))] = v
        terms[((b, "c"), (a, "a"))] = v
    return ManyBodyOperator(terms)


def _expanded_e0(threshold, comm=None):
    """(E0, basis size, truncation_report) of a capped CIPSI expansion on the SIAM."""
    basis = Basis(
        IMPURITY_ORBITALS,
        BATH_STATES,
        nominal_impurity_occ={0: 2},
        truncation_threshold=threshold,
        tau=0.01,
        comm=comm,
        verbose=False,
        delta_impurity_occ={0: 2},
        delta_valence_occ={0: 2},
        delta_conduction_occ={0: 2},
    )
    solver = CIPSISolver(basis)
    solver.expand(_siam_operator(), de2_min=1e-8, dense_cutoff=1000)
    es, _ = solver.get_eigenvectors(_siam_operator(), num_wanted=2, max_energy=1.0, dense_cutoff=1000)
    return float(np.min(es)), basis.size, solver.truncation_report


def test_capped_expand_monotone_and_variational():
    thresholds = [2, 3, 4, 6, 8, 12]
    e0_ref, natural_size, report_ref = _expanded_e0(np.inf)
    assert report_ref is None
    prev_e0 = np.inf
    for threshold in thresholds:
        e0, size, report = _expanded_e0(threshold)
        # The capped solve is variational and the cap is a hard bound that is filled.
        assert e0 >= e0_ref - 1e-9
        assert size <= threshold
        assert size == min(threshold, natural_size)
        # Growing the budget never hurts (fixed-budget refinement keeps the best basis).
        assert e0 <= prev_e0 + 1e-9
        assert report is not None and report["cap_hit"]
        assert report["cycles"] <= 10
        prev_e0 = e0
    # A cap at (or above) the natural size is exactly free.
    e0_full, size_full, _ = _expanded_e0(natural_size)
    assert size_full == natural_size
    assert abs(e0_full - e0_ref) < 1e-9


def test_capped_expand_no_report_when_cap_never_binds():
    _, size, report = _expanded_e0(10**6)
    assert report is None
    assert size < 10**6


@pytest.mark.mpi
def test_capped_expand_mpi_matches_serial():
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("needs >= 2 ranks")
    e0_capped, size, report = _expanded_e0(6, comm=comm)
    assert size == 6 and report["cap_hit"]
    # Same physics as the serial capped run (near-ties may differ, energies must not).
    e0_serial = -4.7813537278
    assert abs(e0_capped - e0_serial) < 1e-6
    # Every rank eigensolves the same Allreduced data, but potentially through a
    # different LAPACK build -- cross-rank agreement, not last-ulp identity.
    cross_rank_tol = 1e-10 * max(1.0, abs(e0_capped))
    assert all(abs(v - e0_capped) < cross_rank_tol for v in comm.allgather(e0_capped))


# ---------------------------------------------------------------------------
# de2 (Epstein-Nesbet PT2) selection score
# ---------------------------------------------------------------------------


def test_de2_denominator_uses_energy_gap_for_candidates_above_ref():
    """de2 = |<Dj|H|psi>|^2 / |E_ref - E_Dj|, finite and gap-weighted for E_Dj > E_ref.

    The historical bug clamped ``de = max(E_ref - E_Dj, 1e-12)``; since ground-state
    candidates sit above E_ref the denominator collapsed to the clamp, making every de2
    an identical ~1e12*|coupling|^2 -- a bare coupling filter blind to the energy gap.
    """
    d0, d1, d2 = _det([0]), _det([1]), _det([2])
    e0, e1, e2, v = 0.0, 1.0, 2.0, 0.3
    # Diagonal energies via number operators; the reference is the only basis state, so
    # d1 and d2 are the "new" candidates. No d1<->d2 coupling => e_Dj is exact.
    H = ManyBodyOperator(
        {
            ((0, "c"), (0, "a")): e0,
            ((1, "c"), (1, "a")): e1,
            ((2, "c"), (2, "a")): e2,
        }
    )
    basis = _make_basis(comm=None)
    basis.clear()
    basis.add_states([d0])
    solver = CIPSISolver(basis)

    hpsi = ManyBodyState({d1: v, d2: v})  # H|psi_ref> projected onto the candidates
    local_Djs, de2 = solver._calc_de2(H, [hpsi], np.array([e0]))
    score = {det: float(np.abs(de2[0, j])) for j, det in enumerate(local_Djs)}

    # Finite and set by the gap: closer candidate (|dE|=1) scores twice the farther (|dE|=2).
    assert score[d1] == pytest.approx(v**2 / abs(e0 - e1), rel=1e-9)
    assert score[d2] == pytest.approx(v**2 / abs(e0 - e2), rel=1e-9)
    assert score[d1] == pytest.approx(2.0 * score[d2], rel=1e-9)
    # The broken clamp would have made both ~1e12 * v^2 and equal.
    assert score[d1] < 1.0


def _oracle_candidate_overlaps_and_energies(solver, H, hpsi_states, slaterWeightMin=0.0):
    """Independent reimplementation of the pre-Phase-9-Step-2 per-state algorithm.

    ``hpsi_states`` is a plain ``list[ManyBodyState]`` (width-1 states, the historical
    representation): builds ``local_Djs`` via the sorted-set-comprehension union and the
    ``overlaps`` matrix via a nested ``.items()`` loop, exactly as
    ``_candidate_overlaps_and_energies`` did before it was rewritten to read the merged
    block through the buffer protocol. Used as a ground truth the new implementation must
    match bit-for-bit -- the block-native rewrite touches ordering and zero-row handling,
    exactly the kind of change a green test suite alone would not catch (see Phase 7/9).
    """
    _index_dict = solver.basis._index_dict
    local_Djs = sorted({state for hp in hpsi_states for state in hp if state not in _index_dict})
    if not local_Djs:
        return local_Djs, np.zeros((len(hpsi_states), 0), dtype=complex)
    Dj_index = {Dj: j for j, Dj in enumerate(local_Djs)}
    overlaps = np.zeros((len(hpsi_states), len(local_Djs)), dtype=complex)
    for i, Hpsi_i in enumerate(hpsi_states):
        for state, amp in Hpsi_i.items():
            j = Dj_index.get(state)
            if j is not None:
                overlaps[i, j] = amp[0]
    return local_Djs, overlaps


def test_candidate_overlaps_block_matches_oracle_serial():
    """The block-native ``_candidate_overlaps_and_energies`` reproduces the old
    per-state algorithm bit-for-bit on a serial (non-distributed) round."""
    d0, d1 = _det([0]), _det([1])
    H = ManyBodyOperator(
        {
            ((0, "c"), (0, "a")): 0.0,
            ((1, "c"), (1, "a")): 1.0,
            ((2, "c"), (2, "a")): 2.0,
            ((3, "c"), (3, "a")): 3.0,
            ((0, "c"), (1, "a")): 0.3,
            ((1, "c"), (0, "a")): 0.3,
            ((0, "c"), (2, "a")): 0.2,
            ((2, "c"), (0, "a")): 0.2,
            ((1, "c"), (3, "a")): 0.1,
            ((3, "c"), (1, "a")): 0.1,
        }
    )
    basis = _make_basis(comm=None)
    basis.clear()
    basis.add_states([d0, d1])
    solver = CIPSISolver(basis)

    psi_ref = [ManyBodyState({d0: 1.0}), ManyBodyState({d1: 1.0})]
    Hpsi_blk = solver._apply_block_and_redistribute(H, psi_ref, 0.0)
    assert isinstance(Hpsi_blk, ManyBodyState)

    local_Djs, overlaps, e_Dj = solver._candidate_overlaps_and_energies(H, Hpsi_blk, 0.0)

    # `_apply_block_and_redistribute` applies H first, so the oracle input is H|psi_i>
    # per reference column, not the raw psi_ref.
    hpsi_states_oracle = [applyOp_test(H, p, cutoff=0.0) for p in psi_ref]
    oracle_Djs, oracle_overlaps = _oracle_candidate_overlaps_and_energies(solver, H, hpsi_states_oracle, 0.0)

    assert local_Djs == oracle_Djs
    assert local_Djs == sorted(local_Djs)
    np.testing.assert_array_equal(overlaps, oracle_overlaps)
    assert e_Dj.shape == (len(local_Djs),)


@pytest.mark.mpi
def test_candidate_overlaps_block_matches_oracle_mpi():
    """The distributed counterpart: block-native path vs. the per-state oracle on a
    real multi-rank round, after a real ``redistribute_block`` (not the non-distributed
    identity branch) -- exercises the row ordering / zero-row-drop equivalence that only
    shows up once determinants actually move across ranks."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("needs >= 2 ranks")

    d0, d1 = _det([0]), _det([1])
    H = ManyBodyOperator(
        {
            ((0, "c"), (0, "a")): 0.0,
            ((1, "c"), (1, "a")): 1.0,
            ((2, "c"), (2, "a")): 2.0,
            ((3, "c"), (3, "a")): 3.0,
            ((4, "c"), (4, "a")): 4.0,
            ((0, "c"), (1, "a")): 0.3,
            ((1, "c"), (0, "a")): 0.3,
            ((0, "c"), (2, "a")): 0.2,
            ((2, "c"), (0, "a")): 0.2,
            ((1, "c"), (3, "a")): 0.1,
            ((3, "c"), (1, "a")): 0.1,
            ((1, "c"), (4, "a")): 0.15,
            ((4, "c"), (1, "a")): 0.15,
        }
    )
    basis = _make_basis(comm=comm)
    basis.clear()
    basis.add_states([d0, d1] if comm.rank == 0 else [])
    solver = CIPSISolver(basis)

    psi0 = ManyBodyState({d0: 1.0} if comm.rank == 0 else {}, width=1)
    psi1 = ManyBodyState({d1: 1.0} if comm.rank == 0 else {}, width=1)
    psi_ref = list(basis.redistribute_psis(psi0, psi1))

    Hpsi_blk = solver._apply_block_and_redistribute(H, psi_ref, 0.0)
    assert isinstance(Hpsi_blk, ManyBodyState)
    local_Djs, overlaps, e_Dj = solver._candidate_overlaps_and_energies(H, Hpsi_blk, 0.0)

    # Oracle: apply H to each reference column locally, redistribute exactly as
    # _apply_block_and_redistribute does, then run the old per-state algorithm.
    hpsi_states_oracle = [applyOp_test(H, p, cutoff=0.0) for p in psi_ref]
    for s in hpsi_states_oracle:
        s.prune(0.0)
    hpsi_states_oracle = basis.redistribute_psis(*hpsi_states_oracle)
    oracle_Djs, oracle_overlaps = _oracle_candidate_overlaps_and_energies(solver, H, hpsi_states_oracle, 0.0)

    assert local_Djs == oracle_Djs
    assert local_Djs == sorted(local_Djs)
    np.testing.assert_array_equal(overlaps, oracle_overlaps)
    assert e_Dj.shape == (len(local_Djs),)


def test_de2_min_selects_by_pt2_contribution():
    """determine_new_Dj admits candidates whose PT2 score reaches de2_min, rejects below."""
    d0, d1, d2 = _det([0]), _det([1]), _det([2])
    # d1 strongly coupled / small gap (large de2); d2 weakly coupled / large gap (tiny de2).
    H = ManyBodyOperator(
        {
            ((0, "c"), (0, "a")): 0.0,
            ((1, "c"), (1, "a")): 1.0,
            ((2, "c"), (2, "a")): 100.0,
            ((0, "c"), (1, "a")): 0.5,
            ((1, "c"), (0, "a")): 0.5,
            ((0, "c"), (2, "a")): 1e-4,
            ((2, "c"), (0, "a")): 1e-4,
        }
    )
    basis = _make_basis(comm=None)
    basis.clear()
    basis.add_states([d0])
    solver = CIPSISolver(basis)

    psi_ref = [ManyBodyState({d0: 1.0})]
    new = solver.determine_new_Dj(np.array([0.0]), psi_ref, H, de2_min=1e-3, gen_ops=None)
    assert d1 in new  # score ~ 0.5^2 / 1 = 0.25 >= 1e-3
    assert d2 not in new  # score ~ (1e-4)^2 / 100 = 1e-12 < 1e-3


# ---------------------------------------------------------------------------
# Diagnostics plumbing (calc_gs -> gs_info -> statistics)
# ---------------------------------------------------------------------------


def _calc_gs_kwargs(truncation_threshold):
    """A minimal calc_gs configuration for the 6-orbital SIAM (see test_sectorization)."""
    Hop = _siam_operator()
    basis_setup = {
        "impurity_orbitals": IMPURITY_ORBITALS,
        "bath_states": BATH_STATES,
        "nominal_impurity_occ": {0: 2},
        "mixed_valence": {0: 2},
        "tau": 0.01,
        "dense_cutoff": 1000,
        "truncation_threshold": truncation_threshold,
    }
    block_structure = BlockStructure(
        blocks=[[0, 1]],
        identical_blocks=[[0]],
        transposed_blocks=[],
        particle_hole_blocks=[],
        particle_hole_transposed_blocks=[],
        inequivalent_blocks=[0],
    )
    return Hop, basis_setup, block_structure


def test_calc_gs_reports_truncation_in_gs_info():
    Hop, basis_setup, block_structure = _calc_gs_kwargs(truncation_threshold=6)
    _, _, _, _, gs_info = calc_gs(
        Hop,
        basis_setup,
        block_structure,
        rot_to_spherical=np.eye(6),
        verbose=False,
        stats_path=None,
    )
    report = gs_info["truncation"]
    assert report is not None and report["cap_hit"]
    assert report["retained"] <= 6
    assert gs_info["statistics"]["truncation"] == report


def test_calc_gs_no_truncation_report_when_uncapped():
    Hop, basis_setup, block_structure = _calc_gs_kwargs(truncation_threshold=np.inf)
    _, _, _, _, gs_info = calc_gs(
        Hop,
        basis_setup,
        block_structure,
        rot_to_spherical=np.eye(6),
        verbose=False,
        stats_path=None,
    )
    assert gs_info["truncation"] is None
    assert gs_info["statistics"]["truncation"] is None
