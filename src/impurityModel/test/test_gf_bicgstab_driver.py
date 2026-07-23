"""Tests for the per-frequency BiCGSTAB Green's function (``gf_method="bicgstab"``).

Three layers, mirroring the Lanczos-path suites:

* kernel: ``block_Green_bicgstab`` against the dense resolvent on a closed sector
  (``test_greens_function``-style), including signs on both spectral sides and axes;
* cap: the ``_CappedBasisProxy`` + ``block_bicgstab`` freeze-growth interaction against the
  dense ``P H P`` resolvent on the retained set — the same strong oracle as
  ``test_gf_truncation``;
* driver: ``get_Greens_function(gf_method="bicgstab")`` against the block-Lanczos path at
  ``reort="partial"`` on a hybridizing system with a two-body term, on both meshes.
"""

import itertools
import os

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.basis_transcription import build_dense_matrix
from impurityModel.ed.cg import block_bicgstab
from impurityModel.ed.gf_solvers import block_Green_bicgstab
from impurityModel.ed.greens_function import (
    _CappedBasisProxy,
    _gf_signed_axes,
    get_Greens_function,
)
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, SlaterDeterminant, block_inner_cy

DELTA = 0.15
OMEGA = np.linspace(-8.0, 8.0, 21)
MATSUBARA = 1j * np.pi * 0.1 * (2 * np.arange(8) + 1)


def _det(occupied):
    """Determinant with the given orbitals occupied (MSB-first: orbital i = bit 7-i)."""
    b = 0
    for i in occupied:
        b |= 1 << (7 - i)
    return SlaterDeterminant.from_bytes(bytes([b]))


def _siam_6():
    """Single-impurity Anderson model, 6 spin-orbitals (0,1 imp; 2,3 val; 4,5 cond)."""
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


_IMP = {0: [[0, 1]]}
_BATHS = ({0: [[2, 3]]}, {0: [[4, 5]]})


def _seeds():
    """Two seed columns in the N=3 sector (reachable space: two 9-det (N_dn, N_up) sectors)."""
    return [
        ManyBodyState({_det([0, 2, 3]): 1.0 + 0j, _det([1, 2, 3]): 0.5 + 0j}),
        ManyBodyState({_det([0, 1, 2]): 1.0 + 0j}),
    ]


def _seed_basis(cap=np.inf, comm=None):
    seed_support = sorted({state for s in _seeds() for state in s})
    return Basis(
        _IMP,
        _BATHS,
        initial_basis=seed_support,
        truncation_threshold=cap,
        comm=comm,
        verbose=False,
    )


def _n3_sector_dets():
    """All 20 determinants with 3 of 6 orbitals occupied (closed under the SIAM-6 H)."""
    return [_det(c) for c in itertools.combinations(range(6), 3)]


def _dense_G_on(dets, z_values, comm=None):
    """G[k, i, j] = <seed_i| (z_k - H)^{-1} |seed_j> on the space spanned by ``dets``."""
    basis = Basis(_IMP, _BATHS, initial_basis=sorted(dets), comm=comm, verbose=False)
    H = np.asarray(build_dense_matrix(basis, _siam_6()))
    index = {det: i for i, det in enumerate(sorted(dets))}
    V = np.zeros((len(index), len(_seeds())), dtype=complex)
    for j, seed in enumerate(_seeds()):
        for det, amp in seed.items():
            V[index[det], j] = amp[0]
    G = np.empty((len(z_values), V.shape[1], V.shape[1]), dtype=complex)
    for k, z in enumerate(z_values):
        G[k] = V.conj().T @ np.linalg.solve(z * np.eye(len(index)) - H, V)
    return G


# --------------------------------------------------------------------------- #
# Kernel: dense-resolvent oracle, both sides and axes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("side_i", [0, 1])
def test_kernel_matches_dense_resolvent(side_i):
    """Uncapped kernel == dense resolvent on the closed N=3 sector, both axes, both signs."""
    e_shift = 0.3
    z_axes = _gf_signed_axes(MATSUBARA, OMEGA, side_i, DELTA)
    G_axes, stats = block_Green_bicgstab(
        _siam_6(),
        _seeds(),
        _seed_basis(),
        [e_shift],
        len(_seeds()),
        z_axes,
        atol=1e-10,
    )
    assert stats["n_unconverged"] == 0
    sector = _n3_sector_dets()
    for ax, z_axis in enumerate(z_axes):
        ref = _dense_G_on(sector, z_axis + e_shift)
        np.testing.assert_allclose(G_axes[ax][0], ref, atol=1e-7 * np.max(np.abs(ref)))


def test_kernel_zero_seed_column():
    """A zero seed column (an annihilated orbital) yields zero G entries, no crash."""
    seeds = [_seeds()[0], ManyBodyState(width=1)]
    z_axes = _gf_signed_axes(MATSUBARA, None, 0, DELTA)
    G_axes, stats = block_Green_bicgstab(_siam_6(), seeds, _seed_basis(), [0.0], 2, z_axes, atol=1e-10)
    assert stats["n_unconverged"] == 0
    np.testing.assert_array_equal(G_axes[0][0][:, 1, :], 0.0)
    np.testing.assert_array_equal(G_axes[0][0][:, :, 1], 0.0)
    assert np.max(np.abs(G_axes[0][0][:, 0, 0])) > 0


def test_kernel_gmres_fallback_rescues_failed_points():
    """With BiCGSTAB disabled (max_iter=0, every point 'fails'), the GMRES fallback must
    solve every point on its own: G still equals the dense resolvent, no point is left
    unconverged, and the stats record the fallback. (The default GF_GMRES_RESTART of 40
    exceeds this sector's dimension, so the rescue is a guaranteed full-GMRES solve.)"""
    e_shift = 0.3
    z_axes = _gf_signed_axes(MATSUBARA, OMEGA, 0, DELTA)
    G_axes, stats = block_Green_bicgstab(
        _siam_6(),
        _seeds(),
        _seed_basis(),
        [e_shift],
        len(_seeds()),
        z_axes,
        atol=1e-10,
        max_iter=0,
    )
    assert stats["gmres_points"] == stats["n_points"]
    assert stats["gmres_iterations"] > 0
    assert stats["n_unconverged"] == 0
    sector = _n3_sector_dets()
    for ax, z_axis in enumerate(z_axes):
        ref = _dense_G_on(sector, z_axis + e_shift)
        np.testing.assert_allclose(G_axes[ax][0], ref, atol=1e-7 * np.max(np.abs(ref)))


def test_kernel_bounded_by_cap():
    """A finite truncation_threshold bounds every per-point basis; the result stays causal."""
    cap = 12
    z_axes = _gf_signed_axes(None, OMEGA, 0, DELTA)
    G_axes, stats = block_Green_bicgstab(_siam_6(), _seeds(), _seed_basis(cap=cap), [0.0], 2, z_axes, atol=1e-10)
    assert stats["cap_hit"]
    assert not stats["seed_overflow"]
    assert stats["max_solve_basis"] <= cap
    assert stats["retained_size"] <= cap
    # retarded axis: Im G_ii <= 0 even under truncation (exact on the retained subspace)
    assert np.all(np.diagonal(G_axes[0][0].imag, axis1=1, axis2=2) <= 1e-12)


# --------------------------------------------------------------------------- #
# Cap: the P H P oracle at the block_bicgstab + _CappedBasisProxy level
# --------------------------------------------------------------------------- #


def _capped_solve_with(solver, cap, z, comm=None):
    """The cap-oracle harness, parametrized over the linear solver (BiCGSTAB / GMRES):
    solve ``(z - H) X = seeds`` through a fresh ``_CappedBasisProxy`` and return the
    seed-projected ``G`` plus the proxy (whose retained keys define ``P``)."""
    basis = _seed_basis(comm=comm)
    proxy = _CappedBasisProxy(basis, cap)
    # redistribute_psis SUMS per-rank contributions, so only rank 0 provides amplitudes.
    # Each seed goes through its own explicit width-1 block rather than a bare
    # ManyBodyState() placeholder on the non-owning ranks: once the flat and block
    # classes merge (Phase 7 step 3), a bare placeholder is the width-0 polymorphic
    # zero, an asymmetric mismatch against the owning rank's populated (eventually
    # width-1) seeds that would deadlock redistribute_psis' collective.
    seeds_full = _seeds()
    owns_seeds = comm is None or comm.rank == 0
    seed_blocks = (
        [ManyBodyState.from_states([s]) for s in seeds_full]
        if owns_seeds
        else [ManyBodyState(width=1) for _ in seeds_full]
    )
    seeds = ManyBodyState.from_states([blk.to_states()[0] for blk in basis.redistribute_psis(seed_blocks)])
    A = z - _siam_6()
    # Restart while unconverged, as the driver does: a near-pole z stagnates a single
    # BiCGSTAB pass (fresh shadow residual each call). GMRES restarts internally, so its
    # first call already converges and the loop is a no-op for it.
    X = ManyBodyState(width=seeds.width)
    info = {}
    for _ in range(10):
        X = solver(A, X, seeds, proxy, 0.0, atol=1e-12, info=info)
        if info["converged"]:
            break
    gram = block_inner_cy(seeds, X)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, gram, op=MPI.SUM)
    return gram, proxy


def _capped_solve(cap, z, comm=None):
    return _capped_solve_with(block_bicgstab, cap, z, comm=comm)


@pytest.mark.parametrize("cap", [6, 12, 17])
def test_capped_solve_equals_dense_php_resolvent(cap):
    """The strong oracle: a capped solve is the exact resolvent of P H P on the retained set."""
    z = 1.7 + 1j * DELTA
    G, proxy = _capped_solve(cap, z)
    assert proxy.cap_hit and proxy.retained_size <= cap
    retained = proxy.retained_keys()
    assert len(retained) == proxy.retained_size
    ref = _dense_G_on(retained, [z])[0]
    np.testing.assert_allclose(G, ref, atol=1e-9)


def test_uncapped_solve_matches_dense_full_sector():
    """Sanity: with the cap above the reachable space the solve is exact on the sector."""
    z = -2.4 + 1j * DELTA
    G, proxy = _capped_solve(1000, z)
    assert not proxy.cap_hit
    ref = _dense_G_on(_n3_sector_dets(), [z])[0]
    np.testing.assert_allclose(G, ref, atol=1e-9)


# --------------------------------------------------------------------------- #
# Driver: get_Greens_function(gf_method="bicgstab") vs block Lanczos (PARTIAL)
# --------------------------------------------------------------------------- #


def _hyb_hop():
    return ManyBodyOperator(
        {
            ((0, "c"), (0, "a")): 0.3,
            ((1, "c"), (1, "a")): 0.7,
            ((2, "c"), (2, "a")): -0.5,
            ((3, "c"), (3, "a")): 0.4,
            ((0, "c"), (2, "a")): 0.25,
            ((2, "c"), (0, "a")): 0.25,
            ((1, "c"), (3, "a")): 0.25,
            ((3, "c"), (1, "a")): 0.25,
            ((0, "c"), (1, "c"), (1, "a"), (0, "a")): 0.6,
        }
    )


def _run_driver(gf_method, reort, comm=None, monkeypatch_env=None):
    state_bytes = [b"\xa0", b"\x50"]  # {0, 2} and {1, 3}
    basis = Basis(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[2, 3]]}, {0: [[]]}),
        initial_basis=state_bytes,
        comm=comm if comm is not None else MPI.COMM_SELF,
    )
    psis = [ManyBodyState({SlaterDeterminant.from_bytes(b): 1.0}) for b in state_bytes]
    if comm is not None and comm.size > 1:
        # production seeds are hash-distributed; rank 0 provides the full amplitudes.
        # Each seed goes through its own explicit width-1 block rather than a bare
        # ManyBodyState() placeholder on the non-owning ranks (see _capped_solve_with's
        # comment for why a bare placeholder is a rename-time asymmetric-width hazard).
        owns_psis = comm.rank == 0
        psi_blocks = (
            [ManyBodyState.from_states([p]) for p in psis] if owns_psis else [ManyBodyState(width=1) for _ in psis]
        )
        psis = [blk.to_states()[0] for blk in basis.redistribute_psis(psi_blocks)]
    old_env = {}
    for key, value in (monkeypatch_env or {}).items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        gs_mat, gs_real, report = get_Greens_function(
            matsubara_mesh=1j * np.pi * 0.5 * (2 * np.arange(12) + 1),
            omega_mesh=np.linspace(-2.0, 2.0, 25),
            psis=psis,
            es=[-0.2, 0.3],
            tau=1.0,
            basis=basis,
            hOp=_hyb_hop(),
            delta=0.1,
            blocks=[[0, 1]],
            verbose=False,
            verbose_extra=False,
            reort=reort,
            dN=1,
            occ_cutoff=1e-6,
            slaterWeightMin=0.0,
            sparse=True,
            gf_method=gf_method,
        )
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return gs_mat, gs_real, report


def test_driver_matches_partial_lanczos():
    """bicgstab G == PARTIAL-reorthogonalized block-Lanczos G, both meshes."""
    m_l, r_l, _ = _run_driver("lanczos", "partial")
    m_b, r_b, report = _run_driver("bicgstab", None)
    np.testing.assert_allclose(m_b[0], m_l[0], atol=1e-7)
    np.testing.assert_allclose(r_b[0], r_l[0], atol=1e-7)
    # the report carries the solver record and the representation-independent checks
    names = {d.name for d in report.diagnostics}
    assert "bicgstab" in names and "causality" in names
    assert all(d.severity.name != "FAIL" for d in report.diagnostics)


def test_driver_grouping_and_operator_split_invariance():
    """Eigenstate grouping is a unit-shape knob only, and the operator-split env (a Lanczos
    decomposition) is ignored on the bicgstab path -- all give the identical G."""
    _, r_ref, _ = _run_driver("bicgstab", None)
    _, r_grouped, _ = _run_driver("bicgstab", None, monkeypatch_env={"GF_EIGENSTATE_GROUP": "2"})
    _, r_split, _ = _run_driver("bicgstab", None, monkeypatch_env={"GF_OPERATOR_SPLIT": "1"})
    np.testing.assert_allclose(r_grouped[0], r_ref[0], atol=1e-9)
    np.testing.assert_allclose(r_split[0], r_ref[0], atol=1e-9)


def test_driver_rejects_unknown_method():
    with pytest.raises(ValueError, match="gf_method"):
        _run_driver("haydock", None)


def test_sliced_driver_matches_partial_lanczos():
    """gf_method='sliced': the Chebyshev window terms sum back to the exact G (partition of
    unity is exact by construction), so the sliced driver must reproduce the PARTIAL-reort
    Lanczos G on both meshes -- and its report must carry the slicing record."""
    m_l, r_l, _ = _run_driver("lanczos", "partial")
    m_s, r_s, report = _run_driver("sliced", None, monkeypatch_env={"GF_SLICES": "3"})
    np.testing.assert_allclose(m_s[0], m_l[0], atol=1e-6)
    np.testing.assert_allclose(r_s[0], r_l[0], atol=1e-6)
    names = {d.name for d in report.diagnostics}
    assert "slicing" in names and "bicgstab" in names
    assert all(d.severity.name != "FAIL" for d in report.diagnostics)


def _reported_windows(report):
    """Window count the slicing diagnostic recorded (guards this file's GF_SLICES tests against
    silently testing the default: the knob is read at call time, so a regression to an
    import-time constant would make the slice-count legs identical and the assertions vacuous)."""
    (slicing,) = [d for d in report.diagnostics if d.name == "slicing"]
    return int(slicing.message.split()[0])


def test_sliced_driver_slice_count_invariance():
    """1 slice vs several: the partition identity makes the result slice-count independent
    (up to the per-solve atol)."""
    _, r_1, rep_1 = _run_driver("sliced", None, monkeypatch_env={"GF_SLICES": "1"})
    _, r_4, rep_4 = _run_driver("sliced", None, monkeypatch_env={"GF_SLICES": "4"})
    assert _reported_windows(rep_1) < _reported_windows(rep_4)
    np.testing.assert_allclose(r_4[0], r_1[0], atol=1e-6)


def test_sliced_driver_slice_tol_is_a_reported_accuracy_trade():
    """GF_SLICE_TOL prunes the filtered slice seeds -- the memory-for-accuracy knob. It must
    stay accurate to the discarded tail (<= sqrt(n_tail)*tol, i.e. far above the atol floor but
    nowhere near an unusable G) and it must never pass silently: the diagnostic warns."""
    _, r_exact, _ = _run_driver("sliced", None, monkeypatch_env={"GF_SLICES": "2"})
    _, r_pruned, report = _run_driver("sliced", None, monkeypatch_env={"GF_SLICES": "2", "GF_SLICE_TOL": "1e-6"})
    np.testing.assert_allclose(r_pruned[0], r_exact[0], atol=1e-3)
    (slicing,) = [d for d in report.diagnostics if d.name == "slicing"]
    assert slicing.severity.name == "WARN" and slicing.value == 1e-6


def test_kernel_bra_seeds_cross_element():
    """block_Green_bicgstab(bra_seeds=...) computes <bra|(z-H)^{-1}|ket> -- checked against
    the dense resolvent with distinct bra and ket blocks."""
    e_shift = 0.3
    z_axes = _gf_signed_axes(MATSUBARA, None, 0, DELTA)
    kets = _seeds()
    bras = [_seeds()[1], _seeds()[0]]  # swapped, so the cross element is genuinely asymmetric
    G_axes, stats = block_Green_bicgstab(
        _siam_6(),
        list(kets),
        _seed_basis(),
        [e_shift],
        2,
        z_axes,
        atol=1e-10,
        bra_seeds=list(bras),
    )
    assert stats["n_unconverged"] == 0
    sector = _n3_sector_dets()
    basis = Basis(_IMP, _BATHS, initial_basis=sorted(sector), verbose=False)
    H_mat = np.asarray(build_dense_matrix(basis, _siam_6()))
    index = {det: i for i, det in enumerate(sorted(sector))}
    K = np.zeros((len(index), 2), dtype=complex)
    B = np.zeros((len(index), 2), dtype=complex)
    for j, (k_state, b_state) in enumerate(zip(kets, bras)):
        for det, amp in k_state.items():
            K[index[det], j] = amp[0]
        for det, amp in b_state.items():
            B[index[det], j] = amp[0]
    for k, z in enumerate(z_axes[0] + e_shift):
        ref = B.conj().T @ np.linalg.solve(z * np.eye(len(index)) - H_mat, K)
        np.testing.assert_allclose(G_axes[0][0, k], ref, atol=1e-7 * max(np.max(np.abs(ref)), 1.0))


# --------------------------------------------------------------------------- #
# MPI
# --------------------------------------------------------------------------- #


@pytest.mark.mpi
def test_driver_mpi_matches_partial_lanczos():
    """Distributed run (2+ ranks): the per-point rebuild/redistribute/solve cycle stays in
    MPI lock-step (empty ranks included) and reproduces the PARTIAL Lanczos G."""
    comm = MPI.COMM_WORLD
    m_l, r_l, _ = _run_driver("lanczos", "partial", comm=comm)
    m_b, r_b, _ = _run_driver("bicgstab", None, comm=comm)
    if comm.rank == 0:
        np.testing.assert_allclose(m_b[0], m_l[0], atol=1e-7)
        np.testing.assert_allclose(r_b[0], r_l[0], atol=1e-7)
    else:
        assert m_b is None and r_b is None


@pytest.mark.mpi
def test_sliced_driver_mpi_matches_partial_lanczos():
    """Distributed sliced run. The bras live *outside* the per-point basis (they enter only
    the closing Gram), so their amplitudes are placed by determinant hash while X is placed
    by the basis partition. This test is what keeps those two orderings honest: if the bra
    ownership and the basis ownership ever disagree, the merge-joined Gram silently drops
    (or double-counts) the determinants they disagree on, and G walks away from Lanczos."""
    comm = MPI.COMM_WORLD
    m_l, r_l, _ = _run_driver("lanczos", "partial", comm=comm)
    m_s, r_s, _ = _run_driver("sliced", None, comm=comm, monkeypatch_env={"GF_SLICES": "3"})
    if comm.rank == 0:
        np.testing.assert_allclose(m_s[0], m_l[0], atol=1e-6)
        np.testing.assert_allclose(r_s[0], r_l[0], atol=1e-6)
    else:
        assert m_s is None and r_s is None


@pytest.mark.mpi
def test_capped_solve_mpi_matches_dense_php():
    """Distributed capped solve: collective cap decisions consistent, P H P oracle holds."""
    comm = MPI.COMM_WORLD
    z = 1.7 + 1j * DELTA
    for cap in (6, 12):
        G, proxy = _capped_solve(cap, z, comm=comm)
        assert proxy.cap_hit and proxy.retained_size <= cap
        gathered = comm.allgather(proxy.retained_keys())
        retained = sorted({k for part in gathered for k in part})
        assert len(retained) == proxy.retained_size
        ref = _dense_G_on(retained, [z])[0]
        np.testing.assert_allclose(G, ref, atol=1e-9)
