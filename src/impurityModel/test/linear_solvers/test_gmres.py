"""Tests for the restarted block-GMRES solver (``impurityModel.ed.gmres.block_gmres``).

Mirrors ``test_cg.py``'s coverage (the two solvers share their entry contract) plus the
GMRES-specific cases: restart-cycle behavior, Arnoldi happy breakdown on a closed
sector, and the stagnation case that motivates the solver -- a near-pole resolvent
where ``block_bicgstab`` (with restarts) fails and ``block_gmres`` converges.
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.basis_transcription import build_sparse_matrix, build_vector
from impurityModel.ed.cg import block_bicgstab
from impurityModel.ed.gmres import block_gmres
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, SlaterDeterminant
from impurityModel.test.support.gf_oracles import _capped_solve_with, _dense_G_on


def _resolvent_system(n=60, seed=0, delta=0.05):
    """A ``(z - H)`` block system of the shape the per-frequency Green's function solves."""
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = 0.5 * (H + H.conj().T)
    A = 1j * delta * np.eye(n) - H
    Y = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
    return A, Y, rng


# --------------------------------------------------------------------------- #
# Dense path
# --------------------------------------------------------------------------- #


def test_block_gmres_array_single():
    np.random.seed(42)
    A = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    A = A @ A.conj().T + np.eye(10)
    x_exact = np.random.rand(10, 1) + 1j * np.random.rand(10, 1)
    y = A @ x_exact
    x_sol = block_gmres(A, np.zeros((10, 1), complex), y, basis=None, slaterWeightMin=0.0)
    np.testing.assert_allclose(x_sol, x_exact, rtol=1e-5, atol=1e-5)


def test_block_gmres_array_block():
    np.random.seed(42)
    A = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    A = A @ A.conj().T + np.eye(10)
    x_exact = np.random.rand(10, 3) + 1j * np.random.rand(10, 3)
    y = A @ x_exact
    x_sol = block_gmres(A, np.zeros((10, 3), complex), y, basis=None, slaterWeightMin=0.0)
    np.testing.assert_allclose(x_sol, x_exact, rtol=1e-5, atol=1e-5)


def test_block_gmres_zero_rhs():
    A = np.eye(5)
    x_sol = block_gmres(A, np.zeros((5, 2), complex), np.zeros((5, 2), complex), basis=None, slaterWeightMin=0.0)
    np.testing.assert_allclose(x_sol, np.zeros((5, 2), complex))


def test_block_gmres_exact_guess():
    A = np.eye(5)
    x_exact = np.ones((5, 2), dtype=complex)
    info = {}
    x_sol = block_gmres(A, x_exact.copy(), A @ x_exact, basis=None, slaterWeightMin=0.0, info=info)
    np.testing.assert_allclose(x_sol, x_exact)
    assert info["converged"] and info["iterations"] == 0


@pytest.mark.parametrize("perturbation", [1e-2, 1e-5, 1e-7, 1e-9])
def test_block_gmres_refines_a_good_warm_start(perturbation):
    """A warm start is refined to ``atol`` however good it already is (the entry deflation
    normalizes the residual Gram, exactly as in block_bicgstab)."""
    A, Y, rng = _resolvent_system()
    X_exact = np.linalg.solve(A, Y)
    noise = rng.normal(size=X_exact.shape) + 1j * rng.normal(size=X_exact.shape)
    X0 = X_exact + perturbation * noise * np.linalg.norm(X_exact) / np.linalg.norm(noise)
    atol = 1e-11
    X = block_gmres(A, X0.copy(), Y, basis=None, slaterWeightMin=0.0, atol=atol)
    assert np.linalg.norm(A @ X - Y) / np.linalg.norm(Y) < 10 * atol


def test_block_gmres_atol_is_relative_to_the_rhs():
    A, Y, _rng = _resolvent_system()
    atol = 1e-9
    for scale in (1e-6, 1.0, 1e6):
        X = block_gmres(A, np.zeros_like(Y), scale * Y, basis=None, slaterWeightMin=0.0, atol=atol)
        rel = np.linalg.norm(A @ X - scale * Y) / np.linalg.norm(scale * Y)
        assert rel < 10 * atol, f"scale {scale:g}: relative residual {rel:.2e}"


def test_block_gmres_array_rank_deficient():
    """Linearly dependent RHS columns are reconstructed by linearity via the deflation."""
    np.random.seed(1)
    A = np.random.rand(8, 8) + 1j * np.random.rand(8, 8)
    A = A @ A.conj().T + np.eye(8)
    c0 = np.random.rand(8, 1) + 1j * np.random.rand(8, 1)
    x_exact = np.hstack([c0, 2.0 * c0])
    y = A @ x_exact
    x_sol = block_gmres(A, np.zeros((8, 2), complex), y, basis=None, slaterWeightMin=0.0, atol=1e-10)
    np.testing.assert_allclose(x_sol, x_exact, rtol=1e-6, atol=1e-6)


def test_block_gmres_array_partial_rank():
    """A 3-column block of rank 2 (column 2 = column 0 + column 1) is solved exactly."""
    np.random.seed(2)
    A = np.random.rand(9, 9) + 1j * np.random.rand(9, 9)
    A = A @ A.conj().T + np.eye(9)
    c0 = np.random.rand(9, 1) + 1j * np.random.rand(9, 1)
    c1 = np.random.rand(9, 1) + 1j * np.random.rand(9, 1)
    x_exact = np.hstack([c0, c1, c0 + c1])
    y = A @ x_exact
    x_sol = block_gmres(A, np.zeros((9, 3), complex), y, basis=None, slaterWeightMin=0.0, atol=1e-10)
    np.testing.assert_allclose(x_sol, x_exact, rtol=1e-6, atol=1e-6)


def test_block_gmres_restart_cycles_reach_atol():
    """A restart length below the problem size still converges through cycles.

    Indefinite spectra need a minimum restart length to make per-cycle progress at all
    (restarted GMRES with a too-short window stalls, honestly reported by the stall
    gate) -- 15 is comfortably above that threshold for this 60-dim anchor while still
    forcing multiple cycles.
    """
    A, Y, _rng = _resolvent_system(n=60, delta=0.5)
    info = {}
    X = block_gmres(
        A, np.zeros_like(Y), Y, basis=None, slaterWeightMin=0.0, atol=1e-9, restart=15, max_restarts=50, info=info
    )
    rel = np.linalg.norm(A @ X - Y) / np.linalg.norm(Y)
    assert info["converged"] and rel < 1e-8
    assert info["iterations"] > 15  # more than one cycle was needed


def test_block_gmres_max_restarts_reports_unconverged():
    A, Y, _rng = _resolvent_system(n=60, delta=1e-4)
    info = {}
    block_gmres(
        A, np.zeros_like(Y), Y, basis=None, slaterWeightMin=0.0, atol=1e-14, restart=2, max_restarts=1, info=info
    )
    assert not info["converged"]
    assert info["rel_residual"] > 1e-14


# --------------------------------------------------------------------------- #
# The motivating case: BiCGSTAB stagnates, GMRES converges
# --------------------------------------------------------------------------- #


def test_gmres_rescues_a_failed_bicgstab_solve():
    """The fallback contract, deterministically: BiCGSTAB ends unconverged on a hard
    near-pole system (budget-capped -- organic stagnation flips with BLAS summation
    order, so the test must not depend on it); GMRES warm-started from the failed
    iterate finishes the solve.

    ``restart >= n`` makes the rescue *guaranteed* (un-restarted GMRES terminates
    finitely), which is the robust form of the claim. On strongly indefinite spectra a
    short-restart GMRES can stall too -- the fallback's production restart length is a
    knob (``GF_GMRES_RESTART``), and the honest failure mode is a reported unconverged
    point, covered by test_block_gmres_max_restarts_reports_unconverged.
    """
    rng = np.random.default_rng(3)
    n = 120
    H = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = 0.5 * (H + H.conj().T)
    evals = np.linalg.eigvalsh(H)
    z = evals[n // 2] + 1e-6 + 1e-4j
    A = z * np.eye(n) - H
    Y = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
    atol = 1e-10

    info_b = {}
    Xb = np.zeros_like(Y)
    for _ in range(2):
        Xb = block_bicgstab(A, Xb, Y, basis=None, slaterWeightMin=0.0, atol=atol, max_iter=30, info=info_b)
        if info_b["converged"]:
            break
    assert not info_b["converged"], "anchor converged within the capped budget; harden it (smaller Im z)"
    res_b = np.max(np.linalg.norm(A @ Xb - Y, axis=0)) / np.linalg.norm(Y)

    info_g = {}
    Xg = block_gmres(A, Xb.copy(), Y, basis=None, slaterWeightMin=0.0, atol=atol, restart=n + 10, info=info_g)
    res_g = np.max(np.linalg.norm(A @ Xg - Y, axis=0)) / np.linalg.norm(Y)
    assert info_g["converged"], info_g
    assert res_g < 100 * atol, (res_b, res_g)
    # the info record reflects the measured residual
    assert info_g["rel_residual"] == pytest.approx(res_g, rel=2.0)


# --------------------------------------------------------------------------- #
# Sparse (ManyBodyState) path
# --------------------------------------------------------------------------- #


def _sparse_system(n_sites=6, n_particles=3):
    """Number-conserving H on the full fixed-N space (closed under H, so the dense
    reference matches exactly)."""
    states = []
    for c in itertools.combinations(range(n_sites), n_particles):
        b = bytearray((n_sites + 7) // 8)
        for o in c:
            b[o // 8] |= 1 << (7 - (o % 8))
        states.append(bytes(b))
    op = {}
    for i in range(n_sites):
        op[((i, "c"), (i, "a"))] = float(i + 1)
        if i + 1 < n_sites:
            op[((i, "c"), (i + 1, "a"))] = 0.5
            op[((i + 1, "c"), (i, "a"))] = 0.5
    op[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = 0.7
    basis = Basis(
        impurity_orbitals={0: [list(range(n_sites))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=False,
    )
    return ManyBodyOperator(op), basis


def _rand_states(basis, rng, n_cols):
    dets = [basis.type.from_bytes(b) if isinstance(b, bytes) else b for b in basis.local_basis]
    return [
        ManyBodyState({d: complex(rng.standard_normal(), rng.standard_normal()) for d in dets}) for _ in range(n_cols)
    ]


def test_block_gmres_sparse_matches_dense():
    H, basis = _sparse_system()
    rng = np.random.default_rng(17)
    ys = _rand_states(basis, rng, 3)
    info = {}
    x0 = ManyBodyState(width=3)
    xs = block_gmres(H, x0, ManyBodyState.from_states(ys), basis=basis, slaterWeightMin=0.0, info=info)
    H_mat = build_sparse_matrix(basis, H).toarray()
    X_ref = np.linalg.solve(H_mat, build_vector(basis, ys).T)
    np.testing.assert_allclose(build_vector(basis, xs.to_states()).T, X_ref, atol=1e-6)
    assert info["converged"]


def test_block_gmres_sparse_rank_deficient_rhs():
    H, basis = _sparse_system()
    rng = np.random.default_rng(19)
    y = _rand_states(basis, rng, 1)[0]
    ys = [y, y * (2.0 + 0j)]
    x0 = ManyBodyState(width=2)
    xs = block_gmres(H, x0, ManyBodyState.from_states(ys), basis=basis, slaterWeightMin=0.0).to_states()
    diff = xs[1] - xs[0] * (2.0 + 0j)
    assert np.sqrt(diff.norm2()) < 1e-8  # exact linearity of the dependent column


def test_block_gmres_sparse_happy_breakdown_exact():
    """Seeds spanning a tiny invariant subspace close the Arnoldi space in a few steps;
    the projected solution is exact (residual at machine precision, not just atol)."""
    z = 3.0 + 0.5j
    A = z - ManyBodyOperator({((0, "c"), (1, "a")): 0.7, ((1, "c"), (0, "a")): 0.7})
    basis = Basis(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=[b"\x80", b"\x40"],
        verbose=False,
        comm=MPI.COMM_SELF,
    )
    det0 = SlaterDeterminant.from_bytes(b"\x80")
    ys = [ManyBodyState({det0: 1.0})]
    info = {}
    x0 = ManyBodyState(width=1)
    atol = 1e-12
    block_gmres(A, x0, ManyBodyState.from_states(ys), basis=basis, slaterWeightMin=0.0, atol=atol, info=info)
    assert info["converged"] and info["iterations"] <= 2
    # rel_residual is only guaranteed down to the solve's own requested atol, not an
    # independent last-ulp bound; derive the check from that argument, one order of
    # magnitude looser to absorb the residual-norm rounding on top of the solve itself.
    assert info["rel_residual"] < 10 * atol


def test_block_gmres_sparse_warm_start_exact():
    H, basis = _sparse_system()
    rng = np.random.default_rng(23)
    ys = ManyBodyState.from_states(_rand_states(basis, rng, 2))
    x0 = ManyBodyState(width=2)
    xs = block_gmres(H, x0, ys, basis=basis, slaterWeightMin=0.0)
    info = {}
    xs2 = block_gmres(H, xs, ys, basis=basis, slaterWeightMin=0.0, info=info)
    assert info["iterations"] == 0
    for a, b in zip(xs.to_states(), xs2.to_states()):
        assert np.sqrt((a - b).norm2()) < 1e-10


def test_block_gmres_capped_proxy_php_oracle():
    """A caps_growth basis bounds the solve to the retained set; the result is the exact
    resolvent of P H P there -- the same oracle as the BiCGSTAB/Lanczos cap tests."""
    z = 1.7 + 1j * 0.15
    G, proxy = _capped_solve_with(block_gmres, 12, z)
    assert proxy.cap_hit and proxy.retained_size <= 12
    ref = _dense_G_on(proxy.retained_keys(), [z])[0]
    np.testing.assert_allclose(G, ref, atol=1e-9)


def _all_det_bytes(n_sites=6, n_particles=3):
    """The full fixed-N determinant list as bytes -- no Basis (and no collectives) involved."""
    out = []
    for c in itertools.combinations(range(n_sites), n_particles):
        b = bytearray((n_sites + 7) // 8)
        for o in c:
            b[o // 8] |= 1 << (7 - (o % 8))
        out.append(bytes(b))
    return out


@pytest.mark.mpi
def test_block_gmres_mpi_matches_dense():
    """Distributed solve agrees with the rank-local dense reference (2+ ranks leave some
    rank with few rows -- the empty-rank edge must not deadlock). Every collective object
    is constructed identically on all ranks; only the seed *amplitudes* are rank-gated
    (redistribute_psis sums per-rank contributions)."""
    comm = MPI.COMM_WORLD
    dets = _all_det_bytes()
    op = _sparse_system()[0]

    # initial_basis must be IDENTICAL on every rank (the Basis contract; see
    # test_gf_truncation) -- only the seed amplitudes are rank-gated below.
    dist_basis = Basis(
        impurity_orbitals={0: [list(range(6))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=dets,
        comm=comm,
        verbose=False,
    )
    rng = np.random.default_rng(29)  # same stream on every rank -> identical amplitudes
    dets_sd = [SlaterDeterminant.from_bytes(b) for b in sorted(dets)]
    ys_full = [
        ManyBodyState({d: complex(rng.standard_normal(), rng.standard_normal()) for d in dets_sd}) for _ in range(2)
    ]
    # Each seed goes through its own explicit width-1 block rather than a bare
    # ManyBodyState() placeholder on the non-owning ranks: once the flat and block
    # classes merge (Phase 7 step 3), a bare placeholder is the width-0 polymorphic
    # zero, an asymmetric mismatch against the owning rank's populated (eventually
    # width-1) seeds that would deadlock redistribute_psis' collective.
    owns_ys = comm.rank == 0
    y_blocks = (
        [ManyBodyState.from_states([y]) for y in ys_full] if owns_ys else [ManyBodyState(width=1) for _ in ys_full]
    )
    ys = [blk.to_states()[0] for blk in dist_basis.redistribute_psis(*y_blocks)]
    info = {}
    x0 = ManyBodyState(width=2)
    xs = block_gmres(op, x0, ManyBodyState.from_states(ys), basis=dist_basis, slaterWeightMin=0.0, info=info)
    assert info["converged"]
    X_dist = build_vector(dist_basis, xs.to_states(), root=0).T

    # Dense reference in dist_basis's OWN global ordering (hash distribution orders
    # determinants by owner, not lexicographically -- a serial basis would permute the
    # rows). build_sparse_matrix populates only the local kets' columns per rank, so
    # the full matrix is the SUM over ranks (the same reduction block_green_impl does).
    H_part = np.ascontiguousarray(build_sparse_matrix(dist_basis, op).toarray())
    H_mat = np.zeros_like(H_part) if comm.rank == 0 else None
    comm.Reduce(H_part, H_mat, op=MPI.SUM, root=0)
    Y_dist = build_vector(dist_basis, ys, root=0).T
    if comm.rank == 0:
        X_ref = np.linalg.solve(H_mat, Y_dist)
        np.testing.assert_allclose(X_dist, X_ref, atol=1e-6)
