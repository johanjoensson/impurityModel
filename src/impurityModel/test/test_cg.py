import itertools

import numpy as np
import pytest

from impurityModel.ed.basis_transcription import build_sparse_matrix, build_vector
from impurityModel.ed.cg import block_bicgstab
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, SlaterDeterminant, inner


def test_block_bicgstab_array_single():
    np.random.seed(42)
    A = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    A = A @ A.conj().T + np.eye(10)

    x_exact = np.random.rand(10, 1) + 1j * np.random.rand(10, 1)
    y = A @ x_exact
    x0 = np.zeros((10, 1), dtype=complex)

    x_sol = block_bicgstab(A, x0, y, basis=None, slaterWeightMin=0.0)
    np.testing.assert_allclose(x_sol, x_exact, rtol=1e-5, atol=1e-5)


def test_block_bicgstab_array_block():
    np.random.seed(42)
    A = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    A = A @ A.conj().T + np.eye(10)

    x_exact = np.random.rand(10, 3) + 1j * np.random.rand(10, 3)
    y = A @ x_exact
    x0 = np.zeros((10, 3), dtype=complex)

    x_sol = block_bicgstab(A, x0, y, basis=None, slaterWeightMin=0.0)
    np.testing.assert_allclose(x_sol, x_exact, rtol=1e-5, atol=1e-5)


def test_block_bicgstab_zero_rhs():
    A = np.eye(5)
    x0 = np.zeros((5, 2), dtype=complex)
    y = np.zeros((5, 2), dtype=complex)
    x_sol = block_bicgstab(A, x0, y, basis=None, slaterWeightMin=0.0)
    np.testing.assert_allclose(x_sol, np.zeros((5, 2), dtype=complex))


def test_block_bicgstab_exact_guess():
    A = np.eye(5)
    x_exact = np.ones((5, 2), dtype=complex)
    x0 = x_exact.copy()
    y = A @ x_exact
    x_sol = block_bicgstab(A, x0, y, basis=None, slaterWeightMin=0.0)
    np.testing.assert_allclose(x_sol, x_exact)


def test_block_bicgstab_max_iter():
    A = np.eye(5)
    A[0, 1] = 0.5
    x_exact = np.ones((5, 2), dtype=complex)
    y = A @ x_exact
    x0 = np.zeros((5, 2), dtype=complex)
    x_sol = block_bicgstab(A, x0, y, basis=None, slaterWeightMin=0.0, max_iter=0)
    assert not np.allclose(x_sol, x_exact)


def _resolvent_system(n=60, seed=0):
    """A ``(z - H)`` block system of the shape the per-frequency Green's function solves."""
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = 0.5 * (H + H.conj().T)
    A = 0.05j * np.eye(n) - H
    Y = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
    return A, Y, rng


@pytest.mark.parametrize("perturbation", [1e-2, 1e-5, 1e-7, 1e-9])
def test_block_bicgstab_refines_a_good_warm_start(perturbation):
    """A warm start is refined to ``atol``, however good it already is.

    ``_cholesky_or_deflate``'s rank floor is absolute (``evals > EPS**(2/3) * max(l_max, 1)``),
    so before the Gram was normalized any initial residual block with ``||R0|| < ~6e-6``
    deflated to rank 0 and ``block_bicgstab`` returned ``x0`` untouched -- silently, whatever
    ``atol`` asked for. That is exactly the regime a frequency-swept warm start lives in.
    """
    A, Y, rng = _resolvent_system()
    X_exact = np.linalg.solve(A, Y)
    noise = rng.normal(size=X_exact.shape) + 1j * rng.normal(size=X_exact.shape)
    X0 = X_exact + perturbation * noise * np.linalg.norm(X_exact) / np.linalg.norm(noise)

    atol = 1e-11
    X = block_bicgstab(A, X0.copy(), Y, basis=None, slaterWeightMin=0.0, atol=atol, rtol=1e-14, max_iter=200)
    # 30*atol, not 10*atol: reaching ~1e-11 relative on a complex resolvent is right at the
    # BLAS-dependent floor. The contract this guards is that the warm start is *refined* (not
    # silently deflated to x0), which a residual near 1e-10 still demonstrates.
    assert np.linalg.norm(A @ X - Y) / np.linalg.norm(Y) < 30 * atol


def test_block_bicgstab_atol_is_relative_to_the_rhs():
    """Scaling ``Y`` scales the delivered residual, so ``atol`` means the same thing."""
    A, Y, _rng = _resolvent_system()
    X0 = np.zeros_like(Y)
    atol = 1e-9
    for scale in (1e-6, 1.0, 1e6):
        X = block_bicgstab(A, X0.copy(), scale * Y, basis=None, slaterWeightMin=0.0, atol=atol, max_iter=200)
        rel = np.linalg.norm(A @ X - scale * Y) / np.linalg.norm(scale * Y)
        assert rel < 10 * atol, f"scale {scale:g}: relative residual {rel:.2e}"


def test_block_bicgstab_converged_warm_start_costs_no_iterations():
    """An ``x0`` that already meets ``atol`` is returned unchanged, not refined further."""
    A, Y, _rng = _resolvent_system()
    X_exact = np.linalg.solve(A, Y)
    X = block_bicgstab(A, X_exact.copy(), Y, basis=None, slaterWeightMin=0.0, atol=1e-6, max_iter=200)
    np.testing.assert_array_equal(X, X_exact)


def test_block_bicgstab_info_cold_solve():
    """``info`` reports a converged cold solve, and its residual estimate is honest."""
    A, Y, _rng = _resolvent_system()
    atol = 1e-9
    info = {}
    X = block_bicgstab(A, np.zeros_like(Y), Y, basis=None, slaterWeightMin=0.0, atol=atol, max_iter=500, info=info)
    assert info["converged"]
    assert info["iterations"] > 0
    true_rel = np.max(np.linalg.norm(A @ X - Y, axis=0)) / np.linalg.norm(Y)
    # The estimate is measured on the deflated system and rescaled by ||R0|| (an upper-bound
    # scale up to sqrt(n)); it must be the right order of magnitude and never optimistic by
    # more than that geometry factor.
    assert info["rel_residual"] < 10 * atol
    assert true_rel < np.sqrt(Y.shape[1]) * info["rel_residual"] + np.finfo(float).eps


def test_block_bicgstab_info_max_iter_reports_unconverged():
    A, Y, _rng = _resolvent_system()
    info = {}
    block_bicgstab(A, np.zeros_like(Y), Y, basis=None, slaterWeightMin=0.0, atol=1e-12, max_iter=1, info=info)
    assert info["iterations"] == 1
    assert not info["converged"]
    assert info["rel_residual"] > 1e-12


def test_block_bicgstab_info_converged_warm_start():
    """An already-converged warm start reports 0 iterations and converged=True."""
    A, Y, _rng = _resolvent_system()
    X_exact = np.linalg.solve(A, Y)
    info = {}
    block_bicgstab(A, X_exact.copy(), Y, basis=None, slaterWeightMin=0.0, atol=1e-6, max_iter=200, info=info)
    assert info["iterations"] == 0
    assert info["converged"]
    assert info["rel_residual"] < 1e-6


# --------------------------------------------------------------------------- #
# Sparse (ManyBodyState) path: real end-to-end solves against a dense
# reference. These replace the old mock-based dict tests, which patched the
# pre-block internals (cg.inner etc.) and never exercised the real solver.
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
        op[((i, "c"), (i, "a"))] = float(i + 1)  # distinct levels -> nonsingular H
        if i + 1 < n_sites:
            op[((i, "c"), (i + 1, "a"))] = 0.5
            op[((i + 1, "c"), (i, "a"))] = 0.5
    op[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = 0.7  # a two-body term
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


def _dense_ref(basis, H, ys):
    H_mat = build_sparse_matrix(basis, H).toarray()
    Y = build_vector(basis, ys).T
    return H_mat, np.linalg.solve(H_mat, Y)


def test_block_bicgstab_sparse_matches_dense():
    H, basis = _sparse_system()
    rng = np.random.default_rng(17)
    ys = _rand_states(basis, rng, 3)
    x0 = ManyBodyState(width=3)
    xs = block_bicgstab(H, x0, ManyBodyState.from_states(ys), basis=basis, slaterWeightMin=0.0)
    _, X_ref = _dense_ref(basis, H, ys)
    X = build_vector(basis, xs.to_states()).T
    np.testing.assert_allclose(X, X_ref, atol=1e-6)


def test_block_bicgstab_sparse_rank_deficient_rhs():
    """A duplicated RHS column must be reconstructed by linearity via the deflation."""
    H, basis = _sparse_system()
    rng = np.random.default_rng(19)
    y = _rand_states(basis, rng, 1)[0]
    ys = [y, y * (2.0 + 0j)]
    x0 = ManyBodyState(width=2)
    xs = block_bicgstab(H, x0, ManyBodyState.from_states(ys), basis=basis, slaterWeightMin=0.0).to_states()
    _, X_ref = _dense_ref(basis, H, ys)
    X = build_vector(basis, xs).T
    np.testing.assert_allclose(X, X_ref, atol=1e-6)
    diff = xs[1] - xs[0] * (2.0 + 0j)
    assert np.sqrt(diff.norm2()) < 1e-8  # exact linearity of the dependent column


def test_block_bicgstab_sparse_warm_start_exact():
    """An exact initial guess returns immediately (zero residual -> rank 0)."""
    H, basis = _sparse_system()
    rng = np.random.default_rng(23)
    ys = ManyBodyState.from_states(_rand_states(basis, rng, 2))
    x0 = ManyBodyState(width=2)
    xs = block_bicgstab(H, x0, ys, basis=basis, slaterWeightMin=0.0)
    xs2 = block_bicgstab(H, xs, ys, basis=basis, slaterWeightMin=0.0)
    for a, b in zip(xs.to_states(), xs2.to_states()):
        diff = a - b
        assert np.sqrt(diff.norm2()) < 1e-10


def test_block_bicgstab_sparse_info_and_rhs_untouched():
    """Sparse path: ``info`` reports convergence, and the caller's RHS block is unmodified
    (guards the ``ri = r0_t = pi = rhs`` aliasing at the core's entry -- an in-place update
    would corrupt the shadow residual *and* the caller's block, which is now passed straight
    through with no defensive copy at the solver's boundary)."""
    H, basis = _sparse_system()
    rng = np.random.default_rng(31)
    ys = _rand_states(basis, rng, 2)
    y_blk = ManyBodyState.from_states(ys)
    y_before = np.asarray(y_blk).copy()
    info = {}
    x0 = ManyBodyState(width=2)
    xs = block_bicgstab(H, x0, y_blk, basis=basis, slaterWeightMin=0.0, info=info)
    assert info["converged"]
    assert info["iterations"] > 0
    assert info["rel_residual"] < 1e-7
    np.testing.assert_array_equal(np.asarray(y_blk), y_before)
    _, X_ref = _dense_ref(basis, H, ys)
    np.testing.assert_allclose(build_vector(basis, xs.to_states()).T, X_ref, atol=1e-6)


def test_block_bicgstab_sparse_max_iter():
    H, basis = _sparse_system()
    rng = np.random.default_rng(29)
    ys = _rand_states(basis, rng, 2)
    x0 = ManyBodyState(width=2)
    xs = block_bicgstab(H, x0, ManyBodyState.from_states(ys), basis=basis, slaterWeightMin=0.0, max_iter=0)
    _, X_ref = _dense_ref(basis, H, ys)
    X = build_vector(basis, xs.to_states()).T
    assert not np.allclose(X, X_ref, atol=1e-6)


# Recovered pre-block tests (real coverage, kept verbatim): array deflation
# cases and the original sparse rank-deficient linearity check.


def test_block_bicgstab_break_active_mask():
    # Test line 110: break if not np.any(active_mask)
    # We want a real array test where r0_norm > eps, but after 1 iteration, r_norms drops below atol
    np.random.seed(42)
    A = np.eye(2)
    x_exact = np.ones((2, 1), dtype=complex)
    y = A @ x_exact
    x0 = np.zeros((2, 1), dtype=complex)

    # We can force a break after 1 iteration because A is identity
    x_sol = block_bicgstab(A, x0, y, basis=None, slaterWeightMin=0.0)
    np.testing.assert_allclose(x_sol, x_exact)


def test_block_bicgstab_array_rank_deficient():
    """A block RHS with linearly dependent columns is solved exactly (A invertible), matching
    the per-column solves -- the case that previously stalled at the zero guess."""
    np.random.seed(1)
    A = np.random.rand(8, 8) + 1j * np.random.rand(8, 8)
    A = A @ A.conj().T + np.eye(8)
    c0 = np.random.rand(8, 1) + 1j * np.random.rand(8, 1)
    x_exact = np.hstack([c0, 2.0 * c0])  # rank-1 block: column 1 == 2 * column 0
    y = A @ x_exact
    x0 = np.zeros((8, 2), dtype=complex)

    x_sol = block_bicgstab(A, x0, y, basis=None, slaterWeightMin=0.0, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(x_sol, x_exact, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(A @ x_sol, y, rtol=1e-6, atol=1e-6)

    per_col = np.hstack(
        [
            block_bicgstab(
                A, np.zeros((8, 1), complex), y[:, k : k + 1], basis=None, slaterWeightMin=0.0, atol=1e-10, rtol=1e-12
            )
            for k in range(2)
        ]
    )
    np.testing.assert_allclose(x_sol, per_col, rtol=1e-6, atol=1e-6)


def test_block_bicgstab_array_partial_rank():
    """A 3-column block of rank 2 (column 2 = column 0 + column 1) is solved exactly."""
    np.random.seed(2)
    A = np.random.rand(9, 9) + 1j * np.random.rand(9, 9)
    A = A @ A.conj().T + np.eye(9)
    c0 = np.random.rand(9, 1) + 1j * np.random.rand(9, 1)
    c1 = np.random.rand(9, 1) + 1j * np.random.rand(9, 1)
    x_exact = np.hstack([c0, c1, c0 + c1])
    y = A @ x_exact
    x0 = np.zeros((9, 3), dtype=complex)

    x_sol = block_bicgstab(A, x0, y, basis=None, slaterWeightMin=0.0, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(x_sol, x_exact, rtol=1e-6, atol=1e-6)


def test_block_bicgstab_sparse_rank_deficient():
    """Sparse (ManyBodyState) path: two proportional RHS states solved as a block match the
    per-column solves to machine precision."""
    from mpi4py import MPI

    from impurityModel.ed.manybody_basis import Basis

    det0 = SlaterDeterminant.from_bytes(b"\x80")  # orbital 0 occupied
    z = 3.0 + 0.5j
    # A = z*I - H, with H a 0<->1 hopping.
    A = z - ManyBodyOperator({((0, "c"), (1, "a")): 0.7, ((1, "c"), (0, "a")): 0.7})

    def _basis():
        b = Basis(
            impurity_orbitals={0: [[0, 1]]},
            bath_states=({0: [[]]}, {0: [[]]}),
            initial_basis=[b"\x80", b"\x40"],
            verbose=False,
            comm=MPI.COMM_SELF,
        )
        b.add_states([b"\x80", b"\x40"])
        return b

    y_block = [ManyBodyState({det0: 1.0}), ManyBodyState({det0: 2.0})]  # proportional -> rank 1

    x_block = block_bicgstab(
        A,
        ManyBodyState.from_states([ManyBodyState(width=1), ManyBodyState(width=1)]),
        ManyBodyState.from_states(y_block),
        basis=_basis(),
        slaterWeightMin=0.0,
        atol=1e-10,
        rtol=1e-12,
    ).to_states()

    ref = [
        block_bicgstab(
            A,
            ManyBodyState(width=1),
            ManyBodyState.from_states([col]),
            basis=_basis(),
            slaterWeightMin=0.0,
            atol=1e-10,
            rtol=1e-12,
        ).to_states()[0]
        for col in y_block
    ]
    for k in range(2):
        diff = x_block[k] - ref[k]
        assert inner(diff, diff).real < 1e-18
    # Column 1 is exactly twice column 0 (linearity preserved through deflation).
    diff10 = x_block[1] - (x_block[0] + x_block[0])
    assert inner(diff10, diff10).real < 1e-18
