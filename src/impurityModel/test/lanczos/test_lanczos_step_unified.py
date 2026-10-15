"""The block-Lanczos step is ONE piece of code for both block representations.

``lanczos_step`` (``_lanczos_step.pxi``) touches its blocks only through the dispatching primitives
of ``_block_ops.pxi``; the ``ndarray`` and the ``ManyBodyState`` front ends differ in how they build
the starting blocks, the matvec and the Krylov store, not in the recurrence. These tests pin that
claim directly: the same operator and the same seed through both front ends give the same block
tridiagonal matrix (compared through gauge-free quantities, because TSQR's R is fixed only up to the
row order it sees), and the primitives and the dense store the array arm is built on behave.

Serial: the multi-rank behaviour of the loop is covered by the existing ``-n 2``/``-n 3`` tests.
"""

import numpy as np
import pytest

from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.BlockLanczosArray import Reort, block_lanczos_array
from impurityModel.ed.BlockLanczosCore import (
    DenseKrylovStore,
    block_add_scaled,
    eigh_block_tridiagonal,
    lanczos_step,
)
from impurityModel.ed.DistributedSparseMatrix import DistributedSparseMatrix
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.test.support.lanczos_fixtures import MockBasis, build_dense_matrix_from_manybody, get_test_system

P = 2
STEPS = 6


def _system():
    h_op, n, _eigvals, basis_states = get_test_system()
    h_dense = build_dense_matrix_from_manybody(h_op, basis_states)
    rng = np.random.default_rng(7)
    seed = rng.standard_normal((n, P)) + 1j * rng.standard_normal((n, P))
    seed, _ = np.linalg.qr(seed)
    return h_op, h_dense, basis_states, seed


def _seed_states(basis_states, seed):
    """The columns of ``seed`` as ManyBodyStates over the same determinants (dense row i = determinant i)."""
    states = []
    for col in range(seed.shape[1]):
        st = ManyBodyState()
        for i, b in enumerate(basis_states):
            st += b * seed[i, col]
        states.append(st)
    return states


def _gauge_free(alphas, betas, widths):
    """Spectrum of T and the singular values of every beta: invariant under the column gauge of each block."""
    ev, _ = eigh_block_tridiagonal(alphas, betas, block_widths=widths, eigvals_only=True)
    sv = [np.linalg.svd(np.asarray(b), compute_uv=False) for b in betas]
    tr = [np.trace(np.asarray(a)).real for a in alphas]
    return np.asarray(ev), sv, tr


@pytest.mark.parametrize("reort", [Reort.NONE, Reort.FULL])
def test_array_and_manybodystate_front_ends_build_the_same_block_tridiagonal(reort):
    h_op, h_dense, basis_states, seed = _system()
    matrix = DistributedSparseMatrix.from_dense(h_dense)

    a_arr, b_arr, _q, w_arr = block_lanczos_array(
        psi0=seed.copy(),
        h_op=matrix,
        converged=lambda *a, **k: False,
        reort=reort,
        max_iter=STEPS,
        return_W=False,
        return_widths=True,
    )[:4]
    a_mbs, b_mbs, _qs, _w, w_mbs = block_lanczos_cy(
        _seed_states(basis_states, seed),
        h_op,
        MockBasis(len(basis_states)),
        lambda *a, **k: False,
        reort=reort,
        max_iter=STEPS,
        return_widths=True,
    )
    assert list(w_arr) == list(w_mbs)
    ev_a, sv_a, tr_a = _gauge_free(a_arr, b_arr, w_arr)
    ev_m, sv_m, tr_m = _gauge_free(a_mbs, b_mbs, w_mbs)
    n = min(len(ev_a), len(ev_m))
    np.testing.assert_allclose(ev_a[:n], ev_m[:n], atol=1e-10)
    for sa, sm in zip(sv_a, sv_m):
        np.testing.assert_allclose(sa, sm, atol=1e-10)
    np.testing.assert_allclose(tr_a, tr_m, atol=1e-10)


def test_one_step_gives_the_same_alpha_for_an_array_block_and_its_manybodystate_block():
    h_op, h_dense, basis_states, seed = _system()
    n = len(basis_states)
    alphas = np.zeros((1, P, P), dtype=complex)
    betas = np.zeros((1, P, P), dtype=complex)
    _qn, alpha_arr, _beta, norm_arr, k_arr, broke = lanczos_step(
        lambda q: h_dense @ q,
        np.zeros((n, P), dtype=complex),
        seed.copy(),
        None,
        alphas,
        betas,
        0,
        Reort.NONE,
        False,
        None,
    )
    assert not broke and k_arr == P

    from impurityModel.ed.BlockLanczosCore import apply_and_redistribute

    q_mbs = ManyBodyState.from_states(_seed_states(basis_states, seed))
    alphas2 = np.zeros((1, P, P), dtype=complex)
    betas2 = np.zeros((1, P, P), dtype=complex)
    basis = MockBasis(n)
    _qn2, alpha_mbs, _beta2, norm_mbs, k_mbs, broke2 = lanczos_step(
        lambda q: apply_and_redistribute(h_op, q, basis, 0.0, False, 1),
        ManyBodyState(width=P),
        q_mbs,
        None,
        alphas2,
        betas2,
        0,
        Reort.NONE,
        False,
        None,
    )
    assert not broke2 and k_mbs == P
    # alpha = X^H A X is gauge-covariant, not invariant: here the two blocks are the same matrix row for row.
    np.testing.assert_allclose(alpha_arr, alpha_mbs, atol=1e-12)
    np.testing.assert_allclose(norm_arr, norm_mbs, atol=1e-10)


@pytest.mark.parametrize("representation", ["array", "manybody"])
def test_the_locked_vectors_are_projected_out_in_both_representations(representation):
    h_op, h_dense, basis_states, seed = _system()
    n = len(basis_states)
    evals, evecs = np.linalg.eigh(h_dense)
    locked = evecs[:, :2]
    # A seed with a component along the locked pair: without the projection q_next keeps it.
    start = seed + 0.7 * locked @ np.array([[1.0, 0.3], [0.2, 1.0]])
    start, _ = np.linalg.qr(start)
    alphas = np.zeros((1, P, P), dtype=complex)
    betas = np.zeros((1, P, P), dtype=complex)
    if representation == "array":
        q_next, *_ = lanczos_step(
            lambda q: h_dense @ q,
            np.zeros((n, P), dtype=complex),
            start.copy(),
            None,
            alphas,
            betas,
            0,
            Reort.NONE,
            False,
            None,
            locked=locked,
        )
        dense_next = q_next
    else:
        from impurityModel.ed.BlockLanczosCore import apply_and_redistribute

        basis = MockBasis(n)
        q_next, *_ = lanczos_step(
            lambda q: apply_and_redistribute(h_op, q, basis, 0.0, False, 1),
            ManyBodyState(width=P),
            ManyBodyState.from_states(_seed_states(basis_states, start)),
            None,
            alphas,
            betas,
            0,
            Reort.NONE,
            False,
            None,
            locked=ManyBodyState.from_states(_seed_states(basis_states, locked)),
        )
        keys = [next(iter(b.keys())) for b in basis_states]
        dense_next = np.zeros((n, q_next.width), dtype=complex)
        for col, st in enumerate(q_next.to_states()):
            for i, k in enumerate(keys):
                amp = st.get(k)
                dense_next[i, col] = 0.0 if amp is None else amp[0]
    np.testing.assert_allclose(locked.conj().T @ dense_next, 0.0, atol=1e-10)


def test_add_scaled_in_place_matches_numpy_on_awkward_inputs():
    rng = np.random.default_rng(3)
    n, p, k = 40, 3, 2
    W = rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k))
    alpha = rng.standard_normal((k, p)) + 1j * rng.standard_normal((k, p))
    base = rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p))
    want = base + W @ alpha

    np.testing.assert_allclose(block_add_scaled(base.copy(), W, alpha), want, atol=1e-13)
    # Fortran order, a non-contiguous slice of a wider array: not the fast path, still right.
    np.testing.assert_allclose(block_add_scaled(np.asfortranarray(base), W, alpha), want, atol=1e-13)
    wide = np.zeros((n, 2 * p), dtype=complex)
    wide[:, ::2] = base
    out = block_add_scaled(wide[:, ::2], W, alpha)
    np.testing.assert_allclose(out, want, atol=1e-13)
    # A strided W is made contiguous first; a real alpha is promoted.
    np.testing.assert_allclose(block_add_scaled(base.copy(), W[::1], alpha.real), base + W @ alpha.real, atol=1e-13)
    # V += V @ alpha: the output overlaps an input, which numpy built a temporary for.
    square = rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p))
    sq_alpha = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
    np.testing.assert_allclose(
        block_add_scaled(square.copy(), square, sq_alpha), square + square @ sq_alpha, atol=1e-12
    )
    # The BLAS call cannot check shapes, so the dispatcher must.
    with pytest.raises(ValueError):
        block_add_scaled(base.copy(), W, alpha[:1])


def test_the_dense_krylov_store_grows_adopts_and_projects():
    rng = np.random.default_rng(11)
    n = 30
    blocks = [rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2)) for _ in range(9)]
    store = DenseKrylovStore(n)
    for b in blocks:
        store.append_block(b)
    full = np.concatenate(blocks, axis=1)
    assert len(store) == full.shape[1]
    np.testing.assert_array_equal(store.array(), full)

    # Adopting a caller's array never writes to it: the first append reallocates.
    mine = full.copy()
    adopted = DenseKrylovStore.from_array(mine)
    adopted.append_block(blocks[0])
    np.testing.assert_array_equal(mine, full)
    assert len(adopted) == full.shape[1] + 2

    # Reorthogonalization against all columns, and against a subset.
    Q, _ = np.linalg.qr(full)
    wp = rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2))
    store_q = DenseKrylovStore(n)
    store_q.append_block(Q[:, :6])
    out, overlap = store_q.reort(wp.copy(), None, 2, None)
    np.testing.assert_allclose(Q[:, :6].conj().T @ out, 0.0, atol=1e-12)
    out_sub, _ = store_q.reort(wp.copy(), [1, 3], 2, None)
    np.testing.assert_allclose(Q[:, [1, 3]].conj().T @ out_sub, 0.0, atol=1e-12)
