r"""TSQR: the orthonormalization primitive that replaced Gram-matrix Cholesky-QR.

What these tests pin down, in the order the algorithm runs:

* the packed triangular storage and the Givens merge that combines two upper triangles
  without ever materializing the stacked ``2p x p`` matrix,
* the panel-blocked local pass (its result must not depend on the panel height, and it must
  handle panels — and whole ranks — with fewer than ``p`` rows),
* the ``Allgather`` reduction, whose result has to be **bitwise** identical on every rank:
  the deflated block width is decided from it independently per rank, and the shrinking-block
  recurrence deadlocks or diverges if two ranks disagree,
* the deflation / breakdown / corruption contract (``k > 0`` / ``k == 0`` / ``k == -1``),
* and the reason the change was made at all: at a condition number the old rank floor
  deliberately *retains* (``kappa`` just under ``DEFLATE_TOL**-1``), a single Cholesky-QR
  leaves an orthonormality error above ``sqrt(EPS)`` — the semi-orthogonality level the
  reort machinery assumes — while TSQR stays at machine precision.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.BlockLanczosArray import _cholesky_or_deflate, block_tsqr
from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant
from impurityModel.ed.TSQR import (
    BREAKDOWN_TOL,
    DEFLATE_TOL,
    DEFLATE_TOL_SEEDS,
    EPS,
    local_r,
    merge_packed_r,
    pack_upper,
    packed_len,
    reduce_r,
    tsqr,
    tsqr_r,
    unpack_upper,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _block(n, p, seed=0):
    rng = _rng(seed)
    return rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p))


def _conditioned_block(n, p, kappa, seed=0):
    """Tall block with prescribed singular values spanning ``kappa``."""
    rng = _rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p)))
    V, _ = np.linalg.qr(rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p)))
    s = np.logspace(0.0, -np.log10(kappa), p)
    return (U * s) @ V.conj().T


def _upper(R):
    return np.allclose(R, np.triu(R), atol=0.0)


# ---------------------------------------------------------------------------
# Packed storage + Givens merge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [1, 2, 3, 5, 8])
def test_pack_unpack_roundtrip(p):
    M = np.triu(_block(p, p, seed=p))
    assert packed_len(p) == p * (p + 1) // 2
    v = pack_upper(M)
    assert v.size == packed_len(p)
    np.testing.assert_array_equal(unpack_upper(v, p), M)


@pytest.mark.parametrize("p", [1, 2, 3, 6])
def test_givens_merge_equals_dense_qr(p):
    """``merge(R1, R2)`` is the triangular factor of the stacked ``[R1; R2]``."""
    R1 = np.triu(_block(p, p, seed=1))
    R2 = np.triu(_block(p, p, seed=2))
    merged = unpack_upper(merge_packed_r(pack_upper(R1), pack_upper(R2)), p)

    assert _upper(merged)
    # The Gram of the stack is the invariant a QR factor must reproduce.
    np.testing.assert_allclose(merged.conj().T @ merged, R1.conj().T @ R1 + R2.conj().T @ R2, atol=1e-12)
    # Same singular values as the dense stack.
    np.testing.assert_allclose(
        np.linalg.svd(merged, compute_uv=False),
        np.linalg.svd(np.vstack([R1, R2]), compute_uv=False),
        rtol=1e-12,
    )


def test_givens_merge_does_not_modify_inputs():
    p = 4
    a, b = pack_upper(np.triu(_block(p, p, seed=3))), pack_upper(np.triu(_block(p, p, seed=4)))
    a0, b0 = a.copy(), b.copy()
    merge_packed_r(a, b)
    np.testing.assert_array_equal(a, a0)
    np.testing.assert_array_equal(b, b0)


def test_givens_merge_with_zero_operand_is_identity():
    """An empty rank contributes ``R = 0``; merging it must change nothing."""
    p = 4
    R = pack_upper(np.triu(_block(p, p, seed=5)))
    np.testing.assert_array_equal(merge_packed_r(R, np.zeros_like(R)), R)


# ---------------------------------------------------------------------------
# Local pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,p", [(200, 4), (37, 3), (5, 5), (3, 7), (1, 1), (1, 4)])
def test_local_r_reproduces_the_gram(n, p):
    """``R^H R == A^H A`` and ``R`` is upper triangular with a real non-negative diagonal."""
    A = _block(n, p, seed=n + p)
    R = unpack_upper(local_r(A), p)
    assert _upper(R)
    diag = np.diag(R)
    np.testing.assert_allclose(diag.imag, 0.0, atol=1e-14)
    assert np.all(diag.real >= 0.0)
    np.testing.assert_allclose(R.conj().T @ R, A.conj().T @ A, atol=1e-11 * max(1.0, n))


@pytest.mark.parametrize("panel", [1, 2, 7, 512, 5000])
def test_local_r_is_panel_invariant(panel):
    """The panel height is a blocking choice, not a numerical one."""
    A = _conditioned_block(997, 5, 1e6, seed=11)
    ref = local_r(A, 5000)
    got = local_r(A, panel)
    np.testing.assert_allclose(got, ref, atol=1e-14 * np.max(np.abs(ref)))


def test_local_r_of_empty_block_is_zero():
    np.testing.assert_array_equal(local_r(np.zeros((0, 3), dtype=complex)), np.zeros(6))


def test_local_r_accepts_non_contiguous_input():
    """Column slices of a Krylov buffer are handed in directly; ``A`` is never copied."""
    big = _block(50, 8, seed=7)
    view = big[:, 2:6]
    assert not view.flags["C_CONTIGUOUS"]
    before = big.copy()
    R = unpack_upper(local_r(view), 4)
    np.testing.assert_allclose(R.conj().T @ R, view.conj().T @ view, atol=1e-12)
    np.testing.assert_array_equal(big, before)


# ---------------------------------------------------------------------------
# Full factorization: accuracy, deflation, breakdown
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,p", [(500, 1), (500, 2), (500, 4), (64, 8), (9, 4)])
def test_tsqr_is_exact_and_orthonormal(n, p):
    A = _block(n, p, seed=n * p)
    Q, beta, k, sv = tsqr(A)
    assert k == p
    assert beta.shape == (p, p)
    assert _upper(beta)
    np.testing.assert_allclose(A, Q @ beta, atol=1e-13 * np.linalg.norm(A))
    np.testing.assert_allclose(Q.conj().T @ Q, np.eye(k), atol=1e-13)
    np.testing.assert_allclose(sv, np.linalg.svd(A, compute_uv=False), rtol=1e-12)


@pytest.mark.parametrize("p", [1, 2, 8, 15, 16, 17, 33])
def test_tsqr_solve_paths_agree_across_the_blas_crossover(p):
    """The back substitution has two implementations — a hand-rolled loop for narrow blocks
    and ``ztrsm`` for wide ones (``TRSM_BLAS_MIN_WIDTH``, chosen because OpenBLAS threads a
    ``m = p, n = rows`` solve by the wrong axis and spends ~1.2 ms of dispatch on ~30 us of
    work). Both sides of the crossover must be equally exact, on well- and ill-conditioned
    blocks alike."""
    for kappa in (None, 1e5):
        A = _block(2000, p, seed=p) if kappa is None else _conditioned_block(2000, p, kappa, seed=p)
        Q, beta, k, _ = tsqr(A)
        assert k == p
        np.testing.assert_allclose(A, Q @ beta, atol=1e-13 * np.linalg.norm(A))
        assert np.linalg.norm(Q.conj().T @ Q - np.eye(k)) < 1e-12


def test_tsqr_does_not_modify_its_input():
    A = _block(100, 4, seed=21)
    before = A.copy()
    tsqr(A)
    np.testing.assert_array_equal(A, before)


@pytest.mark.parametrize("kappa", [1e2, 1e4, 1e5])
def test_tsqr_beats_cholesky_qr_on_ill_conditioned_blocks(kappa):
    """The reason for the change.

    ``DEFLATE_TOL`` retains every direction down to ``kappa = DEFLATE_TOL**-1`` (~1.7e5), so
    these blocks are all *kept*, not deflated. A single Cholesky-QR of such a block leaves an
    orthonormality error of order ``kappa**2 * EPS`` — which is why the second CholeskyQR2
    pass had to exist. TSQR's back substitution is at machine precision without it.
    """
    A = _conditioned_block(400, 4, kappa, seed=int(np.log10(kappa)))
    Q, beta, k, _ = tsqr(A)
    assert k == 4, "the deflation floor should retain this block"
    np.testing.assert_allclose(A, Q @ beta, atol=1e-13 * np.linalg.norm(A))
    orth_tsqr = np.linalg.norm(Q.conj().T @ Q - np.eye(k))
    assert orth_tsqr < 1e-13

    beta_c, beta_inv, k_c = _cholesky_or_deflate(A.conj().T @ A, 4)
    assert k_c == k
    Qc = A @ beta_inv
    orth_chol = np.linalg.norm(Qc.conj().T @ Qc - np.eye(k_c))
    assert orth_tsqr <= orth_chol
    if kappa >= 1e5:
        # At the top of the retained range the single Cholesky pass is already worse than
        # the sqrt(EPS) semi-orthogonality level the reort estimator assumes.
        assert orth_chol > np.sqrt(EPS)


def test_tsqr_of_singular_block_deflates():
    """An exactly dependent column shrinks the block; the retained factor stays exact."""
    A = _block(200, 3, seed=31)
    A = np.hstack([A, 2.0 * A[:, :1]])  # column 3 == 2 * column 0
    Q, beta, k, sv = tsqr(A, scale=1.0)
    assert k == 3
    assert beta.shape == (3, 4)
    np.testing.assert_allclose(A, Q @ beta, atol=1e-12 * np.linalg.norm(A))
    np.testing.assert_allclose(Q.conj().T @ Q, np.eye(k), atol=1e-13)
    assert sv[3] < DEFLATE_TOL * sv[0]


def test_tsqr_rank_matches_the_singular_value_floor():
    """The rank is exactly ``#{sigma > DEFLATE_TOL * sigma_max}`` — no more, no less."""
    p = 5
    for n_keep in range(1, p + 1):
        s = np.ones(p)
        s[n_keep:] = DEFLATE_TOL / 100.0
        rng = _rng(n_keep)
        U, _ = np.linalg.qr(rng.standard_normal((80, p)) + 1j * rng.standard_normal((80, p)))
        V, _ = np.linalg.qr(rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p)))
        _, _, k, _ = tsqr((U * s) @ V.conj().T)
        assert k == n_keep


def _marginal_block(n, p, independent_part, seed=0):
    """``p`` orthonormal columns, the last of which is a copy of the first plus a genuinely
    independent component of size ``independent_part`` — a direction whose rank is a
    question of *where the floor is*, not of exact dependence."""
    rng = _rng(seed)
    u = np.linalg.qr(rng.standard_normal((n, p + 1)) + 1j * rng.standard_normal((n, p + 1)))[0]
    return np.hstack([u[:, : p - 1], u[:, :1] + independent_part * u[:, p : p + 1]])


@pytest.mark.parametrize("independent_part", [1e-4, 1e-7, 1e-10, 1e-13])
def test_deflate_tol_is_a_per_call_property_of_the_block(independent_part):
    """The rank floor is an argument, like ``scale``, because different callers are asking
    about different blocks.

    The default resolves directions down to the factorization's own noise, so that a
    restarted eigensolver can tell apart near-degenerate copies. A *seed* block — the stacked
    polarization components of a spectroscopy — needs the opposite: its symmetry-dependent
    components are zero only to their construction rounding, and retaining one injects a
    noise column into the solve.
    """
    a = _marginal_block(300, 4, independent_part)
    sigma_ratio = tsqr(a)[3][-1] / tsqr(a)[3][0]

    k_default = tsqr(a)[2]
    k_seeds = tsqr(a, deflate_tol=DEFLATE_TOL_SEEDS)[2]

    assert k_default == (4 if sigma_ratio > DEFLATE_TOL else 3)
    assert k_seeds == (4 if sigma_ratio > DEFLATE_TOL_SEEDS else 3)
    # ... and the whole point: between the two floors the answers differ.
    if DEFLATE_TOL < sigma_ratio <= DEFLATE_TOL_SEEDS:
        assert k_default == 4 and k_seeds == 3


def test_seeds_floor_is_looser_than_the_default():
    """Measured on the RIXS tensor benchmark, symmetry-dependent components are discarded at
    ``sigma/sigma_max`` up to 1.2e-9 — four orders above the default floor. The seeds floor
    has to sit above that with room for the rounding to grow with the basis."""
    assert DEFLATE_TOL_SEEDS > 1e-6 > 1e-9 > DEFLATE_TOL


def test_block_tsqr_forwards_the_rank_floor():
    """The dispatcher must pass the knob through for every representation, or the solvers
    that ask for the seeds floor would silently get the default."""
    a = _marginal_block(60, 3, 1e-10)
    dets = _dets(60)
    blk = ManyBodyState.from_states(_states(a, dets))
    for wp in (a, blk, blk.to_states()):
        assert block_tsqr(wp)[2] == 3
        assert block_tsqr(wp, deflate_tol=DEFLATE_TOL_SEEDS)[2] == 2


def test_tsqr_breakdown_is_relative_to_scale():
    """Breakdown asks "is this block zero *against something*"; deflation does not."""
    A = 1e-9 * _block(50, 3, seed=41)
    # Against its own O(1) scale the block is alive...
    assert tsqr(A, scale=1.0)[2] == 3
    # ...and against an operator norm of 1e6 it is numerically zero.
    assert tsqr(A, scale=1e6)[2] == 0
    assert tsqr(np.zeros((50, 3), dtype=complex))[2] == 0


def test_tsqr_breakdown_threshold():
    A = _block(50, 2, seed=42)
    A *= 0.5 * BREAKDOWN_TOL / np.linalg.norm(A, ord=2)
    assert tsqr(A, scale=1.0)[2] == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_tsqr_signals_corruption_not_breakdown(bad):
    """A non-finite factor is a corrupted recurrence (``-1``), not a closed space (``0``)."""
    A = _block(50, 3, seed=51)
    A[7, 1] = bad
    Q, beta, k, sv = tsqr(A)
    assert k == -1
    assert Q is None and beta is None


def test_tsqr_width_zero_block():
    Q, beta, k, sv = tsqr(np.zeros((10, 0), dtype=complex))
    assert k == 0 and Q.shape == (10, 0)


def test_tsqr_refinement_pass_is_folded_into_beta():
    """The conditional second pass must keep ``A = Q @ beta`` exact, not just orthonormalize."""
    A = _conditioned_block(300, 4, 1e5, seed=61)
    Q, beta, k, _ = tsqr(A, refine=True)
    Q1, beta1, k1, _ = tsqr(A, refine=False)
    assert k == k1 == 4
    np.testing.assert_allclose(A, Q @ beta, atol=1e-13 * np.linalg.norm(A))
    np.testing.assert_allclose(A, Q1 @ beta1, atol=1e-13 * np.linalg.norm(A))
    # Refinement can only improve orthonormality.
    assert np.linalg.norm(Q.conj().T @ Q - np.eye(k)) <= np.linalg.norm(Q1.conj().T @ Q1 - np.eye(k1)) + 1e-16


# ---------------------------------------------------------------------------
# block_tsqr: the same factorization through every block representation
# ---------------------------------------------------------------------------


def _states(A, dets):
    """The columns of ``A`` as width-1 ``ManyBodyState`` blocks over ``dets`` (zeros
    dropped). ``width=1`` even for an all-zero column -- a bare ``{}`` would be the
    width-0 polymorphic zero, which ``from_states`` (every caller's next step) rejects."""
    return [
        ManyBodyState({d: complex(A[i, c]) for i, d in enumerate(dets) if A[i, c] != 0}, width=1)
        for c in range(A.shape[1])
    ]


def _dets(n, offset=0):
    return [SlaterDeterminant((i + offset + 1,)) for i in range(n)]


def test_from_keys_and_amps_roundtrip():
    """The write-back keeps the support and accepts a narrower block than it was built from."""
    A = _block(30, 4, seed=91)
    src = ManyBodyState.from_states(_states(A, _dets(30)))
    np.testing.assert_allclose(np.asarray(src), A)

    same = ManyBodyState.from_keys_and_amps(src, A)
    assert [bytes(d) for d in same.keys()] == [bytes(d) for d in src.keys()]
    np.testing.assert_array_equal(np.asarray(same), A)

    narrow = ManyBodyState.from_keys_and_amps(src, A[:, :2])
    assert narrow.width == 2 and len(narrow) == len(src)
    np.testing.assert_array_equal(np.asarray(narrow), A[:, :2])

    with pytest.raises(ValueError):
        ManyBodyState.from_keys_and_amps(src, A[:5])


def test_block_tsqr_agrees_across_representations():
    """Array, ``ManyBodyState`` and ``ManyBodyState``-list inputs give the same factor."""
    A = _block(40, 3, seed=93)
    dets = _dets(40)
    blk = ManyBodyState.from_states(_states(A, dets))

    Q_arr, beta_arr, k_arr, _ = block_tsqr(A)
    Q_blk, beta_blk, k_blk, _ = block_tsqr(blk)
    Q_lst, beta_lst, k_lst, _ = block_tsqr(blk.to_states())

    assert k_arr == k_blk == k_lst == 3
    np.testing.assert_allclose(beta_blk, beta_arr, atol=1e-14)
    np.testing.assert_allclose(beta_lst, beta_arr, atol=1e-14)
    assert isinstance(Q_blk, ManyBodyState)
    assert isinstance(Q_lst, list) and isinstance(Q_lst[0], ManyBodyState)
    np.testing.assert_allclose(np.asarray(Q_blk), Q_arr, atol=1e-14)
    np.testing.assert_allclose(np.asarray(ManyBodyState.from_states(Q_lst)), Q_arr, atol=1e-14)
    # The support is carried through unchanged, and the input block is not disturbed.
    assert [bytes(d) for d in Q_blk.keys()] == [bytes(d) for d in blk.keys()]
    np.testing.assert_allclose(np.asarray(blk), A)


def test_block_tsqr_deflates_the_block_width():
    A = _block(40, 3, seed=95)
    A = np.hstack([A, 3.0 * A[:, :1]])
    blk = ManyBodyState.from_states(_states(A, _dets(40)))
    Q, beta, k, _ = block_tsqr(blk)
    assert k == 3 and Q.width == 3 and beta.shape == (3, 4)
    np.testing.assert_allclose(np.asarray(Q) @ beta, A, atol=1e-12 * np.linalg.norm(A))


def test_block_tsqr_breakdown_returns_no_block():
    blk = ManyBodyState.from_states(_states(np.zeros((10, 3), dtype=complex), _dets(10)))
    Q, beta, k, _ = block_tsqr(blk)
    assert k == 0 and Q is None


def test_block_tsqr_zero_row_block():
    """A block with no determinants has no buffer to export; it must still answer."""
    empty = ManyBodyState.from_states([ManyBodyState({}, width=1) for _ in range(3)])
    assert len(empty) == 0
    assert block_tsqr(empty)[2] == 0


def test_block_tsqr_releases_the_buffer_view():
    """The exported view must be gone by the time the caller mutates the block's rows."""
    A = _block(20, 2, seed=97)
    blk = ManyBodyState.from_states(_states(A, _dets(20)))
    block_tsqr(blk)
    blk.prune_rows(1e-3)  # raises RuntimeError if a view were still exported


# ---------------------------------------------------------------------------
# Distributed
# ---------------------------------------------------------------------------


def _row_slice(A, comm, empty_last=False):
    """This rank's contiguous row block of a globally-known ``A``."""
    n = A.shape[0]
    size = comm.size
    if empty_last and size > 1:
        counts = [0] * size
        base, rem = divmod(n, size - 1)
        for r in range(size - 1):
            counts[r] = base + (1 if r < rem else 0)
    else:
        base, rem = divmod(n, size)
        counts = [base + (1 if r < rem else 0) for r in range(size)]
    off = sum(counts[: comm.rank])
    return np.ascontiguousarray(A[off : off + counts[comm.rank]])


@pytest.mark.mpi
@pytest.mark.parametrize("empty_last", [False, True])
def test_tsqr_mpi_matches_serial(empty_last):
    comm = MPI.COMM_WORLD
    A = _block(120, 4, seed=71)
    R_serial = tsqr_r(A, None)
    R_dist = tsqr_r(_row_slice(A, comm, empty_last), comm)
    np.testing.assert_allclose(R_dist, R_serial, atol=1e-12 * np.max(np.abs(R_serial)))


@pytest.mark.mpi
@pytest.mark.parametrize("empty_last", [False, True])
def test_tsqr_mpi_is_bitwise_identical_across_ranks(empty_last):
    """Every rank decides the deflated block width from ``R`` on its own — so ``R`` must be
    bit-identical, not merely close, or two ranks can pick different widths and the
    shrinking-block recurrence desynchronizes."""
    comm = MPI.COMM_WORLD
    A = _conditioned_block(200, 4, 1e5, seed=73)
    A_local = _row_slice(A, comm, empty_last)
    R = tsqr_r(A_local, comm)
    R_root = comm.bcast(R.tobytes(), root=0)
    assert R.tobytes() == R_root

    _, beta, k, sv = tsqr(A_local, comm)
    assert comm.bcast(k, root=0) == k
    assert comm.bcast(beta.tobytes(), root=0) == beta.tobytes()


@pytest.mark.mpi
@pytest.mark.parametrize("empty_last", [False, True])
def test_tsqr_mpi_q_is_globally_orthonormal(empty_last):
    comm = MPI.COMM_WORLD
    A = _block(120, 3, seed=75)
    A_local = _row_slice(A, comm, empty_last)
    Q, beta, k, _ = tsqr(A_local, comm)
    assert k == 3
    assert Q.shape == (A_local.shape[0], k)

    gram = np.ascontiguousarray(Q.conj().T @ Q)
    comm.Allreduce(MPI.IN_PLACE, gram, op=MPI.SUM)
    np.testing.assert_allclose(gram, np.eye(k), atol=1e-12)

    resid = np.linalg.norm(A_local - Q @ beta) ** 2
    resid = comm.allreduce(resid, op=MPI.SUM)
    assert np.sqrt(resid) < 1e-12 * np.linalg.norm(A)


@pytest.mark.mpi
def test_tsqr_mpi_deflation_agrees_on_every_rank():
    comm = MPI.COMM_WORLD
    A = _block(120, 3, seed=77)
    A = np.hstack([A, A[:, :1] - 3.0 * A[:, 2:3]])
    Q, beta, k, _ = tsqr(_row_slice(A, comm), comm)
    assert k == 3
    assert comm.allreduce(k, op=MPI.MIN) == comm.allreduce(k, op=MPI.MAX) == 3
    gram = np.ascontiguousarray(Q.conj().T @ Q)
    comm.Allreduce(MPI.IN_PLACE, gram, op=MPI.SUM)
    np.testing.assert_allclose(gram, np.eye(k), atol=1e-12)


@pytest.mark.mpi
def test_tsqr_mpi_breakdown_and_corruption_agree():
    comm = MPI.COMM_WORLD
    zero_local = np.zeros((120 // comm.size + 1, 3), dtype=complex)
    assert tsqr(zero_local, comm)[2] == 0

    # One rank's rows are poisoned; every rank must report corruption.
    A_local = _row_slice(_block(120, 3, seed=79), comm)
    if comm.rank == comm.size - 1 and A_local.shape[0] > 0:
        A_local = A_local.copy()
        A_local[0, 0] = np.nan
    assert tsqr(A_local, comm)[2] == -1


@pytest.mark.mpi
@pytest.mark.parametrize("empty_last", [False, True])
def test_block_tsqr_mpi_on_a_distributed_block_state(empty_last):
    """The determinant-distributed block: each rank owns a disjoint slice of the support."""
    comm = MPI.COMM_WORLD
    A = _block(90, 3, seed=99)
    dets = _dets(90)
    A_local = _row_slice(A, comm, empty_last)
    off = comm.scan(A_local.shape[0]) - A_local.shape[0]
    local = ManyBodyState.from_states(_states(A_local, dets[off : off + A_local.shape[0]]))

    Q, beta, k, sv = block_tsqr(local, True, comm)
    assert k == 3
    assert comm.bcast(beta.tobytes(), root=0) == beta.tobytes()

    Q_arr = np.zeros((0, k), dtype=complex) if len(Q) == 0 else np.asarray(Q)
    gram = np.ascontiguousarray(Q_arr.conj().T @ Q_arr)
    comm.Allreduce(MPI.IN_PLACE, gram, op=MPI.SUM)
    np.testing.assert_allclose(gram, np.eye(k), atol=1e-12)

    resid = comm.allreduce(np.linalg.norm(A_local - Q_arr @ beta) ** 2, op=MPI.SUM)
    assert np.sqrt(resid) < 1e-12 * np.linalg.norm(A)
    np.testing.assert_allclose(beta, tsqr(A)[1], atol=1e-12 * np.max(np.abs(beta)))


@pytest.mark.mpi
def test_reduce_r_ignores_rank_order_of_empty_contributions():
    """A rank owning no rows contributes a zero triangle; the global R is unchanged."""
    comm = MPI.COMM_WORLD
    A = _block(60, 3, seed=81)
    all_on_rank0 = A if comm.rank == 0 else np.zeros((0, 3), dtype=complex)
    np.testing.assert_allclose(tsqr_r(all_on_rank0, comm), tsqr_r(A, None), atol=1e-14)
    # reduce_r is collective and its serial path must be a no-op.
    R = local_r(A)
    np.testing.assert_array_equal(reduce_r(R, None), R)
