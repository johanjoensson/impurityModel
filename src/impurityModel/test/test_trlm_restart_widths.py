"""Thick-restart block Lanczos: the retained Ritz block is not always ``nkeep`` wide.

``_trlm_core`` retains ``nkeep = ceil(num_wanted / p) * p`` Ritz vectors per restart and
addresses ``T_full`` and ``Q_basis`` by cumulative block offsets. Both thick-restart
coefficient shortcuts -- ``diag(theta_keep)`` for the retained block and
``beta_res @ Y_last`` for the arrowhead spike -- are derived from ``Q^H Q = I``. Under
``reort=NONE`` the recurrence loses orthogonality, spawns ghost copies of converged Ritz
values, and the retained block ``X = Q_basis[:D] @ Y_k`` can be rank deficient, so
``block_normalize`` deflates it to ``k_ret < nkeep`` columns.

The kernel used to record ``nkeep`` regardless. Two failures followed:

* ``sum(cur_widths)`` outran ``Q_basis``'s column count and the *next* restart's
  ``block_combine`` died with an opaque ``ValueError: matmul: ... size 120 is different
  from 118`` (array path, the clustered system below at ``m=40``);
* where the shapes happened to line up, the restart silently paired Ritz values with the
  wrong vectors -- the ManyBodyState system below returned eigenvalues off by ``5.1e3``
  on a spectrum bounded by ``|E| <= 5``.

These tests pin the invariant (widths track the stored columns) and the accuracy that
falls out of restoring it. ``test_trlm_partial_reort_never_rebuilds_the_restart`` guards
the other direction: the semi-orthogonal modes must keep the free coefficient path.
"""

import contextlib
import io
import re

import numpy as np
import pytest
import scipy.sparse as sps

try:
    from mpi4py import MPI

    _has_mpi = True
except ImportError:  # pragma: no cover - mpi4py is a hard dependency in practice
    _has_mpi = False

from impurityModel.ed.BlockLanczos import (
    _check_width_sync,
    _thick_restart_block_lanczos_array,
    thick_restart_block_lanczos_cy,
)
from impurityModel.ed.BlockLanczosArray import RESTART_ORTH_TOL, Reort, block_normalize
from impurityModel.ed.greens_function import _trim_blocks
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant
from impurityModel.test.test_restarted_lanczos import MockBasis

# A spectrum with three eigenvalues inside 2e-9 of each other. Under reort=NONE the
# recurrence cannot keep them apart: the Krylov basis loses orthogonality entirely
# (measured ||Q^H Q - I|| = 1.0 at the first restart) and ghost copies appear among the
# lowest Ritz values, which is exactly what makes the retained block rank deficient.
_CLUSTER = [-5.0, -5.0 + 1e-9, -5.0 + 2e-9, -4.999]
_NUM_WANTED = 10
_BLOCK_WIDTH = 3


def _clustered_hermitian(n, seed=3):
    """Hermitian ``(n, n)`` with the tight low-lying cluster above and a flat tail."""
    rng = np.random.default_rng(seed)
    evals = np.concatenate([_CLUSTER, np.linspace(-1.0, 5.0, n - len(_CLUSTER))])
    u = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    h = (u * evals) @ np.conj(u.T)
    return 0.5 * (h + np.conj(h.T)), np.sort(evals), rng


def _one_particle_operator(t):
    """A ``ManyBodyOperator`` whose one-particle sector is exactly the matrix ``t``.

    With a single fermion there are no anticommutation signs, so the many-body matrix in
    the ``n_orb`` single-occupancy determinants *is* ``t`` (up to the orbital ordering,
    which leaves the spectrum alone). It is the cheapest way to drive the ManyBodyState
    kernel with a designed spectrum.
    """
    n_orb = t.shape[0]
    n_bytes = (n_orb + 7) // 8
    op = {((i, "c"), (j, "a")): complex(t[i, j]) for i in range(n_orb) for j in range(n_orb)}

    def determinant(i):
        # Orbital i is bit (7 - i % 8) of byte i // 8 -- the MSB-first convention.
        raw = bytearray(n_bytes)
        raw[i // 8] |= 1 << (7 - i % 8)
        return SlaterDeterminant.from_bytes(bytes(raw))

    return ManyBodyOperator(op), [ManyBodyState({determinant(i): 1.0}) for i in range(n_orb)]


def _random_block(basis_states, rng, width):
    block = []
    for _ in range(width):
        state = ManyBodyState()
        for b in basis_states:
            state += b * complex(rng.random() - 0.5, rng.random() - 0.5)
        block.append(state)
    return block


def _restart_reports(captured):
    """(orth_err, k_ret, nkeep) for every restart, from the verbose TRLM log."""
    pattern = r"retained block \|\|Q\^H Q - I\|\| = ([0-9.e+-]+), rank (\d+)/(\d+)"
    return [(float(a), int(b), int(c)) for a, b, c in re.findall(pattern, captured)]


# --------------------------------------------------------------------------------------
# The invariant itself
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("as_list", [False, True])
def test_check_width_sync_accepts_an_uncounted_trailing_residual_block(as_list):
    q = [object()] * 9 if as_list else np.zeros((4, 9))
    _check_width_sync(q, [3, 3], "sweep")  # 6 counted, 9 stored: the residual is extra


@pytest.mark.parametrize("as_list", [False, True])
def test_check_width_sync_rejects_a_counted_but_unstored_block(as_list):
    q = [object()] * 6 if as_list else np.zeros((4, 6))
    with pytest.raises(RuntimeError, match="never stored"):
        _check_width_sync(q, [3, 3, 3], "sweep")


def test_check_width_sync_exact_rejects_a_trailing_block():
    """The restart loop has already split the residual off, so equality is required."""
    q = np.zeros((4, 9))
    _check_width_sync(q, [3, 3, 3], "restart", exact=True)
    with pytest.raises(RuntimeError, match="must be exactly the ones"):
        _check_width_sync(q, [3, 3], "restart", exact=True)


def test_trim_blocks_rejects_a_length_mismatch():
    """``k = len(widths)`` would otherwise silently shorten the continued fraction."""
    alphas = np.zeros((4, 2, 2), dtype=complex)
    betas = np.zeros((4, 2, 2), dtype=complex)
    _trim_blocks(alphas, betas, [2, 2, 2, 2])
    with pytest.raises(ValueError, match="silently use only the first"):
        _trim_blocks(alphas, betas, [2, 2])


# --------------------------------------------------------------------------------------
# Array path
# --------------------------------------------------------------------------------------


def test_array_trlm_survives_a_rank_deficient_retained_ritz_block():
    """Pre-fix: ``ValueError: matmul: ... size 120 is different from 118`` at restart 1."""
    n = 200
    h, evals, rng = _clustered_hermitian(n)

    class _Basis:
        size = n

    psi0 = np.linalg.qr(rng.standard_normal((n, _BLOCK_WIDTH)) + 0j)[0]
    got, vecs = _thick_restart_block_lanczos_array(
        psi0, h, _Basis(), _NUM_WANTED, 40, 1e-13, 30, False, Reort.NONE, None
    )

    assert len(got) == _NUM_WANTED
    np.testing.assert_allclose(np.sort(got.real), evals[:_NUM_WANTED], atol=1e-9)
    # Every returned pair is an eigenpair of h, not just a number that looks right.
    residuals = np.linalg.norm(h @ vecs - vecs * got[None, :], axis=0)
    assert np.max(residuals) < 1e-6, residuals
    np.testing.assert_allclose(np.conj(vecs.T) @ vecs, np.eye(_NUM_WANTED), atol=1e-10)


def test_trlm_rebuilds_the_restart_when_orthogonality_is_lost():
    """reort=NONE loses the ``Q^H Q = I`` premise the coefficient shortcuts rest on."""
    n = 200
    h, evals, rng = _clustered_hermitian(n)

    class _Basis:
        size = n

    psi0 = np.linalg.qr(rng.standard_normal((n, _BLOCK_WIDTH)) + 0j)[0]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got, _ = _thick_restart_block_lanczos_array(
            psi0, h, _Basis(), _NUM_WANTED, 40, 1e-13, 30, True, Reort.NONE, None
        )

    reports = _restart_reports(buf.getvalue())
    assert reports, "TRLM never restarted; the test system stopped exercising the branch"
    assert any(err > RESTART_ORTH_TOL for err, _, _ in reports), reports
    np.testing.assert_allclose(np.sort(got.real), evals[:_NUM_WANTED], atol=1e-9)


@pytest.mark.parametrize("mode", [Reort.PARTIAL, Reort.FULL])
def test_trlm_partial_reort_never_rebuilds_the_restart(mode):
    """The semi-orthogonal modes must keep the free (matvec-less) coefficient path.

    PARTIAL maintains ``||Q^H Q - I|| ~ sqrt(EPS)`` by construction, so it must stay under
    ``RESTART_ORTH_TOL`` and never pay for the Rayleigh-Ritz rebuild.
    """
    n = 200
    h, evals, rng = _clustered_hermitian(n)

    class _Basis:
        size = n

    psi0 = np.linalg.qr(rng.standard_normal((n, _BLOCK_WIDTH)) + 0j)[0]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got, _ = _thick_restart_block_lanczos_array(psi0, h, _Basis(), _NUM_WANTED, 40, 1e-13, 30, True, mode, None)

    for err, k_ret, nkeep in _restart_reports(buf.getvalue()):
        assert err <= RESTART_ORTH_TOL, f"{mode} lost semi-orthogonality: {err:.2e}"
        assert k_ret == nkeep
    np.testing.assert_allclose(np.sort(got.real), evals[:_NUM_WANTED], atol=1e-9)


# --------------------------------------------------------------------------------------
# ManyBodyState path -- the same _trlm_core, driven with list[ManyBodyState] blocks
# --------------------------------------------------------------------------------------


def test_manybodystate_trlm_survives_a_rank_deficient_retained_ritz_block():
    """Pre-fix: eigenvalues off by 5.1e3 on a spectrum bounded by |E| <= 5, silently."""
    n_orb = 100
    t, evals, rng = _clustered_hermitian(n_orb)
    h_op, basis_states = _one_particle_operator(t)
    psi0 = _random_block(basis_states, rng, _BLOCK_WIDTH)

    got, vecs = thick_restart_block_lanczos_cy(
        psi0, h_op, MockBasis(n_orb), _NUM_WANTED, 30, 1e-13, 30, False, 0.0, 0, "none", None
    )

    assert len(got) == _NUM_WANTED
    np.testing.assert_allclose(np.sort(np.real(got)), evals[:_NUM_WANTED], atol=1e-8)
    # The rebuilt restart must leave genuine eigenpairs behind.
    for psi, theta in zip(vecs, np.asarray(got)):
        residual = h_op.apply(psi) - psi * complex(theta)
        assert np.sqrt(residual.norm2()) < 1e-6


# --------------------------------------------------------------------------------------
# MPI: the rebuild branch issues four collectives behind a data-dependent condition
# --------------------------------------------------------------------------------------


@pytest.mark.mpi
@pytest.mark.skipif(not _has_mpi, reason="mpi4py not available")
def test_trlm_rank_deficient_restart_is_collective():
    """Every rank must take the same branch, or the extra collectives deadlock.

    ``k_ret`` and ``orth_err`` are read off an Allreduced Gram matrix, so the decision is
    bit-identical on every rank. If it ever became rank-local this test hangs rather than
    failing -- which is the point: a rank-local branch around ``block_apply`` /
    ``block_inner`` / ``block_orthogonalize`` / ``block_normalize`` is a deadlock.
    """
    comm = MPI.COMM_WORLD
    n = 200
    h, evals, rng = _clustered_hermitian(n)

    counts = [n // comm.size + (1 if r < n % comm.size else 0) for r in range(comm.size)]
    c0 = sum(counts[: comm.rank])
    c1 = c0 + counts[comm.rank]
    h_local = sps.csr_matrix(h[:, c0:c1])

    class _Basis:
        def __init__(self, comm):
            self.comm = comm
            self.size = n

    psi0_full = np.linalg.qr(rng.standard_normal((n, _BLOCK_WIDTH)) + 0j)[0]
    psi0, _ = block_normalize(np.ascontiguousarray(psi0_full[c0:c1, :]), mpi=True, comm=comm)

    got, _ = _thick_restart_block_lanczos_array(
        psi0, h_local, _Basis(comm), _NUM_WANTED, 40, 1e-13, 30, False, Reort.NONE, comm
    )

    np.testing.assert_allclose(np.sort(got.real), evals[:_NUM_WANTED], atol=1e-8)
    # Every rank agrees on the answer (the Ritz values are broadcast from rank 0).
    spread = comm.allreduce(float(np.max(np.abs(got - comm.bcast(got, root=0)))), op=MPI.MAX)
    assert spread == 0.0
