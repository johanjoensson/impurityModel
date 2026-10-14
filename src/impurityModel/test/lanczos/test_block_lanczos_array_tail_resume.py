"""The array kernel resumed from its last two blocks (``tail=``) continues the recurrence it was cut from.

A reort-'none' recurrence keeps no Krylov basis, so the only state a continuation needs is the coefficients so
far, their widths and the (previous, current) blocks. ``block_Green_sparse`` hands a frozen sparse recurrence to
the CSR this way, and a tail-only run that spent its budget continues from the tail ``return_tail`` gives back.
The oracle is one uninterrupted run: stopping after ``k`` blocks and resuming must give the same coefficients,
including across a step where the block narrowed (a seed column whose chain closed).
"""

import numpy as np
import pytest
import scipy.sparse as sps
from mpi4py import MPI

from impurityModel.ed.BlockLanczosArray import Reort, block_lanczos_array
from impurityModel.test.support.lanczos_fixtures import _contiguous_counts_with_empty_last

N_BLOCK_A = 4  # a closed 4-dimensional block: the seed column inside it deflates after 4 steps
N = 40
STEPS = 14


def _hamiltonian():
    rng = np.random.default_rng(3)
    H = np.zeros((N, N), dtype=complex)
    for lo, hi in ((0, N_BLOCK_A), (N_BLOCK_A, N)):
        m = rng.normal(size=(hi - lo, hi - lo)) + 1j * rng.normal(size=(hi - lo, hi - lo))
        H[lo:hi, lo:hi] = m + m.conj().T
    return H


def _seeds(width):
    rng = np.random.default_rng(5)
    psi = np.zeros((N, width), dtype=complex)
    psi[:N_BLOCK_A, 0] = rng.normal(size=N_BLOCK_A)
    psi[N_BLOCK_A:, 1:] = rng.normal(size=(N - N_BLOCK_A, width - 1))
    return np.linalg.qr(psi)[0]


def _partition(comm):
    if comm is None:
        return 0, N
    counts = _contiguous_counts_with_empty_last(N, comm.size)
    lo = int(sum(counts[: comm.rank]))
    return lo, lo + counts[comm.rank]


def _run(h_local, comm, max_iter, **kw):
    return block_lanczos_array(
        h_op=h_local,
        converged=lambda a, b, **_: False,
        max_iter=max_iter,
        reort=Reort.NONE,
        build_krylov_basis=False,
        return_widths=True,
        return_status=True,
        return_tail=True,
        comm=comm,
        **kw,
    )


def _pieces(comm=None, width=3):
    H = _hamiltonian()
    lo, hi = _partition(comm)
    return sps.csr_matrix(H[:, lo:hi]), np.ascontiguousarray(_seeds(width)[lo:hi])


def _resumed(h_local, psi0, comm, k):
    """Run ``k`` blocks, then resume from the returned tail for the rest."""
    if k == 0:
        width = psi0.shape[1]
        alphas = np.empty((0, width, width), dtype=complex)
        betas = np.empty((0, width, width), dtype=complex)
        tail, widths = (np.zeros_like(psi0), psi0), []
    else:
        alphas, betas, _, widths, status, tail = _run(h_local, comm, k, psi0=psi0)
        assert status == "max_iter" and tail is not None
        alphas, betas = alphas.copy(), betas.copy()
    return _run(h_local, comm, STEPS - k, psi0=None, alphas=alphas, betas=betas, tail=tail, block_widths_init=widths)


@pytest.mark.parametrize("k", [0, 1, 3, 4, 5, 9])
def test_a_resumed_recurrence_is_the_uninterrupted_one(k):
    h_local, psi0 = _pieces()
    whole = _run(h_local, None, STEPS, psi0=psi0)
    assert whole[3][:6] == [3, 3, 3, 3, 2, 2], f"premise: the block narrows after step 4 ({whole[3]})"
    resumed = _resumed(h_local, psi0, None, k)
    np.testing.assert_array_equal(resumed[0], whole[0])
    np.testing.assert_array_equal(resumed[1], whole[1])
    assert resumed[3] == whole[3] and resumed[4] == whole[4] == "max_iter"


def test_the_tail_is_returned_only_after_a_budget_stop():
    h_local, psi0 = _pieces()
    *_, status, tail = _run(h_local, None, 3, psi0=psi0)
    assert status == "max_iter" and tail[0].shape == (N, 3) and tail[1].shape == (N, 3)
    stopped = block_lanczos_array(
        psi0=psi0,
        h_op=h_local,
        converged=lambda a, b, **_: len(a) >= 2,
        max_iter=STEPS,
        reort=Reort.NONE,
        build_krylov_basis=False,
        return_widths=True,
        return_status=True,
        return_tail=True,
    )
    assert stopped[4] == "converged" and stopped[5] is None


def test_a_tail_resume_refuses_what_it_cannot_continue():
    h_local, psi0 = _pieces()
    alphas, betas, _, widths, _, tail = _run(h_local, None, 3, psi0=psi0)
    common = dict(h_op=h_local, converged=lambda a, b, **_: False, max_iter=2, alphas=alphas, betas=betas, tail=tail)
    with pytest.raises(ValueError, match="reort='none'"):
        block_lanczos_array(psi0=None, reort=Reort.FULL, block_widths_init=widths, **common)
    with pytest.raises(ValueError, match="block_widths_init"):
        block_lanczos_array(psi0=None, reort=Reort.NONE, build_krylov_basis=False, **common)


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="the partition leaves the last rank empty only at -n 2+")
@pytest.mark.parametrize("k", [0, 4, 5])
def test_a_resumed_recurrence_on_ranks_one_of_which_is_empty(k):
    comm = MPI.COMM_WORLD
    h_local, psi0 = _pieces(comm)
    whole = _run(h_local, comm, STEPS, psi0=psi0)
    resumed = _resumed(h_local, psi0, comm, k)
    np.testing.assert_allclose(resumed[0], whole[0], atol=1e-12)
    np.testing.assert_allclose(resumed[1], whole[1], atol=1e-12)
    assert resumed[3] == whole[3]


def test_a_resumed_recurrence_keeps_the_divergence_guards_scale():
    """The guard is replayed over the coefficients handed in: a resumed step whose blocks are 1000x the scale
    the earlier ones established is caught as a divergence, not taken as the first step of a new scale."""
    h_local, psi0 = _pieces()
    alphas, betas, _, widths, _, tail = _run(h_local, None, 3, psi0=psi0)
    resumed = _run(
        h_local * 1e3,
        None,
        STEPS - 3,
        psi0=None,
        alphas=alphas.copy(),
        betas=betas.copy(),
        tail=tail,
        block_widths_init=widths,
    )
    assert resumed[4] == "diverged", resumed[4]
