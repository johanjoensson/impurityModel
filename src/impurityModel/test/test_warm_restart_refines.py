"""A warm-started restarted Lanczos must *refine* its start block, not return it.

``_cholesky_or_deflate`` used to answer two questions with one threshold,
``evals > DEFLATE_EVAL_TOL * max(evals[-1], 1.0)``. The ``max(..., 1.0)`` made the rank test
**absolute** below ``sqrt(DEFLATE_EVAL_TOL) = EPS**(1/3) = 6.06e-6``, so a residual block
smaller than that deflated to rank 0 whatever the operator norm was. A Lanczos sweep started
from nearly-converged eigenvectors produces exactly such a block on its first step: ``beta_0``
*is* the eigenpair residual. The sweep declared an invariant subspace and both restarted
solvers handed back their input, silently, however tight ``tol`` was.

The two questions have different scales and are now asked separately (``e3ad8c9``):

* **rank deficiency** -- relative, ``evals > DEFLATE_EVAL_TOL * lam_max``. Scale invariant.
* **breakdown** -- absolute, hence needing a reference: ``sqrt(lam_max) <= BREAKDOWN_TOL * scale``.
  The sweeps pass ``~||H||`` because a residual block is zero relative to ``||H||``, not to 1.

So the floor became *relative*: a warm start is refined until its residual reaches
``BREAKDOWN_TOL * ||H||``, i.e. a fixed relative accuracy, rather than a fixed absolute one.

These tests pin that for **both** restarted solvers on **both** paths, since ``_trlm_core`` and
``_irlm_core`` are path-agnostic and share the sweep. The operator is scaled to ``||H|| ~ 3.8e5``
so that the three regimes separate:

    BREAKDOWN_TOL * ||H|| = 3.8e-7  <  ||R0|| = 1.2e-6  <  DEFLATE_TOL = 6.06e-6

The warm start sits between the new relative floor and the old absolute one -- refinable now,
deflated away before. Measured against ``2eeac1b`` (the commit before the deflation work), every
solver below returned its input unrefined at ``1.16e-06``; TRLM additionally *diverged* to
``5.7e+07`` when handed a start block just above the old floor.
"""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sps

try:
    from mpi4py import MPI

    _has_mpi = True
except ImportError:  # pragma: no cover - mpi4py is a hard dependency in practice
    _has_mpi = False

from impurityModel.ed.BlockLanczos import (
    _implicitly_restarted_block_lanczos_array,
    _thick_restart_block_lanczos_array,
    implicitly_restarted_block_lanczos_cy,
    thick_restart_block_lanczos_cy,
)
from impurityModel.ed.BlockLanczosArray import BREAKDOWN_TOL, Reort, block_normalize
from impurityModel.ed.TSQR import DEFLATE_TOL
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant
from impurityModel.test.test_restarted_lanczos import MockBasis

_N = 400
_N_ORB = 120  # the ManyBodyState path carries one determinant per orbital
_P = 3
_SCALE = 1e5  # sets ||H||; see the module docstring for why it has to be large
_NOISE = 1e-13  # perturbs the exact eigenvectors to land ||R0|| in the window
_TOL = 1e-12


def _designer_hermitian(n, seed=0):
    """Dense Hermitian with ``||H|| ~ 3.8e5`` and a well-separated low end."""
    rng = np.random.default_rng(seed)
    d = np.sort(rng.standard_normal(n)) * _SCALE
    u = la.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    h = (u * d) @ np.conj(u.T)
    return 0.5 * (h + np.conj(h.T)), d, u, rng


def _warm_start(u, d, rng, n):
    """The ``_P`` lowest eigenvectors, nudged so the residual is small but nonzero."""
    exact = u[:, np.argsort(d)[:_P]]
    noise = rng.standard_normal((n, _P)) + 1j * rng.standard_normal((n, _P))
    return la.qr(exact + _NOISE * noise, mode="economic")[0]


def _residuals(h, vals, vecs):
    """Per-column ``||H v - theta v||`` for a dense ``(n, k)`` block."""
    return np.linalg.norm(h @ vecs - vecs * np.asarray(vals)[None, :], axis=0)


def _start_residuals(h, q0):
    """The warm start's own eigenpair residuals, using its Rayleigh quotients."""
    thetas = np.array([np.conj(q0[:, j]) @ h @ q0[:, j] for j in range(q0.shape[1])]).real
    return _residuals(h, thetas, q0)


def _one_particle_operator(t):
    """A ``ManyBodyOperator`` whose one-particle sector is exactly the matrix ``t``.

    With a single fermion there are no anticommutation signs, so the many-body matrix in the
    ``n_orb`` single-occupancy determinants *is* ``t``. Orbital ``i`` is bit ``7 - i % 8`` of
    byte ``i // 8`` -- the MSB-first convention.
    """
    n_orb = t.shape[0]
    n_bytes = (n_orb + 7) // 8
    op = {((i, "c"), (j, "a")): complex(t[i, j]) for i in range(n_orb) for j in range(n_orb)}

    def determinant(i):
        raw = bytearray(n_bytes)
        raw[i // 8] |= 1 << (7 - i % 8)
        return SlaterDeterminant.from_bytes(bytes(raw))

    return ManyBodyOperator(op), [ManyBodyState({determinant(i): 1.0}) for i in range(n_orb)]


def _states_from_columns(basis_states, cols):
    """Turn the dense ``(n_orb, k)`` block ``cols`` into ``k`` ManyBodyStates."""
    block = []
    for j in range(cols.shape[1]):
        state = ManyBodyState()
        for i, b in enumerate(basis_states):
            state += b * complex(cols[i, j])
        block.append(state)
    return block


def _mbs_residual(h_op, psi, theta):
    return float(np.sqrt((h_op.apply(psi) - psi * complex(theta)).norm2()))


def _assert_window(h, q0):
    """The warm start must sit strictly between the new relative floor and the old absolute one.

    Without this the test silently stops proving anything: above ``DEFLATE_TOL`` the old code
    also refines, below ``BREAKDOWN_TOL * ||H||`` the new code correctly declares invariance.
    """
    h_norm = np.linalg.norm(h, 2)
    r0 = float(np.max(_start_residuals(h, q0)))
    assert BREAKDOWN_TOL * h_norm < r0 < DEFLATE_TOL, (
        f"warm start ||R0||={r0:.3e} is outside the window "
        f"({BREAKDOWN_TOL * h_norm:.3e}, {DEFLATE_TOL:.3e}); the test no longer bites"
    )
    return r0


# --------------------------------------------------------------------------------------
# Dense array path
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("solver", ["trlm", "irlm"])
def test_array_warm_start_is_refined_not_returned(solver):
    """Pre-fix both returned ``q0`` at ``||r|| = 1.16e-06``; TRLM at a looser start diverged."""
    h, d, u, rng = _designer_hermitian(_N)
    q0 = _warm_start(u, d, rng, _N)
    r0 = _assert_window(h, q0)

    class _Basis:
        size = _N
        comm = None

    if solver == "trlm":
        vals, vecs = _thick_restart_block_lanczos_array(
            q0.copy(), h, _Basis(), _P, 12, _TOL, 60, False, Reort.PARTIAL, None
        )
    else:
        vals, vecs = _implicitly_restarted_block_lanczos_array(
            q0.copy(), h, _Basis(), _P, 12, _TOL, 60, False, Reort.PARTIAL, None
        )

    vals = np.asarray(vals).real
    r = float(np.max(_residuals(h, vals, np.asarray(vecs))))

    # A solver that returns its input passes "r <= r0" trivially. Demand real improvement.
    assert r < r0 / 100, f"{solver} did not refine the warm start: {r:.3e} vs {r0:.3e}"
    np.testing.assert_allclose(np.sort(vals)[:_P], np.sort(d)[:_P], rtol=1e-9)


# --------------------------------------------------------------------------------------
# ManyBodyState path -- the same _trlm_core / _irlm_core, driven with list[ManyBodyState]
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("solver", ["trlm", "irlm"])
def test_manybodystate_warm_start_is_refined_not_returned(solver):
    t, d, u, rng = _designer_hermitian(_N_ORB, seed=1)
    q0 = _warm_start(u, d, rng, _N_ORB)
    r0 = _assert_window(t, q0)

    h_op, basis_states = _one_particle_operator(t)
    psi0 = _states_from_columns(basis_states, q0)
    kernel = thick_restart_block_lanczos_cy if solver == "trlm" else implicitly_restarted_block_lanczos_cy

    vals, vecs = kernel(psi0, h_op, MockBasis(_N_ORB), _P, 12, _TOL, 60, False, 0.0, 0, "partial", None)

    vals = np.asarray(vals).real
    r = max(_mbs_residual(h_op, psi, theta) for psi, theta in zip(vecs, vals))

    assert r < r0 / 100, f"{solver} did not refine the warm start: {r:.3e} vs {r0:.3e}"
    np.testing.assert_allclose(np.sort(vals)[:_P], np.sort(d)[:_P], rtol=1e-9)


# --------------------------------------------------------------------------------------
# The deflation decision is read off an Allreduced Gram: every rank must take the same branch
# --------------------------------------------------------------------------------------


@pytest.mark.mpi
@pytest.mark.skipif(not _has_mpi, reason="mpi4py not available")
@pytest.mark.parametrize("solver", ["trlm", "irlm"])
def test_warm_start_refinement_is_collective(solver):
    """``_cholesky_or_deflate`` runs on the Allreduced Gram, so ``active_k`` agrees on every rank.

    If the deflation decision ever became rank-local, one rank would break out of the sweep while
    the others kept calling ``block_apply``'s collectives, and this test would **hang** rather than
    fail -- which is the point.

    Note the assertion is on the residual, not the eigenvalues: a warm start's Rayleigh quotients
    are already accurate to ``||R0||^2 / gap ~ 2e-17``, so a solver that returns its input
    unrefined reproduces the eigenvalues to full precision. Only the residual sees it.
    """
    comm = MPI.COMM_WORLD
    n = 200  # every rank builds the same H from the same seed; keep it cheap

    h, d, u, rng = _designer_hermitian(n, seed=7)
    q0_full = _warm_start(u, d, rng, n)
    r0 = _assert_window(h, q0_full)

    counts = [n // comm.size + (1 if r < n % comm.size else 0) for r in range(comm.size)]
    c0 = sum(counts[: comm.rank])
    c1 = c0 + counts[comm.rank]
    # Column-sliced, matching the block_apply convention: h_local @ psi_local is a partial sum
    # of the full H @ psi, completed by an Allreduce inside the kernel.
    h_local = sps.csr_matrix(h[:, c0:c1])
    q0_local, _ = block_normalize(np.ascontiguousarray(q0_full[c0:c1, :]), mpi=True, comm=comm)

    class _Basis:
        def __init__(self, comm):
            self.comm = comm
            self.size = n

    entry = _thick_restart_block_lanczos_array if solver == "trlm" else _implicitly_restarted_block_lanczos_array
    vals, vecs = entry(q0_local, h_local, _Basis(comm), _P, 12, _TOL, 60, False, Reort.PARTIAL, comm)

    vals = np.asarray(vals).real
    k = len(vals)

    # Rebuild the global Ritz block and H @ V from the local slices (n = 200, so this is cheap).
    v_full = np.zeros((n, k), dtype=complex)
    v_full[c0:c1, :] = np.asarray(vecs)
    v_full = comm.allreduce(v_full, op=MPI.SUM)
    hv = comm.allreduce(h_local @ np.asarray(vecs), op=MPI.SUM)
    r = float(np.max(np.linalg.norm(hv - v_full * vals[None, :], axis=0)))

    assert r < r0 / 100, f"{solver} did not refine the warm start: {r:.3e} vs {r0:.3e}"
    np.testing.assert_allclose(np.sort(vals)[:_P], np.sort(d)[:_P], rtol=1e-9)

    # Every rank agrees on the answer (the Ritz values are broadcast from rank 0).
    spread = comm.allreduce(float(np.max(np.abs(vals - comm.bcast(vals, root=0)))), op=MPI.MAX)
    assert spread == 0.0
