"""TRLM stops on stagnation, and converges the wanted manifold rather than the padding.

Both behaviours come from one measured failure. On an SrMnO3 ground state
(``impmod_tests/SMO/cubic``), a single TRLM solve burned all 100 restarts moving its max
residual from 6.9e-5 to 6.4e-5 -- three orders of magnitude above ``tol`` -- while the
eigenvalue sat bit-identical for the last 61 of them and the Krylov basis stayed perfectly
orthogonal (``||Q^H Q - I|| ~ 1e-14``, full rank). That one solve was ~10% of every Lanczos
restart in the run, and six more solves showed the same shape.

Two things were wrong:

* **The gate covered states nobody wanted.** ``get_eigenvectors`` asks for
  ``num_wanted + _EIGENSTATE_PAD`` states so that one landing outside the thermal cut
  certifies the kept manifold is whole. Those extra states are bought for their *energies*;
  ``_energy_cut_indices`` keeps a prefix of the sorted spectrum, so the pad is exactly the
  tail that gets discarded. Holding the whole solve to ``tol`` on the tail's residuals let one
  slow padding state stall a solve whose wanted states had long since converged.
  ``num_converge`` narrows the max-over-wanted test to the states the caller will keep.

* **Nothing noticed it had stopped making progress.** The residual is not monotone across
  thick restarts (measured: it swung between 3.9e-5 and 7.7e-4 on that solve), so progress is
  judged on the best value seen so far. Calibrated on the same 69 real solves: every one that
  reached ``tol`` improved its best-so-far by 34x-30000x after restart 10, while the stalled
  one managed 17.6x over 89 restarts. ``STAGNATION_FACTOR = 10`` sits inside that gap.

Stopping early is not worse than the status quo: exhausting ``max_restarts`` already returns
under-converged Ritz pairs. It reaches the same outcome sooner, and ``_TRLM_EXIT`` says which
happened.
"""

import numpy as np
import pytest

from impurityModel.ed import trlm as trlm_mod
from impurityModel.ed.BlockLanczosArray import Reort
from impurityModel.ed.cipsi_solver import SOLVERS
from impurityModel.ed.trlm import _thick_restart_block_lanczos_array, _TRLM_EXIT
from impurityModel.test.support.lanczos_fixtures import MockBasis


def _split_spectrum(n=64, seed=5):
    """Three well-separated low states, then a tight cluster the solver struggles with.

    The low three converge almost immediately; the cluster members are separated by 1e-7 and
    need many more restarts. That gap is the whole point: it makes "how many states must
    converge" decide the restart count.
    """
    rng = np.random.default_rng(seed)
    cluster = 0.0 + 1e-7 * np.arange(n - 3)
    evals = np.concatenate([[-5.0, -4.0, -3.0], cluster])
    u = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    h = (u * evals) @ np.conj(u.T)
    return 0.5 * (h + np.conj(h.T)), np.sort(evals)


def _run(h, num_wanted, tol, num_converge=None, max_restarts=100, seed=0):
    rng = np.random.default_rng(seed)
    n = h.shape[0]
    psi0 = rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2))
    return _thick_restart_block_lanczos_array(
        psi0,
        h,
        MockBasis(n),
        num_wanted,
        max_subspace_blocks=8,
        tol=tol,
        max_restarts=max_restarts,
        verbose=False,
        reort_mode=Reort.FULL,
        comm=None,
        num_converge=num_converge,
    )


def test_num_converge_still_returns_every_requested_state():
    """Narrowing the *gate* must not narrow the *output*.

    ``get_eigenvectors`` reads a short return as ``exhausted`` and answers it with a full
    cold-start re-solve, so a solver that returned fewer states would cost more than it saves.
    """
    h, _ = _split_spectrum()
    eigvals, _ = _run(h, num_wanted=12, tol=1e-10, num_converge=3)
    assert len(eigvals) == 12


def test_num_converge_keeps_the_states_it_gates_on_accurate():
    """The states the caller keeps are as converged as they were before."""
    h, exact = _split_spectrum()
    eigvals, _ = _run(h, num_wanted=12, tol=1e-10, num_converge=3)
    assert np.allclose(np.sort(eigvals.real)[:3], exact[:3], atol=1e-8)


def test_num_converge_reaches_the_gate_sooner_than_the_full_request():
    """Gating on 3 states costs no more restarts than gating on all 12.

    The cluster states need many restarts and the low three do not, so the narrowed gate is
    reached at least as early. Compared as restart counts via ``_TRLM_EXIT`` plus the restart
    cap rather than by wall time, which would be a flaky thing to assert.
    """
    h, _ = _split_spectrum()

    # Enough restarts for the narrow gate, not enough for the full one.
    _run(h, num_wanted=12, tol=1e-10, num_converge=3, max_restarts=6)
    narrow_exit = _TRLM_EXIT[0]
    _run(h, num_wanted=12, tol=1e-10, num_converge=None, max_restarts=6)
    full_exit = _TRLM_EXIT[0]

    assert narrow_exit == "restart_loop_end_converged"
    assert full_exit != "restart_loop_end_converged"


def test_stagnation_stops_before_the_restart_cap(monkeypatch):
    """An unreachable tolerance is abandoned instead of burning every restart.

    ``tol`` below the achievable roundoff floor is the deterministic way to produce the real
    failure: the residual plateaus and no number of restarts gets under the bar. Patience is
    shortened so the test does not have to run 40+ restarts to observe it.
    """
    monkeypatch.setattr(trlm_mod, "STAGNATION_PATIENCE", 3)
    monkeypatch.setattr(trlm_mod, "STAGNATION_FACTOR", 10.0)

    h, _ = _split_spectrum()
    _run(h, num_wanted=12, tol=1e-30, num_converge=None, max_restarts=100)

    assert _TRLM_EXIT[0] == "restart_loop_end_stagnated"


def test_stagnation_does_not_fire_on_a_solve_that_converges(monkeypatch):
    """The detector must not cut off a solve still making real progress."""
    monkeypatch.setattr(trlm_mod, "STAGNATION_PATIENCE", 3)
    monkeypatch.setattr(trlm_mod, "STAGNATION_FACTOR", 10.0)

    h, exact = _split_spectrum()
    eigvals, _ = _run(h, num_wanted=3, tol=1e-10, num_converge=3, max_restarts=100)

    assert _TRLM_EXIT[0] == "restart_loop_end_converged"
    assert np.allclose(np.sort(eigvals.real)[:3], exact[:3], atol=1e-8)


@pytest.mark.parametrize("name", sorted(SOLVERS))
def test_every_registered_solver_accepts_num_converge(name):
    """``get_eigenvectors`` passes ``num_converge`` to whichever solver is selected.

    IRLM accepts and ignores it (its stopping rule is per-pair locking to ``num_wanted``, and
    lowering that target would shorten the returned block), but it must not raise.
    """
    import inspect

    assert "num_converge" in inspect.signature(SOLVERS[name]).parameters
