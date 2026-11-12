"""Importance-pruned block-Lanczos growth (``prune_tol``): the comparator for ``outer``.

The recurrence on the discovered-key matrix admits a step's new determinants only above an amplitude threshold;
a deferred row stays in the frontier and can enter later, so the recurrence restarts from the seeds on the final
``P`` (``gf_recurrence._needs_restart``). That restart is what makes it exact, so the oracle is the same one the
freeze-growth cap satisfies (``test_gf_truncation``): the returned continued fraction must equal
the dense resolvent of ``H`` projected onto the retained set -- for every threshold and every
reorthogonalization mode, not just the full one.

Also pinned: the decision is taken once, on the step's summed output, so row chunking and a
nonzero ``slaterWeightMin`` are allowed, and the seeds' first
H-shell is never pruned.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.gf_solvers import block_Green_sparse
from impurityModel.ed.greens_function import calc_G
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.test.gf.test_gf_truncation import _redistribute_as_width1
from impurityModel.test.support.gf_oracles import (
    DELTA,
    OMEGA,
    _dense_G_on,
    _n3_sector_dets,
    _seed_basis,
    _seeds,
    _siam_6,
)

CLOSURE = 18


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    monkeypatch.setenv("APPLY_ROW_CHUNKS", "1")


def _run(monkeypatch, eta, cap=np.inf, reort=None, comm=None, slater=0.0, shell_tol=None):
    basis = _seed_basis(cap=cap, comm=comm)
    full = _seeds()
    seeds = _redistribute_as_width1(basis, full if comm is None or comm.rank == 0 else None, len(full))
    info = {}
    alphas, betas, r = block_Green_sparse(
        _siam_6(),
        seeds,
        basis,
        DELTA,
        reort=reort,
        verbose=False,
        cap_info=info,
        slaterWeightMin=slater,
        prune_tol=eta,
        first_shell_tol=shell_tol,
    )
    z = OMEGA + 1j * DELTA
    return calc_G(alphas, betas, r, OMEGA, 0.0, DELTA), info, z


def _retained(info, comm=None):
    keys = info["proxy"].retained_keys()
    if comm is not None:
        keys = [k for part in comm.allgather(keys) for k in part]
    return sorted(set(keys))


@pytest.mark.parametrize("reort", [None, True])
# 0.225 and 0.253 are thresholds at which a row rejected on one step clears the threshold on a later
# one -- the case the ban exists for (see the negative control below).
@pytest.mark.parametrize("eta", [0.05, 0.2, 0.225, 0.253, 0.5])
def test_the_pruned_recurrence_is_the_exact_resolvent_of_PHP_on_what_it_kept(monkeypatch, eta, reort):
    G, info, z = _run(monkeypatch, eta, reort=reort)
    retained = _retained(info)
    np.testing.assert_allclose(G, _dense_G_on(retained, z), atol=1e-9)


@pytest.mark.parametrize("reort", [None, True])
def test_without_the_restart_the_recurrence_is_not_exact_which_is_why_it_exists(monkeypatch, reort):
    """Negative control: a row the cut deferred at step ``k`` enters a later step, so ``P_m H q_k != P_{k+1} H q_k``
    and the first pass is the Lanczos of no single operator. At eta=0.225 the retained set is even the whole
    closure, yet G is off by ~1e-2: the error is in the recurrence, not in what was kept. The restart from the
    seeds on the final ``P`` is what makes it the resolvent of ``P H P``."""
    from impurityModel.ed import gf_recurrence

    monkeypatch.setattr(gf_recurrence, "_needs_restart", lambda support, n_seed: False)
    G, info, z = _run(monkeypatch, 0.225, reort=reort)
    assert np.max(np.abs(G - _dense_G_on(_retained(info), z))) > 1e-4


def test_a_vanishing_threshold_reaches_the_closure_and_the_exact_G(monkeypatch):
    G, info, z = _run(monkeypatch, 1e-12)
    retained = _retained(info)
    assert len(retained) == CLOSURE
    np.testing.assert_allclose(G, _dense_G_on(_n3_sector_dets(), z), atol=1e-9)


def test_a_large_threshold_really_prunes_and_changes_the_answer(monkeypatch):
    """Guards against a vacuous oracle: pruning must drop determinants and move G."""
    G, info, z = _run(monkeypatch, 0.5)
    retained = _retained(info)
    assert len(retained) < CLOSURE
    assert np.max(np.abs(G - _dense_G_on(_n3_sector_dets(), z))) > 1e-6


def test_the_seeds_and_their_first_shell_are_never_pruned(monkeypatch):
    _G, info, _z = _run(monkeypatch, 1e6)
    retained = set(_retained(info))
    seeds = _seeds()
    shell = set(_siam_6().apply_block(ManyBodyState.from_states(seeds), 0.0).keys())
    assert {k for s in seeds for k in s.keys()} <= retained and shell <= retained


@pytest.mark.parametrize("shell_tol", [0.3, 0.6])
def test_a_relaxed_first_shell_prunes_it_and_stays_exact(monkeypatch, shell_tol):
    _G, strict, _z = _run(monkeypatch, 1e6)
    G, info, z = _run(monkeypatch, 1e6, shell_tol=shell_tol)
    assert len(_retained(info)) < len(_retained(strict))
    np.testing.assert_allclose(G, _dense_G_on(_retained(info), z), atol=1e-9)


@pytest.mark.parametrize("mode", ["discovered", "retained", "none"])
@pytest.mark.parametrize("cap", [14, 16])
def test_a_unit_the_cap_froze_after_a_deferred_row_came_back_is_still_exact(monkeypatch, mode, cap):
    """At eta=0.225 a row deferred at one step enters a later one, and the cap then freezes the support: the
    first pass is the Lanczos of no single operator, in every matvec mode, until the restart from the seeds on the
    final ``P`` (a frozen retained unit has renumbered its slots, so the seed block follows its keys)."""
    from impurityModel.ed import gf_recurrence
    from impurityModel.test.support.gf_switches import set_gf_switches

    set_gf_switches(monkeypatch, matrix_mode=mode)
    G, info, z = _run(monkeypatch, 0.225, cap=cap)
    assert info["cap_hit"] and info["engine"] == mode
    np.testing.assert_allclose(G, _dense_G_on(_retained(info), z), atol=1e-9)
    monkeypatch.setattr(gf_recurrence, "_needs_restart", lambda support, n_seed: False)
    G, info, z = _run(monkeypatch, 0.225, cap=cap)
    assert np.max(np.abs(G - _dense_G_on(_retained(info), z))) > 1e-4, "premise: the restart is what makes it exact"


def test_a_cap_still_binds_under_pruning(monkeypatch):
    G, info, z = _run(monkeypatch, 1e-3, cap=10)
    assert info["cap_hit"] and info["retained_size"] <= 10
    np.testing.assert_allclose(G, _dense_G_on(_retained(info), z), atol=1e-9)


def test_off_by_default_leaves_the_unpruned_path_alone(monkeypatch):
    _G, info, _z = _run(monkeypatch, 0.0)
    assert not info["proxy"].prunes


@pytest.mark.parametrize("reort", [None, True])
@pytest.mark.parametrize("eta", [0.05, 0.225, 0.5])
def test_a_row_chunked_matvec_is_the_same_exact_PHP_resolvent(monkeypatch, eta, reort):
    """The decision is taken once, on the summed output, so chunking the apply cannot move it."""
    monkeypatch.setenv("APPLY_ROW_CHUNKS", "4")
    G, info, z = _run(monkeypatch, eta, reort=reort)
    np.testing.assert_allclose(G, _dense_G_on(_retained(info), z), atol=1e-9)
    _G1, info1, _ = _run(monkeypatch, eta, reort=reort)
    assert _retained(info1) == _retained(info)


# swm 0.3 / 0.4 are cutoffs at which this model's matvec output has new rows below the cut: the rows the ban must
# cover without the proxy ever seeing them. (At smaller cutoffs nothing is cut, and the tests below are vacuous.)
_CUTS = [0.3, 0.4]


@pytest.mark.parametrize("swm", _CUTS)
@pytest.mark.parametrize("chunks", ["1", "4"])
def test_a_nonzero_slater_weight_min_runs_and_does_not_depend_on_the_chunking(monkeypatch, swm, chunks):
    monkeypatch.setenv("APPLY_ROW_CHUNKS", "1")
    G1, info1, _ = _run(monkeypatch, 0.1, slater=swm)
    monkeypatch.setenv("APPLY_ROW_CHUNKS", chunks)
    G, info, _ = _run(monkeypatch, 0.1, slater=swm)
    assert _retained(info) == _retained(info1)
    np.testing.assert_allclose(G, G1, atol=1e-12)


@pytest.mark.mpi
@pytest.mark.parametrize("eta", [0.05, 0.225, 0.5])
def test_distributed_pruned_recurrence_is_the_exact_PHP_resolvent(monkeypatch, eta):
    comm = MPI.COMM_WORLD
    G, info, z = _run(monkeypatch, eta, reort=True, comm=comm)
    np.testing.assert_allclose(G, _dense_G_on(_retained(info, comm), z), atol=1e-9)


@pytest.mark.mpi
def test_distributed_vanishing_threshold_reaches_the_serial_closure(monkeypatch):
    comm = MPI.COMM_WORLD
    _G, serial, _z = _run(monkeypatch, 1e-12)
    _G, info, _z = _run(monkeypatch, 1e-12, comm=comm)
    assert _retained(info, comm) == _retained(serial)


def test_a_tripped_memory_guard_is_checked_once_per_admission_and_freezes(monkeypatch):
    """The guard is collective and warns: the proxy must ask it once per admission, not twice (which warned
    twice, and could fall through to plain cap admission if the second answer differed)."""
    from impurityModel.ed.gf_primitives import SupportPolicy

    proxy = SupportPolicy(_seed_basis(), 10**9)
    calls = []

    def tripped(self=proxy):
        calls.append(1)
        return True

    monkeypatch.setattr(proxy, "_over_memory_budget", tripped)
    block = ManyBodyState.from_states(_seeds())
    out = proxy._admit(block)
    assert len(calls) == 1
    assert proxy.frozen and proxy.memory_frozen
    assert len(out) == len(proxy.retained_mask)


@pytest.mark.mpi
@pytest.mark.parametrize("swm", _CUTS)
def test_the_pruned_retained_set_does_not_depend_on_the_rank_count(monkeypatch, swm):
    comm = MPI.COMM_WORLD
    monkeypatch.setenv("APPLY_ROW_CHUNKS", "4")
    _G, serial, _z = _run(monkeypatch, 0.1, slater=swm)
    _G, info, _z = _run(monkeypatch, 0.1, slater=swm, comm=comm)
    assert _retained(info, comm) == _retained(serial)


def _run_growth(monkeypatch, eta, growth_tol, reort=None, comm=None):
    """Like :func:`_run`, with the monitor on explicit meshes sampled every block (the default samples every
    ``_GF_CHECK_EVERY`` blocks, more than this 18-determinant model has) and ``growth_tol`` for the growth pass."""
    from impurityModel.ed import gf_convergence
    from impurityModel.ed.gf_convergence import EvalMeshes

    monkeypatch.setattr(gf_convergence, "_GF_CHECK_EVERY", 1)
    basis = _seed_basis(comm=comm)
    full = _seeds()
    seeds = _redistribute_as_width1(basis, full if comm is None or comm.rank == 0 else None, len(full))
    z = OMEGA + 1j * DELTA
    info = {}
    alphas, betas, r = block_Green_sparse(
        _siam_6(),
        seeds,
        basis,
        DELTA,
        reort=reort,
        verbose=False,
        cap_info=info,
        eval_meshes=EvalMeshes([z], [1e-9]),
        growth_tol=growth_tol,
        prune_tol=eta,
    )
    return calc_G(alphas, betas, r, OMEGA, 0.0, DELTA), info, z


@pytest.mark.parametrize("reort", [None, True])
@pytest.mark.parametrize("growth_tol", [0.5, 0.1])
def test_a_looser_growth_tolerance_ends_the_growth_sooner_and_G_is_still_the_exact_PHP_resolvent(
    monkeypatch, growth_tol, reort
):
    _G, full, _z = _run_growth(monkeypatch, 0.05, None, reort=reort)
    G, info, z = _run_growth(monkeypatch, 0.05, growth_tol, reort=reort)
    assert info["passes"]["growth_blocks"] < full["passes"]["growth_blocks"]
    assert info["passes"]["restart_blocks"] > 0
    np.testing.assert_allclose(G, _dense_G_on(_retained(info), z), atol=1e-9)


def test_the_growth_tolerance_is_off_by_default_and_for_an_unpruned_support(monkeypatch):
    _G, off, _z = _run_growth(monkeypatch, 0.05, None)
    _G, unpruned, _z = _run_growth(monkeypatch, 0.0, 0.5)
    assert off["passes"]["growth_blocks"] == 10
    assert unpruned["passes"]["restart_blocks"] == 0, "no pruning: the first pass is on the final P already"


@pytest.mark.parametrize("reort", [None, True])
def test_an_early_stop_that_admitted_nothing_still_restarts(monkeypatch, reort):
    """At growth_tol 0.1 the growth stops after 6 blocks. Whatever it kept, its coefficients are not converged:
    without the restart G is off, with it G is the resolvent of P H P (the first-pass retained set is the same)."""
    from impurityModel.ed import gf_recurrence

    monkeypatch.setattr(gf_recurrence, "_needs_restart", lambda support, n_seed: False)
    G, info, z = _run_growth(monkeypatch, 0.05, 0.1, reort=reort)
    assert info["passes"]["restart_blocks"] > 0
    np.testing.assert_allclose(G, _dense_G_on(_retained(info), z), atol=1e-9)
    monkeypatch.setattr(gf_recurrence, "_restart_after_early_stop", lambda stopped_early: False)
    G, info, z = _run_growth(monkeypatch, 0.05, 0.1, reort=reort)
    assert info["passes"]["restart_blocks"] == 0
    assert np.max(np.abs(G - _dense_G_on(_retained(info), z))) > 1e-6, "premise: the early stop alone is inexact"


@pytest.mark.mpi
@pytest.mark.parametrize("growth_tol", [0.5, 0.1])
def test_distributed_early_growth_stop_is_the_exact_PHP_resolvent(monkeypatch, growth_tol):
    comm = MPI.COMM_WORLD
    G, info, z = _run_growth(monkeypatch, 0.05, growth_tol, reort=True, comm=comm)
    assert comm.allgather(info["passes"]["growth_blocks"]) == [info["passes"]["growth_blocks"]] * comm.size
    np.testing.assert_allclose(G, _dense_G_on(_retained(info, comm), z), atol=1e-9)
