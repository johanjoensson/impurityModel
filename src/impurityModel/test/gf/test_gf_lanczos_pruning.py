"""Importance-pruned block-Lanczos growth (``GF_LANCZOS_ADMIT_TOL``): the comparator for ``outer``.

``_PrunedBasisProxy`` admits a step's new determinants only above an amplitude threshold and bans
the rest for good. The ban is what makes the recurrence exact, so the oracle is the same one the
freeze-growth cap satisfies (``test_gf_truncation``): the returned continued fraction must equal
the dense resolvent of ``H`` projected onto the retained set -- for every threshold and every
reorthogonalization mode, not just the full one.

Also pinned: the decision is taken once, on the step's summed output, so row chunking and a
nonzero ``slaterWeightMin`` are allowed (the rows the cut drops are banned too), and the seeds' first
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
    monkeypatch.setenv("GF_APPLY_ROW_CHUNKS", "1")
    monkeypatch.delenv("GF_LANCZOS_ADMIT_TOL", raising=False)
    monkeypatch.delenv("GF_ADMIT_FIRST_SHELL_TOL", raising=False)


def _run(monkeypatch, eta, cap=np.inf, reort=None, comm=None, slater=0.0):
    monkeypatch.setenv("GF_LANCZOS_ADMIT_TOL", repr(eta))
    basis = _seed_basis(cap=cap, comm=comm)
    full = _seeds()
    seeds = _redistribute_as_width1(basis, full if comm is None or comm.rank == 0 else None, len(full))
    info = {}
    alphas, betas, r = block_Green_sparse(
        _siam_6(), seeds, basis, DELTA, reort=reort, verbose=False, cap_info=info, slaterWeightMin=slater
    )
    z = OMEGA + 1j * DELTA
    return calc_G(alphas, betas, r, OMEGA, 0.0, DELTA), info, z


def _retained(info, comm=None):
    keys = info["proxy"].retained_keys()
    if comm is not None:
        keys = [k for part in comm.allgather(keys) for k in part]
    return sorted(set(keys))


@pytest.mark.parametrize("reort", [None, "full", "partial"])
# 0.225 and 0.253 are thresholds at which a row rejected on one step clears the threshold on a later
# one -- the case the ban exists for (see the negative control below).
@pytest.mark.parametrize("eta", [0.05, 0.2, 0.225, 0.253, 0.5])
def test_the_pruned_recurrence_is_the_exact_resolvent_of_PHP_on_what_it_kept(monkeypatch, eta, reort):
    G, info, z = _run(monkeypatch, eta, reort=reort)
    retained = _retained(info)
    np.testing.assert_allclose(G, _dense_G_on(retained, z), atol=1e-9)


@pytest.mark.parametrize("reort", [None, "full"])
def test_without_the_ban_the_recurrence_is_not_exact_which_is_why_it_exists(monkeypatch, reort):
    """Negative control: forget every ban before each step and the same oracle must fail -- a row
    rejected when ``H q_k`` was formed is admitted later, so ``P_m H q_k != P_{k+1} H q_k`` and the
    recurrence is the Lanczos of no single operator. At eta=0.225 the retained set is even the
    whole closure, yet G is off by ~1e-2: the error is in the recurrence, not in what was kept."""
    from impurityModel.ed import gf_primitives

    original = gf_primitives.SupportPolicy._admit_pruned

    def forgetting(self, block, reached=None):
        ban, self._ban = self._ban, ManyBodyState.from_keys([])
        try:
            return original(self, block, reached)
        finally:
            self._ban = ban

    monkeypatch.setattr(gf_primitives.SupportPolicy, "_admit_pruned", forgetting)
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
    monkeypatch.setenv("GF_ADMIT_FIRST_SHELL_TOL", str(shell_tol))
    G, info, z = _run(monkeypatch, 1e6)
    assert len(_retained(info)) < len(_retained(strict))
    np.testing.assert_allclose(G, _dense_G_on(_retained(info), z), atol=1e-9)


def test_the_ban_mask_is_recorded(monkeypatch):
    _G, info, _z = _run(monkeypatch, 0.5)
    assert info["proxy"].ban_bytes > 0


def test_a_cap_still_binds_under_pruning(monkeypatch):
    G, info, z = _run(monkeypatch, 1e-3, cap=10)
    assert info["cap_hit"] and info["retained_size"] <= 10
    np.testing.assert_allclose(G, _dense_G_on(_retained(info), z), atol=1e-9)


def test_off_by_default_leaves_the_unpruned_path_alone(monkeypatch):
    _G, info, _z = _run(monkeypatch, 0.0)
    assert info["proxy"] is None


@pytest.mark.parametrize("reort", [None, "full"])
@pytest.mark.parametrize("eta", [0.05, 0.225, 0.5])
def test_a_row_chunked_matvec_is_the_same_exact_PHP_resolvent(monkeypatch, eta, reort):
    """The decision is taken once, on the summed output, so chunking the apply cannot move it."""
    monkeypatch.setenv("GF_APPLY_ROW_CHUNKS", "4")
    G, info, z = _run(monkeypatch, eta, reort=reort)
    np.testing.assert_allclose(G, _dense_G_on(_retained(info), z), atol=1e-9)
    G1, info1, _ = _run(monkeypatch, eta, reort=reort)
    assert _retained(info1) == _retained(info)


# swm 0.3 / 0.4 are cutoffs at which this model's matvec output has new rows below the cut: the rows the ban must
# cover without the proxy ever seeing them. (At smaller cutoffs nothing is cut, and the tests below are vacuous.)
_CUTS = [0.3, 0.4]


@pytest.mark.parametrize("swm", _CUTS)
@pytest.mark.parametrize("chunks", ["1", "4"])
def test_a_nonzero_slater_weight_min_runs_and_does_not_depend_on_the_chunking(monkeypatch, swm, chunks):
    monkeypatch.setenv("GF_APPLY_ROW_CHUNKS", "1")
    G1, info1, _ = _run(monkeypatch, 0.1, slater=swm)
    monkeypatch.setenv("GF_APPLY_ROW_CHUNKS", chunks)
    G, info, _ = _run(monkeypatch, 0.1, slater=swm)
    assert _retained(info) == _retained(info1)
    np.testing.assert_allclose(G, G1, atol=1e-12)


@pytest.mark.parametrize("swm", _CUTS)
def test_rows_the_cut_drops_are_banned_too(monkeypatch, swm):
    """Negative control: without the ban on the rows the after-sum cut drops, a row cut from ``H q_k`` can clear
    the threshold on a later step and enter ``P_m`` -- the retained set grows."""
    from impurityModel.ed import gf_primitives

    _G, info, _z = _run(monkeypatch, 0.1, slater=swm)
    banned = _retained(info)
    original = gf_primitives.SupportPolicy.finish_step
    monkeypatch.setattr(
        gf_primitives.SupportPolicy, "finish_step", lambda self, wp, reached=None: original(self, wp, None)
    )
    _G, info, _z = _run(monkeypatch, 0.1, slater=swm)
    assert len(_retained(info)) > len(banned)


def test_pruning_and_the_support_freeze_are_exclusive():
    from impurityModel.ed.gf_primitives import SupportPolicy

    with pytest.raises(ValueError, match="exclusive"):
        SupportPolicy(_seed_basis(), 10**9, eta=0.1).enable_support_freeze(16)


@pytest.mark.mpi
@pytest.mark.parametrize("eta", [0.05, 0.225, 0.5])
def test_distributed_pruned_recurrence_is_the_exact_PHP_resolvent(monkeypatch, eta):
    comm = MPI.COMM_WORLD
    G, info, z = _run(monkeypatch, eta, reort="full", comm=comm)
    np.testing.assert_allclose(G, _dense_G_on(_retained(info, comm), z), atol=1e-9)


@pytest.mark.mpi
def test_distributed_vanishing_threshold_reaches_the_serial_closure(monkeypatch):
    comm = MPI.COMM_WORLD
    _G, serial, _z = _run(monkeypatch, 1e-12)
    _G, info, _z = _run(monkeypatch, 1e-12, comm=comm)
    assert _retained(info, comm) == _retained(serial)


def test_a_tripped_memory_guard_is_checked_once_per_admission_and_freezes(monkeypatch):
    """The guard is collective and warns: the pruned proxy must ask it once per call, not once itself and once
    again through the base class (which warned twice, and could fall through to plain cap admission with no ban
    if the second answer differed)."""
    from impurityModel.ed.gf_primitives import SupportPolicy

    proxy = SupportPolicy(_seed_basis(), 10**9, eta=0.2)
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
    monkeypatch.setenv("GF_APPLY_ROW_CHUNKS", "4")
    _G, serial, _z = _run(monkeypatch, 0.1, slater=swm)
    _G, info, _z = _run(monkeypatch, 0.1, slater=swm, comm=comm)
    assert _retained(info, comm) == _retained(serial)
