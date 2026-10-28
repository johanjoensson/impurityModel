"""The cut rules of importance-pruned growth on the discovered-key matrix: ``eta`` (amplitude) and ``mass`` (a
fraction of the step's frontier weight), the per-step admitted counts (``depth_histogram``) and the kept first
shell (``shell_mask``). Exactness on what was kept is ``test_gf_lanczos_pruning``'s; this file pins what is new
with the mass rule and the records.
"""

import numpy as np
import pytest

from impurityModel.ed.gf_solvers import block_Green_sparse
from impurityModel.ed.greens_function import calc_G
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.test.gf.test_gf_truncation import _redistribute_as_width1
from impurityModel.test.support.comms import COMMS
from impurityModel.test.support.gf_oracles import DELTA, OMEGA, _dense_G_on, _seed_basis, _seeds, _siam_6


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    monkeypatch.setenv("APPLY_ROW_CHUNKS", "1")
    for name in ("GF_LANCZOS_ADMIT_TOL", "GF_ADMIT_FIRST_SHELL_TOL", "GF_PRUNE_RULE"):
        monkeypatch.delenv(name, raising=False)


def _run(monkeypatch, tol, rule="eta", shell_tol=None, cap=np.inf, comm=None):
    monkeypatch.setenv("GF_LANCZOS_ADMIT_TOL", repr(tol))
    monkeypatch.setenv("GF_PRUNE_RULE", rule)
    if shell_tol is not None:
        monkeypatch.setenv("GF_ADMIT_FIRST_SHELL_TOL", repr(shell_tol))
    basis = _seed_basis(cap=cap, comm=comm)
    full = _seeds()
    seeds = _redistribute_as_width1(basis, full if comm is None or comm.rank == 0 else None, len(full))
    info = {}
    alphas, betas, r = block_Green_sparse(_siam_6(), seeds, basis, DELTA, verbose=False, cap_info=info)
    return calc_G(alphas, betas, r, OMEGA, 0.0, DELTA), info


def _gather(keys, comm):
    if comm is not None:
        keys = [k for part in comm.allgather(keys) for k in part]
    return sorted(set(keys))


def _retained(info, comm=None):
    return _gather(info["proxy"].retained_keys(), comm)


def _shell(comm=None):
    seeds = _seeds()
    return set(_siam_6().apply_block(ManyBodyState.from_states(seeds), 0.0).keys()), {
        k for s in seeds for k in s.keys()
    }


@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("tau", [1e-4, 1e-2, 0.1])
def test_the_mass_rule_is_the_exact_resolvent_of_PHP_on_what_it_kept(monkeypatch, comm, tau):
    G, info = _run(monkeypatch, tau, "mass", comm=comm)
    kept = _retained(info, comm)
    np.testing.assert_allclose(G, _dense_G_on(kept, OMEGA + 1j * DELTA), atol=1e-9)


@pytest.mark.parametrize("comm", COMMS)
def test_the_mass_rule_keeps_the_first_shell_whole_by_default_and_prunes_the_rest_as_tau_grows(monkeypatch, comm):
    shell, seeds = _shell()
    sizes = []
    for tau in (0.0, 1e-3, 1e-1, 1.0):
        _G, info = _run(monkeypatch, tau if tau > 0 else 1e-300, "mass", comm=comm)
        kept = set(_retained(info, comm))
        assert (shell | seeds) <= kept
        sizes.append(len(kept))
    assert sizes == sorted(sizes, reverse=True) and sizes[0] > sizes[-1]
    assert sizes[-1] == len(shell | seeds), "tau = 1 leaves out all the later weight"


@pytest.mark.parametrize("comm", COMMS)
def test_the_rules_admit_at_different_depths(monkeypatch, comm):
    """With no ban every rule ends on the closure on this small fixture; what differs is the depth each row enters
    at, which is what the histogram records."""
    hist = {}
    for rule, tol in (("eta", 0.3), ("mass", 0.1), ("mass", 0.3)):
        _G, info = _run(monkeypatch, tol, rule, comm=comm)
        counts = info["proxy"].depth_histogram()
        hist[(rule, tol)] = tuple(counts[: max(i for i, c in enumerate(counts) if c) + 1])
    assert len(set(hist.values())) == len(hist), hist
    assert (5, 6, 3, 1) not in hist.values(), "premise: each one defers rows the unpruned growth admits at once"


@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("rule,tol", [("eta", 0.2), ("mass", 1e-2)])
def test_the_depth_histogram_counts_what_each_step_admitted(monkeypatch, comm, rule, tol):
    _G, info = _run(monkeypatch, tol, rule, comm=comm)
    support = info["proxy"]
    shell, seeds = _shell()
    counts = support.depth_histogram()
    assert counts[0] == len(shell - seeds), "the first step admits the first shell"
    assert len(seeds) + sum(counts) == info["retained_size"]
    assert all(c >= 0 for c in counts) and counts[-1] >= 0


@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("rule", ["eta", "mass"])
def test_the_shell_mask_is_the_first_shell_the_guard_kept(monkeypatch, comm, rule):
    shell, seeds = _shell()
    _G, info = _run(monkeypatch, 1e6 if rule == "eta" else 0.5, rule, comm=comm)
    whole = _gather(list(info["proxy"].shell_mask.keys()), comm)
    assert set(whole) == shell - seeds
    _G, info = _run(
        monkeypatch, 1e6 if rule == "eta" else 0.5, rule, shell_tol=0.6 if rule == "eta" else 0.3, comm=comm
    )
    relaxed = set(_gather(list(info["proxy"].shell_mask.keys()), comm))
    assert relaxed < shell - seeds, "a relaxed first shell is pruned, and the mask says so"


def test_an_unknown_rule_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="GF_PRUNE_RULE"):
        _run(monkeypatch, 0.1, "bogus")


@pytest.mark.parametrize("comm", COMMS)
def test_the_k_over_p_ratio_is_reported_under_pruning_only(monkeypatch, comm):
    from impurityModel.test.support.gf_switches import set_gf_switches

    set_gf_switches(monkeypatch, matrix_mode="discovered")
    _G, pruned = _run(monkeypatch, 0.3, "eta", comm=comm)
    assert pruned["k_over_p"] >= 1.0
    monkeypatch.delenv("GF_LANCZOS_ADMIT_TOL")
    basis = _seed_basis(cap=np.inf, comm=comm)
    full = _seeds()
    seeds = _redistribute_as_width1(basis, full if comm is None or comm.rank == 0 else None, len(full))
    info = {}
    block_Green_sparse(_siam_6(), seeds, basis, DELTA, verbose=False, cap_info=info)
    assert np.isnan(info["k_over_p"])
