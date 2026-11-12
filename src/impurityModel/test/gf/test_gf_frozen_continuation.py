"""A frozen GF recurrence carries on as the ``P H P`` array-kernel recurrence from its last two blocks.

Up to the freeze every block's ``H`` image was admitted whole, and the freezing step was projected onto the
retained set ``P``, so the recurrence so far is already the block Lanczos of ``P H P``: at reort 'none' the array
kernel continues it (``gf_recurrence._continue_on_array_kernel``) instead of recomputing every pre-freeze block
from the seeds. The oracles: the dense ``P H P`` resolvent on the retained set, the matrix-free unit that runs
through the freeze, and a spy on the array kernel showing no block is computed twice.
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import BlockLanczosCore, gf_solvers
from impurityModel.test.gf.test_gf_truncation import _assert_php_oracle, _run_capped
from impurityModel.test.support.gf_switches import set_gf_switches


@pytest.fixture(autouse=True)
def _shipped_defaults(monkeypatch):
    for knob in ("GF_FROZEN_CSR", "APPLY_ROW_CHUNKS", "GF_SUPPORT_FREEZE"):
        monkeypatch.delenv(knob, raising=False)


@pytest.fixture(autouse=True)
def _retained_mode(monkeypatch):
    """The continuation is the ``"retained"`` mode's frozen phase."""
    set_gf_switches(monkeypatch, matrix_mode="retained")


def _spy(monkeypatch):
    """Every run of the array kernel: where it started (the blocks already computed) and where it ended."""
    calls = []
    real_kernel = BlockLanczosCore.block_lanczos

    def recording_kernel(*args, **kwargs):
        out = real_kernel(*args, **kwargs)
        calls.append({"restart": kwargs.get("tail") is None, "start": len(kwargs["alphas_init"]), "end": len(out[0])})
        return out

    monkeypatch.setattr(BlockLanczosCore, "block_lanczos", recording_kernel)
    return calls


@pytest.mark.parametrize("chunks", ["1", "4"])
@pytest.mark.parametrize("cap", [6, 12, 17])
def test_a_frozen_unit_continues_from_where_it_stopped(cap, chunks, monkeypatch):
    monkeypatch.setenv("APPLY_ROW_CHUNKS", chunks)
    calls = _spy(monkeypatch)
    g, info = _run_capped(cap)
    assert info["cap_hit"] and info["engine"] == "retained"
    assert calls and not any(c["restart"] for c in calls), calls
    assert calls[0]["start"] > 0, "premise: the growth loop built blocks before the freeze"
    _assert_php_oracle(g, info)
    set_gf_switches(monkeypatch, matrix_mode="none")
    g_sparse, _ = _run_capped(cap)
    np.testing.assert_allclose(g, g_sparse, atol=1e-10)


def test_a_spent_budget_continues_from_its_own_tail(monkeypatch):
    """With a one-block first budget the frozen recurrence runs in several attempts; each starts where the last
    stopped, so the blocks they add sum to the blocks returned beyond the growth prefix."""
    monkeypatch.setattr(gf_solvers, "_csr_continue_budget", lambda start: 1)
    calls = _spy(monkeypatch)
    g, info = _run_capped(12)
    assert len(calls) > 1, calls
    assert all(b["start"] == a["end"] for a, b in itertools.pairwise(calls)), calls
    _assert_php_oracle(g, info)


def test_a_reorthogonalized_unit_stays_on_the_growth_loop(monkeypatch):
    """The reorthogonalization store lives in the growth loop, so a unit that keeps it never hands off."""
    calls = _spy(monkeypatch)
    g, info = _run_capped(12, reort=True)
    assert info["cap_hit"] and not calls
    _assert_php_oracle(g, info)


@pytest.mark.mpi
@pytest.mark.parametrize("chunks", ["1", "4"])
def test_the_continuation_on_every_rank(chunks, monkeypatch):
    """Multi-rank (an empty rank at -n 3): the continuation equals the matrix-free unit and the P H P oracle."""
    comm = MPI.COMM_WORLD
    monkeypatch.setenv("APPLY_ROW_CHUNKS", chunks)
    calls = _spy(monkeypatch)
    g, info = _run_capped(6, comm=comm)
    assert info["engine"] == "retained" and not any(c["restart"] for c in calls)
    assert len(set(comm.allgather(tuple((c["start"], c["end"]) for c in calls)))) == 1
    _assert_php_oracle(g, info, comm)
    set_gf_switches(monkeypatch, matrix_mode="none")
    g_sparse, _ = _run_capped(6, comm=comm)
    np.testing.assert_allclose(g, g_sparse, atol=1e-9)
