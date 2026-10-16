"""A frozen sparse GF recurrence carries on as the ``P H P`` CSR recurrence from its last two blocks.

Up to the freeze every block's ``H`` image was admitted whole, and the freezing step was projected onto the
retained set ``P``, so the sparse recurrence so far is already the block Lanczos of ``P H P``: at reort 'none'
the CSR continues it (:func:`gf_solvers._frozen_csr_continue`) instead of recomputing every pre-freeze block
from the seeds. The oracles: the dense ``P H P`` resolvent on the retained set, the sparse kernel run through
the freeze, and a spy on the array kernel showing no block is computed twice.
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import gf_solvers
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.test.gf.test_gf_truncation import _assert_php_oracle, _det, _excited_basis, _run_capped


@pytest.fixture(autouse=True)
def _shipped_defaults(monkeypatch):
    for knob in ("GF_FROZEN_CSR", "GF_APPLY_ROW_CHUNKS", "GF_LANCZOS_ADMIT_TOL", "GF_SUPPORT_FREEZE"):
        monkeypatch.delenv(knob, raising=False)


def _spy(monkeypatch):
    calls = []
    real_kernel = gf_solvers.block_lanczos_array

    def recording_kernel(*args, **kwargs):
        out = real_kernel(*args, **kwargs)
        start = len(kwargs["alphas"]) if kwargs.get("tail") is not None else 0
        calls.append({"restart": kwargs.get("psi0") is not None, "start": start, "end": len(out[0])})
        return out

    monkeypatch.setattr(gf_solvers, "block_lanczos_array", recording_kernel)
    return calls


@pytest.mark.parametrize("chunks", ["1", "4"])
@pytest.mark.parametrize("cap", [6, 12, 17])
def test_a_frozen_unit_continues_from_where_it_stopped(cap, chunks, monkeypatch):
    monkeypatch.setenv("GF_APPLY_ROW_CHUNKS", chunks)
    calls = _spy(monkeypatch)
    g, info = _run_capped(cap)
    assert info["cap_hit"] and info["csr_fallback"]
    assert calls and not any(c["restart"] for c in calls), calls
    assert calls[0]["start"] > 0, "premise: the sparse kernel built blocks before the freeze"
    _assert_php_oracle(g, info)
    monkeypatch.setenv("GF_FROZEN_CSR", "0")
    g_sparse, _ = _run_capped(cap)
    np.testing.assert_allclose(g, g_sparse, atol=1e-10)


def test_a_spent_budget_continues_from_its_own_tail(monkeypatch):
    """With a one-block first budget the CSR recurrence runs in several attempts; each starts where the last
    stopped, so the blocks they add sum to the blocks returned beyond the sparse prefix."""
    monkeypatch.setattr(gf_solvers, "_csr_continue_budget", lambda start: 1)
    calls = _spy(monkeypatch)
    g, info = _run_capped(12)
    assert len(calls) > 1, calls
    assert all(b["start"] == a["end"] for a, b in itertools.pairwise(calls)), calls
    _assert_php_oracle(g, info)


@pytest.mark.parametrize("reort", ["full", "partial"])
def test_a_reorthogonalized_unit_still_restarts_from_the_seeds(reort, monkeypatch):
    calls = _spy(monkeypatch)
    g, info = _run_capped(12, reort=reort)
    assert info["csr_fallback"] and calls and calls[0]["restart"]
    _assert_php_oracle(g, info)


def test_a_tail_row_outside_p_is_refused():
    basis = _excited_basis(np.inf)
    outside = _det([3, 4, 5])
    assert outside not in set(basis.local_basis), "premise: a determinant outside the basis"
    mask = ManyBodyState.from_keys(sorted(basis.local_basis))
    tail = [ManyBodyState({basis.local_basis[0]: 1.0}), ManyBodyState({outside: 1.0})]
    with pytest.raises(RuntimeError, match="rows outside P True"):
        gf_solvers._sparse_tail_to_local_dense(basis, mask, tail, [1])


@pytest.mark.mpi
@pytest.mark.parametrize("chunks", ["1", "4"])
def test_the_continuation_on_every_rank(chunks, monkeypatch):
    """Multi-rank (an empty rank at -n 3): the continuation equals the sparse kernel and the P H P oracle."""
    comm = MPI.COMM_WORLD
    monkeypatch.setenv("GF_APPLY_ROW_CHUNKS", chunks)
    calls = _spy(monkeypatch)
    g, info = _run_capped(6, comm=comm)
    assert info["csr_fallback"] and not any(c["restart"] for c in calls)
    assert len(set(comm.allgather(tuple((c["start"], c["end"]) for c in calls)))) == 1
    _assert_php_oracle(g, info, comm)
    monkeypatch.setenv("GF_FROZEN_CSR", "0")
    g_sparse, _ = _run_capped(6, comm=comm)
    np.testing.assert_allclose(g, g_sparse, atol=1e-9)
