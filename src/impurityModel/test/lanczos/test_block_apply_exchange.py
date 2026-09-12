"""``block_apply``'s distributed array branch under both ``GS_MATVEC_EXCHANGE`` spellings.

``graph`` (the default) sends only the structurally nonzero row blocks over a cached distributed
graph and sums in ascending source order; ``reduce`` posts one full-communicator ``Reduce`` per
destination. Both must reproduce the dense product on every rank, with a CSR ``H`` (the
production case, where the neighbourhood is read from the row pointers) and a dense ``H`` (a
complete graph), with an empty last rank, and under a byte budget small enough to force one
column per exchange round. Bitwise identity *between* the two modes is deliberately not asserted:
their floating-point summation orders differ.
"""

import numpy as np
import pytest
import scipy.sparse as sps
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.BlockLanczosCore import block_apply
from impurityModel.ed.solver_trace import tracing
from impurityModel.test.support.lanczos_fixtures import _contiguous_counts_with_empty_last

GLOBAL_N = 13
WIDTH = 3


@pytest.fixture(autouse=True)
def _knobs_unset(monkeypatch):
    monkeypatch.delenv("GS_MATVEC_EXCHANGE", raising=False)
    monkeypatch.delenv("GS_MATVEC_EXCHANGE_BYTES", raising=False)


class _Basis:
    def __init__(self, comm):
        self.comm = comm
        self.size = GLOBAL_N


def _hermitian(seed=11):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((GLOBAL_N, GLOBAL_N)) + 1j * rng.standard_normal((GLOBAL_N, GLOBAL_N))
    A[rng.random((GLOBAL_N, GLOBAL_N)) < 0.6] = 0.0  # sparse enough that some rank pairs decouple
    return A + A.conj().T


def _local(comm, partition):
    size = comm.size
    if partition == "empty_last":
        counts = _contiguous_counts_with_empty_last(GLOBAL_N, size)
    else:
        counts = [GLOBAL_N // size + (1 if r < GLOBAL_N % size else 0) for r in range(size)]
    c0 = sum(counts[: comm.rank])
    c1 = c0 + counts[comm.rank]
    H = _hermitian()
    rng = np.random.default_rng(5)
    V = rng.standard_normal((GLOBAL_N, WIDTH)) + 1j * rng.standard_normal((GLOBAL_N, WIDTH))
    return H, V, c0, c1


@pytest.mark.mpi
@pytest.mark.parametrize("partition", ["balanced", "empty_last"])
@pytest.mark.parametrize("form", ["csr", "dense"])
@pytest.mark.parametrize("mode", ["graph", "reduce"])
def test_block_apply_matches_dense_product(partition, form, mode, monkeypatch):
    comm = MPI.COMM_WORLD
    monkeypatch.setenv("GS_MATVEC_EXCHANGE", mode)
    H, V, c0, c1 = _local(comm, partition)
    h_local = H[:, c0:c1]
    if form == "csr":
        h_local = sps.csr_array(h_local)
    got = block_apply(h_local, np.ascontiguousarray(V[c0:c1]), _Basis(comm), True, 0.0)
    np.testing.assert_allclose(got, (H @ V)[c0:c1], rtol=1e-12, atol=1e-13)


@pytest.mark.mpi
def test_graph_path_under_a_tiny_budget_runs_one_column_per_round(monkeypatch):
    comm = MPI.COMM_WORLD
    monkeypatch.setenv("GS_MATVEC_EXCHANGE", "graph")
    monkeypatch.setenv("GS_MATVEC_EXCHANGE_BYTES", "16")
    H, V, c0, c1 = _local(comm, "balanced")
    with tracing() as trace:
        got = block_apply(sps.csr_array(H[:, c0:c1]), np.ascontiguousarray(V[c0:c1]), _Basis(comm), True, 0.0)
    np.testing.assert_allclose(got, (H @ V)[c0:c1], rtol=1e-12, atol=1e-13)
    notes = [e for e in trace.events if e["kind"] == "matvec_exchange"]
    assert len(notes) == 1 and notes[0]["site"] == "block_apply"
    if comm.size > 1:
        assert notes[0]["w_c"] == 1 and notes[0]["rounds"] == WIDTH
        assert notes[0]["n_src"] >= 1


@pytest.mark.mpi
def test_graph_path_reports_the_neighbourhood_it_read_from_the_csr(monkeypatch):
    """A block-diagonal H couples no rank to any other: the plan must see an empty neighbourhood
    and the result must still be the (purely local) product -- the sparse case the graph exchange
    exists for, in miniature."""
    comm = MPI.COMM_WORLD
    monkeypatch.setenv("GS_MATVEC_EXCHANGE", "graph")
    size = comm.size
    counts = [GLOBAL_N // size + (1 if r < GLOBAL_N % size else 0) for r in range(size)]
    c0 = sum(counts[: comm.rank])
    c1 = c0 + counts[comm.rank]
    H = np.zeros((GLOBAL_N, GLOBAL_N), dtype=complex)
    lo = 0
    for c in counts:
        blk = np.arange(1, c * c + 1, dtype=complex).reshape(c, c)
        H[lo : lo + c, lo : lo + c] = blk + blk.conj().T
        lo += c
    V = np.linspace(0.5, 2.0, GLOBAL_N * WIDTH).reshape(GLOBAL_N, WIDTH).astype(complex)
    with tracing() as trace:
        got = block_apply(sps.csr_array(H[:, c0:c1]), np.ascontiguousarray(V[c0:c1]), _Basis(comm), True, 0.0)
    np.testing.assert_allclose(got, (H @ V)[c0:c1], rtol=1e-12, atol=1e-13)
    note = next(e for e in trace.events if e["kind"] == "matvec_exchange")
    assert note["n_dest"] == 0 and note["n_src"] == 0 and note["send_bytes"] == 0


def test_invalid_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("GS_MATVEC_EXCHANGE", "tree")
    from impurityModel.ed.mpi_comm import matvec_exchange_mode

    with pytest.raises(ValueError, match="GS_MATVEC_EXCHANGE"):
        matvec_exchange_mode()
    assert config.GS_MATVEC_EXCHANGE.default == "graph"
