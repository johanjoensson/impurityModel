"""The auto GF cap priced at the workload's own matvec fan-out (``gf_units.measure_gf_row_fanout``).

``memory_estimate._GF_MATVEC_ROW_FANOUT_DEFAULT`` (8.1) was measured on SrMnO3; on CrI3 the sparse
path then under-predicted its memory ~2x and the cap only survived through the guard's margin. The
same measurement is now repeated per workload at GF entry, in either direction.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import gf_units
from impurityModel.ed import memory_estimate as me
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.test.basis.test_sparse_matrix_build import _all_dets, _basis, _operator


def test_the_estimate_prices_the_given_fanout_and_defaults_to_the_srmno3_one():
    args = (1_000_000, 48, 4, "none")
    default = me.estimate_gf_peak_bytes(*args, ranks=4)
    assert me.estimate_gf_peak_bytes(*args, ranks=4, row_fanout=me._GF_MATVEC_ROW_FANOUT_DEFAULT) == default
    assert me.estimate_gf_peak_bytes(*args, ranks=4, row_fanout=20.0) > default
    assert me.estimate_gf_peak_bytes(*args, ranks=4, row_fanout=4.0) < default


def test_a_larger_fanout_lowers_the_auto_cap_and_a_smaller_one_raises_it(monkeypatch):
    monkeypatch.setattr(me, "available_bytes_per_rank", lambda comm: 4 * 2**30)

    def cap(fanout):
        return me.max_unit_dets_within_budget(48, 4, "none", 4, None, resident_bytes=2**30, row_fanout=fanout)

    assert cap(20.0) < cap(None) < cap(4.0)


@pytest.mark.mpi
def test_the_measured_fanout_is_the_whole_block_one_and_the_same_on_every_rank():
    """Every determinant of a small basis is sampled; the result is what one apply over all of them
    reaches, per determinant, whatever the rank count (the sample is gathered, not local)."""
    comm = MPI.COMM_WORLD
    dets = _all_dets()
    op = _operator()
    got = gf_units.measure_gf_row_fanout(_basis(dets, comm), op)
    want = len(op.apply_block(ManyBodyState({d: 1.0 for d in dets}), 0.0)) / len(dets)
    assert got == pytest.approx(want)
    assert len(set(comm.allgather(got))) == 1


@pytest.mark.mpi
def test_the_sample_is_bounded(monkeypatch):
    comm = MPI.COMM_WORLD
    monkeypatch.setattr(gf_units, "_ROW_FANOUT_SAMPLE_DETS", 10)
    applied = []
    op = _operator()

    class _Counting:
        def apply_block(self, block, cutoff):
            applied.append(len(block))
            return op.apply_block(block, cutoff)

    gf_units.measure_gf_row_fanout(_basis(_all_dets(), comm), _Counting())
    if comm.rank == 0:
        assert applied and max(applied) <= 10 + comm.size, applied


def test_the_image_is_bounded_by_thinning_the_sample(monkeypatch):
    """A pilot apply sizes the full one: with room for ~50 rows, the block applied is about
    50 / fan-out determinants, never the whole sample."""
    monkeypatch.setattr(gf_units, "_ROW_FANOUT_PILOT_DETS", 20)
    monkeypatch.setattr(gf_units, "_ROW_FANOUT_MAX_ROWS", 50)
    applied = []
    op = _operator()

    class _Counting:
        def apply_block(self, block, cutoff):
            out = op.apply_block(block, cutoff)
            applied.append((len(block), len(out)))
            return out

    fanout = gf_units.measure_gf_row_fanout(_basis(_all_dets(), None), _Counting())
    assert len(_all_dets()) > 100, "premise: the sample is much larger than the bound allows"
    assert all(n_in < len(_all_dets()) for n_in, _ in applied)
    assert fanout > 1.0


def test_run_units_distributed_measures_it_once_for_an_auto_cap(monkeypatch):
    calls = []
    monkeypatch.setattr(gf_units, "_is_auto_gf", lambda basis: True)
    monkeypatch.setattr(gf_units, "measure_gf_row_fanout", lambda basis, hOp: calls.append(1) or 11.0)
    basis = _basis(_all_dets(), None)
    psi = ManyBodyState({basis.local_basis[0]: 1.0})
    for _ in range(2):
        gf_units.run_units_distributed(basis, [[psi]], np.ones(1), lambda b, u, s: u, hOp=_operator())
    assert basis.gf_row_fanout == 11.0
    assert calls == [1], "measured once per calculation, then reused like the pinned cap"
