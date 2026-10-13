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


def test_the_bicgstab_estimate_prices_its_unchunked_matvec_at_the_fanout():
    """BiCGSTAB's matvec applies H to the whole block at once: its raw image is priced at the
    fan-out with no row-chunk credit, so the estimate moves by exactly that term."""
    args = (1_000_000, 48, 4, "none")
    low = me.estimate_gf_peak_bytes(*args, ranks=1, method="bicgstab", row_fanout=4.0)
    high = me.estimate_gf_peak_bytes(*args, ranks=1, method="bicgstab", row_fanout=12.0)
    row_bytes = me._COMPLEX_BYTES * 4 + me._key_heap_bytes(48) + me._SD_STRUCT_BYTES
    rows = int(np.ceil(1_000_000 * me._routing_skew_factor(1)))
    assert high - low == pytest.approx(8.0 * rows * row_bytes, rel=1e-6)


def _recording_measurement(monkeypatch):
    """Treat the cap as automatic and record which operator the engine measures the fan-out of."""
    seen = []
    monkeypatch.setattr(gf_units, "_is_auto_gf", lambda basis: True)
    monkeypatch.setattr(gf_units, "measure_gf_row_fanout", lambda basis, hOp: seen.append(hOp) or 8.1)
    return seen


def test_the_bicgstab_driver_has_its_operator_measured(monkeypatch):
    from impurityModel.test.gf.test_gf_bicgstab_driver import _run_driver

    seen = _recording_measurement(monkeypatch)
    _run_driver("bicgstab", None)
    assert seen and all(op is not None for op in seen), "the BiCGSTAB driver did not hand hOp to the engine"


def test_the_rixs_driver_has_its_operator_measured(monkeypatch):
    from impurityModel.test.spectra.test_rixs_tensor import _model, _run_rixs, _thermal_states, _tin_tout

    seen = _recording_measurement(monkeypatch)
    op = _model()
    psis, es, dets, _states, _vecs = _thermal_states(op, 2)
    tin, tout = _tin_tout()
    _run_rixs(op, psis, es, tin, tout, dets)
    assert seen and all(o is op for o in seen), "the RIXS driver did not hand its Hamiltonian to the engine"


def test_the_spectra_driver_has_its_operator_measured(monkeypatch):
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
    from impurityModel.test.spectra.test_spectra_sector_confinement import (
        _anderson_4,
        _fresh_basis,
        _ground_state_half_filled,
        _run,
    )

    seen = _recording_measurement(monkeypatch)
    op = _anderson_4()
    gs, e0, dets = _ground_state_half_filled(op)
    _run(op, [ManyBodyOperator({((0, "a"),): 1.0})], gs, e0, _fresh_basis(dets), True, monkeypatch)
    assert seen and all(o is op for o in seen), "the spectra driver did not hand its Hamiltonian to the engine"


def test_a_pinned_gf_cap_under_an_auto_ground_state_cap_is_not_described_as_auto():
    """``CapPolicy(gs=None, gf=N, from_memory=True)``: the GF cap is the user's, and no fan-out is measured."""
    from types import SimpleNamespace

    from impurityModel.ed.memory_estimate import CapPolicy

    basis = SimpleNamespace(cap_policy=CapPolicy(gs=None, gf=12_421_614, from_memory=True))
    line = gf_units._describe_gf_cap(basis, 12_421_614, "layout")
    assert "set by you" in line and "auto" not in line and "fan-out" not in line, line
