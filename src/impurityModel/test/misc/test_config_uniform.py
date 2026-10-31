"""``config.assert_uniform``: a knob that differs between ranks is an error on every rank, not a deadlock."""

import pytest
from mpi4py import MPI

from impurityModel.ed import config

KNOB = "GS_SECTOR_CERTIFY"  # a bool knob that gates collectives


def test_serial_comm_none_passes(monkeypatch):
    monkeypatch.setenv(KNOB, "0")
    config.assert_uniform(None)
    config.assert_uniform(MPI.COMM_SELF)


def test_unset_and_empty_normalise_alike_for_non_str_knobs(monkeypatch):
    monkeypatch.delenv(KNOB, raising=False)
    unset = config._digest(config._raw_knob_values())
    monkeypatch.setenv(KNOB, "")
    assert config._digest(config._raw_knob_values()) == unset
    monkeypatch.setenv(KNOB, "0")
    assert config._digest(config._raw_knob_values()) != unset


def test_exempt_knob_is_not_hashed(monkeypatch):
    monkeypatch.setenv("GF_SECTOR_CACHE_DIR", "/some/path")
    assert "GF_SECTOR_CACHE_DIR" not in config._raw_knob_values()


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="needs two ranks to disagree")
class TestMultiRank:
    def test_agreeing_ranks_pass(self, monkeypatch):
        monkeypatch.setenv(KNOB, "0")
        config.assert_uniform(MPI.COMM_WORLD)

    def test_knob_on_one_rank_raises_everywhere(self, monkeypatch):
        comm = MPI.COMM_WORLD
        monkeypatch.delenv(KNOB, raising=False)
        if comm.rank == 1:
            monkeypatch.setenv(KNOB, "0")
        with pytest.raises(config.NonUniformKnobError, match=KNOB) as err:
            config.assert_uniform(comm)
        msgs = comm.allgather(str(err.value))
        assert len(set(msgs)) == 1
        assert "rank(s) 1" in msgs[0]

    def test_exempt_knob_may_differ(self, monkeypatch):
        comm = MPI.COMM_WORLD
        monkeypatch.setenv("GF_SECTOR_CACHE_DIR", f"/tmp/rank{comm.rank}")
        config.assert_uniform(comm)
