"""Persistence of the ground-state report: ``ReportFiles``, the text report and the legacy ``stats_path``."""

import json
import os

import pytest

from impurityModel.ed.gs_calc import calc_gs
from impurityModel.ed.gs_report import ReportFiles
from impurityModel.test.basis.test_calc_gs_ensemble import COMMS, _setup


def _rank(comm):
    return comm.rank if comm is not None else 0


def test_for_label_follows_the_dat_file_convention(tmp_path):
    files = ReportFiles.for_label("Mn 1", tmp_path)
    assert files.text == str(tmp_path / "ground_state-Mn 1.txt")
    assert files.json == str(tmp_path / "ground_state-Mn 1.json")
    assert ReportFiles.for_label("Ni").text == "ground_state-Ni.txt"
    assert ReportFiles.coerce(None) == ReportFiles()
    assert ReportFiles.coerce("a.json") == ReportFiles(json="a.json")


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
def test_files_are_written_at_every_verbosity_including_zero(comm, tmp_path):
    h_op, setup, bs, rot, _, _ = _setup(20, comm, "on")
    calc_gs(h_op, setup, bs, rot, 0, slaterWeightMin=1e-12, report_files=ReportFiles.for_label("T", tmp_path))
    if comm is not None:
        comm.Barrier()
    if _rank(comm) == 0:
        txt = (tmp_path / "ground_state-T.txt").read_text()
        assert txt.startswith("=") and "Ground state" in txt and "Thermal expectation values" in txt
        data = json.loads((tmp_path / "ground_state-T.json").read_text())
        assert data["report"]["schema_version"] == 4


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
def test_the_default_writes_nothing(comm, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h_op, setup, bs, rot, _, _ = _setup(20, comm, "on")
    calc_gs(h_op, setup, bs, rot, 0, slaterWeightMin=1e-12)
    if comm is not None:
        comm.Barrier()
    assert os.listdir(tmp_path) == []


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
def test_the_legacy_stats_path_warns_and_writes_only_the_json(comm, tmp_path):
    h_op, setup, bs, rot, _, _ = _setup(20, comm, "on")
    with pytest.warns(DeprecationWarning, match="stats_path"):
        calc_gs(h_op, setup, bs, rot, 0, slaterWeightMin=1e-12, stats_path=str(tmp_path / "gs.json"))
    if comm is not None:
        comm.Barrier()
    if _rank(comm) == 0:
        assert sorted(os.listdir(tmp_path)) == ["gs.json"]


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
def test_a_write_failure_raises_on_every_rank(comm, tmp_path):
    h_op, setup, bs, rot, _, _ = _setup(20, comm, "on")
    files = ReportFiles.for_label("T", tmp_path / "no" / "such" / "directory")
    with pytest.raises(Exception):  # noqa: B017 -- which exception type the OS/MPI layer raises is not the contract
        calc_gs(h_op, setup, bs, rot, 0, slaterWeightMin=1e-12, report_files=files)
