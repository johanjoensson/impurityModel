"""Parse / dispatch tests for the umbrella CLI (:mod:`impurityModel.scripts.cli`).

No solver is invoked -- these only exercise argument parsing and sub-command dispatch, so a
``--help``/flag regression is cheap to catch.
"""

import argparse
import sys

import pytest

from impurityModel.scripts import cli, selfenergy, spectra, susceptibility


def _parse(add_arguments, argv):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(argv)


def test_spectra_defaults_and_optional_radial():
    args = _parse(spectra.add_arguments, ["h0.pickle", "radial.dat"])
    assert args.h0_filename == "h0.pickle"
    assert args.radial_filename == "radial.dat"
    assert args.ls == [1, 2]
    assert args.auto_block_structure is True

    # The radial file is optional now (NIXS is skipped without it).
    args = _parse(spectra.add_arguments, ["h0.pickle"])
    assert args.radial_filename is None


def test_spectra_no_auto_block_structure():
    args = _parse(spectra.add_arguments, ["h0.pickle", "--no-auto-block-structure"])
    assert args.auto_block_structure is False


def test_selfenergy_mesh_and_solver_flags():
    args = _parse(selfenergy.add_arguments, ["h0.pickle"])
    assert args.realaxis is True
    assert args.n_matsubara == 0
    assert args.gf_method == "lanczos"
    assert args.sparse_green is True

    args = _parse(
        selfenergy.add_arguments,
        ["h0.pickle", "--no-realaxis", "--n_matsubara", "16", "--gf-method", "bicgstab", "--no-sparse-green"],
    )
    assert args.realaxis is False
    assert args.n_matsubara == 16
    assert args.gf_method == "bicgstab"
    assert args.sparse_green is False


def test_selfenergy_rejects_bad_gf_method():
    with pytest.raises(SystemExit):
        _parse(selfenergy.add_arguments, ["h0.pickle", "--gf-method", "nope"])


def test_susceptibility_defaults():
    args = _parse(susceptibility.add_arguments, ["h0.pickle"])
    assert args.ls == 2
    assert args.nBaths == 10
    assert args.n_matsubara == 64
    assert args.output == "chi.h5"


def test_selfenergy_from_archive_makes_h0_optional():
    # --from-archive replaces the positional h0 file, which becomes optional.
    args = _parse(selfenergy.add_arguments, ["--from-archive", "arch.h5", "--cluster", "Ni", "--iteration", "3"])
    assert args.h0_filename is None
    assert args.from_archive == "arch.h5"
    assert args.cluster == "Ni"
    assert args.iteration == 3


def test_susceptibility_from_archive_makes_h0_optional():
    args = _parse(susceptibility.add_arguments, ["--from-archive", "arch.h5"])
    assert args.h0_filename is None
    assert args.from_archive == "arch.h5"


def test_cli_dispatches_to_subcommand(monkeypatch):
    recorded = {}

    def fake_add(parser):
        parser.add_argument("h0_filename")

    def fake_run(args):
        recorded["h0"] = args.h0_filename

    monkeypatch.setitem(cli._SUBCOMMANDS, "spectra", (fake_add, fake_run, "help"))
    cli.main(["spectra", "my_h0.pickle"])
    assert recorded["h0"] == "my_h0.pickle"


def test_cli_delegates_plot_subcommand(monkeypatch):
    called = {}
    monkeypatch.setattr(
        cli,
        "_plot_delegates",
        lambda: {"plot-spectra": lambda: called.setdefault("plot", True), "plot-rixs": lambda: None},
    )
    saved_argv = list(sys.argv)
    try:
        cli.main(["plot-spectra", "--filename", "spectra.h5"])
    finally:
        sys.argv = saved_argv
    assert called.get("plot") is True


def test_cli_requires_a_subcommand():
    with pytest.raises(SystemExit):
        cli.main([])


def test_cli_rejects_unknown_subcommand():
    with pytest.raises(SystemExit):
        cli.main(["does-not-exist"])
