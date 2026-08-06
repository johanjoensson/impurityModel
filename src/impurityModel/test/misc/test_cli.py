"""Parse / dispatch tests for the umbrella CLI (:mod:`impurityModel.scripts.cli`).

No solver is invoked -- these only exercise argument parsing and sub-command dispatch, so a
``--help``/flag regression is cheap to catch.
"""

import argparse
import sys

import pytest
from mpi4py import MPI

from impurityModel.scripts import cli, selfenergy, spectra, susceptibility
from impurityModel.scripts._units import convert_energy_args


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


def test_spectra_validate_warns_on_rydberg_scale_xi_2p(capsys):
    """The Ry-into-an-eV-interface mistake this check exists for: xi_2p=0.85 is the eV
    default (11.629) divided by 13.6057, to three digits.

    The warning is rank-0-only by design (mirrors CLAUDE.md's rank-0 print convention), so
    only rank 0's captured stdout carries it; other ranks are asserted silent rather than
    skipped, so the gating itself is under test.
    """
    args = _parse(spectra.add_arguments, ["h0.h0", "--xi_2p", "0.85", "--chargeTransferCorrection", "0.11"])
    spectra._validate(args)
    out = capsys.readouterr().out
    if MPI.COMM_WORLD.rank != 0:
        assert out == ""
        return
    assert "WARNING" in out
    assert "xi_2p" in out
    assert "0.85" in out


def test_spectra_validate_silent_at_default_xi_2p(capsys):
    args = _parse(spectra.add_arguments, ["h0.h0"])
    spectra._validate(args)
    assert capsys.readouterr().out == ""


def test_convert_energy_args_scalar_and_list():
    """The shared helper: eV is a no-op, Ry scales scalars and lists uniformly, None passes
    through untouched (an unset optional argument, e.g. selfenergy's --hField default).
    """
    args = argparse.Namespace(xi=0.5, Fdd=[1.0, 2.0, 3.0], hField=None)

    convert_energy_args(args, ["xi", "Fdd", "hField"], "eV")
    assert args.xi == 0.5
    assert args.Fdd == [1.0, 2.0, 3.0]
    assert args.hField is None

    convert_energy_args(args, ["xi", "Fdd", "hField"], "Ry")
    assert args.xi == pytest.approx(0.5 * 13.605693122994232)
    assert args.Fdd == pytest.approx([v * 13.605693122994232 for v in [1.0, 2.0, 3.0]])
    assert args.hField is None


def test_spectra_unit_ry_matches_the_eV_defaults():
    """The tutorial's original mistaken command line, replayed with --unit Ry: the paper's
    Rydberg values should land within rounding of this script's own eV defaults.
    """
    args = _parse(
        spectra.add_arguments,
        [
            "h0.h0",
            "--unit",
            "Ry",
            "--Fdd",
            "0.55",
            "0.0",
            "0.91",
            "0.0",
            "0.56",
            "--Fpd",
            "0.65",
            "0.0",
            "0.52",
            "--Gpd",
            "0.0",
            "0.38",
            "0.0",
            "0.22",
            "--xi_2p",
            "0.85",
            "--chargeTransferCorrection",
            "0.11",
        ],
    )
    convert_energy_args(args, spectra._ENERGY_FIELDS, args.unit)

    # xi_2p and chargeTransferCorrection are the near-exact matches (the eV defaults divided by
    # 13.6057 to 3 digits); Fdd/Fpd/Gpd are the paper's own physically-reasonable values, not
    # literally this codebase's defaults, so they only need to land in the same ballpark (see
    # the percentages in this plan's Finding 1 table -- up to ~25% apart for Fdd).
    eV_defaults = _parse(spectra.add_arguments, ["h0.h0"])
    assert args.xi_2p == pytest.approx(eV_defaults.xi_2p, rel=0.01)
    assert args.chargeTransferCorrection == pytest.approx(eV_defaults.chargeTransferCorrection, rel=0.01)
    assert args.Fdd == pytest.approx(eV_defaults.Fdd, rel=0.3)
    assert args.Fpd == pytest.approx(eV_defaults.Fpd, rel=0.1)
    assert args.Gpd == pytest.approx(eV_defaults.Gpd, rel=0.15)


def test_spectra_validate_silent_under_unit_ry_with_the_tutorials_original_values(capsys):
    """The end-to-end proof --unit Ry is a genuine fix path for Finding 1, not just a
    differently-shaped warning: converted, xi_2p is back in the normal eV range and B3's check
    does not fire.
    """
    args = _parse(
        spectra.add_arguments,
        ["h0.h0", "--unit", "Ry", "--xi_2p", "0.85", "--chargeTransferCorrection", "0.11"],
    )
    convert_energy_args(args, spectra._ENERGY_FIELDS, args.unit)
    spectra._validate(args)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("mod", [selfenergy, susceptibility])
def test_selfenergy_and_susceptibility_unit_ry_converts_xi(mod):
    args = _parse(mod.add_arguments, ["h0.h0", "--unit", "Ry", "--xi", "0.05"])
    convert_energy_args(args, mod._ENERGY_FIELDS, args.unit)
    assert args.xi == pytest.approx(0.05 * 13.605693122994232)


@pytest.mark.parametrize("mod", [spectra, selfenergy, susceptibility])
def test_unit_defaults_to_eV_and_is_a_noop(mod):
    args = _parse(mod.add_arguments, ["h0.h0"])
    assert args.unit == "eV"
    before = dict(vars(args))
    convert_energy_args(args, mod._ENERGY_FIELDS, args.unit)
    assert vars(args) == before


@pytest.mark.parametrize("mod", [spectra, selfenergy, susceptibility])
def test_unit_rejects_unknown_values(mod):
    with pytest.raises(SystemExit):
        _parse(mod.add_arguments, ["h0.h0", "--unit", "kelvin"])


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
