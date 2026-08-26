"""The ``run`` / ``init`` / ``schema`` verbs.

``init`` and ``schema`` exist because a rigorous format that takes forty lines and a dozen
unguessable keys to replace one shell line is a worse tool, not a better one. Both are
generated from the schema declarations, so the test that matters most is that what ``init``
writes actually validates -- a template that does not parse is worse than no template.
"""

import argparse
import json
import tomllib

import h5py
import pytest

from impurityModel.inputformat import schema
from impurityModel.inputformat.reader import load_input
from impurityModel.scripts import cli, run_cmd


def _args(**overrides):
    defaults = dict(input=None, check=False, show_resolved=False, verbose=0)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ------------------------------------------------------------------------ init


@pytest.mark.parametrize("calculation", schema.CALCULATIONS)
def test_the_generated_template_is_valid_toml(calculation):
    """Python's ``repr`` writes ``None``, ``True`` and 'single quotes'; TOML accepts none."""
    tomllib.loads(run_cmd.render_template(calculation))


@pytest.mark.parametrize("calculation", schema.CALCULATIONS)
def test_the_generated_template_validates(calculation, tmp_path):
    """The stronger claim: it is not merely parseable, it is a usable starting point."""
    path = tmp_path / "in.toml"
    path.write_text(run_cmd.render_template(calculation))
    resolved = load_input(path)
    assert resolved.calculation == calculation


def test_the_template_states_the_switches_it_would_otherwise_hide():
    """Which spectra run is the thing a user most needs to see, so it is written live."""
    text = run_cmd.render_template("spectroscopy")
    for technique in ("pes", "xps", "xas", "rixs", "nixs"):
        assert f"[spectroscopy.{technique}]" in text
    assert "enabled = true" in text and "enabled = false" in text


def test_the_template_says_the_energy_unit_is_required():
    assert "REQUIRED" in run_cmd.render_template("selfenergy")


# ------------------------------------------------------------------ toml_value


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, "true"),
        (False, "false"),
        (1.5, "1.5"),
        (3, "3"),
        ("cluster", '"cluster"'),
        ([1, 2.5], "[1, 2.5]"),
        ({"min": -1.0, "n": 3}, "{ min = -1.0, n = 3 }"),
    ],
)
def test_toml_value_renders_toml_not_python(value, expected):
    assert run_cmd.toml_value(value) == expected


def test_toml_value_refuses_what_toml_cannot_say():
    """TOML has no null, so an unset default must become a comment, not the text 'None'."""
    assert run_cmd.toml_value(None) is None
    assert run_cmd.toml_value([1, None]) is None
    assert run_cmd.toml_value({"a": None}) is None


def test_toml_value_escapes_a_quoted_string():
    assert run_cmd.toml_value('say "hi"') == '"say \\"hi\\""'


# ---------------------------------------------------------------------- schema


def test_schema_prints_one_table_or_all(capsys):
    run_cmd.show_schema(argparse.Namespace(table=None))
    assert "[spectroscopy.rixs]" in capsys.readouterr().out

    run_cmd.show_schema(argparse.Namespace(table="spectroscopy.rixs"))
    out = capsys.readouterr().out
    # Key lines are indented two spaces and followed by their kind in parentheses; the prose
    # legitimately mentions keys of other tables (the RIXS broadening documents its
    # relationship to the shared core-hole one), so match the key line, not the word.
    assert "\n  final_state_broadening  (" in out
    assert "\n  w_loss  (" not in out, "only the requested table's keys are listed"


def test_an_unknown_table_suggests_the_closest_one():
    with pytest.raises(SystemExit, match="spectroscopy.rixs"):
        run_cmd.show_schema(argparse.Namespace(table="spectroscopy.rix"))


# ------------------------------------------------------------------ provenance


def test_the_provenance_record_carries_the_file_and_its_resolved_meaning(tmp_path):
    """A published input file should determine its result; this is what makes that auditable."""
    source = tmp_path / "in.toml"
    source.write_text(run_cmd.render_template("selfenergy"))
    resolved = load_input(source)

    built = type("Built", (), {"notes": ["deduced something"]})()
    target = tmp_path / "out.h5"
    with h5py.File(target, "w") as handle:
        handle.create_dataset("cluster/x", data=[1.0])
    run_cmd._write_provenance(target, resolved, built)

    with h5py.File(target, "r") as handle:
        record = json.loads(handle["provenance"]["resolved"][()])
        assert handle["provenance"]["input_toml"][()].decode() == resolved.raw_text
    assert record["calculation"] == "selfenergy"
    assert record["declared_energy_unit"] == "eV"
    assert record["deduced"] == ["deduced something"]
    assert record["knobs"], "the tuning knobs are part of what determines the result"
    assert "auto" in record["notes_on_completeness"], "say what the record cannot know"


def test_writing_provenance_to_a_missing_file_is_a_no_op(tmp_path):
    """A driver that wrote nothing (a non-root rank, a failed save) must not become a crash."""
    resolved = load_input(tmp_path / "in.toml") if False else None
    run_cmd._write_provenance(tmp_path / "absent.h5", resolved, None)


# ---------------------------------------------------------------------- wiring


@pytest.mark.parametrize("name", ["run", "init", "schema"])
def test_the_verbs_are_registered_on_the_umbrella_cli(name):
    assert name in cli._SUBCOMMANDS
    add_arguments, run, help_text = cli._SUBCOMMANDS[name]
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    assert help_text


def test_run_parses_its_flags():
    parser = argparse.ArgumentParser()
    run_cmd.add_arguments(parser)
    args = parser.parse_args(["in.toml", "--check"])
    assert args.input == "in.toml" and args.check is True
    assert parser.parse_args(["in.toml", "--show-resolved"]).show_resolved is True


def test_show_resolved_names_where_each_value_came_from(tmp_path, capsys):
    """Told plainly, because "why is this 1e-6?" is otherwise a source-reading exercise."""
    source = tmp_path / "in.toml"
    source.write_text(run_cmd.render_template("spectroscopy"))
    resolved = load_input(source)
    run_cmd._print_resolved(resolved)
    out = capsys.readouterr().out
    assert "energies below are in eV" in out
    assert "default for [spectroscopy]" in out, "the per-calculation defaults must be attributed"
    assert "[environment]" in out and "GF_BICGSTAB_ATOL" in out


# ------------------------------------------------------------------- emit-toml


def _spectra_namespace(**overrides):
    from impurityModel.scripts import spectra as spectra_cmd

    parser = argparse.ArgumentParser()
    spectra_cmd.add_arguments(parser)
    args = parser.parse_args(["h0.pickle"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_emit_toml_produces_a_file_that_validates(tmp_path):
    """The bridge is only useful if its output is a working input file."""
    text = run_cmd.emit_toml("spectra", _spectra_namespace())
    path = tmp_path / "emitted.toml"
    path.write_text(text)
    resolved = load_input(path)
    assert resolved.calculation == "spectroscopy"
    assert [shell["role"] for shell in resolved.shells] == ["core", "valence"]


def test_emit_toml_keeps_the_unit_that_was_typed(tmp_path):
    """It runs before the eV conversion, so it translates the command line rather than
    normalising it. Emitting converted numbers under a "Ry" header, or Rydberg numbers under
    an "eV" one, would each be a 13.6x trap in a file the user is told to trust."""
    text = run_cmd.emit_toml("spectra", _spectra_namespace(unit="Ry", xi_2p=0.85))
    assert 'energy = "Ry"' in text
    assert "soc = 0.85" in text

    path = tmp_path / "ry.toml"
    path.write_text(text)
    from impurityModel.ed.h0_format import RY_TO_EV

    core = next(shell for shell in load_input(path).shells if shell["role"] == "core")
    assert core["soc"] == pytest.approx(0.85 * RY_TO_EV), "the reader converts what emit wrote"


def test_emit_toml_translates_the_old_implicit_switches(tmp_path):
    """RIXS was on when its broadening was positive; NIXS when a radial file was supplied.

    Both were side effects rather than choices, so the emitted file states them as choices
    and records what they used to be inferred from.
    """
    off = run_cmd.emit_toml("spectra", _spectra_namespace(deltaRIXS=0.0, radial_filename=None))
    assert "enabled = false    # was: deltaRIXS > 0" in off
    assert "enabled = false    # was: a radial file was supplied" in off

    on = run_cmd.emit_toml("spectra", _spectra_namespace(radial_filename="Ni3d.dat"))
    assert "enabled = true    # was: deltaRIXS > 0" in on
    assert 'radial_file = "Ni3d.dat"' in on


def test_emit_toml_says_what_it_cannot_translate():
    """A bridge that quietly drops half the format would be worse than none."""
    text = run_cmd.emit_toml("spectra", _spectra_namespace())
    assert "Not expressible as a command-line flag" in text
    assert "[environment]" in text


@pytest.mark.parametrize("command", ["selfenergy", "susceptibility"])
def test_emit_toml_covers_the_other_sub_commands(command, tmp_path):
    from impurityModel.scripts import selfenergy as selfenergy_cmd
    from impurityModel.scripts import susceptibility as susceptibility_cmd

    module = {"selfenergy": selfenergy_cmd, "susceptibility": susceptibility_cmd}[command]
    parser = argparse.ArgumentParser()
    module.add_arguments(parser)
    args = parser.parse_args(["h0.h0"])

    path = tmp_path / f"{command}.toml"
    path.write_text(run_cmd.emit_toml(command, args))
    resolved = load_input(path)
    assert resolved.calculation == command
    # The two Matsubara meshes differ in statistics and convention, so each sub-command must
    # emit its own table rather than a shared key.
    assert f"{command}.matsubara" in resolved.tables


@pytest.mark.parametrize("module_name", ["spectra", "selfenergy", "susceptibility"])
def test_every_sub_command_offers_the_flag(module_name):
    import importlib

    module = importlib.import_module(f"impurityModel.scripts.{module_name}")
    parser = argparse.ArgumentParser()
    module.add_arguments(parser)
    assert parser.parse_args(["h0.h0", "--emit-toml"]).emit_toml is True
    assert parser.parse_args(["h0.h0"]).emit_toml is False
