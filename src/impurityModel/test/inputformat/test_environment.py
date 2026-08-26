"""``[environment]``: setting tuning knobs from a file, including outside the CLI.

The RSPt interface drives the solver through a callback with no argument to thread a path
through, so it finds the file by convention and applies only this one table. That path has
requirements the CLI does not: the knobs must be restored afterwards (the callback runs once
per cluster label per self-consistency iteration), and a value the user already set in their
shell must win.
"""

import os

import pytest

from impurityModel.ed import config
from impurityModel.inputformat import reader
from impurityModel.inputformat.reader import InputError, apply_environment, find_environment_file, load_environment

ENVIRONMENT_ONLY = """
[format]
version = [1, 0]

[environment]
GF_BICGSTAB_ATOL = 1e-9
GF_CIPSI_MAX_ROUNDS = 12
"""


def test_load_environment_reads_only_that_table(tmp_path):
    """The rest of the file describes a model, which on the RSPt path RSPt supplies itself."""
    path = tmp_path / "impurityModel.toml"
    path.write_text(ENVIRONMENT_ONLY + '\n[hamiltonian.file]\npath = "nowhere.h0"\n')
    resolved = load_environment(path)
    assert set(resolved) == {"GF_BICGSTAB_ATOL", "GF_CIPSI_MAX_ROUNDS"}
    # Stored as strings, since that is what os.environ takes; compare by value, not spelling.
    assert float(resolved["GF_BICGSTAB_ATOL"]) == pytest.approx(1e-9)
    assert int(resolved["GF_CIPSI_MAX_ROUNDS"]) == 12


def test_an_unknown_knob_name_gets_an_exact_suggestion(tmp_path):
    """Exact, not heuristic: the registry is enumerable, so a near miss can be named."""
    path = tmp_path / "in.toml"
    path.write_text("[format]\nversion = [1, 0]\n[environment]\nGF_BICGSTAB_ATOLL = 1e-9\n")
    with pytest.raises(InputError, match="Did you mean 'GF_BICGSTAB_ATOL'"):
        load_environment(path)


def test_variables_other_libraries_read_are_refused_with_the_reason(tmp_path):
    """Accepting OMP_NUM_THREADS here would silently do nothing: it is read long before this."""
    path = tmp_path / "in.toml"
    path.write_text("[format]\nversion = [1, 0]\n[environment]\nOMP_NUM_THREADS = 4\n")
    with pytest.raises(InputError, match="consumed long before"):
        load_environment(path)


def test_a_value_that_would_be_silently_clamped_is_reported(tmp_path):
    """``Knob.get`` clamps without saying so, which is not acceptable for a run description."""
    knob = next(k for k in config.KNOBS.values() if k.minimum is not None)
    path = tmp_path / "in.toml"
    path.write_text(f"[format]\nversion = [1, 0]\n[environment]\n{knob.name} = {knob.minimum - 1}\n")
    with pytest.raises(InputError, match="silently clamped"):
        load_environment(path)


def test_an_ambiguous_boolean_is_refused(tmp_path):
    """The underlying parser treats every string but a short false-list as true, 'no' included."""
    knob = next(k for k in config.KNOBS.values() if k.kind == "bool")
    path = tmp_path / "in.toml"
    path.write_text(f'[format]\nversion = [1, 0]\n[environment]\n{knob.name} = "no"\n')
    with pytest.raises(InputError, match="ambiguous as a boolean"):
        load_environment(path)


def test_apply_environment_restores_on_exit():
    """Without this, self-consistency iteration two inherits iteration one's knobs."""
    name = "GF_BICGSTAB_ATOL"
    os.environ.pop(name, None)
    with apply_environment({name: "1e-9"}):
        assert os.environ[name] == "1e-9"
    assert name not in os.environ


def test_apply_environment_restores_a_previous_value():
    name = "GF_BICGSTAB_ATOL"
    os.environ[name] = "1e-7"
    try:
        with apply_environment({name: "1e-9"}):
            assert os.environ[name] == "1e-9"
        assert os.environ[name] == "1e-7"
    finally:
        os.environ.pop(name, None)


def test_an_already_set_variable_wins_when_override_is_false():
    """The rule impurityModel_interface already follows: a shell value beats a file value."""
    name = "GF_BICGSTAB_ATOL"
    os.environ[name] = "1e-7"
    try:
        with apply_environment({name: "1e-9"}, override=False) as skipped:
            assert os.environ[name] == "1e-7"
            assert skipped == [name]
    finally:
        os.environ.pop(name, None)


def test_the_knob_is_read_lazily_so_setting_it_late_still_works():
    """Why writing os.environ is enough, and no import ordering has to be arranged."""
    knob = config.KNOBS["GF_BICGSTAB_ATOL"]
    os.environ.pop(knob.name, None)
    assert knob.get() == knob.default
    with apply_environment({knob.name: "1e-9"}):
        assert knob.get() == pytest.approx(1e-9)
    assert knob.get() == knob.default


def test_find_environment_file_uses_the_convention_then_the_override(tmp_path, monkeypatch):
    monkeypatch.delenv(reader.ENVIRONMENT_PATH_VAR, raising=False)
    assert find_environment_file(tmp_path) is None
    (tmp_path / reader.ENVIRONMENT_FILENAME).write_text(ENVIRONMENT_ONLY)
    assert find_environment_file(tmp_path).endswith(reader.ENVIRONMENT_FILENAME)

    other = tmp_path / "elsewhere.toml"
    other.write_text(ENVIRONMENT_ONLY)
    monkeypatch.setenv(reader.ENVIRONMENT_PATH_VAR, str(other))
    assert find_environment_file(tmp_path) == str(other)


def test_the_reader_is_a_leaf_importable_without_the_solver():
    """impmod_interface must be able to use this without dragging in the drivers or argparse."""
    import subprocess
    import sys

    # argparse is deliberately NOT in this list: numpy imports it (via numpy.f2py.diagnose),
    # so its presence says nothing about whether this module is coupled to the CLI. What would
    # say so is impurityModel.scripts, or any solver module, or mpi4py -- the communicator
    # arrives as an argument precisely so this stays importable without one.
    code = (
        "import sys;"
        "import impurityModel.inputformat.reader;"
        "bad=[m for m in sys.modules if m.startswith(('impurityModel.scripts', 'mpi4py'))"
        " or m.endswith(('.groundstate','.greens_function','.selfenergy','.get_spectra',"
        "'.manybody_basis','.spectra','.susceptibility'))];"
        "print(','.join(sorted(bad)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"reader pulled in {out.stdout.strip()}"
