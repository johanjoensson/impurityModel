"""The shipped example inputs, and the launcher scripts that point at them.

An example that has quietly stopped working is worse than no example: it is the first thing a
new user runs. These tests keep every one of them parseable, buildable and reachable.
"""

import re
from pathlib import Path

import pytest

from impurityModel.inputformat.build import build
from impurityModel.inputformat.reader import load_input

REPO = Path(__file__).resolve().parents[4]
EXAMPLES = sorted((REPO / "examples").glob("*.toml")) if (REPO / "examples").is_dir() else []
SCRIPTS = sorted((REPO / "scripts").glob("run_*.sh")) if (REPO / "scripts").is_dir() else []

pytestmark = pytest.mark.skipif(not EXAMPLES, reason="examples/ is not part of an installed package")


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_every_example_validates(path):
    """What `--check` does, which is what CI should run on each of these."""
    load_input(path)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_every_example_builds_a_model(path):
    """Validation only checks the file; this checks it against the Hamiltonian it names."""
    built = build(load_input(path))
    assert built.model.n_spin_orbitals > 0


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_every_example_declares_its_energy_unit(path):
    """The one key with no default, and the one whose absence is a 13.6x error."""
    assert load_input(path).tables["units"]["energy"] in ("eV", "Ry", "Ha")


@pytest.mark.parametrize("path", [p for p in EXAMPLES if "spectra" in p.name], ids=lambda p: p.name)
def test_every_spectroscopy_example_computes_something(path):
    """A file with every technique switched off would validate and produce an empty archive."""
    built = build(load_input(path))
    assert any(getattr(built.spectra, name) for name in ("pes", "xps", "xas", "rixs", "nixs"))


def test_the_examples_cover_all_three_calculations():
    """Self-energy and susceptibility had no worked example at all before this."""
    calculations = {load_input(path).calculation for path in EXAMPLES}
    assert calculations == {"spectroscopy", "selfenergy", "susceptibility"}


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_an_inherited_parameter_is_flagged_where_it_is_inherited(path):
    """The CoO/FeO/MnO conduction bath came from Ni-in-NiO defaults nobody wrote down.

    Now that the ten crystal-field parameters are all stated explicitly, the ones that were
    silently inherited are indistinguishable from measured ones unless the file says so.
    """
    text = path.read_text()
    if "inherited from Ni" in text:
        assert "UNVERIFIED" in text, "an inherited value must say it is unverified"
        for key in ("e_con_eg", "e_con_t2g", "v_con_eg", "v_con_t2g"):
            assert re.search(rf"^{key} = ", text, re.M), f"{key} must be written out, not left to default"


@pytest.mark.skipif(not SCRIPTS, reason="scripts/ is not part of an installed package")
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_launcher_script_points_at_an_example_that_exists(script):
    """The scripts kept the two things a .toml cannot carry: the rank count and the launcher."""
    text = script.read_text()
    referenced = re.findall(r"examples/[\w.]+\.toml", text)
    assert referenced, f"{script.name} references no input file"
    for name in referenced:
        assert (REPO / name).is_file(), f"{script.name} points at a missing {name}"
    assert "mpirun" in text or "mpiexec" in text
    assert "ranks=" in text, "the rank count is the thing the script exists to carry"
