"""Shared minimal input files for the reader tests."""

import pytest

#: A complete, valid spectroscopy input. Deliberately close to the smallest file that works,
#: so a test that needs something extra has to add it visibly.
MINIMAL_SPECTROSCOPY = """
[format]
version = [1, 0]

[units]
energy = "eV"

[hamiltonian.file]
path = "h0.pickle"

[[shell]]
l = 1
role = "core"
nominal_occupation = 6
soc = 11.629

[[shell]]
l = 2
role = "valence"
n_bath = 60
n_valence_bath = 10
nominal_occupation = 8
soc = 0.096

[interaction.slater]
F_vv = [7.5, 0, 9.9, 0, 6.6]
F_cc = [0, 0, 0]
F_cv = [8.9, 0, 6.8]
G_cv = [0, 5.0, 0, 2.8]

[double_counting.mlft]
c = 1.5

[spectroscopy]
"""

MINIMAL_SELFENERGY = """
[format]
version = [1, 0]

[units]
energy = "eV"

[hamiltonian.file]
path = "h0.h0"

[[shell]]
l = 2
role = "valence"
n_bath = 10
n_valence_bath = 10
nominal_occupation = 8

[interaction.slater]
F_vv = [7.5, 0, 9.9, 0, 6.6]

[selfenergy]
"""


@pytest.fixture
def write_input(tmp_path):
    """Write an input file (plus a dummy Hamiltonian so path resolution has a target)."""

    def _write(text, name="input.toml"):
        (tmp_path / "h0.pickle").write_bytes(b"")
        (tmp_path / "h0.h0").write_bytes(b"")
        path = tmp_path / name
        path.write_text(text)
        return path

    return _write
