"""Model interactions and model shells through the TOML front-end.

Every interaction form (Slater-Condon, Kanamori, density-density, explicit terms, a .npy tensor)
compiles to one Coulomb tensor. The load-bearing tests check that equivalent descriptions
build the same model, operator for operator, on every route: a .h0 file (explicit spin or
spin-degenerate), and the spectroscopy assembly. The rest pin the refusals that stop a
plausible-looking input from quietly building a different model.
"""

from pathlib import Path

import numpy as np
import pytest

from impurityModel.ed.interaction_models import kanamori_u4
from impurityModel.ed.lie_algebra import extract_tensors
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.model import atomic_u4, spin_double_solver_matrix
from impurityModel.inputformat.build import build
from impurityModel.inputformat.reader import InputError, load_input

REPO = Path(__file__).resolve().parents[4]
NIO_PICKLE = REPO / "h0" / "h0_NiO_50p10bath.pickle"
GOLDEN_H0 = Path(__file__).resolve().parents[1] / "h0_io" / "golden_h0_v1_index.h0"

#: A spinless SIAM: one impurity orbital, three bath levels, in eV.
SPINLESS_H = np.array(
    [
        [-0.2, 0.3, 0.25, 0.3],
        [0.3, -1.0, 0.0, 0.0],
        [0.25, 0.0, 0.05, 0.0],
        [0.3, 0.0, 0.0, 1.0],
    ]
)


def _write_spinless_h0(path):
    lines = [
        "# impurityModel-h0 v1",
        '{"version": 1, "required_features": ["unit", "energy_reference", "index_convention", "storage"], '
        '"unit": "eV", "energy_reference": "fermi", "n_orb": 4, "index_convention": "impurity-block-first", '
        '"storage": "full", "impurity_orbitals": {"0": [0]}}',
        "--",
    ]
    for i, j in zip(*np.nonzero(SPINLESS_H)):
        lines.append(f"{i} {j} {float(SPINLESS_H[i, j])!r} 0.0")
    path.write_text("\n".join(lines) + "\n")
    return path


SELFENERGY = """
[format]
version = [1, 0]
[units]
energy = "eV"
[hamiltonian.file]
path = "{h0}"
{spin}
[[shell]]
{shell}
role = "valence"
nominal_occupation = {occupation}
{interaction}
[selfenergy]
"""


@pytest.fixture
def run(tmp_path):
    spinless = _write_spinless_h0(tmp_path / "spinless.h0")

    def _run(interaction, *, shell="n_orbitals = 1", spin='spin = "degenerate"', h0=None, occupation=1, extra=""):
        text = SELFENERGY.format(
            h0=h0 or spinless, spin=spin, shell=shell, occupation=occupation, interaction=interaction
        )
        path = tmp_path / "in.toml"
        path.write_text(text + extra)
        return build(load_input(path))

    return _run


def _u4_operator(model):
    return ManyBodyOperator(model.u4)


def assert_same_interaction(model_a, model_b):
    difference = _u4_operator(model_a) - _u4_operator(model_b)
    assert max((abs(v) for v in difference.values()), default=0.0) < 1e-12


def _dense_u4(model):
    n = len(model.impurity_indices)
    _, v, _ = extract_tensors(model.u4, n_orb=n)
    return v + v.transpose(1, 0, 3, 2)


# ---------------------------------------------------------------------------- spin doubling


def test_spin_degenerate_file_doubles_every_orbital(run):
    built = run("[interaction.kanamori]\nU = 2.0")
    model = built.model
    assert model.n_spin_orbitals == 8
    assert model.impurity_orbitals == {0: [0, 1]}
    expected, n_imp = spin_double_solver_matrix(SPINLESS_H, 1)
    assert n_imp == 2
    h = extract_tensors(model.h0, n_orb=8, two_body=False)[0]
    np.testing.assert_allclose(h, expected)
    # the copies really are spin partners: [h, S_z] = [h, S_+] = 0 for the pairing k <-> k + n/2
    # within the impurity block and within the bath block
    sz = np.diag([-0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5])
    splus = np.zeros((8, 8))
    for dn, up in ((0, 1), (2, 5), (3, 6), (4, 7)):
        splus[up, dn] = 1.0
    np.testing.assert_allclose(h @ sz - sz @ h, 0)
    np.testing.assert_allclose(h @ splus - splus @ h, 0)


def test_spinless_file_without_the_declaration_is_refused(run):
    with pytest.raises(InputError, match='spin = "degenerate"'):
        run("[interaction.kanamori]\nU = 2.0", spin="")


def test_bath_counts_are_reported_after_doubling(run):
    built = run("[interaction.kanamori]\nU = 2.0")
    assert any("6 bath orbitals" in note and "2 of them valence" in note for note in built.notes)


# ---------------------------------------------------------------------------- equivalences


def test_every_single_orbital_form_builds_the_same_interaction(run, tmp_path):
    U = 2.5
    spatial = tmp_path / "u_spatial.npy"
    np.save(spatial, np.full((1, 1, 1, 1), U))
    spin_orbital = tmp_path / "u_spin.npy"
    np.save(spin_orbital, atomic_u4(0, [U]))
    reference = run(f"[interaction.kanamori]\nU = {U}").model
    for interaction, shell in [
        (f"[interaction.slater]\nF_vv = [{U}]", "l = 0"),
        (f"[interaction.terms]\nspatial = [[0, 0, 0, 0, {U}]]", "n_orbitals = 1"),
        (f"[interaction.density_density]\nU_opposite_spin = [[{U}]]", "n_orbitals = 1"),
        (f'[interaction.u4_file]\npath = "{spatial}"\nindex_space = "spatial"', "n_orbitals = 1"),
        (f'[interaction.u4_file]\npath = "{spin_orbital}"', "n_orbitals = 1"),
    ]:
        assert_same_interaction(run(interaction, shell=shell).model, reference)


def test_energies_follow_the_units_table(run, tmp_path):
    """U written in Ry is U * 13.6 eV inside, for the inline forms and for a .npy tensor alike."""
    ry = 13.605693122994
    spatial = tmp_path / "u.npy"
    np.save(spatial, np.full((1, 1, 1, 1), 0.1))
    in_ev = run(f"[interaction.kanamori]\nU = {0.1 * ry}").model
    # [units] sits at the top of the template, so rewrite it there
    for interaction in (
        "[interaction.kanamori]\nU = 0.1",
        f'[interaction.u4_file]\npath = "{spatial}"\nindex_space = "spatial"',
    ):
        path = tmp_path / "ry.toml"
        text = SELFENERGY.format(
            h0=_write_spinless_h0(tmp_path / "ry.h0"),
            spin='spin = "degenerate"',
            shell="n_orbitals = 1",
            occupation=1,
            interaction=interaction,
        ).replace('energy = "eV"', 'energy = "Ry"')
        path.write_text(text)
        built = build(load_input(path))
        assert_same_interaction(built.model, in_ev)


def test_two_orbital_explicit_spin_file_takes_kanamori(run):
    """The golden fixture is a 2-orbital shell with spin written out (down first): a model
    shell with n_orbitals = 2 on it gets exactly kanamori_u4."""
    built = run(
        "[interaction.kanamori]\nU = 3.0\nJ = 0.6\nU_prime = 1.9\nJ_pair = 0.4",
        shell="n_orbitals = 2",
        spin="",
        h0=GOLDEN_H0,
        occupation=2,
    )
    np.testing.assert_allclose(
        _dense_u4(built.model),
        kanamori_u4(2, 3.0, J=0.6, U_prime=1.9, J_pair=0.4),
        atol=1e-12,
    )


def test_nominal_dc_of_the_half_filled_hubbard_atom_is_u_over_2(run):
    U = 2.5
    built = run(f"[interaction.kanamori]\nU = {U}", extra="[double_counting.nominal]\n")
    dc = extract_tensors(built.model.dc, n_orb=2, two_body=False)[0]
    np.testing.assert_allclose(dc, U / 2 * np.eye(2), atol=1e-12)


# ---------------------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    "interaction, shell, match",
    [
        ("[interaction.slater]\nF_vv = [2.0]", "n_orbitals = 1", "needs an l shell"),
        ("[interaction.kanamori]\nU = 2.0", "l = 0\nn_orbitals = 1", "exactly one of"),
        (
            "[interaction.terms]\nspin_orbital = [[0, 1, 1, 0, 1.0]]\ncomplete_symmetries = false",
            "n_orbitals = 1",
            "not Hermitian",
        ),
        ("[interaction.terms]\nspatial = [[0, 0, 0, 3, 1.0]]", "n_orbitals = 1", "index 3"),
        ("[interaction.density_density]\nU_opposite_spin = [[1.0, 0.5], [0.5, 1.0]]", "n_orbitals = 1", "1x1"),
        ("[interaction.terms]\n", "n_orbitals = 1", "at least one"),
    ],
)
def test_malformed_interactions_are_refused(run, interaction, shell, match):
    with pytest.raises(InputError, match=match):
        run(interaction, shell=shell)


def test_orbital_basis_is_required_on_an_l_shell(run):
    with pytest.raises(InputError, match="orbital_basis"):
        run("[interaction.kanamori]\nU = 2.0", shell="l = 1", spin="", h0=GOLDEN_H0)


def test_orbital_basis_is_refused_on_a_model_shell(run):
    with pytest.raises(InputError, match="meaningless"):
        run('[interaction.kanamori]\nU = 2.0\norbital_basis = "real_cubic"')


def test_spatial_form_needs_a_down_first_file(run, tmp_path):
    text = GOLDEN_H0.read_text().replace('"spin_ordering": "down_first"', '"spin_ordering": "up_first"')
    up_first = tmp_path / "up_first.h0"
    up_first.write_text(text)
    with pytest.raises(InputError, match="spin_ordering"):
        run("[interaction.kanamori]\nU = 2.0", shell="n_orbitals = 2", spin="", h0=up_first, occupation=2)


def test_core_integrals_outside_spectroscopy_are_refused(run):
    with pytest.raises(InputError, match=r"interaction\.core"):
        run("[interaction.kanamori]\nU = 2.0\n[interaction.core]\nF_cv = [1.0, 0, 0.5]")


# ---------------------------------------------------------------------------- spectroscopy

SPECTROSCOPY = """
[format]
version = [1, 0]
[units]
energy = "eV"
[hamiltonian.file]
path = "{h0}"
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
{interaction}
[double_counting.mlft]
c = 1.5
[spectroscopy]
"""

CORE = "F_cc = [0, 0, 0]\nF_cv = [8.9, 0, 6.8]\nG_cv = [0, 5.0, 0, 2.8]"


@pytest.mark.skipif(not NIO_PICKLE.exists(), reason="needs the shipped NiO Hamiltonian")
def test_spectroscopy_with_a_valence_tensor_matches_slater(tmp_path):
    """The Slater valence tensor, handed over as a .npy in the (l, s, m) order, plus the same
    core integrals in [interaction.core], must build the Slater model term for term -- including
    the MLFT double counting, whose U_vv now comes from the tensor."""
    fvv = [7.5, 0, 9.9, 0, 6.6]
    tensor = tmp_path / "u_dd.npy"
    np.save(tensor, atomic_u4(2, fvv))

    def model(interaction):
        path = tmp_path / "in.toml"
        path.write_text(SPECTROSCOPY.format(h0=NIO_PICKLE, interaction=interaction))
        return build(load_input(path)).model

    slater = model(f"[interaction.slater]\nF_vv = {fvv}\n{CORE}")
    custom = model(f'[interaction.u4_file]\npath = "{tensor}"\n[interaction.core]\n{CORE}')
    difference = ManyBodyOperator(slater.h0) - ManyBodyOperator(custom.h0)
    assert max((abs(v) for v in difference.values()), default=0.0) < 1e-10


def test_model_shell_cannot_drive_spectroscopy(tmp_path):
    path = tmp_path / "in.toml"
    path.write_text(
        SPECTROSCOPY.format(h0=GOLDEN_H0, interaction="[interaction.kanamori]\nU = 2.0").replace(
            "[[shell]]\nl = 2", "[[shell]]\nn_orbitals = 5"
        )
    )
    with pytest.raises(InputError, match="spectroscopy run needs an l valence shell"):
        load_input(path)
