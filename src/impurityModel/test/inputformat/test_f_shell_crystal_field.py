"""An f-shell octahedral crystal field, from TOML file to solved spectra.

The crystal field used to be d-only: ten parameters spelling a single e_g / t_2g splitting.
An f shell cannot be described that way, and not for a naming reason -- the octahedral
invariants of an l=3 shell span a *two*-dimensional space, so one splitting number cannot
place three levels. These tests check the physics that claim rests on, then check that a file
saying so runs.

The oracle is deliberately not the level table: the impurity block is rotated back to the
cubic basis and its eigenvalues are read off, so a wrong transformation matrix or a wrong
column order shows up as levels in the wrong place rather than as agreement with itself.
"""

import numpy as np
import pytest

from impurityModel.ed import hamiltonian_io
from impurityModel.ed.atomic_physics import get_spherical_2_cubic_matrix, octahedral_level_structure
from impurityModel.ed.operator_algebra import daggerOp
from impurityModel.inputformat.build import build
from impurityModel.inputformat.reader import InputError, load_input

F_SHELL = """
[format]
version = [1, 0]

[units]
energy = "eV"

[hamiltonian.crystal_field]
e_imp        = -1.10
e_deltaO_imp =  0.90
e_delta6_imp =  0.21
e_val_t1u = -4.4
e_val_t2u = -5.9
e_val_a2u = -6.5
e_con_t1u = 3.0
e_con_t2u = 2.5
e_con_a2u = 2.0
v_val_t1u = 1.20
v_val_t2u = 0.95
v_val_a2u = 0.70
v_con_t1u = 0.40
v_con_t2u = 0.30
v_con_a2u = 0.20

[[shell]]
l = 2
role = "core"
nominal_occupation = 10
soc = 3.5

[[shell]]
l = 3
role = "valence"
n_bath = 14
n_valence_bath = 14
nominal_occupation = 2
soc = 0.25

[interaction.slater]
F_vv = [5.0, 0.0, 7.0, 0.0, 4.5, 0.0, 3.0]
F_cc = [0.0, 0.0, 0.0, 0.0, 0.0]
F_cv = [6.0, 0.0, 4.0, 0.0, 2.5]
G_cv = [0.0, 3.0, 0.0, 1.8, 0.0, 1.1]

[double_counting.mlft]
c = 1.5

[spectroscopy]
w = {min = -8.0, max = 8.0, n = 41}
w_loss = {min = -1.0, max = 8.0, n = 21}

[spectroscopy.pes]
enabled = false

[spectroscopy.xps]
enabled = false

[spectroscopy.xas]
enabled = true

[spectroscopy.rixs]
enabled = false
"""


@pytest.fixture
def write(tmp_path):
    def _write(text, name="f_shell.toml"):
        path = tmp_path / name
        path.write_text(text)
        return path

    return _write


def _impurity_levels(h0_operator, l):
    """Eigenvalues of the impurity block, per cubic-harmonic column."""
    n = 2 * l + 1
    h = np.zeros((n, n), dtype=complex)
    for process, value in h0_operator.items():
        (i, ci), (j, cj) = process
        if ci == "c" and cj == "a" and len(i) == 3 and len(j) == 3 and i[1] == 0 == j[1]:
            h[i[2] + l, j[2] + l] += value
    u = get_spherical_2_cubic_matrix(l=l)
    return np.real(np.diag(u.conj().T @ h @ u)), np.max(
        np.abs(u.conj().T @ h @ u - np.diag(np.diag(u.conj().T @ h @ u)))
    )


def test_the_f_impurity_block_has_the_three_octahedral_levels():
    """The centre and the two splittings place t_1u, t_2u and a_2u exactly where they claim."""
    l = 3
    e_imp, delta4, delta6 = -1.1, 0.9, 0.21
    parameters = {"e_imp": e_imp, "e_deltaO_imp": delta4, "e_delta6_imp": delta6}
    parameters.update(
        {f"{p}_{irrep}": 0.0 for p in ("e_val", "e_con", "v_val", "v_con") for irrep in ("t1u", "t2u", "a2u")}
    )
    h0 = hamiltonian_io.get_CF_hamiltonian({l: 28}, {l: 14}, parameters, l=l)

    diagonal, off_diagonal = _impurity_levels(h0, l)
    assert off_diagonal < 1e-12, "the cubic basis does not diagonalise the crystal field"

    offset = 0
    for irrep, degeneracy, weights in octahedral_level_structure(l):
        expected = e_imp + weights[0] * delta4 + weights[1] * delta6
        assert np.allclose(diagonal[offset : offset + degeneracy], expected, atol=1e-12), irrep
        offset += degeneracy
    # Traceless splittings: the centre really is the centre.
    assert np.mean(diagonal) == pytest.approx(e_imp, abs=1e-12)


def test_the_two_f_splittings_are_independent():
    """One number cannot place three levels, because the two weight patterns are not parallel.

    "Not equal" would not settle it -- an exactly opposite pattern is still one parameter in
    disguise. The test is proportionality: if the rank-6 weights were any multiple of the
    rank-4 ones, a single splitting with a rescaled value would reproduce every f level and
    ``e_delta6_imp`` would be redundant.
    """
    rank4, rank6 = (np.array([weights[k] for _, _, weights in octahedral_level_structure(3)]) for k in (0, 1))
    assert abs(np.dot(rank4, rank6)) < np.linalg.norm(rank4) * np.linalg.norm(rank6) * (1 - 1e-6)
    # And the independence survives into the assembled Hamiltonian, not just the table.
    base = {"e_imp": 0.0, "e_deltaO_imp": 0.0, "e_delta6_imp": 0.0}
    base.update({f"{p}_{irrep}": 0.0 for p in ("e_val", "e_con", "v_val", "v_con") for irrep in ("t1u", "t2u", "a2u")})
    only_4, _ = _impurity_levels(
        hamiltonian_io.get_CF_hamiltonian({3: 28}, {3: 14}, {**base, "e_deltaO_imp": 1.0}, l=3), 3
    )
    only_6, _ = _impurity_levels(
        hamiltonian_io.get_CF_hamiltonian({3: 28}, {3: 14}, {**base, "e_delta6_imp": 1.0}, l=3), 3
    )
    assert np.linalg.matrix_rank(np.vstack([only_4, only_6]), tol=1e-9) == 2


def test_the_f_crystal_field_carries_no_round_off_couplings():
    """A rotated f matrix leaves ~1e-18 entries where symmetry forbids any coupling at all.

    Keeping them changes no energy, but basis generation walks the H-connectivity closure and
    a 1e-18 hopping is an edge in that graph. The d rotation never produced any, so this only
    ever bit the general path.
    """
    parameters = {"e_imp": -1.1, "e_deltaO_imp": 0.9, "e_delta6_imp": 0.21}
    parameters.update(
        {f"e_val_{irrep}": e for irrep, e in (("t1u", -4.4), ("t2u", -5.9), ("a2u", -6.5))}
        | {f"e_con_{irrep}": e for irrep, e in (("t1u", 3.0), ("t2u", 2.5), ("a2u", 2.0))}
        | {f"v_val_{irrep}": v for irrep, v in (("t1u", 1.2), ("t2u", 0.95), ("a2u", 0.7))}
        | {f"v_con_{irrep}": v for irrep, v in (("t1u", 0.4), ("t2u", 0.3), ("a2u", 0.2))}
    )
    for bath_state_basis in ("spherical", "cubic"):
        h0 = hamiltonian_io.get_CF_hamiltonian({3: 28}, {3: 14}, parameters, bath_state_basis=bath_state_basis, l=3)
        magnitudes = np.abs(np.array(list(h0.values())))
        assert magnitudes.min() > 1e-6, f"{bath_state_basis}: a round-off coupling survived"
        # The rotation is exactly hermitian too: `assert_hermitian` compares operator dicts
        # for equality, not to a tolerance, and the raw f product misses that by 2e-16.
        assert daggerOp(h0) == h0, bath_state_basis


def test_an_f_shell_input_builds_and_the_shells_are_laid_out_by_role(write):
    built = build(load_input(write(F_SHELL)))
    # 3d core block (10 spin-orbitals), then the 4f valence block (14).
    assert built.model.impurity_orbitals == {2: [list(range(10))], 3: [list(range(10, 24))]}
    assert built.model.n_spin_orbitals == 24 + 14
    assert built.model.h0


def test_an_f_shell_input_missing_the_second_splitting_says_so(write):
    with pytest.raises(InputError) as excinfo:
        build(load_input(write(F_SHELL.replace("e_delta6_imp =  0.21\n", ""))))
    assert "e_delta6_imp" in str(excinfo.value)
    assert "l=3 valence shell" in str(excinfo.value)


D_SHELL = """
[format]
version = [1, 0]

[units]
energy = "eV"

[hamiltonian.crystal_field]
e_imp = -1.31796
e_deltaO_imp = 0.60422
e_val_eg = -4.4
e_val_t2g = -6.5
e_con_eg = 3.0
e_con_t2g = 2.0
v_val_eg = 1.883
v_val_t2g = 1.395
v_con_eg = 0.6
v_con_t2g = 0.4

[[shell]]
l = 2
role = "valence"
n_bath = 20
n_valence_bath = 10
nominal_occupation = 8
soc = 0.096

[interaction.slater]
F_vv = [7.5, 0, 9.9, 0, 6.6]

[double_counting.mlft]
c = 1.5

[spectroscopy]
w = {min = -8.0, max = 8.0, n = 41}

[spectroscopy.pes]
enabled = true

[spectroscopy.xps]
enabled = false

[spectroscopy.xas]
enabled = false

[spectroscopy.rixs]
enabled = false
"""


def test_a_d_shell_still_takes_exactly_its_historical_ten_keys(write):
    """The generalisation must not have quietly made any of the d keys optional."""
    assert build(load_input(write(D_SHELL, "d_shell.toml"))).model.h0


def test_a_d_shell_input_refuses_the_f_only_splitting(write):
    """Symmetric to the f case: the rank-6 splitting does not exist for a d shell.

    A d shell has two octahedral levels and one invariant, so a second splitting parameter is
    not merely unused -- there is no operator for it to multiply, and accepting it silently
    would drop a number the user meant to matter.
    """
    text = D_SHELL.replace("e_deltaO_imp = 0.60422", "e_deltaO_imp = 0.60422\ne_delta6_imp = 0.2")
    with pytest.raises(InputError) as excinfo:
        build(load_input(write(text, "d_shell_bad.toml")))
    assert "Not a key for this shell: e_delta6_imp" in str(excinfo.value)
    assert "l=2 valence shell" in str(excinfo.value)


def test_the_required_key_set_follows_the_shell():
    """The single source of truth for "which keys", checked against the level table."""
    assert hamiltonian_io.cf_parameter_names(2) == (
        "e_imp",
        "e_deltaO_imp",
        "e_val_eg",
        "e_val_t2g",
        "e_con_eg",
        "e_con_t2g",
        "v_val_eg",
        "v_val_t2g",
        "v_con_eg",
        "v_con_t2g",
    )
    for l in (0, 1, 2, 3):
        names = hamiltonian_io.cf_parameter_names(l)
        irreps = [irrep for irrep, _, _ in octahedral_level_structure(l)]
        assert len(names) == 1 + (len(irreps) - 1) + 4 * len(irreps)
        assert len(set(names)) == len(names)


F_SHELL_SOLVE = """
[format]
version = [1, 0]

[units]
energy = "eV"

[hamiltonian.crystal_field]
e_imp        = -1.10
e_deltaO_imp =  0.90
e_delta6_imp =  0.21
e_val_t1u = -4.4
e_val_t2u = -5.9
e_val_a2u = -6.5
e_con_t1u = 3.0
e_con_t2u = 2.5
e_con_a2u = 2.0
v_val_t1u = 1.20
v_val_t2u = 0.95
v_val_a2u = 0.70
v_con_t1u = 0.40
v_con_t2u = 0.30
v_con_a2u = 0.20

[[shell]]
l = 3
role = "valence"
n_bath = 14
n_valence_bath = 14
nominal_occupation = 13
soc = 0.25

[interaction.slater]
F_vv = [5.0, 0.0, 7.0, 0.0, 4.5, 0.0, 3.0]

[double_counting.mlft]
c = 1.5

[spectroscopy]
w = {min = -12.0, max = 12.0, n = 41}

[spectroscopy.pes]
enabled = true

[spectroscopy.xps]
enabled = false

[spectroscopy.xas]
enabled = false

[spectroscopy.rixs]
enabled = false
"""


def test_an_f_shell_crystal_field_model_solves(write, tmp_path):
    """The whole chain at l=3: crystal field -> Hamiltonian -> ground state -> spectrum.

    Everything above stops at the assembled operator. This is the one test that runs the
    solver on an f shell, and it is the only place several things are exercised together:
    ``slater_condon_Uop`` and ``dc_MLFT`` at l=3, ``getSOCop(l=3)``, the 7x7 cubic rotation
    inside a real Hamiltonian, and ``assert_hermitian`` on the result -- which the f rotation
    genuinely failed before the rotated blocks were symmetrised, at 2e-16.

    An f13 filling is chosen because it is cheap (a nearly-full shell over a full valence
    bath is a 20-determinant space) while still being a 14-spin-orbital impurity. Even so
    this is the slowest test in the file, around ten seconds: fourteen Green's functions.
    """
    from mpi4py import MPI

    import h5py

    from impurityModel.ed.get_spectra import run_spectra

    built = build(load_input(write(F_SHELL_SOLVE, "f_solve.toml")))
    assert built.model.n_spin_orbitals == 28

    output = tmp_path / "f_shell.h5"
    run_spectra(built.model, built.spectra, built.basis, MPI.COMM_WORLD, verbosity=0, output_filename=str(output))

    if MPI.COMM_WORLD.rank != 0:
        return
    with h5py.File(output, "r") as f:
        energies = np.asarray(f["E"])
        spectrum = np.asarray(f["PS/spectra"])
    assert np.all(np.isfinite(energies))
    # One column per impurity spin-orbital: 14 for an f shell, where a d shell would give 10.
    assert spectrum.shape[1] == 14
    assert np.all(np.isfinite(spectrum))
    assert np.max(np.abs(spectrum)) > 0


def test_two_level_splittings_convert_to_the_two_parameters():
    """The documented inversion, so a user with a paper can set `e_delta6_imp` at all.

    Neither splitting parameter is something a spectrum is quoted in; level positions are.
    ``[hamiltonian.crystal_field]`` documents the closed form, and this checks it round-trips
    through the real assembler rather than through the level table it came from.
    """
    for d1, d2 in ((1.0, 0.3), (0.5, -0.2), (2.0, 1.5)):
        delta_4 = 9 * (3 * d1 - d2) / 22
        delta_6 = 3 * (delta_4 - d1)
        parameters = {"e_imp": 0.0, "e_deltaO_imp": delta_4, "e_delta6_imp": delta_6}
        diagonal, off_diagonal = _impurity_levels(hamiltonian_io.get_CF_hamiltonian({3: 0}, {3: 0}, parameters, l=3), 3)
        assert off_diagonal < 1e-12
        # Column order is t1u (0..2), t2u (3..5), a2u (6).
        assert diagonal[0] - diagonal[6] == pytest.approx(d1, abs=1e-12)
        assert diagonal[3] - diagonal[6] == pytest.approx(d2, abs=1e-12)
