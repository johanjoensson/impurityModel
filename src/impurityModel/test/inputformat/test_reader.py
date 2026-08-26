"""Reading, validating and resolving an input file.

The tests are organised around the failure modes the format exists to prevent, not around
the reader's functions: a wrong unit, a typo that does nothing, a stale key from another
variant, a knob that silently clamps.
"""

import pytest

from impurityModel.ed.average import k_B
from impurityModel.ed.h0_format import RY_TO_EV
from impurityModel.inputformat import reader
from impurityModel.inputformat.capabilities import InvalidShellCombination, UnsupportedCalculation
from impurityModel.inputformat.reader import InputError, load_input

from .conftest import MINIMAL_SELFENERGY, MINIMAL_SPECTROSCOPY


def test_a_minimal_file_resolves(write_input):
    resolved = load_input(write_input(MINIMAL_SPECTROSCOPY))
    assert resolved.calculation == "spectroscopy"
    assert resolved.hamiltonian_source == "file"
    assert resolved.interaction_kind == "slater"
    assert resolved.dc_scheme == "mlft"
    assert [s["role"] for s in resolved.shells] == ["core", "valence"]
    assert not resolved.warnings


def test_the_calculation_is_chosen_by_which_table_is_present(write_input):
    """No `type` string: a tag beside the tables it names can go stale against them."""
    assert load_input(write_input(MINIMAL_SELFENERGY)).calculation == "selfenergy"
    both = MINIMAL_SPECTROSCOPY + "\n[selfenergy]\n"
    with pytest.raises(InputError, match="Exactly one of"):
        load_input(write_input(both))


# --------------------------------------------------------------------------- units


def test_units_energy_is_required(write_input):
    text = MINIMAL_SPECTROSCOPY.replace('[units]\nenergy = "eV"\n', "")
    with pytest.raises(InputError, match=r"\[units\].energy is required"):
        load_input(write_input(text))


def test_rydberg_and_its_hand_converted_ev_twin_agree(write_input):
    """The whole point of `[units]`: the same physics written two ways must resolve equal."""
    ev = load_input(write_input(MINIMAL_SPECTROSCOPY, "ev.toml"))
    ry_text = MINIMAL_SPECTROSCOPY.replace('energy = "eV"', 'energy = "Ry"')
    ry_text = ry_text.replace("soc = 11.629", f"soc = {11.629 / RY_TO_EV!r}")
    ry_text = ry_text.replace("soc = 0.096", f"soc = {0.096 / RY_TO_EV!r}")
    ry_text = ry_text.replace("c = 1.5", f"c = {1.5 / RY_TO_EV!r}")
    ry_text = ry_text.replace(
        "F_vv = [7.5, 0, 9.9, 0, 6.6]", "F_vv = [{}]".format(", ".join(repr(v / RY_TO_EV) for v in [7.5, 0, 9.9, 0, 6.6]))
    )
    ry = load_input(write_input(ry_text, "ry.toml"))
    assert ry.shells[0]["soc"] == pytest.approx(ev.shells[0]["soc"])
    assert ry.shells[1]["soc"] == pytest.approx(ev.shells[1]["soc"])
    assert ry.tables["double_counting.mlft"]["c"] == pytest.approx(ev.tables["double_counting.mlft"]["c"])
    assert ry.tables["interaction.slater"]["F_vv"] == pytest.approx(ev.tables["interaction.slater"]["F_vv"])


def test_hartree_is_accepted_and_is_two_rydberg(write_input):
    text = MINIMAL_SPECTROSCOPY.replace('energy = "eV"', 'energy = "Ha"').replace("soc = 0.096", "soc = 1.0")
    assert load_input(write_input(text)).shells[1]["soc"] == pytest.approx(2 * RY_TO_EV)


@pytest.mark.parametrize(
    "table,key,value",
    [
        ("susceptibility", "energy_cut", 10.0),
        ("double_counting.fixed_occupation", "occupation", 8.0),
    ],
)
def test_dimensionless_keys_survive_a_rydberg_file_unscaled(write_input, table, key, value):
    """R8: these read like energies and are not. A blanket conversion corrupts them."""
    text = MINIMAL_SELFENERGY.replace('energy = "eV"', 'energy = "Ry"')
    if table.startswith("double_counting"):
        text += f"\n[{table}]\n{key} = {value}\n"
    else:
        text = text.replace("[selfenergy]", f"[{table}]\n{key} = {value}")
    resolved = load_input(write_input(text))
    assert resolved.tables[table][key] == pytest.approx(value)


def test_nixs_momentum_transfer_is_not_an_energy(write_input):
    text = MINIMAL_SPECTROSCOPY.replace('energy = "eV"', 'energy = "Ry"')
    text += '\n[spectroscopy.nixs]\nenabled = true\nradial_file = "r.dat"\nq = [[1.0, 0.0, 4.0]]\n'
    resolved = load_input(write_input(text))
    assert resolved.tables["spectroscopy.nixs"]["q"] == [[1.0, 0.0, 4.0]]


def test_temperature_collapses_to_tau_in_ev(write_input):
    """Kelvin and tau are different unit governances, so exactly one may be written."""
    spectroscopy = load_input(write_input(MINIMAL_SPECTROSCOPY))
    assert spectroscopy.tables["temperature"]["tau"] == pytest.approx(k_B * 300)

    explicit = load_input(write_input(MINIMAL_SPECTROSCOPY + "\n[temperature]\nkelvin = 100\n"))
    assert explicit.tables["temperature"]["tau"] == pytest.approx(k_B * 100)

    as_energy = load_input(write_input(MINIMAL_SPECTROSCOPY + "\n[temperature]\ntau = 0.05\n"))
    assert as_energy.tables["temperature"]["tau"] == pytest.approx(0.05)

    with pytest.raises(InputError, match="exactly one"):
        load_input(write_input(MINIMAL_SPECTROSCOPY + "\n[temperature]\nkelvin = 300\ntau = 0.002\n"))


def test_selfenergy_keeps_its_own_temperature_default(write_input):
    """R7: one shared default cannot serve three drivers."""
    assert load_input(write_input(MINIMAL_SELFENERGY)).tables["temperature"]["tau"] == pytest.approx(0.002)


def test_per_calculation_basis_defaults_are_not_shared(write_input):
    """occ_cutoff differs by six orders of magnitude between the two paths."""
    spectroscopy = load_input(write_input(MINIMAL_SPECTROSCOPY, "a.toml"))
    selfenergy = load_input(write_input(MINIMAL_SELFENERGY, "b.toml"))
    assert spectroscopy.tables["many_body_basis"]["occ_cutoff"] == 1e-6
    assert spectroscopy.tables["many_body_basis"]["dN"] == 2
    assert selfenergy.tables["many_body_basis"]["occ_cutoff"] is None
    assert selfenergy.tables["many_body_basis"]["dN"] is None


# ------------------------------------------------------------------ forward compatibility


def test_an_unknown_key_at_or_below_our_minor_is_a_typo(write_input):
    text = MINIMAL_SPECTROSCOPY.replace("soc = 0.096", "sock = 0.096")
    with pytest.raises(InputError, match="unknown key 'sock'.*Did you mean 'soc'"):
        load_input(write_input(text))


def test_an_unknown_key_from_a_newer_minor_is_ignored_with_a_warning(write_input):
    text = MINIMAL_SPECTROSCOPY.replace("version = [1, 0]", "version = [1, 9]").replace("soc = 0.096", "sock = 0.096")
    resolved = load_input(write_input(text))
    assert any("sock" in w for w in resolved.warnings)


def test_a_newer_major_is_refused(write_input):
    text = MINIMAL_SPECTROSCOPY.replace("version = [1, 0]", "version = [2, 0]")
    with pytest.raises(InputError, match="major version"):
        load_input(write_input(text))


def test_an_unrecognised_required_feature_always_fails(write_input):
    text = MINIMAL_SPECTROSCOPY.replace("version = [1, 0]", 'version = [1, 9]\nrequired_features = ["warp_drive"]')
    with pytest.raises(InputError, match="warp_drive"):
        load_input(write_input(text))


# ----------------------------------------------------------------------- tagged unions


def test_two_variants_of_one_union_are_mutually_exclusive(write_input):
    text = MINIMAL_SPECTROSCOPY + '\n[hamiltonian.archive]\npath = "a.h5"\n'
    with pytest.raises(InputError, match="mutually exclusive"):
        load_input(write_input(text))


def test_a_key_directly_in_a_union_root_is_refused(write_input):
    """A stray key in [hamiltonian] would be a stale leftover from another variant."""
    text = MINIMAL_SPECTROSCOPY.replace("[hamiltonian.file]", '[hamiltonian]\nsource = "file"\n[hamiltonian.file]')
    with pytest.raises(InputError, match="selected by which sub-table"):
        load_input(write_input(text))


def test_an_archive_refuses_tables_it_would_override(write_input):
    text = MINIMAL_SELFENERGY.replace('[hamiltonian.file]\npath = "h0.h0"', '[hamiltonian.archive]\npath = "a.h5"')
    with pytest.raises(InputError, match="supplies the model"):
        load_input(write_input(text))


def test_a_double_counting_scheme_the_driver_would_ignore_is_refused(write_input):
    """R1: run_spectra never reads model.dc, so this would be a silent no-op."""
    text = MINIMAL_SPECTROSCOPY.replace("[double_counting.mlft]\nc = 1.5", "[double_counting.fll]\nu = 7.5\nj = 0.9")
    with pytest.raises(InputError, match="never reads model.dc"):
        load_input(write_input(text))


def test_mlft_is_refused_on_the_selfenergy_path(write_input):
    text = MINIMAL_SELFENERGY + "\n[double_counting.mlft]\nc = 1.5\n"
    with pytest.raises(InputError, match="no meaning here"):
        load_input(write_input(text))


# --------------------------------------------------------------------------- shells


def test_exactly_one_valence_shell_is_required(write_input):
    text = MINIMAL_SPECTROSCOPY.replace('role = "valence"', 'role = "core"')
    with pytest.raises(InputError, match='role = "valence"'):
        load_input(write_input(text))


def test_nominal_occupation_cannot_exceed_the_shell(write_input):
    text = MINIMAL_SPECTROSCOPY.replace("nominal_occupation = 8", "nominal_occupation = 11")
    with pytest.raises(InputError, match="exceeds the 10 spin-orbitals"):
        load_input(write_input(text))


def test_bath_counts_may_be_omitted(write_input):
    """They are deduced from the .h0 header; only a non-.h0 source needs them by hand."""
    text = MINIMAL_SPECTROSCOPY.replace("n_bath = 60\nn_valence_bath = 10\n", "")
    resolved = load_input(write_input(text))
    assert "n_bath" not in resolved.shells[1]


def test_a_shell_l_the_solver_cannot_handle_still_parses_then_fails_on_capability(write_input):
    """The future-proofing contract: valid input, unsupported solver, and they read differently."""
    text = MINIMAL_SPECTROSCOPY.replace('l = 1\nrole = "core"', 'l = 2\nrole = "core"')
    text = text.replace('l = 2\nrole = "valence"', 'l = 3\nrole = "valence"')
    text = text.replace("F_vv = [7.5, 0, 9.9, 0, 6.6]", "F_vv = [7.5, 0, 9.9, 0, 6.6, 0, 5.0]")
    with pytest.raises(UnsupportedCalculation) as excinfo:
        load_input(write_input(text))
    assert "The input file is valid" in str(excinfo.value)
    assert "M4,5" in str(excinfo.value)


def test_a_forbidden_transition_is_invalid_input_not_an_unsupported_one(write_input):
    """|l_core - l_valence| = 2 is zero by selection rule, so no generalisation would help."""
    text = MINIMAL_SPECTROSCOPY.replace('l = 2\nrole = "valence"', 'l = 3\nrole = "valence"')
    text = text.replace("F_vv = [7.5, 0, 9.9, 0, 6.6]", "F_vv = [7.5, 0, 9.9, 0, 6.6, 0, 5.0]")
    with pytest.raises(InvalidShellCombination, match="Gaunt selection rule"):
        load_input(write_input(text))


def test_a_core_shell_larger_than_the_valence_shell_is_invalid_input(write_input):
    """Caught before the capability gate: it is a mislabelling, not a missing feature."""
    text = MINIMAL_SPECTROSCOPY.replace('l = 1\nrole = "core"', 'l = 3\nrole = "core"')
    with pytest.raises(InvalidShellCombination, match="exceeds l_valence"):
        load_input(write_input(text))


# -------------------------------------------------------------------- spectroscopy


def test_every_technique_has_an_explicit_switch(write_input):
    resolved = load_input(write_input(MINIMAL_SPECTROSCOPY))
    assert resolved.tables["spectroscopy.xas"]["enabled"] is True
    assert resolved.tables["spectroscopy.rixs"]["enabled"] is False

    text = MINIMAL_SPECTROSCOPY + "\n[spectroscopy.rixs]\nenabled = true\n"
    assert load_input(write_input(text)).tables["spectroscopy.rixs"]["enabled"] is True


def test_disabling_everything_is_refused(write_input):
    text = MINIMAL_SPECTROSCOPY
    for technique in ("pes", "xps", "xas"):
        text += f"\n[spectroscopy.{technique}]\nenabled = false\n"
    with pytest.raises(InputError, match="nothing to compute"):
        load_input(write_input(text))


def test_nixs_requires_its_radial_file_when_enabled(write_input):
    """Previously supplying the file WAS the switch; now the switch is the switch."""
    text = MINIMAL_SPECTROSCOPY + "\n[spectroscopy.nixs]\nenabled = true\n"
    with pytest.raises(InputError, match="no radial_file"):
        load_input(write_input(text))


def test_a_momentum_transfer_along_z_is_warned_about(write_input):
    """It currently yields an all-NaN spectrum with no exception."""
    text = MINIMAL_SPECTROSCOPY + '\n[spectroscopy.nixs]\nenabled = true\nradial_file = "r.dat"\nq = [[0.0, 0.0, 4.0]]\n'
    resolved = load_input(write_input(text))
    assert any("NaN" in w for w in resolved.warnings)


def test_the_shared_meshes_and_broadening_live_on_the_parent(write_input):
    """U1/U2: disabling XAS must not change the RIXS lineshape."""
    text = MINIMAL_SPECTROSCOPY + "\n[spectroscopy.rixs]\nenabled = true\n[spectroscopy.xas]\nenabled = false\n"
    resolved = load_input(write_input(text))
    assert resolved.tables["spectroscopy"]["core_hole_broadening"] == 0.2
    assert resolved.tables["spectroscopy"]["w_loss"]["kind"] == "uniform"


# ------------------------------------------------------------------------- meshes


def test_the_three_mesh_shapes(write_input):
    text = MINIMAL_SELFENERGY + "\n[selfenergy.real_axis]\nmesh = { values = [-1.0, 0.0, 1.0] }\n"
    assert load_input(write_input(text)).tables["selfenergy.real_axis"]["mesh"]["values"] == [-1.0, 0.0, 1.0]

    text = MINIMAL_SELFENERGY + '\n[selfenergy.real_axis]\nmesh = { file = "w.dat" }\n'
    assert load_input(write_input(text)).tables["selfenergy.real_axis"]["mesh"]["file"].endswith("w.dat")

    text = MINIMAL_SELFENERGY + "\n[selfenergy.real_axis]\nmesh = { min = 0, max = 1, n = 0 }\n"
    with pytest.raises(InputError, match="at least one point"):
        load_input(write_input(text))


def test_a_mesh_is_converted_like_any_other_energy(write_input):
    text = MINIMAL_SELFENERGY.replace('energy = "eV"', 'energy = "Ry"')
    text += "\n[selfenergy.real_axis]\nmesh = { min = -1, max = 1, n = 3 }\n"
    mesh = load_input(write_input(text)).tables["selfenergy.real_axis"]["mesh"]
    assert mesh["min"] == pytest.approx(-RY_TO_EV)


def test_matsubara_meshes_are_declared_separately_per_statistics(write_input):
    resolved = load_input(write_input(MINIMAL_SELFENERGY))
    assert resolved.tables["selfenergy.matsubara"]["enabled"] is False
    assert "susceptibility.matsubara" not in resolved.tables


# -------------------------------------------------------------------------- paths


def test_paths_resolve_against_the_input_file_not_the_working_directory(write_input):
    resolved = load_input(write_input(MINIMAL_SPECTROSCOPY))
    assert resolved.tables["hamiltonian.file"]["path"].endswith("h0.pickle")
    assert str(write_input(MINIMAL_SPECTROSCOPY, "x.toml").parent) in resolved.tables["hamiltonian.file"]["path"]
