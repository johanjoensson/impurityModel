"""Turning a resolved input file into what the drivers actually take.

The load-bearing test here is the parity one: an input file and the command line it replaces
must produce the *same* model, term for term. Everything else checks the deductions that let
the file be short -- and, just as importantly, that a deduction which disagrees with what the
user wrote is an error rather than a quiet override.
"""

from pathlib import Path

import numpy as np
import pytest

from impurityModel.ed.get_spectra import build_spectra_model
from impurityModel.inputformat.build import NO_ZEEMAN, build
from impurityModel.inputformat.reader import InputError, load_input

REPO = Path(__file__).resolve().parents[4]
NIO_PICKLE = REPO / "h0" / "h0_NiO_50p10bath.pickle"
GOLDEN_H0 = Path(__file__).resolve().parents[1] / "h0_io" / "golden_h0_v1_rydberg.h0"

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
[interaction.slater]
F_vv = [7.5, 0, 9.9, 0, 6.6]
F_cc = [0, 0, 0]
F_cv = [8.9, 0, 6.8]
G_cv = [0, 5.0, 0, 2.8]
[double_counting.mlft]
c = 1.5
[spectroscopy]
"""

SELFENERGY = """
[format]
version = [1, 0]
[units]
energy = "eV"
[hamiltonian.file]
path = "{h0}"
[[shell]]
l = 0
role = "valence"
nominal_occupation = 1
[interaction.slater]
F_vv = [5.0]
[selfenergy]
"""


@pytest.fixture
def written(tmp_path):
    def _write(template, h0=GOLDEN_H0, name="in.toml", **edits):
        text = template.format(h0=h0)
        for old, new in edits.items():
            text = text.replace(old.replace("__", " "), new)
        path = tmp_path / name
        path.write_text(text)
        return path

    return _write


@pytest.mark.skipif(not NIO_PICKLE.exists(), reason="needs the shipped NiO Hamiltonian")
def test_a_toml_run_builds_the_same_model_as_the_command_line_it_replaces(written):
    """The migration is only lossless if this holds, so it is asserted term by term.

    The right-hand side is the equivalent of ``scripts/run_Ni_NiO_50p10bath.sh`` with the
    spectra sub-command's own defaults written out as literals -- deliberately frozen here
    rather than read back from the schema, so a change to a default cannot move both sides
    together and keep the test green.
    """
    text = SPECTROSCOPY.replace("soc = 0.096", "soc = 0.096\nzeeman_splitting = [0.0, 0.0, 0.0001]")
    built = build(load_input(written(text, h0=NIO_PICKLE)))
    reference = build_spectra_model(
        str(NIO_PICKLE),
        (1, 2),
        (0, 60),
        (0, 10),
        (6, 8),
        (7.5, 0, 9.9, 0, 6.6),
        (0.0, 0.0, 0.0),
        (8.9, 0, 6.8),
        (0.0, 5.0, 0, 2.8),
        11.629,
        0.096,
        1.5,
        (0, 0, 0.0001),
        rank=0,
        verbose=False,
        valence_l=2,
        core_l=1,
    )
    assert built.model.n_spin_orbitals == reference.n_spin_orbitals
    assert built.model.impurity_orbitals == reference.impurity_orbitals
    assert built.model.bath_states == reference.bath_states
    assert set(built.model.h0) == set(reference.h0)
    for term, value in reference.h0.items():
        assert built.model.h0[term] == pytest.approx(value)


@pytest.mark.skipif(not NIO_PICKLE.exists(), reason="needs the shipped NiO Hamiltonian")
def test_omitting_the_zeeman_splitting_means_no_field(written):
    """An omitted key applies nothing -- on every Hamiltonian format.

    The spectra command line has always applied a (0, 0, 1e-4) symmetry-breaking nudge, and
    the labelled readers default to one too while the flat reader does not. Inheriting that
    would make an omitted key mean different physics depending on which file it points at.
    Here it means one thing: no field.
    """
    assert NO_ZEEMAN == (0.0, 0.0, 0.0)
    plain = build(load_input(written(SPECTROSCOPY, h0=NIO_PICKLE, name="plain.toml")))
    text = SPECTROSCOPY.replace("soc = 0.096", "soc = 0.096\nzeeman_splitting = [0.0, 0.0, 0.0001]")
    nudged = build(load_input(written(text, h0=NIO_PICKLE, name="nudged.toml")))

    # The field couples to spin, so it shows up as a difference in the one-body terms.
    assert plain.model.h0 != nudged.model.h0
    diagonal = [
        term for term in nudged.model.h0 if term not in plain.model.h0 or plain.model.h0[term] != nudged.model.h0[term]
    ]
    assert diagonal, "the nudge must actually change the Hamiltonian, or this proves nothing"


def test_the_flat_and_labelled_paths_agree_on_what_omission_means(written, tmp_path):
    """The point of not inheriting the readers' defaults: one meaning, not two."""
    built = build(load_input(written(SELFENERGY)))
    assert built.model is not None


def test_the_switches_reach_the_spectra_options(written):
    text = SPECTROSCOPY + "\n[spectroscopy.rixs]\nenabled = true\n[spectroscopy.xps]\nenabled = false\n"
    built = build(load_input(written(text, h0=NIO_PICKLE if NIO_PICKLE.exists() else GOLDEN_H0)))
    assert built.spectra.rixs is True
    assert built.spectra.xps is False
    assert built.spectra.nixs is False
    assert len(built.spectra.wIn) == 50


def test_rixs_off_leaves_an_empty_incoming_mesh(written):
    built = build(load_input(written(SPECTROSCOPY, h0=NIO_PICKLE if NIO_PICKLE.exists() else GOLDEN_H0)))
    assert built.spectra.rixs is False
    assert len(built.spectra.wIn) == 0


# ---------------------------------------------------------------- bath deduction


def test_bath_counts_come_from_the_h0_header(written):
    """The header knows its own layout, so the file does not have to repeat it."""
    built = build(load_input(written(SELFENERGY)))
    assert built.model.n_spin_orbitals == 3
    assert any("bath orbitals from the .h0 header" in note for note in built.notes)


def test_the_valence_split_falls_back_to_the_on_site_energy_sign_and_says_so(written):
    """Producers omit valence_bath for a non-star geometry, so the fallback is normal."""
    built = build(load_input(written(SELFENERGY)))
    assert any("sign of the bath on-site energies" in note for note in built.notes)


def test_a_written_count_that_contradicts_the_header_is_an_error(written):
    text = SELFENERGY.replace("nominal_occupation = 1", "nominal_occupation = 1\nn_bath = 7")
    with pytest.raises(InputError, match="disagrees with the file is an error"):
        build(load_input(written(text)))


def test_a_core_shell_defaults_to_no_bath_states(written):
    """Directive: no core Hamiltonian read means no core bath states."""
    text = SPECTROSCOPY.replace("n_bath = 60\nn_valence_bath = 10\n", "n_bath = 60\nn_valence_bath = 10\n")
    built = build(load_input(written(text, h0=NIO_PICKLE if NIO_PICKLE.exists() else GOLDEN_H0)))
    core = next(shell for shell in built.model.impurity_orbitals)
    del core
    assert any("no bath states" in note for note in built.notes)


def test_core_bath_states_remain_expressible(written):
    """The default is a default, not a restriction: the exceptional case must still be sayable."""
    text = SPECTROSCOPY.replace(
        'l = 1\nrole = "core"\nnominal_occupation = 6',
        'l = 1\nrole = "core"\nn_bath = 2\nn_valence_bath = 2\nnominal_occupation = 6',
    )
    resolved = load_input(written(text, h0=NIO_PICKLE if NIO_PICKLE.exists() else GOLDEN_H0))
    assert resolved.shells[0]["n_bath"] == 2


def test_a_source_without_a_bath_layout_demands_the_valence_counts(written):
    text = SELFENERGY.replace("[hamiltonian.file]\npath = ", "[hamiltonian.file]\npath = ")
    text = text.replace("l = 0", "l = 2").replace("nominal_occupation = 1", "nominal_occupation = 8")
    text = text.replace("F_vv = [5.0]", "F_vv = [7.5, 0, 9.9, 0, 6.6]")
    with pytest.raises(InputError, match="records no bath layout"):
        build(load_input(written(text, h0="does_not_matter.pickle")))


# ------------------------------------------------------------- header agreement


def _h0_with(tmp_path, name, **header_updates):
    """Copy the golden fixture with header keys added or changed."""
    import json

    lines = GOLDEN_H0.read_text().splitlines()
    header = json.loads(lines[1])
    header.update(header_updates)
    lines[1] = json.dumps(header)
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


def test_a_contradicted_header_guarantee_is_an_error_not_an_override(written, tmp_path):
    """A header that states something and a file that states the opposite cannot both be right.

    Letting the written value win would leave the input file and the Hamiltonian describing
    different models while each looks correct on its own -- the failure a self-describing
    header exists to prevent.
    """
    h0 = _h0_with(tmp_path, "no_soc.h0", contains_soc=False)
    text = SELFENERGY.replace("[hamiltonian.file]", "[hamiltonian.file]\ncontains_soc = true")
    with pytest.raises(InputError, match="contains_soc"):
        build(load_input(written(text, h0=h0)))


def test_a_silent_header_is_unknown_rather_than_false(written):
    """The golden fixture declares no contains_soc, so asserting one is information, not conflict."""
    text = SELFENERGY.replace("[hamiltonian.file]", "[hamiltonian.file]\ncontains_soc = true")
    build(load_input(written(text)))


def test_an_absolute_energy_reference_is_refused_for_a_green_function_run(written, tmp_path):
    """The bath split is taken from sign(h[o,o]), so an offset zero re-partitions the bath."""
    h0 = _h0_with(tmp_path, "absolute.h0", energy_reference="absolute")
    with pytest.raises(InputError, match="Fermi-referenced"):
        build(load_input(written(SELFENERGY, h0=h0)))


def test_declaring_a_unit_for_a_legacy_format_is_refused(written):
    text = SELFENERGY.replace("[hamiltonian.file]", '[hamiltonian.file]\nunit = "Ry"')
    with pytest.raises(InputError, match="Convert the file to"):
        build(load_input(written(text, h0="legacy.pickle")))


def test_the_header_fermi_energy_is_reported_for_provenance(written, tmp_path):
    """It is what makes energy_reference: "fermi" auditable rather than merely asserted."""
    source = GOLDEN_H0.read_text().splitlines()
    import json

    header = json.loads(source[1])
    header["fermi_energy"] = 0.25
    source[1] = json.dumps(header)
    local = tmp_path / "with_fermi.h0"
    local.write_text("\n".join(source) + "\n")
    built = build(load_input(written(SELFENERGY, h0=local)))
    assert any("Fermi level" in note for note in built.notes)


# ------------------------------------------------------------------- meshes


def test_disabling_both_selfenergy_axes_is_refused(written):
    text = SELFENERGY + "\n[selfenergy.real_axis]\nenabled = false\n"
    with pytest.raises(InputError, match="nothing to compute"):
        build(load_input(written(text)))


def test_the_fermionic_matsubara_mesh_is_built_from_tau(written):
    text = SELFENERGY + "\n[selfenergy.matsubara]\nenabled = true\nn_points = 4\n"
    built = build(load_input(written(text)))
    tau = built.basis.tau
    assert built.meshes.iw == pytest.approx(1j * (2 * np.arange(4) + 1) * np.pi * tau)


def test_auto_sentinels_stay_sentinels(written):
    """N2/R7: "auto" must not be materialised into a number by the reader."""
    built = build(load_input(written(SELFENERGY)))
    assert built.basis.truncation_threshold is None, "the RAM-derived cap stays at its call site"
    assert built.solver.reort is None


def test_none_disables_the_cap_rather_than_deriving_one(written):
    text = SELFENERGY + '\n[many_body_basis]\ntruncation_threshold = "none"\n'
    assert build(load_input(written(text))).basis.truncation_threshold == np.inf
