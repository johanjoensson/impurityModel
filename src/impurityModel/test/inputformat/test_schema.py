"""Schema-integrity tests for the TOML input format.

These are the tests that keep the *declaration* honest, independently of any file ever being
read: every key carries a value kind, every default is self-consistent, and no default is a
copied constant that can drift from its source.
"""

import numpy as np
import pytest

from impurityModel.ed import config, h0_format
from impurityModel.inputformat import schema
from impurityModel.inputformat.schema import UNSET, Kind


def _all_keys():
    for path, table in schema.TABLES.items():
        for key in table.keys:
            yield path, key


def test_every_key_declares_a_kind():
    """The rule that makes unit conversion safe: no key may be kind-less.

    ``[units].energy`` converts by kind, so a key without one either escapes conversion it
    needs or receives conversion it must not have -- exactly the rot the hand-maintained
    ``_ENERGY_FIELDS`` tuples suffered.
    """
    for path, key in _all_keys():
        assert isinstance(key.kind, Kind), f"[{path}].{key.name} has no value kind"


def test_every_key_is_documented():
    for path, key in _all_keys():
        assert key.doc.strip(), f"[{path}].{key.name} is undocumented"


def test_enum_keys_declare_choices_and_defaults_are_among_them():
    for path, key in _all_keys():
        if key.kind in (Kind.ENUM, Kind.AUTO_ENUM, Kind.AUTO_COUNT):
            assert key.choices, f"[{path}].{key.name} is an enum with no choices"
        if key.choices and key.default is not UNSET and isinstance(key.default, str):
            assert key.default in key.choices, f"[{path}].{key.name} default is not among its choices"


def test_units_energy_is_required_with_no_default():
    """D1: the one key that must never acquire a default.

    A default lets the TOML and argparse front-ends disagree by 13.6057x silently. Adding one
    later is possible; removing one is not.
    """
    key = next(k for k in schema.TABLES["units"].keys if k.name == "energy")
    assert key.default is UNSET
    assert key.required
    assert set(key.choices) == set(h0_format.ENERGY_UNITS)


def test_energy_unit_choices_track_the_conversion_table():
    """The schema must not offer a unit the converter cannot handle, or hide one it can."""
    for path, key in _all_keys():
        if key.name == "unit" or (path == "units" and key.name == "energy"):
            assert set(key.choices) <= set(h0_format.ENERGY_UNITS)


def test_no_default_duplicates_a_constant_defined_in_ed():
    """N3: a schema default must never be a copied literal of a constant that lives elsewhere.

    The first draft wrote ``slater_weight_min = 1.49011612e-8``, a *truncated* copy of
    ``sqrt(finfo.eps)``, and ``excitation_budget = 4``, a value whose own docstring calls it
    the tightest *measured* value and expects it to be re-measured. Both must be reached by
    omission instead, so the single source of truth stays single.
    """
    forbidden = {
        float(np.sqrt(np.finfo(float).eps)): "sqrt(finfo.eps) (model.BasisOptions.slater_weight_min)",
        h0_format.RY_TO_EV: "h0_format.RY_TO_EV",
    }
    for path, key in _all_keys():
        if isinstance(key.default, float):
            for value, source in forbidden.items():
                assert key.default != pytest.approx(value, rel=1e-6), (
                    f"[{path}].{key.name} duplicates {source}; reach it by omission instead"
                )
    budget = next(k for k in schema.TABLES["many_body_basis"].keys if k.name == "excitation_budget")
    assert budget.default == "auto", "excitation_budget must default to 'auto', not a frozen number"


def test_truncation_threshold_keeps_auto_and_none_distinct():
    """Two different meanings that the solver currently collapses to one value."""
    key = next(k for k in schema.TABLES["many_body_basis"].keys if k.name == "truncation_threshold")
    assert set(key.choices) == {"auto", "none"}
    assert key.default == "auto"


def test_tagged_unions_are_sub_tables_not_string_tags():
    """R10/U5: switching a string tag leaves the old arm's keys behind, valid and ignored."""
    for root, expected in (
        ("hamiltonian", schema.HAMILTONIAN_SOURCES),
        ("interaction", schema.INTERACTION_KINDS),
        ("double_counting", schema.DC_SCHEMES),
    ):
        variants = schema.variants_of(root)
        assert variants, f"{root} declares no variants"
        assert {v.split(".", 1)[1] for v in variants} == set(expected)
        assert root not in schema.TABLES or not any(
            k.name in ("source", "kind", "scheme", "type") for k in schema.TABLES[root].keys
        ), f"[{root}] still carries a string tag"


def test_no_calculation_type_string_exists():
    """U5: the calculation is selected by which table is present, not by a `type` string."""
    assert "calculation" not in schema.TABLES
    run_keys = {k.name for k in schema.TABLES["run"].keys}
    assert "type" not in run_keys and "calculation" not in run_keys
    for name in schema.CALCULATIONS:
        assert name in schema.TABLES


def test_per_calculation_overrides_preserve_todays_divergent_defaults():
    """R7: one flat default set cannot serve three drivers.

    ``occ_cutoff`` is 1e-6 on the spectroscopy path and the 1e-12 dataclass default
    elsewhere; sharing one value would change every self-energy run by six orders of
    magnitude. The overrides carry that divergence explicitly.
    """
    assert schema.TABLES["spectroscopy"].overrides["many_body_basis"]["occ_cutoff"] == 1e-6
    assert schema.TABLES["spectroscopy"].overrides["many_body_basis"]["dN"] == 2
    assert schema.TABLES["spectroscopy"].overrides["temperature"]["kelvin"] == 300.0
    for name in ("selfenergy", "susceptibility"):
        assert schema.TABLES[name].overrides["temperature"]["tau"] == 0.002
        assert "occ_cutoff" not in schema.TABLES[name].overrides.get("many_body_basis", {})


def test_shell_l_is_unrestricted_but_role_is_required():
    """The format must be able to say what the solver cannot yet do."""
    keys = {k.name: k for k in schema.TABLES["shell"].keys}
    assert keys["l"].choices is None, "l must not be restricted to a fixed set"
    assert keys["l"].minimum == 0
    assert keys["role"].required and set(keys["role"].choices) == {"core", "valence"}


def test_bath_counts_are_deduced_not_required():
    """Deduced from the .h0 header when possible; supplied by hand only when there is none."""
    keys = {k.name: k for k in schema.TABLES["shell"].keys}
    for name in ("n_bath", "n_valence_bath"):
        assert keys[name].default is UNSET
        assert keys[name].deduced_from, f"{name} must document where it is deduced from"
        assert not keys[name].required


def test_spectroscopy_shares_the_meshes_and_core_hole_broadening():
    """U1/U2: these are shared in the code, so sharing them here is not a convenience.

    ``delta`` is both the PES/XPS/XAS lineshape and RIXS's intermediate-state broadening, and
    NIXS is evaluated on RIXS's energy-loss mesh. Filing either under one technique would
    make disabling that technique change another one.
    """
    shared = {k.name for k in schema.TABLES["spectroscopy"].keys}
    assert {"w", "w_loss", "core_hole_broadening"} <= shared
    rixs = {k.name for k in schema.TABLES["spectroscopy.rixs"].keys}
    assert "w_loss" not in rixs and "core_hole_broadening" not in rixs
    assert "final_state_broadening" in rixs


def test_every_spectroscopy_technique_has_an_explicit_enabled_switch():
    """The directive: no broadening sign, no empty mesh, no absent file acting as a switch."""
    for technique in ("pes", "xps", "xas", "rixs", "nixs"):
        keys = {k.name: k for k in schema.TABLES[f"spectroscopy.{technique}"].keys}
        assert "enabled" in keys and keys["enabled"].kind is Kind.BOOL
        assert isinstance(keys["enabled"].default, bool)


def test_matsubara_tables_are_separate_per_statistics():
    """N4: fermionic and bosonic differ in statistics *and* convention; one key would lose nu=0."""
    assert "selfenergy.matsubara" in schema.TABLES
    assert "susceptibility.matsubara" in schema.TABLES
    assert schema.TABLES["susceptibility.matsubara"].keys[1].default == 64
    assert schema.TABLES["selfenergy.matsubara"].keys[1].default == 0


def test_dimensionless_keys_are_not_energies():
    """R8: the keys a blanket energy conversion would corrupt."""
    energy_cut = next(k for k in schema.TABLES["susceptibility"].keys if k.name == "energy_cut")
    assert energy_cut.kind is Kind.DIMENSIONLESS
    occupation = next(k for k in schema.TABLES["double_counting.fixed_occupation"].keys if k.name == "occupation")
    assert occupation.kind is Kind.DIMENSIONLESS
    q = next(k for k in schema.TABLES["spectroscopy.nixs"].keys if k.name == "q")
    assert q.kind is Kind.VECTOR_LIST


def test_environment_is_free_form_and_the_knob_registry_is_reachable():
    """[environment] validates names against the registry rather than declaring them twice."""
    assert schema.TABLES["environment"].keys == ()
    assert config.KNOBS, "the knob registry must be non-empty for [environment] to validate against"


def test_dump_renders_every_table():
    text = schema.dump()
    for path in schema.TABLES:
        assert f"`[{path}]`" in text or f"`[[{path}]]`" in text
