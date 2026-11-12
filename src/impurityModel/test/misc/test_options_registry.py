"""The option registry: every field is registered once, and the derived grammar reads and shows it."""

import dataclasses

import pytest

from impurityModel.ed.options import (
    OPTION_SPECS,
    BasisOptions,
    SolverOptions,
    describe_options,
    options_from_attrs,
    options_to_attrs,
    parse_option_tokens,
)

#: Fields the caller supplies; they are not options a user types.
_NOT_OPTIONS = {"nominal_occ", "tau", "sigma_method"}


def test_every_option_field_is_registered_once():
    registered = [(s.group, s.name) for s in OPTION_SPECS]
    assert len(registered) == len(set(registered))
    for group, cls in (("basis", BasisOptions), ("solver", SolverOptions)):
        fields = {f.name for f in dataclasses.fields(cls)} - _NOT_OPTIONS
        assert fields == {name for g, name in registered if g == group}


def test_words_are_unique():
    words = [w for s in OPTION_SPECS for w in (*s.words, *s.flags)]
    assert len(words) == len(set(words))


def test_parse_values_flags_and_leftovers():
    basis, solver, unused = parse_option_tokens(
        ["Slater_Min", "1d-8", "dn", "3", "no_mv", "dense_green", "gf_settle", "off", "gf_admission", "auto"]
        + ["truncation_threshold", "2e6", "gamma", "0.1"]
    )
    assert basis == {"slater_weight_min": 1e-8, "dN": 3, "mixed_valence": False, "truncation_threshold": 2_000_000}
    assert solver == {"sparse_green": False, "gf_settle": False, "gf_admission": None}
    assert unused == ["gamma", "0.1"]


def test_aliases_and_missing_value():
    assert parse_option_tokens(["e_pt2_tol", "1e-9"])[0] == {"e_pt2_tol": 1e-9}
    with pytest.raises(ValueError, match="needs a value"):
        parse_option_tokens(["gf_tol"])
    with pytest.raises(ValueError, match="on or off"):
        parse_option_tokens(["gf_settle", "maybe"])
    with pytest.raises(ValueError, match="positive"):
        parse_option_tokens(["e_pt2", "0"])


def test_parsed_kwargs_build_the_dataclasses():
    basis, solver, _ = parse_option_tokens(["gf_prune_tol", "0", "gf_refine_rounds", "2", "cost_budget", "3"])
    assert SolverOptions(**solver).gf_refine_rounds == 2
    assert BasisOptions(nominal_occ={0: 8}, **basis).cost_budget == 3.0


def test_attrs_round_trip():
    basis = BasisOptions(nominal_occ={0: 8}, dN=2, slater_weight_min=1e-7, e_pt2_tol=1e-9, chain_restrict=False)
    solver = SolverOptions(gf_tol=1e-8, gf_settle=False, gf_prune_tol=0.0)
    attrs = options_to_attrs(basis, solver)
    assert attrs["slater_min"] == 1e-7
    basis_kwargs, solver_kwargs = options_from_attrs(attrs)
    assert BasisOptions(nominal_occ={0: 8}, **basis_kwargs) == basis
    assert SolverOptions(**solver_kwargs) == solver


def test_describe_uses_the_words_of_the_grammar():
    rows = dict(describe_options(BasisOptions(nominal_occ={0: 8}), SolverOptions()))
    assert rows["GF support settle"] == "on"
    assert rows["Determinant cap (truncation_threshold)"] == "auto"
    rows = dict(
        describe_options(
            BasisOptions(nominal_occ={0: 8}, truncation_threshold=float("inf")), SolverOptions(gf_refine_rounds=0)
        )
    )
    assert rows["Determinant cap (truncation_threshold)"] == "unlimited"
    assert rows["GF swap refinement rounds"] == "off"


def test_every_solver_option_is_a_toml_key():
    from impurityModel.inputformat.schema import TABLES

    keys = {key.name for key in TABLES["solver"].keys}
    assert {s.name for s in OPTION_SPECS if s.group == "solver"} <= keys


def test_every_flag_is_a_selfenergy_argument():
    import argparse

    from impurityModel.scripts import selfenergy

    parser = argparse.ArgumentParser()
    selfenergy.add_arguments(parser)
    flags = {flag for action in parser._actions for flag in action.option_strings}
    assert {s.cli for s in OPTION_SPECS if s.cli} <= flags


@pytest.mark.parametrize("word", ["dense_cutoff", "dn", "gf_refine_rounds"])
@pytest.mark.parametrize("text", ["inf", "nan", "1.5"])
def test_an_integer_option_refuses_a_non_integer_with_a_value_error(word, text):
    with pytest.raises(ValueError):
        parse_option_tokens([word, text])


def test_a_mixed_valence_dict_is_refused_by_the_archive_record():
    with pytest.raises(ValueError, match="mixed_valence"):
        options_to_attrs(BasisOptions(nominal_occ={2: 8}, mixed_valence={2: 1}), SolverOptions())
