"""The gate between "this file is valid" and "this run is possible".

The point of these tests is the *future*: when the 2p/3d restriction is lifted, the first
test here flips from "errors clearly" to "runs" by deleting a row from the support table,
and nothing about the input format changes.
"""

import pytest

from impurityModel.inputformat import capabilities
from impurityModel.inputformat.capabilities import InvalidShellCombination, UnsupportedCalculation


def test_the_supported_case_passes():
    capabilities.check("spectroscopy", core_l=1, valence_l=2, techniques=("pes", "xps", "xas", "rixs", "nixs"))


def test_an_unsupported_edge_is_refused_as_not_yet_not_as_invalid():
    """The distinction the user has to be able to act on.

    An M4,5 (3d -> 4f) request is perfectly sensible physics; only this codebase cannot do it
    yet. The message must say so, and must name what would have to change -- otherwise the
    user cannot tell a missing feature from their own mistake.
    """
    with pytest.raises(UnsupportedCalculation) as excinfo:
        capabilities.check("spectroscopy", core_l=2, valence_l=3, techniques=("xas",))
    message = str(excinfo.value)
    assert "M4,5" in message
    assert "The input file is valid" in message
    assert "L2,3" in message, "the message must say what IS supported"
    assert "ed/model.py:461" in message, "the message must name the blocking assumptions"


def test_a_selection_rule_violation_is_invalid_input_not_a_missing_feature():
    """Zero at any level of generality, so no generalisation would ever make it work."""
    with pytest.raises(InvalidShellCombination, match="Gaunt selection rule"):
        capabilities.check("spectroscopy", core_l=2, valence_l=2, techniques=("xas",))


def test_a_core_level_technique_without_a_core_shell_is_invalid_input():
    with pytest.raises(InvalidShellCombination, match="needs a core hole"):
        capabilities.check("spectroscopy", core_l=None, valence_l=2, techniques=("xas",))


def test_techniques_that_need_no_core_hole_do_not_demand_one():
    """PES and NIXS are valence probes; requiring a core shell for them would be wrong."""
    capabilities.check("spectroscopy", core_l=1, valence_l=2, techniques=("pes", "nixs"))


@pytest.mark.parametrize("valence_l", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("calculation", ["selfenergy", "susceptibility"])
def test_selfenergy_and_susceptibility_are_already_l_general(calculation, valence_l):
    """Not pessimism: these paths really are general.

    ``ImpurityModel.from_h0_file`` passes ``l`` straight through to ``getSOCop``,
    ``atomic_u4`` and ``_add_soc_and_field``, none of which pins it. Only the spectroscopy
    path goes through the 2p/3d-specific ``get_hamiltonian_operator``.
    """
    capabilities.check(calculation, core_l=None, valence_l=valence_l)


def test_a_core_shell_on_a_single_shell_calculation_is_refused():
    with pytest.raises(UnsupportedCalculation):
        capabilities.check("selfenergy", core_l=1, valence_l=2)


def test_edge_names_cover_the_common_edges():
    assert capabilities.edge_name(1, 2).startswith("L2,3")
    assert capabilities.edge_name(0, 1).startswith("K")
    assert capabilities.edge_name(2, 3).startswith("M4,5")
    assert "no core-level transition" in capabilities.edge_name(None, 2)


def test_every_blocker_names_a_location():
    """The error message doubles as the to-do list for lifting the restriction."""
    assert capabilities._BLOCKERS
    for text, where in capabilities._BLOCKERS:
        assert text.strip() and where.strip()
        assert "/" in where, f"{where!r} should point at a file"
