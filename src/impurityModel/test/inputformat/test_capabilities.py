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


@pytest.mark.parametrize("core_l, valence_l", [(0, 1), (1, 2), (2, 3), (3, 4)])
def test_every_dipole_allowed_edge_is_accepted(core_l, valence_l):
    """The restriction this table used to carry: spectroscopy was pinned to core_l=1, l_v=2.

    An M4,5 (3d -> 4f) request is perfectly sensible physics, and the solver now assembles it;
    the table must not keep refusing it.
    """
    capabilities.check("spectroscopy", core_l=core_l, valence_l=valence_l, techniques=("xas", "rixs"))


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
    ``atomic_u4`` and ``_add_soc_and_field``, none of which pins it.
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
    """A blocker, if there is one, must say where it lives -- the message is the to-do list.

    The table is empty today. This guards the shape of any row that gets added back, and
    ``test_the_blocker_list_is_empty`` guards the fact that there are none.
    """
    for text, where in capabilities._BLOCKERS:
        assert text.strip() and where.strip()
        assert "/" in where, f"{where!r} should point at a file"


def test_the_blocker_list_is_empty():
    """No angular-momentum assumption narrows the spectroscopy path any more."""
    assert capabilities._BLOCKERS == ()


def test_a_core_shell_is_still_refused_on_a_single_shell_calculation_by_shape():
    """``core_l=None`` in the table means "no core shell", not "any" -- ANY means any."""
    assert capabilities.SUPPORTED[0].core_l is capabilities.ANY
    for row in capabilities.SUPPORTED[1:]:
        assert row.core_l is None
