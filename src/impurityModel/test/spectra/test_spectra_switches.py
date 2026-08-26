"""Per-technique on/off switches for the spectroscopy driver.

Before these existed there was no way to ask for one spectrum: PES and XPS were computed
unconditionally, RIXS was disabled by a non-positive broadening *or* an empty incoming mesh
(two switches, no stated precedence), and NIXS by withholding its radial data. A physics
parameter doubling as a feature flag is unguessable, so each technique now has a switch and
the data conditions became requirements rather than the way to say no.
"""

import inspect

import numpy as np
import pytest

from impurityModel.ed import spectra
from impurityModel.ed.get_spectra import resolve_spectra_switches
from impurityModel.ed.model import SpectraOptions


def test_the_default_options_reproduce_the_historical_behaviour():
    """The compatibility contract: an API caller that sets no switch sees no change."""
    switches, wIn = resolve_spectra_switches(SpectraOptions())
    assert switches == {"pes": True, "xps": True, "xas": True, "rixs": True, "nixs": False}
    assert len(wIn) == 50, "RIXS keeps its historical default incoming mesh"


def test_nixs_still_follows_the_radial_data_when_unset():
    radial = (np.linspace(0, 1, 4), np.ones(4), np.ones(4))
    switches, _ = resolve_spectra_switches(SpectraOptions(radial=radial))
    assert switches["nixs"] is True


def test_a_non_positive_rixs_broadening_still_disables_it_when_unset():
    """Historical inference preserved for callers that never learn about the switch."""
    switches, wIn = resolve_spectra_switches(SpectraOptions(deltaRIXS=0.0))
    assert switches["rixs"] is False
    assert len(wIn) == 0


@pytest.mark.parametrize("technique", ["pes", "xps", "xas", "rixs", "nixs"])
def test_each_technique_can_be_the_only_one(technique):
    """The capability that did not exist at all before: ask for exactly one spectrum."""
    off = {name: False for name in ("pes", "xps", "xas", "rixs", "nixs")}
    options = SpectraOptions(**{**off, technique: True}, radial=(np.linspace(0, 1, 4), np.ones(4), np.ones(4)))
    switches, _ = resolve_spectra_switches(options)
    assert switches[technique] is True
    assert not any(value for name, value in switches.items() if name != technique)


def test_disabling_everything_is_refused():
    with pytest.raises(ValueError, match="nothing to compute"):
        resolve_spectra_switches(SpectraOptions(pes=False, xps=False, xas=False, rixs=False, nixs=False))


def test_an_enabled_technique_demands_its_data_instead_of_silently_skipping():
    with pytest.raises(ValueError, match="no radial data"):
        resolve_spectra_switches(SpectraOptions(nixs=True))
    with pytest.raises(ValueError, match="broadening has to be a broadening"):
        resolve_spectra_switches(SpectraOptions(rixs=True, deltaRIXS=0.0))
    with pytest.raises(ValueError, match="incoming-energy mesh is empty"):
        resolve_spectra_switches(SpectraOptions(rixs=True, wIn=np.array([])))


def test_disabling_xas_does_not_touch_the_rixs_broadening():
    """U1: one `delta` is the XAS lineshape AND the RIXS intermediate-state broadening.

    They are shared in the code, so the switches must not be allowed to imply otherwise --
    turning XAS off changes nothing about RIXS.
    """
    with_xas, _ = resolve_spectra_switches(SpectraOptions(rixs=True))
    without_xas, _ = resolve_spectra_switches(SpectraOptions(rixs=True, xas=False))
    assert with_xas["rixs"] == without_xas["rixs"] is True


def test_simulate_spectra_accepts_a_switch_per_technique():
    """The gating actually reaches the worker, not just the driver."""
    parameters = inspect.signature(spectra.simulate_spectra).parameters
    for technique in ("pes", "xps", "xas", "rixs", "nixs"):
        assert technique in parameters
        assert parameters[technique].default is True


def test_the_switches_are_declared_on_the_option_group():
    fields = SpectraOptions.__dataclass_fields__
    for technique in ("pes", "xps", "xas", "rixs", "nixs"):
        assert fields[technique].default is None, "None must mean 'infer', so old callers are unaffected"
