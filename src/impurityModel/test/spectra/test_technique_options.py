"""The per-technique option classes: each owns its mesh and broadening and validates itself."""

import numpy as np
import pytest

from impurityModel.ed.options import (
    NIXSOptions,
    PESOptions,
    RIXSOptions,
    SpectraOptions,
    XASOptions,
    XPSOptions,
)
from impurityModel.ed.spectroscopy import requests_from_legacy_options

RADIAL = (np.linspace(0, 1, 4), np.ones(4), np.ones(4))


def test_defaults_are_the_historical_grids():
    assert len(PESOptions().w) == 3001 and PESOptions().w[0] == -25
    assert len(RIXSOptions().w_in) == 50
    assert len(RIXSOptions().w_loss) == 4000
    assert XASOptions().delta == 0.2 and RIXSOptions().delta_final == 0.05


@pytest.mark.parametrize("cls", [PESOptions, XPSOptions, XASOptions])
def test_a_non_positive_broadening_is_refused(cls):
    with pytest.raises(ValueError, match="positive finite broadening"):
        cls(delta=0.0)
    with pytest.raises(ValueError, match="positive finite broadening"):
        cls(delta=float("nan"))


@pytest.mark.parametrize("cls", [PESOptions, XPSOptions, XASOptions])
def test_an_empty_mesh_is_refused(cls):
    with pytest.raises(ValueError, match="empty"):
        cls(w=np.array([]))


def test_rixs_refuses_empty_meshes_and_non_positive_broadenings():
    with pytest.raises(ValueError, match="incoming-energy mesh"):
        RIXSOptions(w_in=np.array([]))
    with pytest.raises(ValueError, match="delta_final"):
        RIXSOptions(delta_final=0.0)
    with pytest.raises(ValueError, match="delta_intermediate"):
        RIXSOptions(delta_intermediate=-1.0)


def test_nixs_demands_its_radial_data():
    with pytest.raises(ValueError, match="radial"):
        NIXSOptions(radial=None)
    assert len(NIXSOptions(radial=RADIAL).resolved_qs) == 2


def test_a_dipole_must_be_three_square_matrices():
    XASOptions(dipole=np.zeros((3, 4, 4)))
    with pytest.raises(ValueError, match="shape"):
        XASOptions(dipole=np.zeros((2, 4, 4)))
    with pytest.raises(ValueError, match="shape"):
        RIXSOptions(dipole=np.zeros((3, 4, 5)))


def test_the_legacy_bag_converts_to_requests_in_execution_order():
    requests = requests_from_legacy_options(SpectraOptions(radial=RADIAL, delta=0.3, deltaRIXS=0.07))
    assert [type(r) for r in requests] == [PESOptions, XPSOptions, NIXSOptions, XASOptions, RIXSOptions]
    rixs = requests[-1]
    assert rixs.delta_intermediate == 0.3 and rixs.delta_final == 0.07
    assert requests[2].delta == SpectraOptions().deltaNIXS


def test_the_legacy_bag_honours_its_switches():
    requests = requests_from_legacy_options(SpectraOptions(pes=False, xps=False, rixs=False))
    assert [type(r) for r in requests] == [XASOptions]
