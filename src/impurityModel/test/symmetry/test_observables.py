import numpy as np

from impurityModel.ed.observables import (
    get_Lz_from_rho_spherical,
    get_occupations_from_rho_spherical,
)


def test_rho_spherical_functions():
    rho = np.eye(2)
    N, Ndn, Nup = get_occupations_from_rho_spherical(rho)
    assert N == 2
    assert Ndn == 1
    assert Nup == 1

    Lz = get_Lz_from_rho_spherical(rho, l=0)
    assert Lz == 0.0
