import numpy as np

from impurityModel.ed.atomic_physics import dc_MLFT, get_spherical_2_cubic_matrix


def test_dc_MLFT():
    # Only 3d
    res = dc_MLFT(5, 1.0, Fdd=[5.0, 0, 1.0, 0, 1.0])
    assert 2 in res
    assert np.isclose(res[2], (5.0 - 14.0 / 441 * 2.0) * 5 - 1.0)

    # with 2p
    res = dc_MLFT(5, 1.0, Fdd=[5.0, 0, 1.0, 0, 1.0], n2p_i=6, Fpd=[4.0, 0, 0], Gpd=[0, 1.0, 0, 1.0])
    assert 1 in res
    assert 2 in res


def test_get_spherical_2_cubic_matrix():
    u_p = get_spherical_2_cubic_matrix(l=1)
    assert u_p.shape == (3, 3)

    u_d = get_spherical_2_cubic_matrix(l=2)
    assert u_d.shape == (5, 5)
