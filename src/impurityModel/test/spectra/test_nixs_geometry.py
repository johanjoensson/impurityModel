"""Momentum-transfer geometry of the NIXS transition operator.

The azimuth used to be ``arccos(q_x / (|q| sin(theta)))``, which fails in two ways. Along z
the polar angle is zero, so the expression is 0/0 and every matrix element comes back NaN --
silently, producing an all-NaN spectrum with no exception raised anywhere. Away from the pole
it is still wrong for ``q_y < 0``, because ``arccos`` returns [0, pi] and therefore cannot
represent the lower half-plane at all: +q_y and -q_y came out identical.
"""

from collections import OrderedDict

import numpy as np
import pytest

from impurityModel.ed.transition_operators import nixs_operator


@pytest.fixture
def radial():
    r = np.linspace(0.01, 8.0, 300)
    return r, np.exp(-r / 2)


def _operator(radial, q):
    r, R = radial
    return nixs_operator(OrderedDict([(2, 0)]), np.asarray(q, dtype=float), 2, 2, R, R, r)


@pytest.mark.parametrize("q", [[0, 0, 4.0], [0, 0, -4.0], [0, 0, 1e-9]])
def test_a_momentum_transfer_along_z_is_finite(radial, q):
    """The pole. Previously every element was NaN, and nothing raised."""
    values = np.array(list(_operator(radial, q).values()))
    assert values.size
    assert np.isfinite(values).all()


def test_at_the_pole_only_the_m_diagonal_survives(radial):
    """Physics, not just finiteness: at theta = 0 the azimuth is arbitrary, and only the
    m-conserving elements may be non-zero -- so the answer must not depend on the arbitrary
    choice of phi made there."""
    op = _operator(radial, [0, 0, 4.0])
    for created, annihilated in op:
        # Spin-orbital indices within one l=2 shell, ordered (l, s, m) by c2i: same m means
        # the same offset within a spin block.
        assert (created[0] - annihilated[0]) % 5 == 0


def test_the_lower_half_plane_is_not_the_upper_one(radial):
    """+q_y and -q_y were identical under arccos; they are physically distinct."""
    up = _operator(radial, [1.0, 1.0, 0.0])
    down = _operator(radial, [1.0, -1.0, 0.0])
    assert max(abs(up[k] - down[k]) for k in up) > 1e-3


def test_reflecting_q_y_conjugates_the_operator(radial):
    """The precise statement of the above: y -> -y sends phi -> -phi, so every element
    picks up a complex conjugate. This is what pins the azimuth to arctan2 rather than
    merely 'something that is finite'."""
    up = _operator(radial, [1.0, 1.0, 0.0])
    down = _operator(radial, [1.0, -1.0, 0.0])
    assert set(up) == set(down)
    for key in up:
        assert up[key] == pytest.approx(np.conj(down[key]), abs=1e-12)


def test_rotating_q_about_z_only_changes_phases(radial):
    """|t| is invariant under a rotation of q about z; only the azimuthal phases move."""
    reference = _operator(radial, [2.0, 0.0, 3.0])
    for angle in (0.4, 1.9, -2.7):
        rotated = _operator(radial, [2.0 * np.cos(angle), 2.0 * np.sin(angle), 3.0])
        assert set(rotated) == set(reference)
        for key in reference:
            assert abs(rotated[key]) == pytest.approx(abs(reference[key]), abs=1e-12)


def test_a_zero_momentum_transfer_is_refused(radial):
    """|q| = 0 has no scattering direction; it used to yield NaN via 0/0 in the polar angle."""
    with pytest.raises(ValueError, match="non-zero momentum transfer"):
        _operator(radial, [0.0, 0.0, 0.0])
