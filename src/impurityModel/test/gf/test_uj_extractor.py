"""Tests for the double-counting U/J glue: ``_model_u4_dense`` and ``_model_uj``.

These wire :func:`impurityModel.ed.atomic_physics.uj_from_u4` to an ``ImpurityModel``: recover
the dense impurity Coulomb tensor from ``model.u4`` (an operator dict), rotate it into the
spherical basis with ``model.rot_to_spherical``, and read off the average Coulomb repulsion and
exchange. The FLL/AMF double-counting schemes (C6) derive their U, J from this.
"""

import numpy as np
import pytest

from impurityModel.ed.atomic_physics import get_spherical_2_cubic_matrix
from impurityModel.ed.dc_static import _model_u4_dense, _model_uj
from impurityModel.ed.lie_algebra import rotate_two_body
from impurityModel.ed.model import ImpurityModel, atomic_u4

F0, F2, F4 = 7.5, 9.9, 6.6
L = 2
N_ORB = 2 * L + 1
N_IMP = 2 * N_ORB


def _build_model(u4, rot_to_spherical):
    h0 = np.zeros((N_IMP + 2, N_IMP + 2), dtype=complex)
    return ImpurityModel.from_solver_matrix(
        h0,
        N_IMP,
        dc=None,
        u4=u4,
        rot_to_spherical=rot_to_spherical,
        bath_valence_conduction=([N_IMP, N_IMP + 1], []),
    )


def test_model_u4_dense_round_trips_through_operator_dict():
    u4 = atomic_u4(L, [F0, 0, F2, 0, F4])
    model = _build_model(u4, np.eye(N_IMP, dtype=complex))
    assert np.allclose(_model_u4_dense(model), u4, atol=1e-10)


def test_model_uj_identity_rotation_matches_slater_condon_average():
    u4 = atomic_u4(L, [F0, 0, F2, 0, F4])
    model = _build_model(u4, np.eye(N_IMP, dtype=complex))
    U, J = _model_uj(model)
    assert np.isclose(U, F0, atol=1e-9)
    assert np.isclose(J, (F2 + F4) / 14, atol=1e-9)


def test_model_uj_pins_the_rotation_direction():
    """model.u4 in a non-spherical (cubic) input basis, rot_to_spherical undoing it.

    ``get_spherical_2_cubic_matrix`` is the codebase's own tested spherical -> cubic
    transform. Storing model.u4 in that cubic basis and setting rot_to_spherical to its
    inverse (cubic -> spherical) must recover the same U, J as the identity-rotation case
    -- this pins the einsum direction :func:`_model_uj` uses.
    """
    u4_spherical = atomic_u4(L, [F0, 0, F2, 0, F4])
    u_cubic = get_spherical_2_cubic_matrix(spinpol=True, l=L)
    u4_cubic = rotate_two_body(u4_spherical, u_cubic)

    model = _build_model(u4_cubic, u_cubic.conj().T)
    U, J = _model_uj(model)
    assert np.isclose(U, F0, atol=1e-9)
    assert np.isclose(J, (F2 + F4) / 14, atol=1e-9)


def test_model_uj_wrong_rotation_direction_disagrees():
    """Sanity check that the previous test is actually sensitive to the rotation direction."""
    u4_spherical = atomic_u4(L, [F0, 0, F2, 0, F4])
    u_cubic = get_spherical_2_cubic_matrix(spinpol=True, l=L)
    u4_cubic = rotate_two_body(u4_spherical, u_cubic)

    model = _build_model(u4_cubic, u_cubic)  # wrong direction
    _, J = _model_uj(model)
    assert not np.isclose(J, (F2 + F4) / 14, atol=1e-6)


def test_model_uj_rejects_multigroup_rot_to_spherical():
    u4 = atomic_u4(L, [F0, 0, F2, 0, F4])
    model = _build_model(u4, np.eye(N_IMP, dtype=complex))
    model = ImpurityModel(
        h0=model.h0,
        dc=model.dc,
        u4=model.u4,
        impurity_orbitals=model.impurity_orbitals,
        rot_to_spherical={0: np.eye(N_IMP, dtype=complex)},
        bath_states=model.bath_states,
        n_spin_orbitals=model.n_spin_orbitals,
    )
    with pytest.raises(ValueError, match="multi-group"):
        _model_uj(model)


def test_model_u4_dense_raises_without_u4():
    u4 = atomic_u4(L, [F0, 0, F2, 0, F4])
    model = _build_model(u4, np.eye(N_IMP, dtype=complex))
    model = ImpurityModel(
        h0=model.h0,
        dc=model.dc,
        u4=None,
        impurity_orbitals=model.impurity_orbitals,
        rot_to_spherical=model.rot_to_spherical,
        bath_states=model.bath_states,
        n_spin_orbitals=model.n_spin_orbitals,
    )
    with pytest.raises(ValueError, match="u4"):
        _model_u4_dense(model)
