"""Spin-consistent ``impurity_symmetry_rotation``: a spin-polarised impurity keeps its (dn, up) pairing.

The fixture is the 10x10 Ni impurity block of the collinear AFM NiO archive (group ``NiUp 5``,
``AFM_NiO/impmod/8_BathStates_linked_chainGeometry/outdir-2026-11-09/outdir-3``) with its input rotation to
spherical harmonics. The two spins differ by up to 3e-3 in the crystal field, so diagonalising each
spin on its own gives dn and up different eigenvectors; the rotated ``S_+`` then is no permutation, the
pairing is lost and the layout was graded ``none``.
"""

import numpy as np

from impurityModel.ed.lie_algebra import tensors_to_operator
from impurityModel.ed.spin_pairs import _pairs_from_rotated_splus, spin_layout
from impurityModel.ed.symmetries import impurity_symmetry_rotation

#: Impurity block of ``H solver`` (real; the imaginary part is below 3e-4 and irrelevant here).
H_IMP = np.array(
    [
        [-0.1782320292, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.1761057327, 0.0001996244, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0001996244, -0.1753669281, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.1761057327, 0.0001996244, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0001996244, -0.1753669281, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -0.1785777743, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1763349405, 0.0001831449, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001831449, -0.178239286, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1763349405, 0.0001831449],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001831449, -0.178239286],
    ]
)
#: ``Rot to spherical`` of the archive (spherical -> input).
ROT_REAL = np.array(
    [
        [
            0.0000000000e00,
            0.0000000000e00,
            1.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
        ],
        [
            8.1443166002e-01,
            0.0000000000e00,
            0.0000000000e00,
            4.1030437878e-01,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
        ],
        [
            -4.1030437878e-01,
            0.0000000000e00,
            0.0000000000e00,
            8.1443166002e-01,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
        ],
        [
            0.0000000000e00,
            4.1030645610e-01,
            0.0000000000e00,
            0.0000000000e00,
            -8.1443166002e-01,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
        ],
        [
            0.0000000000e00,
            1.5823513666e-16,
            0.0000000000e00,
            0.0000000000e00,
            4.1030437878e-01,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
        ],
        [
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            1.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
        ],
        [
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            8.2222925968e-01,
            0.0000000000e00,
            0.0000000000e00,
            4.0245578838e-01,
            0.0000000000e00,
        ],
        [
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            -4.0245578838e-01,
            0.0000000000e00,
            0.0000000000e00,
            8.2222925968e-01,
            0.0000000000e00,
        ],
        [
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            4.0245295741e-01,
            0.0000000000e00,
            0.0000000000e00,
            -8.2222925968e-01,
        ],
        [
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            1.5975012473e-16,
            0.0000000000e00,
            0.0000000000e00,
            4.0245578838e-01,
        ],
    ]
)
ROT_IMAG = np.array(
    [
        [
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
        ],
        [
            -9.0420078093e-17,
            0.0000000000e00,
            0.0000000000e00,
            -4.1030645610e-01,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
        ],
        [
            -4.1030645610e-01,
            0.0000000000e00,
            0.0000000000e00,
            -9.0420078093e-17,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
        ],
        [
            0.0000000000e00,
            4.1030437878e-01,
            0.0000000000e00,
            0.0000000000e00,
            -1.1302509762e-16,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
        ],
        [
            0.0000000000e00,
            8.1443166002e-01,
            0.0000000000e00,
            0.0000000000e00,
            4.1030645610e-01,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
        ],
        [
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
        ],
        [
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            -9.1285785561e-17,
            0.0000000000e00,
            0.0000000000e00,
            -4.0245295741e-01,
            0.0000000000e00,
        ],
        [
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            -4.0245295741e-01,
            0.0000000000e00,
            0.0000000000e00,
            -9.1285785561e-17,
            0.0000000000e00,
        ],
        [
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            4.0245578838e-01,
            0.0000000000e00,
            0.0000000000e00,
            -1.1410723195e-16,
        ],
        [
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            0.0000000000e00,
            8.2222925968e-01,
            0.0000000000e00,
            0.0000000000e00,
            4.0245295741e-01,
        ],
    ]
)
ROT = ROT_REAL + 1j * ROT_IMAG
N = 10
EXPECTED_PAIRS = [(k, k + 5) for k in range(5)]


def _rotated(rot_to_spherical):
    w, u = impurity_symmetry_rotation({}, list(range(N)), n_orb=N, h0_matrix=H_IMP, rot_to_spherical=rot_to_spherical)
    return w, u, u.conj().T @ ROT


def test_independent_spin_rotation_loses_the_pairing():
    """Why the fix exists: with no rotation supplied the old per-component ``eigh`` breaks ``S_+``."""
    _, _, rot = _rotated(None)
    assert _pairs_from_rotated_splus(list(range(N)), rot) is None


def test_joint_spin_rotation_keeps_the_pairing():
    _, u, rot = _rotated(ROT)
    assert sorted(_pairs_from_rotated_splus(list(range(N)), rot)) == EXPECTED_PAIRS
    np.testing.assert_allclose(u.conj().T @ u, np.eye(N), atol=1e-12)
    # the same eigenvectors for both spins
    np.testing.assert_allclose(u[:5, :5], u[5:, 5:], atol=1e-14)
    np.testing.assert_allclose(u[:5, 5:], 0.0, atol=1e-14)


def test_the_spin_summed_block_is_diagonal_to_the_spin_splitting():
    _, u, _ = _rotated(ROT)
    h = u.conj().T @ H_IMP @ u
    # Only the dn/up splitting (3e-3) is left off the diagonal of the *spin-summed* basis; the
    # per-spin eigenvectors would remove it but lose the pairing.
    assert np.abs(h - np.diag(np.diag(h))).max() < 1e-3


def test_without_a_clean_pairing_the_old_rotation_is_used():
    bad = np.eye(N, dtype=complex)  # a spin-mixing rotation: S_+ is no permutation
    bad[0, 0] = bad[5, 5] = bad[0, 5] = 2**-0.5
    bad[5, 0] = -(2**-0.5)
    w0, u0 = impurity_symmetry_rotation({}, list(range(N)), n_orb=N, h0_matrix=H_IMP)
    w1, u1 = impurity_symmetry_rotation({}, list(range(N)), n_orb=N, h0_matrix=H_IMP, rot_to_spherical=bad)
    np.testing.assert_allclose(u0, u1)


def test_the_layout_is_collinear_after_the_joint_rotation():
    _, u, rot = _rotated(ROT)
    h_solver = u.conj().T @ H_IMP @ u
    op = tensors_to_operator(h_solver)
    layout = spin_layout(op, {0: [list(range(N))]}, ({}, {}), rot, N)
    assert sorted(layout.imp_pairs) == EXPECTED_PAIRS
    assert layout.grade in ("collinear", "su2")
    assert layout.labelled


def test_the_layout_names_the_rotation_when_it_is_not_clean():
    _, _, _ = _rotated(None)
    w, u = impurity_symmetry_rotation({}, list(range(N)), n_orb=N, h0_matrix=H_IMP)
    rot = u.conj().T @ ROT
    layout = spin_layout(tensors_to_operator(u.conj().T @ H_IMP @ u), {0: [list(range(N))]}, ({}, {}), rot, N)
    assert any(r.startswith("rotation:") for r in layout.rejected)
