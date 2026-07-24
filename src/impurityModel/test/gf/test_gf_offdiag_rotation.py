"""Direct correctness test for rotate_Greens_function (greens_function.py:1247).

The function is a pure array identity, G'(z) = T^H G(z) T, with no basis/hOp/MPI
involvement of its own -- but it stands in for the physical operation "rotate the
correlated orbitals by T" (spherical<->real harmonics, symmetry block-diagonalization,
...), which *does* touch the many-body machinery: replacing the impurity annihilation
operators c_a by rotated combinations sum_b T[b,a] c_b is exactly the same as replacing
the resolvent seeds c_a^dagger|psi> by sum_b T[b,a] (c_b^dagger|psi>). This test checks
that identity directly: G computed from seeds rotated by T must equal G computed from
the original seeds and then rotated by rotate_Greens_function, for a genuinely
non-diagonal, complex (not just real-orthogonal) unitary T.
"""

import numpy as np

from impurityModel.ed.greens_function import rotate_Greens_function
from impurityModel.ed.ManyBodyUtils import ManyBodyState

from impurityModel.test.support.gf_oracles import DELTA, MATSUBARA, OMEGA, _dense_G_on, _n3_sector_dets, _seeds


def _complex_unitary_2x2(theta, phi):
    """A genuinely complex, non-diagonal 2x2 unitary (not just a real rotation)."""
    c, s = np.cos(theta), np.sin(theta)
    T = np.array(
        [
            [c, -s * np.exp(1j * phi)],
            [s * np.exp(-1j * phi), c],
        ],
        dtype=complex,
    )
    np.testing.assert_allclose(T.conj().T @ T, np.eye(2), atol=1e-12)  # sanity: T is unitary
    return T


def _rotated_seeds(T):
    seeds = _seeds()
    rotated = []
    for a in range(T.shape[1]):
        combo = {}
        for b, seed in enumerate(seeds):
            coeff = T[b, a]
            for det, amp in seed.items():
                combo[det] = combo.get(det, 0j) + coeff * amp[0]
        rotated.append(ManyBodyState({det: amp for det, amp in combo.items() if amp != 0}))
    return rotated


def test_rotate_greens_function_matches_seeds_rotated_oracle():
    dets = _n3_sector_dets()
    T = _complex_unitary_2x2(0.37, 1.1)

    z_values = np.concatenate([MATSUBARA, OMEGA + 1j * DELTA])
    G = _dense_G_on(dets, z_values)
    G_rotated_direct = _dense_G_on(dets, z_values, seeds=_rotated_seeds(T))

    G_via_rotate_fn = rotate_Greens_function(G, T)

    np.testing.assert_allclose(G_via_rotate_fn, G_rotated_direct, atol=1e-10)


def test_rotate_greens_function_identity_is_a_no_op():
    dets = _n3_sector_dets()
    G = _dense_G_on(dets, OMEGA + 1j * DELTA)
    np.testing.assert_allclose(rotate_Greens_function(G, np.eye(2, dtype=complex)), G, atol=1e-12)


def test_rotate_greens_function_would_catch_a_conjugate_order_bug():
    """Sanity check that the main test is actually sensitive to getting T vs T^H the
    right way around -- guards against a vacuously-passing comparison. The correct
    identity is G' = T^H G T (both references above use it); the swapped-order
    T G T^H must disagree for a genuinely non-Hermitian, non-diagonal T."""
    dets = _n3_sector_dets()
    T = _complex_unitary_2x2(0.37, 1.1)
    assert not np.allclose(T, T.conj().T), "T must be non-Hermitian for this check to be meaningful"

    z_values = OMEGA + 1j * DELTA
    G = _dense_G_on(dets, z_values)
    G_correct = rotate_Greens_function(G, T)  # T^H G T
    G_wrong_order = T[np.newaxis, :, :] @ G @ np.conj(T.T)[np.newaxis, :, :]  # T G T^H

    assert not np.allclose(G_correct, G_wrong_order, atol=1e-8)
