"""
Tests and benchmark for the optimized density_matrix module.

Verifies that calc_density_matrix returns the same result as the
reference implementation using the full c_j^dag c_i operator construction.
"""

import pytest
import numpy as np
from itertools import product as iproduct
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, SlaterDeterminant, inner
from impurityModel.ed.density_matrix import calc_density_matrix, calc_density_matrices


# ---------------------------------------------------------------------------
# Reference (original) implementation for comparison
# ---------------------------------------------------------------------------

def _reference_calc_density_matrix(psi: ManyBodyState, orbital_indices: list):
    """Original O(n_orb^2) implementation."""
    n_orb = len(orbital_indices)
    rho = np.zeros((n_orb, n_orb), dtype=complex)
    for (i, orb_i), (j, orb_j) in iproduct(enumerate(orbital_indices), repeat=2):
        op = ManyBodyOperator({((orb_j, "c"), (orb_i, "a")): 1.0})
        psi_p = op(psi, cutoff=0)
        amp = inner(psi, psi_p)
        if np.abs(amp) > 0 * np.finfo(float).eps:
            rho[i, j] = amp
    return rho


# ---------------------------------------------------------------------------
# Helper: build a simple Slater-determinant state
# ---------------------------------------------------------------------------

def _make_state(occupied_orbs: list[int], n_orbs: int = 16) -> ManyBodyState:
    """Return a single-Slater-determinant ManyBodyState with given occupied orbitals.

    Orbital k occupies bit (7 - k%8) of byte (k//8), i.e. MSB-first within
    each byte.  This matches how ManyBodyOperator addresses orbitals.
    """
    n_bytes = (n_orbs + 7) // 8
    data = bytearray(n_bytes)
    for orb in occupied_orbs:
        byte_idx = orb // 8
        bit_pos = 7 - (orb % 8)  # MSB-first within each byte
        data[byte_idx] |= (1 << bit_pos)
    sd = SlaterDeterminant.from_bytes(bytes(data))
    return ManyBodyState({sd: 1.0})


def _make_superposition(terms: list[tuple[list[int], complex]], n_orbs: int = 16) -> ManyBodyState:
    """Build a normalised superposition of Slater determinants."""
    n_bytes = (n_orbs + 7) // 8
    d = {}
    for occupied, amp in terms:
        data = bytearray(n_bytes)
        for orb in occupied:
            byte_idx = orb // 8
            bit_pos = 7 - (orb % 8)  # MSB-first within each byte
            data[byte_idx] |= (1 << bit_pos)
        sd = SlaterDeterminant.from_bytes(bytes(data))
        d[sd] = amp
    psi = ManyBodyState(d)
    norm = psi.norm()
    return psi / norm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCalcDensityMatrix:
    def test_single_slater_determinant(self):
        """For a single SD |0,1,2> the density matrix is diagonal with ones
        at the occupied orbitals and zeros elsewhere."""
        occupied = [0, 1, 2]
        psi = _make_state(occupied)
        orbital_indices = [0, 1, 2, 3]
        rho = calc_density_matrix(psi, orbital_indices)

        assert rho.shape == (4, 4)
        np.testing.assert_allclose(np.diag(rho).real, [1, 1, 1, 0], atol=1e-12)
        # Off-diagonals should vanish for a single SD
        mask = ~np.eye(4, dtype=bool)
        np.testing.assert_allclose(rho[mask], 0, atol=1e-12)

    def test_hermitian(self):
        """Density matrix must be Hermitian for any state."""
        psi = _make_superposition([
            ([0, 2], 1.0 + 0j),
            ([1, 3], 0.5 + 0.5j),
            ([0, 3], -0.3 + 0.1j),
        ])
        orbital_indices = [0, 1, 2, 3]
        rho = calc_density_matrix(psi, orbital_indices)
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)

    def test_trace_equals_particle_number(self):
        """Tr(rho) should equal the expected particle number."""
        # Two particles in a superposition of 4-orbital states
        psi = _make_superposition([
            ([0, 1], 1.0),
            ([2, 3], 1.0j),
        ])
        orbital_indices = [0, 1, 2, 3]
        rho = calc_density_matrix(psi, orbital_indices)
        # Each term has 2 particles -> Tr(rho) = 2
        np.testing.assert_allclose(rho.trace().real, 2.0, atol=1e-12)

    def test_agrees_with_reference_single_sd(self):
        """Optimized implementation matches reference for a single SD."""
        psi = _make_state([0, 3, 5])
        orbital_indices = list(range(8))
        rho = calc_density_matrix(psi, orbital_indices)
        rho_ref = _reference_calc_density_matrix(psi, orbital_indices)
        np.testing.assert_allclose(rho, rho_ref, atol=1e-12)

    def test_agrees_with_reference_superposition(self):
        """Optimized implementation matches reference for a non-trivial superposition."""
        psi = _make_superposition([
            ([0, 2, 4], 1.0 + 0j),
            ([1, 3, 5], 0.8 + 0.3j),
            ([0, 1, 4], -0.5 + 0.2j),
            ([2, 3, 4], 0.1 - 0.7j),
        ])
        orbital_indices = list(range(6))
        rho = calc_density_matrix(psi, orbital_indices)
        rho_ref = _reference_calc_density_matrix(psi, orbital_indices)
        np.testing.assert_allclose(rho, rho_ref, atol=1e-12)

    def test_empty_state(self):
        """Requesting an empty orbital_indices list returns an empty matrix."""
        psi = _make_state([0, 1])
        rho = calc_density_matrix(psi, [])
        assert rho.shape == (0, 0)

    def test_single_orbital(self):
        """Single-orbital subspace returns a 1x1 matrix."""
        psi = _make_state([0, 1])
        rho = calc_density_matrix(psi, [0])
        assert rho.shape == (1, 1)
        np.testing.assert_allclose(rho[0, 0].real, 1.0, atol=1e-12)

    def test_subset_of_orbitals(self):
        """Can request a strict subset of the total orbital space."""
        psi = _make_superposition([
            ([0, 2], 1.0),
            ([1, 3], 1.0j),
        ])
        orbital_indices = [0, 1]   # only first two orbitals
        rho = calc_density_matrix(psi, orbital_indices)
        rho_ref = _reference_calc_density_matrix(psi, orbital_indices)
        np.testing.assert_allclose(rho, rho_ref, atol=1e-12)


class TestCalcDensityMatrices:
    def test_batch_matches_individual(self):
        """calc_density_matrices should give same result as calling calc_density_matrix
        for each state individually."""
        psis = [
            _make_state([0, 1]),
            _make_superposition([([0, 2], 1.0), ([1, 3], 1j)]),
            _make_superposition([([0, 1, 2], 1.0), ([1, 2, 3], -1j)]),
        ]
        orbital_indices = list(range(4))
        rhos = calc_density_matrices(psis, orbital_indices)
        for i, psi in enumerate(psis):
            rho_i = calc_density_matrix(psi, orbital_indices)
            np.testing.assert_allclose(rhos[i], rho_i, atol=1e-12)

    def test_output_shape(self):
        """Output has the expected shape (n_psi, n_orb, n_orb)."""
        psis = [_make_state([k]) for k in range(4)]
        orbital_indices = list(range(6))
        rhos = calc_density_matrices(psis, orbital_indices)
        assert rhos.shape == (4, 6, 6)
