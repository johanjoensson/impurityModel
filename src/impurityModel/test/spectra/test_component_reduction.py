"""Unit tests for :func:`symmetries.component_symmetry_reduction` (B2b component dedup).

The dipole transition tensor is invariant under continuous single-particle symmetries of
``h``; the detector represents those symmetries on the 3 Cartesian dipole components and
reduces the tensor accordingly:

* rotationally-invariant (spherical) ``h`` -> all three components collapse to one
  representative (``diagonalizable`` and a single group);
* axial ``h`` (only ``L_z`` continuous) -> the tensor is diagonalised in a symmetry-adapted
  ``Q`` basis but no component is dropped (equating the two in-plane components needs a
  discrete reflection / time reversal, invisible to ``[h, G] = 0``);
* generic ``h`` -> identity reduction, full tensor.

Soundness: whenever ``diagonalizable`` is True, ``chi = Q diag(chi') Q^dagger`` must
reproduce the full tensor for *any* symmetry-respecting ``chi`` in the commutant.
"""

from collections import OrderedDict

import numpy as np
import pytest

from impurityModel.ed import lie_algebra
from impurityModel.ed.operator_algebra import c2i
from impurityModel.ed.spectra import dipole_operators
from impurityModel.ed.symmetries import (
    component_symmetry_reduction,
    extract_tensors,
)

nBaths = OrderedDict({1: 0, 2: 0})  # core p (l=1) + d (l=2), no baths
N_ORB = 2 * (3 + 5)
COMPONENTS = dipole_operators(nBaths, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 1, 2)
CTENSORS = [extract_tensors(op, n_orb=N_ORB)[0] for op in COMPONENTS]


def _diag_h(energy_p, energy_d):
    h = np.zeros((N_ORB, N_ORB), dtype=complex)
    for s in range(2):
        for m in range(-1, 2):
            i = c2i(nBaths, (1, s, m))
            h[i, i] = energy_p(m)
        for m in range(-2, 3):
            i = c2i(nBaths, (2, s, m))
            h[i, i] = 5.0 + energy_d(m)
    return h


def _n_groups(reduction):
    return len(reduction.representatives)


def test_spherical_collapses_to_single_representative():
    h = _diag_h(lambda m: 0.0, lambda m: 0.0)
    r = component_symmetry_reduction(COMPONENTS, h, n_orb=N_ORB)
    assert r.diagonalizable
    assert _n_groups(r) == 1
    assert r.group_of_column == [0, 0, 0]


def test_axial_diagonalises_but_keeps_all_components():
    h = _diag_h(lambda m: 0.2 * abs(m), lambda m: 0.3 * abs(m))
    r = component_symmetry_reduction(COMPONENTS, h, n_orb=N_ORB)
    assert r.diagonalizable  # tensor is diagonal in the symmetry-adapted Q basis
    assert _n_groups(r) == 3  # continuous L_z alone does not equate x and y


def test_generic_falls_back_to_full_tensor():
    rng = np.random.default_rng(0)
    h = _diag_h(lambda m: 0.0, lambda m: 0.0)
    for s in range(2):
        idx = [c2i(nBaths, (2, s, m)) for m in range(-2, 3)]
        a = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
        a = a + a.conj().T
        for p, i in enumerate(idx):
            for q, j in enumerate(idx):
                h[i, j] += a[p, q]
    r = component_symmetry_reduction(COMPONENTS, h, n_orb=N_ORB)
    assert not r.diagonalizable
    assert _n_groups(r) == 3
    np.testing.assert_allclose(r.Q, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("energy_d", [lambda m: 0.0, lambda m: 0.3 * abs(m)])
def test_diagonal_reconstruction_matches_full_tensor(energy_d):
    """For a symmetry-respecting chi (commuting with the discovered representation), the
    diagonal reconstruction chi = Q diag Q^dagger reproduces the full tensor exactly."""
    h = _diag_h(lambda m: 0.2 * abs(m), energy_d)
    r = component_symmetry_reduction(COMPONENTS, h, n_orb=N_ORB)
    if not r.diagonalizable:
        pytest.skip("fallback path uses the full tensor directly")

    # Build a random Hermitian chi that respects the symmetry: chi = Q diag(real) Q^dagger.
    rng = np.random.default_rng(1)
    diag = rng.standard_normal(3)
    # Equal within each group, as the physics guarantees.
    grouped = np.array([diag[r.group_of_column[a]] for a in range(3)])
    chi = r.Q @ np.diag(grouped) @ r.Q.conj().T

    # Reconstruct from representative diagonals only.
    chip = r.Q.conj().T @ chi @ r.Q
    rep_vals = np.array([chip[c, c] for c in r.representatives])
    chi_diag = np.array([rep_vals[r.group_of_column[a]] for a in range(3)])
    chi_rebuilt = np.einsum("a,pa,qa->pq", chi_diag, r.Q, r.Q.conj())
    np.testing.assert_allclose(chi_rebuilt, chi, atol=1e-10)


# --- the streamed residual R factor (memory: the full block is O(n_orb^3), replicated) ---


def _naive_residual_block(Ts, gens, q_t, n_orb):
    """The residual block formed directly -- what :func:`_residual_r_factor` must not build."""
    b = np.empty((len(Ts) * n_orb * n_orb, len(gens)), dtype=complex)
    for k, c in enumerate(gens):
        off = 0
        for T in Ts:
            r = (c @ T - T @ c).reshape(-1)
            r = r - q_t @ (q_t.conj().T @ r)
            b[off : off + r.size, k] = r
            off += r.size
    return b


def _random_residual_inputs(n_orb, n_comp, n_gen, seed):
    rng = np.random.default_rng(seed)

    def mat():
        return rng.standard_normal((n_orb, n_orb)) + 1j * rng.standard_normal((n_orb, n_orb))

    Ts = [mat() for _ in range(n_comp)]
    gens = [mat() for _ in range(n_gen)]
    q_t, _ = np.linalg.qr(np.array([T.reshape(-1) for T in Ts]).T)
    return Ts, gens, q_t


@pytest.mark.parametrize("n_orb,n_comp,n_gen", [(8, 3, 5), (12, 3, 30), (6, 3, 40), (10, 2, 12)])
def test_residual_r_factor_reproduces_a_directly_formed_block(n_orb, n_comp, n_gen):
    """``B = QR`` gives ``B^dagger B = R^dagger R``: same singular values, same right
    singular vectors, so the same null space -- which is all the caller uses."""
    Ts, gens, q_t = _random_residual_inputs(n_orb, n_comp, n_gen, seed=3)
    b = _naive_residual_block(Ts, gens, q_t, n_orb)
    r = lie_algebra._residual_r_factor(Ts, gens, q_t, n_orb)

    assert r.shape[1] == n_gen
    gram = np.linalg.norm(r.conj().T @ r - b.conj().T @ b)
    np.testing.assert_allclose(gram / max(np.linalg.norm(b.conj().T @ b), 1.0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.svd(r, compute_uv=False), np.linalg.svd(b, compute_uv=False), atol=1e-10)


def test_residual_r_factor_does_not_depend_on_the_row_block_size(monkeypatch):
    """Chunking is an implementation detail: a one-row-at-a-time block and a block that
    swallows everything must agree. Catches row-slice arithmetic that is off at a boundary
    (the q_t rows must track the same i range as the commutator rows)."""
    n_orb, n_comp, n_gen = 9, 3, 16
    Ts, gens, q_t = _random_residual_inputs(n_orb, n_comp, n_gen, seed=5)

    monkeypatch.setattr(lie_algebra, "_RESIDUAL_BLOCK_BYTES", 16 * n_orb * n_gen)  # one i per block
    tiny = lie_algebra._residual_r_factor(Ts, gens, q_t, n_orb)
    monkeypatch.setattr(lie_algebra, "_RESIDUAL_BLOCK_BYTES", 1 << 30)  # one block per component
    whole = lie_algebra._residual_r_factor(Ts, gens, q_t, n_orb)

    np.testing.assert_allclose(
        np.linalg.svd(tiny, compute_uv=False), np.linalg.svd(whole, compute_uv=False), atol=1e-10
    )
    np.testing.assert_allclose(tiny.conj().T @ tiny, whole.conj().T @ whole, atol=1e-10)


def test_spherical_collapse_survives_a_one_row_block(monkeypatch):
    """The public result is unchanged when the residual block is streamed one row at a time."""
    h = _diag_h(lambda m: 0.0, lambda m: 0.0)
    monkeypatch.setattr(lie_algebra, "_RESIDUAL_BLOCK_BYTES", 1)
    r = component_symmetry_reduction(COMPONENTS, h, n_orb=N_ORB)
    assert r.diagonalizable
    assert _n_groups(r) == 1
    assert r.group_of_column == [0, 0, 0]
