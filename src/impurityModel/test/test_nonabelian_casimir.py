"""Tests for non-abelian symmetry detection + Casimir reconstruction (companion plan)."""

from itertools import combinations

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp, inner
from impurityModel.ed.symmetries import (
    apply_reconstructed_casimir,
    discover_one_body_symmetries,
    expect_reconstructed_casimir,
    in_span,
    is_abelian,
    structure_constants,
)

# Hubbard dimer, 4 spin-orbitals. Layout: orbital = spin*2 + site, up={0,1}, down={2,3}.
T_HOP = 0.7
_HSITE = np.array([[0.0, -T_HOP], [-T_HOP, 0.0]], dtype=complex)
H_DIMER = np.kron(np.eye(2), _HSITE)  # spin-diagonal hopping

# Single-particle spin operators (spatial pairs (up,dn) = (0,2) and (1,3)).
_SZ = 0.5 * np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)
_SX = np.zeros((4, 4), dtype=complex)
_SY = np.zeros((4, 4), dtype=complex)
for _u, _d in ((0, 2), (1, 3)):
    _SX[_u, _d] += 0.5
    _SX[_d, _u] += 0.5
    _SY[_u, _d] += 0.5 / 1j
    _SY[_d, _u] += -0.5 / 1j


def _sd(occ, n=4):
    b = bytearray((n + 7) // 8)
    for o in occ:
        b[o // 8] |= 1 << (7 - o % 8)
    return SlaterDeterminant.from_bytes(bytes(b))


def _state(terms):
    psi = ManyBodyState({_sd(o): a for o, a in terms})
    return psi / psi.norm()


def test_detect_nonabelian_su2():
    """The Hubbard-dimer commutant is non-abelian; S_x,S_y,S_z form su(2)."""
    gens = discover_one_body_symmetries(H_DIMER)
    assert not is_abelian(gens)  # non-abelian symmetry present

    # The three spin generators are all in the discovered algebra.
    for s in (_SX, _SY, _SZ):
        assert in_span(gens, s)

    # Structure constants of {S_x, S_y, S_z} reproduce [S_a, S_b] = i eps_abc S_c.
    f = structure_constants([_SX, _SY, _SZ])
    eps = np.zeros((3, 3, 3))
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        eps[a, b, c] = 1
        eps[a, c, b] = -1
    np.testing.assert_allclose(f, 1j * eps, atol=1e-10)


def test_detect_abelian():
    """A crystal-field-split (diagonal) one-body H has an abelian commutant."""
    h = np.diag([0.0, 0.3, 0.7, 1.1]).astype(complex)  # distinct energies
    gens = discover_one_body_symmetries(h)
    assert is_abelian(gens)


def test_casimir_matches_handbuilt():
    """Reconstructed S^2 = S_x^2+S_y^2+S_z^2 equals the hand-built S^2 (Phase 1.2)."""
    from impurityModel.ed.observables import expect_casimir, make_spin_operators

    s_plus, s_minus, s_z = make_spin_operators([(2, 0), (3, 1)])  # (dn, up) pairs

    states = [
        _state([([0, 1], 1.0)]),  # both up -> triplet S=1
        _state([([2, 3], 1.0)]),  # both down -> triplet
        _state([([0, 3], 1.0), ([1, 2], -1.0)]),  # singlet S=0
        _state([([0, 3], 1.0), ([1, 2], 1.0)]),  # triplet S_z=0
        _state([([0], 1.0)]),  # one electron -> doublet S=1/2
    ]
    for psi in states:
        reconstructed = expect_reconstructed_casimir(psi, [_SX, _SY, _SZ])
        handbuilt = expect_casimir(psi, s_plus, s_minus, s_z)
        assert np.isclose(reconstructed, handbuilt, atol=1e-10)


def _matrix(apply_fn, dets):
    states = [ManyBodyState({d: 1.0}) for d in dets]
    n = len(states)
    m = np.zeros((n, n), dtype=complex)
    for j, sj in enumerate(states):
        col = apply_fn(sj)
        for i, si in enumerate(states):
            m[i, j] = inner(si, col)
    return m


def test_casimir_commutes_with_H():
    """[S^2, H] = 0 for the reconstructed Casimir on a spin-symmetric interacting H."""
    # Hubbard dimer with on-site U (spin-symmetric).
    terms = {((a, "c"), (b, "a")): -T_HOP for a, b in ((0, 1), (1, 0), (2, 3), (3, 2))}
    for up, dn in ((0, 2), (1, 3)):
        terms[((up, "c"), (dn, "c"), (dn, "a"), (up, "a"))] = 2.5
    hOp = ManyBodyOperator(terms)

    dets = [_sd(list(o)) for ne in range(5) for o in combinations(range(4), ne)]
    h_mat = _matrix(lambda psi: applyOp(hOp, psi), dets)
    c_mat = _matrix(lambda psi: apply_reconstructed_casimir(psi, [_SX, _SY, _SZ]), dets)
    assert np.linalg.norm(h_mat @ c_mat - c_mat @ h_mat) < 1e-10


def test_multiplet_labeling():
    """A degenerate singlet+triplet manifold is labeled S=0 (x1) and S=1 (x3) by the
    reconstructed Casimir, with degeneracy 2S+1."""
    from impurityModel.ed.observables import casimir_to_quantum_number, manifold_observable_values

    # Two spatial orbitals (up:0/dn:2, up:1/dn:3), one electron each -> singlet + triplet.
    manifold = [
        _state([([0, 1], 1.0)]),  # both up (triplet S_z=+1)
        _state([([2, 3], 1.0)]),  # both down (triplet S_z=-1)
        _state([([0, 3], 1.0)]),  # up0 + dn1 (mixes singlet/triplet)
        _state([([2, 1], 1.0)]),  # dn0 + up1 (mixes singlet/triplet)
    ]
    energies = np.zeros(4)  # degenerate manifold

    s2_vals = manifold_observable_values(
        manifold, energies, lambda psi: apply_reconstructed_casimir(psi, [_SX, _SY, _SZ])
    )
    s_labels = sorted(round(casimir_to_quantum_number(v), 6) for v in s2_vals)
    assert s_labels == [0.0, 1.0, 1.0, 1.0]  # one singlet, one triplet (2S+1=3 states)
    assert sum(np.isclose(casimir_to_quantum_number(v), 1.0) for v in s2_vals) == 3
    assert sum(np.isclose(casimir_to_quantum_number(v), 0.0) for v in s2_vals) == 1
