"""Tests for single-particle basis rotation of operators (symmetry plan, Phase 5.1)."""

from itertools import combinations

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp, inner
from impurityModel.ed.symmetries import (
    rotate_hamiltonian,
    rotate_one_body,
    rotate_two_body,
    extract_tensors,
)


def _random_unitary(n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(a)
    # Fix the phase so q is deterministic-ish (not required, just tidy).
    return q @ np.diag(np.exp(-1j * np.angle(np.diag(r))))


def _sd(occupied, n_orbs):
    n_bytes = (n_orbs + 7) // 8
    data = bytearray(n_bytes)
    for orb in occupied:
        data[orb // 8] |= 1 << (7 - (orb % 8))
    return SlaterDeterminant.from_bytes(bytes(data))


def _dense_matrix(op, n_orb, n_elec):
    """<det_a|op|det_b> over the complete n_elec-particle sector of n_orb orbitals."""
    dets = [_sd(occ, n_orb) for occ in combinations(range(n_orb), n_elec)]
    states = [ManyBodyState({d: 1.0}) for d in dets]
    n = len(states)
    mat = np.zeros((n, n), dtype=complex)
    for b, sb in enumerate(states):
        col = applyOp(op, sb)
        for a, sa in enumerate(states):
            mat[a, b] = inner(sa, col)
    return mat


def _hubbard_dimer_operator():
    """2-site Hubbard: orbitals 0,1 = sites (up); 2,3 = sites (down)."""
    t, u = 0.9, 2.3
    terms = {}
    # hopping (spin up: 0<->1, spin down: 2<->3)
    for a, b in ((0, 1), (2, 3)):
        terms[((a, "c"), (b, "a"))] = -t
        terms[((b, "c"), (a, "a"))] = -t
    # on-site U n_up n_dn, normal-ordered: c†_up c†_dn c_dn c_up
    for up, dn in ((0, 2), (1, 3)):
        terms[((up, "c"), (dn, "c"), (dn, "a"), (up, "a"))] = u
    return ManyBodyOperator(terms)


def test_python_basis_rotation_roundtrip():
    """Rotating by U then by U† recovers the original tensors to 1e-12."""
    op = _hubbard_dimer_operator()
    u = _random_unitary(4, seed=1)
    h, v, const = extract_tensors(op, n_orb=4)

    h2 = rotate_one_body(rotate_one_body(h, u), u.conj().T)
    v2 = rotate_two_body(rotate_two_body(v, u), u.conj().T)
    assert np.allclose(h2, h, atol=1e-12)
    assert np.allclose(v2, v, atol=1e-12)


def test_python_basis_rotation_spectrum_invariant():
    """The many-body spectrum is invariant under a single-particle basis rotation.

    This is the decisive check of the two-body rotation index convention: it would
    fail for any inconsistency between rotate_two_body and extract_tensors' V.
    """
    op = _hubbard_dimer_operator()
    u = _random_unitary(4, seed=2)
    op_rot = rotate_hamiltonian(op, u)

    for n_elec in (1, 2, 3):
        ev = np.linalg.eigvalsh(_dense_matrix(op, 4, n_elec))
        ev_rot = np.linalg.eigvalsh(_dense_matrix(op_rot, 4, n_elec))
        np.testing.assert_allclose(ev, ev_rot, atol=1e-10)


def test_rotation_diagonalizes_discovered_generators():
    """In the joint-eigenbasis U, the discovered abelian generators become diagonal."""
    from impurityModel.ed.symmetries import (
        discover_one_body_symmetries,
        cartan_subalgebra,
        joint_diagonalize,
    )

    # SU(2) Hubbard dimer one-body part (spin-diagonal hopping).
    t = 0.7
    hsite = np.array([[0.0, -t], [-t, 0.0]], dtype=complex)
    h = np.kron(np.eye(2), hsite)

    gens = discover_one_body_symmetries(h)
    cartan = cartan_subalgebra(gens)
    u, _ = joint_diagonalize(cartan)

    for g in cartan:
        g_rot = rotate_one_body(g, u)
        off = g_rot - np.diag(np.diag(g_rot))
        assert np.linalg.norm(off) < 1e-9


def test_symmetry_adapted_transformation_bridges_to_phase3():
    """discover -> rotate gives a basis where the Cartan generators are diagonal and
    the many-body spectrum is preserved (the Phase 3 sectorization basis)."""
    from impurityModel.ed.symmetries import symmetry_adapted_transformation

    op = _hubbard_dimer_operator()
    u, rotated_op, cartan_rotated = symmetry_adapted_transformation(op, n_orb=4)

    # Spectrum preserved (rotation is physical).
    for n_elec in (1, 2, 3):
        ev = np.linalg.eigvalsh(_dense_matrix(op, 4, n_elec))
        ev_rot = np.linalg.eigvalsh(_dense_matrix(rotated_op, 4, n_elec))
        np.testing.assert_allclose(ev, ev_rot, atol=1e-10)

    # Every returned Cartan generator is diagonal in the symmetry-adapted basis.
    assert len(cartan_rotated) == 4  # dimer one-body commutant has rank 4
    for g_rot in cartan_rotated:
        assert np.linalg.norm(g_rot - np.diag(np.diag(g_rot))) < 1e-9
