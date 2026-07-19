"""Tests for single-particle basis rotation of operators (symmetry plan, Phase 5.1)."""

from itertools import combinations

import numpy as np

from impurityModel.ed.basis_transcription import build_dense_matrix
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp, inner
from impurityModel.ed.symmetries import (
    extract_tensors,
    rotate_hamiltonian,
    rotate_one_body,
    rotate_two_body,
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
    h, v, _const = extract_tensors(op, n_orb=4)

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


def test_identity_rotation_is_a_noop():
    """Rotating by the identity leaves both tensors bit-for-bit unchanged."""
    op = _hubbard_dimer_operator()
    h, v, _ = extract_tensors(op, n_orb=4)
    eye = np.eye(4, dtype=complex)
    np.testing.assert_allclose(rotate_one_body(h, eye), h, atol=1e-14)
    np.testing.assert_allclose(rotate_two_body(v, eye), v, atol=1e-14)


def test_one_body_rotation_preserves_spectrum_and_hermiticity():
    """A unitary rotation of the one-body tensor is a similarity transform: same
    eigenvalues, and Hermiticity is preserved."""
    op = _hubbard_dimer_operator()
    h, _, _ = extract_tensors(op, n_orb=4)
    u = _random_unitary(4, seed=7)
    h_rot = rotate_one_body(h, u)
    np.testing.assert_allclose(h_rot, h_rot.conj().T, atol=1e-12)
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(h)), np.sort(np.linalg.eigvalsh(h_rot)), atol=1e-12)


def test_rotation_diagonalizes_discovered_generators():
    """In the joint-eigenbasis U, the discovered abelian generators become diagonal."""
    from impurityModel.ed.symmetries import (
        cartan_subalgebra,
        discover_one_body_symmetries,
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
    _u, rotated_op, cartan_rotated = symmetry_adapted_transformation(op, n_orb=4)

    # Spectrum preserved (rotation is physical).
    for n_elec in (1, 2, 3):
        ev = np.linalg.eigvalsh(_dense_matrix(op, 4, n_elec))
        ev_rot = np.linalg.eigvalsh(_dense_matrix(rotated_op, 4, n_elec))
        np.testing.assert_allclose(ev, ev_rot, atol=1e-10)

    # Every returned Cartan generator is diagonal in the symmetry-adapted basis.
    assert len(cartan_rotated) == 4  # dimer one-body commutant has rank 4
    for g_rot in cartan_rotated:
        assert np.linalg.norm(g_rot - np.diag(np.diag(g_rot))) < 1e-9


# ---------------------------------------------------------------------------
# Phase 5.2: pipeline runs in the rotated basis (same physics)
# ---------------------------------------------------------------------------


def _full_fock(n_orb):
    return [_sd(occ, n_orb) for ne in range(n_orb + 1) for occ in combinations(range(n_orb), ne)]


def _trace_spectral_function(op, n_orb, omegas):
    """Tr A(omega) = -Im Tr G(omega) / pi over the full Fock space (basis-invariant)."""
    dets = _full_fock(n_orb)
    states = [ManyBodyState({d: 1.0}) for d in dets]
    n = len(states)

    def mat(o):
        m = np.zeros((n, n), dtype=complex)
        for j, sj in enumerate(states):
            col = applyOp(o, sj)
            for i, si in enumerate(states):
                m[i, j] = inner(si, col)
        return m

    h = mat(op)
    evals, evecs = np.linalg.eigh(h)
    e0, psi = evals[0], evecs[:, 0]
    c = [mat(ManyBodyOperator({((p, "a"),): 1.0})) for p in range(n_orb)]
    cdag = [ci.conj().T for ci in c]
    eye = np.eye(n)
    out = []
    for omega in omegas:
        r_add = np.linalg.inv(omega * eye - (h - e0 * eye))
        r_rem = np.linalg.inv(omega * eye + (h - e0 * eye))
        tr_g = sum(
            psi.conj() @ c[p] @ r_add @ cdag[p] @ psi + psi.conj() @ cdag[p] @ r_rem @ c[p] @ psi for p in range(n_orb)
        )
        out.append(-np.imag(tr_g) / np.pi)
    return np.array(out)


def test_pipeline_rotated_basis_matches_unrotated():
    """Re-expressing H in a rotated single-particle basis and running the real Basis
    pipeline reproduces the ground-state spectrum and the (basis-invariant) spectral
    function."""
    from impurityModel.ed.manybody_basis import Basis

    op = _hubbard_dimer_operator()
    u = _random_unitary(4, seed=5)
    op_rot = rotate_hamiltonian(op, u)

    # Full many-body spectrum through the real Basis.build_dense_matrix machinery.
    all_dets = _full_fock(4)
    kw = dict(impurity_orbitals={0: [[0, 1, 2, 3]]}, bath_states=({0: [[]]}, {0: [[]]}), verbose=False)

    def spectrum(o):
        basis = Basis(initial_basis=all_dets, **kw)
        return np.sort(np.linalg.eigvalsh(np.asarray(build_dense_matrix(basis, o))))

    np.testing.assert_allclose(spectrum(op), spectrum(op_rot), atol=1e-10)

    # The trace spectral function (GF poles) is invariant under the basis rotation.
    omegas = np.array([0.3 + 0.1j, 1.0 + 0.1j, 2.5 + 0.1j, -1.5 + 0.1j])
    np.testing.assert_allclose(
        _trace_spectral_function(op, 4, omegas),
        _trace_spectral_function(op_rot, 4, omegas),
        atol=1e-9,
    )


def test_dmft_symmetry_cache():
    """U is discovered once and reused while the symmetry structure is unchanged."""
    from impurityModel.ed.symmetries import SymmetryRotationCache

    def hubbard(t, u, extra=None):
        terms = {}
        for a, b in ((0, 1), (2, 3)):
            terms[((a, "c"), (b, "a"))] = -t
            terms[((b, "c"), (a, "a"))] = -t
        for up, dn in ((0, 2), (1, 3)):
            terms[((up, "c"), (dn, "c"), (dn, "a"), (up, "a"))] = u
        if extra:
            terms.update(extra)
        return ManyBodyOperator(terms)

    cache = SymmetryRotationCache()

    # Iteration 1: discovery runs.
    u1 = cache.get_rotation(hubbard(0.9, 2.3), 4)
    assert cache.discovery_count == 1

    # Iteration 2: same symmetry, different coefficients -> cache hit, no re-discovery.
    u2 = cache.get_rotation(hubbard(1.5, 3.1), 4)
    assert cache.discovery_count == 1
    assert u2 is u1

    # Iteration 3: a spin-flip term breaks the spin symmetry -> re-discovery.
    cache.get_rotation(hubbard(0.9, 2.3, extra={((0, "c"), (2, "a")): 0.4, ((2, "c"), (0, "a")): 0.4}), 4)
    assert cache.discovery_count == 2
