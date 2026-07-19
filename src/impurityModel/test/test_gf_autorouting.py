"""Tests for symmetry auto-routing of Green's functions (symmetry plan, Phase 4)."""

from itertools import combinations

import numpy as np

import impurityModel.ed.product_state_representation as psr
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp, inner
from impurityModel.ed.symmetries import (
    conserved_subset_charges,
    gf_sector_restrictions,
    green_function_allowed_mask,
    green_function_block_structure,
    measure_conserved_charges,
)


def _sd(occ, n):
    nb = (n + 7) // 8
    data = bytearray(nb)
    for o in occ:
        data[o // 8] |= 1 << (7 - o % 8)
    return SlaterDeterminant.from_bytes(bytes(data))


def _operator_matrix(op, states):
    n = len(states)
    mat = np.zeros((n, n), dtype=complex)
    for j, sj in enumerate(states):
        col = applyOp(op, sj)
        for i, si in enumerate(states):
            mat[i, j] = inner(si, col)
    return mat


def _brute_force_gf(op, n_orb, omega=0.5 + 0.1j):
    """Dense one-body retarded Green's function G_ij(omega) over the full Fock space."""
    dets = [_sd(o, n_orb) for ne in range(n_orb + 1) for o in combinations(range(n_orb), ne)]
    states = [ManyBodyState({d: 1.0}) for d in dets]
    n = len(states)
    h = _operator_matrix(op, states)
    evals, evecs = np.linalg.eigh(h)
    e0, psi = evals[0], evecs[:, 0]
    c = [_operator_matrix(ManyBodyOperator({((p, "a"),): 1.0}), states) for p in range(n_orb)]
    cdag = [ci.conj().T for ci in c]
    eye = np.eye(n)
    r_add = np.linalg.inv(omega * eye - (h - e0 * eye))
    r_rem = np.linalg.inv(omega * eye + (h - e0 * eye))
    gf = np.zeros((n_orb, n_orb), dtype=complex)
    for i in range(n_orb):
        for j in range(n_orb):
            gf[i, j] = psi.conj() @ c[i] @ r_add @ cdag[j] @ psi + psi.conj() @ cdag[j] @ r_rem @ c[i] @ psi
    return gf


def _hubbard_dimer():
    """0,1 = up sites; 2,3 = down sites; spin-diagonal hopping + on-site U."""
    t, u = 0.9, 2.3
    terms = {}
    for a, b in ((0, 1), (2, 3)):
        terms[((a, "c"), (b, "a"))] = -t
        terms[((b, "c"), (a, "a"))] = -t
    for up, dn in ((0, 2), (1, 3)):
        terms[((up, "c"), (dn, "c"), (dn, "a"), (up, "a"))] = u
    return ManyBodyOperator(terms)


def test_automatic_decoupling():
    """Transitions flagged symmetry-decoupled are *exactly* zero in the brute-force GF."""
    op = _hubbard_dimer()
    blocks = green_function_block_structure(op, 4)
    assert blocks == [frozenset({0, 1}), frozenset({2, 3})]  # up / down decoupled

    mask = green_function_allowed_mask(op, 4)
    gf = _brute_force_gf(op, 4)

    # Forbidden (cross-spin) entries are exactly zero; allowed entries are not all zero.
    assert np.allclose(gf[~mask], 0.0, atol=1e-12)
    assert np.max(np.abs(gf[mask])) > 1e-6
    # No nonzero GF element falls outside an allowed block.
    assert np.all((np.abs(gf) > 1e-9) <= mask)


def test_symmetry_breaking_couples_blocks():
    """A spin-flip term merges the spin blocks -> the GF is fully coupled (no forbidden pairs)."""
    op = _hubbard_dimer()
    terms = op.to_dict()
    terms[((0, "c"), (2, "a"))] = 0.4  # up-site0 <-> down-site0
    terms[((2, "c"), (0, "a"))] = 0.4
    op2 = ManyBodyOperator(terms)

    blocks = green_function_block_structure(op2, 4)
    assert blocks == [frozenset({0, 1, 2, 3})]  # single block
    mask = green_function_allowed_mask(op2, 4)
    assert np.all(mask)  # every pair allowed


# ---------------------------------------------------------------------------
# Phase 4.3: Krylov-space sector confinement (addition vs removal)
# ---------------------------------------------------------------------------


def _occset(det_bytes, n):
    return {k for k, bit in enumerate(psr.bytes2bitarray(det_bytes, n)) if bit}


def test_addition_removal_sectors():
    """Addition shifts the orbital's charge by +1, removal by -1; others unchanged."""
    op = _hubbard_dimer()
    charges = conserved_subset_charges(op, 4)  # [{0,1}=N_up, {2,3}=N_down]
    gs_occ = [1, 1]  # half-filled singlet sector

    # Orbital 0 is in the N_up block.
    add = gf_sector_restrictions(charges, gs_occ, 0, "addition")
    rem = gf_sector_restrictions(charges, gs_occ, 0, "removal")
    assert add == {frozenset({0, 1}): (2, 2), frozenset({2, 3}): (1, 1)}  # N_up -> 2
    assert rem == {frozenset({0, 1}): (0, 0), frozenset({2, 3}): (1, 1)}  # N_up -> 0


def test_green_function_explosion_prevention():
    """Confining the addition GF to its sector reproduces it with a smaller basis."""
    op = _hubbard_dimer()
    n = 4
    # Full Fock setup.
    all_dets = [_sd(o, n) for ne in range(n + 1) for o in combinations(range(n), ne)]
    all_bytes = [bytes(d.to_bytearray()) for d in all_dets]
    states = [ManyBodyState({d: 1.0}) for d in all_dets]
    h = _operator_matrix(op, states)
    evals, evecs = np.linalg.eigh(h)
    e0, psi = evals[0], evecs[:, 0]
    cdag0 = _operator_matrix(ManyBodyOperator({((0, "a"),): 1.0}), states).conj().T

    omega = 0.5 + 0.1j
    v = cdag0 @ psi  # c_0^dagger |psi>, lives in the addition sector
    r_full = np.linalg.inv(omega * np.eye(len(states)) - (h - e0 * np.eye(len(states))))
    g_add_full = v.conj() @ r_full @ v

    # Confine to the addition sector.
    charges = conserved_subset_charges(op, n)
    gs_occ = measure_conserved_charges(ManyBodyState({d: psi[i] for i, d in enumerate(all_dets)}), charges, n)
    restr = gf_sector_restrictions(charges, gs_occ, 0, "addition")
    sector_idx = [
        i for i, b in enumerate(all_bytes) if all(lo <= len(s & _occset(b, n)) <= hi for s, (lo, hi) in restr.items())
    ]
    h_sec = h[np.ix_(sector_idx, sector_idx)]
    v_sec = v[sector_idx]
    r_sec = np.linalg.inv(omega * np.eye(len(sector_idx)) - (h_sec - e0 * np.eye(len(sector_idx))))
    g_add_sector = v_sec.conj() @ r_sec @ v_sec

    assert np.isclose(g_add_full, g_add_sector, atol=1e-10)  # same GF
    assert len(sector_idx) < len(all_dets)  # smaller basis
    # c_0^dagger|psi> has no weight outside the addition sector.
    assert np.allclose(np.delete(v, sector_idx), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Phase 4.2: auto-derive block structure (blocks + equivalences) from H
# ---------------------------------------------------------------------------


def _cubic_d_shell():
    """Diagonal cubic crystal field: per spin [eg, eg, t2g, t2g, t2g]; spin down 0-4, up 5-9."""
    e_eg, e_t2g = 0.6, 0.0
    diag = [e_eg, e_eg, e_t2g, e_t2g, e_t2g] * 2
    return ManyBodyOperator({((i, "c"), (i, "a")): diag[i] for i in range(10)})


def test_auto_block_structure_detects_equivalences():
    """auto_block_structure derives the orbital blocks and the t2g/eg identical groups."""
    from impurityModel.ed.symmetries import auto_block_structure, green_function_block_structure

    op = _cubic_d_shell()
    bs = auto_block_structure(op, 10)

    # Blocks agree with the GF selection-rule sectors.
    gf_blocks = {frozenset(b) for b in green_function_block_structure(op, 10)}
    assert {frozenset(b) for b in bs.blocks} == gf_blocks

    # The identical-block detection correctly groups the (degenerate) eg and t2g orbitals.
    eg = {0, 1, 5, 6}
    t2g = {2, 3, 4, 7, 8, 9}
    identical_groups = {frozenset(g) for g in bs.identical_blocks if g}
    assert eg in identical_groups
    assert t2g in identical_groups


def test_advanced_block_equivalence():
    """Reconstructing from only the inequivalent representatives reproduces the full
    block matrix — including the self-particle-hole-symmetric t2g class (the get_inequivalent_blocks fix)."""
    from impurityModel.ed.block_structure import build_block_structure, build_matrix, get_equivalent_blocks

    # A GF-like block matrix: eg orbitals share one value, t2g another (t2g at 0 -> self-PH).
    g_eg, g_t2g = 0.6 + 0.3j, 0.0 + 0.0j
    diag = [g_eg, g_eg, g_t2g, g_t2g, g_t2g] * 2
    full = np.diag(diag).astype(complex)

    bs = build_block_structure(None, mat=full)
    # Both equivalence classes are represented (the bug dropped t2g).
    assert bs.inequivalent_blocks == [0, 2]
    assert {frozenset(g) for g in get_equivalent_blocks(bs)} == {
        frozenset({0, 1, 5, 6}),
        frozenset({2, 3, 4, 7, 8, 9}),
    }

    # Compute only the inequivalent representative blocks, then reconstruct the rest.
    inequivalent_parts = [full[np.ix_(bs.blocks[ib], bs.blocks[ib])] for ib in bs.inequivalent_blocks]
    reconstructed = build_matrix(inequivalent_parts, bs)
    np.testing.assert_allclose(reconstructed, full, atol=1e-12)
