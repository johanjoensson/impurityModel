"""The GF block structure must respect the interaction, not only the one-body part.

``impurity_block_structure`` reads blocks and equivalences off the dressed one-body matrix.
``reconcile_block_structure_with_interaction`` corrects them for an interaction with less
symmetry. The oracle is an exact atomic Green's function: two orbitals in different blocks must
have ``G_ij == 0``, and two orbitals called equivalent must have equal ``G_ii``.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from impurityModel.ed import atomic_physics
from impurityModel.ed.atomic_physics import get_spherical_2_cubic_matrix
from impurityModel.ed.block_structure import build_block_structure
from impurityModel.ed.interaction_models import density_density_u4, kanamori_u4, terms_u4
from impurityModel.ed.lie_algebra import rotate_two_body
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.model import atomic_u4
from impurityModel.ed.symmetries import impurity_two_body_tensor, reconcile_block_structure_with_interaction


def _reconcile(u4, m):
    op = ManyBodyOperator(atomic_physics.getUop_from_rspt_u4(u4))
    two_body = impurity_two_body_tensor(op, list(range(u4.shape[0])))
    before = build_block_structure(None, mat=m)
    after, changed = reconcile_block_structure_with_interaction(before, two_body, m)
    return before, after, changed


def _atomic_greens_function(h1, u4, z, tau=0.3, mu=0.0):
    """Exact thermal G_ij(z) of h1 + U on the full Fock space (sparse Jordan-Wigner)."""
    n = h1.shape[0]
    dim = 2**n
    c = []
    for i in range(n):
        m = sp.lil_matrix((dim, dim))
        for state in range(dim):
            if state >> i & 1:
                m[state ^ (1 << i), state] = (-1) ** bin(state & ((1 << i) - 1)).count("1")
        c.append(m.tocsr())
    cd = [x.T.tocsr() for x in c]
    h = sp.csr_matrix((dim, dim), dtype=complex)
    for i, j in zip(*np.nonzero(np.abs(h1) > 1e-14)):
        h = h + (h1[i, j] - (mu if i == j else 0)) * (cd[i] @ c[j])
    for i, j, k, l in zip(*np.nonzero(np.abs(u4) > 1e-12)):
        h = h + 0.5 * u4[i, j, k, l] * (cd[i] @ cd[j] @ c[l] @ c[k])
    e, v = np.linalg.eigh(h.toarray())
    w = np.exp(-(e - e[0]) / tau)
    w /= w.sum()
    cv = [v.conj().T @ (ci @ v) for ci in c]
    g = np.zeros((n, n), dtype=complex)
    denominator = z + e[:, None] - e[None, :]
    weights = w[:, None] + w[None, :]
    for i in range(n):
        for j in range(n):
            g[i, j] = np.sum(weights * cv[i] * cv[j].conj() / denominator)
    return g


def _assert_structure_is_exact(block_structure, g, tol=1e-10):
    """No G between blocks, and every block called identical to another has the same G block."""
    blocks = block_structure.blocks
    block_of = {o: b for b, blk in enumerate(blocks) for o in blk}
    n = g.shape[0]
    across = max([abs(g[i, j]) for i in range(n) for j in range(n) if block_of[i] != block_of[j]] + [0.0])
    assert across < tol, f"G couples different blocks: {across:.3e}"
    for x, members in enumerate(block_structure.identical_blocks):
        for y in members:
            gx, gy = g[np.ix_(blocks[x], blocks[x])], g[np.ix_(blocks[y], blocks[y])]
            assert np.max(np.abs(gx - gy)) < tol, f"blocks {x} and {y} are called identical but differ"


# ---------------------------------------------------------------------------- Slater: no-op


def _cubic_cf(l):
    u = get_spherical_2_cubic_matrix(spinpol=False, l=l)
    levels = {1: [0.2, 0.2, 0.2], 2: [0.6, 0.6, -0.4, -0.4, -0.4], 3: [0.3, 0.3, 0.3, -0.1, -0.1, -0.1, -0.5]}[l]
    return np.kron(np.eye(2), u @ np.diag(levels) @ u.conj().T)


@pytest.mark.parametrize("l", [1, 2, 3])
@pytest.mark.parametrize("basis", ["spherical", "cubic"])
def test_slater_interaction_leaves_the_block_structure_alone(l, basis):
    """Slater-Condon U is rotationally invariant, so whatever the one-body part separates stays
    separate -- the premise the one-body derivation rests on. Nothing may change."""
    u4 = atomic_u4(l, [6.0, 0, 8.0, 0, 5.0, 0, 4.0][: 2 * l + 1])
    m = _cubic_cf(l)
    if basis == "cubic":
        rot = get_spherical_2_cubic_matrix(spinpol=True, l=l)
        u4, m = rotate_two_body(u4, rot), rot.conj().T @ m @ rot
    before, after, changed = _reconcile(u4, m)
    assert not changed
    assert after is before


def test_soc_eigenbasis_blocks_are_merged_where_the_exact_g_couples_them():
    """With crystal field AND spin-orbit coupling a d shell carries the Gamma_8 irrep twice. In
    the one-body eigenbasis the two copies sit at different energies, so the one-body matrix
    keeps them apart -- but U couples them, and the exact G has elements between them. The
    reconciled structure must be exact where the one-body one is not."""
    l = 2
    labels = [(l, s, m) for s in range(2) for m in range(-l, l + 1)]
    index = {lab: i for i, lab in enumerate(labels)}
    soc = np.zeros((10, 10), dtype=complex)
    for ((a, _), (b, _)), value in atomic_physics.getSOCop(1.0, l=l).items():
        soc[index[a], index[b]] += value
    e, w = np.linalg.eigh(_cubic_cf(l) + 0.3 * soc)
    u4 = rotate_two_body(atomic_u4(l, [6.0, 0, 8.0, 0, 5.0]), w)
    m = np.diag(e).astype(complex)
    before, after, changed = _reconcile(u4, m)
    g = _atomic_greens_function(m, u4, 0.7j, mu=30.0)
    assert changed
    with pytest.raises(AssertionError, match="couples different blocks"):
        _assert_structure_is_exact(before, g)
    _assert_structure_is_exact(after, g)


# ---------------------------------------------------------------------------- model interactions


def _two_orbital(u4, split=0.0):
    """Two degenerate (or split) orbitals, both spins: M diagonal, 1x1 blocks."""
    m = np.diag([0.0, split, 0.0, split]).astype(complex)
    return m, u4


def test_kanamori_on_degenerate_orbitals_changes_nothing():
    """Pair hopping and spin flip keep each orbital's parity, so G stays orbital- and spin-diagonal."""
    m, u4 = _two_orbital(kanamori_u4(2, 3.0, J=0.6, U_prime=1.9, J_pair=0.4))
    _, after, changed = _reconcile(u4, m)
    assert not changed
    _assert_structure_is_exact(after, _atomic_greens_function(m, u4, 0.7j, mu=2.0))


def test_density_assisted_hopping_merges_the_orbitals():
    """<0 1|V|1 1>: an electron hops 0 -> 1 when orbital 1 is occupied. G_01 becomes non-zero."""
    u4 = terms_u4(2, spatial=[[0, 0, 0, 0, 3.0], [1, 1, 1, 1, 3.0], [0, 1, 1, 1, 0.5]])
    m, u4 = _two_orbital(u4)
    before, after, changed = _reconcile(u4, m)
    g = _atomic_greens_function(m, u4, 0.7j, mu=1.5)
    assert changed
    with pytest.raises(AssertionError, match="couples different blocks"):
        _assert_structure_is_exact(before, g)
    _assert_structure_is_exact(after, g)
    assert sorted(map(sorted, after.blocks)) == [[0, 1], [2, 3]]  # merged within each spin only


def test_spin_transfer_term_merges_the_spins():
    """A spin-orbital term flipping orbital 0's spin when orbital 1 (down) is occupied --
    c+_{0 up} c+_{1 dn} c_{1 dn} c_{0 dn}, spin-orbitals 0dn=0, 1dn=1, 0up=2, 1up=3 -- couples
    orbital 0's two spins and nothing else."""
    u4 = terms_u4(2, spatial=[[0, 0, 0, 0, 3.0], [1, 1, 1, 1, 3.0]], spin_orbital=[[2, 1, 0, 1, 0.4]])
    m = np.diag([0.0, 0.5, 0.0, 0.5]).astype(complex)
    before, after, changed = _reconcile(u4, m)
    g = _atomic_greens_function(m, u4, 0.7j, mu=1.5)
    assert changed
    assert sorted(map(sorted, after.blocks)) == [[0, 2], [1], [3]]
    with pytest.raises(AssertionError, match="couples different blocks"):
        _assert_structure_is_exact(before, g)
    _assert_structure_is_exact(after, g)


def test_orbital_dependent_density_interaction_splits_degenerate_orbitals():
    """Degenerate orbitals with different Hubbard U are not equivalent: their G differ, so the
    identical relation the one-body matrix finds must be dropped (blocks stay, copies do not)."""
    u4 = density_density_u4(np.diag([3.0, 1.0]))
    m, u4 = _two_orbital(u4)
    before, after, changed = _reconcile(u4, m)
    assert len(before.inequivalent_blocks) == 1
    assert changed
    assert len(after.inequivalent_blocks) == 2
    g = _atomic_greens_function(m, u4, 0.7j, mu=1.0)
    assert abs(g[0, 0] - g[1, 1]) > 1e-3
    _assert_structure_is_exact(after, g)
