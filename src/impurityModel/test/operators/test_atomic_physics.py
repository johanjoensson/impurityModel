import numpy as np

from impurityModel.ed.atomic_physics import dc_MLFT, get_spherical_2_cubic_matrix, getUop_from_rspt_u4
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, inner


def test_dc_MLFT():
    # Only 3d
    res = dc_MLFT(5, 1.0, Fdd=[5.0, 0, 1.0, 0, 1.0])
    assert 2 in res
    assert np.isclose(res[2], (5.0 - 14.0 / 441 * 2.0) * 5 - 1.0)

    # with 2p
    res = dc_MLFT(5, 1.0, Fdd=[5.0, 0, 1.0, 0, 1.0], n2p_i=6, Fpd=[4.0, 0, 0], Gpd=[0, 1.0, 0, 1.0])
    assert 1 in res
    assert 2 in res


def test_get_spherical_2_cubic_matrix():
    u_p = get_spherical_2_cubic_matrix(l=1)
    assert u_p.shape == (3, 3)

    u_d = get_spherical_2_cubic_matrix(l=2)
    assert u_d.shape == (5, 5)


def _det(occupied, n_orbs):
    """Single-determinant ManyBodyState with the given occupied orbitals (MSB-first bits)."""
    data = bytearray((n_orbs + 7) // 8)
    for orb in occupied:
        data[orb // 8] |= 1 << (7 - orb % 8)
    return ManyBodyState({SlaterDeterminant.from_bytes(bytes(data)): 1.0})


def _random_rspt_u4(n, seed=0):
    """Random tensor with RSPt's symmetries: u(i,j,k,l)=u(j,i,l,k) and u(i,j,k,l)=conj(u(k,l,i,j))."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((n, n, n, n)) + 1j * rng.standard_normal((n, n, n, n))
    r = r + r.transpose((1, 0, 3, 2))  # two-electron exchange symmetry
    return r + np.conj(r.transpose((2, 3, 0, 1)))  # hermiticity


def test_getUop_from_rspt_u4_density_density():
    """u4[i,j,i,j] = <ij|V|ij> is the direct (Hartree) element: <D|U|D> = U n_i n_j."""
    n = 2
    u4 = np.zeros((n, n, n, n), dtype=complex)
    u4[0, 1, 0, 1] = 1.7
    u4[1, 0, 1, 0] = 1.7  # exchange-symmetric partner
    u_op = ManyBodyOperator(getUop_from_rspt_u4(u4))

    both = _det([0, 1], n)
    assert np.isclose(inner(both, u_op(both, 0)), 1.7)
    single = _det([0], n)
    assert np.isclose(inner(single, u_op(single, 0)), 0.0)


def test_getUop_from_rspt_u4_matches_old_convention():
    """The RSPt-order operator equals the old (moveaxis(u4, 1, 0)) operator.

    RSPt's u4[i,j,k,l] multiplies c^dag_i c^dag_j c_l c_k; the previous code
    read the tensor as c^dag_i c^dag_j c_k c_l and relied on the wrapper
    pre-swapping the first two indices. By the exchange symmetry
    u(i,j,k,l) = u(j,i,l,k) both prescriptions define the same operator.
    """
    from itertools import combinations

    n = 4
    u4 = _random_rspt_u4(n)
    new_op = ManyBodyOperator(getUop_from_rspt_u4(u4))

    u4_old = np.moveaxis(u4, 1, 0)
    old_dict = {}
    for i, j, k, l in np.ndindex(u4_old.shape):
        if abs(u4_old[i, j, k, l]) > 1e-10:
            old_dict[((i, "c"), (j, "c"), (k, "a"), (l, "a"))] = u4_old[i, j, k, l] / 2
    old_op = ManyBodyOperator(old_dict)

    dets = [_det(occ, n) for n_el in range(n + 1) for occ in combinations(range(n), n_el)]
    for bra in dets:
        for ket in dets:
            assert np.isclose(inner(bra, new_op(ket, 0)), inner(bra, old_op(ket, 0)), atol=1e-12)
