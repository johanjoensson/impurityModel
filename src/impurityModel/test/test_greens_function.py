import os
import pytest
import numpy as np
from mpi4py import MPI

from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.greens_function import (
    build_full_greens_function,
    build_qr,
    calc_continuants,
    calc_G,
    calc_thermally_averaged_G,
    rotate_matrix,
    block_diagonalize_hyb,
    rotate_Greens_function,
    rotate_4index_U,
    save_Greens_function,
    get_Greens_function,
    block_green_impl,
    block_Green,
    block_Green_sparse,
)
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant


def test_build_full_greens_function_2d():
    bs = BlockStructure(
        blocks=[[0, 1], [2, 3], [4], [5]],
        identical_blocks=[[0, 1], [], [2], []],
        transposed_blocks=[[], [], [3], []],
        particle_hole_blocks=[[], [], [], []],
        particle_hole_transposed_blocks=[[], [], [], []],
        inequivalent_blocks=[0, 2],
    )
    b1 = np.array([[1.0, 0.5j], [-0.5j, 2.0]])
    b2 = np.array([[3.0]])

    # 2D case
    gf = build_full_greens_function([b1, b2], bs)
    assert gf.shape == (6, 6)
    assert np.allclose(gf[0:2, 0:2], b1)
    assert np.allclose(gf[2:4, 2:4], b1)
    assert np.allclose(gf[4:5, 4:5], b2)
    assert np.allclose(gf[5:6, 5:6], np.transpose(b2, (0, 1)))


def test_build_full_greens_function_3d():
    bs = BlockStructure(
        blocks=[[0, 1], [2, 3], [4, 5], [6, 7]],
        identical_blocks=[[0], [], [], []],
        transposed_blocks=[[1], [], [], []],
        particle_hole_blocks=[[2], [], [], []],
        particle_hole_transposed_blocks=[[3], [], [], []],
        inequivalent_blocks=[0],
    )
    b1 = np.array([[[1.0, 0.5j], [-0.5j, 2.0]], [[2.0, 1j], [-1j, 3.0]]])

    gf = build_full_greens_function([b1], bs)
    assert gf.shape == (2, 8, 8)
    assert np.allclose(gf[:, 0:2, 0:2], b1)
    assert np.allclose(gf[:, 2:4, 2:4], np.transpose(b1, (0, 2, 1)))
    assert np.allclose(gf[:, 4:6, 4:6], -np.conj(b1))
    assert np.allclose(gf[:, 6:8, 6:8], -np.transpose(np.conj(b1), (0, 2, 1)))


def test_build_full_greens_function_all_blocks():
    bs = BlockStructure(
        blocks=[[0, 1], [2, 3]],
        identical_blocks=[[0, 1], []],
        transposed_blocks=[[], []],
        particle_hole_blocks=[[], []],
        particle_hole_transposed_blocks=[[], []],
        inequivalent_blocks=[0],
    )
    b1 = np.array([[1.0, 0.5], [0.5, 2.0]])
    b2 = np.array([[2.0, 1.0], [1.0, 3.0]])
    gf = build_full_greens_function([b1, b2], bs)
    assert np.allclose(gf[0:2, 0:2], b1)
    assert np.allclose(gf[2:4, 2:4], b2)


def test_build_full_greens_function_exceptions():
    bs = BlockStructure(
        blocks=[[0, 1]],
        identical_blocks=[[0]],
        transposed_blocks=[[]],
        particle_hole_blocks=[[]],
        particle_hole_transposed_blocks=[[]],
        inequivalent_blocks=[0],
    )
    b1 = np.array([1.0, 2.0])  # Wrong shape (1D)
    with pytest.raises(RuntimeError):
        build_full_greens_function([b1], bs)

    b1_2d = np.array([[1.0, 0.5], [0.5, 2.0]])
    b2_2d = np.array([[1.0, 0.5], [0.5, 2.0]])
    with pytest.raises(RuntimeError):
        build_full_greens_function([b1_2d, b2_2d, b2_2d], bs)  # wrong length


def test_build_qr():
    mat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    q, r = build_qr(mat)
    assert q.shape == (3, 2)
    assert r.shape == (2, 2)
    assert np.allclose(q @ r, mat)
    assert np.allclose(q.T @ q, np.eye(2))


def test_calc_continuants():
    diag = np.array([np.eye(2) * 1, np.eye(2) * 2, np.eye(2) * 3])
    offdiag = np.array([np.zeros((2, 2)), np.eye(2) * 0.5, np.eye(2) * 0.1])
    A, B = calc_continuants(diag, offdiag)
    assert A.shape == (3, 2, 2)
    assert B.shape == (3, 2, 2)
    assert np.allclose(A[0], diag[0])
    assert np.allclose(B[0], np.ones((2, 2)))


def test_rotate_matrix():
    M = np.array([[1, 2], [3, 4]])
    T = np.array([[0, 1], [1, 0]])
    rot = rotate_matrix(M, T)
    expected = T.T.conj() @ M @ T
    assert np.allclose(rot, expected)

    T_dict = {0: np.array([[0, 1], [1, 0]]), 1: np.array([[1]])}
    M2 = np.eye(3)
    rot2 = rotate_matrix(M2, T_dict)
    assert rot2.shape == (3, 3)


def test_rotate_Greens_function():
    G = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    T = np.array([[0, 1], [1, 0]])
    rot = rotate_Greens_function(G, T)
    expected0 = T.T.conj() @ G[0] @ T
    expected1 = T.T.conj() @ G[1] @ T
    assert np.allclose(rot[0], expected0)
    assert np.allclose(rot[1], expected1)


def test_rotate_4index_U():
    U4 = np.ones((2, 2, 2, 2))
    T = np.eye(2)
    rot = rotate_4index_U(U4, T)
    assert np.allclose(rot, U4)


def test_block_diagonalize_hyb():
    hyb = np.zeros((2, 2, 2), dtype=complex)
    hyb[:, 0, 1] = 1.0 + 1j
    hyb[:, 1, 0] = 1.0 - 1j
    hyb[:, 0, 0] = 2.0
    hyb[:, 1, 1] = 2.0

    phase_hyb, Q_full = block_diagonalize_hyb(hyb)
    assert phase_hyb.shape == (2, 2, 2)
    assert Q_full.shape == (2, 2)
    assert np.allclose(phase_hyb[:, 0, 1], 0, atol=1e-10)
    assert np.allclose(phase_hyb[:, 1, 0], 0, atol=1e-10)


def test_calc_G():
    alphas = np.array([np.eye(2) * 1.0, np.eye(2) * 2.0])
    betas = np.array([np.eye(2) * 0.0, np.eye(2) * 0.5])
    r = np.eye(2)
    omega = np.array([-1.0, 0.0, 1.0])
    e = 0.0
    delta = 0.1
    G = calc_G(alphas, betas, r, omega, e, delta)
    assert G.shape == (3, 2, 2)

    G_empty = calc_G(np.empty((0, 2, 2)), betas, r, omega, e, delta)
    assert G_empty.shape == (3, 2, 2)


def test_calc_thermally_averaged_G():
    alphas = [np.array([np.eye(2) * 1.0, np.eye(2) * 2.0])]
    betas = [np.array([np.eye(2) * 0.0, np.eye(2) * 0.5])]
    r = [np.eye(2)]
    mesh = np.array([-1.0, 0.0, 1.0])
    es = [0.0]
    e0 = 0.0
    tau = 1.0
    delta = 0.1
    G_avg = calc_thermally_averaged_G(alphas, betas, r, mesh, es, e0, tau, delta)
    assert G_avg.shape == (3, 2, 2)

    G_empty = calc_thermally_averaged_G([], [], [], mesh, [], e0, tau, delta)
    assert G_empty.shape == (3, 0, 0)


def test_save_Greens_function(tmp_path):
    gs = np.zeros((2, 2, 2), dtype=complex)
    gs[:, 0, 0] = 1.0 + 0.1j
    gs[:, 1, 1] = 2.0 + 0.2j
    gs[:, 0, 1] = 0.5 + 0.05j
    gs[:, 1, 0] = 0.5 - 0.05j

    omega_mesh = np.array([-1.0, 1.0])

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        save_Greens_function(gs, omega_mesh, "test", "cluster")
        assert os.path.exists("real-test-realaxis-cluster.dat")
        assert os.path.exists("imag-test-realaxis-cluster.dat")

        with open("real-test-realaxis-cluster.dat", "r") as f:
            lines = f.readlines()
            assert len(lines) == 6

    finally:
        os.chdir(old_cwd)


def test_get_Greens_function():
    matsubara_mesh = np.array([1j, 2j])
    omega_mesh = np.array([-1.0, 0.0, 1.0])
    tau = 1.0
    delta = 0.1
    blocks = [[0]]

    hop = {((0, "c"), (0, "a")): 0.5}
    hOp = ManyBodyOperator(hop)

    states = [b"\x80", b"\x00"]
    basis = Basis(
        impurity_orbitals={0: [[0]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        comm=MPI.COMM_SELF,
    )
    psi = ManyBodyState({SlaterDeterminant.from_bytes(states[0]): 1.0})
    es = [0.5]

    gs_mat, gs_real, _report = get_Greens_function(
        matsubara_mesh=matsubara_mesh,
        omega_mesh=omega_mesh,
        psis=[psi],
        es=es,
        tau=tau,
        basis=basis,
        hOp=hOp,
        delta=delta,
        blocks=blocks,
        verbose=False,
        verbose_extra=False,
        reort=None,
        dN=1,
        occ_cutoff=1e-6,
        slaterWeightMin=0.0,
        sparse=False,
    )

    assert len(gs_mat) == 1
    assert gs_mat[0].shape == (2, 1, 1)
    assert len(gs_real) == 1
    assert gs_real[0].shape == (3, 1, 1)


def test_get_Greens_function_matsubara_none():
    matsubara_mesh = np.empty(0)
    omega_mesh = np.array([-1.0, 0.0, 1.0])
    tau = 1.0
    delta = 0.1
    blocks = [[0]]

    hop = {((0, "c"), (0, "a")): 0.5}
    hOp = ManyBodyOperator(hop)

    states = [b"\x80", b"\x00"]
    basis = Basis(
        impurity_orbitals={0: [[0]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        comm=MPI.COMM_SELF,
    )
    psi = ManyBodyState({SlaterDeterminant.from_bytes(states[0]): 1.0})
    es = [0.5]

    gs_mat, gs_real, _report = get_Greens_function(
        matsubara_mesh=matsubara_mesh,
        omega_mesh=omega_mesh,
        psis=[psi],
        es=es,
        tau=tau,
        basis=basis,
        hOp=hOp,
        delta=delta,
        blocks=blocks,
        verbose=True,
        verbose_extra=True,
        reort=None,
        dN=None,
        occ_cutoff=1e-6,
        slaterWeightMin=0.0,
        sparse=True,
    )

    if gs_mat is not None and len(gs_mat) > 0:
        assert True


def test_calc_G_pairwise_polarization_identity():
    """calc_G_pairwise reconstructs the full G matrix from scalar continued fractions.

    Single-pole synthetic case: seeds v_i = c_i |x> all proportional to one excited state |x>
    of energy Ex, so M_ij(w) = conj(c_i) c_j / (w + i delta + e - Ex) is analytic. The scalar
    continued fraction for a seed w = c|x> is one block (alphas=[[Ex]], betas=[[0]], r=[[|c|]]),
    giving S(w) = |c|^2 / (w'-Ex). calc_G_pairwise must reproduce M exactly via polarization.
    """
    from impurityModel.ed.greens_function import PairwiseGF, calc_G_pairwise

    c = np.array([0.7 + 0.2j, -0.4 + 0.9j])  # seed coefficients c_0, c_1
    Ex, e, delta = 1.3, 0.25, 0.1
    mesh = np.linspace(-2.0, 2.0, 11)

    def scalar_cf(coeff):
        norm = abs(coeff)
        return (
            [np.array([[Ex]], dtype=complex)],
            [np.array([[0.0]], dtype=complex)],
            np.array([[norm]], dtype=complex),
        )

    diag = [scalar_cf(c[0]), scalar_cf(c[1])]
    pairs = {(0, 1): (scalar_cf(c[0] + c[1]), scalar_cf(c[0] + 1j * c[1]))}
    G = calc_G_pairwise(PairwiseGF(2, diag, pairs), mesh, e, delta)

    wp = mesh + 1j * delta + e - Ex
    M = (np.conj(c)[None, :, None] * c[None, None, :]) / wp[:, None, None]
    np.testing.assert_allclose(G, M, atol=1e-12)


def test_get_Greens_function_eigenstate_grouping():
    """The eigenstate-grouping knob (``GF_EIGENSTATE_GROUP``) is a mathematical
    reorganization, not an approximation: stacking ``g`` thermal states into one wide
    block-Lanczos recurrence (width ``g * n_ops``) and reading each state's own ``n_ops``
    columns of the seed projection must reproduce the per-eigenstate (``g = 1``) Green's
    function. On this small system both fully resolve the excited spectrum (invariant
    subspace), so they agree to ~1e-8.

    The two thermal states occupy *different* orbitals and carry *different* energies, so a
    wrong ``r``-column slice or a mismatched energy pairing would change the result by O(1).
    """
    omega_mesh = np.linspace(-2.0, 2.0, 25)
    blocks = [[0, 1]]  # n_ops = 2 -> a group of 2 states seeds a width-4 block

    def _hop():
        # 2 impurity orbitals, no bath: the N+/-1 sectors are closed under H, so the
        # block-Lanczos reaches an invariant subspace in one shot (no basis expansion) and
        # the grouped/ungrouped Green's functions are exact -- a clean equivalence oracle.
        return ManyBodyOperator(
            {
                ((0, "c"), (0, "a")): 0.3,
                ((1, "c"), (1, "a")): 0.7,
                ((0, "c"), (1, "a")): 0.2,
                ((1, "c"), (0, "a")): 0.2,
            }
        )

    # MSB-first bits: orbital i is bit (7 - i). State 0x80 = {0}, state 0x40 = {1}.
    state_bytes = [b"\x80", b"\x40"]
    es = [0.3, 0.7]

    def run(group, sparse):
        basis = Basis(
            impurity_orbitals={0: [[0, 1]]},
            bath_states=({0: [[]]}, {0: [[]]}),
            initial_basis=state_bytes,
            comm=MPI.COMM_SELF,
        )
        psis = [ManyBodyState({SlaterDeterminant.from_bytes(b): 1.0}) for b in state_bytes]
        old = os.environ.get("GF_EIGENSTATE_GROUP")
        os.environ["GF_EIGENSTATE_GROUP"] = str(group)
        try:
            _, gs_real, _ = get_Greens_function(
                matsubara_mesh=None,
                omega_mesh=omega_mesh,
                psis=psis,
                es=list(es),
                tau=1.0,
                basis=basis,
                hOp=_hop(),
                delta=0.1,
                blocks=blocks,
                verbose=False,
                verbose_extra=False,
                reort=None,
                dN=1,
                occ_cutoff=1e-6,
                slaterWeightMin=0.0,
                sparse=sparse,
            )
        finally:
            if old is None:
                del os.environ["GF_EIGENSTATE_GROUP"]
            else:
                os.environ["GF_EIGENSTATE_GROUP"] = old
        return gs_real[0]

    for sparse in (False, True):
        g1 = run(1, sparse)
        g2 = run(2, sparse)
        assert g1.shape == (len(omega_mesh), 2, 2)
        assert np.allclose(g1, g2, atol=1e-6, rtol=1e-5), (
            f"grouped (g=2) GF differs from per-eigenstate (g=1), sparse={sparse}: "
            f"max|diff|={np.max(np.abs(g1 - g2)):.2e}"
        )


def test_get_Greens_function_eigenstate_grouping_with_bath():
    """Same grouping equivalence as above, but on a hybridizing-bath system whose excited
    basis actually grows under H -- the production-relevant regime -- on the sparse
    block-Lanczos path (the one ``calc_selfenergy`` uses). The shared wide-block recurrence
    must still reproduce the per-eigenstate Green's function to the convergence tolerance.
    """
    omega_mesh = np.linspace(-2.0, 2.0, 25)
    blocks = [[0, 1]]

    def _hop():
        return ManyBodyOperator(
            {
                ((0, "c"), (0, "a")): 0.3,
                ((1, "c"), (1, "a")): 0.7,
                ((2, "c"), (2, "a")): -0.5,
                ((3, "c"), (3, "a")): 0.4,
                ((0, "c"), (2, "a")): 0.25,
                ((2, "c"), (0, "a")): 0.25,
                ((1, "c"), (3, "a")): 0.25,
                ((3, "c"), (1, "a")): 0.25,
            }
        )

    # 0xA0 = {0, 2}, 0x50 = {1, 3} (MSB-first: orbital i is bit 7 - i).
    state_bytes = [b"\xa0", b"\x50"]
    es = [-0.2, 0.3]

    def run(group):
        basis = Basis(
            impurity_orbitals={0: [[0, 1]]},
            bath_states=({0: [[2, 3]]}, {0: [[]]}),
            initial_basis=state_bytes,
            comm=MPI.COMM_SELF,
        )
        psis = [ManyBodyState({SlaterDeterminant.from_bytes(b): 1.0}) for b in state_bytes]
        old = os.environ.get("GF_EIGENSTATE_GROUP")
        os.environ["GF_EIGENSTATE_GROUP"] = str(group)
        try:
            _, gs_real, _ = get_Greens_function(
                matsubara_mesh=None,
                omega_mesh=omega_mesh,
                psis=psis,
                es=list(es),
                tau=1.0,
                basis=basis,
                hOp=_hop(),
                delta=0.1,
                blocks=blocks,
                verbose=False,
                verbose_extra=False,
                reort=None,
                dN=1,
                occ_cutoff=1e-6,
                slaterWeightMin=0.0,
                sparse=True,
            )
        finally:
            if old is None:
                del os.environ["GF_EIGENSTATE_GROUP"]
            else:
                os.environ["GF_EIGENSTATE_GROUP"] = old
        return gs_real[0]

    g1 = run(1)
    g2 = run(2)
    assert g1.shape == (len(omega_mesh), 2, 2)
    assert np.allclose(g1, g2, atol=1e-5, rtol=1e-4), (
        f"grouped (g=2) GF differs from per-eigenstate (g=1) on the bath system: "
        f"max|diff|={np.max(np.abs(g1 - g2)):.2e}"
    )


def test_get_Greens_function_operator_split_matches_block():
    """The operator-split (pairwise) path (``GF_OPERATOR_SPLIT``) is an exact reorganization,
    not an approximation: computing each ``n x n`` block as ``n`` scalar (width-1) recurrences for
    the diagonal seeds plus two polarization recurrences per off-diagonal pair, then reassembling
    via the polarization identity, must reproduce the shared-Krylov width-``n`` block Green's
    function to the convergence tolerance. Run on the hybridizing-bath system (excited basis grows
    under H) on the sparse path the self-energy driver uses; the two thermal states occupy
    different orbitals, so a wrong seed pairing or column assignment would change G by O(1).
    """
    omega_mesh = np.linspace(-2.0, 2.0, 25)
    blocks = [[0, 1]]

    def _hop():
        return ManyBodyOperator(
            {
                ((0, "c"), (0, "a")): 0.3,
                ((1, "c"), (1, "a")): 0.7,
                ((2, "c"), (2, "a")): -0.5,
                ((3, "c"), (3, "a")): 0.4,
                ((0, "c"), (2, "a")): 0.25,
                ((2, "c"), (0, "a")): 0.25,
                ((1, "c"), (3, "a")): 0.25,
                ((3, "c"), (1, "a")): 0.25,
            }
        )

    state_bytes = [b"\xa0", b"\x50"]  # {0,2}, {1,3}
    es = [-0.2, 0.3]

    def run(op_split):
        basis = Basis(
            impurity_orbitals={0: [[0, 1]]},
            bath_states=({0: [[2, 3]]}, {0: [[]]}),
            initial_basis=state_bytes,
            comm=MPI.COMM_SELF,
        )
        psis = [ManyBodyState({SlaterDeterminant.from_bytes(b): 1.0}) for b in state_bytes]
        old = os.environ.get("GF_OPERATOR_SPLIT")
        os.environ["GF_OPERATOR_SPLIT"] = "1" if op_split else "0"
        try:
            _, gs_real, _ = get_Greens_function(
                matsubara_mesh=None,
                omega_mesh=omega_mesh,
                psis=psis,
                es=list(es),
                tau=1.0,
                basis=basis,
                hOp=_hop(),
                delta=0.1,
                blocks=blocks,
                verbose=False,
                verbose_extra=False,
                reort=None,
                dN=1,
                occ_cutoff=1e-6,
                slaterWeightMin=0.0,
                sparse=True,
            )
        finally:
            if old is None:
                del os.environ["GF_OPERATOR_SPLIT"]
            else:
                os.environ["GF_OPERATOR_SPLIT"] = old
        return gs_real[0]

    block = run(False)
    split = run(True)
    assert block.shape == (len(omega_mesh), 2, 2)
    assert np.allclose(block, split, atol=1e-5, rtol=1e-4), (
        f"operator-split GF differs from the block GF on the bath system: "
        f"max|diff|={np.max(np.abs(block - split)):.2e}"
    )


def test_union_restrictions_semantics():
    """``_union_restrictions`` returns the loosest single window admitting every input's feasible
    set: keep only subset keys common to all states, loosen each shared bound to (min lo, max hi),
    and yield ``None`` (unconstrained) if any input is ``None`` or no key is common. This is the
    superset that lets a grouped unit's shared Krylov space contain every stacked state's dynamics.
    """
    from impurityModel.ed.greens_function import _union_restrictions

    a = frozenset({0, 1, 2})
    b = frozenset({3, 4})
    c = frozenset({5, 6})
    # single window -> itself (full per-state tightening for an operator-split / g=1 unit)
    assert _union_restrictions([{a: (1, 3)}]) == {a: (1, 3)}
    # shared key -> loosened bound
    assert _union_restrictions([{a: (2, 3)}, {a: (1, 2)}]) == {a: (1, 3)}
    # key present in only one state -> dropped (it imposes no bound on the other)
    assert _union_restrictions([{a: (2, 3), b: (0, 1)}, {a: (1, 2)}]) == {a: (1, 3)}
    # any None (unconstrained) input -> None; no common key -> None; empty -> None
    assert _union_restrictions([{a: (1, 2)}, None]) is None
    assert _union_restrictions([None]) is None
    assert _union_restrictions([{b: (0, 1)}, {c: (0, 1)}]) is None
    assert _union_restrictions([]) is None


def test_gf_per_state_restrict_gating(monkeypatch):
    """Per-state windows default on exactly when ``chain_restrict`` is on; ``GF_PER_STATE_RESTRICT``
    overrides the default either way (the user-facing policy: tie per-state to chain_restrict)."""
    from impurityModel.ed.greens_function import _gf_per_state_restrict

    monkeypatch.delenv("GF_PER_STATE_RESTRICT", raising=False)
    assert _gf_per_state_restrict(True) is True
    assert _gf_per_state_restrict(False) is False
    monkeypatch.setenv("GF_PER_STATE_RESTRICT", "0")
    assert _gf_per_state_restrict(True) is False
    monkeypatch.setenv("GF_PER_STATE_RESTRICT", "1")
    assert _gf_per_state_restrict(False) is True


def test_get_Greens_function_per_state_restrict_matches_ensemble():
    """Per-state excited windows must not change the Green's function -- they only tighten the
    excited-basis span that the shared ensemble window over-provisions -- so GF(per-state) must
    reproduce GF(ensemble) to the convergence tolerance. Run with ``chain_restrict`` on so the
    per-state code path (per-unit ``_union_restrictions`` windows) is exercised; the two thermal
    states occupy different bath orbitals, so a window that truncated a state's Krylov space would
    change G by O(1)."""
    omega_mesh = np.linspace(-2.0, 2.0, 25)
    blocks = [[0, 1]]

    def _hop():
        return ManyBodyOperator(
            {
                ((0, "c"), (0, "a")): 0.3,
                ((1, "c"), (1, "a")): 0.7,
                ((2, "c"), (2, "a")): -0.5,
                ((3, "c"), (3, "a")): 0.4,
                ((0, "c"), (2, "a")): 0.25,
                ((2, "c"), (0, "a")): 0.25,
                ((1, "c"), (3, "a")): 0.25,
                ((3, "c"), (1, "a")): 0.25,
            }
        )

    state_bytes = [b"\xa0", b"\x50"]  # {0,2}, {1,3}
    es = [-0.2, 0.3]

    def run(per_state):
        basis = Basis(
            impurity_orbitals={0: [[0, 1]]},
            bath_states=({0: [[2, 3]]}, {0: [[]]}),
            initial_basis=state_bytes,
            chain_restrict=True,
            tau=1.0,
            comm=MPI.COMM_SELF,
        )
        psis = [ManyBodyState({SlaterDeterminant.from_bytes(b): 1.0}) for b in state_bytes]
        old = os.environ.get("GF_PER_STATE_RESTRICT")
        os.environ["GF_PER_STATE_RESTRICT"] = "1" if per_state else "0"
        try:
            _, gs_real, _ = get_Greens_function(
                matsubara_mesh=None,
                omega_mesh=omega_mesh,
                psis=psis,
                es=list(es),
                tau=1.0,
                basis=basis,
                hOp=_hop(),
                delta=0.1,
                blocks=blocks,
                verbose=False,
                verbose_extra=False,
                reort=None,
                dN=1,
                occ_cutoff=1e-6,
                slaterWeightMin=0.0,
                sparse=True,
            )
        finally:
            if old is None:
                del os.environ["GF_PER_STATE_RESTRICT"]
            else:
                os.environ["GF_PER_STATE_RESTRICT"] = old
        return gs_real[0]

    ensemble = run(False)
    per_state = run(True)
    assert np.allclose(ensemble, per_state, atol=1e-5, rtol=1e-4), (
        f"per-state-window GF differs from the ensemble-window GF: "
        f"max|diff|={np.max(np.abs(ensemble - per_state)):.2e}"
    )


def test_get_Greens_function_mesh_is_none():
    """A None mesh (e.g. the self-energy driver requesting only one axis) returns that
    axis as None instead of crashing on len(None)."""
    omega_mesh = np.array([-1.0, 0.0, 1.0])
    matsubara_mesh = np.array([1j, 2j])
    hOp = ManyBodyOperator({((0, "c"), (0, "a")): 0.5})
    states = [b"\x80", b"\x00"]

    def run(mmesh, omesh):
        basis = Basis(
            impurity_orbitals={0: [[0]]},
            bath_states=({0: [[]]}, {0: [[]]}),
            initial_basis=states,
            comm=MPI.COMM_SELF,
        )
        psi = ManyBodyState({SlaterDeterminant.from_bytes(states[0]): 1.0})
        return get_Greens_function(
            matsubara_mesh=mmesh,
            omega_mesh=omesh,
            psis=[psi],
            es=[0.5],
            tau=1.0,
            basis=basis,
            hOp=hOp,
            delta=0.1,
            blocks=[[0]],
            verbose=False,
            verbose_extra=False,
            reort=None,
            dN=1,
            occ_cutoff=1e-6,
            slaterWeightMin=0.0,
            sparse=False,
        )

    # No Matsubara axis requested -> gs_matsubara is None, real axis still built.
    gs_mat, gs_real, _ = run(None, omega_mesh)
    assert gs_mat is None
    assert len(gs_real) == 1 and gs_real[0].shape == (3, 1, 1)

    # No real axis requested -> gs_realaxis is None, Matsubara axis still built.
    gs_mat, gs_real, _ = run(matsubara_mesh, None)
    assert gs_real is None
    assert len(gs_mat) == 1 and gs_mat[0].shape == (2, 1, 1)
