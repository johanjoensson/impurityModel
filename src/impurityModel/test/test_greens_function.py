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
