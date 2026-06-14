import pytest
import numpy as np
from mpi4py import MPI
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, SlaterDeterminant
from impurityModel.ed.greens_function import (
    split_comm_and_redistribute_psi,
    calc_Greens_function_with_offdiag,
    get_Greens_function,
)

def test_basis_split_and_redistribute_serial():
    # Setup serial basis
    states = [b"\x80", b"\x40", b"\x20", b"\x10"]
    basis = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        comm=None,
    )
    
    # Test redistribute_psis (serial should return same psis)
    psi0 = [ManyBodyState({SlaterDeterminant.from_bytes(states[0]): 1.0})]
    redist_psi = basis.redistribute_psis(psi0)
    assert redist_psi == psi0
    
    # Test split_basis_and_redistribute_psi
    priorities = [1, 2]
    indices, split_roots, color, items_per_color, split_basis, psis_out, intercomms = \
        basis.split_basis_and_redistribute_psi(priorities, psi0)
    
    assert list(indices) == [0, 1]
    assert split_roots == [0]
    assert color == 0
    assert list(items_per_color) == [2]
    assert split_basis == basis
    assert psis_out == psi0

@pytest.mark.mpi
def test_basis_split_and_redistribute_mpi():
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("This test requires at least 2 MPI ranks")
        
    states = [b"\x80", b"\x40", b"\x20", b"\x10"]
    basis = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        comm=comm,
    )
    
    # Test redistribute_psis
    # Let's create a psi where rank 0 has all states and rank 1 has empty
    if comm.rank == 0:
        psi0 = [ManyBodyState({
            SlaterDeterminant.from_bytes(s): 1.0 for s in states
        })]
    else:
        psi0 = [ManyBodyState({})]
        
    # Redistribute the psis
    redist_psi = basis.redistribute_psis(psi0)
    
    # Gather redistributed to check consistency
    gathered = comm.gather(redist_psi[0], root=0)
    if comm.rank == 0:
        combined = ManyBodyState()
        for p in gathered:
            combined += p
        # Total amplitude sum for each state should be 1.0
        for s in states:
            sd = SlaterDeterminant.from_bytes(s)
            assert np.allclose(combined[sd], 1.0)
            
    # Test split_basis_and_redistribute_psi
    # With priorities for 2 items, rank 0 should get item 0, rank 1 should get item 1
    priorities = [1.0, 1.0]
    indices, split_roots, color, items_per_color, split_basis, psis_out, intercomms = \
        basis.split_basis_and_redistribute_psi(priorities, psi0)
        
    assert len(indices) == 1
    assert indices[0] in [0, 1]
    assert len(split_roots) == 2
    assert items_per_color == [1, 1]
    expected_size = split_roots[color + 1] - split_roots[color] if color + 1 < len(split_roots) else comm.size - split_roots[color]
    assert split_basis.comm.size == expected_size
    if split_basis is not None and split_basis.comm != comm:
        split_basis.free_comm()
    split_basis = None
    intercomms = None

@pytest.mark.mpi
def test_split_comm_and_redistribute_psi_mpi():
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("This test requires at least 2 MPI ranks")
        
    states = [b"\x80", b"\x40", b"\x20", b"\x10"]
    if comm.rank == 0:
        psis = [ManyBodyState({
            SlaterDeterminant.from_bytes(s): 1.0 for s in states
        })]
    else:
        psis = [ManyBodyState({})]
        
    # Split comm for 2 items with equal priorities
    priorities = [1.0, 1.0]
    sub_slice, split_roots, color, items_per_color, split_comm, psis_out = \
        split_comm_and_redistribute_psi(priorities, psis, comm)
        
    expected_size = split_roots[color + 1] - split_roots[color] if color + 1 < len(split_roots) else comm.size - split_roots[color]
    assert split_comm.size == expected_size
    assert len(split_roots) == 2
    assert list(items_per_color) == [1, 1]
    # Check that psis are combined correctly on the roots of split comms
    assert len(psis_out) == 1
    if split_comm.rank == 0:
        for s in states:
            sd = SlaterDeterminant.from_bytes(s)
            assert np.allclose(psis_out[0][sd], 1.0)
    if split_comm is not None and split_comm != MPI.COMM_NULL and split_comm != comm:
        split_comm.Free()
    split_comm = None
    psis_out = None

def test_calc_Greens_function_with_offdiag_serial():
    # Setup simple Hamiltonian H = 0.5 * c_0^\dagger c_0
    eigvals = np.array([0.5])
    hop = {((0, "c"), (0, "a")): 0.5}
    hOp = ManyBodyOperator(hop)
    
    # Basis
    states = [b"\x80", b"\x00"] # 1 orbital, state occupancies: 1, 0
    basis = Basis(
        impurity_orbitals={0: [[0]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        comm=MPI.COMM_SELF,
    )
    
    # Ground state: |1> with energy 0.5
    psi = ManyBodyState({SlaterDeterminant.from_bytes(states[0]): 1.0})
    
    # tOp = c_0 (electron removal operator)
    tOp = ManyBodyOperator({((0, "a"),): 1.0})
    
    alphas, betas, r = calc_Greens_function_with_offdiag(
        hOp=hOp,
        tOps=[tOp],
        psis=[psi],
        es=[0.5],
        block_basis=basis,
        delta=0.01,
        dN=1,
        occ_cutoff=1e-6,
        slaterWeightMin=0.0,
        verbose=False,
        sparse=False,
    )
    
    # alphas should have shape (N, 1, 1) and betas (N-1, 1, 1)
    assert len(alphas) == 1
    assert alphas[0].shape == (1, 1, 1)
    assert betas[0].shape == (1, 1, 1)
    np.testing.assert_allclose(alphas[0], [[[0.0]]], atol=1e-12)
    np.testing.assert_allclose(betas[0], [[[0.0]]], atol=1e-12)
    np.testing.assert_allclose(r[0], [[1.0]], atol=1e-12)


@pytest.mark.mpi
def test_calc_Greens_function_with_offdiag_mpi():
    comm = MPI.COMM_WORLD
    # Setup simple Hamiltonian H = 0.5 * c_0^\dagger c_0
    eigvals = np.array([0.5])
    hop = {((0, "c"), (0, "a")): 0.5}
    hOp = ManyBodyOperator(hop)
    
    # Basis
    states = [b"\x80", b"\x00"]
    basis = Basis(
        impurity_orbitals={0: [[0]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        comm=comm,
    )
    
    # Ground state
    if comm.rank == 0:
        psi = ManyBodyState({SlaterDeterminant.from_bytes(states[0]): 1.0})
    else:
        psi = ManyBodyState({})
    psi = basis.redistribute_psis([psi])[0]
    
    # tOp
    tOp = ManyBodyOperator({((0, "a"),): 1.0})
    
    alphas, betas, r = calc_Greens_function_with_offdiag(
        hOp=hOp,
        tOps=[tOp],
        psis=[psi],
        es=[0.5],
        block_basis=basis,
        delta=0.01,
        dN=1,
        occ_cutoff=1e-6,
        slaterWeightMin=0.0,
        verbose=False,
        sparse=False,
    )
    
    if comm.rank == 0:
        assert len(alphas) == 1
        assert alphas[0].shape == (1, 1, 1)
        assert betas[0].shape == (1, 1, 1)
        np.testing.assert_allclose(alphas[0], [[[0.0]]], atol=1e-12)
        np.testing.assert_allclose(betas[0], [[[0.0]]], atol=1e-12)
        np.testing.assert_allclose(r[0], [[1.0]], atol=1e-12)

    basis = None


@pytest.mark.mpi
def test_calc_Greens_function_with_offdiag_mpi_sparse():
    comm = MPI.COMM_WORLD
    # Setup simple Hamiltonian H = 0.5 * c_0^\dagger c_0
    eigvals = np.array([0.5])
    hop = {((0, "c"), (0, "a")): 0.5}
    hOp = ManyBodyOperator(hop)
    
    # Basis
    states = [b"\x80", b"\x00"]
    basis = Basis(
        impurity_orbitals={0: [[0]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        comm=comm,
    )
    
    # Ground state
    if comm.rank == 0:
        psi = ManyBodyState({SlaterDeterminant.from_bytes(states[0]): 1.0})
    else:
        psi = ManyBodyState({})
    psi = basis.redistribute_psis([psi])[0]
    
    # tOp
    tOp = ManyBodyOperator({((0, "a"),): 1.0})
    
    alphas, betas, r = calc_Greens_function_with_offdiag(
        hOp=hOp,
        tOps=[tOp],
        psis=[psi],
        es=[0.5],
        block_basis=basis,
        delta=0.01,
        dN=1,
        occ_cutoff=1e-6,
        slaterWeightMin=0.0,
        verbose=False,
        sparse=True,
    )
    
    if comm.rank == 0:
        assert len(alphas) == 1
        assert alphas[0].shape == (1, 1, 1)
        assert betas[0].shape == (1, 1, 1)
        np.testing.assert_allclose(alphas[0], [[[0.0]]], atol=1e-12)
        np.testing.assert_allclose(betas[0], [[[0.0]]], atol=1e-6)
        np.testing.assert_allclose(r[0], [[1.0]], atol=1e-12)

    basis = None


@pytest.mark.mpi
def test_Green_freq_bicgstab_mpi():
    comm = MPI.COMM_WORLD
    # Setup simple Hamiltonian H = 0.5 * c_0^\dagger c_0
    eigvals = np.array([0.5])
    hop = {((0, "c"), (0, "a")): 0.5}
    hOp = ManyBodyOperator(hop)
    
    # Basis
    states = [b"\x80", b"\x00"]
    basis = Basis(
        impurity_orbitals={0: [[0]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        comm=comm,
    )
    
    # Ground state
    if comm.rank == 0:
        psi = ManyBodyState({SlaterDeterminant.from_bytes(states[0]): 1.0})
    else:
        psi = ManyBodyState({})
    psi = basis.redistribute_psis([psi])[0]
    
    # w_mesh
    w_mesh = np.array([0.1, 0.2])
    
    from impurityModel.ed.greens_function import Green_freq_bicgstab
    gs = Green_freq_bicgstab(w_mesh, hOp, [psi], 0.5, basis, 0.0)
    
    # Check shape
    if comm.rank == 0:
        assert gs.shape == (2, 1, 1)

    basis = None


@pytest.mark.mpi
def test_getRIXSmap_new_mpi():
    comm = MPI.COMM_WORLD
    # Setup simple Hamiltonian H = 0.5 * c_0^\dagger c_0 + 0.3 * c_1^\dagger c_1
    hop = {
        ((0, "c"), (0, "a")): 0.5,
        ((1, "c"), (1, "a")): 0.3,
    }
    hOp = ManyBodyOperator(hop)
    
    # Basis
    states = [b"\x80", b"\x40", b"\x20", b"\x10", b"\x00"]
    basis = Basis(
        impurity_orbitals={1: [[0]], 2: [[1]]},
        bath_states=(
            {1: [[]], 2: [[]]},
            {1: [[]], 2: [[]]},
        ),
        initial_basis=states,
        comm=comm,
    )
    
    # Ground state psi
    if comm.rank == 0:
        psi1 = ManyBodyState({SlaterDeterminant.from_bytes(states[0]): 1.0})
        psi2 = ManyBodyState({SlaterDeterminant.from_bytes(states[1]): 1.0})
    else:
        psi1 = ManyBodyState({})
        psi2 = ManyBodyState({})
    psis = basis.redistribute_psis([psi1, psi2])
    
    # wIns, wLoss, delta
    # Use multiple wIns and Es to ensure MPI distribution works correctly and does not overwrite intermediate results
    wIns = np.array([0.1, 0.2, 0.3, 0.4])
    wLoss = np.array([0.0])
    delta1 = 0.1
    delta2 = 0.1
    tau = 1.0
    
    # transition operators
    tOpsIn = [ManyBodyOperator({((0, "a"),): 1.0})]
    tOpsOut = [ManyBodyOperator({((0, "c"),): 1.0})]
    
    from impurityModel.ed.spectra import getRIXSmap_new
    gs = getRIXSmap_new(
        hOp=hOp,
        tOpsIn=tOpsIn,
        tOpsOut=tOpsOut,
        psis=psis,
        Es=np.array([0.5, 0.6]),
        tau=tau,
        wIns=wIns,
        wLoss=wLoss,
        delta1=delta1,
        delta2=delta2,
        basis=basis,
        verbose=False,
        slaterWeightMin=0.0,
    )
    
    if comm.rank == 0:
        assert gs.shape == (1, 1, 4, 1)
        expected_gs = np.array([-3.15050801e-14-14.18862669j,  1.74853194e-14-10.49958375j,
                               -8.96683048e-15 -8.07660288j,  0.00000000e+00 -6.40218521j])
        np.testing.assert_allclose(gs.flatten(), expected_gs, atol=1e-13)

    basis = None



