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
    assert split_basis.comm.size == 1

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
        
    assert split_comm.size == 1
    assert len(split_roots) == 2
    assert list(items_per_color) == [1, 1]
    # Check that psis are combined correctly on the roots of split comms
    assert len(psis_out) == 1
    for s in states:
        sd = SlaterDeterminant.from_bytes(s)
        assert np.allclose(psis_out[0][sd], 1.0)

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
    psi = ManyBodyState({SlaterDeterminant.from_bytes(states[0]): 1.0})
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
