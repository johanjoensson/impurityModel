import pytest
import numpy as np
from mpi4py import MPI
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant
from impurityModel.ed.groundstate import calc_gs
from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.density_matrix import calc_density_matrices

@pytest.mark.mpi
def test_groundstate_and_density_matrix_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # 10-orbital system with 5 electrons
    impurity_orbitals = {0: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]}
    bath_states = ({0: [[]]}, {0: [[]]})
    nominal_occ = {0: 5}
    mixed_valence = {0: 0}
    tau = 0.01

    # Hamiltonian with deterministic random hoppings and on-site energies
    np.random.seed(42)
    h_dict = {}
    for i in range(10):
        h_dict[((i, "c"), (i, "a"))] = np.random.uniform(-2, 2)
        for j in range(i + 1, 10):
            val = np.random.uniform(-1, 1)
            h_dict[((i, "c"), (j, "a"))] = val
            h_dict[((j, "c"), (i, "a"))] = val

    Hop = ManyBodyOperator(h_dict)

    block_structure = BlockStructure(
        blocks=[list(range(10))],
        identical_blocks=[[0]],
        transposed_blocks=[[]],
        particle_hole_blocks=[[]],
        particle_hole_transposed_blocks=[[]],
        inequivalent_blocks=[0],
    )

    rot_to_spherical = np.eye(10, dtype=complex)

    # --- 1. Parallel MPI Calculation ---
    basis_setup_mpi = {
        "impurity_orbitals": impurity_orbitals,
        "bath_states": bath_states,
        "N0": nominal_occ,
        "mixed_valence": mixed_valence,
        "tau": tau,
        "chain_restrict": False,
        "dense_cutoff": 10,
        "spin_flip_dj": False,
        "comm": comm,
        "truncation_threshold": 1000,
    }

    psis_mpi, es_mpi, basis_mpi, rho_mpi, gs_info_mpi = calc_gs(
        Hop, basis_setup_mpi, block_structure, rot_to_spherical, verbose=False, slaterWeightMin=1e-12
    )

    # Gather parallel wavefunction to Rank 0
    gathered_psis = comm.gather(psis_mpi, root=0)

    if rank == 0:
        full_psi_mpi = ManyBodyState()
        for r_psis in gathered_psis:
            full_psi_mpi += r_psis[0]

        # --- 2. Serial Calculation ---
        basis_setup_seq = {
            "impurity_orbitals": impurity_orbitals,
            "bath_states": bath_states,
            "N0": nominal_occ,
            "mixed_valence": mixed_valence,
            "tau": tau,
            "chain_restrict": False,
            "dense_cutoff": 10,
            "spin_flip_dj": False,
            "comm": None,  # Serial
            "truncation_threshold": 1000,
        }
        
        psis_seq, es_seq, basis_seq, rho_seq, gs_info_seq = calc_gs(
            Hop, basis_setup_seq, block_structure, rot_to_spherical, verbose=False, slaterWeightMin=1e-12
        )
        full_psi_seq = psis_seq[0]

        # Compare energies
        np.testing.assert_allclose(es_mpi, es_seq, rtol=1e-10, atol=1e-10)

        # Compare density matrices
        np.testing.assert_allclose(rho_mpi, rho_seq, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(gs_info_mpi["rhos"], gs_info_seq["rhos"], rtol=1e-10, atol=1e-10)

        # Compare wavefunctions (up to a global phase)
        dict_mpi = full_psi_mpi.to_dict()
        dict_seq = full_psi_seq.to_dict()

        assert set(dict_mpi.keys()) == set(dict_seq.keys())
        
        ratios = []
        for k in dict_seq:
            val_mpi = dict_mpi[k]
            val_seq = dict_seq[k]
            if abs(val_seq) > 1e-8:
                ratios.append(val_mpi / val_seq)

        # Ratios should have magnitude 1 and be identical (representing same global phase shift)
        if ratios:
            first_ratio = ratios[0]
            for r in ratios:
                np.testing.assert_allclose(abs(r), 1.0, atol=1e-8)
                np.testing.assert_allclose(r, first_ratio, atol=1e-8)

        # Also test the general rectangular case of build_density_matrices
        rect_rho_mpi = basis_mpi.build_density_matrices(psis_mpi, [0, 1, 2], [3, 4, 5])
        rect_rho_seq = basis_seq.build_density_matrices(psis_seq, [0, 1, 2], [3, 4, 5])
        np.testing.assert_allclose(rect_rho_mpi, rect_rho_seq, rtol=1e-10, atol=1e-10)


def test_groundstate_and_density_matrix_serial():
    # 10-orbital system with 5 electrons
    impurity_orbitals = {0: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]}
    bath_states = ({0: [[]]}, {0: [[]]})
    nominal_occ = {0: 5}
    mixed_valence = {0: 0}
    tau = 0.01

    np.random.seed(42)
    h_dict = {}
    for i in range(10):
        h_dict[((i, "c"), (i, "a"))] = np.random.uniform(-2, 2)
        for j in range(i + 1, 10):
            val = np.random.uniform(-1, 1)
            h_dict[((i, "c"), (j, "a"))] = val
            h_dict[((j, "c"), (i, "a"))] = val

    Hop = ManyBodyOperator(h_dict)

    block_structure = BlockStructure(
        blocks=[list(range(10))],
        identical_blocks=[[0]],
        transposed_blocks=[[]],
        particle_hole_blocks=[[]],
        particle_hole_transposed_blocks=[[]],
        inequivalent_blocks=[0],
    )

    rot_to_spherical = np.eye(10, dtype=complex)

    basis_setup = {
        "impurity_orbitals": impurity_orbitals,
        "bath_states": bath_states,
        "N0": nominal_occ,
        "mixed_valence": mixed_valence,
        "tau": tau,
        "chain_restrict": False,
        "dense_cutoff": 10,
        "spin_flip_dj": False,
        "comm": None,
        "truncation_threshold": 1000,
    }

    psis, es, basis, rho, gs_info = calc_gs(
        Hop, basis_setup, block_structure, rot_to_spherical, verbose=False, slaterWeightMin=1e-12
    )

    # Basic assertions
    assert len(es) > 0
    assert len(psis) > 0
    assert rho.shape == (10, 10)
    assert np.allclose(rho, rho.conj().T, atol=1e-12)
    
    # Tr(rho) should equal number of electrons (5)
    np.testing.assert_allclose(rho.trace().real, 5.0, atol=1e-10)

    # Test build_density_matrices
    rho_basis = basis.build_density_matrices(psis, list(range(10)), list(range(10)))
    np.testing.assert_allclose(rho_basis[0], rho, rtol=1e-10, atol=1e-10)
