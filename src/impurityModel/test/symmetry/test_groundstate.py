import numpy as np
import pytest
from mpi4py import MPI

import impurityModel.ed.product_state_representation as psr
from impurityModel.ed import groundstate
from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.density_matrices import build_density_matrices
from impurityModel.ed.eigensolvers import Lowest
from impurityModel.ed.groundstate import calc_energy, calc_gs, find_ground_state_sector, solve_sector
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.test.support.comms import COMMS


def ground_state_support(basis, h_op, dense_cutoff=10):
    """Occupied-orbital tuples carrying amplitude in the ground state on ``basis``.

    A solved sector's basis is returned whole; it is not collapsed onto the
    eigenvector support (that cut determinants at ``sqrt(slaterWeightMin)`` before the final
    expansion could use them, and made a cached basis path-dependent). So a test that means
    "the search found *this* ground state" has to solve for the eigenvector and look at its
    support, rather than read the basis size.
    """
    solver = CIPSISolver(basis)
    _es, psis = solver.get_eigenvectors(h_op, Lowest(), dense_cutoff=dense_cutoff)
    # The state is defined on the whole basis, so iterating it yields every determinant
    # including the ones with zero amplitude; the support is the ones actually carrying weight.
    local = {
        psr.bytes2tuple(bytes(state.to_bytearray())[:8], 64)
        for state, amplitude in (psis.column(0) if hasattr(psis, "width") else psis[0]).items()
        if np.linalg.norm(amplitude) > 1e-8
    }
    if not basis.is_distributed:
        return local
    # Determinants are hash-distributed, one owner each, so each rank sees only its own slice of
    # the support -- a rank owning none of it would read an empty set as "wrong ground state".
    # Allgather (collective, and every rank reaches it) and take the union.
    return set().union(*basis.comm.allgather(local))


def _solved_basis(search, h_op, comm):
    """The basis of the sector a walk settled on, solved for its lowest state."""
    return solve_sector(
        h_op,
        {0: [[0, 1, 2, 3, 4]]},
        ({0: [[]]}, {0: [[]]}),
        search.occupation,
        tau=0.01,
        chain_restrict=False,
        dense_cutoff=10,
        manifold=Lowest(),
        comm=comm,
        verbose=False,
        truncation_threshold=1000,
        slaterWeightMin=1e-12,
    ).basis


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
        "comm": comm,
        "truncation_threshold": 1000,
    }

    psis_mpi, es_mpi, basis_mpi, rho_mpi, gs_info_mpi = calc_gs(
        Hop, basis_setup_mpi, block_structure, rot_to_spherical, verbose=True, slaterWeightMin=1e-12
    )

    # Gather parallel wavefunction to Rank 0
    gathered_psis = comm.gather(psis_mpi, root=0)

    # Also test the general rectangular case of build_density_matrices
    rect_rho_mpi = build_density_matrices(basis_mpi, psis_mpi, [0, 1, 2], [3, 4, 5])

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
            "comm": None,  # Serial
            "truncation_threshold": 1000,
        }

        psis_seq, es_seq, basis_seq, rho_seq, gs_info_seq = calc_gs(
            Hop, basis_setup_seq, block_structure, rot_to_spherical, verbose=True, slaterWeightMin=1e-12
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
        rect_rho_seq = build_density_matrices(basis_seq, psis_seq, [0, 1, 2], [3, 4, 5])
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
        "comm": None,
        "truncation_threshold": 1000,
    }

    psis, es, basis, rho, _gs_info = calc_gs(
        Hop, basis_setup, block_structure, rot_to_spherical, verbose=True, slaterWeightMin=1e-12
    )

    # Basic assertions
    assert len(es) > 0
    assert len(psis) > 0
    assert rho.shape == (10, 10)
    assert np.allclose(rho, rho.conj().T, atol=1e-12)

    # Tr(rho) should equal number of electrons (5)
    np.testing.assert_allclose(rho.trace().real, 5.0, atol=1e-10)

    # Test build_density_matrices
    rho_basis = build_density_matrices(basis, psis, list(range(10)), list(range(10)))
    np.testing.assert_allclose(rho_basis[0], rho, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("comm", COMMS)
def test_calc_energy(comm):
    eigvals = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    hop = {((i, "c"), (i, "a")): val for i, val in enumerate(eigvals)}
    Hop = ManyBodyOperator(hop)

    energy, report, *_ = calc_energy(
        h_op=Hop,
        impurity_indices={0: [[0, 1, 2, 3, 4]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        N0={0: 2},
        chain_restrict=False,
        dense_cutoff=10,
        comm=comm,
        verbose=True,
        truncation_threshold=1000,
        slaterWeightMin=1e-12,
    )
    np.testing.assert_allclose(energy, 1.5, rtol=1e-10, atol=1e-10)
    assert report is None  # the cap never bound


@pytest.mark.parametrize("comm", COMMS)
def test_find_ground_state_sector(comm):
    # Two bound (negative) levels below a gap, three unbound (positive) levels above it, so
    # N=2 (filling the two negative levels) is a genuine local -- and here global -- minimum in
    # total particle number, not merely the value nearest the input N0. (An all-positive-spectrum
    # model would make the unconstrained many-body ground state the vacuum N=0 regardless of
    # N0, which find_ground_state_sector's neighbourhood walk -- HF-seeded or the legacy dN scan
    # alike -- correctly finds; that is not what this test means to exercise.)
    eigvals = np.array([-1.0, -0.5, 0.5, 1.0, 1.5])
    hop = {((i, "c"), (i, "a")): val for i, val in enumerate(eigvals)}
    Hop = ManyBodyOperator(hop)

    search = find_ground_state_sector(
        h_op=Hop,
        impurity_orbitals={0: [[0, 1, 2, 3, 4]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        N0={0: 2},
        chain_restrict=False,
        dense_cutoff=10,
        comm=comm,
        verbose=True,
        truncation_threshold=1000,
        slaterWeightMin=1e-12,
    )
    # N0={0:2}: the two-electron ground state fills the two lowest (negative-energy) orbitals
    # (-1.0 + -0.5 = -1.5). The Hamiltonian is diagonal, so that single determinant carries the
    # whole ground state.
    assert search.energy == pytest.approx(-1.5)
    assert search.occupation == {0: 2}
    assert ground_state_support(_solved_basis(search, Hop, comm), Hop) == {(0, 1)}


def test_find_ground_state_sector_walk_rescues_seed_off_by_two(monkeypatch):
    # Same model as test_find_ground_state_sector: N=2 (filling the two negative-energy
    # levels) is the true minimum. Force the HF seed to miss by 2 (N0=0, the vacuum) -- the
    # scenario the NiO bug report hit (HF pushed 3 electrons into the wrong sector) -- and assert
    # the post-seed walk still corrects it back to the true minimum, matching what the legacy
    # use_hf_seed=False dN scan (the oracle) finds unassisted.
    eigvals = np.array([-1.0, -0.5, 0.5, 1.0, 1.5])
    hop = {((i, "c"), (i, "a")): val for i, val in enumerate(eigvals)}
    Hop = ManyBodyOperator(hop)

    def bad_seed(h_op, impurity_orbitals, bath_states, N0, frozen_occupations=None, comm=None, verbose=False):
        return {0: 0}, True

    monkeypatch.setattr(groundstate, "hartree_fock_seed_occupation", bad_seed)

    common_kwargs = dict(
        h_op=Hop,
        impurity_orbitals={0: [[0, 1, 2, 3, 4]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        N0={0: 0},
        chain_restrict=False,
        dense_cutoff=10,
        comm=None,
        verbose=True,
        truncation_threshold=1000,
        slaterWeightMin=1e-12,
    )

    walked = find_ground_state_sector(use_hf_seed=True, **common_kwargs)
    scanned = find_ground_state_sector(use_hf_seed=False, **common_kwargs)

    for search in (walked, scanned):
        assert search.occupation == {0: 2}
        assert ground_state_support(_solved_basis(search, Hop, None), Hop) == {(0, 1)}


def test_calc_gs_options_serial():
    # Test with the chain_restrict option enabled in serial
    # We shift the eigenvalues so that N=2 is the true global ground state.
    # Energy of N=2 is -1.5 + -1.0 = -2.5. Energy of N=1 is -1.5. Energy of N=3 is -2.5 + 1.5 = -1.0.
    eigvals = np.array([-1.5, -1.0, 1.5, 2.0, 2.5])
    hop = {((i, "c"), (i, "a")): val for i, val in enumerate(eigvals)}
    Hop = ManyBodyOperator(hop)

    basis_setup = {
        "impurity_orbitals": {0: [[0, 1, 2, 3, 4]]},
        "bath_states": ({0: [[]]}, {0: [[]]}),
        "nominal_impurity_occ": {0: 2},
        "tau": 0.01,
        "chain_restrict": True,  # Enable chain restriction
        "dense_cutoff": 10,
        "comm": None,
        "truncation_threshold": 1000,
    }
    block_structure = BlockStructure(
        blocks=[[0, 1, 2, 3, 4]],
        identical_blocks=[[0]],
        transposed_blocks=[[]],
        particle_hole_blocks=[[]],
        particle_hole_transposed_blocks=[[]],
        inequivalent_blocks=[0],
    )
    rot_to_spherical = np.eye(5, dtype=complex)

    psis, es, _basis, thermal_rho, _gs_info = calc_gs(
        Hop,
        basis_setup,
        block_structure,
        rot_to_spherical,
        verbose=True,
        slaterWeightMin=1e-12,
    )
    assert len(es) > 0
    assert len(psis) > 0
    assert thermal_rho.shape == (5, 5)
    np.testing.assert_allclose(es, [-2.5], atol=1e-10)
    expected_rho = np.zeros((5, 5))
    expected_rho[0, 0] = 1.0
    expected_rho[1, 1] = 1.0
    np.testing.assert_allclose(thermal_rho, expected_rho, atol=1e-10)


@pytest.mark.mpi
def test_calc_gs_options_mpi():
    comm = MPI.COMM_WORLD
    # We shift the eigenvalues so that N=2 is the true global ground state.
    eigvals = np.array([-1.5, -1.0, 1.5, 2.0, 2.5])
    hop = {((i, "c"), (i, "a")): val for i, val in enumerate(eigvals)}
    Hop = ManyBodyOperator(hop)

    basis_setup = {
        "impurity_orbitals": {0: [[0, 1, 2, 3, 4]]},
        "bath_states": ({0: [[]]}, {0: [[]]}),
        "nominal_impurity_occ": {0: 2},
        "tau": 0.01,
        "chain_restrict": True,
        "dense_cutoff": 10,
        "comm": comm,
        "truncation_threshold": 1000,
    }
    block_structure = BlockStructure(
        blocks=[[0, 1, 2, 3, 4]],
        identical_blocks=[[0]],
        transposed_blocks=[[]],
        particle_hole_blocks=[[]],
        particle_hole_transposed_blocks=[[]],
        inequivalent_blocks=[0],
    )
    rot_to_spherical = np.eye(5, dtype=complex)

    psis, es, _basis, thermal_rho, _gs_info = calc_gs(
        Hop, basis_setup, block_structure, rot_to_spherical, verbose=True, slaterWeightMin=1e-12
    )
    assert len(es) > 0
    assert len(psis) > 0
    assert thermal_rho.shape == (5, 5)
    np.testing.assert_allclose(es, [-2.5], atol=1e-10)
    expected_rho = np.zeros((5, 5))
    expected_rho[0, 0] = 1.0
    expected_rho[1, 1] = 1.0
    np.testing.assert_allclose(thermal_rho, expected_rho, atol=1e-10)


class _Captured(Exception):
    pass


class _Layout:
    def provenance_note(self):
        return None


def _capture_solve_kwargs(monkeypatch, **call):
    from impurityModel.ed import groundstate

    seen = {}

    def fake(h_op, **kwargs):
        seen.update(kwargs)
        raise _Captured

    monkeypatch.setattr(groundstate, "solve_ground_state", fake)
    with pytest.raises(_Captured):
        groundstate.calc_gs(None, call.pop("basis_setup"), None, None, 0, **call)
    return seen


def test_calc_gs_with_a_spec_hands_solve_ground_state_the_same_keywords_as_the_legacy_dict(monkeypatch):
    """A1(a): plumbing only -- the spec's values reach ``solve_ground_state`` exactly as the dict keys did."""
    from impurityModel.ed.groundstate import GroundStateSpec

    common = {"impurity_orbitals": {0: [[0, 1]]}, "bath_states": ({}, {}), "N0": {0: 1}, "spin_layout": _Layout()}
    legacy = _capture_solve_kwargs(
        monkeypatch,
        basis_setup={
            **common,
            "tau": 0.01,
            "chain_restrict": True,
            "dense_cutoff": 77,
            "truncation_threshold": 1234,
            "e_pt2_tol": 1e-7,
            "de2_min": 1e-9,
            "use_hf_seed": False,
            "weighted_restrictions": None,
        },
        slaterWeightMin=1e-5,
    )
    spec = GroundStateSpec(
        tau=0.01,
        chain_restrict=True,
        dense_cutoff=77,
        truncation_threshold=1234,
        e_pt2_tol=1e-7,
        de2_min=1e-9,
        use_hf_seed=False,
    )
    via_spec = _capture_solve_kwargs(monkeypatch, basis_setup=dict(common), slaterWeightMin=1e-5, spec=spec)
    assert via_spec == legacy


def test_calc_gs_refuses_a_key_set_by_both_the_dict_and_the_spec(monkeypatch):
    from impurityModel.ed.groundstate import GroundStateSpec

    common = {"impurity_orbitals": {0: [[0, 1]]}, "bath_states": ({}, {}), "N0": {0: 1}, "spin_layout": _Layout()}
    with pytest.raises(ValueError, match="dense_cutoff"):
        _capture_solve_kwargs(
            monkeypatch, basis_setup={**common, "dense_cutoff": 5}, spec=GroundStateSpec(tau=0.01, dense_cutoff=7)
        )


def test_every_dense_cutoff_default_has_one_owner():
    """P6: four places used to hold a literal. The user-facing default is ``config.DEFAULT_DENSE_CUTOFF`` (the
    input schema mirrors it as a leaf module), the library fallback is ``config.GS_DENSE_CUTOFF``."""
    import inspect

    from impurityModel.ed import config, groundstate
    from impurityModel.ed.model import SolverOptions

    assert SolverOptions().dense_cutoff == config.DEFAULT_DENSE_CUTOFF == 500
    assert config.GS_DENSE_CUTOFF == 1000
    assert groundstate.GroundStateSpec(tau=0.01).dense_cutoff == config.GS_DENSE_CUTOFF
    for fn in (groundstate.find_ground_state_sector, groundstate.solve_ground_state):
        assert inspect.signature(fn).parameters["dense_cutoff"].default == config.GS_DENSE_CUTOFF
    text = inspect.getsource(__import__("impurityModel.inputformat.schema", fromlist=["x"]))
    assert f'Key("dense_cutoff", Kind.COUNT, {config.DEFAULT_DENSE_CUTOFF},' in text
