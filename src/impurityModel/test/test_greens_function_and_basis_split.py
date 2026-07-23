import os

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.basis_split import split_basis_and_redistribute_psi
from impurityModel.ed.greens_function import (
    calc_Greens_function_with_offdiag,
    get_Greens_function,
)
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyBlockState, ManyBodyOperator, ManyBodyState, SlaterDeterminant


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
    indices, split_roots, color, items_per_color, split_basis, psis_out, _intercomms = split_basis_and_redistribute_psi(
        basis, priorities, psi0
    )

    assert list(indices) == [0, 1]
    assert split_roots == [0]
    assert color == 0
    assert list(items_per_color) == [2]
    assert split_basis == basis
    assert psis_out == psi0


def test_basis_split_and_redistribute_block_serial():
    """Width-1 ManyBodyBlockState lists take the same pass-through path as flat
    ManyBodyState lists (Phase 7 step 2b dual-path dispatch); serial has no MPI
    exchange to exercise, so this only covers the early-return branch."""
    states = [b"\x80", b"\x40", b"\x20", b"\x10"]
    basis = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        comm=None,
    )

    psi0 = [ManyBodyBlockState({SlaterDeterminant.from_bytes(states[0]): 1.0})]
    priorities = [1, 2]
    indices, split_roots, color, items_per_color, split_basis, psis_out, _intercomms = split_basis_and_redistribute_psi(
        basis, priorities, psi0
    )

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
        psi0 = [ManyBodyState({SlaterDeterminant.from_bytes(s): 1.0 for s in states})]
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
    result = split_basis_and_redistribute_psi(basis, priorities, psi0)
    indices, split_roots, color, items_per_color, split_basis, _psis_out, _intercomms = result

    assert len(indices) == 1
    assert indices[0] in [0, 1]
    assert len(split_roots) == 2
    assert items_per_color == [1, 1]
    expected_size = (
        split_roots[color + 1] - split_roots[color] if color + 1 < len(split_roots) else comm.size - split_roots[color]
    )
    assert split_basis.comm.size == expected_size
    if split_basis is not None and split_basis.comm != comm:
        split_basis.free_comm()
    split_basis = None


@pytest.mark.mpi
def test_basis_split_and_redistribute_block_mpi():
    """Width-1 ManyBodyBlockState lists take the same split_basis_and_redistribute_psi
    send/receive path as flat ManyBodyState lists (Phase 7 step 2b): the intercomm wire
    format unwraps each Row to a plain scalar (``v[0]``) and reconstructs a width-1 block
    on receipt, so the total seeded weight must round-trip exactly like the flat path."""
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

    if comm.rank == 0:
        psi0 = [ManyBodyBlockState({SlaterDeterminant.from_bytes(s): 1.0 for s in states})]
    else:
        # A genuine width-0 ManyBodyBlockState({}) is the polymorphic zero, not a
        # width-1 placeholder -- it would trip split_basis_and_redistribute_psi's
        # width guard on this rank only, an asymmetric exception that deadlocks the
        # other rank's matching recv. from_states forces an explicit width-1 zero
        # block instead, the same convention test_mpi_comm.py's block redistribute
        # tests use for an empty-rank placeholder.
        psi0 = [ManyBodyBlockState.from_states([ManyBodyState({})])]

    priorities = [1.0, 1.0]
    result = split_basis_and_redistribute_psi(basis, priorities, psi0)
    indices, split_roots, color, items_per_color, split_basis, psis_out, _intercomms = result

    assert len(indices) == 1
    assert indices[0] in [0, 1]
    assert len(split_roots) == 2
    assert items_per_color == [1, 1]
    expected_size = (
        split_roots[color + 1] - split_roots[color] if color + 1 < len(split_roots) else comm.size - split_roots[color]
    )
    assert split_basis.comm.size == expected_size

    assert len(psis_out) == 1
    assert isinstance(psis_out[0], ManyBodyBlockState)
    total = sum(psis_out[0].get(sd)[0].real for sd in split_basis.local_basis)
    assert np.isclose(total, len(split_basis.local_basis))

    if split_basis is not None and split_basis.comm != comm:
        split_basis.free_comm()
    split_basis = None


@pytest.mark.mpi
def test_memory_budget_caps_unit_split_mpi(monkeypatch):
    """A tiny memory budget must force the unit split down to a single color."""
    from impurityModel.ed import memory_estimate as me
    from impurityModel.ed.gf_units import run_units_distributed

    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("This test requires at least 2 MPI ranks")

    states = [b"\x80", b"\x40", b"\x20", b"\x10"]
    basis = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        comm=comm,
        truncation_threshold=100,
    )
    psi = ManyBodyState({SlaterDeterminant.from_bytes(states[0]): 1.0}) if comm.rank == 0 else ManyBodyState({})
    unit_seeds = [[psi], [psi]]
    unit_weights = np.array([1.0, 1.0])

    observed: list[int] = []

    def kernel(split_basis, u, seeds):
        size = split_basis.comm.size if split_basis.comm is not None else 1
        observed.append(size)
        return size

    # Generous budget: two equal units on >=2 ranks split into two colors, each smaller
    # than the full communicator.
    monkeypatch.setattr(me, "available_bytes_per_rank", lambda c: 2**60)
    run_units_distributed(basis, unit_seeds, unit_weights, kernel)
    assert all(size < comm.size for size in observed)

    # Tiny budget: the memory cap forces a single color (the unified basis on all ranks).
    observed.clear()
    monkeypatch.setattr(me, "available_bytes_per_rank", lambda c: 1)
    run_units_distributed(basis, unit_seeds, unit_weights, kernel)
    assert observed and all(size == comm.size for size in observed)


def test_calc_Greens_function_with_offdiag_serial():
    # Setup simple Hamiltonian H = 0.5 * c_0^\dagger c_0
    np.array([0.5])
    hop = {((0, "c"), (0, "a")): 0.5}
    hOp = ManyBodyOperator(hop)

    # Basis
    states = [b"\x80", b"\x00"]  # 1 orbital, state occupancies: 1, 0
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

    # alphas/betas are now ragged: one entry per state, each a list of variable-dim
    # blocks (padding stripped). Here block size 1, one Lanczos block.
    assert len(alphas) == 1
    assert len(alphas[0]) == 1 and alphas[0][0].shape == (1, 1)
    assert len(betas[0]) == 1 and betas[0][0].shape == (1, 1)
    np.testing.assert_allclose(alphas[0][0], [[0.0]], atol=1e-12)
    np.testing.assert_allclose(betas[0][0], [[0.0]], atol=1e-12)
    np.testing.assert_allclose(r[0], [[1.0]], atol=1e-12)


@pytest.mark.mpi
def test_calc_Greens_function_with_offdiag_mpi():
    comm = MPI.COMM_WORLD
    # Setup simple Hamiltonian H = 0.5 * c_0^\dagger c_0
    np.array([0.5])
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
    psi = ManyBodyState({SlaterDeterminant.from_bytes(states[0]): 1.0}) if comm.rank == 0 else ManyBodyState({})
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
        assert len(alphas[0]) == 1 and alphas[0][0].shape == (1, 1)
        assert len(betas[0]) == 1 and betas[0][0].shape == (1, 1)
        np.testing.assert_allclose(alphas[0][0], [[0.0]], atol=1e-12)
        np.testing.assert_allclose(betas[0][0], [[0.0]], atol=1e-12)
        np.testing.assert_allclose(r[0], [[1.0]], atol=1e-12)

    basis = None


@pytest.mark.mpi
def test_calc_Greens_function_with_offdiag_mpi_sparse():
    comm = MPI.COMM_WORLD
    # Setup simple Hamiltonian H = 0.5 * c_0^\dagger c_0
    np.array([0.5])
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
    psi = ManyBodyState({SlaterDeterminant.from_bytes(states[0]): 1.0}) if comm.rank == 0 else ManyBodyState({})
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
        assert len(alphas[0]) == 1 and alphas[0][0].shape == (1, 1)
        assert len(betas[0]) == 1 and betas[0][0].shape == (1, 1)
        np.testing.assert_allclose(alphas[0][0], [[0.0]], atol=1e-12)
        np.testing.assert_allclose(betas[0][0], [[0.0]], atol=1e-6)
        np.testing.assert_allclose(r[0], [[1.0]], atol=1e-12)

    basis = None


@pytest.mark.mpi
def test_calc_map_mpi():
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

    from impurityModel.ed.spectra import calc_map

    gs = calc_map(
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
        expected_gs = np.array(
            [
                -3.15050801e-14 - 14.18862669j,
                1.74853194e-14 - 10.49958375j,
                -8.96683048e-15 - 8.07660288j,
                0.00000000e00 - 6.40218521j,
            ]
        )
        np.testing.assert_allclose(gs.flatten(), expected_gs, atol=1e-13)

    basis = None


def test_dense_greens_function_basis_expansion():
    """The dense (non-sparse) GF path that expands the basis runs (regression for the
    block_green_impl call that was missing its `reort` argument)."""

    def _sd(occ, n=4):
        b = bytearray((n + 7) // 8)
        for o in occ:
            b[o // 8] |= 1 << (7 - o % 8)
        return SlaterDeterminant.from_bytes(bytes(b))

    # Hopping that connects orbitals so the Krylov/basis expansion actually iterates.
    hop = {}
    for a, b in ((0, 1), (2, 3)):
        hop[((a, "c"), (b, "a"))] = -0.5
        hop[((b, "c"), (a, "a"))] = -0.5
    hOp = ManyBodyOperator(hop)
    psi = ManyBodyState({_sd([0, 3]): 1.0})
    tOp = ManyBodyOperator({((0, "a"),): 1.0})
    basis = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=[bytes(_sd([0, 3]).to_bytearray())],
        comm=MPI.COMM_SELF,
    )
    alphas, _betas, _r = calc_Greens_function_with_offdiag(
        hOp=hOp,
        tOps=[tOp],
        psis=[psi],
        es=[0.0],
        block_basis=basis,
        delta=0.01,
        dN=2,
        occ_cutoff=1e-6,
        slaterWeightMin=0.0,
        verbose=False,
        sparse=False,
    )
    # Ragged return: one entry per state, each a list of (block) 2D arrays.
    assert len(alphas) == 1
    assert all(a.shape == (1, 1) for a in alphas[0])


@pytest.mark.mpi
def test_mpi_load_balancing_gf_split_vs_unified():
    """The Green's function is numerically identical with split vs unified MPI policy
    (Phase 7: split_threshold only changes which ranks do which block, not the result)."""
    comm = MPI.COMM_WORLD

    def _sd(occ, n=2):
        b = bytearray((n + 7) // 8)
        for o in occ:
            b[o // 8] |= 1 << (7 - o % 8)
        return bytes(b)

    hop = {((0, "c"), (1, "a")): -0.5, ((1, "c"), (0, "a")): -0.5}
    hOp = ManyBodyOperator(hop)
    tOps = [ManyBodyOperator({((0, "a"),): 1.0}), ManyBodyOperator({((1, "a"),): 1.0})]
    states = [_sd([0]), _sd([1])]  # 1-electron basis

    def run(split_threshold):
        basis = Basis(
            impurity_orbitals={0: [[0, 1]]},
            bath_states=({0: [[]]}, {0: [[]]}),
            initial_basis=states,
            comm=comm,
            split_threshold=split_threshold,
        )
        if comm.rank == 0:
            psi = ManyBodyState(
                {SlaterDeterminant.from_bytes(states[0]): 1.0, SlaterDeterminant.from_bytes(states[1]): 1.0}
            )
            psi = psi / psi.norm()
        else:
            psi = ManyBodyState({})
        psi = basis.redistribute_psis([psi])[0]
        return calc_Greens_function_with_offdiag(
            hOp=hOp,
            tOps=tOps,
            psis=[psi],
            es=[-0.5],
            block_basis=basis,
            delta=0.01,
            dN=1,
            occ_cutoff=1e-6,
            slaterWeightMin=0.0,
            verbose=False,
            sparse=True,
        )

    a_split, b_split, r_split = run(1e9)  # force maximal split
    a_uni, b_uni, r_uni = run(0.0)  # force unified communicator

    if comm.rank == 0:
        assert len(a_split) == len(a_uni)
        for xs, xu in zip(a_split, a_uni):
            np.testing.assert_allclose(xs, xu, atol=1e-10)
        for ys, yu in zip(b_split, b_uni):
            np.testing.assert_allclose(ys, yu, atol=1e-10)
        for zs, zu in zip(r_split, r_uni):
            np.testing.assert_allclose(zs, zu, atol=1e-10)


@pytest.mark.mpi
def test_get_Greens_function_split_threshold_invariant_mpi():
    """``get_Greens_function`` returns the same per-block Green's function no matter how the
    ``(block x eigenstate)`` work units are distributed across ranks: ``split_threshold`` only
    changes the parallel layout (max-split vs unified communicator), not the result. This is
    the invariant the unified single-split decomposition must preserve. Multiple blocks and
    multiple thermal states exercise the cross-product of work units.
    """
    comm = MPI.COMM_WORLD
    omega = np.linspace(-2.0, 2.0, 17)
    # Diagonal one-body H -> two independent 1-orbital Green's-function blocks.
    hOp = ManyBodyOperator({((0, "c"), (0, "a")): 0.3, ((1, "c"), (1, "a")): 0.7})
    state_bytes = [b"\x80", b"\x40", b"\xc0", b"\x00"]
    es = [0.3, 0.7]  # two thermal states

    def run(split_threshold):
        basis = Basis(
            impurity_orbitals={0: [[0, 1]]},
            bath_states=({0: [[]]}, {0: [[]]}),
            initial_basis=state_bytes,
            comm=comm,
            split_threshold=split_threshold,
        )
        if comm.rank == 0:
            psis = [ManyBodyState({SlaterDeterminant.from_bytes(b): 1.0}) for b in (b"\x80", b"\x40")]
        else:
            psis = [ManyBodyState({}), ManyBodyState({})]
        psis = basis.redistribute_psis(psis)
        _, gs_real, _ = get_Greens_function(
            matsubara_mesh=None,
            omega_mesh=omega,
            psis=psis,
            es=es,
            tau=1.0,
            basis=basis,
            hOp=hOp,
            delta=0.1,
            blocks=[[0], [1]],
            verbose=False,
            verbose_extra=False,
            reort=None,
            dN=1,
            occ_cutoff=1e-6,
            slaterWeightMin=0.0,
            sparse=True,
        )
        return gs_real

    g_split = run(1e9)  # maximal split: one color per unit
    g_uni = run(0.0)  # unified: a single communicator processes every unit
    if comm.rank == 0:
        assert len(g_split) == 2 and len(g_uni) == 2
        for a, b in zip(g_split, g_uni):
            np.testing.assert_allclose(a, b, atol=1e-9)


@pytest.mark.mpi
def test_get_Greens_function_operator_split_matches_block_mpi():
    """The operator-split (pairwise) decomposition reproduces the shared-Krylov block Green's
    function when the seeds and their scalar recurrences are distributed across ranks. Uses a
    2-orbital block (so off-diagonal pairs exercise the polarization seeds) with a hybridizing
    bath, on the sparse path -- the distributed analogue of the serial pairwise oracle.
    """
    comm = MPI.COMM_WORLD
    omega = np.linspace(-2.0, 2.0, 17)
    hOp = ManyBodyOperator(
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
            comm=comm,
        )
        if comm.rank == 0:
            psis = [ManyBodyState({SlaterDeterminant.from_bytes(b): 1.0}) for b in state_bytes]
        else:
            psis = [ManyBodyState({}), ManyBodyState({})]
        psis = basis.redistribute_psis(psis)
        old = os.environ.get("GF_OPERATOR_SPLIT")
        os.environ["GF_OPERATOR_SPLIT"] = "1" if op_split else "0"
        try:
            _, gs_real, _ = get_Greens_function(
                matsubara_mesh=None,
                omega_mesh=omega,
                psis=psis,
                es=es,
                tau=1.0,
                basis=basis,
                hOp=hOp,
                delta=0.1,
                blocks=[[0, 1]],
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
        return gs_real

    g_block = run(False)
    g_split = run(True)
    if comm.rank == 0:
        np.testing.assert_allclose(g_split[0], g_block[0], atol=1e-5, rtol=1e-4)


@pytest.mark.mpi
def test_basis_hash_distribution_partitions_and_stays_sparse():
    """``routing_hash`` gives a complete, disjoint partition of the basis and a
    communication graph whose out-degree is bounded independent of rank count.

    ``routing_hash`` (``SlaterDeterminant.h``) deliberately trades load-balance
    uniformity for a *sparse* communication graph: it is linear over the per-orbital
    occupations, so a hopping term (a few flipped bits) shifts the hash by a fixed
    offset and reaches only a bounded set of target ranks regardless of ``comm.size`` --
    the property that lets the solver scale to 100000+ ranks. Per-rank ownership is
    therefore only *approximately* balanced and can skew at composite / power-of-2 rank
    counts (a cryptographic hash would balance better but densify the graph). The
    load-bearing guarantees, asserted here, are: (1) every determinant owned by exactly
    one rank, ownership being purely ``routing_hash % size``; and (2) the single-hop
    out-degree stays bounded as ``comm.size`` grows.
    """
    from itertools import combinations

    comm = MPI.COMM_WORLD
    # 252 distinct 10-orbital, 5-electron determinants.
    n_orb = 10
    n_el = 5
    dets = []
    for occ in combinations(range(n_orb), n_el):
        b = bytearray((n_orb + 7) // 8)
        for o in occ:
            b[o // 8] |= 1 << (7 - o % 8)
        dets.append(bytes(b))
    basis = Basis(
        impurity_orbitals={0: [list(range(n_orb))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=dets,
        comm=comm,
    )

    # (1) Complete, disjoint partition: every determinant owned by exactly one rank, and
    # every rank agrees on ownership (ownership is purely routing_hash % size).
    local = list(basis.local_basis)
    sizes = comm.allgather(len(local))
    assert sum(sizes) == len(dets)
    owners = {}
    for r, chunk in enumerate(comm.allgather(local)):
        for sd in chunk:
            assert sd not in owners  # no determinant owned by two ranks
            owners[sd] = r
    assert set(owners) == {SlaterDeterminant.from_bytes(d) for d in dets}  # nothing lost
    for sd, r in owners.items():
        assert sd.routing_hash() % comm.size == r

    # (2) Sparse comm graph: the target ranks reached by single-electron hops out of a
    # rank's determinants are bounded and do NOT grow with comm.size. For this 10-orbital
    # problem the out-degree plateaus at ~7 even as ranks -> infinity; at tiny rank counts
    # it is trivially capped by comm.size (you cannot reach more ranks than exist).
    def hop_targets(sd_bytes):
        bits = [(sd_bytes[o // 8] >> (7 - o % 8)) & 1 for o in range(n_orb)]
        occ = [o for o in range(n_orb) if bits[o]]
        emp = [o for o in range(n_orb) if not bits[o]]
        targets = set()
        for a in occ:
            for b in emp:
                arr = bytearray(sd_bytes)
                arr[a // 8] &= ~(1 << (7 - a % 8))
                arr[b // 8] |= 1 << (7 - b % 8)
                targets.add(SlaterDeterminant.from_bytes(bytes(arr)).routing_hash() % comm.size)
        return targets

    # Recover each local determinant's byte image from its known owner-set membership.
    det_bytes_by_sd = {SlaterDeterminant.from_bytes(d): d for d in dets}
    reached = set()
    for sd in local:
        reached |= hop_targets(det_bytes_by_sd[sd])
    out_degree = comm.allreduce(len(reached), op=MPI.MAX)
    assert out_degree <= min(comm.size, 12)  # bounded by distinct hop offsets, not comm.size

    # (3) No rank-0 OOM: ownership is only approximately balanced (uniformity is traded
    # for graph sparsity, see above), but no rank hoards a catastrophic share -- storage
    # still scales down with rank count.
    expected = len(dets) / comm.size
    assert max(sizes) <= 5 * expected + 5


@pytest.mark.mpi
def test_calc_spectra_split_threshold_invariant_mpi():
    """``calc_spectra`` returns the same spectra no matter how the (tOp x eigenstate) work
    units are distributed across ranks (``split_threshold``) or how many eigenstates share one
    wide block-Lanczos recurrence (``GF_EIGENSTATE_GROUP``) -- the flat single-split scheme
    shared with the self-energy path only changes the parallel layout, not the result."""
    from impurityModel.ed.spectra import calc_spectra

    comm = MPI.COMM_WORLD
    w = np.linspace(-2.0, 2.0, 17)
    hOp = ManyBodyOperator({((0, "c"), (0, "a")): 0.3, ((1, "c"), (1, "a")): 0.7})
    state_bytes = [b"\x80", b"\x40", b"\xc0", b"\x00"]
    es = [0.3, 0.7]
    tOps = [ManyBodyOperator({((0, "a"),): 1.0}), ManyBodyOperator({((1, "a"),): 1.0})]

    def run(split_threshold, group):
        basis = Basis(
            impurity_orbitals={0: [[0, 1]]},
            bath_states=({0: [[]]}, {0: [[]]}),
            initial_basis=state_bytes,
            comm=comm,
            split_threshold=split_threshold,
        )
        if comm.rank == 0:
            psis = [ManyBodyState({SlaterDeterminant.from_bytes(b): 1.0}) for b in (b"\x80", b"\x40")]
        else:
            psis = [ManyBodyState({}), ManyBodyState({})]
        psis = basis.redistribute_psis(psis)
        old = os.environ.get("GF_EIGENSTATE_GROUP")
        os.environ["GF_EIGENSTATE_GROUP"] = str(group)
        try:
            gs = calc_spectra(
                hOp,
                tOps,
                psis,
                es,
                1.0,
                w,
                basis,
                0.1,
                0.0,
                False,
                1e-6,
                {0: (1, 1)},
                {0: (1, 0)},
                {0: (0, 1)},
            )
        finally:
            if old is None:
                del os.environ["GF_EIGENSTATE_GROUP"]
            else:
                os.environ["GF_EIGENSTATE_GROUP"] = old
        return gs

    g_ref = run(1e9, 1)  # maximal split, one eigenstate per recurrence
    for threshold, group in ((0.0, 1), (0.5, 1), (1e9, 2)):
        g = run(threshold, group)
        if comm.rank == 0:
            assert g.shape == g_ref.shape
            np.testing.assert_allclose(g, g_ref, atol=1e-9)


@pytest.mark.mpi
def test_calc_map_win_chunk_invariant_mpi():
    """The RIXS map is invariant to the (eigenstate x wIn-chunk) unit granularity
    (``GF_RIXS_WIN_CHUNK``) and to the split policy: chunk boundaries only move where the
    bicgstab warm-start chain cold-starts, which converges to the same resolvent."""
    from impurityModel.ed.spectra import calc_map

    comm = MPI.COMM_WORLD
    hOp = ManyBodyOperator({((0, "c"), (0, "a")): 0.5, ((1, "c"), (1, "a")): 0.3})
    states = [b"\x80", b"\x40", b"\x20", b"\x10", b"\x00"]
    wIns = np.array([0.1, 0.2, 0.3, 0.4])
    wLoss = np.array([0.0])
    tOpsIn = [ManyBodyOperator({((0, "a"),): 1.0})]
    tOpsOut = [ManyBodyOperator({((0, "c"),): 1.0})]

    def run(split_threshold, chunk):
        basis = Basis(
            impurity_orbitals={1: [[0]], 2: [[1]]},
            bath_states=({1: [[]], 2: [[]]}, {1: [[]], 2: [[]]}),
            initial_basis=states,
            comm=comm,
            split_threshold=split_threshold,
        )
        if comm.rank == 0:
            psis = [ManyBodyState({SlaterDeterminant.from_bytes(b): 1.0}) for b in states[:2]]
        else:
            psis = [ManyBodyState({}), ManyBodyState({})]
        psis = basis.redistribute_psis(psis)
        old = os.environ.get("GF_RIXS_WIN_CHUNK")
        os.environ["GF_RIXS_WIN_CHUNK"] = str(chunk)
        try:
            gs = calc_map(
                hOp=hOp,
                tOpsIn=tOpsIn,
                tOpsOut=tOpsOut,
                psis=psis,
                Es=np.array([0.5, 0.6]),
                tau=1.0,
                wIns=wIns,
                wLoss=wLoss,
                delta1=0.1,
                delta2=0.1,
                basis=basis,
                verbose=False,
                slaterWeightMin=0.0,
            )
        finally:
            if old is None:
                del os.environ["GF_RIXS_WIN_CHUNK"]
            else:
                os.environ["GF_RIXS_WIN_CHUNK"] = old
        return gs

    g_ref = run(1e9, 1)  # maximal split: one unit per (eigenstate, wIn point)
    for threshold, chunk in ((0.0, 4), (1.0, 2)):
        g = run(threshold, chunk)
        if comm.rank == 0:
            assert g.shape == g_ref.shape
            np.testing.assert_allclose(g, g_ref, atol=1e-9)
