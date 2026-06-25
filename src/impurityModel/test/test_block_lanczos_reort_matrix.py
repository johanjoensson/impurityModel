import itertools
import pytest
import numpy as np
import scipy.linalg as sp
import scipy.sparse as sps

try:
    from mpi4py import MPI
    _has_mpi = True
except ImportError:
    _has_mpi = False

from impurityModel.ed.BlockLanczosArray import Reort, block_normalize
from impurityModel.ed.trlm import thick_restart_block_lanczos
from impurityModel.ed.irlm import implicitly_restarted_block_lanczos_cy
from impurityModel.ed.ManyBodyUtils import inner_multi, ManyBodyState, SlaterDeterminant
from impurityModel.test.test_restarted_lanczos import get_test_system, MockBasis
from impurityModel.test.test_block_lanczos_array_empty_rank import _contiguous_counts_with_empty_last


def build_dense_matrix_from_manybody(h_op, basis_states):
    N = len(basis_states)
    H_dense = np.zeros((N, N), dtype=complex)
    states = [list(b.keys())[0] for b in basis_states]
    H_basis_states = h_op.apply_multi(basis_states)
    for j in range(N):
        for i, sd_i in enumerate(states):
            H_dense[i, j] = H_basis_states[j].get(sd_i, 0.0)
    return H_dense


def get_mpi_basis(comm):
    from impurityModel.ed.manybody_basis import Basis
    n_sites = 8
    n_particles = 4
    combinations = list(itertools.combinations(range(n_sites), n_particles))
    states_bytes = []
    for c in combinations:
        val = sum(1 << i for i in c)
        states_bytes.append(val.to_bytes(8, byteorder="little"))
    basis = Basis(
        impurity_orbitals={0: [list(range(8))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states_bytes,
        verbose=False,
        comm=comm
    )
    return basis


def assert_orthonormal(eigvecs, path, comm=None):
    if path == "array":
        overlaps = np.conj(eigvecs.T) @ eigvecs
    else:
        overlaps = inner_multi(eigvecs, eigvecs)
    
    if comm is not None:
        overlaps = np.ascontiguousarray(overlaps, dtype=complex)
        comm.Allreduce(MPI.IN_PLACE, overlaps, op=MPI.SUM)
    
    N = overlaps.shape[0]
    expected = np.eye(N, dtype=complex)
    eps = np.finfo(float).eps
    sqrt_eps = np.sqrt(eps)
    
    np.testing.assert_allclose(overlaps, expected, atol=sqrt_eps)


# Both solvers now converge on both paths: IRLM gained EA16 locking + explicit purging
# (Meerbergen & Scott, RAL-TR-2000-011), so it reaches 1e-8 on this tight-binding system
# within the small subspace where it previously stalled. No cells are expected to fail.
@pytest.mark.parametrize("mode", [Reort.NONE, Reort.PARTIAL, Reort.FULL, Reort.SELECTIVE, Reort.PERIODIC])
@pytest.mark.parametrize("path", ["array", "ManyBodyState"])
@pytest.mark.parametrize("solver", ["TRLM", "IRLM"])
def test_reort_matrix(mode, path, solver):
    h_op_mb, N, eigvals_exact, basis_states = get_test_system()
    num_wanted = 4
    max_subspace_blocks = 6
    n_blocks = 2

    np.random.seed(42)
    
    if path == "array":
        H_dense = build_dense_matrix_from_manybody(h_op_mb, basis_states)
        psi0 = np.random.randn(N, n_blocks) + 1j * np.random.randn(N, n_blocks)
        psi0, _ = block_normalize(psi0, mpi=False, comm=None)
        h_op = H_dense
        basis = None
    else:
        psi0 = []
        for _ in range(n_blocks):
            state = ManyBodyState()
            for b in basis_states:
                state += b * (np.random.rand() + 1j * np.random.rand())
            psi0.append(state)
        psi0, _ = block_normalize(psi0, mpi=False, comm=None)
        h_op = h_op_mb
        basis = MockBasis(N)

    if solver == "TRLM":
        eigvals, eigvecs = thick_restart_block_lanczos(
            psi0=psi0,
            h_op=h_op,
            basis=basis,
            num_wanted=num_wanted,
            max_subspace_blocks=max_subspace_blocks,
            tol=1e-8,
            max_restarts=50,
            verbose=False,
            reort=mode,
        )
    else:
        # IRLM
        eigvals, eigvecs = implicitly_restarted_block_lanczos_cy(
            psi0=psi0,
            h_op=h_op,
            basis=basis,
            num_wanted=num_wanted,
            max_subspace_blocks=max_subspace_blocks,
            tol=1e-8,
            max_restarts=50,
            verbose=False,
            reort=mode,
            comm=None,
        )

    # Check eigenvalues
    np.testing.assert_allclose(eigvals, eigvals_exact[:num_wanted], atol=1e-8)
    
    # Check orthonormality (except for NONE mode, which loses it)
    if mode != Reort.NONE:
        assert_orthonormal(eigvecs, path, comm=None)


@pytest.mark.mpi
@pytest.mark.parametrize("mode", [Reort.NONE, Reort.PARTIAL, Reort.FULL, Reort.SELECTIVE, Reort.PERIODIC])
@pytest.mark.parametrize("path", ["array", "ManyBodyState"])
@pytest.mark.parametrize("solver", ["TRLM", "IRLM"])
def test_reort_matrix_mpi(mode, path, solver):
    comm = MPI.COMM_WORLD
    h_op_mb, N, eigvals_exact, basis_states = get_test_system()
    num_wanted = 4
    max_subspace_blocks = 6
    n_blocks = 2

    if path == "array":
        H_dense = build_dense_matrix_from_manybody(h_op_mb, basis_states)
        counts = _contiguous_counts_with_empty_last(N, comm.size)
        offsets = np.array([sum(counts[:r]) for r in range(comm.size)], dtype=int)
        c0 = offsets[comm.rank]
        c1 = c0 + counts[comm.rank]
        
        # Local row/col slice
        h_op = sps.csr_matrix(H_dense[:, c0:c1])
        
        # Globally normalised starting block partitioned contiguously
        # Set seed per rank but construct same global vector
        rng = np.random.default_rng(42)
        psi0_full = rng.standard_normal((N, n_blocks)) + 1j * rng.standard_normal((N, n_blocks))
        psi0_local = np.ascontiguousarray(psi0_full[c0:c1, :], dtype=complex)
        psi0, _ = block_normalize(psi0_local, mpi=True, comm=comm)
        
        # Create a mock basis with communicator for the TRLM/IRLM driver
        class ArrayMockBasis:
            def __init__(self, comm):
                self.comm = comm
        basis = ArrayMockBasis(comm)
    else:
        # ManyBodyState path
        basis = get_mpi_basis(comm)
        h_op = h_op_mb
        
        # Construct psi0 on rank 0, redistribute
        if comm.rank == 0:
            np.random.seed(42)
            psi0_full = []
            for _ in range(n_blocks):
                state = ManyBodyState()
                for b in basis_states:
                    state += b * (np.random.rand() + 1j * np.random.rand())
                psi0_full.append(state)
        else:
            psi0_full = [ManyBodyState() for _ in range(n_blocks)]
            
        psi0 = basis.redistribute_psis(psi0_full)
        psi0, _ = block_normalize(psi0, mpi=True, comm=comm)

    if solver == "TRLM":
        eigvals, eigvecs = thick_restart_block_lanczos(
            psi0=psi0,
            h_op=h_op,
            basis=basis,
            num_wanted=num_wanted,
            max_subspace_blocks=max_subspace_blocks,
            tol=1e-8,
            max_restarts=50,
            verbose=False,
            reort=mode,
        )
    else:
        # IRLM
        eigvals, eigvecs = implicitly_restarted_block_lanczos_cy(
            psi0=psi0,
            h_op=h_op,
            basis=basis,
            num_wanted=num_wanted,
            max_subspace_blocks=max_subspace_blocks,
            tol=1e-8,
            max_restarts=50,
            verbose=False,
            reort=mode,
            comm=comm,
        )

    # Check eigenvalues on all ranks
    np.testing.assert_allclose(eigvals, eigvals_exact[:num_wanted], atol=1e-8)

    # Check orthonormality (except for NONE mode)
    if mode != Reort.NONE:
        assert_orthonormal(eigvecs, path, comm=comm)


@pytest.mark.mpi
@pytest.mark.parametrize("mode", [Reort.PARTIAL, Reort.SELECTIVE])
@pytest.mark.parametrize("path", ["array", "ManyBodyState"])
def test_W_identical_across_ranks(mode, path):
    comm = MPI.COMM_WORLD
    h_op_mb, N, eigvals_exact, basis_states = get_test_system()
    n_blocks = 2
    max_iter = 5

    if path == "array":
        H_dense = build_dense_matrix_from_manybody(h_op_mb, basis_states)
        counts = _contiguous_counts_with_empty_last(N, comm.size)
        offsets = np.array([sum(counts[:r]) for r in range(comm.size)], dtype=int)
        c0 = offsets[comm.rank]
        c1 = c0 + counts[comm.rank]
        h_op = sps.csr_matrix(H_dense[:, c0:c1])
        
        rng = np.random.default_rng(42)
        psi0_full = rng.standard_normal((N, n_blocks)) + 1j * rng.standard_normal((N, n_blocks))
        psi0_local = np.ascontiguousarray(psi0_full[c0:c1, :], dtype=complex)
        psi0, _ = block_normalize(psi0_local, mpi=True, comm=comm)
        
        from impurityModel.ed.BlockLanczosArray import block_lanczos_array
        alphas, betas, Q_list, *W_res = block_lanczos_array(
            psi0=psi0,
            h_op=h_op,
            converged=lambda a, b, **kw: False,
            max_iter=max_iter,
            verbose=False,
            reort=mode,
            return_W=True,
            comm=comm,
        )
        W = W_res[0] if W_res else None
    else:
        basis = get_mpi_basis(comm)
        h_op = h_op_mb
        
        if comm.rank == 0:
            np.random.seed(42)
            psi0_full = []
            for _ in range(n_blocks):
                state = ManyBodyState()
                for b in basis_states:
                    state += b * (np.random.rand() + 1j * np.random.rand())
                psi0_full.append(state)
        else:
            psi0_full = [ManyBodyState() for _ in range(n_blocks)]
            
        psi0 = basis.redistribute_psis(psi0_full)
        psi0, _ = block_normalize(psi0, mpi=True, comm=comm)

        from impurityModel.ed.BlockLanczos import block_lanczos_cy
        alphas, betas, Q_list, W = block_lanczos_cy(
            psi0=psi0,
            h_op=h_op,
            basis=basis,
            converged_fn=lambda a, b, **kw: False,
            verbose=False,
            reort=mode.name.lower(),
            max_iter=max_iter,
            comm=comm,
        )
        
    # comm.bcast rank-0's W and assert every rank's W equals it to 1e-12
    W_root = comm.bcast(W, root=0)
    np.testing.assert_allclose(W, W_root, atol=1e-12)


@pytest.mark.parametrize("p", [1, 2, 4])
@pytest.mark.parametrize("path", ["array", "ManyBodyState"])
def test_deflation_shrinking_block(p, path):
    h_op_mb, N, eigvals_exact, basis_states = get_test_system()
    np.random.seed(42)
    max_iter = 30

    from impurityModel.ed.BlockLanczosArray import _build_full_T

    if path == "array":
        H_dense = build_dense_matrix_from_manybody(h_op_mb, basis_states)
        psi0 = np.random.randn(N, p) + 1j * np.random.randn(N, p)
        if p > 1:
            psi0[:, -1] = psi0[:, 0]
        
        psi0, _ = block_normalize(psi0, mpi=False, comm=None)
        
        from impurityModel.ed.BlockLanczosArray import block_lanczos_array
        alphas, betas, Q_list, block_widths = block_lanczos_array(
            psi0=psi0,
            h_op=H_dense,
            converged=lambda a, b, **kw: False,
            max_iter=max_iter,
            verbose=False,
            reort=Reort.FULL,
            return_widths=True,
            comm=None,
        )
    else:
        psi0_list = []
        for _ in range(p):
            state = ManyBodyState()
            for b in basis_states:
                state += b * (np.random.rand() + 1j * np.random.rand())
            psi0_list.append(state)
        if p > 1:
            psi0_list[-1] = psi0_list[0].copy()
            
        psi0, _ = block_normalize(psi0_list, mpi=False, comm=None)
        
        from impurityModel.ed.BlockLanczos import block_lanczos_cy
        alphas, betas, Q_basis, W, block_widths = block_lanczos_cy(
            psi0=psi0,
            h_op=h_op_mb,
            basis=MockBasis(N),
            converged_fn=lambda a, b, **kw: False,
            verbose=False,
            reort="full",
            max_iter=max_iter,
            return_widths=True,
            comm=None,
        )
        
    for idx in range(len(block_widths) - 1):
        assert block_widths[idx+1] <= block_widths[idx]
        
    T_full = _build_full_T(alphas, betas, block_widths=block_widths)
    eigvals, _ = sp.eigh(T_full)
    np.testing.assert_allclose(eigvals[0], eigvals_exact[0], atol=1e-8)


@pytest.mark.mpi
@pytest.mark.parametrize("p", [1, 2, 4])
@pytest.mark.parametrize("path", ["array", "ManyBodyState"])
def test_deflation_shrinking_block_mpi(p, path):
    comm = MPI.COMM_WORLD
    h_op_mb, N, eigvals_exact, basis_states = get_test_system()
    max_iter = 30

    from impurityModel.ed.BlockLanczosArray import _build_full_T

    if path == "array":
        H_dense = build_dense_matrix_from_manybody(h_op_mb, basis_states)
        counts = _contiguous_counts_with_empty_last(N, comm.size)
        offsets = np.array([sum(counts[:r]) for r in range(comm.size)], dtype=int)
        c0 = offsets[comm.rank]
        c1 = c0 + counts[comm.rank]
        
        h_op = sps.csr_matrix(H_dense[:, c0:c1])
        
        rng = np.random.default_rng(42)
        psi0_full = rng.standard_normal((N, p)) + 1j * rng.standard_normal((N, p))
        if p > 1:
            psi0_full[:, -1] = psi0_full[:, 0]
        psi0_local = np.ascontiguousarray(psi0_full[c0:c1, :], dtype=complex)
        psi0, _ = block_normalize(psi0_local, mpi=True, comm=comm)
        
        from impurityModel.ed.BlockLanczosArray import block_lanczos_array
        alphas, betas, Q_list, block_widths = block_lanczos_array(
            psi0=psi0,
            h_op=h_op,
            converged=lambda a, b, **kw: False,
            max_iter=max_iter,
            verbose=False,
            reort=Reort.FULL,
            return_widths=True,
            comm=comm,
        )
    else:
        basis = get_mpi_basis(comm)
        h_op = h_op_mb
        
        if comm.rank == 0:
            np.random.seed(42)
            psi0_full = []
            for _ in range(p):
                state = ManyBodyState()
                for b in basis_states:
                    state += b * (np.random.rand() + 1j * np.random.rand())
                psi0_full.append(state)
            if p > 1:
                psi0_full[-1] = psi0_full[0].copy()
        else:
            psi0_full = [ManyBodyState() for _ in range(p)]
            
        psi0 = basis.redistribute_psis(psi0_full)
        psi0, _ = block_normalize(psi0, mpi=True, comm=comm)

        from impurityModel.ed.BlockLanczos import block_lanczos_cy
        alphas, betas, Q_basis, W, block_widths = block_lanczos_cy(
            psi0=psi0,
            h_op=h_op,
            basis=basis,
            converged_fn=lambda a, b, **kw: False,
            verbose=False,
            reort="full",
            max_iter=max_iter,
            return_widths=True,
            comm=comm,
        )
        
    for idx in range(len(block_widths) - 1):
        assert block_widths[idx+1] <= block_widths[idx]
        
    T_full = _build_full_T(alphas, betas, block_widths=block_widths)
    eigvals, _ = sp.eigh(T_full)
    if comm.rank == 0:
        with open("deflation_debug.txt", "w") as f:
            f.write(f"block_widths: {block_widths}\n")
            f.write(f"T_full shape: {T_full.shape}\n")
            f.write(f"alphas shape: {alphas.shape}\n")
            f.write(f"betas shape: {betas.shape}\n")
            f.write(f"deflation eigenvalues: {list(eigvals[:5])}\n")
            f.write(f"exact eigenvalues: {list(eigvals_exact[:5])}\n")
            f.write(f"alphas:\n{alphas}\n")
            f.write(f"betas:\n{betas}\n")
    np.testing.assert_allclose(eigvals[0], eigvals_exact[0], atol=1e-8)
