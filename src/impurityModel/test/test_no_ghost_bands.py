import numpy as np
import pytest
import scipy.sparse as sps

try:
    from mpi4py import MPI

    _has_mpi = True
except ImportError:
    _has_mpi = False

from impurityModel.ed.BlockLanczosArray import Reort, block_normalize
from impurityModel.ed.irlm import implicitly_restarted_block_lanczos_cy
from impurityModel.ed.ManyBodyUtils import ManyBodyState, inner_multi, ManyBodyOperator
from impurityModel.ed.trlm import thick_restart_block_lanczos
from impurityModel.test.test_block_lanczos_array_empty_rank import _contiguous_counts_with_empty_last


def create_diagonal_system(eigvals, path, comm=None):
    """Create a diagonal Hamiltonian and starting states.

    If path == 'ManyBodyState', we return:
      h_op (ManyBodyOperator), basis (Basis), psi0 (list of ManyBodyState), basis_states
    If path == 'array', we return:
      h_op (csr_matrix or ndarray), basis (mock or None), psi0 (ndarray), None
    """
    n_states = len(eigvals)
    n_blocks = 2

    if path == "array":
        H_dense = np.diag(eigvals).astype(complex)
        if comm is not None:
            counts = _contiguous_counts_with_empty_last(n_states, comm.size)
            offsets = np.array([sum(counts[:r]) for r in range(comm.size)], dtype=int)
            c0 = offsets[comm.rank]
            c1 = c0 + counts[comm.rank]
            h_op = sps.csr_matrix(H_dense[:, c0:c1])

            rng = np.random.default_rng(42)
            psi0_full = rng.standard_normal((n_states, n_blocks)) + 1j * rng.standard_normal((n_states, n_blocks))
            psi0_local = np.ascontiguousarray(psi0_full[c0:c1, :], dtype=complex)
            psi0, _ = block_normalize(psi0_local, mpi=True, comm=comm)

            class ArrayMockBasis:
                def __init__(self, comm):
                    self.comm = comm

            basis = ArrayMockBasis(comm)
        else:
            np.random.seed(42)
            psi0 = np.random.randn(n_states, n_blocks) + 1j * np.random.randn(n_states, n_blocks)
            psi0, _ = block_normalize(psi0, mpi=False, comm=None)
            h_op = H_dense
            basis = None
        return h_op, basis, psi0, None
    else:
        # ManyBodyState path
        from impurityModel.ed.manybody_basis import Basis

        states_bytes = [(1 << i).to_bytes(8, "little") for i in range(n_states)]
        hop = {((i, "c"), (i, "a")): float(val) for i, val in enumerate(eigvals)}
        h_op = ManyBodyOperator(hop)

        if comm is not None:
            basis = Basis(
                impurity_orbitals={0: [list(range(n_states))]},
                bath_states=({0: [[]]}, {0: [[]]}),
                initial_basis=states_bytes,
                verbose=False,
                comm=comm,
            )

            if comm.rank == 0:
                np.random.seed(42)
                psi0_full = []
                for _ in range(n_blocks):
                    st = ManyBodyState()
                    for s in states_bytes:
                        st[basis.type.from_bytes(s)] = np.random.rand() + 1j * np.random.rand()
                    psi0_full.append(st)
            else:
                psi0_full = [ManyBodyState(width=1) for _ in range(n_blocks)]

            # Each seed goes through its own explicit width-1 block rather than a bare
            # ManyBodyState() placeholder on the non-owning rank: once the flat and
            # block classes merge (Phase 7 step 3), a bare placeholder is the width-0
            # polymorphic zero, an asymmetric mismatch against the owning rank's
            # populated (eventually width-1) seeds that would deadlock
            # redistribute_psis' collective.
            psi0_blocks = [ManyBodyState.from_states([psi]) for psi in psi0_full]
            psi0 = [blk.to_states()[0] for blk in basis.redistribute_psis(psi0_blocks)]
            psi0, _ = block_normalize(psi0, mpi=True, comm=comm)
        else:
            basis = Basis(
                impurity_orbitals={0: [list(range(n_states))]},
                bath_states=({0: [[]]}, {0: [[]]}),
                initial_basis=states_bytes,
                verbose=False,
            )
            np.random.seed(42)
            psi0 = []
            for _ in range(n_blocks):
                st = ManyBodyState()
                for s in states_bytes:
                    st[basis.type.from_bytes(s)] = np.random.rand() + 1j * np.random.rand()
                psi0.append(st)
            psi0, _ = block_normalize(psi0, mpi=False, comm=None)

        return h_op, basis, psi0, None


def assert_orthonormal(eigvecs, path, comm=None):
    overlaps = np.conj(eigvecs.T) @ eigvecs if path == "array" else inner_multi(eigvecs, eigvecs)

    if comm is not None:
        overlaps = np.ascontiguousarray(overlaps, dtype=complex)
        comm.Allreduce(MPI.IN_PLACE, overlaps, op=MPI.SUM)

    N = overlaps.shape[0]
    expected = np.eye(N, dtype=complex)
    eps = np.finfo(float).eps
    sqrt_eps = np.sqrt(eps)

    np.testing.assert_allclose(overlaps, expected, atol=sqrt_eps)


def get_xfail_marker(mode, path, solver, spectrum_type, mpi):
    # On this 12-state spectrum with block size 2 and a tight restart subspace
    # (max_subspace_blocks=5), the array path converges whenever the degeneracies are
    # *split* (near_degenerate): each near-copy is then a distinct Ritz value the block
    # recurrence can resolve one block at a time, so it returns the six lowest cleanly.
    # Those cells run for real (and guard against regressions).
    #
    # IRLM used to be excluded from that: it was xfail on near_degenerate too, because the
    # rank floor sat *above* this spectrum's splitting. The eigenvalues here are 1e-9 apart
    # in relative terms and DEFLATE_TOL was EPS**(1/3) ~ 6.06e-6, so the second copy of each
    # near-degenerate pair looked rank-deficient and was deflated away -- leaving T_full
    # partially filled and producing exactly the spurious Ritz values this test is named
    # after. The floor is now EPS**(2/3) ~ 3.67e-11, below the splitting, and those ten
    # cells (five here, five in the MPI variant) pass. See the DEFLATE_TOL comment in
    # TSQR.pyx.
    #
    # The remaining cells hit a genuine restarted-block-Lanczos limitation, not a
    # reorthogonalization bug and not a deflation threshold: an *exact* high-multiplicity
    # degeneracy (eigenvalue 2.0 has multiplicity 3, exceeding the block size 2) cannot be
    # fully captured inside the tight restart subspace, whatever the floor -- the discarded
    # singular value is exactly zero. T_full is left partially filled and spurious (e.g.
    # zero) Ritz values appear. Deflation itself works (see test_deflation_shrinking_block);
    # this is tracked as future work. strict=False so that if a solver later resolves a
    # cell it surfaces as XPASS rather than failing the suite.
    if path == "array" and spectrum_type == "near_degenerate":
        return None
    return pytest.mark.xfail(
        strict=False,
        reason="restarted block Lanczos cannot resolve a degeneracy exceeding the block "
        "size within a tight restart subspace (partial T_full -> spurious Ritz values)",
    )


@pytest.mark.parametrize("mode", [Reort.NONE, Reort.PARTIAL, Reort.FULL, Reort.SELECTIVE, Reort.PERIODIC])
@pytest.mark.parametrize("path", ["array", "ManyBodyState"])
@pytest.mark.parametrize("solver", ["TRLM", "IRLM"])
@pytest.mark.parametrize("spectrum_type", ["exact_degenerate", "near_degenerate"])
def test_no_ghost_bands(mode, path, solver, spectrum_type, request):
    marker = get_xfail_marker(mode, path, solver, spectrum_type, mpi=False)
    if marker is not None:
        request.node.add_marker(marker)

    if spectrum_type == "exact_degenerate":
        eigvals_exact = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)
    else:
        # near degenerate
        eigvals_exact = np.array(
            [1.0, 1.0 + 1e-9, 2.0, 2.0 + 1e-9, 2.0 + 2e-9, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float
        )

    h_op, basis, psi0, _ = create_diagonal_system(eigvals_exact, path, comm=None)
    num_wanted = 6
    max_subspace_blocks = 5  # force restarts

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

    # Check eigenvalues match exact ones with multiplicity (sorted)
    # NONE is allowed to fail/have ghost bands, but we parameterized it so let's let it xfail if it fails.
    np.testing.assert_allclose(np.sort(eigvals), eigvals_exact[:num_wanted], atol=1e-6)

    if mode != Reort.NONE:
        assert_orthonormal(eigvecs, path, comm=None)


@pytest.mark.mpi
@pytest.mark.parametrize("mode", [Reort.NONE, Reort.PARTIAL, Reort.FULL, Reort.SELECTIVE, Reort.PERIODIC])
@pytest.mark.parametrize("path", ["array", "ManyBodyState"])
@pytest.mark.parametrize("solver", ["TRLM", "IRLM"])
@pytest.mark.parametrize("spectrum_type", ["exact_degenerate", "near_degenerate"])
def test_no_ghost_bands_mpi(mode, path, solver, spectrum_type, request):
    comm = MPI.COMM_WORLD
    marker = get_xfail_marker(mode, path, solver, spectrum_type, mpi=True)
    if marker is not None:
        request.node.add_marker(marker)

    if spectrum_type == "exact_degenerate":
        eigvals_exact = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)
    else:
        # near degenerate
        eigvals_exact = np.array(
            [1.0, 1.0 + 1e-9, 2.0, 2.0 + 1e-9, 2.0 + 2e-9, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float
        )

    h_op, basis, psi0, _ = create_diagonal_system(eigvals_exact, path, comm=comm)
    num_wanted = 6
    max_subspace_blocks = 5  # force restarts

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

    # Check eigenvalues match exact ones with multiplicity (sorted)
    np.testing.assert_allclose(np.sort(eigvals), eigvals_exact[:num_wanted], atol=1e-6)

    if mode != Reort.NONE:
        assert_orthonormal(eigvecs, path, comm=comm)
