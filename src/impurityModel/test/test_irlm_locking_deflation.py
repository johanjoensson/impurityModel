"""Regression tests for EA16 IRLM locking deflation.

Guards two coupled bugs that made the implicitly restarted block Lanczos return
*spurious* eigenvalues — values strictly below the true spectral minimum, which is
variationally impossible for a Rayleigh-Ritz projection and therefore a sure sign of
lost orthogonality:

1. **Missing locking deflation (EA16 §2.6.2).** The inner Lanczos sweep was not kept
   orthogonal to the already-converged ("locked") Ritz vectors. The matvec keeps
   amplifying the dominant locked directions back into the active subspace, so locked
   eigenvalues (and their ``2*theta`` harmonics) reappear as Ritz values *below* the
   true minimum. This struck for *intermediate* subspace sizes (it self-corrects for
   very small or very large ``max_subspace_blocks``) and for FULL reort too, so it is
   not a partial-reorthogonalization issue. Manifested as ``calc_gs`` returning an
   energy below the dense ground state (e.g. ``-20.3`` for a sector whose true minimum
   is ``-10.4``), serially and under MPI.

2. **Duplicate eigenpairs from a converged start (no deflation in final extraction).**
   When IRLM is seeded from already-converged eigenvectors (as
   ``CIPSISolver.get_eigenvectors`` does, restarting from ``psi_refs``), the leftover
   active factorization holds near-copies of the locked Ritz vectors;
   ``_assemble_results`` accepted them, returning each true eigenvalue twice and
   double-counting states in the downstream thermal average.

The fixes deflate the inner sweeps (both array and ManyBodyState kernels) and the final
extraction against the locked set. These tests pin the corrected behaviour across
subspace sizes, reort modes, both operator paths, and serial vs MPI.
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.BlockLanczosArray import Reort, block_normalize
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant
from impurityModel.ed.irlm import implicitly_restarted_block_lanczos_cy
from impurityModel.ed.BlockLanczos import implicitly_restarted_block_lanczos_cy as mbs_irlm
from impurityModel.test.test_restarted_lanczos import MockBasis


def _build_system(n_orb=10, n_part=5, seed=42):
    """Random Hermitian one-body Hamiltonian on a fixed-particle sector.

    Returns the ManyBodyOperator, its dense matrix on the full sector, the basis
    ManyBodyStates (column order matching the dense matrix), and the dense spectrum.
    This is the Hamiltonian from ``test_groundstate_and_density_matrix_mpi`` — the
    case that originally exposed the spurious-eigenvalue bug.
    """
    rng = np.random.RandomState(seed)
    h_dict = {}
    for i in range(n_orb):
        h_dict[((i, "c"), (i, "a"))] = rng.uniform(-2, 2)
        for j in range(i + 1, n_orb):
            val = rng.uniform(-1, 1)
            h_dict[((i, "c"), (j, "a"))] = val
            h_dict[((j, "c"), (i, "a"))] = val
    h_op = ManyBodyOperator(h_dict)

    states = []
    for occ in itertools.combinations(range(n_orb), n_part):
        b = bytearray((n_orb + 7) // 8)
        for o in occ:
            b[o // 8] |= 1 << (7 - o % 8)  # MSB-first orbital->bit convention
        states.append(SlaterDeterminant.from_bytes(bytes(b)))
    basis_states = [ManyBodyState({sd: 1.0}) for sd in states]

    N = len(states)
    index = {sd: i for i, sd in enumerate(states)}
    H = np.zeros((N, N), dtype=complex)
    for j, hpsi in enumerate(h_op.apply_multi(basis_states)):
        for sd, amp in hpsi.items():
            if sd in index:
                H[index[sd], j] = amp
    assert np.max(np.abs(H - H.conj().T)) < 1e-12, "test Hamiltonian must be Hermitian"
    eigvals = np.linalg.eigvalsh(H)
    return h_op, H, basis_states, eigvals


# Intermediate subspace sizes are the dangerous regime: small msb self-corrects via
# frequent restarts, very large msb captures everything before the destructive restart.
_MSB = [30, 60, 100, 200]
_MODES = [Reort.FULL, Reort.PARTIAL]


@pytest.mark.parametrize("msb", _MSB)
@pytest.mark.parametrize("mode", _MODES)
def test_no_eigenvalue_below_spectral_minimum(msb, mode):
    """No returned Ritz value may lie below the dense minimum (Rayleigh-Ritz bound)."""
    import scipy.sparse as sps

    _, H, _, eigvals = _build_system()
    N = H.shape[0]
    rng = np.random.RandomState(1)
    psi0 = rng.standard_normal((N, 1)) + 1j * rng.standard_normal((N, 1))

    ev, _ = implicitly_restarted_block_lanczos_cy(
        psi0=psi0,
        h_op=sps.csr_matrix(H),
        basis=None,
        num_wanted=20,
        max_subspace_blocks=msb,
        tol=1e-8,
        max_restarts=100,
        verbose=False,
        reort=mode,
        comm=None,
    )
    ev = np.sort(np.asarray(ev).real)
    # The hard invariant: nothing below the true minimum (the spurious-eigenvalue bug).
    assert ev[0] >= eigvals[0] - 1e-6, f"spurious eigenvalue {ev[0]} < lambda_min {eigvals[0]}"
    # And the ground state is actually found.
    np.testing.assert_allclose(ev[0], eigvals[0], atol=1e-6)


@pytest.mark.parametrize("mode", _MODES)
def test_lowest_eigenvalues_match_dense_no_duplicates(mode):
    """A random start recovers the distinct lowest eigenvalues with no duplicates."""
    import scipy.sparse as sps

    _, H, _, eigvals = _build_system()
    N = H.shape[0]
    rng = np.random.RandomState(1)
    psi0 = rng.standard_normal((N, 1)) + 1j * rng.standard_normal((N, 1))

    ev, _ = implicitly_restarted_block_lanczos_cy(
        psi0=psi0,
        h_op=sps.csr_matrix(H),
        basis=None,
        num_wanted=15,
        max_subspace_blocks=100,
        tol=1e-8,
        max_restarts=200,
        verbose=False,
        reort=mode,
        comm=None,
    )
    ev = np.sort(np.asarray(ev).real)
    np.testing.assert_allclose(ev, eigvals[:15], atol=1e-6)
    assert np.min(np.diff(ev)) > 1e-6, "returned eigenvalues contain a spurious duplicate"


def test_converged_start_no_duplicate_eigenpairs():
    """Seeding IRLM from exact eigenvectors must not duplicate eigenpairs.

    The exact lowest ``k`` eigenvectors span a ``k``-dimensional invariant subspace, so
    only those ``k`` eigenvalues are reachable; the corrected extraction returns them
    once each (the bug returned every eigenvalue twice)."""
    import scipy.sparse as sps

    _, H, _, eigvals = _build_system()
    N = H.shape[0]
    evals, evecs = np.linalg.eigh(H)
    psi0 = evecs[:, :10].copy()  # exact lowest 10 eigenvectors as the start block

    ev, _ = implicitly_restarted_block_lanczos_cy(
        psi0=psi0,
        h_op=sps.csr_matrix(H),
        basis=None,
        num_wanted=20,
        max_subspace_blocks=100,
        tol=1e-8,
        max_restarts=100,
        verbose=False,
        reort=Reort.PARTIAL,
        comm=None,
    )
    ev = np.sort(np.asarray(ev).real)
    # Each reachable eigenvalue appears exactly once.
    assert len(ev) == len(np.unique(np.round(ev, 6))), "duplicate eigenpairs from converged start"
    np.testing.assert_allclose(ev[:10], evals[:10], atol=1e-7)


@pytest.mark.parametrize("mode", ["full", "partial"])
def test_manybody_path_no_spurious_eigenvalue(mode):
    """The ManyBodyState (hash-distributed) IRLM kernel is deflated against Xl too."""
    h_op, H, basis_states, eigvals = _build_system()
    N = len(basis_states)
    rng = np.random.RandomState(2)
    coeffs = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    psi0 = [sum((b * c for b, c in zip(basis_states, coeffs)), ManyBodyState())]
    psi0, _ = block_normalize(psi0, False, None, 0.0)

    ev, _ = mbs_irlm(
        psi0=psi0,
        h_op=h_op,
        basis=MockBasis(N),
        num_wanted=12,
        max_subspace_blocks=60,
        tol=1e-8,
        max_restarts=100,
        verbose=False,
        reort=mode,
        comm=None,
    )
    ev = np.sort(np.asarray(ev).real)
    assert ev[0] >= eigvals[0] - 1e-6, f"spurious MBS eigenvalue {ev[0]} < lambda_min {eigvals[0]}"
    np.testing.assert_allclose(ev[0], eigvals[0], atol=1e-6)


def _partition(n, size):
    return [n // size + (1 if r < n % size else 0) for r in range(size)]


@pytest.mark.mpi
@pytest.mark.parametrize("mode", [Reort.FULL, Reort.PARTIAL])
def test_array_irlm_mpi_no_spurious_eigenvalue(mode):
    """Row-block-distributed IRLM agrees with serial and stays above the spectral min.

    This is the distilled core of ``test_groundstate_and_density_matrix_mpi``: under MPI
    the non-associative Allreduce perturbs the trajectory, which used to tip the
    undeflated sweep into the runaway. Both the energy and the no-spurious invariant must
    hold."""
    comm = MPI.COMM_WORLD
    _, H, _, eigvals = _build_system()
    N = H.shape[0]
    rng = np.random.RandomState(1)
    psi0_full = rng.standard_normal((N, 1)) + 1j * rng.standard_normal((N, 1))

    counts = _partition(N, comm.size)
    c0 = sum(counts[: comm.rank])
    c1 = c0 + counts[comm.rank]

    import scipy.sparse as sps

    h_local = sps.csr_matrix(np.ascontiguousarray(H[:, c0:c1]))
    psi0_local = np.ascontiguousarray(psi0_full[c0:c1, :], dtype=complex)

    class _Basis:
        def __init__(self, c):
            self.comm = c

    ev_mpi, _ = implicitly_restarted_block_lanczos_cy(
        psi0=psi0_local,
        h_op=h_local,
        basis=_Basis(comm),
        num_wanted=20,
        max_subspace_blocks=100,
        tol=1e-8,
        max_restarts=100,
        verbose=False,
        reort=mode,
        comm=comm,
    )
    ev_mpi = np.sort(np.asarray(ev_mpi).real)
    assert ev_mpi[0] >= eigvals[0] - 1e-6, f"spurious MPI eigenvalue {ev_mpi[0]} < {eigvals[0]}"
    np.testing.assert_allclose(ev_mpi[0], eigvals[0], atol=1e-6)
