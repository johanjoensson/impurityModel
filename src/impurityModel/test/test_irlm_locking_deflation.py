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

from impurityModel.ed.BlockLanczos import implicitly_restarted_block_lanczos_cy as mbs_irlm
from impurityModel.ed.BlockLanczosArray import Reort, block_normalize
from impurityModel.ed.irlm import implicitly_restarted_block_lanczos_cy
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant
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


@pytest.mark.parametrize("msb", _MSB)
@pytest.mark.parametrize("locked_reort", ["full", "partial"])
def test_locked_reort_switch_no_spurious(msb, locked_reort):
    """Both locking-reorth modes ('full' default, 'partial' = EA16 §2.6.2) recover the
    ground state with nothing below the spectral minimum, across subspace sizes."""
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
        reort=Reort.PARTIAL,
        locked_reort=locked_reort,
    )
    ev = np.sort(np.asarray(ev).real)
    assert ev[0] >= eigvals[0] - 1e-6, f"{locked_reort} spurious eigenvalue {ev[0]} < {eigvals[0]}"
    np.testing.assert_allclose(ev[:20], eigvals[:20], atol=1e-6)


@pytest.mark.parametrize("locked_reort", ["full", "partial"])
def test_locked_reort_switch_manybody(locked_reort):
    """The estimate-driven 'partial' locking reorth is wired through the MBS kernel too."""
    h_op, _, basis_states, eigvals = _build_system()
    N = len(basis_states)
    rng = np.random.RandomState(2)
    coeffs = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    psi0 = [sum((b * c for b, c in zip(basis_states, coeffs)), ManyBodyState())]
    psi0, _ = block_normalize(psi0, False, None, 0.0)

    ev, _ = implicitly_restarted_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=MockBasis(N),
        num_wanted=12,
        max_subspace_blocks=80,
        tol=1e-8,
        max_restarts=100,
        verbose=False,
        reort="partial",
        locked_reort=locked_reort,
    )
    ev = np.sort(np.asarray(ev).real)
    assert ev[0] >= eigvals[0] - 1e-6
    np.testing.assert_allclose(ev[:12], eigvals[:12], atol=1e-6)


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

    _, H, _, _eigvals = _build_system()
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
    h_op, _H, basis_states, eigvals = _build_system()
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


def test_select_restart_indices_ghost_filter():
    """The optional ghost filter shifts away locked-eigenvalue copies; default keeps them.

    Eigenvalue-based ghost filtering is a defense-in-depth fallback (the IRLM driver leaves
    it off because the inner-sweep deflation removes ghosts by eigenvector, preserving true
    degeneracies). This pins the API: (a) default behaviour is unchanged, (b) with a tol the
    ghost of a locked eigenvalue is excluded from the kept set, and (c) the kept set never
    starves below n_keep even when ghosts must be used to fill it.
    """
    from impurityModel.ed import ea16

    # index 0 is a ghost of the locked value -5.0; indices 1-3 are genuine.
    theta = np.array([-5.0, -4.0, -3.0, -2.0])
    locked_evals = np.array([-5.0])

    # Default: no ghost filtering -> the two lowest (incl. the -5.0 ghost) are kept.
    kept_def, _ = ea16.select_restart_indices(theta, n_keep=2, locked_local=[])
    assert set(kept_def.tolist()) == {0, 1}

    # With a tol: the -5.0 ghost (index 0) is shifted away; genuine -3.0 (index 2) fills in.
    kept_g, _ = ea16.select_restart_indices(theta, n_keep=2, locked_local=[], locked_evals=locked_evals, ghost_tol=1e-3)
    assert 0 not in set(kept_g.tolist())
    assert set(kept_g.tolist()) == {1, 2}

    # Starvation guard: if everything is a ghost, the kept set is still filled to n_keep.
    theta_all_ghost = np.array([-5.0, -5.0001, -4.9999])
    kept_s, _ = ea16.select_restart_indices(
        theta_all_ghost, n_keep=2, locked_local=[], locked_evals=np.array([-5.0]), ghost_tol=1e-2
    )
    assert len(kept_s) == 2


@pytest.mark.parametrize("msb", [20, 60, 100])
@pytest.mark.parametrize("mode", _MODES)
def test_trlm_array_no_spurious_eigenvalue(msb, mode):
    """Array thick-restart Lanczos: correct lowest eigenvalues, none below the minimum.

    Regression for two TRLM bugs: (a) it crashed when the sweep deflated (shrinking
    blocks) because it assumed a uniform block width ``m_actual * n``; and (b) it diverged
    (T overflowing to ~1e150) because it did not normalize the start block before the
    recurrence."""
    import scipy.sparse as sps

    from impurityModel.ed.trlm import thick_restart_block_lanczos

    _, H, _, eigvals = _build_system()
    N = H.shape[0]
    rng = np.random.RandomState(1)
    psi0 = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))

    ev, _ = thick_restart_block_lanczos(
        psi0=psi0,
        h_op=sps.csr_matrix(H),
        basis=None,
        num_wanted=8,
        max_subspace_blocks=msb,
        tol=1e-8,
        max_restarts=200,
        verbose=False,
        reort=mode,
    )
    ev = np.sort(np.asarray(ev).real)
    assert np.all(np.isfinite(ev)), "TRLM diverged (non-finite eigenvalues)"
    assert ev[0] >= eigvals[0] - 1e-6, f"spurious TRLM eigenvalue {ev[0]} < lambda_min {eigvals[0]}"
    np.testing.assert_allclose(ev[:8], eigvals[:8], atol=1e-5)


def test_trlm_array_unnormalized_start_is_stable():
    """An unnormalized start block must not make TRLM diverge (it now normalizes psi0)."""
    import scipy.sparse as sps

    from impurityModel.ed.trlm import thick_restart_block_lanczos

    _, H, _, eigvals = _build_system()
    N = H.shape[0]
    rng = np.random.RandomState(3)
    # Deliberately large-norm, unnormalized start (norm ~ sqrt(N) per column).
    psi0 = 5.0 * (rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2)))

    ev, _ = thick_restart_block_lanczos(
        psi0=psi0,
        h_op=sps.csr_matrix(H),
        basis=None,
        num_wanted=5,
        max_subspace_blocks=40,
        tol=1e-8,
        max_restarts=200,
        verbose=False,
        reort=Reort.FULL,
    )
    ev = np.sort(np.asarray(ev).real)
    assert np.all(np.isfinite(ev))
    np.testing.assert_allclose(ev[:5], eigvals[:5], atol=1e-5)


@pytest.mark.parametrize("nstart", [1, 2, 3])
@pytest.mark.parametrize("mode", _MODES)
def test_trlm_array_restart_loop_width_agnostic(nstart, mode):
    """The thick-restart continuation loop tracks variable block widths.

    A random Hermitian matrix with ``num_wanted`` well below ``N`` and a small subspace
    forces many real restarts (the block-Krylov does not saturate in one sweep). The
    continuation must stay correct for block starts of width 1, 2, 3 and recover the
    lowest eigenvalues with none below the spectral minimum."""
    import scipy.sparse as sps

    from impurityModel.ed.trlm import thick_restart_block_lanczos

    rng = np.random.RandomState(17)
    N = 40
    M = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = (M + M.conj().T) / 2
    eigvals = np.linalg.eigvalsh(H)
    psi0 = rng.standard_normal((N, nstart)) + 1j * rng.standard_normal((N, nstart))

    ev, _ = thick_restart_block_lanczos(
        psi0=psi0,
        h_op=sps.csr_matrix(H),
        basis=None,
        num_wanted=4,
        max_subspace_blocks=max(5, 4 // nstart + 3),
        tol=1e-9,
        max_restarts=500,
        verbose=False,
        reort=mode,
    )
    ev = np.sort(np.asarray(ev).real)
    assert np.all(np.isfinite(ev))
    assert ev[0] >= eigvals[0] - 1e-6
    np.testing.assert_allclose(ev[:4], eigvals[:4], atol=1e-5)


@pytest.mark.parametrize("N", [13, 14, 15])
def test_trlm_array_block_deflation_in_restart(N):
    """Block deflation that surfaces only in/after a restart must not crash or corrupt.

    With a width-2 start on a modest odd/even ``N`` and a tight subspace, the *residual*
    block can deflate while the diagonal blocks do not (so the run enters the restart
    loop with a padded trailing ``beta``), and continuation blocks can shrink mid-restart.
    Regression for the padded-``beta_res`` broadcast crash and the uniform-width arrowhead
    assumption; the result must stay finite, above the minimum, and match the dense GS."""
    import scipy.sparse as sps

    from impurityModel.ed.trlm import thick_restart_block_lanczos

    rng = np.random.RandomState(N * 13)
    M = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = (M + M.conj().T) / 2
    eigvals = np.linalg.eigvalsh(H)
    psi0 = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))

    ev, _ = thick_restart_block_lanczos(
        psi0=psi0,
        h_op=sps.csr_matrix(H),
        basis=None,
        num_wanted=4,
        max_subspace_blocks=6,
        tol=1e-9,
        max_restarts=800,
        verbose=False,
        reort=Reort.FULL,
    )
    ev = np.sort(np.asarray(ev).real)
    assert np.all(np.isfinite(ev))
    assert ev[0] >= eigvals[0] - 1e-6
    np.testing.assert_allclose(ev[:4], eigvals[:4], atol=1e-5)


@pytest.mark.parametrize("mode", ["full", "partial"])
def test_trlm_manybody_no_spurious_eigenvalue(mode):
    """ManyBodyState thick-restart Lanczos: deflation-aware, no spurious eigenvalue."""
    from impurityModel.ed.trlm import thick_restart_block_lanczos

    h_op, _, basis_states, eigvals = _build_system()
    N = len(basis_states)
    rng = np.random.RandomState(2)
    coeffs = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    psi0 = [sum((b * c for b, c in zip(basis_states, coeffs)), ManyBodyState())]
    psi0, _ = block_normalize(psi0, False, None, 0.0)

    ev, _ = thick_restart_block_lanczos(
        psi0=psi0,
        h_op=h_op,
        basis=MockBasis(N),
        num_wanted=8,
        max_subspace_blocks=60,
        tol=1e-8,
        max_restarts=200,
        verbose=False,
        reort=mode,
    )
    ev = np.sort(np.asarray(ev).real)
    assert np.all(np.isfinite(ev))
    assert ev[0] >= eigvals[0] - 1e-6, f"spurious MBS-TRLM eigenvalue {ev[0]} < {eigvals[0]}"
    np.testing.assert_allclose(ev[:8], eigvals[:8], atol=1e-5)


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


@pytest.mark.mpi
@pytest.mark.parametrize("locked_reort", ["full", "partial"])
def test_locked_reort_switch_mpi(locked_reort):
    """Both locking-reorth modes are MPI-collective-safe and stay above the minimum.

    The §2.6.2 estimate is computed from Allreduced (replicated) band blocks, so the
    trigger decision is identical on every rank and the reorthogonalization Allreduce
    fires collectively — no deadlock and no spurious eigenvalue under distribution."""
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
        reort=Reort.PARTIAL,
        locked_reort=locked_reort,
        comm=comm,
    )
    ev_mpi = np.sort(np.asarray(ev_mpi).real)
    assert ev_mpi[0] >= eigvals[0] - 1e-6, f"{locked_reort} spurious MPI eigenvalue {ev_mpi[0]}"
    np.testing.assert_allclose(ev_mpi[0], eigvals[0], atol=1e-6)


@pytest.mark.mpi
@pytest.mark.parametrize("mode", [Reort.FULL, Reort.PARTIAL])
def test_trlm_array_mpi_no_spurious_eigenvalue(mode):
    """Row-block-distributed thick-restart Lanczos: correct GS, none below the minimum."""
    from impurityModel.ed.trlm import thick_restart_block_lanczos

    comm = MPI.COMM_WORLD
    _, H, _, eigvals = _build_system()
    N = H.shape[0]
    rng = np.random.RandomState(1)
    psi0_full = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))

    counts = _partition(N, comm.size)
    c0 = sum(counts[: comm.rank])
    c1 = c0 + counts[comm.rank]

    import scipy.sparse as sps

    h_local = sps.csr_matrix(np.ascontiguousarray(H[:, c0:c1]))
    psi0_local = np.ascontiguousarray(psi0_full[c0:c1, :], dtype=complex)

    class _Basis:
        def __init__(self, c):
            self.comm = c

    ev_mpi, _ = thick_restart_block_lanczos(
        psi0=psi0_local,
        h_op=h_local,
        basis=_Basis(comm),
        num_wanted=8,
        max_subspace_blocks=60,
        tol=1e-8,
        max_restarts=200,
        verbose=False,
        reort=mode,
    )
    ev_mpi = np.sort(np.asarray(ev_mpi).real)
    assert np.all(np.isfinite(ev_mpi))
    assert ev_mpi[0] >= eigvals[0] - 1e-6, f"spurious MPI TRLM eigenvalue {ev_mpi[0]} < {eigvals[0]}"
    np.testing.assert_allclose(ev_mpi[:8], eigvals[:8], atol=1e-5)
