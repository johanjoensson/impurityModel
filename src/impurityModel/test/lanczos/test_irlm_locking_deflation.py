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
from impurityModel.ed.irlm import (
    FINAL_ACCEPT_SCALE,
    implicitly_restarted_block_lanczos,
    implicitly_restarted_block_lanczos_cy,
)
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant
from impurityModel.test.support.lanczos_fixtures import MockBasis, deflating_start_block


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
                H[index[sd], j] = amp[0]
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


def test_manybody_path_reseed_after_locking_whole_invariant_subspace():
    """``num_wanted`` larger than the dimension psi0 can ever reach must not crash.

    A 4-site tight-binding chain (1-particle sector) is exactly closed under H: the
    Lanczos sweep exhausts its whole 4-dimensional space and stops (invariant
    subspace). ``_irlm_core`` then locks all 4 Ritz pairs, still wants 5 more
    (``num_wanted=9``), and -- since the remaining subspace dimension (0) is smaller
    than a further restart needs -- reseeds from the original ``psi0``, projected
    orthogonal to the now fully-populated locked set ``Xl``.

    ``Xl`` is a ``ManyBodyState`` once anything has locked (Phase 5 step 5). This was
    originally a regression guard for a mixed-representation crash caught by code review:
    at the time, ``psi0``/the reseed vector stayed ``list[ManyBodyState]`` (block_lanczos_cy's
    fresh-start ingestion only accepted a real list), so ``_orth_against_locked`` handed a
    list ``v0`` to ``block_orthogonalize`` against a block ``Xl``, which dispatches on its
    first argument only and fell through to the list-only ``block_orthogonalize_sparse``,
    raising ``TypeError`` on a block second argument. Phase 5 step 6 closed the gap at the
    source instead (``block_lanczos_cy``'s fresh-start ingestion now accepts a
    ``ManyBodyState`` seed directly, so ``psi0``/``v0`` stay block-native throughout),
    but this test is kept as a standing regression guard for the reseed-after-full-locking
    path itself.
    """
    n_sites = 4
    op_dict = {}
    for i in range(n_sites - 1):
        op_dict[((i, "c"), (i + 1, "a"))] = -1.0
        op_dict[((i + 1, "c"), (i, "a"))] = -1.0
    h_op = ManyBodyOperator(op_dict)

    vac = ManyBodyState({SlaterDeterminant((0,)): 1.0})
    states = []
    for i in range(n_sites):
        op = ManyBodyOperator({((i, "c"),): 1.0})
        states.append(next(iter(op.apply(vac).keys())))

    basis = MockBasis(n_sites)
    basis.local_basis = states

    rng = np.random.RandomState(7)
    psi0 = ManyBodyState()
    for s in states:
        psi0[s] = rng.standard_normal() + 1j * rng.standard_normal()
    psi0, _ = block_normalize([psi0], False, None, 0.0)

    # num_wanted (9) exceeds the chain's total dimension (4): every reachable eigenpair
    # gets locked, then the code must reseed from psi0 (now fully deflated against Xl)
    # and stop cleanly instead of crashing.
    ev, _ = mbs_irlm(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=9,
        max_subspace_blocks=15,
        tol=1e-8,
        max_restarts=50,
        verbose=False,
        reort="partial",
        comm=None,
    )
    ev = np.sort(np.asarray(ev).real)
    exact_eigvals = -2 * np.cos(np.pi * np.arange(1, n_sites + 1) / (n_sites + 1))
    exact_eigvals.sort()
    assert len(ev) == n_sites, f"expected all {n_sites} reachable eigenpairs, got {len(ev)}"
    np.testing.assert_allclose(ev, exact_eigvals, atol=1e-7)


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


def test_locked_reort_step_guards_tiny_sigma_min():
    """R7 unified the ``1/sigma_min`` amplification factor in the EA16 2.6.2
    estimate-driven locking reorth onto the EPS-guarded form ``1/max(sv_min, EPS)``.

    Before that, the MBS driver guarded it and the array kernel did not, so on the
    array path a retained singular value below EPS produced an unbounded (or infinite,
    at sv_min == 0) amplification straight into the xi overlap recurrence. TSQR's
    deflation floor is *relative* to the operator scale, so a small-norm operator can
    legitimately retain a singular value that small -- it is not an unreachable state.

    No end-to-end fixture reaches it (the golden case
    ``irlm_array_locked_partial`` observes sv_min in 0.41 .. 2.25), so the guard is
    pinned here directly instead of being left to an argument.
    """
    from impurityModel.ed.BlockLanczosCore import locked_reort_step

    nlock, n, p = 3, 12, 2
    rng = np.random.RandomState(5)
    locked, _ = np.linalg.qr(rng.randn(n, nlock) + 1j * rng.randn(n, nlock))
    q_next, _ = np.linalg.qr(rng.randn(n, p) + 1j * rng.randn(n, p))
    locked_evals = np.array([0.5, 1.5, 2.5])
    alpha_i = np.eye(p, dtype=complex)
    omega_min = 1e-14
    xi = np.full(nlock, omega_min)
    xi_prev = np.full(nlock, omega_min)

    for sv_min in (0.0, 1e-300, 1e-20):
        q_out, xi_prev_out, xi_out = locked_reort_step(
            xi.copy(),
            xi_prev.copy(),
            locked_evals,
            alpha_i,
            sv_min,
            1.0,
            0.0,
            locked,
            q_next.copy(),
            False,
            None,
            omega_min,
        )
        assert np.all(np.isfinite(xi_out)), f"xi went non-finite at sv_min={sv_min}"
        assert np.all(np.isfinite(xi_prev_out)), f"xi_prev went non-finite at sv_min={sv_min}"
        assert np.all(np.isfinite(np.asarray(q_out))), f"q_next went non-finite at sv_min={sv_min}"

    # And the guard is a cap, not a no-op: sv_min far below EPS must give exactly the
    # same result as sv_min == EPS, since both clamp to the same 1/EPS factor.
    from impurityModel.ed.TSQR import EPS

    _, _, xi_tiny = locked_reort_step(
        xi.copy(),
        xi_prev.copy(),
        locked_evals,
        alpha_i,
        1e-300,
        1.0,
        0.0,
        locked,
        q_next.copy(),
        False,
        None,
        omega_min,
    )
    _, _, xi_eps = locked_reort_step(
        xi.copy(),
        xi_prev.copy(),
        locked_evals,
        alpha_i,
        EPS,
        1.0,
        0.0,
        locked,
        q_next.copy(),
        False,
        None,
        omega_min,
    )
    np.testing.assert_array_equal(xi_tiny, xi_eps)


# --------------------------------------------------------------------------------------
# Never report a Ritz pair that was not converged
# --------------------------------------------------------------------------------------


def _true_residuals(h, es, vecs):
    vecs = np.asarray(vecs)
    return np.linalg.norm(h @ vecs - vecs * np.asarray(es)[None, :], axis=0)


@pytest.mark.parametrize("num_wanted,max_blocks", [(8, 6), (10, 8), (6, 5)])
def test_irlm_never_reports_an_unconverged_ritz_pair(num_wanted, max_blocks):
    """A deflated factorization is not a converged one, and IRLM used to hand it over anyway.

    ``total < m_act * p`` breaks out of the restart loop -- IRLM's uniform-width
    purge/restart genuinely cannot continue a factorization whose blocks have narrowed, so
    unlike TRLM it has no width-aware continuation to fall through to. That break is fine.
    What was not fine is where the pairs then came from: ``_assemble_results`` took the
    lowest active Ritz pairs of whatever the last factorization held, and
    ``lock_remaining_and_stop`` locked its own set, neither applying the EA16 eq. (15)
    acceptance test that defines "converged". On this fixture that returned eigenvalues
    wrong by up to 31.6 with residuals of 5.3, against a requested ``tol`` of 1e-10.

    IRLM converges only the two exact ground-state directions here and now says so by
    returning two pairs. Fewer than ``num_wanted`` is the documented contract; wrong is not.
    """
    h, psi0, exact = deflating_start_block()

    es, vecs = implicitly_restarted_block_lanczos(
        psi0=psi0,
        h_op=h,
        basis=None,
        num_wanted=num_wanted,
        max_subspace_blocks=max_blocks,
        tol=1e-10,
        max_restarts=100,
        verbose=False,
        reort="full",
    )

    assert 0 < len(es) <= num_wanted
    # A prefix of the spectrum, not a subset of it: callers read the result as the lowest
    # len(es) eigenvalues, so a gap in the middle would be a wrong answer, not a short one.
    np.testing.assert_allclose(np.sort(np.asarray(es).real), exact[: len(es)], atol=1e-8)
    residuals = _true_residuals(h, es, vecs)
    assert np.max(residuals) < 1e-8, residuals


def test_irlm_reports_a_prefix_when_the_trailing_block_deflates():
    """The other leak: ``lock_remaining_and_stop`` locked without testing convergence.

    Its rationale was that an exact breakdown zeroes every Ritz residual -- true, but it
    fires on ``res_width < p``, which includes a residual block that merely *narrowed* and
    leaves the residuals at ``O(||H||)``. This shape used to return 6 pairs, one of them at
    a true residual of 3.1e-6 against ``tol = 1e-10``; it now returns the 4 it converged.
    """
    import scipy.sparse as sps

    rng = np.random.default_rng(11)
    n = 40
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    a = a + a.conj().T
    h = sps.csr_matrix(a)
    psi0 = np.linalg.qr(rng.standard_normal((n, 3)) + 0j)[0].astype(complex)
    exact = np.sort(np.linalg.eigvalsh(a))

    es, vecs = implicitly_restarted_block_lanczos(
        psi0=psi0,
        h_op=h,
        basis=None,
        num_wanted=6,
        max_subspace_blocks=5,
        tol=1e-10,
        max_restarts=100,
        verbose=False,
        reort="full",
    )

    assert 0 < len(es) <= 6
    np.testing.assert_allclose(np.sort(np.asarray(es).real), exact[: len(es)], atol=1e-8)
    assert np.max(_true_residuals(h, es, vecs)) < 1e-8


def test_final_accept_scale_is_operator_relative_not_the_bare_eps_floor():
    """The acceptance gate must discriminate against real non-convergence only.

    ``EPS * ||T||`` is the residual floor of double precision, not a threshold: converged
    pairs land at a small, arithmetic-dependent multiple of it. A gate at one or ten times
    that floor decides by reduction order -- measured on the warm-restart fixture
    (``||H|| = 2.5e5``), ten times the floor kept 3 of 3 wanted pairs in serial and 1 of 3
    at ``-n 2``. What this gate exists to reject sits five to eleven orders above the floor.
    """
    from impurityModel.ed.BlockLanczosCore import BREAKDOWN_TOL

    assert FINAL_ACCEPT_SCALE == BREAKDOWN_TOL
    assert FINAL_ACCEPT_SCALE > np.finfo(float).eps * 1e3
