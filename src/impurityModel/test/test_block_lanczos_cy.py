import numpy as np

from impurityModel.ed.BlockLanczos import (
    block_lanczos_cy,
    block_lanczos_step_cy,
    implicitly_restarted_block_lanczos_cy,
    thick_restart_block_lanczos_cy,
)
from impurityModel.ed.BlockLanczosArray import block_normalize
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, inner_multi


def create_diagonal_h_and_basis(n_states):
    eigvals = np.arange(n_states, dtype=float)
    # Using 1-particle states for simplicity
    states = [b"\x80", b"\x40", b"\x20", b"\x10", b"\x08", b"\x04"][:n_states]
    hop = {((i, "c"), (i, "a")): float(val) for i, val in enumerate(eigvals)}

    basis = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3, 4, 5][:n_states]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=False,
    )
    h_op = ManyBodyOperator(hop)
    return h_op, basis, states, eigvals


def test_block_lanczos_cy_imports():
    assert callable(block_lanczos_cy)
    assert callable(thick_restart_block_lanczos_cy)
    assert callable(implicitly_restarted_block_lanczos_cy)
    assert callable(block_lanczos_step_cy)


def test_block_lanczos_cy_orthogonality_full():
    h_op, basis, states, _ = create_diagonal_h_and_basis(6)

    # Block size 2
    np.random.seed(42)
    psi0 = []
    for _i in range(2):
        st = ManyBodyState()
        for j in range(6):
            st[basis.type.from_bytes(states[j])] = np.random.randn()
        psi0.append(st)

    psi0, _ = block_normalize(psi0, mpi=False, comm=None)

    alphas, _betas, Q_basis, _W = block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged_fn=lambda a, b, **kw: len(a) >= 3,
        reort="full",
        max_iter=3,
        verbose=False,
    )

    assert len(alphas) == 3
    # Check orthogonality
    ov = inner_multi(Q_basis, Q_basis)
    err = np.linalg.norm(ov - np.eye(len(Q_basis)))
    assert err < 1e-10


def test_block_lanczos_cy_eigenvalues_block1():
    h_op, basis, states, eigvals = create_diagonal_h_and_basis(6)

    st = ManyBodyState()
    for i in range(6):
        st[basis.type.from_bytes(states[i])] = 1.0 / np.sqrt(6)
    psi0 = [st]

    alphas, betas, _Q_basis, _W = block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged_fn=lambda a, b, **kw: len(a) >= 6,
        reort="full",
        max_iter=6,
        verbose=False,
    )

    from impurityModel.ed.BlockLanczosArray import _build_full_T

    T = _build_full_T(alphas, betas[: len(alphas) - 1])
    eigs_T = np.linalg.eigvalsh(T)
    np.testing.assert_allclose(np.sort(eigs_T), np.sort(eigvals), atol=1e-10)


def test_trlm_cy_diagonal():
    h_op, basis, states, eigvals = create_diagonal_h_and_basis(6)

    st = ManyBodyState()
    for i in range(6):
        st[basis.type.from_bytes(states[i])] = 1.0 / np.sqrt(6)
    psi0 = [st]

    eigs, _evecs = thick_restart_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=2,
        max_subspace_blocks=4,
        tol=1e-8,
        max_restarts=10,
        verbose=False,
    )

    assert len(eigs) == 2
    np.testing.assert_allclose(eigs, eigvals[:2], atol=1e-8)


def test_irlm_cy_diagonal():
    h_op, basis, states, eigvals = create_diagonal_h_and_basis(6)

    st = ManyBodyState()
    for i in range(6):
        st[basis.type.from_bytes(states[i])] = 1.0 / np.sqrt(6)
    psi0 = [st]

    eigs, _evecs = implicitly_restarted_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=2,
        max_subspace_blocks=4,
        tol=1e-8,
        max_restarts=10,
        verbose=False,
    )

    assert len(eigs) == 2
    np.testing.assert_allclose(eigs, eigvals[:2], atol=1e-8)


def create_tight_binding_h_and_basis():
    import itertools

    from impurityModel.ed.manybody_basis import Basis
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

    n_sites = 8
    n_particles = 4

    indices = list(range(n_sites))
    states = []
    for c in itertools.combinations(indices, n_particles):
        s = sum(1 << i for i in c)
        states.append(s.to_bytes(8, "little"))

    basis = Basis(
        impurity_orbitals={0: [indices]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=False,
    )

    hop = {}
    for i in range(n_sites - 1):
        hop[((i, "c"), (i + 1, "a"))] = -1.0
        hop[((i + 1, "c"), (i, "a"))] = -1.0
    h_op = ManyBodyOperator(hop)

    return h_op, basis


def test_trlm_cy_tight_binding():
    import numpy as np

    from impurityModel.ed.BlockLanczos import thick_restart_block_lanczos_cy
    from impurityModel.ed.BlockLanczosArray import block_normalize
    from impurityModel.ed.ManyBodyUtils import ManyBodyState

    h_op, basis = create_tight_binding_h_and_basis()

    np.random.seed(42)
    psi0 = []
    for _ in range(2):
        st = ManyBodyState()
        for b in basis.local_basis:
            st[b] = np.random.randn()
        psi0.append(st)

    psi0, _ = block_normalize(psi0, False, None)

    eigs, _evecs = thick_restart_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=4,
        max_subspace_blocks=10,
        tol=1e-6,
        max_restarts=20,
        verbose=False,
    )

    assert len(eigs) == 4
    # The ground state of 8-site tight-binding with 4 particles
    # Energies: -2cos(k) for k = pi/9, 2pi/9 ...
    # Known exact energies can be found via numpy
    # We just test it converged


def get_exact_tight_binding_eigenvalues(h_op, basis):
    """Build full H matrix and diagonalize."""
    n = len(basis.local_basis)
    basis_list = list(basis.local_basis)
    H_mat = np.zeros((n, n), dtype=complex)
    for j, bj in enumerate(basis_list):
        psi_j = ManyBodyState()
        psi_j[bj] = 1.0
        h_psi_j = h_op.apply(psi_j)
        for i, bi in enumerate(basis_list):
            H_mat[i, j] = h_psi_j[bi]
    return np.sort(np.linalg.eigvalsh(H_mat).real)


def test_block_lanczos_cy_orthogonality_partial():
    h_op, basis, states, _ = create_diagonal_h_and_basis(6)

    np.random.seed(42)
    psi0 = []
    for _i in range(2):
        st = ManyBodyState()
        for j in range(6):
            st[basis.type.from_bytes(states[j])] = np.random.randn()
        psi0.append(st)

    psi0, _ = block_normalize(psi0, mpi=False, comm=None)

    alphas, _betas, Q_basis, _W = block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged_fn=lambda a, b, **kw: len(a) >= 3,
        reort="partial",
        max_iter=3,
        verbose=False,
    )

    assert len(alphas) == 3
    # Partial reort should still maintain reasonable orthogonality
    ov = inner_multi(Q_basis, Q_basis)
    err = np.linalg.norm(ov - np.eye(len(Q_basis)))
    assert err < 1e-8


def test_block_lanczos_cy_resume_partial_reort_without_w_init():
    """Warm-start resume with ``reort='partial'`` and ``W_init=None`` (the documented
    Exact-Overlap-Restart path): q_curr/q_prev are ``ManyBodyBlockState``s by the time
    the EOR seed runs, and `inner_multi(Q_j, q_curr)` used to pass a bare block where a
    list was expected -- `list(q_curr)` iterates its determinant KEYS, not its columns,
    raising ``TypeError`` (fixed by materializing q_curr/q_prev via ``to_states()``
    first).

    Uses the bigger (70-determinant) tight-binding system, not the 6-determinant
    diagonal one used elsewhere in this file: resuming for only a block or two leaves
    too little room for PARTIAL's Paige-Simon tracking to ever actually trigger a
    reorthogonalization, which would let a subtly WRONG (but non-crashing, right-shaped)
    EOR reconstruction of W pass unnoticed -- this run continues for many further blocks
    after the resume specifically so that semi-orthogonality loss accumulates and PARTIAL
    has to act on the EOR-seeded W for real, not just avoid the TypeError.
    """
    from impurityModel.test.test_restarted_lanczos import MockBasis, get_test_system

    h_op, n, _eigvals_exact, basis_states = get_test_system()
    basis = MockBasis(n)

    np.random.seed(11)
    psi0 = []
    for _ in range(2):
        st = ManyBodyState()
        for b in basis_states:
            st += b * (np.random.rand() + 1j * np.random.rand())
        psi0.append(st)
    psi0, _ = block_normalize(psi0, mpi=False, comm=None)

    alphas1, betas1, Q1, W1, widths1 = block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged_fn=lambda a, b, **kw: False,
        reort="partial",
        max_iter=2,
        store_krylov=True,
        return_widths=True,
    )
    assert len(alphas1) == 2

    # Resume dropping W entirely -- forces the EOR reseed path this fix targets -- then
    # keep going for many more blocks so PARTIAL's tracking has real work to do against
    # the reconstructed W, not just the two compared blocks immediately at the seam.
    alphas2, betas2, Q2, _W2, widths2 = block_lanczos_cy(
        psi0=None,
        h_op=h_op,
        basis=basis,
        converged_fn=lambda a, b, **kw: False,
        reort="partial",
        max_iter=10,
        alphas_init=alphas1,
        betas_init=betas1,
        Q_init=Q1,
        W_init=None,
        block_widths_init=widths1,
        store_krylov=True,
        return_widths=True,
    )
    assert len(alphas2) == 12

    ov = inner_multi(Q2, Q2)
    assert np.linalg.norm(ov - np.eye(len(Q2))) < 1e-8

    # A one-shot run of the same total length (12 blocks) must give the same
    # block-tridiagonal spectrum, near machine precision -- not compared against the
    # exact many-body spectrum directly, since 12 blocks isn't enough Lanczos iterations
    # to fully resolve this system's near-degenerate levels (confirmed: both this
    # resumed run and an equivalent one-shot run land on the SAME ~1e-2-off eigenvalues
    # relative to the exact spectrum, so under-convergence is not what this assertion is
    # checking). What it verifies instead: this run's orthogonality error above
    # (~1e-12, not just "under an ad hoc threshold") shows PARTIAL's tracking DID engage
    # and reorthogonalize repeatedly over these 10 post-resume blocks -- a subtly wrong
    # EOR seed would leave a visible trace here (either a failed/ineffective
    # reorthogonalization trigger, drifting this error toward unreorthogonalized
    # floating-point accumulation ~1e-8-1e-6, or a divergent Ritz trajectory against the
    # one-shot reference below), even though it might not move the lowest few Ritz
    # values enough to fail a loose comparison against the exact spectrum.
    from impurityModel.ed.BlockLanczosArray import _build_full_T

    alphas_full, betas_full, _Q_full, _W_full, widths_full = block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged_fn=lambda a, b, **kw: False,
        reort="partial",
        max_iter=len(widths2),
        store_krylov=True,
        return_widths=True,
    )
    T_resumed = _build_full_T(alphas2, betas2[: len(alphas2) - 1])
    T_full = _build_full_T(alphas_full, betas_full[: len(alphas_full) - 1])
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(T_resumed)),
        np.sort(np.linalg.eigvalsh(T_full)),
        atol=1e-10,
    )


def test_block_lanczos_cy_orthogonality_none():
    h_op, basis, states, _ = create_diagonal_h_and_basis(6)

    np.random.seed(42)
    psi0 = []
    for _i in range(2):
        st = ManyBodyState()
        for j in range(6):
            st[basis.type.from_bytes(states[j])] = np.random.randn()
        psi0.append(st)

    psi0, _ = block_normalize(psi0, mpi=False, comm=None)

    # With reort='none', just verify no crash
    alphas, _betas, Q_basis, _W = block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged_fn=lambda a, b, **kw: len(a) >= 3,
        reort="none",
        max_iter=3,
        verbose=False,
    )

    assert len(alphas) == 3
    assert len(Q_basis) > 0


def test_block_lanczos_cy_eigenvalues_block2():
    h_op, basis, states, eigvals = create_diagonal_h_and_basis(6)

    # Block size 2: two starting vectors
    np.random.seed(42)
    psi0 = []
    for _i in range(2):
        st = ManyBodyState()
        for j in range(6):
            st[basis.type.from_bytes(states[j])] = np.random.randn()
        psi0.append(st)

    psi0, _ = block_normalize(psi0, mpi=False, comm=None)

    alphas, betas, _Q_basis, _W = block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged_fn=lambda a, b, **kw: len(a) >= 3,
        reort="full",
        max_iter=3,
        verbose=False,
    )

    from impurityModel.ed.BlockLanczosArray import _build_full_T

    T = _build_full_T(alphas, betas[: len(alphas) - 1])
    eigs_T = np.sort(np.linalg.eigvalsh(T))
    np.testing.assert_allclose(eigs_T, np.sort(eigvals), atol=1e-10)


def test_irlm_cy_tight_binding():
    h_op, basis = create_tight_binding_h_and_basis()

    np.random.seed(42)
    psi0 = []
    for _ in range(2):
        st = ManyBodyState()
        for b in basis.local_basis:
            st[b] = np.random.randn()
        psi0.append(st)

    psi0, _ = block_normalize(psi0, False, None)

    eigs, _evecs = implicitly_restarted_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=4,
        max_subspace_blocks=10,
        tol=1e-6,
        max_restarts=20,
        verbose=False,
    )

    assert len(eigs) == 4
    # Eigenvalues should be sorted (lowest first)
    np.testing.assert_array_less(eigs[:-1], eigs[1:] + 1e-12)


def test_trlm_cy_tight_binding_eigenvalue_accuracy():
    from impurityModel.ed.BlockLanczos import thick_restart_block_lanczos_cy

    h_op, basis = create_tight_binding_h_and_basis()

    # Compute exact eigenvalues via full diagonalization
    exact_eigs = get_exact_tight_binding_eigenvalues(h_op, basis)

    np.random.seed(42)
    psi0 = []
    for _ in range(2):
        st = ManyBodyState()
        for b in basis.local_basis:
            st[b] = np.random.randn()
        psi0.append(st)

    psi0, _ = block_normalize(psi0, False, None)

    eigs, _evecs = thick_restart_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=4,
        max_subspace_blocks=10,
        tol=1e-8,
        max_restarts=20,
        verbose=False,
    )

    assert len(eigs) == 4
    np.testing.assert_allclose(eigs, exact_eigs[:4], atol=1e-6)


def test_irlm_cy_partial_reort_convergence():
    n_states = 12
    states = [(1 << i).to_bytes(8, "little") for i in range(n_states)]
    np.random.seed(42)
    H_mat = np.random.randn(n_states, n_states)
    H_mat = H_mat + H_mat.T
    hop = {}
    for i in range(n_states):
        for j in range(n_states):
            if abs(H_mat[i, j]) > 1e-10:
                hop[((i, "c"), (j, "a"))] = float(H_mat[i, j])
    basis = Basis(
        impurity_orbitals={0: [list(range(n_states))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=False,
    )
    h_op = ManyBodyOperator(hop)
    psi0 = [ManyBodyState({b: np.random.randn() for b in basis.local_basis})]
    psi0, _ = block_normalize(psi0, False, None)

    # Run IRLM with partial reortho and verify it converges without blowing up beta to 10^39
    eigvals, _ = implicitly_restarted_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=4,
        max_subspace_blocks=8,
        tol=1e-8,
        max_restarts=10,
        verbose=False,
        reort="partial",
    )
    # verify against exact diagonalization
    exact_eigvals = np.sort(np.linalg.eigvalsh(H_mat))[:4]
    np.testing.assert_allclose(eigvals, exact_eigvals, atol=1e-7)


def test_trlm_cy_partial_reort_convergence():
    n_states = 12
    states = [(1 << i).to_bytes(8, "little") for i in range(n_states)]
    np.random.seed(42)
    H_mat = np.random.randn(n_states, n_states)
    H_mat = H_mat + H_mat.T
    hop = {}
    for i in range(n_states):
        for j in range(n_states):
            if abs(H_mat[i, j]) > 1e-10:
                hop[((i, "c"), (j, "a"))] = float(H_mat[i, j])
    basis = Basis(
        impurity_orbitals={0: [list(range(n_states))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=False,
    )
    h_op = ManyBodyOperator(hop)
    psi0 = [ManyBodyState({b: np.random.randn() for b in basis.local_basis})]
    psi0, _ = block_normalize(psi0, False, None)

    # Run TRLM with partial reortho and verify it converges
    eigvals, _ = thick_restart_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=4,
        max_subspace_blocks=8,
        tol=1e-8,
        max_restarts=10,
        verbose=False,
        reort="partial",
    )
    # verify against exact diagonalization
    exact_eigvals = np.sort(np.linalg.eigvalsh(H_mat))[:4]
    np.testing.assert_allclose(eigvals, exact_eigvals, atol=1e-7)


def test_irlm_cy_selective_reort_orthogonality():
    n_states = 12
    states = [(1 << i).to_bytes(8, "little") for i in range(n_states)]
    np.random.seed(42)
    H_mat = np.random.randn(n_states, n_states)
    H_mat = H_mat + H_mat.T
    hop = {}
    for i in range(n_states):
        for j in range(n_states):
            if abs(H_mat[i, j]) > 1e-10:
                hop[((i, "c"), (j, "a"))] = float(H_mat[i, j])
    basis = Basis(
        impurity_orbitals={0: [list(range(n_states))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=False,
    )
    h_op = ManyBodyOperator(hop)
    psi0 = [ManyBodyState({b: np.random.randn() for b in basis.local_basis})]
    psi0, _ = block_normalize(psi0, False, None)

    _eigvals, eigvecs = implicitly_restarted_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=6,
        max_subspace_blocks=10,
        tol=1e-8,
        max_restarts=15,
        verbose=False,
        reort="selective",
    )

    # Assert eigenvectors are orthogonal
    ov = inner_multi(eigvecs, eigvecs)
    err = np.linalg.norm(ov - np.eye(len(eigvecs)))
    assert err < 1e-10


def test_trlm_cy_selective_reort_orthogonality():
    n_states = 12
    states = [(1 << i).to_bytes(8, "little") for i in range(n_states)]
    np.random.seed(42)
    H_mat = np.random.randn(n_states, n_states)
    H_mat = H_mat + H_mat.T
    hop = {}
    for i in range(n_states):
        for j in range(n_states):
            if abs(H_mat[i, j]) > 1e-10:
                hop[((i, "c"), (j, "a"))] = float(H_mat[i, j])
    basis = Basis(
        impurity_orbitals={0: [list(range(n_states))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=False,
    )
    h_op = ManyBodyOperator(hop)
    psi0 = [ManyBodyState({b: np.random.randn() for b in basis.local_basis})]
    psi0, _ = block_normalize(psi0, False, None)

    _eigvals, eigvecs = thick_restart_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=6,
        max_subspace_blocks=10,
        tol=1e-8,
        max_restarts=15,
        verbose=False,
        reort="selective",
    )

    # Assert eigenvectors are orthogonal
    ov = inner_multi(eigvecs, eigvecs)
    err = np.linalg.norm(ov - np.eye(len(eigvecs)))
    assert err < 1e-10


def test_converged_views_equivalence():
    """block_lanczos_cy passes buffer *views* (alphas_buf[:it+1]) to converged_fn
    instead of rebuilding lists->arrays each step. This pins that the data seen by
    converged_fn at step k equals the first k+1 accumulated alpha/beta blocks (exactly
    what the removed np.array(alphas_list) would have produced) and that convergence /
    eigenvalues are unchanged.
    """
    from impurityModel.ed.BlockLanczosArray import _build_full_T

    h_op, basis, states, eigvals = create_diagonal_h_and_basis(6)
    st = ManyBodyState()
    for i in range(6):
        st[basis.type.from_bytes(states[i])] = 1.0 / np.sqrt(6)
    psi0 = [st]

    snapshots = []

    def recording_converged(alphas, betas, **kw):
        snapshots.append((np.array(alphas, copy=True), np.array(betas, copy=True)))
        return False

    alphas, betas, _Q_basis, _W = block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged_fn=recording_converged,
        reort="full",
        max_iter=6,
        verbose=False,
    )

    alphas = np.asarray(alphas)
    betas = np.asarray(betas)

    # converged_fn was called at least once, and each call k saw exactly the first
    # k+1 accumulated blocks (a prefix view consistent with the final buffers).
    assert len(snapshots) >= 1
    for k, (a_snap, b_snap) in enumerate(snapshots):
        np.testing.assert_array_equal(a_snap, alphas[: k + 1])
        np.testing.assert_array_equal(b_snap, betas[: k + 1])

    # Convergence/eigenvalues unchanged: T's spectrum matches the dense reference.
    T = _build_full_T(alphas, betas[: len(alphas) - 1])
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(T)), np.sort(eigvals), atol=1e-10)


def test_estimate_orthonormality_bounded_buffer_bit_identical():
    """Phase 1: the caller-provided ping-pong buffer (`out`) and the beta-norm history
    (`beta_norms`) must reproduce the allocating/no-history path bit-for-bit."""
    from impurityModel.ed.BlockLanczosArray import estimate_orthonormality

    rng = np.random.default_rng(77)
    p, k = 3, 12
    alphas = rng.standard_normal((k, p, p)) + 1j * rng.standard_normal((k, p, p))
    alphas = 0.5 * (alphas + np.conj(alphas.transpose(0, 2, 1)))
    betas = rng.standard_normal((k, p, p)) + 1j * rng.standard_normal((k, p, p))

    W_ref = np.zeros((2, 1, p, p), dtype=complex)
    W_ref[1, 0] = np.eye(p)
    W_buf = W_ref.copy()
    bufs = (np.empty((2, k + 2, p, p), dtype=complex), np.empty((2, k + 2, p, p), dtype=complex))
    hist = []
    for it in range(k):
        widths = [p] * (it + 2)
        W_ref = estimate_orthonormality(W_ref, alphas[: it + 1], betas[: it + 1], block_widths=widths, N=1000.0)
        W_buf = estimate_orthonormality(
            W_buf,
            alphas[: it + 1],
            betas[: it + 1],
            block_widths=widths,
            N=1000.0,
            out=bufs[it % 2],
            beta_norms=hist,
        )
        np.testing.assert_array_equal(W_buf, W_ref)  # bit-for-bit
        hist.append(float(np.linalg.svd(betas[it, :p, :p], compute_uv=False)[0]))
    # resume-style None placeholders fall back to on-demand norms, still bit-identical
    hist_holes = [None] * len(hist)
    W_h = np.zeros((2, 1, p, p), dtype=complex)
    W_h[1, 0] = np.eye(p)
    for it in range(k):
        widths = [p] * (it + 2)
        W_h = estimate_orthonormality(
            W_h,
            alphas[: it + 1],
            betas[: it + 1],
            block_widths=widths,
            N=1000.0,
            out=bufs[it % 2],
            beta_norms=hist_holes,
        )
    np.testing.assert_array_equal(W_h, W_ref)
