import numpy as np

from impurityModel.ed.BlockLanczos import (
    block_lanczos,
    lanczos_step,
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

    alphas, _betas, Q_basis = block_lanczos(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged=lambda a, b, **kw: len(a) >= 3,
        reort=True,
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

    alphas, betas, _Q_basis = block_lanczos(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged=lambda a, b, **kw: len(a) >= 6,
        reort=True,
        max_iter=6,
        verbose=False,
    )

    from impurityModel.ed.BlockLanczosArray import _build_full_T

    T = _build_full_T(alphas, betas[: len(alphas) - 1])
    eigs_T = np.linalg.eigvalsh(T)
    np.testing.assert_allclose(np.sort(eigs_T), np.sort(eigvals), atol=1e-10)


def create_tight_binding_h_and_basis():
    import itertools

    from impurityModel.ed.manybody_basis import Basis

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
            amp = h_psi_j.get(bi)
            H_mat[i, j] = 0.0 if amp is None else amp[0]
    return np.sort(np.linalg.eigvalsh(H_mat).real)


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

    # With reort=False, just verify no crash
    alphas, _betas, Q_basis = block_lanczos(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged=lambda a, b, **kw: len(a) >= 3,
        reort=False,
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

    alphas, betas, _Q_basis = block_lanczos(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged=lambda a, b, **kw: len(a) >= 3,
        reort=True,
        max_iter=3,
        verbose=False,
    )

    from impurityModel.ed.BlockLanczosArray import _build_full_T

    T = _build_full_T(alphas, betas[: len(alphas) - 1])
    eigs_T = np.sort(np.linalg.eigvalsh(T))
    np.testing.assert_allclose(eigs_T, np.sort(eigvals), atol=1e-10)


def test_converged_views_equivalence():
    """block_lanczos passes buffer *views* (alphas_buf[:it+1]) to converged_fn
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

    alphas, betas, _Q_basis = block_lanczos(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged=recording_converged,
        reort=True,
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


def test_block_lanczos_cy_store_krylov_false_matches_tail_of_stored_run():
    """``store_krylov=False``'s contract (_lanczos_step.pxi's budget-exhaustion branch):
    the returned ``Q_basis`` is exactly the last two blocks a ``store_krylov=True`` run
    would have accumulated, not merely "some" two-block tail.

    R0c gap: this path had no default-run assertion (only the env-gated
    ``RUN_SLICING_PROBE=1`` probe exercises it, e.g. via ``spectral_bounds`` in
    test_slicing_probe.py), yet R6 restructures exactly this branch.
    """
    h_op, basis, states, _ = create_diagonal_h_and_basis(6)
    n_iter = 3  # p=2 * 3 blocks = 6 = the fixture's dimension; more would hit invariant_subspace

    rng = np.random.default_rng(0)
    psi0 = []
    for _ in range(2):
        st = ManyBodyState()
        for s in states:
            st[basis.type.from_bytes(s)] = complex(*rng.standard_normal(2))
        psi0.append(st)
    psi0, _ = block_normalize(psi0, mpi=False, comm=None)

    stored_alphas, stored_betas, stored_Q = block_lanczos(
        [p for p in psi0],
        h_op,
        basis,
        lambda *a, **k: False,
        verbose=False,
        reort=False,
        max_iter=n_iter,
    )
    assert len(stored_Q) == 2 * n_iter  # p=2, n_iter blocks, flat width-1 list

    tail_alphas, tail_betas, tail_Q = block_lanczos(
        [p for p in psi0],
        h_op,
        basis,
        lambda *a, **k: False,
        verbose=False,
        reort=False,
        max_iter=n_iter,
        store_krylov=False,
    )
    assert len(tail_Q) == 4  # the last TWO blocks (p=2 each): q_prev + q_curr

    np.testing.assert_array_equal(np.asarray(stored_alphas), np.asarray(tail_alphas))
    np.testing.assert_array_equal(np.asarray(stored_betas), np.asarray(tail_betas))

    stored_tail = stored_Q[-4:]
    ov = inner_multi(stored_tail, tail_Q)
    np.testing.assert_allclose(ov, np.eye(4), atol=1e-10)


def test_a_resumed_call_keeps_the_divergence_guards_scale(monkeypatch):
    """A resume replays the guard over the blocks it is handed: its first step is guarded against the
    scale the earlier call built, not treated as a fresh run's scale-establishing step (which went
    unchecked and re-seeded the scale on every resume -- every GF window once units ran in windows)."""
    from impurityModel.ed import BlockLanczosCore as BlockLanczos
    from impurityModel.test.support.lanczos_fixtures import MockBasis, get_test_system

    h_op, n, _eigvals_exact, basis_states = get_test_system()
    basis = MockBasis(n)
    np.random.seed(5)
    psi0 = []
    for _ in range(2):
        st = ManyBodyState()
        for b in basis_states:
            st += b * (np.random.rand() + 1j * np.random.rand())
        psi0.append(st)
    psi0, _ = block_normalize(psi0, mpi=False, comm=None)
    calls = []
    real = BlockLanczos.check_divergence

    def recording(beta_norm, alpha_norm, first_step, t_norm_max, h_norm_est, it_abs, comm):
        calls.append((it_abs, bool(first_step), t_norm_max, h_norm_est))
        return real(beta_norm, alpha_norm, first_step, t_norm_max, h_norm_est, it_abs, comm)

    monkeypatch.setattr(BlockLanczos, "check_divergence", recording)
    kw = dict(h_op=h_op, basis=basis, converged=lambda a, b, **k: False, reort=False, return_widths=True)
    a1, b1, q1, wid1 = block_lanczos(psi0=psi0, max_iter=4, **kw)
    whole = list(calls)
    calls.clear()
    block_lanczos(psi0=None, alphas_init=a1, betas_init=b1, Q_init=q1, block_widths_init=wid1, max_iter=2, **kw)
    first_resumed = calls[0]
    assert not first_resumed[1], "the resumed call's first step is not a scale-establishing step"
    assert first_resumed[2] >= max(t for _, _, t, _ in whole), "it carries the earlier call's scale"
