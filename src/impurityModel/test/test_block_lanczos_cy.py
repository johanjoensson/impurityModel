import pytest
import numpy as np
import scipy.linalg as sp


from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import inner_multi, ManyBodyState, ManyBodyOperator
from impurityModel.ed.BlockLanczosArray import block_normalize
from impurityModel.ed.BlockLanczosArray import Reort

from impurityModel.ed.BlockLanczos import (
    block_lanczos_cy,
    thick_restart_block_lanczos_cy,
    implicitly_restarted_block_lanczos_cy,
    block_lanczos_step_cy,
)


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
    for i in range(2):
        st = ManyBodyState()
        for j in range(6):
            st[basis.type.from_bytes(states[j])] = np.random.randn()
        psi0.append(st)

    psi0, _ = block_normalize(psi0, mpi=False, comm=None)

    alphas, betas, Q_basis, W = block_lanczos_cy(
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

    alphas, betas, Q_basis, W = block_lanczos_cy(
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

    eigs, evecs = thick_restart_block_lanczos_cy(
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

    eigs, evecs = implicitly_restarted_block_lanczos_cy(
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
    from impurityModel.ed.BlockLanczosArray import block_normalize
    import numpy as np
    from impurityModel.ed.ManyBodyUtils import ManyBodyState
    from impurityModel.ed.BlockLanczos import thick_restart_block_lanczos_cy

    h_op, basis = create_tight_binding_h_and_basis()

    np.random.seed(42)
    psi0 = []
    for _ in range(2):
        st = ManyBodyState()
        for b in basis.local_basis:
            st[b] = np.random.randn()
        psi0.append(st)

    psi0, _ = block_normalize(psi0, False, None)

    eigs, evecs = thick_restart_block_lanczos_cy(
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
    for i in range(2):
        st = ManyBodyState()
        for j in range(6):
            st[basis.type.from_bytes(states[j])] = np.random.randn()
        psi0.append(st)

    psi0, _ = block_normalize(psi0, mpi=False, comm=None)

    alphas, betas, Q_basis, W = block_lanczos_cy(
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


def test_block_lanczos_cy_orthogonality_none():
    h_op, basis, states, _ = create_diagonal_h_and_basis(6)

    np.random.seed(42)
    psi0 = []
    for i in range(2):
        st = ManyBodyState()
        for j in range(6):
            st[basis.type.from_bytes(states[j])] = np.random.randn()
        psi0.append(st)

    psi0, _ = block_normalize(psi0, mpi=False, comm=None)

    # With reort='none', just verify no crash
    alphas, betas, Q_basis, W = block_lanczos_cy(
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
    for i in range(2):
        st = ManyBodyState()
        for j in range(6):
            st[basis.type.from_bytes(states[j])] = np.random.randn()
        psi0.append(st)

    psi0, _ = block_normalize(psi0, mpi=False, comm=None)

    alphas, betas, Q_basis, W = block_lanczos_cy(
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

    eigs, evecs = implicitly_restarted_block_lanczos_cy(
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

    eigs, evecs = thick_restart_block_lanczos_cy(
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

    eigvals, eigvecs = implicitly_restarted_block_lanczos_cy(
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

    eigvals, eigvecs = thick_restart_block_lanczos_cy(
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
