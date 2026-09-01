"""Shared Lanczos test fixtures: a mock serial basis, a small dense tight-binding
reference system, and a dense-matrix builder for a ManyBodyState-path basis.

Promoted out of individual test files (test_restarted_lanczos.py,
test_block_lanczos_reort_matrix.py, test_block_lanczos_array_empty_rank.py) once several
other test modules started importing them cross-file.
"""

import itertools

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant


class MockBasis:
    def __init__(self, size):
        self.size = size
        self.comm = None
        self._index_dict = {}

    def redistribute_psis(self, *blocks):
        return list(blocks)

    def redistribute_block(self, block):
        return block

    def add_states(self, states):
        pass


def get_test_system():
    n_sites = 8
    n_particles = 4

    # Generate all basis states
    combinations = list(itertools.combinations(range(n_sites), n_particles))
    states = []
    for c in combinations:
        val = sum(1 << i for i in c)
        states.append(SlaterDeterminant.from_bytes(val.to_bytes(8, byteorder="little")))

    # 1D Tight-binding hopping
    op_dict = {}
    for i in range(n_sites - 1):
        op_dict[((i, "c"), (i + 1, "a"))] = -1.0
        op_dict[((i + 1, "c"), (i, "a"))] = -1.0

    h_op = ManyBodyOperator(op_dict)

    # Build dense matrix for exact diagonalization
    N = len(states)
    H_dense = np.zeros((N, N), dtype=complex)

    basis_states = [ManyBodyState({sd: 1.0}) for sd in states]
    H_basis_states = h_op.apply_multi(basis_states)

    for j in range(N):
        for i, sd_i in enumerate(states):
            amp = H_basis_states[j].get(sd_i)
            H_dense[i, j] = 0.0 if amp is None else amp[0]

    eigvals_exact, _ = np.linalg.eigh(H_dense)

    return h_op, N, eigvals_exact, basis_states


def build_dense_matrix_from_manybody(h_op, basis_states):
    N = len(basis_states)
    H_dense = np.zeros((N, N), dtype=complex)
    states = [next(iter(b.keys())) for b in basis_states]
    H_basis_states = h_op.apply_multi(basis_states)
    for j in range(N):
        for i, sd_i in enumerate(states):
            amp = H_basis_states[j].get(sd_i)
            H_dense[i, j] = 0.0 if amp is None else amp[0]
    return H_dense


def _contiguous_counts_with_empty_last(global_N, size):
    """Partition ``global_N`` contiguous indices over ``size`` ranks, last empty.

    The last rank always gets 0; ``global_N`` is spread over the first
    ``size - 1`` ranks.  If ``size - 1 > global_N`` some leading ranks are empty
    too (an even stronger test).  For ``size == 1`` the single rank owns
    everything (no empty rank, but the test still exercises the kernel).
    """
    if size == 1:
        return [global_N]
    base, rem = divmod(global_N, size - 1)
    counts = [base + (1 if r < rem else 0) for r in range(size - 1)] + [0]
    assert sum(counts) == global_N
    assert counts[-1] == 0
    return counts


def deflating_start_block(n=40, seed=7):
    """``(h, psi0, exact)`` for a sweep that deflates at the first step and then runs on.

    ``h`` is diagonal with a two-fold ground state at 0 and a nondegenerate ladder
    ``1, 2, 3, ...`` above it. The start block spans both ground-state directions plus one
    generic vector orthogonal to them, so ``H q - q alpha`` has rank 1 where the block has
    rank 3: the block narrows at the first step and the sweep then runs its full ``m``
    blocks at the reduced width, ending with a live residual block.

    That is *deflation*, not closure of the block-Krylov space -- ``m_actual == m`` and the
    trailing residual block is present -- and both restarted solvers used to conflate the
    two. It is the smallest shape that separates them, so it is shared: TRLM must route it
    to its (width-aware) restart loop rather than extracting directly, and IRLM, whose
    uniform-width purge/restart genuinely cannot continue such a factorization, must report
    only the pairs it actually converged instead of whatever the last factorization held.

    ``exact`` is the full sorted spectrum; the lowest ``k`` of it is the answer for any
    ``num_wanted = k``.
    """
    import numpy as _np
    import scipy.sparse as _sps

    d = _np.concatenate([[0.0, 0.0], _np.arange(1.0, n - 1.0)])
    h = _sps.csr_matrix(_np.diag(d).astype(complex))
    rng = _np.random.default_rng(seed)
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    v[0] = v[1] = 0.0
    v /= _np.linalg.norm(v)
    psi0 = _np.zeros((n, 3), dtype=complex)
    psi0[0, 0] = 1.0
    psi0[1, 1] = 1.0
    psi0[:, 2] = v
    return h, psi0, _np.sort(d)
