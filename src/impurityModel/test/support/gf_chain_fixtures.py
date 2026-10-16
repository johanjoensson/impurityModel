"""A 14-orbital chain whose Green's-function support spreads a shell or two per block.

Shared by the support-freeze tests: the 7-electron sector holds 3432 determinants, so a recurrence is still
growing after several windows of blocks (a dense random hopping matrix saturates its sector in eight blocks),
and with a ``slaterWeightMin`` cutoff the support trickles -- a few determinants per block for hundreds of
blocks -- instead of closing.
"""

import numpy as np

from impurityModel.ed.basis_transcription import build_dense_matrix
from impurityModel.ed.gf_solvers import block_Green_sparse
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.test.basis.test_sparse_matrix_build import _basis, _det
from impurityModel.test.gf.test_gf_truncation import _redistribute_as_width1

N_ORB = 14
SECTOR = 3432  # 7 electrons in 14 orbitals
DELTA = 0.1
OMEGA = np.linspace(-6.0, 6.0, 31)
CAP = 10**9  # finite, so the capped proxy is installed; far above anything these recurrences reach


def chain_operator(n_orb=N_ORB, seed=3):
    """Nearest-neighbour hopping and an on-site interaction on every other bond: the Krylov support spreads a
    shell or two per block."""
    rng = np.random.default_rng(seed)
    terms = {((o, "c"), (o, "a")): float(rng.normal()) for o in range(n_orb)}
    for a in range(n_orb - 1):
        v = 0.4 + 0.1 * float(rng.normal())
        terms[((a, "c"), (a + 1, "a"))] = v
        terms[((a + 1, "c"), (a, "a"))] = v
    for a in range(0, n_orb - 1, 2):
        terms[((a, "c"), (a + 1, "c"), (a + 1, "a"), (a, "a"))] = 2.0
    return ManyBodyOperator(terms)


def chain_seeds():
    return [
        ManyBodyState({_det([0, 1, 2, 3, 4, 5, 6]): 1.0 + 0j, _det([0, 1, 2, 3, 4, 5, 7]): 0.5 + 0j}),
        ManyBodyState({_det([1, 2, 3, 4, 5, 6, 7]): 1.0 + 0j}),
    ]


def run_chain(comm=None, op=None, reort=None, verbose=False, cap=CAP, slater_min=0.0, eval_meshes=None):
    """``block_Green_sparse`` on the chain from its two seeds: ``(alphas, betas, r, cap_info)``."""
    basis = _basis(sorted({d for s in chain_seeds() for d in s}), comm, n_orb=N_ORB)
    basis.truncation_threshold = cap
    full = chain_seeds()
    seeds = _redistribute_as_width1(basis, full if comm is None or comm.rank == 0 else None, len(full))
    info = {}
    alphas, betas, r = block_Green_sparse(
        chain_operator() if op is None else op,
        seeds,
        basis,
        DELTA,
        reort=reort,
        verbose=verbose,
        cap_info=info,
        slaterWeightMin=slater_min,
        eval_meshes=eval_meshes,
    )
    return alphas, betas, r, info


def exact_chain_g(keys, seed=3):
    """G(w) = V^dag ((w + i delta) - H)^{-1} V on the span of ``keys`` (determinants), by one eigh."""
    basis = _basis(sorted(keys), None, n_orb=N_ORB)
    h = np.asarray(build_dense_matrix(basis, chain_operator(seed=seed)))
    index = {bytes(d.to_bytearray()[:2]): i for i, d in enumerate(basis.local_basis)}
    v = np.zeros((h.shape[0], len(chain_seeds())), dtype=complex)
    for j, s in enumerate(chain_seeds()):
        for det, amp in s.items():
            v[index[bytes(det.to_bytearray()[:2])], j] = amp[0]
    assert not np.any(h.imag), "premise: a real Hamiltonian (a real eigh is ~10x cheaper)"
    e, u = np.linalg.eigh(h.real)
    uv = u.conj().T @ v
    return np.einsum("ki,wk,kj->wij", uv.conj(), 1.0 / ((OMEGA[:, None] + 1j * DELTA) - e[None, :]), uv)
