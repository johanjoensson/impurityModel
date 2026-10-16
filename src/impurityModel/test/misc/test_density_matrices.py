"""``density_matrices``: the scatter kernel against the pre-redesign implementation (the oracle).

The oracle fuses the ``c_k|psi>`` into one dense block and is only usable on small states; the new
kernel must reproduce it exactly -- including off-diagonals, which are where the fermionic sign and
the bit order live -- for square and rectangular requests, split amplitudes, and a forced
multi-round scatter.
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.density_matrices import build_density_matrices, occupations
from impurityModel.ed.mpi_comm import graph_alltoall_block
from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant
from impurityModel.test.support.reference_density_matrices import reference_density_matrices

N_ORB = 11  # two bytes: signs and bit order across a byte boundary


class _FakeBasis:
    def __init__(self, comm, n_orb=N_ORB):
        self.comm = comm
        self.is_distributed = comm is not None and comm.size > 1
        self.num_spin_orbitals = n_orb
        self.n_bytes = (n_orb + 7) // 8

    def redistribute_block(self, block):
        return graph_alltoall_block(block, self.n_bytes, self.comm) if self.is_distributed else block

    def redistribute_psis(self, *blocks):
        if not self.is_distributed:
            return list(blocks)
        combined = self.redistribute_block(ManyBodyState.from_states([c for b in blocks for c in b.to_states()]))
        out, off = [], 0
        for b in blocks:
            sub = combined.select(list(range(off, off + b.width)))
            sub.prune_rows(0.0)
            out.append(sub)
            off += b.width
        return out


def _det(occupied, n_orb=N_ORB):
    nbytes = (n_orb + 7) // 8
    b = bytearray(nbytes)
    for i in occupied:
        b[i // 8] |= 1 << (7 - i % 8)
    return SlaterDeterminant.from_bytes(bytes(b))


def _global_states(width, seed=3, n_dets=40, n_el=4):
    rng = np.random.default_rng(seed)
    dets = [c for c in itertools.combinations(range(N_ORB), n_el)]
    pick = rng.choice(len(dets), size=n_dets, replace=False)
    amps = rng.standard_normal((n_dets, width)) + 1j * rng.standard_normal((n_dets, width))
    return [dets[p] for p in pick], amps


def _local_states(comm, width, mode):
    dets, amps = _global_states(width)
    states = []
    for n in range(width):
        d = {}
        for i, occ in enumerate(dets):
            if mode == "owned" and i % comm.size != comm.rank:
                continue
            a = amps[i, n] / (comm.size if mode == "split" else 1)
            d[_det(occ)] = a
        states.append(ManyBodyState(d))
    return states


@pytest.mark.mpi
@pytest.mark.parametrize("mode", ["owned", "split"])
@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize(
    "left,right",
    [(None, None), ([0, 3, 9], [0, 3, 9]), ([1, 8, 2], [10, 0, 9, 8]), ([5, 5, 2], None)],
)
@pytest.mark.parametrize("rounds", [None, 1, 5, 400])  # 400: most rounds are empty on some rank
def test_matches_the_reference(mode, width, left, right, rounds):
    comm = MPI.COMM_WORLD
    basis = _FakeBasis(comm)
    psis = _local_states(comm, width, mode)
    want = reference_density_matrices(basis, psis, left, right)
    got = build_density_matrices(basis, psis, left, right, rounds=rounds)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, atol=1e-12)


@pytest.mark.mpi
def test_is_a_density_matrix():
    comm = MPI.COMM_WORLD
    basis = _FakeBasis(comm)
    rho = build_density_matrices(basis, _local_states(comm, 2, "owned"))
    for r in rho:
        np.testing.assert_allclose(r, r.conj().T, atol=1e-12)
    # Each state has 4 electrons per determinant, so trace(rho_n) = <N> = 4 <psi|psi>.
    dets, amps = _global_states(2)
    np.testing.assert_allclose([np.trace(r).real for r in rho], 4 * (abs(amps) ** 2).sum(axis=0), atol=1e-10)


@pytest.mark.mpi
@pytest.mark.parametrize("mode", ["owned", "split"])
def test_occupations_are_the_diagonal(mode):
    comm = MPI.COMM_WORLD
    basis = _FakeBasis(comm)
    psis = _local_states(comm, 3, mode)
    want = np.real(np.diagonal(reference_density_matrices(basis, psis), axis1=1, axis2=2))
    np.testing.assert_allclose(occupations(basis, psis), want, atol=1e-12)
    np.testing.assert_allclose(occupations(basis, psis, [7, 1]), want[:, [7, 1]], atol=1e-12)


@pytest.mark.mpi
def test_accepts_a_block_and_an_empty_rank():
    comm = MPI.COMM_WORLD
    basis = _FakeBasis(comm)
    # The oracle cannot take an empty rank (its width-0 placeholders deadlock), so the same global
    # state is given to it split over all ranks and to the kernel held entirely by rank 0.
    want = reference_density_matrices(basis, _local_states(comm, 2, "split"))
    dets, amps = _global_states(2)
    psis = [ManyBodyState(width=1) for _ in range(2)]
    if comm.rank == 0:
        psis = [ManyBodyState({_det(o): amps[i, n] for i, o in enumerate(dets)}) for n in range(2)]
    got = build_density_matrices(basis, ManyBodyState.from_states(psis))
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_counting_orbitals_above_gives_the_same_rho():
    """(-1)^(occupied above k) differs from (-1)^(occupied below k) by (-1)^(N-1) per determinant;
    rho multiplies two such signs from the same N-electron sector, so the factor cancels."""
    basis = _FakeBasis(None)
    psis = _local_states(MPI.COMM_SELF, 1, "owned")
    ref = reference_density_matrices(basis, psis)
    dets, amps = _global_states(1)
    # Direct evaluation with the "above" convention.
    rho = np.zeros((N_ORB, N_ORB), dtype=complex)
    phis = {}
    for occ, a in zip(dets, amps[:, 0]):
        for k in occ:
            s = (-1) ** sum(1 for o in occ if o > k)
            phis.setdefault(tuple(o for o in occ if o != k), {})[k] = s * a
    for g in phis.values():
        for i, ai in g.items():
            for j, aj in g.items():
                rho[i, j] += ai * np.conj(aj)
    np.testing.assert_allclose(rho, ref[0], atol=1e-12)
