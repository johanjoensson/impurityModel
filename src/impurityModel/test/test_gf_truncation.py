"""Tests for the truncation_threshold cap on the sparse-kernel Green's function
(_CappedBasisProxy: freeze-growth + importance-ranked boundary admission).

The strong oracle: once the cap freezes the recurrence support, the returned
continued fraction must equal the dense resolvent of H projected onto the retained
determinant set (P H P) — exactly, not approximately."""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.basis_transcription import build_dense_matrix
from impurityModel.ed.gf_solvers import block_Green_sparse
from impurityModel.ed.greens_function import _CappedBasisProxy, calc_G
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, SlaterDeterminant

DELTA = 0.1
OMEGA = np.linspace(-8.0, 8.0, 41)


def _det(occupied):
    """Determinant with the given orbitals occupied (MSB-first: orbital i = bit 7-i)."""
    b = 0
    for i in occupied:
        b |= 1 << (7 - i)
    return SlaterDeterminant.from_bytes(bytes([b]))


def _siam_6():
    """Single-impurity Anderson model, 6 spin-orbitals (0,1 imp; 2,3 val; 4,5 cond)."""
    ed_, u, ev, ec, v = -1.0, 4.0, -3.0, 3.0, 0.5
    terms = {}
    for o in (0, 1):
        terms[((o, "c"), (o, "a"))] = ed_
    for o in (2, 3):
        terms[((o, "c"), (o, "a"))] = ev
    for o in (4, 5):
        terms[((o, "c"), (o, "a"))] = ec
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = u
    for a, b in ((0, 2), (1, 3), (0, 4), (1, 5)):
        terms[((a, "c"), (b, "a"))] = v
        terms[((b, "c"), (a, "a"))] = v
    return ManyBodyOperator(terms)


_IMP = {0: [[0, 1]]}
_BATHS = ({0: [[2, 3]]}, {0: [[4, 5]]})


def _seeds():
    """Two seed columns in the N=3 sector (the reachable space is the full sector, 20 dets)."""
    return [
        ManyBodyState({_det([0, 2, 3]): 1.0 + 0j, _det([1, 2, 3]): 0.5 + 0j}),
        ManyBodyState({_det([0, 1, 2]): 1.0 + 0j}),
    ]


def _excited_basis(cap, comm=None):
    seed_support = sorted({state for s in _seeds() for state in s})
    return Basis(
        _IMP,
        _BATHS,
        initial_basis=seed_support,
        truncation_threshold=cap,
        comm=comm,
        verbose=False,
    )


def _redistribute_as_width1(basis, states, n):
    """Redistribute ``n`` seeds through width-1 blocks on every rank, then unpack back to
    flat states. A bare ManyBodyState() placeholder for a non-owning rank is the width-0
    polymorphic zero once the flat and block classes merge (Phase 7 step 3), an
    asymmetric mismatch against another rank's populated (eventually width-1) seeds that
    would deadlock redistribute_psis' collective. from_states forces an explicit width-1
    block -- empty or not -- on every rank instead."""
    blocks = (
        [ManyBodyState.from_states([s]) for s in states]
        if states is not None
        else [ManyBodyState(width=1) for _ in range(n)]
    )
    return [blk.to_states()[0] for blk in basis.redistribute_psis(blocks)]


def _run_capped(cap, reort=None, comm=None):
    basis = _excited_basis(cap, comm=comm)
    # redistribute_psis SUMS per-rank contributions (production seeds are partial per
    # rank), so only rank 0 may provide the full amplitudes here.
    seeds_full = _seeds()
    seeds = _redistribute_as_width1(basis, seeds_full if comm is None or comm.rank == 0 else None, len(seeds_full))
    info = {}
    alphas, betas, r = block_Green_sparse(
        _siam_6(),
        seeds,
        basis,
        DELTA,
        reort=reort,
        verbose=False,
        cap_info=info,
    )
    return calc_G(alphas, betas, r, OMEGA, 0.0, DELTA), info


def _dense_reference_on(retained_keys, comm=None):
    """G(w) = V^dag ((w + i*delta) - H)^{-1} V on the space spanned by retained_keys."""
    basis = Basis(_IMP, _BATHS, initial_basis=sorted(retained_keys), comm=comm, verbose=False)
    H = np.asarray(build_dense_matrix(basis, _siam_6()))
    index = {det: i for i, det in enumerate(sorted(retained_keys))}
    V = np.zeros((len(index), len(_seeds())), dtype=complex)
    for j, seed in enumerate(_seeds()):
        for det, amp in seed.items():
            V[index[det], j] = amp[0]
    G = np.empty((len(OMEGA), V.shape[1], V.shape[1]), dtype=complex)
    for k, w in enumerate(OMEGA):
        G[k] = V.conj().T @ np.linalg.solve((w + 1j * DELTA) * np.eye(len(index)) - H, V)
    return G


def test_cap_above_reachable_space_is_identity():
    """A cap the recurrence never reaches must not change the result at all."""
    g_uncapped, info_u = _run_capped(np.inf)
    g_capped, info_c = _run_capped(1000)
    assert info_u["proxy"] is None
    assert info_c["proxy"] is not None and not info_c["cap_hit"]
    np.testing.assert_array_equal(g_capped, g_uncapped)


def test_uncapped_matches_dense_full_sector():
    """Sanity: the uncapped GF equals the dense resolvent on the full N=3 sector."""
    g, _ = _run_capped(np.inf)
    # the reachable space: run once capped-with-huge-cap to collect the retained keys
    _, info = _run_capped(1000)
    retained = info["proxy"].retained_keys()
    # H conserves N_up and N_dn separately; the seeds span (N_dn, N_up) = (2,1) + (1,2),
    # two 9-determinant sectors of the 20-determinant N=3 space.
    assert len(retained) == 18
    np.testing.assert_allclose(g, _dense_reference_on(retained), atol=1e-10)


@pytest.mark.parametrize("reort", [None, "full", "partial"])
@pytest.mark.parametrize("cap", [6, 12, 17])
def test_capped_gf_equals_dense_php_resolvent(cap, reort):
    """The oracle: the capped GF is the exact GF of H projected on the retained set."""
    g, info = _run_capped(cap, reort=reort)
    assert info["cap_hit"]
    assert info["retained_size"] <= cap
    retained = info["proxy"].retained_keys()
    assert len(retained) == info["retained_size"]
    np.testing.assert_allclose(g, _dense_reference_on(retained), atol=1e-9)
    # causality: Im G_ii <= 0 on the retarded mesh
    assert np.all(np.diagonal(g.imag, axis1=1, axis2=2) <= 1e-12)


def test_cap_at_seed_size_freezes_immediately():
    """cap == initial basis size: nothing new is ever admitted; still exact on P."""
    basis = _excited_basis(np.inf)
    seed_size = basis.size
    g, info = _run_capped(seed_size)
    assert info["cap_hit"] and info["retained_size"] <= seed_size
    np.testing.assert_allclose(g, _dense_reference_on(info["proxy"].retained_keys()), atol=1e-9)


def test_admission_prefers_large_amplitude_rows():
    """Unit test of the overflow bisection on a hand-built proxy (no MPI)."""

    class _FakeBasis:
        comm = None
        is_distributed = False

        def __init__(self, local):
            self.local_basis = local
            self.size = len(local)
            self.n_bytes = 1

        def redistribute_block(self, block):
            return block

        def redistribute_psis(self, psis):
            return psis

    seed_dets = [_det([0, 1, 2])]
    proxy = _CappedBasisProxy(_FakeBasis(seed_dets), cap=3)
    incoming = ManyBodyState.from_states(
        [
            ManyBodyState(
                {
                    _det([0, 1, 2]): 1.0 + 0j,  # already retained
                    _det([0, 1, 3]): 0.9 + 0j,  # strongest candidate
                    _det([0, 1, 4]): 0.5 + 0j,  # second
                    _det([0, 1, 5]): 0.1 + 0j,  # should be rejected (cap leaves 2 slots)
                    _det([0, 2, 3]): 0.05 + 0j,  # rejected
                }
            )
        ]
    )
    out = proxy.redistribute_block(incoming)
    assert proxy.cap_hit and proxy.retained_size == 3
    kept = set(proxy.retained_keys())
    assert kept == {_det([0, 1, 2]), _det([0, 1, 3]), _det([0, 1, 4])}
    assert len(out) == 3  # the block was projected onto the retained set
    # post-freeze: new rows are dropped, retained rows pass
    later = ManyBodyState.from_states([ManyBodyState({_det([0, 1, 3]): 1.0 + 0j, _det([1, 2, 3]): 2.0 + 0j})])
    out2 = proxy.redistribute_block(later)
    assert len(out2) == 1
    assert proxy.retained_size == 3


@pytest.mark.mpi
def test_capped_gf_mpi_matches_dense_php():
    """Distributed run: cap respected, collective decisions consistent, oracle holds.

    With 2+ ranks and a small cap, some rank will own few or zero retained rows —
    the empty-rank edge case must not deadlock or diverge."""
    comm = MPI.COMM_WORLD
    for cap in (6, 12):
        g, info = _run_capped(cap, comm=comm)
        assert info["cap_hit"]
        assert info["retained_size"] <= cap
        # retained keys are rank-local; gather for the dense reference
        local = info["proxy"].retained_keys()
        gathered = comm.allgather(local)
        retained = sorted({k for part in gathered for k in part})
        assert len(retained) == info["retained_size"]
        g_ref = _dense_reference_on(retained)  # serial dense reference on every rank
        np.testing.assert_allclose(g, g_ref, atol=1e-9)


@pytest.mark.mpi
def test_capped_gf_mpi_matches_serial():
    """Same model, same cap: the MPI result agrees with the serial-equivalent physics
    (identical retained sets need not hold near ties; here amplitudes are distinct)."""
    comm = MPI.COMM_WORLD
    g_mpi, info = _run_capped(12, comm=comm)
    g_serial, info_serial = _run_capped(12, comm=None)
    np.testing.assert_allclose(g_mpi, g_serial, atol=1e-8)
    assert info["retained_size"] == info_serial["retained_size"]


def test_array_path_probe_respects_cap():
    """block_Green (array kernel): the probe loop must stop within ONE H-application
    batch of crossing truncation_threshold, not run all five probe rounds first."""
    from impurityModel.ed.BlockLanczosArray import Reort
    from impurityModel.ed.gf_solvers import block_Green

    cap = 5
    basis = _excited_basis(cap)
    hOp = _siam_6()
    # One full H fanout from the initial basis: the largest support any single
    # probe batch can add before the in-loop cap check fires.
    probe = ManyBodyState.from_states([ManyBodyState(dict.fromkeys(basis.local_basis, 1.0 + 0j))])
    one_fanout = set(basis.local_basis) | set(hOp.apply_block(probe, 0).support_keys(0.0))
    assert cap < len(one_fanout) < 18  # the cap must bind inside the first probe round

    alphas, betas, r = block_Green(hOp, _seeds(), basis, DELTA, Reort.NONE, verbose=False)
    assert basis.size > cap  # the cap was crossed (expansion did happen)
    assert basis.size <= len(one_fanout)  # ...but by at most one apply batch
    g = calc_G(alphas, betas, r, OMEGA, 0.0, DELTA)
    assert np.all(np.diagonal(g.imag, axis1=1, axis2=2) <= 1e-12)
