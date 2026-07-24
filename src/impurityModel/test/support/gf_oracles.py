"""Shared Green's-function test oracles: a small SIAM-6 model, its seeds/basis, and dense
resolvent/capped-basis references.

Promoted out of test_gf_bicgstab_driver.py once test_gmres.py and test_gf_cipsi_driver.py
started importing its helpers cross-file -- this is the canonical home for them now, with
test_gf_bicgstab_driver.py itself importing back like any other consumer.
"""

import itertools
import os

import numpy as np
from mpi4py import MPI

from impurityModel.ed.basis_transcription import build_dense_matrix
from impurityModel.ed.cg import block_bicgstab
from impurityModel.ed.greens_function import _CappedBasisProxy, get_Greens_function
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, block_inner_cy

DELTA = 0.15
OMEGA = np.linspace(-8.0, 8.0, 21)
MATSUBARA = 1j * np.pi * 0.1 * (2 * np.arange(8) + 1)


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
    """Two seed columns in the N=3 sector (reachable space: two 9-det (N_dn, N_up) sectors)."""
    return [
        ManyBodyState({_det([0, 2, 3]): 1.0 + 0j, _det([1, 2, 3]): 0.5 + 0j}),
        ManyBodyState({_det([0, 1, 2]): 1.0 + 0j}),
    ]


def _seed_basis(cap=np.inf, comm=None):
    seed_support = sorted({state for s in _seeds() for state in s})
    return Basis(
        _IMP,
        _BATHS,
        initial_basis=seed_support,
        truncation_threshold=cap,
        comm=comm,
        verbose=False,
    )


def _n3_sector_dets():
    """All 20 determinants with 3 of 6 orbitals occupied (closed under the SIAM-6 H)."""
    return [_det(c) for c in itertools.combinations(range(6), 3)]


def _dense_G_on(dets, z_values, comm=None, seeds=None):
    """G[k, i, j] = <seed_i| (z_k - H)^{-1} |seed_j> on the space spanned by ``dets``.

    ``seeds`` defaults to :func:`_seeds`; pass an explicit list of ``ManyBodyState``
    (e.g. linear combinations of the default seeds) to probe G against a different,
    rotated set of columns without touching the underlying Hamiltonian.
    """
    if seeds is None:
        seeds = _seeds()
    basis = Basis(_IMP, _BATHS, initial_basis=sorted(dets), comm=comm, verbose=False)
    H = np.asarray(build_dense_matrix(basis, _siam_6()))
    index = {det: i for i, det in enumerate(sorted(dets))}
    V = np.zeros((len(index), len(seeds)), dtype=complex)
    for j, seed in enumerate(seeds):
        for det, amp in seed.items():
            V[index[det], j] = amp[0]
    G = np.empty((len(z_values), V.shape[1], V.shape[1]), dtype=complex)
    for k, z in enumerate(z_values):
        G[k] = V.conj().T @ np.linalg.solve(z * np.eye(len(index)) - H, V)
    return G


def _capped_solve_with(solver, cap, z, comm=None):
    """The cap-oracle harness, parametrized over the linear solver (BiCGSTAB / GMRES):
    solve ``(z - H) X = seeds`` through a fresh ``_CappedBasisProxy`` and return the
    seed-projected ``G`` plus the proxy (whose retained keys define ``P``)."""
    basis = _seed_basis(comm=comm)
    proxy = _CappedBasisProxy(basis, cap)
    # redistribute_psis SUMS per-rank contributions, so only rank 0 provides amplitudes.
    # Each seed goes through its own explicit width-1 block rather than a bare
    # ManyBodyState() placeholder on the non-owning ranks: once the flat and block
    # classes merge (Phase 7 step 3), a bare placeholder is the width-0 polymorphic
    # zero, an asymmetric mismatch against the owning rank's populated (eventually
    # width-1) seeds that would deadlock redistribute_psis' collective.
    seeds_full = _seeds()
    owns_seeds = comm is None or comm.rank == 0
    seed_blocks = (
        [ManyBodyState.from_states([s]) for s in seeds_full]
        if owns_seeds
        else [ManyBodyState(width=1) for _ in seeds_full]
    )
    seeds = ManyBodyState.from_states([blk.to_states()[0] for blk in basis.redistribute_psis(*seed_blocks)])
    A = z - _siam_6()
    # Restart while unconverged, as the driver does: a near-pole z stagnates a single
    # BiCGSTAB pass (fresh shadow residual each call). GMRES restarts internally, so its
    # first call already converges and the loop is a no-op for it.
    X = ManyBodyState(width=seeds.width)
    info = {}
    for _ in range(10):
        X = solver(A, X, seeds, proxy, 0.0, atol=1e-12, info=info)
        if info["converged"]:
            break
    gram = block_inner_cy(seeds, X)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, gram, op=MPI.SUM)
    return gram, proxy


def _capped_solve(cap, z, comm=None):
    return _capped_solve_with(block_bicgstab, cap, z, comm=comm)


def _hyb_hop():
    return ManyBodyOperator(
        {
            ((0, "c"), (0, "a")): 0.3,
            ((1, "c"), (1, "a")): 0.7,
            ((2, "c"), (2, "a")): -0.5,
            ((3, "c"), (3, "a")): 0.4,
            ((0, "c"), (2, "a")): 0.25,
            ((2, "c"), (0, "a")): 0.25,
            ((1, "c"), (3, "a")): 0.25,
            ((3, "c"), (1, "a")): 0.25,
            ((0, "c"), (1, "c"), (1, "a"), (0, "a")): 0.6,
        }
    )


def _run_driver(gf_method, reort, comm=None, monkeypatch_env=None):
    state_bytes = [b"\xa0", b"\x50"]  # {0, 2} and {1, 3}
    basis = Basis(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[2, 3]]}, {0: [[]]}),
        initial_basis=state_bytes,
        comm=comm if comm is not None else MPI.COMM_SELF,
    )
    psis = [ManyBodyState({SlaterDeterminant.from_bytes(b): 1.0}) for b in state_bytes]
    if comm is not None and comm.size > 1:
        # production seeds are hash-distributed; rank 0 provides the full amplitudes.
        # Each seed goes through its own explicit width-1 block rather than a bare
        # ManyBodyState() placeholder on the non-owning ranks (see _capped_solve_with's
        # comment for why a bare placeholder is a rename-time asymmetric-width hazard).
        owns_psis = comm.rank == 0
        psi_blocks = (
            [ManyBodyState.from_states([p]) for p in psis] if owns_psis else [ManyBodyState(width=1) for _ in psis]
        )
        psis = [blk.to_states()[0] for blk in basis.redistribute_psis(*psi_blocks)]
    old_env = {}
    for key, value in (monkeypatch_env or {}).items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        gs_mat, gs_real, report = get_Greens_function(
            matsubara_mesh=1j * np.pi * 0.5 * (2 * np.arange(12) + 1),
            omega_mesh=np.linspace(-2.0, 2.0, 25),
            psis=psis,
            es=[-0.2, 0.3],
            tau=1.0,
            basis=basis,
            hOp=_hyb_hop(),
            delta=0.1,
            blocks=[[0, 1]],
            verbose=False,
            verbose_extra=False,
            reort=reort,
            dN=1,
            occ_cutoff=1e-6,
            slaterWeightMin=0.0,
            sparse=True,
            gf_method=gf_method,
        )
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return gs_mat, gs_real, report
