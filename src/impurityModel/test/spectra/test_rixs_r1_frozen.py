r"""The frozen ``P H P`` solves of the RIXS R1 chain (``gf_solvers._FrozenShiftedSystem``).

``rixs._R1SolverChain`` solves one incoming-energy shift at a time on a sparse basis rebuilt per point, with a
right-hand side that is the same for every shift. Once a point's solve admits nothing beyond its starting support,
the remaining points solve on the frozen matrix, each checked against the full ``H``. The contract these tests pin:

* the frozen matvec is ``H`` restricted to ``P``; the leak check is ``(1 - P)(z - H) X`` above the cutoff
  relative to ``||Y||``, against a dense oracle on a ``P`` that is not closed;
* the R1 chain and the full map take the frozen path and give the sparse result; a leaking solution falls back
  to the sparse solve and stays there;
* the weighted window applies to the sparse tier and to the leak check alike;
* every rank takes the same decisions (``-n 2``/``-n 3``).

(Until 2026-11 these tests shared a file with the per-frequency Green's-function driver, ``gf_method="bicgstab"``,
which was removed; only the RIXS half survives.)
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import gf_solvers
from impurityModel.ed.basis_transcription import build_dense_matrix, build_sparse_matrix
from impurityModel.ed.gf_primitives import FrozenColumnMatvec, block_to_local_dense
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.test.gf.test_gf_truncation import _redistribute_as_width1
from impurityModel.test.support import gf_oracles
from impurityModel.test.support.gf_oracles import DELTA, _n3_sector_dets, _seed_basis, _seeds, _siam_6
from impurityModel.test.support.gf_switches import set_gf_switches


@pytest.fixture(autouse=True)
def _hermetic_knobs(monkeypatch):
    for name in ("GF_SUPPORT_FREEZE", "APPLY_ROW_CHUNKS"):
        monkeypatch.delenv(name, raising=False)


def _sparse(monkeypatch, fn, *args, **kwargs):
    set_gf_switches(monkeypatch, closure_freeze="0")
    try:
        return fn(*args, **kwargs)
    finally:
        set_gf_switches(monkeypatch, closure_freeze="1")


def _p_short_of_the_sector(comm=None):
    """P = the seeds and one apply of H around them: not closed, so (z - H) X leaks out of it."""
    basis = _seed_basis(comm=comm)
    op = _siam_6()
    image = basis.redistribute_block(op.apply_block(ManyBodyState.from_states(_seeds()), 0.0))
    basis.add_states(sorted(image.keys()))
    assert basis.size < len(_n3_sector_dets()), "premise: P is a proper subset of the sector"
    return basis, op


def test_the_frozen_matvec_is_h_on_p():
    basis, op = _p_short_of_the_sector()
    matvec = FrozenColumnMatvec(build_sparse_matrix(basis, op), basis.comm, 2)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(len(basis.local_basis), 2)) + 1j * rng.normal(size=(len(basis.local_basis), 2))
    X_block = ManyBodyState.from_keys_and_amps(basis._keys, X)
    want = block_to_local_dense(basis, op.apply_block(X_block, 0.0), 2)
    with matvec.session():
        np.testing.assert_allclose(matvec.apply(X), want, atol=1e-12)


def test_the_leak_check_is_the_part_outside_p_above_the_relative_cutoff():
    """Against a dense oracle on the whole sector, with seeds scaled so the cutoff's scale matters."""
    basis, op = _p_short_of_the_sector()
    scale = 30.0
    seeds = [s * scale for s in _seeds()]
    system = gf_solvers._FrozenShiftedSystem(basis, op, seeds, 2)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(len(basis.local_basis), 2)) + 1j * rng.normal(size=(len(basis.local_basis), 2))
    z = 0.7 + 1j * DELTA

    sector = sorted(_n3_sector_dets())
    full = Basis(gf_oracles._IMP, gf_oracles._BATHS, initial_basis=sector, verbose=False)
    H = np.asarray(build_dense_matrix(full, op))
    row = {d: i for i, d in enumerate(sector)}
    in_p = np.zeros(len(sector), dtype=bool)
    X_full = np.zeros((len(sector), 2), dtype=complex)
    for i, d in enumerate(basis.local_basis):
        in_p[row[d]] = True
        X_full[row[d]] = X[i]
    outside = ((z * np.eye(len(sector)) - H) @ X_full)[~in_p]
    y_norm = np.max(np.sqrt(np.sum(np.abs(system.seeds) ** 2, axis=0)))
    amps = np.max(np.abs(outside), axis=1)
    # A cutoff (relative to ||Y||) that keeps some outside rows and drops others.
    swm = float(np.median(amps[amps > 0])) / y_norm
    kept = outside[amps > swm * y_norm]
    assert 0 < len(kept) < np.count_nonzero(amps), "premise: the cut drops some leaked rows, not all"
    want = np.sqrt(np.sum(np.abs(kept) ** 2, axis=0)) / system.seed_norms

    got = system.boundary(z - op, ManyBodyState.from_keys_and_amps(basis._keys, X), swm)
    np.testing.assert_allclose(got, want, rtol=1e-12)
    uncut = np.sqrt(np.sum(np.abs(outside) ** 2, axis=0)) / system.seed_norms
    assert np.all(got < uncut), "the cut is applied"


# --------------------------------------------------------------------------- RIXS R1


def _rixs_chain_run(freeze, comm=None):
    """The R1 chain with no sector cache, so every wIn point runs the bicgstab tier."""
    from impurityModel.ed.rixs import _new_rixs_solver_stats, _R1SolverChain

    basis = _seed_basis(cap=10**9, comm=comm)
    tmp_basis = basis.clone(initial_basis=[], verbose=False)
    full = _seeds()
    psi1_all = list(_redistribute_as_width1(basis, full if comm is None or comm.rank == 0 else None, len(full)))
    seeds = [s.copy() for s in psi1_all]
    psi2_all = [ManyBodyState(width=1) for _ in psi1_all]
    counters = _new_rixs_solver_stats()
    chain = _R1SolverChain(None, eigenstate=0, counters=counters, freeze=freeze)
    wins = np.linspace(-3.0, 3.0, 7)
    grams = []
    for k, win in enumerate(wins):
        psi2 = chain.solve(tmp_basis, _siam_6(), psi1_all, psi2_all, k, win, wins[k:], 0.3, 0.0, 0.0, False)
        gram = np.zeros((len(seeds), len(psi2)), dtype=complex)
        for i, s1 in enumerate(seeds):
            for j, s2 in enumerate(psi2):
                for d, a in s1.items():
                    b = s2.get(d)
                    if b is not None:
                        gram[i, j] += np.conj(a[0]) * b[0]
        if comm is not None:
            comm.Allreduce(MPI.IN_PLACE, gram, op=MPI.SUM)
        grams.append(gram)
    return np.array(grams), counters


def test_the_rixs_r1_chain_takes_the_frozen_path_and_gives_the_same_states():
    frozen, counters = _rixs_chain_run(True)
    assert counters["r1_frozen"] > 0 and counters["r1_frozen_fallbacks"] == 0
    assert counters["r1_bicgstab"] == 7 and counters["r1_gmres"] == 0
    sparse, sparse_counters = _rixs_chain_run(False)
    assert sparse_counters["r1_frozen"] == 0
    np.testing.assert_allclose(frozen, sparse, atol=1e-5 * np.abs(sparse).max())


def test_the_rixs_r1_fallback(monkeypatch):
    monkeypatch.setattr(gf_solvers._FrozenShiftedSystem, "boundary", lambda self, *a: np.full(self.n_cols, 1.0))
    built = []
    real = gf_solvers._freeze_bicgstab_support
    from impurityModel.ed import rixs

    monkeypatch.setattr(rixs, "_freeze_bicgstab_support", lambda *a, **k: built.append(1) or real(*a, **k))
    grams, counters = _rixs_chain_run(True)
    assert len(built) == 1, "no refreeze after the fallback"
    assert counters["r1_frozen_fallbacks"] == 1 and counters["r1_frozen"] == 0
    sparse, _ = _rixs_chain_run(False)
    np.testing.assert_allclose(grams, sparse, atol=1e-5 * np.abs(sparse).max())


def test_the_rixs_map_passes_the_knob_and_gives_the_sparse_map(monkeypatch):
    """Through calc_map with the dense sector cache and the recycler declined, so R1 runs bicgstab."""
    from impurityModel.ed import rixs
    from impurityModel.test.spectra.test_rixs_tensor import _model, _run_rixs, _thermal_states, _tin_tout

    monkeypatch.setenv("GF_SECTOR_DENSE_MAX", "0")
    monkeypatch.setenv("GF_KRYLOV_RECYCLE_MAX_BYTES", "0")
    flags = []
    real_init = rixs._R1SolverChain.__init__

    def spying(self, *a, **k):
        flags.append(k.get("freeze"))
        real_init(self, *a, **k)

    monkeypatch.setattr(rixs._R1SolverChain, "__init__", spying)
    op = _model()
    psis, es, dets, _states, _vecs = _thermal_states(op, 2)
    tin, tout = _tin_tout()
    frozen = _run_rixs(op, psis, es, tin, tout, dets)
    assert flags and all(flags), flags
    flags.clear()
    sparse = _sparse(monkeypatch, _run_rixs, op, psis, es, tin, tout, dets)
    assert flags and not any(flags), flags
    np.testing.assert_allclose(frozen, sparse, atol=1e-6 * np.abs(sparse).max())


class _SkewedComm:
    """A communicator whose float Allreduce returns rank-dependent last bits (which MPI permits); everything else
    goes to the real one."""

    def __init__(self, comm):
        self._comm = comm

    def Allreduce(self, sendbuf, recvbuf, op=MPI.SUM):
        self._comm.Allreduce(sendbuf, recvbuf, op=op)
        out = recvbuf
        if isinstance(out, np.ndarray) and out.dtype.kind == "f":
            out *= 1.0 + 1e-15 * self._comm.rank

    def __getattr__(self, name):
        return getattr(self._comm, name)


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="the replication is between ranks")
def test_the_frozen_systems_leak_is_the_same_bits_on_every_rank():
    """What the accept/fallback verdict compares is the leak over the seed norms, both reduced over the colour:
    neither may carry a rank's own rounding. Simulated by a communicator whose reductions differ per rank."""
    comm = MPI.COMM_WORLD
    basis, op = _p_short_of_the_sector(comm=comm)
    real_comm = basis.comm
    basis.comm = _SkewedComm(real_comm)
    try:
        system = gf_solvers._FrozenShiftedSystem(basis, op, _seeds(), 2)
        rng = np.random.default_rng(5)
        X = rng.normal(size=(len(basis.local_basis), 2)) + 1j * rng.normal(size=(len(basis.local_basis), 2))
        got = system.boundary((0.7 + 1j * DELTA) - op, ManyBodyState.from_keys_and_amps(basis._keys, X), 0.0)
    finally:
        basis.comm = real_comm
    assert np.all(got > 0), "premise: X leaks out of P"
    assert len({g.tobytes() for g in comm.allgather(got)}) == 1


@pytest.mark.mpi
def test_the_rixs_r1_chain_on_every_rank():
    comm = MPI.COMM_WORLD
    frozen, counters = _rixs_chain_run(True, comm=comm)
    assert counters["r1_frozen"] > 0 and counters["r1_gmres"] == 0
    sparse, _ = _rixs_chain_run(False, comm=comm)
    np.testing.assert_allclose(frozen, sparse, atol=1e-5 * np.abs(sparse).max())


@pytest.mark.parametrize("freeze", [False, True])
def test_the_rixs_r1_solve_respects_the_weighted_window(freeze):
    """``z - hOp`` is a fresh operator: without the weighted mask the R1 solve (sparse tier and the frozen
    leak check alike) reaches determinants outside the window. The window forbids both conduction orbitals
    (4 and 5) being occupied, which the hybridization reaches from the seeds."""
    from impurityModel.ed.rixs import _new_rixs_solver_stats, _R1SolverChain

    window = [({4: 1, 5: 1}, (0, 1))]
    basis = _seed_basis(cap=10**9)
    tmp_basis = basis.clone(initial_basis=[], weighted_restrictions=window, verbose=False)
    psi1_all = list(_redistribute_as_width1(basis, _seeds(), len(_seeds())))
    psi2_all = [ManyBodyState(width=1) for _ in psi1_all]
    counters = _new_rixs_solver_stats()
    chain = _R1SolverChain(None, eigenstate=0, counters=counters, freeze=freeze)
    wins = np.linspace(-3.0, 3.0, 5)
    reached = set()
    for k, win in enumerate(wins):
        psi2 = chain.solve(tmp_basis, _siam_6(), psi1_all, psi2_all, k, win, wins[k:], 0.3, 0.0, 0.0, False)
        reached |= {bytes(d.to_bytearray()[:1])[0] for p in psi2 for d in p.keys()}
    both_conduction = {b for b in reached if (b >> 3) & 1 and (b >> 2) & 1}
    assert reached, "premise: the solve reached something"
    assert not both_conduction, f"determinants outside the weighted window: {sorted(both_conduction)}"
    if freeze:
        # the leak check applies the same mask: rows outside the window are not leaks
        assert counters["r1_frozen"] > 0 and counters["r1_frozen_fallbacks"] == 0
