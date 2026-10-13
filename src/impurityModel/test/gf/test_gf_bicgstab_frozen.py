r"""The BiCGSTAB solves on a frozen ``P H P`` (``gf_solvers._FrozenShiftedSystem``, ``GF_CLOSURE_FREEZE``).

Both BiCGSTAB drivers -- ``gf_method="bicgstab"`` (:func:`block_Green_bicgstab`) and the RIXS R1 fallback
(``rixs._R1SolverChain``) -- solve one shift at a time on a sparse basis rebuilt per point, with a
right-hand side that is the same for every shift of a unit. Once a point's solve admits nothing beyond its
starting support, the remaining points solve on the frozen matrix, each checked against the full ``H``.
The contract these tests pin:

* the frozen solves give the sparse ``G`` (to the solver tolerance), most points take them, and they
  converge in BiCGSTAB itself, on every rank count (a rank-local reduction once sent every multi-rank
  frozen point to GMRES, which hid it by getting ``G`` right anyway);
* the frozen matvec is ``H`` restricted to ``P``; the leak check is ``(1 - P)(z - H) X`` above the cutoff
  relative to ``||Y||``, against a dense oracle on a ``P`` that is not closed;
* a solution that leaks goes back to the sparse solve, warm-started, and stays there;
* no freeze after a cap hit, nor on a unit's last point; a matrix that does not fit is not built;
  ``GF_CLOSURE_FREEZE=0`` keeps every point sparse;
* several eigenstates with complex seeds, and the residual check, on frozen points;
* RIXS: the R1 chain and the full map take the frozen path and give the sparse result; its fallback;
* every rank takes the same decisions (``-n 2``/``-n 3``).
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import gf_solvers
from impurityModel.ed.basis_transcription import build_dense_matrix, build_local_column_operator
from impurityModel.ed.gf_primitives import FrozenColumnMatvec, block_to_local_dense
from impurityModel.ed.gf_solvers import block_Green_bicgstab
from impurityModel.ed.greens_function import _gf_signed_axes
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.test.gf.test_gf_truncation import _redistribute_as_width1
from impurityModel.test.support import gf_oracles
from impurityModel.test.support.gf_oracles import DELTA, OMEGA, _n3_sector_dets, _seed_basis, _seeds, _siam_6

ATOL = 1e-10


@pytest.fixture(autouse=True)
def _hermetic_knobs(monkeypatch):
    for name in ("GF_CLOSURE_FREEZE", "GF_APPLY_ROW_CHUNKS", "GF_BICGSTAB_RESIDUAL_CHECK"):
        monkeypatch.delenv(name, raising=False)


def _complex_seeds():
    """The default seeds with complex, column-specific phases: a missing conjugate shows."""
    a, b = _seeds()
    return [a * np.exp(0.7j), (b + a * 0.3j) * np.exp(-0.4j)]


def _gf(comm=None, seeds=None, es=(0.0,), cap=10**9, omega=OMEGA, **kw):
    basis = _seed_basis(cap=cap, comm=comm)
    full = []
    for _ in es:
        full += list(seeds if seeds is not None else _seeds())
    seeds_r = _redistribute_as_width1(basis, full if comm is None or comm.rank == 0 else None, len(full))
    z_axes = _gf_signed_axes(None, omega, 0, DELTA)
    return block_Green_bicgstab(_siam_6(), seeds_r, basis, list(es), 2, z_axes, atol=ATOL, **kw)


def _assert_same_g(G, G_ref, rel=1e-8):
    scale = max(np.abs(g).max() for g in G_ref)
    for g, g_ref in zip(G, G_ref):
        np.testing.assert_allclose(g, g_ref, atol=rel * scale)


def _sparse(monkeypatch, fn, *args, **kwargs):
    monkeypatch.setenv("GF_CLOSURE_FREEZE", "0")
    try:
        return fn(*args, **kwargs)
    finally:
        monkeypatch.delenv("GF_CLOSURE_FREEZE")


def test_the_frozen_solves_give_the_sparse_g_and_converge_in_bicgstab(monkeypatch):
    G, stats = _gf()
    assert stats["frozen_points"] > len(OMEGA) // 2, stats["frozen_points"]
    assert stats["frozen_fallbacks"] == 0
    assert stats["gmres_points"] == 0, "a frozen solve escalated to GMRES"
    assert sum(pt["frozen"] for pt in stats["points"]) == stats["frozen_points"]
    G_sparse, sparse_stats = _sparse(monkeypatch, _gf)
    assert sparse_stats["frozen_points"] == 0
    _assert_same_g(G, G_sparse)


def test_several_eigenstates_with_complex_seeds(monkeypatch):
    """The frozen system is per eigenstate (same seeds, its own shift), and G is <seed|X>, conjugated."""
    G, stats = _gf(seeds=_complex_seeds(), es=(0.0, 0.4))
    assert stats["frozen_points"] > len(OMEGA), "both eigenstates froze"
    G_sparse, _ = _sparse(monkeypatch, _gf, seeds=_complex_seeds(), es=(0.0, 0.4))
    _assert_same_g(G, G_sparse)


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
    matvec = FrozenColumnMatvec(build_local_column_operator(basis, op), basis.comm, 2)
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


def test_a_leaking_solution_goes_back_to_the_sparse_solve_and_stays(monkeypatch):
    G_sparse, _ = _sparse(monkeypatch, _gf)
    monkeypatch.setattr(gf_solvers._FrozenShiftedSystem, "boundary", lambda self, *a: np.full(self.n_cols, 1.0))
    warm = []
    real = gf_solvers.solve_shifted_block

    def recording(A_op, x0, *a, **k):
        if isinstance(x0, ManyBodyState):  # the sparse solves (the frozen one passes dense rows)
            warm.append(len(x0) > 0)
        return real(A_op, x0, *a, **k)

    monkeypatch.setattr(gf_solvers, "solve_shifted_block", recording)
    G, stats = _gf()
    assert stats["frozen_fallbacks"] == 1 and stats["frozen_points"] == 0, "one fallback, then no refreeze"
    assert all(warm[1:]), "the sparse solves after the fallback are warm-started"
    _assert_same_g(G, G_sparse)


def test_no_freeze_after_a_cap_hit(monkeypatch):
    built = []
    real = gf_solvers._freeze_bicgstab_support
    monkeypatch.setattr(gf_solvers, "_freeze_bicgstab_support", lambda *a, **k: built.append(1) or real(*a, **k))
    _, stats = _gf(cap=8)
    assert stats["cap_hit"]
    assert not built and stats["frozen_points"] == 0 and stats["frozen_fallbacks"] == 0


def test_no_freeze_on_a_units_last_point(monkeypatch):
    built = []
    real = gf_solvers._freeze_bicgstab_support
    monkeypatch.setattr(gf_solvers, "_freeze_bicgstab_support", lambda *a, **k: built.append(1) or real(*a, **k))
    _gf(omega=OMEGA[:1])
    assert not built, "a single point has no later point to use the frozen matrix"


def test_a_matrix_that_does_not_fit_is_not_built(monkeypatch):
    monkeypatch.setattr(gf_solvers, "_frozen_csr_fits", lambda *a, **k: False)
    built = []
    monkeypatch.setattr(gf_solvers, "_FrozenShiftedSystem", lambda *a, **k: built.append(1))
    _, stats = _gf()
    assert not built and stats["frozen_points"] == 0


def test_the_fit_check_budgets_the_solvers_dense_blocks(monkeypatch):
    seen = []
    monkeypatch.setattr(gf_solvers, "_frozen_csr_fits", lambda *a, **k: seen.append(k["work_blocks"]) or True)
    monkeypatch.setenv("GF_GMRES_RESTART", "40")
    _gf()
    assert seen and all(w == 50 for w in seen), seen


def test_the_residual_check_runs_on_frozen_points(monkeypatch):
    monkeypatch.setenv("GF_BICGSTAB_RESIDUAL_CHECK", "1")
    _, stats = _gf()
    frozen = [pt for pt in stats["points"] if pt["frozen"]]
    assert frozen and all("dG_bound" in pt for pt in frozen)
    assert all(np.all(pt["boundary"] <= 1e-9) for pt in frozen), "a frozen point passed its leak check"


def test_off_switch(monkeypatch):
    monkeypatch.setenv("GF_CLOSURE_FREEZE", "0")
    _, stats = _gf()
    assert stats["frozen_points"] == 0 and stats["frozen_size"] is None


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


# --------------------------------------------------------------------------- MPI


@pytest.mark.mpi
def test_every_rank_takes_the_same_decisions_and_bicgstab_converges(monkeypatch):
    comm = MPI.COMM_WORLD
    G, stats = _gf(comm=comm)
    assert stats["frozen_points"] > 0
    assert stats["gmres_points"] == 0, "a frozen solve escalated to GMRES (a rank-local reduction?)"
    assert len(set(comm.allgather((stats["frozen_points"], stats["frozen_size"], stats["iterations"])))) == 1
    serial_iterations = _gf()[1]["iterations"]
    assert stats["iterations"] <= 2 * serial_iterations, (stats["iterations"], serial_iterations)
    G_sparse, _ = _sparse(monkeypatch, _gf, comm=comm)
    _assert_same_g(G, G_sparse, rel=1e-9)


@pytest.mark.mpi
def test_the_rixs_r1_chain_on_every_rank():
    comm = MPI.COMM_WORLD
    frozen, counters = _rixs_chain_run(True, comm=comm)
    assert counters["r1_frozen"] > 0 and counters["r1_gmres"] == 0
    sparse, _ = _rixs_chain_run(False, comm=comm)
    np.testing.assert_allclose(frozen, sparse, atol=1e-5 * np.abs(sparse).max())
