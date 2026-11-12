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
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant
from impurityModel.test.support.gf_switches import set_gf_switches

DELTA = 0.1
OMEGA = np.linspace(-8.0, 8.0, 41)


@pytest.fixture(autouse=True)
def _shipped_defaults(monkeypatch):
    """The knobs these tests set per case are read lazily from the environment: start each test
    from the shipped defaults, whatever the shell exports."""
    for knob in ("GF_FROZEN_CSR", "APPLY_ROW_CHUNKS"):
        monkeypatch.delenv(knob, raising=False)


#: The matvec modes of the recurrence (``gf_recurrence``), forced through ``set_gf_switches(matrix_mode=...)``.
_MODES = ["discovered", "retained", "none"]


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
    return [blk.to_states()[0] for blk in basis.redistribute_psis(*blocks)]


def _run_capped(cap, reort=None, comm=None, basis_attrs=None, **kwargs):
    basis = _excited_basis(cap, comm=comm)
    for name, value in (basis_attrs or {}).items():
        setattr(basis, name, value)
    # redistribute_psis SUMS per-rank contributions (production seeds are partial per
    # rank), so only rank 0 may provide the full amplitudes here.
    seeds_full = _seeds()
    seeds = _redistribute_as_width1(basis, seeds_full if comm is None or comm.rank == 0 else None, len(seeds_full))
    info = {}
    kwargs.setdefault("verbose", False)
    alphas, betas, r = block_Green_sparse(
        _siam_6(),
        seeds,
        basis,
        DELTA,
        reort=reort,
        cap_info=info,
        **kwargs,
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


def test_cap_above_reachable_space_is_identity(monkeypatch):
    """A cap the recurrence never reaches must not change the result at all -- bit for bit,
    which is a property of the one-shot matvec specifically (APPLY_ROW_CHUNKS=1): the
    uncapped run never redistributes at all in serial (no proxy, no caps_growth), while the
    capped-but-never-binding run does one redistribute through _CappedBasisProxy, and
    APPLY_ROW_CHUNKS's default since 2026-09-14 is 4 chunks, whose accumulate (`+=`)
    changes floating-point summation order relative to that single one-shot call (see
    test_gf_apply_row_chunking.py's test_chunked_matches_one_shot_above_the_reachable_space,
    which pins that difference to ~1e-15 explicitly instead of asserting it away)."""
    monkeypatch.setenv("APPLY_ROW_CHUNKS", "1")
    g_uncapped, info_u = _run_capped(np.inf)
    g_capped, info_c = _run_capped(1000)
    assert not info_u["cap_hit"] and not info_c["cap_hit"]
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


@pytest.mark.parametrize("chunks", ["1", "4"])
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("reort", [None, True])
@pytest.mark.parametrize("cap", [6, 12, 17])
def test_capped_gf_equals_dense_php_resolvent(cap, reort, mode, chunks, monkeypatch):
    """The oracle: the capped GF is the exact GF of H projected on the retained set -- in every matvec mode (the
    matrix over every discovered key, the operator applied each step with the ``P H P`` matrix built at the freeze,
    the operator applied every block)."""
    set_gf_switches(monkeypatch, matrix_mode=mode)
    monkeypatch.setenv("APPLY_ROW_CHUNKS", chunks)
    g, info = _run_capped(cap, reort=reort)
    assert info["cap_hit"]
    assert info["engine"] == mode
    assert info["retained_size"] <= cap
    retained = info["proxy"].retained_keys()
    assert len(retained) == info["retained_size"]
    np.testing.assert_allclose(g, _dense_reference_on(retained), atol=1e-9)
    # causality: Im G_ii <= 0 on the retarded mesh
    assert np.all(np.diagonal(g.imag, axis1=1, axis2=2) <= 1e-12)


@pytest.mark.parametrize("mode", _MODES)
def test_cap_at_seed_size_freezes_immediately(mode, monkeypatch):
    """cap == initial basis size: nothing new is ever admitted; still exact on P."""
    set_gf_switches(monkeypatch, matrix_mode=mode)
    basis = _excited_basis(np.inf)
    seed_size = basis.size
    g, info = _run_capped(seed_size)
    assert info["cap_hit"] and info["retained_size"] <= seed_size
    assert info["engine"] == mode
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

        def redistribute_psis(self, *blocks):
            return list(blocks)

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


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_a_tied_slot_at_the_cap_goes_to_the_smaller_key(order):
    """One slot left, two candidates with the same amplitude: the freeze admits the smaller key,
    whatever the row order or rounding -- not neither (under-filling the cap), not whichever rounding
    made larger."""

    class _FakeBasis:
        comm = None
        is_distributed = False

        def __init__(self, local):
            self.local_basis = local
            self.size = len(local)
            self.n_bytes = 1

        def redistribute_block(self, block):
            return block

    seed = _det([0, 1, 2])
    tied = [_det([0, 1, 3]), _det([0, 1, 4])]
    eps = np.finfo(float).eps
    amps = [0.5 * (1 + eps), 0.5 * (1 - eps)]
    proxy = _CappedBasisProxy(_FakeBasis([seed]), cap=2)
    rows = {tied[i]: complex(amps[j]) for i, j in zip((0, 1), order)}
    rows[_det([0, 2, 3])] = 0.1 + 0j
    proxy.redistribute_block(ManyBodyState.from_states([ManyBodyState(rows)]))
    smaller = min(tied, key=lambda k: bytes(k.to_bytearray()))
    assert proxy.cap_hit and proxy.retained_size == 2
    assert set(proxy.retained_keys()) == {seed, smaller}


@pytest.mark.parametrize(
    "chunks",
    [
        # A row's amplitude summed over the chunks that reach it: A gets 0.3 + 0.3.
        [{"A": 0.3, "B": 0.5}, {"A": 0.3, "C": 0.4}],
        [{"A": 0.6}, {"B": 0.5, "C": 0.4}],
        [{"A": 0.6, "B": 0.5, "C": 0.4}],
    ],
)
def test_the_cap_crossing_step_is_decided_on_full_amplitudes_whatever_the_chunking(chunks):
    """Admission at the cap ranks a row on its amplitude summed over the step's chunks.

    Decided chunk by chunk (``APPLY_ROW_CHUNKS`` > 1), the first chunking above froze on B (0.5 beats A's
    partial 0.3), the others on A -- and since how rows split into chunks follows how they are spread over
    ranks, a capped unit froze on a different set in every colour layout. Between ``begin_step`` and
    ``finish_step`` the proxy sees the summed output, so all three keep A.
    """

    class _FakeBasis:
        comm = None
        is_distributed = False

        def __init__(self, local):
            self.local_basis = local
            self.size = len(local)
            self.n_bytes = 1

        def redistribute_block(self, block):
            return block

    names = {"A": _det([0, 1, 3]), "B": _det([0, 1, 4]), "C": _det([0, 1, 5])}
    seed = _det([0, 1, 2])
    proxy = _CappedBasisProxy(_FakeBasis([seed]), cap=2)
    proxy.begin_step()
    wp = None
    for chunk in chunks:
        piece = proxy.redistribute_block(
            ManyBodyState.from_states([ManyBodyState({names[k]: complex(v) for k, v in chunk.items()})])
        )
        wp = piece if wp is None else wp + piece
    wp = proxy.finish_step(wp)
    assert proxy.cap_hit and proxy.retained_size == 2
    assert set(proxy.retained_keys()) == {seed, names["A"]}
    assert len(wp) == 1  # the step's output was projected onto the retained set (A; the seed row is absent)


@pytest.mark.mpi
@pytest.mark.parametrize("mode", _MODES)
def test_capped_gf_mpi_matches_dense_php(mode, monkeypatch):
    """Distributed run: cap respected, collective decisions consistent, oracle holds.

    With 2+ ranks and a small cap, some rank will own few or zero retained rows —
    the empty-rank edge case must not deadlock or diverge."""
    set_gf_switches(monkeypatch, matrix_mode=mode)
    comm = MPI.COMM_WORLD
    for cap in (6, 12):
        g, info = _run_capped(cap, comm=comm)
        assert info["cap_hit"]
        assert info["engine"] == mode
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
    from impurityModel.ed.gf_solvers import block_Green

    cap = 5
    basis = _excited_basis(cap)
    hOp = _siam_6()
    # One full H fanout from the initial basis: the largest support any single
    # probe batch can add before the in-loop cap check fires.
    probe = ManyBodyState.from_states([ManyBodyState(dict.fromkeys(basis.local_basis, 1.0 + 0j))])
    one_fanout = set(basis.local_basis) | set(hOp.apply_block(probe, 0).support_keys(0.0))
    assert cap < len(one_fanout) < 18  # the cap must bind inside the first probe round

    alphas, betas, r = block_Green(hOp, _seeds(), basis, DELTA, False, verbose=False)
    assert basis.size > cap  # the cap was crossed (expansion did happen)
    assert basis.size <= len(one_fanout)  # ...but by at most one apply batch
    g = calc_G(alphas, betas, r, OMEGA, 0.0, DELTA)
    assert np.all(np.diagonal(g.imag, axis1=1, axis2=2) <= 1e-12)


def _kernel_on_proxy(cap, stop_on_freeze, comm=None, max_iter=50):
    """Run the bare sparse kernel on a capped proxy; returns (alphas, betas, status, proxy, resume state)."""
    from impurityModel.ed.basis_transcription import build_state
    from impurityModel.ed.BlockLanczos import block_lanczos
    from impurityModel.ed.gf_primitives import _distributed_seed_qr

    basis = _excited_basis(np.inf, comm=comm)
    seeds_full = _seeds()
    seeds = _redistribute_as_width1(basis, seeds_full if comm is None or comm.rank == 0 else None, len(seeds_full))
    psi_local, _ = _distributed_seed_qr(basis, seeds, 0)
    psi = build_state(basis, psi_local.T, slaterWeightMin=0)
    proxy = _CappedBasisProxy(basis, cap)
    proxy.stop_on_freeze = stop_on_freeze
    alphas, betas, Q, widths, status = block_lanczos(
        psi,
        _siam_6(),
        proxy,
        lambda *a, **k: False,
        verbose=False,
        max_iter=max_iter,
        return_widths=True,
        return_status=True,
        store_krylov=False,
        reort=False,
    )
    return alphas, betas, status, proxy, (psi, Q, widths)


def test_kernel_stops_at_the_freeze_and_resumes_bit_identically(monkeypatch):
    """``stop_on_freeze`` ends the recurrence on the step that froze the support, with a state the
    usual resume protocol continues from: stop + resume == one uninterrupted run, bit for bit."""
    from impurityModel.ed.BlockLanczos import block_lanczos

    monkeypatch.setenv("APPLY_ROW_CHUNKS", "1")
    n_total = 8
    a_full, b_full, status_full, proxy_full, _ = _kernel_on_proxy(12, stop_on_freeze=False, max_iter=n_total)
    assert proxy_full.cap_hit and status_full == "max_iter" and len(a_full) == n_total

    a_stop, b_stop, status, proxy, (psi, Q, widths) = _kernel_on_proxy(12, stop_on_freeze=True, max_iter=n_total)
    assert status == "frozen" and proxy.frozen
    assert 0 < len(a_stop) < n_total
    proxy.stop_on_freeze = False
    a_res, b_res, *_rest, status_res = block_lanczos(
        psi,
        _siam_6(),
        proxy,
        lambda *a, **k: False,
        verbose=False,
        max_iter=n_total - len(a_stop),
        return_widths=True,
        return_status=True,
        alphas_init=a_stop,
        betas_init=b_stop,
        Q_init=Q,
        block_widths_init=widths,
        store_krylov=False,
        reort=False,
    )
    assert status_res == "max_iter"
    np.testing.assert_array_equal(np.asarray(a_res), np.asarray(a_full))
    np.testing.assert_array_equal(np.asarray(b_res), np.asarray(b_full))


def test_kernel_without_stop_on_freeze_runs_through_the_freeze():
    _, _, status, proxy, _ = _kernel_on_proxy(12, stop_on_freeze=False)
    assert proxy.frozen and status != "frozen"


@pytest.mark.mpi
def test_kernel_stops_at_the_freeze_on_the_same_step_on_every_rank():
    comm = MPI.COMM_WORLD
    alphas, _, status, _proxy, _ = _kernel_on_proxy(6, stop_on_freeze=True, comm=comm)
    assert status == "frozen"
    assert len(set(comm.allgather(len(alphas)))) == 1


def _assert_php_oracle(g, info, comm=None):
    local = info["proxy"].retained_keys()
    retained = sorted({k for part in (comm.allgather(local) if comm is not None else [local]) for k in part})
    np.testing.assert_allclose(g, _dense_reference_on(retained), atol=1e-9)


def test_the_frozen_matrix_matches_the_operator_applied_every_block_to_rounding(monkeypatch):
    """Same model, same cap, same retained set: the frozen ``P H P`` matrix and the matrix-free unit agree."""
    monkeypatch.setenv("APPLY_ROW_CHUNKS", "1")
    set_gf_switches(monkeypatch, matrix_mode="none")
    g_sparse, info_sparse = _run_capped(12, reort=True)
    set_gf_switches(monkeypatch, matrix_mode="retained")
    g_csr, info_csr = _run_capped(12, reort=True)
    assert info_csr["engine"] == "retained" and info_sparse["engine"] == "none"
    assert set(info_csr["proxy"].retained_keys()) == set(info_sparse["proxy"].retained_keys())
    np.testing.assert_allclose(g_csr, g_sparse, atol=1e-10)


def test_frozen_csr_under_importance_pruning(monkeypatch):
    """A unit the cap froze under ``prune_tol`` continues on the frozen ``P H P`` matrix, exact on its
    retained set."""
    monkeypatch.setenv("APPLY_ROW_CHUNKS", "1")
    g, info = _run_capped(12, prune_tol=0.05)
    assert info["cap_hit"] and info["csr_fallback"]
    _assert_php_oracle(g, info)


def test_a_memory_guard_freeze_without_room_for_the_matrix_applies_the_operator_every_block():
    """RSS is at budget when the guard freezes; with no looser frozen-CSR budget no matrix fits on top of it, and
    the unit is priced matrix-free."""
    g, info = _run_capped(1000, memory_budget=1)
    assert info["cap_hit"] and info["memory_frozen"]
    assert info["engine"] == "none"
    _assert_php_oracle(g, info)


def test_a_memory_guard_freeze_runs_the_frozen_matrix_when_it_fits():
    """The guard froze the unit, but the frozen-CSR budget has room for the matrix on top of the
    current RSS: the restart runs, and is exact on the retained set."""
    g, info = _run_capped(1000, memory_budget=1, basis_attrs={"gf_csr_memory_budget": 2**62})
    assert info["cap_hit"] and info["memory_frozen"]
    assert info["engine"] == "retained"
    _assert_php_oracle(g, info)


@pytest.mark.parametrize("cap", [6, 12])
def test_a_matrix_that_does_not_fit_leaves_the_operator_applied_every_block(cap):
    """When no matrix fits the budget the unit runs matrix-free and the result is still exact on P."""
    g, info = _run_capped(cap, reort=True, memory_budget=1)
    assert info["cap_hit"] and info["engine"] == "none"
    _assert_php_oracle(g, info)


def test_a_retained_unit_whose_matrix_no_longer_fits_at_the_freeze_continues_matrix_free(monkeypatch):
    """The mode is priced before the first step; the freeze re-measures the worst rank's actual ``P`` and the
    resident set then (a memory-guard freeze arrives with it at the guard's budget), and a matrix that does not fit
    is not built: the unit keeps its answer and reports the mode it ran."""
    set_gf_switches(monkeypatch, matrix_mode="retained")
    _g, info = _run_capped(12, memory_budget=10**12)
    assert info["engine"] == "retained" and info["csr_fallback"]
    g_tight, info_tight = _run_capped(1000, memory_budget=1)
    assert info_tight["memory_frozen"] and info_tight["engine"] == "none" and not info_tight["csr_fallback"]
    _assert_php_oracle(g_tight, info_tight)


def test_csr_fallback_means_the_frozen_recurrence_ran_on_a_stored_matrix(monkeypatch):
    for mode, cap, want in (("retained", 12, True), ("discovered", 12, True), ("none", 12, False)):
        set_gf_switches(monkeypatch, matrix_mode=mode)
        assert _run_capped(cap)[1]["csr_fallback"] is want, (mode, cap)
    set_gf_switches(monkeypatch, matrix_mode="retained")
    assert not _run_capped(1000)[1]["csr_fallback"], "a unit that never froze ran on no stored matrix"


def test_the_csr_fit_check_compares_the_estimate_with_the_budget(monkeypatch):
    from impurityModel.ed import gf_solvers

    basis = _excited_basis(np.inf)
    assert gf_solvers._frozen_csr_fits(basis, _siam_6(), None, None)
    monkeypatch.setattr(gf_solvers, "current_rss_bytes", lambda: 10**9)
    assert gf_solvers._frozen_csr_fits(basis, _siam_6(), 2 * 10**9, None)
    assert not gf_solvers._frozen_csr_fits(basis, _siam_6(), 10**9 + 1, None)


def test_the_csr_fit_check_reports_what_it_decided_on(monkeypatch):
    from impurityModel.ed import gf_solvers

    basis = _excited_basis(np.inf)
    monkeypatch.setattr(gf_solvers, "current_rss_bytes", lambda: 10**9)
    report = {}
    assert gf_solvers._frozen_csr_fits(basis, _siam_6(), 10**10, None, report=report)
    assert report["n_local"] == len(basis.local_basis)
    assert report["rss"] == 10**9 and report["budget"] == 10**10
    assert report["fanout"] > 0.0
    expected = gf_solvers._frozen_matrix_need(report["n_local"], report["fanout"], basis.size, 1, 1)
    assert report["need"] == pytest.approx(expected)
    # The guard off decides nothing, so there is nothing to report.
    empty = {}
    assert gf_solvers._frozen_csr_fits(basis, _siam_6(), None, None, report=empty)
    assert empty == {}


def test_the_frozen_matrix_need_counts_the_exchange_buffers_on_more_than_one_rank():
    """The graph exchange holds row-sized send and receive buffers, at least one column wide: up to
    every other rank's rows out and this rank's rows in from each other rank. With almost no stored
    elements they are what the need is made of; a serial run has none."""
    from impurityModel.ed import gf_solvers

    n_global, n_local, ranks = 4_000_000, 1_000_000, 4
    serial = gf_solvers._frozen_matrix_need(n_local, 1e-6, n_local, 1, 1)
    distributed = gf_solvers._frozen_matrix_need(n_local, 1e-6, n_global, ranks, 1)
    exchange = 16 * ((n_global - n_local) + n_local * (ranks - 1) + n_local)
    assert distributed - serial == pytest.approx(exchange, rel=1e-6)


@pytest.mark.mpi
def test_the_csr_fit_check_runs_on_every_rank_including_empty_ones(monkeypatch):
    """A rank that owns no determinant still enters both reductions, and all ranks agree."""
    from impurityModel.ed import gf_solvers

    comm = MPI.COMM_WORLD
    basis = _excited_basis(np.inf, comm=comm)
    monkeypatch.setattr(gf_solvers, "current_rss_bytes", lambda: 10**9)
    for budget in (10**10, 10**9 + 1):
        report = {}
        fits = gf_solvers._frozen_csr_fits(basis, _siam_6(), budget, comm, report=report)
        assert len(set(comm.allgather(fits))) == 1, "the decision must be the same on every rank"
        reports = comm.allgather(report)
        assert all(r == reports[0] for r in reports), "the report is the reduced worst case, identical everywhere"
        assert reports[0]["n_local"] == max(comm.allgather(len(basis.local_basis)))
    assert fits is False and reports[0]["need"] > 1


@pytest.mark.mpi
def test_a_decision_line_is_printed_on_accept_and_decline(monkeypatch, capsys):
    from impurityModel.ed import memory_estimate

    comm = MPI.COMM_WORLD
    monkeypatch.setattr(memory_estimate, "current_rss_bytes", lambda: 10**9)
    for budget, mode in ((10**10, "retained"), (10**9 + 1, "none")):
        capsys.readouterr()
        _g, info = _run_capped(12, comm=comm, memory_budget=budget, verbose=True)
        out = capsys.readouterr().out
        assert info["engine"] == mode
        if comm.rank == 0:
            assert f"GF matvec mode {mode}" in out and "fan-out" in out and "need" in out


@pytest.mark.mpi
@pytest.mark.parametrize("chunks", ["1", "4"])
def test_frozen_csr_mpi_matches_the_matrix_free_unit(chunks, monkeypatch):
    """Multi-rank: the frozen distributed matrix (empty ranks at -n 3) equals the operator applied every block."""
    comm = MPI.COMM_WORLD
    monkeypatch.setenv("APPLY_ROW_CHUNKS", chunks)
    set_gf_switches(monkeypatch, matrix_mode="none")
    g_sparse, _ = _run_capped(6, reort=True, comm=comm)
    set_gf_switches(monkeypatch, matrix_mode="retained")
    g_csr, info = _run_capped(6, reort=True, comm=comm)
    assert info["engine"] == "retained"
    np.testing.assert_allclose(g_csr, g_sparse, atol=1e-9)
    _assert_php_oracle(g_csr, info, comm)


def test_frozen_csr_starts_from_a_bounded_block_budget(monkeypatch):
    """The array kernel preallocates its coefficient buffers at its block budget, and ``ceil(N/p)``
    at a 10M-determinant frozen basis is GiBs. The handoff starts small and doubles: a 1-block
    floor on a seed-frozen unit still ends exact on P."""
    from impurityModel.ed import BlockLanczosCore, gf_solvers

    budgets = []
    real_kernel = BlockLanczosCore.block_lanczos

    def recording_kernel(*args, **kwargs):
        budgets.append(kwargs["max_iter"])
        return real_kernel(*args, **kwargs)

    monkeypatch.setattr(gf_solvers, "_CSR_INITIAL_BLOCKS", 1)
    monkeypatch.setattr(BlockLanczosCore, "block_lanczos", recording_kernel)
    set_gf_switches(monkeypatch, matrix_mode="retained")
    seed_size = _excited_basis(np.inf).size
    g, info = _run_capped(seed_size, reort=None)
    assert info["engine"] == "retained"
    assert budgets[0] == 1 and len(budgets) > 1
    assert budgets == sorted(budgets)
    _assert_php_oracle(g, info)
