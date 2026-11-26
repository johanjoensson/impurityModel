"""The second-order growth rule (``pt2_tol``) on the six-orbital SIAM, against the dense resolvent.

Pinned: the pruned recurrence is the exact resolvent of ``H`` on what the rule kept (the restart makes it so); the
error in ``G`` is inside the budget when the frontier is scored every block; a vanishing budget reaches the closure;
the cap is filled by score; the retained set does not depend on the rank count; and a unit with no real-frequency
window falls back to the amplitude rule instead of failing.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import gf_growth_score
from impurityModel.ed.gf_convergence import _gf_eval_meshes
from impurityModel.ed.gf_growth_score import GrowthScorer, pt2_mesh
from impurityModel.ed.gf_solvers import block_Green_sparse
from impurityModel.ed.greens_function import calc_G
from impurityModel.test.gf.test_gf_truncation import _redistribute_as_width1
from impurityModel.test.support.gf_oracles import DELTA, MATSUBARA, OMEGA, _dense_G_on, _seed_basis, _seeds, _siam_6

CLOSURE = 18


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    monkeypatch.setenv("APPLY_ROW_CHUNKS", "1")
    monkeypatch.setattr(gf_growth_score, "PT2_SCORE_EVERY", 1)


def _run(pt2_tol, cap=np.inf, comm=None, meshes="real", **kwargs):
    basis = _seed_basis(cap=cap, comm=comm)
    full = _seeds()
    seeds = _redistribute_as_width1(basis, full if comm is None or comm.rank == 0 else None, len(full))
    info = {}
    em = _gf_eval_meshes(None, OMEGA, 0, DELTA, [0.0]) if meshes == "real" else meshes
    a, b, r = block_Green_sparse(
        _siam_6(),
        seeds,
        basis,
        DELTA,
        verbose=False,
        cap_info=info,
        prune_tol=kwargs.pop("prune_tol", 0.0),
        pt2_tol=pt2_tol,
        eval_meshes=em,
        **kwargs,
    )
    return calc_G(a, b, r, OMEGA, 0.0, DELTA), info


def _keys(info, comm=None):
    keys = info["proxy"].retained_keys()
    if comm is not None:
        keys = [k for part in comm.allgather(keys) for k in part]
    return sorted(set(keys))


@pytest.mark.parametrize("tau", [0.3, 0.1, 0.01, 1e-3])
def test_the_rule_is_the_exact_resolvent_of_what_it_kept_and_the_error_is_inside_the_budget(tau):
    g_ref, _ = _run(1e-12)
    g, info = _run(tau)
    z = OMEGA + 1j * DELTA
    np.testing.assert_allclose(g, _dense_G_on(_keys(info), z), atol=1e-9)
    assert abs(g - g_ref).max() / abs(g_ref).max() <= 2 * tau


def test_a_vanishing_budget_reaches_the_closure_and_a_large_one_prunes():
    _g, tight = _run(1e-12)
    _g, loose = _run(0.3)
    assert len(_keys(tight)) == CLOSURE
    assert len(_keys(loose)) < CLOSURE


def test_the_retained_set_grows_as_the_budget_shrinks():
    sizes = [len(_keys(_run(t)[1])) for t in (0.3, 0.01, 1e-12)]
    assert sizes == sorted(sizes)


def test_a_cap_binds_under_the_rule():
    _g, info = _run(1e-12, cap=14)
    assert len(_keys(info)) <= 14


def test_a_matsubara_only_unit_falls_back_to_the_amplitude_rule():
    mats = [np.asarray(MATSUBARA, dtype=complex)]
    g, info = _run(1e-3, meshes=mats)
    g_amp, info_amp = _run(None, meshes=mats, prune_tol=1e-3)
    assert _keys(info) == _keys(info_amp)
    np.testing.assert_allclose(g, g_amp, atol=1e-12)


def test_the_two_rules_are_exclusive():
    with pytest.raises(ValueError, match="replaces the amplitude rule"):
        _run(1e-3, prune_tol=1e-3)


def test_the_mesh_spans_the_union_of_the_states_windows_and_needs_a_real_axis():
    z1 = np.linspace(-2, 2, 9) + 0.1j
    z2 = np.linspace(-1, 5, 9) + 0.1j
    zs, why = pt2_mesh([np.concatenate([z1, z2])], 7)
    assert why is None and len(zs) == 7
    assert zs.real.min() == -2 and zs.real.max() == 5 and np.allclose(zs.imag, 0.1)
    assert pt2_mesh([np.asarray(MATSUBARA)])[0] is None
    assert pt2_mesh(None)[0] is None
    assert pt2_mesh([np.array([0.1j, 0.2j + 1.0])])[0] is None  # too few points


def test_the_mesh_drop_removes_points_that_never_carry_mass():
    sc = GrowthScorer(np.linspace(-1, 1, 5) + 0.1j, 1e-3, None, every=1, drop=1e-2, drop_blocks=2)
    from impurityModel.ed.gf_shift_recycling import ShiftedGalerkin

    sc.acc = ShiftedGalerkin(sc.zs0, 4)
    sc.note_mass(np.array([1.0, 0.0, 1.0, 0.0, 1.0]))
    assert not sc.dropped
    sc.note_mass(np.array([1.0, 0.0, 1.0, 0.0, 1.0]))
    assert sc.dropped and list(sc.kept) == [0, 2, 4] and sc.acc.n_z == 3


@pytest.mark.mpi
@pytest.mark.parametrize("tau", [0.1, 1e-3])
def test_the_retained_set_does_not_depend_on_the_rank_count(tau):
    comm = MPI.COMM_WORLD
    _g, serial = _run(tau)
    g, info = _run(tau, comm=comm)
    assert _keys(info, comm) == _keys(serial)
    np.testing.assert_allclose(g, _run(tau)[0], atol=1e-9)


@pytest.mark.parametrize("tau", [1e-3, 1e-12])
def test_the_quiet_stop_ends_growth_early_and_leaves_the_basis_of_the_unstopped_run(tau, monkeypatch):
    g_stop, stopped = _run(tau)
    monkeypatch.setattr(gf_growth_score, "PT2_QUIET_BLOCKS", 10**9)
    g_full, full = _run(tau)
    # Where the growth reaches every determinant H can (tau 1e-12), the support closes and both runs end there, the
    # closure stop coming no later than the quiet one. Without the quiet stop a growth that admits nothing more still
    # ends, ``stalled`` after |P| vectors (tau 1e-3), which may be the same block.
    if full["closed"] or full["pt2"]["stop"] == "stalled":
        assert stopped["passes"]["growth_blocks"] <= full["passes"]["growth_blocks"]
    else:
        assert stopped["passes"]["growth_blocks"] < full["passes"]["growth_blocks"]
    assert _keys(stopped) == _keys(full)
    np.testing.assert_allclose(g_stop, g_full, atol=1e-10)


def test_the_stop_clock_is_twice_the_last_admission_and_at_least_the_quiet_floor():
    sc = GrowthScorer(np.linspace(-1, 1, 3) + 0.1j, 1e-3, None)
    sc.blocks = 15
    assert not sc.quiet()
    sc.blocks = 16
    assert sc.quiet()  # nothing ever admitted: the floor alone
    sc.last_admission = 20
    sc.blocks = 39
    assert not sc.quiet()
    sc.blocks = 40
    assert sc.quiet()
    sc.last_admission = 4
    sc.blocks = 19
    assert not sc.quiet()
    sc.blocks = 20
    assert sc.quiet()  # the floor (16 quiet blocks) outweighs "twice the last admission" (8)


def test_a_frontier_whose_score_mass_is_still_rising_keeps_the_growth_alive():
    """The rule may admit nothing for many blocks while ``rho`` builds up (SMO unit 7: more than 16): a record of the
    frontier's mass is activity, exactly as an admission is."""
    from impurityModel.ed.gf_shift_recycling import ShiftedGalerkin

    sc = GrowthScorer(np.linspace(-1, 1, 3) + 0.1j, 1e-3, None, drop=0.0)
    sc.acc = ShiftedGalerkin(sc.zs0, 4)
    for block, top in ((4, 1e-6), (8, 1e-5), (12, 1e-4), (16, 5e-5), (20, 2e-4)):
        sc.blocks = block
        sc.note_mass(np.array([0.0, top, top / 2]))
        assert not sc.quiet()
    assert sc.last_rise == 20 and sc.peak_mass == 2e-4
    sc.blocks = 39
    assert not sc.quiet()
    sc.blocks = 40
    assert sc.quiet()


def test_the_unit_reports_where_its_rows_came_from(capsys):
    _g, info = _run(1e-12)
    rec = info["pt2"]
    assert rec["mesh_points"] == gf_growth_score.PT2_MESH_POINTS and rec["mesh_points_kept"] <= rec["mesh_points"]
    # The rule admits every determinant H reaches (CLOSURE, below): the support closes, which ends the growth.
    assert rec["rule_rows"] > 0 and rec["stop"] == "closed"
    n_seed = len({k for seed in _seeds() for k in seed.keys()})
    assert rec["first_shell_rows"] + rec["rule_rows"] + n_seed == CLOSURE
    assert "pt2" not in _run(None, prune_tol=1e-3)[1]


def test_a_unit_that_never_grew_is_not_cut_by_the_quiet_stop(monkeypatch):
    """Started on the closure there is no frontier: nothing is admitted and nothing is restarted, so the recurrence
    must run to its own convergence -- the quiet stop would leave a truncated continued fraction."""
    from impurityModel.ed.manybody_basis import Basis
    from impurityModel.test.support import gf_oracles as O

    closure = _keys(_run(1e-12)[1])

    def run(pt2):
        basis = Basis(O._IMP, O._BATHS, initial_basis=closure, truncation_threshold=np.inf, comm=None, verbose=False)
        em = _gf_eval_meshes(None, OMEGA, 0, DELTA, [0.0])
        a, b, r = block_Green_sparse(
            _siam_6(), _seeds(), basis, DELTA, verbose=False, cap_info={}, prune_tol=0.0 if pt2 else 1e-3,
            pt2_tol=pt2, eval_meshes=em,
        )  # fmt: skip
        return calc_G(a, b, r, OMEGA, 0.0, DELTA)

    monkeypatch.setattr(gf_growth_score, "PT2_QUIET_BLOCKS", 2)
    np.testing.assert_allclose(run(1e-3), run(None), atol=1e-9)


def test_the_capped_unit_books_the_rows_the_cap_let_in_not_the_candidates():
    _g, info = _run(1e-12, cap=12)
    rec = info["pt2"]
    n_seed = len({k for seed in _seeds() for k in seed.keys()})
    assert len(_keys(info)) == 12 and rec["first_shell_rows"] == 5 and rec["rule_rows"] == 12 - n_seed - 5


def test_the_default_cadence_scores_late_and_may_leave_rows_a_dense_closure_has(monkeypatch):
    """Documented limit, pinned: scoring every 4th block admits the first rows after the Krylov space is largely
    used, so on this 18-determinant model the deferred rows carry no coherent weight and the unit ends 14 rows /
    1.6e-3 in G (every block: the closure). tau bounds the error only at the finest cadence."""
    _g1, tight = _run(1e-12)
    monkeypatch.setattr(gf_growth_score, "PT2_SCORE_EVERY", 4)
    g4, info = _run(1e-12)
    assert len(_keys(info)) < len(_keys(tight))
    assert abs(g4 - _g1).max() / abs(_g1).max() < 5e-3


def test_a_finished_growth_releases_the_accumulator(monkeypatch):
    made = []
    real = GrowthScorer.__init__

    def spy(self, *a, **k):
        real(self, *a, **k)
        made.append(self)

    monkeypatch.setattr(GrowthScorer, "__init__", spy)
    _run(1e-3)
    assert made and all(sc.acc is None and sc._stash is None and not sc._queue and not sc._alphas for sc in made)


def _stub_support(budget, n_k, n_z=31):
    import types

    return types.SimpleNamespace(
        decisions=types.SimpleNamespace(memory_budget=budget),
        scorer=types.SimpleNamespace(zs=np.zeros(n_z), kept=np.arange(n_z)),
    ), types.SimpleNamespace(n_k_local=n_k, comm=None)


def test_the_accumulator_is_priced_on_the_measured_k_and_gives_way_mesh_first(monkeypatch):
    """Whole mesh inside MATRIX_MEMORY_SAFETY of the guard's budget, else the halved mesh (16 of 31 points) inside that
    room, else the halved mesh inside the whole budget, else no rule."""
    from impurityModel.ed import gf_recurrence

    monkeypatch.setattr(gf_recurrence, "resident_set_bytes", lambda: 1000)
    support, matrix = _stub_support(None, 10**6)
    assert gf_recurrence._pt2_accumulator_fits(support, matrix, 2)[0] == "full", "no budget: the guard is off"
    per_point = gf_recurrence.REFINE_SKEW * 2 * 10**6 * np.dtype(gf_growth_score.PT2_ACC_DTYPE).itemsize
    full, thin = int(per_point * 31), int(per_point * 16)
    safety = gf_recurrence.MATRIX_MEMORY_SAFETY

    def verdict(budget, n_k=10**6):
        support, matrix = _stub_support(budget, n_k)
        return gf_recurrence._pt2_accumulator_fits(support, matrix, 2)

    v, got, rss, _b = verdict((1000 + full) / safety + 10)
    assert (v, got, rss) == ("full", full, 1000)
    assert verdict((1000 + full) / safety - 10)[0] == "thin"
    assert verdict((1000 + thin) / safety + 10)[0] == "thin"
    assert verdict((1000 + thin) / safety - 10)[0] == "thin", "the halved mesh may use the whole budget"
    assert verdict(1000 + thin + 10)[0] == "thin"
    assert verdict(1000 + thin - 10)[0] == "decline"
    assert verdict(1000 + thin + 10, 2 * 10**6)[0] == "decline", "the same budget, twice the measured K"


def test_thinning_keeps_every_other_point_and_the_ends():
    sc = gf_growth_score.GrowthScorer(np.linspace(-1, 1, 31) + 0.1j, 1e-3, None)
    sc.thin()
    assert sc.thinned and sc.kept.size == 16 and sc.kept[0] == 0 and sc.kept[-1] == 30
    np.testing.assert_array_equal(sc.zs, np.linspace(-1, 1, 31)[::2] + 0.1j)
    assert sc._live.size == 16


def test_a_halved_mesh_keeps_the_rule_and_says_so(monkeypatch):
    from impurityModel.ed import gf_recurrence

    monkeypatch.setattr(gf_recurrence, "_pt2_accumulator_fits", lambda support, matrix, width: ("thin", 0, 0, 0))
    _g, info = _run(1e-3)
    assert "declined" not in info["pt2"] and info["pt2"]["mesh_thinned"]
    assert info["pt2"]["mesh_points_kept"] <= 16
    assert info["passes"]["growth_rule"] == "second-order rule (gf_pt2_tol 0.001, mesh halved to fit the memory)"


@pytest.mark.mpi
def test_an_accumulator_that_does_not_fit_hands_the_unit_to_the_amplitude_rule_in_the_same_mode(monkeypatch):
    """The verdict is taken at the first fold from the measured K; a refusal drops the scorer and the unit grows by
    the amplitude rule at its default: the rows and G of a unit that asked for it, with the reason on the record."""
    from impurityModel.ed import gf_recurrence
    from impurityModel.ed.options import DEFAULT_GF_PRUNE_TOL

    calls = []

    def refuse(support, matrix, width):
        calls.append(width)
        return "decline", 123, 456, 789

    monkeypatch.setattr(gf_recurrence, "_pt2_accumulator_fits", refuse)
    comm = MPI.COMM_WORLD
    g, info = _run(1e-3, comm=comm if comm.size > 1 else None)
    g_amp, info_amp = _run(None, comm=comm if comm.size > 1 else None, prune_tol=DEFAULT_GF_PRUNE_TOL)
    assert len(calls) == 1, "decided once, at the first fold"
    assert "123.0 B" in info["pt2"]["declined"] and "456.0 B" in info["pt2"]["declined"]
    assert "declined" in info["pt2"] and "pt2" not in info_amp
    # The unit summary (the one line every unit gets on rank 0's output) says which rule it grew by, and why.
    assert info["passes"]["growth_rule"].startswith(f"amplitude rule {DEFAULT_GF_PRUNE_TOL:g}, not gf_pt2_tol 0.001: ")
    assert "123.0 B" in info["passes"]["growth_rule"] and "growth_rule" not in info_amp["passes"]
    np.testing.assert_allclose(g, g_amp, atol=1e-12)
    c = comm if comm.size > 1 else None
    assert _keys(info, c) == _keys(info_amp, c)


def test_an_accumulator_that_fits_keeps_the_rule_and_is_priced_on_both_block_widths(monkeypatch):
    from impurityModel.ed import gf_recurrence

    widths = []

    def fits(support, matrix, width):
        widths.append(width)
        return "full", 0, 0, 0

    monkeypatch.setattr(gf_recurrence, "_pt2_accumulator_fits", fits)
    _g, info = _run(1e-12)
    assert "declined" not in info["pt2"] and info["pt2"]["rule_rows"] > 0
    assert info["passes"]["growth_rule"] == "second-order rule (gf_pt2_tol 1e-12)"
    assert widths == [2 * len(_seeds())], "ShiftedGalerkin holds (rows, n_z, p) and (rows, n_z, w): the sum of the two"


def test_the_diagonal_of_K_is_computed_once_per_determinant_and_equals_the_uncached_one():
    """The rule reads ``H_ff`` of the frontier at every scored decision: only the slots appended since the last call
    are computed, and the cached array is the one a fresh computation over all of ``K`` gives."""
    made = []
    from impurityModel.ed import DistributedSparseMatrix as dsm

    real = dsm.DistributedSparseMatrix.k_diagonal

    def spy(self):
        before = len(self._k_diag)
        out = real(self)
        made.append((before, len(out)))
        fresh = np.zeros(self.n_k_local, dtype=complex)
        fresh[self._slot_of_pos] = self.hop.diagonal(self._kstore)
        np.testing.assert_array_equal(out, fresh)
        assert not out.flags.writeable
        return out

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(dsm.DistributedSparseMatrix, "k_diagonal", spy)
        _run(1e-12)
    assert len(made) > 2 and made[0][0] == 0
    assert any(b == n for b, n in made[1:]), "a call with no new slot computes nothing"
    assert any(0 < b < n for b, n in made[1:]), "a grown K computes only its new slots"


def test_scores_are_current_on_the_cadence_whatever_the_fold_batch(monkeypatch):
    """The accumulator is stepped with the queued blocks in one sweep before a scored decision reads it: the retained
    set and G do not depend on how many blocks a sweep folds."""
    monkeypatch.setattr(gf_growth_score, "PT2_SCORE_EVERY", 4)
    g_batched, batched = _run(1e-6)
    monkeypatch.setattr(gf_growth_score, "PT2_FOLD_BATCH", 1)
    g_single, single = _run(1e-6)
    assert _keys(batched) == _keys(single)
    np.testing.assert_allclose(g_batched, g_single, atol=1e-12)


def _restart_with_and_without_the_drop(monkeypatch, tau, comm):
    """``(drops, G with the drop, G without it, info)`` of one unit: the drop is spied on, then disabled."""
    from impurityModel.ed.DistributedSparseMatrix import DistributedSparseMatrix

    real = DistributedSparseMatrix.drop_frontier
    drops = []

    def spy(self):
        n_k = self.n_k_local
        out = real(self)
        drops.append((out, n_k, self.n_k_local, self.n_p_local))
        return out

    monkeypatch.setattr(DistributedSparseMatrix, "drop_frontier", spy)
    g, info = _run(tau, comm=comm)
    monkeypatch.setattr(DistributedSparseMatrix, "drop_frontier", lambda self: False)
    g_kept, info_kept = _run(tau, comm=comm)
    return drops, g, g_kept, info, info_kept


@pytest.mark.parametrize("tau", [0.1, 1e-3])
def test_the_restart_drops_the_rows_outside_p_and_its_green_function_is_bitwise_unchanged(tau, monkeypatch):
    """No refinement follows a second-order growth, so the restart runs on ``P H P`` alone (``drop_frontier``): the
    continued fraction is bitwise the one on the whole discovered matrix, and the unit still reports the |K|/|P| its
    growth reached."""
    drops, g, g_kept, info, info_kept = _restart_with_and_without_the_drop(monkeypatch, tau, None)
    assert len(drops) == 1 and drops[0][0] is True
    _ok, n_k_before, n_k_after, n_p = drops[0]
    assert n_k_before > n_p == n_k_after, "the fixture must have a frontier to drop"
    np.testing.assert_array_equal(g, g_kept)
    assert info["k_over_p"] == info_kept["k_over_p"] > 1.0
    assert info["passes"]["restart_blocks"] == info_kept["passes"]["restart_blocks"] > 0


@pytest.mark.mpi
@pytest.mark.parametrize("tau", [0.1, 1e-3])
def test_the_dropped_frontier_restart_is_bitwise_unchanged_on_every_rank(tau, monkeypatch):
    comm = MPI.COMM_WORLD
    drops, g, g_kept, info, info_kept = _restart_with_and_without_the_drop(monkeypatch, tau, comm)
    assert len(drops) == 1 and drops[0][0] is True
    np.testing.assert_array_equal(g, g_kept)
    assert info["k_over_p"] == info_kept["k_over_p"]
