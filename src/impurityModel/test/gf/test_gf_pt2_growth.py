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


def test_the_unit_reports_where_its_rows_came_from(capsys):
    _g, info = _run(1e-12)
    rec = info["pt2"]
    assert rec["mesh_points"] == gf_growth_score.PT2_MESH_POINTS and rec["mesh_points_kept"] <= rec["mesh_points"]
    assert rec["rule_rows"] > 0 and rec["stop"] == "twice its last admission"
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
    assert made and all(sc.acc is None and sc._stash is None and not sc._alphas for sc in made)
