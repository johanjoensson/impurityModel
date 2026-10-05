r"""Per-axis convergence tolerances of the block-Lanczos Green's function (``gf_tol`` / ``gf_real_tol``).

The monitor used to converge every requested axis to one tolerance, ``max(slaterWeightMin**2, 1e-9)``.
In an RSPt DFT+DMFT run only the Matsubara self-energy drives the self-consistency; the real axis
(broadening ``delta``, where a point can come within ``delta`` of a pole) is output for spectra, yet
it is the axis that sets the Lanczos depth. These tests pin the contract of the split:

* the defaults are bit-identical to the single-tolerance monitor (same stop, same ``G``);
* a looser real-axis tolerance stops earlier while the Matsubara ``G`` stays converged to its own
  tolerance against an exhausted-Krylov reference;
* ``SolverOptions`` validates the options (the TOML and CLI front ends are tested with their peers).
"""

import numpy as np
import pytest
import scipy.linalg as la

from impurityModel.ed.BlockLanczosArray import Reort, block_lanczos_array
from impurityModel.ed.gf_convergence import (
    EvalMeshes,
    _gf_axis_tols,
    _gf_eval_meshes,
    _gf_monitor_tol,
    _gf_rel_tol,
    _make_gf_convergence_monitor,
)
from impurityModel.ed.greens_function import build_qr, calc_G
from impurityModel.ed.model import SolverOptions

_DELTA = 0.2
_TAU = 0.002


@pytest.fixture(autouse=True)
def _hermetic_knobs(monkeypatch):
    for name in ("GF_TOL", "GF_REAL_TOL"):
        monkeypatch.delenv(name, raising=False)


def _matsubara(n, tau=_TAU):
    return 1j * (2 * np.arange(n) + 1) * np.pi * tau


# --- resolution of the tolerances ---------------------------------------------------------------


def test_default_axis_tolerances_are_the_single_historical_tolerance():
    assert _gf_axis_tols(0.0) == (_gf_rel_tol(0.0), _gf_rel_tol(0.0))
    loose_cutoff = 1e-3  # slaterWeightMin**2 = 1e-6 > the 1e-9 floor
    assert _gf_axis_tols(loose_cutoff) == (_gf_rel_tol(loose_cutoff),) * 2


def test_gf_tol_sets_both_axes_and_gf_real_tol_only_the_real_one():
    assert _gf_axis_tols(0.0, gf_tol=1e-7) == (1e-7, 1e-7)
    assert _gf_axis_tols(0.0, gf_real_tol=1e-5) == (_gf_rel_tol(0.0), 1e-5)
    assert _gf_axis_tols(0.0, gf_tol=1e-7, gf_real_tol=1e-4) == (1e-7, 1e-4)


@pytest.mark.parametrize("kwargs", [dict(gf_tol=0.0), dict(gf_real_tol=-1e-6), dict(gf_tol=1.0)])
def test_axis_tolerances_outside_zero_one_are_refused(kwargs):
    with pytest.raises(ValueError, match="must lie in"):
        _gf_axis_tols(0.0, **kwargs)


@pytest.mark.parametrize(
    "has_iw,has_w,expected",
    [(True, True, [1e-8, 1e-4]), (True, False, [1e-8]), (False, True, [1e-4])],
)
def test_eval_meshes_carry_one_tolerance_per_requested_axis(has_iw, has_w, expected):
    iw = _matsubara(16) if has_iw else None
    w = np.linspace(-1.0, 1.0, 16) if has_w else None
    meshes = _gf_eval_meshes(iw, w, side_i=1, delta=_DELTA, es=[0.0, 0.1], axis_tols=(1e-8, 1e-4))
    assert isinstance(meshes, list) and isinstance(meshes, EvalMeshes)
    assert meshes.tols == expected
    assert _gf_monitor_tol(0.0, meshes) == min(expected)


def test_eval_meshes_without_tolerances_fall_back_to_the_single_tolerance():
    meshes = _gf_eval_meshes(_matsubara(8), None, side_i=0, delta=_DELTA, es=[0.0])
    assert meshes.tols is None
    assert _gf_monitor_tol(0.0, meshes) == _gf_rel_tol(0.0)
    assert _gf_monitor_tol(0.0, None) == _gf_rel_tol(0.0)


# --- the monitor -------------------------------------------------------------------------------------


def _model(n=300, seed=4):
    """A spectrum on [1, 50]: G(i w_n) is smooth, G(w + i delta) has to resolve every pole."""
    rng = np.random.default_rng(seed)
    d = np.linspace(1.0, 50.0, n)
    u = la.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    h = (u * d) @ np.conj(u.T)
    h = 0.5 * (h + np.conj(h.T))
    q0, r = build_qr(rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2)))
    return h, q0, r


def _run(h, q0, monitor):
    a, b, _q, _widths = block_lanczos_array(
        psi0=q0, h_op=h, converged=monitor, reort=Reort.FULL, verbose=False, return_widths=True
    )
    return a, b


def test_equal_axis_tolerances_reproduce_the_single_tolerance_monitor_bitwise():
    h, q0, _r = _model()
    iw, w = _matsubara(32), np.linspace(0.0, 52.0, 200)
    plain = _gf_eval_meshes(iw, w, side_i=0, delta=_DELTA, es=[0.0])
    tagged = _gf_eval_meshes(iw, w, side_i=0, delta=_DELTA, es=[0.0], axis_tols=_gf_axis_tols(0.0))

    mon_plain, _f, tol_plain, dg_plain = _make_gf_convergence_monitor(_DELTA, 0.0, eval_meshes=plain)
    mon_tagged, _f, tol_tagged, dg_tagged = _make_gf_convergence_monitor(_DELTA, 0.0, eval_meshes=tagged)
    a_p, b_p = _run(h, q0, mon_plain)
    a_t, b_t = _run(h, q0, mon_tagged)

    assert tol_plain == tol_tagged
    assert len(a_p) == len(a_t)
    assert dg_plain[0] == dg_tagged[0], "the last measured change must be bit-identical"
    for x, y in zip(a_p + b_p, a_t + b_t):
        assert np.array_equal(x, y)


def test_a_loose_real_axis_tolerance_stops_earlier_and_keeps_matsubara_converged():
    # Large enough that the strict run converges before exhausting the Krylov space (at n=300
    # every tolerance runs to exhaustion and nothing is compared): measured 243 / 123 blocks.
    h, q0, r = _model(n=800)
    delta = 1.0
    iw, w = _matsubara(32), np.linspace(0.0, 52.0, 200)
    strict_tols = _gf_axis_tols(0.0)
    loose_tols = _gf_axis_tols(0.0, gf_real_tol=1e-4)

    def monitor(tols):
        meshes = _gf_eval_meshes(iw, w, side_i=0, delta=delta, es=[0.0], axis_tols=tols)
        return _make_gf_convergence_monitor(delta, 0.0, eval_meshes=meshes)

    mon_strict, _f, _t, _dg = monitor(strict_tols)
    mon_loose, flag_loose, tol_loose, _dg = monitor(loose_tols)
    a_s, _b_s = _run(h, q0, mon_strict)
    a_l, b_l = _run(h, q0, mon_loose)
    a_x, b_x = _run(h, q0, lambda *args, **kwargs: False)  # exhaust the Krylov space

    assert flag_loose[0], "the loose run must stop on convergence, not on exhaustion"
    assert tol_loose == strict_tols[0], "the reference tolerance is the strictest axis"
    assert len(a_l) < len(a_s), f"loose real axis used {len(a_l)} blocks against {len(a_s)}"

    g_l, g_x = calc_G(a_l, b_l, r, iw, 0.0, 0.0), calc_G(a_x, b_x, r, iw, 0.0, 0.0)
    err_iw = np.max(np.abs(g_l - g_x)) / np.max(np.abs(g_x))
    assert err_iw < 100 * strict_tols[0], f"Matsubara G lost its own tolerance: {err_iw:.3e}"

    gr_l, gr_x = calc_G(a_l, b_l, r, w, 0.0, delta), calc_G(a_x, b_x, r, w, 0.0, delta)
    err_w = np.max(np.abs(gr_l - gr_x)) / np.max(np.abs(gr_x))
    assert err_w < 1e-2, f"real-axis G far outside its loosened tolerance: {err_w:.3e}"


# --- the front ends -----------------------------------------------------------------------------------


def test_solver_options_accept_and_validate_the_tolerances():
    options = SolverOptions(gf_tol=1e-8, gf_real_tol=1e-5)
    assert (options.gf_tol, options.gf_real_tol) == (1e-8, 1e-5)
    assert SolverOptions().gf_tol is None and SolverOptions().gf_real_tol is None
    with pytest.raises(ValueError, match="must lie in"):
        SolverOptions(gf_real_tol=0.0)
    with pytest.raises(ValueError, match="block-Lanczos convergence tolerance"):
        SolverOptions(gf_method="bicgstab", gf_tol=1e-6)


# --- the driver threads the tolerances to every unit's monitor ----------------------------------------


def _monitor_tolerances(monkeypatch, env=None, **extra):
    """The ``EvalMeshes.tols`` every unit's monitor received during one ``get_Greens_function`` run."""
    from impurityModel.ed import gf_solvers
    from impurityModel.test.support.gf_oracles import _run_driver

    seen = []
    real = gf_solvers._make_gf_convergence_monitor

    def recording(delta, slaterWeightMin, eval_meshes=None):
        seen.append(getattr(eval_meshes, "tols", None))
        return real(delta, slaterWeightMin, eval_meshes)

    monkeypatch.setattr(gf_solvers, "_make_gf_convergence_monitor", recording)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    _run_driver("lanczos", None, extra=extra)
    assert seen, "no unit built a convergence monitor"
    return seen


def test_the_driver_hands_every_unit_the_requested_axis_tolerances(monkeypatch):
    assert all(t == [1e-7, 1e-4] for t in _monitor_tolerances(monkeypatch, gf_tol=1e-7, gf_real_tol=1e-4))


def test_the_knobs_apply_when_the_options_are_unset_and_an_explicit_option_wins(monkeypatch):
    floor = _gf_rel_tol(0.0)
    assert all(t == [floor, floor] for t in _monitor_tolerances(monkeypatch))
    assert all(t == [floor, 1e-5] for t in _monitor_tolerances(monkeypatch, env={"GF_REAL_TOL": "1e-5"}))
    explicit = _monitor_tolerances(monkeypatch, env={"GF_REAL_TOL": "1e-5"}, gf_real_tol=1e-3)
    assert all(t == [floor, 1e-3] for t in explicit)


def test_the_driver_refuses_a_tolerance_on_the_per_frequency_kernel():
    from impurityModel.test.support.gf_oracles import _run_driver

    with pytest.raises(ValueError, match="block-Lanczos tolerances"):
        _run_driver("bicgstab", None, extra={"gf_real_tol": 1e-4})
