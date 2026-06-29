r"""Unit tests for the Green's-function convergence/consistency diagnostics.

These pin the behaviour of each check in :mod:`impurityModel.ed.gf_diagnostics` on small,
exactly-controlled inputs: the anticommutator sum rule, zeroth-moment (mesh-coverage) weight
conservation, the truncated-thermal-ensemble detector, the Lanczos-convergence surface, and
the causality check, plus the report aggregation and the deferred peak-check placeholder.
"""

import numpy as np
import pytest

from impurityModel.ed import gf_diagnostics as gd


def _one_pole_gf(mesh, poles, resid, delta):
    """Diagonal one-orbital GF ``G(w)=sum_k resid_k/(w + i*delta - pole_k)`` as ``(nw,1,1)``."""
    g = (resid[None, :] / (mesh[:, None] + 1j * delta - poles[None, :])).sum(1)
    return g[:, None, None]


# --------------------------------------------------------------------------- #
# sum rule                                                                     #
# --------------------------------------------------------------------------- #


def test_sum_rule_pass_and_fail():
    # occupations n=[0.3,0.8] -> addition seed weights 1-n, removal seed weights n; sum = 2.
    r_ips = np.diag(np.sqrt([0.7, 0.2])).astype(complex)
    r_ps = np.diag(np.sqrt([0.3, 0.8])).astype(complex)
    ok = gd.check_spectral_sum_rule([r_ips, r_ips], [r_ps, r_ps], [0.0, 0.4], 0.0, 0.5, block_dim=2)
    assert ok.severity == gd.Severity.OK and ok.value < 1e-12

    r_bad = (r_ips * 1.3).copy()  # un-normalized seed -> sum != block_dim
    bad = gd.check_spectral_sum_rule([r_bad], [r_ps], [0.0], 0.0, 0.5, block_dim=2)
    assert bad.severity == gd.Severity.FAIL and bad.value > 1e-2


# --------------------------------------------------------------------------- #
# integrated weight (mesh coverage), both sign conventions                     #
# --------------------------------------------------------------------------- #


def test_integrated_weight_mesh_coverage():
    delta = 0.05
    poles = np.array([-1.0, 0.5, 2.0])
    resid = np.array([0.5, 0.3, 0.2])  # zeroth moment = 1
    r1 = np.array([[1.0]], dtype=complex)

    wide = np.linspace(-6, 6, 4001)
    g_wide = _one_pole_gf(wide, poles, resid, delta)
    assert gd.check_integrated_weight(g_wide, [r1], [0.0], 0.0, 1.0, wide, "add").severity == gd.Severity.OK

    narrow = np.linspace(-6, 1.0, 2001)  # misses the pole at +2.0
    g_narrow = _one_pole_gf(narrow, poles, resid, delta)
    miss = gd.check_integrated_weight(g_narrow, [r1], [0.0], 0.0, 1.0, narrow, "add")
    assert miss.severity == gd.Severity.WARN and miss.value > 0.1


def test_mesh_density_thresholds():
    delta = 0.05
    # h/delta ~ 0.08 -> resolves the broadening.
    fine = np.linspace(-8, 8, 4001)
    d_fine = gd.check_mesh_density(fine, delta)
    assert d_fine.severity == gd.Severity.OK and d_fine.value < 0.2

    # h/delta ~ 3 -> too coarse for the 2% trapezoid floor.
    coarse = np.linspace(-8, 8, 107)
    d_coarse = gd.check_mesh_density(coarse, delta)
    assert d_coarse.severity == gd.Severity.WARN and d_coarse.value > 1.36
    # the threshold matches 2*pi/ln(2/rtol).
    assert abs(d_coarse.threshold - 2 * np.pi / np.log(2 / 2e-2)) < 1e-9

    # no broadening (Matsubara) -> not applicable.
    assert gd.check_mesh_density(np.array([1.0, 3.0, 5.0]), 0.0).severity == gd.Severity.OK


def test_integrated_weight_density_vs_coverage_are_distinguished():
    delta = 0.05
    poles = np.array([-1.0, 0.5, 2.0])
    resid = np.array([0.5, 0.3, 0.2])
    r1 = np.array([[1.0]], dtype=complex)

    def gf(mesh):
        return (resid[None, :] / (mesh[:, None] + 1j * delta - poles[None, :])).sum(1)[:, None, None]

    # Coarse but full-extent: the weight check must NOT cry "lost off-mesh" (quadrature floor
    # relaxes the threshold); the mesh_density check carries the warning instead.
    coarse = np.linspace(-8, 8, 107)  # h/delta ~ 3
    w_coarse = gd.check_integrated_weight(gf(coarse), [r1], [0.0], 0.0, 1.0, coarse, "add", delta=delta)
    assert w_coarse.severity == gd.Severity.OK
    assert gd.check_mesh_density(coarse, delta).severity == gd.Severity.WARN

    # Fine density but narrow extent (misses the +2 pole): real coverage loss is still flagged
    # by the weight check, while the density check is happy.
    narrow = np.linspace(-8, 1.0, 2400)  # h/delta ~ 0.08
    w_narrow = gd.check_integrated_weight(gf(narrow), [r1], [0.0], 0.0, 1.0, narrow, "add", delta=delta)
    assert w_narrow.severity == gd.Severity.WARN and w_narrow.value > 0.1
    assert gd.check_mesh_density(narrow, delta).severity == gd.Severity.OK


def test_integrated_weight_removal_sign_robust():
    # The removal GF is evaluated with -delta (Im flips sign); the check must still pass.
    delta = 0.05
    poles = np.array([-1.0, 0.5])
    resid = np.array([0.6, 0.4])
    mesh = np.linspace(-6, 6, 4001)
    g_ps = _one_pole_gf(mesh, poles, resid, -delta)  # note -delta
    r1 = np.array([[1.0]], dtype=complex)
    assert gd.check_integrated_weight(g_ps, [r1], [0.0], 0.0, 1.0, mesh, "rem").severity == gd.Severity.OK


# --------------------------------------------------------------------------- #
# thermal-ensemble truncation                                                  #
# --------------------------------------------------------------------------- #


def test_thermal_cutoff_truncated_vs_complete():
    tau = 0.5
    # Truncated: solver returned exactly as many as wanted, highest still weighty.
    trunc = gd.check_thermal_weight_cutoff([0.0, 0.05, 0.1], 0.0, tau, n_returned=10, num_wanted=10)
    assert trunc.severity == gd.Severity.WARN and trunc.needs_more_states

    # Complete (energy cut bound): solver returned fewer than wanted.
    complete = gd.check_thermal_weight_cutoff([0.0, 1.0, 5.0], 0.0, tau, n_returned=3, num_wanted=10)
    assert complete.severity == gd.Severity.OK and not complete.needs_more_states


def test_thermal_cutoff_unknown_num_wanted_is_safe():
    # No num_wanted (e.g. a T=0 single ground state) must never be flagged as truncated.
    d = gd.check_thermal_weight_cutoff([0.0], 0.0, 0.5, n_returned=1, num_wanted=None)
    assert d.severity == gd.Severity.OK and not d.needs_more_states


# --------------------------------------------------------------------------- #
# lanczos convergence + causality                                              #
# --------------------------------------------------------------------------- #


def test_lanczos_convergence_flag():
    ok = gd.check_lanczos_convergence(True, 1e-9, 12, 50)
    assert ok.severity == gd.Severity.OK and not ok.needs_more_iterations
    bad = gd.check_lanczos_convergence(False, 4.8e-2, 50, 50)
    assert bad.severity == gd.Severity.WARN and bad.needs_more_iterations


def test_causality_sign_convention():
    # retarded convention: Im G_ii <= 0 is causal.
    good = np.array([[[1.0 - 0.5j]]])
    bad = np.array([[[1.0 + 0.5j]]])
    assert gd.check_causality(good).severity == gd.Severity.OK
    assert gd.check_causality(bad).severity == gd.Severity.FAIL


# --------------------------------------------------------------------------- #
# report aggregation                                                           #
# --------------------------------------------------------------------------- #


def test_report_aggregation_and_render():
    rep = gd.DiagnosticReport()
    sr = gd.check_spectral_sum_rule([np.eye(1, dtype=complex)], [np.zeros((1, 1), complex)], [0.0], 0.0, 1.0, 1)
    rep.add("[0,1]", sr)
    rep.add("[0,1]", gd.check_thermal_weight_cutoff([0.0, 0.05], 0.0, 0.5, n_returned=10, num_wanted=10))
    rep.add("[2]", gd.check_lanczos_convergence(False, 0.1, 50, 50))

    assert rep.needs_more_states is True
    assert rep.needs_more_iterations is True
    assert rep.worst_severity == gd.Severity.WARN
    text = rep.render()
    assert "diagnostics" in text and "thermal_cut" in text and "[2]" in text
    # only_problems hides OK rows
    assert "sum_rule" not in rep.render(only_problems=True)


def test_empty_report_renders_pass():
    assert "passed" in gd.DiagnosticReport().render()


# --------------------------------------------------------------------------- #
# deferred peak check                                                          #
# --------------------------------------------------------------------------- #


def test_addition_removal_peak_check_is_deferred():
    with pytest.raises(NotImplementedError):
        gd.addition_removal_peak_check()
