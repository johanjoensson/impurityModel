"""The synthetic metals (``test/support/metal_models.py``) and the replay rig's TOML input and separate caps.

FCC Ni cannot test how the Green's-function basis is truncated while its ground state is itself held by the
determinant cap. These tests pin what the replacement relies on: the model is the one the docstring says
(Gauss-Chebyshev stars, ``sum V^2 = v^2``), it goes through the production input pipeline, the Green's-function
cap can be pinned without touching the ground state, and ``U = 0`` has the exact self-energy zero.
"""

import numpy as np
import pytest

from impurityModel.test.support import freeze_replay as fr
from impurityModel.test.support.metal_models import metal_h0, write_metal


@pytest.fixture(autouse=True)
def _restore_what_the_rig_sets(monkeypatch):
    """``freeze_replay.run`` sets process-wide state (the causality tolerance, the prune rule, the matrix mode):
    undo it after each test, or the self-energy tests that run later see a causality tolerance of 1e9."""
    from impurityModel.ed import gf_solvers, selfenergy

    monkeypatch.setattr(selfenergy, "DEFAULT_CAUSALITY_TOL", selfenergy.DEFAULT_CAUSALITY_TOL)  # the rig sets 1e9
    from impurityModel.ed import gf_recurrence

    monkeypatch.setattr(gf_recurrence, "_PRUNE_RULE", "amplitude")
    monkeypatch.setattr(gf_solvers, "_MATRIX_MODE", gf_solvers._MATRIX_MODE)


def test_the_h0_matrix_is_a_semicircular_star_per_orbital():
    n_orb, n_b, v = 3, 6, 0.8
    h = metal_h0(n_orb, n_b, D=0.5, v=v)
    assert h.shape == (n_orb * (1 + n_b),) * 2
    assert np.array_equal(h, h.T)
    for a in range(n_orb):
        block = slice(n_orb + a * n_b, n_orb + (a + 1) * n_b)
        assert np.sum(h[a, block] ** 2) == pytest.approx(v**2)  # Gauss-Chebyshev weights sum to one
        levels = np.diag(h)[block]
        assert np.allclose(np.sort(levels), -np.sort(levels)[::-1])  # particle-hole symmetric
        assert not np.any(levels == 0.0)
        # An orbital sees only its own star.
        others = [b for b in range(n_orb, h.shape[0]) if not block.start <= b < block.stop]
        assert not np.any(h[a, others])
    assert not np.any(np.diag(h)[:n_orb])  # the double counting sets the impurity level


@pytest.mark.parametrize("n_b", [0, 1, 5])
def test_an_odd_or_empty_star_is_refused(tmp_path, n_b):
    # An odd star has a level at zero, which the input pipeline files under valence or conduction by its sign.
    with pytest.raises(ValueError, match="even n_b"):
        write_metal(tmp_path, "bad", n_b=n_b)


@pytest.mark.parametrize("n_orb, n_b", [(1, 4), (3, 2)])
def test_the_written_model_goes_through_the_production_input_pipeline(tmp_path, n_orb, n_b):
    from impurityModel.inputformat.build import build
    from impurityModel.inputformat.reader import load_input

    toml = write_metal(tmp_path, "m", n_orb=n_orb, n_b=n_b, U=0.4, J=0.05 if n_orb > 1 else 0.0, n_matsubara=8)
    built = build(load_input(str(toml)))
    assert built.basis.nominal_occ == {0: n_orb}  # half filling
    assert built.basis.truncation_threshold is None  # auto, unless a replay pins it
    assert len(built.meshes.w) == 301 and len(built.meshes.iw) == 8


def test_no_caps_leave_the_truncation_threshold_alone():
    assert fr.cap_policy("archive") == "archive"
    assert fr.cap_policy(1234) == 1234


def test_a_gf_cap_alone_leaves_the_ground_state_unlimited():
    policy = fr.cap_policy("archive", gf_cap=300)
    assert policy.gf == 300 and policy.gs == float("inf")
    # CapPolicy has no "final but unlimited" state: the driver's log line cannot describe one.
    assert policy.from_memory


def test_a_gs_cap_alone_leaves_the_gf_units_unlimited():
    policy = fr.cap_policy(77, gs_cap=300)  # the separate caps override `cap`
    assert policy.gs == 300 and policy.gf == float("inf")


def test_two_finite_caps_are_final():
    policy = fr.cap_policy("archive", gs_cap=300, gf_cap=900)
    assert (policy.gs, policy.gf, policy.from_memory) == (300, 900, False)


def test_pruning_replaces_the_settle_in_the_toml_overrides():
    assert fr._solver_overrides(False, 3.0, 1e-3, None, None, None, None) == {"gf_prune_tol": 1e-3}
    assert fr._solver_overrides(False, None, None, None, None, None, None) == {"gf_settle": False}
    assert fr._solver_overrides(None, None, None, 0.1, 3, 0.2, 0.05) == {
        "gf_first_shell_tol": 0.1,
        "gf_refine_rounds": 3,
        "gf_refine_tol": 0.2,
        "gf_refine_swap": 0.05,
    }


def test_the_replay_overrides_reach_the_options_built_from_a_toml_input(tmp_path):
    toml = write_metal(tmp_path, "m", n_b=4, U=0.5, n_matsubara=4)
    policy = fr.cap_policy("archive", gf_cap=500)
    _model, basis, solver, _meshes, label = fr._toml_inputs(
        toml,
        None,
        0,
        swm=1e-6,
        cap=policy,
        cost_budget=None,
        solver_overrides=fr._solver_overrides(None, None, 1e-3, None, 2, None, None),
    )
    assert basis.truncation_threshold is policy
    assert basis.slater_weight_min == 1e-6
    assert solver.gf_prune_tol == 1e-3 and solver.gf_refine_rounds == 2
    assert label == "m"


def test_grade_zero_counts_the_acausal_points():
    sigma = np.zeros((6, 2, 2), dtype=complex)
    sigma[:, 0, 0] = -1e-9j
    sigma[4, 1, 1] = 3e-3j
    g = fr.grade_zero({"sigma": np.full((3, 2, 2), 2e-6, dtype=complex), "sigma_real": sigma})
    assert g["positive_points"] == 1 and g["worst_im_at"] == (4, 1)
    assert g["worst_im"] == pytest.approx(3e-3)
    assert g["real_abs"] == pytest.approx(3e-3) and g["mats_abs"] == pytest.approx(2e-6)
    assert "U = 0" in fr.format_grade_zero(g)


def test_a_non_interacting_metal_has_the_self_energy_zero(tmp_path):
    toml = write_metal(tmp_path, "u0", n_b=4, U=0.0, n_matsubara=4, real_mesh=(-1.5, 1.5, 61))
    res = fr.run(str(toml), str(tmp_path / "u0"), verbosity=0)
    g = fr.grade_zero(res)
    # The solver's own floor (Lanczos tolerance, the G0^-1 - G^-1 subtraction), not truncation: the model is far
    # below every cap. A truncated run is judged against this number.
    assert g["real_abs"] < 1e-4 and g["mats_abs"] < 1e-4


def test_a_gf_cap_binds_the_units_and_leaves_the_ground_state_alone(tmp_path, capfd):
    toml = write_metal(tmp_path, "m", n_b=6, U=0.5, n_matsubara=4, real_mesh=(-1.5, 1.5, 61))
    full = fr.run(str(toml), str(tmp_path / "full"), verbosity=1)
    capfd.readouterr()
    gf = fr.run(str(toml), str(tmp_path / "gf"), verbosity=1, gf_cap=300)
    out = capfd.readouterr().out
    assert "determinant cap: GF 300 per unit" in out
    assert "REPLAY CAPS: GS unlimited, GF 300" in out
    assert "GS basis stopped" not in out
    assert np.array_equal(gf["gs_energies"], full["gs_energies"])  # the ground state was not truncated
    assert np.max(np.abs(gf["sigma_real"] - full["sigma_real"])) > 1e-3  # the Green's functions were

    gs = fr.run(str(toml), str(tmp_path / "gs"), verbosity=1, gs_cap=300)
    out = capfd.readouterr().out
    assert "GS basis stopped at 300 determinants" in out
    assert "determinant cap: GF unlimited per unit" in out
    assert not np.allclose(gs["gs_energies"], full["gs_energies"], atol=1e-6)


def test_derived_excitation_bounds_leave_a_metal_self_energy_alone(tmp_path, monkeypatch):
    """A model WITH conduction baths against the unrestricted reference, with the bounds actually binding.

    Measured at this size: the derived bounds plus the +1 widening agree with the unrestricted run to ~1e-6 on the
    real axis, while the SAME bounds without the widening are off by ~5e-3. So the test fails for a missing +1 as
    well as for bounds that are too tight, and it asserts the precondition that makes either visible: the bounds
    must exclude part of the space (``max_total < n_valence + n_conduction``), or the comparison is vacuous.
    """
    import impurityModel.ed.greens_function as gf_module
    import impurityModel.ed.groundstate as gs_module

    toml = write_metal(tmp_path, "m", n_orb=1, n_b=6, v=0.3, U=0.5, n_matsubara=4, real_mesh=(-1.5, 1.5, 21))
    seen = []
    observe = gs_module.get_effective_excitation_bounds
    monkeypatch.setattr(
        gs_module, "get_effective_excitation_bounds", lambda *a, **k: seen.append(observe(*a, **k)) or seen[-1]
    )
    bounded = fr.run(str(toml), str(tmp_path / "bounded"), verbosity=0)
    assert seen and all(b is not None and b.max_total < 12 for b in seen)  # 6 valence + 6 conduction spin orbitals

    widen = gf_module.widen_weighted_restrictions
    monkeypatch.setattr(gf_module, "widen_weighted_restrictions", lambda wr, extra=None: wr)
    unwidened = fr.run(str(toml), str(tmp_path / "unwidened"), verbosity=0)
    monkeypatch.setattr(gf_module, "widen_weighted_restrictions", widen)
    monkeypatch.setattr(gs_module, "get_effective_excitation_bounds", lambda *a, **k: None)
    free = fr.run(str(toml), str(tmp_path / "free"), verbosity=0)

    np.testing.assert_allclose(bounded["gs_energies"], free["gs_energies"], atol=1e-8)
    assert np.max(np.abs(bounded["sigma_real"] - free["sigma_real"])) < 1e-4
    assert np.max(np.abs(bounded["sigma"] - free["sigma"])) < 1e-4
    # The comparison can see a too-tight window: take the +1 away and the self-energy moves by orders of magnitude.
    assert np.max(np.abs(unwidened["sigma_real"] - free["sigma_real"])) > 1e-3
