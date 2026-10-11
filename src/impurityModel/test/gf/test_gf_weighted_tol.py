r"""Weight-scaled Green's-function tolerances (``GF_WEIGHTED_TOL``).

The thermal Green's function is :math:`G = \sum_n w_n G_n`, so an error in :math:`G_n` enters
:math:`G` multiplied by :math:`w_n`. Every unit used to converge to the dominant state's
tolerance whatever its weight: on the SrMnO3 production run 16 of the 20 capped removal units
were excited states at :math:`w_n \approx 6\times10^{-5}` and ran ~700 blocks past the point where their
weighted error was below tolerance. These tests pin the contract:

* off (the default) is bit-identical, and the helper is the identity for the dominant state;
* an axis is loosened by ``w_max / w_n``, clamped to the ceiling, and never tightened;
* end to end, only the units of low-weight eigenstates receive looser tolerances.
"""

import contextlib
import dataclasses
import io
import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import greens_function as gf
from impurityModel.ed.gf_convergence import _weighted_axis_tols
from impurityModel.test.gf.test_selfenergy_end_to_end import _basis_and_solver, _meshes, _siam_model


@pytest.fixture(autouse=True)
def _hermetic_knobs(monkeypatch):
    for name in ("GF_TOL", "GF_REAL_TOL", "GF_WEIGHTED_TOL", "GF_WEIGHTED_TOL_CEILING"):
        monkeypatch.delenv(name, raising=False)


# --- the helper ----------------------------------------------------------------------------------


def test_the_dominant_state_keeps_the_base_tolerances_exactly():
    assert _weighted_axis_tols((1e-9, 1e-4), 1.0, 1.0, 1e-4) == (1e-9, 1e-4)


def test_a_light_state_is_loosened_by_the_weight_ratio():
    # w_max / w_n = 1e3: the Matsubara tolerance scales 1e-9 -> 1e-6, below the 1e-4 ceiling.
    assert _weighted_axis_tols((1e-9, 1e-9), 1e-3, 1.0, 1e-4) == pytest.approx((1e-6, 1e-6))


def test_the_loosened_tolerance_is_clamped_to_the_ceiling():
    assert _weighted_axis_tols((1e-9, 1e-9), 1e-9, 1.0, 1e-4) == (1e-4, 1e-4)


def test_an_axis_already_looser_than_the_ceiling_is_never_tightened():
    # gf_real_tol = 1e-3 is above a 1e-4 ceiling; scaling must not pull it down to the ceiling.
    assert _weighted_axis_tols((1e-9, 1e-3), 1e-2, 1.0, 1e-4)[1] == 1e-3


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_a_degenerate_weight_leaves_the_tolerances_alone(bad):
    assert _weighted_axis_tols((1e-9, 1e-4), bad, 1.0, 1e-4) == (1e-9, 1e-4)
    assert _weighted_axis_tols((1e-9, 1e-4), 1.0, bad, 1e-4) == (1e-9, 1e-4)


def test_the_helper_is_monotone_in_the_weight():
    weights = [1.0, 0.5, 1e-2, 1e-4, 1e-8]
    tols = [_weighted_axis_tols((1e-9, 1e-9), w, 1.0, 1e-4)[0] for w in weights]
    assert all(a <= b for a, b in itertools.pairwise(tols))


# --- get_Greens_function -------------------------------------------------------------------------

_TAU = 0.4  # a ground doublet (0.498 each) and an excited doublet (0.00167 each)


def _run(monkeypatch, weighted):
    """``calc_selfenergy`` on the SIAM fixture, recording every ``(energies, axis_tols)`` the monitor was handed."""
    seen = []
    real = gf._gf_eval_meshes

    def recording(matsubara_mesh, omega_mesh, side_i, delta, es, *args, **kwargs):
        seen.append((tuple(float(e) for e in es), tuple(kwargs["axis_tols"])))
        return real(matsubara_mesh, omega_mesh, side_i, delta, es, *args, **kwargs)

    monkeypatch.setattr(gf, "_gf_eval_meshes", recording)
    if weighted:
        monkeypatch.setenv("GF_WEIGHTED_TOL", "1")
    basis, solver = _basis_and_solver()
    basis = dataclasses.replace(basis, tau=_TAU)
    with contextlib.redirect_stdout(io.StringIO()):
        from impurityModel.ed.selfenergy import calc_selfenergy

        result = calc_selfenergy(_siam_model(), _meshes(), basis, solver, comm=MPI.COMM_WORLD, verbosity=0)
    return result, MPI.COMM_WORLD.allgather(seen)


@pytest.mark.mpi
def test_off_hands_every_unit_the_base_tolerances(monkeypatch):
    _result, per_rank = _run(monkeypatch, weighted=False)
    calls = [c for rank_calls in per_rank for c in rank_calls]
    assert calls, "the monitor must have been configured at least once"
    assert len({tols for _es, tols in calls}) == 1


@pytest.mark.mpi
def test_on_loosens_only_the_low_weight_units(monkeypatch):
    result, per_rank = _run(monkeypatch, weighted=True)
    calls = [c for rank_calls in per_rank for c in rank_calls]
    energies = sorted({e for es, _tols in calls for e in es})
    assert len(energies) >= 2, "the fixture must retain an excited manifold to be meaningful"
    e_min = energies[0]
    base = min(tols for es, tols in calls if min(es) == e_min)
    assert all(tols == base for es, tols in calls if min(es) == e_min), "the dominant units keep the base"
    light = [tols for es, tols in calls if min(es) > e_min + 1e-9]
    assert light, "an excited unit must have been configured"
    for tols in light:
        assert all(t >= b for t, b in zip(tols, base)), "never tighter than the base"
        assert tols[0] > base[0], "the Matsubara axis must have been loosened"
        assert tols[0] <= 1e-4 * (1 + 1e-12), "clamped to the ceiling"
    if MPI.COMM_WORLD.rank == 0:
        assert np.all(np.isfinite(result["sigma"]))


@pytest.mark.mpi
def test_on_moves_sigma_by_far_less_than_the_loosened_tolerance(monkeypatch):
    off, _ = _run(monkeypatch, weighted=False)
    on, _ = _run(monkeypatch, weighted=True)
    if MPI.COMM_WORLD.rank == 0:
        for key in ("sigma", "sigma_real"):
            change = np.max(np.abs(on[key] - off[key])) / np.max(np.abs(off[key]))
            # Excited weight 3.3e-3 of the ensemble times a <=1e-4 relative error.
            assert change < 1e-6, f"{key}: {change:.3e}"


# --- the cost weights the tolerances imply -------------------------------------------------------


def test_expected_blocks_is_monotone_and_the_base_ratio_is_one():
    from impurityModel.ed.gf_units import expected_blocks, tolerance_cost_ratios

    tols = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    blocks = [expected_blocks(t) for t in tols]
    assert all(a >= b for a, b in itertools.pairwise(blocks))
    ratios = tolerance_cost_ratios(tols, 1e-9)
    assert ratios[0] == 1.0 and np.all(ratios <= 1.0)


def test_a_loosened_unit_weighs_what_the_srmno3_run_measured():
    """~40 blocks at the loosened 8e-6 against ~700 at 1e-9 (SrMnO3 cubic, 2026-10-08)."""
    from impurityModel.ed.gf_units import tolerance_cost_ratios

    ratio = float(tolerance_cost_ratios([8e-6], 1e-9)[0])
    assert 0.04 < ratio < 0.08


def test_the_packer_gives_the_dominant_units_the_ranks():
    """Equal seed mass would split 8 ranks evenly; the cost ratios send most to the two dominant units."""
    from impurityModel.ed.basis_split import _pack_units
    from impurityModel.ed.gf_units import tolerance_cost_ratios

    seed_mass = np.ones(10)
    tols = [1e-9, 1e-9] + [8e-6] * 8
    equal_subgroups, equal_procs = _pack_units(seed_mass, 16, 1.0)
    weighted_subgroups, weighted_procs = _pack_units(seed_mass * tolerance_cost_ratios(tols, 1e-9), 16, 1.0)
    dominant = lambda subgroups, procs: max(int(p) for g, p in zip(subgroups, procs) if 0 in g or 1 in g)  # noqa: E731
    assert dominant(weighted_subgroups, weighted_procs) > dominant(equal_subgroups, equal_procs)


@pytest.mark.mpi
def test_on_scales_the_dispatch_weights_of_the_low_weight_units_only(monkeypatch):
    seen = {}
    real = gf.run_units_distributed

    def recording(basis, unit_seeds, unit_weights, *args, **kwargs):
        seen["weights"] = np.array(unit_weights, dtype=float)
        return real(basis, unit_seeds, unit_weights, *args, **kwargs)

    monkeypatch.setattr(gf, "run_units_distributed", recording)
    _run(monkeypatch, weighted=False)
    off = seen["weights"]
    _run(monkeypatch, weighted=True)
    on = seen["weights"]
    ratio = on / off
    assert np.all(ratio <= 1.0 + 1e-12)
    assert np.isclose(ratio.max(), 1.0), "the dominant units keep their weight exactly"
    assert ratio.min() < 0.5, "an excited-state unit must weigh less"
