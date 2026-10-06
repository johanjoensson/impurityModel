"""Opt-in thermal-weight cutoff for the Green's-function ensemble (``SolverOptions.gf_min_weight``).

The eigensolver keeps every state inside the energy window ``-tau*ln(1e-4)``, and every kept state
costs a full set of Green's-function work units whatever its weight. On the SrMnO3 production run
eight of ten states carried 5.7-6.8e-5 each -- 32 of 40 units for 5.0e-4 of the ensemble. These tests pin
the contract of the cutoff:

* unset, or set below every weight, the result is bit-identical;
* whole degenerate manifolds are kept or dropped, the ground manifold always kept;
* dropping moves the self-energy by an amount of the order of the dropped weight;
* the truncation diagnostic still judges the eigensolver's whole ensemble;
* every front end accepts and validates the option.
"""

import contextlib
import dataclasses
import io

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.model import SolverOptions
from impurityModel.ed.selfenergy import _drop_low_weight_manifolds, calc_selfenergy
from impurityModel.test.gf.test_selfenergy_end_to_end import _basis_and_solver, _meshes, _siam_model

# SrMnO3 (Arrhenius, 2026-10-05): a Kramers ground doublet and four excited doublets at tau = 0.025.
_SMO_TAU = 0.025
_SMO_ES = -12.922717 + np.array([0.0, 0.0, 0.22255, 0.22255, 0.2239, 0.2239, 0.22623, 0.22623, 0.22705, 0.22705])


# --- the filter --------------------------------------------------------------------------------------


def test_the_smo_ensemble_keeps_only_its_ground_doublet():
    psis = [f"psi{i}" for i in range(len(_SMO_ES))]
    kept_psis, kept_es, dropped = _drop_low_weight_manifolds(psis, _SMO_ES, _SMO_TAU, 1e-3, 1e-12)
    assert kept_psis == ["psi0", "psi1"]
    assert isinstance(kept_es, np.ndarray) and np.array_equal(kept_es, _SMO_ES[:2])
    assert len(dropped) == 8
    boltzmann = np.exp(-(_SMO_ES - _SMO_ES[0]) / _SMO_TAU)
    assert sum(w for _de, w in dropped) == pytest.approx(boltzmann[2:].sum() / boltzmann.sum())


def test_a_cutoff_below_every_weight_keeps_everything_in_order():
    es = [0.3, 0.0, 0.1]
    kept_psis, kept_es, dropped = _drop_low_weight_manifolds(["a", "b", "c"], es, 1.0, 1e-6, 1e-12)
    assert (kept_psis, kept_es, dropped) == (["a", "b", "c"], es, [])


def test_the_ground_manifold_is_kept_even_above_its_own_weight():
    # A degenerate ground doublet weighs 0.5 per state; a cutoff of 0.9 must not empty the ensemble.
    kept_psis, _es, dropped = _drop_low_weight_manifolds(["a", "b", "c"], [0.0, 0.0, 0.5], 0.1, 0.9, 1e-12)
    assert kept_psis == ["a", "b"] and len(dropped) == 1


def test_a_degenerate_manifold_straddling_the_cutoff_is_never_split():
    # Two members split by 1e-12 (degenerate to the solver's accuracy) whose weights fall on
    # either side of the cutoff: both go, or both stay.
    tau = 0.1
    e_mid = -tau * np.log(1e-2)  # unnormalised weight exactly 1e-2
    es = [0.0, e_mid - 1e-12, e_mid + 1e-12]
    weights = np.exp(-np.array(es) / tau)
    cutoff = float(np.mean(weights[1:] / weights.sum()))
    kept_psis, _es, _dropped = _drop_low_weight_manifolds(["g", "x", "y"], es, tau, cutoff, 1e-12)
    assert kept_psis in (["g"], ["g", "x", "y"])


def test_the_decision_is_broadcast_from_root():
    comm = MPI.COMM_WORLD
    # Rank 0's third state weighs 1.5e-3 (dropped); elsewhere it would be kept.
    es = [0.0, 0.05, 0.6] if comm.rank == 0 else [0.0, 0.05, 0.05]
    kept_psis, _es, _dropped = _drop_low_weight_manifolds(["a", "b", "c"], es, 0.1, 1e-2, 1e-12, comm=comm)
    assert kept_psis == ["a", "b"], "every rank must follow rank 0's decision"


# --- calc_selfenergy ---------------------------------------------------------------------------------


def _run(solver_changes, tau=0.4):
    # tau = 0.4 retains a ground doublet (0.498 each) and an excited doublet at 0.00167 each.
    basis, solver = _basis_and_solver()
    basis = dataclasses.replace(basis, tau=tau)
    solver = dataclasses.replace(solver, **solver_changes)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = calc_selfenergy(_siam_model(), _meshes(), basis, solver, comm=MPI.COMM_WORLD, verbosity=0)
    return result, out.getvalue()


@pytest.mark.mpi
def test_off_or_below_every_weight_is_bit_identical():
    off, _ = _run({})
    below, _ = _run({"gf_min_weight": 1e-3})
    if MPI.COMM_WORLD.rank == 0:
        assert len(off["gs_energies"]) == 4, "the fixture must retain an excited manifold to be meaningful"
        for key in ("sigma", "sigma_real", "sigma_static", "sigma_moment_1", "sigma_moment_2"):
            assert np.array_equal(off[key], below[key]), key


@pytest.mark.mpi
def test_dropping_a_manifold_moves_sigma_by_the_order_of_its_weight():
    off, _ = _run({})
    cut, text = _run({"gf_min_weight": 1e-2})
    if MPI.COMM_WORLD.rank == 0:
        assert len(cut["gs_energies"]) == 2
        assert "dropped 2 of 4 thermal state(s)" in text
        dropped_weight = 2 * 0.00167
        for key in ("sigma", "sigma_real"):
            change = np.max(np.abs(cut[key] - off[key])) / np.max(np.abs(off[key]))
            assert 0.0 < change < 10 * dropped_weight, f"{key}: relative change {change:.3e}"


def test_the_truncation_check_judges_the_whole_ensemble(monkeypatch):
    """With the filter on, ``get_Greens_function`` must still be handed the unfiltered energies."""
    from impurityModel.ed import selfenergy

    seen = {}
    real = selfenergy.get_Greens_function

    def recording(*args, **kwargs):
        seen["es"], seen["ensemble_es"] = list(kwargs["es"]), list(kwargs["ensemble_es"])
        return real(*args, **kwargs)

    monkeypatch.setattr(selfenergy, "get_Greens_function", recording)
    _run({"gf_min_weight": 1e-2})
    assert len(seen["es"]) == 2 and len(seen["ensemble_es"]) == 4


# --- the front ends ---------------------------------------------------------------------------------


def test_solver_options_validate_the_cutoff():
    assert SolverOptions().gf_min_weight is None
    assert SolverOptions(gf_min_weight=1e-3).gf_min_weight == 1e-3
    for bad in (0.0, 1.0, -1e-3):
        with pytest.raises(ValueError, match="gf_min_weight must lie in"):
            SolverOptions(gf_min_weight=bad)


def test_the_toml_and_cli_take_the_cutoff(tmp_path):
    from impurityModel.inputformat.build import build
    from impurityModel.inputformat.reader import load_input
    from impurityModel.scripts import selfenergy as cli
    from impurityModel.test.inputformat.test_build import GOLDEN_H0, SELFENERGY
    from impurityModel.test.misc.test_cli import _parse

    path = tmp_path / "in.toml"
    path.write_text(SELFENERGY.format(h0=GOLDEN_H0) + "\n[solver]\ngf_min_weight = 1e-3\n")
    assert build(load_input(path)).solver.gf_min_weight == 1e-3

    args = _parse(cli.add_arguments, ["h0.pickle", "--gf-min-weight", "1e-3"])
    assert cli.apply_solver_overrides(SolverOptions(), args).gf_min_weight == 1e-3
