"""CIPSI converges the ground-state energy on the residual PT2 energy, not on ``de2_min``.

``de2_min`` bounds each candidate an expansion leaves out; the energy error follows their *sum*.
On a SIAM with many weakly hybridized bath levels the floor alone stops ~40x above its own value,
and the terminating residual predicts the error it leaves (``doc/plans/cipsi_pt2_convergence.md``).
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import groundstate
from impurityModel.ed.cipsi_solver import DEFAULT_E_PT2_TOL, CIPSISolver
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

N_VAL, N_CON = 5, 4
_rng = np.random.default_rng(7)
E_VAL = -np.sort(_rng.uniform(0.5, 4.0, N_VAL))
E_CON = np.sort(_rng.uniform(0.5, 4.0, N_CON))
# Hybridizations spread over a decade and a half: many candidates, most of them weak.
V_VAL = 10 ** _rng.uniform(-1.5, -0.3, N_VAL)
V_CON = 10 ** _rng.uniform(-1.5, -0.3, N_CON)
VALENCE = [2 + 2 * i + s for i in range(N_VAL) for s in (0, 1)]
CONDUCTION = [2 + 2 * N_VAL + 2 * i + s for i in range(N_CON) for s in (0, 1)]


def _hamiltonian():
    terms = {((0, "c"), (0, "a")): -3.0, ((1, "c"), (1, "a")): -3.0, ((0, "c"), (1, "c"), (1, "a"), (0, "a")): 6.0}
    for orbitals, energies, hoppings in ((VALENCE, E_VAL, V_VAL), (CONDUCTION, E_CON, V_CON)):
        for k, b in enumerate(orbitals):
            spin, level = k % 2, k // 2
            terms[((b, "c"), (b, "a"))] = energies[level]
            terms[((spin, "c"), (b, "a"))] = hoppings[level]
            terms[((b, "c"), (spin, "a"))] = hoppings[level]
    return ManyBodyOperator(terms)


def _expand(comm=None, **kwargs):
    """``(e0, basis size, solver)`` of one expansion from the narrow seed basis."""
    basis = Basis(
        {0: [[0, 1]]},
        ({0: [VALENCE]}, {0: [CONDUCTION]}),
        nominal_impurity_occ={0: 1},
        tau=0.001,
        comm=comm,
        verbose=False,
        delta_impurity_occ={0: 1},
        delta_valence_occ={0: 1},
        delta_conduction_occ={0: 1},
    )
    solver = CIPSISolver(basis)
    solver.expand(_hamiltonian(), dense_cutoff=1000, **kwargs)
    es, _ = solver.get_eigenvectors(
        _hamiltonian(), num_wanted=1, max_energy=0.0, dense_cutoff=1000, psi_refs=solver.psi_refs
    )
    return float(np.min(es)), basis.size, solver


@pytest.fixture(scope="module")
def exact():
    """``(e0, size)`` of the whole reachable space (22,356 determinants): nothing is refused."""
    e0, size, solver = _expand(e_pt2_tol=0.0)
    assert solver.last_selection["residual_pt2"] == 0.0
    return e0, size


@pytest.fixture(scope="module")
def exact_e0(exact):
    return exact[0]


def test_a_per_determinant_floor_does_not_bound_the_energy_error(exact_e0):
    """The defect: at ``de2_min = 1e-8`` alone the energy stops tens of times above 1e-8."""
    e0, _, solver = _expand(de2_min=1e-8, e_pt2_tol=None)
    assert e0 - exact_e0 > 10 * 1e-8
    # ...and the residual it leaves is what predicts that error: the ground state is a Kramers
    # doublet, and the residual is summed over the manifold, so the error is half of it.
    residual = solver.convergence_report["residual_pt2"]
    assert 0.4 * residual <= e0 - exact_e0 <= residual
    assert solver.convergence_report["converged"] is None


def test_the_default_converges_the_energy_to_e_pt2_tol(exact):
    exact_e0, exact_size = exact
    e0, size, solver = _expand()
    report = solver.convergence_report
    assert report["e_pt2_tol"] == DEFAULT_E_PT2_TOL
    assert report["converged"] is True and report["residual_pt2"] <= DEFAULT_E_PT2_TOL
    assert report["limited_by"] is None
    assert -1e-12 <= e0 - exact_e0 <= DEFAULT_E_PT2_TOL
    # Converged by selection, not by admitting everything (measured: 744 of 22,356).
    assert size < exact_size / 10


def test_a_looser_tolerance_is_honoured_and_still_bounds_the_error(exact_e0):
    e0, size, solver = _expand(e_pt2_tol=1e-5)
    _, tight_size, _ = _expand()
    assert solver.convergence_report["converged"] is True
    assert e0 - exact_e0 <= 1e-5
    assert size < tight_size


def test_a_floor_that_stops_short_is_reported(exact_e0, capsys):
    _, _, solver = _expand(de2_min=1e-6)
    report = solver.convergence_report
    assert report["converged"] is False and report["limited_by"] == "de2_min"
    assert report["residual_pt2"] > DEFAULT_E_PT2_TOL
    assert "not converged" in capsys.readouterr().out


@pytest.mark.parametrize("cap", [300, 640, 700])
def test_a_binding_cap_is_never_reported_converged(cap):
    """Caps just below the converged size (744) are the trap: the last capped cycle's own tail is
    within tolerance by construction, but `truncate` then drops determinants, so that residual
    describes a basis that no longer exists. Found by review: 640 and 700 read "converged"."""
    solver = _capped_solver(cap=cap)
    report = solver.convergence_report
    assert solver.truncation_report["cap_hit"]
    assert report["converged"] is False and report["limited_by"] == "cap"
    assert report["residual_is_current"] is False


def test_an_uncapped_report_describes_the_returned_basis():
    _, _, solver = _expand()
    assert solver.convergence_report["residual_is_current"] is True


def _capped_solver(cap):
    basis = Basis(
        {0: [[0, 1]]},
        ({0: [VALENCE]}, {0: [CONDUCTION]}),
        nominal_impurity_occ={0: 1},
        tau=0.001,
        comm=None,
        verbose=False,
        truncation_threshold=cap,
        delta_impurity_occ={0: 1},
        delta_valence_occ={0: 1},
        delta_conduction_occ={0: 1},
    )
    solver = CIPSISolver(basis)
    solver.expand(_hamiltonian(), dense_cutoff=1000)
    return solver


@pytest.mark.mpi
def test_converged_energy_and_residual_are_rank_invariant(exact_e0):
    comm = MPI.COMM_WORLD
    e0, _, solver = _expand(comm=comm)
    report = solver.convergence_report
    assert report["converged"] is True
    assert all(r == report for r in comm.allgather(report))
    assert abs(e0 - exact_e0) <= DEFAULT_E_PT2_TOL


def _solve_ground_state():
    return groundstate.solve_ground_state(
        _hamiltonian(),
        {0: [[0, 1]]},
        ({0: [VALENCE]}, {0: [CONDUCTION]}),
        {0: 1},
        tau=0.001,
        slaterWeightMin=0,
        dense_cutoff=1000,
        use_hf_seed=False,
    )


def test_solve_ground_state_reports_the_refinement_converged():
    _, solver, es, _ = _solve_ground_state()
    report = solver.convergence_report
    assert report["converged"] is True and report["e_pt2_tol"] == groundstate.GS_E_PT2_TOL
    assert len(es) <= solver.last_selection["n_references"]


def test_a_manifold_wider_than_the_references_is_not_reported_converged(monkeypatch, capsys):
    """The residual covers only the states `expand` carried; any the thermal widening adds
    afterwards have none, so the report must stop claiming convergence for the manifold."""
    real_expand = groundstate.CIPSISolver.expand

    def expand_that_scored_one_reference(self, *args, **kwargs):
        real_expand(self, *args, **kwargs)
        self.last_selection["n_references"] = 1

    monkeypatch.setattr(groundstate.CIPSISolver, "expand", expand_that_scored_one_reference)
    _, solver, es, _ = _solve_ground_state()
    assert len(es) > 1
    report = solver.convergence_report
    assert report["converged"] is False and report["limited_by"] == "manifold_widened"
    assert "wider than" in capsys.readouterr().out


class _Captured(Exception):
    pass


@pytest.mark.parametrize("e_pt2_tol", [None, 1e-6])
def test_calc_selfenergy_hands_the_basis_tolerance_to_the_ground_state(e_pt2_tol, monkeypatch):
    """BasisOptions.e_pt2_tol (the RSPt solver line's ``e_pt2 X``) is what the production ground
    state converges to; unset, the ground state's own default."""
    import dataclasses

    from impurityModel.ed.selfenergy import calc_selfenergy
    from impurityModel.test.support._nio_workload import as_calc_selfenergy_args, build_selfenergy_inputs

    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        raise _Captured

    monkeypatch.setattr(groundstate, "solve_ground_state", capture)
    args = as_calc_selfenergy_args(build_selfenergy_inputs(nBaths=10, n_omega=3, dense_cutoff=500))
    args["basis"] = dataclasses.replace(args["basis"], e_pt2_tol=e_pt2_tol)
    with pytest.raises(_Captured):
        calc_selfenergy(**args, comm=None)
    assert seen["e_pt2_tol"] == (groundstate.GS_E_PT2_TOL if e_pt2_tol is None else e_pt2_tol)
