"""The report's manifold tolerance: one floor, never tighter than the solve."""

import numpy as np

from impurityModel.ed.cipsi_solver import _degeneracy_tol
from impurityModel.ed.degeneracy import MANIFOLD_TOL
from impurityModel.ed.gs_report import manifold_tol
from impurityModel.ed.observables import DEGENERACY_TOL


def test_the_observables_tolerance_is_the_shared_floor():
    assert DEGENERACY_TOL == MANIFOLD_TOL == 1e-6


def test_unknown_solver_cutoff_leaves_the_report_floor():
    assert manifold_tol(np.array([0.0, 1.0])) == MANIFOLD_TOL


def test_production_cutoff_gives_the_report_floor():
    es = np.array([-51.78, -51.78, -51.78])
    assert manifold_tol(es, 1e-12) == MANIFOLD_TOL


def test_a_looser_solver_tolerance_wins_over_the_floor():
    # A huge spectrum makes the solver's rounding-noise term exceed the report floor.
    es = np.array([-1e12, 0.0])
    assert _degeneracy_tol(es, 1e-12) > MANIFOLD_TOL
    assert manifold_tol(es, 1e-12) == _degeneracy_tol(es, 1e-12)
