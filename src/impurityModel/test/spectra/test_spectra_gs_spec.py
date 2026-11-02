"""P2: the spectra ground state takes the values the self-energy takes, except chain_restrict."""

import inspect

from impurityModel.ed import get_spectra


def _spec_source():
    src = inspect.getsource(get_spectra.run_spectra)
    start = src.index("gs_spec = GroundStateSpec(")
    return src[start : src.index(")\n    basis_setup", start)]


def test_the_spectra_ground_state_spec_takes_the_solver_and_basis_values():
    spec = _spec_source()
    assert "solver.dense_cutoff" in spec
    assert "basis.e_pt2_tol" in spec and "basis.de2_min" in spec


def test_the_spectra_ground_state_keeps_the_complete_multiplet():
    # chain_restrict truncates the initial multiplet; spectra needs it whole (symmetry over energy).
    assert "chain_restrict=False" in _spec_source()


def test_run_spectra_accepts_the_solver_options():
    assert "solver" in inspect.signature(get_spectra.run_spectra).parameters
