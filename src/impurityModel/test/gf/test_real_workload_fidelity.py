r"""A replay of an archived solver call must pass the archive's options on, not the library's defaults.

``real_workload`` rebuilds a production ``calc_selfenergy`` call from ``impurityModel_data.h5``. It once
dropped the archive's ``excitation_budget`` (8 for the SrMnO3 run) and ``BasisOptions`` fell back to 4: the
replay's ground state had 169k determinants where production's had 277k, E0 sat 1.2 meV high, and every
Green's-function support came out several times smaller. Nothing failed; the numbers were just for another
problem. These tests pin that every solver-relevant attribute the archive records reaches the options, and
that an attribute the helper does not consume is reported instead of vanishing.
"""

import h5py
import numpy as np
import pytest

from impurityModel.ed.model import EXCITATION_BUDGET_DEFAULT
from impurityModel.test.support.real_workload import build_options, load_workload

N_IMP = 10


def _archive(path, **attrs):
    """A minimal one-group archive with a 10-orbital impurity and no bath."""
    base = {"nominal occupation": 3, "tau": 0.025, "delta": 0.01, "slater_min": 1.49e-8}
    with h5py.File(path, "w") as f:
        g = f.create_group("Mn 1")
        g.create_dataset("H solver", data=np.diag(np.linspace(-1.0, 1.0, N_IMP)))
        g.create_dataset("U", data=np.zeros((N_IMP,) * 4))
        g.create_dataset("Matsubara frequency mesh", data=np.linspace(0.1, 5.0, 8))
        g.create_dataset("Real frequency mesh", data=np.linspace(-1.0, 1.0, 9))
        g.create_dataset("Rot to spherical", data=np.eye(N_IMP))
        g.create_dataset("Impurity orbitals", data=np.arange(N_IMP))
        for key, value in {**base, **attrs}.items():
            g.attrs[key] = value
    return path


def test_the_excitation_budget_reaches_the_basis_options(tmp_path):
    workload = load_workload(_archive(tmp_path / "a.h5", excitation_budget=8))
    _model, _meshes, basis, _solver = build_options(workload)
    assert workload["excitation_budget"] == 8
    assert basis.excitation_budget == 8, "the archive's budget, not the library default"


@pytest.mark.parametrize("recorded", [{}, {"excitation_budget": "None"}])
def test_an_unrecorded_budget_is_left_to_the_library_default(tmp_path, recorded):
    workload = load_workload(_archive(tmp_path / "a.h5", **recorded))
    assert workload["excitation_budget"] is None
    assert build_options(workload)[2].excitation_budget == EXCITATION_BUDGET_DEFAULT


def test_the_other_solver_attributes_are_passed_on_too(tmp_path):
    # An archive is self-consistent: the Lanczos tolerances and the thermal cutoff on a Lanczos run ...
    lanczos_archive = _archive(
        tmp_path / "lanczos.h5",
        e_pt2_tol=1e-9,
        de2_min=1e-12,
        gf_real_tol=1e-4,
        gf_min_weight=1e-3,
        gf_method="lanczos",
    )
    workload = load_workload(lanczos_archive)
    _model, _meshes, basis, solver = build_options(workload, gf_real_tol="archive")
    assert (basis.e_pt2_tol, basis.de2_min) == (1e-9, 1e-12)
    assert solver.gf_real_tol == 1e-4
    assert solver.gf_min_weight == 1e-3, "the archive's thermal-weight cutoff, unless the caller overrides it"
    assert build_options(workload, gf_min_weight=0.5)[3].gf_min_weight == 0.5
    # ... and the admission policy on a BiCGSTAB run.
    bicgstab_archive = _archive(tmp_path / "bicgstab.h5", gf_admission="outer", gf_admit_tol=1e-3, gf_method="bicgstab")
    bicgstab = build_options(load_workload(bicgstab_archive), gf_method="bicgstab")[3]
    assert (bicgstab.gf_admission, bicgstab.gf_admit_tol) == ("outer", 1e-3)


def test_an_attribute_nobody_consumes_is_reported_not_dropped(tmp_path):
    workload = load_workload(_archive(tmp_path / "a.h5", excitation_budget=8, mystery_option=3))
    assert workload["ignored_attrs"] == ["mystery_option"]


def test_the_bath_fit_provenance_is_not_flagged(tmp_path):
    path = _archive(
        tmp_path / "a.h5",
        bath_geometry="linked_chain",
        n_baths=4,
        weight_function="unit",
        **{"solver line": "3 4 8 linked_chain", "impurityModel version": "1.0"},
    )
    assert load_workload(path)["ignored_attrs"] == []
