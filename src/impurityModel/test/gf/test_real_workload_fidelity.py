r"""A replay of an archived solver call must pass the archive's options on, not the library's defaults.

``real_workload`` rebuilds a production ``calc_selfenergy`` call from ``impurityModel_data.h5``. It once
dropped the archive's ``excitation_budget`` (8 for the SrMnO3 run) and ``BasisOptions`` fell back to 4 (the
budget is retired now; the same rule holds for every option that remains): the
replay's ground state had 169k determinants where production's had 277k, E0 sat 1.2 meV high, and every
Green's-function support came out several times smaller. Nothing failed; the numbers were just for another
problem. These tests pin that every solver-relevant attribute the archive records reaches the options, and
that an attribute the helper does not consume is reported instead of vanishing.
"""

import h5py
import numpy as np

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


def test_a_retired_excitation_budget_is_neither_replayed_nor_flagged(tmp_path):
    workload = load_workload(_archive(tmp_path / "a.h5", excitation_budget=8))
    assert "excitation_budget" not in workload and workload["ignored_attrs"] == []
    build_options(workload)  # BasisOptions has no such field any more


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


def test_the_kernel_the_archive_recorded_is_the_one_replayed(tmp_path):
    """An archive written by a bicgstab run was once replayed as Lanczos, and its attribute counted as consumed."""
    recorded = load_workload(_archive(tmp_path / "b.h5", gf_method="bicgstab"))
    assert recorded["gf_method"] == "bicgstab"
    assert build_options(recorded)[3].gf_method == "bicgstab"
    assert build_options(recorded, gf_method="lanczos")[3].gf_method == "lanczos", "an explicit choice still wins"
    assert build_options(load_workload(_archive(tmp_path / "u.h5")))[3].gf_method == "lanczos", "unrecorded: Lanczos"


def test_the_archived_prune_options_reach_the_solver_and_an_explicit_argument_beats_them(tmp_path):
    archive = _archive(tmp_path / "p.h5", gf_method="lanczos", gf_prune_tol=1e-4, gf_first_shell_tol=1e-6)
    workload = load_workload(archive)
    assert workload["ignored_attrs"] == []
    solver = build_options(workload)[3]
    assert (solver.gf_prune_tol, solver.gf_first_shell_tol) == (1e-4, 1e-6)
    solver = build_options(workload, gf_prune_tol=3e-3, gf_first_shell_tol=0.0)[3]
    assert (solver.gf_prune_tol, solver.gf_first_shell_tol) == (3e-3, 0.0)


def test_an_explicit_prune_tolerance_replaces_the_archived_settle(tmp_path):
    archive = _archive(tmp_path / "s.h5", gf_method="lanczos", gf_settle=True, gf_settle_ratio=3.0)
    workload = load_workload(archive)
    assert build_options(workload)[3].gf_settle_ratio == 3.0
    solver = build_options(workload, gf_prune_tol=1e-3)[3]
    assert solver.gf_prune_tol == 1e-3 and (solver.gf_settle, solver.gf_settle_ratio) == (None, None)
