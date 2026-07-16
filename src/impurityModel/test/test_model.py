"""Unit tests for :mod:`impurityModel.ed.model` (the impurity-model construction layer)."""

import os
from collections import OrderedDict

import numpy as np
import pytest

from impurityModel.ed import atomic_physics
from impurityModel.ed.model import (
    BasisOptions,
    ImpurityModel,
    Meshes,
    SolverOptions,
    atomic_u4,
    load_selfenergy_archive,
)
from impurityModel.ed.operator_algebra import c2i

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _inline_u4(l, slater):
    """The u4 assembly exactly as it was copy-pasted into the self-energy / susceptibility CLIs."""
    n = 2 * (2 * l + 1)
    u4 = np.zeros((n, n, n, n), dtype=complex)
    u_op = atomic_physics.getUop(l1=l, l2=l, l3=l, l4=l, R=slater)
    n_baths_for_c2i = OrderedDict({l: 0})
    for process, val in u_op.items():
        i = c2i(n_baths_for_c2i, process[0][0])
        j = c2i(n_baths_for_c2i, process[1][0])
        k = c2i(n_baths_for_c2i, process[2][0])
        m = c2i(n_baths_for_c2i, process[3][0])
        u4[i, j, m, k] = 2.0 * val
    return u4


def test_atomic_u4_matches_inline_assembly():
    """atomic_u4 reproduces the inline loop it replaced, bit for bit."""
    slater = [7.5, 0, 9.9, 0, 6.6]
    got = atomic_u4(2, slater)
    expected = _inline_u4(2, slater)
    assert got.shape == (10, 10, 10, 10)
    np.testing.assert_array_equal(got, expected)


def test_atomic_u4_p_shell_shape():
    """A p-shell tensor has the right dimension (dedup must not hard-code the d-shell size)."""
    u4 = atomic_u4(1, [0.0, 2.0, 1.0])
    assert u4.shape == (6, 6, 6, 6)


def test_option_group_defaults():
    """The option groups expose today's effective defaults."""
    solver = SolverOptions()
    assert solver.gf_method == "lanczos"
    assert solver.sparse_green is True
    assert solver.dense_cutoff == 500

    meshes = Meshes(w=np.linspace(-1, 1, 5))
    assert meshes.iw is None
    assert meshes.delta == 0.1

    basis = BasisOptions(nominal_occ={2: 8})
    assert basis.nominal_occ == {2: 8}
    assert basis.dN is None
    assert basis.chain_restrict is True


def test_impurity_model_derives_spin_orbitals_and_indices():
    """n_spin_orbitals and impurity_indices are derived from h0 / the orbital layout."""
    # Two impurity orbitals (0, 1) and one bath orbital (2).
    h0 = {(((0), "c"), ((0), "a")): 1.0, (((2), "c"), ((2), "a")): -1.0}
    model = ImpurityModel(h0=h0, impurity_orbitals={0: [0, 1]}, rot_to_spherical=np.eye(2), u4=None)
    assert model.n_spin_orbitals == 3
    assert model.impurity_indices == [0, 1]


def test_from_h0_file_nio_pickle():
    """from_h0_file builds a coherent single-shell model from the NiO d-shell pickle."""
    h0_file = os.path.join(REPO_ROOT, "h0", "h0_NiO_10bath.pickle")
    if not os.path.exists(h0_file):
        pytest.skip("NiO h0 pickle fixture not available")

    model = ImpurityModel.from_h0_file(h0_file, l=2, n_baths=10, slater=[7.5, 0, 9.9, 0, 6.6], xi=0.0)

    # 10 impurity spin-orbitals (d-shell) + 10 bath = 20.
    assert model.impurity_orbitals == {2: list(range(10))}
    assert model.n_spin_orbitals == 20
    assert model.u4.shape == (10, 10, 10, 10)
    np.testing.assert_array_equal(model.rot_to_spherical, np.eye(10, dtype=complex))
    # h0 is single-index (all keys are integer spin-orbital indices) and non-empty.
    assert model.h0
    for process in model.h0:
        for index, action in process:
            assert isinstance(index, int)
            assert action in ("c", "a")


def test_from_h0_file_matches_nio_workload_inputs():
    """from_h0_file's h0/u4 agree with the hand-built _nio_workload inputs (the golden path)."""
    from impurityModel.test._nio_workload import build_selfenergy_inputs

    h0_file = os.path.join(REPO_ROOT, "h0", "h0_NiO_10bath.pickle")
    if not os.path.exists(h0_file):
        pytest.skip("NiO h0 pickle fixture not available")

    inputs = build_selfenergy_inputs(nBaths=10, xi=0.0, chargeTransferCorrection=None)
    model = ImpurityModel.from_h0_file(h0_file, l=2, n_baths=10, slater=[7.5, 0, 9.9, 0, 6.6], xi=0.0)

    np.testing.assert_array_equal(model.u4, inputs["u4"])
    assert model.h0 == inputs["h0"]
    assert model.impurity_orbitals == inputs["impurity_orbitals"]


def _write_synthetic_archive(path):
    """Write a minimal impurityModel_data.h5 (one 2-orbital impurity + 1 bath) like the interface does."""
    import h5py

    h_solver = np.zeros((3, 3), dtype=complex)
    h_solver[0, 0] = 0.5
    h_solver[1, 1] = 0.5
    h_solver[2, 2] = -1.0
    h_solver[0, 2] = h_solver[2, 0] = 0.3  # impurity-bath hybridization
    u4 = np.zeros((2, 2, 2, 2), dtype=complex)
    u4[0, 1, 1, 0] = 2.0
    with h5py.File(path, "w") as f:
        f.attrs["last iteration"] = 2
        g = f.create_group("X 2")
        g.create_dataset("H solver", data=h_solver)
        g.create_dataset("U", data=u4)
        g.create_dataset("Matsubara frequency mesh", data=np.linspace(0.1, 5.0, 8))
        g.create_dataset("Real frequency mesh", data=np.linspace(-3, 3, 11))
        g.create_dataset("Rot to spherical", data=np.eye(2, dtype=complex))
        g.create_dataset("Impurity orbitals", data=np.array([0, 1]))
        g.attrs["nominal occupation"] = 1
        g.attrs["tau"] = 0.0025
        g.attrs["delta"] = 0.1
        g.attrs["mv"] = "None"
        g.attrs["reort"] = "partial"
        g.attrs["dense_cutoff"] = 500
        g.attrs["spin_flip_dj"] = False
        g.attrs["chain_restrict"] = True
        g.attrs["occ_cutoff"] = 1e-6
        g.attrs["truncation_threshold"] = "None"
        g.attrs["slater_min"] = 0.0
        g.attrs["dN"] = "None"
        g.attrs["sparse_green"] = True


def test_from_hdf5_reads_archive_group(tmp_path):
    """ImpurityModel.from_hdf5 recovers the physics arrays from an archive group."""
    archive = tmp_path / "impurityModel_data.h5"
    _write_synthetic_archive(str(archive))

    model = ImpurityModel.from_hdf5(str(archive))  # defaults to last iteration / first cluster
    assert model.n_spin_orbitals == 3  # H solver is 3x3
    assert model.impurity_orbitals == {0: [0, 1]}
    assert model.u4.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(model.rot_to_spherical, np.eye(2, dtype=complex))
    # h0 holds the non-zero H solver entries, including the hybridization.
    assert model.h0[((0, "c"), (2, "a"))] == 0.3


def test_load_selfenergy_archive_recovers_options(tmp_path):
    """load_selfenergy_archive rebuilds model + meshes + basis + solver from the archive."""
    archive = tmp_path / "impurityModel_data.h5"
    _write_synthetic_archive(str(archive))

    model, meshes, basis, solver, label = load_selfenergy_archive(str(archive))
    assert label == "X 2"
    # The stored Matsubara mesh is real-valued nu; the Meshes.iw is the complex z = i*nu.
    assert meshes.iw is not None and meshes.iw.dtype.kind == "c"
    np.testing.assert_allclose(meshes.iw.imag, np.linspace(0.1, 5.0, 8))
    assert len(meshes.w) == 11
    assert basis.nominal_occ == {0: 1}
    assert basis.mixed_valence is None
    assert basis.tau == 0.0025
    assert basis.chain_restrict is True
    assert basis.dN is None
    assert solver.reort == "partial"
    assert solver.dense_cutoff == 500
    assert isinstance(model, ImpurityModel)
