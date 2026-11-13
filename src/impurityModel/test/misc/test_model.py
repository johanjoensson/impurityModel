"""Unit tests for :mod:`impurityModel.ed.model` (the impurity-model construction layer)."""

import os
from collections import OrderedDict

import numpy as np
import pytest

from impurityModel.ed import atomic_physics
from impurityModel.ed.archive import load_selfenergy_archive
from impurityModel.ed.lie_algebra import extract_tensors
from impurityModel.ed.model import (
    BasisOptions,
    ImpurityModel,
    Meshes,
    SolverOptions,
    atomic_u4,
)
from impurityModel.ed.operator_algebra import c2i, matrixToIOp
from impurityModel.ed.options import DEFAULT_GF_PRUNE_TOL, DEFAULT_GF_REFINE_ROUNDS

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


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
    # u4 is stored as an operator dict (RSPt convention), not the dense tensor.
    assert model.u4 == atomic_physics.getUop_from_rspt_u4(atomic_u4(2, [7.5, 0, 9.9, 0, 6.6]))
    np.testing.assert_array_equal(model.rot_to_spherical, np.eye(10, dtype=complex))
    # h0 is single-index (all keys are integer spin-orbital indices) and non-empty.
    assert model.h0
    for process in model.h0:
        for index, action in process:
            assert isinstance(index, int)
            assert action in ("c", "a")


def test_from_h0_file_matches_nio_workload_inputs():
    """from_h0_file's h0/u4 agree with the hand-built _nio_workload inputs (the golden path)."""
    from impurityModel.test.support._nio_workload import build_selfenergy_inputs

    h0_file = os.path.join(REPO_ROOT, "h0", "h0_NiO_10bath.pickle")
    if not os.path.exists(h0_file):
        pytest.skip("NiO h0 pickle fixture not available")

    inputs = build_selfenergy_inputs(nBaths=10, xi=0.0, chargeTransferCorrection=None)
    model = ImpurityModel.from_h0_file(h0_file, l=2, n_baths=10, slater=[7.5, 0, 9.9, 0, 6.6], xi=0.0)

    # model.u4 is the operator-dict form; inputs["u4"] (the pre-refactor hand-built path) is
    # still the dense RSPt tensor -- compare through the same conversion.
    assert model.u4 == atomic_physics.getUop_from_rspt_u4(inputs["u4"])
    assert model.h0 == inputs["h0"]
    assert model.impurity_orbitals == inputs["impurity_orbitals"]


def _write_synthetic_archive(path, basis_kwargs=None, solver_kwargs=None):
    """Write a minimal impurityModel_data.h5 (one 2-orbital impurity + 1 bath) through the archive writer."""
    import h5py

    from impurityModel.ed.archive import write_selfenergy_group

    h_solver = np.zeros((3, 3), dtype=complex)
    h_solver[0, 0] = 0.5
    h_solver[1, 1] = 0.5
    h_solver[2, 2] = -1.0
    h_solver[0, 2] = h_solver[2, 0] = 0.3  # impurity-bath hybridization
    u4 = np.zeros((2, 2, 2, 2), dtype=complex)
    u4[0, 1, 1, 0] = 2.0
    basis = BasisOptions(nominal_occ={0: 1}, tau=0.0025, slater_weight_min=0.0, **(basis_kwargs or {}))
    solver = SolverOptions(dense_cutoff=500, **(solver_kwargs or {}))
    with h5py.File(path, "w") as f:
        f.attrs["last iteration"] = 2
        write_selfenergy_group(
            f.create_group("X 2"),
            basis=basis,
            solver=solver,
            delta=0.1,
            w=np.linspace(-3, 3, 11),
            iw=np.linspace(0.1, 5.0, 8),
            h_solver=h_solver,
            u4=u4,
            dc=np.zeros((2, 2), dtype=complex),
            rot_to_spherical=np.eye(2, dtype=complex),
            impurity_orbitals=np.array([0, 1]),
            valence_orbitals=np.array([2]),
            conduction_orbitals=np.array([], dtype=int),
            h_dft=h_solver[:2, :2],
            h_bath=h_solver[2:, 2:],
            v=h_solver[:2, 2:],
        )


def _archive_with_attrs(path, **solver_kwargs):
    _write_synthetic_archive(str(path), solver_kwargs=solver_kwargs)


def test_from_hdf5_reads_archive_group(tmp_path):
    """ImpurityModel.from_hdf5 recovers the physics arrays from an archive group."""
    archive = tmp_path / "impurityModel_data.h5"
    _write_synthetic_archive(str(archive))

    model = ImpurityModel.from_hdf5(str(archive))  # defaults to last iteration / first cluster
    assert model.n_spin_orbitals == 3  # H solver is 3x3
    assert model.impurity_orbitals == {0: [0, 1]}
    u4 = np.zeros((2, 2, 2, 2), dtype=complex)
    u4[0, 1, 1, 0] = 2.0
    assert model.u4 == atomic_physics.getUop_from_rspt_u4(u4)
    # The fixture writes an all-zero double counting: an empty operator.
    assert not model.dc
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
    assert solver.dense_cutoff == 500
    assert solver.gf_method == "lanczos"
    assert isinstance(model, ImpurityModel)


def test_archive_e_pt2_tol_round_trips_and_its_absence_means_the_default(tmp_path):
    archive = tmp_path / "impurityModel_data.h5"
    _write_synthetic_archive(str(archive), basis_kwargs={"e_pt2_tol": 1e-6})
    _, _, basis, _, _ = load_selfenergy_archive(str(archive))
    assert basis.e_pt2_tol == 1e-6

    archive = tmp_path / "impurityModel_data_old.h5"
    _write_synthetic_archive(str(archive))
    _, _, basis, _, _ = load_selfenergy_archive(str(archive))
    assert basis.e_pt2_tol is None


@pytest.mark.parametrize("value", [0.0, -1e-8])
def test_a_non_positive_e_pt2_tol_is_refused(value):
    with pytest.raises(ValueError, match="e_pt2_tol must be positive"):
        BasisOptions(nominal_occ={0: 1}, e_pt2_tol=value)


def test_a_negative_de2_min_is_refused_and_zero_means_no_floor():
    with pytest.raises(ValueError, match="de2_min must be non-negative"):
        BasisOptions(nominal_occ={0: 1}, de2_min=-1e-8)
    assert BasisOptions(nominal_occ={0: 1}, de2_min=0.0).de2_min == 0.0


def _blocks():
    """A small (2 impurity + 1 bath) block set for the from_blocks / from_solver_matrix tests."""
    H_imp = np.array([[0.5, 0.1], [0.1, 0.5]], dtype=complex)
    V = np.array([[0.3, 0.2]], dtype=complex)  # (n_bath=1, n_imp=2)
    H_bath = np.array([[-1.0]], dtype=complex)
    return H_imp, V, H_bath


def _stacked(H_imp, V, H_bath):
    n_imp, n_bath = H_imp.shape[0], H_bath.shape[0]
    H = np.zeros((n_imp + n_bath, n_imp + n_bath), dtype=complex)
    H[:n_imp, :n_imp] = H_imp
    H[n_imp:, n_imp:] = H_bath
    H[n_imp:, :n_imp] = V
    H[:n_imp, n_imp:] = V.conj().T
    return H


def test_from_solver_matrix_builds_operator_and_layout():
    """from_solver_matrix converts the matrix to an operator and derives the orbital layout."""
    H_imp, V, H_bath = _blocks()
    H = _stacked(H_imp, V, H_bath)
    u4 = np.zeros((2, 2, 2, 2), dtype=complex)
    model = ImpurityModel.from_solver_matrix(
        H, 2, u4=u4, rot_to_spherical=np.eye(2, dtype=complex), bath_valence_conduction=([2], [])
    )
    assert model.h0 == matrixToIOp(H)
    assert model.impurity_orbitals == {0: [0, 1]}
    assert model.impurity_indices == [0, 1]
    assert model.n_spin_orbitals == 3
    # The valence/conduction split populates bath_states (needed by the DC/spectra paths).
    assert model.bath_states == ({0: [2]}, {0: []})


def test_from_blocks_matches_hand_stacked_matrix():
    """from_blocks stacks [[H_imp, V^dag],[V, H_bath]] and yields that matrix's operator."""
    H_imp, V, H_bath = _blocks()
    model = ImpurityModel.from_blocks(H_imp, V, H_bath, rot_to_spherical=np.eye(2, dtype=complex))
    assert model.h0 == matrixToIOp(_stacked(H_imp, V, H_bath))
    assert model.impurity_orbitals == {0: [0, 1]}
    assert model.n_spin_orbitals == 3
    # Hermitian off-diagonal placement: impurity->bath term is the conjugate of bath->impurity.
    assert model.h0[((2, "c"), (0, "a"))] == np.conj(model.h0[((0, "c"), (2, "a"))])


def test_from_blocks_agrees_with_from_solver_matrix():
    """The block front-door and the matrix core produce the same model."""
    H_imp, V, H_bath = _blocks()
    via_blocks = ImpurityModel.from_blocks(
        H_imp, V, H_bath, rot_to_spherical=np.eye(2, dtype=complex), bath_valence_conduction=([2], [])
    )
    via_matrix = ImpurityModel.from_solver_matrix(
        _stacked(H_imp, V, H_bath), 2, rot_to_spherical=np.eye(2, dtype=complex), bath_valence_conduction=([2], [])
    )
    assert via_blocks.h0 == via_matrix.h0
    assert via_blocks.impurity_orbitals == via_matrix.impurity_orbitals
    assert via_blocks.bath_states == via_matrix.bath_states


def test_from_blocks_without_split_leaves_bath_states_none():
    """Omitting the valence/conduction split leaves bath_states None (the self-energy path re-derives it)."""
    H_imp, V, H_bath = _blocks()
    model = ImpurityModel.from_blocks(H_imp, V, H_bath, rot_to_spherical=np.eye(2, dtype=complex))
    assert model.bath_states is None


def test_from_solver_matrix_dc_and_u4_default_to_none():
    """Omitting dc/u4 (both Optional=None) must not crash -- the model has no DC/interaction."""
    H_imp, V, H_bath = _blocks()
    H = _stacked(H_imp, V, H_bath)
    model = ImpurityModel.from_solver_matrix(H, 2, rot_to_spherical=np.eye(2, dtype=complex))
    assert model.dc is None
    assert model.u4 is None


def test_from_solver_matrix_dc_round_trips_through_operator_dict():
    """dc is stored as an operator dict; extract_tensors recovers the original dense matrix."""
    H_imp, V, H_bath = _blocks()
    H = _stacked(H_imp, V, H_bath)
    dc = np.array([[0.5, 0.1j], [-0.1j, 0.5]], dtype=complex)
    model = ImpurityModel.from_solver_matrix(H, 2, dc=dc, rot_to_spherical=np.eye(2, dtype=complex))
    assert isinstance(model.dc, dict)
    recovered = extract_tensors(model.dc, n_orb=2, two_body=False)[0]
    np.testing.assert_allclose(recovered, dc)


def test_from_hdf5_with_dc_dataset_recovers_dc(tmp_path):
    """A 'DC' dataset in the archive round-trips to model.dc as an operator dict."""
    import h5py

    archive = tmp_path / "impurityModel_data.h5"
    _write_synthetic_archive(str(archive))
    dc = np.array([[0.25, 0.0], [0.0, 0.25]], dtype=complex)
    with h5py.File(str(archive), "a") as f:
        f["X 2"]["DC"][...] = dc

    model = ImpurityModel.from_hdf5(str(archive))
    assert isinstance(model.dc, dict)
    recovered = extract_tensors(model.dc, n_orb=2, two_body=False)[0]
    np.testing.assert_allclose(recovered, dc)


def test_archive_records_and_replays_the_admission_policy(tmp_path):
    archive = tmp_path / "impurityModel_data.h5"
    _write_synthetic_archive(
        str(archive), solver_kwargs={"gf_method": "bicgstab", "gf_admission": "outer", "gf_admit_tol": 1e-5}
    )
    _, _, _, solver, _ = load_selfenergy_archive(str(archive))
    assert (solver.gf_method, solver.gf_admission) == ("bicgstab", "outer")
    assert solver.gf_admit_tol == pytest.approx(1e-5)


def test_archive_records_and_replays_the_support_settle(tmp_path):
    archive = tmp_path / "impurityModel_data.h5"
    _archive_with_attrs(archive, gf_settle=False)
    _, _, _, solver, _ = load_selfenergy_archive(str(archive))
    assert solver.gf_settle is False and solver.gf_settle_ratio is None
    archive = tmp_path / "second.h5"
    _archive_with_attrs(archive, gf_settle=True, gf_settle_ratio=3.0)
    solver = load_selfenergy_archive(str(archive))[3]
    assert solver.gf_settle is True and solver.gf_settle_ratio == pytest.approx(3.0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(gf_settle_ratio=-1.0), "must be >= 0"),
        (dict(gf_settle=False, gf_settle_ratio=2.0), "only applies with the support settle on"),
        (dict(gf_method="bicgstab", gf_settle=True), "applies only to gf_method='lanczos'"),
        (dict(gf_method="bicgstab", gf_settle_ratio=2.0), "applies only to gf_method='lanczos'"),
    ],
)
def test_the_settle_options_refuse_what_would_be_silently_ignored(kwargs, match):
    from impurityModel.ed.model import SolverOptions

    with pytest.raises(ValueError, match=match):
        SolverOptions(**kwargs)


def test_the_settle_options_accept_closure_and_the_defaults():
    from impurityModel.ed.model import SolverOptions

    assert SolverOptions().gf_settle is None and SolverOptions().gf_settle_ratio is None
    assert SolverOptions(gf_settle_ratio=0.0).gf_settle_ratio == 0.0
    assert SolverOptions(gf_settle=False).gf_settle is False


def test_the_settle_retry_option_replays_and_refuses_contradictions(tmp_path):
    from impurityModel.ed.model import SolverOptions

    _archive_with_attrs(tmp_path / "a.h5", gf_settle_retry=False)
    assert load_selfenergy_archive(str(tmp_path / "a.h5"))[3].gf_settle_retry is False
    _write_synthetic_archive(str(tmp_path / "b.h5"))
    assert load_selfenergy_archive(str(tmp_path / "b.h5"))[3].gf_settle_retry is None
    with pytest.raises(ValueError, match="contradicts gf_settle off"):
        SolverOptions(gf_settle=False, gf_settle_retry=True)
    with pytest.raises(ValueError, match="applies only to gf_method='lanczos'"):
        SolverOptions(gf_method="bicgstab", gf_settle_retry=True)
    assert SolverOptions(gf_settle=False, gf_settle_retry=False).gf_settle_retry is False


def test_a_settle_ratio_below_one_warns_that_it_is_a_multiple():
    from impurityModel.ed.model import SolverOptions

    with pytest.warns(UserWarning, match="MULTIPLE of slater_min"):
        SolverOptions(gf_settle_ratio=1e-4)
    with pytest.raises(ValueError):
        SolverOptions(gf_settle_ratio=float("inf"))


def test_archive_records_and_replays_the_pruned_growth(tmp_path):
    archive = tmp_path / "impurityModel_data.h5"
    _archive_with_attrs(archive, gf_prune_tol=1e-4, gf_first_shell_tol=1e-6)
    solver = load_selfenergy_archive(str(archive))[3]
    assert solver.gf_prune_tol == pytest.approx(1e-4) and solver.gf_first_shell_tol == pytest.approx(1e-6)
    archive = tmp_path / "old.h5"
    _write_synthetic_archive(str(archive))
    solver = load_selfenergy_archive(str(archive))[3]
    assert solver.gf_prune_tol is None and solver.gf_first_shell_tol is None


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(gf_prune_tol=-1e-3), "non-negative"),
        (dict(gf_prune_tol=0.0, gf_first_shell_tol=1e-3), "pruning is switched off"),
        (dict(gf_prune_tol=1e-3, gf_first_shell_tol=-1.0), "non-negative"),
        (dict(gf_prune_tol=1e-3, gf_settle=True), "exclusive with the support settle"),
        (dict(gf_prune_tol=1e-3, gf_settle_ratio=2.0), "exclusive with the support settle"),
        (dict(gf_prune_tol=1e-3, gf_settle_retry=True), "exclusive with the support settle"),
        (dict(gf_method="bicgstab", gf_prune_tol=1e-3), "applies only to gf_method='lanczos'"),
    ],
)
def test_the_prune_options_refuse_what_would_be_silently_ignored(kwargs, match):
    from impurityModel.ed.model import SolverOptions

    with pytest.raises(ValueError, match=match):
        SolverOptions(**kwargs)


def test_the_prune_options_accept_the_defaults_a_relaxed_shell_and_an_explicit_settle_off():
    from impurityModel.ed.model import SolverOptions

    assert SolverOptions().gf_prune_tol is None and SolverOptions().gf_first_shell_tol is None
    assert SolverOptions(gf_first_shell_tol=1e-6).resolved_gf_prune_tol == DEFAULT_GF_PRUNE_TOL
    assert SolverOptions(gf_prune_tol=1e-4, gf_first_shell_tol=0.0).gf_first_shell_tol == 0.0
    assert SolverOptions(gf_prune_tol=1e-4, gf_settle=False).gf_settle is False


def test_the_cost_budget_must_be_positive_and_defaults_to_off():
    assert BasisOptions(nominal_occ={0: 8}).cost_budget is None
    assert BasisOptions(nominal_occ={0: 8}, cost_budget=2.0).cost_budget == 2.0
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="cost_budget"):
            BasisOptions(nominal_occ={0: 8}, cost_budget=bad)


def test_archive_records_and_replays_the_swap_refinement(tmp_path):
    archive = tmp_path / "impurityModel_data.h5"
    _archive_with_attrs(archive, gf_refine_rounds=3, gf_refine_tol=1e-4, gf_refine_swap=0.2)
    solver = load_selfenergy_archive(str(archive))[3]
    assert (solver.gf_refine_rounds, solver.gf_refine_tol, solver.gf_refine_swap) == (3, pytest.approx(1e-4), 0.2)
    archive = tmp_path / "old.h5"
    _write_synthetic_archive(str(archive))
    solver = load_selfenergy_archive(str(archive))[3]
    assert (solver.gf_refine_rounds, solver.gf_refine_tol, solver.gf_refine_swap) == (None, None, None)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(gf_refine_rounds=-1), "non-negative integer"),
        (dict(gf_refine_rounds=1.5), "non-negative integer"),
        (dict(gf_refine_rounds=0, gf_refine_tol=1e-3), "need gf_refine_rounds"),
        (dict(gf_refine_rounds=0, gf_refine_swap=0.1), "need gf_refine_rounds"),
        (dict(gf_refine_rounds=2, gf_refine_tol=1.0), "in \\(0, 1\\)"),
        (dict(gf_refine_rounds=2, gf_refine_swap=0.6), "in \\(0, 0.5\\]"),
        (dict(gf_refine_rounds=2, gf_refine_swap=0.0), "in \\(0, 0.5\\]"),
        (dict(gf_refine_rounds=2, gf_method="bicgstab"), "only to gf_method='lanczos'"),
    ],
)
def test_the_refine_options_refuse_a_contradiction_by_name(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SolverOptions(**kwargs)


def test_the_refine_options_default_to_on_for_lanczos_and_accept_a_valid_set():
    assert SolverOptions().gf_refine_rounds is None
    assert SolverOptions().resolved_gf_refine_rounds == DEFAULT_GF_REFINE_ROUNDS > 0
    assert SolverOptions(gf_method="bicgstab").resolved_gf_refine_rounds == 0
    assert SolverOptions(gf_refine_rounds=0).gf_refine_rounds == 0
    ok = SolverOptions(gf_refine_rounds=4, gf_refine_tol=1e-4, gf_refine_swap=0.5)
    assert (ok.gf_refine_rounds, ok.gf_refine_tol, ok.gf_refine_swap) == (4, 1e-4, 0.5)


def test_an_unset_pruning_is_the_default_unless_a_settle_option_asks_for_the_settle():
    assert SolverOptions().resolved_gf_prune_tol == DEFAULT_GF_PRUNE_TOL
    assert SolverOptions(gf_prune_tol=0.0).resolved_gf_prune_tol == 0.0
    assert SolverOptions(gf_prune_tol=2e-4).resolved_gf_prune_tol == 2e-4
    # The settle is what pruning replaces: asking for it explicitly wins over the default.
    for kwargs in (dict(gf_settle=True), dict(gf_settle_ratio=2.0), dict(gf_settle_retry=True)):
        assert SolverOptions(**kwargs).resolved_gf_prune_tol == 0.0, kwargs
    assert SolverOptions(gf_settle=False).resolved_gf_prune_tol == DEFAULT_GF_PRUNE_TOL
    # Leaving the block-Lanczos kernel: nothing to prune, and the explicit 0 is still accepted.
    assert SolverOptions(gf_method="bicgstab").resolved_gf_prune_tol is None
    assert SolverOptions(gf_method="bicgstab", gf_prune_tol=0.0).resolved_gf_prune_tol == 0.0
    assert SolverOptions(gf_refine_rounds=0).resolved_gf_refine_rounds == 0
    assert SolverOptions(gf_refine_rounds=5).resolved_gf_refine_rounds == 5


def test_the_growth_tolerance_is_recorded_replayed_and_absent_means_off(tmp_path):
    archive = tmp_path / "impurityModel_data.h5"
    _archive_with_attrs(archive, gf_prune_tol=1e-4, gf_growth_tol=1e-5)
    assert load_selfenergy_archive(str(archive))[3].gf_growth_tol == pytest.approx(1e-5)
    archive = tmp_path / "old.h5"
    _write_synthetic_archive(str(archive))
    assert load_selfenergy_archive(str(archive))[3].gf_growth_tol is None


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(gf_prune_tol=1e-3, gf_growth_tol=0.0), "must lie in"),
        (dict(gf_prune_tol=1e-3, gf_growth_tol=1.5), "must lie in"),
        (dict(gf_prune_tol=0.0, gf_growth_tol=1e-5), "pruning is switched off"),
        (dict(gf_method="bicgstab", gf_growth_tol=1e-5), "applies only to gf_method='lanczos'"),
    ],
)
def test_the_growth_tolerance_refuses_what_would_be_silently_ignored(kwargs, match):
    from impurityModel.ed.model import SolverOptions

    with pytest.raises(ValueError, match=match):
        SolverOptions(**kwargs)


def test_the_growth_tolerance_defaults_to_off():
    from impurityModel.ed.model import SolverOptions

    assert SolverOptions().gf_growth_tol is None
    assert SolverOptions(gf_growth_tol=1e-5).gf_growth_tol == 1e-5


def test_an_archive_group_without_the_current_version_is_refused(tmp_path):
    import h5py

    archive = tmp_path / "impurityModel_data.h5"
    _write_synthetic_archive(str(archive))
    with h5py.File(archive, "r+") as f:
        del f["X 2"].attrs["archive version"]
    with pytest.raises(ValueError, match="archive version"):
        load_selfenergy_archive(str(archive))
    with pytest.raises(ValueError, match="archive version"):
        ImpurityModel.from_hdf5(str(archive))


def test_the_options_written_to_an_archive_are_the_options_read_back(tmp_path):
    archive = tmp_path / "impurityModel_data.h5"
    basis_kwargs = {"dN": 3, "chain_restrict": False, "mixed_valence": "auto", "cost_budget": 2.0}
    solver_kwargs = {"gf_settle": False, "gf_tol": 1e-7, "gf_refine_rounds": 0}
    _write_synthetic_archive(str(archive), basis_kwargs=basis_kwargs, solver_kwargs=solver_kwargs)
    _, _, basis, solver, _ = load_selfenergy_archive(str(archive))
    assert (basis.dN, basis.chain_restrict, basis.mixed_valence, basis.cost_budget) == (3, False, "auto", 2.0)
    assert (solver.gf_settle, solver.gf_tol, solver.gf_refine_rounds) == (False, 1e-7, 0)


def test_the_quiet_window_is_recorded_and_replayed(tmp_path):
    archive = tmp_path / "impurityModel_data.h5"
    _archive_with_attrs(archive, gf_prune_tol=1e-4, gf_growth_quiet=32)
    assert load_selfenergy_archive(str(archive))[3].gf_growth_quiet == 32
    archive = tmp_path / "old.h5"
    _write_synthetic_archive(str(archive))
    assert load_selfenergy_archive(str(archive))[3].gf_growth_quiet is None


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(gf_prune_tol=1e-3, gf_growth_quiet=-1), "non-negative integer"),
        (dict(gf_prune_tol=1e-3, gf_growth_quiet=2.5), "non-negative integer"),
        (dict(gf_prune_tol=0.0, gf_growth_quiet=8), "pruning is switched off"),
        (dict(gf_method="bicgstab", gf_growth_quiet=8), "applies only to gf_method='lanczos'"),
    ],
)
def test_the_quiet_window_refuses_what_would_be_silently_ignored(kwargs, match):
    from impurityModel.ed.model import SolverOptions

    with pytest.raises(ValueError, match=match):
        SolverOptions(**kwargs)


def test_the_quiet_window_resolves_against_pruning_and_its_default():
    from impurityModel.ed import options
    from impurityModel.ed.model import SolverOptions

    assert options.DEFAULT_GF_GROWTH_QUIET == 64
    assert SolverOptions().resolved_gf_growth_quiet == 64
    assert SolverOptions(gf_growth_quiet=8).resolved_gf_growth_quiet == 8
    assert SolverOptions(gf_growth_quiet=0).resolved_gf_growth_quiet is None
    assert SolverOptions(gf_prune_tol=0.0).resolved_gf_growth_quiet is None
    assert SolverOptions(gf_method="bicgstab").resolved_gf_growth_quiet is None
    assert SolverOptions(gf_settle=True).resolved_gf_growth_quiet is None  # the settle replaces pruning


def test_the_prune_rule_is_recorded_and_replayed(tmp_path):
    archive = tmp_path / "impurityModel_data.h5"
    _archive_with_attrs(archive, gf_prune_tol=1e-4, gf_prune_rule="accumulated")
    assert load_selfenergy_archive(str(archive))[3].gf_prune_rule == "accumulated"
    archive = tmp_path / "old.h5"
    _write_synthetic_archive(str(archive))
    assert load_selfenergy_archive(str(archive))[3].gf_prune_rule is None


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(gf_prune_tol=1e-3, gf_prune_rule="mass"), "expected one of"),
        (dict(gf_prune_tol=0.0, gf_prune_rule="accumulated"), "pruning is switched off"),
        (dict(gf_method="bicgstab", gf_prune_rule="accumulated"), "applies only to gf_method='lanczos'"),
    ],
)
def test_the_prune_rule_refuses_what_would_be_silently_ignored(kwargs, match):
    from impurityModel.ed.model import SolverOptions

    with pytest.raises(ValueError, match=match):
        SolverOptions(**kwargs)


def test_the_prune_rule_resolves_to_its_public_name():
    from impurityModel.ed.model import SolverOptions

    assert SolverOptions().resolved_gf_prune_rule is None  # unset: the kernel default (and the rig switch) apply
    assert SolverOptions(gf_prune_rule="amplitude").resolved_gf_prune_rule == "amplitude"
    assert SolverOptions(gf_prune_rule="accumulated").resolved_gf_prune_rule == "accumulated"
    assert SolverOptions(gf_prune_tol=0.0).resolved_gf_prune_rule is None
    assert SolverOptions(gf_method="bicgstab").resolved_gf_prune_rule is None
