"""The shell model of a multi-shell rotation (``observables.resolve_shell_ls`` and friends)."""

import numpy as np
import pytest

from impurityModel.ed.observables import (
    resolve_shell_ls,
    shell_column_slices,
    shell_index_of_orbitals,
    shell_is_complete,
)


def test_one_shell_is_inferred_from_the_column_count():
    assert resolve_shell_ls(np.eye(10)) == (2,)
    assert resolve_shell_ls(np.eye(6)) == (1,)
    assert resolve_shell_ls(np.zeros((6, 10))) == (2,)  # t2g rows of a d shell


def test_an_unrecognisable_column_count_needs_shell_ls():
    with pytest.raises(ValueError, match="shell_ls"):
        resolve_shell_ls(np.eye(16))


def test_given_shell_ls_must_span_the_columns_and_may_repeat():
    assert resolve_shell_ls(np.eye(16), (1, 2)) == (1, 2)
    assert resolve_shell_ls(np.eye(20), (2, 2)) == (2, 2)  # two d sets are never merged
    with pytest.raises(ValueError, match="spans"):
        resolve_shell_ls(np.eye(16), (2, 2))


def test_column_blocks_follow_the_shell_order():
    assert shell_column_slices((1, 2)) == [slice(0, 6), slice(6, 16)]


def test_orbitals_belong_to_the_shell_with_their_weight():
    R = np.eye(16)
    assert shell_index_of_orbitals(R, (1, 2)).tolist() == [0] * 6 + [1] * 10
    # t2g d rows plus the full p: 12 orbitals, 16 columns
    R = np.zeros((12, 16), dtype=complex)
    R[np.arange(6), np.arange(6)] = 1
    R[np.arange(6, 12), np.arange(6, 12)] = 1
    assert shell_index_of_orbitals(R, (1, 2)).tolist() == [0] * 6 + [1] * 6


def test_an_orbital_straddling_two_shells_raises():
    R = np.eye(16, dtype=complex)
    R[0, 0], R[0, 6] = np.sqrt(0.5), np.sqrt(0.5)
    with pytest.raises(ValueError, match="straddle"):
        shell_index_of_orbitals(R, (1, 2))
    R[0, 0], R[0, 6] = np.sqrt(1 - 1e-8), np.sqrt(1e-8)  # below the tolerance: still shell 0
    assert shell_index_of_orbitals(R, (1, 2))[0] == 0


def test_a_partial_shell_is_not_complete():
    assert shell_is_complete(np.eye(6))
    assert not shell_is_complete(np.eye(10)[:, :6].T)  # 6 rows of 10 columns
    assert not shell_is_complete(np.diag([1.0, 1.0, 0.5]))


def _dict_and_array_two_shell():
    from impurityModel.test.symmetry.test_symmetry_observables import _two_shell_fixture

    impurity_orbitals, rot_dict, rho_imp, _ = _two_shell_fixture()
    return impurity_orbitals, rot_dict, np.eye(16, dtype=complex), rho_imp


def test_the_ndarray_twin_of_the_two_shell_fixture_gives_the_same_shell_rhos():
    from impurityModel.ed.observables import compute_shell_observables, impurity_shell_rhos

    impurity_orbitals, rot_dict, rot_array, rho_imp = _dict_and_array_two_shell()
    want = list(impurity_shell_rhos(rho_imp, rot_dict, impurity_orbitals))
    got = list(impurity_shell_rhos(rho_imp, rot_array, shell_ls=(1, 2)))
    assert [g[0] for g in got] == [w[0] for w in want] == [1, 2]
    for (_, _, a), (_, _, b) in zip(got, want):
        np.testing.assert_allclose(a, b, atol=1e-14)
    obs_dict = compute_shell_observables(rho_imp, rot_dict, impurity_orbitals)
    obs_array = compute_shell_observables(rho_imp, rot_array, shell_ls=(1, 2))
    for sd, sa in zip(obs_dict["shells"], obs_array["shells"]):
        for key in ("n", "lz", "sz", "m_z", "j_z", "l_dot_s", "t_z"):
            assert sa[key] == pytest.approx(sd[key], abs=1e-12)


def test_shell_rhos_are_unchanged_by_a_block_diagonal_rotation_of_the_input_basis():
    from scipy.stats import unitary_group

    from impurityModel.ed.observables import impurity_shell_rhos

    _, _, rot_array, rho_imp = _dict_and_array_two_shell()
    u = np.zeros((16, 16), dtype=complex)
    u[:6, :6] = unitary_group.rvs(6, random_state=1)
    u[6:, 6:] = unitary_group.rvs(10, random_state=2)
    # input basis rotated: rho -> u^dag rho u, R -> u^dag R keeps R^dag rho R fixed
    want = [r for _, _, r in impurity_shell_rhos(rho_imp, rot_array, shell_ls=(1, 2))]
    got = [r for _, _, r in impurity_shell_rhos(u.conj().T @ rho_imp @ u, u.conj().T @ rot_array, shell_ls=(1, 2))]
    for a, b in zip(got, want):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_a_partial_second_shell_gives_a_rank_limited_block():
    from impurityModel.ed.observables import impurity_shell_rhos

    R = np.zeros((12, 16), dtype=complex)
    R[np.arange(12), np.arange(12)] = 1  # full p (0..5) + six d columns
    rho = np.eye(12, dtype=complex)
    (l0, _, r0), (l1, _, r1) = impurity_shell_rhos(rho, R, shell_ls=(1, 2))
    assert (l0, l1) == (1, 2) and r0.shape == (6, 6) and r1.shape == (10, 10)
    assert np.linalg.matrix_rank(r1) == 6 and np.trace(r1).real == pytest.approx(6.0)


def test_multishell_casimir_ops_match_the_dict_path_shell_by_shell():
    from impurityModel.ed.observables import make_impurity_casimir_operators

    impurity_orbitals, rot_dict, rot_array, _ = _dict_and_array_two_shell()
    want = make_impurity_casimir_operators(impurity_orbitals, rot_dict, per_shell=True)
    got = make_impurity_casimir_operators(impurity_orbitals, rot_array, per_shell=True, shell_ls=(1, 2))
    for k in range(3):  # totals: L, S, J
        for g, w in zip(got[k], want[k]):
            assert g.approx_equal(w, 1e-12)
    assert sorted(l for l, *_ in got[3].values()) == sorted(l for l, *_ in want[3].values()) == [1, 2]


def test_a_partial_shell_gives_per_shell_ops_but_no_totals():
    from impurityModel.ed.observables import make_impurity_casimir_operators

    _, _, rot_array, _ = _dict_and_array_two_shell()
    kept = {0: [list(range(6)) + list(range(6, 14))]}
    rot = rot_array[:14, :]  # drop two rows of the d shell: its R_a is rectangular
    with pytest.raises(ValueError):
        make_impurity_casimir_operators(kept, rot, shell_ls=(1, 2))
    l, s, j, per = make_impurity_casimir_operators(kept, rot, per_shell=True, shell_ls=(1, 2))
    assert l is s is j is None
    assert set(per) == {0}  # the complete p shell only; the partial d shell has no operator


def test_model_shell_ls_defaults_to_one_inferred_shell_and_survives_the_constructors():
    from impurityModel.ed.model import ImpurityModel

    h = np.zeros((20, 20))
    one = ImpurityModel.from_solver_matrix(h, 10, rot_to_spherical=np.eye(10))
    assert one.shell_ls is None and one.resolved_shell_ls == (2,)
    two = ImpurityModel.from_solver_matrix(h, 16, rot_to_spherical=np.eye(16), shell_ls=[1, 2])
    assert two.shell_ls == (1, 2) and two.resolved_shell_ls == (1, 2)
    blocks = ImpurityModel.from_blocks(
        np.zeros((16, 16)), np.zeros((4, 16)), np.zeros((4, 4)), rot_to_spherical=np.eye(16), shell_ls=(1, 2)
    )
    assert blocks.resolved_shell_ls == (1, 2)
    assert ImpurityModel.from_solver_matrix(h, 10, rot_to_spherical={2: np.eye(10)}).resolved_shell_ls is None


def test_archive_reads_per_shell_l_and_occupation_when_present_and_not_otherwise(tmp_path):
    h5py = pytest.importorskip("h5py")
    from impurityModel.ed.model import _read_archive_group

    def write(path, multi):
        with h5py.File(path, "w") as f:
            f.attrs["last iteration"] = 1
            g = f.create_group("X 1")
            n = 16 if multi else 10
            g["H solver"] = np.zeros((n + 2, n + 2))
            g["U"] = np.zeros((n,) * 4)
            g["Rot to spherical"] = np.eye(n)
            g["Impurity orbitals"] = np.arange(n)
            g["Matsubara frequency mesh"] = np.zeros(0)
            g["Real frequency mesh"] = np.zeros(0)
            g.attrs.update({"nominal occupation": 8, "tau": 0.1, "delta": 0.1, "mv": 0})
            if multi:
                g["Spherical shell l"] = np.array([1, 2])
                g["nominal occupation per shell"] = np.array([6, 8])

    write(tmp_path / "a.h5", False)
    write(tmp_path / "b.h5", True)
    single = _read_archive_group(tmp_path / "a.h5", with_options=False)
    assert single["shell_ls"] is None and single["nominal_occ"] == {0: 8}
    multi = _read_archive_group(tmp_path / "b.h5", with_options=False)
    assert multi["shell_ls"] == (1, 2) and multi["nominal_occ"] == {0: 6, 1: 8}


def test_whole_shell_spin_pairs_cover_every_shell_of_a_multishell_rotation():
    from impurityModel.ed.spin_pairs import _impurity_candidates, _impurity_pairs_whole_shell

    imp = {0: [list(range(16))]}
    rot = np.eye(16, dtype=complex)
    want = [(0, 3), (1, 4), (2, 5)] + [(6 + i, 11 + i) for i in range(5)]
    assert _impurity_pairs_whole_shell(imp, rot) is None  # without shell_ls: not one l-shell
    assert sorted(_impurity_pairs_whole_shell(imp, rot, shell_ls=(1, 2))) == want
    candidates, _ = _impurity_candidates(imp, rot, 1e-6, shell_ls=(1, 2))
    assert sorted(candidates[0][0]) == want
    # a partial d shell (two rows dropped) is no clean pairing
    assert _impurity_pairs_whole_shell({0: [list(range(14))]}, rot[:14], shell_ls=(1, 2)) is None


def _p_plus_d_model(couple_p=False):
    """16 impurity spin-orbitals (p: 0-5, d: 6-15, each [dn, up]) and 4 bath orbitals hybridised with the d shell."""
    from impurityModel.ed.lie_algebra import tensors_to_operator

    n = 20
    h = np.zeros((n, n), dtype=complex)
    for o in range(6):
        h[o, o] = 5.0 + 0.01 * o  # p above d: energetic filling alone would leave it empty
    for i, o in enumerate(range(6, 16)):
        h[o, o] = 1.0 + 0.1 * (i % 5)
    for b in range(4):
        h[16 + b, 16 + b] = -1.0 if b < 2 else 2.0
    for o, b in ((6, 16), (11, 17), (7, 18), (12, 19)):
        h[o, b] = h[b, o] = 0.3
    if couple_p:
        h[0, 16] = h[16, 0] = 0.3
    return tensors_to_operator(h).to_dict(), np.eye(16, dtype=complex)


def test_prepare_solver_basis_multishell_records_shells_bath_and_per_shell_occupation():
    from impurityModel.ed.solver_basis import prepare_solver_basis

    h0, rot = _p_plus_d_model()
    sb = prepare_solver_basis(
        h0, {}, {}, {0: list(range(16))}, {0: 2, 1: 8}, rot_to_spherical=rot, verbosity=0, shell_ls=(1, 2)
    )
    assert sb.shell_ls == (1, 2)
    assert {sb.impurity_shell_index[o] for o in range(6)} == {0}
    assert {sb.impurity_shell_index[o] for o in range(6, 16)} == {1}
    assert sb.shell_has_bath == {0: False, 1: True}
    assert sb.active_impurity_indices == list(range(6, 16))
    per_shell = dict.fromkeys((0, 1), 0)
    for k, n_k in sb.nominal_occ.items():
        orb = next(o for block in sb.impurity_orbitals[k] for o in block)
        per_shell[sb.impurity_shell_index[orb]] += n_k
    assert per_shell == {0: 2, 1: 8}


def test_prepare_solver_basis_single_shell_leaves_the_shell_fields_empty():
    from impurityModel.ed.lie_algebra import tensors_to_operator
    from impurityModel.ed.solver_basis import prepare_solver_basis

    h = np.zeros((12, 12), dtype=complex)
    for i in range(10):
        h[i, i] = 1.0 + 0.1 * (i % 5)
    h[10, 10], h[11, 11] = -1.0, 2.0
    h[0, 10] = h[10, 0] = h[5, 11] = h[11, 5] = 0.3
    sb = prepare_solver_basis(
        tensors_to_operator(h).to_dict(), {}, {}, {0: list(range(10))}, 8, rot_to_spherical=np.eye(10), verbosity=0
    )
    assert sb.shell_ls is None and sb.impurity_shell_index is None
    assert sb.shell_has_bath is None and sb.active_impurity_indices is None


def test_a_bath_component_hopping_to_two_shells_is_refused():
    from impurityModel.ed.solver_basis import prepare_solver_basis

    h0, rot = _p_plus_d_model(couple_p=True)
    with pytest.raises(ValueError, match=r"spans shells|straddle"):
        prepare_solver_basis(
            h0, {}, {}, {0: list(range(16))}, {0: 2, 1: 8}, rot_to_spherical=rot, verbosity=0, shell_ls=(1, 2)
        )


def test_a_skipped_bath_less_shell_near_the_fermi_level_is_refused_by_the_dc_search():
    from impurityModel.ed.dc_criteria import DC_INACTIVE_SHELL_EF_WINDOW, _check_active_shells
    from impurityModel.ed.solver_basis import prepare_solver_basis

    h0, rot = _p_plus_d_model()  # p at +5: far from E_F, a core-like shell
    sb = prepare_solver_basis(
        h0, {}, {}, {0: list(range(16))}, {0: 2, 1: 8}, rot_to_spherical=rot, verbosity=0, shell_ls=(1, 2)
    )
    _check_active_shells(sb, 0, 0)
    assert DC_INACTIVE_SHELL_EF_WINDOW < 5.0
    near = dict(h0)
    for o in range(6):
        near[((o, "c"), (o, "a"))] = 0.2 * (o % 2) + 0.1
    sb = prepare_solver_basis(
        near, {}, {}, {0: list(range(16))}, {0: 2, 1: 8}, rot_to_spherical=rot, verbosity=0, shell_ls=(1, 2)
    )
    with pytest.raises(ValueError, match="valence shell"):
        _check_active_shells(sb, 0, 0)


def _solve_p_plus_d(tmp_path, shell_ls):
    from impurityModel.ed.groundstate import calc_gs
    from impurityModel.ed.solver_basis import prepare_solver_basis

    h0, rot = _p_plus_d_model()
    sb = prepare_solver_basis(
        h0, {}, {}, {0: list(range(16))}, {0: 2, 1: 8}, rot_to_spherical=rot, verbosity=0, shell_ls=shell_ls
    )
    basis_information = {
        "impurity_orbitals": sb.impurity_orbitals,
        "bath_states": sb.bath_states,
        "N0": sb.nominal_occ,
        "tau": 0.002,
        "chain_restrict": False,
        "dense_cutoff": 1000,
        "comm": None,
        "spin_layout": sb.spin_layout,
        "truncation_threshold": np.inf,
    }
    return calc_gs(
        sb.h,
        basis_information,
        sb.block_structure,
        sb.rot_to_spherical,
        2,
        stats_path=str(tmp_path / "gs.json"),
        shell_ls=sb.shell_ls,
    )


def test_calc_gs_reports_a_two_shell_impurity_shell_by_shell(tmp_path, capsys):
    _solve_p_plus_d(tmp_path, (1, 2))
    out = capsys.readouterr().out
    assert "Per-shell impurity observables" in out
    assert "not reported" not in out and "skipped" not in out
    rows = out.split("Per-shell impurity observables")[1].splitlines()
    assert [r.split()[0] for r in rows[2:4]] == ["l=1", "l=2"]  # one row per shell, p then d


def test_multishell_casimir_ops_follow_the_sorted_orbital_rows_not_the_group_dict_order():
    """The groups ``prepare_solver_basis`` derives list orbitals out of index order; the rotation rows do not."""
    from impurityModel.ed.observables import make_impurity_casimir_operators
    from impurityModel.ed.solver_basis import prepare_solver_basis

    h0, rot = _p_plus_d_model()
    sb = prepare_solver_basis(
        h0, {}, {}, {0: list(range(16))}, {0: 2, 1: 8}, rot_to_spherical=rot, verbosity=0, shell_ls=(1, 2)
    )
    flat = [o for blocks in sb.impurity_orbitals.values() for block in blocks for o in block]
    assert flat != sorted(flat)  # the premise: the dict order is not the row order
    got = make_impurity_casimir_operators(sb.impurity_orbitals, sb.rot_to_spherical, shell_ls=sb.shell_ls)
    want = make_impurity_casimir_operators({0: [sorted(flat)]}, sb.rot_to_spherical, shell_ls=sb.shell_ls)
    for k in range(3):
        for g, w in zip(got[k], want[k]):
            assert g.approx_equal(w, 1e-12)


def test_a_shell_keyed_nominal_occupation_is_not_read_as_group_keyed_when_the_keys_coincide():
    """Two shells of one group each: group keys == shell keys == {0, 1}, but group 0 is the *d* shell (shell 1)."""
    from impurityModel.ed.solver_basis import _per_group_occupation

    impurity_orbitals = {0: [list(range(10))], 1: [list(range(10, 16))]}  # group 0: d, group 1: p
    shell_index = {**dict.fromkeys(range(10), 1), **dict.fromkeys(range(10, 16), 0)}  # p is shell 0, d is shell 1
    got = _per_group_occupation({0: 2, 1: 8}, impurity_orbitals, impurity_shell_index=shell_index)
    assert got == {0: 8, 1: 2}


def _rotating_d_model():
    """A d shell with a crystal-field mixing of orbitals 0 and 1: the symmetry-adapted rotation is not the identity."""
    from impurityModel.ed.lie_algebra import tensors_to_operator

    h = np.zeros((12, 12), dtype=complex)
    for i in range(10):
        h[i, i] = 1.0 + 0.1 * (i % 5)
    h[0, 1] = h[1, 0] = 0.05
    h[5, 6] = h[6, 5] = 0.05
    h[10, 10], h[11, 11] = -1.0, 2.0
    h[0, 10] = h[10, 0] = h[5, 11] = h[11, 5] = 0.3
    return tensors_to_operator(h).to_dict()


@pytest.mark.mpi
@pytest.mark.skipif(__import__("mpi4py.MPI").MPI.COMM_WORLD.size == 1, reason="needs a second rank to disagree")
def test_every_rank_solves_in_rank_zeros_rotation(monkeypatch):
    """H-MS7: the rotation is a rank-local eigh; a rank whose eigh disagrees must still take rank 0's basis."""
    from mpi4py import MPI

    from impurityModel.ed import solver_basis

    comm = MPI.COMM_WORLD
    if comm.rank != 0:
        # Stand-in for a degenerate block whose eigenvectors came out differently on this rank's BLAS.
        monkeypatch.setattr(
            solver_basis,
            "impurity_symmetry_rotation",
            lambda op, imp, n_orb=None, h0_matrix=None: (np.eye(n_orb, dtype=complex), np.eye(len(imp), dtype=complex)),
        )
    sb = solver_basis.prepare_solver_basis(
        _rotating_d_model(), {}, {}, {0: list(range(10))}, 8, rot_to_spherical=np.eye(10), verbosity=0, comm=comm
    )
    root = comm.bcast(sb.rotation_full, root=0)
    assert not np.allclose(root, np.eye(root.shape[0])), "the model must rotate, or the test is vacuous"
    assert np.array_equal(sb.rotation_full, root)
    assert np.array_equal(sb.u_imp, comm.bcast(sb.u_imp, root=0))
    assert sb.block_structure == comm.bcast(sb.block_structure, root=0)
