"""``LocalOperators``: the spherical rebuild, rotations, the dipole forms and the supplied-``S_+`` pairing."""

import numpy as np
import pytest

from impurityModel.ed.local_operators import LocalOperators, dipole_from_circular, single_particle_lsj_matrices
from impurityModel.ed.model import ImpurityModel
from impurityModel.ed.spin_pairs import _impurity_candidates


def _random_unitary(n, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    return q


@pytest.mark.parametrize("l", [1, 2, 3])
def test_from_spherical_identity_is_the_single_shell_matrices(l):
    n = 2 * (2 * l + 1)
    ops = LocalOperators.from_spherical(np.eye(n), (l,))
    lz, lp, _, sz, sp, _ = single_particle_lsj_matrices(l)
    for got, want in ((ops.lz, lz), (ops.l_plus, lp), (ops.sz, sz), (ops.s_plus, sp)):
        np.testing.assert_allclose(got, want, atol=1e-14)
    assert ops.source == "spherical"


def test_from_spherical_partial_shell_is_the_projected_operator():
    l = 2
    R = _random_unitary(10, 1)[:6, :]  # six input orbitals of the d shell (rows kept)
    ops = LocalOperators.from_spherical(R, (l,))
    _, _, _, sz, _, _ = single_particle_lsj_matrices(l)
    np.testing.assert_allclose(ops.sz, R @ sz @ R.conj().T, atol=1e-13)
    assert ops.sz.shape == (6, 6)


def test_two_shells_are_a_direct_sum():
    R = np.eye(16)
    ops = LocalOperators.from_spherical(R, (1, 2))
    np.testing.assert_allclose(ops.sz[:6, :6], single_particle_lsj_matrices(1)[3], atol=1e-14)
    np.testing.assert_allclose(ops.sz[6:, 6:], single_particle_lsj_matrices(2)[3], atol=1e-14)
    assert np.abs(ops.sz[:6, 6:]).max() == 0


def test_rotated_matches_a_change_of_basis():
    ops = LocalOperators.from_spherical(np.eye(10), (2,))
    Q = _random_unitary(10, 2)
    r = ops.rotated(Q)
    np.testing.assert_allclose(r.lz, Q.conj().T @ ops.lz @ Q, atol=1e-13)
    # rotating the spherical set equals building it with R = Q
    direct = LocalOperators.from_spherical(Q.conj().T, (2,))
    np.testing.assert_allclose(r.sz, direct.sz, atol=1e-13)


def test_spin_vector_commutation():
    ops = LocalOperators.from_spherical(np.eye(10), (2,))
    sx, sy, sz = ops.spin_vector()
    np.testing.assert_allclose(sx @ sy - sy @ sx, 1j * sz, atol=1e-13)
    lx, ly, lz = ops.orbital_vector()
    np.testing.assert_allclose(lx @ ly - ly @ lx, 1j * lz, atol=1e-13)


def test_site_rotation_rotates_the_spin_vector():
    ops = LocalOperators.from_spherical(np.eye(6), (1,))
    rot_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)  # 90 degrees about x
    g = LocalOperators.from_spherical(np.eye(6), (1,), site_rotation=rot_x)
    assert g.frame == "global"
    np.testing.assert_allclose(g.sz, ops.spin_vector()[1] * 0 + rot_x[2, 1] * ops.spin_vector()[1], atol=1e-13)
    sg = g.spin_vector()
    np.testing.assert_allclose(sg[0] @ sg[1] - sg[1] @ sg[0], 1j * sg[2], atol=1e-13)
    with pytest.raises(ValueError):
        LocalOperators.from_spherical(np.eye(6), (1,), site_rotation=np.diag([1.0, 1.0, -1.0]))


def test_dipole_circular_roundtrip():
    rng = np.random.default_rng(3)
    d = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))
    ops = LocalOperators(sz=np.zeros((4, 4)), dipole=d)
    dp, dm, dz = ops.dipole_circular()
    np.testing.assert_allclose(dipole_from_circular(dp, dm, dz), d, atol=1e-13)
    herm = d + d.conj().transpose(0, 2, 1)
    dp, dm, _ = LocalOperators(dipole=herm).dipole_circular()
    np.testing.assert_allclose(dm, dp.conj().T, atol=1e-13)  # Dm = Dp^dagger for a Hermitian dipole


def test_validation():
    with pytest.raises(ValueError):
        LocalOperators(sz=np.zeros((4, 4)), s_plus=np.zeros((3, 3)))
    with pytest.raises(ValueError):
        LocalOperators(sz=np.zeros((4, 4)), source="bogus")
    with pytest.raises(ValueError):
        LocalOperators(sz=np.zeros((4, 4)), dipole=np.zeros((2, 4, 4)))


def test_antiunitary_rotation():
    u = _random_unitary(4, 4)
    Q = _random_unitary(4, 5)
    ops = LocalOperators(sz=np.zeros((4, 4)), symmetry_ops=[(u, True)])
    (got, anti), = ops.rotated(Q).symmetry_ops
    assert anti
    np.testing.assert_allclose(got, Q.conj().T @ u @ Q.conj(), atol=1e-13)


def test_from_spin_ordering_orderings():
    for ordering in ("down_first", "up_first", "interleaved"):
        ops = LocalOperators.from_spin_ordering(6, ordering)
        assert ops.source == "index"
        np.testing.assert_allclose(np.diag(ops.sz).real.sum(), 0.0)
        np.testing.assert_allclose(ops.s_plus @ ops.s_plus, 0.0)
    assert LocalOperators.from_spin_ordering(4, "interleaved").s_plus[1, 0] == 1


def test_model_resolves_spherical_operators_and_prefers_supplied():
    n = 10
    H = np.zeros((n, n))
    m = ImpurityModel.from_solver_matrix(H, n, rot_to_spherical=np.eye(n))
    assert m.operators is None
    assert m.resolved_operators.source == "spherical"
    supplied = LocalOperators.from_spin_ordering(n, "interleaved")
    m2 = ImpurityModel.from_solver_matrix(H, n, rot_to_spherical=np.eye(n), operators=supplied)
    assert m2.resolved_operators is supplied


def test_supplied_s_plus_is_the_first_impurity_candidate():
    # Interleaved dn/up ordering: the index convention (dn-then-up) pairs wrongly, a supplied S_+ pairs right.
    imp = {0: [[0, 1, 2, 3]]}
    ops = LocalOperators.from_spin_ordering(4, "interleaved")
    candidates, _ = _impurity_candidates(imp, None, 1e-6, None, ops)
    assert candidates[0] == ([(0, 1), (2, 3)], "operators")
    assert candidates[1][1] == "index"
    # a spherical-source set merely repeats the rotation path and adds no candidate
    sph = LocalOperators.from_spherical(np.eye(6), (1,))
    cands, _ = _impurity_candidates({0: [list(range(6))]}, np.eye(6), 1e-6, None, sph)
    assert all(src != "operators" for _, src in cands)


def test_h0_header_roundtrip(tmp_path):
    from impurityModel.ed.h0_format import read_h0_file, write_h0_file

    ops = LocalOperators.from_spherical(np.eye(6), (1,))
    d = np.random.default_rng(0).normal(size=(3, 6, 6)) + 0j
    ops = LocalOperators(sz=ops.sz, s_plus=ops.s_plus, lz=ops.lz, l_plus=ops.l_plus, dipole=d,
                         symmetry_ops=[(np.eye(6), False), (_random_unitary(6, 9), True)], source="rspt", frame="global")
    path = tmp_path / "x.h0"
    write_h0_file(path, np.diag(np.arange(6.0)), impurity_orbitals={1: list(range(6))}, operators=ops, basis="spherical",
                  spin_ordering="down_first", rot_to_spherical=np.eye(6))
    got = read_h0_file(path).operators
    assert got.source == "rspt" and got.frame == "global"
    for name in ("sz", "s_plus", "lz", "l_plus", "dipole"):
        np.testing.assert_allclose(getattr(got, name), getattr(ops, name), atol=1e-14)
    assert [a for _, a in got.symmetry_ops] == [False, True]
    np.testing.assert_allclose(got.symmetry_ops[1][0], ops.symmetry_ops[1][0], atol=1e-14)
    # a file without the block reads None
    write_h0_file(tmp_path / "y.h0", np.eye(6), impurity_orbitals={1: list(range(6))})
    assert read_h0_file(tmp_path / "y.h0").operators is None


def test_a_transposed_s_plus_is_rejected_not_used_with_swapped_spin_labels():
    # S_+^T is S_-: a clean permutation with dn and up swapped. [S_z, S_+] = S_+ is what exposes it.
    ops = LocalOperators.from_spherical(np.eye(6), (1,))
    bad = LocalOperators(sz=ops.sz, s_plus=ops.s_plus.T.copy(), source="rspt")
    imp = {0: [list(range(6))]}
    candidates, rejected = _impurity_candidates(imp, np.eye(6), 1e-6, (1,), bad)
    assert all(src != "operators" for _, src in candidates)
    assert any("[S_z, S_+]" in r for r in rejected)
    good = LocalOperators(sz=ops.sz, s_plus=ops.s_plus, source="rspt")
    candidates, _ = _impurity_candidates(imp, np.eye(6), 1e-6, (1,), good)
    assert candidates[0][1] == "operators"


def test_operators_of_the_wrong_size_are_refused(tmp_path):
    from impurityModel.ed.h0_format import H0FormatError, read_h0_file, write_h0_file

    small = LocalOperators.from_spherical(np.eye(2), (0,))
    path = tmp_path / "x.h0"
    write_h0_file(path, np.eye(6), impurity_orbitals={1: list(range(6))}, operators=small)
    with pytest.raises(H0FormatError, match="orbitals"):
        read_h0_file(path)
    supplied = LocalOperators(sz=small.sz, s_plus=small.s_plus, source="rspt")
    cands, rejected = _impurity_candidates({0: [list(range(6))]}, None, 1e-6, None, supplied)
    assert any("impurity has 6 orbitals" in r for r in rejected)


@pytest.mark.parametrize("block", [[1, 2], {"symmetry_ops": [[1]]}, {"sz": 3}])
def test_a_malformed_operators_block_is_an_h0_format_error(tmp_path, block):
    import json

    from impurityModel.ed.h0_format import H0FormatError, read_h0_file, write_h0_file

    path = tmp_path / "x.h0"
    write_h0_file(path, np.eye(2), impurity_orbitals={0: [0, 1]})
    lines = path.read_text().splitlines()
    header = json.loads(lines[1])
    header["operators"] = block
    lines[1] = json.dumps(header)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(H0FormatError):
        read_h0_file(path)


def _rotation(seed):
    from scipy.spatial.transform import Rotation

    return Rotation.random(random_state=seed).as_matrix()


@pytest.mark.parametrize("shell_ls", [(1,), (2,), (3,), (1, 2)])
def test_site_frames_recover_the_rotation_of_every_shell(shell_ls):
    from impurityModel.ed.local_operators import site_frames

    n = sum(2 * (2 * l + 1) for l in shell_ls)
    rot = _random_unitary(n, seed=n)
    r = _rotation(seed=len(shell_ls))
    frames = site_frames(LocalOperators.from_spherical(rot, shell_ls, site_rotation=r), rot, shell_ls)
    for frame in frames:
        np.testing.assert_allclose(frame.rotation, r, atol=1e-12)
        assert frame.residual < 1e-12
    # the axis/angle reproduce the matrix (Rodrigues)
    f = frames[0]
    k = np.array([[0, -f.axis[2], f.axis[1]], [f.axis[2], 0, -f.axis[0]], [-f.axis[1], f.axis[0], 0]])
    rebuilt = np.eye(3) + np.sin(f.angle) * k + (1 - np.cos(f.angle)) * k @ k
    np.testing.assert_allclose(rebuilt, r, atol=1e-10)


@pytest.mark.parametrize("kept", [6, 4, 2])
def test_site_frames_of_a_partial_shell_are_exact(kept):
    """Projected L components are not Frobenius-orthogonal; the least-squares fit still recovers R."""
    from impurityModel.ed.local_operators import site_frames

    rot = _random_unitary(10, seed=kept)[:kept, :]
    r = _rotation(seed=kept)
    frame = site_frames(LocalOperators.from_spherical(rot, (2,), site_rotation=r), rot, (2,))[0]
    np.testing.assert_allclose(frame.rotation, r, atol=1e-10)


def test_site_frames_give_the_identity_and_flag_a_non_rotation():
    from impurityModel.ed.local_operators import site_frames

    rot = np.eye(10, dtype=complex)
    ops = LocalOperators.from_spherical(rot, (2,))
    frame = site_frames(ops, rot, (2,))[0]
    assert frame.is_identity and frame.residual < 1e-14 and "global frame" in frame.describe()
    # the basis is not what rot_to_spherical says: L in another basis is no rotation of the reference
    wrong = ops.rotated(_random_unitary(10, seed=7))
    assert site_frames(replace_frame(wrong), rot, (2,))[0].residual > 1e-2
    # no orbital operators, or an s shell: nothing to fit
    assert site_frames(LocalOperators.from_spin_ordering(10), rot, (2,)) == [None]
    assert site_frames(LocalOperators.from_spherical(np.eye(2), (0,)), np.eye(2), (0,)) == [None]


def replace_frame(ops):
    import dataclasses

    return dataclasses.replace(ops, frame="global")


def test_the_model_fits_frames_only_for_global_operators():
    rot = np.eye(10, dtype=complex)
    r = _rotation(seed=5)
    common = dict(
        h0={},
        impurity_orbitals={0: list(range(10))},
        rot_to_spherical=rot,
        bath_states=({0: []}, {0: []}),
        n_spin_orbitals=10,
        shell_ls=(2,),
    )
    assert ImpurityModel(**common).site_frames() == [None]
    local = LocalOperators.from_spherical(rot, (2,))
    assert ImpurityModel(**common, operators=local).site_frames() == [None]
    glob = LocalOperators.from_spherical(rot, (2,), site_rotation=r)
    np.testing.assert_allclose(ImpurityModel(**common, operators=glob).site_frames()[0].rotation, r, atol=1e-12)
