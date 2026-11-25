"""Per-shell static double counting (FLL, AMF, nominal) on a multi-shell impurity, and the FLL fallback the ED
criteria give a bath-less shell.

Fixture: a ``p`` shell (spin-orbitals 0-5) and a ``d`` shell (6-15), each with its own atomic Slater-Condon
tensor, coupled by an inter-shell direct ``Upd`` and same-spin exchange ``Jpd``. On a uniform density each
shell then feels ``N_other (Upd - Jpd/2)`` from the other, in FLL and AMF alike: the two schemes differ only
in the self-interaction, which is inside a shell.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import dc_sector_context
from impurityModel.ed.atomic_physics import get_spherical_2_cubic_matrix, uj_from_u4
from impurityModel.ed.dc_static import (
    _shell_uj,
    amf_dc,
    fll_dc,
    nominal_dc,
    shells_with_bath,
    with_bathless_shells_at_nominal_fll,
)
from impurityModel.ed.lie_algebra import extract_tensors, rotate_two_body
from impurityModel.ed.model import BasisOptions, ImpurityModel, SolverOptions, atomic_u4

P = np.arange(6)
D = np.arange(6, 16)
SLATER_P = [4.0, 0.0, 2.5]
SLATER_D = [6.0, 0.0, 8.0, 0.0, 5.0]
UPD, JPD = 1.5, 0.4


def _u4(upd=UPD, jpd=JPD):
    u4 = np.zeros((16,) * 4, dtype=complex)
    u4[np.ix_(P, P, P, P)] = atomic_u4(1, SLATER_P)
    u4[np.ix_(D, D, D, D)] = atomic_u4(2, SLATER_D)
    spin = {int(o): int(o >= 3) for o in P} | {int(o): int(o >= 11) for o in D}
    for i in P:
        for j in D:
            for a, b in ((i, j), (j, i)):
                u4[a, b, a, b] = upd
                if spin[a] == spin[b]:
                    u4[a, b, b, a] = jpd
    return u4


def _model(u4, *, bath_on_d=True, dc=None):
    h = np.zeros((20, 20), dtype=complex)
    h[P, P] = -60.0
    h[D, D] = 1.0
    for b, e in zip(range(16, 20), (-2.0, -2.0, 2.0, 2.0)):
        h[b, b] = e
    if bath_on_d:
        for o, b in ((6, 16), (11, 17), (7, 18), (12, 19)):
            h[o, b] = h[b, o] = 0.3
    return ImpurityModel.from_solver_matrix(
        h,
        16,
        dc=dc,
        u4=u4,
        rot_to_spherical=np.eye(16, dtype=complex),
        bath_valence_conduction=([16, 17], [18, 19]),
        shell_ls=(1, 2),
    )


def _single_shell(l, slater):
    n = 2 * (2 * l + 1)
    h = np.zeros((n, n), dtype=complex)
    return ImpurityModel.from_solver_matrix(h, n, u4=atomic_u4(l, slater), rot_to_spherical=np.eye(n, dtype=complex))


def _off_shell(dc):
    return np.concatenate([dc[np.ix_(P, D)].ravel(), dc[np.ix_(D, P)].ravel()])


@pytest.mark.parametrize("scheme", [fll_dc, amf_dc])
def test_decoupled_shells_give_block_copies_of_the_single_shell_scheme(scheme):
    dc = scheme(_model(_u4(0.0, 0.0)), n=[6.0, 8.0])
    np.testing.assert_allclose(dc[np.ix_(P, P)], scheme(_single_shell(1, SLATER_P), n=6.0), atol=1e-12)
    np.testing.assert_allclose(dc[np.ix_(D, D)], scheme(_single_shell(2, SLATER_D), n=8.0), atol=1e-12)
    assert np.all(_off_shell(dc) == 0)


def test_each_shell_feels_the_hartree_and_exchange_of_the_other():
    decoupled = fll_dc(_model(_u4(0.0, 0.0)), n=[6.0, 8.0])
    coupled = fll_dc(_model(_u4()), n=[6.0, 8.0])
    shift = coupled - decoupled
    np.testing.assert_allclose(shift[np.ix_(P, P)], 8 * (UPD - JPD / 2) * np.eye(6), atol=1e-12)
    np.testing.assert_allclose(shift[np.ix_(D, D)], 6 * (UPD - JPD / 2) * np.eye(10), atol=1e-12)


def test_fll_and_amf_share_the_inter_shell_part():
    # They differ only in the self-interaction, inside a shell. Not in the potential of a full shell: there the
    # two energies meet, but their slopes do not. p^6 here: FLL U(n-1/2) - J(n-1)/2 = 20.75, AMF U(n-1) - J(n/2-1) = 19.
    n = [6.0, 8.0]
    coupled, decoupled = _model(_u4()), _model(_u4(0.0, 0.0))
    np.testing.assert_allclose(
        fll_dc(coupled, n=n) - fll_dc(decoupled, n=n), amf_dc(coupled, n=n) - amf_dc(decoupled, n=n), atol=1e-12
    )
    np.testing.assert_allclose(fll_dc(decoupled, n=n)[0, 0], 4.0 * 5.5 - 0.25 * 5.0)
    np.testing.assert_allclose(amf_dc(decoupled, n=n)[0, 0], 4.0 * 5.0 - 0.5 * 2.0)


def test_nominal_is_fll_at_the_per_shell_nominal_occupations():
    model = _model(_u4())
    np.testing.assert_allclose(nominal_dc(model, {0: 6, 1: 8}), fll_dc(model, n=[6.0, 8.0]), atol=1e-12)


@pytest.mark.parametrize(
    "call",
    [
        lambda m: fll_dc(m, n=14.0),
        lambda m: amf_dc(m, n=14.0),
        lambda m: nominal_dc(m, 14),
        lambda m: fll_dc(m, n=[6.0, 8.0], u=3.0, j=0.5),
        lambda m: fll_dc(m, n={0: 6.0}),
    ],
)
def test_one_number_for_several_shells_is_refused(call):
    with pytest.raises(ValueError, match="shell"):
        call(_model(_u4()))


def test_explicit_per_shell_u_and_j_need_no_u4_for_the_intra_shell_part():
    model = _model(None)
    dc = fll_dc(model, n=[6.0, 8.0], u=[4.0, 6.0], j=[0.5, 0.9])
    np.testing.assert_allclose(dc[np.ix_(P, P)], (4.0 * 5.5 - 0.25 * 5.0) * np.eye(6))
    np.testing.assert_allclose(dc[np.ix_(D, D)], (6.0 * 7.5 - 0.45 * 7.0) * np.eye(10))


def test_a_partial_set_averages_over_the_orbitals_it_holds():
    # The d shell as its t2g set only, in a basis that mixes the t2g orbitals within each spin: (U, J) must be the
    # average over the six t2g spin-orbitals, whatever that basis is.
    cubic = get_spherical_2_cubic_matrix(spinpol=False, l=2)  # columns: eg (2), t2g (3)
    t2g = cubic[:, 2:5]
    rng = np.random.default_rng(3)
    mix, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    kets = np.zeros((10, 6), dtype=complex)  # spherical kets of the six t2g input orbitals
    kets[:5, :3] = t2g
    kets[5:, 3:] = t2g @ mix
    rot = np.zeros((12, 16), dtype=complex)
    rot[:6, :6] = np.eye(6)
    rot[6:, 6:] = kets.T
    u4_sph = _u4(0.0, 0.0)
    to_input = np.zeros((16, 12), dtype=complex)
    to_input[:6, :6] = np.eye(6)
    to_input[6:, 6:] = kets
    u4_input = rotate_two_body(u4_sph, to_input)
    model = ImpurityModel.from_solver_matrix(
        np.zeros((12, 12), dtype=complex), 12, u4=u4_input, rot_to_spherical=rot.conj(), shell_ls=(1, 2)
    )

    direct = np.zeros((10, 6), dtype=complex)  # the same set, spin-major, no mixing
    direct[:5, :3] = t2g
    direct[5:, 3:] = t2g
    want = uj_from_u4(rotate_two_body(atomic_u4(2, SLATER_D), direct))
    rows = [np.arange(6), np.arange(6, 12)]
    _, (u_t2g, j_t2g) = _shell_uj(model, u4_input, rows)
    np.testing.assert_allclose([u_t2g, j_t2g], want, atol=1e-10)
    # the full-shell average is a different number: a partial set is not averaged over orbitals it lacks
    assert not np.isclose(u_t2g, uj_from_u4(atomic_u4(2, SLATER_D))[0])


def test_shells_with_bath_follows_the_solver_basis_rule():
    assert shells_with_bath(_model(_u4())) == [False, True]


def test_a_bathless_shell_takes_fll_at_the_nominal_occupations_and_the_rest_keeps_the_guess():
    guess = 0.3 * np.eye(16, dtype=complex)
    guess[0, 6] = guess[6, 0] = 0.1
    model = _model(_u4(), dc=guess)
    out = with_bathless_shells_at_nominal_fll(model, {0: 6, 1: 2})
    dc = extract_tensors(out.dc, n_orb=16, two_body=False)[0]
    want_p = fll_dc(model, n=[6.0, 2.0])[np.ix_(P, P)]
    np.testing.assert_allclose(dc[np.ix_(P, P)], want_p, atol=1e-12)
    np.testing.assert_allclose(dc[np.ix_(D, D)], guess[np.ix_(D, D)])
    assert np.all(_off_shell(dc) == 0)


def test_the_fallback_leaves_a_model_alone_when_every_shell_or_no_shell_has_a_bath():
    single = _single_shell(2, SLATER_D)
    assert with_bathless_shells_at_nominal_fll(single, {0: 8}) is single
    no_bath = _model(_u4(), bath_on_d=False)  # every bath orbital falls back to the first impurity orbital (p)
    assert shells_with_bath(no_bath) == [True, False]
    out = with_bathless_shells_at_nominal_fll(no_bath, {0: 6, 1: 2})
    assert out is not no_bath  # the d shell has no bath: it gets FLL


def test_the_sector_search_shifts_the_guess_on_the_bath_shell_and_holds_fll_on_the_other():
    model = _model(_u4(), dc=np.zeros((16, 16), dtype=complex))
    basis = BasisOptions(
        nominal_occ={0: 6, 1: 2},
        mixed_valence=None,
        tau=1e-2,
        slater_weight_min=np.sqrt(np.finfo(float).eps),
        truncation_threshold=int(1e8),
    )
    fallback = with_bathless_shells_at_nominal_fll(model, basis.nominal_occ)
    ctx = dc_sector_context._prepare_sector_context(
        fallback, basis, SolverOptions(dense_cutoff=1000), comm=MPI.COMM_WORLD, memory_label="test"
    )
    dc = ctx.dc(0.7)
    np.testing.assert_allclose(dc[np.ix_(P, P)], fll_dc(model, n=[6.0, 2.0])[np.ix_(P, P)], atol=1e-12)
    np.testing.assert_allclose(dc[np.ix_(D, D)], 0.7 * np.eye(10), atol=1e-12)


def test_the_fallback_needs_no_u_and_j_of_a_shell_with_a_bath():
    # d as four spin-mixed orbitals (|m up> + |m down>)/sqrt(2): no (U, J) can be read off that set, and the
    # fallback never asks for it, since it keeps only the block of the bath-less p shell.
    rot = np.zeros((10, 16), dtype=complex)
    rot[:6, :6] = np.eye(6)
    for k in range(4):
        rot[6 + k, 6 + k] = rot[6 + k, 11 + k] = 1 / np.sqrt(2)
    u4 = np.zeros((10,) * 4, dtype=complex)
    u4[np.ix_(P, P, P, P)] = atomic_u4(1, SLATER_P)
    h = np.zeros((12, 12), dtype=complex)
    h[P, P] = -60.0
    h[6, 10] = h[10, 6] = h[7, 11] = h[11, 7] = 0.3
    model = ImpurityModel.from_solver_matrix(
        h, 10, dc=np.zeros((10, 10)), u4=u4, rot_to_spherical=rot, bath_valence_conduction=([10], [11]), shell_ls=(1, 2)
    )
    assert shells_with_bath(model) == [False, True]
    with pytest.raises(ValueError, match="spin-collinear"):
        fll_dc(model, n=[6.0, 2.0])
    out = with_bathless_shells_at_nominal_fll(model, {0: 6, 1: 2})
    dc = extract_tensors(out.dc, n_orb=10, two_body=False)[0]
    np.testing.assert_allclose(dc[np.ix_(P, P)], (4.0 * 5.5 - 0.25 * 5.0) * np.eye(6), atol=1e-12)
