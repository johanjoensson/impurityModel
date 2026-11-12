"""The DC shift on a multi-shell impurity acts on the hybridising (active) shells only (MS T4.7).

Fixture: a deep, bath-less ``p`` shell (spin-orbitals 0-5, level -60) beside a ``d`` shell (6-15, level +1)
with four bath orbitals hybridised to ``d`` only. The p occupation is pinned at 6 by its energy; a shift on it
moves nothing the criterion measures.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import dc_criteria
from impurityModel.ed.model import BasisOptions, ImpurityModel, SolverOptions

P_ORBS = list(range(6))
D_ORBS = list(range(6, 16))


def _p_plus_d(p_level=-60.0, u=0.0, d_rotation=None):
    h = np.zeros((20, 20), dtype=complex)
    for o in P_ORBS:
        h[o, o] = p_level
    for o in D_ORBS:
        h[o, o] = 1.0
    for b, e in zip(range(16, 20), (-2.0, -2.0, 2.0, 2.0)):
        h[b, b] = e
    for o, b in ((6, 16), (11, 17), (7, 18), (12, 19)):
        h[o, b] = h[b, o] = 0.3
    u4 = np.zeros((16,) * 4, dtype=complex)
    for i in D_ORBS:
        for j in D_ORBS:
            if i != j:
                u4[i, j, i, j] = u
    rot = np.eye(16, dtype=complex)
    if d_rotation is not None:
        # the model as the caller states it: the d shell in a rotated (non-spherical) basis
        big = np.eye(20, dtype=complex)
        big[6:16, 6:16] = d_rotation
        h = big @ h @ big.conj().T
        u4 = np.einsum(
            "ia,jb,kc,ld,abcd->ijkl", big[:16, :16], big[:16, :16], big[:16, :16].conj(), big[:16, :16].conj(), u4
        )
        rot = big[:16, :16].conj().T
    model = ImpurityModel.from_solver_matrix(
        h,
        16,
        dc=np.zeros((16, 16), dtype=complex),
        u4=u4,
        rot_to_spherical=rot,
        bath_valence_conduction=([16, 17], [18, 19]),
        shell_ls=(1, 2),
    )
    basis = BasisOptions(
        nominal_occ={0: 6, 1: 2},
        mixed_valence=None,
        tau=1e-2,
        slater_weight_min=np.sqrt(np.finfo(float).eps),
        truncation_threshold=int(1e8),
    )
    return model, basis, SolverOptions(dense_cutoff=1000)


def test_the_occupation_context_shifts_and_counts_only_the_hybridising_shell():
    model, basis, solver = _p_plus_d()
    ctx = dc_criteria._prepare_occupation_context(model, basis, solver, comm=MPI.COMM_WORLD)
    want = np.diag([0.0] * 6 + [1.0] * 10)
    np.testing.assert_array_equal(ctx.shift_input, want)
    np.testing.assert_array_equal(ctx.shift_solver, want)
    assert ctx.impurity_indices == D_ORBS and ctx.total_impurity_orbitals == 10
    # DFT reference over the active orbitals: the empty d level, not the 6 electrons of the deep p shell
    assert ctx.n0 < 1.0


def test_the_sector_context_shift_leaves_the_p_block_of_the_dc_untouched():
    model, basis, solver = _p_plus_d()
    ctx = dc_criteria._prepare_sector_context(model, basis, solver, comm=MPI.COMM_WORLD, memory_label="test")
    dc = ctx.dc(0.7)
    np.testing.assert_array_equal(dc[:6, :6], np.zeros((6, 6)))
    np.testing.assert_allclose(dc[6:, 6:], 0.7 * np.identity(10))
    assert ctx.active_indices == D_ORBS


def test_a_single_shell_model_keeps_the_exact_identity_and_the_full_index_list():
    from impurityModel.test.gf.test_fixed_dc import common_kwargs

    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    ctx = dc_criteria._prepare_sector_context(
        kwargs["model"], kwargs["basis"], kwargs["solver"], comm=MPI.COMM_WORLD, memory_label="test"
    )
    assert ctx.active_indices is None
    np.testing.assert_array_equal(ctx.shift_input, np.identity(ctx.shift_input.shape[0]))
    assert ctx.shift_solver is ctx.shift_input  # the one object, as before: never rebuilt as u^dagger I u


def test_the_plan_store_measures_the_class_occupation_over_the_active_shell():
    model, basis, solver = _p_plus_d()
    sector = dc_criteria._prepare_sector_context(model, basis, solver, comm=MPI.COMM_WORLD, memory_label="test")
    occupation = dc_criteria._prepare_occupation_context(model, basis, solver, comm=MPI.COMM_WORLD)
    assert sector.plan_store.active_orbitals == tuple(D_ORBS)
    assert occupation.plan_store.active_orbitals == tuple(D_ORBS)


def test_a_single_shell_plan_store_keeps_the_whole_impurity():
    h = np.zeros((12, 12), dtype=complex)
    for o in range(10):
        h[o, o] = 1.0
    h[10, 10], h[11, 11] = -2.0, 2.0
    h[0, 10] = h[10, 0] = h[5, 11] = h[11, 5] = 0.3
    model = ImpurityModel.from_solver_matrix(
        h,
        10,
        dc=np.zeros((10, 10), dtype=complex),
        u4=np.zeros((10,) * 4, dtype=complex),
        rot_to_spherical=np.eye(10, dtype=complex),
        bath_valence_conduction=([10], [11]),
    )
    _m, basis, solver = _p_plus_d()
    basis = BasisOptions(
        nominal_occ=2,
        mixed_valence=None,
        tau=1e-2,
        slater_weight_min=basis.slater_weight_min,
        truncation_threshold=int(1e8),
    )
    ctx = dc_criteria._prepare_sector_context(model, basis, solver, comm=MPI.COMM_WORLD, memory_label="test")
    assert ctx.plan_store.active_orbitals is None


def test_fixed_occupation_dc_shifts_only_the_hybridising_shell():
    """p(6) + d with U: the DC search moves the d block, leaves the deep p block at its guess, and the
    charge state it settles on keeps the p shell full (the sector total is 6 + the target d occupation)."""
    model, basis, solver = _p_plus_d(u=3.0)
    dc, sector = dc_criteria.fixed_occupation_dc(
        model, basis, solver, occupation=1.0, comm=MPI.COMM_WORLD, return_sector=True
    )
    np.testing.assert_array_equal(dc[:6, :6], np.zeros((6, 6)))
    d_block = dc[6:, 6:]
    np.testing.assert_allclose(d_block, d_block[0, 0] * np.identity(10), atol=1e-12)
    # the d occupation is 1 on the plateau mu in (1, 4) of E(n) = n (1 - mu) + 3 n (n - 1) / 2
    assert 1.0 < d_block[0, 0].real < 4.0
    assert sector == 7


def test_the_dc_search_says_which_shells_the_shift_covers(capsys):
    from impurityModel.ed.dc_criteria import _check_active_shells
    from impurityModel.ed.solver_basis import prepare_solver_basis

    model, basis, _solver = _p_plus_d()
    sb = prepare_solver_basis(
        model.h0, model.dc, model.u4, model.impurity_orbitals, basis.nominal_occ,
        rot_to_spherical=model.rot_to_spherical, verbosity=0, shell_ls=model.shell_ls,
    )  # fmt: skip
    _check_active_shells(sb, 0, 1)
    assert "DC shift acts on shell(s) l=2 (hybridizing); excluded: l=1 (no bath)" in capsys.readouterr().out


def test_the_nominal_the_reference_is_compared_with_counts_the_active_shell_only():
    """n0 counts the active d shell, so it is compared with the active nominal (2), not the p+d total (8)."""
    model, basis, solver = _p_plus_d(u=3.0)
    ctx = dc_criteria._prepare_occupation_context(model, basis, solver, comm=MPI.COMM_WORLD)
    assert ctx.nominal_active == 2
    assert sum(ctx.N0.values()) == 8


def test_the_solver_hamiltonian_is_shifted_on_the_active_block_only():
    model, basis, solver = _p_plus_d()
    ctx = dc_criteria._prepare_sector_context(model, basis, solver, comm=MPI.COMM_WORLD, memory_label="test")
    diff = ctx.shifted_h(0.7) - ctx.shifted_h(0.0)
    terms = {k: v for k, v in dict(diff).items() if abs(v) > 1e-12}
    ps = {orb for key in terms for (orb, _t) in key}
    assert ps and ps <= set(D_ORBS)


def test_a_multishell_model_without_a_hybridising_shell_is_refused_up_front():
    from types import SimpleNamespace

    sb = SimpleNamespace(
        active_impurity_indices=[],
        impurity_shell_index={o: 0 if o < 6 else 1 for o in range(16)},
        shell_ls=(1, 2),
        shell_has_bath={0: False, 1: False},
    )
    with pytest.raises(ValueError, match="no shell to act on"):
        dc_criteria._check_active_shells(sb, 0, 1)


def _assert_only_the_d_block_moved(dc):
    np.testing.assert_array_equal(dc[:6, :6], np.zeros((6, 6)))
    d_block = dc[6:, 6:]
    np.testing.assert_allclose(d_block, d_block[0, 0] * np.identity(10), atol=1e-12)
    assert abs(d_block[0, 0]) > 1e-3


def test_fixed_gap_dc_shifts_only_the_hybridising_shell():
    model, basis, solver = _p_plus_d(u=3.0)
    dc, sector = dc_criteria.fixed_gap_dc(model, basis, solver, comm=MPI.COMM_WORLD, return_sector=True)
    _assert_only_the_d_block_moved(dc)
    assert sector == 8  # the p shell stays full: nominal 6 + 2


def test_fixed_peak_dc_shifts_only_the_hybridising_shell():
    model, basis, solver = _p_plus_d(u=3.0)
    dc, _sector = dc_criteria.fixed_peak_dc(
        model, basis, solver, peak_position=2.0, comm=MPI.COMM_WORLD, return_sector=True, allow_charge_state_change=True
    )
    _assert_only_the_d_block_moved(dc)


def test_a_rotated_d_shell_keeps_the_shift_on_the_d_orbitals_in_both_bases():
    """The DC matrix lives in the caller's (rotated) basis, ``sb.h`` in the solver's; ``prepare_solver_basis``
    keeps only rotations inside a shell, so both shifts must still select exactly the d orbitals."""
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(10, 10)))
    model, basis, solver = _p_plus_d(u=3.0, d_rotation=q)
    ctx = dc_criteria._prepare_sector_context(model, basis, solver, comm=MPI.COMM_WORLD, memory_label="test")
    want = np.diag([0.0] * 6 + [1.0] * 10)
    np.testing.assert_array_equal(ctx.shift_input, want)
    np.testing.assert_array_equal(ctx.shift_solver, want)
    assert ctx.active_indices == D_ORBS
    np.testing.assert_array_equal(ctx.dc(0.7)[:6, :6], np.zeros((6, 6)))
    np.testing.assert_allclose(ctx.dc(0.7)[6:, 6:], 0.7 * np.identity(10))


def test_chi_hands_the_seed_window_of_a_bathless_shell_to_the_ground_state(monkeypatch):
    """calc_selfenergy seeds the deep p shell as a window; the chi driver used to seed the whole p+d space and so solved
    a different ground state (review MS1)."""
    from impurityModel.ed import gs_calc, susceptibility
    from impurityModel.ed.basis_generation import SeedWindow
    from impurityModel.ed.model import Meshes

    seen = {}

    class Stop(Exception):
        pass

    def spy(_h, basis_information, *_args, **_kwargs):
        seen["frozen"] = basis_information.get("frozen_occupations")
        raise Stop

    monkeypatch.setattr(gs_calc, "calc_gs", spy)
    model, basis, solver = _p_plus_d()
    meshes = Meshes(iw=None, w=np.linspace(-3.0, 3.0, 5), delta=0.1)
    with pytest.raises(Stop):
        susceptibility.calc_susceptibility_workflow(model, meshes, basis, solver, comm=MPI.COMM_WORLD)
    assert isinstance(seen["frozen"], SeedWindow)


def test_the_cap_probe_seeds_a_bathless_shell_as_the_solve_does(monkeypatch):
    """The auto cap is priced on a measured seed; it must be the window seed the solve uses, not the whole p+d space
    (review MS2)."""
    from impurityModel.ed import gs_connectivity
    from impurityModel.ed.solver_basis import prepare_solver_basis

    model, basis, _solver = _p_plus_d()
    sb = prepare_solver_basis(
        model.h0, model.dc, model.u4, model.impurity_orbitals, basis.nominal_occ,
        rot_to_spherical=model.rot_to_spherical, verbosity=0, shell_ls=model.shell_ls,
    )  # fmt: skip
    assert sb.seed_window is not None
    seen = []
    real = gs_connectivity._probe

    def spy(*args, **kwargs):
        seen.append(args[5] if len(args) > 5 else kwargs.get("frozen_occupations"))
        return real(*args, **kwargs)

    monkeypatch.setattr(gs_connectivity, "_probe", spy)
    gs_connectivity.solver_basis_connectivity(sb, MPI.COMM_WORLD)()
    assert MPI.COMM_WORLD.rank != 0 or seen == [sb.seed_window]


def test_n0_does_not_depend_on_which_shell_sits_at_the_higher_input_indices():
    """With a tiny p-d element the solver may relabel orbitals across shells; n0 is read from the input-basis h0 and so
    must count the active shell by its input indices. The d shell here has the same level and baths whether it comes
    first or last, so n0 must agree (review MS3: it was 0.019 vs 1.010)."""

    def build(p_first):
        p, d = (range(6), range(6, 16)) if p_first else (range(10, 16), range(10))
        p, d = list(p), list(d)
        h = np.zeros((20, 20), dtype=complex)
        for o in p:
            h[o, o] = -60.0
        for o in d:
            h[o, o] = 1.0
        for b, e in zip(range(16, 20), (-2.0, -2.0, 2.0, 2.0)):
            h[b, b] = e
        for o, b in ((d[0], 16), (d[5], 17), (d[1], 18), (d[6], 19)):
            h[o, b] = h[b, o] = 0.3
        h[p[0], d[0]] = h[d[0], p[0]] = 1e-7
        model = ImpurityModel.from_solver_matrix(
            h,
            16,
            dc=np.zeros((16, 16), dtype=complex),
            u4=np.zeros((16,) * 4, dtype=complex),
            rot_to_spherical=np.eye(16, dtype=complex),
            bath_valence_conduction=([16, 17], [18, 19]),
            shell_ls=(1, 2) if p_first else (2, 1),
        )
        basis = BasisOptions(
            nominal_occ={0: 6, 1: 2} if p_first else {0: 2, 1: 6},
            mixed_valence=None,
            tau=1e-2,
            slater_weight_min=np.sqrt(np.finfo(float).eps),
            truncation_threshold=int(1e8),
        )
        return model, basis, SolverOptions(dense_cutoff=1000)

    n0 = [dc_criteria._prepare_occupation_context(*build(first), comm=MPI.COMM_WORLD).n0 for first in (True, False)]
    assert n0[0] == pytest.approx(n0[1], abs=1e-6)


@pytest.mark.parametrize("dc_d", [0.0, 65.0])
def test_dc_trial_sectors_keep_the_bath_less_window_at_its_nominal_filling(dc_d):
    """H-MS4: the DC walk split each trial total by energetic filling over every impurity orbital, so a large d-shell
    double counting moved the deep p window's six electrons into d (window total 0). The self-energy seeds the window
    at its nominal filling; the DC's trial sectors must too, at every total around the nominal one."""
    from mpi4py import MPI

    from impurityModel.ed import dc_criteria

    model, basis, solver = _p_plus_d(u=3.0)
    dc = np.zeros((16, 16), dtype=complex)
    dc[6:, 6:] = dc_d * np.eye(10)
    model = ImpurityModel.from_solver_matrix(
        np.asarray(dc_criteria.extract_tensors(model.h0, n_orb=20, two_body=False)[0]),
        16,
        dc=dc,
        rot_to_spherical=model.rot_to_spherical,
        bath_valence_conduction=([16, 17], [18, 19]),
        shell_ls=(1, 2),
    )
    ctx = dc_criteria._prepare_sector_context(model, basis, solver, comm=MPI.COMM_WORLD, memory_label="test")
    window = sorted(ctx.seed_window)
    nominal = sum(ctx.N0[g] for g in window)
    assert nominal == 6
    n_tot = sum(ctx.N0.values())
    for n in (n_tot - 1, n_tot, n_tot + 1):
        trial = ctx.trial_occupation(n)
        assert sum(trial.values()) == n
        assert sum(trial[g] for g in window) == nominal, (n, trial)
    assert ctx.trial_occupation(nominal - 1) is None  # below the pinned window: no such sector
    # Where the walk at mu held the window elsewhere (Hartree-Fock's filling), the trial sectors at mu follow it.
    ctx.window_filling_at[0.25] = {g: (5 if k == 0 else ctx.N0[g]) for k, g in enumerate(window)}
    held = sum(ctx.window_filling_at[0.25].values())
    trial = ctx.trial_occupation(n_tot, 0.25)
    assert sum(trial[g] for g in window) == held and sum(trial.values()) == n_tot
    assert sum(ctx.trial_occupation(n_tot, 0.5)[g] for g in window) == nominal  # no walk at 0.5: the nominal
