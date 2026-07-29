"""
Tests for the fixed-peak and fixed-occupation double counting criteria.

Analytically solvable model: two impurity spin-orbitals at energy eps with a
Hubbard interaction U, weakly coupled (hopping v) to two valence bath
spin-orbitals at energy eps_b. With the double counting dc entering the
Hamiltonian as -dc * n_imp, and if the total electron number were held fixed
at N_imp + N_bath = 3 (N0 = 1, both bath orbitals always filled):

    E[N_imp = 0] = 2 eps_b
    E[N_imp = 1] = (eps - dc) + 2 eps_b          + O(v^2)
    E[N_imp = 2] = 2 (eps - dc) + U + 2 eps_b    + O(v^2)

giving an electron-addition peak at E[2] - E[1] = eps + U - dc, an
electron-removal peak at E[1] - E[0] = eps - dc, and a charge-transfer
crossing (N_imp: 1 -> 2) at dc > eps + U - eps_b = 6.

Both searches now determine their sector(s) through
``groundstate.find_ground_state_basis`` -- the identical search
``calc_selfenergy`` uses -- rather than by construction at a fixed N0, so
that a dc found here means the same thing calc_selfenergy will find at that
dc (see double_counting.py's module docstring). ``find_ground_state_basis``
does not hold total N fixed: with mixed_valence=0 (the default in this
file), a bath orbital's occupation cannot deviate from nominal at all, so
"N_imp = 2" as a *trial sector* pairs with the bath still fully occupied
(total N = 4, not the charge-transfer-conserving total N = 3 the table
above assumes) -- filling the bound bath level a second time is cheaper
than the true charge-transfer picture, so this alternate sector's energy
drops below the N_imp = 1 sector already at a much smaller dc than 6. Tests
below that push N_imp through this crossing (occupation >= 2 with the
default excitation_budget) hit that alternate sector, not the analytic
dc = 6; their expected values are calibrated against the actual code, with
a comment explaining why. Tests that never push through the crossing (the
peak tests, at very weak hopping v = 0.01) still match the table above,
since the sector search never leaves N_imp = 1 for them -- keep using the
analytic formula there.
"""

from dataclasses import replace

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.basis_transcription import build_density_matrices
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.groundstate import find_ground_state_basis
from impurityModel.ed.lie_algebra import tensors_to_operator
from impurityModel.ed.model import BasisOptions, ImpurityModel, SolverOptions
from impurityModel.ed.selfenergy import fixed_occupation_dc, fixed_peak_dc
from impurityModel.ed.solver_basis import prepare_solver_basis

EPS = -1.0
U = 3.0
EPS_B = -4.0


def build_model(v, dc_scale):
    h0 = np.zeros((4, 4), dtype=complex)
    for s in range(2):
        imp, bath = s, 2 + s
        h0[imp, imp] = EPS
        h0[bath, bath] = EPS_B
        h0[imp, bath] = v
        h0[bath, imp] = v
    dc = np.identity(2, dtype=complex) * dc_scale
    u4 = np.zeros((4, 4, 4, 4), dtype=complex)
    # RSPt convention <ij|V|kl> with pairs (i,k),(j,l): the density-density
    # element U n_0 n_1 sits at u4[0,1,0,1] (and its exchange-symmetric partner).
    u4[0, 1, 0, 1] = U
    u4[1, 0, 1, 0] = U
    return h0, dc, u4


def common_kwargs(v, tau, dc_scale=0.5):
    """``fixed_peak_dc``/``fixed_occupation_dc`` kwargs plus the dense dc guess used to build them.

    Returns ``(kwargs, dc_guess)``: ``kwargs`` is exactly the ``model``/``basis``/``solver``/``comm``
    the search functions accept (no extra keys), ``dc_guess`` is the dense ``model.dc`` matrix for
    the tests' own assertions (:func:`assert_uniform_shift`).
    """
    h0, dc, u4 = build_model(v, dc_scale)
    model = ImpurityModel.from_solver_matrix(
        h0,
        dc.shape[0],
        dc=dc,
        u4=u4,
        rot_to_spherical=np.eye(2, dtype=complex),
        bath_valence_conduction=([2, 3], []),
    )
    basis = BasisOptions(
        nominal_occ={0: 1},
        mixed_valence=None,
        spin_flip_dj=False,
        tau=tau,
        slater_weight_min=np.sqrt(np.finfo(float).eps),
        truncation_threshold=int(1e8),
    )
    solver = SolverOptions(dense_cutoff=1000)
    return dict(model=model, basis=basis, solver=solver, comm=MPI.COMM_WORLD), dc


def assert_uniform_shift(dc, dc_guess):
    """The result must be dc_guess plus a real uniform shift."""
    shift = dc - dc_guess
    assert np.allclose(shift, shift[0, 0] * np.identity(2)), shift
    assert abs(shift[0, 0].imag) < 1e-12


def test_fixed_peak_dc_addition_peak():
    target = 1.2
    kwargs, dc_guess = common_kwargs(v=0.01, tau=1e-3)
    dc = fixed_peak_dc(peak_position=target, **kwargs)
    assert_uniform_shift(dc, dc_guess)
    # E[2] - E[1] = eps + U - dc = target
    expected = EPS + U - target
    assert np.allclose(np.diag(dc).real, expected, atol=5e-3), dc


def test_fixed_peak_dc_removal_peak():
    # A negative peak position must exercise the removal branch, E[1] - E[0]
    target = -1.5
    kwargs, dc_guess = common_kwargs(v=0.01, tau=1e-3, dc_scale=0.2)
    dc = fixed_peak_dc(peak_position=target, **kwargs)
    assert_uniform_shift(dc, dc_guess)
    # E[1] - E[0] = eps - dc = target
    expected = EPS - target
    assert np.allclose(np.diag(dc).real, expected, atol=5e-3), dc


def _split_block_kwargs(dc_scale=0.0):
    """A model whose impurity block structure *splits*, in miniature.

    Four impurity spin-orbitals at two distinct on-site energies, so
    :func:`prepare_solver_basis` derives two orbital-symmetry groups from a single-group input --
    the same thing a cubic crystal field does to a d-shell (``{0: [0..9]}`` -> ``eg``/``t2g``).
    Every NiO-like workload has this shape; the ``build_model`` fixture above does not, because
    ``diag(EPS, EPS)`` is one block.
    """
    h_imp = np.diag([0.4, 0.4, -0.4, -0.4]).astype(complex)
    v = np.eye(4, dtype=complex) * 0.2
    h_bath = np.diag([-2.0, -2.0, -2.1, -2.1]).astype(complex)
    u4 = np.zeros((4,) * 4, dtype=complex)
    for i in range(4):
        for j in range(4):
            if i != j:
                u4[i, j, i, j] = 2.0
    dc = np.identity(4, dtype=complex) * dc_scale
    model = ImpurityModel.from_blocks(
        h_imp,
        v,
        h_bath,
        u4=u4,
        dc=dc,
        rot_to_spherical=np.eye(4, dtype=complex),
        bath_valence_conduction=(list(range(4, 8)), []),
    )
    basis = BasisOptions(
        nominal_occ={0: 2},
        mixed_valence=None,
        spin_flip_dj=False,
        tau=1e-3,
        slater_weight_min=np.sqrt(np.finfo(float).eps),
        truncation_threshold=int(1e6),
    )
    return dict(model=model, basis=basis, solver=SolverOptions(dense_cutoff=1000), comm=MPI.COMM_WORLD), dc


def test_fixed_peak_dc_runs_on_split_block_impurity():
    """Regression: the peak search used to key its sectors on the *input* group.

    ``prepare_solver_basis`` re-derives the impurity grouping, so a single-group input becomes
    several derived groups on any cubic crystal field. The old code took ``group_key`` from the
    input ``N0`` and then indexed the *derived* layout with it, building one-key occupation dicts
    that ``generate_initial_basis`` iterates over every derived group -- ``KeyError``. RSPt's
    generic handler turns that into ``comm.Abort``, so fixed-peak dc killed the run on exactly
    the material class it exists for.
    """
    kwargs, dc_guess = _split_block_kwargs()
    dc = fixed_peak_dc(peak_position=0.5, **kwargs)
    assert dc.shape == (4, 4)
    shift = dc - dc_guess
    np.testing.assert_allclose(shift, shift[0, 0] * np.identity(4), atol=1e-12)


def test_ground_state_basis_reports_its_sector_because_the_basis_cannot():
    """``find_ground_state_basis`` returns the eigenvector support, not a pure sector.

    The CIPSI expansion widens the impurity occupation window (``build_excited_restrictions`` is
    called with ``imp_change=None``, which its own docstring defines as unconstrained), and
    ``calc_energy`` then reduces the basis to the eigenvector support. So the returned basis spans
    several impurity occupations and *no* determinant identifies the sector -- reading one, as
    ``fixed_peak_dc`` used to, centred the peak on whichever came first. The winning occupation
    has to be carried out of the search explicitly, which is what ``ground_state_occupation`` is.
    """
    from impurityModel.ed import product_state_representation as psr
    from impurityModel.ed.groundstate import find_ground_state_basis

    kwargs, _ = _split_block_kwargs()
    model, basis_opts = kwargs["model"], kwargs["basis"]
    sb = prepare_solver_basis(
        model.h0,
        model.dc,
        model.u4,
        model.impurity_orbitals,
        basis_opts.nominal_occ,
        basis_opts.mixed_valence,
        model.rot_to_spherical,
        0,
    )
    imp_idx = [o for blocks in sb.impurity_orbitals.values() for block in blocks for o in block]
    gs = find_ground_state_basis(
        sb.h,
        sb.impurity_orbitals,
        sb.bath_states,
        sb.nominal_occ,
        mixed_valence=sb.mixed_valence,
        tau=basis_opts.tau,
        dense_cutoff=1000,
        spin_flip_dj=False,
        comm=None,
        verbose=False,
        truncation_threshold=int(1e6),
        slaterWeightMin=1e-12,
    )
    per_det = {
        sum(1 for o in psr.bytes2tuple(bytes(state.to_bytearray()), model.n_spin_orbitals) if o in imp_idx)
        for state in gs
    }
    assert len(per_det) > 1, f"fixture no longer mixes occupations ({per_det}); the test is moot"
    assert sum(gs.ground_state_occupation.values()) in per_det


def test_fixed_peak_dc_accepts_multiple_groups():
    """The criterion moves one electron on/off the impurity as a whole, so no group is special.

    It used to raise ``ValueError`` here. The many-body basis is filtered on the *total* impurity
    charge alone (``generate_initial_basis``), so a per-group occupation is not a handle the
    criterion could act on even in principle.
    """
    kwargs, _ = common_kwargs(v=0.01, tau=1e-3, dc_scale=1.0)
    kwargs["basis"] = replace(kwargs["basis"], nominal_occ={0: 1, 1: 1})
    dc = fixed_peak_dc(peak_position=1.0, **kwargs)
    assert dc.shape == (2, 2)


def test_fixed_occupation_dc_already_converged():
    # At the guess (dc=0.5, well below the dc=6 charge-transfer point) the impurity already
    # holds one electron; requesting occupation 1 must return the guess unchanged (mu=0).
    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2)
    dc = fixed_occupation_dc(occupation=1.0, **kwargs)
    np.testing.assert_allclose(dc, dc_guess, atol=1e-8)


def test_fixed_occupation_dc_increases_occupation():
    # Requesting two electrons on the impurity: find_ground_state_basis's sector search hits the
    # "free bath electron" alternate sector (module docstring) well before the analytic
    # charge-transfer crossing at dc = 6; the achieved dc matches the code, not that formula.
    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2, dc_scale=0.5)
    dc = fixed_occupation_dc(occupation=2.0, **kwargs)
    assert_uniform_shift(dc, dc_guess)
    np.testing.assert_allclose(dc[0, 0].real, 2.5, atol=0.05)


def test_fixed_occupation_dc_decreases_occupation():
    # A guess of 7 puts two electrons on the impurity; requesting one electron
    # must bring the double counting back below the charge-transfer point.
    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2, dc_scale=7.0)
    dc = fixed_occupation_dc(occupation=1.0, **kwargs)
    assert_uniform_shift(dc, dc_guess)
    assert dc[0, 0].real < 6.0, dc


def test_fixed_occupation_dc_low_target_lands_on_plateau(capsys):
    # Under the old fixed-N0=1 picture this occupation (below N_imp=1, with both bath orbitals
    # already full) was unreachable and raised. find_ground_state_basis's actual sector search
    # is not fixed-N: it can land on a smaller-total-N sector (module docstring) where a low
    # impurity occupation is a genuine plateau, not an out-of-range target -- the search
    # converges to the closest point on that plateau (with a warning) instead of raising.
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2, dc_scale=0.5)
    dc = fixed_occupation_dc(occupation=0.2, **kwargs)
    assert np.isfinite(dc).all()
    out = capsys.readouterr().out
    if MPI.COMM_WORLD.rank != 0:
        assert "no double-counting shift attains" not in out, out
        return
    # The report must say the criterion has *no solution* here, not merely that the answer is a
    # little uncertain. Measured on this fixture, n(mu) steps 0.0112 -> 1.0018 across a
    # charge-sector boundary and the target 0.2 lies in the gap.
    assert "no double-counting shift attains the requested target 0.2" in out, out
    assert "0.0112" in out and "1.0018" in out, out
    assert "a jump of" in out and "knife-edge" in out, out
    # And it must name the distance to the boundary -- the hazard is that mu lands on it, which a
    # dc cached across CSC iterations turns into a d8/d9 limit cycle.
    assert "from the boundary" in out, out


def test_noninteracting_impurity_occupation_matches_fermi_fill():
    # The h_loc-derived target is the Fermi-filled (mu=0) occupation of the full
    # non-interacting h0. Build a 1-impurity / 1-bath cluster with an impurity
    # level poking above the Fermi level so the answer is genuinely fractional,
    # and compare against an independent per-eigenvector Fermi sum.
    from impurityModel.ed.dc_reference import _noninteracting_impurity_occupation

    e_imp, e_bath, v, tau = 0.5, -2.0, 0.5, 0.1
    h0 = {
        ((0, "c"), (0, "a")): e_imp,
        ((1, "c"), (1, "a")): e_bath,
        ((0, "c"), (1, "a")): v,
        ((1, "c"), (0, "a")): v,
    }
    n = _noninteracting_impurity_occupation(h0, impurity_indices=[0], n_spin_orbitals=2, tau=tau)

    h = np.array([[e_imp, v], [v, e_bath]], dtype=complex)
    energies, vecs = np.linalg.eigh(h)
    f = 1.0 / (1.0 + np.exp(energies / tau))
    expected = float(np.sum(f * np.abs(vecs[0, :]) ** 2))  # <imp| sum_n f_n |v_n><v_n| |imp>
    assert 0.0 < expected < 1.0  # genuinely fractional, not a plateau boundary
    assert np.isclose(n, expected, atol=1e-12), (n, expected)


def test_fixed_occupation_dc_self_consistent_targets_dft_occupation():
    # Omitting `occupation` pins the interacting occupation to the DFT reference N0: the Fermi
    # filling of the RAW h0 (the KS Hamiltonian of the h0 - dc + U contract), computed once --
    # NOT of h0 - dc, which for a realistic dc sinks the impurity levels below E_F and saturates
    # the reference at the full shell (the NiO n0 == 10 bug). For this dimer the raw impurity
    # level eps = -1 already sits below E_F, so N0 = 2 (full shell), exactly like an explicit
    # occupation=2.0 request -- and it hits the same "free bath electron" alternate sector well
    # short of the analytic dc = 6 (module docstring), converging to the same dc as
    # test_fixed_occupation_dc_increases_occupation.
    from impurityModel.ed.dc_reference import _noninteracting_impurity_occupation

    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2)
    model, basis = kwargs["model"], kwargs["basis"]

    n0_ref = _noninteracting_impurity_occupation(model.h0, [0, 1], 4, basis.tau)
    assert np.isclose(n0_ref, 2.0, atol=1e-2), n0_ref

    dc_auto = fixed_occupation_dc(**kwargs)
    assert_uniform_shift(dc_auto, dc_guess)
    np.testing.assert_allclose(dc_auto[0, 0].real, 2.5, atol=0.05)


def test_saturated_reference_warns(capsys):
    # The dimer's raw-h0 filling is N0 = 2.0 = the full impurity shell, i.e. exactly the
    # coarse-bath saturation the warning targets (NiO with 1 or 5 valence-only bath states:
    # n0 == 10). Self-consistent mode must warn and still run; an explicit target must not.
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    fixed_occupation_dc(verbosity=1, **kwargs)
    out = capsys.readouterr().out
    if MPI.COMM_WORLD.rank == 0:  # the warning prints on rank 0 only
        assert "saturated at the full impurity shell" in out

    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    fixed_occupation_dc(occupation=1.0, verbosity=1, **kwargs)
    out = capsys.readouterr().out
    assert "saturated" not in out


def test_fixed_occupation_dc_reference_ignores_dc_guess():
    # NiO in miniature: a large dc_guess must not corrupt the DFT reference. With dc_guess = -3
    # the shifted one-body impurity level eps - dc = +2 pokes above E_F, so the OLD reference
    # fill(h0 - dc) was ~0 -- unreachable (N_imp >= 1 here), the search failed or wandered. The
    # reference is the raw-h0 filling (N0 = 2, independent of the guess), so the search must
    # still converge somewhere on the "free bath electron" alternate-sector plateau (module
    # docstring), the same crossing test_fixed_occupation_dc_increases_occupation hits. It need
    # not be the *same* absolute dc from this guess: the bidirectional bracket search
    # (_solve_dc_shift) finds whichever crossing sits nearest its own starting point, and the
    # sector landscape here is not a single monotone staircase, so different guesses can settle
    # on different (still N_imp = 2) crossings.
    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2, dc_scale=-3.0)
    dc = fixed_occupation_dc(**kwargs)
    assert_uniform_shift(dc, dc_guess)
    np.testing.assert_allclose(dc[0, 0].real, 5.0, atol=0.05)


@pytest.mark.mpi
def test_fixed_peak_dc_ranks_agree():
    # The Newton loop in fixed_peak_dc branches on Lanczos energies, which are
    # only replicated to roundoff across ranks. Every rank must nevertheless run
    # the same iterations and return an identical dc; a per-rank divergence
    # would deadlock on the next collective solve instead of returning here.
    comm = MPI.COMM_WORLD
    kwargs, _ = common_kwargs(v=0.01, tau=1e-3)
    dc = fixed_peak_dc(peak_position=1.2, **kwargs)
    gathered = comm.gather(dc, root=0)
    if comm.rank == 0:
        for other in gathered[1:]:
            assert np.array_equal(dc, other), (dc, other)


@pytest.mark.mpi
def test_fixed_occupation_dc_ranks_agree():
    # Occupation control keys off the Allreduced density matrix, so agreement is
    # by construction; guard it against regressions all the same.
    comm = MPI.COMM_WORLD
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    dc = fixed_occupation_dc(occupation=2.0, **kwargs)
    gathered = comm.gather(dc, root=0)
    if comm.rank == 0:
        for other in gathered[1:]:
            assert np.array_equal(dc, other), (dc, other)


@pytest.mark.mpi
@pytest.mark.parametrize(
    "criterion, call",
    [
        ("peak", lambda kw: fixed_peak_dc(peak_position=1.2, **kw)),
        ("occupation", lambda kw: fixed_occupation_dc(occupation=2.0, **kw)),
    ],
)
def test_dc_search_visits_the_same_mu_sequence_on_every_rank(monkeypatch, criterion, call):
    """The *path*, not just the answer.

    The three ``ranks_agree`` tests above compare only the returned ``dc``, which passes even
    when the ranks took different routes and coincidentally met ``tol`` at the same place. The
    deadlock this guards against is upstream of that: ``_solve_dc_shift``'s residual is the value
    of a *collective* observable replicated only to roundoff, and it gates every branch that
    decides whether the next collective call happens. One ulp of disagreement on a ``tol`` test
    or on the ``g * g_prev < 0`` sign test sends ranks down different sequences of collectives --
    they hang rather than return, so a test comparing return values can never see it. Recording
    the trial ``mu`` sequence and comparing it across ranks is what actually pins the property.
    """
    import impurityModel.ed.dc_criteria as dc_module

    comm = MPI.COMM_WORLD
    kwargs, _ = common_kwargs(v=0.01 if criterion == "peak" else 0.3, tau=1e-3 if criterion == "peak" else 1e-2)

    visited = []
    real_solve = dc_module._solve_dc_shift

    def recording_solve(observable, target, **kw):
        def recording_observable(mu):
            visited.append(mu)
            return observable(mu)

        return real_solve(recording_observable, target, **kw)

    # dc_criteria, not the double_counting shim: the shim's `from dc_search import
    # _solve_dc_shift` is a *new binding*, so patching it there leaves the real call site reading
    # its own and the stub never runs -- with `visited` empty on every rank, the comparison below
    # succeeds vacuously. The non-empty assertion is what makes a dead patch fail loudly.
    monkeypatch.setattr(dc_module, "_solve_dc_shift", recording_solve)
    call(kwargs)
    assert visited, "the _solve_dc_shift stub never ran: the monkeypatch missed the call site"

    gathered = comm.gather(visited, root=0)
    if comm.rank == 0:
        for other in gathered[1:]:
            assert visited == other, f"{criterion} dc search took different mu paths: {visited} vs {other}"


def _capture_prepared_bases(monkeypatch):
    """Monkeypatch the basis-building entry point to record every ``Basis`` built.

    Both ``fixed_peak_dc`` and ``fixed_occupation_dc`` now determine their sector(s) through
    ``groundstate.find_ground_state_basis``/``calc_energy`` (imported function-locally at call
    time, so patching the module attribute here is picked up), which builds each trial's basis
    through ``groundstate.build_basis_and_solver``. Since the walk visits multiple trial
    occupations, one call now captures *every* trial basis, not one per sector -- callers that
    need a specific basis (e.g. the winning one) should index from the end of the list, not
    assume a fixed count.
    """
    import impurityModel.ed.groundstate as gs_module

    captured = []

    original_build = gs_module.build_basis_and_solver

    def build_wrapper(*args, **kwargs):
        basis, solver = original_build(*args, **kwargs)
        captured.append(basis)
        return basis, solver

    monkeypatch.setattr(gs_module, "build_basis_and_solver", build_wrapper)
    return captured


def test_fixed_occupation_dc_derives_cap_when_threshold_is_none(monkeypatch):
    import impurityModel.ed.dc_criteria as dc_module

    calls = []

    def fake_suggest(n_spin_orbitals, **kwargs):
        calls.append(kwargs)
        return 50

    monkeypatch.setattr(dc_module, "suggest_truncation_threshold", fake_suggest)
    monkeypatch.setattr(dc_module, "log_memory_budget", lambda *a, **kw: None)
    captured = _capture_prepared_bases(monkeypatch)

    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2)
    kwargs["basis"] = replace(kwargs["basis"], truncation_threshold=None)
    dc = fixed_occupation_dc(occupation=1.0, **kwargs)

    assert len(calls) == 1
    assert captured and captured[0].truncation_threshold == 50
    np.testing.assert_allclose(dc, dc_guess, atol=1e-8)


def test_fixed_peak_dc_derives_cap_when_threshold_is_none(monkeypatch):
    import impurityModel.ed.dc_criteria as dc_module

    calls = []

    def fake_suggest(n_spin_orbitals, **kwargs):
        calls.append(kwargs)
        return 50

    monkeypatch.setattr(dc_module, "suggest_truncation_threshold", fake_suggest)
    monkeypatch.setattr(dc_module, "log_memory_budget", lambda *a, **kw: None)
    captured = _capture_prepared_bases(monkeypatch)

    kwargs, _ = common_kwargs(v=0.01, tau=1e-3)
    kwargs["basis"] = replace(kwargs["basis"], truncation_threshold=None)
    fixed_peak_dc(peak_position=1.2, **kwargs)

    # One probe (not one per trial basis: truncation_threshold is resolved once, then passed
    # explicitly into every find_ground_state_basis/calc_energy call the walk makes), and every
    # trial basis carries its cap.
    assert len(calls) == 1
    assert calls[0]["safety"] == pytest.approx(dc_module.DEFAULT_MEMORY_SAFETY / 2)
    assert captured
    assert all(b.truncation_threshold == 50 for b in captured)


def test_explicit_threshold_skips_memory_probe(monkeypatch):
    import impurityModel.ed.dc_criteria as dc_module

    calls = []

    def fake_suggest(*args, **kwargs):
        calls.append(1)
        return 50

    monkeypatch.setattr(dc_module, "suggest_truncation_threshold", fake_suggest)
    monkeypatch.setattr(dc_module, "log_memory_budget", lambda *a, **kw: None)

    # Positive control first. Every assertion below is `not calls`, which a monkeypatch that
    # missed its call site satisfies vacuously -- exactly the failure mode a re-export shim
    # invites. Show the stub is reachable before concluding anything from its silence.
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    kwargs["basis"] = replace(kwargs["basis"], truncation_threshold=None)
    fixed_occupation_dc(occupation=1.0, **kwargs)
    assert calls, "the suggest_truncation_threshold stub never ran; the rest of this test is vacuous"
    calls.clear()

    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    fixed_occupation_dc(occupation=1.0, **kwargs)
    assert not calls

    kwargs, _ = common_kwargs(v=0.01, tau=1e-3)
    fixed_peak_dc(peak_position=1.2, **kwargs)
    assert not calls

    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    kwargs["basis"] = replace(kwargs["basis"], truncation_threshold=np.inf)
    fixed_occupation_dc(occupation=1.0, **kwargs)
    assert not calls


@pytest.mark.mpi
def test_fixed_occupation_dc_none_threshold_ranks_agree():
    # Exercises the real (un-monkeypatched) collective memory probe under multiple ranks.
    comm = MPI.COMM_WORLD
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    kwargs["basis"] = replace(kwargs["basis"], truncation_threshold=None)
    dc = fixed_occupation_dc(occupation=2.0, **kwargs)
    gathered = comm.gather(dc, root=0)
    if comm.rank == 0:
        for other in gathered[1:]:
            assert np.array_equal(dc, other), (dc, other)


def test_dc_search_applies_weighted_restrictions(monkeypatch):
    from impurityModel.ed.basis_restrictions import build_weighted_restrictions

    captured = _capture_prepared_bases(monkeypatch)
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    kwargs["basis"] = replace(kwargs["basis"], excitation_budget=2)
    fixed_occupation_dc(occupation=1.0, **kwargs)

    # Both searches now derive bath_states through prepare_solver_basis (the valence/conduction
    # split at solve time), not model.bath_states directly; recompute it the same way to get the
    # expected restriction list.
    model, basis = kwargs["model"], kwargs["basis"]
    sb = prepare_solver_basis(
        model.h0,
        model.dc,
        model.u4,
        model.impurity_orbitals,
        basis.nominal_occ,
        basis.mixed_valence,
        model.rot_to_spherical,
        0,
    )
    expected = build_weighted_restrictions(sb.bath_states, 2)
    assert captured and all(b.weighted_restrictions == expected for b in captured)

    captured.clear()
    kwargs, _ = common_kwargs(v=0.01, tau=1e-3)
    kwargs["basis"] = replace(kwargs["basis"], excitation_budget=2)
    fixed_peak_dc(peak_position=1.2, **kwargs)
    assert captured
    assert all(b.weighted_restrictions == expected for b in captured)


def test_dc_search_excitation_budget_binds():
    # excitation_budget=0 forbids any bath excitation away from the reference (both valence
    # baths filled) *within a sector's CIPSI expansion*: occupation=1 must still converge
    # (trivially, at the guess). occupation=2 is reachable regardless of the budget: the
    # alternate "free bath electron" sector (module docstring) find_ground_state_basis lands on
    # is a single, already-complete determinant (both impurity orbitals and both bath orbitals
    # filled) that needs zero admitted excitations to represent, so excitation_budget=0 does not
    # block it -- the search converges to the same dc as the unrestricted budget
    # (test_fixed_occupation_dc_increases_occupation).
    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2)
    kwargs["basis"] = replace(kwargs["basis"], excitation_budget=0)
    dc = fixed_occupation_dc(occupation=1.0, **kwargs)
    np.testing.assert_allclose(dc, dc_guess, atol=1e-8)

    kwargs, _ = common_kwargs(v=0.3, tau=1e-2, dc_scale=0.5)
    kwargs["basis"] = replace(kwargs["basis"], excitation_budget=0)
    dc = fixed_occupation_dc(occupation=2.0, **kwargs)
    np.testing.assert_allclose(dc[0, 0].real, 2.5, atol=0.05)


def test_dc_search_chain_restrict_forwarded(monkeypatch):
    captured = _capture_prepared_bases(monkeypatch)
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    kwargs["basis"] = replace(kwargs["basis"], chain_restrict=False)
    fixed_occupation_dc(occupation=1.0, **kwargs)
    assert captured and captured[0].chain_restrict is False

    captured.clear()
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    kwargs["basis"] = replace(kwargs["basis"], chain_restrict=True)
    fixed_occupation_dc(occupation=1.0, **kwargs)
    assert captured and captured[0].chain_restrict is True


@pytest.mark.mpi
def test_fixed_occupation_dc_self_consistent_ranks_agree():
    # The self-consistent search targets the DFT reference occupation, computed once from the
    # replicated raw h0 (deterministic NumPy, identical on every rank); the interacting
    # occupation per trial mu must stay rank-invariant for the same reason as the explicit path.
    comm = MPI.COMM_WORLD
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    dc = fixed_occupation_dc(**kwargs)
    gathered = comm.gather(dc, root=0)
    if comm.rank == 0:
        for other in gathered[1:]:
            assert np.array_equal(dc, other), (dc, other)


def _selfenergy_gs_occupation(model, basis_opts, dc):
    """Independently recompute the selfenergy-path ground-state impurity occupation at ``dc``.

    Mirrors what ``calc_selfenergy``/``calc_gs`` do (``prepare_solver_basis`` ->
    ``find_ground_state_basis`` -> thermal rho over the resulting eigenstates), without going
    through any of ``fixed_occupation_dc``'s own machinery (its sector cache, its per-mu HF
    reseeding), so this is a genuine external check of DC <-> GS parity rather than a
    tautological re-check of the same code path.
    """
    sb = prepare_solver_basis(
        model.h0,
        dict(tensors_to_operator(dc)),
        model.u4,
        model.impurity_orbitals,
        basis_opts.nominal_occ,
        basis_opts.mixed_valence,
        model.rot_to_spherical,
        0,
    )
    impurity_indices = [orb for blocks in sb.impurity_orbitals.values() for block in blocks for orb in block]
    gs_basis = find_ground_state_basis(
        h_op=sb.h,
        impurity_orbitals=sb.impurity_orbitals,
        bath_states=sb.bath_states,
        N0=sb.nominal_occ,
        mixed_valence=sb.mixed_valence,
        tau=basis_opts.tau,
        chain_restrict=basis_opts.chain_restrict,
        dense_cutoff=1000,
        spin_flip_dj=basis_opts.spin_flip_dj,
        comm=None,
        verbose=False,
        truncation_threshold=basis_opts.truncation_threshold,
        slaterWeightMin=basis_opts.slater_weight_min,
    )
    solver = CIPSISolver(gs_basis)
    solver.expand(sb.h, dense_cutoff=1000, de2_min=1e-8, slaterWeightMin=basis_opts.slater_weight_min)
    energy_cut = -basis_opts.tau * np.log(1e-4)
    es, psis = solver.get_eigenvectors(
        sb.h,
        num_wanted=10,
        max_energy=energy_cut,
        dense_cutoff=1000,
        slaterWeightMin=basis_opts.slater_weight_min,
        psi_refs=solver.psi_refs,
    )
    rhos = build_density_matrices(gs_basis, psis, impurity_indices, impurity_indices)
    rho = thermal_average_scale_indep(es, rhos, basis_opts.tau)
    return float(np.real(np.trace(rho)))


def test_fixed_occupation_dc_matches_independent_selfenergy_gs():
    # DC <-> GS parity: this is the actual bug the whole rework exists to fix (the DC search and
    # calc_selfenergy previously disagreed on the sector, e.g. NiO's DC search reporting
    # N=8.8411 while the selfenergy GS sat at N~7). dc_scale=0.5 with target=2.0 crosses the
    # charge-transfer point (dc > 6, see the module docstring), so the sector genuinely changes
    # across the mu scan -- not a trivial case where any basis would agree.
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2, dc_scale=0.5)
    target = 2.0
    dc = fixed_occupation_dc(occupation=target, **kwargs)
    n_selfenergy = _selfenergy_gs_occupation(kwargs["model"], kwargs["basis"], dc)
    np.testing.assert_allclose(n_selfenergy, target, atol=2e-2)


def test_reference_far_from_nominal_warns(capsys):
    """The runaway-CSC signature: a reference that is grossly wrong without being pinned.

    The saturation check fires only at exactly 0 or a full shell, so an archive iteration whose
    DFT impurity level has run away reports (measured) n0 = 1.5446 against a nominal 8 and the
    search converges on it silently, returning a dc 3.14 Ry (~43 eV) from the guess. This dimer
    stands in for that: the raw-h0 filling is N0 = 2 while the model is set up at a nominal 1,
    one full charge state away.
    """
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    assert kwargs["basis"].nominal_occ == {0: 1}
    fixed_occupation_dc(verbosity=1, **kwargs)
    out = capsys.readouterr().out
    if MPI.COMM_WORLD.rank != 0:
        assert "different charge state" not in out, out
        return
    # This dimer's reference *is* the full shell, so saturation is the more specific diagnosis
    # and must win -- the two warnings are mutually exclusive by construction.
    assert "saturated at the full impurity shell" in out, out
    assert "different charge state" not in out, out


def test_the_two_reference_warnings_are_mutually_exclusive():
    """Saturation is the specific case; the nominal gap is the catch-all. Never both."""
    from impurityModel.ed.dc_reference import _warn_if_reference_far_from_nominal, _warn_if_reference_saturated

    if MPI.COMM_WORLD.rank != 0:
        pytest.skip("both helpers print on rank 0 only")
    # Saturated at the full shell of 10, nominal 8: saturation fires.
    assert _warn_if_reference_saturated(10.0, 10, "advice.", rank=0) is True
    # The runaway: 1.54 of 10 is neither empty nor full, so saturation stays silent and the
    # nominal-gap check is the only thing that catches it.
    assert _warn_if_reference_saturated(1.5446, 10, "advice.", rank=0) is False
    assert _warn_if_reference_far_from_nominal(1.5446, 8, "advice.", rank=0) is True
    # And a healthy covalent deviation trips neither: NiO's own n0 = 8.6258 against nominal 8.
    assert _warn_if_reference_saturated(8.6258, 10, "advice.", rank=0) is False
    assert _warn_if_reference_far_from_nominal(8.6258, 8, "advice.", rank=0) is False


def test_reference_is_gap_protected_not_nominal_total_matched():
    """R0 decision, pinned: the reference is a genuine grand-canonical fill, not a fill
    constrained to match the many-body basis's *nominal* total (bath-valence-count + N0).

    Investigated on the real NiO archive (nio_15, iteration 1): the raw h0 has an actual
    one-body gap, (-0.0154, +0.0209) Ry, that mu=0 sits well inside, giving a robust,
    gap-protected fill of 148 electrons (n_imp = 8.6252). The ED's own nominal total (142
    valence-classified bath orbitals + 8 nominal impurity = 150) does *not* match that
    gap-fill count. Forcing a match (filling exactly 150 single-particle levels) pushes mu
    onto a fragile 4-fold near-degenerate manifold (66% bath character) sitting right at the
    gap's upper edge -- an artifact of the coarse per-orbital valence/conduction
    classification (each bath orbital classified solely by the sign of its own diagonal
    energy, ignoring hybridization-induced level repulsion), not a more correct number.
    Decision: keep the gap-protected grand-canonical fill (see the module docstring of
    dc_reference.py); do not "correct" the reference to track the nominal total.

    This toy reproduces the same structure at far smaller scale: one impurity orbital and
    two bath orbitals, so that the true one-body gap-fill total (1 electron, almost entirely
    of bath character) differs from the total a naive "bath valence count + nominal N0"
    bookkeeping would assume (2: the one valence-classified bath orbital plus a nominal
    impurity occupation of 1). The reference must follow the gap, not the nominal count.
    """
    from impurityModel.ed.dc_reference import _noninteracting_impurity_occupation

    # Impurity orbital 0 sits ABOVE the Fermi level; bath orbital 1 sits deep below it
    # (always occupied, "valence" by the sign convention); bath orbital 2 sits far above it
    # and decoupled ("conduction"). Naive nominal total: 1 (valence bath) + 1 (nominal
    # impurity, as if this were a d1-like config) = 2. True one-body spectrum: hybridizing
    # orbitals 0 and 1 gives one bonding level (mostly bath character) below the gap and one
    # antibonding level (mostly impurity character) above it; orbital 2 stays above the gap,
    # untouched. Only the bonding level is occupied at mu=0 -- a genuine gap-fill total of 1.
    e_imp, e_bath_valence, e_bath_conduction, v, tau = 0.6, -3.0, 3.0, 0.5, 1e-2
    h0 = {
        ((0, "c"), (0, "a")): e_imp,
        ((1, "c"), (1, "a")): e_bath_valence,
        ((2, "c"), (2, "a")): e_bath_conduction,
        ((0, "c"), (1, "a")): v,
        ((1, "c"), (0, "a")): v,
    }
    h = np.array(
        [[e_imp, v, 0.0], [v, e_bath_valence, 0.0], [0.0, 0.0, e_bath_conduction]],
        dtype=complex,
    )
    energies = np.linalg.eigvalsh(h)
    # A genuine gap straddled by mu=0: exactly one level below zero, both others well above.
    assert sorted(energies)[0] < 0.0 < sorted(energies)[1], energies
    gap = sorted(energies)[1] - sorted(energies)[0]
    assert gap > 100 * tau, "the toy must be deep in the gap at this tau, or the fill is not robust"

    n0 = _noninteracting_impurity_occupation(h0, impurity_indices=[0], n_spin_orbitals=3, tau=tau)
    total_at_mu0 = float(np.sum(1.0 / (1.0 + np.exp(energies / tau))))

    # The gap-fill total is 1 (the bonding level only), not the naive nominal total of 2.
    assert np.isclose(total_at_mu0, 1.0, atol=1e-6), total_at_mu0
    # Most of that one electron is bath character (e_bath_valence << e_imp), so the impurity's
    # share is small -- nowhere near a "nominal" occupation of 1.
    assert 0.0 < n0 < 0.3, n0

