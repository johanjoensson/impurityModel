"""
Tests for the fixed-peak and fixed-occupation double counting criteria.

Analytically solvable model: two impurity spin-orbitals at energy eps with a
Hubbard interaction U, weakly coupled (hopping v) to two valence bath
spin-orbitals at energy eps_b. With the double counting dc entering the
Hamiltonian as -dc * n_imp:

    E[N_imp = 0] = 2 eps_b
    E[N_imp = 1] = (eps - dc) + 2 eps_b          + O(v^2)
    E[N_imp = 2] = 2 (eps - dc) + U + 2 eps_b    + O(v^2)

so the electron-addition peak sits at E[2] - E[1] = eps + U - dc and the
electron-removal peak at E[1] - E[0] = eps - dc. The total electron number is
conserved (N_imp + N_bath = 3 with N0 = 1), and the impurity occupation
switches from 1 to 2 through charge transfer when eps - dc + U < eps_b, i.e.
for dc > eps + U - eps_b = 6.
"""

from dataclasses import replace

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.model import BasisOptions, ImpurityModel, SolverOptions
from impurityModel.ed.selfenergy import fixed_occupation_dc, fixed_peak_dc

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


def test_fixed_peak_dc_multiple_groups_raises():
    kwargs, _ = common_kwargs(v=0.01, tau=1e-3, dc_scale=1.0)
    kwargs["basis"] = replace(kwargs["basis"], nominal_occ={0: 1, 1: 1})
    with pytest.raises(ValueError, match="single impurity group"):
        fixed_peak_dc(peak_position=1.0, **kwargs)


def test_fixed_occupation_dc_already_converged():
    # At the guess (dc=0.5, well below the dc=6 charge-transfer point) the impurity already
    # holds one electron; requesting occupation 1 must return the guess unchanged (mu=0).
    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2)
    dc = fixed_occupation_dc(occupation=1.0, **kwargs)
    np.testing.assert_allclose(dc, dc_guess, atol=1e-8)


def test_fixed_occupation_dc_increases_occupation():
    # Requesting two electrons on the impurity requires pushing the doubly
    # occupied impurity below the bath: dc > eps + U - eps_b = 6.
    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2, dc_scale=0.5)
    dc = fixed_occupation_dc(occupation=2.0, **kwargs)
    assert_uniform_shift(dc, dc_guess)
    assert dc[0, 0].real > 6.0, dc


def test_fixed_occupation_dc_decreases_occupation():
    # A guess of 7 puts two electrons on the impurity; requesting one electron
    # must bring the double counting back below the charge-transfer point.
    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2, dc_scale=7.0)
    dc = fixed_occupation_dc(occupation=1.0, **kwargs)
    assert_uniform_shift(dc, dc_guess)
    assert dc[0, 0].real < 6.0, dc


def test_fixed_occupation_dc_unreachable_raises():
    # Only two bath spin-orbitals and three electrons in total: the impurity
    # occupation cannot drop below one.
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2, dc_scale=0.5)
    with pytest.raises(RuntimeError, match="Could not bracket"):
        fixed_occupation_dc(occupation=0.2, **kwargs)


def test_noninteracting_impurity_occupation_matches_fermi_fill():
    # The h_loc-derived target is the Fermi-filled (mu=0) occupation of the full
    # non-interacting h0. Build a 1-impurity / 1-bath cluster with an impurity
    # level poking above the Fermi level so the answer is genuinely fractional,
    # and compare against an independent per-eigenvector Fermi sum.
    from impurityModel.ed.double_counting import _noninteracting_impurity_occupation

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
    # level eps = -1 already sits below E_F, so N0 = 2 (full shell) and the search must cross
    # the charge-transfer point dc > eps + U - eps_b = 6, exactly like an explicit
    # occupation=2.0 request.
    from impurityModel.ed.double_counting import _noninteracting_impurity_occupation

    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2)
    model, basis = kwargs["model"], kwargs["basis"]

    n0_ref = _noninteracting_impurity_occupation(model.h0, [0, 1], 4, basis.tau)
    assert np.isclose(n0_ref, 2.0, atol=1e-2), n0_ref

    dc_auto = fixed_occupation_dc(**kwargs)
    assert_uniform_shift(dc_auto, dc_guess)
    assert dc_auto[0, 0].real > 6.0, dc_auto


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
    # converge to the same physics as from any other guess: past the charge-transfer point,
    # dc > 6.
    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2, dc_scale=-3.0)
    dc = fixed_occupation_dc(**kwargs)
    assert_uniform_shift(dc, dc_guess)
    assert dc[0, 0].real > 6.0, dc


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


def _capture_prepared_bases(monkeypatch):
    """Monkeypatch ``_prepare_dc_solver`` to record every ``Basis`` it builds.

    Returns the list the ``Basis`` objects are appended to, in call order.
    """
    import impurityModel.ed.double_counting as dc_module

    captured = []
    original = dc_module._prepare_dc_solver

    def wrapper(*args, **kwargs):
        basis, solver = original(*args, **kwargs)
        captured.append(basis)
        return basis, solver

    monkeypatch.setattr(dc_module, "_prepare_dc_solver", wrapper)
    return captured


def test_fixed_occupation_dc_derives_cap_when_threshold_is_none(monkeypatch):
    import impurityModel.ed.double_counting as dc_module

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
    import impurityModel.ed.double_counting as dc_module

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

    # One probe (not one per sector basis), and both sector bases carry its cap.
    assert len(calls) == 1
    assert calls[0]["safety"] == pytest.approx(dc_module.DEFAULT_MEMORY_SAFETY / 2)
    assert len(captured) == 2
    assert all(b.truncation_threshold == 50 for b in captured)


def test_explicit_threshold_skips_memory_probe(monkeypatch):
    import impurityModel.ed.double_counting as dc_module

    calls = []

    def fake_suggest(*args, **kwargs):
        calls.append(1)
        return 50

    monkeypatch.setattr(dc_module, "suggest_truncation_threshold", fake_suggest)

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
    from impurityModel.ed.double_counting import _normalize_dc_orbitals

    captured = _capture_prepared_bases(monkeypatch)
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    kwargs["basis"] = replace(kwargs["basis"], excitation_budget=2)
    fixed_occupation_dc(occupation=1.0, **kwargs)

    _, bath_states = _normalize_dc_orbitals(kwargs["model"].impurity_orbitals, kwargs["model"].bath_states)
    expected = build_weighted_restrictions(bath_states, 2)
    assert captured and captured[0].weighted_restrictions == expected

    captured.clear()
    kwargs, _ = common_kwargs(v=0.01, tau=1e-3)
    kwargs["basis"] = replace(kwargs["basis"], excitation_budget=2)
    fixed_peak_dc(peak_position=1.2, **kwargs)
    assert len(captured) == 2
    assert all(b.weighted_restrictions == expected for b in captured)


def test_dc_search_excitation_budget_binds():
    # excitation_budget=0 forbids any bath excitation away from the reference (both valence
    # baths filled): with 3 total electrons and 2 bath spin-orbitals, the impurity is then
    # pinned to exactly N_imp=1 for every basis determinant -- occupation=1 must still converge
    # (trivially, at the guess), but occupation=2 becomes unreachable, unlike with the default
    # (non-binding) budget where test_fixed_occupation_dc_increases_occupation reaches it.
    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2)
    kwargs["basis"] = replace(kwargs["basis"], excitation_budget=0)
    dc = fixed_occupation_dc(occupation=1.0, **kwargs)
    np.testing.assert_allclose(dc, dc_guess, atol=1e-8)

    kwargs, _ = common_kwargs(v=0.3, tau=1e-2, dc_scale=0.5)
    kwargs["basis"] = replace(kwargs["basis"], excitation_budget=0)
    with pytest.raises(RuntimeError, match="Could not bracket"):
        fixed_occupation_dc(occupation=2.0, **kwargs)


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
