"""Does the gap criterion's edge carry impurity spectral weight? -- the diagnostic's own gate.

:mod:`impurityModel.test.support.dc_weights` measures, per charge sector and per eigenstate, the
impurity occupation and the Lehmann weight ``w_n = sum_d |<n|c_d^dagger|0>|^2``. It exists to
answer whether :func:`dc_criteria.fixed_gap_dc`'s ``omega_+-`` -- the *lowest* eigenvalue of the
``N +- 1`` sectors, taken with no impurity projection at all -- are poles of ``A_imp`` or states
invisible to it.

A measurement is not evidence until it reproduces a case whose answer is known independently, and
a previous campaign in this repository lost days to oracle bugs that only a closed-form check
would have caught. So the first test here is the two-level model solved on paper; the other two
are the two regimes the diagnostic has to tell apart.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.test.support.dc_weights import sector_weight_table

from .test_fixed_dc import EPS, U, charge_transfer_kwargs, common_kwargs

#: ``dc_scale`` the fixtures below are built at, so the analytic energies can name it.
DC = 0.5

#: The measurements here are serial-only (see :mod:`dc_weights`), and the MPI gate runs the whole
#: suite on **every** rank -- so without this every test below fails at ``-n 2`` and ``-n 3``, on
#: the non-root ranks where rank 0's summary does not show it. Not ``@pytest.mark.mpi``, which
#: means the opposite (run *only* under ``--with-mpi``).
serial_only = pytest.mark.skipif(
    MPI.COMM_WORLD.size > 1, reason="dc_weights measures determinant-support overlaps and is serial only"
)


def _table(kwargs, **overrides):
    # `comm` comes from the fixture's kwargs but stays overridable -- passing it in both places
    # is a TypeError for multiple values, which is how the distributed-guard test below managed
    # to fail for a reason other than the one it was checking.
    options = dict(mu=0.0, num_states=40, max_energy=1e3, comm=kwargs["comm"])
    options.update(overrides)
    return sector_weight_table(kwargs["model"], kwargs["basis"], kwargs["solver"], **options)


@serial_only
def test_the_weight_table_reproduces_the_two_level_model_solved_on_paper():
    """Signs, fermionic ordering and normalisation, against closed form.

    At hopping ``v -> 0`` the fixture is exactly solvable (see ``test_fixed_dc``'s module
    docstring): with the impurity at ``eps``, a Hubbard ``U`` and the bath filled and far below,

        omega_+ = E[2] - E[1] = eps + U - dc,      omega_- = E[1] - E[0] = eps - dc,

    the added/removed electron is entirely on the impurity (``delta = +-1``) and the whole
    addition and removal sum rules sit on that one state. Every quantity the diagnostic reports
    is pinned here at once -- an operator built with ``"a"``/``"c"`` swapped, a missing conjugate,
    or an unnormalised eigenvector each break a different line below.
    """
    kwargs, _dc_guess = common_kwargs(v=0.01, tau=1e-3, dc_scale=DC)
    result = _table(kwargs)

    assert result["omega_plus"] == pytest.approx(EPS + U - DC, abs=1e-3)
    assert result["omega_minus"] == pytest.approx(EPS - DC, abs=1e-3)
    # "Put the double counting in the middle of the gap" is eps + U/2 - dc, which the fixture is
    # built to make zero.
    assert result["centre"] == pytest.approx(0.0, abs=1e-3)

    addition = result["states"][result["n_center"] + 1]
    removal = result["states"][result["n_center"] - 1]
    assert addition[0]["delta"] == pytest.approx(1.0, abs=1e-3)
    assert removal[0]["delta"] == pytest.approx(-1.0, abs=1e-3)
    # The whole sum rule on the lowest state of each side: nothing else is reachable at v -> 0.
    assert addition[0]["weight_fraction"] == pytest.approx(1.0, abs=1e-3)
    assert removal[0]["weight_fraction"] == pytest.approx(1.0, abs=1e-3)

    # The seed lives inside the sector basis, and the enumeration exhausts the exact sum rule
    # (sum_n w_n = sum_d (1 - n_d) for addition, sum_d n_d for removal). Both are checks on the
    # *diagnostic*, not on the physics: without them a weight is a number with no denominator.
    for n in (result["n_center"] + 1, result["n_center"] - 1):
        assert result["representability"][n] == pytest.approx(1.0, abs=1e-3)
        assert result["enumerated_weight"][n] == pytest.approx(result["sum_rule"][n], rel=1e-3)


@serial_only
def test_the_weighted_and_unweighted_edges_agree_when_the_edge_is_impurity_like():
    """The regime in which ``fixed_gap_dc``'s shortcut is exactly right.

    With no conduction bath the added electron has nowhere to go but the impurity, so the lowest
    addition state *is* the impurity addition pole. Projecting on impurity weight must then
    change nothing -- if it did, the diagnostic would be manufacturing a correction where there
    is none, and every difference it reports elsewhere would be suspect.
    """
    kwargs, _dc_guess = common_kwargs(v=0.3, tau=1e-3, dc_scale=DC)
    result = _table(kwargs)

    assert result["omega_plus_w"] == result["omega_plus"]
    assert result["omega_minus_w"] == result["omega_minus"]
    assert result["centre_w"] == pytest.approx(result["centre"])
    addition = result["states"][result["n_center"] + 1]
    assert addition[0]["delta"] > 0.9, addition[0]
    assert addition[0]["weight_fraction"] > 0.9, addition[0]


@serial_only
def test_a_bath_like_edge_is_visible_as_a_minority_of_the_addition_weight():
    """The regime the criterion is blind to, and the number that makes it visible.

    ``charge_transfer_kwargs`` puts a conduction bath level just above ``E_F``, below where the
    impurity addition level ``eps + U - dc`` lands. The lowest addition state is then the bath
    level: mostly *not* on the impurity, and carrying a minority of the addition sum rule, while
    the state that carries the majority sits well above it. ``fixed_gap_dc`` fixes ``mu`` against
    the former and calls it the gap edge.

    Asserted as an ordering and a separation rather than against measured constants: the point is
    that the two states are distinguishable and in this order, not that they sit at particular
    energies of this particular fixture.
    """
    kwargs, _dc_guess = charge_transfer_kwargs()
    result = _table(kwargs)
    addition = result["states"][result["n_center"] + 1]

    lowest = addition[0]
    dominant = max(addition, key=lambda row: row["weight"])

    assert lowest["delta"] < 0.2, f"the lowest addition state should be bath-like: {lowest}"
    assert lowest["weight_fraction"] < 0.3, f"the lowest addition state should be a minority pole: {lowest}"
    assert dominant["delta"] > 0.7, f"the dominant pole should be impurity-like: {dominant}"
    assert dominant["weight_fraction"] > 0.5, f"the dominant pole should carry the sum rule: {dominant}"
    assert dominant["omega"] > lowest["omega"], (dominant, lowest)

    # The whole point, as one number: the criterion's edge and the impurity spectrum's centre of
    # gravity are far apart on a scale set by the gap the criterion thinks it is measuring.
    assert result["omega_plus_centroid"] - result["omega_plus"] > 0.5 * abs(result["omega_plus"])

    # The removal side of this same fixture is impurity-like, so the defect is one-sided here.
    # Stated because a diagnostic that flagged *both* sides on every model would be flagging the
    # model rather than the edge.
    removal = result["states"][result["n_center"] - 1]
    assert removal[0]["weight_fraction"] > 0.9, removal[0]


@serial_only
def test_the_enumerated_weight_is_normalised_against_the_exact_sum_rule():
    """A short window must show up as a small cumulative, never as a rescaled spectrum.

    Dividing each weight by the *enumerated* total instead of the exact sum rule made the
    charge-transfer fixture's bath level read 0.93 of the addition weight when its true share is
    0.15 -- the enumeration had found 16 % of the spectrum and the normalisation hid it. This
    pins the denominator: cut the window down and the fractions must *shrink*, not renormalise.
    """
    kwargs, _dc_guess = charge_transfer_kwargs()
    narrow = _table(kwargs, num_states=2, max_energy=1.0)
    addition = narrow["states"][narrow["n_center"] + 1]

    assert narrow["enumerated_weight"][narrow["n_center"] + 1] < 0.5 * narrow["sum_rule"][narrow["n_center"] + 1]
    assert sum(row["weight_fraction"] for row in addition) < 0.5
    assert addition[0]["weight_fraction"] < 0.3, addition[0]


@serial_only
def test_the_exact_moments_agree_with_the_enumerated_spectrum_where_it_is_complete():
    """``M_1/M_0`` by completeness, against the same number summed state by state.

    The moments close analytically (``M_0^+ = 1 - n_d``, ``M_1^+ = <0|c H c^dag|0> - E_0 M_0^+``),
    so they cost one ``H`` apply per orbital and are exact. The enumerated centroid is not: it
    inherits whatever the window missed. Where the enumeration *is* complete the two must agree,
    which is what makes the moments trustworthy where it is not -- and on NiO's removal side it
    reaches 55 % at best.
    """
    kwargs, _dc_guess = common_kwargs(v=0.01, tau=1e-3, dc_scale=DC)
    result = _table(kwargs)

    # Single-pole limit: the centre of gravity of each side *is* the analytic pole.
    assert result["centroid_plus"] == pytest.approx(EPS + U - DC, abs=1e-3)
    assert result["centroid_minus"] == pytest.approx(EPS - DC, abs=1e-3)
    assert result["centre_moment"] == pytest.approx(0.0, abs=1e-3)
    # M_0 is the sum rule, exactly -- the same number the enumeration is normalised against.
    for tag, n in (("plus", result["n_center"] + 1), ("minus", result["n_center"] - 1)):
        assert result[f"m0_{tag}"] == pytest.approx(result["sum_rule"][n], rel=1e-6)
    # And where the enumeration closes, so do the centroids.
    assert result["centroid_plus"] == pytest.approx(result["omega_plus_centroid"], abs=1e-3)


def test_the_diagnostic_refuses_to_run_distributed():
    """Serial-only, and it says so rather than reducing something wrong.

    The overlaps and the representability check are sums over determinant *supports*, which have
    no distributed implementation here. Determinants are hash-distributed so one is possible, but
    a silent partial sum on each rank is the failure this guard exists to prevent -- and the last
    pass through this code shipped an MPI deadlock, so the bar for adding collectives is high.
    """
    from mpi4py import MPI

    if MPI.COMM_WORLD.size == 1:
        pytest.skip("needs more than one rank to exercise the guard")
    kwargs, _dc_guess = common_kwargs(v=0.01, tau=1e-3, dc_scale=DC)
    with pytest.raises(RuntimeError, match="serial only"):
        _table(kwargs, comm=MPI.COMM_WORLD)


@serial_only
def test_the_analytic_edges_move_with_the_double_counting_shift():
    """``omega_+-`` must respond to ``mu`` the way the model says, or the diagnostic is not
    measuring the criterion's observable.

    Both edges are ``eps (+ U) - dc`` with ``dc = dc_guess + mu``, so both shift by ``-mu`` and
    the gap *width* does not move at all. This is the Hellmann-Feynman slope the record reports
    as ``chi``, at the one point where its exact value is known: ``delta_+ = delta_- = 1``, so
    ``d(centre)/dmu = -1``.
    """
    kwargs, _dc_guess = common_kwargs(v=0.01, tau=1e-3, dc_scale=DC)
    step = 0.1
    at_zero = _table(kwargs)
    shifted = _table(kwargs, mu=step)

    assert shifted["omega_plus"] == pytest.approx(at_zero["omega_plus"] - step, abs=2e-3)
    assert shifted["omega_minus"] == pytest.approx(at_zero["omega_minus"] - step, abs=2e-3)
    width_zero = at_zero["omega_plus"] - at_zero["omega_minus"]
    width_step = shifted["omega_plus"] - shifted["omega_minus"]
    assert width_step == pytest.approx(width_zero, abs=2e-3)
    chi = (shifted["centre"] - at_zero["centre"]) / step
    assert chi == pytest.approx(-1.0, abs=2e-2)


@serial_only
def test_solve_sector_and_calc_energy_agree_on_the_ground_state_energy():
    """The extraction that made this diagnostic possible did not move the number.

    ``groundstate.solve_sector`` is ``calc_energy``'s body with the eigenvectors kept; the latter
    is now the former plus ``min`` plus a broadcast. Different truncation conventions between the
    two would make every energy in this module incomparable with the criterion's own.
    """
    from mpi4py import MPI

    from impurityModel.ed.dc_criteria import _per_group_occupation, _prepare_sector_context
    from impurityModel.ed.groundstate import calc_energy, solve_sector

    kwargs, _dc_guess = common_kwargs(v=0.01, tau=1e-3, dc_scale=DC)
    ctx = _prepare_sector_context(
        kwargs["model"], kwargs["basis"], kwargs["solver"], comm=MPI.COMM_WORLD, verbosity=0, memory_label="test"
    )
    h_op = ctx.shifted_h(0.0)
    occ = _per_group_occupation(ctx.nominal_total, ctx.impurity_orbitals, ctx.h_solver_matrix)
    args = (
        h_op,
        ctx.impurity_orbitals,
        ctx.bath_states,
        occ,
        ctx.mixed_valence,
        ctx.tau,
        ctx.chain_restrict,
        ctx.spin_flip_dj,
        ctx.dense_cutoff,
    )
    common = dict(
        comm=MPI.COMM_WORLD,
        verbose=False,
        truncation_threshold=ctx.truncation_threshold,
        slaterWeightMin=ctx.slater_weight_min,
        weighted_restrictions=ctx.weighted_restrictions,
        frozen_occupations=ctx.frozen_occupations,
    )
    energy, _basis = calc_energy(*args, **common)
    es, _psis, _basis = solve_sector(*args, **common)
    assert float(np.min(es)) == pytest.approx(energy, abs=1e-10)
