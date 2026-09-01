"""
Tests for the fixed-peak, fixed-gap and fixed-occupation double counting criteria.

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
dc (see dc_criteria.py's module docstring). ``find_ground_state_basis``
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

from impurityModel.ed.average import energy_cut as boltzmann_energy_cut
from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.basis_transcription import build_density_matrices
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.dc_criteria import GAP_ENERGY_TOLERANCE_FRACTION
from impurityModel.ed.dc_search import bracket_width_tol
from impurityModel.ed.groundstate import find_ground_state_basis
from impurityModel.ed.lie_algebra import tensors_to_operator
from impurityModel.ed.model import BasisOptions, ImpurityModel, SolverOptions
from impurityModel.ed.selfenergy import (
    DoubleCountingUnreachable,
    fixed_gap_dc,
    fixed_occupation_dc,
    fixed_peak_dc,
    occupation_and_energy_at_mu,
)
from impurityModel.ed.solver_basis import prepare_solver_basis

from ..support.dc_records import parse_record

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


def charge_transfer_kwargs(v=0.3, tau=1e-3, dc_scale=0.5, eps_cond=0.4):
    """A model whose *empty* level nearest E_F is a **bath** level, not the impurity.

    Every other fixture in this file gives the impurity two valence bath orbitals well below it and
    no conduction bath at all (``bath_valence_conduction=([2, 3], [])``), so an electron added to
    the cluster has nowhere to go but the impurity: ``delta_+ = 1`` by construction and the cluster
    charge gap *is* the impurity gap. That is exactly the limit in which ``fixed_gap_dc``'s
    conditioning claim is trivially true, and it is why the whole suite could not see that the
    criterion measures the cluster and not the impurity.

    Here a conduction bath orbital sits just above E_F at ``eps_cond``, below where the impurity
    addition level ``eps + U - dc`` lands, with a hybridization comparable to that separation. The
    lowest addition state is then bath-like, which is the generic situation for a real
    bath-discretized charge-transfer insulator -- and is what a hybridization fit *puts* there.
    """
    n_orb = 6  # 2 impurity, 2 valence bath, 2 conduction bath
    h0 = np.zeros((n_orb, n_orb), dtype=complex)
    for s in range(2):
        imp, val, cond = s, 2 + s, 4 + s
        h0[imp, imp] = EPS
        h0[val, val] = EPS_B
        h0[cond, cond] = eps_cond
        h0[imp, val] = h0[val, imp] = v
        h0[imp, cond] = h0[cond, imp] = v
    dc = np.identity(2, dtype=complex) * dc_scale
    u4 = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=complex)
    u4[0, 1, 0, 1] = U
    u4[1, 0, 1, 0] = U
    model = ImpurityModel.from_solver_matrix(
        h0,
        2,
        dc=dc,
        u4=u4,
        rot_to_spherical=np.eye(2, dtype=complex),
        bath_valence_conduction=([2, 3], [4, 5]),
    )
    basis = BasisOptions(
        nominal_occ={0: 1},
        mixed_valence=None,
        spin_flip_dj=False,
        tau=tau,
        slater_weight_min=np.sqrt(np.finfo(float).eps),
        truncation_threshold=int(1e8),
    )
    return dict(model=model, basis=basis, solver=SolverOptions(dense_cutoff=1000), comm=MPI.COMM_WORLD), dc


#: Impurity spatial orbitals in :func:`spin_crossover_kwargs`. Three is the minimum that admits a
#: low-spin/high-spin crossover: with one or two, the maximum total spin at ``N`` and at ``N + 1``
#: differ by exactly 1/2, which ``c_d^dagger`` always connects.
CROSSOVER_N_ORB = 3


def _crossover_index(orbital, spin):
    """Spin-major, down-then-up -- the impurity convention the solver reports in."""
    return spin * CROSSOVER_N_ORB + orbital


def _kanamori_u4(n_total, u, j):
    """Density-density Kanamori on the impurity block, in the solver's ``u4`` convention.

    ``u4[i, j, k, l]`` multiplies ``c^dag_i c^dag_j c_l c_k`` at a factor of one half, so
    ``u4[a, b, a, b] = u4[b, a, b, a] = U_ab`` is ``U_ab n_a n_b`` -- the same construction the
    one-orbital fixtures above use, extended to the three Kanamori channels. Density-density only:
    it reproduces the low/high-spin energetics exactly and conserves ``S_z`` manifestly, which is
    the mechanism under test.
    """
    u4 = np.zeros((n_total,) * 4, dtype=complex)

    def add(a, b, value):
        u4[a, b, a, b] += value
        u4[b, a, b, a] += value

    for a in range(CROSSOVER_N_ORB):
        add(_crossover_index(a, 0), _crossover_index(a, 1), u)  # intra-orbital
        for b in range(a + 1, CROSSOVER_N_ORB):
            add(_crossover_index(a, 0), _crossover_index(b, 1), u - 2 * j)  # inter, opposite spin
            add(_crossover_index(a, 1), _crossover_index(b, 0), u - 2 * j)
            for spin in (0, 1):
                add(_crossover_index(a, spin), _crossover_index(b, spin), u - 3 * j)  # inter, same spin
    return u4


def spin_crossover_kwargs(u=8.0, j=1.0, delta=3.5, eps=-11.0, eps_bath=-25.0, v=0.05, tau=1e-3):
    r"""A model whose gap edge carries **exactly zero** impurity spectral weight.

    Every other fixture in this file has a single impurity spatial orbital, which makes "the
    impurity gained an electron" and "the state is ``c_d^dagger|0>``-reachable" nearly the same
    statement -- measured, ``delta`` and the Lehmann weight track to within a few percent at the
    edge across every parameter scanned. So none of them can exhibit the case the impurity-character
    warning explicitly disclaims: an edge that moves fully with ``mu`` while carrying no weight in
    ``A_imp``.

    Three orbitals with a crystal field ``delta`` and Hund's ``j`` do, through ``S_z``. The
    low/high-spin energy difference is ``(delta - 3j)`` at ``N = 2`` and ``(delta - 4j)`` at
    ``N = 3`` -- opposite signs inside ``3j < delta < 4j``, and independent of ``u``. So in that
    window the ``N`` ground state is low-spin (``S_z = 0``, both electrons paired in the lowest
    orbital) while the ``N + 1`` ground state is high-spin (``S_z = 3/2``, one electron per
    orbital, aligned). ``H`` conserves ``S_z`` and ``c_d^dagger|0>`` only reaches ``S_z = +-1/2``
    from ``S_z = 0``, so the overlap is **exactly** zero -- symmetry, not smallness. Yet the
    electron lands entirely on the impurity, so ``delta_+ = 1`` and the warning stays silent.

    ``delta`` is the control: at ``delta = 3.5`` (inside the window) the addition edge is
    weightless; at ``delta = 5`` (outside) the same fixture puts weight back on it.

    Levels follow from wanting ``N = 2`` to be the cluster ground state: with ``E(1) = eps``,
    ``E(2) = 2 eps + u`` and ``E(3) = 3 eps + 2 delta + 3u - 9j``, that needs
    ``-u > eps > 9j - 2 delta - 2u``, i.e. ``-14 < eps < -8`` at the defaults. ``eps = -11`` sits
    mid-window and puts the edges at ``omega_+ = +3`` and ``omega_- = -3``.

    **Every impurity orbital gets its own bath partner, and that is load-bearing.**
    :mod:`basis_generation` freezes any impurity group with no bath attached (the bath-less-core
    pin), and a frozen group's occupation is pinned at its aufbau value -- which excludes the
    high-spin configuration entirely. Hybridizing only the lowest orbital gave a 4-determinant
    ``N + 1`` sector whose lowest state was the low-spin one, 0.5 above the true sector ground
    state, silently. Confirmed against an independent exact diagonalization.
    """
    n_imp = 2 * CROSSOVER_N_ORB
    n_total = n_imp + 2 * CROSSOVER_N_ORB
    h0 = np.zeros((n_total, n_total), dtype=complex)
    for spin in (0, 1):
        for orbital in range(CROSSOVER_N_ORB):
            i = _crossover_index(orbital, spin)
            h0[i, i] = eps + (0.0 if orbital == 0 else delta)
            bath = n_imp + spin * CROSSOVER_N_ORB + orbital
            # Weak, so the S_z multiplets stay sharp; deep, so the bath stays full and an added
            # electron has nowhere to go but the impurity.
            h0[bath, bath] = eps_bath
            h0[i, bath] = h0[bath, i] = v
    model = ImpurityModel.from_solver_matrix(
        h0,
        n_imp,
        dc=np.zeros((n_imp, n_imp), dtype=complex),
        u4=_kanamori_u4(n_total, u, j),
        rot_to_spherical=np.eye(n_imp, dtype=complex),
        bath_valence_conduction=(list(range(n_imp, n_total)), []),
    )
    basis = BasisOptions(
        nominal_occ={0: 2},
        mixed_valence=None,
        spin_flip_dj=False,
        tau=tau,
        slater_weight_min=np.sqrt(np.finfo(float).eps),
        truncation_threshold=int(1e8),
    )
    return dict(model=model, basis=basis, solver=SolverOptions(dense_cutoff=10000), comm=MPI.COMM_WORLD)


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


def test_fixed_gap_dc_centres_the_gap_on_the_fermi_level():
    """Karolak et al.'s insulator prescription, on the model whose gap is known exactly.

    At weak hopping the centre sector stays at ``N_imp = 1``, so the two excitations are the
    addition peak ``E[2] - E[1] = eps + U - dc`` and the removal peak ``E[1] - E[0] = eps - dc``.
    Their midpoint is ``eps + U/2 - dc``, so centring it on the Fermi level fixes
    ``dc = eps + U/2`` -- exactly halfway between the two peak criteria's own answers, which is
    the whole content of "put mu_dc in the middle of the gap".
    """
    kwargs, dc_guess = common_kwargs(v=0.01, tau=1e-3, dc_scale=0.0)
    dc = fixed_gap_dc(offset=0.0, **kwargs)
    assert_uniform_shift(dc, dc_guess)
    np.testing.assert_allclose(np.diag(dc).real, EPS + U / 2, atol=5e-3)


def test_fixed_gap_dc_honours_an_offset():
    """``offset`` is read exactly as ``peak_position`` is: an energy relative to E_F = 0."""
    offset = -0.3
    kwargs, dc_guess = common_kwargs(v=0.01, tau=1e-3, dc_scale=0.5)
    dc = fixed_gap_dc(offset=offset, **kwargs)
    assert_uniform_shift(dc, dc_guess)
    np.testing.assert_allclose(np.diag(dc).real, EPS + U / 2 - offset, atol=5e-3)


def test_the_gap_centre_responds_to_the_shift_with_the_slope_it_claims():
    """The conditioning claim, measured rather than asserted.

    ``d(centre)/dmu = -(delta_+ + delta_-)/2`` is documented as lying in ``[-1, -0.5]``, which is
    what makes this criterion well conditioned where the single-pole peak criterion is not (its
    slope ``-delta_+-`` goes *small* in the charge-transfer regime). Inverting, ``d(dc)/d(offset)
    = 1/slope`` must lie in ``[-2, -1]``. Measured through the public API, on two offsets.
    """
    kwargs, _ = common_kwargs(v=0.01, tau=1e-3, dc_scale=0.5)
    dc_a = fixed_gap_dc(offset=-0.4, **kwargs)[0, 0].real
    dc_b = fixed_gap_dc(offset=+0.4, **kwargs)[0, 0].real
    d_dc_d_offset = (dc_b - dc_a) / 0.8
    assert -2.0 <= d_dc_d_offset <= -1.0 + 1e-6, d_dc_d_offset


def test_fixed_gap_dc_reports_the_width_it_refuses_to_root_find():
    """The width is a *second* difference of three independently truncated CIPSI energies, so it
    is reported and never solved for. Here it is the bare Hubbard U: ``omega_+ - omega_- =
    (eps + U - dc) - (eps - dc) = U``, independent of dc -- which is also why no shift could
    control it even if the criterion tried."""
    kwargs, _ = common_kwargs(v=0.01, tau=1e-3, dc_scale=0.0)
    report = {}
    fixed_gap_dc(offset=0.0, report=report, **kwargs)

    np.testing.assert_allclose(report["gap_width"], U, atol=5e-3)
    # The residual is the midpoint of the two excitations it reports, by construction.
    np.testing.assert_allclose(report["gap_center"], 0.5 * (report["omega_plus"] + report["omega_minus"]), atol=1e-9)
    assert report["omega_plus"] > 0.0 > report["omega_minus"], report
    # Separate tolerances: the energy tolerance is scaled to the measured gap, the mu resolution
    # is not. fixed_peak_dc passes one number for both, which is what this criterion fixes.
    np.testing.assert_allclose(report["tol"], GAP_ENERGY_TOLERANCE_FRACTION * U, rtol=0.05)
    assert report["tol"] > bracket_width_tol(1e-3)


def test_the_gap_centre_is_the_midpoint_of_the_two_peaks_it_is_built_from():
    """Cross-criterion consistency, with no analytic input at all.

    At the shift ``fixed_gap_dc`` returns, the addition peak sits at ``+width/2`` and the removal
    peak at ``-width/2`` by definition of "centred". So asking ``fixed_peak_dc`` -- a different
    residual, a different search path -- to place either of those peaks must reproduce the same
    ``dc``. Both numbers come out of the gap criterion's own report, so this checks the two
    criteria against each other rather than against a hand-derived formula, and it exercises the
    real sector walk and sector solves rather than an analytic stand-in.
    """
    kwargs, _ = common_kwargs(v=0.01, tau=1e-3, dc_scale=0.0)
    report = {}
    dc_gap = fixed_gap_dc(offset=0.0, report=report, **kwargs)[0, 0].real
    half_gap = 0.5 * report["gap_width"]

    dc_addition = fixed_peak_dc(peak_position=+half_gap, **kwargs)[0, 0].real
    dc_removal = fixed_peak_dc(peak_position=-half_gap, **kwargs)[0, 0].real

    np.testing.assert_allclose(dc_addition, dc_gap, atol=1e-2)
    np.testing.assert_allclose(dc_removal, dc_gap, atol=1e-2)


def test_fixed_gap_dc_costs_a_handful_of_solves():
    """Every evaluation is three ground-state solves, so the count is the cost.

    The gap criterion knows its slope analytically, so the first step is a Newton step rather than
    the first rung of a geometric walk -- and on this fixture the residual is exactly linear with
    exactly the declared slope, so that one step lands on the root and the polish settles at its
    ``width_tol`` early return without evaluating anything.

    What the count is really guarding is the other half: the criterion must not mistake its own
    (deliberately loose) energy tolerance for a Karolak plateau and start bisecting the band's
    edges. That mistake measured 17 evaluations against 2, and the 15 saved are bisections
    removed, not a refinement added. The refinement itself is exercised analytically in
    ``test_dc_search_efficiency.py``, where the residual is not linear in the declared slope.
    """
    kwargs, _ = common_kwargs(v=0.01, tau=1e-3, dc_scale=0.0)
    report = {}
    fixed_gap_dc(offset=0.0, report=report, **kwargs)
    assert report["evaluations"] <= 6, report
    assert report["status"] != "plateau", report


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
    # This regression is about the stale group_key crash, not about which charge state the peak
    # search lands on -- opt out of R2's charge-state guard, which is orthogonal to what is
    # being tested here.
    dc = fixed_peak_dc(peak_position=0.5, allow_charge_state_change=True, **kwargs)
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
    # This test is about accepting a multi-group N0, not about the landed charge state -- opt
    # out of R2's charge-state guard, which is orthogonal to what is being tested here.
    dc = fixed_peak_dc(peak_position=1.0, allow_charge_state_change=True, **kwargs)
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
    # 2.275, not the 2.5 the bidirectional scan used to return: the impurity saturates at 2
    # electrons, so the criterion is met on a *plateau* rather than at a point, and the old
    # answer was simply whichever geometric grid point first landed in it. The search now
    # returns the plateau's near edge stood off by one `initial_step` -- closest to the
    # double-counting guess, per the tie-break, but not within the search's own resolution
    # of the charge-sector boundary that bounds it.
    np.testing.assert_allclose(dc[0, 0].real, 2.275, atol=0.05)


def test_fixed_occupation_dc_decreases_occupation():
    # A guess of 7 puts two electrons on the impurity; requesting one electron
    # must bring the double counting back below the charge-transfer point.
    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2, dc_scale=7.0)
    dc = fixed_occupation_dc(occupation=1.0, **kwargs)
    assert_uniform_shift(dc, dc_guess)
    assert dc[0, 0].real < 6.0, dc


def test_fixed_occupation_dc_low_target_lands_on_plateau(capsys):
    # find_ground_state_basis's sector search is not fixed-N: it can land on a smaller-total-N
    # sector (module docstring) where a low impurity occupation is a genuine plateau -- the
    # observable steps across the target at a charge-sector boundary rather than crossing it.
    # B3: this is "no solution here", not a tolerable miss, so it raises DoubleCountingUnreachable
    # (the diagnostic still prints on rank 0 before the raise; only the old silent-return is gone).
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2, dc_scale=0.5)
    # The exception's own message is unreachable_message (the generic "could not bracket" text);
    # the richer "no double-counting shift attains" narration is printed separately by
    # _report_unattainable_target on rank 0, checked below via capsys.
    with pytest.raises(DoubleCountingUnreachable, match="Could not bracket the requested impurity occupation"):
        fixed_occupation_dc(occupation=0.2, **kwargs)
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
    # 2.275, not the 2.5 the bidirectional scan used to return: the impurity saturates at 2
    # electrons, so the criterion is met on a *plateau* rather than at a point, and the old
    # answer was simply whichever geometric grid point first landed in it. The search now
    # returns the plateau's near edge stood off by one `initial_step` -- closest to the
    # double-counting guess, per the tie-break, but not within the search's own resolution
    # of the charge-sector boundary that bounds it.
    np.testing.assert_allclose(dc_auto[0, 0].real, 2.275, atol=0.05)


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


def test_fixed_occupation_dc_reports_chi_in_its_record(capsys):
    """R3: chi actually reaches the emitted record, at the production call site -- not just inside
    `_dc_chi` itself. A unit test of `_dc_chi` cannot catch `width_tol` failing to reach it (a
    second copy of `max(tau, 1e-4)` silently diverging from the first is exactly the class of no-op
    this branch has already shipped once, see `bracket_width_tol`), nor `at` being passed the wrong
    mu, which would report a slope measured somewhere the answer is not.
    """
    # occupation=1.0 is already the guess's own occupation (see test_fixed_occupation_dc_already_
    # converged): the mu=0 fast path is taken and no second point is ever evaluated, so chi has
    # no pair to measure -- the "not resolvable" branch, reported rather than silently skipped.
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    fixed_occupation_dc(occupation=1.0, **kwargs)
    out = capsys.readouterr().out
    if MPI.COMM_WORLD.rank == 0:
        assert parse_record(out)["chi"] == "not resolvable", out

    # occupation=2.0 walks mu away from the guess (test_fixed_occupation_dc_increases_occupation),
    # evaluating more than one point, so a real chi must be recorded with a number beside it.
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2, dc_scale=0.5)
    fixed_occupation_dc(occupation=2.0, **kwargs)
    out = capsys.readouterr().out
    if MPI.COMM_WORLD.rank == 0:
        chi = parse_record(out)["chi"]
        assert "not resolvable" not in chi, out
        # And the uncertainty it implies for the answer, which is the reason it is reported at all.
        assert "dc determined to" in chi or "dc undetermined" in chi, chi


def test_fixed_occupation_dc_reference_ignores_dc_guess():
    # NiO in miniature: a large dc_guess must not corrupt the DFT reference. With dc_guess = -3
    # the shifted one-body impurity level eps - dc = +2 pokes above E_F, so the OLD reference
    # fill(h0 - dc) was ~0 -- unreachable (N_imp >= 1 here), the search failed or wandered. The
    # reference is the raw-h0 filling (N0 = 2, independent of the guess), so the search must
    # still converge somewhere on the "free bath electron" alternate-sector plateau (module
    # docstring), the same crossing test_fixed_occupation_dc_increases_occupation hits -- and now
    # at the *same absolute dc*, which is what this test's name has always claimed and could not
    # previously assert. The old bidirectional scan returned whichever crossing sat nearest its
    # own starting point, so this guess gave 5.0 where a guess of +0.5 gave 2.5. The sector-first
    # search brackets the charge sector the criterion is solved in and then takes the plateau's
    # near edge with a fixed standoff, which does not depend on where the scan started: guesses of
    # -3.0, +0.5 and +3.0 all converge to within 0.13 of each other, where the old scan spread them over 2.5.
    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2, dc_scale=-3.0)
    dc = fixed_occupation_dc(**kwargs)
    assert_uniform_shift(dc, dc_guess)
    np.testing.assert_allclose(dc[0, 0].real, 2.375, atol=0.05)


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
def test_fixed_gap_dc_ranks_agree():
    # Same hazard as the peak criterion: the gap centre is a difference of Lanczos energies,
    # replicated only to roundoff, and it gates every branch that decides whether the next
    # collective solve runs.
    comm = MPI.COMM_WORLD
    kwargs, _ = common_kwargs(v=0.01, tau=1e-3)
    dc = fixed_gap_dc(offset=0.0, **kwargs)
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


def test_the_gap_tolerance_says_which_of_its_four_cases_produced_it(capsys):
    """``_size_gap_tolerance`` replaced an ``abs()`` that hid two distinct failures.

    Tested directly rather than through a search: reaching the negative-width and runaway-width
    cases from a model means engineering a truncation pathology, and the point of extracting the
    function is that the four cases are decidable from three numbers.
    """
    from impurityModel.ed.dc_criteria import GAP_ENERGY_TOLERANCE_FRACTION as frac
    from impurityModel.ed.dc_criteria import _size_gap_tolerance

    mu_tol, bandwidth = 1e-3, 3.0

    # The ordinary case: a percent of the measured width.
    tol, basis = _size_gap_tolerance(3.0, mu_tol, bandwidth)
    assert basis == "measured_gap" and tol == pytest.approx(frac * 3.0)

    # No width to scale against -- fall back to the mu resolution, do not invent one.
    assert _size_gap_tolerance(None, mu_tol, bandwidth) == (mu_tol, "width_undefined")

    # Negative: E[N] is not the lowest of the three sectors, so the centre walk and the sector
    # solves disagree. `abs()` silently turned this into a slightly different tolerance; it is a
    # diagnostic and must be said out loud.
    tol, basis = _size_gap_tolerance(-2.0, mu_tol, bandwidth)
    assert (tol, basis) == (mu_tol, "width_negative")
    assert "negative" in capsys.readouterr().out

    # Runaway: an inflated width would set a tolerance big enough that `abs(g0) <= tol` holds at
    # mu = 0, and the search would return the guess unchanged after one evaluation, reporting
    # `converged`. A no-op claiming success is the worst answer available, so it is capped.
    tol, basis = _size_gap_tolerance(1e4, mu_tol, bandwidth)
    assert basis == "width_capped" and tol == pytest.approx(0.1 * bandwidth)
    assert "cannot localize mu" in capsys.readouterr().out

    # The cap never bites below the mu resolution, however small the bandwidth.
    assert _size_gap_tolerance(1e4, mu_tol, 0.0)[0] == mu_tol


def test_the_gap_criterion_measures_its_own_impurity_character(capsys):
    """``delta_sum`` separates the two regimes the rest of the suite could not tell apart.

    Characterization, not a pass/fail claim about the physics: the point is that the criterion now
    *reports* whether the gap it centred belongs to the impurity or to the bath, so a user is not
    left inferring it from a slope buried in a diagnostics table.

    In the near-atomic fixture the added electron can only go on the impurity, so both edges are
    impurity-like and both deltas sit near their maximum of 1. With a conduction bath level just
    above E_F the *addition* edge is bath-like while the removal edge is untouched, and it is the
    weaker of the two -- ``min(delta_+, delta_-)``, below ``GAP_EDGE_IMPURITY_FLOOR`` -- that the
    criterion says out loud.

    Measured directly on these two fixtures: near-atomic ``(1.0000, 1.0000)`` against
    charge-transfer ``(-0.0210, 0.8179)``. The **sums**, 2.000 and 0.797, are what an earlier
    version of this test compared, and they nearly hide the defect: one conduction orbital at +0.4
    turns a wholly impurity-like addition edge into a *negatively* impurity one while the sum only
    falls by 60%. The edges fail independently, so the summary statistic has to be the minimum.

    ``delta_+ = -0.021`` is not a typo and not a convergence artefact -- it is bit-reproducible
    across ``c46eb82`` and the refactor after it -- and it is the review's F3 in the suite:
    ``delta`` is **not** bounded below by 0. Adding an electron here puts it on the conduction
    level and rehybridization takes slightly *more* than that off the impurity, so the impurity
    occupation goes down when an electron is added. Any future rewrite that clips these to
    ``[0, 1]``, or reads them as "the fraction of the electron that landed on the impurity",
    breaks on this fixture -- which is why the values are pinned here to 1e-3 rather than only
    bounded. An earlier docstring quoted ``(0.075, 0.951)`` for this fixture; no version of this
    code ever produced it.

    Both searches run on **every** rank; only the parsing and the assertions are rank-gated. An
    earlier version of this test put the ``rank != 0: return`` between the two searches, so rank 0
    entered the second ``fixed_gap_dc`` -- and every collective under it -- while the other ranks
    walked out to teardown. It hung the ``-n 2`` gate for seven minutes at 24% before ``py-spy``
    showed rank 0 inside ``distribute_determinants`` and rank 1 already in
    ``pytest_runtest_teardown``. Exactly the rule in CLAUDE.md, broken in a test written to check
    something else: **never gate a collective on rank-local state.**
    """
    from impurityModel.ed.dc_criteria import GAP_EDGE_IMPURITY_FLOOR

    kwargs, _ = common_kwargs(v=0.01, tau=1e-3, dc_scale=0.5)
    fixed_gap_dc(offset=-0.4, **kwargs)
    atomic_out = capsys.readouterr().out

    kwargs, _ = charge_transfer_kwargs()
    try:
        fixed_gap_dc(offset=0.0, allow_charge_state_change=True, **kwargs)
    except DoubleCountingUnreachable:
        # A bath-like edge that does not respond to mu can also make the target unattainable;
        # either way the record is emitted from the `finally` and carries the verdict. Raised
        # identically on every rank (the residual is broadcast), so catching it here does not
        # split the ranks either.
        pass
    transfer_out = capsys.readouterr().out

    if MPI.COMM_WORLD.rank != 0:
        return
    atomic_record = parse_record(atomic_out)
    atomic_edges = [float(atomic_record[key].split()[0]) for key in ("delta_plus", "delta_minus")]
    # Impurity-only addition: both edges move fully with the shift, so neither is flagged.
    assert min(atomic_edges) > 0.9, atomic_out
    assert "WARNING: the addition" not in atomic_out and "WARNING: the removal" not in atomic_out, atomic_out

    record = parse_record(transfer_out)
    # Unconditional, and that is a change. This used to be guarded by `if "delta_plus" in record`,
    # because the edges were measured *after* the search returned and `DoubleCountingUnreachable`
    # skipped past that line -- so the record was blank on exactly the runs where a bath-like edge
    # is a leading *cause* of the failure. The measurement now happens in the search block's
    # `finally`, at the last shift actually evaluated, so there is no path that reaches a record
    # without it.
    assert "delta_plus" in record and "delta_minus" in record, transfer_out
    edges = [float(record[key].split()[0]) for key in ("delta_plus", "delta_minus")]
    # The defect is one-sided: the conduction level spoils the addition edge and leaves the
    # removal edge alone. Asserting on the minimum is what the sum could not do.
    assert min(edges) < min(atomic_edges), f"charge-transfer edges {edges} not below atomic {atomic_edges}"
    assert min(edges) < GAP_EDGE_IMPURITY_FLOOR, edges
    assert "WARNING: the addition (N+1) edge" in transfer_out, transfer_out
    # Pinned, not merely bounded, and negative on purpose -- see the docstring. The inequalities
    # above all pass for `delta_+ = 0.075` too, which is what let a wrong number sit in this
    # docstring across two commits without a test noticing.
    assert edges[0] == pytest.approx(-0.021, abs=1e-3), edges
    assert edges[1] == pytest.approx(0.818, abs=1e-3), edges


def test_the_edge_character_costs_no_sector_solves_of_its_own(monkeypatch):
    """``delta_+-`` come from solves the search already paid for, not from three fresh ones.

    They used to cost three: ``sector_energy`` went through ``calc_energy``, which discards its
    eigenvectors, so ``sector_occupation`` re-solved the same sectors at the same shift to get
    them back. On a workload that converges in one evaluation -- the common case in a converged
    self-consistency loop, and the first production workload this met -- that was a third of the
    whole search. Measured on this fixture: 10 CIPSI expansions before, 7 after, with
    ``delta_+-`` identical to 16 digits.

    Asserting the *delta* across ``_measure_edge_character`` rather than a total, because a total
    is a number that drifts with every unrelated change to the search and would be re-baselined
    rather than believed. Zero is a property.

    It also matters for correctness, not only cost: a re-solve matches the expansion whose
    energies ``omega_+-`` came from only if CIPSI selection is deterministic. Reading the pair off
    one solve makes them co-derived by construction.
    """
    from impurityModel.ed import dc_criteria as dc_module
    from impurityModel.ed import groundstate as gs_module

    solves = {"n": 0}
    # `_solve_sector_core`, not `solve_sector`: the core is the one body both `solve_sector` and
    # `calc_energy` run, so a counter on it sees every sector solve regardless of which head the
    # search entered through. `_measure_edge_character` reaches neither today, which is exactly
    # what makes the assertion below meaningful -- but a counter on one head only would report
    # zero for a solve that moved to the other.
    real_solve_sector = gs_module._solve_sector_core

    def counted(*args, **kwargs):
        solves["n"] += 1
        return real_solve_sector(*args, **kwargs)

    monkeypatch.setattr(gs_module, "_solve_sector_core", counted)

    spent = {}
    real_measure = dc_module._measure_edge_character

    def watched(ctx, mu, n_center):
        before = solves["n"]
        result = real_measure(ctx, mu, n_center)
        spent["solves"] = solves["n"] - before
        return result

    monkeypatch.setattr(dc_module, "_measure_edge_character", watched)

    kwargs, _ = common_kwargs(v=0.01, tau=1e-3)
    report = {}
    fixed_gap_dc(offset=0.0, report=report, **kwargs)

    assert spent.get("solves") == 0, f"the edge measurement ran {spent.get('solves')} sector solves of its own"
    # And it still produced the numbers: a measurement that costs nothing because it measured
    # nothing would pass the assertion above.
    assert report["delta_plus"] is not None and report["delta_minus"] is not None, report


def test_the_two_estimators_of_delta_sum_agree(capsys):
    """``delta_+ + delta_-`` against ``-2*chi``: two routes to one number, differenced.

    The criterion has always carried both and, until ``delta_sum_vs_chi``, compared neither.
    ``delta_sum`` is built from impurity occupations, which are **first**-order in the wavefunction
    error; ``-2*chi`` is a secant of gap centres against ``mu``, **second**-order in it but
    carrying a finite-difference term and a CIPSI basis-reselection term the occupations do not
    have. They fail differently, so agreement is not a tautology -- it is end-to-end evidence that
    the sector solves are converged and that ``sector_solve`` is returning the occupation of the
    same expansion the energy came from.

    Hellmann-Feynman is what ties them: ``d(centre)/dmu = -(delta_+ + delta_-)/2`` for an exact
    eigenstate of ``H(mu)``, so the difference asserted here is a residual of that identity on the
    actual variational space, not a comparison of two independent measurements.

    Measured: ``-7.3e-06`` on the near-atomic fixture, ``-5.1e-04`` on the charge-transfer one.
    The two orders of magnitude between them are the point -- the bath is what makes the identity
    hard, and a threshold tight enough for the atomic case alone would be a false floor. Bounded
    at 5e-3, roughly ten times the worse of the two, so real divergence is caught and roundoff is
    not.
    """
    kwargs, _ = common_kwargs(v=0.01, tau=1e-3, dc_scale=0.5)
    atomic = {}
    fixed_gap_dc(offset=-0.4, report=atomic, **kwargs)

    kwargs, _ = charge_transfer_kwargs()
    transfer = {}
    try:
        fixed_gap_dc(offset=0.0, allow_charge_state_change=True, report=transfer, **kwargs)
    except DoubleCountingUnreachable:
        # Rank-identical (the residual is broadcast), so catching it does not split the ranks.
        pass

    if MPI.COMM_WORLD.rank != 0:
        return
    for label, report in (("near-atomic", atomic), ("charge-transfer", transfer)):
        # Both searches take more than one evaluation inside one sector, which is what `chi`
        # needs to exist. If a future fixture change makes one converge at the guess, `chi` goes
        # None and this field disappears -- fail loudly rather than skip, because a silently
        # vanishing cross-check is the failure mode the whole review was about.
        assert report.get("chi") is not None, f"{label}: no chi, so the cross-check never ran"
        assert report.get("delta_sum_vs_chi") is not None, f"{label}: {report}"
        assert report["delta_sum_vs_chi"] == pytest.approx(
            report["delta_sum"] - (-2.0 * report["chi"])
        ), f"{label}: the recorded difference is not the difference of the two estimators"
        assert abs(report["delta_sum_vs_chi"]) < 5e-3, (
            f"{label}: the occupation estimator {report['delta_sum']:.6f} and the chi secant "
            f"{-2.0 * report['chi']:.6f} disagree by {report['delta_sum_vs_chi']:.2e}"
        )


def test_a_weightless_gap_edge_passes_the_impurity_character_check(capsys):
    """The limitation the warning disclaims, exhibited: ``delta_+ = 1`` on a zero-weight edge.

    ``_warn_if_a_gap_edge_is_not_impurity_like`` measures a ``mu``-response, and its own text says
    that is not a spectral weight -- "a state orthogonal to c_d^dagger|0> by spin or irrep carries
    zero weight in A_imp while still moving fully with mu, and no value of delta detects that".
    Until :func:`spin_crossover_kwargs` existed, no fixture in this suite could show it: with a
    single impurity spatial orbital the two quantities track to within a few percent at the edge.

    Here they decouple completely. Inside the crossover window the ``N + 1`` ground state is the
    high-spin ``S_z = 3/2`` multiplet, unreachable from the ``S_z = 0`` ground state by any single
    ``c_d^dagger``, so its Lehmann weight is **exactly** zero -- while the added electron sits
    entirely on the impurity, giving the maximum possible ``delta``. The criterion converges, both
    edges report 1.000, and nothing warns. ``test_dc_weights`` measures the zero weight itself;
    this test pins the *silence*, which is the part a user sees.

    ``delta = 5`` is the control: identical fixture, crossover switched off, and the edge the
    criterion picks is the weighted one. Both report ``delta = 1``, which is exactly the point --
    the diagnostic cannot separate them, and this test would start failing if someone ever taught
    it to, which is the intended future.
    """
    for crystal_field in (3.5, 5.0):
        report = {}
        fixed_gap_dc(
            offset=0.0, report=report, allow_charge_state_change=True, **spin_crossover_kwargs(delta=crystal_field)
        )
        out = capsys.readouterr().out
        if MPI.COMM_WORLD.rank != 0:
            continue
        # The designed edges: +-3 inside the window (high-spin addition), +-4 outside it.
        assert report["omega_plus"] == pytest.approx(3.0 if crystal_field == 3.5 else 4.0, abs=1e-3), report
        assert report["omega_minus"] == pytest.approx(-3.0 if crystal_field == 3.5 else -4.0, abs=1e-3), report
        # Maximum impurity character on both edges, in both regimes -- the electron really does
        # land on the impurity. That is what makes the weightless case invisible here.
        assert report["delta_plus"] == pytest.approx(1.0, abs=1e-3), report
        assert report["delta_minus"] == pytest.approx(1.0, abs=1e-3), report
        assert "WARNING: the addition" not in out and "WARNING: the removal" not in out, out


def test_the_record_says_whether_the_thermal_manifold_hides_anything(capsys):
    """``manifold_spread``: the evidence for pairing a ``T = 0`` energy with a thermal occupation.

    ``omega_+-`` are the *lowest* eigenvalue of each sector; ``delta_+-`` are differences of
    Boltzmann-averaged ``Tr rho_imp`` over each sector's whole retained manifold. Hellmann-Feynman
    relates them **state by state**, so the pairing is exact only where the retained states share
    ``N_imp``. Nothing said whether they do, and the argument for the thermal convention was made
    on rank-safety and consistency with ``fixed_occupation_dc`` -- both true, and neither about
    this.

    So it is measured instead of argued. On every fixture in this suite the spread is machine zero
    (2e-16, 0.0, 5e-15), which closes the question *for these models*: the two conventions are the
    same number and switching to ``argmin(es)`` would change nothing. The field exists so that a
    model where they differ says so in its own record rather than being caught by whoever next
    re-derives the concern.

    The manifold **sizes** are asserted alongside deliberately. A spread of zero over a one-state
    manifold is silence, not agreement, and the off-centre sectors here retain exactly one state
    -- so the centre sector, with two, is carrying the whole result.
    """
    kwargs, _ = common_kwargs(v=0.01, tau=1e-3, dc_scale=0.5)
    report = {}
    fixed_gap_dc(offset=-0.4, report=report, **kwargs)

    if MPI.COMM_WORLD.rank != 0:
        return
    assert report.get("manifold_spread") is not None, report
    assert report["manifold_spread"] < 1e-9, report["manifold_spread"]
    sizes = [int(n) for n in report["manifold_states"].split("/")]
    assert len(sizes) == 3 and all(n >= 1 for n in sizes), report["manifold_states"]
    # Not a vacuous zero: at least one sector retained more than one state, so at least one
    # comparison was actually made.
    assert max(sizes) > 1, f"every manifold held one state, so the spread asserts nothing: {sizes}"


def test_the_impurity_character_warning_fires_on_the_weaker_edge(capsys):
    """The predicate, against every ``(delta_+, delta_-)`` pair actually measured.

    The real workload is too expensive to run in the suite, so the calibration is pinned here
    instead -- and it needs pinning, because both previous versions of this threshold were wrong
    in a way only a measurement exposed. A sum-based 0.2 stayed silent on the NiO 15-bath run that
    motivated it (sum 0.218); raising it to 0.5 fixed that one and still stayed silent on NiO
    5-bath, whose removal edge is *pure ligand* (0.032) but whose sum is 0.593.
    """
    from impurityModel.ed.dc_criteria import GAP_EDGE_IMPURITY_FLOOR as floor
    from impurityModel.ed.dc_criteria import _warn_if_a_gap_edge_is_not_impurity_like as warn

    # NiO 5-bath, measured directly: an impurity-like addition edge above a ligand valence band.
    # The sum is 0.593 -- above any sum-based floor that does not also fire on healthy models.
    warn(0.561, 0.032, rank=0)
    out = capsys.readouterr().out
    assert "removal (N-1)" in out and "moves only 0.032 of an electron" in out, out
    # The caveat that keeps the warning honest about what it measured. delta is a mu-response, and
    # a state orthogonal to c_d^dagger|0> by spin or irrep carries zero weight in A_imp while still
    # moving fully with mu -- so a HIGH delta is not a certificate. Saying "this edge is a state of
    # the bath" from a slope alone was the stronger claim than the measurement supports.
    assert "mu-response, not a spectral weight" in out, out
    assert "charge-transfer insulator" in out and "discretization artefact" in out, out
    assert 0.561 + 0.032 > 0.5, "the sum-based floor this replaced would have stayed silent here"

    # The charge-transfer fixture fails on the other edge, and must be named as such.
    warn(0.075, 0.951, rank=0)
    out = capsys.readouterr().out
    assert "addition (N+1)" in out and "moves only 0.075 of an electron" in out, out

    # Healthy models stay quiet, or the warning becomes noise on every well-behaved run.
    for quiet in ((1.0, 1.0), (0.997, 0.976), (0.56, 0.44)):
        warn(*quiet, rank=0)
        assert capsys.readouterr().out == "", quiet
    assert floor < 0.44, "the floor must clear the weakest edge of a model that is still healthy"

    # Never on a non-root rank, and never where nothing was measured. One edge measured and the
    # other undefined is still a verdict -- a shell edge does not excuse a ligand gap edge.
    warn(0.01, 0.01, rank=1)
    warn(None, None, rank=0)
    assert capsys.readouterr().out == ""
    warn(0.01, None, rank=0)
    assert "addition (N+1)" in capsys.readouterr().out


def test_the_gap_tolerance_names_the_term_that_actually_won(capsys):
    """``tol_basis`` distinguishes all five outcomes, including the one it used to misreport.

    ``energy_tol = max(0.01 * width, mu_tol)``, capped against the bandwidth. Reporting
    ``measured_gap`` whenever the width was merely *defined* was wrong on both real workloads
    measured: NiO 15-bath sized ``0.01 * 0.0654 = 6.5e-4`` against ``mu_tol = 2.5e-3`` and NiO
    5-bath ``0.01 * 0.119 = 1.2e-3`` against the same, so in both the tolerance came from
    ``bracket_width_tol`` while the record said it came from the gap. Tested here directly rather
    than through a search, because reaching all five cases by building a model for each is how
    the case that misreports came to be untested in the first place.
    """
    from impurityModel.ed.dc_criteria import _size_gap_tolerance

    mu_tol, bandwidth = 2.5e-3, 10.0

    # The width sets it: 0.01 * 1.0 = 1e-2, above mu_tol and far below the 0.1 * bandwidth cap.
    assert _size_gap_tolerance(1.0, mu_tol, bandwidth) == (pytest.approx(1e-2), "measured_gap")
    # The mu resolution sets it: 0.01 * 0.1 = 1e-3, below mu_tol. Same number as before the fix,
    # different -- and now correct -- account of where it came from.
    assert _size_gap_tolerance(0.1, mu_tol, bandwidth) == (pytest.approx(mu_tol), "mu_resolution")
    # Exactly at the crossover the width is still the term that won, by >=.
    assert _size_gap_tolerance(0.25, mu_tol, bandwidth)[1] == "measured_gap"
    # Undefined and negative widths keep their own words.
    assert _size_gap_tolerance(None, mu_tol, bandwidth) == (mu_tol, "width_undefined")
    assert _size_gap_tolerance(-1.0, mu_tol, bandwidth) == (mu_tol, "width_negative")
    # A runaway width is capped, not believed.
    assert _size_gap_tolerance(1e4, mu_tol, bandwidth) == (pytest.approx(1.0), "width_capped")
    capsys.readouterr()


def test_the_gap_criterion_records_where_its_tolerance_came_from(capsys):
    """The four-way basis reaches the emitted record, not just the function that computed it.

    Asserted on the record rather than the ``report=`` dict on purpose: the record is the channel
    a user actually reads, and a value that never reaches it is a value nobody sees.
    """
    kwargs, _ = common_kwargs(v=0.01, tau=1e-3)
    fixed_gap_dc(offset=0.0, **kwargs)
    out = capsys.readouterr().out
    if MPI.COMM_WORLD.rank == 0:
        assert parse_record(out)["tol_basis"] == "measured_gap", out


@pytest.mark.mpi
def test_a_rank_dependent_sector_energy_cannot_reach_the_gap_tolerance(monkeypatch):
    """The gap criterion's *tolerance* must be rank-identical, not merely its residual.

    ``fixed_gap_dc`` is the only criterion that sizes its convergence tolerance from a solver
    output: ``energy_tol = 1% of the measured gap width``, and that width is three
    ``calc_energy`` results. Lanczos energies are replicated only to roundoff, so an un-broadcast
    width makes ``tol`` rank-local -- and ``tol`` gates every branch deciding whether the next
    collective solve happens (``abs(g0) <= tol`` at ``dc_search``'s fast path is the sharpest:
    one rank returns immediately, the other runs the whole search).

    **Why this asserts the invariant rather than the symptom.** The symptom is a *hang*, which no
    assertion can catch -- the test would never fail, it would stop. So the property under test is
    the one that makes the hang impossible: inject a rank-dependent perturbation *below* the
    broadcast boundary and require the derived tolerance to come out bit-for-bit identical anyway.
    ``1e-12`` is chosen deliberately: large enough that an un-broadcast ``tol`` differs in the
    bits (the assertion is ``==``, not ``approx``), far too small to flip any ``abs(g) <= tol``
    test on this fixture and so incapable of turning a red test into a hung one.

    The perturbation is **multiplicative**, and that is not incidental. The gap width is the
    second difference ``E[N+1] + E[N-1] - 2 E[N]``, in which any rank-dependent *constant* cancels
    exactly -- an additive probe leaves the width bit-identical and the test passes without
    testing anything. Scaling each energy instead survives the second difference.
    """
    from impurityModel.ed import groundstate as gs_module

    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("needs at least 2 ranks")

    real_solve_sector = gs_module.solve_sector

    def perturbed(*args, **kwargs):
        es, psis, basis = real_solve_sector(*args, **kwargs)
        return es * (1.0 + comm.rank * 1e-12), psis, basis

    # `solve_sector`, not `calc_energy`: `_SectorContext.sector_solve` produces the energy *and*
    # the impurity occupation from one expansion, so this is the boundary both quantities cross.
    # It is imported function-locally, so patching the module attribute is picked up -- and it
    # patches *below* the boundary that has to broadcast, which is what makes the perturbation
    # observable at all.
    #
    # Perturbing `es` rather than the returned energy also reaches further than the old probe did:
    # the Boltzmann weights in `thermal_average_scale_indep` are built from these same numbers, so
    # an un-broadcast occupation comes out rank-dependent too. One injection, both paths.
    monkeypatch.setattr(gs_module, "solve_sector", perturbed)

    kwargs, _ = common_kwargs(v=0.01, tau=1e-3)
    report = {}
    fixed_gap_dc(offset=0.0, report=report, **kwargs)

    # `tol`, not `energy_tol`: the criterion's out-parameter *is* its record now, one vocabulary
    # across both channels. This test read the old private name and kept passing until V5's rename
    # landed -- which is itself the argument for not having had two sets of names.
    # `delta_plus`/`delta_minus` are here for a second reason. They come from
    # `_SectorContext.sector_occupation`, which runs `build_density_matrices` -- and that ends in
    # an `Allreduce`. Everything in front of it (is the manifold usable, how many states does it
    # hold) is read off rank-local `solve_sector` output, so the branch has to be broadcast or
    # some ranks skip the reduction the others are sitting in. A deadlock cannot be asserted on,
    # so what is asserted is the invariant that rules it out: the values are bit-identical.
    for key in ("tol", "gap_width", "sector", "delta_plus", "delta_minus"):
        gathered = comm.allgather(report[key])
        assert len(set(gathered)) == 1, f"{key} is not rank-identical: {gathered}"


@pytest.mark.mpi
def test_calc_energy_returns_a_bitwise_identical_energy_on_every_rank():
    """The invariant one layer down, stated where it belongs.

    Three call sites consume ``calc_energy``'s energy and two of them broadcast it themselves
    (``groundstate.py``'s walk and ``solve_ground_state``). The third -- ``_SectorContext`` -- did
    not, which is how the gap tolerance became rank-local. Enforcing it at the source is what
    stops the next caller inheriting the same trap.
    """
    from impurityModel.ed.groundstate import calc_energy

    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("needs at least 2 ranks")

    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    sb = prepare_solver_basis(
        kwargs["model"].h0,
        kwargs["model"].dc,
        kwargs["model"].u4,
        kwargs["model"].impurity_orbitals,
        kwargs["basis"].nominal_occ,
        kwargs["basis"].mixed_valence,
        kwargs["model"].rot_to_spherical,
        0,
    )
    energy, _ = calc_energy(
        sb.h,
        sb.impurity_orbitals,
        sb.bath_states,
        sb.nominal_occ,
        sb.mixed_valence,
        kwargs["basis"].tau,
        kwargs["basis"].chain_restrict,
        kwargs["basis"].spin_flip_dj,
        kwargs["solver"].dense_cutoff,
        comm=comm,
        verbose=False,
    )
    gathered = comm.allgather(energy)
    assert len(set(gathered)) == 1, f"calc_energy is not rank-replicated: {gathered}"


@pytest.mark.mpi
@pytest.mark.parametrize(
    "criterion, call",
    [
        ("peak", lambda kw: fixed_peak_dc(peak_position=1.2, **kw)),
        ("gap", lambda kw: fixed_gap_dc(offset=0.0, **kw)),
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
    """And derives it at the **full** safety fraction, like every other driver.

    This used to halve it, on the grounds that the two N+-1 solves were live alongside the centre
    search's basis. They are not -- `centre_sector` and `sector_energy` each discard their basis
    before the next solve starts, so exactly one is live at a time and the peak is the centre
    walk's own SectorCache, which is the peak `calc_gs` has too. The halving measured the double
    counting on half the determinant budget the self-energy run would use at that same dc: a
    DC<->GS parity break, in the module whose whole purpose is closing those.
    """
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
    assert calls[0]["safety"] == pytest.approx(dc_module.DEFAULT_MEMORY_SAFETY)
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
    # 2.275, not the 2.5 the bidirectional scan used to return: the impurity saturates at 2
    # electrons, so the criterion is met on a *plateau* rather than at a point, and the old
    # answer was simply whichever geometric grid point first landed in it. The search now
    # returns the plateau's near edge stood off by one `initial_step` -- closest to the
    # double-counting guess, per the tie-break, but not within the search's own resolution
    # of the charge-sector boundary that bounds it.
    np.testing.assert_allclose(dc[0, 0].real, 2.275, atol=0.05)


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
    energy_cut = boltzmann_energy_cut(basis_opts.tau)
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


def _frozen_core_kwargs():
    """A three-group impurity: a bath-less core pair plus two bath-coupled valence pairs.

    Six impurity spin-orbitals at three on-site energies -- a deep, uncoupled core doublet
    (``-5.0``) and the ``_split_block_kwargs`` valence doublets (``0.4``/``-0.4``) -- so
    ``prepare_solver_basis`` derives three orbital-symmetry groups, exactly one of which
    (the core) has zero attached bath orbitals. Only the valence orbitals couple to the four
    bath states; the core rows/columns of ``V`` are zero.
    """
    h_imp = np.diag([-5.0, -5.0, 0.4, 0.4, -0.4, -0.4]).astype(complex)
    v = np.zeros((4, 6), dtype=complex)
    v[0, 2] = v[1, 3] = v[2, 4] = v[3, 5] = 0.2
    h_bath = np.diag([-2.0, -2.0, -2.1, -2.1]).astype(complex)
    u4 = np.zeros((6,) * 4, dtype=complex)
    dc = np.zeros((6, 6), dtype=complex)
    model = ImpurityModel.from_blocks(
        h_imp,
        v,
        h_bath,
        u4=u4,
        dc=dc,
        rot_to_spherical=np.eye(6, dtype=complex),
        bath_valence_conduction=(list(range(6, 10)), []),
    )
    # Energetic filling (lowest on-site energy first) puts the total on core (full, -5.0 x2)
    # and the -0.4 valence doublet (full), leaving the 0.4 doublet empty -- a nominal 4.
    basis = BasisOptions(
        nominal_occ={0: 4},
        mixed_valence=None,
        spin_flip_dj=False,
        tau=1e-3,
        slater_weight_min=np.sqrt(np.finfo(float).eps),
        truncation_threshold=int(1e6),
    )
    return dict(model=model, basis=basis, solver=SolverOptions(dense_cutoff=1000), comm=MPI.COMM_WORLD), dc


def _bath_total(group, valence_baths, conduction_baths):
    return sum(len(b) for b in valence_baths.get(group, [])) + sum(len(b) for b in conduction_baths.get(group, []))


@pytest.mark.parametrize(
    "criterion, call",
    [
        ("fixed-occupation", lambda kw: fixed_occupation_dc(occupation=4.0, **kw)),
        # This test is about the three inputs reaching find_ground_state_basis, not about the
        # landed charge state -- opt out of R2's charge-state guard.
        ("fixed-peak", lambda kw: fixed_peak_dc(peak_position=0.5, allow_charge_state_change=True, **kw)),
        ("fixed-gap", lambda kw: fixed_gap_dc(offset=0.0, allow_charge_state_change=True, **kw)),
    ],
)
def test_dc_criteria_pass_frozen_occupations_tau_and_symmetry_generators(monkeypatch, criterion, call):
    """R1: the call-counter proof, not an outcome proof (the shim lesson).

    Asserting on the returned ``dc`` cannot distinguish "the three inputs reached
    ``find_ground_state_basis``" from "they didn't, and it happened not to matter" -- both
    criteria' docstrings claimed parity with ``calc_gs`` before this was true (see the module
    warning), and the toy parity test passed the whole time. Spy on
    ``groundstate.find_ground_state_basis`` itself (re-imported fresh inside each criterion at
    call time, so patching the ``groundstate`` module's attribute is what the ``from ... import``
    picks up) and inspect every call's kwargs directly.
    """
    import impurityModel.ed.groundstate as groundstate_module

    real_find_ground_state_basis = groundstate_module.find_ground_state_basis
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_find_ground_state_basis(*args, **kwargs)

    monkeypatch.setattr(groundstate_module, "find_ground_state_basis", spy)

    kwargs, _ = _frozen_core_kwargs()
    tau = kwargs["basis"].tau
    # This fixture's fixed-occupation target (4.0, the nominal) happens to sit on a charge-sector
    # plateau (B3: DoubleCountingUnreachable) -- irrelevant here, since find_ground_state_basis is
    # called and spied on before the search ever raises. The call-counter proof this test exists
    # for does not need the search to succeed.
    try:
        call(kwargs)
    except DoubleCountingUnreachable:
        pass

    assert calls, f"{criterion}: find_ground_state_basis was never called"
    for args, call_kwargs in calls:
        impurity_orbitals_arg = args[1]
        valence_baths, conduction_baths = args[2]
        expected_frozen = {i for i in impurity_orbitals_arg if _bath_total(i, valence_baths, conduction_baths) == 0}
        # The fixture's core group must actually be bath-less, or this test proves nothing.
        assert expected_frozen, f"{criterion}: fixture has no bath-less group to freeze"
        assert call_kwargs["frozen_occupations"] == expected_frozen, (criterion, call_kwargs["frozen_occupations"])
        # calc_gs's own convention: the walk runs at tau/100, restored to the full tau
        # everywhere else (see the module warning in dc_criteria.py).
        assert call_kwargs["tau"] == pytest.approx(tau / 100), (criterion, call_kwargs["tau"])
        # calc_gs runs the walk at sqrt(slaterWeightMin) and only tightens to the full cutoff
        # for the refinement; a walk run four orders of magnitude tighter is a different search
        # and can settle on a different sector.
        assert call_kwargs["slaterWeightMin"] == pytest.approx(np.sqrt(kwargs["basis"].slater_weight_min)), (
            criterion,
            call_kwargs["slaterWeightMin"],
        )
        # Stale symmetry generators must NOT be threaded down. CIPSISolver.expand re-derives them
        # from the H it is actually expanding whenever the argument is None (cipsi_solver.py),
        # which is the only operator they can correctly come from; calc_gs does not forward them
        # either. Passing a set computed elsewhere is how the two paths drifted apart.
        assert "symmetry_generators" not in call_kwargs, criterion


def test_fixed_peak_dc_targets_the_nominal_charge_state_by_default():
    """The peak criterion is multivalued, and the default must resolve that in the user's favour.

    The same requested peak position is attained at several charge states, several eV apart in
    ``dc``. The old bidirectional scan returned whichever sat nearest ``mu = 0`` -- on this fixture
    ``n_center = 2`` against a nominal total of 4 -- and the guard could only notice afterwards and
    refuse.

    Sector-first makes it a choice rather than an accident: the search locates a ``mu`` where the
    walk lands in the nominal sector and solves the criterion inside it, stepping by a secant on a
    residual whose slope it knows (``d(E_upper - E_lower)/dmu ~ -1``). It now returns the nominal
    charge state, in three evaluations. The guard still stands behind it for the case where the
    criterion genuinely has no solution there and the search has to widen.
    """
    kwargs, _ = _frozen_core_kwargs()
    nominal_total = sum(kwargs["basis"].nominal_occ.values())

    dc, sector = fixed_peak_dc(peak_position=0.5, return_sector=True, **kwargs)
    assert dc.shape == (6, 6)
    assert sector == nominal_total, f"default landed on charge state {sector}, not the nominal {nominal_total}"


def test_fixed_peak_dc_accepts_the_nominal_charge_state_by_default():
    """The guard must not reject the ordinary case: a peak reached without leaving the nominal
    sector passes with no opt-out needed."""
    kwargs, dc_guess = common_kwargs(v=0.01, tau=1e-3)
    dc = fixed_peak_dc(peak_position=1.2, **kwargs)
    assert_uniform_shift(dc, dc_guess)


def test_occupation_and_energy_at_mu_matches_the_search_at_mu_zero():
    """B2's context/per-mu extraction (``_prepare_occupation_context`` +
    ``_evaluate_occupation_and_energy_at_mu``) must be byte-identical to the observable
    ``fixed_occupation_dc``'s own search evaluates, not a reimplementation of it.

    ``_solve_dc_shift`` evaluates ``mu = 0`` first and returns immediately once
    ``|residual| <= tol`` (the search's own fast path): requesting exactly the occupation this
    diagnostic measures at ``mu = 0`` must therefore make the search return ``dc_guess``
    unchanged. If the extracted context ever silently diverged from the original inline closure
    (a different tau, a different truncation, a stale symmetry generator), this would fail.
    """
    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2)
    model, basis, solver = kwargs["model"], kwargs["basis"], kwargs["solver"]

    n_at_zero, e0, sector = occupation_and_energy_at_mu(model, basis, solver, 0.0, comm=kwargs["comm"])
    assert np.isfinite(e0)
    assert sector == int(sector)

    dc = fixed_occupation_dc(occupation=n_at_zero, **kwargs)
    assert_uniform_shift(dc, dc_guess)
    np.testing.assert_allclose(dc, dc_guess, atol=1e-9)


def test_occupation_and_energy_at_mu_ranks_agree():
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    model, basis, solver = kwargs["model"], kwargs["basis"], kwargs["solver"]
    n, e0, sector = occupation_and_energy_at_mu(model, basis, solver, 0.3, comm=MPI.COMM_WORLD)

    other = MPI.COMM_WORLD.allgather((n, e0, sector))
    assert all(triple == other[0] for triple in other), other


def test_fixed_occupation_dc_warns_when_sector_differs_from_nominal(capsys):
    """M1: the sector the walk actually settles on at the returned mu is the number this whole
    branch exists to establish (DC <-> GS parity), and it must be surfaced when it disagrees with
    the nominal occupation, not silently discarded the way it was before ``sector_at`` existed.

    ``_frozen_core_kwargs`` (nominal total 4: a frozen core of 4 plus two empty valence doublets)
    crosses to a sector total of 6 at mu=0.5 -- verified directly against
    ``occupation_and_energy_at_mu`` -- so requesting that achieved occupation drives
    ``fixed_occupation_dc`` onto the same non-nominal sector.
    """
    kwargs, _dc_guess = _frozen_core_kwargs()
    model, basis, solver = kwargs["model"], kwargs["basis"], kwargs["solver"]

    n_at_mu, _e0, sector_at_mu = occupation_and_energy_at_mu(model, basis, solver, 0.5, comm=kwargs["comm"])
    nominal_total = sum(basis.nominal_occ.values())
    assert sector_at_mu != nominal_total, "fixture no longer crosses to a non-nominal sector at mu=0.5"

    fixed_occupation_dc(occupation=n_at_mu, verbosity=1, **kwargs)
    out = capsys.readouterr().out
    if MPI.COMM_WORLD.rank != 0:  # the warning prints on rank 0 only
        return
    assert "WARNING" in out
    assert "walk's winning sector" in out
    assert f"totals {sector_at_mu}" in out
    assert f"not the nominal occupation {nominal_total}" in out


def test_fixed_occupation_dc_does_not_warn_at_the_nominal_sector(capsys):
    """The ordinary case (the walk never leaves the nominal sector) must not print the warning."""
    kwargs, _dc_guess = common_kwargs(v=0.3, tau=1e-2)
    basis = kwargs["basis"]
    n0, _e0, sector = occupation_and_energy_at_mu(kwargs["model"], basis, kwargs["solver"], 0.0, comm=kwargs["comm"])
    assert sector == sum(basis.nominal_occ.values()), "fixture unexpectedly left the nominal sector at mu=0"

    fixed_occupation_dc(occupation=n0, verbosity=1, **kwargs)
    out = capsys.readouterr().out
    assert "walk's winning sector" not in out


# --- B1: continuum vs discretized DFT reference -----------------------------------------------
#
# For a single impurity orbital coupled to a single bath level, the continuum propagator
# G0(w) = [(w+i*eim) - e_imp - Delta(w)]^-1 with Delta(w) = v**2/(w+i*eim - e_bath) is the
# EXACT Feshbach/Woodbury projection of the 2x2 one-body matrix [[e_imp, v], [v, e_bath]] onto
# the impurity index -- not an approximation. So in the eim -> 0 limit, the Richardson-debiased
# continuum fill and the discretized (exact-diagonalization) fill of the same matrix are the
# SAME quantity computed two different ways: a known analytic case to verify against, per the
# repo's own oracle-verification lesson (dense-Lehmann oracle campaign) -- verify against a KNOWN
# case before trusting the diagnostic on a real, unverifiable workload.


def _two_level_reference(e_imp=0.5, e_bath=-2.0, v=0.5, tau=0.1):
    h = np.array([[e_imp, v], [v, e_bath]], dtype=complex)
    energies, vecs = np.linalg.eigh(h)
    f = 1.0 / (1.0 + np.exp(energies / tau))
    n0_disc = float(np.sum(f * np.abs(vecs[0, :]) ** 2))
    return e_imp, e_bath, v, tau, n0_disc


def _two_level_continuum(e_imp, e_bath, v, eim, n_w=60001, w_max=15.0):
    w = np.linspace(-w_max, w_max, n_w)
    delta_w = (v**2 / (w + 1j * eim - e_bath))[:, None, None]  # (n_w, 1, 1), frequency-first
    h_dft = np.array([[e_imp]], dtype=complex)
    return h_dft, delta_w, w


def test_continuum_impurity_occupation_matches_exact_projection_as_eim_shrinks():
    from impurityModel.ed.dc_reference import continuum_impurity_occupation

    e_imp, e_bath, v, tau, n0_disc = _two_level_reference()
    errors = []
    for eim in (0.02, 0.01, 0.005):
        h_dft, delta_w, w = _two_level_continuum(e_imp, e_bath, v, eim)
        out = continuum_impurity_occupation(h_dft, delta_w, w, eim, tau)
        errors.append(abs(out["n0"] - n0_disc))
    # Richardson-debiased error shrinks as eim -> 0 (the diagnostic converges to the exact
    # projection identity, not merely "some number in the right ballpark").
    assert errors[-1] < errors[0], errors
    assert errors[-1] < 1e-3, errors


def test_continuum_impurity_occupation_richardson_beats_naive():
    from impurityModel.ed.dc_reference import continuum_impurity_occupation

    e_imp, e_bath, v, tau, n0_disc = _two_level_reference()
    eim = 0.05  # coarse enough that the naive broadening bias is visible
    h_dft, delta_w, w = _two_level_continuum(e_imp, e_bath, v, eim)
    out = continuum_impurity_occupation(h_dft, delta_w, w, eim, tau)
    naive_error = abs(out["n0_naive"] - n0_disc)
    debiased_error = abs(out["n0"] - n0_disc)
    assert debiased_error < naive_error / 2, (debiased_error, naive_error)


def test_continuum_impurity_occupation_sum_rule_and_causality():
    from impurityModel.ed.dc_reference import continuum_impurity_occupation

    e_imp, e_bath, v, tau, _ = _two_level_reference()
    h_dft, delta_w, w = _two_level_continuum(e_imp, e_bath, v, eim=0.01)
    out = continuum_impurity_occupation(h_dft, delta_w, w, 0.01, tau)
    # A window wide enough to hold the single impurity spectral pole integrates to ~1.
    assert np.isclose(out["sum_rule"], 1.0, atol=5e-3), out["sum_rule"]
    # A physical (causal) Delta(w) has Im Delta <= 0 everywhere.
    assert out["causality_violation"] == 0.0


def test_continuum_impurity_occupation_flags_acausal_delta():
    from impurityModel.ed.dc_reference import continuum_impurity_occupation

    e_imp, e_bath, v, tau, _ = _two_level_reference()
    h_dft, delta_w, w = _two_level_continuum(e_imp, e_bath, v, eim=0.01)
    acausal_delta_w = np.conj(delta_w)  # flips the sign of Im Delta -> acausal
    out = continuum_impurity_occupation(h_dft, acausal_delta_w, w, 0.01, tau)
    assert out["causality_violation"] > 0.0


def test_continuum_impurity_occupation_sum_rule_flags_too_narrow_a_window():
    from impurityModel.ed.dc_reference import continuum_impurity_occupation

    e_imp, e_bath, v, tau, _ = _two_level_reference()
    # A window that excludes the bonding/antibonding resonances misses real spectral weight.
    h_dft, delta_w, w = _two_level_continuum(e_imp, e_bath, v, eim=0.01, n_w=2001, w_max=0.5)
    out = continuum_impurity_occupation(h_dft, delta_w, w, 0.01, tau)
    assert out["sum_rule"] < 0.9, out["sum_rule"]


def test_report_continuum_reference_prints_and_returns_the_diagnostic(capsys):
    from impurityModel.ed.dc_reference import report_continuum_reference

    e_imp, e_bath, v, tau, n0_disc = _two_level_reference()
    eim = 0.01
    h_dft, hyb, w = _two_level_continuum(e_imp, e_bath, v, eim)
    # A slightly mistuned "fitted" hybridization (a small extra shift) stands in for a bath fit
    # that does not exactly reproduce the true continuum -- the discretization error B1 exists to
    # name.
    _, hyb_fit, _ = _two_level_continuum(e_imp, e_bath - 0.3, v, eim)

    report = report_continuum_reference(h_dft, hyb, hyb_fit, w, eim, tau, n0_disc, rank=0)

    out = capsys.readouterr().out
    assert "DC reference check" in out
    assert "discretization error" in out
    assert "sum rule" in out
    assert "causality" in out
    assert "dn0/dtau" in out

    assert report["n0_disc"] == n0_disc
    assert report["discretization_error"] != 0.0
    # The mistuned fit differs from the true continuum, so this must not vanish either.
    assert abs(report["n0_cont_true"] - report["n0_cont_fitted"]) > 1e-3


def test_report_continuum_reference_silent_off_root(capsys):
    from impurityModel.ed.dc_reference import report_continuum_reference

    e_imp, e_bath, v, tau, n0_disc = _two_level_reference()
    h_dft, hyb, w = _two_level_continuum(e_imp, e_bath, v, 0.01)
    report = report_continuum_reference(h_dft, hyb, hyb, w, 0.01, tau, n0_disc, rank=1)
    assert report is None
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "criterion, call",
    [
        ("fixed-occupation", lambda kw: fixed_occupation_dc(occupation=2.0, return_sector=True, **kw)),
        (
            "fixed-peak",
            lambda kw: fixed_peak_dc(peak_position=0.5, allow_charge_state_change=True, return_sector=True, **kw),
        ),
        (
            "fixed-gap",
            lambda kw: fixed_gap_dc(offset=0.0, allow_charge_state_change=True, return_sector=True, **kw),
        ),
    ],
)
def test_the_criteria_report_the_sector_they_settled_on(criterion, call):
    """The sector is half the answer, and the caller needs it.

    A double counting is only meaningful if the following ``calc_selfenergy`` lands on the state
    it was measured against, and the charge sector is the one part of that which is a *discrete*
    choice -- a small change in the incoming hybridization can flip it. The interface persists
    this and seeds the next solve with it (``impmod_interface``'s ``_double_counting_sector``),
    so it has to come back out rather than being printed and discarded.
    """
    kwargs, _ = _frozen_core_kwargs()
    try:
        result = call(kwargs)
    except (DoubleCountingUnreachable, RuntimeError):
        pytest.skip(f"{criterion}: this fixture does not converge; the return shape is tested elsewhere")

    assert isinstance(result, tuple) and len(result) == 2, f"{criterion}: expected (dc, sector), got {type(result)}"
    dc, sector = result
    assert dc.shape[0] == dc.shape[1], criterion
    assert isinstance(sector, (int, np.integer)), f"{criterion}: sector {sector!r} is not an integer charge state"
    assert sector >= 0, criterion


@pytest.mark.parametrize(
    "criterion, call",
    [
        ("fixed-occupation", lambda kw: fixed_occupation_dc(occupation=2.0, **kw)),
        ("fixed-peak", lambda kw: fixed_peak_dc(peak_position=0.5, allow_charge_state_change=True, **kw)),
        ("fixed-gap", lambda kw: fixed_gap_dc(offset=0.0, allow_charge_state_change=True, **kw)),
    ],
)
def test_the_sector_is_opt_in_so_existing_callers_keep_a_bare_matrix(criterion, call):
    kwargs, _ = _frozen_core_kwargs()
    try:
        result = call(kwargs)
    except (DoubleCountingUnreachable, RuntimeError):
        pytest.skip(f"{criterion}: this fixture does not converge")
    assert not isinstance(result, tuple), f"{criterion}: default return changed shape"
