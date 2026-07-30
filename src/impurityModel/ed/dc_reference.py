r"""The DFT reference filling every double-counting scheme measures against.

``model.h0`` is the KS/DFT Hamiltonian of the ``h0 - dc + U`` contract, so its *raw* Fermi
filling is the DFT impurity occupation -- the target :func:`dc_criteria.fixed_occupation_dc`
pins when given none, and the point at which :mod:`dc_static`'s three formulas are evaluated.
No double counting is subtracted before filling: subtracting a realistic ``dc`` first sinks the
impurity levels far below the Fermi level and saturates the reference at the full shell (NiO:
``n0`` pinned at 10 instead of ~8.6 for a ~4 Ry dc).

That reference is a property of the *discretized* bath, and it pins at the full (or empty) shell
whenever the fit places no impurity weight across the Fermi level. A saturated reference is
wrong by O(1) electron, which for FLL is several eV -- enough to move NiO from Mott-insulating
to nearly metallic -- and it is wrong silently, so the check belongs here, to the reference,
rather than to any one scheme that consumes it.

Every computation in this module is deterministic NumPy on the replicated ``h0``: no MPI
collective, identical on every rank.

.. admonition:: Decision (R0): grand-canonical fill at mu=0, not a fill matched to the ED's
   nominal total

   This module fills ``h0`` **grand-canonically** at ``mu = 0``, with no constraint on the
   resulting total electron count. That total need not (and on the real NiO archive, does not)
   match the many-body basis's own nominal total -- the bath-valence-orbital count plus the
   nominal impurity occupation the basis is seeded from (``generate_initial_basis``). This was
   investigated because it looked like a bug: on the ``nio_15`` archive (iteration 1, 152
   spin-orbitals, 10 impurity), the basis's nominal total is 150 (142 valence-classified bath
   orbitals + 8 nominal impurity), while the grand-canonical fill at ``mu = 0`` gives a total of
   only ``147.9925`` -- a ~2-electron gap.

   The resolution: the raw one-body spectrum of ``h0`` has a **genuine gap**, ``(-0.0154,
   +0.0209)`` Ry (~0.036 Ry wide), that ``mu = 0`` sits well inside -- 148 single-particle levels
   lie below it (occupied to round-off at ``tau = 0.0025``), and exactly one 4-fold-degenerate
   manifold sits just above it (occupation ``~0.0002`` each at ``mu = 0``, negligible). That
   148-electron fill is what a real DFT calculation would give for this system: robust,
   gap-protected, and insensitive to exactly where ``mu`` sits inside the gap. Forcing the fill to
   match the basis's nominal total of 150 instead requires pushing ``mu`` up to ``+0.0209`` Ry,
   landing it **exactly on** that near-degenerate manifold and half-filling it -- a number that is
   66% bath character and 34% impurity character, and is fragile in the sense that it depends on
   precisely how the bath discretization placed that one manifold, not on a gap. The 150-total
   mismatch is therefore not evidence the grand-canonical fill is wrong; it reflects that the
   per-orbital valence/conduction classification (:func:`symmetries.classify_bath_occupation`,
   which looks only at the sign of each bath orbital's own diagonal energy) is a coarse
   bookkeeping device for seeding the many-body basis, not a claim about the true hybridized
   single-particle spectrum. The resulting ``n0 = 8.6252`` (a fractional, hybridization-driven
   impurity occupation) is consistent with published covalent d-occupations for NiO.

   :func:`test.gf.test_fixed_dc.test_reference_is_gap_protected_not_nominal_total_matched` pins
   this on a toy model with the same structure (a genuine gap, and a nominal total that
   deliberately disagrees with the gap-fill total), so a future change cannot silently make the
   reference track the basis's nominal total instead of the true one-body gap.
"""

import numpy as np
from mpi4py import MPI

from impurityModel.ed.lie_algebra import extract_tensors
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator


def _noninteracting_impurity_rho(h0_op, impurity_indices, n_spin_orbitals, tau):
    r"""Thermal impurity density matrix of the non-interacting ``h0`` at the Fermi level.

    Diagonalise the full one-body Hamiltonian ``h0`` (impurity *and* bath), occupy the
    single-particle levels with Fermi-Dirac statistics at chemical potential ``mu = 0`` -- the
    RSPt convention places the Fermi level at zero -- and return the impurity block of the
    resulting one-particle density matrix:

    .. math::
        \rho = \sum_n f(\epsilon_n)\, |v_n\rangle\langle v_n|,\quad
        f(\epsilon) = \frac{1}{1 + e^{\epsilon / \tau}}.

    Because it hybridises the impurity with the bath before tracing, ``Tr rho_imp`` is the DFT
    impurity occupation of a wide-window p-d model, which is the target :func:`fixed_occupation_dc`
    pins when the caller supplies none. ``model.h0`` is the KS/DFT Hamiltonian of the
    ``h0 - dc + U`` model contract (it already contains the DFT mean-field interaction -- that is
    what the double counting corrects for), so the DFT reference is the filling of the *raw*
    ``h0`` and this function deliberately takes no double counting: subtracting a realistic dc
    first sinks the impurity levels far below the Fermi level and saturates the occupation at the
    full shell (NiO: n0 pinned at 10 instead of ~8.6 for a ~4 Ry dc). It is a deterministic NumPy
    computation on the replicated ``h0`` (no MPI collective), so every rank obtains an identical
    value.

    Parameters
    ----------
    h0_op : dict or ManyBodyOperator
        Non-interacting Hamiltonian in single-index operator form (``model.h0``).
    impurity_indices : sequence of int
        Impurity spin-orbital indices (the block traced over).
    n_spin_orbitals : int
        Total number of spin-orbitals (impurity + bath).
    tau : float
        Fundamental temperature ``k_B T`` in the energy units of ``h0``. ``tau <= 0`` fills
        every level below the Fermi level (a zero-temperature step).

    Returns
    -------
    numpy.ndarray
        The impurity block of the density matrix, ``(n_imp, n_imp)`` complex.
    """
    h = extract_tensors(ManyBodyOperator(h0_op), n_orb=n_spin_orbitals, two_body=False)[0]
    energies, vecs = np.linalg.eigh(h)
    if tau > 0:
        # 1/(1 + exp(e/tau)) without overflow warnings: exp saturates to inf/0, giving f -> 0/1.
        with np.errstate(over="ignore"):
            occupations = 1.0 / (1.0 + np.exp(energies / tau))
    else:
        occupations = (energies < 0).astype(float)
    rho = (vecs * occupations) @ vecs.conj().T
    impurity_ix = np.ix_(list(impurity_indices), list(impurity_indices))
    return rho[impurity_ix]


def _noninteracting_impurity_occupation(h0_op, impurity_indices, n_spin_orbitals, tau):
    """Thermal impurity occupation ``Tr rho_imp``; see :func:`_noninteracting_impurity_rho`."""
    rho = _noninteracting_impurity_rho(h0_op, impurity_indices, n_spin_orbitals, tau)
    return float(np.real(np.trace(rho)))


# What to do about a saturated reference, per scheme family. The searches take a target on the
# double-counting line; the static formulas take the occupation (or density matrix) directly.
_SATURATION_ADVICE = {
    "search": (
        "so the self-consistent occupation criterion is meaningless and the search will very "
        "likely fail as unreachable. Supply an explicit occupation target instead (RSPt "
        "interface: 'occ <N>' on the double-counting line), or improve the bath discretization "
        "around the Fermi level."
    ),
    "static": (
        "so the double counting is evaluated at a filling the material does not have. Supply the "
        "occupation (n=) or density matrix (rho=) explicitly instead, or improve the bath "
        "discretization around the Fermi level."
    ),
}


def _warn_if_reference_saturated(n0, total_impurity_orbitals, advice, occ_tol=1e-2, rank=None):
    """Warn when the DFT reference filling has pinned at the full or empty shell.

    Saturation is a threshold phenomenon of coarse bath discretizations, not a smooth error:
    impurity weight crosses the Fermi level only when level repulsion pushes a mixed state
    through it (NiO: ``n0 = 8.63`` at 15 bath states per block, exactly ``10.0`` at 1 and 5). A
    saturated reference is silently wrong rather than noisily wrong, and it is wrong by O(1)
    electron -- for FLL that is ``U * dN - J * dN / 2``, several eV, enough to move NiO from
    Mott-insulating to nearly metallic. Every scheme that *defaults* to the reference routes
    through here; a caller supplying its own occupation is not warned, having chosen it.

    Printing only (no collective), so gating on rank 0 is safe.
    """
    if rank is None:
        rank = MPI.COMM_WORLD.rank
    saturated = n0 >= total_impurity_orbitals - occ_tol or n0 <= occ_tol
    if rank != 0 or not saturated:
        return saturated
    print(
        f"WARNING: the DFT reference occupation N0 = {n0:.4f} is saturated at the "
        f"{'full' if n0 > occ_tol else 'empty'} impurity shell "
        f"(of {total_impurity_orbitals} spin-orbitals): the discretized bath places no "
        "impurity spectral weight across the Fermi level (typical for coarse "
        f"valence-only bath fits), {advice}",
        flush=True,
    )
    return True


#: How far the DFT reference may sit from the nominal occupation before it is reported as
#: suspect: **one electron**. Not a tolerance to be tuned -- it is the spacing of the quantity
#: itself. The nominal occupation names a charge state (d7 / d8 / d9), so a reference more than
#: one electron away points at a *different* charge state than the model was set up for, whatever
#: the covalency. Genuine covalent deviation sits well inside it: NiO at 15 bath states gives
#: n0 = 8.6258 against a nominal 8, i.e. 0.63.
_NOMINAL_GAP_ELECTRONS = 1.0


def _warn_if_reference_far_from_nominal(n0, nominal_total, advice, rank=None):
    """Warn when the DFT reference names a different charge state than the nominal occupation.

    The complement of :func:`_warn_if_reference_saturated`, which fires only at exactly 0 or a
    full shell and so passes a reference that is grossly wrong without being pinned. That gap is
    not hypothetical: the *second* CSC iteration of a NiO archive reports ``n0 = 1.5446`` against
    a nominal 8 -- a runaway iterate whose DFT impurity level had moved 0.25 Ry (~3.4 eV) above
    the first's -- and the search converged on it silently, returning a double counting 3.14 Ry
    (~43 eV) from the guess. Neither the saturation check nor the convergence test had anything
    to say about it.

    Printing only (no collective), so gating on rank 0 is safe.
    """
    if rank is None:
        rank = MPI.COMM_WORLD.rank
    if rank != 0 or abs(n0 - nominal_total) <= _NOMINAL_GAP_ELECTRONS:
        return False
    print(
        f"WARNING: the DFT reference occupation N0 = {n0:.4f} lies {abs(n0 - nominal_total):.4f} "
        f"electrons from the nominal occupation {nominal_total}, i.e. it names a different charge "
        "state than the model was set up for. That is the signature of a diverging charge "
        f"self-consistency loop, or of an archive iteration that has run away, {advice}",
        flush=True,
    )
    return True


def _warn_if_not_fermi_referenced(h0_op, n_spin_orbitals, rank=None):
    """Verify the E_F = 0 convention that every sector comparison silently depends on.

    ``h0`` is *asserted* to be Fermi-referenced throughout this package -- the reference filling
    occupies levels below zero, and the occupation walk compares charge sectors at chemical
    potential zero -- and until now nothing checked it. An ``h0`` carrying an absolute energy
    offset (a Hartree- rather than Fermi-referenced KS Hamiltonian, or the wrong units) still
    diagonalises fine and still produces a number; it is simply the wrong number, and every
    downstream sector comparison inherits the error.

    The necessary condition is cheap: the one-body spectrum must **straddle** zero. If every level
    lies on one side, the Fermi level is not at zero (or the model is trivially full or empty) and
    no filling computed from it means anything. This catches the gross violation, which is the one
    that occurs; a spectrum that straddles zero at the wrong place is not detectable from ``h0``
    alone.

    Printing only (no collective), so gating on rank 0 is safe.
    """
    if rank is None:
        rank = MPI.COMM_WORLD.rank
    h = extract_tensors(ManyBodyOperator(h0_op), n_orb=n_spin_orbitals, two_body=False)[0]
    energies = np.linalg.eigvalsh(h)
    if rank != 0 or (energies.min() < 0.0 < energies.max()):
        return False
    side = "below" if energies.max() <= 0.0 else "above"
    print(
        f"WARNING: every one-body level of h0 lies {side} the Fermi level "
        f"(spectrum {energies.min():.4f} .. {energies.max():.4f}), so the E_F = 0 convention this "
        "package asserts does not hold for this Hamiltonian. The DFT reference filling and every "
        "charge-sector comparison in the ground-state search are computed at chemical potential "
        "zero and are meaningless if the Fermi level is elsewhere. Check that h0 is the "
        "Fermi-referenced KS Hamiltonian and that its units match tau.",
        flush=True,
    )
    return True


def _reference_impurity_occupation(model, tau, *, warn=True):
    """The DFT impurity occupation the static schemes default to, saturation-checked."""
    n = _noninteracting_impurity_occupation(model.h0, model.impurity_indices, model.n_spin_orbitals, tau)
    if warn:
        _warn_if_reference_saturated(n, len(model.impurity_indices), _SATURATION_ADVICE["static"])
    return n


def discretized_impurity_occupation(model, tau):
    """The DFT reference occupation ``n0`` that :func:`dc_criteria.fixed_occupation_dc` pins to
    when given no explicit target -- exposed for external callers (e.g. the RSPt interface's B1
    diagnostic, :func:`report_continuum_reference`) that need the same "n0 the criterion uses
    today" without duplicating the Fermi fill and its saturation check. Unwarned: the caller is
    expected to have its own reporting (or to be calling this purely to feed a comparison), not to
    silently re-trigger the saturation warning a second time per call.
    """
    return _reference_impurity_occupation(model, tau, warn=False)


def _continuum_fermi_filling(h_dft, delta_w, w, eim, tau):
    r"""Grand-canonical ``mu = 0`` filling of the continuum impurity propagator.

    .. math::
        G_0(\omega) = \left[(\omega + i\,\text{eim})\mathbb{1} - h_\text{dft} - \Delta(\omega)\right]^{-1}

    with :math:`\Delta(\omega)` a hybridization function (either RSPt's continuum one, or the
    fitted bath's own reconstruction of it -- both are valid inputs, which is what makes the
    differential comparison in :func:`continuum_impurity_occupation` possible).

    Parameters
    ----------
    h_dft : (n_orb, n_orb) complex ndarray
        Impurity block of the DFT Hamiltonian, same basis as ``delta_w``.
    delta_w : (n_w, n_orb, n_orb) complex ndarray
        Hybridization function on the real mesh ``w``, evaluated ``i*eim`` above the axis.
        Frequency-first, matching this package's Green's-function convention (and
        ``impmod_interface``'s own ``hyb``/``hyb_fit`` layout after its ``moveaxis``).
    w : (n_w,) real ndarray
        Real-frequency mesh, ascending.
    eim : float
        Broadening for the ``(w + i*eim)`` term. Need not equal the broadening ``delta_w`` was
        itself evaluated at -- doubling it here (with ``delta_w`` held fixed) is exactly the
        finite-difference handle :func:`continuum_impurity_occupation` uses to debias.
    tau : float
        Fermi-Dirac temperature for the fill.

    Returns
    -------
    n : float
        ``Tr rho_imp`` from Fermi-filling the continuum spectral function.
    sum_rule : float
        ``\int Tr A(w) dw``, which should be close to ``n_orb`` if the mesh/window captures the
        impurity spectral weight; a wide-window fit whose window misses real weight fails this.
    causality_violation : float
        ``max(0, max Im Delta(w))``; zero iff :math:`\Delta` is causal (``Im Delta <= 0``)
        everywhere on the mesh.
    """
    n_orb = h_dft.shape[0]
    identity = np.eye(n_orb, dtype=complex)
    z = (w + 1j * eim)[:, None, None] * identity[None, :, :]
    g0 = np.linalg.inv(z - delta_w - h_dft[None, :, :])
    dos = -np.trace(g0, axis1=1, axis2=2).imag / np.pi
    if tau > 0:
        with np.errstate(over="ignore"):
            f = 1.0 / (1.0 + np.exp(w / tau))
    else:
        f = (w < 0).astype(float)
    n = float(np.trapezoid(dos * f, w))
    sum_rule = float(np.trapezoid(dos, w))
    causality_violation = float(max(0.0, np.max(delta_w.imag)))
    return n, sum_rule, causality_violation


def continuum_impurity_occupation(h_dft, delta_w, w, eim, tau, *, richardson=True):
    """Continuum DFT impurity occupation from a real-axis hybridization function.

    A naive real-axis integral at finite ``eim`` is biased toward half filling (measured:
    ``eim=0.02`` -> bias -0.24 e, 24x a typical ``occ_tol``): the Lorentzian broadening smears
    weight across the Fermi level regardless of where the true spectral weight sits. Richardson
    extrapolation using a second fill at matrix broadening ``2*eim`` (``delta_w`` held fixed --
    only the ``(w + i*eim)`` term changes) cancels the leading-order bias, since the naive fill is
    ``n(eim) = n_true + C*eim + O(eim**2)``: ``n_true ~= 2*n(eim) - n(2*eim)``.

    Returns a dict with ``n0`` (Richardson-debiased if requested, else the naive fill),
    ``n0_naive``, ``n0_2eim``, ``sum_rule``, ``n_orb`` and ``causality_violation`` -- see
    :func:`_continuum_fermi_filling` for the latter three.
    """
    n0_naive, sum_rule, causality_violation = _continuum_fermi_filling(h_dft, delta_w, w, eim, tau)
    if richardson:
        n0_2eim, _, _ = _continuum_fermi_filling(h_dft, delta_w, w, 2 * eim, tau)
        n0 = 2 * n0_naive - n0_2eim
    else:
        n0_2eim = None
        n0 = n0_naive
    return {
        "n0": n0,
        "n0_naive": n0_naive,
        "n0_2eim": n0_2eim,
        "sum_rule": sum_rule,
        "n_orb": h_dft.shape[0],
        "causality_violation": causality_violation,
    }


def continuum_dn0_dtau(h_dft, delta_w, w, eim, tau, *, richardson=True):
    """Finite-difference ``dn0/dtau`` of the (optionally Richardson-debiased) continuum fill.

    Gap-protected materials (a genuine gap straddling ``mu=0``) are insensitive to ``tau``; a
    metal is not, and nothing has checked that sensitivity before -- see the module docstring's
    R0 admonition for why the gap protection matters for the *discretized* reference, which this
    is the continuum-side analogue of.
    """
    n_tau = continuum_impurity_occupation(h_dft, delta_w, w, eim, tau, richardson=richardson)["n0"]
    n_2tau = continuum_impurity_occupation(h_dft, delta_w, w, eim, 2 * tau, richardson=richardson)["n0"]
    return (n_2tau - n_tau) / tau


def report_continuum_reference(h_dft, hyb, hyb_fit, w, eim, tau, n0_disc, *, rank=None):
    """Print and return the B1 two-way DFT-reference diagnostic: continuum vs discretized ``n0``.

    Computes the continuum occupation twice, from the same ``h_dft``/mesh/``eim``/quadrature but
    two different hybridization functions:

    - ``hyb`` -- RSPt's true continuum :math:`\\Delta(\\omega)`.
    - ``hyb_fit`` -- the fitted bath's own reconstruction of it (already computed by the caller
      for the existing ``fit_dev`` report).

    Differencing these two **cancels the broadening bias**, since both are integrated through the
    identical quadrature at the identical ``eim`` -- what remains is the discretization error of
    the bath fit alone, which is the number this diagnostic exists to name.

    As a self-consistency check on the diagnostic itself (not the physics): in the ``eim -> 0``
    limit, ``continuum_impurity_occupation(h_dft, hyb_fit, ...)`` and ``n0_disc`` (the exact
    diagonalization of the same finite one-body problem) are the *same quantity* computed two
    different ways -- a Feshbach/Woodbury projection identity, not an approximation -- so a large
    disagreement between them indicates a bug in this diagnostic, not a physics finding.

    This is a **diagnostic only**: it does not change what :func:`dc_criteria.fixed_occupation_dc`
    pins to. Printing only (no collective), so gating on rank 0 is safe.
    """
    if rank is None:
        rank = MPI.COMM_WORLD.rank
    if rank != 0:
        return None
    cont_true = continuum_impurity_occupation(h_dft, hyb, w, eim, tau)
    cont_fit = continuum_impurity_occupation(h_dft, hyb_fit, w, eim, tau)
    dn0_dtau_true = continuum_dn0_dtau(h_dft, hyb, w, eim, tau)
    discretization_error = cont_true["n0"] - cont_fit["n0"]
    fit_consistency_check = cont_fit["n0"] - n0_disc
    n_orb = cont_true["n_orb"]
    print(
        "DC reference check (B1): discretized n0 (used by the criterion) = "
        f"{n0_disc:.4f}; continuum n0 from the true Delta(w) = {cont_true['n0']:.4f}; from the "
        f"fitted Delta(w) = {cont_fit['n0']:.4f}\n"
        f"  discretization error (true continuum - fitted continuum) = {discretization_error:+.4f} "
        "(the bath-fit artefact the criterion cannot see)\n"
        "  diagnostic self-consistency (fitted continuum - discretized; should be small "
        f"as eim -> 0) = {fit_consistency_check:+.4f}\n"
        f"  sum rule (expect ~{n_orb}): true {cont_true['sum_rule']:.3f}, "
        f"fitted {cont_fit['sum_rule']:.3f}\n"
        f"  causality (max Im Delta(w), expect <= 0): true {cont_true['causality_violation']:.2e}, "
        f"fitted {cont_fit['causality_violation']:.2e}\n"
        f"  dn0/dtau (true continuum) = {dn0_dtau_true:+.4f} per unit tau "
        "(large values flag a metal-like Fermi-smearing sensitivity)",
        flush=True,
    )
    return {
        "n0_disc": n0_disc,
        "n0_cont_true": cont_true["n0"],
        "n0_cont_fitted": cont_fit["n0"],
        "discretization_error": discretization_error,
        "fit_consistency_check": fit_consistency_check,
        "sum_rule_true": cont_true["sum_rule"],
        "sum_rule_fitted": cont_fit["sum_rule"],
        "causality_true": cont_true["causality_violation"],
        "causality_fitted": cont_fit["causality_violation"],
        "dn0_dtau_true": dn0_dtau_true,
    }
