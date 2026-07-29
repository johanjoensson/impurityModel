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
