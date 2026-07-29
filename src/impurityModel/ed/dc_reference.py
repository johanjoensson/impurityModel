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
    if rank != 0:
        return
    if not (n0 >= total_impurity_orbitals - occ_tol or n0 <= occ_tol):
        return
    print(
        f"WARNING: the DFT reference occupation N0 = {n0:.4f} is saturated at the "
        f"{'full' if n0 > occ_tol else 'empty'} impurity shell "
        f"(of {total_impurity_orbitals} spin-orbitals): the discretized bath places no "
        "impurity spectral weight across the Fermi level (typical for coarse "
        f"valence-only bath fits), {advice}",
        flush=True,
    )


def _reference_impurity_occupation(model, tau, *, warn=True):
    """The DFT impurity occupation the static schemes default to, saturation-checked."""
    n = _noninteracting_impurity_occupation(model.h0, model.impurity_indices, model.n_spin_orbitals, tau)
    if warn:
        _warn_if_reference_saturated(n, len(model.impurity_indices), _SATURATION_ADVICE["static"])
    return n
