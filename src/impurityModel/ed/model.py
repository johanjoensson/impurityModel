"""The impurity model and the option groups the solvers consume.

This module is the single construction point for the *physics* of an impurity problem --
the non-interacting Hamiltonian ``h0``, the Coulomb tensor ``u4``, the impurity orbital
layout and the rotation to spherical harmonics -- bundled into one :class:`ImpurityModel`.
The solver entry points (``calc_selfenergy``, ``calc_susceptibility_workflow`` and the
spectra driver) take an :class:`ImpurityModel` together with a few small option-group
dataclasses (:class:`Meshes`, :class:`BasisOptions`, :class:`SolverOptions`) instead of a
long flat argument list. Embedded callers (the RSPt interface) build the model from arrays
they already hold in memory; the CLIs build it from a file via :meth:`ImpurityModel.from_h0_file`.

Layering: this is a bottom-layer module. It imports only ``atomic_physics``,
``operator_algebra`` and ``hamiltonian_io`` -- never a solver -- so it can be constructed
without pulling in the many-body machinery.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

from impurityModel.ed import atomic_physics
from impurityModel.ed.hamiltonian_io import get_noninteracting_hamiltonian_operator
from impurityModel.ed.operator_algebra import c2i

__all__ = [
    "ImpurityModel",
    "Meshes",
    "BasisOptions",
    "SolverOptions",
    "SpectraOptions",
    "atomic_u4",
]


def atomic_u4(l: int, slater) -> np.ndarray:
    """Assemble the single-shell atomic Coulomb tensor ``u4`` in the RSPt convention.

    Builds the spherical-harmonics Slater-Condon interaction of one ``l``-shell (via
    :func:`atomic_physics.getUop`) and lays it out as the dense rank-4 tensor the solver
    expects, following RSPt's storage convention: ``u4[i, j, k, l]`` multiplies
    :math:`c^\\dagger_i c^\\dagger_j c_l c_k`, so the process
    :math:`c^\\dagger_i c^\\dagger_j c_k c_l` is stored with ``k`` and ``l`` swapped.

    This is the assembly that used to be copy-pasted into the self-energy and
    susceptibility CLIs; both now call this helper.

    Parameters
    ----------
    l : int
        Angular momentum of the correlated shell (e.g. 2 for a d-shell).
    slater : sequence of float
        Slater-Condon parameters ``F^0, F^1, ...`` passed to
        :func:`atomic_physics.getUop` as ``R``.

    Returns
    -------
    numpy.ndarray
        Complex ``(n, n, n, n)`` tensor with ``n = 2 * (2 * l + 1)`` spin-orbitals.
    """
    n_imp_spin_orbitals = 2 * (2 * l + 1)
    u4 = np.zeros((n_imp_spin_orbitals,) * 4, dtype=complex)
    u_op = atomic_physics.getUop(l1=l, l2=l, l3=l, l4=l, R=slater)
    n_baths_for_c2i = OrderedDict({l: 0})
    for process, val in u_op.items():
        i = c2i(n_baths_for_c2i, process[0][0])
        j = c2i(n_baths_for_c2i, process[1][0])
        k = c2i(n_baths_for_c2i, process[2][0])
        m = c2i(n_baths_for_c2i, process[3][0])
        u4[i, j, m, k] = 2.0 * val
    return u4


@dataclass(frozen=True)
class ImpurityModel:
    """The physics of one impurity problem: ``h0``, ``u4`` and the orbital layout.

    ``h0`` is the *non-interacting* Hamiltonian in single-index operator form (the format
    produced by :func:`operator_algebra.matrixToIOp` and the readers in
    :mod:`impurityModel.ed.hamiltonian_io`); the interacting Hamiltonian ``h0 + U(u4)`` is
    assembled inside the solver. Every array is in the caller's input (correlated) basis;
    results are rotated back to it.

    Attributes
    ----------
    h0 : dict
        Non-interacting Hamiltonian, ``{process: amplitude}`` with integer spin-orbital
        indices. The impurity orbitals come first, then the bath orbitals.
    u4 : numpy.ndarray or None
        Dense ``(n_imp, n_imp, n_imp, n_imp)`` Coulomb tensor over the impurity
        spin-orbitals, in the RSPt convention (see :func:`atomic_u4`). ``None`` for a
        non-interacting model.
    impurity_orbitals : dict[int, list[int]]
        Flat impurity spin-orbital index lists, keyed by an arbitrary group label. The bath
        orbitals (everything else in ``h0``) and their valence/conduction split are derived
        from the Hamiltonian inside the solver.
    rot_to_spherical : numpy.ndarray or dict
        Rotation from the impurity input basis to spherical harmonics, used only for the
        L/S/J observable reporting. An identity when the input basis already is spherical.
    bath_states : tuple or None
        Explicit ``(valence_baths, conduction_baths)`` layout (each a
        ``dict[group, list[list[int]]]``), used by the spectra driver where the multi-shell
        bath partition is built up-front. ``None`` in the self-energy path, where the solver
        derives the bath orbitals and their valence/conduction split from ``h0``.
    n_spin_orbitals : int, optional
        Total number of spin-orbitals (impurity + bath). Derived from ``h0`` when not given.
    """

    h0: dict
    impurity_orbitals: dict
    rot_to_spherical: Union[np.ndarray, dict]
    u4: Optional[np.ndarray] = None
    bath_states: Optional[tuple] = None
    n_spin_orbitals: int = field(default=0)

    def __post_init__(self):
        # Derive the spin-orbital count from h0 when the caller did not supply it. h0 is a
        # frozen field, so set the derived value through object.__setattr__.
        if not self.n_spin_orbitals:
            object.__setattr__(self, "n_spin_orbitals", _count_spin_orbitals(self.h0))

    @property
    def impurity_indices(self) -> list:
        """Sorted flat list of impurity spin-orbital indices across all groups."""
        return sorted(orb for orbs in self.impurity_orbitals.values() for orb in orbs)

    @classmethod
    def from_h0_file(
        cls,
        h0_filename: str,
        l: int,
        n_baths: int,
        slater,
        xi: float = 0.0,
        h_field=(0.0, 0.0, 0.0001),
        n_val_baths: Optional[int] = None,
        rank: int = 0,
        verbose: bool = False,
    ) -> "ImpurityModel":
        """Build a single-correlated-shell model from a non-interacting ``h0`` file.

        Reads ``h0_filename`` (pickle / ``.dat`` / ``.json``, see
        :func:`hamiltonian_io.read_h0_operator`), dresses it with the shell's spin-orbit
        coupling and magnetic field, converts to single-index operator form, and pairs it
        with the atomic Coulomb tensor from :func:`atomic_u4`. This is the construction the
        self-energy and susceptibility CLIs use; the correlated shell is treated as the
        ``l``-shell that :func:`hamiltonian_io.get_noninteracting_hamiltonian_operator`
        dresses (SOC ``xi`` on it, magnetic field on the d-shell).

        Parameters
        ----------
        h0_filename : str
            File holding the non-interacting Hamiltonian.
        l : int
            Angular momentum of the correlated shell.
        n_baths : int
            Total number of bath spin-orbitals coupled to the shell.
        slater : sequence of float
            Slater-Condon parameters for the atomic Coulomb tensor.
        xi : float, optional
            Spin-orbit coupling of the correlated shell.
        h_field : tuple of float, optional
            Magnetic field ``(hx, hy, hz)``.
        n_val_baths : int, optional
            Number of valence bath states (only needed by the ``.json`` CF reader);
            defaults to ``n_baths``.
        rank, verbose
            Forwarded to the reader for rank-0 logging.

        Returns
        -------
        ImpurityModel
        """
        sum_baths = OrderedDict({l: n_baths})
        val_baths = OrderedDict({l: n_baths if n_val_baths is None else n_val_baths})
        # get_noninteracting_hamiltonian_operator dresses the 2p (l=1) and 3d (l=2) shells
        # with SOC and the d-shell with the magnetic field; the correlated shell's SOC is
        # supplied as the 3d entry, matching the self-energy CLI's [0, xi] call.
        h0_op = get_noninteracting_hamiltonian_operator(
            sum_baths, val_baths, [0.0, xi], tuple(h_field), h0_filename, rank, verbose
        )
        # Drop identically-zero terms before mapping to integer indices:
        # get_noninteracting_hamiltonian_operator unconditionally adds a 2p (l=1) SOC
        # operator whose terms are all 0.0 when xi_2p=0. Those carry l=1 labels that c2i
        # cannot map against a single l-shell nBaths dict, and they contribute nothing to H.
        h0_single_index = {
            tuple((c2i(sum_baths, spin_orb), action) for spin_orb, action in process): value
            for process, value in h0_op.items()
            if abs(value) != 0
        }
        n_imp_spin_orbitals = 2 * (2 * l + 1)
        return cls(
            h0=h0_single_index,
            u4=atomic_u4(l, slater),
            impurity_orbitals={l: list(range(n_imp_spin_orbitals))},
            rot_to_spherical=np.eye(n_imp_spin_orbitals, dtype=complex),
        )


def _count_spin_orbitals(h0) -> int:
    """Highest spin-orbital index in ``h0`` plus one (0 for an empty operator)."""
    items = h0.items() if hasattr(h0, "items") else h0.to_dict().items()
    highest = -1
    for process, _value in items:
        for index, _action in process:
            if index > highest:
                highest = index
    return highest + 1


@dataclass(frozen=True)
class Meshes:
    """Frequency meshes and the real-axis broadening.

    Attributes
    ----------
    iw : numpy.ndarray or None
        Matsubara frequency mesh (imaginary part; the solver multiplies by ``1j``). ``None``
        skips the Matsubara output.
    w : numpy.ndarray or None
        Real frequency mesh (eV). ``None`` skips the real-axis output.
    delta : float
        Smearing above the real axis (HWHM, eV).
    """

    iw: Optional[np.ndarray] = None
    w: Optional[np.ndarray] = None
    delta: float = 0.1


@dataclass(frozen=True)
class BasisOptions:
    """Many-body basis construction: occupation, restrictions and the determinant budget.

    Attributes
    ----------
    nominal_occ : dict or int
        Nominal impurity occupation. A dict keyed by the model's impurity groups, or a
        scalar/total that the solver distributes by energetic filling.
    mixed_valence : dict, int or None
        Mixed-valence scalar, forwarded per group to the basis.
    dN : int or None
        Allowed impurity occupation window (+-dN) for the Green's-function excited bases.
    truncation_threshold : int, float or None
        Global cap on the number of determinants per basis. ``None`` derives the cap from
        available per-rank memory; ``numpy.inf`` disables capping.
    chain_restrict : bool
        Whether to apply chain occupation restrictions.
    spin_flip_dj : bool
        Whether to generate spin-flipped determinants.
    occ_cutoff : float
        Occupation cutoff.
    slater_weight_min : float
        Minimum Slater-determinant weight retained.
    tau : float
        Fundamental temperature ``k_B * T`` (eV).
    """

    nominal_occ: Any
    mixed_valence: Any = None
    dN: Optional[int] = None
    truncation_threshold: Optional[Union[int, float]] = None
    chain_restrict: bool = True
    spin_flip_dj: bool = False
    occ_cutoff: float = 1e-12
    slater_weight_min: float = float(np.sqrt(np.finfo(float).eps))
    tau: float = 0.002


@dataclass(frozen=True)
class SolverOptions:
    """Green's-function kernel and eigensolver knobs.

    Attributes
    ----------
    reort : str, float or None
        Reorthogonalization mode of the block-Lanczos Green's function.
    dense_cutoff : int
        Use a dense eigensolver below this matrix size.
    sparse_green : bool
        Whether the Green's function uses the sparse block-Lanczos path.
    gf_method : {"lanczos", "bicgstab", "sliced"}
        Green's-function kernel. See :func:`impurityModel.ed.greens_function.get_Greens_function`.
    """

    reort: Any = None
    dense_cutoff: int = 500
    sparse_green: bool = True
    gf_method: str = "lanczos"


@dataclass(frozen=True)
class SpectraOptions:
    """Spectroscopy meshes, broadenings, polarizations and projectors for the spectra driver.

    Every field defaults to ``None`` for the arrays the driver fills in with its standard
    grids/polarizations (so an API caller can override just the pieces it cares about) and to
    the historical literal for the scalars. The frequency meshes are resolved once inside
    :func:`impurityModel.ed.get_spectra.run_spectra`.

    Attributes
    ----------
    w : numpy.ndarray or None
        XAS / PES / XPS energy mesh (eV). ``None`` -> ``linspace(-25, 25, 3001)``.
    delta : float
        Core-hole-lifetime broadening (HWHM, eV).
    epsilons : sequence or None
        XAS polarization vectors. ``None`` -> the Cartesian ``x, y, z`` triple.
    wLoss : numpy.ndarray or None
        RIXS energy-loss mesh. ``None`` -> ``linspace(-2, 12, 4000)``.
    wIn : numpy.ndarray or None
        RIXS incoming-energy mesh. ``None`` with ``deltaRIXS > 0`` -> ``linspace(-10, 20, 50)``;
        an empty mesh disables RIXS.
    deltaRIXS, deltaNIXS : float
        Excited-state-lifetime broadenings for RIXS / NIXS (HWHM, eV). ``deltaRIXS <= 0``
        disables RIXS.
    epsilonsRIXSin, epsilonsRIXSout : sequence or None
        RIXS in/out polarization vectors. ``None`` -> the Cartesian ``x, y, z`` triple.
    qsNIXS : sequence or None
        NIXS momentum-transfer vectors. ``None`` -> the driver's two default ``|q|``.
    liNIXS, ljNIXS : int
        Angular momenta of the final/initial orbitals in the NIXS excitation.
    radial : tuple or None
        ``(radial_mesh, Ri, Rj)`` for the NIXS radial integral. ``None`` skips NIXS.
    energy_cut : float
        How many ``k_B * T`` above the ground state to keep (thermal window).
    nPsiMax : int
        Maximum number of eigenstates to consider.
    auto_block_structure : bool
        Derive the block structure (and the symmetry-adapted solver basis) from the
        hybridization-dressed impurity matrix instead of the hand-coded one.
    XAS_projectors, RIXS_projectors : object or None
        Optional transition projectors; ``None`` computes the full polarization tensor.
    """

    w: Optional[np.ndarray] = None
    delta: float = 0.2
    epsilons: Any = None
    wLoss: Optional[np.ndarray] = None
    wIn: Optional[np.ndarray] = None
    deltaRIXS: float = 0.050
    deltaNIXS: float = 0.100
    epsilonsRIXSin: Any = None
    epsilonsRIXSout: Any = None
    qsNIXS: Any = None
    liNIXS: int = 2
    ljNIXS: int = 2
    radial: Any = None
    energy_cut: float = 10.0
    nPsiMax: int = 5
    auto_block_structure: bool = True
    XAS_projectors: Any = None
    RIXS_projectors: Any = None
