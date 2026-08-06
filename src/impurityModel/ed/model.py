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

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

from impurityModel.ed import atomic_physics, h0_format
from impurityModel.ed.hamiltonian_io import get_noninteracting_hamiltonian_operator
from impurityModel.ed.lie_algebra import tensors_to_operator
from impurityModel.ed.operator_algebra import addOps, c2i, matrixToIOp

__all__ = [
    "EXCITATION_BUDGET_DEFAULT",
    "BasisOptions",
    "ImpurityModel",
    "Meshes",
    "SolverOptions",
    "SpectraOptions",
    "atomic_u4",
    "load_selfenergy_archive",
    "resolve_excitation_budget",
]

# Default cap on the total number of bath excitations per determinant (see
# ``BasisOptions.excitation_budget``). 4 is the tightest measured-lossless value: on the
# metallic tier (FCC Ni) it shrinks the ground-state basis 1.43x at eigenvector rotation
# ~2e-4 (budget 3 is 3.45x but rotates the ground state past the 1e-3 accuracy bar), and on
# localized insulators the basis already carries fewer excitations so it is inert. A negative
# value (CLI) or ``None`` disables the budget entirely; see doc/plans/restrictions_redux.md.
EXCITATION_BUDGET_DEFAULT = 4


def resolve_excitation_budget(value: Optional[int]) -> Optional[int]:
    """Map a CLI-style excitation budget to ``BasisOptions.excitation_budget``.

    ``None`` (flag not given) resolves to ``EXCITATION_BUDGET_DEFAULT``; a negative value
    means "explicitly disabled" and resolves to ``None``.
    """
    if value is None:
        return EXCITATION_BUDGET_DEFAULT
    return value if value >= 0 else None


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
    dc : dict or None
        Double counting correction, in the RSPt convention, ``{process: amplitude}``.
        ``None`` for no DC correction.
    u4 : dict or None
        Coulomb tensor over the impurity spin-orbitals, in the RSPt convention (see
        :func:`atomic_u4`), ``{process: amplitude}``. ``None`` for a non-interacting model.
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
    dc: Optional[dict] = None
    u4: Optional[dict] = None
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
            u4=atomic_physics.getUop_from_rspt_u4(atomic_u4(l, slater)),
            impurity_orbitals={l: list(range(n_imp_spin_orbitals))},
            rot_to_spherical=np.eye(n_imp_spin_orbitals, dtype=complex),
        )

    @classmethod
    def from_h0_text(
        cls,
        path,
        *,
        l: Optional[int] = None,
        slater=None,
        n_impurity_orbitals: Optional[int] = None,
        xi: float = 0.0,
        h_field=(0.0, 0.0, 0.0),
        allow_noninteracting: bool = False,
    ) -> "ImpurityModel":
        """Build a model from a ``.h0`` file (or the legacy bare-integer form).

        Reads the flat single-index format described in ``doc/h0_file_format.md`` and
        delegates to :meth:`from_solver_matrix`. Deliberately **not** routed through
        :meth:`from_h0_file`: that path calls
        :func:`hamiltonian_io.get_noninteracting_hamiltonian_operator`, which ``addOps`` the
        ``(l, s, m)``-labelled spin-orbit and magnetic-field operators from
        :mod:`atomic_physics`. A flat-index ``h0`` cannot be added to those, and forcing it
        through ``c2i`` would reorder the orbitals silently.

        Parameters
        ----------
        path : str or pathlib.Path
            The ``.h0`` (or legacy ``.dict``/``.dat``) file.
        l : int, optional
            Angular momentum of the correlated shell. Taken from the header's ``interaction``
            block or ``impurity_l`` when not given.
        slater : sequence of float, optional
            Slater-Condon parameters for the atomic Coulomb tensor. Taken from the header's
            ``interaction`` block when not given.
        n_impurity_orbitals : int, optional
            Required for the legacy format, which carries no orbital layout. Validated
            against the sparsity pattern by the reader.
        xi : float, optional
            Spin-orbit coupling of the correlated shell. Built directly into the flat matrix
            (see Raises) when the header guarantees the basis and spin ordering it needs.
        h_field : tuple of float, optional
            Magnetic field ``(hx, hy, hz)`` on the correlated shell. Same gating as ``xi``.
        allow_noninteracting : bool, optional
            Permit ``u4=None``. Off by default, because a model with no Coulomb tensor runs a
            *non-interacting* solve to completion and prints a plausible-looking self-energy.

        Returns
        -------
        ImpurityModel

        Raises
        ------
        ValueError
            If no interaction is available and ``allow_noninteracting`` is False; if the
            declared shell disagrees with the impurity block size; if the file already contains
            spin-orbit coupling and a non-zero ``xi`` was requested, which would double-count
            it; or if a non-zero ``xi``/``h_field`` was requested but the header does not
            guarantee ``basis: "spherical"``, ``spin_ordering: "down_first"`` and a single
            correlated shell. ``xi``/``h_field`` are built in ``(l, s, m)`` labels via
            :func:`atomic_physics.getSOCop`/:func:`atomic_physics.gethHfieldop` and mapped
            through :func:`operator_algebra.c2i`, which only lines up with the flat matrix's
            own layout under those three guarantees -- refusing otherwise is deliberate, since
            guessing wrong flips the magnetization with no error anywhere. A missing
            ``spin_ordering`` (files written before the .h0 format tracked it) also refuses,
            rather than assume it.
        """
        parsed = (
            h0_format.read_h0_file(path)
            if h0_format.is_h0_format(path)
            else h0_format.read_legacy_flat_h0(path, _require_n_imp(path, n_impurity_orbitals))
        )

        if parsed.contains_soc and xi:
            raise ValueError(f"{path}: the file already contains spin-orbit coupling; xi={xi} would double-count it.")

        interaction = parsed.interaction or {}
        if l is None:
            l = interaction.get("l", parsed.header.get("impurity_l"))
        if slater is None:
            slater = interaction.get("F")

        n_imp = parsed.n_imp
        if l is not None and 2 * (2 * int(l) + 1) != n_imp:
            raise ValueError(
                f"{path}: the file's impurity block holds {n_imp} spin-orbitals, but l={l} implies "
                f"{2 * (2 * int(l) + 1)}. One of the two is wrong; refusing to guess."
            )

        dress = bool(xi) or any(h_field)
        if dress:
            if l is None:
                raise ValueError(
                    f"{path}: xi/h_field requested but the correlated shell's l could not be "
                    "determined (pass l=..., or the header needs 'impurity_l' or an 'interaction' "
                    "block); it is needed to build the spin-orbit/field operator."
                )
            if parsed.basis != "spherical":
                raise ValueError(
                    f"{path}: xi/h_field requested, but the header's basis is {parsed.basis!r}, not "
                    "'spherical'. Indexing a spherical-basis spin-orbit/field operator into a "
                    "non-spherical block would be wrong regardless of spin ordering; refusing."
                )
            if parsed.spin_ordering != "down_first":
                raise ValueError(
                    f"{path}: xi/h_field requested, but the header's spin_ordering is "
                    f"{parsed.spin_ordering!r}, not 'down_first'. c2i (and getSOCop/gethHfieldop) "
                    "assume spin down first; a file that does not declare this convention cannot "
                    "be dressed without risking a silently flipped magnetization."
                )
            if len(parsed.impurity_orbitals) != 1 or parsed.header.get("shell_layout") == "multi":
                raise ValueError(
                    f"{path}: xi/h_field requested, but the file declares more than one impurity "
                    "shell/group. c2i interleaves multiple shells, so a single-shell (l, s, m) "
                    "operator cannot be mapped into the flat matrix unambiguously; refusing."
                )

        if slater is None:
            if not allow_noninteracting:
                raise ValueError(
                    f"{path}: no Coulomb interaction available -- the header carries no 'interaction' "
                    "block and no slater parameters were passed. A model with u4=None runs a "
                    "*non-interacting* solve to completion and reports a plausible-looking Sigma, so "
                    "this is refused rather than defaulted. Pass slater=... (and l=...), or "
                    "allow_noninteracting=True if that is genuinely what you want."
                )
            u4 = None
        else:
            u4 = atomic_u4(int(l), slater)

        bath_valence_conduction = None
        if parsed.valence_bath is not None and parsed.conduction_bath is not None:
            bath_valence_conduction = (parsed.valence_bath, parsed.conduction_bath)

        rot = parsed.rot_to_spherical
        if rot is None:
            rot = np.eye(n_imp, dtype=complex)

        h = parsed.to_matrix()
        if dress:
            _add_soc_and_field(h, int(l), xi, h_field)

        return cls.from_solver_matrix(
            h,
            n_imp,
            u4=u4,
            rot_to_spherical=rot,
            bath_valence_conduction=bath_valence_conduction,
        )

    @classmethod
    def from_hdf5(cls, path, cluster: Optional[str] = None, iteration: Optional[int] = None) -> "ImpurityModel":
        """Build a model from a group of an ``impurityModel_data.h5`` archive.

        The RSPt interface (:mod:`impurityModel_interface`) writes one group per
        ``(cluster, DMFT iteration)`` holding the full one-particle solver Hamiltonian
        (``H solver``), the double counting correction (``DC``), the Coulomb tensor (``U``),
        ``Rot to spherical`` and the impurity orbital indices, which is exactly the physics of
        an :class:`ImpurityModel`. This makes an archived DFT-embedded run re-runnable offline.
        Use :func:`load_selfenergy_archive`
        to also recover the frequency meshes and the recorded solver/basis options.

        Parameters
        ----------
        path : str or path-like
            Path to the archive.
        cluster : str, optional
            Cluster label (e.g. ``"Ni"``); defaults to the label of the first group.
        iteration : int, optional
            DMFT iteration; defaults to the archive's ``last iteration`` attribute.

        Returns
        -------
        ImpurityModel
        """
        raw = _read_archive_group(path, cluster, iteration)
        return cls(
            h0=raw["h0"],
            u4=raw["u4"],
            dc=raw["dc"],
            impurity_orbitals=raw["impurity_orbitals"],
            rot_to_spherical=raw["rot_to_spherical"],
            n_spin_orbitals=raw["n_spin_orbitals"],
        )

    @classmethod
    def from_solver_matrix(
        cls,
        H,
        n_impurity_orbitals: int,
        *,
        dc: Optional[np.ndarray] = None,
        u4: Optional[np.ndarray] = None,
        rot_to_spherical,
        bath_valence_conduction=None,
    ) -> "ImpurityModel":
        """Build a model from the full one-particle solver matrix ``H`` (impurity block first).

        ``H`` is the dense ``(n, n)`` single-particle Hamiltonian with the impurity orbitals in
        the first ``n_impurity_orbitals`` rows/columns and the bath orbitals after -- the layout
        :func:`rspt2spectra.h0.assemble_h0` produces and the ``impurityModel_data.h5`` archive
        stores as ``H solver``. The impurity orbital layout is derived from
        ``n_impurity_orbitals``; the bath orbitals (and their valence/conduction split, when the
        solver needs it) are everything after.

        Parameters
        ----------
        H : numpy.ndarray
            Full ``(n, n)`` one-particle Hamiltonian, impurity block first.
        n_impurity_orbitals : int
            Number of impurity spin-orbitals (the leading block dimension).
        u4 : numpy.ndarray, optional
            Impurity Coulomb tensor (``n_imp`` per axis), in the same basis as ``H``.
        rot_to_spherical : numpy.ndarray or dict
            Rotation from the impurity input basis to spherical harmonics (observable reporting).
        bath_valence_conduction : tuple of sequence, optional
            ``(valence_indices, conduction_indices)`` classifying the bath spin-orbitals; when
            given it populates :attr:`bath_states`. The self-energy path re-derives its own split
            from ``h0`` and ignores this; the double-counting and spectra paths require it.

        Returns
        -------
        ImpurityModel
        """
        H = np.asarray(H)
        n_imp = int(n_impurity_orbitals)
        bath_states = None
        if bath_valence_conduction is not None:
            valence, conduction = bath_valence_conduction
            bath_states = ({0: [int(i) for i in valence]}, {0: [int(i) for i in conduction]})
        return cls(
            h0=_operator_from_matrix(H),
            dc=tensors_to_operator(np.asarray(dc, dtype=complex)).to_dict() if dc is not None else None,
            u4=atomic_physics.getUop_from_rspt_u4(u4) if u4 is not None else None,
            impurity_orbitals={0: list(range(n_imp))},
            rot_to_spherical=rot_to_spherical,
            bath_states=bath_states,
            n_spin_orbitals=H.shape[0],
        )

    @classmethod
    def from_blocks(
        cls,
        H_imp,
        V,
        H_bath,
        *,
        dc: Optional[np.ndarray] = None,
        u4: Optional[np.ndarray] = None,
        rot_to_spherical,
        bath_valence_conduction=None,
    ) -> "ImpurityModel":
        """Build a model from the impurity / hybridization / bath blocks.

        Stacks the block form ``H = [[H_imp, V^dagger], [V, H_bath]]`` (impurity block first,
        matching :func:`rspt2spectra.h0.assemble_h0`) and delegates to
        :meth:`from_solver_matrix`. ``H_imp`` must be the *effective* impurity block, i.e. the
        DFT block with the fitted constant hybridization offset already folded in
        (``E_imp = H_imp + C``); ``assemble_h0`` returns exactly that.

        Parameters
        ----------
        H_imp : numpy.ndarray
            Effective impurity block, ``(n_imp, n_imp)``.
        V : numpy.ndarray
            Impurity-bath hopping, ``(n_bath, n_imp)`` (the bath-row / impurity-column block, as
            ``assemble_h0`` returns ``v_solver``).
        H_bath : numpy.ndarray
            Bath block, ``(n_bath, n_bath)``.
        dc, u4, rot_to_spherical, bath_valence_conduction
            As in :meth:`from_solver_matrix`.

        Returns
        -------
        ImpurityModel
        """
        H_imp = np.asarray(H_imp)
        V = np.asarray(V)
        H_bath = np.asarray(H_bath)
        n_imp = H_imp.shape[0]
        n_bath = H_bath.shape[0]
        H = np.zeros((n_imp + n_bath, n_imp + n_bath), dtype=complex)
        H[:n_imp, :n_imp] = H_imp
        H[n_imp:, n_imp:] = H_bath
        H[n_imp:, :n_imp] = V
        H[:n_imp, n_imp:] = V.conj().T
        return cls.from_solver_matrix(
            H,
            n_imp,
            dc=dc,
            u4=u4,
            rot_to_spherical=rot_to_spherical,
            bath_valence_conduction=bath_valence_conduction,
        )


#: Extensions handled by the labelled (l, s, m) readers in :mod:`hamiltonian_io`.
_LABELLED_EXTENSIONS = {".pickle", ".json"}
#: Extensions handled by the flat single-index reader in :mod:`h0_format`.
_FLAT_EXTENSIONS = {".h0", ".dict"}


def load_model(
    h0_filename,
    *,
    l=None,
    n_baths=None,
    slater=None,
    xi: float = 0.0,
    h_field=None,
    n_val_baths=None,
    n_impurity_orbitals=None,
    allow_noninteracting: bool = False,
    rank: int = 0,
    verbose: bool = False,
) -> "ImpurityModel":
    """Build an :class:`ImpurityModel` from any supported h0 file.

    The single place that answers "which h0 formats does impurityModel read". Both the
    self-energy and susceptibility CLIs route through it, so a new format is added once
    rather than in each of them.

    Dispatch is on *content* first and extension second, because the extensions are
    overloaded: a ``.dat`` may hold either the labelled ``(l, s, m)`` operator list that
    :mod:`op_parser` reads or the legacy flat-index terms, since renaming a ``.dict`` to
    ``.dat`` was the standard way around the extension check that used to reject it.

    Parameters
    ----------
    h0_filename : str or pathlib.Path
        The Hamiltonian file.
    l : int, optional
        Angular momentum of the correlated shell. Required by the labelled readers; the
        flat format can take it from its header.
    n_baths, n_val_baths : int, optional
        Bath counts, for the labelled readers only (the flat format carries its own layout).
    slater : sequence of float, optional
        Slater-Condon parameters for the atomic Coulomb tensor.
    xi : float, optional
        Spin-orbit coupling of the correlated shell.
    h_field : tuple of float, optional
        Magnetic field. ``None`` (the default) means "whatever this format's usual default is":
        the labelled readers get their usual ``(0, 0, 0.0001)`` symmetry-breaking nudge, the
        flat-index reader gets no field. An *explicit* non-zero field on a flat file is applied
        when the header guarantees ``basis: "spherical"`` and ``spin_ordering: "down_first"``
        (see :meth:`ImpurityModel.from_h0_text`), and raises otherwise rather than being
        silently dropped or guessed at.
    n_impurity_orbitals : int, optional
        Impurity block size, required only by the legacy flat format.
    allow_noninteracting : bool, optional
        Permit a model with no Coulomb tensor; see :meth:`ImpurityModel.from_h0_text`.
    rank, verbose
        Forwarded to the labelled reader for rank-0 logging.

    Returns
    -------
    ImpurityModel

    Raises
    ------
    ValueError
        For an unrecognised format, naming the ones that are supported.
    """
    path = str(h0_filename)
    ext = os.path.splitext(path)[1].lower()

    flat = ext in _FLAT_EXTENSIONS or h0_format.is_h0_format(path)
    if not flat and ext == ".dat":
        # A .dat that is really the flat form -- the renamed-.dict case.
        flat = h0_format.looks_like_flat_terms(path)

    if flat:
        return ImpurityModel.from_h0_text(
            path,
            l=l,
            slater=slater,
            n_impurity_orbitals=n_impurity_orbitals,
            xi=xi,
            h_field=(0.0, 0.0, 0.0) if h_field is None else tuple(h_field),
            allow_noninteracting=allow_noninteracting,
        )

    if ext in _LABELLED_EXTENSIONS or ext == ".dat":
        if l is None or n_baths is None or slater is None:
            raise ValueError(
                f"{path}: the labelled {ext} format carries no orbital layout or interaction; "
                "l, n_baths and slater are all required."
            )
        return ImpurityModel.from_h0_file(
            path,
            l=l,
            n_baths=n_baths,
            slater=slater,
            xi=xi,
            h_field=(0.0, 0.0, 0.0001) if h_field is None else tuple(h_field),
            n_val_baths=n_val_baths,
            rank=rank,
            verbose=verbose,
        )

    raise ValueError(
        f"{path}: unrecognised h0 format {ext!r}. Supported: .h0 (self-describing flat index, "
        "read by the selfenergy and susceptibility sub-commands), .dict/.dat (legacy flat index), "
        ".pickle/.json/.dat (labelled (l, s, m), the only formats the spectra sub-command takes). "
        "See doc/h0_file_format.md."
    )


def _require_n_imp(path, n_impurity_orbitals):
    """The legacy format carries no orbital layout, so the caller must supply the boundary."""
    if n_impurity_orbitals is None:
        raise ValueError(
            f"{path} is in the legacy bare-integer h0 format, which records no impurity/bath "
            "boundary. Pass n_impurity_orbitals=..., or regenerate the file with build_h0 to get "
            "a self-describing .h0 (see doc/h0_file_format.md)."
        )
    return int(n_impurity_orbitals)


def _add_soc_and_field(h, l, xi, h_field):
    """Add spin-orbit coupling and a magnetic field for one correlated shell into ``h``, in place.

    Only valid when ``h``'s impurity block is laid out exactly as :func:`operator_algebra.c2i`
    would place a single ``l``-shell (spin-major, ``s=0`` first, ``m`` ascending within each spin
    block) -- the caller (:meth:`ImpurityModel.from_h0_text`) is responsible for checking the
    file's header actually guarantees that before calling this.

    Parameters
    ----------
    h : numpy.ndarray
        Dense one-particle Hamiltonian, modified in place. Only entries within the ``l``-shell's
        ``2 * (2 * l + 1)`` leading rows/columns are touched.
    l : int
        Angular momentum of the correlated shell.
    xi : float
        Spin-orbit coupling constant.
    h_field : tuple of float
        Magnetic field ``(hx, hy, hz)``.
    """
    ops = addOps([atomic_physics.getSOCop(xi, l), atomic_physics.gethHfieldop(*h_field, l=l)])
    n_baths = OrderedDict({l: 0})
    for (op_c, op_a), value in ops.items():
        i = c2i(n_baths, op_c[0])
        j = c2i(n_baths, op_a[0])
        h[i, j] += value


def _operator_from_matrix(h_matrix) -> dict:
    """Single-index operator dict of a dense one-particle Hamiltonian (the one matrix->op site).

    Wraps :func:`operator_algebra.matrixToIOp`; shared by :meth:`ImpurityModel.from_solver_matrix`
    and the archive reader so the matrix -> ``{((i, "c"), (j, "a")): value}`` conversion lives in
    exactly one place.
    """
    return matrixToIOp(np.asarray(h_matrix))


def _archive_attr(attrs, key, default=None):
    """Group attribute with the interface's ``None``-stored-as-the-string-``"None"`` undone."""
    value = attrs.get(key, default)
    if isinstance(value, (str, bytes)) and str(value) == "None":
        return None
    return value


def _read_archive_group(path, cluster=None, iteration=None) -> dict:
    """Parse one ``(cluster, iteration)`` group of an ``impurityModel_data.h5`` archive.

    Returns the raw physics arrays (``h0`` operator dict, ``u4``, ``impurity_orbitals``,
    ``rot_to_spherical``, ``n_spin_orbitals``), both frequency meshes and every recorded
    solver/basis option, plus the group ``label``. Shared by :meth:`ImpurityModel.from_hdf5`
    and :func:`load_selfenergy_archive`.
    """
    import h5py

    with h5py.File(path, "r") as f:
        if iteration is None:
            iteration = int(f.attrs.get("last iteration", 1))
        if cluster is None:
            labels = sorted(f.keys())
            if not labels:
                raise ValueError(f"{path}: archive holds no cluster groups")
            cluster = labels[0].rsplit(" ", 1)[0]
        name = f"{cluster} {iteration}"
        if name not in f:
            raise ValueError(f"{path}: no group {name!r}; available: {sorted(f.keys())}")
        g = f[name]
        attrs = dict(g.attrs)

        h_solver = np.asarray(g["H solver"])
        _u4 = np.asarray(g["U"])
        _dc = np.asarray(g["DC"]) if "DC" in g else None
        iw_mesh = np.asarray(g["Matsubara frequency mesh"])
        w_mesh = np.asarray(g["Real frequency mesh"])
        rot_to_spherical = np.asarray(g["Rot to spherical"])
        impurity_indices = [int(i) for i in np.asarray(g["Impurity orbitals"])]

    # The interface hands calc_selfenergy the Hamiltonian as a second-quantized operator.
    h0 = _operator_from_matrix(h_solver)
    u4 = atomic_physics.getUop_from_rspt_u4(_u4)
    dc = tensors_to_operator(_dc).to_dict() if _dc is not None else None

    truncation_threshold = _archive_attr(attrs, "truncation_threshold")
    if truncation_threshold is not None:
        truncation_threshold = float(truncation_threshold)
    dN = _archive_attr(attrs, "dN")
    if dN is not None:
        dN = int(dN)
    # Missing attribute (older archives) falls back to the default budget; a stored
    # negative value means the producing run explicitly disabled it.
    excitation_budget = _archive_attr(attrs, "excitation_budget", EXCITATION_BUDGET_DEFAULT)
    if excitation_budget is not None:
        excitation_budget = int(excitation_budget) if int(excitation_budget) >= 0 else None

    return {
        "label": name,
        "h0": h0,
        "u4": u4,
        "dc": dc,
        "n_spin_orbitals": int(h_solver.shape[0]),
        "impurity_orbitals": {0: impurity_indices},
        "rot_to_spherical": rot_to_spherical,
        # Meshes are stored real-valued; the Matsubara mesh is z = i * nu (see Meshes.iw).
        "iw_mesh": iw_mesh if iw_mesh.size else None,
        "w_mesh": w_mesh if w_mesh.size else None,
        "nominal_occ": {0: int(attrs["nominal occupation"])},
        "mixed_valence": _archive_attr(attrs, "mv"),
        "tau": float(attrs["tau"]),
        "delta": float(attrs["delta"]),
        "reort": _archive_attr(attrs, "reort"),
        "dense_cutoff": int(_archive_attr(attrs, "dense_cutoff", 1000)),
        "spin_flip_dj": bool(_archive_attr(attrs, "spin_flip_dj", False)),
        "chain_restrict": bool(_archive_attr(attrs, "chain_restrict", False)),
        "occ_cutoff": float(_archive_attr(attrs, "occ_cutoff", 1e-6)),
        "truncation_threshold": truncation_threshold,
        "slater_weight_min": float(_archive_attr(attrs, "slater_min", 0.0)),
        "dN": dN,
        "excitation_budget": excitation_budget,
        "sparse_green": bool(_archive_attr(attrs, "sparse_green", True)),
    }


def load_selfenergy_archive(path, cluster=None, iteration=None):
    """Reconstruct a full ``calc_selfenergy`` call from an ``impurityModel_data.h5`` archive.

    Returns ``(model, meshes, basis, solver, cluster_label)`` -- the :class:`ImpurityModel`
    together with the frequency meshes and the recorded basis/solver options, so an archived
    DFT-embedded self-energy run can be reproduced offline (see :meth:`ImpurityModel.from_hdf5`
    for the physics-only view).

    Parameters
    ----------
    path : str or path-like
        Path to the archive.
    cluster : str, optional
        Cluster label; defaults to the first group's label.
    iteration : int, optional
        DMFT iteration; defaults to the archive's ``last iteration`` attribute.

    Returns
    -------
    (ImpurityModel, Meshes, BasisOptions, SolverOptions, str)
    """
    raw = _read_archive_group(path, cluster, iteration)
    model = ImpurityModel(
        h0=raw["h0"],
        dc=raw["dc"],
        u4=raw["u4"],
        impurity_orbitals=raw["impurity_orbitals"],
        rot_to_spherical=raw["rot_to_spherical"],
        n_spin_orbitals=raw["n_spin_orbitals"],
    )
    meshes = Meshes(
        iw=1j * raw["iw_mesh"] if raw["iw_mesh"] is not None else None,
        w=raw["w_mesh"],
        delta=raw["delta"],
    )
    basis = BasisOptions(
        nominal_occ=raw["nominal_occ"],
        mixed_valence=raw["mixed_valence"],
        dN=raw["dN"],
        truncation_threshold=raw["truncation_threshold"],
        chain_restrict=raw["chain_restrict"],
        spin_flip_dj=raw["spin_flip_dj"],
        occ_cutoff=raw["occ_cutoff"],
        slater_weight_min=raw["slater_weight_min"],
        tau=raw["tau"],
        excitation_budget=raw["excitation_budget"],
    )
    solver = SolverOptions(
        reort=raw["reort"],
        dense_cutoff=raw["dense_cutoff"],
        sparse_green=raw["sparse_green"],
    )
    return model, meshes, basis, solver, raw["label"]


def _count_spin_orbitals(h0) -> int:
    """Highest spin-orbital index in ``h0`` plus one (0 for an empty operator)."""
    items = h0.items() if hasattr(h0, "items") else h0.to_dict().items()
    highest = -1
    for process, _value in items:
        for index, _action in process:
            highest = max(highest, index)
    return highest + 1


@dataclass(frozen=True)
class Meshes:
    """Frequency meshes and the real-axis broadening.

    Attributes
    ----------
    iw : numpy.ndarray or None
        Complex Matsubara mesh :math:`z = i\\nu_n` (the solver evaluates the resolvent at
        ``iw`` directly, so pass ``1j * nu``). ``None`` skips the Matsubara output.
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
    excitation_budget : int or None
        Cap on the total number of bath excitations (holes in filled-valence + electrons in
        empty-conduction orbitals) a determinant may carry, enforced as a weighted
        restriction on the ground-state basis and inherited (widened) by the
        Green's-function / spectra excited bases. Defaults to ``EXCITATION_BUDGET_DEFAULT``
        (4, the tightest measured-lossless value); ``None`` disables it. A memory lever on
        metals; judge its accuracy on the eigenvector/spectral criterion, not ``E0`` (see
        ``doc/plans/restrictions_redux.md``).
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
    excitation_budget: Optional[int] = EXCITATION_BUDGET_DEFAULT


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
    gf_method : {"lanczos", "bicgstab", "sliced", "cipsi"}
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
