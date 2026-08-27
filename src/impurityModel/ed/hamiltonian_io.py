"""
Construction and file I/O of the (non-interacting and interacting) impurity
Hamiltonian: readers for pickled/.dat/.json h0 formats and the builders that
combine h0 with SOC, magnetic field, Coulomb, and double counting.
"""

import json
import os
from collections.abc import Mapping
import pickle
from collections import OrderedDict
from typing import NamedTuple

import numpy as np

from impurityModel.ed import atomic_physics, h0_format, op_parser, symmetries
from impurityModel.ed.operator_algebra import addOps, assert_hermitian, c2i, i2c


def get_noninteracting_hamiltonian_operator(
    nBaths,
    nValBaths,
    h0_filename,
    rank,
    verbose=True,
    *,
    valence_l,
    xi_valence,
    hField=(0.0, 0.0, 0.0),
    core_l=None,
    xi_core=0.0,
):
    """
    Build the non-interacting Hamiltonian operator.

    Combines spin-orbit coupling, magnetic field, and the non-interacting
    Hamiltonian read from a file.

    The shells are named explicitly rather than inferred from ``nBaths``' key order: which
    shell is the core one decides where the spin-orbit coupling and the Zeeman field land,
    and no ordering convention on a dict should be able to change that.

    Parameters
    ----------
    nBaths : dict
        Number of bath orbitals, keyed by angular momentum.
    nValBaths : dict
        Number of valence bath orbitals, keyed by angular momentum.
    h0_filename : str
        Filename of the non-interacting Hamiltonian.
    rank : int
        MPI rank.
    verbose : bool, optional
        Whether to print output on rank 0. Default is True.
    valence_l : int
        Angular momentum of the valence (correlated) shell. Carries the Zeeman field.
    xi_valence : float
        Spin-orbit coupling constant of the valence shell.
    hField : tuple[float, float, float], optional
        Magnetic field components (hx, hy, hz), applied to the valence shell.
    core_l : int or None, optional
        Angular momentum of the core shell, or ``None`` when the model has no core shell.
    xi_core : float, optional
        Spin-orbit coupling constant of the core shell. Ignored when ``core_l`` is ``None``.

    Returns
    -------
    hOperator : dict
        The total non-interacting Hamiltonian operator.
    """
    hx, hy, hz = hField

    # Magnetic field, on the valence shell only.
    operators = [atomic_physics.gethHfieldop(hx, hy, hz, l=valence_l)]

    # Add SOC, in spherical harmonics basis. Core shell before valence: `addOps` builds its
    # result by insertion, and the spectra matvec accumulates terms in that order, so the
    # operand order is observable in the last bits of every Green's function. Keeping the
    # historical order (field, core SOC, valence SOC, h0) keeps those bit-reproducible.
    if core_l is not None:
        operators.append(atomic_physics.getSOCop(xi_core, l=core_l))
    operators.append(atomic_physics.getSOCop(xi_valence, l=valence_l))

    # Read the non-relativistic non-interacting Hamiltonian operator from file.
    h0_operator = read_h0_operator(h0_filename, nBaths, nValBaths, rank=rank, verbose=verbose, valence_l=valence_l)

    if rank == 0 and verbose:
        print(f"Non-interacting, non-relativistic Hamiltonian (h0): {len(h0_operator)} terms.")
    operators.append(h0_operator)
    return addOps(operators)


def read_h0_operator(filename, nBaths, nValBaths=None, rank=0, verbose=True, *, valence_l=None):
    """
    Read the non-interacting Hamiltonian from a pickled (.pickle) or text (.dat) file.

    Parameters
    ----------
    filename : str or Mapping
        The path to the file, or a mapping of crystal-field parameters.
    nBaths : dict
        Number of bath orbitals.
    nValBaths : dict
        Number of valence bath orbitals (needed for .json CF).
    rank, verbose : int, bool, optional
        Forwarded to :func:`flat_h0_to_labelled` for the Kramers-degeneracy warning.
    valence_l : int, optional
        Angular momentum of the shell a crystal-field parametrisation describes. Named rather
        than inferred: "the shell that has a bath" is wrong for a bath-less valence shell,
        which is the Hubbard-I model. Only the crystal-field path reads it.

    Returns
    -------
    dict
        The non-interacting Hamiltonian operator dictionary.
    """

    # Crystal-field parameters supplied directly rather than through a .json file. The TOML
    # input format takes this path so it can require every parameter of that shell explicitly
    # (see `cf_parameter_names`): the file reader below fills an absent d-shell key from a
    # hard-coded Ni-in-NiO value, which silently gives another material Ni's conduction bath.
    def crystal_field(source):
        """Both crystal-field entry points -- a mapping and a .json file -- need the shell."""
        if valence_l is None:
            raise ValueError(
                f"{source if isinstance(source, str) else 'Crystal-field parameters'} describes "
                "a crystal field, but no valence_l was given. The shell a crystal-field "
                "parametrisation applies to is named, not inferred: 'the shell that has a "
                "bath' cannot name a bath-less valence shell (the Hubbard-I model)."
            )
        return get_CF_hamiltonian(nBaths, nValBaths, source, l=valence_l)

    if isinstance(filename, Mapping):
        return crystal_field(filename)

    _, ext = os.path.splitext(filename)
    if ext.lower() == ".pickle":
        return read_pickled_file(filename)
    if ext.lower() == ".dat":
        return read_h0_dict(filename)
    if ext.lower() == ".json":
        return crystal_field(filename)
    if ext.lower() in (".h0", ".dict"):
        if not h0_format.is_h0_format(filename):
            # A legacy headerless flat file: no basis, no spin ordering, nothing to
            # justify the relabelling below.
            raise RuntimeError(
                f"{filename}: this is the legacy headerless flat h0 format, which records no "
                "basis or spin ordering. The labelled readers cannot interpret it without those "
                "guarantees; regenerate it with build_h0 to get a self-describing .h0. "
                "See doc/h0_file_format.md."
            )
        return flat_h0_to_labelled(h0_format.read_h0_file(filename), nBaths, path=filename, rank=rank, verbose=verbose)
    raise RuntimeError(f"Unknown file h0 file extension {ext}")


def flat_h0_to_labelled(parsed, nBaths, path=None, rank=0, verbose=True):
    """Relabel a single-shell, flat-indexed ``.h0`` file into ``(l, s, m)`` / ``(l, b)`` labels.

    The flat format's impurity-block-first layout is, for a *single* correlated shell, the
    exact inverse of :func:`operator_algebra.i2c` applied with a one-shell ``nBaths`` dict --
    provided the header guarantees ``basis: "spherical"`` and ``spin_ordering: "down_first"``,
    which is what makes ``m`` and spin land where ``i2c`` expects them. That is the only
    permutation this function performs; the caller's ``nBaths`` (potentially multi-shell) is
    used only to identify which shell the file belongs to and validate its bath count.

    Parameters
    ----------
    parsed : h0_format.H0File
        Already-parsed flat file.
    nBaths : dict
        Number of bath orbitals per shell, keyed by angular momentum -- the multi-shell layout
        the caller (:func:`get_hamiltonian_operator`) assembles into.
    path : str, optional
        File path to name in error messages. Falls back to ``parsed.header.get('producer')``
        (typically ``None``, since ``build_h0`` does not set that key) when omitted.
    rank, verbose : int, bool, optional
        Forwarded from the caller so the Kramers-degeneracy warning below prints only once,
        on rank 0 -- reading a file is not an MPI collective, so this gates a print, not
        control flow (unlike the collective-gating rule in CLAUDE.md).

    Returns
    -------
    dict
        ``{((l, s, m) | (l, b), 'c'), (..., 'a')): amplitude}``.

    Raises
    ------
    RuntimeError
        If the header does not guarantee spherical basis, down-first spin ordering and a
        Fermi-referenced energy zero; if the file declares more than one impurity shell; if
        its shell's ``l`` is not one of ``nBaths``; or if its bath count disagrees with
        ``nBaths[l]``.
    """
    if parsed.basis != "spherical":
        raise RuntimeError(
            f"{path or parsed.header.get('producer')}: header basis is {parsed.basis!r}, not "
            "'spherical'; relabelling into (l, s, m) assumes a spherical-harmonics m-ordering."
        )
    if parsed.spin_ordering != "down_first":
        raise RuntimeError(
            f"{path or parsed.header.get('producer')}: header spin_ordering is {parsed.spin_ordering!r}, "
            "not 'down_first'; a file that does not declare this convention cannot be relabelled "
            "without risking a silently flipped spin."
        )
    if parsed.energy_reference != "fermi":
        raise RuntimeError(
            f"{path or parsed.header.get('producer')}: header energy_reference is "
            f"{parsed.energy_reference!r}, not 'fermi'; an absolute-referenced block would shift "
            "the whole shell relative to what get_hamiltonian_operator assumes."
        )
    if len(parsed.impurity_orbitals) != 1 or parsed.header.get("shell_layout") == "multi":
        raise RuntimeError(
            f"{path or parsed.header.get('producer')}: file declares more than one impurity shell/group; "
            "c2i interleaves multiple shells, so a single-shell relabelling is not unambiguous."
        )

    l = parsed.header.get("impurity_l")
    if l is None or int(l) not in nBaths:
        raise RuntimeError(
            f"{path or parsed.header.get('producer')}: header impurity_l={l!r} is not one of the "
            f"requested shells {sorted(nBaths)}."
        )
    l = int(l)
    if 2 * (2 * l + 1) != parsed.n_imp:
        raise RuntimeError(
            f"{path or parsed.header.get('producer')}: impurity_l={l} implies {2 * (2 * l + 1)} "
            f"impurity spin-orbitals, but the file's impurity block holds {parsed.n_imp}."
        )
    n_bath_in_file = parsed.n_orb - parsed.n_imp
    if n_bath_in_file != nBaths[l]:
        raise RuntimeError(
            f"{path or parsed.header.get('producer')}: file has {n_bath_in_file} bath orbitals for l={l}, "
            f"but nBaths[{l}] = {nBaths[l]} was requested."
        )

    # SOC is time-reversal *even*, so a SOC-free Hamiltonian's exact Kramers degeneracy
    # surviving a star-bath fit is structural (the up/down blocks are literally identical and
    # share one fit); with SOC present nothing currently enforces the pairing between the
    # time-reversal-conjugate blocks the fit treats independently, so losing exact degeneracy
    # is expected there, not a defect (see build_h0.py's Kramers check for the full reasoning).
    # Only warn when the header *positively* declares no SOC (`contains_soc is False`): every
    # file written before this check existed has no `contains_soc` key at all (None), and
    # warning on that unknown case would cry wolf on nearly every real transition-metal
    # workload already on disk.
    if rank == 0 and parsed.contains_soc is False:
        violations = symmetries.check_kramers_degeneracy(parsed.to_matrix())
        if violations:
            print(
                f"WARNING: {path or parsed.header.get('producer')}: {len(violations)} "
                f"odd-multiplicity eigenvalue cluster(s) -- this h0 breaks time-reversal "
                f"(Kramers) symmetry. The header declares no spin-orbit coupling, so this is not "
                "the expected SOC-driven loss of exact degeneracy -- it usually means the cluster "
                "is genuinely spin-polarised or field-dressed, or that the bath fit that produced "
                "the file has a problem (see doc/h0_file_format.md, Basis and spin ordering)."
            )
            if verbose:
                detail = ", ".join(f"E={v['energy']:.4g} (x{v['multiplicity']})" for v in violations)
                print(f"  clusters: {detail}")

    shell = OrderedDict({l: n_bath_in_file})
    operator = {}
    for ((i, _), (j, _)), value in parsed.h0.items():
        operator[((i2c(shell, i), "c"), (i2c(shell, j), "a"))] = value
    return operator


def get_hamiltonian_operator(
    nBaths,
    nValBaths,
    slaterCondon,
    DCinfo,
    h0_filename,
    rank,
    verbose=True,
    *,
    valence_l,
    xi_valence,
    hField=(0.0, 0.0, 0.0),
    core_l=None,
    xi_core=0.0,
):
    """
    Return the Hamiltonian, in operator form.

    Parameters
    ----------
    nBaths : dict
        Number of bath states for each angular momentum.
    nValBaths : dict
        Number of valence bath states for each angular momentum.
    slaterCondon : sequence
        ``(Fvv, Fcc, Fcv, Gcv)`` Slater-Condon parameters. The three core-valence arrays are
        ignored (and may be ``None``) when ``core_l`` is ``None``.
    DCinfo : sequence
        ``(n0imps, chargeTransferCorrection)``: nominal occupation per angular momentum, and
        the many-body correction to the charge-transfer energy.
    h0_filename : str
        Filename of non-interacting, non-relativistic operator.
    rank : int
        MPI rank.
    verbose : bool, optional
        Whether to print output on rank 0.
    valence_l, xi_valence, hField, core_l, xi_core
        The shell layout; see :func:`get_noninteracting_hamiltonian_operator`.

    Returns
    -------
    hOp : dict
        The Hamiltonian in operator form.
        tuple : complex,
        where each tuple describes a process of several steps.
        Each step is described by a tuple of the form: (i,'c') or (i,'a'),
        where i is a spin-orbital index.

    """
    # Divide up input parameters to more concrete variables
    Fvv, Fcc, Fcv, Gcv = slaterCondon
    n0imps, chargeTransferCorrection = DCinfo
    if core_l is None:
        Fcc = Fcv = Gcv = None

    h_non_interacting = get_noninteracting_hamiltonian_operator(
        nBaths,
        nValBaths,
        h0_filename,
        rank,
        verbose,
        valence_l=valence_l,
        xi_valence=xi_valence,
        hField=hField,
        core_l=core_l,
        xi_core=xi_core,
    )
    # Calculate the U operator, in spherical harmonics basis.
    uOperator = atomic_physics.slater_condon_Uop(valence_l, core_l, Fvv, Fcc=Fcc, Fcv=Fcv, Gcv=Gcv)
    dc = atomic_physics.dc_MLFT(
        valence_l,
        n0imps[valence_l],
        chargeTransferCorrection,
        Fvv,
        lc=core_l,
        n_core_i=None if core_l is None else n0imps[core_l],
        Fcv=Fcv,
        Gcv=Gcv,
    )
    eDCOperator = {}
    for l in dc:
        for s in range(2):
            for m in range(-l, l + 1):
                eDCOperator[(((l, s, m), "c"), ((l, s, m), "a"))] = -dc[l]

    # Add Hamiltonian terms to one operator.
    hOperator = addOps([uOperator, eDCOperator, h_non_interacting])

    # Convert spin-orbital and bath state indices to a single index notation.
    hOp = {}
    for process, value in hOperator.items():
        hOp[tuple((c2i(nBaths, spinOrb), action) for spinOrb, action in process)] = value

    assert_hermitian(hOp)
    return hOp


def read_pickled_file(filename: str):
    """
    Load content from a pickled file.

    Parameters
    ----------
    filename : str
        The path to the pickle file.

    Returns
    -------
    any
        The deserialized Python object.
    """
    with open(filename, "rb") as handle:
        content = pickle.load(handle)
    return content


def read_h0_dict(h0_filename):
    r"""
    Reads the non-interacting Hamiltoninan from file.
    Parameters
    ----------
        h0_filename : String
        File containing the non-interacting Hamiltonian.
    """
    h0_dict = {}
    for _, op in op_parser.parse_file(h0_filename).items():
        for key, val in op.items():
            if key in h0_dict:
                h0_dict[key] += val
            else:
                h0_dict[key] = val
    return h0_dict


class CFBathBlocks(NamedTuple):
    """Which of the two crystal-field bath blocks a model has.

    A named pair rather than two bare booleans: ``cf_parameter_names(2, False, False)`` said
    nothing about which flag was which, and the call sites unpacked it as ``*blocks``.
    """

    valence: bool
    conduction: bool


def cf_shell(nBaths, l):
    """Validate the angular momentum a crystal-field parametrisation describes.

    The shell is **named**, never inferred. "The shell that has a bath" cannot name a
    bath-less valence shell -- the Hubbard-I approximation -- and inferring a role from the
    model's shape is the assumption this format exists to outlive.

    Parameters
    ----------
    nBaths : dict
        Number of bath states for each angular momentum.
    l : int
        The shell being parametrised.

    Raises
    ------
    ValueError
        If ``l`` is not among the shells, or has no tabulated octahedral level structure.
    """
    if l not in nBaths:
        raise ValueError(f"get_CF_hamiltonian was asked for l={l}, which is not among the shells {sorted(nBaths)}.")
    atomic_physics.octahedral_level_structure(l)  # raises with the reason when l has none
    return l


def cf_bath_blocks(nBaths, nValBaths, l):
    """Which of the two crystal-field bath blocks this model actually has.

    The parametrisation gives each impurity spin-orbital one valence and one conduction bath
    partner, so each block is present in full or not at all. Zero of both is the Hubbard-I
    model: a bare correlated shell with no hybridization.

    Returns
    -------
    CFBathBlocks

    Raises
    ------
    ValueError
        If a block is partially present. Silently dropping half a bath would be far worse
        than refusing: the model would run and quietly not be the one that was asked for.
    """
    width = 2 * (2 * l + 1)
    n_val = nValBaths[l]
    n_con = nBaths[l] - n_val
    for name, count in (("valence", n_val), ("conduction", n_con)):
        if count not in (0, width):
            raise ValueError(
                f"The crystal-field parametrisation gives every one of the {width} impurity "
                f"spin-orbitals of the l={l} shell its own {name} bath partner, so that block "
                f"holds either 0 or {width} states -- not {count}. Supply a .h0 file to "
                "describe a bath of another size."
            )
    return CFBathBlocks(valence=n_val == width, conduction=n_con == width)


def _per_orbital(levels, per_level):
    """One value per O_h level -> one value per cubic-harmonic orbital.

    The column order lives in the level table rather than in a positional tuple written out
    beside ``np.fill_diagonal``; that tuple silently encoded the column order of
    ``get_spherical_2_cubic_matrix``, and a second shell would have needed a second tuple.
    """
    return [value for (_, degeneracy, _), value in zip(levels, per_level) for _ in range(degeneracy)]


def _by_irrep(levels, values):
    """A ``{irrep: value}`` mapping, ordered to match the level table."""
    return [values[irrep] for irrep, _, _ in levels]


def _hermitian(matrix):
    """Average a rotated matrix against its own adjoint.

    ``u D u^dagger`` is hermitian in exact arithmetic, but the f-shell rotation mixes four
    spherical harmonics per column and leaves the two halves differing in the last bit --
    enough for ``assert_hermitian``, which compares the assembled operator dicts for exact
    equality, to reject a perfectly good Hamiltonian. Where the product already came out
    exactly hermitian (the d-shell rotation does) this is bit-for-bit the identity: doubling
    and halving a double are both exact.
    """
    return (matrix + np.conj(matrix.T)) / 2


def _negligible(n_orb, *matrices):
    """Magnitude below which a rotated matrix element is round-off, not physics.

    Rotating a diagonal matrix by the 7x7 f transformation leaves ~1e-18 entries where
    symmetry forbids any coupling at all -- the d rotation happens not to. Keeping them would
    be harmless arithmetically but not structurally: basis generation walks the
    H-connectivity closure, and a 1e-18 hopping is an edge in that graph.

    The floor is derived from the matrices' own scale rather than written as a literal, and it
    is never stricter than the arithmetic that produced them: elements start being dropped
    only once the level splitting falls below ``eps * n`` of the overall scale, which is where
    double precision has already lost the splitting itself.
    """
    scale = max((float(np.max(np.abs(m))) for m in matrices), default=0.0)
    return np.finfo(float).eps * n_orb * scale


def _cf_impurity_operator(l, levels, e_imp, deltas, u):
    """The correlated shell's own block: its O_h levels, rotated to spherical harmonics."""
    n_orb = 2 * l + 1
    e_imp_levels = [e_imp + sum(w * delta for w, delta in zip(weights, deltas)) for _, _, weights in levels]
    h_imp = np.zeros((n_orb, n_orb))
    np.fill_diagonal(h_imp, _per_orbital(levels, e_imp_levels))
    h_imp = _hermitian(np.dot(u, np.dot(h_imp, np.conj(u.T))))
    floor = _negligible(n_orb, h_imp)

    operator = {}
    for i, mi in enumerate(range(-l, l + 1)):
        for j, mj in enumerate(range(-l, l + 1)):
            if abs(h_imp[i, j]) > floor:
                for s in range(2):
                    operator[(((l, s, mi), "c"), ((l, s, mj), "a"))] = h_imp[i, j]
    return operator


def _cf_bath_operators(l, levels, u, bath_state_basis, blocks, e_val, e_con, v_val, v_con):
    """Bath on-site energies, and the impurity-bath hopping.

    Returns ``(hopping, bath_energies)``. Either may be empty: a block the model does not have
    is skipped rather than emitted with zero amplitude, since its labels would not map.
    """
    n_orb = 2 * l + 1
    vVal, vCon, eBathVal, eBathCon = (np.zeros((n_orb, n_orb)) for _ in range(4))
    if blocks.valence:
        np.fill_diagonal(vVal, _per_orbital(levels, _by_irrep(levels, v_val)))
        np.fill_diagonal(eBathVal, _per_orbital(levels, _by_irrep(levels, e_val)))
    if blocks.conduction:
        np.fill_diagonal(vCon, _per_orbital(levels, _by_irrep(levels, v_con)))
        np.fill_diagonal(eBathCon, _per_orbital(levels, _by_irrep(levels, e_con)))

    # For the bath states, we can rotate to any basis. Which one is selected here.
    if bath_state_basis == "spherical":
        u_bath = u
    elif bath_state_basis == "cubic":
        u_bath = np.eye(np.shape(u)[0])
    else:
        raise Exception("Design of this basis is not (yet) implemented.")
    # Rotate the bath energies and the hopping parameters
    vVal = np.dot(u_bath, np.dot(vVal, np.conj(u.T)))
    vCon = np.dot(u_bath, np.dot(vCon, np.conj(u.T)))
    eBathVal = _hermitian(np.dot(u_bath, np.dot(eBathVal, np.conj(u_bath.T))))
    eBathCon = _hermitian(np.dot(u_bath, np.dot(eBathCon, np.conj(u_bath.T))))
    hopp_floor = _negligible(n_orb, vVal, vCon)
    bath_floor = _negligible(n_orb, eBathVal, eBathCon)

    hopping, bath_energies = {}, {}
    # Loop over spin
    for s in range(2):
        # Loop over impurity orbitals
        for i, _mi in enumerate(range(-l, l + 1)):
            # Bath state index for valence bath states, then for conduction bath states.
            bi_val = s * n_orb + i
            bi_con = 2 * n_orb + bi_val
            # Loop over impurity orbitals
            for j, mj in enumerate(range(-l, l + 1)):
                bj_val = s * n_orb + j
                bj_con = 2 * n_orb + bj_val
                # Hamiltonian values related to valence bath states.
                vHopp = vVal[i, j] if blocks.valence else 0.0
                eBath = eBathVal[i, j] if blocks.valence else 0.0
                if abs(vHopp) > hopp_floor:
                    hopping[(((l, bi_val), "c"), ((l, s, mj), "a"))] = vHopp
                    hopping[(((l, s, mj), "c"), ((l, bi_val), "a"))] = vHopp.conjugate()
                if abs(eBath) > bath_floor:
                    bath_energies[(((l, bi_val), "c"), ((l, bj_val), "a"))] = eBath
                # Only add the processes related to the conduction bath states if they are in
                # the basis: one conduction bath orbital per impurity spin-orbital.
                if blocks.conduction:
                    vHopp = vCon[i, j]
                    eBath = eBathCon[i, j]
                    if abs(vHopp) > hopp_floor:
                        hopping[(((l, bi_con), "c"), ((l, s, mj), "a"))] = vHopp
                        hopping[(((l, s, mj), "c"), ((l, bi_con), "a"))] = vHopp.conjugate()
                    if abs(eBath) > bath_floor:
                        bath_energies[(((l, bi_con), "c"), ((l, bj_con), "a"))] = eBath
    return hopping, bath_energies


def get_CF_hamiltonian(nBaths, nValBaths, h0_CF_filename, bath_state_basis="spherical", *, l):
    """
    Construct non-relativistic and non-interacting Hamiltonian, from CF parameters.

    The shell is named by the caller rather than fixed at l=2, and its level structure comes
    from :func:`atomic_physics.octahedral_level_structure`. A d shell therefore still splits
    into e_g and t_2g under a single ``e_deltaO_imp`` = 10Dq, while an f shell splits into
    t_1u, t_2u and a_2u under two independent parameters. That second parameter is not a
    naming choice: the octahedral invariants of an l=3 shell span a two-dimensional space, so
    one number cannot place three levels.

    Parameters
    ----------
    nBaths : dict
        Number of bath states for each angular momentum.
    nValBaths : dict
        Number of valence bath states for each angular momentum.
    h0_CF_filename : str
        Filename of the non-relativistic non-interacting CF Hamiltonian operator, in json-format.
    bath_state_basis : str
        'spherical' or 'cubic'.
        Which basis to use for the bath states.
    l : int, keyword-only, required
        The shell being parametrised.

    Returns
    -------
    h0_operator : dict
        The non-relativistic non-interacting Hamiltonian in operator form.
        Hamiltonian describes the correlated orbitals and bath orbitals.
        tuple : complex,
        where each tuple describes a process of two steps (annihilation and then creation).
        Each step is described by a tuple of the form:
        (spin_orb, 'c') or (spin_orb, 'a'),
        where spin_orb is a tuple of the form (l, s, m) or (l, b) or ((l_a, l_b), b).

    Raises
    ------
    ValueError
        If ``l`` is not a shell of this model, has no octahedral level structure, or carries a
        partially-filled bath block.
    """
    cf_shell(nBaths, l)
    levels = atomic_physics.octahedral_level_structure(l)
    blocks = cf_bath_blocks(nBaths, nValBaths, l)
    e_imp, deltas, e_val, e_con, v_val, v_con = read_h0_CF_file(h0_CF_filename, l, blocks)
    u = atomic_physics.get_spherical_2_cubic_matrix(spinpol=False, l=l)

    impurity = _cf_impurity_operator(l, levels, e_imp, deltas, u)
    hopping, bath_energies = _cf_bath_operators(l, levels, u, bath_state_basis, blocks, e_val, e_con, v_val, v_con)
    # Add Hamiltonian terms to one operator. The order is observable: `addOps` builds by
    # insertion and the matvec accumulates in that order.
    return addOps([impurity, hopping, bath_energies])


#: Ni-in-NiO fallbacks for the d shell, kept because the legacy ``.json`` reader has always
#: filled absent keys from them. They are d-shell numbers, so no other shell gets defaults --
#: a missing f-shell key is an error rather than silently Ni's bath.
_NIO_CF_DEFAULTS = {
    "e_imp": -1.31796,
    "e_deltaO_imp": 0.60422,
    "e_val_eg": -4.4,
    "e_val_t2g": -6.5,
    "e_con_eg": 3,
    "e_con_t2g": 2,
    "v_val_eg": 1.883,
    "v_val_t2g": 1.395,
    "v_con_eg": 0.6,
    "v_con_t2g": 0.4,
}

#: Key naming the splitting of each octahedral invariant, in rank order (4, then 6). The
#: first keeps its historical name, so a d shell has exactly the keys it always had.
CF_SPLITTING_KEYS = ("e_deltaO_imp", "e_delta6_imp")


def cf_parameter_names(l, blocks=None):
    """Every ``[hamiltonian.crystal_field]`` key this model needs, in a stable order.

    A d shell with both bath blocks yields the historical ten. An f shell yields fifteen: one
    more splitting (``e_delta6_imp``) and three irreps instead of two on each bath row. s and
    p have a single octahedral level, so no splitting parameter exists at all for them.

    Absent bath blocks drop their rows rather than requiring values nothing reads. A
    bath-less valence shell -- the Hubbard-I approximation -- therefore needs only the
    impurity level: ``e_imp`` plus its splittings.

    Parameters
    ----------
    l : int
        Angular momentum of the shell being parametrised.
    blocks : CFBathBlocks, optional
        Which bath blocks the model has, from :func:`cf_bath_blocks`. Defaults to both, which
        is every key the shell could take.
    """
    if blocks is None:
        blocks = CFBathBlocks(valence=True, conduction=True)
    levels = atomic_physics.octahedral_level_structure(l)
    names = ["e_imp"]
    names += list(CF_SPLITTING_KEYS[: atomic_physics.n_octahedral_splittings(l)])
    rows = (
        ("e_val", blocks.valence),
        ("e_con", blocks.conduction),
        ("v_val", blocks.valence),
        ("v_con", blocks.conduction),
    )
    for prefix, present in rows:
        if present:
            names += [f"{prefix}_{irrep}" for irrep, _, _ in levels]
    return tuple(names)


def read_h0_CF_file(h0_CF_filename, l, blocks=None):
    """
    Reads CF Hamiltonian from json-file.

    Parameters
    ----------
    h0_CF_filename : str or Mapping
        Filename of the non-relativistic non-interacting CF Hamiltonian operator in
        json-format, or a mapping holding the same parameters directly.
    l : int
        Angular momentum of the shell being parametrised. Required: it decides which keys are
        read -- the octahedral irreps of that shell, and one splitting parameter per
        independent invariant (see :func:`cf_parameter_names`). It used to default to 2, a
        d-shell assumption hiding in a function that is no longer about d shells.
    blocks : CFBathBlocks, optional
        Which bath blocks exist. An absent block's keys are neither required nor read; its
        dict comes back empty. Defaults to both.

    Returns
    -------
    e_imp : float
        Average on-site energy of the correlated shell.
    deltas : tuple of float
        Octahedral splitting parameters, in invariant-rank order. Each is the full spread
        (highest level minus lowest) that invariant alone produces, which is what makes the
        single d-shell entry the historical 10Dq.
    e_val, e_con, v_val, v_con : dict
        Bath level positions and hybridizations, keyed by octahedral irrep name.

    Raises
    ------
    ValueError
        If a required key is absent and the shell is not the d shell whose absent keys the
        Ni-in-NiO defaults describe.

    Note
    ----
    If a parameter is not specified in the json-file, a default value will used. Only for
    ``l = 2``: the defaults are Ni-in-NiO numbers and mean nothing for another shell.

    """
    if blocks is None:
        blocks = CFBathBlocks(valence=True, conduction=True)
    if isinstance(h0_CF_filename, Mapping):
        parameters = dict(h0_CF_filename)
    else:
        with open(h0_CF_filename, "r") as file_handle:
            parameters = json.loads(file_handle.read())

    levels = atomic_physics.octahedral_level_structure(l)

    def read(name):
        if name in parameters:
            return parameters[name]
        if l == 2 and name in _NIO_CF_DEFAULTS:
            # Default values are for Ni in NiO.
            return _NIO_CF_DEFAULTS[name]
        raise ValueError(
            f"{name!r} is missing from the crystal-field parameters for the l={l} shell. "
            f"Required: {', '.join(cf_parameter_names(l, blocks))}. "
            "Defaults exist only for l=2, where they are Ni-in-NiO values."
        )

    e_imp = read("e_imp")
    deltas = tuple(read(key) for key in CF_SPLITTING_KEYS[: atomic_physics.n_octahedral_splittings(l)])
    e_val, e_con, v_val, v_con = (
        {irrep: read(f"{prefix}_{irrep}") for irrep, _, _ in levels} if present else {}
        for prefix, present in (
            ("e_val", blocks.valence),
            ("e_con", blocks.conduction),
            ("v_val", blocks.valence),
            ("v_con", blocks.conduction),
        )
    )
    return e_imp, deltas, e_val, e_con, v_val, v_con
