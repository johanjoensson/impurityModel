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

import numpy as np

from impurityModel.ed import atomic_physics, h0_format, op_parser, symmetries
from impurityModel.ed.operator_algebra import addOps, assert_hermitian, c2i, i2c


def get_noninteracting_hamiltonian_operator(nBaths, nValBaths, SOCs, hField, h0_filename, rank, verbose=True):
    """
    Build the non-interacting Hamiltonian operator.

    Combines spin-orbit coupling, magnetic field, and the non-interacting
    Hamiltonian read from a file.

    Parameters
    ----------
    nBaths : dict
        Number of bath orbitals.
    SOCs : tuple[float, float]
        Spin-orbit coupling constants for 2p and 3d shells.
    hField : tuple[float, float, float]
        Magnetic field components (hx, hy, hz).
    h0_filename : str
        Filename of the non-interacting Hamiltonian.
    rank : int
        MPI rank.
    verbose : bool, optional
        Whether to print output on rank 0. Default is True.

    Returns
    -------
    hOperator : dict
        The total non-interacting Hamiltonian operator.
    """
    # Divide up input parameters to more concrete variables
    xi_2p, xi_3d = SOCs
    hx, hy, hz = hField
    # Add SOC, in spherical harmonics basis.
    SOC2pOperator = atomic_physics.getSOCop(xi_2p, l=1)
    SOC3dOperator = atomic_physics.getSOCop(xi_3d, l=2)

    # Magnetic field
    hHfieldOperator = atomic_physics.gethHfieldop(hx, hy, hz, l=2)

    # Read the non-relativistic non-interacting Hamiltonian operator from file.
    h0_operator = read_h0_operator(h0_filename, nBaths, nValBaths, rank=rank, verbose=verbose)

    if rank == 0 and verbose:
        print(f"Non-interacting, non-relativistic Hamiltonian (h0): {len(h0_operator)} terms.")
    hOperator = addOps([hHfieldOperator, SOC2pOperator, SOC3dOperator, h0_operator])
    return hOperator


def read_h0_operator(filename, nBaths, nValBaths=None, rank=0, verbose=True):
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

    Returns
    -------
    dict
        The non-interacting Hamiltonian operator dictionary.
    """
    # Crystal-field parameters supplied directly rather than through a .json file. The TOML
    # input format takes this path so it can require all ten parameters explicitly: the file
    # reader below fills each absent one from a hard-coded Ni-in-NiO value, which silently
    # gives another material Ni's conduction bath.
    if isinstance(filename, Mapping):
        return get_CF_hamiltonian(nBaths, nValBaths, filename)

    _, ext = os.path.splitext(filename)
    if ext.lower() == ".pickle":
        return read_pickled_file(filename)
    if ext.lower() == ".dat":
        return read_h0_dict(filename)
    if ext.lower() == ".json":
        return get_CF_hamiltonian(nBaths, nValBaths, filename)
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


def get_hamiltonian_operator(nBaths, nValBaths, slaterCondon, SOCs, DCinfo, hField, h0_filename, rank, verbose=True):
    """
    Return the Hamiltonian, in operator form.

    Parameters
    ----------
    nBaths : dict
        Number of bath states for each angular momentum.
    nValBaths : dict
        Number of valence bath states for each angular momentum.
    slaterCondon : list
        List of Slater-Condon parameters.
    SOCs : list
        List of SOC parameters.
    DCinfo : list
        Contains information needed for the double counting energy.
    hField : list
        External magnetic field.
        Elements hx,hy,hz
    h0_filename : str
        Filename of non-interacting, non-relativistic operator.

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
    Fdd, Fpp, Fpd, Gpd = slaterCondon
    n0imps, chargeTransferCorrection = DCinfo

    h_non_interacting = get_noninteracting_hamiltonian_operator(
        nBaths, nValBaths, SOCs, hField, h0_filename, rank, verbose
    )
    # Calculate the U operator, in spherical harmonics basis.
    uOperator = atomic_physics.get2p3dSlaterCondonUop(Fdd=Fdd, Fpp=Fpp, Fpd=Fpd, Gpd=Gpd)
    dc = atomic_physics.dc_MLFT(n3d_i=n0imps[2], c=chargeTransferCorrection, Fdd=Fdd, n2p_i=n0imps[1], Fpd=Fpd, Gpd=Gpd)
    eDCOperator = {}
    for l in [2, 1]:
        # for il, l in enumerate([2, 1]):
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


def get_CF_hamiltonian(nBaths, nValBaths, h0_CF_filename, bath_state_basis="spherical"):
    """
    Construct non-relativistic and non-interacting Hamiltonian, from CF parameters.

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

    Returns
    -------
    h0_operator : dict
        The non-relativistic non-interacting Hamiltonian in operator form.
        Hamiltonian describes 3d orbitals and bath orbitals.
        tuple : complex,
        where each tuple describes a process of two steps (annihilation and then creation).
        Each step is described by a tuple of the form:
        (spin_orb, 'c') or (spin_orb, 'a'),
        where spin_orb is a tuple of the form (l, s, m) or (l, b) or ((l_a, l_b), b).

    """
    (
        e_imp,
        e_deltaO_imp,
        e_val_eg,
        e_val_t2g,
        e_con_eg,
        e_con_t2g,
        v_val_eg,
        v_val_t2g,
        v_con_eg,
        v_con_t2g,
    ) = read_h0_CF_file(h0_CF_filename)

    # Calculate impurity 3d Hamiltonian.
    # First formulate in cubic harmonics basis and then rotate to
    # the spherical harmonics basis.
    l = 2
    e_imp_eg = e_imp + 3 / 5 * e_deltaO_imp
    e_imp_t2g = e_imp - 2 / 5 * e_deltaO_imp
    h_imp_3d = np.zeros((2 * l + 1, 2 * l + 1))
    np.fill_diagonal(h_imp_3d, (e_imp_eg, e_imp_eg, e_imp_t2g, e_imp_t2g, e_imp_t2g))
    # Convert to spherical harmonics basis
    u = atomic_physics.get_spherical_2_cubic_matrix(spinpol=False, l=l)
    h_imp_3d = np.dot(u, np.dot(h_imp_3d, np.conj(u.T)))
    # Convert from matrix to operator form.
    # Also add spin.
    h_imp_3d_operator = {}
    for i, mi in enumerate(range(-l, l + 1)):
        for j, mj in enumerate(range(-l, l + 1)):
            if h_imp_3d[i, j] != 0:
                for s in range(2):
                    h_imp_3d_operator[(((l, s, mi), "c"), ((l, s, mj), "a"))] = h_imp_3d[i, j]

    # Bath (3d) on-site energies and hoppings.
    # Calculate hopping terms between bath and impurity.
    # First formulate the terms in the cubic harmonics basis.
    l = 2
    vVal3d = np.zeros((2 * l + 1, 2 * l + 1))
    vCon3d = np.zeros((2 * l + 1, 2 * l + 1))
    eBathVal3d = np.zeros((2 * l + 1, 2 * l + 1))
    eBathCon3d = np.zeros((2 * l + 1, 2 * l + 1))
    np.fill_diagonal(vVal3d, (v_val_eg, v_val_eg, v_val_t2g, v_val_t2g, v_val_t2g))
    np.fill_diagonal(vCon3d, (v_con_eg, v_con_eg, v_con_t2g, v_con_t2g, v_con_t2g))
    np.fill_diagonal(eBathVal3d, (e_val_eg, e_val_eg, e_val_t2g, e_val_t2g, e_val_t2g))
    np.fill_diagonal(eBathCon3d, (e_con_eg, e_con_eg, e_con_t2g, e_con_t2g, e_con_t2g))
    # For the bath states, we can rotate to any basis.
    # Which bath state basis to use is determined selected here.
    if bath_state_basis == "spherical":
        # One example is to use spherical harmonics basis for the bath states.
        # This implies the following rotation matrix:
        u_bath = u
    elif bath_state_basis == "cubic":
        # One example is to keep the cubic harmonics basis for the bath states.
        # This implies the following rotation matrix:
        u_bath = np.eye(np.shape(u)[0])
    else:
        raise Exception("Design of this basis is not (yet) implemented.")
    # Rotate the bath energies and the hopping parameters
    vVal3d = np.dot(u_bath, np.dot(vVal3d, np.conj(u.T)))
    vCon3d = np.dot(u_bath, np.dot(vCon3d, np.conj(u.T)))
    eBathVal3d = np.dot(u_bath, np.dot(eBathVal3d, np.conj(u_bath.T)))
    eBathCon3d = np.dot(u_bath, np.dot(eBathCon3d, np.conj(u_bath.T)))
    # Convert from matrix to operator form.
    # Also introduce spin.
    h_hopp_operator = {}
    e_bath_3d_operator = {}
    # Loop over spin
    for s in range(2):
        # Loop over impurity orbitals
        for i, _mi in enumerate(range(-l, l + 1)):
            # Bath state index for valence bath states.
            bi_val = s * (2 * l + 1) + i
            # Bath state index for conduction bath states.
            bi_con = 2 * (2 * l + 1) + bi_val
            # Loop over impurity orbitals
            for j, mj in enumerate(range(-l, l + 1)):
                # Bath state index for valence bath states.
                bj_val = s * (2 * l + 1) + j
                # Bath state index for conduction bath states.
                bj_con = 2 * (2 * l + 1) + bj_val
                # Hamiltonian values related to valence bath states.
                vHopp = vVal3d[i, j]
                eBath = eBathVal3d[i, j]
                if vHopp != 0:
                    h_hopp_operator[(((l, bi_val), "c"), ((l, s, mj), "a"))] = vHopp
                    h_hopp_operator[(((l, s, mj), "c"), ((l, bi_val), "a"))] = vHopp.conjugate()
                if eBath != 0:
                    e_bath_3d_operator[(((l, bi_val), "c"), ((l, bj_val), "a"))] = eBath
                # Only add the processes related to the conduction bath states if they are
                # in the basis.
                if nBaths[l] - nValBaths[l] == 10:
                    # Hamiltonian values related to conduction bath states.
                    vHopp = vCon3d[i, j]
                    eBath = eBathCon3d[i, j]
                    if vHopp != 0:
                        h_hopp_operator[(((l, bi_con), "c"), ((l, s, mj), "a"))] = vHopp
                        h_hopp_operator[(((l, s, mj), "c"), ((l, bi_con), "a"))] = vHopp.conjugate()
                    if eBath != 0:
                        e_bath_3d_operator[(((l, bi_con), "c"), ((l, bj_con), "a"))] = eBath

    # Add Hamiltonian terms to one operator.
    h0_operator = addOps([h_imp_3d_operator, h_hopp_operator, e_bath_3d_operator])
    return h0_operator


def read_h0_CF_file(h0_CF_filename):
    """
    Reads CF Hamiltonian from json-file.

    Parameters
    ----------
    h0_CF_filename : str or Mapping
        Filename of the non-relativistic non-interacting CF Hamiltonian operator in
        json-format, or a mapping holding the same parameters directly.

    Returns
    -------
    e_imp : float
        Average 3d onsite energy.
    e_deltaO_imp : float
        Energy split of 3d orbitals into eg and t2g orbitals.
    e_val_eg : float
        Energy position of valence bath states, coupled to eg orbitals.
    e_val_t2g : float
        Energy position of valence bath states, coupled to t2g orbitals.
    e_con_eg : float
        Energy position of conduction bath states, coupled to eg orbitals.
    e_con_t2g : float
        Energy position of conduction bath states, coupled to t2g orbitals.
    v_val_eg : float
        Hybridization/hopping strength of valence bath states with eg orbitals.
    v_val_t2g : float
        Hybridization/hopping strength of valence bath states with t2g orbitals.
    v_con_eg : float
        Hybridization/hopping strength of conduction bath states with eg orbitals.
    v_con_t2g : float
        Hybridization/hopping strength of conduction bath states with t2g orbitals.

    Note
    ----
    If a parameter is not specified in the json-file, a default value will used.

    """
    if isinstance(h0_CF_filename, Mapping):
        parameters = dict(h0_CF_filename)
    else:
        with open(h0_CF_filename, "r") as file_handle:
            parameters = json.loads(file_handle.read())
    # Default values are for Ni in NiO.
    e_imp = parameters.get("e_imp", -1.31796)
    e_deltaO_imp = parameters.get("e_deltaO_imp", 0.60422)
    e_val_eg = parameters.get("e_val_eg", -4.4)
    e_val_t2g = parameters.get("e_val_t2g", -6.5)
    e_con_eg = parameters.get("e_con_eg", 3)
    e_con_t2g = parameters.get("e_con_t2g", 2)
    v_val_eg = parameters.get("v_val_eg", 1.883)
    v_val_t2g = parameters.get("v_val_t2g", 1.395)
    v_con_eg = parameters.get("v_con_eg", 0.6)
    v_con_t2g = parameters.get("v_con_t2g", 0.4)
    return (
        e_imp,
        e_deltaO_imp,
        e_val_eg,
        e_val_t2g,
        e_con_eg,
        e_con_t2g,
        v_val_eg,
        v_val_t2g,
        v_con_eg,
        v_con_t2g,
    )
