"""Library functions for calculating various spectra.

``build_spectra_model`` assembles the full interacting :class:`ImpurityModel` from a
non-interacting ``h0`` file; ``run_spectra`` solves the ground state and writes ``spectra.h5``.
The command-line interface lives in :mod:`impurityModel.scripts.spectra`.
"""

from collections import OrderedDict

import h5py
import numpy as np

# Local stuff
from impurityModel.ed import spectra
from impurityModel.ed.basis_restrictions import build_weighted_restrictions
from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.groundstate import calc_gs
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.model import load_model
from impurityModel.ed.symmetries import (
    extract_tensors,
    impurity_block_structure,
    impurity_symmetry_rotation,
    rotate_hamiltonian,
)
from impurityModel.ed.utils import V_RESULT, Reporter


def build_spectra_model(
    h0_filename,
    ls,
    nBaths,
    nValBaths,
    n0imps,
    F_vv,
    F_cc,
    F_cv,
    G_cv,
    xi_core,
    xi_valence,
    chargeTransferCorrection,
    hField,
    rank=0,
    verbose=True,
    *,
    valence_l,
    core_l=None,
):
    """Assemble the full interacting spectra model from a non-interacting ``h0`` file.

    Thin wrapper over :func:`model.load_model`'s multi-shell path
    (:meth:`ImpurityModel.from_shells`), kept so :mod:`impurityModel.scripts.spectra` and its
    tests do not need to build ``OrderedDict``s themselves. ``h0_filename`` accepts either a
    labelled ``.pickle``/``.json``/``.dat`` file or a self-describing flat ``.h0`` (see
    :func:`hamiltonian_io.read_h0_operator`).

    Unlike the self-energy path (``h0`` + separate ``u4``), the spectra driver works with the
    *full* single-index interacting operator that :func:`hamiltonian_io.get_hamiltonian_operator`
    builds (core + correlated shells, SOC, magnetic field, atomic Coulomb and double counting all
    folded in). It is carried as :attr:`ImpurityModel.h0` with ``u4=None``; the explicit
    multi-shell ``(valence_baths, conduction_baths)`` partition is stored in
    :attr:`ImpurityModel.bath_states`.

    Parameters
    ----------
    h0_filename : str
        Non-interacting Hamiltonian file.
    ls : sequence of int
        Angular momenta of the shells (e.g. ``(1, 2)`` for a 2p core + 3d valence shell).
        This fixes the ``c2i`` index layout only; which shell is the core one is said by
        ``core_l``, never by this sequence's order.
    nBaths, nValBaths : sequence of int
        Total / valence bath-state counts, one per shell in ``ls``.
    n0imps : sequence of int
        Nominal impurity occupation per shell (used for the double-counting term).
    F_vv, F_cc, F_cv, G_cv : sequence of float
        Slater-Condon parameters: valence-valence direct, core-core direct, core-valence
        direct and core-valence exchange. The last three are ignored when ``core_l`` is
        ``None``.
    xi_core, xi_valence : float
        Spin-orbit couplings of the core and valence shells.
    chargeTransferCorrection : float
        Double-counting parameter.
    hField : sequence of float
        Magnetic field ``(hx, hy, hz)``, applied to the valence shell.
    rank : int
        MPI rank, forwarded to the reader for rank-0 logging.
    verbose : bool
        Whether the reader logs on rank 0.
    valence_l : int
        Angular momentum of the valence (correlated) shell.
    core_l : int, optional
        Angular momentum of the core shell; ``None`` when the model has no core shell.

    Returns
    -------
    ImpurityModel
        With ``h0`` = the full interacting operator, ``u4=None``, ``impurity_orbitals`` the
        per-shell block lists, and ``bath_states = (valence_baths, conduction_baths)``.
    """
    return load_model(
        h0_filename,
        shells=OrderedDict(zip(ls, nBaths)),
        val_shells=OrderedDict(zip(ls, nValBaths)),
        n0imps=OrderedDict(zip(ls, n0imps)),
        slater_condon=(F_vv, F_cc, F_cv, G_cv),
        socs=(xi_core, xi_valence),
        charge_transfer_correction=chargeTransferCorrection,
        h_field=hField,
        valence_l=valence_l,
        core_l=core_l,
        rank=rank,
        verbose=verbose,
    )


def resolve_spectra_switches(options):
    """Decide which spectra to compute, and the RIXS incoming mesh that goes with the answer.

    Each technique has its own switch. A ``None`` switch reproduces the historical inference
    exactly -- PES, XPS and XAS always; RIXS when its broadening is positive and its mesh
    non-empty; NIXS when radial data was supplied -- so an API caller that sets none of them
    sees no change. An explicit switch is authoritative, and what used to *disable* a
    technique as a side effect becomes a requirement of an enabled one: a non-positive
    broadening or an empty mesh is now an error rather than a silent skip, because a
    broadening doubling as a feature flag is unguessable and gave RIXS two independent
    switches with no stated precedence.

    Parameters
    ----------
    options : impurityModel.ed.model.SpectraOptions

    Returns
    -------
    switches : dict
        ``{"pes": bool, "xps": bool, "xas": bool, "rixs": bool, "nixs": bool}``.
    wIn : numpy.ndarray or list
        The RIXS incoming-energy mesh, defaulted when unset and empty when RIXS is off.

    Raises
    ------
    ValueError
        If a technique is enabled but the data it needs is absent or non-physical.
    """
    rixs = options.rixs
    if rixs is None:
        rixs = options.deltaRIXS > 0 and (options.wIn is None or len(options.wIn) > 0)
    elif rixs and options.deltaRIXS <= 0:
        raise ValueError(
            f"RIXS is enabled but deltaRIXS is {options.deltaRIXS}. A non-positive broadening "
            "used to be how RIXS was switched off; now the switch is the switch, and a "
            "broadening has to be a broadening."
        )

    nixs = options.nixs
    if nixs is None:
        nixs = options.radial is not None
    elif nixs and options.radial is None:
        raise ValueError(
            "NIXS is enabled but no radial data was supplied. Supplying it used to be what "
            "switched NIXS on; now it is a requirement of having switched NIXS on."
        )
    if nixs and options.deltaNIXS <= 0:
        raise ValueError(f"NIXS is enabled but deltaNIXS is {options.deltaNIXS}.")

    if options.wIn is not None:
        wIn = options.wIn
    elif rixs:
        wIn = np.linspace(-10, 20, 50)
    else:
        wIn = []
    if rixs and len(wIn) == 0:
        raise ValueError("RIXS is enabled but its incoming-energy mesh is empty.")

    switches = {
        "pes": True if options.pes is None else options.pes,
        "xps": True if options.xps is None else options.xps,
        "xas": True if options.xas is None else options.xas,
        "rixs": rixs,
        "nixs": nixs,
    }
    if not any(switches.values()):
        raise ValueError("Every spectrum is disabled, so there is nothing to compute.")
    return switches, wIn


def _resolve_shell_roles(nBaths):
    """Return ``(l_valence, l_core)`` for a spectra model, from the bath counts.

    A last resort, for a model that does not carry :attr:`ImpurityModel.valence_l` -- one
    built from an archive or a bare matrix rather than through
    :meth:`ImpurityModel.from_shells`, which is told the roles outright. The rule is that the
    core shell is the one with no bath of its own, which is what every RSPt-produced workload
    looks like (one hybridization file per orbital group, and a core group has none).

    It is a guess, and a bath-less *valence* shell -- the Hubbard-I approximation -- is exactly
    the case it would get backwards. So a layout it cannot read unambiguously raises and says
    to carry the roles instead.

    Parameters
    ----------
    nBaths : Mapping
        ``{l: n_bath}`` for every shell in the model.

    Returns
    -------
    tuple
        ``(l_valence, l_core)``; ``l_core`` is ``None`` when the model has no core shell.

    Raises
    ------
    ValueError
        If the shells do not resolve to exactly one valence shell and at most one core shell.
    """
    with_bath = [l for l, n in nBaths.items() if n > 0]
    without_bath = [l for l, n in nBaths.items() if n == 0]

    if len(nBaths) == 1:
        return next(iter(nBaths)), None
    if len(with_bath) == 1 and len(without_bath) == 1:
        return with_bath[0], without_bath[0]
    raise ValueError(
        f"Cannot tell which shell is the core one from the bath counts {dict(nBaths)}. Read this "
        "way, a spectra model needs exactly one shell with bath states (the valence shell) and at "
        "most one without (the core shell) -- which a bath-less valence shell (Hubbard-I) "
        "alongside a core shell is not. Build the model through ImpurityModel.from_shells, which "
        "is told valence_l/core_l outright, rather than leaving them to be inferred."
    )


def run_spectra(model, spectra_options, basis, comm, *, verbosity=None, output_filename="spectra.h5"):
    """Find the lowest eigenstates of ``model`` and calculate the requested spectra.

    Extracted verbatim from the historical ``get_spectra.main``: builds the many-body ground
    state, derives (or keeps) the block structure, optionally rotates the correlated shell into
    a symmetry-adapted basis, writes ``output_filename`` and calls
    :func:`impurityModel.ed.spectra.simulate_spectra`.

    Parameters
    ----------
    model : ImpurityModel
        The full interacting model from :func:`build_spectra_model` (``bath_states`` set).
    spectra_options : SpectraOptions
        Meshes, broadenings, polarizations and (optional) NIXS radial data. ``None`` array
        fields are filled with the historical default grids here.
    basis : BasisOptions
        Nominal occupation, determinant budget and ``tau = k_B * T``.
    comm : mpi4py communicator
        MPI communicator (``MPI.COMM_WORLD`` for the CLI).
    output_filename : str, optional
        Where to write the results. The path was hard-coded relative to the working directory,
        which left a caller wanting them elsewhere with only ``os.chdir`` -- process-global,
        and unsafe under MPI where every rank shares the interpreter's working directory.
        Mirrors the keyword :func:`impurityModel.ed.susceptibility.calc_susceptibility_workflow`
        already had.
    verbosity : int, optional
        Printing level (rank-uniform; printing itself is gated on rank downstream).
        ``None`` -> ``V_RESULT`` (0), the terse default.
    """
    rank = comm.rank if comm is not None else 0
    if verbosity is None:
        verbosity = V_RESULT
    # The stage one-liners below are the provenance heartbeat users pipe to logs; they stay
    # ungated (level V_RESULT) regardless of verbosity, so `report` here is only for the
    # shared rank gate and formatting, not for hiding anything at the default level.
    report = Reporter(verbosity, rank)

    hOp = ManyBodyOperator(model.h0)
    impurity_orbitals = model.impurity_orbitals
    valence_baths, conduction_baths = model.bath_states
    rot_to_spherical = dict(model.rot_to_spherical)

    ls = list(impurity_orbitals.keys())
    nBaths = OrderedDict(
        (l, sum(len(b) for b in valence_baths[l]) + sum(len(b) for b in conduction_baths[l])) for l in ls
    )
    n_spin_orbitals = model.n_spin_orbitals

    # Fall-back block structure -- one block per impurity shell -- used when
    # auto_block_structure is off. Derived from impurity_orbitals rather than written out as
    # [range(6), range(6, 16)], which was the 2p + 3d spin-orbital counts spelled as literals.
    shell_blocks = [sorted(orb for block in impurity_orbitals[l] for orb in block) for l in ls]
    n_shell_blocks = len(shell_blocks)
    block_structure = BlockStructure(
        blocks=shell_blocks,
        identical_blocks=[[i] for i in range(n_shell_blocks)],
        transposed_blocks=[[] for _ in range(n_shell_blocks)],
        particle_hole_blocks=[[] for _ in range(n_shell_blocks)],
        particle_hole_transposed_blocks=[[] for _ in range(n_shell_blocks)],
        inequivalent_blocks=list(range(n_shell_blocks)),
    )

    # -- Spectra meshes / polarizations: fill the unset (None) fields with today's defaults. --
    w = spectra_options.w if spectra_options.w is not None else np.linspace(-25, 25, 3001)
    delta = spectra_options.delta
    cartesian = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    epsilons = spectra_options.epsilons if spectra_options.epsilons is not None else cartesian
    epsilonsRIXSin = spectra_options.epsilonsRIXSin if spectra_options.epsilonsRIXSin is not None else cartesian
    epsilonsRIXSout = spectra_options.epsilonsRIXSout if spectra_options.epsilonsRIXSout is not None else cartesian
    deltaRIXS = spectra_options.deltaRIXS
    deltaNIXS = spectra_options.deltaNIXS
    switches, wIn = resolve_spectra_switches(spectra_options)
    wLoss = spectra_options.wLoss if spectra_options.wLoss is not None else np.linspace(-2.0, 12.0, 4000)
    qsNIXS = (
        spectra_options.qsNIXS
        if spectra_options.qsNIXS is not None
        else [2 * np.array([1, 1, 1]) / np.sqrt(3), 7 * np.array([1, 1, 1]) / np.sqrt(3)]
    )
    liNIXS, ljNIXS = spectra_options.liNIXS, spectra_options.ljNIXS
    XAS_projectors = spectra_options.XAS_projectors
    RIXS_projectors = spectra_options.RIXS_projectors

    # NIXS needs the radial part of the correlated orbitals; it is optional. When no radial data
    # was supplied the radial arrays stay None and simulate_spectra skips the NIXS block.
    if spectra_options.radial is not None:
        radialMesh, RiNIXS, RjNIXS = spectra_options.radial
    else:
        radialMesh = RiNIXS = RjNIXS = None

    report(f"Number of spin-orbitals: {n_spin_orbitals}", level=V_RESULT)

    # Default: derive the block structure from the hybridization-dressed impurity matrix
    # (impurity_block_structure) rather than the hand-coded one. It matches or strictly refines
    # the manual structure (e.g. SOC / crystal field splits each shell into sub-blocks) and
    # fixes bath-mediated coupling. Pass auto_block_structure=False to keep the hand-coded one.
    #
    # Adaptive symmetry-adapted solver basis: rotate the correlated 3d shell into the basis that
    # diagonalises its one-body block, IF that keeps the Coulomb term roughly as sparse (the
    # fill-ratio gate; a d-shell with SOC densifies ~8x and stays spherical). The scalar XAS /
    # PES / NIXS / RIXS spectra are basis-invariant, so simulate_spectra just rotates the one-body
    # transition operators to match and deduplicates the now-degenerate PES/IPS operators (B2a).
    # Which shell is which is carried by the model, because from_shells was told. Only a
    # model built some other way (an archive, a bare matrix) falls back to reading the roles
    # off the bath counts.
    if model.valence_l is not None:
        l_valence, l_core = model.valence_l, model.core_l
    else:
        l_valence, l_core = _resolve_shell_roles(nBaths)

    rotation = None
    correlated_block_structure = None
    if spectra_options.auto_block_structure:
        impurity_indices = sorted(orb for blocks in impurity_orbitals.values() for block in blocks for orb in block)
        h_matrix = extract_tensors(hOp, n_orb=n_spin_orbitals, two_body=False)[0]
        block_structure = impurity_block_structure(hOp, impurity_indices, h0_matrix=h_matrix)
        report(f"Auto-derived block structure: {len(block_structure.blocks)} blocks", level=V_RESULT)

        if l_valence in impurity_orbitals:
            d_indices = sorted(orb for block in impurity_orbitals[l_valence] for orb in block)
            W, u_imp = impurity_symmetry_rotation(hOp, d_indices, n_orb=n_spin_orbitals, h0_matrix=h_matrix)
            h_rotated = rotate_hamiltonian(hOp, W, tol=spectra._ROTATION_TRIM_TOL)
            fill_ratio = len(h_rotated) / max(1, len(hOp))
            if fill_ratio <= spectra._MAX_ROTATION_FILL:
                rotation = W
                hOp = h_rotated
                h_matrix = extract_tensors(hOp, n_orb=n_spin_orbitals, two_body=False)[0]
                block_structure = impurity_block_structure(hOp, impurity_indices, h0_matrix=h_matrix)
                correlated_block_structure = impurity_block_structure(hOp, d_indices, h0_matrix=h_matrix)
                # rot_to_spherical maps the (rotated) computational basis back to spherical harmonics
                # for the L/S/J Casimir reporting in calc_gs; identity on the un-rotated core p shell.
                rot_to_spherical[l_valence] = u_imp.conj().T
                n_classes = len(correlated_block_structure.inequivalent_blocks)
                report(
                    f"Rotated the l={l_valence} shell into symmetry-adapted basis "
                    f"(fill {fill_ratio:.2f}x); "
                    f"{n_classes} inequivalent PES/IPS classes.",
                    level=V_RESULT,
                )
            else:
                report(
                    f"Kept spherical basis (rotation would densify {fill_ratio:.2f}x"
                    f" > {spectra._MAX_ROTATION_FILL}).",
                    level=V_RESULT,
                )
                correlated_block_structure = impurity_block_structure(hOp, d_indices, h0_matrix=h_matrix)
    # Measure how many physical processes the Hamiltonian contains.
    report(f"Hamiltonian contains {len(hOp)} terms.", level=V_RESULT)
    # Many body basis for the ground state
    report("Creating the many-body basis ...", level=V_RESULT, flush=True)
    tau = basis.tau
    basis_setup = {
        "impurity_orbital": impurity_orbitals,
        "bath_states": (valence_baths, conduction_baths),
        "nominal_impurity_occ": basis.nominal_occ,
        # Pin the core shell at generation time so it cannot drain into the valence shell (a
        # 2p^4 d^10 configuration otherwise wins by ~23 eV and zeroes every core-level
        # spectrum -- see doc/basis_and_restrictions.md). Keyed on the shell the model says is
        # the core one, not on "the shell with no bath": those agreed only while both were the
        # same inference, and a core shell that does carry bath states would be left unpinned.
        "frozen_occupations": set() if l_core is None else {l_core},
        # None = "as many determinants as fit in RAM", resolved against the per-rank
        # available memory inside find_ground_state_basis (see memory_estimate).
        "truncation_threshold": basis.truncation_threshold,
        "tau": tau,
        "comm": comm,
        # Optional excitation-budget weighted restriction on the ground-state basis; the XAS/PES
        # excited bases inherit it (widened) via spectra/greens_function; RIXS attaches it
        # explicitly (rixs.py).
        "weighted_restrictions": build_weighted_restrictions(
            (valence_baths, conduction_baths), basis.excitation_budget
        ),
    }
    psis, es, ground_state_basis, _rho, _ = calc_gs(
        hOp,
        basis_setup,
        block_structure,
        rot_to_spherical,
        verbosity,
    )

    # Save some of the arrays. HDF5-format does not directly support dictionaries.
    h5f = None
    if rank == 0:
        h5f = h5py.File(output_filename, "w")
        h5f.create_dataset("E", data=es)
        h5f.create_dataset("w", data=w)
        h5f.create_dataset("wIn", data=wIn)
        h5f.create_dataset("wLoss", data=wLoss)
        h5f.create_dataset("qsNIXS", data=qsNIXS)
        if radialMesh is not None:
            h5f.create_dataset("r", data=radialMesh)
            h5f.create_dataset("RiNIXS", data=RiNIXS)
            h5f.create_dataset("RjNIXS", data=RjNIXS)

    report.banner("Spectra", level=V_RESULT)
    report(f"Considering {len(es)} eigenstate(s) for the spectra.", level=V_RESULT)
    report("Calculating spectra ...", level=V_RESULT, flush=True)
    spectra.simulate_spectra(
        es,
        psis,
        hOp,
        tau,
        w,
        delta,
        epsilons,
        wLoss,
        deltaNIXS,
        qsNIXS,
        liNIXS,
        ljNIXS,
        RiNIXS,
        RjNIXS,
        radialMesh,
        wIn,
        deltaRIXS,
        epsilonsRIXSin,
        epsilonsRIXSout,
        ground_state_basis.restrictions,
        h5f,
        nBaths,
        XAS_projectors,
        RIXS_projectors,
        ground_state_basis,
        basis.occ_cutoff,
        basis.dN if basis.dN is not None else 2,
        basis.slater_weight_min,
        verbosity >= 1,
        rotation=rotation,
        correlated_block_structure=correlated_block_structure,
        l_valence=l_valence,
        l_core=l_core,
        **switches,
    )

    if h5f is not None:
        h5f.close()
    if comm is not None:
        comm.Barrier()
    report("\nDone.", level=V_RESULT)
