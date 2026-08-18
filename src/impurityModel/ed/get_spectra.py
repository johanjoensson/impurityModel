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
    Fdd,
    Fpp,
    Fpd,
    Gpd,
    xi_2p,
    xi_3d,
    chargeTransferCorrection,
    hField,
    rank=0,
    verbose=True,
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
        Angular momenta of the correlated shells (e.g. ``(1, 2)`` for 2p + 3d).
    nBaths, nValBaths : sequence of int
        Total / valence bath-state counts, one per shell in ``ls``.
    n0imps : sequence of int
        Nominal impurity occupation per shell (used for the double-counting term).
    Fdd, Fpp, Fpd, Gpd : sequence of float
        Slater-Condon parameters.
    xi_2p, xi_3d : float
        Spin-orbit couplings for the p- and d-shell.
    chargeTransferCorrection : float
        Double-counting parameter.
    hField : sequence of float
        Magnetic field ``(hx, hy, hz)``.
    rank : int
        MPI rank, forwarded to the reader for rank-0 logging.
    verbose : bool
        Whether the reader logs on rank 0.

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
        slater_condon=(Fdd, Fpp, Fpd, Gpd),
        socs=(xi_2p, xi_3d),
        charge_transfer_correction=chargeTransferCorrection,
        h_field=hField,
        rank=rank,
        verbose=verbose,
    )


def run_spectra(model, spectra_options, basis, comm, *, verbosity=None):
    """Find the lowest eigenstates of ``model`` and calculate the requested spectra.

    Extracted verbatim from the historical ``get_spectra.main``: builds the many-body ground
    state, derives (or keeps) the block structure, optionally rotates the correlated shell into
    a symmetry-adapted basis, writes ``spectra.h5`` and calls
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

    # Hand-coded fall-back block structure (2p + 3d): used when auto_block_structure is off.
    block_structure = BlockStructure(
        blocks=[list(range(6)), list(range(6, 16))],
        identical_blocks=[[i] for i in range(2)],
        transposed_blocks=[[] for _ in range(2)],
        particle_hole_blocks=[[] for _ in range(2)],
        particle_hole_transposed_blocks=[[] for _ in range(2)],
        inequivalent_blocks=list(range(2)),
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
    if spectra_options.wIn is not None:
        wIn = spectra_options.wIn
    elif deltaRIXS > 0:
        wIn = np.linspace(-10, 20, 50)
    else:
        wIn = []
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
    rotation = None
    correlated_block_structure = None
    correlated_l = 2
    if spectra_options.auto_block_structure:
        impurity_indices = sorted(orb for blocks in impurity_orbitals.values() for block in blocks for orb in block)
        h_matrix = extract_tensors(hOp, n_orb=n_spin_orbitals, two_body=False)[0]
        block_structure = impurity_block_structure(hOp, impurity_indices, h0_matrix=h_matrix)
        report(f"Auto-derived block structure: {len(block_structure.blocks)} blocks", level=V_RESULT)

        if correlated_l in impurity_orbitals:
            d_indices = sorted(orb for block in impurity_orbitals[correlated_l] for orb in block)
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
                rot_to_spherical[correlated_l] = u_imp.conj().T
                n_classes = len(correlated_block_structure.inequivalent_blocks)
                report(
                    f"Rotated 3d shell into symmetry-adapted basis (fill {fill_ratio:.2f}x); "
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
        "frozen_occupations": {i for i in nBaths if nBaths[i] == 0},
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
        h5f = h5py.File("spectra.h5", "w")
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
        correlated_l=correlated_l,
        correlated_block_structure=correlated_block_structure,
    )

    if h5f is not None:
        h5f.close()
    if comm is not None:
        comm.Barrier()
    report("\nDone.", level=V_RESULT)
