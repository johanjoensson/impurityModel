"""
This module contains functions for calculating various spectra.
"""

import time

import numpy as np
from mpi4py import MPI

# Local imports
import impurityModel.ed.greens_function as gf
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, inner
from impurityModel.ed.ManyBodyUtils import applyOp as applyOp_test
from impurityModel.ed.operator_algebra import arrayOp2Dict, c2i, combineOp
from impurityModel.ed.symmetries import (
    ComponentReduction,
    component_symmetry_reduction,
    conserved_subset_charges,
    extract_tensors,
    measure_conserved_charges,
    rotate_hamiltonian,
    transition_sector_restrictions,
)

# The transition-operator builders now live in their own physics module. They are re-exported
# here (and referenced by the bare names below) so that simulate_spectra's calls resolve and
# existing callers/tests that reach them via ``spectra.dipole_operators`` etc. keep working.
from impurityModel.ed.transition_operators import (  # noqa: F401
    daggered_dipole_operators,
    dipole_operator,
    dipole_operators,
    inverse_photoemission_operators,
    nixs_operator,
    nixs_operators,
    photoemission_operators,
    sph_harm,
)

# MPI variables
comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size

# Adaptive-rotation gate, mirroring selfenergy: rotate the correlated shell into its
# symmetry-adapted basis only when doing so keeps the operator roughly as sparse (a d-shell
# with SOC densifies the Coulomb term ~8x and must stay in spherical harmonics). See the
# "symmetry-rotation-densifies-coulomb" note.
_ROTATION_TRIM_TOL = 1e-8
_MAX_ROTATION_FILL = 2.0


def _rotate_op(tOp, rotation):
    """Rotate a one-body transition operator into the symmetry-adapted basis (U† T U).

    Accepts a ``ManyBodyOperator`` or the ``{process: amplitude}`` dict the
    :mod:`transition_operators` builders return; always returns an operator.
    """
    if not isinstance(tOp, ManyBodyOperator):
        tOp = ManyBodyOperator(tOp)
    return rotate_hamiltonian(tOp, rotation, tol=_ROTATION_TRIM_TOL)


def _shell_orbitals(nBaths, l):
    """Sorted global spin-orbital indices of impurity shell ``l`` (spin-up then spin-down m's)."""
    return sorted(c2i(nBaths, (l, s, m)) for s in range(2) for m in range(-l, l + 1))


def _pes_ips_equivalence_groups(nBaths, l, block_structure):
    """Symmetry-equivalence label per PES/IPS operator of shell ``l`` (B2a dedup).

    ``photoemission_operators`` / ``inverse_photoemission_operators`` emit one operator per
    ``(s, m)`` in the order ``for s in 0,1: for m in -l..l``. ``block_structure`` is the impurity
    block structure of shell ``l`` in the symmetry-adapted basis (local indices into the sorted
    shell). Operators whose orbital lands in the same ``identical_blocks`` class get the same
    label, so :func:`calc_spectra` computes one representative per class.
    """
    shell = _shell_orbitals(nBaths, l)
    local_of_global = {orb: k for k, orb in enumerate(shell)}
    class_of_block = {}
    for cls_id, cls in enumerate(block_structure.identical_blocks):
        for b in cls:
            class_of_block[b] = cls_id
    label_of_local = {}
    for b, block in enumerate(block_structure.blocks):
        for local in block:
            label_of_local[local] = class_of_block.get(b, b)
    return [label_of_local[local_of_global[c2i(nBaths, (l, s, m))]] for s in range(2) for m in range(-l, l + 1)]


def simulate_spectra(
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
    restrictions,
    h5f,
    nBaths,
    XAS_projectors,
    RIXS_projectors,
    basis,
    occ_cutoff,
    dN,
    slaterWeightMin,
    verbose,
    rotation=None,
    correlated_l=2,
    correlated_block_structure=None,
):
    """
    Simulate various spectra.

    Parameters
    ----------
    es : tuple
        Eigen-energy (in eV).
    psis : tuple
        Many-body eigen-states.
    hOp : dict
        The Hamiltonian in operator form.
        tuple : complex,
        where each tuple describes a process of several steps.
        Each step is described by a tuple of the form: (i,'c') or (i,'a'),
        where i is a spin-orbital index.
    T : float
        Temperature (in Kelvin).
    w : ndarray
        Real-energy mesh (in eV).
    delta : float
        Distance above the real axis (in eV).
        Gives smearing to spectra.
    epsilons : list
        Each element is a XAS polarization vector. Only consumed when ``XAS_projectors`` is
        given (projected operators are not linear in the polarization); otherwise the XAS
        spectral tensor is stored and polarization contraction is a post-processing step (see
        ``impurityModel.ed.polarization``).
    wLoss : ndarray
        Real-energy mesh (in eV).
        Incoming minus outgoing photon energy.
    deltaNIXS : float
        Distance above the real axis (in eV).
        Gives smearing to NIXS spectra.
    qsNIXS : list
        Various momenta used in NIXS.
    liNIXS : int
        Angular momentum of final orbitals in the NIXS excitation process.
    ljNIXS : int
        Angular momentum of initial orbitals in the NIXS excitation process.
    RiNIXS : ndarray
        Radial part of final correlated orbitals.
    RjNIXS : ndarray
        Radial part of initial correlated orbitals.
    radialMesh : ndarray
        Radial mesh, using in NIXS.
    wIn : ndarray
        Incoming photon energies in RIXS.
    deltaRIXS : float
        Distance above the real axis (in eV).
        Gives smearing to RIXS spectra.
    epsilonsRIXSin : list
        Polarization vectors of in-going photon. Only consumed when ``RIXS_projectors`` is
        given; otherwise the RIXS Kramers-Heisenberg tensor is stored and polarization
        contraction is a post-processing step (see ``impurityModel.ed.polarization``).
    epsilonsRIXSout : list
        Polarization vectors of out-going photon. Same caveat as ``epsilonsRIXSin``.
    restrictions : dict
        Restriction the occupation of generated
        product states.
    h5f : h5py file-handle
        Will be used to write data to disk. This is the single output of a spectra run --
        each spectrum/tensor is written under its own group (``PS/spectra``, ``XPS/spectra``,
        ``NIXS/spectra``, ``XAS/tensor`` or ``XAS/projected``, ``RIXS/tensor`` or
        ``RIXS/projected``) as complex arrays; no separate quick-look ``.dat``/``.bin`` files
        are written. See ``impurityModel.ed.polarization`` for turning a tensor into an
        intensity for one or more polarizations.
    nBaths : OrderedDict
        Angular momentum : number of bath states.
    RIXS_projectors : dict
        dict of dicts representing the projections to apply for the calculation of the RIXS spectra
    rotation : np.ndarray, optional
        Full-space single-particle unitary rotating the correlated shell into its symmetry-adapted
        basis. When given, ``hOp`` and ``psis`` are assumed already expressed in that basis, so the
        one-body transition operators (dipole/NIXS/RIXS) are rotated to match; the scalar spectra
        are basis-invariant and need no un-rotation. ``None`` keeps the spherical-harmonics basis.
    correlated_l : int, optional
        Angular momentum of the rotated correlated shell (default 2, the 3d shell).
    correlated_block_structure : BlockStructure, optional
        Impurity block structure of the correlated shell in the symmetry-adapted basis. When given
        (with ``rotation``), degenerate PES/IPS operators of that shell are deduplicated (B2a).

    """

    # One-body transition operators must be rotated into the same basis as hOp/psis; PES/IPS are
    # bare ladder operators whose integer indices already refer to the rotated orbitals, so they
    # are left as-is and instead deduplicated via the shell's symmetry-equivalence classes.
    def _prep_one_body(tOp_dicts):
        if rotation is None:
            return [ManyBodyOperator(t) for t in tOp_dicts]
        return [_rotate_op(t, rotation) for t in tOp_dicts]

    if rotation is not None and correlated_block_structure is not None:
        correlated_groups = _pes_ips_equivalence_groups(nBaths, correlated_l, correlated_block_structure)
    else:
        correlated_groups = None

    if rank == 0:
        t0 = time.perf_counter()

    if rank == 0:
        print("Create 3d inverse photoemission and photoemission spectra...")
    # Transition operators
    tOpsIPS = inverse_photoemission_operators(nBaths, l=2)
    tOpsPS = photoemission_operators(nBaths, l=2)
    if rank == 0:
        print("Inverse photoemission Green's function..")
    assert isinstance(hOp, ManyBodyOperator)
    gsIPS = calc_spectra(
        hOp,
        [ManyBodyOperator(t) for t in tOpsIPS],
        psis,
        es,
        tau,
        w,
        basis,
        delta,
        slaterWeightMin,
        verbose,
        occ_cutoff,
        dN_imp={1: (0, 0), 2: (0, 1)},
        dN_val={1: (0, 0), 2: (1, 0)},
        dN_con={1: (0, 0), 2: (0, 1)},
        equivalence_groups=correlated_groups,
    )
    if rank == 0:
        print("Photoemission Green's function..")
    gsPS = calc_spectra(
        hOp,
        [ManyBodyOperator(t) for t in tOpsPS],
        psis,
        es,
        tau,
        -w,
        basis,
        -delta,
        slaterWeightMin,
        verbose,
        occ_cutoff,
        dN_imp={1: (0, 0), 2: (1, 0)},
        dN_val={1: (0, 0), 2: (1, 0)},
        dN_con={1: (0, 0), 2: (0, 1)},
        equivalence_groups=correlated_groups,
    )
    gsPS *= -1
    gs = gsPS + gsIPS
    if rank == 0:
        print("#spin orbitals = {:d}".format(np.shape(gs)[1]))
        print("#mesh points = {:d}".format(np.shape(gs)[0]))
    if rank == 0 and h5f:
        h5f.create_dataset("PS/spectra", data=gs)
    if rank == 0:
        print("time(PS) = {:.2f} seconds \n".format(time.perf_counter() - t0))
        t0 = time.perf_counter()

    if rank == 0:
        print("Create core 2p x-ray photoemission spectra (XPS) ...")
    # Transition operators
    tOpsPS = photoemission_operators(nBaths, l=1)
    # Photoemission Green's function
    gs = calc_spectra(
        hOp,
        [ManyBodyOperator(t) for t in tOpsPS],
        psis,
        es,
        tau,
        -w,
        basis,
        -delta,
        slaterWeightMin,
        verbose,
        occ_cutoff,
        dN_imp={1: (1, 0), 2: (1, 1)},
        dN_val={1: (0, 0), 2: (1, 0)},
        dN_con={1: (0, 0), 2: (0, 1)},
    )
    gs *= -1
    if rank == 0:
        print("#spin orbitals = {:d}".format(np.shape(gs)[1]))
        print("#mesh points = {:d}".format(np.shape(gs)[0]))
    if rank == 0 and h5f:
        h5f.create_dataset("XPS/spectra", data=gs)
    if rank == 0:
        print("time(XPS) = {:.2f} seconds \n".format(time.perf_counter() - t0))
        t0 = time.perf_counter()

    # NIXS needs the radial part of the correlated orbitals; skip it when no radial data was
    # supplied (RiNIXS is None). Every other spectrum is independent of it.
    if RiNIXS is not None:
        if rank == 0:
            print("Create NIXS spectra...")
        # Transition operator: exp(iq*r)
        tOps = nixs_operators(nBaths, qsNIXS, liNIXS, ljNIXS, RiNIXS, RjNIXS, radialMesh)
        # Green's function
        gs = calc_spectra(
            hOp,
            _prep_one_body(tOps),
            psis,
            es,
            tau,
            wLoss,
            basis,
            deltaNIXS,
            slaterWeightMin,
            verbose,
            occ_cutoff,
            dN_imp={liNIXS: (1, 1), ljNIXS: (1, 1)},
            dN_val={liNIXS: (1, 0), ljNIXS: (1, 0)},
            dN_con={liNIXS: (0, 1), ljNIXS: (0, 1)},
        )
        if rank == 0:
            print("#q-points = {:d}".format(np.shape(gs)[1]))
            print("#mesh points = {:d}".format(np.shape(gs)[0]))
        if rank == 0 and h5f:
            h5f.create_dataset("NIXS/spectra", data=gs)

        if rank == 0:
            print("time(NIXS) = {:.2f} seconds \n".format(time.perf_counter() - t0))
            t0 = time.perf_counter()

    if rank == 0:
        print("Create XAS spectra...")
    dN_XAS = dict(
        dN_imp={1: (1, 0), 2: (0, 1)},
        dN_val={1: (0, 0), 2: (1, 0)},
        dN_con={1: (0, 0), 2: (0, 1)},
    )
    if XAS_projectors:
        # Projected operators are not a plain Cartesian linear combination -> keep the
        # per-operator path.
        tOps = dipole_operators(nBaths, epsilons)
        iBasisProjectors = arrayOp2Dict(nBaths, XAS_projectors.values())
        projectedTOps = []
        for proj in iBasisProjectors:
            for op in tOps:
                projectedTOps.append(combineOp(nBaths, proj, op))
        gs = calc_spectra(
            hOp,
            _prep_one_body(projectedTOps),
            psis,
            es,
            tau,
            w,
            basis,
            delta,
            slaterWeightMin,
            verbose,
            occ_cutoff,
            **dN_XAS,
        )
        if rank == 0:
            print("#projected operators = {:d}".format(np.shape(gs)[1]))
            print("#mesh points = {:d}".format(np.shape(gs)[0]))
        if rank == 0 and h5f:
            h5f.create_dataset("XAS/projected", data=gs)
    else:
        # Dipole is linear in the polarization: compute and store the spectral tensor over the
        # 3 Cartesian components once (symmetry-reduced). Polarization contraction (arbitrary /
        # circular, dichroism, ...) is left to post-processing (impurityModel.ed.polarization),
        # so it never requires re-running the solve.
        cartesian_ops = _prep_one_body(dipole_operators(nBaths, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        n_orb = basis.num_spin_orbitals
        h_onebody = extract_tensors(hOp, n_orb=n_orb, two_body=False)[0]
        reduction = component_symmetry_reduction(cartesian_ops, h_onebody, n_orb=n_orb)
        chi = calc_spectra_tensor(
            hOp,
            cartesian_ops,
            psis,
            es,
            tau,
            w,
            basis,
            delta,
            slaterWeightMin,
            verbose,
            occ_cutoff,
            reduction=reduction,
            **dN_XAS,
        )
        if rank == 0:
            print("#Cartesian components = {:d}".format(chi.shape[1]))
            print("#mesh points = {:d}".format(chi.shape[0]))
        if rank == 0 and h5f:
            h5f.create_dataset("XAS/tensor", data=chi)
    if rank == 0:
        print("time(XAS) = {:.2f} seconds \n".format(time.perf_counter() - t0))
        t0 = time.perf_counter()

    if len(wIn) > 0:
        if rank == 0:
            print("Create RIXS spectra...")

        if RIXS_projectors:
            # Projected operators are not a plain Cartesian linear combination -> keep the
            # per-operator Kramers-Heisenberg path.
            tOpsIn = dipole_operators(nBaths, epsilonsRIXSin)
            tOpsOut = daggered_dipole_operators(nBaths, epsilonsRIXSout)
            iBasisProjectors = arrayOp2Dict(nBaths, RIXS_projectors.values())
            projectedTOpsIn = []
            projectedTOpsOut = []
            for proj in iBasisProjectors:
                for opIn in tOpsIn:
                    projectedTOpsIn.append(combineOp(nBaths, proj, opIn))
                for opOut in tOpsOut:
                    projectedTOpsOut.append(combineOp(nBaths, opOut, proj))
            gs = calc_map(
                hOp,
                _prep_one_body(projectedTOpsIn),
                _prep_one_body(projectedTOpsOut),
                psis,
                es,
                tau,
                wIn,
                wLoss,
                delta,
                deltaRIXS,
                basis,
                verbose,
                slaterWeightMin=slaterWeightMin,
            )
            if rank == 0:
                print("RIXS projectors = {}".format(RIXS_projectors.keys()))
                print(f"shape(gs) = {np.shape(gs)}")
            if rank == 0 and h5f:
                h5f.create_dataset("RIXS/projected", data=gs)
                g = h5f.create_group("RIXSprojectors")
                for key, proj in RIXS_projectors:
                    g.create_dataset(key, data=str(proj))
        else:
            # Dipole is linear in the polarization: compute and store the full rank-4
            # Kramers-Heisenberg tensor over the 3 Cartesian in/out components once (R4 -- the
            # RIXS analogue of B2b). Polarization contraction (arbitrary/circular, dichroism,
            # ...) is left to post-processing (impurityModel.ed.polarization).
            in_component_ops = _prep_one_body(dipole_operators(nBaths, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            out_component_ops = _prep_one_body(daggered_dipole_operators(nBaths, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            C = calc_tensor_map(
                hOp,
                in_component_ops,
                out_component_ops,
                psis,
                es,
                tau,
                wIn,
                wLoss,
                delta,
                deltaRIXS,
                basis,
                verbose,
                slaterWeightMin=slaterWeightMin,
            )
            if rank == 0:
                print(f"shape(C) = {np.shape(C)}")
                print("#in-components = {:d}".format(C.shape[0]))
                print("#out-components = {:d}".format(C.shape[1]))
                print("#mesh points of input energy = {:d}".format(C.shape[4]))
                print("#mesh points of energy loss = {:d}".format(C.shape[5]))
            if rank == 0 and h5f:
                # complex64: ~1e-7 relative precision, below the R1 solve tolerance (1e-6);
                # halves the storage of the default-mesh tensor (~260 -> ~130 MB).
                h5f.create_dataset("RIXS/tensor", data=C.astype(np.complex64))

        if rank == 0:
            print("time(RIXS) = {:.2f} seconds \n".format(time.perf_counter() - t0))
            t0 = time.perf_counter()

    if rank == 0 and h5f:
        h5f.close()


def _sector_restrictions_per_top(hOp, tOps, psis, basis):
    r"""Conserved-charge sector restrictions for each transition operator (or ``None``).

    Each Lanczos seed ``tOp|\psi\rangle`` lives in a definite conserved-charge sector
    (``q_ψ`` shifted by the operator's charge change). Confining the excited basis to that
    sector prunes determinants the per-shell occupation window would otherwise admit -- the
    spectra analogue of :func:`symmetries.gf_sector_restrictions` used by the self-energy.

    Returns a list aligned with ``tOps``; an entry is ``None`` when the operator has no
    definite sector (its terms disagree) so the caller falls back to the occupation window.
    Returns ``None`` (whole list) when the ground states do not share a single charge
    signature -- then no per-operator sector is well defined.
    """
    n_orb = basis.num_spin_orbitals
    comm = basis.comm
    charges = conserved_subset_charges(hOp, n_orb=n_orb)
    gs_occ = None
    for psi in psis:
        occ = measure_conserved_charges(psi, charges, n_orb, comm=comm)
        if gs_occ is None:
            gs_occ = occ
        elif occ != gs_occ:
            # Thermally-populated states span different sectors: no shared confinement.
            return None
    if gs_occ is None:
        return None
    return [transition_sector_restrictions(charges, gs_occ, tOp) for tOp in tOps]


def calc_spectra(
    hOp,
    tOps,
    psis,
    es,
    tau,
    w,
    basis,
    delta,
    slaterWeightMin,
    verbose,
    occ_cutoff,
    dN_imp,
    dN_val,
    dN_con,
    equivalence_groups=None,
    extra_meshes=None,
    seed_transform=None,
):
    """
    Calculate the Green's function spectra for a list of transition operators.

    Supports both single-process and distributed parallel calculations over MPI.

    Parameters
    ----------
    hOp : ManyBodyOperator
        The Hamiltonian operator.
    tOps : list of ManyBodyOperator
        List of transition operators.
    psis : list of ManyBodyState
        List of many-body eigenstates.
    es : list of float
        Total energies of the eigenstates.
    tau : float
        Temperature parameter for Boltzmann averaging.
    w : ndarray
        Real energy mesh points.
    basis : Basis
        The basis container.
    delta : float
        Broadening/resolution parameter (distance from the real axis).
    slaterWeightMin : float
        Minimum weight of Slater determinants to retain in basis expansion.
    verbose : bool
        If True, prints progress and diagnostic messages.
    occ_cutoff : float
        Occupation cutoff for state pruning.
    dN_imp : dict
        Restrictions on particle number change in the impurity shell.
    dN_val : dict
        Restrictions on particle number change in the valence shell.
    dN_con : dict
        Restrictions on particle number change in the conduction shell.
    equivalence_groups : list, optional
        One hashable label per transition operator. Operators sharing a label are guaranteed
        (by symmetry) to yield identical spectra, so the Lanczos is run for one representative
        per label and the result is broadcast to every member -- the B2a degeneracy dedup.
        ``None`` (default) computes every operator independently.
    extra_meshes : list of (ndarray, float), optional
        Additional ``(mesh, delta)`` pairs on which the *same* accumulated Lanczos
        coefficients are evaluated (evaluation is pointwise in ``mesh + 1j*delta``, so a
        complex mesh with ``delta=0`` gives e.g. a Matsubara evaluation) -- no extra
        Lanczos work. When given, the return value is a **list** ``[G(w, delta),
        G(mesh_1, delta_1), ...]``.
    seed_transform : callable, optional
        ``seed_transform(ei, i_op, seed) -> ManyBodyState`` applied to each enumerated
        Lanczos seed column ``tOps[i_op] |psi_ei>`` before the work-unit split (e.g. the
        susceptibility driver's projection of the seed out of the degenerate ground
        manifold). Must be linear in ``seed`` and collective-safe: it is invoked in the
        identical order on every rank, so it may perform collectives on ``basis.comm``.
        Incompatible with ``equivalence_groups``; forces the non-pairwise unit
        decomposition.

    Returns
    -------
    ndarray or list of ndarray
        A 2D array of shape `(len(w), len(tOps))` containing the complex-valued
        spectra on the real axis (a list of such arrays, one per mesh, when
        ``extra_meshes`` is given). Only returned on root process rank 0; other ranks
        return an empty array (or a list of empty arrays).
    """
    if equivalence_groups is not None:
        assert seed_transform is None, "seed_transform is incompatible with equivalence_groups"
        # Compute one representative per equivalence class, then broadcast columns. Every rank
        # takes the same reduced path (the recursion is collective), so MPI stays in lock-step.
        first_index = {}
        rep_order = []
        for i, label in enumerate(equivalence_groups):
            if label not in first_index:
                first_index[label] = i
                rep_order.append(label)
        reduced = calc_spectra(
            hOp,
            [tOps[first_index[label]] for label in rep_order],
            psis,
            es,
            tau,
            w,
            basis,
            delta,
            slaterWeightMin,
            verbose,
            occ_cutoff,
            dN_imp,
            dN_val,
            dN_con,
            equivalence_groups=None,
            extra_meshes=extra_meshes,
        )
        reduced_list = reduced if extra_meshes is not None else [reduced]
        if reduced_list[0].size == 0:  # non-root ranks return (a list of) empty arrays
            return reduced
        pos = {label: k for k, label in enumerate(rep_order)}
        full_list = []
        for reduced_mesh in reduced_list:
            full = np.empty((reduced_mesh.shape[0], len(equivalence_groups)), dtype=complex)
            for i, label in enumerate(equivalence_groups):
                full[:, i] = reduced_mesh[:, pos[label]]
            full_list.append(full)
        return full_list if extra_meshes is not None else full_list[0]

    comm = basis.comm
    # Conserved-charge sector confinement, one restriction per transition operator (computed
    # on the full communicator before any basis split, since it measures the ground states).
    sector_restrictions = _sector_restrictions_per_top(hOp, tOps, psis, basis)

    # Shared excited-sector occupation window (identical for every operator); the per-operator
    # window intersects it with that operator's charge sector -- it can only tighten, never
    # loosen. Built on the full basis before the split, so every rank holds the identical list.
    base_restrictions, weighted_restrictions = gf._build_excited_restrictions(
        basis,
        hOp,
        psis,
        es,
        None,
        occ_cutoff,
        dN_imp=dN_imp,
        dN_val=dN_val,
        dN_con=dN_con,
        slater_weight_min=slaterWeightMin,
    )
    if sector_restrictions is None:
        group_restrictions = [base_restrictions] * len(tOps)
    else:
        group_restrictions = [
            base_restrictions if sec is None else gf._intersect_restrictions(base_restrictions, sec)
            for sec in sector_restrictions
        ]

    # Flat work units = (transition operator x eigenstate chunk), distributed in ONE split with
    # the shared cost-model weights -- the same scheme as the self-energy path
    # (gf.get_Greens_function); the engine handles the serial path internally.
    op_groups = [([tOp], delta) for tOp in tOps]
    units, unit_seeds, unit_restrictions = gf.enumerate_gf_units(
        op_groups,
        psis,
        group_restrictions,
        weighted_restrictions,
        slaterWeightMin,
        # The pairwise decomposition combines seed columns, which would hide the
        # (eigenstate, operator) identity the transform needs.
        pairwise=False if seed_transform is not None else None,
    )
    if seed_transform is not None:
        # One operator per group here, so group_i identifies the transition operator and
        # the unit's seed columns are in eigenstate (chunk) order.
        for u, unit in enumerate(units):
            unit_seeds[u] = [seed_transform(ei, unit.group_i, seed) for ei, seed in zip(unit.chunk, unit_seeds[u])]
    unit_weights = gf.unit_cost_weights(unit_seeds, comm)

    def kernel(split_basis, u, seeds):
        unit = units[u]
        alphas, betas, r, _cap_stats = gf._block_green_group(
            split_basis,
            hOp,
            seeds,
            None,
            unit.delta,
            slaterWeightMin,
            True,
            verbose,
            unit_restrictions[u],
            weighted_restrictions,
        )
        if verbose:
            print(f"Expanded excited state basis contains {_cap_stats['retained_size']} elements.")
        return alphas, betas, [r[:, p * unit.n_ops : (p + 1) * unit.n_ops] for p in range(len(unit.chunk))]

    results = gf.run_units_distributed(basis, unit_seeds, unit_weights, kernel, verbose=verbose)
    if results is None:  # non-root rank of a distributed run
        empty = np.empty((0, 0), dtype=complex)
        return [empty] * (1 + len(extra_meshes)) if extra_meshes is not None else empty

    # Reassemble per-(tOp, eigenstate) coefficients (unit.group_i indexes tOps; at width 1 the
    # operator-split mode emits only diagonal units, so the same reassembly covers both modes),
    # then evaluate the thermal average on the frequency mesh -- on the root rank only, matching
    # the self-energy path and shrinking the gather payload to the Lanczos coefficients.
    acc_alphas = [[None] * len(psis) for _ in tOps]
    acc_betas = [[None] * len(psis) for _ in tOps]
    acc_r = [[None] * len(psis) for _ in tOps]
    for unit, (alphas, betas, r_slices) in zip(units, results):
        for p, ei in enumerate(unit.chunk):
            acc_alphas[unit.group_i][ei] = alphas
            acc_betas[unit.group_i][ei] = betas
            acc_r[unit.group_i][ei] = r_slices[p]

    e0 = np.min(es)
    Z = np.sum(np.exp(-(es - e0) / tau))
    meshes = [(w, delta)] + list(extra_meshes or [])
    gs_per_mesh = []
    for mesh, mesh_delta in meshes:
        gs_mesh = np.empty((len(mesh), len(tOps)), dtype=complex)
        for i in range(len(tOps)):
            G_tOp = gf.calc_thermally_averaged_G(acc_alphas[i], acc_betas[i], acc_r[i], mesh, es, e0, tau, mesh_delta)
            gs_mesh[:, i] = G_tOp[:, 0, 0] / Z
        gs_per_mesh.append(gs_mesh)
    return gs_per_mesh if extra_meshes is not None else gs_per_mesh[0]


def _combine_component_ops(component_ops, coeffs):
    r"""Linear combination :math:`\sum_\alpha c_\alpha T_\alpha` of one-body component operators."""
    return sum(
        (coeff * op for coeff, op in zip(coeffs, component_ops) if abs(coeff) != 0),
        ManyBodyOperator(),
    )


def _component_seed_moments(hOp, comp_ops, psis, es, e0, tau, basis, slaterWeightMin):
    r"""Thermal seed norms ``<seed|seed>`` and energies ``<seed|H|seed>`` per component.

    Used as the point-group-dedup safety net: symmetry predicts equal moments within a
    dedup group, so a mismatch flags an *incomplete* symmetry multiplet (an ensemble that is
    not actually symmetric) and the caller falls back to the full tensor. States are applied
    locally then redistributed onto a shared working basis so the inner products are complete
    under MPI (the apply-local -> redistribute -> local-inner -> Allreduce pattern).
    """
    m = len(comp_ops)
    comm = basis.comm
    weights = np.exp(-(np.asarray(es) - e0) / tau)
    m0 = np.zeros(m)
    m1 = np.zeros(m)
    work = basis.clone(initial_basis=[], verbose=False, comm=comm)
    for psi, wgt in zip(psis, weights):
        seeds = [applyOp_test(op, psi) for op in comp_ops]
        hseeds = [applyOp_test(hOp, s) for s in seeds]
        work.clear()
        for s in seeds + hseeds:
            work.add_states(s.keys())
        red = work.redistribute_psis(seeds + hseeds)
        for a in range(m):
            m0[a] += wgt * np.real(inner(red[a], red[a]))
            m1[a] += wgt * np.real(inner(red[a], red[m + a]))
    if comm is not None and comm.size > 1:
        comm.Allreduce(MPI.IN_PLACE, m0, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, m1, op=MPI.SUM)
    return m0, m1


def _moments_consistent(m0, m1, group_of_column, tol=1e-6):
    r"""Whether seed moments are equal within every dedup group (symmetry actually holds)."""
    groups = {}
    for a, g in enumerate(group_of_column):
        groups.setdefault(g, []).append(a)
    for members in groups.values():
        if len(members) == 1:
            continue
        ref = members[0]
        scale0 = max(abs(m0[ref]), 1.0)
        for a in members[1:]:
            if abs(m0[a] - m0[ref]) > tol * scale0:
                return False
            # Compare mean energies m1/m0 where the seed is non-trivial.
            if (
                m0[ref] > tol
                and m0[a] > tol
                and abs(m1[a] / m0[a] - m1[ref] / m0[ref]) > tol * max(abs(m1[ref] / m0[ref]), 1.0)
            ):
                return False
    return True


def calc_spectra_tensor(
    hOp,
    component_ops,
    psis,
    es,
    tau,
    w,
    basis,
    delta,
    slaterWeightMin,
    verbose,
    occ_cutoff,
    dN_imp,
    dN_val,
    dN_con,
    reduction=None,
):
    r"""One-body spectral tensor over Cartesian transition components (B2b).

    A dipole (or NIXS) transition operator is *linear* in the polarization,
    :math:`T_\varepsilon = \sum_\alpha \varepsilon_\alpha T_\alpha`, so every polarization's
    spectrum is a contraction (see :func:`impurityModel.ed.polarization.contract_spectra_tensor`)
    of a single Hermitian spectral tensor

    .. math:: \chi_{\alpha\beta}(\omega) = \langle g| T_\alpha^\dagger (\omega - H)^{-1}
              T_\beta |g\rangle .

    This function computes and returns ``chi`` itself (not a polarization contraction), so any
    number of polarizations -- including ones chosen after the fact, e.g. for dichroism -- can
    be evaluated as a cheap post-processing step instead of re-running the solve.

    The tensor is computed with **one** block-Lanczos recurrence over the (symmetry-reduced)
    component operators, confined to the conserved-charge sector of the seeds (B1).
    ``reduction`` (from :func:`symmetries.component_symmetry_reduction`) optionally collapses
    symmetry-equivalent components so the block shrinks further; a seed-moment spot-check
    guards against an incomplete symmetry multiplet and falls back to the full tensor when it
    fails.

    Parameters
    ----------
    hOp : ManyBodyOperator
        The Hamiltonian.
    component_ops : list of ManyBodyOperator
        The Cartesian component transition operators (e.g. the 3 dipole components).
    reduction : ComponentReduction, optional
        Point-group reduction of the components. ``None`` computes the full tensor.
    **kwargs
        The remaining parameters match :func:`calc_spectra`.

    Returns
    -------
    ndarray
        ``(len(w), len(component_ops), len(component_ops))`` complex spectral tensor on rank 0;
        empty array elsewhere.
    """
    m = len(component_ops)
    if reduction is None:
        reduction = ComponentReduction(np.eye(m, dtype=complex), list(range(m)), list(range(m)), m <= 1)
    Q = np.asarray(reduction.Q, dtype=complex)
    diagonalizable = reduction.diagonalizable

    # Representative component operators to actually run the block-Lanczos over.
    rep_ops = [_combine_component_ops(component_ops, Q[:, c]) for c in reduction.representatives]

    # Safety net: if the dedup would drop columns, verify the seed moments really match within
    # each group (the ensemble is a complete symmetry multiplet). Otherwise fall back to full.
    if diagonalizable and len(rep_ops) < m:
        all_rot_ops = [_combine_component_ops(component_ops, Q[:, a]) for a in range(m)]
        e0 = np.min(es)
        m0, m1 = _component_seed_moments(hOp, all_rot_ops, psis, es, e0, tau, basis, slaterWeightMin)
        if not _moments_consistent(m0, m1, reduction.group_of_column):
            diagonalizable = False
            Q = np.eye(m, dtype=complex)
            rep_ops = list(component_ops)
            reduction = reduction._replace(
                Q=Q, representatives=list(range(m)), group_of_column=list(range(m)), diagonalizable=False
            )

    # One conserved-charge sector for the whole block (all components share the charge shift).
    sector = _sector_restrictions_per_top(hOp, [rep_ops[0]], psis, basis)
    extra = None if sector is None else sector[0]

    comm = basis.comm
    e0 = np.min(es)
    Z = np.sum(np.exp(-(es - e0) / tau))
    alphas, betas, r = gf.calc_Greens_function_with_offdiag(
        hOp,
        rep_ops,
        psis,
        es,
        basis,
        delta,
        occ_cutoff=occ_cutoff,
        slaterWeightMin=slaterWeightMin,
        verbose=verbose,
        sparse=True,
        dN_imp=dN_imp,
        dN_val=dN_val,
        dN_con=dN_con,
        extra_restrictions=extra,
    )
    if comm is not None and comm.rank != 0:
        return np.empty((0, 0, 0), dtype=complex)

    chi_red = gf.calc_thermally_averaged_G(alphas, betas, r, w, es, e0, tau, delta) / Z  # (n_w, r, r)

    if diagonalizable:
        rep_diag = np.diagonal(chi_red, axis1=1, axis2=2)  # (n_w, r)
        chi_diag = rep_diag[:, reduction.group_of_column]  # (n_w, m)
        chi_full = np.einsum("wa,pa,qa->wpq", chi_diag, Q, Q.conj(), optimize=True)
    else:
        chi_full = chi_red  # full m x m tensor in the Cartesian basis (Q = I)

    return chi_full


# The RIXS drivers live in their own module; re-export them so simulate_spectra's calls and
# existing spectra.getRIXSmap_* callers resolve unchanged.
from impurityModel.ed.rixs import calc_map, calc_tensor_map  # noqa: E402
