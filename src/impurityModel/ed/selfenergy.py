import argparse
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from impurityModel.ed import atomic_physics
from impurityModel.ed.hamiltonian_io import get_noninteracting_hamiltonian_operator
from impurityModel.ed.symmetries import (
    classify_bath_occupation,
    extract_tensors,
    group_orbitals_by_blocks,
    impurity_block_structure,
    impurity_symmetry_rotation,
    rotate_hamiltonian,
)
from impurityModel.ed.operator_algebra import c2i
from impurityModel.ed.greens_function import build_full_greens_function, get_Greens_function, save_Greens_function
from impurityModel.ed.groundstate import calc_gs
from impurityModel.ed.memory_estimate import log_memory_budget, log_peak_vs_predicted, suggest_truncation_threshold
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

# The double-counting search and the self-energy extraction were split into their own modules;
# re-export their public entry points so calc_selfenergy's calls and existing
# selfenergy.<name> callers (and their test patches) resolve here unchanged.
from impurityModel.ed.double_counting import fixed_occupation_dc, fixed_peak_dc  # noqa: F401
from impurityModel.ed.sigma import (  # noqa: F401
    UnphysicalGreensFunctionError,
    check_greens_function,
    get_hcorr_v_hbath,
    get_Sigma_static,
    get_sigma,
    hyb,
)

EV_TO_RY = 1 / 13.605693122994

# Adaptive symmetry-adapted-basis rotation (calc_selfenergy): drop rotated operator terms below
# this magnitude (eV; removes rotation round-off fill), and rotate into the symmetry-adapted
# basis only if it keeps the operator term count within this factor of the input basis.
_ROTATION_TRIM_TOL = 1e-8
_MAX_ROTATION_FILL = 2.0


def _per_group_occupation(nominal_occ, impurity_orbitals, h=None):
    """Map ``nominal_occ`` onto the derived orbital-symmetry groups.

    Accepts a dict already keyed by the group indices (used as-is), or any other dict / a
    scalar interpreted as the *total* impurity occupation. When the one-body Hamiltonian ``h``
    is supplied, the total is distributed by **energetic filling** — the lowest on-site-energy
    impurity spin-orbitals (``h[o, o]``) are occupied first — so e.g. a cubic d-shell fills the
    lower ``t2g`` manifold before ``eg`` (giving ``t2g=6``, ``eg=2`` for ``d8``) and the split
    is spin-symmetric. Without ``h`` it falls back to a size-proportional split (remainder to
    the largest groups). The prescan refines the per-group split, so this only needs to be a
    sensible starting point.
    """
    keys = list(impurity_orbitals)
    if isinstance(nominal_occ, dict) and set(nominal_occ) == set(keys):
        return {k: int(nominal_occ[k]) for k in keys}
    total = int(sum(nominal_occ.values()) if isinstance(nominal_occ, dict) else nominal_occ)

    if h is not None:
        # Energetic filling: occupy the lowest on-site-energy impurity spin-orbitals first and
        # count how many land in each group. Ties broken by orbital index for determinism.
        orb_to_group = {orb: k for k in keys for block in impurity_orbitals[k] for orb in block}
        ordered = sorted(orb_to_group, key=lambda o: (np.real(h[o, o]), o))
        alloc = {k: 0 for k in keys}
        for orb in ordered[: max(0, min(total, len(ordered)))]:
            alloc[orb_to_group[orb]] += 1
        return alloc

    sizes = {k: sum(len(block) for block in impurity_orbitals[k]) for k in keys}
    tot_size = sum(sizes.values()) or 1
    alloc = {k: int(total * sizes[k] // tot_size) for k in keys}
    remainder = total - sum(alloc.values())
    for k in sorted(keys, key=lambda k: sizes[k], reverse=True):
        if remainder <= 0:
            break
        alloc[k] += 1
        remainder -= 1
    return alloc


def _per_group_scalar(value, impurity_orbitals, default=0):
    """Map a per-group scalar setting (e.g. mixed_valence) onto the derived group keys."""
    keys = list(impurity_orbitals)
    if isinstance(value, dict) and set(value) == set(keys):
        return dict(value)
    return {k: default for k in keys}


def _raise_together(comm, message):
    """Turn a rank-local validation verdict into a *collective* raise.

    .. warning:: **Collective on** ``comm`` (broadcast of the verdict from rank 0). Call it
       unconditionally on every rank, outside the ``if gs is not None`` guard.

    ``get_Greens_function`` gathers its results to global rank 0, so ``gs_matsubara`` /
    ``gs_realaxis`` (and the self-energies built from them) are ``None`` on every other rank.
    The physicality checks therefore run on rank 0 alone. Raising there unwinds rank 0 out of
    :func:`calc_selfenergy` and into ``MPI_Finalize`` while the remaining ranks walk on to the
    next collective -- ``log_peak_vs_predicted``'s ``Allreduce`` -- and block there forever.
    An unphysical Green's function then presents as a *hang* rather than an error.

    Broadcasting the verdict makes every rank raise the same exception at the same point.

    Parameters
    ----------
    comm : MPI communicator or None
    message : str or None
        The failure message on rank 0; ``None`` on the other ranks and when the check passed.
        Only rank 0's value is used.
    """
    if comm is not None:
        message = comm.bcast(message, root=0)
    if message is not None:
        raise UnphysicalGreensFunctionError(message)


def calc_selfenergy(
    h0,
    u4,
    iw,
    w,
    delta,
    nominal_occ,
    mixed_valence,
    impurity_orbitals,
    tau,
    verbosity,
    rot_to_spherical,
    cluster_label,
    reort,
    dense_cutoff,
    spin_flip_dj,
    comm,
    chain_restrict,
    occ_cutoff,
    truncation_threshold,
    slaterWeightMin,
    dN,
    sparse_green,
    gf_method="lanczos",
):
    """Calculate the self energy of the impurity.

    Parameters
    ----------
    h0 : dict or ManyBodyOperator
        The non-interacting Hamiltonian.
    u4 : np.ndarray
        The Coulomb interaction matrix.
    iw : np.ndarray or None
        Matsubara frequency mesh.
    w : np.ndarray or None
        Real frequency mesh.
    delta : float
        Smearing parameter for real frequencies.
    nominal_occ : dict
        Nominal occupation.
    mixed_valence : bool
        Whether to consider mixed valence.
    impurity_orbitals : dict[int, list[int]]
        Flat impurity spin-orbital index lists per group; re-grouped into conserved-charge blocks
        internally by :func:`symmetries.group_orbitals_by_charges`. The bath orbitals (everything
        else in ``h0``) and their valence/conduction (occupied/empty) split are derived from the
        Hamiltonian via :func:`symmetries.classify_bath_occupation`.
    tau : float
        Temperature parameter.
    verbosity : int
        Verbosity level.
    rot_to_spherical : np.ndarray
        Rotation matrix to spherical harmonics.
    cluster_label : str
        Label for the cluster.
    reort : float or None
        Reorthogonalization parameter.
    dense_cutoff : int
        Cutoff for dense matrix representation.
    spin_flip_dj : bool
        Whether to include spin-flip terms.
    comm : MPI.Comm or None
        MPI communicator.
    chain_restrict : bool
        Whether to restrict to chain geometry.
    occ_cutoff : float
        Cutoff for occupation numbers.
    truncation_threshold : float or None
        Global cap on the number of Slater determinants per basis (ground state and each
        Green's-function excited basis). ``None`` derives the cap from available per-rank
        memory (collective probe; see :mod:`impurityModel.ed.memory_estimate`), ``np.inf``
        disables capping.
    slaterWeightMin : float
        Minimum weight for Slater determinants.
    dN : int or None
        Particle number constraint.
    sparse_green : bool
        Whether to use sparse representation for Green's function.
    gf_method : str
        Green's-function kernel: ``"lanczos"`` (default, one block-Lanczos recurrence per
        work unit serving the whole mesh), ``"bicgstab"`` (one linear solve per frequency
        point on a rebuilt-and-discarded basis -- the memory-first path; ``sparse_green``
        and ``reort`` do not apply to it), or ``"sliced"`` (Chebyshev spectral-window
        decomposition with per-slice bases; real-axis meshes only). See
        :func:`impurityModel.ed.greens_function.get_Greens_function`.

    Returns
    -------
    dict
        Dictionary containing self-energy, Green's function, thermal density matrix, and ground state info.
    """
    # MPI variables
    rank = comm.rank if comm is not None else 0

    def log(msg="", *, level=1, **kwargs):
        if verbosity >= level:
            print(msg, **kwargs)

    def banner(title, *, level=1):
        if verbosity >= level:
            print("\n" + "=" * 80)
            print(f"  {title}")
            print("=" * 80, flush=verbosity >= 2)

    # construct local, interacting, hamiltonian (in the caller's input/correlated basis B)
    u = atomic_physics.getUop_from_rspt_u4(u4)
    h_input = ManyBodyOperator(h0) + ManyBodyOperator(u)

    # Flatten the impurity orbital dict (dict[int, list[int]]) into a plain spin-orbital index
    # list; the total orbital count is inferred from the Hamiltonian (impurity + bath). The bath
    # orbitals and their valence/conduction split are derived below, not passed in.
    impurity_indices = sorted(o for orbs in impurity_orbitals.values() for o in orbs)
    h_input_matrix = extract_tensors(h_input, two_body=False)[0]
    n_spin_orbitals = h_input_matrix.shape[0]

    # Adaptive symmetry-adapted basis: diagonalising the impurity one-body block collapses the
    # Green's-function block structure to its finest form (e.g. 1x1 eg/t2g blocks) BUT can
    # express the Coulomb interaction more densely. h0 and u4 are in the caller's "correlated"
    # input basis (NOT assumed spherical); the fill test below is measured *relative to that
    # input basis*, so we rotate only when it does not densify the operator (fill <= threshold)
    # and keep the input basis otherwise (e.g. a j,m_j eigenbasis under spin-orbit coupling
    # densifies the Coulomb tensor). Every output is rotated back to the input basis B before
    # returning; nothing here presumes a spherical-harmonic input.
    rotation_full, u_imp = impurity_symmetry_rotation(
        h_input, impurity_indices, n_orb=n_spin_orbitals, h0_matrix=h_input_matrix
    )
    h_rotated = rotate_hamiltonian(h_input, rotation_full, tol=_ROTATION_TRIM_TOL)
    n_terms_input = sum(1 for v in h_input.values() if abs(v) > _ROTATION_TRIM_TOL)
    fill_ratio = len(h_rotated) / max(n_terms_input, 1)

    rotate = fill_ratio <= _MAX_ROTATION_FILL
    if rotate:
        h = h_rotated
        h0_solve = rotate_hamiltonian(ManyBodyOperator(h0), rotation_full, tol=_ROTATION_TRIM_TOL).to_dict()
        # Observable rotation for the solve (spherical -> S): compose the caller's input rotation
        # R_in (spherical -> B) with W^dag (B -> S). On the impurity block, R = u_imp^dag @ R_in.
        rot_to_spherical = u_imp.conj().T @ np.asarray(rot_to_spherical, dtype=complex)
    else:
        # Stay in the input basis; make the output rotation below a no-op.
        h = h_input
        h0_solve = h0
        rotation_full = np.eye(n_spin_orbitals, dtype=complex)
        u_imp = np.eye(len(impurity_indices), dtype=complex)

    # One-body matrix of the solver-basis Hamiltonian, extracted once and shared by the
    # classification/grouping helpers below (each would otherwise re-walk the full operator
    # and allocate its own dense n_orb x n_orb copy).
    h_matrix = extract_tensors(h, n_orb=n_spin_orbitals, two_body=False)[0] if rotate else h_input_matrix

    # Derive the bath orbitals (complement of the impurity set) and their initial occupation:
    # baths below the Fermi level (h[o, o] < 0) are valence (initially occupied), the rest are
    # conduction (initially empty). The bath one-body diagonal is unchanged by the impurity-only
    # rotation, so this is consistent whether measured in the input or solver basis.
    valence_flat, conduction_flat = classify_bath_occupation(
        h, impurity_indices, n_orb=n_spin_orbitals, h0_matrix=h_matrix
    )

    # GF block structure from the hybridization-dressed impurity matrix (h[imp,imp] + V^dag V),
    # in whichever basis we solve in (fixes bath-mediated coupling; 1x1 blocks when rotated).
    # Derived from h *after* any rotation, so the blocks label the sectors of the solver basis.
    block_structure = impurity_block_structure(h, impurity_indices, h0_matrix=h_matrix)

    # Group the flat orbital lists into orbital-symmetry manifolds (the inequivalent blocks and
    # their spin-degenerate partners, e.g. eg / t2g) **in the solver basis** h. Grouping by the
    # block structure keeps both spins of a manifold in one group, so the many-body basis spans
    # all S_z sectors (spin multiplets stay degenerate); the impurity occupation window is tied
    # across groups by the restriction machinery, not pinned per group.
    impurity_orbitals, bath_states = group_orbitals_by_blocks(
        h, impurity_indices, valence_flat, conduction_flat, block_structure, n_orb=n_spin_orbitals, h0_matrix=h_matrix
    )
    nominal_occ = _per_group_occupation(nominal_occ, impurity_orbitals, h_matrix)
    mixed_valence = _per_group_scalar(mixed_valence, impurity_orbitals, default=0)

    valence_baths, conduction_baths = bath_states
    total_impurity_orbitals = {i: sum(len(orbs) for orbs in impurity_orbitals[i]) for i in impurity_orbitals}
    sum_bath_states = {
        i: sum(len(orbs) for orbs in valence_baths[i]) + sum(len(orbs) for orbs in conduction_baths[i])
        for i in valence_baths
    }

    if verbosity > 0:
        basis_note = f"symmetry-adapted (fill {fill_ratio:.1f}x)" if rotate else f"input basis (fill {fill_ratio:.1f}x)"
        print(f"Block structure: {len(block_structure.blocks)} blocks, solving in {basis_note}")
    # Resolve the basis cap: None means "as many determinants as fit in RAM". Both the
    # suggestion and the budget log are collective on comm (memory probe + allreduce), so
    # they run unconditionally on every rank; only the printing is verbosity-gated.
    gf_block_width = max(4, *(len(block) for block in block_structure.blocks))
    if truncation_threshold is None:
        truncation_threshold = suggest_truncation_threshold(
            n_spin_orbitals, comm=comm, block_width=gf_block_width, reort=reort, method=gf_method
        )
    memory_budget = log_memory_budget(
        truncation_threshold,
        n_spin_orbitals,
        comm=comm,
        block_width=gf_block_width,
        reort=reort,
        verbose=verbosity > 0,
        label=cluster_label,
        method=gf_method,
    )
    basis_information = {
        "impurity_orbitals": impurity_orbitals,
        "bath_states": bath_states,
        "N0": nominal_occ,
        "mixed_valence": mixed_valence,
        "tau": tau,
        "chain_restrict": chain_restrict,
        "dense_cutoff": dense_cutoff,
        "spin_flip_dj": spin_flip_dj,
        "rank": rank,
        "comm": comm,
        "truncation_threshold": truncation_threshold,
    }
    # Compute the thermal ground state and the interacting Green's function, with a single
    # auto-retry: the diagnostics report (gf_diagnostics) can detect that the thermal
    # ensemble was truncated (the highest retained state still carries Boltzmann weight); if
    # so we re-run the eigensolver with more requested states (num_wanted) once.
    num_wanted = 10
    max_retries = 2
    for _attempt in range(max_retries + 1):
        psis, es, ground_state_basis, thermal_rho, gs_info = calc_gs(
            h,
            basis_information,
            block_structure,
            rot_to_spherical,
            verbosity >= 2,
            slaterWeightMin=slaterWeightMin,
            num_wanted=num_wanted,
        )
        restrictions = ground_state_basis.restrictions

        if restrictions is not None:
            log("Restrictions on ground-state occupation:", level=2)
            for indices, limits in restrictions.items():
                log(f"  {sorted(indices)} : {limits}", level=2)

        banner("Interacting Green's function")
        log(f"Considering {len(es)} eigenstate(s) for the spectra.")
        log("Calculating interacting Green's function ...", flush=verbosity >= 2)

        gs_matsubara, gs_realaxis, gf_report = get_Greens_function(
            matsubara_mesh=iw,
            omega_mesh=w,
            psis=psis,
            es=es,
            tau=tau,
            basis=ground_state_basis,
            hOp=h,
            delta=delta,
            blocks=[block_structure.blocks[block_i] for block_i in block_structure.inequivalent_blocks],
            verbose=verbosity >= 1,
            verbose_extra=verbosity >= 2,
            reort=reort,
            dN=dN,
            occ_cutoff=occ_cutoff,
            slaterWeightMin=slaterWeightMin,
            sparse=sparse_green,
            num_wanted=num_wanted,
            gf_method=gf_method,
        )

        # Root rank renders the diagnostics report and decides whether to retry; the decision
        # is broadcast so every rank re-enters calc_gs collectively (or all break).
        retry = False
        if rank == 0 and gf_report is not None:
            log(gf_report.render())
            retry = gf_report.needs_more_states and _attempt < max_retries
        if comm is not None:
            retry = comm.bcast(retry, root=0)
        if not retry:
            break
        num_wanted *= 2
        log(f"\nThermal ensemble appears truncated; retrying with num_wanted = {num_wanted}.\n", flush=True)
    # The four physicality checks below only have data on rank 0; `_raise_together` broadcasts
    # each verdict so every rank raises (or continues) as one. See its docstring: raising on
    # rank 0 alone deadlocks the survivors in the next collective.
    message = None
    if gs_matsubara is not None:
        try:
            for gs in gs_matsubara:
                if gs is None:
                    continue
                check_greens_function(gs)
        except UnphysicalGreensFunctionError as err:
            message = "Matsubara interacting Greens function:\n" + str(err)
    _raise_together(comm, message)

    message = None
    if gs_realaxis is not None:
        try:
            for gs in gs_realaxis:
                if gs is None:
                    continue
                check_greens_function(gs)
        except UnphysicalGreensFunctionError as err:
            message = "Real frequency interacting Greens function:\n" + str(err)
    _raise_together(comm, message)

    banner("Self-energy")
    log("Calculating self-energy ...")
    message = None
    if gs_realaxis is not None:
        sigma_real = get_sigma(
            omega_mesh=w,
            impurity_orbitals=total_impurity_orbitals,
            nBaths=sum_bath_states,
            gs=gs_realaxis,
            h0op=h0_solve,
            delta=delta,
            clustername=cluster_label,
            blocks=[block_structure.blocks[block_i] for block_i in block_structure.inequivalent_blocks],
        )
        try:
            for sig in sigma_real:
                check_greens_function(sig)
        except UnphysicalGreensFunctionError as err:
            for i, sig in enumerate(sigma_real):
                save_Greens_function(sig, w, f"sig+dc-{i}", cluster_label)
            message = "Real frequency self-energy:\n" + str(err)
    else:
        sigma_real = None
    _raise_together(comm, message)

    message = None
    if gs_matsubara is not None:
        sigma = get_sigma(
            omega_mesh=iw,
            impurity_orbitals=total_impurity_orbitals,
            nBaths=sum_bath_states,
            gs=gs_matsubara,
            h0op=h0_solve,
            delta=0,
            clustername=cluster_label,
            blocks=[block_structure.blocks[block_i] for block_i in block_structure.inequivalent_blocks],
        )
        try:
            for sig in sigma:
                check_greens_function(sig)
        except UnphysicalGreensFunctionError as err:
            for i, sig in enumerate(sigma):
                save_Greens_function(sig, iw, f"sig+dc-{i}", cluster_label)
            message = "Matsubara self-energy:\n" + str(err)
    else:
        sigma = None
    _raise_together(comm, message)

    log("Calculating static self-energy ...")
    # Sort the flattened indices: the groups enumerate the impurity orbitals in
    # block order (e.g. eg [0,1,5,6] before t2g [2,3,4,7,8,9]), but thermal_rho
    # and u4 below are in the input-basis orbital order. Indexing rho with the
    # unsorted list permutes it against u4 and yields a wrong static self-energy
    # (Sigma_static would no longer equal the iw -> inf limit of Sigma).
    impurity_indices = sorted(
        orb
        for impurity_blocks in ground_state_basis.impurity_orbitals.values()
        for block in impurity_blocks
        for orb in block
    )
    impurity_ix = np.ix_(impurity_indices, impurity_indices)

    # Rotate every result from the solver basis S back to the caller's input basis B
    # (O_B = W O_S W^dag; impurity block u_imp). When the adaptive test kept the input basis,
    # W and u_imp are identity and these are no-ops. The density matrix is full-space (rotate
    # with W); the self-energies / Green's functions are impurity-only (rotate with u_imp).
    thermal_rho = rotation_full @ thermal_rho @ rotation_full.conj().T

    def _to_input_basis(block_list):
        """Reassemble per-inequivalent-block matrices (basis S) and rotate to input basis B."""
        if block_list is None:
            return None
        full_s = build_full_greens_function(block_list, block_structure)
        if full_s.ndim == 3:  # (n_omega, n_imp, n_imp)
            return np.einsum("ij,wjk,lk->wil", u_imp, full_s, u_imp.conj())
        return u_imp @ full_s @ u_imp.conj().T

    sigma_full = _to_input_basis(sigma)
    sigma_real_full = _to_input_basis(sigma_real)
    gs_matsubara_full = _to_input_basis(gs_matsubara)
    gs_realaxis_full = _to_input_basis(gs_realaxis)

    # Static (Hartree-Fock) self-energy from the input-basis density matrix and u4 (input basis).
    sigma_static = get_Sigma_static(u4, thermal_rho[impurity_ix])

    # Predicted-vs-measured peak feedback for re-calibrating the byte model on
    # production-size runs (doc/plans/truncation_reliability.md). Collective on comm,
    # so it runs unconditionally; only the printing is verbosity-gated.
    log_peak_vs_predicted(memory_budget, comm=comm, verbose=verbosity > 0, label=cluster_label)

    return {
        "sigma": sigma_full,
        "sigma_real": sigma_real_full,
        "sigma_static": sigma_static,
        "gs_matsubara": gs_matsubara_full,
        "gs_realaxis": gs_realaxis_full,
        "thermal_rho": thermal_rho,
        "rhos": gs_info["rhos"],
        "gs_energies": np.asarray(es),
        "block_structure": block_structure,
        # None unless the truncation_threshold bound the ground-state basis; a dict with
        # the fixed-budget CIPSI refinement summary otherwise (see CIPSISolver.expand).
        "gs_truncation": gs_info.get("truncation"),
    }


@dataclass(frozen=True)
class HamiltonianParameters:
    """Non-interacting Hamiltonian file plus the atomic interaction parameters.

    Attributes
    ----------
    h0_filename : str
        Filename of the non-interacting Hamiltonian.
    ls : int
        Angular momentum of correlated orbitals.
    nBaths, nValBaths : int
        Total number of bath states / number of valence bath states.
    Fdd : list of float
        Slater-Condon parameters.
    xi : float
        Spin-orbit coupling value.
    chargeTransferCorrection : float
        Double counting parameter.
    hField : tuple of float
        Magnetic field vector (hx, hy, hz).
    """

    h0_filename: str
    ls: int
    nBaths: int
    nValBaths: int
    Fdd: object
    xi: float
    chargeTransferCorrection: float
    hField: tuple


@dataclass(frozen=True)
class OccupationParameters:
    """Nominal impurity occupation and the allowed deviations.

    Attributes
    ----------
    n0imps : int
        Nominal impurity occupation.
    dnTols : int
        Max deviation from nominal impurity occupation.
    dnValBaths : int
        Max number of electrons to leave valence bath orbitals.
    dnConBaths : int
        Max number of electrons to enter conduction bath orbitals.
    """

    n0imps: int
    dnTols: int
    dnValBaths: int
    dnConBaths: int


@dataclass(frozen=True)
class SolverParameters:
    """Output label, eigenstate budget, temperature/smearing, and the GF kernel.

    Attributes
    ----------
    clustername : str
        Label for the cluster.
    nPsiMax : int
        Maximum number of eigenstates to consider.
    nPrintSlaterWeights : int
        Printing parameter for Slater weights.
    tau : float
        Fundamental temperature.
    energy_cut : float
        Energy cutoff for eigenstates.
    delta : float
        Smearing parameter.
    verbose : bool
        Verbosity flag.
    gf_method : str
        Green's-function kernel, ``"lanczos"`` (default), ``"bicgstab"`` (per-frequency
        linear solves, memory-first), or ``"sliced"`` (Chebyshev spectral-window
        decomposition, real-axis meshes only). See :func:`calc_selfenergy`.
    """

    clustername: str
    nPsiMax: int
    nPrintSlaterWeights: int
    tau: float
    energy_cut: float
    delta: float
    verbose: bool
    gf_method: str = "lanczos"


def get_selfenergy(
    hamiltonian: HamiltonianParameters,
    occupation: OccupationParameters,
    solver: SolverParameters,
):
    """Calculate the self energy from the grouped CLI parameters.

    Parameters
    ----------
    hamiltonian : HamiltonianParameters
        Non-interacting Hamiltonian file and the atomic interaction parameters.
    occupation : OccupationParameters
        Nominal impurity occupation and its allowed deviations.
    solver : SolverParameters
        Output label, eigenstate budget, temperature/smearing, and the GF kernel.
    """
    # Unpack the grouped parameters into the local names used below.
    h0_filename = hamiltonian.h0_filename
    ls = hamiltonian.ls
    nBaths = hamiltonian.nBaths
    Fdd = hamiltonian.Fdd
    xi = hamiltonian.xi
    hField = hamiltonian.hField
    n0imps = occupation.n0imps
    clustername = solver.clustername
    tau = solver.tau
    delta = solver.delta
    verbose = solver.verbose
    gf_method = solver.gf_method

    # MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # omega_mesh = np.linspace(-25, 25, 2000)
    omega_mesh = np.linspace(-1.83, 1.83, 2000)
    # omega_mesh = 1j*np.pi*tau*np.arange(start = 1, step = 2, stop = 2*375)

    # if rank == 0:
    #     t0 = time.perf_counter()
    # -- System information --

    sum_baths = OrderedDict({ls: nBaths})

    # -- Basis occupation information --
    nominal_occ = {ls: n0imps}

    # Construct u4 and rot_to_spherical, mixed_valence, etc.
    n_imp_spin_orbitals = 2 * (2 * ls + 1)
    u4 = np.zeros((n_imp_spin_orbitals, n_imp_spin_orbitals, n_imp_spin_orbitals, n_imp_spin_orbitals), dtype=complex)
    uOp = atomic_physics.getUop(l1=ls, l2=ls, l3=ls, l4=ls, R=Fdd)
    nBaths_for_c2i = OrderedDict({ls: 0})
    for process, val in uOp.items():
        i = c2i(nBaths_for_c2i, process[0][0])
        j = c2i(nBaths_for_c2i, process[1][0])
        k = c2i(nBaths_for_c2i, process[2][0])
        l = c2i(nBaths_for_c2i, process[3][0])
        # RSPt convention: u4[i,j,k,l] multiplies c^dag_i c^dag_j c_l c_k, so
        # the process c^dag_i c^dag_j c_k c_l is stored with k and l swapped.
        u4[i, j, l, k] = 2.0 * val

    # Flat impurity spin-orbital index list (dict[int, list[int]]); calc_selfenergy re-groups the
    # orbitals into per-conserved-charge blocks, derives the bath orbitals + their valence/
    # conduction split from the Hamiltonian, and derives the block structure internally.
    impurity_orbitals = {ls: list(range(n_imp_spin_orbitals))}
    mixed_valence = {ls: 0}

    rot_to_spherical = np.eye(n_imp_spin_orbitals, dtype=complex)

    # Hamiltonian
    if rank == 0 and verbose:
        print("Constructing the Hamiltonian operator ...")
    hOp = get_noninteracting_hamiltonian_operator(
        sum_baths,
        [0, xi],
        hField,
        h0_filename,
        rank,
        verbose,
    )
    # Convert spin-orbital and bath state indices to a single index notation.
    hOp_new = {}
    for process, value in hOp.items():
        new_process = []
        for spinOrb, action in process:
            try:
                new_process.append((c2i(sum_baths, spinOrb), action))
            except Exception as e:
                print(f"FAILED on spinOrb: {spinOrb} in process {process}", flush=True)
                raise e
        hOp_new[tuple(new_process)] = value
    hOp = hOp_new

    # calc_selfenergy returns a result dict, not a tuple. Keys: "sigma"/"sigma_real" and
    # "gs_matsubara"/"gs_realaxis" (full (n_omega, n_imp, n_imp) matrices rotated back to the
    # caller's input basis, or None), "sigma_static", "thermal_rho", "rhos", "block_structure".
    result = calc_selfenergy(
        h0=hOp,
        u4=u4,
        iw=None,
        w=omega_mesh,
        delta=delta,
        nominal_occ=nominal_occ,
        mixed_valence=mixed_valence,
        impurity_orbitals=impurity_orbitals,
        tau=tau,
        verbosity=2 if verbose else 0,
        rot_to_spherical=rot_to_spherical,
        cluster_label=clustername,
        reort=None,
        dense_cutoff=500,
        spin_flip_dj=False,
        comm=comm,
        chain_restrict=False,
        occ_cutoff=1e-12,
        truncation_threshold=None,
        slaterWeightMin=1e-12,
        dN=None,
        sparse_green=True,
        gf_method=gf_method,
    )

    if rank == 0 and verbose:
        print(f"Self-energy computed for cluster '{clustername}'.")
    # To persist the results, save the relevant entries of `result`, e.g.:
    #     if rank == 0:
    #         np.savetxt(f"real-sig_static-{clustername}.dat", np.real(result["sigma_static"]))
    #         np.savetxt(f"imag-sig_static-{clustername}.dat", np.imag(result["sigma_static"]))
    #         save_Greens_function(
    #             gs=result["sigma_real"], omega_mesh=omega_mesh, label=f"Sigma-{clustername}", e_scale=1
    #         )
    return result


if __name__ == "__main__":
    # Parse input parameters
    parser = argparse.ArgumentParser(description="Calculate selfenergy")
    parser.add_argument(
        "h0_filename",
        type=str,
        help="Filename of non-interacting Hamiltonian.",
    )
    parser.add_argument(
        "--clustername",
        type=str,
        default="cluster",
        help="Id of cluster, used for generating the filename in which to store the calculated self-energy.",
    )
    parser.add_argument(
        "--ls",
        type=int,
        default=2,
        help="Angular momenta of correlated orbitals.",
    )
    parser.add_argument(
        "--nBaths",
        type=int,
        default=10,
        help="Total number of bath states, for the correlated orbitals.",
    )
    parser.add_argument(
        "--nValBaths",
        type=int,
        default=10,
        help="Number of valence bath states for the correlated orbitals.",
    )
    parser.add_argument(
        "--n0imps",
        type=int,
        default=8,
        help="Nominal impurity occupation.",
    )
    parser.add_argument(
        "--dnTols",
        type=int,
        default=2,
        help=("Max devation from nominal impurity occupation."),
    )
    parser.add_argument(
        "--dnValBaths",
        type=int,
        default=2,
        help=("Max number of electrons to leave valence bath orbitals."),
    )
    parser.add_argument(
        "--dnConBaths",
        type=int,
        default=0,
        help=("Max number of electrons to enter conduction bath orbitals."),
    )
    parser.add_argument(
        "--Fdd",
        type=float,
        nargs="+",
        default=[7.5, 0, 9.9, 0, 6.6],
        help="Slater-Condon parameters Fdd. d-orbitals are assumed.",
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=0,
        help="SOC value for valence orbitals. Assumed to be d-orbitals",
    )
    parser.add_argument(
        "--chargeTransferCorrection",
        type=float,
        default=None,
        help="Double counting parameter.",
    )
    parser.add_argument(
        "--hField",
        type=float,
        nargs="+",
        default=[0, 0, 0.0001],
        help="Magnetic field. (h_x, h_y, h_z)",
    )
    parser.add_argument(
        "--nPsiMax",
        type=int,
        default=5,
        help="Maximum number of eigenstates to consider.",
    )
    parser.add_argument("--nPrintSlaterWeights", type=int, default=3, help="Printing parameter.")
    parser.add_argument("--tau", type=float, default=0.002, help="Fundamental temperature (kb*T).")
    parser.add_argument(
        "--energy_cut",
        type=float,
        default=10,
        help="How many k_B*T above lowest eigenenergy to consider.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.2,
        help=("Smearing, half width half maximum (HWHM). Due to short core-hole lifetime."),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=("Set verbose output (very loud...)"),
    )
    parser.add_argument(
        "--gf_method",
        type=str,
        choices=("lanczos", "bicgstab", "sliced"),
        default="lanczos",
        help=(
            "Green's-function kernel: 'lanczos' runs one block-Lanczos recurrence per work unit "
            "for the whole mesh; 'bicgstab' solves one linear system per frequency point on a "
            "rebuilt-and-discarded basis; 'sliced' adds Chebyshev spectral-window decomposition "
            "with per-slice bases (real-axis meshes; see GF_SLICES/GF_SLICE_TOL)."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    assert args.nBaths >= args.nValBaths
    assert args.n0imps >= 0
    assert args.n0imps <= 2 * (2 * args.ls + 1)
    assert len(args.Fdd) == 5
    assert len(args.hField) == 3

    get_selfenergy(
        hamiltonian=HamiltonianParameters(
            h0_filename=args.h0_filename,
            ls=(args.ls),
            nBaths=(args.nBaths),
            nValBaths=(args.nValBaths),
            Fdd=(args.Fdd),
            xi=args.xi,
            chargeTransferCorrection=args.chargeTransferCorrection,
            hField=tuple(args.hField),
        ),
        occupation=OccupationParameters(
            n0imps=(args.n0imps),
            dnTols=(args.dnTols),
            dnValBaths=(args.dnValBaths),
            dnConBaths=(args.dnConBaths),
        ),
        solver=SolverParameters(
            clustername=args.clustername,
            nPsiMax=args.nPsiMax,
            nPrintSlaterWeights=args.nPrintSlaterWeights,
            tau=args.tau,
            energy_cut=args.energy_cut,
            delta=args.delta,
            verbose=args.verbose,
            gf_method=args.gf_method,
        ),
    )
