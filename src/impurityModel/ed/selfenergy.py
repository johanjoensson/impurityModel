import argparse
import itertools
from collections import OrderedDict

import numpy as np
from mpi4py import MPI

# from impurityModel.ed.get_spectra import get_noninteracting_hamiltonian_operator
from impurityModel.ed import finite
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.greens_function import build_full_greens_function, get_Greens_function, save_Greens_function
from impurityModel.ed.groundstate import calc_gs
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.utils import matrix_print

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


class UnphysicalGreensFunctionError(Exception):
    """
    Excpetion signalling an unphysical Greens function, i.e. the imaginary part is positive for some frequencies.
    """


def fixed_peak_dc(
    h0_op,
    N0,
    mixed_valence,
    impurity_orbitals,
    bath_states,
    u4,
    peak_position,
    dc_guess,
    spin_flip_dj,
    tau,
    rank,
    verbose,
    dense_cutoff,
    slaterWeightMin,
    truncation_threshold,
):
    """
    Calculate double counting correction using a fixed peak position criterion.

    Adjusts the double counting potential to align the peak position of the
    self-energy or Green's function.

    Parameters
    ----------
    h0_op : ManyBodyOperator
        The non-interacting Hamiltonian.
    N0 : dict
        Nominal impurity occupation.
    mixed_valence : bool
        Whether system is mixed valence.
    impurity_orbitals : dict
        Impurity orbital description.
    bath_states : tuple
        Valence and conduction bath states.
    u4 : ndarray
        Coulomb interaction U matrix.
    peak_position : float
        Target peak position to adjust to.
    dc_guess : ndarray
        Initial guess for double counting.
    spin_flip_dj : bool
        Whether to allow spin flips.
    tau : float
        Temperature scale/energy broadening.
    rank : int
        MPI process rank.
    verbose : bool
        Verbosity flag.
    dense_cutoff : int
        Cutoff dimension for dense solver.
    slaterWeightMin : float
        Minimum Slater determinant weight.
    truncation_threshold : float
        Truncation threshold.

    Returns
    -------
    dc : ndarray
        Calculated double counting correction matrix.
    """
    sum(len(block) for imp_orbs in impurity_orbitals.values() for block in imp_orbs)
    peak_position = max(peak_position, 4 * tau)
    valence_baths, conduction_baths = bath_states
    u = finite.getUop_from_rspt_u4(u4)
    h_op_i = ManyBodyOperator(finite.addOps([h0_op, u]))
    dc_trial = dc_guess

    Np = {l: N0[l] + 1 for l in N0}
    Nm = {l: N0[l] - 1 for l in N0}
    if peak_position >= 0:
        basis_upper = Basis(
            impurity_orbitals,
            bath_states,
            nominal_impurity_occ=Np,
            mixed_valence=mixed_valence,
            truncation_threshold=truncation_threshold,
            verbose=verbose,
            comm=MPI.COMM_WORLD,
            spin_flip_dj=spin_flip_dj,
            tau=tau,
        )
        solver_upper = CIPSISolver(basis_upper)
        solver_upper.truncate_initial(h_op_i)

        basis_lower = Basis(
            impurity_orbitals,
            bath_states,
            nominal_impurity_occ=N0,
            mixed_valence=mixed_valence,
            truncation_threshold=truncation_threshold,
            verbose=verbose,
            comm=MPI.COMM_WORLD,
            spin_flip_dj=spin_flip_dj,
            tau=tau,
        )
        solver_lower = CIPSISolver(basis_lower)
        solver_lower.truncate_initial(h_op_i)
    else:
        basis_upper = Basis(
            impurity_orbitals,
            bath_states,
            nominal_impurity_occ=N0,
            mixed_valence=mixed_valence,
            truncation_threshold=truncation_threshold,
            verbose=verbose,
            comm=MPI.COMM_WORLD,
            spin_flip_dj=spin_flip_dj,
            tau=tau,
        )
        solver_upper = CIPSISolver(basis_upper)
        solver_upper.truncate_initial(h_op_i)

        basis_lower = Basis(
            impurity_orbitals,
            bath_states,
            nominal_impurity_occ=Nm,
            mixed_valence=mixed_valence,
            truncation_threshold=truncation_threshold,
            verbose=verbose,
            comm=MPI.COMM_WORLD,
            spin_flip_dj=spin_flip_dj,
            tau=tau,
        )
        solver_lower = CIPSISolver(basis_lower)
        solver_lower.truncate_initial(h_op_i)

    # basis_upper.restrictions = basis_upper.build_initial_restrictions(h_op_i)
    # basis_lower.restrictions = basis_lower.build_initial_restrictions(h_op_i)
    dc_op_i = ManyBodyOperator(
        {
            ((i, "c"), (j, "a")): -dc_trial[i, j] + 0j
            for i in range(dc_trial.shape[0])
            for j in range(dc_trial.shape[1])
            if abs(dc_trial[i, j]) > 0
        }
    )
    h_op = h_op_i + dc_op_i
    solver_upper.expand(h_op, dense_cutoff=dense_cutoff, de2_min=1e-3, slaterWeightMin=slaterWeightMin)
    solver_lower.expand(h_op, dense_cutoff=dense_cutoff, de2_min=1e-3, slaterWeightMin=slaterWeightMin)

    energy_cut = -tau * np.log(1e-4)

    impurity_indices = [orb for orb_blocks in impurity_orbitals.values() for block in orb_blocks for orb in block]
    np.ix_(impurity_indices, impurity_indices)

    def F(dc_fac):
        """
        Evaluate the peak shift function as a function of double counting factor.
        """
        dc = dc_fac * dc_trial
        dc_op_i = {
            ((i, "c"), (j, "a")): -dc[i, j] + 0j
            for i in range(dc_trial.shape[0])
            for j in range(dc_trial.shape[1])
            if abs(dc_trial[i, j]) > 0
        }
        h_op = h_op_i + ManyBodyOperator(dc_op_i)

        e_upper, psi_upper = solver_upper.get_eigenvectors(
            h_op,
            num_wanted=1,
            max_energy=energy_cut,
            dense_cutoff=dense_cutoff,
            slaterWeightMin=slaterWeightMin,
            solver="trlm",
        )
        e_lower, psi_lower = solver_lower.get_eigenvectors(
            h_op,
            num_wanted=1,
            max_energy=energy_cut,
            dense_cutoff=dense_cutoff,
            slaterWeightMin=slaterWeightMin,
            solver="trlm",
        )
        rho_lower = basis_lower.build_density_matrices(
            psi_lower,
            orbital_indices_left=impurity_indices,
            orbital_indices_right=impurity_indices,
        )
        rho_upper = basis_upper.build_density_matrices(
            psi_upper,
            orbital_indices_left=impurity_indices,
            orbital_indices_right=impurity_indices,
        )
        rho_lower = finite.thermal_average_scale_indep(e_lower, rho_lower, basis_lower.tau)
        rho_upper = finite.thermal_average_scale_indep(e_upper, rho_upper, basis_upper.tau)
        avg_dc_lower = np.real(np.trace(rho_lower @ dc))
        avg_dc_upper = np.real(np.trace(rho_upper @ dc))
        if abs(avg_dc_upper - avg_dc_lower) < max(tau, 1e-2):
            return 0
        return (e_upper[0] - e_lower[0] - peak_position) / (avg_dc_upper - avg_dc_lower)

    dc_fac = 1
    for _ in range(5):
        dc_fac += F(dc_fac)
    if not 0.1 < abs(dc_fac) < 2:
        dc_fac = 1
    if verbose and rank == 0:
        print(f"Fixed-peak double counting (peak position = {peak_position}):")
        matrix_print(dc_guess, label="DC guess:")
        matrix_print(dc_fac * dc_trial, label="DC found:")

    return dc_fac * dc_trial


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
    truncation_threshold : float
        Threshold for truncating the basis.
    slaterWeightMin : float
        Minimum weight for Slater determinants.
    dN : int or None
        Particle number constraint.
    sparse_green : bool
        Whether to use sparse representation for Green's function.

    Returns
    -------
    dict
        Dictionary containing self-energy, Green's function, thermal density matrix, and ground state info.
    """
    # MPI variables
    rank = comm.rank if comm is not None else 0
    # Confine this section's logging to the master rank: silencing the others keeps the
    # output readable under MPI (the verbose flags below are only forwarded to printing,
    # never to collective operations).
    if rank != 0:
        verbosity = 0

    def log(msg="", *, level=1, **kwargs):
        if verbosity >= level:
            print(msg, **kwargs)

    def banner(title, *, level=1):
        if verbosity >= level:
            print("\n" + "=" * 80)
            print(f"  {title}")
            print("=" * 80, flush=verbosity >= 2)

    # construct local, interacting, hamiltonian (in the caller's input/correlated basis B)
    u = finite.getUop_from_rspt_u4(u4)
    h_input = ManyBodyOperator(h0) + ManyBodyOperator(u)

    from impurityModel.ed.symmetries import (
        classify_bath_occupation,
        extract_tensors,
        group_orbitals_by_blocks,
        impurity_block_structure,
        impurity_symmetry_rotation,
        rotate_hamiltonian,
    )

    # Flatten the impurity orbital dict (dict[int, list[int]]) into a plain spin-orbital index
    # list; the total orbital count is inferred from the Hamiltonian (impurity + bath). The bath
    # orbitals and their valence/conduction split are derived below, not passed in.
    impurity_indices = sorted(o for orbs in impurity_orbitals.values() for o in orbs)
    n_spin_orbitals = extract_tensors(h_input, two_body=False)[0].shape[0]

    # Adaptive symmetry-adapted basis: diagonalising the impurity one-body block collapses the
    # Green's-function block structure to its finest form (e.g. 1x1 eg/t2g blocks) BUT can
    # express the Coulomb interaction more densely. h0 and u4 are in the caller's "correlated"
    # input basis (NOT assumed spherical); the fill test below is measured *relative to that
    # input basis*, so we rotate only when it does not densify the operator (fill <= threshold)
    # and keep the input basis otherwise (e.g. a j,m_j eigenbasis under spin-orbit coupling
    # densifies the Coulomb tensor). Every output is rotated back to the input basis B before
    # returning; nothing here presumes a spherical-harmonic input.
    rotation_full, u_imp = impurity_symmetry_rotation(h_input, impurity_indices, n_orb=n_spin_orbitals)
    h_rotated = rotate_hamiltonian(h_input, rotation_full, tol=_ROTATION_TRIM_TOL)
    n_terms_input = sum(1 for v in h_input.to_dict().values() if abs(v) > _ROTATION_TRIM_TOL)
    fill_ratio = len(h_rotated.to_dict()) / max(n_terms_input, 1)

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

    # Derive the bath orbitals (complement of the impurity set) and their initial occupation:
    # baths below the Fermi level (h[o, o] < 0) are valence (initially occupied), the rest are
    # conduction (initially empty). The bath one-body diagonal is unchanged by the impurity-only
    # rotation, so this is consistent whether measured in the input or solver basis.
    valence_flat, conduction_flat = classify_bath_occupation(h, impurity_indices, n_orb=n_spin_orbitals)

    # GF block structure from the hybridization-dressed impurity matrix (h[imp,imp] + V^dag V),
    # in whichever basis we solve in (fixes bath-mediated coupling; 1x1 blocks when rotated).
    # Derived from h *after* any rotation, so the blocks label the sectors of the solver basis.
    block_structure = impurity_block_structure(h, impurity_indices)

    # Group the flat orbital lists into orbital-symmetry manifolds (the inequivalent blocks and
    # their spin-degenerate partners, e.g. eg / t2g) **in the solver basis** h. Grouping by the
    # block structure keeps both spins of a manifold in one group, so the many-body basis spans
    # all S_z sectors (spin multiplets stay degenerate); the impurity occupation window is tied
    # across groups by the restriction machinery, not pinned per group.
    impurity_orbitals, bath_states = group_orbitals_by_blocks(
        h, impurity_indices, valence_flat, conduction_flat, block_structure, n_orb=n_spin_orbitals
    )
    nominal_occ = _per_group_occupation(
        nominal_occ, impurity_orbitals, extract_tensors(h, n_orb=n_spin_orbitals, two_body=False)[0]
    )
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
            verbosity >= 1,
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
    if gs_matsubara is not None:
        try:
            for gs in gs_matsubara:
                if gs is None:
                    continue
                check_greens_function(gs)
        except UnphysicalGreensFunctionError as err:
            raise UnphysicalGreensFunctionError("Matsubara interacting Greens function:\n" + str(err)) from None
    if gs_realaxis is not None:
        try:
            for gs in gs_realaxis:
                if gs is None:
                    continue
                check_greens_function(gs)
        except UnphysicalGreensFunctionError as err:
            raise UnphysicalGreensFunctionError("Real frequency interacting Greens function:\n" + str(err)) from None

    banner("Self-energy")
    log("Calculating self-energy ...")
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
            raise UnphysicalGreensFunctionError("Real frequency self-energy:\n" + str(err)) from None
    else:
        sigma_real = None
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
            raise UnphysicalGreensFunctionError("Matsubara self-energy:\n" + str(err)) from None
    else:
        sigma = None
    log("Calculating static self-energy ...")
    impurity_indices = [
        orb
        for impurity_blocks in ground_state_basis.impurity_orbitals.values()
        for block in impurity_blocks
        for orb in block
    ]
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

    return {
        "sigma": sigma_full,
        "sigma_real": sigma_real_full,
        "sigma_static": sigma_static,
        "gs_matsubara": gs_matsubara_full,
        "gs_realaxis": gs_realaxis_full,
        "thermal_rho": thermal_rho,
        "rhos": gs_info["rhos"],
        "block_structure": block_structure,
    }


def check_greens_function(G):
    """Verify that the Green's function makes physical sense.

    Raises an exception if the diagonal elements of the imaginary part are positive.

    Parameters
    ----------
    G : np.ndarray
        The Green's function matrix.

    Raises
    ------
    UnphysicalGreensFunctionError
        If the diagonal term has a positive imaginary part.
    """
    if np.any(np.diagonal(G, axis1=1, axis2=2).imag > 0):
        raise UnphysicalGreensFunctionError("Diagonal term has positive imaginary part.")


def get_hcorr_v_hbath(h0op, impurity_orbitals, sum_bath_states):
    """Extract the correlation Hamiltonian, hybridization, and bath Hamiltonian.

    The matrix form of h0op can be written as:
      [  hcorr  V^+    ]
      [  V      hbath  ]

    Parameters
    ----------
    h0op : dict or ManyBodyOperator
        The non-interacting Hamiltonian operator.
    impurity_orbitals : dict
        Dictionary of impurity orbitals.
    sum_bath_states : dict
        Dictionary of total bath states.

    Returns
    -------
    hcorr : np.ndarray
        Hamiltonian for the correlated impurity orbitals.
    v : np.ndarray
        Hopping from impurity to bath orbitals.
    v_dagger : np.ndarray
        Hopping from bath to impurity orbitals.
    h_bath : np.ndarray
        Hamiltonian for the non-interacting bath orbitals.
    """

    num_spin_orbitals = sum(impurity_orbitals[i] + sum_bath_states[i] for i in impurity_orbitals)
    n_corr = sum(ni for ni in impurity_orbitals.values())
    h0Matrix = np.zeros((num_spin_orbitals, num_spin_orbitals), dtype=complex)
    for ((i, opi), (j, opj)), val in h0op.items():
        if opi == "c" and opj == "a":
            h0Matrix[i, j] = val
        elif opj == "c" and opi == "a":
            if i == j:
                h0Matrix[i, j] = 1 - val
            else:
                h0Matrix[i, j] = -val
    hcorr = h0Matrix[0:n_corr, 0:n_corr]
    v_dagger = h0Matrix[0:n_corr, n_corr:]
    v = h0Matrix[n_corr:, 0:n_corr]
    h_bath = h0Matrix[n_corr:, n_corr:]
    return hcorr, v, v_dagger, h_bath


def hyb(ws, v, hbath, delta):
    """Calculate hybridization function from hopping parameters and bath energies.

    Δ(w) = V^dag [(w + i*delta)I - hbath]^-1 V

    Parameters
    ----------
    ws : np.ndarray
        Frequency mesh.
    v : np.ndarray
        Hopping matrix V.
    hbath : np.ndarray
        Bath Hamiltonian matrix.
    delta : float
        Smearing parameter.

    Returns
    -------
    np.ndarray
        The hybridization function.
    """
    return np.conj(v.T) @ np.linalg.solve(
        (ws + 1j * delta)[:, None, None] * np.identity(hbath.shape[0], dtype=complex)[None, :, :] - hbath[None, :, :],
        v[None, :, :],
    )


def get_sigma(
    omega_mesh,
    impurity_orbitals,
    nBaths,
    gs,
    h0op,
    delta,
    blocks,
    clustername="",
):
    """Calculate self-energy from interacting Greens function and local hamiltonian.

    Parameters
    ----------
    omega_mesh : np.ndarray
        Frequency mesh.
    impurity_orbitals : dict
        Dictionary of impurity orbitals.
    nBaths : dict
        Dictionary of total bath states.
    gs : list of np.ndarray
        List of block Green's function matrices.
    h0op : dict or ManyBodyOperator
        The non-interacting Hamiltonian operator.
    delta : float
        Smearing parameter.
    blocks : list of list of int
        List of blocks.
    clustername : str, optional
        Label for the cluster.

    Returns
    -------
    list of np.ndarray
        The self-energy matrices for each block.
    """
    hcorr, v_full, _, h_bath = get_hcorr_v_hbath(h0op, impurity_orbitals, nBaths)

    res = []
    for block, g in zip(blocks, gs):
        block_ix = np.ix_(block, block)
        wIs = (omega_mesh + 1j * delta)[:, np.newaxis, np.newaxis] * np.eye(len(block))[np.newaxis, :, :]
        g0_inv = wIs - hcorr[block_ix] - hyb(omega_mesh, v_full[:, block], h_bath, delta)
        res.append(g0_inv - np.linalg.inv(g))

    return res


def get_Sigma_static(U4, rho):
    """Calculate the static (Hartree-Fock) self-energy.

    Parameters
    ----------
    U4 : np.ndarray
        Coulomb interaction tensor.
    rho : np.ndarray
        Density matrix.

    Returns
    -------
    np.ndarray
        The static self-energy.
    """
    sigma_static = np.zeros_like(rho)
    for i, j in itertools.product(range(rho.shape[0]), range(rho.shape[1])):
        sigma_static += (U4[j, :, :, i] - U4[j, :, i, :]) * rho[i, j]

    return sigma_static


def get_selfenergy(
    clustername,
    h0_filename,
    ls,
    nBaths,
    nValBaths,
    n0imps,
    dnTols,
    dnValBaths,
    dnConBaths,
    Fdd,
    xi,
    chargeTransferCorrection,
    hField,
    nPsiMax,
    nPrintSlaterWeights,
    tau,
    energy_cut,
    delta,
    verbose,
):
    """Calculate the self energy starting from a large number of arguments.

    Parameters
    ----------
    clustername : str
        Label for the cluster.
    h0_filename : str
        Filename of the non-interacting Hamiltonian.
    ls : int
        Angular momentum of correlated orbitals.
    nBaths : int
        Total number of bath states.
    nValBaths : int
        Number of valence bath states.
    n0imps : int
        Nominal impurity occupation.
    dnTols : int
        Max deviation from nominal impurity occupation.
    dnValBaths : int
        Max number of electrons to leave valence bath orbitals.
    dnConBaths : int
        Max number of electrons to enter conduction bath orbitals.
    Fdd : list of float
        Slater-Condon parameters.
    xi : float
        Spin-orbit coupling value.
    chargeTransferCorrection : float
        Double counting parameter.
    hField : tuple of float
        Magnetic field vector (hx, hy, hz).
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
    """
    # MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.rank
    from impurityModel.ed.get_spectra import get_noninteracting_hamiltonian_operator

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
    uOp = finite.getUop(l1=ls, l2=ls, l3=ls, l4=ls, R=Fdd)
    nBaths_for_c2i = OrderedDict({ls: 0})
    for process, val in uOp.items():
        i = finite.c2i(nBaths_for_c2i, process[0][0])
        j = finite.c2i(nBaths_for_c2i, process[1][0])
        k = finite.c2i(nBaths_for_c2i, process[2][0])
        l = finite.c2i(nBaths_for_c2i, process[3][0])
        u4[i, j, k, l] = 2.0 * val

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
                new_process.append((finite.c2i(sum_baths, spinOrb), action))
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
        truncation_threshold=1000,
        slaterWeightMin=1e-12,
        dN=None,
        sparse_green=True,
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
    args = parser.parse_args()

    # Sanity checks
    assert args.nBaths >= args.nValBaths
    assert args.n0imps >= 0
    assert args.n0imps <= 2 * (2 * args.ls + 1)
    assert len(args.Fdd) == 5
    assert len(args.hField) == 3

    get_selfenergy(
        clustername=args.clustername,
        h0_filename=args.h0_filename,
        ls=(args.ls),
        nBaths=(args.nBaths),
        nValBaths=(args.nValBaths),
        n0imps=(args.n0imps),
        dnTols=(args.dnTols),
        dnValBaths=(args.dnValBaths),
        dnConBaths=(args.dnConBaths),
        Fdd=(args.Fdd),
        xi=args.xi,
        chargeTransferCorrection=args.chargeTransferCorrection,
        hField=tuple(args.hField),
        nPsiMax=args.nPsiMax,
        nPrintSlaterWeights=args.nPrintSlaterWeights,
        tau=args.tau,
        energy_cut=args.energy_cut,
        delta=args.delta,
        verbose=args.verbose,
    )
