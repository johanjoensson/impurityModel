"""
This module contains functions for calculating various spectra.
"""

from typing import Optional

import numpy as np
from mpi4py import MPI

# Local imports
import impurityModel.ed.greens_function as gf
from impurityModel.ed.average import ThermalEnsemble
from impurityModel.ed.basis_restrictions import intersect_sector_window
from impurityModel.ed.basis_transcription import comm_support_filter
from impurityModel.ed.gf_convergence import _gf_eval_meshes
from impurityModel.ed.gf_engine import lanczos_unit_kernel, states_by_group
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, inner
from impurityModel.ed.ManyBodyUtils import applyOp as applyOp_test
from impurityModel.ed.operator_algebra import c2i
from impurityModel.ed.options import DEFAULT_GF_PRUNE_TOL

# The RIXS drivers live in their own module; re-exported here for callers.
from impurityModel.ed.rixs import calc_map, calc_tensor_map  # noqa: F401
from impurityModel.ed.symmetries import (
    ComponentReduction,
    conserved_subset_charges,
    definite_conserved_charges,
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


def _shell_orbitals(nBaths, l):
    """Sorted global spin-orbital indices of impurity shell ``l`` (spin-up then spin-down m's)."""
    return sorted(c2i(nBaths, (l, s, m)) for s in range(2) for m in range(-l, l + 1))


def _pes_ips_equivalence_groups(nBaths, l, block_structure):
    """Symmetry-equivalence label per PES/IPS operator of shell ``l`` (B2a dedup).

    ``photoemission_operators`` / ``inverse_photoemission_operators`` emit one operator per
    ``(s, m)`` in the order ``for s in 0,1: for m in -l..l``. ``block_structure`` is the impurity
    block structure of shell ``l`` in the basis of the Hamiltonian (local indices into the sorted
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


#: The three Cartesian polarization directions the spectral tensors are stored over.
CARTESIAN = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def _shell_windows(l_valence, l_core, *, imp, val, con, core_imp=(0, 0)):
    """Build the ``dN_imp``/``dN_val``/``dN_con`` basis-expansion windows for one technique.

    The windows say how far each shell's occupation may move while the excited basis is
    generated. They used to be written as ``{1: ..., 2: ...}`` at every call site, which made
    them silently wrong for any edge but L2,3; here the shells are named. A core shell has no
    bath of its own, so its valence/conduction windows are always ``(0, 0)``.
    """
    # Core shell first, then valence. The consumers look these up by key, so the order is not
    # semantic -- but basis generation walks the restrictions built from them, and a different
    # walk order reorders the determinants and so the floating-point summation inside the
    # Krylov recurrences. Keeping the historical order keeps the L2,3 spectra bit-reproducible.
    dN_imp, dN_val, dN_con = {}, {}, {}
    if l_core is not None:
        dN_imp[l_core] = core_imp
        dN_val[l_core] = (0, 0)
        dN_con[l_core] = (0, 0)
    dN_imp[l_valence] = imp
    dN_val[l_valence] = val
    dN_con[l_valence] = con
    return dict(dN_imp=dN_imp, dN_val=dN_val, dN_con=dN_con)


def _sector_restrictions_per_top(hOp, tOps, psis, basis):
    r"""Conserved-charge sector restrictions for each transition operator (or ``None``).

    Each Lanczos seed ``tOp|\psi\rangle`` lives in a definite conserved-charge sector
    (``q_ψ`` shifted by the operator's charge change). Confining the excited basis to that
    sector prunes determinants the per-shell occupation window would otherwise admit -- the
    spectra analogue of :func:`symmetries.gf_sector_restrictions` used by the self-energy.

    Returns a list aligned with ``tOps``; an entry is ``None`` when the operator has no
    definite sector (its terms disagree) so the caller falls back to the occupation window.
    Returns ``None`` (whole list) when the ground states do not share a single charge
    signature, or when any of them has no definite charge (a vector mixing sectors, as a
    degenerate multiplet may be returned) -- then no per-operator sector is well defined, and
    confining the seeds to a rounded one would prune everything outside it (review ledger C7).
    """
    n_orb = basis.num_spin_orbitals
    comm = basis.comm
    charges = conserved_subset_charges(hOp, n_orb=n_orb)
    gs_occ = None
    for psi in psis:
        # Collective; every rank reaches every call (no early exit before the loop ends would be
        # rank-safe either, but the verdicts are identical on every rank, so this one is).
        occ = definite_conserved_charges(psi, charges, n_orb, comm=comm)
        if occ is None:
            return None
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
    unit_report_label=None,
    gf_pt2_tol=None,
    _unit_row_indices=None,
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
        Incompatible with ``equivalence_groups``.
    gf_pt2_tol : float, optional
        The second-order growth rule (``SolverOptions.gf_pt2_tol``), scoring over the mesh ``w`` at broadening
        ``delta``; it replaces the amplitude default. ``None``: the amplitude rule.
    unit_report_label : str, optional
        When given, the completed calculation prints the maximum excited basis size each
        work unit reached -- one line per transition operator -- under a heading naming this
        spectrum (:func:`greens_function._report_max_unit_basis`). ``None`` (the default)
        prints nothing, which is what a caller that runs this function several times for one
        result wants (the susceptibility driver calls it once per spin sector).
    _unit_row_indices : list of int, optional
        Internal. The caller's operator index for each entry of ``tOps``, used only to label
        the report's rows. The ``equivalence_groups`` recursion below passes the original
        indices of the representatives it kept, so a row names the operator the reader has
        rather than a position in a symmetry-reduced list.

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
            unit_report_label=unit_report_label,
            gf_pt2_tol=gf_pt2_tol,
            _unit_row_indices=[first_index[label] for label in rep_order],
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
            base_restrictions if sec is None else intersect_sector_window(base_restrictions, sec)
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
        basis=basis,
    )
    if seed_transform is not None:
        # One operator per group here, so group_i identifies the transition operator and
        # the unit's seed columns are in eigenstate (chunk) order.
        for u, unit in enumerate(units):
            unit_seeds[u] = [seed_transform(ei, unit.group_i, seed) for ei, seed in zip(unit.chunk, unit_seeds[u])]
    unit_weights = gf.unit_cost_weights(unit_seeds, comm)

    # Spectra always run the sparse kernel without reorthogonalization.
    kernel = lanczos_unit_kernel(
        units,
        hOp,
        unit_restrictions,
        weighted_restrictions,
        reort=None,
        sparse=True,
        slaterWeightMin=slaterWeightMin,
        solver_verbose=verbose,
        print_size=verbose,
        prune_tol=0.0 if gf_pt2_tol else DEFAULT_GF_PRUNE_TOL,  # the second-order rule replaces the amplitude rule
        pt2_tol=gf_pt2_tol,
        eval_meshes_for=(
            None if not gf_pt2_tol else lambda unit: _gf_eval_meshes(None, w, 0, delta, [es[ei] for ei in unit.chunk])
        ),
    )

    results = gf.run_units_distributed(
        basis, unit_seeds, unit_weights, kernel, verbose=verbose, unit_windows=unit_restrictions, hOp=hOp
    )
    if results is None:  # non-root rank of a distributed run
        empty = np.empty((0, 0), dtype=complex)
        return [empty] * (1 + len(extra_meshes)) if extra_meshes is not None else empty

    # Reassemble per-(tOp, eigenstate) coefficients (unit.group_i indexes tOps), then evaluate the
    # thermal average on the frequency mesh -- on the root rank only, matching the self-energy
    # path and shrinking the gather payload to the Lanczos coefficients.
    by_op = states_by_group(units, results, len(tOps), len(psis))
    # Largest excited basis per transition operator, over the eigenstate chunks feeding it;
    # within one unit the basis only grows, so its final size is that unit's maximum.
    max_basis: dict[int, tuple[Optional[int], bool]] = {}
    for unit, (_alphas, _betas, _r_slices, cap_stats, _conv) in zip(units, results):
        gf._merge_unit_basis(max_basis, unit.group_i, cap_stats["retained_size"], cap_stats["cap_hit"])
    if unit_report_label is not None:
        # One row per transition operator, named by the CALLER's operator index: under
        # `equivalence_groups` the recursion above reduced `tOps` to one representative per
        # symmetry class, so `group_i` is a position in that reduced list and labelling rows
        # with it would point a reader at the wrong orbital.
        names = {i: f"operator {i if _unit_row_indices is None else _unit_row_indices[i]}" for i in max_basis}
        width = max((len(name) for name in names.values()), default=0)
        gf._report_max_unit_basis(
            f"Maximum excited basis size per unit -- {unit_report_label}",
            [(names[i].ljust(width), *max_basis[i]) for i in sorted(max_basis)],
        )

    thermal = ThermalEnsemble(es, tau)
    e0, Z = thermal.e0, thermal.Z
    meshes = [(w, delta)] + list(extra_meshes or [])
    gs_per_mesh = []
    for mesh, mesh_delta in meshes:
        gs_mesh = np.empty((len(mesh), len(tOps)), dtype=complex)
        for i in range(len(tOps)):
            G_tOp = gf.calc_thermally_averaged_G(*by_op[i], mesh, es, e0, tau, mesh_delta)
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
    weights = ThermalEnsemble(es, tau).weights
    m0 = np.zeros(m)
    m1 = np.zeros(m)
    work = basis.clone(initial_basis=[], verbose=False, comm=comm)
    for psi, wgt in zip(psis, weights):
        seeds = [applyOp_test(op, psi) for op in comp_ops]
        # <seed_a|H|seed_a> reads H|seed_a> only on the seeds' supports: generate nothing else. The
        # filter is collective, built once per state on every rank.
        keep = comm_support_filter(seeds, comm if basis.is_distributed else None)
        hseeds = [hOp.apply_block(s, 0, keep) for s in seeds]
        work.clear()
        for s in seeds + hseeds:
            work.add_states(s.keys())
        red = work.redistribute_psis(*seeds, *hseeds)
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
    unit_report_label=None,
    gf_pt2_tol=None,
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
        e0 = ThermalEnsemble(es, tau).e0
        m0, m1 = _component_seed_moments(hOp, all_rot_ops, psis, es, e0, tau, basis, slaterWeightMin)
        if not _moments_consistent(m0, m1, reduction.group_of_column):
            diagonalizable = False
            Q = np.eye(m, dtype=complex)
            rep_ops = list(component_ops)
            reduction = reduction._replace(
                Q=Q, representatives=list(range(m)), group_of_column=list(range(m)), diagonalizable=False
            )

    # One excited basis serves the whole block, so it can be confined to a conserved-charge
    # sector only when every component's seed lives in that same sector. This used to take
    # component 0's sector for all of them -- "all components share the charge shift" -- which
    # holds for the Cartesian dipole components (each sums over every core orbital) but not in
    # general: two components with definite but different sectors (c2^dag c0 and c3^dag c1 when
    # the core orbitals are separately conserved) had every other component's seed pruned to
    # nothing, and its spectrum came back identically zero. Disagreement now falls back to the
    # occupation window, exactly as a component with no definite sector already did.
    sectors = _sector_restrictions_per_top(hOp, rep_ops, psis, basis)
    shared = sectors is not None and sectors[0] is not None and all(s == sectors[0] for s in sectors)
    extra = sectors[0] if shared else None

    comm = basis.comm
    thermal = ThermalEnsemble(es, tau)
    e0, Z = thermal.e0, thermal.Z
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
        unit_report_label=unit_report_label,
        gf_pt2_tol=gf_pt2_tol,
        eval_omega=w,
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
