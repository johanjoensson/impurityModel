import itertools
from typing import Optional

import numpy as np
import scipy as sp
from mpi4py import MPI

# from impurityModel.ed import spectra
from impurityModel.ed.block_structure import BlockStructure, get_blocks
from impurityModel.ed.BlockLanczosArray import (
    Reort,
    block_lanczos_array,
    BETA_BLOWUP_FACTOR,
)
from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.symmetries import widen_weighted_restrictions
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyOperator,
    ManyBodyState,
)
from impurityModel.ed.mpi_comm import gather_distributed_results
from impurityModel.ed import gf_diagnostics as _gfd

comm = MPI.COMM_WORLD
rank = comm.rank


def build_full_greens_function(block_gf, block_structure: BlockStructure):
    """
    Assemble the full Green's function from individual blocks and block symmetries.

    Parameters
    ----------
    block_gf : list of ndarray
        Green's functions for each inequivalent block.
    block_structure : BlockStructure
        The block structure defining mapping and symmetry relationships.

    Returns
    -------
    res : ndarray
        The full Green's function matrix.
    """
    (
        blocks,
        identical_blocks,
        transposed_blocks,
        particle_hole_blocks,
        particle_hole_transposed_blocks,
        inequivalent_blocks,
    ) = block_structure
    n_orb = sum(len(block) for block in block_structure.blocks)
    if len(block_gf[0].shape) == 2:
        res = np.zeros((n_orb, n_orb), dtype=block_gf[0].dtype)
    elif len(block_gf[0].shape) == 3:
        res = np.zeros((block_gf[0].shape[0], n_orb, n_orb), dtype=block_gf[0].dtype)
    else:
        raise RuntimeError(
            f"Unknown data shape {block_gf[0].shape}. Should be 3 index (n_freq, n_orb,n_orb) or 2 index (n_orb,n_orb)"
        )
    if len(block_gf) == len(inequivalent_blocks):
        # block_gf contains only symmetrically inequivalent blocks
        for inequiv_i, gf_i in enumerate(block_gf):
            for block_i in identical_blocks[inequivalent_blocks[inequiv_i]]:
                if len(gf_i.shape) == 2:
                    block_idx = np.ix_(blocks[block_i], blocks[block_i])
                elif len(gf_i.shape) == 3:
                    block_idx = np.ix_(range(gf_i.shape[0]), blocks[block_i], blocks[block_i])
                res[block_idx] = gf_i
            for block_i in transposed_blocks[inequivalent_blocks[inequiv_i]]:
                if len(gf_i.shape) == 2:
                    block_idx = np.ix_(blocks[block_i], blocks[block_i])
                    res[block_idx] = np.transpose(gf_i, (1, 0))
                elif len(gf_i.shape) == 3:
                    block_idx = np.ix_(range(gf_i.shape[0]), blocks[block_i], blocks[block_i])
                    res[block_idx] = np.transpose(gf_i, (0, 2, 1))
            for block_i in particle_hole_blocks[inequivalent_blocks[inequiv_i]]:
                if len(gf_i.shape) == 2:
                    block_idx = np.ix_(blocks[block_i], blocks[block_i])
                elif len(gf_i.shape) == 3:
                    block_idx = np.ix_(range(gf_i.shape[0]), blocks[block_i], blocks[block_i])
                res[block_idx] = -np.conj(gf_i)
            for block_i in particle_hole_transposed_blocks[inequivalent_blocks[inequiv_i]]:
                if len(gf_i.shape) == 2:
                    block_idx = np.ix_(blocks[block_i], blocks[block_i])
                    res[block_idx] = -np.transpose(np.conj(gf_i), (1, 0))
                elif len(gf_i.shape) == 3:
                    block_idx = np.ix_(range(gf_i.shape[0]), blocks[block_i], blocks[block_i])
                    res[block_idx] = -np.transpose(np.conj(gf_i), (0, 2, 1))
    elif len(block_gf) == len(blocks):
        # block_gf contains all blocks
        for block_i, gf_i in enumerate(block_gf):
            if len(gf_i.shape) == 2:
                block_idx = np.ix_(blocks[block_i], blocks[block_i])
            elif len(gf_i.shape) == 3:
                block_idx = np.ix_(range(gf_i.shape[0]), blocks[block_i], blocks[block_i])
            res[block_idx] = gf_i
    else:
        raise RuntimeError(f"Block structure does not match block_gf.\n{block_structure=} {len(block_gf)=}")
    return res


def get_Greens_function(
    matsubara_mesh: np.ndarray,
    omega_mesh: np.ndarray,
    psis: list[ManyBodyState],
    es: list[float],
    tau: float,
    basis: Basis,
    hOp: ManyBodyOperator,
    delta: float,
    blocks: list[list[int]],
    verbose: bool,
    verbose_extra: bool,
    reort: Optional,
    dN: Optional[int],
    occ_cutoff: float,
    slaterWeightMin: float,
    sparse: bool,
    num_wanted: int = None,
):
    """
    Calculate interacting Greens function.

    Returns ``(gs_matsubara, gs_realaxis, report)`` on the root rank, where ``report`` is a
    :class:`gf_diagnostics.DiagnosticReport` of per-block convergence/consistency checks
    (``(None, None, None)`` on non-root ranks). ``num_wanted`` is the number of thermal states
    the eigensolver was asked for, used by the ensemble-truncation check.
    """
    (
        block_indices,
        block_roots,
        block_color,
        blocks_per_color,
        block_basis,
        psis,
        _,  # block_intercomms — freed collectively by gc after barrier in conftest
    ) = basis.split_basis_and_redistribute_psi([len(block) ** 2 for block in blocks], psis)
    if verbose:
        print(f"New block roots: {block_roots}")
        print(f"Blocks per color: {blocks_per_color}")
        print("=" * 80)
    block_indices_per_color = gather_distributed_results(
        basis.comm,
        block_basis.comm.rank if block_basis.comm is not None else 0,
        block_roots,
        blocks_per_color,
        np.array(block_indices),
        is_array=True,
    )

    local_gs_matsubara = []
    local_gs_realaxis = []
    local_block_diags = {}  # block_i -> list[Diagnostic]; filled on each color's root rank
    IPS_ops = ([ManyBodyOperator({((orb, "c"),): 1}) for orb in blocks[bi]] for bi in block_indices)
    PS_ops = ([ManyBodyOperator({((orb, "a"),): 1}) for orb in blocks[bi]] for bi in block_indices)
    for block_i, IPS_ops, PS_ops in zip(block_indices, IPS_ops, PS_ops):
        alphas_IPS, betas_IPS, r_IPS = calc_Greens_function_with_offdiag(
            hOp,
            IPS_ops,
            psis,
            es,
            block_basis,
            delta,
            reort=reort,
            dN=dN,
            occ_cutoff=occ_cutoff,
            slaterWeightMin=slaterWeightMin,
            verbose=verbose_extra,
            sparse=sparse,
        )
        alphas_PS, betas_PS, r_PS = calc_Greens_function_with_offdiag(
            hOp,
            PS_ops,
            psis,
            es,
            block_basis,
            -delta,
            reort=reort,
            dN=dN,
            occ_cutoff=occ_cutoff,
            slaterWeightMin=slaterWeightMin,
            verbose=verbose_extra,
            sparse=sparse,
        )

        e0 = np.min(es)
        Z = np.sum(np.exp(-(es - e0) / tau))
        if matsubara_mesh is not None and block_basis.comm.rank == 0:
            G_IPS = calc_thermally_averaged_G(alphas_IPS, betas_IPS, r_IPS, matsubara_mesh, es, e0, tau, 0)
            G_PS = calc_thermally_averaged_G(alphas_PS, betas_PS, r_PS, -matsubara_mesh, es, e0, tau, 0)
            local_gs_matsubara.append(
                (
                    G_IPS
                    - np.transpose(
                        G_PS,
                        (
                            0,
                            2,
                            1,
                        ),
                    )
                )
                / Z
            )
        G_IPS_real = G_PS_real = combined_real = None
        if omega_mesh is not None and block_basis.comm.rank == 0:
            G_IPS_real = calc_thermally_averaged_G(alphas_IPS, betas_IPS, r_IPS, omega_mesh, es, e0, tau, delta)
            G_PS_real = calc_thermally_averaged_G(alphas_PS, betas_PS, r_PS, -omega_mesh, es, e0, tau, -delta)
            combined_real = (G_IPS_real - np.transpose(G_PS_real, (0, 2, 1))) / Z
            local_gs_realaxis.append(combined_real)

        # --- per-block convergence / consistency diagnostics (color root rank) ---------
        if block_basis.comm.rank == 0:
            block_dim = len(blocks[block_i])
            diags = [
                _gfd.check_spectral_sum_rule(r_IPS, r_PS, es, e0, tau, block_dim),
                _gfd.check_thermal_weight_cutoff(es, e0, tau, n_returned=len(es), num_wanted=num_wanted),
            ]
            conv_add = _lanczos_convergence_summary(alphas_IPS, betas_IPS, delta)
            conv_rem = _lanczos_convergence_summary(alphas_PS, betas_PS, delta)
            converged = conv_add[0] and conv_rem[0]
            worst_dg = max(conv_add[1], conv_rem[1])
            n_blocks = max(conv_add[2], conv_rem[2])
            diags.append(_gfd.check_lanczos_convergence(converged, worst_dg, n_blocks, n_blocks))
            if G_IPS_real is not None:
                diags.append(_gfd.check_mesh_density(omega_mesh, delta))
                diags.append(
                    _gfd.check_integrated_weight(G_IPS_real, r_IPS, es, e0, tau, omega_mesh, "add", delta=delta)
                )
                diags.append(
                    _gfd.check_integrated_weight(G_PS_real, r_PS, es, e0, tau, -omega_mesh, "rem", delta=delta)
                )
                diags.append(_gfd.check_causality(combined_real, "G"))
            local_block_diags[int(block_i)] = diags
    gathered_matsubara = gather_distributed_results(
        basis.comm,
        block_basis.comm.rank if block_basis.comm is not None else 0,
        block_roots,
        blocks_per_color,
        local_gs_matsubara,
        is_array=False,
    )
    gathered_realaxis = gather_distributed_results(
        basis.comm,
        block_basis.comm.rank if block_basis.comm is not None else 0,
        block_roots,
        blocks_per_color,
        local_gs_realaxis,
        is_array=False,
    )
    # Gather the per-block diagnostics (collective on basis.comm; only color-root ranks hold
    # non-empty dicts) and merge them into one report on the root rank.
    gathered_diags = basis.comm.gather(local_block_diags, root=0) if basis.comm is not None else [local_block_diags]
    if basis.comm.rank == 0:
        gs_matsubara = [np.empty((len(matsubara_mesh), len(block), len(block)), dtype=complex) for block in blocks]
        gs_realaxis = [np.empty((len(omega_mesh), len(block), len(block)), dtype=complex) for block in blocks]
        for i, block_idx in enumerate(block_indices_per_color):
            gs_matsubara[block_idx][:] = gathered_matsubara[i]
            gs_realaxis[block_idx][:] = gathered_realaxis[i]
        report = _gfd.DiagnosticReport()
        merged = {}
        for part in gathered_diags:
            merged.update(part)
        for block_idx in sorted(merged):
            report.extend(str(blocks[block_idx]), merged[block_idx])

    # Free the split communicator collectively before returning.
    # block_basis.comm is a split comm created by split_basis_and_redistribute_psi.
    # MPI_Comm_free is collective — it must be called by all ranks in the comm
    # at the same time.  Leaving it for Python gc risks non-collective freeing.
    if block_basis is not None and block_basis.comm != basis.comm:
        block_basis.free_comm()

    return (gs_matsubara, gs_realaxis, report) if basis.comm.rank == 0 else (None, None, None)


def calc_Greens_function_with_offdiag(
    hOp,
    tOps,
    psis,
    es,
    block_basis,
    delta,
    reort: Optional = None,
    dN: Optional[int] = None,
    occ_cutoff: float = 1e-6,
    slaterWeightMin: float = 0,
    verbose: bool = True,
    sparse: bool = False,
    dN_imp=None,
    dN_val=None,
    dN_con=None,
):
    r"""
        Return Green's function for states with low enough energy.

        For states :math:`|psi \rangle`, calculate:

        :math:`g(w+1j*delta) =
        = \langle psi| tOp^\dagger ((w+1j*delta+e)*\hat{1} - hOp)^{-1} tOp
        |psi \rangle`,

        where :math:`e = \langle psi| hOp |psi \rangle`
    ,
        Lanczos algorithm is used.

        Parameters
        ----------
        n_spin_orbitals : int
            Total number of spin-orbitals in the system.
        hOp : dict
            Operator
        tOps : list
            List of dict operators
        psis : list
            List of Multi state dictionaries
        es : list
            Total energies
        w : list
            Real axis energy mesh
        delta : float
            Deviation from real axis.
            Broadening/resolution parameter.
        restrictions : dict
            Restriction the occupation of generated
            product states.
        krylovSize : int
            Size of the Krylov space
        slaterWeightMin : float
            Restrict the number of product states by
            looking at `|amplitudes|^2`.
        parallelization_mode : str
                "eigen_states" or "H_build".

    """

    # Set limits for change occupation, if any.
    # limits are pairs of integers (max_holes, max_el)
    # These limits are imposed on top of the (effective) ground state limitations.
    if dN_imp is None:
        if dN is not None:
            dN_imp = dict.fromkeys(block_basis.impurity_orbitals, (dN, dN))
    else:
        dN_imp = {i: dN_imp.get(i) for i in block_basis.impurity_orbitals}

    if dN_val is None:
        if dN is not None:
            dN_val = dict.fromkeys(block_basis.impurity_orbitals, (dN, 0))
    else:
        dN_val = {i: dN_val.get(i) for i in block_basis.impurity_orbitals}

    if dN_con is None:
        if dN is not None:
            dN_con = dict.fromkeys(block_basis.impurity_orbitals, (0, dN))
    else:
        dN_con = {i: dN_con.get(i) for i in block_basis.impurity_orbitals}

    excited_restrictions = block_basis.build_excited_restrictions(
        hOp,
        psis,
        es,
        imp_change=dN_imp,
        val_change=dN_val,
        con_change=dN_con,
        cutoff=occ_cutoff,
    )
    # Weighted (e.g. S_z) restriction for the excited sector: widen the ground-state
    # bounds by one orbital weight so the addition (c_j†) / removal (c_j) sectors
    # q_psi ± w_j are admitted while still confining the basis.
    excited_weighted_restrictions = widen_weighted_restrictions(block_basis.weighted_restrictions)
    block_v = [[ManyBodyState({}) for _ in tOps] for _ in psis]
    for i_tOp, tOp in enumerate(tOps):
        tOp.set_restrictions(excited_restrictions)
        tOp.set_weighted_restrictions(excited_weighted_restrictions)
        res_psis = tOp.apply_multi(psis, cutoff=slaterWeightMin)
        for j_psi, res_psi in enumerate(res_psis):
            block_v[j_psi][i_tOp] += res_psi
    block_v_lengths = np.array([sum(len(t_psi) for t_psi in t_psis) for t_psis in block_v])
    block_basis.comm.Allreduce(MPI.IN_PLACE, block_v_lengths, op=MPI.SUM)

    (
        excited_indices,
        excited_roots,
        excited_color,
        excited_states_per_color,
        split_original_basis,
        split_original_psis,
        _,
    ) = block_basis.split_basis_and_redistribute_psi(
        np.log10(block_v_lengths + 1) + 1, [t_psi for t_psis in block_v for t_psi in t_psis]
    )
    if verbose:
        print(f"New excited state roots: {excited_roots}")
        print(f"excited states per color: {excited_states_per_color}")
        print("=" * 80, flush=True)
    excited_indices_per_color = gather_distributed_results(
        block_basis.comm,
        split_original_basis.comm.rank if split_original_basis.comm is not None else 0,
        excited_roots,
        excited_states_per_color,
        np.array(excited_indices),
        is_array=True,
    )

    excited_block_psis = [[ManyBodyState({}) for _ in vs] for vs in block_v]
    for i, j in itertools.product(range(len(tOps)), range(len(psis))):
        excited_block_psis[j][i] += split_original_psis[j * len(tOps) + i]
    local_alphas = []
    local_betas = []
    local_r = []
    if verbose and excited_restrictions is not None:
        print("Excited state restrictions:")
        for indices, occupations in excited_restrictions.items():
            print(f"---> {sorted(indices)} : {occupations}")
    for excited_psis in (excited_block_psis[ei] for ei in excited_indices):
        excited_basis = split_original_basis.clone(
            initial_basis=set(state for p in excited_psis for state in p),
            restrictions=excited_restrictions,
            weighted_restrictions=excited_weighted_restrictions,
            verbose=False,
        )

        if excited_basis.restrictions is not None:
            hOp.set_restrictions(excited_basis.restrictions)
        if excited_basis.weighted_restrictions is not None:
            hOp.set_weighted_restrictions(excited_basis.weighted_restrictions)
        if sparse:
            alphas, betas, r = block_Green_sparse(
                reort=reort,
                hOp=hOp,
                psi_arr=excited_basis.redistribute_psis(excited_psis),
                basis=excited_basis,
                delta=delta,
                slaterWeightMin=slaterWeightMin,
                verbose=verbose,
            )
        else:
            alphas, betas, r = block_Green(
                reort=reort,
                hOp=hOp,
                psi_arr=excited_basis.redistribute_psis(excited_psis),
                basis=excited_basis,
                delta=delta,
                slaterWeightMin=slaterWeightMin,
                verbose=verbose,
            )
        local_alphas.append(alphas)
        local_betas.append(betas)
        local_r.append(r)
        if verbose:
            print(f"Expanded excited state basis contains {len(excited_basis)} elements.")

    gathered_alphas = gather_distributed_results(
        block_basis.comm,
        excited_basis.comm.rank if excited_basis.comm is not None else 0,
        excited_roots,
        excited_states_per_color,
        local_alphas,
        is_array=False,
    )
    gathered_betas = gather_distributed_results(
        block_basis.comm,
        excited_basis.comm.rank if excited_basis.comm is not None else 0,
        excited_roots,
        excited_states_per_color,
        local_betas,
        is_array=False,
    )
    gathered_r = gather_distributed_results(
        block_basis.comm,
        excited_basis.comm.rank if excited_basis.comm is not None else 0,
        excited_roots,
        excited_states_per_color,
        local_r,
        is_array=False,
    )
    excited_alphas = None
    excited_betas = None
    excited_r = None
    if block_basis.comm.rank == 0:
        excited_alphas = [None for _ in psis]
        excited_betas = [None for _ in psis]
        excited_r = [None for _ in psis]
        for i, excited_i in enumerate(excited_indices_per_color):
            excited_alphas[excited_i] = gathered_alphas[i]
            excited_betas[excited_i] = gathered_betas[i]
            excited_r[excited_i] = gathered_r[i]
        assert not any(alpha is None for alpha in excited_alphas), f"{excited_alphas=}"
        assert not any(beta is None for beta in excited_betas), f"{excited_betas=}"
        assert not any(r is None for r in excited_r), f"{excited_r=}"

    # Free the split communicator collectively before returning.
    if split_original_basis is not None and split_original_basis.comm != block_basis.comm:
        split_original_basis.free_comm()

    return excited_alphas, excited_betas, excited_r


def build_qr(psi):
    """
    Perform an economic QR decomposition of a state matrix.

    Parameters
    ----------
    psi : ndarray
        The input state matrix.

    Returns
    -------
    psi_orthogonal : ndarray
        The orthogonalized matrix Q.
    r : ndarray
        The upper triangular matrix R.
    """
    # Do a QR decomposition of the starting block.
    # Later on, use r to restore the psi block
    psi, r = sp.linalg.qr(psi.copy(), mode="economic", overwrite_a=True, check_finite=False, pivoting=False)
    return np.ascontiguousarray(psi), r


def calc_continuants(diagonal, offdiagonal):
    """
    Calculate continued fraction continuants.

    """

    An = np.empty_like(diagonal)
    Bn = np.empty_like(An)
    An[-1] = np.eye(diagonal.shape[1])
    Bn[-1] = 0
    An[0] = diagonal[0]
    Bn[0] = 1
    for n in range(1, diagonal.shape[0]):
        An[n] = diagonal[n] * An[n - 1] - np.conj(offdiagonal[n]) * An[n - 2] * offdiagonal[n]
        Bn[n] = diagonal[n] * Bn[n - 1] - np.conj(offdiagonal[n]) * Bn[n - 2] * offdiagonal[n]
    return An, Bn


def block_Green(
    hOp,
    psi_arr,
    basis,
    delta,
    reort,
    slaterWeightMin=0,
    verbose=True,
):
    """
    calculate  one block of the Greens function. This function builds the many body basis iteratively. Reducing memory requrements.
    """

    len(basis)
    n = len(psi_arr)

    # alphas/betas stay padded (k, P, P) here so the cross-expansion elementwise
    # diff below has matching shapes; they are trimmed to true block widths before
    # any continued-fraction evaluation and at the final return.
    alphas, betas, r, last_q, widths = block_green_impl(
        basis, hOp, basis.redistribute_psis(psi_arr), delta, reort, slaterWeightMin, verbose
    )
    done = False
    while not done:
        old_size = basis.size
        new_psis = last_q
        for i in range(5):
            new_psis = hOp.apply_multi(new_psis, cutoff=slaterWeightMin)
            basis.add_states(
                set(state for p in new_psis for state in p if state not in basis.local_basis),
            )
        if basis.size == old_size or basis.size > basis.truncation_threshold:
            break
        if verbose:
            print(f"    expanded basis contains {basis.size} states")
        alphas_prev = alphas
        betas_prev = betas
        widths_prev = widths
        alphas, betas, r, last_q, widths = block_green_impl(
            basis, hOp, basis.redistribute_psis(psi_arr), delta, reort, slaterWeightMin, verbose
        )

        n_test = min(alphas.shape[0], alphas_prev.shape[0])
        # relatively large changes in alpha and/or betas means we have not converged
        if np.any(np.abs(alphas[:n_test] - alphas_prev[:n_test]) > 1e-12) or np.any(
            np.abs(betas[:n_test] - betas_prev[:n_test]) > 1e-12
        ):
            done = False
            continue

        # alphas seem decently converged, check the Greens function to be sure
        a_t, b_t = _trim_blocks(alphas, betas, widths)
        ap_t, bp_t = _trim_blocks(alphas_prev, betas_prev, widths_prev)
        ws = np.concatenate([np.diagonal(a) for a in a_t])[: n_test * n] if a_t else np.zeros(0, dtype=complex)
        G_prev = calc_G(ap_t, bp_t, np.identity(n), ws, 0, delta)
        G = calc_G(a_t, b_t, np.identity(n), ws, 0, delta)
        done = (
            np.all(np.diagonal(G.imag, axis1=1, axis2=2) * np.sign(delta) <= 0) and np.max(np.abs(G - G_prev)) < 1e-12
        )
    return _trim_blocks(alphas, betas, widths) + (r,)


def _scatter_qr_columns(comm, psi_dense, r, local_size):
    """Scatter the row-distributed QR factor ``Q`` (held on rank 0) across MPI ranks.

    Rank 0 holds the full ``(N, n)`` ``Q`` and the ``(n, n)`` ``R`` after :func:`build_qr`.
    Broadcast ``R`` and the column count, then ``Scatterv`` ``Q``'s rows onto each rank's
    local partition (``local_size`` rows). Shared by ``block_green_impl`` (sparse branch)
    and ``block_Green_sparse``.

    Returns
    -------
    psi_dense_local : ndarray
        This rank's ``(local_size, n)`` slice of ``Q``.
    r : ndarray
        The ``(n, n)`` ``R`` factor (replicated on every rank).
    """
    rank = comm.rank
    r = comm.bcast(r if rank == 0 else None, root=0)
    columns = comm.bcast(psi_dense.shape[1] if rank == 0 else None, root=0)
    psi_dense_local = np.empty((local_size, columns), dtype=complex, order="C")
    send_counts = np.empty((comm.size), dtype=int) if rank == 0 else None
    comm.Gather(np.array([psi_dense_local.size]), send_counts, root=0)
    offsets = np.array([np.sum(send_counts[:rr]) for rr in range(comm.size)], dtype=int) if rank == 0 else None
    comm.Scatterv(
        [psi_dense, send_counts, offsets, MPI.C_DOUBLE_COMPLEX] if rank == 0 else None,
        psi_dense_local,
        root=0,
    )
    return psi_dense_local, r


def _make_gf_convergence_monitor(delta, slaterWeightMin):
    r"""Frozen-mesh relative-change convergence monitor for the block-Lanczos Green's function.

    Shared by both GF kernels (``block_green_impl``, ``block_Green_sparse``). Returns
    ``(converged_fn, converged_flag)``: ``converged_fn(alphas, betas, verbose, block_widths)``
    estimates ``G`` on a mesh frozen once a few blocks have resolved the spectral edges
    (:func:`_gf_sample_mesh`) and reports convergence when the relative change
    (:func:`_greens_function_change`, with the cross-step ``gs_cache``) drops below
    ``max(slaterWeightMin**2, 1e-8)``. ``converged_flag[0]`` records whether tolerance was ever
    met, for the non-convergence warning.
    """
    delta_min = max(slaterWeightMin**2, 1e-8)
    converged_flag = [False]
    mesh_cache = [None]
    gs_cache = [None, 0]

    def converged(alphas, betas, verbose=False, block_widths=None, **kwargs):
        if len(alphas) <= 1:
            return False
        if mesh_cache[0] is None:
            if len(alphas) < 3:  # let the extremal Ritz values settle before freezing
                return False
            A_trim, _ = (
                _trim_blocks(alphas, betas, block_widths)
                if (block_widths is not None and len(block_widths) == len(alphas))
                else ([np.asarray(a) for a in alphas], None)
            )
            mesh_cache[0] = _gf_sample_mesh(A_trim, delta)
        d_g = _greens_function_change(alphas, betas, block_widths, delta, omegaP=mesh_cache[0], cache=gs_cache)
        if d_g is None:  # spurious (wrong-sign) imaginary part -> not converged
            return False
        if verbose:
            print(rf"$\delta$ = {d_g}", flush=True)
        is_conv = d_g < delta_min
        converged_flag[0] = converged_flag[0] or is_conv
        return is_conv

    return converged, converged_flag


def block_green_impl(basis, hOp, psi_arr, delta, reort, slaterWeightMin, verbose):
    """
    Internal block Green's function implementation.

    Parameters
    ----------
    basis : Basis
        The many-body basis.
    hOp : dict
        Hamiltonian operator.
    psi_arr : list of ManyBodyState
        Input state vectors.
    delta : float or ndarray
        Imaginary part/mesh info.
    reort : Reort
        Reorthogonalization method.
    slaterWeightMin : float
        Slater determinant cutoff weight.
    verbose : bool
        Whether to print verbose output.

    Returns
    -------
    gs_matsubara : ndarray
        Matsubara Green's function.
    gs_realaxis : ndarray
        Real axis Green's function.
    r : ndarray
        R matrix from QR.
    psi_arr : list
        Resulting states.
    """
    len(basis)
    n = len(psi_arr)

    comm = basis.comm
    rank = comm.rank if comm is not None else 0

    dense = len(basis) < 500
    if dense:
        psi_dense = basis.build_vector(psi_arr, slaterWeightMin=0).T
        psi_dense_local, r = build_qr(psi_dense)
    else:
        psi_dense = basis.build_vector(psi_arr, root=0, slaterWeightMin=0).T
        if rank == 0:
            psi_dense, r = build_qr(psi_dense)
        psi_dense_local, r = _scatter_qr_columns(
            comm, psi_dense if rank == 0 else None, r if rank == 0 else None, len(basis.local_basis)
        )

    if psi_dense_local.shape[1] == 0:
        return np.zeros((0, n, n), dtype=complex), np.zeros((0, n, n), dtype=complex), r, psi_arr, []

    converged, converged_flag = _make_gf_convergence_monitor(delta, slaterWeightMin)

    if dense:
        H = basis.build_dense_matrix(hOp)
        alphas, betas, Q_list, widths = block_lanczos_array(
            psi0=psi_dense_local,
            h_op=H,
            converged=converged,
            verbose=False and verbose,
            reort=reort if reort is not None else Reort.NONE,
            return_widths=True,
        )
    else:
        h_local = basis.build_sparse_matrix(hOp)[:, basis.local_indices]

        def matmat(v):
            """
            Perform matrix-matrix multiplication with the local Hamiltonian.

            Applies the local Hamiltonian to a set of state vectors and performs
            an MPI reduction across MPI processes to accumulate the results.

            Parameters
            ----------
            v : ndarray
                Input vectors to multiply.

            Returns
            -------
            res : ndarray
                The resulting matrix product after MPI reduction.
            """
            res = h_local @ v
            if comm is not None:
                comm.Reduce(MPI.IN_PLACE if rank == 0 else res, res, op=MPI.SUM, root=0)
            return res.reshape(h_local.shape[0], v.shape[1])

        H = sp.sparse.linalg.LinearOperator(
            (len(basis), len(basis.local_indices)),
            matvec=matmat,
            rmatvec=matmat,
            matmat=matmat,
            rmatmat=matmat,
            dtype=complex,
        )

        # Run Lanczos on psi0^T* [wI - j*delta - H]^-1 psi0
        alphas, betas, Q_list, widths = block_lanczos_array(
            psi0=psi_dense_local,
            h_op=H,
            converged=converged,
            reort=reort if reort is not None else Reort.NONE,
            verbose=False and verbose,
            comm=comm,
            return_widths=True,
        )
    if not converged_flag[0] and verbose and rank == 0:
        print(
            f"warning: block Green's function did not reach the convergence tolerance "
            f"{max(slaterWeightMin ** 2, 1e-8):.1e} in {len(alphas)} block(s). The continued fraction uses the "
            f"subspace built so far.",
            flush=True,
        )
    # Keep alphas/betas padded (k, P, P) for the caller's elementwise cross-expansion diff;
    # only drop a corrupted trailing tail (whole blocks + widths) so it never reaches the
    # continued fraction. Norms of padded blocks equal those of the true blocks (zeros add
    # nothing), so the scan is valid on the padded arrays.
    keep = len(_sanitize_continued_fraction(list(alphas), list(betas), verbose=verbose, rank=rank)[0])
    if keep < len(alphas):
        alphas, betas, widths = alphas[:keep], betas[:keep], widths[:keep]
    q_last = Q_list[:, -1:]
    return alphas, betas, r, basis.build_state(q_last.T, slaterWeightMin=slaterWeightMin), widths


def calc_thermally_averaged_G(alphas, betas, r, mesh, es, e0, tau, delta):
    """
    Calculate the thermally averaged Green's function over multiple initial states.

    Parameters
    ----------
    alphas : list of list of ndarray
    betas : list of list of ndarray
    r : list of ndarray
    mesh : ndarray
    es : list of float
    e0 : float
    tau : float
    delta : float

    Returns
    -------
    G_avg : ndarray
    """
    if len(alphas) == 0:
        return np.zeros((len(mesh), 0, 0), dtype=complex)

    n_ops = r[0].shape[-1]
    G_avg = np.zeros((len(mesh), n_ops, n_ops), dtype=complex)

    for e, alphas_e, betas_e, r_e in zip(es, alphas, betas, r):
        G_avg += calc_G(alphas_e, betas_e, r_e, mesh, e, delta) * np.exp(-(e - e0) / tau)

    return G_avg


def block_Green_sparse(
    hOp,
    psi_arr,
    basis,
    delta,
    reort: Optional = None,
    slaterWeightMin=0,
    verbose=True,
):
    """
    calculate  one block of the Greens function. This function builds the many body basis iteratively. Reducing memory requrements.
    """
    mpi = basis.comm is not None
    comm = basis.comm if mpi else None
    rank = comm.rank if mpi else 0

    N = len(basis)
    n = len(psi_arr)

    if N == 0 or n == 0:
        return np.empty((0, n, n), dtype=complex), np.empty((0, n, n), dtype=complex), np.zeros((n, n), dtype=complex)
    psi_dense = basis.build_vector(psi_arr, root=0, slaterWeightMin=slaterWeightMin).T
    if rank == 0:
        psi_dense, r = build_qr(psi_dense)
    if mpi:
        psi_dense_local, r = _scatter_qr_columns(
            comm, psi_dense if rank == 0 else None, r if rank == 0 else None, len(basis.local_basis)
        )
    else:
        psi_dense_local = psi_dense
    psi_arr = basis.build_state(psi_dense_local.T, slaterWeightMin=0)
    if len(psi_arr) == 0:
        return np.empty((0, n, n), dtype=complex), np.empty((0, n, n), dtype=complex), r

    converged, converged_flag = _make_gf_convergence_monitor(delta, slaterWeightMin)

    # Cap the Krylov dimension: without convergence the recurrence would otherwise run the
    # whole (possibly large) excited basis. basis.size // n is the kernel default; we make
    # it explicit so we can detect and report a non-converged solve below.
    max_iter = max(int(getattr(basis, "size", 0)) // max(n, 1), 1)
    alphas, betas, _, _, widths = block_lanczos_cy(
        psi_arr,
        hOp,
        basis,
        converged,
        verbose=verbose,
        reort=reort if reort is not None else Reort.NONE,
        slaterWeightMin=slaterWeightMin,
        max_iter=max_iter,
        return_widths=True,
    )

    if not converged_flag[0] and verbose and rank == 0:
        print(
            f"warning: block Green's function did not reach the convergence tolerance "
            f"{max(slaterWeightMin ** 2, 1e-8):.1e} in {len(alphas)} block(s) (max_iter={max_iter}). The "
            f"continued fraction uses the subspace built so far.",
            flush=True,
        )

    alphas, betas = _trim_blocks(alphas, betas, widths)
    alphas, betas = _sanitize_continued_fraction(alphas, betas, verbose=verbose, rank=rank)
    return alphas, betas, r


def _trim_blocks(alphas, betas, block_widths):
    r"""Strip the zero padding from block-Lanczos coefficients (shrinking blocks).

    The Lanczos kernels store every block into a fixed ``(P, P)`` pre-allocated
    buffer, zero-padding the inactive rows/columns whenever a block deflates
    (``block_widths[i] < P``).  This returns the true variable-dimension blocks:
    ``alphas[i] -> (w_i, w_i)`` and ``betas[i] -> (w_{i+1}, w_i)`` where
    ``w_i = block_widths[i]`` (the trailing ``betas[-1]`` residual block keeps its
    stored row count — it is the coupling beyond the subspace and is unused by the
    continued fraction).

    Args:
        alphas: Diagonal blocks, ``(k, P, P)`` ndarray (or length-``k`` sequence).
        betas: Off-diagonal blocks, same outer length.
        block_widths: True width ``w_i`` of every block.

    Returns:
        tuple[list, list]: ragged ``(alphas, betas)`` lists of 2D arrays.
    """
    widths = [int(w) for w in block_widths]
    k = len(widths)
    a = [np.asarray(alphas[i])[: widths[i], : widths[i]] for i in range(k)]
    b = []
    for i in range(k):
        rows = widths[i + 1] if i + 1 < k else np.asarray(betas[i]).shape[0]
        b.append(np.asarray(betas[i])[:rows, : widths[i]])
    return a, b


def _sanitize_continued_fraction(alphas, betas, verbose=False, rank=0):
    r"""Drop a corrupted trailing tail from the block-Lanczos coefficients.

    Defense-in-depth before the continued fraction / self-energy: the Lanczos kernels now
    truncate a diverging recurrence at the source (CholeskyQR2 + the ``BETA_BLOWUP_FACTOR``
    guard), but should a non-finite or runaway block ever reach here it must *not* be fed
    silently into :func:`calc_G` and ``sig_static``.  Scans the (trimmed) blocks and keeps
    only the leading run whose norms stay bounded relative to the healthy part; the trailing
    ``beta`` of the kept run is the (ignored) residual coupling, so dropping the tail is
    consistent with the continued fraction's own convention.

    Returns the (possibly shortened) ``(alphas, betas)`` and warns when a tail is dropped.
    """
    norm_max = 0.0
    keep = len(alphas)
    for i in range(len(alphas)):
        a = np.asarray(alphas[i])
        b = np.asarray(betas[i])
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            keep = i
            break
        a_norm = float(np.linalg.norm(a, 2)) if a.size else 0.0
        b_norm = float(np.linalg.norm(b, 2)) if b.size else 0.0
        if i > 0 and max(a_norm, b_norm) > BETA_BLOWUP_FACTOR * max(norm_max, 1.0):
            keep = i
            break
        norm_max = max(norm_max, a_norm, b_norm)
    if keep < len(alphas):
        if verbose and rank == 0:
            print(
                f"warning: discarding {len(alphas) - keep} corrupted block(s) from the "
                f"Green's-function continued fraction before computing the self-energy.",
                flush=True,
            )
        return alphas[:keep], betas[:keep]
    return alphas, betas


def _block_cf_inverse(alphas, betas, omegaP):
    r"""Level-0 inverse resolvent of a block-tridiagonal :math:`T` by continued fraction.

    Builds, for every frequency ``omegaP`` (already shifted, i.e.
    :math:`\omega + i\delta + e`),

    .. math::

        G^{-1}_0(\omega) = \omega I - \alpha_0
            - \beta_0^\dagger \big(\omega I - \alpha_1 - \cdots\big)^{-1} \beta_0 ,

    where ``alphas[i]`` is the ``(n_i, n_i)`` diagonal block and ``betas[i]`` the
    ``(n_{i+1}, n_i)`` sub-diagonal block coupling block ``i`` to ``i+1``.  Block
    dimensions may vary from level to level (shrinking-block deflation) and the
    ``betas`` may be rectangular; the identity at each level is sized from that
    level's diagonal block, so no fixed block dimension is assumed.  The trailing
    ``betas[-1]`` (residual coupling beyond the retained subspace) is ignored.

    Args:
        alphas: Length-``k`` sequence of square diagonal blocks.
        betas: Length-``k`` sequence of sub-diagonal blocks.
        omegaP: ``(n_w,)`` complex frequency mesh (shift already applied).

    Returns:
        numpy.ndarray: ``(n_w, n_0, n_0)`` inverse resolvent at the first block.
    """
    nw = omegaP.shape[0]

    def wI(n):
        return omegaP[:, np.newaxis, np.newaxis] * np.identity(n, dtype=complex)[np.newaxis]

    a_last = np.asarray(alphas[-1])
    G_inv = wI(a_last.shape[0]) - a_last[np.newaxis]
    for alpha, beta in zip(alphas[-2::-1], betas[-2::-1]):
        alpha = np.asarray(alpha)
        beta = np.asarray(beta)
        n_i = alpha.shape[0]
        beta_b = np.broadcast_to(beta, (nw,) + beta.shape)
        G_inv = wI(n_i) - alpha[np.newaxis] - np.conj(beta.T)[np.newaxis] @ np.linalg.solve(G_inv, beta_b)
    return G_inv


def calc_G(alphas, betas, r, omega, e, delta):
    r"""Green's function from block-Lanczos continued-fraction coefficients.

    Computes :math:`G(\omega) = r^\dagger (\omega + i\delta + e - T)^{-1} r` where
    ``T`` is the block-tridiagonal matrix with diagonal blocks ``alphas`` and
    sub-diagonal blocks ``betas``.  ``alphas`` / ``betas`` may be either a uniform
    ``(k, p, p)`` ndarray (no deflation) or ragged sequences of variable-dimension
    2D blocks (after :func:`_trim_blocks`); rectangular ``betas`` from shrinking-block
    deflation are handled — no fixed block dimension is assumed.

    Parameters
    ----------
    alphas : ndarray or sequence of ndarray
        Diagonal continued-fraction blocks.
    betas : ndarray or sequence of ndarray
        Off-diagonal continued-fraction blocks (``betas[i]`` couples block ``i`` to
        ``i+1`` with shape ``(n_{i+1}, n_i)``).
    r : ndarray
        ``(n_0, n_ops)`` projection of the seed block onto the first Lanczos block.
    omega : ndarray
        Frequency mesh.
    e : float
        Energy offset.
    delta : float
        Broadening factor.

    Returns
    -------
    G : ndarray
        ``(len(omega), n_ops, n_ops)`` Green's function.
    """
    r = np.asarray(r)
    if len(alphas) == 0:
        n_ops = r.shape[-1]
        return np.zeros((len(omega), n_ops, n_ops), dtype=complex)
    omegaP = np.asarray(omega) + 1j * delta + e
    G_inv = _block_cf_inverse(alphas, betas, omegaP)
    r_b = np.broadcast_to(r, (omegaP.shape[0],) + r.shape)
    return np.conj(r.T)[np.newaxis] @ np.linalg.solve(G_inv, r_b)


def _greens_function_change(alphas, betas, block_widths, delta, omegaP=None, cache=None):
    r"""Relative change in the block resolvent when the last Lanczos block is added.

    Block-Lanczos convergence monitor for the Green's function.  It compares the
    seed-block resolvent :math:`G = (G^{-1}_0)^{-1}` built from all ``k`` blocks against
    the one from the first ``k-1`` blocks, on sample frequencies drawn from the (trimmed)
    diagonal blocks (the broadened Ritz values — where the spectral weight sits).  The
    measure is the *relative* change

    .. math::

        d_g = \frac{\max_\omega \lVert G_k(\omega) - G_{k-1}(\omega)\rVert}
                   {\max_\omega \lVert G_k(\omega)\rVert},

    so it is scale-invariant and reflects the spectral function the self-energy actually
    needs — unlike the absolute change in :math:`G^{-1}_0`, whose leading
    :math:`\omega I - \alpha_0` term (:math:`\sim\lvert\omega\rvert`) is identical between
    the two and which therefore never decays to a tight absolute tolerance even when the
    spectrum is fully resolved.  Shrinking-block deflation is handled by trimming to
    ``block_widths`` first, so no fixed block dimension is assumed.

    The optional ``cache`` (a ``[gs_value, n_blocks]`` list) lets the convergence loop reuse
    work: this step's ``G^{-1}`` over ``k-1`` blocks is *identical* to the previous step's
    over ``k-1`` blocks on the same frozen mesh, so the ``gs_prev`` continued fraction is taken
    from the cache instead of rebuilt — halving the per-step continued-fraction cost. Exact
    (no behavior change); pass the same list each step.

    Returns:
        float or None: the relative change, or ``None`` if the freshly added block yields
        a wrong-sign spectral weight (not yet stabilized).
    """
    if block_widths is not None and len(block_widths) == len(alphas):
        A, B = _trim_blocks(alphas, betas, block_widths)
    else:
        A = [np.asarray(alphas[i]) for i in range(len(alphas))]
        B = [np.asarray(betas[i]) for i in range(len(betas))]
    if omegaP is None:
        # Default (back-compat / standalone use): sample at the current Ritz values. The
        # convergence loop should instead pass a *frozen* mesh (see _gf_sample_mesh): the
        # Ritz set grows every step, so each new block adds a pole at a fresh sample point
        # and the change never decays — measuring on a fixed mesh is what converges.
        n0 = A[0].shape[0]
        ws = np.concatenate([np.diagonal(a) for a in A])[: 15 * n0]
        omegaP = ws.real + delta * 1j
    gs_new = _block_cf_inverse(A, B, omegaP)
    if cache is not None and cache[0] is not None and cache[1] == len(A) - 1:
        gs_prev = cache[0]  # == previous step's gs_new (CF over the same k-1 blocks/mesh)
    else:
        gs_prev = _block_cf_inverse(A[:-1], B[:-1], omegaP)
    if cache is not None:
        cache[0], cache[1] = gs_new, len(A)
    if np.any(np.diagonal(gs_new.imag, axis1=1, axis2=2) * np.sign(delta) < 0):
        return None
    # Compare the resolvents G = (G^{-1})^{-1}, not their inverses: the broadening
    # (Im omega = delta) keeps G^{-1} non-singular, so the inverse is well defined.
    G_new = np.linalg.inv(gs_new)
    G_prev = np.linalg.inv(gs_prev)
    scale = np.max(np.abs(G_new))
    return np.max(np.abs(G_new - G_prev)) / max(scale, np.finfo(float).tiny)


def _lanczos_convergence_summary(alphas_list, betas_list, delta, tol=1e-6):
    r"""Post-hoc block-Lanczos convergence summary over the per-thermal-state coefficients.

    Avoids threading run-time monitor state out of ``block_Green_sparse``: for each thermal
    state's trimmed ``(alphas, betas)`` it re-evaluates the final relative resolvent change on
    a frozen mesh (the same measure the run-time monitor uses, via
    :func:`_greens_function_change`).  A state whose final change exceeds ``tol`` (and was not
    a short invariant-subspace run) is not fully resolved.

    Args:
        alphas_list, betas_list: Per-thermal-state lists of trimmed Lanczos blocks.
        delta: Broadening used to place the frozen sample mesh off the real axis.
        tol: Relative-change threshold below which a state counts as converged.

    Returns:
        tuple[bool, float, int]: ``(all_converged, worst_final_change, max_blocks)``.
    """
    worst = 0.0
    max_blocks = 0
    all_converged = True
    for A, B in zip(alphas_list, betas_list):
        A = list(A)
        B = list(B)
        max_blocks = max(max_blocks, len(A))
        if len(A) <= 2:  # invariant subspace reached almost immediately -> exact
            continue
        mesh = _gf_sample_mesh(A, delta if delta else 1.0)
        d_g = _greens_function_change(A, B, None, delta if delta else 1.0, omegaP=mesh)
        if d_g is None:
            all_converged = False
            continue
        worst = max(worst, float(d_g))
        if d_g >= tol:
            all_converged = False
    return all_converged, worst, max_blocks


def _gf_sample_mesh(alphas, delta, n_points=64):
    r"""Frozen real-frequency mesh for the block-Lanczos Green's-function convergence test.

    Spans the current Ritz range (the diagonal entries of the ``alphas`` blocks — Lanczos
    resolves the spectral *edges* within a few blocks) padded by a margin, on the line
    :math:`\omega + i\,\mathrm{sign}(\delta)\,\lvert\delta\rvert`.  The caller builds this
    once and reuses it, so the convergence measure is evaluated at *fixed* frequencies and
    actually decays as the spectrum fills in.
    """
    ws = np.concatenate([np.real(np.diagonal(np.asarray(a))) for a in alphas])
    lo, hi = float(np.min(ws)), float(np.max(ws))
    span = hi - lo
    margin = 0.05 * span + 10.0 * abs(delta)
    grid = np.linspace(lo - margin, hi + margin, n_points)
    return grid + 1j * delta


def rotate_matrix(M, T):
    r"""
    Rotate the matrix, M, using the matrix T.
    Returns M' = T^\dagger M T
    Parameters
    ==========
    M : NDArray - Matrix to rotate
    T : NDArray or dict - Rotation matrix to use, or dict of rotation matrices for blocks.
    Returns
    =======
    M' : NDArray - The rotated matrix
    """
    if isinstance(T, dict):
        from scipy.linalg import block_diag

        sorted_keys = sorted(T.keys())
        T_matrix = block_diag(*(T[k] for k in sorted_keys))
        return np.conj(T_matrix.T) @ M @ T_matrix
    return np.conj(T.T) @ M @ T


def block_diagonalize_hyb(hyb):
    """
    Block diagonalize the hybridization function matrix.

    Parameters
    ----------
    hyb : ndarray of shape (n_freq, n_orb, n_orb)
        The hybridization matrix.

    Returns
    -------
    Q_full : ndarray of shape (n_orb, n_orb)
        The transformation matrix.
    """
    hyb_herm = 1 / 2 * (hyb + np.conj(np.transpose(hyb, (0, 2, 1))))
    blocks = get_blocks(hyb_herm)
    Q_full = np.zeros((hyb.shape[1], hyb.shape[2]), dtype=complex)
    treated_orbitals = 0
    for block in blocks:
        block_idx = np.ix_(range(hyb.shape[0]), block, block)
        if len(block) == 1:
            Q_full[block_idx[1:], treated_orbitals] = 1
            treated_orbitals += 1
            continue
        block_hyb = hyb_herm[block_idx]
        upper_triangular_hyb = np.triu(hyb_herm, k=1)
        ind_max_offdiag = np.unravel_index(np.argmax(np.abs(upper_triangular_hyb)), upper_triangular_hyb.shape)
        eigvals, Q = np.linalg.eigh(block_hyb[ind_max_offdiag[0], :, :])
        sorted_indices = np.argsort(eigvals)
        Q = Q[:, sorted_indices]
        for column in range(Q.shape[1]):
            j = np.argmax(np.abs(Q[:, column]))
            Q_full[block, treated_orbitals + column] = Q[:, column] * abs(Q[j, column]) / Q[j, column]
        treated_orbitals += Q.shape[1]
    phase_hyb = np.conj(Q_full.T)[np.newaxis, :, :] @ hyb @ Q_full[np.newaxis, :, :]
    return phase_hyb, Q_full


def rotate_Greens_function(G, T):
    r"""
    Rotate the Greens function, G, using the matrix T.
    Returns G'(\omega) = T^\dagger G(\omega) T
    Parameters
    ==========
    G : NDArray - Greens function to rotate
    T : NDArray - Rotation matrix to use
    Returns
    =======
    G' : NDArray - The rotated Greens function
    """
    return np.conj(T.T)[np.newaxis, :, :] @ G @ T[np.newaxis, :, :]


def rotate_4index_U(U4, T):
    r"""
    Rotate the four index tensor, U4, using the matrix T.
    Returns U4' = T^\daggerT^\dagger U4 TT
    Parameters
    ==========
    U4 : NDArray - Tensor function to rotate
    T : NDArray - Rotation matrix to use
    Returns
    =======
    U4' : NDArray - The rotated tensor function
    """
    return np.einsum("ij,kl, jlmo, mn, op", np.conj(T.T), np.conj(T.T), U4, T, T)


def save_Greens_function(gs, omega_mesh, label, cluster_label, e_scale=1, tol=1e-8):
    """
    Save Greens function to file, using RSPt .dat format. Including offdiagonal elements.
    """
    n_orb = gs.shape[1]
    axis_label = "-realaxis"
    if np.all(np.abs(np.imag(omega_mesh)) > 1e-6):
        omega_mesh = np.imag(omega_mesh)
        axis_label = ""

    off_diags = []
    for column in range(gs.shape[2]):
        for row in range(gs.shape[1]):
            if row == column:
                continue
            if np.any(np.abs(gs[:, row, column]) > tol):
                off_diags.append((row, column))

    print(f"Writing {label}{axis_label}-{cluster_label} to files")
    with open(f"real-{label}{axis_label}-{cluster_label}.dat", "w") as fg_real, open(
        f"imag-{label}{axis_label}-{cluster_label}.dat", "w"
    ) as fg_imag:
        header = "# Frequency, total, spin down, spin up\n"
        header += "# indexmap: (column index of projected elements)"
        for row in range(gs.shape[1]):
            header += "\n# "
            for column in range(gs.shape[2]):
                if row == column:
                    header += f"{5 + row:< 4d}"
                elif (row, column) in off_diags:
                    header += f"{5 + n_orb + off_diags.index((row, column)):< 4d}"
                else:
                    header += f"{0:< 4d}"
        fg_real.write(header + "\n")
        fg_imag.write(header + "\n")
        for i, w in enumerate(omega_mesh):
            fg_real.write(
                f"{w * e_scale} {np.real(np.sum(np.diag(gs[i, :, :]))) / e_scale} "
                + f"{np.real(np.sum(np.diag(gs[i, : n_orb // 2, : n_orb // 2]))) / e_scale} "
                + f"{np.real(np.sum(np.diag(gs[i, n_orb // 2 :, n_orb // 2 :]))) / e_scale} "
                + " ".join(f"{np.real(el) / e_scale}" for el in np.diag(gs[i, :, :]))
                + " "
                + " ".join(f"{np.real(gs[i, row, column]) / e_scale}" for row, column in off_diags)
                + "\n"
            )
            fg_imag.write(
                f"{w * e_scale} {np.imag(np.sum(np.diag(gs[i, :, :]))) / e_scale} "
                + f"{np.imag(np.sum(np.diag(gs[i, : n_orb // 2, : n_orb // 2]))) / e_scale} "
                + f"{np.imag(np.sum(np.diag(gs[i, n_orb // 2 :, n_orb // 2 :]))) / e_scale} "
                + " ".join(f"{np.imag(el) / e_scale}" for el in np.diag(gs[i, :, :]))
                + " "
                + " ".join(f"{np.imag(gs[i, row, column]) / e_scale}" for row, column in off_diags)
                + "\n"
            )
