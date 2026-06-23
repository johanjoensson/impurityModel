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
)
from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyOperator,
    ManyBodyState,
)
from impurityModel.ed.mpi_comm import gather_distributed_results

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
):
    """
    Calculate interacting Greens function.
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
        if omega_mesh is not None and block_basis.comm.rank == 0:
            G_IPS = calc_thermally_averaged_G(alphas_IPS, betas_IPS, r_IPS, omega_mesh, es, e0, tau, delta)
            G_PS = calc_thermally_averaged_G(alphas_PS, betas_PS, r_PS, -omega_mesh, es, e0, tau, -delta)
            local_gs_realaxis.append(
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
    if basis.comm.rank == 0:
        gs_matsubara = [np.empty((len(matsubara_mesh), len(block), len(block)), dtype=complex) for block in blocks]
        gs_realaxis = [np.empty((len(omega_mesh), len(block), len(block)), dtype=complex) for block in blocks]
        for i, block_idx in enumerate(block_indices_per_color):
            gs_matsubara[block_idx][:] = gathered_matsubara[i]
            gs_realaxis[block_idx][:] = gathered_realaxis[i]

    # Free the split communicator collectively before returning.
    # block_basis.comm is a split comm created by split_basis_and_redistribute_psi.
    # MPI_Comm_free is collective — it must be called by all ranks in the comm
    # at the same time.  Leaving it for Python gc risks non-collective freeing.
    if block_basis is not None and block_basis.comm != basis.comm:
        block_basis.free_comm()

    return (gs_matsubara, gs_realaxis) if basis.comm.rank == 0 else (None, None)


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
    block_v = [[ManyBodyState({}) for _ in tOps] for _ in psis]
    for i_tOp, tOp in enumerate(tOps):
        tOp.set_restrictions(excited_restrictions)
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
            verbose=False,
        )

        if excited_basis.restrictions is not None:
            hOp.set_restrictions(excited_basis.restrictions)
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

    alphas, betas, r, last_q = block_green_impl(
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
        alphas, betas, r, last_q = block_green_impl(
            basis, hOp, basis.redistribute_psis(psi_arr), delta, slaterWeightMin, verbose
        )

        n_test = min(alphas.shape[0], alphas_prev.shape[0])
        # relatively large changes in alpha and/or betas means we have not converged
        if np.any(np.abs(alphas[:n_test] - alphas_prev[:n_test]) > 1e-12) or np.any(
            np.abs(betas[:n_test] - betas_prev[:n_test]) > 1e-12
        ):
            done = False
            continue

        # alphas seem decently converged, check the Greens function to be sure
        ws = np.diagonal(alphas, axis1=1, axis2=2).flat[: n_test * alphas.shape[1]]
        G_prev = calc_G(alphas_prev, betas_prev, np.identity(n), ws, 0, delta)
        G = calc_G(alphas, betas, np.identity(n), ws, 0, delta)
        done = (
            np.all(np.diagonal(G.imag, axis1=1, axis2=2) * np.sign(delta) <= 0) and np.max(np.abs(G - G_prev)) < 1e-12
        )
    return alphas, betas, r


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
        r = comm.bcast(r if rank == 0 else None, root=0)
        rows, columns = comm.bcast(psi_dense.shape if rank == 0 else None, root=0)
        psi_dense_local = np.empty((len(basis.local_basis), columns), dtype=complex, order="C")
        send_counts = np.empty((comm.size), dtype=int) if rank == 0 else None
        comm.Gather(np.array([psi_dense_local.size]), send_counts, root=0)
        offsets = np.array([np.sum(send_counts[:r]) for r in range(comm.size)], dtype=int) if rank == 0 else None
        comm.Scatterv(
            [psi_dense, send_counts, offsets, MPI.C_DOUBLE_COMPLEX] if rank == 0 else None,
            psi_dense_local,
            root=0,
        )

    if psi_dense_local.shape[1] == 0:
        return np.zeros((0, n, n), dtype=complex), np.zeros((0, n, n), dtype=complex), r, psi_arr

    delta_min = max(slaterWeightMin**2, 1e-8)

    def converged(alphas, betas, verbose=False):
        """
        Check convergence of the block Lanczos algorithm.

        It monitors convergence by estimating the Green's function at selected
        frequencies and checking whether the maximum change between subsequent
        iterations is within a specified tolerance.

        Parameters
        ----------
        alphas : ndarray
            The diagonal block matrices (alpha) generated by block Lanczos.
        betas : ndarray
            The off-diagonal block matrices (beta) generated by block Lanczos.
        verbose : bool, optional
            If True, print the convergence difference at each step.

        Returns
        -------
        converged : bool
            True if the calculation has converged, False otherwise.
        """
        if alphas.shape[0] <= 1:
            return False

        # For scalar valued Lanczos, calculate convergents to check for convergence.
        if n == 1 and False:
            An, Bn = calc_continuants(alphas[-1] + 1j * delta - alphas, betas)
            if abs(Bn[-1]) < 1e-6:
                return (
                    abs(Bn[-1] * An[-2] - An[-1] * Bn[-2]) <= abs(delta_min * Bn[-1] * Bn[-2])
                    and An[-1].imag * Bn[-1].real - An[-1].real * Bn[-1].imag <= 0
                )

            return abs(An[-2] / Bn[-2] - An[-1] / Bn[-1]) < delta_min and (An[-1] / Bn[-1]).imag <= 0

        # For matrix valued (block) Lanczos, continued fractions are harder to estimate convergence for
        wIs = (np.diagonal(alphas, axis1=1, axis2=2).flat[: 15 * alphas.shape[1]] + delta * 1j)[
            :, None, None
        ] * np.identity(alphas.shape[1], dtype=complex)[np.newaxis]
        gs_new = wIs - alphas[-1]
        gs_new = (
            wIs
            - alphas[-2]
            - np.conj(betas[-2].T)[np.newaxis, :, :] @ np.linalg.solve(gs_new, betas[-2][np.newaxis, :, :])
        )
        gs_prev = wIs - alphas[-2]
        for alpha, beta in zip(alphas[-3::-1], betas[-3::-1]):
            gs_new = wIs - alpha - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve(gs_new, beta[np.newaxis, :, :])
            gs_prev = wIs - alpha - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve(gs_prev, beta[np.newaxis, :, :])

        if np.any(np.diagonal(gs_new.imag, axis1=1, axis2=2) * np.sign(delta) < 0):
            return False
        d_g = np.max(np.abs(gs_new - gs_prev))
        return d_g < delta_min

    if dense:
        H = basis.build_dense_matrix(hOp)
        alphas, betas, Q_list, *_ = block_lanczos_array(
            psi0=psi_dense_local,
            h_op=H,
            converged=converged,
            verbose=False and verbose,
            reort=reort if reort is not None else Reort.NONE,
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
        alphas, betas, Q_list, *_ = block_lanczos_array(
            psi0=psi_dense_local,
            h_op=H,
            converged=converged,
            reort=reort if reort is not None else Reort.NONE,
            verbose=False and verbose,
            comm=comm,
        )
    q_last = Q_list[:, -1:]
    return alphas, betas, r, basis.build_state(q_last.T, slaterWeightMin=slaterWeightMin)


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
        r = comm.bcast(r if rank == 0 else None, root=0)
        rows, columns = comm.bcast(psi_dense.shape if rank == 0 else None, root=0)
        psi_dense_local = np.empty((len(basis.local_basis), columns), dtype=complex, order="C")
        send_counts = np.empty((comm.size), dtype=int) if rank == 0 else None
        comm.Gather(np.array([psi_dense_local.size]), send_counts, root=0)
        offsets = np.array([np.sum(send_counts[:r]) for r in range(comm.size)], dtype=int) if rank == 0 else None
        comm.Scatterv(
            [psi_dense, send_counts, offsets, MPI.C_DOUBLE_COMPLEX] if rank == 0 else None,
            psi_dense_local,
            root=0,
        )
    else:
        psi_dense_local = psi_dense
    psi_arr = basis.build_state(psi_dense_local.T, slaterWeightMin=0)
    if len(psi_arr) == 0:
        return np.empty((0, n, n), dtype=complex), np.empty((0, n, n), dtype=complex), r

    delta_min = max(slaterWeightMin**2, 1e-8)

    def converged(alphas, betas, verbose=False):
        """
        Check convergence of the block Lanczos algorithm.

        It monitors convergence by estimating the Green's function at selected
        frequencies and checking whether the maximum change between subsequent
        iterations is within a specified tolerance.

        Parameters
        ----------
        alphas : ndarray
            The diagonal block matrices (alpha) generated by block Lanczos.
        betas : ndarray
            The off-diagonal block matrices (beta) generated by block Lanczos.
        verbose : bool, optional
            If True, print the convergence difference at each step.

        Returns
        -------
        converged : bool
            True if the calculation has converged, False otherwise.
        """
        if alphas.shape[0] <= 1:
            return False

        # For scalar valued Lanczos, calculate convergents to check for convergence.
        if n == 1 and False:
            An, Bn = calc_continuants(alphas[-1] + 1j * delta - alphas, betas)
            if abs(Bn[-1]) < 1e-6:
                return (
                    abs(Bn[-1] * An[-2] - An[-1] * Bn[-2]) <= abs(delta_min * Bn[-1] * Bn[-2])
                    and An[-1].imag * Bn[-1].real - An[-1].real * Bn[-1].imag <= 0
                )

            return abs(An[-2] / Bn[-2] - An[-1] / Bn[-1]) < delta_min and (An[-1] / Bn[-1]).imag <= 0

        # For matrix valued (block) Lanczos, continued fractions are harder to estimate convergence for
        wIs = (np.diagonal(alphas, axis1=1, axis2=2).flat[: 15 * alphas.shape[1]] + delta * 1j)[
            :, None, None
        ] * np.identity(alphas.shape[1], dtype=complex)[np.newaxis]
        gs_new = wIs - alphas[-1]
        gs_new = (
            wIs
            - alphas[-2]
            - np.conj(betas[-2].T)[np.newaxis, :, :] @ np.linalg.solve(gs_new, betas[-2][np.newaxis, :, :])
        )
        gs_prev = wIs - alphas[-2]
        for alpha, beta in zip(alphas[-3::-1], betas[-3::-1]):
            gs_new = wIs - alpha - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve(gs_new, beta[np.newaxis, :, :])
            gs_prev = wIs - alpha - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve(gs_prev, beta[np.newaxis, :, :])

        if np.any(np.diagonal(gs_new.imag, axis1=1, axis2=2) * np.sign(delta) < 0):
            return False
        d_g = np.max(np.abs(gs_new - gs_prev))
        print(rf"$\delta$ = {d_g}", flush=True)
        return d_g < delta_min

    alphas, betas, _, _ = block_lanczos_cy(
        psi_arr,
        hOp,
        basis,
        converged,
        verbose=verbose,
        reort=reort if reort is not None else Reort.NONE,
        slaterWeightMin=slaterWeightMin,
    )

    return alphas, betas, r


def calc_G(alphas, betas, r, omega, e, delta):
    """
    Calculate the Green's function using continued fraction parameters.

    Parameters
    ----------
    alphas : ndarray
        Alpha continued fraction coefficients.
    betas : ndarray
        Beta continued fraction coefficients.
    r : ndarray
        R matrix projection.
    omega : ndarray
        Frequency mesh.
    e : float
        Energy offset.
    delta : float
        Broadening factor.

    Returns
    -------
    G : ndarray
        Calculated Green's function.
    """
    if alphas.shape[0] == 0:
        return np.zeros((len(omega), alphas.shape[1], alphas.shape[1]), dtype=complex)
    I = np.identity(alphas.shape[1], dtype=complex)
    omegaP = omega + 1j * delta + e
    # G_inv = np.zeros((len(omega), alphas.shape[1], alphas.shape[1]), dtype=complex)
    wIs = omegaP[:, np.newaxis, np.newaxis] * I[np.newaxis, :, :]
    G_inv = wIs - alphas[-1][np.newaxis]
    for alpha, beta in zip(alphas[-2::-1], betas[-2::-1]):
        G_inv = (
            wIs
            - alpha[np.newaxis, :, :]
            - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve(G_inv, beta[np.newaxis, :, :])
        )
    return np.conj(r.T)[np.newaxis] @ np.linalg.solve(G_inv, r[np.newaxis])


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
