import numpy as np
import scipy.linalg as sp
from mpi4py import MPI

from impurityModel.ed.BlockLanczosArray import (
    block_apply,
    block_combine,
    block_inner,
    block_normalize,
    block_orthogonalize,
    is_array,
)
from impurityModel.ed.BlockLanczosArray import Reort
from impurityModel.ed.ManyBodyUtils import inner_multi


def thick_restart_block_lanczos(
    psi0,
    h_op,
    basis,
    num_wanted: int,
    max_subspace_blocks: int,
    tol: float = 1e-8,
    max_restarts: int = 100,
    verbose: bool = True,
    slaterWeightMin: float = 0,
    reort: Reort = Reort.PARTIAL,
):
    mpi = basis is not None and getattr(basis, "comm", None) is not None
    comm = basis.comm if mpi else None

    if isinstance(reort, str):
        reort = Reort[reort.upper()]

    is_arr = is_array(h_op)
    n = psi0.shape[1] if is_arr and len(psi0.shape) == 2 else len(psi0)

    k_blocks = int(np.ceil(num_wanted / n))
    m = max_subspace_blocks

    track_W = reort in (Reort.PARTIAL, Reort.SELECTIVE)

    if not is_arr:
        from impurityModel.ed.BlockLanczos import thick_restart_block_lanczos_cy
        return thick_restart_block_lanczos_cy(
            psi0=psi0,
            h_op=h_op,
            basis=basis,
            num_wanted=num_wanted,
            max_subspace_blocks=max_subspace_blocks,
            tol=tol,
            max_restarts=max_restarts,
            verbose=verbose,
            slaterWeightMin=slaterWeightMin,
            reort=reort,
            comm=comm,
        )

    from impurityModel.ed.BlockLanczosArray import block_lanczos_array

    # block_lanczos_array assumes an orthonormal start block (it does not normalize
    # internally). An unnormalized psi0 makes the three-term recurrence operate on
    # non-unit vectors, so the betas grow geometrically and T overflows within ~10-15
    # steps. The IRLM driver normalizes before sweeping; do the same here.
    psi0 = np.ascontiguousarray(psi0 if (is_arr and psi0.ndim == 2) else np.reshape(psi0, (-1, 1)), dtype=complex)
    psi0, _ = block_normalize(psi0, mpi, comm, 0.0)

    alphas, betas, Q_list, *rest = block_lanczos_array(
        psi0=psi0,
        h_op=h_op,
        converged=lambda a, b, **kw: False,
        max_iter=m,
        verbose=verbose,
        reort=reort,
        return_W=track_W,
        return_widths=True,
        comm=comm,
    )
    # block_lanczos_array returns (..., [W], block_widths); W only when return_W.
    widths = rest[-1]
    W = rest[0] if track_W and len(rest) > 1 else None

    m_actual = len(alphas)
    # The sweep can shrink blocks (rank-deficient beta -> deflation), so the true
    # subspace dimension is sum(widths), not the padded m_actual * n. All slicing and
    # T construction must use the real widths or they desynchronize from Q_list.
    total = int(sum(widths)) if widths is not None else m_actual * n
    deflated = total < m_actual * n

    if Q_list.shape[1] > total:
        q_m = Q_list[:, total : total + n].copy()
        Q_list = Q_list[:, :total]
    else:
        q_m = None

    from impurityModel.ed.BlockLanczosArray import _build_full_T

    _betas_off = betas[: m_actual - 1] if len(betas) == m_actual else betas
    T_full = _build_full_T(alphas, _betas_off, block_widths=widths)

    if m_actual < m or deflated:
        # Either a genuine invariant subspace (fewer blocks than asked) or block
        # deflation (rank-deficient residual): in both cases the spanned block-Krylov
        # space is (near-)invariant, so its Ritz pairs are accurate eigenpairs and we
        # extract directly instead of thick-restarting. Crucially this also avoids the
        # uniform-width restart loop below, whose arrowhead/T_full bookkeeping assumes a
        # constant block width n and is invalid once the blocks have shrunk.
        if verbose and (not mpi or comm.rank == 0):
            reason = "Invariant subspace" if m_actual < m else "Block deflation"
            print(f"{reason} (dim {total}). Extracting directly.")
        eigvals, eigvecs = sp.eigh(T_full)
        wanted_indices = np.argsort(eigvals)[:num_wanted]
        final_eigvals = eigvals[wanted_indices]

        final_eigvecs = block_combine(Q_list, eigvecs[:, wanted_indices], slaterWeightMin)
        return final_eigvals, final_eigvecs

    # --- Width-aware thick restart -------------------------------------------------
    # The blocks can shrink mid-restart (rank-deficient residual -> block_normalize
    # returns a rectangular beta and a narrower q_next). The loop therefore tracks the
    # actual width of every block (``cur_widths``) and addresses T_full / Q_list by
    # cumulative column offsets instead of the constant ``n``. nkeep = k_blocks * n is a
    # *count* of retained Ritz vectors (a scalar, not a block width), kept as one
    # diagonal super-block coupled to the residual by the thick-restart spike.
    nkeep = k_blocks * n
    p_resid = q_m.shape[1] if q_m is not None else n
    # betas[-1] is the trailing coupling block, padded to (n, n) by the kernel. The
    # *residual* block can deflate (rank p_resid < n) even when the diagonal blocks do
    # not -- that leaves total == m_actual*n, so deflated is False and we still enter the
    # restart loop, but q_m has only p_resid columns. Slice off the phantom padded rows
    # so beta_res is the true (p_resid, n) coupling.
    beta_res = betas[-1][:p_resid, :]
    cur_widths = list(widths)  # block widths of the current factorization (all n here)

    def _extract(T, Q, dim):
        e, V = sp.eigh(T[:dim, :dim])
        idx = np.argsort(e)[:num_wanted]
        return e[idx], block_combine(Q[:, :dim], V[:, idx], slaterWeightMin)

    for restart in range(max_restarts):
        D = int(sum(cur_widths))
        p_last = cur_widths[-1]
        eigvals, eigvecs = sp.eigh(T_full[:D, :D])
        res_norms = np.linalg.norm(beta_res @ eigvecs[D - p_last : D, :], axis=0)
        wanted_indices = np.argsort(eigvals)[:num_wanted]
        max_wanted_res = np.max(res_norms[wanted_indices])

        if verbose and (not mpi or comm.rank == 0):
            print(f"Restart {restart:3d} | Min Eigval: {eigvals[0]:.6f} | Max Wanted Residual: {max_wanted_res:.2e}")

        if max_wanted_res < tol:
            if verbose:
                print("Converged!")
            break

        keep_indices = np.argsort(eigvals)[:nkeep]
        Y_k = eigvecs[:, keep_indices]
        Y_last = Y_k[D - p_last : D, :]  # last-block rows -> thick-restart spike
        T_k = np.diag(eigvals[keep_indices])

        Q_ret = block_combine(Q_list[:, :D], Y_k, 0.0)
        Q_ret, _ = block_normalize(Q_ret, mpi, comm, 0.0)

        # Worst case the continuation adds (m - k_blocks) full-width-n blocks.
        T_full = np.zeros((nkeep + (m - k_blocks) * n, nkeep + (m - k_blocks) * n), dtype=complex)
        T_full[:nkeep, :nkeep] = T_k

        # Residual seed block. Normally carried over as q_m; recompute if absent.
        if q_m is None:
            q_seed = Q_ret[:, max(0, nkeep - n) : nkeep]
            wp = block_apply(h_op, q_seed, basis, mpi)
            # Thick-restart always full-reorthogonalizes the residual seed against the
            # whole retained basis (all modes): the arrowhead T_full requires it, and
            # the PRO W-recurrence is not maintained across restart.
            wp, _ = block_orthogonalize(wp, Q_ret, mpi=mpi, comm=comm)
            wp, _ = block_orthogonalize(wp, Q_ret, mpi=mpi, comm=comm)
            q_m, beta_res = block_normalize(wp, mpi, comm, 0.0)
            p_resid = q_m.shape[1]

        Q_list = np.concatenate([Q_ret, q_m], axis=1)
        cross_term = beta_res @ Y_last  # (p_resid, nkeep)
        T_full[nkeep : nkeep + p_resid, :nkeep] = cross_term
        T_full[:nkeep, nkeep : nkeep + p_resid] = np.conj(cross_term.T)

        cur_widths = [nkeep, p_resid]
        off = nkeep  # column start of the current block q1
        w1 = p_resid
        q1 = q_m
        q_m = None  # consumed; the new trailing residual is set at the last inner step

        for i in range(k_blocks, m):
            wp = block_apply(h_op, q1, basis, mpi)

            overlaps = block_inner(Q_list, wp, mpi, comm)
            alpha_i = overlaps[-w1:, :]  # q1^H H q1  (w1, w1)
            T_full[off : off + w1, off : off + w1] = alpha_i

            wp, _ = block_orthogonalize(wp, Q_list, overlaps=overlaps, mpi=mpi, comm=comm)
            wp, _ = block_orthogonalize(wp, Q_list, mpi=mpi, comm=comm)

            try:
                q_next, beta_i = block_normalize(wp, mpi, comm, 0.0)
            except ValueError:
                q_next = None

            # Full collapse, or a near-invariant subspace: extract from what we have.
            if q_next is None or np.linalg.norm(beta_i, ord=2) < 1e-5:
                if verbose and (not mpi or comm.rank == 0):
                    print(f"Invariant subspace found during restart at block {i}. Stopping early.")
                return _extract(T_full, Q_list, off + w1)

            # Partial deflation shrinks the block: beta_i is (w_next, w1), q_next has
            # w_next <= w1 columns. The arrowhead placement below uses those widths.
            w_next = q_next.shape[1]
            if i < m - 1:
                T_full[off + w1 : off + w1 + w_next, off : off + w1] = beta_i
                T_full[off : off + w1, off + w1 : off + w1 + w_next] = np.conj(beta_i.T)

                Q_list = np.concatenate([Q_list, q_next], axis=1)
                cur_widths.append(w_next)
                off += w1
                w1 = w_next
                q1 = q_next
            else:
                beta_res = beta_i
                q_m = q_next
                p_resid = w_next

    final_eigvals, final_eigvecs = _extract(T_full, Q_list, int(sum(cur_widths)))

    if verbose and (not mpi or comm.rank == 0):
        print(f"Final eigvals:\n{final_eigvals}")

    return final_eigvals, final_eigvecs


