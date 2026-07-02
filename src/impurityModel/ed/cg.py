import numpy as np
from mpi4py import MPI

from impurityModel.ed.BlockLanczosArray import (
    _cholesky_or_deflate,
    block_add_scaled,
    block_apply,
    block_combine,
    block_inner,
    is_array,
)
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState, inner


def block_bicgstab(A, x0, y, basis: Basis, slaterWeightMin: float, atol=1e-8, rtol=1e-12, **kwargs):
    """
    Solve a linear system ``A X = Y`` with Block BiCGSTAB.

    Works for dense (``numpy`` array) and fully sparse (``ManyBodyState``) blocks, and for an
    arbitrary -- possibly **rank-deficient** -- right-hand side. A rank-deficient block RHS
    (linearly dependent columns) would otherwise make the block coefficient matrix singular and
    stall the solver at the initial guess; here the initial residual block is deflated to its
    independent directions, the full-rank reduced system is solved, and the dependent columns
    are reconstructed by linearity.

    Parameters
    ----------
    A : ManyBodyOperator or ndarray
        The linear operator.
    x0 : list of ManyBodyState or ndarray
        Initial guess block (warm start).
    y : list of ManyBodyState or ndarray
        Right-hand side block.
    basis : Basis
        The many-body state basis object (``None`` for the dense path).
    slaterWeightMin : float
        Slater determinant cutoff weight.
    atol : float, optional
        Absolute tolerance. Default is 1e-8.
    rtol : float, optional
        Relative tolerance. Default is 1e-12.

    Returns
    -------
    list of ManyBodyState or ndarray
        The solution block ``X``.
    """
    is_arr = is_array(x0)
    n = x0.shape[1] if is_arr and len(x0.shape) == 2 else len(x0)
    mpi = basis is not None and getattr(basis, "is_distributed", False)
    comm = basis.comm if mpi else None

    if not is_arr and hasattr(A, "set_restrictions"):
        A.set_restrictions(basis.restrictions)

    def matmat(v):
        return block_apply(A, v, basis=basis, mpi=mpi, slaterWeightMin=slaterWeightMin)

    # Initial residual block R0 = Y - A x0.
    Axi = matmat(x0.copy() if is_arr else [st.copy() for st in x0])
    ri = y.copy() if is_arr else [st.copy() for st in y]
    block_add_scaled(ri, Axi, -np.eye(n, dtype=complex), slaterWeightMin=slaterWeightMin)
    if not is_arr:
        basis.add_states(
            state
            for r in ri
            for state, amp in r.items()
            if abs(amp) > slaterWeightMin and state not in basis.local_basis
        )

    # Deflate R0 = Q @ beta_j into rank independent directions (Q = R0 @ beta_inv, orthonormal).
    # Solve the full-rank reduced system A Zq = Q, then reconstruct the correction
    # Z = Zq @ beta_j so that A Z = A Zq @ beta_j = Q @ beta_j = R0, exactly (the dependent
    # directions are recovered by linearity, never solved for). Reuses the same Gram-matrix
    # deflation as the block-Lanczos recurrence (:func:`_cholesky_or_deflate`).
    gram = block_inner(ri, ri, mpi=mpi, comm=comm)
    beta_j, beta_inv, rank = _cholesky_or_deflate(gram, n)
    if rank == 0:
        return x0  # zero residual: x0 already solves the system

    q_block = block_combine(ri, beta_inv, slaterWeightMin)
    z_block = _block_bicgstab_core(
        matmat, q_block, basis, slaterWeightMin, atol, rtol, mpi, comm, is_arr, rank,
        kwargs.get("max_iter", np.inf),
    )
    correction = block_combine(z_block, beta_j, slaterWeightMin)
    xi = x0.copy() if is_arr else [st.copy() for st in x0]
    block_add_scaled(xi, correction, np.eye(n, dtype=complex), slaterWeightMin=slaterWeightMin)
    return xi


def _block_bicgstab_core(matmat, rhs, basis, slaterWeightMin, atol, rtol, mpi, comm, is_arr, n, max_iter):
    """Block BiCGSTAB inner iteration for a **full-rank** RHS block with a zero initial guess.

    Assumes ``rhs`` has full column rank (guaranteed by the deflation in
    :func:`block_bicgstab`), so the block coefficient systems are well posed. The active-column
    coefficient solves use least squares (robust to any rank loss that still develops as columns
    converge at different rates), and the loop exits only on genuine convergence, ``max_iter``,
    or basis exhaustion -- never discarding accumulated progress on a conditioning number.
    """

    def b_inner(B1, B2):
        return block_inner(B1, B2, mpi=mpi, comm=comm)

    xi = np.zeros_like(rhs) if is_arr else [ManyBodyState() for _ in range(n)]
    ri = rhs.copy() if is_arr else [st.copy() for st in rhs]  # A x0 = 0 -> residual is the RHS
    r0_t = ri.copy() if is_arr else [st.copy() for st in ri]
    pi = ri.copy() if is_arr else [st.copy() for st in ri]

    def block_norm(v):
        if is_arr:
            return np.max(np.linalg.norm(v, axis=0))
        norms = np.array([inner(vi, vi).real for vi in v])
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, norms, op=MPI.SUM)
        return np.sqrt(np.max(norms))

    r0_norm = block_norm(r0_t)
    if r0_norm < np.finfo(float).eps:
        return xi

    if not is_arr:
        seen_states = set()
        for state in rhs:
            seen_states.update(state.keys())
        global_seen_size = np.array([len(seen_states)], dtype=int)
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, global_seen_size, op=MPI.SUM)
    else:
        global_seen_size = np.array([np.inf])

    def masked_lstsq(mat, rhs_mat, active):
        # Solve mat @ X = rhs_mat restricted to active rows/cols (inactive rows of X -> 0),
        # via least squares so a singular active sub-block is handled gracefully.
        out = np.zeros_like(rhs_mat)
        act = np.where(active)[0]
        if act.size:
            out[np.ix_(act, np.arange(out.shape[1]))] = np.linalg.lstsq(
                mat[np.ix_(act, act)], rhs_mat[act, :], rcond=None
            )[0]
        return out

    it = 0
    while it * n < global_seen_size[0] and it < max_iter:
        it += 1

        if is_arr:
            r_norms2 = np.linalg.norm(ri, axis=0) ** 2
        else:
            r_norms2 = np.array([inner(vi, vi).real for vi in ri])
            if mpi:
                comm.Allreduce(MPI.IN_PLACE, r_norms2, op=MPI.SUM)
        r_norms = np.sqrt(r_norms2)

        active_mask = (r_norms >= atol) & (r_norms / r0_norm >= rtol)
        if not np.any(active_mask):
            break

        vi = matmat(pi)

        if not is_arr:
            basis.add_states(
                state
                for v in vi
                for state, amp in v.items()
                if abs(amp) > slaterWeightMin and state not in basis.local_basis
            )
            for state in vi:
                seen_states.update(state.keys())
            global_seen_size[0] = len(seen_states)
            if mpi:
                comm.Allreduce(MPI.IN_PLACE, global_seen_size, op=MPI.SUM)

        R0_V = b_inner(r0_t, vi)
        R0_R = b_inner(r0_t, ri)
        R0_R[:, ~active_mask] = 0
        ai = masked_lstsq(R0_V, R0_R, active_mask)

        si = ri.copy() if is_arr else [r.copy() for r in ri]
        block_add_scaled(si, vi, -ai, slaterWeightMin=slaterWeightMin)

        if is_arr:
            s_norms2 = np.linalg.norm(si, axis=0) ** 2
        else:
            s_norms2 = np.array([inner(vsi, vsi).real for vsi in si])
            if mpi:
                comm.Allreduce(MPI.IN_PLACE, s_norms2, op=MPI.SUM)

        active_mask_s = np.sqrt(s_norms2) >= atol
        if not np.any(active_mask_s):
            xip = xi.copy() if is_arr else [st.copy() for st in xi]
            block_add_scaled(xip, pi, ai, slaterWeightMin=slaterWeightMin)
            xi = xip
            break

        ti = matmat(si)
        if not is_arr:
            basis.add_states(state for t in ti for state in t if state not in basis.local_basis)
            for state in ti:
                seen_states.update(state.keys())
            global_seen_size[0] = len(seen_states)
            if mpi:
                comm.Allreduce(MPI.IN_PLACE, global_seen_size, op=MPI.SUM)

        if is_arr:
            ts = np.sum(np.conj(ti[:, active_mask_s]) * si[:, active_mask_s], axis=0).sum()
            tt = np.sum(np.conj(ti[:, active_mask_s]) * ti[:, active_mask_s], axis=0).sum()
        else:
            ts = sum(inner(ti[j], si[j]) for j in range(n) if active_mask_s[j])
            tt = sum(inner(ti[j], ti[j]) for j in range(n) if active_mask_s[j])
            if mpi:
                ts_arr = np.array(ts, dtype=complex)
                tt_arr = np.array(tt, dtype=complex)
                comm.Allreduce(MPI.IN_PLACE, ts_arr, op=MPI.SUM)
                comm.Allreduce(MPI.IN_PLACE, tt_arr, op=MPI.SUM)
                ts = ts_arr.item()
                tt = tt_arr.item()

        wi = 0.0 if abs(tt) < np.finfo(float).eps else ts / tt

        xip = xi.copy() if is_arr else [st.copy() for st in xi]
        block_add_scaled(xip, si, wi * np.eye(n, dtype=complex), slaterWeightMin=slaterWeightMin)
        block_add_scaled(xip, pi, ai, slaterWeightMin=slaterWeightMin)

        rip = si.copy() if is_arr else [st.copy() for st in si]
        block_add_scaled(rip, ti, -wi * np.eye(n, dtype=complex), slaterWeightMin=slaterWeightMin)

        R0_T = b_inner(r0_t, ti)
        R0_T[:, ~active_mask_s] = 0
        bi = masked_lstsq(R0_V, -R0_T, active_mask)

        pip = rip.copy() if is_arr else [st.copy() for st in rip]
        block_add_scaled(pip, pi, bi, slaterWeightMin=slaterWeightMin)
        block_add_scaled(pip, vi, -wi * bi, slaterWeightMin=slaterWeightMin)

        xi = xip
        ri = rip
        pi = pip

    return xi
