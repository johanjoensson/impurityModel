import numpy as np
from mpi4py import MPI

from impurityModel.ed.BlockLanczosArray import block_add_scaled, block_apply, block_inner, is_array
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import inner


def block_bicgstab(A, x0, y, basis: Basis, slaterWeightMin: float, atol=1e-8, rtol=1e-12, **kwargs):
    """
    Solve a linear system with Block BiCGSTAB using fully sparse ManyBodyStates.

    Parameters
    ----------
    A : dict
        Many-body operator.
    x0 : list
        Initial guess states.
    y : list
        Right-hand side states.
    basis : Basis
        The many-body state basis object.
    slaterWeightMin : float
        Slater determinant cutoff weight.
    atol : float, optional
        Absolute tolerance. Default is 1e-8.
    rtol : float, optional
        Relative tolerance. Default is 1e-12.

    Returns
    -------
    list
        The solved solution states.
    """
    n = x0.shape[1] if is_array(x0) and len(x0.shape) == 2 else len(x0)
    is_arr = is_array(x0)
    mpi = basis is not None and getattr(basis, "is_distributed", False)
    comm = basis.comm if mpi else None

    if not is_arr and hasattr(A, "set_restrictions"):
        A.set_restrictions(basis.restrictions)

    def b_inner(B1, B2):
        return block_inner(B1, B2, mpi=mpi, comm=comm)

    def matmat(v):
        return block_apply(A, v, basis=basis, mpi=mpi, slaterWeightMin=slaterWeightMin)

    xi = x0.copy() if is_arr else [st.copy() for st in x0]

    Axi = matmat(xi)
    ri = y.copy() if is_arr else [st.copy() for st in y]
    block_add_scaled(ri, Axi, -np.eye(n, dtype=complex), slaterWeightMin=slaterWeightMin)

    if not is_arr:
        basis.add_states(
            state
            for r in ri
            for state, amp in r.items()
            if abs(amp) > slaterWeightMin and state not in basis.local_basis
        )

    r0_t = ri.copy() if is_arr else [st.copy() for st in ri]
    pi = ri.copy() if is_arr else [st.copy() for st in ri]

    def block_norm(v):
        if is_arr:
            norms = np.linalg.norm(v, axis=0)
            return np.max(norms)
        else:
            norms = np.array([inner(vi, vi).real for vi in v])
            if mpi:
                comm.Allreduce(MPI.IN_PLACE, norms, op=MPI.SUM)
            return np.sqrt(np.max(norms))

    r0_norm = block_norm(r0_t)
    if r0_norm < np.finfo(float).eps:
        return x0

    max_iter = kwargs.get("max_iter", np.inf)
    if not is_arr:
        seen_states = set()
        for state in x0:
            seen_states.update(state.keys())
        for state in y:
            seen_states.update(state.keys())
        global_seen_size = np.array([len(seen_states)], dtype=int)
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, global_seen_size, op=MPI.SUM)
    else:
        global_seen_size = np.array([np.inf])

    it = 0
    active_mask = np.ones(n, dtype=bool)

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

        # Deflate converged columns
        R0_V[~active_mask, :] = 0
        R0_V[:, ~active_mask] = 0
        R0_V[~active_mask, ~active_mask] = 1.0
        R0_R[:, ~active_mask] = 0

        if np.linalg.cond(R0_V) > 1 / np.finfo(float).eps:
            break

        ai = np.linalg.solve(R0_V, R0_R)

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

        if abs(tt) < np.finfo(float).eps:
            wi = 0.0
        else:
            wi = ts / tt

        xip = xi.copy() if is_arr else [st.copy() for st in xi]
        block_add_scaled(xip, si, wi * np.eye(n, dtype=complex), slaterWeightMin=slaterWeightMin)
        block_add_scaled(xip, pi, ai, slaterWeightMin=slaterWeightMin)

        rip = si.copy() if is_arr else [st.copy() for st in si]
        block_add_scaled(rip, ti, -wi * np.eye(n, dtype=complex), slaterWeightMin=slaterWeightMin)

        R0_T = b_inner(r0_t, ti)
        R0_T[:, ~active_mask_s] = 0
        bi = np.linalg.solve(R0_V, -R0_T)

        pip = rip.copy() if is_arr else [st.copy() for st in rip]
        block_add_scaled(pip, pi, bi, slaterWeightMin=slaterWeightMin)
        block_add_scaled(pip, vi, -wi * bi, slaterWeightMin=slaterWeightMin)

        xi = xip
        ri = rip
        pi = pip

    return xi
