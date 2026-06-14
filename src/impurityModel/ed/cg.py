import numpy as np
import scipy as sp

import itertools
import impurityModel.ed.finite as finite
from mpi4py import MPI
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState, applyOp, inner, inner_multi, add_scaled_multi


def cg(A, x, y, atol=1e-5):
    """
    Solve the linear system A * x = y using the Conjugate Gradient (CG) method.

    Parameters
    ----------
    A : ndarray or sparse matrix
        The linear operator/matrix of the system.
    x : ndarray
        Initial guess for the solution, updated in-place.
    y : ndarray
        Right-hand side vector.
    atol : float, optional
        Absolute tolerance for convergence. Default is 1e-5.

    Returns
    -------
    x : ndarray
        The solved solution vector.
    info : int
        Convergence status (0 for success, -1 if max iterations reached).
    """
    info = -1
    n = A.shape[0]
    r = y - A @ x
    p = r.copy()
    prev_r2 = abs(np.vdot(r, r))
    for i in range(10 * n):
        r_i = A @ r
        dr = prev_r2**2 / (np.vdot(r, r_i))
        x += dr * r
        r -= dr * r_i
        r2 = abs(np.vdot(r, r))
        if r2 < atol**2:
            info = 0
            break
        p = r + (r2 / prev_r2) * p
        prev_r2 = r2
    return x, info


def cg_2(A, x, y, atol=1e-5):
    """
    Alternative Conjugate Gradient solver formulation.

    Parameters
    ----------
    A : ndarray or sparse matrix
        The linear operator/matrix.
    x : ndarray
        Initial guess, updated in-place.
    y : ndarray
        Right-hand side vector.
    atol : float, optional
        Absolute tolerance. Default is 1e-5.

    Returns
    -------
    x : ndarray
        The solved solution vector.
    info : int
        Convergence status.
    """
    info = 0
    n = A.shape[0]
    r = y - A @ x
    p = r.copy()
    for it in range(10 * n):
        p_i = A @ p
        dr = np.vdot(r, r) / np.vdot(r, p_i)
        x += dr * p
        r -= dr * p_i
        r2 = abs(np.vdot(r, r))
        if r2 < atol:
            info = 0
            break
        dp = -np.vdot(p_i, r) / np.vdot(p, p_i)
        p = r + dp * p

    return x, info


def bicgstab(A_op, x_0, y, basis, slaterWeightMin, atol=1e-8, **kwargs):
    """
    Solve a linear system with BiCGSTAB using a many-body state representation.

    Parameters
    ----------
    A_op : dict
        Many-body operator defining the linear system.
    x_0 : list[ManyBodyState]
        Initial guess states.
    y : list[ManyBodyState]
        Right-hand side states.
    basis : Basis
        The many-body state basis object.
    slaterWeightMin : float
        Cutoff weight for Slater determinants.
    atol : float, optional
        Absolute tolerance. Default is 1e-8.

    Returns
    -------
    x_i : list[ManyBodyState]
        The solved solution states.
    """

    n = len(x_0)
    if hasattr(A_op, "set_restrictions"):
        A_op.set_restrictions(basis.restrictions)
    Ax = A_op.apply_multi(x_0, cutoff=slaterWeightMin)
    Ax = basis.redistribute_psis(Ax)

    r_0 = [yi.copy() for yi in y]
    add_scaled_multi(r_0, Ax, -np.eye(n, dtype=complex))

    x_i = [xi.copy() for xi in x_0]
    r_i = [ri.copy() for ri in r_0]
    rho_i = np.array([ri.norm2() for ri in r_i], dtype=complex)
    if basis.is_distributed:
        basis.comm.Allreduce(MPI.IN_PLACE, rho_i)
    max_iter = kwargs.get("max_iter", np.inf)
    seen_states = set()
    for state in x_0:
        seen_states.update(state.keys())
    for state in y:
        seen_states.update(state.keys())
    global_seen_size = np.array([len(seen_states)], dtype=int)
    if basis.is_distributed:
        basis.comm.Allreduce(MPI.IN_PLACE, global_seen_size, op=MPI.SUM)

    p_i = [ri.copy() for ri in r_i]
    it = 0
    
    active_mask = np.ones(n, dtype=bool)

    while it * n < global_seen_size[0] and it < max_iter:
        it += 1
        
        r2 = np.array([ri.norm2() for ri in r_i], dtype=complex)
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, r2)
        active_mask = np.abs(r2) > atol**2
        if not np.any(active_mask):
            break
            
        nu = A_op.apply_multi(p_i, cutoff=slaterWeightMin)
        nu = basis.redistribute_psis(nu)
        rnui = np.array([inner(ri, nui) for ri, nui in zip(r_0, nu)], dtype=complex)
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, rnui)
            
        # Prevent division by zero for inactive states
        rnui[~active_mask] = 1.0
        
        if np.any(np.abs(rnui[active_mask]) < np.finfo(float).eps):
            print(f"Breakdown in BICGSTAB at iteration {it}: rnui is zero for an active state")
            break
            
        alpha = np.zeros(n, dtype=complex)
        alpha[active_mask] = rho_i[active_mask] / rnui[active_mask]
        
        h = [xi.copy() for xi in x_i]
        add_scaled_multi(h, p_i, np.diag(alpha))
        
        s = [ri.copy() for ri in r_i]
        add_scaled_multi(s, nu, np.diag(-alpha))
        
        s2 = np.array([si.norm2() for si in s], dtype=complex)
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, s2)
        
        active_mask_s = np.abs(s2) > atol**2
        if not np.any(active_mask_s):
            x_i = h
            break
            
        t = A_op.apply_multi(s, cutoff=slaterWeightMin)
        t = basis.redistribute_psis(t)
        
        ts = np.array([inner(ti, si) for ti, si in zip(t, s)], dtype=complex)
        t2 = np.array([ti.norm2() for ti in t], dtype=complex)
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, ts)
            basis.comm.Allreduce(MPI.IN_PLACE, t2)
            
        # Prevent division by zero
        t2[~active_mask_s] = 1.0
        
        omega = np.zeros(n, dtype=complex)
        omega[active_mask_s] = ts[active_mask_s] / t2[active_mask_s]
        
        x_i = [hi.copy() for hi in h]
        add_scaled_multi(x_i, s, np.diag(omega))
        
        r_i = [si.copy() for si in s]
        add_scaled_multi(r_i, t, np.diag(-omega))

        basis.add_states(state for xi in x_i for state, amp in xi.items() if abs(amp) > slaterWeightMin)
        tmp = basis.redistribute_psis(r_0 + p_i + x_i + r_i)

        for state in tmp:
            seen_states.update(state.keys())
        global_seen_size[0] = len(seen_states)
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, global_seen_size, op=MPI.SUM)

        r_0 = tmp[:n]
        p_i = tmp[n : 2 * n]
        x_i = tmp[2 * n : 3 * n]
        r_i = tmp[3 * n : 4 * n]

        r2_new = np.array([ri.norm2() for ri in r_i], dtype=complex)
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, r2_new)
        
        active_mask_new = np.abs(r2_new) > atol**2
        if not np.any(active_mask_new):
            break

        rho_ip = np.array([inner(r0i, ri) for r0i, ri in zip(r_0, r_i)], dtype=complex)
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, rho_ip)
            
        beta = np.zeros(n, dtype=complex)
        valid = active_mask_new & active_mask & (np.abs(omega) > 0)
        beta[valid] = (rho_ip[valid] / rho_i[valid]) * (alpha[valid] / omega[valid])
        
        p_i_new = [ri.copy() for ri in r_i]
        add_scaled_multi(p_i_new, p_i, np.diag(beta))
        add_scaled_multi(p_i_new, nu, np.diag(-beta * omega))
        p_i = p_i_new
        
        rho_i = rho_ip

    return x_i


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
    n = len(x0)
    if hasattr(A, "set_restrictions"):
        A.set_restrictions(basis.restrictions)

    def block_inner(B1, B2):
        """
        Compute the block inner product matrix between two state blocks B1 and B2.

        M[i, j] = <B1[i] | B2[j]>. If the basis is distributed, the inner products
        are reduced across all MPI ranks.

        Parameters
        ----------
        B1 : list of ManyBodyState
            First block of states.
        B2 : list of ManyBodyState
            Second block of states.

        Returns
        -------
        M : ndarray
            The complex matrix of inner products.
        """

        M = inner_multi(B1, B2)
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, M, op=MPI.SUM)
        return M

    def matmat(v):
        """
        Apply the operator A to each state in block v and redistribute.

        Parameters
        ----------
        v : list of ManyBodyState
            Input block of states.

        Returns
        -------
        mv : list of ManyBodyState
            Output block of states after applying operator and redistributing.
        """

        mv = A.apply_multi(v, cutoff=slaterWeightMin)
        return basis.redistribute_psis(mv)

    xi = [st.copy() for st in x0]

    Axi = matmat(xi)
    ri = [st.copy() for st in y]
    add_scaled_multi(ri, Axi, -np.eye(n, dtype=complex))

    basis.add_states(
        state for r in ri for state, amp in r.items() if abs(amp) > slaterWeightMin and state not in basis.local_basis
    )

    r0_t = [st.copy() for st in ri]
    pi = [st.copy() for st in ri]

    def block_norm(v):
        """
        Calculate the norm of the state block v.

        Computes the maximum L2 norm of the individual states in the block,
        reducing across all MPI ranks if the basis is distributed.

        Parameters
        ----------
        v : list of ManyBodyState
            Input block of states.

        Returns
        -------
        norm : float
            The square root of the maximum norm in the block.
        """

        norms = np.array([inner(vi, vi).real for vi in v])
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, norms, op=MPI.SUM)
        return np.sqrt(np.max(norms))

    r0_norm = block_norm(r0_t)
    if r0_norm < np.finfo(float).eps:
        return x0

    max_iter = kwargs.get("max_iter", np.inf)
    seen_states = set()
    for state in x0:
        seen_states.update(state.keys())
    for state in y:
        seen_states.update(state.keys())
    global_seen_size = np.array([len(seen_states)], dtype=int)
    if basis.is_distributed:
        basis.comm.Allreduce(MPI.IN_PLACE, global_seen_size, op=MPI.SUM)

    it = 0
    active_mask = np.ones(n, dtype=bool)

    while it * n < global_seen_size[0] and it < max_iter:
        it += 1
        
        r_norms2 = np.array([inner(vi, vi).real for vi in ri])
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, r_norms2, op=MPI.SUM)
        r_norms = np.sqrt(r_norms2)
        
        active_mask = (r_norms >= atol) & (r_norms / r0_norm >= rtol)
        
        if not np.any(active_mask):
            break

        vi = matmat(pi)

        basis.add_states(
            state
            for v in vi
            for state, amp in v.items()
            if abs(amp) > slaterWeightMin and state not in basis.local_basis
        )

        for state in vi:
            seen_states.update(state.keys())
        global_seen_size[0] = len(seen_states)
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, global_seen_size, op=MPI.SUM)

        R0_V = block_inner(r0_t, vi)
        R0_R = block_inner(r0_t, ri)

        # Deflate converged columns
        R0_V[~active_mask, :] = 0
        R0_V[:, ~active_mask] = 0
        R0_V[~active_mask, ~active_mask] = 1.0
        R0_R[:, ~active_mask] = 0

        if np.linalg.cond(R0_V) > 1 / np.finfo(float).eps:
            break

        ai = np.linalg.solve(R0_V, R0_R)

        si = [r.copy() for r in ri]
        add_scaled_multi(si, vi, -ai)

        s_norms2 = np.array([inner(vi, vi).real for vi in si])
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, s_norms2, op=MPI.SUM)
        
        active_mask_s = np.sqrt(s_norms2) >= atol
        if not np.any(active_mask_s):
            xip = [st.copy() for st in xi]
            add_scaled_multi(xip, pi, ai)
            xi = xip
            break

        ti = matmat(si)
        basis.add_states(state for t in ti for state in t if state not in basis.local_basis)

        for state in ti:
            seen_states.update(state.keys())
        global_seen_size[0] = len(seen_states)
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, global_seen_size, op=MPI.SUM)

        ts = sum(inner(ti[j], si[j]) for j in range(n) if active_mask_s[j])
        tt = sum(inner(ti[j], ti[j]) for j in range(n) if active_mask_s[j])
        if basis.is_distributed:
            ts_arr = np.array(ts, dtype=complex)
            tt_arr = np.array(tt, dtype=complex)
            basis.comm.Allreduce(MPI.IN_PLACE, ts_arr, op=MPI.SUM)
            basis.comm.Allreduce(MPI.IN_PLACE, tt_arr, op=MPI.SUM)
            ts = ts_arr.item()
            tt = tt_arr.item()

        if abs(tt) < np.finfo(float).eps:
            wi = 0.0
        else:
            wi = ts / tt

        xip = [st.copy() for st in xi]
        add_scaled_multi(xip, si, wi * np.eye(n, dtype=complex))
        add_scaled_multi(xip, pi, ai)

        rip = [st.copy() for st in si]
        add_scaled_multi(rip, ti, -wi * np.eye(n, dtype=complex))

        R0_T = block_inner(r0_t, ti)
        R0_T[:, ~active_mask_s] = 0
        bi = np.linalg.solve(R0_V, -R0_T)

        pip = [st.copy() for st in rip]
        add_scaled_multi(pip, pi, bi)
        add_scaled_multi(pip, vi, -wi * bi)

        xi = xip
        ri = rip
        pi = pip

        for v in (xi, ri, pi):
            for st in v:
                st.prune(slaterWeightMin)

    return xi


def cg_phys(A_op, A_dict, n_spin_orbitals, x_psi, y_psi, w, delta, basis, atol=1e-5):
    """
    delta is a small imaginary part added to the diagonal of H in order to form A.
    CG algorithm, reformulated to only use matrix vector products of the form A.x
    (the sought after solution). This might allow for an approximation of x based on physics
    rather than pure numerics.
    """

    n = basis.size
    A = basis.build_sparse_matrix(A_op, A_dict)
    x, y = basis.build_vector([x_psi, y_psi])
    Ax = A @ x
    if basis.is_distributed:
        basis.comm.Allreduce(MPI.IN_PLACE, Ax, op=MPI.SUM)
    r = y - Ax
    p = r
    r_prev = r
    alpha_guess = (1 - delta * 1j) / (1 + delta**2)
    for it in range(10 * n):
        x += alpha_guess * p

        p_psi, r_prev_psi, x_psi = basis.build_state([p, r_prev, x], distribute=True)

        basis.expand_at(w, x_psi, A_op, A_dict)
        A = basis.build_sparse_matrix(A_op, A_dict)
        x, y, r_prev, p = basis.build_vector([x_psi, y_psi, r_prev_psi, p_psi])

        Ax = A @ x
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, Ax, op=MPI.SUM)
        r = y - Ax

        ad = (r_prev - r) / alpha_guess
        alpha = np.conj(r_prev) @ r_prev / (np.conj(r_prev) @ ad)

        x += (alpha - alpha_guess) * p
        r -= (alpha - alpha_guess) * ad
        if it % 10 == 0:
            Ax = A @ x
            if basis.is_distributed:
                basis.comm.Allreduce(MPI.IN_PLACE, Ax, op=MPI.SUM)
            r = y - Ax
        if (np.conj(r) @ r).real < atol**2:
            break
        dad = np.conj(p) @ ad
        da = np.conj(ad - 2j * delta * p)
        dar = da @ r
        p = r - dar / dad * p
        r_prev = r
    return x, {"rnorm2": np.conj(r) @ r, "it": it + 1}
