import numpy as np
import scipy as sp
from time import perf_counter
import itertools
import impurityModel.ed.finite as finite
from mpi4py import MPI
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState, applyOp, inner


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
            print(f"breaking early r {it}")
            break
        dp = -np.vdot(p_i, r) / np.vdot(p, p_i)
        p = r + dp * p

    return x, info


def bicgstab(A_op, x_0, y, basis, slaterWeightMin, atol=1e-8):
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
    Ax = [
        applyOp(
            A_op,
            xi,
            cutoff=slaterWeightMin,
            restrictions=basis.restrictions,
        )
        for xi in x_0
    ]
    Ax = basis.redistribute_psis(Ax)

    r_0 = [yi - Axi for yi, Axi in zip(y, Ax)]

    x_i = [ManyBodyState({state: amp for state, amp in xi.items()}) for xi in x_0]
    r_i = [ManyBodyState({state: amp for state, amp in ri.items()}) for ri in r_0]
    rho_i = np.array([ri.norm2() for ri in r_i], dtype=complex)
    if basis.is_distributed:
        basis.comm.Allreduce(MPI.IN_PLACE, rho_i)
    p_i = [ManyBodyState(dict(ri.items())) for ri in r_i]
    it = 0
    while True:
        it += 1
        nu = [
            applyOp(
                A_op,
                pi,
                cutoff=slaterWeightMin,
                restrictions=basis.restrictions,
            )
            for pi in p_i
        ]
        nu = basis.redistribute_psis(nu)
        rnui = np.array([inner(ri, nui) for ri, nui in zip(r_0, nu)], dtype=complex)
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, rnui)
        alpha = rho_i / rnui
        h = [xi + a * pi for xi, a, pi in zip(x_i, alpha.tolist(), p_i)]
        s = [ri - a * nui for ri, a, nui in zip(r_i, alpha.tolist(), nu)]
        s2 = np.array([si.norm2() for si in s], dtype=complex)
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, s2)
        if np.all(np.abs(s2) < atol**2):
            x_i = h
            break
        t = [
            applyOp(
                A_op,
                si,
                cutoff=slaterWeightMin,
                restrictions=basis.restrictions,
            )
            for si in s
        ]
        t = basis.redistribute_psis(t)
        ts = np.array([inner(ti, si) for ti, si in zip(t, s)], dtype=complex)
        t2 = np.array([ti.norm2() for ti in t])
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, ts)
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, t2)
        omega = ts / t2
        x_i = [hi + w * si for hi, w, si in zip(h, omega.tolist(), s)]
        r_i = [si - w * ti for si, w, ti in zip(s, omega.tolist(), t)]

        basis.add_states(state for xi in x_i for state, amp in xi.items() if abs(amp) > slaterWeightMin)
        tmp = basis.redistribute_psis(r_0 + p_i + x_i + r_i)

        r_0 = tmp[:n]
        p_i = tmp[n : 2 * n]
        x_i = tmp[2 * n : 3 * n]
        r_i = tmp[3 * n : 4 * n]

        r2 = np.array([ri.norm2() for ri in r_i], dtype=complex)
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, r2)
        if np.all(np.abs(r2) < atol**2):
            break

        rho_ip = np.array([inner(r0i, ri) for r0i, ri in zip(r_0, r_i)], dtype=complex)
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, rho_ip)
        beta = (rho_ip / rho_i) * (alpha / omega)
        p_i = [ri + b * (pi - w * nui) for ri, b, pi, w, nui in zip(r_i, beta.tolist(), p_i.copy(), omega.tolist(), nu)]
        rho_i = rho_ip

    return x_i


def block_bicgstab(A, x0, y, basis: Basis, slaterWeightMin: float, atol=1e-8, rtol=1e-12):
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

    def block_inner(B1, B2):
        """
        Documentation for block_inner.
        """
        M = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                M[i, j] = inner(B1[i], B2[j])
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, M, op=MPI.SUM)
        return M

    def matmat(v):
        """
        Documentation for matmat.
        """
        mv = [applyOp(A, vi, cutoff=slaterWeightMin, restrictions=basis.restrictions) for vi in v]
        return basis.redistribute_psis(mv)

    xi = [st.copy() for st in x0]
    
    Axi = matmat(xi)
    ri = [yi - axi for yi, axi in zip(y, Axi)]
    
    basis.add_states(state for r in ri for state, amp in r.items() if abs(amp) > slaterWeightMin and state not in basis.local_basis)

    r0_t = [st.copy() for st in ri]
    pi = [st.copy() for st in ri]
    
    def block_norm(v):
        """
        Documentation for block_norm.
        """
        norms = np.array([inner(vi, vi).real for vi in v])
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, norms, op=MPI.SUM)
        return np.sqrt(np.max(norms))

    r0_norm = block_norm(r0_t)
    if r0_norm < np.finfo(float).eps:
        return x0

    while True:
        r_norm = block_norm(ri)
        if r_norm < atol or r_norm / r0_norm < rtol:
            break
            
        vi = matmat(pi)
        
        basis.add_states(state for v in vi for state, amp in v.items() if abs(amp) > slaterWeightMin and state not in basis.local_basis)

        R0_V = block_inner(r0_t, vi)
        R0_R = block_inner(r0_t, ri)
        
        if np.linalg.cond(R0_V) > 1 / np.finfo(float).eps:
            print("Breakdown in Block BICGSTAB")
            break
            
        ai = np.linalg.solve(R0_V, R0_R)
        
        si = [ri[j].copy() for j in range(n)]
        for j in range(n):
            for k in range(n):
                si[j] -= vi[k] * ai[k, j]
                
        if block_norm(si) < atol:
            for j in range(n):
                for k in range(n):
                    xi[j] += pi[k] * ai[k, j]
            break
            
        ti = matmat(si)
        basis.add_states(state for t in ti for state, amp in t.items() if abs(amp) > slaterWeightMin and state not in basis.local_basis)

        ts = sum(inner(ti[j], si[j]) for j in range(n))
        tt = sum(inner(ti[j], ti[j]) for j in range(n))
        if basis.is_distributed:
            ts_arr = np.array(ts, dtype=complex)
            tt_arr = np.array(tt, dtype=complex)
            basis.comm.Allreduce(MPI.IN_PLACE, ts_arr, op=MPI.SUM)
            basis.comm.Allreduce(MPI.IN_PLACE, tt_arr, op=MPI.SUM)
            ts = ts_arr.item()
            tt = tt_arr.item()
        
        wi = ts / tt
        
        xip = [xi[j] + wi * si[j] for j in range(n)]
        for j in range(n):
            for k in range(n):
                xip[j] += pi[k] * ai[k, j]
                
        rip = [si[j] - wi * ti[j] for j in range(n)]
        
        R0_T = block_inner(r0_t, ti)
        bi = np.linalg.solve(R0_V, -R0_T)
        
        pip = [rip[j].copy() for j in range(n)]
        for j in range(n):
            for k in range(n):
                pip[j] += (pi[k] - wi * vi[k]) * bi[k, j]
                
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
    t_cg = perf_counter()
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
    t_expansion = 0
    t_build_sparse_mat = 0
    t_build_vectors_separate = 0
    t_matrix_mul = 0
    t_rest_of_cg = 0
    for it in range(10 * n):
        x += alpha_guess * p

        p_psi, r_prev_psi, x_psi = basis.build_state([p, r_prev, x], distribute=True)

        t_expand = perf_counter()
        basis.expand_at(w, x_psi, A_op, A_dict)
        t_expansion += perf_counter() - t_expand
        t_build_sparse = perf_counter()
        A = basis.build_sparse_matrix(A_op, A_dict)
        t_build_sparse_mat += perf_counter() - t_build_sparse
        t_build_vectors = perf_counter()
        x, y, r_prev, p = basis.build_vector([x_psi, y_psi, r_prev_psi, p_psi])
        t_build_vectors_separate += perf_counter() - t_build_vectors

        t_matmul = perf_counter()
        Ax = A @ x
        if basis.is_distributed:
            basis.comm.Allreduce(MPI.IN_PLACE, Ax, op=MPI.SUM)
        t_matrix_mul += perf_counter() - t_matmul
        t_rest = perf_counter()
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
        t_rest_of_cg = perf_counter() - t_rest
    t_cg = perf_counter() - t_cg
    print(f"Converged in {it+1} iterations!")
    print(f"Took {t_cg:.3f} seconds")
    print(f"--->Expanding the basis took {t_expansion:.4f} seconds")
    print(f"--->Building the matrix took {t_build_sparse_mat:.4f} seconds")
    print(f"--->Building the vectors took {t_build_vectors_separate:.4f} seconds")
    print(f"--->Matrix multiplication took {t_matrix_mul:.4f} seconds")
    print(f"--->The rest of the CG algorithm took {t_rest_of_cg:.4f} seconds")
    return x, {"rnorm2": np.conj(r) @ r, "it": it + 1}
