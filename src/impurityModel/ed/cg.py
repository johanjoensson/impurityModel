import numpy as np
import scipy as sp
from time import perf_counter
import itertools
import impurityModel.ed.finite as finite
from mpi4py import MPI
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState, applyOp, inner


def cg(A, x, y, atol=1e-5):
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
    def matmat(M, v):
        mv = [None for _ in v]
        for i in range(len(v)):
            mv[i] = applyOp(M, v[i], cutoff=slaterWeightMin, restrictions=basis.restrictions)
        return basis.redistribute_psis(mv)

    def matnorm(A, B):
        return np.vdot(A, B)

    n = len(x0)
    xi_sparse = x0
    ri_sparse = [yi - Ax0i for yi, Ax0i in zip(y, matmat(A, x0))]
    basis.add_states(state for r in ri_sparse for state in r if state not in basis.local_basis)

    ri = basis.build_vector(basis.redistribute_psis(ri_sparse)).T
    xi = basis.build_vector(basis.redistribute_psis(x0)).T
    r0_t = ri
    # r0_t = np.random.rand(ri.shape[0], n) + 1j * np.random.rand(ri.shape[0], n)  # or possibly ri
    pi = ri
    if np.max(np.linalg.norm(r0_t, axis=0)) < np.finfo(float).eps:
        return x0

    while True:
        if (
            np.max(np.linalg.norm(ri, axis=0)) < atol
            or np.max(np.linalg.norm(ri, axis=0)) / np.max(np.linalg.norm(r0_t, axis=0)) < rtol
        ):
            # Converged, residuals are small
            break
        pi_sparse = basis.build_state(pi.T)
        vi_sparse = matmat(A, pi_sparse)

        r0_t_sparse = basis.build_state(r0_t.T)
        ri_sparse = basis.build_state(ri.T)
        xi_sparse = basis.build_state(xi.T)
        basis.add_states(state for v in vi_sparse for state in v if state not in basis.local_basis)
        r0_t = basis.build_vector(basis.redistribute_psis(r0_t_sparse)).T
        ri = basis.build_vector(basis.redistribute_psis(ri_sparse)).T
        pi = basis.build_vector(basis.redistribute_psis(pi_sparse)).T
        xi = basis.build_vector(basis.redistribute_psis(xi_sparse)).T

        vi = basis.build_vector(basis.redistribute_psis(vi_sparse)).T
        if np.linalg.cond(np.conj(r0_t.T) @ vi) > 1 / np.finfo(float).eps:
            # BREAKDOWN, ill conditioned matrix!
            print("Breakdown in Block BICGSTAB")
            break
        ai = np.linalg.solve(np.conj(r0_t.T) @ vi, np.conj(r0_t.T) @ ri)
        si = ri - vi @ ai
        if np.max(np.linalg.norm(si, axis=0)) < atol:
            # Converged, correction is small enough without further additions
            xi = xi + pi @ ai
            break

        ti_sparse = matmat(A, basis.build_state(si.T))
        si_sparse = basis.build_state(si.T)

        basis.add_states(state for t in ti_sparse for state in t if state not in basis.local_basis)

        r0_t = basis.build_vector(basis.redistribute_psis(r0_t_sparse)).T
        ri = basis.build_vector(basis.redistribute_psis(ri_sparse)).T
        pi = basis.build_vector(basis.redistribute_psis(pi_sparse)).T
        xi = basis.build_vector(basis.redistribute_psis(xi_sparse)).T
        ti = basis.build_vector(basis.redistribute_psis(ti_sparse)).T
        si = basis.build_vector(basis.redistribute_psis(si_sparse)).T
        vi = basis.build_vector(basis.redistribute_psis(vi_sparse)).T

        wi = np.vdot(ti, si) / np.vdot(ti, ti)
        xip = xi + pi @ ai + wi * si
        rip = si - wi * ti
        bi = np.linalg.solve(np.conj(r0_t.T) @ vi, -np.conj(r0_t.T) @ ti)
        pip = rip + (pi - wi * vi) @ bi

        xi = xip
        ri = rip
        pi = pip
    return basis.build_state(xi.T)


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
