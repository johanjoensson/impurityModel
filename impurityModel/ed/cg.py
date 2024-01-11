import numpy as np
import scipy as sp
from time import perf_counter
import impurityModel.ed.finite as finite


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


def bicgstab(A, x, y, atol=1e-5):
    return x, 0


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
        basis.comm.Allreduce(Ax.copy(), Ax)
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
            basis.comm.Allreduce(Ax.copy(), Ax)
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
                basis.comm.Allreduce(Ax.copy(), Ax)
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
