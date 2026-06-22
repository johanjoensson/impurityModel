import numpy as np
import scipy.linalg as sp
import scipy.sparse as sps

from impurityModel.ed.ManyBodyUtils import ManyBodyState, add_scaled_multi, inner_multi


def is_array(V):
    """Check if the vector representation is a dense/sparse array (as opposed to ManyBodyState list)."""
    if isinstance(V, (np.ndarray, sps.spmatrix, sps.sparray)):
        return True
    if isinstance(V, list) and len(V) > 0 and isinstance(V[0], (np.ndarray, sps.spmatrix, sps.sparray)):
        return True
    return False


def block_inner(V, W, mpi=False, comm=None):
    r"""Computes V^\dagger W."""
    if is_array(V):
        if isinstance(V, list):
            V = np.column_stack(V)
        if isinstance(W, list):
            W = np.column_stack(W)
        return V.conj().T @ W
    else:
        res = inner_multi(V, W)
        if mpi and comm is not None:
            from mpi4py import MPI

            comm.Allreduce(MPI.IN_PLACE, res, op=MPI.SUM)
        return res


def block_apply(H, V, basis=None, mpi=False, slaterWeightMin=0.0):
    """Applies Hamiltonian H to block vector V."""
    if (
        is_array(V)
        or getattr(H, "is_array_operator", False)
        or isinstance(H, np.ndarray)
        or isinstance(H, sps.spmatrix)
    ):
        if isinstance(V, list) and isinstance(V[0], np.ndarray):
            V_arr = np.column_stack(V)
            res = H @ V_arr
            return res
        return H @ V
    else:
        wp = H.apply_multi(V, cutoff=slaterWeightMin)
        if mpi and basis is not None and basis.comm is not None:
            wp = basis.redistribute_psis(wp)
        return wp


def block_add_scaled(V, W, alpha, slaterWeightMin=0.0):
    """Computes V = V + W @ alpha in place."""
    if is_array(V):
        V += W @ alpha
    else:
        add_scaled_multi(V, W, alpha)
        if slaterWeightMin > 0:
            for st in V:
                st.prune(slaterWeightMin)
    return V


def block_combine(Q, Y, slaterWeightMin=0.0):
    """Combines basis blocks Q with Ritz vectors Y (Q @ Y)."""
    if is_array(Q):
        if isinstance(Q, list):
            Q_arr = np.column_stack(Q)
            return Q_arr @ np.ascontiguousarray(Y, dtype=complex)
        return Q @ np.ascontiguousarray(Y, dtype=complex)
    else:
        n_out = Y.shape[1]
        out = [ManyBodyState() for _ in range(n_out)]
        add_scaled_multi(out, Q, np.ascontiguousarray(Y, dtype=complex))
        if slaterWeightMin > 0:
            for st in out:
                st.prune(slaterWeightMin)
        return out


def block_orthogonalize(wp, Q, overlaps=None, mpi=False, comm=None):
    """Orthogonalizes wp against Q using Gram-Schmidt (wp = wp - Q @ Q^T wp)."""
    if mpi and comm is not None:
        from mpi4py import MPI
    if is_array(wp):
        if isinstance(Q, list):
            Q = np.column_stack(Q)
        if overlaps is None:
            overlaps = Q.conj().T @ wp
        wp -= Q @ overlaps
    else:
        if overlaps is None:
            overlaps = inner_multi(Q, wp)
            if mpi and comm is not None:

                comm.Allreduce(MPI.IN_PLACE, overlaps, op=MPI.SUM)
        add_scaled_multi(wp, Q, -overlaps)
    return wp, overlaps


def block_normalize(wp, mpi=False, comm=None, slaterWeightMin=0.0):
    r"""Computes M = wp^\dagger wp, then cholesky factorizes to find beta, and normalizes wp into q_next."""
    M = block_inner(wp, wp, mpi, comm)
    n = M.shape[0]
    cond = np.linalg.cond(M)
    if cond > 1.0 / (100.0 * np.finfo(float).eps):
        raise sp.LinAlgError(f"Ill-conditioned matrix: cond={cond:.2e}")
    try:
        L = sp.cholesky(M, lower=True)
    except sp.LinAlgError:
        raise sp.LinAlgError("Cholesky decomposition failed")
    beta = np.conj(L.T)
    beta_inv = sp.inv(beta)

    if is_array(wp):
        q_next = wp @ beta_inv
    else:
        q_next = [ManyBodyState() for _ in range(n)]
        add_scaled_multi(q_next, wp, beta_inv)
        if slaterWeightMin > 0:
            for st in q_next:
                st.prune(slaterWeightMin)
    return q_next, beta
