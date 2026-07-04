"""
Transcription between the distributed determinant basis and dense/sparse
linear-algebra objects: wavefunction vectors, operator matrices, and
single-particle density matrices.

All functions take the basis as their first argument. The ones documented as
reducing over ranks (`build_vector` without root, `build_dense_matrix`,
`build_density_matrices`) contain MPI collectives and must be called by all
ranks of ``basis.comm``.
"""

import itertools
from typing import Any, Optional, Union

import numpy as np
import scipy as sp
from mpi4py import MPI

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, applyOp, inner_multi


def build_vector(
    basis, psis: list[ManyBodyState], root: Optional[int] = None, slaterWeightMin: float = 0
) -> np.ndarray:
    """Build a dense matrix representation of wavefunctions in the basis.

    Parameters
    ----------
    psis : list of ManyBodyState
        The wavefunctions to represent.
    root : int, optional
        MPI rank to reduce the vector to. If None, it is reduced to all ranks.
    slaterWeightMin : float, default 0
        Minimum amplitude threshold below which coefficients are ignored.

    Returns
    -------
    np.ndarray
        The 2D dense matrix representation of the wavefunctions.
    """
    v = np.zeros((len(psis), basis.size), dtype=complex, order="C")
    # psis = basis.redistribute_psis(psis)
    # row_states_in_basis: list[bytes] = []
    # row_dict = {state: basis._index_dict[state] for state in basis.local_basis}
    # col_dict = dict(zip(basis.local_basis, range(basis.local_indices.start, basis.local_indices.stop)))
    _index_dict = basis._index_dict
    for row, psi in enumerate(psis):
        for state, val in psi.items():
            idx = _index_dict.get(state)
            if idx is None or abs(val) < slaterWeightMin:
                continue
            v[row, idx] = val

    if basis.is_distributed and root is None:
        basis.comm.Allreduce(MPI.IN_PLACE, v, op=MPI.SUM)
    elif basis.is_distributed:
        basis.comm.Reduce(MPI.IN_PLACE if basis.comm.rank == root else v, v, op=MPI.SUM, root=root)
    return v


def build_distributed_vector(basis, psis: list[ManyBodyState], dtype: Any = complex) -> np.ndarray:
    """Build the MPI-local portion of a wavefunction vector.

    Parameters
    ----------
    psis : list of ManyBodyState
        The wavefunctions to represent.
    dtype : Any, default complex
        The data type of the returned array.

    Returns
    -------
    np.ndarray
        The 2D array containing the local amplitudes.
    """
    psis = basis.redistribute_psis(psis)
    v = np.empty((len(psis), len(basis.local_basis)), dtype=dtype, order="C")
    for (row, psi), (col, state) in itertools.product(enumerate(psis), enumerate(basis.local_basis)):
        v[row, col] = psi.get(state, 0)
    return v


def build_state(basis, vs: Union[list[np.ndarray], np.ndarray], slaterWeightMin: float = 0) -> list[ManyBodyState]:
    """Convert dense vectors back to a list of ManyBodyState objects.

    Parameters
    ----------
    vs : list of np.ndarray or np.ndarray
        The dense vector representations.
    slaterWeightMin : float, default 0
        Minimum amplitude threshold to keep.

    Returns
    -------
    list of ManyBodyState
        The corresponding list of many-body states.
    """
    if isinstance(vs, np.matrix):
        vs = vs.A
    if isinstance(vs, np.ndarray) and len(vs.shape) == 1:
        vs = vs.reshape((1, vs.shape[0]))
    if isinstance(vs, list):
        vs = np.array(vs)
    res = [ManyBodyState({}) for _ in range(vs.shape[0])]
    if vs.shape[1] == basis.size:
        for j, i in np.argwhere(np.abs(vs[:, basis.local_indices]) > slaterWeightMin):
            res[j][basis.local_basis[i]] = vs[j, i + basis.offset]
    elif vs.shape[1] == len(basis.local_basis):
        for j, i in np.argwhere(np.abs(vs) > slaterWeightMin):
            res[j][basis.local_basis[i]] = vs[j, i]
    else:
        raise RuntimeError(
            f"The dimensions of the input dense vector does not match a distributed, or full vector.\n{vs.shape} != ({vs.shape[0]}, {basis.size}) || ({vs.shape[0]}, {len(basis.local_basis)})"
        )
    return res


def build_local_operator_list(basis, op, slaterWeightMin):
    """
    Apply the operator to all (MPI local) basis states, in order.
    Return the results in a list.
    """
    res = []
    unit_state = ManyBodyState()
    for state in basis.local_basis:
        unit_state[state] = 1.0
        res.append(applyOp(op, unit_state, cutoff=slaterWeightMin))
        unit_state.erase(state)
    return res


def build_dense_matrix(basis, op, distribute=True):
    """
    Get the operator as a dense matrix in the current basis.
    by default the dense matrix is distributed to all ranks.
    """
    h_local = build_sparse_matrix(basis, op)
    if basis.is_distributed:
        h = np.empty(h_local.shape, dtype=h_local.dtype)
        basis.comm.Allreduce(h_local.todense(), h, op=MPI.SUM)
    else:
        h = h_local.todense(order="C")
    return h


def build_sparse_matrix(basis, op: ManyBodyOperator):
    """
    Get the operator as a sparse matrix in the current basis.
    The sparse matrix is distributed over all ranks.
    """
    if isinstance(op, dict):
        op = ManyBodyOperator(op)

    rows = []
    cols = []
    vals = []
    _index_dict = basis._index_dict
    if not basis.is_distributed:
        for ket, ket_state in zip(basis.local_basis, build_local_operator_list(basis, op, 0)):
            col = _index_dict[ket]
            for bra, val in ket_state.items():
                row = _index_dict.get(bra)
                if row is not None:
                    rows.append(row)
                    cols.append(col)
                    vals.append(val)
    else:
        columns = []
        bras = []
        values = []
        for ket, ket_state in zip(basis.local_basis, build_local_operator_list(basis, op, 0)):
            col = _index_dict[ket]
            for bra, val in ket_state.items():
                columns.append(col)
                bras.append(bra)
                values.append(val)

        global_rows = list(basis._index_sequence(bras))
        _size = basis.size
        for row, col, val in zip(global_rows, columns, values):
            if row != _size:
                rows.append(row)
                cols.append(col)
                vals.append(val)

    n = len(basis)
    if rows:
        res = sp.sparse.csc_array((vals, (rows, cols)), shape=(n, n), dtype=complex)
    else:
        res = sp.sparse.csc_array((n, n), dtype=complex)
    return res


def build_density_matrices(basis, psis, orbital_indices_left=None, orbital_indices_right=None):
    r"""Compute single-particle density matrices for a list of many-body states.

    rho[n, i, j] = <psi_n| c_{orb_j}^dagger c_{orb_i} |psi_n>

    For the square case (orbital_indices_left == orbital_indices_right) the
    identity rho[i, j] = <phi_j | phi_i>  (where |phi_k> = c_{orb_k}|psi>)
    is exploited, cutting operator applications from O(n^2) to O(n) and
    halving inner products via Hermitian symmetry.

    For the general rectangular case we also exploit this decomposition
    rho[i, j] = <chi_j | phi_i> (where |chi_k> = c_{orb_k}|psi> and
    |phi_k> = c_{orb_k}|psi>), reducing operator applications to O(n).
    """
    if orbital_indices_left is None:
        orbital_indices_left = list(range(basis.num_spin_orbitals))
    if orbital_indices_right is None:
        orbital_indices_right = list(range(basis.num_spin_orbitals))
    n_left, n_right = len(orbital_indices_left), len(orbital_indices_right)
    rhos = np.zeros((len(psis), n_left, n_right), dtype=complex)

    square = orbital_indices_left == orbital_indices_right

    for n, psi_n in enumerate(psis):
        phi = [ManyBodyOperator({((orb, "a"),): 1.0})(psi_n, 0) for orb in orbital_indices_left]
        if square:
            chi = phi
        else:
            chi = [ManyBodyOperator({((orb, "a"),): 1.0})(psi_n, 0) for orb in orbital_indices_right]

        if basis.is_distributed:
            phi = basis.redistribute_psis(phi)
            if square:
                chi = phi
            else:
                chi = basis.redistribute_psis(chi)

        rhos[n] = inner_multi(chi, phi).T

    if basis.is_distributed:
        basis.comm.Allreduce(MPI.IN_PLACE, rhos, op=MPI.SUM)

    return rhos
