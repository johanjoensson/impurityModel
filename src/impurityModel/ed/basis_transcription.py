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

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, Row, applyOp, inner_multi


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
            # psi may be a ManyBodyState (val is a width-1 Row) or a plain dict of
            # scalars (a documented duck-typed input): only a Row needs unwrapping.
            amp = val[0] if isinstance(val, Row) else val
            if idx is None or abs(amp) < slaterWeightMin:
                continue
            v[row, idx] = amp

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
    psis = basis.redistribute_psis(*psis)
    v = np.empty((len(psis), len(basis.local_basis)), dtype=dtype, order="C")
    for (row, psi), (col, state) in itertools.product(enumerate(psis), enumerate(basis.local_basis)):
        row_amp = psi.get(state)
        v[row, col] = 0 if row_amp is None else row_amp[0]
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

    Notes
    -----
    Still a flat-state producer, not yet flipped to width-1 ``ManyBodyState``
    (Phase 7 step 2a): its output flows unchanged (``.items()``/arithmetic on scalars)
    into many not-yet-flipped consumers (GF stack, groundstate, spectra, rixs,
    susceptibility, sectorization -- steps 2c-2f), so flipping this producer alone
    breaks ~75 tests across those modules. Flip once its consumers are block-tolerant,
    or as part of the mechanical rename in step 3.
    """
    if isinstance(vs, np.matrix):
        vs = vs.A
    if isinstance(vs, np.ndarray) and len(vs.shape) == 1:
        vs = vs.reshape((1, vs.shape[0]))
    if isinstance(vs, list):
        vs = np.array(vs)
    # width=1: a row that never receives a setitem below (every entry <= slaterWeightMin,
    # a real occurrence for a rank-deficient/deflated column) must not stay the width-0
    # polymorphic zero -- every element of this list is expected to be width 1.
    res = [ManyBodyState(width=1) for _ in range(vs.shape[0])]
    if vs.shape[1] == basis.size:
        for j, i in np.argwhere(np.abs(vs[:, basis.local_indices]) > slaterWeightMin):
            res[j][basis.local_basis[i]] = vs[j, i + basis.offset]
    elif vs.shape[1] == len(basis.local_basis):
        for j, i in np.argwhere(np.abs(vs) > slaterWeightMin):
            res[j][basis.local_basis[i]] = vs[j, i]
    else:
        raise RuntimeError(
            f"The dimensions of the input dense vector does not match a distributed, or full vector.\n"
            f"{vs.shape} != ({vs.shape[0]}, {basis.size}) || ({vs.shape[0]}, {len(basis.local_basis)})"
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
                    vals.append(val[0])
    else:
        columns = []
        bras = []
        values = []
        for ket, ket_state in zip(basis.local_basis, build_local_operator_list(basis, op, 0)):
            col = _index_dict[ket]
            for bra, val in ket_state.items():
                columns.append(col)
                bras.append(bra)
                values.append(val[0])

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

    ``rho[n, i, j] = <psi_n| c_{orb_j}^dagger c_{orb_i} |psi_n>``

    For the square case (``orbital_indices_left == orbital_indices_right``) the
    identity ``rho[i, j] = <phi_j | phi_i>`` (where ``|phi_k> = c_{orb_k}|psi>``)
    is exploited, cutting operator applications from O(n^2) to O(n) and
    halving inner products via Hermitian symmetry.

    For the general rectangular case we also exploit this decomposition
    ``rho[i, j] = <chi_j | phi_i>`` (where ``|chi_k> = c_{orb_k}|psi>`` and
    ``|phi_k> = c_{orb_k}|psi>``), reducing operator applications to O(n).

    ``psis`` may also be a ``ManyBodyState`` (Phase 6a of the state-unification
    refactor); it is unpacked to a list once on entry and the body below is otherwise
    unchanged. Rewriting the per-state loop as one block-apply per orbital was
    considered and rejected, re-evaluated after ``ManyBodyState.select``/``column``
    got a direct O(rows) gather (no longer routed through a selection-matrix matvec) and
    still rejected: reading each orbital's per-state value back out no longer needs a
    full ``width x width`` Gram matrix (cheap columns make a per-state diagonal
    extraction possible), but that only matches this loop's existing per-state
    inner-product cost -- it doesn't beat it. The apply side is the one place
    ``apply_block`` could still win (near-flat cost in block width vs one term/sign/
    restriction/hash pass per state), and that gain is genuinely small here: every
    operator applied is a *trivial* single-term annihilator, so there is little shared
    per-determinant work left to amortize across states in the first place. See
    doc/plans/manybodystate_block_unification.md, Phase 6a, item 3b.
    """
    if isinstance(psis, ManyBodyState):
        psis = psis.to_states()
    if orbital_indices_left is None:
        orbital_indices_left = list(range(basis.num_spin_orbitals))
    if orbital_indices_right is None:
        orbital_indices_right = list(range(basis.num_spin_orbitals))
    n_left, n_right = len(orbital_indices_left), len(orbital_indices_right)
    rhos = np.zeros((len(psis), n_left, n_right), dtype=complex)

    square = orbital_indices_left == orbital_indices_right

    # Built once, not once per state: each annihilator is a pure function of orb, not
    # of psi_n, so constructing it inside the loop below paid len(psis)x redundant
    # ManyBodyOperator construction for no reason.
    left_ops = [ManyBodyOperator({((orb, "a"),): 1.0}) for orb in orbital_indices_left]
    right_ops = left_ops if square else [ManyBodyOperator({((orb, "a"),): 1.0}) for orb in orbital_indices_right]

    for n, psi_n in enumerate(psis):
        phi = [op(psi_n, 0) for op in left_ops]
        chi = phi if square else [op(psi_n, 0) for op in right_ops]

        if basis.is_distributed:
            phi = basis.redistribute_psis(*phi)
            chi = phi if square else basis.redistribute_psis(*chi)

        rhos[n] = inner_multi(chi, phi).T

    if basis.is_distributed:
        basis.comm.Allreduce(MPI.IN_PLACE, rhos, op=MPI.SUM)

    return rhos
