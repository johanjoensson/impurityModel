"""
Dense eigensolver driver for the small-basis ground-state path.

Above ``dense_cutoff`` determinants the low spectrum comes from the thick-restart block Lanczos
solver (:mod:`impurityModel.ed.trlm`); this module only diagonalizes a matrix it can densify.
"""

from typing import Any, Optional

import numpy as np
from mpi4py import MPI


def dense_eigensystem(
    h_local: Any, return_eigvecs: bool = True, comm: Optional[MPI.Comm] = None
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Solve the eigenvalue problem using dense matrix diagonalization.

    Parameters
    ----------
    h_local : Any
        The matrix to diagonalize.
    return_eigvecs : bool, default True
        Whether to return eigenvectors.
    comm : MPI.Comm, optional
        MPI communicator.

    Returns
    -------
    es : np.ndarray
        Array of eigenvalues.
    vecs : np.ndarray, optional
        Array of eigenvectors, returned if return_eigvecs is True.
    """
    rank = comm.rank if comm is not None else 0
    if hasattr(h_local, "toarray"):
        h = h_local.toarray()
    elif hasattr(h_local, "todense"):
        h = h_local.todense()
    else:
        h = h_local
    if comm is not None:
        comm.Reduce(h if rank != 0 else MPI.IN_PLACE, h, root=0, op=MPI.SUM)
    if return_eigvecs:
        if rank == 0:
            es, vecs = np.linalg.eigh(h, UPLO="L")
        else:
            es = np.empty((h_local.shape[0]), dtype=float, order="C")
            vecs = np.empty(h_local.shape, dtype=h_local.dtype, order="C")
    else:
        es = np.linalg.eigvalsh(h, UPLO="L") if rank == 0 else np.empty(h_local.shape[0], dtype=float)
    if comm is not None:
        comm.Bcast(es, root=0)
        if return_eigvecs:
            comm.Bcast(vecs, root=0)
    if return_eigvecs:
        return es, vecs
    return es


def eigensystem(h_local, e_max, eigenValueTol=0, return_eigvecs=True, comm=None):
    """
    Return the eigen-energies and eigenstates of a Hamiltonian matrix, densely diagonalized.

    Parameters
    ----------
    h_local : scipy.sparse sparse array (any kind) or ndarray
        Contains part of the full many-body Hamiltonian, local to this MPI rank; the parts are
        summed onto rank 0 and the result broadcast (see :func:`dense_eigensystem`).
    e_max : float or None
        Maximum energy difference for excited states. ``None`` keeps every computed state.
    eigenValueTol : float
        Floor on ``e_max``: the precision of the returned eigenvalues.
    return_eigvecs : bool
        If True, return eigenvalues and eigenvectors for all states with energy within e_max of the lowest energy state.
        If False, return only the calculated eigenvalues.
    comm : MPI communicator to use for any MPI communication
    """

    # e_max is limited by the accuracy of the calculated eigenvalues and machine precision.
    # e_max=None means "no energy cutoff" (get_eigenvectors passes max_energy=None): keep every
    # computed state. Guard the None here, otherwise max(None, ...) raises TypeError -- a live
    # crash on the dense (basis < dense_cutoff) path, which does not otherwise touch e_max.
    e_max = np.inf if e_max is None else max(e_max, eigenValueTol, np.finfo(float).eps * 100)

    if return_eigvecs:
        es, vecs = dense_eigensystem(h_local, return_eigvecs, comm)
    else:
        es = dense_eigensystem(h_local, return_eigvecs, comm)
        vecs = None
    indices = np.argsort(es)
    es = es[indices]
    if return_eigvecs and vecs is not None:
        vecs = vecs[:, indices]
    mask = es - np.min(es) <= e_max

    if return_eigvecs:
        return es[mask], vecs[:, mask]
    return es[mask]
