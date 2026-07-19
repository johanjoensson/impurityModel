"""
Eigensolver drivers for the low-energy spectrum: dense (numpy), ARPACK
(scipy.sparse), and block-Lanczos (TRLM) paths, plus the MPI-aware
:class:`HermitianOperator` wrapper used to feed them.
"""

import time
import warnings
from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse
from mpi4py import MPI
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigsh


class HermitianOperator(scipy.sparse.linalg.LinearOperator):
    """A LinearOperator representing a Hermitian operator defined by its diagonal and lower triangular part.

    This class enables efficient matrix-vector products without storing the full dense matrix.
    """

    def __init__(self, diagonal: np.ndarray, diagonal_indices: np.ndarray, triangular_part: scipy.sparse.csr_matrix):
        self.diagonal = diagonal if len(diagonal.shape) == 1 else diagonal.reshape(-1)
        self.diagonal_indices = diagonal_indices
        self.triangular_part = triangular_part
        # Delegate dtype/shape (and, on scipy>=1.15, the array-namespace ``_xp``
        # attribute that ``LinearOperator.dot`` now requires) to the base initializer
        # instead of setting them by hand.
        super().__init__(dtype=triangular_part.dtype, shape=triangular_part.shape)

    def _matvec(self, v):
        v = v.reshape(-1)
        res = np.zeros(v.shape[0], dtype=v.dtype)
        res[self.diagonal_indices] = self.diagonal * v[self.diagonal_indices]
        return res + self.triangular_part @ v + self.triangular_part.getH() @ v

    def _matmat(self, m):
        res = np.zeros((self.shape[0], m.shape[1]), dtype=self.dtype)
        for col in range(m.shape[1]):
            res[self.diagonal_indices, col] = self.diagonal * m[self.diagonal_indices, col]
        return res + self.triangular_part @ m + self.triangular_part.getH() @ m

    def _adjoint(self):
        """Return the adjoint of the operator (which is itself)."""
        return self


def mpi_matmat(m: Any, comm: Optional[MPI.Comm]) -> Callable[[np.ndarray], np.ndarray]:
    """Create a parallel MPI matrix-matrix multiplication wrapper.

    Parameters
    ----------
    m : Any
        The local matrix operator (supporting matrix multiplication `@`).
    comm : MPI.Comm, optional
        The MPI communicator.

    Returns
    -------
    f : callable
        A function that takes a vector/matrix `v` and returns `m @ v` reduced across ranks.
    """

    def f(v: np.ndarray) -> np.ndarray:
        """Perform the local matrix product and MPI reduction.

        Parameters
        ----------
        v : np.ndarray
            The input vector or matrix.

        Returns
        -------
        np.ndarray
            The result of the multiplication.
        """
        res = m @ v
        if comm is not None:
            comm.Allreduce(MPI.IN_PLACE, res)
        return res.reshape(v.shape)

    return f


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
    elif isinstance(h_local, scipy.sparse.linalg.LinearOperator):
        h = h_local @ np.eye(h_local.shape[0], dtype=h_local.dtype)
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


def scipy_eigensystem(
    h_local: Any,
    e_max: float,
    k: int = 10,
    v0: Optional[np.ndarray] = None,
    eigenValueTol: float = 0,
    return_eigvecs: bool = True,
    comm: Optional[MPI.Comm] = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Solve the eigenvalue problem using SciPy's sparse solver (ARPACK).

    Parameters
    ----------
    h_local : Any
        The local sparse matrix.
    e_max : float
        The maximum energy above the ground state to resolve.
    k : int, default 10
        Number of eigenvalues to request.
    v0 : np.ndarray, optional
        Initial guess eigenvectors.
    eigenValueTol : float, default 0
        Tolerance for eigenvalue convergence.
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
    h = scipy.sparse.linalg.LinearOperator(
        h_local.shape,
        matvec=mpi_matmat(h_local, comm),
        rmatvec=mpi_matmat(h_local, comm),
        dtype=h_local.dtype,
    )
    h_diag = h_local.diagonal() if callable(getattr(h_local, "diagonal", None)) else h_local.diagonal
    h_diag_inv = h_diag.copy()
    nonzeros = np.nonzero(h_diag_inv)
    h_diag_inv[nonzeros] = 1.0 / h_diag_inv[nonzeros]
    diag_h_inv = scipy.sparse.diags_array(h_diag_inv, shape=h_local.shape)
    scipy.sparse.linalg.LinearOperator(
        h_local.shape,
        matvec=mpi_matmat(diag_h_inv, comm),
        rmatvec=mpi_matmat(diag_h_inv, comm),
        dtype=h_local.dtype,
    )

    es = np.array([0])
    rng = np.random.default_rng()
    if v0 is not None:
        norm_mask = np.linalg.norm(v0, axis=0) > np.sqrt(np.finfo(float).eps)
        v0 = v0[:, norm_mask]
        if v0.shape[1] == 0:
            v0 = None
    if v0 is None:
        v0 = rng.uniform(size=(h.shape[0], 1)) + 1j * rng.uniform(size=(h.shape[0], 1))
        if comm is not None:
            comm.Allreduce(MPI.IN_PLACE, v0, op=MPI.SUM)

    vecs = v0 / np.linalg.norm(v0)
    ncv = None
    conv_fail = False
    k = min(k, h.shape[1] - 2)

    def done(energies: np.ndarray) -> bool:
        """Check if convergence criteria are met.

        Parameters
        ----------
        energies : np.ndarray
            Calculated energies.

        Returns
        -------
        bool
            True if target number of eigenvalues above e_max is resolved.
        """
        return len(energies) > 2 + np.sum(energies - np.min(energies) <= e_max)

    while not done(es) and len(es) < h.shape[0] - 2:
        time.perf_counter()
        try:
            es, vecs = eigsh(
                h,
                k=min(k, h.shape[1] - 2),
                which="SA",
                v0=vecs[:, 0] if len(vecs.shape) > 1 else vecs,
                ncv=ncv,
                tol=eigenValueTol if conv_fail else 0,
            )
            # eigsh does not guarantee that the eigenvectors are orthonormal. therefore we do a
            # QR decomposition on them.
            vecs, _ = np.linalg.qr(vecs, mode="reduced")
            k *= 2
        except ArpackNoConvergence as e:
            # Reqested accuracy was not reached
            # increase eigenvalueTol and try again, starting from the already obtained eigenvectors
            es = e.eigenvalues
            vecs = e.eigenvectors
            if vecs.size == 0:
                vecs = rng.uniform(size=(h.shape[0], 1)) + 1j * rng.uniform(size=(h.shape[0], 1))
                if comm is not None:
                    comm.Allreduce(MPI.IN_PLACE, vecs, op=MPI.SUM)
                vecs, _ = np.linalg.qr(vecs, mode="reduced")
            eigenValueTol = max(eigenValueTol, np.finfo(float).eps) if not conv_fail else eigenValueTol * 10
            conv_fail = True
        except ArpackError:
            # Something went horribly wrong
            # Increase ncv and generate new random starting vectors
            ncv = min(h.shape[0], max(2 * k + 3, 20)) if ncv is None else min(ncv * 2, h.shape[0])
            es = np.array([0])
            vecs = rng.uniform(size=(h.shape[0], 1)) + 1j * rng.uniform(size=(h.shape[0], 1))
            if comm is not None:
                comm.Allreduce(MPI.IN_PLACE, vecs, op=MPI.SUM)
            vecs, _ = np.linalg.qr(vecs, mode="reduced")
        if es is None or len(es) == 0:
            es = np.array([0])

        if comm is not None:
            comm.barrier()

        indices = np.argsort(es)
        es = es[indices]
        vecs = vecs[:, indices]
        if done(es) and 5 * vecs.shape[1] < h.shape[0]:
            # In principle, lobpcg should be able to correct some errors in the eigenvectors ad
            # eigenvalues found by eigsh (which uses ARPACK behind the scenes).
            # eigsh struggles with degenerate or nearly degenerate eigenstates, so do one round of
            # lobpcg to correct any errors.
            # lobpcg is robust as long as the preconditioner is very good (is this what robust
            # means?). We don't have a good preconditioner, so we ignore any warnings from lobpcg
            # instead.
            # if comm.rank == 0:
            time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                es, vecs = scipy.sparse.linalg.lobpcg(
                    h,
                    vecs,
                    # vecs[:, : min(2 * (2 + sum(es - np.min(es) <= e_max)), vecs.shape[1])],
                    # M=OPinv,
                    largest=False,
                    tol=max(eigenValueTol, 1e-12),
                    maxiter=500,
                )
    indices = np.argsort(es)
    es = es[indices]
    if return_eigvecs:
        vecs = vecs[:, indices]
        return es, np.ascontiguousarray(vecs)
    return es


def eigensystem(h_local, e_max, k=10, e0=None, v0=None, eigenValueTol=0, return_eigvecs=True, comm=None, dense=False):
    """
    Return eigen-energies and eigenstates of a Hamiltonian matrix.

    This function automatically chooses between a dense eigensolver, SciPy's sparse solver (ARPACK),
    and a custom thick-restarted block Lanczos solver based on the matrix size and options.


    Parameters
    ----------
    h_local : scipy.sparse sparse array (any kind)
        Contains part of the full many-body Hamiltonian, local to this MPI rank.
    e_max : float
        Maximum energy difference for excited states
    k : int
        Calculate at least k eigenstates.
    eigenValueTol : float
        The precision of the returned eigenvalues.
    return_eigvecs : bool
        If True, return eigenvalues and eigenvectors for all states with energy within e_max of the lowest energy state.
        If False, return only the calculated eigenvalues.
    comm : MPI communicator to use for any MPI communication
    dende : Convert h_local to dense form and use standard np.linalg.eigh to calculate the full spectra
    """

    # e_max is limited by the accuracy of the calculated eigenvalues and machine precision.
    # e_max=None means "no energy cutoff" (get_eigenvectors passes max_energy=None): keep every
    # computed state. Guard the None here, otherwise max(None, ...) raises TypeError -- a live
    # crash on the dense (basis < dense_cutoff) path, which does not otherwise touch e_max.
    e_max = np.inf if e_max is None else max(e_max, eigenValueTol, np.finfo(float).eps * 100)

    N = h_local.shape[0]
    # Set up random initial vectors
    np.random.seed(42)  # For reproducibility in testing, might want to remove in prod
    n_blocks = 1  # Simple 1-block for now unless block size is needed
    if v0 is not None:
        psi0 = v0
        if len(psi0.shape) == 1:
            psi0 = psi0.reshape(-1, 1)
    else:
        psi0 = np.random.rand(N, n_blocks) + 1j * np.random.rand(N, n_blocks)

    psi0, _ = np.linalg.qr(psi0)

    # We want to find eigenvalues up to e_max above ground state.
    # Since we don't know the ground state yet, we just find k eigenvalues.
    if dense or N <= 20:
        if return_eigvecs:
            es, vecs = dense_eigensystem(h_local, return_eigvecs, comm)
        else:
            es = dense_eigensystem(h_local, return_eigvecs, comm)
            vecs = None
    else:
        if return_eigvecs:
            es, vecs = scipy_eigensystem(h_local, e_max, k, v0, 0, return_eigvecs, comm)
        else:
            es = scipy_eigensystem(h_local, e_max, k, v0, 0, return_eigvecs, comm)
            vecs = None
    indices = np.argsort(es)
    es = es[indices]
    if return_eigvecs and vecs is not None:
        vecs = vecs[:, indices]
    mask = es - np.min(es) <= e_max

    if return_eigvecs:
        return es[mask], vecs[:, mask]
    return es[mask]
