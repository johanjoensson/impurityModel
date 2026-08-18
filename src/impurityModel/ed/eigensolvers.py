"""
Eigensolver drivers for the low-energy spectrum: dense (numpy) and ARPACK
(scipy.sparse) paths, plus the MPI-aware :class:`HermitianOperator` wrapper used to
feed them.
"""

import warnings
from typing import Any, Optional

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


class _RootDrivenApply:
    """Bcast/Reduce apply loop that confines ARPACK/lobpcg's control flow -- and therefore the
    number of collectives it decides to post -- to a single driving rank (rank 0).

    ``h_local`` is a column-distributed operator: each rank's copy has nonzero entries only in
    the columns it owns (see ``build_sparse_matrix``), so ``h_local @ v`` on a full-length ``v``
    gives this rank's partial contribution to the global product, and summing those partial
    results across ranks reconstructs it exactly.

    Only the driving rank ever decides *whether* another apply happens (that decision lives
    inside ARPACK/lobpcg's own convergence logic, called only on that rank via ``root_apply``);
    every other rank just mirrors it by looping in ``worker_loop`` until ``done()`` sends the
    sentinel. Previously every rank ran its own ARPACK instance and relied on all of them
    independently reaching the same number of ``Allreduce`` calls -- which threaded-BLAS FP
    nondeterminism inside ARPACK's internal convergence bookkeeping does not guarantee, and the
    resulting desync is exactly the deadlock this class fixes (see the memory note
    n3-arpack-truncate-initial-deadlock).
    """

    _APPLY = 0
    _DONE = 1

    def __init__(self, h_local: Any, comm: MPI.Comm):
        self.h_local = h_local
        self.comm = comm

    def _local_apply(self, buf: np.ndarray) -> np.ndarray:
        res = self.h_local @ buf
        res_dtype = np.promote_types(self.h_local.dtype, buf.dtype)
        return np.ascontiguousarray(res, dtype=res_dtype)

    def root_apply(self, v: np.ndarray) -> np.ndarray:
        """Broadcast one apply to the workers and return the reduced result. Driving rank only."""
        v = np.asarray(v)
        buf = np.ascontiguousarray(v)
        self.comm.bcast((self._APPLY, buf.shape, buf.dtype), root=0)
        self.comm.Bcast(buf, root=0)
        res = self._local_apply(buf)
        self.comm.Reduce(MPI.IN_PLACE, res, op=MPI.SUM, root=0)
        return res.reshape(v.shape)

    def worker_loop(self) -> None:
        """Mirror the driving rank's applies until it signals done. Non-driving ranks only."""
        while True:
            kind, shape, dtype = self.comm.bcast(None, root=0)
            if kind == self._DONE:
                return
            buf = np.empty(shape, dtype=dtype)
            self.comm.Bcast(buf, root=0)
            res = self._local_apply(buf)
            self.comm.Reduce(res, None, op=MPI.SUM, root=0)

    def done(self) -> None:
        """Signal workers to stop mirroring. Driving rank only; must always be called."""
        self.comm.bcast((self._DONE, None, None), root=0)


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


#: Fixed seed for scipy_eigensystem's fallback start/restart vectors. These are generated only
#: on the rank that drives ARPACK (see _RootDrivenApply) and reach every other rank purely
#: through the broadcast that is already part of each matvec, so a fixed seed is what makes the
#: whole solve rank-count independent -- an unseeded RNG previously produced a different vector
#: on COMM_SELF than on COMM_WORLD for what should be the same solve.
_SCIPY_EIGENSYSTEM_SEED = 0


def _scipy_eigensystem_solve(
    h: scipy.sparse.linalg.LinearOperator,
    e_max: float,
    k: int,
    v0: Optional[np.ndarray],
    eigenValueTol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the ARPACK/lobpcg control flow against ``h`` and return ``(es, vecs)``.

    Must be called on exactly one rank/process: ``scipy_eigensystem`` calls this directly when
    unparallelized, and only on the driving rank (via ``h`` wrapping ``_RootDrivenApply.root_apply``)
    otherwise -- every ``h`` matvec/matmat call this makes is what decides the collectives posted,
    so if two ranks both ran this independently they could post a different number of them.
    """
    rng = np.random.default_rng(_SCIPY_EIGENSYSTEM_SEED)
    if v0 is not None:
        norm_mask = np.linalg.norm(v0, axis=0) > np.sqrt(np.finfo(float).eps)
        v0 = v0[:, norm_mask]
        if v0.shape[1] == 0:
            v0 = None
    if v0 is None:
        v0 = rng.uniform(size=(h.shape[0], 1)) + 1j * rng.uniform(size=(h.shape[0], 1))

    es = np.array([0])
    vecs = v0 / np.linalg.norm(v0)
    ncv = None
    conv_fail = False
    k = min(k, h.shape[1] - 2)

    def done(energies: np.ndarray) -> bool:
        """True if the target number of eigenvalues above e_max is resolved."""
        return len(energies) > 2 + np.sum(energies - np.min(energies) <= e_max)

    while not done(es) and len(es) < h.shape[0] - 2:
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
                vecs, _ = np.linalg.qr(vecs, mode="reduced")
            eigenValueTol = max(eigenValueTol, np.finfo(float).eps) if not conv_fail else eigenValueTol * 10
            conv_fail = True
        except ArpackError:
            # Something went horribly wrong
            # Increase ncv and generate new random starting vectors
            ncv = min(h.shape[0], max(2 * k + 3, 20)) if ncv is None else min(ncv * 2, h.shape[0])
            es = np.array([0])
            vecs = rng.uniform(size=(h.shape[0], 1)) + 1j * rng.uniform(size=(h.shape[0], 1))
            vecs, _ = np.linalg.qr(vecs, mode="reduced")
        if es is None or len(es) == 0:
            es = np.array([0])

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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                es, vecs = scipy.sparse.linalg.lobpcg(
                    h,
                    vecs,
                    largest=False,
                    tol=max(eigenValueTol, 1e-12),
                    maxiter=500,
                )
    indices = np.argsort(es)
    es = es[indices]
    return es, np.ascontiguousarray(vecs[:, indices])


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
        The local sparse matrix (column-distributed: nonzero only in this rank's owned columns,
        see ``build_sparse_matrix``).
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
    if comm is None or comm.size == 1:

        def local_apply(v: np.ndarray) -> np.ndarray:
            return (h_local @ v).reshape(v.shape)

        h = scipy.sparse.linalg.LinearOperator(
            h_local.shape,
            matvec=local_apply,
            rmatvec=local_apply,
            dtype=h_local.dtype,
        )
        es, vecs = _scipy_eigensystem_solve(h, e_max, k, v0, eigenValueTol)
        return (es, vecs) if return_eigvecs else es

    apply = _RootDrivenApply(h_local, comm)
    if comm.rank != 0:
        apply.worker_loop()
        ok = comm.bcast(None, root=0)
        if not ok:
            message = comm.bcast(None, root=0)
            raise RuntimeError(f"scipy_eigensystem failed on rank 0: {message}")
        es_shape, es_dtype, vecs_shape, vecs_dtype = comm.bcast(None, root=0)
        es = np.empty(es_shape, dtype=es_dtype)
        vecs = np.empty(vecs_shape, dtype=vecs_dtype)
        comm.Bcast(es, root=0)
        comm.Bcast(vecs, root=0)
        return (es, vecs) if return_eigvecs else es

    # Driving rank (rank 0): run the solver against an operator wrapping root_apply, which
    # broadcasts every matvec/matmat this makes to the workers -- so the number of collectives
    # posted is decided here, once, and mirrored exactly by worker_loop above, rather than each
    # rank's own ARPACK deciding independently (see _RootDrivenApply's docstring).
    h = scipy.sparse.linalg.LinearOperator(
        h_local.shape,
        matvec=apply.root_apply,
        rmatvec=apply.root_apply,
        dtype=h_local.dtype,
    )
    error: Optional[Exception] = None
    try:
        es, vecs = _scipy_eigensystem_solve(h, e_max, k, v0, eigenValueTol)
    except Exception as exc:  # must reach every rank via bcast, not just crash rank 0
        error = exc
    finally:
        # Always signal the workers out of worker_loop, success or failure -- otherwise a rank-0
        # exception here hangs every other rank forever inside worker_loop's collective wait.
        apply.done()

    comm.bcast(error is None, root=0)
    if error is not None:
        comm.bcast(str(error), root=0)
        raise error

    comm.bcast((es.shape, es.dtype, vecs.shape, vecs.dtype), root=0)
    comm.Bcast(es, root=0)
    comm.Bcast(vecs, root=0)
    return (es, vecs) if return_eigvecs else es


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
