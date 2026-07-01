"""
This module contains functions doing the bulk of the calculations.
"""

import itertools
import time
import warnings
from collections import deque
from math import pi, sqrt
from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse
from mpi4py import MPI
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigsh
from sympy.physics.wigner import gaunt

# Local imports
from impurityModel.ed.average import k_B, thermal_average, thermal_average_scale_indep  # noqa: F401
from impurityModel.ed.block_structure import get_equivalent_blocks
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, inner


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


def get_job_tasks(rank, ranks, tasks_tot):
    """
    Return a tuple of job task indices for a particular rank.

    This function distribute the job tasks in tasks_tot
    over all the ranks.

    Note
    ----
    This is a primerly a MPI help function.

    Parameters
    ----------
    rank : int
        Current MPI rank/worker.
    ranks : int
        Number of MPI ranks/workers in total.
    tasks_tot : list
        List of task indices.
        Length is the total number of job tasks.

    """
    n_tot = len(tasks_tot)
    nj = n_tot // ranks
    rest = n_tot % ranks
    tasks = [tasks_tot[i] for i in range(nj * rank, nj * rank + nj)]
    if rank < rest:
        tasks.append(tasks_tot[n_tot - rest + rank])
    return tuple(tasks)


def rotate_matrix(M, T):
    r"""
    Rotate the matrix, M, using the matrix T.
    Returns :math:`M' = T^{\dagger} M T`
    Parameters
    ==========
    M : NDArray - Matrix to rotate
    T : NDArray or dict - Rotation matrix to use, or dict of rotation matrices for blocks.
    Returns
    =======
    M' : NDArray - The rotated matrix
    """
    if isinstance(T, dict):
        from scipy.linalg import block_diag

        sorted_keys = sorted(T.keys())
        T_matrix = block_diag(*(T[k] for k in sorted_keys))
        return np.conj(T_matrix.T) @ M @ T_matrix
    return np.conj(T.T) @ M @ T


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
        if rank == 0:
            es = np.linalg.eigvalsh(h, UPLO="L")
        else:
            es = np.empty((h_local.shape[0]), dtype=float)
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
            # eigsh does not guarantee that the eigenvectors are orthonormal. therefore we do a QR decomposition on them.
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
            # In principle, lobpcg should be able to correct some errors in the eigenvectors ad eigenvalues found by eigsh (which uses ARPACK behind the scenes).
            # eigsh struggles with degenerate or nearly degenerate eigenstates, so do one round of lobpcg to correct any errors.
            # lobpcg is robust as long as the preconditioner is very good (is this what robust means?). We don't have a good preconditioner, so we ignore any warnings from lobpcg instead.
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

    # e_max is limited by the accuracy of the calculated eigenvalues and machine precision
    e_max = max(e_max, eigenValueTol, np.finfo(float).eps * 100)

    from impurityModel.ed.trlm import thick_restart_block_lanczos

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
    num_wanted = k
    max_subspace_blocks = max(40, int(np.ceil(num_wanted * 2)))

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
        # try:
        #     es, vecs = thick_restarted_block_lanczos(
        #         psi0=psi0,
        #         h_op=h_local,
        #         basis=None,
        #         num_wanted=num_wanted,
        #         max_subspace_blocks=max_subspace_blocks,
        #         tol=max(1e-8, eigenValueTol),
        #         max_restarts=100,
        #         verbose=False,
        #     )
        # except Exception:
        #     # Fallback to scipy if TRLM fails
        #     if return_eigvecs:
        #         es, vecs = scipy_eigensystem(h_local, e_max, k, v0, 0, return_eigvecs, comm)
        #     else:
        #         es = scipy_eigensystem(h_local, e_max, k, v0, 0, return_eigvecs, comm)
        #         vecs = None

    indices = np.argsort(es)
    es = es[indices]
    if return_eigvecs and vecs is not None:
        vecs = vecs[:, indices]
    mask = es - np.min(es) <= e_max

    if return_eigvecs:
        return es[mask], vecs[:, mask]
    return es[mask]


def print_expectation_values(
    rhos, es, rot_to_spherical, block_structure, s_values=None, l_values=None, j_values=None, sisb_values=None
):
    """
    print several expectation values, e.g. E, N, L^2.

    If ``s_values`` / ``l_values`` / ``j_values`` are given (one impurity ``S`` / ``L``
    / ``J`` quantum number per eigenstate, e.g. from :func:`manifold_observable_values`
    with :func:`make_impurity_casimir_operators` + :func:`casimir_to_quantum_number`),
    the corresponding columns are appended. When all are ``None`` the output is
    identical to before (used when the eigenstates are not available, e.g. on non-root
    ranks).
    """
    orb_offset = min(orb for block in block_structure.blocks for orb in block)
    equivalent_blocks = get_equivalent_blocks(block_structure)
    print(f"E0 = {es[0]:9.6f}")
    block_N_string = [f"N({','.join(f'{b}' for b in blocks)})" for blocks in equivalent_blocks]
    # Each block-occupation column is right-aligned to a width that fits both its header
    # and the 7-8 char ``.5f`` value below it, so header and numbers line up.
    block_N_widths = [max(len(Ns), 8) for Ns in block_N_string]
    block_N_string_formatted = [f"{Ns:>{w}s}" for Ns, w in zip(block_N_string, block_N_widths)]
    extra = [
        (name, vals)
        for name, vals in (("S", s_values), ("L", l_values), ("J", j_values), ("Si.Sb", sisb_values))
        if vals is not None
    ]
    # Lz/Sz/L.S and the S/L/J/Si.Sb columns are printed with the space-flag format
    # ``{x: 8.6f}``, which is 9 characters wide (the sign column sits on top of the 8),
    # so their headers must be 9 wide to line up with the numbers below.
    extra_header = "".join(f"  {name:>9s}" for name, _ in extra)
    print(
        f"{'i':>3s}  {'E-E0':>11s}  {'N':>8s}  {'N(Dn)':>8s}  {'N(Up)':>8s}  {'  '.join(block_N_string_formatted)}  {'Lz':>9s}  {'Sz':>9s}  {'L.S':>9s}{extra_header}"
    )
    for i, (e, rho) in enumerate(zip(es - es[0], rhos)):
        block_occs = [
            np.sum(np.diag(rho)[list(orb - orb_offset for block in blocks for orb in block_structure.blocks[block])])
            for blocks in equivalent_blocks
        ]
        block_occ_string_formatted = ["" for _ in block_occs]
        for ib, b_occ in enumerate(block_occs):
            block_occ_string_formatted[ib] = f"{np.real(b_occ):>{block_N_widths[ib]}.5f}"
        rho_spherical = rotate_matrix(rho, rot_to_spherical)
        N, Ndn, Nup = get_occupations_from_rho_spherical(rho_spherical)
        Lz = get_Lz_from_rho_spherical(rho_spherical)
        Sz = get_Sz_from_rho_spherical(rho_spherical)
        LS = get_LS_from_rho_spherical(rho_spherical)
        extra_fields = "".join(f"  {vals[i]: 8.6f}" for _, vals in extra)
        print(
            f"{i:>3d}  {e:11.8f}  {N:8.5f}  {Ndn:8.5f}  {Nup:8.5f}  {'  '.join(block_occ_string_formatted)}  {Lz: 8.6f}  {Sz: 8.6f}  {LS: 8.6f}{extra_fields}"
        )
    print("\n")


def get_occupations_from_rho_spherical(rho):
    """
    Calculate the (spin polarized) occupation from the density matrix.
    """
    n_orbs = rho.shape[0]
    return (
        np.real(np.trace(rho)),
        np.real(np.trace(rho[: n_orbs // 2, : n_orbs // 2])),
        np.real(np.trace(rho[n_orbs // 2 :, n_orbs // 2 :])),
    )


def get_Lz_from_rho_spherical(rho: np.ndarray, l: Optional[int] = None) -> float:
    """Calculate the expectation value of L_z from the density matrix in spherical basis.

    Parameters
    ----------
    rho : np.ndarray
        The density matrix.
    l : int, optional
        The orbital angular momentum quantum number. If None, it is calculated from rho's shape.

    Returns
    -------
    float
        The expectation value <L_z>.
    """
    if l is None:
        l = (rho.shape[0] // 2 - 1) // 2
    return np.real(
        sum(ml * (rho[i, i] + rho[i + (2 * l + 1), i + (2 * l + 1)]) for i, ml in enumerate(range(-l, l + 1)))
    )


def get_Lplus_from_rho_spherical(rho: np.ndarray, l: int) -> complex:
    """Calculate the expectation value of L_+ from the density matrix in spherical basis.

    Parameters
    ----------
    rho : np.ndarray
        The density matrix.
    l : int
        The orbital angular momentum quantum number.

    Returns
    -------
    complex
        The expectation value <L_+>.
    """
    # L+ |l, ml> = sqrt(l*(l+1) - ml*(ml+1))|l, ml+1>
    llp1 = l * (l + 1)
    #   L+    |2, -2>,  |2, -1>, |2,  0>, |2,  1>, |2,  2>
    # <2, -2|    0         0        0        0        0
    # <2, -1| sqrt(8)      0        0        0        0
    # <2,  0|    0      sqrt(6)     0        0        0
    # <2,  1|    0         0     sqrt(6)     0        0
    # <2,  2|    0         0        0     sqrt(8)     0
    Lplus = np.diag([np.sqrt(llp1 - ml * (ml + 1)) for ml in range(-l, l)], k=-1)
    return np.trace(
        rho @ np.block([[Lplus, np.zeros((2 * l + 1, 2 * l + 1))], [np.zeros((2 * l + 1, 2 * l + 1)), Lplus]])
    )


def get_Sminus_from_rho_spherical(rho: np.ndarray, l: int, s: float = 0.5) -> complex:
    """Calculate the expectation value of S_- from the density matrix in spherical basis.

    Parameters
    ----------
    rho : np.ndarray
        The density matrix.
    l : int
        The orbital angular momentum quantum number.
    s : float, default 0.5
        The spin quantum number.

    Returns
    -------
    complex
        The expectation value <S_->.
    """
    # S+ |s, ms> = sqrt(s*(s+1) - ms*(ms+1))|s, ms+1>
    ssp1 = s * (s + 1)
    ms = +1 / 2
    #   S-      |1/2,-1/2>,  |1/2, 1/2>
    # <1/2,-1/2|    0            1
    # <1/2, 1/2|    0            0
    # S- = [[0   S-],
    #        0   0 ]]
    Sminus = np.diag(np.repeat(np.sqrt(ssp1 - ms * (ms - 1)), 2 * l), k=1)
    return np.trace(
        rho
        @ np.block(
            [
                [np.zeros((2 * l + 1, 2 * l + 1)), Sminus],
                [np.zeros((2 * l + 1, 2 * l + 1)), np.zeros((2 * l + 1, 2 * l + 1))],
            ]
        )
    )


def get_Lminus_from_rho_spherical(rho: np.ndarray, l: int) -> complex:
    """Calculate the expectation value of L_- from the density matrix in spherical basis.

    Parameters
    ----------
    rho : np.ndarray
        The density matrix.
    l : int
        The orbital angular momentum quantum number.

    Returns
    -------
    complex
        The expectation value <L_->.
    """
    # L- |l, ml> = sqrt(l*(l+1) - ml*(ml-1))|l, ml-1>
    llp1 = l * (l + 1)
    #   L+    |2, -2>,  |2, -1>, |2,  0>, |2,  1>, |2,  2>
    # <2, -2|    0      sqrt(4)     0        0        0
    # <2, -1|    0         0     sqrt(6)     0        0
    # <2,  0|    0         0        0     sqrt(6)     0
    # <2,  1|    0         0        0        0     sqrt(4)
    # <2,  2|    0         0        0        0        0
    Lminus = np.diag([np.sqrt(llp1 - ml * (ml - 1)) for ml in range(-l + 1, l + 1)], k=1)
    return np.trace(
        rho @ np.block([[Lminus, np.zeros((2 * l + 1, 2 * l + 1))], [np.zeros((2 * l + 1, 2 * l + 1)), Lminus]])
    )


def get_Splus_from_rho_spherical(rho: np.ndarray, l: int, s: float = 0.5) -> complex:
    """Calculate the expectation value of S_+ from the density matrix in spherical basis.

    Parameters
    ----------
    rho : np.ndarray
        The density matrix.
    l : int
        The orbital angular momentum quantum number.
    s : float, default 0.5
        The spin quantum number.

    Returns
    -------
    complex
        The expectation value <S_+>.
    """
    # S+ |s, ms> = sqrt(s*(s+1) - ms*(ms+1))|s, ms+1>
    ssp1 = s * (s + 1)
    ms = -1 / 2
    #   S+      |1/2,-1/2>,  |1/2, 1/2>
    # <1/2,-1/2|    0            0
    # <1/2, 1/2|    1            0
    # S+ = [[0   0],
    #        S+  0]]
    Splus = np.diag(np.repeat(np.sqrt(ssp1 - ms * (ms + 1)), 2 * l), k=-1)
    return np.trace(
        rho
        @ np.block(
            [
                [np.zeros((2 * l + 1, 2 * l + 1)), np.zeros((2 * l + 1, 2 * l + 1))],
                [Splus, np.zeros((2 * l + 1, 2 * l + 1))],
            ]
        )
    )


def get_Sz_from_rho_spherical(rho: np.ndarray, l: Optional[int] = None) -> float:
    """Calculate the expectation value of S_z from the density matrix in spherical basis.

    Parameters
    ----------
    rho : np.ndarray
        The density matrix.
    l : int, optional
        The orbital angular momentum quantum number. If None, it is calculated from rho's shape.

    Returns
    -------
    float
        The expectation value <S_z>.
    """
    if l is None:
        l = (rho.shape[0] // 2 - 1) // 2
    return 1 / 2 * np.real(sum(-rho[i, i] + rho[i + (2 * l + 1), i + (2 * l + 1)] for i in range(2 * l + 1)))


def _single_particle_lsj_matrices(l):
    r"""Single-particle ``L`` and ``S`` operator matrices in the spherical basis.

    Layout: a ``2*(2l+1)`` space whose first ``2l+1`` orbitals are spin-down
    (:math:`m_s=-1/2`, ``ml=-l..l``) and the next ``2l+1`` are spin-up — matching the
    ``get_*_from_rho_spherical`` helpers.

    Returns
    -------
    (lz, lplus, lminus, sz, splus, sminus) : tuple of np.ndarray
        Each of shape ``(2*(2l+1), 2*(2l+1))``.
    """
    n = 2 * l + 1
    mls = np.arange(-l, l + 1)
    llp1 = l * (l + 1)
    zeros = np.zeros((n, n))
    eye = np.eye(n)
    lz = np.diag(np.concatenate((mls, mls)).astype(float)).astype(complex)
    sz = np.diag(np.concatenate((-0.5 * np.ones(n), 0.5 * np.ones(n)))).astype(complex)
    lplus_block = np.diag([np.sqrt(llp1 - ml * (ml + 1)) for ml in mls[:-1]], k=-1)
    lplus = np.block([[lplus_block, zeros], [zeros, lplus_block]]).astype(complex)
    lminus = lplus.conj().T
    splus = np.block([[zeros, zeros], [eye, zeros]]).astype(complex)
    sminus = splus.conj().T
    return lz, lplus, lminus, sz, splus, sminus


def get_LS_from_rho_spherical(rho: np.ndarray, l: Optional[int] = None) -> float:
    r"""Calculate the expectation value of the one-body spin-orbit coupling
    :math:`\langle \mathbf{L}\cdot\mathbf{S}\rangle` from the density matrix in the
    spherical basis.

    :math:`\mathbf{L}\cdot\mathbf{S}` is a one-body operator in the single-particle
    space, so its expectation value is the contraction
    :math:`\langle \mathbf{L}\cdot\mathbf{S}\rangle = \mathrm{Tr}(\rho\, (l\cdot s))`,
    where :math:`l\cdot s = l_z s_z + \tfrac{1}{2}(l_+ s_- + l_- s_+)` is the
    single-particle spin-orbit matrix. No many-body solver is required.

    The spherical basis layout matches the other ``get_*_from_rho_spherical``
    helpers: a ``2*(2l+1)`` matrix whose first ``2l+1`` rows/columns are the
    spin-down (:math:`m_s=-1/2`) orbitals ``ml = -l..l`` and whose last ``2l+1``
    are the spin-up (:math:`m_s=+1/2`) orbitals.

    Parameters
    ----------
    rho : np.ndarray
        The density matrix in the spherical basis.
    l : int, optional
        The orbital angular momentum quantum number. If None, it is calculated
        from rho's shape.

    Returns
    -------
    float
        The expectation value :math:`\langle \mathbf{L}\cdot\mathbf{S}\rangle`.
    """
    if l is None:
        l = (rho.shape[0] // 2 - 1) // 2
    n = 2 * l + 1
    # Contract against the leading 2*(2l+1) sub-block, matching the index-based
    # get_Lz/get_Sz helpers (robust when rho is not exactly spin-doubled, e.g. an
    # odd-sized block).
    rho = rho[: 2 * n, : 2 * n]
    lz, lplus, lminus, sz, splus, sminus = _single_particle_lsj_matrices(l)
    ls = lz @ sz + 0.5 * (lplus @ sminus + lminus @ splus)
    return np.real(np.trace(rho @ ls))


def make_spin_operators(spin_pairs):
    r"""Build the one-body spin ladder/Cartan operators for a set of spatial orbitals.

    Each spatial orbital contributes a spin doublet; ``spin_pairs`` lists its
    ``(dn_index, up_index)`` spin-orbital indices. The returned operators are

    .. math::
        \hat S_+ = \sum_a c^\dagger_{a\uparrow} c_{a\downarrow}, \quad
        \hat S_- = \sum_a c^\dagger_{a\downarrow} c_{a\uparrow}, \quad
        \hat S_z = \tfrac12 \sum_a (n_{a\uparrow} - n_{a\downarrow}).

    Parameters
    ----------
    spin_pairs : iterable of (int, int)
        ``(dn_index, up_index)`` spin-orbital index pairs, one per spatial orbital.

    Returns
    -------
    (ManyBodyOperator, ManyBodyOperator, ManyBodyOperator)
        The operators :math:`(\hat S_+, \hat S_-, \hat S_z)`.
    """
    s_plus, s_minus, s_z = {}, {}, {}
    for dn, up in spin_pairs:
        s_plus[((up, "c"), (dn, "a"))] = 1.0
        s_minus[((dn, "c"), (up, "a"))] = 1.0
        s_z[((up, "c"), (up, "a"))] = 0.5
        s_z[((dn, "c"), (dn, "a"))] = -0.5
    return ManyBodyOperator(s_plus), ManyBodyOperator(s_minus), ManyBodyOperator(s_z)


def make_orbital_angular_momentum_operators(channels):
    r"""Build the one-body orbital angular-momentum operators for a set of shells.

    Each *channel* is one ``(l, spin)`` block: an ordered list of the orbital
    indices for ``ml = -l, -l+1, ..., l`` at fixed spin. The returned operators are

    .. math::
        \hat L_+ = \sum c^\dagger_{m_l+1} c_{m_l}\,\sqrt{l(l+1)-m_l(m_l+1)}, \quad
        \hat L_z = \sum m_l\, n_{m_l},

    summed over every channel (both spins), and :math:`\hat L_- = \hat L_+^\dagger`.

    Parameters
    ----------
    channels : iterable of sequence[int]
        Each element lists the ``2l+1`` orbital indices ordered by ``ml`` from
        ``-l`` to ``+l`` for one spin of one ``l``-shell. ``l`` is inferred from
        the length.

    Returns
    -------
    (ManyBodyOperator, ManyBodyOperator, ManyBodyOperator)
        The operators :math:`(\hat L_+, \hat L_-, \hat L_z)`.
    """
    l_plus, l_minus, l_z = {}, {}, {}
    for indices in channels:
        indices = list(indices)
        l = (len(indices) - 1) // 2
        llp1 = l * (l + 1)
        for a, ml in enumerate(range(-l, l + 1)):
            l_z[((indices[a], "c"), (indices[a], "a"))] = float(ml)
            if ml < l:
                coeff = np.sqrt(llp1 - ml * (ml + 1))
                l_plus[((indices[a + 1], "c"), (indices[a], "a"))] = coeff
                l_minus[((indices[a], "c"), (indices[a + 1], "a"))] = coeff
    return ManyBodyOperator(l_plus), ManyBodyOperator(l_minus), ManyBodyOperator(l_z)


def make_impurity_casimir_operators(impurity_orbitals, rot_to_spherical):
    r"""Build the total impurity ``(L, S, J)`` ladder/Cartan operators in the
    **computational** basis.

    For each impurity ``l``-shell the single-particle ``L``/``S`` matrices are built in
    the spherical basis (:func:`_single_particle_lsj_matrices`, where the ``ml``/spin
    structure is explicit) and rotated to the computational basis via
    ``rot_to_spherical`` (``O_comp = R\,O_sph\,R^\dagger`` with ``R`` the
    spherical→computational rotation, matching :func:`rotate_matrix`), then summed over
    shells. This makes ``L²``/``J²``/``S²`` evaluable on states stored in the
    computational basis — the ``ml`` dependence of ``L`` is carried by the rotation, and
    the construction is robust to whatever spin ordering the computational basis uses
    (Phase 5 unblocks this for the deferred ``L²``/``J²`` reporting).

    Parameters
    ----------
    impurity_orbitals : dict
        ``Basis.impurity_orbitals`` (``partition -> list of orbital-index blocks``).
        The shell's ``l`` is inferred from the orbital count ``2*(2l+1)``.
    rot_to_spherical : np.ndarray or dict
        The spherical→computational rotation: a single ``2(2l+1)`` matrix, or a dict
        ``{partition: matrix}`` (as in ``get_spectra``).

    Returns
    -------
    (L, S, J) : tuple
        Each is ``(plus, minus, z)`` as ``ManyBodyOperator``s, ready for
        :func:`apply_casimir` / :func:`expect_casimir`. ``J = L + S``.
    """
    l_plus, l_minus, l_z = {}, {}, {}
    s_plus, s_minus, s_z = {}, {}, {}
    for partition, blocks in impurity_orbitals.items():
        orbs = [orb for block in blocks for orb in block]
        n_so = len(orbs)
        shell_l = (n_so // 2 - 1) // 2
        if 2 * (2 * shell_l + 1) != n_so:
            raise ValueError(
                f"Impurity partition {partition} has {n_so} spin-orbitals, which is not a "
                f"spin-doubled l-shell (2*(2l+1)); cannot build L/S/J operators for it."
            )
        lz_m, lp_m, lm_m, sz_m, sp_m, sm_m = _single_particle_lsj_matrices(shell_l)
        rot = rot_to_spherical[partition] if isinstance(rot_to_spherical, dict) else rot_to_spherical
        rot = np.asarray(rot, dtype=complex)
        for target, matrix in (
            (l_z, lz_m),
            (l_plus, lp_m),
            (l_minus, lm_m),
            (s_z, sz_m),
            (s_plus, sp_m),
            (s_minus, sm_m),
        ):
            computational = rot @ matrix @ rot.conj().T
            for i in range(n_so):
                for j in range(n_so):
                    if abs(computational[i, j]) > 1e-12:
                        key = ((orbs[i], "c"), (orbs[j], "a"))
                        target[key] = target.get(key, 0.0) + computational[i, j]
    l_ops = (ManyBodyOperator(l_plus), ManyBodyOperator(l_minus), ManyBodyOperator(l_z))
    s_ops = (ManyBodyOperator(s_plus), ManyBodyOperator(s_minus), ManyBodyOperator(s_z))
    j_ops = (l_ops[0] + s_ops[0], l_ops[1] + s_ops[1], l_ops[2] + s_ops[2])
    return l_ops, s_ops, j_ops


def apply_casimir(psi, j_plus, j_minus, j_z):
    r"""Apply a su(2) Casimir operator to ``psi`` and return the resulting state.

    Uses the ladder identity :math:`\hat J^2 = \hat J_- \hat J_+ + \hat J_z^2 +
    \hat J_z` (with :math:`\hat J_- = \hat J_+^\dagger`), so only the one-body
    ladder/Cartan operators are needed — no explicit two-body operator product is
    constructed. Each factor is applied sequentially to the state.

    Parameters
    ----------
    psi : ManyBodyState
        The state to act on.
    j_plus, j_minus, j_z : ManyBodyOperator
        The raising, lowering, and Cartan operators of the su(2) algebra.

    Returns
    -------
    ManyBodyState
        :math:`\hat J^2 |\psi\rangle`.
    """
    jz_psi = j_z(psi, 0)
    result = j_minus(j_plus(psi, 0), 0)
    result += j_z(jz_psi, 0)
    result += jz_psi
    return result


def apply_spin_correlation(psi, ops_a, ops_b):
    r"""Apply the spin-correlation operator :math:`\hat{\mathbf S}_A\cdot\hat{\mathbf S}_B`.

    For two **disjoint** orbital sets A and B the spin operators commute, so

    .. math::
        \hat{\mathbf S}_A\cdot\hat{\mathbf S}_B
            = \hat S^A_z \hat S^B_z
            + \tfrac12\left(\hat S^A_+ \hat S^B_- + \hat S^A_- \hat S^B_+\right),

    with no normal-ordering correction. Each factor is applied sequentially.

    Parameters
    ----------
    psi : ManyBodyState
        The state to act on.
    ops_a, ops_b : (ManyBodyOperator, ManyBodyOperator, ManyBodyOperator)
        The ``(S_+, S_-, S_z)`` operators for set A and set B (see
        :func:`make_spin_operators`). A and B must address disjoint orbitals.

    Returns
    -------
    ManyBodyState
        :math:`\hat{\mathbf S}_A\cdot\hat{\mathbf S}_B\,|\psi\rangle`.
    """
    a_plus, a_minus, a_z = ops_a
    b_plus, b_minus, b_z = ops_b
    result = a_z(b_z(psi, 0), 0)
    result += 0.5 * a_plus(b_minus(psi, 0), 0)
    result += 0.5 * a_minus(b_plus(psi, 0), 0)
    return result


def expect_spin_correlation(psi, ops_a, ops_b, comm=None):
    r"""Return :math:`\langle\psi|\hat{\mathbf S}_A\cdot\hat{\mathbf S}_B|\psi\rangle`.

    A negative value signals impurity-bath singlet (Kondo) screening. See
    :func:`apply_spin_correlation` for the operator and disjointness requirement.
    """
    val = inner(psi, apply_spin_correlation(psi, ops_a, ops_b))
    if comm is not None:
        val = comm.allreduce(val)
    return np.real(val)


def expect_casimir(psi, j_plus, j_minus, j_z, comm=None):
    r"""Return :math:`\langle\psi|\hat J^2|\psi\rangle` for a (possibly distributed) state.

    Parameters
    ----------
    psi : ManyBodyState
        The state. Assumed normalised (``inner(psi, psi) == 1``).
    j_plus, j_minus, j_z : ManyBodyOperator
        The su(2) ladder/Cartan operators (see :func:`make_spin_operators`).
    comm : MPI.Comm, optional
        If given, the local inner products are summed across ranks. The state must
        be hash-distributed so that every basis determinant reachable by the
        operators is owned by exactly one rank.

    Returns
    -------
    float
        The expectation value :math:`\langle \hat J^2\rangle`.
    """
    val = inner(psi, apply_casimir(psi, j_plus, j_minus, j_z))
    if comm is not None:
        val = comm.allreduce(val)
    return np.real(val)


def _group_degenerate(energies, tol):
    """Group indices of (ascending-sorted) ``energies`` into near-degenerate blocks.

    Returns a list of lists of indices into ``energies``; consecutive energies
    within ``tol`` of the block's first energy share a block.
    """
    groups = []
    current = [0]
    for i in range(1, len(energies)):
        if abs(energies[i] - energies[current[0]]) <= tol:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    return groups


def manifold_observable_values(eigenstates, energies, apply_op, degeneracy_tol=1e-6, comm=None, redistribute=None):
    r"""Per-state physical values of an observable on a low-energy manifold.

    Block Lanczos returns a *block* spanning a (near-)degenerate eigenspace; any
    single returned vector is an arbitrary combination within a degenerate
    manifold, so :math:`\langle\psi|\hat O|\psi\rangle` on it is not well defined
    when :math:`[\hat O, H]\neq 0` inside the manifold. For each degenerate
    subspace this builds the small matrix :math:`O_{mn}=\langle m|\hat O|n\rangle`
    and diagonalises it; the eigenvalues are the physical observable values.

    This may be evaluated distributed: pass the rank-local states together with
    ``comm`` and a ``redistribute`` callback (e.g. ``Basis.redistribute_psis``). Then
    :math:`\hat O|n\rangle` is formed on each rank's local partition, redistributed to
    align with the bra partition, the local contributions to :math:`O_{mn}` are summed,
    and the small matrix is ``Allreduce``-d. The result is identical on every rank, so no
    state-vector gather is needed. With ``comm=None`` the ``eigenstates`` are treated as
    full (single-rank) states, as before.

    Parameters
    ----------
    eigenstates : sequence of ManyBodyState
        Orthonormal manifold basis (e.g. a Lanczos block); rank-local when ``comm`` is
        given, otherwise full.
    energies : array_like of shape (N,)
        Energy of each eigenstate (used only to group degenerate subspaces).
    apply_op : callable
        ``apply_op(psi)`` returns :math:`\hat O|\psi\rangle` as a ManyBodyState.
    degeneracy_tol : float, default 1e-6
        Energies within this tolerance are treated as degenerate.
    comm : MPI.Comm, optional
        Communicator over which the states are distributed. If ``None`` the computation
        is purely local.
    redistribute : callable, optional
        ``redistribute(op_states)`` returns the operator-applied states reshuffled onto
        the same determinant partition as ``eigenstates`` (e.g. ``Basis.redistribute_psis``).
        Required when the states are distributed.

    Returns
    -------
    np.ndarray of shape (N,)
        Real observable values aligned with ``eigenstates``. States in the same
        degenerate subspace receive that subspace's eigenvalues (their assignment
        within the subspace is arbitrary, the values being physically interchangeable).
    """
    n = len(eigenstates)
    energies = np.asarray(energies, dtype=float)
    op_states = [apply_op(psi) for psi in eigenstates]
    if redistribute is not None:
        op_states = redistribute(op_states)
    o_matrix = np.array(
        [[inner(eigenstates[i], op_states[j]) for j in range(n)] for i in range(n)],
        dtype=complex,
    )
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, o_matrix, op=MPI.SUM)
    # Hermitise to kill rounding asymmetry before diagonalising.
    o_matrix = 0.5 * (o_matrix + o_matrix.conj().T)

    order = np.argsort(energies, kind="stable")
    values = np.empty(n)
    for block in _group_degenerate(energies[order], degeneracy_tol):
        idx = order[block]
        sub = o_matrix[np.ix_(idx, idx)]
        evals = np.linalg.eigvalsh(sub)
        for k, state_index in enumerate(idx):
            values[state_index] = evals[k]
    return values


def thermal_observable_value(values, energies, tau):
    r"""Boltzmann-weighted average of per-state observable ``values``.

    :math:`\langle\hat O\rangle = \sum_n e^{-\beta E_n} o_n / Z`, evaluated with
    the energy-scale convention of :func:`average.thermal_average_scale_indep`.
    ``values`` should be the physical per-state values from
    :func:`manifold_observable_values`, so degenerate manifolds contribute correctly.
    """
    return thermal_average_scale_indep(np.asarray(energies, dtype=float), np.asarray(values), tau)


def casimir_to_quantum_number(jj_plus_1):
    r"""Invert :math:`j(j+1)` to recover the angular-momentum quantum number ``j``.

    Parameters
    ----------
    jj_plus_1 : float
        The Casimir eigenvalue :math:`j(j+1)`.

    Returns
    -------
    float
        ``j = (-1 + sqrt(1 + 4 j(j+1))) / 2`` (clamped at 0 for tiny negatives).
    """
    return 0.5 * (-1.0 + np.sqrt(max(1.0 + 4.0 * np.real(jj_plus_1), 0.0)))


def impurity_spin_pairs(impurity_orbitals):
    r"""Return the ``(dn_index, up_index)`` impurity spin-orbital pairs.

    Within each angular-momentum partition of ``impurity_orbitals`` the first half
    of the spin-orbitals are spin-down and the second half spin-up (the basis
    layout, matching ``Basis._generate_spin_flipped_determinants``), so orbital
    ``k`` pairs with orbital ``k + n//2``.

    Parameters
    ----------
    impurity_orbitals : dict
        Mapping ``partition -> list of orbital-index blocks`` (``Basis.impurity_orbitals``).

    Returns
    -------
    list of (int, int)
        ``(dn, up)`` global spin-orbital index pairs, suitable for
        :func:`make_spin_operators`.
    """
    pairs = []
    for orb_blocks in impurity_orbitals.values():
        orbs = [orb for block in orb_blocks for orb in block]
        n = len(orbs)
        for k in range(n // 2):
            pairs.append((orbs[k], orbs[k + n // 2]))
    return pairs


def bath_spin_pairs(bath_states):
    r"""Return the ``(dn_index, up_index)`` bath spin-orbital pairs.

    Same down-then-up convention as :func:`impurity_spin_pairs`, applied to each
    valence and conduction bath block independently (the ``get_CF_hamiltonian`` /
    ``c2i`` layout, where a bath block is ``[down(2l+1), up(2l+1)]``). Odd-sized blocks
    are skipped. The result is only *trusted* after
    :func:`spin_pairs_consistent_with_h` confirms the induced spin operators commute
    with the one-body Hamiltonian.

    Parameters
    ----------
    bath_states : tuple of dict
        ``(valence_baths, conduction_baths)`` (``Basis.bath_states``).

    Returns
    -------
    list of (int, int)
    """
    pairs = []
    for baths in bath_states:
        for blocks in baths.values():
            for block in blocks:
                n = len(block)
                if n % 2 != 0:
                    continue
                for k in range(n // 2):
                    pairs.append((block[k], block[k + n // 2]))
    return pairs


def spin_pairs_consistent_with_h(h_op, spin_pairs, n_orb, tol=1e-6):
    r"""Whether the spin operators from ``spin_pairs`` commute with the one-body ``h``.

    Builds the single-particle ``S_z`` and ``S_+`` matrices implied by the ``(dn, up)``
    pairing and checks ``[h, S_z] = [h, S_+] = 0``. If both hold, the spin labelling
    **and** the down↔up pairing are consistent with the Hamiltonian's spin symmetry, so
    the spin operators are physically correct. If either fails (spin-orbit coupling, a
    non-standard orbital ordering, …), the pairing is **not** trustworthy and spin-spin
    observables built from it should be skipped rather than reported wrong.

    Parameters
    ----------
    h_op : ManyBodyOperator or dict
        The Hamiltonian (its one-body part is used).
    spin_pairs : sequence of (int, int)
        ``(dn, up)`` global spin-orbital pairs (impurity + bath).
    n_orb : int
        Total number of spin-orbitals.
    tol : float, optional
        Commutator norm tolerance.

    Returns
    -------
    bool
    """
    from impurityModel.ed.symmetries import extract_tensors

    h, _, _ = extract_tensors(h_op, n_orb=n_orb)
    sz = np.zeros((n_orb, n_orb), dtype=complex)
    splus = np.zeros((n_orb, n_orb), dtype=complex)
    for dn, up in spin_pairs:
        sz[up, up] += 0.5
        sz[dn, dn] -= 0.5
        splus[up, dn] += 1.0  # S_+ = c†_up c_dn
    return bool(np.linalg.norm(h @ sz - sz @ h) <= tol and np.linalg.norm(h @ splus - splus @ h) <= tol)


def derive_spin_pairs(h_op, impurity_orbitals, rot_to_spherical, n_orb, tol=1e-6):
    r"""Derive the global ``(down, up)`` spin-orbital pairing from the one-body Hamiltonian.

    Geometry-agnostic alternative to the down-then-up index convention of
    :func:`impurity_spin_pairs` / :func:`bath_spin_pairs`, which is only valid in the
    spherical-harmonics representation. It is needed for bath geometries (e.g. the linked
    double-chain / Haverkort bath) where the computational orbital order is not
    down-then-up.

    Two ingredients:

    1. **Impurity** — the pairing is read from the spherical spin-raising operator
       :math:`S_+` rotated to the computational basis via ``rot_to_spherical``: the single
       non-zero in each column marks a ``(down, up)`` partner. This carries no index-order
       assumption. If the rotated :math:`S_+` is not a clean permutation (spin-orbit
       coupling mixes the impurity spins) the derivation gives up.
    2. **Bath** — propagated outward along the Hamiltonian's hopping graph. Because the
       one-body ``h`` is spin-blind for a collinear model (``h_up == h_dn``), the spin-down
       and spin-up sectors form identical connectivity blocks. Starting from each impurity
       ``(down, up)`` pair, a simultaneous breadth-first search matches each spin-down bath
       orbital with the structurally identical (same hopping magnitude and on-site energy)
       spin-up bath orbital.

    The result is a *candidate* that must still be confirmed with
    :func:`spin_pairs_consistent_with_h`. Returns ``None`` when the pairing cannot be
    determined unambiguously (spin-orbit coupling, a bath orbital disconnected from the
    impurity, or a structurally ambiguous match).

    Parameters
    ----------
    h_op : ManyBodyOperator or dict
        The Hamiltonian (its one-body part is used).
    impurity_orbitals : dict
        ``Basis.impurity_orbitals`` (``partition -> list of orbital-index blocks``).
    rot_to_spherical : np.ndarray or dict
        The spherical->computational rotation (single matrix or ``{partition: matrix}``).
    n_orb : int
        Total number of spin-orbitals.
    tol : float, optional
        Magnitude tolerance for graph edges and structural matching.

    Returns
    -------
    (imp_pairs, bath_pairs) : tuple of list of (int, int), or None
    """
    from impurityModel.ed.symmetries import extract_tensors

    h, _, _ = extract_tensors(h_op, n_orb=n_orb)
    impurity_orbs = set(orb for blocks in impurity_orbitals.values() for block in blocks for orb in block)

    # --- impurity (dn, up) pairs from the rotated spherical S_+ ---
    imp_pairs = []
    for partition, blocks in impurity_orbitals.items():
        orbs = [orb for block in blocks for orb in block]
        n_so = len(orbs)
        shell_l = (n_so // 2 - 1) // 2
        if 2 * (2 * shell_l + 1) != n_so:
            return None
        _, _, _, _, sp_m, _ = _single_particle_lsj_matrices(shell_l)
        rot = rot_to_spherical[partition] if isinstance(rot_to_spherical, dict) else rot_to_spherical
        rot = np.asarray(rot, dtype=complex)
        sp_comp = rot @ sp_m @ rot.conj().T  # S_+ in the computational basis
        downs = {}  # local down index -> local up index
        for j in range(n_so):
            rows = [i for i in range(n_so) if abs(sp_comp[i, j]) > 0.5]
            if len(rows) > 1 or (len(rows) == 1 and abs(abs(sp_comp[rows[0], j]) - 1.0) > 1e-3):
                return None  # not a clean spin-eigen pairing (e.g. SOC)
            if len(rows) == 1:
                downs[j] = rows[0]
        ups = set(downs.values())
        if len(downs) != n_so // 2 or len(ups) != n_so // 2 or (set(downs) & ups):
            return None
        for j, i in downs.items():
            imp_pairs.append((orbs[j], orbs[i]))

    # --- bath pairs by simultaneous BFS over the hopping graph, seeded by imp_pairs ---
    h_abs = np.abs(h)
    matched = {}
    for dn, up in imp_pairs:
        matched[dn] = up
        matched[up] = dn

    def bath_neighbors(x):
        return [y for y in range(n_orb) if y != x and y not in impurity_orbs and h_abs[x, y] > tol]

    bath_pairs = []
    queue = deque(imp_pairs)
    while queue:
        dn, up = queue.popleft()
        up_candidates = [y for y in bath_neighbors(up) if y not in matched]
        for ndn in bath_neighbors(dn):
            if ndn in matched:
                continue
            cands = [
                nup
                for nup in up_candidates
                if nup not in matched
                and abs(h_abs[dn, ndn] - h_abs[up, nup]) <= tol
                and abs(h[ndn, ndn] - h[nup, nup]) <= tol
            ]
            if len(cands) != 1:
                return None  # no match or ambiguous
            nup = cands[0]
            matched[ndn] = nup
            matched[nup] = ndn
            bath_pairs.append((ndn, nup))
            queue.append((ndn, nup))

    bath_orbs = set(range(n_orb)) - impurity_orbs
    if any(orb not in matched for orb in bath_orbs):
        return None  # some bath orbital disconnected from the impurity
    return imp_pairs, bath_pairs


def print_thermal_expectation_values(
    rho_thermal,
    e_thermal,
    rot_to_spherical,
    block_structure,
    s_thermal=None,
    l_thermal=None,
    j_thermal=None,
    sisb_thermal=None,
):
    """
    print several thermal expectation values, e.g. E, N, Sz, Lz.

    If ``s_thermal`` / ``l_thermal`` / ``j_thermal`` are given (the thermally-averaged
    impurity ``S(S+1)`` / ``L(L+1)`` / ``J(J+1)``), the corresponding ``<S^2>`` /
    ``<L^2>`` / ``<J^2>`` lines (with the quantum number) are appended. When all are
    ``None`` the output is identical to before.
    """
    orb_offset = min(orb for block in block_structure.blocks for orb in block)
    equivalent_blocks = get_equivalent_blocks(block_structure)
    rho_thermal_spherical = rotate_matrix(rho_thermal, rot_to_spherical)
    N, Ndn, Nup = get_occupations_from_rho_spherical(rho_thermal_spherical)

    # Collect (label, value, suffix) rows, then print with the '=' signs aligned and the
    # numbers right-aligned (sign-padded), so the column reads as a tidy table.
    rows = [
        ("<E-E0>", e_thermal, ""),
        ("<N>", N, ""),
        ("<N(Dn)>", Ndn, ""),
        ("<N(Up)>", Nup, ""),
    ]
    for blocks in equivalent_blocks:
        occ = np.sum(
            np.diag(rho_thermal)[list(orb - orb_offset for block in blocks for orb in block_structure.blocks[block])]
        ).real
        rows.append((f"<N({','.join(str(orb) for orb in blocks)})>", occ, ""))
    rows.append(("<Lz>", get_Lz_from_rho_spherical(rho_thermal_spherical), ""))
    rows.append(("<Sz>", get_Sz_from_rho_spherical(rho_thermal_spherical), ""))
    rows.append(("<L.S>", get_LS_from_rho_spherical(rho_thermal_spherical), ""))
    for label, value in (("S", s_thermal), ("L", l_thermal), ("J", j_thermal)):
        if value is not None:
            rows.append((f"<{label}^2>", np.real(value), f"({label} = {casimir_to_quantum_number(value): 6.4f})"))
    if sisb_thermal is not None:
        rows.append(("<S_imp.S_bath>", np.real(sisb_thermal), ""))

    label_width = max(len(label) for label, _, _ in rows)
    for label, value, suffix in rows:
        line = f"{label:<{label_width}} = {value: 12.7f}"
        if suffix:
            line += f"  {suffix}"
        print(line)


def dc_MLFT(n3d_i, c, Fdd, n2p_i=None, Fpd=None, Gpd=None):
    r"""
    Return double counting (DC) in multiplet ligand field theory.

    Parameters
    ----------
    n3d_i : int
        Nominal (integer) 3d occupation.
    c : float
        Many-body correction to the charge transfer energy.
    n2p_i : int
        Nominal (integer) 2p occupation.
    Fdd : list
        Slater integrals {F_{dd}^k}, k \in [0,1,2,3,4]
    Fpd : list
        Slater integrals {F_{pd}^k}, k \in [0,1,2]
    Gpd : list
        Slater integrals {G_{pd}^k}, k \in [0,1,2,3]

    Notes
    -----
    The `c` parameter is related to the charge-transfer
    energy :math:`\Delta_{CT}` by:

    .. math:: \Delta_{CT} = (e_d-e_b) + c.

    """
    if not int(n3d_i) == n3d_i:
        raise ValueError("3d occupation should be an integer")
    if n2p_i is not None and int(n2p_i) != n2p_i:
        raise ValueError("2p occupation should be an integer")

    # Average repulsion energy defines Udd and Upd
    Udd = Fdd[0] - 14.0 / 441 * (Fdd[2] + Fdd[4])
    if n2p_i is None and Fpd is None and Gpd is None:
        return {2: Udd * n3d_i - c}
    if n2p_i == 6 and Fpd is not None and Gpd is not None:
        Upd = Fpd[0] - (1 / 15.0) * Gpd[1] - (3 / 70.0) * Gpd[3]
        return {2: Udd * n3d_i + Upd * n2p_i - c, 1: Upd * (n3d_i + 1) - c}
    else:
        raise ValueError("double counting input wrong.")


def get_spherical_2_cubic_matrix(spinpol=False, l=2):
    r"""
    Return unitary ndarray for transforming from spherical to cubic harmonics.

    Parameters
    ----------
    spinpol : boolean
        If transformation involves spin.
    l : integer
        Angular momentum number. p: l=1, d: l=2.

    Returns
    -------
    u : (M,M) ndarray
        The unitary matrix from spherical to cubic harmonics.

    Notes
    -----
    Element :math:`u_{i,j}` represents the contribution of spherical
    harmonics :math:`i` to the cubic harmonic :math:`j`:

    .. math:: \lvert l_j \rangle  = \sum_{i=0}^4 u_{d,(i,j)}
        \lvert Y_{d,i} \rangle.

    """
    if l == 1:
        # u = np.zeros((3,3),dtype=complex)
        u = np.zeros((3, 3), dtype=complex)
        u[0, 0] = 1j / np.sqrt(2)
        u[2, 0] = 1j / np.sqrt(2)
        u[0, 1] = 1 / np.sqrt(2)
        u[2, 1] = -1 / np.sqrt(2)
        u[1, 2] = 1
    elif l == 2:
        # u = np.zeros((5,5),dtype=complex)
        u = np.zeros((5, 5), dtype=complex)
        u[2, 0] = 1
        u[[0, -1], 1] = 1 / np.sqrt(2)
        u[1, 2] = -1j / np.sqrt(2)
        u[-2, 2] = -1j / np.sqrt(2)
        u[1, 3] = 1 / np.sqrt(2)
        u[-2, 3] = -1 / np.sqrt(2)
        u[0, 4] = 1j / np.sqrt(2)
        u[-1, 4] = -1j / np.sqrt(2)
    if spinpol:
        n, m = np.shape(u)
        # U = np.zeros((2*n,2*m),dtype=complex)
        U = np.zeros((2 * n, 2 * m), dtype=complex)
        U[0:n, 0:m] = u
        U[n:, m:] = u
        u = U
    return u


def daggerOp(op):
    """
    return op^dagger
    """
    opDagger = {}
    for process, value in op.items():
        processNew = []
        for e in process[::-1]:
            if e[1] == "a":
                processNew.append((e[0], "c"))
            elif e[1] == "c":
                processNew.append((e[0], "a"))
            else:
                raise Exception("Operator type unknown: {}".format(e[1]))
        processNew = tuple(processNew)
        opDagger[processNew] = value.conjugate()
    return opDagger


def assert_hermitian(op: dict[tuple, int | float | complex]) -> None:
    """Assert that the operator is Hermitian (equal to its adjoint).

    Parameters
    ----------
    op : dict
        The operator representing a mapping from excitation tuples to amplitudes.
    """
    assert daggerOp(op) == op


def gauntC(k, l, m, lp, mp, prec=16):
    """
    return "nonvanishing" Gaunt coefficients of
    Coulomb interaction expansion.
    """
    c = sqrt(4 * pi / (2 * k + 1)) * (-1) ** m * gaunt(l, k, lp, -m, m - mp, mp, prec=prec)
    return float(c)


def getU(l1, m1, l2, m2, l3, m3, l4, m4, R):
    r"""
    Return Hubbard U term for four spherical harmonics functions.

    Scattering process:

    :math:`u_{l_1,m_1,l_2,m_2,l_3,m_3,l_4,m_4}
    * c_{l_1,m_1}^\dagger c_{l_2,m_2}^\dagger c_{l_3,m_3} c_{l_4,m_4}`.

    Parameters
    ----------
    l1 : int
        angular momentum of orbital 1
    m1 : int
        z projected angular momentum of orbital 1
    l2 : int
        angular momentum of orbital 2
    m2 : int
        z projected angular momentum of orbital 2
    l3 : int
        angular momentum of orbital 3
    m3 : int
        z projected angular momentum of orbital 3
    l4 : int
        angular momentum of orbital 4
    m4 : int
        z projected angular momentum of orbital 4
    R : list
        Slater-Condon parameters.
        Elements R[k] fullfill
        :math:`0<=k<=\textrm{min}(|l_1+l_4|,|l_2+l_3|)`.
        Note, U is nonzero if :math:`k+l_1+l_4` is an even integer
        and :math:`k+l_3+l_2` is an even integer.
        For example: if :math:`l_1=l_2=l_3=l_4=2`,
        R = [R0,R1,R2,R3,R4] and only R0,R2 and R4 will
        give nonzero contribution.

    Returns
    -------
    u - float
        Hubbard U term.
    """
    # Check if angular momentum is conserved
    if m1 + m2 == m3 + m4:
        u = 0
        for k, Rk in enumerate(R):
            u += Rk * gauntC(k, l1, m1, l4, m4) * gauntC(k, l3, m3, l2, m2)
    else:
        u = 0
    return u


def getUop_from_rspt_u4(u4: np.ndarray) -> dict:
    """Convert a 4-index U matrix from RSPT format to an operator dictionary.

    Parameters
    ----------
    u4 : np.ndarray
        The 4D array representing the Hubbard U matrix.

    Returns
    -------
    uDict : dict
        The converted operator dictionary.
    """
    l1, l2, l3, l4 = u4.shape
    l1 = ((l1 // 2) - 1) // 2
    l2 = ((l2 // 2) - 1) // 2
    l3 = ((l3 // 2) - 1) // 2
    l4 = ((l4 // 2) - 1) // 2
    uDict = {}
    for i, j, k, l in itertools.product(range(u4.shape[0]), range(u4.shape[1]), range(u4.shape[2]), range(u4.shape[3])):
        u = u4[i, j, k, l]
        if abs(u) > 1e-10:
            proccess = (
                (i, "c"),
                (j, "c"),
                (k, "a"),
                (l, "a"),
            )
            uDict[proccess] = u / 2
    return uDict


def getUop(l1, l2, l3, l4, R):
    r"""
    Return U operator.

    Scattering processes:
    :math:`1/2 \sum_{m_1,m_2,m_3,m_4} u_{l_1,m_1,l_2,m_2,l_3,m_3,l_4,m_4}
    * \sum_{s,sp} c_{l_1, s, m_1}^\dagger c_{l_2, sp, m_2}^\dagger
    c_{l_3, sp, m_3} c_{l_4, s, m_4}`.

    Spin polarization is considered, thus basis: (l, s, m),
    where :math:`s \in \{0, 1 \}` and these indices respectively
    corresponds to the physical values
    :math:`\{-\frac{1}{2},\frac{1}{2} \}`.

    Returns
    -------
    uDict : dict
        Elements of the form:
        ((sorb1,'c'),(sorb2,'c'),(sorb3,'a'),(sorb4,'a')) : u/2
        where sorb1 is a superindex of (l, s, m).

    """
    uDict = {}
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            for m3 in range(-l3, l3 + 1):
                for m4 in range(-l4, l4 + 1):
                    u = getU(l1, m1, l2, m2, l3, m3, l4, m4, R)
                    if u != 0:
                        for s in range(2):
                            for sp in range(2):
                                proccess = (
                                    ((l1, s, m1), "c"),
                                    ((l2, sp, m2), "c"),
                                    ((l3, sp, m3), "a"),
                                    ((l4, s, m4), "a"),
                                )
                                # Pauli exclusion principle
                                if not (s == sp and ((l1, m1) == (l2, m2) or (l3, m3) == (l4, m4))):
                                    uDict[proccess] = u / 2.0
    return uDict


def addOps(ops):
    """
    Return one operator, represented as a dictonary.

    Parameters
    ----------
    ops : list
        Operators

    Returns
    -------
    opSum : dict

    """
    opSum = {}
    for op in ops:
        for sOp, value in op.items():
            opSum[sOp] = opSum.get(sOp, 0) + value
            # if np.abs(value) > 1e-12:
            #     if sOp in opSum:
            #         opSum[sOp] += value
            #     else:
            #         opSum[sOp] = value
    return opSum


def get2p3dSlaterCondonUop(Fdd=(9, 0, 8, 0, 6), Fpp=(20, 0, 8), Fpd=(10, 0, 8), Gpd=(0, 3, 0, 2)):
    """
    Return a 2p-3d U operator containing a sum of
    different Slater-Condon proccesses.

    Parameters
    ----------
    Fdd : tuple
    Fpp : tuple
    Fpd : tuple
    Gpd : tuple

    """
    # Calculate F_dd^{0,2,4}
    FddOp = {}
    if Fdd is not None:
        FddOp = getUop(l1=2, l2=2, l3=2, l4=2, R=Fdd)
    # Calculate F_pp^{0,2}
    FppOp = {}
    if Fpp is not None:
        FppOp = getUop(l1=1, l2=1, l3=1, l4=1, R=Fpp)
    # Calculate F_pd^{0,2}
    FpdOp = {}
    if Fpd is not None:
        FpdOp1 = getUop(l1=1, l2=2, l3=2, l4=1, R=Fpd)
        FpdOp2 = getUop(l1=2, l2=1, l3=1, l4=2, R=Fpd)
        FpdOp = addOps([FpdOp1, FpdOp2])
    # Calculate G_pd^{1,3}
    GpdOp = {}
    if Gpd is not None:
        GpdOp1 = getUop(l1=1, l2=2, l3=1, l4=2, R=Gpd)
        GpdOp2 = getUop(l1=2, l2=1, l3=2, l4=1, R=Gpd)
        GpdOp = addOps([GpdOp1, GpdOp2])
    # Add operators
    uOp = addOps([FddOp, FppOp, FpdOp, GpdOp])
    return uOp


def getSOCop(xi, l=2):
    """
    Return SOC operator for one l-shell.

    Parameters
    ----------
    xi : float
        Spin-orbit coupling constant.
    l : int, default 2
        Angular momentum quantum number.

    Returns
    -------
    uDict : dict
        Elements of the form:
        (((l, s1, m1),'c'), ((l, s2, m2),'a')) : h_value
        where (l, s, m) is the state.
    """
    opDict = {}
    for m in range(-l, l + 1):
        for s in range(2):
            value = xi * m * (1 / 2.0 if s == 1 else -1 / 2.0)
            opDict[(((l, s, m), "c"), ((l, s, m), "a"))] = value
    for m in range(-l, l):
        value = xi / 2.0 * sqrt((l - m) * (l + m + 1))
        opDict[(((l, 1, m), "c"), ((l, 0, m + 1), "a"))] = value
        opDict[(((l, 0, m + 1), "c"), ((l, 1, m), "a"))] = value
    return opDict


def gethHfieldop(hx, hy, hz, l=2):
    """
    Return magnetic field operator for one l-shell.

    Parameters
    ----------
    hx : float
        Magnetic field x-component.
    hy : float
        Magnetic field y-component.
    hz : float
        Magnetic field z-component.
    l : int, default 2
        Angular momentum quantum number.

    Returns
    -------
    hHfieldOperator : dict
        Elements of the form:
        (((l, s1, m1),'c'), ((l, s2, m2),'a')) : h_value
    """
    hHfieldOperator = {}
    for m in range(-l, l + 1):
        hHfieldOperator[(((l, 1, m), "c"), ((l, 0, m), "a"))] = hx / 2
        hHfieldOperator[(((l, 0, m), "c"), ((l, 1, m), "a"))] = hx / 2
        hHfieldOperator[(((l, 1, m), "c"), ((l, 0, m), "a"))] += -hy * 1j / 2
        hHfieldOperator[(((l, 0, m), "c"), ((l, 1, m), "a"))] += hy * 1j / 2
        for s in range(2):
            hHfieldOperator[(((l, s, m), "c"), ((l, s, m), "a"))] = hz / 2 if s == 1 else -hz / 2
    return hHfieldOperator


def c2i(nBaths, spinOrb):
    """
    Return an index, representing a spin-orbital or a bath state.

    Parameters
    ----------
    nBaths : ordered dict
        An elements is either of the form:
        angular momentum : number of bath spin-orbitals
        or of the form:
        (angular momentum_a, angular momentum_b, ...) : number of bath states.
        The latter form is used if impurity orbitals from different
        angular momenta share the same bath states.
    spinOrb : tuple
        (l, s, m), (l, b) or ((l_a, l_b, ...), b)

    Returns
    -------
    i : int
        An index denoting a spin-orbital or a bath state.

    """
    # Counting index and return variable.
    i = 0
    # Check if spinOrb is an impurity spin-orbital.
    # Loop through all impurity spin-orbitals.
    for lp in nBaths.keys():
        if isinstance(lp, int):
            for sp in range(2):
                for mp in range(-lp, lp + 1):
                    if (lp, sp, mp) == spinOrb:
                        return i
                    i += 1
        elif isinstance(lp, tuple):
            # Loop over all different angular momenta in lp.
            for lp_int in lp:
                for sp in range(2):
                    for mp in range(-lp_int, lp_int + 1):
                        if (lp_int, sp, mp) == spinOrb:
                            return i
                        i += 1
    # If reach this point it means spinOrb is a bath state.
    # Need to figure out which one index is has.
    for lp, nBath in nBaths.items():
        for b in range(nBath):
            if (lp, b) == spinOrb:
                return i
            i += 1
    raise Exception("Can not find index corresponding to spin-orbital state")


def i2c(nBaths, i):
    """
    Return an coordinate tuple, representing a spin-orbital.

    Parameters
    ----------
    nBaths : ordered dict
        An elements is either of the form:
        angular momentum : number of bath spin-orbitals
        or of the form:
        (angular momentum_a, angular momentum_b, ...) : number of bath states.
        The latter form is used if impurity orbitals from different
        angular momenta share the same bath states.
    i : int
        An index denoting a spin-orbital or a bath state.

    Returns
    -------
    spinOrb : tuple
        (l, s, m), (l, b) or ((l_a, l_b, ...), b)

    """
    # Counting index.
    k = 0
    # Check if index "i" belong to an impurity spin-orbital.
    # Loop through all impurity spin-orbitals.
    for lp in nBaths.keys():
        if isinstance(lp, int):
            # Check if index "i" belong to impurity spin-orbital having lp.
            if i - k < 2 * (2 * lp + 1):
                for sp in range(2):
                    for mp in range(-lp, lp + 1):
                        if k == i:
                            return (lp, sp, mp)
                        k += 1
            k += 2 * (2 * lp + 1)
        elif isinstance(lp, tuple):
            # Loop over all different angular momenta in lp.
            for lp_int in lp:
                # Check if index "i" belong to impurity spin-orbital having lp_int.
                if i - k < 2 * (2 * lp_int + 1):
                    for sp in range(2):
                        for mp in range(-lp_int, lp_int + 1):
                            if k == i:
                                return (lp_int, sp, mp)
                            k += 1
                k += 2 * (2 * lp_int + 1)
    # If reach this point it means index "i" belong to a bath state.
    # Need to figure out which one.
    for lp, nBath in nBaths.items():
        b = i - k
        # Check if bath state belong to bath states having lp.
        if b < nBath:
            # The index "b" will have a value between 0 and nBath-1
            return (lp, b)
        k += nBath
    raise Exception("Can not find spin-orbital state corresponding to index.")


def arrayOp2Dict(nbaths, opsArray):
    r"""
    Return an array of dicts of the form {(i,j):val ...} corresponding to the
    operators c^dagger_i c_j, stored in the opsArray
    Parameters
    ----------
    nbaths : dict
        l : nb
    opsArray : [dict]
        [{(((l, m, s, b), 'c'), ((l, m, s, b), 'a'))}, ...]
    """
    res = []
    for ops in opsArray:
        res.append(op2Dict(nbaths, ops))
    return res


def op2Dict(nBaths, ops):
    r"""
    returns a dict of the form {(i,j):val,...} correspoding to the opeator c^dagger_i c_j
    where i, j are obtained from c2i
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    ops: dict
        Multi configurational state.
    """
    d = {}
    for t, val in ops.items():
        # Only one particle terms
        if len(t) == 2:
            # Accept both ((l,m,s,b),'c/a') and (l,m,s,b)
            if len(t[0]) == 2:
                if t[0][1] == "c" and t[1][1] == "a":
                    d[((c2i(nBaths, t[0][0]), "c"), (c2i(nBaths, t[1][0]), "a"))] = val
                elif t[0][1] == "a" and t[1][1] == "c":
                    if t[0][0] == t[1][0]:
                        d[((c2i(nBaths, t[1][0]), "c"), (c2i(nBaths, t[0][0]), "a"))] = 1.0 - val
                    else:
                        d[((c2i(nBaths, t[1][0]), "c"), (c2i(nBaths, t[0][0]), "a"))] = -val
            else:
                d[((c2i(nBaths, t[0]), "c"), (c2i(nBaths, t[1]), "a"))] = val
    return d


def combineOp(nBaths, op1, op2):
    r"""
    Return a dict of the form {(i, j) : val, ...} corresponding to the
    operator op1*op2

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    op1 : dict
        Operator dictionary {(i, j) : val}
    op2 : dict
        Operator dictionary {(i, j) : val}

    Returns
    -------
    newOp : dict
        Combined operator dictionary
    """
    mOp1 = iOpToMatrix(nBaths, op1)
    mOp2 = iOpToMatrix(nBaths, op2)

    newOp = np.matmul(mOp1, mOp2)

    return matrixToIOp(newOp)


def iOpToMatrix(nBaths, op):
    r"""
    Return the matrix representation of op

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    op : dict
        Operator dictionary {(i, j) : val}

    Returns
    -------
    m : numpy.ndarray
        Dense matrix representation of the operator
    """
    dsize = 0
    for l, nb in nBaths.items():
        dsize += nb + (2 * l + 1) * 2
    m = np.zeros((dsize, dsize), dtype=complex)
    for ((i, opi), (j, opj)), val in op.items():
        if opi == "c" and opj == "a":
            m[i, j] = val
        elif opj == "c" and opi == "a":
            if i == j:
                m[i, j] = 1 - val
            else:
                m[i, j] = -val
    return m


def matrixToIOp(mat):
    r"""
    Return a dict containing the non-zero elements of the matrix mat

    Parameters
    ----------
    mat : numpy.ndarray
        Matrix representation of the operator

    Returns
    -------
    res : dict
        Operator dictionary {(i, j) : val}
    """
    rows, columns = mat.shape
    res = {}
    for i in range(rows):
        for j in range(columns):
            if abs(mat[i, j]) > 0:
                res[((i, "c"), (j, "a"))] = mat[i, j]
    return res


def i2c_op(nBaths: dict[int, int], i_op: dict) -> dict:
    """Convert an operator dictionary from flat indices to spin-orbital labels.

    Parameters
    ----------
    nBaths : dict
        Number of bath states for each l quantum number.
    i_op : dict
        Operator dictionary with flat index keys.

    Returns
    -------
    dict
        Operator dictionary with spin-orbital label keys.
    """
    c_op = {}
    for ((i, opi), (j, opj)), val in i_op.items():
        c_op[((i2c(nBaths, i), opi), (i2c(nBaths, j), opj))] = val
    return c_op
