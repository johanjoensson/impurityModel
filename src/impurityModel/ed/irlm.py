import numpy as np
import scipy.linalg as sp

from impurityModel.ed.BlockLanczosArray import block_apply, block_combine, block_normalize, block_orthogonalize, is_array


def implicitly_restarted_block_lanczos_cy(
    psi0,
    h_op,
    basis,
    num_wanted: int,
    max_subspace_blocks: int,
    tol: float = 1e-8,
    max_restarts: int = 100,
    verbose: bool = True,
    slaterWeightMin: float = 0.0,
    reort=None,
    comm=None,
):
    """Thin wrapper delegating to the Cython IRLM implementation.

    Finds the ``num_wanted`` algebraically smallest eigenvalues of ``h_op``
    operating on ``ManyBodyState`` objects via the implicitly-restarted
    block Lanczos (IRLM) algorithm implemented in
    :mod:`impurityModel.ed.BlockLanczos`.

    Args:
        psi0: Starting block of ``ManyBodyState`` objects (length ``p``).
        h_op: ``ManyBodyOperator`` Hamiltonian.
        basis: ``Basis`` providing ``redistribute_psis`` and ``comm``.
        num_wanted: Number of wanted lowest eigenvalues.
        max_subspace_blocks: Maximum Krylov subspace size in blocks.
        tol: Convergence tolerance. Default ``1e-8``.
        max_restarts: Maximum implicit restarts. Default ``100``.
        verbose: Verbosity. Default ``True``.
        slaterWeightMin: Amplitude cutoff. Default ``0.0``.
        reort: Reorthogonalization mode (``Reort`` enum, string, or
            ``None`` which defaults to ``'partial'``).
        comm: ``mpi4py`` communicator or ``None``.

    Returns:
        tuple[numpy.ndarray, list]: ``(eigvals, eigvecs)``.
    """
    from impurityModel.ed.BlockLanczos import (
        implicitly_restarted_block_lanczos_cy as _cy,
    )
    from impurityModel.ed.BlockLanczosArray import Reort

    if reort is None:
        _reort_str = "partial"
    elif isinstance(reort, Reort):
        _reort_str = reort.name.lower()
    else:
        _reort_str = str(reort).lower()

    return _cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=num_wanted,
        max_subspace_blocks=max_subspace_blocks,
        tol=tol,
        max_restarts=max_restarts,
        verbose=verbose,
        slaterWeightMin=slaterWeightMin,
        reort=_reort_str,
        comm=comm,
    )
