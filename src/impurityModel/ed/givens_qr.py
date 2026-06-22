import numpy as np
import scipy.linalg as sp


def implicit_qr_step_block(T, n, shift, U_k=None):
    """Apply a single implicit QR shift to a block tridiagonal matrix T.

    Applies a bulge-chasing implicit QR step with a given shift:

    .. math::

        T \\leftarrow U^\\dagger T U

    where :math:`U` is the accumulated sequence of unitary block Givens
    rotations. This avoids forming the full dense QR decomposition, reducing
    complexity from :math:`\\mathcal{O}((mn)^3)` to :math:`\\mathcal{O}(m n^3)`.
    The transformations are applied directly in-place to the block-tridiagonal
    matrix ``T``.

    If the transformation matrix ``U_k`` is provided, it is updated as
    :math:`U_k \\leftarrow U_k U`.

    Args:
        T: The full block-tridiagonal matrix of shape ``(m*n, m*n)``. Modified
            in-place.
        n: The block size :math:`p`.
        shift: The implicit shift (typically an unwanted Ritz value) to apply.
        U_k: Optional transformation matrix of shape ``(m*n, m*n)`` to accumulate
            the unitary rotations. Modified in-place.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray | None]: A tuple ``(T, U_k)`` where:
            * ``T`` is the updated block-tridiagonal matrix.
            * ``U_k`` is the updated transformation matrix, or ``None``.
    """
    N = T.shape[0]
    m = N // n

    # 1. Initial block rotation
    # x is 2n x n block
    x = T[0 : 2 * n, 0:n].copy()
    x -= shift * np.eye(2 * n, n)
    q0, r0 = sp.qr(x)

    T[0 : 2 * n, :] = q0.conj().T @ T[0 : 2 * n, :]
    T[:, 0 : 2 * n] = T[:, 0 : 2 * n] @ q0
    if U_k is not None:
        U_k[:, 0 : 2 * n] = U_k[:, 0 : 2 * n] @ q0

    # 2. Bulge chasing
    for j in range(1, m - 1):
        # We want to annihilate the subdiagonal block T[j+1, j-1]
        x = T[j * n : (j + 2) * n, (j - 1) * n : j * n].copy()
        q, r = sp.qr(x)

        T[j * n : (j + 2) * n, :] = q.conj().T @ T[j * n : (j + 2) * n, :]
        T[:, j * n : (j + 2) * n] = T[:, j * n : (j + 2) * n] @ q
        if U_k is not None:
            U_k[:, j * n : (j + 2) * n] = U_k[:, j * n : (j + 2) * n] @ q

    return T, U_k
