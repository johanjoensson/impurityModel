import numpy as np
import scipy.linalg as sp

def implicit_qr_step_block(T, n, shift, U_k=None):
    """
    Applies a single implicit QR shift to a block tridiagonal matrix T
    using block bulge chasing.
    T is modified in place.
    If U_k is provided, the transformations are accumulated into it.
    
    This avoids forming a full dense N x N QR decomposition, reducing
    complexity from O((mn)^3) to O(m * n^3).
    """
    N = T.shape[0]
    m = N // n
    
    # 1. Initial block rotation
    # x is 2n x n block
    x = T[0:2*n, 0:n].copy()
    x -= shift * np.eye(2*n, n)
    q0, r0 = sp.qr(x)
    
    T[0:2*n, :] = q0.conj().T @ T[0:2*n, :]
    T[:, 0:2*n] = T[:, 0:2*n] @ q0
    if U_k is not None:
        U_k[:, 0:2*n] = U_k[:, 0:2*n] @ q0

    # 2. Chase the block bulges
    for i in range(1, m - 1):
        # The bulge is in T[ (i+1)n : (i+2)n, (i-1)n : i*n ]
        # We zero it out using the diagonal block above it: T[ i*n : (i+1)n, (i-1)n : i*n ]
        bulge_col = T[i*n:(i+2)*n, (i-1)*n:i*n].copy()
        qi, ri = sp.qr(bulge_col)
        
        T[i*n:(i+2)*n, :] = qi.conj().T @ T[i*n:(i+2)*n, :]
        T[:, i*n:(i+2)*n] = T[:, i*n:(i+2)*n] @ qi
        if U_k is not None:
            U_k[:, i*n:(i+2)*n] = U_k[:, i*n:(i+2)*n] @ qi
            
    return T, U_k

