import numpy as np

# Dimension
N = 70
n = 2

# Random H
H = np.random.randn(N, N)
H = H + H.T

# Random psi0 (not orthogonal!)
psi0 = np.random.randn(N, n)
psi0 = psi0 / np.linalg.norm(psi0, axis=0)

# Lanczos step 0
wp = H @ psi0
alphas = psi0.T @ wp
wp = wp - psi0 @ alphas

M = wp.T @ wp
try:
    L = np.linalg.cholesky(M)
    print("Cholesky SUCCESS")
except np.linalg.LinAlgError:
    print("Cholesky FAILED")
