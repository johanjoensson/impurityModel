import numpy as np
np.random.seed(42)
n_blocks = 2
states = [np.random.rand() + 1j * np.random.rand() for _ in range(70)]
v1 = np.array(states)
v1 /= np.linalg.norm(v1)

states2 = [np.random.rand() + 1j * np.random.rand() for _ in range(70)]
v2 = np.array(states2)
v2 /= np.linalg.norm(v2)

print("v1^T v2 =", np.vdot(v1, v2))
