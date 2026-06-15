import numpy as np
import itertools
from impurityModel.ed.ManyBodyUtils import SlaterDeterminant, ManyBodyState, ManyBodyOperator

n_sites = 8
n_particles = 4

combinations = list(itertools.combinations(range(n_sites), n_particles))
states = [SlaterDeterminant(c) for c in combinations]

op_dict = {}
for i in range(n_sites - 1):
    op_dict[((i, 'c'), (i+1, 'a'))] = -1.0
    op_dict[((i+1, 'c'), (i, 'a'))] = -1.0
    
h_op = ManyBodyOperator(op_dict)

N = len(states)
H_dense = np.zeros((N, N), dtype=complex)

basis_states = [ManyBodyState({sd: 1.0}) for sd in states]
H_basis_states = h_op.apply_multi(basis_states)

for j in range(N):
    for i, sd_i in enumerate(states):
        H_dense[i, j] = H_basis_states[j].get(sd_i, 0.0)

print(np.sum(np.abs(H_dense)))
