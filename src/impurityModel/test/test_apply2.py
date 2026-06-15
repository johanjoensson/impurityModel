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

state = ManyBodyState()
state[states[0]] = 1.0

mv1 = h_op(state, 1e-10)
print("mv1 len:", len(mv1))
