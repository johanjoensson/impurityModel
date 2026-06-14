import numpy as np
import itertools
from impurityModel.ed.ManyBodyUtils import SlaterDeterminant

n_sites = 8
n_particles = 4

combinations = list(itertools.combinations(range(n_sites), n_particles))
states = [SlaterDeterminant(c) for c in combinations]

print("State 0 elements:", [i for i in combinations[0]])
sd = states[0]
print("SD len:", len(sd))
print("SD [0]:", bin(sd[0]))
