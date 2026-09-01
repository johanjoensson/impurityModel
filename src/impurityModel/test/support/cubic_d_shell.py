"""A cubic d-shell impurity model with a genuine 3-fold ``S = 1`` ground triplet.

Promoted out of ``test/symmetry/test_ground_state_symmetry.py`` once the Lanczos manifold
tests needed the same model: it is the only fixture in the suite that carries a real
degenerate manifold of a *fully interacting* Hamiltonian on a real ``Basis``, which is what
``CIPSISolver.get_eigenvectors``' Krylov branch needs (it builds its matrix with
``build_sparse_matrix(basis, H)``, so the dense/scipy-sparse fixtures in
``lanczos_fixtures.py`` cannot reach it).

An ``eg``/``t2g`` crystal field on 5 spatial orbitals in the down-then-up spin layout, with a
genuine Slater-Condon interaction and a hybridising bath. Its continuous symmetries are total
charge and total spin: the octahedral point group is discrete, so the orbital part has no Lie
generators at all, and everything the one-body commutant offers beyond spin is broken by ``U``.
"""

import numpy as np

from impurityModel.ed import atomic_physics
from impurityModel.ed.lie_algebra import tensors_to_operator
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.model import atomic_u4

N_IMP = 10
SLATER = (9.0, 0.0, 8.0, 0.0, 6.0)
EG_SPATIAL = (0, 1)


def cubic_d_shell(*, n_bath_sets=2, soc=0.0):
    """(h_op, H_full, impurity_orbitals, bath_states) for a cubic d-shell.

    ``n_bath_sets=2`` gives the production layout, a valence *and* a conduction bath -- the case
    the old one-body-commutant path silently returned ``[]`` for.
    """
    n_orb = N_IMP * (1 + n_bath_sets)
    h = np.zeros((n_orb, n_orb), dtype=complex)
    for orb in range(N_IMP):
        # Cubic field splits the SPATIAL orbitals; both spin channels see the same splitting.
        h[orb, orb] = 0.6 if (orb % 5) in EG_SPATIAL else -0.4
        for which in range(n_bath_sets):
            partner = N_IMP * (which + 1) + orb
            h[partner, partner] = -2.0 if which == 0 else 2.0
            h[orb, partner] = h[partner, orb] = 0.3 if which == 0 else 0.2
    if soc:
        # A spin-mixing term: S_+- stops being a symmetry, and the pairing must be rejected.
        for spatial in range(5):
            h[spatial, spatial + 5] += soc
            h[spatial + 5, spatial] += soc
    h_op = tensors_to_operator(h)
    u_op = ManyBodyOperator(atomic_physics.getUop_from_rspt_u4(atomic_u4(2, SLATER)))
    impurity_orbitals = {0: [list(range(N_IMP))]}
    bath_sets = [{0: [[N_IMP * (w + 1) + o for o in range(N_IMP)]]} for w in range(n_bath_sets)]
    while len(bath_sets) < 2:
        bath_sets.append({0: []})
    return h_op, h_op + u_op, impurity_orbitals, tuple(bath_sets)
