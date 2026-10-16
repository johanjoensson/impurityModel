"""Reference (oracle) single-particle density matrices.

The pre-redesign ``build_density_matrices``, moved here verbatim: one ``c_k|psi>`` per orbital,
fused ``redistribute_psis`` and ``inner_multi``. Correct but it densifies the union support
(OOM on large states), so it serves only as the oracle the new kernel is tested against.
"""

import numpy as np
from mpi4py import MPI

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, inner_multi


def reference_density_matrices(basis, psis, orbital_indices_left=None, orbital_indices_right=None):
    r"""Compute single-particle density matrices for a list of many-body states.

    ``rho[n, i, j] = <psi_n| c_{orb_j}^dagger c_{orb_i} |psi_n>``

    For the square case (``orbital_indices_left == orbital_indices_right``) the
    identity ``rho[i, j] = <phi_j | phi_i>`` (where ``|phi_k> = c_{orb_k}|psi>``)
    is exploited, cutting operator applications from O(n^2) to O(n) and
    halving inner products via Hermitian symmetry.

    For the general rectangular case we also exploit this decomposition
    ``rho[i, j] = <chi_j | phi_i>`` (where ``|chi_k> = c_{orb_k}|psi>`` and
    ``|phi_k> = c_{orb_k}|psi>``), reducing operator applications to O(n).

    ``psis`` may also be a ``ManyBodyState`` (Phase 6a of the state-unification
    refactor); it is unpacked to a list once on entry and the body below is otherwise
    unchanged. Rewriting the per-state loop as one block-apply per orbital was
    considered and rejected, re-evaluated after ``ManyBodyState.select``/``column``
    got a direct O(rows) gather (no longer routed through a selection-matrix matvec) and
    still rejected: reading each orbital's per-state value back out no longer needs a
    full ``width x width`` Gram matrix (cheap columns make a per-state diagonal
    extraction possible), but that only matches this loop's existing per-state
    inner-product cost -- it doesn't beat it. The apply side is the one place
    ``apply_block`` could still win (near-flat cost in block width vs one term/sign/
    restriction/hash pass per state), and that gain is genuinely small here: every
    operator applied is a *trivial* single-term annihilator, so there is little shared
    per-determinant work left to amortize across states in the first place. See
    doc/plans/manybodystate_block_unification.md, Phase 6a, item 3b.
    """
    if isinstance(psis, ManyBodyState):
        psis = psis.to_states()
    if orbital_indices_left is None:
        orbital_indices_left = list(range(basis.num_spin_orbitals))
    if orbital_indices_right is None:
        orbital_indices_right = list(range(basis.num_spin_orbitals))
    n_left, n_right = len(orbital_indices_left), len(orbital_indices_right)
    rhos = np.zeros((len(psis), n_left, n_right), dtype=complex)

    square = orbital_indices_left == orbital_indices_right

    # Built once, not once per state: each annihilator is a pure function of orb, not
    # of psi_n, so constructing it inside the loop below paid len(psis)x redundant
    # ManyBodyOperator construction for no reason.
    left_ops = [ManyBodyOperator({((orb, "a"),): 1.0}) for orb in orbital_indices_left]
    right_ops = left_ops if square else [ManyBodyOperator({((orb, "a"),): 1.0}) for orb in orbital_indices_right]

    for n, psi_n in enumerate(psis):
        phi = [op(psi_n, 0) for op in left_ops]
        chi = phi if square else [op(psi_n, 0) for op in right_ops]

        if basis.is_distributed:
            phi = basis.redistribute_psis(*phi)
            chi = phi if square else basis.redistribute_psis(*chi)

        rhos[n] = inner_multi(chi, phi).T

    if basis.is_distributed:
        basis.comm.Allreduce(MPI.IN_PLACE, rhos, op=MPI.SUM)

    return rhos
