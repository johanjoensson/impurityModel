"""The degenerate-manifold guarantee, on the branch that actually implements it.

``CIPSISolver.get_eigenvectors`` keeps a thermal manifold whole by never letting the energy cut
fall inside a group of (near-)degenerate eigenvalues: a degenerate manifold has no preferred
basis, so keeping only part of one leaves the CIPSI candidate scores, the thermal average and
the Green's-function seed support at the mercy of whichever rotation the eigensolver returned --
and hence of the MPI rank count. That guarantee is the reason five separate loops in the
ground-state search grow their eigenstate request.

Every existing test of it runs the **dense** branch (``basis.size < dense_cutoff``):
``test_ground_state_symmetry.py`` and ``test_groundstate.py`` both diagonalise exactly, and
``test_eigenstate_cold_retry.py`` replaces the eigensolver with a fake. So the guarantee had
never been exercised on the restarted-Lanczos path that production takes, which is also the only
path where ``_energy_cut_indices`` runs at all -- the dense branch applies a hard
``es - min(es) <= e_max`` mask (``eigensolvers.py``) with no manifold-absorbing step.

Both the expansion and the final solve run at ``dense_cutoff=DENSE_CUTOFF`` here, so the whole
CIPSI walk goes through TRLM. Expanding densely and only solving through Krylov would show that
the Krylov path *preserves* a manifold it was handed, never that it can find one.

Marked ``mpi``: it costs ~17 s, which is a lot to add to a default gate of ~85 s, and the MPI
legs (1, 2 and 3 ranks) are where it earns its keep -- a manifold that is whole serially and
bisected at 3 ranks is exactly the failure this guarantee exists to prevent, and ``-n 1`` covers
the serial case.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.manybody_basis import Basis
from impurityModel.test.support.cubic_d_shell import cubic_d_shell

# Below the 800-determinant basis the expansion reaches, so every solve takes the Krylov branch.
DENSE_CUTOFF = 100
THRESHOLD = 800
SLATER_WEIGHT_MIN = 1e-12

# The cut goes at the geometric centre of the gap between the ground triplet's internal spread
# (measured 1.4e-8 on this capped basis) and the second triplet (1.13e-5 above): a factor ~29
# either side, the widest margin available. Placed below the spread it would bisect the triplet --
# correctly, since at that scale these are distinct eigenvalues of the *truncated* Hamiltonian
# rather than a manifold; placed above 1e-5 it would swallow the second triplet and the test would
# stop being about a boundary at all. The premise assertions below re-check both gaps, so a
# fixture whose spectrum moves fails with that diagnosis instead of a bare count mismatch.
CUT = 4e-7
MANIFOLD = 3


@pytest.fixture(scope="module")
def expanded():
    """The cubic d-shell walked all the way through TRLM, with its warm start left in place.

    Module-scoped: the expansion is ~25 s and both tests want the same converged basis. Every
    rank builds it, collectively, at first use -- the same call the tests would make themselves.
    """
    h_op, h_full, impurity_orbitals, bath_states = cubic_d_shell(n_bath_sets=1)
    basis = Basis(
        impurity_orbitals,
        bath_states,
        nominal_impurity_occ={0: 2},
        mixed_valence={0: 0},
        tau=1e-3,
        truncation_threshold=THRESHOLD,
        comm=MPI.COMM_WORLD,
        verbose=False,
    )
    solver = CIPSISolver(basis)
    solver.expand(h_full, dense_cutoff=DENSE_CUTOFF, de2_min=1e-10, slaterWeightMin=SLATER_WEIGHT_MIN)
    yield solver, h_full


@pytest.mark.mpi
def test_the_krylov_branch_returns_the_whole_ground_manifold_and_stops_at_the_cut(expanded):
    solver, h_full = expanded

    assert len(solver.basis) >= DENSE_CUTOFF, (
        f"the basis came back at {len(solver.basis)} determinants, under dense_cutoff="
        f"{DENSE_CUTOFF}: this test would silently run the dense branch, which is the one case "
        "it exists to avoid"
    )
    assert solver.psi_refs is not None, "expand left no warm start; the block width premise is moot"
    # `get_eigenvectors` appends one cold full-support column to the warm block, so the Lanczos
    # block width is len(psi_refs) + 1. It has to exceed the multiplicity: a block-Krylov space
    # satisfies dim(K(H, V) intersect E_lambda) <= rank(P_lambda V) <= p at any subspace depth,
    # restarts included (see test_no_ghost_bands.py), so a narrower block cannot represent the
    # whole manifold however hard it works.
    width = len(solver.psi_refs) + 1
    assert width > MANIFOLD, f"block width {width} cannot hold a {MANIFOLD}-fold manifold"

    uncut, _ = solver.get_eigenvectors(
        h_full,
        num_wanted=6,
        max_energy=None,
        dense_cutoff=DENSE_CUTOFF,
        slaterWeightMin=SLATER_WEIGHT_MIN,
        psi_refs=solver.psi_refs,
    )
    uncut = np.sort(np.asarray(uncut).real)
    gaps = uncut - uncut[0]
    # Premise, asserted rather than assumed: the fixture still has a MANIFOLD-fold ground group
    # well inside the cut and its next state well outside it.
    assert gaps[MANIFOLD - 1] < CUT / 5, f"the ground manifold no longer fits inside the cut: {gaps[:6]}"
    assert gaps[MANIFOLD] > 5 * CUT, f"the next state is no longer outside the cut: {gaps[:6]}"

    es, psis = solver.get_eigenvectors(
        h_full,
        num_wanted=6,
        max_energy=CUT,
        dense_cutoff=DENSE_CUTOFF,
        slaterWeightMin=SLATER_WEIGHT_MIN,
        psi_refs=solver.psi_refs,
    )

    assert len(es) == len(psis) == MANIFOLD, f"expected the whole {MANIFOLD}-fold manifold, got {len(es)}: {es}"
    assert np.ptp(np.asarray(es).real) < CUT
    # Trimmed, not merely short: a solver that returned only these states would not certify
    # anything, because nothing would lie beyond the cut.
    assert len(uncut) > len(es), "nothing was computed outside the cut, so the manifold is uncertified"


@pytest.mark.mpi
def test_the_krylov_and_dense_branches_agree_on_the_manifold(expanded):
    """A free exact oracle for the test above: the same basis, diagonalised.

    Compared at ``max_energy=None``. The dense branch has no manifold-absorbing step, so at the
    tiny ``CUT`` above it would return however many states happen to sit below a hard threshold
    and the comparison would fail for a reason that has nothing to do with the Krylov path.
    """
    solver, h_full = expanded
    kwargs = dict(num_wanted=6, max_energy=None, slaterWeightMin=SLATER_WEIGHT_MIN, psi_refs=solver.psi_refs)

    krylov, _ = solver.get_eigenvectors(h_full, dense_cutoff=DENSE_CUTOFF, **kwargs)
    dense, _ = solver.get_eigenvectors(h_full, dense_cutoff=10 * len(solver.basis), **kwargs)

    k = np.sort(np.asarray(krylov).real)[:MANIFOLD]
    d = np.sort(np.asarray(dense).real)[:MANIFOLD]
    np.testing.assert_allclose(k, d, atol=1e-9, err_msg=f"Krylov {k} vs dense {d}")
