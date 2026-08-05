"""The winning sector must be rebuilt, not resumed from the walk's collapsed basis.

``find_ground_state_basis`` ends by clearing ``energy_cache`` and re-calling ``calc_energy`` at
the winning occupation -- the comment there calls it a rebuild. It was not one: ``sector_cache``
was left populated, so that call was a cache **hit** returning

* the walk's basis, already collapsed to the eigenvector support by ``calc_energy``'s tail, and
* the stale ``solver.psi_refs`` built on the *pre-collapse* space, whose out-of-basis amplitudes
  ``build_distributed_vector`` then drops without a word.

Worse, whether the hit happened depended on how many *distinct* trial occupations the walk
visited in between, because ``SectorCache`` is an LRU of size 3 (``groundstate.py:50-86``):
resident -> collapsed basis, evicted -> fresh one. Same physical sector, two different starting
spaces, chosen by walk order.

The observable consequence (which basis the final expansion starts from) is not reliably visible
in the returned size -- on a small model the expansion regrows the collapsed support to the same
space. So the test is on the mechanism: the final rebuild must be a cache **miss**.
"""

import numpy as np
import pytest

from impurityModel.ed import groundstate
from impurityModel.ed.groundstate import find_ground_state_basis
from impurityModel.ed.lie_algebra import tensors_to_operator

N_ORB = 12


def _model():
    """A 2-group impurity with one valence and one conduction bath pair per group."""
    h = np.zeros((N_ORB, N_ORB), dtype=complex)
    impurity_orbitals = {0: [[0, 1]], 1: [[2, 3]]}
    valence = {0: [[4, 5]], 1: [[6, 7]]}
    conduction = {0: [[8, 9]], 1: [[10, 11]]}
    for eps, imp in ((-0.6, (0, 1)), (0.4, (2, 3))):
        for orb in imp:
            h[orb, orb] = eps
    for orb in (4, 5, 6, 7):
        h[orb, orb] = -2.0
    for orb in (8, 9, 10, 11):
        h[orb, orb] = 2.0
    for imp, val, con in ((0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11)):
        for other in (val, con):
            h[imp, other] = h[other, imp] = 0.35
    return tensors_to_operator(h), impurity_orbitals, (valence, conduction)


def _run(monkeypatch, *, max_size=64):
    """One ground-state search; returns (basis, [hit?, hit?, ...] per SectorCache.get)."""
    original = groundstate.SectorCache
    forced_size = max_size
    lookups = []

    class _Recording(original):
        # Ignore the caller's size on purpose: find_ground_state_basis constructs the cache with
        # an explicit max_size=3, and on this fixture the walk visits five distinct splits, so
        # the winner would be evicted before the final rebuild and the LRU -- not the fix --
        # would be what makes that a miss. Forcing the size is what puts the winner on the
        # resident side, which is the case the fix exists for.
        def __init__(self, max_size=None):
            super().__init__(max_size=forced_size)

        def get(self, key):
            value = super().get(key)
            lookups.append(value is not None)
            return value

    monkeypatch.setattr(groundstate, "SectorCache", _Recording)

    h_op, impurity_orbitals, bath_states = _model()
    basis = find_ground_state_basis(
        h_op,
        impurity_orbitals,
        bath_states,
        {0: 1, 1: 1},
        tau=1e-3,
        dense_cutoff=1000,
        comm=None,
        verbose=False,
        truncation_threshold=np.inf,
        slaterWeightMin=1e-12,
    )
    return basis, lookups


def test_the_final_rebuild_is_a_miss(monkeypatch):
    """The rebuild at the winning occupation must build a fresh seed basis.

    A generous cache (64 entries, so nothing is ever evicted) is the worst case: before the fix
    the winning sector was always still resident and the "rebuild" always resolved to its
    collapsed basis plus stale ``psi_refs``.
    """
    _basis, lookups = _run(monkeypatch, max_size=64)

    assert lookups, "the walk never consulted the sector cache -- fixture no longer exercises it"
    assert not lookups[-1], (
        "the final calc_energy at the winning occupation hit the sector cache; it must be a "
        "rebuild, since the cached basis is the walk's collapsed eigenvector support and its "
        "solver carries psi_refs from the pre-collapse space"
    )


@pytest.mark.parametrize("max_size", [1, 3, 64])
def test_the_result_does_not_depend_on_the_cache_size(max_size, monkeypatch):
    """The cache is a dedupe for the walk; it must not decide what the search returns."""
    reference, _ = _run(monkeypatch, max_size=1)
    result, _ = _run(monkeypatch, max_size=max_size)

    assert result.ground_state_occupation == reference.ground_state_occupation
    assert result.size == reference.size, (
        f"sector cache size {max_size} returned a basis of {result.size} determinants where "
        f"size 1 returned {reference.size}: the returned basis depends on the cache, not on the "
        "physics."
    )
