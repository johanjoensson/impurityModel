"""``None`` must mean one thing across the restriction helpers (ledger row C4).

Three helpers used to disagree about ``None``: the union of windows returned it to mean
*unrestricted*, ``Basis.clone(restrictions=None)`` read it as *inherit the parent's window* (and
every GF kernel clones the ground-state basis, so a unit meant to be unrestricted ran under the
unwidened ground-state window), and the intersection never checked that its bounds were still
feasible, silently producing an excited basis with no determinants. Now ``None`` is unrestricted
everywhere, ``Basis.clone`` inherits only on the explicit ``INHERIT`` default, and an infeasible
intersection raises.
"""

import pytest
from mpi4py import MPI

from impurityModel.ed.basis_restrictions import intersect_windows, union_windows
from impurityModel.ed.manybody_basis import Basis
from impurityModel.test.support.gf_branch_oracle import _det

C4 = "C4: None means unrestricted / inherit / empty depending on the helper (doc/reviews/gf_review.md)"

GS_WINDOW = {frozenset({2, 3}): (2, 2)}  # a ground-state window: the valence pair filled


def _gs_basis():
    basis = Basis({0: [[0, 1]]}, ({0: [[2, 3]]}, {0: [[]]}), initial_basis=[_det((0, 2, 3), 4)], comm=MPI.COMM_SELF)
    basis.restrictions = dict(GS_WINDOW)
    return basis


def test_a_union_over_disjoint_keys_is_unrestricted():
    """The premise: stacking two states whose windows share no key leaves nothing enforceable."""
    assert union_windows([{frozenset({2}): (1, 1)}, {frozenset({3}): (1, 1)}]) is None


def test_an_unrestricted_unit_window_is_not_replaced_by_the_ground_state_window():
    """What _block_green_group does with that None: clone(restrictions=None) inherits."""
    union = union_windows([{frozenset({2}): (1, 1)}, {frozenset({3}): (1, 1)}])
    excited = _gs_basis().clone(initial_basis=[_det((0, 1, 2), 4)], restrictions=union)
    assert not excited.restrictions, f"unit meant to be unrestricted runs under {excited.restrictions}"


def test_an_infeasible_intersection_is_rejected():
    """(0,1) AND (2,3) on the same orbitals admits nothing; the helper must say so."""
    key = frozenset({0, 1})
    with pytest.raises(ValueError):
        intersect_windows({key: (0, 1)}, {key: (2, 3)})


def test_omitting_the_window_still_inherits_it():
    """The default is the explicit INHERIT sentinel: a caller that does not pass a window (the
    spectra seed-moment scratch basis) keeps the parent's, exactly as before."""
    child = _gs_basis().clone(initial_basis=[_det((0, 2, 3), 4)])
    assert child.restrictions == GS_WINDOW


def test_intersecting_with_nothing_is_the_other_window():
    key = frozenset({0, 1})
    assert intersect_windows(None, {key: (0, 1)}) == {key: (0, 1)}
    assert intersect_windows({key: (0, 1)}, None) == {key: (0, 1)}
    assert intersect_windows(None, None) is None
