"""``None`` must mean one thing across the restriction helpers (ledger row C4).

Three helpers disagree about ``None``:

* ``gf_units._union_restrictions`` returns ``None`` to mean *unrestricted* (no key is common to
  every stacked state, so nothing can be enforced for the group);
* ``Basis.clone(restrictions=None)`` reads ``None`` as *inherit the parent's window* -- and every
  GF kernel clones the ground-state basis, whose window is the (unwidened) ground-state one;
* ``greens_function._intersect_restrictions`` treats ``None`` as empty and never checks that the
  intersected bounds are still feasible (``lo > hi``).

Composed, a unit whose union window is "unrestricted" runs under the ground-state window instead,
and an infeasible intersection silently produces an excited basis with no determinants.
"""

import pytest
from mpi4py import MPI

from impurityModel.ed.gf_units import _union_restrictions
from impurityModel.ed.greens_function import _intersect_restrictions
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
    assert _union_restrictions([{frozenset({2}): (1, 1)}, {frozenset({3}): (1, 1)}]) is None


@pytest.mark.xfail(strict=True, reason=C4)
def test_an_unrestricted_unit_window_is_not_replaced_by_the_ground_state_window():
    """What _block_green_group does with that None: clone(restrictions=None) inherits."""
    union = _union_restrictions([{frozenset({2}): (1, 1)}, {frozenset({3}): (1, 1)}])
    excited = _gs_basis().clone(initial_basis=[_det((0, 1, 2), 4)], restrictions=union)
    assert not excited.restrictions, f"unit meant to be unrestricted runs under {excited.restrictions}"


@pytest.mark.xfail(strict=True, reason=C4)
def test_an_infeasible_intersection_is_rejected():
    """(0,1) AND (2,3) on the same orbitals admits nothing; the helper must say so."""
    key = frozenset({0, 1})
    with pytest.raises(ValueError):
        _intersect_restrictions({key: (0, 1)}, {key: (2, 3)})
