"""The bath layout ``ImpurityModel.from_shells`` builds.

These are property tests rather than golden values: the valence and conduction sets must
partition the bath block. That is the invariant a real bug violated for years -- the
conduction states were indexed from the start of the bath block instead of from after the
valence states, so the two sets overlapped and the last ``n_val`` orbitals belonged to
neither. It stayed invisible because every shipped workload but one has ``n_bath == n_val``,
which makes the conduction list empty and the mistake unobservable.
"""

from collections import OrderedDict

import pytest

from impurityModel.ed.model import ImpurityModel


def _layout(shells, val_shells, h0_filename):
    model = ImpurityModel.from_shells(
        h0_filename,
        OrderedDict(shells),
        OrderedDict(val_shells),
        # dc_MLFT insists on a full 2p core and an integer 3d filling.
        OrderedDict((l, 6 if l == 1 else 8) for l in shells),
        ((0,) * 5, (0,) * 3, (0,) * 3, (0,) * 4),
        (0.0, 0.0),
        0.0,
        h_field=(0.0, 0.0, 0.0),
        rank=0,
        verbose=False,
    )
    valence, conduction = model.bath_states
    flat = lambda d, l: sorted(i for block in d[l] for i in block)  # noqa: E731
    return model, valence, conduction, flat


@pytest.fixture
def nio_pickle():
    from pathlib import Path

    path = Path(__file__).resolve().parents[4] / "h0" / "h0_NiO_50p10bath.pickle"
    if not path.is_file():
        pytest.skip("needs the shipped NiO Hamiltonian")
    return str(path)


# Splits of the 60 bath states this Hamiltonian has. (60, 60) is the degenerate case that
# hid the bug: no conduction states, so an overlapping conduction range is unobservable.
@pytest.mark.parametrize("n_bath,n_val", [(60, 50), (60, 10), (60, 30), (60, 60), (60, 0)])
def test_valence_and_conduction_partition_the_bath_block(nio_pickle, n_bath, n_val):
    """Disjoint, and together exactly the bath block. Neither held before the fix.

    With n_bath = 60 and n_val = 50 the conduction states used to come back as indices 16-25
    -- the *valence* range -- while 66-75 were in neither set.
    """
    shells = OrderedDict([(1, 0), (2, n_bath)])
    model, valence, conduction, flat = _layout(shells, OrderedDict([(1, 0), (2, n_val)]), nio_pickle)

    v, c = flat(valence, 2), flat(conduction, 2)
    assert len(v) == n_val
    assert len(c) == n_bath - n_val
    assert not set(v) & set(c), "valence and conduction must not overlap"

    n_impurity = sum(len(block) for blocks in model.impurity_orbitals.values() for block in blocks)
    assert sorted(v + c) == list(range(n_impurity, n_impurity + n_bath)), "the bath block must be covered exactly"


def test_the_conduction_block_follows_the_valence_block(nio_pickle):
    """The bath is laid out valence-first, so the conduction states start after them."""
    shells = OrderedDict([(1, 0), (2, 60)])
    _model, valence, conduction, flat = _layout(shells, OrderedDict([(1, 0), (2, 50)]), nio_pickle)
    v, c = flat(valence, 2), flat(conduction, 2)
    assert max(v) + 1 == min(c)


def test_no_bath_orbital_is_orphaned_across_several_shells(nio_pickle):
    """Two shells with baths: each shell's block must still partition, and not collide."""
    shells = OrderedDict([(1, 4), (2, 60)])
    model, valence, conduction, flat = _layout(shells, OrderedDict([(1, 2), (2, 50)]), nio_pickle)

    seen = []
    for l in shells:
        seen += flat(valence, l) + flat(conduction, l)
    assert len(seen) == len(set(seen)), "no orbital may belong to two shells"
    n_impurity = sum(len(block) for blocks in model.impurity_orbitals.values() for block in blocks)
    assert sorted(seen) == list(range(n_impurity, n_impurity + sum(shells.values())))
