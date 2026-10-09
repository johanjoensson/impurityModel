"""``MATVEC_PRUNE``: a sparse matvec whose output is redistributed cuts ``slaterWeightMin`` on each
row's summed amplitude (``apply_and_redistribute``, ``block_apply``), not on the per-rank partials.

The oracle is a serial ``apply_block(v, cutoff)`` over the whole block, which sees every row's
whole sum. The multi-rank tests split the input rows over the ranks by position, not by owner, so
determinants reached from rows on different ranks arrive as partial sums -- the case the knob is
about. ``test_gf_apply_row_chunking.py`` covers the block-Lanczos step itself.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.BlockLanczosCore import apply_and_redistribute, block_apply, matvec_cut_after_sum
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant

_IMP = {0: [[0, 1]]}
_BATHS = ({0: [[2, 3]]}, {0: [[4, 5]]})

# Cutoffs at which this model's H v has rows whose per-rank partials fall below the cutoff while
# their sums do not, at 2, 3 and 4 ranks and in both forms (before_sum then loses them; checked below).
_CUTOFFS = (0.4, 0.7)


@pytest.fixture(autouse=True)
def _knob_unset(monkeypatch):
    monkeypatch.delenv("MATVEC_PRUNE", raising=False)


def _det(occupied):
    """Determinant with the given orbitals occupied (MSB-first: orbital i = bit 7-i)."""
    b = 0
    for i in occupied:
        b |= 1 << (7 - i)
    return SlaterDeterminant.from_bytes(bytes([b]))


def _siam_6():
    """Single-impurity Anderson model, 6 spin-orbitals (0,1 imp; 2,3 val; 4,5 cond)."""
    terms = {}
    for o, e in ((0, -1.0), (1, -1.0), (2, -3.0), (3, -3.0), (4, 3.0), (5, 3.0)):
        terms[((o, "c"), (o, "a"))] = e
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = 4.0
    for a, b in ((0, 2), (1, 3), (0, 4), (1, 5)):
        terms[((a, "c"), (b, "a"))] = 0.5
        terms[((b, "c"), (a, "a"))] = 0.5
    return ManyBodyOperator(terms)


def _states():
    """Two columns over the N=3 sector, unequal amplitudes so the cut lands between rows."""
    rng = np.random.default_rng(7)
    dets = [_det(o) for o in ((0, 2, 3), (1, 2, 3), (0, 1, 2), (0, 1, 3), (0, 2, 4), (1, 3, 5), (2, 3, 4), (2, 3, 5))]
    return [ManyBodyState({d: complex(a) for d, a in zip(dets, rng.uniform(0.1, 1.0, len(dets)))}) for _ in range(2)]


def _rows(block):
    """{determinant: (amplitude per column)} of a shared-support block."""
    cols = [{k: complex(a[0]) for k, a in c.items()} for c in block.to_states()]
    keys = sorted({k for c in cols for k in c})
    return {k: tuple(c.get(k, 0j) for c in cols) for k in keys}


def _gathered(rows, comm):
    out = {}
    for part in comm.allgather(rows):
        out.update(part)
    return out


def _my_share(states, comm):
    """This rank's slice of every column's rows, split by position (not by owner)."""
    keys = sorted({k for s in states for k in s.keys()})
    mine = keys[comm.rank :: comm.size]
    return [ManyBodyState({k: a for k, a in s.items() if k in mine}) for s in states]


def _assert_rows_match(got, want):
    assert set(got) == set(want)
    for k in want:
        np.testing.assert_allclose(got[k], want[k], rtol=1e-12, atol=1e-14)


def test_the_cut_is_after_the_sum_only_when_something_is_summed(monkeypatch):
    assert matvec_cut_after_sum(True, 0.1)
    assert not matvec_cut_after_sum(False, 0.1)  # no redistribute: the apply sees whole sums
    assert not matvec_cut_after_sum(True, 0.0)  # nothing to cut
    monkeypatch.setenv("MATVEC_PRUNE", "before_sum")
    assert not matvec_cut_after_sum(True, 0.1)


def test_matvec_prune_rejects_an_unknown_value(monkeypatch):
    monkeypatch.setenv("MATVEC_PRUNE", "sometimes")
    with pytest.raises(ValueError, match="MATVEC_PRUNE"):
        matvec_cut_after_sum(True, 0.1)


def test_no_redistribute_is_the_plain_apply():
    H, block = _siam_6(), ManyBodyState.from_states(_states())
    _assert_rows_match(_rows(apply_and_redistribute(H, block, None, 0.3, False)), _rows(H.apply_block(block, 0.3)))


def _distributed(comm):
    keys = sorted({k for s in _states() for k in s.keys()})
    return Basis(_IMP, _BATHS, initial_basis=keys, comm=comm, verbose=False)


def _block_reference(cutoff):
    """One serial apply over the whole block: a row survives if any column's whole sum does."""
    return _rows(_siam_6().apply_block(ManyBodyState.from_states(_states()), cutoff))


def _list_reference(cutoff):
    """The state-list form cuts each state on its own (apply_multi's per-state cut)."""
    return _rows(
        ManyBodyState.from_states(
            [_siam_6().apply_block(ManyBodyState.from_states([st]), cutoff).to_states()[0] for st in _states()]
        )
    )


def _block_result(cutoff, comm):
    out = apply_and_redistribute(
        _siam_6(), ManyBodyState.from_states(_my_share(_states(), comm)), _distributed(comm), cutoff, True
    )
    return _gathered(_rows(out), comm)


def _list_result(cutoff, comm):
    out = block_apply(_siam_6(), _my_share(_states(), comm), _distributed(comm), True, cutoff)
    return _gathered(_rows(ManyBodyState.from_states(list(out))), comm)


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="needs input rows on more than one rank")
@pytest.mark.parametrize("cutoff", _CUTOFFS)
@pytest.mark.parametrize(
    "result, reference",
    [(_block_result, _block_reference), (_list_result, _list_reference)],
    ids=["block", "state_list"],
)
def test_a_redistributed_matvec_cuts_the_summed_row(result, reference, cutoff):
    _assert_rows_match(result(cutoff, MPI.COMM_WORLD), reference(cutoff))


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="needs input rows on more than one rank")
@pytest.mark.parametrize("cutoff", _CUTOFFS)
@pytest.mark.parametrize(
    "result, reference",
    [(_block_result, _block_reference), (_list_result, _list_reference)],
    ids=["block", "state_list"],
)
def test_before_sum_cuts_each_ranks_partial(result, reference, cutoff, monkeypatch):
    """The fallback keeps the old order, and these cases are ones where the order matters -- so the
    after_sum test above discriminates."""
    monkeypatch.setenv("MATVEC_PRUNE", "before_sum")
    with pytest.raises(AssertionError):
        _assert_rows_match(result(cutoff, MPI.COMM_WORLD), reference(cutoff))
