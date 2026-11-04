"""``merge_measurements``: one member is passed through untouched; several concatenate unshifted energies."""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.gs_report import measure_ground_state, merge_measurements
from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant


def _det(i):
    return SlaterDeterminant((i,))


def _member(keys, es, total):
    psis = [ManyBodyState({k: 1.0}) for k in keys]
    return measure_ground_state(f"basis{total}", np.array(es), psis, total=total)


def test_one_member_is_returned_untouched():
    m = _member([_det(1)], [-1.0], 1)
    basis, es, psis = merge_measurements([m])
    assert basis is m.basis and es is m.es and psis is m.psis


def test_members_concatenate_with_absolute_energies_dominant_first():
    a = _member([_det(1), _det(2)], [-2.0, -1.9], 1)
    b = _member([_det(4)], [-1.95], 2)
    basis, es, psis = merge_measurements([a, b])
    assert basis == "basis1"
    assert list(es) == [-2.0, -1.9, -1.95]
    assert len(psis) == 3


def test_empty_raises():
    with pytest.raises(ValueError):
        merge_measurements([])


class _Basis:
    def __init__(self, comm):
        self.comm = comm


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="needs a rank that holds none of the data (width-0 block)")
def test_width_zero_block_on_a_rank_still_yields_len_es_states():
    """A rank whose redistributed block has width 0 (``to_states() == []``) must not return a short list:
    the merged list has len(es) entries on every rank, so the collectives after it stay in step."""

    comm = MPI.COMM_WORLD
    basis = _Basis(comm)
    # rank 0 holds the data; every other rank holds the width-0 polymorphic zero
    if comm.rank == 0:
        psis_a = ManyBodyState.from_states([ManyBodyState({_det(1): 1.0}), ManyBodyState({_det(2): 1.0})])
    else:
        psis_a = ManyBodyState.from_states([])
    a = measure_ground_state(basis, np.array([-2.0, -1.9]), psis_a, total=1)
    b = measure_ground_state(
        basis, np.array([-1.95]), ManyBodyState.from_states([ManyBodyState({_det(4): 1.0})]), total=2
    )
    _, es, psis = merge_measurements([a, b])
    assert len(psis) == len(es) == 3
    # what report_ground_state does next, on every rank: the padded entries must be width-1 states
    block = ManyBodyState.from_states(list(psis))
    assert int(block.width) == len(es)
