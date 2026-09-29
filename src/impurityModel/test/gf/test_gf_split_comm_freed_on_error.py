"""A failing GF unit must not leak the split communicator (ledger row M4).

``run_units_distributed`` frees the per-color communicator after its ``try/finally`` block, not
inside it: a kernel that raises -- on every rank, so there is no deadlock -- propagates past the
``free_comm()`` call, and the split communicator is never freed. A calling driver that recovers
from the error (the self-energy retry loop, the double-counting search) then leaks one
communicator per failure.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import memory_estimate as me
from impurityModel.ed.gf_units import run_units_distributed
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant

M4 = "M4: the split communicator is freed outside the finally (doc/reviews/gf_review.md)"


class _KernelFailure(RuntimeError):
    pass


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="the unit split only runs on more than one rank")
def test_the_split_communicator_is_freed_when_a_kernel_raises(monkeypatch):
    comm = MPI.COMM_WORLD
    states = [b"\x80", b"\x40", b"\x20", b"\x10"]
    basis = Basis(
        {0: [[0, 1, 2, 3]]}, ({0: [[]]}, {0: [[]]}), initial_basis=states, comm=comm, truncation_threshold=100
    )
    psi = ManyBodyState.from_states(
        [ManyBodyState({SlaterDeterminant.from_bytes(states[0]): 1.0} if comm.rank == 0 else {}, width=1)]
    )
    monkeypatch.setattr(me, "available_bytes_per_rank", lambda c: 2**60)  # two units -> two colors

    freed = []
    original = Basis.free_comm

    def recording_free(self):
        freed.append(self.comm)
        original(self)

    monkeypatch.setattr(Basis, "free_comm", recording_free)

    def kernel(split_basis, u, seeds):
        assert split_basis.comm is not comm, "premise: the units ran on a split communicator"
        raise _KernelFailure("unit failed on every rank")

    with pytest.raises(_KernelFailure):
        run_units_distributed(basis, [[psi], [psi]], np.array([1.0, 1.0]), kernel)
    assert freed, "the split communicator was not freed on the error path"
