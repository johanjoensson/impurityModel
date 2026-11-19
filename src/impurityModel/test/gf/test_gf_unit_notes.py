"""The root-side unit notes of the Lanczos Green's-function path.

Every work unit reaches a traced rank 0 as one ``gf_unit_basis`` note, including the units other colours ran
(the kernel's own ``gf_unit_memory`` note is emitted per colour, which is why the basis-size harness does not
read it).  Until 2026-11 this file was shared with the per-point BiCGSTAB driver, which is gone.
"""

from collections import Counter

import pytest
from mpi4py import MPI

from impurityModel.ed import solver_trace
from impurityModel.test.support.gf_oracles import _run_driver


def _unit_notes(comm):
    with solver_trace.tracing() as trace:
        _run_driver(True, comm=comm)
    return trace.of_kind("gf_unit_basis")


def test_every_unit_reaches_root_as_one_note():
    notes = _unit_notes(None)
    assert notes, "no gf_unit_basis note recorded"
    assert {n["method"] for n in notes} == {"lanczos"}
    assert {n["side"] for n in notes} == {0, 1}


@pytest.mark.mpi
def test_root_sees_every_color_units():
    """Distributed, a traced rank 0 must still see every unit -- the same multiset of (block, side)
    notes a serial run records -- not only the units of its own colour."""
    comm = MPI.COMM_WORLD
    serial = Counter((n["block"], n["side"]) for n in _unit_notes(MPI.COMM_SELF))
    distributed = _unit_notes(comm)
    if comm.rank == 0:
        assert Counter((n["block"], n["side"]) for n in distributed) == serial
    else:
        assert distributed == []
