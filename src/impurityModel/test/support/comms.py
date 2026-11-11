"""Communicator parametrization shared by tests that run both undistributed and distributed.

A test that used to exist twice -- a ``comm=None`` copy and a ``COMM_WORLD`` copy differing in
one keyword -- is one test parametrized over :data:`COMMS`. The ``COMM_WORLD`` case carries the
``mpi`` marker, so it runs only under ``--with-mpi`` and is genuinely distributed at ``-n 2``/``-n 3``.
"""

import pytest
from mpi4py import MPI

#: ``comm=None`` is a purely local computation, so at ``-n 2``/``-n 3`` every rank would run the identical case
#: (all ranks at once, sharing the cores) -- it is covered at one rank, where it is skipped nowhere. The mpi case
#: runs at every rank count.
COMMS = [
    pytest.param(
        None,
        id="serial",
        marks=pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="comm=None: serial, covered at one rank"),
    ),
    pytest.param(MPI.COMM_WORLD, id="mpi", marks=pytest.mark.mpi),
]
