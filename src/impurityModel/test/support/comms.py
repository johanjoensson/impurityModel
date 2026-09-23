"""Communicator parametrization shared by tests that run both undistributed and distributed.

A test that used to exist twice -- a ``comm=None`` copy and a ``COMM_WORLD`` copy differing in
one keyword -- is one test parametrized over :data:`COMMS`. The ``COMM_WORLD`` case carries the
``mpi`` marker, so it runs only under ``--with-mpi`` and is genuinely distributed at ``-n 2``/``-n 3``.
"""

import pytest
from mpi4py import MPI

COMMS = [pytest.param(None, id="serial"), pytest.param(MPI.COMM_WORLD, id="mpi", marks=pytest.mark.mpi)]
