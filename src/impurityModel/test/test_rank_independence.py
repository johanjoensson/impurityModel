"""Serial and MPI must agree on the ground-state basis and on the Green's function.

`de08d77` made the adaptively-selected CIPSI basis independent of `comm.size`, but nothing in the
suite pins that end to end, so it can rot silently: the ground-state *energy* is reproducible even
when the selected determinants are not, which is exactly why the original bug survived so long.

Each test runs the production path twice -- once on the real communicator, once on `MPI.COMM_SELF`
-- and compares. Every rank recomputes the serial reference on its own `COMM_SELF`, which is
redundant but symmetric: no collective can be reached by a subset of ranks, so a mistake here fails
rather than deadlocks. The reference is computed *in the test*, not hard-coded, because a frozen
constant rots the moment the workload is retuned.

Tolerances. The determinant set must be **bit-identical** -- it is a discrete object, and any
difference is structural. The Green's function must not be: MPI reduction order alone perturbs it.
Measured on this workload, serial vs 2 and 3 ranks:

    gs_energies    rel 9.1e-16 / 1.7e-15
    gs_realaxis    rel 1.9e-12 / 2.8e-12
    sigma_real     rel 2.1e-12 / 3.0e-12

so `_GF_RTOL = 1e-10` leaves ~30x margin over roundoff while still catching the class of bug this
file exists for: before `de08d77` the same comparison read 5.3e-9 at 2 ranks and 1.6e-7 at 3.
"""

import numpy as np
import pytest

try:
    from mpi4py import MPI

    _has_mpi = True
except ImportError:  # pragma: no cover - mpi4py is a hard dependency in practice
    _has_mpi = False

from impurityModel.ed.selfenergy import calc_selfenergy
from impurityModel.test._nio_workload import (
    as_calc_selfenergy_args,
    build_ground_state_workload,
    build_selfenergy_inputs,
)

_MASK = (1 << 64) - 1
_GF_RTOL = 1e-10
_E_RTOL = 1e-12

_NBATHS = 10
_DE2_MIN = 1e-6
_DENSE_CUTOFF = 50
_TRUNCATION = 1000
_N_OMEGA = 32


def _basis_fingerprint(basis, comm):
    """Order-independent fingerprint ``(count, sum, xor)`` of the *global* determinant set.

    Determinants are hash-routed with one owner per rank, so the local sets are disjoint and a
    SUM/XOR reduction reconstructs the global set exactly, independently of how it was partitioned.
    Both a sum and an xor, because either alone can collide.
    """
    total = 0
    parity = 0
    for det in basis.local_basis:
        h = det.get_hash() & _MASK
        total = (total + h) & _MASK
        parity ^= h
    count = len(basis.local_basis)
    if comm is not None and comm.size > 1:
        total = comm.allreduce(total, op=MPI.SUM) & _MASK
        parity = comm.allreduce(parity, op=MPI.BXOR)
        count = comm.allreduce(count, op=MPI.SUM)
    return count, total, parity


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}"
    scale = max(float(np.max(np.abs(a))), 1e-300)
    return float(np.max(np.abs(a - b))) / scale


@pytest.mark.mpi
@pytest.mark.skipif(not _has_mpi, reason="mpi4py not available")
def test_cipsi_ground_state_basis_is_rank_independent():
    """The selected determinant set is a discrete object: it must be bit-identical, not close."""
    world = MPI.COMM_WORLD

    distributed = build_ground_state_workload(
        nBaths=_NBATHS, de2_min=_DE2_MIN, dense_cutoff=_DENSE_CUTOFF, comm=world, verbose=False
    )
    serial = build_ground_state_workload(
        nBaths=_NBATHS, de2_min=_DE2_MIN, dense_cutoff=_DENSE_CUTOFF, comm=MPI.COMM_SELF, verbose=False
    )

    got = _basis_fingerprint(distributed["basis"], world)
    ref = _basis_fingerprint(serial["basis"], MPI.COMM_SELF)

    assert got == ref, (
        f"the CIPSI ground-state basis depends on the rank count: {world.size} ranks selected "
        f"(count, sum, xor) = {got}, serial selected {ref}"
    )


@pytest.mark.mpi
@pytest.mark.skipif(not _has_mpi, reason="mpi4py not available")
def test_greens_function_and_selfenergy_are_rank_independent():
    """Transitively pins the *excited* basis too.

    `_block_green_group` seeds it from the whole thermal manifold, which is rotation invariant.
    Seeded instead from a single eigenvector of a degenerate manifold -- as a benchmark harness
    once did -- it differed by one determinant across rank counts and `G` moved by 1.6e-7, three
    orders above the tolerance here.
    """
    world = MPI.COMM_WORLD

    kwargs = build_selfenergy_inputs(
        nBaths=_NBATHS, n_omega=_N_OMEGA, truncation_threshold=_TRUNCATION, rank=world.rank, verbose=False
    )
    distributed = calc_selfenergy(**as_calc_selfenergy_args(kwargs), comm=world)

    kwargs_serial = build_selfenergy_inputs(
        nBaths=_NBATHS, n_omega=_N_OMEGA, truncation_threshold=_TRUNCATION, rank=0, verbose=False
    )
    serial = calc_selfenergy(**as_calc_selfenergy_args(kwargs_serial), comm=MPI.COMM_SELF)

    if world.rank != 0:
        return  # calc_selfenergy returns the result on the root of the passed communicator

    assert _rel(serial["gs_energies"], distributed["gs_energies"]) < _E_RTOL

    for key in ("gs_realaxis", "sigma_real", "sigma_static"):
        ref, got = serial.get(key), distributed.get(key)
        if ref is None or np.asarray(ref).shape == ():
            continue
        err = _rel(ref, got)
        assert err < _GF_RTOL, f"{key} depends on the rank count: rel {err:.3e} at {world.size} ranks"


@pytest.mark.mpi
@pytest.mark.skipif(not _has_mpi, reason="mpi4py not available")
def test_warm_started_eigensolver_delivers_more_than_its_start_block():
    """A warm start must not cap the number of eigenstates at its own block width.

    Before the deflation fix, `get_eigenvectors` warm-started from `psi_refs` produced a first
    Lanczos block whose residual `beta_0` *is* the eigenpair residual; `_cholesky_or_deflate`'s
    absolute rank floor deflated it to rank 0, the sweep declared an invariant subspace, and TRLM
    returned exactly `len(psi_refs)` pairs whatever `num_wanted` asked for. Measured on the 50-bath
    workload: 4 states returned for `num_wanted` = 1 *and* 10, against 20 from a cold start.

    That capped the thermal manifold at the warm start's width, and a partially-kept degenerate
    manifold has no rotation-invariant basis -- which is precisely how the Green's function became
    rank dependent.
    """
    world = MPI.COMM_WORLD
    workload = build_ground_state_workload(
        nBaths=_NBATHS, de2_min=_DE2_MIN, dense_cutoff=_DENSE_CUTOFF, comm=world, verbose=False
    )
    solver = workload["solver"]
    assert solver.psi_refs is not None, "the workload no longer leaves a warm start; the test is moot"
    width = len(solver.psi_refs)

    e_ref, _ = solver.get_eigenvectors(
        workload["h"], num_wanted=10, max_energy=None, dense_cutoff=_DENSE_CUTOFF, slaterWeightMin=1e-12
    )

    assert len(e_ref) > width, (
        f"warm-started eigensolver returned {len(e_ref)} states from a width-{width} start block; "
        "the sweep is deflating the warm start away instead of refining it"
    )
