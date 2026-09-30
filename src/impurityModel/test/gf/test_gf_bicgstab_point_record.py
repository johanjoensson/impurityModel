"""The per-point basis record of the per-frequency BiCGSTAB path, and the root-side unit notes.

What these pin, for the basis-size comparison harness (``doc/plans/gf_basis_size_comparison.md``):

* ``block_Green_bicgstab``'s ``stats["points"]`` has one entry per solve, with the seed support
  as the floor under the rebuilt basis and the rebuilt basis as the floor under the solve basis;
* ``GF_BICGSTAB_WARM_HISTORY=0`` really cold-starts: the rebuilt basis is *exactly* the seed
  support, while the default warm start carries its extrapolation's support in on top -- the
  sliding-window union that makes a warm-started "per-point" size overstate the point;
* every work unit of either method reaches a traced rank 0 as one ``gf_unit_basis`` note,
  including the units other colors ran. The Lanczos kernel's own ``gf_unit_memory`` note is
  emitted per color, which is why the harness does not read it.
"""

from collections import Counter

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import solver_trace
from impurityModel.ed.gf_solvers import block_Green_bicgstab
from impurityModel.ed.greens_function import _gf_signed_axes
from impurityModel.test.support.gf_oracles import DELTA, MATSUBARA, OMEGA, _run_driver, _seed_basis, _seeds, _siam_6

ATOL = 1e-10


@pytest.fixture(autouse=True)
def _default_warm_history(monkeypatch):
    # Knobs are read lazily, so an exported value would silently change what these tests measure.
    monkeypatch.delenv("GF_BICGSTAB_WARM_HISTORY", raising=False)


def _solve(z_axes):
    return block_Green_bicgstab(_siam_6(), _seeds(), _seed_basis(), [0.3], len(_seeds()), z_axes, atol=ATOL)


def test_points_record_one_entry_per_solve():
    z_axes = _gf_signed_axes(MATSUBARA, OMEGA, 0, DELTA)
    _G, stats = _solve(z_axes)
    points = stats["points"]
    assert len(points) == stats["n_points"] == sum(len(z) for z in z_axes)
    assert {(p["axis"], p["k"]) for p in points} == {(ax, k) for ax, z in enumerate(z_axes) for k in range(len(z))}
    for p in points:
        assert 0 < p["seed_size"] <= p["rebuild_size"] <= p["solve_size"]
        assert p["converged"]
    assert max(p["solve_size"] for p in points) == stats["max_solve_basis"]
    assert max(p["rebuild_size"] for p in points) == stats["max_rebuild_basis"]


def test_cold_start_rebuilds_on_the_seed_support_alone(monkeypatch):
    """History 0 must rebuild every point on exactly the seed support, and still give the same G."""
    z_axes = _gf_signed_axes(MATSUBARA, OMEGA, 0, DELTA)
    G_warm, warm = _solve(z_axes)
    monkeypatch.setenv("GF_BICGSTAB_WARM_HISTORY", "0")
    G_cold, cold = _solve(z_axes)

    assert all(p["rebuild_size"] == p["seed_size"] for p in cold["points"])
    # The discriminating half: the default warm start does carry support in, or the cold check
    # above would pass on a knob that changed nothing.
    assert any(p["rebuild_size"] > p["seed_size"] for p in warm["points"])

    # Both solves stop at atol relative to the block's largest seed column, so they may differ by
    # up to twice that through the resolvent, whose norm is at most 1/|Im z|.
    seed_norm2 = max(float(np.real(s.norm2())) for s in _seeds())
    for ax, z_axis in enumerate(z_axes):
        bound = 2 * ATOL * seed_norm2 / np.abs(np.imag(z_axis))
        err = np.max(np.abs(G_cold[ax][0] - G_warm[ax][0]), axis=(1, 2))
        assert np.all(err <= bound), (ax, err.max(), bound.min())


def _unit_notes(gf_method, comm):
    with solver_trace.tracing() as trace:
        _run_driver(gf_method, "partial", comm=comm)
    return trace.of_kind("gf_unit_basis")


@pytest.mark.parametrize("gf_method", ["lanczos", "bicgstab"])
def test_every_unit_reaches_root_as_one_note(gf_method):
    notes = _unit_notes(gf_method, None)
    assert notes, "no gf_unit_basis note recorded"
    assert {n["method"] for n in notes} == {gf_method}
    assert {n["side"] for n in notes} == {0, 1}
    if gf_method == "bicgstab":
        for n in notes:
            assert n["retained_size"] == max(p["solve_size"] for p in n["points"])
            assert n["retained_size"] >= max(p["seed_size"] for p in n["points"])


@pytest.mark.mpi
@pytest.mark.parametrize("gf_method", ["lanczos", "bicgstab"])
def test_root_sees_every_color_units(gf_method):
    """Distributed, a traced rank 0 must still see every unit -- the same multiset of (block, side)
    notes a serial run records -- not only the units of its own color."""
    comm = MPI.COMM_WORLD
    serial = Counter((n["block"], n["side"]) for n in _unit_notes(gf_method, MPI.COMM_SELF))
    distributed = _unit_notes(gf_method, comm)
    if comm.rank == 0:
        assert Counter((n["block"], n["side"]) for n in distributed) == serial
    else:
        assert distributed == []
