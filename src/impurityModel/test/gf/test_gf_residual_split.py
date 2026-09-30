"""The residual split of a projected resolvent solve, and the ``G`` error bound built on it.

``gf_primitives.residual_split`` recovers, from one cutoff-0 matvec, the part of the true residual
``s - (z-H) X`` inside the solve basis ``P`` (the solver residual, recomputed) and the part
outside it (the *boundary residual* ``b = (1-P) H X``, which the solver never sees).
``resolvent_error_bound`` turns those into a rigorous elementwise bound on ``|G_exact - G|``.

What is pinned, on the closed SIAM-6 sector where the dense resolvent is exact:

* the split equals the dense ``R = Y - (z-H)X`` cut at the retained keys, serially and
  distributed (the owner-side summation of ``redistribute_block`` only matters multi-rank);
* the bound holds -- actual ``|dG|`` never exceeds it -- at binding caps, where ``b`` is large;
* uncapped at ``slaterWeightMin = 0`` the boundary residual is **exactly** zero: the solver adds
  every row it produces to the basis, so any nonzero ``b`` is a bookkeeping bug, not roundoff;
* a seed that is not real up to one phase falls back to the first-order bound.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.cg import block_bicgstab
from impurityModel.ed.gf_primitives import _CappedBasisProxy, residual_split, resolvent_error_bound
from impurityModel.ed.gf_solvers import block_Green_bicgstab
from impurityModel.ed.greens_function import _gf_signed_axes
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.ed.basis_transcription import build_dense_matrix
from impurityModel.ed.manybody_basis import Basis
from impurityModel.test.support.gf_oracles import (
    _BATHS,
    _IMP,
    DELTA,
    OMEGA,
    _dense_G_on,
    _n3_sector_dets,
    _seed_basis,
    _seeds,
    _siam_6,
)

ATOL = 1e-12


@pytest.fixture(autouse=True)
def _hermetic_knobs(monkeypatch):
    monkeypatch.delenv("GF_BICGSTAB_WARM_HISTORY", raising=False)
    monkeypatch.setenv("GF_BICGSTAB_RESIDUAL_CHECK", "1")


def _key(det):
    """A determinant as its padded byte string: hashable, picklable, and comparable across objects
    (``SlaterDeterminant`` is none of those reliably, and ``from_bytes`` of the padded form is a
    *different* determinant)."""
    return bytes(det.to_bytearray())


def _dense_H_and_index():
    dets = sorted(_n3_sector_dets())
    basis = Basis(_IMP, _BATHS, initial_basis=dets, comm=None, verbose=False)
    return np.asarray(build_dense_matrix(basis, _siam_6())), {_key(d): i for i, d in enumerate(dets)}


def _to_dense(states, index, comm):
    """Dense columns of distributed width-1 states (test-only gather).

    Determinants cross the wire as bytes: ``SlaterDeterminant`` is not picklable, and a pickling
    error on one rank inside a collective leaves the other ranks waiting in it forever."""
    local = [{_key(det): complex(amp[0]) for det, amp in s.items()} for s in states]
    parts = comm.allgather(local) if comm is not None else [local]
    V = np.zeros((len(index), len(states)), dtype=complex)
    for part in parts:
        for j, col in enumerate(part):
            for key, amp in col.items():
                V[index[key], j] += amp
    return V


def _capped_solve(cap, z, comm):
    basis = _seed_basis(comm=comm)
    proxy = _CappedBasisProxy(basis, cap)
    owns = comm is None or comm.rank == 0
    blocks = [ManyBodyState.from_states([s]) for s in _seeds()] if owns else [ManyBodyState(width=1) for _ in _seeds()]
    Y = ManyBodyState.from_states([blk.to_states()[0] for blk in basis.redistribute_psis(*blocks)])
    A = z - _siam_6()
    X = ManyBodyState(width=Y.width)
    info = {}
    for _ in range(10):
        X = block_bicgstab(A, X, Y, proxy, 0.0, atol=ATOL, info=info)
        if info["converged"]:
            break
    return A, X, Y, basis, proxy


@pytest.mark.mpi
@pytest.mark.parametrize("cap", [6, 12, 17])
def test_split_equals_the_dense_residual_cut_at_the_retained_keys(cap):
    comm = MPI.COMM_WORLD if MPI.COMM_WORLD.size > 1 else None
    z = 1.7 + 1j * DELTA
    A, X, Y, basis, proxy = _capped_solve(cap, z, comm)
    r_p, b = residual_split(A, X, Y, basis, proxy.retained_mask, Y.width, comm)

    H, index = _dense_H_and_index()
    Xd, Yd = _to_dense(X.to_states(), index, comm), _to_dense(Y.to_states(), index, comm)
    R = Yd - (z * np.eye(len(index)) - H) @ Xd
    kept = [_key(d) for d in proxy.retained_keys()]
    kept = {k for part in (comm.allgather(kept) if comm is not None else [kept]) for k in part}
    in_p = np.array([key in kept for key in sorted(index)], dtype=bool)
    np.testing.assert_allclose(r_p, np.linalg.norm(R[in_p], axis=0), rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(b, np.linalg.norm(R[~in_p], axis=0), rtol=1e-10, atol=1e-13)
    assert np.all(b > 0), "cap below the closure must leave a boundary residual"


def _points(cap, seeds=None):
    z_axes = _gf_signed_axes(None, OMEGA, 0, DELTA)
    seeds = _seeds() if seeds is None else seeds
    G_axes, stats = block_Green_bicgstab(_siam_6(), seeds, _seed_basis(cap=cap), [0.0], len(seeds), z_axes, atol=ATOL)
    return z_axes[0], G_axes[0][0], stats


@pytest.mark.parametrize("cap", [6, 12, 17, np.inf])
def test_bound_holds_elementwise_at_every_point(cap):
    z_axis, G, stats = _points(cap)
    ref = _dense_G_on(_n3_sector_dets(), z_axis)
    assert len(stats["points"]) == len(z_axis)
    for p in stats["points"]:
        err = np.abs(G[p["k"]] - ref[p["k"]])
        bound = np.asarray(p["dG_bound"])
        assert np.all(err <= bound * (1 + 1e-9) + 1e-14), (p["k"], err.max(), bound.max())
    if np.isfinite(cap):
        # Binding cap: the bound must be informative, not vacuous.
        assert max(np.max(p["dG_bound"]) for p in stats["points"]) > 1e-8


@pytest.mark.parametrize("cap", [np.inf, 10**6])
def test_uncapped_boundary_residual_is_exactly_zero(cap):
    """Both unconstrained routes: the raw basis (``inf``: no proxy at all) and a proxy whose cap
    never binds (the route every production solve takes under the memory guard)."""
    _z, _G, stats = _points(cap)
    assert not stats["cap_hit"]
    for p in stats["points"]:
        assert all(bj == 0.0 for bj in p["boundary"]), p["boundary"]


def test_complex_seed_falls_back_to_the_first_order_bound():
    s0, s1 = _seeds()
    twisted = ManyBodyState({det: amp[0] * (1j if n == 0 else 1.0) for n, (det, amp) in enumerate(s0.items())})
    _z, _G, stats = _points(12, seeds=[twisted, s1])
    kinds = {tuple(p["second_order"]) for p in stats["points"]}
    assert kinds == {(False, True)}, kinds


@pytest.mark.mpi
def test_error_bar_reaches_the_driver_report(monkeypatch):
    """End to end through ``get_Greens_function``: with the check on the report carries the bound
    (a finite, non-negative number), and with it off the diagnostic is absent rather than zero --
    an unmeasured error bar must not read as a perfect one."""
    from impurityModel.test.support.gf_oracles import _run_driver

    comm = MPI.COMM_WORLD if MPI.COMM_WORLD.size > 1 else None

    def bound_entries(report):
        return [d for d in report.diagnostics if d.name == "truncation_error_bound"] if report is not None else []

    _mat, _real, report = _run_driver("bicgstab", "none", comm=comm)
    if comm is None or comm.rank == 0:
        entries = bound_entries(report)
        assert entries and all(np.isfinite(d.value) and d.value >= 0.0 for d in entries)

    monkeypatch.setenv("GF_BICGSTAB_RESIDUAL_CHECK", "0")
    _mat, _real, report = _run_driver("bicgstab", "none", comm=comm)
    assert bound_entries(report) == []
