"""Importance-ranked per-point basis admission (``GF_BICGSTAB_ADMISSION=outer``).

The contract under test (``gf_admission``): a point is solved by an outer loop of *frozen-basis*
solves -- measure the residual outside ``P``, admit what scores above the threshold, re-solve -- so
the answer is always the exact resolvent of ``P H P`` on the final retained set, whatever that set
is. Everything is checked against the dense resolvent on the closed SIAM-6 sector:

* the strong oracle: ``G`` equals the dense ``P H P`` resolvent on the retained keys, for every
  threshold and cap -- this is what a skipped final solve, or a solve on the wrong basis, breaks;
* the endpoints: threshold 0 admits everything nonzero and so reaches the full connectivity
  closure (the depth ceiling the retired per-frequency CIPSI hit cannot bite), a huge threshold
  admits nothing beyond the start set;
* the start set always holds the seeds and their first H-shell (the ``Sigma`` tail's moments);
* the cap is never exceeded and reports ``budget``;
* ``all`` -- the default -- is bit-identical to the unset knob.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.gf_admission import solve_point_outer
from impurityModel.ed.gf_solvers import block_Green_bicgstab, solve_shifted_block
from impurityModel.ed.greens_function import _gf_signed_axes
from impurityModel.ed.ManyBodyUtils import ManyBodyState, block_inner_cy
from impurityModel.test.support.gf_oracles import (
    DELTA,
    OMEGA,
    _dense_G_on,
    _n3_sector_dets,
    _seed_basis,
    _seeds,
    _siam_6,
)

ATOL = 1e-12
Z = 1.7 + 1j * DELTA
CLOSURE = 18  # two 9-determinant sectors: what the two seed columns reach


@pytest.fixture(autouse=True)
def _hermetic_knobs(monkeypatch):
    for name in (
        "GF_BICGSTAB_ADMISSION",
        "GF_BICGSTAB_ADMIT_SCORER",
        "GF_BICGSTAB_ADMIT_TOL_AMP",
        "GF_BICGSTAB_ADMIT_TOL_JACOBI",
        "GF_BICGSTAB_ADMIT_ROUNDS",
        "GF_BICGSTAB_ADMIT_SHELLS",
        "GF_BICGSTAB_ADMIT_CARRY_TOL",
        "GF_BICGSTAB_ADMIT_EN_TOL",
        "GF_ADMIT_FIRST_SHELL_TOL",
        "GF_BICGSTAB_WARM_HISTORY",
    ):
        monkeypatch.delenv(name, raising=False)


def _outer_point(monkeypatch, cap=np.inf, eta=1e-4, scorer="amplitude", comm=None, z=Z, **knobs):
    """One ``solve_point_outer`` on SIAM-6: ``(G, record, proxy, basis, A_op, seeds)``."""
    monkeypatch.setenv("GF_BICGSTAB_ADMIT_SCORER", scorer)
    monkeypatch.setenv(
        "GF_BICGSTAB_ADMIT_TOL_AMP" if scorer == "amplitude" else "GF_BICGSTAB_ADMIT_TOL_JACOBI", repr(eta)
    )
    for name, value in knobs.items():
        monkeypatch.setenv(f"GF_BICGSTAB_ADMIT_{name}", str(value))
    basis = _seed_basis(cap=cap, comm=comm)
    # redistribute_psis SUMS per-rank contributions: only rank 0 provides amplitudes.
    owns = comm is None or comm.rank == 0
    blocks = [ManyBodyState.from_states([s]) for s in _seeds()] if owns else [ManyBodyState(width=1) for _ in _seeds()]
    seeds = [blk.to_states()[0] for blk in basis.redistribute_psis(*blocks)]
    hOp = _siam_6()
    A_op = z - hOp
    x0 = [ManyBodyState(width=1) for _ in seeds]
    X, seeds, info, record, proxy = solve_point_outer(
        A_op, hOp, z, seeds, x0, basis, cap, 0.0, ATOL, 500, comm, len(seeds), solve_shifted_block
    )
    gram = block_inner_cy(ManyBodyState.from_states(seeds), X)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, gram, op=MPI.SUM)
    assert info["converged"]
    return gram, record, proxy, basis, A_op, seeds


def _retained(proxy, comm=None):
    keys = proxy.retained_keys()
    if comm is not None:
        keys = [k for part in comm.allgather(keys) for k in part]
    return sorted(set(keys))


@pytest.mark.parametrize("eta", [1e-1, 1e-2, 1e-3, 1e-6])
def test_the_answer_is_the_exact_resolvent_of_PHP_on_the_retained_set(monkeypatch, eta):
    G, record, proxy, basis, *_ = _outer_point(monkeypatch, eta=eta)
    retained = _retained(proxy)
    assert len(retained) == proxy.retained_size == int(basis.size) == record["final_size"]
    np.testing.assert_allclose(G, _dense_G_on(retained, [Z])[0], atol=1e-9)


def test_threshold_zero_reaches_the_full_closure_and_the_exact_G(monkeypatch):
    G, record, *_ = _outer_point(monkeypatch, eta=0.0)
    assert record["exit_reason"] == "converged"
    assert record["final_size"] == CLOSURE
    np.testing.assert_allclose(G, _dense_G_on(_n3_sector_dets(), [Z])[0], atol=1e-9)


def test_a_huge_threshold_admits_nothing_beyond_the_start_set(monkeypatch):
    _G, record, _proxy, _b, _A, _s = _outer_point(monkeypatch, eta=1e6)
    assert record["exit_reason"] == "converged"
    assert record["rounds"] == 0
    assert record["final_size"] == record["start_size"] < CLOSURE


def test_the_start_set_holds_the_seeds_and_their_first_shell(monkeypatch):
    """Whatever the threshold, the moments of G through H^2 (the Sigma tail) stay exact."""
    _G, _record, proxy, _basis, A_op, seeds = _outer_point(monkeypatch, eta=1e6)
    shell = set(A_op.apply_block(ManyBodyState.from_states(seeds), 0.0).keys())
    seed_keys = {k for s in seeds for k in s.keys()}
    retained = set(_retained(proxy))
    assert seed_keys <= retained and shell <= retained


@pytest.mark.parametrize("cap", [6, 10, 14])
def test_the_cap_is_never_exceeded_and_reports_budget(monkeypatch, cap):
    G, record, proxy, *_ = _outer_point(monkeypatch, cap=cap, eta=0.0)
    assert proxy.retained_size <= cap or record["start_size"] > cap
    assert record["exit_reason"] == "budget" and record["cap_hit"]
    retained = _retained(proxy)
    np.testing.assert_allclose(G, _dense_G_on(retained, [Z])[0], atol=1e-9)


@pytest.mark.parametrize("shell_tol", [0.3, 0.6])
def test_a_relaxed_first_shell_shrinks_the_start_set_and_stays_exact(monkeypatch, shell_tol):
    """``GF_ADMIT_FIRST_SHELL_TOL`` prunes weak first-shell rows; the answer is still the exact PHP
    resolvent on whatever was kept."""
    _G, strict, *_ = _outer_point(monkeypatch, eta=1e6)
    monkeypatch.setenv("GF_ADMIT_FIRST_SHELL_TOL", str(shell_tol))
    G, record, proxy, *_ = _outer_point(monkeypatch, eta=1e6)
    assert record["start_size"] < strict["start_size"]
    np.testing.assert_allclose(G, _dense_G_on(_retained(proxy), [Z])[0], atol=1e-9)


def test_the_jacobi_scorer_gives_the_same_contract(monkeypatch):
    G, _record, proxy, _b, _A, _s = _outer_point(monkeypatch, eta=1e-3, scorer="jacobi")
    np.testing.assert_allclose(G, _dense_G_on(_retained(proxy), [Z])[0], atol=1e-9)


def test_one_shell_per_round_reaches_the_same_closure_in_more_rounds(monkeypatch):
    _G, two, *_ = _outer_point(monkeypatch, eta=0.0)
    _G, one, *_ = _outer_point(monkeypatch, eta=0.0, SHELLS=1)
    assert one["final_size"] == two["final_size"] == CLOSURE
    assert one["rounds"] >= two["rounds"]


def test_the_estimate_stops_admission_early_without_changing_the_contract(monkeypatch):
    G, record, proxy, *_ = _outer_point(monkeypatch, eta=0.0, EN_TOL=0.5)
    assert record["exit_reason"] == "estimate"
    np.testing.assert_allclose(G, _dense_G_on(_retained(proxy), [Z])[0], atol=1e-9)


# --- the driver ---------------------------------------------------------------------------------


def _driver(monkeypatch, **env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    z_axes = _gf_signed_axes(None, OMEGA, 0, DELTA)
    G, stats = block_Green_bicgstab(_siam_6(), _seeds(), _seed_basis(), [0.0], 2, z_axes, atol=ATOL)
    return G[0][0], stats, z_axes[0]


def test_all_is_bit_identical_to_the_unset_knob(monkeypatch):
    G0, s0, _ = _driver(monkeypatch)
    G1, s1, _ = _driver(monkeypatch, GF_BICGSTAB_ADMISSION="all")
    assert np.array_equal(G0, G1)
    assert [p["solve_size"] for p in s0["points"]] == [p["solve_size"] for p in s1["points"]]


def test_the_driver_under_outer_matches_the_dense_resolvent_and_records_each_point(monkeypatch):
    G, stats, z = _driver(monkeypatch, GF_BICGSTAB_ADMISSION="outer", GF_BICGSTAB_ADMIT_TOL_AMP="0")
    np.testing.assert_allclose(G, _dense_G_on(_n3_sector_dets(), z), atol=1e-9)
    assert stats["n_unconverged"] == 0 and len(stats["points"]) == len(z)
    for p in stats["points"]:
        assert p["admission"]["exit_reason"] == "converged"
        assert p["seed_size"] <= p["rebuild_size"] <= p["solve_size"] == CLOSURE


def test_a_cold_point_starts_from_the_seeds_and_their_shell_not_the_whole_warm_support(monkeypatch):
    """The warm-start guess contributes only the determinants that still matter at this z."""
    _G, stats, _z = _driver(monkeypatch, GF_BICGSTAB_ADMISSION="outer", GF_BICGSTAB_ADMIT_TOL_AMP="1e-2")
    _G, warm_all, _z = _driver(monkeypatch, GF_BICGSTAB_ADMISSION="all")
    assert max(p["rebuild_size"] for p in stats["points"]) <= max(p["rebuild_size"] for p in warm_all["points"])


@pytest.mark.parametrize("name,value", [("GF_BICGSTAB_ADMISSION", "inner"), ("GF_BICGSTAB_ADMIT_SCORER", "residual")])
def test_an_unknown_policy_raises_rather_than_falling_back(monkeypatch, name, value):
    monkeypatch.setenv("GF_BICGSTAB_ADMISSION", "outer")
    monkeypatch.setenv(name, value)
    z_axes = _gf_signed_axes(None, OMEGA[:2], 0, DELTA)
    with pytest.raises(ValueError, match=name):
        block_Green_bicgstab(_siam_6(), _seeds(), _seed_basis(), [0.0], 2, z_axes, atol=ATOL)


# --- MPI ----------------------------------------------------------------------------------------


@pytest.mark.mpi
@pytest.mark.parametrize("eta", [1e-2, 1e-6])
def test_distributed_answer_is_the_exact_PHP_resolvent_on_the_gathered_retained_set(monkeypatch, eta):
    comm = MPI.COMM_WORLD
    G, record, proxy, *_ = _outer_point(monkeypatch, eta=eta, comm=comm)
    retained = _retained(proxy, comm)
    assert len(retained) == proxy.retained_size == record["final_size"]
    np.testing.assert_allclose(G, _dense_G_on(retained, [Z])[0], atol=1e-9)


@pytest.mark.mpi
def test_distributed_threshold_zero_reaches_the_same_closure_as_serial(monkeypatch):
    comm = MPI.COMM_WORLD
    _G, _r, serial_proxy, *_ = _outer_point(monkeypatch, eta=0.0)
    G, record, proxy, *_ = _outer_point(monkeypatch, eta=0.0, comm=comm)
    assert _retained(proxy, comm) == _retained(serial_proxy)
    assert record["final_size"] == CLOSURE and record["exit_reason"] == "converged"
    np.testing.assert_allclose(G, _dense_G_on(_n3_sector_dets(), [Z])[0], atol=1e-9)


@pytest.mark.mpi
@pytest.mark.parametrize("cap", [6, 12])
def test_distributed_cap_is_respected(monkeypatch, cap):
    comm = MPI.COMM_WORLD
    G, record, proxy, *_ = _outer_point(monkeypatch, cap=cap, eta=0.0, comm=comm)
    retained = _retained(proxy, comm)
    assert len(retained) <= max(cap, record["start_size"])
    np.testing.assert_allclose(G, _dense_G_on(retained, [Z])[0], atol=1e-9)
