"""Opt-in benchmark of the block apply on the real NiO 10-bath solver Hamiltonian.

Run:  python -m pytest -m benchmark -s src/impurityModel/test/operators/test_apply_nio_bench.py

The operator is the one the Green's-function Lanczos applies (``prepare_solver_basis`` of the archived model,
excited-sector restrictions as the run logged them); the block is the support the matvec reaches from the
d8 seed after a few generations. It prints microseconds per input row for block widths 1, 2 and 4 (what the
Green's-function path uses), the per-class cost of each operator subset, and checks the block apply against the
independent reference (``apply_reference``) on a sample of rows. Skipped when the archive is absent.

Reference numbers on the development laptop, serial release build, 103k-row block, width 1:
2.27 us/row (before ``doc/plans/apply_large_support_perf.md`` part 2) -> 1.21 us/row.
"""

import os
import random
import time
from pathlib import Path

import numpy as np
import pytest

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant

from .apply_reference import (
    assert_matches,
    block_from_orbital_sets,
    block_to_orbital_sets,
    orbitals_from_chunks,
    reference_apply,
)

pytestmark = pytest.mark.benchmark

ARCHIVE = Path(
    os.environ.get(
        "APPLY_BENCH_ARCHIVE",
        "~/Programming/impmod_tests/NiO/impmod/10_BathStates_linked_chainGeometry_noneReorthonormalization_3_processors_/"
        "impurityModel_data.h5",
    )
).expanduser()
# Excited-sector windows of that run (its log, "chain_restrict" graded windows) and the excitation-budget form.
CHAIN_WINDOWS = {
    frozenset([10, 11, 12, 22, 23, 24, 58, 59, 60, 70, 71, 72]): (11, 12),
    frozenset([13, 25, 61, 73]): (2, 4),
    frozenset([34, 42, 50, 82, 90, 98]): (5, 6),
    frozenset([35, 43, 51, 83, 91, 99]): (4, 6),
}
N_IMP = 10


def _solver_basis():
    """``(model, sb)``: the archived model and its ``prepare_solver_basis`` result (skips without the archive)."""
    if not ARCHIVE.exists():
        pytest.skip(f"archive {ARCHIVE} not available")
    from impurityModel.ed.solver_basis import prepare_solver_basis
    from impurityModel.test.support import real_workload as rw

    wl = rw.load_workload(str(ARCHIVE), iteration=1)
    model, _, basis, _ = rw.build_options(wl)
    sb = prepare_solver_basis(
        model.h0,
        model.dc,
        model.u4,
        model.impurity_orbitals,
        basis.nominal_occ,
        rot_to_spherical=model.rot_to_spherical,
        verbosity=0,
    )
    return model, sb


@pytest.fixture(scope="module")
def nio():
    from impurityModel.ed.basis_restrictions import build_weighted_restrictions

    _model, sb = _solver_basis()
    weights, (lo, hi) = build_weighted_restrictions(sb.bath_states, 6)[0]
    weighted = [(weights, (lo - 1, hi + 1))]
    h = sb.h
    h.set_restrictions(CHAIN_WINDOWS)
    h.set_weighted_restrictions(weighted)
    valence = sorted(o for o, w in weights.items() if w < 0)
    chunks = [0, 0]
    for o in {0, 5, 2, 3, 4, 7, 8, 9} | set(valence):
        chunks[o // 64] |= 1 << (63 - o % 64)
    psi = ManyBodyState({SlaterDeterminant(tuple(chunks)): 1.0 + 0j})
    for _ in range(8):
        psi = h.apply_block(psi, 0.0)
        if len(psi) > 100_000:
            break
    return h, weighted, psi


def _timed(op, blk, reps=4):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        op.apply_block(blk, 0.0)
        best = min(best, time.perf_counter() - t0)
    return best


def test_nio_block_apply_matches_reference_on_a_sample(nio):
    h, weighted, psi = nio
    keys = list(psi.keys())
    random.seed(2)
    rng = np.random.default_rng(1)
    states = {
        orbitals_from_chunks(tuple(k)): rng.standard_normal(2) + 1j * rng.standard_normal(2)
        for k in random.sample(keys, 400)
    }
    got = block_to_orbital_sets(h.apply_block(block_from_orbital_sets(states, 2), 0.0))
    ref = reference_apply(h.to_dict(), states, 2, CHAIN_WINDOWS, weighted)
    assert len(ref) > 1000
    assert_matches(got, ref)


def test_nio_block_apply_speed(nio, capsys):
    h, _, psi = nio
    n = len(psi)
    rng = np.random.default_rng(0)
    terms = h.to_dict()
    isimp = lambda k: all(o < N_IMP for o, _ in k)  # noqa: E731
    isdiag = lambda k: sorted(o for o, c in k if c == "c") == sorted(o for o, c in k if c == "a")  # noqa: E731
    subsets = {
        "impurity off-diagonal": lambda k: isimp(k) and not isdiag(k),
        "impurity diagonal": lambda k: isimp(k) and isdiag(k),
        "bath density": lambda k: not isimp(k) and isdiag(k),
        "bath/hybridisation hops": lambda k: not isimp(k) and not isdiag(k),
    }
    with capsys.disabled():
        print(f"\n[apply-nio] {n} input rows, {h.num_flat_terms()} terms")
        for p in (1, 2, 4):
            blk = ManyBodyState.from_keys_and_amps(psi, rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p)))
            print(f"[apply-nio] p={p}: {1e6 * _timed(h, blk) / n:.2f} us/row (full operator)")
        blk = ManyBodyState.from_keys_and_amps(psi, rng.standard_normal((n, 1)) + 1j * rng.standard_normal((n, 1)))
        for name, pred in subsets.items():
            sub = ManyBodyOperator({k: v for k, v in terms.items() if pred(k)})
            sub.set_restrictions(CHAIN_WINDOWS)
            print(f"[apply-nio]   {name:26s} {sub.num_flat_terms():4d} terms: {1e6 * _timed(sub, blk) / n:.2f} us/row")


def test_nio_cost_form_per_row_cost(nio, capsys):
    """What a second weighted form with ~100 groups costs per input row (the cost budget's form), against the
    excitation budget's form alone. The plan coarsens the weight scale if the cost form adds 5 % or more."""
    from impurityModel.ed.basis_restrictions import build_weighted_restrictions

    h, weighted, psi = nio
    _model, sb = _solver_basis()
    both = build_weighted_restrictions(
        sb.bath_states,
        6,
        cost_budget=0.3,
        h_op=h,
        impurity_orbitals=sb.impurity_orbitals,
        slater_weight_min=1e-10,
    )
    assert len(both) == 2
    cost_weights, cost_bounds = both[1]
    # the bound is irrelevant to the timing; widen it so the form never rejects and the output work is equal
    cost_form = (cost_weights, (-(10**6), 10**6))
    groups = len({w for w in cost_weights.values() if w})
    rng = np.random.default_rng(0)
    n = len(psi)
    blk = ManyBodyState.from_keys_and_amps(psi, rng.standard_normal((n, 1)) + 1j * rng.standard_normal((n, 1)))
    times = {}
    for name, forms in (("excitation form only", weighted), ("excitation + cost form", weighted + [cost_form])):
        h.set_weighted_restrictions(forms)
        times[name] = _timed(h, blk, reps=6)
    h.set_weighted_restrictions(weighted)
    ratio = times["excitation + cost form"] / times["excitation form only"]
    with capsys.disabled():
        for name, t in times.items():
            print(f"[apply-nio-cost] {name:24s} {1e6 * t / n:.2f} us/row")
        print(f"[apply-nio-cost] cost form: {groups} weight groups, ratio {ratio:.3f}")
    assert ratio < 1.25
