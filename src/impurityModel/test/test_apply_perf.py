"""Phase 0 of the ManyBodyOperator::apply performance plan
(doc/plans/manybodyoperator_apply_performance.md).

Provides:

* **Representative fixtures** (0.1) — a 1-/2-body number-conserving Hamiltonian, an
  unpaired transition operator, and a constant shift, applied to a multi-thousand-SD
  state at ``n_orbs = 160`` (3 ``uint64`` chunks).
* **Golden-output oracle** (0.2) — the regression gate every later phase must keep
  green. The reference is the sorted ``(key_chunks, re, im)`` list of
  ``op.apply_multi([psi], 0.0)``; it is regenerated when missing or when
  ``REGEN_APPLY_GOLDEN=1`` is set, and committed alongside this file.
* **Timing harness** (0.3) — median wall-time per ``apply_multi`` printed under
  ``pytest -s``; no assertion on absolute time (machine-dependent), only that repeats
  agree.

The fixtures are fully deterministic (seeded ``random.Random``) so the oracle is
stable across runs and machines (up to floating-point summation order, hence the
tolerance compare).
"""

import json
import os
import random
import time
from pathlib import Path

import pytest

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant

N_ORBS = 160
N_CHUNKS = (N_ORBS + 63) // 64
N_ELEC = 80

# Small fixtures back the committed golden oracle (keeps the reference file tiny while
# still exercising every code path); large fixtures back the timing harness only.
ORACLE_STATES = 40
ORACLE_1BODY = 30
ORACLE_2BODY = 30
TIMING_STATES = 2000
TIMING_1BODY = 300
TIMING_2BODY = 300

GOLDEN_PATH = Path(__file__).parent / "apply_perf_golden.json"
TOL = 1e-9


# --------------------------------------------------------------------------- #
# Fixture construction
# --------------------------------------------------------------------------- #
def _sd_from_orbitals(orbitals):
    """Pack occupied orbital indices into a SlaterDeterminant.

    Orbital ``idx`` -> chunk ``idx // 64``, bit ``63 - (idx % 64)`` (MSB-first within
    the chunk), matching the C++ ``create``/``annihilate`` bit convention
    (``bit_idx = num_bits - 1 - idx % num_bits``).
    """
    chunks = [0] * N_CHUNKS
    for idx in orbitals:
        chunks[idx // 64] |= 1 << (63 - (idx % 64))
    return SlaterDeterminant(tuple(chunks))


def _make_state(rng, n_states):
    """A number-conserving state of ``n_states`` distinct ``N_ELEC``-electron SDs."""
    d = {}
    while len(d) < n_states:
        orbs = tuple(sorted(rng.sample(range(N_ORBS), N_ELEC)))
        d[_sd_from_orbitals(orbs)] = complex(rng.gauss(0, 1), rng.gauss(0, 1))
    return ManyBodyState(d)


def _make_hamiltonian(rng, n1, n2):
    """Random 1-/2-body number-conserving terms: c^d_i c_j and c^d_i c^d_j c_k c_l."""
    op = {}
    for _ in range(n1):
        key = ((rng.randrange(N_ORBS), "c"), (rng.randrange(N_ORBS), "a"))
        op[key] = op.get(key, 0) + complex(rng.gauss(0, 1), rng.gauss(0, 1))
    for _ in range(n2):
        i, j, k, l = (rng.randrange(N_ORBS) for _ in range(4))
        key = ((i, "c"), (j, "c"), (k, "a"), (l, "a"))
        op[key] = op.get(key, 0) + complex(rng.gauss(0, 1), rng.gauss(0, 1))
    return ManyBodyOperator(op)


def _make_transition(rng, n):
    """Unpaired single-operator terms (the self-energy / spectra path)."""
    op = {}
    for _ in range(n):
        kc = ((rng.randrange(N_ORBS), "c"),)
        ka = ((rng.randrange(N_ORBS), "a"),)
        op[kc] = op.get(kc, 0) + complex(rng.gauss(0, 1), rng.gauss(0, 1))
        op[ka] = op.get(ka, 0) + complex(rng.gauss(0, 1), rng.gauss(0, 1))
    return ManyBodyOperator(op)


def _make_constant(rng):
    """A pure scalar shift: zero elementary operators."""
    return ManyBodyOperator({(): complex(rng.gauss(0, 1), rng.gauss(0, 1))})


def _make_diagonal(rng, n):
    """Density-density / number operators: c^d_i c^d_j c_j c_i (= n_i n_j) and c^d_i c_i.

    Every term is diagonal (conserves each orbital's occupation), exercising the Phase 2b
    single-insert fast path; the output is a subset of the input determinants.
    """
    op = {}
    for _ in range(n):
        i, j = rng.randrange(N_ORBS), rng.randrange(N_ORBS)
        if i == j:
            continue
        key = ((i, "c"), (j, "c"), (j, "a"), (i, "a"))
        op[key] = op.get(key, 0) + complex(rng.gauss(0, 1), rng.gauss(0, 1))
    for _ in range(n):
        i = rng.randrange(N_ORBS)
        key = ((i, "c"), (i, "a"))
        op[key] = op.get(key, 0) + complex(rng.gauss(0, 1), rng.gauss(0, 1))
    return ManyBodyOperator(op)


def build_fixtures(n_states, n1, n2):
    """Return ``{name: (operator, state)}`` for all fixtures, deterministically."""
    rng = random.Random(20260628)
    psi = _make_state(rng, n_states)
    return {
        "hamiltonian": (_make_hamiltonian(rng, n1, n2), psi),
        "transition": (_make_transition(rng, n1), psi),
        "constant": (_make_constant(rng), psi),
        "diagonal": (_make_diagonal(rng, n1 + n2), psi),
    }


FIXTURE_NAMES = ["hamiltonian", "transition", "constant", "diagonal"]


def build_oracle_fixtures():
    return build_fixtures(ORACLE_STATES, ORACLE_1BODY, ORACLE_2BODY)


def build_timing_fixtures():
    return build_fixtures(TIMING_STATES, TIMING_1BODY, TIMING_2BODY)


# --------------------------------------------------------------------------- #
# Golden-oracle serialization
# --------------------------------------------------------------------------- #
def _serialize(state):
    """Sorted list of ``[chunk_tuple, re, im]`` — order-independent canonical form."""
    items = []
    for sd, amp in state.items():
        items.append((tuple(int(c) for c in sd), float(amp.real), float(amp.imag)))
    items.sort(key=lambda t: t[0])
    return [[list(chunks), re, im] for chunks, re, im in items]


def _apply(op, psi):
    return op.apply_multi([psi], 0.0)[0]


def _load_golden():
    if not GOLDEN_PATH.exists():
        return None
    with open(GOLDEN_PATH) as fh:
        return json.load(fh)


def _regen_golden():
    golden = {name: _serialize(_apply(op, psi)) for name, (op, psi) in build_oracle_fixtures().items()}
    with open(GOLDEN_PATH, "w") as fh:
        json.dump(golden, fh)
    return golden


def _assert_matches(name, produced, reference):
    """Compare two serialized states key-by-key within TOL."""
    prod = {tuple(c): (re, im) for c, re, im in produced}
    ref = {tuple(c): (re, im) for c, re, im in reference}
    assert prod.keys() == ref.keys(), (
        f"[{name}] key set differs: "
        f"{len(prod.keys() - ref.keys())} extra, {len(ref.keys() - prod.keys())} missing"
    )
    for key, (re, im) in ref.items():
        pre, pim = prod[key]
        assert abs(pre - re) < TOL and abs(pim - im) < TOL, f"[{name}] amplitude mismatch at {key}"


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def oracle_fixtures():
    return build_oracle_fixtures()


@pytest.fixture(scope="module")
def timing_fixtures():
    return build_timing_fixtures()


@pytest.fixture(scope="module")
def golden():
    if os.environ.get("REGEN_APPLY_GOLDEN") == "1" or not GOLDEN_PATH.exists():
        return _regen_golden()
    return _load_golden()


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_apply_matches_golden(name, oracle_fixtures, golden):
    op, psi = oracle_fixtures[name]
    produced = _serialize(_apply(op, psi))
    _assert_matches(name, produced, golden[name])


def test_apply_is_deterministic(oracle_fixtures):
    """Same operator + state -> same result across repeated applications."""
    op, psi = oracle_fixtures["hamiltonian"]
    first = _serialize(_apply(op, psi))
    second = _serialize(_apply(op, psi))
    _assert_matches("hamiltonian", second, first)


def _occupied_orbitals(sd):
    """Set of occupied orbital indices of a SlaterDeterminant (MSB-first per chunk)."""
    occ = set()
    for chunk_idx, chunk in enumerate(sd):
        for bit in range(64):
            if chunk & (1 << (63 - bit)):
                occ.add(chunk_idx * 64 + bit)
    return occ


def test_diagonal_independent(oracle_fixtures):
    """Independent oracle for the Phase 2b diagonal fast path.

    The diagonal fixture is built only from ``n_i n_j`` (``c^d_i c^d_j c_j c_i`` with
    i!=j, which equals ``n_i n_j`` exactly since the two number operators commute) and
    ``n_i`` terms. Each has eigenvalue ``prod_o occ(o)`` and sign +1, so the expected
    output amplitude is computable from occupancies alone -- no fermion-sign machinery,
    making this a true cross-check of the C++ apply rather than a self-comparison.
    """
    op, psi = oracle_fixtures["diagonal"]
    terms = [({p[0] for p in processes}, coeff) for processes, coeff in op.items()]

    expected = {}
    for sd, amp in psi.items():
        occ = _occupied_orbitals(sd)
        val = sum(coeff for orbs, coeff in terms if orbs <= occ) * amp
        if val != 0:
            expected[tuple(int(c) for c in sd)] = val

    produced = {tuple(int(c) for c in k): v for k, v in _apply(op, psi).items()}
    assert expected.keys() == produced.keys(), "diagonal support differs from occupancy oracle"
    for key, ev in expected.items():
        assert abs(ev - produced[key]) < TOL, f"diagonal amplitude mismatch at {key}"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_apply_timing(name, timing_fixtures, capsys):
    """Phase 0.3 timing harness: print median ms/apply (no absolute-time assertion)."""
    op, psi = timing_fixtures[name]
    reps = 7
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = _apply(op, psi)
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    median = times[len(times) // 2]
    with capsys.disabled():
        print(f"\n[apply-perf] {name:11s} n_out={len(out):6d}  median={median:8.2f} ms  best={times[0]:8.2f} ms")
