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
        f"[{name}] key set differs: " f"{len(prod.keys() - ref.keys())} extra, {len(ref.keys() - prod.keys())} missing"
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


def _make_non_normal_ordered(rng, n):
    """Contraction-heavy operator with annihilations left of creations (Phase 3 input)."""
    op = {}
    for _ in range(n):
        a, b, c, d = (rng.randrange(N_ORBS) for _ in range(4))
        key = ((a, "a"), (b, "c"), (c, "a"), (d, "c"))  # c_a c^d_b c_c c^d_d
        op[key] = op.get(key, 0) + complex(rng.gauss(0, 1), rng.gauss(0, 1))
    return ManyBodyOperator(op)


def test_normal_ordering_contraction():
    """c_i c^d_i = 1 - n_i: the build-time contraction must produce the constant + (-n_i)."""
    i = 5
    op = ManyBodyOperator({((i, "a"), (i, "c")): 1.0 + 0j})  # c_i c^d_i, product order
    assert op.num_flat_terms() == 2  # constant (1) and -n_i
    s_occ = _sd_from_orbitals([i, 7, 70])
    s_emp = _sd_from_orbitals([7, 70])
    out_occ = _apply(op, ManyBodyState({s_occ: 3.0 + 0j}))
    out_emp = _apply(op, ManyBodyState({s_emp: 3.0 + 0j}))
    assert abs(out_occ.get(s_occ, 0) - 0.0) < TOL  # 1 - n_i = 0 when occupied
    assert abs(out_emp.get(s_emp, 0) - 3.0) < TOL  # 1 - n_i = 1 when empty


def test_normal_ordering_ab_invariance(oracle_fixtures):
    """Normal ordering is a representation change: apply() must be identical on/off,
    including on a contraction-heavy non-normal-ordered operator."""
    _, psi = oracle_fixtures["hamiltonian"]
    op = _make_non_normal_ordered(random.Random(7), 60)

    op.set_normal_ordering(True)
    on = _serialize(_apply(op, psi))
    op.set_normal_ordering(False)
    off = _serialize(_apply(op, psi))
    _assert_matches("non_normal_ordered", on, off)


def test_normal_ordering_multiplier(oracle_fixtures, capsys):
    """Report the term-count multiplier (Phase 3b). Already-normal-ordered Hamiltonian
    fixtures must not expand (multiplier ~1); the report flags any blow-up."""
    with capsys.disabled():
        for name in FIXTURE_NAMES:
            op, _ = oracle_fixtures[name]
            op.set_normal_ordering(False)
            raw = op.num_flat_terms()
            op.set_normal_ordering(True)
            normal = op.num_flat_terms()
            mult = normal / raw if raw else 1.0
            assert mult <= 1.5, f"{name} normal-order expansion {mult:.2f} exceeds 1.5"


@pytest.mark.benchmark
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_apply_timing(name, timing_fixtures, capsys):
    """Phase 0.3 timing harness: print median ms/apply (no absolute-time assertion).

    Marked ``benchmark`` so it is skipped by default (see pytest.ini); run with
    ``pytest -m benchmark``. Because ``timing_fixtures`` is used only here, the
    expensive fixture is never built during a standard run.
    """
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


@pytest.mark.benchmark
def test_apply_block_width_scaling(timing_fixtures, capsys):
    """Phase 2.0 of the block-state matvec plan (blocklanczos_partial_perf_memory.md):
    apply_multi cost vs block width p on shared-support states — the Lanczos block
    shape. Today the p matvecs run independently, so wall time scales ~linearly in p;
    the ManyBodyBlockState target is near-flat in p (term/sign/accumulator work done
    once per determinant, p FMAs per emission). This baseline is what the block
    container is measured against."""
    from impurityModel.ed.ManyBodyUtils import ManyBodyBlockState

    op, psi = timing_fixtures["hamiltonian"]
    support = list(psi.to_dict().keys())
    rows = []
    for p in (1, 2, 4, 8):
        rng = random.Random(97 + p)
        psis = [ManyBodyState({sd: complex(rng.random(), rng.random()) for sd in support}) for _ in range(p)]
        blk = ManyBodyBlockState.from_states(psis)
        times, btimes = [], []
        for _ in range(5):
            t0 = time.perf_counter()
            out = op.apply_multi(psis, 0.0)
            times.append((time.perf_counter() - t0) * 1e3)
            t0 = time.perf_counter()
            bout = op.apply_block(blk, 0.0)
            btimes.append((time.perf_counter() - t0) * 1e3)
        times.sort()
        btimes.sort()
        rows.append((p, times[len(times) // 2], btimes[len(btimes) // 2], len(out[0])))
    t1 = rows[0][1]
    with capsys.disabled():
        print()
        for p, med, bmed, n_out in rows:
            print(
                f"[apply-block] p={p}  multi={med:8.2f} ms ({med / t1:5.2f}x p=1)"
                f"  block={bmed:8.2f} ms  speedup={med / bmed:5.2f}x  n_out={n_out}"
            )
