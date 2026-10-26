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
        a = amp[0]
        items.append((tuple(int(c) for c in sd), float(a.real), float(a.imag)))
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
    """Compare two serialized states key-by-key within TOL.

    A determinant present on only one side is a mismatch only if its amplitude is
    resolvable: ``apply`` drops an output whose contributions cancel to *exactly* 0.0
    but keeps one that cancels to a rounding residue, so two summation orders over the
    same operator legitimately differ by keys carrying ~1e-16. Below TOL the two states
    are numerically indistinguishable, and a genuine regression moves weight far above it.
    """
    prod = {tuple(c): (re, im) for c, re, im in produced}
    ref = {tuple(c): (re, im) for c, re, im in reference}
    for label, only in (("extra", prod.keys() - ref.keys()), ("missing", ref.keys() - prod.keys())):
        side = prod if label == "extra" else ref
        significant = [k for k in only if abs(complex(*side[k])) >= TOL]
        assert not significant, f"[{name}] {len(significant)} {label} determinant(s) with amplitude >= {TOL}"
    for key, (re, im) in ref.items():
        pre, pim = prod.get(key, (0.0, 0.0))
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
    ``n_i`` terms. Each has eigenvalue ``prod_o occ(o)`` up to a sign fixed purely by the
    term's *shape*, so the expected output amplitude is computable from occupancies alone
    -- no fermion-sign machinery, making this a true cross-check of the C++ apply rather
    than a self-comparison.

    The sign: terms are stored in canonical normal order, which sorts the annihilators
    ascending, whereas ``n_{a_1}..n_{a_k} = c^d_{a_1}..c^d_{a_k} c_{a_k}..c_{a_1}`` has
    them descending. Reversing k elements costs ``(-1)^(k(k-1)/2)``, so a canonical
    k-body diagonal term with coefficient ``c`` has eigenvalue
    ``c * (-1)^(k(k-1)/2) * prod_o occ(o)`` -- +1 for k=1, -1 for k=2.
    """
    op, psi = oracle_fixtures["diagonal"]
    terms = []
    for processes, coeff in op.items():
        orbs = {p[0] for p in processes}
        k = sum(1 for p in processes if p[1] == "c")
        terms.append((orbs, coeff * (-1) ** ((k * (k - 1)) // 2)))

    expected = {}
    for sd, amp in psi.items():
        occ = _occupied_orbitals(sd)
        val = sum(coeff for orbs, coeff in terms if orbs <= occ) * amp[0]
        if val != 0:
            expected[tuple(int(c) for c in sd)] = val

    produced = {tuple(int(c) for c in k): v[0] for k, v in _apply(op, psi).items()}
    assert expected.keys() == produced.keys(), "diagonal support differs from occupancy oracle"
    for key, ev in expected.items():
        assert abs(ev - produced[key]) < TOL, f"diagonal amplitude mismatch at {key}"


def _make_non_normal_ordered(rng, n):
    """Contraction-heavy operator with annihilations left of creations (Phase 3 input).

    Built through ``__setitem__`` rather than the constructor **on purpose**: the
    constructor canonicalizes (see ``ManyBodyOperator.canonicalize``), so a
    dict-constructed operator is normal-ordered before it is ever applied and could not
    exercise the flat-representation ordering path at all. ``__setitem__`` is the one
    route that leaves raw, unordered term strings in storage.
    """
    op = ManyBodyOperator()
    amps = {}
    for _ in range(n):
        a, c = rng.randrange(N_ORBS), rng.randrange(N_ORBS)
        # Half the adjacent (annihilation, creation) pairs share an orbital, so the
        # anticommutator c_a c^d_a = 1 - n_a actually fires and the recursion emits
        # contraction terms -- with four independent draws over N_ORBS orbitals a
        # collision is vanishingly rare and the path would go untested.
        b = a if rng.random() < 0.5 else rng.randrange(N_ORBS)
        d = c if rng.random() < 0.5 else rng.randrange(N_ORBS)
        key = ((a, "a"), (b, "c"), (c, "a"), (d, "c"))  # c_a c^d_b c_c c^d_d
        amps[key] = amps.get(key, 0) + complex(rng.gauss(0, 1), rng.gauss(0, 1))
        op[key] = amps[key]
    assert not op.is_canonical(), "fixture must stay non-canonical to be a meaningful A/B"
    return op


def test_normal_ordering_contraction():
    """c_i c^d_i = 1 - n_i: the build-time contraction must produce the constant + (-n_i)."""
    i = 5
    op = ManyBodyOperator({((i, "a"), (i, "c")): 1.0 + 0j})  # c_i c^d_i, product order
    assert op.num_flat_terms() == 2  # constant (1) and -n_i
    s_occ = _sd_from_orbitals([i, 7, 70])
    s_emp = _sd_from_orbitals([7, 70])
    out_occ = _apply(op, ManyBodyState({s_occ: 3.0 + 0j}))
    out_emp = _apply(op, ManyBodyState({s_emp: 3.0 + 0j}))
    occ_amp = out_occ.get(s_occ)
    emp_amp = out_emp.get(s_emp)
    occ_val = 0.0 if occ_amp is None else occ_amp[0]
    emp_val = 0.0 if emp_amp is None else emp_amp[0]
    assert abs(occ_val - 0.0) < TOL  # 1 - n_i = 0 when occupied
    assert abs(emp_val - 3.0) < TOL  # 1 - n_i = 1 when empty


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


def test_normal_ordering_multiplier(oracle_fixtures):
    """Guard against a normal-order term-count blow-up.

    The constructor-built fixtures are canonical, so their flat representation is a copy
    of their stored terms by construction -- an on/off A/B on them measures nothing. The
    measurement that still has content is on a raw, non-canonical operator, where the
    anticommutator recursion actually runs: a contraction-heavy input is *allowed* to
    expand (each ``c_a c^d_b`` crossing emits a contraction term), but not without bound.
    """
    for name in FIXTURE_NAMES:
        op, _ = oracle_fixtures[name]
        assert op.is_canonical(), f"{name} fixture should be canonical from its constructor"
        assert op.num_flat_terms() == len(op), f"{name} flat rep should mirror canonical storage"

    raw_op = _make_non_normal_ordered(random.Random(11), 60)
    raw_op.set_normal_ordering(False)
    raw = raw_op.num_flat_terms()
    raw_op.set_normal_ordering(True)
    normal = raw_op.num_flat_terms()
    mult = normal / raw if raw else 1.0
    assert mult <= 4.0, f"normal-order expansion {mult:.2f} exceeds 4.0"


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
    the ManyBodyState target is near-flat in p (term/sign/accumulator work done
    once per determinant, p FMAs per emission). This baseline is what the block
    container is measured against."""

    op, psi = timing_fixtures["hamiltonian"]
    support = list(psi.to_dict().keys())
    rows = []
    for p in (1, 2, 4, 8):
        rng = random.Random(97 + p)
        psis = [ManyBodyState({sd: complex(rng.random(), rng.random()) for sd in support}) for _ in range(p)]
        blk = ManyBodyState.from_states(psis)
        times, btimes = [], []
        for _ in range(5):
            t0 = time.perf_counter()
            out = op.apply_multi(psis, 0.0)
            times.append((time.perf_counter() - t0) * 1e3)
            t0 = time.perf_counter()
            op.apply_block(blk, 0.0)
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


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_block_apply_restriction_deltas_match_the_per_state_check(seed):
    """The block apply checks a restriction as ``row value + per-term delta``; the scalar apply popcounts the
    output. Count and weighted restrictions (overlapping groups, weights of both signs) must keep and drop
    exactly the same output determinants with the same amplitudes."""
    rng = random.Random(seed)
    op = _make_hamiltonian(rng, 60, 60)
    psi = _make_state(rng, 60)
    # N_ELEC = 80 of 160 orbitals: bounds straddle the typical occupation so some outputs are cut.
    op.set_restrictions({frozenset(range(0, 40)): (18, 22), frozenset(range(30, 90)): (28, 33)})
    op.set_weighted_restrictions(
        [
            ({o: (1 if o % 2 else -1) for o in range(0, 80)}, (-3, 3)),
            ({o: 2 for o in range(10, 30)} | {o: -1 for o in range(25, 60)}, (-12, 14)),
        ]
    )
    blk = ManyBodyState.from_states([psi])
    got = op.apply_block(blk, 0.0).to_states()[0].to_dict()
    ref = op.apply_multi([psi], 0.0)[0].to_dict()
    assert 0 < len(ref)
    unrestricted = _make_hamiltonian(random.Random(seed), 60, 60).apply_multi([psi], 0.0)[0].to_dict()
    assert len(ref) < len(unrestricted), "the restrictions cut nothing, so the test discriminates nothing"
    assert set(got) == set(ref)
    for k, v in ref.items():
        assert abs(got[k] - v) <= 1e-12 * max(1.0, abs(v))


def test_block_apply_matches_scalar_apply_on_terms_that_rarely_or_never_fire():
    """The block apply runs a term's create/annihilate sequence only where an occupancy mask says it fires.
    Strings that revisit an orbital (c_i c^d_i, c^d_i c_i c_j ...), annihilate-then-create chains and
    repeated creators never fire or fire conditionally; the block result must equal the scalar one."""
    rng = random.Random(11)
    terms = {}
    for _ in range(40):
        i, j, k = (rng.randrange(0, 12) for _ in range(3))
        for key in (
            ((i, "c"), (i, "a")),
            ((i, "a"), (i, "c")),
            ((i, "c"), (j, "c"), (j, "a"), (k, "a")),
            ((i, "a"), (j, "c"), (j, "a"), (i, "c")),
            ((i, "c"), (i, "c")),
            ((i, "c"), (j, "a"), (i, "a"), (k, "c")),
        ):
            terms[key] = terms.get(key, 0) + complex(rng.gauss(0, 1), rng.gauss(0, 1))
    # an orbital beyond the determinant's width never fires
    terms[((N_ORBS + 5, "c"), (3, "a"))] = 1.0 + 0.5j
    op = ManyBodyOperator(terms)
    states = {}
    while len(states) < 50:
        occ = tuple(sorted(rng.sample(range(0, 12), rng.randrange(3, 9))))
        states[_sd_from_orbitals(occ)] = complex(rng.gauss(0, 1), rng.gauss(0, 1))
    psi = ManyBodyState(states)
    got = op.apply_block(ManyBodyState.from_states([psi]), 0.0).to_states()[0].to_dict()
    ref = op.apply_multi([psi], 0.0)[0].to_dict()
    assert set(got) == set(ref)
    for k, v in ref.items():
        assert abs(got[k] - v) <= 1e-12 * max(1.0, abs(v))


def test_wide_block_apply_with_restrictions_matches_scalar_apply():
    """A block big enough for the threaded apply (in an ``IMPURITYMODEL_PARALLEL=1`` build) to split across
    threads, with restrictions on: the threaded accumulate/merge/sort must agree with the scalar apply
    column by column (this is also the serial path's wide-block check)."""
    rng = random.Random(23)
    op = _make_hamiltonian(rng, 40, 40)
    op.set_restrictions({frozenset(range(0, 60)): (26, 34)})
    op.set_weighted_restrictions([({o: (1 if o % 2 else -1) for o in range(0, 100)}, (-4, 4))])
    psis = []
    support = list(_make_state(rng, 1500).to_dict())
    for _ in range(3):
        psis.append(ManyBodyState({sd: complex(rng.gauss(0, 1), rng.gauss(0, 1)) for sd in support}))
    got = op.apply_block(ManyBodyState.from_states(psis), 0.0).to_states()
    ref = op.apply_multi(psis, 0.0)
    for g, r in zip(got, ref):
        gd, rd = g.to_dict(), r.to_dict()
        assert set(gd) == set(rd)
        for k, v in rd.items():
            assert abs(gd[k] - v) <= 1e-11 * max(1.0, abs(v))
