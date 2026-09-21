import numpy as np
import pytest

from impurityModel.ed.ManyBodyUtils import (
    ManyBodyOperator,
    ManyBodyState,
    SlaterDeterminant,
)


def all_isclose(dict1, dict2, **kwargs):
    return all(pytest.approx(dict1[key][0]) == dict2[key][0] for key in dict1.keys()) and all(
        pytest.approx(dict1[key][0]) == dict2[key][0] for key in dict2.keys()
    )


def test_ManyBodyOperator():
    d = {((1, "c"),): 1.0, ((0, "a"),): 1j}
    op = ManyBodyOperator(d)

    for process, value in d.items():
        assert pytest.approx(value) == op[process]
    for process in op:
        assert pytest.approx(d[process]) == op[process]


def test_ManyBodyOperator_2():
    d = {((1, "c"),): 1.0, ((0, "a"),): 1j}
    op = ManyBodyOperator()

    for process, amp in d.items():
        op[process] = amp

    for process, value in d.items():
        assert pytest.approx(value) == op[process]
    for process in op:
        assert pytest.approx(d[process]) == op[process]


def test_ManyBodyOperator_arithmetic():
    add = {((1, "c"),): 1.0, ((0, "a"),): 1j}
    a = ManyBodyOperator({((1, "c"),): 1.0})
    b = ManyBodyOperator({((0, "a"),): 1j})

    op = a + b
    for state, value in add.items():
        assert pytest.approx(value) == op[state]
    for state in op:
        assert pytest.approx(add[state]) == op[state]

    sub = {((1, "c"),): 1.0, ((0, "a"),): -1j}
    op = a - b
    for state, value in sub.items():
        assert pytest.approx(value) == op[state]
    for state in op:
        assert pytest.approx(sub[state]) == op[state]

    scale = {((1, "c"),): 2.5, ((0, "a"),): 2.5j}
    op = (a + b) * 2.5
    for state, value in scale.items():
        assert pytest.approx(value) == op[state]
    for state, value in scale.items():
        assert pytest.approx(value) == op[state]

    op = 2.5 * (a + b)
    for state, value in scale.items():
        assert pytest.approx(value) == op[state]
    for state, value in scale.items():
        assert pytest.approx(value) == op[state]

    op = (a + b) / 0.4
    for state, value in scale.items():
        assert pytest.approx(value) == op[state]
    for state, value in scale.items():
        assert pytest.approx(value) == op[state]


def test_ManyBodyOperator_apply():
    #                      1010          1011
    psi = ManyBodyState({SlaterDeterminant.from_bytes(b"\xa0\x00"): 1.0, SlaterDeterminant.from_bytes(b"\xbf"): 1.0j})

    op = ManyBodyOperator({((0, "a"), (3, "c")): 1.0, ((1, "a"), (1, "c")): 1.0j})
    #                      0011          1010           1011
    res = ManyBodyState(
        {
            SlaterDeterminant.from_bytes(b"\x30"): 1.0,
            SlaterDeterminant.from_bytes(b"\xa0"): 1.0j,
            SlaterDeterminant.from_bytes(b"\xbf"): -1.0,
        }
    )

    assert all_isclose(res, op(psi, 0))


def test_ManyBodyOperator_apply2():
    #                      1010          1011           1110
    psi = ManyBodyState(
        {
            SlaterDeterminant.from_bytes(b"\xa0"): 1.0,
            SlaterDeterminant.from_bytes(b"\xbf"): 1.0j,
            SlaterDeterminant.from_bytes(b"\xe0"): 1e-13,
        }
    )

    op = ManyBodyOperator({((0, "a"), (3, "c")): 1.0, ((1, "a"), (1, "c")): 1.0j})
    #                      0011
    res = ManyBodyState(
        {
            SlaterDeterminant.from_bytes(b"\x30"): 1.0,
            SlaterDeterminant.from_bytes(b"\xa0"): 1.0j,
            SlaterDeterminant.from_bytes(b"\xbf"): -1.0,
        }
    )

    assert all_isclose(res, op(psi, 1e-12))


def test_ManyBodyOperator_apply3():
    #                      1010          1011
    psi = ManyBodyState({SlaterDeterminant.from_bytes(b"\xa0"): 1.0, SlaterDeterminant.from_bytes(b"\xbf"): 1.0j})

    op = ManyBodyOperator({((0, "a"), (3, "c")): 1.0, ((1, "a"), (1, "c")): 1.0j})
    #                      1010
    res = ManyBodyState({SlaterDeterminant.from_bytes(b"\xa0"): 1.0j})

    op.set_restrictions({frozenset([2, 3]): (1, 1)})
    assert all_isclose(res, op(psi, 0))


def test_ManyBodyOperator_pickle():
    import pickle

    op = ManyBodyOperator({((0, "a"), (3, "c")): 1.0, ((1, "a"), (1, "c")): 1.0j})
    pickled_op = pickle.dumps(op)
    new_op = pickle.loads(pickled_op)

    assert op == new_op


def test_SlaterDeterminant_operators():
    sd1 = SlaterDeterminant.from_bytes(b"\x01")
    sd2 = SlaterDeterminant.from_bytes(b"\x02")
    sd1_copy = SlaterDeterminant.from_bytes(b"\x01")

    assert sd1 < sd2
    assert sd2 > sd1
    assert sd1 == sd1_copy
    assert sd1 != sd2
    assert hash(sd1) == hash(sd1_copy)
    assert len(sd1) > 0
    assert repr(sd1).startswith("SlaterDeterminant")


def test_SlaterDeterminant_extra():
    sd = SlaterDeterminant.from_bytes(b"\x01\x02")
    assert len(sd) > 0
    # test __getitem__ and __setitem__
    val = sd[0]
    sd[0] = val + 1
    assert sd[0] == val + 1
    # test __iter__
    chunks = list(sd)
    assert len(chunks) == len(sd)
    # test __copy__ and __deepcopy__
    import copy

    sd_copy = copy.copy(sd)
    sd_deepcopy = copy.deepcopy(sd)
    assert sd == sd_copy
    assert sd == sd_deepcopy
    # test to_bytearray
    ba = sd.to_bytearray()
    assert isinstance(ba, bytearray)
    assert ba[0] == 1
    assert ba[1] == 2
    assert ba[7] == 1
    assert all(x == 0 for x in ba[2:7])
    assert all(x == 0 for x in ba[8:])


def test_ManyBodyOperator_extra():
    key1 = ((1, "c"),)
    key2 = ((0, "a"),)
    d = {key1: 1.0, key2: 2.0j}
    op = ManyBodyOperator(d)

    # test __contains__
    assert key1 in op
    assert ((2, "a"),) not in op

    # test keys(), values(), items(), to_dict()
    assert set(op.keys()) == {key1, key2}
    assert set(op.values()) == {1.0, 2.0j}
    assert dict(op.items()) == d
    assert op.to_dict() == d

    # test operator *= and /=
    op1 = ManyBodyOperator(d)
    op1 *= 2.0
    assert op1[key1] == 2.0
    assert op1[key2] == 4.0j

    op1 /= 2.0
    assert op1[key1] == 1.0
    assert op1[key2] == 2.0j

    # test unary -
    op_neg = -op
    assert op_neg[key1] == -1.0
    assert op_neg[key2] == -2.0j

    # test __eq__ and __ne__
    op2 = ManyBodyOperator(d)
    assert op == op2
    assert op != op_neg

    # test size() and len()
    assert op.size() == 2
    assert len(op) == 2

    # test erase
    op.erase(key1)
    assert key1 not in op
    assert op.size() == 1


def get_random_state(num_terms):
    s = ManyBodyState()
    for _ in range(num_terms):
        key = tuple(sorted(np.random.randint(0, 100, size=np.random.randint(1, 5))))
        val = np.random.rand() + 1j * np.random.rand()
        s.add_scaled(ManyBodyState({SlaterDeterminant(key): 1.0}), val)
    return s


def test_apply_multi():
    n_states = 3
    num_terms_op = 10
    num_terms_state = 10

    # Build random many-body operator
    op_dict = {}
    for _ in range(num_terms_op):
        num_c = np.random.randint(1, 3)
        num_a = np.random.randint(1, 3)
        k_c = tuple((int(np.random.randint(0, 50)), "c") for _ in range(num_c))
        k_a = tuple((int(np.random.randint(0, 50)), "a") for _ in range(num_a))
        op_dict[k_c + k_a] = np.random.rand() + 1j * np.random.rand()

    op = ManyBodyOperator(op_dict)

    # Create random states
    states = [get_random_state(num_terms_state) for _ in range(n_states)]

    # 1. Apply multi
    results_multi = op.apply_multi(states)

    # 2. Apply in loop
    results_loop = [op(s) for s in states]

    # Assert equality
    for r_multi, r_loop in zip(results_multi, results_loop):
        assert len(r_multi) == len(r_loop)
        for key in r_multi:
            assert key in r_loop
            np.testing.assert_allclose(r_multi[key][0], r_loop[key][0], atol=1e-12)


def _random_operator(n_orb, n_terms, seed):
    """A mix of density, one-body and general two-body terms, so `diagonal` meets every
    branch of the term loop -- the density fast path, the one-body hops it must skip, and
    the general strings it has to sign-evaluate."""
    rng = np.random.default_rng(seed)
    terms = {}
    while len(terms) < n_terms:
        kind = rng.integers(0, 3)
        if kind == 0:  # density n_i n_j
            i, j = rng.integers(0, n_orb, 2)
            key = ((int(i), "c"), (int(i), "a"), (int(j), "c"), (int(j), "a"))
        elif kind == 1:  # one-body hop
            i, j = rng.integers(0, n_orb, 2)
            key = ((int(i), "c"), (int(j), "a"))
        else:  # general two-body
            i, j, k, m = rng.integers(0, n_orb, 4)
            key = ((int(i), "c"), (int(j), "c"), (int(k), "a"), (int(m), "a"))
        terms[key] = complex(rng.standard_normal(), rng.standard_normal())
    return ManyBodyOperator(terms)


def _determinants(n_orb, n_dets, seed):
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    n_bytes = (n_orb + 7) // 8
    while len(out) < n_dets:
        b = bytes(rng.integers(0, 256, n_bytes).tolist())
        if b not in seen:
            seen.add(b)
            out.append(SlaterDeterminant.from_bytes(b))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_diagonal_matches_applying_to_each_determinant_alone(seed):
    """`diagonal` must return exactly what reading row D out of `op({D: 1})` returns.

    Not approximately: a term maps D to itself only when its created and annihilated
    multisets agree, which makes it diagonal for *every* determinant, so no off-diagonal
    term can ever contribute to that row. `diagonal` therefore accumulates the identical
    contributions in the identical term order, and the equality is bit-for-bit. Asserting
    `approx` here would let a reordering through, and CIPSI's candidate ranking is
    sensitive to exactly that.
    """
    op = _random_operator(n_orb=12, n_terms=60, seed=seed)
    dets = _determinants(n_orb=12, n_dets=40, seed=seed + 100)
    block = ManyBodyState(dict.fromkeys(dets, 1.0 + 0j), width=1)

    got = op.diagonal(block)
    assert got.shape == (len(block),)

    # `diagonal` reports in the block's ROW order, which is sorted, not insertion order.
    expected = []
    for key in block.keys():
        row = op(ManyBodyState({key: 1.0 + 0j}, width=1), 0).get(key)
        expected.append(0j if row is None else row[0])
    assert got.tobytes() == np.array(expected, dtype=complex).tobytes()


def test_diagonal_ignores_amplitudes_and_reads_only_the_keys():
    """Scaling the block must not move the answer: `<D|H|D>` is a property of `D`."""
    op = _random_operator(n_orb=10, n_terms=40, seed=7)
    dets = _determinants(n_orb=10, n_dets=25, seed=707)
    ones = ManyBodyState(dict.fromkeys(dets, 1.0 + 0j), width=1)
    scaled = ManyBodyState(dict.fromkeys(dets, 0.5 - 3j), width=1)
    assert op.diagonal(ones).tobytes() == op.diagonal(scaled).tobytes()


def test_diagonal_of_an_empty_block_is_an_empty_array():
    op = _random_operator(n_orb=8, n_terms=20, seed=3)
    out = op.diagonal(ManyBodyState(width=1))
    assert out.shape == (0,)
    assert out.dtype == complex


def test_diagonal_of_a_hermitian_operator_is_real():
    """A Hermitian `H` has real diagonal elements, which is why CIPSI takes `.real` of
    this without a tolerance check."""
    op = _random_operator(n_orb=10, n_terms=40, seed=11).hermitian_part()
    dets = _determinants(n_orb=10, n_dets=30, seed=1111)
    got = op.diagonal(ManyBodyState(dict.fromkeys(dets, 1.0 + 0j), width=1))
    assert np.max(np.abs(got.imag)) < 1e-14


# ---------------------------------------------------------------------------
# The threaded block apply
# ---------------------------------------------------------------------------


def _threaded_apply_active():
    """Whether this process will actually take the threaded branch.

    Both conditions are required and neither is the default: the extension must be built
    with ``IMPURITYMODEL_PARALLEL=1``, and ``OMP_NUM_THREADS`` must exceed 1 -- the thread
    cap defaults to 1 and the whole threaded block sits behind ``if (num_threads > 1)``.
    CI pairs them in the ThreadSanitizer job for exactly this reason. Reported rather than
    assumed, so a skip here is never mistaken for a pass.
    """
    import os

    from impurityModel.ed.ManyBodyUtils import parallel_apply_build

    return parallel_apply_build() and int(os.environ.get("OMP_NUM_THREADS", "1")) > 1


@pytest.mark.skipif(not _threaded_apply_active(), reason="needs IMPURITYMODEL_PARALLEL=1 and OMP_NUM_THREADS>1")
def test_threaded_block_apply_matches_a_row_by_row_reference():
    """The threaded merge must reproduce the serial image.

    The reference applies one row at a time: a single-row block takes ``num_threads == 1``
    and therefore the serial branch, so this compares the two code paths rather than the
    same path twice. The row count and width are chosen so more than one thread is actually
    spawned (``num_threads = min(cap, rows / max(1, 256 / p))``).

    A tolerance, not equality: the threaded merge sums duplicate contributions in bucket
    order, which is a different order from the serial accumulator's.
    """
    n_orb, n_rows, p = 24, 600, 4
    rng = np.random.default_rng(11)

    terms = {}
    for i in range(n_orb):
        for j in range(n_orb):
            terms[((i, "c"), (j, "a"))] = complex(rng.standard_normal(), rng.standard_normal())
    op = ManyBodyOperator(terms)

    keys = set()
    while len(keys) < n_rows:
        word = bytearray(8)
        for orb in rng.choice(n_orb, size=6, replace=False):
            word[orb // 8] |= 1 << (7 - orb % 8)
        keys.add(bytes(word))
    dets = [SlaterDeterminant.from_bytes(k) for k in sorted(keys)]
    amps = rng.standard_normal((n_rows, p)) + 1j * rng.standard_normal((n_rows, p))
    block = ManyBodyState.from_states(
        [ManyBodyState({d: complex(amps[r, c]) for r, d in enumerate(dets)}, width=1) for c in range(p)]
    )
    assert block.width == p

    got = op.apply_block(block, 0.0)

    expected = {}
    for r, d in enumerate(dets):
        one = ManyBodyState.from_states([ManyBodyState({d: complex(amps[r, c])}, width=1) for c in range(p)])
        image = op.apply_block(one, 0.0)
        for k, row in image.items():
            acc = expected.setdefault(k, np.zeros(p, dtype=complex))
            acc += np.asarray([row[c] for c in range(p)])

    assert len(got) == len(expected), f"threaded image has {len(got)} rows, reference has {len(expected)}"
    for k, row in got.items():
        ref = expected[k]
        np.testing.assert_allclose(np.asarray([row[c] for c in range(p)]), ref, rtol=1e-10, atol=1e-12)
