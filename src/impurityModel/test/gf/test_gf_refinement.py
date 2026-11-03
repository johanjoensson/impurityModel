"""``gf_refinement``: the importance probe's scores against a dense oracle that shares no code with it.

The oracle builds the whole operator on the 12-orbital half-filled space by applying it to every determinant in
Python, restricts it to ``P`` and evaluates the module's definitions with dense linear algebra:
``s_ret(D) = max_j |X_j(D)|^2 |z - H_DD - Sigma_D|`` on ``P`` and ``s_cand(D) = max_j |(H_FP X)_j(D)|^2 /
|z - H_DD - Sigma_D|`` on the discovered rows. The removal identity ``dG_ij = -X_i(D) X_j(D) / R_DD`` that the
score is built on is checked on its own against a direct re-inversion.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.DistributedSparseMatrix import DistributedSparseMatrix
from impurityModel.ed.gf_refinement import (
    ScoringPlan,
    _round_worse,
    g0_check_shifts,
    importance_scores,
    sigma_error,
)
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, applyOp
from impurityModel.test.basis.test_sparse_matrix_build import N_ORB, _all_dets, _basis

_N_BYTES = 2
ZS = (0.4 + 0.15j, -1.1 + 0.3j)


def _real_operator(seed=5):
    rng = np.random.default_rng(seed)
    terms = {((o, "c"), (o, "a")): float(rng.normal()) for o in range(N_ORB)}
    for a in range(N_ORB):
        for b in range(a + 1, N_ORB):
            if rng.random() < 0.3:
                v = float(rng.normal()) * 0.4
                terms[((a, "c"), (b, "a"))] = v
                terms[((b, "c"), (a, "a"))] = v
    for a in range(4):
        terms[((a, "c"), (a + 4, "c"), (a + 4, "a"), (a, "a"))] = 2.0
    return ManyBodyOperator(terms)


def _key(det):
    return bytes(det.to_bytearray()[:_N_BYTES])


def _dense_operator(op, dets):
    index = {_key(d): i for i, d in enumerate(dets)}
    h = np.zeros((len(dets), len(dets)), dtype=complex)
    for col, det in enumerate(dets):
        image = applyOp(op, ManyBodyState({det: 1.0 + 0j}), cutoff=0)
        for state, amp in zip(image.keys(), np.asarray(image)[:, 0]):
            row = index.get(_key(state))
            if row is not None:
                h[row, col] += amp
    return h, index


def _setup(comm, drop=False):
    dets = _all_dets()
    op = _real_operator()
    basis = _basis(dets[::3], comm)
    matrix = DistributedSparseMatrix(op, (basis.n_bytes + 7) // 8, comm if comm.size > 1 else None)
    matrix.grow(list(basis.local_basis))
    h, index = _dense_operator(op, dets)
    in_p = np.zeros(len(dets), dtype=bool)
    in_p[[index[_key(d)] for d in dets[::3]]] = True
    if drop:
        # Dead slots: every retained determinant but the seeds' whose index is odd leaves P (its row stays in K).
        pkeys = matrix.p_keys()
        gone = [i for i, k in enumerate(pkeys) if index[_key(k)] % 2 == 1 and index[_key(k)] not in (0, 90)]
        matrix.drop_columns(gone)
        for part in comm.allgather([index[_key(pkeys[i])] for i in gone]):
            in_p[part] = False
    return dets, op, basis, matrix, h, index, in_p


def _seeds(matrix, index, dets):
    """Two seed columns: the first and the 31st retained determinant (a rank owns the one it owns)."""
    block = np.zeros((matrix.n_p_local, 2), dtype=complex)
    k_to_p = matrix.k_to_p_slot()
    for j, det in enumerate((dets[0], dets[90])):
        slot = matrix.key_slots([det])
        if slot[0] >= 0 and k_to_p[slot[0]] >= 0:
            block[k_to_p[slot[0]], j] = 1.0
    return block


def _dense_scores(h, in_p, seeds_global, zs, first_order=True):
    n = len(in_p)
    p = np.flatnonzero(in_p)
    hpp = h[np.ix_(p, p)]
    s_ret = np.full(n, -np.inf)
    s_cand = np.full(n, -np.inf)
    diag = np.diag(h)
    for z in zs:
        x = np.linalg.solve(z * np.eye(len(p)) - hpp, seeds_global[p])
        inv = 1.0 / (z - diag[p])
        sigma = (np.abs(h[:, p]) ** 2) @ inv
        sigma[p] -= np.abs(diag[p]) ** 2 * inv
        if not first_order:
            sigma = 0 * sigma
        denom = np.abs(z - diag - sigma)
        flow = h[:, p] @ x
        cand = (np.abs(flow) ** 2).max(axis=1) / denom
        ret = np.full(n, -np.inf)
        ret[p] = (np.abs(x) ** 2).max(axis=1) * denom[p]
        s_ret = np.maximum(s_ret, ret)
        s_cand = np.maximum(s_cand, np.where(in_p, -np.inf, cand))
    return s_ret, s_cand


def _global_seeds(matrix, block, index, comm):
    """The seed block over the whole space, gathered from the ranks' P-slot blocks (test-side only)."""
    full = np.zeros((len(index), block.shape[1]), dtype=complex)
    keys = matrix.p_keys()
    for slot, key in enumerate(keys):
        full[index[_key(key)]] += block[slot]
    if comm.size > 1:
        comm.Allreduce(MPI.IN_PLACE, full, op=MPI.SUM)
    return full


def _dense_solutions(matrix, h, in_p, index, block, zs, conj=False):
    """``X_z = (z - H_PP)^-1 s`` per shift from the dense operator, on this rank's P slots (``conj``: at ``z*``)."""
    comm = matrix.comm if matrix.comm is not None else MPI.COMM_WORLD
    p = np.flatnonzero(in_p)
    seeds_g = _global_seeds(matrix, block, index, comm)
    pos = {int(g): i for i, g in enumerate(p)}
    rows = [pos.get(index[_key(k)], 0) for k in matrix.p_keys()]  # a dead slot reads row 0: zeroed by `alive`
    out = []
    for z in zs:
        zz = np.conj(z) if conj else z
        x = np.linalg.solve(zz * np.eye(len(p)) - h[np.ix_(p, p)], seeds_g[p])
        out.append(x[rows])
    return out


@pytest.mark.mpi
def test_abs2_matvec_is_the_entrywise_square():
    comm = MPI.COMM_WORLD
    _dets, _op, _b, matrix, h, index, _in_p = _setup(comm)
    rng = np.random.default_rng(2)
    x = rng.normal(size=(matrix.n_p_local, 3)) + 0j
    got = matrix.matvec(x, rows="all", squared=True)
    plain = matrix.matvec(x, rows="all")
    # Against the dense |H|^2 on the K rows this rank owns.
    kkeys = matrix.keys_of(np.arange(matrix.n_k_local))
    pkeys = matrix.p_keys()
    xg = np.zeros((len(index), 3), dtype=complex)
    for slot, key in enumerate(pkeys):
        xg[index[_key(key)]] += x[slot]
    if comm.size > 1:
        comm.Allreduce(MPI.IN_PLACE, xg, op=MPI.SUM)
    want = (np.abs(h) ** 2) @ xg
    rows = [index[_key(k)] for k in kkeys]
    np.testing.assert_allclose(got, want[rows], atol=1e-12)
    assert not np.allclose(got, plain)


@pytest.mark.mpi
@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("first_order", [True, False])
def test_the_scores_equal_the_dense_definitions(first_order, drop):
    comm = MPI.COMM_WORLD
    dets, _op, _b, matrix, h, index, in_p = _setup(comm, drop)
    block = _seeds(matrix, index, dets)
    xs = _dense_solutions(matrix, h, in_p, index, block, ZS)
    got = importance_scores(matrix, block, ZS, xs, first_order=first_order)
    s_ret, s_cand = _dense_scores(h, in_p, _global_seeds(matrix, block, index, comm), ZS, first_order)
    pkeys = matrix.p_keys()
    kkeys = matrix.keys_of(np.arange(matrix.n_k_local))
    want_ret = s_ret[[index[_key(k)] for k in pkeys]]
    want_cand = s_cand[[index[_key(k)] for k in kkeys]]
    np.testing.assert_allclose(got.s_ret, want_ret, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(got.s_cand, want_cand, rtol=1e-6, atol=1e-12)
    # The premise: there are discovered rows with a score, and the self-energy changes the ranking's values.
    n_cand = comm.allreduce(int(np.isfinite(got.s_cand).sum()))
    assert n_cand > 0


@pytest.mark.mpi
def test_the_self_energy_term_matters():
    comm = MPI.COMM_WORLD
    dets, _op, _b, matrix, h, index, in_p = _setup(comm)
    block = _seeds(matrix, index, dets)
    xs = _dense_solutions(matrix, h, in_p, index, block, ZS)
    a = importance_scores(matrix, block, ZS, xs)
    b = importance_scores(matrix, block, ZS, xs, first_order=False)
    finite = np.isfinite(a.s_ret)
    differs = comm.allreduce(int(not np.allclose(a.s_ret[finite], b.s_ret[finite], rtol=1e-3)), op=MPI.SUM)
    assert differs > 0


def test_the_removal_identity_is_exact_for_a_real_hamiltonian():
    """dG_ij = -X_i(D) X_j(D) / R_DD for a symmetric real H and a complex z: the score's premise."""
    rng = np.random.default_rng(1)
    n = 30
    h = rng.normal(size=(n, n))
    h = (h + h.T) / 2
    z = 0.3 + 0.2j
    s = rng.normal(size=(n, 2))
    r = np.linalg.inv(z * np.eye(n) - h)
    x = r @ s
    g = s.T @ x
    for d in (0, 7, 19):
        keep = [i for i in range(n) if i != d]
        r2 = np.linalg.inv(z * np.eye(n - 1) - h[np.ix_(keep, keep)])
        g2 = s[keep].T @ r2 @ s[keep]
        np.testing.assert_allclose(g2 - g, -np.outer(x[d], x[d]) / r[d, d], atol=1e-12)


@pytest.mark.mpi
@pytest.mark.parametrize("with_extra", [False, True])
def test_the_exempt_rows_are_the_seeds_the_heavy_first_shell_and_last_rounds_admissions(with_extra):
    from impurityModel.ed.gf_refinement import FIRST_SHELL_EXEMPT_TAIL, _exempt_pslots

    comm = MPI.COMM_WORLD
    dets, _op, _b, matrix, h, index, in_p = _setup(comm)
    block = _seeds(matrix, index, dets)
    seed_keys = [k for k, row in zip(matrix.p_keys(), block) if np.abs(row).max() > 0]
    extra = [dets[30]] if with_extra else []
    own = {_key(k) for k in matrix.p_keys()}
    extra = [k for k in extra if _key(k) in own]
    got = _exempt_pslots(matrix, seed_keys, block, extra)
    # Dense oracle on the global P ordering.
    p = np.flatnonzero(in_p)
    seeds_g = _global_seeds(matrix, block, index, comm)
    flow = h[np.ix_(p, p)] @ seeds_g[p]
    weight = (np.abs(flow) ** 2).max(axis=1)
    is_seed = np.abs(seeds_g[p]).max(axis=1) > 0
    weight[is_seed] = 0.0
    order = np.argsort(weight)
    tail = np.cumsum(weight[order])
    cutoff = weight[order][np.searchsorted(tail, FIRST_SHELL_EXEMPT_TAIL * weight.sum(), side="right") - 1]
    heavy = {int(p[i]) for i in range(len(p)) if weight[i] > cutoff}
    want = {index[_key(k)] for k in seed_keys} | heavy | {index[_key(k)] for k in extra}
    mine = {index[_key(k)] for k, e in zip(matrix.p_keys(), got) if e}
    union = set().union(*comm.allgather(mine))
    assert union == set().union(*comm.allgather(want))
    assert any(e for e in got) or comm.size > 1


@pytest.mark.mpi
def test_the_retire_ranking_resolves_scores_many_decades_below_the_largest():
    """The ``k`` retired rows are exactly the ``k`` lowest ``s_ret`` even when most scores sit 1e-10 and more below
    the largest (selections bin scores to 32 mantissa bits: a ranking by ``1 / (1 + s / s_max)`` would put them
    all in one bin and retire by key order)."""
    from impurityModel.ed.gf_refinement import ImportanceScores, _select_swap

    comm = MPI.COMM_WORLD
    _dets, _op, _basis, matrix, _h, _index, _in_p = _setup(comm)
    n_p = matrix.n_p_local
    # distinct scores over 40 decades, spread over the ranks in an order unrelated to the keys
    ranks_n = np.array(comm.allgather(n_p))
    total = int(ranks_n.sum())
    first = int(ranks_n[: comm.rank].sum())
    order = np.random.default_rng(3).permutation(total)[first : first + n_p]
    s_ret = 10.0 ** (-40.0 * order / max(total - 1, 1))
    s_cand = np.full(matrix.n_k_local, 1.0)
    s_cand[matrix.p_to_k_slot()] = -np.inf  # every candidate beats every retained row
    scores = ImportanceScores(s_ret, s_cand)
    k = 7
    retire, _admit, _gain = _select_swap(matrix, scores, np.zeros(n_p, dtype=bool), k)
    got = np.sort(np.concatenate(comm.allgather(s_ret[retire])))
    want = np.sort(np.concatenate(comm.allgather(s_ret)))[:k]
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("hermitian_complex", [False, True])
def test_the_removal_identity_for_complex_seeds_needs_the_conjugate_shift_solve(hermitian_complex):
    """``dG_ij = -conj(Y_i(D)) X_j(D) / R_DD`` with ``X = R(z) s`` and ``Y = R(z*) s``, for complex seeds and a real or
    complex Hermitian ``H``; the real-seed form ``X_i X_j`` is not it."""
    rng = np.random.default_rng(2)
    n = 30
    h = rng.normal(size=(n, n)) + (1j * rng.normal(size=(n, n)) if hermitian_complex else 0)
    h = (h + h.conj().T) / 2
    z = 0.3 + 0.2j
    s = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
    r = np.linalg.inv(z * np.eye(n) - h)
    x = r @ s
    y = np.linalg.inv(np.conj(z) * np.eye(n) - h) @ s
    g = s.conj().T @ x
    for d in (0, 7, 19):
        keep = [i for i in range(n) if i != d]
        r2 = np.linalg.inv(z * np.eye(n - 1) - h[np.ix_(keep, keep)])
        g2 = s[keep].conj().T @ r2 @ s[keep]
        np.testing.assert_allclose(g2 - g, -np.outer(y[d].conj(), x[d]) / r[d, d], atol=1e-11)
        assert not np.allclose(g2 - g, -np.outer(x[d].conj(), x[d]) / r[d, d], atol=1e-6)


def _dense_scores_hermitian(h, in_p, seeds_global, zs):
    """``max_i |Y_i| max_j |X_j|`` form of the scores (``Y = R(z*) s``), dense."""
    n = len(in_p)
    p = np.flatnonzero(in_p)
    hpp = h[np.ix_(p, p)]
    diag = np.diag(h)
    s_ret = np.full(n, -np.inf)
    s_cand = np.full(n, -np.inf)
    for z in zs:
        x = np.linalg.solve(z * np.eye(len(p)) - hpp, seeds_global[p])
        y = np.linalg.solve(np.conj(z) * np.eye(len(p)) - hpp, seeds_global[p])
        inv = 1.0 / (z - diag[p])
        sigma = (np.abs(h[:, p]) ** 2) @ inv
        sigma[p] -= np.abs(diag[p]) ** 2 * inv
        denom = np.abs(z - diag - sigma)
        cand = np.abs(h[:, p] @ x).max(axis=1) * np.abs(h[:, p] @ y).max(axis=1) / denom
        ret = np.full(n, -np.inf)
        ret[p] = np.abs(x).max(axis=1) * np.abs(y).max(axis=1) * denom[p]
        s_ret = np.maximum(s_ret, ret)
        s_cand = np.maximum(s_cand, np.where(in_p, -np.inf, cand))
    return s_ret, s_cand


@pytest.mark.mpi
def test_complex_seeds_are_scored_with_the_conjugate_shift_solve():
    comm = MPI.COMM_WORLD
    dets, _op, _b, matrix, h, index, in_p = _setup(comm)
    block = _seeds(matrix, index, dets)
    # a relative phase between determinants of one column: not real up to a global phase
    rng = np.random.default_rng(4)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=len(index)))
    pkeys = matrix.p_keys()
    block = block * np.array([phases[index[_key(k)]] for k in pkeys])[:, None]
    block[:, 1] += 0.3 * block[:, 0]  # a genuinely mixed second column
    xs = _dense_solutions(matrix, h, in_p, index, block, ZS)
    ys = _dense_solutions(matrix, h, in_p, index, block, ZS, conj=True)
    got = importance_scores(matrix, block, ZS, xs, ys)
    s_ret, s_cand = _dense_scores_hermitian(h, in_p, _global_seeds(matrix, block, index, comm), ZS)
    kkeys = matrix.keys_of(np.arange(matrix.n_k_local))
    np.testing.assert_allclose(got.s_ret, s_ret[[index[_key(k)] for k in pkeys]], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(got.s_cand, s_cand[[index[_key(k)] for k in kkeys]], rtol=1e-6, atol=1e-12)
    # Without the conjugate solutions the complex seeds' scores are the real-seed form, which is not the same.
    wrong = importance_scores(matrix, block, ZS, xs)
    finite = np.isfinite(got.s_ret)
    assert comm.allreduce(int(not np.allclose(wrong.s_ret[finite], got.s_ret[finite], rtol=1e-3)), op=MPI.SUM) > 0


@pytest.mark.mpi
def test_the_weights_scale_the_scores_and_pick_the_shift_that_counts():
    comm = MPI.COMM_WORLD
    dets, _op, _b, matrix, h, index, in_p = _setup(comm)
    block = _seeds(matrix, index, dets)
    xs = _dense_solutions(matrix, h, in_p, index, block, ZS)
    plain = importance_scores(matrix, block, ZS, xs)
    finite = np.isfinite(plain.s_ret)
    half = importance_scores(matrix, block, ZS, xs, weights=[0.5, 0.5])
    np.testing.assert_allclose(half.s_ret[finite], 0.5 * plain.s_ret[finite], rtol=1e-12)
    # A weight picks the shift that counts: zero on the second leaves the first shift's scores alone.
    first_only = importance_scores(matrix, block, ZS[:1], xs[:1])
    masked = importance_scores(matrix, block, ZS, xs, weights=[1.0, 0.0])
    np.testing.assert_allclose(masked.s_ret[finite], first_only.s_ret[finite], rtol=1e-12)


def _dense_dg2(h, in_p, seeds_global, z):
    """The second-order frontier correction ``dG_ij = sum_f conj((H Y_i)_f) (H X_j)_f / (z - H_ff - Sigma_f)``."""
    p, f = np.flatnonzero(in_p), np.flatnonzero(~in_p)
    hpp, hfp = h[np.ix_(p, p)], h[np.ix_(f, p)]
    x = np.linalg.solve(z * np.eye(len(p)) - hpp, seeds_global[p])
    y = np.linalg.solve(np.conj(z) * np.eye(len(p)) - hpp, seeds_global[p])
    sigma = (np.abs(hfp) ** 2) @ (1.0 / (z - np.diag(h)[p]))
    return np.einsum("fi,fj,f->ij", (hfp @ y).conj(), hfp @ x, 1.0 / (z - np.diag(h)[f] - sigma))


@pytest.mark.mpi
@pytest.mark.parametrize("drop", [False, True])
def test_the_sigma_error_equals_its_dense_definition(drop):
    comm = MPI.COMM_WORLD
    dets, _op, _b, matrix, h, index, in_p = _setup(comm, drop)
    block = _seeds(matrix, index, dets)
    seeds_g = _global_seeds(matrix, block, index, comm)
    xs = _dense_solutions(matrix, h, in_p, index, block, ZS)
    ys = _dense_solutions(matrix, h, in_p, index, block, ZS, conj=True)
    weights = [3.0, 0.5]
    want = [w * np.abs(_dense_dg2(h, in_p, seeds_g, z)).max() for z, w in zip(ZS, weights)]
    np.testing.assert_allclose(sigma_error(matrix, ZS, xs, ys, weights), want, rtol=1e-8)
    # Real H and real seeds: Y = conj(X), the conjugate solve is not needed.
    np.testing.assert_allclose(sigma_error(matrix, ZS, xs, None, weights), want, rtol=1e-8)


def test_the_second_order_correction_estimates_what_the_frontier_adds_to_g():
    """``dG`` against the exact change of ``G`` when the coupled rows join P: within 30 % at these shifts, where the
    bare denominator ``z - H_ff`` is off by 1.5-26x. The estimate is perturbative: next to a pole at small broadening
    (``2 + 0.05i`` on this fixture) it is not (measured 4.6x off)."""
    comm = MPI.COMM_SELF
    dets, _op, _b, matrix, h, index, in_p = _setup(comm)
    seeds_g = _global_seeds(matrix, _seeds(matrix, index, dets), index, comm)
    p = np.flatnonzero(in_p)
    f = np.flatnonzero(~in_p)
    u = np.concatenate([p, f[np.abs(h[np.ix_(f, p)]).sum(axis=1) > 0]])

    def g_on(rows, z):
        return seeds_g[rows].conj().T @ np.linalg.solve(z * np.eye(len(rows)) - h[np.ix_(rows, rows)], seeds_g[rows])

    for z in ZS + (0.5j,):
        true = g_on(u, z) - g_on(p, z)
        err = np.abs(_dense_dg2(h, in_p, seeds_g, z) - true).max() / np.abs(true).max()
        assert err < 0.3, (z, err)


def test_a_round_is_judged_on_the_largest_error_of_each_axis():
    """Review F4/R1, the M1 65k pattern: the real-axis estimate grows 3.5x while the Matsubara one falls 4.7x from a
    larger value. A sum over the shifts falls (the rule that accepted every M1 round); per axis the round is
    rejected."""
    axes = [0, 0, 0, 1, 1]
    before = np.array([2.2, 0.4, 0.1, 7.0, 1.0])
    after = np.array([7.8, 0.5, 0.1, 1.5, 0.3])
    assert after.sum() < before.sum()
    assert _round_worse(before, after, axes)
    # SMO 40k round 1: a few points get worse while each axis's worst point improves -- accepted, where a test per
    # shift rejects.
    after = np.array([1.5, 0.45, 0.12, 5.0, 1.1])
    assert (after > before).any()
    assert not _round_worse(before, after, axes)
    assert not _round_worse(before, before * 1.1, axes)  # within the slack
    assert _round_worse(before, np.where(axes == np.int64(1), np.nan, before), axes)  # a broken estimate rejects


def test_the_check_shifts_are_mesh_points_at_the_broadening_and_the_lowest_matsubara_points():
    def g0_inv(zs):
        return (np.asarray(zs)[:, None, None] - 0.3) * np.eye(2)[None]

    w = np.linspace(-2.0, 2.0, 101)
    iw = 1j * np.pi * (2 * np.arange(10) + 1) / 20.0
    got = g0_check_shifts(g0_inv, w[::-1], iw[::-1], 0.05, n_real=5, n_matsubara=3)
    zs = [z for z, _w, _a in got]
    assert [a for _z, _w, a in got] == [0] * 5 + [1] * 3
    np.testing.assert_allclose(zs[:5], np.linspace(-2.0, 2.0, 5) + 0.05j)
    np.testing.assert_allclose(zs[5:], iw[:3])
    np.testing.assert_allclose([wt for _z, wt, _a in got], np.abs(np.asarray(zs) - 0.3) ** 2)


@pytest.mark.parametrize("need_y", [False, True])
def test_a_check_shift_that_is_also_a_scoring_shift_is_solved_once(need_y):
    zs = (0.1 + 0.2j, 0.5j)
    plan = ScoringPlan(zs, (1.0, 1.0), need_y, 1e-3, checks=(0.3 + 0.2j, 0.5j, 0.7 + 0.2j), check_axes=(0, 1, 0))
    shifts = plan.accumulator_shifts
    assert len(shifts) == (4 if not need_y else 8)  # 0.5j is shared
    solutions = [np.array([[z]]) for z in shifts]  # each "solution" carries its own shift
    xs, ys = plan.split(solutions)
    assert [complex(x[0, 0]) for x in xs] == list(zs)
    cx, cy = plan.check_solutions(solutions)
    assert [complex(x[0, 0]) for x in cx] == list(plan.checks)
    if need_y:
        assert [complex(y[0, 0]) for y in ys] == [np.conj(z) for z in zs]
        assert [complex(y[0, 0]) for y in cy] == [np.conj(z) for z in plan.checks]
    else:
        assert ys is None and cy is None
