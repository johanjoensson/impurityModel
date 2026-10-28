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
from impurityModel.ed.gf_refinement import importance_scores, solve_shifted
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
    got = importance_scores(matrix, block, ZS, rtol=1e-11, first_order=first_order)
    s_ret, s_cand = _dense_scores(h, in_p, _global_seeds(matrix, block, index, comm), ZS, first_order)
    pkeys = matrix.p_keys()
    kkeys = matrix.keys_of(np.arange(matrix.n_k_local))
    want_ret = s_ret[[index[_key(k)] for k in pkeys]]
    want_cand = s_cand[[index[_key(k)] for k in kkeys]]
    np.testing.assert_allclose(got.s_ret, want_ret, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(got.s_cand, want_cand, rtol=1e-6, atol=1e-12)
    assert max(got.residuals) < 1e-10
    # The premise: there are discovered rows with a score, and the self-energy changes the ranking's values.
    n_cand = comm.allreduce(int(np.isfinite(got.s_cand).sum()))
    assert n_cand > 0


@pytest.mark.mpi
def test_the_self_energy_term_matters():
    comm = MPI.COMM_WORLD
    dets, _op, _b, matrix, _h, index, _in_p = _setup(comm)
    block = _seeds(matrix, index, dets)
    a = importance_scores(matrix, block, ZS, rtol=1e-11)
    b = importance_scores(matrix, block, ZS, rtol=1e-11, first_order=False)
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
def test_the_solver_agrees_with_a_direct_solve_and_leaves_on_one_iteration_on_every_rank():
    comm = MPI.COMM_WORLD
    rng = np.random.default_rng(3)
    n = 40
    a = rng.normal(size=(n, n))
    a = (a + a.T) / 2
    comm.Bcast(a, root=0)
    lo, hi = comm.rank * n // comm.size, (comm.rank + 1) * n // comm.size
    z = 0.2 + 0.5j
    b = np.arange(n, dtype=complex) + 1

    def apply(v):
        full = np.concatenate(comm.allgather(v))
        return a[lo:hi] @ full

    x, it, rel = solve_shifted(apply, z, b[lo:hi], comm, rtol=1e-12)
    want = np.linalg.solve(z * np.eye(n) - a, b)
    np.testing.assert_allclose(x, want[lo:hi], atol=1e-9)
    assert len(set(comm.allgather(it))) == 1 and rel < 1e-11


# -- the seam: gf_engine._block_green_group calls the probe on a capped unit that kept its matrix ---------------


def _unit(comm, cap, probe):
    from impurityModel.ed.gf_engine import _block_green_group
    from impurityModel.test.gf.test_gf_truncation import _redistribute_as_width1
    from impurityModel.test.support.gf_chain_fixtures import DELTA, chain_operator, chain_seeds
    from impurityModel.test.support.gf_chain_fixtures import N_ORB as N14

    basis = _basis(sorted({d for s in chain_seeds() for d in s}), comm, n_orb=N14)
    basis.truncation_threshold = cap
    full = chain_seeds()
    seeds = _redistribute_as_width1(basis, full if comm is None or comm.rank == 0 else None, len(full))
    return _block_green_group(
        basis,
        chain_operator(),
        seeds,
        False,
        DELTA,
        0.0,
        True,
        False,
        None,
        None,
        support_settle=False,
        unit_probe=probe,
    )


@pytest.mark.mpi
def test_the_probe_runs_on_a_capped_unit_with_its_matrix():
    comm = MPI.COMM_WORLD
    calls = []

    def probe(matrix, support, seeds, hop, basis, info):
        calls.append((matrix, support, seeds, basis))
        return {"called": True, "n_p": int(matrix.n_p_local)}

    *_, cap_stats = _unit(comm, 900, probe)
    assert cap_stats["cap_hit"] and len(calls) == 1
    assert isinstance(calls[0][0], DistributedSparseMatrix) and cap_stats["probe"]["called"]
    assert calls[0][1].retained_size == cap_stats["retained_size"]


@pytest.mark.mpi
@pytest.mark.parametrize("cap", [10**9])
def test_the_probe_is_not_called_on_an_uncapped_unit_or_without_one(cap):
    comm = MPI.COMM_WORLD
    calls = []
    *_, uncapped = _unit(comm, cap, lambda *a: calls.append(a))
    assert not uncapped["cap_hit"] and not calls and "probe" not in uncapped
    *_, none = _unit(comm, 900, None)
    assert "probe" not in none


@pytest.mark.mpi
def test_probe_unit_scores_a_real_capped_unit():
    comm = MPI.COMM_WORLD
    *_, cap_stats = _unit(comm, 900, _probe_with_zs)
    record = cap_stats["probe"]
    assert record["residual"] < 1e-5
    assert record["top_s_cand"] > 0 and np.isfinite(record["low_s_ret"])


def _probe_with_zs(matrix, support, seeds, hop, basis, info):
    from impurityModel.ed.gf_refinement import probe_unit

    return probe_unit(matrix, support, seeds, hop, basis, info, zs=(0.5 + 0.1j, -1.0 + 0.1j))


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
@pytest.mark.parametrize("x0", [False, True])
def test_the_block_solver_equals_a_direct_solve_per_column_and_leaves_together_on_every_rank(x0):
    from impurityModel.ed.gf_refinement import solve_shifted_block

    comm = MPI.COMM_WORLD
    rng = np.random.default_rng(4)
    n = 45
    a = rng.normal(size=(n, n))
    a = (a + a.T) / 2
    b = rng.normal(size=(n, 4)) + 0j
    b[:, 2] = 0.0  # a zero column stays zero and does not stall the others
    comm.Bcast(a, root=0)
    comm.Bcast(b, root=0)
    lo, hi = comm.rank * n // comm.size, (comm.rank + 1) * n // comm.size
    z = 0.2 + 0.5j
    guess = (rng.normal(size=(n, 4)) * 0.1 + 0j) if x0 else None
    if x0:
        comm.Bcast(guess, root=0)

    def apply(v):
        full = np.zeros((n, v.shape[1]), dtype=complex)
        full[lo:hi] = v
        if comm.size > 1:
            comm.Allreduce(MPI.IN_PLACE, full, op=MPI.SUM)
        return a[lo:hi] @ full

    x, it, rel = solve_shifted_block(apply, z, b[lo:hi], comm, x0=None if guess is None else guess[lo:hi], rtol=1e-12)
    want = np.linalg.solve(z * np.eye(n) - a, b)
    got = np.zeros_like(want)
    got[lo:hi] = x
    if comm.size > 1:
        comm.Allreduce(MPI.IN_PLACE, got, op=MPI.SUM)
    np.testing.assert_allclose(got, want, atol=1e-9)
    assert rel.max() < 1e-10 and rel[2] == 0.0
    assert comm.allreduce(it, op=MPI.MAX) == comm.allreduce(it, op=MPI.MIN)
