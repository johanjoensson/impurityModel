"""RIXS correctness backbone (T0 for the RIXS tensor refactor).

:func:`spectra.getRIXSmap_new` implements the Kramers-Heisenberg map

    A_{ij}(w_in, w_loss) = sum_g (weight_g / Z)
        <g| Tin_i^dagger R1(w_in) Tout_j^dagger R2(w_loss) Tout_j R1(w_in) Tin_i |g>,

with R1 = (w_in + i d1 + E_g - H)^-1 and R2 = (w_loss + i d2 + E_g - H)^-1.

These tests pin that behaviour with an independent dense reference (so the in-progress
tensor refactor is checked against physics, not just against itself) and verify that the
map's component sum is invariant under a single-particle basis rotation.
"""

from itertools import combinations

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import spectra
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp, inner
from impurityModel.ed.symmetries import impurity_symmetry_rotation, rotate_hamiltonian

# orbitals 0,1 = impurity "3d" (block key 2); orbital 2 = core "2p" (block key 1)
N_ORB = 3
TAU = 0.02
D1, D2 = 0.4, 0.2
WIN = np.array([-7.0, -6.5])
WLOSS = np.linspace(-1.0, 3.0, 9)


def _model():
    ei, ec, t, u = 0.5, -8.0, 0.3, 2.0
    return ManyBodyOperator(
        {
            ((0, "c"), (0, "a")): ei,
            ((1, "c"), (1, "a")): ei + 0.2,
            ((2, "c"), (2, "a")): ec,
            ((0, "c"), (1, "a")): t,
            ((1, "c"), (0, "a")): t,
            ((0, "c"), (1, "c"), (1, "a"), (0, "a")): u,
        }
    )


def _bytes(occ):
    b = bytearray(1)
    for o in occ:
        b[0] |= 1 << (7 - o)
    return bytes(b)


def _dets(ne):
    return [_bytes(o) for o in combinations(range(N_ORB), ne)]


def _states(dets):
    return [ManyBodyState({SlaterDeterminant.from_bytes(d): 1.0}) for d in dets]


def _matrix(op, states):
    n = len(states)
    m = np.zeros((n, n), dtype=complex)
    for j, sj in enumerate(states):
        col = applyOp(op, sj)
        for i, si in enumerate(states):
            m[i, j] = inner(si, col)
    return m


def _thermal_states(op, ne):
    dets = _dets(ne)
    states = _states(dets)
    h = _matrix(op, states)
    ev, vec = np.linalg.eigh(h)
    psis = [
        ManyBodyState(
            {SlaterDeterminant.from_bytes(dets[i]): vec[i, k] for i in range(len(dets)) if abs(vec[i, k]) > 1e-14}
        )
        for k in range(len(ev))
    ]
    return psis, list(ev), dets, states, vec


def _basis(dets):
    return Basis(
        impurity_orbitals={2: [[0, 1]], 1: [[2]]},
        bath_states=({2: [[]], 1: [[]]}, {2: [[]], 1: [[]]}),
        initial_basis=list(dets),
        verbose=False,
        comm=MPI.COMM_SELF,
    )


def _dense_rixs(op, ne, tin, tout, es, vecs, states):
    """Independent dense Kramers-Heisenberg map, thermally averaged."""
    n = len(states)
    eye = np.eye(n, dtype=complex)
    H = _matrix(op, states)
    Tin = [_matrix(t, states) for t in tin]
    Tout = [_matrix(t, states) for t in tout]
    e0 = min(es)
    Z = float(np.sum(np.exp(-(np.asarray(es) - e0) / TAU)))
    out = np.zeros((len(tin), len(tout), len(WIN), len(WLOSS)), dtype=complex)
    for g, eg in enumerate(es):
        gvec = vecs[:, g]
        wg = np.exp(-(eg - e0) / TAU)
        for i in range(len(tin)):
            for ki, win in enumerate(WIN):
                psi2 = np.linalg.solve((win + 1j * D1 + eg) * eye - H, Tin[i] @ gvec)
                for j in range(len(tout)):
                    psi3 = Tout[j] @ psi2
                    for kl, wl in enumerate(WLOSS):
                        r2 = np.linalg.solve((wl + 1j * D2 + eg) * eye - H, psi3)
                        out[i, j, ki, kl] += wg * (psi3.conj() @ r2)
    return out / Z


def _tin_tout():
    tin = [ManyBodyOperator({((0, "c"), (2, "a")): 1.0}), ManyBodyOperator({((1, "c"), (2, "a")): 1.0})]
    tout = [ManyBodyOperator({((2, "c"), (0, "a")): 1.0}), ManyBodyOperator({((2, "c"), (1, "a")): 1.0})]
    return tin, tout


def _run_rixs(op, psis, es, tin, tout, dets):
    return spectra.getRIXSmap_new(
        op,
        tin,
        tout,
        psis,
        es,
        tau=TAU,
        wIns=WIN,
        wLoss=WLOSS,
        delta1=D1,
        delta2=D2,
        basis=_basis(dets),
        verbose=False,
        slaterWeightMin=0.0,
    )


def test_rixs_matches_dense_reference():
    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    tin, tout = _tin_tout()
    got = _run_rixs(op, psis, es, tin, tout, dets)
    ref = _dense_rixs(op, 2, tin, tout, es, vecs, states)
    np.testing.assert_allclose(got, ref, atol=1e-9)


def test_rixs_component_sum_is_rotation_invariant():
    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    tin, tout = _tin_tout()
    base = _run_rixs(op, psis, es, tin, tout, dets).sum(axis=(0, 1))

    # Rotate the impurity block (0,1); rotate H, the operators and re-solve the states.
    W, _ = impurity_symmetry_rotation(op, [0, 1], n_orb=N_ORB)
    op_rot = rotate_hamiltonian(op, W)
    tin_rot = [ManyBodyOperator(spectra._rotate_op_dict(t.to_dict(), W)) for t in tin]
    tout_rot = [ManyBodyOperator(spectra._rotate_op_dict(t.to_dict(), W)) for t in tout]
    psis_rot, es_rot, dets_rot, _, _ = _thermal_states(op_rot, 2)
    rot = _run_rixs(op_rot, psis_rot, es_rot, tin_rot, tout_rot, dets_rot).sum(axis=(0, 1))

    np.testing.assert_allclose(base, rot, atol=1e-8)


# --- tests for the full rank-4 polarization tensor (getRIXSmap_tensor) ---

# In/out polarization vectors (length = #components = 2), including non-axis and circular ones.
EPS_IN = [[1.0, 0.0], [0.0, 1.0], [1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 1j / np.sqrt(2)]]
EPS_OUT = [[1.0, 0.0], [0.0, 1.0], [1 / np.sqrt(2), -1 / np.sqrt(2)], [1 / np.sqrt(2), 1j / np.sqrt(2)]]


def _dense_rixs_pol(op, tin_comp, tout_comp, epsIn, epsOut, es, vecs, states):
    """Independent dense Kramers-Heisenberg map contracted with arbitrary in/out polarizations.

    T_in(e_in)  = sum_a e_in[a]      Tin_a      (in operators carry no dagger)
    T_out(e_out)= sum_b e_out[b]^*   Tout_b     (out operators are the daggered dipole components)
    A(e_in, e_out) = sum_g (w_g / Z) <g| T_in^dag R1 T_out^dag R2 T_out R1 T_in |g>.
    """
    n = len(states)
    eye = np.eye(n, dtype=complex)
    H = _matrix(op, states)
    Tin = [_matrix(t, states) for t in tin_comp]
    Tout = [_matrix(t, states) for t in tout_comp]
    nin, nout = len(Tin), len(Tout)
    ein = np.asarray(epsIn, dtype=complex)
    eout = np.asarray(epsOut, dtype=complex)
    e0 = min(es)
    Z = float(np.sum(np.exp(-(np.asarray(es) - e0) / TAU)))
    out = np.zeros((len(epsIn), len(epsOut), len(WIN), len(WLOSS)), dtype=complex)
    for g, eg in enumerate(es):
        gvec = vecs[:, g]
        wg = np.exp(-(eg - e0) / TAU)
        for ki, win in enumerate(WIN):
            psi2 = [np.linalg.solve((win + 1j * D1 + eg) * eye - H, Tin[a] @ gvec) for a in range(nin)]
            for pin, e_in in enumerate(ein):
                psi2_in = sum(e_in[a] * psi2[a] for a in range(nin))
                for pout, e_out in enumerate(eout):
                    psi3 = sum(np.conj(e_out[b]) * (Tout[b] @ psi2_in) for b in range(nout))
                    for kl, wl in enumerate(WLOSS):
                        r2 = np.linalg.solve((wl + 1j * D2 + eg) * eye - H, psi3)
                        out[pin, pout, ki, kl] += wg * (psi3.conj() @ r2)
    return out / Z


def _run_rixs_tensor(op, psis, es, tin, tout, dets, epsIn, epsOut):
    return spectra.getRIXSmap_tensor(
        op,
        tin,
        tout,
        epsIn,
        epsOut,
        psis,
        es,
        tau=TAU,
        wIns=WIN,
        wLoss=WLOSS,
        delta1=D1,
        delta2=D2,
        basis=_basis(dets),
        verbose=False,
        slaterWeightMin=0.0,
    )


def test_rixs_tensor_matches_dense_reference():
    """Full tensor contracted with arbitrary/circular polarizations vs an independent dense KH."""
    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    tin, tout = _tin_tout()
    got = _run_rixs_tensor(op, psis, es, tin, tout, dets, EPS_IN, EPS_OUT)
    ref = _dense_rixs_pol(op, tin, tout, EPS_IN, EPS_OUT, es, vecs, states)
    np.testing.assert_allclose(got, ref, atol=1e-8)


def test_rixs_tensor_matches_getRIXSmap_new():
    """Contracting the tensor reproduces the validated per-pair getRIXSmap_new for the same
    polarizations (built as the linear combinations of the component operators)."""
    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    tin, tout = _tin_tout()
    # Per-pair reference: T_in(e) = sum e_a Tin_a; T_out(e) = sum e_b^* Tout_b (daggered dipole).
    ref_in = [spectra._combine_component_ops(tin, e) for e in EPS_IN]
    ref_out = [spectra._combine_component_ops(tout, np.conj(e)) for e in EPS_OUT]
    ref = _run_rixs(op, psis, es, ref_in, ref_out, dets)
    got = _run_rixs_tensor(op, psis, es, tin, tout, dets, EPS_IN, EPS_OUT)
    np.testing.assert_allclose(got, ref, atol=1e-8)


def test_rixs_tensor_is_rotation_invariant():
    """Summed over a complete orthonormal polarization basis, the tensor map (a trace over the
    component span) is invariant under a single-particle basis rotation."""
    ident = np.eye(2, dtype=complex)
    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    tin, tout = _tin_tout()
    base = _run_rixs_tensor(op, psis, es, tin, tout, dets, ident, ident).sum(axis=(0, 1))

    W, _ = impurity_symmetry_rotation(op, [0, 1], n_orb=N_ORB)
    op_rot = rotate_hamiltonian(op, W)
    tin_rot = [ManyBodyOperator(spectra._rotate_op_dict(t.to_dict(), W)) for t in tin]
    tout_rot = [ManyBodyOperator(spectra._rotate_op_dict(t.to_dict(), W)) for t in tout]
    psis_rot, es_rot, dets_rot, _, _ = _thermal_states(op_rot, 2)
    rot = _run_rixs_tensor(op_rot, psis_rot, es_rot, tin_rot, tout_rot, dets_rot, ident, ident).sum(axis=(0, 1))

    np.testing.assert_allclose(base, rot, atol=1e-8)


# --- tests for the adaptive greedy-AAA wIn sampler ---

WIN_ADAPTIVE = np.linspace(-9.0, -5.0, 25)


def test_rixs_map_adaptive_synthetic():
    """The adaptive driver reconstructs a shared-pole map from a fraction of the solves."""
    rng = np.random.default_rng(3)
    poles = np.array([-8.2 + 0.3j, -7.0 + 0.25j, -6.1 + 0.4j])
    n_i, n_o, n_l = 2, 2, 17
    numerators = rng.standard_normal((n_i * n_o * n_l, 3)) + 1j * rng.standard_normal((n_i * n_o * n_l, 3))
    x = WIN_ADAPTIVE

    def truth(wins):
        cauchy = 1.0 / (np.asarray(wins)[:, None] - poles[None, :])
        flat = cauchy @ numerators.T  # (n_w, K)
        return np.moveaxis(flat.reshape(len(wins), n_i, n_o, n_l), 0, 2)

    calls = []

    def map_fn(wins):
        calls.append(len(wins))
        return truth(wins)

    got = spectra._rixs_map_adaptive(map_fn, x, None, tol=1e-8, verbose=False)
    ref = truth(x)
    assert np.max(np.abs(got - ref)) <= 1e-6 * np.max(np.abs(ref))
    assert sum(calls) < len(x), f"adaptive solved every point: {calls}"


def test_rixs_tensor_adaptive_matches_dense(monkeypatch):
    """End-to-end: getRIXSmap_tensor(adaptive_wIn_tol=...) matches the dense sweep on the
    model while solving fewer wIn points."""
    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    tin, tout = _tin_tout()

    def run(adaptive_tol):
        return spectra.getRIXSmap_tensor(
            op, tin, tout, EPS_IN, EPS_OUT, psis, es,
            tau=TAU, wIns=WIN_ADAPTIVE, wLoss=WLOSS, delta1=D1, delta2=D2,
            basis=_basis(dets), verbose=False, slaterWeightMin=0.0,
            adaptive_wIn_tol=adaptive_tol,
        )

    dense = run(None)

    solved_counts = []
    real_flat = spectra._rixs_map_flat

    def counting_flat(*args, **kwargs):
        # positional arg 5 is the wIn subset (hOp, in_ops, psis, Es, tau, wIns, ...)
        solved_counts.append(len(args[5]))
        return real_flat(*args, **kwargs)

    monkeypatch.setattr(spectra, "_rixs_map_flat", counting_flat)
    adaptive = run(1e-8)

    scale = np.max(np.abs(dense))
    assert np.max(np.abs(adaptive - dense)) <= 1e-6 * scale
    assert sum(solved_counts) < len(WIN_ADAPTIVE), f"no savings: {solved_counts}"


def test_rixs_tensor_adaptive_short_grid_stays_dense(monkeypatch):
    """Grids below _RIXS_ADAPTIVE_MIN_GRID are solved densely even with a tolerance set."""
    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    tin, tout = _tin_tout()
    solved_counts = []
    real_flat = spectra._rixs_map_flat

    def counting_flat(*args, **kwargs):
        solved_counts.append(len(args[5]))
        return real_flat(*args, **kwargs)

    monkeypatch.setattr(spectra, "_rixs_map_flat", counting_flat)
    spectra.getRIXSmap_tensor(
        op, tin, tout, EPS_IN, EPS_OUT, psis, es,
        tau=TAU, wIns=WIN, wLoss=WLOSS, delta1=D1, delta2=D2,
        basis=_basis(dets), verbose=False, slaterWeightMin=0.0,
        adaptive_wIn_tol=1e-6,
    )
    assert solved_counts == [len(WIN)]


def test_rixs_tensor_adaptive_env_knob(monkeypatch):
    """GF_RIXS_ADAPTIVE_TOL enables the sampler without a code change."""
    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    tin, tout = _tin_tout()
    solved_counts = []
    real_flat = spectra._rixs_map_flat

    def counting_flat(*args, **kwargs):
        solved_counts.append(len(args[5]))
        return real_flat(*args, **kwargs)

    monkeypatch.setattr(spectra, "_rixs_map_flat", counting_flat)
    monkeypatch.setenv("GF_RIXS_ADAPTIVE_TOL", "1e-8")
    spectra.getRIXSmap_tensor(
        op, tin, tout, EPS_IN, EPS_OUT, psis, es,
        tau=TAU, wIns=WIN_ADAPTIVE, wLoss=WLOSS, delta1=D1, delta2=D2,
        basis=_basis(dets), verbose=False, slaterWeightMin=0.0,
    )
    assert sum(solved_counts) < len(WIN_ADAPTIVE)


@pytest.mark.mpi
def test_rixs_tensor_adaptive_distributed_matches_dense():
    """The greedy selection loop stays collective on a genuinely distributed basis: the
    rank-0 fit is broadcast each round, every rank calls the solver with the same subset."""
    comm = MPI.COMM_WORLD
    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    tin, tout = _tin_tout()

    def run(adaptive_tol):
        world_basis = Basis(
            impurity_orbitals={2: [[0, 1]], 1: [[2]]},
            bath_states=({2: [[]], 1: [[]]}, {2: [[]], 1: [[]]}),
            initial_basis=list(dets),
            verbose=False,
            comm=comm,
        )
        return spectra.getRIXSmap_tensor(
            op, tin, tout, EPS_IN, EPS_OUT, psis, es,
            tau=TAU, wIns=WIN_ADAPTIVE, wLoss=WLOSS, delta1=D1, delta2=D2,
            basis=world_basis, verbose=False, slaterWeightMin=0.0,
            adaptive_wIn_tol=adaptive_tol,
        )

    dense = run(None)
    adaptive = run(1e-8)
    if comm.rank == 0:
        scale = np.max(np.abs(dense))
        assert np.max(np.abs(adaptive - dense)) <= 1e-6 * scale
    else:
        assert adaptive is None and dense is None


# --- tests for the R2 sector resolvent cache and the spurious-pole guard ---


def test_sector_resolvent_cache_matches_dense_solve():
    """One eigendecomposition serves different seed blocks and evaluation shifts."""
    from impurityModel.ed import greens_function as gf

    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    H = _matrix(op, states)
    cache = gf.SectorResolventCache()
    basis = _basis([])  # cache expands from the seeds
    eye = np.eye(len(states), dtype=complex)

    # the second seed set's support is nested inside the first's sector (the model's
    # {0,1} determinant is H-disconnected, so an unrelated seed would legitimately rebuild)
    for shift, seeds_src in ((0.7 + 0.3j, psis), (-1.1 + 0.2j, psis[:2])):
        zs = WLOSS + shift
        got = cache.try_eval(basis, op, list(seeds_src), zs)
        s_dense = np.array([[inner(sk, psi) for psi in seeds_src] for sk in states])
        ref = np.array([s_dense.conj().T @ np.linalg.solve(z * eye - H, s_dense) for z in zs])
        np.testing.assert_allclose(got, ref, atol=1e-10)
    assert cache._n_builds == 1, "second evaluation must reuse the eigendecomposition"
    assert cache._n_solves == 2


def test_sector_resolvent_cache_declines_oversized_sector(monkeypatch):
    """Above GF_SECTOR_DENSE_MAX the cache declines and the tensor path falls back to
    block-Lanczos -- and still matches the independent dense reference."""
    from impurityModel.ed import greens_function as gf

    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    monkeypatch.setenv("GF_SECTOR_DENSE_MAX", "1")
    cache = gf.SectorResolventCache()
    oversized_basis = _basis([])
    assert cache.try_eval(oversized_basis, op, list(psis[:2]), WLOSS + 0.3j) is None
    # the expansion probe must stop early, not complete the closure of a sector it
    # is about to decline (a massive sector would otherwise grow to truncation_threshold)
    assert len(oversized_basis) <= 2 + len(psis[0])
    # and the decline is sticky: later evaluations short-circuit without re-expanding
    assert cache._declined
    assert cache.try_solve(_basis([]), op, list(psis[:2]), 0.5 + 0.3j) is None

    tin, tout = _tin_tout()
    got = _run_rixs_tensor(op, psis, es, tin, tout, dets, EPS_IN, EPS_OUT)
    ref = _dense_rixs_pol(op, tin, tout, EPS_IN, EPS_OUT, es, vecs, states)
    np.testing.assert_allclose(got, ref, atol=1e-8)


def test_adaptive_blowup_guard_forces_solving_spurious_pole(monkeypatch):
    """A fit with a barycentric denominator zero between support points must not survive:
    the guard forces a solve at the blown-up node instead of trusting the reconstruction.

    The first fit is doctored to weights (1, 1) on the two outermost solved points -- its
    denominator vanishes at their midpoint, the classic Froissart signature. The surrogate
    alone would then compare the NEXT fit against this one and could stop; the guard must
    instead sample the artifact region and converge to the truth.
    """
    rng = np.random.default_rng(11)
    poles = np.array([-8.4 + 0.35j, -6.6 + 0.3j])
    n_i, n_o, n_l = 1, 1, 9
    numerators = rng.standard_normal((n_i * n_o * n_l, 2)) + 1j * rng.standard_normal((n_i * n_o * n_l, 2))
    x = np.linspace(-9.0, -5.0, 33)

    def truth(wins):
        cauchy = 1.0 / (np.asarray(wins)[:, None] - poles[None, :])
        flat = cauchy @ numerators.T
        return np.moveaxis(flat.reshape(len(wins), n_i, n_o, n_l), 0, 2)

    solved_wins = []

    def map_fn(wins):
        solved_wins.extend(np.atleast_1d(wins).tolist())
        return truth(wins)

    real_aaa = spectra.set_valued_aaa
    doctored = {"used": False}

    def poisoned_aaa(xs, F, rtol):
        if not doctored["used"]:
            doctored["used"] = True
            return [0, len(xs) - 1], np.array([1.0, 1.0], dtype=complex)
        return real_aaa(xs, F, rtol=rtol)

    monkeypatch.setattr(spectra, "set_valued_aaa", poisoned_aaa)
    got = spectra._rixs_map_adaptive(map_fn, x, None, tol=1e-8, verbose=False)
    ref = truth(x)
    assert np.max(np.abs(got - ref)) <= 1e-6 * np.max(np.abs(ref))
    # the denominator zero of the poisoned fit sits midway between the outermost initial
    # samples; the guard must have forced solves in that region
    assert len(solved_wins) > 5, "guard never forced additional solves"


def test_sector_resolvent_cache_solve_matches_dense():
    """try_solve returns exact (z - H)^-1 rhs on the sector, reusing the eigendecomposition."""
    from impurityModel.ed import greens_function as gf

    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    H = _matrix(op, states)
    cache = gf.SectorResolventCache()
    basis = _basis([])
    eye = np.eye(len(states), dtype=complex)

    for z in (0.5 + 0.4j, -7.3 + 0.4j):
        got = cache.try_solve(basis, op, list(psis), z)
        rhs = np.array([[inner(sk, psi) for psi in psis] for sk in states])
        ref = np.linalg.solve(z * eye - H, rhs)
        got_dense = np.array([[inner(sk, x) for x in got] for sk in states])
        np.testing.assert_allclose(got_dense, ref, atol=1e-10)
    assert cache._n_builds == 1


def test_sector_resolvent_cache_disk_persistence(monkeypatch, tmp_path):
    """GF_SECTOR_CACHE_DIR persists the eigendecomposition across cache instances; a
    changed Hamiltonian gets a different digest (no false hit)."""
    from impurityModel.ed import greens_function as gf

    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    monkeypatch.setenv("GF_SECTOR_CACHE_DIR", str(tmp_path))
    zs = WLOSS + 0.3j

    first = gf.SectorResolventCache()
    g1 = first.try_eval(_basis([]), op, list(psis), zs)
    files = list(tmp_path.glob("sector_*.npz"))
    assert len(files) == 1

    # a fresh instance must load, not re-eigendecompose
    def no_eigh(*a, **k):
        raise AssertionError("eigh called despite a valid disk cache")

    monkeypatch.setattr(np.linalg, "eigh", no_eigh)
    second = gf.SectorResolventCache()
    g2 = second.try_eval(_basis([]), op, list(psis), zs)
    np.testing.assert_allclose(g2, g1, atol=1e-12)
    monkeypatch.undo()
    monkeypatch.setenv("GF_SECTOR_CACHE_DIR", str(tmp_path))

    # perturbing the Hamiltonian changes the digest -> a new build, a new file
    op2 = ManyBodyOperator({**op.to_dict(), ((0, "c"), (0, "a")): 0.51 + 0j})
    third = gf.SectorResolventCache()
    third.try_eval(_basis([]), op2, list(psis), zs)
    assert len(list(tmp_path.glob("sector_*.npz"))) == 2

    # a corrupt file is ignored and rebuilt, not fatal
    files[0].write_bytes(b"garbage")
    fourth = gf.SectorResolventCache()
    g4 = fourth.try_eval(_basis([]), op, list(psis), zs)
    np.testing.assert_allclose(g4, g1, atol=1e-12)


# --- tests for the shift-recycled Krylov resolvent and the GMRES escalation ---


def test_krylov_shifted_resolvent_matches_dense_solve():
    """One block-Lanczos recurrence solves (z - H) x = y for every shift, matching the
    dense solve on the seeds' H-closed sector."""
    from impurityModel.ed import greens_function as gf

    op = _model()
    psis, es, dets, states, vecs = _thermal_states(op, 2)
    H = _matrix(op, states)
    eye = np.eye(len(states), dtype=complex)
    zs = np.array([0.7 + 0.3j, -1.1 + 0.2j, 2.5 + 0.05j, 0.7 + 0.3j])
    rhs = list(psis)  # spans both connected components, so the closure is the full sector

    sols = gf.KrylovShiftedResolvent().solve(_basis([]), op, rhs, zs, slaterWeightMin=0.0, atol=1e-10)
    assert sols is not None
    s_dense = np.array([[inner(sk, psi) for psi in rhs] for sk in states])
    for z, xs in zip(zs, sols):
        ref = np.linalg.solve(z * eye - H, s_dense)
        got = np.array([[inner(sk, x) for x in xs] for sk in states])
        np.testing.assert_allclose(got, ref, atol=1e-8)
