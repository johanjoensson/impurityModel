"""Tests for the Chebyshev filter stage (``impurityModel.ed.chebyshev_filter``).

Oracles are dense: the filtered many-body seed must equal ``p_s(H_dense) v`` built from
the same coefficients via ``numpy.polynomial.chebyshev``, the tiling windows must sum to
one identically on the spectrum, and a ``caps_growth`` basis must bound the recurrence
(the filtered seeds become those of ``P H P``).
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI
from numpy.polynomial import chebyshev as npcheb

from impurityModel.ed.basis_transcription import build_sparse_matrix, build_vector
from impurityModel.ed.chebyshev_filter import chebyshev_apply, partition_of_unity, spectral_bounds
from impurityModel.ed.greens_function import _CappedBasisProxy
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, SlaterDeterminant


def _det(occupied):
    b = 0
    for i in occupied:
        b |= 1 << (7 - i)
    return SlaterDeterminant.from_bytes(bytes([b]))


def _siam_6():
    """Single-impurity Anderson model, 6 spin-orbitals (0,1 imp; 2,3 val; 4,5 cond)."""
    ed_, u, ev, ec, v = -1.0, 4.0, -3.0, 3.0, 0.5
    terms = {}
    for o in (0, 1):
        terms[((o, "c"), (o, "a"))] = ed_
    for o in (2, 3):
        terms[((o, "c"), (o, "a"))] = ev
    for o in (4, 5):
        terms[((o, "c"), (o, "a"))] = ec
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = u
    for a, b in ((0, 2), (1, 3), (0, 4), (1, 5)):
        terms[((a, "c"), (b, "a"))] = v
        terms[((b, "c"), (a, "a"))] = v
    return ManyBodyOperator(terms)


def _sector_basis():
    """The full N=3 sector (closed under the SIAM-6 H), 20 determinants."""
    dets = []
    for c in itertools.combinations(range(6), 3):
        b = bytearray(1)
        for o in c:
            b[o // 8] |= 1 << (7 - (o % 8))
        dets.append(bytes(b))
    return Basis(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[2, 3]]}, {0: [[4, 5]]}),
        initial_basis=dets,
        verbose=False,
    )


def _seeds():
    return [
        ManyBodyState({_det([0, 2, 3]): 1.0 + 0j, _det([1, 2, 3]): 0.5 + 0j}),
        ManyBodyState({_det([0, 1, 2]): 1.0 + 0j}),
    ]


def _redistribute_as_width1(basis, states, n):
    """Redistribute ``n`` seeds through width-1 blocks on every rank, then unpack back
    to flat states. Some ranks in these tests hold no amplitudes at all (``states`` is
    None there) -- a bare ``ManyBodyState()`` placeholder for that case is the width-0
    polymorphic zero once the flat and block classes merge (Phase 7 step 3), an
    asymmetric mismatch against another rank's populated (eventually width-1) seeds
    that would deadlock redistribute_psis' collective. from_states forces an explicit
    width-1 block -- empty or not -- on every rank instead, so the collective sees the
    same representation everywhere; to_states() immediately unpacks the result back to
    the flat list chebyshev_apply still expects (not yet Row-safe, Phase 7 step 3.4)."""
    blocks = (
        [ManyBodyState.from_states([s]) for s in states]
        if states is not None
        else [ManyBodyState(width=1) for _ in range(n)]
    )
    return [blk.to_states()[0] for blk in basis.redistribute_psis(blocks)]


def test_spectral_bounds_bracket_the_spectrum():
    basis = _sector_basis()
    H = _siam_6()
    lo, hi = spectral_bounds(H, basis, n_iter=25)
    ev = np.linalg.eigvalsh(build_sparse_matrix(basis, H).toarray())
    assert lo < ev[0] and hi > ev[-1]
    # padded, but not absurdly: within half a bandwidth on each side
    width = ev[-1] - ev[0]
    assert lo > ev[0] - 0.5 * width and hi < ev[-1] + 0.5 * width


def test_partition_of_unity_is_exact():
    """The tiling windows' polynomials sum to 1 identically on [-1, 1] (telescoping,
    Jackson damping included): p_0 sums to 1, every higher coefficient cancels."""
    bounds = (-7.3, 9.1)
    edges = np.linspace(-5.0, 5.0, 7)
    coeff_sets, window_edges, edge_width = partition_of_unity(bounds, edges, degree=200)
    total = np.sum(coeff_sets, axis=0)
    assert abs(total[0] - 1.0) < 1e-13
    assert np.max(np.abs(total[1:])) < 1e-13
    # windows tile [lo, hi] exactly
    assert window_edges[0][0] == bounds[0] and window_edges[-1][1] == bounds[1]
    assert edge_width > 0


def _dense_filter(H_mat, coeffs, bounds):
    """Matrix polynomial ``p(H)`` via eigendecomposition (chebval on a matrix argument is
    elementwise, NOT the matrix polynomial)."""
    ev, U = np.linalg.eigh(H_mat)
    x = (ev - 0.5 * (bounds[1] + bounds[0])) / (0.5 * (bounds[1] - bounds[0]))
    return (U * npcheb.chebval(x, coeffs)) @ U.conj().T


def test_filtered_seed_matches_dense_oracle():
    """chebyshev_apply == p_s(H_dense) v, column by column, window by window."""
    basis = _sector_basis()
    H = _siam_6()
    H_mat = build_sparse_matrix(basis, H).toarray()
    ev = np.linalg.eigvalsh(H_mat)
    bounds = (ev[0] - 0.5, ev[-1] + 0.5)
    coeff_sets, _, _ = partition_of_unity(bounds, np.linspace(ev[0], ev[-1], 4), degree=120)

    seeds = _seeds()
    filtered = chebyshev_apply(H, basis, seeds, coeff_sets, 0.0, bounds)

    V = build_vector(basis, seeds).T
    for s, c in enumerate(coeff_sets):
        ref = _dense_filter(H_mat, c, bounds) @ V
        got = build_vector(basis, filtered[s]).T
        np.testing.assert_allclose(got, ref, atol=1e-10)


def test_filter_bank_sums_to_the_seed():
    """sum_s p_s(H) v == v (the partition-of-unity identity applied to a state)."""
    basis = _sector_basis()
    H = _siam_6()
    ev = np.linalg.eigvalsh(build_sparse_matrix(basis, H).toarray())
    bounds = (ev[0] - 0.5, ev[-1] + 0.5)
    coeff_sets, _, _ = partition_of_unity(bounds, np.linspace(ev[0] + 1, ev[-1] - 1, 5), degree=150)
    seeds = _seeds()
    filtered = chebyshev_apply(H, basis, seeds, coeff_sets, 0.0, bounds)
    for col in range(len(seeds)):
        total = ManyBodyState()
        for s in range(len(coeff_sets)):
            total = total + filtered[s][col]
        diff = total - seeds[col]
        assert np.sqrt(diff.norm2()) < 1e-12


def test_filter_localizes_spectral_weight():
    """A window's filtered seed has ~all its H-eigenbasis weight inside the window
    (up to the Jackson edge broadening)."""
    basis = _sector_basis()
    H = _siam_6()
    H_mat = build_sparse_matrix(basis, H).toarray()
    ev, U = np.linalg.eigh(H_mat)
    bounds = (ev[0] - 0.5, ev[-1] + 0.5)
    mid = 0.5 * (ev[0] + ev[-1])
    coeff_sets, window_edges, edge_width = partition_of_unity(bounds, [mid], degree=400)
    seeds = _seeds()
    filtered = chebyshev_apply(H, basis, seeds, coeff_sets, 0.0, bounds)
    for s, (lo_w, hi_w) in enumerate(window_edges):
        v = build_vector(basis, filtered[s]).T[:, 0]
        weights = np.abs(U.conj().T @ v) ** 2
        inside = (ev >= lo_w - 2 * edge_width) & (ev <= hi_w + 2 * edge_width)
        leak = np.sum(weights[~inside])
        assert leak < 1e-3 * max(np.sum(weights), 1e-30), (s, leak)


def test_capped_recurrence_is_php_filter():
    """Under a caps_growth basis the recurrence never exceeds the cap, and the filtered
    seeds equal p_s(PHP) v on the retained set — the shared truncation contract."""
    basis = _sector_basis()
    H = _siam_6()
    # localized seed so the cap binds through growth, not through the seed itself
    seeds = [ManyBodyState({_det([0, 2, 3]): 1.0 + 0j})]
    small = Basis(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[2, 3]]}, {0: [[4, 5]]}),
        initial_basis=sorted(seeds[0].keys()),
        verbose=False,
    )
    cap = 8
    proxy = _CappedBasisProxy(small, cap)
    ev = np.linalg.eigvalsh(build_sparse_matrix(basis, H).toarray())
    bounds = (ev[0] - 0.5, ev[-1] + 0.5)
    coeff_sets, _, _ = partition_of_unity(bounds, [0.5 * (ev[0] + ev[-1])], degree=100)
    filtered = chebyshev_apply(H, proxy, seeds, coeff_sets, 0.0, bounds)
    assert proxy.cap_hit and proxy.retained_size <= cap
    retained = sorted(proxy.retained_keys())
    # dense PHP oracle on the retained set
    php_basis = Basis(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[2, 3]]}, {0: [[4, 5]]}),
        initial_basis=retained,
        verbose=False,
    )
    Hp = build_sparse_matrix(php_basis, H).toarray()
    Vp = build_vector(php_basis, seeds).T
    for s, c in enumerate(coeff_sets):
        ref = _dense_filter(Hp, c, bounds) @ Vp
        got = build_vector(php_basis, php_basis.redistribute_psis(filtered[s])).T
        np.testing.assert_allclose(got, ref, atol=1e-10)


def test_seed_pruning_bounds_live_support():
    """slaterWeightMin > 0 prunes the recurrence blocks (the memory knob is honored)."""
    basis = _sector_basis()
    H = _siam_6()
    ev = np.linalg.eigvalsh(build_sparse_matrix(basis, H).toarray())
    bounds = (ev[0] - 0.5, ev[-1] + 0.5)
    coeff_sets, _, _ = partition_of_unity(bounds, [0.0], degree=60)
    filtered = chebyshev_apply(H, basis, _seeds(), coeff_sets, 1e-3, bounds)
    exact = chebyshev_apply(H, basis, _seeds(), coeff_sets, 0.0, bounds)
    # pruned result differs from exact by at most ~degree * cutoff in norm
    for s in range(len(coeff_sets)):
        for col in range(2):
            diff = filtered[s][col] - exact[s][col]
            assert np.sqrt(diff.norm2()) < 60 * 1e-3


@pytest.mark.mpi
def test_chebyshev_apply_mpi_matches_serial():
    comm = MPI.COMM_WORLD
    H = _siam_6()
    dets = []
    for c in itertools.combinations(range(6), 3):
        b = bytearray(1)
        for o in c:
            b[o // 8] |= 1 << (7 - (o % 8))
        dets.append(bytes(b))
    dist = Basis(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[2, 3]]}, {0: [[4, 5]]}),
        initial_basis=dets,
        comm=comm,
        verbose=False,
    )
    ev_bounds = spectral_bounds(_siam_6(), dist, n_iter=20)
    coeff_sets, _, _ = partition_of_unity(ev_bounds, np.linspace(ev_bounds[0] + 1, ev_bounds[1] - 1, 3), degree=80)
    seeds_full = _seeds()
    seeds = _redistribute_as_width1(dist, seeds_full if comm.rank == 0 else None, len(seeds_full))
    filtered = chebyshev_apply(H, dist, list(seeds), coeff_sets, 0.0, ev_bounds)
    # partition identity holds distributed: sum_s p_s v == v on this rank's rows
    for col in range(2):
        total = ManyBodyState()
        for s in range(len(coeff_sets)):
            total = total + filtered[s][col]
        diff = total - seeds[col]
        assert np.sqrt(diff.norm2()) < 1e-10


@pytest.mark.mpi
def test_chebyshev_apply_mpi_redistributes_misplaced_seeds():
    """Seeds whose amplitudes sit on the rank that GENERATED them, not the rank that OWNS the
    determinant -- the state a rank-local c/c^dagger apply leaves them in.

    The recurrence redistributes ``H t`` but not ``t``, so a misplaced row leaves ``H t`` on
    the owner and ``t`` on the generator: the recurrence decouples across ranks and diverges
    to inf (this is exactly what the sliced driver hit in production). ``chebyshev_apply``
    must therefore redistribute its own seed block on entry.

    The sibling test above cannot catch this: it seeds the *closed* C(6,3) sector and hands
    over already-redistributed seeds, so generator == owner there.
    """
    comm = MPI.COMM_WORLD
    H = _siam_6()
    dets = []
    for c in itertools.combinations(range(6), 3):
        b = bytearray(1)
        for o in c:
            b[o // 8] |= 1 << (7 - (o % 8))
        dets.append(bytes(b))
    dist = Basis(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[2, 3]]}, {0: [[4, 5]]}),
        initial_basis=dets,
        comm=comm,
        verbose=False,
    )
    ev_bounds = spectral_bounds(_siam_6(), dist, n_iter=20)
    coeff_sets, _, _ = partition_of_unity(ev_bounds, np.linspace(ev_bounds[0] + 1, ev_bounds[1] - 1, 3), degree=80)

    # Deliberately misplaced: rank 0 holds every amplitude, no rank owns what it holds.
    misplaced = _seeds() if comm.rank == 0 else [ManyBodyState(width=1) for _ in _seeds()]
    filtered = chebyshev_apply(H, dist, list(misplaced), coeff_sets, 0.0, ev_bounds)

    # sum_s p_s v == v, compared against the *properly* distributed seeds: the entry
    # redistribute must have moved every row to its owner.
    owned_full = _seeds()
    owned = _redistribute_as_width1(dist, owned_full if comm.rank == 0 else None, len(owned_full))
    for col in range(2):
        total = ManyBodyState()
        for s in range(len(coeff_sets)):
            total = total + filtered[s][col]
        diff = total - owned[col]
        assert np.sqrt(diff.norm2()) < 1e-10
        # and the filter did not run away (the failure mode was inf, not a small error)
        assert np.isfinite(total.norm2())
