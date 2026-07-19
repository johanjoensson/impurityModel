"""Rank-count reproducibility of the CIPSI basis selection.

The adaptively selected CIPSI basis used to depend on how many MPI ranks the run was launched
with -- the ground-state *energy* was reproducible (which is what the other tests compare), but
the selected determinants were not, and every excited basis and Green's function built on them
inherited the difference. Three independent causes, one test each:

1. ``_calc_de2``'s diagonal probe applied ``H`` to each rank's *local* candidate slice, so the
   candidate-candidate couplings ``<Dj|H|Dk>`` with ``Dk`` on another rank were missing -- and
   which ones were missing depended on ``comm.size``.
2. The Lanczos start vector came from ``random.seed(42 + rank)``, i.e. from the partition.
3. The candidate score ``max_i |<Dj|H|psi_i>|^2 / de`` is not invariant under rotations *within*
   a degenerate reference manifold, and a restarted Lanczos returns an arbitrary basis of such a
   manifold (all rotations share the same residual).
"""

import itertools

import numpy as np
import pytest

from impurityModel.ed.cipsi_solver import (
    _EIGEN_TOL_FLOOR,
    _EIGEN_TOL_MAX,
    DEGENERACY_TOL,
    _amplitude_from_hash,
    _degenerate_groups,
    _eigen_tol,
    _energy_cut_indices,
    _splitmix64,
)


def test_eigen_tol_follows_slater_weight_min():
    """The eigenvector must be converged below the amplitude cutoff, or its own noise survives
    the prune and the state's support becomes rank-count dependent."""
    # a tight cutoff demands a tight eigensolve
    assert _eigen_tol(1e-12) == 1e-12
    assert _eigen_tol(1e-10) == 1e-10
    # ...but never looser than the historical default, even for a loose cutoff
    assert _eigen_tol(1e-5) == _EIGEN_TOL_MAX
    assert _eigen_tol(1.0) == _EIGEN_TOL_MAX
    # ...and never chasing roundoff
    assert _eigen_tol(1e-20) == _EIGEN_TOL_FLOOR
    # no pruning -> the noise is harmless, keep the cheap default
    assert _eigen_tol(0.0) == _EIGEN_TOL_MAX
    assert _eigen_tol(None) == _EIGEN_TOL_MAX
    # monotone: a tighter cutoff never yields a looser tolerance
    cuts = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
    tols = [_eigen_tol(c) for c in cuts]
    assert all(a >= b for a, b in itertools.pairwise(tols))


def test_amplitude_from_hash_is_deterministic_and_rank_independent():
    """The start amplitude must depend only on the determinant, never on the process."""
    assert _amplitude_from_hash(12345) == _amplitude_from_hash(12345)
    vals = [_amplitude_from_hash(h) for h in range(5000)]
    assert len(set(vals)) == len(vals), "amplitudes collide"
    a = np.array(vals)
    # Uniform on [0,1) in both components, with uncorrelated real/imaginary streams.
    assert a.real.min() >= 0.0 and a.real.max() < 1.0
    assert a.imag.min() >= 0.0 and a.imag.max() < 1.0
    assert abs(a.real.mean() - 0.5) < 0.02 and abs(a.imag.mean() - 0.5) < 0.02
    assert abs(np.corrcoef(a.real, a.imag)[0, 1]) < 0.05


def test_splitmix64_is_a_bijection_on_small_inputs():
    assert len({_splitmix64(x) for x in range(10000)}) == 10000
    assert all(0 <= _splitmix64(x) < 2**64 for x in range(100))


def test_degenerate_groups():
    assert _degenerate_groups([]) == []
    assert _degenerate_groups([1.0]) == [[0]]
    # ascending, one exactly-degenerate pair
    assert _degenerate_groups([-111.17311714, -111.173059, -111.173059]) == [[0], [1, 2]]
    assert _degenerate_groups([-1.0, -1.0, -1.0]) == [[0, 1, 2]]
    # splittings well above the tolerance stay separate
    assert _degenerate_groups([0.0, 10 * DEGENERACY_TOL, 1.0]) == [[0], [1], [2]]


def test_energy_cut_never_bisects_a_degenerate_manifold():
    d = DEGENERACY_TOL / 10  # inside the manifold tolerance
    # The cut at 0.5 would keep only the first two of a degenerate triplet at ~1.0 ... except the
    # triplet starts below the cut, so all three must come along.
    e = [0.0, 0.4, 0.4 + d, 0.4 + 2 * d, 5.0]
    idx, need_more = _energy_cut_indices(e, max_energy=0.5)
    assert idx == [0, 1, 2, 3], "a degenerate manifold straddling the cut must be kept whole"
    assert need_more is False  # e=5.0 lies outside: the boundary manifold is certified complete


def test_energy_cut_requests_more_states_when_none_lie_outside():
    """If every computed state is inside the cut, the boundary manifold is not certified."""
    idx, need_more = _energy_cut_indices([0.0, 0.1, 0.2], max_energy=10.0)
    assert idx == [0, 1, 2]
    assert need_more is True

    idx, need_more = _energy_cut_indices([0.0, 0.1, 20.0], max_energy=10.0)
    assert idx == [0, 1]
    assert need_more is False


def test_energy_cut_handles_edges():
    assert _energy_cut_indices([], max_energy=1.0) == ([], False)
    assert _energy_cut_indices([3.0], max_energy=1.0) == ([0], True)  # always keep the lowest
    # unsorted input is handled
    idx, _ = _energy_cut_indices([5.0, 0.0, 0.1], max_energy=1.0)
    assert idx == [1, 2]
    # no cut requested -> everything, nothing to certify
    assert _energy_cut_indices([0.0, 9.0], max_energy=None) == ([0, 1], False)


def _score(de2, e_ref):
    """The production candidate score: max over manifolds of the manifold-summed de2."""
    de2_abs = np.abs(de2)
    return np.max(np.stack([de2_abs[g, :].sum(axis=0) for g in _degenerate_groups(e_ref)]), axis=0)


def test_score_is_invariant_under_rotations_within_a_degenerate_manifold():
    """A restarted Lanczos returns an arbitrary basis of a degenerate eigenspace; the candidate
    score must not move when that basis rotates. ``max_i`` does move -- that is the bug."""
    rng = np.random.default_rng(0)
    n_cand = 40
    e_ref = np.array([-2.0, -1.0, -1.0])  # one singlet + one degenerate pair
    de = np.abs(e_ref[:, None] - rng.normal(size=(1, n_cand)))

    overlaps = rng.normal(size=(3, n_cand)) + 1j * rng.normal(size=(3, n_cand))
    de2 = np.abs(overlaps) ** 2 / de

    # Rotate the degenerate pair (rows 1,2) by an arbitrary angle: physically the same subspace.
    theta = 0.7
    c, s = np.cos(theta), np.sin(theta)
    rot = overlaps.copy()
    rot[1] = c * overlaps[1] + s * overlaps[2]
    rot[2] = -s * overlaps[1] + c * overlaps[2]
    de2_rot = np.abs(rot) ** 2 / de

    np.testing.assert_allclose(_score(de2, e_ref), _score(de2_rot, e_ref), rtol=1e-12)

    # The old score (max over individual reference states) is *not* invariant.
    old = np.max(np.abs(de2), axis=0)
    old_rot = np.max(np.abs(de2_rot), axis=0)
    assert not np.allclose(old, old_rot, rtol=1e-6), "expected the max-over-states score to move"


def test_score_reduces_to_the_max_when_nondegenerate():
    rng = np.random.default_rng(1)
    e_ref = np.array([-3.0, -2.0, -1.0])  # no degeneracy -> every manifold is a singleton
    de2 = np.abs(rng.normal(size=(3, 25))) ** 2
    np.testing.assert_allclose(_score(de2, e_ref), np.max(np.abs(de2), axis=0), rtol=0, atol=0)


def test_score_equals_the_projection_norm_not_a_degeneracy_factor():
    """Sum over a manifold is ||v||^2/de -- the squared norm of H|Dj> projected onto the
    eigenspace. When the coupling lies along a single direction it equals the max exactly:
    there is no factor of the degeneracy."""
    e_ref = np.array([-1.0, -1.0])
    de = np.ones((2, 1))
    aligned = np.array([[3.0], [0.0]])  # all weight on one member of the pair
    de2 = np.abs(aligned) ** 2 / de
    assert _score(de2, e_ref)[0] == pytest.approx(9.0)
    assert np.max(np.abs(de2), axis=0)[0] == pytest.approx(9.0)  # identical, no d-scaling

    # Smearing the same physical coupling across the manifold leaves the score unchanged...
    smeared = np.array([[3.0 / np.sqrt(2)], [3.0 / np.sqrt(2)]])
    de2_s = np.abs(smeared) ** 2 / de
    assert _score(de2_s, e_ref)[0] == pytest.approx(9.0)
    # ...while the old max understates it by 1/d.
    assert np.max(np.abs(de2_s), axis=0)[0] == pytest.approx(4.5)
