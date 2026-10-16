"""A degenerate level wider than the Krylov block must still come back whole.

Block Lanczos started from ``p`` vectors can reach at most ``p`` copies of an exactly degenerate
eigenvalue: the Krylov space ``span{Q0, H Q0, H^2 Q0, ...}`` meets the eigenspace in at most
``rank(P Q0)`` dimensions, where ``P`` projects onto it. CIPSI warm-starts every solve from the
previous cycle's converged eigenvectors (capped at ``GS_MAX_BLOCK_WIDTH``) plus one cold vector, so
the block's reach into an excited degenerate level is typically *one* direction. Found on the
SrMnO3 gap double-counting search (cap 8000, tau = 0.0025, thermal window 23 meV): a 10-fold level
17.6 meV above an exactly 5-fold ground level came back with 4 to 9 of its 10 copies, every one
converged to 1e-13. ``_energy_cut_indices`` certifies a manifold only by finding a state beyond the
cut, which it did, so nothing downstream saw the hole -- and a thermal manifold with missing copies
gives wrong Boltzmann weights, a wrong CIPSI selection and wrong Green's-function seeds.

The Hamiltonian here is that spectrum with no physics attached: ``U diag(lambda) U^H`` over a
one-particle determinant space, handed to ``get_eigenvectors`` as ``h_matrix``. The warm start is
the exact ground manifold, as a converged CIPSI cycle would hand over.
"""

import numpy as np
import pytest
import scipy.sparse as sps
from mpi4py import MPI

from impurityModel.ed.basis_transcription import build_state
from impurityModel.ed.BlockLanczosCore import block_inner
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.DistributedSparseMatrix import DistributedSparseMatrix
from impurityModel.ed.eigensolvers import Lowest, Within
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

N_ORB = 64
GROUND = 5  # exactly degenerate ground level
EXCITED = 10  # exactly degenerate level inside the window, wider than the start block
E_EXCITED = 0.0176
CUT = 0.023  # the SrMnO3 window: energy_cut(tau=0.0025)
N_WINDOW = GROUND + EXCITED


def _singlet(orbital):
    b = bytearray(N_ORB // 8)
    b[orbital // 8] |= 1 << (7 - orbital % 8)
    return bytes(b)


@pytest.fixture(scope="module")
def system():
    comm = MPI.COMM_WORLD
    basis = Basis(
        impurity_orbitals={0: [list(range(N_ORB))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=[_singlet(o) for o in range(N_ORB)],
        verbose=False,
        comm=comm,
    )
    n = len(basis)
    rng = np.random.default_rng(11)
    evals = np.concatenate([np.zeros(GROUND), np.full(EXCITED, E_EXCITED), np.linspace(0.12, 3.0, n - N_WINDOW)])
    U = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    h = (U * evals) @ U.conj().T
    h = 0.5 * (h + h.conj().T)
    return basis, sps.csr_matrix(h), U, evals


def _solve(basis, h, U, manifold, warm=True, dense_cutoff=10):
    """``get_eigenvectors`` on ``h``, warm-started from the exact ground manifold (as a converged CIPSI
    cycle hands over) or, with ``warm=False``, cold: the width-1 hash vector, as the cold-start solves."""
    psi_refs = None
    if warm:
        local = np.asarray(list(basis.local_indices), dtype=int)
        psi_refs = build_state(basis, U[local, :GROUND].T)
    solver = CIPSISolver(basis)
    dummy = ManyBodyOperator({((0, "c"), (0, "a")): 0.0})
    h = DistributedSparseMatrix.from_scipy(
        sps.csc_array(h)[:, list(basis.local_indices)], basis.comm if basis.is_distributed else None, basis.offset
    )
    return solver.get_eigenvectors(
        dummy, manifold, psi_refs=psi_refs, h_matrix=h, dense_cutoff=dense_cutoff, slaterWeightMin=1e-12
    )


def _assert_orthonormal_and_ascending(basis, e_ref, psis):
    """No eigenstate returned twice, and the order callers take columns by position in."""
    e = np.real(np.asarray(e_ref))
    assert np.all(np.diff(e) >= 0), f"eigenvalues not ascending: {e}"
    gram = block_inner(psis, psis, basis.is_distributed, basis.comm)
    np.testing.assert_allclose(gram, np.eye(len(e)), atol=1e-8)


@pytest.mark.mpi
@pytest.mark.parametrize("warm", [True, False], ids=["warm", "cold"])
def test_a_degenerate_level_wider_than_the_block_comes_back_whole(system, warm):
    """Cold is the width-1 start of every cold-start solve: the probe must not inherit that width."""
    basis, h, U, evals = system
    e_ref, psis = _solve(basis, h, U, Within(CUT), warm=warm)
    e = np.real(e_ref)
    assert len(e) == N_WINDOW, (
        f"the thermal window holds {N_WINDOW} states ({GROUND} ground + {EXCITED} at {E_EXCITED}); "
        f"got {len(e)}: {np.round(e, 5)}"
    )
    np.testing.assert_allclose(e, evals[:N_WINDOW], atol=1e-9)
    _assert_orthonormal_and_ascending(basis, e_ref, psis)


@pytest.mark.mpi
def test_a_lowest_request_on_a_degenerate_level_returns_exactly_one_state(system):
    """``Lowest`` -- the occupation walk and every double-counting sector solve. The one state asked
    for belongs to a 5-fold level. The completeness probe once ran here, locked only that one,
    "found" the other computed copies in the complement and returned them next to their originals:
    9 ground states for a 5-fold level, Gram error 0.76. A ``Lowest`` request claims nothing about
    the level's multiplicity and returns one state."""
    basis, h, U, _evals = system
    e_ref, psis = _solve(basis, h, U, Lowest())
    assert len(e_ref) == psis.width == 1
    np.testing.assert_allclose(np.real(e_ref), [0.0], atol=1e-9)
    _assert_orthonormal_and_ascending(basis, e_ref, psis)


@pytest.mark.mpi
def test_a_level_just_above_the_cut_is_not_returned_twice():
    """A level within the degeneracy tolerance above the cut: outside the window, inside the probe's
    boundary. Left out of the locked set it was re-found, the found copies' energies landed at or
    below the cut, and the final cut kept both them and their originals -- a thermal level counted
    twice."""
    comm = MPI.COMM_WORLD
    basis = Basis(
        impurity_orbitals={0: [list(range(N_ORB))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=[_singlet(o) for o in range(N_ORB)],
        verbose=False,
        comm=comm,
    )
    n = len(basis)
    rng = np.random.default_rng(12)
    edge = CUT + 5e-10  # the degeneracy tolerance here is 1e-9
    evals = np.concatenate([np.zeros(GROUND), np.full(3, edge), np.linspace(0.12, 3.0, n - GROUND - 3)])
    U = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    h = (U * evals) @ U.conj().T
    h = sps.csr_matrix(0.5 * (h + h.conj().T))
    e_ref, psis = _solve(basis, h, U, Within(CUT))
    _assert_orthonormal_and_ascending(basis, e_ref, psis)
    assert np.sum(np.abs(np.real(e_ref) - edge) < 1e-6) <= 3


@pytest.mark.mpi
def test_a_complete_window_is_certified_without_a_fallback_solve(system, monkeypatch):
    """The common case -- nothing missing -- costs the one window solve and no fallback.

    Warm columns span the ground level, so ``k = 5 < rank(F^H W) + R`` and the certificate holds
    from the first solve."""
    import impurityModel.ed.eigensolvers as eigensolvers

    basis, h, U, evals = system
    calls = []
    real = eigensolvers.trlm

    def counting(*args, **kwargs):
        calls.append(kwargs.get("locked") is not None)
        return real(*args, **kwargs)

    monkeypatch.setattr(eigensolvers, "trlm", counting)
    solver_eig = _solve(basis, h, U, Within(E_EXCITED / 2))
    np.testing.assert_allclose(np.real(solver_eig[0]), evals[:GROUND], atol=1e-9)
    assert calls == [False]


def _window_solve(system, *, n_random, warm, max_energy=CUT):
    import impurityModel.ed.eigensolvers as eigensolvers

    basis, h, U, evals = system
    comm = basis.comm if basis.is_distributed else None
    h_dist = DistributedSparseMatrix.from_scipy(sps.csc_array(h)[:, list(basis.local_indices)], comm, basis.offset)
    hashes = np.fromiter((state.get_hash() for state in basis.local_basis), dtype=np.uint64)
    local = np.asarray(list(basis.local_indices), dtype=int)
    return eigensolvers.solve_window(
        h_dist,
        comm,
        hashes,
        warm=np.ascontiguousarray(U[local, :GROUND]) if warm else None,
        max_energy=max_energy,
        deg_tol=1e-8,
        tol=1e-11,
        num_wanted=4,
        n_global=len(basis),
        n_random=n_random,
    )


@pytest.mark.mpi
@pytest.mark.parametrize("warm", [True, False], ids=["warm", "cold"])
def test_a_level_wider_than_the_random_block_is_completed_by_the_fallback(system, warm):
    """10 copies against ``R = 2`` generic columns: the first solve saturates, the fallback finishes."""
    eig = _window_solve(system, n_random=2, warm=warm)
    evals = system[3]
    assert eig.certified
    assert eig.n_in == N_WINDOW
    np.testing.assert_allclose(eig.e[:N_WINDOW], evals[:N_WINDOW], atol=1e-9)
    assert eig.max_multiplicity == EXCITED


@pytest.mark.mpi
def test_without_the_certificate_the_wide_level_loses_copies(system, monkeypatch):
    """Injected bug: a certificate that always passes must lose copies here, or the test above proves nothing."""
    import impurityModel.ed.eigensolvers as eigensolvers

    monkeypatch.setattr(
        eigensolvers, "_level_overlap_ranks", lambda F, groups, W, mpi, comm, floor: [10**6] * len(groups)
    )
    eig = _window_solve(system, n_random=2, warm=False)
    assert eig.n_in < N_WINDOW


@pytest.mark.mpi
def test_the_multiplicity_found_sizes_the_next_solves_random_block(system):
    """A second solve with ``R = max_multiplicity + 1`` certifies without any fallback."""
    import impurityModel.ed.eigensolvers as eigensolvers

    first = _window_solve(system, n_random=2, warm=False)
    real = eigensolvers.trlm
    locked_calls = []

    def counting(*args, **kwargs):
        locked_calls.append(kwargs.get("locked") is not None)
        return real(*args, **kwargs)

    eigensolvers.trlm = counting
    try:
        second = _window_solve(system, n_random=first.max_multiplicity + 1, warm=False)
    finally:
        eigensolvers.trlm = real
    assert second.certified and second.n_in == N_WINDOW
    assert locked_calls == [False]


def test_trlm_with_a_locked_set_solves_in_its_complement():
    """``locked`` restricts TRLM to the orthogonal complement: it returns the next states up.

    Locks the ground level and part of the degenerate one; the solve must return the remaining
    copies first, then the rest of the spectrum, all exact, and none of it may overlap the locked
    set. The start block is as wide as the remaining multiplicity (7), since a block of width p
    reaches at most p copies of a degenerate level -- the limit the completeness probe works around
    by locking and probing again. ``m`` is small so the answer has to come through the restart
    continuation; the locked set used to leak back in there (the ground level reappeared as the
    "lowest" state) when the continuation projected against it before the CGS against its own basis.
    """
    from impurityModel.ed.trlm import trlm

    rng = np.random.default_rng(5)
    n = 120
    evals = np.concatenate([np.zeros(GROUND), np.full(EXCITED, E_EXCITED), np.linspace(0.12, 3.0, n - N_WINDOW)])
    U = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    h = sps.csr_matrix(0.5 * ((U * evals) @ U.conj().T + ((U * evals) @ U.conj().T).conj().T))
    n_locked = GROUND + 3
    locked = U[:, :n_locked]
    psi0 = rng.standard_normal((n, 7)) + 1j * rng.standard_normal((n, 7))

    vals, vecs, status, _ = trlm(
        psi0, h, None, num_wanted=9, max_subspace_blocks=4, tol=1e-10, max_restarts=200, locked=locked
    )

    assert status == "converged" or status.endswith("converged"), status
    np.testing.assert_allclose(np.sort(vals.real), evals[n_locked : n_locked + 9], atol=1e-9)
    np.testing.assert_allclose(locked.conj().T @ vecs, 0, atol=1e-10)


def test_the_vectorized_probe_hash_matches_the_scalar_one():
    """The probe's start block must be the same pseudo-random numbers at any rank count and on any
    platform, so its vectorized splitmix64 has to agree with the scalar one bit for bit."""
    from impurityModel.ed.eigensolvers import _splitmix64, _splitmix64_array

    x = np.random.default_rng(0).integers(0, 2**63, 1000, dtype=np.int64).astype(np.uint64) * np.uint64(
        2
    ) + np.uint64(1)
    x = np.concatenate([x, np.array([0, 1, 2**64 - 1], dtype=np.uint64)])
    np.testing.assert_array_equal(_splitmix64_array(x), np.array([_splitmix64(int(v)) for v in x], dtype=np.uint64))


def _lowest_solve(system, *, warm, calls=None, monkeypatch=None):
    import impurityModel.ed.eigensolvers as eigensolvers

    basis, h, U, evals = system
    comm = basis.comm if basis.is_distributed else None
    h_dist = DistributedSparseMatrix.from_scipy(sps.csc_array(h)[:, list(basis.local_indices)], comm, basis.offset)
    hashes = np.fromiter((state.get_hash() for state in basis.local_basis), dtype=np.uint64)
    local = np.asarray(list(basis.local_indices), dtype=int)
    if calls is not None:
        real = eigensolvers.trlm

        def counting(psi0, *args, **kwargs):
            calls.append(psi0.shape[1])
            return real(psi0, *args, **kwargs)

        monkeypatch.setattr(eigensolvers, "trlm", counting)
    return eigensolvers.solve_lowest(
        h_dist,
        comm,
        hashes,
        warm=np.ascontiguousarray(U[local, :GROUND]) if warm else None,
        tol=1e-11,
        n_global=len(basis),
    )


@pytest.mark.mpi
@pytest.mark.parametrize("warm", [True, False], ids=["warm", "cold"])
def test_lowest_returns_one_pair_of_a_degenerate_level_in_one_narrow_solve(system, warm, monkeypatch):
    """A 5-fold ground level: the energy is exact, one eigensolve runs, and its block is at most two wide."""
    calls = []
    eig = _lowest_solve(system, warm=warm, calls=calls, monkeypatch=monkeypatch)
    assert eig.certified
    assert len(eig.e) == 1 and eig.vecs.shape[1] == 1 and eig.n_in == 1
    np.testing.assert_allclose(eig.e[0], system[3][0], atol=1e-9)
    assert len(calls) == 1 and calls[0] <= 2


def test_within_thermal_is_the_energy_cut():
    from impurityModel.ed.average import energy_cut
    from impurityModel.ed.eigensolvers import Lowest, Within

    assert Within.thermal(0.0025).max_energy == energy_cut(0.0025)
    assert Within(0.1) == Within(0.1) and Within(0.1) != Lowest()


@pytest.mark.mpi
def test_the_dense_branch_keeps_a_degenerate_group_straddling_the_window_edge_whole(system):
    """The dense branch cuts with the same rule as the Krylov branch: a window edge inside the 10-fold
    excited level absorbs the whole level instead of bisecting it (``eigensystem``'s own cut did)."""
    basis, h, U, evals = system
    ground = _solve(basis, h, U, Within(E_EXCITED / 2), dense_cutoff=10**6)[0]
    assert len(ground) == GROUND
    # The edge sits exactly on the excited level, where roundoff scatters its copies to either side of it.
    inside = _solve(basis, h, U, Within(E_EXCITED), dense_cutoff=10**6)[0]
    assert len(inside) == N_WINDOW
    np.testing.assert_allclose(np.real(inside), evals[:N_WINDOW], atol=1e-9)


@pytest.mark.mpi
def test_a_lowest_result_is_read_back_not_solved_again(system, monkeypatch):
    """``reuse_last`` serves a ``Lowest`` request from the previous ``Lowest`` solve (same generation)."""
    import impurityModel.ed.cipsi_solver as cs

    basis, h, U, _evals = system
    solver = CIPSISolver(basis)
    dummy = ManyBodyOperator({((0, "c"), (0, "a")): 0.0})
    h_dist = DistributedSparseMatrix.from_scipy(
        sps.csc_array(h)[:, list(basis.local_indices)], basis.comm if basis.is_distributed else None, basis.offset
    )
    calls = []
    real = cs.solve_lowest
    monkeypatch.setattr(cs, "solve_lowest", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    # `reuse_last` refuses a caller-supplied h_matrix (it cannot vouch for the operator), so build it the normal way.
    monkeypatch.setattr(solver, "_build_h_matrix", lambda H: h_dist)
    first = solver.get_eigenvectors(dummy, Lowest(), dense_cutoff=10, slaterWeightMin=1e-12)
    second = solver.get_eigenvectors(dummy, Lowest(), dense_cutoff=10, slaterWeightMin=1e-12, reuse_last=True)
    assert len(calls) == 1
    np.testing.assert_allclose(np.real(first[0]), np.real(second[0]))
    # A `Lowest` result does not cover a window request.
    solver.get_eigenvectors(dummy, Within(0.0), dense_cutoff=10, slaterWeightMin=1e-12, reuse_last=True)
    assert solver.eig.manifold == Within(0.0)
