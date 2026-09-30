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


def _solve(basis, h, U, warm=True, **kwargs):
    """``get_eigenvectors`` on ``h``, warm-started from the exact ground manifold (as a converged CIPSI
    cycle hands over) or, with ``warm=False``, cold: the width-1 hash vector, as ``dc_frozen`` solves."""
    psi_refs = None
    if warm:
        local = np.asarray(list(basis.local_indices), dtype=int)
        psi_refs = build_state(basis, U[local, :GROUND].T)
    solver = CIPSISolver(basis)
    dummy = ManyBodyOperator({((0, "c"), (0, "a")): 0.0})
    return solver.get_eigenvectors(
        dummy, psi_refs=psi_refs, h_matrix=h, dense_cutoff=10, slaterWeightMin=1e-12, **kwargs
    )


def _assert_orthonormal_and_ascending(basis, e_ref, psis):
    """No eigenstate returned twice, and the order callers take columns by position in."""
    e = np.real(np.asarray(e_ref))
    assert np.all(np.diff(e) >= 0), f"eigenvalues not ascending: {e}"
    gram = block_inner(psis, psis, basis.is_distributed, basis.comm)
    np.testing.assert_allclose(gram, np.eye(len(e)), atol=1e-8)


@pytest.mark.mpi
@pytest.mark.parametrize("warm", [True, False], ids=["warm", "cold"])
@pytest.mark.parametrize("solver", ["trlm", "irlm"])
def test_a_degenerate_level_wider_than_the_block_comes_back_whole(system, solver, warm):
    """Cold is the width-1 start of every ``dc_frozen`` solve: the probe must not inherit that width."""
    basis, h, U, evals = system
    e_ref, psis = _solve(basis, h, U, warm=warm, num_wanted=10, max_energy=CUT, solver=solver)
    e = np.real(e_ref)
    assert len(e) == N_WINDOW, (
        f"the thermal window holds {N_WINDOW} states ({GROUND} ground + {EXCITED} at {E_EXCITED}); "
        f"got {len(e)}: {np.round(e, 5)}"
    )
    np.testing.assert_allclose(e, evals[:N_WINDOW], atol=1e-9)
    _assert_orthonormal_and_ascending(basis, e_ref, psis)


@pytest.mark.mpi
@pytest.mark.parametrize("solver", ["trlm", "irlm"])
def test_a_count_cut_inside_a_degenerate_level_does_not_return_copies_twice(system, solver):
    """No cut, ``num_wanted=1`` -- ``calc_energy``'s occupation walk. The one state asked for belongs to
    a 5-fold level. The completeness probe once ran here, locked only that one, "found" the other
    computed copies in the complement and returned them next to their originals: 9 ground states for
    a 5-fold level, Gram error 0.76. No-cut solves are no longer probed; this pins that the path
    returns each state once."""
    basis, h, U, _evals = system
    e_ref, psis = _solve(basis, h, U, num_wanted=1, max_energy=None, solver=solver)
    e = np.real(e_ref)
    # At most the true multiplicity: complete is not promised without a cut (see get_eigenvectors).
    assert np.sum(np.abs(e) < 1e-6) <= GROUND, np.round(e, 5)
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
    e_ref, psis = _solve(basis, h, U, num_wanted=10, max_energy=CUT)
    _assert_orthonormal_and_ascending(basis, e_ref, psis)
    assert np.sum(np.abs(np.real(e_ref) - edge) < 1e-6) <= 3


@pytest.mark.mpi
def test_a_complete_window_is_certified_without_a_probe_solve(system, monkeypatch):
    """The common case -- nothing missing -- must cost the locked sweep and no probe solve.

    What this cannot catch: a sweep run without its locked set. The kept vectors here are exact
    eigenvectors, so ``H`` maps their complement into itself and an unlocked sweep stays in it anyway;
    in production the kept vectors are converged only to ``tol``, their components leak back in, and
    the probe would "find" them every time and pay for a solve to learn nothing."""
    import impurityModel.ed.cipsi_solver as cipsi_module

    basis, h, U, evals = system
    calls = []
    real = cipsi_module.thick_restart_block_lanczos

    def counting(**kwargs):
        if kwargs.get("locked") is not None:
            calls.append(kwargs["locked"].shape[1])
        return real(**kwargs)

    monkeypatch.setattr(cipsi_module, "thick_restart_block_lanczos", counting)
    e_ref, _ = _solve(basis, h, U, num_wanted=4, max_energy=E_EXCITED / 2)
    np.testing.assert_allclose(np.real(e_ref), evals[:GROUND], atol=1e-9)
    assert calls == []


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
    from impurityModel.ed.trlm import _TRLM_EXIT, thick_restart_block_lanczos

    rng = np.random.default_rng(5)
    n = 120
    evals = np.concatenate([np.zeros(GROUND), np.full(EXCITED, E_EXCITED), np.linspace(0.12, 3.0, n - N_WINDOW)])
    U = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    h = sps.csr_matrix(0.5 * ((U * evals) @ U.conj().T + ((U * evals) @ U.conj().T).conj().T))
    n_locked = GROUND + 3
    locked = U[:, :n_locked]
    psi0 = rng.standard_normal((n, 7)) + 1j * rng.standard_normal((n, 7))

    vals, vecs = thick_restart_block_lanczos(
        psi0, h, None, num_wanted=9, max_subspace_blocks=4, tol=1e-10, max_restarts=200, reort="full", locked=locked
    )

    assert _TRLM_EXIT[0].startswith("restart_loop_end") or _TRLM_EXIT[0] == "continuation_converged", _TRLM_EXIT
    np.testing.assert_allclose(np.sort(vals.real), evals[n_locked : n_locked + 9], atol=1e-9)
    np.testing.assert_allclose(locked.conj().T @ vecs, 0, atol=1e-10)


def test_the_vectorized_probe_hash_matches_the_scalar_one():
    """The probe's start block must be the same pseudo-random numbers at any rank count and on any
    platform, so its vectorized splitmix64 has to agree with the scalar one bit for bit."""
    from impurityModel.ed.cipsi_solver import _splitmix64, _splitmix64_array

    x = np.random.default_rng(0).integers(0, 2**63, 1000, dtype=np.int64).astype(np.uint64) * np.uint64(
        2
    ) + np.uint64(1)
    x = np.concatenate([x, np.array([0, 1, 2**64 - 1], dtype=np.uint64)])
    np.testing.assert_array_equal(_splitmix64_array(x), np.array([_splitmix64(int(v)) for v in x], dtype=np.uint64))
