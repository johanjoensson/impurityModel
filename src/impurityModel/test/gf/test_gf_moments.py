"""Direct correctness tests for get_greens_function_moments (greens_function.py:141)
against an independent dense-Lehmann-sum oracle built from full diagonalization of
small toy models.

test_selfenergy.py already has two analytic references (non-interacting resolvent
powers, and the Hubbard-atom two-pole formula), but both are degenerate special
cases: a single eigenstate (or a manifold with equal energies, so tau never actually
enters the weights) where the two impurity indices are Sz-related and the off-diagonal
block is trivially zero by symmetry. This file exercises the gaps: multi-order
shape/hermiticity checks, a model with genuine inter-orbital mixing (so the
off-diagonal block is not just correctly zero but correctly *nonzero*), finite-tau
thermal averaging over non-degenerate eigenstates, and MPI rank-invariance.

The oracle (``_dense_lehmann_setup`` / ``_lehmann_moments``) implements the same
M_n[a,b] = (1/Z) sum_e w_e ( <e|c_a (H-E_e)^n c_b^dagger|e>
                              + (-1)^n <e|c_b^dagger (H-E_e)^n c_a|e> )
formula from the function's own docstring, but evaluates it as a dense double sum
over *all* exact eigenstates of a fully diagonalized toy Hamiltonian, rather than via
the production code's (H-E)^n Krylov recursion -- an independent computation path
that shares no code with greens_function.py's side_krylov_moments.
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.basis_transcription import build_dense_matrix
from impurityModel.ed.greens_function import get_greens_function_moments
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState

from impurityModel.test.support.gf_oracles import _BATHS, _IMP, _det, _siam_6


def _full_fock_dets(n_orb):
    """All 2**n_orb determinants over orbitals 0..n_orb-1, paired with their particle count."""
    dets = []
    n_of_det = []
    for n in range(n_orb + 1):
        for combo in itertools.combinations(range(n_orb), n):
            dets.append(_det(combo))
            n_of_det.append(n)
    return dets, n_of_det


def _siam_offdiag():
    """2-orbital impurity with a direct inter-orbital hop t01 (so the impurity block is
    not Sz-diagonal, unlike _siam_6) plus one bath orbital per impurity orbital and a
    density-density U between the two impurity orbitals. The U term matters: without any
    interaction the GF moments reduce to single-particle quantities that are exactly
    state-independent (as the existing non-interacting analytic test already covers), so a
    purely quadratic model here couldn't exercise genuine multi-eigenstate thermal
    averaging -- adding U makes the moments state-dependent."""
    eps0, eps1, t01, ebath, v, U = -0.3, 0.6, 0.25, 1.2, 0.4, 2.0
    terms = {
        ((0, "c"), (0, "a")): eps0,
        ((1, "c"), (1, "a")): eps1,
        ((0, "c"), (1, "a")): t01,
        ((1, "c"), (0, "a")): t01,
        ((2, "c"), (2, "a")): ebath,
        ((3, "c"), (3, "a")): ebath,
        ((0, "c"), (2, "a")): v,
        ((2, "c"), (0, "a")): v,
        ((1, "c"), (3, "a")): v,
        ((3, "c"), (1, "a")): v,
        ((0, "c"), (1, "c"), (1, "a"), (0, "a")): U,
    }
    return ManyBodyOperator(terms)


_OFFDIAG_IMP = {0: [[0, 1]]}
_OFFDIAG_BATHS = ({0: [[2, 3]]}, {0: [[]]})


def _dense_lehmann_setup(hOp, impurity_orbitals, bath_states, n_orb, impurity_indices, comm=None):
    """Full diagonalization of ``hOp`` over the whole ``n_orb``-orbital Fock space.

    Returns ``(dets, eigvals, eigvecs, n_of_eig, C, Cdag)``: eigenvalues/eigenvectors of the
    *single* diagonalization (reused everywhere below -- a second, separate ``eigh`` call on
    the same matrix is not guaranteed to return the same rotation within a degenerate
    subspace), each eigenstate's (near-integer) particle number, and the dense
    annihilation/creation matrices for ``impurity_indices``, rotated into the H-eigenbasis
    (``C[orb][m, m']`` = ``<m|c_orb|m'>``).
    """
    unsorted_dets, n_of_det = _full_fock_dets(n_orb)
    basis = Basis(
        impurity_orbitals, bath_states, initial_basis=unsorted_dets, comm=comm or MPI.COMM_SELF, verbose=False
    )
    index = basis._index_dict
    # Basis sorts/reindexes initial_basis internally, so eigvecs' row order follows
    # basis.local_basis, NOT the caller's construction order -- return that ordering
    # (not unsorted_dets) so eigvecs[:, m] can be zipped back into a ManyBodyState correctly.
    dets = list(basis.local_basis)
    n_of_index = np.zeros(len(dets), dtype=float)
    for det, n in zip(unsorted_dets, n_of_det):
        n_of_index[index[det]] = n

    H = np.asarray(build_dense_matrix(basis, hOp))
    eigvals, eigvecs = np.linalg.eigh(H)
    # n_of_index is indexed by basis (row) position; contract over that axis to get each
    # eigenstate's (column) particle-number expectation -- not eigvecs @ n_of_index, which
    # contracts over the wrong (eigenstate) axis and silently mislabels every sector.
    n_of_eig = n_of_index @ np.abs(eigvecs) ** 2

    C, Cdag = {}, {}
    for orb in impurity_indices:
        c_dense = np.asarray(build_dense_matrix(basis, ManyBodyOperator({((orb, "a"),): 1.0})))
        cdag_dense = np.asarray(build_dense_matrix(basis, ManyBodyOperator({((orb, "c"),): 1.0})))
        C[orb] = eigvecs.conj().T @ c_dense @ eigvecs
        Cdag[orb] = eigvecs.conj().T @ cdag_dense @ eigvecs
    return dets, eigvals, eigvecs, n_of_eig, C, Cdag


def _lehmann_moments(e_indices, tau, max_order, impurity_indices, eigvals, C, Cdag):
    """Dense-Lehmann-sum reference for M_n, see module docstring for the formula."""
    es_sel = eigvals[e_indices]
    w = np.exp(-(es_sel - es_sel.min()) / tau)
    Z = w.sum()
    n_corr = len(impurity_indices)
    M = np.zeros((max_order + 1, n_corr, n_corr), dtype=complex)
    M[0] = np.eye(n_corr)
    for order in range(1, max_order + 1):
        acc = np.zeros((n_corr, n_corr), dtype=complex)
        for wi, ie in zip(w, e_indices):
            dEn = (eigvals - eigvals[ie]) ** order
            for ai, a in enumerate(impurity_indices):
                for bi, b in enumerate(impurity_indices):
                    term1 = np.sum(C[a][ie, :] * dEn * Cdag[b][:, ie])
                    term2 = np.sum(Cdag[b][ie, :] * dEn * C[a][:, ie])
                    acc[ai, bi] += wi * (term1 + ((-1) ** order) * term2)
        M[order] = acc / Z
    return M


def _psis_from_eigvecs(dets, eigvecs, e_indices):
    return [
        ManyBodyState({det: complex(amp) for det, amp in zip(dets, eigvecs[:, ie]) if abs(amp) > 1e-13})
        for ie in e_indices
    ]


def _ground_index_in_sector(n_of_eig, eigvals, n_target):
    in_sector = np.flatnonzero(np.round(n_of_eig) == n_target)
    assert in_sector.size > 0
    return int(in_sector[np.argmin(eigvals[in_sector])])


def _low_indices_in_sector(n_of_eig, eigvals, n_target, count):
    in_sector = np.flatnonzero(np.round(n_of_eig) == n_target)
    assert in_sector.size >= count
    order = in_sector[np.argsort(eigvals[in_sector])]
    return order[:count].tolist()


@pytest.mark.parametrize("max_order", [0, 1, 2, 3])
def test_gf_moments_shapes_and_hermiticity(max_order):
    """Ground state of the N=3 sector of _siam_6, a single eigenstate: shape, M[0]=I,
    and Hermiticity of M[n] hold for every max_order, and the function agrees with the
    dense-Lehmann oracle exactly (both are exact for a genuine eigenstate)."""
    impurity_indices = [0, 1]
    dets, eigvals, eigvecs, n_of_eig, C, Cdag = _dense_lehmann_setup(_siam_6(), _IMP, _BATHS, 6, impurity_indices)
    ie = _ground_index_in_sector(n_of_eig, eigvals, 3)

    psis = _psis_from_eigvecs(dets, eigvecs, [ie])
    basis = Basis(_IMP, _BATHS, initial_basis=list(psis[0].keys()), comm=MPI.COMM_SELF, verbose=False)
    M = get_greens_function_moments(
        psis, [eigvals[ie]], tau=1.0, basis=basis, hOp=_siam_6(), impurity_indices=impurity_indices, max_order=max_order
    )
    M_ref = _lehmann_moments([ie], 1.0, max_order, impurity_indices, eigvals, C, Cdag)

    assert M.shape == (max_order + 1, 2, 2)
    np.testing.assert_allclose(M[0], np.eye(2), atol=1e-10)
    for n in range(max_order + 1):
        np.testing.assert_allclose(M[n], M[n].conj().T, atol=1e-9, err_msg=f"order {n} not Hermitian")
    np.testing.assert_allclose(M, M_ref, atol=1e-9)


def test_gf_moments_offdiagonal_impurity_indices():
    """A model with a genuine inter-orbital hop (t01): the off-diagonal moment block is
    not just correctly zero (as it trivially is for the Sz-block-diagonal _siam_6) but
    correctly *nonzero*, and matches the dense-Lehmann oracle."""
    impurity_indices = [0, 1]
    hOp = _siam_offdiag()
    dets, eigvals, eigvecs, n_of_eig, C, Cdag = _dense_lehmann_setup(
        hOp, _OFFDIAG_IMP, _OFFDIAG_BATHS, 4, impurity_indices
    )
    ie = _ground_index_in_sector(n_of_eig, eigvals, 2)

    psis = _psis_from_eigvecs(dets, eigvecs, [ie])
    basis = Basis(_OFFDIAG_IMP, _OFFDIAG_BATHS, initial_basis=list(psis[0].keys()), comm=MPI.COMM_SELF, verbose=False)
    M = get_greens_function_moments(
        psis, [eigvals[ie]], tau=1.0, basis=basis, hOp=hOp, impurity_indices=impurity_indices, max_order=3
    )
    M_ref = _lehmann_moments([ie], 1.0, 3, impurity_indices, eigvals, C, Cdag)

    # Genuinely nonzero -- this is the point of the test, not an incidental property.
    assert abs(M[1][0, 1]) > 1e-3
    np.testing.assert_allclose(M, M_ref, atol=1e-9)


def test_gf_moments_thermal_averaging_matches_weighted_lehmann():
    """Three non-degenerate low-lying eigenstates of the N=2 sector fed in together with
    a finite tau: the thermally-averaged moments must match the weighted-Lehmann oracle,
    and must differ from the T=0 (ground-state-only) result -- proving tau is actually
    exercised rather than the ensemble silently collapsing to the ground state."""
    impurity_indices = [0, 1]
    hOp = _siam_offdiag()
    dets, eigvals, eigvecs, n_of_eig, C, Cdag = _dense_lehmann_setup(
        hOp, _OFFDIAG_IMP, _OFFDIAG_BATHS, 4, impurity_indices
    )
    e_indices = _low_indices_in_sector(n_of_eig, eigvals, 2, 3)
    assert len(set(np.round(eigvals[e_indices], 8))) == 3  # genuinely non-degenerate
    tau = 0.3

    psis = _psis_from_eigvecs(dets, eigvecs, e_indices)
    support = sorted({det for psi in psis for det in psi.keys()})
    basis = Basis(_OFFDIAG_IMP, _OFFDIAG_BATHS, initial_basis=support, comm=MPI.COMM_SELF, verbose=False)
    M = get_greens_function_moments(
        psis, eigvals[e_indices], tau=tau, basis=basis, hOp=hOp, impurity_indices=impurity_indices, max_order=2
    )
    M_ref = _lehmann_moments(e_indices, tau, 2, impurity_indices, eigvals, C, Cdag)
    np.testing.assert_allclose(M, M_ref, atol=1e-9)

    M_ground_only = _lehmann_moments([e_indices[0]], tau, 2, impurity_indices, eigvals, C, Cdag)
    assert not np.allclose(M[1], M_ground_only[1], atol=1e-4), "thermal average collapsed to the ground state only"


@pytest.mark.mpi
def test_gf_moments_mpi_rank_invariant():
    """The collective Allreduce path (basis.is_distributed) must return the identical
    moment tensor on every rank, and it must still match the serial dense-Lehmann
    reference computed independently on COMM_SELF."""
    comm = MPI.COMM_WORLD
    impurity_indices = [0, 1]
    dets, eigvals, eigvecs, n_of_eig, C, Cdag = _dense_lehmann_setup(
        _siam_6(), _IMP, _BATHS, 6, impurity_indices, comm=MPI.COMM_SELF
    )
    ie = _ground_index_in_sector(n_of_eig, eigvals, 3)
    M_ref = _lehmann_moments([ie], 1.0, 3, impurity_indices, eigvals, C, Cdag)

    psi = _psis_from_eigvecs(dets, eigvecs, [ie])[0]
    basis = Basis(_IMP, _BATHS, initial_basis=sorted(psi.keys()), comm=comm, verbose=False)
    owns_psi = comm.rank == 0
    psi_block = ManyBodyState.from_states([psi]) if owns_psi else ManyBodyState(width=1)
    (redistributed,) = basis.redistribute_psis(psi_block)
    psis = [redistributed.to_states()[0]]

    M = get_greens_function_moments(
        psis, [eigvals[ie]], tau=1.0, basis=basis, hOp=_siam_6(), impurity_indices=impurity_indices, max_order=3
    )

    all_M = comm.allgather(M)
    for other in all_M[1:]:
        np.testing.assert_allclose(other, all_M[0], atol=1e-12)
    np.testing.assert_allclose(M, M_ref, atol=1e-9)
