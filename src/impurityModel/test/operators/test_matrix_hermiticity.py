"""Occupation restrictions cannot change the assembled matrix on a basis that satisfies them.

``build_sparse_matrix`` materialises ``H`` column by column, and the restriction masks are
tested on the **output** determinant of each term (``ManyBodyOperator.cpp:654, 684, 828,
859``; the input-side tests at ``:637`` and ``:811`` are the diagonal/density path, where
in == out). Independently, the builder drops every bra that is not in the basis
(``basis_transcription.py:179``/``:198``).

Put together: on a basis whose determinants all satisfy the restrictions, *every determinant
the mask rejects is one the in-basis row filter would have dropped anyway*. The masks cannot
move a single matrix element -- pinned here as elementwise equality, not a tolerance -- and
the assembled matrix therefore inherits the operator's Hermiticity exactly rather than
approximately.

The precondition is load-bearing, and is pinned too: a basis determinant *outside* the
allowed window does desymmetrise the matrix (last test), because that ket's column keeps its
in-window rows while no in-window column ever gets a row back. Anything that assumes the
assembled matrix is Hermitian -- a conjugate-transpose fill, a triangular storage scheme, an
eigensolver that only reads one triangle -- has to be gated on the basis respecting the
restrictions, not merely on the operator being Hermitian.

A restriction that is set but never rejects anything would make all of this pass vacuously.
This model conserves both N and S_z, so a plain S_z mask never bites -- each test therefore
asserts that its mask actually rejects something, and the S_z tests switch on a spin-flip
hybridisation to give it something to reject.
"""

from itertools import combinations

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.basis_transcription import build_sparse_matrix
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp
from impurityModel.ed.manybody_basis import Basis

# 6 spin-orbitals: 0,1 = impurity (dn, up); 2,3 = bath site 1; 4,5 = bath site 2.
N_ORB = 6
_IMP = frozenset({0, 1})
_UP = (1, 3, 5)
_DN = (0, 2, 4)
# 2*S_z weights: +1 on the up orbitals, -1 on the down ones.
_SZ2_WEIGHTS = {o: 1 for o in _UP} | {o: -1 for o in _DN}
# A deliberately non-uniform weighted sum: the impurity counted double, the bath once.
_DOUBLED_IMP_WEIGHTS = {0: 2, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1}


def _sd(occupied):
    """SlaterDeterminant with ``occupied`` set, MSB-first within each byte."""
    data = bytearray((N_ORB + 7) // 8)
    for orb in occupied:
        data[orb // 8] |= 1 << (7 - (orb % 8))
    return SlaterDeterminant.from_bytes(bytes(data))


def _occ(det):
    """The set of occupied orbitals of a determinant."""
    raw = bytes(det.to_bytearray())
    return {o for o in range(N_ORB) if raw[o // 8] & (1 << (7 - (o % 8)))}


def _anderson_6(spin_flip=0.0):
    """Impurity + two bath sites: on-site energies, hybridisation, U n_dn n_up.

    ``spin_flip`` adds a spin-off-diagonal hybridisation (impurity down <-> bath up), which
    breaks S_z conservation -- without it an S_z mask has nothing to reject.
    """
    e_imp, e_b1, e_b2, v1, v2, u = -1.7, 0.4, 1.1, 0.63, -0.29, 3.1
    terms = {}
    for orb, eps in ((0, e_imp), (1, e_imp), (2, e_b1), (3, e_b1), (4, e_b2), (5, e_b2)):
        terms[((orb, "c"), (orb, "a"))] = eps
    for imp, bath, v in ((0, 2, v1), (1, 3, v1), (0, 4, v2), (1, 5, v2)):
        terms[((imp, "c"), (bath, "a"))] = v
        terms[((bath, "c"), (imp, "a"))] = np.conj(v)
    if spin_flip:
        terms[((0, "c"), (3, "a"))] = spin_flip
        terms[((3, "c"), (0, "a"))] = np.conj(spin_flip)
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = u
    return ManyBodyOperator(terms)


def _basis(dets, restrictions=None, weighted_restrictions=None, comm=MPI.COMM_SELF):
    """A ``Basis`` over an explicit determinant list."""
    return Basis(
        impurity_orbitals={0: [list(range(N_ORB))]},
        bath_states=({0: []}, {0: []}),
        initial_basis=dets,
        restrictions=restrictions,
        weighted_restrictions=weighted_restrictions,
        comm=comm,
        verbose=False,
    )


def _matrix(basis, op):
    """What ``get_eigenvectors`` builds: restrictions onto the operator, then assemble."""
    if basis.restrictions is not None:
        op.set_restrictions(basis.restrictions)
    if basis.weighted_restrictions is not None:
        op.set_weighted_restrictions(basis.weighted_restrictions)
    return build_sparse_matrix(basis, op).toarray()


def _rejected_images(op, dets, allowed):
    """Determinants ``op`` reaches from ``dets`` that the allowed-set predicate rejects.

    Computed with an *unrestricted* operator, so it measures what the mask has to reject --
    i.e. whether the restriction binds at all on this basis.
    """
    out = set()
    for det in dets:
        for image, _ in applyOp(op, ManyBodyState({det: 1.0})).items():
            if not allowed(_occ(image)):
                out.add(image)
    return out


def _two_sz(occ):
    return len(occ.intersection(_UP)) - len(occ.intersection(_DN))


def _dets_where(allowed, n_elec=3):
    return [_sd(occ) for occ in combinations(range(N_ORB), n_elec) if allowed(set(occ))]


def test_operator_is_hermitian():
    """The precondition on the operator itself, before any mask is applied."""
    assert _anderson_6().is_hermitian(1e-12)
    assert _anderson_6(spin_flip=0.21).is_hermitian(1e-12)


def test_subset_restriction_binds_and_matrix_stays_hermitian():
    """Impurity occupation pinned to [1, 2]: hops out of the window are rejected, H stays Hermitian."""
    restrictions = {_IMP: (1, 2)}

    def allowed(occ):
        return 1 <= len(occ & _IMP) <= 2

    dets = _dets_where(allowed)
    assert len(dets) > 1
    # The mask bites: the unrestricted operator reaches determinants outside the window
    # (e.g. c^dag_2 a_0 emptying a singly-occupied impurity).
    assert _rejected_images(_anderson_6(), dets, allowed)

    h = _matrix(_basis(dets, restrictions=restrictions), _anderson_6())
    assert np.max(np.abs(h - h.conj().T)) < 1e-14


def test_weighted_restriction_binds_and_matrix_stays_hermitian():
    """A non-uniform weighted sum (impurity counted double) that the hybridisation moves."""
    weighted = [(_DOUBLED_IMP_WEIGHTS, (3, 4))]

    def allowed(occ):
        return 3 <= sum(_DOUBLED_IMP_WEIGHTS[o] for o in occ) <= 4

    dets = _dets_where(allowed)
    assert len(dets) > 1
    assert _rejected_images(_anderson_6(), dets, allowed)

    h = _matrix(_basis(dets, weighted_restrictions=weighted), _anderson_6())
    assert np.max(np.abs(h - h.conj().T)) < 1e-14


def test_sz_restriction_binds_under_spin_flip_and_stays_hermitian():
    """An S_z mask only bites once a spin-flip term breaks S_z -- then it still stays Hermitian."""
    weighted = [(_SZ2_WEIGHTS, (1, 1))]

    def allowed(occ):
        return _two_sz(occ) == 1

    dets = _dets_where(allowed)
    assert len(dets) > 1
    # Without the spin flip this H conserves S_z and the mask is vacuous; with it, it bites.
    assert not _rejected_images(_anderson_6(), dets, allowed)
    assert _rejected_images(_anderson_6(spin_flip=0.21), dets, allowed)

    h = _matrix(_basis(dets, weighted_restrictions=weighted), _anderson_6(spin_flip=0.21))
    assert np.max(np.abs(h - h.conj().T)) < 1e-14


def test_both_masks_together_stay_hermitian():
    """Subset and weighted restrictions combined -- the configuration CIPSI actually runs."""
    restrictions = {_IMP: (1, 2)}
    weighted = [(_SZ2_WEIGHTS, (1, 1))]

    def allowed(occ):
        return 1 <= len(occ & _IMP) <= 2 and _two_sz(occ) == 1

    dets = _dets_where(allowed)
    assert len(dets) > 1
    assert _rejected_images(_anderson_6(spin_flip=0.21), dets, allowed)

    basis = _basis(dets, restrictions=restrictions, weighted_restrictions=weighted)
    h = _matrix(basis, _anderson_6(spin_flip=0.21))
    assert np.max(np.abs(h - h.conj().T)) < 1e-14


def test_masks_change_nothing_on_a_basis_that_satisfies_them():
    """The real guarantee: on a closed basis the mask rejects only already-dropped bras.

    Elementwise equality, not a tolerance -- every determinant the restriction rejects is
    outside the window, hence outside the basis, hence dropped by the in-basis row filter
    regardless. This is why the restricted matrix inherits the operator's Hermiticity exactly
    rather than approximately, and why the incremental assembly's conjugate-transpose fill
    cannot be perturbed by the masks.
    """
    restrictions = {_IMP: (1, 2)}
    weighted = [(_SZ2_WEIGHTS, (1, 1))]

    def allowed(occ):
        return 1 <= len(occ & _IMP) <= 2 and _two_sz(occ) == 1

    dets = _dets_where(allowed)
    op = _anderson_6(spin_flip=0.21)
    assert _rejected_images(op, dets, allowed)

    masked = _matrix(_basis(dets, restrictions=restrictions, weighted_restrictions=weighted), op)
    free = _matrix(_basis(dets), _anderson_6(spin_flip=0.21))
    assert np.array_equal(masked, free)


def test_out_of_window_basis_determinant_breaks_hermiticity():
    """The precondition, stated as a failure: a basis determinant outside the mask desymmetrises H.

    Not a defect -- the masks filter outputs, so an out-of-window *ket* still emits its
    in-window rows while no in-window ket ever emits a row back to it.
    """
    restrictions = {_IMP: (1, 2)}

    def allowed(occ):
        return 1 <= len(occ & _IMP) <= 2

    dets = _dets_where(allowed)
    # One determinant with an empty impurity: outside the window, and hybridisation-connected
    # to the in-window ones.
    intruder = _sd((2, 3, 4))
    assert not allowed(_occ(intruder))

    h = _matrix(_basis(dets + [intruder], restrictions=restrictions), _anderson_6())
    assert np.max(np.abs(h - h.conj().T)) > 1e-8


@pytest.mark.mpi
def test_restricted_matrix_is_hermitian_distributed():
    """The same guarantee for the distributed assembly (rows resolved by the routed lookup)."""
    comm = MPI.COMM_WORLD
    restrictions = {_IMP: (1, 2)}

    def allowed(occ):
        return 1 <= len(occ & _IMP) <= 2

    basis = _basis(_dets_where(allowed), restrictions=restrictions, comm=comm)
    op = _anderson_6()
    op.set_restrictions(basis.restrictions)
    # Only this rank's columns are populated; sum them to get the whole matrix.
    h_local = build_sparse_matrix(basis, op).toarray()
    h = np.zeros_like(h_local)
    comm.Allreduce(h_local, h, op=MPI.SUM)
    assert np.max(np.abs(h - h.conj().T)) < 1e-14
