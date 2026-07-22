"""Algebra on ``ManyBodyOperator``: canonical form, constants, products, brackets, adjoint.

The dict-level helpers in :mod:`impurityModel.ed.operator_algebra` are covered by
``test_operator_algebra.py``; this file covers the C++/Cython operator itself.

Where possible each property is checked against something independent of the code under
test -- the fermionic anticommutation relations, a naive product loop, the dict-level
``daggerOp``, or the sequential ``observables.apply_casimir`` -- rather than against
another call into the same machinery.
"""

import itertools
import random

import numpy as np
import pytest

from impurityModel.ed.ManyBodyUtils import (
    ManyBodyOperator,
    ManyBodyState,
    SlaterDeterminant,
    anticommutator,
    commutator,
)
from impurityModel.ed.operator_algebra import daggerOp

TOL = 1e-12
N_ORB = 5


def sd(orbitals):
    """Determinant with ``orbitals`` occupied (orbital i is bit 7-i of byte i//8)."""
    data = bytearray(1)
    for o in orbitals:
        data[o // 8] |= 1 << (7 - o % 8)
    return SlaterDeterminant.from_bytes(bytes(data))


def all_determinants(n_orb=N_ORB):
    """Every determinant over ``n_orb`` orbitals -- a full basis to compare actions on."""
    for k in range(n_orb + 1):
        for orbs in itertools.combinations(range(n_orb), k):
            yield sd(orbs)


def random_operator(rng, n_terms=4, max_len=3, n_orb=N_ORB):
    """Random operator, deliberately including odd-length (non-number-conserving) strings."""
    terms = {}
    for _ in range(n_terms):
        key = tuple((rng.randrange(n_orb), rng.choice("ca")) for _ in range(rng.randrange(1, max_len + 1)))
        terms[key] = terms.get(key, 0) + complex(rng.gauss(0, 1), rng.gauss(0, 1))
    return ManyBodyOperator(terms)


def max_action_difference(op_a, op_b, n_orb=N_ORB):
    """Largest amplitude difference between two operators applied across the full basis."""
    worst = 0.0
    for state in all_determinants(n_orb):
        psi = ManyBodyState({state: 1.0 + 0j})
        left, right = op_a(psi).to_dict(), op_b(psi).to_dict()
        for key in set(left) | set(right):
            worst = max(worst, abs(left.get(key, 0) - right.get(key, 0)))
    return worst


def c(i):
    return ManyBodyOperator({((i, "a"),): 1.0 + 0j})


def c_dag(i):
    return ManyBodyOperator({((i, "c"),): 1.0 + 0j})


def n(i):
    return ManyBodyOperator({((i, "c"), (i, "a")): 1.0 + 0j})


# --------------------------------------------------------------------------- #
# Canonical form
# --------------------------------------------------------------------------- #
def test_constructor_canonicalizes():
    """c_i c^d_i is stored as 1 - n_i, and reports itself canonical."""
    op = ManyBodyOperator({((0, "a"), (0, "c")): 1.0 + 0j})
    assert op.is_canonical()
    assert op.to_dict() == {(): 1.0 + 0j, ((0, "c"), (0, "a")): -1.0 + 0j}


def test_canonicalization_preserves_the_action():
    """Rewriting to normal order must not change what the operator does to any state.

    Built through __setitem__ so the terms stay raw; the constructor would have
    canonicalized them before there was anything to compare.
    """
    rng = random.Random(4)
    for _ in range(20):
        raw = ManyBodyOperator()
        for _ in range(5):
            key = tuple((rng.randrange(N_ORB), rng.choice("ca")) for _ in range(rng.randrange(1, 4)))
            raw[key] = complex(rng.gauss(0, 1), rng.gauss(0, 1))
        assert not raw.is_canonical()

        canonical = ManyBodyOperator(raw.to_dict())
        canonical.canonicalize()
        assert canonical.is_canonical()
        assert max_action_difference(raw, canonical) < TOL


def test_setitem_breaks_and_canonicalize_restores_the_invariant():
    op = n(0)
    assert op.is_canonical()
    op[((1, "a"), (1, "c"))] = 1.0 + 0j
    assert not op.is_canonical()
    op.canonicalize()
    assert op.is_canonical()
    assert op.to_dict() == {
        (): 1.0 + 0j,
        ((0, "c"), (0, "a")): 1.0 + 0j,
        ((1, "c"), (1, "a")): -1.0 + 0j,
    }


def test_setitem_invalidates_the_flat_representation():
    """Writing a new amplitude must reach apply(), not leave it on the cached one."""
    psi = ManyBodyState({sd([0]): 1.0 + 0j})
    op = n(0)
    assert op(psi).to_dict()[sd([0])] == pytest.approx(1.0)
    op[((0, "c"), (0, "a"))] = 5.0 + 0j
    assert op(psi).to_dict()[sd([0])] == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
def test_identity_and_constant_round_trip():
    assert ManyBodyOperator.identity(2.5).to_dict() == {(): 2.5 + 0j}
    assert ManyBodyOperator.identity().to_dict() == {(): 1.0 + 0j}
    # An empty operator is ZERO, not the identity.
    assert ManyBodyOperator().is_empty()
    assert (ManyBodyOperator() * n(0)).to_dict() == {}

    op = n(0)
    assert op.constant == 0
    assert (op + 2.5).constant - op.constant == pytest.approx(2.5)
    op.constant = 3.0
    assert op.constant == pytest.approx(3.0)
    assert op.is_canonical()
    op.constant = 0.0
    assert () not in op.to_dict()


@pytest.mark.parametrize("z", [2.5, 2.5 + 1j, np.complex128(2.5 + 1j), np.float64(2.5), 3])
def test_scalar_shift_matches_the_two_term_identity_construction(z):
    """`z - H` reproduces the retired `z*(n_0 + (1 - n_0)) - H` trick exactly.

    Includes numpy scalars: a frequency read off a mesh is a numpy scalar, and without
    __array_ufunc__ = None the left operand would try to broadcast the operator.
    """
    h = ManyBodyOperator({((0, "c"), (1, "a")): 0.7, ((1, "c"), (0, "a")): 0.7})
    legacy = ManyBodyOperator({((0, "c"), (0, "a")): z, ((0, "a"), (0, "c")): z}) - h
    assert (z - h).to_dict() == legacy.to_dict()
    assert max_action_difference(z - h, legacy) < TOL


def test_scalar_addition_is_commutative_and_signed():
    op = n(0)
    assert (op + 2.0).to_dict() == (2.0 + op).to_dict()
    assert (2.0 - op).approx_equal(-(op - 2.0))
    in_place = n(0)
    in_place += 2.0
    in_place -= 0.5
    assert in_place.constant == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# Products
# --------------------------------------------------------------------------- #
def test_product_is_composition_not_its_reverse():
    """(A*B)(psi) == A(B(psi)).

    The operand-order check: terms are stored rightmost-first, so concatenating the
    factors the wrong way round yields B*A. The second assertion keeps this test honest
    by confirming the two conventions actually disagree on this input.
    """
    rng = random.Random(0)
    for _ in range(50):
        a, b = random_operator(rng), random_operator(rng)
        assert max_action_difference(a * b, _Composed(a, b)) < TOL

    a, b = c_dag(0) * n(1), c(2) + c_dag(1)
    assert max_action_difference(a * b, _Composed(b, a)) > 1e-3


class _Composed:
    """Sequential application A(B(psi)), independent of the operator product."""

    def __init__(self, outer, inner):
        self.outer, self.inner = outer, inner

    def __call__(self, psi, cutoff=0):
        return self.outer(self.inner(psi, cutoff), cutoff)


def test_matmul_is_a_synonym_for_the_product():
    rng = random.Random(1)
    a, b = random_operator(rng), random_operator(rng)
    assert (a @ b).to_dict() == (a * b).to_dict()


def test_fermionic_anticommutation_relations():
    """{c_i, c^d_j} = delta_ij, {c_i, c_j} = 0 -- the defining algebra."""
    for i, j in itertools.product(range(3), repeat=2):
        expected = ManyBodyOperator.identity() if i == j else ManyBodyOperator()
        assert anticommutator(c(i), c_dag(j)).approx_equal(expected, TOL)
        assert anticommutator(c(i), c(j)).approx_equal(ManyBodyOperator(), TOL)
        assert anticommutator(c_dag(i), c_dag(j)).approx_equal(ManyBodyOperator(), TOL)


def test_number_operator_is_a_projector():
    assert (n(0) ** 2).approx_equal(n(0), TOL)
    assert (n(0) ** 3).approx_equal(n(0), TOL)
    assert (n(0) ** 0).approx_equal(ManyBodyOperator.identity(), TOL)
    assert (c(0) * c(0)).approx_equal(ManyBodyOperator(), TOL)  # Pauli


def test_power_matches_repeated_multiplication():
    rng = random.Random(5)
    op = random_operator(rng, n_terms=3, max_len=2, n_orb=3)
    expected = ManyBodyOperator.identity()
    for k in range(4):
        assert (op**k).approx_equal(expected, TOL), k
        expected = expected * op


def test_negative_power_is_rejected():
    with pytest.raises(ValueError):
        n(0) ** -1


# --------------------------------------------------------------------------- #
# Commutator / anticommutator
# --------------------------------------------------------------------------- #
def test_brackets_match_the_naive_definition():
    """The disjoint-support skip is exact: it drops only identically-zero pairs."""
    rng = random.Random(7)
    for _ in range(100):
        a, b = random_operator(rng), random_operator(rng)

        naive_c = a * b - b * a
        naive_c.prune(1e-13)
        naive_a = a * b + b * a
        naive_a.prune(1e-13)

        assert commutator(a, b).approx_equal(naive_c, 1e-11)
        assert anticommutator(a, b).approx_equal(naive_a, 1e-11)


def test_brackets_cancel_exactly_rather_than_to_zero_coefficients():
    """[A,A] must be empty, not full of terms carrying 0j."""
    rng = random.Random(8)
    a = random_operator(rng)
    assert commutator(a, a).to_dict() == {}
    assert commutator(n(0), n(1)).to_dict() == {}
    assert commutator(n(0), ManyBodyOperator.identity(3.0)).to_dict() == {}


def test_commutator_antisymmetry_and_jacobi():
    rng = random.Random(9)
    a, b, d = (random_operator(rng, n_terms=3, max_len=2, n_orb=4) for _ in range(3))
    assert commutator(a, b).approx_equal(-commutator(b, a), 1e-11)
    jacobi = commutator(a, commutator(b, d)) + commutator(b, commutator(d, a)) + commutator(d, commutator(a, b))
    jacobi.prune(1e-10)
    assert jacobi.to_dict() == {}


def test_commutator_of_a_one_body_hamiltonian_with_an_annihilation_operator():
    r"""[H, c_k] = -sum_j h_kj c_j for a one-body H = sum_ij h_ij c^d_i c_j.

    An exact closed form to check the bracket against, and the construction the
    improved-estimator work needs. Only the terms of H touching orbital k survive, which
    is what the disjoint-support skip exploits -- though the *result* is supported away
    from k, so locality of the input is not locality of the output.
    """
    rng = random.Random(21)
    h_matrix = {(i, j): complex(rng.gauss(0, 1), rng.gauss(0, 1)) for i in range(N_ORB) for j in range(N_ORB)}
    h = ManyBodyOperator({((i, "c"), (j, "a")): v for (i, j), v in h_matrix.items()})

    for k in range(N_ORB):
        expected = ManyBodyOperator({((j, "a"),): -h_matrix[(k, j)] for j in range(N_ORB)})
        assert commutator(h, c(k)).approx_equal(expected, 1e-11)


def test_prune_drops_the_zeros_addition_leaves_behind():
    residue = n(0) - n(0)
    assert residue.to_dict() == {((0, "c"), (0, "a")): 0j}
    residue.prune(TOL)
    assert residue.to_dict() == {}


# --------------------------------------------------------------------------- #
# Adjoint
# --------------------------------------------------------------------------- #
def test_adjoint_agrees_with_the_dict_level_daggerop():
    """Cross-check against operator_algebra.daggerOp, an independent implementation."""
    rng = random.Random(3)
    for _ in range(100):
        op = random_operator(rng)
        assert op.adjoint().approx_equal(ManyBodyOperator(daggerOp(op.to_dict())), TOL)


def test_adjoint_is_an_involution_and_reverses_products():
    rng = random.Random(11)
    for _ in range(50):
        a, b = random_operator(rng), random_operator(rng)
        assert a.adjoint().adjoint().approx_equal(a, TOL)
        assert (a * b).adjoint().approx_equal(b.adjoint() * a.adjoint(), 1e-11)


def test_hermiticity():
    assert n(0).is_hermitian()
    assert not c(0).is_hermitian()
    assert c(0).adjoint().to_dict() == {((0, "c"),): 1.0 + 0j}

    rng = random.Random(12)
    for _ in range(20):
        assert random_operator(rng).hermitian_part().is_hermitian(1e-11)


# --------------------------------------------------------------------------- #
# Introspection
# --------------------------------------------------------------------------- #
def test_orbitals_and_body_rank():
    op = ManyBodyOperator({((0, "c"), (1, "c"), (1, "a"), (0, "a")): 1.0 + 0j, ((3, "c"), (2, "a")): 0.5 + 0j})
    assert op.orbitals() == (0, 1, 2, 3)
    assert op.body_rank() == 2
    assert n(0).body_rank() == 1
    assert c(0).body_rank() == 1
    assert ManyBodyOperator.identity(2.0).body_rank() == 0
    assert ManyBodyOperator().body_rank() == 0
    assert ManyBodyOperator().orbitals() == ()


def test_approx_equal_is_tolerant_where_eq_is_exact():
    perturbed = n(0) * (1 + 1e-13)
    assert perturbed != n(0)
    assert perturbed.approx_equal(n(0), 1e-9)
    assert not perturbed.approx_equal(n(0), 1e-15)
    # A key present on only one side counts as a zero coefficient there.
    assert (n(0) - n(0)).approx_equal(ManyBodyOperator(), TOL)


# --------------------------------------------------------------------------- #
# Physics cross-check
# --------------------------------------------------------------------------- #
def test_casimir_built_as_an_operator_product_matches_the_sequential_form():
    r"""J^2 = J_- J_+ + J_z^2 + J_z assembled with ``*`` must agree with apply_casimir.

    ``observables.apply_casimir`` deliberately never forms the two-body product -- it
    applies the ladder/Cartan factors to the state one after another. Building the same
    Casimir as an explicit operator product exercises the product, the scalar algebra and
    the constant against an already-trusted independent implementation, on a real
    cubic d-shell rather than a toy operator.
    """
    from impurityModel.ed.atomic_physics import get_spherical_2_cubic_matrix
    from impurityModel.ed.ManyBodyUtils import inner
    from impurityModel.ed.observables import apply_casimir, make_impurity_casimir_operators

    rot = get_spherical_2_cubic_matrix(spinpol=True, l=2)
    _, s_ops, j_ops = make_impurity_casimir_operators({0: [list(range(10))]}, rot.conj().T)

    def d_shell_state(occupied):
        data = bytearray(2)
        for o in occupied:
            data[o // 8] |= 1 << (7 - o % 8)
        return ManyBodyState({SlaterDeterminant.from_bytes(bytes(data)): 1.0 + 0j})

    # High-spin d8: t2g^6 + eg-up^2, S = 1 so <S^2> = 2.
    psi = d_shell_state([2, 3, 4, 5, 6, 7, 8, 9])

    for plus, minus, z in (s_ops, j_ops):
        casimir = minus * plus + z * z + z
        sequential = apply_casimir(psi, plus, minus, z)
        produced = casimir(psi)
        keys = set(sequential.to_dict()) | set(produced.to_dict())
        worst = max(abs(sequential.to_dict().get(k, 0) - produced.to_dict().get(k, 0)) for k in keys)
        assert worst < 1e-9

    s_plus, s_minus, s_z = s_ops
    s_squared = s_minus * s_plus + s_z * s_z + s_z
    assert float(np.real(inner(psi, s_squared(psi)))) == pytest.approx(2.0, abs=1e-9)
