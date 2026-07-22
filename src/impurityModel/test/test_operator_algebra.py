from collections import OrderedDict

import numpy as np
import pytest

from impurityModel.ed.atomic_physics import get_spherical_2_cubic_matrix
from impurityModel.ed.operator_algebra import assert_hermitian, c2i, combineOp, daggerOp, iOpToMatrix, op2Dict
from impurityModel.ed.transition_operators import dipole_operator

# 2p first, then 3d with 10 bath states -- the shell order get_spectra uses, which makes the
# dipole operator c^dagger_3d c_2p (absorption). The projectors below act on the 3d block.
_NBATHS = OrderedDict([(1, 0), (2, 10)])
_D_LABELS = [(2, s, m) for s in range(2) for m in range(-2, 3)]
# Cubic harmonics for l=2 in the order [x2-y2, z2, xy, xz, yz] per spin.
_CUBIC_COLUMNS = {"eg": [0, 1], "t2g": [2, 3, 4]}


def _irrep_projector(name):
    """The eg or t2g projector on the 3d shell, keyed by ``(l, s, m)`` labels."""
    u = get_spherical_2_cubic_matrix(spinpol=True, l=2)  # spherical <- cubic
    cols = [c + 5 * s for s in range(2) for c in _CUBIC_COLUMNS[name]]
    p = u[:, cols] @ u[:, cols].conj().T
    return {
        ((_D_LABELS[i], "c"), (_D_LABELS[j], "a")): p[i, j]
        for i in range(10)
        for j in range(10)
        if abs(p[i, j]) > 1e-12
    }


def test_daggerOp():
    op = {((0, "c"), (1, "a")): 1.0 + 0.5j}
    dag = daggerOp(op)
    assert len(dag) == 1
    # Note: dagger reverses the tuple and swaps 'c' and 'a', conjugates value
    assert ((1, "c"), (0, "a")) in dag
    assert dag[((1, "c"), (0, "a"))] == 1.0 - 0.5j


def test_assert_hermitian():
    op_hermitian = {((0, "c"), (1, "a")): 1.0j, ((1, "c"), (0, "a")): -1.0j}
    assert_hermitian(op_hermitian)

    op_nonhermitian = {((0, "c"), (1, "a")): 1.0j}
    with pytest.raises(AssertionError):
        assert_hermitian(op_nonhermitian)


def test_op2Dict_accepts_both_spellings():
    """The labelled and bare forms of the same one-body term agree."""
    a, b = (2, 0, -2), (2, 0, -1)
    ia, ib = c2i(_NBATHS, a), c2i(_NBATHS, b)
    assert op2Dict(_NBATHS, {((a, "c"), (b, "a")): 2.0}) == {((ia, "c"), (ib, "a")): 2.0}
    assert op2Dict(_NBATHS, {(a, b): 2.0}) == {((ia, "c"), (ib, "a")): 2.0}


def test_op2Dict_keeps_bare_bath_pairs():
    """A bare bath pair must survive: ``(l, b)`` is itself a 2-tuple.

    Dispatching on ``len(element) == 2`` cannot tell the bath label ``(2, 0)`` from a
    ``(label, action)`` factor, so bare bath pairs used to be dropped silently, giving a
    projector that quietly ignored part of what the user asked for.
    """
    i, j = c2i(_NBATHS, (2, 0)), c2i(_NBATHS, (2, 1))
    assert op2Dict(_NBATHS, {((2, 0), (2, 1)): 1.0}) == {((i, "c"), (j, "a")): 1.0}


def test_op2Dict_normal_orders_and_sums_colliding_terms():
    """Anti-normal-ordered off-diagonal terms flip sign and merge with their partner.

    ``c_a c^dagger_b = -c^dagger_b c_a`` for ``a != b``, so writing both spellings with the
    same amplitude must cancel. Assigning into a plain dict would have kept only the last.
    """
    a, b = (2, 0, -2), (2, 0, -1)
    ia, ib = c2i(_NBATHS, a), c2i(_NBATHS, b)
    assert op2Dict(_NBATHS, {((a, "a"), (b, "c")): 2.0}) == {((ib, "c"), (ia, "a")): -2.0}
    collide = {((a, "c"), (b, "a")): 1.0, ((b, "a"), (a, "c")): 1.0}
    assert op2Dict(_NBATHS, collide) == {((ia, "c"), (ib, "a")): 0.0}


def test_op2Dict_rejects_terms_it_cannot_project():
    """An identity part or a two-body term is an error, not a silently mangled projector.

    ``c_a c^dagger_a = 1 - n_a`` carries a constant, and the projection is applied as a
    single-particle matrix product (:func:`combineOp`) that cannot represent a multiple of
    the identity. The old code mapped this to ``1 - val``, which is neither the operator nor
    the complement projector; two-body terms were dropped without a word.
    """
    a, b = (2, 0, -2), (2, 0, -1)
    with pytest.raises(ValueError, match="identity part"):
        op2Dict(_NBATHS, {((a, "a"), (a, "c")): 0.2})
    with pytest.raises(ValueError, match="one-body"):
        op2Dict(_NBATHS, {((a, "c"), (b, "c"), (a, "a"), (b, "a")): 1.0})
    with pytest.raises(ValueError, match="same way"):
        op2Dict(_NBATHS, {((a, "c"), b): 1.0})


def test_irrep_projectors_are_orthogonal_projectors():
    """eg/t2g come out idempotent, mutually orthogonal, and complete on the 3d shell."""
    p = {name: iOpToMatrix(_NBATHS, op2Dict(_NBATHS, _irrep_projector(name))) for name in _CUBIC_COLUMNS}
    d_first = c2i(_NBATHS, (2, 0, -2))
    d_block = np.ix_(range(d_first, d_first + 10), range(d_first, d_first + 10))
    for name, expected_trace in (("eg", 4), ("t2g", 6)):
        np.testing.assert_allclose(p[name] @ p[name], p[name], atol=1e-12)
        assert np.trace(p[name]).real == pytest.approx(expected_trace, abs=1e-10)
    np.testing.assert_allclose(p["eg"] @ p["t2g"], 0, atol=1e-12)
    np.testing.assert_allclose((p["eg"] + p["t2g"])[d_block], np.eye(10), atol=1e-12)


def test_irrep_projected_transition_operators_sum_to_the_total():
    """The point of the XAS/RIXS projectors: the irrep pieces must add back up.

    ``P_eg + P_t2g`` is the identity on the 3d shell and the dipole operator creates only
    there, so projecting onto the two irreps and summing has to return the unprojected
    transition operator exactly. This is the end-to-end check that the label conversion,
    the normal ordering and the matrix product all agree.
    """
    t_op = dipole_operator(_NBATHS, [0, 0, 1])
    projected = [combineOp(_NBATHS, op2Dict(_NBATHS, _irrep_projector(name)), t_op) for name in _CUBIC_COLUMNS]
    reconstructed = sum(iOpToMatrix(_NBATHS, p) for p in projected)
    np.testing.assert_allclose(reconstructed, iOpToMatrix(_NBATHS, t_op), atol=1e-14)
    # ...and neither irrep is trivially the whole thing.
    for p in projected:
        assert 0 < np.max(np.abs(iOpToMatrix(_NBATHS, p))) < np.max(np.abs(iOpToMatrix(_NBATHS, t_op))) + 1e-12
