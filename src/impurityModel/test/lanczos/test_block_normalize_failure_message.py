r"""``block_normalize`` must say *which* of ``tsqr``'s two failure codes fired.

:func:`impurityModel.ed.TSQR.tsqr` distinguishes two outcomes that both come back as a
non-positive rank, and its own docstring says "the callers act differently":

``k == 0``
    the block is numerically zero (:math:`\sigma_{\max} \le` ``BREAKDOWN_TOL * scale``) --
    a genuine invariant subspace / closed Krylov space, or an emptied basis.
``k == -1``
    the factor is **non-finite** -- NaN/Inf in the block, i.e. a corrupted recurrence
    upstream, which is not a statement about rank at all.

``block_normalize`` used to flatten both into the single string "Block collapsed to zero
rank". That sent a production SrMnO3 double-counting crash (the cap ladder's rung 7, a
warm-started CIPSI start block) down the wrong diagnosis, because the message named the
one cause that could not have produced it. These tests pin the two messages apart, and
pin that the block's shape travels with them.
"""

import numpy as np
import pytest

from impurityModel.ed.BlockLanczosArray import BlockBreakdown, block_normalize
from impurityModel.ed.ManyBodyUtils import ManyBodyState


def test_zero_block_reports_the_breakdown_floor():
    """An all-zero block is the ``k == 0`` case: numerically zero, not corrupted."""
    with pytest.raises(ValueError) as excinfo:
        block_normalize(np.zeros((7, 3), dtype=complex))
    message = str(excinfo.value)
    assert "zero rank" in message
    assert "numerically zero" in message
    assert "non-finite" not in message
    # The shape is what tells a reader whether an emptied basis or a genuine invariant
    # subspace is in front of them, and it is exactly what the flattened message dropped.
    assert "width 3" in message and "7 local rows" in message


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_block_is_not_reported_as_a_rank_collapse(bad):
    """NaN/Inf is the ``k == -1`` case: the message must not blame the rank."""
    block = np.eye(5, 2, dtype=complex)
    block[0, 0] = bad
    with pytest.raises(ValueError) as excinfo:
        block_normalize(block)
    message = str(excinfo.value)
    assert "non-finite" in message
    assert "NaN/Inf" in message
    assert "width 2" in message and "5 local rows" in message


def test_a_healthy_block_still_normalizes():
    """The disambiguation is on the raise path only; the success path is unchanged."""
    rng = np.random.default_rng(0)
    block = rng.normal(size=(9, 3)) + 1j * rng.normal(size=(9, 3))
    q, beta = block_normalize(block)
    assert q.shape == (9, 3)
    np.testing.assert_allclose(q.conj().T @ q, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(q @ beta, block, atol=1e-12)


def test_a_breakdown_raises_the_dedicated_type_and_is_still_a_value_error():
    """The recovery paths catch :class:`BlockBreakdown`, not ``ValueError``.

    Every caller that *recovers* from a breakdown (``irlm``'s three lock/restart sites,
    ``cipsi_solver._normalize_start_block``'s cold-start fallback) sits around a collective
    call, and the recovery is MPI-safe only because this failure is rank-symmetric -- it is
    decided by ``block_tsqr``'s ``k``, which TSQR replicates bitwise. Subclassing ``ValueError``
    keeps every handler that predates the type working.
    """
    with pytest.raises(BlockBreakdown):
        block_normalize(np.zeros((7, 3), dtype=complex))
    assert issubclass(BlockBreakdown, ValueError)


def test_a_rank_local_raise_is_not_a_breakdown():
    """The one that must NOT be recovered from, because it is not rank-symmetric.

    ``block_normalize`` also raises *before* reaching its collective: ``from_states`` rejects a
    width-0 block, and a rank owning no determinants is exactly what builds the polymorphic zero
    while its peers do not. Recovered from, that would send one rank into the fallback's own
    collectives while the others were still inside the first -- an asymmetric deadlock. It stays
    a plain ``ValueError``, so the narrowed ``except`` lets it through.
    """
    with pytest.raises(ValueError) as excinfo:
        block_normalize([ManyBodyState({})], False, None, 0.0)
    assert not isinstance(excinfo.value, BlockBreakdown)
    assert "width-1" in str(excinfo.value)


def test_the_message_formats_for_a_list_of_arrays_too():
    """``is_array`` accepts ``list[np.ndarray]``, so the failure formatter must survive one.

    ``block_cols`` read ``.shape[1]`` unconditionally on that branch, which a list does not have:
    the raise came back as ``AttributeError`` from *inside the error formatter* rather than as
    the breakdown the caller was meant to see. Columns are counted the way the rest of the module
    consumes this representation (``np.column_stack``): one per 1-D entry.
    """
    with pytest.raises(BlockBreakdown) as excinfo:
        block_normalize([np.zeros(7, dtype=complex), np.zeros(7, dtype=complex)])
    assert "width 2" in str(excinfo.value) and "7 local rows" in str(excinfo.value)
