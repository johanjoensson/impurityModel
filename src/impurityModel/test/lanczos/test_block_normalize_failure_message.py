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

from impurityModel.ed.BlockLanczosArray import block_normalize


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
