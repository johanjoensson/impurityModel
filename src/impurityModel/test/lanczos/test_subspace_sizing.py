"""``_size_subspace``: how deep a Krylov subspace ``get_eigenvectors`` asks for.

This arithmetic sets the block count every production ground-state solve runs at, and until now
it had no test at all. The one default-suite test that reaches the call site
(``test_eigenstate_cold_retry.py``) passes ``max_subspace_blocks`` into a fake that ignores it,
and the formula itself lived twice inside ``get_eigenvectors`` -- once inline, once in a closure
-- where nothing could call it.

The interesting property is the last one here, and it is red: see the xfail.
"""

import itertools

import numpy as np
import pytest

from impurityModel.ed.cipsi_solver import _size_subspace

# Spans both regimes: the padding term wins on large bases, the basis-size bound on small ones.
CAPS = (5, 12, 30, 60, 200, 800, 1771, 20000, 63504)
WIDTHS = (1, 2, 6, 7, 11, 21, 31, 59)
REQUESTS = (1, 3, 10, 11, 20, 21, 30, 110)


def _grid():
    for cap, width, num_wanted in itertools.product(CAPS, WIDTHS, REQUESTS):
        yield cap, width, min(num_wanted + 10, cap)  # get_eigenvectors pads before sizing


def test_the_subspace_always_has_room_for_a_trailing_residual_block():
    """``blocks >= 2``: one block of Krylov vectors plus somewhere to put ``beta``.

    Both kernels require ``m > k_blocks >= 1``, and the sweep writes a trailing residual block
    that the restart loop continues from.
    """
    for cap, width, num_wanted in _grid():
        blocks, _ = _size_subspace(num_wanted, width, cap)
        assert blocks >= 2, f"cap={cap} width={width} num_wanted={num_wanted} -> {blocks} blocks"


def test_the_returned_request_always_fits_the_returned_subspace():
    """``num_wanted <= (blocks - 1) * width`` -- the invariant the clamp exists to enforce.

    Asking for more pairs than the subspace can hold is not an error the kernels raise; it comes
    back as a short return, which ``get_eigenvectors`` reads as Krylov exhaustion and answers with
    a cold retry. So this must hold by construction.
    """
    for cap, width, num_wanted in _grid():
        blocks, out = _size_subspace(num_wanted, width, cap)
        assert out <= (blocks - 1) * width, f"cap={cap} width={width} num_wanted={num_wanted}"


def test_a_request_is_never_reduced_to_nothing():
    for cap, width, num_wanted in _grid():
        _, out = _size_subspace(num_wanted, width, cap)
        assert out >= 1, f"cap={cap} width={width} num_wanted={num_wanted} -> {out}"


def test_width_zero_is_treated_as_one_rather_than_dividing_by_it():
    """Unreachable in production -- ``block_normalize`` raises on a collapsed block and the start
    block always carries the cold column -- but the guard is load-bearing if that ever changes."""
    assert _size_subspace(10, 0, 1000) == _size_subspace(10, 1, 1000)


def test_a_bigger_basis_never_buys_a_smaller_subspace():
    for width, num_wanted in itertools.product(WIDTHS, REQUESTS):
        sizes = [_size_subspace(num_wanted, width, cap)[0] for cap in sorted(CAPS) if cap >= num_wanted]
        assert sizes == sorted(sizes), f"width={width} num_wanted={num_wanted}: {sizes}"


def test_the_padding_term_alone_never_forces_a_clamp():
    """With the basis-size bound out of the way the clamp is provably inert.

    ``blocks >= 2 * ceil(2 * n / w) + 20`` gives ``(blocks - 1) * w >= 4 * n``, four times the
    request. So every case where the clamp *does* bite is the basis-size bound biting -- which is
    what the xfail below is about.
    """
    for width, num_wanted in itertools.product(WIDTHS, REQUESTS):
        cap = 10**6  # far above any bound the basis size could impose
        blocks, out = _size_subspace(num_wanted, width, cap)
        assert out == num_wanted
        assert (blocks - 1) * width >= 4 * num_wanted


@pytest.mark.xfail(strict=True, reason="the basis-size bound silently trims the certified request; see C8")
def test_the_clamp_is_a_no_op():
    """The property that actually matters, stated as the defect it is.

    ``num_wanted`` is the *certified* output of the thermal-manifold search: the whole point of
    the widening loop is to prove the kept manifold is complete, and a request trimmed on the way
    in can hand back half a manifold precisely when the code is trying hardest to certify one.
    Asserting the clamp *after* it has run is vacuous -- ``_size_subspace`` clamps ``num_wanted``
    down until it holds. The non-vacuous statement is that it never had to.

    It binds whenever the basis-size bound wins: ``(blocks - 1) * width ~ cap - 2 * width``, so
    any request above that is trimmed, and at ``cap < 3 * width`` the bound collapses to two
    blocks and the request is trimmed to ``width``. Measured over the grid here, ~30% of points
    trim. The fix is not to loosen the bound -- it is real -- but to give up ``width`` instead
    (drop warm columns from the start block) or fall through to the dense branch, neither of
    which sacrifices the certified output. Recorded in ``doc/plans/eigenstate_expansion_tightening.md``.

    Two related dead spots fall out of the same arithmetic, and are the reason this is worth a
    marker rather than a comment: the clamp always returns strictly below ``cap``, so
    ``get_eigenvectors``' ``num_wanted >= cap`` break is unreachable whenever the bound binds.
    """
    for cap, width, num_wanted in _grid():
        blocks, _ = _size_subspace(num_wanted, width, cap)
        assert (blocks - 1) * width >= num_wanted, f"cap={cap} width={width} num_wanted={num_wanted}"


def test_the_clamp_binds_only_through_the_basis_size_bound():
    """Pins *where* the defect above lives, so a fix can be checked against a specific mechanism."""
    trimmed = [(c, w, n) for c, w, n in _grid() if _size_subspace(n, w, c)[1] < n]
    assert trimmed, "premise: the grid still reaches the capped regime"
    for cap, width, num_wanted in trimmed:
        blocks, _ = _size_subspace(num_wanted, width, cap)
        assert blocks == max(2, cap // width - 1), f"cap={cap} width={width}: trimmed without the basis bound"
        assert 2 * int(np.ceil(min(max(2 * num_wanted, num_wanted + 10), cap) / width)) + 20 > blocks
