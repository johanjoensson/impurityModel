"""The ground state against a dense oracle: every sector the run solves, the winner and its window.

Small random impurity models (``support/gs_oracle.py``) are diagonalized exactly for every electron
count; ``solve_ground_state`` must agree with that for each sector it solves, not only for the winner
(``doc/plans/gs_review_2026-10.md``, the oracle). ``PASSING`` are models that agree today; models the
review found wrong are added to the lists below by the commit that fixes them.
"""

import numpy as np
import pytest

from impurityModel.ed import groundstate
from impurityModel.ed.lie_algebra import tensors_to_operator
from impurityModel.test.support import gs_oracle
from impurityModel.test.support.comms import COMMS

OPTIONS = {"hf": dict(use_hf_seed=True), "scan": dict(use_hf_seed=False)}

# (seed, index, complex_h): real and complex, small (about 0.2 s each) and mid-sized (about 1 s).
PASSING = [(1, 3, False), (1, 5, False), (1, 12, False), (1, 13, False), (2, 0, True), (2, 3, True), (2, 6, True)]
# Models the review found wrong, by the fix that repairs them.
FIXED_BY_P7 = [(1, 37, False), (2, 2, True)]
# (1, 40) and (1, 54) are also P7 cases (the scan picks the wrong sector); S1 leaves a non-winning sector off.
FIXED_BY_S1 = [(11, 77, False), (11, 80, False), (1, 40, False), (1, 54, False)]


@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("option", sorted(OPTIONS))
@pytest.mark.parametrize("seed, index, complex_h", PASSING + FIXED_BY_P7 + FIXED_BY_S1)
def test_every_solved_sector_winner_and_window_match_the_dense_answer(comm, option, seed, index, complex_h):
    h_op, imp, bath, N0, n_orb, _n_val, _params = gs_oracle.pinned_model(seed, index, complex_h)
    failures = gs_oracle.check_model(h_op, imp, bath, N0, n_orb, comm=comm, **OPTIONS[option])
    assert not failures, failures


# P7: the legacy dN scan's extension guard made occupation 0 unreachable and let a group at 0 block the rest.
SCAN_CASES = [(1, 37, False), (1, 40, False), (1, 54, False), (2, 2, True)]


def _exact_total(h_op, n_orb):
    spectra = gs_oracle.exact_spectra(h_op, n_orb)
    e = {n: float(v[0]) for n, v in spectra.items() if len(v)}
    return min(e, key=e.get)


def _scan_total(h_op, imp, bath, N0, n_val, **kw):
    found = groundstate.find_ground_state_sector(
        h_op, imp, bath, dict(N0), comm=None, truncation_threshold=np.inf, use_hf_seed=False, verbose=0, **kw
    )
    return sum(found.occupation.values()) + n_val


@pytest.mark.parametrize("seed, index, complex_h", SCAN_CASES)
def test_the_scan_walk_reaches_the_exact_sector(seed, index, complex_h):
    h_op, imp, bath, N0, n_orb, n_val, _ = gs_oracle.pinned_model(seed, index, complex_h)
    assert _scan_total(h_op, imp, bath, N0, n_val) == _exact_total(h_op, n_orb)


@pytest.mark.parametrize("seed, index, complex_h", SCAN_CASES)
def test_the_scan_walk_reaches_the_exact_sector_with_two_groups_and_a_split_that_may_go_negative(
    seed, index, complex_h
):
    h_op, imp, bath, N0, n_orb, n_val, _ = gs_oracle.pinned_model(seed, index, complex_h)
    imp2 = {0: [[0, 2]], 1: [[1, 3]]}
    bath2 = ({0: bath[0][0], 1: [[]]}, {0: bath[1][0], 1: [[]]})
    n = N0[0]
    N0_2 = {0: min(n, 2), 1: n - min(n, 2)}
    assert _scan_total(h_op, imp2, bath2, N0_2, n_val) == _exact_total(h_op, n_orb)


@pytest.mark.parametrize("seed, index, complex_h", SCAN_CASES)
def test_a_frozen_empty_shell_does_not_block_the_scan(seed, index, complex_h):
    h_op, imp, bath, N0, n_orb, n_val, _ = gs_oracle.pinned_model(seed, index, complex_h)
    shell = np.zeros((n_orb + 2, n_orb + 2))
    shell[n_orb, n_orb] = shell[n_orb + 1, n_orb + 1] = 5.0
    h_a = h_op + tensors_to_operator(shell)
    imp_a = {0: imp[0], 1: [[n_orb, n_orb + 1]]}
    bath_a = ({0: bath[0][0], 1: [[]]}, {0: bath[1][0], 1: [[]]})
    total = _scan_total(h_a, imp_a, bath_a, {0: N0[0], 1: 0}, n_val, frozen_occupations={1})
    assert total == _exact_total(h_op, n_orb)


@pytest.mark.parametrize(
    "occupation, sizes, frozen, expected",
    [
        ({0: 2, 1: 1}, {0: 4, 1: 4}, (), {0: 2, 1: 1}),
        ({0: -1, 1: 4}, {0: 4, 1: 4}, (), {0: 0, 1: 3}),
        ({0: 5, 1: -1}, {0: 4, 1: 4}, (), {0: 4, 1: 0}),
        ({0: 6, 1: 2}, {0: 4, 1: 4}, (), {0: 4, 1: 4}),
        ({0: -2, 1: 2, 2: 3}, {0: 4, 1: 4, 2: 4}, (), {0: 0, 1: 0, 2: 3}),
        ({0: -1, 1: 0, 2: 3}, {0: 4, 1: 4, 2: 4}, {1}, {0: 0, 1: 0, 2: 2}),
    ],
)
def test_canonical_split_clips_keeps_the_total_and_leaves_frozen_groups(occupation, sizes, frozen, expected):
    out = groundstate.canonical_split(occupation, sizes, frozen)
    assert out == expected
    assert sum(out.values()) == sum(occupation.values())
    assert all(out[i] == occupation[i] for i in frozen)


@pytest.mark.benchmark
@pytest.mark.parametrize("complex_h", [False, True], ids=["real", "complex"])
def test_sweep_random_models_every_sector_against_the_dense_answer(complex_h):
    """The review's sweep (opt-in, ``pytest -m benchmark``): 60 random models, both walks, every failing
    model reported at once, not the first."""
    import numpy as np

    rng = np.random.default_rng(100 + complex_h)
    failures = []
    for index in range(60):
        h_op, imp, bath, N0, n_orb, _n_val, _params = gs_oracle.random_model(rng, complex_h=complex_h)
        spectra = gs_oracle.exact_spectra(h_op, n_orb)
        for name, options in OPTIONS.items():
            found = gs_oracle.check_model(h_op, imp, bath, N0, n_orb, spectra=spectra, **options)
            if found:
                failures.append((index, name, found))
    assert not failures, "\n".join(map(str, failures))
