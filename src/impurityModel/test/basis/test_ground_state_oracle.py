"""The ground state against a dense oracle: every sector the run solves, the winner and its window.

Small random impurity models (``support/gs_oracle.py``) are diagonalized exactly for every electron
count; ``solve_ground_state`` must agree with that for each sector it solves, not only for the winner
(``doc/plans/gs_review_2026-10.md``, the oracle). ``PASSING`` are models that agree today; models the
review found wrong are added to the lists below by the commit that fixes them.
"""

import pytest

from impurityModel.test.support import gs_oracle
from impurityModel.test.support.comms import COMMS

OPTIONS = {"hf": dict(use_hf_seed=True), "scan": dict(use_hf_seed=False)}

# (seed, index, complex_h): real and complex, small (about 0.2 s each) and mid-sized (about 1 s).
PASSING = [(1, 3, False), (1, 5, False), (1, 12, False), (1, 13, False), (2, 0, True), (2, 3, True), (2, 6, True)]
# Models the review found wrong, by the fix that repairs them.
FIXED_BY_P7 = []
FIXED_BY_S1 = []


@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("option", sorted(OPTIONS))
@pytest.mark.parametrize("seed, index, complex_h", PASSING + FIXED_BY_P7 + FIXED_BY_S1)
def test_every_solved_sector_winner_and_window_match_the_dense_answer(comm, option, seed, index, complex_h):
    h_op, imp, bath, N0, n_orb, _n_val, _params = gs_oracle.pinned_model(seed, index, complex_h)
    failures = gs_oracle.check_model(h_op, imp, bath, N0, n_orb, comm=comm, **OPTIONS[option])
    assert not failures, failures
