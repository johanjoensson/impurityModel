"""The excited-sector window must admit the ground state it was derived from (ledger row C2).

``build_excited_restrictions`` removes the far (freeze-eligible) chain orbitals from the
valence/conduction groups into their own chain windows, then bounds the remaining *near* group.
For the conduction group that bound is an upper bound, ``n_near <= max_con``. The removed far
orbitals are (nominally) empty, so they contribute ~0 to the group occupation and the near-group
bound should stay ``max_con``. The code subtracts their *count* instead
(``basis_restrictions.py:476``), which can drive the bound to 0 and pin the near conduction bath
empty -- excluding determinants of the very ground-state basis the window was widened from.
It fires whenever ``con_change`` is set: every spectra path, RIXS, and the self-energy with ``dN``.
"""

import itertools

import pytest

from impurityModel.ed.basis_restrictions import build_excited_restrictions
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.test.support.gf_branch_oracle import _det

C2 = "C2: near-conduction upper bound subtracts the far-orbital count (doc/reviews/gf_review.md)"

N_CON = 7
N_ORB = 1 + N_CON
CON = list(range(1, N_ORB))


def _chain_op():
    """Impurity 0 hybridizes with the head of a nearest-neighbour conduction chain 1-2-...-7."""
    terms = {((0, "c"), (0, "a")): -1.0}
    for o in CON:
        terms[((o, "c"), (o, "a"))] = 3.0  # above E_F -> nominally empty conduction
    for a, b in itertools.pairwise([0, *CON]):
        terms[((a, "c"), (b, "a"))] = 0.3
        terms[((b, "c"), (a, "a"))] = 0.3
    return ManyBodyOperator(terms)


# A ground-state basis with the electron either on the impurity or hopped onto the chain head.
GS_OCCUPATIONS = [(0,), (1,)]


def _satisfies(occupied, restrictions):
    occupied = set(occupied)
    return all(lo <= len(occupied & set(key)) <= hi for key, (lo, hi) in restrictions.items())


@pytest.mark.xfail(strict=True, reason=C2)
def test_the_excited_window_admits_the_ground_state_basis():
    basis = Basis(
        impurity_orbitals={0: [[0]]},
        bath_states=({0: [[]]}, {0: [CON]}),
        initial_basis=[_det(occ, N_ORB) for occ in GS_OCCUPATIONS],
        chain_restrict=True,
        verbose=False,
    )
    # The spectra drivers' per-shell change (_shell_windows): one particle more or less.
    change = {0: (1, 1)}
    # Hop-count metric with min_dist=2 makes the chain tail freeze-eligible (as in
    # test_excitation_budget's chain fixtures); psis=None takes the nominal occupations.
    restrictions = build_excited_restrictions(
        basis, _chain_op(), psis=None, imp_change=change, con_change=change, min_dist=2, coupling_cutoff=None
    )
    assert restrictions is not None
    for occ in GS_OCCUPATIONS:
        assert _satisfies(occ, restrictions), f"ground-state determinant {occ} excluded by {restrictions}"
