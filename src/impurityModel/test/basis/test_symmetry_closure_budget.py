"""The CIPSI symmetry closure must use real symmetries, and must never make a capped run worse.

``get_symmetry_generators`` discovers generators from the **one-body** block alone
(``extract_tensors(h_op, two_body=False)``) and gates them against that same one-body matrix.
The commutant of ``h_imp`` is generally far larger than the full Hamiltonian's: any degenerate
impurity block (a spherical d-shell, an ``eg``/``t2g`` manifold, a ``j=5/2`` level) hands over
rotations that the Coulomb tensor does not preserve.

That is not benign under a budget. The closure only enlarges the candidate set -- it never
touches the ``de2`` ranking -- so with no cap a non-symmetry merely wastes determinants. With a
binding ``truncation_threshold`` the junk images compete for the same budget and displace
determinants carrying real weight, and the "symmetry-adapted" run comes back with a **higher**
variational energy than the plain one. Adding determinants can only lower a variational energy,
so that is the signature.

``expand`` therefore re-gates every generator against the full ``H``.
"""

import numpy as np
import pytest

from impurityModel.ed.cipsi_solver import CIPSISolver, _commutes_with
from impurityModel.ed.lie_algebra import tensors_to_operator
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.symmetries import discover_one_body_symmetries

N_IMP, N_VAL, N_CON = 4, 4, 4
N_ORB = N_IMP + N_VAL + N_CON


def _one_body():
    """A *degenerate* impurity block -- so its commutant is the whole of u(4) -- hybridised with
    a valence and a conduction bath orbital per impurity orbital."""
    h = np.zeros((N_ORB, N_ORB), dtype=complex)
    for orb in range(N_IMP):
        h[orb, orb] = -0.5
    for orb in range(N_IMP, N_IMP + N_VAL):
        h[orb, orb] = -2.0
    for orb in range(N_IMP + N_VAL, N_ORB):
        h[orb, orb] = 2.0
    for orb in range(N_IMP):
        for partner in (N_IMP + orb, N_IMP + N_VAL + orb):
            h[orb, partner] = h[partner, orb] = 0.4
    return h


def _hamiltonian(orbital_dependent):
    """``h + U``. ``orbital_dependent`` decides whether ``U`` breaks the one-body degeneracy."""
    h = _one_body()
    u_terms = {}
    for i in range(N_IMP):
        for j in range(N_IMP):
            if i == j:
                continue
            u_terms[((i, "c"), (j, "c"), (j, "a"), (i, "a"))] = 2.0 + (0.5 * i if orbital_dependent else 0.0)
    return tensors_to_operator(h) + ManyBodyOperator(u_terms), h


def _discovered(h):
    generators = discover_one_body_symmetries(h[np.ix_(range(N_IMP), range(N_IMP))])
    assert generators, "fixture no longer produces one-body symmetries -- the test would be vacuous"
    return generators


def _solve(h_full, generators, threshold):
    basis = Basis(
        {0: [list(range(N_IMP))]},
        ({0: [list(range(N_IMP, N_IMP + N_VAL))]}, {0: [list(range(N_IMP + N_VAL, N_ORB))]}),
        nominal_impurity_occ={0: 2},
        mixed_valence={0: 2},
        tau=1e-3,
        truncation_threshold=threshold,
        verbose=False,
        comm=None,
    )
    solver = CIPSISolver(basis)
    solver.expand(h_full, dense_cutoff=1000, de2_min=1e-10, slaterWeightMin=1e-12, symmetry_generators=generators)
    es, _psis = solver.get_eigenvectors(h_full, num_wanted=1, dense_cutoff=1000, slaterWeightMin=1e-12)
    return float(np.min(es)), basis.size


def test_the_gate_rejects_generators_that_u_does_not_preserve():
    """The one-body commutant over-counts, and the gate is what notices."""
    h_full, h = _hamiltonian(orbital_dependent=True)
    generators = _discovered(h)

    kept = [g for g in generators if _commutes_with(h_full, tensors_to_operator(g, tol=1e-12))]

    assert len(kept) < len(generators), (
        f"all {len(generators)} generators discovered from the one-body block were accepted "
        "against a Hamiltonian whose orbital-dependent U breaks that degeneracy"
    )


def test_the_gate_accepts_a_genuine_symmetry():
    """And it is not simply rejecting everything.

    Checked on a hand-built symmetry rather than a discovered one, because an *impurity-only*
    rotation is never a symmetry of a hybridised model: mixing impurity orbitals 0 and 1 has to
    mix their bath partners too, which is what ``get_symmetry_generators``'s bath propagation is
    for. The total number operator needs no such propagation -- it commutes with every term.
    """
    h_full, _h = _hamiltonian(orbital_dependent=True)

    total_number = tensors_to_operator(np.eye(N_ORB, dtype=complex), tol=1e-12)
    impurity_only_rotation = np.zeros((N_ORB, N_ORB), dtype=complex)
    impurity_only_rotation[0, 1] = impurity_only_rotation[1, 0] = 1.0

    assert _commutes_with(h_full, total_number), "the gate rejected the total number operator"
    assert not _commutes_with(h_full, tensors_to_operator(impurity_only_rotation, tol=1e-12)), (
        "the gate accepted an impurity-only rotation, which cannot commute with a hybridization "
        "that pairs each impurity orbital with its own bath orbital"
    )


@pytest.mark.parametrize("threshold", [60, 120, np.inf])
@pytest.mark.parametrize("orbital_dependent", [True, False])
def test_the_closure_never_raises_the_ground_state_energy(threshold, orbital_dependent):
    """The variational property, at and below the cap.

    Before the gate, ``threshold=60`` with an orbital-dependent ``U`` gave ``E0 = -6.83305657``
    with the closure against ``-6.83306211`` without it, and left the basis at 42 determinants
    against a cap of 60 -- room it truncated for and then never used.
    """
    h_full, h = _hamiltonian(orbital_dependent)
    generators = _discovered(h)

    e_off, size_off = _solve(h_full, [], threshold)
    e_on, size_on = _solve(h_full, generators, threshold)

    assert e_on <= e_off + 1e-10, (
        f"threshold={threshold}, orbital_dependent={orbital_dependent}: the closure raised E0 "
        f"from {e_off:.10f} to {e_on:.10f}"
    )
    if np.isfinite(threshold):
        assert size_on <= threshold
        assert size_on >= size_off, (
            f"the closure run kept {size_on} determinants where the plain run kept {size_off}; "
            "a closure that only adds candidates cannot end up smaller unless it over-truncated"
        )
