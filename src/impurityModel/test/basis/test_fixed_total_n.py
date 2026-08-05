"""A sector basis must hold exactly one total electron number.

``calc_energy(N0)`` is used to compare *charge sectors* -- by the ground-state occupation walk
and, in ``dc_criteria``, to form ``E[N+1] - E[N]`` and ``E[N] - E[N-1]``. ``H`` conserves the
total electron number, so if the generated basis spans several total-N sectors the returned
"energy of sector N" is really the minimum over the whole window, and every trial in the window
returns the same number: the sector comparison goes blind and the peak observable collapses to
a constant.

``generate_initial_basis`` builds each determinant as
``impurity_occupation + valence_occupation + conduction_occupation`` per group, with

    impurity_occupation = nominal_occ + delta_valence - delta_conduction
    valence_occupation  = len(valence) - delta_valence
    conduction_occupation = delta_conduction

so the ``delta_*`` (dN) mechanism is total-conserving by construction -- an electron moved onto
the impurity is taken off the bath. Widening ``nominal_occ`` itself is not: it adds impurity
electrons out of nowhere. Mixed valence therefore has to be expressed through the bath.
"""

import pytest

from impurityModel.ed.basis_generation import generate_initial_basis


def _layout():
    """Two impurity spin-orbitals, two valence bath, two conduction bath -- one group."""
    impurity_orbitals = {0: [[0, 1]]}
    valence_baths = {0: [[2, 3]]}
    conduction_baths = {0: [[4, 5]]}
    return impurity_orbitals, (valence_baths, conduction_baths)


def _two_group_layout():
    """An eg/t2g-shaped split impurity -- the NiO case.

    Worth testing separately: with two unfrozen groups ``generate_initial_basis`` takes the
    ``redistribute`` branch, where each group's occupation ranges over its whole
    ``[0, group_size]`` and only the cross-group total is filtered. That is a different code
    path from the single-group one, and it is the path the golden d2-collapse regression lived
    on.
    """
    impurity_orbitals = {0: [[0, 1]], 1: [[2, 3, 4, 5]]}
    valence_baths = {0: [[6, 7]], 1: [[8, 9, 10, 11]]}
    conduction_baths = {0: [[12, 13]], 1: [[14, 15]]}
    return impurity_orbitals, (valence_baths, conduction_baths)


def _occupations_and_totals(basis, impurity_orbs, n_orbitals):
    """(impurity occupations seen, total electron numbers seen) over the whole basis."""
    impurity_occupations, totals = set(), set()
    for det in basis:
        raw = bytes(det.to_bytearray())
        bits = [(raw[o // 8] >> (7 - o % 8)) & 1 for o in range(n_orbitals)]
        impurity_occupations.add(sum(bits[o] for o in impurity_orbs))
        totals.add(sum(bits))
    return impurity_occupations, totals


def _total_n_histogram(basis, n_bytes=1):
    """Electron count of every generated determinant, as {N: multiplicity}."""
    counts = {}
    for det in basis:
        n = sum(bin(byte).count("1") for byte in bytes(det.to_bytearray()))
        counts[n] = counts.get(n, 0) + 1
    return counts


@pytest.mark.parametrize("mv", [0, 1])
def test_basis_holds_a_single_total_electron_number(mv):
    """The property the sector comparison depends on.

    With ``mixed_valence = 1`` the old generator produced determinants at three different total
    electron numbers, so ``calc_energy`` returned ``min(E[N-1], E[N], E[N+1])`` for every trial
    and ``E[N+1] - E[N]`` came out as exactly ``0.0``.
    """
    impurity_orbitals, bath_states = _layout()
    basis, _n_spin_orbitals = generate_initial_basis(
        impurity_orbitals,
        bath_states,
        delta_valence_occ=None,
        delta_conduction_occ=None,
        delta_impurity_occ=None,
        n_bytes=1,
        nominal_impurity_occ={0: 1},
        mixed_valence={0: mv},
        verbose=False,
    )

    histogram = _total_n_histogram(basis)
    assert len(histogram) == 1, (
        f"mixed_valence={mv} produced determinants at total electron numbers "
        f"{sorted(histogram)}; H conserves N, so calc_energy would return the minimum over "
        "the whole window instead of the energy of the requested sector."
    )


def test_mixed_valence_still_widens_the_impurity_occupation():
    """Fixing the total must not silently disable mixed valence.

    The impurity occupation is still allowed to fluctuate -- it just has to do so against the
    bath, which is what a charge-transfer system physically does.
    """
    impurity_orbitals, bath_states = _layout()
    impurity_mask = 0b11000000  # orbitals 0 and 1 -> bits 7 and 6 (MSB-first, see the memory)

    occupations = set()
    basis, _ = generate_initial_basis(
        impurity_orbitals,
        bath_states,
        delta_valence_occ=None,
        delta_conduction_occ=None,
        delta_impurity_occ=None,
        n_bytes=1,
        nominal_impurity_occ={0: 1},
        mixed_valence={0: 1},
        verbose=False,
    )
    for det in basis:
        occupations.add(bin(bytes(det.to_bytearray())[0] & impurity_mask).count("1"))

    assert occupations == {0, 1, 2}, (
        f"mixed_valence=1 around a nominal impurity occupation of 1 should reach impurity "
        f"occupations {{0, 1, 2}}, got {sorted(occupations)}"
    )


def test_no_mixed_valence_pins_the_impurity_occupation():
    impurity_orbitals, bath_states = _layout()
    impurity_mask = 0b11000000

    basis, _ = generate_initial_basis(
        impurity_orbitals,
        bath_states,
        delta_valence_occ=None,
        delta_conduction_occ=None,
        delta_impurity_occ=None,
        n_bytes=1,
        nominal_impurity_occ={0: 1},
        mixed_valence={0: 0},
        verbose=False,
    )
    occupations = {bin(bytes(det.to_bytearray())[0] & impurity_mask).count("1") for det in basis}

    assert occupations == {1}, f"expected the impurity pinned at 1, got {sorted(occupations)}"


@pytest.mark.parametrize("mv", [0, 1])
def test_the_split_impurity_path_also_fixes_the_total(mv):
    """The NiO-shaped case: the ``redistribute`` branch must obey the same rule.

    Here each group's nominal slides over its whole range and only the cross-group total is
    pruned, so the electron count is not fixed by construction the way it is for one group --
    it is the leaf filter that pins it.
    """
    impurity_orbitals, bath_states = _two_group_layout()
    basis, _ = generate_initial_basis(
        impurity_orbitals,
        bath_states,
        delta_valence_occ=None,
        delta_conduction_occ=None,
        delta_impurity_occ=None,
        n_bytes=2,
        nominal_impurity_occ={0: 1, 1: 3},
        mixed_valence={0: mv, 1: mv},
        verbose=False,
    )

    impurity_occupations, totals = _occupations_and_totals(basis, impurity_orbs=[0, 1, 2, 3, 4, 5], n_orbitals=16)

    assert totals == {10}, f"mixed_valence={mv} spans total electron numbers {sorted(totals)}, expected only 10"
    # And mixed valence must still do its job on this path -- the fix must not silently pin the
    # impurity charge of a split shell, which is exactly the covalency the NiO model needs.
    expected = {4} if mv == 0 else {3, 4, 5}
    assert (
        impurity_occupations == expected
    ), f"mixed_valence={mv} reached impurity occupations {sorted(impurity_occupations)}, expected {sorted(expected)}"
