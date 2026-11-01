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


def test_basis_holds_a_single_total_electron_number():
    """The property the sector comparison depends on.

    A generator that spans several total electron numbers makes ``calc_energy`` return
    ``min(E[N-1], E[N], E[N+1])`` for every trial, and ``E[N+1] - E[N]`` comes out as exactly ``0.0``.
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
        verbose=False,
    )

    histogram = _total_n_histogram(basis)
    assert len(histogram) == 1, (
        f"the generator produced determinants at total electron numbers "
        f"{sorted(histogram)}; H conserves N, so calc_energy would return the minimum over "
        "the whole window instead of the energy of the requested sector."
    )


def test_the_impurity_occupation_is_pinned_without_a_window():
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
        verbose=False,
    )
    occupations = {bin(bytes(det.to_bytearray())[0] & impurity_mask).count("1") for det in basis}

    assert occupations == {1}, f"expected the impurity pinned at 1, got {sorted(occupations)}"


def test_the_split_impurity_path_also_fixes_the_total():
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
        verbose=False,
    )

    impurity_occupations, totals = _occupations_and_totals(basis, impurity_orbs=[0, 1, 2, 3, 4, 5], n_orbitals=16)

    assert totals == {10}, f"spans total electron numbers {sorted(totals)}, expected only 10"
    # The split shell's charge still redistributes between its groups (the covalency the NiO model needs); only
    # the total is fixed.
    assert impurity_occupations == {4}, f"reached impurity occupations {sorted(impurity_occupations)}, expected {{4}}"
