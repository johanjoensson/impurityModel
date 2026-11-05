"""``get_effective_excitation_bounds`` / ``excitation_bound_restrictions``: the bounds derived from a basis."""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import product_state_representation as psr
from impurityModel.ed.basis_restrictions import (
    ExcitationBounds,
    excitation_bound_restrictions,
    get_effective_excitation_bounds,
)
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import SlaterDeterminant

_N = 8
_VAL = {0, 1, 2, 3}
_CON = {4, 5, 6, 7}
_BATHS = ({0: [[0, 1, 2, 3]]}, {0: [[4, 5, 6, 7]]})
_NO_CON = ({0: [[0, 1, 2, 3]]}, {0: []})


def _bytes(occ):
    b = bytearray((_N + 7) // 8)
    for o in occ:
        b[o // 8] |= 1 << (7 - o % 8)
    return bytes(b)


def _basis(occs, bath_states=_BATHS, impurity=None):
    return Basis(
        impurity_orbitals={0: [[]]} if impurity is None else impurity,
        bath_states=bath_states,
        initial_basis=[_bytes(o) for o in occs],
        verbose=False,
        comm=MPI.COMM_WORLD,
    )


def _occ(det):
    return {k for k, bit in enumerate(psr.bytes2bitarray(bytes(det.to_bytearray()), _N)) if bit}


# reference, one hole, one electron, hole+electron, two holes + one electron
_OCCS = [
    {0, 1, 2, 3},
    {0, 1, 2},
    {0, 1, 2, 3, 4},
    {0, 1, 2, 4},
    {0, 1, 4, 6},
]


def _brute(occs):
    holes = max(len(_VAL - o) for o in occs)
    elec = max(len(o & _CON) for o in occs)
    total = max(len(_VAL - o) + len(o & _CON) for o in occs)
    return holes, elec, total


def test_bounds_match_the_brute_force_maxima():
    bounds = get_effective_excitation_bounds(_basis(_OCCS))
    assert (bounds.max_holes, bounds.max_electrons, bounds.max_total) == _brute(_OCCS) == (2, 2, 4)
    assert bounds.max_total < bounds.max_holes + bounds.max_electrons + 1


def test_total_is_not_the_sum_of_the_maxima():
    # holes and electrons peak in different determinants: total (2) < max_holes (2) + max_electrons (2)
    occs = [{0, 1, 2}, {0, 1, 2, 3, 4, 5}]
    bounds = get_effective_excitation_bounds(_basis(occs))
    assert (bounds.max_holes, bounds.max_electrons, bounds.max_total) == (1, 2, 2)


def test_the_restrictions_admit_every_basis_determinant_and_reject_one_more():
    bounds = get_effective_excitation_bounds(_basis(_OCCS))
    entries = excitation_bound_restrictions(_BATHS, bounds)
    assert len(entries) == 3

    def admitted(occ):
        for weights, (q_min, q_max) in entries:
            q = sum(w for o, w in weights.items() if o in occ)
            if not q_min <= q <= q_max:
                return False
        return True

    assert all(admitted(o) for o in _OCCS)
    # one more hole than any determinant has (3 holes, no electron: total 3 <= 4, only the hole entry bites)
    assert not admitted({0})
    # one more electron (3 electrons, no hole)
    assert not admitted({0, 1, 2, 3, 4, 5, 6})
    # holes 2 and electrons 2 are each inside their own bounds, total 4 is the bound: still admitted
    assert admitted({0, 1, 4, 5})
    assert not admitted({0, 4, 5, 6})  # holes 3 > 2


def test_a_conduction_free_model_gets_no_electron_entry():
    bounds = get_effective_excitation_bounds(_basis([{0, 1, 2, 3}, {0, 1, 2}], _NO_CON))
    assert bounds == ExcitationBounds(max_holes=1, max_electrons=0, max_total=1, cap_bound=False)
    entries = excitation_bound_restrictions(_NO_CON, bounds)
    assert len(entries) == 2 and all(e is not None for e in entries)


def test_no_bath_orbitals_gives_none():
    assert get_effective_excitation_bounds(_basis([{0}], ({0: []}, {0: []}), impurity={0: [[0, 1]]})) is None


def test_the_cap_flag_is_carried():
    assert get_effective_excitation_bounds(_basis(_OCCS), cap_bound=True).cap_bound is True


def test_an_injected_min_for_max_would_be_caught():
    # the maxima are over the whole basis: dropping the last determinant must change the answer
    full = get_effective_excitation_bounds(_basis(_OCCS))
    short = get_effective_excitation_bounds(_basis(_OCCS[:-1]))
    assert short.max_holes < full.max_holes or short.max_total < full.max_total


@pytest.mark.mpi
def test_bounds_are_rank_invariant_and_match_the_fixed_value():
    comm = MPI.COMM_WORLD
    bounds = get_effective_excitation_bounds(_basis(_OCCS))
    assert bounds is not None
    assert (bounds.max_holes, bounds.max_electrons, bounds.max_total) == (2, 2, 4)
    assert len(set(comm.allgather((bounds.max_holes, bounds.max_electrons, bounds.max_total)))) == 1
