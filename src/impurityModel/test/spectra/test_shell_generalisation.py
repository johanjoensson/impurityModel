"""The shell-agnostic Hamiltonian assembly, against the 2p/3d code it replaced.

Two obligations, and they pull in opposite directions:

* the L2,3 edge must come out *bit-identically* -- generalising the assembly is not allowed
  to move a single NiO number, so the reference here is rebuilt from the old
  ``get2p3dSlaterCondonUop`` and the literal ``Udd``/``Upd`` expressions rather than from a
  golden file, which would only prove the new code agrees with itself;
* another edge must actually assemble, on the shells it was told about and not on the ones a
  length-derived guess would have picked.
"""

from collections import OrderedDict
from unittest.mock import patch

import pytest

from impurityModel.ed import atomic_physics, hamiltonian_io
from impurityModel.ed.operator_algebra import addOps, c2i

FDD = (7.5, 0.0, 9.9, 0.0, 6.6)
FPP = (0.0, 0.0, 0.0)
FPD = (8.9, 0.0, 6.8)
GPD = (0.0, 5.0, 0.0, 2.8)
CTC = 1.5
SHELLS = OrderedDict({1: 0, 2: 10})
N0IMPS = OrderedDict({1: 6, 2: 8})


def _reference_2p3d_interacting_part():
    """The Coulomb + double-counting operator exactly as the 2p/3d-specific code built it."""
    u_op = atomic_physics.get2p3dSlaterCondonUop(Fdd=FDD, Fpp=FPP, Fpd=FPD, Gpd=GPD)

    # The literal expressions dc_MLFT used to carry, before they became spherical averages.
    Udd = FDD[0] - 14.0 / 441 * (FDD[2] + FDD[4])
    Upd = FPD[0] - (1 / 15.0) * GPD[1] - (3 / 70.0) * GPD[3]
    dc = {2: Udd * N0IMPS[2] + Upd * N0IMPS[1] - CTC, 1: Upd * (N0IMPS[2] + 1) - CTC}

    e_dc = {}
    for l in (2, 1):
        for s in range(2):
            for m in range(-l, l + 1):
                e_dc[(((l, s, m), "c"), ((l, s, m), "a"))] = -dc[l]
    return addOps([u_op, e_dc])


def test_l23_hamiltonian_is_bit_identical_to_the_2p3d_assembly():
    """Gate 1: identical keys, maximum difference exactly zero -- no tolerance."""
    interacting = _reference_2p3d_interacting_part()
    reference = {
        tuple((c2i(SHELLS, spin_orb), action) for spin_orb, action in process): value
        for process, value in interacting.items()
    }

    with patch.object(hamiltonian_io, "get_noninteracting_hamiltonian_operator", return_value={}):
        built = hamiltonian_io.get_hamiltonian_operator(
            SHELLS,
            SHELLS,
            (FDD, FPP, FPD, GPD),
            (N0IMPS, CTC),
            "unused.pickle",
            0,
            False,
            valence_l=2,
            xi_valence=0.096,
            core_l=1,
            xi_core=11.629,
        )

    assert set(built) == set(reference)
    assert max(abs(built[key] - reference[key]) for key in reference) == 0.0


def _noninteracting(**kwargs):
    with patch.object(hamiltonian_io, "read_h0_operator", return_value={}):
        return hamiltonian_io.get_noninteracting_hamiltonian_operator(
            OrderedDict({1: 0, 2: 10}),
            OrderedDict({1: 0, 2: 10}),
            "unused.pickle",
            0,
            False,
            **kwargs,
        )


def test_soc_and_field_land_on_the_shells_they_were_told_about():
    """Neither ``nBaths`` key order nor an array length may decide where SOC goes."""
    built = _noninteracting(valence_l=2, xi_valence=0.5, hField=(0.0, 0.0, 0.25), core_l=1, xi_core=11.0)
    reference = addOps(
        [
            atomic_physics.gethHfieldop(0.0, 0.0, 0.25, l=2),
            atomic_physics.getSOCop(11.0, l=1),
            atomic_physics.getSOCop(0.5, l=2),
        ]
    )
    assert built == reference


def test_the_term_order_is_the_historical_one():
    """Not cosmetic: the operand order is observable in the last bits of every spectrum.

    ``addOps`` builds its result by insertion and the spectra matvec accumulates terms in
    that order, so reordering the operands moves every Green's function by ~1e-8 -- measured
    on the NiO L2,3 example, where assembling the valence SOC before the field and the core
    SOC shifted XPS by 1.4e-8 with an identical Hamiltonian and identical ground state. The
    order is (Zeeman field, core SOC, valence SOC, h0); this pins it.
    """
    built = _noninteracting(valence_l=2, xi_valence=0.5, hField=(0.0, 0.0, 0.25), core_l=1, xi_core=11.0)
    reference = addOps(
        [
            atomic_physics.gethHfieldop(0.0, 0.0, 0.25, l=2),
            atomic_physics.getSOCop(11.0, l=1),
            atomic_physics.getSOCop(0.5, l=2),
        ]
    )
    assert list(built) == list(reference), "term insertion order changed"


def test_a_single_shell_model_builds_no_core_operator():
    """``core_l=None`` must omit the core SOC term, not request it with a zero amplitude."""
    with patch.object(hamiltonian_io, "read_h0_operator", return_value={}):
        built = hamiltonian_io.get_noninteracting_hamiltonian_operator(
            OrderedDict({2: 10}),
            OrderedDict({2: 10}),
            "unused.pickle",
            0,
            False,
            valence_l=2,
            xi_valence=0.5,
        )
    assert {label[0] for process in built for label, _ in process} == {2}


@pytest.mark.parametrize("core_l, valence_l", [(0, 1), (2, 3)])
def test_another_edge_assembles_on_its_own_shells(core_l, valence_l):
    """A K edge and an M4,5 edge: neither has the 3/3/4 array shape of the L2,3 case."""
    shells = OrderedDict({core_l: 0, valence_l: 6})
    n0imps = OrderedDict({core_l: 2 * (2 * core_l + 1), valence_l: 3})
    slater = (
        tuple(1.0 if k == 0 else 0.5 for k in range(2 * valence_l + 1)),
        tuple(0.2 for _ in range(2 * core_l + 1)),
        tuple(0.3 for _ in range(2 * core_l + 1)),
        tuple(0.1 for _ in range(2 * core_l + 2)),
    )

    with patch.object(hamiltonian_io, "get_noninteracting_hamiltonian_operator", return_value={}):
        built = hamiltonian_io.get_hamiltonian_operator(
            shells,
            shells,
            slater,
            (n0imps, 1.0),
            "unused.pickle",
            0,
            False,
            valence_l=valence_l,
            xi_valence=0.05,
            core_l=core_l,
            xi_core=1.0,
        )

    assert built
    # Every index must fall inside the two impurity blocks these shells define.
    n_imp = 2 * (2 * core_l + 1) + 2 * (2 * valence_l + 1)
    assert max(i for process in built for i, _ in process) < n_imp
