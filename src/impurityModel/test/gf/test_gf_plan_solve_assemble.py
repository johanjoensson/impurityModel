"""``get_Greens_function`` is plan -> solve -> assemble: the pieces run alone, on a unit subset, and re-assemble.

The safety net that re-solves a few work units of a block (``doc/plans/gf_support_freeze_redesign.md``) needs
exactly this: any subset of the plan's units can be solved again and written into the full results list, and
assembling that list gives the same ``G`` as a clean full run.
"""

import numpy as np
from mpi4py import MPI

from impurityModel.ed import greens_function as gfm
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant
from impurityModel.test.support.gf_oracles import _hyb_hop

MATSUBARA = 1j * np.pi * 0.5 * (2 * np.arange(12) + 1)
OMEGA = np.linspace(-2.0, 2.0, 25)
ES = [-0.2, 0.3]
KW = dict(delta=0.1, blocks=[[0, 1]], dN=1, occ_cutoff=1e-6, slaterWeightMin=0.0)


def _inputs():
    state_bytes = [b"\xa0", b"\x50"]
    basis = Basis(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[2, 3]]}, {0: [[]]}),
        initial_basis=state_bytes,
        comm=MPI.COMM_SELF,
    )
    psis = [ManyBodyState({SlaterDeterminant.from_bytes(b): 1.0}) for b in state_bytes]
    return basis, psis


def _plan(basis, psis):
    return gfm._gf_lanczos_plan(
        psis, ES, basis, _hyb_hop(), KW["delta"], KW["blocks"], False, KW["dN"], KW["occ_cutoff"],
        KW["slaterWeightMin"], "lanczos", None, None, None, None,
    )  # fmt: skip


def _solve(plan, basis, subset=None):
    return gfm._gf_solve_units(
        plan, basis, _hyb_hop(), MATSUBARA, OMEGA, ES, KW["delta"], None, True, KW["slaterWeightMin"], False,
        subset=subset,
    )  # fmt: skip


def _assemble(plan, results):
    return gfm._gf_assemble(plan, results, KW["blocks"], MATSUBARA, OMEGA, ES, 1.0, KW["delta"], None, None)


def _same_unit(a, b):
    (al_a, be_a, r_a), (al_b, be_b, r_b) = a[:3], b[:3]
    assert len(al_a) == len(al_b)
    for x, y in zip(list(al_a) + list(be_a) + list(r_a), list(al_b) + list(be_b) + list(r_b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_the_pieces_in_sequence_are_get_greens_function():
    basis, psis = _inputs()
    gm, gr, _ = gfm.get_Greens_function(
        MATSUBARA, OMEGA, psis, ES, 1.0, basis, _hyb_hop(), verbose=False, verbose_extra=False, reort=None,
        sparse=True, **KW,
    )  # fmt: skip
    plan = _plan(basis, psis)
    assert len(plan.units) == 4
    pm, pr, _ = _assemble(plan, _solve(plan, basis))
    np.testing.assert_array_equal(pm[0], gm[0])
    np.testing.assert_array_equal(pr[0], gr[0])


def test_a_unit_subset_is_the_same_solve_and_splices_back():
    basis, psis = _inputs()
    plan = _plan(basis, psis)
    full = _solve(plan, basis)
    subset = [3, 1]
    again = _solve(plan, basis, subset=subset)
    assert len(again) == 2
    for pos, u in enumerate(subset):
        _same_unit(again[pos], full[u])
    spliced = list(full)
    for pos, u in enumerate(subset):
        spliced[u] = again[pos]
    ref = _assemble(plan, full)
    out = _assemble(plan, spliced)
    np.testing.assert_array_equal(out[0][0], ref[0][0])
    np.testing.assert_array_equal(out[1][0], ref[1][0])


def test_an_empty_subset_solves_nothing():
    basis, psis = _inputs()
    plan = _plan(basis, psis)
    assert _solve(plan, basis, subset=[]) in ([], None)
