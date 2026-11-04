"""Every cut of a Within expansion leaves a basis made of whole orbits, at the default slaterWeightMin."""

import numpy as np

from impurityModel.ed import cipsi_solver
from impurityModel.ed.eigensolvers import Within
from impurityModel.ed.groundstate import _orbits_for, build_basis_and_solver
from impurityModel.ed.orbits import _bits, _keys_of
from impurityModel.test.support.cubic_d_shell import cubic_d_shell

SWMIN_DEFAULT = float(np.sqrt(np.finfo(float).eps))


def _missing_members(solver):
    orb, basis = solver.orbits, solver.basis
    dets = list(basis.local_basis)
    rep, _ = orb.canonical(orb.pattern(_bits(dets, orb.n_orb)))
    keys = _keys_of(orb.members(np.unique(rep, axis=0)), orb.n_bytes)
    present = np.fromiter(basis.contains(keys), dtype=bool, count=len(keys))
    return int((~present).sum())


def test_truncate_keeps_orbits_whole_when_members_fall_below_the_slater_weight_cutoff(monkeypatch):
    """``truncate`` ranked orbits by the members present in the block's support, but ``build_state`` drops a
    row whose every amplitude is <= slaterWeightMin: a member in the basis with no row was dropped from a kept
    orbit (118 missing members on the capped cubic d-shell at the default cutoff, 0 at slaterWeightMin=0)."""
    seen = []
    original = cipsi_solver.CIPSISolver.truncate

    def spy(self, psis, e_ref=None, target=None, slaterWeightMin=0):
        out = original(self, psis, e_ref, target, slaterWeightMin)
        if self.orbits is not None:
            seen.append(_missing_members(self))
        return out

    monkeypatch.setattr(cipsi_solver.CIPSISolver, "truncate", spy)
    _h0, H, imp, bath = cubic_d_shell(n_bath_sets=2)
    manifold = Within(0.05)
    basis, solver = build_basis_and_solver(
        H,
        imp,
        bath,
        {0: 8},
        tau=0.0,
        chain_restrict=False,
        truncation_threshold=2000,
        comm=None,
        verbose=False,
        frozen_occupations=None,
        weighted_restrictions=None,
        slaterWeightMin=SWMIN_DEFAULT,
        manifold=manifold,
        dense_cutoff=500,
    )
    solver.orbits = _orbits_for(H, basis, manifold, None)
    assert solver.orbits is not None
    solver.expand(H, manifold, dense_cutoff=500, de2_min=0.0, e_pt2_tol=1e-8, slaterWeightMin=SWMIN_DEFAULT)
    assert seen, "the cap must bind: no truncate ran"
    assert max(seen) == 0, seen
