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


def test_the_seed_size_truncation_builds_its_orbits_with_the_spin_layout(monkeypatch):
    """A seed above the cap is cut by whole orbits; the orbits must carry the solve's spin layout (S+- closure),
    as the expansion's do (review GS10: they were built with spin_layout=None)."""
    from impurityModel.ed import groundstate
    from impurityModel.ed.eigensolvers import Within
    from impurityModel.ed.groundstate import build_basis_and_solver

    _h0, H, imp, bath = cubic_d_shell(n_bath_sets=1)
    seen = []
    real = groundstate._orbits_for

    def spy(h_op, basis, manifold, layout):
        seen.append(layout)
        return real(h_op, basis, manifold, layout)

    monkeypatch.setattr(groundstate, "_orbits_for", spy)
    marker = object()
    try:
        build_basis_and_solver(
            H, imp, bath, {0: 8}, tau=0.0, chain_restrict=False, truncation_threshold=10, comm=None, verbose=False,
            frozen_occupations=None, weighted_restrictions=None, slaterWeightMin=0, manifold=Within(1.0),
            spin_layout=marker,
        )  # fmt: skip
    except Exception:  # the marker is not a layout: building the orbits may reject it; the call is what is pinned
        pass
    assert seen and seen[0] is marker


def test_a_degraded_orbit_closure_is_reported_once(monkeypatch, capsys):
    """``Orbits.notes`` say the closure is not what the guarantee assumes; they must reach the user (review GS11)."""
    from impurityModel.ed import groundstate
    from impurityModel.ed.eigensolvers import Within

    class _Fake:
        def __init__(self):
            self.notes = ["a symmetry does not map spin pairs onto spin pairs: left out"]

    monkeypatch.setattr(groundstate, "build_orbits", lambda *a, **k: _Fake())
    monkeypatch.setattr(groundstate, "_REPORTED_ORBIT_NOTES", set())
    basis = type("B", (), {"is_distributed": False})()
    assert groundstate._orbits_for(None, basis, Within(1.0), None).notes
    groundstate._orbits_for(None, basis, Within(1.0), None)
    out = capsys.readouterr().out
    assert out.count("symmetry orbit closure is degraded") == 1
    assert "left out" in out


def test_no_orbits_means_nothing_to_report(monkeypatch):
    """``build_orbits`` returns ``None`` when there is no symmetry to close over (review GS11 regression)."""
    from impurityModel.ed import groundstate
    from impurityModel.ed.eigensolvers import Within

    monkeypatch.setattr(groundstate, "build_orbits", lambda *a, **k: None)
    basis = type("B", (), {"is_distributed": False})()
    assert groundstate._orbits_for(None, basis, Within(1.0), None) is None


def test_retain_fills_the_room_with_zero_importance_orbits_whole():
    """S2b: an orbit with no importance at all used to be dropped even when the cap had room (FeO: 163,974 ->
    47,200 determinants). The room goes to those orbits in hash order, whole."""
    _h0, H, imp, bath = cubic_d_shell(n_bath_sets=1)
    manifold = Within(0.05)
    basis, solver = build_basis_and_solver(
        H,
        imp,
        bath,
        {0: 8},
        tau=0.0,
        chain_restrict=False,
        truncation_threshold=None,
        comm=None,
        verbose=False,
        frozen_occupations=None,
        weighted_restrictions=None,
        slaterWeightMin=SWMIN_DEFAULT,
        manifold=manifold,
        dense_cutoff=500,
    )
    orbits = _orbits_for(H, basis, manifold, None)
    assert orbits is not None
    solver.orbits = orbits
    solver.expand(H, manifold, dense_cutoff=500, de2_min=0.0, e_pt2_tol=1e-8, slaterWeightMin=SWMIN_DEFAULT)
    dets = list(basis.local_basis)
    importance = np.zeros(len(dets))
    importance[:3] = 1.0  # only three determinants (a few orbits) carry weight
    rep, _ = orbits.canonical(orbits.pattern(_bits(dets, orbits.n_orb)))
    largest = max(len(orbits.members(r[None, :])) for r in np.unique(rep, axis=0))
    target = len(dets) - 1
    retained = orbits.retain(basis, dets, importance, target)
    assert len(dets) - largest <= len(retained) <= target, (len(retained), target, largest)
    basis.clear()
    basis.add_states(retained)
    assert _missing_members(solver) == 0
