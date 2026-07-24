"""Tests for the excitation-budget weighted restriction (Phase 3a).

``basis_restrictions.excitation_budget_restriction`` turns a "max total bath excitations"
budget into a single weighted restriction consumed by
``ManyBodyOperator.set_weighted_restrictions``. Reference occupation: valence baths filled,
conduction baths empty; a hole in a filled-valence orbital or an electron in an
empty-conduction orbital each costs one (default), and the budget caps their sum.
"""

from itertools import pairwise

import numpy as np

from impurityModel.ed.basis_restrictions import build_weighted_restrictions, excitation_budget_restriction
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant


def _sd(occ, n=8):
    b = bytearray((n + 7) // 8)
    for o in occ:
        b[o // 8] |= 1 << (7 - o % 8)
    return SlaterDeterminant.from_bytes(bytes(b))


def _occset(det, n=8):
    import impurityModel.ed.product_state_representation as psr

    return {k for k, bit in enumerate(psr.bytes2bitarray(bytes(det.to_bytearray()), n)) if bit}


# 8 spin-orbitals. Valence baths (nominally filled): {0,1,2,3}. Conduction (empty): {4,5,6,7}.
_BATHS = ({0: [[0, 1, 2, 3]]}, {0: [[4, 5, 6, 7]]})
_VAL = {0, 1, 2, 3}
_CON = {4, 5, 6, 7}


def _excitations(occ):
    """Number of holes in valence + electrons in conduction."""
    return len(_VAL - occ) + len(occ & _CON)


def test_budget_weights_and_bounds():
    """The construction encodes E = C + sum w n with the documented weights/bounds."""
    weights, (q_lo, q_hi) = excitation_budget_restriction(_BATHS, budget=2)
    assert {weights[o] for o in _VAL} == {-1}
    assert {weights[o] for o in _CON} == {1}
    # C = sum of valence costs = 4; bounds = (-C, budget - C) = (-4, -2).
    assert (q_lo, q_hi) == (-4, -2)


def test_budget_filters_by_excitation_order():
    """Only determinants within the excitation budget survive the mask (exact, not a superset)."""
    from itertools import combinations

    n_total = ManyBodyOperator({((i, "c"), (i, "a")): 1.0 for i in range(8)})
    # Every 4-electron determinant of the 8 orbitals spans excitation orders 0..4.
    psi = ManyBodyState({_sd(list(occ)): 1.0 for occ in combinations(range(8), 4)})

    for budget in (0, 1, 2, 3, 4):
        rest = excitation_budget_restriction(_BATHS, budget=budget)
        n_total.set_weighted_restrictions([rest])
        survivors = [_occset(d) for d, _ in n_total(psi).items()]
        assert survivors, f"budget {budget} kept nothing"
        assert all(_excitations(o) <= budget for o in survivors), budget
        # Everything within budget is kept (the mask is exact, not a superset).
        expected = {frozenset(occ) for occ in combinations(range(8), 4) if _excitations(set(occ)) <= budget}
        assert {frozenset(o) for o in survivors} == expected


def test_budget_infinite_is_noop():
    """A budget >= the largest possible cost keeps every determinant (oracle)."""
    from itertools import combinations

    n_total = ManyBodyOperator({((i, "c"), (i, "a")): 1.0 for i in range(8)})
    psi = ManyBodyState({_sd(list(occ)): 1.0 for occ in combinations(range(8), 4)})
    all_dets = {frozenset(_occset(d)) for d, _ in n_total(psi).items()}

    rest = excitation_budget_restriction(_BATHS, budget=8)  # max cost = 4 holes + 4 electrons
    n_total.set_weighted_restrictions([rest])
    survivors = {frozenset(_occset(d)) for d, _ in n_total(psi).items()}
    assert survivors == all_dets


def test_budget_graded_cost():
    """A per-orbital cost function weights the budget (graded distance form)."""
    # Cost 2 for the outermost orbitals (3 in valence, 7 in conduction), 1 otherwise.
    cost = {3: 2, 7: 2}
    weights, (q_lo, q_hi) = excitation_budget_restriction(_BATHS, budget=3, cost_fn=lambda o: cost.get(o, 1))
    assert weights[3] == -2 and weights[7] == 2
    assert weights[0] == -1 and weights[4] == 1
    # C = 1+1+1+2 = 5; bounds = (-5, 3 - 5) = (-5, -2).
    assert (q_lo, q_hi) == (-5, -2)


def test_budget_no_baths_returns_none():
    """With no bath orbitals there is nothing to bound."""
    assert excitation_budget_restriction(({0: [[]]}, {0: [[]]}), budget=1) is None


def test_budget_composes_in_basis_expand():
    """A Basis carrying the budget keeps expand within the excitation budget."""
    from impurityModel.ed.manybody_basis import Basis

    imp = {0: [[]]}  # no impurity orbitals; all 8 are baths
    # A hopping operator that moves electrons between valence and conduction (raises excitation).
    op = ManyBodyOperator(
        {
            ((4, "c"), (3, "a")): 1.0,  # val3 -> con4 (one excitation)
            ((3, "c"), (4, "a")): 1.0,
            ((5, "c"), (2, "a")): 1.0,  # val2 -> con5 (another excitation)
            ((2, "c"), (5, "a")): 1.0,
        }
    )
    seed = [bytes(_sd([0, 1, 2, 3]).to_bytearray())]  # reference: 0 excitations
    budget1 = excitation_budget_restriction(_BATHS, budget=1)
    restricted = Basis(
        impurity_orbitals=imp,
        bath_states=_BATHS,
        initial_basis=list(seed),
        weighted_restrictions=[budget1],
        verbose=False,
    )
    restricted.expand(op)
    assert all(_excitations(_occset(d)) <= 1 for d in restricted.local_basis)

    free = Basis(impurity_orbitals=imp, bath_states=_BATHS, initial_basis=list(seed), verbose=False)
    free.expand(op)
    assert any(_excitations(_occset(d)) > 1 for d in free.local_basis)  # leaks past 1 without the budget


# ---- build_weighted_restrictions helper (what the drivers call) ----------------------


def test_build_weighted_restrictions_none_when_unset():
    """No budget -> None, so the drivers leave Basis.weighted_restrictions unset."""
    assert build_weighted_restrictions(_BATHS, excitation_budget=None) is None


def test_build_weighted_restrictions_wraps_budget_in_list():
    """A budget -> a one-element list holding the excitation-budget restriction."""
    got = build_weighted_restrictions(_BATHS, excitation_budget=2)
    assert isinstance(got, list) and len(got) == 1
    assert got[0] == excitation_budget_restriction(_BATHS, 2)


def test_build_weighted_restrictions_none_without_baths():
    """No bath orbitals -> nothing to bound even with a budget set."""
    assert build_weighted_restrictions(({0: [[]]}, {0: [[]]}), excitation_budget=1) is None


# ---- calc_gs enforcement (the path calc_selfenergy / run_spectra build the GS with) --


def _siam_with_baths():
    """A 6-orbital SIAM: imp {0,1}, valence bath {2,3} (below E_F), conduction bath {4,5}."""
    from impurityModel.ed.block_structure import BlockStructure

    ed, U, ev, ec, V = -2.0, 6.0, -4.0, 4.0, 1.0
    terms = {((o, "c"), (o, "a")): ed for o in (0, 1)}
    terms.update({((o, "c"), (o, "a")): ev for o in (2, 3)})
    terms.update({((o, "c"), (o, "a")): ec for o in (4, 5)})
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = U
    for a, b in ((0, 2), (1, 3), (0, 4), (1, 5)):
        terms[((a, "c"), (b, "a"))] = V
        terms[((b, "c"), (a, "a"))] = V
    Hop = ManyBodyOperator(terms)
    bath = ({0: [[2, 3]]}, {0: [[4, 5]]})
    bs = BlockStructure(
        blocks=[[0, 1]],
        identical_blocks=[[0]],
        transposed_blocks=[[]],
        particle_hole_blocks=[[]],
        particle_hole_transposed_blocks=[[]],
        inequivalent_blocks=[0],
    )
    return Hop, {0: [[0, 1]]}, bath, bs


def _run_calc_gs(weighted_restrictions):
    from impurityModel.ed.groundstate import calc_gs

    Hop, imp, bath, bs = _siam_with_baths()
    basis_setup = dict(
        impurity_orbitals=imp,
        bath_states=bath,
        N0={0: 1},
        mixed_valence={0: 1},
        tau=0.01,
        dense_cutoff=1000,
        spin_flip_dj=False,
        comm=None,
        truncation_threshold=100000,
        weighted_restrictions=weighted_restrictions,
    )
    return calc_gs(Hop, basis_setup, bs, np.eye(2, dtype=complex), verbose=False, slaterWeightMin=1e-12)


def _bath_excitations(det, n=6, val=(2, 3), con=(4, 5)):
    occ = _occset(det, n)
    return len(set(val) - occ) + len(occ & set(con))


def test_calc_gs_excitation_budget_enforced():
    """A ground state built with an excitation budget respects it on every determinant."""
    budget = 1
    weighted = build_weighted_restrictions(({0: [[2, 3]]}, {0: [[4, 5]]}), excitation_budget=budget)
    _psis, _es, basis, _, _ = _run_calc_gs(weighted)
    for det in basis.local_basis:
        assert _bath_excitations(det) <= budget


def test_calc_gs_large_budget_reproduces_unrestricted_energy():
    """A budget >= the largest reachable excitation leaves E0 unchanged (Q=inf oracle)."""
    _, es_free, _, _, _ = _run_calc_gs(None)
    weighted = build_weighted_restrictions(({0: [[2, 3]]}, {0: [[4, 5]]}), excitation_budget=4)
    _, es_big, _, _, _ = _run_calc_gs(weighted)
    np.testing.assert_allclose(min(es_free), min(es_big), atol=1e-10)


# ---- calc_selfenergy end-to-end (the driver threading + GF inheritance) --------------


def _nio_selfenergy(excitation_budget):
    """Run calc_selfenergy on the small NiO d-shell workload with the given budget."""
    import dataclasses

    from impurityModel.ed.selfenergy import calc_selfenergy
    from impurityModel.test.support._nio_workload import as_calc_selfenergy_args, build_selfenergy_inputs

    inputs = build_selfenergy_inputs(nBaths=10, n_omega=3, dense_cutoff=100000, truncation_threshold=100000)
    args = as_calc_selfenergy_args(inputs)
    args["basis"] = dataclasses.replace(args["basis"], excitation_budget=excitation_budget)
    return calc_selfenergy(**args, comm=None)


def test_calc_selfenergy_excitation_budget_oracle():
    """Through calc_selfenergy: an unset budget and a >=max budget give the same self-energy.

    Exercises the full driver threading (basis_information -> calc_gs) and the automatic GF
    inheritance (greens_function widens basis.weighted_restrictions onto the excited bases).
    """
    base = _nio_selfenergy(None)
    big = _nio_selfenergy(50)  # 50 >> the 10 valence orbitals: admits everything
    np.testing.assert_allclose(base["sigma_real"], big["sigma_real"], atol=1e-8)
    np.testing.assert_allclose(np.min(base["gs_energies"]), np.min(big["gs_energies"]), atol=1e-10)


# ---- CHAIN_MAX_HOLES / CHAIN_MAX_ELECTRONS: length-independent per-chain caps ----------
#
# The excited-sector chain window defaults to a *fraction* of the chain length
# (CHAIN_FILLED_HOLE_FRACTION = 0.5 -> L//2 holes), which grows with the chain. The absolute
# caps replace that with a length-independent bound (the tighter of the two wins), so a long
# chain no longer inflates the window. These build a genuine chain -- impurity coupled only
# to the chain head, nearest-neighbour bath hopping -- so the far sites are freeze-eligible
# and receive a chain occupation window.


def _chain_excited_window(n_val=6, max_holes=None):
    """Valence-chain window from build_excited_restrictions, optionally with CHAIN_MAX_HOLES set."""
    from impurityModel.ed import basis_restrictions as br
    from impurityModel.ed.basis_restrictions import build_excited_restrictions
    from impurityModel.ed.manybody_basis import Basis

    n = 1 + n_val
    val = list(range(1, n))
    terms = {((0, "c"), (0, "a")): -1.0}
    for o in val:
        terms[((o, "c"), (o, "a"))] = -3.0  # below E_F -> nominally filled valence
    chain = [0, *val]
    for a, b in pairwise(chain):
        terms[((a, "c"), (b, "a"))] = 0.3  # nearest-neighbour chain hopping
        terms[((b, "c"), (a, "a"))] = 0.3
    op = ManyBodyOperator(terms)
    seed = bytes(_sd(list(range(n)), n=n).to_bytearray())  # impurity + whole chain filled
    basis = Basis(
        impurity_orbitals={0: [[0]]},
        bath_states=({0: [val]}, {0: [[]]}),
        initial_basis=[seed],
        chain_restrict=True,
        verbose=False,
    )
    saved = br.CHAIN_MAX_HOLES
    br.CHAIN_MAX_HOLES = max_holes
    try:
        # Legacy hop-count metric (coupling_cutoff=None) + min_dist=2 so the chain interior
        # is freeze-eligible; psis=None takes the "valence nominally filled" reference.
        rest = build_excited_restrictions(basis, op, psis=None, min_dist=2, coupling_cutoff=None)
    finally:
        br.CHAIN_MAX_HOLES = saved
    # The chain window is the largest restricted subset of the valence orbitals.
    chain_keys = [k for k in (rest or {}) if k and set(k) <= set(val) and len(k) > 2]
    assert chain_keys, f"no valence-chain window produced (restrictions={rest})"
    key = max(chain_keys, key=len)
    lo, hi = rest[key]
    return len(key), lo, hi  # (chain length windowed, floor, ceil==length)


def test_chain_max_holes_caps_excited_window():
    """CHAIN_MAX_HOLES bounds the holes per chain independently of the chain length."""
    length, lo0, hi0 = _chain_excited_window(n_val=6, max_holes=None)
    holes_uncapped = hi0 - lo0
    # Default fraction lets the window lose ~L//2 electrons.
    assert holes_uncapped >= length // 2

    _, lo2, hi2 = _chain_excited_window(n_val=6, max_holes=2)
    assert hi2 - lo2 <= 2  # at most two holes with the cap
    assert lo2 >= lo0  # the cap only tightens (raises the floor)

    _, lo1, _ = _chain_excited_window(n_val=6, max_holes=1)
    assert lo1 >= lo2  # a tighter cap raises the floor further


def test_chain_max_holes_none_is_noop():
    """With the cap unset (None) the window is exactly the historical fraction window."""
    _, lo_none, hi_none = _chain_excited_window(n_val=6, max_holes=None)
    # A cap at least as large as the chain removes nothing, matching the unset case.
    _, lo_big, hi_big = _chain_excited_window(n_val=6, max_holes=6)
    assert (lo_none, hi_none) == (lo_big, hi_big)


# ---- CHAIN_GRADED_RESTRICT: hopping-derived three-zone chain restriction ---------------
#
# The graded scheme splits a chain by the accumulated hopping a(o)=exp(-dist(o)) into a free
# head, an intermediate band (capped at CHAIN_INTERMEDIATE_MAX_EXCITATIONS), and a deep zone
# capped at CHAIN_DEEP_MAX_EXCITATIONS (default 1 -- the single-excitation sector must stay
# complete so every chain eigenmode / hybridization peak remains representable; pinning the
# deep zone was measured to make the self-energy non-causal on hybridization peaks). Uses the
# weighted metric (coupling_cutoff set) with hopping that DECAYS along the chain, so a(o)
# drops with depth and all three zones appear.


def _graded_chain_valence_windows(n_val=16, graded=True, slater_weight_min=1e-8, hop0=0.4, decay=0.85, deep_cap=None):
    """All valence-chain windows from build_excited_restrictions, graded on/off.

    Returns ``{frozenset(sites): (lo, hi)}`` restricted to valence orbitals. The chain hopping
    decays geometrically from the head so the accumulated hopping (and thus the zone) varies
    with depth. ``deep_cap`` overrides CHAIN_DEEP_MAX_EXCITATIONS when not None.
    """
    from impurityModel.ed import basis_restrictions as br
    from impurityModel.ed.basis_restrictions import build_excited_restrictions
    from impurityModel.ed.manybody_basis import Basis

    n = 1 + n_val
    val = list(range(1, n))
    terms = {((0, "c"), (0, "a")): -1.0}
    for o in val:
        terms[((o, "c"), (o, "a"))] = -3.0
    for k, (a, b) in enumerate(pairwise([0, *val])):
        t = hop0 * (decay**k)  # decaying hops -> accumulated hopping decays along the chain
        terms[((a, "c"), (b, "a"))] = t
        terms[((b, "c"), (a, "a"))] = t
    op = ManyBodyOperator(terms)
    seed = bytes(_sd(list(range(n)), n=n).to_bytearray())
    basis = Basis(
        impurity_orbitals={0: [[0]]},
        bath_states=({0: [val]}, {0: [[]]}),
        initial_basis=[seed],
        chain_restrict=True,
        verbose=False,
    )
    saved = br.CHAIN_GRADED_RESTRICT
    saved_cap = br.CHAIN_DEEP_MAX_EXCITATIONS
    br.CHAIN_GRADED_RESTRICT = graded
    if deep_cap is not None:
        br.CHAIN_DEEP_MAX_EXCITATIONS = deep_cap
    try:
        rest = build_excited_restrictions(basis, op, psis=None, es=None, slater_weight_min=slater_weight_min)
    finally:
        br.CHAIN_GRADED_RESTRICT = saved
        br.CHAIN_DEEP_MAX_EXCITATIONS = saved_cap
    return {k: v for k, v in (rest or {}).items() if set(k) <= set(val)}, set(val)


def _deep_windows(windows, cap):
    """The deep-zone windows: slack (hi - lo) equal to the deep cap on a group larger than it."""
    return {k: (lo, hi) for k, (lo, hi) in windows.items() if hi - lo == cap and len(k) > cap}


def test_graded_chain_has_deep_and_intermediate_zones():
    """Graded mode caps the deep chain at CHAIN_DEEP_MAX_EXCITATIONS and the middle band at K."""
    from impurityModel.ed.basis_restrictions import CHAIN_DEEP_MAX_EXCITATIONS, CHAIN_INTERMEDIATE_MAX_EXCITATIONS

    windows, _val = _graded_chain_valence_windows(graded=True)
    deep = _deep_windows(windows, CHAIN_DEEP_MAX_EXCITATIONS)
    assert deep, f"no deep zone produced (windows={windows})"
    # No window pins its group (lo == hi == len would forbid the propagating GF excitation).
    assert not any(lo == hi == len(k) for k, (lo, hi) in windows.items())
    # Every valence window allows at most the intermediate cap of holes.
    for lo, hi in windows.values():
        assert hi - lo <= CHAIN_INTERMEDIATE_MAX_EXCITATIONS


def test_graded_deep_zone_admits_a_single_excitation():
    """A lone hole on ANY chain site (even the deepest) satisfies the graded windows.

    This is the causality guarantee: the complete single-excitation sector keeps every
    eigenmode of the chain (every hybridization peak) representable.
    """
    windows, val = _graded_chain_valence_windows(graded=True)
    for site in val:
        occupied = val - {site}  # one hole at `site`
        for k, (lo, hi) in windows.items():
            n_occ = len(k & occupied)
            assert lo <= n_occ <= hi, f"single hole at {site} violates window {sorted(k)}: {(lo, hi)}"


def test_graded_deep_cap_zero_reproduces_hard_freeze():
    """CHAIN_DEEP_MAX_EXCITATIONS=0 restores the old pinned (len, len) deep window."""
    windows, _val = _graded_chain_valence_windows(graded=True, deep_cap=0)
    pinned = [k for k, (lo, hi) in windows.items() if lo == hi == len(k)]
    assert pinned, f"no pinned deep zone at cap 0 (windows={windows})"


def test_graded_off_reproduces_binary_window():
    """CHAIN_GRADED_RESTRICT=False leaves the single fraction window untouched."""
    from impurityModel.ed.basis_restrictions import CHAIN_DEEP_MAX_EXCITATIONS

    off, _ = _graded_chain_valence_windows(graded=False)
    # Binary mode emits one far-chain window (no deep-cap group, nothing pinned).
    assert not _deep_windows(off, CHAIN_DEEP_MAX_EXCITATIONS)
    assert not any(lo == hi == len(k) for k, (lo, hi) in off.items())


def test_graded_freeze_boundary_tracks_amplitude_cutoff():
    """A tighter amplitude cutoff pushes the freeze boundary deeper (smaller deep zone)."""
    from impurityModel.ed.basis_restrictions import CHAIN_DEEP_MAX_EXCITATIONS

    loose, _ = _graded_chain_valence_windows(graded=True, slater_weight_min=1e-8)
    tight, _ = _graded_chain_valence_windows(graded=True, slater_weight_min=1e-12)
    n_deep = lambda w: sum(len(k) for k in _deep_windows(w, CHAIN_DEEP_MAX_EXCITATIONS))  # noqa: E731
    assert n_deep(tight) <= n_deep(loose)
