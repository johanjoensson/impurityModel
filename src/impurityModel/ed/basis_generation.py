"""
Enumeration of the initial Slater-determinant basis from occupation windows,
and spin-flip completion of determinant sets. Pure, rank-deterministic code:
no MPI communication happens here.
"""

import itertools
from typing import Iterable, Optional

from impurityModel.ed import product_state_representation as psr
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp


def _window_str(lo: int, hi: int) -> str:
    """``"3"`` for a pinned occupation, ``"2-4"`` for a window."""
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def _bounds(patterns: list[tuple[int, int, int]], slot: int) -> tuple[int, int]:
    """``(min, max)`` of one component of the enumerated (impurity, valence, conduction) patterns."""
    values = [pattern[slot] for pattern in patterns]
    return min(values), max(values)


def generate_initial_basis(
    impurity_orbitals: dict[int, list[list[int]]],
    bath_states: tuple[dict[int, list[list[int]]], dict[int, list[list[int]]]],
    delta_valence_occ: Optional[dict[int, int]],
    delta_conduction_occ: Optional[dict[int, int]],
    delta_impurity_occ: Optional[dict[int, int]],
    nominal_impurity_occ: dict[int, int],
    mixed_valence: dict[int, int],
    n_bytes: int,
    verbose: bool,
    frozen_occupations: Optional[set] = None,
    total_charge_slack: int = 0,
) -> tuple[list[SlaterDeterminant], int]:
    """Construct the initial basis of Slater determinants.

    Parameters
    ----------
    impurity_orbitals : dict
        Impurity orbitals grouped by l quantum number.
    bath_states : tuple of dict
        Valence and conduction bath states grouped by l quantum number.
    delta_valence_occ : dict, optional
        Allowed valence bath occupation variations.
    delta_conduction_occ : dict, optional
        Allowed conduction bath occupation variations.
    delta_impurity_occ : dict, optional
        Allowed impurity occupation variations.
    nominal_impurity_occ : dict
        Nominal impurity occupations.
    mixed_valence : dict
        Per group, how far the *impurity* occupation may fluctuate around its nominal value.
        Realised as charge transfer with the bath, so the total electron number is unchanged;
        a value of ``k`` buys ``k`` bath charge-transfer excitations. Use ``total_charge_slack``
        for the different question of spanning several total-charge sectors.
    verbose : bool
        Whether to print configuration details.
    frozen_occupations : set, optional
        Orbital-set keys whose impurity occupation is pinned at exactly
        ``nominal_impurity_occ[i]`` (e.g. a bath-less core shell). Pinned shells are
        excluded from the multi-group redistribution: without this, the cross-group
        total filter alone lets a core shell drain into a lower-lying valence shell
        (2p4 3d10 on the NiO L-edge), and since no Hamiltonian term moves charge
        between shells, the drained sector is H-disconnected from the physical one.
    total_charge_slack : int, optional
        Half-width, in electrons, of the *total charge* window. ``0`` (the default) makes the
        basis a single charge sector, which is what every energy comparison needs: ``H``
        conserves ``N``, so a basis spanning several sectors makes ``calc_energy(N0)`` report
        the minimum over the window rather than the energy of ``N0``. Only a caller that
        deliberately wants a multi-sector space -- :func:`dc_criteria.build_union_space`, which
        sweeps ``H(mu) = H(0) - mu*N`` on one fixed space -- should set it non-zero. It is a
        different concept from ``mixed_valence``, which fluctuates the *impurity* charge against
        the bath at fixed total.

    Returns
    -------
    basis : list of SlaterDeterminant
        The list of constructed initial Slater determinants.
    num_spin_orbitals : int
        The total number of spin orbitals.
    """
    valence_baths, conduction_baths = bath_states
    total_baths = {
        i: sum(len(orbs) for orbs in valence_baths[i]) + sum(len(orbs) for orbs in conduction_baths[i])
        for i in valence_baths
    }

    if delta_valence_occ is None:
        delta_valence_occ = dict.fromkeys(impurity_orbitals.keys(), 0)
    if delta_conduction_occ is None:
        delta_conduction_occ = dict.fromkeys(impurity_orbitals.keys(), 0)
    if delta_impurity_occ is None:
        delta_impurity_occ = dict.fromkeys(impurity_orbitals.keys(), 0)
    if frozen_occupations is None:
        frozen_occupations = set()

    # Mixed valence is charge transfer *with the bath*, not charge from nowhere. An impurity
    # occupation of ``nominal + d`` has to be paid for by d electrons taken off the valence bath
    # (or put onto the conduction bath) -- which is precisely what the ``delta_*`` mechanism
    # below does, and what keeps the total electron number fixed. Widening the ``nominal_occ``
    # loop instead created determinants at several total electron numbers; since H conserves N,
    # ``calc_energy`` then returned the minimum over the whole window rather than the energy of
    # the requested sector, and ``fixed_peak_dc``'s observable ``E[N+1] - E[N]`` collapsed to
    # exactly 0.0. Fold the requested window into the bath-transfer windows and keep the nominal
    # impurity charge pinned.
    # The excitation budget stays honest: a mixed valence of k buys k bath charge-transfer
    # excitations, so it reaches impurity occupations nominal +/- k but does not smuggle in the
    # charge-neutral (valence -> conduction) double excitation that widening all three windows
    # independently would allow. The caller's own dN windows are untouched -- their pairwise sum
    # is already their bound -- so dN behaviour is bit-identical.
    excitation_budget = {
        i: max(delta_valence_occ[i] + delta_conduction_occ[i], abs(mixed_valence[i])) for i in impurity_orbitals
    }
    delta_valence_occ = {i: max(delta_valence_occ[i], abs(mixed_valence[i])) for i in impurity_orbitals}
    delta_conduction_occ = {i: max(delta_conduction_occ[i], abs(mixed_valence[i])) for i in impurity_orbitals}
    delta_impurity_occ = {i: max(delta_impurity_occ[i], abs(mixed_valence[i])) for i in impurity_orbitals}

    total_impurity_orbitals = {i: sum(len(orbs) for orbs in impurity_orbitals[i]) for i in impurity_orbitals}
    # Per group, materialise the allowed configurations tagged with their impurity
    # occupation, as (impurity_occupation, occupied_orbital_tuple). Materialising (rather
    # than keeping lazy nested itertools iterators) avoids re-consuming an exhausted
    # iterator when several groups each admit multiple occupations, and lets the cross-group
    # combination below be filtered by *total* impurity charge.
    # When the impurity is split into several orbital-symmetry manifolds (this grouping), they
    # are one correlated shell that must freely redistribute charge among manifolds at fixed
    # *total* occupation. A single group already enumerates every whole-impurity arrangement
    # through ``combinations`` below, so its per-group nominal stays pinned; but with >= 2 groups
    # each group's occupation ranges over the whole [0, group_size] and the cross-group *total*
    # filter keeps only the arrangements in the occupation window. Gating the per-group range in
    # the grouped case instead pins each manifold and collapses the seed to a single frozen
    # configuration -- the NiO covalency / magnetic-moment regression. Redistribution moves charge
    # between manifolds and so leaves the electron count alone; the total-electron filter at the
    # DFS leaf is what stops the sliding nominal from changing it. Frozen shells never
    # redistribute (their window is pinned below) and contribute no slack to the total window.
    redistribute = len([i for i in impurity_orbitals if i not in frozen_occupations]) > 1
    total_nominal = sum(int(nominal_impurity_occ[i]) for i in valence_baths)
    # ``mixed_valence`` was folded into ``delta_impurity_occ`` above, so it is not added again
    # here. The two contributions are independent questions -- how far the impurity charge may
    # fluctuate against the bath, and how many total-charge sectors the space spans -- so the
    # impurity-charge window has to admit both. Only the latter loosens the electron-count filter
    # at the DFS leaf, which stays the binding constraint.
    total_slack = total_charge_slack + max(
        (abs(delta_impurity_occ[i]) for i in valence_baths if i not in frozen_occupations),
        default=0,
    )
    group_configurations = {}
    # Per-group occupation windows actually enumerated, for the one-line report at the end:
    # {group: [imp_lo, imp_hi, val_lo, val_hi, con_lo, con_hi, n_patterns]}. Reported as ranges
    # rather than one block per (impurity, valence, conduction) pattern: this function is
    # re-entered once per trial occupation of the ground-state search, so anything printed per
    # pattern here is multiplied by the number of patterns *and* the number of trials -- which
    # is how a four-line block per pattern became hundreds of lines of scrollback per solve.
    occupation_patterns: dict[int, list[tuple[int, int, int]]] = {i: [] for i in valence_baths}
    for i in valence_baths:
        configs = []
        impurity_electron_indices = [orb for imp_orbs in impurity_orbitals[i] for orb in imp_orbs]
        valence_electron_indices = [orb for val_orbs in valence_baths[i] for orb in val_orbs]
        conduction_electron_indices = [orb for con_orbs in conduction_baths[i] for orb in con_orbs]
        if i in frozen_occupations:
            occ_lo = occ_hi = nominal_impurity_occ[i]
        else:
            # Pinned unless the groups redistribute among themselves: the impurity charge window
            # is generated by the delta_* (bath charge-transfer) loops below, at fixed total
            # electron number, not by sliding this nominal. Sliding it is what changes the total,
            # so it is allowed only by exactly as much as `total_charge_slack` permits.
            occ_lo = 0 if redistribute else max(0, nominal_impurity_occ[i] - total_charge_slack)
            occ_hi = (
                total_impurity_orbitals[i]
                if redistribute
                else min(total_impurity_orbitals[i], nominal_impurity_occ[i] + total_charge_slack)
            )
        for nominal_occ in range(occ_lo, occ_hi + 1):
            for delta_valence in range(delta_valence_occ[i] + 1):
                for delta_conduction in range(delta_conduction_occ[i] + 1):
                    delta_impurity = delta_valence - delta_conduction
                    if (
                        abs(delta_impurity) <= abs(delta_impurity_occ[i])
                        and delta_valence + delta_conduction <= excitation_budget[i]
                        and nominal_occ + delta_impurity <= total_impurity_orbitals[i]
                        and nominal_occ + delta_impurity >= 0
                        and delta_valence <= len(valence_electron_indices)
                    ):
                        impurity_occupation = nominal_occ + delta_impurity
                        valence_occupation = len(valence_electron_indices) - delta_valence
                        conduction_occupation = delta_conduction
                        occupation_patterns[i].append((impurity_occupation, valence_occupation, conduction_occupation))
                        for imp_c, val_c, con_c in itertools.product(
                            itertools.combinations(impurity_electron_indices, impurity_occupation),
                            itertools.combinations(valence_electron_indices, valence_occupation),
                            itertools.combinations(conduction_electron_indices, conduction_occupation),
                        ):
                            configs.append((impurity_occupation, imp_c + val_c + con_c))
        group_configurations[i] = configs
    num_spin_orbitals = sum(total_impurity_orbitals[i] + total_baths[i] for i in total_baths)

    # Filter the cross-group combinations on the whole-impurity charge window computed above,
    # so wide per-manifold windows cannot leak total charge: the manifolds redistribute at
    # fixed impurity count, while a single group keeps its full impurity/bath charge-transfer
    # range (the filter is then a no-op).
    lo_tot = max(0, total_nominal - total_slack)
    hi_tot = total_nominal + total_slack

    # Combine the per-group configurations, keeping only determinants whose *total* impurity
    # occupation lies in the window [lo_tot, hi_tot]. Rather than materialise the full
    # itertools.product of the per-group configs (up to ~2^n_imp arrangements in the
    # multi-group ``redistribute`` branch, where each group ranges over its whole
    # [0, group_size]) and discard the out-of-window majority, enumerate incrementally with
    # running-total pruning: at each group only keep partial choices that can still reach a
    # total inside the window, given the min/max impurity occupation attainable from the
    # remaining groups. The surviving determinant set is identical to the product-then-filter
    # result, but the cost is proportional to the in-window output rather than the full
    # product -- decisive for large impurities / long manifolds.
    group_lists = list(group_configurations.values())
    n_groups = len(group_lists)
    # suffix_min/max[t] = min/max total impurity occupation attainable from groups t.. onward.
    suffix_min = [0] * (n_groups + 1)
    suffix_max = [0] * (n_groups + 1)
    for t in range(n_groups - 1, -1, -1):
        occs = [imp_occ for imp_occ, _ in group_lists[t]]
        suffix_min[t] = suffix_min[t + 1] + (min(occs) if occs else 0)
        suffix_max[t] = suffix_max[t + 1] + (max(occs) if occs else 0)

    # The one quantity H actually conserves. Every determinant kept below holds exactly this
    # many electrons, so the basis is a single charge sector and `calc_energy(N0)` is the energy
    # *of that sector* rather than the minimum over a window of them. The per-group count is
    # `nominal_impurity_occ + len(valence)`: the delta_* loops move charge between the impurity
    # and the bath, which leaves it invariant, and the redistribute branch moves charge between
    # impurity groups, which leaves it invariant too. Only sliding a group's nominal changes it.
    n_electrons = sum(nominal_impurity_occ[i] + sum(len(orbs) for orbs in valence_baths[i]) for i in valence_baths)

    basis = []
    # Iterative DFS; a frame is (group_index, partial_impurity_occ, partial_occupied_orbitals).
    stack: list[tuple[int, int, tuple[int, ...]]] = [(0, 0, ())]
    while stack:
        t, partial_occ, occupied = stack.pop()
        if t == n_groups:
            # The last group's prune already guarantees lo_tot <= partial_occ <= hi_tot. The
            # electron count is the length of the accumulated orbital tuple; the redistribute
            # branch lets a group's nominal slide, so it is checked here rather than assumed.
            if abs(len(occupied) - n_electrons) <= total_charge_slack:
                basis.append(psr.tuple2bytes(occupied, 8 * n_bytes))
            continue
        for imp_occ, orbs in group_lists[t]:
            next_occ = partial_occ + imp_occ
            # Prune unless the remaining groups can still land the total inside the window.
            if next_occ + suffix_min[t + 1] > hi_tot or next_occ + suffix_max[t + 1] < lo_tot:
                continue
            stack.append((t + 1, next_occ, occupied + orbs))

    if verbose:
        groups = "; ".join(
            f"{i}: imp {_window_str(*_bounds(patterns, 0))}, val {_window_str(*_bounds(patterns, 1))},"
            f" con {_window_str(*_bounds(patterns, 2))} ({len(patterns)} cfg)"
            for i, patterns in sorted(occupation_patterns.items())
            if patterns
        )
        # Indented one level: this is the detail *behind* a row of the occupation-search table
        # in `groundstate.find_ground_state_basis`, which is re-entered once per trial sector.
        print(
            f"    seed basis: {len(basis)} determinants, {n_electrons} electrons, "
            f"impurity occupation {_window_str(lo_tot, hi_tot)}"
        )
        if groups:
            print(f"      per group -- {groups}")

    return [SlaterDeterminant.from_bytes(bytestring) for bytestring in basis], num_spin_orbitals


def spin_flipped_determinants(
    impurity_orbitals: dict[int, list[list[int]]], determinants: Iterable[SlaterDeterminant]
) -> set[SlaterDeterminant]:
    """Generate spin-flipped counterparts for a collection of determinants.

    Parameters
    ----------
    determinants : Iterable of SlaterDeterminant
        The starting Slater determinants to spin-flip.

    Returns
    -------
    set of SlaterDeterminant
        The original determinants plus their spin-flipped counterparts.
    """
    n_dn_op = {
        ((i, "c"), (i, "a")): 1.0
        for l in impurity_orbitals
        for i in range(sum(len(orbs) for orbs in impurity_orbitals[l]) // 2)
    }
    n_up_op = {
        ((i, "c"), (i, "a")): 1.0
        for l in impurity_orbitals
        for i in range(
            sum(len(orbs) for orbs in impurity_orbitals[l]) // 2,
            sum(len(orbs) for orbs in impurity_orbitals[l]),
        )
    }
    n_dn_mbo = ManyBodyOperator(n_dn_op)
    n_up_mbo = ManyBodyOperator(n_up_op)
    spin_flip = set()
    for det in determinants:
        n_dn = _real_occupation(applyOp(n_dn_mbo, ManyBodyState({det: 1.0}), cutoff=0), det)
        n_up = _real_occupation(applyOp(n_up_mbo, ManyBodyState({det: 1.0}), cutoff=0), det)
        spin_flip.add(det)
        to_flip = {det}
        for _l, orb_groups in impurity_orbitals.items():
            n_orb = sum(len(orbs) for orbs in orb_groups)
            for i in range(n_orb // 2):
                spin_flip_op = {
                    ((i + n_orb // 2, "c"), (i, "a")): 1.0,
                    ((i, "c"), (i + n_orb // 2, "a")): 1.0,
                }
                spin_flip_mbo = ManyBodyOperator(spin_flip_op)
                for state in list(to_flip):
                    flipped = applyOp(spin_flip_mbo, ManyBodyState({state: 1.0}), cutoff=0)
                    to_flip.update(flipped.keys())
                    if len(flipped) == 0:
                        continue
                    flipped_state = next(iter(flipped.keys()))
                    new_n_dn = _real_occupation(
                        applyOp(n_dn_mbo, ManyBodyState({flipped_state: 1.0}), cutoff=0), flipped_state
                    )
                    new_n_up = _real_occupation(
                        applyOp(n_up_mbo, ManyBodyState({flipped_state: 1.0}), cutoff=0), flipped_state
                    )
                    if (new_n_dn == n_dn and new_n_up == n_up) or (new_n_dn == n_up and new_n_up == n_dn):
                        spin_flip.update(flipped.keys())

    return spin_flip


def _real_occupation(psi: ManyBodyState, det: SlaterDeterminant) -> int:
    """Read back a width-1 block's real amplitude for ``det`` (0 if absent)."""
    row = psi.get(det)
    return int(0 if row is None else row[0].real)
