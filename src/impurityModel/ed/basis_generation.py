"""
Enumeration of the initial Slater-determinant basis from occupation windows,
and spin-flip completion of determinant sets. Pure, rank-deterministic code:
no MPI communication happens here.
"""

import itertools
from typing import Iterable, Optional

from impurityModel.ed import product_state_representation as psr
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp


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
        Allowed mixed valence variations.
    verbose : bool
        Whether to print configuration details.
    frozen_occupations : set, optional
        Orbital-set keys whose impurity occupation is pinned at exactly
        ``nominal_impurity_occ[i]`` (e.g. a bath-less core shell). Pinned shells are
        excluded from the multi-group redistribution: without this, the cross-group
        total filter alone lets a core shell drain into a lower-lying valence shell
        (2p4 3d10 on the NiO L-edge), and since no Hamiltonian term moves charge
        between shells, the drained sector is H-disconnected from the physical one.

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

    total_impurity_orbitals = {i: sum(len(orbs) for orbs in impurity_orbitals[i]) for i in impurity_orbitals}
    # Per group, materialise the allowed configurations tagged with their impurity
    # occupation, as (impurity_occupation, occupied_orbital_tuple). Materialising (rather
    # than keeping lazy nested itertools iterators) avoids re-consuming an exhausted
    # iterator when several groups each admit multiple occupations, and lets the cross-group
    # combination below be filtered by *total* impurity charge.
    # When the impurity is split into several orbital-symmetry manifolds (this grouping), they
    # are one correlated shell that must freely redistribute charge among manifolds at fixed
    # *total* occupation. A single group already enumerates every whole-impurity arrangement
    # through ``combinations`` below, so its per-group occupation stays pinned to
    # ``nominal +/- mixed_valence`` (preserving the seed count for the un-grouped case); but with
    # >= 2 groups each group's occupation ranges over the whole [0, group_size] and the
    # cross-group *total* filter keeps only the arrangements in the occupation window. Gating the
    # per-group range by ``mixed_valence[i]`` in the grouped case instead pins each manifold and
    # collapses the seed to a single frozen configuration -- the NiO covalency / magnetic-moment
    # regression. ``mixed_valence`` still widens the *total* window via ``total_slack``.
    # Frozen shells never redistribute (their window is pinned below) and contribute no
    # slack to the total window.
    redistribute = len([i for i in impurity_orbitals if i not in frozen_occupations]) > 1
    total_nominal = sum(int(nominal_impurity_occ[i]) for i in valence_baths)
    total_slack = max(
        (abs(mixed_valence[i]) + abs(delta_impurity_occ[i]) for i in valence_baths if i not in frozen_occupations),
        default=0,
    )
    group_configurations = {}
    for i in valence_baths:
        configs = []
        impurity_electron_indices = [orb for imp_orbs in impurity_orbitals[i] for orb in imp_orbs]
        valence_electron_indices = [orb for val_orbs in valence_baths[i] for orb in val_orbs]
        conduction_electron_indices = [orb for con_orbs in conduction_baths[i] for orb in con_orbs]
        if i in frozen_occupations:
            occ_lo = occ_hi = nominal_impurity_occ[i]
        else:
            occ_lo = 0 if redistribute else max(0, nominal_impurity_occ[i] - abs(mixed_valence[i]))
            occ_hi = (
                total_impurity_orbitals[i]
                if redistribute
                else min(total_impurity_orbitals[i], nominal_impurity_occ[i] + abs(mixed_valence[i]))
            )
        for nominal_occ in range(occ_lo, occ_hi + 1):
            for delta_valence in range(delta_valence_occ[i] + 1):
                for delta_conduction in range(delta_conduction_occ[i] + 1):
                    delta_impurity = delta_valence - delta_conduction
                    if (
                        abs(delta_impurity) <= abs(delta_impurity_occ[i])
                        and nominal_occ + delta_impurity <= total_impurity_orbitals[i]
                        and nominal_occ + delta_impurity >= 0
                        and delta_valence <= len(valence_electron_indices)
                    ):
                        impurity_occupation = nominal_occ + delta_impurity
                        valence_occupation = len(valence_electron_indices) - delta_valence
                        conduction_occupation = delta_conduction
                        if verbose:
                            print(f"Partition {i} occupations")
                            print(f"Impurity occupation:   {impurity_occupation:d}")
                            print(f"Valence occupation:   {valence_occupation:d}")
                            print(f"Conduction occupation: {conduction_occupation:d}")
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

    basis = []
    # Iterative DFS; a frame is (group_index, partial_impurity_occ, partial_occupied_orbitals).
    stack: list[tuple[int, int, tuple[int, ...]]] = [(0, 0, ())]
    while stack:
        t, partial_occ, occupied = stack.pop()
        if t == n_groups:
            # The last group's prune already guarantees lo_tot <= partial_occ <= hi_tot.
            basis.append(psr.tuple2bytes(occupied, 8 * n_bytes))
            continue
        for imp_occ, orbs in group_lists[t]:
            next_occ = partial_occ + imp_occ
            # Prune unless the remaining groups can still land the total inside the window.
            if next_occ + suffix_min[t + 1] > hi_tot or next_occ + suffix_max[t + 1] < lo_tot:
                continue
            stack.append((t + 1, next_occ, occupied + orbs))

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
        n_dn = int(applyOp(n_dn_mbo, ManyBodyState({det: 1.0}), cutoff=0).get(det, 0).real)
        n_up = int(applyOp(n_up_mbo, ManyBodyState({det: 1.0}), cutoff=0).get(det, 0).real)
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
                    new_n_dn = int(
                        applyOp(n_dn_mbo, ManyBodyState({flipped_state: 1.0}), cutoff=0).get(flipped_state, 0).real
                    )
                    new_n_up = int(
                        applyOp(n_up_mbo, ManyBodyState({flipped_state: 1.0}), cutoff=0).get(flipped_state, 0).real
                    )
                    if (new_n_dn == n_dn and new_n_up == n_up) or (new_n_dn == n_up and new_n_up == n_dn):
                        spin_flip.update(flipped.keys())

    return spin_flip
