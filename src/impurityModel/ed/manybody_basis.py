from math import ceil
from typing import Any, Optional, Union

try:
    from collections.abc import Iterable, Sequence
except ModuleNotFoundError:
    from collections import Iterable, Sequence
import itertools
from heapq import merge

import numpy as np
import scipy as sp
from mpi4py import MPI

from impurityModel.ed import product_state_representation as psr
from impurityModel.ed.finite import (
    thermal_average_scale_indep,
)
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyOperator,
    ManyBodyState,
    SlaterDeterminant,
)
from impurityModel.ed.ManyBodyUtils import (
    applyOp as applyOp_test,
)
from impurityModel.ed.mpi_comm import graph_alltoall, graph_alltoall_psis
from impurityModel.ed.utils import matrix_print


def _pack_units(
    weights, comm_size: int, split_threshold: float
) -> tuple[Optional[list[tuple[int, ...]]], Optional[np.ndarray]]:
    """Pack work units into per-color bins and allocate ranks to each color.

    Pure packing math behind :meth:`Basis.split_basis_and_redistribute_psi` — no MPI,
    so it is unit-testable, and every rank computes the identical packing from the
    (already Allreduced) weights.

    The number of colors is capped by the participation ratio (Σw)²/Σw² — the
    effective number of equally-weighted units — scaled by ``split_threshold``, so a
    few dominant units are not starved of ranks: better to run them on a larger
    sub-communicator (or unified). ``split_threshold=0`` forces a single unified
    communicator; ``=1`` is the legacy max-split for equal weights.

    Units are packed with LPT (Longest Processing Time): the next-heaviest unit goes
    to the currently lightest bin, ties to the lowest bin index. This bounds the
    heaviest bin at 4/3 of the optimal packing and reduces to round-robin dealing on
    uniform weights. Ranks are then apportioned to bins proportionally to bin mass by
    largest remainder, every bin keeping at least one rank.

    Parameters
    ----------
    weights : array_like of float
        Cost weight per unit (identical on every rank).
    comm_size : int
        Number of MPI ranks to distribute over.
    split_threshold : float
        Scale factor on the participation-ratio cap of the number of colors.

    Returns
    -------
    subgroups : list of tuple of int, or None
        Unit indices assigned to each color; ``None`` when the packing collapses to
        a single color (the caller should not split).
    procs_per_color : ndarray of int, or None
        Ranks per color, each at least 1, summing to ``comm_size``.
    """
    normalized = np.abs(np.asarray(weights, dtype=float))
    normalized /= np.sum(normalized)
    n_colors = min(comm_size, len(normalized))
    participation = 1.0 / np.sum(normalized**2)
    n_colors = min(n_colors, max(1, int(np.ceil(participation * split_threshold))))
    if n_colors <= 1:
        return None, None

    # LPT packing. The first n_colors units land in distinct empty bins, so no bin
    # is ever empty (n_colors <= number of units).
    sorted_idxs = np.argsort(normalized, kind="stable")[::-1]
    subgroups: list[tuple[int, ...]] = [tuple() for _ in range(n_colors)]
    bin_mass = np.zeros(n_colors)
    for u in sorted_idxs:
        c = int(np.argmin(bin_mass))
        subgroups[c] += (int(u),)
        bin_mass[c] += normalized[u]

    # Largest-remainder rank apportionment on the bin masses (they sum to 1).
    raw = comm_size * bin_mass
    floors = np.floor(raw).astype(int)
    procs_per_color = np.maximum(floors, 1)
    remainder = comm_size - int(np.sum(procs_per_color))
    if remainder > 0:
        # The floors sum to within n_colors of comm_size, so one pass over the
        # largest fractional parts places every leftover rank.
        order = np.argsort(-(raw - floors), kind="stable")
        procs_per_color[order[:remainder]] += 1
    else:
        # The max(1, .) floors over-allocated: reclaim ranks from the lightest bins
        # that can spare one, so the heaviest bins keep their proportional share.
        order = np.argsort(bin_mass, kind="stable")
        while remainder < 0:
            reclaimed = False
            for c in order:
                if procs_per_color[c] > 1:
                    procs_per_color[c] -= 1
                    remainder += 1
                    reclaimed = True
                    if remainder == 0:
                        break
            assert reclaimed, "rank apportionment failed to converge"
    assert np.sum(procs_per_color) == comm_size
    return subgroups, procs_per_color


class Basis:
    """Many-body basis of Slater determinants.

    This class manages the Slater determinant basis states for exact diagonalization,
    supporting distributed states over MPI, restrictions, and basis extensions.
    """

    def _get_initial_basis(
        self,
        impurity_orbitals: dict[int, list[list[int]]],
        bath_states: tuple[dict[int, list[list[int]]], dict[int, list[list[int]]]],
        delta_valence_occ: Optional[dict[int, int]],
        delta_conduction_occ: Optional[dict[int, int]],
        delta_impurity_occ: Optional[dict[int, int]],
        nominal_impurity_occ: dict[int, int],
        mixed_valence: dict[int, int],
        verbose: bool,
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
        redistribute = len(impurity_orbitals) > 1
        total_nominal = sum(int(nominal_impurity_occ[i]) for i in valence_baths)
        total_slack = max((abs(mixed_valence[i]) + abs(delta_impurity_occ[i]) for i in valence_baths), default=0)
        group_configurations = {}
        for i in valence_baths:
            configs = []
            impurity_electron_indices = [orb for imp_orbs in impurity_orbitals[i] for orb in imp_orbs]
            valence_electron_indices = [orb for val_orbs in valence_baths[i] for orb in val_orbs]
            conduction_electron_indices = [orb for con_orbs in conduction_baths[i] for orb in con_orbs]
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
        stack = [(0, 0, ())]
        while stack:
            t, partial_occ, occupied = stack.pop()
            if t == n_groups:
                # The last group's prune already guarantees lo_tot <= partial_occ <= hi_tot.
                basis.append(psr.tuple2bytes(occupied, 8 * self.n_bytes))
                continue
            for imp_occ, orbs in group_lists[t]:
                next_occ = partial_occ + imp_occ
                # Prune unless the remaining groups can still land the total inside the window.
                if next_occ + suffix_min[t + 1] > hi_tot or next_occ + suffix_max[t + 1] < lo_tot:
                    continue
                stack.append((t + 1, next_occ, occupied + orbs))

        return [SlaterDeterminant.from_bytes(bytestring) for bytestring in basis], num_spin_orbitals

    def get_effective_restrictions(self) -> dict[frozenset[int], tuple[int, int]]:
        """Calculate the actual min/max occupations observed across the current basis.

        Returns
        -------
        restrictions : dict of frozenset of int to (int, int)
            Dictionary mapping orbital subsets to their observed (min, max) occupations.
        """
        valence_baths, conduction_baths = self.bath_states
        restrictions = {}

        # Impurity occupation is restricted on the *whole* impurity as a single subset (never
        # per orbital-symmetry group). A per-group impurity pin would fix S_z (spin groups) or
        # the eg/t2g ratio (manifold groups) and destroy spin-multiplet degeneracy / confine the
        # charge; the union bound confines only the total impurity charge, leaving the manifolds
        # free to redistribute at fixed count.
        all_impurity_indices = frozenset(
            sorted(ind for imp_blocks in self.impurity_orbitals.values() for imp_ind in imp_blocks for ind in imp_ind)
        )
        valence_index_sets = {
            i: frozenset(sorted(ind for val_ind in valence_baths[i] for ind in val_ind)) for i in valence_baths
        }
        conduction_index_sets = {
            i: frozenset(sorted(ind for con_ind in conduction_baths[i] for ind in con_ind)) for i in conduction_baths
        }

        max_imp, min_imp = 0, len(all_impurity_indices)
        min_val = {i: len(valence_index_sets[i]) for i in valence_baths}
        max_con = {i: 0 for i in conduction_baths}
        for state in self.local_basis:
            bits = psr.bytes2bitarray(bytes(state.to_bytearray()), self.num_spin_orbitals)
            n_imp = sum(bits[i] for i in all_impurity_indices)
            max_imp = max(max_imp, n_imp)
            min_imp = min(min_imp, n_imp)
            for i in valence_baths:
                n_val = sum(bits[j] for j in valence_index_sets[i])
                n_con = sum(bits[j] for j in conduction_index_sets[i])
                min_val[i] = min(min_val[i], n_val)
                max_con[i] = max(max_con[i], n_con)
        if self.is_distributed:
            max_imp = self.comm.allreduce(max_imp, op=MPI.MAX)
            min_imp = self.comm.allreduce(min_imp, op=MPI.MIN)
        if len(all_impurity_indices) > 0:
            restrictions[all_impurity_indices] = (min_imp, max_imp)
        for i in sorted(valence_baths.keys()):
            v_min, c_max = min_val[i], max_con[i]
            if self.is_distributed:
                v_min = self.comm.allreduce(v_min, op=MPI.MIN)
                c_max = self.comm.allreduce(c_max, op=MPI.MAX)
            # Valence baths sit filled (max = full), conduction empty (min = 0) by convention.
            if len(valence_index_sets[i]) > 0:
                restrictions[valence_index_sets[i]] = (v_min, len(valence_index_sets[i]))
            if len(conduction_index_sets[i]) > 0:
                restrictions[conduction_index_sets[i]] = (0, c_max)
        return restrictions

    def _get_updated_occ_restrictions(
        self, restrictions: dict[frozenset[int], tuple[int, int]], orbs: frozenset[int], dN: Optional[tuple[int, int]]
    ) -> tuple[int, int]:
        """Compute the updated occupation bounds after an operator excitation.

        Parameters
        ----------
        restrictions : dict of frozenset of int to (int, int)
            The existing occupation restrictions.
        orbs : frozenset of int
            The set of orbitals being modified.
        dN : tuple of (int, int), optional
            (occupations_decreased, occupations_increased) bounds, or None.

        Returns
        -------
        min_occ : int
            The new minimum occupation allowed.
        max_occ : int
            The new maximum occupation allowed.
        """
        if dN is None or orbs not in restrictions:
            return 0, len(orbs)
        occ_dec, occ_inc = dN
        min_occ, max_occ = restrictions[orbs]
        return max(min_occ - occ_dec, 0), min(max_occ + occ_inc, len(orbs))

    def _impurity_coupling_distance(self, op, tot_orb, all_impurity_orbitals, coupling_cutoff, min_dist):
        r"""Distance-from-impurity matrix classifying how weakly each orbital couples in.

        The default (``coupling_cutoff`` set) is a *physics-derived* metric: each hopping edge
        gets weight :math:`-\log(|h_{ij}| / h_{\max})` (``h_max`` = largest off-diagonal
        hopping), so the weighted shortest path from the impurity accumulates
        :math:`-\log` of the product of couplings along the best path and
        :math:`e^{-\text{dist}}` is the effective coupling strength. An orbital counts as
        decoupled (freeze-eligible) when ``dist > -log(coupling_cutoff)``. This follows the
        actual coupling *strength* rather than graph hop-count, so a strongly-hybridised long
        chain stays free while a weakly-coupled near orbital is frozen.

        With ``coupling_cutoff=None`` it falls back to the legacy unweighted hop-count distance
        with threshold ``min_dist``.

        Returns ``(dist_matrix, threshold)`` with ``dist_matrix`` rows ordered as
        ``all_impurity_orbitals`` (the callers index it with impurity orbital indices, valid for
        the 0-based impurity layout) and an orbital frozen when ``dist > threshold``.
        """
        hop = np.zeros((tot_orb, tot_orb))
        for i, j in itertools.product(range(tot_orb), repeat=2):
            key = ((i, "c"), (j, "a"))
            if key in op:
                hop[i, j] = abs(op[key])
        if coupling_cutoff is None:
            graph = hop > 1e-8
            dist = sp.sparse.csgraph.shortest_path(
                graph, directed=False, unweighted=True, indices=all_impurity_orbitals
            )
            return dist, min_dist
        np.fill_diagonal(hop, 0.0)
        h_max = hop.max()
        mask = hop > 1e-8
        graph = np.zeros((tot_orb, tot_orb))
        if h_max > 0:
            # weight = -log(|h| / h_max) >= 0; + small offset so the strongest edges (weight 0)
            # are not read as "no edge" by csgraph (which drops values <~1e-8). The offset is
            # negligible for the cutoff (it takes thousands of hops to accumulate to it).
            graph[mask] = -np.log(hop[mask] / h_max) + 1e-3
        dist = sp.sparse.csgraph.shortest_path(graph, directed=False, indices=all_impurity_orbitals)
        return dist, -np.log(coupling_cutoff)

    def build_initial_restrictions(
        self, op: ManyBodyOperator, min_dist: int = 4, coupling_cutoff: float = 1e-3
    ) -> Optional[dict[frozenset[int], tuple[int, int]]]:
        """Construct the initial occupation restrictions based on Hamiltonian connectivity.

        Parameters
        ----------
        op : ManyBodyOperator
            The Hamiltonian operator.
        min_dist : int, default 4
            Minimum shortest-path distance from the impurity to consider a bath state.

        Returns
        -------
        restrictions : dict of frozenset of int to (int, int), optional
            The initial ground state restrictions, or None if no restrictions were built.
        """
        ground_state_restrictions = {}
        valence_baths, conduction_baths = self.bath_states

        filled_bath_states = []
        empty_bath_states = []

        all_impurity_orbitals = [
            orb for orb_blocks in self.impurity_orbitals.values() for orb_block in orb_blocks for orb in orb_block
        ]
        all_valence_orbitals = [
            orb for orb_blocks in valence_baths.values() for orb_block in orb_blocks for orb in orb_block
        ]
        all_conduction_orbitals = [
            orb for orb_blocks in conduction_baths.values() for orb_block in orb_blocks for orb in orb_block
        ]
        tot_orb = len(all_impurity_orbitals) + len(all_valence_orbitals) + len(all_conduction_orbitals)
        dist_matrix, dist_cutoff = self._impurity_coupling_distance(
            op, tot_orb, all_impurity_orbitals, coupling_cutoff, min_dist
        )
        for i, impurity_orbitals in self.impurity_orbitals.items():
            for imp_orb_block, val_orb_block, con_orb_block in zip(
                impurity_orbitals, valence_baths[i], conduction_baths[i]
            ):
                # Identify filled and empty bath states
                # Only restrict states that couple weakly to the impurity (dist above cutoff).
                filled_valence_states = [
                    orb for orb in val_orb_block if np.min(dist_matrix[np.ix_(imp_orb_block, [orb])]) > dist_cutoff
                ]
                filled_states = frozenset(sorted(filled_valence_states))
                empty_conduction_states = [
                    orb for orb in con_orb_block if np.min(dist_matrix[np.ix_(imp_orb_block, [orb])]) > dist_cutoff
                ]
                empty_states = frozenset(sorted(empty_conduction_states))
                filled_bath_states.append(filled_states)
                empty_bath_states.append(empty_states)
        for filled_orbitals, empty_orbitals in zip(filled_bath_states, empty_bath_states):
            if len(filled_orbitals) > 1:
                ground_state_restrictions[filled_orbitals] = (len(filled_orbitals) - 1, len(filled_orbitals))
            if len(empty_orbitals) > 1:
                ground_state_restrictions[empty_orbitals] = (0, 1)
        if sum(len(rest) for rest in ground_state_restrictions.keys()) == 0:
            return None
        if self.verbose:
            print("Ground state restrictions:")
            for indices, occupations in ground_state_restrictions.items():
                print(f"---> {sorted(indices)} : {occupations}")
        return ground_state_restrictions

    def build_excited_restrictions(
        self,
        op: ManyBodyOperator,
        psis: Optional[list[ManyBodyState] | ManyBodyState] = None,
        es: Optional[list[float]] = None,
        imp_change: Optional[dict[int, tuple[int, int]]] = None,
        val_change: Optional[dict[int, tuple[int, int]]] = None,
        con_change: Optional[dict[int, tuple[int, int]]] = None,
        cutoff: float = 1e-6,
        min_dist=4,
        coupling_cutoff: float = 1e-3,
    ):
        """
        Construct restrictions for impurity occupation, valence bath occupation, and conduction bath occupation.
        Restrictions are formed by identifying which bath states are filled, empty, and partially filled, using the density matrices of states psis.
        Filled states are restricted to containing a maximum of one hole. Empty states can have a maximum of one electron. Partially filled states can be empty or filled.
        The total occupations of the impurity, valence band, and conduction band is limited by the ground state occupations with an optional occupation change.
        Arguments:
        =========
        psis: list[ManyBodyState] | ManyBodyState - Eigenstates used to calculate the density matrices.
        imp_change: Optional[tuple[int, int]] - Tuple containing the maximum deviation of the impurity occupation from the ground state occupations, (max_decrease, max_increasess). Default = None
        val_change: Optional[tuple[int, int]] - Tuple containing the maximum deviation of the valence bath occupation from the ground state occupations, (max_decrease, max_increasess). Default = None
        con_change: Optional[tuple[int, int]] - Tuple containing the maximum deviation of the conduction occupation from the ground state occupations, (max_decrease, max_increasess). Default = None
        occ_cutoff: float - Cutoff separating filled, partially filled, and empty states. Filled stated have occupation > 1-occ_cutoff, empty states have occupation < occ_cutoff, partially filled states lie in between
        Returns:
        ========
        excited restrictions: Optional[dict[frozenset[int], tuple[int, int]]] - The occupation restrictions of various orbital indices, or None if no restrictions are generated
        """
        if isinstance(psis, ManyBodyState):
            psis = [psis]
        if psis is not None and len(psis) == 0:
            return None
        if imp_change is None:
            imp_change = dict.fromkeys(self.impurity_orbitals)
        if val_change is None:
            val_change = dict.fromkeys(self.impurity_orbitals)
        if con_change is None:
            con_change = dict.fromkeys(self.impurity_orbitals)

        ground_state_restrictions = self.get_effective_restrictions()
        excited_restrictions = {}

        valence_baths, conduction_baths = self.bath_states

        filled_bath_states = []
        empty_bath_states = []

        all_impurity_orbitals = [
            orb for orb_blocks in self.impurity_orbitals.values() for orb_block in orb_blocks for orb in orb_block
        ]
        all_valence_orbitals = [
            orb for orb_blocks in valence_baths.values() for orb_block in orb_blocks for orb in orb_block
        ]
        all_conduction_orbitals = [
            orb for orb_blocks in conduction_baths.values() for orb_block in orb_blocks for orb in orb_block
        ]
        tot_orb = len(all_impurity_orbitals) + len(all_valence_orbitals) + len(all_conduction_orbitals)
        dist_matrix, dist_cutoff = self._impurity_coupling_distance(
            op, tot_orb, all_impurity_orbitals, coupling_cutoff, min_dist
        )
        if False and self.verbose:
            matrix_print(dist_matrix[:, :], "Orbital distance matrix", flush=True)

        # Impurity occupation is confined on the *whole* impurity as one subset (never per
        # orbital-symmetry group), so the window bounds only the total impurity charge and does
        # not pin S_z or the eg/t2g ratio. Combine the per-group requested changes into a single
        # (max decrease, max increase); if any group is unconstrained (None), the union is too.
        all_imp_orbs = frozenset(sorted(all_impurity_orbitals))
        imp_changes = [imp_change[i] for i in self.impurity_orbitals]
        if imp_changes and all(c is not None for c in imp_changes):
            combined_imp_change = (max(c[0] for c in imp_changes), max(c[1] for c in imp_changes))
        else:
            combined_imp_change = None
        imp_min, imp_max = self._get_updated_occ_restrictions(
            ground_state_restrictions, all_imp_orbs, combined_imp_change
        )

        for i, impurity_orbitals in self.impurity_orbitals.items():
            val_orbs = frozenset(sorted(orb for block in valence_baths[i] for orb in block))
            min_val, max_val = self._get_updated_occ_restrictions(ground_state_restrictions, val_orbs, val_change[i])
            con_orbs = frozenset(sorted(orb for block in conduction_baths[i] for orb in block))
            min_con, max_con = self._get_updated_occ_restrictions(ground_state_restrictions, con_orbs, con_change[i])

            if self.chain_restrict:
                for imp_orb_block, val_orb_block, con_orb_block in zip(
                    impurity_orbitals, valence_baths[i], conduction_baths[i]
                ):
                    if psis is not None:
                        val_rhos = self.build_density_matrices(psis, val_orb_block, val_orb_block)
                        con_rhos = self.build_density_matrices(psis, con_orb_block, con_orb_block)
                        valence_occupations = thermal_average_scale_indep(
                            es, np.diagonal(val_rhos.real, axis1=1, axis2=2), self.tau
                        )
                        conduction_occupations = thermal_average_scale_indep(
                            es, np.diagonal(con_rhos.real, axis1=1, axis2=2), self.tau
                        )
                    else:
                        valence_occupations = np.ones(len(val_orb_block))
                        conduction_occupations = np.zeros(len(con_orb_block))

                    # Identify filled and empty bath states
                    # Ignore states that are too close to the impurity
                    filled_valence_states = [
                        val_orb_block[orb]
                        for orb in np.nonzero(valence_occupations > 1 - cutoff)[0]
                        if np.min(dist_matrix[np.ix_(imp_orb_block, [val_orb_block[orb]])]) > dist_cutoff
                    ]
                    filled_conduction_states = [
                        con_orb_block[orb]
                        for orb in np.nonzero(conduction_occupations > 1 - cutoff)[0]
                        if np.min(dist_matrix[np.ix_(imp_orb_block, [con_orb_block[orb]])]) > dist_cutoff
                    ]
                    filled_states = frozenset(sorted(filled_valence_states + filled_conduction_states))
                    empty_valence_states = [
                        val_orb_block[orb]
                        for orb in np.nonzero(valence_occupations < cutoff)[0]
                        if np.min(dist_matrix[np.ix_(imp_orb_block, [val_orb_block[orb]])]) > dist_cutoff
                    ]
                    empty_conduction_states = [
                        con_orb_block[orb]
                        for orb in np.nonzero(conduction_occupations < cutoff)[0]
                        if np.min(dist_matrix[np.ix_(imp_orb_block, [con_orb_block[orb]])]) > dist_cutoff
                    ]
                    min_val = max(min_val - len(filled_valence_states) - len(empty_valence_states), 0)
                    max_con = max(max_con - len(empty_conduction_states) - len(filled_conduction_states), 0)
                    empty_states = frozenset(sorted(empty_valence_states + empty_conduction_states))
                    filled_bath_states.append(filled_states)
                    empty_bath_states.append(empty_states)

                if self.collapse_chains:
                    filled_bath_states = [
                        frozenset(sorted(orbs for filled_orbs in filled_bath_states for orbs in filled_orbs))
                    ]
                    empty_bath_states = [
                        frozenset(sorted(orbs for empty_orbs in empty_bath_states for orbs in empty_orbs))
                    ]
            else:
                empty_bath_states = [frozenset()]
                filled_bath_states = [frozenset()]
            new_valence_indices = frozenset(
                sorted(orb for orb in val_orbs if not any(orb in s for s in filled_bath_states + empty_bath_states))
            )
            new_conduction_indices = frozenset(
                sorted(orb for orb in con_orbs if not any(orb in s for s in filled_bath_states + empty_bath_states))
            )
            if len(new_valence_indices) > 0 and min_val > 0:
                excited_restrictions[new_valence_indices] = (min_val, len(new_valence_indices))
            if len(new_conduction_indices) > 0 and max_con < len(new_conduction_indices):
                excited_restrictions[new_conduction_indices] = (0, max_con)
            for filled_orbitals, empty_orbitals in zip(filled_bath_states, empty_bath_states):
                if len(filled_orbitals) > 2:
                    excited_restrictions[filled_orbitals] = (len(filled_orbitals) // 2, len(filled_orbitals))
                    # excited_restrictions[filled_orbitals] = (len(filled_orbitals) - 1, len(filled_orbitals))
                if len(empty_orbitals) > 2:
                    excited_restrictions[empty_orbitals] = (0, ceil(len(empty_orbitals) / 2))
                    # excited_restrictions[empty_orbitals] = (0, 1)
        # Emit the single whole-impurity occupation window (only when it actually confines).
        if len(all_imp_orbs) > 0 and (imp_min > 0 or imp_max < len(all_imp_orbs)):
            excited_restrictions[all_imp_orbs] = (imp_min, imp_max)
        if sum(len(rest) for rest in excited_restrictions.keys()) == 0:
            return None
        return excited_restrictions

    def __init__(
        self,
        impurity_orbitals,
        bath_states,
        nominal_impurity_occ=None,
        mixed_valence=None,
        initial_basis=None,
        restrictions=None,
        weighted_restrictions=None,
        split_threshold=1.0,
        delta_valence_occ=None,
        delta_conduction_occ=None,
        delta_impurity_occ=None,
        truncation_threshold=np.inf,
        spin_flip_dj=False,
        tau=0,
        chain_restrict=False,
        collapse_chains=False,
        comm=None,
        verbose=True,
        debug=False,
    ):
        """Initialize the Basis class.

        Parameters
        ----------
        impurity_orbitals : dict
            Impurity orbitals.
        bath_states : tuple of dict
            Valence and conduction bath states.
        nominal_impurity_occ : dict, optional
            Nominal impurity occupations.
        mixed_valence : dict, optional
            Mixed valence bounds.
        initial_basis : list of SlaterDeterminant or bytes, optional
            Predefined initial states.
        restrictions : dict, optional
            Initial occupation restrictions.
        delta_valence_occ : dict, optional
            Allowed valence occupation variations.
        delta_conduction_occ : dict, optional
            Allowed conduction occupation variations.
        delta_impurity_occ : dict, optional
            Allowed impurity occupation variations.
        truncation_threshold : float, default np.inf
            Threshold for truncating states.
        spin_flip_dj : bool, default False
            Whether to enable spin-flip states.
        tau : float, default 0
            Tau parameter.
        chain_restrict : bool, default False
            Whether to restrict chain states.
        collapse_chains : bool, default False
            Whether to collapse chains.
        comm : MPI.Comm, optional
            MPI communicator.
        verbose : bool, default True
            Whether to print info.
        debug : bool, default False
            Debug flag.
        """
        assert (
            impurity_orbitals is not None
        ), "You need to supply the number of impurity orbitals in each set in impurity_orbitals"
        assert bath_states is not None, "You need to supply the number of bath states for each l quantum number"

        self.num_spin_orbitals = sum(
            sum(len(orbs) for orbs in impurity_orbitals[i])
            + sum(len(orbs) for orbs in bath_states[0][i])
            + sum(len(orbs) for orbs in bath_states[1][i])
            for i in bath_states[0]
        )
        test = ManyBodyState({SlaterDeterminant.from_bytes(b"\x00"): 1.0})
        slater_det = list(test.keys())[0]
        self.type = type(slater_det)
        self.n_bytes = int(ceil(ceil(self.num_spin_orbitals / 8) / len(slater_det)) * len(slater_det))

        self.truncation_threshold = truncation_threshold
        self.is_distributed = comm is not None and comm.size > 1
        self.tau = tau

        if initial_basis is not None:
            assert nominal_impurity_occ is None
            assert delta_valence_occ is None
            assert delta_conduction_occ is None
            assert delta_impurity_occ is None
            initial_basis = [
                self.type.from_bytes(state) if isinstance(state, bytes) else state for state in initial_basis
            ]
        else:
            assert nominal_impurity_occ is not None
            initial_basis, num_spin_orbitals = self._get_initial_basis(
                impurity_orbitals=impurity_orbitals,
                bath_states=bath_states,
                delta_valence_occ=delta_valence_occ,
                delta_conduction_occ=delta_conduction_occ,
                delta_impurity_occ=delta_impurity_occ,
                nominal_impurity_occ=nominal_impurity_occ,
                mixed_valence=mixed_valence if mixed_valence is not None else dict.fromkeys(nominal_impurity_occ, 0),
                verbose=verbose,
            )
        self.impurity_orbitals = impurity_orbitals
        self.bath_states = bath_states
        self.spin_flip_dj = spin_flip_dj
        self.chain_restrict = chain_restrict
        self.collapse_chains = collapse_chains
        self.verbose = verbose
        self.debug = debug
        self.comm = comm
        self.restrictions = restrictions
        # Weighted-sum restrictions (e.g. S_z), list of (weights, (q_min, q_max)); see
        # ManyBodyOperator.set_weighted_restrictions. None = none.
        self.weighted_restrictions = weighted_restrictions
        # Adaptive MPI split policy (Phase 7): cap the number of split colors near the
        # participation ratio of the block costs, scaled by split_threshold. Larger =>
        # split more aggressively; 0 => never split (unified communicator). 1.0 keeps the
        # legacy max-split behaviour for equally-weighted blocks.
        self.split_threshold = split_threshold

        # Distributed determinant storage (formerly SimpleDistributedStateContainer):
        # the rank-local sorted determinant list, its state -> global-index dict, and the
        # rank-partition bookkeeping. States are hash-distributed across ranks; lookups and
        # retrievals use sparse point-to-point communication (graph_alltoall).
        self.rng = np.random.default_rng()
        self.local_basis = []
        self.offset = 0
        self.size = 0
        self.local_indices = range(0, 0)
        self._index_dict: dict = {}
        self.index_bounds = [None] * comm.size if self.is_distributed else [None]
        self.state_bounds = [None] * comm.size if self.is_distributed else [None]
        self.add_states(initial_basis)

    def clone(self, initial_basis=None, restrictions=None, weighted_restrictions=None, verbose=None, comm=None):
        """Create a new Basis instance, optionally overriding initial_basis and restrictions.

        If initial_basis is None, the new basis will start with self.local_basis.
        If restrictions is None, the new basis will inherit self.restrictions.
        If weighted_restrictions is None, the new basis inherits self.weighted_restrictions.
        If comm is None, the new basis will inherit self.comm.
        """
        return Basis(
            impurity_orbitals=self.impurity_orbitals,
            bath_states=self.bath_states,
            initial_basis=initial_basis if initial_basis is not None else list(self.local_basis),
            restrictions=restrictions if restrictions is not None else self.restrictions,
            weighted_restrictions=(
                weighted_restrictions if weighted_restrictions is not None else self.weighted_restrictions
            ),
            split_threshold=self.split_threshold,
            truncation_threshold=self.truncation_threshold,
            spin_flip_dj=self.spin_flip_dj,
            tau=self.tau,
            chain_restrict=self.chain_restrict,
            collapse_chains=self.collapse_chains,
            comm=comm if comm is not None else self.comm,
            verbose=verbose if verbose is not None else self.verbose,
            debug=self.debug,
        )

    def free_comm(self):
        """
        Free the split/custom MPI communicator associated with this Basis.
        This must be called collectively by all ranks sharing the communicator.
        """
        if self.comm is not None and self.comm != MPI.COMM_NULL:
            self.comm.Free()
            self.comm = None

    @staticmethod
    def _point2point(send_list, comm):
        """Sparse point-to-point MPI exchange of per-rank data lists."""
        return graph_alltoall(send_list, comm)

    def add_states(self, new_states: Iterable[bytes], unique_sorted=False) -> None:
        """
        Extend the current basis by adding the new_states to it.
        """
        new_states = [self.type.from_bytes(state) if isinstance(state, bytes) else state for state in new_states]
        if not self.is_distributed:
            existing_set = self._index_dict
            unique_new = [s for s in sorted(set(new_states)) if s not in existing_set]
            if unique_new:
                self.local_basis = list(merge(self.local_basis, unique_new))
                self.size = len(self.local_basis)
                self.offset = 0
                self.local_indices = range(0, len(self.local_basis))
                self._index_dict = {state: i for i, state in enumerate(self.local_basis)}
                if __debug__:
                    assert all(self.local_basis[i] < self.local_basis[i + 1] for i in range(len(self.local_basis) - 1))
            return

        from impurityModel.ed.mpi_comm import distribute_determinants

        unique_new_states = list(set(new_states))
        received_list = distribute_determinants(unique_new_states, self.n_bytes, self.comm)

        all_received = []
        for r_data in received_list:
            if r_data:
                all_received.extend(r_data)

        existing_set = self._index_dict
        unique_received = sorted(set(all_received))
        unique_new = [s for s in unique_received if s not in existing_set]

        local_added = len(unique_new)
        any_added = self.comm.allreduce(local_added, op=MPI.SUM)
        if any_added == 0:
            return

        if unique_new:
            self.local_basis = list(merge(self.local_basis, unique_new))

        local_length = len(self.local_basis)
        size_arr = np.array(self.comm.allgather(local_length), dtype=int)
        self.size = np.sum(size_arr)
        self.offset = np.sum(size_arr[: self.comm.rank])
        self.local_indices = range(self.offset, self.offset + len(self.local_basis))
        self.index_bounds = [np.sum(size_arr[: r + 1]) if size_arr[r] > 0 else None for r in range(self.comm.size)]
        self._index_dict = {state: self.offset + i for i, state in enumerate(self.local_basis)}
        state_bounds = list(self._getitem_sequence([i for i in self.index_bounds if i is not None and i < self.size]))
        self.state_bounds = state_bounds + [None] * (self.comm.size - len(state_bounds))
        self.state_bounds = [
            (
                self.state_bounds[r]
                if r < self.comm.size - 1 and self.state_bounds[r] != self.state_bounds[r + 1]
                else None
            )
            for r in range(self.comm.size)
        ]
        if __debug__:
            assert all(self.local_basis[i] < self.local_basis[i + 1] for i in range(len(self.local_basis) - 1))

    def redistribute_psis(self, psis: list[ManyBodyState]) -> list[ManyBodyState]:
        """Redistribute wavefunctions across MPI ranks based on state ownership.

        Parameters
        ----------
        psis : list of ManyBodyState
            The wavefunctions to redistribute.

        Returns
        -------
        list of ManyBodyState
            The redistributed wavefunctions.
        """
        if isinstance(psis, ManyBodyState):
            print("WARNING in redistribute_psi:")
            print(
                "Expetced a list of ManyBodyStates, received a single ManyBodyState. Remaking into list of one ManyBodyState"
            )
            psis = [psis]
        psis = [
            (
                psi
                if isinstance(psi, ManyBodyState)
                else ManyBodyState(
                    {(SlaterDeterminant.from_bytes(k) if isinstance(k, bytes) else k): v for k, v in psi.items()}
                )
            )
            for psi in psis
        ]
        if not self.is_distributed:
            return psis

        comm = self.comm

        res = graph_alltoall_psis(psis, self.n_bytes, comm)
        return res

    def _generate_spin_flipped_determinants(self, determinants: Iterable[SlaterDeterminant]) -> set[SlaterDeterminant]:
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
        valence_baths, conduction_baths = self.bath_states
        n_dn_op = {
            ((i, "c"), (i, "a")): 1.0
            for l in self.impurity_orbitals
            for i in range(sum(len(orbs) for orbs in self.impurity_orbitals[l]) // 2)
        }
        n_up_op = {
            ((i, "c"), (i, "a")): 1.0
            for l in self.impurity_orbitals
            for i in range(
                sum(len(orbs) for orbs in self.impurity_orbitals[l]) // 2,
                sum(len(orbs) for orbs in self.impurity_orbitals[l]),
            )
        }
        n_dn_mbo = ManyBodyOperator(n_dn_op)
        n_up_mbo = ManyBodyOperator(n_up_op)
        spin_flip = set()
        for det in determinants:
            n_dn = int(applyOp_test(n_dn_mbo, ManyBodyState({det: 1.0}), cutoff=0).get(det, 0).real)
            n_up = int(applyOp_test(n_up_mbo, ManyBodyState({det: 1.0}), cutoff=0).get(det, 0).real)
            spin_flip.add(det)
            to_flip = {det}
            for l in self.impurity_orbitals:
                n_orb = sum(len(orbs) for orbs in self.impurity_orbitals[l])
                for i in range(n_orb // 2):
                    spin_flip_op = {
                        ((i + n_orb // 2, "c"), (i, "a")): 1.0,
                        ((i, "c"), (i + n_orb // 2, "a")): 1.0,
                    }
                    spin_flip_mbo = ManyBodyOperator(spin_flip_op)
                    for state in list(to_flip):
                        flipped = applyOp_test(spin_flip_mbo, ManyBodyState({state: 1.0}), cutoff=0)
                        to_flip.update(flipped.keys())
                        if len(flipped) == 0:
                            continue
                        flipped_state = list(flipped.keys())[0]
                        new_n_dn = int(
                            applyOp_test(n_dn_mbo, ManyBodyState({flipped_state: 1.0}), cutoff=0)
                            .get(flipped_state, 0)
                            .real
                        )
                        new_n_up = int(
                            applyOp_test(n_up_mbo, ManyBodyState({flipped_state: 1.0}), cutoff=0)
                            .get(flipped_state, 0)
                            .real
                        )
                        if (new_n_dn == n_dn and new_n_up == n_up) or (new_n_dn == n_up and new_n_up == n_dn):
                            spin_flip.update(flipped.keys())

        return spin_flip

    def expand(self, op, slaterWeightMin=0, max_it=5):
        """
        Expand the basis by repeatedly applying an operator
        to the basis states, thus generating new basis states.
        The basis will probably explode!

        Parameters:
        ===========
        op: ManyBodyOperator - The operator to apply, over and over again.
        slaterWeightMin: float - Minimum weight for slater determinants to be kept (default: 0).
        max_it: int - Apply the operator at most this number of times (default: 5)

        Returns:
        ========
        op_dict: dict - Dictionary of slater determinants as keys and ManyBodyStates as values.
                        Applying the operator to a key results in the ManyBodyState mapped to
                        by that key.
        """
        if isinstance(op, dict):
            op = ManyBodyOperator(op)
        op.set_restrictions(self.restrictions)
        # Unconditional (like set_restrictions): passing None clears any stale weighted
        # mask left on a reused operator object.
        op.set_weighted_restrictions(self.weighted_restrictions)
        old_size = self.size - 1

        it = 0
        max_inner_loops = 2

        local_states = set(self.local_basis)
        apply_h_to_these = local_states
        while old_size < self.size and it < max(max_it // max_inner_loops, 1):
            for _ in range(max_inner_loops):
                new_local_states = set()
                for state in apply_h_to_these:
                    res = applyOp_test(
                        op,
                        ManyBodyState({state: 1}),
                        cutoff=slaterWeightMin,
                    )
                    new_local_states |= set(state for state in res.keys()) - local_states
                if len(new_local_states) == 0:
                    break
                apply_h_to_these = new_local_states
                local_states |= new_local_states
            new_states = local_states - set(self.local_basis)
            if self.spin_flip_dj:
                new_states = self._generate_spin_flipped_determinants(new_states)
            old_size = self.size

            n_new_states = len(new_states)
            if self.is_distributed:
                n_new_states = self.comm.allreduce(n_new_states, op=MPI.SUM)
            if self.size + n_new_states > self.truncation_threshold:
                break
            self.add_states(new_states)
            apply_h_to_these = apply_h_to_these ^ (set(self.local_basis) - local_states)
            it += 1
        if self.verbose:
            print(f"After expansion, the basis contains {self.size} elements.")

    def index(self, val: SlaterDeterminant) -> int:
        """Find the global index of a Slater determinant in the basis.

        Parameters
        ----------
        val : SlaterDeterminant
            The Slater determinant to look up.

        Returns
        -------
        int
            The global index of the determinant.

        Raises
        ------
        ValueError
            If any state is not found in the basis.
        TypeError
            If the query type is invalid.
        """
        if isinstance(val, bytes):
            val = self.type.from_bytes(val)
        if isinstance(val, self.type):
            res = next(self._index_sequence([val]))
            if res == self.size:
                raise ValueError(f"Could not find {val} in basis!")
            return res
        elif isinstance(val, Sequence) or isinstance(val, Iterable):
            converted = [self.type.from_bytes(x) if isinstance(x, bytes) else x for x in val]
            res = list(self._index_sequence(converted))
            for i, v in enumerate(res):
                if v >= self.size:
                    raise ValueError(f"Could not find {list(val)[i]} in basis!")
            return (i for i in res)
        else:
            raise TypeError(f"Invalid query type {type(val)}! Valid types are {self.type} and sequences thereof.")
        return None

    def __getitem__(self, key: int | slice) -> SlaterDeterminant | list[SlaterDeterminant]:
        """Get the Slater determinant(s) at the specified index or slice.

        Parameters
        ----------
        key : int or slice
            The index or slice of basis states to retrieve.

        Returns
        -------
        SlaterDeterminant or list of SlaterDeterminant
            The Slater determinant at the index, or list of determinants for a slice.

        Raises
        ------
        IndexError
            If the index is out of bounds or the state cannot be found.
        TypeError
            If the index type is invalid.
        """
        if isinstance(key, slice):
            start = key.start
            if start is None:
                start = 0
            elif start < 0:
                start = self.size + start
            stop = key.stop
            if stop is None:
                stop = self.size
            elif stop < 0:
                stop = self.size + stop
            step = key.step
            if step is None and start < stop:
                step = 1
            elif step is None:
                step = -1
            query = range(start, stop, step)
            result = list(self._getitem_sequence(query))
            for i, res in enumerate(result):
                if res == SlaterDeterminant.from_bytes(bytes(0)):
                    raise IndexError(f"Could not find index {query[i]} in basis with size {self.size}!")
            return (state for state in result)
        elif isinstance(key, Sequence) or isinstance(key, Iterable):
            result = list(self._getitem_sequence(key))
            for i, res in enumerate(result):
                if res == SlaterDeterminant.from_bytes(bytes(0)):
                    raise IndexError(f"Could not find index {key[i]} in basis with size {self.size}!")
            return (state for state in result)
        elif isinstance(key, int):
            result = next(self._getitem_sequence([key]))
            if result == SlaterDeterminant.from_bytes(bytes(0)):
                raise IndexError(f"Could not find index {key} in basis with size {self.size}!")
            return result
        else:
            raise TypeError(f"Invalid index type {type(key)}. Valid types are slice, Sequence and int")
        return None

    def __len__(self) -> int:
        """Get the total size of the basis.

        Returns
        -------
        int
            The total number of Slater determinants in the basis.
        """
        return self.size

    def __contains__(self, item: SlaterDeterminant | bytes) -> bool:
        """Check if a Slater determinant or its byte representation is in the basis.

        Parameters
        ----------
        item : SlaterDeterminant or bytes
            The state to search for.

        Returns
        -------
        bool
            True if the state is in the basis, False otherwise.
        """
        if isinstance(item, bytes):
            item = self.type.from_bytes(item)
        if not self.is_distributed:
            return item in self._index_dict
        return next(self._index_sequence([item])) != self.size

    def contains(self, item: Iterable[SlaterDeterminant | bytes]) -> np.ndarray:
        """Check containment for an iterable of states.

        Parameters
        ----------
        item : Iterable of SlaterDeterminant or bytes
            The collection of states to check.

        Returns
        -------
        np.ndarray of bool
            Boolean array indicating containment for each state.
        """
        if isinstance(item, bytes):
            item = self.type.from_bytes(item)
        if isinstance(item, self.type):
            return next(self._contains_sequence([item]))
        elif isinstance(item, Sequence) or isinstance(item, Iterable):
            converted = [self.type.from_bytes(x) if isinstance(x, bytes) else x for x in item]
            return self._contains_sequence(converted)
        return None

    def __iter__(self) -> Iterable[SlaterDeterminant]:
        """Iterate over all Slater determinants in the basis.

        Yields
        ------
        SlaterDeterminant
            The next Slater determinant in the basis.
        """
        chunk_size = 10000
        for i in range(0, self.size, chunk_size):
            chunk_end = min(i + chunk_size, self.size)
            chunk = self._getitem_sequence(range(i, chunk_end))
            for state in chunk:
                yield state

    def _getitem_sequence(self, l: Iterable[int]) -> Iterable[SlaterDeterminant]:
        """Retrieve the states for a sequence of global indices (sparse point-to-point)."""
        if not self.is_distributed:
            return (self.local_basis[i] for i in l)

        l = np.fromiter((i if i >= 0 else self.size + i for i in l), dtype=int)

        send_list: list[list[int]] = [[] for _ in range(self.comm.size)]
        send_to_ranks = np.empty((len(l)), dtype=int)
        send_to_ranks[:] = self.size
        for idx, i in enumerate(l):
            for r in range(self.comm.size):
                if self.index_bounds[r] is not None and i < self.index_bounds[r]:
                    send_list[r].append(i)
                    send_to_ranks[idx] = r
                    break
        send_order = np.argsort(send_to_ranks, kind="stable")

        queries = Basis._point2point(send_list, self.comm)

        results = [[] for _ in range(self.comm.size)]
        for r in range(len(queries)):
            for i, query in enumerate(queries[r]):
                if query >= self.offset and query < self.offset + len(self.local_basis):
                    results[r].append(self.local_basis[query - self.offset])

        result = [state for r_results in Basis._point2point(results, self.comm) for state in r_results]

        return (result[i] for i in np.argsort(send_order))

    def _index_sequence(self, s: Iterable[SlaterDeterminant]) -> Iterable[int]:
        """Find the global indices for a sequence of states (hash-routed lookups)."""
        if not self.is_distributed:
            return (self._index_dict.get(val, self.size) for val in s)

        s = list(s)
        send_list: list[list[SlaterDeterminant]] = [[] for _ in range(self.comm.size)]
        send_to_ranks = np.empty((len(s)), dtype=int)
        send_to_ranks[:] = self.size
        for i, val in enumerate(s):
            r = val.get_hash() % self.comm.size
            send_list[r].append(val)
            send_to_ranks[i] = r

        send_order = np.argsort(send_to_ranks, kind="stable")

        queries = Basis._point2point(send_list, self.comm)

        results = [[] for _ in range(self.comm.size)]
        for r in range(self.comm.size):
            for query in queries[r]:
                results[r].append(self._index_dict.get(query, self.size))
        result = np.array([i for r_i in Basis._point2point(results, self.comm) for i in r_i], dtype=int)
        if len(result) > 0:
            max_retries = 3
            retry_count = 0
            while np.any(np.logical_or(result > self.size, result < 0)) and retry_count < max_retries:
                mask = np.logical_or(result > self.size, result < 0)
                result[mask] = np.fromiter(
                    self._index_sequence(itertools.compress(s, mask)), dtype=int, count=int(np.sum(mask))
                )
                retry_count += 1

            if retry_count >= max_retries:
                import warnings

                warnings.warn(f"Failed to resolve all indices after {max_retries} retries")

        return (res for res in result[np.argsort(send_order)])

    def _contains_sequence(self, items) -> Iterable[bool]:
        """Check membership for a sequence of states."""
        if not self.is_distributed:
            return (item in self._index_dict for item in items)
        return (index < self.size for index in self._index_sequence(items))

    def copy(self) -> "Basis":
        """Create a copy of this Basis.

        Returns
        -------
        Basis
            A new Basis object with identical states and parameters.
        """
        return Basis(
            self.impurity_orbitals,
            self.bath_states,
            initial_basis=self.local_basis,
            restrictions=self.restrictions,
            weighted_restrictions=self.weighted_restrictions,
            split_threshold=self.split_threshold,
            spin_flip_dj=self.spin_flip_dj,
            chain_restrict=self.chain_restrict,
            collapse_chains=self.collapse_chains,
            comm=self.comm,
            truncation_threshold=self.truncation_threshold,
            verbose=self.verbose,
        )

    def clear(self) -> None:
        """Clear all states from the basis."""
        self.local_basis.clear()
        self.offset = 0
        self.size = 0
        self.local_indices = range(0, 0)
        self._index_dict = {}
        self.index_bounds = [None] * self.comm.size if self.is_distributed else [None]
        self.state_bounds = [None] * self.comm.size if self.is_distributed else [None]
        self.add_states([])

    def build_vector(
        self, psis: list[ManyBodyState], root: Optional[int] = None, slaterWeightMin: float = 0
    ) -> np.ndarray:
        """Build a dense matrix representation of wavefunctions in the basis.

        Parameters
        ----------
        psis : list of ManyBodyState
            The wavefunctions to represent.
        root : int, optional
            MPI rank to reduce the vector to. If None, it is reduced to all ranks.
        slaterWeightMin : float, default 0
            Minimum amplitude threshold below which coefficients are ignored.

        Returns
        -------
        np.ndarray
            The 2D dense matrix representation of the wavefunctions.
        """
        v = np.zeros((len(psis), self.size), dtype=complex, order="C")
        # psis = self.redistribute_psis(psis)
        # row_states_in_basis: list[bytes] = []
        # row_dict = {state: self._index_dict[state] for state in self.local_basis}
        # col_dict = dict(zip(self.local_basis, range(self.local_indices.start, self.local_indices.stop)))
        _index_dict = self._index_dict
        for row, psi in enumerate(psis):
            for state, val in psi.items():
                idx = _index_dict.get(state)
                if idx is None or abs(val) < slaterWeightMin:
                    continue
                v[row, idx] = val

        if self.is_distributed and root is None:
            self.comm.Allreduce(MPI.IN_PLACE, v, op=MPI.SUM)
        elif self.is_distributed:
            self.comm.Reduce(MPI.IN_PLACE if self.comm.rank == root else v, v, op=MPI.SUM, root=root)
        return v

    def build_distributed_vector(self, psis: list[ManyBodyState], dtype: Any = complex) -> np.ndarray:
        """Build the MPI-local portion of a wavefunction vector.

        Parameters
        ----------
        psis : list of ManyBodyState
            The wavefunctions to represent.
        dtype : Any, default complex
            The data type of the returned array.

        Returns
        -------
        np.ndarray
            The 2D array containing the local amplitudes.
        """
        psis = self.redistribute_psis(psis)
        v = np.empty((len(psis), len(self.local_basis)), dtype=dtype, order="C")
        for (row, psi), (col, state) in itertools.product(enumerate(psis), enumerate(self.local_basis)):
            v[row, col] = psi.get(state, 0)
        return v

    def build_state(self, vs: Union[list[np.ndarray], np.ndarray], slaterWeightMin: float = 0) -> list[ManyBodyState]:
        """Convert dense vectors back to a list of ManyBodyState objects.

        Parameters
        ----------
        vs : list of np.ndarray or np.ndarray
            The dense vector representations.
        slaterWeightMin : float, default 0
            Minimum amplitude threshold to keep.

        Returns
        -------
        list of ManyBodyState
            The corresponding list of many-body states.
        """
        if isinstance(vs, np.matrix):
            vs = vs.A
        if isinstance(vs, np.ndarray) and len(vs.shape) == 1:
            vs = vs.reshape((1, vs.shape[0]))
        if isinstance(vs, list):
            vs = np.array(vs)
        res = [ManyBodyState({}) for _ in range(vs.shape[0])]
        if vs.shape[1] == self.size:
            for j, i in np.argwhere(np.abs(vs[:, self.local_indices]) > slaterWeightMin):
                res[j][self.local_basis[i]] = vs[j, i + self.offset]
        elif vs.shape[1] == len(self.local_basis):
            for j, i in np.argwhere(np.abs(vs) > slaterWeightMin):
                res[j][self.local_basis[i]] = vs[j, i]
        else:
            raise RuntimeError(
                f"The dimensions of the input dense vector does not match a distributed, or full vector.\n{vs.shape} != ({vs.shape[0]}, {self.size}) || ({vs.shape[0]}, {len(self.local_basis)})"
            )
        return res

    def build_local_operator_list(self, op, slaterWeightMin):
        """
        Apply the operator to all (MPI local) basis states, in order.
        Return the results in a list.
        """
        res = []
        unit_state = ManyBodyState()
        for state in self.local_basis:
            unit_state[state] = 1.0
            res.append(applyOp_test(op, unit_state, cutoff=slaterWeightMin))
            unit_state.erase(state)
        return res

    def build_dense_matrix(self, op, distribute=True):
        """
        Get the operator as a dense matrix in the current basis.
        by default the dense matrix is distributed to all ranks.
        """
        h_local = self.build_sparse_matrix(op)
        if self.is_distributed:
            h = np.empty(h_local.shape, dtype=h_local.dtype)
            self.comm.Allreduce(h_local.todense(), h, op=MPI.SUM)
        else:
            h = h_local.todense(order="C")
        return h

    def build_sparse_matrix(self, op: ManyBodyOperator):
        """
        Get the operator as a sparse matrix in the current basis.
        The sparse matrix is distributed over all ranks.
        """
        if isinstance(op, dict):
            op = ManyBodyOperator(op)

        rows = []
        cols = []
        vals = []
        _index_dict = self._index_dict
        if not self.is_distributed:
            for ket, ket_state in zip(self.local_basis, self.build_local_operator_list(op, 0)):
                col = _index_dict[ket]
                for bra, val in ket_state.items():
                    row = _index_dict.get(bra)
                    if row is not None:
                        rows.append(row)
                        cols.append(col)
                        vals.append(val)
        else:
            columns = []
            bras = []
            values = []
            for ket, ket_state in zip(self.local_basis, self.build_local_operator_list(op, 0)):
                col = _index_dict[ket]
                for bra, val in ket_state.items():
                    columns.append(col)
                    bras.append(bra)
                    values.append(val)

            global_rows = list(self._index_sequence(bras))
            _size = self.size
            for row, col, val in zip(global_rows, columns, values):
                if row != _size:
                    rows.append(row)
                    cols.append(col)
                    vals.append(val)

        n = len(self)
        if rows:
            res = sp.sparse.csc_array((vals, (rows, cols)), shape=(n, n), dtype=complex)
        else:
            res = sp.sparse.csc_array((n, n), dtype=complex)
        return res

    @property
    def impurity_spin_orbital_indices(self):
        """Flat, sorted-by-orbital-set list of all impurity spin-orbital indices."""
        return [orb for blocks in self.impurity_orbitals.values() for block in blocks for orb in block]

    @property
    def valence_spin_orbital_indices(self):
        """Flat list of all valence-bath spin-orbital indices."""
        valence_baths, _conduction_baths = self.bath_states
        return [orb for blocks in valence_baths.values() for block in blocks for orb in block]

    @property
    def conduction_spin_orbital_indices(self):
        """Flat list of all conduction-bath spin-orbital indices."""
        _valence_baths, conduction_baths = self.bath_states
        return [orb for blocks in conduction_baths.values() for block in blocks for orb in block]

    def build_density_matrices(self, psis, orbital_indices_left=None, orbital_indices_right=None):
        r"""Compute single-particle density matrices for a list of many-body states.

        rho[n, i, j] = <psi_n| c_{orb_j}^dagger c_{orb_i} |psi_n>

        For the square case (orbital_indices_left == orbital_indices_right) the
        identity rho[i, j] = <phi_j | phi_i>  (where |phi_k> = c_{orb_k}|psi>)
        is exploited, cutting operator applications from O(n^2) to O(n) and
        halving inner products via Hermitian symmetry.

        For the general rectangular case we also exploit this decomposition
        rho[i, j] = <chi_j | phi_i> (where |chi_k> = c_{orb_k}|psi> and
        |phi_k> = c_{orb_k}|psi>), reducing operator applications to O(n).
        """
        if orbital_indices_left is None:
            orbital_indices_left = list(range(self.num_spin_orbitals))
        if orbital_indices_right is None:
            orbital_indices_right = list(range(self.num_spin_orbitals))
        n_left, n_right = len(orbital_indices_left), len(orbital_indices_right)
        rhos = np.zeros((len(psis), n_left, n_right), dtype=complex)

        square = orbital_indices_left == orbital_indices_right

        for n, psi_n in enumerate(psis):
            phi = [ManyBodyOperator({((orb, "a"),): 1.0})(psi_n, 0) for orb in orbital_indices_left]
            if square:
                chi = phi
            else:
                chi = [ManyBodyOperator({((orb, "a"),): 1.0})(psi_n, 0) for orb in orbital_indices_right]

            if self.is_distributed:
                phi = self.redistribute_psis(phi)
                if square:
                    chi = phi
                else:
                    chi = self.redistribute_psis(chi)

            from impurityModel.ed.ManyBodyUtils import inner_multi

            rhos[n] = inner_multi(chi, phi).T

        if self.is_distributed:
            self.comm.Allreduce(MPI.IN_PLACE, rhos, op=MPI.SUM)

        return rhos

    def split_basis_and_redistribute_psi(
        self, priorities: list[float], psis: Optional[list[ManyBodyState]]
    ) -> tuple[
        list[int], list[int], int, list[int], "Basis", Optional[list[ManyBodyState]], list[Optional[MPI.Intercomm]]
    ]:
        """Split the basis and redistribute wavefunctions over a split communicator.

        Parameters
        ----------
        priorities : list of float
            The split priority weights for each block.
        psis : list of ManyBodyState, optional
            The wavefunctions to redistribute, or None.

        Returns
        -------
        indices : list of int
            Representative indices.
        split_roots : list of int
            The roots for the split communicators.
        color : int
            The split communicator color rank.
        items_per_color : list of int
            Number of items assigned to each color group.
        split_basis : Basis
            The new Basis associated with the split communicator.
        psis : list of ManyBodyState, optional
            Redistributed wavefunctions.
        intercomms : list of MPI.Intercomm
            MPI intercommunicators (all set to None after being freed).
        """

        if (not self.is_distributed) or len(priorities) <= 1:
            return list(range(len(priorities))), [0], 0, [len(priorities)], self, psis, [None]

        comm = self.comm
        # All packing math (participation-ratio color cap, LPT unit packing,
        # largest-remainder rank apportionment) lives in _pack_units; it is pure and
        # deterministic, so every rank computes the identical packing.
        subgroups, procs_per_color = _pack_units(priorities, comm.size, self.split_threshold)
        if subgroups is None:
            # Unified: all ranks process every block together (no actual split).
            return list(range(len(priorities))), [0], 0, [len(priorities)], self, psis, [None]

        proc_cutoffs = np.cumsum(procs_per_color)
        color = int(np.argmax(comm.rank < proc_cutoffs))

        split_comm = comm.Split(color=color, key=comm.rank)
        split_roots = [0] + proc_cutoffs[:-1].tolist()
        items_per_color = [len(subgroup) for subgroup in subgroups]
        assert sum(items_per_color) == len(priorities)

        intercomms = []
        for c, c_root in enumerate(split_roots):
            if c == color:
                intercomms.append(None)
                continue
            intercomms.append(split_comm.Create_intercomm(0, comm, c_root))
        indices = sorted(subgroups[color])

        if split_comm.rank == 0:
            assert comm.rank in split_roots

        new_states = set(self.local_basis)
        # Distribute my local basis states among all other colors
        for c, c_root in enumerate(split_roots):
            # I will send  states to this color
            if color != c:
                serialized_local_basis = bytearray().join(
                    state.to_bytearray()[: self.n_bytes] for state in self.local_basis
                )
                intercomms[c].send(serialized_local_basis, dest=split_comm.rank % procs_per_color[c])
            # I will receive states from all other colors
            else:
                for send_color in range(len(split_roots)):
                    if send_color == color:
                        continue
                    for sender in range(procs_per_color[send_color]):
                        if sender % procs_per_color[c] != split_comm.rank:
                            continue
                        received_bytes = intercomms[send_color].recv(source=sender)
                        new_states.update(
                            self.type.from_bytes(bytes(received_bytes[i : i + self.n_bytes]))
                            for i in range(0, len(received_bytes), self.n_bytes)
                        )

        split_basis = Basis(
            self.impurity_orbitals,
            self.bath_states,
            initial_basis=list(new_states),
            restrictions=self.restrictions,
            weighted_restrictions=self.weighted_restrictions,
            split_threshold=self.split_threshold,
            chain_restrict=self.chain_restrict,
            collapse_chains=self.collapse_chains,
            comm=split_comm,
            verbose=self.verbose,
            truncation_threshold=self.truncation_threshold,
            tau=self.tau,
            spin_flip_dj=self.spin_flip_dj,
        )

        if psis is not None:
            new_psis = [p.copy() for p in psis]
            for c, c_root in enumerate(split_roots):
                if color != c:
                    serialized_psis = [{bytes(k.to_bytearray()[: self.n_bytes]): v for k, v in p.items()} for p in psis]
                    intercomms[c].send(serialized_psis, dest=split_comm.rank % procs_per_color[c])
                else:
                    for send_color in range(len(split_roots)):
                        if send_color == color:
                            continue
                        for sender in range(procs_per_color[send_color]):
                            if sender % procs_per_color[color] != split_comm.rank:
                                continue
                            received_psis = intercomms[send_color].recv(source=sender)
                            for i, received_psi in enumerate(received_psis):
                                new_psis[i] += ManyBodyState(
                                    {self.type.from_bytes(k): v for k, v in received_psi.items()}
                                )
            psis = split_basis.redistribute_psis(new_psis)

        # Free the intercommunicators collectively while all ranks are still
        # synchronised here.  MPI_Comm_free is collective — leaving the objects
        # for Python gc means they may be freed at different times on different
        # ranks, causing crashes.  The split_comm itself (split_basis.comm) must
        # NOT be freed here because the caller still needs split_basis.
        for ic in intercomms:
            if ic is not None and ic != MPI.COMM_NULL:
                ic.Free()

        return indices, split_roots, color, items_per_color, split_basis, psis, [None] * len(intercomms)
