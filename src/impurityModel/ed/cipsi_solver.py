import itertools

import numpy as np
from mpi4py import MPI

from impurityModel.ed.eigensolvers import eigensystem
from impurityModel.ed.irlm import implicitly_restarted_block_lanczos_cy
from impurityModel.ed.manybody_basis import Basis, collective_amplitude_cutoff
from impurityModel.ed.ManyBodyUtils import ManyBodyBlockState, ManyBodyOperator, ManyBodyState
from impurityModel.ed.ManyBodyUtils import applyOp as applyOp_test
from impurityModel.ed.BlockLanczosArray import Reort, block_normalize
from impurityModel.ed.trlm import thick_restart_block_lanczos
from impurityModel.ed.basis_transcription import (
    build_distributed_vector,
    build_sparse_matrix,
    build_state,
    build_vector,
)

SOLVERS = {
    "trlm": thick_restart_block_lanczos,
    "irlm": implicitly_restarted_block_lanczos_cy,
}


class CIPSISolver:
    def __init__(self, basis: Basis):
        self.basis = basis
        self.psi_refs = None
        # Diagnostics of the latest candidate selection / basis truncation (see
        # determine_new_Dj / truncate); None until the corresponding event happens.
        self.last_selection = None
        self.last_truncation = None
        self.truncation_report = None

    def _allreduce_sum(self, value):
        if self.basis.is_distributed:
            return self.basis.comm.allreduce(value, op=MPI.SUM)
        return value

    def _allreduce_max(self, value):
        if self.basis.is_distributed:
            return self.basis.comm.allreduce(value, op=MPI.MAX)
        return value

    def truncate_initial(self, H: ManyBodyOperator) -> None:
        """Perform an initial truncation if the basis exceeds the truncation threshold."""
        if self.basis.size > self.basis.truncation_threshold and H is not None:
            if self.basis.verbose:
                print("Truncating basis!")
            H_sparse = build_sparse_matrix(self.basis, H)
            # Rank over the low-energy manifold (up to 10 states within energy_cut), not a
            # single eigenvector: the downstream expansion / get_eigenvectors keep the whole
            # near-degenerate ground manifold, so truncating to one state's support would
            # bias the retained determinants toward one member of the multiplet.
            e_ref, psi_ref = eigensystem(
                H_sparse,
                e_max=-self.basis.tau * np.log(1e-4),
                k=min(10, self.basis.size),
                eigenValueTol=0,
                comm=self.basis.comm,
                dense=False,
            )
            # eigensystem returns eigenvectors as columns (N, k); build_state wants one
            # row per state vector, so transpose (matches CIPSISolver.expand's usage).
            self.truncate(build_state(self.basis, psi_ref.T))

    def truncate(self, psis: list[ManyBodyState], target=None) -> list[ManyBodyState]:
        """Keep the globally top-``target`` determinants by eigenvector amplitude.

        Importance is the max ``|amplitude|^2`` over ``psis`` (each determinant counted
        once, on its hash-owner rank -- ``psis`` must be redistributed). The cutoff comes
        from the collective amplitude bisection, so every rank retains the identical set
        and the global count never exceeds ``target`` (ties at the cutoff are
        under-admitted). Collective on ``basis.comm``.
        """
        if target is None:
            target = self.basis.truncation_threshold
        keys, norms2 = ManyBodyBlockState.from_states(list(psis)).row_max_norms2()
        cutoff2 = collective_amplitude_cutoff(norms2, int(target), self.basis.comm)
        keep_mask = norms2 > cutoff2
        if self._allreduce_sum(int(np.count_nonzero(keep_mask))) == 0:
            # The bisection under-admits ties: if every candidate ties at the maximum
            # score the strict cutoff retains nothing. Keep the max-score tie class
            # (possibly exceeding target) rather than emptying the basis.
            global_max = self._allreduce_max(float(norms2.max()) if norms2.size else 0.0)
            if global_max > 0.0:
                keep_mask = norms2 >= global_max
        retained = set(itertools.compress(keys, keep_mask))
        kept_weight = self._allreduce_sum(float(norms2[keep_mask].sum()))
        total_weight = self._allreduce_sum(float(norms2.sum()))
        self.last_truncation = {
            "target": int(target),
            "retained": self._allreduce_sum(len(retained)),
            "discarded_weight": 1.0 - kept_weight / total_weight if total_weight > 0.0 else 0.0,
        }
        # Use the full container reset, not list.clear(): clearing only the local_basis
        # *list* leaves self.size and the _index_dict stale, so the subsequent
        # add_states() dedupes the (subset) trimmed states against the still-populated
        # index and repopulates nothing -- leaving size > 0 with an empty local_basis.
        # That desync later crashes build_state (IndexError) and collapses the Lanczos
        # seed block ("Block collapsed to zero rank").
        self.basis.clear()
        self.basis.add_states(retained)
        psis = [{state: amp for state, amp in psi.items() if state in retained} for psi in psis]
        return self.basis.redistribute_psis(psis)

    def _calc_de2(self, H, Hpsi_ref, e_ref: float, slaterWeightMin: float = 0):
        if isinstance(H, dict):
            H = ManyBodyOperator(H)

        _index_dict = self.basis._index_dict
        local_Djs = sorted({state for hp in Hpsi_ref for state in hp if state not in _index_dict})
        if not local_Djs:
            return local_Djs, np.zeros((len(Hpsi_ref), 0), dtype=complex)

        Dj_index = {Dj: j for j, Dj in enumerate(local_Djs)}
        overlaps = np.zeros((len(Hpsi_ref), len(local_Djs)), dtype=complex)
        for i, Hpsi_i in enumerate(Hpsi_ref):
            for state, amp in Hpsi_i.items():
                j = Dj_index.get(state)
                if j is not None:
                    overlaps[i, j] = amp

        # Diagonal probe <Dj|H|Dj> from a single H application to one superposition of
        # all candidates. Unit-modulus pseudo-random phases (derived from the
        # determinant hash, so deterministic and rank-independent) make the
        # candidate-candidate couplings enter with quasi-random phases instead of the
        # systematic offset an all-ones probe would add to the diagonal estimate.
        phases = np.exp(2j * np.pi * np.array([(hash(Dj) & 0xFFFF) / 65536.0 for Dj in local_Djs]))
        psi_all_Dj = ManyBodyState({Dj: phases[j] for j, Dj in enumerate(local_Djs)})
        H_psi_all = applyOp_test(H, psi_all_Dj, cutoff=slaterWeightMin)
        e_Dj = np.array(
            [np.real(np.conj(phases[j]) * H_psi_all.get(Dj, 0.0)) for j, Dj in enumerate(local_Djs)], dtype=float
        )

        # Epstein-Nesbet importance |<Dj|H|psi>|^2 / |E_ref - E_Dj| (the magnitude of
        # the second-order energy contribution). The denominator must not be a signed
        # clamp: candidates sit *above* E_ref for a ground-state search, so
        # max(E_ref - E_Dj, eps) would collapse to eps and turn the selection into a
        # bare coupling filter.
        de = np.maximum(np.abs(e_ref[:, None] - e_Dj[None, :]), 1e-12)
        de2 = np.zeros_like(overlaps)
        mask = np.abs(overlaps) > 1e-12
        de2[mask] = np.square(np.abs(overlaps[mask])) / de[mask]
        return local_Djs, de2

    def determine_new_Dj(
        self, e_ref, psi_ref, H, de2_min, slater_cutoff=0, return_Hpsi_ref=False, gen_ops=None, max_new=None
    ):
        """Select the candidate determinants to add to the basis.

        Candidates connected to ``psi_ref`` through ``H`` are kept when their de2
        importance (max over the reference states) reaches ``de2_min``. ``max_new``
        optionally caps the *global* number of selected candidates: the top ``max_new``
        by de2 importance are kept (collective bisection cutoff, ties under-admitted)
        before the symmetry closure, and ``self.last_selection`` records
        ``{"n_candidates", "n_admitted", "discarded_de2_mass"}``. Collective on
        ``basis.comm``.
        """
        Hpsi_ref = [applyOp_test(H, psi_i, cutoff=slater_cutoff) for psi_i in psi_ref]
        Hpsi_ref = self.basis.redistribute_psis(Hpsi_ref)
        local_Djs, de2 = self._calc_de2(H, Hpsi_ref, e_ref)
        scores = np.max(np.abs(de2), axis=0) if len(local_Djs) else np.zeros(0)
        de2_mask = scores >= de2_min
        n_candidates = self._allreduce_sum(int(np.count_nonzero(de2_mask)))
        discarded_de2_mass = 0.0
        if max_new is not None and n_candidates > max_new:
            comm = self.basis.comm if self.basis.is_distributed else None
            cutoff = collective_amplitude_cutoff(scores[de2_mask], int(max_new), comm)
            admitted = de2_mask & (scores > cutoff)
            if self._allreduce_sum(int(np.count_nonzero(admitted))) == 0:
                # All candidates tie at the maximum importance (the bisection
                # under-admits ties): admit the max-score tie class instead of nothing.
                global_max = self._allreduce_max(float(scores[de2_mask].max()) if np.any(de2_mask) else 0.0)
                if global_max > 0.0:
                    admitted = de2_mask & (scores >= global_max)
            discarded_de2_mass = self._allreduce_sum(float(scores[de2_mask & ~admitted].sum()))
            de2_mask = admitted
        self.last_selection = {
            "n_candidates": n_candidates,
            "n_admitted": self._allreduce_sum(int(np.count_nonzero(de2_mask))),
            "discarded_de2_mass": discarded_de2_mass,
        }
        new_Dj = set(itertools.compress(local_Djs, de2_mask))

        if gen_ops:
            import random

            unexplored_list = list(new_Dj)
            chunk_size = 1000

            while unexplored_list:
                chunk = unexplored_list[:chunk_size]
                unexplored_list = unexplored_list[chunk_size:]

                # Use random superpositions to avoid destructive interference
                chunk_state = ManyBodyState({state: random.random() + 1j * random.random() for state in chunk})

                while chunk_state:
                    next_chunk_state = ManyBodyState()
                    for op in gen_ops:
                        # Apply generator (cutoff=1e-12 to prune float noise)
                        psi_op = applyOp_test(op, chunk_state, cutoff=1e-12)

                        for state, amp in psi_op.items():
                            if state not in new_Dj:
                                new_Dj.add(state)
                                next_chunk_state[state] = amp

                    chunk_state = next_chunk_state

        if return_Hpsi_ref:
            return new_Dj, Hpsi_ref
        return new_Dj

    def expand(
        self,
        H,
        de2_min=1e-10,
        dense_cutoff=1e3,
        slaterWeightMin=0,
        solver="trlm",
        reort=Reort.PARTIAL,
        symmetry_generators=None,
        cap_e_tol=1e-8,
        max_cap_cycles=10,
    ):
        """Expand the basis variationally (CIPSI) until it stops growing.

        With a finite ``basis.truncation_threshold`` the expansion becomes a
        **fixed-budget CIPSI** once the cap binds: each cycle prunes the currently
        least important determinants (by eigenvector amplitude, collective top-K),
        admits the best de2-ranked candidates into the freed room, and
        re-diagonalizes; cycles stop when the ground-state energy changes by less
        than ``cap_e_tol`` or after ``max_cap_cycles`` cycles. ``truncation_report``
        records whether (and how) the cap bound the expansion.
        """
        if self.basis.restrictions is not None:
            H.set_restrictions(self.basis.restrictions)
        if self.basis.weighted_restrictions is not None:
            H.set_weighted_restrictions(self.basis.weighted_restrictions)
        de0_max = -self.basis.tau * np.log(1e-4)
        psi_refs = None

        if isinstance(H, dict):
            H = ManyBodyOperator(H)

        from impurityModel.ed.symmetries import (
            extract_tensors,
            discover_one_body_symmetries,
            tensors_to_operator,
        )

        if symmetry_generators is None:
            imp_orbs = []
            if getattr(self.basis, "impurity_orbitals", None):
                for orbs in self.basis.impurity_orbitals.values():
                    for o in orbs:
                        imp_orbs.extend(o)

            h, _, _ = extract_tensors(H, two_body=False)

            if imp_orbs:
                imp = sorted(list(set(imp_orbs)))
                h_imp = h[np.ix_(imp, imp)]
                imp_generators = discover_one_body_symmetries(h_imp)

                imp_map = {}
                for group, imp_blocks in self.basis.impurity_orbitals.items():
                    imp_blk = imp_blocks[0]
                    for idx_in_grp, o in enumerate(imp_blk):
                        imp_map[o] = (group, idx_in_grp)

                generators = []
                for g_imp in imp_generators:
                    g_full = np.zeros_like(h)
                    for i, oi in enumerate(imp):
                        for j, oj in enumerate(imp):
                            g_full[oi, oj] = g_imp[i, j]

                    for bath_dict in self.basis.bath_states:
                        if not bath_dict:
                            continue
                        n_bath = max([len(blks) for blks in bath_dict.values()], default=0)

                        for k in range(n_bath):
                            site_k_map = {}
                            for i, o_imp in enumerate(imp):
                                group, idx_in_grp = imp_map[o_imp]
                                if group in bath_dict and k < len(bath_dict[group]):
                                    if idx_in_grp < len(bath_dict[group][k]):
                                        site_k_map[i] = bath_dict[group][k][idx_in_grp]

                            for i in range(len(imp)):
                                if i not in site_k_map:
                                    continue
                                oi = site_k_map[i]
                                for j in range(len(imp)):
                                    if j not in site_k_map:
                                        continue
                                    oj = site_k_map[j]
                                    g_full[oi, oj] = g_imp[i, j]

                    if np.linalg.norm(h @ g_full - g_full @ h) < 1e-9:
                        generators.append(g_full)
            else:
                generators = discover_one_body_symmetries(h)
        else:
            generators = symmetry_generators

        gen_ops = []
        for g in generators:
            op = tensors_to_operator(g, tol=1e-12)
            if self.basis.restrictions is not None:
                op.set_restrictions(self.basis.restrictions)
            if self.basis.weighted_restrictions is not None:
                op.set_weighted_restrictions(self.basis.weighted_restrictions)
            gen_ops.append(op)

        threshold = self.basis.truncation_threshold
        capped = np.isfinite(threshold)
        cap_cycles = 0
        no_improve = 0
        e0 = np.inf
        best_e0 = None
        best_basis = None
        best_psis = None
        self.truncation_report = None
        while True:
            if solver in SOLVERS and self.basis.size >= dense_cutoff:
                restarted_lanczos = SOLVERS[solver]

                if psi_refs is None:
                    import random

                    # Per-rank seed (42 + rank) gives a different local random start on each
                    # rank. The *final energy* is reproducible across rank counts (it is what
                    # the tests compare), but the adaptively-selected basis *trajectory* is
                    # not bit-for-bit reproducible vs a different np, since the start vector
                    # differs. If cross-np basis reproducibility is ever required, seed from a
                    # single rank-independent global vector and scatter it instead.
                    rank = self.basis.comm.rank if self.basis.comm is not None else 0
                    random.seed(42 + rank)
                    local_states = list(self.basis.local_basis)
                    psi0_dict = {state: random.random() + 1j * random.random() for state in local_states}
                    psi0 = [ManyBodyState(psi0_dict)] if psi0_dict else [ManyBodyState()]
                    psi0 = self.basis.redistribute_psis(psi0)

                    N2s = np.array([psi.norm2() for psi in psi0], dtype=float)
                    if self.basis.is_distributed:
                        self.basis.comm.Allreduce(MPI.IN_PLACE, N2s, op=MPI.SUM)
                    psi0 = [psi / np.sqrt(N2s[i]) if N2s[i] > 0 else psi for i, psi in enumerate(psi0)]
                else:
                    psi0 = psi_refs

                num_wanted = min(2 * len(psi_refs) if psi_refs is not None else 10, len(self.basis))

                max_subspace = min(max(2 * num_wanted, num_wanted + 10), len(self.basis))
                if len(psi0) > 0:
                    try:
                        psi0, _ = block_normalize(psi0, self.basis.is_distributed, self.basis.comm, slaterWeightMin)
                    except Exception as e:
                        pass

                max_subspace_blocks = 2 * int(np.ceil(max_subspace / max(1, len(psi0)))) + 20
                if len(psi0) > 0:
                    max_blocks = max(2, len(self.basis) // len(psi0) - 1)
                    max_subspace_blocks = min(max_subspace_blocks, max_blocks)

                num_wanted = min(num_wanted, (max_subspace_blocks - 1) * len(psi0))

                # Iterative CIPSI uses the *array* IRLM (sparse H_mat, column-distributed
                # via local_indices). Memory note: the array MPI matvec forms the full
                # (global_N, n) dense partial product on every rank before the Allreduce
                # (see the guardrail in BlockLanczosArray.pyx), so per-rank memory scales
                # with the *global* sector dimension, not the local partition. This path is
                # for sectors small enough to also hold the sparse H_mat; very large sectors
                # should use the hash-distributed ManyBodyState kernel instead.
                H_mat = build_sparse_matrix(self.basis, H)
                if self.basis.is_distributed:
                    H_mat = H_mat[:, self.basis.local_indices]
                psi0_arr = (
                    build_distributed_vector(self.basis, psi0).T
                    if len(psi0) > 0
                    else np.zeros((len(self.basis.local_basis), 1), dtype=complex)
                )

                e_ref, psi_refs_arr = restarted_lanczos(
                    psi0=psi0_arr,
                    h_op=H_mat,
                    basis=self.basis,
                    num_wanted=num_wanted,
                    max_subspace_blocks=max_subspace_blocks,
                    tol=de2_min / 10,
                    max_restarts=10,
                    verbose=False,
                    slaterWeightMin=slaterWeightMin,
                    reort=reort,
                )

                if len(e_ref) > 0:
                    psi_refs = build_state(self.basis, psi_refs_arr.T, slaterWeightMin=slaterWeightMin)
                    e_min = np.min(e_ref)
                    valid_idx = [i for i, e in enumerate(e_ref) if e - e_min <= de0_max]
                else:
                    valid_idx = []
                if not valid_idx:
                    if self.basis.verbose:
                        print("No eigenvalues below energy threshold found.")
                    break
                e_ref = e_ref[valid_idx]
                psi_refs = [psi_refs[i] for i in valid_idx]
            else:
                H_mat = build_sparse_matrix(self.basis, H)
                v0 = (
                    build_vector(self.basis, psi_refs).T
                    if psi_refs is not None and self.basis.size >= dense_cutoff
                    else None
                )
                e_ref, psi_ref_dense = eigensystem(
                    H_mat,
                    e_max=de0_max,
                    k=2 * len(psi_refs) if psi_refs is not None else 10,
                    e0=None,
                    v0=v0,
                    eigenValueTol=0,
                    comm=self.basis.comm,
                    dense=self.basis.size < dense_cutoff,
                )
                psi_refs = build_state(self.basis, psi_ref_dense.T)

            if len(e_ref) == 0:
                break
            e0 = float(np.min(e_ref))
            if cap_cycles > 0:
                # Fixed-budget refinement: keep the best capped basis seen so far and
                # stop once cycles stop lowering the (variational) ground-state energy.
                improved = best_e0 is None or e0 < best_e0 - cap_e_tol
                if best_e0 is None or e0 < best_e0:
                    best_e0 = e0
                    best_basis = list(self.basis.local_basis)
                    best_psis = psi_refs
                no_improve = 0 if improved else no_improve + 1
                if no_improve >= 2 or cap_cycles >= max_cap_cycles:
                    break

            admit_target = None
            if capped:
                # Cap the selection so one cycle turns over at most ~10% of the basis
                # (or fills the remaining budget, whichever is larger).
                budget = int(threshold) - self.basis.size
                admit_target = max(budget, -(-int(threshold) // 10))
            new_Dj = self.determine_new_Dj(
                e_ref, psi_refs, H, de2_min, slater_cutoff=slaterWeightMin, gen_ops=gen_ops, max_new=admit_target
            )
            old_size = self.basis.size
            n_new = self._allreduce_sum(len(new_Dj))
            if capped and self.basis.size + n_new > threshold:
                # Fixed-budget CIPSI cycle: make room by dropping the currently least
                # important determinants (by eigenvector amplitude), then admit the
                # de2-ranked candidates; the loop head re-diagonalizes and the cycle
                # repeats until the energy stabilizes.
                cap_cycles += 1
                keep = max(int(threshold) - n_new, int(threshold) // 2, 1)
                psi_refs = self.truncate(psi_refs, target=keep)
                if self.basis.verbose:
                    print(
                        f"------> Basis truncated! (cycle {cap_cycles}: kept "
                        f"{self.last_truncation['retained']:,} determinants, admitting {n_new:,} candidates)"
                    )
            self.basis.add_states(new_Dj)
            psi_refs = self.basis.redistribute_psis(psi_refs)
            if cap_cycles == 0 and self.basis.size == old_size:
                break
            e0 = np.inf  # the basis was mutated; e0 no longer describes it
        if best_basis is not None and e0 > best_e0:
            # The last refinement cycle left a worse basis (e.g. score/amplitude
            # ping-pong): restore the best capped basis seen during the cycles.
            self.basis.clear()
            self.basis.add_states(best_basis)
            psi_refs = self.basis.redistribute_psis(best_psis)
        self.psi_refs = psi_refs
        if capped and self.basis.size > threshold and self.psi_refs is not None:
            # The symmetry closure can push an admission slightly past the cap; enforce
            # the hard threshold on exit (downstream get_eigenvectors re-solves).
            self.psi_refs = self.truncate(self.psi_refs)
        if cap_cycles > 0:
            sel = self.last_selection or {}
            self.truncation_report = {
                "cap_hit": True,
                "cycles": cap_cycles,
                "retained": int(self.basis.size),
                "threshold": int(threshold),
                "discarded_de2_mass": float(sel.get("discarded_de2_mass", 0.0)),
                "n_candidates_last": int(sel.get("n_candidates", 0)),
            }
            rank = self.basis.comm.rank if self.basis.is_distributed else 0
            if self.basis.verbose and rank == 0:
                rep = self.truncation_report
                print(
                    f"GS basis cap hit: fixed-budget CIPSI held the basis at "
                    f"{rep['retained']:,} determinants (truncation_threshold={rep['threshold']:,}) "
                    f"over {rep['cycles']} refinement cycle(s); the last cycle discarded "
                    f"candidates carrying {rep['discarded_de2_mass']:.3e} of PT2 importance. "
                    f"The ground state is exact on the retained subspace.",
                    flush=True,
                )

        if self.basis.verbose:
            print(f"After expansion, the basis contains {self.basis.size} elements.", flush=True)

    def get_eigenvectors(
        self,
        H,
        num_wanted: int,
        max_energy=None,
        dense_cutoff=1e3,
        slaterWeightMin=0,
        solver="trlm",
        reort=Reort.PARTIAL,
    ):
        if self.basis.restrictions is not None:
            H.set_restrictions(self.basis.restrictions)
        if self.basis.weighted_restrictions is not None:
            H.set_weighted_restrictions(self.basis.weighted_restrictions)

        if solver in SOLVERS and self.basis.size >= dense_cutoff:
            restarted_lanczos = SOLVERS[solver]

            import random

            rank = self.basis.comm.rank if self.basis.comm is not None else 0
            random.seed(42 + rank)
            local_states = list(self.basis.local_basis)
            if hasattr(self, "psi_refs") and self.psi_refs is not None:
                psi0 = self.psi_refs
            else:
                psi0 = [
                    ManyBodyState({state: random.random() + 1j * random.random() for state in local_states})
                    for _ in range(1)
                ]

            num_wanted = min(num_wanted + 10, len(self.basis))

            max_subspace = min(max(2 * num_wanted, num_wanted + 10), len(self.basis))
            if len(psi0) > 0:
                try:
                    psi0, _ = block_normalize(psi0, self.basis.is_distributed, self.basis.comm, slaterWeightMin)
                except Exception as e:
                    pass

            max_subspace_blocks = 2 * int(np.ceil(max_subspace / max(1, len(psi0)))) + 20
            if len(psi0) > 0:
                max_blocks = max(2, len(self.basis) // len(psi0) - 1)
                max_subspace_blocks = min(max_subspace_blocks, max_blocks)

            num_wanted = min(num_wanted, (max_subspace_blocks - 1) * len(psi0))

            H_mat = build_sparse_matrix(self.basis, H)
            if self.basis.is_distributed:
                H_mat = H_mat[:, self.basis.local_indices]

            psi0_arr = (
                build_distributed_vector(self.basis, psi0).T
                if len(psi0) > 0
                else np.zeros((len(self.basis.local_basis), 1), dtype=complex)
            )

            e_ref, psi_refs_arr = restarted_lanczos(
                psi0=psi0_arr,
                h_op=H_mat,
                basis=self.basis,
                num_wanted=num_wanted,
                max_subspace_blocks=max_subspace_blocks,
                tol=1e-8,
                max_restarts=100,
                verbose=self.basis.verbose and (self.basis.comm is None or self.basis.comm.rank == 0),
                slaterWeightMin=slaterWeightMin,
                reort=reort,
            )
            if len(e_ref) > 0:
                psi_refs = build_state(self.basis, psi_refs_arr.T, slaterWeightMin=slaterWeightMin)
            if max_energy is not None and len(e_ref) > 0:
                e_min = np.min(e_ref)
                valid_idx = [i for i, e in enumerate(e_ref) if e - e_min <= max_energy]
                e_ref = e_ref[valid_idx]
                psi_refs = [psi_refs[i] for i in valid_idx]

        else:
            H_mat = build_sparse_matrix(self.basis, H)
            e_ref, psi_ref_dense = eigensystem(
                H_mat,
                e_max=max_energy,
                k=num_wanted,
                e0=None,
                v0=None,
                eigenValueTol=0,
                comm=self.basis.comm,
                dense=self.basis.size < dense_cutoff,
                return_eigvecs=True,
            )
            psi_refs = build_state(self.basis, psi_ref_dense.T, slaterWeightMin=slaterWeightMin)

        return e_ref, psi_refs
