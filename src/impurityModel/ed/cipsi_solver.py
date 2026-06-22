import itertools

import numpy as np
from mpi4py import MPI

from impurityModel.ed.finite import eigensystem
from impurityModel.ed.irlm import implicitly_restarted_block_lanczos
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed.ManyBodyUtils import applyOp as applyOp_test
from impurityModel.ed.block_math import block_normalize
from impurityModel.ed.lanczos import Reort
from impurityModel.ed.trlm import thick_restart_block_lanczos

SOLVERS = {
    "trlm": thick_restart_block_lanczos,
    "irlm": implicitly_restarted_block_lanczos,
}


class CIPSISolver:
    def __init__(self, basis: Basis):
        self.basis = basis

    def truncate_initial(self, H: ManyBodyOperator) -> None:
        """Perform an initial truncation if the basis exceeds the truncation threshold."""
        if self.basis.size > self.basis.truncation_threshold and H is not None:
            if self.basis.verbose:
                print("Truncating basis!")
            H_sparse = self.basis.build_sparse_matrix(H)
            e_ref, psi_ref = eigensystem(
                H_sparse,
                e_max=-self.basis.tau * np.log(1e-4),
                k=1,
                eigenValueTol=0,
                comm=self.basis.comm,
                dense=False,
            )
            self.truncate(self.basis.build_state(psi_ref))

    def truncate(self, psis: list[ManyBodyState]) -> list[ManyBodyState]:
        cutoff = np.finfo(float).eps
        self.basis.local_basis.clear()
        num_states = self.basis.comm.allreduce(max(len(psi) for psi in psis))
        while num_states > self.basis.truncation_threshold:
            psis = [{state: amp for state, amp in psi.items() if abs(amp) > cutoff} for psi in psis]
            num_states = self.basis.comm.allreduce(max(len(psi) for psi in psis))
            cutoff *= 10
        self.basis.add_states(state for psi in psis for state in psi)
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

        psi_all_Dj = ManyBodyState(dict.fromkeys(local_Djs, 1.0))
        H_psi_all = applyOp_test(H, psi_all_Dj, cutoff=slaterWeightMin)
        e_Dj = np.array([np.real(H_psi_all.get(Dj, 0.0)) for Dj in local_Djs], dtype=float)

        de = e_ref[:, None] - e_Dj[None, :]
        de = np.maximum(de, 1e-12)
        de2 = np.zeros_like(overlaps)
        mask = np.abs(overlaps) > 1e-12
        de2[mask] = np.square(np.abs(overlaps[mask])) / de[mask]
        return local_Djs, de2

    def determine_new_Dj(self, e_ref, psi_ref, H, de2_min, slater_cutoff=0, return_Hpsi_ref=False):
        Hpsi_ref = [applyOp_test(H, psi_i, cutoff=slater_cutoff) for psi_i in psi_ref]
        Hpsi_ref = self.basis.redistribute_psis(Hpsi_ref)
        local_Djs, de2 = self._calc_de2(H, Hpsi_ref, e_ref)
        de2_mask = np.any(np.abs(de2) >= de2_min, axis=0)
        new_Dj = set(itertools.compress(local_Djs, de2_mask))
        if return_Hpsi_ref:
            return new_Dj, Hpsi_ref
        return new_Dj

    def expand(self, H, de2_min=1e-10, dense_cutoff=1e3, slaterWeightMin=0, solver="trlm", reort=Reort.PARTIAL):
        if self.basis.restrictions is not None:
            H.set_restrictions(self.basis.restrictions)
        de0_max = -self.basis.tau * np.log(1e-4)
        psi_refs = None

        if isinstance(H, dict):
            H = ManyBodyOperator(H)

        old_size = self.basis.size - 1
        while old_size != self.basis.size:
            if solver in SOLVERS and self.basis.size >= dense_cutoff:
                restarted_lanczos = SOLVERS[solver]

                if psi_refs is None:
                    import random

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

                H_mat = self.basis.build_sparse_matrix(H)
                psi0_arr = (
                    self.basis.build_vector(psi0).T if len(psi0) > 0 else np.zeros((self.basis.size, 1), dtype=complex)
                )

                e_ref, psi_refs_arr = restarted_lanczos(
                    psi0=psi0_arr,
                    h_op=H_mat,
                    basis=self.basis,
                    num_wanted=num_wanted,
                    max_subspace_blocks=max_subspace_blocks,
                    tol=de2_min / 10,
                    max_restarts=10,
                    verbose=self.basis.verbose,
                    slaterWeightMin=slaterWeightMin,
                    reort=reort,
                )

                if len(e_ref) > 0:
                    psi_refs = self.basis.build_state(psi_refs_arr.T, slaterWeightMin=slaterWeightMin)
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
                H_mat = self.basis.build_sparse_matrix(H)
                v0 = (
                    self.basis.build_vector(psi_refs).T
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
                psi_refs = self.basis.build_state(psi_ref_dense.T)

            new_Dj = self.determine_new_Dj(e_ref, psi_refs, H, de2_min, slater_cutoff=slaterWeightMin)
            old_size = self.basis.size
            self.basis.add_states(new_Dj)
            psi_refs = self.basis.redistribute_psis(psi_refs)
            if self.basis.size > self.basis.truncation_threshold:
                psi_refs = self.truncate(psi_refs)
                if self.basis.verbose:
                    print("------> Basis truncated!")
                break
        self.psi_refs = psi_refs

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

            H_mat = self.basis.build_sparse_matrix(H)
            psi0_arr = (
                self.basis.build_vector(psi0).T if len(psi0) > 0 else np.zeros((self.basis.size, 1), dtype=complex)
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
                psi_refs = self.basis.build_state(psi_refs_arr.T, slaterWeightMin=slaterWeightMin)
            if max_energy is not None and len(e_ref) > 0:
                e_min = np.min(e_ref)
                valid_idx = [i for i, e in enumerate(e_ref) if e - e_min <= max_energy]
                e_ref = e_ref[valid_idx]
                psi_refs = [psi_refs[i] for i in valid_idx]

        else:
            H_mat = self.basis.build_sparse_matrix(H)
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
            psi_refs = self.basis.build_state(psi_ref_dense.T, slaterWeightMin=slaterWeightMin)

        return e_ref, psi_refs
