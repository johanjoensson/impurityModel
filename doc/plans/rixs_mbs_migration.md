# RIXS: status of the solver chain, and moving R3 off the array path

Written 2026-09-12, at the end of the sparse-matvec-exchange series
(`doc/plans/dc_smo_performance.md`, Phase 1c). Purpose: record what the RIXS map computes with
today, which of its stages already run on the `ManyBodyState` (MBS) kernels -- batched
`apply_block` plus the sparse `Neighbor_alltoallv` redistribution -- and which stage still runs
the array-path Green's function driver, so a later session can move that stage over. The
finding that makes this more than housekeeping is in the third section: **the array-path
driver's operator branch does not work on a communicator of more than one rank.**

## The chain today (`rixs.py`, `spectra._R1SolverChain`, `gf_solvers`)

`calc_tensor_map` (the production entry: Cartesian tensors, `getRIXSmap_tensor`) and
`calc_map` (per-operator-pair) share the structure; per eigenstate `|psi_e>` with energy `E_e`:

| stage | what | solver, in tier order | kernel family |
|---|---|---|---|
| seeds | `Tin_k |psi_e>` for every in-component `k` | `applyOp` per operator (`rixs.py:472`) | scalar `applyOp` (not batched) |
| **R1** | `(wIn + i delta1 + E_e - H)^-1 Tin|psi_e>` in the core-excited sector, per `wIn` | (1) `SectorResolventCache.try_solve` -- dense spectral, **declines any `comm.size > 1`**; (2) `KrylovShiftedResolvent` -- one distributed block-Lanczos recurrence, all shifts; (3) restarted per-point `block_bicgstab` with GMRES escalation | (2), (3): MBS -- `apply_block` + `redistribute_block` |
| **R2** | the in-component block of R1 solutions, one Krylov space per `wIn` | `cg.block_bicgstab` (`rixs.py:300-360`) | MBS |
| out seeds | `Tout_b |psi2_a>` for every (in `a`, out `b`) pair | `applyOp` per operator (`rixs.py:821`) | scalar `applyOp` |
| **R3** | `<s_ab| (wLoss + i delta2 + E_e - H)^-1 |s_a'b'>`, the full out-block resolvent | (1) `SectorResolventCache.try_eval` -- dense, **declines any `comm.size > 1`**; (2) **`gf_solvers.block_Green`** -- the array-path driver (`rixs.py:832`; `:688` in `calc_map`) | **array path** |

Two things follow from the table. Everything that touches the intermediate state (R1, R2) is
already on the MBS kernels, so it inherits the batched apply and the sparse graph exchange
through `redistribute_block` -> `graph_alltoall_block` -- nothing in the 2026-09 exchange work
touched it, and nothing needed to. R3, the emission-side resolvent, is the odd one out: on a
distributed colour it goes to `block_Green`, and `block_Green` is the one Green's-function driver
still built on dense per-rank vectors.

What the adaptive-`wIn` sampler (`GF_RIXS_ADAPTIVE_TOL`, `rixs._rixs_map_adaptive`) and the work
splitter (`gf_units.run_units_distributed`, `GF_RIXS_WIN_CHUNK`) do to this: each work unit is
(eigenstate x contiguous `wIn` chunk) and runs on a colour sub-communicator. **When there are at
least as many units as ranks, every colour is a single rank**, the dense caches serve both R1
and R3, and the array driver is never reached above the dense threshold. That is the
configuration every validation so far ran in (NiO L3 at `-n 2`, `rixs-r1-solver-chain` /
`rixs-adaptive-win-sampler-shipped` in the memory notes). The multi-rank-colour regime -- more
ranks than units, or colours merged by the memory budget -- is the production regime on a
cluster and is where the finding below applies.

## What `block_Green` (array path) does

`gf_solvers.block_Green` (`gf_solvers.py:44-312`):

- **Below 500 determinants** (`dense = len(basis) < 500`): `build_vector` gathers the seeds as
  a dense global `(n, N)` array on every rank (an `Allreduce` of the full vector),
  `build_dense_matrix` `Allreduce`s the full `(N, N)` Hamiltonian, and `block_lanczos_array`
  runs **without a communicator** -- a replicated serial Lanczos on every rank.
- **At 500 and above**: `_distributed_seed_qr` gives each rank its `(N_local, n)` slice of the
  QR'd seed block; `build_sparse_matrix(basis, hOp)[:, local_indices]` gives the
  `(global_N, N_local)` CSR; that CSR is wrapped in a `scipy.sparse.linalg.LinearOperator` whose
  `matmat` computes `h_local @ v` -- the **full `(global_N, n)` product** -- and `Reduce`s it to
  rank 0. The kernel receives an operator, so it takes its generic `h_op.dot` branch:
  neither the row-chunked reduce-scatter nor the new `MatvecExchangePlan` is involved.

### The operator branch fails on more than one rank

Probed 2026-09-12 (scratch script, reproduced below) on a 12-spin-orbital, 6-electron model --
924 determinants, so the operator branch -- with two width-1 seeds, `Reort.NONE`, `delta = 0.2`:

| driver | serial | `-n 2` |
|---|---|---|
| `block_Green` (array) | OK | **`ValueError: could not broadcast input array from shape (924,2) into shape (476,2)`** on rank 0, `(448,2)` on rank 1 |
| `block_Green_sparse` (MBS) | OK | OK; serial vs `-n 2` `max|dG| = 1.3e-9`; vs the array driver's serial result `1.3e-9` |

The mechanism is visible in the shapes: the wrapper returns `global_N` rows, the kernel's
`wp` buffer has `N_local` rows (`psi_dense_local` is the rank's slice), and the assignment
`wp_arr[:] = h_op.dot(q1)` in `block_lanczos_array_cy`'s generic branch cannot broadcast. Even
if it could, only rank 0 holds the reduced product; the other ranks would carry their partial
sums into the Gram matrices. So the branch is not "slow" or "unsparse" -- it does not run. It has
never been exercised by the suite on more than one rank: `test_gf_truncation.py`'s MPI test
drives `block_Green` at 12 determinants (the dense branch), and the RIXS MPI tests run with
single-rank colours.

Consequence for RIXS: any distributed run whose colours span two or more ranks and whose R3
`green_basis` exceeds 500 determinants (every real workload; NiO L3's R2 sector is 5,565) fails
in `eval_out`. That is the case `rixs-r1-solver-chain` recorded as a remaining follow-up
("distributed/oversized R2 still re-runs block-Lanczos per point") -- it re-runs a driver that
cannot complete.

Repro (serial OK, `mpiexec -n 2` fails; `which` is `array` or `sparse`):

```python
import itertools, sys
import numpy as np
from mpi4py import MPI
from impurityModel.ed.gf_solvers import block_Green, block_Green_sparse
from impurityModel.ed.greens_function import calc_G
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant
from impurityModel.ed.BlockLanczosCore import Reort

NSO, NE = 12, 6                       # C(12,6) = 924 determinants > 500
comm = MPI.COMM_WORLD if "--mpi" in sys.argv else None
def det(occ):
    b = [0, 0]
    for i in occ:
        b[i // 8] |= 1 << (7 - i % 8)
    return SlaterDeterminant.from_bytes(bytes(b))
terms = {((o, "c"), (o, "a")): -1.0 + 0.3 * o for o in range(NSO)}
for a in (0, 1):
    for b in range(2, NSO):
        terms[((a, "c"), (b, "a"))] = terms[((b, "c"), (a, "a"))] = 0.4
terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = 3.0
hOp = ManyBodyOperator(terms)
basis = Basis({0: [[0, 1]]}, ({0: [list(range(2, 7))]}, {0: [list(range(7, 12))]}),
              initial_basis=[det(o) for o in itertools.combinations(range(NSO), NE)], comm=comm, verbose=False)
full = [ManyBodyState({det([0, 2, 3, 4, 5, 6]): 1.0, det([1, 2, 3, 4, 5, 6]): 0.5}), ManyBodyState({det([0, 1, 2, 3, 4, 5]): 1.0})]
seeds = [ManyBodyState(dict(s.items()) if comm is None or comm.rank == 0 else {}, width=1) for s in full]
if comm is not None:
    seeds = basis.redistribute_psis(*seeds)   # SUMS per-rank copies: only rank 0 supplies amplitudes
fn = block_Green if sys.argv[1] == "array" else block_Green_sparse
alphas, betas, r = fn(hOp, seeds, basis, 0.2, Reort.NONE, verbose=False)
G = calc_G(alphas, betas, r, np.linspace(-6, 6, 25), 0.0, 0.2)
```

A strict-`xfail` MPI test pinning this (`test/gf/test_block_green_array_multirank.py`) ships
with this document so the migration flips it rather than rediscovering it.

## What "move R3 to the MBS path" means

The MBS driver `block_Green_sparse` already computes the same object (probe above: the two
drivers agree to `1e-9` serially) on a `Basis` it is allowed to grow, freezing growth at
`truncation_threshold` through `_CappedBasisProxy` (`doc/plans/gf_truncation` notes, PHP-exact
on the retained set). R3's `green_basis` is built by hand in `eval_out` -- `add_states` of every
out-seed's support, then `redistribute_psis` -- and then handed to `block_Green`, which does not
grow it. The migration is therefore:

1. **R3 solve**: replace `gf.block_Green(hOp, seeds, green_basis, delta2, Reort.NONE, ...)` at
   `rixs.py:832` (and `:688`) by `gf.block_Green_sparse(...)` on the same basis with its
   growth semantics decided explicitly: either keep today's "seed support only" basis by
   wrapping it in `_CappedBasisProxy(green_basis, green_basis.size)` (freeze immediately; PHP on
   the seed support -- what the array path computes today, and what `block_Green_cipsi` does at
   `gf_solvers.py:979`), or let it grow to the sector's `truncation_threshold` (what
   `greens_function`'s `sparse=True` path does for ordinary spectra, and strictly more accurate).
   The `info` dict contract (`converged`, `d_g`, `n_blocks`, `tol`) is the same on both drivers,
   so `solver_stats` needs no change. `calc_G` consumes the same `(alphas, betas, r)`.
2. **Out seeds**: `applyOp` per `(a, b)` pair at `rixs.py:821` walks the operator once per
   pair. `ManyBodyOperator.apply_block` on `from_states(psi2_all)` per out-operator (or the
   `Tout` block on the whole in-block) is the batched form the rest of the code uses; it also
   removes the per-pair `add_states` loop, since the block's union support is one call. Same
   for the in-seeds at `:472`.
3. **Dense tiers stay**: `SectorResolventCache.try_eval` is the fast path on single-rank
   colours and is untouched; it still declines `comm.size > 1`, so on multi-rank colours R3
   becomes `block_Green_sparse` instead of a crash.
4. **Retire `block_Green`'s operator branch**, or fix it -- once RIXS is moved, `block_Green`'s
   only remaining callers are `greens_function.py:1069` (`sparse=False`, the non-default
   `sparse_green=False` archive flag) and `test_gf_truncation.py`. The honest fix on the array
   side is to hand the kernel the `(global_N, N_local)` CSR directly (the kernel's sparse branch
   handles exactly that layout, with the row partition from its own `Allgather`, and now the
   graph exchange); the `LinearOperator` wrapper is the only thing standing in the way. Whether
   to do that or delete the branch is the migration session's call; deleting removes the last
   full-vector `Reduce`-to-root per matvec in the tree.

Verification for that session: the strict-xfail test flips to a pass (delete the marker);
`test_rixs_tensor.py`'s distributed tests at `-n 2` **and** `-n 3` (the latter is where a colour
can span two ranks when units are few -- check `GF_RIXS_WIN_CHUNK` forces fewer units than
ranks in at least one case, or add one); the NiO L3 map against `rixsgateB/rixs_dense.npz`
(`max rel err` was `1.5e-4` on the adaptive path).

## Hazards recorded elsewhere that apply here

- `redistribute_psis` SUMS replicated copies: seeds supplied on every rank double-count
  (`replicated-psis-double-count-in-mpi-tests`). The probe above and `test_gf_truncation.py`
  supply amplitudes on rank 0 only.
- Width-0 `ManyBodyState({})` placeholders on non-root ranks deadlock the redistribution;
  use `width=1` (`phase7-rename-landed-width0-deadlock-class`).
- `eval_out` runs hundreds of times per map; its solver warnings are aggregated in
  `solver_stats`, not printed (309 identical lines on the NiO validation run).
- `_R1SolverChain`'s tiers must stay collective-safe on every rank of a colour; a per-rank
  `verbose` must never gate a collective (`no-collectives-under-per-rank-verbose`).
