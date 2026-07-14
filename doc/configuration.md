# Configuration reference

Every runtime-tunable parameter of the solver stack is an environment variable read through
the central registry in `impurityModel/ed/config.py`. Set them in the shell (or a run script)
before launching a calculation:

```bash
GF_EIGENSTATE_GROUP=2 GF_SECTOR_CACHE_DIR=/scratch/sectors \
    mpiexec -n 16 python -m impurityModel.ed.selfenergy ...
```

Knobs are read **lazily** on every access, so a variable set at any point takes effect on
the next read. A default of *derived* means the value is computed at the call site (from the
available per-rank memory or the communicator size) unless the variable overrides it.

> This table is generated from `config.dump()`; edit the `Knob` declarations in
> `impurityModel/ed/config.py`, not this file. Regenerate with
> `python -m impurityModel.ed.config > doc/configuration.md` (then re-add this header).

### Per-frequency BiCGSTAB solver (``gf_method="bicgstab"``)

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `GF_BICGSTAB_ATOL` | float | `1e-08` | Absolute residual tolerance of a per-frequency BiCGSTAB solve. The default sits at the block-Lanczos reference's accuracy (doc/plans/bicgstab_per_frequency_gf.md, Phase 3a) -- inside the 2.5e-8 spread PARTIAL-vs-FULL reorthogonalization itself shows on the real workloads. The reliability diagnostics (gf_diagnostics.check_bicgstab_convergence) derive their thresholds from the value actually used; never re-hardcode it. |
| `GF_BICGSTAB_MAX_ITER` | int | `500` | Hard per-point iteration bound. Warm-started production solves measure ~3 iterations and a cold start ~6, so 500 is pathology headroom: a stagnating solve (a real-axis point within `delta` of a pole) ends and is *reported* by the diagnostics instead of iterating until the growing seen-support exhaustion bound -- which a solve that keeps discovering determinants may never reach. |
| `GF_BICGSTAB_RESTARTS` | int | `10` | Restarts granted to a non-converged point before the GMRES fallback. Restarting is the standard cure for BiCGSTAB's r0-orthogonality stagnation, which real-axis points within ~delta of a pole do hit. Progress-gated: a restart must shrink the residual by at least half to earn the next one, so a genuinely stuck point stops early and is reported. |
| `GF_GMRES_RESTART` | int | `40` | Krylov restart length of the GMRES fallback (the solver for points BiCGSTAB leaves unconverged: its shadow-residual recurrence stagnates near a pole, GMRES minimizes the residual and has no such mode). Bounds the fallback's live Krylov blocks, so the memory model (memory_estimate.estimate_gf_peak_bytes, method="bicgstab") reads the same knob. |
| `GF_GMRES_MAX_RESTARTS` | int | `25` | Maximum GMRES restart cycles before the point is reported as unconverged. |

### Spectrum slicing (``gf_method="sliced"``)

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `GF_SLICES` | int | `8` | Number of Chebyshev windows tiling the real-axis evaluation band. |
| `GF_SLICE_DEGREE` | int | `0` | Chebyshev filter degree; 0 = auto (derived from the bandwidth / slice-width ratio). |
| `GF_SLICE_TOL` | float | `0.0` | Amplitude truncation applied to the filtered slice seeds; 0 = no truncation. |

### Green's-function work-unit decomposition

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `GF_EIGENSTATE_GROUP` | int | `1` | Eigenstates stacked into one block-Lanczos work unit. Stacking shares the matvec/Krylov build across eigenstates but grows the per-step reorthogonalization with the block width, so the optimum is workload-dependent (doc/plans/calc_selfenergy_performance.md). The default (1) gives each eigenstate its own unit and its own Krylov space. |
| `GF_OPERATOR_SPLIT` | bool | `False` | Split each orbital block's Green's function into scalar (pairwise) continued fractions, one per operator column, instead of one block recurrence. Multiplies the number of independent work units -- better load balance for few large blocks -- at the cost of redundant Krylov building (no subspace shared across columns). Mutually exclusive with eigenstate grouping; the operator split wins when both are requested. |
| `GF_PER_STATE_RESTRICT` | bool | *derived* | Build the excited-sector occupation window per eigenstate rather than once for the thermal ensemble. Unset, it follows the basis's ``chain_restrict`` flag. It only matters when the bath classification is state-dependent (long chains, where distant sites clear the coupling-distance filter); for a directly-hybridizing single bath shell the per-state and ensemble windows are identical and this is a no-op. |

### Block-Lanczos convergence monitor

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `GF_CHECK_EVERY` | int | `8` | Blocks between convergence tests during the long approach. The test rebuilds the block continued fraction each call -- the single largest cost of the block-Lanczos Green's function (~53% of runtime at reort=NONE, measured) -- so while convergence is still far away it is sampled sparsely. Set to 1 to test every block. Once a check lands within GF_NEAR_FACTOR x tol the monitor switches to every block regardless, so the exact convergence point is caught with no added Lanczos steps and the converged G is unchanged. |
| `GF_NEAR_FACTOR` | float | `2.0` | Switch from sparse to per-block convergence sampling once the relative change is within this factor of the tolerance. Kept small: the relative change typically sits on a long noisy plateau a decade or two above tolerance before its final descent, and that plateau must stay in the sparse regime for the sampling to pay off. |

### RIXS shift-recycling solver tiers

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `GF_SECTOR_DENSE_MAX` | int | *derived* | Largest sector the RIXS R1 spectral cache (SectorResolventCache) may densify and eigendecompose. The eigendecomposition holds ~3 dense (N, N) complex arrays (H, the eigenvector matrix, LAPACK workspace); unset, the cap is derived so that fits in a quarter of the available per-rank memory. 0 disables the tier. |
| `GF_SECTOR_CACHE_DIR` | str | `''` | Directory persisting SectorResolventCache eigendecompositions across runs. Empty = in-memory only. With it, the dominant one-time `eigh` cost (measured ~450 s at 5565 determinants; OpenBLAS's Hermitian eigensolvers are bound by their non-parallelizing reduction stage, and the measured alternatives are no faster with eigenvectors) is paid once per material instead of once per run. |
| `GF_KRYLOV_RECYCLE_MAX_BYTES` | int | *derived* | Per-rank byte cap on a recycled Krylov store (KrylovShiftedResolvent: one block-Lanczos recurrence serving every shift of a fixed right-hand side). The retained Krylov basis is that tier's dominant allocation; unset, it is capped at a quarter of the available per-rank memory, mirroring GF_SECTOR_DENSE_MAX's budget. 0 disables the tier. |

### RIXS incoming-energy sampling

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `GF_RIXS_WIN_CHUNK` | int | *derived* | Incoming-energy points per RIXS work unit. A unit is (eigenstate x contiguous wIn-chunk); contiguity preserves the warm-start locality of consecutive points, and a unit is atomic (the engine never reorders within one). Unset, the default targets ~3 units per rank so the packing has slack to balance without fragmenting the warm-start chains; a serial run gets one unit per eigenstate (maximal locality). |
| `GF_RIXS_ADAPTIVE_TOL` | float | *derived* | Stop tolerance of the greedy adaptive wIn sampler (set-valued AAA): solve only the incoming energies the rational interpolant cannot yet predict to within this tolerance. Unset/empty disables it (dense sweep). Measured on NiO L3: 28 of 121 solves at 1e-4 relative error. |
| `GF_RIXS_ADAPTIVE_BATCH` | int | `1` | New wIn solves per adaptive round. Above 1 trades interpolation sharpness (each round's greedy pick is made with less information) for parallel width. |

