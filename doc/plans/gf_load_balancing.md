# GF unit load balancing: a pull queue

## Problem

`run_units_distributed` packs Green's-function units statically. `_pack_units` runs LPT on
`unit_cost_weights = seed_mass * width` and then gives each colour ranks in proportion to its packed
mass, so it needs correct *absolute* weights twice. Seed mass cannot see the cost asymmetry between
the two spectral sides:

- **SrMnO3:** 76 removal units took about 3,350 s each and 76 addition units 20–450 s.
- **NiO:** electron addition pushes the impurity to d9 and above.
- **Empty conduction orbitals:** these make addition expensive in other systems too.

These cases are hard to predict in general, so the scheduler has to tolerate a wrong prediction.

## Evidence: replaying measured unit walls

These are the per-unit walls from the round-2 cluster kit (SrMnO3 archive, 152 units, one rank per
colour). Each run replays its own walls through list scheduling (`work_queue.queue_makespan`).

| ranks | measured GF wall | lower bound | static LPT, perfect weights | queue, ranked by actual size | queue, random order |
|---|---|---|---|---|---|
| 32 | 13,200 s | 7,656 s | 8,534 s | 8,745 s | 9,727 s |
| 64 | ~9,800 s | 5,260 s (one 5,260 s unit) | 5,260 s | 5,260 s | 5,933 s |
| 128 | 3,692 s | 3,784 s at 1 rank/colour | – | – | – |

- **No prediction needed for the main win.** A queue with no prediction at all beats the static
  packing 1.36x at 32 ranks and 1.65x at 64.
- **A good order gets close to the oracle.** The "actual size" column ranks by the measured retained
  size, which is only known after the fact. It is within 3% of the oracle.
- **128 ranks is bound by unit size, not scheduling.** There the static packer gave some heavy units
  2-rank colours (1.84x faster), so a queue on one-rank colours is about 2.5% slower until the head
  of the queue gets wider colours.

## Counter progress (measured 2026-10-07)

The counter is an int64 in an RMA window on rank 0, taken with `Fetch_and_op`. In the measurement
below, the host computed for 3.8 s without making any MPI call, as a 1-rank colour does for a whole
unit.

| transport | no poke | `Testall([])` | `Iprobe` on the window's comm | size-1 `Allreduce` |
|---|---|---|---|---|
| Open MPI 5.0.9, shared memory | 0.15 s | – | – | – |
| Open MPI 5.0.9, `osc rdma` + `btl tcp` | 3.80 s | 3.80 s | 0.001 s | 3.60 s |
| MPICH 4.2.2, default (one node) | 3.81 s | 3.80 s | 0.017 s (also with 4 VCIs) | 3.60 s |

- **The host has to poke.** `work_queue.queue_progress` is an `Iprobe` on the queue's own `Dup`,
  and it is a no-op on every rank that does not host a counter.
- **A progress thread would have worked but is ruled out.** It measured 0.1–0.3 s when the host
  released the GIL. However, RSPt initialises MPI with a plain `MPI_Init`, which gives THREAD_SINGLE.
- **The stall test only discriminates on MPICH.**
  `test_a_busy_host_that_calls_queue_progress_does_not_stall_fetches`, with its poke removed, fails
  under MPICH (1.40 s wait) and passes under Open MPI over shared memory.

## Dispatch order: the excited-sector dimension (measured 2026-10-07)

**Candidates.** All were scored on the production SrMnO3 archive, which is the same archive as the
cluster runs (md5 `79548b1c`, 152 units). The local proxies were matched to the cluster walls by
spectral side, because the cluster lines carry no unit index. A removal unit retains more than 100k
determinants, an addition unit about 17k.

- `dim`: `window_dimension` of the unit's excited window at its seeds' electron number.
- `h1`: the global support of `H` applied once to the seeds.
- `mass`: today's `seed_mass * width` weight.

Each makespan below is the median over 200 random assignments of each side's measured walls.

| proxy | AUC (removal ranked above addition) | queue makespan @32 | @64 |
|---|---|---|---|
| seed mass x width (current) | 0.959 | 10,267 s | 6,171 s |
| random | – | 9,860 s | 5,905 s |
| sector dimension | **1.000** | 9,312 s | 5,805 s |
| `H` probe (`h1`, `h1_new`, `h1/seed`) | 1.000 | 9,288–9,294 s | 5,791–5,817 s |

**What the dimensions look like.** Removal sectors hold 2,760,681 determinants (32 electrons) and
addition sectors 73,815 (34 electrons). That is a ratio of 37, close to the measured cost ratio.

**Why the current weight orders badly.** It ranks the sides almost correctly. The few heavy units it
misranks are dispatched last and become the stragglers, which is why it does worse than random.

**Why the dimension.** It ranks as well as the probe and costs nothing: no communication beyond one
small Allreduce, and no apply. The probe costs one `H` apply per unit on the full communicator, 28 s
in total here, and its memory grows with the ground state.

**NiO 10-bath star archive, cap 3e4 (local):** every unit hit the cap and took the frozen-CSR path,
so the walls (2–8 s) cannot rank anything.
- In this archive the *removal* window is the larger one (2.4e10 against 1.0e8 for addition).
- Removal units also ran more blocks (785–986 against 498–623), so the ordering is consistent there.
- Above the cap, the dimension says nothing about depth, so ties within a side are broken by seed mass.
- At the archive's auto cap (376k), and at 1e5, the run was OOM-killed on a 15 GB box after the
  ground state and before the GF units. This is unrelated to the scheduler and not investigated.

## Local A/B on a real workload (2026-10-07)

SrMnO3 `smo_causality_check` archive, -n 3 (three 1-rank colours), `GF_REAL_TOL=1e-6`. The TCP leg
runs over `--mca osc rdma --mca btl self,tcp --mca pml ob1`, the transport where the counter stalls
without a poke.

| kernel | case | static GF | queue GF | queue over TCP | Σ (both axes) | longest wait |
|---|---|---|---|---|---|---|
| Lanczos | cap 2e4, 8 units | 14.8 s | 11.1 s | 9.3 s | bit-identical | 0.12 s, 0.03 s (TCP) |
| BiCGSTAB | cap 3e3, 4 units, 8+20 points | 878.8 s | – | 543.2 s | bit-identical | **12.40 s** (TCP), flagged |

- **Lanczos.** The monitor hook keeps the waits at block length.
- **BiCGSTAB.** It pokes once per frequency point, and one point's solve took up to 12 s without
  returning to Python. That is about 2% of this run.
  - A per-iteration hook would go inside the Cython `block_bicgstab` loop, which needs a rebuild.
  - Measure on the cluster before adding it.

## Cluster A/B (2026-10-07)

`debug/gf_queue_kit/` on tree 89686a00 with Intel MPI 2021.16 (`FI_PROVIDER=cxi`), a `safe` build,
the SrMnO3 production archive and the Lanczos self-energy. Each job ran static then queue on the same
node.

| ranks | static GF | queue GF | gain | longest wait for a unit | queue colours |
|---|---|---|---|---|---|
| 32 | – | – | – | – | (pending) |
| 64 | 7,350 s | 5,560 s | 1.32x | 7.79 s | 64 x 1 rank |
| 128 | 3,710 s | 3,661 s | 1.3% faster | 0.40 s | 98, 30 of them with 2 ranks |

- **64 ranks.** The queue finished below the replayed 5,805 s. Static ran faster than the ~9,800 s
  of the earlier kit round, so the gain is 1.32x rather than the replayed 1.65x.
- **128 ranks.** The predicted 2.5% slowdown did not happen. The memory-sized colour count was 98,
  not 128, so 30 colours got a second rank. Which unit landed on those is not controlled (Step 5).
- **Dimension ordering without a window.** This archive has no occupation window. The count over all
  orbitals still separates the sides by electron number (2,760,681 against 73,815), and the measured
  bases agree (about 590k removal, 18k addition).
- **The 7.79 s wait at 64 ranks.** Its source is unknown. It is 0.14% of the phase and below the
  10 s flag.
- **Σ was not compared between schedulers on the cluster.** The kit does not save Σ. Σ is
  bit-identical only in the local A/B.

**Counter probe.** Rank 0 was busy for 20 s while every other rank fetched. All ranks were on one
node at every size, so the cross-node case is still unmeasured.

| ranks | no poke | `Iprobe` poke |
|---|---|---|
| 32 | 19.0 s | 0.001 s |
| 64 | 19.0 s | 0.002 s |
| 128 | 19.0 s | 0.006 s |

Intel MPI stalls the counter even within a node, so the poke is required in production.

## Status

`GF_SCHEDULER=queue` is the default after the 64- and 128-rank A/B, and `static` remains as an
opt-out. Still open:

- **The 32-rank row.**
- **One queue run each of BiCGSTAB, spectra and RIXS.** The A/B covered only the Lanczos
  self-energy.
- **A cross-node counter probe.**
- **Wider colours for the head of the queue.** At 128 ranks there are fewer heavy units (76) than
  ranks, so the run is bound by the slowest heavy unit. In the A/B, 30 colours had 2 ranks only
  because of the memory-sized colour count. Handing the head of the queue 2-rank colours on purpose
  (measured 1.84x per unit) would make that deliberate.
- **More hook sites, only if the A/B asks for them.** `queue_progress` is called from the GF
  convergence monitor (once per block, both Lanczos kernels) and from BiCGSTAB (once per frequency
  point). The frozen-CSR build is not hooked. Every queue stage prints `GF unit queue: ... longest
  wait for a unit`, flagged when over 10 s.
- **The dimension count is bounded.** `window_dimension` gives up past 20,000 dynamic-program states
  (six random overlapping sets ran over a minute), and the queue then orders by seed mass. The
  solver's windows (disjoint or nested sets) count in milliseconds.
