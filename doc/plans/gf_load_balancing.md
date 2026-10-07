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

## Status

`GF_SCHEDULER=queue` is opt-in, and the default stays `static` until the cluster A/B
(`debug/gf_queue_kit/`) confirms the replay. Still open:

- **Flip the default.** Do it if the A/B lands near the replayed makespans at 32 and 64 ranks, and
  stays within about 3% of static at 128.
- **Wider colours for the head of the queue.** At 128 ranks there are fewer heavy units (76) than
  ranks, so the run is bound by the slowest heavy unit on one rank. Giving the units at the head of the queue
  2-rank colours (measured 1.84x) is what static packing got by accident.
- **More hook sites, only if the A/B asks for them.** `queue_progress` is called from the GF
  convergence monitor (once per block, both Lanczos kernels) and from BiCGSTAB (once per frequency
  point). The frozen-CSR build is not hooked. Every queue stage prints `GF unit queue: ... longest
  wait for a unit`, flagged when over 10 s.
- **What the A/B covers.** The kit runs only the Lanczos self-energy. Before flipping the default for
  BiCGSTAB, spectra and RIXS, check their queue wait lines on one run each.
- **The dimension count is bounded.** `window_dimension` gives up past 20,000 dynamic-program states
  (six random overlapping sets ran over a minute), and the queue then orders by seed mass. The
  solver's windows (disjoint or nested sets) count in milliseconds.
