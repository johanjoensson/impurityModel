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
