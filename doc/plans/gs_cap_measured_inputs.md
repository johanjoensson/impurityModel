# The ground-state cap from measured inputs

**Problem.** NiO linked-chain, iteration 1, 2 ranks, 5 GiB per rank: the auto ground-state cap was 429k
determinants, the Green's-function cap 4.25M. The run measured 1.1 GiB peak against 2.5 GiB predicted.
Two model-dependent inputs of `estimate_gs_peak_bytes` were constants measured once on SrMnO3 and applied to
every model: stored elements per determinant (100) and the CIPSI selection fan-out (40).

## What is measured now, and what is not

| input | source | NiO | SrMnO3 |
|---|---|---|---|
| stored elements per determinant | probe: mean rows of `H|d>` over a 15,000-determinant closure of the nominal seed | 16.7 | 41 |
| selection fan-out | probe: de-duplicated rows of one block `H|psi>` per determinant | 4.3 | 9 |
| width of the thermal window | probe: degeneracy of the seed's lowest level (exact by symmetry, a lower bound) | 3 | n/a |
| block width, states to retain | `GS_MAX_BLOCK_WIDTH`, `GS_NUM_WANTED` when set, else from the window width | | 5, 222 |
| byte sizes (24 B/CSR element, 16 B/complex, 161 B/determinant, 50 B/selection pair) | structural | | |
| Krylov copies in the window solve (2) | measured on NiO, see below | | |

The probe is `gs_connectivity.measure_gs_connectivity`. It never looks at an energy, so it is invariant under
the chemical-potential shift a double-counting search applies. Every driver measures on the same solver basis,
so the DC search and the self-energy size the same cap. It runs only when a cap is sized or reported (0.6 s on
NiO). A caller with no operator gets the SrMnO3 constants, and the log says `UNMEASURED`.

The seed's *thermal-window* state count was rejected as the window width: the seed's excited levels are not
converged, it counts 21 states on NiO where 3 are kept, and it cut the cap below the unmeasured one.

## Two phases, measured

Each ground-state solve now prints and traces (`gs_phase_memory`) the peak RSS of its class expansions and of
its thermal window solve. NiO iteration 1, serial, cap 20,000:

| phase | added memory per determinant |
|---|---|
| class expansions | 1.0 to 5.0 kB (model 3.3) |
| thermal window solve | 9.4 kB (model 7.5) |

The window solve holds the Krylov basis twice: the restart store (196 columns at a block of 7 and 13 retained;
3.1 kB per determinant) and the initial sweep's own basis. The estimate is the larger of the two phases.

## The thermal union and the cap

Before, `solve_certified` added the winner and all its spin partners before the first cap check and only the
expansion's exit truncated back, so the window solve could run on 2 to 3 times the cap at its higher
per-determinant price. The union is now built within the cap, decided on the exact replicated size. Classes at
the winning energy are cut by one common share of their determinants (ranked by the winner's amplitudes), with
partner rows regenerated from the cut classes. A union that fits is unchanged.

The price: when the cap is so tight that the winning level cannot fit, the thermal manifold comes back short
and says so (the d7 cubic fixture: 4 of 8 states at a cap of 200, where the old overshoot gave 8). NiO at a cap
of 20,000 returns all 3 states and the same energy. Production caps are far above the winner's size (115k of
680k), so the fit path is the one that runs.

## Caps

| | old model | measured model |
|---|---|---|
| NiO, 2 ranks, 5 GiB | 425,512 | 680,222 |
| NiO, 1 rank, 5 GiB | 214,937 | 350,045 |
| GF cap, NiO, 2 ranks | 4,249,820 | unchanged |
| SrMnO3 point (128 ranks, 2,745,510, W=5, nw=222): predicted vs measured 3.5 to 4.6 GiB | 0.35 GiB | 2.9 GiB |

The GS cap stays below the GF cap because a determinant costs the ground state a CSR row, a full-reorthogonalised
Krylov basis and a selection transient, and the Green's-function path about 0.5 kB. The ratio on NiO is 1/6.
SrMnO3 is still up to 1.6 times under-predicted at the production point; the measured-RSS guard of
`CIPSISolver.expand` polices what the model misses.
