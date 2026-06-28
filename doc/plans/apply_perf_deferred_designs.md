# Design Sketches: Deferred `apply` / MPI Optimizations

Companion to `manybodyoperator_apply_performance.md`. Two deferred items, written up as
**designs with implementation sketches** (not yet weak-model checkboxes): the off-diagonal
**two-body masked sign** (Phase 2d) and the **NBX sparse count exchange** (Phase 4c). Both
include a correctness strategy so they can be implemented safely.

---

## A. Off-diagonal two-body masked sign

### A.1 The key result (it's not as hard as 2c's deferral implied)

For **any** product of creation/annihilation operators applied to `|state>` in stored
order `o_1 … o_n` (o_1 acts on the ket first), with target orbitals `p_1 … p_n`, the
fermion sign factorises into a **single state-dependent masked popcount plus a build-time
constant**:

```
sign(state) = (-1)^( mask_parity(state, M) + C )
```

Derivation. Each elementary operator contributes `(-1)^(occupied orbitals < p_m in the
state as it is when o_m acts)`. The state at step m differs from the original only at the
orbitals toggled by `o_1 … o_{m-1}` — and those positions are **fixed** (build-time
known). Split each step's prefix count into (i) the original-state prefix and (ii) a
correction from already-toggled orbitals below `p_m`:

```
E = Σ_m prefix_orig(p_m)            (state-dependent)
  + Σ_m #{ q in {p_1..p_{m-1}} : q < p_m }   (build-time constant)   (mod 2)
```

* `parity` is linear over `+`, so `Σ_m parity(prefix_orig(p_m)) = mask_parity(state, M)`
  with **`M = mask_lt(p_1) XOR mask_lt(p_2) XOR … XOR mask_lt(p_n)`** (bitwise XOR of the
  "orbitals strictly below `p_m`" masks — overlaps cancel in parity, exactly what XOR
  encodes). `mask_lt(p)` and `M` are built once.
* The second sum is **`C`** — a constant 0/1 from the fixed target ordering.

This **generalises the existing kernels**: 1-body (`c†_i c_j`) is the n=2 special case
(`M = mask_lt(i) XOR mask_lt(j)` = the "between" mask, `C = [j<i]`), and it extends to
2-body and beyond with no new runtime cost — still **one popcount**.

### A.2 The nonzero (occupancy) condition

The masked kernel must also cheaply reject terms that annihilate `|state>`. For a
normal-ordered term `c†_{i}c†_{j} c_{k}c_{l}` (Phase 3 gives `i<j`, `k<l`):

* **All four orbitals distinct** (the dominant pair-hop / exchange case): nonzero iff
  `k,l` occupied and `i,j` empty. Output toggles all four. → `occ_mask = {k,l}`,
  `vac_mask = {i,j}`.
* **One creation orbital == one annihilation orbital** (e.g. `c†_i c†_j c_j c_l = n_j ·
  c†_i c_l`): an occupancy-weighted 1-body hop. Reduces to the 1-body kernel between the
  *residual* orbitals plus an extra "shared orbital occupied" condition.

Deriving these masks by hand is error-prone for the overlap cases — so we don't trust the
derivation; we **probe-validate** it (next).

### A.3 Correctness guarantee: build-time probe (same pattern as the density kernel)

For each candidate term, after deriving `M`, `C`, `occ_mask`, `vac_mask` at build time,
**verify against the ground-truth `create`/`annihilate` sequence** on a handful of random
determinants, and only enable the masked kernel if every probe matches; otherwise fall
back to the general path. This makes correctness independent of derivation bugs — exactly
how `m_flat_density` already probes the all-occupied determinant.

```cpp
// build_flat_representation(), after deriving M/C/occ/vac for a 2-body candidate:
bool ok = true;
std::mt19937_64 rng(0x2b0d ^ op_idx);            // deterministic per term
for (int t = 0; t < N_PROBE && ok; ++t) {        // N_PROBE ~ 16
    key_type s = random_determinant(rng, n_chunks);
    // force the occupancy condition so we exercise the nonzero branch too:
    set_bits(s, occ_orbs); clear_bits(s, vac_orbs);
    // masked prediction
    bool nz = mask_occupied(s, occ_mask) && mask_empty(s, vac_mask);
    double masked = nz ? ((mask_parity(s, M) ^ C) ? -1.0 : 1.0) : 0.0;
    // ground truth via the existing in-place sequence
    double truth = reference_sign(s, indices);   // create/annihilate, 0 if it vanishes
    if (masked != truth) ok = false;
    // also probe a few states that VIOLATE occupancy -> truth must be 0
}
m_flat_twobody.push_back(ok ? 1 : 0);
```

`reference_sign` is a tiny helper that runs the current `create`/`annihilate` loop on a
copy and returns the signed result (0 if Pauli-blocked). Cost is build-time only.

### A.4 Apply sketch (mirrors the 1-body branch, both serial and PARALLEL paths)

```cpp
if (m_flat_twobody[op_idx]) {
    if (mask_occupied(slater, m_tb_occ[op_idx]) &&
        mask_empty(slater, m_tb_vac[op_idx])) {
        const double sgn = (mask_parity(slater, m_tb_M[op_idx]) ^ m_tb_C[op_idx])
                               ? -1.0 : 1.0;
        for (size_t o : m_tb_out_toggle[op_idx]) toggle_bit(out_slater_determinant, o);
        if (!check_restrictions ||
            state_is_within_restrictions(out_slater_determinant)) {
            emit(out_slater_determinant, m_flat_coeffs[op_idx] * amp * sgn); // map_res in serial
        }
        for (size_t o : m_tb_out_toggle[op_idx]) toggle_bit(out_slater_determinant, o);
    }
    continue;
}
```

New per-term arrays (parallel to `m_onebody_*`): `m_flat_twobody` (uint8),
`m_tb_M`/`m_tb_occ`/`m_tb_vac` (`key_type` masks), `m_tb_C` (uint8), `m_tb_out_toggle`
(small list of orbitals to flip for the output). Need a `mask_empty` helper
(`(state & mask) == 0`, the dual of `mask_occupied`).

### A.5 Scope, expected gain, risks

* **Scope v1:** enable only the all-four-distinct case; route overlaps to the general
  path (still correct). Add overlap cases later if the probe validates them and the
  coverage measurement justifies it.
* **Expected gain:** bounded by the *two-body producing fraction* (~6% of applications on
  a half-filled determinant) × the per-term sign saving. On the current hopping fixture
  (½ two-body) the 1-body kernel gave ~4.6%; two-body all-distinct should be a similar
  few percent — **measure on the real Hamiltonian's term mix before committing**, since a
  density-dominated Coulomb H has few off-diagonal all-distinct two-body terms.
* **Risks:** sign derivation (mitigated by the probe → automatic fallback); the
  classification must be exhaustive (probe failures must fall back, never silently
  mis-sign). The golden oracle + an independent occupancy/sign cross-check gate it, as in
  2b/2c.

### A.6 Test plan

1. `test_apply_perf.py`: the existing golden already contains two-body terms — masked
   results must match within tolerance.
2. Add a dedicated fixture of **off-diagonal all-distinct two-body terms** (pair hops /
   exchange) and an independent occupancy+sign oracle (compute the sign in Python by the
   between-count formula, no `create`/`annihilate`).
3. A/B microbench (kernel on vs off) on that fixture and on the realistic H.
4. Serial + `IMPURITYMODEL_PARALLEL=1` + MPI parity.

---

## B. NBX sparse count exchange (Dynamic Sparse Data Exchange)

### B.1 What it replaces and why

`graph_alltoall_psis` today does, every redistribute:

```
comm.Alltoall(send_counts, recv_counts)          # dense, O(P) per rank, synchronizing
graph_comm = _cached_dist_graph(comm, src, dst)  # cached (Phase 4b)
graph_comm.Neighbor_alltoallv(... one BYTE msg ...)
```

The `Alltoall` of counts is O(P) memory and a full collective **regardless of sparsity** —
the part that does not scale to 100s–1000s of ranks when each rank only really talks to a
few peers. NBX (Hoefler et al. 2010, "Scalable Communication Protocols for Dynamic Sparse
Data Exchange") discovers each rank's *sources* and moves the data in **O(degree)**
messages, with **no dense collective and no pre-exchanged counts** — the message carries
its own size (`Get_count` on probe).

Because NBX needs neither the count `Alltoall` nor the `dist_graph`, the cleaner design
sends **the data directly** (DSDE), subsuming Phase 4a's single buffer and making 4b's
graph-comm caching unnecessary.

### B.2 Algorithm (deadlock-free by construction)

```
post MPI_Issend(payload_to[d], d)  for each destination d with data   # synchronous: completes when matched
barrier_active = False
done = False
while not done:
    while MPI_Iprobe(ANY_SOURCE, tag) -> (have_msg, status):
        n = status.Get_count(BYTE); src = status.Get_source()
        buf = empty(n); MPI_Recv(buf, src, tag); stash[src] = buf
    if not barrier_active:
        if MPI_Testall(my_issends):          # all my sends were received
            MPI_Ibarrier(); barrier_active = True
    else:
        if MPI_Test(ibarrier_request):       # everyone is done sending & all msgs drained
            done = True
```

The `Issend`→`Testall`→`Ibarrier` chain is the consensus: the non-blocking barrier can
only complete once every rank has finished sending **and** every message has been matched
by a receive, so the final probe drain is exhaustive. No counts, no graph, no deadlock.

### B.3 mpi4py implementation sketch (replacing steps 2–6 of `graph_alltoall_psis`)

```python
TAG = 0xD5DE

def _nbx_exchange(comm, send_bufs):
    """send_bufs: dict {dest_rank: np.uint8 buffer}. Returns {src_rank: np.uint8 buffer}."""
    me = comm.rank
    recv = {}
    # handle self-send locally (hash%size == me); never Issend to self
    if me in send_bufs:
        recv[me] = send_bufs.pop(me)
    reqs = [comm.Issend([buf, MPI.BYTE], dest=d, tag=TAG) for d, buf in send_bufs.items()]

    status = MPI.Status()
    barrier_req = None
    done = False
    while not done:
        while comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAG, status=status):
            n = status.Get_count(MPI.BYTE); src = status.Get_source()
            buf = np.empty(n, dtype=np.uint8)
            comm.Recv([buf, MPI.BYTE], source=src, tag=TAG)
            recv[src] = buf
        if barrier_req is None:
            if MPI.Request.Testall(reqs):
                barrier_req = comm.Ibarrier()
        elif barrier_req.Test():
            done = True
    return recv
```

Integration: pack per-destination slices (reuse `pack_psis_fused` but keep the rank
boundaries from `send_counts` rather than concatenating), build `send_bufs`, call
`_nbx_exchange`, then `unpack_psis_fused_cy` over each received buffer. `recv_counts` /
`Alltoall` / `_cached_dist_graph` / `Neighbor_alltoallv` all disappear from this path.

### B.4 The adaptive caveat (important — NBX is not always a win)

NBX wins when the pattern is **sparse** (degree ≪ P). For the **dense** regime
(`n_out/rank ≥ P`, which hash distribution produces at moderate P) every rank talks to
~all ranks, so NBX is O(P) point-to-point messages and the optimized
`Neighbor_alltoallv` collective likely beats it. Therefore:

* Keep **both** paths and pick by neighbour degree: if
  `#destinations < α·P` (α ~ 0.3, tunable) use NBX; else the dist_graph collective.
  Degree is known locally after packing (`send_counts > 0`); agree on the choice with one
  cheap `Allreduce(max degree)` or just decide per-rank (both paths are matched-safe only
  if all ranks pick the same — so reduce the degree decision).
* NBX's real payoff is **paired with basis locality** (sparse pattern by construction) or
  at extreme P. Standalone at moderate P it mainly removes the O(P) count `Alltoall`.

### B.5 Validation & risks

* **Correctness** is testable at n=3–6: NBX output must be identical to the current path
  (same per-rank ownership, same summed amplitudes). The existing `test_mpi_comm`
  round-trip / ownership / ring tests are the gate; add an `n`-rank all-to-all-dense and a
  sparse (each rank → 1 neighbour) case.
* **Scaling** benefit is **only observable at 100s–1000s of ranks** — must be benchmarked
  on a real allocation, not inferred from n≤6.
* **Risks:** buffer lifetime (Issend buffers must outlive the request — keep refs until
  `Testall`); MPI progress (must keep calling `Iprobe`); tag isolation; the adaptive
  switch must be a *collective* decision so all ranks use the same protocol (else
  Issend/Neighbor mismatch). NBX itself is deadlock-free; the danger is the
  collective-consistency of the path choice — guard it like the Phase-4b rebuild.

### B.6 Recommended sequencing

1. Land NBX behind an env/arg flag, **off by default**, with the adaptive switch.
2. Correctness-gate at n=3–6 against the existing path.
3. Benchmark at real scale; only then consider making it (or the adaptive policy) default.
4. Revisit together with any **basis-locality** work — that is what makes the pattern
   genuinely sparse and unlocks NBX's asymptotic advantage.
