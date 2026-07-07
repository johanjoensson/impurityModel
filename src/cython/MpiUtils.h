#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include "ManyBodyBlockState.h"
#include "ManyBodyState.h"
#include "SlaterDeterminant.h"
#include <vector>
#include <complex>
#include <cstdint>

namespace mpi_utils {

// Partition and pack a vector of SlaterDeterminants
void pack_determinants(
    const std::vector<SlaterDeterminant<uint64_t>>& dets,
    int comm_size,
    std::vector<int64_t>& send_counts,
    std::vector<uint64_t>& state_buf);

// Unpack into a vector of vectors (one per rank) of SlaterDeterminants
std::vector<std::vector<SlaterDeterminant<uint64_t>>> unpack_determinants(
    int comm_size,
    const std::vector<int64_t>& recv_counts,
    const std::vector<uint64_t>& state_buf,
    size_t chunks_per_state);

// Partition and pack multiple ManyBodyStates
void pack_psis(
    const std::vector<const ManyBodyState*>& psis,
    int comm_size,
    std::vector<int64_t>& send_counts,
    std::vector<uint64_t>& state_buf,
    std::vector<double>& amp_buf,
    std::vector<int32_t>& psi_buf);

// Unpack received buffers into existing ManyBodyStates
void unpack_psis(
    std::vector<ManyBodyState*>& psis,
    int comm_size,
    const std::vector<int64_t>& recv_counts,
    const std::vector<uint64_t>& state_buf,
    const std::vector<double>& amp_buf,
    const std::vector<int32_t>& psi_buf,
    size_t chunks_per_state);

// Fused single-buffer variants: each entry is serialized as
// [state: chunks_per_state x uint64][amp: 2 x double][psi_idx: int32] so the whole
// redistribute moves in ONE Neighbor_alltoallv(MPI_BYTE) instead of three. At 100s-1000s
// of ranks the dense personalized exchange is latency-bound, so collapsing 3 message
// rounds into 1 cuts the dominant cost ~3x. send_buf is rank-ordered; bytes_per_entry =
// chunks_per_state*8 + 20.
void pack_psis_fused(
    const std::vector<const ManyBodyState*>& psis,
    int comm_size,
    size_t chunks_per_state,
    std::vector<int64_t>& send_counts,
    std::vector<char>& send_buf);

void unpack_psis_fused(
    std::vector<ManyBodyState*>& psis,
    int comm_size,
    const std::vector<int64_t>& recv_counts,
    const std::vector<char>& recv_buf,
    size_t chunks_per_state);

// Block-state variants: one entry per shared-support ROW, serialized as
// [state: chunks_per_state x uint64][p x complex amp] — no per-entry psi index
// (the column position within the row identifies the vector), so the wire cost
// per determinant is state_bytes + 16*p instead of p * (state_bytes + 20).
// Ownership uses the same routing_hash() % comm_size as the scalar path.
// Two-phase pack (Phase 4): count computes the per-rank row counts and caches the
// owners; fill serializes straight into the caller-provided buffer (the numpy array
// handed to MPI), eliminating the intermediate std::vector wire buffer and its full
// copy-out per redistribute.
void pack_block_count(
    const ManyBodyBlockState& block,
    int comm_size,
    std::vector<int64_t>& send_counts,
    std::vector<int>& owners);

void pack_block_fill(
    const ManyBodyBlockState& block,
    int comm_size,
    size_t chunks_per_state,
    const std::vector<int64_t>& send_counts,
    const std::vector<int>& owners,
    char* send_buf);

// Rebuild a block from the received buffer. Rows for the same determinant
// arriving from different ranks are summed; the summation order per column is
// the arrival order (stable sort), matching unpack_psis_fused's
// insert-then-accumulate semantics bit-for-bit.
ManyBodyBlockState unpack_block_fused(
    int comm_size,
    size_t width,
    const std::vector<int64_t>& recv_counts,
    const char* recv_buf,
    size_t chunks_per_state);

} // namespace mpi_utils
#endif
