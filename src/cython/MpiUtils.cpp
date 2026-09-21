#include "MpiUtils.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <unordered_set>

namespace mpi_utils {

void pack_determinants(const std::vector<SlaterDeterminant<uint64_t>> &dets,
                       int comm_size, std::vector<int64_t> &send_counts,
                       std::vector<uint64_t> &state_buf) {

  send_counts.assign(comm_size, 0);
  if (dets.empty())
    return;

  size_t chunks_per_state = dets[0].size();

  // Group determinants by destination rank
  std::vector<std::vector<const SlaterDeterminant<uint64_t> *>> rank_dets(
      comm_size);
  // Determine unique states and their target ranks
  std::unordered_set<SlaterDeterminant<uint64_t>> unique_dets(dets.begin(),
                                                              dets.end());
  for (const auto &det : unique_dets) {
    int rank = det.routing_hash() % comm_size;
    rank_dets[rank].push_back(&det);
    send_counts[rank]++;
  }

  // Pack into flat buffer
  size_t total_states = unique_dets.size();
  state_buf.reserve(total_states * chunks_per_state);
  state_buf.clear();

  for (int r = 0; r < comm_size; r++) {
    for (const auto *det : rank_dets[r]) {
      state_buf.insert(state_buf.end(), det->begin(), det->end());
    }
  }
}

std::vector<std::vector<SlaterDeterminant<uint64_t>>>
unpack_determinants(int comm_size, const std::vector<int64_t> &recv_counts,
                    const std::vector<uint64_t> &state_buf,
                    size_t chunks_per_state) {

  std::vector<std::vector<SlaterDeterminant<uint64_t>>> result(comm_size);
  size_t offset = 0;

  for (int r = 0; r < comm_size; r++) {
    int64_t count = recv_counts[r];
    result[r].reserve(count);
    for (int64_t i = 0; i < count; i++) {
      SlaterDeterminant<uint64_t> det;
      det.reserve(chunks_per_state);
      for (size_t c = 0; c < chunks_per_state; c++) {
        det.push_back(state_buf[offset++]);
      }
      result[r].push_back(det);
    }
  }
  return result;
}

void pack_block_count(const ManyBodyBlockState &block, int comm_size,
                      std::vector<int64_t> &send_counts,
                      std::vector<int> &owners) {

  send_counts.assign(comm_size, 0);
  owners.resize(block.rows());
  for (size_t r = 0; r < block.rows(); ++r) {
    owners[r] = static_cast<int>(block.key(r).routing_hash() % comm_size);
    send_counts[owners[r]]++;
  }
}

void pack_block_fill(const ManyBodyBlockState &block, int comm_size,
                     size_t chunks_per_state,
                     const std::vector<int64_t> &send_counts,
                     const std::vector<int> &owners, char *send_buf) {

  const size_t p = block.width();
  const size_t state_bytes = chunks_per_state * sizeof(uint64_t);
  const size_t amp_bytes = p * sizeof(ManyBodyBlockState::Value);
  const size_t bpe = state_bytes + amp_bytes;

  // Rank-ordered entry offsets, then a single fill pass (rows keep their block
  // order within each destination rank, like the scalar packer's per-rank
  // lists).
  std::vector<size_t> next(comm_size, 0);
  size_t total = 0;
  for (int rk = 0; rk < comm_size; ++rk) {
    next[rk] = total;
    total += static_cast<size_t>(send_counts[rk]);
  }
  for (size_t r = 0; r < block.rows(); ++r) {
    char *dst = send_buf + (next[owners[r]]++) * bpe;
    std::memcpy(dst, block.key(r).data(), state_bytes);
    std::memcpy(dst + state_bytes, block.row(r).data(), amp_bytes);
  }
}

ManyBodyBlockState unpack_block_fused(int comm_size, size_t width,
                                      const std::vector<int64_t> &recv_counts,
                                      const char *recv_buf,
                                      size_t chunks_per_state) {

  const size_t state_bytes = chunks_per_state * sizeof(uint64_t);
  const size_t amp_bytes = width * sizeof(ManyBodyBlockState::Value);
  const size_t bpe = state_bytes + amp_bytes;
  size_t total = 0;
  for (int r = 0; r < comm_size; ++r) {
    total += static_cast<size_t>(recv_counts[r]);
  }

  // Only the KEYS come out of the receive buffer, and into one flat array rather than a
  // vector-of-vectors: `total * chunks_per_state * 8` B against the `total * width * 16` B
  // the amplitudes used to cost, and no per-key heap allocation. The amplitudes stay in
  // `recv_buf` -- which MPI keeps alive for the whole call -- and are read once, at emit
  // time, so the coefficients are never resident twice. Measured at total=60k, width=320:
  // 517.4 MiB peak for a 220.4 MiB result before, because `amps` alone was 293 MiB.
  const size_t cps = chunks_per_state;
  std::vector<uint64_t> flat_keys(total * cps);
  for (size_t e = 0; e < total; ++e) {
    std::memcpy(flat_keys.data() + e * cps, recv_buf + e * bpe, state_bytes);
  }
  const uint64_t *fk = flat_keys.data();

  // Stable sort keeps duplicates (the same determinant from several source
  // ranks) in arrival order, so the left-to-right summation below reproduces
  // the scalar unpack's insert-then-accumulate order bit-for-bit per column.
  // The comparison is element-wise over uint64 because that is what
  // `std::vector<uint64_t>::operator<` does; a byte compare would disagree with
  // it on a little-endian machine, and the key order is what every rank's
  // `offset`-based global index arithmetic depends on.
  std::vector<size_t> idx(total);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [fk, cps](size_t a, size_t b) {
    return std::lexicographical_compare(fk + a * cps, fk + (a + 1) * cps, fk + b * cps,
                                        fk + (b + 1) * cps);
  });

  // Count the distinct determinants before reserving. `total` over-reserves by the
  // duplicate fraction, and `std::vector` never hands that capacity back -- the result
  // is long-lived, so the slack would be retained rather than transient.
  size_t n_unique = 0;
  for (size_t t = 0; t < total; ++t) {
    if (t == 0 || !std::equal(fk + idx[t] * cps, fk + (idx[t] + 1) * cps, fk + idx[t - 1] * cps)) {
      ++n_unique;
    }
  }

  std::vector<ManyBodyBlockState::Key> out_keys;
  out_keys.reserve(n_unique);
  std::vector<ManyBodyBlockState::Value> out_amps;
  out_amps.reserve(n_unique * width);
  // One row's worth of scratch, so the duplicate branch reads `recv_buf` through a
  // `memcpy` rather than a reinterpret_cast of `char*` to `std::complex<double>*`.
  std::vector<ManyBodyBlockState::Value> scratch(width);
  for (size_t t = 0; t < total; ++t) {
    const size_t e = idx[t];
    const char *src_amp = recv_buf + e * bpe + state_bytes;
    if (!out_keys.empty() && std::equal(fk + e * cps, fk + (e + 1) * cps, out_keys.back().data())) {
      ManyBodyBlockState::Value *dst = out_amps.data() + (out_keys.size() - 1) * width;
      if (amp_bytes > 0) {
        std::memcpy(scratch.data(), src_amp, amp_bytes);
      }
      for (size_t c = 0; c < width; ++c) {
        dst[c] += scratch[c];
      }
    } else {
      // `SlaterDeterminant` inherits `std::vector` without `using vector::vector`, so it
      // has only the default constructor -- no iterator-range form.
      out_keys.emplace_back();
      out_keys.back().assign(fk + e * cps, fk + (e + 1) * cps);
      const size_t base = out_amps.size();
      out_amps.resize(base + width);
      if (amp_bytes > 0) {
        std::memcpy(out_amps.data() + base, src_amp, amp_bytes);
      }
    }
  }
  return ManyBodyBlockState(std::move(out_keys), std::move(out_amps), width);
}

} // namespace mpi_utils
