#include "MpiUtils.h"
#include <cstring>
#include <numeric>
#include <unordered_set>

namespace mpi_utils {

namespace {
// Per-entry byte layout shared by pack_psis_fused / unpack_psis_fused.
inline size_t fused_bytes_per_entry(size_t chunks_per_state) {
  return chunks_per_state * sizeof(uint64_t) + 2 * sizeof(double) +
         sizeof(int32_t);
}
} // namespace

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

void pack_psis(const std::vector<const ManyBodyState *> &psis, int comm_size,
               std::vector<int64_t> &send_counts,
               std::vector<uint64_t> &state_buf,
               std::vector<double> &amp_buf_reim,
               std::vector<int32_t> &psi_buf) {

  send_counts.assign(comm_size, 0);

  size_t chunks_per_state = 0;

  struct Entry {
    const SlaterDeterminant<uint64_t> *state;
    std::complex<double> amp;
    int32_t psi_idx;
  };

  std::vector<std::vector<Entry>> rank_entries(comm_size);

  for (size_t pi = 0; pi < psis.size(); pi++) {
    const auto *psi = psis[pi];
    if (!psi)
      continue;
    for (auto it = psi->begin(); it != psi->end(); ++it) {
      const auto &state = it->first;
      if (chunks_per_state == 0)
        chunks_per_state = state.size();
      int rank = state.routing_hash() % comm_size;
      rank_entries[rank].push_back(
          {&state, it->second, static_cast<int32_t>(pi)});
      send_counts[rank]++;
    }
  }

  size_t total_entries = 0;
  for (int64_t c : send_counts)
    total_entries += c;

  state_buf.reserve(total_entries * chunks_per_state);
  state_buf.clear();
  amp_buf_reim.reserve(total_entries * 2);
  amp_buf_reim.clear();
  psi_buf.reserve(total_entries);
  psi_buf.clear();

  for (int r = 0; r < comm_size; r++) {
    for (const auto &entry : rank_entries[r]) {
      state_buf.insert(state_buf.end(), entry.state->begin(),
                       entry.state->end());
      amp_buf_reim.push_back(entry.amp.real());
      amp_buf_reim.push_back(entry.amp.imag());
      psi_buf.push_back(entry.psi_idx);
    }
  }
}

void unpack_psis(std::vector<ManyBodyState *> &psis, int comm_size,
                 const std::vector<int64_t> &recv_counts,
                 const std::vector<uint64_t> &state_buf,
                 const std::vector<double> &amp_buf_reim,
                 const std::vector<int32_t> &psi_buf, size_t chunks_per_state) {

  size_t offset = 0;
  for (int r = 0; r < comm_size; r++) {
    int64_t count = recv_counts[r];
    for (int64_t i = 0; i < count; i++) {
      SlaterDeterminant<uint64_t> det;
      det.reserve(chunks_per_state);
      for (size_t c = 0; c < chunks_per_state; c++) {
        det.push_back(state_buf[offset * chunks_per_state + c]);
      }

      int32_t pi = psi_buf[offset];
      std::complex<double> amp(amp_buf_reim[2 * offset],
                               amp_buf_reim[2 * offset + 1]);

      if (pi >= 0 && pi < static_cast<int32_t>(psis.size()) && psis[pi]) {
        auto [it, inserted] = psis[pi]->try_emplace(det, amp);
        if (!inserted) {
          it->second += amp;
        }
      }
      offset++;
    }
  }
}

void pack_psis_fused(const std::vector<const ManyBodyState *> &psis,
                     int comm_size, size_t chunks_per_state,
                     std::vector<int64_t> &send_counts,
                     std::vector<char> &send_buf) {

  send_counts.assign(comm_size, 0);

  struct Entry {
    const SlaterDeterminant<uint64_t> *state;
    std::complex<double> amp;
    int32_t psi_idx;
  };
  std::vector<std::vector<Entry>> rank_entries(comm_size);

  for (size_t pi = 0; pi < psis.size(); pi++) {
    const auto *psi = psis[pi];
    if (!psi)
      continue;
    for (auto it = psi->begin(); it != psi->end(); ++it) {
      const auto &state = it->first;
      int rank = state.routing_hash() % comm_size;
      rank_entries[rank].push_back(
          {&state, it->second, static_cast<int32_t>(pi)});
      send_counts[rank]++;
    }
  }

  size_t total_entries = 0;
  for (int64_t c : send_counts)
    total_entries += c;

  const size_t bpe = fused_bytes_per_entry(chunks_per_state);
  const size_t state_bytes = chunks_per_state * sizeof(uint64_t);
  send_buf.clear();
  send_buf.resize(total_entries * bpe);

  char *p = send_buf.data();
  for (int r = 0; r < comm_size; r++) {
    for (const auto &e : rank_entries[r]) {
      std::memcpy(p, e.state->data(), state_bytes);
      p += state_bytes;
      const double re = e.amp.real();
      const double im = e.amp.imag();
      std::memcpy(p, &re, sizeof(double));
      p += sizeof(double);
      std::memcpy(p, &im, sizeof(double));
      p += sizeof(double);
      std::memcpy(p, &e.psi_idx, sizeof(int32_t));
      p += sizeof(int32_t);
    }
  }
}

void unpack_psis_fused(std::vector<ManyBodyState *> &psis, int comm_size,
                       const std::vector<int64_t> &recv_counts,
                       const std::vector<char> &recv_buf,
                       size_t chunks_per_state) {

  const size_t state_bytes = chunks_per_state * sizeof(uint64_t);
  const char *p = recv_buf.data();
  for (int r = 0; r < comm_size; r++) {
    const int64_t count = recv_counts[r];
    for (int64_t i = 0; i < count; i++) {
      SlaterDeterminant<uint64_t> det;
      det.resize(chunks_per_state);
      std::memcpy(det.data(), p, state_bytes);
      p += state_bytes;
      double re;
      double im;
      std::memcpy(&re, p, sizeof(double));
      p += sizeof(double);
      std::memcpy(&im, p, sizeof(double));
      p += sizeof(double);
      int32_t pi;
      std::memcpy(&pi, p, sizeof(int32_t));
      p += sizeof(int32_t);

      const std::complex<double> amp(re, im);
      if (pi >= 0 && pi < static_cast<int32_t>(psis.size()) && psis[pi]) {
        auto [it, inserted] = psis[pi]->try_emplace(det, amp);
        if (!inserted) {
          it->second += amp;
        }
      }
    }
  }
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
    std::memcpy(dst + state_bytes, block.row(r), amp_bytes);
  }
}

ManyBodyBlockState unpack_block_fused(int comm_size, size_t width,
                                      const std::vector<int64_t> &recv_counts,
                                      const char *recv_buf,
                                      size_t chunks_per_state) {

  const size_t state_bytes = chunks_per_state * sizeof(uint64_t);
  const size_t amp_bytes = width * sizeof(ManyBodyBlockState::Value);
  size_t total = 0;
  for (int r = 0; r < comm_size; ++r) {
    total += static_cast<size_t>(recv_counts[r]);
  }

  std::vector<ManyBodyBlockState::Key> keys(total);
  std::vector<ManyBodyBlockState::Value> amps(total * width);
  const char *src = recv_buf;
  for (size_t e = 0; e < total; ++e) {
    keys[e].resize(chunks_per_state);
    std::memcpy(keys[e].data(), src, state_bytes);
    src += state_bytes;
    std::memcpy(amps.data() + e * width, src, amp_bytes);
    src += amp_bytes;
  }

  // Stable sort keeps duplicates (the same determinant from several source
  // ranks) in arrival order, so the left-to-right summation below reproduces
  // the scalar unpack's insert-then-accumulate order bit-for-bit per column.
  std::vector<size_t> idx(total);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(),
                   [&keys](size_t a, size_t b) { return keys[a] < keys[b]; });

  std::vector<ManyBodyBlockState::Key> out_keys;
  out_keys.reserve(total);
  std::vector<ManyBodyBlockState::Value> out_amps;
  out_amps.reserve(total * width);
  for (size_t t = 0; t < total; ++t) {
    const size_t e = idx[t];
    if (!out_keys.empty() && out_keys.back() == keys[e]) {
      ManyBodyBlockState::Value *dst =
          out_amps.data() + (out_keys.size() - 1) * width;
      const ManyBodyBlockState::Value *add = amps.data() + e * width;
      for (size_t c = 0; c < width; ++c) {
        dst[c] += add[c];
      }
    } else {
      out_keys.push_back(std::move(keys[e]));
      out_amps.insert(out_amps.end(), amps.begin() + e * width,
                      amps.begin() + (e + 1) * width);
    }
  }
  return ManyBodyBlockState(std::move(out_keys), std::move(out_amps), width);
}

} // namespace mpi_utils
