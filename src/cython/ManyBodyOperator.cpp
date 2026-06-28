#include <boost/unordered/unordered_flat_map.hpp>
#include "ManyBodyOperator.h"
#include "ManyBodyState.h"
#include <algorithm>
#include <cassert>
#include <numeric>
#if __cplusplus >= 202002L
#include <bit>
#else
#include <bitset>
#include <climits>
#endif
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <thread>

int set_bits(ManyBodyState::key_type::value_type byte) noexcept {
#if __cplusplus >= 202002L
  return std::popcount<ManyBodyState::key_type::value_type>(byte);
#else
  return std::bitset<sizeof(ManyBodyState::key_type::value_type) * CHAR_BIT>(
             byte)
      .count();
#endif
}

[[nodiscard]] int create(ManyBodyState::key_type &state,
                         size_t idx) /*noexcept*/ {
  const size_t num_bits{8 * sizeof(ManyBodyState::key_type::value_type)};
  const size_t state_idx = idx / num_bits;
  if (state_idx >= state.size()) {
    return 0;
  }
  const size_t bit_idx = num_bits - 1 - (idx % num_bits);
  const ManyBodyState::key_type::value_type mask =
      static_cast<ManyBodyState::key_type::value_type>(1) << bit_idx;
  if (state[state_idx] & mask) {
    return 0;
  }
  size_t n_set = 0;
  for (size_t i = 0; i < state_idx; i++) {
    n_set += set_bits(state[i]);
  }
  n_set += set_bits(state[state_idx] >> bit_idx);
  state[state_idx] ^= mask;
  return n_set % 2 ? -1 : 1;
}

[[nodiscard]] int annihilate(ManyBodyState::key_type &state,
                             size_t idx) /*noexcept*/ {
  const size_t num_bits = 8 * sizeof(ManyBodyState::key_type::value_type);
  const size_t state_idx = idx / num_bits;
  if (state_idx >= state.size()) {
    return 0;
  }
  const size_t bit_idx = num_bits - 1 - (idx % num_bits);
  const ManyBodyState::key_type::value_type mask =
      static_cast<ManyBodyState::key_type::value_type>(1) << bit_idx;
  if (!(state[state_idx] & mask)) {
    return 0;
  }
  int n_set = 0;
  for (size_t i = 0; i < state_idx; i++) {
    n_set += set_bits(state[i]);
  }
  state[state_idx] ^= mask;
  n_set += set_bits(state[state_idx] >> bit_idx);
  return n_set % 2 ? -1 : 1;
}

// Undo helper for the in-place apply (Phase 1a): toggle the occupation bit of orbital
// `idx`. Only ever called on orbitals whose create/annihilate already succeeded, so
// `state_idx` is in range. Bit-toggle is its own inverse and order-independent, so
// re-toggling every successfully-applied operator restores the determinant to its
// input value without copying it again per term.
static inline void toggle_bit(ManyBodyState::key_type &state,
                              size_t idx) noexcept {
  const size_t num_bits = 8 * sizeof(ManyBodyState::key_type::value_type);
  const size_t state_idx = idx / num_bits;
  const size_t bit_idx = num_bits - 1 - (idx % num_bits);
  state[state_idx] ^=
      (static_cast<ManyBodyState::key_type::value_type>(1) << bit_idx);
}

// Phase 2b fast-path test: are all orbitals in `mask` occupied in `state`? Chunks of
// `mask` beyond `state.size()` would require occupied orbitals the determinant cannot
// hold, so they fail. Zero mask chunks (and the empty mask of a constant term) pass.
static inline bool mask_occupied(const ManyBodyState::key_type &state,
                                 const ManyBodyState::key_type &mask) noexcept {
  for (size_t j = 0; j < mask.size(); j++) {
    if (mask[j] == 0) {
      continue;
    }
    if (j >= state.size() || (state[j] & mask[j]) != mask[j]) {
      return false;
    }
  }
  return true;
}

// Is orbital `idx` occupied in `state`? (chunks beyond the determinant are empty.)
static inline bool bit_set(const ManyBodyState::key_type &state,
                           size_t idx) noexcept {
  const size_t num_bits = 8 * sizeof(ManyBodyState::key_type::value_type);
  const size_t state_idx = idx / num_bits;
  if (state_idx >= state.size()) {
    return false;
  }
  const size_t bit_idx = num_bits - 1 - (idx % num_bits);
  return (state[state_idx] >> bit_idx) & 1U;
}

// Parity (0/1) of the number of occupied orbitals selected by `mask` -- the one-body
// fermion sign exponent. Loops over the actual key_size chunks.
static inline int mask_parity(const ManyBodyState::key_type &state,
                              const ManyBodyState::key_type &mask) noexcept {
  int pc = 0;
  const size_t lim = std::min(state.size(), mask.size());
  for (size_t j = 0; j < lim; j++) {
    pc += set_bits(state[j] & mask[j]);
  }
  return pc & 1;
}

static void initialize_from_ops(std::vector<ManyBodyOperator::value_type> &m_ops) {
  std::sort(m_ops.begin(), m_ops.end(),
            [](const ManyBodyOperator::value_type &a,
               const ManyBodyOperator::value_type &b) {
              return a.first < b.first;
            });
  if (!m_ops.empty()) {
    size_t merge_at_idx = 0;
    for (size_t read_from_idx = 1; read_from_idx < m_ops.size(); read_from_idx++) {
      if (m_ops[merge_at_idx].first == m_ops[read_from_idx].first) {
        m_ops[merge_at_idx].second += m_ops[read_from_idx].second;
      } else {
        merge_at_idx++;
        if (merge_at_idx != read_from_idx) {
          m_ops[merge_at_idx] = std::move(m_ops[read_from_idx]);
        }
      }
    }
    m_ops.resize(merge_at_idx + 1);
  }
}

ManyBodyOperator::ManyBodyOperator(const std::vector<value_type> &ops)
    : m_ops(ops) {
  initialize_from_ops(m_ops);
}

bool ManyBodyOperator::empty() const noexcept {
    return m_ops.empty();
}

bool ManyBodyOperator::clear() {
    m_ops.clear();
    m_flat_dirty = true;
    return true;
}

ManyBodyOperator::ManyBodyOperator(std::vector<value_type> &&ops)
    : m_ops(std::move(ops)) {
  initialize_from_ops(m_ops);
}

ManyBodyOperator::ManyBodyOperator(const OPS_VEC &ops, const SCALAR_VEC &amps)
    : m_ops() {
  m_ops.reserve(ops.size());
  for (size_t i = 0; i < ops.size(); i++) {
    m_ops.emplace_back(ops[i], amps[i]);
  }
  initialize_from_ops(m_ops);
}
ManyBodyOperator::ManyBodyOperator(OPS_VEC &&ops, SCALAR_VEC &&amps) : m_ops() {
  m_ops.reserve(ops.size());
  for (size_t i = 0; i < ops.size(); i++) {
    m_ops.emplace_back(std::move(ops[i]), std::move(amps[i]));
  }
  initialize_from_ops(m_ops);
}

ManyBodyOperator::iterator ManyBodyOperator::find(iterator first, iterator last,
                                                  const key_type &key) {
  return std::lower_bound(
      first, last, key,
      [](const value_type &a, const key_type &b) { return a.first < b; });
}

ManyBodyOperator::iterator ManyBodyOperator::find(const key_type &key) {
  return find(m_ops.begin(), m_ops.end(), key);
}

ManyBodyOperator::const_iterator
ManyBodyOperator::find(const_iterator first, const_iterator last,
                       const key_type &key) const {
  return std::lower_bound(
      first, last, key,
      [](const value_type &a, const key_type &b) { return a.first < b; });
}

ManyBodyOperator::const_iterator
ManyBodyOperator::find(const key_type &key) const {
  return find(m_ops.cbegin(), m_ops.cend(), key);
}

template <class K>
ManyBodyOperator::iterator ManyBodyOperator::find(const K &key) {
  return find(m_ops.begin(), m_ops.end(), static_cast<key_type>(key));
}

template <class K>
ManyBodyOperator::const_iterator ManyBodyOperator::find(const K &key) const {
  return find(m_ops.cbegin(), m_ops.cend(), static_cast<key_type>(key));
}

ManyBodyOperator::mapped_type &
ManyBodyOperator::operator[](const key_type &key) {
  auto it = find(key);
  if (it == m_ops.end() || it->first != key) {
    it = m_ops.emplace(it, key, mapped_type{});
  }
  return it->second;
}

ManyBodyOperator::mapped_type &ManyBodyOperator::operator[](key_type &&key) {
  auto it = find(key);
  if (it == m_ops.end() || it->first != key) {
    it = m_ops.emplace(it, std::move(key), mapped_type{});
  }
  return it->second;
}

ManyBodyOperator::mapped_type &ManyBodyOperator::at(const key_type &key) {
  auto it = find(key);
  if (it == m_ops.end() || it->first != key) {
    throw std::out_of_range("Element not found!");
  }
  return it->second;
}

const ManyBodyOperator::mapped_type &
ManyBodyOperator::at(const key_type &key) const {
  const auto it = find(key);
  if (it == m_ops.end() || it->first != key) {
    throw std::out_of_range("Element not found!");
  }
  return it->second;
}

std::pair<ManyBodyOperator::iterator, bool>
ManyBodyOperator::insert(const value_type &val) {
  auto it = find(val.first);
  if (it != m_ops.end() && it->first == val.first) {
    return {it, false};
  }
  it = m_ops.emplace(it, val);
  m_flat_dirty = true;
  return {it, true};
}
std::pair<ManyBodyOperator::iterator, bool>
ManyBodyOperator::insert(value_type &&val) {
  auto it = find(val.first);
  if (it != m_ops.end() && it->first == val.first) {
    return {it, false};
  }
  it = m_ops.emplace(it, std::move(val));
  m_flat_dirty = true;
  return {it, true};
}
ManyBodyOperator::iterator ManyBodyOperator::insert(iterator pos,
                                                    const value_type &val) {
  auto it = find(m_ops.begin(), pos, val.first);
  m_flat_dirty = true;
  return m_ops.emplace(it, val);
}

ManyBodyOperator::iterator ManyBodyOperator::insert(iterator pos,
                                                    value_type &&val) {
  auto it = find(m_ops.begin(), pos, val.first);
  m_flat_dirty = true;
  return m_ops.emplace(it, std::move(val));
}
template <class InputIt>
void ManyBodyOperator::insert(InputIt first, InputIt last) {
  for (auto it = first; it != last; it++) {
    insert(std::move(*it));
  }
}

void ManyBodyOperator::insert(std::initializer_list<value_type> l) {
  for (auto val : l) {
    insert(std::move(val));
  }
}

template <class... Args>
std::pair<ManyBodyOperator::iterator, bool>
ManyBodyOperator::emplace(Args &&...args) {
  value_type val{std::forward<Args>(args)...};
  auto it = find(val.first);
  if (it != m_ops.end() && it->first == val.first) {
    return {it, false};
  }
  it = m_ops.emplace(it, std::move(val));
  m_flat_dirty = true;
  return {it, true};
}

template <class... Args>
ManyBodyOperator::iterator ManyBodyOperator::emplace_hint(const_iterator hint,
                                                          Args &&...args) {
  value_type val{std::forward<Args>(args)...};
  auto it = find(m_ops.begin(), hint, val.first);
  if (it == m_ops.end() || it->first == val.first) {
    return m_ops.begin() + (it - m_ops.begin());
  }
  it = m_ops.emplace(it, std::move(val));
  return m_ops.begin() + (it - m_ops.begin());
}

ManyBodyOperator::iterator ManyBodyOperator::erase(iterator pos) {
  m_flat_dirty = true;
  return m_ops.erase(pos);
}

ManyBodyOperator::iterator ManyBodyOperator::erase(const_iterator pos) {
  m_flat_dirty = true;
  return m_ops.erase(pos);
}
ManyBodyOperator::iterator ManyBodyOperator::erase(const_iterator first,
                                                   const_iterator last) {
  m_flat_dirty = true;
  return m_ops.erase(first, last);
}
ManyBodyOperator::size_type ManyBodyOperator::erase(const key_type &key) {
  auto it = find(key);
  if (it == m_ops.end() || it->first != key) {
    return 0;
  }
  m_ops.erase(it);
  m_flat_dirty = true;
  return 1;
}

ManyBodyOperator::iterator ManyBodyOperator::lower_bound(const key_type &key) {
  return std::lower_bound(
      m_ops.begin(), m_ops.end(), key,
      [](const value_type &a, const key_type &b) { return a.first < b; });
}
ManyBodyOperator::const_iterator
ManyBodyOperator::lower_bound(const key_type &key) const {
  return std::lower_bound(
      m_ops.cbegin(), m_ops.cend(), key,
      [](const value_type &a, const key_type &b) { return a.first < b; });
}
template <class K>
ManyBodyOperator::iterator ManyBodyOperator::lower_bound(const K &key) {
  return std::lower_bound(
      m_ops.begin(), m_ops.end(), key,
      [](const value_type &a, const key_type &b) { return a.first < b; });
}
template <class K>
ManyBodyOperator::const_iterator
ManyBodyOperator::lower_bound(const K &key) const {
  return std::lower_bound(
      m_ops.cbegin(), m_ops.cend(), key,
      [](const value_type &a, const key_type &b) { return a.first < b; });
}

ManyBodyOperator::iterator ManyBodyOperator::upper_bound(const key_type &key) {
  return std::upper_bound(
      m_ops.begin(), m_ops.end(), key,
      [](const key_type &a, const value_type &b) { return a < b.first; });
}
ManyBodyOperator::const_iterator
ManyBodyOperator::upper_bound(const key_type &key) const {
  return std::upper_bound(
      m_ops.cbegin(), m_ops.cend(), key,
      [](const key_type &a, const value_type &b) { return a < b.first; });
}
template <class K>
ManyBodyOperator::iterator ManyBodyOperator::upper_bound(const K &key) {
  return std::upper_bound(
      m_ops.begin(), m_ops.end(), key,
      [](const key_type &a, const value_type &b) { return a < b.first; });
}
template <class K>
ManyBodyOperator::const_iterator
ManyBodyOperator::upper_bound(const K &key) const {
  return std::upper_bound(
      m_ops.cbegin(), m_ops.cend(), key,
      [](const key_type &a, const value_type &b) { return a < b.first; });
}

[[nodiscard]] ManyBodyOperator::size_type
ManyBodyOperator::size() const noexcept {
  return m_ops.size();
}

namespace {
// Build the bitmask (a SlaterDeterminant key) marking the given orbital indices.
ManyBodyState::key_type
build_orbital_mask(const std::vector<size_t> &orbitals) {
  const size_t num_bits = 8 * sizeof(ManyBodyState::key_type::value_type);
  std::vector<size_t> sorted_orbitals(orbitals);
  std::sort(sorted_orbitals.begin(), sorted_orbitals.end());

  ManyBodyState::key_type mask;
  ManyBodyState::key_type::value_type current = 0;
  size_t mask_i = 0;
  for (auto idx : sorted_orbitals) {
    size_t mask_j = idx / num_bits;
    size_t local_idx = (num_bits - 1 - (idx % num_bits));
    while (mask_j > mask_i) {
      mask.push_back(current);
      current = 0;
      mask_i += 1;
    }
    current |=
        (static_cast<ManyBodyState::key_type::value_type>(1) << local_idx);
  }
  if (current != 0) {
    mask.push_back(current);
  }
  return mask;
}
} // namespace

void ManyBodyOperator::build_restriction_mask(
    const Restrictions &restrictions) noexcept {
  std::vector<ManyBodyState::key_type> masks;
  std::vector<size_t> min_vals;
  std::vector<size_t> max_vals;
  masks.reserve(restrictions.size());
  min_vals.reserve(restrictions.size());
  max_vals.reserve(restrictions.size());
  for (const auto &restriction : restrictions) {
    masks.push_back(build_orbital_mask(restriction.first));
    min_vals.push_back(restriction.second.first);
    max_vals.push_back(restriction.second.second);
  }

  this->m_restrictions_mask =
      std::tuple<std::vector<ManyBodyState::key_type>, std::vector<size_t>,
                 std::vector<size_t>>{std::move(masks), std::move(min_vals),
                                      std::move(max_vals)};
}

void ManyBodyOperator::build_weighted_restriction_mask(
    const WeightedRestrictions &restrictions) noexcept {
  m_weighted_restrictions_mask.clear();
  m_weighted_restrictions_mask.reserve(restrictions.size());
  for (const auto &restriction : restrictions) {
    std::vector<std::pair<long, ManyBodyState::key_type>> groups;
    groups.reserve(restriction.first.size());
    for (const auto &[weight, orbitals] : restriction.first) {
      groups.emplace_back(weight, build_orbital_mask(orbitals));
    }
    m_weighted_restrictions_mask.emplace_back(std::move(groups),
                                              restriction.second);
  }
}

bool ManyBodyOperator::state_is_within_restrictions(
    const ManyBodyState::key_type &state) const noexcept {

  const auto &[masks, min_vals, max_vals] = m_restrictions_mask;
  for (size_t i = 0; i < masks.size(); i++) {
    size_t bit_count = 0;
    const size_t limit = std::min(masks[i].size(), state.size());
    for (size_t j = 0; j < limit; j++) {
      bit_count += set_bits(state[j] & masks[i][j]);
    }
    if (bit_count < min_vals[i] || bit_count > max_vals[i]) {
      return false;
    }
  }

  // Weighted-sum restrictions: sum_w  w * (#occupied orbitals in group w).
  for (const auto &[groups, bounds] : m_weighted_restrictions_mask) {
    long weighted_sum = 0;
    for (const auto &[weight, mask] : groups) {
      size_t bit_count = 0;
      const size_t limit = std::min(mask.size(), state.size());
      for (size_t j = 0; j < limit; j++) {
        bit_count += set_bits(state[j] & mask[j]);
      }
      weighted_sum += weight * static_cast<long>(bit_count);
    }
    if (weighted_sum < bounds.first || weighted_sum > bounds.second) {
      return false;
    }
  }
  return true;
}

namespace {
// boost::unordered_flat_map is an open-addressing (flat) hash map: contiguous storage,
// no per-node allocation, far better cache behavior than std::unordered_map for the
// apply accumulator. boost::hash does not pick up the std::hash<SlaterDeterminant>
// specialization in SlaterDeterminant.h, so wrap it explicitly.
struct SlaterKeyHash {
  std::size_t operator()(const ManyBodyState::key_type &k) const noexcept {
    return std::hash<ManyBodyState::key_type>{}(k);
  }
};
using ResultMap =
    boost::unordered_flat_map<ManyBodyState::key_type,
                              ManyBodyState::mapped_type, SlaterKeyHash>;

#if defined(PARALLEL)
// Maximum threads the apply path may use. Prefer SLURM_CPUS_PER_TASK (the cores the
// scheduler granted *this* MPI task) so one rank per task never oversubscribes the node;
// fall back to hardware_concurrency only when unset. Resolved once.
unsigned int apply_thread_cap() {
  static const unsigned int cap = []() -> unsigned int {
    if (const char *s = std::getenv("SLURM_CPUS_PER_TASK")) {
      char *end = nullptr;
      const long v = std::strtol(s, &end, 10);
      if (end != s && v > 0) {
        return static_cast<unsigned int>(v);
      }
    }
    return std::max(1u, std::thread::hardware_concurrency());
  }();
  return cap;
}
#endif
} // namespace

[[nodiscard]] ManyBodyState ManyBodyOperator::apply(const ManyBodyState &state,
                                                    double cutoff) const {
  std::vector<std::pair<ManyBodyState::Key, ManyBodyState::Value>> local_res;
  const double cutoff2 = cutoff * cutoff;

  // Phase 1c: most operators carry no occupation restrictions, so hoist the emptiness
  // test out of the per-output hot path and skip state_is_within_restrictions entirely
  // when there is nothing to check.
  const bool check_restrictions =
      !std::get<0>(m_restrictions_mask).empty() ||
      !m_weighted_restrictions_mask.empty();

#if defined(PARALLEL)
  if (m_flat_dirty) {
    build_flat_representation();
  }
  // Phase 5b: scale the thread count to the workload instead of always grabbing every
  // core. Small states (few SDs) don't amortize thread spawn + the bucket merge, and under
  // MPI every rank would otherwise oversubscribe the node; require >= MIN_SD_PER_THREAD
  // input determinants per thread, so tiny applies run on a single thread.
  constexpr size_t MIN_SD_PER_THREAD = 256;
  const ManyBodyState::size_type num_slater = state.size();
  const unsigned int hw = apply_thread_cap(); // SLURM_CPUS_PER_TASK, else hardware_concurrency
  const unsigned int want = static_cast<unsigned int>(
      std::max<size_t>(1, num_slater / MIN_SD_PER_THREAD));
  const unsigned int num_threads = std::max(1u, std::min(hw, want));
  // Output is partitioned across `num_buckets` by key hash so the merge below runs one
  // lock-free thread per bucket (disjoint key sets), instead of the old serial merge
  // that re-did every insert on one thread and capped the speedup.
  const unsigned int num_buckets = num_threads;
  const SlaterKeyHash hasher;
  // local_buckets[t * num_buckets + b]: outputs from compute-thread t hashing to bucket b.
  std::vector<ResultMap> local_buckets(static_cast<size_t>(num_threads) *
                                       num_buckets);
  std::vector<std::thread> threads;
  size_t chunk_size = (num_slater + num_threads - 1) / num_threads;
  for (unsigned int t = 0; t < num_threads; t++) {
    size_t start_slater = t * chunk_size;
    size_t end_slater = std::min(start_slater + chunk_size, num_slater);
    if (start_slater >= num_slater) {
      break;
    }
    threads.push_back(std::thread([&, t, start_slater, end_slater]() {
      ResultMap *buckets = &local_buckets[static_cast<size_t>(t) * num_buckets];
      auto emit = [&](const ManyBodyState::key_type &k,
                      ManyBodyState::mapped_type v) {
        buckets[hasher(k) % num_buckets][k] += v;
      };
      ManyBodyState::key_type out_slater_determinant;

      ManyBodyState::const_iterator it = state.begin(), end = state.begin();
      std::advance(it, start_slater);
      std::advance(end, end_slater);
      for (; it != end; it++) {
        const auto &[slater, amp] = *it;

        // Copy the input determinant ONCE per SD; each term applies in place and
        // undoes its operators afterwards (Phase 1a), instead of copying per term.
        out_slater_determinant = slater;
        // Phase 2b: accumulate all diagonal terms into one scalar -> single insert.
        ManyBodyState::mapped_type diag_accum{0.0, 0.0};
        for (size_t op_idx = 0; op_idx < m_flat_coeffs.size(); op_idx++) {
          // Phase 2b fast path: pure number-operator product -> one occupancy AND-test
          // and a constant-signed scalar add, no create/annihilate.
          if (m_flat_density[op_idx]) {
            if (mask_occupied(slater, m_density_mask[op_idx]) &&
                (!check_restrictions || state_is_within_restrictions(slater))) {
              diag_accum += m_density_coeff[op_idx] * amp;
            }
            continue;
          }
          // Phase 2c fast path: off-diagonal one-body hop c^d_i c_j -> bit tests + a
          // masked-popcount sign, no create/annihilate.
          if (m_flat_onebody[op_idx]) {
            const size_t ob_i = m_onebody_i[op_idx];
            const size_t ob_j = m_onebody_j[op_idx];
            if (bit_set(slater, ob_j) && !bit_set(slater, ob_i)) {
              const double sgn =
                  mask_parity(slater, m_onebody_between[op_idx]) ? -1.0 : 1.0;
              toggle_bit(out_slater_determinant, ob_j); // remove j
              toggle_bit(out_slater_determinant, ob_i); // add i
              if (!check_restrictions ||
                  state_is_within_restrictions(out_slater_determinant)) {
                emit(out_slater_determinant, m_flat_coeffs[op_idx] * amp * sgn);
              }
              toggle_bit(out_slater_determinant, ob_j); // restore scratch
              toggle_bit(out_slater_determinant, ob_i);
            }
            continue;
          }
          double sign = 1;
          const size_t start_idx = m_flat_offsets[op_idx];
          const size_t end_idx = m_flat_offsets[op_idx + 1];
          const auto coeff = m_flat_coeffs[op_idx];

          size_t i = start_idx;
          for (; i < end_idx; i++) {
            const int64_t idx = m_flat_indices[i];
            const int s =
                idx >= 0
                    ? create(out_slater_determinant, static_cast<size_t>(idx))
                    : annihilate(out_slater_determinant,
                                 static_cast<size_t>(-(idx + 1)));
            if (s == 0) {
              sign = 0;
              break;
            }
            sign *= s;
          }
          if (sign != 0 &&
              (!check_restrictions ||
               state_is_within_restrictions(out_slater_determinant))) {
            const auto contribution = coeff * amp * sign;
            if (m_flat_diagonal[op_idx]) {
              diag_accum += contribution; // out == slater
            } else {
              emit(out_slater_determinant, contribution);
            }
          }
          // Restore out_slater_determinant to `slater`: undo the [start_idx, i)
          // operators that actually toggled a bit.
          for (size_t j = start_idx; j < i; j++) {
            const int64_t idx = m_flat_indices[j];
            toggle_bit(out_slater_determinant,
                       idx >= 0 ? static_cast<size_t>(idx)
                                : static_cast<size_t>(-(idx + 1)));
          }
        }
        if (diag_accum != ManyBodyState::mapped_type(0.0, 0.0)) {
          emit(slater, diag_accum);
        }
      }
    }));
  }
  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  // Phase 5a: parallel merge -- one thread per output bucket. Bucket b owns a disjoint
  // set of keys (all hashing to b), so the threads never touch the same entry and need no
  // locks. Each thread also applies the cutoff while collecting its bucket's survivors.
  std::vector<std::vector<std::pair<ManyBodyState::Key, ManyBodyState::Value>>>
      bucket_vecs(num_buckets);
  std::vector<std::thread> merge_threads;
  for (unsigned int b = 0; b < num_buckets; b++) {
    merge_threads.push_back(std::thread([&, b]() {
      ResultMap acc;
      for (unsigned int t = 0; t < num_threads; t++) {
        for (auto &[k, v] :
             local_buckets[static_cast<size_t>(t) * num_buckets + b]) {
          acc[k] += v;
        }
      }
      auto &outv = bucket_vecs[b];
      outv.reserve(acc.size());
      for (auto &[k, v] : acc) {
        if (std::norm(v) > cutoff2) {
          outv.emplace_back(k, v);
        }
      }
    }));
  }
  for (auto &thread : merge_threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  size_t total_out = 0;
  for (const auto &v : bucket_vecs) {
    total_out += v.size();
  }
  local_res.reserve(total_out);
  for (auto &v : bucket_vecs) {
    for (auto &kv : v) {
      local_res.emplace_back(std::move(kv));
    }
  }

#else
  if (m_flat_dirty) {
    build_flat_representation();
  }
  ResultMap map_res;
  map_res.reserve(state.size());
  ManyBodyState::key_type out_slater_determinant;
  for (const auto &[slater, amp] : state) {
    // Copy the input determinant ONCE per SD; each term applies its operators in
    // place and undoes them afterwards (Phase 1a), instead of copying per term.
    out_slater_determinant = slater;
    // Phase 2b: sum every diagonal term's contribution (all map to `slater`) into one
    // scalar and emit a single insert, instead of one colliding insert per term.
    ManyBodyState::mapped_type diag_accum{0.0, 0.0};
    for (size_t op_idx = 0; op_idx < m_flat_coeffs.size(); op_idx++) {
      // Phase 2b fast path: pure number-operator product -> one occupancy AND-test and a
      // constant-signed scalar add, no create/annihilate.
      if (m_flat_density[op_idx]) {
        if (mask_occupied(slater, m_density_mask[op_idx]) &&
            (!check_restrictions || state_is_within_restrictions(slater))) {
          diag_accum += m_density_coeff[op_idx] * amp;
        }
        continue;
      }
      // Phase 2c fast path: off-diagonal one-body hop c^d_i c_j -> bit tests + a masked-
      // popcount sign, no create/annihilate.
      if (m_flat_onebody[op_idx]) {
        const size_t ob_i = m_onebody_i[op_idx];
        const size_t ob_j = m_onebody_j[op_idx];
        if (bit_set(slater, ob_j) && !bit_set(slater, ob_i)) {
          const double sgn =
              mask_parity(slater, m_onebody_between[op_idx]) ? -1.0 : 1.0;
          toggle_bit(out_slater_determinant, ob_j); // remove j
          toggle_bit(out_slater_determinant, ob_i); // add i
          if (!check_restrictions ||
              state_is_within_restrictions(out_slater_determinant)) {
            map_res[out_slater_determinant] += m_flat_coeffs[op_idx] * amp * sgn;
          }
          toggle_bit(out_slater_determinant, ob_j); // restore scratch
          toggle_bit(out_slater_determinant, ob_i);
        }
        continue;
      }
      double sign = 1;
      const size_t start_idx = m_flat_offsets[op_idx];
      const size_t end_idx = m_flat_offsets[op_idx + 1];
      const auto coeff = m_flat_coeffs[op_idx];

      size_t i = start_idx;
      for (; i < end_idx; i++) {
        const int64_t idx = m_flat_indices[i];
        const int s =
            idx >= 0
                ? create(out_slater_determinant, static_cast<size_t>(idx))
                : annihilate(out_slater_determinant,
                             static_cast<size_t>(-(idx + 1)));
        if (s == 0) {
          sign = 0;
          break;
        }
        sign *= s;
      }
      if (sign != 0 && (!check_restrictions ||
                        state_is_within_restrictions(out_slater_determinant))) {
        const auto contribution = coeff * amp * sign;
        if (m_flat_diagonal[op_idx]) {
          // out_slater_determinant == slater here (occupation conserved).
          diag_accum += contribution;
        } else {
          map_res[out_slater_determinant] += contribution;
        }
      }
      // Restore out_slater_determinant to `slater`: undo the [start_idx, i)
      // operators that actually toggled a bit. A failing op (s == 0) at index i did
      // not modify the determinant, so it is excluded.
      for (size_t j = start_idx; j < i; j++) {
        const int64_t idx = m_flat_indices[j];
        toggle_bit(out_slater_determinant,
                   idx >= 0 ? static_cast<size_t>(idx)
                            : static_cast<size_t>(-(idx + 1)));
      }
    }
    if (diag_accum != ManyBodyState::mapped_type(0.0, 0.0)) {
      map_res[slater] += diag_accum;
    }
  }
  local_res.reserve(map_res.size());
  for (auto &[k, v] : map_res) {
    if (std::norm(v) > cutoff2) {
      local_res.emplace_back(std::move(k), v);
    }
  }
#endif

  // Sort vector to move duplicate keys next to each other
  std::sort(local_res.begin(), local_res.end(),
            [](const std::pair<ManyBodyState::Key, ManyBodyState::Value> &a,
               const std::pair<ManyBodyState::Key, ManyBodyState::Value> &b) {
               return a.first < b.first;
            });

  // Build the final ManyBodyState
  ManyBodyState res{};
  res.reserve(local_res.size());
  for (auto &[slater, amp] : local_res) {
    res.emplace(std::move(slater), amp);
  }
  return res;
}

ManyBodyOperator &
ManyBodyOperator::operator+=(const ManyBodyOperator &other) noexcept {
  m_flat_dirty = true;
  if (m_ops.empty()) {
    m_ops = other.m_ops;
    return *this;
  }
  auto current = m_ops.begin();
  for (const auto &op : other.m_ops) {
    current = find(current, m_ops.end(), op.first);
    if (current == m_ops.end() || current->first != op.first) {
      current = m_ops.emplace(current, op);
    } else {
      current->second += op.second;
    }
  }

  return *this;
}

ManyBodyOperator &
ManyBodyOperator::operator-=(const ManyBodyOperator &other) noexcept {
  m_flat_dirty = true;
  if (m_ops.empty()) {
    m_ops = other.m_ops;
    for (auto &op : m_ops) {
      op.second = -op.second;
    }
    return *this;
  }
  auto current = m_ops.begin();
  for (const auto &op : other.m_ops) {
    current = find(current, m_ops.end(), op.first);
    if (current == m_ops.end() || current->first != op.first) {
      current = m_ops.emplace(current, op.first, -op.second);
    } else {
      current->second -= op.second;
    }
  }
  return *this;
}

ManyBodyOperator &ManyBodyOperator::operator*=(mapped_type s) noexcept {
  m_flat_dirty = true;
  for (auto &p : m_ops) {
    p.second *= s;
  }
  return *this;
}

ManyBodyOperator &ManyBodyOperator::operator/=(mapped_type s) noexcept {
  m_flat_dirty = true;
  for (auto &p : m_ops) {
    p.second /= s;
  }
  return *this;
}

ManyBodyOperator ManyBodyOperator::operator-() const noexcept {
  ManyBodyOperator res(*this);
  for (auto &p : res.m_ops) {
    p.second = -p.second;
  }
  return res;
}

void ManyBodyOperator::set_normal_ordering(bool enable) noexcept {
  if (enable != m_normal_order) {
    m_normal_order = enable;
    m_flat_dirty = true;
  }
}

bool ManyBodyOperator::normal_ordering() const noexcept { return m_normal_order; }

ManyBodyOperator::size_type ManyBodyOperator::num_flat_terms() const {
  if (m_flat_dirty) {
    build_flat_representation();
  }
  return m_flat_coeffs.size();
}

namespace {
using Term = std::pair<std::vector<int64_t>, std::complex<double>>;

// Rewrite one operator string (in product / left-to-right reading order) into a list of
// normal-ordered terms (creations before annihilations, each group ascending in orbital)
// using the fermionic anticommutators:
//   c_p c^d_q = delta_pq - c^d_q c_p   (contraction when p == q)
//   c^d_p c^d_q = -c^d_q c^d_p,  c_p c_q = -c_q c_p   (p != q; equal-orbital pair = 0).
// The recursion fixes the leftmost out-of-order adjacent pair and terminates because each
// swap reduces the inversion count and each contraction shortens the term.
void normal_order_recurse(std::vector<int64_t> ops, std::complex<double> coeff,
                          std::vector<Term> &out) {
  for (size_t k = 0; k + 1 < ops.size(); k++) {
    const int64_t a = ops[k];
    const int64_t b = ops[k + 1];
    const bool a_create = a >= 0;
    const bool b_create = b >= 0;
    const int64_t a_orb = a_create ? a : -(a + 1);
    const int64_t b_orb = b_create ? b : -(b + 1);

    // Pauli: two adjacent same-type operators on the same orbital annihilate the term.
    if (a_create == b_create && a_orb == b_orb) {
      return;
    }

    const int rank_a = a_create ? 0 : 1; // creations sort before annihilations
    const int rank_b = b_create ? 0 : 1;
    bool inverted;
    if (rank_a != rank_b) {
      inverted = rank_a > rank_b; // annihilation sitting left of a creation
    } else {
      inverted = a_orb > b_orb; // same type, orbital out of ascending order
    }
    if (!inverted) {
      continue;
    }

    if (rank_a == 1 && rank_b == 0) {
      // c_{a_orb} c^d_{b_orb}: contraction (only if same orbital) + anticommuted term.
      if (a_orb == b_orb) {
        std::vector<int64_t> contracted;
        contracted.reserve(ops.size() - 2);
        for (size_t m = 0; m < ops.size(); m++) {
          if (m != k && m != k + 1) {
            contracted.push_back(ops[m]);
          }
        }
        normal_order_recurse(std::move(contracted), coeff, out);
      }
    }
    std::vector<int64_t> swapped = ops;
    std::swap(swapped[k], swapped[k + 1]);
    normal_order_recurse(std::move(swapped), -coeff, out);
    return; // the recursive calls finish ordering the rest
  }
  out.emplace_back(std::move(ops), coeff);
}

// Build the list of terms apply() flattens: the raw operator terms, or their normal-
// ordered expansion (merged, with zero/Pauli terms dropped) when normal ordering is on.
std::vector<Term>
collect_flat_terms(const std::vector<ManyBodyOperator::value_type> &m_ops,
                   bool normal_order) {
  std::vector<Term> terms;
  if (!normal_order) {
    terms.reserve(m_ops.size());
    for (const auto &op : m_ops) {
      terms.emplace_back(op.first, op.second);
    }
    return terms;
  }

  std::vector<Term> expanded;
  for (const auto &op : m_ops) {
    // m_ops stores operators rightmost-first (apply order); reverse to product order.
    std::vector<int64_t> product(op.first.rbegin(), op.first.rend());
    normal_order_recurse(std::move(product), op.second, expanded);
  }
  // Back to apply/stored order, then merge identical terms and drop ~zero coefficients.
  for (auto &[ops, c] : expanded) {
    std::reverse(ops.begin(), ops.end());
  }
  std::sort(expanded.begin(), expanded.end(),
            [](const Term &x, const Term &y) { return x.first < y.first; });
  for (auto &[ops, c] : expanded) {
    if (!terms.empty() && terms.back().first == ops) {
      terms.back().second += c;
    } else {
      terms.emplace_back(std::move(ops), c);
    }
  }
  terms.erase(std::remove_if(terms.begin(), terms.end(),
                             [](const Term &t) { return std::norm(t.second) < 1e-24; }),
              terms.end());
  return terms;
}
} // namespace

void ManyBodyOperator::build_flat_representation() const {
  if (!m_flat_dirty) return;
  m_flat_indices.clear();
  m_flat_offsets.clear();
  m_flat_coeffs.clear();
  m_flat_diagonal.clear();
  m_flat_density.clear();
  m_density_mask.clear();
  m_density_coeff.clear();
  m_flat_onebody.clear();
  m_onebody_i.clear();
  m_onebody_j.clear();
  m_onebody_between.clear();

  const std::vector<Term> terms = collect_flat_terms(m_ops, m_normal_order);

  m_flat_offsets.push_back(0);
  std::vector<int64_t> creators;     // reused scratch
  std::vector<int64_t> annihilators; // reused scratch
  for (const auto& op : terms) {
    const auto& indices = op.first;
    m_flat_indices.insert(m_flat_indices.end(), indices.begin(), indices.end());
    m_flat_offsets.push_back(m_flat_indices.size());
    m_flat_coeffs.push_back(op.second);

    // Diagonal classification (Phase 2a): equal created/annihilated orbital
    // multisets <=> every orbital's occupation is conserved <=> diagonal. The empty
    // (constant) term has both multisets empty, so it is diagonal.
    creators.clear();
    annihilators.clear();
    for (int64_t idx : indices) {
      if (idx >= 0) {
        creators.push_back(idx);
      } else {
        annihilators.push_back(-(idx + 1));
      }
    }
    std::sort(creators.begin(), creators.end());
    std::sort(annihilators.begin(), annihilators.end());
    const bool diagonal = (creators == annihilators);
    m_flat_diagonal.push_back(diagonal ? 1 : 0);

    // Pure-number-product fast path (Phase 2b). Require balanced (== diagonal) and
    // all-distinct orbitals, then *probe*: apply the term to a determinant with exactly
    // the involved orbitals occupied. A nonzero sign proves the term is a pure product of
    // number operators (n_i, n_i n_j, ...) with an occupancy-independent constant sign;
    // anything that needs an empty orbital (e.g. c_i c^d_i = 1 - n_i) fails the probe and
    // is left to the general diagonal path. So correctness never depends on this firing.
    bool all_distinct = true;
    for (size_t k = 1; k < creators.size() && all_distinct; k++) {
      if (creators[k] == creators[k - 1]) {
        all_distinct = false;
      }
    }
    bool is_density = diagonal && all_distinct;
    ManyBodyState::key_type mask; // occupancy mask over the involved orbitals
    std::complex<double> signed_coeff{0.0, 0.0};
    if (is_density) {
      const size_t num_bits = 8 * sizeof(ManyBodyState::key_type::value_type);
      for (int64_t o : creators) {
        const size_t chunk = static_cast<size_t>(o) / num_bits;
        const size_t bit = num_bits - 1 - (static_cast<size_t>(o) % num_bits);
        if (chunk >= mask.size()) {
          mask.resize(chunk + 1, 0);
        }
        mask[chunk] |= static_cast<ManyBodyState::key_type::value_type>(1) << bit;
      }
      ManyBodyState::key_type probe = mask; // involved orbitals occupied
      double sign = 1;
      for (int64_t idx : indices) {
        const int s = idx >= 0
                          ? create(probe, static_cast<size_t>(idx))
                          : annihilate(probe, static_cast<size_t>(-(idx + 1)));
        if (s == 0) {
          sign = 0;
          break;
        }
        sign *= s;
      }
      if (sign == 0) {
        is_density = false; // not a pure n-product; defer to general diagonal path
      } else {
        signed_coeff = op.second * sign;
      }
    }
    m_flat_density.push_back(is_density ? 1 : 0);
    m_density_mask.push_back(std::move(mask));
    m_density_coeff.push_back(signed_coeff);

    // Off-diagonal one-body hop c^d_i c_j (i != j): exactly one creator and one
    // annihilator on different orbitals (Phase 2c). Precompute the between-mask so the
    // fermion sign is one popcount at apply time.
    const bool is_onebody = !diagonal && creators.size() == 1 &&
                            annihilators.size() == 1 &&
                            creators[0] != annihilators[0];
    size_t ob_i = 0;
    size_t ob_j = 0;
    ManyBodyState::key_type between;
    if (is_onebody) {
      ob_i = static_cast<size_t>(creators[0]);     // created orbital
      ob_j = static_cast<size_t>(annihilators[0]); // annihilated orbital
      const size_t num_bits = 8 * sizeof(ManyBodyState::key_type::value_type);
      const size_t lo = std::min(ob_i, ob_j);
      const size_t hi = std::max(ob_i, ob_j);
      for (size_t o = lo + 1; o < hi; o++) {
        const size_t chunk = o / num_bits;
        const size_t bit = num_bits - 1 - (o % num_bits);
        if (chunk >= between.size()) {
          between.resize(chunk + 1, 0);
        }
        between[chunk] |= static_cast<ManyBodyState::key_type::value_type>(1) << bit;
      }
    }
    m_flat_onebody.push_back(is_onebody ? 1 : 0);
    m_onebody_i.push_back(ob_i);
    m_onebody_j.push_back(ob_j);
    m_onebody_between.push_back(std::move(between));
  }
  m_flat_dirty = false;
}
