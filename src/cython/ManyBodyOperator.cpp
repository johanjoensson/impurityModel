#include "ManyBodyOperator.h"
#include "ManyBodyState.h"
#include <algorithm>
#include <boost/unordered/unordered_flat_map.hpp>
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

// Undo helper for the in-place apply (Phase 1a): toggle the occupation bit of
// orbital `idx`. Only ever called on orbitals whose create/annihilate already
// succeeded, so `state_idx` is in range. Bit-toggle is its own inverse and
// order-independent, so re-toggling every successfully-applied operator
// restores the determinant to its input value without copying it again per
// term.
static inline void toggle_bit(ManyBodyState::key_type &state,
                              size_t idx) noexcept {
  const size_t num_bits = 8 * sizeof(ManyBodyState::key_type::value_type);
  const size_t state_idx = idx / num_bits;
  const size_t bit_idx = num_bits - 1 - (idx % num_bits);
  state[state_idx] ^=
      (static_cast<ManyBodyState::key_type::value_type>(1) << bit_idx);
}

// Phase 2b fast-path test: are all orbitals in `mask` occupied in `state`?
// Chunks of `mask` beyond `state.size()` would require occupied orbitals the
// determinant cannot hold, so they fail. Zero mask chunks (and the empty mask
// of a constant term) pass.
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

// Is orbital `idx` occupied in `state`? (chunks beyond the determinant are
// empty.)
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

// Parity (0/1) of the number of occupied orbitals selected by `mask` -- the
// one-body fermion sign exponent. Loops over the actual key_size chunks.
static inline int mask_parity(const ManyBodyState::key_type &state,
                              const ManyBodyState::key_type &mask) noexcept {
  int pc = 0;
  const size_t lim = std::min(state.size(), mask.size());
  for (size_t j = 0; j < lim; j++) {
    pc += set_bits(state[j] & mask[j]);
  }
  return pc & 1;
}

// Every constructor leaves the operator canonical (see canonicalize()), so the
// stored terms of anything built from Python or from the algebra are in normal
// order. canonicalize() subsumes initialize_from_ops: it sorts, merges equal
// terms and drops the zeros as part of the rewrite.
ManyBodyOperator::ManyBodyOperator(const std::vector<value_type> &ops)
    : m_ops(ops) {
  canonicalize();
}

bool ManyBodyOperator::empty() const noexcept { return m_ops.empty(); }

bool ManyBodyOperator::clear() {
  m_ops.clear();
  m_flat_dirty = true;
  m_canonical = true; // the empty (zero) operator is trivially canonical
  return true;
}

ManyBodyOperator::ManyBodyOperator(std::vector<value_type> &&ops)
    : m_ops(std::move(ops)) {
  canonicalize();
}

ManyBodyOperator::ManyBodyOperator(const OPS_VEC &ops, const SCALAR_VEC &amps)
    : m_ops() {
  m_ops.reserve(ops.size());
  for (size_t i = 0; i < ops.size(); i++) {
    m_ops.emplace_back(ops[i], amps[i]);
  }
  canonicalize();
}
ManyBodyOperator::ManyBodyOperator(OPS_VEC &&ops, SCALAR_VEC &&amps) : m_ops() {
  m_ops.reserve(ops.size());
  for (size_t i = 0; i < ops.size(); i++) {
    m_ops.emplace_back(std::move(ops[i]), std::move(amps[i]));
  }
  canonicalize();
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

// The raw map-style accessors hand out a mutable reference to a coefficient and
// may insert an arbitrary (possibly non-normal-ordered) key, so they have to
// invalidate BOTH caches conservatively: the flat representation, because the
// caller is about to write a new amplitude through the returned reference, and
// the canonical flag, because the new key need not be in normal order.
ManyBodyOperator::mapped_type &
ManyBodyOperator::operator[](const key_type &key) {
  auto it = find(key);
  if (it == m_ops.end() || it->first != key) {
    it = m_ops.emplace(it, key, mapped_type{});
    m_canonical = false;
  }
  m_flat_dirty = true;
  return it->second;
}

ManyBodyOperator::mapped_type &ManyBodyOperator::operator[](key_type &&key) {
  auto it = find(key);
  if (it == m_ops.end() || it->first != key) {
    it = m_ops.emplace(it, std::move(key), mapped_type{});
    m_canonical = false;
  }
  m_flat_dirty = true;
  return it->second;
}

ManyBodyOperator::mapped_type &ManyBodyOperator::at(const key_type &key) {
  auto it = find(key);
  if (it == m_ops.end() || it->first != key) {
    throw std::out_of_range("Element not found!");
  }
  m_flat_dirty = true;
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
  m_canonical = false;
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
  m_canonical = false;
  return {it, true};
}
ManyBodyOperator::iterator ManyBodyOperator::insert(iterator pos,
                                                    const value_type &val) {
  auto it = find(m_ops.begin(), pos, val.first);
  m_flat_dirty = true;
  m_canonical = false;
  return m_ops.emplace(it, val);
}

ManyBodyOperator::iterator ManyBodyOperator::insert(iterator pos,
                                                    value_type &&val) {
  auto it = find(m_ops.begin(), pos, val.first);
  m_flat_dirty = true;
  m_canonical = false;
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
  m_canonical = false;
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
// Build the bitmask (a SlaterDeterminant key) marking the given orbital
// indices.
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
// boost::unordered_flat_map is an open-addressing (flat) hash map: contiguous
// storage, no per-node allocation, far better cache behavior than
// std::unordered_map for the apply accumulator. boost::hash does not pick up
// the std::hash<SlaterDeterminant> specialization in SlaterDeterminant.h, so
// wrap it explicitly.
struct SlaterKeyHash {
  std::size_t operator()(const ManyBodyState::key_type &k) const noexcept {
    return std::hash<ManyBodyState::key_type>{}(k);
  }
};
using ResultMap =
    boost::unordered_flat_map<ManyBodyState::key_type,
                              ManyBodyState::mapped_type, SlaterKeyHash>;

#if defined(PARALLEL)
// Maximum threads the apply path may use. Prefer OMP_NUM_THREADS (the cores the
// user requested for *this* MPI task) so one rank per task never oversubscribes
// the node; fall back to a single thread only when unset. Resolved once.
unsigned int apply_thread_cap() {
  static const unsigned int cap = []() -> unsigned int {
    if (const char *s = std::getenv("OMP_NUM_THREADS")) {
      char *end = nullptr;
      const long v = std::strtol(s, &end, 10);
      if (end != s && v > 0) {
        return static_cast<unsigned int>(v);
      }
    }
    return 1u;
  }();
  return cap;
}
#endif
} // namespace

[[nodiscard]] ManyBodyState ManyBodyOperator::apply(const ManyBodyState &state,
                                                    double cutoff) const {
  std::vector<std::pair<ManyBodyState::Key, ManyBodyState::Value>> local_res;
  const double cutoff2 = cutoff * cutoff;

  // Phase 1c: most operators carry no occupation restrictions, so hoist the
  // emptiness test out of the per-output hot path and skip
  // state_is_within_restrictions entirely when there is nothing to check.
  const bool check_restrictions = !std::get<0>(m_restrictions_mask).empty() ||
                                  !m_weighted_restrictions_mask.empty();

#if defined(PARALLEL)
  if (m_flat_dirty) {
    build_flat_representation();
  }
  // Phase 5b: scale the thread count to the workload instead of always grabbing
  // every core. Small states (few SDs) don't amortize thread spawn + the bucket
  // merge, and under MPI every rank would otherwise oversubscribe the node;
  // require >= MIN_SD_PER_THREAD input determinants per thread, so tiny applies
  // run on a single thread.
  constexpr size_t MIN_SD_PER_THREAD = 256;
  const ManyBodyState::size_type num_slater = state.size();
  const unsigned int hw = apply_thread_cap(); // OMP_NUM_THREADS, else 1
  const unsigned int want = static_cast<unsigned int>(
      std::max<size_t>(1, num_slater / MIN_SD_PER_THREAD));
  const unsigned int num_threads = std::max(1u, std::min(hw, want));
  // Output is partitioned across `num_buckets` by key hash so the merge below
  // runs one lock-free thread per bucket (disjoint key sets), instead of the
  // old serial merge that re-did every insert on one thread and capped the
  // speedup.
  const unsigned int num_buckets = num_threads;
  const SlaterKeyHash hasher;
  // local_buckets[t * num_buckets + b]: outputs from compute-thread t hashing
  // to bucket b.
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

        // Copy the input determinant ONCE per SD; each term applies in place
        // and undoes its operators afterwards (Phase 1a), instead of copying
        // per term.
        out_slater_determinant = slater;
        // Phase 2b: accumulate all diagonal terms into one scalar -> single
        // insert.
        ManyBodyState::mapped_type diag_accum{0.0, 0.0};
        for (size_t op_idx = 0; op_idx < m_flat_coeffs.size(); op_idx++) {
          // Phase 2b fast path: pure number-operator product -> one occupancy
          // AND-test and a constant-signed scalar add, no create/annihilate.
          if (m_flat_density[op_idx]) {
            if (mask_occupied(slater, m_density_mask[op_idx]) &&
                (!check_restrictions || state_is_within_restrictions(slater))) {
              diag_accum += m_density_coeff[op_idx] * amp;
            }
            continue;
          }
          // Phase 2c fast path: off-diagonal one-body hop c^d_i c_j -> bit
          // tests + a masked-popcount sign, no create/annihilate.
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

  // Phase 5a: parallel merge -- one thread per output bucket. Bucket b owns a
  // disjoint set of keys (all hashing to b), so the threads never touch the
  // same entry and need no locks. Each thread also applies the cutoff while
  // collecting its bucket's survivors.
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
    // Copy the input determinant ONCE per SD; each term applies its operators
    // in place and undoes them afterwards (Phase 1a), instead of copying per
    // term.
    out_slater_determinant = slater;
    // Phase 2b: sum every diagonal term's contribution (all map to `slater`)
    // into one scalar and emit a single insert, instead of one colliding insert
    // per term.
    ManyBodyState::mapped_type diag_accum{0.0, 0.0};
    for (size_t op_idx = 0; op_idx < m_flat_coeffs.size(); op_idx++) {
      // Phase 2b fast path: pure number-operator product -> one occupancy
      // AND-test and a constant-signed scalar add, no create/annihilate.
      if (m_flat_density[op_idx]) {
        if (mask_occupied(slater, m_density_mask[op_idx]) &&
            (!check_restrictions || state_is_within_restrictions(slater))) {
          diag_accum += m_density_coeff[op_idx] * amp;
        }
        continue;
      }
      // Phase 2c fast path: off-diagonal one-body hop c^d_i c_j -> bit tests +
      // a masked- popcount sign, no create/annihilate.
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
            map_res[out_slater_determinant] +=
                m_flat_coeffs[op_idx] * amp * sgn;
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
            idx >= 0 ? create(out_slater_determinant, static_cast<size_t>(idx))
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
      // operators that actually toggled a bit. A failing op (s == 0) at index i
      // did not modify the determinant, so it is excluded.
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

// True when the extension was compiled with the opt-in threaded apply
// (IMPURITYMODEL_PARALLEL=1 at install time). Exposed to Python so tests can
// choose exact vs tolerance assertions: the threaded merge changes the
// duplicate-accumulation order, so bit-for-bit equality is a serial-build
// property (same as for the scalar threaded apply).
bool apply_parallel_build() noexcept {
#if defined(PARALLEL)
  return true;
#else
  return false;
#endif
}

[[nodiscard]] ManyBodyBlockState
ManyBodyOperator::apply(const ManyBodyBlockState &block, double cutoff) const {
  // Block variant of the serial apply above: one pass over the shared support,
  // per-(determinant, term) work done once, p amplitudes emitted per hit. The
  // accumulator maps out-determinant -> row index of a growing row-major buffer
  // (one hash op per (det, term) instead of p). Per-column arithmetic mirrors
  // the scalar path exactly: same term order, same += sequence per column.
  if (m_flat_dirty) {
    build_flat_representation();
  }
  const std::size_t p = block.width();
  const double cutoff2 = cutoff * cutoff;
  const bool check_restrictions = !std::get<0>(m_restrictions_mask).empty() ||
                                  !m_weighted_restrictions_mask.empty();

  boost::unordered_flat_map<ManyBodyState::key_type, std::size_t, SlaterKeyHash>
      row_of;
  row_of.reserve(block.rows());
  std::vector<ManyBodyBlockState::Value> acc; // row-major, row_of.size() * p
  acc.reserve(block.rows() * p);

  // Row of `k` in acc, appending a zero row on first sight. resize() grows the
  // buffer geometrically, so the amortized cost stays O(1) per new row.
  const auto emit_row = [&](const ManyBodyState::key_type &k) {
    const auto [it, inserted] = row_of.try_emplace(k, row_of.size());
    if (inserted) {
      acc.resize(acc.size() + p);
    }
    return acc.data() + it->second * p;
  };

#if defined(PARALLEL)
  // Threaded block apply (Phase 2.5): partition the input ROWS across threads,
  // one block accumulator per (thread, bucket) pair, lock-free per-bucket merge
  // — the same structure as the scalar threaded apply above. Each row carries
  // ~p times the scalar work, so the per-thread workload floor shrinks
  // accordingly.
  constexpr size_t MIN_SD_PER_THREAD = 256;
  const size_t min_rows_per_thread =
      std::max<size_t>(1, MIN_SD_PER_THREAD / std::max<size_t>(1, p));
  const unsigned int hw = apply_thread_cap();
  const unsigned int want = static_cast<unsigned int>(
      std::max<size_t>(1, block.rows() / min_rows_per_thread));
  const unsigned int num_threads = std::max(1u, std::min(hw, want));
  if (num_threads > 1) {
    struct BucketAcc {
      boost::unordered_flat_map<ManyBodyState::key_type, std::size_t,
                                SlaterKeyHash>
          row_of;
      std::vector<ManyBodyBlockState::Value> acc;
    };
    const unsigned int num_buckets = num_threads;
    const SlaterKeyHash hasher;
    std::vector<BucketAcc> local_buckets(static_cast<size_t>(num_threads) *
                                         num_buckets);
    std::vector<std::thread> threads;
    const size_t chunk_size = (block.rows() + num_threads - 1) / num_threads;
    for (unsigned int t = 0; t < num_threads; t++) {
      const size_t start_row = t * chunk_size;
      const size_t end_row = std::min(start_row + chunk_size, block.rows());
      if (start_row >= block.rows()) {
        break;
      }
      threads.push_back(std::thread([&, t, start_row, end_row]() {
        BucketAcc *buckets =
            &local_buckets[static_cast<size_t>(t) * num_buckets];
        const auto emit_row_t = [&](const ManyBodyState::key_type &k) {
          BucketAcc &ba = buckets[hasher(k) % num_buckets];
          const auto [it, inserted] =
              ba.row_of.try_emplace(k, ba.row_of.size());
          if (inserted) {
            ba.acc.resize(ba.acc.size() + p);
          }
          return ba.acc.data() + it->second * p;
        };
        // Per-row term loop, duplicated from the serial body below (the same
        // precedent as the scalar threaded apply: the loop reads private
        // flat-term members, so a shared free function would need an unwieldy
        // signature).
        std::vector<ManyBodyBlockState::Value> diag_accum(p);
        ManyBodyState::key_type out_sd;
        for (std::size_t r = start_row; r < end_row; ++r) {
          const auto &slater = block.key(r);
          const ManyBodyBlockState::Value *amp = block.row(r);
          out_sd = slater;
          std::fill(diag_accum.begin(), diag_accum.end(),
                    ManyBodyBlockState::Value{0.0, 0.0});
          for (size_t op_idx = 0; op_idx < m_flat_coeffs.size(); op_idx++) {
            if (m_flat_density[op_idx]) {
              if (mask_occupied(slater, m_density_mask[op_idx]) &&
                  (!check_restrictions ||
                   state_is_within_restrictions(slater))) {
                const auto coeff = m_density_coeff[op_idx];
                for (std::size_t c = 0; c < p; ++c) {
                  diag_accum[c] += coeff * amp[c];
                }
              }
              continue;
            }
            if (m_flat_onebody[op_idx]) {
              const size_t ob_i = m_onebody_i[op_idx];
              const size_t ob_j = m_onebody_j[op_idx];
              if (bit_set(slater, ob_j) && !bit_set(slater, ob_i)) {
                const double sgn =
                    mask_parity(slater, m_onebody_between[op_idx]) ? -1.0 : 1.0;
                toggle_bit(out_sd, ob_j);
                toggle_bit(out_sd, ob_i);
                if (!check_restrictions ||
                    state_is_within_restrictions(out_sd)) {
                  const auto coeff = m_flat_coeffs[op_idx] * sgn;
                  ManyBodyBlockState::Value *dst = emit_row_t(out_sd);
                  for (std::size_t c = 0; c < p; ++c) {
                    dst[c] += coeff * amp[c];
                  }
                }
                toggle_bit(out_sd, ob_j);
                toggle_bit(out_sd, ob_i);
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
                      ? create(out_sd, static_cast<size_t>(idx))
                      : annihilate(out_sd, static_cast<size_t>(-(idx + 1)));
              if (s == 0) {
                sign = 0;
                break;
              }
              sign *= s;
            }
            if (sign != 0 &&
                (!check_restrictions || state_is_within_restrictions(out_sd))) {
              const auto scaled = coeff * sign;
              if (m_flat_diagonal[op_idx]) {
                for (std::size_t c = 0; c < p; ++c) {
                  diag_accum[c] += scaled * amp[c];
                }
              } else {
                ManyBodyBlockState::Value *dst = emit_row_t(out_sd);
                for (std::size_t c = 0; c < p; ++c) {
                  dst[c] += scaled * amp[c];
                }
              }
            }
            for (size_t j = start_idx; j < i; j++) {
              const int64_t idx = m_flat_indices[j];
              toggle_bit(out_sd, idx >= 0 ? static_cast<size_t>(idx)
                                          : static_cast<size_t>(-(idx + 1)));
            }
          }
          bool any_diag = false;
          for (std::size_t c = 0; c < p; ++c) {
            if (diag_accum[c] != ManyBodyBlockState::Value(0.0, 0.0)) {
              any_diag = true;
              break;
            }
          }
          if (any_diag) {
            ManyBodyBlockState::Value *dst = emit_row_t(slater);
            for (std::size_t c = 0; c < p; ++c) {
              dst[c] += diag_accum[c];
            }
          }
        }
      }));
    }
    for (auto &thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }

    // Lock-free per-bucket merge (disjoint key sets); threads iterate the
    // compute threads in index order, then apply the whole-row cutoff.
    std::vector<std::vector<ManyBodyBlockState::Key>> bucket_keys(num_buckets);
    std::vector<std::vector<ManyBodyBlockState::Value>> bucket_amps(
        num_buckets);
    std::vector<std::thread> merge_threads;
    for (unsigned int b = 0; b < num_buckets; b++) {
      merge_threads.push_back(std::thread([&, b]() {
        BucketAcc macc;
        for (unsigned int t = 0; t < num_threads; t++) {
          const BucketAcc &src =
              local_buckets[static_cast<size_t>(t) * num_buckets + b];
          for (const auto &[k, row] : src.row_of) {
            const auto [it, inserted] =
                macc.row_of.try_emplace(k, macc.row_of.size());
            if (inserted) {
              macc.acc.resize(macc.acc.size() + p);
            }
            ManyBodyBlockState::Value *dst = macc.acc.data() + it->second * p;
            const ManyBodyBlockState::Value *add = src.acc.data() + row * p;
            for (std::size_t c = 0; c < p; ++c) {
              dst[c] += add[c];
            }
          }
        }
        auto &okeys = bucket_keys[b];
        auto &oamps = bucket_amps[b];
        okeys.reserve(macc.row_of.size());
        for (auto &[k, row] : macc.row_of) {
          const ManyBodyBlockState::Value *src_row = macc.acc.data() + row * p;
          for (std::size_t c = 0; c < p; ++c) {
            if (std::norm(src_row[c]) > cutoff2) {
              okeys.push_back(k);
              oamps.insert(oamps.end(), src_row, src_row + p);
              break;
            }
          }
        }
      }));
    }
    for (auto &thread : merge_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }

    // Assemble: gather the surviving (key, row) pairs, sort by key, build.
    size_t total_out = 0;
    for (const auto &v : bucket_keys) {
      total_out += v.size();
    }
    std::vector<
        std::pair<ManyBodyBlockState::Key, const ManyBodyBlockState::Value *>>
        survivors;
    survivors.reserve(total_out);
    for (unsigned int b = 0; b < num_buckets; b++) {
      for (size_t i = 0; i < bucket_keys[b].size(); ++i) {
        survivors.emplace_back(std::move(bucket_keys[b][i]),
                               bucket_amps[b].data() + i * p);
      }
    }
    std::sort(survivors.begin(), survivors.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    std::vector<ManyBodyBlockState::Key> keys;
    keys.reserve(total_out);
    std::vector<ManyBodyBlockState::Value> amps;
    amps.reserve(total_out * p);
    for (auto &[k, src_row] : survivors) {
      keys.push_back(std::move(k));
      amps.insert(amps.end(), src_row, src_row + p);
    }
    return ManyBodyBlockState(std::move(keys), std::move(amps), p);
  }
#endif

  std::vector<ManyBodyBlockState::Value> diag_accum(p);
  ManyBodyState::key_type out_slater_determinant;
  for (std::size_t r = 0; r < block.rows(); ++r) {
    const auto &slater = block.key(r);
    const ManyBodyBlockState::Value *amp = block.row(r);
    out_slater_determinant = slater;
    std::fill(diag_accum.begin(), diag_accum.end(),
              ManyBodyBlockState::Value{0.0, 0.0});
    for (size_t op_idx = 0; op_idx < m_flat_coeffs.size(); op_idx++) {
      if (m_flat_density[op_idx]) {
        if (mask_occupied(slater, m_density_mask[op_idx]) &&
            (!check_restrictions || state_is_within_restrictions(slater))) {
          const auto coeff = m_density_coeff[op_idx];
          for (std::size_t c = 0; c < p; ++c) {
            diag_accum[c] += coeff * amp[c];
          }
        }
        continue;
      }
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
            const auto coeff = m_flat_coeffs[op_idx] * sgn;
            ManyBodyBlockState::Value *dst = emit_row(out_slater_determinant);
            for (std::size_t c = 0; c < p; ++c) {
              dst[c] += coeff * amp[c];
            }
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
            idx >= 0 ? create(out_slater_determinant, static_cast<size_t>(idx))
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
        const auto scaled = coeff * sign;
        if (m_flat_diagonal[op_idx]) {
          // out_slater_determinant == slater here (occupation conserved).
          for (std::size_t c = 0; c < p; ++c) {
            diag_accum[c] += scaled * amp[c];
          }
        } else {
          ManyBodyBlockState::Value *dst = emit_row(out_slater_determinant);
          for (std::size_t c = 0; c < p; ++c) {
            dst[c] += scaled * amp[c];
          }
        }
      }
      for (size_t j = start_idx; j < i; j++) {
        const int64_t idx = m_flat_indices[j];
        toggle_bit(out_slater_determinant,
                   idx >= 0 ? static_cast<size_t>(idx)
                            : static_cast<size_t>(-(idx + 1)));
      }
    }
    bool any_diag = false;
    for (std::size_t c = 0; c < p; ++c) {
      if (diag_accum[c] != ManyBodyBlockState::Value(0.0, 0.0)) {
        any_diag = true;
        break;
      }
    }
    if (any_diag) {
      ManyBodyBlockState::Value *dst = emit_row(slater);
      for (std::size_t c = 0; c < p; ++c) {
        dst[c] += diag_accum[c];
      }
    }
  }

  // Keep rows where ANY column survives the cutoff, then sort by determinant to
  // establish the container's sorted-unique invariant.
  std::vector<std::pair<ManyBodyState::key_type, std::size_t>> survivors;
  survivors.reserve(row_of.size());
  for (auto &[k, row] : row_of) {
    const ManyBodyBlockState::Value *src = acc.data() + row * p;
    for (std::size_t c = 0; c < p; ++c) {
      if (std::norm(src[c]) > cutoff2) {
        survivors.emplace_back(std::move(k), row);
        break;
      }
    }
  }
  std::sort(survivors.begin(), survivors.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });

  std::vector<ManyBodyBlockState::Key> keys;
  keys.reserve(survivors.size());
  std::vector<ManyBodyBlockState::Value> amps;
  amps.reserve(survivors.size() * p);
  for (auto &[k, row] : survivors) {
    keys.push_back(std::move(k));
    const ManyBodyBlockState::Value *src = acc.data() + row * p;
    amps.insert(amps.end(), src, src + p);
  }
  return ManyBodyBlockState(std::move(keys), std::move(amps), p);
}

ManyBodyOperator &
ManyBodyOperator::operator+=(const ManyBodyOperator &other) noexcept {
  m_flat_dirty = true;
  // Merging two canonical term sets keeps the result canonical: no term string
  // changes, and equal keys are combined in place.
  m_canonical = m_canonical && other.m_canonical;
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
  m_canonical = m_canonical && other.m_canonical;
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

ManyBodyOperator &ManyBodyOperator::operator*=(const ManyBodyOperator &other) {
  m_flat_dirty = true;
  // An empty operator is the ZERO operator (the identity is the empty *term*,
  // {} -> 1), so either factor being empty annihilates the product.
  if (m_ops.empty() || other.m_ops.empty()) {
    m_ops.clear();
    m_canonical = true;
    return *this;
  }
  std::vector<value_type> combined;
  combined.reserve(m_ops.size() * other.m_ops.size());
  for (const auto &left : m_ops) {
    for (const auto &right : other.m_ops) {
      key_type ops;
      ops.reserve(left.first.size() + right.first.size());
      // Terms are stored rightmost-first, i.e. in the order apply() runs them,
      // so composing A*B (act with B, then with A) lays down B's string first.
      ops.insert(ops.end(), right.first.begin(), right.first.end());
      ops.insert(ops.end(), left.first.begin(), left.first.end());
      combined.emplace_back(std::move(ops), left.second * right.second);
    }
  }
  m_ops = std::move(combined);
  // Restores the sorted/merged storage invariant AND collapses the product:
  // concatenated strings are generally not in normal order, and this is where
  // Pauli-vanishing and contracting pairs cancel.
  canonicalize();
  return *this;
}

ManyBodyOperator ManyBodyOperator::power(unsigned n) const {
  ManyBodyOperator res;
  res.set_constant(mapped_type{1.0, 0.0}); // n == 0 -> identity
  for (unsigned k = 0; k < n; k++) {
    res *= *this;
  }
  return res;
}

ManyBodyOperator::mapped_type ManyBodyOperator::constant() const noexcept {
  // The empty key sorts before every other term, so this is the front element
  // whenever a constant is present at all.
  if (!m_ops.empty() && m_ops.front().first.empty()) {
    return m_ops.front().second;
  }
  return mapped_type{0.0, 0.0};
}

void ManyBodyOperator::set_constant(mapped_type c) noexcept {
  m_flat_dirty = true;
  const bool have = !m_ops.empty() && m_ops.front().first.empty();
  const bool want = c != mapped_type{0.0, 0.0};
  if (have && want) {
    m_ops.front().second = c;
  } else if (have) {
    m_ops.erase(m_ops.begin());
  } else if (want) {
    m_ops.emplace(m_ops.begin(), key_type{}, c);
  }
  // m_canonical is deliberately left alone: the empty term is canonical and is
  // written at the front, which is exactly where the ordering puts it.
}

ManyBodyOperator &ManyBodyOperator::operator+=(mapped_type s) noexcept {
  set_constant(constant() + s);
  return *this;
}

ManyBodyOperator &ManyBodyOperator::operator-=(mapped_type s) noexcept {
  set_constant(constant() - s);
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

bool ManyBodyOperator::normal_ordering() const noexcept {
  return m_normal_order;
}

ManyBodyOperator::size_type ManyBodyOperator::num_flat_terms() const {
  if (m_flat_dirty) {
    build_flat_representation();
  }
  return m_flat_coeffs.size();
}

namespace {
using Term = std::pair<std::vector<int64_t>, std::complex<double>>;

// Rewrite one operator string (in product / left-to-right reading order) into a
// list of normal-ordered terms (creations before annihilations, each group
// ascending in orbital) using the fermionic anticommutators:
//   c_p c^d_q = delta_pq - c^d_q c_p   (contraction when p == q)
//   c^d_p c^d_q = -c^d_q c^d_p,  c_p c_q = -c_q c_p   (p != q; equal-orbital
//   pair = 0).
// The recursion fixes the leftmost out-of-order adjacent pair and terminates
// because each swap reduces the inversion count and each contraction shortens
// the term.
void normal_order_recurse(std::vector<int64_t> ops, std::complex<double> coeff,
                          std::vector<Term> &out) {
  for (size_t k = 0; k + 1 < ops.size(); k++) {
    const int64_t a = ops[k];
    const int64_t b = ops[k + 1];
    const bool a_create = a >= 0;
    const bool b_create = b >= 0;
    const int64_t a_orb = a_create ? a : -(a + 1);
    const int64_t b_orb = b_create ? b : -(b + 1);

    // Pauli: two adjacent same-type operators on the same orbital annihilate
    // the term.
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
      // c_{a_orb} c^d_{b_orb}: contraction (only if same orbital) +
      // anticommuted term.
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

// Build the list of terms apply() flattens: the raw operator terms, or their
// normal- ordered expansion (merged, with zero/Pauli terms dropped) when normal
// ordering is on.
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
    // m_ops stores operators rightmost-first (apply order); reverse to product
    // order.
    std::vector<int64_t> product(op.first.rbegin(), op.first.rend());
    normal_order_recurse(std::move(product), op.second, expanded);
  }
  // Back to apply/stored order, then merge identical terms and drop ~zero
  // coefficients.
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
  terms.erase(
      std::remove_if(terms.begin(), terms.end(),
                     [](const Term &t) { return std::norm(t.second) < 1e-24; }),
      terms.end());
  return terms;
}
} // namespace

void ManyBodyOperator::canonicalize() {
  // collect_flat_terms already does the whole job: it normal-orders every term
  // string, sorts by key, merges terms equal up to ordering and drops the ones
  // whose coefficient cancelled. Its Term type is layout-identical to
  // value_type, so the result is the new m_ops.
  m_ops = collect_flat_terms(m_ops, true);
  m_canonical = true;
  m_flat_dirty = true;
}

bool ManyBodyOperator::is_canonical() const noexcept { return m_canonical; }

void ManyBodyOperator::build_flat_representation() const {
  if (!m_flat_dirty)
    return;
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

  // Canonical storage is already normal-ordered and merged, so the flat rep is
  // a plain copy and the (potentially expensive) anticommutator recursion is
  // skipped. It only has work to do for terms injected through the raw
  // mutators.
  const std::vector<Term> terms =
      collect_flat_terms(m_ops, m_normal_order && !m_canonical);

  m_flat_offsets.push_back(0);
  std::vector<int64_t> creators;     // reused scratch
  std::vector<int64_t> annihilators; // reused scratch
  for (const auto &op : terms) {
    const auto &indices = op.first;
    m_flat_indices.insert(m_flat_indices.end(), indices.begin(), indices.end());
    m_flat_offsets.push_back(m_flat_indices.size());
    m_flat_coeffs.push_back(op.second);

    // Diagonal classification (Phase 2a): equal created/annihilated orbital
    // multisets <=> every orbital's occupation is conserved <=> diagonal. The
    // empty (constant) term has both multisets empty, so it is diagonal.
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

    // Pure-number-product fast path (Phase 2b). Require balanced (== diagonal)
    // and all-distinct orbitals, then *probe*: apply the term to a determinant
    // with exactly the involved orbitals occupied. A nonzero sign proves the
    // term is a pure product of number operators (n_i, n_i n_j, ...) with an
    // occupancy-independent constant sign; anything that needs an empty orbital
    // (e.g. c_i c^d_i = 1 - n_i) fails the probe and is left to the general
    // diagonal path. So correctness never depends on this firing.
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
        mask[chunk] |= static_cast<ManyBodyState::key_type::value_type>(1)
                       << bit;
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
        is_density =
            false; // not a pure n-product; defer to general diagonal path
      } else {
        signed_coeff = op.second * sign;
      }
    }
    m_flat_density.push_back(is_density ? 1 : 0);
    m_density_mask.push_back(std::move(mask));
    m_density_coeff.push_back(signed_coeff);

    // Off-diagonal one-body hop c^d_i c_j (i != j): exactly one creator and one
    // annihilator on different orbitals (Phase 2c). Precompute the between-mask
    // so the fermion sign is one popcount at apply time.
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
        between[chunk] |= static_cast<ManyBodyState::key_type::value_type>(1)
                          << bit;
      }
    }
    m_flat_onebody.push_back(is_onebody ? 1 : 0);
    m_onebody_i.push_back(ob_i);
    m_onebody_j.push_back(ob_j);
    m_onebody_between.push_back(std::move(between));
  }
  m_flat_dirty = false;
}

void ManyBodyOperator::prune(double tol) noexcept {
  const double tol2 = tol * tol;
  m_ops.erase(std::remove_if(m_ops.begin(), m_ops.end(),
                             [tol2](const value_type &t) {
                               return std::norm(t.second) <= tol2;
                             }),
              m_ops.end());
  m_flat_dirty = true;
  // Dropping terms cannot disturb the ordering of the ones that remain.
}

ManyBodyOperator ManyBodyOperator::adjoint() const {
  std::vector<value_type> terms;
  terms.reserve(m_ops.size());
  for (const auto &[ops, coeff] : m_ops) {
    key_type dag;
    dag.reserve(ops.size());
    // (o_n ... o_1)^dagger = o_1^dagger ... o_n^dagger: reverse the string and
    // dagger each factor. In the index encoding (i >= 0 creates orbital i,
    // i < 0 annihilates orbital -(i+1)) daggering is -(i+1) either way, since
    // that maps create(i) <-> annihilate(i) involutively.
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
      dag.push_back(-(*it + 1));
    }
    terms.emplace_back(std::move(dag), std::conj(coeff));
  }
  return ManyBodyOperator{std::move(terms)};
}

bool ManyBodyOperator::is_hermitian(double tol) const {
  return approx_equal(adjoint(), tol);
}

ManyBodyOperator ManyBodyOperator::hermitian_part() const {
  ManyBodyOperator res = *this;
  res += adjoint();
  res *= mapped_type{0.5, 0.0};
  return res;
}

std::vector<int64_t> ManyBodyOperator::orbitals() const {
  std::vector<int64_t> orbs;
  for (const auto &[ops, coeff] : m_ops) {
    for (int64_t idx : ops) {
      orbs.push_back(idx >= 0 ? idx : -(idx + 1));
    }
  }
  std::sort(orbs.begin(), orbs.end());
  orbs.erase(std::unique(orbs.begin(), orbs.end()), orbs.end());
  return orbs;
}

size_t ManyBodyOperator::body_rank() const noexcept {
  size_t rank = 0;
  for (const auto &[ops, coeff] : m_ops) {
    rank = std::max(rank, (ops.size() + 1) / 2);
  }
  return rank;
}

bool ManyBodyOperator::approx_equal(const ManyBodyOperator &other,
                                    double tol) const {
  // Both sides are sorted by key, so walk them together and treat a key present
  // on only one side as having coefficient zero on the other.
  const double tol2 = tol * tol;
  auto a = m_ops.cbegin();
  auto b = other.m_ops.cbegin();
  while (a != m_ops.cend() || b != other.m_ops.cend()) {
    if (b == other.m_ops.cend() || (a != m_ops.cend() && a->first < b->first)) {
      if (std::norm(a->second) > tol2) {
        return false;
      }
      ++a;
    } else if (a == m_ops.cend() || b->first < a->first) {
      if (std::norm(b->second) > tol2) {
        return false;
      }
      ++b;
    } else {
      if (std::norm(a->second - b->second) > tol2) {
        return false;
      }
      ++a;
      ++b;
    }
  }
  return true;
}

namespace {
// Orbital support of one term, precomputed once per term instead of per pair.
struct TermSupport {
  // Cheap reject mask: bit (orbital % 64). Two terms sharing an orbital always
  // share that bit, so a zero AND proves the supports are disjoint; the reverse
  // can alias (orbitals 64 apart), which is why a nonzero AND falls through to
  // the exact test rather than concluding anything.
  uint64_t signature;
  std::vector<int64_t> orbitals; // sorted, unique
  size_t length;                 // number of ladder operators in the string
};

TermSupport make_support(const ManyBodyOperator::key_type &ops) {
  TermSupport s;
  s.signature = 0;
  s.length = ops.size();
  s.orbitals.reserve(ops.size());
  for (int64_t idx : ops) {
    const int64_t orb = idx >= 0 ? idx : -(idx + 1);
    s.signature |= (uint64_t{1} << (static_cast<uint64_t>(orb) & 63U));
    s.orbitals.push_back(orb);
  }
  std::sort(s.orbitals.begin(), s.orbitals.end());
  s.orbitals.erase(std::unique(s.orbitals.begin(), s.orbitals.end()),
                   s.orbitals.end());
  return s;
}

bool supports_disjoint(const TermSupport &a, const TermSupport &b) noexcept {
  if ((a.signature & b.signature) == 0) {
    return true;
  }
  // Exact merge of two sorted orbital lists; stops at the first shared orbital.
  size_t i = 0;
  size_t j = 0;
  while (i < a.orbitals.size() && j < b.orbitals.size()) {
    if (a.orbitals[i] < b.orbitals[j]) {
      i++;
    } else if (b.orbitals[j] < a.orbitals[i]) {
      j++;
    } else {
      return false;
    }
  }
  return true;
}

// Shared engine for [A,B] (anti == false) and {A,B} (anti == true).
ManyBodyOperator bracket(const ManyBodyOperator &A, const ManyBodyOperator &B,
                         bool anti) {
  if (A.empty() || B.empty()) {
    return ManyBodyOperator{};
  }
  std::vector<TermSupport> support_a;
  std::vector<TermSupport> support_b;
  support_a.reserve(A.size());
  support_b.reserve(B.size());
  for (const auto &t : A) {
    support_a.push_back(make_support(t.first));
  }
  for (const auto &t : B) {
    support_b.push_back(make_support(t.first));
  }

  const ManyBodyOperator::mapped_type second_sign =
      anti ? ManyBodyOperator::mapped_type{1.0, 0.0}
           : ManyBodyOperator::mapped_type{-1.0, 0.0};

  std::vector<ManyBodyOperator::value_type> terms;
  size_t ia = 0;
  for (const auto &a : A) {
    size_t ib = 0;
    for (const auto &b : B) {
      const TermSupport &sa = support_a[ia];
      const TermSupport &sb = support_b[ib];
      ib++;
      // Disjoint strings obey a b = (-1)^(len_a * len_b) b a, so the pair's two
      // orderings either cancel or reinforce depending on that parity alone.
      // Skipping is exact: it drops only pairs contributing identically zero.
      if (supports_disjoint(sa, sb)) {
        const bool both_odd = (sa.length % 2 == 1) && (sb.length % 2 == 1);
        if (anti ? both_odd : !both_odd) {
          continue;
        }
      }
      const ManyBodyOperator::mapped_type coeff = a.second * b.second;
      // Stored order is apply order (rightmost first), so A*B lays down B's
      // string first, and B*A lays down A's first.
      ManyBodyOperator::key_type ab;
      ab.reserve(a.first.size() + b.first.size());
      ab.insert(ab.end(), b.first.begin(), b.first.end());
      ab.insert(ab.end(), a.first.begin(), a.first.end());
      terms.emplace_back(std::move(ab), coeff);

      ManyBodyOperator::key_type ba;
      ba.reserve(a.first.size() + b.first.size());
      ba.insert(ba.end(), a.first.begin(), a.first.end());
      ba.insert(ba.end(), b.first.begin(), b.first.end());
      terms.emplace_back(std::move(ba), second_sign * coeff);
    }
    ia++;
  }
  // One canonicalization over the combined term list: the two orderings of a
  // pair only cancel once they are both in normal form, so building the
  // products separately and subtracting would leave the cancellation undone.
  return ManyBodyOperator{std::move(terms)};
}
} // namespace

ManyBodyOperator commutator(const ManyBodyOperator &A,
                            const ManyBodyOperator &B) {
  return bracket(A, B, false);
}

ManyBodyOperator anticommutator(const ManyBodyOperator &A,
                                const ManyBodyOperator &B) {
  return bracket(A, B, true);
}
