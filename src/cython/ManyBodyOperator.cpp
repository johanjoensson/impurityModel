#include <unordered_map>
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

[[nodiscard]] ManyBodyState ManyBodyOperator::apply(const ManyBodyState &state,
                                                    double cutoff) const {
  std::vector<std::pair<ManyBodyState::Key, ManyBodyState::Value>> local_res;
  std::unordered_map<ManyBodyState::key_type, ManyBodyState::mapped_type> map_res;

#if defined(PARALLEL)
  if (m_flat_dirty) {
    build_flat_representation();
  }
  unsigned int num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;
  std::vector<std::unordered_map<ManyBodyState::key_type, ManyBodyState::mapped_type>>
      local_maps(num_threads);
  ManyBodyState::size_type num_slater = state.size();
  size_t chunk_size = (num_slater + num_threads - 1) / num_threads;
  for (unsigned int t = 0; t < num_threads; t++) {
    size_t start_slater = t * chunk_size;
    size_t end_slater = std::min(start_slater + chunk_size, num_slater);
    if (start_slater >= num_slater) {
      break;
    }
    threads.push_back(std::thread([&, t, start_slater, end_slater]() {
      auto &tmp_map = local_maps[t];
      ManyBodyState::key_type out_slater_determinant;

      ManyBodyState::const_iterator it = state.begin(), end = state.begin();
      std::advance(it, start_slater);
      std::advance(end, end_slater);
      for (; it != end; it++) {
        const auto &[slater, amp] = *it;

        for (size_t op_idx = 0; op_idx < m_flat_coeffs.size(); op_idx++) {
          out_slater_determinant = slater;
          double sign = 1;
          const size_t start_idx = m_flat_offsets[op_idx];
          const size_t end_idx = m_flat_offsets[op_idx + 1];
          const auto coeff = m_flat_coeffs[op_idx];

          for (size_t i = start_idx; i < end_idx; i++) {
            const int64_t idx = m_flat_indices[i];
            if (idx >= 0) {
              sign *= create(out_slater_determinant, static_cast<size_t>(idx));
            } else {
              sign *= annihilate(out_slater_determinant,
                                 static_cast<size_t>(-(idx + 1)));
            }

            if (sign == 0) {
              break;
            }
          }
          if (sign != 0 &&
              state_is_within_restrictions(out_slater_determinant)) {
            tmp_map[out_slater_determinant] += coeff * amp * sign;
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

  for (auto &tmp_map : local_maps) {
    for (auto &[k, v] : tmp_map) {
      map_res[k] += v;
    }
  }

#else
  if (m_flat_dirty) {
    build_flat_representation();
  }
  ManyBodyState::key_type out_slater_determinant;
  for (const auto &[slater, amp] : state) {
    for (size_t op_idx = 0; op_idx < m_flat_coeffs.size(); op_idx++) {
      out_slater_determinant = slater;
      double sign = 1;
      const size_t start_idx = m_flat_offsets[op_idx];
      const size_t end_idx = m_flat_offsets[op_idx + 1];
      const auto coeff = m_flat_coeffs[op_idx];

      for (size_t i = start_idx; i < end_idx; i++) {
        const int64_t idx = m_flat_indices[i];
        if (idx >= 0) {
          sign *= create(out_slater_determinant, static_cast<size_t>(idx));
        } else {
          sign *= annihilate(out_slater_determinant,
                             static_cast<size_t>(-(idx + 1)));
        }

        if (sign == 0) {
          break;
        }
      }
      if (sign != 0 && state_is_within_restrictions(out_slater_determinant)) {
        map_res[out_slater_determinant] += coeff * amp * sign;
      }
    }
  }
#endif
  double cutoff2 = cutoff * cutoff;
  local_res.reserve(map_res.size());
  for (auto &[k, v] : map_res) {
    if (std::norm(v) > cutoff2) {
      local_res.emplace_back(std::move(k), v);
    }
  }

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

void ManyBodyOperator::build_flat_representation() const {
  if (!m_flat_dirty) return;
  m_flat_indices.clear();
  m_flat_offsets.clear();
  m_flat_coeffs.clear();
  
  m_flat_offsets.push_back(0);
  for (const auto& op : m_ops) {
    const auto& indices = op.first;
    m_flat_indices.insert(m_flat_indices.end(), indices.begin(), indices.end());
    m_flat_offsets.push_back(m_flat_indices.size());
    m_flat_coeffs.push_back(op.second);
  }
  m_flat_dirty = false;
}
