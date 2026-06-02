#ifdef PARALLEL_STL
#include <execution>
#define PAR std::execution::par_unseq,
#else
#define PAR
#endif

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
#include <cstddef>

constexpr int set_bits(ManyBodyState::key_type::value_type byte) noexcept {
#if __cplusplus >= 202002L
  return std::popcount<ManyBodyState::key_type::value_type>(byte);
#else
  return std::bitset<sizeof(ManyBodyState::key_type::value_type) * CHAR_BIT>(
             byte)
      .count();
#endif
}

[[nodiscard]] std::pair<int, ManyBodyState::key_type>
create(const ManyBodyState::key_type &in_state, size_t idx) /*noexcept*/ {
  ManyBodyState::key_type state = in_state;
  const size_t num_bits{8 * sizeof(ManyBodyState::key_type::value_type)};
  const size_t state_idx = idx / num_bits;
  const size_t bit_idx = num_bits - 1 - (idx % num_bits);
  const ManyBodyState::key_type::value_type mask =
      static_cast<ManyBodyState::key_type::value_type>(1) << bit_idx;
  if (state[state_idx] & mask) {
    return {0, state};
  }
  size_t sign = 0;
  for (size_t i = 0; i < state_idx; i++) {
    sign += set_bits(state[i]) % 2;
  }
  sign += set_bits(state[state_idx] >> bit_idx) % 2;
  state[state_idx] ^= mask;
  return {sign % 2 ? -1 : 1, state};
}

[[nodiscard]] std::pair<int, ManyBodyState::key_type>
annihilate(const ManyBodyState::key_type &in_state, size_t idx) /*noexcept*/ {
  ManyBodyState::key_type state = in_state;
  const size_t num_bits = 8 * sizeof(ManyBodyState::key_type::value_type);
  const size_t state_idx = idx / num_bits;
  const size_t bit_idx = num_bits - 1 - (idx % num_bits);
  const ManyBodyState::key_type::value_type mask =
      static_cast<ManyBodyState::key_type::value_type>(1) << bit_idx;
  if (!(state[state_idx] & mask)) {
    return {0, state};
  }
  int sign = 0;
  for (size_t i = 0; i < state_idx; i++) {
    sign += set_bits(state[i]) % 2;
  }
  state[state_idx] ^= mask;
  sign += set_bits(state[state_idx] >> bit_idx) % 2;
  return {sign % 2 ? -1 : 1, state};
}

ManyBodyOperator::ManyBodyOperator(const std::vector<value_type> &ops)
    : m_ops() {
  m_ops.reserve(ops.size());
  for (const auto &val : ops) {
    insert(val);
  }
}

ManyBodyOperator::ManyBodyOperator(std::vector<value_type> &&ops) : m_ops() {
  m_ops.reserve(ops.size());
  for (auto &&val : ops) {
    insert(std::move(val));
  }
}

ManyBodyOperator::ManyBodyOperator(const OPS_VEC &ops, const SCALAR_VEC &amps)
    : m_ops() {
  m_ops.reserve(ops.size());
  for (size_t i = 0; i < ops.size(); i++) {
    emplace(ops[i], amps[i]);
  }
}
ManyBodyOperator::ManyBodyOperator(OPS_VEC &&ops, SCALAR_VEC &&amps) : m_ops() {
  m_ops.reserve(ops.size());
  for (size_t i = 0; i < ops.size(); i++) {
    emplace(std::move(ops[i]), std::move(amps[i]));
  }
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
ManyBodyOperator::operator[](const key_type &key) noexcept {
  auto it = find(key);
  if (it == m_ops.end() || it->first != key) {
    it = m_ops.emplace(it, key, mapped_type{});
  }
  return it->second;
}

ManyBodyOperator::mapped_type &
ManyBodyOperator::operator[](key_type &&key) noexcept {
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
  return {it, true};
}
std::pair<ManyBodyOperator::iterator, bool>
ManyBodyOperator::insert(value_type &&val) {
  auto it = find(val.first);
  if (it != m_ops.end() && it->first != val.first) {
    return {it, false};
  }
  it = m_ops.emplace(it, std::move(val));
  return {it, true};
}
ManyBodyOperator::iterator ManyBodyOperator::insert(iterator pos,
                                                    const value_type &val) {
  auto it = find(m_ops.begin(), pos, val.first);
  return m_ops.emplace(it, val);
}

ManyBodyOperator::iterator ManyBodyOperator::insert(iterator pos,
                                                    value_type &&val) {
  auto it = find(m_ops.begin(), pos, val.first);
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
  return m_ops.erase(pos);
}

ManyBodyOperator::iterator ManyBodyOperator::erase(const_iterator pos) {
  return m_ops.erase(pos);
}
ManyBodyOperator::iterator ManyBodyOperator::erase(const_iterator first,
                                                   const_iterator last) {
  return m_ops.erase(first, last);
}
ManyBodyOperator::size_type ManyBodyOperator::erase(const key_type &key) {
  auto it = find(key);
  if (it == m_ops.end() || it->first != key) {
    return 0;
  }
  m_ops.erase(it);
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

void ManyBodyOperator::build_restriction_mask(
    const Restrictions &restrictions) noexcept {
  const size_t num_bits = 8 * sizeof(ManyBodyState::key_type::value_type);

  std::vector<ManyBodyState::key_type> masks;
  std::vector<size_t> min_vals;
  std::vector<size_t> max_vals;
  masks.reserve(restrictions.size());
  min_vals.reserve(restrictions.size());
  max_vals.reserve(restrictions.size());
  for (const auto &restriction : restrictions) {
    ManyBodyState::key_type mask;
    ManyBodyState::key_type::value_type current = 0;
    size_t mask_i = 0;
    for (auto idx : restriction.first) {
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
    masks.push_back(mask);
    min_vals.push_back(restriction.second.first);
    max_vals.push_back(restriction.second.second);
  }

  this->m_restrictions_mask =
      std::tuple<std::vector<ManyBodyState::key_type>, std::vector<size_t>,
                 std::vector<size_t>>{masks, min_vals, max_vals};
}

bool ManyBodyOperator::state_is_within_restrictions(
    const ManyBodyState::key_type &state) const noexcept {

  const auto &masks = std::get<0>(m_restrictions_mask);
  const auto &min_vals = std::get<1>(m_restrictions_mask);
  const auto &max_vals = std::get<2>(m_restrictions_mask);
  for (size_t i = 0; i < masks.size(); i++) {
    size_t bit_count = 0;
    for (size_t j = 0; j < masks[i].size(); j++) {
      bit_count += set_bits(state[j] & masks[i][j]);
    }
    if (bit_count < min_vals[i] || bit_count > max_vals[i]) {
      return false;
    }
  }
  return true;
}

ManyBodyState ManyBodyOperator::apply_op_determinant(
    const ManyBodyState::key_type &in_slater_determinant) const noexcept {
  ManyBodyState tmp;
  std::pair<int, ManyBodyState::key_type> ac_res;
  for (auto [indices, coeff] : m_ops) {
    // for (auto op_it = m_ops.cbegin(); op_it != m_ops.cend(); op_it++) {
    ManyBodyState::key_type out_slater_determinant{in_slater_determinant};
    int sign = 1;
    for (const int64_t idx : indices) {
      if (idx >= 0) {
        ac_res = create(out_slater_determinant, static_cast<size_t>(idx));
      } else {
        ac_res =
            annihilate(out_slater_determinant, static_cast<size_t>(-(idx + 1)));
      }
      sign *= ac_res.first;
      out_slater_determinant = std::move(ac_res.second);
      if (sign == 0 || !state_is_within_restrictions(out_slater_determinant)) {
        sign = 0;
        break;
      }
    }
    if (sign != 0) {
      tmp[std::move(out_slater_determinant)] +=
          static_cast<double>(sign) * coeff;
    }
  }
  return tmp;
}

[[nodiscard]] ManyBodyState ManyBodyOperator::apply(const ManyBodyState &state,
                                                    double cutoff) const {
  return std::transform_reduce(
      PAR m_ops.begin(), m_ops.end(), ManyBodyState{},
      [](auto &&a, auto &&b) {
        return std::forward<decltype(a)>(a) += std::forward<decltype(b)>(b);
      },
      [this, &state, cutoff](ManyBodyOperator::const_reference op_amp) {
        const auto &[indices, coeff] = op_amp;
        std::pair<int, ManyBodyState::key_type> ac_res;
        ManyBodyState tmp{};
        tmp.reserve(indices.size() * state.size());
        for (const auto &[slater, amp] : state) {
          ManyBodyState::key_type out_slater_determinant{slater};
          double sign = 1;
          for (const int64_t idx : indices) {
            if (idx >= 0) {
              ac_res = create(out_slater_determinant, static_cast<size_t>(idx));
            } else {
              ac_res = annihilate(out_slater_determinant,
                                  static_cast<size_t>(-(idx + 1)));
            }
            sign *= ac_res.first;
            out_slater_determinant = std::move(ac_res.second);

            if (sign == 0 ||
                !state_is_within_restrictions(out_slater_determinant)) {
              sign = 0;
              break;
            }
          }
          if (sign != 0 && abs(coeff * amp) > cutoff) {
            auto p = tmp.try_emplace(std::move(out_slater_determinant),
                                     sign * coeff * amp);
            if (!p.second) {
              p.first->second += sign * coeff * amp;
            }
          }
        }
        return tmp;
      });
}

ManyBodyOperator &
ManyBodyOperator::operator+=(const ManyBodyOperator &other) noexcept {
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

ManyBodyOperator &ManyBodyOperator::operator*=(const SCALAR &s) noexcept {
  for (auto &p : m_ops) {
    p.second *= s;
  }
  return *this;
}

ManyBodyOperator &ManyBodyOperator::operator/=(const SCALAR &s) noexcept {
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
