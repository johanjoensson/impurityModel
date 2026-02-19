#ifdef PARALLEL_STL
#include <execution>
#define PAR std::execution::par,
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

[[nodiscard]] double create(ManyBodyState::key_type &state,
                            size_t idx) noexcept {
  const size_t num_bits{8 * sizeof(ManyBodyState::key_type::value_type)};
  const size_t state_idx = idx / num_bits;
  const size_t bit_idx = num_bits - 1 - (idx % num_bits);
  const ManyBodyState::key_type::value_type mask =
      static_cast<ManyBodyState::key_type::value_type>(1) << bit_idx;
  if (state[state_idx] & mask) {
    return 0;
  }
  size_t sign = 0;
  for (size_t i = 0; i < state_idx; i++) {
    sign += set_bits(state[i]) % 2;
  }
  sign += set_bits(state[state_idx] >> bit_idx) % 2;
  state[state_idx] ^= mask;
  return sign % 2 ? -1 : 1;
}

[[nodiscard]] double annihilate(ManyBodyState::key_type &state,
                                size_t idx) noexcept {
  const size_t num_bits = 8 * sizeof(ManyBodyState::key_type::value_type);
  const size_t state_idx = idx / num_bits;
  const size_t bit_idx = num_bits - 1 - (idx % num_bits);
  const ManyBodyState::key_type::value_type mask =
      static_cast<ManyBodyState::key_type::value_type>(1) << bit_idx;
  if (!(state[state_idx] & mask)) {
    return 0;
  }
  int sign = 0;
  for (size_t i = 0; i < state_idx; i++) {
    sign += set_bits(state[i]) % 2;
  }
  state[state_idx] ^= mask;
  sign += set_bits(state[state_idx] >> bit_idx) % 2;
  return sign % 2 ? -1 : 1;
}

ManyBodyOperator::ManyBodyOperator(const std::vector<value_type> &ops)
    : m_ops() {
  m_ops.reserve(ops.size());
  for (const auto &p : ops) {
    insert(p);
  }
}

ManyBodyOperator::ManyBodyOperator(std::vector<value_type> &&ops) : m_ops() {
  m_ops.reserve(ops.size());
  for (auto &p : ops) {
    insert(std::move(p));
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
  return std::lower_bound(first, last, key,
                          [](const value_type &a, const key_type &b) {
                            return compare_type{}(a.first, b);
                          });
}

ManyBodyOperator::iterator ManyBodyOperator::find(const key_type &key) {
  return find(m_ops.begin(), m_ops.end(), key);
}

ManyBodyOperator::const_iterator
ManyBodyOperator::find(const_iterator first, const_iterator last,
                       const key_type &key) const {
  return std::lower_bound(first, last, key,
                          [](const value_type &a, const key_type &b) {
                            return compare_type{}(a.first, b);
                          });
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
    it = m_ops.emplace(it, key, mapped_type());
  }
  return it->second;
}

ManyBodyOperator::mapped_type &
ManyBodyOperator::operator[](key_type &&key) noexcept {
  auto it = find(key);
  if (it == m_ops.end() || it->first != key) {
    it = m_ops.emplace(it, std::move(key), mapped_type());
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
  for (auto &&val : l) {
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
  return std::lower_bound(m_ops.begin(), m_ops.end(), key,
                          [](const value_type &a, const key_type &b) {
                            return compare_type()(a.first, b);
                          });
}
ManyBodyOperator::const_iterator
ManyBodyOperator::lower_bound(const key_type &key) const {
  return std::lower_bound(m_ops.cbegin(), m_ops.cend(), key,
                          [](const value_type &a, const key_type &b) {
                            return compare_type()(a.first, b);
                          });
}
template <class K>
ManyBodyOperator::iterator ManyBodyOperator::lower_bound(const K &key) {
  return std::lower_bound(m_ops.begin(), m_ops.end(), key,
                          [](const value_type &a, const key_type &b) {
                            return compare_type()(a.first, b);
                          });
}
template <class K>
ManyBodyOperator::const_iterator
ManyBodyOperator::lower_bound(const K &key) const {
  return std::lower_bound(m_ops.cbegin(), m_ops.cend(), key,
                          [](const value_type &a, const key_type &b) {
                            return compare_type()(a.first, b);
                          });
}

ManyBodyOperator::iterator ManyBodyOperator::upper_bound(const key_type &key) {
  return std::upper_bound(m_ops.begin(), m_ops.end(), key,
                          [](const key_type &a, const value_type &b) {
                            return compare_type()(a, b.first);
                          });
}
ManyBodyOperator::const_iterator
ManyBodyOperator::upper_bound(const key_type &key) const {
  return std::upper_bound(m_ops.cbegin(), m_ops.cend(), key,
                          [](const key_type &a, const value_type &b) {
                            return compare_type()(a, b.first);
                          });
}
template <class K>
ManyBodyOperator::iterator ManyBodyOperator::upper_bound(const K &key) {
  return std::upper_bound(m_ops.begin(), m_ops.end(), key,
                          [](const key_type &a, const value_type &b) {
                            return compare_type()(a, b.first);
                          });
}
template <class K>
ManyBodyOperator::const_iterator
ManyBodyOperator::upper_bound(const K &key) const {
  return std::upper_bound(m_ops.cbegin(), m_ops.cend(), key,
                          [](const key_type &a, const value_type &b) {
                            return compare_type()(a, b.first);
                          });
}

[[nodiscard]] ManyBodyOperator::size_type
ManyBodyOperator::size() const noexcept {
  return m_ops.size();
}

std::tuple<std::vector<ManyBodyState::key_type>, std::vector<size_t>,
           std::vector<size_t>>
ManyBodyOperator::build_restriction_mask(
    const Restrictions &restrictions) const noexcept {
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

  return std::tuple<std::vector<ManyBodyState::key_type>, std::vector<size_t>,
                    std::vector<size_t>>{masks, min_vals, max_vals};
}

bool ManyBodyOperator::state_is_within_restrictions(
    const ManyBodyState::key_type &state,
    const std::tuple<std::vector<ManyBodyState::key_type>, std::vector<size_t>,
                     std::vector<size_t>> &restrictions) noexcept {

  const auto &masks = std::get<0>(restrictions);
  const auto &min_vals = std::get<1>(restrictions);
  const auto &max_vals = std::get<2>(restrictions);
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
    const ManyBodyState::key_type &in_slater_determinant,
    const std::tuple<std::vector<ManyBodyState::key_type>, std::vector<size_t>,
                     std::vector<size_t>> &restrictions) const noexcept {
  // std::chrono::microseconds t_create{0}, t_annihilate{0}, t_increment{0},
  //     t_copy{0};
  ManyBodyState tmp;
  // auto t0 = std::chrono::high_resolution_clock::now();
  // auto t_total = std::chrono::high_resolution_clock::now();
  for (auto op_it = m_ops.cbegin(); op_it != m_ops.cend(); op_it++) {
    // t0 = std::chrono::high_resolution_clock::now();
    ManyBodyState::key_type out_slater_determinant{in_slater_determinant};
    // t_copy += (std::chrono::duration_cast<std::chrono::microseconds>(
    //     std::chrono::high_resolution_clock::now() - t0));
    int sign = 1;
    for (const int64_t idx : (*op_it).first) {
      // auto t0 = std::chrono::high_resolution_clock::now();
      if (idx >= 0) {
        sign *= create(out_slater_determinant, static_cast<size_t>(idx));
        // t_create += (std::chrono::duration_cast<std::chrono::microseconds>(
        //     std::chrono::high_resolution_clock::now() - t0));
      } else {
        sign *=
            annihilate(out_slater_determinant, static_cast<size_t>(-(idx + 1)));
        // t_annihilate +=
        // (std::chrono::duration_cast<std::chrono::microseconds>(
        //     std::chrono::high_resolution_clock::now() - t0));
      }
      if (sign == 0 ||
          !state_is_within_restrictions(out_slater_determinant, restrictions)) {
        sign = 0;
        break;
      }
    }
    // t0 = std::chrono::high_resolution_clock::now();
    if (sign != 0) {
      tmp[out_slater_determinant] +=
          static_cast<double>(sign) * (*op_it).second;
    }
    // t_increment += (std::chrono::duration_cast<std::chrono::microseconds>(
    //     std::chrono::high_resolution_clock::now() - t0));
  }
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
  //     std::chrono::high_resolution_clock::now() - t_total);
  // if (duration.count() > 1000) {
  //   std::cout << "apply_op_determinant took " << duration.count()
  //             << " micros\n";
  //   std::cout << "---> copying slater_in took " << t_copy.count()
  //             << " micros\n";
  //   std::cout << "---> create took " << t_create.count() << " micros\n";
  //   std::cout << "---> annihilate took " << t_annihilate.count() << "
  //   micros\n"; std::cout << "---> incrementing psi took " <<
  //   t_increment.count()
  //             << " micros\n";
  // }
  return tmp;
}

[[nodiscard]] ManyBodyState ManyBodyOperator::apply(
    const ManyBodyState &state, double cutoff,
    const ManyBodyOperator::Restrictions &restrictions) const noexcept {
  const auto restriction_mask = build_restriction_mask(restrictions);
  // ManyBodyState initial{};
  // initial.reserve(state.size());
  ManyBodyState res = std::transform_reduce(
      PAR state.cbegin(), state.cend(), ManyBodyState{},
      [](auto &&a, const auto &b) { return std::forward<decltype(a)>(a += b); },
      [this, &restriction_mask,
       cutoff](ManyBodyState::const_reference state_amp) {
        ManyBodyState tmp{};
        for (auto op_it = m_ops.cbegin(); op_it != m_ops.cend(); op_it++) {
          ManyBodyState::key_type out_slater_determinant{state_amp.first};
          double sign = 1;
          for (const int64_t idx : (*op_it).first) {
            if (idx >= 0) {
              sign *= create(out_slater_determinant, static_cast<size_t>(idx));
            } else {
              sign *= annihilate(out_slater_determinant,
                                 static_cast<size_t>(-(idx + 1)));
            }

            if (sign == 0 || !state_is_within_restrictions(
                                 out_slater_determinant, restriction_mask)) {
              sign = 0;
              break;
            }
          }
          if (abs(sign * op_it->second * state_amp.second) > cutoff) {
            tmp[std::move(out_slater_determinant)] +=
                sign * op_it->second * state_amp.second;
          }
        }
        return tmp;
      });

  // res.prune(cutoff);
  return res;
}

[[nodiscard]] std::vector<ManyBodyState> ManyBodyOperator::apply(
    const std::vector<ManyBodyState> &psis, double cutoff,
    const ManyBodyOperator::Restrictions &restrictions) const noexcept {
  const auto restriction_mask = build_restriction_mask(restrictions);
  std::vector<ManyBodyState> res;
  res.reserve(psis.size());
  for (const ManyBodyState &psi : psis) {
    res.push_back(std::transform_reduce(
        PAR psi.begin(), psi.end(), ManyBodyState{},
        [](auto &&a, const ManyBodyState &b) {
          return std::forward<ManyBodyState>(a += b);
        },
        [&](ManyBodyState::const_reference state_it) {
          ManyBodyState tmp;
          for (auto op_it = m_ops.cbegin(); op_it != m_ops.cend(); op_it++) {
            ManyBodyState::key_type out_slater_determinant{state_it.first};
            double sign = 1;
            for (const int64_t idx : (*op_it).first) {
              if (idx >= 0) {
                sign *=
                    create(out_slater_determinant, static_cast<size_t>(idx));
              } else {
                sign *= annihilate(out_slater_determinant,
                                   static_cast<size_t>(-(idx + 1)));
              }
              if (sign == 0 || !state_is_within_restrictions(
                                   out_slater_determinant, restriction_mask)) {
                sign = 0;
                break;
              }
            }
            if (sign != 0 && abs(op_it->second * state_it.second) > cutoff) {
              tmp[out_slater_determinant] +=
                  static_cast<double>(sign) * op_it->second * state_it.second;
            }
          }
          return tmp;
        }));
  }
  // for (auto &psi : res) {
  //   psi.prune(cutoff);
  // }

  return res;
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
