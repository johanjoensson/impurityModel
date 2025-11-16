#include "ManyBodyOperator.h"
#include "ManyBodyState.h"
#include <algorithm>
#include <cassert>
#if __cplusplus >= 202002L
#include <bit>
#else
#include <bitset>
#include <climits>
#endif

inline size_t set_bits(ManyBodyState::key_type::value_type byte) {
#if __cplusplus >= 202002L
  return std::popcount(byte);
#else
  return std::bitset<sizeof(ManyBodyState::key_type::value_type) * CHAR_BIT>(
             byte)
      .count();
#endif
}

int create(ManyBodyState::key_type &state, size_t idx) {
  const size_t num_bits = 8 * sizeof(ManyBodyState::key_type::value_type);
  const size_t state_idx = idx / num_bits;
  const size_t bit_idx = num_bits - 1 - (idx % num_bits);
  const ManyBodyState::key_type::value_type mask = 1 << bit_idx;
  if (state[state_idx] & mask) {
    return 0;
  }
  int sign = 1;
  state[state_idx] |= mask;
  for (size_t i = 0; i < state_idx; i++) {
    sign *= set_bits(state[i]) % 2 ? -1 : 1;
  }
  return sign * (set_bits(state[state_idx] >> (bit_idx + 1)) % 2 ? -1 : 1);
}

int annihilate(ManyBodyState::key_type &state, size_t idx) {
  const size_t num_bits = 8 * sizeof(ManyBodyState::key_type::value_type);
  const size_t state_idx = idx / num_bits;
  const size_t bit_idx = num_bits - 1 - (idx % num_bits);
  const ManyBodyState::key_type::value_type mask = 1 << bit_idx;
  if (!(state[state_idx] & mask)) {
    return 0;
  }
  int sign = 1;
  state[state_idx] &= ~mask;
  for (size_t i = 0; i < state_idx; i++) {
    sign *= set_bits(state[i]) % 2 ? -1 : 1;
  }
  return sign * (set_bits(state[state_idx] >> (bit_idx + 1)) % 2 ? -1 : 1);
}

ManyBodyOperator::ManyBodyOperator(const std::vector<value_type> &ops)
    : m_ops(), m_memory() {
  m_ops.reserve(ops.size());
  for (const auto &p : ops) {
    insert(p);
  }
}

ManyBodyOperator::ManyBodyOperator(std::vector<value_type> &&ops)
    : m_ops(), m_memory() {
  m_ops.reserve(ops.size());
  for (auto &p : ops) {
    insert(std::move(p));
  }
}

ManyBodyOperator::ManyBodyOperator(const OPS_VEC &ops, const SCALAR_VEC &amps)
    : m_ops(), m_memory() {
  m_ops.reserve(ops.size());
  for (size_t i = 0; i < ops.size(); i++) {
    emplace(ops[i], amps[i]);
  }
}
ManyBodyOperator::ManyBodyOperator(OPS_VEC &&ops, SCALAR_VEC &&amps)
    : m_ops(), m_memory() {
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
ManyBodyOperator::operator[](const key_type &key) {
  auto it = find(key);
  if (it == m_ops.end() || it->first != key) {
    it = m_ops.emplace(it, key, mapped_type());
  }
  return it->second;
}

ManyBodyOperator::mapped_type &ManyBodyOperator::operator[](key_type &&key) {
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

ManyBodyOperator::size_type ManyBodyOperator::size() const {
  return m_ops.size();
}

void ManyBodyOperator::clear_memory() noexcept { m_memory.clear(); }

bool ManyBodyOperator::state_is_within_restrictions(
    const ManyBodyState::key_type &state, const Restrictions &restrictions) {
  const size_t num_bits = 8 * sizeof(ManyBodyState::key_type::value_type);
  for (const auto &p : restrictions) {
    size_t bit_count = 0;
    const Restrictions::value_type::first_type indices = p.first;
    const Restrictions::value_type::second_type limits = p.second;
    ManyBodyState::key_type mask(state.size(), 0);
    for (auto idx : indices) {
      const auto state_idx = idx / num_bits;
      const auto bit_idx = num_bits - 1 - (idx % num_bits);
      mask[state_idx] |= (1 << bit_idx);
    }
    for (size_t i = 0; i < state.size(); i++) {
      bit_count += set_bits(state[i] & mask[i]);
    }
    if (bit_count < limits.first || bit_count > limits.second) {
      return false;
    }
  }
  return true;
}

ManyBodyState ManyBodyOperator::operator()(
    const ManyBodyState &state, double cutoff = 0,
    const ManyBodyOperator::Restrictions &restrictions = {}) {
  ManyBodyState::key_type new_state;
  ManyBodyState res, tmp;
  int sign = 0;
  for (const auto &key_amp : state) {
    if (m_memory.find(key_amp.first) != m_memory.end()) {
      res += key_amp.second * m_memory[key_amp.first];
      continue;
    }
    ManyBodyState::Map tmp;
#pragma omp parallel
    {
      ManyBodyState::Map tmp_local;
      ManyBodyOperator::Memory local_memory;
#pragma omp for schedule(dynamic)
      // Use iterator loop construct for OpenMP parallelization
      for (auto it = m_ops.cbegin(); it != m_ops.cend(); it++) {
        const auto &ops_scalar = *it;
        new_state = key_amp.first;
        sign = 1;
        for (const int64_t idx : ops_scalar.first) {
          if (idx >= 0) {
            sign *= create(new_state, static_cast<size_t>(idx));
          } else {
            sign *= annihilate(new_state, static_cast<size_t>(-(idx + 1)));
          }
          if (sign == 0) {
            break;
          }
        }
        if (sign == 0) {
          local_memory.insert({key_amp.first, ManyBodyState()});
          continue;
        }
        if (!state_is_within_restrictions(new_state, restrictions)) {
          continue;
        }
        tmp_local[new_state] += static_cast<double>(sign) * ops_scalar.second;
      }
#pragma omp critical
      for (const auto &state_amp : tmp_local) {
        tmp[state_amp.first] += state_amp.second;
      }
      for (const auto &state_res : local_memory) {
        m_memory[state_res.first] += state_res.second;
      }
    }
    m_memory[key_amp.first] += ManyBodyState(tmp);
    res += ManyBodyState(tmp) * key_amp.second;
  }

  res.prune(cutoff);
  return res;
}

ManyBodyOperator::Memory ManyBodyOperator::memory() const { return m_memory; }

ManyBodyOperator &ManyBodyOperator::operator+=(const ManyBodyOperator &other) {
  auto current = m_ops.begin();
  for (const auto &op : other.m_ops) {
    current = find(current, m_ops.end(), op.first);
    if (current->first == op.first) {
      current->second += op.second;
    } else {
      current = m_ops.emplace(current, op);
    }
  }

  Memory new_mem;
  auto left_it = this->m_memory.cbegin(), right_it = other.m_memory.cbegin();
  while (left_it != this->m_memory.cend() &&
         right_it != other.m_memory.cend()) {
    if (left_it->first < right_it->first) {
      left_it++;
    } else if (right_it->first < left_it->first) {
      right_it++;
    } else {
      new_mem.emplace(left_it->first, left_it->second + right_it->second);
    }
  }
  this->m_memory = std::move(new_mem);

  return *this;
}

ManyBodyOperator &ManyBodyOperator::operator-=(const ManyBodyOperator &other) {
  auto current = m_ops.begin();
  for (const auto &op : other.m_ops) {
    current = find(current, m_ops.end(), op.first);
    if (current->first == op.first) {
      current->second -= op.second;
    } else {
      current = m_ops.emplace(current, op.first, -op.second);
    }
  }
  Memory new_mem;
  auto left_it = this->m_memory.cbegin(), right_it = other.m_memory.cbegin();
  while (left_it != this->m_memory.cend() &&
         right_it != other.m_memory.cend()) {
    if (left_it->first < right_it->first) {
      left_it++;
    } else if (right_it->first < left_it->first) {
      right_it++;
    } else {
      new_mem.emplace(left_it->first, left_it->second - right_it->second);
    }
  }
  this->m_memory = std::move(new_mem);
  return *this;
}

ManyBodyOperator &ManyBodyOperator::operator*=(const SCALAR &s) {
  for (auto &p : m_ops) {
    p.second *= s;
  }
  for (auto &p : m_memory) {
    p.second *= s;
  }
  return *this;
}

ManyBodyOperator &ManyBodyOperator::operator/=(const SCALAR &s) {
  for (auto &p : m_ops) {
    p.second /= s;
  }
  for (auto &p : m_memory) {
    p.second /= s;
  }
  return *this;
}

ManyBodyOperator ManyBodyOperator::operator-() const {
  ManyBodyOperator res(*this);
  for (auto &p : res.m_ops) {
    p.second = -p.second;
  }
  for (auto &p : res.m_memory) {
    p.second = -p.second;
  }
  return res;
}
