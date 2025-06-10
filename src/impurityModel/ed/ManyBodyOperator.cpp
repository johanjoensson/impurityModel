#include "ManyBodyOperator.h"
#include "ManyBodyState.h"
#include <bitset>
#include <cassert>

inline size_t set_bits(const ManyBodyState::key_type::value_type &byte) {
  const std::bitset<8 * sizeof(ManyBodyState::key_type::value_type)> bits(byte);
  return bits.count();
}
inline size_t set_bits(ManyBodyState::key_type::value_type &&byte) {
  const std::bitset<8 * sizeof(ManyBodyState::key_type::value_type)> bits(
      std::move(byte));
  return bits.count();
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

ManyBodyOperator::ManyBodyOperator(
    const std::vector<std::pair<OPS, SCALAR>> &ops)
    : m_ops(), m_memory() {
  for (const auto &p : ops) {
    m_ops.insert(p);
  }
}

ManyBodyOperator::ManyBodyOperator(std::vector<std::pair<OPS, SCALAR>> &&ops)
    : m_ops(), m_memory() {
  for (auto &p : ops) {
    m_ops.insert(std::move(p));
  }
}

ManyBodyOperator::ManyBodyOperator(const Map &m) : m_ops(m), m_memory() {}
ManyBodyOperator::ManyBodyOperator(Map &&m) : m_ops(std::move(m)), m_memory() {}

void ManyBodyOperator::add_ops(const std::vector<std::pair<OPS, SCALAR>> &ops) {
  for (const auto &op : ops) {
    m_ops[op.first] += op.second;
  }
}

void ManyBodyOperator::add_ops(std::vector<std::pair<OPS, SCALAR>> &&ops) {
  for (auto &op : ops) {
    m_ops[std::move(op.first)] += std::move(op.second);
  }
}

ManyBodyOperator::Map::size_type ManyBodyOperator::size() const {
  return m_ops.size();
}

void ManyBodyOperator::clear_memory() noexcept { m_memory.clear(); }

bool ManyBodyOperator::state_is_within_restrictions(
    const ManyBodyState::key_type &state, const Restrictions &restrictions) {
  const size_t num_bits = 8 * sizeof(ManyBodyState::key_type::value_type);
  for (const auto &p : restrictions) {
    size_t bit_count = 0;
    const Restrictions::key_type indices = p.first;
    const Restrictions::mapped_type limits = p.second;
    ManyBodyState::key_type mask(state.size(), 0);
    for (auto idx : indices) {
      auto state_idx = idx / num_bits;
      auto bit_idx = num_bits - 1 - (idx % num_bits);
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

ManyBodyState
ManyBodyOperator::operator()(const ManyBodyState &state, double cutoff = 0,
                             const Restrictions &restrictions = {}) {
  ManyBodyState::key_type new_state;
  ManyBodyState res, tmp;
  int sign = 0;
  for (const auto &key_amp : state) {
    if (m_memory.find(key_amp.first) != m_memory.end()) {
      res += key_amp.second * m_memory[key_amp.first];
      continue;
    }
    tmp.clear();
    for (const auto &ops_scalar : m_ops) {
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
        m_memory.insert({key_amp.first, ManyBodyState()});
        continue;
      }
      if (!state_is_within_restrictions(new_state, restrictions)) {
        continue;
      }
      tmp[new_state] += static_cast<double>(sign) * ops_scalar.second;
    }
    m_memory[key_amp.first] += tmp;
    tmp *= key_amp.second;
    res += tmp;
  }

  res.prune(cutoff);
  return res;
}

ManyBodyOperator::Memory ManyBodyOperator::memory() const { return m_memory; }

ManyBodyOperator &ManyBodyOperator::operator+=(const ManyBodyOperator &other) {

  for (const auto &p : other.m_ops) {
    m_ops[p.first] += p.second;
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
      new_mem.insert({left_it->first, left_it->second + right_it->second});
    }
  }
  this->m_memory = std::move(new_mem);

  return *this;
}

ManyBodyOperator &ManyBodyOperator::operator-=(const ManyBodyOperator &other) {
  for (const auto &p : other.m_ops) {
    m_ops[p.first] -= p.second;
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
      new_mem.insert({left_it->first, left_it->second - right_it->second});
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

int main() {

  const std::vector<uint8_t> state_orig{0xFF, 0x80};
  auto state = state_orig;
  int sign = create(state, 1);
  assert(sign == 0);
  assert(state == state_orig);
  std::vector<uint8_t> state_res{0xFF, 0xA0};
  sign = create(state, 10);
  assert(sign == -1);
  assert(state == state_res);
  state = state_orig;
  state_res = {0xFF, 0x81};
  sign = create(state, 15);
  assert(sign == -1);
  assert(state == state_res);
  state = state_orig;
  state_res = {0xFF, 0x00};
  sign = annihilate(state, 8);
  assert(sign == 1);
  assert(state == state_res);
  state = state_orig;
  state_res = {0b11111110, 0x80};
  sign = annihilate(state, 7);
  assert(sign == -1);
  assert(state == state_res);
  state = state_orig;
  state_res = state_orig;
  sign = annihilate(state, 9);
  assert(sign == 0);
  assert(state == state_res);

  std::vector<int64_t> indices({0, 1, 2, -1, -2, -3});
  std::complex<double> scalar(1, 0);
  std::vector<std::pair<std::vector<int64_t>, std::complex<double>>> ops(
      {std::pair<std::vector<int64_t>, std::complex<double>>(
          std::move(indices), std::move(scalar))});
  ManyBodyOperator op(std::move(ops));
  std::vector<std::vector<uint8_t>> keys{{1}, {6}};
  std::vector<std::complex<double>> amps{{1, 0}, {0, 0.5}};
  ManyBodyState psi(std::move(keys), std::move(amps));
  ManyBodyState psi_new = op(psi);
  return 0;
}
