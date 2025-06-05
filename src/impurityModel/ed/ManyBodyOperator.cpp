#include "ManyBodyOperator.h"
#include "ManyBodyState.h"
#include <bitset>
#include <cassert>
#include <iostream>

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
    : m_ops(ops), m_memory() {}

ManyBodyOperator::ManyBodyOperator(std::vector<std::pair<OPS, SCALAR>> &&ops)
    : m_ops(std::move(ops)), m_memory() {}

void ManyBodyOperator::add_ops(const std::vector<std::pair<OPS, SCALAR>> &ops) {
  for (const auto &op : ops) {
    m_ops.push_back(op);
  }
}

void ManyBodyOperator::add_ops(std::vector<std::pair<OPS, SCALAR>> &&ops) {
  for (auto &op : ops) {
    m_ops.push_back(std::move(op));
  }
}

ManyBodyState ManyBodyOperator::operator()(const ManyBodyState &state) const {
  ManyBodyState::key_type new_state;
  ManyBodyState res, tmp;
  int sign;
  for (const auto &key_amp : state) {
    // if (m_memory.find(key_amp.first) != m_memory.end()) {
    //   res += key_amp.second * m_memory[key_amp.first];
    //   continue;
    // }
    for (const auto &ops_scalar : m_ops) {
      tmp.clear();
      new_state = key_amp.first;
      for (const int64_t idx : ops_scalar.first) {
        if (idx >= 0) {
          sign = create(new_state, static_cast<size_t>(idx));
        } else {
          sign = annihilate(new_state, static_cast<size_t>(-(idx + 1)));
        }
        if (sign == 0) {
          // m_memory[key_amp.first] = ManyBodyState();
          break;
        }
      }
      tmp[std::move(new_state)] +=
          static_cast<double>(sign) * ops_scalar.second;
    }
    // m_memory[key_amp.first] = tmp;
    tmp *= key_amp.second;
    res += tmp;
  }

  res.prune(0);
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
