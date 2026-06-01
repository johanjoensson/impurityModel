// #include <bits/utility.h>
#include "ManyBodyState.h"
#include <bits/utility.h>
#include <bitset>
#include <functional>
#include <numeric>
#ifdef PARALLEL_STL
#include <execution>
#define STATE_PAR std::execution::par_unseq,
#else // PAR
#define STATE_PAR
#endif // PAR

#ifndef SORTED_UNIQUE
#define SORTED_UNIQUE std::sorted_unique,
#endif

ManyBodyState::Map merge_flat_maps(ManyBodyState::Map &&map1,
                                   ManyBodyState::Map &&map2, auto &&op) {

  using K = ManyBodyState::Key;
  using V = ManyBodyState::Value;
  // 1. Extract the underlying sorted vectors
  auto [k1, v1] = std::move(map1).extract();
  auto [k2, v2] = std::move(map2).extract();

  // 2. Pre-allocate maximum possible size
  std::vector<K> res_keys;
  std::vector<V> res_values;
  res_keys.reserve(k1.size() + k2.size());
  res_values.reserve(v1.size() + v2.size());

  size_t i = 0, j = 0;

  // 3. Two-pointer linear merge
  while (i < k1.size() && j < k2.size()) {
    if (k1[i] < k2[j]) {
      res_keys.push_back(std::move(k1[i]));
      res_values.push_back(op(std::move(v1[i]), V{}));
      i++;
    } else if (k2[j] < k1[i]) {
      res_keys.push_back(std::move(k2[j]));
      res_values.push_back(op(V{}, std::move(v2[j])));
      j++;
    } else { // Duplicate key found: Sum the values
      res_keys.push_back(std::move(k1[i]));
      res_values.push_back(op(std::move(v1[i]), std::move(v2[j])));
      i++;
      j++;
    }
  }

  // 4. Append remaining elements
  while (i < k1.size()) {
    res_keys.push_back(std::move(k1[i]));
    res_values.push_back(op(std::move(v1[i]), V{}));
    i++;
  }
  while (j < k2.size()) {
    res_keys.push_back(std::move(k2[j]));
    res_values.push_back(op(V{}, std::move(v2[j])));
    j++;
  }

  // Shrink to fit actual size after duplicates are removed
  res_keys.shrink_to_fit();
  res_values.shrink_to_fit();

  // 5. Rebuild flat_map using the sorted_unique tag (guaranteed sorted and
  // unique)
  return ManyBodyState::Map(std::sorted_unique, std::move(res_keys),
                            std::move(res_values));
}

ManyBodyState::Map merge_flat_maps(
    ManyBodyState::Map &&map1,      // Flattened/moved (non-const)
    const ManyBodyState::Map &map2, // Supports both const and non-const
    auto &&op) {
  using K = ManyBodyState::Key;
  using V = ManyBodyState::Value;
  // 1. Extract underlying vectors from the first map (zero-copy)
  auto [k1, v1] = std::move(map1).extract();

  // 2. Access underlying containers of the const map via keys() and values()
  const auto &k2 = map2.keys();
  const auto &v2 = map2.values();

  // 3. Pre-allocate maximum possible size
  std::vector<K> res_keys;
  std::vector<V> res_values;
  res_keys.reserve(k1.size() + k2.size());
  res_values.reserve(v1.size() + v2.size());

  size_t i = 0, j = 0;

  // 4. Two-pointer linear merge
  while (i < k1.size() && j < k2.size()) {
    if (k1[i] < k2[j]) {
      res_keys.push_back(std::move(k1[i]));
      res_values.push_back(op(std::move(v1[i]), V{}));
      i++;
    } else if (k2[j] < k1[i]) {
      res_keys.push_back(k2[j]);            // Copy from const map
      res_values.push_back(op(V{}, v2[j])); // Copy from const map
      j++;
    } else { // Duplicate key: Sum values
      res_keys.push_back(std::move(k1[i]));
      res_values.push_back(op(std::move(v1[i]), v2[j]));
      i++;
      j++;
    }
  }

  // 5. Append remaining elements
  while (i < k1.size()) {
    res_keys.push_back(std::move(k1[i]));
    res_values.push_back(op(std::move(v1[i]), V{}));
    i++;
  }
  while (j < k2.size()) {
    res_keys.push_back(k2[j]);
    res_values.push_back(op(V{}, v2[j]));
    j++;
  }

  res_keys.shrink_to_fit();
  res_values.shrink_to_fit();

  // 6. Rebuild using sorted_unique
  return ManyBodyState::Map(std::sorted_unique, std::move(res_keys),
                            std::move(res_values));
}

ManyBodyState::ManyBodyState(const std::vector<key_type> &keys,
                             const std::vector<mapped_type> &values)
    : m_map(keys, values) {}

ManyBodyState::ManyBodyState(std::vector<key_type> &&keys,
                             std::vector<mapped_type> &&values)
    : m_map(std::move(keys), std::move(values)) {}

double ManyBodyState::norm2() const {
  return std::transform_reduce(
      STATE_PAR this->cbegin(), this->cend(), 0.,
      [](double a, double b) { return a + b; },
      [](const_reference a) { return std::pow(abs(a.second), 2); });
}

double ManyBodyState::norm() const { return sqrt(norm2()); }

ManyBodyState &ManyBodyState::operator+=(ManyBodyState &&other) {
  m_map = merge_flat_maps(std::move(this->m_map), std::move(other.m_map),
                          std::plus());
  return *this;
}

ManyBodyState &ManyBodyState::operator+=(const ManyBodyState &other) {
  m_map = merge_flat_maps(std::move(this->m_map), other.m_map, std::plus());
  return *this;
}

ManyBodyState &ManyBodyState::operator-=(ManyBodyState &&other) {
  m_map = merge_flat_maps(std::move(this->m_map), std::move(other.m_map),
                          std::minus());
  return *this;
}

ManyBodyState &ManyBodyState::operator-=(const ManyBodyState &other) {
  m_map = merge_flat_maps(std::move(this->m_map), other.m_map, std::minus());
  return *this;
}

ManyBodyState &ManyBodyState::operator*=(mapped_type s) {
  for (reference p : *this) {
    p.second *= s;
  }
  return *this;
}
ManyBodyState &ManyBodyState::operator/=(mapped_type s) {
  for (reference p : *this) {
    p.second /= s;
  }
  return *this;
}

ManyBodyState ManyBodyState::operator-() const {
  ManyBodyState res(*this);
  for (reference p : res) {
    p.second = -p.second;
  }
  return res;
}

std::complex<double> inner(const ManyBodyState &a, const ManyBodyState &b) {
  std::complex<double> res = 0;

  const auto &ka = a.m_map.keys();
  const auto &va = a.m_map.values();
  const auto &kb = b.m_map.keys();
  const auto &vb = b.m_map.values();

  size_t i = 0, j = 0;
  while (i < ka.size() && j < kb.size()) {
    if (ka[i] < kb[j]) {
      i++;
    } else if (ka[i] > kb[j]) {
      j++;
    } else {
      res += conj(va[i]) * vb[j];
      i++;
      j++;
    }
  }

  return res;
}

ManyBodyState &ManyBodyState::prune(double cutoff) {
  std::erase_if(m_map, [cutoff](ManyBodyState::const_reference pair) {
    return abs(pair.second) <= cutoff;
  });
  return *this;
}

std::string ManyBodyState::to_string() const {
  std::string res;
  res += "[";
  for (const auto &key_amp : *this) {
    res += "(" + std::to_string(key_amp.second.real()) + " + " +
           std::to_string(key_amp.second.imag()) + "i) " + "|";
    for (const auto &k : key_amp.first) {
      res += (std::bitset<sizeof(ManyBodyState::key_type::value_type)>(k))
                 .to_string() +
             " ";
    }
    res += ">, ";
  }
  return res + "]";
}
