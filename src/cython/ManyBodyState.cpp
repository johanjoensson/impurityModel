// #include <bits/utility.h>
#include "ManyBodyState.h"
#include <algorithm>
#include <bits/utility.h>
#include <bitset>
#include <functional>
#include <iterator>
#include <numeric>
#ifdef PARALLEL_STL
#include <execution>
#define STATE_PAR std::execution::par_unseq,
#else // PAR
#define STATE_PAR
#endif // PAR

template <typename K, typename V, typename OP>
boost::container::flat_map<K, V>
merge_flat_maps(boost::container::flat_map<K, V> &&map1,
                const boost::container::flat_map<K, V> &map2, OP &&op) {

  using Sequence = typename boost::container::flat_map<K, V>::sequence_type;
  Sequence data1 = map1.extract_sequence();
  Sequence merged_data;
  merged_data.reserve(data1.size() + map2.size());

  auto it1 = data1.begin();
  auto it2 = map2.begin();
  while (it1 < data1.end() && it2 < map2.end()) {
    if (it1->first < it2->first) {
      merged_data.emplace_back(
          std::move(it1->first),
          std::forward<OP>(op)(std::move(it1->second), V{}));
      it1++;
    } else if (it1->first > it2->first) {
      merged_data.emplace_back(it2->first,
                               std::forward<OP>(op)(V{}, it2->second));
      it2++;
    } else {
      merged_data.emplace_back(
          std::move(it1->first),
          std::forward<OP>(op)(std::move(it1->second), it2->second));
      it1++;
      it2++;
    }
  }
  for (; it1 < data1.end(); it1++) {
    merged_data.emplace_back(std::move(it1->first),
                             std::forward<OP>(op)(std::move(it1->second), V{}));
  }
  for (; it2 < map2.end(); it2++) {
    merged_data.emplace_back(it2->first,
                             std::forward<OP>(op)(V{}, it2->second));
  }
  merged_data.shrink_to_fit();
  boost::container::flat_map<K, V> merged_map;
  merged_map.adopt_sequence(boost::container::ordered_unique_range,
                            std::move(merged_data));
  return merged_map;
}

template <typename K, typename V, typename OP>
boost::container::flat_map<K, V>
merge_flat_maps(boost::container::flat_map<K, V> &&map1,
                boost::container::flat_map<K, V> &&map2, OP &&op) {

  using Sequence = typename boost::container::flat_map<K, V>::sequence_type;
  Sequence data1 = map1.extract_sequence();
  Sequence data2 = map2.extract_sequence();
  Sequence merged_data;
  merged_data.reserve(map1.size() + map2.size());

  auto it1 = data1.begin();
  auto it2 = data2.begin();
  while (it1 < data1.end() && it2 < data2.end()) {
    if (it1->first < it2->first) {
      merged_data.emplace_back(
          std::move(it1->first),
          std::forward<OP>(op)(std::move(it1->second), V{}));
      it1++;
    } else if (it1->first > it2->first) {
      merged_data.emplace_back(
          std::move(it2->first),
          std::forward<OP>(op)(V{}, std::move(it2->second)));
      it2++;
    } else {
      merged_data.emplace_back(
          std::move(it1->first),
          std::forward<OP>(op)(std::move(it1->second), std::move(it2->second)));
      it1++;
      it2++;
    }
  }

  for (; it1 < data1.end(); it1++) {
    merged_data.emplace_back(std::move(it1->first),
                             std::forward<OP>(op)(std::move(it1->second), V{}));
  }
  for (; it2 < data2.end(); it2++) {
    merged_data.emplace_back(std::move(it2->first),
                             std::forward<OP>(op)(V{}, std::move(it2->second)));
  }
  merged_data.shrink_to_fit();
  boost::container::flat_map<K, V> merged_map;
  merged_map.adopt_sequence(boost::container::ordered_unique_range,
                            std::move(merged_data));
  return merged_map;
}

// If we have access to c++23 or later, prefer to use std::flat_map instead of
// boost::container::flat_map
#if __cplusplus >= 202302L
template <typename K, typename V, typename OP>
std::flat_map<K, V> merge_flat_maps(std::flat_map<K, V> &&map1,
                                    std::flat_map<K, V> &&map2, OP &&op) {

  // Extract data from maps
  auto [k1, v1] = std::move(map1).extract();
  auto [k2, v2] = std::move(map2).extract();

  std::vector<K> res_keys;
  std::vector<V> res_values;
  res_keys.reserve(k1.size() + k2.size());
  res_values.reserve(v1.size() + v2.size());

  // Merge by iterating through both maps simultaneously
  size_t i = 0, j = 0;
  while (i < k1.size() && j < k2.size()) {
    if (k1[i] < k2[j]) {
      res_keys.push_back(std::move(k1[i]));
      res_values.push_back(std::forward<OP>(op)(std::move(v1[i]), V{}));
      i++;
    } else if (k2[j] < k1[i]) {
      res_keys.push_back(std::move(k2[j]));
      res_values.push_back(std::forward<OP>(op)(V{}, std::move(v2[j])));
      j++;
    } else {
      res_keys.push_back(std::move(k1[i]));
      res_values.push_back(
          std::forward<OP>(op)(std::move(v1[i]), std::move(v2[j])));
      i++;
      j++;
    }
  }

  // Treat leftover items
  while (i < k1.size()) {
    res_keys.push_back(std::move(k1[i]));
    res_values.push_back(std::forward<OP>(op)(std::move(v1[i]), V{}));
    i++;
  }
  while (j < k2.size()) {
    res_keys.push_back(std::move(k2[j]));
    res_values.push_back(std::forward<OP>(op)(V{}, std::move(v2[j])));
    j++;
  }

  res_keys.shrink_to_fit();
  res_values.shrink_to_fit();

  return ManyBodyState::Map(std::sorted_unique, std::move(res_keys),
                            std::move(res_values));
}

template <typename K, typename V, typename OP>
std::flat_map<K, V> merge_flat_maps(
    std::flat_map<K, V> &&map1,      // Flattened/moved (non-const)
    const std::flat_map<K, V> &map2, // Supports both const and non-const
    OP &&op) {
  // Extract data from map1
  auto [k1, v1] = std::move(map1).extract();

  const auto &k2 = map2.keys();
  const auto &v2 = map2.values();

  std::vector<K> res_keys;
  std::vector<V> res_values;
  res_keys.reserve(k1.size() + k2.size());
  res_values.reserve(v1.size() + v2.size());

  // Merge the two maps by iterating through them simultaneously
  size_t i = 0, j = 0;
  while (i < k1.size() && j < k2.size()) {
    if (k1[i] < k2[j]) {
      res_keys.push_back(std::move(k1[i]));
      res_values.push_back(std::forward<OP>(op)(std::move(v1[i]), V{}));
      i++;
    } else if (k2[j] < k1[i]) {
      res_keys.push_back(k2[j]); // Copy from const map
      res_values.push_back(
          std::forward<OP>(op)(V{}, v2[j])); // Copy from const map
      j++;
    } else { // Duplicate key: Sum values
      res_keys.push_back(std::move(k1[i]));
      res_values.push_back(std::forward<OP>(op)(std::move(v1[i]), v2[j]));
      i++;
      j++;
    }
  }

  // Treat leftover items
  while (i < k1.size()) {
    res_keys.push_back(std::move(k1[i]));
    res_values.push_back(std::forward<OP>(op)(std::move(v1[i]), V{}));
    i++;
  }
  while (j < k2.size()) {
    res_keys.push_back(k2[j]);
    res_values.push_back(std::forward<OP>(op)(V{}, v2[j]));
    j++;
  }

  res_keys.shrink_to_fit();
  res_values.shrink_to_fit();

  return ManyBodyState::Map(std::sorted_unique, std::move(res_keys),
                            std::move(res_values));
}
#endif

ManyBodyState::ManyBodyState(const std::vector<key_type> &keys,
                             const std::vector<mapped_type> &values)
    : m_map() {
  if (!keys.empty()) {
    reserve(keys.size());
    std::vector<size_t> indices(keys.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&keys](size_t i, size_t j) { return keys[i] < keys[j]; });
    for (size_t idx : indices) {
      if (m_map.size() > 0 && m_map.rbegin()->first == keys[idx]) {
        m_map.rbegin()->second += values[idx];
      } else {
        m_map.emplace_hint(m_map.end(), keys[idx], values[idx]);
      }
    }
  }
}

ManyBodyState::ManyBodyState(std::vector<key_type> &&keys,
                             std::vector<mapped_type> &&values)
    : m_map() {
  if (!keys.empty()) {
    reserve(keys.size());
    std::vector<size_t> indices(keys.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&keys](size_t i, size_t j) { return keys[i] < keys[j]; });
    for (size_t idx : indices) {
      if (m_map.size() > 0 && m_map.rbegin()->first == keys[idx]) {
        m_map.rbegin()->second += values[idx];
      } else {
        m_map.emplace_hint(m_map.end(), std::move(keys[idx]), values[idx]);
      }
    }
  }
}

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
  for (auto &&[k, v] : m_map) {
    v *= s;
  }
  return *this;
}
ManyBodyState &ManyBodyState::operator/=(mapped_type s) {
  for (auto &&[k, v] : m_map) {
    v /= s;
  }
  return *this;
}

ManyBodyState ManyBodyState::operator-() const {
  ManyBodyState res(*this);
  for (auto &&[k, v] : res.m_map) {
    v = -v;
  }
  return res;
}

std::complex<double> inner(const ManyBodyState &a, const ManyBodyState &b) {
  std::complex<double> res = 0;

  auto i = a.begin(), j = b.begin();
  while (i < a.end() && j < b.end()) {
    auto [ka, va] = *i;
    auto [kb, vb] = *j;
    if (ka < kb) {
      i++;
    } else if (ka > kb) {
      j++;
    } else {
      res += conj(va) * vb;
      i++;
      j++;
    }
  }

  return res;
}

ManyBodyState &ManyBodyState::prune(double cutoff) {
#if __cplusplus >= 202302L
  erase_if(m_map, [cutoff](ManyBodyState::const_reference pair) {
    return abs(pair.second) <= cutoff;
  });
#else
  auto new_end = std::remove_if(m_map.begin(), m_map.end(),
                                [cutoff](ManyBodyState::const_reference pair) {
                                  return abs(pair.second) <= cutoff;
                                });
  m_map.erase(new_end, m_map.end());
#endif
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
