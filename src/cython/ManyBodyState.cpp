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

#if not(defined SORTED_UNIQUE) && __cplusplus >= 202302L
#define SORTED_UNIQUE std::sorted_unique,
#else
#define SORTED_UNIQUE boost::container::ordered_unique_range_t(),
#endif

#if __cplusplus >= 202302L
template <typename OP>
ManyBodyState::Map merge_flat_maps(ManyBodyState::Map &&map1,
                                   ManyBodyState::Map &&map2, OP &&op) {

  using K = ManyBodyState::Key;
  using V = ManyBodyState::Value;
  // 1. Extract the underlying sorted vectors
  auto [k1, v1] = std::move(map1).extract();
  auto k2, v2 = std::move(map2).extract();

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
  return ManyBodyState::Map(SORTED_UNIQUE std::move(res_keys),
                            std::move(res_values));
}

template <typename OP>
ManyBodyState::Map merge_flat_maps(
    ManyBodyState::Map &&map1,      // Flattened/moved (non-const)
    const ManyBodyState::Map &map2, // Supports both const and non-const
    OP &&op) {
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
  return ManyBodyState::Map(SORTED_UNIQUE std::move(res_keys),
                            std::move(res_values));
}

ManyBodyState::ManyBodyState(const std::vector<key_type> &keys,
                             const std::vector<mapped_type> &values)
    : m_map(keys, values) {}

ManyBodyState::ManyBodyState(std::vector<key_type> &&keys,
                             std::vector<mapped_type> &&values)
    : m_map(std::move(keys), std::move(values)) {}

#else

template <typename OP>
ManyBodyState::Map merge_flat_maps(ManyBodyState::Map &&map1,
                                   ManyBodyState::Map &&map2, OP &&op) {
  using V = ManyBodyState::Value;
  // 1. Prepare an output vector for the merged elements
  std::vector<ManyBodyState::Map::value_type> merged_elements;
  merged_elements.reserve(map1.size() + map2.size());

  auto it1 = map1.begin();
  auto it2 = map2.begin();

  while (it1 < map1.end() && it2 < map2.end()) {
    if (it1->first < it2->first) {
      merged_elements.emplace_back(std::move(it1->first),
                                   op(std::move(it1->second), V{}));
      it1++;
    } else if (it2->first < it1->first) {
      merged_elements.emplace_back(std::move(it2->first),
                                   op(V{}, std::move(it2->second)));
      it2++;
    } else {
      merged_elements.emplace_back(
          std::move(it1->first),
          op(std::move(it1->second), std::move(it2->second)));
      it1++;
      it2++;
    }
  }
  while (it1 < map1.end()) {
    merged_elements.emplace_back(std::move(it1->first),
                                 op(std::move(it1->second), V{}));
    it1++;
  }
  while (it2 < map2.end()) {
    merged_elements.emplace_back(std::move(it2->first),
                                 op(V{}, std::move(it2->second)));
    it2++;
  }

  // 4. Construct the final flat_map from the summed vector
  return ManyBodyState::Map(
      SORTED_UNIQUE std::move_iterator(merged_elements.begin()),
      std::move_iterator(merged_elements.end()));
}

template <typename OP>
ManyBodyState::Map merge_flat_maps(ManyBodyState::Map &&map1,
                                   const ManyBodyState::Map &map2, OP &&op) {
  using V = ManyBodyState::Value;
  // 1. Prepare an output vector for the merged elements
  std::vector<ManyBodyState::Map::value_type> merged_elements;
  merged_elements.reserve(map1.size() + map2.size());

  auto it1 = map1.begin();
  auto it2 = map2.begin();

  while (it1 < map1.end() && it2 < map2.end()) {

    if (it1->first < it2->first) {
      merged_elements.emplace_back(std::move(it1->first),
                                   op(std::move(it1->second), V{}));
      it1++;
    } else if (it2->first < it1->first) {
      merged_elements.emplace_back(it2->first, op(V{}, it2->second));
      it2++;
    } else {
      merged_elements.emplace_back(std::move(it1->first),
                                   op(std::move(it1->second), it2->second));
      it1++;
      it2++;
    }
  }
  while (it1 < map1.end()) {
    merged_elements.emplace_back(std::move(it1->first),
                                 op(std::move(it1->second), V{}));
    it1++;
  }
  while (it2 < map2.end()) {
    merged_elements.emplace_back(it2->first, op(V{}, it2->second));
    it2++;
  }

  // 4. Construct the final flat_map from the summed vector
  return ManyBodyState::Map(
      SORTED_UNIQUE std::move_iterator(merged_elements.begin()),
      std::move_iterator(merged_elements.end()));
}

ManyBodyState::ManyBodyState(const std::vector<key_type> &keys,
                             const std::vector<mapped_type> &values)
    : m_map() {
  m_map.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); i++) {
    m_map.emplace(keys[i], values[i]);
  }
}

ManyBodyState::ManyBodyState(std::vector<key_type> &&keys,
                             std::vector<mapped_type> &&values)
    : m_map() {
  m_map.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); i++) {
    m_map.emplace(std::move(keys[i]), std::move(values[i]));
  }
}

#endif

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
  for (auto &[k, v] : *this) {
    v *= s;
  }
  return *this;
}
ManyBodyState &ManyBodyState::operator/=(mapped_type s) {
  for (auto &[k, v] : *this) {
    v /= s;
  }
  return *this;
}

ManyBodyState ManyBodyState::operator-() const {
  ManyBodyState res(*this);
  for (auto &[k, v] : res) {
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
  auto new_end = std::remove_if(m_map.begin(), m_map.end(),
                                [cutoff](ManyBodyState::const_reference pair) {
                                  return abs(pair.second) <= cutoff;
                                });
  m_map.erase(new_end, m_map.end());
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
