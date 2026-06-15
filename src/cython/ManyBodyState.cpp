#include "ManyBodyState.h"
#include <algorithm>
#include <bitset>
#include <functional>
#include <numeric>

template <typename OP>
void merge_maps(ManyBodyState::Map &map1, const ManyBodyState::Map &map2,
                OP &&op) {

  // map1.reserve(map1.size() + map2.size()); // std::flat_map does not have
  // reserve
  using V = ManyBodyState::Value;
  for (auto &&[key, value] : map2) {
    auto [it, inserted] = map1.try_emplace(key, op(V{}, value));
    if (!inserted) {
      it->second = op(it->second, value);
    }
  }
}

template <typename OP>
void merge_maps(ManyBodyState::Map &map1, ManyBodyState::Map &&map2, OP &&op) {

  // map1.reserve(map1.size() + map2.size()); // std::flat_map does not have
  // reserve
  using V = ManyBodyState::Value;
  for (auto &&[key, value] : map2) {
    auto [it, inserted] = map1.try_emplace(std::move(key), op(V{}, value));
    if (!inserted) {
      it->second = op(it->second, value);
    }
  }
}

ManyBodyState::ManyBodyState(const std::vector<key_type> &keys,
                             const std::vector<mapped_type> &values)
    : m_map() {
  if (!keys.empty()) {
    std::vector<std::pair<key_type, mapped_type>> temp;
    temp.reserve(keys.size());
    for (size_t idx = 0; idx < keys.size(); idx++) {
      temp.emplace_back(keys[idx], values[idx]);
    }
    m_map.insert(temp.begin(), temp.end());
  }
}

ManyBodyState::ManyBodyState(std::vector<key_type> &&keys,
                             std::vector<mapped_type> &&values)
    : m_map() {
  if (!keys.empty()) {
    std::vector<std::pair<key_type, mapped_type>> temp;
    temp.reserve(keys.size());
    for (size_t idx = 0; idx < keys.size(); idx++) {
      temp.emplace_back(std::move(keys[idx]), std::move(values[idx]));
    }
    m_map.insert(temp.begin(), temp.end());
  }
}

double ManyBodyState::norm2() const {
  return std::transform_reduce(
      this->cbegin(), this->cend(), 0.,
      [](double a, double b) { return a + b; },
      [](const_reference a) { return std::norm(a.second); });
}

double ManyBodyState::norm() const { return sqrt(norm2()); }

ManyBodyState &ManyBodyState::operator+=(ManyBodyState &&other) {
  if (this->m_map.empty()) {
    this->m_map = std::move(other.m_map);
    return *this;
  }
  this->reserve(m_map.size() + other.m_map.size());
  merge_maps(this->m_map, std::move(other.m_map), std::plus());
  return *this;
}

ManyBodyState &ManyBodyState::operator+=(const ManyBodyState &other) {
  if (this->m_map.empty()) {
    this->m_map = other.m_map;
    return *this;
  }
  this->reserve(m_map.size() + other.m_map.size());
  merge_maps(this->m_map, other.m_map, std::plus());
  return *this;
}

ManyBodyState &ManyBodyState::operator-=(ManyBodyState &&other) {
  if (this->m_map.empty()) {
    this->m_map = std::move(other.m_map);
    for (auto &&[k, v] : this->m_map) {
      v = -v;
    }
    return *this;
  }
  this->reserve(m_map.size() + other.m_map.size());
  merge_maps(this->m_map, std::move(other.m_map), std::minus());
  return *this;
}

ManyBodyState &ManyBodyState::operator-=(const ManyBodyState &other) {
  if (this->m_map.empty()) {
    this->m_map = other.m_map;
    for (auto &&[k, v] : this->m_map) {
      v = -v;
    }
    return *this;
  }
  this->reserve(m_map.size() + other.m_map.size());
  merge_maps(this->m_map, other.m_map, std::minus());
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

void ManyBodyState::add_scaled(const ManyBodyState &other, mapped_type scale) {
  if (other.m_map.empty()) {
    return;
  }
  if (scale == mapped_type{0.}) {
    return;
  }
  this->reserve(m_map.size() + other.m_map.size());
  for (auto &&[key, value] : other.m_map) {
    auto [it, inserted] = m_map.try_emplace(key, value * scale);
    if (!inserted) {
      it->second += value * scale;
    }
  }
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

  if (a.size() <= b.size()) {
    for (auto &&[key, value] : a) {
      auto it = b.find(key);
      if (it != b.end()) {
        res += std::conj(value) * it->second;
      }
    }
  } else {
    for (auto &&[key, value] : b) {
      auto it = a.find(key);
      if (it != a.end()) {
        res += std::conj(it->second) * value;
      }
    }
  }

  return res;
}

void ManyBodyState::reserve(size_t n) {
#if __cplusplus >= 202302L && __has_include(<flat_map>)
  auto [keys, vals] = std::move(m_map).extract();
  keys.reserve(n);
  vals.reserve(n);
  m_map.replace(std::move(keys), std::move(vals));
#else
  m_map.reserve(n);
#endif
}

ManyBodyState &ManyBodyState::prune(double cutoff) {
  double cutoff2 = cutoff * cutoff;
#if __cplusplus >= 202302L && __has_include(<flat_map>)
  std::erase_if(m_map, [cutoff2](ManyBodyState::const_reference pair) {
#else
  boost::container::erase_if(m_map,
                             [cutoff2](ManyBodyState::const_reference pair) {
#endif
    return std::norm(pair.second) <= cutoff2;
  });
  return *this;
}

std::string ManyBodyState::to_string() const {
  std::string res;
  res += "ManyBodyState{";
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
