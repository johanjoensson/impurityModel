// #include <bits/utility.h>
#include "ManyBodyState.h"
#include <bits/utility.h>
#include <bitset>
#include <iostream>
#include <iterator>
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

ManyBodyState::ManyBodyState(const std::vector<value_type> &values) : m_map() {
  std::vector<size_t> indices(values.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::stable_sort(indices.begin(), indices.end(),
                   [&values](const size_t &a, const size_t &b) {
                     return values[a].first < values[b].first;
                   });
  for (size_t idx : indices) {
    m_map.insert(values[idx]);
  }
}

ManyBodyState::ManyBodyState(std::vector<value_type> &&values) : m_map() {
  std::vector<size_t> indices(values.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::stable_sort(indices.begin(), indices.end(),
                   [&values](const size_t &a, const size_t &b) {
                     return values[a].first < values[b].first;
                   });
  for (size_t idx : indices) {
    m_map.insert(std::move(values[idx]));
  }
}

ManyBodyState::ManyBodyState(const std::vector<key_type> &keys,
                             const std::vector<mapped_type> &values)
    : m_map() {
  std::vector<size_t> indices(keys.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::stable_sort(
      indices.begin(), indices.end(),
      [&keys](const size_t &a, const size_t &b) { return keys[a] < keys[b]; });
  for (size_t idx : indices) {
    m_map.insert({keys[idx], values[idx]});
  }
}

ManyBodyState::ManyBodyState(std::vector<key_type> &&keys,
                             std::vector<mapped_type> &&values)
    : m_map() {
  std::vector<size_t> indices(keys.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::stable_sort(
      indices.begin(), indices.end(),
      [&keys](const size_t &a, const size_t &b) { return keys[a] < keys[b]; });
  for (size_t idx : indices) {
    m_map.insert({std::move(keys[idx]), std::move(values[idx])});
  }
}

double ManyBodyState::norm2() const {
  return std::transform_reduce(
      STATE_PAR this->cbegin(), this->cend(), 0.,
      [](double a, double b) { return a + b; },
      [](const_reference a) { return std::pow(abs(a.second), 2); });
}

double ManyBodyState::norm() const { return sqrt(norm2()); }

// ManyBodyState &ManyBodyState::operator+=(ManyBodyState &&other) {
//   if (other.size() > this->size()) {
//     m_map.reserve(other.size());
//   }
//   for (auto it = std::make_move_iterator(other.begin());
//        it != std::make_move_iterator(other.end()); it++) {
//     (*this)[it->first] += it->second;
//   }
//   return *this;
// }

ManyBodyState &ManyBodyState::operator+=(const ManyBodyState &other) {
  if (other.size() > this->size()) {
    m_map.reserve(other.size());
  }
  for (const auto &val : other) {
    (*this)[val.first] += val.second;
  }
  return *this;
}

ManyBodyState &ManyBodyState::operator-=(const ManyBodyState &other) {
  if (other.size() > this->size()) {
    m_map.reserve(other.size());
  }
  for (const auto &state_amp : other) {
    (*this)[state_amp.first] -= state_amp.second;
  }

  return *this;
}

ManyBodyState &ManyBodyState::operator*=(const mapped_type &s) {
  for (reference p : *this) {
    p.second *= s;
  }
  return *this;
}
ManyBodyState &ManyBodyState::operator/=(const mapped_type &s) {
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

  for (ManyBodyState::const_reference p : b) {
    if (!a.contains(p.first)) {
      continue;
    }
    res += conj(a.at(p.first)) * p.second;
  }

  return res;
}

ManyBodyState &ManyBodyState::prune(double cutoff) {
#ifdef BOOST
  boost::unordered::erase_if(m_map,
                             [cutoff](ManyBodyState::const_reference pair) {
                               return abs(pair.second) <= cutoff;
                             });
#else
  std::erase_if(m_map, [cutoff](ManyBodyState::const_reference pair) {
    return abs(pair.second) <= cutoff;
  });
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
