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

// ManyBodyState::ManyBodyState(std::initializer_list<value_type> &&values)
//     : m_map(values) {
//   // std::vector<size_t> indices(values.size());
//   // std::iota(indices.begin(), indices.end(), 0);

//   // std::stable_sort(indices.begin(), indices.end(),
//   //                  [&values](const size_t &a, const size_t &b) {
//   //                    return values[a].first < values[b].first;
//   //                  });
//   // for (size_t idx : indices) {
//   //   m_map.insert(values[idx]);
//   // }
// }

// ManyBodyState::ManyBodyState(const std::vector<value_type> &values) : m_map()
// {
//   std::vector<size_t> indices(values.size());
//   std::iota(indices.begin(), indices.end(), 0);

//   std::stable_sort(indices.begin(), indices.end(),
//                    [&values](const size_t &a, const size_t &b) {
//                      return values[a].first < values[b].first;
//                    });
//   for (size_t idx : indices) {
//     m_map.insert(values[idx]);
//   }
// }

// ManyBodyState::ManyBodyState(std::vector<value_type> &&values) : m_map() {
//   std::vector<size_t> indices(values.size());
//   std::iota(indices.begin(), indices.end(), 0);
//   std::stable_sort(indices.begin(), indices.end(),
//                    [&values](const size_t &a, const size_t &b) {
//                      return values[a].first < values[b].first;
//                    });
//   for (size_t idx : indices) {
//     m_map.insert(std::move(values[idx]));
//   }
// }

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

ManyBodyState &ManyBodyState::operator+=(ManyBodyState &&other) {
  for (auto &&[key, amp] : other) {
    auto [it, inserted] = m_map.try_emplace(key, amp);
    if (!inserted) {
      it->second += amp;
    }
  }
  return *this;
}

ManyBodyState &ManyBodyState::operator+=(const ManyBodyState &other) {
  for (auto &&[key, amp] : other) {
    auto [it, inserted] = m_map.try_emplace(key, amp);
    if (!inserted) {
      it->second += amp;
    }
  }
  return *this;
}

ManyBodyState &ManyBodyState::operator-=(ManyBodyState &&other) {
  for (auto &&[key, amp] : other) {
    auto [it, inserted] = m_map.try_emplace(key, -amp);
    if (!inserted) {
      it->second -= amp;
    }
  }
  return *this;
}

ManyBodyState &ManyBodyState::operator-=(const ManyBodyState &other) {
  for (auto &&[key, amp] : other) {
    auto [it, inserted] = m_map.try_emplace(key, -amp);
    if (!inserted) {
      it->second -= amp;
    }
  }
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

  for (auto &&[key, amp] : b) {
    if (!a.contains(key)) {
      continue;
    }
    res += conj(a.at(key)) * amp;
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
