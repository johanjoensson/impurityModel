#include "ManyBodyState.h"
#include <numeric>

ManyBodyState::ManyBodyState(const std::vector<std::vector<uint8_t>> &keys,
                             const std::vector<std::complex<double>> &values)
    : m_map() {
  for (size_t i = 0; i < keys.size(); i++) {
    m_map[keys[i]] = values[i];
    // m_map.insert({keys[i], values[i]});
  }
}
ManyBodyState::ManyBodyState(const std::vector<std::vector<uint8_t>> &&keys,
                             const std::vector<std::complex<double>> &&values)
    : m_map() {
  for (size_t i = 0; i < keys.size(); i++) {
    m_map[keys[i]] = std::move(values[i]);
    // m_map.insert({std::move(keys[i]), std::move(values[i])});
  }
}
ManyBodyState::ManyBodyState(const Map &m) : m_map(m) {}
ManyBodyState::ManyBodyState(Map &&m) : m_map(std::move(m)) {}

double ManyBodyState::norm2() const {
  return std::accumulate(m_map.cbegin(), m_map.cend(), 0.,
                         [](double acc, const_reference a) {
                           return acc + pow(abs(a.second), 2.);
                         });
}

double ManyBodyState::norm() const { return sqrt(norm2()); }

ManyBodyState &ManyBodyState::operator+=(const ManyBodyState &other) {
  for (const auto &p : other) {
    (*this)[p.first] += p.second;
  }
  return *this;
}
ManyBodyState &ManyBodyState::operator-=(const ManyBodyState &other) {
  for (const auto &p : other) {
    (*this)[p.first] -= p.second;
  }
  return *this;
}
ManyBodyState &ManyBodyState::operator*=(const std::complex<double> &s) {
  for (auto &p : *this) {
    p.second *= s;
  }
  return *this;
}
ManyBodyState &ManyBodyState::operator/=(const std::complex<double> &s) {
  for (auto &p : *this) {
    p.second /= s;
  }
  return *this;
}

ManyBodyState ManyBodyState::operator-() const {
  ManyBodyState res(*this);
  for (auto &p : res) {
    p.second = -p.second;
  }
  return res;
}

std::complex<double> inner(const ManyBodyState &a, const ManyBodyState &b) {
  std::complex<double> res = 0;
  for (const auto &p : a) {
    auto it = b.find(p.first);
    if (it != b.end()) {
      res += conj(p.second) * it->second;
    }
  }
  return res;
}

void ManyBodyState::prune(double cutoff) {
  for (auto it = m_map.begin(); it != m_map.end();) {
    if (abs(it->second) <= cutoff) {
      it = m_map.erase(it);
    } else {
      ++it;
    }
  }
}
