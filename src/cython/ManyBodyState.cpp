// #include <bits/utility.h>
#ifdef PARALLEL_STL
#include <execution>
#define STATE_PAR std::execution::par_unseq,
#else // PAR
#define STATE_PAR
#endif // PAR

#include "ManyBodyState.h"
#include <bitset>
#include <numeric>

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
  return std::accumulate(this->cbegin(), this->cend(), 0.,
                         [](double acc, const_reference a) {
                           return acc + std::pow(abs(a.second), 2);
                         });
}

double ManyBodyState::norm() const { return sqrt(norm2()); }

ManyBodyState &ManyBodyState::operator+=(auto &&other) {
  if (other.size() == 0) {
    return *this;
  }

  auto other_it = other.begin();
  iterator my_it = this->begin();
  my_it = this->lower_bound(other_it->first);
  while (other_it != other.end()) {
    if (my_it == this->end()) {
      this->insert(other_it, other.end());
      other_it = other.end();
    } else if (my_it->first > other_it->first) {
      auto other_end = other.lower_bound(my_it->first);
      this->insert(other_it, other_end);
      std::advance(my_it, std::distance(other_it, other_end));
      other_it = other_end;

    } else if (my_it->first == other_it->first) {
      while (my_it != this->end() && other_it != other.end() &&
             my_it->first == other_it->first) {
        (my_it++)->second += (other_it++)->second;
      }
    } else {
      my_it = this->lower_bound(other_it->first);
    }
  }
  return *this;
}

// ManyBodyState &ManyBodyState::operator+=(const ManyBodyState &other) {
//   if (other.size() == 0) {
//     return *this;
//   }

//   auto other_it = other.begin();
//   iterator my_it = this->begin();
//   my_it = this->lower_bound(other_it->first);
//   while (other_it != other.end()) {
//     if (my_it == this->end()) {
//       this->insert(other_it, other.end());
//       other_it = other.end();
//     } else if (my_it->first > other_it->first) {
//       auto other_end = other.lower_bound(my_it->first);
//       this->insert(other_it, other_end);
//       my_it += other_end - other_it;
//       other_it = other_end;

//     } else if (my_it->first == other_it->first) {
//       while (my_it != this->end() && other_it != other.end() &&
//              my_it->first == other_it->first) {
//         (my_it++)->second += (other_it++)->second;
//       }
//     } else {
//       my_it = this->lower_bound(other_it->first);
//     }
//   }
//   return *this;
// }

ManyBodyState &ManyBodyState::operator-=(auto &&other) {
  if (other.size() == 0) {
    return *this;
  }

  auto other_it = other.begin();
  iterator my_it = this->begin();
  my_it = this->lower_bound(other_it->first);
  while (other_it != other.end()) {
    if (my_it == this->end()) {
      for (auto it = other_it; it != other.end(); it++) {
        this->insert({it->first, -it->second});
      }
      other_it = other.end();
    } else if (my_it->first > other_it->first) {
      auto other_end = other.lower_bound(my_it->first);
      std::pair<iterator, bool> tmp;
      for (auto it = other_it; it != other_end; it++) {
        tmp = this->insert({it->first, -it->second});
      }
      my_it = tmp.first;
      other_it = other_end;

    } else if (my_it->first == other_it->first) {
      while (my_it != this->end() && other_it != other.end() &&
             my_it->first == other_it->first) {
        (my_it++)->second -= (other_it++)->second;
      }
    } else {
      my_it = this->lower_bound(other_it->first);
    }
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

void ManyBodyState::prune(double cutoff) {
  std::erase_if(m_map, [cutoff](ManyBodyState::const_reference pair) {
    return abs(pair.second) <= cutoff;
  });
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
