#ifdef PARALLEL_STL
#include <execution>
#define STATE_PAR std::execution::par_unseq,
#else // PAR
#define STATE_PAR
#endif // PAR

#include "ManyBodyState.h"
#include <bitset>
#include <iterator>
#include <numeric>

ManyBodyState::ManyBodyState(const std::vector<value_type> &values)
    : m_keys(), m_values() {
  m_keys.reserve(values.size());
  m_values.reserve(values.size());
  std::vector<size_t> indices(values.size());
  std::iota(indices.begin(), indices.end(), 0);

  KeyComparer comparer{};
  std::stable_sort(indices.begin(), indices.end(),
                   [&comparer, &values](const size_t &a, const size_t &b) {
                     return comparer(values[a].first, values[b].first);
                   });
  for (size_t idx : indices) {
    m_keys.push_back(values[idx].first);
    m_values.push_back(values[idx].second);
  }
}

ManyBodyState::ManyBodyState(std::vector<value_type> &&values)
    : m_keys(), m_values() {
  m_keys.reserve(values.size());
  m_values.reserve(values.size());
  std::vector<size_t> indices(values.size());
  std::iota(indices.begin(), indices.end(), 0);

  KeyComparer comparer{};
  std::stable_sort(indices.begin(), indices.end(),
                   [&comparer, &values](const size_t &a, const size_t &b) {
                     return comparer(values[a].first, values[b].first);
                   });
  for (size_t idx : indices) {
    m_keys.push_back(std::move(values[idx].first));
    m_values.push_back(std::move(values[idx].second));
  }
}

ManyBodyState::ManyBodyState(const std::vector<key_type> &keys,
                             const std::vector<mapped_type> &values)
    : m_keys(), m_values() {
  m_keys.reserve(keys.size());
  m_values.reserve(values.size());
  std::vector<size_t> indices(keys.size());
  std::iota(indices.begin(), indices.end(), 0);

  KeyComparer comparer{};
  std::stable_sort(indices.begin(), indices.end(),
                   [&comparer, &keys](const size_t &a, const size_t &b) {
                     return comparer(keys[a], keys[b]);
                   });
  for (size_t idx : indices) {
    m_keys.push_back(keys[idx]);
    m_values.push_back(values[idx]);
  }
}

ManyBodyState::ManyBodyState(std::vector<key_type> &&keys,
                             std::vector<mapped_type> &&values)
    : m_keys(), m_values() {
  m_keys.reserve(keys.size());
  m_values.reserve(values.size());
  std::vector<size_t> indices(keys.size());
  std::iota(indices.begin(), indices.end(), 0);

  KeyComparer comparer{};
  std::stable_sort(indices.begin(), indices.end(),
                   [&comparer, &keys](const size_t &a, const size_t &b) {
                     return comparer(keys[a], keys[b]);
                   });
  for (size_t idx : indices) {
    m_keys.push_back(std::move(keys[idx]));
    m_values.push_back(std::move(values[idx]));
  }
}

double ManyBodyState::norm2() const {
  return std::accumulate(this->cbegin(), this->cend(), 0.,
                         [](double acc, const_reference a) {
                           return acc + std::pow(abs(a.second), 2);
                         });
}

double ManyBodyState::norm() const { return sqrt(norm2()); }

ManyBodyState &ManyBodyState::operator+=(const ManyBodyState &other) {
  auto comparer = ManyBodyState::KeyComparer();
  // this->m_keys.reserve(other.size() + this->size());
  // this->m_values.reserve(other.size() + this->size());
  // this->m_keys.reserve(std::max(other.size(), this->size()));
  // this->m_values.reserve(std::max(other.size(), this->size()));
  iterator my_it = this->begin();
  for (auto other_it = other.cbegin(); other_it != other.cend(); other_it++) {
    my_it =
        std::lower_bound(my_it, this->end(), *other_it,
                         [&](const const_reference a, const const_reference b) {
                           return comparer(a.first, b.first);
                         });

    /* If we are at the end of *this, just insert whatever elements are left
     * in other*/
    if (my_it == this->end()) {
      // for (; other_it != other.cend(); other_it++) {
      //   this->push_back(*other_it);
      // }
      this->insert(this->cend(), other_it, other.cend());
      break;
      /* Element in other_it points to an element that should be inserted
       * before my_it */
    } else if (comparer((*other_it).first, (*my_it).first)) {
      // my_it = this->insert(iterator(my_it.m_it.first, my_it.m_it.second),
      //                      *other_it);
      auto other_end = std::lower_bound(other_it, other.end(), *my_it,
                                        [&](const auto &a, const auto &b) {
                                          return comparer(a.first, b.first);
                                        });
      my_it = this->insert(const_iterator(my_it.m_it.first, my_it.m_it.second),
                           other_it, other_end);
      my_it += other_end - other_it;
      other_it = --other_end;
    } else if ((*my_it).first == (*other_it).first) {
      (*my_it++).second += (*other_it).second;
    }
  }
  return *this;
}

ManyBodyState &ManyBodyState::operator-=(const ManyBodyState &other) {
  auto comparer = ManyBodyState::KeyComparer();
  // this->m_keys.reserve(std::max(other.size(), this->size()));
  // this->m_values.reserve(std::max(other.size(), this->size()));
  iterator my_it = this->begin();
  for (auto other_it = other.cbegin(); other_it != other.cend(); other_it++) {
    my_it = std::lower_bound(my_it, this->end(), *other_it,
                             [&](const_reference a, const_reference b) {
                               return comparer(a.first, b.first);
                             });

    /* If we are at the end of *this, just insert whatever elements are left
     * in other*/
    if (my_it == this->end()) {

      my_it = this->insert(this->cend(), other_it, other.cend());
      std::transform(STATE_PAR my_it, this->end(), my_it, [](auto &&p) {
        return std::forward<std::pair<Key, Value>>({p.first, -p.second});
      });
      break;
      /* Element in other_it points to an element that should be inserted
       * before my_it */
    } else if (comparer((*other_it).first, (*my_it).first)) {
      auto other_end = std::lower_bound(other_it, other.end(), *my_it,
                                        [&](const auto &a, const auto &b) {
                                          return comparer(a.first, b.first);
                                        });
      my_it = this->insert(const_iterator(my_it.m_it.first, my_it.m_it.second),
                           other_it, other_end);
      std::transform(STATE_PAR my_it, my_it + (other_end - other_it), my_it,
                     [](const auto &p) {
                       return std::forward<std::pair<Key, Value>>(
                           {p.first, -p.second});
                     });
      // my_it += other_end - other_it;
      other_it = --other_end;
    } else if ((*my_it).first == (*other_it).first) {
      (*my_it++).second -= (*other_it).second;
    }
  }
  return *this;
}

ManyBodyState &ManyBodyState::operator*=(const mapped_type &s) {
  std::transform(
      STATE_PAR this->begin(), this->end(), this->begin(), [s](auto &&p) {
        return std::forward<std::pair<Key, Value>>({p.first, p.second * s});
      });
  return *this;
}
ManyBodyState &ManyBodyState::operator/=(const mapped_type &s) {
  std::transform(
      STATE_PAR this->begin(), this->end(), this->begin(), [s](auto &&p) {
        return std::forward<std::pair<Key, Value>>({p.first, p.second / s});
      });
  return *this;
}

ManyBodyState ManyBodyState::operator-() const {
  ManyBodyState res(*this);
  std::transform(STATE_PAR res.begin(), res.end(), res.begin(), [](auto &&p) {
    return std::forward<std::pair<Key, Value>>({p.first, -p.second});
  });
  return res;
}

std::complex<double> inner(const ManyBodyState &a, const ManyBodyState &b) {
  std::complex<double> res = 0;

  for (auto a_it = a.begin(), b_it = b.begin();
       a_it != a.end() && b_it != b.end(); a_it++) {
    b_it = std::lower_bound(
        b_it, b.end(), *a_it,
        [](ManyBodyState::const_reference a, ManyBodyState::const_reference b) {
          return a.first < b.first;
        });
    if (b_it == b.end()) {
      break;
    } else if ((*a_it).first == (*b_it).first) {
      res += conj((*a_it).second) * (*b_it).second;
    }
  }
  return res;
}

void ManyBodyState::prune(double cutoff) {
  this->erase(std::remove_if(STATE_PAR this->begin(), this->end(),
                             [cutoff](const ManyBodyState::reference &pair) {
                               return abs(pair.second) <= cutoff;
                             }),
              this->end());
}

void ManyBodyState::reserve(size_t capacity) {
  this->m_keys.reserve(capacity);
  this->m_values.reserve(capacity);
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

#if __cplusplus >= 202302L
static_assert(std::random_access_iterator<ManyBodyState::iterator>);
static_assert(std::random_access_iterator<ManyBodyState::const_iterator>);
#else
static_assert(std::bidirectional_iterator<ManyBodyState::iterator>);
// static_assert(std::bidirectional_iterator<ManyBodyState::const_iterator>);
#endif
