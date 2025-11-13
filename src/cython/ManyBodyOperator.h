#ifndef MANYBODYOPERATOR_H
#define MANYBODYOPERATOR_H
#include "ManyBodyState.h"
#include <complex>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

class ManyBodyOperator {

public:
  template <typename T> struct Comparer {
    inline bool operator()(const std::vector<T> &a,
                           const std::vector<T> &b) const noexcept {
      for (size_t i = 0; i < std::min(a.size(), b.size()); i++) {
        if (i >= a.size()) {
          return true;
        } else if (i >= b.size()) {
          return false;
        }
        if (a[i] < b[i]) {
          return true;
        } else if (a[i] > b[i]) {
          return false;
        }
      }
      return false;
    }
  };
  using SCALAR = std::complex<double>;
  using OPS = std::vector<int64_t>;
  using OPS_VEC = std::vector<OPS>;
  using SCALAR_VEC = std::vector<SCALAR>;
  // using Map = std::map<OPS, SCALAR, Comparer<int64_t>>;
  using Memory = std::map<ManyBodyState::key_type, ManyBodyState,
                          ManyBodyState::key_compare>;
  using Restrictions =
      std::vector<std::pair<std::vector<size_t>, std::pair<size_t, size_t>>>;

private:
  // Map m_ops;
  std::vector<std::pair<OPS, SCALAR>> m_ops;
  Memory m_memory;
  static bool state_is_within_restrictions(const ManyBodyState::key_type &,
                                           const Restrictions &);

public:
  using key_type = const OPS;
  using mapped_type = SCALAR;
  using value_type = std::pair<OPS, mapped_type>;
  using size_type = std::vector<value_type>::size_type;
  using difference_type = std::vector<std::pair<OPS, SCALAR>>::difference_type;
  using reference = value_type &;
  using const_reference = const value_type &;
  using compare_type = Comparer<OPS::value_type>;
  using iterator = std::vector<value_type>::iterator;
  using const_iterator = std::vector<value_type>::const_iterator;
  using reverse_iterator = std::vector<value_type>::reverse_iterator;
  using const_reverse_iterator =
      std::vector<value_type>::const_reverse_iterator;

  ManyBodyOperator() = default;
  ManyBodyOperator(const ManyBodyOperator &) = default;
  ManyBodyOperator(ManyBodyOperator &&) = default;
  ~ManyBodyOperator() = default;
  ManyBodyOperator &operator=(const ManyBodyOperator &) = default;
  ManyBodyOperator &operator=(ManyBodyOperator &&) = default;

  ManyBodyOperator(const std::vector<value_type> &);
  ManyBodyOperator(std::vector<value_type> &&);
  ManyBodyOperator(const OPS_VEC &, const SCALAR_VEC &);
  ManyBodyOperator(OPS_VEC &&, SCALAR_VEC &&);

  ManyBodyState operator()(const ManyBodyState &, double, const Restrictions &);

  Memory memory() const;

  size_type size() const;
  bool empty() const;
  bool clear();

  void clear_memory() noexcept;

  ManyBodyOperator &operator+=(const ManyBodyOperator &);
  ManyBodyOperator &operator-=(const ManyBodyOperator &);
  ManyBodyOperator &operator*=(const mapped_type &);
  ManyBodyOperator &operator/=(const mapped_type &);
  ManyBodyOperator operator-() const;

  inline ManyBodyOperator operator+(const ManyBodyOperator &other) const {
    ManyBodyOperator res(*this);
    return res += other;
  }

  inline ManyBodyOperator operator-(const ManyBodyOperator &other) const {
    ManyBodyOperator res(*this);
    return res -= other;
  }

  inline ManyBodyOperator operator*(const mapped_type &s) const {
    ManyBodyOperator res(*this);
    return res *= s;
  }

  friend inline ManyBodyOperator operator*(const mapped_type &s,
                                           const ManyBodyOperator &o) {
    return o * s;
  }

  inline ManyBodyOperator operator/(const mapped_type &s) const {
    ManyBodyOperator res(*this);
    return res /= s;
  }

  inline bool operator==(const ManyBodyOperator &other) {
    return this->m_ops == other.m_ops;
  }
  inline bool operator!=(const ManyBodyOperator &other) {
    return this->m_ops != other.m_ops;
  }
  mapped_type &operator[](const key_type &key);
  mapped_type &operator[](key_type &&key);

  mapped_type &at(const key_type &key);
  const mapped_type &at(const key_type &key) const;

  std::pair<iterator, bool> insert(const value_type &val);
  std::pair<iterator, bool> insert(value_type &&val);
  iterator insert(iterator pos, const value_type &val);
  iterator insert(iterator pos, value_type &&val);
  template <class InputIt> void insert(InputIt first, InputIt last);
  void insert(std::initializer_list<value_type> l);

  template <class... Args> std::pair<iterator, bool> emplace(Args &&...args);
  template <class... Args>
  iterator emplace_hint(const_iterator hint, Args &&...args);

  iterator erase(iterator pos);
  iterator erase(const_iterator pos);
  iterator erase(const_iterator first, const_iterator last);
  size_type erase(const key_type &key);

  inline void swap(ManyBodyOperator &other) {
    m_ops.swap(other.m_ops);
    m_memory.swap(other.m_memory);
  }

  iterator find(const key_type &);
  iterator find(iterator, iterator, const key_type &);
  const_iterator find(const key_type &key) const;
  const_iterator find(const_iterator, const_iterator, const key_type &) const;
  template <class K> iterator find(const K &);
  template <class K> iterator find(iterator, iterator, const K &);
  template <class K> const_iterator find(const K &) const;
  template <class K>
  const_iterator find(const_iterator, const_iterator, const K &) const;

  iterator lower_bound(const key_type &);
  const_iterator lower_bound(const key_type &) const;
  template <class K> iterator lower_bound(const K &);
  template <class K> const_iterator lower_bound(const K &) const;

  iterator upper_bound(const key_type &ey);
  const_iterator upper_bound(const key_type &ey) const;
  template <class K> iterator upper_bound(const K &);
  template <class K> const_iterator upper_bound(const K &) const;

  // inline compare_type key_comp() const { return compare_type(); }

  inline iterator begin() { return m_ops.begin(); }
  inline const_iterator begin() const { return m_ops.begin(); }
  inline const_iterator cbegin() const noexcept { return m_ops.cbegin(); }
  inline iterator end() { return m_ops.end(); }
  inline const_iterator end() const { return m_ops.end(); }
  inline const_iterator cend() const noexcept { return m_ops.cend(); }
  inline reverse_iterator rbegin() { return m_ops.rbegin(); }
  inline const_reverse_iterator rbegin() const { return m_ops.rbegin(); }
  inline const_reverse_iterator crbegin() const noexcept {
    return m_ops.crbegin();
  }
  inline reverse_iterator rend() { return m_ops.rend(); }
  inline const_reverse_iterator rend() const { return m_ops.rend(); }
  inline const_reverse_iterator crend() const noexcept { return m_ops.crend(); }
};
#endif // MANYBODYOPERATOR_H
