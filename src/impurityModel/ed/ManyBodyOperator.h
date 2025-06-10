#ifndef MANYBODYOPERATOR_H
#define MANYBODYOPERATOR_H
#include "ManyBodyState.h"
#include <complex>
#include <cstdint>
#include <map>
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
  using Map = std::map<OPS, SCALAR, Comparer<int64_t>>;
  using Memory = std::map<ManyBodyState::key_type, ManyBodyState,
                          ManyBodyState::key_compare>;
  using Restrictions = std::map<std::vector<size_t>, std::pair<size_t, size_t>,
                                Comparer<size_t>>;

private:
  Map m_ops;
  Memory m_memory;
  static bool state_is_within_restrictions(const ManyBodyState::key_type &,
                                           const Restrictions &);

public:
  using key_type = Map::key_type;
  using mapped_type = Map::mapped_type;
  using value_type = Map::value_type;
  using size_type = Map::size_type;
  using difference_type = Map::difference_type;
  using key_compare = Map::key_compare;
  using value_compare = Map::value_compare;
  using allocator_type = Map::allocator_type;
  using reference = Map::reference;
  using const_reference = Map::const_reference;
  using pointer = Map::pointer;
  using const_pointer = Map::const_pointer;
  using iterator = Map::iterator;
  using const_iterator = Map::const_iterator;
  using reverse_iterator = Map::reverse_iterator;
  using const_reverse_iterator = Map::const_reverse_iterator;

  ManyBodyOperator() = default;
  ManyBodyOperator(const ManyBodyOperator &) = default;
  ManyBodyOperator(ManyBodyOperator &&) = default;
  ~ManyBodyOperator() = default;
  ManyBodyOperator &operator=(const ManyBodyOperator &) = default;
  ManyBodyOperator &operator=(ManyBodyOperator &&) = default;

  ManyBodyOperator(const std::vector<std::pair<key_type, mapped_type>> &);
  ManyBodyOperator(std::vector<std::pair<key_type, mapped_type>> &&);
  ManyBodyOperator(const Map &);
  ManyBodyOperator(Map &&);

  void add_ops(const std::vector<std::pair<key_type, mapped_type>> &ops);
  void add_ops(std::vector<std::pair<key_type, mapped_type>> &&ops);
  ManyBodyState operator()(const ManyBodyState &, double, const Restrictions &);

  Memory memory() const;

  Map::size_type size() const;
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
  inline allocator_type get_allocator() const { return m_ops.get_allocator(); }
  inline mapped_type &operator[](const key_type &key) { return m_ops[key]; }
  inline mapped_type &operator[](key_type &&key) {
    return m_ops[std::forward<key_type>(key)];
  }
  inline mapped_type &at(const key_type &key) { return m_ops.at(key); }
  inline const mapped_type &at(const key_type &key) const {
    return m_ops.at(key);
  }
  inline std::pair<iterator, bool> insert(const value_type &val) {
    return m_ops.insert(val);
  }
  inline std::pair<iterator, bool> insert(value_type &&val) {
    return m_ops.insert(std::forward<value_type>(val));
  }
  inline iterator insert(iterator pos, const value_type &val) {
    return m_ops.insert(pos, val);
  }
  inline iterator insert(iterator pos, value_type &&val) {
    return m_ops.insert(pos, std::forward<value_type>(val));
  }
  template <class InputIt> inline void insert(InputIt first, InputIt last) {
    return m_ops.insert(first, last);
  }
  inline void insert(std::initializer_list<value_type> l) {
    return m_ops.insert(l);
  }

  template <class... Args>
  std::pair<iterator, bool> inline emplace(Args &&...args) {
    return m_ops.emplace(std::forward(args...));
  }
  template <class... Args>
  inline iterator emplace_hint(const_iterator hint, Args &&...args) {
    return m_ops.emplace_hint(hint, std::forward(args...));
  }

  inline iterator erase(iterator pos) { return m_ops.erase(pos); }
  inline iterator erase(const_iterator pos) { return m_ops.erase(pos); }
  inline iterator erase(const_iterator first, const_iterator last) {
    return m_ops.erase(first, last);
  }
  inline size_type erase(const key_type &key) { return m_ops.erase(key); }

  inline void swap(ManyBodyOperator &other) {
    m_ops.swap(other.m_ops);
    m_memory.swap(other.m_memory);
  }

  inline size_type count(const key_type &key) const { return m_ops.count(key); }
  template <class K> inline size_type count(const K &x) const {
    return m_ops.count(x);
  }

  inline iterator find(const key_type &key) { return m_ops.find(key); }
  inline const_iterator find(const key_type &key) const {
    return m_ops.find(key);
  }
  template <class K> inline iterator find(const K &x) { return m_ops.find(x); }
  template <class K> inline const_iterator find(const K &x) const {
    return m_ops.find(x);
  }

  inline std::pair<iterator, iterator> equal_range(const key_type &key) {
    return m_ops.equal_range(key);
  }
  inline std::pair<const_iterator, const_iterator>
  equal_range(const key_type &key) const {
    return m_ops.equal_range(key);
  }
  template <class K>
  inline std::pair<iterator, iterator> equal_range(const K &x) {
    return m_ops.equal_range(x);
  }
  template <class K>
  inline std::pair<const_iterator, const_iterator>
  equal_range(const K &x) const {
    return m_ops.equal_range(x);
  }

  inline iterator lower_bound(const key_type &key) {
    return m_ops.lower_bound(key);
  }
  inline const_iterator lower_bound(const key_type &key) const {
    return m_ops.lower_bound(key);
  }
  template <class K> inline iterator lower_bound(const K &x) {
    return m_ops.lower_bound(x);
  }
  template <class K> inline const_iterator lower_bound(const K &x) const {
    return m_ops.lower_bound(x);
  }

  inline iterator upper_bound(const key_type &key) {
    return m_ops.upper_bound(key);
  }
  inline const_iterator upper_bound(const key_type &key) const {
    return m_ops.upper_bound(key);
  }
  template <class K> inline iterator upper_bound(const K &x) {
    return m_ops.upper_bound(x);
  }
  template <class K> inline const_iterator upper_bound(const K &x) const {
    return m_ops.upper_bound(x);
  }

  inline key_compare key_comp() const { return m_ops.key_comp(); }

  inline value_compare value_comp() const { return m_ops.value_comp(); }

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
