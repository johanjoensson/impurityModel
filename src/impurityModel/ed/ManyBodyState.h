#ifndef MANYBODY_STATE_H
#define MANYBODY_STATE_H

#include <complex>
#include <cstdint>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

class ManyBodyState {

private:
  using Key = std::vector<uint8_t>;
  using Value = std::complex<double>;
  struct Comparer {
    inline bool operator()(const Key &a, const Key &b) const noexcept {
      for (size_t i = 0; i < a.size(); i++) {
        if (a[i] < b[i]) {
          return true;
        } else if (a[i] > b[i]) {
          return false;
        }
      }
      return false;
    }
  };
  using Map = std::map<Key, Value, Comparer>;
  Map m_map;

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

  ManyBodyState() = default;
  ManyBodyState(const ManyBodyState &) = default;
  ManyBodyState(ManyBodyState &&other) : m_map(std::move(other.m_map)) {}
  ManyBodyState &operator=(const ManyBodyState &) = default;
  ManyBodyState &operator=(ManyBodyState &&) = default;
  ~ManyBodyState() = default;

  ManyBodyState(const Map &);
  ManyBodyState(Map &&);

  ManyBodyState(const std::vector<std::vector<uint8_t>> &keys,
                const std::vector<std::complex<double>> &values);
  ManyBodyState(const std::vector<std::vector<uint8_t>> &&keys,
                const std::vector<std::complex<double>> &&values);

  double norm2() const;
  double norm() const;

  ManyBodyState &operator+=(const ManyBodyState &);
  ManyBodyState &operator-=(const ManyBodyState &);
  ManyBodyState &operator*=(const std::complex<double> &);
  ManyBodyState &operator/=(const std::complex<double> &);
  ManyBodyState operator-() const;
  inline ManyBodyState operator+(const ManyBodyState &b) const {
    ManyBodyState res(*this);
    return res += b;
  }
  inline ManyBodyState operator-(const ManyBodyState &b) const {
    ManyBodyState res(*this);
    return res -= b;
  }
  friend inline ManyBodyState operator*(const std::complex<double> &s,
                                        const ManyBodyState &a) {
    return a * s;
  }
  inline ManyBodyState operator*(const std::complex<double> &s) const {
    ManyBodyState res(*this);
    return res *= s;
  }
  inline ManyBodyState operator/(const std::complex<double> &s) const {
    ManyBodyState res(*this);
    return res /= s;
  }

  friend std::complex<double> inner(const ManyBodyState &,
                                    const ManyBodyState &);

  // Wrap the member functions from std::map
  inline bool operator==(const ManyBodyState &other) {
    return this->m_map == other.m_map;
  }
  inline bool operator!=(const ManyBodyState &other) {
    return this->m_map != other.m_map;
  }
  inline allocator_type get_allocator() const { return m_map.get_allocator(); }
  inline Value &operator[](const Key &key) { return m_map[key]; }
  inline Value &operator[](Key &&key) { return m_map[std::forward<Key>(key)]; }
  inline Value &at(const Key &key) { return m_map.at(key); }
  inline const Value &at(const Key &key) const { return m_map.at(key); }

  inline bool empty() const { return m_map.empty(); }
  inline size_type size() const { return m_map.size(); }
  inline size_type max_size() const { return m_map.max_size(); }

  inline void clear() { m_map.clear(); }
  void prune(double cutoff);

  inline std::pair<iterator, bool> insert(const value_type &val) {
    return m_map.insert(val);
  }
  inline std::pair<iterator, bool> insert(value_type &&val) {
    return m_map.insert(std::forward<value_type>(val));
  }
  inline iterator insert(iterator pos, const value_type &val) {
    return m_map.insert(pos, val);
  }
  inline iterator insert(iterator pos, value_type &&val) {
    return m_map.insert(pos, std::forward<value_type>(val));
  }
  template <class InputIt> inline void insert(InputIt first, InputIt last) {
    return m_map.insert(first, last);
  }
  inline void insert(std::initializer_list<value_type> l) {
    return m_map.insert(l);
  }

  template <class... Args>
  std::pair<iterator, bool> inline emplace(Args &&...args) {
    return m_map.emplace(std::forward(args...));
  }
  template <class... Args>
  inline iterator emplace_hint(const_iterator hint, Args &&...args) {
    return m_map.emplace_hint(hint, std::forward(args...));
  }

  inline iterator erase(iterator pos) { return m_map.erase(pos); }
  inline iterator erase(const_iterator pos) { return m_map.erase(pos); }
  inline iterator erase(const_iterator first, const_iterator last) {
    return m_map.erase(first, last);
  }
  inline size_type erase(const Key &key) { return m_map.erase(key); }

  inline void swap(ManyBodyState &other) { m_map.swap(other.m_map); }

  inline size_type count(const Key &key) const { return m_map.count(key); }
  template <class K> inline size_type count(const K &x) const {
    return m_map.count(x);
  }

  inline iterator find(const Key &key) { return m_map.find(key); }
  inline const_iterator find(const Key &key) const { return m_map.find(key); }
  template <class K> inline iterator find(const K &x) { return m_map.find(x); }
  template <class K> inline const_iterator find(const K &x) const {
    return m_map.find(x);
  }

  inline std::pair<iterator, iterator> equal_range(const Key &key) {
    return m_map.equal_range(key);
  }
  inline std::pair<const_iterator, const_iterator>
  equal_range(const Key &key) const {
    return m_map.equal_range(key);
  }
  template <class K>
  inline std::pair<iterator, iterator> equal_range(const K &x) {
    return m_map.equal_range(x);
  }
  template <class K>
  inline std::pair<const_iterator, const_iterator>
  equal_range(const K &x) const {
    return m_map.equal_range(x);
  }

  inline iterator lower_bound(const Key &key) { return m_map.lower_bound(key); }
  inline const_iterator lower_bound(const Key &key) const {
    return m_map.lower_bound(key);
  }
  template <class K> inline iterator lower_bound(const K &x) {
    return m_map.lower_bound(x);
  }
  template <class K> inline const_iterator lower_bound(const K &x) const {
    return m_map.lower_bound(x);
  }

  inline iterator upper_bound(const Key &key) { return m_map.upper_bound(key); }
  inline const_iterator upper_bound(const Key &key) const {
    return m_map.upper_bound(key);
  }
  template <class K> inline iterator upper_bound(const K &x) {
    return m_map.upper_bound(x);
  }
  template <class K> inline const_iterator upper_bound(const K &x) const {
    return m_map.upper_bound(x);
  }

  inline key_compare key_comp() const { return m_map.key_comp(); }

  inline value_compare value_comp() const { return m_map.value_comp(); }

  inline iterator begin() { return m_map.begin(); }
  inline const_iterator begin() const { return m_map.begin(); }
  inline const_iterator cbegin() const noexcept { return m_map.cbegin(); }
  inline iterator end() { return m_map.end(); }
  inline const_iterator end() const { return m_map.end(); }
  inline const_iterator cend() const noexcept { return m_map.cend(); }
  inline reverse_iterator rbegin() { return m_map.rbegin(); }
  inline const_reverse_iterator rbegin() const { return m_map.rbegin(); }
  inline const_reverse_iterator crbegin() const noexcept {
    return m_map.crbegin();
  }
  inline reverse_iterator rend() { return m_map.rend(); }
  inline const_reverse_iterator rend() const { return m_map.rend(); }
  inline const_reverse_iterator crend() const noexcept { return m_map.crend(); }
};

namespace std {
inline void swap(ManyBodyState &a, ManyBodyState &b) { a.swap(b); }
}; // namespace std
#endif // MANYBODY_STATE_H
