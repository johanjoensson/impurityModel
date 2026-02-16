#ifndef MANYBODY_STATE_H
#define MANYBODY_STATE_H

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#if __cplusplus >= 202302L
#include <flat_map>
#define SORTED_UNIQUE std::sorted_unique,
#else
#define SORTED_UNIQUE
#include <map>
#endif
#include <string>
#include <utility>
#include <vector>

class ManyBodyState {

public:
  using Key = std::vector<uint64_t>;
  using Value = std::complex<double>;
  struct KeyComparer {
    bool operator()(const Key &a, const Key &b) const noexcept { return a < b; }
  };
  struct KeyHash {
    std::size_t operator()(const Key &key) const {
      std::size_t res = 0;
      const std::size_t MAGIC = 0x9e3779b9;
      std::hash<Key::value_type> hasher{};
      for (const Key::value_type &k : key) {
        res ^= hasher(k) + MAGIC + (res << 6) + (res >> 2);
      }
      return res;
    }
  };
  struct ValueComparer {
    bool operator()(const Value &a, const Value &b) const noexcept {
      if (std::abs(a) < std::abs(b)) {
        return true;
      }
      if (std::abs(b) < std::abs(a)) {
        return false;
      }
      return std::arg(a) < std::arg(b);
    }
  };
#if __cplusplus >= 202302L
  using Map = std::flat_map<Key, Value>;
#else
  using Map = std::map<Key, Value>;
#endif

private:
  Map m_map;

public:
  using key_type = Map::key_type;
  using mapped_type = Map::mapped_type;
  using value_type = Map::value_type;
  using size_type = Map::size_type;
  using difference_type = Map::difference_type;
  using key_compare = KeyComparer;
  using value_compare = ValueComparer;
  using reference = Map::reference;
  using const_reference = Map::const_reference;

  using iterator = Map::iterator;
  using const_iterator = Map::const_iterator;

  ManyBodyState() = default;
  ManyBodyState(const ManyBodyState &) = default;
  ManyBodyState(ManyBodyState &&) noexcept = default;
  ManyBodyState &operator=(const ManyBodyState &) = default;
  ManyBodyState &operator=(ManyBodyState &&) = default;
  ~ManyBodyState() = default;

  explicit ManyBodyState(const std::vector<value_type> &);
  explicit ManyBodyState(std::vector<value_type> &&);
  explicit ManyBodyState(const std::vector<key_type> &keys,
                         const std::vector<mapped_type> &values);
  explicit ManyBodyState(std::vector<key_type> &&keys,
                         std::vector<mapped_type> &&values);

  double norm2() const;
  double norm() const;

  ManyBodyState &operator+=(auto &&);
  // ManyBodyState &operator+=(const ManyBodyState &);
  ManyBodyState &operator-=(auto &&);
  // ManyBodyState &operator-=(const ManyBodyState &);
  ManyBodyState &operator*=(const std::complex<double> &);
  ManyBodyState &operator/=(const std::complex<double> &);
  ManyBodyState operator-() const;
  friend ManyBodyState operator+(const ManyBodyState &a,
                                 const ManyBodyState &b) {
    return (ManyBodyState{a} += b);
  }

  friend ManyBodyState operator-(const ManyBodyState &a,
                                 const ManyBodyState &b) {
    return (ManyBodyState{a} -= b);
  }
  friend ManyBodyState operator*(const std::complex<double> &s,
                                 const ManyBodyState &a) {
    return (ManyBodyState{a} *= s);
  }
  friend ManyBodyState operator*(auto &&a, const std::complex<double> &s) {
    return (ManyBodyState{a} *= s);
  }
  friend ManyBodyState operator/(auto &&a, const std::complex<double> &s) {
    return (ManyBodyState{a} /= s);
  }

  friend std::complex<double> inner(const ManyBodyState &,
                                    const ManyBodyState &);

  // Wrap the member functions from std::map
  friend bool operator==(const ManyBodyState &self,
                         const ManyBodyState &other) {
    return self.m_map == other.m_map;
  }
  friend bool operator!=(const ManyBodyState &self,
                         const ManyBodyState &other) {
    return !(self == other);
  }
  mapped_type &operator[](const Key &key) { return m_map[key]; }
  mapped_type &operator[](Key &&key) { return m_map[std::move(key)]; }
  mapped_type &at(const Key &key) { return m_map.at(key); }
  const mapped_type &at(const Key &key) const { return m_map.at(key); }

  bool empty() const { return m_map.empty(); }
  size_type size() const { return m_map.size(); }
  size_type max_size() const { return m_map.max_size(); }

  void clear() { m_map.clear(); }
  void prune(double cutoff);

  iterator begin() { return m_map.begin(); }
  const_iterator begin() const { return m_map.begin(); }
  const_iterator cbegin() const noexcept { return m_map.cbegin(); }

  iterator end() { return m_map.end(); }
  const_iterator end() const { return m_map.end(); }
  const_iterator cend() const noexcept { return m_map.cend(); }

  std::pair<iterator, bool> insert(const value_type &val) {
    return m_map.insert(val);
  }

  std::pair<iterator, bool> insert(value_type &&val) {
    return m_map.insert(std::move(val));
  }

  iterator insert(const_iterator pos, const value_type &val) {
    return m_map.insert(pos, val);
  }
  iterator insert(const_iterator pos, value_type &&val) {
    return m_map.insert(pos, std::move(val));
  }

  template <class InputIt> void insert(InputIt first, InputIt last) {
    m_map.insert(SORTED_UNIQUE first, last);
  }
  iterator erase(iterator pos) { return m_map.erase(pos); }
  iterator erase(const_iterator pos) { return m_map.erase(pos); }

  iterator erase(const_iterator first, const_iterator last) {
    return m_map.erase(first, last);
  }

  size_type erase(const key_type &key) { return m_map.erase(key); }

  void swap(ManyBodyState &other) noexcept { m_map.swap(other.m_map); }

  iterator find(const key_type &key) { return m_map.find(key); }

  const_iterator find(const key_type &key) const { return m_map.find(key); }

  template <class K> iterator find(const K &key) { return m_map.find(key); }

  template <class K> const_iterator find(const K &key) const {
    return m_map.find(key);
  }

  bool contains(const key_type &k) const { return m_map.contains(k); }
  template <typename K> bool contains(const K &k) const {
    return m_map.contains(k);
  }

  iterator lower_bound(const key_type &key) { return m_map.lower_bound(key); }

  const_iterator lower_bound(const key_type &key) const {
    return m_map.lower_bound(key);
  }

  template <class K> iterator lower_bound(const K &key) {
    return m_map.lower_bound(key);
  }
  template <class K> const_iterator lower_bound(const K &key) const {
    return m_map.lower_bound(key);
  }

  iterator upper_bound(const key_type &key) { return m_map.upper_bound(key); }

  const_iterator upper_bound(const key_type &key) const {
    return m_map.upper_bound(key);
  }

  template <class K> iterator upper_bound(const K &key) {
    return m_map.upper_bound(key);
  }

  template <class K> const_iterator upper_bound(const K &key) const {
    return m_map.upper_bound(key);
  }
  std::string to_string() const;
};

namespace std {
inline void swap(ManyBodyState &a, ManyBodyState &b) noexcept { a.swap(b); }
// #if __cplusplus >= 202302L
// template <template <class> class TQual, template <class> class UQual>
// struct basic_common_reference<ManyBodyState::const_reference,
//                               ManyBodyState::value_type, TQual, UQual> {
//   using type = ManyBodyState::value_type;
// };

// template <template <class> class TQual, template <class> class UQual>
// struct basic_common_reference<ManyBodyState::value_type,
//                               ManyBodyState::const_reference, TQual, UQual>
//                               {
//   using type = ManyBodyState::value_type;
// };
// #endif
}; // namespace std
#endif // MANYBODY_STATE_H
