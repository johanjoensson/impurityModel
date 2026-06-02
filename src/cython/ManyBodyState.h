#ifndef MANYBODY_STATE_H
#define MANYBODY_STATE_H

#include <boost/container/flat_map.hpp>
#include <complex>
#include <cstdint>
#if __cplusplus >= 202302L
#include <flat_map>
#endif
#include <string>
#include <utility>
#include <vector>

class ManyBodyState {

public:
  // using Key = std::vector<uint8_t>;
  using Key = std::vector<uint64_t>;
  using Value = std::complex<double>;
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
#if __cplusplus >= 202302L
  using Map = std::flat_map<Key, Value>;
#else
  using Map = boost::container::flat_map<Key, Value>;
#endif

private:
  Map m_map;

public:
  using key_type = Map::key_type;
  using mapped_type = Map::mapped_type;
  using value_type = Map::value_type;
  using size_type = Map::size_type;
  using difference_type = Map::difference_type;
  using key_compare = Map::key_compare;
  using hasher = KeyHash;
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

  explicit ManyBodyState(std::initializer_list<value_type> l) : m_map(l) {}
  explicit ManyBodyState(const std::vector<key_type> &keys,
                         const std::vector<mapped_type> &values);
  explicit ManyBodyState(std::vector<key_type> &&keys,
                         std::vector<mapped_type> &&values);

  double norm2() const;
  double norm() const;

  // The expensive part of this code is creating copies of
  // keys, to insert into the new state. We cannot move the keys
  // into the result since value_type is pair<const Key, Amplitude>,
  // and we cant move a const value.
  ManyBodyState &operator+=(ManyBodyState &&);
  ManyBodyState &operator+=(const ManyBodyState &);
  ManyBodyState &operator-=(ManyBodyState &&);
  ManyBodyState &operator-=(const ManyBodyState &);
  ManyBodyState &operator*=(mapped_type);
  ManyBodyState &operator/=(mapped_type);
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
  template <typename A>
  friend ManyBodyState operator*(A &&a, const std::complex<double> &s) {
    return (ManyBodyState{std::forward<decltype(a)>(a)} *= s);
  }
  template <typename A>
  friend ManyBodyState operator/(A &&a, const std::complex<double> &s) {
    return (ManyBodyState{std::forward<decltype(a)>(a)} /= s);
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
  ManyBodyState &prune(double cutoff);

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
    m_map.insert(first, last);
  }

  template <class... Arg> std::pair<iterator, bool> emplace(Arg &&...args) {
    return m_map.emplace(std::forward<Arg>(args)...);
  }

  std::pair<iterator, bool> try_emplace(const Key &key, Value value) {
    return m_map.try_emplace(key, value);
  }

  std::pair<iterator, bool> try_emplace(Key &&key, Value value) {
    return m_map.try_emplace(std::move(key), value);
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
  void reserve(size_type count) {

#if __cplusplus >= 202302L
    auto [keys, values] = std::move(m_map).extract();
    keys.reserve(count);
    values.reserve(count);
    m_map = Map(std::move(keys), std::move(values));
#else
    m_map.reserve(count);
#endif
  }

  key_compare key_comp() const { return m_map.key_comp(); }
};

namespace std {
inline void swap(ManyBodyState &a, ManyBodyState &b) noexcept { a.swap(b); }
}; // namespace std
#endif // MANYBODY_STATE_H
