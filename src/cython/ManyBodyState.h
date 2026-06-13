#ifndef MANYBODY_STATE_H
#define MANYBODY_STATE_H

#ifdef BOOST
#include <boost/unordered/unordered_flat_map.hpp>
#endif
#include "SlaterDeterminant.h"

#include <complex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * @class ManyBodyState
 * @brief Represents a quantum many-body state in a second-quantized Slater determinant basis.
 *
 * This class stores the coefficients (amplitudes) of a many-body wavefunction
 * mapped from each SlaterDeterminant. It supports standard vector space operations
 * (addition, scalar multiplication) and quantum mechanical operations like inner products.
 *
 * Compilation flags:
 * - BOOST: switches the underlying hash map from std::unordered_map to boost::unordered_flat_map for improved performance.
 */
class ManyBodyState {

public:
  using Key = SlaterDeterminant<>;
  using Value = std::complex<double>;
#ifndef BOOST
  using Map = std::unordered_map<Key, Value, std::hash<Key>>;
#else
  using Map = boost::unordered::unordered_flat_map<Key, Value, std::hash<Key>>;
#endif

private:
  Map m_map;

public:
  using key_type = Map::key_type;
  using mapped_type = Map::mapped_type;
  using value_type = Map::value_type;
  using size_type = Map::size_type;
  using difference_type = Map::difference_type;
  using key_equal = Map::key_equal;
  using hasher = std::hash<key_type>;
  using reference = Map::reference;
  using const_reference = Map::const_reference;

  using iterator = Map::iterator;
  using const_iterator = Map::const_iterator;
#ifndef BOOST
  using local_iterator = Map::local_iterator;
  using const_local_iterator = Map::const_local_iterator;
#endif

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

  /**
   * @brief Compute the squared norm (L2 norm) of the state.
   */
  double norm2() const;

  /**
   * @brief Compute the norm (L2 norm) of the state.
   */
  double norm() const;

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

  /**
   * @brief Compute the inner product (bra-ket) of two states: <a|b>.
   */
  friend std::complex<double> inner(const ManyBodyState &,
                                    const ManyBodyState &);

  // Wrap the member functions from std::map / boost::unordered_flat_map
  friend bool operator==(const ManyBodyState &self,
                         const ManyBodyState &other) {
    return self.m_map == other.m_map;
  }
  friend bool operator!=(const ManyBodyState &self,
                         const ManyBodyState &other) {
    return !(self == other);
  }
  mapped_type &operator[](const key_type &key) { return m_map[key]; }
  mapped_type &operator[](key_type &&key) { return m_map[std::move(key)]; }
  mapped_type &at(const key_type &key) { return m_map.at(key); }
  const mapped_type &at(const key_type &key) const { return m_map.at(key); }

  bool empty() const { return m_map.empty(); }
  size_type size() const { return m_map.size(); }
  size_type max_size() const { return m_map.max_size(); }

  void clear() { m_map.clear(); }

  /**
   * @brief Prune the state by removing components with amplitudes squared <= cutoff.
   */
  ManyBodyState &prune(double cutoff);

  size_type bucket_count() const { return m_map.bucket_count(); }
  float load_factor() const { return m_map.load_factor(); }
  float max_load_factor() const { return m_map.max_load_factor(); }
  void max_load_factor(float ml) { m_map.max_load_factor(ml); }
  iterator begin() { return m_map.begin(); }
  const_iterator begin() const { return m_map.begin(); }
  const_iterator cbegin() const noexcept { return m_map.cbegin(); }

  iterator end() { return m_map.end(); }
  const_iterator end() const { return m_map.end(); }
  const_iterator cend() const noexcept { return m_map.cend(); }

#ifndef BOOST
  // Local iterators (over buckets)
  local_iterator begin(size_t n) { return m_map.begin(n); }
  const_local_iterator begin(size_t n) const { return m_map.begin(n); }
  const_local_iterator cbegin(size_t n) const noexcept {
    return m_map.cbegin(n);
  }

  local_iterator end(size_t n) { return m_map.end(n); }
  const_local_iterator end(size_t n) const { return m_map.end(n); }
  const_local_iterator cend(size_t n) const noexcept { return m_map.cend(n); }
#endif

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

  template <class... Args>
  iterator emplace_hint(const_iterator hint, Args &&...args) {
    return m_map.emplace_hint(hint, std::forward<Args>(args)...);
  }

  std::pair<iterator, bool> try_emplace(const key_type &key,
                                        mapped_type value) {
    return m_map.try_emplace(key, value);
  }

  std::pair<iterator, bool> try_emplace(key_type &&key, mapped_type value) {
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

  bool contains(const key_type &k) const {
#if __cpluplus >= 202002L
    return m_map.contains(k);
#else
    auto it = m_map.find(k);
    return it != m_map.end();
#endif
  }
  template <typename K> bool contains(const K &k) const {
#if __cpluplus >= 202002L
    return m_map.contains(k);
#else
    auto it = m_map.find(k);
    return it != m_map.end();
#endif
  }

  /**
   * @brief String representation of the state for debugging.
   */
  std::string to_string() const;
  void reserve(size_type count) { m_map.reserve(count); }
};

namespace std {
inline void swap(ManyBodyState &a, ManyBodyState &b) noexcept { a.swap(b); }
}; // namespace std
#endif // MANYBODY_STATE_H
