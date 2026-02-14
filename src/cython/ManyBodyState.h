#ifndef MANYBODY_STATE_H
#define MANYBODY_STATE_H

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class ManyBodyState {

public:
  using Key = std::vector<uint64_t>;
  using Value = std::complex<double>;
  struct KeyComparer {
    bool operator()(const Key &a, const Key &b) const noexcept {
      for (size_t i = 0; i < a.size(); i++) {
        if (a[i] < b[i]) {
          return true;
        }
        if (a[i] > b[i]) {
          return false;
        }
      }
      return false;
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
  using Map = std::map<Key, Value, KeyComparer>;
  // using Values = std::deque<Value>;
  // using Keys = std::deque<Key>;
  using Values = std::vector<Value>;
  using Keys = std::vector<Key>;

private:
  // Map m_map;
  Keys m_keys;
  Values m_values;

public:
  using key_type = Key;
  using mapped_type = Value;
  using value_type = std::pair<Key, Value>;
  using size_type = Keys::size_type;
  using difference_type = Map::difference_type;
  using key_compare = KeyComparer;
  using value_compare = ValueComparer;
  using pointer = std::pair<Keys::pointer, Values::pointer>;
  using const_pointer = std::pair<Keys::const_pointer, Values::const_pointer>;
  using reference = std::pair<Key &, Value &>;
  using const_reference = std::pair<const Key &, const Value &>;
  using compare_type = KeyComparer;

  struct iterator {
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = ManyBodyState::value_type;
    using pointer = ManyBodyState::pointer;
    using const_pointer = ManyBodyState::const_pointer;
    using reference = ManyBodyState::reference;
    using const_referece = ManyBodyState::const_reference;

    iterator() : m_it() {}
    iterator(const Keys::iterator &k_it, const Values::iterator &v_it)
        : m_it(k_it, v_it) {}

    reference operator*() const {
      return {*this->m_it.first, *this->m_it.second};
    }
    std::pair<Keys::iterator, Values::iterator> *operator->() {
      return &this->m_it;
    };

    iterator &operator++() {
      this->m_it.first++;
      this->m_it.second++;
      return *this;
    }

    iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    iterator &operator+=(difference_type n) {
      this->m_it.first += n;
      this->m_it.second += n;
      return *this;
    }
    friend iterator operator+(const iterator &a, ptrdiff_t n) {
      iterator tmp = a;
      return tmp += n;
    }

    friend iterator operator+(ptrdiff_t n, const iterator &a) {
      iterator tmp = a;
      return tmp += n;
    }

    iterator &operator--() {
      this->m_it.first--;
      this->m_it.second--;
      return *this;
    }

    iterator operator--(int) {
      iterator tmp = *this;
      --(*this);
      return tmp;
    }

    iterator &operator-=(difference_type n) {
      this->m_it.first -= n;
      this->m_it.second -= n;
      return *this;
    }

    friend iterator operator-(const iterator &a, ptrdiff_t n) {
      iterator tmp = a;
      return tmp -= n;
    }

    difference_type operator-(const iterator &other) const {
      return this->m_it.first - other.m_it.first;
    }

    reference operator[](ptrdiff_t n) const { return *(*this + n); }

    friend bool operator==(const iterator &a, const iterator &b) {
      return a.m_it == b.m_it;
    }

    friend bool operator!=(const iterator &a, const iterator &b) {
      return !(a == b);
    }

    friend bool operator<(const iterator &a, const iterator &b) {
      return a.m_it.first < b.m_it.first;
    }
    friend bool operator>=(const iterator &a, const iterator &b) {
      return !(a.m_it.first < b.m_it.first);
    }
    friend bool operator>(const iterator &a, const iterator &b) {
      return a.m_it.first > b.m_it.first;
    }
    friend bool operator<=(const iterator &a, const iterator &b) {
      return !(a.m_it.first > b.m_it.first);
    }
    std::pair<Keys::iterator, Values::iterator> m_it;
  };

  struct const_iterator {
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    // using value_type = ManyBodyState::value_type;
    using value_type = std::pair<const Key, const Value>;
    using pointer = ManyBodyState::const_pointer;
    using const_pointer = ManyBodyState::const_pointer;
    using reference = ManyBodyState::const_reference;
    using const_referece = ManyBodyState::const_reference;

    explicit const_iterator() : m_it() {}
    explicit const_iterator(const Keys::iterator &k_it,
                            const Values::iterator &v_it)
        : m_it(Keys::const_iterator{k_it}, Values::const_iterator{v_it}) {}

    explicit const_iterator(const Keys::const_iterator &k_it,
                            const Values::const_iterator &v_it)
        : m_it(k_it, v_it) {}

    reference operator*() const {
      return {*this->m_it.first, *this->m_it.second};
    }
    const std::pair<Keys::const_iterator, Values::const_iterator> *
    operator->() {
      return &this->m_it;
    };

    const_iterator &operator++() {
      this->m_it.first++;
      this->m_it.second++;
      return *this;
    }

    const_iterator operator++(int) {
      const_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    const_iterator &operator+=(difference_type n) {
      this->m_it.first += n;
      this->m_it.second += n;
      return *this;
    }
    friend const_iterator operator+(const const_iterator &a, ptrdiff_t n) {
      const_iterator tmp = a;
      return tmp += n;
    }

    friend const_iterator operator+(ptrdiff_t n, const const_iterator &a) {
      const_iterator tmp = a;
      return tmp += n;
    }

    const_iterator &operator--() {
      this->m_it.first--;
      this->m_it.second--;
      return *this;
    }

    const_iterator operator--(int) {
      const_iterator tmp = *this;
      --(*this);
      return tmp;
    }

    const_iterator &operator-=(difference_type n) {
      this->m_it.first -= n;
      this->m_it.second -= n;
      return *this;
    }

    friend const_iterator operator-(const const_iterator &a, ptrdiff_t n) {
      const_iterator tmp = a;
      return tmp -= n;
    }

    difference_type operator-(const const_iterator &other) const {
      return this->m_it.first - other.m_it.first;
    }

    reference operator[](ptrdiff_t n) const { return *(*this + n); }

    friend bool operator==(const const_iterator &a, const const_iterator &b) {
      return a.m_it == b.m_it;
    }

    friend bool operator!=(const const_iterator &a, const const_iterator &b) {
      return !(a == b);
    }

    friend bool operator<(const const_iterator &a, const const_iterator &b) {
      return a.m_it.first < b.m_it.first;
    }
    friend bool operator>=(const const_iterator &a, const const_iterator &b) {
      return !(a.m_it.first < b.m_it.first);
    }
    friend bool operator>(const const_iterator &a, const const_iterator &b) {
      return a.m_it.first > b.m_it.first;
    }
    friend bool operator<=(const const_iterator &a, const const_iterator &b) {
      return !(a.m_it.first > b.m_it.first);
    }
    std::pair<Keys::const_iterator, Values::const_iterator> m_it;
  };

  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  explicit ManyBodyState() : m_keys(), m_values() {
    m_keys.reserve(1000);
    m_values.reserve(1000);
  };
  // ManyBodyState() = default;
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

  ManyBodyState &operator+=(const ManyBodyState &);
  ManyBodyState &operator-=(const ManyBodyState &);
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
    // return this->m_map == other.m_map;
    return self.m_keys == other.m_keys && self.m_values == other.m_values;
  }
  friend bool operator!=(const ManyBodyState &self,
                         const ManyBodyState &other) {
    return !(self == other);
  }
  //  allocator_type get_allocator() const { return m_map.get_allocator();
  // }
  Values::reference operator[](const Key &key) {
    auto it = this->lower_bound(key);
    if (it == this->end() || (*it).first != key) {
      it = this->insert(it, {key, Value{}});
    }
    return (*it).second;
  }
  Values::reference operator[](Key &&key) {
    auto it = this->lower_bound(key);
    if ((*it).first != key) {
      it = this->insert(it, {std::move(key), Value{}});
    }
    return (*it).second;
  }
  Values::reference at(const Key &key) {
    auto it = this->lower_bound(key);
    if ((*it).first != key) {
      throw std::out_of_range("Key not present in ManyBodyState");
    }
    return (*it).second;
  }
  Values::const_reference at(const Key &key) const {
    auto it = this->lower_bound(key);
    if ((*it).first != key) {
      throw std::out_of_range("Key not present in ManyBodyState");
    }
    return (*it).second;
  }

  bool empty() const { return m_values.empty(); }
  size_type size() const { return m_values.size(); }
  size_type max_size() const { return m_values.max_size(); }

  void clear() {
    m_values.clear();
    m_keys.clear();
  }
  void prune(double cutoff);
  void reserve(size_t capacity);

  iterator begin() {
    return iterator{this->m_keys.begin(), this->m_values.begin()};
  }
  const_iterator begin() const {
    return const_iterator(this->m_keys.begin(), this->m_values.begin());
  }
  const_iterator cbegin() const noexcept {
    return const_iterator{this->m_keys.cbegin(), this->m_values.cbegin()};
  }

  iterator end() { return iterator{this->m_keys.end(), this->m_values.end()}; }
  const_iterator end() const {
    return const_iterator{this->m_keys.end(), this->m_values.end()};
  }
  const_iterator cend() const noexcept {
    return const_iterator{this->m_keys.cend(), this->m_values.cend()};
  }

  reverse_iterator rbegin() { return reverse_iterator{this->end()}; }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(this->end());
  }

  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(this->cend());
  }

  reverse_iterator rend() { return reverse_iterator(this->begin()); }

  const_reverse_iterator rend() const {
    return const_reverse_iterator(this->begin());
  }

  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(this->cbegin());
  }
  std::pair<iterator, bool> insert(const value_type &val) {
    auto key_insert_it = std::lower_bound(
        this->m_keys.begin(), this->m_keys.end(), val.first, KeyComparer{});
    auto val_insert_it =
        this->m_values.begin() + (key_insert_it - this->m_keys.begin());
    if (*key_insert_it == val.first) {
      return std::pair<iterator, bool>{iterator{key_insert_it, val_insert_it},
                                       false};
    }

    key_insert_it = this->m_keys.insert(key_insert_it, val.first);
    val_insert_it = this->m_values.insert(val_insert_it, val.second);
    return std::pair<iterator, bool>{iterator{key_insert_it, val_insert_it},
                                     true};
  }

  std::pair<iterator, bool> insert(value_type &&val) {
    auto key_insert_it = std::lower_bound(
        this->m_keys.begin(), this->m_keys.end(), val.first, KeyComparer{});
    auto val_insert_it =
        this->m_values.begin() + (key_insert_it - this->m_keys.begin());
    if (*key_insert_it == val.first) {
      return std::pair<iterator, bool>{iterator{key_insert_it, val_insert_it},
                                       false};
    }

    key_insert_it = this->m_keys.insert(key_insert_it, std::move(val.first));
    val_insert_it = this->m_values.insert(val_insert_it, std::move(val.second));
    return std::pair<iterator, bool>{iterator{key_insert_it, val_insert_it},
                                     true};
  }

  iterator insert(iterator pos, const value_type &val) {
    return iterator{this->m_keys.insert(pos->first, val.first),
                    this->m_values.insert(pos->second, val.second)};
  }
  iterator insert(iterator pos, value_type &&val) {
    return iterator{this->m_keys.insert(pos->first, std::move(val.first)),
                    this->m_values.insert(pos->second, std::move(val.second))};
  }

  template <class InputIt>
  iterator insert(const_iterator pos, InputIt first, InputIt last) {
    return iterator{
        this->m_keys.insert(pos.m_it.first, first.m_it.first, last.m_it.first),
        this->m_values.insert(pos.m_it.second, first.m_it.second,
                              last.m_it.second)};
  }
  // iterator insert(const_iterator pos, std::initializer_list<value_type> l) {
  //   Values values = Values(l.size());
  //   Keys keys = Values(keys.size());
  //   for (auto &val : l) {
  //   }
  //   this->m_values.insert(pos.m_it.first, values.begin(), values.end());
  // }

  template <class... Args> std::pair<iterator, bool> emplace(Args &&...args) {
    return this->insert(value_type{std::forward<Args>(args)...});
  }
  iterator erase(iterator pos) {
    return iterator{this->m_keys.erase(pos->first),
                    this->m_values.erase(pos->second)};
  }
  iterator erase(const_iterator pos) {
    return iterator{this->m_keys.erase(pos->first),
                    this->m_values.erase(pos->second)};
  }

  iterator erase(iterator first, iterator last) {
    return iterator{this->m_keys.erase(first.m_it.first, last.m_it.first),
                    this->m_values.erase(first.m_it.second, last.m_it.second)};
  }

  const_iterator erase(const_iterator first, const_iterator last) {
    return const_iterator{
        this->m_keys.erase(first.m_it.first, last.m_it.first),
        this->m_values.erase(first.m_it.second, last.m_it.second)};
  }
  iterator erase(const key_type &key) {
    auto key_insert_it = std::lower_bound(
        this->m_keys.begin(), this->m_keys.end(), key, KeyComparer{});
    auto val_insert_it =
        this->m_values.begin() + (key_insert_it - this->m_keys.begin());
    if (*key_insert_it == key) {
      return this->erase(iterator({key_insert_it, val_insert_it}));
    }
    return this->end();
  }

  void swap(ManyBodyState &other) noexcept {
    this->m_keys.swap(other.m_keys);
    this->m_values.swap(other.m_values);
  }

  iterator find(const key_type &key) {
    auto key_insert_it = std::lower_bound(
        this->m_keys.begin(), this->m_keys.end(), key, KeyComparer{});
    auto val_insert_it =
        this->m_values.begin() + (key_insert_it - this->m_keys.begin());
    if (key_insert_it == this->m_keys.end()) {
      return this->end();
    }
    if (*key_insert_it == key) {
      return iterator({key_insert_it, val_insert_it});
    }
    return this->end();
  }

  const_iterator find(const key_type &key) const {
    auto key_insert_it = std::lower_bound(
        this->m_keys.cbegin(), this->m_keys.cend(), key, KeyComparer{});
    auto val_insert_it =
        this->m_values.cbegin() + (key_insert_it - this->m_keys.cbegin());
    if (key_insert_it == this->m_keys.end()) {
      return this->end();
    }
    if (*key_insert_it == key) {
      return const_iterator{key_insert_it, val_insert_it};
    }
    return this->cend();
  }

  template <class K> iterator find(const K &key) {
    auto key_insert_it = std::lower_bound(
        this->m_keys.begin(), this->m_keys.end(), key, KeyComparer{});
    auto val_insert_it =
        this->m_values.begin() + (key_insert_it - this->m_keys.begin());
    if (key_insert_it == this->m_keys.end()) {
      return this->end();
    }
    if (*key_insert_it == key) {
      return iterator({key_insert_it, val_insert_it});
    }
    return this->end();
  }

  template <class K> const_iterator find(const K &key) const {
    auto key_insert_it = std::lower_bound(
        this->m_keys.cbegin(), this->m_keys.cend(), key, KeyComparer{});
    auto val_insert_it =
        this->m_values.cbegin() + (key_insert_it - this->m_keys.cbegin());
    if (key_insert_it == this->m_keys.end()) {
      return this->end();
    }
    if (*key_insert_it == key) {
      return const_iterator({key_insert_it, val_insert_it});
    }
    return this->cend();
  }

  iterator lower_bound(const key_type &key) {
    auto key_insert_it = std::lower_bound(
        this->m_keys.begin(), this->m_keys.end(), key, KeyComparer{});
    return this->begin() + (key_insert_it - this->m_keys.begin());
  }

  const_iterator lower_bound(const key_type &key) const {
    auto key_insert_it = std::lower_bound(
        this->m_keys.cbegin(), this->m_keys.cend(), key, KeyComparer{});
    return this->cbegin() + (key_insert_it - this->m_keys.cbegin());
  }

  template <class K> iterator lower_bound(const K &key) {
    auto key_insert_it = std::lower_bound(
        this->m_keys.begin(), this->m_keys.end(), key, KeyComparer{});
    return this->begin() + (key_insert_it - this->m_keys.begin());
  }
  template <class K> const_iterator lower_bound(const K &key) const {
    auto key_insert_it = std::lower_bound(
        this->m_keys.cbegin(), this->m_keys.cend(), key, KeyComparer{});
    return this->cbegin() + (key_insert_it - this->m_keys.cbegin());
  }

  iterator upper_bound(const key_type &key) {
    auto key_insert_it = std::upper_bound(
        this->m_keys.begin(), this->m_keys.end(), key, KeyComparer{});
    auto val_insert_it =
        this->m_values.begin() + (key_insert_it - this->m_keys.begin());
    return iterator({key_insert_it, val_insert_it});
  }

  const_iterator upper_bound(const key_type &key) const {
    auto key_insert_it = std::upper_bound(
        this->m_keys.begin(), this->m_keys.end(), key, KeyComparer{});
    auto val_insert_it =
        this->m_values.begin() + (key_insert_it - this->m_keys.cbegin());
    return const_iterator(key_insert_it, val_insert_it);
  }

  template <class K> iterator upper_bound(const K &key) {
    auto key_insert_it = std::upper_bound(
        this->m_keys.begin(), this->m_keys.end(), key, KeyComparer{});
    auto val_insert_it =
        this->m_values.begin() + (key_insert_it - this->m_keys.begin());
    return iterator({key_insert_it, val_insert_it});
  }

  template <class K> const_iterator upper_bound(const K &key) const {
    auto key_insert_it = std::upper_bound(
        this->m_keys.cbegin(), this->m_keys.cend(), key, KeyComparer{});
    auto val_insert_it =
        this->m_values.cbegin() + (key_insert_it - this->m_keys.cbegin());
    return const_iterator({key_insert_it, val_insert_it});
  }
  std::string to_string() const;
};

namespace std {
inline void swap(ManyBodyState &a, ManyBodyState &b) noexcept { a.swap(b); }
#if __cplusplus >= 202302L
template <template <class> class TQual, template <class> class UQual>
struct basic_common_reference<ManyBodyState::const_reference,
                              ManyBodyState::value_type, TQual, UQual> {
  using type = ManyBodyState::value_type;
};

template <template <class> class TQual, template <class> class UQual>
struct basic_common_reference<ManyBodyState::value_type,
                              ManyBodyState::const_reference, TQual, UQual> {
  using type = ManyBodyState::value_type;
};
#endif
}; // namespace std
#endif // MANYBODY_STATE_H
