#ifndef MANYBODYOPERATOR_H
#define MANYBODYOPERATOR_H

#include "ManyBodyState.h"
#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

class ManyBodyOperator {

public:
  template <typename T> struct Comparer {
    bool operator()(const std::vector<T> &a,
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
  using SLATER = ManyBodyState::key_type;
  using Restrictions =
      std::vector<std::pair<std::vector<size_t>, std::pair<size_t, size_t>>>;

private:
  std::vector<std::pair<OPS, SCALAR>> m_ops;

  [[nodiscard]] std::tuple<std::vector<ManyBodyState::key_type>,
                           std::vector<size_t>, std::vector<size_t>>
  build_restriction_mask(const Restrictions &restrictions) const noexcept;

  [[nodiscard]] static bool state_is_within_restrictions(
      const ManyBodyState::key_type &,
      const std::tuple<std::vector<ManyBodyState::key_type>,
                       std::vector<size_t>, std::vector<size_t>> &) noexcept;

  [[nodiscard]] ManyBodyState apply_op_determinant(
      const ManyBodyState::Key &slater_determinant,
      const std::tuple<std::vector<ManyBodyState::key_type>,
                       std::vector<size_t>, std::vector<size_t>> &)
      const noexcept;

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

  [[nodiscard]] ManyBodyState operator()(
      const ManyBodyState &psi, double cutoff = 0,
      const ManyBodyOperator::Restrictions &restrictions = {}) const noexcept {
    return apply(psi, cutoff, restrictions);
  }

  [[nodiscard]] std::vector<ManyBodyState> operator()(
      const std::vector<ManyBodyState> &psis, double cutoff = 0,
      const ManyBodyOperator::Restrictions &restrictions = {}) const noexcept {
    return apply(psis, cutoff, restrictions);
  }

  [[nodiscard]] ManyBodyState
  apply(const ManyBodyState &, double cutoff = 0,
        const ManyBodyOperator::Restrictions &restrictions = {}) const noexcept;

  [[nodiscard]] std::vector<ManyBodyState>
  apply(const std::vector<ManyBodyState> &, double cutoff = 0,
        const ManyBodyOperator::Restrictions &restrictions = {}) const noexcept;

  [[nodiscard]] size_type size() const noexcept;
  [[nodiscard]] bool empty() const noexcept;
  bool clear();

  ManyBodyOperator &operator+=(const ManyBodyOperator &) noexcept;
  ManyBodyOperator &operator-=(const ManyBodyOperator &) noexcept;
  ManyBodyOperator &operator*=(const mapped_type &) noexcept;
  ManyBodyOperator &operator/=(const mapped_type &) noexcept;
  [[nodiscard]] ManyBodyOperator operator-() const noexcept;

  [[nodiscard]] ManyBodyOperator
  operator+(const ManyBodyOperator &other) const {
    ManyBodyOperator res(*this);
    return res += other;
  }

  [[nodiscard]] ManyBodyOperator
  operator-(const ManyBodyOperator &other) const {
    ManyBodyOperator res(*this);
    return res -= other;
  }

  [[nodiscard]] ManyBodyOperator operator*(const mapped_type &s) const {
    ManyBodyOperator res(*this);
    return res *= s;
  }

  [[nodiscard]] friend ManyBodyOperator operator*(const mapped_type &s,
                                                  const ManyBodyOperator &o) {
    return o * s;
  }

  [[nodiscard]] ManyBodyOperator operator/(const mapped_type &s) const {
    ManyBodyOperator res(*this);
    return res /= s;
  }

  [[nodiscard]] bool operator==(const ManyBodyOperator &other) const {
    return this->m_ops == other.m_ops;
  }
  [[nodiscard]] bool operator!=(const ManyBodyOperator &other) const {
    return !(*this == other);
  }
  mapped_type &operator[](const key_type &key) noexcept;
  mapped_type &operator[](key_type &&key) noexcept;

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

  void swap(ManyBodyOperator &other) noexcept { m_ops.swap(other.m_ops); }

  [[nodiscard]] iterator find(const key_type &);
  [[nodiscard]] iterator find(iterator, iterator, const key_type &);
  [[nodiscard]] const_iterator find(const key_type &key) const;
  [[nodiscard]] const_iterator find(const_iterator, const_iterator,
                                    const key_type &) const;
  template <class K> [[nodiscard]] iterator find(const K &);
  template <class K> [[nodiscard]] iterator find(iterator, iterator, const K &);
  template <class K> [[nodiscard]] const_iterator find(const K &) const;
  template <class K>
  [[nodiscard]] const_iterator find(const_iterator, const_iterator,
                                    const K &) const;

  [[nodiscard]] iterator lower_bound(const key_type &);
  [[nodiscard]] const_iterator lower_bound(const key_type &) const;
  template <class K> [[nodiscard]] iterator lower_bound(const K &);
  template <class K> [[nodiscard]] const_iterator lower_bound(const K &) const;

  [[nodiscard]] iterator upper_bound(const key_type &ey);
  [[nodiscard]] const_iterator upper_bound(const key_type &ey) const;
  template <class K> [[nodiscard]] iterator upper_bound(const K &);
  template <class K> [[nodiscard]] const_iterator upper_bound(const K &) const;

  //  compare_type key_comp() const { return compare_type(); }

  [[nodiscard]] iterator begin() { return m_ops.begin(); }
  [[nodiscard]] const_iterator begin() const { return m_ops.begin(); }
  [[nodiscard]] const_iterator cbegin() const noexcept {
    return m_ops.cbegin();
  }
  [[nodiscard]] iterator end() { return m_ops.end(); }
  [[nodiscard]] const_iterator end() const { return m_ops.end(); }
  [[nodiscard]] const_iterator cend() const noexcept { return m_ops.cend(); }
  [[nodiscard]] reverse_iterator rbegin() { return m_ops.rbegin(); }
  [[nodiscard]] const_reverse_iterator rbegin() const { return m_ops.rbegin(); }
  [[nodiscard]] const_reverse_iterator crbegin() const noexcept {
    return m_ops.crbegin();
  }
  [[nodiscard]] reverse_iterator rend() { return m_ops.rend(); }
  [[nodiscard]] const_reverse_iterator rend() const { return m_ops.rend(); }
  [[nodiscard]] const_reverse_iterator crend() const noexcept {
    return m_ops.crend();
  }
};
#endif // MANYBODYOPERATOR_H
