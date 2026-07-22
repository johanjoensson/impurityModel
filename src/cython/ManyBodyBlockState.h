#ifndef MANYBODY_BLOCK_STATE_H
#define MANYBODY_BLOCK_STATE_H

#include "SlaterDeterminant.h"

#include <algorithm>
#include <bitset>
#include <complex>
#include <cstddef>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if __cplusplus >= 202002L && __has_include(<span>)
#include <span>
#define MANYBODY_HAVE_STD_SPAN 1
#endif

#if defined(MANYBODY_HAVE_STD_SPAN)
/** @brief A view over one determinant's amplitudes; std::span where available. */
template <typename T> using RowSpan = std::span<T>;
#else
/**
 * @brief Minimal std::span stand-in (the build defaults to -std=c++17).
 *
 * Only the subset the state layer uses: contiguous view, indexing, iteration,
 * and the non-const -> const conversion. Deliberately non-owning; see the
 * invalidation rule on ManyBodyBlockState::row.
 */
template <typename T> class RowSpan {
public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using pointer = T *;
  using reference = T &;
  using iterator = T *;

  constexpr RowSpan() noexcept = default;
  constexpr RowSpan(pointer data, size_type size) noexcept
      : m_data(data), m_size(size) {}

  /** Non-const -> const conversion, mirroring std::span. */
  template <typename U,
            typename = std::enable_if_t<std::is_same_v<const U, T>>>
  constexpr RowSpan(const RowSpan<U> &other) noexcept
      : m_data(other.data()), m_size(other.size()) {}

  constexpr pointer data() const noexcept { return m_data; }
  constexpr size_type size() const noexcept { return m_size; }
  constexpr bool empty() const noexcept { return m_size == 0; }
  constexpr reference operator[](size_type i) const { return m_data[i]; }
  constexpr reference front() const { return m_data[0]; }
  constexpr reference back() const { return m_data[m_size - 1]; }
  constexpr iterator begin() const noexcept { return m_data; }
  constexpr iterator end() const noexcept { return m_data + m_size; }

private:
  pointer m_data{nullptr};
  size_type m_size{0};
};
#endif

/**
 * @class ManyBodyBlockState
 * @brief A block of p many-body vectors over ONE shared Slater-determinant
 * support.
 *
 * Stores the union support as a sorted, strictly-increasing key vector and the
 * coefficients as a row-major (rows() x width()) dense array: row r holds the p
 * amplitudes of determinant key(r), one per block vector. This is the hot-loop
 * (block Lanczos / Green's function) counterpart of ManyBodyState: the shared
 * support lets ManyBodyOperator::apply do the term loop, sign work, restriction
 * checks and accumulator hashing once per (determinant, term) and emit p scaled
 * amplitudes, and it lets block inner products / axpy run as dense row-block
 * BLAS when two blocks share a support. ManyBodyState remains the single-vector
 * boundary type everywhere outside the hot loop.
 *
 * Invariants: m_keys is sorted and unique; m_amps.size() == m_keys.size() *
 * m_width. The block width is a runtime value (no compile-time bound).
 */
class ManyBodyBlockState {
public:
  using Key = SlaterDeterminant<>;
  using Value = std::complex<double>;
  /** @brief One determinant's `width()` amplitudes; see the invalidation rule on row(). */
  using Row = RowSpan<Value>;
  using ConstRow = RowSpan<const Value>;

  // flat_map-compatible spellings, so the map surface below reads the same as
  // the container this class replaces.
  using key_type = Key;
  using mapped_type = Value;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

private:
  std::vector<Key> m_keys;   // sorted, strictly increasing
  std::vector<Value> m_amps; // row-major, m_keys.size() * m_width
  std::size_t m_width{0};

public:
  /**
   * @brief Iterator over row ENTRIES: `*it` is `(key, amplitudes-of-that-key)`.
   *
   * The pair is built on dereference rather than stored -- keys and amplitudes
   * live in separate arrays -- so it is returned by value (a proxy reference)
   * and `operator->` goes through the standard arrow proxy. Random access, so
   * `std::advance` / `std::distance` stay O(1) (the threaded apply relies on
   * advancing into the middle of a state).
   */
  template <bool Const> class BasicIterator {
  public:
    using StateType =
        std::conditional_t<Const, const ManyBodyBlockState, ManyBodyBlockState>;
    using RowType = std::conditional_t<Const, ConstRow, Row>;
    using value_type = std::pair<const Key &, RowType>;
    using reference = value_type; // proxy: materialized on dereference
    using pointer = void;         // use operator-> (arrow proxy) instead
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    /** @brief Keeps the proxy pair alive for the duration of `it->member`. */
    struct ArrowProxy {
      value_type entry;
      const value_type *operator->() const noexcept { return &entry; }
    };

    BasicIterator() noexcept = default;
    BasicIterator(StateType *state, difference_type pos) noexcept
        : m_state(state), m_pos(pos) {}
    /** @brief Mutable -> const conversion, mirroring the standard containers. */
    template <bool OtherConst,
              typename = std::enable_if_t<Const && !OtherConst>>
    BasicIterator(const BasicIterator<OtherConst> &other) noexcept
        : m_state(other.state()), m_pos(other.pos()) {}

    StateType *state() const noexcept { return m_state; }
    difference_type pos() const noexcept { return m_pos; }

    reference operator*() const {
      const auto r = static_cast<std::size_t>(m_pos);
      return value_type(m_state->key(r), m_state->row(r));
    }
    ArrowProxy operator->() const { return ArrowProxy{**this}; }
    reference operator[](difference_type n) const { return *(*this + n); }

    BasicIterator &operator++() noexcept { ++m_pos; return *this; }
    BasicIterator operator++(int) noexcept { auto t = *this; ++m_pos; return t; }
    BasicIterator &operator--() noexcept { --m_pos; return *this; }
    BasicIterator operator--(int) noexcept { auto t = *this; --m_pos; return t; }
    BasicIterator &operator+=(difference_type n) noexcept { m_pos += n; return *this; }
    BasicIterator &operator-=(difference_type n) noexcept { m_pos -= n; return *this; }

    friend BasicIterator operator+(BasicIterator it, difference_type n) noexcept { return it += n; }
    friend BasicIterator operator+(difference_type n, BasicIterator it) noexcept { return it += n; }
    friend BasicIterator operator-(BasicIterator it, difference_type n) noexcept { return it -= n; }
    friend difference_type operator-(const BasicIterator &a, const BasicIterator &b) noexcept {
      return a.m_pos - b.m_pos;
    }
    friend bool operator==(const BasicIterator &a, const BasicIterator &b) noexcept { return a.m_pos == b.m_pos; }
    friend bool operator!=(const BasicIterator &a, const BasicIterator &b) noexcept { return a.m_pos != b.m_pos; }
    friend bool operator<(const BasicIterator &a, const BasicIterator &b) noexcept { return a.m_pos < b.m_pos; }
    friend bool operator>(const BasicIterator &a, const BasicIterator &b) noexcept { return a.m_pos > b.m_pos; }
    friend bool operator<=(const BasicIterator &a, const BasicIterator &b) noexcept { return a.m_pos <= b.m_pos; }
    friend bool operator>=(const BasicIterator &a, const BasicIterator &b) noexcept { return a.m_pos >= b.m_pos; }

  private:
    StateType *m_state{nullptr};
    difference_type m_pos{0};
  };

  using iterator = BasicIterator<false>;
  using const_iterator = BasicIterator<true>;

  ManyBodyBlockState() = default;
  ManyBodyBlockState(const ManyBodyBlockState &) = default;
  ManyBodyBlockState(ManyBodyBlockState &&) noexcept = default;
  ManyBodyBlockState &operator=(const ManyBodyBlockState &) = default;
  ManyBodyBlockState &operator=(ManyBodyBlockState &&) noexcept = default;
  ~ManyBodyBlockState() = default;

  /** @brief Empty state of the given width (no rows). */
  explicit ManyBodyBlockState(std::size_t width) : m_width(width) {}

  /**
   * @brief Adopt pre-built storage. `keys` must be sorted and unique and
   * `amps.size() == keys.size() * width` (unchecked in release builds).
   */
  ManyBodyBlockState(std::vector<Key> keys, std::vector<Value> amps,
                     std::size_t width)
      : m_keys(std::move(keys)), m_amps(std::move(amps)), m_width(width) {}

  /**
   * @brief Width-1 state from parallel (key, amplitude) arrays in ANY order.
   *
   * Sorts into the row order this container maintains; on duplicate keys the
   * first occurrence wins, matching the flat_map range-insert this replaces.
   */
  ManyBodyBlockState(const std::vector<Key> &keys,
                     const std::vector<Value> &values);

  std::size_t width() const noexcept { return m_width; }
  std::size_t rows() const noexcept { return m_keys.size(); }
  /** @brief Number of stored determinants -- rows(), spelled for the map surface. */
  size_type size() const noexcept { return m_keys.size(); }
  bool empty() const noexcept { return m_keys.empty(); }

  const Key &key(std::size_t r) const { return m_keys[r]; }
  const std::vector<Key> &keys() const noexcept { return m_keys; }
  Value *data() noexcept { return m_amps.data(); }
  const Value *data() const noexcept { return m_amps.data(); }

  /**
   * @brief The amplitudes of row `r`.
   *
   * Non-owning: invalidated by anything that reallocates the storage
   * (prune_rows, keep_rows, merge_keys, erase, and inserting a new key).
   */
  Row row(std::size_t r) noexcept {
    return Row(m_amps.data() + r * m_width, m_width);
  }
  ConstRow row(std::size_t r) const noexcept {
    return ConstRow(m_amps.data() + r * m_width, m_width);
  }

  iterator begin() noexcept { return iterator(this, 0); }
  iterator end() noexcept {
    return iterator(this, static_cast<difference_type>(rows()));
  }
  const_iterator begin() const noexcept { return const_iterator(this, 0); }
  const_iterator end() const noexcept {
    return const_iterator(this, static_cast<difference_type>(rows()));
  }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator cend() const noexcept { return end(); }

  /** @brief Row index of `k`, or rows() when absent (binary search). */
  std::size_t find_row(const Key &k) const noexcept {
    auto it = std::lower_bound(m_keys.begin(), m_keys.end(), k);
    if (it != m_keys.end() && *it == k) {
      return static_cast<std::size_t>(it - m_keys.begin());
    }
    return m_keys.size();
  }

  // --- map surface -------------------------------------------------------
  // Lookup is a binary search over the sorted keys; insertion of a NEW key
  // shifts the tail of both arrays, so building a state key-by-key is O(n^2)
  // -- construct from sorted/bulk input instead.

  iterator find(const Key &k) noexcept {
    return iterator(this, static_cast<difference_type>(find_row(k)));
  }
  const_iterator find(const Key &k) const noexcept {
    return const_iterator(this, static_cast<difference_type>(find_row(k)));
  }
  bool contains(const Key &k) const noexcept { return find_row(k) != rows(); }

  Row at(const Key &k) {
    const std::size_t r = find_row(k);
    if (r == rows()) {
      throw std::out_of_range("ManyBodyBlockState::at: determinant not present");
    }
    return row(r);
  }
  ConstRow at(const Key &k) const {
    const std::size_t r = find_row(k);
    if (r == rows()) {
      throw std::out_of_range("ManyBodyBlockState::at: determinant not present");
    }
    return row(r);
  }

  /** @brief Row of `k`, inserting a zero row when absent (O(rows) then). */
  Row operator[](const Key &k) { return row(insert_row(k).first); }

  /**
   * @brief Row index of `k`, inserting a zero row when absent.
   * @return (row index, true if a row was inserted).
   */
  std::pair<std::size_t, bool> insert_row(const Key &k) {
    auto it = std::lower_bound(m_keys.begin(), m_keys.end(), k);
    const auto pos = static_cast<std::size_t>(it - m_keys.begin());
    if (it != m_keys.end() && *it == k) {
      return {pos, false};
    }
    m_keys.insert(it, k);
    m_amps.insert(m_amps.begin() +
                      static_cast<difference_type>(pos * m_width),
                  m_width, Value{0.0, 0.0});
    return {pos, true};
  }

  /** @brief Remove the row of `k`; returns how many rows went (0 or 1). */
  size_type erase(const Key &k) {
    const std::size_t r = find_row(k);
    if (r == rows()) {
      return 0;
    }
    erase_row(r);
    return 1;
  }

  void erase_row(std::size_t r) {
    m_keys.erase(m_keys.begin() + static_cast<difference_type>(r));
    const auto first = m_amps.begin() + static_cast<difference_type>(r * m_width);
    m_amps.erase(first, first + static_cast<difference_type>(m_width));
  }

  void clear() noexcept {
    m_keys.clear();
    m_amps.clear();
  }

  void reserve(size_type n) {
    m_keys.reserve(n);
    m_amps.reserve(n * m_width);
  }

  void swap(ManyBodyBlockState &other) noexcept {
    m_keys.swap(other.m_keys);
    m_amps.swap(other.m_amps);
    std::swap(m_width, other.m_width);
  }

  // --- vector space ------------------------------------------------------
  // Widths must match, except that a row-less operand is the additive identity
  // and adopts the other's width (so summing onto a default-constructed state
  // works regardless of the block width).

  /** @brief Frobenius norm squared: the sum over every stored amplitude. */
  double norm2() const noexcept {
    double res = 0.0;
    for (const Value &v : m_amps) {
      res += std::norm(v);
    }
    return res;
  }
  double norm() const noexcept { return std::sqrt(norm2()); }

  /** @brief this += scale * other, over the union support. */
  ManyBodyBlockState &add_scaled(const ManyBodyBlockState &other, Value scale);

  ManyBodyBlockState &operator+=(const ManyBodyBlockState &other) {
    return add_scaled(other, Value{1.0, 0.0});
  }
  ManyBodyBlockState &operator-=(const ManyBodyBlockState &other) {
    return add_scaled(other, Value{-1.0, 0.0});
  }
  ManyBodyBlockState &operator*=(Value s) {
    for (Value &v : m_amps) {
      v *= s;
    }
    return *this;
  }
  ManyBodyBlockState &operator/=(Value s) {
    for (Value &v : m_amps) {
      v /= s;
    }
    return *this;
  }
  ManyBodyBlockState operator-() const {
    ManyBodyBlockState res(*this);
    return res *= Value{-1.0, 0.0};
  }

  friend ManyBodyBlockState operator+(const ManyBodyBlockState &a,
                                      const ManyBodyBlockState &b) {
    return ManyBodyBlockState{a} += b;
  }
  friend ManyBodyBlockState operator-(const ManyBodyBlockState &a,
                                      const ManyBodyBlockState &b) {
    return ManyBodyBlockState{a} -= b;
  }
  friend ManyBodyBlockState operator*(const ManyBodyBlockState &a, Value s) {
    return ManyBodyBlockState{a} *= s;
  }
  friend ManyBodyBlockState operator*(Value s, const ManyBodyBlockState &a) {
    return ManyBodyBlockState{a} *= s;
  }
  friend ManyBodyBlockState operator/(const ManyBodyBlockState &a, Value s) {
    return ManyBodyBlockState{a} /= s;
  }

  /** @brief Largest |amplitude|^2 anywhere in the block. */
  double max_norm2() const noexcept {
    double res = 0.0;
    for (const Value &v : m_amps) {
      res = std::max(res, std::norm(v));
    }
    return res;
  }

  /** @brief Rows whose largest column |amplitude|^2 exceeds `cutoff2`. */
  size_type count_above(double cutoff2) const noexcept {
    size_type count = 0;
    for (std::size_t r = 0; r < rows(); ++r) {
      if (row_max2(r) > cutoff2) {
        ++count;
      }
    }
    return count;
  }

  /**
   * @brief Keep the `max_rows` rows with the largest row-max |amplitude|^2.
   *
   * Ties at the cutoff are all kept, so the result can exceed `max_rows` -- the
   * behaviour of the flat_map truncate this replaces. `max_rows == 0` is a no-op.
   */
  void truncate(std::size_t max_rows);

  /** @brief String representation of the state, for debugging. */
  std::string to_string() const;

  /**
   * @brief Drop every row whose amplitudes ALL satisfy the ManyBodyState::prune
   * test (|amp|^2 <= cutoff^2): a row survives if ANY column survives. Keeping
   * whole rows preserves the shared support across the block — the deliberate
   * semantic difference vs per-column pruning of independent states.
   */
  void prune_rows(double cutoff) {
    const double cutoff2 = cutoff * cutoff;
    std::size_t out = 0;
    for (std::size_t r = 0; r < rows(); ++r) {
      bool keep = false;
      const ConstRow src = row(r);
      for (const Value &v : src) {
        if (std::norm(v) > cutoff2) {
          keep = true;
          break;
        }
      }
      if (keep) {
        if (out != r) {
          m_keys[out] = std::move(m_keys[r]);
          std::copy(src.begin(), src.end(), m_amps.data() + out * m_width);
        }
        ++out;
      }
    }
    m_keys.resize(out);
    m_amps.resize(out * m_width);
  }

  /**
   * @brief Keep only rows whose key appears in `keep` (sorted, strictly
   * increasing). Linear merge over the two sorted sequences — the
   * set-intersection complement of prune_rows, used by the capped
   * Green's-function recurrence to project a block onto the retained
   * determinant set.
   */
  void keep_rows(const std::vector<Key> &keep) {
    std::size_t out = 0;
    std::size_t ik = 0;
    for (std::size_t r = 0; r < rows(); ++r) {
      while (ik < keep.size() && keep[ik] < m_keys[r]) {
        ++ik;
      }
      if (ik < keep.size() && keep[ik] == m_keys[r]) {
        if (out != r) {
          const ConstRow src = row(r);
          m_keys[out] = std::move(m_keys[r]);
          std::copy(src.begin(), src.end(), m_amps.data() + out * m_width);
        }
        ++out;
        ++ik;
      }
    }
    m_keys.resize(out);
    m_amps.resize(out * m_width);
  }

  /** @brief Per-row max column |amp|^2 into out[0..rows()). */
  void row_max_norm2(double *out) const noexcept {
    for (std::size_t r = 0; r < rows(); ++r) {
      out[r] = row_max2(r);
    }
  }

  /** @brief Row max |amp|^2 (helper for the capped-recurrence primitives). */
  double row_max2(std::size_t r) const noexcept {
    double m = 0.0;
    for (const Value &v : row(r)) {
      m = std::max(m, std::norm(v));
    }
    return m;
  }

  /** @brief Number of rows whose key appears in `keep` (sorted, unique). */
  std::size_t count_rows_in(const std::vector<Key> &keep) const noexcept {
    std::size_t n = 0;
    std::size_t ik = 0;
    for (std::size_t r = 0; r < rows(); ++r) {
      while (ik < keep.size() && keep[ik] < m_keys[r]) {
        ++ik;
      }
      if (ik < keep.size() && keep[ik] == m_keys[r]) {
        ++n;
        ++ik;
      }
    }
    return n;
  }

  /**
   * @brief Max |amp|^2 of every row whose key is NOT in `keep`, appended to
   * `out` in row order. With count_rows_in this gives the candidate-importance
   * array for the capped recurrence's overflow-step ranking without any
   * per-row Python traffic.
   */
  void new_row_max_norm2(const std::vector<Key> &keep,
                         std::vector<double> &out) const {
    std::size_t ik = 0;
    for (std::size_t r = 0; r < rows(); ++r) {
      while (ik < keep.size() && keep[ik] < m_keys[r]) {
        ++ik;
      }
      if (ik < keep.size() && keep[ik] == m_keys[r]) {
        ++ik;
      } else {
        out.push_back(row_max2(r));
      }
    }
  }

  /**
   * @brief Width-0 key-only block of the rows NOT in `keep` (sorted, unique)
   * whose max |amp|^2 exceeds `cutoff2` — the admitted boundary determinants
   * once the overflow bisection has fixed the amplitude cutoff.
   */
  ManyBodyBlockState keys_new_above(const std::vector<Key> &keep,
                                    double cutoff2) const {
    std::vector<Key> out;
    std::size_t ik = 0;
    for (std::size_t r = 0; r < rows(); ++r) {
      while (ik < keep.size() && keep[ik] < m_keys[r]) {
        ++ik;
      }
      if (ik < keep.size() && keep[ik] == m_keys[r]) {
        ++ik;
      } else if (row_max2(r) > cutoff2) {
        out.push_back(m_keys[r]);
      }
    }
    return ManyBodyBlockState(std::move(out), {}, 0);
  }

  /**
   * @brief Width-0 key-only block holding the sorted union of this block's and
   * `other`'s keys (amplitudes of both are ignored). Used as the retained-set
   * mask of the capped recurrence: merge each admitted step's support, then
   * project later blocks with keep_rows.
   */
  ManyBodyBlockState key_union(const ManyBodyBlockState &other) const {
    std::vector<Key> out;
    out.reserve(rows() + other.rows());
    std::set_union(m_keys.begin(), m_keys.end(), other.m_keys.begin(),
                   other.m_keys.end(), std::back_inserter(out));
    return ManyBodyBlockState(std::move(out), {}, 0);
  }

  /**
   * @brief In-place key_union for width-0 mask blocks: append `other`'s keys
   * not already present, then inplace_merge. Copies only the genuinely new
   * keys (the existing ones are moved, not reallocated) — the per-step
   * retained-mask accumulate of the capped recurrence. Requires width() == 0
   * (amplitude storage must stay empty); checked by the Cython wrapper.
   */
  void merge_keys(const ManyBodyBlockState &other) {
    std::vector<Key> add;
    std::size_t ik = 0;
    for (std::size_t r = 0; r < other.rows(); ++r) {
      while (ik < m_keys.size() && m_keys[ik] < other.m_keys[r]) {
        ++ik;
      }
      if (ik >= m_keys.size() || other.m_keys[r] < m_keys[ik]) {
        add.push_back(other.m_keys[r]);
      } else {
        ++ik;
      }
    }
    if (add.empty()) {
      return;
    }
    const auto mid = static_cast<std::ptrdiff_t>(m_keys.size());
    m_keys.insert(m_keys.end(), std::make_move_iterator(add.begin()),
                  std::make_move_iterator(add.end()));
    std::inplace_merge(m_keys.begin(), m_keys.begin() + mid, m_keys.end());
  }

  /** @brief Per-column sum of |amp|^2 into out[0..width). */
  void col_norm2(double *out) const noexcept {
    for (std::size_t c = 0; c < m_width; ++c) {
      out[c] = 0.0;
    }
    for (std::size_t r = 0; r < rows(); ++r) {
      const ConstRow src = row(r);
      for (std::size_t c = 0; c < m_width; ++c) {
        out[c] += std::norm(src[c]);
      }
    }
  }

  friend bool operator==(const ManyBodyBlockState &a,
                         const ManyBodyBlockState &b) {
    return a.m_width == b.m_width && a.m_keys == b.m_keys &&
           a.m_amps == b.m_amps;
  }
  friend bool operator!=(const ManyBodyBlockState &a,
                         const ManyBodyBlockState &b) {
    return !(a == b);
  }
};

inline ManyBodyBlockState::ManyBodyBlockState(const std::vector<Key> &keys,
                                              const std::vector<Value> &values)
    : m_width(1) {
  const std::size_t n = keys.size();
  std::vector<std::size_t> order(n);
  for (std::size_t i = 0; i < n; ++i) {
    order[i] = i;
  }
  // Stable, so that among equal keys the earliest input entry sorts first and
  // the dedup below keeps it -- the flat_map range-insert semantics.
  std::stable_sort(order.begin(), order.end(),
                   [&keys](std::size_t a, std::size_t b) {
                     return keys[a] < keys[b];
                   });
  m_keys.reserve(n);
  m_amps.reserve(n);
  for (const std::size_t i : order) {
    if (!m_keys.empty() && m_keys.back() == keys[i]) {
      continue;
    }
    m_keys.push_back(keys[i]);
    m_amps.push_back(values[i]);
  }
}

inline ManyBodyBlockState &
ManyBodyBlockState::add_scaled(const ManyBodyBlockState &other, Value scale) {
  if (other.rows() == 0 || scale == Value{0.0, 0.0}) {
    return *this;
  }
  // A width-0, row-less state is the polymorphic zero (what a default
  // construction gives) and adopts the other operand's width, so summing onto
  // it works at any block width. A state with an explicit width stays strict --
  // silently widening that would hide a genuine width mismatch.
  if (rows() == 0 && m_width == 0) {
    *this = other;
    return *this *= scale;
  }
  if (m_width != other.m_width) {
    throw std::invalid_argument(
        "ManyBodyBlockState::add_scaled: block widths differ");
  }
  if (rows() == 0) {
    *this = other;
    return *this *= scale;
  }
  // Linear merge over the two sorted supports into fresh storage: rows present
  // only in `other` have to be inserted, which an in-place update cannot do
  // without repeatedly shifting the tail.
  std::vector<Key> keys;
  std::vector<Value> amps;
  keys.reserve(rows() + other.rows());
  amps.reserve((rows() + other.rows()) * m_width);
  std::size_t ia = 0;
  std::size_t ib = 0;
  const auto emit = [&](const Key &k, ConstRow a, ConstRow b) {
    keys.push_back(k);
    for (std::size_t c = 0; c < m_width; ++c) {
      Value v = a.data() != nullptr ? a[c] : Value{0.0, 0.0};
      if (b.data() != nullptr) {
        v += scale * b[c];
      }
      amps.push_back(v);
    }
  };
  const ConstRow absent{};
  while (ia < rows() || ib < other.rows()) {
    if (ib >= other.rows() || (ia < rows() && m_keys[ia] < other.m_keys[ib])) {
      emit(m_keys[ia], row(ia), absent);
      ++ia;
    } else if (ia >= rows() || other.m_keys[ib] < m_keys[ia]) {
      emit(other.m_keys[ib], absent, other.row(ib));
      ++ib;
    } else {
      emit(m_keys[ia], row(ia), other.row(ib));
      ++ia;
      ++ib;
    }
  }
  m_keys = std::move(keys);
  m_amps = std::move(amps);
  return *this;
}

inline void ManyBodyBlockState::truncate(std::size_t max_rows) {
  if (max_rows == 0 || rows() <= max_rows) {
    return;
  }
  std::vector<double> norms;
  norms.reserve(rows());
  for (std::size_t r = 0; r < rows(); ++r) {
    norms.push_back(row_max2(r));
  }
  std::nth_element(norms.begin(),
                   norms.begin() + static_cast<difference_type>(max_rows - 1),
                   norms.end(), std::greater<double>());
  const double cutoff2 = norms[max_rows - 1];
  std::size_t out = 0;
  for (std::size_t r = 0; r < rows(); ++r) {
    if (row_max2(r) < cutoff2) {
      continue;
    }
    if (out != r) {
      const ConstRow src = row(r);
      m_keys[out] = std::move(m_keys[r]);
      std::copy(src.begin(), src.end(), m_amps.data() + out * m_width);
    }
    ++out;
  }
  m_keys.resize(out);
  m_amps.resize(out * m_width);
}

inline std::string ManyBodyBlockState::to_string() const {
  std::string res = "ManyBodyBlockState{";
  for (const auto &[det, amps] : *this) {
    res += "|";
    for (const auto &chunk : det) {
      res += std::bitset<8 * sizeof(Key::value_type)>(chunk).to_string() + " ";
    }
    res += ">: (";
    for (std::size_t c = 0; c < amps.size(); ++c) {
      res += std::to_string(amps[c].real()) + " + " +
             std::to_string(amps[c].imag()) + "i";
      if (c + 1 < amps.size()) {
        res += ", ";
      }
    }
    res += "), ";
  }
  return res + "}";
}

/**
 * @brief Block Gram matrix C = A^H B: C[i, j] = sum_det conj(A[det, i]) *
 * B[det, j].
 *
 * Merge-join over the two sorted supports (linear in rows); shared determinants
 * contribute a rank-1 update. The determinant order equals the sorted flat_map
 * iteration order of the scalar path, so the per-(i, j) summation sequence
 * matches `inner_multi` over lists bit-for-bit. `C` must hold A.width() *
 * B.width() values (row-major, stride B.width()).
 */
inline void block_inner(const ManyBodyBlockState &A,
                        const ManyBodyBlockState &B,
                        ManyBodyBlockState::Value *C) {
  const std::size_t wa = A.width();
  const std::size_t wb = B.width();
  std::fill(C, C + wa * wb, ManyBodyBlockState::Value{0.0, 0.0});
  std::size_t ia = 0;
  std::size_t ib = 0;
  while (ia < A.rows() && ib < B.rows()) {
    if (A.key(ia) < B.key(ib)) {
      ++ia;
    } else if (B.key(ib) < A.key(ia)) {
      ++ib;
    } else {
      const ManyBodyBlockState::ConstRow ra = A.row(ia);
      const ManyBodyBlockState::ConstRow rb = B.row(ib);
      for (std::size_t i = 0; i < wa; ++i) {
        const ManyBodyBlockState::Value cai = std::conj(ra[i]);
        ManyBodyBlockState::Value *crow = C + i * wb;
        for (std::size_t j = 0; j < wb; ++j) {
          crow[j] += cai * rb[j];
        }
      }
      ++ia;
      ++ib;
    }
  }
}

/**
 * @brief OUT = A + B * C over the union support: out[det, j] = A[det, j] +
 * sum_i B[det, i] * C[i, j], with C row-major (B.width() x A.width()).
 *
 * The i-ascending accumulation per determinant matches `add_scaled_multi`'s
 * source-loop order bit-for-bit. Rows only in A are copied; rows only in B get
 * the pure B * C contribution (the scalar path creates those flat_map entries
 * the same way).
 */
inline ManyBodyBlockState block_add_scaled(const ManyBodyBlockState &A,
                                           const ManyBodyBlockState &B,
                                           const ManyBodyBlockState::Value *C) {
  const std::size_t wa = A.width();
  const std::size_t wb = B.width();
  std::vector<ManyBodyBlockState::Key> keys;
  keys.reserve(A.rows() + B.rows());
  std::vector<ManyBodyBlockState::Value> amps;
  amps.reserve((A.rows() + B.rows()) * wa);

  // A default-constructed (null) row means "this side has no entry for this
  // determinant" -- the same sentinel the raw-pointer version used.
  const ManyBodyBlockState::ConstRow absent{};
  const auto emit_bc = [&](ManyBodyBlockState::ConstRow base,
                           ManyBodyBlockState::ConstRow rb) {
    for (std::size_t j = 0; j < wa; ++j) {
      ManyBodyBlockState::Value v =
          base.data() != nullptr ? base[j] : ManyBodyBlockState::Value{0.0, 0.0};
      if (rb.data() != nullptr) {
        for (std::size_t i = 0; i < wb; ++i) {
          const ManyBodyBlockState::Value c = C[i * wa + j];
          if (c.real() != 0 || c.imag() != 0) {
            v += c * rb[i];
          }
        }
      }
      amps.push_back(v);
    }
  };

  std::size_t ia = 0;
  std::size_t ib = 0;
  while (ia < A.rows() || ib < B.rows()) {
    if (ib >= B.rows() || (ia < A.rows() && A.key(ia) < B.key(ib))) {
      keys.push_back(A.key(ia));
      emit_bc(A.row(ia), absent);
      ++ia;
    } else if (ia >= A.rows() || B.key(ib) < A.key(ia)) {
      keys.push_back(B.key(ib));
      emit_bc(absent, B.row(ib));
      ++ib;
    } else {
      keys.push_back(A.key(ia));
      emit_bc(A.row(ia), B.row(ib));
      ++ia;
      ++ib;
    }
  }
  return ManyBodyBlockState(std::move(keys), std::move(amps), wa);
}

/**
 * @brief OUT = A * Y on A's support: out[det, k] = sum_j A[det, j] * Y[j, k],
 * with Y row-major (A.width() x width_out). The j-ascending accumulation
 * matches `block_combine_sparse` (add_scaled_multi) bit-for-bit, including the
 * skip-exact-zero-coefficient behavior.
 */
inline ManyBodyBlockState block_combine_cols(const ManyBodyBlockState &A,
                                             const ManyBodyBlockState::Value *Y,
                                             std::size_t width_out) {
  const std::size_t wa = A.width();
  std::vector<ManyBodyBlockState::Key> keys(A.keys());
  std::vector<ManyBodyBlockState::Value> amps(
      A.rows() * width_out, ManyBodyBlockState::Value{0.0, 0.0});
  for (std::size_t r = 0; r < A.rows(); ++r) {
    const ManyBodyBlockState::ConstRow ra = A.row(r);
    ManyBodyBlockState::Value *out = amps.data() + r * width_out;
    for (std::size_t j = 0; j < wa; ++j) {
      const ManyBodyBlockState::Value aj = ra[j];
      if (aj.real() == 0 && aj.imag() == 0) {
        continue;
      }
      const ManyBodyBlockState::Value *yrow = Y + j * width_out;
      for (std::size_t k = 0; k < width_out; ++k) {
        const ManyBodyBlockState::Value y = yrow[k];
        if (y.real() != 0 || y.imag() != 0) {
          out[k] += aj * y;
        }
      }
    }
  }
  return ManyBodyBlockState(std::move(keys), std::move(amps), width_out);
}

#endif // MANYBODY_BLOCK_STATE_H
