#ifndef MANYBODY_BLOCK_STATE_H
#define MANYBODY_BLOCK_STATE_H

#include "SlaterDeterminant.h"

#include <algorithm>
#include <bitset>
#include <complex>
#include <numeric>
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
/**
 * @brief A sorted, strictly-increasing set of fixed-width determinant keys, in ONE buffer.
 *
 * Replaces `std::vector<SlaterDeterminant<>>`, where every key is itself a heap-owning
 * vector, so an N-determinant block was N separate small allocations plus the pointer array
 * indexing them. Here the keys are a row-major `N x n_chunks` matrix of `uint64_t`, which is
 * the shape the amplitudes (`m_amps`) already had -- so this makes keys match amplitudes
 * rather than introducing a new idiom.
 *
 * Measured at N = 1M, n_chunks = 2 (`doc/plans/flat_key_store.md`): `lower_bound` 595 -> 178
 * ns/lookup and the key store 53.41 -> 15.26 MiB, i.e. 56 B/det down to 16.
 *
 * **The ordering contract is load-bearing across MPI ranks.** `Basis` derives global indices
 * as `offset + position` with `offset` from an allgather of local lengths, so every rank must
 * compute the same total order. Comparison here is therefore element-wise over `uint64` --
 * exactly what `std::vector<uint64_t>::operator<` did -- and NEVER `memcmp`, which is faster
 * and disagrees on a little-endian machine (`0x1` ranks above `0x100` byte-wise and below it
 * element-wise). `test_block_state.py` pins this.
 *
 * **Width is a hard invariant**, not an inherited accident: every key in one store has exactly
 * `n_chunks` chunks. That was already assumed (`MpiUtils.cpp:16` takes `chunks_per_state` from
 * `dets[0].size()` for a whole set, and `Basis` normalizes every input to `n_bytes`), and it
 * forecloses the variable-length ambiguity where the same occupation built from byte strings
 * of different length yields two unequal keys. An empty store has `n_chunks == 0` and adopts
 * the width of whatever is first put into it.
 */
class FlatKeyStore {
public:
  using Key = SlaterDeterminant<>;

  /**
   * @brief A non-owning, ordered view of one key. Trivially copyable, no allocation.
   *
   * Invalidated by anything that reallocates the owning store, exactly like an iterator.
   * Callers that need to outlive a mutation must materialize with `to_key()`.
   */
  struct View {
    const uint64_t *ptr{nullptr};
    std::size_t len{0};

    const uint64_t *data() const noexcept { return ptr; }
    std::size_t size() const noexcept { return len; }
    const uint64_t &operator[](std::size_t i) const noexcept { return ptr[i]; }
    Key to_key() const {
      Key out;
      out.assign(ptr, ptr + len);
      return out;
    }
  };

  /** @brief Element-wise over `uint64`, the one comparison every rank must agree on. */
  static bool less(const uint64_t *a, std::size_t na, const uint64_t *b, std::size_t nb) noexcept {
    const std::size_t n = na < nb ? na : nb;
    for (std::size_t i = 0; i < n; ++i) {
      if (a[i] != b[i]) {
        return a[i] < b[i];
      }
    }
    return na < nb;
  }
  static bool equal(const uint64_t *a, std::size_t na, const uint64_t *b, std::size_t nb) noexcept {
    if (na != nb) {
      return false;
    }
    for (std::size_t i = 0; i < na; ++i) {
      if (a[i] != b[i]) {
        return false;
      }
    }
    return true;
  }
  static bool less(View a, View b) noexcept { return less(a.ptr, a.len, b.ptr, b.len); }
  static bool less(View a, const Key &b) noexcept { return less(a.ptr, a.len, b.data(), b.size()); }
  static bool less(const Key &a, View b) noexcept { return less(a.data(), a.size(), b.ptr, b.len); }
  static bool equal(View a, const Key &b) noexcept { return equal(a.ptr, a.len, b.data(), b.size()); }
  static bool equal(View a, View b) noexcept { return equal(a.ptr, a.len, b.ptr, b.len); }

  std::size_t chunks() const noexcept { return m_n_chunks; }
  std::size_t rows() const noexcept { return m_n_chunks == 0 ? 0 : m_data.size() / m_n_chunks; }
  bool empty() const noexcept { return rows() == 0; }

  View view(std::size_t r) const noexcept {
    return View{m_data.data() + r * m_n_chunks, m_n_chunks};
  }
  Key key(std::size_t r) const { return view(r).to_key(); }

  /** @brief First row whose key is not less than `k`; `rows()` when there is none. */
  std::size_t lower_bound(const uint64_t *k, std::size_t n) const noexcept {
    std::size_t lo = 0, hi = rows();
    while (lo < hi) {
      const std::size_t mid = lo + (hi - lo) / 2;
      if (less(m_data.data() + mid * m_n_chunks, m_n_chunks, k, n)) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  }
  std::size_t lower_bound(const Key &k) const noexcept { return lower_bound(k.data(), k.size()); }
  std::size_t lower_bound(View v) const noexcept { return lower_bound(v.ptr, v.len); }

  /** @brief Row holding `k`, or `rows()` when absent. */
  std::size_t find(const uint64_t *k, std::size_t n) const noexcept {
    const std::size_t pos = lower_bound(k, n);
    if (pos < rows() && equal(view(pos), View{k, n})) {
      return pos;
    }
    return rows();
  }
  std::size_t find(const Key &k) const noexcept { return find(k.data(), k.size()); }

  /**
   * @brief Adopt `n` as this store's chunk width, or verify it against the existing one.
   *
   * **Enforced, not assumed.** This previously assigned only when the width was still 0 and
   * silently ignored a mismatch, so pushing a differently-sized key inserted `n` chunks into
   * a buffer striding by `m_n_chunks`: every later row misaligned, the support stopped being
   * sorted (making every binary search here unsound), and `from_columns`' amplitude scatter
   * wrote one `Value` past the end of its array -- a heap-buffer-overflow reachable from
   * three lines of Python. `SlaterDeterminant::from_bytes` takes its chunk count from the
   * input length with no normalization, so mixed widths are one call away; only `Basis` pads.
   * The class comment asserted this invariant and nothing checked it.
   */
  void set_chunks(std::size_t n) {
    if (n == 0) {
      throw std::invalid_argument(
          "ManyBodyBlockState: a determinant key must have at least one chunk");
    }
    if (n > m_n_chunks) {
      widen_to(n);
    }
  }

  /**
   * @brief Re-stride every stored row to `n` chunks, zero-extending on the right.
   *
   * A flat buffer is fixed-stride by construction, but `ManyBodyState` has always accepted
   * keys of differing chunk counts (`test_manybody_utils.py::get_random_state` builds them
   * from 1-4 random chunks on purpose), and the old `vector<vector<uint64_t>>` ordered them
   * lexicographically with a shorter prefix ranking first. Rejecting them would be a public
   * contract change; silently inserting them was memory corruption. Normalizing is the third
   * option, and it is the one the design claimed to deliver.
   *
   * **Zero-extension preserves the order.** For two keys differing before the shorter one
   * ends, the deciding chunk is untouched. For a prefix pair, the extended key has 0 where
   * the longer has its first extra chunk, so it still sorts first unless that tail is all
   * zeros -- in which case the two are the SAME occupation and become equal, which is the
   * variable-length ambiguity this was meant to foreclose. Equal keys land adjacent, so the
   * dedup below is a linear pass.
   */
  void widen_to(std::size_t n) {
    if (n <= m_n_chunks) {
      return;
    }
    if (m_n_chunks == 0 || m_data.empty()) {
      m_n_chunks = n;
      m_data.clear();
      return;
    }
    const std::size_t old_rows = rows();
    std::vector<uint64_t> wide(old_rows * n, 0);
    for (std::size_t r = 0; r < old_rows; ++r) {
      const uint64_t *src = m_data.data() + r * m_n_chunks;
      std::copy(src, src + m_n_chunks, wide.data() + r * n);
    }
    m_data.swap(wide);
    m_n_chunks = n;
    dedup_sorted();
  }

  /** @brief Drop adjacent duplicates. Only widening can create them. */
  void dedup_sorted() {
    const std::size_t n_rows = rows();
    std::size_t out = 0;
    for (std::size_t r = 0; r < n_rows; ++r) {
      if (out == 0 || !equal(view_at(m_data.data(), out - 1, m_n_chunks),
                             view_at(m_data.data(), r, m_n_chunks))) {
        move_row(out, r);
        ++out;
      }
    }
    resize_rows(out);
  }

  static View view_at(const uint64_t *base, std::size_t r, std::size_t n) noexcept {
    return View{base + r * n, n};
  }

  /** @brief Zero-extend `k` into this store's stride and append it. */
  void push_padded(const uint64_t *k, std::size_t n) {
    set_chunks(n);
    const std::size_t base = m_data.size();
    m_data.resize(base + m_n_chunks, 0);
    std::copy(k, k + n, m_data.data() + base);
  }

  /** @brief The width two stores share; throws when both are set and disagree. */
  /** @brief The stride two stores must share to be combined: the wider of the two. */
  static std::size_t common_chunks(const FlatKeyStore &a, const FlatKeyStore &b) noexcept {
    return a.m_n_chunks > b.m_n_chunks ? a.m_n_chunks : b.m_n_chunks;
  }

  // Sequence spellings, so the mask walks below can take either this or a `std::vector<Key>`
  // without rebuilding one from the other.
  std::size_t size() const noexcept { return rows(); }
  View operator[](std::size_t i) const noexcept { return view(i); }
  void clear() noexcept {
    m_data.clear();
    m_n_chunks = 0;
  }
  void reserve_rows(std::size_t n) { m_data.reserve(n * (m_n_chunks == 0 ? 1 : m_n_chunks)); }
  void resize_rows(std::size_t n) { m_data.resize(n * m_n_chunks); }
  void shrink_to_fit() { m_data.shrink_to_fit(); }
  void swap(FlatKeyStore &o) noexcept {
    m_data.swap(o.m_data);
    std::swap(m_n_chunks, o.m_n_chunks);
  }

  void push_back(const uint64_t *k, std::size_t n) { push_padded(k, n); }
  void push_back(const Key &k) { push_back(k.data(), k.size()); }
  void push_back(View v) { push_back(v.ptr, v.len); }

  /** @brief Copy row `from` onto row `to`; used by the in-place compactions. */
  void move_row(std::size_t to, std::size_t from) noexcept {
    if (to == from) {
      return;
    }
    uint64_t *dst = m_data.data() + to * m_n_chunks;
    const uint64_t *src = m_data.data() + from * m_n_chunks;
    for (std::size_t c = 0; c < m_n_chunks; ++c) {
      dst[c] = src[c];
    }
  }

  void insert_at(std::size_t pos, const uint64_t *k, std::size_t n) {
    set_chunks(n);
    std::vector<uint64_t> padded(m_n_chunks, 0);
    std::copy(k, k + (n < m_n_chunks ? n : m_n_chunks), padded.begin());
    m_data.insert(m_data.begin() + static_cast<std::ptrdiff_t>(pos * m_n_chunks), padded.begin(),
                  padded.end());
  }
  void insert_at(std::size_t pos, const Key &k) { insert_at(pos, k.data(), k.size()); }

  void erase_at(std::size_t r) {
    const auto first = m_data.begin() + static_cast<std::ptrdiff_t>(r * m_n_chunks);
    m_data.erase(first, first + static_cast<std::ptrdiff_t>(m_n_chunks));
  }

  /** @brief Replace contents with rows `[lo, hi)` of `src`. */
  void assign_range(const FlatKeyStore &src, std::size_t lo, std::size_t hi) {
    m_n_chunks = src.m_n_chunks;
    m_data.assign(src.m_data.begin() + static_cast<std::ptrdiff_t>(lo * m_n_chunks),
                  src.m_data.begin() + static_cast<std::ptrdiff_t>(hi * m_n_chunks));
  }

  /** @brief Raw buffer access, for bulk packing and the buffer protocol. */
  const std::vector<uint64_t> &data() const noexcept { return m_data; }
  std::vector<uint64_t> &data() noexcept { return m_data; }

  std::size_t row_capacity() const noexcept {
    return m_n_chunks == 0 ? 0 : m_data.capacity() / m_n_chunks;
  }

  bool operator==(const FlatKeyStore &o) const noexcept {
    return m_n_chunks == o.m_n_chunks && m_data == o.m_data;
  }
  bool operator!=(const FlatKeyStore &o) const noexcept { return !(*this == o); }

private:
  std::vector<uint64_t> m_data;
  std::size_t m_n_chunks{0};
};

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
  FlatKeyStore m_keys;       // sorted, strictly increasing, one flat buffer
  std::vector<Value> m_amps; // row-major, m_keys.rows() * m_width
  std::size_t m_width{0};

public:
  /**
   * @brief Iterator over row ENTRIES: `*it` is `(key, amplitudes-of-that-key)`.
   *
   * The pair is built on dereference rather than stored -- keys and amplitudes
   * live in separate arrays -- so it is returned by value (a proxy reference)
   * and `operator->` goes through the standard arrow proxy. Random access, so
   * `std::advance` / `std::distance` stay O(1) -- a property future threaded
   * chunking can rely on; nothing advances into the middle of a state today
   * (the current threaded apply partitions by row index directly).
   *
   * `reference` is `value_type` (a proxy returned by value), which is not a
   * conforming C++17 random-access iterator's `reference` -- algorithms that
   * bind through it (`std::sort`, `std::rotate`, ...) would not compile or
   * would be ill-formed. `std::advance`, `std::distance` and range-for, which
   * is everything this iterator is used for, are unaffected.
   */
  template <bool Const> class BasicIterator {
  public:
    using StateType =
        std::conditional_t<Const, const ManyBodyBlockState, ManyBodyBlockState>;
    using RowType = std::conditional_t<Const, ConstRow, Row>;
    // `Key` by value, not `const Key&`: `ManyBodyBlockState::key(r)` returns a prvalue now
    // that the keys live in one flat buffer, and a reference MEMBER does not get lifetime
    // extension -- the pair would bind to a temporary that dies at the end of the full
    // expression, which ASan reports as a stack-use-after-scope in `to_string()`.
    using value_type = std::pair<Key, RowType>;
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
  ManyBodyBlockState(const std::vector<Key> &keys, std::vector<Value> amps,
                     std::size_t width)
      : m_amps(std::move(amps)), m_width(width) {
    // Pack the owning keys into the flat buffer. Every key must have the same chunk count --
    // the invariant `FlatKeyStore` makes explicit rather than inherits.
    if (!keys.empty()) {
      m_keys.set_chunks(keys[0].size());
      m_keys.reserve_rows(keys.size());
      for (const Key &k : keys) {
        m_keys.push_back(k);
      }
    }
  }

  /** @brief Adopt a flat key buffer directly -- no per-key allocation on the way in. */
  ManyBodyBlockState(FlatKeyStore keys, std::vector<Value> amps, std::size_t width)
      : m_keys(std::move(keys)), m_amps(std::move(amps)), m_width(width) {}

  /**
   * @brief A width-`cols.size()` block over the UNION support of width-1 columns.
   *
   * The Cython `from_states` used to do this with a `std::vector<Key> support`: one heap
   * allocation per determinant per column, a `lower_bound` per row that materialized a key
   * only to search with it, and -- because `erase` after the dedup keeps the allocation --
   * a key store left holding `width` times the capacity it needs, for the block's whole
   * life. Building it here keeps everything in the flat buffer.
   *
   * Missing determinants of a column are exact zeros, and every stored coefficient
   * round-trips bit-identically, which is the contract `to_states` relies on.
   */
  static ManyBodyBlockState
  from_columns(const std::vector<const ManyBodyBlockState *> &cols) {
    const std::size_t p = cols.size();
    ManyBodyBlockState out(p);
    if (p == 0) {
      return out;
    }
    std::size_t total = 0, n_chunks = 0;
    for (const ManyBodyBlockState *c : cols) {
      total += c->rows();
      // Every column, not just the first: taking the width from column 0 and then pushing
      // another column's differently-sized keys is what desynced the buffer.
      // The WIDEST column sets the stride; narrower keys are zero-extended into it.
      // Taking column 0's width and pushing another column's wider keys is what desynced
      // the buffer and wrote past the end of the amplitude array.
      if (c->m_keys.chunks() > n_chunks) {
        n_chunks = c->m_keys.chunks();
      }
    }
    if (total == 0 || n_chunks == 0) {
      return out;
    }

    // Gather every column's keys into one buffer, then order by index and emit the distinct
    // rows. Sized once from the exact total, so the dedup leaves no capacity behind.
    FlatKeyStore gathered;
    gathered.set_chunks(n_chunks);
    gathered.reserve_rows(total);
    for (const ManyBodyBlockState *c : cols) {
      for (std::size_t r = 0; r < c->rows(); ++r) {
        // `push_back` zero-extends into `gathered`'s stride, so a narrower column's keys
        // land correctly rather than misaligning every row after them.
        gathered.push_back(c->m_keys.view(r));
      }
    }
    std::vector<std::size_t> order(total);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&gathered](std::size_t a, std::size_t b) {
      return FlatKeyStore::less(gathered.view(a), gathered.view(b));
    });
    out.m_keys.set_chunks(n_chunks);
    out.m_keys.reserve_rows(total);
    for (std::size_t i = 0; i < total; ++i) {
      if (i != 0 && FlatKeyStore::equal(gathered.view(order[i]), gathered.view(order[i - 1]))) {
        continue;
      }
      out.m_keys.push_back(gathered.view(order[i]));
    }
    out.m_keys.shrink_to_fit();

    const std::size_t ns = out.m_keys.rows();
    const std::size_t ns_chunks = out.m_keys.chunks();
    std::vector<uint64_t> padded(ns_chunks, 0);
    out.m_amps.assign(ns * p, Value{0.0, 0.0});
    for (std::size_t ci = 0; ci < p; ++ci) {
      const ManyBodyBlockState *c = cols[ci];
      for (std::size_t r = 0; r < c->rows(); ++r) {
        // Search with the key padded to the output stride: an unpadded narrower view
        // compares as a shorter vector and misses, and the miss returns `rows()`, which
        // would index one element past the end of `out.m_amps`.
        padded.assign(ns_chunks, 0);
        const FlatKeyStore::View kv = c->m_keys.view(r);
        std::copy(kv.data(), kv.data() + kv.size(), padded.begin());
        const std::size_t pos = out.m_keys.lower_bound(padded.data(), padded.size());
        if (pos >= out.m_keys.rows()) {
          throw std::logic_error("ManyBodyBlockState::from_columns: key absent from its own union");
        }
        out.m_amps[pos * p + ci] = c->m_amps[r];
      }
    }
    return out;
  }

  /**
   * @brief State from parallel (key, row) arrays given in ANY order.
   *
   * `amps` holds `width` amplitudes per key, in the same order as `keys`. Sorts
   * into the row order this container maintains; on duplicate keys the first
   * occurrence wins, matching the flat_map range-insert this replaces.
   */
  static ManyBodyBlockState from_unsorted(const std::vector<Key> &keys,
                                          const std::vector<Value> &amps,
                                          std::size_t width);

  std::size_t width() const noexcept { return m_width; }
  std::size_t rows() const noexcept { return m_keys.rows(); }
  /** @brief Number of stored determinants -- rows(), spelled for the map surface. */
  size_type size() const noexcept { return m_keys.rows(); }
  bool empty() const noexcept { return m_keys.empty(); }
  /** @brief Theoretical row-count bound (the key vector's own max_size()) -- a
   * container-capacity figure, not a real usable limit; spelled for map-surface parity
   * with the flat_map class's forwarded std::map::max_size(). */
  size_type max_size() const noexcept { return m_keys.data().max_size(); }

  /** @brief Row `r`'s key, materialized. Prefer `key_view` in a loop: this allocates. */
  Key key(std::size_t r) const { return m_keys.key(r); }
  /** @brief Row `r`'s key as a non-owning view -- no allocation, invalidated by any
   * mutation of this block, exactly like an iterator. */
  FlatKeyStore::View key_view(std::size_t r) const noexcept { return m_keys.view(r); }
  /** @brief The flat key store itself, for bulk packing and ordered search. */
  const FlatKeyStore &key_store() const noexcept { return m_keys; }
  /** @brief Every key, materialized. O(rows) allocations; `key_store()` avoids them. */
  std::vector<Key> keys() const {
    std::vector<Key> out;
    out.reserve(m_keys.rows());
    for (std::size_t r = 0; r < m_keys.rows(); ++r) {
      out.push_back(m_keys.key(r));
    }
    return out;
  }
  Value *data() noexcept { return m_amps.data(); }
  const Value *data() const noexcept { return m_amps.data(); }

  /**
   * @brief The amplitudes of row `r`.
   *
   * Non-owning: invalidated by anything that changes the row count or order
   * (prune_rows, keep_rows, merge_keys, truncate, clear, swap, erase,
   * add_scaled/+=/-= -- which rebuild storage over the union support --, and
   * operator[] when it inserts a new key). In-place scaling (*=, /=) does NOT
   * invalidate a row: the layout is unchanged, only the values are.
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
    return m_keys.find(k);
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
    const std::size_t pos = m_keys.lower_bound(k);
    if (pos < m_keys.rows() && FlatKeyStore::equal(m_keys.view(pos), k)) {
      return {pos, false};
    }
    m_keys.insert_at(pos, k);
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
    m_keys.erase_at(r);
    const auto first = m_amps.begin() + static_cast<difference_type>(r * m_width);
    m_amps.erase(first, first + static_cast<difference_type>(m_width));
  }

  void clear() noexcept {
    m_keys.clear();
    m_amps.clear();
  }

  void reserve(size_type n) {
    m_keys.reserve_rows(n);
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
  // Plain-named aliases for operator*=/operator/=: Cython's cppclass declarations do
  // not support compound-assignment operators, so these are what the Python wrapper's
  // true in-place __imul__/__itruediv__ (no reallocation; a live Row survives them,
  // unlike add_scaled/+=/-=) actually call.
  void scale(Value s) noexcept { *this *= s; }
  void scale_inv(Value s) noexcept { *this /= s; }
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
          m_keys.move_row(out, r);
          std::copy(src.begin(), src.end(), m_amps.data() + out * m_width);
        }
        ++out;
      }
    }
    m_keys.resize_rows(out);
    m_amps.resize(out * m_width);
  }

  /**
   * @brief Keep only rows whose key appears in `keep` (sorted, strictly
   * increasing). Linear merge over the two sorted sequences — the
   * set-intersection complement of prune_rows, used by the capped
   * Green's-function recurrence to project a block onto the retained
   * determinant set.
   */
  // Templated on the container so a mask can be passed as its `FlatKeyStore` -- the
  // Cython callers used to hand over `mask.keys()`, which rebuilds the whole
  // `vector<vector<uint64_t>>` this store exists to delete, once per call.
  template <typename KeepT> void keep_rows(const KeepT &keep) {
    std::size_t out = 0;
    std::size_t ik = 0;
    for (std::size_t r = 0; r < rows(); ++r) {
      while (ik < keep.size() && FlatKeyStore::less(keep[ik], m_keys.view(r))) {
        ++ik;
      }
      if (ik < keep.size() && FlatKeyStore::equal(m_keys.view(r), keep[ik])) {
        if (out != r) {
          const ConstRow src = row(r);
          m_keys.move_row(out, r);
          std::copy(src.begin(), src.end(), m_amps.data() + out * m_width);
        }
        ++out;
        ++ik;
      }
    }
    m_keys.resize_rows(out);
    m_amps.resize(out * m_width);
  }

  /**
   * @brief A new block holding rows `[lo, hi)` of this one, same width.
   *
   * Rows are stored in sorted key order, so a contiguous row range is already a
   * valid support (sorted, unique) and both arrays copy as flat ranges -- no
   * merge, no per-row work, and the result allocates exactly `hi - lo` rows.
   *
   * This exists for the row-chunked matvec (`_lanczos_step.pxi`) and the CIPSI
   * selection round (`cipsi_solver._apply_block_and_redistribute`), which used
   * to build a chunk as `copy()` + `keep_rows(mask)`. That allocates a
   * FULL-SIZE duplicate of the block and then shrinks its logical length --
   * `keep_rows`' `resize` does not release `std::vector` capacity -- so the
   * duplicate stayed resident at full size for the whole chunk, defeating the
   * point of chunking (measured: chunking ran slower with a HIGHER peak than
   * the one-shot path, see doc/plans/dc_smo_memory.md). It also required
   * materializing one Python key object per row to build the mask.
   *
   * `lo`/`hi` are clamped to `[0, rows()]` and an empty or inverted range gives
   * an empty block of the same width -- never the width-0 polymorphic zero,
   * which would be an asymmetric value on a rank that owns no rows.
   */
  ManyBodyBlockState row_slice(std::size_t lo, std::size_t hi) const {
    const std::size_t n = rows();
    lo = std::min(lo, n);
    hi = std::min(hi, n);
    ManyBodyBlockState out;
    out.m_width = m_width;
    if (hi <= lo) {
      return out;
    }
    out.m_keys.assign_range(m_keys, lo, hi);
    out.m_amps.assign(m_amps.begin() + static_cast<std::ptrdiff_t>(lo * m_width),
                      m_amps.begin() + static_cast<std::ptrdiff_t>(hi * m_width));
    return out;
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
  template <typename KeepT>
  std::size_t count_rows_in(const KeepT &keep) const noexcept {
    std::size_t n = 0;
    std::size_t ik = 0;
    for (std::size_t r = 0; r < rows(); ++r) {
      while (ik < keep.size() && FlatKeyStore::less(keep[ik], m_keys.view(r))) {
        ++ik;
      }
      if (ik < keep.size() && FlatKeyStore::equal(m_keys.view(r), keep[ik])) {
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
  template <typename KeepT>
  void new_row_max_norm2(const KeepT &keep,
                         std::vector<double> &out) const {
    std::size_t ik = 0;
    for (std::size_t r = 0; r < rows(); ++r) {
      while (ik < keep.size() && FlatKeyStore::less(keep[ik], m_keys.view(r))) {
        ++ik;
      }
      if (ik < keep.size() && FlatKeyStore::equal(m_keys.view(r), keep[ik])) {
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
  template <typename KeepT>
  ManyBodyBlockState keys_new_above(const KeepT &keep,
                                    double cutoff2) const {
    std::vector<Key> out;
    std::size_t ik = 0;
    for (std::size_t r = 0; r < rows(); ++r) {
      while (ik < keep.size() && FlatKeyStore::less(keep[ik], m_keys.view(r))) {
        ++ik;
      }
      if (ik < keep.size() && FlatKeyStore::equal(m_keys.view(r), keep[ik])) {
        ++ik;
      } else if (row_max2(r) > cutoff2) {
        out.push_back(m_keys.key(r));
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
    FlatKeyStore out;
    const std::size_t n_chunks = FlatKeyStore::common_chunks(m_keys, other.m_keys);
    if (n_chunks != 0) {
      out.set_chunks(n_chunks);
    }
    out.reserve_rows(rows() + other.rows());
    std::size_t ia = 0, ib = 0;
    while (ia < rows() || ib < other.rows()) {
      if (ib >= other.rows()) {
        out.push_back(m_keys.view(ia++));
      } else if (ia >= rows()) {
        out.push_back(other.m_keys.view(ib++));
      } else if (FlatKeyStore::less(m_keys.view(ia), other.m_keys.view(ib))) {
        out.push_back(m_keys.view(ia++));
      } else if (FlatKeyStore::less(other.m_keys.view(ib), m_keys.view(ia))) {
        out.push_back(other.m_keys.view(ib++));
      } else {
        out.push_back(m_keys.view(ia++));
        ++ib;
      }
    }
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
    // Count first so the merged buffer is allocated exactly once and carries no slack --
    // `erase`/`resize` never return capacity, and nothing in this layer had a shrink_to_fit.
    std::size_t n_new = 0, ik = 0;
    for (std::size_t r = 0; r < other.rows(); ++r) {
      while (ik < m_keys.rows() && FlatKeyStore::less(m_keys.view(ik), other.m_keys.view(r))) {
        ++ik;
      }
      if (ik >= m_keys.rows() || FlatKeyStore::less(other.m_keys.view(r), m_keys.view(ik))) {
        ++n_new;
      } else {
        ++ik;
      }
    }
    if (n_new == 0) {
      return;
    }
    FlatKeyStore merged;
    merged.set_chunks(FlatKeyStore::common_chunks(m_keys, other.m_keys));
    merged.reserve_rows(m_keys.rows() + n_new);
    std::size_t ia = 0, ib = 0;
    while (ia < m_keys.rows() || ib < other.rows()) {
      if (ib >= other.rows()) {
        merged.push_back(m_keys.view(ia++));
      } else if (ia >= m_keys.rows()) {
        merged.push_back(other.m_keys.view(ib++));
      } else if (FlatKeyStore::less(m_keys.view(ia), other.m_keys.view(ib))) {
        merged.push_back(m_keys.view(ia++));
      } else if (FlatKeyStore::less(other.m_keys.view(ib), m_keys.view(ia))) {
        merged.push_back(other.m_keys.view(ib++));
      } else {
        merged.push_back(m_keys.view(ia++));
        ++ib;
      }
    }
    m_keys.swap(merged);
  }

  /**
   * @brief Gather columns `cols` into a new block on the SAME support (same
   * `keys()`, including rows that are zero in every selected column).
   * `out.row(r)[k] = row(r)[cols[k]]` -- bit-for-bit what
   * `block_combine_cols(*this, Y, cols.size())` produces for a 0/1 selection
   * matrix `Y`, but O(rows() * cols.size()) instead of
   * O(rows() * width() * cols.size()). `cols` entries are assumed already
   * validated (in range, non-negative) by the caller.
   */
  ManyBodyBlockState select_cols(const std::vector<std::size_t> &cols) const {
    const std::size_t n = cols.size();
    std::vector<Value> out_amps(rows() * n);
    for (std::size_t r = 0; r < rows(); ++r) {
      const ConstRow src = row(r);
      Value *dst = out_amps.data() + r * n;
      for (std::size_t k = 0; k < n; ++k) {
        dst[k] = src[cols[k]];
      }
    }
    return ManyBodyBlockState(m_keys, std::move(out_amps), n);  // flat store, copied as a buffer
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

inline ManyBodyBlockState
ManyBodyBlockState::from_unsorted(const std::vector<Key> &keys,
                                  const std::vector<Value> &amps,
                                  std::size_t width) {
  const std::size_t n = keys.size();
  if (amps.size() != n * width) {
    throw std::invalid_argument(
        "ManyBodyBlockState::from_unsorted: amps size does not match keys * width");
  }
  std::vector<std::size_t> order(n);
  for (std::size_t i = 0; i < n; ++i) {
    order[i] = i;
  }
  // Stable, so that among equal keys the earliest input entry sorts first and
  // the dedup below keeps it -- the flat_map range-insert semantics.
  std::stable_sort(
      order.begin(), order.end(),
      [&keys](std::size_t a, std::size_t b) { return keys[a] < keys[b]; });
  ManyBodyBlockState res(width);
  if (n > 0) {
    res.m_keys.set_chunks(keys[0].size());
  }
  res.m_keys.reserve_rows(n);
  res.m_amps.reserve(n * width);
  for (const std::size_t i : order) {
    if (!res.m_keys.empty() &&
        FlatKeyStore::equal(res.m_keys.view(res.m_keys.rows() - 1), keys[i])) {
      continue;
    }
    res.m_keys.push_back(keys[i]);
    res.m_amps.insert(res.m_amps.end(), amps.begin() + static_cast<difference_type>(i * width),
                      amps.begin() + static_cast<difference_type>((i + 1) * width));
  }
  return res;
}

inline ManyBodyBlockState &
ManyBodyBlockState::add_scaled(const ManyBodyBlockState &other, Value scale) {
  // A width-0, row-less state is the polymorphic zero (what a default
  // construction gives) and adopts the other operand's width, so summing onto
  // it works at any block width -- even when `other` itself has zero rows
  // (e.g. an operator applied to a column and pruned to nothing: still a real
  // width-1 result, not a "no width information" zero). A state with an
  // explicit width stays strict -- silently widening that would hide a
  // genuine width mismatch. This check must precede the empty-`other`
  // short-circuit below, or the polymorphic zero never adopts a width when
  // `other` happens to be empty.
  if (rows() == 0 && m_width == 0) {
    *this = other;
    return *this *= scale;
  }
  if (other.rows() == 0 || scale == Value{0.0, 0.0}) {
    return *this;
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
  FlatKeyStore keys;
  const std::size_t merged_chunks = FlatKeyStore::common_chunks(m_keys, other.m_keys);
  if (merged_chunks != 0) {
    keys.set_chunks(merged_chunks);
  }
  std::vector<Value> amps;
  keys.reserve_rows(rows() + other.rows());
  amps.reserve((rows() + other.rows()) * m_width);
  std::size_t ia = 0;
  std::size_t ib = 0;
  const auto emit = [&](FlatKeyStore::View k, ConstRow a, ConstRow b) {
    keys.push_back(k);
    for (std::size_t c = 0; c < m_width; ++c) {
      Value v = a.empty() ? Value{0.0, 0.0} : a[c];
      if (!b.empty()) {
        v += scale * b[c];
      }
      amps.push_back(v);
    }
  };
  const ConstRow absent{};
  while (ia < rows() || ib < other.rows()) {
    if (ib >= other.rows() ||
        (ia < rows() && FlatKeyStore::less(m_keys.view(ia), other.m_keys.view(ib)))) {
      emit(m_keys.view(ia), row(ia), absent);
      ++ia;
    } else if (ia >= rows() ||
               FlatKeyStore::less(other.m_keys.view(ib), m_keys.view(ia))) {
      emit(other.m_keys.view(ib), absent, other.row(ib));
      ++ib;
    } else {
      emit(m_keys.view(ia), row(ia), other.row(ib));
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
  // norms[r] stays index-aligned with row r; nth_element reorders a SEPARATE
  // copy to find the cutoff, so row_max2 is computed once per row rather than
  // once to rank it and again to compact (the naive two-pass approach).
  std::vector<double> norms(rows());
  for (std::size_t r = 0; r < rows(); ++r) {
    norms[r] = row_max2(r);
  }
  std::vector<double> ranked(norms);
  std::nth_element(ranked.begin(),
                   ranked.begin() + static_cast<difference_type>(max_rows - 1),
                   ranked.end(), std::greater<double>());
  const double cutoff2 = ranked[max_rows - 1];
  std::size_t out = 0;
  for (std::size_t r = 0; r < rows(); ++r) {
    if (norms[r] < cutoff2) {
      continue;
    }
    if (out != r) {
      const ConstRow src = row(r);
      m_keys.move_row(out, r);
      std::copy(src.begin(), src.end(), m_amps.data() + out * m_width);
    }
    ++out;
  }
  m_keys.resize_rows(out);
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
    // `key_view`, not `key`: the latter materializes a `SlaterDeterminant`, and this is
    // `block_inner`'s merge join -- once per Lanczos iteration and again per locked block
    // inside reorthogonalization. Two allocations per merge step measured 4.0x slower than
    // the reference comparison this replaced.
    if (FlatKeyStore::less(A.key_view(ia), B.key_view(ib))) {
      ++ia;
    } else if (FlatKeyStore::less(B.key_view(ib), A.key_view(ia))) {
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
  FlatKeyStore keys;
  const std::size_t out_chunks = FlatKeyStore::common_chunks(A.key_store(), B.key_store());
  if (out_chunks != 0) {
    keys.set_chunks(out_chunks);
  }
  keys.reserve_rows(A.rows() + B.rows());
  std::vector<ManyBodyBlockState::Value> amps;
  amps.reserve((A.rows() + B.rows()) * wa);

  // A default-constructed (empty) row means "this side has no entry for this
  // determinant" -- the same sentinel the raw-pointer version used.
  const ManyBodyBlockState::ConstRow absent{};
  const auto emit_bc = [&](ManyBodyBlockState::ConstRow base,
                           ManyBodyBlockState::ConstRow rb) {
    for (std::size_t j = 0; j < wa; ++j) {
      ManyBodyBlockState::Value v =
          base.empty() ? ManyBodyBlockState::Value{0.0, 0.0} : base[j];
      if (!rb.empty()) {
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
    if (ib >= B.rows() ||
        (ia < A.rows() && FlatKeyStore::less(A.key_view(ia), B.key_view(ib)))) {
      keys.push_back(A.key_view(ia));
      emit_bc(A.row(ia), absent);
      ++ia;
    } else if (ia >= A.rows() ||
               FlatKeyStore::less(B.key_view(ib), A.key_view(ia))) {
      keys.push_back(B.key_view(ib));
      emit_bc(absent, B.row(ib));
      ++ib;
    } else {
      keys.push_back(A.key_view(ia));
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
