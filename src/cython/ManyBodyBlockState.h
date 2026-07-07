#ifndef MANYBODY_BLOCK_STATE_H
#define MANYBODY_BLOCK_STATE_H

#include "SlaterDeterminant.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>

/**
 * @class ManyBodyBlockState
 * @brief A block of p many-body vectors over ONE shared Slater-determinant support.
 *
 * Stores the union support as a sorted, strictly-increasing key vector and the
 * coefficients as a row-major (rows() x width()) dense array: row r holds the p
 * amplitudes of determinant key(r), one per block vector. This is the hot-loop
 * (block Lanczos / Green's function) counterpart of ManyBodyState: the shared
 * support lets ManyBodyOperator::apply do the term loop, sign work, restriction
 * checks and accumulator hashing once per (determinant, term) and emit p scaled
 * amplitudes, and it lets block inner products / axpy run as dense row-block BLAS
 * when two blocks share a support. ManyBodyState remains the single-vector
 * boundary type everywhere outside the hot loop.
 *
 * Invariants: m_keys is sorted and unique; m_amps.size() == m_keys.size() * m_width.
 * The block width is a runtime value (no compile-time bound).
 */
class ManyBodyBlockState {
public:
  using Key = SlaterDeterminant<>;
  using Value = std::complex<double>;

private:
  std::vector<Key> m_keys;   // sorted, strictly increasing
  std::vector<Value> m_amps; // row-major, m_keys.size() * m_width
  std::size_t m_width{0};

public:
  ManyBodyBlockState() = default;
  ManyBodyBlockState(const ManyBodyBlockState &) = default;
  ManyBodyBlockState(ManyBodyBlockState &&) noexcept = default;
  ManyBodyBlockState &operator=(const ManyBodyBlockState &) = default;
  ManyBodyBlockState &operator=(ManyBodyBlockState &&) noexcept = default;
  ~ManyBodyBlockState() = default;

  /**
   * @brief Adopt pre-built storage. `keys` must be sorted and unique and
   * `amps.size() == keys.size() * width` (unchecked in release builds).
   */
  ManyBodyBlockState(std::vector<Key> keys, std::vector<Value> amps,
                     std::size_t width)
      : m_keys(std::move(keys)), m_amps(std::move(amps)), m_width(width) {}

  std::size_t width() const noexcept { return m_width; }
  std::size_t rows() const noexcept { return m_keys.size(); }
  bool empty() const noexcept { return m_keys.empty(); }

  const Key &key(std::size_t r) const { return m_keys[r]; }
  const std::vector<Key> &keys() const noexcept { return m_keys; }
  Value *data() noexcept { return m_amps.data(); }
  const Value *data() const noexcept { return m_amps.data(); }
  Value *row(std::size_t r) noexcept { return m_amps.data() + r * m_width; }
  const Value *row(std::size_t r) const noexcept {
    return m_amps.data() + r * m_width;
  }

  /** @brief Row index of `k`, or rows() when absent (binary search). */
  std::size_t find_row(const Key &k) const noexcept {
    auto it = std::lower_bound(m_keys.begin(), m_keys.end(), k);
    if (it != m_keys.end() && *it == k) {
      return static_cast<std::size_t>(it - m_keys.begin());
    }
    return m_keys.size();
  }

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
      const Value *src = row(r);
      for (std::size_t c = 0; c < m_width; ++c) {
        if (std::norm(src[c]) > cutoff2) {
          keep = true;
          break;
        }
      }
      if (keep) {
        if (out != r) {
          m_keys[out] = std::move(m_keys[r]);
          std::copy(src, src + m_width, m_amps.data() + out * m_width);
        }
        ++out;
      }
    }
    m_keys.resize(out);
    m_amps.resize(out * m_width);
  }

  /** @brief Per-column sum of |amp|^2 into out[0..width). */
  void col_norm2(double *out) const noexcept {
    for (std::size_t c = 0; c < m_width; ++c) {
      out[c] = 0.0;
    }
    for (std::size_t r = 0; r < rows(); ++r) {
      const Value *src = row(r);
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

/**
 * @brief Block Gram matrix C = A^H B: C[i, j] = sum_det conj(A[det, i]) * B[det, j].
 *
 * Merge-join over the two sorted supports (linear in rows); shared determinants
 * contribute a rank-1 update. The determinant order equals the sorted flat_map
 * iteration order of the scalar path, so the per-(i, j) summation sequence matches
 * `inner_multi` over lists bit-for-bit. `C` must hold A.width() * B.width() values
 * (row-major, stride B.width()).
 */
inline void block_inner(const ManyBodyBlockState &A, const ManyBodyBlockState &B,
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
      const ManyBodyBlockState::Value *ra = A.row(ia);
      const ManyBodyBlockState::Value *rb = B.row(ib);
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

  const auto emit_bc = [&](const ManyBodyBlockState::Value *base,
                           const ManyBodyBlockState::Value *rb) {
    for (std::size_t j = 0; j < wa; ++j) {
      ManyBodyBlockState::Value v = base ? base[j]
                                         : ManyBodyBlockState::Value{0.0, 0.0};
      if (rb != nullptr) {
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
      emit_bc(A.row(ia), nullptr);
      ++ia;
    } else if (ia >= A.rows() || B.key(ib) < A.key(ia)) {
      keys.push_back(B.key(ib));
      emit_bc(nullptr, B.row(ib));
      ++ib;
    } else {
      keys.push_back(A.key(ia));
      emit_bc(A.row(ia), B.row(ib));
      ++ia;
      ++ib;
    }
  }
  return ManyBodyBlockState(std::move(keys), std::move(amps),
                            wa);
}

/**
 * @brief OUT = A * Y on A's support: out[det, k] = sum_j A[det, j] * Y[j, k],
 * with Y row-major (A.width() x width_out). The j-ascending accumulation matches
 * `block_combine_sparse` (add_scaled_multi) bit-for-bit, including the
 * skip-exact-zero-coefficient behavior.
 */
inline ManyBodyBlockState block_combine_cols(const ManyBodyBlockState &A,
                                             const ManyBodyBlockState::Value *Y,
                                             std::size_t width_out) {
  const std::size_t wa = A.width();
  std::vector<ManyBodyBlockState::Key> keys(A.keys());
  std::vector<ManyBodyBlockState::Value> amps(A.rows() * width_out,
                                              ManyBodyBlockState::Value{0.0,
                                                                        0.0});
  for (std::size_t r = 0; r < A.rows(); ++r) {
    const ManyBodyBlockState::Value *ra = A.row(r);
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
