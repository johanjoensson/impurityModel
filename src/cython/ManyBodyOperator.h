#ifndef MANYBODYOPERATOR_H
#define MANYBODYOPERATOR_H

#include "ManyBodyBlockState.h"
#include "ManyBodyState.h"
#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

/**
 * @brief True when compiled with the opt-in threaded apply
 * (IMPURITYMODEL_PARALLEL=1).
 *
 * The threaded merge changes the duplicate-accumulation order, so bit-for-bit
 * reproducibility of apply results is a serial-build property; tests use this
 * to choose exact vs tolerance assertions.
 */
bool apply_parallel_build() noexcept;

/**
 * @class ManyBodyOperator
 * @brief Represents a quantum many-body operator in second quantization.
 *
 * This class stores a collection of creation and annihilation operator
 * sequences along with their corresponding complex amplitudes. It provides
 * functionality to apply the operator to a ManyBodyState, with support for
 * multithreading (when PARALLEL is defined) and restriction masks to filter
 * allowed states.
 */
class ManyBodyOperator {

public:
  template <typename T> struct Comparer {
    bool operator()(const std::vector<T> &a,
                    const std::vector<T> &b) const noexcept {
      size_t i;
      for (i = 0; i < std::min(a.size(), b.size()); i++) {
        if (a[i] < b[i]) {
          return true;
        } else if (a[i] > b[i]) {
          return false;
        }
      }
      return i >= a.size();
    }
  };
  using SCALAR = std::complex<double>;
  using OPS = std::vector<int64_t>;
  using OPS_VEC = std::vector<OPS>;
  using SCALAR_VEC = std::vector<SCALAR>;
  using SLATER = ManyBodyState::key_type;
  using Restrictions =
      std::vector<std::pair<std::vector<size_t>, std::pair<size_t, size_t>>>;
  // Weighted-sum restrictions: each entry is a list of (integer weight, orbital
  // subset) groups together with an inclusive [q_min, q_max] bound on the
  // weighted occupation sum  sum_w  w * (#occupied orbitals in that group).
  // Lets a charge like S_z = sum +-1 n_i (after scaling to integer weights) or
  // L_z = sum m_l n_i be expressed, which a plain subset-occupation bound
  // cannot.
  using WeightedRestrictions =
      std::vector<std::pair<std::vector<std::pair<long, std::vector<size_t>>>,
                            std::pair<long, long>>>;

public:
  using key_type = OPS;
  using mapped_type = SCALAR;
  using value_type = std::pair<key_type, mapped_type>;
  using size_type = std::vector<value_type>::size_type;
  using difference_type = std::vector<std::pair<OPS, SCALAR>>::difference_type;
  using reference = std::pair<const key_type &, mapped_type &>;
  using const_reference = std::pair<const key_type &, const mapped_type &>;
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

  [[nodiscard]] ManyBodyState operator()(const ManyBodyState &psi,
                                         double cutoff = 0) const /*noexcept*/ {
    return apply(psi, cutoff);
  }
  [[nodiscard]] std::vector<ManyBodyState>
  operator()(const std::vector<ManyBodyState> &psis, double cutoff = 0) const
  /*noexcept*/ {
    return apply(psis, cutoff);
  }

  /**
   * @brief Configure orbital occupation constraints.
   */
  void build_restriction_mask(const Restrictions &restrictions) noexcept;

  /**
   * @brief Configure weighted-sum occupation constraints (e.g. S_z, L_z).
   */
  void build_weighted_restriction_mask(
      const WeightedRestrictions &restrictions) noexcept;

  /**
   * @brief Apply this operator to a ManyBodyState.
   *
   * @param psi The input many-body state.
   * @param cutoff Threshold for pruning components with negligible amplitudes.
   * @return ManyBodyState The resulting many-body state.
   */
  [[nodiscard]] ManyBodyState apply(const ManyBodyState &,
                                    double cutoff = 0) const /*noexcept*/;

  [[nodiscard]] std::vector<ManyBodyState>
  apply(const std::vector<ManyBodyState> &psis, double cutoff) const
  /*noexcept*/ {
    std::vector<ManyBodyState> res;
    res.reserve(psis.size());
    for (const ManyBodyState &psi : psis) {
      res.push_back(this->apply(psi, cutoff));
    }
    return res;
  }

  [[nodiscard]] std::vector<ManyBodyState>
  apply(const std::vector<const ManyBodyState *> &psis, double cutoff) const
  /*noexcept*/ {
    std::vector<ManyBodyState> res;
    res.reserve(psis.size());
    for (const ManyBodyState *psi : psis) {
      res.push_back(this->apply(*psi, cutoff));
    }
    return res;
  }

  /**
   * @brief Apply this operator to a shared-support block of p vectors.
   *
   * The term loop, fermion sign, restriction check and accumulator hash
   * operation run once per (determinant, term); the p amplitudes are emitted
   * with p fused multiply-adds. Per-column arithmetic is identical to p
   * independent apply() calls (bit-for-bit at cutoff 0). The cutoff acts on
   * whole rows: a row is kept when ANY column survives the |amp|^2 > cutoff^2
   * test, so the output block keeps its shared support (sub-cutoff residuals
   * in the other columns are retained, unlike per-column pruning).
   */
  [[nodiscard]] ManyBodyBlockState apply(const ManyBodyBlockState &block,
                                         double cutoff = 0) const;

  [[nodiscard]] size_type size() const noexcept;
  [[nodiscard]] bool empty() const noexcept;
  bool clear();

  /**
   * @brief Rewrite the STORED terms into canonical normal order.
   *
   * Canonical form: creations before annihilations, each group ascending in
   * orbital, Pauli-vanishing terms dropped, terms equal up to ordering merged,
   * and coefficients below 1e-12 in magnitude removed. Contractions emitted
   * when a creation crosses an annihilation of the same orbital produce shorter
   * terms, down to the constant (empty) term.
   *
   * This is a representation change only: the operator's action on any state is
   * unchanged. Every constructor and every algebraic operation
   * (product/commutator/adjoint/...) leaves the operator canonical, so that
   * algebra actually cancels -- without it A*B - B*A never simplifies. Only the
   * raw map-style mutators (operator[], insert, emplace, non-const at) can
   * break the invariant; they clear the flag and apply() falls back to
   * normal-ordering the flat representation, so correctness never depends on
   * the invariant holding.
   */
  void canonicalize();
  /** @brief Whether the stored terms are known to be in canonical normal order.
   */
  [[nodiscard]] bool is_canonical() const noexcept;

  /**
   * @brief Enable/disable build-time normal ordering of the apply()
   * representation.
   *
   * Only has an observable effect on an operator whose stored terms are NOT
   * canonical, i.e. one mutated through operator[] / insert / emplace after
   * construction -- everything built through a constructor or through the
   * algebra is canonical already, and its flat representation is a plain copy
   * of the stored terms either way. Disabling it on such an operator makes
   * apply() run the raw, unordered term strings.
   */
  void set_normal_ordering(bool enable) noexcept;
  [[nodiscard]] bool normal_ordering() const noexcept;
  /** @brief Number of terms in the (possibly normal-ordered) apply()
   * representation. */
  [[nodiscard]] size_type num_flat_terms() const;

  ManyBodyOperator &operator+=(const ManyBodyOperator &) noexcept;
  ManyBodyOperator &operator-=(const ManyBodyOperator &) noexcept;
  ManyBodyOperator &operator*=(mapped_type) noexcept;
  ManyBodyOperator &operator/=(mapped_type) noexcept;
  [[nodiscard]] ManyBodyOperator operator-() const noexcept;

  [[nodiscard]] friend ManyBodyOperator
  operator+(const ManyBodyOperator &self, const ManyBodyOperator &other) {
    return ManyBodyOperator{self} += other;
  }

  [[nodiscard]] friend ManyBodyOperator
  operator-(const ManyBodyOperator &self, const ManyBodyOperator &other) {
    return ManyBodyOperator{self} -= other;
  }

  [[nodiscard]] friend ManyBodyOperator operator*(const ManyBodyOperator &self,
                                                  mapped_type s) {
    return ManyBodyOperator{self} *= s;
  }

  [[nodiscard]] friend ManyBodyOperator operator*(mapped_type s,
                                                  const ManyBodyOperator &o) {
    return o * s;
  }

  [[nodiscard]] friend ManyBodyOperator operator/(const ManyBodyOperator &self,
                                                  mapped_type s) {
    return ManyBodyOperator{self} /= s;
  }

  [[nodiscard]] bool operator==(const ManyBodyOperator &other) const {
    return this->m_ops == other.m_ops;
  }
  [[nodiscard]] bool operator!=(const ManyBodyOperator &other) const {
    return !(*this == other);
  }
  mapped_type &operator[](const key_type &key);
  mapped_type &operator[](key_type &&key);

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

private:
  std::vector<std::pair<key_type, mapped_type>> m_ops;

  std::tuple<std::vector<ManyBodyState::key_type>, std::vector<size_t>,
             std::vector<size_t>>
      m_restrictions_mask;

  // For each weighted restriction: a list of (weight, precomputed bitmask)
  // groups and the inclusive [q_min, q_max] bound on the weighted occupation
  // sum.
  std::vector<std::pair<std::vector<std::pair<long, ManyBodyState::key_type>>,
                        std::pair<long, long>>>
      m_weighted_restrictions_mask;

  bool m_normal_order{true};
  // True when m_ops is known to be in canonical normal order (see
  // canonicalize()). Set by canonicalize(), preserved by the algebraic
  // operations, cleared by the raw map-style mutators. Purely an optimization
  // hint for build_flat_representation: when set, the flat rep is a copy of
  // m_ops and the normal-ordering recursion is skipped.
  bool m_canonical{false};
  mutable bool m_flat_dirty{true};
  mutable std::vector<int64_t> m_flat_indices;
  mutable std::vector<size_t> m_flat_offsets;
  mutable std::vector<std::complex<double>> m_flat_coeffs;
  // Per-term flag (1 = diagonal): the created-orbital multiset equals the
  // annihilated-orbital multiset, so the term maps every occupation basis state
  // to itself up to a scalar. Lets apply() accumulate all diagonal terms of one
  // input SD into a single output insert. Constants (zero operators) are
  // diagonal.
  mutable std::vector<uint8_t> m_flat_diagonal;
  // Per-term flag (1 = pure number-operator product, e.g. n_i, n_i n_j, or a
  // constant): balanced + all-distinct orbitals AND the build-time all-occupied
  // probe gives a nonzero (occupancy-independent) sign. These get the Phase 2b
  // fast path: a single occupancy AND-test against m_density_mask instead of
  // running create/annihilate, with the constant sign folded into
  // m_density_coeff. Balanced terms that are NOT pure n-products (e.g. c_i
  // c^d_i = 1 - n_i) fail the probe and fall back to the general diagonal path,
  // so correctness never depends on this optimization firing.
  mutable std::vector<uint8_t> m_flat_density;
  mutable std::vector<ManyBodyState::key_type> m_density_mask;
  mutable std::vector<std::complex<double>> m_density_coeff;
  // Per-term flag (1 = off-diagonal one-body hop c^d_i c_j, i != j). These get
  // a masked kernel: occupancy/vacancy bit tests + the fermion sign as
  // (-1)^popcount(state & m_onebody_between), where the between-mask marks the
  // orbitals strictly between i and j -- one popcount instead of the two prefix
  // scans inside create/annihilate. m_onebody_i = created orbital, m_onebody_j
  // = annihilated orbital.
  mutable std::vector<uint8_t> m_flat_onebody;
  mutable std::vector<size_t> m_onebody_i;
  mutable std::vector<size_t> m_onebody_j;
  mutable std::vector<ManyBodyState::key_type> m_onebody_between;

  void build_flat_representation() const;

  [[nodiscard]] bool
  state_is_within_restrictions(const ManyBodyState::key_type &) const noexcept;
};
#endif // MANYBODYOPERATOR_H
