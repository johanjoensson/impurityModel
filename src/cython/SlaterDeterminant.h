#ifndef SLATER_DETERMINANT_H
#define SLATER_DETERMINANT_H
#include <bitset>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <random>
#include <string>
#include <vector>

/**
 * @struct SlaterDeterminant
 * @brief Represents a many-body state in the form of a Slater determinant.
 *
 * This struct inherits from std::vector<CHUNK> to compactly represent
 * spin-orbital occupations using integer chunks (typically 64-bit).
 *
 * @tparam CHUNK The integer type used for bit-packing the occupations. Defaults
 * to uint64_t.
 */
template <typename CHUNK = uint64_t>
struct SlaterDeterminant : public std::vector<CHUNK> {

  /**
   * @brief Computes the hash of the SlaterDeterminant.
   * @return A size_t representing the hash value.
   */
  std::size_t hash() const {
    return std::hash<SlaterDeterminant<CHUNK>>{}(*this);
  }

  /**
   * @brief The nibble-popcount routing hash (the default mode).
   *
   * A sum over every 4-orbital nibble of popcount(nibble) * W, one weight per
   * nibble position. Hopping terms change a few bytes, so the number of target
   * MPI ranks any single rank communicates with is strictly bounded, yielding a
   * sparse communication graph at the thousands-of-ranks scale.
   *
   * Because the weight does not depend on WHICH orbital of a nibble is
   * occupied, every determinant with the same per-nibble occupation counts
   * hashes alike. On an LMO-shaped basis (10 impurity + 22 bath orbitals, up to
   * 4 valence holes, 1.26M determinants) that is only ~1,500 distinct hash
   * values, the largest class 1.6x the per-rank mean at 240 ranks, and the
   * busiest rank holds 2.6x the mean (logged: local[min,max] = [1,156, 12,224]
   * of mean 4,752). `routing_hash_orbital` removes that coarseness.
   */
  uint64_t routing_hash_nibble() const {
    static const std::vector<uint64_t> byte_hash_table = []() {
      std::vector<uint64_t> t(1024 * 256); // Supports up to 8192 orbitals
      std::mt19937_64 rng(
          0x1337BEEF); // Fixed seed ensures consistent routing across MPI ranks
      for (size_t i = 0; i < 1024; ++i) {
        uint64_t W0 = rng();
        uint64_t W1 = rng();
        for (int v = 0; v < 256; ++v) {
          uint64_t pop0 = __builtin_popcount(v & 0x0F);
          uint64_t pop1 = __builtin_popcount(v >> 4);
          t[i * 256 + v] = pop0 * W0 + pop1 * W1;
        }
      }
      return t;
    }();

    uint64_t h = 0;
    const uint8_t *ptr = reinterpret_cast<const uint8_t *>(this->data());
    size_t num_bytes = this->size() * sizeof(CHUNK);
    for (size_t i = 0; i < num_bytes; ++i) {
      h += byte_hash_table[i * 256 + ptr[i]];
    }
    return h;
  }

  /**
   * @brief The per-orbital routing hash: sum over occupied orbitals of an
   * independent 64-bit weight.
   *
   * Same cost as the nibble hash (one table lookup per byte) and the same
   * additive structure, so a hop c^dag_i c_j shifts the hash by the
   * determinant-independent constant W_i - W_j and the peer set of a rank stays
   * bounded exactly as before (MatvecExchangePlan relies on that). Every
   * occupation pattern gets its own weight, so the partition is balanced to
   * within sampling noise (max/mean 1.03 at 240 ranks on the basis above,
   * against 2.58). The price is a denser communication graph at the ground-state
   * rank counts (240 peers of 240, against 57-80), where a complete graph costs
   * ~7.6 MiB/rank; at GF colour widths (<= 40) the nibble graph is already complete.
   */
  uint64_t routing_hash_orbital() const {
    static const std::vector<uint64_t> byte_hash_table = []() {
      std::vector<uint64_t> t(1024 * 256);
      std::mt19937_64 rng(0x0B17A1ED); // fixed: identical on every MPI rank
      for (size_t i = 0; i < 1024; ++i) {
        uint64_t W[8];
        for (int b = 0; b < 8; ++b)
          W[b] = rng();
        for (int v = 0; v < 256; ++v) {
          uint64_t sum = 0;
          for (int b = 0; b < 8; ++b)
            if (v & (1 << b))
              sum += W[b];
          t[i * 256 + v] = sum;
        }
      }
      return t;
    }();

    uint64_t h = 0;
    const uint8_t *ptr = reinterpret_cast<const uint8_t *>(this->data());
    size_t num_bytes = this->size() * sizeof(CHUNK);
    for (size_t i = 0; i < num_bytes; ++i) {
      h += byte_hash_table[i * 256 + ptr[i]];
    }
    return h;
  }

  /**
   * @brief The hash that decides which MPI rank owns this determinant
   * (``routing_hash() % comm_size``).
   *
   * Dispatches on IMPURITYMODEL_ROUTING_HASH = ``nibble`` (default) or
   * ``orbital``, read ONCE per process on first use. Every rank of a run must
   * read the same value (ownership is a pure function of the determinant and the
   * rank count); ``config.assert_uniform`` checks that on the Python side.
   * An unrecognised value aborts rather than silently falling back: a rank
   * that fell back would disagree with the others about who owns what.
   */
  uint64_t routing_hash() const {
    static const bool orbital = []() {
      const char *mode = std::getenv("IMPURITYMODEL_ROUTING_HASH");
      if (mode == nullptr || *mode == '\0' || std::string(mode) == "nibble")
        return false;
      if (std::string(mode) == "orbital")
        return true;
      std::fprintf(stderr,
                   "IMPURITYMODEL_ROUTING_HASH=%s: expected 'nibble' or "
                   "'orbital'\n",
                   mode);
      std::abort();
    }();
    return orbital ? routing_hash_orbital() : routing_hash_nibble();
  }

  /**
   * @brief Returns a string representation of the Slater determinant.
   * @return std::string formatted as "SlaterDeterminant((chunk1, chunk2,
   * ...))".
   */
  std::string to_string() const {
    std::string res = "SlaterDeterminant(";
    if (!this->empty()) {
      res += "(";
      for (size_t i = 0; i < this->size() - 1; i++) {
        res += std::to_string((*this)[i]) + ", ";
      }
      res += std::to_string(this->back());
      res += ")";
    }
    res += ")";
    return res;
  }
};

/**
 * @brief Specialization of std::hash for SlaterDeterminant<uint64_t>.
 *
 * Provides a custom hash function for use in std::unordered_map or other
 * hash-based containers.
 */
template <> struct std::hash<SlaterDeterminant<uint64_t>> {
  /**
   * @brief Hash function operator for SlaterDeterminant<uint64_t>.
   * @param det The Slater determinant to hash.
   * @return The hash value.
   */
  std::size_t operator()(const SlaterDeterminant<uint64_t> &det) const {
    uint64_t res = 0x9e3779b97f4a7c15ULL;

    for (uint64_t c : det) {
      c ^= c >> 30;
      c *= 0xbf58476d1ce4e5b9ULL;
      c ^= c >> 27;
      c *= 0x94d049bb133111ebULL;
      c ^= c >> 31;

      res ^= c + 0x9e3779b9 + (res << 6) + (res >> 2);
    }
    return static_cast<std::size_t>(res);
  }
};
#endif // SLATER_DETERMINANT_H
