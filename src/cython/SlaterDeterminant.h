#ifndef SLATER_DETERMINANT_H
#define SLATER_DETERMINANT_H
#include <bitset>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>
#include <random>

/**
 * @struct SlaterDeterminant
 * @brief Represents a many-body state in the form of a Slater determinant.
 *
 * This struct inherits from std::vector<CHUNK> to compactly represent
 * spin-orbital occupations using integer chunks (typically 64-bit).
 *
 * @tparam CHUNK The integer type used for bit-packing the occupations. Defaults to uint64_t.
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
   * @brief Computes a locality-preserving linear hash for MPI routing.
   *
   * Unlike the standard cryptographic-style hash() which disperses states uniformly
   * (leading to a dense all-to-all communication graph), routing_hash() is a linear
   * function over GF(2^64). Hopping terms only change a few bytes, so the number of
   * target MPI ranks any single rank communicates with is strictly bounded, yielding
   * an extremely sparse communication graph that scales to 100,000+ ranks.
   */
  uint64_t routing_hash() const {
    static const std::vector<uint64_t> byte_hash_table = []() {
        std::vector<uint64_t> t(1024 * 256); // Supports up to 8192 orbitals
        std::mt19937_64 rng(0x1337BEEF); // Fixed seed ensures consistent routing across MPI ranks
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
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(this->data());
    size_t num_bytes = this->size() * sizeof(CHUNK);
    for (size_t i = 0; i < num_bytes; ++i) {
        h += byte_hash_table[i * 256 + ptr[i]];
    }
    return h;
  }

  /**
   * @brief Returns a string representation of the Slater determinant.
   * @return std::string formatted as "SlaterDeterminant((chunk1, chunk2, ...))".
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
 * Provides a custom hash function for use in std::unordered_map or other hash-based containers.
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
