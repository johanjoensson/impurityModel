#ifndef SLATER_DETERMINANT_H
#define SLATER_DETERMINANT_H
#include <bitset>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

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
