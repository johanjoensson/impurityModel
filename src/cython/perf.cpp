#include "ManyBodyOperator.h"
#include "ManyBodyState.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using Key = ManyBodyState::key_type;
using Amp = ManyBodyState::mapped_type;

// Representative 1-/2-body number-conserving Hamiltonian fixture, matching the Python
// oracle/timing fixture in src/impurityModel/test/test_apply_perf.py (n_orbs = 160,
// 80 electrons, ~2000 SDs, 300 1-body + 300 2-body terms). This is the local-profiling
// twin of that test; the Python side is the durable golden-output regression gate.
//
// Build (serial):
//   g++ -O3 -std=c++17 -I<boost-include> perf.cpp ManyBodyOperator.cpp ManyBodyState.cpp -o perf
// Build (threaded apply, PARALLEL branch in ManyBodyOperator::apply):
//   g++ -O3 -std=c++17 -DPARALLEL -pthread -I<boost-include> \
//       perf.cpp ManyBodyOperator.cpp ManyBodyState.cpp -o perf
// (Set -I to your Boost headers, e.g. $BOOST_ROOT/include, when std::flat_map is absent.)
//
// Operator encoding (ManyBodyOperator::OPS, int64): create orbital o -> o;
// annihilate orbital o -> -(o+1); terms are stored rightmost-operator-first, so
// c^d_i c_j -> {-(j+1), i} and c^d_i c^d_j c_k c_l -> {-(l+1), -(k+1), j, i}.
std::pair<ManyBodyState, ManyBodyOperator> setup() {
  std::mt19937 gen{20260628u}; // fixed seed: reproducible timings
  std::normal_distribution n{0., 1.0};

  const size_t n_states = 2000;
  const uint64_t n_orbs = 160;
  const size_t n_elec = 80;
  const size_t n_1body = 300;
  const size_t n_2body = 300;
  std::uniform_int_distribution<size_t> orb{0, n_orbs - 1};

  const size_t bits_per_chunk = 8 * sizeof(Key::value_type);
  size_t key_size = n_orbs / bits_per_chunk;
  if (n_orbs % bits_per_chunk) {
    key_size++;
  }

  auto set_bit = [&](Key &k, size_t o) {
    // Orbital o -> chunk o/bits_per_chunk, bit (bits_per_chunk-1 - o%bits_per_chunk),
    // matching the C++ create/annihilate MSB-first convention.
    k[o / bits_per_chunk] |=
        static_cast<Key::value_type>(1)
        << (bits_per_chunk - 1 - (o % bits_per_chunk));
  };

  std::vector<Key> keys;
  std::vector<Amp> amps;
  keys.reserve(n_states);
  amps.reserve(n_states);
  for (size_t s = 0; s < n_states; s++) {
    Key tmp;
    tmp.resize(key_size, 0);
    size_t placed = 0;
    while (placed < n_elec) {
      size_t o = orb(gen);
      const auto mask = static_cast<Key::value_type>(1)
                        << (bits_per_chunk - 1 - (o % bits_per_chunk));
      if (!(tmp[o / bits_per_chunk] & mask)) {
        set_bit(tmp, o);
        placed++;
      }
    }
    keys.push_back(std::move(tmp));
    amps.push_back(Amp{n(gen), n(gen)});
  }

  std::vector<ManyBodyOperator::OPS> ops;
  std::vector<ManyBodyOperator::SCALAR> scalars;
  ops.reserve(n_1body + n_2body);
  scalars.reserve(n_1body + n_2body);
  for (size_t t = 0; t < n_1body; t++) {
    const int64_t i = static_cast<int64_t>(orb(gen));
    const int64_t j = static_cast<int64_t>(orb(gen));
    ops.push_back({-(j + 1), i}); // c^d_i c_j
    scalars.push_back(Amp{n(gen), n(gen)});
  }
  for (size_t t = 0; t < n_2body; t++) {
    const int64_t i = static_cast<int64_t>(orb(gen));
    const int64_t j = static_cast<int64_t>(orb(gen));
    const int64_t k = static_cast<int64_t>(orb(gen));
    const int64_t l = static_cast<int64_t>(orb(gen));
    ops.push_back({-(l + 1), -(k + 1), j, i}); // c^d_i c^d_j c_k c_l
    scalars.push_back(Amp{n(gen), n(gen)});
  }

  ManyBodyState psi{std::move(keys), std::move(amps)};
  ManyBodyOperator op{std::move(ops), std::move(scalars)};
  return {psi, op};
}

int main() {
  constexpr size_t n_reps = 10;
  constexpr double cutoff = 0.0;
  auto [psi, op] = setup();

  // Warm up the flat operator representation and report the output size.
  ManyBodyState out = op(psi, cutoff);
  std::cout << "n_in = " << psi.size() << ", n_out = " << out.size() << "\n";

  std::vector<double> ms;
  ms.reserve(n_reps);
  for (size_t i = 0; i < n_reps; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    ManyBodyState res = op(psi, cutoff);
    auto end = std::chrono::high_resolution_clock::now();
    ms.push_back(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1000.0);
  }
  std::sort(ms.begin(), ms.end());
  std::cout << "apply: median = " << ms[n_reps / 2] << " ms, best = " << ms.front()
            << " ms (over " << n_reps << " reps)\n";
  return 0;
}
