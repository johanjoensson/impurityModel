#ifndef MANYBODYOPERATOR_H
#define MANYBODYOPERATOR_H
#include "ManyBodyState.h"
#include <complex>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

class ManyBodyOperator {

private:
  using SCALAR = std::complex<double>;
  using OPS = std::vector<int64_t>;
  std::vector<std::pair<OPS, SCALAR>> m_ops;
  std::map<ManyBodyState::key_type, ManyBodyState, ManyBodyState::key_compare>
      m_memory;

public:
  ManyBodyOperator() = default;
  ManyBodyOperator(const ManyBodyOperator &) = default;
  ManyBodyOperator(ManyBodyOperator &&) = default;
  ~ManyBodyOperator() = default;

  ManyBodyOperator &operator=(const ManyBodyOperator &) = default;
  ManyBodyOperator &operator=(ManyBodyOperator &&) = default;
  ManyBodyOperator(const std::vector<std::pair<OPS, SCALAR>> &);
  ManyBodyOperator(std::vector<std::pair<OPS, SCALAR>> &&);

  void add_ops(const std::vector<std::pair<OPS, SCALAR>> &ops);
  void add_ops(std::vector<std::pair<OPS, SCALAR>> &&ops);
  ManyBodyState operator()(const ManyBodyState &) const;
};
#endif // MANYBODYOPERATOR_H
