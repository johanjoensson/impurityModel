#include "ManyBodyState.h"
#include "ManyBodyOperator.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Running C++ unit tests..." << std::endl;

    // 1. Test SlaterDeterminant
    SlaterDeterminant<> s1;
    s1.push_back(1);
    s1.push_back(2);
    assert(s1.size() == 2);
    assert(s1[0] == 1);
    assert(s1[1] == 2);

    SlaterDeterminant<> s2;
    s2.push_back(1);
    s2.push_back(2);
    assert(s1 == s2);

    // 2. Test ManyBodyState
    ManyBodyState state;
    state[s1] = 1.0;
    assert(state.size() == 1);
    assert(state[s1] == 1.0);
    assert(state.norm2() == 1.0);
    assert(state.norm() == 1.0);

    ManyBodyState state2;
    state2[s1] = 2.0;
    
    ManyBodyState sum_state = state + state2;
    assert(sum_state[s1] == 3.0);

    // 3. Test ManyBodyOperator
    // Define a simple operator: c(0), a(1) with amplitude 1.5
    ManyBodyOperator::key_type op_key = {0, -2}; // 0 is c(0), -2 is a(1)
    ManyBodyOperator op;
    op[op_key] = 1.5;
    assert(op.size() == 1);

    SlaterDeterminant<> initial_det;
    initial_det.push_back(1ULL << 62); // bit index for orbital 1: 64 - 1 - 1 = 62.
    
    ManyBodyState psi;
    psi[initial_det] = 1.0;

    ManyBodyState res = op(psi);
    
    std::cout << "res size: " << res.size() << std::endl;
    std::cout << "res string: " << res.to_string() << std::endl;

    SlaterDeterminant<> expected_det;
    expected_det.push_back(1ULL << 63);
    
    std::cout << "expected det string: " << expected_det.to_string() << std::endl;
    std::cout << "expected det chunk 0: " << expected_det[0] << std::endl;
    if (res.size() > 0) {
        for (const auto &p : res) {
            std::cout << "res contains det: " << p.first.to_string() << " with amp " << p.second << std::endl;
            std::cout << "det chunk 0: " << p.first[0] << std::endl;
        }
    }

    assert(res.size() == 1);
    assert(res[expected_det] == -1.5);

    std::cout << "All C++ unit tests passed successfully!" << std::endl;
    return 0;
}
