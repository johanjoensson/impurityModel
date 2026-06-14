#include "MpiUtils.h"
#include <unordered_set>
#include <numeric>

namespace mpi_utils {

void pack_determinants(
    const std::vector<SlaterDeterminant<uint64_t>>& dets,
    int comm_size,
    std::vector<int64_t>& send_counts,
    std::vector<uint64_t>& state_buf) {
    
    send_counts.assign(comm_size, 0);
    if (dets.empty()) return;
    
    size_t chunks_per_state = dets[0].size();
    
    // Group determinants by destination rank
    std::vector<std::vector<const SlaterDeterminant<uint64_t>*>> rank_dets(comm_size);
    // Determine unique states and their target ranks
    std::unordered_set<SlaterDeterminant<uint64_t>> unique_dets(dets.begin(), dets.end());
    for (const auto& det : unique_dets) {
        int rank = det.hash() % comm_size;
        rank_dets[rank].push_back(&det);
        send_counts[rank]++;
    }
    
    // Pack into flat buffer
    size_t total_states = unique_dets.size();
    state_buf.reserve(total_states * chunks_per_state);
    state_buf.clear();
    
    for (int r = 0; r < comm_size; r++) {
        for (const auto* det : rank_dets[r]) {
            state_buf.insert(state_buf.end(), det->begin(), det->end());
        }
    }
}

std::vector<std::vector<SlaterDeterminant<uint64_t>>> unpack_determinants(
    int comm_size,
    const std::vector<int64_t>& recv_counts,
    const std::vector<uint64_t>& state_buf,
    size_t chunks_per_state) {
    
    std::vector<std::vector<SlaterDeterminant<uint64_t>>> result(comm_size);
    size_t offset = 0;
    
    for (int r = 0; r < comm_size; r++) {
        int64_t count = recv_counts[r];
        result[r].reserve(count);
        for (int64_t i = 0; i < count; i++) {
            SlaterDeterminant<uint64_t> det;
            det.reserve(chunks_per_state);
            for (size_t c = 0; c < chunks_per_state; c++) {
                det.push_back(state_buf[offset++]);
            }
            result[r].push_back(det);
        }
    }
    return result;
}

void pack_psis(
    const std::vector<const ManyBodyState*>& psis,
    int comm_size,
    std::vector<int64_t>& send_counts,
    std::vector<uint64_t>& state_buf,
    std::vector<double>& amp_buf_reim,
    std::vector<int32_t>& psi_buf) {
    
    send_counts.assign(comm_size, 0);
    
    size_t chunks_per_state = 0;
    
    struct Entry {
        const SlaterDeterminant<uint64_t>* state;
        std::complex<double> amp;
        int32_t psi_idx;
    };
    
    std::vector<std::vector<Entry>> rank_entries(comm_size);
    
    for (size_t pi = 0; pi < psis.size(); pi++) {
        const auto* psi = psis[pi];
        if (!psi) continue;
        for (auto it = psi->begin(); it != psi->end(); ++it) {
            const auto& state = it->first;
            if (chunks_per_state == 0) chunks_per_state = state.size();
            int rank = state.hash() % comm_size;
            rank_entries[rank].push_back({&state, it->second, static_cast<int32_t>(pi)});
            send_counts[rank]++;
        }
    }
    
    size_t total_entries = 0;
    for (int64_t c : send_counts) total_entries += c;
    
    state_buf.reserve(total_entries * chunks_per_state);
    state_buf.clear();
    amp_buf_reim.reserve(total_entries * 2);
    amp_buf_reim.clear();
    psi_buf.reserve(total_entries);
    psi_buf.clear();
    
    for (int r = 0; r < comm_size; r++) {
        for (const auto& entry : rank_entries[r]) {
            state_buf.insert(state_buf.end(), entry.state->begin(), entry.state->end());
            amp_buf_reim.push_back(entry.amp.real());
            amp_buf_reim.push_back(entry.amp.imag());
            psi_buf.push_back(entry.psi_idx);
        }
    }
}

void unpack_psis(
    std::vector<ManyBodyState*>& psis,
    int comm_size,
    const std::vector<int64_t>& recv_counts,
    const std::vector<uint64_t>& state_buf,
    const std::vector<double>& amp_buf_reim,
    const std::vector<int32_t>& psi_buf,
    size_t chunks_per_state) {
    
    size_t offset = 0;
    for (int r = 0; r < comm_size; r++) {
        int64_t count = recv_counts[r];
        for (int64_t i = 0; i < count; i++) {
            SlaterDeterminant<uint64_t> det;
            det.reserve(chunks_per_state);
            for (size_t c = 0; c < chunks_per_state; c++) {
                det.push_back(state_buf[offset * chunks_per_state + c]);
            }
            
            int32_t pi = psi_buf[offset];
            std::complex<double> amp(amp_buf_reim[2*offset], amp_buf_reim[2*offset+1]);
            
            if (pi >= 0 && pi < static_cast<int32_t>(psis.size()) && psis[pi]) {
                auto [it, inserted] = psis[pi]->try_emplace(det, amp);
                if (!inserted) {
                    it->second += amp;
                }
            }
            offset++;
        }
    }
}

} // namespace mpi_utils
