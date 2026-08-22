#pragma once
#include "quant/types.h"
#include <vector>

namespace quant {

enum class HybridAttnKind { KDA, MLA };

struct HybridSchedule {
    std::vector<HybridAttnKind> layers;
    int num_kda = 0;
    int num_mla = 0;
    int total() const { return (int)layers.size(); }
};

HybridSchedule build_hybrid_schedule(int total_layers, int kda_per_mla = 3);
HybridSchedule build_k3_schedule(); // 93 layers, 69 KDA + 24 MLA (3:1)

} // namespace quant
