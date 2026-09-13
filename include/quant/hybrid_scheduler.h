#pragma once
#include "quant/types.h"
#include <vector>

namespace quant {

// Per-layer attention-kind map. STD = full softmax attention (only kind).
// Owner purge 2026-09-07: variant kinds removed; schedule is all-STD.
enum class HybridAttnKind { STD };

struct HybridSchedule {
    std::vector<HybridAttnKind> layers;
    int num_std = 0;
    int total() const { return (int)layers.size(); }
};

// Generic N-layer layout: all STD. Keeps signature for callers.
HybridSchedule build_hybrid_schedule(int total_layers, int reserved = 3);

} // namespace quant
