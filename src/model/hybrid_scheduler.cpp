#include "quant/hybrid_scheduler.h"

namespace quant {

// All-STD layout (owner purge 2026-09-07). Keeps signature for callers.
HybridSchedule build_hybrid_schedule(int total_layers, int reserved) {
    HybridSchedule s;
    if (total_layers <= 0) return s;
    if (reserved < 0) reserved = 0;
    if (reserved > total_layers) reserved = total_layers;
    s.layers.reserve((size_t)total_layers);
    for (int i = 0; i < total_layers; ++i) {
        s.layers.push_back(HybridAttnKind::STD);
        s.num_std++;
    }
    return s;
}

} // namespace quant
