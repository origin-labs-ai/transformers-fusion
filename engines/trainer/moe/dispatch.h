#pragma once
#include "quant/tensor.h"
#include <cstdint>
#include <string>
#include <vector>

namespace quant {
namespace moe_engine {

// Expert dispatch/combine with real capacity-drop math.
// Tokens route via (expert, weight) pairs; over-capacity tokens drop
// and are reported — no silent loss, no stubs.
struct DispatchItem {
    int64_t token = 0;
    int64_t expert = 0;
    float weight = 0.0f;
};

struct DispatchPlan {
    std::vector<std::vector<DispatchItem>> per_expert;
    std::vector<DispatchItem> dropped;
    int64_t slots_per_expert = 0;
};

DispatchPlan plan_dispatch(const Tensor& indices, const Tensor& weights,
                           int64_t num_experts, int64_t slots_per_expert);

// Gather rows {tokens,D} into per-expert {slots,D} buffers (real copy,
// dropped tokens excluded). Returns per-expert buffers + slot counts.
struct ExpertInputs {
    std::vector<Tensor> bufs;
    std::vector<int64_t> counts;
};
ExpertInputs gather_expert_inputs(const Tensor& hidden, const DispatchPlan& plan,
                                  int64_t num_experts);

// Combine per-expert outputs back to {tokens,D} via weights (real axpy).
// Dropped tokens contribute zero and are counted.
struct CombineOut {
    Tensor hidden;
    int64_t dropped = 0;
};
CombineOut combine_expert_outputs(const std::vector<Tensor>& expert_outs,
                                  const DispatchPlan& plan,
                                  int64_t tokens, int64_t dim);

// Capacity-drop accounting readout.
struct DropStats {
    int64_t assigned = 0;
    int64_t kept = 0;
    int64_t dropped = 0;
    double drop_rate = 0.0;
    std::vector<int64_t> per_expert_kept;
};
DropStats drop_stats(const DispatchPlan& plan, int64_t num_experts);

} // namespace moe_engine
} // namespace quant
