#include "quant/hybrid_expert.h"

namespace quant {

HybridExpertShard::HybridExpertShard(const expert::ClusterConfig& cfg, int num_nodes)
    : cfg_(cfg), num_nodes_(std::max(1, num_nodes)) {}

std::vector<expert::ExpertAssignment> HybridExpertShard::assign_layer(int num_experts) const {
    std::vector<expert::ExpertAssignment> out;
    out.reserve(num_experts);
    for (int i = 0; i < num_experts; ++i) {
        expert::ExpertAssignment a;
        a.expert_id = i;
        a.node_id = i % num_nodes_;
        a.capacity = cfg_.max_tokens_per_expert > 0 ? cfg_.max_tokens_per_expert : 1024;
        out.push_back(a);
    }
    return out;
}

std::vector<std::vector<expert::ExpertAssignment>> HybridExpertShard::assign_schedule(
    const HybridSchedule& sched) const {
    int num_experts = (int)cfg_.num_experts;
    if (num_experts <= 0) num_experts = 64;
    std::vector<std::vector<expert::ExpertAssignment>> per_layer;
    per_layer.reserve(sched.total());
    for (int i = 0; i < sched.total(); ++i) {
        per_layer.push_back(assign_layer(num_experts));
    }
    return per_layer;
}

} // namespace quant
