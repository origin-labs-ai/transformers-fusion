#pragma once
#include "quant/hybrid_scheduler.h"
#include "quant/expert_parallel.h"
#include <vector>

namespace quant {

// HybridExpertShard — maps hybrid experts onto cluster nodes for HybridModel.
// Uses round-robin assignment balanced per layer-kind so each
// node's expert load is statistically uniform. Wraps ExpertParallel cluster
// but testable without network (dry-run assignments).
class HybridExpertShard {
public:
    explicit HybridExpertShard(const expert::ClusterConfig& cfg, int num_nodes);

    // Per-layer expert assignments for a full HybridSchedule (93 layers).
    // Each layer shares same expert set but assignment is recomputed per layer
    // for load isolation.
    std::vector<std::vector<expert::ExpertAssignment>> assign_schedule(
        const HybridSchedule& sched) const;

    // Single layer assignment: num_experts distributed round-robin.
    std::vector<expert::ExpertAssignment> assign_layer(int num_experts) const;

    int num_nodes() const { return num_nodes_; }

private:
    expert::ClusterConfig cfg_;
    int num_nodes_;
};

} // namespace quant
