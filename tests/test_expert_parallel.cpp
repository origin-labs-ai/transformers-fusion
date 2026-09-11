// test_expert_parallel.cpp — ExpertDispatcher routing + ExpertParallel partition
#include "quant/expert_parallel.h"
#include "quant/moe_model.h"
#include "quant/moe_variants.h"
#include "quant/tensor.h"
#include "quant/test.h"
#include <vector>
#include <cmath>
#include <cstdio>

using namespace quant;

int main() {
    TEST_SUITE("expert_parallel");
    printf("=== Expert parallel test ===\n\n");

    // --- ExpertDispatcher: routing math (no sockets) ---
    expert::ClusterConfig ccfg;
    ccfg.hidden_size = 16;
    ccfg.num_experts = 4;
    ccfg.top_k = 2;

    expert::ExpertDispatcher dispatcher(ccfg);
    dispatcher.assign_experts_to_nodes(2);
    auto assignments = dispatcher.get_all_assignments();
    TEST_CHECK(assignments.size() == 4, "4 experts assigned");
    for (auto& a : assignments) {
        TEST_CHECK(a.node_id >= 0 && a.node_id < 2, "expert node in range");
    }

    // Route 8 tokens over 4 experts
    const int64_t T = 8, E = 4;
    Tensor logits(Shape{T, E}, DType::F32);
    unsigned int seed = 3;
    for (int64_t i = 0; i < T * E; i++) {
        seed = seed * 1103515245u + 12345u;
        logits.data<float>()[i] = (float)((seed >> 16) & 0xFFFF) / 65535.0f;
    }

    expert::RoutingDecision r = dispatcher.route_tokens(logits, T, 0, 2);
    TEST_CHECK(r.expert_indices.numel() == T * ccfg.top_k, "expert indices T*K");
    TEST_CHECK(r.tokens_per_expert.size() == E, "per-expert token counts");
    float lb = 0.0f, zl = 0.0f;
    dispatcher.compute_load_balance(r, lb, zl);
    TEST_CHECK(lb >= 0.0f, "load balance loss >= 0");
    TEST_CHECK(zl >= 0.0f, "z loss >= 0");

    // AllToAll plan for local node
    expert::AllToAllPlan plan = dispatcher.plan_all_to_all(r, 0);
    TEST_CHECK(plan.total_bytes >= 0, "all-to-all plan non-negative bytes");

    // --- ExpertParallel (moe_model.h): partition experts across ranks ---
    ExpertParallel ep(8, 2, 0);
    TEST_CHECK(ep.num_experts() == 8, "total experts 8");
    TEST_CHECK(ep.num_ranks() == 2, "2 ranks");
    TEST_CHECK(ep.num_local_experts() == 4, "4 local experts per rank");
    TEST_CHECK(ep.local_experts().size() == 4, "local expert list size 4");

    ExpertParallel ep2(8, 2, 1);
    auto l0 = ep.local_experts();
    auto l1 = ep2.local_experts();
    bool disjoint = true;
    for (int e0 : l0)
        for (int e1 : l1)
            if (e0 == e1) disjoint = false;
    TEST_CHECK(disjoint, "rank expert sets are disjoint");

    // --- PipelineScheduler: stage bookkeeping ---
    // (PROD: was TEST_CHECK(true) after reset — vacuous. reset() clears
    // ready flags AND step count, so re-scheduling must work like fresh.)
    expert::PipelineScheduler ps(2, 4);
    Tensor input(Shape{4, 16}, DType::F32);
    ps.schedule_forward(input, 0);
    TEST_CHECK(ps.all_stages_done() == false, "pipeline not done after one stage");
    ps.reset();
    TEST_CHECK(ps.all_stages_done() == true, "pipeline all-done after reset (flags cleared)");
    ps.schedule_forward(input, 0);
    TEST_CHECK(ps.all_stages_done() == false, "pipeline re-schedules after reset");

    int failures = TEST_REPORT();
    printf("\nEXPERT PARALLEL TEST %s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures > 0 ? 1 : 0;
}
