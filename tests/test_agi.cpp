// test_agi.cpp — AGI subsystems: monitoring, reflection, metacognition, safety
#include "quant/agi.h"
#include "quant/transformer.h"
#include "quant/model.h"
#include "quant/tensor.h"
#include "quant/test.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <string>

using namespace quant;

int main() {
    TEST_SUITE("agi");
    printf("=== AGI subsystems test ===\n\n");

    TransformerConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = 8;
    cfg.ffn_hidden_size = 32;
    cfg.max_seq_len = 32;

    DenseModel model(cfg);
    Tensor logits(Shape{2, 64}, DType::F32);
    for (int64_t i = 0; i < logits.numel(); i++)
        logits.data<float>()[i] = (float)(i % 9) / 9.0f - 0.4f;

    // SelfMonitor
    {
        agi::SelfMonitor monitor(&model);
        float conf = monitor.estimate_confidence(logits);
        TEST_CHECK(conf >= 0.0f && conf <= 1.0f, "confidence in [0,1]");

        auto state = monitor.analyze("hello world", "hi there");
        TEST_CHECK(!state.recommendation.empty(), "monitor produces recommendation");
        monitor.get_confidence_stats().update(conf);
        monitor.get_confidence_stats().update(conf * 0.9f);
        TEST_CHECK(monitor.get_confidence_stats().mean > 0.0f, "rolling stats mean > 0");

        bool drift = monitor.detect_drift(logits, logits);
        TEST_CHECK(!drift, "identical logits not flagged as drift");
    }

    // SelfReflector
    {
        agi::SelfReflector reflector(&model);
        std::vector<float> confs = {0.9f, 0.8f, 0.7f, 0.6f};
        std::vector<bool> corr = {true, true, false, false};
        float ece = reflector.ece_calibration(confs, corr);
        printf("  ECE: %.4f\n", ece);
        TEST_CHECK(ece >= 0.0f && ece <= 1.0f, "ECE in [0,1]");
        float cons = reflector.consistency_check("a b c", "a b c");
        TEST_CHECK(cons > 0.5f, "identical outputs high consistency");
    }

    // MetaCognition
    {
        agi::MetaCognition meta(&model);
        float kbs = meta.knowledge_boundary_score("what is the weather");
        TEST_CHECK(kbs > 0.0f, "knowledge boundary score > 0");
        float ep = meta.epistemic_uncertainty(logits);
        float al = meta.aleatoric_uncertainty(logits);
        float tot = meta.total_uncertainty(logits);
        TEST_CHECK(std::isfinite(ep) && std::isfinite(al), "uncertainties finite");
        TEST_CHECK(tot >= std::max(ep, al) - 1e-3f, "total >= max(epistemic, aleatoric)");
        TEST_CHECK(!meta.should_refuse("what is 2+2", 0.99f), "simple query not refused");
        meta.self_model_update("what is 2+2", 0.9f, true);
        TEST_CHECK(meta.get_self_model_accuracy() > 0.0f, "self model accuracy updated");
    }

    // SafetyGuardrails
    {
        agi::SafetyGuardrails sg;
        TEST_CHECK(sg.check_output("regular output"), "regular output passes");
        TEST_CHECK(sg.check_input("hello"), "regular input passes");
        sg.set_invariant("no_harm", "must not contain harm");
        TEST_CHECK(sg.get_invariants().size() == 1, "one invariant registered");
        TEST_CHECK(sg.check_invariant("no_harm"), "invariant checked");
        TEST_CHECK(sg.human_override("apply", "test override"), "human override granted");
        sg.audit_log("modify", "test change", "ok");
        TEST_CHECK(sg.get_audit_log().size() == 2, "audit log records override + change");
        sg.set_kill_switch(true);
        TEST_CHECK(sg.is_killed(), "kill switch engaged");
    }

    // HITL + Alignment
    {
        agi::HITL hitl;
        TEST_CHECK(!hitl.is_paused(), "HITL not paused by default");
        hitl.pause();
        TEST_CHECK(hitl.is_paused(), "HITL paused");

        agi::AlignmentSystem align;
        float score = align.value_alignment_score("helpful response");
        TEST_CHECK(score >= 0.0f && score <= 1.0f, "alignment score in [0,1]");
    }

    // MultiAgentCoordinator — message passing (no model call)
    {
        agi::MultiAgentCoordinator coord(&model);
        coord.send_message(0, 1, "analyze task", 0.9f);
        coord.send_message(1, 0, "done", 0.8f);
        TEST_CHECK(coord.get_message_queue(1).size() == 1, "agent 1 has 1 message");
        TEST_CHECK(coord.get_message_queue(0).size() == 1, "agent 0 has 1 message");
        auto msg = coord.receive_message(1);
        TEST_CHECK(msg.sender_id == 0 && msg.receiver_id == 1, "message routed correctly");
        std::string rn = coord.role_name(agi::MultiAgentCoordinator::ANALYST);
        TEST_CHECK(rn == "Analyst", "role name mapping");
    }

    // PlanningEngine — topological sort + confidence
    {
        agi::PlanningEngine planner(&model);
        std::vector<agi::PlanStep> steps;
        agi::PlanStep b, a;
        b.action = "build b"; b.dependencies = {"build a"};
        a.action = "build a";
        steps.push_back(b);
        steps.push_back(a);
        auto sorted = planner.topological_sort(steps);
        TEST_CHECK(sorted.size() == 2, "topological sort returns all steps");
        float conf = planner.estimate_confidence(sorted);
        TEST_CHECK(conf >= 0.0f && conf <= 1.0f, "plan confidence in [0,1]");
        planner.clear_execution_history();
        TEST_CHECK(planner.get_execution_history().empty(), "execution history cleared");
    }

    // MemorySystem — store 2 similar entries, consolidate merges them.
    // (PROD: was TEST_CHECK(true) — vacuous. consolidate() merges entries
    // with cosine sim > 0.95, so retrieval still works after merging.)
    {
        agi::MemorySystem mem(100);
        Tensor key(Shape{1, 4}, DType::F32);
        Tensor val(Shape{1, 4}, DType::F32);
        key.fill(1.0f);
        val.fill(2.0f);
        mem.store(key, val);
        Tensor out = mem.retrieve(key, 1);
        TEST_CHECK(out.numel() > 0, "memory retrieves stored entry");
        mem.store(key, val); // near-duplicate (sim = 1.0 > 0.95)
        mem.consolidate();
        Tensor out2 = mem.retrieve(key, 1);
        TEST_CHECK(out2.numel() > 0, "memory retrieves after consolidation");
        const float* d = out2.data<float>();
        bool finite = true;
        for (int64_t i = 0; i < out2.numel(); i++)
            if (!std::isfinite(d[i])) finite = false;
        TEST_CHECK(finite, "post-consolidation retrieval finite");
    }

    // ToolUse
    {
        agi::ToolUse tools(&model);
        auto avail = tools.get_available_tools();
        TEST_CHECK(!avail.empty(), "tool registry non-empty");
    }

    // RollingStats anomaly detection
    {
        agi::RollingStats rs;
        rs.window_size = 10;
        for (int i = 0; i < 10; i++) rs.update(1.0f);
        TEST_CHECK(!rs.is_anomalous(1.05f, 3.0f), "in-distribution value not anomalous");
        TEST_CHECK(rs.is_anomalous(100.0f, 3.0f), "outlier flagged anomalous");
    }

    int failures = TEST_REPORT();
    printf("\nAGI TEST %s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures > 0 ? 1 : 0;
}
