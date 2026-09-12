#pragma once
#include "quant/tensor.h"
#include "quant/model.h"
#include "quant/trainer.h"
#include "quant/reward.h"
#include "quant/world_model.h"
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <chrono>
#include <fstream>
#include <deque>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <queue>
#include <map>
#include <set>
#include <sstream>

namespace quant {
namespace agi {

struct MetaCognitionState {
    float confidence = 0;
    float uncertainty = 0;
    int reasoning_depth = 0;
    bool needs_reanalysis = false;
    std::string recommendation;
    std::vector<std::string> reasoning_chain;
    std::vector<float> token_confidences;
    float epistemic_uncertainty = 0;
    float aleatoric_uncertainty = 0;
    float knowledge_boundary_score = 0;
    float reasoning_quality_score = 0;
    bool should_refuse = false;
    std::string refusal_reason;
};

struct RollingStats {
    std::deque<float> recent_values;
    size_t window_size = 100;
    float mean = 0.0f;
    float variance = 0.0f;
    float min_val = INFINITY;
    float max_val = -INFINITY;
    void update(float value);
    float z_score(float value) const;
    bool is_anomalous(float value, float threshold = 3.0f) const;
};

struct AlertEntry {
    int64_t timestamp;
    std::string severity;
    std::string source;
    std::string message;
    float value;
};

struct AuditEntry {
    int64_t timestamp;
    std::string action;
    std::string detail;
    std::string result;
};

struct Message {
    int sender_id;
    int receiver_id;
    std::string role;
    std::string content;
    int64_t round;
    float confidence;
};

struct PlanStep {
    std::string action;
    std::vector<std::string> dependencies;
    float confidence = 1.0f;
    bool completed = false;
    std::string result;
};

struct SandboxResult {
    bool compiled = false;
    bool passed = false;
    float score = 0.0f;
    std::string stdout_capture;
    std::string stderr_capture;
    double runtime_ms = 0.0;
    int exit_code = -1;
};

struct FlywheelIteration {
    int iter = 0;
    std::string task;
    std::string solution;
    bool compiled = false;
    bool verified = false;
    float delta = 0.0f;
    bool applied = false;
    bool rolled_back = false;
    std::string file;
    int line = 0;
    std::string proof;
    double runtime_ms = 0.0;
};

// Subsystem 1: SelfMonitor — confidence, drift, anomaly, rolling stats, alerts
class SelfMonitor {
public:
    SelfMonitor(Model* model);
    MetaCognitionState analyze(const std::string& input, const std::string& output);
    float estimate_confidence(const Tensor& logits);
    bool detect_drift(const Tensor& current_logits, const Tensor& reference_logits);
    float distribution_shift(const Tensor& current, const Tensor& reference);
    bool check_anomaly(const MetaCognitionState& state);
    RollingStats& get_confidence_stats() { return confidence_stats_; }
    RollingStats& get_uncertainty_stats() { return uncertainty_stats_; }
    std::vector<AlertEntry> get_recent_alerts(int count = 10) const;
    void clear_alerts();
private:
    Model* model_;
    RollingStats confidence_stats_;
    RollingStats uncertainty_stats_;
    std::vector<AlertEntry> alerts_;
    mutable std::mutex alert_mutex_;
    void add_alert(const std::string& severity, const std::string& source,
                   const std::string& message, float value);
};

// Subsystem 2: SelfReflection — consistency, ECE calibration, error ID
class SelfReflector {
public:
    SelfReflector(Model* model);
    std::string reflect(const std::string& input, const std::string& output);
    std::string refine(const std::string& original, const std::string& reflection);
    float consistency_check(const std::string& output1, const std::string& output2);
    float ece_calibration(const std::vector<float>& confidences,
                          const std::vector<bool>& correctness, int n_bins = 10);
    std::vector<std::string> identify_errors(const std::string& input,
                                              const std::string& output,
                                              const std::string& expected);
private:
    Model* model_;
};

// Subsystem 3: MetaCognition — knowledge boundaries, uncertainty, refusal, reasoning quality, self-model update
class MetaCognition {
public:
    MetaCognition(Model* model);
    float knowledge_boundary_score(const std::string& query);
    float epistemic_uncertainty(const Tensor& logits);
    float aleatoric_uncertainty(const Tensor& logits);
    float total_uncertainty(const Tensor& logits);
    bool should_refuse(const std::string& query, float uncertainty_threshold = 0.7f);
    float reasoning_quality(const std::vector<std::string>& chain);
    void self_model_update(const std::string& query, float confidence,
                           bool was_correct);
    float get_self_model_accuracy() const { return self_model_accuracy_; }
private:
    Model* model_;
    float self_model_accuracy_ = 0.0f;
    int self_model_updates_ = 0;
    std::unordered_map<std::string, float> knowledge_map_;
};

// Subsystem 4: RSI — evaluate->weaknesses->candidates->evaluate->accept->repeat
class RecursiveSelfImprover {
public:
    RecursiveSelfImprover(Model* model, Trainer* trainer);
    void improvement_cycle(int iterations = 10);
    bool self_modify(const std::string& analysis);
    std::vector<std::string> evaluate_weaknesses(const std::string& eval_data);
    std::vector<std::string> generate_candidates(const std::vector<std::string>& weaknesses);
    float evaluate_candidate(const std::string& candidate);
    bool accept_candidate(const std::string& candidate, float score);
    void adjust_lr(const std::string& weakness);
    void apply_perturbation(float magnitude = 0.01f);
    void adjust_format(const std::string& analysis);
    void apply_dropout(float rate = 0.1f);
    void apply_pruning(float threshold = 0.01f);
private:
    Model* model_;
    Trainer* trainer_;
    float learning_rate_ = 1e-4f;
    float dropout_rate_ = 0.0f;
};

// Subsystem 5: Sandbox — static analysis, resource limits, correctness verification.
//
// HONESTY CONTRACT (bug census): compile_and_test / verify_correctness /
// benchmark / run_with_timeout REQUIRE compiling+executing untrusted code,
// which needs an owner-gated process-execution path that does not exist in
// this tree. They are fail-closed stubs: always report failure (false /
// compiled=false / exit_code=-1) + stderr note, NEVER a fake pass. Callers
// must treat false as "could not verify", not "code is bad". static_analysis
// and check_resource_limits ARE real (pure text analysis).
class Sandbox {
public:
    Sandbox();
    std::vector<std::string> static_analysis(const std::string& code);
    bool check_resource_limits(const std::string& code, int64_t max_memory_mb = 256,
                               double max_time_sec = 10.0);
    bool verify_correctness(const std::string& code, const std::vector<std::string>& test_cases);
    SandboxResult compile_and_test(const std::string& code, const std::string& task);
    bool benchmark(const std::string& code, double& ops_per_sec, double& avg_latency_ms);
    std::string get_sandbox_path() const;
private:
    std::string sandbox_dir_;
    std::mutex sandbox_mutex_;
    std::string generate_test_program(const std::string& code, const std::string& task);
    bool run_with_timeout(const std::string& binary, double timeout_sec,
                          std::string& stdout_out, std::string& stderr_out, int& exit_code);
    int count_tests_passed(const std::string& stdout_str);
};

// Subsystem 6: SafetyGuardrails — modification/output checks, immutable invariants, human override, audit log
class SafetyGuardrails {
public:
    SafetyGuardrails();
    bool check_output(const std::string& output);
    bool check_input(const std::string& input);
    bool check_modification(const std::string& target, const std::string& change);
    bool check_invariant(const std::string& invariant_name);
    void set_invariant(const std::string& name, const std::string& expression);
    std::vector<std::string> get_invariants() const;
    bool human_override(const std::string& action, const std::string& reason);
    void audit_log(const std::string& action, const std::string& detail, const std::string& result);
    std::vector<AuditEntry> get_audit_log(int count = 50) const;
    void set_kill_switch(bool kill) { kill_switch_ = kill; }
    bool is_killed() const { return kill_switch_; }
    bool is_paused() const { return paused_; }
    void set_paused(bool p) { paused_ = p; }
private:
    bool kill_switch_ = false;
    bool paused_ = false;
    std::vector<std::string> blocked_patterns_;
    std::unordered_map<std::string, std::string> invariants_;
    std::vector<AuditEntry> audit_log_;
    mutable std::mutex audit_mutex_;
    int64_t current_timestamp();
};

// Subsystem 7: MultiAgentCoordinator — Analyst, Implementer, Verifier, Critic, Synthesizer
class MultiAgentCoordinator {
public:
    enum AgentRole {
        ANALYST = 0,
        IMPLEMENTER = 1,
        VERIFIER = 2,
        CRITIC = 3,
        SYNTHESIZER = 4,
        NUM_ROLES = 5
    };

    MultiAgentCoordinator(Model* model);
    std::string run_task(const std::string& task_description, int max_critic_rounds = 3);
    void send_message(int sender, int receiver, const std::string& content, float confidence);
    Message receive_message(int agent_id);
    std::vector<Message> get_message_queue(int agent_id) const;
    std::vector<std::string> get_agent_history(int agent_id) const;
    void clear_agent_history(int agent_id);
    bool is_converged() const { return converged_; }
    int get_round() const { return current_round_; }

    static std::string role_name(AgentRole role) {
        static const char* names[] = {"Analyst", "Implementer", "Verifier", "Critic", "Synthesizer"};
        return names[role];
    }

private:
    Model* model_;
    std::vector<std::vector<Message>> message_queues_;
    std::vector<std::vector<std::string>> agent_histories_;
    int current_round_ = 0;
    bool converged_ = false;
    std::mutex coord_mutex_;

    std::string analyst_phase(const std::string& task);
    std::string implementer_phase(const std::string& analysis);
    std::string verifier_phase(const std::string& implementation);
    std::string critic_phase(const std::string& implementation, const std::string& verification);
    std::string synthesizer_phase(const std::vector<std::string>& inputs);
    std::string call_model(const std::string& prompt);
};

// Subsystem 8: PlanningEngine — decomposition, topological sort, confidence, replanning, execution
class PlanningEngine {
public:
    PlanningEngine(Model* model);
    std::vector<PlanStep> plan(const std::string& goal, int max_steps = 20);
    bool execute(const std::vector<PlanStep>& plan);
    std::vector<PlanStep> decompose(const std::string& goal, int depth = 3);
    std::vector<PlanStep> topological_sort(const std::vector<PlanStep>& steps);
    float estimate_confidence(const std::vector<PlanStep>& plan);
    std::vector<PlanStep> replan(const std::vector<PlanStep>& failed_plan,
                                  const std::string& failure_reason, int max_steps = 10);
    std::vector<PlanStep> get_execution_history() const { return execution_history_; }
    void clear_execution_history();
private:
    Model* model_;
    std::vector<PlanStep> execution_history_;
    std::mutex plan_mutex_;
    bool execute_step(const PlanStep& step);
};

// HITL
class HITL {
public:
    HITL();
    bool request_approval(const std::string& action);
    void pause() { paused_ = true; }
    void resume() { paused_ = false; }
    bool is_paused() const { return paused_; }
    std::vector<AuditEntry> get_approval_history() const { return approval_history_; }
private:
    bool paused_ = false;
    std::vector<AuditEntry> approval_history_;
};

// Alignment
class AlignmentSystem {
public:
    AlignmentSystem();
    float value_alignment_score(const std::string& output);
    static constexpr int max_loop_iterations = 100;
};

// CodeGenSelfImprover
class CodeGenSelfImprover {
public:
    CodeGenSelfImprover(Model* model);
    std::string generate_kernel(const std::string& op, int64_t M, int64_t N, int64_t K);
    bool compile_and_test(const std::string& code);
    bool replace_kernel(const std::string& op, const std::string& new_code);
private:
    Model* model_;
};

// SelfVerifier
class SelfVerifier {
public:
    SelfVerifier(Model* model);
    bool verify(const std::string& problem, const std::string& solution);
    std::vector<std::string> find_edge_cases(const std::string& solution);
private:
    Model* model_;
};

// CapabilityAmplifier
class CapabilityAmplifier {
public:
    CapabilityAmplifier(Model* model);
    float measure(const std::string& capability);
    bool improve(const std::string& capability, int steps = 100);
private:
    Model* model_;
};

// WorldModel — canonical definition lives in quant/world_model.h (DIFFUSION T48,
// full predict/plan/memory API plus legacy Tensor simulate_step/plan compat).
// agi.h includes it at top; the old G13 stub was removed to end the ODR clash.
// CuriosityDrivenExplorer
class CuriosityDrivenExplorer {
public:
    CuriosityDrivenExplorer(Model* model);
    Tensor intrinsic_reward(const Tensor& state);
    std::vector<int> explore(int64_t n_steps);
private:
    Model* model_;
    std::vector<Tensor> visited_states_;
};

// NAS
struct Architecture { int layers; int hidden; float score; };
class NeuralArchitectureSearch {
public:
    NeuralArchitectureSearch();
    Architecture search(int population = 50, int generations = 20);
private:
    Architecture mutate(const Architecture& arch);
    float evaluate(const Architecture& arch);
};

// HPOptimizer
class HPOptimizer {
public:
    HPOptimizer(Trainer* trainer);
    void population_based_training(int n_population = 8, int n_generations = 10);
private:
    Trainer* trainer_;
};

// ContinuousLearner
class ContinuousLearner {
public:
    ContinuousLearner(Model* model);
    void update(const Tensor& new_data);
    bool prevent_forgetting(float threshold = 0.95f);
private:
    Model* model_;
    std::vector<Tensor> exemplars_;
};

// KnowledgeDistillation
class KnowledgeDistillation {
public:
    KnowledgeDistillation(Model* teacher, Model* student);
    void distill(const DataLoader& data, int steps);
private:
    Model *teacher_, *student_;
};

// PromptOptimizer
class PromptOptimizer {
public:
    PromptOptimizer(Model* model);
    std::string optimize(const std::string& task, int n_iterations = 20);
    float evaluate(const std::string& prompt, const std::string& task);
private:
    Model* model_;
};

// ChainOfThought
class ChainOfThought {
public:
    ChainOfThought(Model* model);
    std::string reason(const std::string& problem, int max_steps = 10);
    std::vector<std::string> get_chain() const { return chain_; }
private:
    Model* model_;
    std::vector<std::string> chain_;
};

// ToolUse
class ToolUse {
public:
    ToolUse(Model* model);
    struct Tool { std::string name; std::string description; };
    std::string call_tool(const std::string& tool_name, const std::string& args);
    std::vector<std::string> get_available_tools() const;
private:
    Model* model_;
    std::vector<Tool> tools_;
};

// MemorySystem
class MemorySystem {
public:
    MemorySystem(int64_t capacity = 10000);
    void store(const Tensor& key, const Tensor& value);
    Tensor retrieve(const Tensor& query, int k = 5);
    void consolidate();
private:
    std::vector<std::pair<Tensor, Tensor>> memory_;
    int64_t capacity_;
};

// EvaluationHarness
class EvaluationHarness {
public:
    EvaluationHarness(Model* model);
    struct Result { float accuracy; float loss; int samples; };
    Result evaluate(const std::string& benchmark_name, int n_samples = 100);
    std::vector<Result> evaluate_all();
private:
    Model* model_;
};

// Flywheel
class Flywheel {
public:
    Flywheel(Model* model, Trainer* trainer, CodeGenSelfImprover* codegen = nullptr,
             SelfVerifier* verifier = nullptr, CapabilityAmplifier* amplifier = nullptr,
             SafetyGuardrails* safety = nullptr);
    void run(int max_iters = 100);
    const std::vector<FlywheelIteration>& get_history() const { return history_; }
    int get_no_improvement_count() const { return no_improvement_count_; }
    std::string get_log_path() const;
    // P17 test hook (additive): public wrapper over private self_play() so
    // tests can assert non-empty / round-robin without friending or macros.
    std::string self_play_for_test();
private:
    std::string self_play();
    SandboxResult sandbox_compile_and_test(const std::string& code, const std::string& task);
    float measure_improvement(const std::string& task, const std::string& solution);
    bool apply_improvement(const std::string& original, const std::string& improved, const std::string& target_file);
    bool rollback(const std::string& file_path, const std::string& backup_path);
    void log_iteration(const FlywheelIteration& iter);
    std::string generate_fallback_solution(const std::string& task);
    std::string generate_test_program(const std::string& code, const std::string& task);
    std::string extract_proof(const std::string& solution);
    bool run_with_timeout(const std::string& binary, double timeout_sec, std::string& stdout_out, std::string& stderr_out, int& exit_code);
    std::string make_diff(const std::string& original, const std::string& improved);
    bool check_convergence();
    std::string sandbox_path() const;
    int count_tests_passed(const std::string& stdout_str);
    std::string generate_benchmark_harness(const std::string& code, const std::string& task, int n_iterations = 1000);
    int calculate_cyclomatic_complexity(const std::string& code);
    int measure_nesting_depth(const std::string& code);
    float estimate_code_quality(const std::string& code);
    bool sandbox_benchmark(const std::string& code, const std::string& task, double& ops_per_sec, double& avg_latency_ms);
    std::string generate_multi_file_test(const std::vector<std::pair<std::string, std::string>>& files, const std::string& task);
    Model* model_;
    Trainer* trainer_;
    CodeGenSelfImprover* codegen_;
    SelfVerifier* verifier_;
    CapabilityAmplifier* amplifier_;
    SafetyGuardrails* safety_;
    int no_improvement_count_ = 0;
    int converged_count_ = 0;
    std::vector<FlywheelIteration> history_;
};

struct RLHFIntegration {
    static void run_with_flywheel(Model* model, Model* ref_model,
                                   Tokenizer* tokenizer, Trainer* trainer,
                                   RewardModel* reward_model,
                                   Optimizer* policy_opt, Optimizer* rm_opt,
                                   int n_rounds = 3, int n_prompts = 50);
};

namespace util {
    inline std::vector<int> simple_encode(const std::string& text, int vocab_size) {
        std::vector<int> ids;
        int offset = 5;
        int mod = std::max(1, vocab_size - offset);
        for (char c : text) ids.push_back((int)(unsigned char)c % mod + offset);
        return ids;
    }
    inline std::string simple_decode(const std::vector<int>& ids) {
        std::string s;
        for (int id : ids) { int c = id - 5; s += (c >= 0 && c < 256) ? (char)c : '?'; }
        return s;
    }
} // namespace util

} // namespace agi

// Legacy top-level alias: quant::WorldModel == quant::agi::WorldModel.
// Wave-7 smoke (test_wave7_missing) constructs quant::WorldModel(nullptr) and
// calls Tensor simulate_step/plan; the alias keeps one implementation.
using WorldModel = agi::WorldModel;

} // namespace quant