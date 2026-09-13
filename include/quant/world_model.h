#pragma once
#include "quant/tensor.h"
#include "quant/model.h"
#include "quant/trainer.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <deque>
#include <functional>
#include <random>
#include <algorithm>

namespace quant {
namespace agi {

// ========================================================================
// World Model (DIFFUSION T48)
// ========================================================================

struct StateEntry {
    std::string key;
    float value = 0.0f;
};

struct WorldState {
    std::vector<StateEntry> entries;
    int64_t timestamp = 0;
    float importance = 0.0f;

    float get_value(const std::string& key) const;
    void set_value(const std::string& key, float val);
    std::vector<std::string> get_keys() const;
    std::vector<float> to_vector() const;
    void from_vector(const std::vector<float>& v, const std::vector<std::string>& keys);
};

struct Transition {
    WorldState state;
    std::string action;
    WorldState next_state;
    float reward = 0.0f;
    float predicted_reward = 0.0f;
    float prediction_error = 0.0f;
    int64_t timestamp = 0;
};

struct PlanningBeam {
    std::vector<std::string> action_sequence;
    float cumulative_reward = 0.0f;
    WorldState final_state;
    float confidence = 0.0f;
};

struct PredictionResult {
    WorldState predicted_state;
    float predicted_reward = 0.0f;
    float uncertainty = 0.0f;
    float confidence = 0.0f;
    std::vector<WorldState> ensemble_predictions;
};

struct CounterfactualResult {
    WorldState actual_state;
    WorldState counterfactual_state;
    std::string intervention_description;
    float state_difference = 0.0f;
    float reward_difference = 0.0f;
};

struct MPCPlan {
    std::vector<std::string> action_sequence;
    std::vector<WorldState> predicted_states;
    std::vector<float> predicted_rewards;
    float total_expected_reward = 0.0f;
    int64_t planning_horizon = 0;
};

struct UncertaintyEstimate {
    float epistemic_uncertainty = 0.0f;
    float aleatoric_uncertainty = 0.0f;
    float total_uncertainty = 0.0f;
    int64_t ensemble_size = 0;
    float disagreement_magnitude = 0.0f;
};

struct StateMemoryEntry {
    WorldState state;
    float importance_weight = 0.0f;
    int64_t access_count = 0;
    int64_t last_accessed = 0;
};

struct WorldModelStats {
    int64_t total_transitions = 0;
    int64_t total_predictions = 0;
    float avg_prediction_error = 0.0f;
    float avg_reward_prediction_error = 0.0f;
    int64_t memory_entries = 0;
    int64_t planning_calls = 0;
    float avg_planning_time_ms = 0.0f;
};

class WorldModel {
public:
    WorldModel(Model* model, int64_t state_dim = 64, int64_t ensemble_size = 5);

    // Legacy tensor-step API (G13, kept for quant::WorldModel compat).
    // Null-model safe: returns zeros shaped like state; plan returns {}.
    Tensor simulate_step(const Tensor& state, const Tensor& action);
    std::vector<Tensor> plan(int64_t horizon);

    PredictionResult predict(const WorldState& state, const std::string& action);
    float predict_reward(const WorldState& state, const std::string& action);
    PredictionResult predict_with_uncertainty(const WorldState& state, const std::string& action);

    std::vector<Transition> simulate_trajectory(const WorldState& start_state,
                                                 const std::vector<std::string>& actions);
    void update_transition_model(const Transition& transition);
    void train_on_transitions(const std::vector<Transition>& transitions, int epochs = 10);

    PlanningBeam beam_search_plan(const WorldState& current_state, int64_t horizon,
                                   int64_t beam_width = 10, int64_t n_actions = 10);
    MPCPlan model_predictive_control(const WorldState& current_state, int64_t horizon,
                                      int64_t n_samples = 50);
    std::vector<std::string> plan_actions(const WorldState& current_state, int64_t horizon);

    UncertaintyEstimate estimate_uncertainty(const WorldState& state, const std::string& action);
    float compute_ensemble_disagreement(const std::vector<WorldState>& predictions);

    CounterfactualResult counterfactual_analysis(const WorldState& actual_state,
                                                  const std::string& actual_action,
                                                  const std::string& counterfactual_action);
    std::vector<CounterfactualResult> batch_counterfactual(const WorldState& state,
                                                            const std::string& actual_action,
                                                            const std::vector<std::string>& alternatives);

    void store_state(const WorldState& state, float importance = 1.0f);
    WorldState retrieve_most_relevant(const std::string& query_key);
    WorldState retrieve_most_relevant(const std::vector<float>& query_vector);
    void consolidate_memory(int64_t target_size = 1000);
    std::deque<StateMemoryEntry>& get_memory();

    void add_transition(const Transition& transition);
    std::vector<Transition>& get_transition_buffer();
    WorldState get_current_state() const;
    void set_current_state(const WorldState& state);

    WorldModelStats get_stats() const;
    void reset_stats();

    static constexpr int64_t max_transition_buffer = 100000;
    static constexpr int64_t max_state_memory = 10000;

private:
    Model* model_;
    int64_t state_dim_;
    int64_t ensemble_size_;
    WorldState current_state_;
    std::vector<Transition> transition_buffer_;
    std::deque<StateMemoryEntry> state_memory_;
    WorldModelStats stats_;

    std::vector<std::string> possible_actions_;
    std::mt19937 rng_;

    float state_distance(const WorldState& a, const WorldState& b);
    float cosine_similarity_vec(const std::vector<float>& a, const std::vector<float>& b);
    std::vector<std::string> get_action_space(int64_t n_actions);
    float evaluate_sequence_reward(const WorldState& start, const std::vector<std::string>& actions);
    std::vector<WorldState> ensemble_predict(const WorldState& state, const std::string& action);
    void update_importance_weights();
    std::vector<float> state_to_model_input(const WorldState& state, const std::string& action);
    WorldState model_output_to_state(const float* output, int64_t output_len);
};

} // namespace agi
} // namespace quant
