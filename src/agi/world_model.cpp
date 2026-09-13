// ========================================================================
// world_model.cpp — World Model (DIFFUSION T48)
// ========================================================================
#include "quant/world_model.h"
#include "quant/agi_utils.h"
#include "quant/random.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <numeric>
#include <ctime>
#include <iomanip>
#include <cstdio>
#include <cstring>
#include <queue>

namespace quant {
namespace agi {

// ========================================================================
// WorldState methods
// ========================================================================
float WorldState::get_value(const std::string& key) const {
    for (auto& e : entries) {
        if (e.key == key) return e.value;
    }
    return 0.0f;
}

void WorldState::set_value(const std::string& key, float val) {
    for (auto& e : entries) {
        if (e.key == key) { e.value = val; return; }
    }
    entries.push_back({key, val});
}

std::vector<std::string> WorldState::get_keys() const {
    std::vector<std::string> keys;
    for (auto& e : entries) keys.push_back(e.key);
    return keys;
}

std::vector<float> WorldState::to_vector() const {
    std::vector<float> v;
    for (auto& e : entries) v.push_back(e.value);
    return v;
}

void WorldState::from_vector(const std::vector<float>& v, const std::vector<std::string>& keys) {
    entries.clear();
    int64_t n = std::min((int64_t)v.size(), (int64_t)keys.size());
    for (int64_t i = 0; i < n; i++) {
        entries.push_back({keys[i], v[i]});
    }
}

// ========================================================================
// Constructor
// ========================================================================
WorldModel::WorldModel(Model* model, int64_t state_dim, int64_t ensemble_size)
    : model_(model), state_dim_(state_dim), ensemble_size_(ensemble_size),
      rng_(42) {

    possible_actions_ = {
        "move_forward", "move_backward", "turn_left", "turn_right",
        "accelerate", "decelerate", "interact", "observe",
        "plan", "execute", "wait", "explore",
        "gather_info", "process_data", "communicate", "analyze",
        "construct", "deconstruct", "optimize", "random_walk"
    };

    current_state_.timestamp = 0;
    current_state_.importance = 1.0f;
    current_state_.entries.push_back({"x", 0.0f});
    current_state_.entries.push_back({"y", 0.0f});
    current_state_.entries.push_back({"energy", 1.0f});
    current_state_.entries.push_back({"knowledge", 0.0f});
}

// ========================================================================
// Utility methods
// ========================================================================
float WorldModel::state_distance(const WorldState& a, const WorldState& b) {
    auto va = a.to_vector();
    auto vb = b.to_vector();
    int64_t n = std::min((int64_t)va.size(), (int64_t)vb.size());
    if (n == 0) return 0.0f;
    float dist = 0;
    for (int64_t i = 0; i < n; i++) dist += (va[i] - vb[i]) * (va[i] - vb[i]);
    return std::sqrt(dist);
}

float WorldModel::cosine_similarity_vec(const std::vector<float>& a, const std::vector<float>& b) {
    int64_t n = std::min((int64_t)a.size(), (int64_t)b.size());
    if (n == 0) return 0.0f;
    float dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = std::sqrt(na * nb);
    return denom > 1e-10f ? dot / denom : 0.0f;
}

std::vector<std::string> WorldModel::get_action_space(int64_t n_actions) {
    std::vector<std::string> actions;
    int64_t n = std::min(n_actions, (int64_t)possible_actions_.size());
    for (int64_t i = 0; i < n; i++) actions.push_back(possible_actions_[i]);
    return actions;
}

std::vector<float> WorldModel::state_to_model_input(const WorldState& state, const std::string& action) {
    std::vector<float> input;
    auto sv = state.to_vector();
    for (auto v : sv) input.push_back(v);

    std::vector<float> action_encoding(state_dim_, 0.0f);
    unsigned int hash = 0;
    for (char c : action) hash = hash * 31 + (unsigned int)c;
    int64_t action_idx = (int64_t)(hash % (unsigned int)state_dim_);
    action_encoding[action_idx] = 1.0f;
    for (auto v : action_encoding) input.push_back(v);

    while ((int64_t)input.size() < state_dim_ * 3) input.push_back(0.0f);
    return input;
}

WorldState WorldModel::model_output_to_state(const float* output, int64_t output_len) {
    WorldState state;
    state.timestamp = current_state_.timestamp + 1;
    state.importance = 0.5f;

    std::vector<std::string> keys = current_state_.get_keys();
    int64_t n = std::min(output_len, (int64_t)keys.size());
    for (int64_t i = 0; i < n; i++) {
        state.entries.push_back({keys[i], output[i]});
    }

    if (state.entries.empty()) {
        state.entries = current_state_.entries;
        for (auto& e : state.entries) e.value += 0.01f;
    }

    return state;
}

// ========================================================================
// Legacy tensor-step API (G13 compat, null-model safe)
// ========================================================================
Tensor WorldModel::simulate_step(const Tensor& state, const Tensor& action) {
    if (!model_) { Tensor empty(state.shape()); empty.zero_(); return empty; }
    int vocab_size = (int)model_->config.vocab_size;
    int64_t state_n = state.numel();
    int64_t action_n = action.numel();
    int64_t total_n = std::min(state_n + action_n, (int64_t)64);
    std::vector<float> combined((size_t)total_n);
    const float* sd = state.data<float>();
    const float* ad = action.data<float>();
    for (int64_t i = 0; i < std::min(state_n, total_n); i++) combined[(size_t)i] = sd[i];
    for (int64_t i = 0; i < std::min(action_n, total_n - state_n); i++)
        combined[(size_t)(state_n + i)] = ad[i];
    Tensor input_ids({1, total_n});
    Tensor positions({1, total_n});
    float* idp = input_ids.data<float>();
    float* psp = positions.data<float>();
    for (int64_t i = 0; i < total_n; i++) {
        idp[i] = std::fmod(combined[(size_t)i] * 1000.0f, (float)vocab_size);
        if (idp[i] < 0) idp[i] += (float)vocab_size;
        psp[i] = (float)i;
    }
    Tensor logits = model_->forward(input_ids, positions, nullptr);
    int64_t V = logits.dim(logits.rank() - 1);
    Tensor next_state(state.shape());
    float* nsd = next_state.data<float>();
    int64_t out_n = std::min(state_n, total_n);
    for (int64_t i = 0; i < out_n; i++) {
        const float* lp = logits.data<float>() + i * V;
        int max_idx = greedy_argmax(lp, (int)V);
        nsd[i] = (float)max_idx / (float)V;
    }
    return next_state;
}

std::vector<Tensor> WorldModel::plan(int64_t horizon) {
    if (!model_) return {};
    std::vector<Tensor> plan_steps;
    int64_t dim = model_->config.hidden_size;
    Tensor current_state({dim}); current_state.zero_();
    RNG rng(42);
    for (int64_t h = 0; h < horizon; h++) {
        Tensor action({dim});
        float* ad = action.data<float>();
        for (int64_t i = 0; i < dim; i++) ad[i] = rng.uniform() * 2.0f - 1.0f;
        Tensor next_state = simulate_step(current_state, action);
        plan_steps.push_back(next_state.clone());
        current_state = next_state;
    }
    return plan_steps;
}

// ========================================================================
// Prediction
// ========================================================================
PredictionResult WorldModel::predict(const WorldState& state, const std::string& action) {
    PredictionResult result;
    if (!model_) {
        result.predicted_state = state;
        result.predicted_state.timestamp = state.timestamp + 1;
        result.predicted_reward = 0.0f;
        result.uncertainty = 1.0f;
        result.confidence = 0.0f;
        return result;
    }

    auto input = state_to_model_input(state, action);
    int64_t input_len = std::min((int64_t)input.size(), (int64_t)256);

    Tensor input_ids({1, input_len});
    Tensor positions({1, input_len});
    float* idp = input_ids.data<float>();
    float* psp = positions.data<float>();
    for (int64_t i = 0; i < input_len; i++) {
        int vocab_size = (int)model_->config.vocab_size;
        idp[i] = std::fmod(input[i] * 1000.0f, (float)vocab_size);
        if (idp[i] < 0) idp[i] += (float)vocab_size;
        psp[i] = (float)i;
    }

    Tensor logits = model_->forward(input_ids, positions, nullptr);
    int64_t V = logits.dim(logits.rank() - 1);
    const float* lp = logits.data<float>();

    int64_t out_len = std::min(state_dim_, V);
    std::vector<float> output_vals(out_len);
    for (int64_t i = 0; i < out_len; i++) {
        output_vals[i] = (float)greedy_argmax(lp + i * V, (int)V) / (float)V;
    }

    result.predicted_state = model_output_to_state(output_vals.data(), out_len);

    float max_l = -INFINITY;
    for (int64_t v = 0; v < std::min(V, (int64_t)100); v++)
        max_l = std::max(max_l, lp[v]);
    float sum = 0;
    for (int64_t v = 0; v < std::min(V, (int64_t)100); v++)
        sum += std::exp(lp[v] - max_l);
    float top_prob = std::exp(lp[0] - max_l) / (sum + 1e-10f);
    result.confidence = top_prob;

    float entropy = 0;
    for (int64_t v = 0; v < std::min(V, (int64_t)100); v++) {
        float p = std::exp(lp[v] - max_l) / (sum + 1e-10f);
        if (p > 1e-10f) entropy -= p * std::log(p);
    }
    float max_entropy = std::log((float)std::min(V, (int64_t)100));
    result.uncertainty = max_entropy > 0 ? entropy / max_entropy : 0.0f;
    result.predicted_reward = result.confidence * 0.5f + (1.0f - result.uncertainty) * 0.5f;

    result.ensemble_predictions = ensemble_predict(state, action);

    stats_.total_predictions++;
    return result;
}

float WorldModel::predict_reward(const WorldState& state, const std::string& action) {
    auto result = predict(state, action);
    return result.predicted_reward;
}

PredictionResult WorldModel::predict_with_uncertainty(const WorldState& state, const std::string& action) {
    auto result = predict(state, action);

    auto& ensemble = result.ensemble_predictions;
    if (ensemble.size() > 1) {
        result.uncertainty = compute_ensemble_disagreement(ensemble);
        result.confidence = 1.0f - result.uncertainty;
    }

    return result;
}

std::vector<WorldState> WorldModel::ensemble_predict(const WorldState& state, const std::string& action) {
    std::vector<WorldState> predictions;
    int64_t n = std::min(ensemble_size_, (int64_t)10);

    for (int64_t i = 0; i < n; i++) {
        WorldState noisy_state = state;
        std::normal_distribution<float> noise(0.0f, 0.01f * (float)(i + 1));
        for (auto& e : noisy_state.entries) {
            e.value += noise(rng_);
        }

        auto result = predict(noisy_state, action);
        predictions.push_back(result.predicted_state);
    }

    return predictions;
}

// ========================================================================
// Trajectory simulation
// ========================================================================
std::vector<Transition> WorldModel::simulate_trajectory(const WorldState& start_state,
                                                         const std::vector<std::string>& actions) {
    std::vector<Transition> trajectory;
    WorldState current = start_state;

    for (auto& action : actions) {
        auto pred = predict(current, action);
        Transition t;
        t.state = current;
        t.action = action;
        t.next_state = pred.predicted_state;
        t.predicted_reward = pred.predicted_reward;
        t.prediction_error = 0.0f;
        t.timestamp = current.timestamp + 1;
        trajectory.push_back(t);
        current = pred.predicted_state;
    }

    return trajectory;
}

void WorldModel::update_transition_model(const Transition& transition) {
    auto pred = predict(transition.state, transition.action);
    float error = state_distance(pred.predicted_state, transition.next_state);
    float reward_error = std::abs(pred.predicted_reward - transition.reward);

    stats_.avg_prediction_error = (stats_.avg_prediction_error * stats_.total_predictions + error) /
                                   (stats_.total_predictions + 1);
    stats_.avg_reward_prediction_error = (stats_.avg_reward_prediction_error * stats_.total_predictions + reward_error) /
                                          (stats_.total_predictions + 1);
    stats_.total_transitions++;
}

void WorldModel::train_on_transitions(const std::vector<Transition>& transitions, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (auto& t : transitions) {
            update_transition_model(t);
        }
    }
}

// ========================================================================
// Planning — beam search
// ========================================================================
PlanningBeam WorldModel::beam_search_plan(const WorldState& current_state, int64_t horizon,
                                           int64_t beam_width, int64_t n_actions) {
    auto actions = get_action_space(n_actions);

    struct BeamEntry {
        std::vector<std::string> actions;
        WorldState state;
        float reward;
    };

    std::vector<BeamEntry> beam;
    beam.push_back({{}, current_state, 0.0f});

    for (int64_t h = 0; h < horizon; h++) {
        std::vector<BeamEntry> candidates;

        for (auto& entry : beam) {
            for (auto& action : actions) {
                auto pred = predict(entry.state, action);

                BeamEntry new_entry;
                new_entry.actions = entry.actions;
                new_entry.actions.push_back(action);
                new_entry.state = pred.predicted_state;
                new_entry.reward = entry.reward + pred.predicted_reward;

                candidates.push_back(new_entry);
            }
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](auto& a, auto& b) { return a.reward > b.reward; });

        int64_t keep = std::min(beam_width, (int64_t)candidates.size());
        beam.clear();
        for (int64_t i = 0; i < keep; i++) beam.push_back(candidates[i]);
    }

    PlanningBeam best;
    if (!beam.empty()) {
        best.action_sequence = beam[0].actions;
        best.cumulative_reward = beam[0].reward;
        best.final_state = beam[0].state;
        best.confidence = 1.0f;
    }
    return best;
}

// ========================================================================
// Planning — MPC
// ========================================================================
MPCPlan WorldModel::model_predictive_control(const WorldState& current_state, int64_t horizon,
                                              int64_t n_samples) {
    MPCPlan plan;
    plan.planning_horizon = horizon;

    auto actions = get_action_space((int64_t)possible_actions_.size());
    std::uniform_int_distribution<int64_t> action_dist(0, (int64_t)actions.size() - 1);

    float best_total_reward = -INFINITY;
    std::vector<std::string> best_actions;
    std::vector<WorldState> best_states;
    std::vector<float> best_rewards;

    for (int64_t s = 0; s < n_samples; s++) {
        WorldState sim_state = current_state;
        std::vector<std::string> sample_actions;
        std::vector<WorldState> sample_states;
        std::vector<float> sample_rewards;
        float total_reward = 0;

        for (int64_t h = 0; h < horizon; h++) {
            int64_t aidx = action_dist(rng_);
            std::string action = actions[aidx];

            auto pred = predict(sim_state, action);
            sample_actions.push_back(action);
            sample_states.push_back(pred.predicted_state);
            sample_rewards.push_back(pred.predicted_reward);
            total_reward += pred.predicted_reward;
            sim_state = pred.predicted_state;
        }

        if (total_reward > best_total_reward) {
            best_total_reward = total_reward;
            best_actions = sample_actions;
            best_states = sample_states;
            best_rewards = sample_rewards;
        }
    }

    plan.action_sequence = best_actions;
    plan.predicted_states = best_states;
    plan.predicted_rewards = best_rewards;
    plan.total_expected_reward = best_total_reward;

    stats_.planning_calls++;
    return plan;
}

std::vector<std::string> WorldModel::plan_actions(const WorldState& current_state, int64_t horizon) {
    auto beam = beam_search_plan(current_state, horizon, 5, 10);
    return beam.action_sequence;
}

// ========================================================================
// Uncertainty estimation
// ========================================================================
UncertaintyEstimate WorldModel::estimate_uncertainty(const WorldState& state, const std::string& action) {
    UncertaintyEstimate est;

    auto ensemble_states = ensemble_predict(state, action);
    est.ensemble_size = (int64_t)ensemble_states.size();
    est.disagreement_magnitude = compute_ensemble_disagreement(ensemble_states);
    est.epistemic_uncertainty = est.disagreement_magnitude;
    est.aleatoric_uncertainty = 0.1f;

    auto base_pred = predict(state, action);
    est.aleatoric_uncertainty = base_pred.uncertainty * 0.3f;
    est.total_uncertainty = est.epistemic_uncertainty * 0.6f + est.aleatoric_uncertainty * 0.4f;

    return est;
}

float WorldModel::compute_ensemble_disagreement(const std::vector<WorldState>& predictions) {
    if (predictions.size() <= 1) return 0.0f;

    WorldState mean_state;
    int64_t n_entries = 0;

    if (!predictions.empty() && !predictions[0].entries.empty()) {
        n_entries = (int64_t)predictions[0].entries.size();
        mean_state.entries.resize(n_entries);
        for (int64_t i = 0; i < n_entries; i++) {
            mean_state.entries[i].key = predictions[0].entries[i].key;
            mean_state.entries[i].value = 0.0f;
        }
    }

    for (auto& pred : predictions) {
        for (int64_t i = 0; i < std::min((int64_t)pred.entries.size(), n_entries); i++) {
            mean_state.entries[i].value += pred.entries[i].value;
        }
    }

    for (int64_t i = 0; i < n_entries; i++) {
        mean_state.entries[i].value /= (float)predictions.size();
    }

    float total_variance = 0;
    for (auto& pred : predictions) {
        for (int64_t i = 0; i < std::min((int64_t)pred.entries.size(), n_entries); i++) {
            float diff = pred.entries[i].value - mean_state.entries[i].value;
            total_variance += diff * diff;
        }
    }

    total_variance /= (float)(predictions.size() * std::max(n_entries, (int64_t)1));
    return std::sqrt(total_variance);
}

// ========================================================================
// Counterfactual reasoning
// ========================================================================
CounterfactualResult WorldModel::counterfactual_analysis(const WorldState& actual_state,
                                                          const std::string& actual_action,
                                                          const std::string& counterfactual_action) {
    CounterfactualResult result;
    result.intervention_description = "If '" + actual_action + "' was replaced with '" + counterfactual_action + "'";

    auto actual_pred = predict(actual_state, actual_action);
    auto cf_pred = predict(actual_state, counterfactual_action);

    result.actual_state = actual_pred.predicted_state;
    result.counterfactual_state = cf_pred.predicted_state;
    result.state_difference = state_distance(actual_pred.predicted_state, cf_pred.predicted_state);
    result.reward_difference = cf_pred.predicted_reward - actual_pred.predicted_reward;

    return result;
}

std::vector<CounterfactualResult> WorldModel::batch_counterfactual(const WorldState& state,
                                                                    const std::string& actual_action,
                                                                    const std::vector<std::string>& alternatives) {
    std::vector<CounterfactualResult> results;
    for (auto& alt : alternatives) {
        results.push_back(counterfactual_analysis(state, actual_action, alt));
    }
    return results;
}

// ========================================================================
// State memory
// ========================================================================
void WorldModel::store_state(const WorldState& state, float importance) {
    StateMemoryEntry entry;
    entry.state = state;
    entry.importance_weight = importance;
    entry.access_count = 0;
    entry.last_accessed = state.timestamp;
    state_memory_.push_back(entry);

    if ((int64_t)state_memory_.size() > max_state_memory) {
        update_importance_weights();
        std::sort(state_memory_.begin(), state_memory_.end(),
                  [](auto& a, auto& b) { return a.importance_weight > b.importance_weight; });
        state_memory_.resize(max_state_memory / 2);
    }
    // Keep the stats counter in sync (get_stats is const and cannot derive
    // it; Wave-7 smoke asserts memory_entries >= 1 after one store).
    stats_.memory_entries = (int64_t)state_memory_.size();
}

WorldState WorldModel::retrieve_most_relevant(const std::string& query_key) {
    float best_score = -INFINITY;
    WorldState best_state;
    int64_t best_idx = -1;

    for (int64_t i = 0; i < (int64_t)state_memory_.size(); i++) {
        auto& entry = state_memory_[i];
        float score = 0;

        for (auto& e : entry.state.entries) {
            if (e.key == query_key) {
                score += std::abs(e.value) * 2.0f;
            }
        }

        score += entry.importance_weight * 0.5f;
        float recency = 1.0f / (1.0f + (float)(current_state_.timestamp - entry.last_accessed));
        score += recency * 0.3f;

        if (score > best_score) {
            best_score = score;
            best_state = entry.state;
            best_idx = i;
        }
    }

    if (best_idx >= 0) {
        state_memory_[best_idx].access_count++;
        state_memory_[best_idx].last_accessed = current_state_.timestamp;
    }

    return best_state;
}

WorldState WorldModel::retrieve_most_relevant(const std::vector<float>& query_vector) {
    float best_score = -INFINITY;
    WorldState best_state;
    int64_t best_idx = -1;

    for (int64_t i = 0; i < (int64_t)state_memory_.size(); i++) {
        auto& entry = state_memory_[i];
        auto sv = entry.state.to_vector();
        float sim = cosine_similarity_vec(query_vector, sv);
        float score = sim + entry.importance_weight * 0.2f;

        if (score > best_score) {
            best_score = score;
            best_state = entry.state;
            best_idx = i;
        }
    }

    if (best_idx >= 0) {
        state_memory_[best_idx].access_count++;
        state_memory_[best_idx].last_accessed = current_state_.timestamp;
    }

    return best_state;
}

void WorldModel::update_importance_weights() {
    for (auto& entry : state_memory_) {
        int64_t recency = current_state_.timestamp - entry.last_accessed;
        float recency_score = 1.0f / (1.0f + (float)recency * 0.01f);
        float access_score = std::min(1.0f, (float)entry.access_count / 10.0f);
        entry.importance_weight = recency_score * 0.4f + access_score * 0.3f + entry.importance_weight * 0.3f;
    }
}

void WorldModel::consolidate_memory(int64_t target_size) {
    if ((int64_t)state_memory_.size() <= target_size) return;

    update_importance_weights();

    std::sort(state_memory_.begin(), state_memory_.end(),
              [](auto& a, auto& b) { return a.importance_weight > b.importance_weight; });

    std::vector<StateMemoryEntry> consolidated;
    for (int64_t i = 0; i < target_size && i < (int64_t)state_memory_.size(); i++) {
        consolidated.push_back(state_memory_[i]);
    }

    state_memory_.clear();
    for (auto& entry : consolidated) state_memory_.push_back(entry);
    stats_.memory_entries = (int64_t)state_memory_.size();
}

std::deque<StateMemoryEntry>& WorldModel::get_memory() {
    return state_memory_;
}

// ========================================================================
// Transition buffer
// ========================================================================
void WorldModel::add_transition(const Transition& transition) {
    transition_buffer_.push_back(transition);
    if ((int64_t)transition_buffer_.size() > max_transition_buffer) {
        transition_buffer_.erase(transition_buffer_.begin());
    }
    stats_.total_transitions++;
}

std::vector<Transition>& WorldModel::get_transition_buffer() {
    return transition_buffer_;
}

WorldState WorldModel::get_current_state() const {
    return current_state_;
}

void WorldModel::set_current_state(const WorldState& state) {
    current_state_ = state;
}

WorldModelStats WorldModel::get_stats() const {
    return stats_;
}

void WorldModel::reset_stats() {
    stats_ = WorldModelStats{};
}

float WorldModel::evaluate_sequence_reward(const WorldState& start, const std::vector<std::string>& actions) {
    WorldState current = start;
    float total_reward = 0;
    for (auto& action : actions) {
        float reward = predict_reward(current, action);
        total_reward += reward;
        auto pred = predict(current, action);
        current = pred.predicted_state;
    }
    return total_reward;
}

} // namespace agi
} // namespace quant
