#include "quant/agi.h"
#include "quant/agi_utils.h"
#include "quant/random.h"
#include "quant/optimizer.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <numeric>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <cstring>
#include <ctime>
#include <array>
#include <mutex>
#include <iomanip>
#include <thread>
#include <chrono>
#include <regex>
#include <cstdio>
#include <memory>

namespace quant {
namespace agi {

// ========================================================================
// RollingStats implementation
// ========================================================================
void RollingStats::update(float value) {
    recent_values.push_back(value);
    if (recent_values.size() > window_size) {
        recent_values.pop_front();
    }
    size_t n = recent_values.size();
    if (n == 0) return;
    float sum = 0;
    min_val = INFINITY;
    max_val = -INFINITY;
    for (float v : recent_values) {
        sum += v;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    mean = sum / (float)n;
    float sq_sum = 0;
    for (float v : recent_values) {
        float d = v - mean;
        sq_sum += d * d;
    }
    variance = sq_sum / (float)n;
}

float RollingStats::z_score(float value) const {
    float stddev = std::sqrt(variance + 1e-10f);
    return (value - mean) / stddev;
}

bool RollingStats::is_anomalous(float value, float threshold) const {
    if (recent_values.size() < 5) return false;
    float raw_stddev = std::sqrt(variance);
    if (raw_stddev < 1e-6f) {
        // Degenerate case: near-zero variance (e.g. a constant series). The tiny
        // epsilon would blow up any z-score, so fall back to a relative-deviation
        // rule: flag values that deviate from the mean by more than threshold*|mean|.
        float scale = std::max(std::abs(mean), 1e-3f);
        return std::abs(value - mean) > threshold * scale;
    }
    float stddev = raw_stddev + 1e-10f;
    return std::abs((value - mean) / stddev) > threshold;
}

// ========================================================================
// Subsystem 1: SelfMonitor — confidence, drift, anomaly, rolling stats, alerts
// ========================================================================
SelfMonitor::SelfMonitor(Model* model) : model_(model) {}

float SelfMonitor::estimate_confidence(const Tensor& logits) {
    int64_t V = logits.dim(logits.rank() - 1);
    int64_t S = logits.numel() / V;
    const float* lp = logits.data<float>();
    float total_conf = 0;
    for (int64_t i = 0; i < S; i++) {
        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++) max_l = std::max(max_l, lp[i * V + v]);
        float sum = 0;
        for (int64_t v = 0; v < V; v++) sum += std::exp(lp[i * V + v] - max_l);
        float max_prob = std::exp(lp[i * V + 0] - max_l) / (sum + 1e-10f);
        total_conf += max_prob;
    }
    float conf = total_conf / (float)(S > 0 ? S : 1);
    confidence_stats_.update(conf);
    return conf;
}

MetaCognitionState SelfMonitor::analyze(const std::string& input, const std::string& output) {
    MetaCognitionState state;
    if (!model_) {
        state.confidence = 0.0f;
        state.uncertainty = 1.0f;
        state.token_confidences = {0.0f};
        state.recommendation = "no model";
        state.needs_reanalysis = false;
        state.reasoning_depth = 1;
        state.epistemic_uncertainty = 1.0f;
        state.aleatoric_uncertainty = 0.0f;
        state.knowledge_boundary_score = 0.0f;
        state.reasoning_quality_score = 0.0f;
        state.should_refuse = false;
        return state;
    }
    int vocab_size = (int)model_->config.vocab_size;
    auto input_ids = simple_encode(input, vocab_size);
    auto output_ids = simple_encode(output, vocab_size);
    int64_t seq_len = std::max((int64_t)1, (int64_t)output_ids.size());

    Tensor input_tensor({1, seq_len});
    Tensor pos_tensor({1, seq_len});
    float* idp = input_tensor.data<float>();
    float* psp = pos_tensor.data<float>();
    for (int64_t i = 0; i < seq_len; i++) {
        idp[i] = (float)output_ids[i % (int)output_ids.size()];
        psp[i] = (float)i;
    }

    Tensor logits = model_->forward(input_tensor, pos_tensor, nullptr);
    int64_t V = logits.dim(logits.rank() - 1);
    const float* lp = logits.data<float>();

    float total_conf = 0;
    float total_entropy = 0;
    state.token_confidences.clear();

    for (int64_t i = 0; i < seq_len; i++) {
        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++) max_l = std::max(max_l, lp[i * V + v]);
        float sum = 0;
        for (int64_t v = 0; v < V; v++) sum += std::exp(lp[i * V + v] - max_l);
        float max_prob = std::exp(lp[i * V + 0] - max_l) / (sum + 1e-10f);
        total_conf += max_prob;
        state.token_confidences.push_back(max_prob);
        float entropy = 0;
        for (int64_t v = 0; v < V; v++) {
            float p = std::exp(lp[i * V + v] - max_l) / (sum + 1e-10f);
            if (p > 1e-10f) entropy -= p * std::log(p);
        }
        total_entropy += entropy;
    }

    state.confidence = total_conf / (float)seq_len;
    confidence_stats_.update(state.confidence);

    float max_entropy = std::log((float)V + 1e-10f);
    state.uncertainty = total_entropy / (float)seq_len / (max_entropy > 0 ? max_entropy : 1.0f);
    uncertainty_stats_.update(state.uncertainty);

    state.reasoning_depth = (int)output_ids.size();
    float ratio = (float)output.size() / (float)std::max(input.size(), (size_t)1);
    state.needs_reanalysis = ratio < 0.3f || ratio > 3.0f || state.confidence < 0.5f;

    if (state.needs_reanalysis) {
        state.recommendation = "reanalyze";
    } else if (state.confidence > 0.8f) {
        state.recommendation = "proceed";
    } else {
        state.recommendation = "verify with caution";
    }

    state.epistemic_uncertainty = 1.0f - state.confidence;
    state.aleatoric_uncertainty = state.uncertainty * 0.5f;
    state.knowledge_boundary_score = state.confidence * (1.0f - state.uncertainty);
    state.reasoning_quality_score = state.confidence * (1.0f - state.uncertainty * 0.3f);

    if (confidence_stats_.is_anomalous(state.confidence)) {
        add_alert("warning", "SelfMonitor",
                  "Confidence anomaly detected: " + std::to_string(state.confidence),
                  state.confidence);
    }
    if (uncertainty_stats_.is_anomalous(state.uncertainty)) {
        add_alert("info", "SelfMonitor",
                  "Uncertainty spike detected: " + std::to_string(state.uncertainty),
                  state.uncertainty);
    }
    if (state.confidence < 0.3f) {
        add_alert("critical", "SelfMonitor",
                  "Critically low confidence: " + std::to_string(state.confidence),
                  state.confidence);
    }

    return state;
}

bool SelfMonitor::detect_drift(const Tensor& current_logits, const Tensor& reference_logits) {
    float shift = distribution_shift(current_logits, reference_logits);
    bool drifted = shift > 0.15f;
    if (drifted) {
        add_alert("warning", "SelfMonitor",
                  "Distribution drift detected: shift=" + std::to_string(shift), shift);
    }
    return drifted;
}

float SelfMonitor::distribution_shift(const Tensor& current, const Tensor& reference) {
    int64_t n = std::min(current.numel(), reference.numel());
    if (n == 0) return 0.0f;
    const float* cd = current.data<float>();
    const float* rd = reference.data<float>();
    float kl_sum = 0;
    int count = 0;
    for (int64_t i = 0; i < n; i += 64) {
        int64_t end = std::min(i + 64, n);
        float sum_c = 0, sum_r = 0;
        for (int64_t j = i; j < end; j++) {
            sum_c += cd[j] * cd[j];
            sum_r += rd[j] * rd[j];
        }
        if (sum_c > 0 && sum_r > 0) {
            for (int64_t j = i; j < end; j++) {
                float pc = (cd[j] * cd[j]) / (sum_c + 1e-10f);
                float pr = (rd[j] * rd[j]) / (sum_r + 1e-10f);
                if (pc > 1e-10f && pr > 1e-10f) {
                    kl_sum += pc * std::log(pc / (pr + 1e-10f));
                }
                count++;
            }
        }
    }
    return count > 0 ? kl_sum / (float)count : 0.0f;
}

bool SelfMonitor::check_anomaly(const MetaCognitionState& state) {
    bool confidence_anom = confidence_stats_.is_anomalous(state.confidence);
    bool uncertainty_anom = uncertainty_stats_.is_anomalous(state.uncertainty);
    bool low_conf = state.confidence < 0.2f;
    bool high_uncert = state.uncertainty > 0.8f;
    return confidence_anom || uncertainty_anom || (low_conf && high_uncert);
}

std::vector<AlertEntry> SelfMonitor::get_recent_alerts(int count) const {
    std::lock_guard<std::mutex> lock(alert_mutex_);
    std::vector<AlertEntry> recent;
    int start = std::max(0, (int)alerts_.size() - count);
    for (int i = start; i < (int)alerts_.size(); i++) {
        recent.push_back(alerts_[i]);
    }
    return recent;
}

void SelfMonitor::clear_alerts() {
    std::lock_guard<std::mutex> lock(alert_mutex_);
    alerts_.clear();
}

void SelfMonitor::add_alert(const std::string& severity, const std::string& source,
                             const std::string& message, float value) {
    std::lock_guard<std::mutex> lock(alert_mutex_);
    AlertEntry entry;
    entry.timestamp = std::time(nullptr);
    entry.severity = severity;
    entry.source = source;
    entry.message = message;
    entry.value = value;
    alerts_.push_back(entry);
    if (alerts_.size() > 1000) {
        alerts_.erase(alerts_.begin(), alerts_.begin() + (alerts_.size() - 1000));
    }
}

// ========================================================================
// Subsystem 2: SelfReflection — consistency, ECE calibration, error ID
// ========================================================================
SelfReflector::SelfReflector(Model* model) : model_(model) {}

std::string SelfReflector::reflect(const std::string& input, const std::string& output) {
    if (!model_) return "Reflection: No model available";
    int vocab_size = (int)model_->config.vocab_size;
    std::string prompt = "Reflect on this output for input \"" + input + "\": " + output + ". How could this output be improved?";
    auto ids = simple_encode(prompt, vocab_size);
    auto gen = generate_new_tokens(model_, ids, vocab_size, 50);
    return simple_decode(gen);
}

std::string SelfReflector::refine(const std::string& original, const std::string& reflection) {
    if (!model_) return "[Refined]: No model available";
    int vocab_size = (int)model_->config.vocab_size;
    std::string prompt = "Original: " + original + "\nReflection: " + reflection + "\nRefined:";
    auto ids = simple_encode(prompt, vocab_size);
    auto gen = generate_new_tokens(model_, ids, vocab_size, 50);
    return simple_decode(gen);
}

float SelfReflector::consistency_check(const std::string& output1, const std::string& output2) {
    if (output1.empty() || output2.empty()) return 0.0f;
    std::string o1 = output1, o2 = output2;
    std::transform(o1.begin(), o1.end(), o1.begin(), ::tolower);
    std::transform(o2.begin(), o2.end(), o2.begin(), ::tolower);
    std::vector<std::string> tokens1, tokens2;
    std::istringstream ss1(o1), ss2(o2);
    std::string word;
    while (ss1 >> word) tokens1.push_back(word);
    while (ss2 >> word) tokens2.push_back(word);
    if (tokens1.empty() || tokens2.empty()) return 0.0f;
    std::set<std::string> set1(tokens1.begin(), tokens1.end());
    std::set<std::string> set2(tokens2.begin(), tokens2.end());
    int intersection = 0;
    for (const auto& w : set1) {
        if (set2.find(w) != set2.end()) intersection++;
    }
    int denom = (int)std::max(set1.size(), set2.size());
    float jaccard = denom > 0 ? (float)intersection / (float)denom : 0.0f;
    float edit_sim = 0.0f;
    if (o1.size() < 1000 && o2.size() < 1000) {
        int m = (int)o1.size(), n = (int)o2.size();
        std::vector<int> dp(n + 1, 0);
        for (int j = 0; j <= n; j++) dp[j] = j;
        for (int i = 1; i <= m; i++) {
            int prev = dp[0];
            dp[0] = i;
            for (int j = 1; j <= n; j++) {
                int temp = dp[j];
                if (o1[i - 1] == o2[j - 1]) {
                    dp[j] = prev;
                } else {
                    dp[j] = 1 + std::min({prev, dp[j - 1], dp[j]});
                }
                prev = temp;
            }
        }
        int edit_dist = dp[n];
        float max_len = (float)std::max(m, n);
        edit_sim = max_len > 0 ? 1.0f - (float)edit_dist / max_len : 0.0f;
    }
    return jaccard * 0.4f + edit_sim * 0.6f;
}

float SelfReflector::ece_calibration(const std::vector<float>& confidences,
                                      const std::vector<bool>& correctness, int n_bins) {
    if (confidences.empty() || confidences.size() != correctness.size()) return 1.0f;
    int n = (int)confidences.size();
    float ece = 0.0f;
    for (int b = 0; b < n_bins; b++) {
        float bin_start = (float)b / (float)n_bins;
        float bin_end = (float)(b + 1) / (float)n_bins;
        float bin_acc = 0.0f;
        float bin_conf = 0.0f;
        int bin_count = 0;
        for (int i = 0; i < n; i++) {
            if (confidences[i] >= bin_start && confidences[i] < bin_end) {
                bin_acc += correctness[i] ? 1.0f : 0.0f;
                bin_conf += confidences[i];
                bin_count++;
            }
        }
        if (bin_count > 0) {
            float acc = bin_acc / (float)bin_count;
            float conf = bin_conf / (float)bin_count;
            float weight = (float)bin_count / (float)n;
            ece += weight * std::abs(acc - conf);
        }
    }
    return ece;
}

std::vector<std::string> SelfReflector::identify_errors(const std::string& input,
                                                         const std::string& output,
                                                         const std::string& expected) {
    std::vector<std::string> errors;
    if (output.empty()) {
        errors.push_back("empty output: model produced no response");
        return errors;
    }
    if (expected.empty()) return errors;
    std::string out_lower = output;
    std::string exp_lower = expected;
    std::transform(out_lower.begin(), out_lower.end(), out_lower.begin(), ::tolower);
    std::transform(exp_lower.begin(), exp_lower.end(), exp_lower.begin(), ::tolower);
    for (char c : {'?', '.', '!', ',', ';', ':'}) {
        std::replace(out_lower.begin(), out_lower.end(), c, ' ');
        std::replace(exp_lower.begin(), exp_lower.end(), c, ' ');
    }
    std::istringstream exp_ss(exp_lower);
    std::set<std::string> expected_words;
    std::string word;
    while (exp_ss >> word) {
        if (word.size() > 2) expected_words.insert(word);
    }
    int missing_count = 0;
    int extra_count = 0;
    std::istringstream out_ss(out_lower);
    std::set<std::string> output_words;
    while (out_ss >> word) {
        if (word.size() > 2) output_words.insert(word);
    }
    for (const auto& ew : expected_words) {
        if (output_words.find(ew) == output_words.end()) {
            missing_count++;
            if (missing_count <= 3) {
                errors.push_back("missing keyword: \"" + ew + "\" expected but not found");
            }
        }
    }
    for (const auto& ow : output_words) {
        if (expected_words.find(ow) == expected_words.end()) {
            extra_count++;
            if (extra_count <= 2) {
                errors.push_back("unexpected token: \"" + ow + "\" in output but not expected");
            }
        }
    }
    if (missing_count > 0) {
        errors.push_back("total missing: " + std::to_string(missing_count) + " keywords absent");
    }
    if (extra_count > 3) {
        errors.push_back("excessive extra tokens: " + std::to_string(extra_count) + " unexpected words");
    }
    float consistency = consistency_check(output, expected);
    if (consistency < 0.5f) {
        errors.push_back("low consistency with expected output: jaccard=" + std::to_string(consistency));
    }
    if (output.size() > expected.size() * 3 && expected.size() > 10) {
        errors.push_back("output excessively verbose: " + std::to_string(output.size()) +
                         " chars vs " + std::to_string(expected.size()) + " expected");
    }
    if (output.size() < expected.size() / 3 && expected.size() > 10) {
        errors.push_back("output too brief: " + std::to_string(output.size()) +
                         " chars vs " + std::to_string(expected.size()) + " expected");
    }
    return errors;
}

// ========================================================================
// Subsystem 3: MetaCognition — knowledge boundaries, uncertainty, refusal, reasoning quality, self-model update
// ========================================================================
MetaCognition::MetaCognition(Model* model) : model_(model) {}

float MetaCognition::knowledge_boundary_score(const std::string& query) {
    if (knowledge_map_.empty()) {
        knowledge_map_["what"] = 0.9f;
        knowledge_map_["how"] = 0.6f;
        knowledge_map_["why"] = 0.5f;
        knowledge_map_["define"] = 0.8f;
        knowledge_map_["explain"] = 0.7f;
        knowledge_map_["list"] = 0.85f;
        knowledge_map_["compare"] = 0.6f;
        knowledge_map_["solve"] = 0.5f;
        knowledge_map_["calculate"] = 0.4f;
        knowledge_map_["unknown"] = 0.2f;
    }
    std::string q = query;
    std::transform(q.begin(), q.end(), q.begin(), ::tolower);
    std::istringstream ss(q);
    std::string word;
    float total_score = 0;
    int count = 0;
    while (ss >> word) {
        for (auto& [key, val] : knowledge_map_) {
            if (word.find(key) != std::string::npos || key.find(word) != std::string::npos) {
                total_score += val;
                count++;
                break;
            }
        }
    }
    if (!model_) {
        return count > 0 ? total_score / (float)count : 0.3f;
    }
    int vocab_size = (int)model_->config.vocab_size;
    auto ids = simple_encode("answer knowledge about: " + query, vocab_size);
    auto gen = generate_new_tokens(model_, ids, vocab_size, 20);
    std::string response = simple_decode(gen);
    float response_len_score = std::min(1.0f, (float)response.size() / 50.0f);
    float word_score = count > 0 ? total_score / (float)count : 0.3f;
    return word_score * 0.6f + response_len_score * 0.4f;
}

float MetaCognition::epistemic_uncertainty(const Tensor& logits) {
    int64_t V = logits.dim(logits.rank() - 1);
    int64_t S = logits.numel() / V;
    const float* lp = logits.data<float>();
    float top2_margin_sum = 0;
    int count = 0;
    for (int64_t i = 0; i < S; i++) {
        float top1_val = -INFINITY, top2_val = -INFINITY;
        for (int64_t v = 0; v < V; v++) {
            float val = lp[i * V + v];
            if (val > top1_val) {
                top2_val = top1_val;
                top1_val = val;
            } else if (val > top2_val) {
                top2_val = val;
            }
        }
        top2_margin_sum += top1_val - top2_val;
        count++;
    }
    float avg_margin = count > 0 ? top2_margin_sum / (float)count : 0.0f;
    return 1.0f / (1.0f + std::exp(avg_margin));
}

float MetaCognition::aleatoric_uncertainty(const Tensor& logits) {
    int64_t V = logits.dim(logits.rank() - 1);
    int64_t S = logits.numel() / V;
    const float* lp = logits.data<float>();
    float total_entropy = 0;
    int count = 0;
    for (int64_t i = 0; i < S; i++) {
        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++) max_l = std::max(max_l, lp[i * V + v]);
        float sum = 0;
        for (int64_t v = 0; v < V; v++) sum += std::exp(lp[i * V + v] - max_l);
        float entropy = 0;
        for (int64_t v = 0; v < V; v++) {
            float p = std::exp(lp[i * V + v] - max_l) / (sum + 1e-10f);
            if (p > 1e-10f) entropy -= p * std::log(p + 1e-10f);
        }
        total_entropy += entropy;
        count++;
    }
    float avg_entropy = count > 0 ? total_entropy / (float)count : 0.0f;
    float max_entropy = std::log((float)V + 1e-10f);
    return max_entropy > 0 ? avg_entropy / max_entropy : 0.0f;
}

float MetaCognition::total_uncertainty(const Tensor& logits) {
    float epi = epistemic_uncertainty(logits);
    float alea = aleatoric_uncertainty(logits);
    return std::min(1.0f, epi + alea - epi * alea);
}

bool MetaCognition::should_refuse(const std::string& query, float uncertainty_threshold) {
    if (!model_) return true;
    int vocab_size = (int)model_->config.vocab_size;
    auto ids = simple_encode(query, vocab_size);
    int64_t seq_len = (int64_t)ids.size();
    if (seq_len == 0) return true;
    Tensor input_tensor({1, seq_len});
    Tensor pos_tensor({1, seq_len});
    float* idp = input_tensor.data<float>();
    float* psp = pos_tensor.data<float>();
    for (int64_t i = 0; i < seq_len; i++) {
        idp[i] = (float)ids[i % (int)ids.size()];
        psp[i] = (float)i;
    }
    Tensor logits = model_->forward(input_tensor, pos_tensor, nullptr);
    float unc = total_uncertainty(logits);
    float kb = knowledge_boundary_score(query);
    if (unc > uncertainty_threshold || kb < 0.3f) return true;
    std::string q = query;
    std::transform(q.begin(), q.end(), q.begin(), ::tolower);
    std::vector<std::string> refuse_patterns = {
        "how to make a bomb", "instructions for illegal", "how to hack",
        "bypass security", "how to commit", "illegal drug synthesis",
        "how to build a weapon", "terrorist attack", "child exploitation"
    };
    for (auto& p : refuse_patterns) {
        if (q.find(p) != std::string::npos) return true;
    }
    return false;
}

float MetaCognition::reasoning_quality(const std::vector<std::string>& chain) {
    if (chain.empty()) return 0.0f;
    float total_score = 0;
    int n = (int)chain.size();
    for (int i = 0; i < n; i++) {
        std::string step = chain[i];
        std::transform(step.begin(), step.end(), step.begin(), ::tolower);
        float step_score = 0.3f;
        if (step.find("step") != std::string::npos) step_score += 0.1f;
        if (step.find("because") != std::string::npos || step.find("since") != std::string::npos) step_score += 0.15f;
        if (step.find("therefore") != std::string::npos || step.find("thus") != std::string::npos) step_score += 0.15f;
        if (step.find("if") != std::string::npos || step.find("then") != std::string::npos) step_score += 0.1f;
        if (step.size() > 20) step_score += 0.1f;
        if (step.size() > 100) step_score -= 0.05f;
        int digit_count = 0;
        for (char c : step) if (c >= '0' && c <= '9') digit_count++;
        if (digit_count > 0) step_score += 0.1f;
        total_score += std::min(1.0f, step_score);
    }
    float length_bonus = std::min(1.0f, (float)n / 5.0f) * 0.2f;
    return std::min(1.0f, total_score / (float)n + length_bonus);
}

void MetaCognition::self_model_update(const std::string& query, float confidence, bool was_correct) {
    float prediction_error = std::abs(confidence - (was_correct ? 1.0f : 0.0f));
    float alpha = 0.1f / (1.0f + 0.01f * self_model_updates_);
    float update = was_correct ? alpha * prediction_error : -alpha * (1.0f - prediction_error);
    self_model_accuracy_ = self_model_accuracy_ * (1.0f - alpha) + (was_correct ? 1.0f : 0.0f) * alpha;
    self_model_updates_++;
    std::string q = query;
    std::transform(q.begin(), q.end(), q.begin(), ::tolower);
    std::istringstream ss(q);
    std::string word;
    while (ss >> word) {
        auto it = knowledge_map_.find(word);
        if (it != knowledge_map_.end()) {
            it->second = std::max(0.1f, std::min(0.99f,
                it->second + (was_correct ? 0.05f : -0.1f) / (1.0f + 0.1f * self_model_updates_)));
        } else if (knowledge_map_.size() < 500) {
            knowledge_map_[word] = was_correct ? 0.6f : 0.3f;
        }
    }
}

// ========================================================================
// Subsystem 4: RSI — evaluate->weaknesses->candidates->evaluate->accept->repeat
// ========================================================================
RecursiveSelfImprover::RecursiveSelfImprover(Model* model, Trainer* trainer)
    : model_(model), trainer_(trainer) {}

void RecursiveSelfImprover::improvement_cycle(int iterations) {
    if (!model_) return;
    // Save the FULL training config before the loop: continual-loop mutation must
    // never leak out of a single improvement_cycle call. Catastrophic forgetting
    // happens when rope_theta / norm_eps / learning-rate tweaks compound across
    // calls, permanently degrading the model. All exit paths restore the snapshot.
    const int64_t orig_hidden = model_->config.hidden_size;
    const int64_t orig_layers = model_->config.num_layers;
    const float orig_rope_theta = model_->config.rope_theta;
    const float orig_norm_eps = model_->config.norm_eps;
    const float orig_learning_rate = learning_rate_;
    const float orig_dropout_rate = dropout_rate_;
    auto restore_config = [&]() {
        model_->config.hidden_size = orig_hidden;
        model_->config.num_layers = orig_layers;
        model_->config.rope_theta = orig_rope_theta;
        model_->config.norm_eps = orig_norm_eps;
        learning_rate_ = orig_learning_rate;
        dropout_rate_ = orig_dropout_rate;
    };
    int no_improvement_count = 0;
    float best_perplexity = 1e10f;

    for (int i = 0; i < iterations && i < AlignmentSystem::max_loop_iterations; i++) {
        if (no_improvement_count >= 10) {
            restore_config();
            break;
        }

        int vocab_size = (int)model_->config.vocab_size;
        std::string test_input = "Self-evaluation test input iteration " + std::to_string(i);
        auto test_ids = simple_encode(test_input, vocab_size);
        int64_t len = std::max((int64_t)1, (int64_t)test_ids.size());
        Tensor input_tensor({1, len});
        Tensor pos_tensor({1, len});
        float* idp = input_tensor.data<float>();
        float* psp = pos_tensor.data<float>();
        for (int64_t j = 0; j < len; j++) {
            idp[j] = (float)test_ids[j % (int)test_ids.size()];
            psp[j] = (float)j;
        }
        Tensor logits = model_->forward(input_tensor, pos_tensor, nullptr);
        int64_t V = logits.dim(logits.rank() - 1);
        float loss = 0;
        int count = 0;
        for (int64_t j = 1; j < len; j++) {
            int target = test_ids[j % (int)test_ids.size()];
            const float* lp = logits.data<float>() + (j - 1) * V;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < V; v++) max_l = std::max(max_l, lp[v]);
            float sum = 0;
            for (int64_t v = 0; v < V; v++) sum += std::exp(lp[v] - max_l);
            float prob = std::exp(lp[target % (int)V] - max_l) / (sum + 1e-10f);
            loss -= std::log(std::max(prob, 1e-10f));
            count++;
        }
        float perplexity = std::exp(loss / std::max(1, count));

        std::string eval_data = "perplexity=" + std::to_string(perplexity) +
                                "|iteration=" + std::to_string(i) +
                                "|hidden_size=" + std::to_string(model_->config.hidden_size);

        std::vector<std::string> weaknesses = evaluate_weaknesses(eval_data);
        std::vector<std::string> candidates = generate_candidates(weaknesses);

        for (auto& candidate : candidates) {
            float score = evaluate_candidate(candidate);
            if (score > 0.5f) {
                accept_candidate(candidate, score);
                break;
            }
        }

        if (perplexity > 20.0f) {
            apply_perturbation(0.02f);
            adjust_lr("high_perplexity");
        } else if (perplexity > 12.0f) {
            adjust_format(eval_data);
            apply_dropout(0.05f);
        }

        if (perplexity < best_perplexity) {
            best_perplexity = perplexity;
            no_improvement_count = 0;
        } else {
            no_improvement_count++;
            if (no_improvement_count >= 5) {
                apply_pruning(0.005f);
            }
        }

        std::string analysis = "iteration=" + std::to_string(i) +
            "|perplexity=" + std::to_string(perplexity) +
            "|vocab_size=" + std::to_string(vocab_size) +
            "|hidden_size=" + std::to_string(model_->config.hidden_size) +
            "|num_layers=" + std::to_string(model_->config.num_layers);

        if (!self_modify(analysis)) {
            restore_config();
            break;
        }
    }
    restore_config();
}

std::vector<std::string> RecursiveSelfImprover::evaluate_weaknesses(const std::string& eval_data) {
    std::vector<std::string> weaknesses;
    float perplexity = -1.0f;
    auto ppos = eval_data.find("perplexity=");
    if (ppos != std::string::npos) {
        ppos += 11;
        auto pend = eval_data.find('|', ppos);
        try {
            perplexity = std::stof(eval_data.substr(ppos, pend - ppos));
        } catch (const std::exception& e) {
            (void)e;
            perplexity = 20.0f;
        } catch (...) {
            perplexity = 20.0f;
        }
    }
    if (perplexity > 20.0f) {
        weaknesses.push_back("high_perplexity: model struggling with basic predictions");
        weaknesses.push_back("poor_calibration: confidence does not match accuracy");
    } else if (perplexity > 10.0f) {
        weaknesses.push_back("moderate_perplexity: need better representation learning");
        weaknesses.push_back("insufficient_context_utilization");
    } else if (perplexity > 5.0f) {
        weaknesses.push_back("mild_perplexity: minor improvements in attention patterns");
    } else {
        weaknesses.push_back("low_perplexity: maintain current performance, explore capacity");
    }
    int64_t hidden = model_ ? model_->config.hidden_size : 4096;
    int64_t layers = model_ ? model_->config.num_layers : 32;
    if (hidden < 2048) weaknesses.push_back("limited_model_capacity: hidden_size too small");
    if (layers < 12) weaknesses.push_back("shallow_architecture: too few layers for complex reasoning");
    weaknesses.push_back("generalization_gap: evaluate on held-out data");
    return weaknesses;
}

std::vector<std::string> RecursiveSelfImprover::generate_candidates(const std::vector<std::string>& weaknesses) {
    std::vector<std::string> candidates;
    for (auto& w : weaknesses) {
        if (w.find("high_perplexity") != std::string::npos) {
            candidates.push_back("increase_learning_rate");
            candidates.push_back("apply_weight_perturbation");
            candidates.push_back("adjust_rope_theta");
        } else if (w.find("moderate_perplexity") != std::string::npos) {
            candidates.push_back("adjust_norm_epsilon");
            candidates.push_back("apply_structured_dropout");
            candidates.push_back("learning_rate_warmup");
        } else if (w.find("limited_model_capacity") != std::string::npos) {
            candidates.push_back("increase_hidden_size");
            candidates.push_back("increase_num_layers");
        } else if (w.find("shallow_architecture") != std::string::npos) {
            candidates.push_back("add_transformer_layer");
            candidates.push_back("increase_ffn_size");
        } else if (w.find("generalization") != std::string::npos) {
            candidates.push_back("apply_weight_decay");
            candidates.push_back("add_dropout_regularization");
        } else if (w.find("calibration") != std::string::npos) {
            candidates.push_back("temperature_scaling");
            candidates.push_back("confidence_penalization");
        } else {
            candidates.push_back("apply_small_perturbation");
            candidates.push_back("adjust_learning_rate");
        }
    }
    if (candidates.empty()) {
        candidates.push_back("no_change");
    }
    return candidates;
}

float RecursiveSelfImprover::evaluate_candidate(const std::string& candidate) {
    if (!model_) return 0.0f;
    int vocab_size = (int)model_->config.vocab_size;
    std::string eval_prompt = "Evaluate candidate improvement: " + candidate + " Score (0-10):";
    auto ids = simple_encode(eval_prompt, vocab_size);
    auto gen = generate_new_tokens(model_, ids, vocab_size, 5);
    std::string response = simple_decode(gen);
    float score = 0.5f;
    if (!response.empty() && response[0] >= '0' && response[0] <= '9') {
        score = (float)(response[0] - '0') / 10.0f;
    }
    if (candidate == "increase_hidden_size" || candidate == "increase_num_layers") {
        score = 0.3f;
    }
    if (candidate == "no_change") {
        score = 0.1f;
    }
    if (candidate.find("perturb") != std::string::npos) {
        score = 0.4f;
    }
    return std::max(0.0f, std::min(1.0f, score));
}

bool RecursiveSelfImprover::accept_candidate(const std::string& candidate, float score) {
    if (score < 0.4f) return false;
    if (candidate == "increase_learning_rate") {
        adjust_lr("explicit_increase");
    } else if (candidate == "apply_weight_perturbation") {
        apply_perturbation(0.01f * score);
    } else if (candidate == "apply_structured_dropout") {
        apply_dropout(0.1f * score);
    } else if (candidate == "temperature_scaling") {
        if (model_) model_->config.rope_theta *= (1.0f + 0.1f * score);
    } else if (candidate == "confidence_penalization") {
        learning_rate_ *= (1.0f - 0.05f * score);
    } else if (candidate == "apply_small_perturbation") {
        apply_perturbation(0.005f);
    } else if (candidate == "adjust_learning_rate") {
        adjust_lr("adaptive");
    } else if (candidate == "apply_weight_decay") {
        if (model_) model_->config.norm_eps *= (1.0f + 0.1f);
    } else if (candidate == "adjust_norm_epsilon") {
        if (model_) model_->config.norm_eps = std::max(1e-7f, model_->config.norm_eps * 0.8f);
    } else if (candidate == "adjust_rope_theta") {
        if (model_) model_->config.rope_theta = std::min(100000.0f, model_->config.rope_theta * 1.2f);
    }
    return true;
}

bool RecursiveSelfImprover::self_modify(const std::string& analysis) {
    if (!model_) return false;
    auto extract_val = [&](const std::string& key) -> float {
        auto pos = analysis.find(key + "=");
        if (pos == std::string::npos) return -1.0f;
        pos += key.size() + 1;
        auto end = analysis.find('|', pos);
        try { return std::stof(analysis.substr(pos, end - pos)); }
        catch (...) { return -1.0f; }
    };
    float perplexity = extract_val("perplexity");
    if (perplexity < 0) return false;
    bool modified = false;
    if (perplexity > 15.0f) {
        model_->config.rope_theta = std::min(100000.0f, model_->config.rope_theta * 1.5f);
        modified = true;
    } else if (perplexity > 8.0f) {
        model_->config.norm_eps = std::max(1e-7f, model_->config.norm_eps * 0.5f);
        modified = true;
    } else if (perplexity < 3.0f) {
        model_->config.norm_eps = std::min(1e-3f, model_->config.norm_eps * 2.0f);
        modified = true;
    }
    return modified;
}

void RecursiveSelfImprover::adjust_lr(const std::string& weakness) {
    if (weakness == "high_perplexity" || weakness == "explicit_increase") {
        learning_rate_ = std::min(1e-3f, learning_rate_ * 1.5f);
    } else if (weakness == "adaptive") {
        learning_rate_ = learning_rate_ * (0.8f + 0.4f * (float)rand() / (float)RAND_MAX);
    } else {
        learning_rate_ = std::max(1e-6f, learning_rate_ * 0.9f);
    }
    if (model_ && model_->config.hidden_size > 0) {
        model_->config.norm_eps = std::max(1e-7f, std::min(1e-3f,
            learning_rate_ / (1e-4f) * 1e-5f));
    }
}

void RecursiveSelfImprover::apply_perturbation(float magnitude) {
    if (!model_) return;
    int vocab_size = (int)model_->config.vocab_size;
    if (vocab_size <= 0) return;
    int64_t n = model_->config.hidden_size;
    if (n <= 0) return;
    std::mt19937 rng((unsigned)std::time(nullptr));
    std::normal_distribution<float> noise(0.0f, magnitude);
    Tensor dummy_input({1, n});
    Tensor dummy_pos({1, n});
    float* idp = dummy_input.data<float>();
    float* psp = dummy_pos.data<float>();
    for (int64_t i = 0; i < n; i++) {
        idp[i] = (float)(rng() % std::max(1, vocab_size));
        psp[i] = (float)i;
    }
    Tensor dummy_output = model_->forward(dummy_input, dummy_pos, nullptr);
    float* od = dummy_output.data<float>();
    int64_t total = dummy_output.numel();
    for (int64_t i = 0; i < total; i++) {
        od[i] += noise(rng) * 0.01f;
    }
}

void RecursiveSelfImprover::adjust_format(const std::string& analysis) {
    float perplexity = -1.0f;
    auto ppos = analysis.find("perplexity=");
    if (ppos != std::string::npos) {
        ppos += 11;
        try { perplexity = std::stof(analysis.substr(ppos, analysis.find('|', ppos) - ppos)); }
        catch (...) {}
    }
    if (perplexity > 0 && model_) {
        float scale = std::min(2.0f, perplexity / 5.0f);
        model_->config.rope_theta = std::min(200000.0f, model_->config.rope_theta * (1.0f + 0.05f * scale));
    }
}

void RecursiveSelfImprover::apply_dropout(float rate) {
    dropout_rate_ = std::min(0.5f, rate);
    if (model_) {
        model_->config.norm_eps = std::max(1e-7f, model_->config.norm_eps * (1.0f - dropout_rate_ * 0.5f));
    }
}

void RecursiveSelfImprover::apply_pruning(float threshold) {
    if (!model_) return;
    int64_t n = model_->config.hidden_size;
    if (n <= 0) return;
    std::mt19937 rng(42);
    int vocab_size = std::max(1, (int)model_->config.vocab_size);
    Tensor dummy_input({1, n});
    Tensor dummy_pos({1, n});
    float* idp = dummy_input.data<float>();
    float* psp = dummy_pos.data<float>();
    for (int64_t i = 0; i < n; i++) {
        idp[i] = (float)(rng() % vocab_size);
        psp[i] = (float)i;
    }
    Tensor output = model_->forward(dummy_input, dummy_pos, nullptr);
    float* od = output.data<float>();
    int64_t total = output.numel();
    float ref_val = 0;
    for (int64_t i = 0; i < std::min(total, (int64_t)100); i++) ref_val += std::abs(od[i]);
    ref_val = ref_val / (float)std::min(total, (int64_t)100) + 1e-10f;
    for (int64_t i = 0; i < total; i++) {
        if (std::abs(od[i]) < ref_val * threshold) {
            od[i] = 0.0f;
        }
    }
}

class TripleLoopVerifier {
public:
    struct VerificationResult {
        std::string output;
        float confidence;
        bool passed_symbolic;
        bool passed_sandbox;
        int consistency_votes;
        int total_paths;
    };
    
    VerificationResult verify(const std::string& prompt, Model* model, Tokenizer* tok, int num_paths = 10, float consistency_threshold = 0.99f) {
        VerificationResult res;
        res.total_paths = num_paths;
        
        std::vector<std::string> paths = generate_paths(prompt, model, tok, num_paths, 0.8f);
        if (paths.empty()) {
            res.output = "";
            res.confidence = 0.0f;
            res.passed_symbolic = false;
            res.passed_sandbox = false;
            res.consistency_votes = 0;
            return res;
        }

        std::string best_output = consistency_vote(paths, consistency_threshold);
        res.output = best_output;
        
        res.passed_symbolic = symbolic_verify(best_output);
        res.passed_sandbox = sandbox_verify(best_output);
        
        int votes = 0;
        for (const auto& p : paths) {
            if (p == best_output) votes++;
        }
        res.consistency_votes = votes;
        res.confidence = (float)votes / (float)num_paths;
        
        return res;
    }
    
private:
    std::vector<std::string> generate_paths(const std::string& prompt, Model* model, Tokenizer* tok, int n, float temperature = 0.8f) {
        std::vector<std::string> paths;
        if (!model) return paths;
        int vocab_size = (int)model->config.vocab_size;
        auto prompt_ids = simple_encode(prompt, vocab_size);
        for (int i = 0; i < n; i++) {
            auto gen_ids = generate_new_tokens(model, prompt_ids, vocab_size, 50);
            paths.push_back(simple_decode(gen_ids));
        }
        return paths;
    }

    bool symbolic_verify(const std::string& output) {
        int brackets = 0, parens = 0, braces = 0;
        for (char c : output) {
            if (c == '[') brackets++; else if (c == ']') brackets--;
            if (c == '(') parens++;   else if (c == ')') parens--;
            if (c == '{') braces++;   else if (c == '}') braces--;
            if (brackets < 0 || parens < 0 || braces < 0) return false;
        }
        if (brackets != 0 || parens != 0 || braces != 0) return false;

        std::string lower = output;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find("not not") != std::string::npos) return false;
        if (lower.find("true and false") != std::string::npos) return false;
        return true;
    }

    bool sandbox_verify(const std::string& output) {
        std::string lower = output;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find("system(") != std::string::npos) return false;
        if (lower.find("exec(") != std::string::npos) return false;
        if (lower.find("rm -rf") != std::string::npos) return false;
        if (lower.find("drop table") != std::string::npos) return false;
        return true;
    }

    std::string consistency_vote(const std::vector<std::string>& outputs, float threshold) {
        if (outputs.empty()) return "";
        std::unordered_map<std::string, int> votes;
        for (const auto& o : outputs) votes[o]++;
        
        std::string best = outputs[0];
        int max_votes = 0;
        for (const auto& [k, v] : votes) {
            if (v > max_votes) {
                max_votes = v;
                best = k;
            }
        }
        return best;
    }
};

class ASTGate {
public:
    struct GateResult { bool passed; std::string reason; };
    GateResult validate(const std::string& output) {
        if (!check_length_bounds(output)) return {false, "Length bounds exceeded"};
        if (!check_balanced_delimiters(output)) return {false, "Unbalanced delimiters"};
        if (!check_no_code_injection(output)) return {false, "Code injection detected"};
        if (!check_citation_format(output)) return {false, "Invalid citation format"};
        return {true, "Passed"};
    }
private:
    bool check_balanced_delimiters(const std::string& s) {
        std::vector<char> stack;
        for (char c : s) {
            if (c == '(' || c == '[' || c == '{') stack.push_back(c);
            else if (c == ')') { if (stack.empty() || stack.back() != '(') return false; stack.pop_back(); }
            else if (c == ']') { if (stack.empty() || stack.back() != '[') return false; stack.pop_back(); }
            else if (c == '}') { if (stack.empty() || stack.back() != '{') return false; stack.pop_back(); }
        }
        return stack.empty();
    }
    bool check_no_code_injection(const std::string& s) {
        std::string lower = s;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find("<script>") != std::string::npos) return false;
        if (lower.find("eval(") != std::string::npos) return false;
        if (lower.find("os.system") != std::string::npos) return false;
        return true;
    }
    bool check_citation_format(const std::string& s) {
        if (s.find("[cite") != std::string::npos && s.find("]") == std::string::npos) return false;
        return true;
    }
    bool check_length_bounds(const std::string& s, size_t min_len = 1, size_t max_len = 100000) {
        return s.size() >= min_len && s.size() <= max_len;
    }
};

class DynamicToolSynthesizer {
public:
    struct Tool { std::string name; std::string code; bool verified; };
    Tool synthesize(const std::string& task_description, Model* model, Tokenizer* tok) {
        Tool t;
        t.name = "generated_tool_" + std::to_string(std::hash<std::string>{}(task_description));
        if (model) {
            int vocab_size = (int)model->config.vocab_size;
            auto ids = simple_encode("Write tool code for: " + task_description, vocab_size);
            auto gen = generate_new_tokens(model, ids, vocab_size, 100);
            t.code = simple_decode(gen);
        } else {
            t.code = "def " + t.name + "(): pass";
        }
        t.verified = verify_tool(t);
        if (t.verified) {
            tool_cache_.push_back(t);
        }
        return t;
    }
    bool verify_tool(const Tool& tool) {
        if (tool.code.empty()) return false;
        if (tool.code.find("syntax error") != std::string::npos) return false;
        if (tool.code.find("import os") != std::string::npos) return false;
        return true;
    }
private:
    std::vector<Tool> tool_cache_;
};

} // namespace agi
} // namespace quant