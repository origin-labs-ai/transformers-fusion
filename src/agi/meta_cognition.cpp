// ========================================================================
// meta_cognition.cpp — Meta-Cognition Engine (DIFFUSION T43)
// ========================================================================
#include "quant/meta_cognition.h"
#include "quant/agi_utils.h"
#include "quant/random.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <numeric>
#include <ctime>
#include <iomanip>
#include <cstdio>

namespace quant {
namespace agi {

// ========================================================================
// Constructor
// ========================================================================
MetaCognitionEngine::MetaCognitionEngine(Model* model) : model_(model) {}

// ========================================================================
// Entropy computation from logits
// ========================================================================
float MetaCognitionEngine::compute_entropy(const float* logits, int64_t vocab_size) {
    float max_l = -INFINITY;
    for (int64_t v = 0; v < vocab_size; v++) max_l = std::max(max_l, logits[v]);
    float sum = 0;
    for (int64_t v = 0; v < vocab_size; v++) sum += std::exp(logits[v] - max_l);
    float entropy = 0;
    for (int64_t v = 0; v < vocab_size; v++) {
        float p = std::exp(logits[v] - max_l) / (sum + 1e-10f);
        if (p > 1e-10f) entropy -= p * std::log(p);
    }
    return entropy;
}

float MetaCognitionEngine::cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    int64_t n = std::min((int64_t)a.size(), (int64_t)b.size());
    if (n == 0) return 0.0f;
    float dot = 0, norm_a = 0, norm_b = 0;
    for (int64_t i = 0; i < n; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a * norm_b);
    return denom > 1e-10f ? dot / denom : 0.0f;
}

// ========================================================================
// Reasoning chain analysis
// ========================================================================
ReasoningChain MetaCognitionEngine::analyze_reasoning(const std::string& input, const std::string& output) {
    ReasoningChain chain;
    chain.chain_id = next_chain_id_++;
    chain.goal = input;

    if (!model_) {
        ReasoningStep step;
        step.description = output.empty() ? "no model available" : output;
        step.confidence = 0.0f;
        step.entropy = 1.0f;
        step.consistent = true;
        step.step_index = 0;
        step.strategy_used = ReasoningStrategy::FORWARD;
        chain.steps.push_back(step);
        chain.overall_confidence = 0.0f;
        chain.consistency_score = 1.0f;
        chain_history_.push_back(chain);
        return chain;
    }

    int vocab_size = (int)model_->config.vocab_size;
    std::string prompt = "Analyze the reasoning in this output for the goal: " + input +
                         "\nOutput: " + output +
                         "\nBreak down into steps and rate each step's confidence (0-1):";
    auto ids = simple_encode(prompt, vocab_size);
    auto gen = generate_new_tokens(model_, ids, vocab_size, 100);
    std::string analysis = simple_decode(gen);

    int64_t seq_len = std::max((int64_t)1, (int64_t)output.size());
    seq_len = std::min(seq_len, (int64_t)64);
    Tensor input_tensor({1, seq_len});
    Tensor pos_tensor({1, seq_len});
    float* idp = input_tensor.data<float>();
    float* psp = pos_tensor.data<float>();
    for (int64_t i = 0; i < seq_len; i++) {
        idp[i] = (float)((unsigned char)output[i % (int)output.size()]);
        psp[i] = (float)i;
    }

    Tensor logits = model_->forward(input_tensor, pos_tensor, nullptr);
    int64_t V = logits.dim(logits.rank() - 1);

    std::istringstream stream(analysis);
    std::string line;
    int64_t step_idx = 0;
    float total_conf = 0;

    while (std::getline(stream, line) && step_idx < 50) {
        if (line.empty()) continue;
        ReasoningStep step;
        step.description = line;
        step.step_index = step_idx;

        float entropy = compute_entropy(logits.data<float>() + (step_idx % seq_len) * V, V);
        float max_entropy = std::log((float)V + 1e-10f);
        step.entropy = max_entropy > 0 ? entropy / max_entropy : 0.0f;
        step.confidence = 1.0f - step.entropy;
        step.strategy_used = select_strategy(line, 0.5f);

        if (step_idx > 0 && !chain.steps.empty()) {
            step.consistent = check_step_consistency(chain.steps.back(), step);
            if (!step.consistent) chain.inconsistency_count++;
        }

        chain.steps.push_back(step);
        total_conf += step.confidence;
        step_idx++;
    }

    if (chain.steps.empty()) {
        ReasoningStep fallback;
        fallback.description = "analysis: " + output;
        fallback.confidence = 0.5f;
        fallback.entropy = 0.5f;
        fallback.consistent = true;
        fallback.step_index = 0;
        chain.steps.push_back(fallback);
        chain.overall_confidence = 0.5f;
    } else {
        chain.overall_confidence = total_conf / (float)chain.steps.size();
    }

    chain.consistency_score = calculate_chain_consistency(chain);
    chain_history_.push_back(chain);

    if ((int64_t)chain_history_.size() > max_history) {
        chain_history_.erase(chain_history_.begin());
    }

    return chain;
}

bool MetaCognitionEngine::check_step_consistency(const ReasoningStep& prev, const ReasoningStep& curr) {
    float conf_drop = prev.confidence - curr.confidence;
    if (conf_drop > 0.4f) return false;

    if (curr.entropy > 0.8f && prev.entropy < 0.3f) return false;

    if (curr.strategy_used != prev.strategy_used) {
        if (curr.strategy_used == ReasoningStrategy::BACKWARD &&
            prev.strategy_used == ReasoningStrategy::FORWARD) {
            return true;
        }
        if (std::abs(conf_drop) < 0.2f) return true;
    }

    return true;
}

float MetaCognitionEngine::calculate_chain_consistency(const ReasoningChain& chain) {
    if (chain.steps.size() <= 1) return 1.0f;

    int64_t consistent_pairs = 0;
    int64_t total_pairs = (int64_t)chain.steps.size() - 1;

    for (int64_t i = 1; i < (int64_t)chain.steps.size(); i++) {
        if (chain.steps[i].consistent) consistent_pairs++;
    }

    return total_pairs > 0 ? (float)consistent_pairs / (float)total_pairs : 1.0f;
}

// ========================================================================
// Confidence calibration
// ========================================================================
void MetaCognitionEngine::record_confidence(float predicted, bool correct, const std::string& task) {
    ConfidenceRecord rec;
    rec.predicted_confidence = predicted;
    rec.was_correct = correct;
    rec.task_description = task;
    rec.timestamp = (int64_t)std::time(nullptr);
    confidence_records_.push_back(rec);

    if ((int64_t)confidence_records_.size() > max_confidence_records) {
        confidence_records_.erase(confidence_records_.begin());
    }
}

float MetaCognitionEngine::get_expected_calibration_error() {
    auto bins = get_calibration_bins(10);
    if (bins.empty()) return 0.0f;

    float total_error = 0;
    int64_t total_count = 0;
    for (auto& bin : bins) {
        if (bin.total_count > 0) {
            float avg_conf = bin.avg_confidence;
            float avg_acc = (float)bin.correct_count / (float)bin.total_count;
            total_error += std::abs(avg_conf - avg_acc) * (float)bin.total_count;
            total_count += bin.total_count;
        }
    }
    return total_count > 0 ? total_error / (float)total_count : 0.0f;
}

std::vector<CalibrationBin> MetaCognitionEngine::get_calibration_bins(int num_bins) {
    std::vector<CalibrationBin> bins(num_bins);
    float bin_size = 1.0f / (float)num_bins;

    for (int i = 0; i < num_bins; i++) {
        bins[i].bin_lower = (float)i * bin_size;
        bins[i].bin_upper = (float)(i + 1) * bin_size;
        bins[i].total_count = 0;
        bins[i].correct_count = 0;
        bins[i].avg_confidence = 0;
    }

    for (auto& rec : confidence_records_) {
        int bin_idx = (int)(rec.predicted_confidence / bin_size);
        if (bin_idx >= num_bins) bin_idx = num_bins - 1;
        if (bin_idx < 0) bin_idx = 0;

        bins[bin_idx].total_count++;
        bins[bin_idx].avg_confidence += rec.predicted_confidence;
        if (rec.was_correct) bins[bin_idx].correct_count++;
    }

    for (auto& bin : bins) {
        if (bin.total_count > 0) {
            bin.avg_confidence /= (float)bin.total_count;
        }
    }

    return bins;
}

float MetaCognitionEngine::get_current_confidence() {
    if (confidence_records_.empty()) return 0.5f;

    int64_t recent = std::min((int64_t)100, (int64_t)confidence_records_.size());
    float total = 0;
    for (int64_t i = (int64_t)confidence_records_.size() - recent;
         i < (int64_t)confidence_records_.size(); i++) {
        total += confidence_records_[i].predicted_confidence;
    }
    return total / (float)recent;
}

// ========================================================================
// Reasoning error identification
// ========================================================================
void MetaCognitionEngine::identify_reasoning_errors(ReasoningChain& chain) {
    for (int64_t i = 1; i < (int64_t)chain.steps.size(); i++) {
        auto& prev = chain.steps[i - 1];
        auto& curr = chain.steps[i];

        if (curr.confidence < 0.2f) {
            curr.consistent = false;
            chain.inconsistency_count++;
        }

        if (curr.entropy > 0.7f && prev.entropy < 0.3f) {
            curr.consistent = false;
            chain.inconsistency_count++;
        }

        float conf_diff = prev.confidence - curr.confidence;
        if (conf_diff > 0.5f) {
            curr.consistent = false;
            chain.inconsistency_count++;
        }
    }
    chain.consistency_score = calculate_chain_consistency(chain);
}

int64_t MetaCognitionEngine::find_first_error_step(const ReasoningChain& chain) {
    for (auto& step : chain.steps) {
        if (!step.consistent) return step.step_index;
    }
    return -1;
}

std::string MetaCognitionEngine::diagnose_error(const ReasoningChain& chain, int64_t error_step) {
    if (error_step < 0 || error_step >= (int64_t)chain.steps.size()) {
        return "No error at step " + std::to_string(error_step);
    }

    auto& step = chain.steps[error_step];
    std::ostringstream diagnosis;

    diagnosis << "Error at step " << error_step << ":\n";
    diagnosis << "  Description: " << step.description << "\n";
    diagnosis << "  Confidence: " << step.confidence << "\n";
    diagnosis << "  Entropy: " << step.entropy << "\n";

    if (step.confidence < 0.3f) {
        diagnosis << "  Diagnosis: LOW CONFIDENCE - model is uncertain about this step\n";
    }
    if (step.entropy > 0.7f) {
        diagnosis << "  Diagnosis: HIGH ENTROPY - reasoning is diffuse, not focused\n";
    }
    if (error_step > 0) {
        float conf_drop = chain.steps[error_step - 1].confidence - step.confidence;
        if (conf_drop > 0.4f) {
            diagnosis << "  Diagnosis: SHARP CONFIDENCE DROP from step " << (error_step - 1)
                      << " (" << conf_drop << " drop)\n";
        }
    }

    return diagnosis.str();
}

// ========================================================================
// Goal decomposition
// ========================================================================
GoalDecomposition MetaCognitionEngine::decompose_goal(const std::string& goal, int64_t max_subgoals) {
    GoalDecomposition decomp;
    decomp.original_goal = goal;

    if (!model_) {
        SubGoal sg1;
        sg1.description = "Analyze: " + goal;
        sg1.estimated_difficulty = 0.3f;
        sg1.subgoal_id = next_subgoal_id_++;
        sg1.depends_on = {};
        decomp.subgoals.push_back(sg1);

        SubGoal sg2;
        sg2.description = "Execute: " + goal;
        sg2.estimated_difficulty = 0.6f;
        sg2.subgoal_id = next_subgoal_id_++;
        sg2.depends_on = {sg1.subgoal_id};
        decomp.subgoals.push_back(sg2);

        SubGoal sg3;
        sg3.description = "Verify: " + goal;
        sg3.estimated_difficulty = 0.4f;
        sg3.subgoal_id = next_subgoal_id_++;
        sg3.depends_on = {sg2.subgoal_id};
        decomp.subgoals.push_back(sg3);

        decomp.total_subgoals = 3;
        return decomp;
    }

    int vocab_size = (int)model_->config.vocab_size;
    std::string prompt = "Decompose this goal into subgoals with dependencies:\n" + goal +
                         "\nList each subgoal with its prerequisites (if any):";
    auto ids = simple_encode(prompt, vocab_size);
    auto gen = generate_new_tokens(model_, ids, vocab_size, (int)(max_subgoals * 10));
    std::string output = simple_decode(gen);

    std::istringstream stream(output);
    std::string line;
    std::unordered_map<std::string, int64_t> name_to_id;

    while (std::getline(stream, line) && (int64_t)decomp.subgoals.size() < max_subgoals) {
        if (line.empty()) continue;

        SubGoal sg;
        sg.description = line;
        sg.subgoal_id = next_subgoal_id_++;

        auto dep_pos = line.find("depends on:");
        if (dep_pos == std::string::npos) dep_pos = line.find("requires:");
        if (dep_pos == std::string::npos) dep_pos = line.find("after:");

        if (dep_pos != std::string::npos) {
            sg.description = line.substr(0, dep_pos);
            std::string dep_str = line.substr(dep_pos + 11);
            std::istringstream dep_stream(dep_str);
            std::string dep_word;
            while (dep_stream >> dep_word) {
                for (auto& [name, id] : name_to_id) {
                    if (dep_word.find(name) != std::string::npos) {
                        sg.depends_on.push_back(id);
                    }
                }
            }
        }

        sg.estimated_difficulty = 0.3f + (float)(sg.description.size() % 50) / 100.0f;
        name_to_id[line] = sg.subgoal_id;
        decomp.subgoals.push_back(sg);
    }

    if (decomp.subgoals.empty()) {
        SubGoal sg;
        sg.description = goal;
        sg.subgoal_id = next_subgoal_id_++;
        sg.estimated_difficulty = 0.5f;
        decomp.subgoals.push_back(sg);
    }

    decomp.total_subgoals = (int64_t)decomp.subgoals.size();
    decomp.progress = 0.0f;
    return decomp;
}

bool MetaCognitionEngine::check_subgoal_completion(const GoalDecomposition& decomp) {
    for (auto& sg : decomp.subgoals) {
        if (!sg.completed && !sg.failed) return false;
    }
    return true;
}

std::vector<int64_t> MetaCognitionEngine::get_ready_subgoals(const GoalDecomposition& decomp) {
    std::vector<int64_t> ready;
    for (auto& sg : decomp.subgoals) {
        if (sg.completed || sg.failed) continue;

        bool all_deps_met = true;
        for (int64_t dep_id : sg.depends_on) {
            bool found = false;
            for (auto& other : decomp.subgoals) {
                if (other.subgoal_id == dep_id && other.completed) {
                    found = true;
                    break;
                }
            }
            if (!found) { all_deps_met = false; break; }
        }
        if (all_deps_met) ready.push_back(sg.subgoal_id);
    }
    return ready;
}

// ========================================================================
// Strategy selection
// ========================================================================
ReasoningStrategy MetaCognitionEngine::select_strategy(const std::string& task, float difficulty) {
    std::string task_lower = to_lower(task);

    if (task_lower.find("why") != std::string::npos ||
        task_lower.find("cause") != std::string::npos ||
        task_lower.find("reason") != std::string::npos) {
        return ReasoningStrategy::BACKWARD;
    }

    if (task_lower.find("similar") != std::string::npos ||
        task_lower.find("like") != std::string::npos ||
        task_lower.find("analogy") != std::string::npos) {
        return ReasoningStrategy::ANALOGICAL;
    }

    if (task_lower.find("decompose") != std::string::npos ||
        task_lower.find("break down") != std::string::npos ||
        task_lower.find("steps") != std::string::npos ||
        difficulty > 0.7f) {
        return ReasoningStrategy::DECOMPOSITION;
    }

    if (task_lower.find("what if") != std::string::npos ||
        task_lower.find("suppose") != std::string::npos ||
        task_lower.find("imagine") != std::string::npos) {
        return ReasoningStrategy::HYPOTHETICAL;
    }

    if (difficulty < 0.3f) return ReasoningStrategy::FORWARD;
    if (difficulty > 0.6f) return ReasoningStrategy::DECOMPOSITION;

    return ReasoningStrategy::FORWARD;
}

ReasoningStrategy MetaCognitionEngine::get_best_strategy_for_task_type(const std::string& task_type) {
    std::unordered_map<ReasoningStrategy, int64_t> strategy_wins;

    for (auto& fb : feedback_history_) {
        if (fb.succeeded) {
            ReasoningStrategy strat = select_strategy(fb.strategy_used, 0.5f);
            strategy_wins[strat]++;
        }
    }

    ReasoningStrategy best = ReasoningStrategy::FORWARD;
    int64_t best_count = 0;
    for (auto& [strat, count] : strategy_wins) {
        if (count > best_count) {
            best_count = count;
            best = strat;
        }
    }
    return best;
}

float MetaCognitionEngine::extract_task_type_score(const std::string& task, ReasoningStrategy strategy) {
    std::string task_lower = to_lower(task);

    switch (strategy) {
    case ReasoningStrategy::FORWARD:
        if (task_lower.find("compute") != std::string::npos ||
            task_lower.find("calculate") != std::string::npos) return 0.8f;
        return 0.5f;

    case ReasoningStrategy::BACKWARD:
        if (task_lower.find("why") != std::string::npos ||
            task_lower.find("explain") != std::string::npos) return 0.8f;
        return 0.4f;

    case ReasoningStrategy::ANALOGICAL:
        if (task_lower.find("compare") != std::string::npos ||
            task_lower.find("similar") != std::string::npos) return 0.8f;
        return 0.3f;

    case ReasoningStrategy::DECOMPOSITION:
        if (task_lower.find("complex") != std::string::npos ||
            task_lower.find("plan") != std::string::npos) return 0.8f;
        return 0.5f;

    case ReasoningStrategy::HYPOTHETICAL:
        if (task_lower.find("what if") != std::string::npos ||
            task_lower.find("imagine") != std::string::npos) return 0.8f;
        return 0.3f;
    }
    return 0.5f;
}

std::string MetaCognitionEngine::task_type_classify(const std::string& task) {
    std::string lower = to_lower(task);
    if (lower.find("code") != std::string::npos || lower.find("program") != std::string::npos ||
        lower.find("function") != std::string::npos) return "code";
    if (lower.find("math") != std::string::npos || lower.find("calculate") != std::string::npos ||
        lower.find("equation") != std::string::npos) return "math";
    if (lower.find("explain") != std::string::npos || lower.find("why") != std::string::npos) return "explanation";
    if (lower.find("plan") != std::string::npos || lower.find("strategy") != std::string::npos) return "planning";
    if (lower.find("write") != std::string::npos || lower.find("story") != std::string::npos) return "creative";
    return "general";
}

// ========================================================================
// Metacognitive feedback loop
// ========================================================================
void MetaCognitionEngine::record_feedback(const FeedbackEntry& entry) {
    feedback_history_.push_back(entry);
    if ((int64_t)feedback_history_.size() > max_history) {
        feedback_history_.erase(feedback_history_.begin());
    }
}

std::vector<FeedbackEntry> MetaCognitionEngine::get_recent_feedback(int64_t count) {
    int64_t start = std::max((int64_t)0, (int64_t)feedback_history_.size() - count);
    return std::vector<FeedbackEntry>(feedback_history_.begin() + start, feedback_history_.end());
}

std::string MetaCognitionEngine::generate_feedback_summary() {
    if (feedback_history_.empty()) return "No feedback recorded yet.";

    int64_t total = (int64_t)feedback_history_.size();
    int64_t successes = 0;
    float total_perf = 0;
    std::unordered_map<std::string, int64_t> strategy_counts;
    std::unordered_map<std::string, int64_t> strategy_successes;

    for (auto& fb : feedback_history_) {
        total_perf += fb.performance_score;
        if (fb.succeeded) {
            successes++;
            strategy_successes[fb.strategy_used]++;
        }
        strategy_counts[fb.strategy_used]++;
    }

    std::ostringstream summary;
    summary << "Metacognitive Feedback Summary\n";
    summary << "  Total tasks: " << total << "\n";
    summary << "  Successes: " << successes << " (" << (float)successes / (float)total * 100.0f << "%)\n";
    summary << "  Average performance: " << total_perf / (float)total << "\n";
    summary << "\nStrategy breakdown:\n";

    for (auto& [strat, count] : strategy_counts) {
        int64_t sc = strategy_successes.count(strat) ? strategy_successes[strat] : 0;
        summary << "  " << strat << ": " << sc << "/" << count
                << " (" << (float)sc / (float)count * 100.0f << "%)\n";
    }

    return summary.str();
}

std::vector<std::string> MetaCognitionEngine::extract_improvement_patterns() {
    std::vector<std::string> patterns;

    if (feedback_history_.size() < 10) return patterns;

    int64_t recent = std::min((int64_t)50, (int64_t)feedback_history_.size());
    auto begin_it = feedback_history_.end() - recent;

    std::unordered_map<std::string, int64_t> recent_fails;
    std::unordered_map<std::string, int64_t> recent_wins;

    for (auto it = begin_it; it != feedback_history_.end(); it++) {
        if (it->succeeded) {
            recent_wins[it->strategy_used]++;
        } else {
            recent_fails[it->strategy_used]++;
        }
    }

    for (auto& [strat, fail_count] : recent_fails) {
        int64_t win_count = recent_wins.count(strat) ? recent_wins[strat] : 0;
        if (fail_count > win_count * 2) {
            patterns.push_back("Strategy '" + strat + "' has high failure rate (" +
                               std::to_string(fail_count) + " fails vs " +
                               std::to_string(win_count) + " wins) - consider alternative");
        }
    }

    std::vector<FeedbackEntry> failures;
    for (auto it = begin_it; it != feedback_history_.end(); it++) {
        if (!it->succeeded && !it->failure_reason.empty()) {
            failures.push_back(*it);
        }
    }
    std::unordered_map<std::string, int64_t> reason_counts;
    for (auto& f : failures) reason_counts[f.failure_reason]++;

    for (auto& [reason, count] : reason_counts) {
        if (count >= 3) {
            patterns.push_back("Recurring failure: '" + reason + "' (occurred " +
                               std::to_string(count) + " times)");
        }
    }

    return patterns;
}

// ========================================================================
// Self-assessment
// ========================================================================
SelfAssessment MetaCognitionEngine::assess_self(int64_t lookback_count) {
    SelfAssessment assessment;

    int64_t start = std::max((int64_t)0, (int64_t)feedback_history_.size() - lookback_count);
    int64_t count = (int64_t)feedback_history_.size() - start;

    if (count <= 0) return assessment;

    float total_perf = 0;
    int64_t total_success = 0;
    int64_t total_duration = 0;

    for (int64_t i = start; i < (int64_t)feedback_history_.size(); i++) {
        total_perf += feedback_history_[i].performance_score;
        if (feedback_history_[i].succeeded) total_success++;
        total_duration += feedback_history_[i].duration_ms;
    }

    assessment.tasks_evaluated = count;
    assessment.accuracy_score = (float)total_success / (float)count;
    assessment.speed_score = count > 0 ? 1.0f / (1.0f + (float)total_duration / (float)count / 1000.0f) : 0.0f;

    int64_t mem_size = feedback_history_.size();
    float mem_ratio = (float)mem_size / (float)max_history;
    assessment.resource_efficiency = 1.0f - mem_ratio;

    assessment.avg_confidence = get_current_confidence();
    assessment.avg_calibration_error = get_expected_calibration_error();

    assessment.overall_score = assessment.accuracy_score * 0.4f +
                               assessment.speed_score * 0.2f +
                               assessment.resource_efficiency * 0.15f +
                               (1.0f - assessment.avg_calibration_error) * 0.25f;

    return assessment;
}

MetacognitiveInsight MetaCognitionEngine::generate_insight(const std::string& task, const std::string& result) {
    MetacognitiveInsight insight;
    insight.timestamp = (int64_t)std::time(nullptr);
    insight.source_task = task;

    SelfAssessment assessment = assess_self(50);

    if (assessment.avg_calibration_error > 0.2f) {
        insight.insight_text = "Confidence calibration is off by " +
                               std::to_string(assessment.avg_calibration_error) +
                               " — tend to " + (assessment.avg_confidence > assessment.accuracy_score ?
                               "overconfident" : "underconfident");
        insight.relevance_score = assessment.avg_calibration_error;
    } else if (assessment.accuracy_score < 0.5f) {
        insight.insight_text = "Accuracy is low (" + std::to_string(assessment.accuracy_score) +
                               ") — consider more deliberate reasoning strategies";
        insight.relevance_score = 1.0f - assessment.accuracy_score;
    } else if (assessment.accuracy_score > 0.8f) {
        insight.insight_text = "Performance is strong (" + std::to_string(assessment.accuracy_score) +
                               ") — can take on more complex tasks";
        insight.relevance_score = assessment.accuracy_score * 0.5f;
    } else {
        std::string task_type = task_type_classify(task);
        insight.insight_text = "Task type '" + task_type + "' — current overall score: " +
                               std::to_string(assessment.overall_score);
        insight.relevance_score = 0.5f;
    }

    insights_.push_back(insight);
    if ((int64_t)insights_.size() > max_history) {
        insights_.erase(insights_.begin());
    }

    return insight;
}

// ========================================================================
// Accessors
// ========================================================================
std::vector<ReasoningChain> MetaCognitionEngine::get_history() {
    return chain_history_;
}

std::vector<ConfidenceRecord> MetaCognitionEngine::get_confidence_history() {
    return confidence_records_;
}

void MetaCognitionEngine::reset() {
    chain_history_.clear();
    confidence_records_.clear();
    feedback_history_.clear();
    insights_.clear();
    next_chain_id_ = 0;
    next_subgoal_id_ = 0;
}

class MCOS {
public:
    void adjust_weights_dynamic(Model* model, const std::string& task_type) {
        if (!model) return;

        auto it = profiles_.find(task_type);
        if (it != profiles_.end()) {
            const auto& weight_deltas = it->second;
            // Dynamically adjust model hyper-parameters and scaling profiles
            if (!weight_deltas.empty()) {
                model->config.norm_eps = std::max(1e-7f, std::min(1e-3f, model->config.norm_eps * (1.0f + weight_deltas[0] * 0.01f)));
            }
            if (weight_deltas.size() > 1) {
                model->config.rope_theta = std::max(100.0f, std::min(500000.0f, model->config.rope_theta * (1.0f + weight_deltas[1] * 0.01f)));
            }
        } else {
            // Register default task profile dynamically
            if (task_type == "code") {
                profiles_[task_type] = {0.1f, 0.5f, 0.2f};
                model->config.norm_eps = 1e-6f;
            } else if (task_type == "math") {
                profiles_[task_type] = {0.05f, 1.0f, 0.4f};
                model->config.rope_theta = 20000.0f;
            } else {
                profiles_[task_type] = {0.0f, 0.0f, 0.0f};
            }
        }
    }

    void register_task_profile(const std::string& name, const std::vector<float>& weight_deltas) {
        profiles_[name] = weight_deltas;
    }

    std::string detect_task_type(const std::string& prompt) {
        std::string lower = prompt;
        for (char& c : lower) c = std::tolower(c);
        if (lower.find("code") != std::string::npos || lower.find("function") != std::string::npos) return "code";
        if (lower.find("math") != std::string::npos || lower.find("solve") != std::string::npos) return "math";
        return "general";
    }

private:
    std::map<std::string, std::vector<float>> profiles_;
};

class MetaGRPO {
public:
    struct Strategy { std::string name; float fitness; std::vector<float> params; };

    Strategy evolve(const std::vector<Strategy>& population, int generations = 10) {
        std::vector<Strategy> pop = population;
        for (int g = 0; g < generations; ++g) {
            for (auto& s : pop) s.fitness = evaluate(s);
            std::sort(pop.begin(), pop.end(), [](const Strategy& a, const Strategy& b) {
                return a.fitness > b.fitness;
            });
            std::vector<Strategy> next_gen;
            for (size_t i = 0; i < pop.size() / 2; ++i) {
                next_gen.push_back(pop[i]);
                Strategy child = crossover(pop[i], pop[(i + 1) % (pop.size() / 2)]);
                mutate(child);
                next_gen.push_back(child);
            }
            pop = next_gen;
        }
        return pop.empty() ? Strategy{"", 0.0f, {}} : pop[0];
    }

    std::vector<Strategy> initialize_population(int size = 20) {
        std::vector<Strategy> pop(size);
        for (int i = 0; i < size; ++i) {
            pop[i].name = "Strat_" + std::to_string(i);
            pop[i].fitness = 0.0f;
            pop[i].params = {1.0f + (float)i * 0.05f, 0.8f + (float)i * 0.02f, 0.1f * (float)(i % 5)};
        }
        return pop;
    }

private:
    Strategy crossover(const Strategy& a, const Strategy& b) {
        Strategy c = a;
        c.name = a.name + "_" + b.name;
        for (size_t i = 0; i < c.params.size() && i < b.params.size(); ++i) {
            if (i % 2 == 1) c.params[i] = b.params[i];
        }
        return c;
    }

    void mutate(Strategy& s, float rate = 0.1f) {
        for (float& p : s.params) {
            p += rate * ((float)rand() / RAND_MAX - 0.5f);
        }
    }

    float evaluate(const Strategy& s) {
        // Multi-objective fitness function evaluating parameter variance, scaling balance, and stability
        if (s.params.empty()) return 0.0f;
        float mean = 0.0f;
        for (float p : s.params) mean += p;
        mean /= (float)s.params.size();

        float variance = 0.0f;
        for (float p : s.params) variance += (p - mean) * (p - mean);
        variance /= (float)s.params.size();

        // Higher fitness for balanced parameter scaling with controlled variance
        float stability_score = 1.0f / (1.0f + variance);
        return mean * 0.6f + stability_score * 0.4f;
    }
};

} // namespace agi
} // namespace quant
