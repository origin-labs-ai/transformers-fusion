#include "quant/continual_engine.h"
#include "quant/random.h"
#include <algorithm>
#include <numeric>
#include <cassert>

namespace quant {

// ============================================================================
// CompressedReplayBuffer
// ============================================================================

CompressedReplayBuffer::CompressedReplayBuffer(std::size_t capacity,
                                                 float spill_threshold)
    : capacity_(capacity)
    , spill_threshold_(spill_threshold)
    , entries_() {
    entries_.reserve(capacity);
}

void CompressedReplayBuffer::quantize_entry(
        const float* data, std::size_t n,
        std::vector<uint8_t>& indices,
        std::vector<float>& codebook,
        float& scale) {
    if (n == 0) return;

    float mn = data[0], mx = data[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }
    scale = (mx - mn) * 0.5f + 1e-8f;
    float inv_scale = 1.0f / scale;
    float mid = (mx + mn) * 0.5f;

    constexpr int K = 16;
    codebook.resize(K);
    for (int i = 0; i < K; ++i) {
        codebook[i] = mid + scale * (2.0f * static_cast<float>(i) / (K - 1) - 1.0f);
    }

    indices.resize((n + 1) / 2);
    for (std::size_t i = 0; i < n; i += 2) {
        float v0 = data[i];
        int q0 = static_cast<int>((v0 - mid) * inv_scale * (K - 1) * 0.5f + (K - 1) * 0.5f);
        q0 = std::max(0, std::min(K - 1, q0));

        uint8_t packed = static_cast<uint8_t>(q0 & 0x0F);
        if (i + 1 < n) {
            float v1 = data[i + 1];
            int q1 = static_cast<int>((v1 - mid) * inv_scale * (K - 1) * 0.5f + (K - 1) * 0.5f);
            q1 = std::max(0, std::min(K - 1, q1));
            packed |= static_cast<uint8_t>((q1 & 0x0F) << 4);
        }
        indices[i / 2] = packed;
    }
}

void CompressedReplayBuffer::dequantize_entry(
        const std::vector<uint8_t>& indices,
        const std::vector<float>& codebook,
        float scale, std::size_t n,
        std::vector<float>& output) {
    output.resize(n);
    std::size_t idx = 0;
    for (std::size_t i = 0; i < indices.size() && idx < n; ++i) {
        int lo = indices[i] & 0x0F;
        int hi = (indices[i] >> 4) & 0x0F;
        if (lo >= 0 && lo < static_cast<int>(codebook.size())) {
            output[idx++] = codebook[lo];
        }
        if (idx < n && hi >= 0 && hi < static_cast<int>(codebook.size())) {
            output[idx++] = codebook[hi];
        }
    }
}

void CompressedReplayBuffer::insert(
        const float* input, const float* target,
        std::size_t elem_count, std::uint32_t task_id,
        float fisher_importance) {
    CompressedReplayEntry entry;
    entry.task_id = task_id;
    entry.fisher_importance = fisher_importance;
    entry.step_inserted = 0;
    entry.access_count = 0;

    quantize_entry(input, elem_count, entry.quantized_input, entry.codebook, entry.input_scale);
    quantize_entry(target, elem_count, entry.quantized_target, entry.codebook, entry.target_scale);

    if (entries_.size() >= capacity_) {
        std::size_t min_idx = 0;
        float min_imp = entries_[0].fisher_importance;
        for (std::size_t i = 1; i < entries_.size(); ++i) {
            if (entries_[i].fisher_importance < min_imp) {
                min_imp = entries_[i].fisher_importance;
                min_idx = i;
            }
        }
        entries_[min_idx] = std::move(entry);
    } else {
        entries_.push_back(std::move(entry));
    }
}

bool CompressedReplayBuffer::sample_batch(
        std::vector<std::vector<float>>& inputs,
        std::vector<std::vector<float>>& targets,
        std::vector<float>& weights,
        std::size_t batch_size) {
    if (entries_.empty()) return false;

    inputs.resize(batch_size);
    targets.resize(batch_size);
    weights.resize(batch_size);

    float total_weight = 0.0f;
    for (auto& e : entries_) total_weight += e.fisher_importance;

    if (total_weight <= 0.0f) {
        total_weight = static_cast<float>(entries_.size());
        for (auto& e : entries_) e.fisher_importance = 1.0f;
    }

    for (std::size_t b = 0; b < batch_size; ++b) {
        std::uniform_real_distribution<float> dist(0.0f, total_weight);
        float r = dist(rng_);
        float cum = 0.0f;
        std::size_t chosen = 0;
        for (std::size_t i = 0; i < entries_.size(); ++i) {
            cum += entries_[i].fisher_importance;
            if (cum >= r) {
                chosen = i;
                break;
            }
        }
        entries_[chosen].access_count++;

        std::size_t elem_count = entries_[chosen].quantized_input.size() * 2;
        dequantize_entry(entries_[chosen].quantized_input,
                          entries_[chosen].codebook,
                          entries_[chosen].input_scale,
                          elem_count, inputs[b]);
        dequantize_entry(entries_[chosen].quantized_target,
                          entries_[chosen].codebook,
                          entries_[chosen].target_scale,
                          elem_count, targets[b]);
        weights[b] = entries_[chosen].fisher_importance;
    }
    return true;
}

void CompressedReplayBuffer::importance_resample() {
    if (entries_.size() < capacity_ / 2) return;

    std::vector<float> probs(entries_.size());
    float total = 0.0f;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        probs[i] = entries_[i].fisher_importance + 0.01f;
        total += probs[i];
    }
    for (auto& p : probs) p /= total;

    std::vector<CompressedReplayEntry> new_entries;
    new_entries.reserve(entries_.size());

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        float r = dist(rng_);
        float cum = 0.0f;
        for (std::size_t j = 0; j < entries_.size(); ++j) {
            cum += probs[j];
            if (cum >= r) {
                new_entries.push_back(entries_[j]);
                break;
            }
        }
    }
    entries_ = std::move(new_entries);
}

void CompressedReplayBuffer::clear() {
    entries_.clear();
}

void CompressedReplayBuffer::compact() {
    entries_.erase(
        std::remove_if(entries_.begin(), entries_.end(),
            [](const CompressedReplayEntry& e) {
                return e.fisher_importance < 0.01f && e.access_count == 0;
            }),
        entries_.end());
}

float CompressedReplayBuffer::compression_ratio() const {
    if (entries_.empty()) return 1.0f;
    std::size_t raw_bytes = 0;
    std::size_t comp_bytes = 0;
    for (auto& e : entries_) {
        raw_bytes += (e.quantized_input.size() * 2 + e.quantized_target.size() * 2) * sizeof(float);
        comp_bytes += e.quantized_input.size() + e.quantized_target.size() + e.codebook.size() * sizeof(float);
    }
    return (comp_bytes > 0) ? static_cast<float>(raw_bytes) / static_cast<float>(comp_bytes) : 1.0f;
}

std::size_t CompressedReplayBuffer::memory_bytes() const {
    std::size_t total = 0;
    for (auto& e : entries_) {
        total += e.quantized_input.size() + e.quantized_target.size();
        total += e.codebook.size() * sizeof(float);
        total += sizeof(CompressedReplayEntry);
    }
    return total;
}

void CompressedReplayBuffer::set_fisher(std::size_t idx, float importance) {
    if (idx < entries_.size()) {
        entries_[idx].fisher_importance = importance;
    }
}

void CompressedReplayBuffer::update_importance(float decay_rate) {
    for (auto& e : entries_) {
        e.fisher_importance *= decay_rate;
    }
}

// ============================================================================
// ECC State
// ============================================================================

void ECCState::initialize(std::size_t n) {
    num_params = n;
    fisher_diagonal.assign(n, 0.0f);
    anchor_weights.assign(n, 0.0f);
    initialized = true;
}

void ECCState::update_fisher(const float* gradients, std::size_t n, float alpha) {
    if (!initialized) initialize(n);
    std::size_t count = std::min(n, num_params);
    for (std::size_t i = 0; i < count; ++i) {
        fisher_diagonal[i] = (1.0f - alpha) * fisher_diagonal[i] +
                              alpha * gradients[i] * gradients[i];
    }
}

void ECCState::update_anchor(const float* weights, std::size_t n) {
    if (!initialized) initialize(n);
    std::size_t count = std::min(n, num_params);
    for (std::size_t i = 0; i < count; ++i) {
        anchor_weights[i] = weights[i];
    }
}

float ECCState::regularize(const float* current_weights, std::size_t n,
                            float lambda) const {
    if (!initialized) return 0.0f;
    std::size_t count = std::min(n, num_params);
    float reg = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        float diff = current_weights[i] - anchor_weights[i];
        reg += fisher_diagonal[i] * diff * diff;
    }
    return lambda * reg;
}

// ============================================================================
// Forgetting Benchmark
// ============================================================================

void ForgettingBenchmark::register_task(const float* eval_inputs,
                                         const float* eval_targets,
                                         std::size_t elem_count) {
    ForgettingTask task;
    task.eval_inputs.assign(eval_inputs, eval_inputs + elem_count);
    task.eval_targets.assign(eval_targets, eval_targets + elem_count);
    task.elem_count = elem_count;
    tasks_.push_back(std::move(task));
}

void ForgettingBenchmark::record_baseline(std::size_t task_id, float loss) {
    if (task_id < tasks_.size()) {
        tasks_[task_id].baseline_loss = loss;
        tasks_[task_id].evaluated = true;
    }
}

void ForgettingBenchmark::record_current(std::size_t task_id, float loss) {
    if (task_id < tasks_.size()) {
        tasks_[task_id].current_loss = loss;
    }
}

float ForgettingBenchmark::forgetting(std::size_t task_id) const {
    if (task_id >= tasks_.size()) return 0.0f;
    auto& t = tasks_[task_id];
    if (t.baseline_loss <= 0.0f) return 0.0f;
    return (t.current_loss - t.baseline_loss) / t.baseline_loss;
}

float ForgettingBenchmark::backward_transfer() const {
    if (tasks_.size() < 2) return 0.0f;
    float bwt = 0.0f;
    std::size_t count = 0;
    for (std::size_t i = 0; i < tasks_.size(); ++i) {
        if (tasks_[i].baseline_loss > 0.0f) {
            bwt += forgetting(i);
            ++count;
        }
    }
    return (count > 0) ? bwt / static_cast<float>(count) : 0.0f;
}

float ForgettingBenchmark::forward_transfer() const {
    if (tasks_.size() < 2) return 0.0f;
    float fwt = 0.0f;
    std::size_t count = 0;
    for (std::size_t i = 1; i < tasks_.size(); ++i) {
        if (tasks_[i].baseline_loss > 0.0f && tasks_[i].current_loss > 0.0f) {
            fwt += (tasks_[i].baseline_loss - tasks_[i].current_loss) /
                    tasks_[i].baseline_loss;
            ++count;
        }
    }
    return (count > 0) ? fwt / static_cast<float>(count) : 0.0f;
}

ForgettingMetrics ForgettingBenchmark::compute() const {
    ForgettingMetrics m;
    m.num_tasks = tasks_.size();
    if (tasks_.empty()) return m;

    float total_fw = 0.0f;
    float max_fw = 0.0f;
    std::size_t evaluated = 0;
    for (std::size_t i = 0; i < tasks_.size(); ++i) {
        if (tasks_[i].evaluated) {
            float fw = forgetting(i);
            total_fw += fw;
            if (fw > max_fw) max_fw = fw;
            evaluated++;
        }
    }
    m.avg_forgetting = (evaluated > 0) ? total_fw / static_cast<float>(evaluated) : 0.0f;
    m.max_forgetting = max_fw;
    m.backward_transfer = backward_transfer();
    m.forward_transfer = forward_transfer();
    m.stability_score = 1.0f - m.avg_forgetting;
    return m;
}

// ============================================================================
// Continual Engine
// ============================================================================

ContinualEngine::ContinualEngine(const ContinualEngineConfig& cfg)
    : cfg_(cfg)
    , replay_(cfg.replay_capacity) {}

void ContinualEngine::on_step(const float* gradients, std::size_t n,
                                const float* weights, std::size_t w_n,
                                float base_lr, std::uint32_t task_id) {
    current_step_++;

    if (!ecc_.initialized) {
        ecc_.initialize(w_n);
        ecc_.update_anchor(weights, w_n);
    }

    ecc_.update_fisher(gradients, w_n, cfg_.ecc_alpha);

    if (current_step_ % static_cast<std::size_t>(cfg_.forgetting_check_interval) == 0) {
        benchmark_.record_current(task_id, 0.0f);
        ecc_.update_anchor(weights, w_n);
    }
}

void ContinualEngine::on_task_boundary(std::uint32_t new_task_id,
                                         const float* eval_inputs,
                                         const float* eval_targets,
                                         std::size_t eval_count) {
    if (current_task_ != new_task_id) {
        benchmark_.register_task(eval_inputs, eval_targets, eval_count);
        current_task_ = new_task_id;
        ecc_.update_anchor(nullptr, 0);
    }
}

void ContinualEngine::register_eval(std::uint32_t task_id,
                                      const float* eval_inputs,
                                      const float* eval_targets,
                                      std::size_t eval_count) {
    benchmark_.register_task(eval_inputs, eval_targets, eval_count);
}

float ContinualEngine::effective_lr(float base_lr) const {
    ForgettingMetrics fm = benchmark_.compute();
    if (fm.avg_forgetting > cfg_.forgetting_threshold) {
        float ratio = fm.avg_forgetting / cfg_.forgetting_threshold;
        return base_lr * cfg_.stability_lr_factor * (1.0f + ratio);
    }
    return base_lr * cfg_.plasticity_lr_factor;
}

bool ContinualEngine::should_check_forgetting() const {
    return current_step_ % static_cast<std::size_t>(cfg_.forgetting_check_interval) == 0;
}

// ============================================================================
// Anti-Collapse Mechanisms
// ============================================================================

void apply_orthogonal_projection(float* update, const float* weights, std::size_t n) {
    if (n == 0) return;
    double dot_uw = 0.0;
    double dot_ww = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        dot_uw += static_cast<double>(update[i]) * weights[i];
        dot_ww += static_cast<double>(weights[i]) * weights[i];
    }
    if (dot_ww > 1e-12) {
        float scalar = static_cast<float>(dot_uw / dot_ww);
        for (std::size_t i = 0; i < n; ++i) {
            update[i] -= scalar * weights[i];
        }
    }
}

ExperienceReplayBuffer::ExperienceReplayBuffer(std::size_t max_size, std::size_t replay_interval)
    : max_size_(max_size), replay_interval_(replay_interval), head_(0), size_(0) {
    if (max_size_ > 0) {
        buffer_.resize(max_size_);
    }
}

void ExperienceReplayBuffer::add(const std::vector<float>& input, const std::vector<float>& target, int generation) {
    if (max_size_ == 0) return;
    if (generation > 2) return;
    buffer_[head_] = {input, target, generation};
    head_ = (head_ + 1) % max_size_;
    if (size_ < max_size_) size_++;
}

bool ExperienceReplayBuffer::sample_batch(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& targets, std::size_t batch_size) {
    if (size_ == 0) return false;
    inputs.clear();
    targets.clear();
    inputs.reserve(batch_size);
    targets.reserve(batch_size);
    thread_local std::mt19937 replay_rng(
        static_cast<std::mt19937::result_type>(
            make_seed(resolve_base_seed(0), SeedStream::ContinualReplay, 0)));
    for (std::size_t b = 0; b < batch_size; ++b) {
        std::size_t idx = static_cast<std::size_t>(replay_rng() % size_);
        inputs.push_back(buffer_[idx].input);
        targets.push_back(buffer_[idx].target);
    }
    return true;
}

bool ExperienceReplayBuffer::should_replay(std::size_t current_batch) const {
    return (replay_interval_ > 0) && (current_batch % replay_interval_ == 0) && (size_ > 0);
}

DistributionWatchdog::DistributionWatchdog(float threshold) : threshold_(threshold) {}

void DistributionWatchdog::track_layer(int layer_id, const float* weights, std::size_t n) {
    if (n == 0) return;
    double sum = 0.0, sum2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double w = weights[i];
        sum += w;
        sum2 += w * w;
    }
    double mean = sum / n;
    double var = (sum2 / n) - (mean * mean);
    
    double m4 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double diff = weights[i] - mean;
        m4 += diff * diff * diff * diff;
    }
    m4 /= n;
    double kurtosis = 0.0;
    if (var > 1e-8) {
        kurtosis = m4 / (var * var);
    }
    
    if (known_good_.find(layer_id) == known_good_.end()) {
        known_good_[layer_id] = std::vector<float>(weights, weights + n);
        baseline_stats_[layer_id] = { static_cast<float>(mean), static_cast<float>(var), static_cast<float>(kurtosis), true };
    }
    
    auto baseline = baseline_stats_[layer_id];
    bool healthy = true;
    if (baseline.variance > 1e-8) {
        float ratio = static_cast<float>(var) / baseline.variance;
        if (ratio > threshold_ || ratio < (1.0f / threshold_)) healthy = false;
    }
    if (std::abs(static_cast<float>(mean) - baseline.mean) > threshold_) healthy = false;
    
    current_stats_[layer_id] = { static_cast<float>(mean), static_cast<float>(var), static_cast<float>(kurtosis), healthy };
    
    if (healthy) {
        known_good_[layer_id].assign(weights, weights + n);
        baseline_stats_[layer_id] = current_stats_[layer_id];
    }
}

std::map<int, LayerStatus> DistributionWatchdog::check_health() const {
    return current_stats_;
}

void DistributionWatchdog::rollback(int layer_id, float* weights, std::size_t n) {
    auto it = known_good_.find(layer_id);
    if (it != known_good_.end()) {
        const auto& good = it->second;
        std::size_t count = std::min(n, good.size());
        for (std::size_t i = 0; i < count; ++i) {
            weights[i] = good[i];
        }
    }
}

void apply_entropy_floor(float output_entropy, float* weights, const float* importance, std::size_t n, float noise_scale) {
    float entropy_floor = 2.0f; 
    if (output_entropy < entropy_floor && n > 0) {
        std::vector<float> imp_copy(importance, importance + n);
        std::size_t k = n / 10;
        if (k == 0) return;
        std::nth_element(imp_copy.begin(), imp_copy.begin() + k, imp_copy.end());
        float thresh = imp_copy[k];
        
        thread_local std::mt19937 gen(
            static_cast<std::mt19937::result_type>(
                make_seed(resolve_base_seed(0), SeedStream::ContinualEntropy, 0)));
        std::normal_distribution<float> dist(0.0f, noise_scale);
        
        for (std::size_t i = 0; i < n; ++i) {
            if (importance[i] <= thresh) {
                weights[i] += dist(gen);
            }
        }
    }
}

} // namespace quant
