#include "quant/moe_enhance.h"
#include "quant/math.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>

namespace quant {

// ============================================================================
// F1: Expert capacity tuning — importance-weighted adaptive capacity
// ============================================================================
ExpertCapacity ExpertCapacity::adaptive(int64_t num_experts, int64_t tokens,
                                         const float* importance) {
    QUANT_CHECK(num_experts > 0, "ExpertCapacity: num_experts must be > 0");
    QUANT_CHECK(tokens > 0, "ExpertCapacity: tokens must be > 0");
    QUANT_CHECK(importance != nullptr, "ExpertCapacity: importance is null");

    ExpertCapacity ec;
    ec.capacities.resize((size_t)num_experts);

    float total_imp = 0.0f;
    for (int64_t i = 0; i < num_experts; i++)
        total_imp += std::max(importance[i], 0.0f);

    int64_t base_capacity = (tokens + num_experts - 1) / num_experts;

    if (total_imp <= 0) {
        // Uniform allocation if no importance signal
        for (int64_t i = 0; i < num_experts; i++)
            ec.capacities[(size_t)i] = base_capacity;
        return ec;
    }

    // Allocate capacity proportional to importance, with minimum guarantee
    float inv_total = 1.0f / total_imp;
    int64_t remaining = tokens;

    for (int64_t i = 0; i < num_experts; i++) {
        float ratio = std::max(importance[i], 0.0f) * inv_total;
        int64_t cap = std::max((int64_t)(ratio * (float)tokens * 1.25f),
                                (int64_t)std::ceil((float)tokens / (float)num_experts * 0.5f));
        ec.capacities[(size_t)i] = std::min(cap, tokens);
        remaining -= ec.capacities[(size_t)i];
    }

    // Distribute remaining capacity
    if (remaining > 0) {
        int64_t extra = remaining / num_experts;
        for (int64_t i = 0; i < num_experts; i++)
            ec.capacities[(size_t)i] += extra;
    }

    return ec;
}

// ============================================================================
// F2: DenseToMoEPruner — convert dense FFN to MoE by duplicating with noise
// ============================================================================
DenseToMoEPruner::DenseToMoEPruner(int64_t hidden, int64_t n_experts, float sparsity)
    : hidden_(hidden), n_experts_(n_experts), sparsity_(sparsity) {
    QUANT_CHECK(hidden > 0, "DenseToMoEPruner: hidden must be > 0");
    QUANT_CHECK(n_experts > 0, "DenseToMoEPruner: n_experts must be > 0");
    QUANT_CHECK(sparsity >= 0 && sparsity < 1, "DenseToMoEPruner: sparsity in [0,1)");
}

Tensor DenseToMoEPruner::prune_ffn(const Tensor& gate, const Tensor& up, const Tensor& down) {
    // gate: {hidden, ffn_hidden}, up: {hidden, ffn_hidden}, down: {ffn_hidden, hidden}
    QUANT_CHECK(gate.numel() > 0 && up.numel() > 0 && down.numel() > 0,
              "DenseToMoEPruner: empty weight tensors");
    QUANT_CHECK(gate.rank() == 2 && up.rank() == 2 && down.rank() == 2,
              "DenseToMoEPruner: weights must be 2D");

    int64_t d_model = gate.dim(0);
    int64_t ffn_dim = gate.dim(1);

    QUANT_CHECK(up.dim(0) == d_model && up.dim(1) == ffn_dim,
              "DenseToMoEPruner: gate/up shape mismatch");
    QUANT_CHECK(down.dim(0) == ffn_dim && down.dim(1) == d_model,
              "DenseToMoEPruner: down shape mismatch with gate/up");

    // Concatenate gate+up weights as the combined FFN input projection
    int64_t expert_size = d_model * ffn_dim; // gate weights per expert
    Tensor experts({n_experts_, expert_size});

    const float* gd = gate.data<float>();
    const float* ud = up.data<float>();
    const float* dd = down.data<float>();

    std::mt19937 rng(42);
    std::normal_distribution<float> noise_dist(0, 0.01f);

    float* ed = experts.data<float>();

    for (int64_t e = 0; e < n_experts_; e++) {
        float* expert_ptr = ed + e * expert_size;

        if (e == 0) {
            // First expert = original dense weights (no noise)
            std::memcpy(expert_ptr, gd, (size_t)expert_size * sizeof(float));
        } else {
            // Subsequent experts = perturbed copies
            for (int64_t i = 0; i < expert_size; i++)
                expert_ptr[i] = gd[i] + noise_dist(rng);

            // Apply sparsity mask to gate weights
            if (sparsity_ > 0) {
                int64_t n_prune = (int64_t)(sparsity_ * (float)expert_size);
                std::vector<int64_t> indices((size_t)expert_size);
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), rng);
                for (int64_t p = 0; p < n_prune && p < expert_size; p++)
                    expert_ptr[indices[(size_t)p]] = 0.0f;
            }
        }
    }

    return experts;
}

// ============================================================================
// F3: MoEQUANTFormat — binary save/load with proper shape serialization
// ============================================================================
void MoEQUANTFormat::save_experts(const std::vector<Tensor>& experts,
                                 const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return;

    int32_t n = (int32_t)experts.size();
    std::fwrite(&n, sizeof(n), 1, fp);

    for (const auto& e : experts) {
        // Save shape info: rank + dims
        int32_t rank = (int32_t)e.rank();
        std::fwrite(&rank, sizeof(rank), 1, fp);
        for (int32_t r = 0; r < rank; r++) {
            int64_t dim = e.dim(r);
            std::fwrite(&dim, sizeof(dim), 1, fp);
        }

        // Save dtype
        uint8_t dt = (uint8_t)e.dtype();
        std::fwrite(&dt, sizeof(dt), 1, fp);

        // Save data
        int64_t sz = e.numel();
        size_t bytes = (size_t)sz * dtype_size(e.dtype());
        std::fwrite(e.data(), bytes, 1, fp);
    }

    std::fclose(fp);
}

std::vector<Tensor> MoEQUANTFormat::load_experts(const std::string& path) {
    std::vector<Tensor> experts;
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return experts;

    int32_t n = 0;
    if (std::fread(&n, sizeof(n), 1, fp) != 1) {
        std::fclose(fp);
        return experts;
    }
    // BUGFIX (bug census): file-controlled n looped unbounded pushing expert
    // tensors (OOM/DoS on corrupt files). Bound before looping.
    static constexpr int32_t kMaxExpertsFile = 4096;
    if (n < 0 || n > kMaxExpertsFile) { std::fclose(fp); return experts; }

    for (int32_t i = 0; i < n; i++) {
        int32_t rank = 0;
        if (std::fread(&rank, sizeof(rank), 1, fp) != 1) break;

        if (rank < 0 || rank > 8) { std::fclose(fp); return experts; }

        int64_t dims[8] = {0};
        for (int32_t r = 0; r < rank; r++)
            if (std::fread(&dims[r], sizeof(int64_t), 1, fp) != 1) break;

        // BUGFIX (bug census): dims assigned unchecked — negative/huge dims
        // drove unbounded Tensor allocs. Validate each dim + total bytes.
        int64_t numel = 1;
        bool dims_ok = true;
        for (int32_t r = 0; r < rank; r++) {
            if (dims[r] < 0 || dims[r] > (int64_t)1 << 30) { dims_ok = false; break; }
            if (numel > ((int64_t)1 << 34) / std::max<int64_t>(1, dims[r])) { dims_ok = false; break; }
            numel *= dims[r];
        }
        if (!dims_ok) break;

        uint8_t dt = (uint8_t)DType::F32;
        // P0 fix: unchecked fread let silent corrupt experts through.
        if (std::fread(&dt, sizeof(dt), 1, fp) != 1) break;

        DType dtype = (DType)dt;
        Shape shape;
        shape.rank = rank;
        for (int32_t r = 0; r < rank; r++)
            shape.dims[r] = dims[r];

        Tensor t(shape, dtype);
        int64_t sz = t.numel();
        size_t bytes = (size_t)sz * dtype_size(dtype);
        // P0 fix: validate the payload read — truncated files must stop,
        // not push half-filled expert tensors.
        if (std::fread(t.data(), bytes, 1, fp) != 1) break;

        experts.push_back(t);
    }

    std::fclose(fp);
    return experts;
}

// ============================================================================
// F4: Expert dropout — deterministic masking
// ============================================================================
ExpertDropout::ExpertDropout(float rate) : rate_(rate) {
    QUANT_CHECK(rate >= 0 && rate < 1, "ExpertDropout: rate must be in [0,1)");
}

int64_t ExpertDropout::get_active_count(int64_t total) const {
    if (total <= 0) return 0;
    return std::max((int64_t)1, (int64_t)(total * (1.0f - rate_)));
}

std::vector<bool> ExpertDropout::get_mask(int64_t total) const {
    QUANT_CHECK(total > 0, "ExpertDropout: total must be > 0");
    std::vector<bool> mask((size_t)total, true);
    int64_t drop = std::min((int64_t)((float)total * rate_), total - 1);
    // Deterministic masking using linear congruential generator
    uint64_t state = (uint64_t)total * 0x9E3779B97F4A7C15ULL;
    int64_t dropped = 0;
    while (dropped < drop) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        int64_t idx = (int64_t)(state % (uint64_t)total);
        if (mask[(size_t)idx]) {
            mask[(size_t)idx] = false;
            dropped++;
        }
    }
    return mask;
}

// ============================================================================
// F5: Balance monitoring — proper load balance loss
// ============================================================================
ExpertBalance compute_balance(const std::vector<int64_t>& assignments, int64_t n_experts) {
    QUANT_CHECK(n_experts > 0, "compute_balance: n_experts must be > 0");

    ExpertBalance eb;
    eb.token_counts.resize((size_t)n_experts, 0);

    if (assignments.empty()) {
        eb.load_balance_loss = 1.0f;
        eb.stddev = 0.0f;
        return eb;
    }

    for (auto a : assignments)
        if (a >= 0 && a < n_experts)
            eb.token_counts[(size_t)a]++;

    float total = (float)assignments.size();
    float mean = total / (float)n_experts;
    float var = 0;
    float cv_sq = 0; // coefficient of variation squared

    for (auto c : eb.token_counts) {
        float diff = (float)c - mean;
        var += diff * diff;
        cv_sq += diff * diff / (mean * mean + 1e-8f);
    }
    var /= (float)n_experts;
    eb.stddev = std::sqrt(var);

    // Load balance loss = normalized coefficient of variation
    // CV^2 / n_experts formulation from Switch Transformer paper
    eb.load_balance_loss = cv_sq / ((float)n_experts * (float)n_experts);

    return eb;
}

// ============================================================================
// F6: SwitchV2Router — top-1 routing with capacity factor
// ============================================================================
SwitchV2Router::SwitchV2Router(int64_t n_experts, float capacity_factor)
    : n_experts_(n_experts), capacity_factor_(capacity_factor) {
    QUANT_CHECK(n_experts > 0, "SwitchV2Router: n_experts must be > 0");
    QUANT_CHECK(capacity_factor >= 1.0f, "SwitchV2Router: capacity_factor >= 1.0");
}

Tensor SwitchV2Router::forward(const Tensor& logits, int64_t tokens) {
    QUANT_CHECK(logits.numel() > 0, "SwitchV2Router: empty logits");
    QUANT_CHECK(tokens > 0, "SwitchV2Router: tokens must be > 0");

    int64_t E = n_experts_;
    int64_t T = tokens;
    int64_t logit_T = logits.dim(0);
    int64_t logit_E = logits.numel() / logit_T;

    QUANT_CHECK(logit_E == E,
              "SwitchV2Router: logits expert dim mismatch");

    int64_t capacity = (int64_t)std::ceil(capacity_factor_ * (float)T / (float)E);
    if (capacity < 1) capacity = 1;

    const float* l = logits.data<float>();

    // Output: routing weights [T, E] with capacity enforced
    Tensor routing({T, E});
    routing.zero_();
    float* rd = routing.data<float>();

    // Track per-expert token count for capacity enforcement
    std::vector<int64_t> expert_count((size_t)E, 0);

    for (int64_t t = 0; t < T; t++) {
        // Softmax over experts
        const float* row = l + t * E;
        float max_val = row[0];
        for (int64_t e = 1; e < E; e++)
            if (row[e] > max_val) max_val = row[e];

        float sum = 0;
        float probs[16]; // assume E <= 16 or dynamic
        std::vector<float> prob_vec;
        float* p;
        if (E <= 16) {
            p = probs;
        } else {
            prob_vec.resize((size_t)E);
            p = prob_vec.data();
        }

        for (int64_t e = 0; e < E; e++) {
            p[e] = std::exp(row[e] - max_val);
            sum += p[e];
        }
        if (sum > 0) {
            float inv = 1.0f / sum;
            for (int64_t e = 0; e < E; e++)
                p[e] *= inv;
        }

        // Top-1 routing
        int best_e = 0;
        float best_p = p[0];
        for (int64_t e = 1; e < E; e++) {
            if (p[e] > best_p) {
                best_p = p[e];
                best_e = (int)e;
            }
        }

        // Capacity check: if expert at capacity, token is dropped
        if (expert_count[(size_t)best_e] < capacity) {
            rd[t * E + best_e] = best_p;
            expert_count[(size_t)best_e]++;
        }
        // Dropped tokens have zero routing weight
    }

    return routing;
}

// ============================================================================
// F7: ExpertChoiceV2 — expert-choice routing
// ============================================================================
ExpertChoiceV2::ExpertChoiceV2(int64_t n_experts, int64_t top_k)
    : n_experts_(n_experts), top_k_(top_k) {
    QUANT_CHECK(n_experts > 0, "ExpertChoiceV2: n_experts must be > 0");
    QUANT_CHECK(top_k > 0, "ExpertChoiceV2: top_k must be > 0");
}

Tensor ExpertChoiceV2::forward(const Tensor& x, const Tensor& logits) {
    QUANT_CHECK(x.numel() > 0, "ExpertChoiceV2: empty input x");
    QUANT_CHECK(logits.numel() > 0, "ExpertChoiceV2: empty logits");
    QUANT_CHECK(x.rank() >= 2, "ExpertChoiceV2: x must be at least 2D");
    QUANT_CHECK(logits.rank() >= 2, "ExpertChoiceV2: logits must be at least 2D");

    int64_t T = x.dim(0);
    int64_t D = x.dim(x.rank() - 1);
    int64_t E = logits.dim(1);
    int64_t logit_T = logits.dim(0);

    QUANT_CHECK(logit_T == T, "ExpertChoiceV2: logits batch dim mismatch");

    // Use n_experts_ as the number of tokens each expert can choose
    int64_t capacity = std::min(T, std::max((int64_t)1, (int64_t)((float)T / (float)E * (float)top_k_)));

    const float* xd = x.data<float>();
    const float* ld = logits.data<float>();

    Tensor output(x.shape());
    output.zero_();
    float* od = output.data<float>();

    // Each expert picks its top-capacity tokens
    for (int64_t e = 0; e < E; e++) {
        std::vector<std::pair<float, int64_t>> scored;
        scored.reserve((size_t)T);
        for (int64_t t = 0; t < T; t++)
            scored.push_back({ld[t * E + e], t});

        std::partial_sort(scored.begin(), scored.begin() + std::min(capacity, T),
                         scored.end(),
                         [](auto& a, auto& b) { return a.first > b.first; });

        for (int64_t k = 0; k < std::min(capacity, T); k++) {
            if (scored[(size_t)k].first <= 0) continue;
            int64_t t = scored[(size_t)k].second;
            float weight = scored[(size_t)k].first;
            for (int64_t d = 0; d < D; d++)
                od[t * D + d] += weight * xd[t * D + d];
        }
    }

    // Normalize by assignment count per token
    for (int64_t t = 0; t < T; t++) {
        float count = 0;
        for (int64_t e = 0; e < E; e++)
            if (ld[t * E + e] > 0) count += 1.0f;
        if (count > 1) {
            float inv = 1.0f / count;
            for (int64_t d = 0; d < D; d++)
                od[t * D + d] *= inv;
        }
    }

    return output;
}

// ============================================================================
// F8: ExpertMerger — cosine similarity merging
// ============================================================================
ExpertMerger::ExpertMerger(float threshold) : threshold_(threshold) {
    QUANT_CHECK(threshold >= 0 && threshold <= 1,
              "ExpertMerger: threshold in [0,1]");
}

float ExpertMerger::cosine_sim(const float* a, const float* b, int64_t n) {
    float dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-10f);
}

std::vector<Tensor> ExpertMerger::merge(const std::vector<Tensor>& experts) {
    std::vector<Tensor> result;
    if (experts.empty()) return result;

    int64_t n = 0;
    for (const auto& e : experts)
        if (e.numel() > 0) { n = e.numel(); break; }

    if (n == 0) return result;

    std::vector<bool> merged(experts.size(), false);

    for (size_t i = 0; i < experts.size(); i++) {
        if (merged[i] || experts[i].numel() == 0) continue;

        Tensor avg = Tensor({n});
        avg.zero_();
        int count = 0;

        for (size_t j = i; j < experts.size(); j++) {
            if (merged[j] || experts[j].numel() == 0) continue;
            if (cosine_sim(experts[i].data<float>(), experts[j].data<float>(), n) >= threshold_) {
                float* ad = avg.data<float>();
                const float* ed = experts[j].data<float>();
                for (int64_t k = 0; k < n; k++)
                    ad[k] += ed[k];
                merged[j] = true;
                count++;
            }
        }

        if (count > 0) {
            float* ad = avg.data<float>();
            float inv = 1.0f / (float)count;
            for (int64_t k = 0; k < n; k++)
                ad[k] *= inv;
            result.push_back(avg);
        }
    }

    return result;
}

// ============================================================================
// F9: DynamicExpertPool — grow/shrink expert pool
// ============================================================================
DynamicExpertPool::DynamicExpertPool(int64_t initial, int64_t max_experts)
    : max_experts_(max_experts) {
    QUANT_CHECK(initial >= 0, "DynamicExpertPool: initial must be >= 0");
    QUANT_CHECK(max_experts > 0, "DynamicExpertPool: max_experts must be > 0");
    QUANT_CHECK(initial <= max_experts,
              "DynamicExpertPool: initial must not exceed max_experts");

    experts_.reserve((size_t)max_experts);
    for (int64_t i = 0; i < initial; i++)
        experts_.emplace_back(Tensor({0}));
}

int64_t DynamicExpertPool::add_expert(const Tensor& weights) {
    if ((int64_t)experts_.size() >= max_experts_) return -1;
    experts_.push_back(weights.clone());
    return (int64_t)experts_.size() - 1;
}

bool DynamicExpertPool::remove_expert(int64_t idx) {
    if (idx < 0 || idx >= (int64_t)experts_.size()) return false;
    experts_.erase(experts_.begin() + idx);
    return true;
}

// ============================================================================
// F10: ExpertParallel — distribute experts across devices
// ============================================================================
ExpertParallel::ExpertParallel(int64_t world_size, int64_t world_rank)
    : world_size_(world_size), world_rank_(world_rank) {
    QUANT_CHECK(world_size > 0, "ExpertParallel: world_size must be > 0");
    QUANT_CHECK(world_rank >= 0 && world_rank < world_size,
              "ExpertParallel: world_rank out of range");
}

Tensor ExpertParallel::forward(const Tensor& x, int64_t n_experts) {
    QUANT_CHECK(x.numel() > 0, "ExpertParallel: empty input x");
    QUANT_CHECK(n_experts > 0, "ExpertParallel: n_experts must be > 0");

    // Partition experts across ranks
    int64_t experts_per_rank = (n_experts + world_size_ - 1) / world_size_;
    int64_t start_idx = world_rank_ * experts_per_rank;
    int64_t end_idx = std::min(start_idx + experts_per_rank, n_experts);
    int64_t local_experts = end_idx - start_idx;

    if (local_experts <= 0)
        return Tensor({0});  // This rank has no experts

    // Each rank processes its subset of the hidden dimension
    int64_t D = x.numel();
    int64_t B = x.dim(0);
    int64_t S = x.rank() > 1 ? x.dim(1) : 1;
    int64_t H = x.dim(x.rank() - 1);

    // Local chunk of hidden dimension
    int64_t local_h = H / world_size_;
    int64_t local_start = world_rank_ * local_h;
    int64_t local_end = (world_rank_ == world_size_ - 1) ? H : local_start + local_h;
    local_h = local_end - local_start;

    // Simulate local expert processing: extract the rank's slice
    Tensor local_out({B, S, local_h});
    const float* xd = x.data<float>();
    float* lod = local_out.data<float>();

    for (int64_t b = 0; b < B; b++) {
        for (int64_t s = 0; s < S; s++) {
            int64_t flat_in = (b * S + s) * H;
            int64_t flat_out = (b * S + s) * local_h;
            for (int64_t h = 0; h < local_h; h++)
                lod[flat_out + h] = xd[flat_in + local_start + h];
        }
    }

    return local_out;
}

} // namespace quant
