#pragma once
// ============================================================================
// speculative_decoder.h — Speculative Decoding with KV Rollback
// ============================================================================
// Production speculative decoding engine with:
//   - Draft-then-verify loop with KV cache checkpoint/rewind
//   - Small-batch verification kernel
//   - Draft-length auto-tuner
//   - Theorem 13 proof
// ============================================================================

#include "quant/types.h"
#include "quant/kv_cache.h"
#include "quant/inferno_integration.h"
#include <vector>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <random>

namespace quant {

// ── Draft Model Interface ─────────────────────────────────────────────────
// A lightweight model that proposes draft tokens.

class DraftModel {
public:
    virtual ~DraftModel() = default;
    virtual std::vector<int> propose(const std::vector<int>& context,
                                      std::size_t num_tokens) = 0;
    virtual int vocab_size() const = 0;
};

// ── Target Model Interface ────────────────────────────────────────────────
// The full model that verifies draft tokens.

class TargetModel {
public:
    virtual ~TargetModel() = default;
    virtual std::vector<float> logits(const std::vector<int>& context) = 0;
    virtual int vocab_size() const = 0;
};

// ── KV Cache Checkpoint ───────────────────────────────────────────────────
// Snapshot of KV cache state for rollback after failed verification.

struct KVCheckpoint {
    std::vector<int> positions;
    std::size_t kv_context_len = 0;
    std::vector<std::vector<float>> saved_k;
    std::vector<std::vector<float>> saved_v;
    bool valid = false;
};

class SpeculativeDecoder {
public:
    struct Config {
        int draft_k = 3;
        int min_draft_k = 1;
        int max_draft_k = 8;
        double target_acceptance_rate = 0.7;
        int auto_tune_window = 100;
        float temperature = 1.0f;
        int top_k = 50;
        float top_p = 0.9f;
        Config() = default;
    };

    explicit SpeculativeDecoder(const Config& cfg);
    SpeculativeDecoder() : SpeculativeDecoder(Config{}) {}
    ~SpeculativeDecoder() = default;

    std::vector<int> generate(
        DraftModel& draft,
        TargetModel& target,
        const std::vector<int>& prompt,
        std::size_t max_tokens,
        KVCache* cache = nullptr);

    struct StepResult {
        std::vector<int> tokens_accepted;
        int tokens_rejected = 0;
        int total_proposed = 0;
        double acceptance_rate = 0.0;
    };

    StepResult decode_step(
        DraftModel& draft,
        TargetModel& target,
        std::vector<int>& context,
        KVCache* cache = nullptr);

    [[nodiscard]] int current_draft_k() const { return cfg_.draft_k; }
    [[nodiscard]] int total_accepted() const { return total_accepted_; }
    [[nodiscard]] int total_proposed() const { return total_proposed_; }
    [[nodiscard]] double acceptance_rate() const;

    void update_draft_k(int new_k);

private:
    void checkpoint_kv(KVCache* cache, KVCheckpoint& ckpt);
    void rewind_kv(KVCache* cache, const KVCheckpoint& ckpt, std::size_t new_len);

    int sample_token(const std::vector<float>& logits) const;
    std::vector<int> topk_sample(const std::vector<float>& logits, int k) const;

    void auto_tune_draft_k();

    Config cfg_;
    int total_accepted_ = 0;
    int total_proposed_ = 0;
    std::vector<double> acceptance_window_;
    mutable std::mt19937_64 rng_{42};
};

// ── Theorem 13: Speculative Decoding Speedup ──────────────────────────────
// Statement: Let α be the acceptance rate of the draft model, k the draft
// length, and c the cost ratio (draft_cost / target_cost). Then the expected
// speedup of speculative decoding is:
//
//   S(k, α) = (1 + k * α) / (1 + k * c)
//
// The optimal draft length is:
//
//   k* = sqrt(2 / (1 - α)) when α > c
//
// Proof: See publication/whitepaper/chapters/ch07_mtp_speculative.tex

struct Theorem13Result {
    double speedup;
    int optimal_k;
    double acceptance_rate;
    double cost_ratio;

    static Theorem13Result compute(double alpha, double cost_ratio) {
        Theorem13Result r;
        r.acceptance_rate = alpha;
        r.cost_ratio = cost_ratio;

        if (alpha <= cost_ratio) {
            r.speedup = 1.0;
            r.optimal_k = 1;
            return r;
        }

        double k_opt_d = std::sqrt(2.0 / (1.0 - alpha));
        r.optimal_k = std::max(1, static_cast<int>(std::round(k_opt_d)));

        r.speedup = (1.0 + r.optimal_k * alpha) /
                    (1.0 + r.optimal_k * cost_ratio);

        return r;
    }
};

// ── Small-Batch Verification Kernel ───────────────────────────────────────
// Verifies a batch of draft tokens against the target model.
// Returns per-position acceptance flags.

struct VerificationResult {
    std::vector<bool> accepted;
    int first_rejection_pos = -1;
    int total_accepted = 0;
    double verify_flops = 0.0;
    double verify_inferno = 0.0;
};

class SmallBatchVerifier {
public:
    explicit SmallBatchVerifier(int max_batch = 8);

    VerificationResult verify(
        const std::vector<std::vector<float>>& draft_logits,
        const std::vector<std::vector<float>>& target_logits,
        float temperature = 1.0f);

    void set_format_index(int idx) { format_index_ = idx; }

private:
    bool token_match(const std::vector<float>& draft_dist,
                      const std::vector<float>& target_dist,
                      float temperature) const;

    int max_batch_;
    int format_index_ = 5;
};

} // namespace quant
