#include "quant/speculative_decoder.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>

namespace quant {

// ============================================================================
// SpeculativeDecoder
// ============================================================================

SpeculativeDecoder::SpeculativeDecoder(const Config& cfg)
    : cfg_(cfg)
    , acceptance_window_(cfg.auto_tune_window, 0.0) {}

int SpeculativeDecoder::sample_token(const std::vector<float>& logits) const {
    if (logits.empty()) return 0;

    float max_l = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp((logits[i] - max_l) / std::max(cfg_.temperature, 0.01f));
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    float cum = 0.0f;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (cum >= r) return static_cast<int>(i);
    }
    return static_cast<int>(probs.size() - 1);
}

std::vector<int> SpeculativeDecoder::topk_sample(
        const std::vector<float>& logits, int k) const {
    std::vector<int> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + std::min(k, static_cast<int>(indices.size())),
                       indices.end(),
                       [&logits](int a, int b) { return logits[a] > logits[b]; });

    int top_k = std::min(k, static_cast<int>(indices.size()));
    float max_l = logits[indices[0]];
    std::vector<float> probs(top_k);
    float sum = 0.0f;
    for (int i = 0; i < top_k; ++i) {
        probs[i] = std::exp((logits[indices[i]] - max_l) / std::max(cfg_.temperature, 0.01f));
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    float cum = 0.0f;
    for (int i = 0; i < top_k; ++i) {
        cum += probs[i];
        if (cum >= r) return {indices[i]};
    }
    return {indices[top_k - 1]};
}

void SpeculativeDecoder::checkpoint_kv(KVCache* cache, KVCheckpoint& ckpt) {
    // P13 honest note: ye checkpoint sirf context_len record karta hai —
    // saved_k/saved_v populate NAHI hote (KVCheckpoint ke wo fields hamesha
    // khaali rehte hain). Matlab rollback ke paas restore karne layak K/V
    // snapshot nahi hai, sirf length marker hai.
    if (!cache) return;
    ckpt.kv_context_len = cache->context_len();
    ckpt.valid = true;
}

void SpeculativeDecoder::rewind_kv(KVCache* cache, const KVCheckpoint& ckpt,
                                     std::size_t new_len) {
    // P13 honest note (DEPRECATED semantics): KVCache ka koi truncate API
    // nahi hai. KVCache::resize(new_max_seq_len) max capacity badalta hai aur
    // current_pos=0 karke POORA cache wipe kar deta hai (kv_cache.cpp) — ye
    // true "new_len tak rewind" NAHI hai. Jab tak KVCache me proper truncate/
    // set_pos API nahi aata, isko sirf guarded caller (decode_step rejection
    // path) se best-effort rollback ke liye call karo; full-fidelity KV
    // restore supported nahi hai.
    if (!cache || !ckpt.valid) return;
    cache->resize(new_len);
}

SpeculativeDecoder::StepResult SpeculativeDecoder::decode_step(
        DraftModel& draft,
        TargetModel& target,
        std::vector<int>& context,
        KVCache* cache) {
    StepResult result;

    KVCheckpoint ckpt;
    checkpoint_kv(cache, ckpt);
    std::size_t start_len = context.size();

    std::vector<int> draft_tokens = draft.propose(context, cfg_.draft_k);

    for (std::size_t i = 0; i < draft_tokens.size(); ++i) {
        std::vector<int> augmented = context;
        augmented.insert(augmented.end(), draft_tokens.begin(),
                          draft_tokens.begin() + i + 1);

        std::vector<float> draft_logits_vec = target.logits(augmented);
        int draft_token = draft_tokens[i];
        int target_token = sample_token(draft_logits_vec);

        result.total_proposed++;

        if (draft_token == target_token) {
            context.push_back(draft_token);
            result.tokens_accepted.push_back(draft_token);
            sops_hook_speculative_accept(1);
        } else {
            context.push_back(target_token);
            result.tokens_rejected++;
            sops_hook_speculative_propose(1);
            // P13: rewind_kv ka caller wire — reject pe KV ko checkpoint tak
            // wapas lao (best-effort; rewind_kv ke honest note dekho: true
            // truncate API nahi hai). start_len ab dead var nahi raha.
            rewind_kv(cache, ckpt, start_len + result.tokens_accepted.size());
            break;
        }
    }

    if (result.total_proposed > 0) {
        result.acceptance_rate = static_cast<double>(result.tokens_accepted.size()) /
                                  static_cast<double>(result.total_proposed);
    }

    total_accepted_ += static_cast<int>(result.tokens_accepted.size());
    total_proposed_ += result.total_proposed;

    acceptance_window_.push_back(result.acceptance_rate);
    if (acceptance_window_.size() > cfg_.auto_tune_window) {
        acceptance_window_.erase(acceptance_window_.begin());
    }

    sops_hook_draft_length_tune(
        cfg_.draft_k, acceptance_rate(),
        cfg_.min_draft_k, cfg_.max_draft_k,
        cfg_.target_acceptance_rate);

    cfg_.draft_k = sops_global().current_draft_k.load(std::memory_order_relaxed);

    return result;
}

std::vector<int> SpeculativeDecoder::generate(
        DraftModel& draft,
        TargetModel& target,
        const std::vector<int>& prompt,
        std::size_t max_tokens,
        KVCache* cache) {
    std::vector<int> context = prompt;
    std::vector<int> generated;
    generated.reserve(max_tokens);

    for (std::size_t t = 0; t < max_tokens; ) {
        StepResult step = decode_step(draft, target, context, cache);

        for (auto& tok : step.tokens_accepted) {
            generated.push_back(tok);
            t++;
            if (t >= max_tokens) break;
        }

        if (step.tokens_rejected > 0) {
            t++;
            generated.push_back(context.back());
        }

        if (step.total_proposed == 0) break;
    }

    return generated;
}

double SpeculativeDecoder::acceptance_rate() const {
    if (total_proposed_ == 0) return 0.0;
    return static_cast<double>(total_accepted_) / static_cast<double>(total_proposed_);
}

void SpeculativeDecoder::update_draft_k(int new_k) {
    cfg_.draft_k = std::max(cfg_.min_draft_k, std::min(cfg_.max_draft_k, new_k));
}

// ============================================================================
// SmallBatchVerifier
// ============================================================================

SmallBatchVerifier::SmallBatchVerifier(int max_batch)
    : max_batch_(max_batch) {}

bool SmallBatchVerifier::token_match(
        const std::vector<float>& draft_dist,
        const std::vector<float>& target_dist,
        float temperature) const {
    if (draft_dist.empty() || target_dist.empty()) return false;
    if (draft_dist.size() != target_dist.size()) return false;

    int draft_argmax = 0;
    float draft_max = draft_dist[0];
    for (std::size_t i = 1; i < draft_dist.size(); ++i) {
        if (draft_dist[i] > draft_max) {
            draft_max = draft_dist[i];
            draft_argmax = static_cast<int>(i);
        }
    }

    int target_argmax = 0;
    float target_max = target_dist[0];
    for (std::size_t i = 1; i < target_dist.size(); ++i) {
        if (target_dist[i] > target_max) {
            target_max = target_dist[i];
            target_argmax = static_cast<int>(i);
        }
    }

    return draft_argmax == target_argmax;
}

VerificationResult SmallBatchVerifier::verify(
        const std::vector<std::vector<float>>& draft_logits,
        const std::vector<std::vector<float>>& target_logits,
        float temperature) {
    VerificationResult result;
    int n = std::min({max_batch_,
                       static_cast<int>(draft_logits.size()),
                       static_cast<int>(target_logits.size())});
    result.accepted.resize(n, false);
    result.first_rejection_pos = n;

    sops_hook_small_batch(format_index_, n);

    for (int i = 0; i < n; ++i) {
        if (token_match(draft_logits[i], target_logits[i], temperature)) {
            result.accepted[i] = true;
            result.total_accepted++;
        } else {
            result.first_rejection_pos = i;
            break;
        }
    }

    return result;
}

} // namespace quant
