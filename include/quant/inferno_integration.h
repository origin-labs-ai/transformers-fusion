#pragma once
// ============================================================================
// inferno_integration.h — Inferno Global Integration Hooks
// ============================================================================
// Connects Inferno counter into training loop, inference engine, quantizer,
// small-batch verification kernel, stability-plasticity tuner, and
// draft-length auto-tuner.
//
// All hooks are zero-cost when disabled (compile-time QUANT_INFERNO_ENABLED).
// When enabled, they accumulate Inferno counters at every critical path.
// ============================================================================

#include "quant/inferno.h"
#include "quant/format_registry.h"
#include <cstdint>
#include <cstddef>
#include <functional>
#include <atomic>

namespace quant {

// ── Compile-time gate ─────────────────────────────────────────────────────
#ifndef QUANT_INFERNO_ENABLED
#define QUANT_INFERNO_ENABLED 1
#endif

// ── Global Inferno state ─────────────────────────────────────────────────────

struct InfernoGlobalState {
    InfernoCounter training;
    InfernoCounter inference;
    InfernoCounter quantization;

    std::atomic<int64_t> small_batch_calls{0};
    std::atomic<int64_t> small_batch_elements{0};
    std::atomic<double>  small_batch_avg_inferno{0.0};

    std::atomic<int64_t> stability_plasticity_updates{0};
    std::atomic<double>  current_stability_factor{1.0};
    std::atomic<double>  current_plasticity_factor{1.0};

    std::atomic<int>     current_draft_k{1};
    std::atomic<double>  current_acceptance_rate{0.0};
    std::atomic<int64_t> speculative_accepted{0};
    std::atomic<int64_t> speculative_proposed{0};

    void reset() {
        training.reset();
        inference.reset();
        quantization.reset();
        small_batch_calls.store(0);
        small_batch_elements.store(0);
        small_batch_avg_inferno.store(0.0);
        stability_plasticity_updates.store(0);
        current_stability_factor.store(1.0);
        current_plasticity_factor.store(1.0);
        current_draft_k.store(1);
        current_acceptance_rate.store(0.0);
        speculative_accepted.store(0);
        speculative_proposed.store(0);
    }
};

InfernoGlobalState& inferno_global();

// ── Training Hook ─────────────────────────────────────────────────────────
// Called at each training step to record Inferno for the given format.

inline void inferno_hook_training_start() {
#if QUANT_INFERNO_ENABLED
    inferno_global().training.start();
#endif
}

inline void inferno_hook_training_stop(int format_index, int64_t elements) {
#if QUANT_INFERNO_ENABLED
    inferno_global().training.stop();
    inferno_global().training.record(format_index, elements);
#endif
}

inline void inferno_hook_training_stop_mixed(const int* format_indices,
                                           const int64_t* counts, int n_tiers) {
#if QUANT_INFERNO_ENABLED
    inferno_global().training.stop();
    for (int i = 0; i < n_tiers; ++i) {
        if (format_indices[i] < INFERNO_NUM_FORMATS) {
            inferno_global().training.record(format_indices[i], counts[i]);
        }
    }
#endif
}

// ── Inference Hook ────────────────────────────────────────────────────────

inline void inferno_hook_inference_start() {
#if QUANT_INFERNO_ENABLED
    inferno_global().inference.start();
#endif
}

inline void inferno_hook_inference_stop(int format_index, int64_t elements) {
#if QUANT_INFERNO_ENABLED
    inferno_global().inference.stop();
    inferno_global().inference.record(format_index, elements);
#endif
}

// ── Quantization Hook ─────────────────────────────────────────────────────

inline void inferno_hook_quantize_start() {
#if QUANT_INFERNO_ENABLED
    inferno_global().quantization.start();
#endif
}

inline void inferno_hook_quantize_stop(int format_index, int64_t elements) {
#if QUANT_INFERNO_ENABLED
    inferno_global().quantization.stop();
    inferno_global().quantization.record(format_index, elements);
#endif
}

// ── Small-Batch Verification Kernel Hook ──────────────────────────────────
// For speculative decoding: verify a small batch of draft tokens
// against the target model. Records Inferno for the verification pass.

inline void inferno_hook_small_batch(int format_index, int64_t elements) {
#if QUANT_INFERNO_ENABLED
    auto& gs = inferno_global();
    gs.small_batch_calls.fetch_add(1, std::memory_order_relaxed);
    gs.small_batch_elements.fetch_add(elements, std::memory_order_relaxed);

    InfernoCounter local;
    local.cpu_ghz = gs.training.cpu_ghz;
    local.start();
    local.record(format_index, elements);
    local.stop();
    double s = local.inferno();

    double prev = gs.small_batch_avg_inferno.load(std::memory_order_relaxed);
    int64_t calls = gs.small_batch_calls.load(std::memory_order_relaxed);
    if (calls <= 1) {
        gs.small_batch_avg_inferno.store(s, std::memory_order_relaxed);
    } else {
        double new_avg = prev + (s - prev) / static_cast<double>(calls);
        gs.small_batch_avg_inferno.store(new_avg, std::memory_order_relaxed);
    }
#endif
}

// ── Stability-Plasticity Hook ─────────────────────────────────────────────
// Adjusts learning rate factors based on forgetting metrics.
// Called by ContinualTrainer after each step.

struct StabilityPlasticityParams {
    float forgetting_weight;
    float current_lr_factor;
    float stability_lr_factor;
    float plasticity_lr_factor;
    float forgetting_threshold;
};

inline StabilityPlasticityParams inferno_hook_stability_plasticity(
        float forgetting_weight, float base_lr,
        float stability_factor, float plasticity_factor,
        float threshold) {
#if QUANT_INFERNO_ENABLED
    auto& gs = inferno_global();
    gs.stability_plasticity_updates.fetch_add(1, std::memory_order_relaxed);

    float effective_lr = base_lr;
    float sf = stability_factor;
    float pf = plasticity_factor;

    if (forgetting_weight > threshold) {
        sf *= (1.0f + (forgetting_weight - threshold));
        pf *= (1.0f - 0.5f * (forgetting_weight - threshold));
    }

    sf = (sf < 0.1f) ? 0.1f : ((sf > 5.0f) ? 5.0f : sf);
    pf = (pf < 0.1f) ? 0.1f : ((pf > 5.0f) ? 5.0f : pf);

    gs.current_stability_factor.store(sf, std::memory_order_relaxed);
    gs.current_plasticity_factor.store(pf, std::memory_order_relaxed);

    effective_lr *= (forgetting_weight > threshold) ? sf : pf;

    return StabilityPlasticityParams{
        forgetting_weight, effective_lr, sf, pf, threshold
    };
#else
    return StabilityPlasticityParams{
        forgetting_weight, base_lr, stability_factor, plasticity_factor, threshold
    };
#endif
}

// ── Draft-Length Auto-Tuner Hook ──────────────────────────────────────────
// Adjusts speculative decoding draft-k based on measured acceptance rate.

inline int inferno_hook_draft_length_tune(int current_k, double acceptance_rate,
                                        int min_k, int max_k,
                                        double target_rate) {
#if QUANT_INFERNO_ENABLED
    auto& gs = inferno_global();
    gs.current_acceptance_rate.store(acceptance_rate, std::memory_order_relaxed);

    int new_k = current_k;
    if (acceptance_rate > target_rate + 0.05) {
        new_k = current_k + 1;
    } else if (acceptance_rate < target_rate - 0.10) {
        new_k = current_k - 1;
    }

    if (new_k < min_k) new_k = min_k;
    if (new_k > max_k) new_k = max_k;

    gs.current_draft_k.store(new_k, std::memory_order_relaxed);
    return new_k;
#else
    (void)acceptance_rate; (void)target_rate;
    return current_k;
#endif
}

// ── Speculative Decode Tracking ───────────────────────────────────────────

inline void inferno_hook_speculative_propose(int count) {
#if QUANT_INFERNO_ENABLED
    inferno_global().speculative_proposed.fetch_add(count, std::memory_order_relaxed);
#endif
}

inline void inferno_hook_speculative_accept(int count) {
#if QUANT_INFERNO_ENABLED
    inferno_global().speculative_accepted.fetch_add(count, std::memory_order_relaxed);
#endif
}

// ── Reporting ─────────────────────────────────────────────────────────────

struct InfernoReport {
    double training_flops;
    double inference_flops;
    double quantization_flops;
    double training_inferno;
    double inference_inferno;
    double quantization_inferno;
    double effective_training_gflops;
    double effective_inference_gflops;

    int64_t small_batch_calls;
    double  small_batch_avg_inferno;

    int64_t stability_updates;
    double  stability_factor;
    double  plasticity_factor;

    int     draft_k;
    double  acceptance_rate;
    int64_t speculative_accepted;
    int64_t speculative_proposed;
    double  speculative_efficiency;
};

inline InfernoReport inferno_report() {
#if QUANT_INFERNO_ENABLED
    auto& gs = inferno_global();
    int64_t prop = gs.speculative_proposed.load(std::memory_order_relaxed);
    int64_t acc  = gs.speculative_accepted.load(std::memory_order_relaxed);

    return InfernoReport{
        gs.training.flops(),
        gs.inference.flops(),
        gs.quantization.flops(),
        gs.training.inferno(),
        gs.inference.inferno(),
        gs.quantization.inferno(),
        gs.training.effective_gflops(),
        gs.inference.effective_gflops(),

        gs.small_batch_calls.load(std::memory_order_relaxed),
        gs.small_batch_avg_inferno.load(std::memory_order_relaxed),

        gs.stability_plasticity_updates.load(std::memory_order_relaxed),
        gs.current_stability_factor.load(std::memory_order_relaxed),
        gs.current_plasticity_factor.load(std::memory_order_relaxed),

        gs.current_draft_k.load(std::memory_order_relaxed),
        gs.current_acceptance_rate.load(std::memory_order_relaxed),
        acc, prop,
        (prop > 0) ? static_cast<double>(acc) / static_cast<double>(prop) : 0.0
    };
#else
    return InfernoReport{};
#endif
}

} // namespace quant
