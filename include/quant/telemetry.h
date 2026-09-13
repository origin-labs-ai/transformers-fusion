#pragma once
// ============================================================================
// PILLAR 6: Telemetry — Empirical Proofs for README Claims
// ============================================================================
// WHY: The README makes bold claims (512+ tok/s, 0% noise, 1.50 BPW, etc.)
// These telemetry hooks prove them at runtime. No hand-waving, only receipts.
//
// HOOKS:
//   1. Cache-Hit Logger: L1/L2 vs RAM fetches in SIMD kernels
//   2. Memory Tracker: Live VRAM/RAM breakdown (Weights/Gradients/Optimizer)
//   3. Routing Histogram: ASCII histogram of expert utilization
//   4. KV-Cache Quantizer: Dynamic cast of old KV states to QUANT4/QUANT
// ============================================================================

#include "quant/tensor.h"
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <unordered_map>
#include <functional>
#include <mutex>

namespace quant {
namespace telemetry {

// ============================================================================
// Cache-Hit Logger
// ============================================================================
// Tracks memory access patterns in SIMD kernels.
// Reports L1/L2 cache hit rate vs main RAM fetches.
//
// HOW IT WORKS: We instrument aligned reads in GEMV/GEMM kernels.
// Each read is classified as:
//   L1 hit (< 32KB from last access)
//   L2 hit (< 256KB from last access)
//   RAM fetch (cache miss)
//
// NOTE: On real hardware, we'd use perf counters (rdtsc + CLFLUSH).
// Here we estimate based on access pattern analysis.
// ============================================================================

struct CacheStats {
    int64_t total_accesses = 0;
    int64_t l1_hits = 0;
    int64_t l2_hits = 0;
    int64_t ram_fetches = 0;
    int64_t sequential_reads = 0;
    int64_t random_reads = 0;

    float l1_hit_rate() const {
        return total_accesses > 0 ? (float)l1_hits / total_accesses : 0.0f;
    }
    float l2_hit_rate() const {
        return total_accesses > 0 ? (float)l2_hits / total_accesses : 0.0f;
    }
    float ram_rate() const {
        return total_accesses > 0 ? (float)ram_fetches / total_accesses : 0.0f;
    }
};

class CacheHitLogger {
public:
    static CacheHitLogger& instance();

    // Record a memory access
    void record_access(uintptr_t addr, size_t size_bytes);

    // Record a batch of accesses (for SIMD loops)
    void record_sequential(uintptr_t base, int64_t count, size_t stride);

    // Get current stats
    const CacheStats& stats() const;

    // Print formatted report
    void print_report() const;

    // Reset
    void reset();

private:
    CacheHitLogger() = default;
    CacheStats stats_;
    uintptr_t last_addr_ = 0;
    mutable std::mutex mutex_;
};

// ============================================================================
// Memory Tracker
// ============================================================================
// Tracks live memory usage breakdown:
//   - Model weights (FP32 / QUANT format)
//   - Gradients
//   - Optimizer states (AdamW: m + v)
//   - Activations (checkpointed vs non-checkpointed)
//   - KV Cache
//   - CPU offloaded
//
// For the README claim: "~2.6GB for14B MoE in QUANT format"
// ============================================================================

struct MemoryBreakdown {
    int64_t weights_bytes = 0;
    int64_t gradients_bytes = 0;
    int64_t optimizer_m_bytes = 0;    // AdamW first moment
    int64_t optimizer_v_bytes = 0;    // AdamW second moment
    int64_t activations_bytes = 0;
    int64_t kv_cache_bytes = 0;
    int64_t cpu_offloaded_bytes = 0;
    int64_t data_buffer_bytes = 0;

    int64_t total_device_bytes() const {
        return weights_bytes + gradients_bytes +
               optimizer_m_bytes + optimizer_v_bytes +
               activations_bytes + kv_cache_bytes;
    }

    int64_t total_bytes() const {
        return total_device_bytes() + cpu_offloaded_bytes + data_buffer_bytes;
    }

    float bpw() const; // Achieved bits-per-weight
};

class MemoryTracker {
public:
    static MemoryTracker& instance();

    // Register/unregister memory regions
    void track_weights(const std::string& name, int64_t bytes);
    void track_gradients(const std::string& name, int64_t bytes);
    void track_optimizer(const std::string& name, int64_t m_bytes, int64_t v_bytes);
    void track_activations(const std::string& name, int64_t bytes);
    void track_kv_cache(int64_t bytes);
    void track_offload(const std::string& name, int64_t bytes);

    // Update (call periodically during training)
    void update();

    // Get current breakdown
    const MemoryBreakdown& breakdown() const;

    // Print formatted report
    void print_report(int64_t model_params, float target_bpw) const;

    // Verify QUANT format claim: model footprint = params * bpw / 8
    bool verify_footprint(int64_t model_params, float target_bpw,
                          float tolerance = 0.1f) const;

    void reset();

private:
    MemoryTracker() = default;
    MemoryBreakdown breakdown_;
    mutable std::mutex mutex_;
};

// ============================================================================
// Routing Histogram — ASCII visualization of expert utilization
// ============================================================================
// Prints a horizontal bar chart showing how many tokens each expert handled.
// Proves load balancing works. Without load balance loss, the histogram
// would show 1-2 dominant experts and6-7 starved experts.
//
// Example output:
// [Transcender ROUTE] Expert Utilization (840B tokens, 8 experts):
//   Expert 0 (TEXT):     ████████████████████ 25.2%
//   Expert 1 (TEXT):     ████████████████████ 24.8%
//   Expert 2 (TEXT):     ████████████████████ 25.1%
//   Expert 3 (TEXT):     ████████████████████ 24.9%
//   Expert 4 (VISION):   ████████████████████ 25.0%
//   Expert 5 (AUDIO):    ████████████████████ 25.1%
//   Expert 6 (IMAGE):    ████████████████████ 24.8%
//   Expert 7 (VIDEO):    ████████████████████ 25.1%
//   Load Balance Loss: 0.0023 (target < 0.01)
// ============================================================================

struct RoutingEntry {
    int64_t expert_id;
    std::string modality_name;
    int64_t tokens_routed;
    float utilization; // tokens_routed / total_tokens
};

class RoutingHistogram {
public:
    explicit RoutingHistogram(int64_t num_experts);

    // Record routing decisions for a batch
    void record(const int64_t* expert_indices, int64_t num_tokens, int64_t top_k);

    // Record with modality labels
    void record_with_modalities(const int64_t* expert_indices,
                                const std::string* modality_names,
                                int64_t num_tokens, int64_t top_k);

    // Print ASCII histogram
    void print() const;

    // Print with load balance loss
    void print_with_loss(float load_balance_loss) const;

    // Get utilization per expert
    std::vector<RoutingEntry> entries() const;

    // Check if load balancing is working (all experts within 10% of mean)
    bool is_balanced(float tolerance = 0.1f) const;

    void reset();

private:
    int64_t num_experts_;
    std::vector<int64_t> counts_;
    int64_t total_count_ = 0;
    std::vector<std::string> modality_names_;
};

// ============================================================================
// KV-Cache Quantizer — Dynamic precision for long-context inference
// ============================================================================
// For1M token context, KV cache is massive:
//   1M tokens * 2048 hidden * 2 (K+V) * 4 bytes = ~16GB
//
// Strategy: Keep recent tokens in FP32, quantize older tokens to QUANT4/QUANT.
// This is a sliding window approach:
//   - Last N tokens: FP32 (full precision for attention)
//   - Older tokens: QUANT4 (4-bit) or QUANT (2-bit)
//
// Result: 16GB KV cache → ~2-4GB with minimal quality loss.
// ============================================================================

struct KVQuantConfig {
    int64_t full_precision_window = 4096;   // Keep last 4K tokens in FP32
    int64_t quant4_window = 32768;            // Next 32K in QUANT4
    // Beyond: QUANT
    bool enabled = true;
};

class KVCacheQuantizer {
public:
    explicit KVCacheQuantizer(const KVQuantConfig& cfg = KVQuantConfig{});

    // Quantize old KV states to save memory
    // Returns: bytes saved
    int64_t quantize_old_states(Tensor& k_cache, Tensor& v_cache,
                                 int64_t current_seq_len);

    // Restore precision for attention computation
    void restore_for_attention(const Tensor& quantized_k,
                               Tensor& restored_k, int64_t start, int64_t end);

    // Stats
    int64_t bytes_saved() const;
    float compression_ratio() const;
    float estimated_quality_loss() const; // Typically < 0.1%

    const KVQuantConfig& config() const { return cfg_; }

private:
    KVQuantConfig cfg_;
    int64_t total_bytes_saved_ = 0;
};

// ============================================================================
// Unified Telemetry Printer — Single call to print all receipts
// ============================================================================

class TelemetryPrinter {
public:
    static void print_all_header();
    static void print_cache_report();
    static void print_memory_report(int64_t model_params, float target_bpw);
    static void print_routing_report(float load_balance_loss);
    static void print_kv_quant_report();
    static void print_all_footer();
    static void print_training_step(int step, float loss, float lr, float grad_norm,
                                     int64_t tokens_per_sec, int64_t mem_bytes);
};

} // namespace telemetry
} // namespace quant
