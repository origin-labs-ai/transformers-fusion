#pragma once
// ============================================================================
// constants.h — Named Constants for Transcender
// ============================================================================
// Eliminates magic numbers across the entire codebase.
// Every numeric literal with semantic meaning lives here.
// ============================================================================

#include <cstdint>
#include <cstddef>

namespace quant {

// ── Model Defaults ────────────────────────────────────────────────────────

constexpr int    DEFAULT_VOCAB_SIZE     = 32000;
constexpr int    DEFAULT_MAX_SEQ_LEN    = 2048;
constexpr int    DEFAULT_NUM_LAYERS     = 12;
constexpr int    DEFAULT_NUM_HEADS      = 12;
constexpr int    DEFAULT_HIDDEN_SIZE    = 768;
constexpr int    DEFAULT_FF_SIZE        = 3072;
constexpr int    DEFAULT_BLOCK_SIZE     = 128;
constexpr int    DEFAULT_BATCH_SIZE     = 8;
constexpr int    DEFAULT_MAX_TOKENS     = 1024;
constexpr int    DEFAULT_MAX_EXPERTS    = 8;

// ── Model Size Presets ────────────────────────────────────────────────────

constexpr int64_t TINY_MODEL_PARAMS    = 85'000'000;      // 85M
constexpr int64_t SMALL_MODEL_PARAMS   = 350'000'000;     // 350M
constexpr int64_t MEDIUM_MODEL_PARAMS  = 1'000'000'000LL; // 1B
constexpr int64_t LARGE_MODEL_PARAMS   = 7'000'000'000LL; // 7B

// ── Quantization Constants ────────────────────────────────────────────────

constexpr int    QUANT4_CODEBOOK_SIZE    = 16;
constexpr int    QUANT8_CODEBOOK_SIZE    = 256;
constexpr int    Q1_CENTROIDS        = 1;
constexpr int    Q2_CENTROIDS        = 4;
constexpr int    QUANT4_CENTROIDS        = 16;
constexpr int    QUANT8_CENTROIDS        = 256;
constexpr int    Q1_5_CENTROIDS    = 4;
constexpr int    QUANT16_CENTROIDS       = 0;    // QUANT16 stores raw FP16, no centroids
constexpr int    QUANT_FORMAT_COUNT      = 15;

constexpr float  FP16_MAX_VALUE        = 65504.0f;
constexpr float  FP32_EPSILON          = 1e-8f;
constexpr float  Q1_5_BPW          = 1.50f;

// ── Training Defaults ─────────────────────────────────────────────────────

constexpr float  DEFAULT_LR            = 1e-4f;
constexpr float  DEFAULT_WEIGHT_DECAY  = 1e-2f;
constexpr float  DEFAULT_BETA1         = 0.9f;
constexpr float  DEFAULT_BETA2         = 0.999f;
constexpr float  DEFAULT_EPSILON       = 1e-8f;
constexpr float  DEFAULT_GRAD_CLIP     = 1.0f;
constexpr float  DEFAULT_WARMUP_FACTOR = 0.01f;
constexpr int    DEFAULT_WARMUP_STEPS  = 100;
constexpr int    DEFAULT_EVAL_INTERVAL = 100;
constexpr int    DEFAULT_SAVE_INTERVAL = 1000;
constexpr int    DEFAULT_LOG_INTERVAL  = 10;
constexpr int    DEFAULT_MAX_STEPS     = 10000;
constexpr float  DEFAULT_TEMPERATURE   = 1.0f;
constexpr int    DEFAULT_TOP_K         = 50;
constexpr float  DEFAULT_TOP_P         = 0.9f;

// ── Continual Learning Defaults ───────────────────────────────────────────

constexpr std::size_t DEFAULT_REPLAY_CAPACITY = 65536;
constexpr float  DEFAULT_ECC_LAMBDA    = 1.0f;
constexpr float  DEFAULT_ECC_ALPHA     = 0.01f;
constexpr float  DEFAULT_FORGETTING_THRESHOLD = 0.1f;
constexpr float  DEFAULT_STABILITY_LR  = 0.5f;
constexpr float  DEFAULT_PLASTICITY_LR = 2.0f;
constexpr float  DEFAULT_FORGETTING_TEMP = 100.0f;
constexpr float  DEFAULT_IMPORTANCE_DECAY = 0.99f;
constexpr int    DEFAULT_FORGET_CHECK_INTERVAL = 100;

// ── Speculative Decoding Defaults ─────────────────────────────────────────

constexpr int    DEFAULT_DRAFT_K       = 3;
constexpr int    MIN_DRAFT_K           = 1;
constexpr int    MAX_DRAFT_K           = 8;
constexpr double DEFAULT_ACCEPTANCE_RATE = 0.7;
constexpr int    DEFAULT_TUNER_WINDOW  = 100;

// ── Memory / Buffer Constants ─────────────────────────────────────────────

constexpr std::size_t DEFAULT_POOL_SIZE     = 64 * 1024 * 1024;   // 64MB
constexpr std::size_t INITIAL_BUF_RESERVE   = 16 * 1024;           // 16KB
constexpr std::size_t L1_CACHE_BYTES        = 32 * 1024;           // 32KB
constexpr std::size_t L2_CACHE_BYTES        = 256 * 1024;          // 256KB
constexpr std::size_t L3_CACHE_BYTES        = 8 * 1024 * 1024;     // 8MB
constexpr std::size_t ONE_MIB               = 1024 * 1024;
constexpr std::size_t DEFAULT_PHYSICAL_MEM   = 8ULL * 1024 * 1024 * 1024; // 8GB

// ── HTTP / Production Constants ───────────────────────────────────────────

constexpr int    HTTP_TIMEOUT_MS       = 15000;   // 15 seconds
constexpr int    CORS_MAX_AGE          = 86400;   // 24 hours in seconds
constexpr int    DEFAULT_PORT          = 8080;
constexpr int    MAX_CONNECTIONS       = 128;

// ── Dataset / Tokenizer Constants ─────────────────────────────────────────

constexpr int64_t DEFAULT_EST_DATASET_SIZE = 1'000'000;
constexpr int    MAX_BPE_MERGES         = 50000;
constexpr int    DEFAULT_MAX_NGRAM      = 4;

// ── Expert Parallelism Constants ──────────────────────────────────────────

constexpr int    MAX_EXPERT_COUNT      = 10000;
constexpr int    DEFAULT_TOP_K_EXPERTS = 2;

// ── Paged KV Cache Constants ──────────────────────────────────────────────

constexpr int64_t KV_TABLE_ENTRIES     = 4096;
constexpr int64_t DEFAULT_KV_BLOCK_SIZE = 16;
constexpr int64_t KV_4M_MAX_TOKENS     = (int64_t(1) << 22);  // ~4M
constexpr int64_t KV_1T_MAX_TOKENS     = (int64_t(1) << 40);  // 1T+
constexpr int    FP8_BLOCK_SIZE        = 64;
constexpr float  FP8_MAX_VALUE         = 127.0f;

// ── Random / Reward Constants ─────────────────────────────────────────────

constexpr int    RANDOM_RANGE          = 2000;
constexpr int    REWARD_BASE           = 1000;

// ── Telemetry / Cache Thresholds ──────────────────────────────────────────

constexpr std::size_t TELEMETRY_CACHE_LARGE  = 256 * 1024;    // 256KB
constexpr std::size_t TELEMETRY_CACHE_HUGE   = 1024 * 1024;   // 1MB

// ── Architecture Search Constants ─────────────────────────────────────────

constexpr int    NAS_SEARCH_RANGE      = 4080;
constexpr int    NAS_STEP_SIZE         = 24;

} // namespace quant
