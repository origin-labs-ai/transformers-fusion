#pragma once

// ============================================================================
// Transcender Fine-Tuning Engine — three native strategies.
//
//   1. SelectiveFineTuner        — "where to overwrite": gradient-saliency +
//                                   Fisher-prior driven block selection.
//                                   Base weights are updated ONLY in blocks
//                                   the gradients prove need changing.
//
//   2. RankAdapterEngine         — Transcender-native low-rank delta adapters
//                                   (QUANT-Rank): per-layer LOW-RANK delta
//                                   ΔW ≈ B·A trained with autograd and
//                                   stored quantized in QUANT/QUANT formats.
//                                   100% native implementation — no external
//                                   adapter code of any kind is used.
//
//   3. KnowledgeExpansionEngine  — hallucination guard: NEW weight slots are
//                                   ADDED (never overwriting the base). Old
//                                   knowledge stays bit-identical (zero
//                                   catastrophic forgetting), new knowledge is
//                                   learned inside fresh slots. A confidence
//                                   gate decides when the slots may speak.
//
// All three are real training loops over the existing autograd / optimizer /
// QUANT-format stack. No stubs.
// ============================================================================

#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/model.h"
#include "quant/transformer.h"
#include "quant/trainer.h"
#include "quant/optimizer.h"
#include "quant/autograd.h"
#include "quant/format_registry.h"
#include "quant/ste_quantizer.h"
#include <vector>
#include <string>
#include <cstdint>

namespace quant {

// ============================================================================
// METHOD 1 — Selective fine-tuning
// ============================================================================
struct SelectiveTunerConfig {
    float learning_rate = 1e-5f;
    int num_epochs = 1;
    int warmup_steps = 10;
    int log_interval = 10;
    int save_interval = 100;
    int block_size = 256;            // weights per selection unit
    float max_grad_norm = 1.0f;      // global gradient clipping
    float saliency_sigma = 2.0f;     // blocks above mean + sigma*std selected
    float fisher_bias = 0.35f;       // Fisher prior weight inside saliency
    float fisher_decay = 0.95f;      // EMA decay when accumulating Fisher
    float min_select_fraction = 0.01f; // floor on the selected block fraction
    // STE roundtrip target: QUANT16 (FP16-class) by default. Coarser lattice
    // formats (e.g. QUANT8) quantize every selected block with ONE rms scale,
    // which collapses blocks that mix an outlier with small values — far too
    // lossy for trained weights. FP16 keeps the fine-tuned weights exactly
    // representable while preserving training quality.
    Format target_format = Format::Q16;
    bool ste_roundtrip = true;       // quantize selected blocks each step (STE)
    std::string output_path = "finetuned_selective.quant";
};

class SelectiveFineTuner {
public:
    struct Stats {
        float selected_fraction = 0.0f;
        int selected_blocks = 0;
        int total_blocks = 0;
        float last_loss = 0.0f;
        int steps = 0;
    };

    explicit SelectiveFineTuner(DenseModel* model);
    ~SelectiveFineTuner() = default;

    void configure(const SelectiveTunerConfig& cfg);
    void freeze_all(bool freeze);

    // Fisher importance of OLD knowledge (one anchor batch).
    void accumulate_fisher(const Tensor& input_ids, const Tensor& positions,
                           const Tensor& target_ids);
    void accumulate_fisher(DataLoader& dl, int max_batches);

    // Fine-tune on raw batches (loop `steps` times over the given batch).
    void fine_tune(const Tensor& input_ids, const Tensor& positions,
                   const Tensor& target_ids, int steps);
    // Fine-tune by pulling batches from a DataLoader.
    void fine_tune(DataLoader& dl, int max_batches);

    // Saliency-driven block selection helper (used internally, exposed for tests).
    // Returns one entry per weight block (true = update this block).
    static std::vector<bool> select_blocks(
        const float* grad, const float* fisher, int64_t n,
        int block_size, float sigma, float fisher_bias,
        float min_frac, float* threshold_out);

    // Quantized QUANT save of the updated model.
    void save(const std::string& path) const;

    const Stats& stats() const { return stats_; }
    const std::vector<std::vector<float>>& fisher_diag() const { return fisher_; }

private:
    DenseModel* model_;
    SelectiveTunerConfig cfg_;
    std::vector<Tensor*> params_;
    std::vector<std::vector<float>> fisher_;   // per-param Fisher diagonal
    // Adafactor drives the actual parameter updates; opt_views_ holds the
    // per-parameter 2-D views registered with it (they alias each model
    // weight, so the step lands on the real weights). Block selection is
    // implemented by zeroing the gradients of non-selected blocks before
    // optimizer_.step().
    Adafactor optimizer_;
    std::vector<Tensor> opt_views_;
    Stats stats_;
    int step_ = 0;

    float lr_at(int step) const;
    void ste_roundtrip_block(float* data, int64_t n, Format fmt);
};

// ============================================================================
// METHOD 2 — Transcender-native low-rank delta adapters ("QUANT-Rank")
// ============================================================================
struct RankAdapterConfig {
    int rank = 16;                     // adapter rank (width of the delta path)
    float alpha = 1.0f;                // scaling = alpha / rank, folded into A
    float learning_rate = 3e-4f;
    float weight_decay = 0.0f;
    int warmup_steps = 10;
    Format factor_format = Format::Q8; // deployment quantization of A and B
    std::string output_path = "adapters.nrad";
    // DoRA (weight-decomposed low-rank adaptation): keep a trainable per-output
    // magnitude vector m and renormalise the adapted weight to it —
    //   W' = m ⊙ (W0 + B·A) / ||W0 + B·A||_c
    // with ||·||_c the L2 norm down each output column. m is initialised to
    // ||W0||_c, and freezing m there makes W' collapse to plain LoRA exactly.
    bool use_dora = false;
};

class RankAdapterEngine {
public:
    struct LayerAdapter {
        int64_t layer_index = -1;
        int64_t in_dim = 0;    // input width of the adapted projection
        int64_t out_dim = 0;   // output width of the adapted projection
        int64_t rank = 0;
        Tensor A;              // {rank, in_dim}, pre-scaled by alpha/rank
        Tensor B;              // {out_dim, rank}, init 0 => zero initial delta
        // DoRA only: {out_dim, 1} magnitude vector, initialised to ||W0[:,o]||_c
        // (the column norms of the frozen base). Empty when use_dora is false.
        Tensor magnitude;
        std::string name;
    };

    explicit RankAdapterEngine(DenseModel* model);
    ~RankAdapterEngine() = default;

    void configure(const RankAdapterConfig& cfg);
    // Build one adapter per layer (delta on the FFN down-projection) and
    // freeze every base weight. Base is never modified during training.
    void init_adapters();
    void freeze_base(bool freeze);

    // Differentiable forward with adapters active (base frozen).
    Tensor forward_with_adapters(const Tensor& input_ids, const Tensor& positions,
                                 KVCache* cache = nullptr);

    // One training step on one batch. Optimizes adapter factors only.
    void train_step(const Tensor& input_ids, const Tensor& positions,
                    const Tensor& target_ids);
    void clear_grads();

    // Fold the deltas into the base down-projections. Plain LoRA: W' = W0 + ΔW.
    // DoRA: W' = m ⊙ (W0 + ΔW) / ||W0 + ΔW||_c, so every merged column has norm
    // exactly m[o] by construction.
    void merge_into_base();

    // DoRA magnitude support -------------------------------------------------
    bool dora_enabled() const { return cfg_.use_dora; }
    // Number of trainable magnitude scalars (0 unless use_dora).
    int64_t dora_param_count() const;
    // Exact DoRA magnitude gradient step. `dL_dW` holds one gradient tensor per
    // adapter, in the same {in_dim, out_dim} layout as the merged base weight
    // (position k*out_dim + o), i.e. dL/dW' for W' = m ⊙ V / ||V||_c with V and
    // ||V||_c held fixed:
    //     dL/dm[o] = Σ_k  dL/dW'[k,o] · V[k,o] / ||V[:,o]||
    // Exposed as an explicit call rather than folded into the autograd graph
    // because the engine has no norm/div op to carry it through. Layers whose
    // gradient is absent or the wrong size are skipped, not guessed at.
    void magnitude_step(const std::vector<Tensor>& dL_dW, float lr);

    // Quantized adapter store (.nrad): A/B saved in cfg.factor_format.
    void save(const std::string& path) const;
    bool load(const std::string& path);

    int64_t adapter_param_count() const;
    float last_loss() const { return last_loss_; }
    int step() const { return step_; }
    const std::vector<LayerAdapter>& adapters() const { return adapters_; }

private:
    DenseModel* model_;
    RankAdapterConfig cfg_;
    std::vector<LayerAdapter> adapters_;
    Adafactor optimizer_;
    float last_loss_ = 0.0f;
    int step_ = 0;

    Tensor block_forward(int64_t layer_idx, const Tensor& x,
                         const Tensor& positions, const Tensor& mask,
                         KVCache& cache) const;
    Tensor ffn_forward(int64_t layer_idx, const Tensor& x) const;
};

// ============================================================================
// METHOD 3 — Knowledge expansion (new weights, no forgetting, guard)
// ============================================================================
struct KnowledgeSlot {
    std::string name;
    int64_t hidden_size = 0;
    int64_t width = 0;
    Tensor w_in;    // {width, hidden}
    Tensor w_out;   // {hidden, width}; gate~0 makes the slot silent at init
    Tensor gate;    // {1} fixed scale (1.0 => tanh ~ 0.76), not trained
    float usage = 0.0f;
    float avg_confidence = 0.0f;
};

struct KnowledgeExpansionConfig {
    int64_t slot_width = 256;
    float learning_rate = 3e-4f;
    int warmup_steps = 10;
    float confidence_threshold = 0.35f; // base confidence below => consult slots
    float slot_gate_threshold = 0.5f;   // normalized slot gate activation floor
    std::string output_path = "knowledge_slots.kexp";
};

class KnowledgeExpansionEngine {
public:
    explicit KnowledgeExpansionEngine(DenseModel* model);
    ~KnowledgeExpansionEngine() = default;

    void configure(const KnowledgeExpansionConfig& cfg);
    void freeze_base(bool freeze);

    // Add a fresh weight slot for new knowledge. Returns the slot index.
    int64_t add_slot(const std::string& name, int64_t width = 0);

    // Forward with slot corrections applied to the final hidden state.
    Tensor forward_with_slots(const Tensor& input_ids, const Tensor& positions,
                              KVCache* cache = nullptr);

    // Hallucination guard: per-token it trusts the base when the base is
    // confident, otherwise the slots may correct. Optionally returns the
    // per-token confidence and a "slot consulted" mask.
    Tensor forward_guarded(const Tensor& input_ids, const Tensor& positions,
                           KVCache* cache = nullptr, Tensor* confidence_out = nullptr,
                           Tensor* slot_used_out = nullptr);

    // Train one step on one batch. Slot weights only; base untouched.
    void train_step(const Tensor& input_ids, const Tensor& positions,
                    const Tensor& target_ids);
    void clear_grads();

    void save(const std::string& path) const;
    bool load(const std::string& path);

    int64_t slot_param_count() const;
    float last_loss() const { return last_loss_; }
    int step() const { return step_; }
    size_t slot_count() const { return slots_.size(); }
    const std::vector<KnowledgeSlot>& slots() const { return slots_; }

private:
    DenseModel* model_;
    KnowledgeExpansionConfig cfg_;
    std::vector<KnowledgeSlot> slots_;
    Adafactor optimizer_;
    float last_loss_ = 0.0f;
    int step_ = 0;
    size_t opt_registered_ = 0;  // slots whose params are already in optimizer_

    // Replicated forward through the (frozen) base up to the final norm.
    Tensor extract_hidden(const Tensor& input_ids, const Tensor& positions,
                          KVCache* cache) const;
    Tensor apply_slots(const Tensor& h) const;
};

} // namespace quant
