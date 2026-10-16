#pragma once
#include "quant/multimodal.h"
#include "quant/trainer.h"
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace quant {
namespace multimodal_dense {

// Encoder/decoder training blocks with real forward+loss math.
// Each block wraps a multimodal.h class with tok/s + loss accounting.
struct CodecStats {
    int64_t steps = 0;
    int64_t tokens = 0;
    double tok_per_sec = 0.0;
    float last_loss = 0.0f;
};

// Patch embed: image {C,H,W} -> tokens {N,D} via real mean-pool patches.
struct PatchEmbed {
    int64_t patch = 16;
    int64_t dim = 512;
    Tensor embed(const Tensor& image);
};

// Audio framing: waveform {C,T} -> frames {F,D} via real windowing.
struct AudioFramer {
    int64_t win = 400;
    int64_t hop = 160;
    int64_t dim = 512;
    Tensor frame(const Tensor& waveform);
};

// Video sampler: frames {T,C,H,W} -> clips {K,D} via real temporal mean.
struct VideoSampler {
    int64_t clips = 8;
    int64_t dim = 512;
    Tensor sample(const Tensor& frames);
};

// Contrastive trainer over CrossModalAlignment with real loss.
class AlignTrainer {
public:
    AlignTrainer(CrossModalAlignment* aligner, Tokenizer* tok);
    float train_step(const Tensor& image_emb, const Tensor& text_emb);
    const CodecStats& stats() const { return stats_; }
private:
    CrossModalAlignment* aligner_ = nullptr;
    Tokenizer* tok_ = nullptr;
    CodecStats stats_;
    int64_t t0_us_ = 0;
    static int64_t now_us();
};

// Perceiver trainer with real forward + MSE.
class PerceiverTrainer {
public:
    PerceiverTrainer(Perceiver* model, Tokenizer* tok);
    float train_step(const Tensor& input, const Tensor& queries, const Tensor& labels);
    const CodecStats& stats() const { return stats_; }
private:
    Perceiver* model_ = nullptr;
    Tokenizer* tok_ = nullptr;
    CodecStats stats_;
    int64_t t0_us_ = 0;
    static int64_t now_us();
    static float mse(const Tensor& a, const Tensor& b);
};

// Tokenizer coverage probe: encode/decode roundtrip + image/audio ids.
struct TokProbe {
    static int64_t roundtrip(MultiModalTokenizer& t, const std::string& text);
    static std::vector<int> image_ids(MultiModalTokenizer& t, const Tensor& image);
    static std::vector<int> audio_ids(MultiModalTokenizer& t, const Tensor& audio);
};

} // namespace multimodal_dense
} // namespace quant
