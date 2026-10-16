#include "codecs.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace quant {
namespace multimodal_dense {

static int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

Tensor PatchEmbed::embed(const Tensor& image) {
    if (image.rank() != 3) throw std::runtime_error("[codecs] image needs {C,H,W}");
    int64_t C = image.dim(0), H = image.dim(1), W = image.dim(2);
    if (H % patch != 0 || W % patch != 0) throw std::runtime_error("[codecs] image not patch-aligned");
    int64_t gh = H / patch, gw = W / patch, N = gh * gw;
    const float* im = image.data<float>();
    Tensor o = Tensor::zeros({N, dim});
    float* oo = o.data<float>();
    for (int64_t n = 0; n < N; n++) {
        int64_t py = (n / gw) * patch, px = (n % gw) * patch;
        for (int64_t d = 0; d < dim; d++) {
            double acc = 0.0;
            int64_t cnt = 0;
            for (int64_t y = 0; y < patch; y++) {
                for (int64_t x = 0; x < patch; x++) {
                    int64_t c = d % C;
                    acc += im[c * H * W + (py + y) * W + (px + x)];
                    cnt++;
                }
            }
            oo[n * dim + d] = (float)(acc / (double)cnt);
        }
    }
    return o;
}

Tensor AudioFramer::frame(const Tensor& waveform) {
    if (waveform.rank() != 2) throw std::runtime_error("[codecs] waveform needs {C,T}");
    int64_t C = waveform.dim(0), T = waveform.dim(1);
    if (win <= 0 || hop <= 0) throw std::runtime_error("[codecs] bad win/hop");
    int64_t F = (T - win) / hop + 1;
    if (F <= 0) throw std::runtime_error("[codecs] waveform too short");
    const float* w = waveform.data<float>();
    Tensor o = Tensor::zeros({F, dim});
    float* oo = o.data<float>();
    for (int64_t f = 0; f < F; f++) {
        for (int64_t d = 0; d < dim; d++) {
            double acc = 0.0;
            for (int64_t k = 0; k < win; k++) {
                int64_t t = f * hop + k;
                acc += w[(d % C) * T + t] * (0.54 - 0.46 * std::cos(2.0 * 3.14159265 * k / (win - 1)));
            }
            oo[f * dim + d] = (float)(acc / (double)win);
        }
    }
    return o;
}

Tensor VideoSampler::sample(const Tensor& frames) {
    if (frames.rank() != 4) throw std::runtime_error("[codecs] frames need {T,C,H,W}");
    int64_t T = frames.dim(0), C = frames.dim(1), H = frames.dim(2), W = frames.dim(3);
    if (clips <= 0 || clips > T) throw std::runtime_error("[codecs] bad clips");
    const float* f = frames.data<float>();
    Tensor o = Tensor::zeros({clips, dim});
    float* oo = o.data<float>();
    int64_t per = T / clips;
    for (int64_t k = 0; k < clips; k++) {
        for (int64_t d = 0; d < dim; d++) {
            double acc = 0.0;
            int64_t cnt = 0;
            for (int64_t t = k * per; t < (k + 1) * per; t++) {
                int64_t c = d % C;
                for (int64_t y = 0; y < H; y++)
                    for (int64_t x = 0; x < W; x++) {
                        acc += f[t * C * H * W + c * H * W + y * W + x];
                        cnt++;
                    }
            }
            oo[k * dim + d] = (float)(acc / (double)cnt);
        }
    }
    return o;
}

int64_t AlignTrainer::now_us() { return multimodal_dense::now_us(); }
int64_t PerceiverTrainer::now_us() { return multimodal_dense::now_us(); }

AlignTrainer::AlignTrainer(CrossModalAlignment* aligner, Tokenizer* tok)
    : aligner_(aligner), tok_(tok) {
    if (!aligner_ || !tok_) throw std::runtime_error("[codecs] AlignTrainer needs aligner+tok");
    t0_us_ = now_us();
}

float AlignTrainer::train_step(const Tensor& image_emb, const Tensor& text_emb) {
    float loss = aligner_->contrastive_loss(image_emb, text_emb);
    stats_.steps++;
    stats_.tokens += image_emb.numel();
    stats_.last_loss = loss;
    int64_t dt = now_us() - t0_us_;
    if (dt > 0) stats_.tok_per_sec = (double)stats_.tokens * 1e6 / (double)dt;
    return loss;
}

PerceiverTrainer::PerceiverTrainer(Perceiver* model, Tokenizer* tok)
    : model_(model), tok_(tok) {
    if (!model_ || !tok_) throw std::runtime_error("[codecs] PerceiverTrainer needs model+tok");
    t0_us_ = now_us();
}

float PerceiverTrainer::mse(const Tensor& a, const Tensor& b) {
    int64_t n = std::min(a.numel(), b.numel());
    if (n <= 0) return 0.0f;
    const float* x = a.data<float>();
    const float* y = b.data<float>();
    double se = 0.0;
    for (int64_t i = 0; i < n; i++) { double d = (double)x[i] - y[i]; se += d * d; }
    return (float)(se / (double)n);
}

float PerceiverTrainer::train_step(const Tensor& input, const Tensor& queries, const Tensor& labels) {
    Tensor out = model_->forward(input, queries);
    float loss = mse(out, labels);
    stats_.steps++;
    stats_.tokens += input.numel();
    stats_.last_loss = loss;
    int64_t dt = now_us() - t0_us_;
    if (dt > 0) stats_.tok_per_sec = (double)stats_.tokens * 1e6 / (double)dt;
    return loss;
}

int64_t TokProbe::roundtrip(MultiModalTokenizer& t, const std::string& text) {
    std::vector<int> ids = t.encode(text);
    std::string back = t.decode(ids);
    std::vector<int> ids2 = t.encode(back);
    int64_t match = 0;
    for (size_t i = 0; i < std::min(ids.size(), ids2.size()); i++)
        if (ids[i] == ids2[i]) match++;
    return match;
}

std::vector<int> TokProbe::image_ids(MultiModalTokenizer& t, const Tensor& image) {
    return t.encode_image(image);
}

std::vector<int> TokProbe::audio_ids(MultiModalTokenizer& t, const Tensor& audio) {
    return t.encode_audio(audio);
}

} // namespace multimodal_dense
} // namespace quant
