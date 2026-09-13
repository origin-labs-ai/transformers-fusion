#include "quant/multimodal.h"
#include "quant/math.h"
#include "quant/tokenizer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace quant {

// ============================================================================
// H1: Cross-modal attention — proper scaled dot-product cross-attention
// ============================================================================
CrossModalAttention::CrossModalAttention(int64_t hidden) : hidden_(hidden) {}

Tensor CrossModalAttention::forward(const std::vector<Tensor>& modalities) {
    if (modalities.empty())
        return Tensor({hidden_});
    if (modalities.size() == 1)
        return modalities[0].clone();

    int64_t M = (int64_t)modalities.size();

    for (int64_t m = 0; m < M; m++) {
        QUANT_CHECK(modalities[m].numel() % hidden_ == 0,
            "CrossModalAttention: modality " + std::to_string(m) +
            " numel not divisible by hidden");
    }

    std::vector<Tensor> attended;
    attended.reserve(M);

    for (int64_t src = 0; src < M; src++) {
        const Tensor& src_mod = modalities[src];
        int64_t src_seq = src_mod.numel() / hidden_;
        Tensor src_flat = src_mod.reshape({src_seq, hidden_});
        const float* sq = src_flat.data<float>();

        struct TargetView { const float* data; int64_t seq; };
        std::vector<TargetView> targets;
        targets.reserve(M - 1);
        for (int64_t tgt = 0; tgt < M; tgt++) {
            if (tgt == src) continue;
            const Tensor& tgt_mod = modalities[tgt];
            int64_t tgt_seq = tgt_mod.numel() / hidden_;
            Tensor tgt_flat = tgt_mod.reshape({tgt_seq, hidden_});
            targets.push_back({tgt_flat.data<float>(), tgt_seq});
        }

        Tensor result({src_seq, hidden_});
        result.zero_();
        float* rd = result.data<float>();
        float inv_sqrt_d = 1.0f / std::sqrt((float)hidden_);

        for (int64_t s = 0; s < src_seq; s++) {
            const float* q = sq + s * hidden_;
            float total_weight = 0.0f;
            std::vector<float> ctx(hidden_, 0.0f);

            for (const auto& tgt : targets) {
                const float* kv = tgt.data;
                int64_t t_seq = tgt.seq;

                float max_score = -1e30f;
                std::vector<float> scores((size_t)t_seq);
                for (int64_t t = 0; t < t_seq; t++) {
                    float dot = 0;
                    const float* kv_t = kv + t * hidden_;
                    for (int64_t d = 0; d < hidden_; d++)
                        dot += q[d] * kv_t[d];
                    dot *= inv_sqrt_d;
                    scores[(size_t)t] = dot;
                    if (dot > max_score) max_score = dot;
                }

                float sum_exp = 0;
                for (int64_t t = 0; t < t_seq; t++) {
                    scores[(size_t)t] = std::exp(scores[(size_t)t] - max_score);
                    sum_exp += scores[(size_t)t];
                }

                if (sum_exp > 0) {
                    float inv_sum = 1.0f / sum_exp;
                    for (int64_t t = 0; t < t_seq; t++) {
                        float attn = scores[(size_t)t] * inv_sum;
                        total_weight += attn;
                        const float* kv_t = kv + t * hidden_;
                        for (int64_t d = 0; d < hidden_; d++)
                            ctx[(size_t)d] += attn * kv_t[d];
                    }
                }
            }

            if (total_weight > 0) {
                float norm = 1.0f / total_weight;
                for (int64_t d = 0; d < hidden_; d++)
                    rd[s * hidden_ + d] = ctx[(size_t)d] * norm;
            }
        }
        attended.push_back(result);
    }

    // Combine: modalities may have different seq lengths (e.g. text 16 vs
    // image 8 vs audio 12), so elementwise averaging over attended[0]'s shape
    // would read out-of-bounds. Mean-pool each modality over its seq dim,
    // then average across modalities. Returns {hidden_}. Fixes flaky NaN.
    Tensor combined({hidden_});
    combined.zero_();
    float* cd = combined.data<float>();
    for (int64_t m = 0; m < M; m++) {
        const float* ad = attended[(size_t)m].data<float>();
        int64_t seq_m = attended[(size_t)m].numel() / hidden_;
        if (seq_m <= 0) continue;
        for (int64_t s = 0; s < seq_m; s++)
            for (int64_t d = 0; d < hidden_; d++)
                cd[d] += ad[s * hidden_ + d] / ((float)seq_m * (float)M);
    }

    return combined;
}

// ============================================================================
// H2: Joint multi-modal model
// ============================================================================
JointMultimodalModel::JointMultimodalModel(Model* text_encoder, int64_t hidden)
    : text_encoder_(text_encoder), cross_attn_(hidden) {
    // D6 W7: differentiate null-encoder (throw) from empty-input (empty out).
    if (text_encoder_ == nullptr) {
        std::cerr << "[JointMultimodalModel] null text_encoder" << std::endl;
        throw std::invalid_argument("JointMultimodalModel: null text_encoder");
    }
}

Tensor JointMultimodalModel::forward(const Tensor& text, const Tensor& image, const Tensor& audio) {
    std::vector<Tensor> mods;
    if (text.numel() > 0) mods.push_back(text);
    if (image.numel() > 0) mods.push_back(image);
    if (audio.numel() > 0) mods.push_back(audio);
    return cross_attn_.forward(mods);
}

// ============================================================================
// H3: ImageNet classifier
// ============================================================================
ImageNetClassifier::ImageNetClassifier(Model* vit, int64_t n_classes)
    : vit_(vit), classifier_head_(Tensor({n_classes, 768})) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0, 0.02f);
    float* w = classifier_head_.data<float>();
    for (int64_t i = 0; i < n_classes * 768; i++)
        w[i] = dist(rng);
}

Tensor ImageNetClassifier::classify(const Tensor& image) {
    // D6 W7: null encoder throws; empty input returns empty (never fallback
    // zeros that masquerade as plausible logits). Modes logged distinctly.
    if (vit_ == nullptr) {
        std::cerr << "[ImageNetClassifier] null vit encoder" << std::endl;
        throw std::invalid_argument("ImageNetClassifier: null vit encoder");
    }
    if (image.numel() == 0) {
        std::cerr << "[ImageNetClassifier] empty image input" << std::endl;
        return Tensor({0});
    }
    if (image.rank() < 1) {
        std::cerr << "[ImageNetClassifier] malformed image rank" << std::endl;
        return Tensor({0});
    }

    Tensor feat = vit_->forward(image, Tensor({1, 1}));
    if (feat.numel() == 0) {
        std::cerr << "[ImageNetClassifier] feature extraction failed" << std::endl;
        return Tensor({0});
    }
    int64_t B = image.dim(0);
    if (B <= 0) {
        std::cerr << "[ImageNetClassifier] invalid batch dim" << std::endl;
        return Tensor({0});
    }
    int64_t n_classes = classifier_head_.dim(0);
    int64_t feat_dim = feat.numel() / B;
    Tensor feat_flat = (feat_dim != 768) ? feat.reshape({B, 768}) : feat;

    Tensor logits({B, n_classes});
    const float* w = classifier_head_.data<float>();
    const float* f = feat_flat.data<float>();
    float* ld = logits.data<float>();
    for (int64_t b = 0; b < B; b++)
        for (int64_t c = 0; c < n_classes; c++) {
            float sum = 0;
            for (int64_t d = 0; d < 768; d++)
                sum += f[b * 768 + d] * w[c * 768 + d];
            ld[b * n_classes + c] = sum;
        }
    return logits;
}

float ImageNetClassifier::evaluate(DataLoader& val_dl) {
    int64_t correct = 0, total = 0;
    Tensor inputs, labels;
    while (val_dl.next_batch(inputs, labels)) {
        if (inputs.numel() == 0) continue;
        Tensor logits = classify(inputs);
        const float* ld = logits.data<float>();
        int64_t B = logits.dim(0);
        int64_t C = logits.dim(1);
        const float* lb = labels.data<float>();
        for (int64_t b = 0; b < B; b++) {
            int64_t pred = 0;
            for (int64_t c = 1; c < C; c++)
                if (ld[b * C + c] > ld[b * C + pred]) pred = c;
            if (std::abs((float)pred - lb[b]) < 0.5f) correct++;
            total++;
        }
    }
    return total > 0 ? (float)correct / (float)total : 0.0f;
}

// ============================================================================
// H4: Speech recognition with CTC decoding
// ============================================================================
SpeechRecognizer::SpeechRecognizer(Model* audio_encoder)
    : audio_encoder_(audio_encoder) {
}

std::string SpeechRecognizer::transcribe(const Tensor& audio) {
    // D6 W7: null encoder throws; empty input returns empty (never a
    // plausible-English constant); feature/decode failures return empty with
    // a distinct log per mode.
    if (audio_encoder_ == nullptr) {
        std::cerr << "[SpeechRecognizer] null audio encoder" << std::endl;
        throw std::invalid_argument("SpeechRecognizer: null audio_encoder");
    }
    if (audio.numel() == 0) {
        std::cerr << "[SpeechRecognizer] empty audio input" << std::endl;
        return "";
    }
    Tensor feat = audio_encoder_->forward(audio, Tensor({1, 1}));
    if (feat.numel() == 0) {
        std::cerr << "[SpeechRecognizer] feature extraction failed" << std::endl;
        return "";
    }
    if (feat.rank() < 2) {
        std::cerr << "[SpeechRecognizer] feature decode failure: rank<2" << std::endl;
        return "";
    }
    int64_t T = feat.dim(feat.rank() - 2);
    int64_t V = feat.dim(feat.rank() - 1);
    if (T <= 0 || V <= 0) {
        std::cerr << "[SpeechRecognizer] feature decode failure: bad T/V" << std::endl;
        return "";
    }

    const float* fd = feat.data<float>();
    if (fd == nullptr) {
        std::cerr << "[SpeechRecognizer] null feature data" << std::endl;
        return "";
    }
    std::string result;
    int prev = -1;
    for (int64_t t = 0; t < T; t++) {
        int max_c = 0;
        float max_val = fd[t * V];
        for (int64_t v = 1; v < V; v++)
            if (fd[t * V + v] > max_val) { max_val = fd[t * V + v]; max_c = (int)v; }
        if (max_c == 0) { prev = -1; continue; }
        if (max_c != prev) {
            if (max_c - 1 < 26)
                result += (char)('a' + max_c - 1);
            else if (max_c - 1 < 36)
                result += (char)('0' + max_c - 27);
            else if (max_c - 1 == 36)
                result += ' ';
        }
        prev = max_c;
    }
    if (result.empty()) {
        std::cerr << "[SpeechRecognizer] decode produced no tokens" << std::endl;
    }
    return result;
}

// ============================================================================
// H5: OCR pipeline
// ============================================================================
OCRPipeline::OCRPipeline(Model* ocr_encoder)
    : ocr_encoder_(ocr_encoder) {
}

std::string OCRPipeline::recognize(const Tensor& image) {
    // D6 W7: null encoder throws; empty input returns empty (never a
    // plausible-English constant); feature/decode failures return empty with
    // a distinct log per mode.
    if (ocr_encoder_ == nullptr) {
        std::cerr << "[OCRPipeline] null ocr encoder" << std::endl;
        throw std::invalid_argument("OCRPipeline: null ocr_encoder");
    }
    if (image.numel() == 0) {
        std::cerr << "[OCRPipeline] empty image input" << std::endl;
        return "";
    }
    Tensor feat = ocr_encoder_->forward(image, Tensor({1, 1}));
    if (feat.numel() == 0) {
        std::cerr << "[OCRPipeline] feature extraction failed" << std::endl;
        return "";
    }
    if (feat.rank() < 2) {
        std::cerr << "[OCRPipeline] feature decode failure: rank<2" << std::endl;
        return "";
    }
    int64_t S = feat.dim(feat.rank() - 2);
    int64_t V = feat.dim(feat.rank() - 1);
    if (S <= 0 || V <= 0) {
        std::cerr << "[OCRPipeline] feature decode failure: bad S/V" << std::endl;
        return "";
    }

    const float* fd = feat.data<float>();
    if (fd == nullptr) {
        std::cerr << "[OCRPipeline] null feature data" << std::endl;
        return "";
    }
    std::string result;
    int prev = -1;
    for (int64_t s = 0; s < S; s++) {
        int max_c = 0;
        float max_val = fd[s * V];
        for (int64_t v = 1; v < V; v++)
            if (fd[s * V + v] > max_val) { max_val = fd[s * V + v]; max_c = (int)v; }
        if (max_c == prev || max_c == 0) continue;
        prev = max_c;
        if (max_c > 0 && max_c < 128)
            result += (char)max_c;
    }
    if (result.empty()) {
        std::cerr << "[OCRPipeline] decode produced no tokens" << std::endl;
    }
    return result;
}

// ============================================================================
// H6: Video understanding
// ============================================================================
VideoUnderstanding::VideoUnderstanding(Model* video_encoder)
    : video_encoder_(video_encoder) {
}

std::string VideoUnderstanding::describe(const Tensor& video_frames) {
    // D6 W7: null encoder throws; empty input returns empty (never a
    // plausible-English bracket constant); feature/decode failures return
    // empty with a distinct log per mode.
    if (video_encoder_ == nullptr) {
        std::cerr << "[VideoUnderstanding] null video encoder" << std::endl;
        throw std::invalid_argument("VideoUnderstanding: null video_encoder");
    }
    if (video_frames.numel() == 0) {
        std::cerr << "[VideoUnderstanding] empty video input" << std::endl;
        return "";
    }
    Tensor feat = video_encoder_->forward(video_frames, Tensor({1, 1}));
    if (feat.numel() == 0) {
        std::cerr << "[VideoUnderstanding] feature extraction failed" << std::endl;
        return "";
    }
    if (feat.rank() < 2) {
        std::cerr << "[VideoUnderstanding] feature decode failure: rank<2" << std::endl;
        return "";
    }

    int64_t last_dim = feat.dim(feat.rank() - 1);
    if (last_dim <= 0) {
        std::cerr << "[VideoUnderstanding] feature decode failure: bad last_dim" << std::endl;
        return "";
    }
    int64_t seq_len = feat.numel() / last_dim;
    if (seq_len <= 0) {
        std::cerr << "[VideoUnderstanding] feature decode failure: bad seq_len" << std::endl;
        return "";
    }
    const float* fd = feat.data<float>();
    if (fd == nullptr) {
        std::cerr << "[VideoUnderstanding] null feature data" << std::endl;
        return "";
    }
    std::string result;
    int prev = -1;

    for (int64_t t = 0; t < seq_len; t++) {
        int max_c = 0;
        float max_val = fd[t * last_dim];
        for (int64_t d = 1; d < std::min(last_dim, (int64_t)200); d++)
            if (fd[t * last_dim + d] > max_val) { max_val = fd[t * last_dim + d]; max_c = (int)d; }
        if (max_c > 0 && max_c < 128 && max_c != prev)
            result += (char)max_c;
        prev = max_c;
        if (result.size() > 128) break;
    }

    if (result.empty()) {
        std::cerr << "[VideoUnderstanding] decode produced no tokens" << std::endl;
    }
    return result;
}

// ============================================================================
// H7: Image captioning
// ============================================================================
ImageCaptioning::ImageCaptioning(Model* vision_encoder, Model* text_decoder)
    : vision_encoder_(vision_encoder), text_decoder_(text_decoder) {
}

std::string ImageCaptioning::caption(const Tensor& image, int max_tokens) {
    // D6 W7: null vision/text handles throw distinctly; empty image returns
    // empty (never a plausible-English constant); vocab/decode failures empty.
    if (vision_encoder_ == nullptr) {
        std::cerr << "[ImageCaptioning] null vision encoder" << std::endl;
        throw std::invalid_argument("ImageCaptioning: null vision_encoder");
    }
    if (text_decoder_ == nullptr) {
        std::cerr << "[ImageCaptioning] null text decoder" << std::endl;
        throw std::invalid_argument("ImageCaptioning: null text_decoder");
    }
    if (image.numel() == 0) {
        std::cerr << "[ImageCaptioning] empty image input" << std::endl;
        return "";
    }
    if (max_tokens <= 0) max_tokens = 32;

    Tensor image_feat = vision_encoder_->forward(image, Tensor({1, 1}));
    if (image_feat.numel() == 0) {
        std::cerr << "[ImageCaptioning] feature extraction failed" << std::endl;
        return "";
    }

    static BPETokenizer bpe;
    std::vector<int> tokens = {bpe.bos_id()};
    int64_t V = text_decoder_->vocab_size();
    if (V <= 0) {
        std::cerr << "[ImageCaptioning] invalid vocab size" << std::endl;
        return "";
    }

    {
        int64_t img_len = std::min(image_feat.numel(), (int64_t)64);
        const float* img_data = image_feat.data<float>();
        if (img_data == nullptr) {
            std::cerr << "[ImageCaptioning] null feature data" << std::endl;
            return "";
        }
        for (int64_t i = 0; i < img_len; i++)
            tokens.push_back((int)(std::fabs(img_data[i]) * 1000) % std::max((int64_t)1, V));
    }

    for (int i = 0; i < max_tokens; i++) {
        Tensor input({1, (int64_t)tokens.size()});
        float* id = input.data<float>();
        if (id == nullptr) {
            std::cerr << "[ImageCaptioning] input alloc failed" << std::endl;
            return "";
        }
        for (size_t t = 0; t < tokens.size(); t++)
            id[t] = (float)tokens[t];

        Tensor logits = text_decoder_->forward(input, Tensor({1, (int64_t)tokens.size()}));
        int64_t numel = logits.numel();
        if (numel < V) {
            std::cerr << "[ImageCaptioning] logits shorter than vocab" << std::endl;
            return "";
        }
        const float* lbase = logits.data<float>();
        if (lbase == nullptr) {
            std::cerr << "[ImageCaptioning] null logits data" << std::endl;
            return "";
        }
        const float* ld = lbase + numel - V;

        int next = 0;
        float max_val = ld[0];
        for (int64_t v = 1; v < V; v++)
            if (ld[v] > max_val) { max_val = ld[v]; next = (int)v; }

        if (next == bpe.eos_id()) break;
        tokens.push_back(next);
    }

    return bpe.decode(tokens);
}

// ============================================================================
// H8: Visual QA
// ============================================================================
VisualQA::VisualQA(Model* vision_encoder, Model* text_decoder)
    : vision_encoder_(vision_encoder), text_decoder_(text_decoder) {
}

std::string VisualQA::answer(const Tensor& image, const std::string& question) {
    // D6 W7: null vision/text handles throw distinctly; empty image/question
    // returns empty (never a plausible-English constant); decode failures empty.
    if (vision_encoder_ == nullptr) {
        std::cerr << "[VisualQA] null vision encoder" << std::endl;
        throw std::invalid_argument("VisualQA: null vision_encoder");
    }
    if (text_decoder_ == nullptr) {
        std::cerr << "[VisualQA] null text decoder" << std::endl;
        throw std::invalid_argument("VisualQA: null text_decoder");
    }
    if (image.numel() == 0) {
        std::cerr << "[VisualQA] empty image input" << std::endl;
        return "";
    }
    if (question.empty()) {
        std::cerr << "[VisualQA] empty question input" << std::endl;
        return "";
    }

    Tensor image_feat = vision_encoder_->forward(image, Tensor({1, 1}));
    if (image_feat.numel() == 0) {
        std::cerr << "[VisualQA] feature extraction failed" << std::endl;
        return "";
    }

    static BPETokenizer bpe;
    std::vector<int> q_tokens = bpe.encode(question);
    std::vector<int> tokens = {bpe.bos_id()};

    int64_t img_prefix_len = 0;
    {
        int64_t img_len = std::min(image_feat.numel(), (int64_t)64);
        const float* img_data = image_feat.data<float>();
        if (img_data == nullptr) {
            std::cerr << "[VisualQA] null feature data" << std::endl;
            return "";
        }
        int64_t V = text_decoder_->vocab_size();
        if (V <= 0) {
            std::cerr << "[VisualQA] invalid vocab size" << std::endl;
            return "";
        }
        for (int64_t i = 0; i < img_len; i++)
            tokens.push_back((int)(std::fabs(img_data[i]) * 1000) % std::max((int64_t)1, V));
        img_prefix_len = img_len;
    }

    tokens.insert(tokens.end(), q_tokens.begin(), q_tokens.end());

    int64_t V = text_decoder_->vocab_size();
    if (V <= 0) {
        std::cerr << "[VisualQA] invalid vocab size" << std::endl;
        return "";
    }

    for (int i = 0; i < 64; i++) {
        Tensor input({1, (int64_t)tokens.size()});
        float* id = input.data<float>();
        if (id == nullptr) {
            std::cerr << "[VisualQA] input alloc failed" << std::endl;
            return "";
        }
        for (size_t t = 0; t < tokens.size(); t++)
            id[t] = (float)tokens[t];

        Tensor logits = text_decoder_->forward(input, Tensor({1, (int64_t)tokens.size()}));
        int64_t numel = logits.numel();
        if (numel < V) {
            std::cerr << "[VisualQA] logits shorter than vocab" << std::endl;
            return "";
        }
        const float* lbase = logits.data<float>();
        if (lbase == nullptr) {
            std::cerr << "[VisualQA] null logits data" << std::endl;
            return "";
        }
        const float* ld = lbase + numel - V;

        int next = 0;
        float max_val = ld[0];
        for (int64_t v = 1; v < V; v++)
            if (ld[v] > max_val) { max_val = ld[v]; next = (int)v; }

        if (next == bpe.eos_id()) break;
        tokens.push_back(next);
    }

    std::vector<int> answer_tokens(tokens.begin() + 1 + img_prefix_len + (int64_t)q_tokens.size(), tokens.end());
    return bpe.decode(answer_tokens);
}

// ============================================================================
// H9: Text-to-image — DDIM diffusion model
// ============================================================================
TextToImage::TextToImage(int64_t latent_dim, int64_t image_size)
    : latent_dim_(latent_dim), image_size_(image_size) {
    // Build UNet blocks
    int64_t chans[] = {64, 128, 256};
    for (int64_t ch : chans) {
        UNetBlock blk;
        blk.in_channels = ch;
        blk.out_channels = ch;
        blk.conv_weight = Tensor(Shape{ch, ch, 3, 3});
        blk.conv_bias = Tensor(Shape{ch});
        blk.bn_weight = Tensor(Shape{ch});
        blk.bn_bias = Tensor(Shape{ch});
        unet_.push_back(std::move(blk));
    }
}

Tensor TextToImage::generate(const std::string& prompt, int steps) {
    if (image_size_ <= 0 || latent_dim_ <= 0)
        return Tensor({3, 64, 64});
    if (steps <= 0) steps = 50;
    if (steps > 200) steps = 200;

    // Text embedding for conditioning
    std::vector<float> text_emb((size_t)latent_dim_, 0.0f);
    for (size_t i = 0; i < prompt.size(); i++)
        text_emb[i % text_emb.size()] += (float)(unsigned char)prompt[i] / 255.0f;
    float mean = 0;
    for (auto v : text_emb) mean += v;
    mean /= (float)text_emb.size();
    for (auto& v : text_emb) v -= mean;

    int64_t total_steps = std::min<int64_t>(steps, 100);
    int64_t H = std::max((int64_t)8, image_size_ / 8);
    int64_t W = std::max((int64_t)8, image_size_ / 8);
    int64_t spatial = H * W;
    int64_t num_channels = 3;

    std::mt19937 rng(42);
    Tensor latent({num_channels, H, W});
    float* ld = latent.data<float>();
    int64_t latent_size = latent.numel();
    for (int64_t i = 0; i < latent_size; i++)
        ld[i] = (float)rng() / (float)UINT32_MAX * 2.0f - 1.0f;

    std::vector<float> alphas_cumprod((size_t)total_steps + 1, 1.0f);
    for (int s = 0; s < total_steps; s++) {
        float t = (float)(s + 1) / (float)total_steps;
        float beta = 0.0001f + (0.02f - 0.0001f) * t;
        alphas_cumprod[(size_t)(s + 1)] = alphas_cumprod[(size_t)s] * (1.0f - beta);
    }

    for (int64_t s = total_steps - 1; s >= 0; s--) {
        float alpha_t = alphas_cumprod[(size_t)(s + 1)];
        float alpha_s = alphas_cumprod[(size_t)s];
        float sigma_t = std::sqrt(std::max(1.0f - alpha_t, 1e-8f));
        float sigma_s = std::sqrt(std::max(1.0f - alpha_s, 1e-8f));
        float sqrt_alpha_t = std::sqrt(std::max(alpha_t, 1e-8f));
        float sqrt_alpha_s = std::sqrt(std::max(alpha_s, 1e-8f));

        // UNet noise prediction via local smoothing (proxy for actual UNet)
        Tensor noise_pred({num_channels, H, W});
        float* nd = noise_pred.data<float>();
        for (int64_t c = 0; c < num_channels; c++) {
            for (int64_t i = 0; i < H; i++) {
                for (int64_t j = 0; j < W; j++) {
                    int64_t idx = c * spatial + i * W + j;
                    float local_sum = ld[idx];
                    int local_count = 1;
                    if (i > 0) { local_sum += ld[c * spatial + (i - 1) * W + j]; local_count++; }
                    if (i < H - 1) { local_sum += ld[c * spatial + (i + 1) * W + j]; local_count++; }
                    if (j > 0) { local_sum += ld[c * spatial + i * W + (j - 1)]; local_count++; }
                    if (j < W - 1) { local_sum += ld[c * spatial + i * W + (j + 1)]; local_count++; }
                    nd[idx] = (ld[idx] - local_sum / (float)local_count) * 0.5f;
                }
            }
        }

        // Text conditioning
        for (int64_t i = 0; i < latent_size; i++)
            ld[i] += text_emb[(size_t)(i % text_emb.size())] * 0.5f;

        // DDIM update
        std::vector<float> pred_x0((size_t)latent_size);
        for (int64_t i = 0; i < latent_size; i++)
            pred_x0[(size_t)i] = (ld[i] - sigma_t * nd[i]) / sqrt_alpha_t;
        for (int64_t i = 0; i < latent_size; i++)
            ld[i] = sqrt_alpha_s * pred_x0[(size_t)i] + sigma_s * nd[i];
    }

    // Decode latent to pixel space
    Tensor image({3, image_size_, image_size_});
    float* id = image.data<float>();
    for (int64_t c = 0; c < 3; c++)
        for (int64_t i = 0; i < image_size_; i++)
            for (int64_t j = 0; j < image_size_; j++) {
                int64_t li = c * spatial + (i * H / image_size_) * W + (j * W / image_size_);
                id[c * image_size_ * image_size_ + i * image_size_ + j] = std::tanh(ld[li]);
            }
    return image;
}

// ============================================================================
// H10: Audio synthesis — Griffin-Lim phase reconstruction
// ============================================================================
AudioSynthesizer::AudioSynthesizer() {}

Tensor AudioSynthesizer::synthesize(const Tensor& mel) {
    if (mel.rank() < 2 || mel.numel() == 0)
        return Tensor({0});

    int64_t n_mels = mel.dim(0);
    int64_t n_frames = mel.dim(1);
    if (n_mels <= 0 || n_frames <= 0)
        return Tensor({0});

    int64_t n_fft = 1024;
    int64_t hop = n_fft / 4;
    int64_t n_bins = n_fft / 2 + 1;
    int64_t n_samples = (n_frames - 1) * hop + n_fft;

    const float* md = mel.data<float>();

    // Griffin-Lim: iterative phase reconstruction
    Tensor waveform({n_samples});
    waveform.zero_();
    float* wd = waveform.data<float>();

    // Initial random phase
    std::mt19937 rng(42);
    Tensor phase({n_frames, n_bins});
    float* pd = phase.data<float>();
    for (int64_t i = 0; i < n_frames * n_bins; i++)
        pd[i] = (float)rng() / (float)UINT32_MAX * 2.0f * 3.14159265f;

    for (int iter = 0; iter < 8; iter++) {
        waveform.zero_();

        // ISTFT
        for (int64_t f = 0; f < n_frames; f++) {
            int64_t offset = f * hop;
            for (int64_t k = 0; k < n_bins; k++) {
                float mag = std::exp(std::min(md[f * n_mels + k % n_mels], 10.0f));
                float phase_val = pd[f * n_bins + k];
                float re = mag * std::cos(phase_val);
                float im = mag * std::sin(phase_val);
                float hann = 0.5f * (1.0f - std::cos(2.0f * 3.14159265f * (float)k / (float)(n_fft - 1)));
                for (int64_t n = 0; n < n_fft && offset + n < n_samples; n++) {
                    float angle = 2.0f * 3.14159265f * (float)k * (float)n / (float)n_fft;
                    float val = (re * std::cos(angle) - im * std::sin(angle)) * hann / (float)n_fft;
                    wd[offset + n] += val;
                }
            }
        }

        // STFT to update phase
        for (int64_t f = 0; f < n_frames; f++) {
            int64_t offset = f * hop;
            for (int64_t k = 0; k < n_bins; k++) {
                float re = 0, im = 0;
                float angle_k = 2.0f * 3.14159265f * (float)k / (float)n_fft;
                for (int64_t n = 0; n < n_fft && offset + n < n_samples; n++) {
                    float hann = 0.5f * (1.0f - std::cos(2.0f * 3.14159265f * (float)n / (float)(n_fft - 1)));
                    re += wd[offset + n] * hann * std::cos(angle_k * n);
                    im -= wd[offset + n] * hann * std::sin(angle_k * n);
                }
                pd[f * n_bins + k] = std::atan2(im, re);
            }
        }
    }

    return waveform;
}

// ============================================================================
// H11: Mel spectrogram
// ============================================================================
MelSpectrogram::MelSpectrogram(int sample_rate, int n_mels)
    : sample_rate_(sample_rate), n_mels_(n_mels) {
    QUANT_CHECK(sample_rate > 0, "MelSpectrogram: sample_rate must be positive");
    QUANT_CHECK(n_mels > 0, "MelSpectrogram: n_mels must be positive");
}

std::vector<float> MelSpectrogram::mel_filterbank(int n_filters, int fft_size) {
    int64_t n_bins = fft_size / 2 + 1;
    std::vector<float> fb((size_t)n_filters * (size_t)n_bins, 0.0f);
    float mel_max = 2595.0f * std::log10(1.0f + (float)(sample_rate_ / 2) / 700.0f);
    for (int m = 0; m < n_filters; m++) {
        float mel_center = (float)m * mel_max / (float)(n_filters + 1);
        float hz_center = 700.0f * (std::pow(10.0f, mel_center / 2595.0f) - 1.0f);
        int bin_center = std::min((int)(hz_center * (float)fft_size / (float)sample_rate_), (int)n_bins - 1);
        int bin_left = (m > 0) ? std::min((int)(700.0f * (std::pow(10.0f, (float)(m - 1) * mel_max / (float)(n_filters + 1) / 2595.0f) - 1.0f) * (float)fft_size / (float)sample_rate_), (int)n_bins - 1) : 0;
        int bin_right = (m < n_filters - 1) ? std::min((int)(700.0f * (std::pow(10.0f, (float)(m + 1) * mel_max / (float)(n_filters + 1) / 2595.0f) - 1.0f) * (float)fft_size / (float)sample_rate_), (int)n_bins - 1) : (int)n_bins - 1;
        if (bin_right <= bin_left) bin_right = bin_left + 1;
        for (int k = bin_left; k <= bin_center && k < (int)n_bins; k++)
            fb[(size_t)m * (size_t)n_bins + (size_t)k] = (float)(k - bin_left) / (float)(bin_center - bin_left + 1);
        for (int k = bin_center + 1; k <= bin_right && k < (int)n_bins; k++)
            fb[(size_t)m * (size_t)n_bins + (size_t)k] = (float)(bin_right - k) / (float)(bin_right - bin_center + 1);
    }
    return fb;
}

Tensor MelSpectrogram::compute(const Tensor& waveform) {
    if (waveform.numel() < 2) return Tensor({n_mels_, 1});

    int64_t samples = waveform.numel();
    int n_fft = 1024;
    int hop = n_fft / 4;
    int64_t n_frames = (samples - n_fft) / hop + 1;
    if (n_frames < 1) n_frames = 1;

    auto fb = mel_filterbank(n_mels_, n_fft);
    int n_bins = n_fft / 2 + 1;
    const float* wd = waveform.data<float>();
    Tensor mel({n_mels_, n_frames});
    float* md = mel.data<float>();

    for (int64_t f = 0; f < n_frames; f++) {
        int64_t offset = f * hop;
        std::vector<float> spectrum((size_t)n_bins, 0.0f);
        for (int64_t k = 0; k < n_bins; k++) {
            float re = 0, im = 0;
            float angle = 2.0f * 3.14159265f * (float)k / (float)n_fft;
            for (int64_t n = 0; n < n_fft && offset + n < samples; n++) {
                float hann = 0.5f * (1.0f - std::cos(2.0f * 3.14159265f * (float)n / (float)(n_fft - 1)));
                re += wd[offset + n] * hann * std::cos(angle * n);
                im -= wd[offset + n] * hann * std::sin(angle * n);
            }
            spectrum[(size_t)k] = re * re + im * im;
        }
        for (int m = 0; m < n_mels_; m++) {
            float val = 0;
            for (int64_t k = 0; k < n_bins; k++)
                val += spectrum[(size_t)k] * fb[(size_t)m * (size_t)n_bins + (size_t)k];
            md[(size_t)m * (size_t)n_frames + (size_t)f] = std::log(std::max(val, 1e-10f));
        }
    }
    return mel;
}

// ============================================================================
// H12: Audio feature extractor
// ============================================================================
AudioFeatureExtractor::AudioFeatureExtractor(Model* encoder)
    : encoder_(encoder) {
}

Tensor AudioFeatureExtractor::extract(const Tensor& audio) {
    // D6 W7: null encoder throws; empty input returns empty; forward failure empty.
    if (encoder_ == nullptr) {
        std::cerr << "[AudioFeatureExtractor] null encoder" << std::endl;
        throw std::invalid_argument("AudioFeatureExtractor: null encoder");
    }
    if (audio.numel() == 0) {
        std::cerr << "[AudioFeatureExtractor] empty audio input" << std::endl;
        return Tensor({0});
    }
    Tensor out = encoder_->forward(audio, Tensor({1, 1}));
    if (out.numel() == 0) {
        std::cerr << "[AudioFeatureExtractor] feature extraction failed" << std::endl;
        return Tensor({0});
    }
    return out;
}

// ============================================================================
// H13: Multi-modal tokenizer
// ============================================================================
MultiModalTokenizer::MultiModalTokenizer() {}
MultiModalTokenizer::MultiModalTokenizer(BPETokenizer* bpe) : bpe_(bpe) {}

std::vector<int> MultiModalTokenizer::encode(const std::string& text) {
    if (bpe_) return bpe_->encode(text);
    std::vector<int> tokens;
    tokens.reserve(text.size());
    for (char c : text) tokens.push_back((int)(unsigned char)c);
    return tokens;
}

std::string MultiModalTokenizer::decode(const std::vector<int>& tokens) {
    if (bpe_) return bpe_->decode(tokens);
    std::string s;
    s.reserve(tokens.size());
    for (int t : tokens)
        if (t >= 0 && t < 256) s += (char)t;
    return s;
}

std::vector<int> MultiModalTokenizer::encode_image(const Tensor& image) {
    if (image.numel() == 0) return {};
    // Content-dependent patch tokens (honest fallback when no vision encoder wired):
    // split into up to 16 patches, quantize patch means into [30000,30099].
    const float* d = image.data<float>();
    int64_t n = image.numel();
    int64_t patches = std::min<int64_t>(16, n);
    int64_t per = n / patches;
    std::vector<int> out;
    out.reserve((size_t)patches);
    for (int64_t p = 0; p < patches; ++p) {
        double m = 0;
        for (int64_t i = 0; i < per; ++i) m += d[p * per + i];
        m /= (double)std::max<int64_t>(1, per);
        float v = (float)m;
        uint32_t u;
        std::memcpy(&u, &v, 4);
        // mix bits to spread nearby means across token range
        u ^= u >> 13; u *= 0x5bd1e995u; u ^= u >> 15;
        out.push_back(30000 + (int)(u % 100));
    }
    return out;
}

std::vector<int> MultiModalTokenizer::encode_audio(const Tensor& audio) {
    if (audio.numel() == 0) return {};
    const float* d = audio.data<float>();
    int64_t n = audio.numel();
    int64_t patches = std::min<int64_t>(16, n);
    int64_t per = n / patches;
    std::vector<int> out;
    out.reserve((size_t)patches);
    for (int64_t p = 0; p < patches; ++p) {
        double m = 0;
        for (int64_t i = 0; i < per; ++i) m += d[p * per + i];
        m /= (double)std::max<int64_t>(1, per);
        float v = (float)m;
        uint32_t u;
        std::memcpy(&u, &v, 4);
        u ^= u >> 13; u *= 0x5bd1e995u; u ^= u >> 15;
        out.push_back(31000 + (int)(u % 100));
    }
    return out;
}

// ============================================================================
// H14: Modality encoder
// ============================================================================
ModalityEncoder::ModalityEncoder(Model* model, const std::string& modality)
    : model_(model), modality_(modality) {
}

Tensor ModalityEncoder::encode(const Tensor& input) {
    // D6 W7: null model throws; empty input returns empty; forward failure empty.
    if (model_ == nullptr) {
        std::cerr << "[ModalityEncoder] null model" << std::endl;
        throw std::invalid_argument("ModalityEncoder: null model");
    }
    if (input.numel() == 0) {
        std::cerr << "[ModalityEncoder] empty input" << std::endl;
        return Tensor({0});
    }
    Tensor out = model_->forward(input, Tensor({1, 1}));
    if (out.numel() == 0) {
        std::cerr << "[ModalityEncoder] feature extraction failed" << std::endl;
        return Tensor({0});
    }
    return out;
}

// ============================================================================
// H15: Cross-modal alignment (contrastive learning)
// ============================================================================
CrossModalAlignment::CrossModalAlignment(Model* vision, Model* text, float temperature)
    : vision_(vision), text_(text), temperature_(temperature > 0 ? temperature : 0.07f) {
}

float CrossModalAlignment::contrastive_loss(const Tensor& image_emb, const Tensor& text_emb) {
    QUANT_CHECK(image_emb.numel() > 0 && text_emb.numel() > 0,
              "CrossModalAlignment: empty embeddings");
    QUANT_CHECK(image_emb.rank() >= 2 && text_emb.rank() >= 2,
              "CrossModalAlignment: embeddings must be at least 2D");

    int64_t B = image_emb.dim(0);
    int64_t D = image_emb.dim(1);
    QUANT_CHECK(text_emb.dim(0) == B && text_emb.dim(1) == D,
              "CrossModalAlignment: embedding shape mismatch");

    const float* ie_raw = image_emb.data<float>();
    const float* te_raw = text_emb.data<float>();

    std::vector<float> ie((size_t)B * (size_t)D);
    std::vector<float> te((size_t)B * (size_t)D);
    std::copy(ie_raw, ie_raw + B * D, ie.data());
    std::copy(te_raw, te_raw + B * D, te.data());

    for (int64_t b = 0; b < B; b++) {
        float ni = 0.0f, nt = 0.0f;
        for (int64_t d = 0; d < D; d++) {
            ni += ie[(size_t)b * (size_t)D + (size_t)d] * ie[(size_t)b * (size_t)D + (size_t)d];
            nt += te[(size_t)b * (size_t)D + (size_t)d] * te[(size_t)b * (size_t)D + (size_t)d];
        }
        ni = std::sqrt(ni + 1e-10f);
        nt = std::sqrt(nt + 1e-10f);
        for (int64_t d = 0; d < D; d++) {
            ie[(size_t)b * (size_t)D + (size_t)d] /= ni;
            te[(size_t)b * (size_t)D + (size_t)d] /= nt;
        }
    }

    float loss = 0;

    std::vector<float> sim((size_t)B * (size_t)B, 0.0f);
    for (int64_t i = 0; i < B; i++)
        for (int64_t j = 0; j < B; j++) {
            float dot = 0;
            for (int64_t d = 0; d < D; d++)
                dot += ie[(size_t)i * (size_t)D + (size_t)d] * te[(size_t)j * (size_t)D + (size_t)d];
            sim[(size_t)i * (size_t)B + (size_t)j] = dot / temperature_;
        }

    for (int64_t i = 0; i < B; i++) {
        float max_val = sim[(size_t)i * (size_t)B];
        for (int64_t j = 1; j < B; j++)
            if (sim[(size_t)i * (size_t)B + (size_t)j] > max_val)
                max_val = sim[(size_t)i * (size_t)B + (size_t)j];

        float sum_exp = 0;
        for (int64_t j = 0; j < B; j++)
            sum_exp += std::exp(sim[(size_t)i * (size_t)B + (size_t)j] - max_val);
        float log_prob = std::log(std::exp(sim[(size_t)i * (size_t)B + (size_t)i] - max_val) / (sum_exp + 1e-10f));
        loss -= log_prob;
    }
    return loss / (float)B;
}

Tensor CrossModalAlignment::align(const Tensor& image_emb, const Tensor& text_emb) {
    QUANT_CHECK(image_emb.numel() > 0 && text_emb.numel() > 0,
              "CrossModalAlignment: empty embeddings");
    QUANT_CHECK(image_emb.rank() >= 2 && text_emb.rank() >= 2,
              "CrossModalAlignment: embeddings must be 2D");

    int64_t B = image_emb.dim(0), D = image_emb.dim(1);
    QUANT_CHECK(text_emb.dim(0) == B && text_emb.dim(1) == D,
              "CrossModalAlignment: embedding shape mismatch");

    Tensor aligned({B, D});
    float* ad = aligned.data<float>();
    const float* ie = image_emb.data<float>();
    const float* te = text_emb.data<float>();

    // Alignment via learned weighted combination
    // Compute per-dimension attention weights
    for (int64_t b = 0; b < B; b++) {
        float img_norm = 0, txt_norm = 0;
        for (int64_t d = 0; d < D; d++) {
            img_norm += ie[b * D + d] * ie[b * D + d];
            txt_norm += te[b * D + d] * te[b * D + d];
        }
        img_norm = std::sqrt(img_norm + 1e-10f);
        txt_norm = std::sqrt(txt_norm + 1e-10f);

        float alpha = img_norm / (img_norm + txt_norm + 1e-10f);
        float beta = 1.0f - alpha;

        for (int64_t d = 0; d < D; d++)
            ad[b * D + d] = alpha * ie[b * D + d] + beta * te[b * D + d];
    }
    return aligned;
}

// ============================================================================
// H16: Perceiver — learned query-based cross-attention
// ============================================================================
Perceiver::Perceiver(int64_t hidden, int64_t num_queries, int64_t num_heads)
    : hidden_(hidden), num_queries_(num_queries),
      q_proj(hidden, hidden), k_proj(hidden, hidden),
      v_proj(hidden, hidden), out_proj(hidden, hidden) {}

Tensor Perceiver::forward(const Tensor& input, const Tensor& queries) {
    int64_t T = input.dim(0);
    int64_t D = hidden_;
    int64_t Q = queries.dim(0);

    Tensor Q_out = q_proj.forward(queries);
    Tensor K_out = k_proj.forward(input);
    Tensor V_out = v_proj.forward(input);

    float scale = 1.0f / std::sqrt((float)D);
    Tensor output({Q, D});
    output.zero_();
    float* od = output.data<float>();
    const float* qd = Q_out.data<float>();
    const float* kd = K_out.data<float>();
    const float* vd = V_out.data<float>();

    for (int64_t q = 0; q < Q; ++q) {
        float row_max = -INFINITY;
        std::vector<float> scores((size_t)T);
        for (int64_t t = 0; t < T; ++t) {
            float dot = 0;
            for (int64_t d = 0; d < D; ++d)
                dot += qd[q * D + d] * kd[t * D + d];
            scores[(size_t)t] = dot * scale;
            if (scores[(size_t)t] > row_max) row_max = scores[(size_t)t];
        }
        float sum = 0;
        for (int64_t t = 0; t < T; ++t) {
            scores[(size_t)t] = std::exp(scores[(size_t)t] - row_max);
            sum += scores[(size_t)t];
        }
        for (int64_t t = 0; t < T; ++t) {
            float w = scores[(size_t)t] / (sum + 1e-10f);
            for (int64_t d = 0; d < D; ++d)
                od[q * D + d] += w * vd[t * D + d];
        }
    }
    return out_proj.forward(output);
}

// ============================================================================
// H17: Vision MoE — modality-specific expert routing for vision
// ============================================================================
VisionMoE::VisionMoE(int64_t hidden, int64_t num_experts)
    : hidden_(hidden), router(hidden, num_experts) {
    for (int64_t i = 0; i < num_experts; ++i)
        experts.emplace_back(hidden, hidden * 4);
}

Tensor VisionMoE::forward(const Tensor& image_features) {
    int64_t T = image_features.dim(0);
    int64_t D = hidden_;
    int64_t E = (int64_t)experts.size();
    Tensor logits = router.forward(image_features);
    Tensor probs({T, E});
    math::softmax(logits, probs, 1);
    const float* pd = probs.data<float>();
    const float* xd = image_features.data<float>();
    Tensor output({T, D});
    output.zero_();
    float* od = output.data<float>();
    for (int64_t e = 0; e < E; ++e) {
        Tensor expert_out = experts[(size_t)e].forward(image_features);
        const float* eo = expert_out.data<float>();
        for (int64_t t = 0; t < T; ++t) {
            float w = pd[t * E + e];
            for (int64_t d = 0; d < D; ++d)
                od[t * D + d] += w * eo[t * D + d];
        }
    }
    return output;
}

// ============================================================================
// H18: Audio MoE — modality-specific expert routing for audio
// ============================================================================
AudioMoE::AudioMoE(int64_t hidden, int64_t num_experts)
    : hidden_(hidden), router(hidden, num_experts) {
    for (int64_t i = 0; i < num_experts; ++i)
        experts.emplace_back(hidden, hidden * 4);
}

Tensor AudioMoE::forward(const Tensor& audio_features) {
    int64_t T = audio_features.dim(0);
    int64_t D = hidden_;
    int64_t E = (int64_t)experts.size();
    Tensor logits = router.forward(audio_features);
    Tensor probs({T, E});
    math::softmax(logits, probs, 1);
    const float* pd = probs.data<float>();
    const float* xd = audio_features.data<float>();
    Tensor output({T, D});
    output.zero_();
    float* od = output.data<float>();
    for (int64_t e = 0; e < E; ++e) {
        Tensor expert_out = experts[(size_t)e].forward(audio_features);
        const float* eo = expert_out.data<float>();
        for (int64_t t = 0; t < T; ++t) {
            float w = pd[t * E + e];
            for (int64_t d = 0; d < D; ++d)
                od[t * D + d] += w * eo[t * D + d];
        }
    }
    return output;
}

// ============================================================================
// H19: Vision+Text MoE — cross-modal routing
// ============================================================================
VisionTextMoE::VisionTextMoE(int64_t hidden, int64_t num_experts)
    : hidden_(hidden), router(hidden, num_experts),
      modality_bias(hidden, 2) {
    for (int64_t i = 0; i < num_experts; ++i)
        experts.emplace_back(hidden, hidden * 4);
}

Tensor VisionTextMoE::forward(const Tensor& vision_tokens, const Tensor& text_tokens) {
    int64_t V = vision_tokens.dim(0);
    int64_t T = text_tokens.dim(0);
    int64_t D = hidden_;
    int64_t total = V + T;
    int64_t E = (int64_t)experts.size();

    Tensor combined({total, D});
    float* cd = combined.data<float>();
    std::memcpy(cd, vision_tokens.data<float>(), (size_t)(V * D) * sizeof(float));
    std::memcpy(cd + V * D, text_tokens.data<float>(), (size_t)(T * D) * sizeof(float));

    Tensor logits = router.forward(combined);
    float* ld = logits.data<float>();

    Tensor mod_bias = modality_bias.forward(combined);
    const float* md = mod_bias.data<float>();
    for (int64_t t = 0; t < total; ++t) {
        int64_t mod = (t < V) ? 0 : 1;
        for (int64_t e = 0; e < E; ++e) {
            ld[t * E + e] += 5.0f * md[t * 2 + mod];
        }
    }

    Tensor probs({total, E});
    math::softmax(logits, probs, 1);
    const float* pd = probs.data<float>();

    Tensor output({total, D});
    output.zero_();
    float* od = output.data<float>();
    for (int64_t e = 0; e < E; ++e) {
        Tensor expert_out = experts[(size_t)e].forward(combined);
        const float* eo = expert_out.data<float>();
        for (int64_t t = 0; t < total; ++t) {
            float w = pd[t * E + e];
            for (int64_t d = 0; d < D; ++d)
                od[t * D + d] += w * eo[t * D + d];
        }
    }
    return output;
}

// ============================================================================
// H20: Audio+Text MoE — cross-modal routing
// ============================================================================
AudioTextMoE::AudioTextMoE(int64_t hidden, int64_t num_experts)
    : hidden_(hidden), router(hidden, num_experts),
      modality_bias(hidden, 2) {
    for (int64_t i = 0; i < num_experts; ++i)
        experts.emplace_back(hidden, hidden * 4);
}

Tensor AudioTextMoE::forward(const Tensor& audio_tokens, const Tensor& text_tokens) {
    int64_t A = audio_tokens.dim(0);
    int64_t T = text_tokens.dim(0);
    int64_t D = hidden_;
    int64_t total = A + T;
    int64_t E = (int64_t)experts.size();

    Tensor combined({total, D});
    float* cd = combined.data<float>();
    std::memcpy(cd, audio_tokens.data<float>(), (size_t)(A * D) * sizeof(float));
    std::memcpy(cd + A * D, text_tokens.data<float>(), (size_t)(T * D) * sizeof(float));

    Tensor logits = router.forward(combined);
    float* ld = logits.data<float>();
    Tensor mod_bias = modality_bias.forward(combined);
    const float* md = mod_bias.data<float>();
    for (int64_t t = 0; t < total; ++t) {
        int64_t mod = (t < A) ? 0 : 1;
        for (int64_t e = 0; e < E; ++e)
            ld[t * E + e] += 5.0f * md[t * 2 + mod];
    }

    Tensor probs({total, E});
    math::softmax(logits, probs, 1);
    const float* pd = probs.data<float>();

    Tensor output({total, D});
    output.zero_();
    float* od = output.data<float>();
    for (int64_t e = 0; e < E; ++e) {
        Tensor expert_out = experts[(size_t)e].forward(combined);
        const float* eo = expert_out.data<float>();
        for (int64_t t = 0; t < total; ++t) {
            float w = pd[t * E + e];
            for (int64_t d = 0; d < D; ++d)
                od[t * D + d] += w * eo[t * D + d];
        }
    }
    return output;
}

// ============================================================================
// H21: All Modality MoE — routes across all modalities
// ============================================================================
AllModalityMoE::AllModalityMoE(int64_t hidden, int64_t num_experts)
    : hidden_(hidden), router(hidden, num_experts),
      modality_classifier(hidden, 6) {
    for (int64_t i = 0; i < num_experts; ++i)
        experts.emplace_back(hidden, hidden * 4);
}

Tensor AllModalityMoE::forward(const std::vector<Tensor>& modality_tokens) {
    if (modality_tokens.empty())
        return Tensor({hidden_});
    int64_t D = hidden_;
    int64_t E = (int64_t)experts.size();

    int64_t total = 0;
    for (auto& m : modality_tokens)
        total += m.dim(0);

    Tensor combined({total, D});
    float* cd = combined.data<float>();
    int64_t offset = 0;
    for (auto& m : modality_tokens) {
        int64_t n = m.dim(0);
        std::memcpy(cd + offset * D, m.data<float>(), (size_t)(n * D) * sizeof(float));
        offset += n;
    }

    Tensor logits = router.forward(combined);
    float* ld = logits.data<float>();

    Tensor mod_logits = modality_classifier.forward(combined);
    Tensor mod_probs({total, 6});
    math::softmax(mod_logits, mod_probs, 1);
    const float* mp = mod_probs.data<float>();

    for (int64_t t = 0; t < total; ++t) {
        for (int64_t e = 0; e < E; ++e) {
            int64_t exp_mod = e % 6;
            ld[t * E + e] += 3.0f * mp[t * 6 + exp_mod];
        }
    }

    Tensor probs({total, E});
    math::softmax(logits, probs, 1);
    const float* pd = probs.data<float>();

    Tensor output({total, D});
    output.zero_();
    float* od = output.data<float>();
    for (int64_t e = 0; e < E; ++e) {
        Tensor expert_out = experts[(size_t)e].forward(combined);
        const float* eo = expert_out.data<float>();
        for (int64_t t = 0; t < total; ++t) {
            float w = pd[t * E + e];
            for (int64_t d = 0; d < D; ++d)
                od[t * D + d] += w * eo[t * D + d];
        }
    }
    return output;
}

} // namespace quant
