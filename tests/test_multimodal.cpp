// test_multimodal.cpp — multimodal models, fusion, tokenizer
#include "quant/multimodal.h"
#include "quant/tensor.h"
#include "quant/tokenizer.h"
#include "quant/test.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <string>

using namespace quant;

static bool all_finite(const Tensor& t) {
    if (t.numel() == 0) return true;
    const float* d = t.data<float>();
    for (int64_t i = 0; i < t.numel(); i++)
        if (!std::isfinite(d[i])) return false;
    return true;
}

int main() {
    fprintf(stderr, "MM0 start\n");
    TEST_SUITE("multimodal");
    printf("=== Multimodal test ===\n\n");

    // H1: Cross-modal attention
    {
        fprintf(stderr, "MM1 cma\n");
        CrossModalAttention cma(16);
        Tensor text(Shape{2, 8, 16}, DType::F32);
        Tensor image(Shape{2, 4, 16}, DType::F32);
        Tensor audio(Shape{2, 6, 16}, DType::F32);
        text.fill(0.1f); image.fill(0.2f); audio.fill(0.3f);
        Tensor out = cma.forward({text, image, audio});
        TEST_CHECK(out.numel() > 0 && all_finite(out), "cross-modal attention finite");
    }

    // H2: Joint multimodal model
    {
        fprintf(stderr, "MM2 joint\n");
        TransformerConfig cfg;
        cfg.vocab_size = 64;
        cfg.hidden_size = 16;
        cfg.num_layers = 1;
        cfg.num_heads = 2;
        cfg.head_dim = 8;
        cfg.ffn_hidden_size = 32;
        cfg.max_seq_len = 32;

        DenseModel text_enc(cfg);
        JointMultimodalModel joint(&text_enc, 16);
        Tensor text(Shape{2, 4, 16}, DType::F32);
        Tensor image(Shape{2, 4, 16}, DType::F32);
        Tensor audio(Shape{2, 4, 16}, DType::F32);
        text.fill(0.1f); image.fill(0.2f); audio.fill(0.3f);
        Tensor out = joint.forward(text, image, audio);
        TEST_CHECK(out.numel() > 0 && all_finite(out), "joint multimodal forward finite");
    }

    // H13: Multi-modal tokenizer
    {
        fprintf(stderr, "MM3 tokenizer\n");
        MultiModalTokenizer tok;
        TEST_CHECK(tok.vocab_size() == 32000, "default vocab 32000");
        TEST_CHECK(tok.image_token_id() == 30000, "image token id");
        TEST_CHECK(tok.audio_token_id() == 31000, "audio token id");

        BPETokenizer bpe;
        MultiModalTokenizer tok2(&bpe);
        auto ids = tok2.encode("hello world");
        TEST_CHECK(!ids.empty(), "text encodes to ids");
        std::string dec = tok2.decode(ids);
        TEST_CHECK(!dec.empty(), "ids decode to text");
    }

    // H14: Modality encoder + H12 audio extractor
    {
        fprintf(stderr, "MM4 encoders\n");
        TransformerConfig cfg;
        cfg.vocab_size = 64;
        cfg.hidden_size = 16;
        cfg.num_layers = 1;
        cfg.num_heads = 2;
        cfg.head_dim = 8;
        cfg.ffn_hidden_size = 32;
        cfg.max_seq_len = 32;

        DenseModel enc(cfg);
        ModalityEncoder me(&enc, "vision");
        Tensor input(Shape{3, 16}, DType::F32);
        input.fill(0.5f);
        Tensor out = me.encode(input);
        TEST_CHECK(out.numel() > 0 && all_finite(out), "modality encoder finite");

        AudioFeatureExtractor afx(&enc);
        Tensor audio(Shape{4, 16}, DType::F32);
        audio.fill(0.5f);
        Tensor aout = afx.extract(audio);
        TEST_CHECK(aout.numel() > 0, "audio feature extractor runs");
    }

    // H16: Perceiver
    {
        fprintf(stderr, "MM5 perceiver\n");
        Perceiver perc(16, 4, 2);
        Tensor input(Shape{8, 16}, DType::F32);
        Tensor queries(Shape{4, 16}, DType::F32);
        input.fill(0.1f); queries.fill(0.2f);
        Tensor out = perc.forward(input, queries);
        TEST_CHECK(out.numel() == 4 * 16 && all_finite(out), "perceiver finite");
    }

    // H17/18: Vision/Audio MoE
    {
        fprintf(stderr, "MM6 moe\n");
        VisionMoE vmoe(16, 4);
        Tensor features(Shape{6, 16}, DType::F32);
        features.fill(0.3f);
        Tensor out = vmoe.forward(features);
        TEST_CHECK(out.numel() > 0 && all_finite(out), "vision MoE finite");

        AudioMoE amoe(16, 4);
        Tensor afeat(Shape{6, 16}, DType::F32);
        afeat.fill(0.3f);
        Tensor aout = amoe.forward(afeat);
        TEST_CHECK(aout.numel() > 0 && all_finite(aout), "audio MoE finite");
    }

    // H11: Mel spectrogram
    {
        fprintf(stderr, "MM7 mel\n");
        MelSpectrogram mel(8000, 8);
        Tensor waveform(Shape{1, 8000}, DType::F32);
        float* wd = waveform.data<float>();
        for (int64_t i = 0; i < 8000; i++)
            wd[i] = (float)std::sin(2.0 * 3.14159 * 440.0 * (double)i / 8000.0);
        Tensor m = mel.compute(waveform);
        TEST_CHECK(m.numel() > 0 && all_finite(m), "mel spectrogram finite");
    }

    // Fusion config
    {
        fprintf(stderr, "MM8 fusion\n");
        FusionConfig fcfg;
        fcfg.hidden_size = 16;
        fcfg.num_heads = 2;
        fcfg.head_dim = 8;
        MultimodalFusion fusion(fcfg);
        TEST_CHECK(fusion.param_count() > 0, "fusion has parameters");

        Tensor vision(Shape{2, 16, 16}, DType::F32);
        Tensor audio(Shape{2, 16, 16}, DType::F32);
        Tensor text(Shape{2, 16, 16}, DType::F32);
        vision.fill(0.1f); audio.fill(0.2f); text.fill(0.3f);

        Tensor v = fusion.fuse_vision_audio(vision, audio);
        fprintf(stderr, "MM8a fuse_va ok\n");
        Tensor t = fusion.fuse_vision_text(vision, text);
        fprintf(stderr, "MM8b fuse_vt ok\n");
        Tensor a = fusion.fuse_audio_text(audio, text);
        fprintf(stderr, "MM8c fuse_at ok\n");
        Tensor all = fusion.fuse_all(vision, audio, text);
        fprintf(stderr, "MM8d fuse_all ok\n");
        TEST_CHECK(v.numel() > 0 && all_finite(v), "vision-audio fusion finite");
        TEST_CHECK(t.numel() > 0 && all_finite(t), "vision-text fusion finite");
        TEST_CHECK(a.numel() > 0 && all_finite(a), "audio-text fusion finite");
        TEST_CHECK(all.numel() > 0 && all_finite(all), "all-modality fusion finite");
    }

    // H9: TextToImage DDIM pipeline — honest contract (PROD round-8).
    // The pipeline is structurally real (DDIM schedule + text conditioning +
    // tanh decode) but the noise predictor is a local-smoothing proxy, NOT a
    // trained UNet — so assert pipeline properties, never image quality:
    // (a) output shape {3,S,S} + finite, (b) deterministic (seed-42),
    // (c) prompt-sensitive (text conditioning is wired, not ignored).
    {
        fprintf(stderr, "MM9 t2i\n");
        TextToImage t2i(16, 32);
        Tensor img1 = t2i.generate("a red sunset over mountains", 5);
        TEST_CHECK(img1.dim(0) == 3 && img1.dim(1) == 32 && img1.dim(2) == 32,
                   "t2i output shape {3,32,32}");
        TEST_CHECK(all_finite(img1), "t2i output finite");
        Tensor img1b = t2i.generate("a red sunset over mountains", 5);
        bool same = img1.numel() == img1b.numel();
        if (same) {
            const float* d1 = img1.data<float>();
            const float* d2 = img1b.data<float>();
            for (int64_t i = 0; i < img1.numel() && same; i++)
                if (d1[i] != d2[i]) same = false;
        }
        TEST_CHECK(same, "t2i deterministic for same prompt+steps");
        Tensor img2 = t2i.generate("a blue ocean with white waves", 5);
        bool differs = false;
        if (img2.numel() == img1.numel()) {
            const float* d1 = img1.data<float>();
            const float* d2 = img2.data<float>();
            for (int64_t i = 0; i < img1.numel(); i++)
                if (d1[i] != d2[i]) { differs = true; break; }
        }
        TEST_CHECK(differs, "t2i prompt-sensitive (conditioning wired)");
    }

    int failures = TEST_REPORT();
    printf("\nMULTIMODAL TEST %s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures > 0 ? 1 : 0;
}
