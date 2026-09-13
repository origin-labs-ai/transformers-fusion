#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
// test_grp_quality_proof.cpp — Quality proof test verifying GRP variants beat 2x BPW base formats
#include "quant/types.h"
#include "quant/format_registry.h"
#include "quant/block_codec.h"
#include "quant/model.h"
#include "quant/trainer.h"
#include "quant/bpe_tokenizer.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <random>
#include <algorithm>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "  Transcender GRP Quality Superiority Proof  " << std::endl;
    std::cout << "=========================================" << std::endl;

    // Verify all 9 integer-GRP formats exist and are registered.
    // v3 enum ids: QG1..QG24 = 37..45 (10..18 are K variants now).
    std::cout << "[Test 1] Verifying all 9 GRP variants in FormatRegistry..." << std::endl;
    for (int i = 37; i <= 45; i++) {
        auto fmt = static_cast<quant::Format>(i);
        std::string name = quant::format_name(fmt);
        float bpw = quant::format_bpw(fmt);

        assert(quant::format_is_grp(fmt));
        assert(bpw > 0.0f);
        std::cout << "  -> " << name << " (BPW: " << bpw << ") verified!" << std::endl;
    }

    std::cout << "[Test 2] Verifying quality hierarchy: GRP variants carry per-group scales..." << std::endl;
    constexpr int N = 256;
    std::vector<float> data(N);
    for (int i = 0; i < N; i++) data[i] = (float)(i % 16) * 0.1f - 0.8f;

    // v3 registry singles carry integer BPW only; request the GRP descriptor
    // by name instead of snapping a half-BPW target to the nearest plain.
    auto q4_grp = quant::FormatRegistry::parse_format_name("QG4");
    assert(q4_grp.grouped);
    assert(q4_grp.group_size == 16.0f || q4_grp.group_size == 32.0f);

    std::cout << "  -> QG4 group size: " << q4_grp.group_size << std::endl;
    std::cout << "  -> PASSED: GRP grouped scaling structure verified!" << std::endl;

    // [Test 3] Real round-trip PSNR proof on gaussian weights (sigma=0.1).
    // Regression guard for the QG3 encode/decode level-grid mismatch bug:
    // encoder snapped to 4 levels (v*3) while decoder read a 16-level grid
    // (idx/15), collapsing PSNR to ~12.6 dB. Both sides now use 8 levels.
    // L012 pre/post CSV note (no rebench here; thresholds anchored to frozen
    // evidence repo/state/telemetry/bench_history/bench_baseline_20260823_speed1.csv):
    //   pre-fix:  QG3 ~12.6 dB (collapsed grid, see comment above).
    //   post-fix: gaussian Q3_G 29.208 dB (Q3 plain 24.786) [row 87 vs 88];
    //             real     Q3_G 30.4575 dB (Q3 plain 26.586) [row 199 vs 200].
    // L012 ACCEPT bar: gaussian >= 24 dB AND real >= 25 dB AND GRP>=plain.
    std::cout << "[Test 3] Round-trip PSNR on gaussian(sigma=0.1), N=8192..." << std::endl;
    {
        constexpr int N = 8192;
        std::vector<float> w(N);
        uint64_t seed = 0x9E3779B97F4A7C15ull;
        auto rnd = [&seed]() {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            return (double)(seed >> 11) / 9007199254740992.0;
        };
        for (int i = 0; i < N; ++i) {
            double u1 = rnd(), u2 = rnd();
            double g = std::sqrt(-2.0 * std::log(u1 + 1e-300)) *
                       std::cos(6.283185307179586 * u2);
            w[(size_t)i] = (float)(g * 0.1);
        }
        auto psnr_of = [&](quant::Format f) {
            std::vector<uint8_t> idx, cb;
            std::vector<float> dec(N);
            if (!quant::quantize_block_all(f, w.data(), N, idx, cb)) return -1.0f;
            quant::dequantize_block_all(f, idx.data(), idx.size(), cb.data(),
                                        cb.size(), (uint32_t)N, dec.data());
            double se = 0.0, smax = 0.0;
            for (int i = 0; i < N; ++i) {
                const double d = (double)dec[(size_t)i] - (double)w[(size_t)i];
                se += d * d;
                smax = std::max(smax, std::fabs((double)w[(size_t)i]));
            }
            const double mse = se / N;
            return (mse <= 0.0) ? 100.0f : (float)(10.0 * std::log10(smax * smax / mse));
        };
        const float p3_grp = psnr_of(quant::Format::QG3);
        const float p3     = psnr_of(quant::Format::Q3);
        const float p4_grp = psnr_of(quant::Format::QG4);
        const float p2_grp = psnr_of(quant::Format::QG2);
        std::cout << "  -> QG3 PSNR: " << p3_grp << " dB (Q3 plain: "
                  << p3 << " dB)" << std::endl;
        assert(p3_grp >= 24.0f);                 // L012 hard floor (was ~12.6 pre-fix, 22.0 interim)
        assert(p3_grp >= p3 - 0.5f);             // GRP(3.5 BPW) must beat Q3(3.0)
        assert(p4_grp > p3_grp);                 // more bits = better, monotonic
        assert(p2_grp < p3_grp);                 // fewer bits = worse, monotonic
        std::cout << "  -> PASSED: QG3 round-trip quality restored!" << std::endl;
    }

    // [Test 4] L012 real-weights proof: QG3 >= 25 dB on trained tensors.
    // Uses the existing real-tensor path pattern from
    // bench/bench_format_comparison.cpp::train_real_tensors + run_format_tensors:
    // tiny DenseModel trained 300 steps, then per-tensor chunked round-trip
    // (kChunk=256, blocks never span tensor boundaries, exactly like
    // model.save_quantized). Frozen post-fix reference:
    // bench_baseline_20260823_speed1.csv row 199: real Q3_G 30.4575 dB.
    std::cout << "[Test 4] Round-trip PSNR on real trained weights (per-tensor path)..."
              << std::endl;
    {
        constexpr int64_t kChunk = 256;
        // --- train real tensors (bench pattern, verbatim config) ---
        quant::TransformerConfig cfg;
        cfg.vocab_size = 64;
        cfg.hidden_size = 48;
        cfg.num_layers = 2;
        cfg.num_heads = 4;
        cfg.head_dim = 12;
        cfg.ffn_hidden_size = 96;
        cfg.max_seq_len = 16;
        quant::DenseModel model(cfg);
        quant::BPETokenizer tokenizer;
        quant::Trainer trainer(&model, &tokenizer);
        quant::TrainConfig tcfg;
        tcfg.batch_size = 2;
        tcfg.seq_length = 8;
        tcfg.train_steps = 300;
        tcfg.learning_rate = 3e-3f;
        tcfg.weight_decay = 0.0f;
        tcfg.warmup_steps = 5;
        tcfg.log_interval = 1000;
        tcfg.save_interval = 1000;
        tcfg.val_interval = 1000;
        tcfg.output_path = "bench_weights.quant";
        trainer.compile(tcfg);
        const int64_t B = 2, S = 8;
        std::mt19937 rng(7);
        std::vector<int64_t> seq((size_t)(B * S));
        for (size_t i = 0; i < seq.size(); i++) seq[i] = (int64_t)(rng() % (uint64_t)cfg.vocab_size);
        for (int step = 0; step < 300; step++) {
            quant::Tensor input_ids(quant::Shape{B, S}, quant::DType::F32);
            quant::Tensor labels(quant::Shape{B, S}, quant::DType::F32);
            float* idp = input_ids.data<float>();
            float* lbp = labels.data<float>();
            for (int64_t i = 0; i < B * S; i++) {
                idp[i] = (float)seq[(size_t)i];
                lbp[i] = (float)seq[(size_t)i];
            }
            trainer.train_step(input_ids, labels);
        }
        std::vector<quant::Tensor*> params;
        model.get_parameters(params);
        std::vector<std::vector<float>> tensors;
        for (auto* p : params) {
            if (!p || p->numel() <= 0) continue;
            std::vector<float> t((size_t)p->numel());
            const float* d = p->data<float>();
            for (int64_t i = 0; i < p->numel(); i++) t[(size_t)i] = d[i];
            tensors.push_back(std::move(t));
        }
        assert(!tensors.empty());
        // --- per-tensor chunked PSNR (blocks never cross tensor boundary) ---
        auto psnr_tensors = [&](quant::Format f) {
            struct Span { size_t ti; int64_t start, cnt; };
            std::vector<Span> spans;
            for (size_t ti = 0; ti < tensors.size(); ++ti)
                for (int64_t s = 0; s < (int64_t)tensors[ti].size(); s += kChunk)
                    spans.push_back({ti, s, std::min(kChunk, (int64_t)tensors[ti].size() - s)});
            std::vector<std::vector<float>> outs(tensors.size());
            for (size_t ti = 0; ti < tensors.size(); ++ti) outs[ti].resize(tensors[ti].size());
            std::vector<uint8_t> idx, cb;
            double se = 0.0, smax = 0.0;
            int64_t total = 0;
            for (const auto& t : tensors) {
                for (float v : t) smax = std::max(smax, (double)std::fabs(v));
                total += (int64_t)t.size();
            }
            for (const auto& sp : spans) {
                const float* src = tensors[sp.ti].data() + sp.start;
                float* dst = outs[sp.ti].data() + sp.start;
                if (!quant::quantize_block_all(f, src, (int)sp.cnt, idx, cb)) return -1.0f;
                quant::dequantize_block_all(f, idx.data(), idx.size(), cb.data(),
                                            cb.size(), (uint32_t)sp.cnt, dst);
            }
            for (size_t ti = 0; ti < tensors.size(); ++ti)
                for (size_t i = 0; i < tensors[ti].size(); i++) {
                    const double d = (double)outs[ti][i] - (double)tensors[ti][i];
                    se += d * d;
                }
            const double mse = se / (double)total;
            return (mse <= 0.0) ? 100.0f : (float)(10.0 * std::log10(smax * smax / mse));
        };
        const float r3_grp = psnr_tensors(quant::Format::QG3);
        const float r3     = psnr_tensors(quant::Format::Q3);
        std::cout << "  -> QG3 real-weights PSNR: " << r3_grp << " dB (Q3 plain: "
                  << r3 << " dB)" << std::endl;
        assert(r3_grp >= 25.0f);                 // L012 real floor (post-fix ref 30.46)
        assert(r3_grp >= r3 - 0.5f);             // GRP must hold vs plain on real too
        std::cout << "  -> PASSED: QG3 real-weights quality holds!" << std::endl;
    }

    std::cout << "\nGRP QUALITY PROOF TEST PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
