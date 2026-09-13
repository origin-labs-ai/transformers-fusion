// test_fine_tuning.cpp — selective FT, rank adapters, knowledge expansion
#include "quant/fine_tuning.h"
#include "quant/model.h"
#include "quant/transformer.h"
#include "quant/tensor.h"
#include "quant/autograd.h"
#include "quant/test.h"
#include <vector>
#include <cmath>
#include <cstdio>

using namespace quant;

static TransformerConfig make_cfg() {
    TransformerConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = cfg.hidden_size / cfg.num_heads;
    cfg.ffn_hidden_size = 32;
    cfg.norm_eps = 1e-5f;
    cfg.max_seq_len = 32;
    return cfg;
}

static void fill_batch(Tensor& ids, Tensor& pos, Tensor& tgt, int vocab) {
    for (int64_t i = 0; i < ids.numel(); i++) {
        int64_t tok = (i + 1) % (vocab - 1);
        ids.data<float>()[i] = (float)(tok + 1);
        pos.data<float>()[i] = (float)(i % 8);
        tgt.data<float>()[i] = (float)((tok + 2) % vocab);
    }
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    TEST_SUITE("fine_tuning");
    printf("=== Fine-tuning engines test ===\n\n");

    TransformerConfig cfg = make_cfg();
    DenseModel model(cfg);
    printf("[trace] model ctor done\n");

    const int64_t B = 2, S = 8;
    Tensor ids(Shape{B, S}, DType::F32);
    Tensor pos(Shape{B, S}, DType::F32);
    Tensor tgt(Shape{B, S}, DType::F32);
    fill_batch(ids, pos, tgt, (int)cfg.vocab_size);
    printf("[trace] tensors done\n");

    // --- Method 1: Selective fine-tuning ---
    {
        SelectiveFineTuner ft(&model);
        SelectiveTunerConfig fcfg;
        fcfg.num_epochs = 1;
        fcfg.learning_rate = 1e-4f;
        ft.configure(fcfg);
        ft.freeze_all(true);
        printf("[trace] ft configured\n");

        AutogradEngine::set_enabled(true);
        printf("[trace] model.forward (enabled)...\n");
        Tensor lg = model.forward(ids, pos);
        printf("[trace] forward done numel=%lld\n", (long long)lg.numel());
        Tensor ls = AutogradEngine::cross_entropy_op(lg, tgt);
        printf("[trace] loss done\n");
        AutogradEngine::instance().backward(ls);
        printf("[trace] backward done\n");
        AutogradEngine::instance().clear();
        AutogradEngine::set_enabled(false);

        ft.accumulate_fisher(ids, pos, tgt);
        printf("[trace] fisher done\n");
        TEST_CHECK(!ft.fisher_diag().empty(), "fisher diagonal populated");

        float before = 0.0f;
        {
            AutogradEngine::set_enabled(false);
            Tensor logits = model.forward(ids, pos);
            Tensor l = AutogradEngine::cross_entropy_op(logits, tgt);
            before = *(const float*)l.data();
            AutogradEngine::set_enabled(true);
        }
        ft.fine_tune(ids, pos, tgt, 5);
        float after = ft.stats().last_loss;
        printf("  selective FT loss: %.4f -> %.4f\n", before, after);
        TEST_CHECK(std::isfinite(after), "selective FT loss finite");
        TEST_CHECK(after < before + 1e-2f, "selective FT does not blow up loss");

        // select_blocks returns one flag per 64-weight block (it also needs the
        // fisher diagonal, not nullptr, to select anything).
        int64_t n_weights = (int64_t)ft.fisher_diag()[0].size();
        std::vector<bool> sel = SelectiveFineTuner::select_blocks(
            ft.fisher_diag()[0].data(), ft.fisher_diag()[0].data(), n_weights,
            64, 2.0f, 0.35f, 0.01f, nullptr);
        int64_t n_blocks = (n_weights + 63) / 64;
        TEST_CHECK((int64_t)sel.size() == n_blocks, "block selection size matches (one flag per block)");
    }

    // --- Method 2: Rank adapters ---
    {
        DenseModel m2(make_cfg());
        RankAdapterEngine ra(&m2);
        RankAdapterConfig rcfg;
        rcfg.rank = 4;
        rcfg.learning_rate = 3e-4f;
        ra.configure(rcfg);
        ra.init_adapters();
        TEST_CHECK(ra.adapter_param_count() > 0, "adapter params > 0");
        TEST_CHECK(!ra.adapters().empty(), "one adapter per layer");

        ra.freeze_base(true);
        Tensor out = ra.forward_with_adapters(ids, pos);
        TEST_CHECK(out.numel() == B * S * cfg.vocab_size, "adapter forward logits shape");

        ra.train_step(ids, pos, tgt);
        printf("  rank adapter step loss: %.4f (step %d)\n", ra.last_loss(), ra.step());
        TEST_CHECK(std::isfinite(ra.last_loss()), "adapter step loss finite");
        TEST_CHECK(ra.step() == 1, "adapter step counter advanced");
        // PROD: was TEST_CHECK(true) — vacuous. merge_into_base() adds
        // ΔW=B·A into down_proj. NOTE (LoRA-correct): B inits to 0, so one
        // step leaves ΔW≈0 — run several steps so B moves, then assert.
        for (int s = 0; s < 9; s++) ra.train_step(ids, pos, tgt);
        TEST_CHECK(ra.step() == 10, "adapter ran 10 steps");
        const float* wb0 = m2.layers[0]->ffn.down_proj.weight.data<float>();
        int64_t nwb = m2.layers[0]->ffn.down_proj.weight.numel();
        std::vector<float> snap(wb0, wb0 + nwb);
        ra.merge_into_base();
        const float* wb1 = m2.layers[0]->ffn.down_proj.weight.data<float>();
        double delta = 0.0;
        for (int64_t i = 0; i < nwb; i++) {
            double d = (double)wb1[i] - (double)snap[(size_t)i];
            delta += d * d;
        }
        printf("  merge delta^2: %.6g\n", delta);
        TEST_CHECK(delta > 0.0, "adapter merge changes base weights");
        Tensor merged_out = m2.forward(ids, pos);
        bool mfinite = true;
        const float* md = merged_out.data<float>();
        for (int64_t i = 0; i < merged_out.numel(); i++)
            if (!std::isfinite(md[i])) mfinite = false;
        TEST_CHECK(mfinite, "merged model forward finite");
    }

    // --- Method 2b: DoRA (weight-decomposed rank adapters) ---
    // W' = m ⊙ (W0 + B·A) / ||W0 + B·A||_c. Three invariants are pinned here:
    //   (1) at init (delta == 0) DoRA reproduces the base weight exactly, i.e. it
    //       reduces to plain LoRA with the magnitude frozen at ||W0||_c;
    //   (2) after any merge, every output column has L2 norm exactly m[o] — that
    //       renormalisation IS DoRA, so if it holds the decomposition is real;
    //   (3) magnitude_step() performs gradient descent on m.
    {
        DenseModel m4(make_cfg());
        RankAdapterEngine ra(&m4);
        RankAdapterConfig rcfg;
        rcfg.rank = 4;
        rcfg.learning_rate = 3e-4f;
        rcfg.use_dora = true;
        ra.configure(rcfg);
        ra.init_adapters();

        TEST_CHECK(ra.dora_enabled(), "dora_enabled() reports the configured mode");
        TEST_CHECK(!ra.adapters().empty(), "dora: one adapter per layer");

        const auto& ad0 = ra.adapters()[0];
        const int64_t out_dim = ad0.out_dim, in_dim = ad0.in_dim;
        TEST_CHECK(ra.dora_param_count() == (int64_t)ra.adapters().size() * out_dim,
                   "dora: one magnitude scalar per output column per layer");
        TEST_CHECK(ad0.magnitude.numel() == out_dim, "dora: magnitude vector sized out_dim");

        // (1) magnitude must equal the base column norms at init.
        {
            const float* w0 = m4.layers[0]->ffn.down_proj.weight.data<float>();
            const float* mag = ad0.magnitude.data<float>();
            double worst = 0.0;
            for (int64_t o = 0; o < out_dim; ++o) {
                double s = 0.0;
                for (int64_t k = 0; k < in_dim; ++k) {
                    double d = (double)w0[k * out_dim + o];
                    s += d * d;
                }
                worst = std::max(worst, std::fabs(mag[o] - std::sqrt(s)));
            }
            printf("  dora: |m - ||W0||_c| max = %.6g\n", worst);
            TEST_CHECK(worst < 1e-4, "dora: m initialised to base column norms");
        }

        // (1) zero delta + renormalise => identity on the base weight.
        {
            const float* w0 = m4.layers[0]->ffn.down_proj.weight.data<float>();
            int64_t n = m4.layers[0]->ffn.down_proj.weight.numel();
            std::vector<float> snap(w0, w0 + n);
            ra.merge_into_base();
            const float* w1 = m4.layers[0]->ffn.down_proj.weight.data<float>();
            double worst = 0.0;
            for (int64_t i = 0; i < n; ++i)
                worst = std::max(worst, (double)std::fabs((double)w1[i] - (double)snap[(size_t)i]));
            printf("  dora: identity-merge max abs diff = %.6g\n", worst);
            TEST_CHECK(worst < 1e-4, "dora: zero-delta merge reproduces the base (reduces to LoRA)");
        }

        // Train so B moves away from zero, then merge and check (2).
        ra.freeze_base(true);
        Tensor out = ra.forward_with_adapters(ids, pos);
        TEST_CHECK(out.numel() == B * S * cfg.vocab_size, "dora forward logits shape");
        for (int s = 0; s < 10; ++s) ra.train_step(ids, pos, tgt);
        TEST_CHECK(std::isfinite(ra.last_loss()), "dora step loss finite");

        // (3) magnitude_step must apply exactly the analytic DoRA gradient
        //     dL/dm[o] = Σ_k dL/dW'[k,o] · V[k,o] / ||V[:,o]||,
        // with V = W0 + B·A at the current factors. Checking the arithmetic
        // beats checking a sign: Σ_k V[k,o] can be negative, so a uniformly
        // positive dL/dW' does not imply m moves the same way in every column.
        {
            const int64_t r = ad0.rank;
            const float* w0 = m4.layers[0]->ffn.down_proj.weight.data<float>();
            const float* aa = ad0.A.data<float>();
            const float* bb = ad0.B.data<float>();
            std::vector<double> v((size_t)in_dim * (size_t)out_dim), nrm((size_t)out_dim, 0.0);
            for (int64_t k = 0; k < in_dim; ++k)
                for (int64_t o = 0; o < out_dim; ++o) {
                    double acc = 0.0;
                    for (int64_t t = 0; t < r; ++t)
                        acc += (double)aa[k * r + t] * (double)bb[t * out_dim + o];
                    double merged = (double)w0[k * out_dim + o] + acc;
                    v[(size_t)k * out_dim + o] = merged;
                    nrm[(size_t)o] += merged * merged;
                }
            for (int64_t o = 0; o < out_dim; ++o) nrm[(size_t)o] = std::sqrt(nrm[(size_t)o]);

            const float lr = 0.01f;
            std::vector<Tensor> grads;
            for (size_t li = 0; li < ra.adapters().size(); ++li) {
                Tensor g(Shape{in_dim, out_dim}, DType::F32);
                g.fill(1.0f);
                grads.push_back(g);
            }
            std::vector<float> before(ad0.magnitude.data<float>(),
                                      ad0.magnitude.data<float>() + out_dim);
            ra.magnitude_step(grads, lr);

            const float* after = ra.adapters()[0].magnitude.data<float>();
            double worst = 0.0;
            bool moved = false;
            for (int64_t o = 0; o < out_dim; ++o) {
                double col_sum = 0.0;
                for (int64_t k = 0; k < in_dim; ++k) col_sum += v[(size_t)k * out_dim + o];
                double expected = (double)before[(size_t)o] - (double)lr * col_sum / nrm[(size_t)o];
                worst = std::max(worst, std::fabs((double)after[o] - expected));
                if (std::fabs((double)after[o] - (double)before[(size_t)o]) > 1e-9) moved = true;
            }
            printf("  dora: magnitude_step vs analytic max abs err = %.6g\n", worst);
            TEST_CHECK(worst < 1e-5, "dora: magnitude_step matches the analytic gradient");
            TEST_CHECK(moved, "dora: magnitude_step actually moves m");
        }

        ra.merge_into_base();
        {
            const float* w1 = m4.layers[0]->ffn.down_proj.weight.data<float>();
            const float* mag = ra.adapters()[0].magnitude.data<float>();
            double worst = 0.0;
            for (int64_t o = 0; o < out_dim; ++o) {
                double s = 0.0;
                for (int64_t k = 0; k < in_dim; ++k) {
                    double d = (double)w1[k * out_dim + o];
                    s += d * d;
                }
                worst = std::max(worst, std::fabs(std::sqrt(s) - (double)mag[o]));
            }
            printf("  dora: ||W'[:,o]||_c - m[o] max = %.6g\n", worst);
            TEST_CHECK(worst < 1e-4, "dora: merged columns are renormalised to m exactly");
        }

        Tensor merged_out = m4.forward(ids, pos);
        bool dfinite = true;
        const float* md = merged_out.data<float>();
        for (int64_t i = 0; i < merged_out.numel(); i++)
            if (!std::isfinite(md[i])) dfinite = false;
        TEST_CHECK(dfinite, "dora merged model forward finite");
    }

    // --- Method 3: Knowledge expansion ---
    {
        DenseModel m3(make_cfg());
        KnowledgeExpansionEngine ke(&m3);
        KnowledgeExpansionConfig kcfg;
        kcfg.slot_width = 8;
        kcfg.learning_rate = 3e-4f;
        ke.configure(kcfg);
        ke.freeze_base(true);

        int64_t slot = ke.add_slot("new_knowledge");
        TEST_CHECK(slot == 0, "first slot index 0");
        TEST_CHECK(ke.slot_count() == 1, "one slot registered");
        TEST_CHECK(ke.slot_param_count() > 0, "slot params > 0");

        Tensor out = ke.forward_with_slots(ids, pos);
        TEST_CHECK(out.numel() == B * S * cfg.vocab_size, "slot forward logits shape");

        Tensor conf, used;
        Tensor gout = ke.forward_guarded(ids, pos, nullptr, &conf, &used);
        TEST_CHECK(gout.numel() == B * S * cfg.vocab_size, "guarded forward shape");

        ke.train_step(ids, pos, tgt);
        printf("  knowledge expansion step loss: %.4f\n", ke.last_loss());
        TEST_CHECK(std::isfinite(ke.last_loss()), "slot train loss finite");
    }

    int failures = TEST_REPORT();
    printf("\nFINE TUNING TEST %s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures > 0 ? 1 : 0;
}
