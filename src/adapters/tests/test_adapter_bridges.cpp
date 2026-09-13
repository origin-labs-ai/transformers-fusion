// ============================================================================
// test_adapter_bridges.cpp — Smoke tests for all ADAPTER-EDITION bridges
// ============================================================================
#include "adapters/adapter_core.h"
#include "adapters/ptq_bridge.h"
#include "adapters/gguf_bridge.h"
#include "adapters/safetensors_bridge.h"

#include <cstdio>
#include <cstdlib>
// ctime removed — using <random> instead
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

#include "quant/quant_format.h"
#include "quant/block_codec.h"
#include "quant/tensor.h"

using namespace quant::adapters;
using namespace quant;

static int g_failures = 0;

static void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); g_failures++; }
    else      { std::fprintf(stdout, "PASS: %s\n", msg); }
}

static void test_fp_conversions() {
    std::fprintf(stdout, "\n=== FP Conversion Tests ===\n");

    float f16_1 = fp16_to_float(0x3C00);
    check(std::fabs(f16_1 - 1.0f) < 0.01f, "FP16 1.0");

    float f16_neg = fp16_to_float(0xBC00);
    check(std::fabs(f16_neg - (-1.0f)) < 0.01f, "FP16 -1.0");

    float f16_half = fp16_to_float(0x3800);
    check(std::fabs(f16_half - 0.5f) < 0.01f, "FP16 0.5");

    float bf16_1 = bf16_to_float(0x3F80);
    check(std::fabs(bf16_1 - 1.0f) < 0.01f, "BF16 1.0");

    float fp8_1 = fp8_e4m3_to_float(0x38);
    check(std::fabs(fp8_1 - 1.0f) < 0.01f, "FP8 E4M3 1.0");

    float fp8_1e5m2 = fp8_e5m2_to_float(0x3C);
    check(std::fabs(fp8_1e5m2 - 1.0f) < 0.01f, "FP8 E5M2 1.0");
}

static void test_format_detection() {
    std::fprintf(stdout, "\n=== Format Detection Tests ===\n");

    check(detect_format("test.bin") == ExternalFormat::RAW_FP32, ".bin = RAW_FP32");
    check(detect_format("test.fp32") == ExternalFormat::RAW_FP32, ".fp32 = RAW_FP32");
    check(detect_format("test.fp16") == ExternalFormat::RAW_FP16, ".fp16 = RAW_FP16");
    check(detect_format("test.gguf") == ExternalFormat::GGUF, ".gguf = GGUF");
    check(detect_format("test.safetensors") == ExternalFormat::SAFETENSORS, ".safetensors = SAFETENSORS");
    check(detect_format("test.quant") == ExternalFormat::QUANT, ".quant = QUANT");
    check(detect_format("test.fp8e4m3") == ExternalFormat::RAW_FP8_E4M3, ".fp8e4m3 = FP8");
    check(detect_format("test.unknown") == ExternalFormat::UNKNOWN, ".unknown = UNKNOWN");
}

static void test_mixed_write() {
    std::fprintf(stdout, "\n=== Mixed-Precision Write Tests ===\n");

    BridgeConfig cfg;
    cfg.target_bpw = 2.0f;
    cfg.block_size = 256;
    cfg.output_path = "test_adapter.quant";

    std::vector<AdapterTensor> tensors;

    AdapterTensor t1;
    t1.name = "layer0.weight";
    t1.shape = {64, 64};
    t1.data.resize(64 * 64);
    static thread_local std::mt19937 rng(42);
    for (auto& v : t1.data) v = ((float)(std::uniform_int_distribution<int>(0, 1999)(rng)) / 1000.0f - 1.0f) * 0.1f;
    tensors.push_back(std::move(t1));

    AdapterTensor t2;
    t2.name = "layer1.weight";
    t2.shape = {64, 64};
    t2.data.resize(64 * 64);
    for (auto& v : t2.data) v = ((float)(std::uniform_int_distribution<int>(0, 1999)(rng)) / 1000.0f - 1.0f) * 0.1f;
    tensors.push_back(std::move(t2));

    bool ok = write_quant_mixed(tensors, cfg);
    check(ok, "write_quant_mixed returns true");

    float bpw = estimate_mixed_bpw(64 * 64, 256, 2.0f);
    check(bpw > 1.0f && bpw < 4.0f, "estimated BPW in valid range");
}

static void test_qg_mx_budget_at_scale() {
    std::fprintf(stdout, "\n=== QG_MX Budget Tests ===\n");

    constexpr int64_t N = 1 << 20; // 1M weights, 4096 blocks of 256
    std::vector<float> data((size_t)N);
    std::mt19937 rng(20260802);
    for (int64_t g = 0; g < N / 1024; ++g) {
        const float scale = 0.20f + 0.15f * (float)(g + 1);
        std::normal_distribution<float> dist(0.0f, scale);
        for (int64_t k = 0; k < 1024; ++k) data[(size_t)(g * 1024 + k)] = dist(rng);
    }

    const MixDescriptor* mix = nullptr;
    for (const auto& m : FormatRegistry::get_all_four_mixes())
        if (m.name == "QG_MX_3.5") mix = &m;
    check(mix != nullptr, "QG_MX_3.5 registered");
    if (!mix) return;

    const std::vector<Format> fmts =
        allocate_tensor_formats("layer0.weight", N, data.data(), 256, Format::Q2, mix);

    size_t total_bytes = 0;
    bool budget_ok = true;
    for (int b = 0; b < (int)fmts.size(); ++b) {
        const Format bfmt = fmts[(size_t)b];
        const int start = b * 256;
        const int n = (int)std::min<int64_t>(256, N - start);
        std::vector<uint8_t> indices, codebook;
        quantize_block(bfmt, data.data() + start, n, indices, codebook);
        const size_t stored = indices.size() + codebook.size();
        const size_t cap = block_claimed_bytes(bfmt, (uint32_t)n);
        if (stored > cap) budget_ok = false;
        total_bytes += stored;
    }
    check(budget_ok, "every QG_MX block fits its claimed byte budget");

    const double expected = 3.78125 * (double)N / 8.0;
    const double actual = (double)total_bytes;
    char message[256];
    std::snprintf(message, sizeof(message),
                  "QG_MX file bytes %.0f vs claimed %.0f (%.2f%% off)",
                  actual, expected, 100.0 * (actual - expected) / expected);
    check(std::fabs(actual - expected) / expected < 0.01, message);
    std::fprintf(stdout, "QG_MX_3.5 @ 1M weights: %zu bytes (claim 3.78125 BPW -> %.0f)\n",
                 total_bytes, expected);
}

static void test_raw_load() {
    std::fprintf(stdout, "\n=== Raw Load Tests ===\n");

    std::vector<float> data = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f};
    std::string path = "test_adapter_raw.fp32";
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    f.close();

    AdapterTensor tensor = load_raw_blob(path, ExternalFormat::RAW_FP32);
    check(!tensor.data.empty(), "load_raw_blob loads FP32");
    if (!tensor.data.empty()) {
        check(tensor.data.size() == 6, "raw load gets correct count");
    }
}

// ── L070 LLaMA-family loader tests (Phase 16 Wave 7) ────────────────────────

static AdapterTensor make_llama_tensor(const std::string& name,
                                       std::vector<int64_t> shape) {
    AdapterTensor t;
    t.name = name;
    t.shape = std::move(shape);
    int64_t n = 1;
    for (int64_t d : t.shape) n *= d;
    t.data.assign((size_t)n, 0.01f);
    return t;
}

static std::vector<AdapterTensor> make_tiny_llama(bool qwen_marker) {
    std::vector<AdapterTensor> ts;
    ts.push_back(make_llama_tensor("model.embed_tokens.weight", {32, 16}));
    for (int l = 0; l < 2; l++) {
        std::string p = "model.layers." + std::to_string(l) + ".";
        ts.push_back(make_llama_tensor(p + "self_attn.q_proj.weight", {16, 16}));
        ts.push_back(make_llama_tensor(p + "self_attn.k_proj.weight", {16, 16}));
        ts.push_back(make_llama_tensor(p + "self_attn.v_proj.weight", {16, 16}));
        ts.push_back(make_llama_tensor(p + "self_attn.o_proj.weight", {16, 16}));
        ts.push_back(make_llama_tensor(p + "mlp.gate_proj.weight", {32, 16}));
        ts.push_back(make_llama_tensor(p + "mlp.up_proj.weight", {32, 16}));
        ts.push_back(make_llama_tensor(p + "mlp.down_proj.weight", {16, 32}));
        ts.push_back(make_llama_tensor(p + "input_layernorm.weight", {16}));
        ts.push_back(make_llama_tensor(p + "post_attention_layernorm.weight", {16}));
        if (qwen_marker && l == 0)
            ts.push_back(make_llama_tensor(p + "self_attn.q_norm.weight", {16}));
    }
    ts.push_back(make_llama_tensor("model.norm.weight", {16}));
    ts.push_back(make_llama_tensor("lm_head.weight", {32, 16}));
    return ts;
}

static void test_llama_loader() {
    std::fprintf(stdout, "\n=== L070 LLaMA Loader Tests ===\n");
    auto ts = make_tiny_llama(false);
    check(detect_llama_family(ts) == LlamaFamily::Llama, "llama family detected");
    std::string err;
    check(validate_llama_tensors(ts, &err), "tiny llama validates");
    LlamaArchConfig c = infer_llama_arch(ts);
    check(c.valid, "llama arch valid");
    check(c.num_layers == 2, "llama 2 layers inferred");
    check(c.hidden_size == 16, "llama hidden 16 inferred");
    check(c.vocab_size == 32, "llama vocab 32 inferred");
    check(c.ffn_hidden == 32, "llama ffn 32 inferred");
    check(!c.tie_embeddings, "llama head present (not tied)");

    auto qwen = make_tiny_llama(true);
    check(detect_llama_family(qwen) == LlamaFamily::QwenDense, "qwen-dense marker detected");
    check(validate_llama_tensors(qwen, &err), "tiny qwen validates");

    std::vector<AdapterTensor> garbage;
    AdapterTensor g;
    g.name = "some_random_weight";
    g.shape = {8, 8};
    g.data.assign(64, 0.1f);
    garbage.push_back(g);
    check(detect_llama_family(garbage) == LlamaFamily::Unknown, "garbage is Unknown family");
    check(!validate_llama_tensors(garbage, &err), "garbage fails validation");
    check(!err.empty(), "validation error message set");
}

// ── L071 MoE arch loader tests (Phase 16 Wave 7) ────────────────────────────

static std::vector<AdapterTensor> make_tiny_mixtral() {
    std::vector<AdapterTensor> ts;
    ts.push_back(make_llama_tensor("model.embed_tokens.weight", {32, 16}));
    std::string p = "model.layers.0.";
    ts.push_back(make_llama_tensor(p + "self_attn.q_proj.weight", {16, 16}));
    ts.push_back(make_llama_tensor(p + "block_sparse_moe.gate.weight", {4, 16}));
    for (int e = 0; e < 4; e++) {
        std::string ep = p + "block_sparse_moe.experts." + std::to_string(e) + ".";
        ts.push_back(make_llama_tensor(ep + "w1.weight", {32, 16}));
        ts.push_back(make_llama_tensor(ep + "w2.weight", {16, 32}));
        ts.push_back(make_llama_tensor(ep + "w3.weight", {32, 16}));
    }
    ts.push_back(make_llama_tensor("model.norm.weight", {16}));
    return ts;
}

static void test_moe_loader() {
    std::fprintf(stdout, "\n=== L071 MoE Loader Tests ===\n");
    auto ts = make_tiny_mixtral();
    check(is_moe_tensors(ts), "mixtral experts detected");
    MoeArchConfig c = infer_moe_arch(ts);
    check(c.valid, "moe arch valid");
    check(c.is_moe, "moe flag set");
    check(c.num_experts == 4, "4 experts inferred");
    check(c.num_layers_with_moe == 1, "1 moe layer inferred");
    check(c.pattern == "mixtral", "mixtral pattern named");
    std::string err;
    check(validate_moe_tensors(ts, &err), "tiny mixtral validates");

    auto dense = make_tiny_llama(false);
    check(!is_moe_tensors(dense), "dense llama is not moe");
    check(!validate_moe_tensors(dense, &err), "dense fails moe validation");
}

// ── L059 Block-FP8 pair loader tests (Phase 15 Wave 6) ───────────────────
// Qwen3.8-class split layout: raw F8_E4M3 weight bytes + one F32 scale per
// `block` elements. This test crafts a minimal safetensors file by hand
// (header JSON + raw segment) and verifies bit-exact scaled loads plus
// honest failures (missing keys, scale-count mismatch).

static void write_u64_le(std::ofstream& f, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        f.put((char)(v & 0xFF));
        v >>= 8;
    }
}

static void test_block_fp8_loader() {
    std::fprintf(stdout, "\n=== L059 Block-FP8 Loader Tests ===\n");
    const std::string path = "test_block_fp8.safetensors";
    // weight "w": 4x F8_E4M3 bytes [0x38, 0x38, 0xBC, 0x00] = [1, 1, -1, 0]
    // scale "w_scale": 2x F32 [0.5, 2.0], block = 2
    // expected: [0.5, 0.5, -2.0, 0.0]
    const std::string header =
        "{\"w\":{\"dtype\":\"F8_E4M3\",\"shape\":[4],\"data_offsets\":[0,4]},"
        "\"w_scale\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[4,12]}}";
    {
        std::ofstream f(path, std::ios::binary);
        write_u64_le(f, (uint64_t)header.size());
        f.write(header.data(), (std::streamsize)header.size());
        const uint8_t wbytes[4] = {0x38, 0x38, 0xBC, 0x00};
        f.write(reinterpret_cast<const char*>(wbytes), 4);
        const float scales[2] = {0.5f, 2.0f};
        f.write(reinterpret_cast<const char*>(scales), 8);
    }
    BlockFp8Group g = load_block_fp8_pair(path, "w", "w_scale", 2);
    check(g.ok, "block-fp8 pair loads ok");
    check(g.use_e4m3, "e4m3 variant detected");
    check(g.block_size == 2, "block size kept");
    if (g.ok && g.data.size() == 4) {
        check(std::fabs(g.data[0] - 0.5f) < 1e-6f, "w[0] = 1.0 * 0.5");
        check(std::fabs(g.data[1] - 0.5f) < 1e-6f, "w[1] = 1.0 * 0.5");
        check(std::fabs(g.data[2] + 2.0f) < 1e-6f, "w[2] = -1.0 * 2.0");
        check(std::fabs(g.data[3] - 0.0f) < 1e-6f, "w[3] = 0.0 * 2.0");
    } else {
        check(false, "block-fp8 data has 4 elements");
    }
    // Plain per-element F8 path sees the same bytes (regression: decoder fix).
    auto plain = load_safetensors(path, false);
    for (const auto& t : plain) {
        if (t.name == "w" && t.data.size() == 4) {
            check(std::fabs(t.data[0] - 1.0f) < 1e-6f, "plain F8_E4M3 0x38 = 1.0 (decoder fix)");
            check(std::fabs(t.data[2] + 1.0f) < 1e-6f, "plain F8_E4M3 0xBC = -1.0 (decoder fix)");
        }
    }
    // Honest failures.
    BlockFp8Group miss = load_block_fp8_pair(path, "nope", "w_scale", 2);
    check(!miss.ok, "missing weight key => !ok");
    BlockFp8Group miss_scale = load_block_fp8_pair(path, "w", "nope_scale", 2);
    check(!miss_scale.ok, "missing scale key => !ok");
    BlockFp8Group bad_block = load_block_fp8_pair(path, "w", "w_scale", 1);
    check(!bad_block.ok, "scale-count mismatch (block=1 needs 4 scales, has 2) => !ok");
    std::remove(path.c_str());
}

int main() {
    std::fprintf(stdout, "========================================\n");
    std::fprintf(stdout, "  ADAPTER-EDITION SMOKE TESTS\n");
    std::fprintf(stdout, "========================================\n");

    test_fp_conversions();
    test_format_detection();
    test_mixed_write();
    test_qg_mx_budget_at_scale();
    test_raw_load();
    test_llama_loader();
    test_moe_loader();
    test_block_fp8_loader();

    std::fprintf(stdout, "\n========================================\n");
    if (g_failures == 0)
        std::fprintf(stdout, "  ALL TESTS PASSED\n");
    else
        std::fprintf(stdout, "  %d TEST(S) FAILED\n", g_failures);
    std::fprintf(stdout, "========================================\n");

    return g_failures;
}
