#include "quant/qwen35_engine.h"
#include "quant/block_codec.h"
#include <cmath>
#include <cstring>
#include <memory>
#include <new>
#include <thread>
#include <algorithm>
#include <cstdio>

namespace quant {

static const int HID = 4096;
static const int MAX_BATCH = 512;

// ---- block decode ---------------------------------------------------------
// Every QUANT/QUANT format (0..14) is decoded via the canonical block codec, so
// the engine runs QG_MX files too (they are per-block formats).
void Qwen35Engine::decode_block(const QUANTReader& rd, uint32_t block_id, uint32_t nw, float* out) {
    for (uint32_t i = 0; i < 256; i++) out[i] = 0.0f;
    const uint8_t* raw = rd.block_ptr(block_id);
    if (!raw) return;
    uint32_t cb;
    std::memcpy(&cb, raw + 4, 4);
    uint32_t idx_bytes;
    std::memcpy(&idx_bytes, raw + 8 + cb, 4);
    const uint8_t* idx = raw + 8 + cb + 4;
    const Format fmt = (Format)(uint8_t)rd.format_entry(block_id).format;
    dequantize_block_all(fmt, idx, idx_bytes, raw + 8, cb, nw, out);
}

// ---- small helpers --------------------------------------------------------
static float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static float softplusf(float x) {
    if (x > 30.0f) return x;
    if (x < -30.0f) return 0.0f;
    return std::log1pf(std::exp(x));
}
static float siluf(float x) { return x * sigmoidf(x); }

void Qwen35Engine::rmsnorm(float* out, const float* x, const float* w, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    float r = 1.0f / std::sqrt(s / n + 1e-6f);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * (1.0f + w[i]);
}

// ---- gemm: Y[B][rows] = X[B][cols] @ W^T ---------------------------------
void Qwen35Engine::gemm(const Th& t, const float* X, float* Y, int B) {
    const int rows = (int)t.rows, cols = (int)t.cols;
    const int bpr = cols / 256;
    int nt = (int)std::thread::hardware_concurrency();
    if (nt < 1) nt = 1;
    if (nt > 8) nt = 8;

    std::vector<std::thread> threads;
    int chunk = (rows + nt - 1) / nt;
    std::vector<float> scratch(nt * 256);
    std::vector<float> acc(nt * MAX_BATCH);

    for (int tid = 0; tid < nt; tid++) {
        int r0 = tid * chunk;
        int r1 = std::min(rows, r0 + chunk);
        if (r0 >= r1) continue;
        threads.emplace_back([&, tid, r0, r1]() {
            float* sc = &scratch[(size_t)tid * 256];
            float* ac = &acc[(size_t)tid * MAX_BATCH];
            for (int r = r0; r < r1; r++) {
                for (int b = 0; b < B; b++) ac[b] = 0.0f;
                for (int blk = 0; blk < bpr; blk++) {
                    uint32_t bid = t.start + (uint32_t)r * (uint32_t)bpr + (uint32_t)blk;
                    decode_block(*reader_, bid, 256, sc);
                    for (int b = 0; b < B; b++) {
                        const float* xb = X + (size_t)b * cols;
                        float a = 0.0f;
                        const float* s = sc;
                        const float* x = xb + (size_t)blk * 256;
                        for (int j = 0; j < 256; j++) a += s[j] * x[j];
                        ac[b] += a;
                    }
                }
                for (int b = 0; b < B; b++) Y[(size_t)b * rows + r] = ac[b];
            }
        });
    }
    for (auto& t2 : threads) t2.join();
}

// ---- constructor / load ---------------------------------------------------
Qwen35Engine::Qwen35Engine()
    : reader_(nullptr), ok_(false), vocab_(0), pos_(0), max_kv_(4096), scratch_(nullptr),
      rng_(0x9E3779B97F4A7C15ull) {
    skip_full_ = getenv("QUANT_SKIP_FULL") != nullptr;
    skip_gla_ = getenv("QUANT_SKIP_GLA") != nullptr;
}

Qwen35Engine::~Qwen35Engine() {
    delete reader_;
    delete[] scratch_;
}

bool Qwen35Engine::resolve(const std::string& name, uint32_t rows, uint32_t cols) {
    uint32_t start, count;
    if (!reader_->tensor_blocks(name, start, count)) {
        std::fprintf(stderr, "qwen35_engine: missing tensor %s\n", name.c_str());
        return false;
    }
    Th t;
    t.start = start;
    t.count = count;
    t.fmt = (Format)reader_->format_entry(start).format;
    t.rows = rows;
    t.cols = cols;
    th_[name] = t;
    return true;
}

bool Qwen35Engine::load_layer(int L) {
    char buf[256];
    const char* p = "model.language_model.layers.";
    std::snprintf(buf, sizeof(buf), "%s%d.input_layernorm.weight", p, L);
    if (!resolve(buf, 4096, 4096)) return false;
    std::snprintf(buf, sizeof(buf), "%s%d.post_attention_layernorm.weight", p, L);
    if (!resolve(buf, 4096, 4096)) return false;
    std::snprintf(buf, sizeof(buf), "%s%d.mlp.gate_proj.weight", p, L);
    if (!resolve(buf, 12288, 4096)) return false;
    std::snprintf(buf, sizeof(buf), "%s%d.mlp.up_proj.weight", p, L);
    if (!resolve(buf, 12288, 4096)) return false;
    std::snprintf(buf, sizeof(buf), "%s%d.mlp.down_proj.weight", p, L);
    if (!resolve(buf, 4096, 12288)) return false;

    bool full = ((L + 1) % 4) == 0;
            if (full) {
                std::fprintf(stderr, "q "); std::fflush(stderr);
                std::snprintf(buf, sizeof(buf), "%s%d.self_attn.q_proj.weight", p, L);
        if (!resolve(buf, 8192, 4096)) return false;
        std::snprintf(buf, sizeof(buf), "%s%d.self_attn.k_proj.weight", p, L);
        if (!resolve(buf, 1024, 4096)) return false;
        std::snprintf(buf, sizeof(buf), "%s%d.self_attn.v_proj.weight", p, L);
        if (!resolve(buf, 1024, 4096)) return false;
        std::snprintf(buf, sizeof(buf), "%s%d.self_attn.o_proj.weight", p, L);
        if (!resolve(buf, 4096, 4096)) return false;
        std::snprintf(buf, sizeof(buf), "%s%d.self_attn.q_norm.weight", p, L);
        if (!resolve(buf, 256, 256)) return false;
        std::snprintf(buf, sizeof(buf), "%s%d.self_attn.k_norm.weight", p, L);
        if (!resolve(buf, 256, 256)) return false;
    } else {
        std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.in_proj_qkv.weight", p, L);
        if (!resolve(buf, 8192, 4096)) return false;
        std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.in_proj_z.weight", p, L);
        if (!resolve(buf, 4096, 4096)) return false;
        std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.in_proj_a.weight", p, L);
        if (!resolve(buf, 32, 4096)) return false;
        std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.in_proj_b.weight", p, L);
        if (!resolve(buf, 32, 4096)) return false;
        std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.out_proj.weight", p, L);
        if (!resolve(buf, 4096, 4096)) return false;
        std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.conv1d.weight", p, L);
        if (!resolve(buf, 8192, 4)) return false;
        std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.A_log", p, L);
        if (!resolve(buf, 32, 1)) return false;
        std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.dt_bias", p, L);
        if (!resolve(buf, 32, 1)) return false;
        std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.norm.weight", p, L);
        if (!resolve(buf, 128, 1)) return false;
    }
    return true;
}

bool Qwen35Engine::load(const std::string& quant_path) {
    // PROD fix: double-load leaked the old reader + scratch (raw new
    // overwrite), and mid-load `return false` left a half-loaded reader
    // behind. Free first; any failure rolls back to clean not-ok state.
    auto fail = [&]() -> bool {
        delete reader_; reader_ = nullptr;
        delete[] scratch_; scratch_ = nullptr;
        ok_ = false;
        return false;
    };
    delete reader_; reader_ = nullptr;
    delete[] scratch_; scratch_ = nullptr;
    ok_ = false;
    reader_ = new (std::nothrow) QUANTReader(quant_path);
    if (!reader_ || !reader_->valid()) return fail();

    if (!resolve("model.language_model.embed_tokens.weight", 248320, 4096)) return fail();
    if (!resolve("model.language_model.norm.weight", 4096, 4096)) return fail();
    if (!resolve("lm_head.weight", 248320, 4096)) return fail();

    for (int L = 0; L < 32; L++) {
        if (!load_layer(L)) return fail();
    }
    vocab_ = 248320;

    // small constant tensors
    Tensor t;
    char buf[256];
    const char* p = "model.language_model.layers.";
    for (int L = 0; L < 32; L++) {
        bool full = ((L + 1) % 4) == 0;
        if (full) {
            std::snprintf(buf, sizeof(buf), "%s%d.self_attn.q_norm.weight", p, L);
            t = reader_->read_tensor(buf);
            qnorm_w_[L].assign((const float*)t.data(), (const float*)t.data() + t.numel());
            std::snprintf(buf, sizeof(buf), "%s%d.self_attn.k_norm.weight", p, L);
            t = reader_->read_tensor(buf);
            knorm_w_[L].assign((const float*)t.data(), (const float*)t.data() + t.numel());
        } else {
            std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.A_log", p, L);
            t = reader_->read_tensor(buf);
            A_log_[L].assign((const float*)t.data(), (const float*)t.data() + t.numel());
            std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.dt_bias", p, L);
            t = reader_->read_tensor(buf);
            dt_bias_[L].assign((const float*)t.data(), (const float*)t.data() + t.numel());
            std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.norm.weight", p, L);
            t = reader_->read_tensor(buf);
            gnorm_w_[L].assign((const float*)t.data(), (const float*)t.data() + t.numel());
            std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.conv1d.weight", p, L);
            t = reader_->read_tensor(buf);
            conv_w_[L].assign((const float*)t.data(), (const float*)t.data() + t.numel());
        }
        std::snprintf(buf, sizeof(buf), "%s%d.input_layernorm.weight", p, L);
        t = reader_->read_tensor(buf);
        ln1_w_[L].assign((const float*)t.data(), (const float*)t.data() + t.numel());
        std::snprintf(buf, sizeof(buf), "%s%d.post_attention_layernorm.weight", p, L);
        t = reader_->read_tensor(buf);
        ln2_w_[L].assign((const float*)t.data(), (const float*)t.data() + t.numel());
    }
    t = reader_->read_tensor("model.language_model.norm.weight");
    final_norm_w_.assign((const float*)t.data(), (const float*)t.data() + t.numel());

    // state buffers (indexed by layer number L in 0..31; GLA layers subset)
    conv_state_.assign(32 * 8192 * 3, 0.0f);
    gla_state_.assign(32 * 32 * 128 * 128, 0.0f);
    kv_k_.assign(8 * 4 * max_kv_ * 256, 0.0f);
    kv_v_.assign(8 * 4 * max_kv_ * 256, 0.0f);
    h_.assign(HID, 0.0f);
    logits_.assign(vocab_, 0.0f);
    qkv_.assign(MAX_BATCH * 64 + 12288 + 12288, 0.0f); // mlp 2x12288 + prefill a/b B*64
    xb_.assign(MAX_BATCH * HID, 0.0f);
    ob_.assign(MAX_BATCH * (8192 + 1024 + 1024 + 4096), 0.0f);
    ga_.assign(MAX_BATCH * 12288, 0.0f);
    up_.assign(MAX_BATCH * 12288, 0.0f);
    tmp_.assign(MAX_BATCH * HID, 0.0f);
    float* scratch = new (std::nothrow) float[256];
    if (!scratch) return fail(); // ok_ stays false, reader rolled back
    scratch_ = scratch;
    ok_ = true;
    return true;
}

void Qwen35Engine::reset() {
    std::fill(conv_state_.begin(), conv_state_.end(), 0.0f);
    std::fill(gla_state_.begin(), gla_state_.end(), 0.0f);
    std::fill(kv_k_.begin(), kv_k_.end(), 0.0f);
    std::fill(kv_v_.begin(), kv_v_.end(), 0.0f);
    pos_ = 0;
}

// ---- GLA single-token step -------------------------------------------------
void Qwen35Engine::gla_step(int L, const float* x, float* y) {
    char buf[256];
    const char* p = "model.language_model.layers.";
    std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.in_proj_qkv.weight", p, L);
    gemm(*th(buf), x, qkv_.data(), 1);
    std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.in_proj_z.weight", p, L);
    gemm(*th(buf), x, qkv_.data() + 8192, 1);
    std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.in_proj_a.weight", p, L);
    gemm(*th(buf), x, tmp_.data(), 1);
    std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.in_proj_b.weight", p, L);
    gemm(*th(buf), x, tmp_.data() + 64, 1);

    const float* wa = tmp_.data();
    const float* wb = tmp_.data() + 64;
    const float* zz = qkv_.data() + 8192;
    const float* alog = A_log_[L].data();
    const float* dti = dt_bias_[L].data();
    const float* cw = conv_w_[L].data();
    float* cs = conv_state_.data() + (size_t)L * 8192 * 3;

    // depthwise causal conv1d (kernel 4, silu)
    float* qkv = qkv_.data();
    for (int c = 0; c < 8192; c++) {
        const float* w = cw + (size_t)c * 4;
        float v = w[0] * cs[c * 3] + w[1] * cs[c * 3 + 1] + w[2] * cs[c * 3 + 2] + w[3] * qkv[c];
        cs[c * 3] = cs[c * 3 + 1];
        cs[c * 3 + 1] = cs[c * 3 + 2];
        cs[c * 3 + 2] = qkv[c];
        qkv[c] = siluf(v);
    }

    // split q,k,v (16 k-heads, 32 v-heads), repeat q/k x2
    float qa[32][128], ka[32][128], va[32][128];
    for (int h = 0; h < 16; h++) {
        for (int d = 0; d < 128; d++) {
            float qv = qkv[(size_t)h * 128 + d];
            float kv = qkv[2048 + (size_t)h * 128 + d];
            qa[h][d] = qa[h + 16][d] = qv;
            ka[h][d] = ka[h + 16][d] = kv;
        }
    }
    for (int h = 0; h < 32; h++)
        for (int d = 0; d < 128; d++) va[h][d] = qkv[4096 + (size_t)h * 128 + d];

    float beta[32], g[32];
    for (int h = 0; h < 32; h++) {
        beta[h] = sigmoidf(wb[h]);
        g[h] = -std::exp(alog[h]) * softplusf(wa[h] + dti[h]);
    }

    const float inv = 1.0f / std::sqrt(128.0f);
    for (int h = 0; h < 32; h++) {
        float sq = 0.0f, sk = 0.0f;
        for (int d = 0; d < 128; d++) { sq += qa[h][d] * qa[h][d]; sk += ka[h][d] * ka[h][d]; }
        float rq = inv / std::sqrt(sq + 1e-6f);
        float rk = 1.0f / std::sqrt(sk + 1e-6f);
        for (int d = 0; d < 128; d++) { qa[h][d] *= rq; ka[h][d] *= rk; }
    }

    float* st = gla_state_.data() + (size_t)L * 32 * 128 * 128;
    for (int h = 0; h < 32; h++) {
        float e = std::exp(g[h]);
        float* st_h = st + (size_t)h * 128 * 128;
        for (int j = 0; j < 128; j++)
            for (int v = 0; v < 128; v++) st_h[(size_t)j * 128 + v] *= e;

        float kv_mem[128], delta[128], out[128];
        for (int v = 0; v < 128; v++) {
            float a = 0.0f;
            for (int j = 0; j < 128; j++) a += st_h[(size_t)j * 128 + v] * ka[h][j];
            kv_mem[v] = a;
        }
        for (int v = 0; v < 128; v++) delta[v] = (va[h][v] - kv_mem[v]) * beta[h];
        for (int j = 0; j < 128; j++)
            for (int v = 0; v < 128; v++) st_h[(size_t)j * 128 + v] += ka[h][j] * delta[v];
        for (int v = 0; v < 128; v++) {
            float a = 0.0f;
            for (int j = 0; j < 128; j++) a += st_h[(size_t)j * 128 + v] * qa[h][j];
            out[v] = a;
        }

        // RMSNormGated: norm * silu(z)
        float ms = 0.0f;
        for (int v = 0; v < 128; v++) ms += out[v] * out[v];
        float r = 1.0f / std::sqrt(ms / 128.0f + 1e-6f);
        const float* gw = gnorm_w_[L].data();
        for (int v = 0; v < 128; v++)
            qkv[4096 + (size_t)h * 128 + v] = gw[v] * (out[v] * r) * siluf(zz[(size_t)h * 128 + v]);
    }

    std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.out_proj.weight", p, L);
    gemm(*th(buf), qkv + 4096, y, 1);
}

// ---- GLA prefill token (projections already batched) ----------------------
void Qwen35Engine::gla_prefill(int L, const float* qkv_vec, const float* z_vec, const float* a_vec,
                               const float* b_vec, float* y) {
    const float* alog = A_log_[L].data();
    const float* dti = dt_bias_[L].data();
    const float* cw = conv_w_[L].data();
    float* cs = conv_state_.data() + (size_t)L * 8192 * 3;

    float qkv[8192];
    std::memcpy(qkv, qkv_vec, 8192 * 4);
    for (int c = 0; c < 8192; c++) {
        const float* w = cw + (size_t)c * 4;
        float v = w[0] * cs[c * 3] + w[1] * cs[c * 3 + 1] + w[2] * cs[c * 3 + 2] + w[3] * qkv[c];
        cs[c * 3] = cs[c * 3 + 1];
        cs[c * 3 + 1] = cs[c * 3 + 2];
        cs[c * 3 + 2] = qkv[c];
        qkv[c] = siluf(v);
    }

    float qa[32][128], ka[32][128], va[32][128];
    for (int h = 0; h < 16; h++) {
        for (int d = 0; d < 128; d++) {
            float qv = qkv[(size_t)h * 128 + d];
            float kv = qkv[2048 + (size_t)h * 128 + d];
            qa[h][d] = qa[h + 16][d] = qv;
            ka[h][d] = ka[h + 16][d] = kv;
        }
    }
    for (int h = 0; h < 32; h++)
        for (int d = 0; d < 128; d++) va[h][d] = qkv[4096 + (size_t)h * 128 + d];

    float beta[32], g[32];
    for (int h = 0; h < 32; h++) {
        beta[h] = sigmoidf(b_vec[h]);
        g[h] = -std::exp(alog[h]) * softplusf(a_vec[h] + dti[h]);
    }

    const float inv = 1.0f / std::sqrt(128.0f);
    for (int h = 0; h < 32; h++) {
        float sq = 0.0f, sk = 0.0f;
        for (int d = 0; d < 128; d++) { sq += qa[h][d] * qa[h][d]; sk += ka[h][d] * ka[h][d]; }
        float rq = inv / std::sqrt(sq + 1e-6f);
        float rk = 1.0f / std::sqrt(sk + 1e-6f);
        for (int d = 0; d < 128; d++) { qa[h][d] *= rq; ka[h][d] *= rk; }
    }

    float* st = gla_state_.data() + (size_t)L * 32 * 128 * 128;
    for (int h = 0; h < 32; h++) {
        float e = std::exp(g[h]);
        float* st_h = st + (size_t)h * 128 * 128;
        for (int j = 0; j < 128; j++)
            for (int v = 0; v < 128; v++) st_h[(size_t)j * 128 + v] *= e;

        float kv_mem[128], delta[128], out[128];
        for (int v = 0; v < 128; v++) {
            float a = 0.0f;
            for (int j = 0; j < 128; j++) a += st_h[(size_t)j * 128 + v] * ka[h][j];
            kv_mem[v] = a;
        }
        for (int v = 0; v < 128; v++) delta[v] = (va[h][v] - kv_mem[v]) * beta[h];
        for (int j = 0; j < 128; j++)
            for (int v = 0; v < 128; v++) st_h[(size_t)j * 128 + v] += ka[h][j] * delta[v];
        for (int v = 0; v < 128; v++) {
            float a = 0.0f;
            for (int j = 0; j < 128; j++) a += st_h[(size_t)j * 128 + v] * qa[h][j];
            out[v] = a;
        }

        float ms = 0.0f;
        for (int v = 0; v < 128; v++) ms += out[v] * out[v];
        float r = 1.0f / std::sqrt(ms / 128.0f + 1e-6f);
        const float* gw = gnorm_w_[L].data();
        for (int v = 0; v < 128; v++)
            y[(size_t)h * 128 + v] = gw[v] * (out[v] * r) * siluf(z_vec[(size_t)h * 128 + v]);
    }
}

// ---- full attention single-token step -------------------------------------
void Qwen35Engine::attn_step(int L, const float* x, float* y, int pos) {
    char buf[256];
    const char* p = "model.language_model.layers.";
    std::snprintf(buf, sizeof(buf), "%s%d.self_attn.q_proj.weight", p, L);
    gemm(*th(buf), x, qkv_.data(), 1);
    std::snprintf(buf, sizeof(buf), "%s%d.self_attn.k_proj.weight", p, L);
    gemm(*th(buf), x, qkv_.data() + 8192, 1);
    std::snprintf(buf, sizeof(buf), "%s%d.self_attn.v_proj.weight", p, L);
    gemm(*th(buf), x, qkv_.data() + 8192 + 1024, 1);

    const float* qg = qkv_.data();
    const float* kg = qkv_.data() + 8192;
    const float* vg = qkv_.data() + 8192 + 1024;
    const float* qw = qnorm_w_[L].data();
    const float* kw = knorm_w_[L].data();

    float qq[16][256], kk[4][256], vv[4][256], gate[16][256];
    for (int h = 0; h < 16; h++) {
        float ms = 0.0f;
        for (int d = 0; d < 256; d++) ms += qg[(size_t)h * 256 + d] * qg[(size_t)h * 256 + d];
        float r = 1.0f / std::sqrt(ms / 256.0f + 1e-6f);
        for (int d = 0; d < 256; d++) {
            qq[h][d] = qg[(size_t)h * 256 + d] * r * (1.0f + qw[d]);
            gate[h][d] = qg[4096 + (size_t)h * 256 + d];
        }
    }
    for (int h = 0; h < 4; h++) {
        float ms = 0.0f;
        for (int d = 0; d < 256; d++) ms += kg[(size_t)h * 256 + d] * kg[(size_t)h * 256 + d];
        float r = 1.0f / std::sqrt(ms / 256.0f + 1e-6f);
        for (int d = 0; d < 256; d++) {
            kk[h][d] = kg[(size_t)h * 256 + d] * r * (1.0f + kw[d]);
            vv[h][d] = vg[(size_t)h * 256 + d];
        }
    }

    // RoPE (rotary_dim 64, theta 1e7)
    float cosv[64], sinv[64];
    {
        const float theta = 1e7f;
        for (int i = 0; i < 32; i++) {
            float fr = (float)pos / std::pow(theta, 2.0f * i / 64.0f);
            cosv[i] = std::cos(fr);
            sinv[i] = std::sin(fr);
            cosv[i + 32] = cosv[i];
            sinv[i + 32] = sinv[i];
        }
    }
    for (int h = 0; h < 16; h++) {
        for (int i = 0; i < 32; i++) {
            float a = qq[h][i], b = qq[h][i + 32];
            qq[h][i] = a * cosv[i] - b * sinv[i];
            qq[h][i + 32] = a * sinv[i] + b * cosv[i];
        }
    }
    for (int h = 0; h < 4; h++) {
        for (int i = 0; i < 32; i++) {
            float a = kk[h][i], b = kk[h][i + 32];
            kk[h][i] = a * cosv[i] - b * sinv[i];
            kk[h][i + 32] = a * sinv[i] + b * cosv[i];
        }
    }

    int li = (L - 3) / 4;
    float* kc = kv_k_.data() + ((size_t)li * 4) * max_kv_ * 256;
    float* vc = kv_v_.data() + ((size_t)li * 4) * max_kv_ * 256;
    for (int h = 0; h < 4; h++) {
        std::memcpy(kc + ((size_t)h * max_kv_ + pos) * 256, kk[h], 256 * 4);
        std::memcpy(vc + ((size_t)h * max_kv_ + pos) * 256, vv[h], 256 * 4);
    }

    const float scale = 1.0f / std::sqrt(256.0f);
    float attn[4096], out[16][256];
    for (int h = 0; h < 16; h++) {
        int kvh = h / 4;
        const float* kbase = kc + (size_t)kvh * max_kv_ * 256;
        const float* vbase = vc + (size_t)kvh * max_kv_ * 256;
        float mx = -1e30f;
        for (int t = 0; t <= pos; t++) {
            const float* kt = kbase + (size_t)t * 256;
            float s = 0.0f;
            for (int d = 0; d < 256; d++) s += qq[h][d] * kt[d];
            attn[t] = s * scale;
            if (attn[t] > mx) mx = attn[t];
        }
        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) {
            attn[t] = std::exp(attn[t] - mx);
            sum += attn[t];
        }
        float inv = 1.0f / sum;
        for (int d = 0; d < 256; d++) {
            float a = 0.0f;
            for (int t = 0; t <= pos; t++) a += attn[t] * inv * vbase[(size_t)t * 256 + d];
            out[h][d] = a;
        }
    }

    for (int h = 0; h < 16; h++)
        for (int d = 0; d < 256; d++) qkv_[(size_t)h * 256 + d] = out[h][d] * sigmoidf(gate[h][d]);

    std::snprintf(buf, sizeof(buf), "%s%d.self_attn.o_proj.weight", p, L);
    gemm(*th(buf), qkv_.data(), y, 1);
}

// ---- full attention prefill token -----------------------------------------
void Qwen35Engine::attn_prefill(int L, const float* qvec, const float* kvec, const float* vvec,
                                int pos, float* y) {
    const float* qw = qnorm_w_[L].data();
    const float* kw = knorm_w_[L].data();

    float qq[16][256], kk[4][256], vv[4][256], gate[16][256];
    for (int h = 0; h < 16; h++) {
        float ms = 0.0f;
        for (int d = 0; d < 256; d++) ms += qvec[(size_t)h * 256 + d] * qvec[(size_t)h * 256 + d];
        float r = 1.0f / std::sqrt(ms / 256.0f + 1e-6f);
        for (int d = 0; d < 256; d++) {
            qq[h][d] = qvec[(size_t)h * 256 + d] * r * (1.0f + qw[d]);
            gate[h][d] = qvec[4096 + (size_t)h * 256 + d];
        }
    }
    for (int h = 0; h < 4; h++) {
        float ms = 0.0f;
        for (int d = 0; d < 256; d++) ms += kvec[(size_t)h * 256 + d] * kvec[(size_t)h * 256 + d];
        float r = 1.0f / std::sqrt(ms / 256.0f + 1e-6f);
        for (int d = 0; d < 256; d++) {
            kk[h][d] = kvec[(size_t)h * 256 + d] * r * (1.0f + kw[d]);
            vv[h][d] = vvec[(size_t)h * 256 + d];
        }
    }

    float cosv[64], sinv[64];
    {
        const float theta = 1e7f;
        for (int i = 0; i < 32; i++) {
            float fr = (float)pos / std::pow(theta, 2.0f * i / 64.0f);
            cosv[i] = std::cos(fr);
            sinv[i] = std::sin(fr);
            cosv[i + 32] = cosv[i];
            sinv[i + 32] = sinv[i];
        }
    }
    for (int h = 0; h < 16; h++) {
        for (int i = 0; i < 32; i++) {
            float a = qq[h][i], b = qq[h][i + 32];
            qq[h][i] = a * cosv[i] - b * sinv[i];
            qq[h][i + 32] = a * sinv[i] + b * cosv[i];
        }
    }
    for (int h = 0; h < 4; h++) {
        for (int i = 0; i < 32; i++) {
            float a = kk[h][i], b = kk[h][i + 32];
            kk[h][i] = a * cosv[i] - b * sinv[i];
            kk[h][i + 32] = a * sinv[i] + b * cosv[i];
        }
    }

    int li = (L - 3) / 4;
    float* kc = kv_k_.data() + ((size_t)li * 4) * max_kv_ * 256;
    float* vc = kv_v_.data() + ((size_t)li * 4) * max_kv_ * 256;
    for (int h = 0; h < 4; h++) {
        std::memcpy(kc + ((size_t)h * max_kv_ + pos) * 256, kk[h], 256 * 4);
        std::memcpy(vc + ((size_t)h * max_kv_ + pos) * 256, vv[h], 256 * 4);
    }

    const float scale = 1.0f / std::sqrt(256.0f);
    float attn[4096];
    for (int h = 0; h < 16; h++) {
        int kvh = h / 4;
        const float* kbase = kc + (size_t)kvh * max_kv_ * 256;
        const float* vbase = vc + (size_t)kvh * max_kv_ * 256;
        float mx = -1e30f;
        for (int t = 0; t <= pos; t++) {
            const float* kt = kbase + (size_t)t * 256;
            float s = 0.0f;
            for (int d = 0; d < 256; d++) s += qq[h][d] * kt[d];
            attn[t] = s * scale;
            if (attn[t] > mx) mx = attn[t];
        }
        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) {
            attn[t] = std::exp(attn[t] - mx);
            sum += attn[t];
        }
        float inv = 1.0f / sum;
        for (int d = 0; d < 256; d++) {
            float a = 0.0f;
            for (int t = 0; t <= pos; t++) a += attn[t] * inv * vbase[(size_t)t * 256 + d];
            y[(size_t)h * 256 + d] = a * sigmoidf(gate[h][d]);
        }
    }
}

// ---- MLP ------------------------------------------------------------------
void Qwen35Engine::mlp_step(int L, const float* x, float* y) {
    char buf[256];
    const char* p = "model.language_model.layers.";
    std::snprintf(buf, sizeof(buf), "%s%d.mlp.gate_proj.weight", p, L);
    gemm(*th(buf), x, qkv_.data(), 1);
    std::snprintf(buf, sizeof(buf), "%s%d.mlp.up_proj.weight", p, L);
    gemm(*th(buf), x, qkv_.data() + 12288, 1);
    for (int i = 0; i < 12288; i++) qkv_[i] = siluf(qkv_[i]) * qkv_[12288 + i];
    std::snprintf(buf, sizeof(buf), "%s%d.mlp.down_proj.weight", p, L);
    gemm(*th(buf), qkv_.data(), y, 1);
}

// ---- logits ---------------------------------------------------------------
void Qwen35Engine::run_logits(const float* h) {
    const Th& t = th_.at("lm_head.weight");
    gemm(t, h, logits_.data(), 1);
}

// ---- prefill --------------------------------------------------------------
void Qwen35Engine::prefill(const std::vector<int>& ids) {
    const Th& emb = th_.at("model.language_model.embed_tokens.weight");
    size_t i = 0;
    const size_t n = ids.size();
    while (i < n) {
        int B = (int)std::min((size_t)MAX_BATCH, n - i);
        float* X = xb_.data();
        for (int b = 0; b < B; b++) {
            int tok = ids[i + (size_t)b];
            float* row = X + (size_t)b * 4096;
            for (int blk = 0; blk < 16; blk++) {
                decode_block(*reader_, emb.start + (uint32_t)tok * 16 + (uint32_t)blk, 256, row + (size_t)blk * 256);
            }
        }

        for (int L = 0; L < 32; L++) {
            std::fprintf(stderr, "L%d ", L);
            std::fflush(stderr);
            char buf[256];
            const char* p = "model.language_model.layers.";
            bool full = ((L + 1) % 4) == 0;

            for (int b = 0; b < B; b++)
                rmsnorm(tmp_.data() + (size_t)b * 4096, X + (size_t)b * 4096, ln1_w_[L].data(), 4096);

            if (full) {
                if (!skip_full_) {
                std::snprintf(buf, sizeof(buf), "%s%d.self_attn.q_proj.weight", p, L);
                gemm(*th(buf), tmp_.data(), ob_.data(), B);
                std::snprintf(buf, sizeof(buf), "%s%d.self_attn.k_proj.weight", p, L);
                gemm(*th(buf), tmp_.data(), ob_.data() + (size_t)B * 8192, B);
                std::snprintf(buf, sizeof(buf), "%s%d.self_attn.v_proj.weight", p, L);
                gemm(*th(buf), tmp_.data(), ob_.data() + (size_t)B * (8192 + 1024), B);
                float* outb = ob_.data() + (size_t)B * (8192 + 1024 + 1024);
                for (int b = 0; b < B; b++) {
                    attn_prefill(L,
                                 ob_.data() + (size_t)b * 8192,
                                 ob_.data() + (size_t)B * 8192 + (size_t)b * 1024,
                                 ob_.data() + (size_t)B * (8192 + 1024) + (size_t)b * 1024,
                                 (int)pos_ + b, outb + (size_t)b * 4096);
                }
                std::snprintf(buf, sizeof(buf), "%s%d.self_attn.o_proj.weight", p, L);
                gemm(*th(buf), outb, tmp_.data(), B);
                } else {
                    for (int b = 0; b < B; b++)
                        std::memcpy(tmp_.data() + (size_t)b * 4096, X + (size_t)b * 4096, 4096 * 4);
                }
            } else if (!skip_gla_) {
                std::fprintf(stderr, "g1 "); std::fflush(stderr);
                std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.in_proj_qkv.weight", p, L);
                gemm(*th(buf), tmp_.data(), ob_.data(), B);
                std::fprintf(stderr, "g2 "); std::fflush(stderr);
                std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.in_proj_z.weight", p, L);
                gemm(*th(buf), tmp_.data(), ob_.data() + (size_t)B * 8192, B);
                std::fprintf(stderr, "g3 "); std::fflush(stderr);
                std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.in_proj_a.weight", p, L);
                gemm(*th(buf), tmp_.data(), qkv_.data(), B);
                std::fprintf(stderr, "g4 "); std::fflush(stderr);
                std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.in_proj_b.weight", p, L);
                gemm(*th(buf), tmp_.data(), qkv_.data() + (size_t)B * 32, B);
                std::fprintf(stderr, "g5 "); std::fflush(stderr);
                for (int b = 0; b < B; b++) {
                    gla_prefill(L,
                                ob_.data() + (size_t)b * 8192,
                                ob_.data() + (size_t)B * 8192 + (size_t)b * 4096,
                                qkv_.data() + (size_t)b * 32,
                                qkv_.data() + (size_t)B * 32 + (size_t)b * 32,
                                tmp_.data() + (size_t)b * 4096);
                }
                std::snprintf(buf, sizeof(buf), "%s%d.linear_attn.out_proj.weight", p, L);
                gemm(*th(buf), tmp_.data(), ob_.data(), B);
            } else {
                for (int b = 0; b < B; b++)
                    std::memcpy(tmp_.data() + (size_t)b * 4096, X + (size_t)b * 4096, 4096 * 4);
            }

            const float* src = full ? tmp_.data() : ob_.data();
            for (int b = 0; b < B; b++) {
                float* hb = X + (size_t)b * 4096;
                const float* s = src + (size_t)b * 4096;
                for (int d = 0; d < 4096; d++) hb[d] += s[d];
            }
            if (L % 4 == 0) {
                double ms = 0;
                for (int d = 0; d < 4096; d++) ms += (double)X[d] * X[d];
                std::fprintf(stderr, "[h%d rms %.3f] ", L, (float)std::sqrt(ms / 4096));
                std::fflush(stderr);
            }
            if (trace_.size() != 32 * 4096) trace_.assign(32 * 4096, 0.0f);
            std::memcpy(trace_.data() + (size_t)L * 4096, X + (size_t)(B - 1) * 4096, 4096 * 4);
            for (int b = 0; b < B; b++)
                rmsnorm(tmp_.data() + (size_t)b * 4096, X + (size_t)b * 4096, ln2_w_[L].data(), 4096);

            std::snprintf(buf, sizeof(buf), "%s%d.mlp.gate_proj.weight", p, L);
            gemm(*th(buf), tmp_.data(), ga_.data(), B);
            std::snprintf(buf, sizeof(buf), "%s%d.mlp.up_proj.weight", p, L);
            gemm(*th(buf), tmp_.data(), up_.data(), B);
            for (int i = 0; i < B * 12288; i++) ga_[i] = siluf(ga_[i]) * up_[i];
            std::snprintf(buf, sizeof(buf), "%s%d.mlp.down_proj.weight", p, L);
            gemm(*th(buf), ga_.data(), ob_.data(), B);
            for (int b = 0; b < B; b++) {
                float* hb = X + (size_t)b * 4096;
                const float* m = ob_.data() + (size_t)b * 4096;
                for (int d = 0; d < 4096; d++) hb[d] += m[d];
            }
        }

        // final norm + logits (only last token needed)
        for (int b = 0; b < B; b++)
            rmsnorm(tmp_.data() + (size_t)b * 4096, X + (size_t)b * 4096, final_norm_w_.data(), 4096);
        {
            double ms = 0;
            for (int d = 0; d < 4096; d++) ms += (double)tmp_[d] * tmp_[d];
            std::fprintf(stderr, "\n[final rms %.3f]\n", (float)std::sqrt(ms / 4096));
        }
        if (last_h_.size() != 4096) last_h_.assign(4096, 0.0f);
        std::memcpy(last_h_.data(), tmp_.data(), 4096 * 4);
        const Th& lh = th_.at("lm_head.weight");
        gemm(lh, tmp_.data() + (size_t)(B - 1) * 4096, logits_.data(), 1);

        pos_ += (size_t)B;
        i += (size_t)B;
    }
}

// ---- decode step ----------------------------------------------------------
void Qwen35Engine::append_token(int id) {
    const Th& emb = th_.at("model.language_model.embed_tokens.weight");
    for (int blk = 0; blk < 16; blk++) {
        decode_block(*reader_, emb.start + (uint32_t)id * 16 + (uint32_t)blk, 256, h_.data() + (size_t)blk * 256);
    }
    for (int L = 0; L < 32; L++) {
        rmsnorm(tmp_.data(), h_.data(), ln1_w_[L].data(), 4096);
        if (((L + 1) % 4) == 0) {
            if (skip_full_) { std::memset(ob_.data(), 0, 4096 * 4); }
            else attn_step(L, tmp_.data(), ob_.data(), (int)pos_);
        } else {
            if (skip_gla_) { std::memset(ob_.data(), 0, 4096 * 4); }
            else gla_step(L, tmp_.data(), ob_.data());
        }
        for (int d = 0; d < 4096; d++) h_[d] += ob_[d];
        if (trace_.size() != 32 * 4096) trace_.assign(32 * 4096, 0.0f);
        std::memcpy(trace_.data() + (size_t)L * 4096, h_.data(), 4096 * 4);
        rmsnorm(tmp_.data(), h_.data(), ln2_w_[L].data(), 4096);
        mlp_step(L, tmp_.data(), ob_.data());
        for (int d = 0; d < 4096; d++) h_[d] += ob_[d];
    }
    rmsnorm(tmp_.data(), h_.data(), final_norm_w_.data(), 4096);
    run_logits(tmp_.data());
    if (last_h_.size() != 4096) last_h_.assign(4096, 0.0f);
    std::memcpy(last_h_.data(), tmp_.data(), 4096 * 4);
    pos_++;
}

int Qwen35Engine::sample(float temperature, int top_k) {
    if (top_k <= 0 || top_k > vocab_) top_k = vocab_;
    struct Cand { float v; int id; };
    int nt = (int)std::thread::hardware_concurrency();
    if (nt < 1) nt = 1;
    if (nt > 8) nt = 8;
    std::vector<Cand> finals;
    {
        int chunk = (vocab_ + nt - 1) / nt;
        std::vector<std::thread> threads;
        std::vector<std::vector<Cand>> heaps(nt);
        for (int tid = 0; tid < nt; tid++) {
            int i0 = tid * chunk;
            int i1 = std::min(vocab_, i0 + chunk);
            if (i0 >= i1) continue;
            threads.emplace_back([&, tid, i0, i1]() {
                auto& h2 = heaps[tid];
                h2.reserve(top_k + 1);
                auto cmp = [](const Cand& a, const Cand& b) { return a.v > b.v; };
                for (int i = i0; i < i1; i++) {
                    float v = temperature > 0.0f ? logits_[i] / temperature : logits_[i];
                    if ((int)h2.size() < top_k) {
                        h2.push_back({v, i});
                        std::push_heap(h2.begin(), h2.end(), cmp);
                    } else if (v > h2.front().v) {
                        std::pop_heap(h2.begin(), h2.end(), cmp);
                        h2.back() = {v, i};
                        std::push_heap(h2.begin(), h2.end(), cmp);
                    }
                }
            });
        }
        for (auto& t : threads) t.join();
        for (auto& h2 : heaps)
            for (auto& c : h2) finals.push_back(c);
    }
    if (finals.empty()) return 0;

    float mx = finals[0].v;
    for (auto& c : finals) if (c.v > mx) mx = c.v;
    double sum = 0.0;
    for (auto& c : finals) { c.v = std::exp(c.v - mx); sum += c.v; }
    rng_ ^= rng_ << 13; rng_ ^= rng_ >> 7; rng_ ^= rng_ << 17;
    double r = (double)(rng_ & 0x7FFFFFFFFFFFFFFFull) / 9.2233720368547758e18 * sum;
    double acc = 0.0;
    for (auto& c : finals) {
        acc += c.v;
        if (r < acc) return c.id;
    }
    return finals.back().id;
}

} // namespace quant

