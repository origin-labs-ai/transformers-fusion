// quant_refcheck — ground-truth check against original BF16 safetensors.
// Tests:
//  1) lm_head logits from ORIGINAL weights on the engine's final hidden state.
//  2) layer-0 GLA reference vs the engine's layer-0 output (trace0.bin).
#include "quant/json_parser.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <unordered_map>
#include <thread>
#include <algorithm>

static float bf16_to_fp32(uint16_t b) {
    uint32_t s = (uint32_t)(b & 0x8000u) << 16;
    uint32_t e = (uint32_t)(b & 0x7F80u) << 16;
    uint32_t m = (uint32_t)(b & 0x007Fu) << 16;
    if (e == 0 && m != 0) {
        // subnormal
        uint32_t exp = 127 - 126;
        while (!(m & 0x800000u)) { m <<= 1; exp--; }
        m &= 0x7FFFFFu;
        e = ((uint32_t)exp + 127) << 23;
    }
    float f;
    uint32_t bits = s | e | m;
    std::memcpy(&f, &bits, 4);
    return f;
}

struct TensorInfo {
    std::string shard;
    int64_t off0;
    int64_t numel;
    std::vector<int64_t> shape;
};

class SfReader {
public:
    bool init(const std::string& model_dir, const std::string& index_path) {
        dir_ = model_dir;
        std::string json;
        FILE* f = fopen(index_path.c_str(), "rb");
        if (!f) return false;
        char buf[1 << 16];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) json.append(buf, n);
        fclose(f);
        std::string err;
        quant::JsonValue root = quant::JsonValue::parse(json, &err);
        if (!err.empty() || !root.is_object()) {
            fprintf(stderr, "refcheck: bad index json: %s\n", err.c_str());
            return false;
        }
        const quant::JsonValue& wm = root["weight_map"];
        for (const auto& kv : wm.obj) {
            TensorInfo ti;
            ti.shard = kv.second.as_string();
            auto it = meta_.find(ti.shard);
            map_[kv.first] = (int)shards_.size();
            if (it == meta_.end()) {
                shards_.push_back(ti.shard);
            }
        }
        // parse each shard header once
        for (size_t i = 0; i < shards_.size(); i++) {
            if (!parse_shard_header((int)i)) return false;
        }
        for (const auto& kv : wm.obj) {
            TensorInfo& ti = info_[kv.first];
            ti.shard = kv.second.as_string();
            int si = map_[kv.first];
            const auto& hdr = headers_[si];
            auto tit = hdr.obj.find(kv.first);
            if (tit == hdr.obj.end()) { fprintf(stderr, "refcheck: tensor %s not in shard\n", kv.first.c_str()); return false; }
            ti.off0 = tit->second["data_offsets"][0].as_int();
            ti.numel = tit->second["data_offsets"][1].as_int() - tit->second["data_offsets"][0].as_int();
            ti.numel /= 2; // bf16
            for (const auto& s : tit->second["shape"].arr) ti.shape.push_back(s.as_int());
        }
        return true;
    }

    // read numel bf16 elements at elem_offset (row-major), convert to fp32
    bool read(const std::string& name, int64_t elem_offset, int64_t count, std::vector<float>& out) {
        auto it = info_.find(name);
        if (it == info_.end()) { fprintf(stderr, "refcheck: no tensor %s\n", name.c_str()); return false; }
        const TensorInfo& ti = it->second;
        int si = map_[ti.shard];
        const quant::JsonValue& hdr = headers_[si];
        int64_t data_start = hdr["data_offsets"][1].as_int() + hdr["data_offsets"][0].as_int() - hdr["data_offsets"][0].as_int();
        data_start = 8 + (int64_t)hdr_json_len_[si];
        int64_t byte_off = ti.off0 + elem_offset * 2;
        // BUGFIX (bug census): "\\" separator breaks POSIX + unchecked fseek
        // (wrong offset → silent bad comparison). filesystem join + fseek check.
        std::filesystem::path shard_path = std::filesystem::path(dir_) / ti.shard;
        FILE* f = fopen(shard_path.string().c_str(), "rb");
        if (!f) { fprintf(stderr, "refcheck: can't open shard %s\n", ti.shard.c_str()); return false; }
        if (fseek(f, data_start + byte_off, SEEK_SET) != 0) {
            fprintf(stderr, "refcheck: seek failed\n");
            fclose(f);
            return false;
        }
        std::vector<uint16_t> raw((size_t)count);
        size_t rd = fread(raw.data(), 2, (size_t)count, f);
        fclose(f);
        if (rd != (size_t)count) { fprintf(stderr, "refcheck: short read %zu/%lld\n", rd, (long long)count); return false; }
        out.resize((size_t)count);
        for (int64_t i = 0; i < count; i++) out[(size_t)i] = bf16_to_fp32(raw[(size_t)i]);
        return true;
    }

    bool shape_of(const std::string& name, int64_t& rows, int64_t& cols) {
        auto it = info_.find(name);
        if (it == info_.end()) return false;
        if (it->second.shape.size() == 1) { rows = it->second.shape[0]; cols = 1; }
        else if (it->second.shape.size() == 2) { rows = it->second.shape[0]; cols = it->second.shape[1]; }
        else if (it->second.shape.size() == 3) { rows = it->second.shape[0]; cols = it->second.shape[1] * it->second.shape[2]; }
        else return false;
        return true;
    }

private:
    bool parse_shard_header(int si) {
        // BUGFIX (bug census): same "\\" join breakage as read().
        std::filesystem::path shard_path = std::filesystem::path(dir_) / shards_[si];
        FILE* f = fopen(shard_path.string().c_str(), "rb");
        if (!f) return false;
        uint64_t len;
        if (fread(&len, 8, 1, f) != 1) { fclose(f); return false; }
        std::string json;
        json.resize((size_t)len);
        if (fread(&json[0], 1, (size_t)len, f) != len) { fclose(f); return false; }
        fclose(f);
        std::string err;
        headers_.push_back(quant::JsonValue::parse(json, &err));
        if (!err.empty()) { fprintf(stderr, "refcheck: shard header parse: %s\n", err.c_str()); return false; }
        hdr_json_len_.push_back((int64_t)len);
        return true;
    }

    std::string dir_;
    std::vector<std::string> shards_;
    std::unordered_map<std::string, int> map_;
    std::vector<quant::JsonValue> headers_;
    std::vector<int64_t> hdr_json_len_;
    std::unordered_map<std::string, TensorInfo> info_;
};

static float rsqrt(float x) { return 1.0f / std::sqrt(x); }

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: quant_refcheck <model_dir> <engine_h.bin> <trace0.bin>\n");
        return 1;
    }
    std::string dir = argv[1];
    SfReader r;
    if (!r.init(dir, dir + "\\model.safetensors.index.json")) { fprintf(stderr, "refcheck: init failed\n"); return 1; }

    // ---- load engine outputs ----
    std::vector<float> eng_h(4096), eng_t0(4096);
    {
        FILE* f = fopen(argv[2], "rb");
        if (!f) { fprintf(stderr, "refcheck: no %s\n", argv[2]); return 1; }
        if (fread(eng_h.data(), 4, 4096, f) != 4096) { fclose(f); return 1; }
        fclose(f);
    }
    {
        FILE* f = fopen(argv[3], "rb");
        if (!f) { fprintf(stderr, "refcheck: no %s\n", argv[3]); return 1; }
        if (fread(eng_t0.data(), 4, 4096, f) != 4096) { fclose(f); return 1; }
        fclose(f);
    }

    // ---- test 1: original lm_head logits on engine h ----
    {
        int64_t rows, cols;
        r.shape_of("lm_head.weight", rows, cols);
        fprintf(stderr, "[lm_head] %lld x %lld\n", (long long)rows, (long long)cols);
        int nt = (int)std::thread::hardware_concurrency();
        if (nt < 1) nt = 1; if (nt > 8) nt = 8;
        std::vector<float> logits((size_t)rows);
        std::vector<std::thread> ths;
        int64_t chunk = (rows + nt - 1) / nt;
        for (int tid = 0; tid < nt; tid++) {
            int64_t r0 = tid * chunk, r1 = std::min(rows, r0 + chunk);
            if (r0 >= r1) continue;
            ths.emplace_back([&, r0, r1]() {
                std::vector<float> row;
                for (int64_t i = r0; i < r1; i++) {
                    r.read("lm_head.weight", i * cols, cols, row);
                    float s = 0;
                    for (int64_t j = 0; j < cols; j++) s += row[(size_t)j] * eng_h[(size_t)j];
                    logits[(size_t)i] = s;
                }
            });
        }
        for (auto& t : ths) t.join();
        float mx = -1e30f; int mxi = 0;
        double sum = 0;
        for (int64_t i = 0; i < rows; i++) {
            if (logits[(size_t)i] > mx) { mx = logits[(size_t)i]; mxi = (int)i; }
            sum += (double)logits[(size_t)i] * logits[(size_t)i];
        }
        fprintf(stderr, "[orig lm_head] argmax=%d mx=%.3f rms=%.3f\n", mxi, mx, (float)std::sqrt(sum / rows));
        std::vector<std::pair<float, int>> top;
        for (int64_t i = 0; i < rows; i++) top.push_back({logits[(size_t)i], (int)i});
        std::partial_sort(top.begin(), top.begin() + std::min(10, (int)top.size()), top.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        for (int q = 0; q < 10; q++) fprintf(stderr, "[orig top%d] %d %.3f\n", q + 1, top[q].second, top[q].first);
    }

    // ---- test 2: layer-0 GLA reference ----
    {
        const int HID = 4096, NQKV = 16384, NZ = 4096;
        std::vector<float> embed_row, ln1, wqkv, wz, wa, wb, conv, alog, dtb, norm, outproj;
        r.read("model.language_model.embed_tokens.weight", 9419 * 4096, 4096, embed_row);
        r.read("model.language_model.layers.0.input_layernorm.weight", 0, 4096, ln1);
        r.read("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", 0, NQKV * HID, wqkv);
        r.read("model.language_model.layers.0.linear_attn.in_proj_z.weight", 0, NZ * HID, wz);
        r.read("model.language_model.layers.0.linear_attn.in_proj_a.weight", 0, 32 * HID, wa);
        r.read("model.language_model.layers.0.linear_attn.in_proj_b.weight", 0, 32 * HID, wb);
        r.read("model.language_model.layers.0.linear_attn.conv1d.weight", 0, 8192 * 4, conv);
        r.read("model.language_model.layers.0.linear_attn.A_log", 0, 32, alog);
        r.read("model.language_model.layers.0.linear_attn.dt_bias", 0, 32, dtb);
        r.read("model.language_model.layers.0.linear_attn.norm.weight", 0, 128, norm);
        r.read("model.language_model.layers.0.linear_attn.out_proj.weight", 0, HID * HID, outproj);

        // ln1
        std::vector<float> h(4096), qkv(8192), z(4096), a(32), b(32);
        {
            double ms = 0;
            for (int i = 0; i < HID; i++) ms += (double)embed_row[i] * embed_row[i];
            float rinv = rsqrt((float)(ms / HID) + 1e-6f);
            for (int i = 0; i < HID; i++) h[i] = embed_row[i] * rinv * (1.0f + ln1[i]);
        }
        // projections
        auto gemm = [](const std::vector<float>& W, const std::vector<float>& x, int rows, int cols, std::vector<float>& y) {
            y.assign((size_t)rows, 0.0f);
            for (int rr = 0; rr < rows; rr++) {
                float s = 0;
                for (int c = 0; c < cols; c++) s += W[(size_t)rr * cols + c] * x[c];
                y[rr] = s;
            }
        };
        gemm(wqkv, h, NQKV, HID, qkv);
        std::vector<float> qkvz;
        gemm(wz, h, NZ, HID, z);
        gemm(wa, h, 32, HID, a);
        gemm(wb, h, 32, HID, b);
        // conv1d t=0: silu(w[3]*x) (padding zeros)
        for (int c = 0; c < 8192; c++) {
            float v = conv[(size_t)c * 4 + 3] * qkv[c];
            qkv[c] = v * (1.0f / (1.0f + std::exp(-v)));
        }
        // split + repeat
        float qa[32][128], ka[32][128], va[32][128];
        for (int hh = 0; hh < 16; hh++) {
            for (int d = 0; d < 128; d++) {
                float qv = qkv[(size_t)hh * 128 + d];
                float kv = qkv[2048 + (size_t)hh * 128 + d];
                qa[hh][d] = qa[hh + 16][d] = qv;
                ka[hh][d] = ka[hh + 16][d] = kv;
            }
        }
        for (int hh = 0; hh < 32; hh++)
            for (int d = 0; d < 128; d++) va[hh][d] = qkv[4096 + (size_t)hh * 128 + d];
        float beta[32], g[32];
        for (int hh = 0; hh < 32; hh++) {
            beta[hh] = 1.0f / (1.0f + std::exp(-b[hh]));
            g[hh] = -std::exp(alog[hh]) * std::log1p(std::exp(a[hh] + dtb[hh]));
        }
        const float inv = 1.0f / std::sqrt(128.0f);
        for (int hh = 0; hh < 32; hh++) {
            double sq = 0, sk = 0;
            for (int d = 0; d < 128; d++) { sq += (double)qa[hh][d] * qa[hh][d]; sk += (double)ka[hh][d] * ka[hh][d]; }
            float rq = inv / std::sqrt((float)sq + 1e-6f);
            float rk = 1.0f / std::sqrt((float)sk + 1e-6f);
            for (int d = 0; d < 128; d++) { qa[hh][d] *= rq; ka[hh][d] *= rk; }
        }
        std::vector<float> y(4096);
        for (int hh = 0; hh < 32; hh++) {
            float e = std::exp(g[hh]);
            float st[128][128];
            for (int j = 0; j < 128; j++) for (int v = 0; v < 128; v++) st[j][v] = 0.0f;
            float kv_mem[128], delta[128], out[128];
            for (int v = 0; v < 128; v++) kv_mem[v] = 0.0f;
            for (int v = 0; v < 128; v++) delta[v] = va[hh][v] * beta[hh];
            for (int j = 0; j < 128; j++) for (int v = 0; v < 128; v++) st[j][v] += ka[hh][j] * delta[v];
            for (int v = 0; v < 128; v++) {
                float s = 0;
                for (int j = 0; j < 128; j++) s += st[j][v] * qa[hh][j];
                out[v] = s;
            }
            float ms = 0;
            for (int v = 0; v < 128; v++) ms += out[v] * out[v];
            float rr = rsqrt(ms / 128.0f + 1e-6f);
            for (int v = 0; v < 128; v++)
                y[(size_t)hh * 128 + v] = norm[v] * out[v] * rr * (z[(size_t)hh * 128 + v] /
                    (1.0f + std::exp(-z[(size_t)hh * 128 + v])));
        }
        std::vector<float> gla_out(4096);
        gemm(outproj, y, HID, HID, gla_out);
        std::vector<float> h0(4096);
        for (int i = 0; i < HID; i++) h0[i] = embed_row[i] + gla_out[i];

        // compare
        auto stats = [](const std::vector<float>& a, const std::vector<float>& b) {
            double d = 0, ms = 0;
            double ca = 0, cb = 0, cab = 0;
            for (size_t i = 0; i < a.size(); i++) {
                d += std::abs((double)a[i] - b[i]);
                ms += (double)a[i] * a[i];
                ca += a[i]; cb += b[i]; cab += (double)a[i] * b[i];
            }
            double ra = 0, rb = 0;
            for (size_t i = 0; i < a.size(); i++) { ra += (double)a[i] * a[i]; rb += (double)b[i] * b[i]; }
            double corr = ra > 0 && rb > 0 ? cab / (std::sqrt(ra) * std::sqrt(rb)) : 0.0;
            return std::make_tuple(d, std::sqrt(ms / a.size()), corr);
        };
        auto [d0, r0, c0] = stats(h0, eng_t0);
        fprintf(stderr, "[layer0] ref rms=%.4f eng rms=%.4f maxabsdiff=%.6f corr=%.4f\n",
                r0, std::sqrt([&]() { double ms = 0; for (auto v : eng_t0) ms += (double)v * v; return ms / 4096; }()),
                d0, c0);
        // also h vec compare (engine h vs ref... engine h is post-32-layer; skip)
    }
    return 0;
}
