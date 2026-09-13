// ============================================================================
// safetensors_bridge.cpp — Safetensors JSON+raw parser -> AdapterTensors -> QUANT
// ============================================================================
#include "adapters/safetensors_bridge.h"
#include "adapters/adapter_core.h"

#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <algorithm>

namespace quant {
namespace adapters {

// ── Minimal JSON helpers (safetensors header is flat & predictable) ──────────

static std::string read_until(const std::string& s, size_t& pos, char delim) {
    size_t start = pos;
    while (pos < s.size() && s[pos] != delim) pos++;
    return s.substr(start, pos - start);
}

static std::string extract_json_string(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto p = json.find(needle);
    if (p == std::string::npos) return {};
    p = json.find(':', p + needle.size());
    if (p == std::string::npos) return {};
    // skip ws
    p++; while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n' || json[p] == '\r')) p++;
    if (p >= json.size() || json[p] != '"') return {};
    p++;
    std::string val;
    while (p < json.size()) {
        if (json[p] == '\\' && p + 1 < json.size()) { val += json[p+1]; p += 2; continue; }
        if (json[p] == '"') { p++; break; }
        val += json[p++];
    }
    return val;
}

static int64_t extract_json_int(const std::string& json, const std::string& key) {
    auto p = json.find("\"" + key + "\"");
    if (p == std::string::npos) return 0;
    p = json.find(':', p + key.size() + 2);
    if (p == std::string::npos) return 0;
    p++;
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) p++;
    return std::atoll(json.c_str() + p);
}

// ── Per-tensor dtype info ─────────────────────────────────────────────────────

struct STDTypeInfo {
    const char* name;
    int bytes_per_elem;
};

static const STDTypeInfo st_dtypes[] = {
    { "F32",    4 },
    { "F16",    2 },
    { "BF16",   2 },
    { "I64",    8 },
    { "I32",    4 },
    { "I16",    2 },
    { "I8",     1 },
    { "U8",     1 },
    { "BOOL",   1 },
    { "F8_E4M3", 1 },
    { "F8_E5M2", 1 },
    { nullptr,  0 },
};

static int st_bytes_per_elem(const std::string& dtype_name) {
    for (const STDTypeInfo* p = st_dtypes; p->name; ++p)
        if (dtype_name == p->name) return p->bytes_per_elem;
    return 0;
}

// ── Dequantizers ─────────────────────────────────────────────────────────────

static float st_f16_to_f32(uint16_t h) {
    int sign = (h >> 15) & 1;
    int exp  = (h >> 10) & 0x1F;
    int mant = h & 0x3FF;
    if (exp == 0) return (mant ? (float)mant / 16384.0f * 2.0f : 0.0f) * (sign ? -1.0f : 1.0f);
    if (exp == 31) return mant ? NAN : INFINITY;
    return ((float)(mant | 0x400) / 8388608.0f) * std::ldexp(1.0f, exp - 15) * (sign ? -1.0f : 1.0f);
}

static float st_bf16_to_f32(uint16_t b) {
    uint32_t bits = ((uint32_t)b) << 16;
    float f; std::memcpy(&f, &bits, 4);
    return f;
}

static float st_fp8_e4m3(uint8_t x) {
    // L059 fix: delegate to the verified adapter_core decoder (bit-exact;
    // covered by test_adapter_bridges "FP8 E4M3 1.0"). The previous local
    // formula scaled normals by an extra 2^4 (0x38 decoded to 8.0, not 1.0).
    return fp8_e4m3_to_float(x);
}

static float st_fp8_e5m2(uint8_t x) {
    // L059 fix: same as above (previous local formula scaled E5M2 normals
    // by an extra 2^1; 0x3C decoded to 2.0, not 1.0).
    return fp8_e5m2_to_float(x);
}

// ── Safetensors parser ───────────────────────────────────────────────────────

std::vector<AdapterTensor> load_safetensors(const std::string& path, bool verbose) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};

    f.seekg(0, std::ios::end);
    std::streamsize file_size = f.tellg();
    f.seekg(0, std::ios::beg);

    if (file_size < 8) return {};

    uint64_t hdr_len_u64 = 0;
    f.read(reinterpret_cast<char*>(&hdr_len_u64), 8);
    if (f.gcount() != 8) return {};
    size_t hdr_len = (size_t)hdr_len_u64;
    if (hdr_len == 0 || hdr_len > (size_t)(file_size - 8)) return {};

    std::string json(hdr_len, '\0');
    f.read(&json[0], (std::streamsize)hdr_len);
    if ((std::streamsize)hdr_len != f.gcount()) return {};

    if (verbose)
        std::fprintf(stderr, "Safetensors header: %zu bytes\n", hdr_len);

    // Find tensor keys by scanning for top-level JSON keys that are not "__metadata__".
    std::vector<std::string> tensor_keys;
    size_t pos = 0;
    while (true) {
        pos = json.find('"', pos);
        if (pos == std::string::npos) break;
        pos++; // skip opening quote
        std::string key;
        while (pos < json.size() && json[pos] != '"') key += json[pos++];
        if (pos >= json.size()) break; // unclosed quote
        pos++; // skip closing quote
        if (key.empty() || key == "__metadata__") continue;
        size_t after = pos;
        while (after < json.size() && (json[after] == ':' || json[after] == ' ' || json[after] == '\n')) after++;
        if (after < json.size() && json[after] == '{') tensor_keys.push_back(key);
    }

    std::vector<AdapterTensor> result;
    result.reserve(tensor_keys.size());

    size_t data_start = 8 + hdr_len;

    for (const auto& key : tensor_keys) {
        // Locate the JSON object for this tensor key.
        size_t obj_start = json.find("\"" + key + "\"");
        if (obj_start == std::string::npos) continue;
        obj_start = json.find('{', obj_start);
        if (obj_start == std::string::npos) continue;

        int depth = 0;
        size_t obj_end = obj_start;
        for (; obj_end < json.size(); obj_end++) {
            if (json[obj_end] == '{') depth++;
            else if (json[obj_end] == '}') { depth--; if (depth == 0) { obj_end++; break; } }
        }
        std::string obj = json.substr(obj_start, obj_end - obj_start);

        // Extract dtype from within the object.
        std::string dtype_str = extract_json_string(obj, "dtype");
        if (dtype_str.empty()) continue;

        // Extract shape array.
        std::vector<int64_t> shape;
        size_t sp = obj.find("\"shape\"");
        if (sp != std::string::npos) {
            sp = obj.find('[', sp);
            if (sp != std::string::npos) {
                sp++;
                while (sp < obj.size()) {
                    while (sp < obj.size() && (obj[sp] == ' ' || obj[sp] == '\t')) sp++;
                    if (sp >= obj.size() || obj[sp] == ']') break;
                    char* endp = nullptr;
                    long long v = std::strtoll(obj.c_str() + sp, &endp, 10);
                    if (endp == obj.c_str() + sp) break;
                    shape.push_back((int64_t)v);
                    sp = (size_t)(endp - obj.c_str());
                    while (sp < obj.size() && (obj[sp] == ' ' || obj[sp] == ',')) sp++;
                    if (sp < obj.size() && obj[sp] == ']') { sp++; break; }
                }
            }
        }

        // Extract data_offsets.
        int64_t off_start = 0, off_end = 0;
        {
            size_t op = obj.find("\"data_offsets\"");
            if (op != std::string::npos) {
                op = obj.find('[', op);
                if (op != std::string::npos) {
                    op++;
                    while (op < obj.size() && (obj[op] == ' ' || obj[op] == '\t')) op++;
                    off_start = std::atoll(obj.c_str() + op);
                    op = obj.find(',', op);
                    if (op != std::string::npos) {
                        op++;
                        while (op < obj.size() && (obj[op] == ' ' || obj[op] == '\t')) op++;
                        off_end = std::atoll(obj.c_str() + op);
                    }
                }
            }
        }

        if (shape.empty() || off_end <= off_start) {
            if (verbose) std::fprintf(stderr, "  skip %s: empty shape/offsets\n", key.c_str());
            continue;
        }

        int64_t numel = 1;
        for (int64_t d : shape) numel *= d;
        if (numel <= 0) continue;

        int bpe = st_bytes_per_elem(dtype_str);
        if (bpe == 0) {
            if (verbose) std::fprintf(stderr, "  skip %s: unsupported dtype %s\n", key.c_str(), dtype_str.c_str());
            continue;
        }

        AdapterTensor at;
        at.name = key;
        at.shape = shape;
        at.data.resize((size_t)numel);

        size_t byte_off = (size_t)(data_start + off_start);
        f.seekg((std::streamoff)byte_off, std::ios::beg);

        std::vector<uint8_t> raw((size_t)(off_end - off_start));
        f.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)raw.size());

        for (int64_t i = 0; i < numel; i++) {
            const uint8_t* p = &raw[(size_t)i * bpe];
            if (dtype_str == "F32") {
                float v; std::memcpy(&v, p, 4); at.data[(size_t)i] = v;
            } else if (dtype_str == "F16") {
                uint16_t h; std::memcpy(&h, p, 2); at.data[(size_t)i] = st_f16_to_f32(h);
            } else if (dtype_str == "BF16") {
                uint16_t b; std::memcpy(&b, p, 2); at.data[(size_t)i] = st_bf16_to_f32(b);
            } else if (dtype_str == "F8_E4M3") {
                at.data[(size_t)i] = st_fp8_e4m3(p[0]);
            } else if (dtype_str == "F8_E5M2") {
                at.data[(size_t)i] = st_fp8_e5m2(p[0]);
            } else if (dtype_str == "I64") {
                int64_t v; std::memcpy(&v, p, 8); at.data[(size_t)i] = (float)v;
            } else if (dtype_str == "I32") {
                int32_t v; std::memcpy(&v, p, 4); at.data[(size_t)i] = (float)v;
            } else if (dtype_str == "I16") {
                int16_t v; std::memcpy(&v, p, 2); at.data[(size_t)i] = (float)v;
            } else if (dtype_str == "I8") {
                at.data[(size_t)i] = (float)(int8_t)p[0];
            } else if (dtype_str == "U8") {
                at.data[(size_t)i] = (float)p[0];
            } else if (dtype_str == "BOOL") {
                at.data[(size_t)i] = p[0] ? 1.0f : 0.0f;
            }
        }
        result.push_back(std::move(at));
        if (verbose)
            std::fprintf(stderr, "  tensor %s dtype=%s shape=[%s] ne=%lld\n",
                          key.c_str(), dtype_str.c_str(),
                          [&]() {
                              std::string s;
                              for (size_t i = 0; i < shape.size(); ++i) { if (i) s += ","; s += std::to_string(shape[i]); }
                              return s;
                          }().c_str(), (long long)numel);
    }
    return result;
}

bool safetensors_to_quant(const std::string& input_path, const BridgeConfig& cfg) {
    auto tensors = load_safetensors(input_path, cfg.verbose);
    if (tensors.empty()) return false;
    BridgeConfig c = cfg;
    return write_quant_mixed(tensors, c);
}

// L059: split block-FP8 pair loader. Reuses load_safetensors() for parsing
// (weight bytes decode as raw FP8 values, scales as floats), then applies
// the per-block scale. Key lookup is exact; any mismatch => ok=false.
BlockFp8Group load_block_fp8_pair(const std::string& path,
                                  const std::string& weight_key,
                                  const std::string& scale_key,
                                  int64_t block) {
    BlockFp8Group out;
    out.block_size = block > 0 ? block : 128;
    if (weight_key.empty() || scale_key.empty()) return out;

    // Recover the weight dtype from the raw header (load_safetensors hides
    // it): re-scan for the weight key's dtype string.
    std::string weight_dtype;
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) return out;
        uint64_t hdr_len_u64 = 0;
        f.read(reinterpret_cast<char*>(&hdr_len_u64), 8);
        if (f.gcount() != 8 || hdr_len_u64 == 0 || hdr_len_u64 > (1u << 26)) return out;
        std::string json((size_t)hdr_len_u64, '\0');
        f.read(&json[0], (std::streamsize)hdr_len_u64);
        if ((std::streamsize)hdr_len_u64 != f.gcount()) return out;
        size_t kp = json.find("\"" + weight_key + "\"");
        if (kp == std::string::npos) return out;
        size_t dp = json.find("\"dtype\"", kp);
        if (dp == std::string::npos || dp > kp + 4096) return out;
        size_t q1 = json.find('"', dp + 7);
        if (q1 == std::string::npos) return out;
        size_t q2 = json.find('"', q1 + 1);
        if (q2 == std::string::npos) return out;
        weight_dtype = json.substr(q1 + 1, q2 - q1 - 1);
    }
    bool e4m3;
    if (weight_dtype == "F8_E4M3") e4m3 = true;
    else if (weight_dtype == "F8_E5M2") e4m3 = false;
    else return out; // not a block-FP8 weight tensor
    out.use_e4m3 = e4m3;

    auto tensors = load_safetensors(path, false);
    const AdapterTensor* w = nullptr;
    const AdapterTensor* s = nullptr;
    for (const auto& t : tensors) {
        if (t.name == weight_key) w = &t;
        if (t.name == scale_key) s = &t;
    }
    if (!w || !s) return out;
    int64_t wn = w->numel();
    int64_t need_scales = (wn + out.block_size - 1) / out.block_size;
    if ((int64_t)s->data.size() != need_scales) return out;

    out.name = weight_key;
    out.shape = w->shape;
    out.data.resize((size_t)wn);
    // w->data holds the exact FP8 bit-decode (st_fp8_* delegate to the
    // verified adapter_core decoders), so scaling here is bit-exact:
    // out[i] == fp8(w[i]) * scale[i / block].
    for (int64_t i = 0; i < wn; i++)
        out.data[(size_t)i] = w->data[(size_t)i] * s->data[(size_t)(i / out.block_size)];
    out.ok = true;
    return out;
}

bool block_fp8_safetensors_to_quant(const std::string& input_path,
                                    const std::string& weight_key,
                                    const std::string& scale_key,
                                    const BridgeConfig& cfg,
                                    int64_t block) {
    BlockFp8Group g = load_block_fp8_pair(input_path, weight_key, scale_key, block);
    if (!g.ok) return false;
    AdapterTensor at;
    at.name = g.name;
    at.shape = g.shape;
    at.data = g.data;
    BridgeConfig c = cfg;
    return write_quant_mixed({at}, c);
}

} // namespace adapters
} // namespace quant
