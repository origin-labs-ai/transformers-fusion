// ============================================================================
// quant_verify_roundtrip.cpp — verify a .quant file against the original
// safetensors weights, tensor-by-tensor, WITHOUT loading full shards.
// Peak memory = one original tensor + one dequantized tensor.
// ============================================================================
#include "quant/quant_format.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// ── Minimal JSON helpers (safetensors headers are flat) ────────────────────

static std::string json_str(const std::string& s, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t p = s.find(needle);
    if (p == std::string::npos) return {};
    p = s.find(':', p + needle.size());
    if (p == std::string::npos) return {};
    p++;
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) p++;
    if (p >= s.size() || s[p] != '"') return {};
    p++;
    std::string v;
    while (p < s.size() && s[p] != '"') {
        if (s[p] == '\\' && p + 1 < s.size()) { v += s[p + 1]; p += 2; continue; }
        v += s[p++];
    }
    return v;
}

// Parse [a,b] style arrays: "shape":[1,2] or "data_offsets":[x,y]
static std::vector<int64_t> json_int_array(const std::string& s, const std::string& key) {
    std::vector<int64_t> out;
    size_t p = s.find("\"" + key + "\"");
    if (p == std::string::npos) return out;
    p = s.find('[', p);
    if (p == std::string::npos) return out;
    p++;
    while (p < s.size()) {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == ',')) p++;
        if (p >= s.size() || s[p] == ']') break;
        char* endp = nullptr;
        long long v = std::strtoll(s.c_str() + p, &endp, 10);
        if (endp == s.c_str() + p) break;
        out.push_back((int64_t)v);
        p = (size_t)(endp - s.c_str());
    }
    return out;
}

// Parse safetensors index.json weight_map: name -> shard file.
static std::vector<std::pair<std::string, std::string>> parse_weight_map(const std::string& json) {
    std::vector<std::pair<std::string, std::string>> out;
    size_t p = json.find("\"weight_map\"");
    if (p == std::string::npos) return out;
    p = json.find('{', p);
    if (p == std::string::npos) return out;
    p++;
    while (p < json.size()) {
        while (p < json.size() && json[p] != '"') p++;
        if (p >= json.size()) break;
        p++;
        std::string key;
        while (p < json.size() && json[p] != '"') key += json[p++];
        p++;
        while (p < json.size() && json[p] != ':') p++;
        p++;
        while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n' || json[p] == '\r')) p++;
        if (p >= json.size() || json[p] != '"') break;
        p++;
        std::string val;
        while (p < json.size() && json[p] != '"') val += json[p++];
        p++;
        out.push_back({ key, val });
    }
    return out;
}

static std::string read_text_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Read ONE tensor's raw slice from a safetensors shard and dequantize to FP32.
// Supported: F32, BF16, F16. Returns false on any structural problem.
static bool read_tensor_slice(const std::string& shard_path, const std::string& name,
                              std::vector<float>& out) {
    std::ifstream f(shard_path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    std::streamsize file_size = f.tellg();
    f.seekg(0, std::ios::beg);
    if (file_size < 8) return false;

    uint64_t hdr_len_u64 = 0;
    f.read(reinterpret_cast<char*>(&hdr_len_u64), 8);
    size_t hdr_len = (size_t)hdr_len_u64;
    if (hdr_len == 0 || hdr_len > (size_t)(file_size - 8)) return false;

    std::string json(hdr_len, '\0');
    f.read(&json[0], (std::streamsize)hdr_len);
    if ((std::streamsize)hdr_len != f.gcount()) return false;

    // Locate the tensor object { ... }.
    size_t obj_start = json.find("\"" + name + "\"");
    if (obj_start == std::string::npos) return false;
    obj_start = json.find('{', obj_start);
    if (obj_start == std::string::npos) return false;
    int depth = 0;
    size_t obj_end = obj_start;
    for (; obj_end < json.size(); obj_end++) {
        if (json[obj_end] == '{') depth++;
        else if (json[obj_end] == '}') { depth--; if (depth == 0) { obj_end++; break; } }
    }
    std::string obj = json.substr(obj_start, obj_end - obj_start);

    std::string dtype = json_str(obj, "dtype");
    auto shape = json_int_array(obj, "shape");
    auto offs = json_int_array(obj, "data_offsets");
    if (dtype.empty() || shape.empty() || offs.size() < 2) return false;

    int64_t numel = 1;
    for (int64_t d : shape) numel *= d;
    if (numel <= 0 || offs[1] <= offs[0]) return false;
    int64_t span = offs[1] - offs[0];

    size_t data_start = 8 + hdr_len;
    f.seekg((std::streamoff)(data_start + offs[0]), std::ios::beg);
    std::vector<uint8_t> raw((size_t)span);
    f.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)raw.size());
    if (f.gcount() != (std::streamsize)raw.size()) return false;

    out.resize((size_t)numel);
    if (dtype == "F32") {
        std::memcpy(out.data(), raw.data(), (size_t)numel * 4);
    } else if (dtype == "BF16") {
        for (int64_t i = 0; i < numel; i++) {
            uint32_t bits = (uint32_t)raw[(size_t)i * 2] | ((uint32_t)raw[(size_t)i * 2 + 1] << 8);
            float v;
            uint32_t f32 = bits << 16;
            std::memcpy(&v, &f32, 4);
            out[(size_t)i] = v;
        }
    } else if (dtype == "F16") {
        for (int64_t i = 0; i < numel; i++) {
            uint16_t h = (uint16_t)raw[(size_t)i * 2] | ((uint16_t)raw[(size_t)i * 2 + 1] << 8);
            uint32_t sign = (uint32_t)(h >> 15) << 31;
            int32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t bits;
            if (exp == 0) {
                float m = (mant ? (float)mant / 1024.0f * 1.5258789e-5f : 0.0f);
                bits = sign;
                if (m != 0) { float fv = m; std::memcpy(&bits, &fv, 4); bits |= sign; }
            } else if (exp == 31) {
                float fv = mant ? NAN : INFINITY;
                std::memcpy(&bits, &fv, 4);
            } else {
                bits = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
            }
            float v;
            std::memcpy(&v, &bits, 4);
            out[(size_t)i] = v;
        }
    } else {
        out.clear();
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: quant_verify_roundtrip <model_dir> <model.quant> [tensors_to_check] [name_filter]\n");
        return 1;
    }
    std::string dir = argv[1];
    std::string quant_path = argv[2];
    // BUGFIX (bug census): bare atoi fail-open (garbage → 0 → checks nothing,
    // exits 0 = fake pass). Validated; garbage exits 2.
    int limit = 5;
    if (argc >= 4) {
        char* end = nullptr;
        long v = std::strtol(argv[3], &end, 10);
        if (!end || *end != '\0' || v < 0 || v > 1000000) {
            std::printf("ERROR: tensors_to_check needs 0..1000000, got '%s'\n", argv[3]);
            return 2;
        }
        limit = (int)v;
    }
    std::string filter = argc >= 5 ? argv[4] : "";

    // name -> shard, from index.json
    std::string index_path = dir + "/model.safetensors.index.json";
    auto wm = parse_weight_map(read_text_file(index_path));
    if (wm.empty()) { std::printf("ERROR: no weight_map in %s\n", index_path.c_str()); return 1; }

    quant::QUANTReader reader(quant_path);
    if (!reader.valid()) { std::printf("ERROR: cannot open %s\n", quant_path.c_str()); return 1; }
    auto names = reader.tensor_names();
    std::printf("QUANT file: %zu tensors; weight_map: %zu\n", names.size(), wm.size());

    int checked = 0;
    double total_mse = 0.0;
    double total_w = 0.0;
    for (const auto& name : names) {
        if (checked >= limit) break;
        if (!filter.empty() && name.find(filter) == std::string::npos) continue;

        std::string shard;
        for (const auto& e : wm) if (e.first == name) { shard = e.second; break; }
        if (shard.empty()) { std::printf("  %-52s NOT IN WEIGHT MAP\n", name.c_str()); continue; }

        std::vector<float> orig;
        if (!read_tensor_slice(dir + "/" + shard, name, orig)) {
            std::printf("  %-52s SLICE READ FAILED\n", name.c_str());
            continue;
        }

        auto t = reader.read_tensor(name);
        if (t.data() == nullptr || t.numel() == 0) { std::printf("  %-52s READ FAILED\n", name.c_str()); continue; }
        const float* deq = (const float*)t.data();
        int64_t n = t.numel();
        if ((int64_t)orig.size() != n) { std::printf("  %-52s SIZE MISMATCH orig=%zu quant=%lld\n", name.c_str(), orig.size(), (long long)n); continue; }

        double s = 0.0, maxe = 0.0, pow_sig = 0.0;
        int64_t z = 0;
        for (int64_t i = 0; i < n; i++) {
            double d = (double)orig[(size_t)i] - (double)deq[i];
            s += d * d;
            double a = std::fabs(d);
            if (a > maxe) maxe = a;
            pow_sig += (double)orig[(size_t)i] * (double)orig[(size_t)i];
            if (deq[i] == 0.0f) z++;
        }
        double mse = s / (double)n;
        double snr = (pow_sig > 1e-30 && mse > 1e-30) ? 10.0 * std::log10(pow_sig / s) : 99.0;
        std::printf("  %-52s n=%lld MSE=%.6e max_err=%.6e SNR=%.1fdB zeroed=%.2f%%\n",
                    name.c_str(), (long long)n, mse, maxe, snr, 100.0 * (double)z / (double)n);
        total_mse += mse * (double)n;
        total_w += (double)n;
        checked++;
    }
    if (total_w > 0) {
        std::string wstr = std::to_string(total_w);
        std::printf("WEIGHTED MSE over %d tensors (%s weights): %.6e\n",
                    checked, wstr.c_str(), total_mse / total_w);
    } else std::printf("No tensors matched.\n");
    return 0;
}
