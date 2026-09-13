// ============================================================================
// bench_compare.cpp — QUANT Mixed vs GGUF Q4_K_M benchmark (PURE C++)
// ============================================================================
// Build:
//   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang-cl
//   cmake --build build --target bench_compare
//
// Usage:
//   bench_compare --input-dir benchmark_results/raw_weights --output results.json
//   bench_compare --input-dir benchmark_results/raw_weights --output results.json --max-layers 10
// ============================================================================
#include "adapters/adapter_core.h"
#include "adapters/ptq_bridge.h"
#include "quant/quant_format.h"
#include "quant/tensor.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <map>

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/stat.h>
#include <dirent.h>
#endif

using namespace quant::adapters;

// ============================================================================
// GGUF Q4_K_M Quantization — Pure C++ Implementation
// ============================================================================
// Based on llama.cpp ggml-quants.c quantization algorithm
// Block structure: 256 weights per super-block, 8 sub-blocks of 32 each

namespace gguf_q4 {

static inline uint16_t f32_to_f16(float v) {
    if (v == 0.0f) return 0;
    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint16_t sign = (uint16_t)((bits >> 31) & 1);
    int exp = (int)((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp > 15) return sign | 0x7C00;
    if (exp < -14) {
        if (exp < -24) return sign;
        int m = mant | 0x800000;
        int shift = -exp - 14 + 13;
        int mf = m >> shift;
        int round_bit = (m >> (shift - 1)) & 1;
        int sticky = ((m & ((1 << (shift - 1)) - 1)) ? 1 : 0);
        int lsb = mf & 1;
        if (round_bit && (sticky || lsb)) mf += 1;
        if (mf >= 0x400) return sign | (1 << 10) | (mf & 0x3FF);
        return sign | (uint16_t)mf;
    }
    int m = (mant >> 13) | 0x400;
    int round_bit = (mant >> 12) & 1;
    int sticky = (mant & 0xFFF) ? 1 : 0;
    int lsb = m & 1;
    if (round_bit && (sticky || lsb)) m += 1;
    if (m >= 0x800) { m >>= 1; exp += 1; if (exp > 15) return sign | 0x7C00; }
    return sign | ((uint16_t)(exp + 15) << 10) | (uint16_t)(m & 0x3FF);
}

static inline float f16_to_f32(uint16_t h) {
    if (h == 0) return 0.0f;
    float result;
    uint32_t bits = ((uint32_t)(h & 0x8000) << 16) | ((uint32_t)(h & 0x7FFF) << 13);
    memcpy(&result, &bits, 4);
    return (h & 0x8000) ? -result : result;
}

static inline int nearest_int(float f) { return (int)std::round(f); }

// Quantize one sub-block (32 weights) to 4-bit with optimal affine mapping
static float quantize_subblock(const float* wsub, int n, uint8_t* qout, float& min_out) {
    float w_min = *std::min_element(wsub, wsub + n);
    float w_max = *std::max_element(wsub, wsub + n);

    if (w_max == w_min) {
        min_out = 0.0f;
        for (int i = 0; i < n; i++) qout[i] = 8;
        return 0.0f;
    }

    float scale = (w_max - w_min) / 15.0f;
    float min_val = w_min;
    float best_scale = scale, best_min = min_val;
    float best_err = 1e30f;

    for (int iter = 0; iter < 20; iter++) {
        float err = 0.0f;
        for (int i = 0; i < n; i++) {
            int q = nearest_int((wsub[i] + min_val) / scale);
            q = std::max(0, std::min(15, q));
            float recon = scale * q - min_val;
            float diff = wsub[i] - recon;
            err += diff * diff;
        }
        if (err < best_err) {
            best_err = err;
            best_scale = scale;
            best_min = min_val;
        }
        // Refit via least-squares
        float sum_q = 0, sum_q2 = 0, sum_w = 0, sum_wq = 0;
        for (int i = 0; i < n; i++) {
            int q = nearest_int((wsub[i] + min_val) / scale);
            q = std::max(0, std::min(15, q));
            sum_q += q; sum_q2 += (float)q * q;
            sum_w += wsub[i]; sum_wq += wsub[i] * q;
        }
        float denom = (float)n * sum_q2 - sum_q * sum_q;
        if (std::fabs(denom) > 1e-10f) {
            scale = ((float)n * sum_wq - sum_w * sum_q) / denom;
            scale = std::max(scale, 1e-10f);
        }
        min_val = (sum_w - scale * sum_q) / (float)n;
        if (min_val > 0.0f) min_val = 0.0f;
    }

    min_out = best_min;
    for (int i = 0; i < n; i++) {
        int q = nearest_int((wsub[i] + best_min) / best_scale);
        q = std::max(0, std::min(15, q));
        qout[i] = (uint8_t)q;
    }
    return best_scale;
}

// Pack 256 weights into one Q4_K block (144 bytes)
struct Q4KBlock {
    uint8_t data[144];
};

static void quantize_superblock(const float* w256, Q4KBlock& block) {
    float sub_scales[8];
    float sub_mins[8];
    uint8_t sub_quants[8][32];
    float max_scale = 0.0f, max_min = 0.0f;

    for (int j = 0; j < 8; j++) {
        sub_scales[j] = quantize_subblock(w256 + j * 32, 32, sub_quants[j], sub_mins[j]);
        max_scale = std::max(max_scale, std::fabs(sub_scales[j]));
        max_min = std::max(max_min, std::fabs(sub_mins[j]));
    }

    float inv_scale = max_scale > 0 ? 63.0f / max_scale : 0.0f;
    float inv_min = max_min > 0 ? 63.0f / max_min : 0.0f;

    uint8_t packed_scales[12] = {0};
    for (int j = 0; j < 8; j++) {
        uint8_t ls = (uint8_t)std::min(63, nearest_int(inv_scale * std::fabs(sub_scales[j])));
        uint8_t lm = (uint8_t)std::min(63, nearest_int(inv_min * std::fabs(sub_mins[j])));
        if (j < 4) {
            packed_scales[j] = ls;
            packed_scales[j + 4] = lm;
        } else {
            packed_scales[j + 4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
            packed_scales[j - 4] |= ((ls >> 4) << 6);
            packed_scales[j] |= ((lm >> 4) << 6);
        }
    }

    uint16_t d_half = f32_to_f16(max_scale / 63.0f);
    uint16_t dmin_half = f32_to_f16(max_min / 63.0f);
    memcpy(block.data, &d_half, 2);
    memcpy(block.data + 2, &dmin_half, 2);
    memcpy(block.data + 4, packed_scales, 12);

    memset(block.data + 16, 0, 128);
    for (int j = 0; j < 8; j++) {
        for (int l = 0; l < 32; l++) {
            int byte_idx = j * 16 + l / 2;
            if (l % 2 == 0)
                block.data[16 + byte_idx] = sub_quants[j][l];
            else
                block.data[16 + byte_idx] |= (sub_quants[j][l] << 4);
        }
    }
}

static float dequantize_superblock(const Q4KBlock& block, float* w256) {
    uint16_t d_half, dmin_half;
    memcpy(&d_half, block.data, 2);
    memcpy(&dmin_half, block.data + 2, 2);
    float d_val = f16_to_f32(d_half);
    float dmin_val = f16_to_f32(dmin_half);
    const uint8_t* scales = block.data + 4;
    const uint8_t* quants = block.data + 16;

    for (int j = 0; j < 8; j++) {
        uint8_t sc, m;
        if (j < 4) {
            sc = scales[j] & 63;
            m = scales[j + 4] & 63;
        } else {
            sc = (uint8_t)((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
            m = (uint8_t)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
        }
        float scale = d_val * sc;
        float min_val = dmin_val * m;
        for (int l = 0; l < 32; l++) {
            int byte_idx = j * 16 + l / 2;
            uint8_t q = (l % 2 == 0) ? (quants[byte_idx] & 0xF) : ((quants[byte_idx] >> 4) & 0xF);
            w256[j * 32 + l] = scale * q - min_val;
        }
    }
    return d_val;
}

// Quantize entire tensor (M x K) to Q4_K bytes
static std::vector<uint8_t> quantize_tensor_q4k(const float* data, int M, int K) {
    int k_padded = K;
    if (k_padded % 256 != 0) k_padded = ((k_padded + 255) / 256) * 256;
    int n_blocks_per_row = k_padded / 256;
    int total_blocks = M * n_blocks_per_row;
    std::vector<uint8_t> out(total_blocks * 144);
    std::vector<float> padded(k_padded);

    for (int m = 0; m < M; m++) {
        const float* row = data + m * K;
        for (int sb = 0; sb < n_blocks_per_row; sb++) {
            for (int k = 0; k < 256; k++) {
                int src_k = sb * 256 + k;
                padded[k] = (src_k < K) ? row[src_k] : 0.0f;
            }
            Q4KBlock block;
            quantize_superblock(padded.data(), block);
            memcpy(out.data() + (m * n_blocks_per_row + sb) * 144, block.data, 144);
        }
    }
    return out;
}

static float measure_q4k_error(const float* orig, const std::vector<uint8_t>& qdata, int M, int K) {
    int k_padded = K;
    if (k_padded % 256 != 0) k_padded = ((k_padded + 255) / 256) * 256;
    int n_blocks_per_row = k_padded / 256;
    double mse_sum = 0.0;
    int64_t n_elem = 0;
    std::vector<float> recon(k_padded);

    for (int m = 0; m < M; m++) {
        for (int sb = 0; sb < n_blocks_per_row; sb++) {
            Q4KBlock block;
            memcpy(block.data, qdata.data() + (m * n_blocks_per_row + sb) * 144, 144);
            dequantize_superblock(block, recon.data());
            for (int k = 0; k < 256; k++) {
                int src_k = sb * 256 + k;
                if (src_k < K) {
                    float diff = orig[m * K + src_k] - recon[k];
                    mse_sum += diff * diff;
                    n_elem++;
                }
            }
        }
    }
    return (float)(mse_sum / (double)n_elem);
}

} // namespace gguf_q4

// ============================================================================
// Utility: directory listing
// ============================================================================

static std::vector<std::string> list_files(const std::string& dir, const char* ext) {
    std::vector<std::string> files;
#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    std::string pattern = dir + "\\*" + ext;
    HANDLE hFind = FindFirstFileA(pattern.c_str(), &fd);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                files.push_back(dir + "\\" + fd.cFileName);
            }
        } while (FindNextFileA(hFind, &fd));
        FindClose(hFind);
    }
#else
    DIR* d = opendir(dir.c_str());
    if (d) {
        struct dirent* ent;
        while ((ent = readdir(d)) != NULL) {
            std::string name = ent->d_name;
            if (name.size() > strlen(ext) && name.substr(name.size() - strlen(ext)) == ext) {
                files.push_back(dir + "/" + name);
            }
        }
        closedir(d);
    }
#endif
    std::sort(files.begin(), files.end());
    return files;
}

static std::vector<float> load_raw_fp32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    size_t sz = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> data(sz / 4);
    if (sz > 0) f.read(reinterpret_cast<char*>(data.data()), sz);
    return data;
}

static void save_json(const std::string& path, const std::string& json) {
    std::ofstream f(path);
    f << json;
}

// ============================================================================
// Benchmark
// ============================================================================

struct LayerResult {
    std::string name;
    int M = 0, K = 0;
    float quant_mse = 0.0f;
    float q4_mse = 0.0f;
    float quant_rmse = 0.0f;
    float q4_rmse = 0.0f;
    float quant_bpw = 0.0f;
    float quant_time = 0.0f;
    float q4_time = 0.0f;
    size_t quant_bytes = 0;
    size_t q4_bytes = 0;
    size_t orig_bytes = 0;
};

int main(int argc, char** argv) {
    std::string input_dir = "benchmark_results/raw_weights";
    std::string output_path = "benchmark_results/cpp_results.json";
    int max_layers = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input-dir") == 0 && i + 1 < argc)
            input_dir = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_path = argv[++i];
        else if (strcmp(argv[i], "--max-layers") == 0 && i + 1 < argc)
            max_layers = std::atoi(argv[++i]);
    }

    printf("=== QUANT Mixed vs GGUF Q4_K_M Benchmark (PURE C++) ===\n");
    printf("Input dir: %s\n", input_dir.c_str());

    auto files = list_files(input_dir, ".fp32");
    if (files.empty()) {
        fprintf(stderr, "ERROR: No .fp32 files found in %s\n", input_dir.c_str());
        return 1;
    }
    printf("Found %zu weight files\n", files.size());

    std::vector<LayerResult> results;
    double quant_mse_sum = 0.0, q4_mse_sum = 0.0;
    size_t quant_bytes_sum = 0, q4_bytes_sum = 0, orig_bytes_sum = 0;
    int64_t n_elem_total = 0;

    int layer_idx = 0;
    for (const auto& path : files) {
        if (max_layers > 0 && layer_idx >= max_layers) break;

        auto data = load_raw_fp32(path);
        if (data.empty()) continue;

        int M = 1, K = (int)data.size();
        size_t name_start = path.find_last_of("\\/") + 1;
        std::string name = path.substr(name_start);
        name = name.substr(0, name.size() - 5); // remove .fp32

        printf("\n[%d] %s shape=(%d, %d)\n", layer_idx + 1, name.c_str(), M, K);
        fflush(stdout);

        LayerResult lr;
        lr.name = name;
        lr.M = M; lr.K = K;
        lr.orig_bytes = K * 4;

        AdapterTensor t;
        t.name = name;
        t.data = data;  // copy, don't move — needed for GGUF section too
        t.shape = { (int64_t)K };

        // ── QUANT Mixed Precision ──
        {
            BridgeConfig cfg;
            cfg.target_bpw = 1.50f;
            cfg.block_size = 256;
            cfg.output_path = input_dir + "/" + name + "_mixed.quant";
            cfg.verbose = false;

            auto t0 = std::chrono::high_resolution_clock::now();
            bool ok = write_quant_mixed({ t }, cfg);
            auto t1 = std::chrono::high_resolution_clock::now();
            lr.quant_time = std::chrono::duration<float>(t1 - t0).count();

            if (ok) {
                std::ifstream f(cfg.output_path, std::ios::binary);
                f.seekg(0, std::ios::end);
                lr.quant_bytes = (size_t)f.tellg();

                quant::QUANTReader reader(cfg.output_path);
                if (reader.valid()) {
                    auto tn = reader.tensor_names();
                    if (!tn.empty()) {
                        quant::Tensor loaded = reader.read_tensor(tn[0]);
                        const float* recon = loaded.data<float>();
                        double mse = 0.0;
                        for (int i = 0; i < K; i++) {
                            float diff = t.data[i] - recon[i];
                            mse += diff * diff;
                        }
                        lr.quant_mse = (float)(mse / K);
                        lr.quant_rmse = sqrtf(lr.quant_mse);
                    }
                }
            }
            printf("    QUANT Mixed:  MSE=%.6f RMSE=%.6f bytes=%zu time=%.3fs\n",
                   lr.quant_mse, lr.quant_rmse, lr.quant_bytes, lr.quant_time);
        }

        // ── GGUF Q4_K_M ──
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto qdata = gguf_q4::quantize_tensor_q4k(t.data.data(), M, K);
            auto t1 = std::chrono::high_resolution_clock::now();
            lr.q4_time = std::chrono::duration<float>(t1 - t0).count();
            lr.q4_bytes = qdata.size();
            lr.q4_mse = gguf_q4::measure_q4k_error(t.data.data(), qdata, M, K);
            lr.q4_rmse = sqrtf(lr.q4_mse);
            printf("    GGUF Q4_K:  MSE=%.6f RMSE=%.6f bytes=%zu time=%.3fs\n",
                   lr.q4_mse, lr.q4_rmse, lr.q4_bytes, lr.q4_time);
        }

        results.push_back(lr);
        quant_mse_sum += lr.quant_mse * K;
        q4_mse_sum += lr.q4_mse * K;
        quant_bytes_sum += lr.quant_bytes;
        q4_bytes_sum += lr.q4_bytes;
        orig_bytes_sum += lr.orig_bytes;
        n_elem_total += K;
        layer_idx++;
    }

    if (n_elem_total == 0 || results.empty()) {
        fprintf(stderr, "ERROR: No layers benchmarked\n");
        return 1;
    }

    // ── Aggregate ──
    float quant_avg_mse = (float)(quant_mse_sum / n_elem_total);
    float q4_avg_mse = (float)(q4_mse_sum / n_elem_total);
    float quant_ratio = orig_bytes_sum > 0 ? (float)quant_bytes_sum / orig_bytes_sum : 0.0f;
    float q4_ratio = orig_bytes_sum > 0 ? (float)q4_bytes_sum / orig_bytes_sum : 0.0f;

    printf("\n%s\n", std::string(60, '=').c_str());
    printf("BENCHMARK RESULTS (PURE C++)\n");
    printf("%s\n", std::string(60, '=').c_str());
    printf("Layers benchmarked: %d\n", (int)results.size());
    printf("Total weights: %.1fM\n", n_elem_total / 1e6f);
    printf("\nQUANT Mixed (95%% QUANT, 4%% Q4, 1%% Q8):\n");
    printf("  Avg MSE:   %.6f\n", quant_avg_mse);
    printf("  Avg RMSE:  %.6f\n", sqrtf(quant_avg_mse));
    printf("  Size:      %.1fKB / %.1fKB (%.1f%%)\n",
           quant_bytes_sum / 1024.0f, orig_bytes_sum / 1024.0f, quant_ratio * 100.0f);
    printf("\nGGUF Q4_K_M:\n");
    printf("  Avg MSE:   %.6f\n", q4_avg_mse);
    printf("  Avg RMSE:  %.6f\n", sqrtf(q4_avg_mse));
    printf("  Size:      %.1fKB / %.1fKB (%.1f%%)\n",
           q4_bytes_sum / 1024.0f, orig_bytes_sum / 1024.0f, q4_ratio * 100.0f);

    int quant_wins = 0;
    for (const auto& lr : results) {
        if (lr.quant_mse < lr.q4_mse) quant_wins++;
    }
    printf("\nWinner: QUANT wins %d/%d layers (%.0f%%)\n",
           quant_wins, (int)results.size(), results.empty() ? 0 : 100.0f * quant_wins / results.size());

    if (quant_avg_mse < q4_avg_mse)
        printf("  >>> QUANT WINS by %.1f%% lower reconstruction error\n",
               (q4_avg_mse - quant_avg_mse) / q4_avg_mse * 100.0f);
    else
        printf("  >>> GGUF Q4_K_M WINS by %.1f%%\n",
               (quant_avg_mse - q4_avg_mse) / quant_avg_mse * 100.0f);

    // ── JSON output ──
    char json_buf[65536];
    int pos = 0;
    pos += snprintf(json_buf + pos, sizeof(json_buf) - pos,
        "{\n  \"model\":\"bloom-560m\",\n"
        "  \"layers\":%d,\n"
        "  \"quant_avg_mse\":%.6f,\n"
        "  \"q4_avg_mse\":%.6f,\n"
        "  \"quant_compression\":%.3f,\n"
        "  \"q4_compression\":%.3f,\n"
        "  \"quant_wins\":%d,\n"
        "  \"q4_wins\":%d,\n",
        (int)results.size(), quant_avg_mse, q4_avg_mse, quant_ratio, q4_ratio, quant_wins, (int)results.size() - quant_wins);

    pos += snprintf(json_buf + pos, sizeof(json_buf) - pos, "  \"layers\":[\n");
    for (size_t i = 0; i < results.size(); i++) {
        const auto& lr = results[i];
        pos += snprintf(json_buf + pos, sizeof(json_buf) - pos,
            "    {\"name\":\"%s\",\"quant_mse\":%.6f,\"q4_mse\":%.6f,"
            "\"quant_bytes\":%zu,\"q4_bytes\":%zu}\n%s",
            lr.name.c_str(), lr.quant_mse, lr.q4_mse, lr.quant_bytes, lr.q4_bytes,
            i + 1 < results.size() ? "," : "");
    }
    pos += snprintf(json_buf + pos, sizeof(json_buf) - pos, "  ]\n}\n");

    save_json(output_path, json_buf);
    printf("\n[+] Results saved to %s\n", output_path.c_str());

    return 0;
}
