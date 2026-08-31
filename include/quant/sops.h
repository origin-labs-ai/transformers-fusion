#pragma once
// ============================================================================
// sops.h — SOPS: Sextillion Operations Per Second
// ============================================================================
// A NEW COMPUTE UNIT for quantized model training on CPU.
//
// FORMULA:
//   SOPS = (elements x info_weight) / time / 10^21
//   info_weight = 32 / bits_per_weight_element
//
// All 15 base formats: Q1 through Q32, GRP variants, plus mixes.
// ============================================================================

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <chrono>
#include <atomic>
#include <thread>

namespace quant {

// ── Format BPW lookup (matches RegFormat enum order) ──────────────────────

struct SopsFormat {
    const char* name;
    double      bpw;
    double      info_weight; // = 32 / bpw
};

// All base formats from format_registry.h
inline const SopsFormat sops_formats[] = {
    {"Q1",            1.0,   32.000},
    {"Q0",        1.50,  21.333},
    {"Q1K",    2.0,   16.000},
    {"Q2",            2.0,   16.000},
    {"Q4",            4.0,    8.000},
    {"Q8",            8.0,    4.000},
    {"Q16",          16.0,    2.000},
    {"Q32",          32.0,    1.000},
    {"Q1_G",        1.0,   32.000},
    {"Q2_G",        2.625, 12.190},
    {"Q4_G",        4.50,   7.111},
    {"Q8_G",        8.50,   3.765},
    {"Q16_G",      16.0,    2.000},
    {"Q0_G",    1.50,  21.333},
    {"Q1K_G",2.0,   16.000},
};
inline constexpr int SOPS_NUM_FORMATS = 15;

// Mix format effective BPW (from format_registry.cpp)
inline const SopsFormat sops_mix_formats[] = {
    {"Q8+Q2_1_99",      2.06,  15.534},
    {"Q8+Q4_5_95",      4.20,   7.619},
    {"Q4+Q2_5_95",      2.10,  15.238},
    {"Q4+Q2_10_90",     2.20,  14.545},
    {"Q8+Q2_5_95",      2.30,  13.913},
    {"Q8+Q2_10_90",     2.60,  12.308},
    {"Q4+Q8_5_95",     7.675,  4.169},
    {"Q16+Q4_1_99",     4.12,   7.767},
    {"Q16+Q8_5_95",     8.40,   3.810},
    {"Q32+Q8_1_99",     8.24,   3.883},
};
inline constexpr int SOPS_NUM_MIXES = 10;

// ── SIMD detection ───────────────────────────────────────────────────────

enum class SopsISA : uint8_t {
    SCALAR = 0,
    SSE4   = 1,
    AVX2   = 2,
    AVX512 = 3,
};

inline const char* sops_isa_name(SopsISA isa) {
    switch (isa) {
        case SopsISA::SCALAR: return "scalar";
        case SopsISA::SSE4:   return "SSE4";
        case SopsISA::AVX2:   return "AVX2";
        case SopsISA::AVX512: return "AVX512";
        default:              return "unknown";
    }
}

#if defined(__AVX512F__)
inline constexpr SopsISA SOPS_DEFAULT_ISA = SopsISA::AVX512;
#elif defined(__AVX2__)
inline constexpr SopsISA SOPS_DEFAULT_ISA = SopsISA::AVX2;
#elif defined(__SSE4_1__)
inline constexpr SopsISA SOPS_DEFAULT_ISA = SopsISA::SSE4;
#else
inline constexpr SopsISA SOPS_DEFAULT_ISA = SopsISA::SCALAR;
#endif

// ── Timer ────────────────────────────────────────────────────────────────

inline uint64_t sops_rdtsc() {
#if defined(_MSC_VER)
    return __rdtsc();
#elif defined(__x86_64__) || defined(__i386__)
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#else
    return (uint64_t)std::chrono::high_resolution_clock::now()
                .time_since_epoch().count();
#endif
}

// ── SOPS Counter ─────────────────────────────────────────────────────────

struct SopsCounter {
    std::atomic<uint64_t> total_info_ops{0};
    std::atomic<uint64_t> total_elements{0};
    uint64_t start_tsc = 0;
    uint64_t end_tsc   = 0;
    double   cpu_ghz   = 3.0;
    int      num_threads = 1;

    void reset() {
        total_info_ops.store(0);
        total_elements.store(0);
        start_tsc = end_tsc = 0;
    }

    void start() { start_tsc = sops_rdtsc(); }
    void stop()  { end_tsc = sops_rdtsc(); }

    void record(int format_index, int64_t count) {
        if (format_index < 0 || format_index >= SOPS_NUM_FORMATS) return;
        double iw = sops_formats[format_index].info_weight;
        total_info_ops.fetch_add((uint64_t)(count * iw));
        total_elements.fetch_add((uint64_t)count);
    }

    void record_mix(int mix_index, int64_t count) {
        if (mix_index < 0 || mix_index >= SOPS_NUM_MIXES) return;
        double iw = sops_mix_formats[mix_index].info_weight;
        total_info_ops.fetch_add((uint64_t)(count * iw));
        total_elements.fetch_add((uint64_t)count);
    }

    void record_iw(double info_weight, int64_t count) {
        total_info_ops.fetch_add((uint64_t)(count * info_weight));
        total_elements.fetch_add((uint64_t)count);
    }

    double elapsed_sec() const {
        if (start_tsc == 0 || end_tsc == 0) return 0.0;
        return (double)(end_tsc - start_tsc) / (cpu_ghz * 1e9);
    }

    double raw_ops_per_sec() const {
        double dt = elapsed_sec();
        if (dt <= 0.0) return 0.0;
        return (double)total_info_ops.load() / dt;
    }

    double sops()  const { return raw_ops_per_sec() / 1e21; }
    double psops() const { return raw_ops_per_sec() / 1e15; }
    double nsops() const { return raw_ops_per_sec() / 1e12; }
    double usops() const { return raw_ops_per_sec() / 1e9; }
    double msops() const { return raw_ops_per_sec() / 1e6; }

    double effective_gflops() const { return raw_ops_per_sec() / 1e9; }

    const char* best_unit() const {
        double v = sops();
        if (v >= 1.0)  return "SOPS";
        v *= 1e3; if (v >= 1.0) return "mSOPS";
        v *= 1e3; if (v >= 1.0) return "uSOPS";
        v *= 1e3; if (v >= 1.0) return "nSOPS";
        v *= 1e3; if (v >= 1.0) return "pSOPS";
        return "fSOPS";
    }

    double best_value() const {
        double v = sops();
        if (v >= 1.0)  return v;
        v *= 1e3; if (v >= 1.0) return v;
        v *= 1e3; if (v >= 1.0) return v;
        v *= 1e3; if (v >= 1.0) return v;
        v *= 1e3; return v;
    }
};

// ── Helpers ──────────────────────────────────────────────────────────────

inline const char* sops_unit_name(double val) {
    if (val >= 1.0)  return "SOPS";
    val *= 1e3; if (val >= 1.0) return "mSOPS";
    val *= 1e3; if (val >= 1.0) return "uSOPS";
    val *= 1e3; if (val >= 1.0) return "nSOPS";
    val *= 1e3; if (val >= 1.0) return "pSOPS";
    return "fSOPS";
}

inline double sops_best_value(double val) {
    if (val >= 1.0)  return val;
    val *= 1e3; if (val >= 1.0) return val;
    val *= 1e3; if (val >= 1.0) return val;
    val *= 1e3; if (val >= 1.0) return val;
    val *= 1e3; return val;
}

} // namespace quant
