#pragma once
// ============================================================================
// inferno.h — Inferno on FLOPS base: raw FLOPS measured, Inferno derived.
// ============================================================================
// FLOPS = 2 * elements / seconds (1 MAC = 2 FLOPS, FP32 GEMM standard).
// Inferno = (elements x info_weight) / time / 10^21
//         = FLOPS x info_weight / 2 / 10^21
//   info_weight = 32 / bits_per_weight_element
// TransFormaers-Fusion rule: every calculation runs in FLOPS first,
// Inferno is only a scaled view of that FLOPS number.
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

struct InfernoFormat {
    const char* name;
    double      bpw;
    double      info_weight; // = 32 / bpw
};

// Canonical singles wired to quant::Format one-truth (types.h:22-67,
// FORMAT_COUNT=105, wire BPW via format_bpw). Inferno tables carry only
// these names — no QUANT*/Q0/Q1K aliases. IW = 32 / wire BPW.
inline const InfernoFormat inferno_formats[] = {
    {"Q1",   1.0,     32.0},
    {"Q2",   2.0,     16.0},
    {"Q3",   3.0,     32.0 / 3.0},
    {"Q4",   4.0,     8.0},
    {"Q6",   6.0,     32.0 / 6.0},
    {"Q8",   8.0,     4.0},
    {"Q12",  12.0,    32.0 / 12.0},
    {"Q16",  16.0,    2.0},
    {"Q24",  24.0,    32.0 / 24.0},
    {"Q32",  32.0,    1.0},
    {"QG1",  1.0,     32.0},
    {"QG2",  2.625,   32.0 / 2.625},
    {"QG3",  3.5,     32.0 / 3.5},
    {"QG4",  4.5,     32.0 / 4.5},
    {"QG6",  6.5625,  32.0 / 6.5625},
    {"QG8",  8.5,     32.0 / 8.5},
    {"QG12", 12.5,    32.0 / 12.5},
    {"QG16", 16.5,    32.0 / 16.5},
    {"QG24", 24.5,    32.0 / 24.5},
};
inline constexpr int INFERNO_NUM_FORMATS = 19;

// Mix effective BPW from format_registry.cpp (Q_MX_*/QG_MX_* wire BPW).
// Names are canonical v3 display names (Q_MX_3.5 etc).
inline const InfernoFormat inferno_mix_formats[] = {
    {"Q_MX_3.5",   3.5,     32.0 / 3.5},
    {"Q_MX_4.5",   4.5,     32.0 / 4.5},
    {"Q_MX_6.5",   6.5,     32.0 / 6.5},
    {"Q_MX_8.5",   8.5,     32.0 / 8.5},
    {"Q_MX_12.5",  12.5,    32.0 / 12.5},
    {"Q_MX_16.5",  16.5,    32.0 / 16.5},
    {"Q_MX_24.5",  24.5,    32.0 / 24.5},
    {"QG_MX_3.5",  3.78125, 32.0 / 3.78125},
    {"QG_MX_4.5",  4.5,     32.0 / 4.5},
    {"QG_MX_6.5",  6.5,     32.0 / 6.5},
    {"QG_MX_8.5",  8.5,     32.0 / 8.5},
    {"QG_MX_12.5", 12.5,    32.0 / 12.5},
    {"QG_MX_16.5", 16.5,    32.0 / 16.5},
    {"QG_MX_24.5", 24.5,    32.0 / 24.5},
};
inline constexpr int INFERNO_NUM_MIXES = 14;

// ── SIMD detection ───────────────────────────────────────────────────────

enum class InfernoISA : uint8_t {
    SCALAR = 0,
    SSE4   = 1,
    AVX2   = 2,
    AVX512 = 3,
};

inline const char* inferno_isa_name(InfernoISA isa) {
    switch (isa) {
        case InfernoISA::SCALAR: return "scalar";
        case InfernoISA::SSE4:   return "SSE4";
        case InfernoISA::AVX2:   return "AVX2";
        case InfernoISA::AVX512: return "AVX512";
        default:              return "unknown";
    }
}

#if defined(__AVX512F__)
inline constexpr InfernoISA INFERNO_DEFAULT_ISA = InfernoISA::AVX512;
#elif defined(__AVX2__)
inline constexpr InfernoISA INFERNO_DEFAULT_ISA = InfernoISA::AVX2;
#elif defined(__SSE4_1__)
inline constexpr InfernoISA INFERNO_DEFAULT_ISA = InfernoISA::SSE4;
#else
inline constexpr InfernoISA INFERNO_DEFAULT_ISA = InfernoISA::SCALAR;
#endif

// ── Timer ────────────────────────────────────────────────────────────────

inline uint64_t inferno_rdtsc() {
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

// ── Inferno Counter ─────────────────────────────────────────────────────────

struct InfernoCounter {
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

    void start() { start_tsc = inferno_rdtsc(); }
    void stop()  { end_tsc = inferno_rdtsc(); }

    void record(int format_index, int64_t count) {
        if (format_index < 0 || format_index >= INFERNO_NUM_FORMATS) return;
        double iw = inferno_formats[format_index].info_weight;
        total_info_ops.fetch_add((uint64_t)(count * iw));
        total_elements.fetch_add((uint64_t)count);
    }

    void record_mix(int mix_index, int64_t count) {
        if (mix_index < 0 || mix_index >= INFERNO_NUM_MIXES) return;
        double iw = inferno_mix_formats[mix_index].info_weight;
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

    // FLOPS base: 1 element = 1 MAC = 2 FLOPS (FP32 GEMM standard).
    // Every number below derives from this FLOPS value first.
    double flops() const {
        double dt = elapsed_sec();
        if (dt <= 0.0) return 0.0;
        return (double)total_elements.load() * 2.0 / dt;
    }
    double gflops() const { return flops() / 1e9; }
    double mean_info_weight() const {
        uint64_t el = total_elements.load();
        if (el == 0) return 1.0;
        return (double)total_info_ops.load() / (double)el;
    }
    double raw_ops_per_sec() const { return flops() * mean_info_weight() / 2.0; }

    double inferno()  const { return raw_ops_per_sec() / 1e21; }
    double pinferno() const { return raw_ops_per_sec() / 1e15; }
    double ninferno() const { return raw_ops_per_sec() / 1e12; }
    double uinferno() const { return raw_ops_per_sec() / 1e9; }
    double minferno() const { return raw_ops_per_sec() / 1e6; }

    double effective_gflops() const { return gflops(); }

    const char* best_unit() const {
        double v = inferno();
        if (v >= 1.0)  return "Inferno";
        v *= 1e3; if (v >= 1.0) return "mInferno";
        v *= 1e3; if (v >= 1.0) return "uInferno";
        v *= 1e3; if (v >= 1.0) return "nInferno";
        v *= 1e3; if (v >= 1.0) return "pInferno";
        return "fInferno";
    }

    double best_value() const {
        double v = inferno();
        if (v >= 1.0)  return v;
        v *= 1e3; if (v >= 1.0) return v;
        v *= 1e3; if (v >= 1.0) return v;
        v *= 1e3; if (v >= 1.0) return v;
        v *= 1e3; return v;
    }
};

// ── Helpers ──────────────────────────────────────────────────────────────

inline const char* inferno_unit_name(double val) {
    if (val >= 1.0)  return "Inferno";
    val *= 1e3; if (val >= 1.0) return "mInferno";
    val *= 1e3; if (val >= 1.0) return "uInferno";
    val *= 1e3; if (val >= 1.0) return "nInferno";
    val *= 1e3; if (val >= 1.0) return "pInferno";
    return "fInferno";
}

inline double inferno_best_value(double val) {
    if (val >= 1.0)  return val;
    val *= 1e3; if (val >= 1.0) return val;
    val *= 1e3; if (val >= 1.0) return val;
    val *= 1e3; if (val >= 1.0) return val;
    val *= 1e3; return val;
}

} // namespace quant
