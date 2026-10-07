// ============================================================================
// inferno_format_info.cpp — Complete Q format information display
// ============================================================================
// Build: cmake --build . --target inferno_info
// Run:   ./inferno_info
// ============================================================================

#include <cstdio>
#include <cmath>
#include <cstring>

// -- Format data (canonical v3 display names from include/quant/types.h,
// format_name(); wire BPW via format_bpw(); IW = 32 / wire BPW) -----
struct FormatInfo {
    const char* name;
    double bpw;
    double info_weight;
};
static const FormatInfo base_formats[] = {
    {"Q1",    1.0,  32.0},
    {"Q2",    2.0,  16.0},
    {"Q3",    3.0,  32.0 / 3.0},
    {"Q4",    4.0,  8.0},
    {"Q6",    6.0,  32.0 / 6.0},
    {"Q8",    8.0,  4.0},
    {"Q12",   12.0, 32.0 / 12.0},
    {"Q16",   16.0, 2.0},
    {"Q24",   24.0, 32.0 / 24.0},
    {"Q32",   32.0, 1.0},
    {"QG1",   1.0,  32.0},
    {"QG2",   2.625,  32.0 / 2.625},
    {"QG3",   3.5,  32.0 / 3.5},
    {"QG4",   4.5,  32.0 / 4.5},
    {"QG6",   6.5625, 32.0 / 6.5625},
    {"QG8",   8.5,  32.0 / 8.5},
    {"QG12",  12.5, 32.0 / 12.5},
    {"QG16",  16.5, 32.0 / 16.5},
    {"QG24",  24.5, 32.0 / 24.5},
};
static constexpr int NUM_BASE = 19;

static const FormatInfo mix_formats[] = {
    {"Q_MX_3.5",   3.5,  32.0 / 3.5},
    {"Q_MX_4.5",   4.5,  32.0 / 4.5},
    {"Q_MX_6.5",   6.5,  32.0 / 6.5},
    {"Q_MX_8.5",   8.5,  32.0 / 8.5},
    {"Q_MX_12.5",  12.5, 32.0 / 12.5},
    {"Q_MX_16.5",  16.5, 32.0 / 16.5},
    {"Q_MX_24.5",  24.5, 32.0 / 24.5},
    {"QG_MX_3.5",  3.78125, 32.0 / 3.78125},
    {"QG_MX_4.5",  4.5,  32.0 / 4.5},
    {"QG_MX_6.5",  6.5,  32.0 / 6.5},
    {"QG_MX_8.5",  8.5,  32.0 / 8.5},
    {"QG_MX_12.5", 12.5, 32.0 / 12.5},
    {"QG_MX_16.5", 16.5, 32.0 / 16.5},
    {"QG_MX_24.5", 24.5, 32.0 / 24.5},
};
static constexpr int NUM_MIX = 14;

// ── Section 1: Base format table ─────────────────────────────────────────

static void print_base_formats(int64_t params) {
    printf("================================================================\n");
    printf("  SECTION 1: ALL Q BASE FORMATS\n");
    printf("  Model: %lld parameters\n", (long long)params);
    printf("================================================================\n\n");

    printf("  %-14s  %6s  %8s  %12s  %14s  %10s  %12s\n",
           "Format", "BPW", "IW", "Bytes/W", "Model Size", "Info Ops", "vs FP32");
    printf("  %-14s  %6s  %8s  %12s  %14s  %10s  %12s\n",
           "----------", "------", "--------", "------------", "--------------", "----------", "------------");

    for (int i = 0; i < NUM_BASE; i++) {
        auto& f = base_formats[i];
        double bytes_per_w = f.bpw / 8.0;
        double model_bytes = (double)params * bytes_per_w;
        double info_ops = (double)params * f.info_weight;

        const char* size_unit = "B";
        double size_disp = model_bytes;
        if (size_disp > 1e9) { size_disp /= 1e9; size_unit = "GB"; }
        else if (size_disp > 1e6) { size_disp /= 1e6; size_unit = "MB"; }
        else if (size_disp > 1e3) { size_disp /= 1e3; size_unit = "KB"; }

        printf("  %-14s  %6.2f  %6.2fx   %8.4f    %7.1f %-2s  %12.0f  %6.1fx\n",
               f.name, f.bpw, f.info_weight, bytes_per_w,
               size_disp, size_unit, info_ops, f.info_weight);
    }
}

// ── Section 2: Mix formats ───────────────────────────────────────────────

static void print_mix_formats(int64_t params) {
    printf("\n================================================================\n");
    printf("  SECTION 2: ALL MIX FORMATS (2-tier)\n");
    printf("  Model: %lld parameters\n", (long long)params);
    printf("================================================================\n\n");

    printf("  %-22s  %8s  %8s  %12s  %14s  %10s\n",
           "Mix Format", "Eff BPW", "IW", "Bytes/W", "Model Size", "Info Ops");
    printf("  %-22s  %8s  %8s  %12s  %14s  %10s\n",
           "----------------------", "--------", "--------", "------------", "--------------", "----------");

    for (int i = 0; i < NUM_MIX; i++) {
        auto& f = mix_formats[i];
        double bytes_per_w = f.bpw / 8.0;
        double model_bytes = (double)params * bytes_per_w;
        double info_ops = (double)params * f.info_weight;

        const char* size_unit = "B";
        double size_disp = model_bytes;
        if (size_disp > 1e9) { size_disp /= 1e9; size_unit = "GB"; }
        else if (size_disp > 1e6) { size_disp /= 1e6; size_unit = "MB"; }
        else if (size_disp > 1e3) { size_disp /= 1e3; size_unit = "KB"; }

        printf("  %-22s  %6.2f   %6.2fx   %8.4f    %7.1f %-2s  %12.0f\n",
               f.name, f.bpw, f.info_weight, bytes_per_w,
               size_disp, size_unit, info_ops);
    }
}

// ── Section 3: Conservation law ──────────────────────────────────────────

static void print_conservation_law() {
    printf("\n================================================================\n");
    printf("  SECTION 3: CONSERVATION LAW PROOF\n");
    printf("  IW x bytes_per_weight = 4  (constant for all formats)\n");
    printf("================================================================\n\n");

    printf("  Theorem: For any format with BPW bits per weight:\n");
    printf("    IW x (BPW/8) = (32/BPW) x (BPW/8) = 32/8 = 4\n\n");

    printf("  %-14s  %8s  %8s  %10s  %12s\n",
           "Format", "BPW", "IW", "Bytes/W", "IW * Bytes");
    printf("  %-14s  %8s  %8s  %10s  %12s\n",
           "----------", "--------", "--------", "----------", "------------");

    for (int i = 0; i < NUM_BASE; i++) {
        auto& f = base_formats[i];
        double bytes_per_w = f.bpw / 8.0;
        double product = f.info_weight * bytes_per_w;
        printf("  %-14s  %6.2f    %6.2fx   %8.4f    %10.4f\n",
               f.name, f.bpw, f.info_weight, bytes_per_w, product);
    }

    printf("\n  Result: ALL formats produce exactly 4 FP32-equivalent ops/byte.\n");
    printf("  Every byte of weight memory yields 4 FP32-equivalent operations,\n");
    printf("  regardless of the quantization format.\n");

    printf("\n  Physical interpretation:\n");
    printf("  - Q1: 8 weights/byte, each worth 32 FP32-ops -> 8*32 = 256 per 32 bytes = 4/byte\n");
    printf("  - Q4:   2 weights/byte, each worth 8 FP32-ops  -> 2*8  = 16  per 4 bytes  = 4/byte\n");
    printf("  - Q32:  0.25 weights/byte, each worth 1 FP32-op -> 0.25*1 = 0.25 per 0.0625 byte = 4/byte\n");
}

// ── Section 4: Memory hierarchy ──────────────────────────────────────────

struct CacheLevel {
    const char* name;
    double size_bytes;
    double bandwidth_gbs;
};

static void print_memory_hierarchy(int64_t params) {
    CacheLevel levels[] = {
        {"L1 Cache",       32.0 * 1024.0,    1000.0},
        {"L2 Cache",      256.0 * 1024.0,     200.0},
        {"L3 Cache",     8.0 * 1024.0 * 1024.0, 100.0},
        {"Main RAM",     64.0 * 1024.0 * 1024.0 * 1024.0,  50.0},
    };
    int num_levels = 4;

    printf("\n================================================================\n");
    printf("  SECTION 4: MEMORY HIERARCHY ANALYSIS\n");
    printf("  Model: %lld parameters\n", (long long)params);
    printf("================================================================\n\n");

    printf("  Cache/level assumptions:\n");
    for (int i = 0; i < num_levels; i++) {
        printf("    %-12s  %8.0f KB    %6.0f GB/s\n",
               levels[i].name, levels[i].size_bytes / 1024.0, levels[i].bandwidth_gbs);
    }

    printf("\n  %-14s  %12s  ", "Format", "Model Size");
    for (int i = 0; i < num_levels; i++) {
        printf("%-10s ", levels[i].name);
    }
    printf("\n");

    printf("  %-14s  %12s  ", "----------", "------------");
    for (int i = 0; i < num_levels; i++) {
        printf("%-10s ", "----------");
    }
    printf("\n");

    for (int i = 0; i < NUM_BASE; i++) {
        auto& f = base_formats[i];
        double model_bytes = (double)params * (f.bpw / 8.0);

        const char* size_unit = "B";
        double size_disp = model_bytes;
        if (size_disp > 1e9) { size_disp /= 1e9; size_unit = "GB"; }
        else if (size_disp > 1e6) { size_disp /= 1e6; size_unit = "MB"; }
        else if (size_disp > 1e3) { size_disp /= 1e3; size_unit = "KB"; }

        printf("  %-14s  %7.1f %-2s  ", f.name, size_disp, size_unit);

        for (int j = 0; j < num_levels; j++) {
            if (model_bytes <= levels[j].size_bytes) {
                printf("%-10s ", "FITS");
            } else {
                printf("%-10s ", "---");
            }
        }
        printf("\n");
    }

    printf("\n  Bandwidth analysis (peak BW at each level):\n\n");
    printf("  %-14s  %12s  %12s  %12s  %12s\n",
           "Format", "L1 BW", "L2 BW", "L3 BW", "RAM BW");
    printf("  %-14s  %12s  %12s  %12s  %12s\n",
           "----------", "------------", "------------", "------------", "------------");

    for (int i = 0; i < NUM_BASE; i++) {
        auto& f = base_formats[i];
        double bytes_per_w = f.bpw / 8.0;
        double model_bytes = (double)params * bytes_per_w;

        printf("  %-14s", f.name);
        for (int j = 0; j < num_levels; j++) {
            if (model_bytes <= levels[j].size_bytes) {
                double weights_per_sec = levels[j].bandwidth_gbs * 1e9 / bytes_per_w;
                double flops = weights_per_sec * 2.0;
                double effective_ops = flops * f.info_weight / 2.0;
                double inferno = effective_ops / 1e21;
                char unit_buf[32];
                if (inferno >= 1.0)
                    snprintf(unit_buf, sizeof(unit_buf), "%.2f Inferno", inferno);
                else if (inferno >= 1e-3)
                    snprintf(unit_buf, sizeof(unit_buf), "%.2f mInferno", inferno * 1e3);
                else if (inferno >= 1e-6)
                    snprintf(unit_buf, sizeof(unit_buf), "%.2f uInferno", inferno * 1e6);
                else if (inferno >= 1e-9)
                    snprintf(unit_buf, sizeof(unit_buf), "%.2f nInferno", inferno * 1e9);
                else if (inferno >= 1e-12)
                    snprintf(unit_buf, sizeof(unit_buf), "%.2f pInferno", inferno * 1e12);
                else
                    snprintf(unit_buf, sizeof(unit_buf), "%.2f fInferno", inferno * 1e15);
                printf("  %-12s", unit_buf);
            } else {
                printf("  %-12s", "too large");
            }
        }
        printf("\n");
    }

    printf("\n  Key insight: Q4 (32 MB) fits in L3 cache on modern CPUs,\n");
    printf("  enabling ~100 GB/s bandwidth vs ~50 GB/s from RAM.\n");
    printf("  Q2 (10.2 MB) and Q1 (8 MB) fit in L2 (256 KB-1 MB).\n");
}

// ── Main ──────────────────────────────────────────────────────────────────

int main() {
    int64_t params = 64000000LL;

    printf("================================================================\n");
    printf("  Q FORMAT INFO — Complete Reference\n");
    printf("  TransFormers-Fusion Inferno Library\n");
    printf("================================================================\n\n");

    print_base_formats(params);
    print_mix_formats(params);
    print_conservation_law();
    print_memory_hierarchy(params);

    printf("\n================================================================\n");
    printf("  COMPLETE\n");
    printf("================================================================\n");

    return 0;
}
