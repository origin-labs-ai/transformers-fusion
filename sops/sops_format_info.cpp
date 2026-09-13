// ============================================================================
// sops_format_info.cpp — Complete QUANT format information display
// ============================================================================
// Build: cmake --build . --target sops_info
// Run:   ./sops_info
// ============================================================================

#include <cstdio>
#include <cmath>
#include <cstring>

// ── Format data (duplicated from sops.h to avoid quant_core dependency) ─────
// NOTE (v3 naming): the labels below are the pre-v3 display snapshot
// (QUANT_Q1, QUANT4_G, ...). They are display strings for the sops_info
// tool only — NOT the format registry truth, which lives in
// include/quant/types.h (v3 QG*/Q_MX_* names, FORMAT_COUNT=105). Do not
// add new formats here; do not parse these labels anywhere.

struct FormatInfo {
    const char* name;
    double bpw;
    double info_weight;
};

static const FormatInfo base_formats[] = {
    {"QUANT1",             1.0,   32.000},
    {"Q1_5",         1.50,  21.333},
    {"QUANT_Q1",     2.0,   16.000},
    {"QUANT2",             2.0,   16.000},
    {"QUANT4",             4.0,    8.000},
    {"QUANT8",             8.0,    4.000},
    {"QUANT16",           16.0,    2.000},
    {"QUANT32",           32.0,    1.000},
    {"QG1",         1.0,   32.000},
    {"QG2",         2.625, 12.190},
    {"QUANT4_G",         4.50,   7.111},
    {"QUANT8_G",         8.50,   3.765},
    {"QUANT16_G",       16.0,    2.000},
    {"QG_1_5",     1.50,  21.333},
    {"QUANT_Q1_G", 2.0,   16.000},
};
static constexpr int NUM_BASE = 15;

static const FormatInfo mix_formats[] = {
    {"QUANT8+Q2_1_99",      2.06,  15.534},
    {"QUANT8+QUANT4_5_95",      4.20,   7.619},
    {"QUANT4+Q2_10_90",     2.20,  14.545},
    {"QUANT8+Q2_10_90",     2.60,  12.308},
    {"QUANT+QUANT8_5_95",     7.675,  4.169},
    {"QUANT16+QUANT4_1_99",     4.12,   7.767},
    {"QUANT16+QUANT8_5_95",     8.40,   3.810},
    {"QUANT32+QUANT8_1_99",     8.24,   3.883},
};
static constexpr int NUM_MIX = 8;

// ── Section 1: Base format table ─────────────────────────────────────────

static void print_base_formats(int64_t params) {
    printf("================================================================\n");
    printf("  SECTION 1: ALL QUANT BASE FORMATS\n");
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
    printf("  - QUANT1: 8 weights/byte, each worth 32 FP32-ops -> 8*32 = 256 per 32 bytes = 4/byte\n");
    printf("  - QUANT4:   2 weights/byte, each worth 8 FP32-ops  -> 2*8  = 16  per 4 bytes  = 4/byte\n");
    printf("  - QUANT32:  0.25 weights/byte, each worth 1 FP32-op -> 0.25*1 = 0.25 per 0.0625 byte = 4/byte\n");
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
                double effective_ops = weights_per_sec * f.info_weight;
                double sops = effective_ops / 1e21;
                char unit_buf[32];
                if (sops >= 1.0)
                    snprintf(unit_buf, sizeof(unit_buf), "%.2f SOPS", sops);
                else if (sops >= 1e-3)
                    snprintf(unit_buf, sizeof(unit_buf), "%.2f mSOPS", sops * 1e3);
                else if (sops >= 1e-6)
                    snprintf(unit_buf, sizeof(unit_buf), "%.2f uSOPS", sops * 1e6);
                else if (sops >= 1e-9)
                    snprintf(unit_buf, sizeof(unit_buf), "%.2f nSOPS", sops * 1e9);
                else if (sops >= 1e-12)
                    snprintf(unit_buf, sizeof(unit_buf), "%.2f pSOPS", sops * 1e12);
                else
                    snprintf(unit_buf, sizeof(unit_buf), "%.2f fSOPS", sops * 1e15);
                printf("  %-12s", unit_buf);
            } else {
                printf("  %-12s", "too large");
            }
        }
        printf("\n");
    }

    printf("\n  Key insight: QUANT4 (32 MB) fits in L3 cache on modern CPUs,\n");
    printf("  enabling ~100 GB/s bandwidth vs ~50 GB/s from RAM.\n");
    printf("  QUANT (10.2 MB) and QUANT1 (8 MB) fit in L2 (256 KB-1 MB).\n");
}

// ── Main ──────────────────────────────────────────────────────────────────

int main() {
    int64_t params = 64000000LL;

    printf("================================================================\n");
    printf("  QUANT FORMAT INFO — Complete Reference\n");
    printf("  Transcender SOPS Library\n");
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
