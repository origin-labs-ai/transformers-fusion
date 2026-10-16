#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cassert>
namespace quant {
enum class Activation : uint8_t { None, ReLU, GELU, SiLU, SwiGLU, GeGLU };
enum class RoPEScalingMode : uint8_t { None, Linear, NTK, YARN };
// ============================================================
// Format enum — Fusion canonical quantization formats (Q_ / Q_G_ / Q_MX / Q_G_MX)
// Rules: exact BPW (name == claimed BPW), 3 K variants per BPW (L/M/H),
// GRP exact at same BPW (no extra), half-BPW GRP as Q_G_X_Y,
// Mix only as Q_MX_* / QG_MX_* (formerly known as MXQ & formerly known as QMX; 4-variant mix, no TWI), no TWI_MIX.
// ============================================================
enum class Format : uint8_t {
    // --- Base integer BPW (10) ---
    Q_1              = 0,   // 1.00
    Q_2              = 1,   // 2.00
    Q_3              = 2,   // 3.00
    Q_4              = 3,   // 4.00
    Q_6              = 4,   // 6.00
    Q_8              = 5,   // 8.00
    Q_12             = 6,   // 12.00
    Q_16             = 7,   // 16.00
    Q_24             = 8,   // 24.00
    Q_32             = 9,   // 32.00
    // --- K variants integer BPW (30) ---
    Q_1_K_L          = 10, Q_1_K_M          = 11, Q_1_K_H          = 12,
    Q_2_K_L          = 13, Q_2_K_M          = 14, Q_2_K_H          = 15,
    Q_3_K_L          = 16, Q_3_K_M          = 17, Q_3_K_H          = 18,
    Q_4_K_L          = 19, Q_4_K_M          = 20, Q_4_K_H          = 21,
    Q_6_K_L          = 22, Q_6_K_M          = 23, Q_6_K_H          = 24,
    Q_8_K_L          = 25, Q_8_K_M          = 26, Q_8_K_H          = 27,
    Q_12_K_L         = 28, Q_12_K_M         = 29, Q_12_K_H         = 30,
    Q_16_K_L         = 31, Q_16_K_M         = 32, Q_16_K_H         = 33,
    Q_24_K_L         = 34, Q_24_K_M         = 35, Q_24_K_H         = 36,
    // --- GRP exact integer BPW (9) ---
    Q_G_1          = 37, Q_G_2          = 38, Q_G_3          = 39, Q_G_4          = 40, Q_G_6          = 41,
    Q_G_8          = 42, Q_G_12         = 43, Q_G_16         = 44, Q_G_24         = 45,
    // --- K_G exact integer BPW (27) ---
    Q_G_1_K_L      = 46, Q_G_1_K_M      = 47, Q_G_1_K_H      = 48,
    Q_G_2_K_L      = 49, Q_G_2_K_M      = 50, Q_G_2_K_H      = 51,
    Q_G_3_K_L      = 52, Q_G_3_K_M      = 53, Q_G_3_K_H      = 54,
    Q_G_4_K_L      = 55, Q_G_4_K_M      = 56, Q_G_4_K_H      = 57,
    Q_G_6_K_L      = 58, Q_G_6_K_M      = 59, Q_G_6_K_H      = 60,
    Q_G_8_K_L      = 61, Q_G_8_K_M      = 62, Q_G_8_K_H      = 63,
    Q_G_12_K_L     = 64, Q_G_12_K_M     = 65, Q_G_12_K_H     = 66,
    Q_G_16_K_L     = 67, Q_G_16_K_M     = 68, Q_G_16_K_H     = 69,
    Q_G_24_K_L     = 70, Q_G_24_K_M     = 71, Q_G_24_K_H     = 72,
    // --- Half BPW plain (9) ---
    Q_1_5            = 73, Q_2_5            = 74, Q_3_5            = 75, Q_4_5            = 76, Q_6_5            = 77,
    Q_8_5            = 78, Q_12_5           = 79, Q_16_5           = 80, Q_24_5           = 81,
    // --- Half BPW GRP (9) ---
    Q_G_1_5       = 82, Q_G_2_5       = 83, Q_G_3_5       = 84, Q_G_4_5       = 85, Q_G_6_5       = 86,
    Q_G_8_5       = 87, Q_G_12_5      = 88, Q_G_16_5      = 89, Q_G_24_5      = 90,
    // --- Mix 4-variant (7 plain Q_MX + 7 grouped QG_MX, formerly MXQ & formerly QMX) ---
    Q_MX_3_5         = 91, Q_MX_4_5         = 92, Q_MX_6_5         = 93, Q_MX_8_5         = 94, Q_MX_12_5        = 95, Q_MX_16_5        = 96, Q_MX_24_5        = 97,
    Q_G_MX_3_5     = 98, Q_G_MX_4_5     = 99, Q_G_MX_6_5     = 100, Q_G_MX_8_5     = 101, Q_G_MX_12_5    = 102, Q_G_MX_16_5    = 103, Q_G_MX_24_5    = 104,

    // --- Legacy pre-Fusion aliases (deprecated, same IDs) ---
    Q1 = 0,  // legacy alias for Q_1
    Q2 = 1,  // legacy alias for Q_2
    Q3 = 2,  // legacy alias for Q_3
    Q4 = 3,  // legacy alias for Q_4
    Q6 = 4,  // legacy alias for Q_6
    Q8 = 5,  // legacy alias for Q_8
    Q12 = 6,  // legacy alias for Q_12
    Q16 = 7,  // legacy alias for Q_16
    Q24 = 8,  // legacy alias for Q_24
    Q32 = 9,  // legacy alias for Q_32
    Q1_K_L = 10,  // legacy alias for Q_1_K_L
    Q1_K_M = 11,  // legacy alias for Q_1_K_M
    Q1_K_H = 12,  // legacy alias for Q_1_K_H
    Q2_K_L = 13,  // legacy alias for Q_2_K_L
    Q2_K_M = 14,  // legacy alias for Q_2_K_M
    Q2_K_H = 15,  // legacy alias for Q_2_K_H
    Q3_K_L = 16,  // legacy alias for Q_3_K_L
    Q3_K_M = 17,  // legacy alias for Q_3_K_M
    Q3_K_H = 18,  // legacy alias for Q_3_K_H
    Q4_K_L = 19,  // legacy alias for Q_4_K_L
    Q4_K_M = 20,  // legacy alias for Q_4_K_M
    Q4_K_H = 21,  // legacy alias for Q_4_K_H
    Q6_K_L = 22,  // legacy alias for Q_6_K_L
    Q6_K_M = 23,  // legacy alias for Q_6_K_M
    Q6_K_H = 24,  // legacy alias for Q_6_K_H
    Q8_K_L = 25,  // legacy alias for Q_8_K_L
    Q8_K_M = 26,  // legacy alias for Q_8_K_M
    Q8_K_H = 27,  // legacy alias for Q_8_K_H
    Q12_K_L = 28,  // legacy alias for Q_12_K_L
    Q12_K_M = 29,  // legacy alias for Q_12_K_M
    Q12_K_H = 30,  // legacy alias for Q_12_K_H
    Q16_K_L = 31,  // legacy alias for Q_16_K_L
    Q16_K_M = 32,  // legacy alias for Q_16_K_M
    Q16_K_H = 33,  // legacy alias for Q_16_K_H
    Q24_K_L = 34,  // legacy alias for Q_24_K_L
    Q24_K_M = 35,  // legacy alias for Q_24_K_M
    Q24_K_H = 36,  // legacy alias for Q_24_K_H
    QG1 = 37,  // legacy alias for Q_G_1
    QG2 = 38,  // legacy alias for Q_G_2
    QG3 = 39,  // legacy alias for Q_G_3
    QG4 = 40,  // legacy alias for Q_G_4
    QG6 = 41,  // legacy alias for Q_G_6
    QG8 = 42,  // legacy alias for Q_G_8
    QG12 = 43,  // legacy alias for Q_G_12
    QG16 = 44,  // legacy alias for Q_G_16
    QG24 = 45,  // legacy alias for Q_G_24
    QG_1_K_L = 46,  // legacy alias for Q_G_1_K_L
    QG_1_K_M = 47,  // legacy alias for Q_G_1_K_M
    QG_1_K_H = 48,  // legacy alias for Q_G_1_K_H
    QG_2_K_L = 49,  // legacy alias for Q_G_2_K_L
    QG_2_K_M = 50,  // legacy alias for Q_G_2_K_M
    QG_2_K_H = 51,  // legacy alias for Q_G_2_K_H
    QG_3_K_L = 52,  // legacy alias for Q_G_3_K_L
    QG_3_K_M = 53,  // legacy alias for Q_G_3_K_M
    QG_3_K_H = 54,  // legacy alias for Q_G_3_K_H
    QG_4_K_L = 55,  // legacy alias for Q_G_4_K_L
    QG_4_K_M = 56,  // legacy alias for Q_G_4_K_M
    QG_4_K_H = 57,  // legacy alias for Q_G_4_K_H
    QG_6_K_L = 58,  // legacy alias for Q_G_6_K_L
    QG_6_K_M = 59,  // legacy alias for Q_G_6_K_M
    QG_6_K_H = 60,  // legacy alias for Q_G_6_K_H
    QG_8_K_L = 61,  // legacy alias for Q_G_8_K_L
    QG_8_K_M = 62,  // legacy alias for Q_G_8_K_M
    QG_8_K_H = 63,  // legacy alias for Q_G_8_K_H
    QG_12_K_L = 64,  // legacy alias for Q_G_12_K_L
    QG_12_K_M = 65,  // legacy alias for Q_G_12_K_M
    QG_12_K_H = 66,  // legacy alias for Q_G_12_K_H
    QG_16_K_L = 67,  // legacy alias for Q_G_16_K_L
    QG_16_K_M = 68,  // legacy alias for Q_G_16_K_M
    QG_16_K_H = 69,  // legacy alias for Q_G_16_K_H
    QG_24_K_L = 70,  // legacy alias for Q_G_24_K_L
    QG_24_K_M = 71,  // legacy alias for Q_G_24_K_M
    QG_24_K_H = 72,  // legacy alias for Q_G_24_K_H
    Q1_5 = 73,  // legacy alias for Q_1_5
    Q2_5 = 74,  // legacy alias for Q_2_5
    Q3_5 = 75,  // legacy alias for Q_3_5
    Q4_5 = 76,  // legacy alias for Q_4_5
    Q6_5 = 77,  // legacy alias for Q_6_5
    Q8_5 = 78,  // legacy alias for Q_8_5
    Q12_5 = 79,  // legacy alias for Q_12_5
    Q16_5 = 80,  // legacy alias for Q_16_5
    Q24_5 = 81,  // legacy alias for Q_24_5
    QG_1_5 = 82,  // legacy alias for Q_G_1_5
    QG_2_5 = 83,  // legacy alias for Q_G_2_5
    QG_3_5 = 84,  // legacy alias for Q_G_3_5
    QG_4_5 = 85,  // legacy alias for Q_G_4_5
    QG_6_5 = 86,  // legacy alias for Q_G_6_5
    QG_8_5 = 87,  // legacy alias for Q_G_8_5
    QG_12_5 = 88,  // legacy alias for Q_G_12_5
    QG_16_5 = 89,  // legacy alias for Q_G_16_5
    QG_24_5 = 90,  // legacy alias for Q_G_24_5
    QG_MX_3_5 = 98,  // legacy alias for Q_G_MX_3_5
    QG_MX_4_5 = 99,  // legacy alias for Q_G_MX_4_5
    QG_MX_6_5 = 100,  // legacy alias for Q_G_MX_6_5
    QG_MX_8_5 = 101,  // legacy alias for Q_G_MX_8_5
    QG_MX_12_5 = 102,  // legacy alias for Q_G_MX_12_5
    QG_MX_16_5 = 103,  // legacy alias for Q_G_MX_16_5
    QG_MX_24_5 = 104,  // legacy alias for Q_G_MX_24_5
};
constexpr int FORMAT_COUNT = 105;
inline const char* format_name(Format f) {
    switch(f) {
        case Format::Q_1: return "Q_1"; case Format::Q_2: return "Q_2"; case Format::Q_3: return "Q_3"; case Format::Q_4: return "Q_4"; case Format::Q_6: return "Q_6"; case Format::Q_8: return "Q_8"; case Format::Q_12: return "Q_12"; case Format::Q_16: return "Q_16"; case Format::Q_24: return "Q_24"; case Format::Q_32: return "Q_32";
        case Format::Q_1_K_L: return "Q_1_K_L"; case Format::Q_1_K_M: return "Q_1_K_M"; case Format::Q_1_K_H: return "Q_1_K_H";
        case Format::Q_2_K_L: return "Q_2_K_L"; case Format::Q_2_K_M: return "Q_2_K_M"; case Format::Q_2_K_H: return "Q_2_K_H";
        case Format::Q_3_K_L: return "Q_3_K_L"; case Format::Q_3_K_M: return "Q_3_K_M"; case Format::Q_3_K_H: return "Q_3_K_H";
        case Format::Q_4_K_L: return "Q_4_K_L"; case Format::Q_4_K_M: return "Q_4_K_M"; case Format::Q_4_K_H: return "Q_4_K_H";
        case Format::Q_6_K_L: return "Q_6_K_L"; case Format::Q_6_K_M: return "Q_6_K_M"; case Format::Q_6_K_H: return "Q_6_K_H";
        case Format::Q_8_K_L: return "Q_8_K_L"; case Format::Q_8_K_M: return "Q_8_K_M"; case Format::Q_8_K_H: return "Q_8_K_H";
        case Format::Q_12_K_L: return "Q_12_K_L"; case Format::Q_12_K_M: return "Q_12_K_M"; case Format::Q_12_K_H: return "Q_12_K_H";
        case Format::Q_16_K_L: return "Q_16_K_L"; case Format::Q_16_K_M: return "Q_16_K_M"; case Format::Q_16_K_H: return "Q_16_K_H";
        case Format::Q_24_K_L: return "Q_24_K_L"; case Format::Q_24_K_M: return "Q_24_K_M"; case Format::Q_24_K_H: return "Q_24_K_H";
        case Format::Q_G_1: return "Q_G_1"; case Format::Q_G_2: return "Q_G_2"; case Format::Q_G_3: return "Q_G_3"; case Format::Q_G_4: return "Q_G_4"; case Format::Q_G_6: return "Q_G_6"; case Format::Q_G_8: return "Q_G_8"; case Format::Q_G_12: return "Q_G_12"; case Format::Q_G_16: return "Q_G_16"; case Format::Q_G_24: return "Q_G_24";
        case Format::Q_G_1_K_L: return "Q_G_1_K_L"; case Format::Q_G_1_K_M: return "Q_G_1_K_M"; case Format::Q_G_1_K_H: return "Q_G_1_K_H";
        case Format::Q_G_2_K_L: return "Q_G_2_K_L"; case Format::Q_G_2_K_M: return "Q_G_2_K_M"; case Format::Q_G_2_K_H: return "Q_G_2_K_H";
        case Format::Q_G_3_K_L: return "Q_G_3_K_L"; case Format::Q_G_3_K_M: return "Q_G_3_K_M"; case Format::Q_G_3_K_H: return "Q_G_3_K_H";
        case Format::Q_G_4_K_L: return "Q_G_4_K_L"; case Format::Q_G_4_K_M: return "Q_G_4_K_M"; case Format::Q_G_4_K_H: return "Q_G_4_K_H";
        case Format::Q_G_6_K_L: return "Q_G_6_K_L"; case Format::Q_G_6_K_M: return "Q_G_6_K_M"; case Format::Q_G_6_K_H: return "Q_G_6_K_H";
        case Format::Q_G_8_K_L: return "Q_G_8_K_L"; case Format::Q_G_8_K_M: return "Q_G_8_K_M"; case Format::Q_G_8_K_H: return "Q_G_8_K_H";
        case Format::Q_G_12_K_L: return "Q_G_12_K_L"; case Format::Q_G_12_K_M: return "Q_G_12_K_M"; case Format::Q_G_12_K_H: return "Q_G_12_K_H";
        case Format::Q_G_16_K_L: return "Q_G_16_K_L"; case Format::Q_G_16_K_M: return "Q_G_16_K_M"; case Format::Q_G_16_K_H: return "Q_G_16_K_H";
        case Format::Q_G_24_K_L: return "Q_G_24_K_L"; case Format::Q_G_24_K_M: return "Q_G_24_K_M"; case Format::Q_G_24_K_H: return "Q_G_24_K_H";
        case Format::Q_1_5: return "Q_1_5"; case Format::Q_2_5: return "Q_2_5"; case Format::Q_3_5: return "Q_3_5"; case Format::Q_4_5: return "Q_4_5"; case Format::Q_6_5: return "Q_6_5"; case Format::Q_8_5: return "Q_8_5"; case Format::Q_12_5: return "Q_12_5"; case Format::Q_16_5: return "Q_16_5"; case Format::Q_24_5: return "Q_24_5";
        case Format::Q_G_1_5: return "Q_G_1_5"; case Format::Q_G_2_5: return "Q_G_2_5"; case Format::Q_G_3_5: return "Q_G_3_5"; case Format::Q_G_4_5: return "Q_G_4_5"; case Format::Q_G_6_5: return "Q_G_6_5"; case Format::Q_G_8_5: return "Q_G_8_5"; case Format::Q_G_12_5: return "Q_G_12_5"; case Format::Q_G_16_5: return "Q_G_16_5"; case Format::Q_G_24_5: return "Q_G_24_5";
        case Format::Q_MX_3_5: return "Q_MX_3_5"; case Format::Q_MX_4_5: return "Q_MX_4_5"; case Format::Q_MX_6_5: return "Q_MX_6_5"; case Format::Q_MX_8_5: return "Q_MX_8_5"; case Format::Q_MX_12_5: return "Q_MX_12_5"; case Format::Q_MX_16_5: return "Q_MX_16_5"; case Format::Q_MX_24_5: return "Q_MX_24_5";
        case Format::Q_G_MX_3_5: return "Q_G_MX_3_5"; case Format::Q_G_MX_4_5: return "Q_G_MX_4_5"; case Format::Q_G_MX_6_5: return "Q_G_MX_6_5"; case Format::Q_G_MX_8_5: return "Q_G_MX_8_5"; case Format::Q_G_MX_12_5: return "Q_G_MX_12_5"; case Format::Q_G_MX_16_5: return "Q_G_MX_16_5"; case Format::Q_G_MX_24_5: return "Q_G_MX_24_5";
        default: return "unknown";
    }
}
inline float format_bpw(Format f) {
    switch(f) {
        case Format::Q_1: return 1.0f; case Format::Q_2: return 2.0f; case Format::Q_3: return 3.0f; case Format::Q_4: return 4.0f; case Format::Q_6: return 6.0f; case Format::Q_8: return 8.0f; case Format::Q_12: return 12.0f; case Format::Q_16: return 16.0f; case Format::Q_24: return 24.0f; case Format::Q_32: return 32.0f;
        case Format::Q_1_K_L: case Format::Q_1_K_M: case Format::Q_1_K_H: return 1.0f;
        case Format::Q_2_K_L: case Format::Q_2_K_M: case Format::Q_2_K_H: return 2.0f;
        case Format::Q_3_K_L: case Format::Q_3_K_M: case Format::Q_3_K_H: return 3.0f;
        case Format::Q_4_K_L: case Format::Q_4_K_M: case Format::Q_4_K_H: return 4.0f;
        case Format::Q_6_K_L: case Format::Q_6_K_M: case Format::Q_6_K_H: return 6.0f;
        case Format::Q_8_K_L: case Format::Q_8_K_M: case Format::Q_8_K_H: return 8.0f;
        case Format::Q_12_K_L: case Format::Q_12_K_M: case Format::Q_12_K_H: return 12.0f;
        case Format::Q_16_K_L: case Format::Q_16_K_M: case Format::Q_16_K_H: return 16.0f;
        case Format::Q_24_K_L: case Format::Q_24_K_M: case Format::Q_24_K_H: return 24.0f;
        // TRUTHFUL BPW (audit 2026-08-26 option-c): these report the wire
        // budget the canonical encoder actually spends at n%256 (probe:
        // tests/test_format_audit.cpp). Names stay Q*_G for API stability.
        case Format::Q_G_1: return 1.0f; case Format::Q_G_2: return 2.625f; case Format::Q_G_3: return 3.5f; case Format::Q_G_4: return 4.5f; case Format::Q_G_6: return 6.5625f; case Format::Q_G_8: return 8.5f; case Format::Q_G_12: return 12.5f; case Format::Q_G_16: return 16.5f; case Format::Q_G_24: return 24.5f;
        case Format::Q_G_1_K_L: case Format::Q_G_1_K_M: case Format::Q_G_1_K_H: return 1.0f;
        case Format::Q_G_2_K_L: case Format::Q_G_2_K_M: case Format::Q_G_2_K_H: return 2.4375f;
        case Format::Q_G_3_K_L: case Format::Q_G_3_K_M: case Format::Q_G_3_K_H: return 3.4375f;
        case Format::Q_G_4_K_L: case Format::Q_G_4_K_M: case Format::Q_G_4_K_H: return 4.4375f;
        case Format::Q_G_6_K_L: case Format::Q_G_6_K_M: case Format::Q_G_6_K_H: return 6.5625f;
        case Format::Q_G_8_K_L: case Format::Q_G_8_K_M: case Format::Q_G_8_K_H: return 8.5f;
        case Format::Q_G_12_K_L: case Format::Q_G_12_K_M: case Format::Q_G_12_K_H: return 12.5f;
        case Format::Q_G_16_K_L: case Format::Q_G_16_K_M: case Format::Q_G_16_K_H: return 16.0f;
        case Format::Q_G_24_K_L: case Format::Q_G_24_K_M: case Format::Q_G_24_K_H: return 24.0f;
        case Format::Q_1_5: return 1.5f; case Format::Q_2_5: return 2.4375f; case Format::Q_3_5: return 3.5f; case Format::Q_4_5: return 4.5f; case Format::Q_6_5: return 6.4375f; case Format::Q_8_5: return 8.5f; case Format::Q_12_5: return 12.5f; case Format::Q_16_5: return 16.5f; case Format::Q_24_5: return 24.5f;
        case Format::Q_G_1_5: return 1.5f; case Format::Q_G_2_5: return 2.4375f; case Format::Q_G_3_5: return 3.5f; case Format::Q_G_4_5: return 4.5f; case Format::Q_G_6_5: return 6.4375f; case Format::Q_G_8_5: return 8.5f; case Format::Q_G_12_5: return 12.5f; case Format::Q_G_16_5: return 16.5f; case Format::Q_G_24_5: return 24.5f;
        case Format::Q_MX_3_5: return 3.5f; case Format::Q_MX_4_5: return 4.5f; case Format::Q_MX_6_5: return 6.5f; case Format::Q_MX_8_5: return 8.5f; case Format::Q_MX_12_5: return 12.5f; case Format::Q_MX_16_5: return 16.5f; case Format::Q_MX_24_5: return 24.5f;
        // QG_MX true wire BPW at canonical n=256 (per-32 dom scales + per-tier
        // non-dom scales). QG_MX_3.5 has 92% sign tier → 8 dom scales → 3.78125.
        case Format::Q_G_MX_3_5: return 3.78125f; case Format::Q_G_MX_4_5: return 4.5f; case Format::Q_G_MX_6_5: return 6.5f; case Format::Q_G_MX_8_5: return 8.5f; case Format::Q_G_MX_12_5: return 12.5f; case Format::Q_G_MX_16_5: return 16.5f; case Format::Q_G_MX_24_5: return 24.5f;
        default: return 0;
    }
}
// Rule (BPW dual-accessor): names are nominal family labels, format_bpw()
// is wire truth (measured canonical spend at n%256==0, audit 2026-08-26
// option-c). format_nominal_bpw() reports the name-implied density (family
// label, no wire overhead); format_wire_bpw() reports the measured canonical
// wire and always equals format_bpw().
inline float format_nominal_bpw(Format f) {
    switch(f) {
        case Format::Q_1: return 1.0f; case Format::Q_2: return 2.0f; case Format::Q_3: return 3.0f; case Format::Q_4: return 4.0f; case Format::Q_6: return 6.0f; case Format::Q_8: return 8.0f; case Format::Q_12: return 12.0f; case Format::Q_16: return 16.0f; case Format::Q_24: return 24.0f; case Format::Q_32: return 32.0f;
        case Format::Q_1_K_L: case Format::Q_1_K_M: case Format::Q_1_K_H: return 1.0f;
        case Format::Q_2_K_L: case Format::Q_2_K_M: case Format::Q_2_K_H: return 2.0f;
        case Format::Q_3_K_L: case Format::Q_3_K_M: case Format::Q_3_K_H: return 3.0f;
        case Format::Q_4_K_L: case Format::Q_4_K_M: case Format::Q_4_K_H: return 4.0f;
        case Format::Q_6_K_L: case Format::Q_6_K_M: case Format::Q_6_K_H: return 6.0f;
        case Format::Q_8_K_L: case Format::Q_8_K_M: case Format::Q_8_K_H: return 8.0f;
        case Format::Q_12_K_L: case Format::Q_12_K_M: case Format::Q_12_K_H: return 12.0f;
        case Format::Q_16_K_L: case Format::Q_16_K_M: case Format::Q_16_K_H: return 16.0f;
        case Format::Q_24_K_L: case Format::Q_24_K_M: case Format::Q_24_K_H: return 24.0f;
        case Format::Q_G_1: return 1.0f; case Format::Q_G_2: return 2.0f; case Format::Q_G_3: return 3.0f; case Format::Q_G_4: return 4.0f; case Format::Q_G_6: return 6.0f; case Format::Q_G_8: return 8.0f; case Format::Q_G_12: return 12.0f; case Format::Q_G_16: return 16.0f; case Format::Q_G_24: return 24.0f;
        case Format::Q_G_1_K_L: case Format::Q_G_1_K_M: case Format::Q_G_1_K_H: return 1.0f;
        case Format::Q_G_2_K_L: case Format::Q_G_2_K_M: case Format::Q_G_2_K_H: return 2.0f;
        case Format::Q_G_3_K_L: case Format::Q_G_3_K_M: case Format::Q_G_3_K_H: return 3.0f;
        case Format::Q_G_4_K_L: case Format::Q_G_4_K_M: case Format::Q_G_4_K_H: return 4.0f;
        case Format::Q_G_6_K_L: case Format::Q_G_6_K_M: case Format::Q_G_6_K_H: return 6.0f;
        case Format::Q_G_8_K_L: case Format::Q_G_8_K_M: case Format::Q_G_8_K_H: return 8.0f;
        case Format::Q_G_12_K_L: case Format::Q_G_12_K_M: case Format::Q_G_12_K_H: return 12.0f;
        case Format::Q_G_16_K_L: case Format::Q_G_16_K_M: case Format::Q_G_16_K_H: return 16.0f;
        case Format::Q_G_24_K_L: case Format::Q_G_24_K_M: case Format::Q_G_24_K_H: return 24.0f;
        case Format::Q_1_5: return 1.5f; case Format::Q_2_5: return 2.5f; case Format::Q_3_5: return 3.5f; case Format::Q_4_5: return 4.5f; case Format::Q_6_5: return 6.5f; case Format::Q_8_5: return 8.5f; case Format::Q_12_5: return 12.5f; case Format::Q_16_5: return 16.5f; case Format::Q_24_5: return 24.5f;
        case Format::Q_G_1_5: return 1.5f; case Format::Q_G_2_5: return 2.5f; case Format::Q_G_3_5: return 3.5f; case Format::Q_G_4_5: return 4.5f; case Format::Q_G_6_5: return 6.5f; case Format::Q_G_8_5: return 8.5f; case Format::Q_G_12_5: return 12.5f; case Format::Q_G_16_5: return 16.5f; case Format::Q_G_24_5: return 24.5f;
        case Format::Q_MX_3_5: return 3.5f; case Format::Q_MX_4_5: return 4.5f; case Format::Q_MX_6_5: return 6.5f; case Format::Q_MX_8_5: return 8.5f; case Format::Q_MX_12_5: return 12.5f; case Format::Q_MX_16_5: return 16.5f; case Format::Q_MX_24_5: return 24.5f;
        case Format::Q_G_MX_3_5: return 3.5f; case Format::Q_G_MX_4_5: return 4.5f; case Format::Q_G_MX_6_5: return 6.5f; case Format::Q_G_MX_8_5: return 8.5f; case Format::Q_G_MX_12_5: return 12.5f; case Format::Q_G_MX_16_5: return 16.5f; case Format::Q_G_MX_24_5: return 24.5f;
        default: return 0;
    }
}
inline float format_wire_bpw(Format f) {
    return format_bpw(f);
}
inline bool format_is_base(Format f) { auto v=(int)f; return v<=9; }
inline bool format_is_k(Format f) { auto v=(int)f; return (v>=10&&v<=36)||(v>=46&&v<=72); }
inline bool format_is_grp(Format f) { auto v=(int)f; return (v>=37&&v<=45)||(v>=82&&v<=90)||(v>=98&&v<=104); }
inline bool format_is_half(Format f) { auto v=(int)f; return (v>=73&&v<=81)||(v>=82&&v<=90); }
inline bool format_is_mx(Format f) { auto v=(int)f; return v>=91; }
inline bool format_is_mx_plain(Format f){ auto v=(int)f; return v>=91&&v<=97; }
inline bool format_is_mixed(Format f){ return format_is_mx(f); }
inline bool format_is_twi_mix(Format f){ return false; }
inline bool format_is_quad_mix(Format f){ return format_is_mx(f); }
inline int format_codebook_size(Format f){
    switch(f){
        case Format::Q_1: case Format::Q_1_K_L: case Format::Q_1_K_M: case Format::Q_1_K_H: case Format::Q_G_1: case Format::Q_G_1_K_L: case Format::Q_G_1_K_M: case Format::Q_G_1_K_H: case Format::Q_1_5: case Format::Q_G_1_5: return 1;
        case Format::Q_2: case Format::Q_2_K_L: case Format::Q_2_K_M: case Format::Q_2_K_H: case Format::Q_G_2: case Format::Q_G_2_K_L: case Format::Q_G_2_K_M: case Format::Q_G_2_K_H: case Format::Q_2_5: case Format::Q_G_2_5: return 4;
        case Format::Q_3: case Format::Q_3_K_L: case Format::Q_3_K_M: case Format::Q_3_K_H: case Format::Q_G_3: case Format::Q_G_3_K_L: case Format::Q_G_3_K_M: case Format::Q_G_3_K_H: case Format::Q_3_5: case Format::Q_G_3_5: return 8;
        case Format::Q_4: case Format::Q_4_K_L: case Format::Q_4_K_M: case Format::Q_4_K_H: case Format::Q_G_4: case Format::Q_G_4_K_L: case Format::Q_G_4_K_M: case Format::Q_G_4_K_H: case Format::Q_4_5: case Format::Q_G_4_5: return 16;
        case Format::Q_6: case Format::Q_6_K_L: case Format::Q_6_K_M: case Format::Q_6_K_H: case Format::Q_G_6: case Format::Q_G_6_K_L: case Format::Q_G_6_K_M: case Format::Q_G_6_K_H: case Format::Q_6_5: case Format::Q_G_6_5: return 64;
        case Format::Q_8: case Format::Q_8_K_L: case Format::Q_8_K_M: case Format::Q_8_K_H: case Format::Q_G_8: case Format::Q_G_8_K_L: case Format::Q_G_8_K_M: case Format::Q_G_8_K_H: case Format::Q_8_5: case Format::Q_G_8_5: return 256;
        case Format::Q_12: case Format::Q_12_K_L: case Format::Q_12_K_M: case Format::Q_12_K_H: case Format::Q_G_12: case Format::Q_G_12_K_L: case Format::Q_G_12_K_M: case Format::Q_G_12_K_H: case Format::Q_12_5: case Format::Q_G_12_5: return 4096;
        default: return 0;
    }
}
enum class DType : uint8_t { I64,I32,U8,U4,U16,F16,F32 };
inline size_t dtype_size(DType dt){ switch(dt){case DType::I64:return 8;case DType::I32:return 4;case DType::U8:return 1;case DType::U4:return 1;case DType::U16:return 2;case DType::F16:return 2;case DType::F32:return 4;default:return 0;}}
inline DType format_to_dtype(Format f){
    switch(f){
        case Format::Q_1: case Format::Q_1_K_L: case Format::Q_1_K_M: case Format::Q_1_K_H: case Format::Q_G_1: case Format::Q_G_1_K_L: case Format::Q_G_1_K_M: case Format::Q_G_1_K_H: case Format::Q_1_5: case Format::Q_G_1_5: return DType::U8;
        case Format::Q_2: case Format::Q_2_K_L: case Format::Q_2_K_M: case Format::Q_2_K_H: case Format::Q_G_2: case Format::Q_G_2_K_L: case Format::Q_G_2_K_M: case Format::Q_G_2_K_H: case Format::Q_2_5: case Format::Q_G_2_5: return DType::U8;
        case Format::Q_3: case Format::Q_3_K_L: case Format::Q_3_K_M: case Format::Q_3_K_H: case Format::Q_G_3: case Format::Q_G_3_K_L: case Format::Q_G_3_K_M: case Format::Q_G_3_K_H: case Format::Q_3_5: case Format::Q_G_3_5: return DType::U8;
        case Format::Q_4: case Format::Q_4_K_L: case Format::Q_4_K_M: case Format::Q_4_K_H: case Format::Q_G_4: case Format::Q_G_4_K_L: case Format::Q_G_4_K_M: case Format::Q_G_4_K_H: case Format::Q_4_5: case Format::Q_G_4_5: return DType::U4;
        case Format::Q_6: case Format::Q_6_K_L: case Format::Q_6_K_M: case Format::Q_6_K_H: case Format::Q_G_6: case Format::Q_G_6_K_L: case Format::Q_G_6_K_M: case Format::Q_G_6_K_H: case Format::Q_6_5: case Format::Q_G_6_5: return DType::U8;
        case Format::Q_8: case Format::Q_8_K_L: case Format::Q_8_K_M: case Format::Q_8_K_H: case Format::Q_G_8: case Format::Q_G_8_K_L: case Format::Q_G_8_K_M: case Format::Q_G_8_K_H: case Format::Q_8_5: case Format::Q_G_8_5: return DType::U8;
        case Format::Q_12: case Format::Q_12_K_L: case Format::Q_12_K_M: case Format::Q_12_K_H: case Format::Q_G_12: case Format::Q_G_12_K_L: case Format::Q_G_12_K_M: case Format::Q_G_12_K_H: case Format::Q_12_5: case Format::Q_G_12_5: return DType::U16;
        case Format::Q_16: case Format::Q_16_K_L: case Format::Q_16_K_M: case Format::Q_16_K_H: case Format::Q_G_16: case Format::Q_G_16_K_L: case Format::Q_G_16_K_M: case Format::Q_G_16_K_H: case Format::Q_16_5: case Format::Q_G_16_5: return DType::F16;
        case Format::Q_24: case Format::Q_24_K_L: case Format::Q_24_K_M: case Format::Q_24_K_H: case Format::Q_G_24: case Format::Q_G_24_K_L: case Format::Q_G_24_K_M: case Format::Q_G_24_K_H: case Format::Q_24_5: case Format::Q_G_24_5: return DType::U8;
        case Format::Q_32: return DType::F32;
        default: return DType::U8;
    }
}
struct Shape{int64_t dims[8];int rank;Shape():rank(0){dims[0]=dims[1]=dims[2]=dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0;}explicit Shape(int64_t d0):rank(1){dims[0]=d0;dims[1]=dims[2]=dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0;}Shape(int64_t d0,int64_t d1):rank(2){dims[0]=d0;dims[1]=d1;dims[2]=dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0;}Shape(int64_t d0,int64_t d1,int64_t d2):rank(3){dims[0]=d0;dims[1]=d1;dims[2]=d2;dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0;}Shape(std::initializer_list<int64_t> l):rank((int)l.size()){dims[0]=dims[1]=dims[2]=dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0;if(rank>8)throw std::runtime_error("Shape: rank exceeds maximum of 8");int i=0;for(auto x:l)dims[i++]=x;}int64_t& operator[](int i){if(i<0||i>=rank)throw std::out_of_range("Shape index out of range");return dims[i];}const int64_t& operator[](int i) const{if(i<0||i>=rank)throw std::out_of_range("Shape index out of range");return dims[i];}int64_t numel() const{int64_t n=1;for(int i=0;i<rank;i++)n*=dims[i];return n;}bool operator==(const Shape& o) const{if(rank!=o.rank)return false;for(int i=0;i<rank;i++)if(dims[i]!=o.dims[i])return false;return true;}bool operator!=(const Shape& o) const{return !(*this==o);}std::string to_string() const{std::string s="[";for(int i=0;i<rank;i++){if(i)s+=",";s+=std::to_string(dims[i]);}s+="]";return s;}};
struct Status{bool ok;std::string msg;Status():ok(true){}Status(const std::string& e):ok(false),msg(e){}static Status error(const std::string& m){return Status(m);}static Status success(){return Status();}explicit operator bool() const{return ok;}};
struct Config{int num_threads=1;uint64_t seed=42;size_t pool_size=64*1024*1024;bool verbose=false;};
class Error:public std::runtime_error{public:explicit Error(const std::string& m):std::runtime_error(m){}};
#ifdef QUANT_THROW_ABORT
#define QUANT_CHECK(cond,msg) do{if(!(cond)){fprintf(stderr,"QUANT_CHECK FAIL: %s\n",std::string(msg).c_str());fflush(stderr);std::abort();}}while(0)
#else
#define QUANT_CHECK(cond,msg) do{if(!(cond))throw quant::Error(msg);}while(0)
#endif
} // namespace quant
