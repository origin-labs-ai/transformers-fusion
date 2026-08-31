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
// Format enum — Q-series quantization formats — v3 RE-ARRANGED 2026-08-23
// Rules: exact BPW (name == claimed BPW), 3 K variants per BPW (L/M/H),
// GRP exact at same BPW (no extra), half-BPW GRP as Q_G_X_Y,
// MXQ only as MXQ_(BPW)_G (4-variant mix, no TWI), no TWI_MIX.
// ============================================================
enum class Format : uint8_t {
    // --- Base integer BPW (10) ---
    Q1              = 0,   // 1.00
    Q2              = 1,   // 2.00
    Q3              = 2,   // 3.00
    Q4              = 3,   // 4.00
    Q6              = 4,   // 6.00
    Q8              = 5,   // 8.00
    Q12             = 6,   // 12.00
    Q16             = 7,   // 16.00
    Q24             = 8,   // 24.00
    Q32             = 9,   // 32.00
    // --- K variants integer BPW (30) ---
    Q1_K_L          = 10, Q1_K_M          = 11, Q1_K_H          = 12,
    Q2_K_L          = 13, Q2_K_M          = 14, Q2_K_H          = 15,
    Q3_K_L          = 16, Q3_K_M          = 17, Q3_K_H          = 18,
    Q4_K_L          = 19, Q4_K_M          = 20, Q4_K_H          = 21,
    Q6_K_L          = 22, Q6_K_M          = 23, Q6_K_H          = 24,
    Q8_K_L          = 25, Q8_K_M          = 26, Q8_K_H          = 27,
    Q12_K_L         = 28, Q12_K_M         = 29, Q12_K_H         = 30,
    Q16_K_L         = 31, Q16_K_M         = 32, Q16_K_H         = 33,
    Q24_K_L         = 34, Q24_K_M         = 35, Q24_K_H         = 36,
    // --- GRP exact integer BPW (9) ---
    Q1_G          = 37, Q2_G          = 38, Q3_G          = 39, Q4_G          = 40, Q6_G          = 41,
    Q8_G          = 42, Q12_G         = 43, Q16_G         = 44, Q24_G         = 45,
    // --- K_G exact integer BPW (27) ---
    Q1_K_L_G      = 46, Q1_K_M_G      = 47, Q1_K_H_G      = 48,
    Q2_K_L_G      = 49, Q2_K_M_G      = 50, Q2_K_H_G      = 51,
    Q3_K_L_G      = 52, Q3_K_M_G      = 53, Q3_K_H_G      = 54,
    Q4_K_L_G      = 55, Q4_K_M_G      = 56, Q4_K_H_G      = 57,
    Q6_K_L_G      = 58, Q6_K_M_G      = 59, Q6_K_H_G      = 60,
    Q8_K_L_G      = 61, Q8_K_M_G      = 62, Q8_K_H_G      = 63,
    Q12_K_L_G     = 64, Q12_K_M_G     = 65, Q12_K_H_G     = 66,
    Q16_K_L_G     = 67, Q16_K_M_G     = 68, Q16_K_H_G     = 69,
    Q24_K_L_G     = 70, Q24_K_M_G     = 71, Q24_K_H_G     = 72,
    // --- Half BPW plain (9) ---
    Q1_5            = 73, Q2_5            = 74, Q3_5            = 75, Q4_5            = 76, Q6_5            = 77,
    Q8_5            = 78, Q12_5           = 79, Q16_5           = 80, Q24_5           = 81,
    // --- Half BPW GRP (9) ---
    Q_G_1_5       = 82, Q_G_2_5       = 83, Q_G_3_5       = 84, Q_G_4_5       = 85, Q_G_6_5       = 86,
    Q_G_8_5       = 87, Q_G_12_5      = 88, Q_G_16_5      = 89, Q_G_24_5      = 90,
    // --- MXQ mix 4-variant (7 plain + 7 GRP) ---
    MXQ_3_5         = 91, MXQ_4_5         = 92, MXQ_6_5         = 93, MXQ_8_5         = 94, MXQ_12_5        = 95, MXQ_16_5        = 96, MXQ_24_5        = 97,
    MXQ_3_5_G     = 98, MXQ_4_5_G     = 99, MXQ_6_5_G     = 100, MXQ_8_5_G     = 101, MXQ_12_5_G    = 102, MXQ_16_5_G    = 103, MXQ_24_5_G    = 104,
};
constexpr int FORMAT_COUNT = 105;
inline const char* format_name(Format f) {
    switch(f) {
        case Format::Q1: return "Q1"; case Format::Q2: return "Q2"; case Format::Q3: return "Q3"; case Format::Q4: return "Q4"; case Format::Q6: return "Q6"; case Format::Q8: return "Q8"; case Format::Q12: return "Q12"; case Format::Q16: return "Q16"; case Format::Q24: return "Q24"; case Format::Q32: return "Q32";
        case Format::Q1_K_L: return "Q1_K_L"; case Format::Q1_K_M: return "Q1_K_M"; case Format::Q1_K_H: return "Q1_K_H";
        case Format::Q2_K_L: return "Q2_K_L"; case Format::Q2_K_M: return "Q2_K_M"; case Format::Q2_K_H: return "Q2_K_H";
        case Format::Q3_K_L: return "Q3_K_L"; case Format::Q3_K_M: return "Q3_K_M"; case Format::Q3_K_H: return "Q3_K_H";
        case Format::Q4_K_L: return "Q4_K_L"; case Format::Q4_K_M: return "Q4_K_M"; case Format::Q4_K_H: return "Q4_K_H";
        case Format::Q6_K_L: return "Q6_K_L"; case Format::Q6_K_M: return "Q6_K_M"; case Format::Q6_K_H: return "Q6_K_H";
        case Format::Q8_K_L: return "Q8_K_L"; case Format::Q8_K_M: return "Q8_K_M"; case Format::Q8_K_H: return "Q8_K_H";
        case Format::Q12_K_L: return "Q12_K_L"; case Format::Q12_K_M: return "Q12_K_M"; case Format::Q12_K_H: return "Q12_K_H";
        case Format::Q16_K_L: return "Q16_K_L"; case Format::Q16_K_M: return "Q16_K_M"; case Format::Q16_K_H: return "Q16_K_H";
        case Format::Q24_K_L: return "Q24_K_L"; case Format::Q24_K_M: return "Q24_K_M"; case Format::Q24_K_H: return "Q24_K_H";
        case Format::Q1_G: return "Q1_G"; case Format::Q2_G: return "Q2_G"; case Format::Q3_G: return "Q3_G"; case Format::Q4_G: return "Q4_G"; case Format::Q6_G: return "Q6_G"; case Format::Q8_G: return "Q8_G"; case Format::Q12_G: return "Q12_G"; case Format::Q16_G: return "Q16_G"; case Format::Q24_G: return "Q24_G";
        case Format::Q1_K_L_G: return "Q1_K_L_G"; case Format::Q1_K_M_G: return "Q1_K_M_G"; case Format::Q1_K_H_G: return "Q1_K_H_G";
        case Format::Q2_K_L_G: return "Q2_K_L_G"; case Format::Q2_K_M_G: return "Q2_K_M_G"; case Format::Q2_K_H_G: return "Q2_K_H_G";
        case Format::Q3_K_L_G: return "Q3_K_L_G"; case Format::Q3_K_M_G: return "Q3_K_M_G"; case Format::Q3_K_H_G: return "Q3_K_H_G";
        case Format::Q4_K_L_G: return "Q4_K_L_G"; case Format::Q4_K_M_G: return "Q4_K_M_G"; case Format::Q4_K_H_G: return "Q4_K_H_G";
        case Format::Q6_K_L_G: return "Q6_K_L_G"; case Format::Q6_K_M_G: return "Q6_K_M_G"; case Format::Q6_K_H_G: return "Q6_K_H_G";
        case Format::Q8_K_L_G: return "Q8_K_L_G"; case Format::Q8_K_M_G: return "Q8_K_M_G"; case Format::Q8_K_H_G: return "Q8_K_H_G";
        case Format::Q12_K_L_G: return "Q12_K_L_G"; case Format::Q12_K_M_G: return "Q12_K_M_G"; case Format::Q12_K_H_G: return "Q12_K_H_G";
        case Format::Q16_K_L_G: return "Q16_K_L_G"; case Format::Q16_K_M_G: return "Q16_K_M_G"; case Format::Q16_K_H_G: return "Q16_K_H_G";
        case Format::Q24_K_L_G: return "Q24_K_L_G"; case Format::Q24_K_M_G: return "Q24_K_M_G"; case Format::Q24_K_H_G: return "Q24_K_H_G";
        case Format::Q1_5: return "Q1.5"; case Format::Q2_5: return "Q2.5"; case Format::Q3_5: return "Q3.5"; case Format::Q4_5: return "Q4.5"; case Format::Q6_5: return "Q6.5"; case Format::Q8_5: return "Q8.5"; case Format::Q12_5: return "Q12.5"; case Format::Q16_5: return "Q16.5"; case Format::Q24_5: return "Q24.5";
        case Format::Q_G_1_5: return "Q_G_1.5"; case Format::Q_G_2_5: return "Q_G_2.5"; case Format::Q_G_3_5: return "Q_G_3.5"; case Format::Q_G_4_5: return "Q_G_4.5"; case Format::Q_G_6_5: return "Q_G_6.5"; case Format::Q_G_8_5: return "Q_G_8.5"; case Format::Q_G_12_5: return "Q_G_12.5"; case Format::Q_G_16_5: return "Q_G_16.5"; case Format::Q_G_24_5: return "Q_G_24.5";
        case Format::MXQ_3_5: return "MXQ_3.5"; case Format::MXQ_4_5: return "MXQ_4.5"; case Format::MXQ_6_5: return "MXQ_6.5"; case Format::MXQ_8_5: return "MXQ_8.5"; case Format::MXQ_12_5: return "MXQ_12.5"; case Format::MXQ_16_5: return "MXQ_16.5"; case Format::MXQ_24_5: return "MXQ_24.5";
        case Format::MXQ_3_5_G: return "MXQ_3.5_G"; case Format::MXQ_4_5_G: return "MXQ_4.5_G"; case Format::MXQ_6_5_G: return "MXQ_6.5_G"; case Format::MXQ_8_5_G: return "MXQ_8.5_G"; case Format::MXQ_12_5_G: return "MXQ_12.5_G"; case Format::MXQ_16_5_G: return "MXQ_16.5_G"; case Format::MXQ_24_5_G: return "MXQ_24.5_G";
        default: return "unknown";
    }
}
inline float format_bpw(Format f) {
    switch(f) {
        case Format::Q1: return 1.0f; case Format::Q2: return 2.0f; case Format::Q3: return 3.0f; case Format::Q4: return 4.0f; case Format::Q6: return 6.0f; case Format::Q8: return 8.0f; case Format::Q12: return 12.0f; case Format::Q16: return 16.0f; case Format::Q24: return 24.0f; case Format::Q32: return 32.0f;
        case Format::Q1_K_L: case Format::Q1_K_M: case Format::Q1_K_H: return 1.0f;
        case Format::Q2_K_L: case Format::Q2_K_M: case Format::Q2_K_H: return 2.0f;
        case Format::Q3_K_L: case Format::Q3_K_M: case Format::Q3_K_H: return 3.0f;
        case Format::Q4_K_L: case Format::Q4_K_M: case Format::Q4_K_H: return 4.0f;
        case Format::Q6_K_L: case Format::Q6_K_M: case Format::Q6_K_H: return 6.0f;
        case Format::Q8_K_L: case Format::Q8_K_M: case Format::Q8_K_H: return 8.0f;
        case Format::Q12_K_L: case Format::Q12_K_M: case Format::Q12_K_H: return 12.0f;
        case Format::Q16_K_L: case Format::Q16_K_M: case Format::Q16_K_H: return 16.0f;
        case Format::Q24_K_L: case Format::Q24_K_M: case Format::Q24_K_H: return 24.0f;
        // TRUTHFUL BPW (audit 2026-08-26 option-c): these report the wire
        // budget the canonical encoder actually spends at n%256 (probe:
        // tests/test_format_audit.cpp). Names stay Q*_G for API stability.
        case Format::Q1_G: return 1.0f; case Format::Q2_G: return 2.625f; case Format::Q3_G: return 3.5f; case Format::Q4_G: return 4.5f; case Format::Q6_G: return 6.5625f; case Format::Q8_G: return 8.5f; case Format::Q12_G: return 12.5f; case Format::Q16_G: return 16.5f; case Format::Q24_G: return 24.5f;
        case Format::Q1_K_L_G: case Format::Q1_K_M_G: case Format::Q1_K_H_G: return 1.0f;
        case Format::Q2_K_L_G: case Format::Q2_K_M_G: case Format::Q2_K_H_G: return 2.4375f;
        case Format::Q3_K_L_G: case Format::Q3_K_M_G: case Format::Q3_K_H_G: return 3.4375f;
        case Format::Q4_K_L_G: case Format::Q4_K_M_G: case Format::Q4_K_H_G: return 4.4375f;
        case Format::Q6_K_L_G: case Format::Q6_K_M_G: case Format::Q6_K_H_G: return 6.5625f;
        case Format::Q8_K_L_G: case Format::Q8_K_M_G: case Format::Q8_K_H_G: return 8.5f;
        case Format::Q12_K_L_G: case Format::Q12_K_M_G: case Format::Q12_K_H_G: return 12.5f;
        case Format::Q16_K_L_G: case Format::Q16_K_M_G: case Format::Q16_K_H_G: return 16.0f;
        case Format::Q24_K_L_G: case Format::Q24_K_M_G: case Format::Q24_K_H_G: return 24.0f;
        case Format::Q1_5: return 1.5f; case Format::Q2_5: return 2.5f; case Format::Q3_5: return 3.5f; case Format::Q4_5: return 4.5f; case Format::Q6_5: return 6.5f; case Format::Q8_5: return 8.5f; case Format::Q12_5: return 12.5f; case Format::Q16_5: return 16.5f; case Format::Q24_5: return 24.5f;
        case Format::Q_G_1_5: return 1.5f; case Format::Q_G_2_5: return 2.5f; case Format::Q_G_3_5: return 3.5f; case Format::Q_G_4_5: return 4.5f; case Format::Q_G_6_5: return 6.5f; case Format::Q_G_8_5: return 8.5f; case Format::Q_G_12_5: return 12.5f; case Format::Q_G_16_5: return 16.5f; case Format::Q_G_24_5: return 24.5f;
        case Format::MXQ_3_5: return 3.5f; case Format::MXQ_4_5: return 4.5f; case Format::MXQ_6_5: return 6.5f; case Format::MXQ_8_5: return 8.5f; case Format::MXQ_12_5: return 12.5f; case Format::MXQ_16_5: return 16.5f; case Format::MXQ_24_5: return 24.5f;
        // MXQ_G true wire BPW at canonical n=256 (per-32 dom scales + per-tier
        // non-dom scales). MXQ_3.5_G has 92% sign tier → 8 dom scales → 3.78125.
        case Format::MXQ_3_5_G: return 3.78125f; case Format::MXQ_4_5_G: return 4.5f; case Format::MXQ_6_5_G: return 6.5f; case Format::MXQ_8_5_G: return 8.5f; case Format::MXQ_12_5_G: return 12.5f; case Format::MXQ_16_5_G: return 16.5f; case Format::MXQ_24_5_G: return 24.5f;
        default: return 0;
    }
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
        case Format::Q1: case Format::Q1_K_L: case Format::Q1_K_M: case Format::Q1_K_H: case Format::Q1_G: case Format::Q1_K_L_G: case Format::Q1_K_M_G: case Format::Q1_K_H_G: case Format::Q1_5: case Format::Q_G_1_5: return 1;
        case Format::Q2: case Format::Q2_K_L: case Format::Q2_K_M: case Format::Q2_K_H: case Format::Q2_G: case Format::Q2_K_L_G: case Format::Q2_K_M_G: case Format::Q2_K_H_G: case Format::Q2_5: case Format::Q_G_2_5: return 4;
        case Format::Q3: case Format::Q3_K_L: case Format::Q3_K_M: case Format::Q3_K_H: case Format::Q3_G: case Format::Q3_K_L_G: case Format::Q3_K_M_G: case Format::Q3_K_H_G: case Format::Q3_5: case Format::Q_G_3_5: return 8;
        case Format::Q4: case Format::Q4_K_L: case Format::Q4_K_M: case Format::Q4_K_H: case Format::Q4_G: case Format::Q4_K_L_G: case Format::Q4_K_M_G: case Format::Q4_K_H_G: case Format::Q4_5: case Format::Q_G_4_5: return 16;
        case Format::Q6: case Format::Q6_K_L: case Format::Q6_K_M: case Format::Q6_K_H: case Format::Q6_G: case Format::Q6_K_L_G: case Format::Q6_K_M_G: case Format::Q6_K_H_G: case Format::Q6_5: case Format::Q_G_6_5: return 64;
        case Format::Q8: case Format::Q8_K_L: case Format::Q8_K_M: case Format::Q8_K_H: case Format::Q8_G: case Format::Q8_K_L_G: case Format::Q8_K_M_G: case Format::Q8_K_H_G: case Format::Q8_5: case Format::Q_G_8_5: return 256;
        case Format::Q12: case Format::Q12_K_L: case Format::Q12_K_M: case Format::Q12_K_H: case Format::Q12_G: case Format::Q12_K_L_G: case Format::Q12_K_M_G: case Format::Q12_K_H_G: case Format::Q12_5: case Format::Q_G_12_5: return 4096;
        default: return 0;
    }
}
enum class DType : uint8_t { I64,I32,U8,U4,U16,F16,F32 };
inline size_t dtype_size(DType dt){ switch(dt){case DType::I64:return 8;case DType::I32:return 4;case DType::U8:return 1;case DType::U4:return 1;case DType::U16:return 2;case DType::F16:return 2;case DType::F32:return 4;default:return 0;}}
inline DType format_to_dtype(Format f){
    switch(f){
        case Format::Q1: case Format::Q1_K_L: case Format::Q1_K_M: case Format::Q1_K_H: case Format::Q1_G: case Format::Q1_K_L_G: case Format::Q1_K_M_G: case Format::Q1_K_H_G: case Format::Q1_5: case Format::Q_G_1_5: return DType::U8;
        case Format::Q2: case Format::Q2_K_L: case Format::Q2_K_M: case Format::Q2_K_H: case Format::Q2_G: case Format::Q2_K_L_G: case Format::Q2_K_M_G: case Format::Q2_K_H_G: case Format::Q2_5: case Format::Q_G_2_5: return DType::U8;
        case Format::Q3: case Format::Q3_K_L: case Format::Q3_K_M: case Format::Q3_K_H: case Format::Q3_G: case Format::Q3_K_L_G: case Format::Q3_K_M_G: case Format::Q3_K_H_G: case Format::Q3_5: case Format::Q_G_3_5: return DType::U8;
        case Format::Q4: case Format::Q4_K_L: case Format::Q4_K_M: case Format::Q4_K_H: case Format::Q4_G: case Format::Q4_K_L_G: case Format::Q4_K_M_G: case Format::Q4_K_H_G: case Format::Q4_5: case Format::Q_G_4_5: return DType::U4;
        case Format::Q6: case Format::Q6_K_L: case Format::Q6_K_M: case Format::Q6_K_H: case Format::Q6_G: case Format::Q6_K_L_G: case Format::Q6_K_M_G: case Format::Q6_K_H_G: case Format::Q6_5: case Format::Q_G_6_5: return DType::U8;
        case Format::Q8: case Format::Q8_K_L: case Format::Q8_K_M: case Format::Q8_K_H: case Format::Q8_G: case Format::Q8_K_L_G: case Format::Q8_K_M_G: case Format::Q8_K_H_G: case Format::Q8_5: case Format::Q_G_8_5: return DType::U8;
        case Format::Q12: case Format::Q12_K_L: case Format::Q12_K_M: case Format::Q12_K_H: case Format::Q12_G: case Format::Q12_K_L_G: case Format::Q12_K_M_G: case Format::Q12_K_H_G: case Format::Q12_5: case Format::Q_G_12_5: return DType::U16;
        case Format::Q16: case Format::Q16_K_L: case Format::Q16_K_M: case Format::Q16_K_H: case Format::Q16_G: case Format::Q16_K_L_G: case Format::Q16_K_M_G: case Format::Q16_K_H_G: case Format::Q16_5: case Format::Q_G_16_5: return DType::F16;
        case Format::Q24: case Format::Q24_K_L: case Format::Q24_K_M: case Format::Q24_K_H: case Format::Q24_G: case Format::Q24_K_L_G: case Format::Q24_K_M_G: case Format::Q24_K_H_G: case Format::Q24_5: case Format::Q_G_24_5: return DType::U8;
        case Format::Q32: return DType::F32;
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
