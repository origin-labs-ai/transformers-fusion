#pragma once
// detail/fp16.h — single home for IEEE-754 binary16 <-> float conversion.
//
// DEDUP (2026-09-11): was copy-pasted in kernel_quant4.cpp,
// kernel_q12.cpp (as q12_fp16_to_float), kernel_production.cpp,
// trainer_data.cpp, core/tensor.cpp (as half_to_float) — all with the same
// magic 2^-24 literal. One named constant + one inline function.
#include <cstdint>
#include <cmath>
#include <cstring>

namespace quant {
namespace detail {

// 2^-24: subnormal FP16 mantissa step (mant * 2^-24).
inline constexpr float kFp16SubnormalStep = 0.000000059604644775390625f;

inline float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    if (exp == 0) {
        float v = (float)mant * kFp16SubnormalStep;
        return sign ? -v : v;
    }
    if (exp == 31) {
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    }
    uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &f32, sizeof(result));
    return result;
}

} // namespace detail
} // namespace quant
