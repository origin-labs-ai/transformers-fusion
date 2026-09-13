#pragma once
// detail/exp_avx2.h — single home for the AVX2 exp approximation.
//
// DEDUP (2026-09-11): was triplicated byte-identical in math_avx2.cpp,
// math_avx2_tensor.cpp, math_avx2_tiled.cpp ("kept self-contained" drift
// risk — three copies of polynomial constants). Include under
// #if defined(QUANT_AVX2) after <immintrin.h>.
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__) || \
    defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

#if defined(QUANT_AVX2) || defined(__AVX2__)
namespace quant {
namespace math {
namespace detail {

// Fast exp approximation (polynomial, float-precision):
// exp(x) = 2^(x * log2(e)), polynomial on the fractional part.
inline __m256 exp_ps(__m256 x) {
    __m256 ln2 = _mm256_set1_ps(1.4426950408889634f);
    __m256 t = _mm256_mul_ps(x, ln2);

    __m256 t_floor = _mm256_floor_ps(t);
    __m256 frac = _mm256_sub_ps(t, t_floor);

    __m256 c1 = _mm256_set1_ps(0.6931471805599453f);
    __m256 c2 = _mm256_set1_ps(0.240226506959101f);
    __m256 c3 = _mm256_set1_ps(0.055504108664672f);
    __m256 c4 = _mm256_set1_ps(0.009618129107628f);
    __m256 c5 = _mm256_set1_ps(0.001333355814643f);

    __m256 p = _mm256_fmadd_ps(c5, frac, c4);
    p = _mm256_fmadd_ps(p, frac, c3);
    p = _mm256_fmadd_ps(p, frac, c2);
    p = _mm256_fmadd_ps(p, frac, c1);
    p = _mm256_fmadd_ps(p, frac, _mm256_set1_ps(1.0f));

    __m256i exp_part = _mm256_slli_epi32(
        _mm256_add_epi32(_mm256_cvttps_epi32(t_floor), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(exp_part));
}

} // namespace detail
} // namespace math
} // namespace quant
#endif // defined(QUANT_AVX2) || defined(__AVX2__)
