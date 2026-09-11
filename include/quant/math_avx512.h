#pragma once
// math_avx512.h — AVX-512 kernel declarations (compiled only when QUANT_AVX512).
//
// Contracts (PROD round-4 docs; previously undocumented):
// - All pointers must be non-null; n >= 0 (n == 0 is a no-op).
// - dst may alias src for unary ops (relu/gelu/silu/sigmoid/tanh/softmax/
//   norms/scale) but must NOT overlap a/b on vec_* binary ops.
// - vec_div: zeros in b produce inf/nan (no guard — caller checks).
// - softmax/layer_norm/rms_norm operate on ONE row of length n; eps > 0.
// - gemv: A is {M,N} row-major, x {N}, y {M}; alpha/beta follow BLAS gemv.
// - Alignment: _mm512_loadu (unaligned-safe); 64B alignment recommended.
// - Thread-safety: pure functions, no globals — safe to call concurrently.

#include "quant/tensor.h"

namespace quant {
namespace math {

/// result = dot(a, b) over n elements.
void dot_avx512(float* result, const float* a, const float* b, int64_t n);
/// y += alpha * x (n elements).
void axpy_avx512(float alpha, const float* x, float* y, int64_t n);

/// Elementwise add/sub/mul/div/scale over n elements (see aliasing rule).
void vec_add_avx512(float* dst, const float* a, const float* b, int64_t n);
void vec_sub_avx512(float* dst, const float* a, const float* b, int64_t n);
void vec_mul_avx512(float* dst, const float* a, const float* b, int64_t n);
void vec_div_avx512(float* dst, const float* a, const float* b, int64_t n);
void vec_scale_avx512(float* dst, const float* src, float factor, int64_t n);

/// Activations over n elements (in-place safe).
void relu_avx512(float* dst, const float* src, int64_t n);
void gelu_avx512(float* dst, const float* src, int64_t n);
void silu_avx512(float* dst, const float* src, int64_t n);
void sigmoid_avx512(float* dst, const float* src, int64_t n);
void tanh_avx512(float* dst, const float* src, int64_t n);

/// Softmax / norms over one row of length n (eps > 0).
void softmax_avx512(float* dst, const float* src, int64_t n);
void layer_norm_avx512(float* dst, const float* src, const float* gamma,
                       const float* beta, int64_t n, float eps);
void rms_norm_avx512(float* dst, const float* src, const float* gamma,
                     int64_t n, float eps);

/// y = alpha*A*x + beta*y (BLAS semantics; shapes above).
void gemv_avx512(float alpha, const Tensor& A, const Tensor& x,
                 float beta, Tensor& y);

} // namespace math
} // namespace quant
