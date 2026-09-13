#include "quant/gpu_compute_openvino.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#if defined(QUANT_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace quant {
namespace gpu {

static void* load_openvino_lib() {
#if defined(_WIN32)
    return LoadLibraryA("openvino_c.dll");
#else
    return dlopen("libopenvino_c.so", RTLD_LAZY | RTLD_LOCAL);
#endif
}

static void close_openvino_lib(void* handle) {
#if defined(_WIN32)
    if (handle) FreeLibrary((HMODULE)handle);
#else
    if (handle) dlclose(handle);
#endif
}

static void* get_openvino_sym(void* handle, const char* name) {
#if defined(_WIN32)
    return (void*)GetProcAddress((HMODULE)handle, name);
#else
    return dlsym(handle, name);
#endif
}

GPUComputeOpenVINO::~GPUComputeOpenVINO() {
    if (handle_) {
        close_openvino_lib(handle_);
    }
}

bool GPUComputeOpenVINO::init(int device_id) {
    if (initialized_) return true;
    handle_ = load_openvino_lib();
    if (!handle_) return false;
    // Real implementation would resolve functions here
    initialized_ = true;
    return true;
}

// REAL ONLY: malloc-backed "device" memory is HOST emulation, documented
// here so no one mistakes it for NPU memory. Compute ops always fail loud
// via the backend wrapper (CPU triple-loop stub, no graph kernels).
void* GPUComputeOpenVINO::alloc(size_t size) {
    if (!initialized_) return nullptr;
    return std::malloc(size);
}

void GPUComputeOpenVINO::free_buf(void* ptr) {
    std::free(ptr);
}

void GPUComputeOpenVINO::upload(const void* src, void* dst, size_t size) {
    std::memcpy(dst, src, size);
}

void GPUComputeOpenVINO::download(const void* src, void* dst, size_t size) {
    std::memcpy(dst, src, size);
}

void GPUComputeOpenVINO::gemm(float alpha, const void* A, const void* B, float beta, void* C, int64_t M, int64_t N, int64_t K) {
    // Host reference (NOT NPU execution). Delegates to the tuned reference.
    reference_gemm(alpha, static_cast<const float*>(A), static_cast<const float*>(B),
                   beta, static_cast<float*>(C), M, N, K);
}

// ========================================================================
// Host reference kernels. Tuning: OpenVINO NPU tiles weights in 48-row
// groups; host mirror uses 48-row tiles with K-slices of 96. i-k-j order.
// AVX2 (8-wide) on host for elementwise work; transcendentals scalar.
// Beta/alpha-correct, null-guarded.
// ========================================================================

void GPUComputeOpenVINO::reference_gemm(float alpha, const float* a, const float* b,
                                         float beta, float* c, int64_t M, int64_t N, int64_t K) {
    if (!a || !b || !c || M <= 0 || N <= 0 || K <= 0) return;
    const int64_t total = M * N;
    if (beta != 0.0f) {
        for (int64_t i = 0; i < total; ++i) c[i] *= beta;
    } else {
        for (int64_t i = 0; i < total; ++i) c[i] = 0.0f;
    }
    constexpr int64_t BM = 48, BN = 64, BK = 96;
    for (int64_t m0 = 0; m0 < M; m0 += BM) {
        const int64_t m1 = (m0 + BM < M) ? m0 + BM : M;
        for (int64_t n0 = 0; n0 < N; n0 += BN) {
            const int64_t n1 = (n0 + BN < N) ? n0 + BN : N;
            for (int64_t k0 = 0; k0 < K; k0 += BK) {
                const int64_t k1 = (k0 + BK < K) ? k0 + BK : K;
                for (int64_t i = m0; i < m1; ++i) {
                    float* crow = c + i * N;
                    const float* arow = a + i * K;
                    for (int64_t l = k0; l < k1; ++l) {
                        const float aval = alpha * arow[l];
                        if (aval == 0.0f) continue;
                        const float* brow = b + l * N;
                        int64_t j = n0;
#if defined(QUANT_AVX2) || defined(__AVX2__)
                        const __m256 av = _mm256_set1_ps(aval);
                        for (; j + 8 <= n1; j += 8) {
                            const __m256 bv = _mm256_loadu_ps(brow + j);
                            const __m256 cv = _mm256_loadu_ps(crow + j);
                            _mm256_storeu_ps(crow + j, _mm256_add_ps(_mm256_mul_ps(av, bv), cv));
                        }
#endif
                        for (; j < n1; ++j) crow[j] += aval * brow[j];
                    }
                }
            }
        }
    }
}

void GPUComputeOpenVINO::reference_gemv(float alpha, const float* A, const float* x,
                                         float beta, float* y, int64_t M, int64_t N) {
    if (!A || !x || !y || M <= 0 || N <= 0) return;
    for (int64_t i = 0; i < M; ++i) {
        const float* row = A + i * N;
        float s = 0.0f;
        int64_t j = 0;
#if defined(QUANT_AVX2) || defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (; j + 8 <= N; j += 8) {
            const __m256 av = _mm256_loadu_ps(row + j);
            const __m256 xv = _mm256_loadu_ps(x + j);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(av, xv));
        }
        float tail[8];
        _mm256_storeu_ps(tail, acc);
        s = tail[0] + tail[1] + tail[2] + tail[3] + tail[4] + tail[5] + tail[6] + tail[7];
#endif
        for (; j < N; ++j) s += row[j] * x[j];
        y[i] = alpha * s + beta * y[i];
    }
}

void GPUComputeOpenVINO::reference_relu(const float* x, float* y, int64_t n) {
    if (!x || !y) return;
    int64_t i = 0;
#if defined(QUANT_AVX2) || defined(__AVX2__)
    const __m256 z = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(y + i, _mm256_max_ps(v, z));
    }
#endif
    for (; i < n; ++i) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void GPUComputeOpenVINO::reference_gelu(const float* x, float* y, int64_t n) {
    if (!x || !y) return;
    constexpr float c0 = 0.7978845608028654f;
    constexpr float c1 = 0.044715f;
    for (int64_t i = 0; i < n; ++i) {
        const float v = x[i];
        y[i] = 0.5f * v * (1.0f + std::tanh(c0 * (v + c1 * v * v * v)));
    }
}

void GPUComputeOpenVINO::reference_silu(const float* x, float* y, int64_t n) {
    if (!x || !y) return;
    for (int64_t i = 0; i < n; ++i) {
        const float v = x[i];
        y[i] = v / (1.0f + std::exp(-v));
    }
}

void GPUComputeOpenVINO::reference_add(const float* a, const float* b, float* c, int64_t n) {
    if (!a || !b || !c) return;
    int64_t i = 0;
#if defined(QUANT_AVX2) || defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        const __m256 av = _mm256_loadu_ps(a + i);
        const __m256 bv = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(av, bv));
    }
#endif
    for (; i < n; ++i) c[i] = a[i] + b[i];
}

void GPUComputeOpenVINO::reference_mul(const float* a, const float* b, float* c, int64_t n) {
    if (!a || !b || !c) return;
    int64_t i = 0;
#if defined(QUANT_AVX2) || defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        const __m256 av = _mm256_loadu_ps(a + i);
        const __m256 bv = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_mul_ps(av, bv));
    }
#endif
    for (; i < n; ++i) c[i] = a[i] * b[i];
}

void GPUComputeOpenVINO::reference_scale(float s, const float* x, float* y, int64_t n) {
    if (!x || !y) return;
    int64_t i = 0;
#if defined(QUANT_AVX2) || defined(__AVX2__)
    const __m256 sv = _mm256_set1_ps(s);
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(sv, v));
    }
#endif
    for (; i < n; ++i) y[i] = s * x[i];
}

void GPUComputeOpenVINO::reference_softmax(const float* x, float* y, int64_t rows, int64_t cols) {
    if (!x || !y || rows <= 0 || cols <= 0) return;
    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr = y + r * cols;
        float mx = xr[0];
        for (int64_t c = 1; c < cols; ++c) mx = (std::max)(mx, xr[c]);
        float sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            yr[c] = std::exp(xr[c] - mx);
            sum += yr[c];
        }
        const float inv = 1.0f / (sum + 1e-10f);
        for (int64_t c = 0; c < cols; ++c) yr[c] *= inv;
    }
}

void GPUComputeOpenVINO::reference_rms_norm(const float* x, const float* weight, float* y,
                                             float eps, int64_t rows, int64_t cols) {
    if (!x || !weight || !y || rows <= 0 || cols <= 0) return;
    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr = y + r * cols;
        float ss = 0.0f;
        for (int64_t c = 0; c < cols; ++c) ss += xr[c] * xr[c];
        const float rs = 1.0f / std::sqrt(ss / cols + eps);
        for (int64_t c = 0; c < cols; ++c) yr[c] = xr[c] * rs * weight[c];
    }
}

void GPUComputeOpenVINO::reference_layer_norm(const float* x, const float* gamma, const float* beta,
                                               float* y, float eps, int64_t rows, int64_t cols) {
    if (!x || !gamma || !beta || !y || rows <= 0 || cols <= 0) return;
    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr = y + r * cols;
        float mn = 0.0f;
        for (int64_t c = 0; c < cols; ++c) mn += xr[c];
        mn /= cols;
        float vr = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            const float d = xr[c] - mn;
            vr += d * d;
        }
        vr /= cols;
        const float iv = 1.0f / std::sqrt(vr + eps);
        for (int64_t c = 0; c < cols; ++c) yr[c] = (xr[c] - mn) * iv * gamma[c] + beta[c];
    }
}

// REAL ONLY: dummy 1GB constants REMOVED (fake capacity). No NPU memory
// query exists in this stub; 0 until real ov_* queries land.
int64_t GPUComputeOpenVINO::memory_free() const {
    return 0;
}

int64_t GPUComputeOpenVINO::memory_total() const {
    return 0;
}

void GPUComputeOpenVINO::synchronize() {
    // No-op
}

GPUComputeOpenVINO& get_openvino_compute() {
    static GPUComputeOpenVINO instance;
    return instance;
}

} // namespace gpu
} // namespace quant
