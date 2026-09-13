#include "quant/gpu_compute_musa.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

#if defined(QUANT_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace quant {

namespace {
    void* load_lib(const char* name) {
#ifdef _WIN32
        return LoadLibraryA(name);
#else
        return dlopen(name, RTLD_LAZY);
#endif
    }
    void* get_sym(void* handle, const char* name) {
#ifdef _WIN32
        return (void*)GetProcAddress((HMODULE)handle, name);
#else
        return dlsym(handle, name);
#endif
    }
}

GpuComputeMusa::GpuComputeMusa() : lib_handle_(nullptr) {}

GpuComputeMusa::~GpuComputeMusa() {}

bool GpuComputeMusa::init() {
#ifdef _WIN32
    lib_handle_ = load_lib("musa_runtime.dll");
#else
    lib_handle_ = load_lib("libmusa.so");
#endif

    if (!lib_handle_) return false;

    musa_malloc_ = (decltype(musa_malloc_))get_sym(lib_handle_, "musaMalloc");
    musa_free_ = (decltype(musa_free_))get_sym(lib_handle_, "musaFree");
    musa_memcpy_ = (decltype(musa_memcpy_))get_sym(lib_handle_, "musaMemcpy");
    musa_launch_kernel_ = (decltype(musa_launch_kernel_))get_sym(lib_handle_, "musaLaunchKernel");

    return (musa_malloc_ != nullptr);
}

void* GpuComputeMusa::alloc(size_t size) {
    if (musa_malloc_) {
        void* ptr = nullptr;
        if (musa_malloc_(&ptr, size) == 0 && ptr != nullptr) return ptr;
    }
    return new uint8_t[size];
}

void GpuComputeMusa::free(void* ptr) {
    if (musa_free_) {
        musa_free_(ptr);
    } else {
        delete[] static_cast<uint8_t*>(ptr);
    }
}

void GpuComputeMusa::copy_to_device(void* dst, const void* src, size_t size) {
    if (musa_memcpy_) {
        musa_memcpy_(dst, src, size, 1); // host to device = 1
    } else {
        std::memcpy(dst, src, size);
    }
}

void GpuComputeMusa::copy_to_host(void* dst, const void* src, size_t size) {
    if (musa_memcpy_) {
        musa_memcpy_(dst, src, size, 2); // device to host = 2
    } else {
        std::memcpy(dst, src, size);
    }
}

void GpuComputeMusa::launch_gemm(int m, int n, int k, const float* a, const float* b, float* c) {
    // Host path (alpha=1, beta=0). NOTE: the previous loop accumulated with
    // += into uninitialized C (wrong unless caller zeroed it); the reference
    // below owns beta handling. Real musaLaunchKernel dispatch is not
    // implemented — GPU_MUSA compute fails loud via ComputeBackend.
    reference_gemm(1.0f, a, b, 0.0f, c, m, n, k);
}

// ========================================================================
// Host reference kernels. Tuning: Moore Threads MTP clusters favour square
// 64x64 tiles (one thread-block worth of C) with K-slices of 64; i-k-j
// order. AVX2 (8-wide) on host for elementwise work; transcendentals
// scalar. Beta/alpha-correct, null-guarded.
// ========================================================================

void GpuComputeMusa::reference_gemm(float alpha, const float* a, const float* b,
                                     float beta, float* c, int m, int n, int k) {
    if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0) return;
    const int total = m * n;
    if (beta != 0.0f) {
        for (int i = 0; i < total; ++i) c[i] *= beta;
    } else {
        for (int i = 0; i < total; ++i) c[i] = 0.0f;
    }
    constexpr int BM = 64, BN = 64, BK = 64;
    for (int m0 = 0; m0 < m; m0 += BM) {
        const int m1 = (m0 + BM < m) ? m0 + BM : m;
        for (int n0 = 0; n0 < n; n0 += BN) {
            const int n1 = (n0 + BN < n) ? n0 + BN : n;
            for (int k0 = 0; k0 < k; k0 += BK) {
                const int k1 = (k0 + BK < k) ? k0 + BK : k;
                for (int i = m0; i < m1; ++i) {
                    float* crow = c + i * n;
                    const float* arow = a + i * k;
                    for (int l = k0; l < k1; ++l) {
                        const float aval = alpha * arow[l];
                        if (aval == 0.0f) continue;
                        const float* brow = b + l * n;
                        int j = n0;
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

void GpuComputeMusa::reference_gemv(float alpha, const float* A, const float* x,
                                     float beta, float* y, int m, int n) {
    if (!A || !x || !y || m <= 0 || n <= 0) return;
    for (int i = 0; i < m; ++i) {
        const float* row = A + i * n;
        float s = 0.0f;
        int j = 0;
#if defined(QUANT_AVX2) || defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (; j + 8 <= n; j += 8) {
            const __m256 av = _mm256_loadu_ps(row + j);
            const __m256 xv = _mm256_loadu_ps(x + j);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(av, xv));
        }
        float tail[8];
        _mm256_storeu_ps(tail, acc);
        s = tail[0] + tail[1] + tail[2] + tail[3] + tail[4] + tail[5] + tail[6] + tail[7];
#endif
        for (; j < n; ++j) s += row[j] * x[j];
        y[i] = alpha * s + beta * y[i];
    }
}

void GpuComputeMusa::reference_relu(const float* x, float* y, size_t n) {
    if (!x || !y) return;
    size_t i = 0;
#if defined(QUANT_AVX2) || defined(__AVX2__)
    const __m256 z = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(y + i, _mm256_max_ps(v, z));
    }
#endif
    for (; i < n; ++i) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void GpuComputeMusa::reference_gelu(const float* x, float* y, size_t n) {
    if (!x || !y) return;
    constexpr float c0 = 0.7978845608028654f;
    constexpr float c1 = 0.044715f;
    for (size_t i = 0; i < n; ++i) {
        const float v = x[i];
        y[i] = 0.5f * v * (1.0f + std::tanh(c0 * (v + c1 * v * v * v)));
    }
}

void GpuComputeMusa::reference_silu(const float* x, float* y, size_t n) {
    if (!x || !y) return;
    for (size_t i = 0; i < n; ++i) {
        const float v = x[i];
        y[i] = v / (1.0f + std::exp(-v));
    }
}

void GpuComputeMusa::reference_add(const float* a, const float* b, float* c, size_t n) {
    if (!a || !b || !c) return;
    size_t i = 0;
#if defined(QUANT_AVX2) || defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        const __m256 av = _mm256_loadu_ps(a + i);
        const __m256 bv = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(av, bv));
    }
#endif
    for (; i < n; ++i) c[i] = a[i] + b[i];
}

void GpuComputeMusa::reference_mul(const float* a, const float* b, float* c, size_t n) {
    if (!a || !b || !c) return;
    size_t i = 0;
#if defined(QUANT_AVX2) || defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        const __m256 av = _mm256_loadu_ps(a + i);
        const __m256 bv = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_mul_ps(av, bv));
    }
#endif
    for (; i < n; ++i) c[i] = a[i] * b[i];
}

void GpuComputeMusa::reference_scale(float s, const float* x, float* y, size_t n) {
    if (!x || !y) return;
    size_t i = 0;
#if defined(QUANT_AVX2) || defined(__AVX2__)
    const __m256 sv = _mm256_set1_ps(s);
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(sv, v));
    }
#endif
    for (; i < n; ++i) y[i] = s * x[i];
}

void GpuComputeMusa::reference_softmax(const float* x, float* y, int rows, int cols) {
    if (!x || !y || rows <= 0 || cols <= 0) return;
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr = y + r * cols;
        float mx = xr[0];
        for (int c = 1; c < cols; ++c) mx = (std::max)(mx, xr[c]);
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            yr[c] = std::exp(xr[c] - mx);
            sum += yr[c];
        }
        const float inv = 1.0f / (sum + 1e-10f);
        for (int c = 0; c < cols; ++c) yr[c] *= inv;
    }
}

void GpuComputeMusa::reference_rms_norm(const float* x, const float* gamma, float* y,
                                         float eps, int rows, int cols) {
    if (!x || !gamma || !y || rows <= 0 || cols <= 0) return;
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr = y + r * cols;
        float ss = 0.0f;
        for (int c = 0; c < cols; ++c) ss += xr[c] * xr[c];
        const float rs = 1.0f / std::sqrt(ss / cols + eps);
        for (int c = 0; c < cols; ++c) yr[c] = xr[c] * rs * gamma[c];
    }
}

void GpuComputeMusa::reference_layer_norm(const float* x, const float* gamma, const float* beta,
                                           float* y, float eps, int rows, int cols) {
    if (!x || !gamma || !beta || !y || rows <= 0 || cols <= 0) return;
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr = y + r * cols;
        float mn = 0.0f;
        for (int c = 0; c < cols; ++c) mn += xr[c];
        mn /= cols;
        float vr = 0.0f;
        for (int c = 0; c < cols; ++c) {
            const float d = xr[c] - mn;
            vr += d * d;
        }
        vr /= cols;
        const float iv = 1.0f / std::sqrt(vr + eps);
        for (int c = 0; c < cols; ++c) yr[c] = (xr[c] - mn) * iv * gamma[c] + beta[c];
    }
}

} // namespace quant
