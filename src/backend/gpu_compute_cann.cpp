#include "quant/gpu_compute_cann.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

#if defined(QUANT_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace quant {
namespace gpu {

typedef int aclError;
typedef void* aclrtStream;

typedef aclError (*PFN_aclInit)(const char* configPath);
typedef aclError (*PFN_aclFinalize)();
typedef aclError (*PFN_aclrtSetDevice)(int32_t deviceId);
typedef aclError (*PFN_aclrtResetDevice)(int32_t deviceId);
typedef aclError (*PFN_aclrtCreateStream)(aclrtStream* stream);
typedef aclError (*PFN_aclrtDestroyStream)(aclrtStream stream);
typedef aclError (*PFN_aclrtMalloc)(void** devPtr, size_t size, int32_t policy);
typedef aclError (*PFN_aclrtFree)(void* devPtr);
typedef aclError (*PFN_aclrtMemcpy)(void* dst, size_t destMax, const void* src, size_t count, int32_t kind);
typedef aclError (*PFN_aclrtSynchronizeStream)(aclrtStream stream);
typedef aclError (*PFN_aclopExecuteV2)(const char* opType, int numInputs, void* inputDesc, void* inputs, int numOutputs, void* outputDesc, void* outputs, void* attr, aclrtStream stream);

struct CANNAPI {
    PFN_aclInit aclInit = nullptr;
    PFN_aclFinalize aclFinalize = nullptr;
    PFN_aclrtSetDevice aclrtSetDevice = nullptr;
    PFN_aclrtResetDevice aclrtResetDevice = nullptr;
    PFN_aclrtCreateStream aclrtCreateStream = nullptr;
    PFN_aclrtDestroyStream aclrtDestroyStream = nullptr;
    PFN_aclrtMalloc aclrtMalloc = nullptr;
    PFN_aclrtFree aclrtFree = nullptr;
    PFN_aclrtMemcpy aclrtMemcpy = nullptr;
    PFN_aclrtSynchronizeStream aclrtSynchronizeStream = nullptr;
    PFN_aclopExecuteV2 aclopExecuteV2 = nullptr;
    void* handle = nullptr;
};

struct GPUComputeCann::Impl {
    CANNAPI api;
    bool initialized = false;
    aclrtStream stream = nullptr;
    int32_t device_id = 0;

    bool load_cann() {
        if (api.handle) return true;
#if defined(_WIN32)
        api.handle = LoadLibraryA("libascendcl.dll");
        if (!api.handle) return false;
        #define LOAD_SYM(name) api.name = (PFN_##name)GetProcAddress((HMODULE)api.handle, #name)
#else
        api.handle = dlopen("libascendcl.so", RTLD_LAZY | RTLD_LOCAL);
        if (!api.handle) return false;
        #define LOAD_SYM(name) api.name = (PFN_##name)dlsym(api.handle, #name)
#endif
        LOAD_SYM(aclInit);
        LOAD_SYM(aclFinalize);
        LOAD_SYM(aclrtSetDevice);
        LOAD_SYM(aclrtResetDevice);
        LOAD_SYM(aclrtCreateStream);
        LOAD_SYM(aclrtDestroyStream);
        LOAD_SYM(aclrtMalloc);
        LOAD_SYM(aclrtFree);
        LOAD_SYM(aclrtMemcpy);
        LOAD_SYM(aclrtSynchronizeStream);
        LOAD_SYM(aclopExecuteV2);
        return true;
    }

    // REAL ONLY: the old body called aclopExecuteV2 with NULL tensor
    // descriptors (a silent no-op that left outputs untouched while the
    // caller believed a MatMul ran). That is FAKE. Until real aclTensorDesc
    // graphs are built, every compute entry fails loud.
    [[noreturn]] void no_cann_kernel(const char* op_type) {
        std::string msg = std::string("[REAL-ONLY] GPUComputeCann::") + op_type +
            " has no CANN tensor descriptors (null-descriptor stub); refusing to fake execution";
        std::fprintf(stderr, "%s\n", msg.c_str());
        throw std::runtime_error(msg);
    }
    void execute_op(const char* op_type) {
        (void)op_type;
        no_cann_kernel(op_type);
    }
};

GPUComputeCann::GPUComputeCann() : impl_(new Impl()) {}
GPUComputeCann::~GPUComputeCann() {
    shutdown();
    delete impl_;
}

bool GPUComputeCann::init(int64_t device_id) {
    if (impl_->initialized) return true;
    if (!impl_->load_cann()) return false;
    if (!impl_->api.aclInit) return false;

    if (impl_->api.aclInit(nullptr) != 0) return false;
    if (impl_->api.aclrtSetDevice((int32_t)device_id) != 0) return false;
    if (impl_->api.aclrtCreateStream(&impl_->stream) != 0) return false;
    
    impl_->device_id = (int32_t)device_id;
    impl_->initialized = true;
    return true;
}

bool GPUComputeCann::is_initialized() const { return impl_->initialized; }

void GPUComputeCann::shutdown() {
    if (impl_->initialized) {
        if (impl_->api.aclrtDestroyStream) impl_->api.aclrtDestroyStream(impl_->stream);
        if (impl_->api.aclrtResetDevice) impl_->api.aclrtResetDevice(impl_->device_id);
        if (impl_->api.aclFinalize) impl_->api.aclFinalize();
        impl_->initialized = false;
    }
}

void* GPUComputeCann::alloc(int64_t bytes) {
    if (!impl_->initialized || !impl_->api.aclrtMalloc) return nullptr;
    void* devPtr = nullptr;
    // 0 is ACL_MEM_MALLOC_HUGE_FIRST
    if (impl_->api.aclrtMalloc(&devPtr, bytes, 0) != 0) return nullptr;
    return devPtr;
}

void GPUComputeCann::free_buf(void* ptr) {
    if (!impl_->initialized || !impl_->api.aclrtFree || !ptr) return;
    impl_->api.aclrtFree(ptr);
}

void GPUComputeCann::upload(const Tensor& src, void* dst) {
    if (!impl_->initialized || !impl_->api.aclrtMemcpy || !src.data()) return;
    // 1 is ACL_MEMCPY_HOST_TO_DEVICE
    impl_->api.aclrtMemcpy(dst, src.numel() * sizeof(float), src.data(), src.numel() * sizeof(float), 1);
}

void GPUComputeCann::download(void* src, Tensor& dst) {
    if (!impl_->initialized || !impl_->api.aclrtMemcpy || !dst.data()) return;
    // 2 is ACL_MEMCPY_DEVICE_TO_HOST
    impl_->api.aclrtMemcpy(dst.data(), dst.numel() * sizeof(float), src, dst.numel() * sizeof(float), 2);
}

void GPUComputeCann::gemm(float alpha, const void* A, const void* B, float beta, void* C, int64_t M, int64_t N, int64_t K) {
    if (impl_->initialized) impl_->execute_op("MatMul");
}

void GPUComputeCann::gemv(float alpha, const void* A, const void* x, float beta, void* y, int64_t M, int64_t N) {
    if (impl_->initialized) impl_->execute_op("MatMul");
}

void GPUComputeCann::relu(const void* x, void* y, int64_t n) {
    if (impl_->initialized) impl_->execute_op("Relu");
}

void GPUComputeCann::gelu(const void* x, void* y, int64_t n) {
    if (impl_->initialized) impl_->execute_op("Gelu");
}

void GPUComputeCann::silu(const void* x, void* y, int64_t n) {
    if (impl_->initialized) impl_->execute_op("Swish");
}

void GPUComputeCann::softmax(const void* x, void* y, int64_t rows, int64_t cols) {
    if (impl_->initialized) impl_->execute_op("SoftmaxV2");
}

void GPUComputeCann::rms_norm(const void* x, const void* weight, void* y, int64_t rows, int64_t cols, float eps) {
    if (impl_->initialized) impl_->execute_op("LayerNorm");
}

void GPUComputeCann::layer_norm(const void* x, const void* gamma, const void* beta, void* y, int64_t rows, int64_t cols, float eps) {
    if (impl_->initialized) impl_->execute_op("LayerNorm");
}

void GPUComputeCann::add(const void* a, const void* b, void* c, int64_t n) {
    if (impl_->initialized) impl_->execute_op("Add");
}

void GPUComputeCann::mul(const void* a, const void* b, void* c, int64_t n) {
    if (impl_->initialized) impl_->execute_op("Mul");
}

void GPUComputeCann::scale(float s, const void* x, void* y, int64_t n) {
    if (impl_->initialized) impl_->execute_op("Scale");
}

void GPUComputeCann::fill(float val, void* x, int64_t n) {
    if (impl_->initialized) impl_->execute_op("Fills");
}

void GPUComputeCann::copy_buf(const void* src, void* dst, int64_t n) {
    if (impl_->initialized && impl_->api.aclrtMemcpy) {
        // 3 is ACL_MEMCPY_DEVICE_TO_DEVICE
        impl_->api.aclrtMemcpy(dst, n * sizeof(float), src, n * sizeof(float), 3);
    }
}

void GPUComputeCann::synchronize() {
    if (impl_->initialized && impl_->api.aclrtSynchronizeStream)
        impl_->api.aclrtSynchronizeStream(impl_->stream);
}

// REAL ONLY: unconditional 16GB constants REMOVED (fake capacity — no ACL
// memory query exists here). 0 until real aclrt queries land.
int64_t GPUComputeCann::memory_free() const {
    return 0;
}

int64_t GPUComputeCann::memory_total() const {
    return 0;
}

// ========================================================================
// Host reference kernels. Tuning: Ascend cube units natively multiply
// 16x16 tiles; host mirror uses 32-row tiles with K-slices of 128.
// i-k-j order. AVX2 (8-wide) on host for elementwise work;
// transcendentals scalar. Beta/alpha-correct, null-guarded. Verification
// baseline for CI without Ascend hardware — NEVER CANN execution.
// ========================================================================

void GPUComputeCann::reference_gemm(float alpha, const float* a, const float* b,
                                     float beta, float* c, int64_t M, int64_t N, int64_t K) {
    if (!a || !b || !c || M <= 0 || N <= 0 || K <= 0) return;
    const int64_t total = M * N;
    if (beta != 0.0f) {
        for (int64_t i = 0; i < total; ++i) c[i] *= beta;
    } else {
        for (int64_t i = 0; i < total; ++i) c[i] = 0.0f;
    }
    constexpr int64_t BM = 32, BN = 64, BK = 128;
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

void GPUComputeCann::reference_gemv(float alpha, const float* A, const float* x,
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

void GPUComputeCann::reference_relu(const float* x, float* y, int64_t n) {
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

void GPUComputeCann::reference_gelu(const float* x, float* y, int64_t n) {
    if (!x || !y) return;
    constexpr float c0 = 0.7978845608028654f;
    constexpr float c1 = 0.044715f;
    for (int64_t i = 0; i < n; ++i) {
        const float v = x[i];
        y[i] = 0.5f * v * (1.0f + std::tanh(c0 * (v + c1 * v * v * v)));
    }
}

void GPUComputeCann::reference_silu(const float* x, float* y, int64_t n) {
    if (!x || !y) return;
    for (int64_t i = 0; i < n; ++i) {
        const float v = x[i];
        y[i] = v / (1.0f + std::exp(-v));
    }
}

void GPUComputeCann::reference_add(const float* a, const float* b, float* c, int64_t n) {
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

void GPUComputeCann::reference_mul(const float* a, const float* b, float* c, int64_t n) {
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

void GPUComputeCann::reference_scale(float s, const float* x, float* y, int64_t n) {
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

void GPUComputeCann::reference_softmax(const float* x, float* y, int64_t rows, int64_t cols) {
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

void GPUComputeCann::reference_rms_norm(const float* x, const float* weight, float* y,
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

void GPUComputeCann::reference_layer_norm(const float* x, const float* gamma, const float* beta,
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

static GPUComputeCann g_cann_compute;
GPUComputeCann& get_cann_compute() { return g_cann_compute; }

} // namespace gpu
} // namespace quant
