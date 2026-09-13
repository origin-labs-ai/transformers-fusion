#include "quant/gpu_compute_zendnn.h"
#include <cstring>
#include <cstdlib>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(__AVX2__) || defined(QUANT_AVX2)
#include <immintrin.h>
#endif

namespace quant {
namespace gpu {

namespace {
    void* load_lib(const char* name) {
#if defined(_WIN32)
        return LoadLibraryA(name);
#else
        return dlopen(name, RTLD_LAZY | RTLD_LOCAL);
#endif
    }
    void* get_sym(void* handle, const char* name) {
#if defined(_WIN32)
        return (void*)GetProcAddress((HMODULE)handle, name);
#else
        return dlsym(handle, name);
#endif
    }
    void close_lib(void* handle) {
#if defined(_WIN32)
        if (handle) FreeLibrary((HMODULE)handle);
#else
        if (handle) dlclose(handle);
#endif
    }

    typedef void (*zendnn_sgemm_fn)(char*, char*, int*, int*, int*,
                                    float*, const float*, int*,
                                    const float*, int*, float*, float*, int*);
}

GPUComputeZenDNN::~GPUComputeZenDNN() {
    close_lib(handle_);
    handle_ = nullptr;
}

bool GPUComputeZenDNN::init(int device_id) {
    (void)device_id;
    if (initialized_) return true;
#ifdef _WIN32
    handle_ = load_lib("zendnn.dll");
#else
    handle_ = load_lib("libzendnn.so");
#endif
    if (!handle_) return false;
    sgemm_ = (zendnn_sgemm_fn)get_sym(handle_, "zendnn_sgemm");
    initialized_ = true;
    return true;
}

void* GPUComputeZenDNN::alloc(size_t size) {
    return std::malloc(size ? size : 1);
}

void GPUComputeZenDNN::free_buf(void* ptr) {
    std::free(ptr);
}

void GPUComputeZenDNN::upload(const void* src, void* dst, size_t size) {
    if (src && dst && size) std::memcpy(dst, src, size);
}

void GPUComputeZenDNN::download(const void* src, void* dst, size_t size) {
    if (src && dst && size) std::memcpy(dst, src, size);
}

void GPUComputeZenDNN::gemm(float alpha, const void* A, const void* B,
                            float beta, void* C,
                            int64_t M, int64_t N, int64_t K) {
    const float* a = static_cast<const float*>(A);
    const float* b = static_cast<const float*>(B);
    float* c = static_cast<float*>(C);
    if (!a || !b || !c || M <= 0 || N <= 0 || K <= 0) return;

    if (sgemm_) {
        char transA = 'N', transB = 'N';
        int m = (int)M, n = (int)N, k = (int)K;
        float beta0 = beta;
        ((zendnn_sgemm_fn)sgemm_)(&transA, &transB, &m, &n, &k, &alpha, a, &k, b, &n, &beta0, c, &n);
        return;
    }

    if (beta != 0.0f) {
        for (int64_t i = 0; i < M * N; i++) c[i] *= beta;
    } else {
        std::memset(c, 0, (size_t)(M * N) * sizeof(float));
    }
    for (int64_t i = 0; i < M; i++) {
        for (int64_t l = 0; l < K; l++) {
            float aval = alpha * a[i * K + l];
            if (aval == 0.0f) continue;
            const float* brow = b + l * N;
            float* crow = c + i * N;
            int64_t j = 0;
#if defined(__AVX2__) || defined(QUANT_AVX2)
            __m256 av = _mm256_set1_ps(aval);
            for (; j + 8 <= N; j += 8) {
                __m256 bv = _mm256_loadu_ps(brow + j);
                __m256 cv = _mm256_loadu_ps(crow + j);
                _mm256_storeu_ps(crow + j, _mm256_fmadd_ps(av, bv, cv));
            }
#endif
            for (; j < N; j++) crow[j] += aval * brow[j];
        }
    }
}

int64_t GPUComputeZenDNN::memory_free() const {
#if defined(_WIN32)
    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    GlobalMemoryStatusEx(&mem);
    return (int64_t)mem.ullAvailPhys;
#else
    return (int64_t)0;
#endif
}

int64_t GPUComputeZenDNN::memory_total() const {
#if defined(_WIN32)
    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    GlobalMemoryStatusEx(&mem);
    return (int64_t)mem.ullTotalPhys;
#else
    return (int64_t)0;
#endif
}

void GPUComputeZenDNN::synchronize() {}

GPUComputeZenDNN& get_zendnn_compute() {
    static GPUComputeZenDNN instance;
    return instance;
}

} // namespace gpu
} // namespace quant
