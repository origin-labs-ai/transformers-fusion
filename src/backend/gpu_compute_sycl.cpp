#include "quant/gpu_compute_sycl.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace quant {
namespace gpu {

// Real oneAPI Level Zero / SYCL runtime PI (Plugin Interface) symbols.
// These are the actual exported symbols from the SYCL runtime (libsycl.so / sycl7.dll).
// The PI layer provides a C-API abstraction over Level Zero, OpenCL, and CUDA backends.
typedef void* pi_platform;
typedef void* pi_device;
typedef void* pi_context;
typedef void* pi_queue;
typedef void* pi_mem;
typedef void* pi_program;
typedef void* pi_kernel;
typedef int32_t pi_result;

// PI device type enum
constexpr uint64_t PI_DEVICE_TYPE_GPU = 4;

// PI mem flags
constexpr uint64_t PI_MEM_FLAGS_ACCESS_RW = 1;

// PI USM type
constexpr int PI_USM_TYPE_DEVICE = 2;

// Real PI function signatures from SYCL runtime
typedef pi_result (*PFN_piPlatformsGet)(uint32_t, pi_platform*, uint32_t*);
typedef pi_result (*PFN_piDevicesGet)(pi_platform, uint64_t, uint32_t, pi_device*, uint32_t*);
typedef pi_result (*PFN_piContextCreate)(const void*, uint32_t, const pi_device*, void(*)(const char*, const void*, size_t, void*), void*, pi_context*);
typedef pi_result (*PFN_piQueueCreate)(pi_context, pi_device, uint64_t, pi_queue*);
typedef pi_result (*PFN_piQueueFinish)(pi_queue);
typedef pi_result (*PFN_piQueueRelease)(pi_queue);
typedef pi_result (*PFN_piContextRelease)(pi_context);
typedef pi_result (*PFN_piextUSMDeviceAlloc)(void**, pi_context, pi_device, const void*, size_t, uint32_t);
typedef pi_result (*PFN_piextUSMFree)(pi_context, void*);
typedef pi_result (*PFN_piextUSMEnqueueMemcpy)(pi_queue, int, void*, const void*, size_t, uint32_t, const void*, void*);
typedef pi_result (*PFN_piDeviceGetInfo)(pi_device, uint32_t, size_t, void*, size_t*);

// PI device info queries
constexpr uint32_t PI_DEVICE_INFO_GLOBAL_MEM_SIZE = 0x101F;
constexpr uint32_t PI_DEVICE_INFO_GLOBAL_MEM_FREE = 0x1044;

struct SYCLAPI {
    PFN_piPlatformsGet piPlatformsGet = nullptr;
    PFN_piDevicesGet piDevicesGet = nullptr;
    PFN_piContextCreate piContextCreate = nullptr;
    PFN_piQueueCreate piQueueCreate = nullptr;
    PFN_piQueueFinish piQueueFinish = nullptr;
    PFN_piQueueRelease piQueueRelease = nullptr;
    PFN_piContextRelease piContextRelease = nullptr;
    PFN_piextUSMDeviceAlloc piextUSMDeviceAlloc = nullptr;
    PFN_piextUSMFree piextUSMFree = nullptr;
    PFN_piextUSMEnqueueMemcpy piextUSMEnqueueMemcpy = nullptr;
    PFN_piDeviceGetInfo piDeviceGetInfo = nullptr;
    void* handle = nullptr;
    bool runtime_usable = false;
};

// ========================================================================
// CPU fallback implementations (used when SYCL runtime is unavailable or
// for operations that require the SYCL compiler for GPU dispatch)
// ========================================================================

static void cpu_gemm(float alpha, const float* A, const float* B, float beta, float* C,
                      int64_t M, int64_t N, int64_t K) {
    constexpr int64_t TILE = 64;
    for (int64_t m0 = 0; m0 < M; m0 += TILE) {
        int64_t m1 = std::min(m0 + TILE, M);
        for (int64_t n0 = 0; n0 < N; n0 += TILE) {
            int64_t n1 = std::min(n0 + TILE, N);
            for (int64_t k0 = 0; k0 < K; k0 += TILE) {
                int64_t k1 = std::min(k0 + TILE, K);
                for (int64_t m = m0; m < m1; m++) {
                    for (int64_t n = n0; n < n1; n++) {
                        float s = (k0 == 0) ? 0.0f : C[m * N + n];
                        for (int64_t k = k0; k < k1; k++)
                            s += A[m * K + k] * B[k * N + n];
                        if (k0 == 0)
                            C[m * N + n] = alpha * s + beta * C[m * N + n];
                        else
                            C[m * N + n] += alpha * s;
                    }
                }
            }
        }
    }
}

static void cpu_gemv(float alpha, const float* A, const float* x, float beta, float* y,
                      int64_t M, int64_t N) {
    for (int64_t m = 0; m < M; m++) {
        float s = 0;
        for (int64_t n = 0; n < N; n++) s += A[m * N + n] * x[n];
        y[m] = alpha * s + beta * y[m];
    }
}

static void cpu_relu(const float* x, float* y, int64_t n) {
    for (int64_t i = 0; i < n; i++) y[i] = x[i] > 0 ? x[i] : 0;
}

static void cpu_gelu(const float* x, float* y, int64_t n) {
    const float c = 0.7978845608028654f;
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
    }
}

static void cpu_silu(const float* x, float* y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v / (1.0f + std::exp(-v));
    }
}

static void cpu_softmax(const float* x, float* y, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; r++) {
        float mx = x[r * cols];
        for (int64_t c = 1; c < cols; c++) mx = std::max(mx, x[r * cols + c]);
        float sum = 0;
        for (int64_t c = 0; c < cols; c++) {
            y[r * cols + c] = std::exp(x[r * cols + c] - mx);
            sum += y[r * cols + c];
        }
        float inv = 1.0f / (sum + 1e-10f);
        for (int64_t c = 0; c < cols; c++) y[r * cols + c] *= inv;
    }
}

static void cpu_rms_norm(const float* x, const float* weight, float* y,
                          int64_t rows, int64_t cols, float eps) {
    for (int64_t i = 0; i < rows; i++) {
        float ss = 0;
        for (int64_t j = 0; j < cols; j++) ss += x[i * cols + j] * x[i * cols + j];
        float rs = 1.0f / std::sqrt(ss / cols + eps);
        for (int64_t j = 0; j < cols; j++) y[i * cols + j] = x[i * cols + j] * rs * weight[j];
    }
}

static void cpu_layer_norm(const float* x, const float* gamma, const float* beta,
                             float* y, int64_t rows, int64_t cols, float eps) {
    for (int64_t i = 0; i < rows; i++) {
        float mn = 0;
        for (int64_t j = 0; j < cols; j++) mn += x[i * cols + j];
        mn /= cols;
        float vr = 0;
        for (int64_t j = 0; j < cols; j++) {
            float df = x[i * cols + j] - mn;
            vr += df * df;
        }
        vr /= cols;
        float iv = 1.0f / std::sqrt(vr + eps);
        for (int64_t j = 0; j < cols; j++)
            y[i * cols + j] = (x[i * cols + j] - mn) * iv * gamma[j] + beta[j];
    }
}

static void cpu_add(const float* a, const float* b, float* c, int64_t n) {
    for (int64_t i = 0; i < n; i++) c[i] = a[i] + b[i];
}

static void cpu_mul(const float* a, const float* b, float* c, int64_t n) {
    for (int64_t i = 0; i < n; i++) c[i] = a[i] * b[i];
}

static void cpu_scale(float s, const float* x, float* y, int64_t n) {
    for (int64_t i = 0; i < n; i++) y[i] = s * x[i];
}

static void cpu_fill(float val, float* x, int64_t n) {
    for (int64_t i = 0; i < n; i++) x[i] = val;
}

// ========================================================================
// GPUComputeSycl::Impl
// ========================================================================

struct GPUComputeSycl::Impl {
    SYCLAPI api;
    bool initialized = false;
    bool gpu_ok = false;
    pi_platform platform = nullptr;
    pi_device device = nullptr;
    pi_context context = nullptr;
    pi_queue queue = nullptr;

    bool load_sycl() {
        if (api.handle) return true;
#if defined(_WIN32)
        api.handle = LoadLibraryA("sycl7.dll");
        if (!api.handle) api.handle = LoadLibraryA("sycl6.dll");
        if (!api.handle) api.handle = LoadLibraryA("sycl.dll");
        if (!api.handle) return false;
        #define LOAD_PI(name) api.name = (PFN_##name)GetProcAddress((HMODULE)api.handle, #name)
#else
        api.handle = dlopen("libsycl.so.7", RTLD_LAZY | RTLD_LOCAL);
        if (!api.handle) api.handle = dlopen("libsycl.so.6", RTLD_LAZY | RTLD_LOCAL);
        if (!api.handle) api.handle = dlopen("libsycl.so", RTLD_LAZY | RTLD_LOCAL);
        if (!api.handle) return false;
        #define LOAD_PI(name) api.name = (PFN_##name)dlsym(api.handle, #name)
#endif
        LOAD_PI(piPlatformsGet);
        LOAD_PI(piDevicesGet);
        LOAD_PI(piContextCreate);
        LOAD_PI(piQueueCreate);
        LOAD_PI(piQueueFinish);
        LOAD_PI(piQueueRelease);
        LOAD_PI(piContextRelease);
        LOAD_PI(piextUSMDeviceAlloc);
        LOAD_PI(piextUSMFree);
        LOAD_PI(piextUSMEnqueueMemcpy);
        LOAD_PI(piDeviceGetInfo);
#undef LOAD_PI

        api.runtime_usable = (api.piPlatformsGet && api.piDevicesGet &&
                              api.piContextCreate && api.piQueueCreate);
        return true;
    }

    bool init_gpu(int64_t device_id) {
        if (!api.runtime_usable) return false;

        uint32_t num_platforms = 0;
        if (api.piPlatformsGet(0, nullptr, &num_platforms) != 0 || num_platforms == 0)
            return false;

        std::vector<pi_platform> platforms(num_platforms);
        if (api.piPlatformsGet(num_platforms, platforms.data(), nullptr) != 0)
            return false;

        for (uint32_t p = 0; p < num_platforms; p++) {
            uint32_t num_devices = 0;
            if (api.piDevicesGet(platforms[p], PI_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) != 0)
                continue;
            if (num_devices == 0) continue;

            std::vector<pi_device> devices(num_devices);
            if (api.piDevicesGet(platforms[p], PI_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr) != 0)
                continue;

            uint32_t idx = (device_id >= 0 && (uint32_t)device_id < num_devices)
                           ? (uint32_t)device_id : 0;
            device = devices[idx];
            platform = platforms[p];

            if (api.piContextCreate(nullptr, 1, &device, nullptr, nullptr, &context) != 0)
                continue;

            if (api.piQueueCreate(context, device, 0, &queue) != 0) {
                if (api.piContextRelease) api.piContextRelease(context);
                context = nullptr;
                continue;
            }

            return true;
        }
        return false;
    }
};

GPUComputeSycl::GPUComputeSycl() : impl_(new Impl()) {}
GPUComputeSycl::~GPUComputeSycl() {
    shutdown();
    delete impl_;
}

bool GPUComputeSycl::init(int64_t device_id) {
    if (impl_->initialized) return impl_->gpu_ok;

    impl_->load_sycl();

    if (impl_->init_gpu(device_id)) {
        impl_->gpu_ok = true;
    }

    impl_->initialized = true;
    // REAL ONLY: report the DEVICE, not "library touched". Previous code
    // returned true (and is_initialized() likewise) even with no SYCL GPU,
    // letting the backend claim availability while every kernel ran on CPU.
    return impl_->gpu_ok;
}

// REAL ONLY: true ONLY when a real SYCL GPU context+queue exists.
bool GPUComputeSycl::is_initialized() const { return impl_->initialized && impl_->gpu_ok; }

void GPUComputeSycl::shutdown() {
    if (!impl_->initialized) return;
    if (impl_->gpu_ok) {
        if (impl_->queue && impl_->api.piQueueFinish)
            impl_->api.piQueueFinish(impl_->queue);
        if (impl_->queue && impl_->api.piQueueRelease)
            impl_->api.piQueueRelease(impl_->queue);
        if (impl_->context && impl_->api.piContextRelease)
            impl_->api.piContextRelease(impl_->context);
        impl_->queue = nullptr;
        impl_->context = nullptr;
        impl_->gpu_ok = false;
    }
    impl_->initialized = false;
}

void* GPUComputeSycl::alloc(int64_t bytes) {
    if (!impl_->initialized) return nullptr;
    if (impl_->gpu_ok && impl_->api.piextUSMDeviceAlloc) {
        void* ptr = nullptr;
        if (impl_->api.piextUSMDeviceAlloc(&ptr, impl_->context, impl_->device,
                                           nullptr, (size_t)bytes, 0) == 0)
            return ptr;
    }
    return std::malloc((size_t)bytes);
}

void GPUComputeSycl::free_buf(void* ptr) {
    if (!impl_->initialized || !ptr) return;
    if (impl_->gpu_ok && impl_->api.piextUSMFree) {
        impl_->api.piextUSMFree(impl_->context, ptr);
        return;
    }
    std::free(ptr);
}

void GPUComputeSycl::upload(const Tensor& src, void* dst) {
    if (!impl_->initialized || !src.data() || !dst) return;
    size_t sz = src.numel() * sizeof(float);
    if (impl_->gpu_ok && impl_->api.piextUSMEnqueueMemcpy) {
        impl_->api.piextUSMEnqueueMemcpy(impl_->queue, 1, dst, src.data(), sz, 0, nullptr, nullptr);
        if (impl_->api.piQueueFinish) impl_->api.piQueueFinish(impl_->queue);
        return;
    }
    std::memcpy(dst, src.data(), sz);
}

void GPUComputeSycl::download(void* src, Tensor& dst) {
    if (!impl_->initialized || !dst.data() || !src) return;
    size_t sz = dst.numel() * sizeof(float);
    if (impl_->gpu_ok && impl_->api.piextUSMEnqueueMemcpy) {
        impl_->api.piextUSMEnqueueMemcpy(impl_->queue, 1, dst.data(), src, sz, 0, nullptr, nullptr);
        if (impl_->api.piQueueFinish) impl_->api.piQueueFinish(impl_->queue);
        return;
    }
    std::memcpy(dst.data(), src, sz);
}

// ========================================================================
// Compute operations — REAL ONLY.
// SYCL GPU kernel dispatch requires the DPC++ compiler to compile SYCL
// kernel source; without it there are NO device kernels. The cpu_* helpers
// above are HOST reference kernels for CI verification ONLY (see
// tests/test_backends_realonly.cpp) and must NEVER run disguised as SYCL
// device work: every compute entry fails loud unless a real kernel lands.
// (The ComputeBackend wrapper already throws; this is defense in depth for
// direct gpu:: callers.)
// ========================================================================

namespace {
[[noreturn]] void throw_no_sycl_kernel(const char* op) {
    std::string msg = std::string("[REAL-ONLY] GPUComputeSycl::") + op +
        " has no SYCL device kernel (DPC++ compiler required); host reference not substituted";
    std::fprintf(stderr, "%s\n", msg.c_str());
    throw std::runtime_error(msg);
}
} // namespace

void GPUComputeSycl::gemm(float alpha, const void* A, const void* B, float beta, void* C,
                           int64_t M, int64_t N, int64_t K) {
    (void)alpha; (void)A; (void)B; (void)beta; (void)C; (void)M; (void)N; (void)K;
    if (!impl_->initialized || !impl_->gpu_ok) throw_no_sycl_kernel("uninitialized");
    throw_no_sycl_kernel("gemm");
}

void GPUComputeSycl::gemv(float alpha, const void* A, const void* x, float beta, void* y,
                           int64_t M, int64_t N) {
    (void)alpha; (void)A; (void)x; (void)beta; (void)y; (void)M; (void)N;
    if (!impl_->initialized || !impl_->gpu_ok) throw_no_sycl_kernel("uninitialized");
    throw_no_sycl_kernel("gemv");
}

void GPUComputeSycl::relu(const void* x, void* y, int64_t n) {
    (void)x; (void)y; (void)n;
    if (!impl_->initialized || !impl_->gpu_ok) throw_no_sycl_kernel("uninitialized");
    throw_no_sycl_kernel("relu");
}

void GPUComputeSycl::gelu(const void* x, void* y, int64_t n) {
    (void)x; (void)y; (void)n;
    if (!impl_->initialized || !impl_->gpu_ok) throw_no_sycl_kernel("uninitialized");
    throw_no_sycl_kernel("gelu");
}

void GPUComputeSycl::silu(const void* x, void* y, int64_t n) {
    (void)x; (void)y; (void)n;
    if (!impl_->initialized || !impl_->gpu_ok) throw_no_sycl_kernel("uninitialized");
    throw_no_sycl_kernel("silu");
}

void GPUComputeSycl::softmax(const void* x, void* y, int64_t rows, int64_t cols) {
    (void)x; (void)y; (void)rows; (void)cols;
    if (!impl_->initialized || !impl_->gpu_ok) throw_no_sycl_kernel("uninitialized");
    throw_no_sycl_kernel("softmax");
}

void GPUComputeSycl::rms_norm(const void* x, const void* weight, void* y,
                               int64_t rows, int64_t cols, float eps) {
    (void)x; (void)weight; (void)y; (void)rows; (void)cols; (void)eps;
    if (!impl_->initialized || !impl_->gpu_ok) throw_no_sycl_kernel("uninitialized");
    throw_no_sycl_kernel("rms_norm");
}

void GPUComputeSycl::layer_norm(const void* x, const void* gamma, const void* beta,
                                 void* y, int64_t rows, int64_t cols, float eps) {
    (void)x; (void)gamma; (void)beta; (void)y; (void)rows; (void)cols; (void)eps;
    if (!impl_->initialized || !impl_->gpu_ok) throw_no_sycl_kernel("uninitialized");
    throw_no_sycl_kernel("layer_norm");
}

void GPUComputeSycl::add(const void* a, const void* b, void* c, int64_t n) {
    (void)a; (void)b; (void)c; (void)n;
    if (!impl_->initialized || !impl_->gpu_ok) throw_no_sycl_kernel("uninitialized");
    throw_no_sycl_kernel("add");
}

void GPUComputeSycl::mul(const void* a, const void* b, void* c, int64_t n) {
    (void)a; (void)b; (void)c; (void)n;
    if (!impl_->initialized || !impl_->gpu_ok) throw_no_sycl_kernel("uninitialized");
    throw_no_sycl_kernel("mul");
}

void GPUComputeSycl::scale(float s, const void* x, void* y, int64_t n) {
    (void)s; (void)x; (void)y; (void)n;
    if (!impl_->initialized || !impl_->gpu_ok) throw_no_sycl_kernel("uninitialized");
    throw_no_sycl_kernel("scale");
}

void GPUComputeSycl::fill(float val, void* x, int64_t n) {
    if (!impl_->initialized) return;
    cpu_fill(val, (float*)x, n);
}

void GPUComputeSycl::copy_buf(const void* src, void* dst, int64_t n) {
    if (!impl_->initialized || !src || !dst) return;
    size_t sz = (size_t)n * sizeof(float);
    if (impl_->gpu_ok && impl_->api.piextUSMEnqueueMemcpy) {
        impl_->api.piextUSMEnqueueMemcpy(impl_->queue, 1, dst, src, sz, 0, nullptr, nullptr);
        if (impl_->api.piQueueFinish) impl_->api.piQueueFinish(impl_->queue);
        return;
    }
    std::memcpy(dst, src, sz);
}

void GPUComputeSycl::synchronize() {
    if (impl_->gpu_ok && impl_->api.piQueueFinish && impl_->queue)
        impl_->api.piQueueFinish(impl_->queue);
}

int64_t GPUComputeSycl::memory_free() const {
    if (!impl_->gpu_ok || !impl_->api.piDeviceGetInfo) return 0;
    uint64_t free_mem = 0;
    size_t ret_size = 0;
    if (impl_->api.piDeviceGetInfo(impl_->device, PI_DEVICE_INFO_GLOBAL_MEM_FREE,
                                   sizeof(free_mem), &free_mem, &ret_size) == 0)
        return (int64_t)free_mem;
    return 0;
}

int64_t GPUComputeSycl::memory_total() const {
    if (!impl_->gpu_ok || !impl_->api.piDeviceGetInfo) return 0;
    uint64_t total_mem = 0;
    size_t ret_size = 0;
    if (impl_->api.piDeviceGetInfo(impl_->device, PI_DEVICE_INFO_GLOBAL_MEM_SIZE,
                                   sizeof(total_mem), &total_mem, &ret_size) == 0)
        return (int64_t)total_mem;
    return 0;
}

static GPUComputeSycl g_sycl_compute;
GPUComputeSycl& get_sycl_compute() { return g_sycl_compute; }

} // namespace gpu
} // namespace quant
