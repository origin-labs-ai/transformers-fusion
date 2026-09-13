#include "quant/gpu_compute_hip.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#undef min
#undef max
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

// HIP memcpy kinds
constexpr int hipMemcpyHostToDevice = 1;
constexpr int hipMemcpyDeviceToHost = 2;

// ========================================================================
// REAL HIP kernels: compiled at runtime with hiprtc (libhiprtc), loaded via
// hipModuleLoadData, launched via hipModuleLaunchKernel.
//
// HONESTY NOTE (replaces removed fake data): this file previously contained
// eight byte-identical 64-byte arrays presented as "embedded AMDGCN ISA code
// objects". They were bare ELF headers with no sections, no ISA and no
// kernel symbols, so hipModuleGetFunction could never succeed and every
// launch silently ran host CPU code while reporting GPU execution. Those
// arrays are DELETED. What remains is real: HIP C++ kernel sources below,
// genuine hiprtc compilation, genuine module launch, and fail-loud errors
// when any step is unavailable.
//
// Target arch: gfx906 (Vega 20 / Radeon VII class). Production use should
// query the device arch via hipGetDeviceProperties and recompile;
// see compile_kernel_module().
// ========================================================================

static const char* kHipKernelSource = R"HIP(
extern "C" __global__ void relu_kernel(float* d, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (; i < n; i += stride) d[i] = fmaxf(d[i], 0.0f);
}
extern "C" __global__ void gelu_kernel(float* d, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        float x = d[i];
        d[i] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}
extern "C" __global__ void silu_kernel(float* d, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        float x = d[i];
        d[i] = x / (1.0f + expf(-x));
    }
}
extern "C" __global__ void add_kernel(const float* a, const float* b, float* c, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (; i < n; i += stride) c[i] = a[i] + b[i];
}
extern "C" __global__ void mul_kernel(const float* a, const float* b, float* c, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (; i < n; i += stride) c[i] = a[i] * b[i];
}
extern "C" __global__ void scale_kernel(float* d, float s, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (; i < n; i += stride) d[i] *= s;
}
extern "C" __global__ void fill_kernel(float* d, float v, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (; i < n; i += stride) d[i] = v;
}
extern "C" __global__ void softmax_kernel(float* d, int rows, int cols) {
    int r = (int)blockIdx.x;
    if (r >= rows) return;
    extern __shared__ float sm[];
    int tid = (int)threadIdx.x;
    int bd = (int)blockDim.x;
    float mx = -1e30f;
    for (int c = tid; c < cols; c += bd) mx = fmaxf(mx, d[r * cols + c]);
    sm[tid] = mx;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < (int)s) sm[tid] = fmaxf(sm[tid], sm[tid + s]);
        __syncthreads();
    }
    float rowmax = sm[0];
    float sum = 0.0f;
    for (int c = tid; c < cols; c += bd) {
        float e = expf(d[r * cols + c] - rowmax);
        d[r * cols + c] = e;
        sum += e;
    }
    sm[tid] = sum;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < (int)s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float inv = 1.0f / (sm[0] + 1e-10f);
    for (int c = tid; c < cols; c += bd) d[r * cols + c] *= inv;
}
extern "C" __global__ void rmsnorm_kernel(float* d, const float* w, int rows, int cols, float eps) {
    int r = (int)blockIdx.x;
    if (r >= rows) return;
    extern __shared__ float acc[];
    int tid = (int)threadIdx.x;
    int bd = (int)blockDim.x;
    float ss = 0.0f;
    for (int c = tid; c < cols; c += bd) {
        float v = d[r * cols + c];
        ss += v * v;
    }
    acc[tid] = ss;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < (int)s) acc[tid] += acc[tid + s];
        __syncthreads();
    }
    float rs = rsqrtf(acc[0] / (float)cols + eps);
    for (int c = tid; c < cols; c += bd) d[r * cols + c] *= rs * w[c];
}
extern "C" __global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row < M && col < N) {
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s += A[row * K + k] * B[k * N + col];
        C[row * N + col] = s;
    }
}
extern "C" __global__ void rope_kernel(float* q, float* k, int seq_len, int head_dim, unsigned long long num_heads) {
    int half = head_dim / 2;
    unsigned long long total = num_heads * (unsigned long long)seq_len * (unsigned long long)half;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (; i < total; i += stride) {
        unsigned long long t = i;
        int dd = (int)(t % (unsigned long long)half);
        t /= (unsigned long long)half;
        int s = (int)(t % (unsigned long long)seq_len);
        unsigned long long h = t / (unsigned long long)seq_len;
        float freq = 1.0f / powf(10000.0f, (2.0f * (float)dd) / (float)head_dim);
        float ang = (float)s * freq;
        float co = cosf(ang), si = sinf(ang);
        unsigned long long base = (h * (unsigned long long)seq_len + (unsigned long long)s) * (unsigned long long)head_dim;
        float q0 = q[base + dd], q1 = q[base + dd + half];
        float k0 = k[base + dd], k1 = k[base + dd + half];
        q[base + dd] = q0 * co - q1 * si;
        q[base + dd + half] = q0 * si + q1 * co;
        k[base + dd] = k0 * co - k1 * si;
        k[base + dd + half] = k0 * si + k1 * co;
    }
}
extern "C" __global__ void attention_kernel(const float* q, const float* k, const float* v, float* out, int seq_len, int head_dim) {
    int i = (int)blockIdx.x;
    if (i >= seq_len) return;
    extern __shared__ float scores[];
    int tid = (int)threadIdx.x;
    int bd = (int)blockDim.x;
    for (int j = tid; j < seq_len; j += bd) {
        float s;
        if (j <= i) {
            float dot = 0.0f;
            for (int dd = 0; dd < head_dim; ++dd) dot += q[i * head_dim + dd] * k[j * head_dim + dd];
            s = dot * rsqrtf((float)head_dim);
        } else {
            s = -1e9f;
        }
        scores[j] = s;
    }
    __syncthreads();
    if (tid == 0) {
        float mx = scores[0];
        for (int j = 1; j < seq_len; ++j) mx = fmaxf(mx, scores[j]);
        float sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            scores[j] = expf(scores[j] - mx);
            sum += scores[j];
        }
        float inv = 1.0f / (sum + 1e-10f);
        for (int dd = 0; dd < head_dim; ++dd) {
            float acc = 0.0f;
            for (int j = 0; j < seq_len; ++j) acc += scores[j] * v[j * head_dim + dd];
            out[i * head_dim + dd] = acc * inv;
        }
    }
}
)HIP";

// ========================================================================
// Host reference kernels — verification baseline ONLY (never substituted
// for device work; every launch_* below throws instead of calling these).
// Extern linkage (not static) so -Wunused-function can never fire.
// ========================================================================

void cpu_gemm(int m, int n, int k, const float* a, const float* b, float* c) {
    constexpr int TILE = 64;
    for (int m0 = 0; m0 < m; m0 += TILE) {
        int m1 = std::min(m0 + TILE, m);
        for (int n0 = 0; n0 < n; n0 += TILE) {
            int n1 = std::min(n0 + TILE, n);
            for (int k0 = 0; k0 < k; k0 += TILE) {
                int k1 = std::min(k0 + TILE, k);
                for (int mi = m0; mi < m1; mi++) {
                    for (int ni = n0; ni < n1; ni++) {
                        float sum = (k0 == 0) ? 0.0f : c[mi * n + ni];
                        for (int ki = k0; ki < k1; ki++)
                            sum += a[mi * k + ki] * b[ki * n + ni];
                        c[mi * n + ni] = sum;
                    }
                }
            }
        }
    }
}

void cpu_relu(float* data, size_t size) {
    for (size_t i = 0; i < size; i++) if (data[i] < 0.0f) data[i] = 0.0f;
}

void cpu_silu(float* data, size_t size) {
    for (size_t i = 0; i < size; i++) data[i] = data[i] / (1.0f + std::exp(-data[i]));
}

void cpu_gelu(float* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        float x = data[i];
        data[i] = 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}

void cpu_softmax(float* data, size_t size) {
    if (size == 0) return;
    float max_v = data[0];
    for (size_t i = 1; i < size; i++) if (data[i] > max_v) max_v = data[i];
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        data[i] = std::exp(data[i] - max_v);
        sum += data[i];
    }
    float inv = 1.0f / (sum + 1e-10f);
    for (size_t i = 0; i < size; i++) data[i] *= inv;
}

void cpu_rmsnorm(float* data, size_t size) {
    if (size == 0) return;
    float sum_sq = 0.0f;
    for (size_t i = 0; i < size; i++) sum_sq += data[i] * data[i];
    float rms = std::sqrt(sum_sq / (float)size + 1e-6f);
    float inv = 1.0f / rms;
    for (size_t i = 0; i < size; i++) data[i] *= inv;
}

void cpu_add(const float* a, const float* b, float* c, size_t size) {
    for (size_t i = 0; i < size; i++) c[i] = a[i] + b[i];
}

void cpu_mul(const float* a, const float* b, float* c, size_t size) {
    for (size_t i = 0; i < size; i++) c[i] = a[i] * b[i];
}

void cpu_scale(float* data, float scale, size_t size) {
    for (size_t i = 0; i < size; i++) data[i] *= scale;
}

void cpu_fill(float* data, float value, size_t size) {
    for (size_t i = 0; i < size; i++) data[i] = value;
}

void cpu_rope(float* q, float* k, int seq_len, int head_dim, size_t num_heads) {
    int half_d = head_dim / 2;
    for (size_t h = 0; h < num_heads; h++) {
        for (int s = 0; s < seq_len; s++) {
            float* q_head = q + (h * seq_len + s) * head_dim;
            float* k_head = k + (h * seq_len + s) * head_dim;
            for (int d = 0; d < half_d; d++) {
                float freq = 1.0f / std::pow(10000.0f, (float)(2 * d) / (float)head_dim);
                float theta = (float)s * freq;
                float cos_v = std::cos(theta), sin_v = std::sin(theta);
                float q0 = q_head[d], q1 = q_head[d + half_d];
                q_head[d] = q0 * cos_v - q1 * sin_v;
                q_head[d + half_d] = q0 * sin_v + q1 * cos_v;
                float k0 = k_head[d], k1 = k_head[d + half_d];
                k_head[d] = k0 * cos_v - k1 * sin_v;
                k_head[d + half_d] = k0 * sin_v + k1 * cos_v;
            }
        }
    }
}

void cpu_attention(const float* q, const float* k, const float* v, float* out,
                           int seq_len, int head_dim) {
    float scale = 1.0f / std::sqrt((float)head_dim);
    std::vector<float> scores(seq_len * seq_len, 0.0f);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j <= i; j++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            scores[i * seq_len + j] = dot * scale;
        }
        for (int j = i + 1; j < seq_len; j++)
            scores[i * seq_len + j] = -1e9f;
        float max_s = scores[i * seq_len];
        for (int j = 1; j < seq_len; j++)
            if (scores[i * seq_len + j] > max_s) max_s = scores[i * seq_len + j];
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[i * seq_len + j] = std::exp(scores[i * seq_len + j] - max_s);
            sum_exp += scores[i * seq_len + j];
        }
        float inv = 1.0f / (sum_exp + 1e-10f);
        for (int j = 0; j < seq_len; j++) scores[i * seq_len + j] *= inv;
        for (int d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (int j = 0; j < seq_len; j++)
                val += scores[i * seq_len + j] * v[j * head_dim + d];
            out[i * head_dim + d] = val;
        }
    }
}

// ========================================================================
// GpuComputeHip implementation with GPU kernel dispatch via
// hipModuleLoadData + hipModuleGetFunction + hipModuleLaunchKernel
// ========================================================================

GpuComputeHip::GpuComputeHip() : lib_handle_(nullptr), rtc_handle_(nullptr),
    hip_init_(nullptr), hip_set_device_(nullptr), hip_malloc_(nullptr),
    hip_free_(nullptr), hip_memcpy_(nullptr), hip_module_load_data_(nullptr),
    hip_module_get_function_(nullptr), hip_module_launch_kernel_(nullptr),
    hiprtc_create_program_(nullptr), hiprtc_compile_program_(nullptr),
    hiprtc_get_code_size_(nullptr), hiprtc_get_code_(nullptr),
    hiprtc_destroy_program_(nullptr) {}

GpuComputeHip::~GpuComputeHip() {}

bool GpuComputeHip::init() {
#ifdef _WIN32
    lib_handle_ = load_lib("amdhip64.dll");
    rtc_handle_ = load_lib("hiprtc.dll");
#else
    lib_handle_ = load_lib("libamdhip64.so");
    rtc_handle_ = load_lib("libhiprtc.so");
#endif

    if (!lib_handle_) return false;

    hip_init_ = (decltype(hip_init_))get_sym(lib_handle_, "hipInit");
    hip_set_device_ = (decltype(hip_set_device_))get_sym(lib_handle_, "hipSetDevice");
    hip_malloc_ = (decltype(hip_malloc_))get_sym(lib_handle_, "hipMalloc");
    hip_free_ = (decltype(hip_free_))get_sym(lib_handle_, "hipFree");
    hip_memcpy_ = (decltype(hip_memcpy_))get_sym(lib_handle_, "hipMemcpy");
    hip_module_load_data_ = (decltype(hip_module_load_data_))get_sym(lib_handle_, "hipModuleLoadData");
    hip_module_get_function_ = (decltype(hip_module_get_function_))get_sym(lib_handle_, "hipModuleGetFunction");
    hip_module_launch_kernel_ = (decltype(hip_module_launch_kernel_))get_sym(lib_handle_, "hipModuleLaunchKernel");

    if (rtc_handle_) {
        hiprtc_create_program_ = (decltype(hiprtc_create_program_))get_sym(rtc_handle_, "hiprtcCreateProgram");
        hiprtc_compile_program_ = (decltype(hiprtc_compile_program_))get_sym(rtc_handle_, "hiprtcCompileProgram");
        hiprtc_get_code_size_ = (decltype(hiprtc_get_code_size_))get_sym(rtc_handle_, "hiprtcGetCodeSize");
        hiprtc_get_code_ = (decltype(hiprtc_get_code_))get_sym(rtc_handle_, "hiprtcGetCode");
        hiprtc_destroy_program_ = (decltype(hiprtc_destroy_program_))get_sym(rtc_handle_, "hiprtcDestroyProgram");
    }

    // REAL ONLY: the old code ignored hipInit's return value (a missing
    // driver still reported success). A device is claimed only when hipInit
    // returns 0 AND the module-launch trio resolves (needed for every
    // kernel) AND hiprtc resolves (kernels are JIT-compiled, no embedded
    // binaries remain).
    if (!hip_init_ || hip_init_(0) != 0) return false;
    if (hip_set_device_) hip_set_device_(0);
    if (!hip_malloc_ || !hip_free_ || !hip_memcpy_) return false;
    if (!hip_module_load_data_ || !hip_module_get_function_ || !hip_module_launch_kernel_) return false;
    if (!hiprtc_create_program_ || !hiprtc_compile_program_ || !hiprtc_get_code_size_ ||
        !hiprtc_get_code_ || !hiprtc_destroy_program_)
        return false;
    return true;
}

void* GpuComputeHip::alloc(size_t size) {
    void* ptr = nullptr;
    if (hip_malloc_) hip_malloc_(&ptr, size);
    return ptr;
}

void GpuComputeHip::free(void* ptr) {
    if (hip_free_) hip_free_(ptr);
}

void GpuComputeHip::copy_to_device(void* dst, const void* src, size_t size) {
    if (hip_memcpy_) hip_memcpy_(dst, src, size, hipMemcpyHostToDevice);
}

void GpuComputeHip::copy_to_host(void* dst, const void* src, size_t size) {
    if (hip_memcpy_) hip_memcpy_(dst, src, size, hipMemcpyDeviceToHost);
}

// ========================================================================
// REAL ONLY fail-loud + hiprtc JIT compile helper.
// compile_kernel_module() compiles kHipKernelSource once per process
// (gfx906; see arch note above), loads it, and caches the module per
// kernel name. Returns nullptr on ANY failure — callers throw, never run
// host code disguised as device work.
// ========================================================================

namespace {
[[noreturn]] void throw_hip_unavailable(const char* op, const char* reason) {
    std::string msg = std::string("[REAL-ONLY] GpuComputeHip::") + op +
        " failed: " + reason + " (no host substitution)";
    std::fprintf(stderr, "%s\n", msg.c_str());
    throw std::runtime_error(msg);
}
} // namespace

void* GpuComputeHip::compile_kernel_module(const char* kernel_name) {
    std::lock_guard<std::mutex> lock(rtc_mu_);
    auto it = rtc_modules_.find(kernel_name);
    if (it != rtc_modules_.end()) return it->second;
    if (!hiprtc_create_program_ || !hiprtc_compile_program_ || !hiprtc_get_code_size_ ||
        !hiprtc_get_code_ || !hiprtc_destroy_program_ || !hip_module_load_data_ ||
        !hip_module_get_function_) {
        return nullptr;
    }
    void* prog = nullptr;
    if (hiprtc_create_program_(&prog, kHipKernelSource, "transcender_kernels",
                               0, nullptr, nullptr) != 0 || !prog) {
        return nullptr;
    }
    const char* opts[] = {"--gpu-architecture=gfx906"};
    int crc = hiprtc_compile_program_(prog, 1, opts);
    if (crc != 0) {
        hiprtc_destroy_program_(&prog);
        return nullptr;
    }
    size_t code_size = 0;
    if (hiprtc_get_code_size_(prog, &code_size) != 0 || code_size == 0) {
        hiprtc_destroy_program_(&prog);
        return nullptr;
    }
    std::vector<char> code(code_size);
    if (hiprtc_get_code_(prog, code.data()) != 0) {
        hiprtc_destroy_program_(&prog);
        return nullptr;
    }
    hiprtc_destroy_program_(&prog);
    void* module = nullptr;
    if (hip_module_load_data_(&module, code.data()) != 0 || !module) return nullptr;
    void* function = nullptr;
    if (hip_module_get_function_(&function, module, kernel_name) != 0 || !function) {
        return nullptr;
    }
    rtc_modules_[kernel_name] = module;
    return module;
}

// Launches an already-compiled kernel by name. Returns true on success.
bool GpuComputeHip::launch_named_kernel(void* module, const char* kernel_name,
                                        unsigned int gx, unsigned int gy, unsigned int gz,
                                        unsigned int bx, unsigned int by, unsigned int bz,
                                        unsigned int shared_mem, void** args) {
    if (!module || !hip_module_get_function_ || !hip_module_launch_kernel_) return false;
    void* function = nullptr;
    if (hip_module_get_function_(&function, module, kernel_name) != 0 || !function)
        return false;
    int rc = hip_module_launch_kernel_(function, gx, gy, gz, bx, by, bz,
                                       shared_mem, nullptr, args, nullptr);
    return rc == 0;
}

// ========================================================================
// Kernel launch methods — genuine HIP dispatch; throw on any failure.
// ========================================================================

// In-place elementwise launch helper: upload, grid-stride kernel, download.
// Throws on any failure (REAL ONLY — no host substitution).
void GpuComputeHip::launch_inplace_kernel(float* data, size_t size, const char* kernel_name,
                                          unsigned int shared_mem) {
    if (!data || size == 0) throw_hip_unavailable(kernel_name, "null/empty buffer");
    void* module = compile_kernel_module(kernel_name);
    if (!module) throw_hip_unavailable(kernel_name, "hiprtc compile/module load failed");
    size_t sz = size * sizeof(float);
    void* d_data = nullptr;
    if (!hip_malloc_ || hip_malloc_(&d_data, sz) != 0 || !d_data)
        throw_hip_unavailable(kernel_name, "hipMalloc failed");
    copy_to_device(d_data, data, sz);
    constexpr unsigned int block = 256;
    unsigned int grid = (unsigned int)((size + block - 1) / block);
    if (grid == 0) grid = 1;
    void* args[] = {&d_data, &size};
    bool ok = launch_named_kernel(module, kernel_name, grid, 1, 1, block, 1, 1, shared_mem, args);
    if (ok) copy_to_host(data, d_data, sz);
    if (hip_free_) hip_free_(d_data);
    if (!ok) throw_hip_unavailable(kernel_name, "kernel launch failed");
}

void GpuComputeHip::launch_gemm(int m, int n, int k, const float* a, const float* b, float* c) {
    if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0)
        throw_hip_unavailable("gemm", "null/empty GEMM input");
    void* module = compile_kernel_module("gemm_kernel");
    if (!module) throw_hip_unavailable("gemm", "hiprtc compile/module load failed");
    size_t sz_a = (size_t)m * (size_t)k * sizeof(float);
    size_t sz_b = (size_t)k * (size_t)n * sizeof(float);
    size_t sz_c = (size_t)m * (size_t)n * sizeof(float);
    void* d_a = nullptr; void* d_b = nullptr; void* d_c = nullptr;
    if (!hip_malloc_ || hip_malloc_(&d_a, sz_a) != 0 ||
        hip_malloc_(&d_b, sz_b) != 0 || hip_malloc_(&d_c, sz_c) != 0) {
        if (hip_free_) { if (d_a) hip_free_(d_a); if (d_b) hip_free_(d_b); if (d_c) hip_free_(d_c); }
        throw_hip_unavailable("gemm", "hipMalloc failed");
    }
    copy_to_device(d_a, a, sz_a);
    copy_to_device(d_b, b, sz_b);
    constexpr unsigned int bx = 16, by = 16;
    unsigned int gx = ((unsigned int)n + bx - 1) / bx;
    unsigned int gy = ((unsigned int)m + by - 1) / by;
    void* args[] = {&d_a, &d_b, &d_c, &m, &n, &k};
    bool ok = launch_named_kernel(module, "gemm_kernel", gx, gy, 1, bx, by, 1, 0, args);
    if (ok) copy_to_host((void*)c, d_c, sz_c);
    if (hip_free_) { hip_free_(d_a); hip_free_(d_b); hip_free_(d_c); }
    if (!ok) throw_hip_unavailable("gemm", "kernel launch failed");
}

void GpuComputeHip::launch_relu(float* data, size_t size) {
    launch_inplace_kernel(data, size, "relu_kernel", 0);
}

void GpuComputeHip::launch_silu(float* data, size_t size) {
    launch_inplace_kernel(data, size, "silu_kernel", 0);
}

void GpuComputeHip::launch_gelu(float* data, size_t size) {
    launch_inplace_kernel(data, size, "gelu_kernel", 0);
}

void GpuComputeHip::launch_softmax(float* data, int rows, int cols) {
    if (!data || rows <= 0 || cols <= 0)
        throw_hip_unavailable("softmax", "null/empty input");
    void* module = compile_kernel_module("softmax_kernel");
    if (!module) throw_hip_unavailable("softmax", "hiprtc compile/module load failed");
    size_t sz = (size_t)rows * (size_t)cols * sizeof(float);
    void* d_data = nullptr;
    if (!hip_malloc_ || hip_malloc_(&d_data, sz) != 0 || !d_data)
        throw_hip_unavailable("softmax", "hipMalloc failed");
    copy_to_device(d_data, data, sz);
    constexpr unsigned int block = 256;
    void* args[] = {&d_data, &rows, &cols};
    bool ok = launch_named_kernel(module, "softmax_kernel", (unsigned int)rows, 1, 1,
                                  block, 1, 1, block * (unsigned int)sizeof(float), args);
    if (ok) copy_to_host(data, d_data, sz);
    if (hip_free_) hip_free_(d_data);
    if (!ok) throw_hip_unavailable("softmax", "kernel launch failed");
}

void GpuComputeHip::launch_rmsnorm(float* data, const float* weight, int rows, int cols, float eps) {
    if (!data || !weight || rows <= 0 || cols <= 0)
        throw_hip_unavailable("rmsnorm", "null/empty input");
    void* module = compile_kernel_module("rmsnorm_kernel");
    if (!module) throw_hip_unavailable("rmsnorm", "hiprtc compile/module load failed");
    size_t sz = (size_t)rows * (size_t)cols * sizeof(float);
    void* d_data = nullptr; void* d_w = nullptr;
    if (!hip_malloc_ || hip_malloc_(&d_data, sz) != 0 || !d_data)
        throw_hip_unavailable("rmsnorm", "hipMalloc failed");
    if (hip_malloc_(&d_w, (size_t)cols * sizeof(float)) != 0 || !d_w) {
        if (hip_free_) hip_free_(d_data);
        throw_hip_unavailable("rmsnorm", "hipMalloc failed");
    }
    copy_to_device(d_data, data, sz);
    copy_to_device(d_w, weight, (size_t)cols * sizeof(float));
    constexpr unsigned int block = 256;
    void* args[] = {&d_data, &d_w, &rows, &cols, &eps};
    bool ok = launch_named_kernel(module, "rmsnorm_kernel", (unsigned int)rows, 1, 1,
                                  block, 1, 1, block * (unsigned int)sizeof(float), args);
    if (ok) copy_to_host(data, d_data, sz);
    if (hip_free_) { hip_free_(d_data); hip_free_(d_w); }
    if (!ok) throw_hip_unavailable("rmsnorm", "kernel launch failed");
}

// Out-of-place binary elementwise launch helper (add/mul). Throws on failure.
void GpuComputeHip::launch_binary_kernel(const float* a, const float* b, float* c,
                                         size_t size, const char* kernel_name) {
    if (!a || !b || !c || size == 0)
        throw_hip_unavailable(kernel_name, "null/empty input");
    void* module = compile_kernel_module(kernel_name);
    if (!module) throw_hip_unavailable(kernel_name, "hiprtc compile/module load failed");
    size_t sz = size * sizeof(float);
    void* d_a = nullptr; void* d_b = nullptr; void* d_c = nullptr;
    if (!hip_malloc_ || hip_malloc_(&d_a, sz) != 0 || hip_malloc_(&d_b, sz) != 0 ||
        hip_malloc_(&d_c, sz) != 0) {
        if (hip_free_) { if (d_a) hip_free_(d_a); if (d_b) hip_free_(d_b); if (d_c) hip_free_(d_c); }
        throw_hip_unavailable(kernel_name, "hipMalloc failed");
    }
    copy_to_device(d_a, a, sz);
    copy_to_device(d_b, b, sz);
    constexpr unsigned int block = 256;
    unsigned int grid = (unsigned int)((size + block - 1) / block);
    if (grid == 0) grid = 1;
    void* args[] = {&d_a, &d_b, &d_c, &size};
    bool ok = launch_named_kernel(module, kernel_name, grid, 1, 1, block, 1, 1, 0, args);
    if (ok) copy_to_host((void*)c, d_c, sz);
    if (hip_free_) { hip_free_(d_a); hip_free_(d_b); hip_free_(d_c); }
    if (!ok) throw_hip_unavailable(kernel_name, "kernel launch failed");
}

void GpuComputeHip::launch_add(const float* a, const float* b, float* c, size_t size) {
    launch_binary_kernel(a, b, c, size, "add_kernel");
}

void GpuComputeHip::launch_mul(const float* a, const float* b, float* c, size_t size) {
    launch_binary_kernel(a, b, c, size, "mul_kernel");
}

void GpuComputeHip::launch_scale(float* data, float scale, size_t size) {
    if (!data || size == 0) throw_hip_unavailable("scale", "null/empty input");
    void* module = compile_kernel_module("scale_kernel");
    if (!module) throw_hip_unavailable("scale", "hiprtc compile/module load failed");
    size_t sz = size * sizeof(float);
    void* d_data = nullptr;
    if (!hip_malloc_ || hip_malloc_(&d_data, sz) != 0 || !d_data)
        throw_hip_unavailable("scale", "hipMalloc failed");
    copy_to_device(d_data, data, sz);
    constexpr unsigned int block = 256;
    unsigned int grid = (unsigned int)((size + block - 1) / block);
    if (grid == 0) grid = 1;
    void* args[] = {&d_data, &scale, &size};
    bool ok = launch_named_kernel(module, "scale_kernel", grid, 1, 1, block, 1, 1, 0, args);
    if (ok) copy_to_host(data, d_data, sz);
    if (hip_free_) hip_free_(d_data);
    if (!ok) throw_hip_unavailable("scale", "kernel launch failed");
}

void GpuComputeHip::launch_fill(float* data, float value, size_t size) {
    if (!data || size == 0) throw_hip_unavailable("fill", "null/empty input");
    void* module = compile_kernel_module("fill_kernel");
    if (!module) throw_hip_unavailable("fill", "hiprtc compile/module load failed");
    size_t sz = size * sizeof(float);
    void* d_data = nullptr;
    if (!hip_malloc_ || hip_malloc_(&d_data, sz) != 0 || !d_data)
        throw_hip_unavailable("fill", "hipMalloc failed");
    constexpr unsigned int block = 256;
    unsigned int grid = (unsigned int)((size + block - 1) / block);
    if (grid == 0) grid = 1;
    void* args[] = {&d_data, &value, &size};
    bool ok = launch_named_kernel(module, "fill_kernel", grid, 1, 1, block, 1, 1, 0, args);
    if (ok) copy_to_host(data, d_data, sz);
    if (hip_free_) hip_free_(d_data);
    if (!ok) throw_hip_unavailable("fill", "kernel launch failed");
}

void GpuComputeHip::launch_rope(float* q, float* k, int seq_len, int head_dim, size_t num_heads) {
    if (!q || !k || seq_len <= 0 || head_dim <= 0 || num_heads == 0)
        throw_hip_unavailable("rope", "null/empty input");
    void* module = compile_kernel_module("rope_kernel");
    if (!module) throw_hip_unavailable("rope", "hiprtc compile/module load failed");
    size_t sz = num_heads * (size_t)seq_len * (size_t)head_dim * sizeof(float);
    void* d_q = nullptr; void* d_k = nullptr;
    if (!hip_malloc_ || hip_malloc_(&d_q, sz) != 0 || !d_q)
        throw_hip_unavailable("rope", "hipMalloc failed");
    if (hip_malloc_(&d_k, sz) != 0 || !d_k) {
        if (hip_free_) hip_free_(d_q);
        throw_hip_unavailable("rope", "hipMalloc failed");
    }
    copy_to_device(d_q, q, sz);
    copy_to_device(d_k, k, sz);
    constexpr unsigned int block = 256;
    unsigned long long total = num_heads * (unsigned long long)seq_len * (unsigned long long)(head_dim / 2);
    unsigned int grid = (unsigned int)((total + block - 1) / block);
    if (grid == 0) grid = 1;
    void* args[] = {&d_q, &d_k, &seq_len, &head_dim, &num_heads};
    bool ok = launch_named_kernel(module, "rope_kernel", grid, 1, 1, block, 1, 1, 0, args);
    if (ok) {
        copy_to_host(q, d_q, sz);
        copy_to_host(k, d_k, sz);
    }
    if (hip_free_) { hip_free_(d_q); hip_free_(d_k); }
    if (!ok) throw_hip_unavailable("rope", "kernel launch failed");
}

void GpuComputeHip::launch_attention(const float* q, const float* k, const float* v, float* out,
                                      int seq_len, int head_dim) {
    if (!q || !k || !v || !out || seq_len <= 0 || head_dim <= 0)
        throw_hip_unavailable("attention", "null/empty input");
    void* module = compile_kernel_module("attention_kernel");
    if (!module) throw_hip_unavailable("attention", "hiprtc compile/module load failed");
    size_t sz_qkv = (size_t)seq_len * (size_t)head_dim * sizeof(float);
    void* d_q = nullptr; void* d_k = nullptr; void* d_v = nullptr; void* d_o = nullptr;
    if (!hip_malloc_ || hip_malloc_(&d_q, sz_qkv) != 0 || !d_q)
        throw_hip_unavailable("attention", "hipMalloc failed");
    bool mem_ok = hip_malloc_(&d_k, sz_qkv) == 0 && d_k &&
                  hip_malloc_(&d_v, sz_qkv) == 0 && d_v &&
                  hip_malloc_(&d_o, sz_qkv) == 0 && d_o;
    if (!mem_ok) {
        if (hip_free_) { hip_free_(d_q); if (d_k) hip_free_(d_k); if (d_v) hip_free_(d_v); if (d_o) hip_free_(d_o); }
        throw_hip_unavailable("attention", "hipMalloc failed");
    }
    copy_to_device(d_q, q, sz_qkv);
    copy_to_device(d_k, k, sz_qkv);
    copy_to_device(d_v, v, sz_qkv);
    constexpr unsigned int block = 256;
    void* args[] = {&d_q, &d_k, &d_v, &d_o, &seq_len, &head_dim};
    bool ok = launch_named_kernel(module, "attention_kernel", (unsigned int)seq_len, 1, 1,
                                  block, 1, 1, (unsigned int)seq_len * (unsigned int)sizeof(float), args);
    if (ok) copy_to_host((void*)out, d_o, sz_qkv);
    if (hip_free_) { hip_free_(d_q); hip_free_(d_k); hip_free_(d_v); hip_free_(d_o); }
    if (!ok) throw_hip_unavailable("attention", "kernel launch failed");
}

} // namespace quant
