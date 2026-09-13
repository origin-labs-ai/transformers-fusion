#pragma once
#include <cstdint>
#include <cstddef>
#include <map>
#include <mutex>
#include <string>

namespace quant {

class GpuComputeHip {
public:
    GpuComputeHip();
    ~GpuComputeHip();

    bool init();
    void* alloc(size_t size);
    void free(void* ptr);
    void copy_to_device(void* dst, const void* src, size_t size);
    void copy_to_host(void* dst, const void* src, size_t size);

    // Device launches. Every method attempts genuine HIP dispatch (hiprtc-
    // compiled kernels) and throws std::runtime_error on any failure —
    // NEVER silent host substitution (REAL ONLY).
    void launch_gemm(int m, int n, int k, const float* a, const float* b, float* c);
    void launch_relu(float* data, size_t size);
    void launch_silu(float* data, size_t size);
    void launch_gelu(float* data, size_t size);
    void launch_softmax(float* data, int rows, int cols);
    void launch_rmsnorm(float* data, const float* weight, int rows, int cols, float eps);
    void launch_add(const float* a, const float* b, float* c, size_t size);
    void launch_mul(const float* a, const float* b, float* c, size_t size);
    void launch_scale(float* data, float scale, size_t size);
    void launch_fill(float* data, float value, size_t size);
    void launch_rope(float* q, float* k, int seq_len, int head_dim, size_t num_heads);
    void launch_attention(const float* q, const float* k, const float* v, float* out, int seq_len, int head_dim);

private:
    void* lib_handle_;
    void* rtc_handle_;

    // Function pointers
    int (*hip_init_)(unsigned int);
    int (*hip_set_device_)(int);
    int (*hip_malloc_)(void**, size_t);
    int (*hip_free_)(void*);
    int (*hip_memcpy_)(void*, const void*, size_t, int);
    int (*hip_module_load_data_)(void**, const void*);
    int (*hip_module_get_function_)(void**, void*, const char*);
    int (*hip_module_launch_kernel_)(void*, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int,
                                     unsigned int, void*, void**, void**);
    // hiprtc JIT (opaque program handles as void*; signatures per hiprtc.h)
    int (*hiprtc_create_program_)(void**, const char*, const char*, int, const char**, const char**);
    int (*hiprtc_compile_program_)(void*, int, const char**);
    int (*hiprtc_get_code_size_)(void*, size_t*);
    int (*hiprtc_get_code_)(void*, char*);
    int (*hiprtc_destroy_program_)(void**);
    // Process-lifetime compiled-module cache (never unloaded; standard JIT).
    std::map<std::string, void*> rtc_modules_;
    std::mutex rtc_mu_;
    // Compiles kHipKernelSource once (gfx906) and returns the module for the
    // named kernel, or nullptr on any failure. Caller must hold no locks.
    void* compile_kernel_module(const char* kernel_name);
    // Launches a named kernel from an already-compiled module. True on success.
    bool launch_named_kernel(void* module, const char* kernel_name,
                             unsigned int gx, unsigned int gy, unsigned int gz,
                             unsigned int bx, unsigned int by, unsigned int bz,
                             unsigned int shared_mem, void** args);
    // Upload/grid-stride-kernel/download for in-place elementwise kernels.
    // Throws on any failure (REAL ONLY).
    void launch_inplace_kernel(float* data, size_t size, const char* kernel_name,
                               unsigned int shared_mem);
    // Upload/launch/download for out-of-place binary kernels (add/mul).
    // Throws on any failure (REAL ONLY).
    void launch_binary_kernel(const float* a, const float* b, float* c,
                              size_t size, const char* kernel_name);
};

} // namespace quant
