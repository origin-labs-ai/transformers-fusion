#include "quant/gpu_compute_opencl.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>

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

    const char* kOpenCLGEMMKernel = R"(
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_kernel(const int M, const int N, const int K,
                 __global const float* A,
                 __global const float* B,
                 __global float* C) {
        __local float As[16][16];
        __local float Bs[16][16];
        int row = get_group_id(1) * 16 + get_local_id(1);
        int col = get_group_id(0) * 16 + get_local_id(0);
        int lx = get_local_id(0);
        int ly = get_local_id(1);
        float acc = 0.0f;
        for (int t0 = 0; t0 < K; t0 += 16) {
            int aCol = t0 + lx;
            int bRow = t0 + ly;
            As[ly][lx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
            Bs[ly][lx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int k = 0; k < 16; ++k) acc += As[ly][k] * Bs[k][lx];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (row < M && col < N) C[row * N + col] = acc;
    }
)";
    const char* kOpenCLGEMVKernel = R"(
__kernel void gemv_kernel(const int M, const int N,
                          __global const float* A,
                          __global const float* X,
                          __global float* Y) {
        int row = get_global_id(0);
        if (row < M) {
            float sum = 0.0f;
            for (int n = 0; n < N; ++n) sum += A[row * N + n] * X[n];
            Y[row] = sum;
        }
    }
)";
    const char* kOpenCLEltKernel = R"(
__kernel void elt_kernel(const int op, const float s, const int n,
                         __global const float* A,
                         __global const float* B,
                         __global float* C) {
        int i = get_global_id(0);
        if (i >= n) return;
        float a = A[i];
        float b = B ? B[i] : 0.0f;
        float y = 0.0f;
        if (op == 0) y = a > 0.0f ? a : 0.0f;
        else if (op == 1) y = 0.5f * a * (1.0f + erf(a * 0.70710678f));
        else if (op == 2) y = a / (1.0f + exp(-a));
        else if (op == 3) y = a + b;
        else if (op == 4) y = a * b;
        else if (op == 5) y = s * a;
        C[i] = y;
    }
)";
    const char* kOpenCLNormKernel = R"(
__kernel void norm_kernel(const int op, const float eps, const int rows, const int cols,
                          __global const float* X,
                          __global const float* G,
                          __global const float* B,
                          __global float* Y) {
        int r = get_global_id(0);
        if (r >= rows) return;
        __global const float* xr = X + (int64_t)r * cols;
        float m = xr[0];
        for (int c = 1; c < cols; ++c) m = max(m, xr[c]);
        float sum = 0.0f;
        float sq = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float e = exp(xr[c] - m);
            sum += e;
            float v = xr[c];
            sq += v * v;
        }
        float mean_sq = sq / (float)cols;
        float rms = sqrt(mean_sq + eps);
        float mean = 0.0f;
        if (op == 2) {
            for (int c = 0; c < cols; ++c) mean += xr[c];
            mean /= (float)cols;
        }
        float var = 0.0f;
        if (op == 2) {
            for (int c = 0; c < cols; ++c) { float d = xr[c] - mean; var += d * d; }
            var /= (float)cols;
        }
        float inv = 1.0f / sqrt(var + eps);
        __global float* yr = Y + (int64_t)r * cols;
        for (int c = 0; c < cols; ++c) {
            if (op == 0) yr[c] = exp(xr[c] - m) / sum;
            else if (op == 1) yr[c] = (xr[c] / rms) * G[c];
            else yr[c] = (xr[c] - mean) * inv * G[c] + B[c];
        }
    }
)";
}

GpuComputeOpenCL::GpuComputeOpenCL() : lib_handle_(nullptr), context_(nullptr), command_queue_(nullptr) {}

GpuComputeOpenCL::~GpuComputeOpenCL() {}

bool GpuComputeOpenCL::init() {
#ifdef _WIN32
    lib_handle_ = load_lib("OpenCL.dll");
#else
    lib_handle_ = load_lib("libOpenCL.so");
#endif

    if (!lib_handle_) return false;

    clGetPlatformIDs_ = (decltype(clGetPlatformIDs_))get_sym(lib_handle_, "clGetPlatformIDs");
    clGetDeviceIDs_ = (decltype(clGetDeviceIDs_))get_sym(lib_handle_, "clGetDeviceIDs");
    clCreateContext_ = (decltype(clCreateContext_))get_sym(lib_handle_, "clCreateContext");
    clCreateCommandQueue_ = (decltype(clCreateCommandQueue_))get_sym(lib_handle_, "clCreateCommandQueue");
    clCreateProgramWithSource_ = (decltype(clCreateProgramWithSource_))get_sym(lib_handle_, "clCreateProgramWithSource");
    clBuildProgram_ = (decltype(clBuildProgram_))get_sym(lib_handle_, "clBuildProgram");
    clCreateKernel_ = (decltype(clCreateKernel_))get_sym(lib_handle_, "clCreateKernel");
    clSetKernelArg_ = (decltype(clSetKernelArg_))get_sym(lib_handle_, "clSetKernelArg");
    clEnqueueNDRangeKernel_ = (decltype(clEnqueueNDRangeKernel_))get_sym(lib_handle_, "clEnqueueNDRangeKernel");
    clFinish_ = (decltype(clFinish_))get_sym(lib_handle_, "clFinish");
    clReleaseKernel_ = (decltype(clReleaseKernel_))get_sym(lib_handle_, "clReleaseKernel");
    clReleaseProgram_ = (decltype(clReleaseProgram_))get_sym(lib_handle_, "clReleaseProgram");
    clCreateBuffer_ = (decltype(clCreateBuffer_))get_sym(lib_handle_, "clCreateBuffer");
    clEnqueueWriteBuffer_ = (decltype(clEnqueueWriteBuffer_))get_sym(lib_handle_, "clEnqueueWriteBuffer");
    clEnqueueReadBuffer_ = (decltype(clEnqueueReadBuffer_))get_sym(lib_handle_, "clEnqueueReadBuffer");
    clReleaseMemObject_ = (decltype(clReleaseMemObject_))get_sym(lib_handle_, "clReleaseMemObject");

    if (!clGetPlatformIDs_ || !clGetDeviceIDs_ || !clCreateContext_) return false;

    uint32_t num_platforms = 0;
    void* platform = nullptr;
    if (clGetPlatformIDs_(1, &platform, &num_platforms) != 0 || num_platforms == 0) return false;

    uint32_t num_devices = 0;
    void* device = nullptr;
    if (clGetDeviceIDs_(platform, 0xFFFFFFFF, 1, &device, &num_devices) != 0 || num_devices == 0) return false;

    int err = 0;
    context_ = clCreateContext_(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != 0 || !context_) return false;

    command_queue_ = clCreateCommandQueue_(context_, device, 0, &err);
    return (err == 0 && command_queue_ != nullptr);
}

void* GpuComputeOpenCL::alloc(size_t size) {
    if (clCreateBuffer_ && context_) {
        int err = 0;
        return clCreateBuffer_(context_, 4, size, nullptr, &err); // CL_MEM_READ_WRITE = 4
    }
    return new uint8_t[size];
}

void GpuComputeOpenCL::free(void* ptr) {
    if (clReleaseMemObject_ && context_) {
        clReleaseMemObject_(ptr);
    } else {
        delete[] static_cast<uint8_t*>(ptr);
    }
}

void GpuComputeOpenCL::copy_to_device(void* dst, const void* src, size_t size) {
    if (clEnqueueWriteBuffer_ && command_queue_) {
        clEnqueueWriteBuffer_(command_queue_, dst, 1, 0, size, src, 0, nullptr, nullptr);
    } else {
        std::memcpy(dst, src, size);
    }
}

void GpuComputeOpenCL::copy_to_host(void* dst, const void* src, size_t size) {
    if (clEnqueueReadBuffer_ && command_queue_) {
        // (queue, device-buffer=src, blocking, offset, size, host-ptr=dst).
        // Was swapped (host ptr passed as cl_mem) — segfaulted on any live
        // OpenCL platform at the first device->host readback.
        clEnqueueReadBuffer_(command_queue_, const_cast<void*>(src), 1, 0, size, dst, 0, nullptr, nullptr);
    } else {
        std::memcpy(dst, src, size);
    }
}

void GpuComputeOpenCL::launch_gemm(int m, int n, int k, const float* a, const float* b, float* c) {
    // REAL-ONLY: every device-launch prerequisite must be present and every
    // step checked. The old code enqueued the kernel WITHOUT setting any
    // kernel args (clSetKernelArg was never even loaded) and read back
    // before completion — segfaulting on any live OpenCL platform — with a
    // silent CPU fallback masking unavailable plumbing as GPU results.
    // Incomplete plumbing now throws fail-loud instead.
    if (!clCreateProgramWithSource_ || !clBuildProgram_ || !clCreateKernel_ ||
        !clSetKernelArg_ || !clEnqueueNDRangeKernel_ || !clFinish_ ||
        !clReleaseKernel_ || !clReleaseProgram_ || !clCreateBuffer_ ||
        !clEnqueueWriteBuffer_ || !clEnqueueReadBuffer_ || !clReleaseMemObject_ ||
        !context_ || !command_queue_) {
        throw std::runtime_error(
            "[REAL-ONLY] GPU_OPENCL::launch_gemm unavailable: incomplete OpenCL "
            "launch plumbing on this host (no silent CPU fallback)");
    }

    int err = 0;
    const char* src = kOpenCLGEMMKernel;
    size_t length = std::strlen(src);
    void* program = clCreateProgramWithSource_(context_, 1, &src, &length, &err);
    if (err != 0 || !program)
        throw std::runtime_error("[REAL-ONLY] GPU_OPENCL::launch_gemm: clCreateProgramWithSource failed");
    if (clBuildProgram_(program, 0, nullptr, nullptr, nullptr, nullptr) != 0) {
        clReleaseProgram_(program);
        throw std::runtime_error("[REAL-ONLY] GPU_OPENCL::launch_gemm: OpenCL program build failed");
    }
    void* kernel = clCreateKernel_(program, "gemm_kernel", &err);
    if (err != 0 || !kernel) {
        clReleaseProgram_(program);
        throw std::runtime_error("[REAL-ONLY] GPU_OPENCL::launch_gemm: clCreateKernel failed");
    }

    int arg_err = 0;
    void* d_a = nullptr;
    void* d_b = nullptr;
    void* d_c = nullptr;
    try {
        d_a = alloc(m * k * sizeof(float));
        d_b = alloc(k * n * sizeof(float));
        d_c = alloc(m * n * sizeof(float));
        if (!d_a || !d_b || !d_c) throw std::runtime_error("clCreateBuffer failed");

        copy_to_device(d_a, a, m * k * sizeof(float));
        copy_to_device(d_b, b, k * n * sizeof(float));

        arg_err |= clSetKernelArg_(kernel, 0, sizeof(int), &m);
        arg_err |= clSetKernelArg_(kernel, 1, sizeof(int), &n);
        arg_err |= clSetKernelArg_(kernel, 2, sizeof(int), &k);
        arg_err |= clSetKernelArg_(kernel, 3, sizeof(void*), &d_a);
        arg_err |= clSetKernelArg_(kernel, 4, sizeof(void*), &d_b);
        arg_err |= clSetKernelArg_(kernel, 5, sizeof(void*), &d_c);
        if (arg_err != 0) throw std::runtime_error("clSetKernelArg failed");

        size_t global_work_size[2] = {(size_t)m, (size_t)n};
        if (clEnqueueNDRangeKernel_(command_queue_, kernel, 2, nullptr,
                                    global_work_size, nullptr, 0, nullptr, nullptr) != 0)
            throw std::runtime_error("clEnqueueNDRangeKernel failed");
        // Must complete before the blocking read below (old code raced).
        if (clFinish_(command_queue_) != 0)
            throw std::runtime_error("clFinish failed");

        copy_to_host(c, d_c, m * n * sizeof(float));
    } catch (...) {
        if (d_a) free(d_a);
        if (d_b) free(d_b);
        if (d_c) free(d_c);
        clReleaseKernel_(kernel);
        clReleaseProgram_(program);
        throw;
    }
    free(d_a);
    free(d_b);
    free(d_c);
    clReleaseKernel_(kernel);
    clReleaseProgram_(program);
}

void GpuComputeOpenCL::launch_gemv(int m, int n, const float* a, const float* x, float* y) {
    if (!clCreateProgramWithSource_ || !clBuildProgram_ || !clCreateKernel_ ||
        !clSetKernelArg_ || !clEnqueueNDRangeKernel_ || !clFinish_ ||
        !clReleaseKernel_ || !clReleaseProgram_ || !clCreateBuffer_ ||
        !clEnqueueWriteBuffer_ || !clEnqueueReadBuffer_ || !clReleaseMemObject_ ||
        !context_ || !command_queue_) {
        throw std::runtime_error("[REAL-ONLY] GPU_OPENCL::launch_gemv unavailable: incomplete OpenCL launch plumbing");
    }
    int err = 0;
    const char* src = kOpenCLGEMVKernel;
    size_t length = std::strlen(src);
    void* program = clCreateProgramWithSource_(context_, 1, &src, &length, &err);
    if (err != 0 || !program) throw std::runtime_error("[REAL-ONLY] GPU_OPENCL::launch_gemv: program failed");
    if (clBuildProgram_(program, 0, nullptr, nullptr, nullptr, nullptr) != 0) {
        clReleaseProgram_(program);
        throw std::runtime_error("[REAL-ONLY] GPU_OPENCL::launch_gemv: build failed");
    }
    void* kernel = clCreateKernel_(program, "gemv_kernel", &err);
    if (err != 0 || !kernel) {
        clReleaseProgram_(program);
        throw std::runtime_error("[REAL-ONLY] GPU_OPENCL::launch_gemv: kernel failed");
    }
    void* d_a = nullptr;
    void* d_x = nullptr;
    void* d_y = nullptr;
    try {
        d_a = alloc(m * n * sizeof(float));
        d_x = alloc(n * sizeof(float));
        d_y = alloc(m * sizeof(float));
        if (!d_a || !d_x || !d_y) throw std::runtime_error("clCreateBuffer failed");
        copy_to_device(d_a, a, m * n * sizeof(float));
        copy_to_device(d_x, x, n * sizeof(float));
        int ae = 0;
        ae |= clSetKernelArg_(kernel, 0, sizeof(int), &m);
        ae |= clSetKernelArg_(kernel, 1, sizeof(int), &n);
        ae |= clSetKernelArg_(kernel, 2, sizeof(void*), &d_a);
        ae |= clSetKernelArg_(kernel, 3, sizeof(void*), &d_x);
        ae |= clSetKernelArg_(kernel, 4, sizeof(void*), &d_y);
        if (ae != 0) throw std::runtime_error("clSetKernelArg failed");
        size_t gws[1] = {(size_t)m};
        if (clEnqueueNDRangeKernel_(command_queue_, kernel, 1, nullptr, gws, nullptr, 0, nullptr, nullptr) != 0)
            throw std::runtime_error("clEnqueueNDRangeKernel failed");
        if (clFinish_(command_queue_) != 0) throw std::runtime_error("clFinish failed");
        copy_to_host(y, d_y, m * sizeof(float));
    } catch (...) {
        if (d_a) free(d_a);
        if (d_x) free(d_x);
        if (d_y) free(d_y);
        clReleaseKernel_(kernel);
        clReleaseProgram_(program);
        throw;
    }
    free(d_a);
    free(d_x);
    free(d_y);
    clReleaseKernel_(kernel);
    clReleaseProgram_(program);
}

void GpuComputeOpenCL::run_elt_impl(const char* src, const char* name, int op, float s, int n,
                                    const float* a, const float* b, float* c) {
    if (!clCreateProgramWithSource_ || !clBuildProgram_ || !clCreateKernel_ ||
        !clSetKernelArg_ || !clEnqueueNDRangeKernel_ || !clFinish_ ||
        !clReleaseKernel_ || !clReleaseProgram_ || !clCreateBuffer_ ||
        !clEnqueueWriteBuffer_ || !clEnqueueReadBuffer_ || !clReleaseMemObject_ ||
        !context_ || !command_queue_) {
        throw std::runtime_error("[REAL-ONLY] GPU_OPENCL elt unavailable: incomplete launch plumbing");
    }
    int err = 0;
    size_t length = std::strlen(src);
    void* program = clCreateProgramWithSource_(context_, 1, &src, &length, &err);
    if (err != 0 || !program) throw std::runtime_error("[REAL-ONLY] GPU_OPENCL elt: program failed");
    if (clBuildProgram_(program, 0, nullptr, nullptr, nullptr, nullptr) != 0) {
        clReleaseProgram_(program);
        throw std::runtime_error("[REAL-ONLY] GPU_OPENCL elt: build failed");
    }
    void* kernel = clCreateKernel_(program, name, &err);
    if (err != 0 || !kernel) {
        clReleaseProgram_(program);
        throw std::runtime_error("[REAL-ONLY] GPU_OPENCL elt: kernel failed");
    }
    void* d_a = nullptr;
    void* d_b = nullptr;
    void* d_c = nullptr;
    static const float kZero = 0.0f;
    try {
        d_a = alloc(n * sizeof(float));
        d_b = alloc(n * sizeof(float));
        d_c = alloc(n * sizeof(float));
        if (!d_a || !d_b || !d_c) throw std::runtime_error("clCreateBuffer failed");
        copy_to_device(d_a, a, n * sizeof(float));
        copy_to_device(d_b, b ? b : &kZero, b ? n * sizeof(float) : sizeof(float));
        int ae = 0;
        ae |= clSetKernelArg_(kernel, 0, sizeof(int), &op);
        ae |= clSetKernelArg_(kernel, 1, sizeof(float), &s);
        ae |= clSetKernelArg_(kernel, 2, sizeof(int), &n);
        ae |= clSetKernelArg_(kernel, 3, sizeof(void*), &d_a);
        ae |= clSetKernelArg_(kernel, 4, sizeof(void*), &d_b);
        ae |= clSetKernelArg_(kernel, 5, sizeof(void*), &d_c);
        if (ae != 0) throw std::runtime_error("clSetKernelArg failed");
        size_t gws[1] = {(size_t)n};
        if (clEnqueueNDRangeKernel_(command_queue_, kernel, 1, nullptr, gws, nullptr, 0, nullptr, nullptr) != 0)
            throw std::runtime_error("clEnqueueNDRangeKernel failed");
        if (clFinish_(command_queue_) != 0) throw std::runtime_error("clFinish failed");
        copy_to_host(c, d_c, n * sizeof(float));
    } catch (...) {
        if (d_a) free(d_a);
        if (d_b) free(d_b);
        if (d_c) free(d_c);
        clReleaseKernel_(kernel);
        clReleaseProgram_(program);
        throw;
    }
    free(d_a);
    free(d_b);
    free(d_c);
    clReleaseKernel_(kernel);
    clReleaseProgram_(program);
}

static void run_elt_kernel(GpuComputeOpenCL* self, const char* src, const char* name,
                           int op, float s, int n, const float* a, const float* b, float* c) {
    self->run_elt_impl(src, name, op, s, n, a, b, c);
}

void GpuComputeOpenCL::launch_relu(int n, const float* x, float* y) { run_elt_kernel(this, kOpenCLEltKernel, "elt_kernel", 0, 0.0f, n, x, nullptr, y); }
void GpuComputeOpenCL::launch_gelu(int n, const float* x, float* y) { run_elt_kernel(this, kOpenCLEltKernel, "elt_kernel", 1, 0.0f, n, x, nullptr, y); }
void GpuComputeOpenCL::launch_silu(int n, const float* x, float* y) { run_elt_kernel(this, kOpenCLEltKernel, "elt_kernel", 2, 0.0f, n, x, nullptr, y); }
void GpuComputeOpenCL::launch_add(int n, const float* a, const float* b, float* c) { run_elt_kernel(this, kOpenCLEltKernel, "elt_kernel", 3, 0.0f, n, a, b, c); }
void GpuComputeOpenCL::launch_mul(int n, const float* a, const float* b, float* c) { run_elt_kernel(this, kOpenCLEltKernel, "elt_kernel", 4, 0.0f, n, a, b, c); }
void GpuComputeOpenCL::launch_scale(int n, float s, const float* x, float* y) { run_elt_kernel(this, kOpenCLEltKernel, "elt_kernel", 5, s, n, x, nullptr, y); }

void GpuComputeOpenCL::launch_softmax(int rows, int cols, const float* x, float* y) {
    run_norm_impl(kOpenCLNormKernel, "norm_kernel", 0, 0.0f, rows, cols, x, nullptr, nullptr, y);
}

void GpuComputeOpenCL::launch_rms_norm(int rows, int cols, const float* x, const float* g, float eps, float* y) {
    run_norm_impl(kOpenCLNormKernel, "norm_kernel", 1, eps, rows, cols, x, g, nullptr, y);
}

void GpuComputeOpenCL::launch_layer_norm(int rows, int cols, const float* x, const float* g, const float* b, float eps, float* y) {
    run_norm_impl(kOpenCLNormKernel, "norm_kernel", 2, eps, rows, cols, x, g, b, y);
}

void GpuComputeOpenCL::run_norm_impl(const char* src, const char* name, int op, float eps,
                                     int rows, int cols, const float* x, const float* g,
                                     const float* b, float* y) {
    if (!clCreateProgramWithSource_ || !clBuildProgram_ || !clCreateKernel_ ||
        !clSetKernelArg_ || !clEnqueueNDRangeKernel_ || !clFinish_ ||
        !clReleaseKernel_ || !clReleaseProgram_ || !clCreateBuffer_ ||
        !clEnqueueWriteBuffer_ || !clEnqueueReadBuffer_ || !clReleaseMemObject_ ||
        !context_ || !command_queue_) {
        throw std::runtime_error("[REAL-ONLY] GPU_OPENCL norm unavailable: incomplete launch plumbing");
    }
    int err = 0;
    size_t length = std::strlen(src);
    void* program = clCreateProgramWithSource_(context_, 1, &src, &length, &err);
    if (err != 0 || !program) throw std::runtime_error("[REAL-ONLY] GPU_OPENCL norm: program failed");
    if (clBuildProgram_(program, 0, nullptr, nullptr, nullptr, nullptr) != 0) {
        clReleaseProgram_(program);
        throw std::runtime_error("[REAL-ONLY] GPU_OPENCL norm: build failed");
    }
    void* kernel = clCreateKernel_(program, name, &err);
    if (err != 0 || !kernel) {
        clReleaseProgram_(program);
        throw std::runtime_error("[REAL-ONLY] GPU_OPENCL norm: kernel failed");
    }
    void* d_x = nullptr;
    void* d_g = nullptr;
    void* d_b = nullptr;
    void* d_y = nullptr;
    static const float kZero = 0.0f;
    static const float kOne = 1.0f;
    try {
        d_x = alloc(rows * cols * sizeof(float));
        d_g = alloc(cols * sizeof(float));
        d_b = alloc(cols * sizeof(float));
        d_y = alloc(rows * cols * sizeof(float));
        if (!d_x || !d_g || !d_b || !d_y) throw std::runtime_error("clCreateBuffer failed");
        copy_to_device(d_x, x, rows * cols * sizeof(float));
        copy_to_device(d_g, g ? g : &kOne, g ? cols * sizeof(float) : sizeof(float));
        copy_to_device(d_b, b ? b : &kZero, b ? cols * sizeof(float) : sizeof(float));
        int ae = 0;
        ae |= clSetKernelArg_(kernel, 0, sizeof(int), &op);
        ae |= clSetKernelArg_(kernel, 1, sizeof(float), &eps);
        ae |= clSetKernelArg_(kernel, 2, sizeof(int), &rows);
        ae |= clSetKernelArg_(kernel, 3, sizeof(int), &cols);
        ae |= clSetKernelArg_(kernel, 4, sizeof(void*), &d_x);
        ae |= clSetKernelArg_(kernel, 5, sizeof(void*), &d_g);
        ae |= clSetKernelArg_(kernel, 6, sizeof(void*), &d_b);
        ae |= clSetKernelArg_(kernel, 7, sizeof(void*), &d_y);
        if (ae != 0) throw std::runtime_error("clSetKernelArg failed");
        size_t gws[1] = {(size_t)rows};
        if (clEnqueueNDRangeKernel_(command_queue_, kernel, 1, nullptr, gws, nullptr, 0, nullptr, nullptr) != 0)
            throw std::runtime_error("clEnqueueNDRangeKernel failed");
        if (clFinish_(command_queue_) != 0) throw std::runtime_error("clFinish failed");
        copy_to_host(y, d_y, rows * cols * sizeof(float));
    } catch (...) {
        if (d_x) free(d_x);
        if (d_g) free(d_g);
        if (d_b) free(d_b);
        if (d_y) free(d_y);
        clReleaseKernel_(kernel);
        clReleaseProgram_(program);
        throw;
    }
    free(d_x);
    free(d_g);
    free(d_b);
    free(d_y);
    clReleaseKernel_(kernel);
    clReleaseProgram_(program);
}

} // namespace quant
