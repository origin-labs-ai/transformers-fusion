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
__kernel void gemm_kernel(const int M, const int N, const int K,
                          __global const float* A,
                          __global const float* B,
                          __global float* C) {
        int row = get_global_id(0);
        int col = get_global_id(1);

        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
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

} // namespace quant
