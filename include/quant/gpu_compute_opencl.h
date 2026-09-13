#pragma once
#include <cstdint>
#include <cstddef>

namespace quant {

class GpuComputeOpenCL {
public:
    GpuComputeOpenCL();
    ~GpuComputeOpenCL();

    bool init();
    void* alloc(size_t size);
    void free(void* ptr);
    void copy_to_device(void* dst, const void* src, size_t size);
    void copy_to_host(void* dst, const void* src, size_t size);

    void launch_gemm(int m, int n, int k, const float* a, const float* b, float* c);

private:
    void* lib_handle_;
    
    // Function pointers
    int (*clGetPlatformIDs_)(unsigned int, void*, unsigned int*);
    int (*clGetDeviceIDs_)(void*, unsigned long long, unsigned int, void*, unsigned int*);
    void* (*clCreateContext_)(const void*, unsigned int, const void*, void*, void*, int*);
    void* (*clCreateCommandQueue_)(void*, void*, unsigned long long, int*);
    void* (*clCreateProgramWithSource_)(void*, unsigned int, const char**, const size_t*, int*);
    int (*clBuildProgram_)(void*, unsigned int, const void*, const char*, void*, void*);
    void* (*clCreateKernel_)(void*, const char*, int*);
    int (*clSetKernelArg_)(void*, unsigned int, size_t, const void*);
    int (*clEnqueueNDRangeKernel_)(void*, void*, unsigned int, const size_t*, const size_t*, const size_t*, unsigned int, const void*, void**);
    int (*clFinish_)(void*);
    int (*clReleaseKernel_)(void*);
    int (*clReleaseProgram_)(void*);
    void* (*clCreateBuffer_)(void*, unsigned long long, size_t, void*, int*);
    int (*clEnqueueWriteBuffer_)(void*, void*, unsigned int, size_t, size_t, const void*, unsigned int, const void*, void**);
    int (*clEnqueueReadBuffer_)(void*, void*, unsigned int, size_t, size_t, void*, unsigned int, const void*, void**);
    int (*clReleaseMemObject_)(void*);

    void* context_;
    void* command_queue_;
};

} // namespace quant
