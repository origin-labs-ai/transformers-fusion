#pragma once
#include <cstdint>
#include <cstddef>

namespace quant {

class GpuComputeHexagon {
public:
    GpuComputeHexagon();
    ~GpuComputeHexagon();

    bool init();
    void* alloc(size_t size);
    void free(void* ptr);
    void copy_to_device(void* dst, const void* src, size_t size);
    void copy_to_host(void* dst, const void* src, size_t size);

    void launch_gemm(int m, int n, int k, const float* a, const float* b, float* c);

    // Host reference kernels (REAL CPU, AVX2+scalar, Hexagon-TCM-tuned
    // blocking): verification baseline for CI without DSP hardware. NEVER
    // claimed as HVX execution — DSP_HEXAGON compute fails loud via the
    // ComputeBackend wrapper. launch_gemm delegates here (alpha=1,beta=0).
    void reference_gemm(float alpha, const float* a, const float* b, float beta,
                        float* c, int m, int n, int k);
    void reference_gemv(float alpha, const float* A, const float* x, float beta,
                        float* y, int m, int n);
    void reference_relu(const float* x, float* y, size_t n);
    void reference_gelu(const float* x, float* y, size_t n);
    void reference_silu(const float* x, float* y, size_t n);
    void reference_add(const float* a, const float* b, float* c, size_t n);
    void reference_mul(const float* a, const float* b, float* c, size_t n);
    void reference_scale(float s, const float* x, float* y, size_t n);
    void reference_softmax(const float* x, float* y, int rows, int cols);
    void reference_rms_norm(const float* x, const float* gamma, float* y,
                            float eps, int rows, int cols);
    void reference_layer_norm(const float* x, const float* gamma, const float* beta,
                              float* y, float eps, int rows, int cols);

private:
    void* lib_handle_;
    
    // Function pointers
    int (*hexagon_nn_init_)(void);
    int (*hexagon_nn_prepare_)(void);
    int (*hexagon_nn_execute_)(void);
};

} // namespace quant
