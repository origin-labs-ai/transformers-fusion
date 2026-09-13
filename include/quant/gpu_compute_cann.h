#pragma once
#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/backend.h"

namespace quant {
namespace gpu {

class GPUComputeCann {
public:
    GPUComputeCann();
    ~GPUComputeCann();

    bool init(int64_t device_id = 0);
    bool is_initialized() const;
    void shutdown();

    void* alloc(int64_t bytes);
    void free_buf(void* ptr);

    void upload(const Tensor& src, void* dst);
    void download(void* src, Tensor& dst);

    void gemm(float alpha, const void* A, const void* B, float beta, void* C, int64_t M, int64_t N, int64_t K);
    void gemv(float alpha, const void* A, const void* x, float beta, void* y, int64_t M, int64_t N);
    
    void relu(const void* x, void* y, int64_t n);
    void gelu(const void* x, void* y, int64_t n);
    void silu(const void* x, void* y, int64_t n);
    void softmax(const void* x, void* y, int64_t rows, int64_t cols);
    void rms_norm(const void* x, const void* weight, void* y, int64_t rows, int64_t cols, float eps);
    void layer_norm(const void* x, const void* gamma, const void* beta, void* y, int64_t rows, int64_t cols, float eps);
    
    void add(const void* a, const void* b, void* c, int64_t n);
    void mul(const void* a, const void* b, void* c, int64_t n);
    void scale(float s, const void* x, void* y, int64_t n);
    void fill(float val, void* x, int64_t n);
    void copy_buf(const void* src, void* dst, int64_t n);

    // Host reference kernels (REAL CPU, AVX2+scalar, cube-tuned blocking):
    // verification baseline for CI without Ascend hardware. NEVER claimed
    // as CANN execution — GPU_CANN compute fails loud (null descriptors).
    void reference_gemm(float alpha, const float* a, const float* b, float beta,
                        float* c, int64_t M, int64_t N, int64_t K);
    void reference_gemv(float alpha, const float* A, const float* x, float beta,
                        float* y, int64_t M, int64_t N);
    void reference_relu(const float* x, float* y, int64_t n);
    void reference_gelu(const float* x, float* y, int64_t n);
    void reference_silu(const float* x, float* y, int64_t n);
    void reference_add(const float* a, const float* b, float* c, int64_t n);
    void reference_mul(const float* a, const float* b, float* c, int64_t n);
    void reference_scale(float s, const float* x, float* y, int64_t n);
    void reference_softmax(const float* x, float* y, int64_t rows, int64_t cols);
    void reference_rms_norm(const float* x, const float* weight, float* y,
                            float eps, int64_t rows, int64_t cols);
    void reference_layer_norm(const float* x, const float* gamma, const float* beta,
                              float* y, float eps, int64_t rows, int64_t cols);

    void synchronize();
    int64_t memory_free() const;
    int64_t memory_total() const;

private:
    struct Impl;
    Impl* impl_;
};

GPUComputeCann& get_cann_compute();

} // namespace gpu
} // namespace quant
