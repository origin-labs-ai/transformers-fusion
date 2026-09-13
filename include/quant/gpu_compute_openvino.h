#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace quant {
namespace gpu {

class GPUComputeOpenVINO {
public:
    GPUComputeOpenVINO() = default;
    ~GPUComputeOpenVINO();

    bool init(int device_id = 0);
    bool is_initialized() const { return initialized_; }

    void* alloc(size_t size);
    void free_buf(void* ptr);

    void upload(const void* src, void* dst, size_t size);
    void download(const void* src, void* dst, size_t size);

    void gemm(float alpha, const void* A, const void* B, float beta, void* C, int64_t M, int64_t N, int64_t K);

    // Host reference kernels (REAL CPU, AVX2+scalar, cache-tuned blocking):
    // verification baseline for CI without NPU hardware. NEVER claimed as
    // OpenVINO execution — NPU_OPENVINO compute fails loud via backend.
    // gemm() delegates to reference_gemm.
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

    int64_t memory_free() const;
    int64_t memory_total() const;
    void synchronize();

private:
    bool initialized_ = false;
    void* handle_ = nullptr;
};

GPUComputeOpenVINO& get_openvino_compute();

} // namespace gpu
} // namespace quant
