#pragma once
#include "quant/tensor.h"
#include <vector>
#include <cstdint>

namespace quant {
namespace gpu {

enum class GPUType {
    DIRECTX12,
    VULKAN
};

struct GPUConfig {
    GPUType type = GPUType::DIRECTX12;
    int64_t device_id = 0;
    bool enable_fp16 = true;
    bool enable_tf32 = true;
    int64_t num_warps = 32;
    int64_t workgroup_size = 256;
};

class GPUBackendInterface {
public:
    virtual ~GPUBackendInterface() = default;
    virtual bool init(int64_t device_id = 0) = 0;
    virtual bool is_initialized() const = 0;
    virtual void shutdown() = 0;
    virtual void upload(const Tensor& src, void* dst) = 0;
    virtual void download(void* src, Tensor& dst) = 0;
    virtual void rms_norm(const void* x, const void* gamma, void* y,
                          float eps, int64_t n, int64_t d) = 0;
    virtual void layer_norm(const void* x, const void* gamma, const void* beta,
                            void* y, float eps, int64_t n, int64_t d) = 0;
    virtual void synchronize() = 0;
    virtual int64_t memory_free() const = 0;
    virtual int64_t memory_total() const = 0;
};

class DirectXCompute : public GPUBackendInterface {
public:
    DirectXCompute();
    ~DirectXCompute() override;

    bool init(int64_t device_id = 0) override;
    bool is_initialized() const override;
    void shutdown() override;

    void* allocate(size_t bytes);
    void free(void* ptr);
    void upload(const Tensor& src, void* dst) override;
    void download(void* src, Tensor& dst) override;
    void copy(void* dst, const void* src, size_t bytes);

    void gemm(float alpha, const void* A, const void* B, float beta, void* C,
              int64_t M, int64_t N, int64_t K);

    void gemv(float alpha, const void* A, const void* x, float beta, void* y,
              int64_t M, int64_t N);

    void relu(const void* x, void* y, int64_t n);
    void gelu(const void* x, void* y, int64_t n);
    void silu(const void* x, void* y, int64_t n);
    void add(const void* a, const void* b, void* c, int64_t n);
    void mul(const void* a, const void* b, void* c, int64_t n);
    void scale(float s, const void* x, void* y, int64_t n);

    void softmax(const void* x, void* y, int64_t rows, int64_t cols);
    void rms_norm(const void* x, const void* gamma, void* y, float eps, int64_t n, int64_t d) override;
    void layer_norm(const void* x, const void* gamma, const void* beta, void* y, float eps, int64_t n, int64_t d) override;

    void moe_gather(const void* x, const int64_t* indices, const float* weights,
                    void* out, int64_t T, int64_t K, int64_t D);
    void moe_scatter_add(void* out, const int64_t* indices, const float* weights,
                         const void* expert_out, int64_t T, int64_t K, int64_t D);

    void synchronize() override;
    int64_t memory_free() const override;
    int64_t memory_total() const override;

private:
    struct Impl;
    Impl* impl_;
};

// ========================================================================
// Vulkan compute backend (dynamically loaded at runtime)
// ========================================================================

class VulkanBackend {
public:
    VulkanBackend();
    ~VulkanBackend();

    bool init(int64_t device_id = 0);
    bool is_initialized() const;
    // True when the device is up AND the embedded compute shaders passed
    // structural validation. is_initialized() only speaks about the device.
    bool compute_ready() const;
    void shutdown();

    void* allocate(size_t bytes);
    void free(void* ptr);
    void upload(const Tensor& src, void* dst);
    void download(void* src, Tensor& dst);

    void gemm(float alpha, const void* A, const void* B, float beta, void* C,
              int64_t M, int64_t N, int64_t K);
    void gemv(float alpha, const void* A, const void* x, float beta, void* y,
              int64_t M, int64_t N);

    void relu(const void* x, void* y, int64_t n);
    void gelu(const void* x, void* y, int64_t n);
    void silu(const void* x, void* y, int64_t n);
    void add(const void* a, const void* b, void* c, int64_t n);
    void mul(const void* a, const void* b, void* c, int64_t n);
    void scale(float s, const void* x, void* y, int64_t n);

    void softmax(const void* x, void* y, int64_t rows, int64_t cols);
    void rms_norm(const void* x, const void* gamma, void* y, float eps,
                  int64_t n, int64_t d);
    void layer_norm(const void* x, const void* gamma, const void* beta, void* y,
                    float eps, int64_t n, int64_t d);

    void moe_gather(const void* x, const int64_t* indices, const float* weights,
                    void* out, int64_t T, int64_t K, int64_t D);
    void moe_scatter_add(void* out, const int64_t* indices, const float* weights,
                         const void* expert_out, int64_t T, int64_t K, int64_t D);

    void flash_attention(void* out, const void* Q, const void* K, const void* V,
                        int64_t B, int64_t H, int64_t N, int64_t D, bool causal);

    void pipeline_cache_clear();
    size_t pipeline_cache_size() const;

    void synchronize();
    int64_t memory_free() const;
    int64_t memory_total() const;

private:
    struct Impl;
    Impl* impl_;
};

// ========================================================================
// GPU autodetection and factory
// ========================================================================

GPUType detect_best_gpu();
DirectXCompute& get_dx_compute();
VulkanBackend& get_vulkan_backend();
bool gpu_available();

void init_gpu(GPUType type = GPUType::DIRECTX12, int64_t device = 0);
void shutdown_gpu();

// ========================================================================
// Tensor-level GPU kernel wrappers (dispatched to best available backend)
// ========================================================================

Tensor vk_flash_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                            int64_t B, int64_t H, int64_t N, int64_t D,
                            bool causal = true);

Tensor vk_cross_entropy(const Tensor& logits, const Tensor& targets);

Tensor vk_cross_entropy_grad(const Tensor& logits, const Tensor& targets);

Tensor vk_swiglu(const Tensor& gate, const Tensor& up);

Tensor vk_rms_norm(const Tensor& x, const Tensor& gamma, float eps = 1e-5f);

Tensor vk_rms_norm_add(const Tensor& x, const Tensor& residual,
                         const Tensor& gamma, float eps = 1e-5f);

void vk_rope(Tensor& Q, Tensor& K, const Tensor& cos_cache,
               const Tensor& sin_cache, int64_t seq_start);

std::pair<Tensor, Tensor> vk_topk_softmax(const Tensor& logits, int64_t k);

Tensor vk_gelu(const Tensor& x);

Tensor vk_softmax(const Tensor& x);

Tensor vk_layer_norm(const Tensor& x, const Tensor& gamma,
                       const Tensor& beta, float eps = 1e-5f);

} // namespace gpu
} // namespace quant
