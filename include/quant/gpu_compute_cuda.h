#pragma once
#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/backend.h"

namespace quant {
namespace gpu {

class GPUComputeCuda {
public:
    GPUComputeCuda();
    ~GPUComputeCuda();

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

    void rope(const void* x, void* q, int64_t seq_len, int64_t head_dim);
    void attention(const void* q, const void* k, const void* v, void* o, int64_t seq_len, int64_t head_dim);
    void reduce_sum(const void* x, void* result, int64_t n);
    void reduce_max(const void* x, void* result, int64_t n);
    void moe_gather(const void* expert_outputs, const void* routing_indices, void* output, int64_t num_tokens, int64_t expert_dim);
    void moe_scatter(const void* input, const void* routing_indices, void* expert_inputs, int64_t num_tokens, int64_t token_dim);

    // MoE page-locking and async-stream upload/download primitives.
    // No speedup figure is claimed here: consult bench/ for measured numbers.
    bool register_host_memory(void* ptr, size_t bytes);
    bool unregister_host_memory(void* ptr);
    void* create_stream();
    void destroy_stream(void* stream);
    void synchronize_stream(void* stream);
    void async_upload(const void* src, void* dst, size_t bytes, void* stream = nullptr);
    void async_download(const void* src, void* dst, size_t bytes, void* stream = nullptr);

    void synchronize();
    int64_t memory_free() const;
    int64_t memory_total() const;

private:
    struct Impl;
    Impl* impl_;
};

GPUComputeCuda& get_cuda_compute();

} // namespace gpu
} // namespace quant
