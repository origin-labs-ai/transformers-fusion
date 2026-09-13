#include "quant/ddp.h"
#include "quant/random.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <thread>

namespace quant {

DistributedDataParallel::DistributedDataParallel(Model* model, int world_size,
                                                   int world_rank, DDPMode mode)
    : ctx_(world_size, world_rank, DistributedContext::Mode::DDP),
      model_(model), mode_(mode) {
    collect_params();
    build_buckets();

    if (mode_ == DDPMode::ASYNC) {
        async_running_ = true;
        async_thread_ = std::thread(&DistributedDataParallel::async_sync_loop, this);
    }
}

DistributedDataParallel::~DistributedDataParallel() {
    if (async_running_.exchange(false)) {
        if (async_thread_.joinable())
            async_thread_.join();
    }
}

void DistributedDataParallel::collect_params() {
    all_params_.clear();
    DenseModel* dm = dynamic_cast<DenseModel*>(model_);
    if (!dm) return;

    all_params_.push_back(&dm->tok_embeddings->weight);
    for (auto& layer : dm->layers) {
        all_params_.push_back(&layer->attention_norm.weight);
        all_params_.push_back(&layer->attention.q_proj.weight);
        all_params_.push_back(&layer->attention.q_proj.bias);
        all_params_.push_back(&layer->attention.k_proj.weight);
        all_params_.push_back(&layer->attention.k_proj.bias);
        all_params_.push_back(&layer->attention.v_proj.weight);
        all_params_.push_back(&layer->attention.v_proj.bias);
        all_params_.push_back(&layer->attention.o_proj.weight);
        all_params_.push_back(&layer->attention.o_proj.bias);
        all_params_.push_back(&layer->ffn_norm.weight);
        all_params_.push_back(&layer->ffn.gate_proj.weight);
        all_params_.push_back(&layer->ffn.gate_proj.bias);
        all_params_.push_back(&layer->ffn.up_proj.weight);
        all_params_.push_back(&layer->ffn.up_proj.bias);
        all_params_.push_back(&layer->ffn.down_proj.weight);
        all_params_.push_back(&layer->ffn.down_proj.bias);
    }
    all_params_.push_back(&dm->norm->weight);
    all_params_.push_back(&dm->lm_head->weight);
    all_params_.push_back(&dm->lm_head->bias);
}

void DistributedDataParallel::build_buckets() {
    buckets_.clear();
    Bucket current;
    int64_t current_numel = 0;

    for (auto* p : all_params_) {
        int64_t p_numel = p->numel();
        if (current_numel + p_numel > bucket_numel_ && current_numel > 0) {
            current.size = current_numel;
            current.buffer = Tensor({current_numel});
            buckets_.push_back(std::move(current));
            current = Bucket();
            current_numel = 0;
        }
        current.params.push_back(p);
        current_numel += p_numel;
    }

    if (current_numel > 0) {
        current.size = current_numel;
        current.buffer = Tensor({current_numel});
        buckets_.push_back(std::move(current));
    }
}

void DistributedDataParallel::set_grad_compression(GradCompression comp,
                                                    float sparse_ratio) {
    compression_ = comp;
    sparse_ratio_ = sparse_ratio;
}

void DistributedDataParallel::set_bucket_size(int64_t bucket_numel) {
    bucket_numel_ = bucket_numel;
    build_buckets();
}

Tensor DistributedDataParallel::forward(const Tensor& input,
                                         const Tensor& positions) {
    return model_->forward(input, positions);
}

void DistributedDataParallel::sync_gradients() {
    if (ctx_.world_size() <= 1) return;

    if (mode_ == DDPMode::ASYNC) {
        std::lock_guard<std::mutex> lock(async_mutex_);
    }

    for (auto& bucket : buckets_) {
        allreduce_bucket(bucket);
    }
}

void DistributedDataParallel::allreduce_bucket(Bucket& bucket) {
    if (bucket.params.empty()) return;

    // Pack grads into bucket buffer
    float* buf = bucket.buffer.data<float>();
    int64_t offset = 0;
    for (auto* p : bucket.params) {
        if (p->has_grad()) {
            int64_t n = p->grad().numel();
            std::memcpy(&buf[offset], p->grad().data<float>(), n * sizeof(float));
            offset += n;
        } else {
            int64_t n = p->numel();
            std::memset(&buf[offset], 0, n * sizeof(float));
            offset += n;
        }
    }

    // Apply compression if enabled
    if (compression_ == GradCompression::TOP_K_SPARSE && bucket.numel > 0) {
        int64_t k = std::max((int64_t)1, (int64_t)(bucket.numel * sparse_ratio_));
        Tensor values({k});
        Tensor indices({k});
        compress_topk_sparse(bucket.buffer, values, indices, k);
        ctx_.all_reduce(values);
        decompress_topk_sparse(bucket.buffer, values, indices, bucket.numel);
    } else if (compression_ == GradCompression::FP8_QUANT) {
        std::vector<uint8_t> compressed(bucket.numel);
        float scale = 1.0f;
        compress_fp8(bucket.buffer, compressed, scale);
        Tensor scale_t({1});
        scale_t.data<float>()[0] = scale;
        ctx_.all_reduce(scale_t);
        // Broadcast compressed data (in shared memory, just sum)
        for (int64_t i = 0; i < bucket.numel; i++)
            buf[i] += (float)((int8_t)compressed[i]) * scale;
        // Finalize: average
        for (int64_t i = 0; i < bucket.numel; i++)
            buf[i] /= (float)ctx_.world_size();
    } else {
        // Standard allreduce
        ctx_.all_reduce(buf, bucket.numel);
    }

    // Unpack grads back from bucket buffer
    offset = 0;
    for (auto* p : bucket.params) {
        int64_t n = p->numel();
        if (p->has_grad()) {
            float* g = p->grad().data<float>();
            for (int64_t i = 0; i < n; i++)
                g[i] = buf[offset + i];
        } else {
            p->set_grad(Tensor(p->shape()));
            float* g = p->grad().data<float>();
            for (int64_t i = 0; i < n; i++)
                g[i] = buf[offset + i];
        }
        offset += n;
    }
}

void DistributedDataParallel::compress_topk_sparse(Tensor& grad, Tensor& values,
                                                    Tensor& indices, int64_t k) {
    int64_t n = grad.numel();
    float* g = grad.data<float>();
    float* v = values.data<float>();
    int64_t* idx = indices.data<int64_t>();

    if (k >= n) {
        for (int64_t i = 0; i < n; i++) { v[i] = g[i]; idx[i] = i; }
        return;
    }

    std::vector<std::pair<float, int64_t>> sorted(n);
    for (int64_t i = 0; i < n; i++)
        sorted[i] = {std::abs(g[i]), i};
    std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    for (int64_t i = 0; i < k; i++) {
        v[i] = g[sorted[i].second];
        idx[i] = sorted[i].second;
    }

    // Zero out non-top-k gradients
    for (int64_t i = k; i < n; i++)
        g[sorted[i].second] = 0.0f;
}

void DistributedDataParallel::decompress_topk_sparse(Tensor& grad,
                                                      const Tensor& values,
                                                      const Tensor& indices,
                                                      int64_t n) {
    const float* v = values.data<float>();
    const int64_t* idx = indices.data<int64_t>();
    int64_t k = values.numel();

    grad.zero_();
    float* g = grad.data<float>();
    for (int64_t i = 0; i < std::min(k, n); i++)
        g[idx[i]] = v[i];
}

void DistributedDataParallel::compress_fp8(const Tensor& src,
                                            std::vector<uint8_t>& dst,
                                            float& scale) {
    int64_t n = src.numel();
    const float* s = src.data<float>();

    float max_abs = 0.0f;
    for (int64_t i = 0; i < n; i++)
        max_abs = std::max(max_abs, std::abs(s[i]));

    if (max_abs < 1e-10f) {
        scale = 1.0f;
        std::memset(dst.data(), 0, n);
        return;
    }

    scale = max_abs / 127.0f;
    float inv_scale = 1.0f / scale;

    for (int64_t i = 0; i < n; i++) {
        int8_t q = (int8_t)std::round(s[i] * inv_scale);
        q = std::max((int8_t)-127, std::min((int8_t)127, q));
        dst[i] = (uint8_t)(uint8_t)(int8_t)q;
    }
}

void DistributedDataParallel::decompress_fp8(std::vector<uint8_t>& src,
                                              float scale, Tensor& dst) {
    int64_t n = dst.numel();
    float* d = dst.data<float>();
    for (int64_t i = 0; i < n; i++)
        d[i] = (float)(int8_t)src[i] * scale;
}

void DistributedDataParallel::sync_buffers(const std::vector<Tensor*>& buffers) {
    if (ctx_.world_size() <= 1) return;
    for (auto* buf : buffers) {
        ctx_.all_reduce(*buf);
        float* d = buf->data<float>();
        int64_t n = buf->numel();
        for (int64_t i = 0; i < n; i++)
            d[i] /= (float)ctx_.world_size();
    }
}

void DistributedDataParallel::init_grad_accum_buffers() {
    grad_accum_buffers_.clear();
    for (auto* p : all_params_) {
        grad_accum_buffers_.push_back(Tensor(p->shape()));
        grad_accum_buffers_.back().zero_();
    }
}

void DistributedDataParallel::zero_grad() {
    for (auto* p : all_params_) {
        if (p->has_grad())
            p->zero_grad();
    }
    for (auto& buf : grad_accum_buffers_)
        buf.zero_();
    grad_accum_count_ = 0;
}

float DistributedDataParallel::global_grad_norm() {
    float total = 0.0f;
    for (auto* p : all_params_) {
        if (p->has_grad()) {
            const float* g = p->grad().data<float>();
            int64_t n = p->grad().numel();
            double sum = 0.0;
            for (int64_t i = 0; i < n; i++)
                sum += (double)g[i] * (double)g[i];
            total += (float)sum;
        }
    }
    if (ctx_.world_size() > 1) {
        float total_buf = total;
        ctx_.all_reduce(&total_buf, 1);
        total = total_buf;
    }
    return std::sqrt(total);
}

void DistributedDataParallel::clip_gradients(float max_norm) {
    float norm = global_grad_norm();
    if (norm <= max_norm || norm < 1e-8f) return;
    float scale = max_norm / norm;
    for (auto* p : all_params_) {
        if (p->has_grad()) {
            float* g = p->grad().data<float>();
            int64_t n = p->grad().numel();
            for (int64_t i = 0; i < n; i++)
                g[i] *= scale;
        }
    }
}

void DistributedDataParallel::average_model(float weight) {
    for (auto* p : all_params_) {
        if (p->numel() == 0) continue;
        float* d = p->data<float>();
        int64_t n = p->numel();
        for (int64_t i = 0; i < n; i++)
            d[i] *= weight;
    }
}

void DistributedDataParallel::broadcast_model(int src_rank) {
    if (ctx_.world_size() <= 1) return;
    for (auto* p : all_params_) {
        if (p->numel() == 0) continue;
        ctx_.broadcast(*p, src_rank);
    }
}

void DistributedDataParallel::inject_gradient_noise(int step, float eta, float gamma) {
    std::mt19937 rng(
        static_cast<std::mt19937::result_type>(
            make_seed(resolve_base_seed(0), SeedStream::DdpNoise, static_cast<uint64_t>(step))));
    float noise_var = eta / std::pow((float)(step + 1), gamma);
    if (noise_var < 1e-10f) return;
    float noise_std = std::sqrt(noise_var);
    for (auto* p : all_params_) {
        if (p->has_grad()) {
            float* g = p->grad().data<float>();
            int64_t n = p->grad().numel();
            for (int64_t i = 0; i < n; i++)
                g[i] += noise_std * (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) - 0.5f) * 2.0f;
        }
    }
}

void DistributedDataParallel::async_sync_loop() {
    while (async_running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (!async_running_.load()) break;
        std::lock_guard<std::mutex> lock(async_mutex_);
        sync_gradients();
    }
}

} // namespace quant
