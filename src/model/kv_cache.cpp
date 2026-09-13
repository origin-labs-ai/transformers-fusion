#include "quant/kv_cache.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <limits>
#include <stdexcept>
#ifndef _WIN32
#include <sys/stat.h>
#endif
#include <filesystem>

// D6 W6/W7: D-* diagnostics gated behind NDEBUG (release silence) AND runtime
// flag QUANT_PAGED_KV_DEBUG. Header ABI frozen to void, so bool load status
// lives in file-local helpers + resident checks below (load_from_disk→bool
// semantics without header change; void members delegate and callers verify
// state as return proxy).
namespace {
inline bool paged_kv_debug_enabled() {
#ifndef NDEBUG
    const char* f = std::getenv("QUANT_PAGED_KV_DEBUG");
    return f && f[0] != '\0' && f[0] != '0';
#else
    return false;
#endif
}
#define PAGED_KV_DLOG(msg) do { if (paged_kv_debug_enabled()) { std::cerr << msg; } } while (0)
// load_from_disk→bool helper: exact-size read; false keeps caller state intact.
inline bool read_exact_bytes(std::ifstream& ifs, char* dst, std::streamsize n) {
    if (n < 0) return false;
    if (n == 0) return static_cast<bool>(ifs);
    if (dst == nullptr) return false;
    ifs.read(dst, n);
    return static_cast<bool>(ifs) && ifs.gcount() == n;
}
// evict_to_disk helper: exact-size write; false keeps caller state intact.
inline bool write_exact_bytes(std::ofstream& ofs, const char* src, std::streamsize n) {
    if (n < 0) return false;
    if (n == 0) return static_cast<bool>(ofs);
    if (src == nullptr) return false;
    ofs.write(src, n);
    return static_cast<bool>(ofs);
}
// Resident-size gate before any memcpy (never OOB).
inline bool block_resident_ok(const std::vector<float>& k,
                              const std::vector<float>& v, int64_t need) {
    if (need <= 0) return false;
    if (k.data() == nullptr || v.data() == nullptr) return false;
    if ((int64_t)k.size() < need) return false;
    if ((int64_t)v.size() < need) return false;
    return true;
}
}  // namespace

namespace quant {

void KVCache::quantize_fp8_block(const float* src, uint8_t* dst, float* scale,
                                  int64_t n) {
    float max_abs = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float a = std::fabs(src[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-10f) {
        *scale = 1.0f;
        std::memset(dst, 0, (size_t)n);
        return;
    }
    *scale = max_abs / FP8_MAX;
    float inv_scale = 1.0f / *scale;
    for (int64_t i = 0; i < n; i++) {
        float q = src[i] * inv_scale;
        q = std::max(-FP8_MAX, std::min(FP8_MAX, q));
        dst[i] = (uint8_t)(int8_t)std::round(q);
    }
}

void KVCache::dequantize_fp8_block(const uint8_t* src, float scale,
                                     float* dst, int64_t n) {
    if (scale < 1e-10f) {
        std::memset(dst, 0, (size_t)n * sizeof(float));
        return;
    }
    for (int64_t i = 0; i < n; i++)
        dst[i] = (float)(int8_t)src[i] * scale;
}

KVCache::KVCache(KVCache&& other) noexcept
    : caches_(std::move(other.caches_)),
      num_layers_(other.num_layers_),
      max_seq_len_(other.max_seq_len_),
      num_heads_(other.num_heads_),
      head_dim_(other.head_dim_),
      quantized_(other.quantized_) {
    other.num_layers_ = 0;
    other.max_seq_len_ = 0;
    other.num_heads_ = 0;
    other.head_dim_ = 0;
    other.quantized_ = false;
}

KVCache& KVCache::operator=(KVCache&& other) noexcept {
    if (this != &other) {
        caches_ = std::move(other.caches_);
        num_layers_ = other.num_layers_;
        max_seq_len_ = other.max_seq_len_;
        num_heads_ = other.num_heads_;
        head_dim_ = other.head_dim_;
        quantized_ = other.quantized_;
        other.num_layers_ = 0;
        other.max_seq_len_ = 0;
        other.num_heads_ = 0;
        other.head_dim_ = 0;
        other.quantized_ = false;
    }
    return *this;
}

KVCache::KVCache(int num_layers, int64_t max_seq_len, int64_t num_heads,
                  int64_t head_dim, bool quantized)
    : num_layers_(num_layers), max_seq_len_(max_seq_len),
      num_heads_(num_heads), head_dim_(head_dim), quantized_(quantized)
{
    init(num_layers, max_seq_len, num_heads, head_dim, quantized);
}

void KVCache::init(int num_layers, int64_t max_seq_len, int64_t num_heads,
                    int64_t head_dim, bool quantized) {
    num_layers_ = num_layers;
    max_seq_len_ = max_seq_len;
    num_heads_ = num_heads;
    head_dim_ = head_dim;
    quantized_ = quantized;
    caches_.resize(num_layers);
    for (int i = 0; i < num_layers; i++) {
        auto& c = caches_[i];
        int64_t h = num_heads_;
        int64_t d = head_dim_;
        c.k = Tensor::zeros(Shape{1, h, max_seq_len_, d});
        c.v = Tensor::zeros(Shape{1, h, max_seq_len_, d});
        c.current_pos = 0;
        if (quantized_) {
            int64_t total_vals = h * max_seq_len_ * d;
            int64_t num_blocks = (total_vals + FP8_BLOCK_SIZE - 1) / FP8_BLOCK_SIZE;
            c.k_quant.resize((size_t)total_vals);
            c.v_quant.resize((size_t)total_vals);
            c.k_scales.resize((size_t)num_blocks);
            c.v_scales.resize((size_t)num_blocks);
        }
    }
}

void KVCache::append(int layer, const Tensor& k, const Tensor& v) {
    if (layer >= num_layers_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& c = caches_[layer];
    int64_t seq_len = k.shape().dims[2];
    int64_t pos = c.current_pos;
    if (pos >= max_seq_len_) return;
    seq_len = std::min(seq_len, max_seq_len_ - pos);
    int64_t h = num_heads_;
    int64_t d = head_dim_;

    if (quantized_) {
        int64_t total_vals = h * seq_len * d;
        int64_t num_blocks = (total_vals + FP8_BLOCK_SIZE - 1) / FP8_BLOCK_SIZE;
        std::vector<uint8_t> k_tmp((size_t)total_vals);
        std::vector<uint8_t> v_tmp((size_t)total_vals);
        std::vector<float> k_scales_tmp((size_t)num_blocks);
        std::vector<float> v_scales_tmp((size_t)num_blocks);

        const float* ksrc = k.data<float>();
        const float* vsrc = v.data<float>();

        for (int64_t b = 0; b < num_blocks; b++) {
            int64_t start = b * FP8_BLOCK_SIZE;
            int64_t end = std::min(start + FP8_BLOCK_SIZE, total_vals);
            int64_t n = end - start;
            quantize_fp8_block(ksrc + start, k_tmp.data() + start,
                                &k_scales_tmp[(size_t)b], n);
            quantize_fp8_block(vsrc + start, v_tmp.data() + start,
                                &v_scales_tmp[(size_t)b], n);
        }

        int64_t dst_off = (int64_t)c.current_pos * h * d;
        int64_t copy_n = total_vals;
        if (dst_off + copy_n > (int64_t)c.k_quant.size())
            copy_n = (int64_t)c.k_quant.size() - dst_off;
        if (copy_n <= 0) return;
        if (copy_n > 0) {
            std::memcpy(c.k_quant.data() + dst_off, k_tmp.data(), (size_t)copy_n);
            std::memcpy(c.v_quant.data() + dst_off, v_tmp.data(), (size_t)copy_n);
            int64_t scale_off = dst_off / FP8_BLOCK_SIZE;
            int64_t num_scales = (copy_n + FP8_BLOCK_SIZE - 1) / FP8_BLOCK_SIZE;
            std::memcpy(c.k_scales.data() + scale_off, k_scales_tmp.data(),
                        (size_t)num_scales * sizeof(float));
            std::memcpy(c.v_scales.data() + scale_off, v_scales_tmp.data(),
                        (size_t)num_scales * sizeof(float));
        }
    } else {
        float* kdst = (float*)c.k.data();
        float* vdst = (float*)c.v.data();
        const float* ksrc = (const float*)k.data();
        const float* vsrc = (const float*)v.data();

        for (int64_t s = 0; s < seq_len && pos + s < max_seq_len_; s++) {
            for (int64_t hh = 0; hh < h; hh++) {
                int64_t dst_offset = hh * max_seq_len_ * d + (pos + s) * d;
                int64_t src_offset = hh * seq_len * d + s * d;
                memcpy(kdst + dst_offset, ksrc + src_offset, (size_t)d * sizeof(float));
                memcpy(vdst + dst_offset, vsrc + src_offset, (size_t)d * sizeof(float));
            }
        }
    }
    c.current_pos += (int)std::min(seq_len, max_seq_len_ - pos);
}

std::pair<Tensor, Tensor> KVCache::get_range(int layer, int start, int end) const {
    if (layer >= num_layers_) return {};
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& c = caches_[layer];
    int64_t len = (end - start > max_seq_len_) ? max_seq_len_ : end - start;
    Tensor k_out(Shape{1, num_heads_, len, head_dim_});
    Tensor v_out(Shape{1, num_heads_, len, head_dim_});

    if (quantized_) {
        int64_t h = num_heads_;
        int64_t d = head_dim_;
        for (int64_t s = 0; s < len; s++) {
            int64_t src_pos = start + s;
            if (src_pos >= max_seq_len_) break;
            for (int64_t hh = 0; hh < h; hh++) {
                int64_t read_off = hh * max_seq_len_ * d + src_pos * d;
                int64_t write_off = hh * len * d + s * d;
                float* kdst = k_out.data<float>();
                float* vdst = v_out.data<float>();
                for (int64_t i = 0; i < d; i++) {
                    int64_t off = read_off + i;
                    int64_t bi = off / FP8_BLOCK_SIZE;
                    // NOTE (bug census): `bo = off % FP8_BLOCK_SIZE` was
                    // computed then (void)-discarded — dead remainder. The
                    // block-float scaling below uses per-block scales only;
                    // remainder kept out (no sub-block interpolation).
                    kdst[write_off + i] = (float)(int8_t)c.k_quant[(size_t)off] * c.k_scales[(size_t)bi];
                    vdst[write_off + i] = (float)(int8_t)c.v_quant[(size_t)off] * c.v_scales[(size_t)bi];
                }
            }
        }
    } else {
        const float* ksrc = (const float*)c.k.data();
        const float* vsrc = (const float*)c.v.data();
        float* kdst = (float*)k_out.data();
        float* vdst = (float*)v_out.data();

        for (int64_t s = 0; s < len; s++) {
            int64_t src_pos = start + s;
            if (src_pos >= max_seq_len_) break;
            for (int64_t hh = 0; hh < num_heads_; hh++) {
                int64_t src_off = hh * max_seq_len_ * head_dim_ + src_pos * head_dim_;
                int64_t dst_off = hh * len * head_dim_ + s * head_dim_;
                memcpy(kdst + dst_off, ksrc + src_off, (size_t)head_dim_ * sizeof(float));
                memcpy(vdst + dst_off, vsrc + src_off, (size_t)head_dim_ * sizeof(float));
            }
        }
    }
    return {k_out, v_out};
}

std::pair<Tensor, Tensor> KVCache::get_all(int layer) const {
    return get_range(layer, 0, caches_[layer].current_pos);
}

int KVCache::context_len() const {
    // BUGFIX (bug census): lock-free read of current_pos while append writes
    // under mutex_ (torn read). Accessors take mutex_ now.
    std::lock_guard<std::mutex> lock(mutex_);
    return caches_.empty() ? 0 : caches_[0].current_pos;
}

int KVCache::context_len(int layer) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return (layer >= 0 && layer < (int)caches_.size()) ? caches_[layer].current_pos : 0;
}

int KVCache::max_seq_len() const { return (int)max_seq_len_; }

size_t KVCache::size_bytes() const {
    // BUGFIX (bug census): lock-free vector walk while append mutates
    // (race/crash). Snapshot under mutex_.
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (auto& c : caches_) {
        if (quantized_) {
            total += c.k_quant.size() + c.v_quant.size();
            total += c.k_scales.size() * sizeof(float);
            total += c.v_scales.size() * sizeof(float);
        } else {
            total += c.k.size_bytes() + c.v.size_bytes();
        }
    }
    return total;
}

void KVCache::clear() {
    for (auto& c : caches_) {
        if (quantized_) {
            std::memset(c.k_quant.data(), 0, c.k_quant.size());
            std::memset(c.v_quant.data(), 0, c.v_quant.size());
            std::memset(c.k_scales.data(), 0, c.k_scales.size() * sizeof(float));
            std::memset(c.v_scales.data(), 0, c.v_scales.size() * sizeof(float));
        } else {
            c.k.zero_();
            c.v.zero_();
        }
        c.current_pos = 0;
    }
}

void KVCache::resize(int64_t new_max_seq_len) {
    max_seq_len_ = new_max_seq_len;
    for (auto& c : caches_) {
        c.k = Tensor::zeros(Shape{1, num_heads_, max_seq_len_, head_dim_});
        c.v = Tensor::zeros(Shape{1, num_heads_, max_seq_len_, head_dim_});
        c.current_pos = 0;
        int64_t total_vals = num_heads_ * max_seq_len_ * head_dim_;
        c.k_quant.resize((size_t)total_vals);
        c.v_quant.resize((size_t)total_vals);
        int64_t num_blocks = (total_vals + FP8_BLOCK_SIZE - 1) / FP8_BLOCK_SIZE;
        c.k_scales.resize((size_t)num_blocks);
        c.v_scales.resize((size_t)num_blocks);
    }
}

// ===========================================================================
// PagedKVCacheBase — shared implementation
// ===========================================================================

PagedKVCacheBase::PagedKVCacheBase(int num_layers, int64_t num_heads, int64_t head_dim,
                                   int64_t block_size, size_t physical_memory_bytes,
                                   const std::string& disk_path)
    : num_layers_(num_layers), num_heads_(num_heads), head_dim_(head_dim),
      block_size_(block_size), physical_memory_limit_(physical_memory_bytes),
      current_memory_used_(0), disk_path_(disk_path), access_counter_(0),
      next_block_id_(0)
{
    layers_.resize(num_layers);
    async_worker_ = std::thread(&PagedKVCacheBase::async_worker_loop, this);
}

PagedKVCacheBase::~PagedKVCacheBase() {
    {
        std::lock_guard<std::mutex> lk(async_mtx_);
        async_stop_ = true;
    }
    async_cv_.notify_all();
    if (async_worker_.joinable()) async_worker_.join();
}

// ===========================================================================
// Async disk pipeline (C-20)
// ===========================================================================

void PagedKVCacheBase::async_worker_loop() {
    for (;;) {
        AsyncRequest req;
        {
            std::unique_lock<std::mutex> lk(async_mtx_);
            async_cv_.wait(lk, [this] { return async_stop_ || !async_queue_.empty(); });
            if (async_queue_.empty()) {
                if (async_stop_) return;   // drained and told to stop
                continue;
            }
            req = async_queue_.front();
            async_queue_.pop_front();
        }

        // I/O happens here, off the caller's thread. evict_to_disk()/load_from_disk()
        // already keep state consistent on failure (a failed load leaves
        // on_disk == true), so a failed request simply leaves the block where it
        // was and ensure_resident() reports the timeout.
        if (req.evict) {
            evict_to_disk(req.layer, req.block_id);
        } else {
            load_from_disk(req.layer, req.block_id);
        }

        {
            std::lock_guard<std::mutex> lk(async_mtx_);
            async_inflight_.erase({req.layer, req.block_id});
            if (!req.evict) ++async_loads_;
        }
        async_done_cv_.notify_all();
    }
}

bool PagedKVCacheBase::block_is_resident(int layer, int64_t block_id) const {
    // BUGFIX (bug census): lock-free read of layers_/blocks/on_disk while
    // evict/load mutate them (data race). Lookup under async_mtx_.
    std::lock_guard<std::mutex> lock(async_mtx_);
    if (layer < 0 || layer >= num_layers_) return false;
    auto& ls = layers_[(size_t)layer];
    auto it = ls.blocks.find(block_id);
    if (it == ls.blocks.end()) return false;
    return !it->second.on_disk;
}

bool PagedKVCacheBase::enqueue_async(int layer, int64_t block_id, bool evict) const {
    if (layer < 0 || layer >= num_layers_ || block_id < 0) return false;
    std::lock_guard<std::mutex> lk(async_mtx_);
    auto key = std::make_pair(layer, block_id);
    if (async_inflight_.count(key)) return false;   // already on its way
    async_inflight_.insert(key);
    async_queue_.push_back(AsyncRequest{layer, block_id, evict});
    async_cv_.notify_one();
    return true;
}

void PagedKVCacheBase::prefetch(int layer, int64_t block_id) const {
    if (block_is_resident(layer, block_id)) return;
    enqueue_async(layer, block_id, false);
}

void PagedKVCacheBase::prefetch_range(int layer, int64_t start, int64_t end) const {
    if (layer < 0 || layer >= num_layers_ || block_size_ <= 0) return;
    if (start < 0) start = 0;
    if (end <= start) return;
    const int64_t first = start / block_size_;
    const int64_t last = (end - 1) / block_size_;
    for (int64_t b = first; b <= last; ++b) {
        int64_t block_id = resolve_block_id(layer, b * block_size_);
        if (block_id >= 0) prefetch(layer, block_id);
    }
}

bool PagedKVCacheBase::ensure_resident(int layer, int64_t block_id, int timeout_ms) const {
    if (block_is_resident(layer, block_id)) return true;

    const auto key = std::make_pair(layer, block_id);
    enqueue_async(layer, block_id, false);   // false if already in flight

    std::unique_lock<std::mutex> lk(async_mtx_);
    const bool ok = async_done_cv_.wait_for(
        lk, std::chrono::milliseconds(timeout_ms),
        [this, &key] { return async_inflight_.count(key) == 0; });
    lk.unlock();
    // Either the worker finished (ok) or we gave up. Re-check the block itself:
    // a completed-but-failed load leaves on_disk == true, which must be reported
    // as not resident rather than trusted because the request drained.
    return ok && block_is_resident(layer, block_id);
}

bool PagedKVCacheBase::async_worker_running() const {
    return async_worker_.joinable();
}

uint64_t PagedKVCacheBase::async_loads_completed() const {
    std::lock_guard<std::mutex> lk(async_mtx_);
    return async_loads_;
}

size_t PagedKVCacheBase::async_queue_depth() const {
    std::lock_guard<std::mutex> lk(async_mtx_);
    return async_queue_.size() + async_inflight_.size();
}

void PagedKVCacheBase::init_layer_roots() {
    for (int i = 0; i < num_layers_; i++) {
        layers_[i].root = create_root();
        layers_[i].current_pos = 0;
    }
}

void PagedKVCacheBase::cleanup_layer_roots() {
    for (auto& ls : layers_) {
        destroy_root(ls.root);
        ls.root = nullptr;
    }
}

int64_t PagedKVCacheBase::tokens_per_block() const {
    return block_size_;
}

int64_t PagedKVCacheBase::logical_capacity() const {
    return max_logical_tokens_per_layer();
}

int64_t PagedKVCacheBase::num_physical_blocks() const {
    int64_t total = 0;
    for (auto& ls : layers_) total += (int64_t)ls.blocks.size();
    return total;
}

int64_t PagedKVCacheBase::num_disk_blocks() const {
    int64_t total = 0;
    for (auto& ls : layers_) {
        for (auto& [id, blk] : ls.blocks) {
            if (blk.on_disk) total++;
        }
    }
    return total;
}

size_t PagedKVCacheBase::physical_memory_used() const {
    return current_memory_used_;
}

size_t PagedKVCacheBase::physical_memory_limit() const {
    return physical_memory_limit_;
}

int PagedKVCacheBase::context_len() const {
    return layers_.empty() ? 0 : (int)layers_[0].current_pos;
}

void PagedKVCacheBase::append(int layer, int64_t logical_pos, const Tensor& k, const Tensor& v) {
    if (layer < 0 || layer >= num_layers_) return;
    if (logical_pos < 0) return;

    int64_t block_id = resolve_block_id(layer, logical_pos);
    if (block_id < 0) {
        block_id = alloc_block_id(layer, logical_pos);
        if (block_id < 0) return;
    }

    auto& ls = layers_[layer];
    auto it = ls.blocks.find(block_id);
    if (it == ls.blocks.end()) return;
    auto& blk = it->second;

    if (blk.on_disk) {
        // C-20: the load runs on the async worker, not inline here. We wait only
        // for this one block, and a prefetch_range() issued ahead of the access
        // normally means there is nothing to wait for.
        // D6 W6/W7: void ABI -> state is the return proxy. Failure keeps
        // on_disk==true; never touch buffers after failed load.
        if (!ensure_resident(layer, block_id)) return;
        if (blk.on_disk) return;
    }
    int64_t per_block_floats = num_heads_ * block_size_ * head_dim_;
    if (!block_resident_ok(blk.k_data, blk.v_data, per_block_floats)) return;
    blk.last_access = ++access_counter_;
    blk.dirty = true;

    int64_t offset_in_block = logical_pos % block_size_;
    if (offset_in_block < 0 || offset_in_block >= block_size_) return;
    int64_t k_num = k.numel();
    int64_t v_num = v.numel();
    if (k_num <= 0 || v_num <= 0) return;
    int64_t max_floats = num_heads_ * block_size_ * head_dim_;

    const float* ksrc = k.data<float>();
    const float* vsrc = v.data<float>();
    if (ksrc == nullptr || vsrc == nullptr) return;

    int64_t write_offset = offset_in_block * num_heads_ * head_dim_;
    if (write_offset < 0 || write_offset >= max_floats) return;
    int64_t copy_k = std::min(k_num, max_floats - write_offset);
    int64_t copy_v = std::min(v_num, max_floats - write_offset);
    // D6 W6/W7: resident-size gate before memcpy (never OOB).
    if (write_offset + copy_k > (int64_t)blk.k_data.size())
        copy_k = (int64_t)blk.k_data.size() - write_offset;
    if (write_offset + copy_v > (int64_t)blk.v_data.size())
        copy_v = (int64_t)blk.v_data.size() - write_offset;

    if (copy_k > 0) {
        if (blk.k_data.data() == nullptr) return;
        std::memcpy(blk.k_data.data() + write_offset, ksrc, (size_t)copy_k * sizeof(float));
    }
    if (copy_v > 0) {
        if (blk.v_data.data() == nullptr) return;
        std::memcpy(blk.v_data.data() + write_offset, vsrc, (size_t)copy_v * sizeof(float));
    }

    // current_pos tracks the appended TOKEN extent. The copy above accepts a
    // whole block ({heads, tokens, dim}) — advance by the token count actually
    // written, not by 1, or get_range clamps multi-token appends to one token.
    const int64_t tokens_written =
        (num_heads_ > 0 && head_dim_ > 0) ? k_num / (num_heads_ * head_dim_) : 0;
    const int64_t new_end = logical_pos + (tokens_written > 0 ? tokens_written : 1);
    if (new_end > ls.current_pos) ls.current_pos = new_end;
}

std::pair<Tensor, Tensor> PagedKVCacheBase::get_range(int layer, int64_t start, int64_t end) const {
    if (layer < 0 || layer >= num_layers_) return {Tensor(), Tensor()};
    if (start < 0) start = 0;
    int64_t current = layers_[layer].current_pos;
    if (end > current) end = current;
    if (end <= start) return {Tensor(), Tensor()};
    if (num_heads_ <= 0 || head_dim_ <= 0 || block_size_ <= 0)
        return {Tensor(), Tensor()};

    int64_t len = end - start;
    Tensor k_out(Shape{1, num_heads_, len, head_dim_});
    Tensor v_out(Shape{1, num_heads_, len, head_dim_});
    float* k_base = k_out.data<float>();
    float* v_base = v_out.data<float>();
    // D6 W6/W7: empty (never partial-OOB) if outputs not resident.
    if (k_base == nullptr || v_base == nullptr) return {Tensor(), Tensor()};
    if (k_out.numel() < len * num_heads_ * head_dim_ ||
        v_out.numel() < len * num_heads_ * head_dim_)
        return {Tensor(), Tensor()};

    for (int64_t pos = start; pos < end; pos++) {
        int64_t block_id = resolve_block_id(layer, pos);
        if (block_id < 0) continue;

        auto& ls = layers_[layer];
        auto it = ls.blocks.find(block_id);
        if (it == ls.blocks.end()) continue;
        auto& blk = it->second;

        if (blk.on_disk) {
            load_from_disk(layer, block_id);
            // D6 W6/W7: check state-as-return + resident size; skip on failure.
            if (blk.on_disk) continue;
        }
        int64_t max_floats = num_heads_ * block_size_ * head_dim_;
        if (!block_resident_ok(blk.k_data, blk.v_data, max_floats)) continue;
        blk.last_access = ++access_counter_;

        int64_t offset_in_block = pos % block_size_;
        if (offset_in_block < 0 || offset_in_block >= block_size_) continue;
        int64_t head_offset = offset_in_block * num_heads_ * head_dim_;
        int64_t out_offset = (pos - start) * num_heads_ * head_dim_;
        int64_t copy_n = num_heads_ * head_dim_;
        if (copy_n <= 0) continue;
        if (head_offset < 0 || head_offset + copy_n > max_floats) continue;
        // D6 W6/W7: resident + output gates before memcpy (never OOB).
        if (head_offset + copy_n > (int64_t)blk.k_data.size()) continue;
        if (head_offset + copy_n > (int64_t)blk.v_data.size()) continue;
        if (out_offset < 0 || out_offset + copy_n > k_out.numel()) continue;
        if (out_offset + copy_n > v_out.numel()) continue;

        std::memcpy(k_base + out_offset, blk.k_data.data() + head_offset, (size_t)copy_n * sizeof(float));
        std::memcpy(v_base + out_offset, blk.v_data.data() + head_offset, (size_t)copy_n * sizeof(float));
    }

    return {k_out, v_out};
}

void PagedKVCacheBase::evict_to_disk(int layer, int64_t block_id) const {
    // Locking wrapper: internal paths already holding async_mtx_ must call
    // evict_to_disk_locked (below). Public/external callers come here.
    std::lock_guard<std::mutex> lock(async_mtx_);
    evict_to_disk_locked(layer, block_id);
}

// Internal: caller MUST hold async_mtx_ (nested-locking the non-recursive
// mutex is UB — MSVC aborts 0xc0000409, caught by J5-eviction_stress).
void PagedKVCacheBase::evict_to_disk_locked(int layer, int64_t block_id) const {
    if (layer < 0 || layer >= num_layers_) return;
    auto& ls = layers_[layer];
    auto it = ls.blocks.find(block_id);
    if (it == ls.blocks.end()) return;
    auto& blk = it->second;
    if (blk.on_disk) return;
    int64_t per_block_floats = num_heads_ * block_size_ * head_dim_;
    if (per_block_floats <= 0) return;
    // D6 W6/W7: only resident, fully-sized blocks are evictable; otherwise keep
    // state (never write partial buffers to disk).
    if (!block_resident_ok(blk.k_data, blk.v_data, per_block_floats)) return;

    std::string path = block_disk_path(layer, block_id);
    size_t parent_end = path.find_last_of("/\\");
    if (parent_end != std::string::npos) {
        std::string dir = path.substr(0, parent_end);
        try {
#ifdef _WIN32
        std::filesystem::create_directories(dir);
#else
        if (!dir.empty()) std::filesystem::create_directories(dir);
#endif
        } catch (...) {
            return;  // Failure keeps state consistent: still resident.
        }
    }
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return;  // D6 W6/W7: ofs error -> keep resident state.

    int64_t ksize = (int64_t)blk.k_data.size();
    int64_t vsize = (int64_t)blk.v_data.size();
    // D6 W6/W7: check every write; on any failure remove partial file and keep
    // block resident (no memory/size mutation).
    if (!write_exact_bytes(ofs, (const char*)&ksize, sizeof(ksize))) {
        ofs.close();
        std::remove(path.c_str());
        return;
    }
    if (!write_exact_bytes(ofs, (const char*)blk.k_data.data(), ksize * (std::streamsize)sizeof(float))) {
        ofs.close();
        std::remove(path.c_str());
        return;
    }
    if (!write_exact_bytes(ofs, (const char*)&vsize, sizeof(vsize))) {
        ofs.close();
        std::remove(path.c_str());
        return;
    }
    if (!write_exact_bytes(ofs, (const char*)blk.v_data.data(), vsize * (std::streamsize)sizeof(float))) {
        ofs.close();
        std::remove(path.c_str());
        return;
    }
    ofs.close();
    if (!ofs) {  // D6 W6/W7: close/flush error -> keep resident state.
        std::remove(path.c_str());
        return;
    }

    size_t block_bytes = (size_t)per_block_floats * 2 * sizeof(float);
    // Saturating subtract (never wrap to huge on double-evict paths).
    if (current_memory_used_ >= block_bytes) current_memory_used_ -= block_bytes;
    else current_memory_used_ = 0;

    blk.k_data.clear();
    blk.v_data.clear();
    blk.k_data.shrink_to_fit();
    blk.v_data.shrink_to_fit();
    blk.on_disk = true;
    blk.disk_file = path;
}

void PagedKVCacheBase::load_from_disk(int layer, int64_t block_id) const {
    // Locking wrapper (see evict_to_disk above): internal callers use
    // load_from_disk_locked.
    std::lock_guard<std::mutex> lock(async_mtx_);
    load_from_disk_locked(layer, block_id);
}

void PagedKVCacheBase::load_from_disk_locked(int layer, int64_t block_id) const {
    // D6 W6/W7 load_from_disk→bool semantics: header ABI frozen to void, so
    // this wrapper reports status via block state (on_disk==false + resident
    // buffers == success). Every failure path below returns WITHOUT mutating
    // blk/memory/disk_file — state stays consistent for caller retry/skip.
    if (layer < 0 || layer >= num_layers_) return;
    auto& ls = layers_[layer];
    auto it = ls.blocks.find(block_id);
    if (it == ls.blocks.end()) return;
    auto& blk = it->second;
    if (!blk.on_disk) return;
    if (blk.disk_file.empty()) return;  // Missing-file: keep on_disk==true.
    if (num_heads_ <= 0 || block_size_ <= 0 || head_dim_ <= 0) return;

    int64_t per_block_floats = num_heads_ * block_size_ * head_dim_;
    if (per_block_floats <= 0) return;
    size_t block_bytes = (size_t)per_block_floats * 2 * sizeof(float);
    // D6 W6/W7: evict loop must break when no evictable block remains.
    while (physical_memory_limit_ > 0 && current_memory_used_ + block_bytes > physical_memory_limit_) {
        size_t prev_mem = current_memory_used_;
        evict_lru_locked(layer);
        if (current_memory_used_ >= prev_mem) break;
    }
    // Still over budget and no progress possible -> keep on disk (failure).
    if (physical_memory_limit_ > 0 && current_memory_used_ + block_bytes > physical_memory_limit_)
        return;

    std::ifstream ifs(blk.disk_file, std::ios::binary);
    if (!ifs) return;  // D6 W6/W7: missing-file proof: no mutation here.

    // D6 W6/W7: stage into temps first; mutate blk only after ALL reads verify.
    int64_t ksize = 0, vsize = 0;
    if (!read_exact_bytes(ifs, (char*)&ksize, sizeof(ksize))) return;
    if (ksize != per_block_floats) return;  // Size mismatch -> keep state.
    std::vector<float> k_tmp((size_t)ksize);
    if (ksize > 0 &&
        !read_exact_bytes(ifs, (char*)k_tmp.data(), ksize * (std::streamsize)sizeof(float)))
        return;
    if (!read_exact_bytes(ifs, (char*)&vsize, sizeof(vsize))) return;
    if (vsize != per_block_floats) return;
    std::vector<float> v_tmp((size_t)vsize);
    if (vsize > 0 &&
        !read_exact_bytes(ifs, (char*)v_tmp.data(), vsize * (std::streamsize)sizeof(float)))
        return;
    if (!ifs) return;  // D6 W6/W7: any ifs error -> keep on_disk state.
    ifs.close();

    // Success path only: commit, account memory, then unlink backing file.
    blk.k_data = std::move(k_tmp);
    blk.v_data = std::move(v_tmp);
    blk.on_disk = false;
    current_memory_used_ += block_bytes;
    std::remove(blk.disk_file.c_str());
    blk.disk_file.clear();
}

void PagedKVCacheBase::evict_lru(int layer) const { std::lock_guard<std::mutex> lock(async_mtx_); evict_lru_locked(layer); }

void PagedKVCacheBase::evict_lru_locked(int layer) const {
    if (layer < 0 || layer >= num_layers_) return;
    auto& ls = layers_[layer];
    if (ls.blocks.empty()) return;

    int64_t lru_id = -1;
    int64_t oldest_access = INT64_MAX;
    for (auto& [id, blk] : ls.blocks) {
        if (!blk.on_disk && blk.last_access < oldest_access) {
            oldest_access = blk.last_access;
            lru_id = id;
        }
    }
    if (lru_id >= 0) {
        evict_to_disk_locked(layer, lru_id);
    }
}

std::string PagedKVCacheBase::block_disk_path(int layer, int64_t block_id) const {
    std::ostringstream ss;
    if (!disk_path_.empty()) {
        ss << disk_path_;
        if (disk_path_.back() != '/' && disk_path_.back() != '\\')
            ss << "/";
    } else {
#ifdef _WIN32
        const char* tmp = std::getenv("TEMP");
        ss << (tmp ? tmp : "C:\\Temp") << "\\Transcender_" << disk_name() << "\\";
#else
        ss << "/tmp/Transcender_" << disk_name() << "/";
#endif
    }
    ss << disk_name() << "_l" << layer << "_b" << block_id << ".bin";
    return ss.str();
}

void PagedKVCacheBase::flush_to_disk() {
    for (int layer = 0; layer < num_layers_; layer++) {
        auto& ls = layers_[layer];
        std::vector<int64_t> to_offload;
        for (auto& [id, blk] : ls.blocks) {
            if (!blk.on_disk && blk.dirty) to_offload.push_back(id);
        }
        for (int64_t id : to_offload) evict_to_disk_locked(layer, id);
    }
}

void PagedKVCacheBase::load_from_disk() {
    // D6 W6/W7: header ABI frozen to void; per-block bool status is observed
    // via state (on_disk==false == success). Failures keep block state.
    PAGED_KV_DLOG("D-lfd-enter" << std::endl);
    for (int layer = 0; layer < num_layers_; layer++) {
        auto& ls = layers_[layer];
        PAGED_KV_DLOG("D-lfd-layer n=" << ls.blocks.size() << std::endl);
        for (auto& [id, blk] : ls.blocks) {
            if (blk.on_disk) load_from_disk_locked(layer, id);
        }
    }
    PAGED_KV_DLOG("D-lfd-exit" << std::endl);
}

void PagedKVCacheBase::clear() {
    PAGED_KV_DLOG("D-clear-enter" << std::endl);
    for (auto& ls : layers_) {
        PAGED_KV_DLOG("D-clear-blocks n=" << ls.blocks.size() << std::endl);
        for (auto& [id, blk] : ls.blocks) {
            if (blk.on_disk && !blk.disk_file.empty()) {
                std::remove(blk.disk_file.c_str());
            }
        }
        ls.blocks.clear();
        PAGED_KV_DLOG("D-clear-pos" << std::endl);
        ls.current_pos = 0;
        PAGED_KV_DLOG("D-destroy-root" << std::endl);
        destroy_root(ls.root);
        PAGED_KV_DLOG("D-new-root" << std::endl);
        ls.root = create_root();
    }
    current_memory_used_ = 0;
    next_block_id_ = 0;
    access_counter_ = 0;
    PAGED_KV_DLOG("D-clear-exit" << std::endl);
}

void PagedKVCacheBase::unload_memory() {
    // Flush dirty blocks, then drop resident data but keep the block map,
    // disk files and positions so load_from_disk() restores contents. P19.
    flush_to_disk();
    for (int layer = 0; layer < num_layers_; layer++) {
        auto& ls = layers_[layer];
        std::vector<int64_t> ids;
        for (auto& kv : ls.blocks) ids.push_back(kv.first);
        for (int64_t id : ids) {
            auto it = ls.blocks.find(id);
            if (it == ls.blocks.end()) continue;
            if (!it->second.on_disk) evict_to_disk_locked(layer, id);
            else {
                it->second.k_data.clear(); it->second.k_data.shrink_to_fit();
                it->second.v_data.clear(); it->second.v_data.shrink_to_fit();
            }
        }
    }
}

// ===========================================================================
// PagedKVCache4M — 2-level page table specifics
// ===========================================================================

PagedKVCache4M::L2Table::L2Table() {
    for (int i = 0; i < TABLE_ENTRIES; i++) entries[i] = -1;
}

PagedKVCache4M::L1Table::L1Table() {
    for (int i = 0; i < TABLE_ENTRIES; i++) entries[i] = nullptr;
}

PagedKVCache4M::L1Table::~L1Table() {
    for (int i = 0; i < TABLE_ENTRIES; i++) delete entries[i];
}

PagedKVCache4M::PagedKVCache4M(int num_layers, int64_t num_heads, int64_t head_dim,
                                int64_t block_size, size_t physical_memory_bytes,
                                const std::string& disk_path)
    : PagedKVCacheBase(num_layers, num_heads, head_dim, block_size,
                       physical_memory_bytes, disk_path)
{
    init_layer_roots();
}

PagedKVCache4M::~PagedKVCache4M() {
    cleanup_layer_roots();
}

void* PagedKVCache4M::create_root() const {
    return new L1Table();
}

void PagedKVCache4M::destroy_root(void* root) const {
    delete static_cast<L1Table*>(root);
}

int64_t PagedKVCache4M::max_logical_tokens_per_layer() const {
    return TABLE_ENTRIES * TABLE_ENTRIES * block_size_;
}

int64_t PagedKVCache4M::resolve_block_id(int layer, int64_t logical_pos) const {
    if (layer < 0 || layer >= num_layers_) return -1;
    if (logical_pos < 0 || block_size_ <= 0) return -1;
    int64_t block_idx = logical_pos / block_size_;
    if (block_idx < 0) return -1;
    int64_t l1_idx = block_idx / TABLE_ENTRIES;
    int64_t l2_idx = block_idx % TABLE_ENTRIES;

    if (l1_idx < 0 || l1_idx >= TABLE_ENTRIES) return -1;
    if (l2_idx < 0 || l2_idx >= TABLE_ENTRIES) return -1;

    const auto& ls = layers_[layer];
    if (ls.root == nullptr) return -1;  // D6 W6/W7: null-check ls.root.
    L1Table* l1 = static_cast<L1Table*>(ls.root);
    if (l1 == nullptr) return -1;
    L2Table* l2 = l1->entries[(size_t)l1_idx];
    if (!l2) return -1;
    return l2->entries[(size_t)l2_idx];
}

int64_t PagedKVCache4M::alloc_block_id(int layer, int64_t logical_pos) {
    if (layer < 0 || layer >= num_layers_) return -1;
    if (logical_pos < 0 || block_size_ <= 0) return -1;
    if (num_heads_ <= 0 || head_dim_ <= 0) return -1;
    int64_t block_idx = logical_pos / block_size_;
    if (block_idx < 0) return -1;
    int64_t l1_idx = block_idx / TABLE_ENTRIES;
    int64_t l2_idx = block_idx % TABLE_ENTRIES;

    if (l1_idx < 0 || l1_idx >= TABLE_ENTRIES) return -1;
    if (l2_idx < 0 || l2_idx >= TABLE_ENTRIES) return -1;
    // D6 W6/W7: bound next_block_id_ (overflow + capacity cap).
    if (next_block_id_ < 0 || next_block_id_ == INT64_MAX ||
        next_block_id_ == std::numeric_limits<int64_t>::max())
        return -1;
    if (block_idx * block_size_ >= max_logical_tokens_per_layer()) return -1;

    auto& ls = layers_[layer];
    if (ls.root == nullptr) return -1;  // D6 W6/W7: null-check ls.root.
    L1Table* l1 = static_cast<L1Table*>(ls.root);
    if (l1 == nullptr) return -1;
    L2Table*& l2 = l1->entries[(size_t)l1_idx];
    if (!l2) l2 = new L2Table();

    int64_t id = next_block_id_++;
    l2->entries[(size_t)l2_idx] = id;

    PhysicalBlock blk;
    blk.id = id;
    int64_t per_block_floats = num_heads_ * block_size_ * head_dim_;
    if (per_block_floats <= 0) {
        l2->entries[(size_t)l2_idx] = -1;
        return -1;
    }
    blk.k_data.resize((size_t)per_block_floats, 0.0f);
    blk.v_data.resize((size_t)per_block_floats, 0.0f);
    blk.last_access = ++access_counter_;

    size_t block_bytes = (size_t)per_block_floats * 2 * sizeof(float);
    // D6 W6/W7: evict loop breaks when no evictable block remains.
    if (physical_memory_limit_ > 0) {
        while (current_memory_used_ + block_bytes > physical_memory_limit_) {
            size_t prev = current_memory_used_;
            evict_lru_locked(layer);
            if (current_memory_used_ >= prev) break;
        }
        if (current_memory_used_ + block_bytes > physical_memory_limit_) {
            // Keep state consistent: roll back page-table slot, no insert.
            l2->entries[(size_t)l2_idx] = -1;
            return -1;
        }
    }

    ls.blocks[id] = std::move(blk);
    current_memory_used_ += block_bytes;
    return id;
}

std::pair<Tensor, Tensor> PagedKVCache4M::get_block(int layer, int64_t logical_pos) const {
    if (layer < 0 || layer >= num_layers_) return {Tensor(), Tensor()};
    if (logical_pos < 0) return {Tensor(), Tensor()};
    int64_t block_id = resolve_block_id(layer, logical_pos);
    if (block_id < 0) return {Tensor(), Tensor()};

    auto& ls = layers_[layer];
    auto it = ls.blocks.find(block_id);
    if (it == ls.blocks.end()) return {Tensor(), Tensor()};
    auto& blk = it->second;

    if (blk.on_disk) {
        load_from_disk(layer, block_id);
        // D6 W6/W7: check state-as-return; empty (never OOB) on failure.
        if (blk.on_disk) return {Tensor(), Tensor()};
    }
    if (num_heads_ <= 0 || block_size_ <= 0 || head_dim_ <= 0)
        return {Tensor(), Tensor()};
    int64_t per_head = block_size_ * head_dim_;
    int64_t need = num_heads_ * per_head;
    // D6 W6/W7: resident-size gate before memcpy (never OOB).
    if (!block_resident_ok(blk.k_data, blk.v_data, need))
        return {Tensor(), Tensor()};
    Tensor k_out(Shape{1, num_heads_, block_size_, head_dim_});
    Tensor v_out(Shape{1, num_heads_, block_size_, head_dim_});
    float* k_dst = k_out.data<float>();
    float* v_dst = v_out.data<float>();
    if (k_dst == nullptr || v_dst == nullptr) return {Tensor(), Tensor()};
    if (k_out.numel() < need || v_out.numel() < need)
        return {Tensor(), Tensor()};

    std::memcpy(k_dst, blk.k_data.data(), (size_t)need * sizeof(float));
    std::memcpy(v_dst, blk.v_data.data(), (size_t)need * sizeof(float));

    blk.last_access = ++access_counter_;
    return {k_out, v_out};
}

bool PagedKVCache4M::verify_retrieval(int layer, int64_t pos,
                                       const Tensor& expected_k,
                                       const Tensor& expected_v) const {
    auto [k, v] = get_range(layer, pos, pos + 1);
    if (k.numel() == 0 || v.numel() == 0) return false;

    const float* kd = k.data<float>();
    const float* vd = v.data<float>();
    const float* ek = expected_k.data<float>();
    const float* ev = expected_v.data<float>();
    // D6 W6/W7: null-gate before compare (never OOB on empty tensors).
    if (kd == nullptr || vd == nullptr || ek == nullptr || ev == nullptr)
        return false;

    int64_t n = std::min(k.numel(), expected_k.numel());
    for (int64_t i = 0; i < n; i++) {
        if (std::fabs(kd[i] - ek[i]) > 1e-5f) return false;
    }
    n = std::min(v.numel(), expected_v.numel());
    for (int64_t i = 0; i < n; i++) {
        if (std::fabs(vd[i] - ev[i]) > 1e-5f) return false;
    }
    return true;
}

const char* PagedKVCache4M::disk_name() const {
    return "paged_kv";
}

// ===========================================================================
// PagedKVCache1T — 3-level page table specifics
// ===========================================================================

PagedKVCache1T::L3Table::L3Table() {
    for (int i = 0; i < TABLE_ENTRIES; i++) entries[i] = -1;
}

PagedKVCache1T::L2Table::L2Table() {
    for (int i = 0; i < TABLE_ENTRIES; i++) entries[i] = nullptr;
}

PagedKVCache1T::L2Table::~L2Table() {
    for (int i = 0; i < TABLE_ENTRIES; i++) delete entries[i];
}

PagedKVCache1T::L1Table::L1Table() {
    for (int i = 0; i < TABLE_ENTRIES; i++) entries[i] = nullptr;
}

PagedKVCache1T::L1Table::~L1Table() {
    for (int i = 0; i < TABLE_ENTRIES; i++) delete entries[i];
}

PagedKVCache1T::PagedKVCache1T(int num_layers, int64_t num_heads, int64_t head_dim,
                                int64_t block_size, size_t physical_memory_bytes,
                                const std::string& disk_path)
    : PagedKVCacheBase(num_layers, num_heads, head_dim, block_size,
                       physical_memory_bytes, disk_path)
{
    init_layer_roots();
}

PagedKVCache1T::~PagedKVCache1T() {
    cleanup_layer_roots();
}

void* PagedKVCache1T::create_root() const {
    return new L1Table();
}

void PagedKVCache1T::destroy_root(void* root) const {
    delete static_cast<L1Table*>(root);
}

int64_t PagedKVCache1T::max_logical_tokens_per_layer() const {
    return TABLE_ENTRIES * TABLE_ENTRIES * TABLE_ENTRIES * block_size_;
}

int64_t PagedKVCache1T::resolve_block_id(int layer, int64_t logical_pos) const {
    if (layer < 0 || layer >= num_layers_) return -1;
    if (logical_pos < 0 || block_size_ <= 0) return -1;
    int64_t block_idx = logical_pos / block_size_;
    if (block_idx < 0) return -1;
    int64_t l1_idx = block_idx / (TABLE_ENTRIES * TABLE_ENTRIES);
    int64_t rem = block_idx % (TABLE_ENTRIES * TABLE_ENTRIES);
    int64_t l2_idx = rem / TABLE_ENTRIES;
    int64_t l3_idx = rem % TABLE_ENTRIES;

    if (l1_idx < 0 || l1_idx >= TABLE_ENTRIES) return -1;
    if (l2_idx < 0 || l2_idx >= TABLE_ENTRIES) return -1;
    if (l3_idx < 0 || l3_idx >= TABLE_ENTRIES) return -1;

    const auto& ls = layers_[layer];
    if (ls.root == nullptr) return -1;  // D6 W6/W7: null-check ls.root.
    L1Table* l1 = static_cast<L1Table*>(ls.root);
    if (l1 == nullptr) return -1;
    L2Table* l2 = l1->entries[(size_t)l1_idx];
    if (!l2) return -1;
    L3Table* l3 = l2->entries[(size_t)l2_idx];
    if (!l3) return -1;
    return l3->entries[(size_t)l3_idx];
}

int64_t PagedKVCache1T::alloc_block_id(int layer, int64_t logical_pos) {
    if (layer < 0 || layer >= num_layers_) return -1;
    if (logical_pos < 0 || block_size_ <= 0) return -1;
    if (num_heads_ <= 0 || head_dim_ <= 0) return -1;
    int64_t block_idx = logical_pos / block_size_;
    if (block_idx < 0) return -1;
    int64_t l1_idx = block_idx / (TABLE_ENTRIES * TABLE_ENTRIES);
    int64_t rem = block_idx % (TABLE_ENTRIES * TABLE_ENTRIES);
    int64_t l2_idx = rem / TABLE_ENTRIES;
    int64_t l3_idx = rem % TABLE_ENTRIES;

    if (l1_idx < 0 || l1_idx >= TABLE_ENTRIES) return -1;
    if (l2_idx < 0 || l2_idx >= TABLE_ENTRIES) return -1;
    if (l3_idx < 0 || l3_idx >= TABLE_ENTRIES) return -1;
    // D6 W6/W7: bound next_block_id_ (overflow + capacity cap).
    if (next_block_id_ < 0 || next_block_id_ == INT64_MAX ||
        next_block_id_ == std::numeric_limits<int64_t>::max())
        return -1;
    if (block_idx * block_size_ >= max_logical_tokens_per_layer()) return -1;

    auto& ls = layers_[layer];
    if (ls.root == nullptr) return -1;  // D6 W6/W7: null-check ls.root.
    L1Table* l1 = static_cast<L1Table*>(ls.root);
    if (l1 == nullptr) return -1;
    L2Table*& l2 = l1->entries[(size_t)l1_idx];
    if (!l2) l2 = new L2Table();
    L3Table*& l3 = l2->entries[(size_t)l2_idx];
    if (!l3) l3 = new L3Table();

    int64_t id = next_block_id_++;
    l3->entries[(size_t)l3_idx] = id;

    PhysicalBlock blk;
    blk.id = id;
    int64_t per_block_floats = num_heads_ * block_size_ * head_dim_;
    if (per_block_floats <= 0) {
        l3->entries[(size_t)l3_idx] = -1;
        return -1;
    }
    blk.k_data.resize((size_t)per_block_floats, 0.0f);
    blk.v_data.resize((size_t)per_block_floats, 0.0f);
    blk.last_access = ++access_counter_;

    size_t block_bytes = (size_t)per_block_floats * 2 * sizeof(float);
    // D6 W6/W7: evict loop breaks when no evictable block remains.
    if (physical_memory_limit_ > 0) {
        while (current_memory_used_ + block_bytes > physical_memory_limit_) {
            size_t prev = current_memory_used_;
            evict_lru_locked(layer);
            if (current_memory_used_ >= prev) break;
        }
        if (current_memory_used_ + block_bytes > physical_memory_limit_) {
            // Keep state consistent: roll back page-table slot, no insert.
            l3->entries[(size_t)l3_idx] = -1;
            return -1;
        }
    }

    ls.blocks[id] = std::move(blk);
    current_memory_used_ += block_bytes;
    return id;
}

const char* PagedKVCache1T::disk_name() const {
    return "paged_kv1t";
}

void PagedKVCache::init(int num_layers, int64_t max_seq_len, int64_t num_heads,
                        int64_t head_dim) {
    max_seq_len_ = max_seq_len;
    cache_ = std::make_unique<PagedKVCache4M>(num_layers, num_heads, head_dim,
                                              PagedKVCache4M::DEFAULT_BLOCK_SIZE);
}

int64_t PagedKVCache::logical_capacity() const {
    return cache_ ? cache_->logical_capacity() : 0;
}

} // namespace quant
