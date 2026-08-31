#include "quant/kv_cache.h"
#include <cstring>
#include <iostream> // DEBUG-A
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstdio>
#ifndef _WIN32
#include <sys/stat.h>
#endif
#include <filesystem>

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
                    int64_t bo = off % FP8_BLOCK_SIZE; (void)bo;
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
    return caches_.empty() ? 0 : caches_[0].current_pos;
}

int KVCache::context_len(int layer) const {
    return (layer >= 0 && layer < (int)caches_.size()) ? caches_[layer].current_pos : 0;
}

int KVCache::max_seq_len() const { return (int)max_seq_len_; }

size_t KVCache::size_bytes() const {
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
}

PagedKVCacheBase::~PagedKVCacheBase() = default;

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
        load_from_disk(layer, block_id);
    }
    blk.last_access = ++access_counter_;
    blk.dirty = true;

    int64_t offset_in_block = logical_pos % block_size_;
    int64_t k_num = k.numel();
    int64_t v_num = v.numel();
    int64_t max_floats = num_heads_ * block_size_ * head_dim_;

    const float* ksrc = k.data<float>();
    const float* vsrc = v.data<float>();

    int64_t write_offset = offset_in_block * num_heads_ * head_dim_;
    int64_t copy_k = std::min(k_num, max_floats - write_offset);
    int64_t copy_v = std::min(v_num, max_floats - write_offset);

    if (copy_k > 0) std::memcpy(blk.k_data.data() + write_offset, ksrc, (size_t)copy_k * sizeof(float));
    if (copy_v > 0) std::memcpy(blk.v_data.data() + write_offset, vsrc, (size_t)copy_v * sizeof(float));

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

    int64_t len = end - start;
    Tensor k_out(Shape{1, num_heads_, len, head_dim_});
    Tensor v_out(Shape{1, num_heads_, len, head_dim_});

    for (int64_t pos = start; pos < end; pos++) {
        int64_t block_id = resolve_block_id(layer, pos);
        if (block_id < 0) continue;

        auto& ls = layers_[layer];
        auto it = ls.blocks.find(block_id);
        if (it == ls.blocks.end()) continue;
        auto& blk = it->second;

        if (blk.on_disk) {
            load_from_disk(layer, block_id);
        }
        blk.last_access = ++access_counter_;

        int64_t offset_in_block = pos % block_size_;
        int64_t head_offset = offset_in_block * num_heads_ * head_dim_;
        int64_t out_offset = (pos - start) * num_heads_ * head_dim_;
        int64_t copy_n = num_heads_ * head_dim_;
        int64_t max_floats = num_heads_ * block_size_ * head_dim_;
        if (head_offset + copy_n > max_floats) copy_n = max_floats - head_offset;

        std::memcpy(k_out.data<float>() + out_offset, blk.k_data.data() + head_offset, (size_t)copy_n * sizeof(float));
        std::memcpy(v_out.data<float>() + out_offset, blk.v_data.data() + head_offset, (size_t)copy_n * sizeof(float));
    }

    return {k_out, v_out};
}

void PagedKVCacheBase::evict_to_disk(int layer, int64_t block_id) const {
    auto& ls = layers_[layer];
    auto it = ls.blocks.find(block_id);
    if (it == ls.blocks.end()) return;
    auto& blk = it->second;
    if (blk.on_disk) return;

    std::string path = block_disk_path(layer, block_id);
    size_t parent_end = path.find_last_of("/\\");
    if (parent_end != std::string::npos) {
        std::string dir = path.substr(0, parent_end);
#ifdef _WIN32
        std::filesystem::create_directories(dir);
#else
        mkdir(dir.c_str(), 0755);
#endif
    }
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return;

    int64_t ksize = (int64_t)blk.k_data.size();
    int64_t vsize = (int64_t)blk.v_data.size();
    ofs.write((const char*)&ksize, sizeof(ksize));
    ofs.write((const char*)blk.k_data.data(), ksize * sizeof(float));
    ofs.write((const char*)&vsize, sizeof(vsize));
    ofs.write((const char*)blk.v_data.data(), vsize * sizeof(float));
    ofs.close();

    int64_t per_block_floats = num_heads_ * block_size_ * head_dim_;
    size_t block_bytes = (size_t)per_block_floats * 2 * sizeof(float);
    current_memory_used_ -= block_bytes;

    blk.k_data.clear();
    blk.v_data.clear();
    blk.k_data.shrink_to_fit();
    blk.v_data.shrink_to_fit();
    blk.on_disk = true;
    blk.disk_file = path;
}

void PagedKVCacheBase::load_from_disk(int layer, int64_t block_id) const {
    auto& ls = layers_[layer];
    auto it = ls.blocks.find(block_id);
    if (it == ls.blocks.end()) return;
    auto& blk = it->second;
    if (!blk.on_disk) return;

    int64_t per_block_floats = num_heads_ * block_size_ * head_dim_;
    size_t block_bytes = (size_t)per_block_floats * 2 * sizeof(float);
    while (physical_memory_limit_ > 0 && current_memory_used_ + block_bytes > physical_memory_limit_) {
        size_t prev_mem = current_memory_used_;
        evict_lru(layer);
        if (current_memory_used_ >= prev_mem) break;
    }

    std::ifstream ifs(blk.disk_file, std::ios::binary);
    if (!ifs) return;

    int64_t ksize, vsize;
    ifs.read((char*)&ksize, sizeof(ksize));
    blk.k_data.resize((size_t)ksize);
    ifs.read((char*)blk.k_data.data(), ksize * sizeof(float));
    ifs.read((char*)&vsize, sizeof(vsize));
    blk.v_data.resize((size_t)vsize);
    ifs.read((char*)blk.v_data.data(), vsize * sizeof(float));
    ifs.close();

    blk.on_disk = false;
    current_memory_used_ += block_bytes;
    std::remove(blk.disk_file.c_str());
    blk.disk_file.clear();
}

void PagedKVCacheBase::evict_lru(int layer) const {
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
        evict_to_disk(layer, lru_id);
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
        ss << (tmp ? tmp : "C:\\Temp") << "\\InNova_" << disk_name() << "\\";
#else
        ss << "/tmp/InNova_" << disk_name() << "/";
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
        for (int64_t id : to_offload) evict_to_disk(layer, id);
    }
}

void PagedKVCacheBase::load_from_disk() {
    std::cerr << "D-lfd-enter" << std::endl;
    for (int layer = 0; layer < num_layers_; layer++) {
        auto& ls = layers_[layer];
        std::cerr << "D-lfd-layer n=" << ls.blocks.size() << std::endl;
        for (auto& [id, blk] : ls.blocks) {
            if (blk.on_disk) load_from_disk(layer, id);
        }
    }
    std::cerr << "D-lfd-exit" << std::endl;
}

void PagedKVCacheBase::clear() {
    std::cerr << "D-clear-enter" << std::endl;
    for (auto& ls : layers_) {
        std::cerr << "D-clear-blocks n=" << ls.blocks.size() << std::endl;
        for (auto& [id, blk] : ls.blocks) {
            if (blk.on_disk && !blk.disk_file.empty()) {
                std::remove(blk.disk_file.c_str());
            }
        }
        ls.blocks.clear();
        std::cerr << "D-clear-pos" << std::endl;
        ls.current_pos = 0;
        std::cerr << "D-destroy-root" << std::endl;
        destroy_root(ls.root);
        std::cerr << "D-new-root" << std::endl;
        ls.root = create_root();
    }
    current_memory_used_ = 0;
    next_block_id_ = 0;
    access_counter_ = 0;
    std::cerr << "D-clear-exit" << std::endl;
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
    int64_t block_idx = logical_pos / block_size_;
    int64_t l1_idx = block_idx / TABLE_ENTRIES;
    int64_t l2_idx = block_idx % TABLE_ENTRIES;

    if (l1_idx >= TABLE_ENTRIES) return -1;

    const auto& ls = layers_[layer];
    L1Table* l1 = static_cast<L1Table*>(ls.root);
    L2Table* l2 = l1->entries[(size_t)l1_idx];
    if (!l2) return -1;
    return l2->entries[(size_t)l2_idx];
}

int64_t PagedKVCache4M::alloc_block_id(int layer, int64_t logical_pos) {
    int64_t block_idx = logical_pos / block_size_;
    int64_t l1_idx = block_idx / TABLE_ENTRIES;
    int64_t l2_idx = block_idx % TABLE_ENTRIES;

    if (l1_idx >= TABLE_ENTRIES) return -1;

    auto& ls = layers_[layer];
    L1Table* l1 = static_cast<L1Table*>(ls.root);
    L2Table*& l2 = l1->entries[(size_t)l1_idx];
    if (!l2) l2 = new L2Table();

    int64_t id = next_block_id_++;
    l2->entries[(size_t)l2_idx] = id;

    PhysicalBlock blk;
    blk.id = id;
    int64_t per_block_floats = num_heads_ * block_size_ * head_dim_;
    blk.k_data.resize((size_t)per_block_floats, 0.0f);
    blk.v_data.resize((size_t)per_block_floats, 0.0f);
    blk.last_access = ++access_counter_;

    size_t block_bytes = (size_t)per_block_floats * 2 * sizeof(float);
    while (current_memory_used_ + block_bytes > physical_memory_limit_ && !ls.blocks.empty()) {
        evict_lru(layer);
    }

    ls.blocks[id] = std::move(blk);
    current_memory_used_ += block_bytes;
    return id;
}

std::pair<Tensor, Tensor> PagedKVCache4M::get_block(int layer, int64_t logical_pos) const {
    int64_t block_id = resolve_block_id(layer, logical_pos);
    if (block_id < 0) return {Tensor(), Tensor()};

    auto& ls = layers_[layer];
    auto it = ls.blocks.find(block_id);
    if (it == ls.blocks.end()) return {Tensor(), Tensor()};
    auto& blk = it->second;

    if (blk.on_disk) {
        load_from_disk(layer, block_id);
    }
    blk.last_access = ++access_counter_;

    int64_t per_head = block_size_ * head_dim_;
    Tensor k_out(Shape{1, num_heads_, block_size_, head_dim_});
    Tensor v_out(Shape{1, num_heads_, block_size_, head_dim_});

    std::memcpy(k_out.data<float>(), blk.k_data.data(), (size_t)num_heads_ * per_head * sizeof(float));
    std::memcpy(v_out.data<float>(), blk.v_data.data(), (size_t)num_heads_ * per_head * sizeof(float));

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
    int64_t block_idx = logical_pos / block_size_;
    int64_t l1_idx = block_idx / (TABLE_ENTRIES * TABLE_ENTRIES);
    int64_t rem = block_idx % (TABLE_ENTRIES * TABLE_ENTRIES);
    int64_t l2_idx = rem / TABLE_ENTRIES;
    int64_t l3_idx = rem % TABLE_ENTRIES;

    if (l1_idx >= TABLE_ENTRIES) return -1;

    const auto& ls = layers_[layer];
    L1Table* l1 = static_cast<L1Table*>(ls.root);
    L2Table* l2 = l1->entries[(size_t)l1_idx];
    if (!l2) return -1;
    L3Table* l3 = l2->entries[(size_t)l2_idx];
    if (!l3) return -1;
    return l3->entries[(size_t)l3_idx];
}

int64_t PagedKVCache1T::alloc_block_id(int layer, int64_t logical_pos) {
    int64_t block_idx = logical_pos / block_size_;
    int64_t l1_idx = block_idx / (TABLE_ENTRIES * TABLE_ENTRIES);
    int64_t rem = block_idx % (TABLE_ENTRIES * TABLE_ENTRIES);
    int64_t l2_idx = rem / TABLE_ENTRIES;
    int64_t l3_idx = rem % TABLE_ENTRIES;

    if (l1_idx >= TABLE_ENTRIES) return -1;

    auto& ls = layers_[layer];
    L1Table* l1 = static_cast<L1Table*>(ls.root);
    L2Table*& l2 = l1->entries[(size_t)l1_idx];
    if (!l2) l2 = new L2Table();
    L3Table*& l3 = l2->entries[(size_t)l2_idx];
    if (!l3) l3 = new L3Table();

    int64_t id = next_block_id_++;
    l3->entries[(size_t)l3_idx] = id;

    PhysicalBlock blk;
    blk.id = id;
    int64_t per_block_floats = num_heads_ * block_size_ * head_dim_;
    blk.k_data.resize((size_t)per_block_floats, 0.0f);
    blk.v_data.resize((size_t)per_block_floats, 0.0f);
    blk.last_access = ++access_counter_;

    size_t block_bytes = (size_t)per_block_floats * 2 * sizeof(float);
    while (current_memory_used_ + block_bytes > physical_memory_limit_ && !ls.blocks.empty()) {
        evict_lru(layer);
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
