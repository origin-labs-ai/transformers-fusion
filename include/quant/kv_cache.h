#pragma once
#include "quant/types.h"
#include "quant/tensor.h"
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <deque>
#include <set>
#include <thread>
#include <condition_variable>

namespace quant {

class KVCache {
public:
    KVCache() = default;
    KVCache(KVCache&& other) noexcept;
    KVCache& operator=(KVCache&& other) noexcept;
    KVCache(int num_layers, int64_t max_seq_len, int64_t num_heads, 
            int64_t head_dim, bool quantized = false);
    
    void init(int num_layers, int64_t max_seq_len, int64_t num_heads,
              int64_t head_dim, bool quantized = false);
    
    void append(int layer, const Tensor& k, const Tensor& v);
    
    std::pair<Tensor, Tensor> get_range(int layer, int start, int end) const;
    std::pair<Tensor, Tensor> get_all(int layer) const;
    
    int context_len() const;
    int context_len(int layer) const;
    int max_seq_len() const;
    
    size_t size_bytes() const;
    
    void clear();
    
    void resize(int64_t new_max_seq_len);

    // Truncate the cache back to `new_len` tokens, discarding everything
    // appended after it. Unlike resize() (which re-allocates and wipes the
    // whole cache), truncate keeps the buffers and capacity intact — only
    // `current_pos` moves back and the vacated tail rows are zeroed so a
    // later re-append is deterministic. Clamps: new_len < 0 is treated as 0,
    // new_len beyond the current position is a no-op. Thread-safe (mutex_).
    // Primary consumer: SpeculativeDecoder::rewind_kv on draft rejection.
    void truncate(int64_t new_len);
    
    static constexpr int FP8_BLOCK_SIZE = 64;
    static constexpr float FP8_MAX = 127.0f;

    static void quantize_fp8_block(const float* src, uint8_t* dst, float* scale,
                                    int64_t n);
    static void dequantize_fp8_block(const uint8_t* src, float scale,
                                      float* dst, int64_t n);

private:
    struct LayerCache {
        Tensor k;
        Tensor v;
        int current_pos = 0;
        std::vector<uint8_t> k_quant;
        std::vector<uint8_t> v_quant;
        std::vector<float> k_scales;
        std::vector<float> v_scales;
    };
    std::vector<LayerCache> caches_;
    int num_layers_ = 0;
    int64_t max_seq_len_ = 0;
    int64_t num_heads_ = 0;
    int64_t head_dim_ = 0;
    bool quantized_ = false;
    mutable std::mutex mutex_;
};

// ===========================================================================
// PagedKVCacheBase — shared implementation for hierarchical paged KV caches
//
// Both PagedKVCache4M (2-level) and PagedKVCache1T (3-level) inherit from
// this base. All shared logic (LRU tracking, disk I/O, block management,
// append, get_range, flush, clear) lives here. Only the page-table-specific
// operations (resolve, alloc, max_logical) are virtual pure.
// ===========================================================================

class PagedKVCacheBase {
public:
    PagedKVCacheBase(int num_layers, int64_t num_heads, int64_t head_dim,
                     int64_t block_size, size_t physical_memory_bytes,
                     const std::string& disk_path);
    virtual ~PagedKVCacheBase();
    void init_layer_roots();
    void cleanup_layer_roots();

    PagedKVCacheBase(const PagedKVCacheBase&) = delete;
    PagedKVCacheBase& operator=(const PagedKVCacheBase&) = delete;

    void append(int layer, int64_t logical_pos, const Tensor& k, const Tensor& v);
    std::pair<Tensor, Tensor> get_range(int layer, int64_t start, int64_t end) const;

    int64_t logical_capacity() const;
    int64_t num_physical_blocks() const;
    int64_t num_disk_blocks() const;
    size_t physical_memory_used() const;
    size_t physical_memory_limit() const;

    void flush_to_disk();
    void load_from_disk();
    void clear();
    // Drop all resident blocks to disk but keep the block map, disk files and
    // positions, so a later load_from_disk() restores contents. Unlike clear()
    // (full wipe incl. disk files), this is the correct flush→reload sequence. P19.
    void unload_memory();

    // ---- Async disk pipeline (C-20) ----------------------------------------
    // Before this, every disk read happened inline on the attention path:
    // append() called load_from_disk() and blocked on fread. A single background
    // worker now owns block loads, and callers talk to it through:
    //   prefetch()        — enqueue a load, never blocks (fire and forget)
    //   prefetch_range()  — warm every block covering [start, end)
    //   ensure_resident() — wait until the block is in RAM (correctness path)
    // The worker serialises requests, so overlapping prefetches cannot race on
    // the same block, and the hot path only pays a wait when no prefetch got
    // there first. Correctness is unchanged: a block is either resident or the
    // caller waits for it.
    //
    // Threading contract: the cache keeps its original single-caller contract
    // (one thread drives append/get_range). The worker is the only other thread
    // and it never touches a block while the caller can reach it — the caller
    // waits for the in-flight request to complete before proceeding, so no
    // block-level lock is needed on the data itself.
    void prefetch(int layer, int64_t block_id) const;
    void prefetch_range(int layer, int64_t start, int64_t end) const;
    bool ensure_resident(int layer, int64_t block_id, int timeout_ms = 5000) const;
    bool async_worker_running() const;
    uint64_t async_loads_completed() const;
    size_t async_queue_depth() const;

    int context_len() const;
    int64_t block_size() const { return block_size_; }
    int num_layers() const { return num_layers_; }
    int64_t num_heads() const { return num_heads_; }
    int64_t head_dim() const { return head_dim_; }

protected:
    struct PhysicalBlock {
        int64_t id = -1;
        mutable std::vector<float> k_data;
        mutable std::vector<float> v_data;
        bool dirty = false;
        mutable int64_t last_access = 0;
        mutable bool on_disk = false;
        mutable std::string disk_file;
    };

    struct LayerState {
        void* root = nullptr;
        std::unordered_map<int64_t, PhysicalBlock> blocks;
        int64_t current_pos = 0;
    };

    virtual int64_t resolve_block_id(int layer, int64_t logical_pos) const = 0;
    virtual int64_t alloc_block_id(int layer, int64_t logical_pos) = 0;
    virtual int64_t max_logical_tokens_per_layer() const = 0;
    virtual void* create_root() const = 0;
    virtual void destroy_root(void* root) const = 0;
    virtual const char* disk_name() const = 0;

    int num_layers_;
    int64_t num_heads_;
    int64_t head_dim_;
    int64_t block_size_;
    size_t physical_memory_limit_;
    mutable size_t current_memory_used_;
    std::string disk_path_;
    std::vector<LayerState> layers_;
    mutable int64_t access_counter_;
    int64_t next_block_id_;

    void evict_lru(int layer) const;
    void evict_to_disk(int layer, int64_t block_id) const;
    void load_from_disk(int layer, int64_t block_id) const;
    // Internal: caller MUST hold async_mtx_ (the mutex is non-recursive;
    // nested locking is UB — MSVC aborts 0xc0000409). Public wrappers above
    // lock + delegate; internal callers (evict_lru, alloc paths, flush,
    // load_from_disk's own evict loop) use these.
    void evict_lru_locked(int layer) const;
    void evict_to_disk_locked(int layer, int64_t block_id) const;
    void load_from_disk_locked(int layer, int64_t block_id) const;
    std::string block_disk_path(int layer, int64_t block_id) const;
    int64_t tokens_per_block() const;

    // Async worker plumbing. All of it is guarded by async_mtx_; async_cv_ wakes
    // the worker, async_done_cv_ wakes callers waiting in ensure_resident().
    struct AsyncRequest {
        int layer = -1;
        int64_t block_id = -1;
        bool evict = false;
    };
    void async_worker_loop();
    bool block_is_resident(int layer, int64_t block_id) const;
    bool enqueue_async(int layer, int64_t block_id, bool evict) const;

    mutable std::mutex async_mtx_;
    mutable std::condition_variable async_cv_;
    mutable std::condition_variable async_done_cv_;
    mutable std::deque<AsyncRequest> async_queue_;
    mutable std::set<std::pair<int, int64_t>> async_inflight_;
    mutable bool async_stop_ = false;
    mutable uint64_t async_loads_ = 0;
    std::thread async_worker_;
};

// ===========================================================================
// PagedKVCache4M — 4M context via 2-level hierarchical paging
//
// Logical token space: up to 2^22 (~4M) tokens, configurable
// Hierarchical paging: L1(4096) -> L2(4096) -> physical block
// Capacity = 4096^2 * block_size = 268M tokens (block_size=16)
// ===========================================================================

class PagedKVCache4M : public PagedKVCacheBase {
public:
    static constexpr int64_t TABLE_ENTRIES = 4096;
    static constexpr int64_t DEFAULT_BLOCK_SIZE = 16;
    static constexpr int64_t MAX_LOGICAL_TOKENS = (int64_t)1 << 22;

    PagedKVCache4M(int num_layers, int64_t num_heads, int64_t head_dim,
                   int64_t block_size = DEFAULT_BLOCK_SIZE,
                   size_t physical_memory_bytes = 8ULL * 1024 * 1024 * 1024,
                   const std::string& disk_path = "");
    ~PagedKVCache4M() override;

    std::pair<Tensor, Tensor> get_block(int layer, int64_t logical_pos) const;

    bool verify_retrieval(int layer, int64_t pos, const Tensor& expected_k,
                          const Tensor& expected_v) const;

    int64_t max_logical_tokens_per_layer() const override;

private:
    struct L2Table {
        int64_t entries[TABLE_ENTRIES];
        L2Table();
    };

    struct L1Table {
        L2Table* entries[TABLE_ENTRIES];
        L1Table();
        ~L1Table();
    };

    int64_t resolve_block_id(int layer, int64_t logical_pos) const override;
    int64_t alloc_block_id(int layer, int64_t logical_pos) override;
    void* create_root() const override;
    void destroy_root(void* root) const override;
    const char* disk_name() const override;
};

// ===========================================================================
// PagedKVCache1T — 1T+ logical tokens via 3-level hierarchical paging
//
// Like PagedKVCache4M but with an extra page table level:
//   L1(4096) -> L2(4096) -> L3(4096) -> physical block
//   Capacity = 4096^3 * block_size = 1T tokens (block_size=16)
// ===========================================================================

class PagedKVCache1T : public PagedKVCacheBase {
public:
    static constexpr int64_t TABLE_ENTRIES = 4096;
    static constexpr int64_t DEFAULT_BLOCK_SIZE = 16;
    static constexpr int64_t MAX_LOGICAL_TOKENS = (int64_t)1 << 40;
    static constexpr int64_t MIN_LOGICAL_TOKENS = (int64_t)1 << 40;

    PagedKVCache1T(int num_layers, int64_t num_heads, int64_t head_dim,
                   int64_t block_size = DEFAULT_BLOCK_SIZE,
                   size_t physical_memory_bytes = 8ULL * 1024 * 1024 * 1024,
                   const std::string& disk_path = "");
    ~PagedKVCache1T() override;

    int64_t max_logical_tokens_per_layer() const override;

private:
    struct L3Table {
        int64_t entries[TABLE_ENTRIES];
        L3Table();
    };

    struct L2Table {
        L3Table* entries[TABLE_ENTRIES];
        L2Table();
        ~L2Table();
    };

    struct L1Table {
        L2Table* entries[TABLE_ENTRIES];
        L1Table();
        ~L1Table();
    };

    int64_t resolve_block_id(int layer, int64_t logical_pos) const override;
    int64_t alloc_block_id(int layer, int64_t logical_pos) override;
    void* create_root() const override;
    void destroy_root(void* root) const override;
    const char* disk_name() const override;
};

// ===========================================================================
// PagedKVCache — convenience facade over the hierarchical paged engine
//
// Wraps PagedKVCache4M behind a default-constructible + init() API.
// ===========================================================================

class PagedKVCache {
public:
    PagedKVCache() = default;
    ~PagedKVCache() = default;

    PagedKVCache(const PagedKVCache&) = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;

    void init(int num_layers, int64_t max_seq_len, int64_t num_heads,
              int64_t head_dim);

    int64_t logical_capacity() const;
    int64_t max_seq_len() const { return max_seq_len_; }

private:
    std::unique_ptr<PagedKVCache4M> cache_;
    int64_t max_seq_len_ = 0;
};

} // namespace quant