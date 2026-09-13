#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
#include "quant/kv_cache.h"
#include <iostream>
#include <cassert>
#include <thread>
#include <cmath>
#include <vector>
#include <filesystem>
#include <limits>
#include <system_error>

// J5 helpers: memory-invariant + empty-tensor predicates (ASAN-clean).
// resolve_block_id/evict_lru are protected, so J5 exercises them via the
// public surface: get_block/get_range return empty on resolve -1 (null root,
// unallocated L2, OOB layer/pos), tiny-limit appends force evict_lru.
namespace {
inline size_t j5_block_bytes(int64_t nh, int64_t bs, int64_t hd) {
    return (size_t)nh * (size_t)bs * (size_t)hd * 2 * sizeof(float);
}
inline bool j5_is_empty(const quant::Tensor& t) {
    // Default Tensor() has rank-0 shape (numel()==1) but null buffer/data,
    // so empty must be detected via buffer/data, not numel()==0 alone.
    if (t.buffer() == nullptr) return true;
    if (t.data() == nullptr) return true;
    if (t.numel() == 0) return true;
    return false;
}
inline void j5_mem_strong(const quant::PagedKVCache4M& c, const char* tag) {
    assert(c.num_disk_blocks() <= c.num_physical_blocks());
    assert(c.physical_memory_used() <= c.physical_memory_limit());
    size_t b = j5_block_bytes(c.num_heads(), c.block_size(), c.head_dim());
    if (b > 0) {
        int64_t resident = c.num_physical_blocks() - c.num_disk_blocks();
        assert(resident >= 0);
        assert(c.physical_memory_used() == (size_t)resident * b);
    }
    std::cout << tag << "-mem used=" << c.physical_memory_used()
              << " phys=" << c.num_physical_blocks()
              << " disk=" << c.num_disk_blocks() << std::endl;
}
inline void j5_mem_weak(const quant::PagedKVCache4M& c, const char* tag) {
    assert(c.num_disk_blocks() <= c.num_physical_blocks());
    if (c.physical_memory_limit() > 0)
        assert(c.physical_memory_used() <= c.physical_memory_limit());
    std::cout << tag << "-mem used=" << c.physical_memory_used()
              << " phys=" << c.num_physical_blocks()
              << " disk=" << c.num_disk_blocks() << std::endl;
}
}  // namespace

int main() {
    std::cout << "[Test] Running Paged KV Cache test..." << std::endl;

    const int64_t num_layers = 1;
    const int64_t num_heads = 8;
    const int64_t head_dim = 64;
    const int64_t block_size = 16;

    quant::PagedKVCache4M cache(num_layers, num_heads, head_dim, block_size,
                                256ULL * 1024 * 1024);

    // Capacity contract: L1 x L2 table product scaled by block size.
    // TABLE_ENTRIES was raised 512->4096 at some point without a test update;
    // assert the structural invariant rather than a stale magic constant.
    assert(cache.max_logical_tokens_per_layer() ==
           quant::PagedKVCache4M::TABLE_ENTRIES *
               quant::PagedKVCache4M::TABLE_ENTRIES * block_size);
    assert(cache.logical_capacity() == cache.max_logical_tokens_per_layer());
    assert(cache.block_size() == block_size);
    assert(cache.num_layers() == num_layers);
    assert(cache.num_heads() == num_heads);
    assert(cache.head_dim() == head_dim);

    // Write a block worth of K/V at the start of the logical space.
    int64_t pos0 = 0;
    quant::Tensor k0({num_heads, block_size, head_dim}, quant::DType::F32);
    quant::Tensor v0({num_heads, block_size, head_dim}, quant::DType::F32);
    float* kd = k0.data<float>();
    float* vd = v0.data<float>();
    for (int64_t i = 0; i < num_heads * block_size * head_dim; i++) {
        kd[i] = (float)(i % 7) * 0.25f;
        vd[i] = (float)(i % 5) * 0.5f;
    }
    std::cout << "S2" << std::endl;
    j5_mem_strong(cache, "S2");
    assert(cache.physical_memory_used() == 0);
    assert(cache.num_physical_blocks() == 0);
    assert(cache.num_disk_blocks() == 0);
    cache.append(0, pos0, k0, v0);
    std::cout << "S3" << std::endl;
    // J5 invariant: 1 resident block, none on disk, used == 1*block_bytes.
    j5_mem_strong(cache, "S3");
    assert(cache.num_physical_blocks() == 1);
    assert(cache.num_disk_blocks() == 0);
    assert(cache.physical_memory_used() ==
           j5_block_bytes(num_heads, block_size, head_dim));

    // Read it back and verify byte-identical roundtrip.
    auto got = cache.get_range(0, pos0, pos0 + block_size);
    std::cout << "S4" << std::endl;
    j5_mem_strong(cache, "S4");
    const quant::Tensor& rk = got.first;
    const quant::Tensor& rv = got.second;
    assert(rk.numel() == num_heads * block_size * head_dim);
    assert(rv.numel() == num_heads * block_size * head_dim);
    const float* rkd = rk.data<float>();
    const float* rvd = rv.data<float>();
    for (int64_t i = 0; i < rk.numel(); i++) {
        assert(std::abs(rkd[i] - kd[i]) < 1e-6f);
    }
    for (int64_t i = 0; i < rv.numel(); i++) {
        assert(std::abs(rvd[i] - vd[i]) < 1e-6f);
    }
    std::cout << "S5" << std::endl;
    j5_mem_strong(cache, "S5");
    assert(cache.verify_retrieval(0, pos0, k0, v0));
    std::cout << "S6" << std::endl;
    j5_mem_strong(cache, "S6");
    assert(cache.num_physical_blocks() == 1);
    assert(cache.num_disk_blocks() == 0);

    // Write to a far-away logical position (exercises hierarchical paging).
    int64_t far_pos = ((int64_t)1 << 20) * block_size;
    assert(cache.context_len() >= 0);
    std::cout << "S7" << std::endl;
    j5_mem_strong(cache, "S7");
    cache.append(0, far_pos, k0, v0);
    std::cout << "S8" << std::endl;
    // J5 invariant: 2 resident blocks under the 256MB limit, none evicted.
    j5_mem_strong(cache, "S8");
    assert(cache.num_physical_blocks() == 2);
    assert(cache.num_disk_blocks() == 0);
    assert(cache.physical_memory_used() ==
           2 * j5_block_bytes(num_heads, block_size, head_dim));
    auto got2 = cache.get_range(0, far_pos, far_pos + block_size);
    const quant::Tensor& rk2 = got2.first;
    std::cout << "S8b rk2.numel=" << rk2.numel() << " curpos=" << cache.context_len() << std::endl;
    j5_mem_strong(cache, "S8b");
    const float* rkd2 = rk2.data<float>();
    for (int64_t i = 0; i < rk2.numel(); i++) {
        assert(std::abs(rkd2[i] - kd[i]) < 1e-6f);
    }

    // Flush to disk and reload — contents must survive. NOTE: use
    // unload_memory(), not clear() — clear() is a full wipe (deletes disk
    // files, resets positions), so flush→clear→load can never restore. P19.
    std::cout << "S9" << std::endl;
    j5_mem_strong(cache, "S9");
    cache.flush_to_disk();
    cache.unload_memory();
    std::cout << "S11" << std::endl;
    // J5 invariant: after flush+unload all blocks on disk, memory == 0.
    j5_mem_strong(cache, "S11");
    assert(cache.num_physical_blocks() == 2);
    assert(cache.num_disk_blocks() == 2);
    assert(cache.physical_memory_used() == 0);
    cache.load_from_disk();
    std::cout << "S12" << std::endl;
    // J5 invariant: after reload all blocks resident again.
    j5_mem_strong(cache, "S12");
    assert(cache.num_physical_blocks() == 2);
    assert(cache.num_disk_blocks() == 0);
    assert(cache.physical_memory_used() ==
           2 * j5_block_bytes(num_heads, block_size, head_dim));
    auto got3 = cache.get_range(0, pos0, pos0 + block_size);
    std::cout << "S13" << std::endl;
    j5_mem_strong(cache, "S13");
    const quant::Tensor& rk3 = got3.first;
    const float* rkd3 = rk3.data<float>();
    std::cout << "S14 numel=" << rk3.numel() << std::endl;
    j5_mem_strong(cache, "S14");
    for (int64_t i = 0; i < rk3.numel(); i++) {
        assert(std::abs(rkd3[i] - kd[i]) < 1e-6f);
    }

    // ── C-20: async disk pipeline ─────────────────────────────────────────
    // Block loads must be owned by the background worker: prefetch() is
    // fire-and-forget, and the hot path (append) waits for the block instead of
    // doing fread inline. Nothing else touches the cache in this block, so a
    // completed load is unambiguous.
    {
        std::cout << "S15 async" << std::endl;
        assert(cache.async_worker_running());
        assert(cache.async_queue_depth() == 0);
        const uint64_t loads_before = cache.async_loads_completed();

        cache.flush_to_disk();
        cache.unload_memory();
        assert(cache.num_disk_blocks() == 2);
        assert(cache.physical_memory_used() == 0);

        cache.prefetch_range(0, pos0, pos0 + block_size);
        // Bounded spin: wait for the queue to drain rather than sleeping a
        // fixed amount, so the assertion is about the pipeline, not the clock.
        for (int spin = 0; spin < 200000 && cache.async_queue_depth() > 0; ++spin)
            std::this_thread::yield();
        assert(cache.async_queue_depth() == 0);
        assert(cache.async_loads_completed() > loads_before);
        assert(cache.num_disk_blocks() == 1);   // the prefetched block is back
        assert(cache.physical_memory_used() ==
               j5_block_bytes(num_heads, block_size, head_dim));

        // Contents must survive an async reload, byte-for-byte.
        auto got_async = cache.get_range(0, pos0, pos0 + block_size);
        const float* rkd4 = got_async.first.data<float>();
        assert(got_async.first.numel() == num_heads * block_size * head_dim);
        for (int64_t i = 0; i < got_async.first.numel(); i++) {
            assert(std::abs(rkd4[i] - kd[i]) < 1e-6f);
        }

        // append() must also ride the pipeline: park the block on disk again and
        // require append() to bring it back through ensure_resident().
        cache.flush_to_disk();
        cache.unload_memory();
        assert(cache.num_disk_blocks() == 2);
        cache.append(0, pos0, k0, v0);
        assert(cache.num_disk_blocks() == 1);
        assert(cache.physical_memory_used() ==
               j5_block_bytes(num_heads, block_size, head_dim));
        auto got_async2 = cache.get_range(0, pos0, pos0 + block_size);
        assert(!j5_is_empty(got_async2.first));
        const float* rkd5 = got_async2.first.data<float>();
        for (int64_t i = 0; i < got_async2.first.numel(); i++) {
            assert(std::abs(rkd5[i] - kd[i]) < 1e-6f);
        }
        std::cout << "S16 async ok, loads=" << cache.async_loads_completed()
                  << std::endl;
    }

    // ── J5: reload_missing_file (delete disk file → graceful, ASAN-clean) ──
    {
        std::cout << "J5-reload_missing_file" << std::endl;
        namespace fs = std::filesystem;
        fs::path j5dir;
        try {
            j5dir = fs::temp_directory_path() / "TransCender_j5_missing";
        } catch (...) {
            j5dir = fs::path("TransCender_j5_missing");
        }
        std::error_code ec;
        fs::remove_all(j5dir, ec);
        fs::create_directories(j5dir, ec);
        std::string j5path = j5dir.string();
        quant::PagedKVCache4M mc(1, num_heads, head_dim, block_size,
                                 256ULL * 1024 * 1024, j5path);
        mc.append(0, pos0, k0, v0);
        assert(mc.num_physical_blocks() == 1);
        mc.flush_to_disk();
        mc.unload_memory();
        assert(mc.num_disk_blocks() == 1);
        assert(mc.physical_memory_used() == 0);
        j5_mem_weak(mc, "J5-missing-predelete");
        // Delete the backing file(s) to simulate missing/corrupt disk.
        fs::remove_all(j5dir, ec);
        assert(!fs::exists(j5dir));
        // Must not throw/crash; void ABI reports via state (stays on_disk).
        mc.load_from_disk();
        j5_mem_weak(mc, "J5-missing-postload");
        assert(mc.physical_memory_used() == 0);
        assert(mc.num_disk_blocks() == 1);
        // get_block on missing file → graceful empty (resolve ok, load fails).
        auto gb = mc.get_block(0, pos0);
        assert(j5_is_empty(gb.first) && j5_is_empty(gb.second));
        // get_range on missing file → graceful: zeros (clamped len) or empty,
        // never original bytes, never OOB. ASAN-clean: null/numel guards.
        auto gr = mc.get_range(0, pos0, pos0 + block_size);
        if (j5_is_empty(gr.first)) {
            assert(j5_is_empty(gr.second));
        } else {
            assert(gr.first.numel() == num_heads * block_size * head_dim);
            const float* p = gr.first.data<float>();
            const float* q = gr.second.data<float>();
            assert(p != nullptr && q != nullptr);
            bool matches_orig = true;
            bool all_zero = true;
            for (int64_t i = 0; i < gr.first.numel(); i++) {
                if (p[i] != 0.0f) all_zero = false;
                if (std::abs(p[i] - kd[i]) >= 1e-6f) matches_orig = false;
            }
            // Missing backing store must not resurrect original bytes.
            assert(!matches_orig);
            (void)all_zero;
            (void)q;
        }
        // Second load still graceful (idempotent, no crash).
        mc.load_from_disk();
        j5_mem_weak(mc, "J5-missing-reload2");
        assert(mc.physical_memory_used() == 0);
        mc.clear();
        fs::remove_all(j5dir, ec);
        std::cout << "J5-reload_missing_file ok" << std::endl;
    }

    // ── J5: eviction_stress (tiny memory forcing evict_lru, roundtrip) ──
    {
        std::cout << "J5-eviction_stress" << std::endl;
        namespace fs = std::filesystem;
        fs::path edir;
        try {
            edir = fs::temp_directory_path() / "TransCender_j5_evict";
        } catch (...) {
            edir = fs::path("TransCender_j5_evict");
        }
        std::error_code ec;
        fs::remove_all(edir, ec);
        fs::create_directories(edir, ec);
        const int64_t eh = 4, ed = 16, eb = 8;
        const size_t ebb = j5_block_bytes(eh, eb, ed);
        const size_t elim = ebb * 2;  // room for 2 resident blocks
        quant::PagedKVCache4M xc(1, eh, ed, eb, elim, edir.string());
        assert(xc.physical_memory_limit() == elim);
        const int kNBlk = 6;  // > resident capacity → forces evict_lru
        std::vector<std::vector<float>> expK((size_t)kNBlk), expV((size_t)kNBlk);
        for (int b = 0; b < kNBlk; b++) {
            quant::Tensor kk({eh, eb, ed}, quant::DType::F32);
            quant::Tensor vv({eh, eb, ed}, quant::DType::F32);
            float* kpd = kk.data<float>();
            float* vpd = vv.data<float>();
            assert(kpd != nullptr && vpd != nullptr);
            assert(kk.numel() == eh * eb * ed);
            for (int64_t i = 0; i < kk.numel(); i++) {
                kpd[i] = (float)(b * 10007 + (i % 13)) * 0.25f;
                vpd[i] = (float)(b * 7919 + (i % 7)) * 0.5f;
            }
            expK[(size_t)b].assign(kpd, kpd + kk.numel());
            expV[(size_t)b].assign(vpd, vpd + vv.numel());
            xc.append(0, (int64_t)b * eb, kk, vv);
            // Per-append memory invariants (ASAN-clean, no OOB).
            assert(xc.physical_memory_used() <= xc.physical_memory_limit());
            assert(xc.num_disk_blocks() <= xc.num_physical_blocks());
            assert(xc.num_physical_blocks() == (int64_t)(b + 1));
        }
        assert(xc.num_physical_blocks() == kNBlk);
        assert(xc.num_disk_blocks() > 0);  // evict_lru actually fired
        assert(xc.physical_memory_used() <= elim);
        j5_mem_weak(xc, "J5-evict-post-append");
        // Byte-identical roundtrip across evicted + resident blocks.
        for (int b = 0; b < kNBlk; b++) {
            auto g = xc.get_range(0, (int64_t)b * eb, (int64_t)b * eb + eb);
            assert(!j5_is_empty(g.first) && !j5_is_empty(g.second));
            assert(g.first.numel() == eh * eb * ed);
            assert(g.second.numel() == eh * eb * ed);
            const float* rp = g.first.data<float>();
            const float* rq = g.second.data<float>();
            assert(rp != nullptr && rq != nullptr);
            for (int64_t i = 0; i < g.first.numel(); i++) {
                assert(std::abs(rp[i] - expK[(size_t)b][(size_t)i]) < 1e-6f);
            }
            for (int64_t i = 0; i < g.second.numel(); i++) {
                assert(std::abs(rq[i] - expV[(size_t)b][(size_t)i]) < 1e-6f);
            }
            assert(xc.physical_memory_used() <= xc.physical_memory_limit());
            assert(xc.num_disk_blocks() <= xc.num_physical_blocks());
        }
        j5_mem_weak(xc, "J5-evict-post-verify");
        xc.clear();
        assert(xc.num_physical_blocks() == 0);
        assert(xc.physical_memory_used() == 0);
        fs::remove_all(edir, ec);
        std::cout << "J5-eviction_stress ok" << std::endl;
    }

    // ── J5: resolve_null_root + out_of_range_layer (empty, no crash) ──
    {
        std::cout << "J5-resolve_null_root" << std::endl;
        // Fresh cache: L2 slots null, nothing allocated → resolve -1 path.
        quant::PagedKVCache4M fc(1, num_heads, head_dim, block_size,
                                 256ULL * 1024 * 1024);
        auto e1 = fc.get_range(0, 0, block_size);  // current_pos==0 → empty
        assert(j5_is_empty(e1.first) && j5_is_empty(e1.second));
        auto e2 = fc.get_block(0, 0);  // unallocated L2 → resolve -1 → empty
        assert(j5_is_empty(e2.first) && j5_is_empty(e2.second));
        auto e3 = fc.get_range(0, far_pos, far_pos + block_size);
        assert(j5_is_empty(e3.first) && j5_is_empty(e3.second));
        // verify_retrieval on unallocated pos → false, no crash/OOB.
        assert(!fc.verify_retrieval(0, 0, k0, v0));
        auto e4 = fc.get_block(0, far_pos);
        assert(j5_is_empty(e4.first) && j5_is_empty(e4.second));
        fc.clear();
        std::cout << "J5-resolve_null_root ok" << std::endl;

        std::cout << "J5-out_of_range_layer" << std::endl;
        quant::PagedKVCache4M oc(1, num_heads, head_dim, block_size,
                                 256ULL * 1024 * 1024);
        oc.append(0, pos0, k0, v0);
        // OOB layers → empty / false / no-op, never crash.
        auto o1 = oc.get_range(-1, pos0, pos0 + block_size);
        assert(j5_is_empty(o1.first) && j5_is_empty(o1.second));
        auto o2 = oc.get_range(1, pos0, pos0 + block_size);
        assert(j5_is_empty(o2.first) && j5_is_empty(o2.second));
        auto o3 = oc.get_range(std::numeric_limits<int>::max(), pos0, pos0 + block_size);
        assert(j5_is_empty(o3.first) && j5_is_empty(o3.second));
        auto b1 = oc.get_block(-1, pos0);
        assert(j5_is_empty(b1.first) && j5_is_empty(b1.second));
        auto b2 = oc.get_block(1, pos0);
        assert(j5_is_empty(b2.first) && j5_is_empty(b2.second));
        auto b3 = oc.get_block(1000000, pos0);
        assert(j5_is_empty(b3.first) && j5_is_empty(b3.second));
        oc.append(-1, pos0, k0, v0);
        oc.append(1, pos0, k0, v0);
        oc.append(std::numeric_limits<int>::max(), pos0, k0, v0);
        assert(!oc.verify_retrieval(-1, pos0, k0, v0));
        assert(!oc.verify_retrieval(1, pos0, k0, v0));
        // Negative / past-capacity positions → graceful empty, no crash.
        oc.append(0, -1, k0, v0);
        auto n1 = oc.get_block(0, -1);
        assert(j5_is_empty(n1.first) && j5_is_empty(n1.second));
        int64_t oob = oc.max_logical_tokens_per_layer() + 1024;
        auto ob1 = oc.get_block(0, oob);
        assert(j5_is_empty(ob1.first) && j5_is_empty(ob1.second));
        auto ob2 = oc.get_range(0, oob, oob + block_size);
        assert(j5_is_empty(ob2.first) && j5_is_empty(ob2.second));
        // Negative-start clamp path must not crash either.
        auto ng = oc.get_range(0, -100, block_size);
        assert(!j5_is_empty(ng.first));  // clamps to [0, len), valid block
        assert(oc.context_len() >= 0);
        j5_mem_weak(oc, "J5-oob");
        oc.clear();
        std::cout << "J5-out_of_range_layer ok" << std::endl;
    }

    cache.clear();
    std::cout << "Paged KV Cache test passed!" << std::endl;
    return 0;
}
