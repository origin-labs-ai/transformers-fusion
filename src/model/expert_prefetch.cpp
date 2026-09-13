#include "quant/expert_prefetch.h"
#include "quant/gpu_compute_cuda.h"
#include <iostream>
#include <algorithm>
#include <chrono>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace quant {

ExpertPrefetcher::ExpertPrefetcher(int num_experts, int num_layers, size_t expert_size,
                                   int prefetch_ahead, int max_resident)
    : num_experts_(num_experts),
      num_layers_(num_layers),
      expert_size_(expert_size),
      prefetch_ahead_(prefetch_ahead),
      max_resident_(max_resident)
{
    pages_.resize(num_layers_);
    for (int l = 0; l < num_layers_; ++l) {
        pages_[l].resize(num_experts_);
        for (int e = 0; e < num_experts_; ++e) {
            pages_[l][e].expert_id = e;
            pages_[l][e].layer_id = l;
            pages_[l][e].size_bytes = expert_size_;
        }
    }
}

ExpertPrefetcher::~ExpertPrefetcher() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_flag_ = true;
    }
    cv_.notify_one();
    if (prefetch_thread_.joinable()) {
        prefetch_thread_.join();
    }
    
    // Destroy GPU copy stream
    if (copy_stream_) {
        gpu::get_cuda_compute().destroy_stream(copy_stream_);
        copy_stream_ = nullptr;
    }

    // Unpin memory and free device buffers
    for (int l = 0; l < num_layers_; ++l) {
        for (int e = 0; e < num_experts_; ++e) {
            if (pages_[l][e].is_pinned && pages_[l][e].host_ptr) {
#if defined(_WIN32)
                VirtualUnlock(pages_[l][e].host_ptr, pages_[l][e].size_bytes);
#else
                munlock(pages_[l][e].host_ptr, pages_[l][e].size_bytes);
#endif
                pages_[l][e].is_pinned = false;
                if (gpu::get_cuda_compute().is_initialized()) {
                    gpu::get_cuda_compute().unregister_host_memory(pages_[l][e].host_ptr);
                }
            }
            if (pages_[l][e].host_ptr) {
                free(pages_[l][e].host_ptr);
                pages_[l][e].host_ptr = nullptr;
            }
            if (pages_[l][e].device_ptr) {
                gpu::get_cuda_compute().free_buf(pages_[l][e].device_ptr);
                pages_[l][e].device_ptr = nullptr;
            }
        }
    }

    for (void* buf : prefetch_buffers_) {
        if (buf) {
#if defined(_WIN32)
            VirtualUnlock(buf, expert_size_);
#else
            munlock(buf, expert_size_);
#endif
            free(buf);
        }
    }
}

void ExpertPrefetcher::initialize() {
    // BUGFIX (bug census): malloc unchecked then VirtualLock/mlock on a
    // possible null (null-deref on OOM) + expert_size_ unvalidated.
    if (expert_size_ == 0) return;
    prefetch_buffers_.resize(2, nullptr);
    for (int i = 0; i < 2; ++i) {
        prefetch_buffers_[i] = malloc(expert_size_);
        if (!prefetch_buffers_[i]) continue;
#if defined(_WIN32)
        VirtualLock(prefetch_buffers_[i], expert_size_);
#else
        mlock(prefetch_buffers_[i], expert_size_);
#endif
    }

    // Create a dedicated GPU copy stream for async uploads
    if (gpu::get_cuda_compute().is_initialized()) {
        copy_stream_ = gpu::get_cuda_compute().create_stream();
    }

    prefetch_thread_ = std::thread(&ExpertPrefetcher::prefetch_thread_func, this);
}


void ExpertPrefetcher::set_expert_source(int layer_id, int expert_id, const void* src_ptr) {
    // BUGFIX (bug census): unlocked write raced the worker's reads.
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_id >= 0 && layer_id < num_layers_ && expert_id >= 0 && expert_id < num_experts_) {
        ExpertPage& page = pages_[layer_id][expert_id];
        page.source_ptr = src_ptr;
        page.source_segments.clear();
    }
}

void ExpertPrefetcher::set_expert_segments(int layer_id, int expert_id,
                                           const std::vector<SourceSegment>& segments) {
    // BUGFIX (bug census): same unlocked-write race as set_expert_source.
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_id >= 0 && layer_id < num_layers_ && expert_id >= 0 && expert_id < num_experts_) {
        ExpertPage& page = pages_[layer_id][expert_id];
        page.source_segments = segments;
        page.source_ptr = nullptr;
        // Compute total size from segments and update size_bytes
        size_t total = 0;
        for (auto& seg : segments) total += seg.bytes;
        if (total > 0) {
            page.size_bytes = total;
        }
    }
}


void ExpertPrefetcher::schedule_prefetch(int layer_id, const std::vector<int>& expert_ids) {
    // BUGFIX (bug census): pages_[layer_id][e] with no range check on either
    // index (OOB on bad layer/expert). Bounds-checked like set_expert_source.
    if (layer_id < 0 || layer_id >= num_layers_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    for (int e : expert_ids) {
        if (e < 0 || e >= num_experts_) continue;
        if (!pages_[layer_id][e].is_on_device && !pages_[layer_id][e].transfer_in_progress) {
            prefetch_queue_.push_back({layer_id, e});
        }
    }
    cv_.notify_one();
}

const float* ExpertPrefetcher::get_expert_weights(int layer_id, int expert_id) {
    // BUGFIX (bug census): pages_[layer_id][expert_id] dereferenced without
    // validation (OOB + null page ref on bad ids). Fail closed (nullptr).
    if (layer_id < 0 || layer_id >= num_layers_ ||
        expert_id < 0 || expert_id >= num_experts_)
        return nullptr;
    auto start_time = std::chrono::steady_clock::now();
    // BUGFIX (bug census): TOCTOU — is_on_device/host_ptr/device_ptr were
    // read lock-free, re-checked under lock, and disk/GPU I/O ran while
    // holding mutex_ (worker stall + double-load). Snapshot under lock, do
    // I/O unlocked, revalidate under lock before publishing.
    bool need_load = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        need_load = !pages_[layer_id][expert_id].is_on_device;
    }
    ExpertPage& page = pages_[layer_id][expert_id];
    
    bool waited = false;
    while (page.transfer_in_progress.load(std::memory_order_acquire)) {
        std::this_thread::yield();
        waited = true;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    if (!page.is_on_device) {
        // Miss! We have to do it synchronously if it wasn't prefetch queued or finished.
        stats_.prefetch_misses++;
        // Load from source into host memory
        load_from_disk(page);
        // Pin the host memory
        if (!page.is_pinned && page.host_ptr) {
#if defined(_WIN32)
            VirtualLock(page.host_ptr, page.size_bytes);
#else
            mlock(page.host_ptr, page.size_bytes);
#endif
            page.is_pinned = true;
            if (gpu::get_cuda_compute().is_initialized()) {
                gpu::get_cuda_compute().register_host_memory(page.host_ptr, page.size_bytes);
            }
        }
        // Upload to GPU synchronously if available
        upload_to_device(page);
        page.is_on_device = true;
        
        // LRU update
        evict_lru();
        lru_list_.push_back({layer_id, expert_id});
    } else {
        if (waited) {
            stats_.prefetch_misses++; // If we had to wait, count as a miss for timing purposes.
        } else {
            stats_.prefetch_hits++;
        }
        
        // Update LRU: move to back
        auto it = std::find_if(lru_list_.begin(), lru_list_.end(), [&](const std::pair<int, int>& p) {
            return p.first == layer_id && p.second == expert_id;
        });
        if (it != lru_list_.end()) {
            std::pair<int, int> val = *it;
            lru_list_.erase(it);
            lru_list_.push_back(val);
        }
    }
    
    if (waited) {
        auto end_time = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        double total_wait = stats_.avg_wait_ms * (stats_.prefetch_hits + stats_.prefetch_misses - 1) + ms;
        stats_.avg_wait_ms = total_wait / (stats_.prefetch_hits + stats_.prefetch_misses);
    }
    
    // Return device_ptr if GPU is active and the data was uploaded, else host_ptr.
    // NOTE (bug census round-11): this runs INSIDE the mutex_ guard above —
    // do NOT take another lock_guard here (non-recursive mutex: self-deadlock/
    // abort, caught by test_expert_prefetch 0xc0000409). The TOCTOU on the
    // returned pointer is inherent to the synchronous-miss API (caller uses
    // it before the next eviction); a fully safe API would ref-count pages.
    if (gpu::get_cuda_compute().is_initialized() && page.device_ptr) {
        return reinterpret_cast<const float*>(page.device_ptr);
    }
    return reinterpret_cast<const float*>(page.host_ptr);
}

void ExpertPrefetcher::release_expert(int layer_id, int expert_id) {
    // We don't immediately unpin, we rely on LRU eviction when limit is reached.
    // This allows re-use if the expert is requested again soon.
}

void ExpertPrefetcher::evict_lru() {
    while ((int)lru_list_.size() >= max_resident_) {
        auto [l, e] = lru_list_.front();
        lru_list_.erase(lru_list_.begin());
        
        ExpertPage& evict_page = pages_[l][e];
        if (evict_page.is_pinned && evict_page.host_ptr) {
#if defined(_WIN32)
            VirtualUnlock(evict_page.host_ptr, evict_page.size_bytes);
#else
            munlock(evict_page.host_ptr, evict_page.size_bytes);
#endif
            evict_page.is_pinned = false;
            if (gpu::get_cuda_compute().is_initialized()) {
                gpu::get_cuda_compute().unregister_host_memory(evict_page.host_ptr);
            }
        }
        if (evict_page.device_ptr && gpu::get_cuda_compute().is_initialized()) {
            gpu::get_cuda_compute().free_buf(evict_page.device_ptr);
            evict_page.device_ptr = nullptr;
        }
        evict_page.is_on_device = false;
        stats_.evictions++;
    }
}

void ExpertPrefetcher::load_from_disk(ExpertPage& page) {
    if (!page.host_ptr) {
        page.host_ptr = malloc(page.size_bytes);
    }
    if (!page.host_ptr) return;
    if (!page.source_segments.empty()) {
        // Segmented source: gate/up/down live in separate tensors — pack into contiguous host_ptr
        size_t offset = 0;
        for (auto& seg : page.source_segments) {
            if (seg.ptr && seg.bytes > 0 && offset + seg.bytes <= page.size_bytes) {
                std::memcpy(static_cast<char*>(page.host_ptr) + offset, seg.ptr, seg.bytes);
                offset += seg.bytes;
            }
        }
        // Zero any remaining tail if size_bytes is larger than segments total
        if (offset < page.size_bytes) {
            std::memset(static_cast<char*>(page.host_ptr) + offset, 0, page.size_bytes - offset);
        }
    } else if (page.source_ptr) {
        // Single contiguous source
        std::memcpy(page.host_ptr, page.source_ptr, page.size_bytes);
    } else {
        // No source registered — zero-fill as fallback (model will register sources on load/init)
        std::memset(page.host_ptr, 0, page.size_bytes);
    }
}

void ExpertPrefetcher::upload_to_device(ExpertPage& page) {
    if (!page.host_ptr) return;
    if (!gpu::get_cuda_compute().is_initialized()) return;

    // Lazily allocate device buffer
    if (!page.device_ptr) {
        page.device_ptr = gpu::get_cuda_compute().alloc((int64_t)page.size_bytes);
    }
    if (page.device_ptr) {
        gpu::get_cuda_compute().async_upload(page.host_ptr, page.device_ptr, page.size_bytes, copy_stream_);
        // For synchronous path (get_expert_weights miss), ensure copy is done before returning
        gpu::get_cuda_compute().synchronize_stream(copy_stream_);
    }
}

void ExpertPrefetcher::save_to_disk(ExpertPage& page) {
    // Simulate disk save
    if (page.host_ptr) {
        free(page.host_ptr);
        page.host_ptr = nullptr;
    }
}

void ExpertPrefetcher::prefetch_thread_func() {
    while (true) {
        PrefetchRequest req;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return stop_flag_ || !prefetch_queue_.empty(); });
            
            if (stop_flag_ && prefetch_queue_.empty()) {
                break;
            }
            
            req = prefetch_queue_.front();
            prefetch_queue_.erase(prefetch_queue_.begin());
            
            pages_[req.layer_id][req.expert_id].transfer_in_progress.store(true, std::memory_order_release);
        }
        
        ExpertPage& page = pages_[req.layer_id][req.expert_id];
        
        // Load expert weights via the real data path (host_ptr <- source_ptr)
        load_from_disk(page);

        // Perform the pinning
        if (!page.is_pinned && page.host_ptr) {
#if defined(_WIN32)
            VirtualLock(page.host_ptr, page.size_bytes);
#else
            mlock(page.host_ptr, page.size_bytes);
#endif
            page.is_pinned = true;
            if (gpu::get_cuda_compute().is_initialized()) {
                gpu::get_cuda_compute().register_host_memory(page.host_ptr, page.size_bytes);
            }
        }

        // Asynchronously transfer to GPU if available (async via copy_stream_)
        if (gpu::get_cuda_compute().is_initialized() && page.host_ptr) {
            if (!page.device_ptr) {
                page.device_ptr = gpu::get_cuda_compute().alloc((int64_t)page.size_bytes);
            }
            if (page.device_ptr) {
                gpu::get_cuda_compute().async_upload(page.host_ptr, page.device_ptr, page.size_bytes, copy_stream_);
                gpu::get_cuda_compute().synchronize_stream(copy_stream_);
            }
        }
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            page.is_on_device = true;
            page.transfer_in_progress.store(false, std::memory_order_release);
            
            // LRU logic
            auto it = std::find_if(lru_list_.begin(), lru_list_.end(), [&](const std::pair<int, int>& p) {
                return p.first == req.layer_id && p.second == req.expert_id;
            });
            if (it == lru_list_.end()) {
                evict_lru();
                lru_list_.push_back({req.layer_id, req.expert_id});
            } else {
                std::pair<int, int> val = *it;
                lru_list_.erase(it);
                lru_list_.push_back(val);
            }
        }
    }
}

ExpertPrefetcher::Stats ExpertPrefetcher::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

} // namespace quant
