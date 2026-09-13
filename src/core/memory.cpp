#include "quant/memory.h"
#include <cstdlib>
#include <cstring>

namespace quant {

// MemoryPool and StackAllocator are primarily inline in memory.h.
// This file provides additional utility functions and thread-local support.

static thread_local MemoryPool* tls_pool_ = nullptr;

MemoryPool* get_tls_memory_pool(size_t pool_size = 1024 * 1024) {
    if (!tls_pool_) {
        tls_pool_ = ThreadLocalPoolRegistry::instance().get_or_create_pool(pool_size);
    }
    return tls_pool_;
}

void release_tls_memory_pool() {
    ThreadLocalPoolRegistry::instance().release_pool();
    tls_pool_ = nullptr;
}

void* tls_allocate(size_t bytes, size_t alignment) {
    MemoryPool* pool = get_tls_memory_pool();
    return pool ? pool->allocate(bytes, alignment) : nullptr;
}

void tls_reset_pool() {
    MemoryPool* pool = get_tls_memory_pool();
    if (pool) pool->reset();
}

} // namespace quant
