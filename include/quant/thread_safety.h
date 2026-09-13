#pragma once
// ============================================================================
// thread_safety.h — Lock-Free & Thread-Safe Primitives for Transcender
// ============================================================================
// Provides:
//   1. SPSCQueue<T>  — Single-Producer Single-Consumer lock-free ring buffer
//   2. SharedMutex    — Read-write lock (shared_mutex wrapper)
//   3. AtomicFloat    — Lock-free atomic floating-point accumulator
//   4. SpinLock       — Lightweight spinlock for short critical sections
//   5. ThreadSafeCounter — Atomic counter with per-thread accumulation
// ============================================================================

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <new>
#include <shared_mutex>
#include <cstring>

namespace quant {

// ── SPSC Lock-Free Queue ──────────────────────────────────────────────────
// Bounded single-producer single-consumer ring buffer.
// No locks. No CAS loops. Cache-line padded to avoid false sharing.

template<typename T>
class SPSCQueue {
public:
    explicit SPSCQueue(size_t capacity)
        : capacity_(round_up_pow2(capacity))
        , mask_(capacity_ - 1)
        , buffer_(static_cast<T*>(::operator new(sizeof(T) * capacity_,
                    std::align_val_t{64})))
        , head_(0)
        , tail_(0) {}

    ~SPSCQueue() {
        while (!empty()) {
            T* ptr = buffer_ + (tail_ & mask_);
            ptr->~T();
            ++tail_;
        }
        ::operator delete(buffer_, std::align_val_t{64});
    }

    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;

    bool push(const T& item) {
        size_t h = head_.load(std::memory_order_relaxed);
        if (h - tail_.load(std::memory_order_acquire) >= capacity_) {
            return false;
        }
        new (buffer_ + (h & mask_)) T(item);
        head_.store(h + 1, std::memory_order_release);
        return true;
    }

    bool push(T&& item) {
        size_t h = head_.load(std::memory_order_relaxed);
        if (h - tail_.load(std::memory_order_acquire) >= capacity_) {
            return false;
        }
        new (buffer_ + (h & mask_)) T(std::move(item));
        head_.store(h + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        size_t t = tail_.load(std::memory_order_relaxed);
        if (t >= head_.load(std::memory_order_acquire)) {
            return false;
        }
        T* ptr = buffer_ + (t & mask_);
        item = std::move(*ptr);
        ptr->~T();
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }

    bool empty() const {
        return tail_.load(std::memory_order_acquire) >=
               head_.load(std::memory_order_acquire);
    }

    size_t size() const {
        return head_.load(std::memory_order_acquire) -
               tail_.load(std::memory_order_acquire);
    }

    size_t capacity() const { return capacity_; }

private:
    static size_t round_up_pow2(size_t v) {
        if (v == 0) return 1;
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        return v + 1;
    }

    const size_t capacity_;
    const size_t mask_;
    T* const buffer_;

    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
};

// ── SharedMutex (Read-Write Lock) ─────────────────────────────────────────
// Thin wrapper around std::shared_mutex for clarity.

class SharedMutex {
public:
    void lock_shared() { mtx_.lock_shared(); }
    void unlock_shared() { mtx_.unlock_shared(); }
    void lock() { mtx_.lock(); }
    void unlock() { mtx_.unlock(); }
    bool try_lock() { return mtx_.try_lock(); }
    bool try_lock_shared() { return mtx_.try_lock_shared(); }

private:
    std::shared_mutex mtx_;
};

// ── RAII Shared Lock Guard ────────────────────────────────────────────────

class SharedLockGuard {
public:
    explicit SharedLockGuard(SharedMutex& m) : mtx_(m) { mtx_.lock_shared(); }
    ~SharedLockGuard() { mtx_.unlock_shared(); }
    SharedLockGuard(const SharedLockGuard&) = delete;
    SharedLockGuard& operator=(const SharedLockGuard&) = delete;

private:
    SharedMutex& mtx_;
};

class UniqueLockGuard {
public:
    explicit UniqueLockGuard(SharedMutex& m) : mtx_(m) { mtx_.lock(); }
    ~UniqueLockGuard() { mtx_.unlock(); }
    UniqueLockGuard(const UniqueLockGuard&) = delete;
    UniqueLockGuard& operator=(const UniqueLockGuard&) = delete;

private:
    SharedMutex& mtx_;
};

// ── AtomicFloat — Lock-Free Float Accumulator ─────────────────────────────
// Uses compare_exchange on the underlying int64_t for lock-free float math.

class AtomicFloat {
public:
    AtomicFloat() = default;
    explicit AtomicFloat(float v) : val_(float_to_bits(v)) {}

    float load(std::memory_order mo = std::memory_order_seq_cst) const {
        return bits_to_float(val_.load(mo));
    }

    void store(float v, std::memory_order mo = std::memory_order_seq_cst) {
        val_.store(float_to_bits(v), mo);
    }

    float fetch_add(float v, std::memory_order mo = std::memory_order_seq_cst) {
        int64_t old_val = val_.load(mo);
        while (true) {
            float old_f = bits_to_float(old_val);
            float new_f = old_f + v;
            int64_t new_val = float_to_bits(new_f);
            if (val_.compare_exchange_weak(old_val, new_val, mo)) {
                return old_f;
            }
        }
    }

    float operator+=(float v) { return fetch_add(v) + v; }

    AtomicFloat(const AtomicFloat&) = delete;
    AtomicFloat& operator=(const AtomicFloat&) = delete;

private:
    static int64_t float_to_bits(float f) {
        int32_t i;
        std::memcpy(&i, &f, sizeof(f));
        return static_cast<int64_t>(i);
    }

    static float bits_to_float(int64_t bits) {
        int32_t i = static_cast<int32_t>(bits);
        float f;
        std::memcpy(&f, &i, sizeof(f));
        return f;
    }

    std::atomic<int64_t> val_{0};
};

// ── SpinLock ──────────────────────────────────────────────────────────────
// For very short critical sections where mutex overhead is too high.

class SpinLock {
public:
    void lock() noexcept {
        for (;;) {
            if (!flag_.exchange(true, std::memory_order_acquire)) {
                return;
            }
            while (flag_.load(std::memory_order_relaxed)) {
#if defined(_MSC_VER)
                _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
                __asm__ __volatile__("pause");
#elif defined(__aarch64__)
                __asm__ __volatile__("yield");
#endif
            }
        }
    }

    void unlock() noexcept {
        flag_.store(false, std::memory_order_release);
    }

    bool try_lock() noexcept {
        return !flag_.exchange(true, std::memory_order_acquire);
    }

private:
    std::atomic<bool> flag_{false};
};

class SpinLockGuard {
public:
    explicit SpinLockGuard(SpinLock& s) : lock_(s) { lock_.lock(); }
    ~SpinLockGuard() { lock_.unlock(); }
    SpinLockGuard(const SpinLockGuard&) = delete;
    SpinLockGuard& operator=(const SpinLockGuard&) = delete;

private:
    SpinLock& lock_;
};

// ── ThreadSafeCounter — Per-Thread Accumulated Counter ────────────────────
// Lock-free approximate counter using per-thread accumulators and periodic
// merge. Suitable for metrics that don't need exact real-time counts.

template<int NUM_THREADS = 64>
class ThreadSafeCounter {
public:
    void add(int thread_id, int64_t delta) {
        int idx = thread_id & (NUM_THREADS - 1);
        slots_[idx].fetch_add(delta, std::memory_order_relaxed);
    }

    int64_t total() const {
        int64_t sum = 0;
        for (int i = 0; i < NUM_THREADS; ++i) {
            sum += slots_[i].load(std::memory_order_relaxed);
        }
        return sum;
    }

    void reset() {
        for (int i = 0; i < NUM_THREADS; ++i) {
            slots_[i].store(0, std::memory_order_relaxed);
        }
    }

private:
    std::atomic<int64_t> slots_[NUM_THREADS] = {};
};

} // namespace quant
