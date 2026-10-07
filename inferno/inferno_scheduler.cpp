#include "inferno_scheduler.h"
#include <algorithm>
#include <chrono>

namespace quant {

using Clock = std::chrono::high_resolution_clock;
using Microseconds = std::chrono::microseconds;

static int64_t now_us() {
    return std::chrono::duration_cast<Microseconds>(
        Clock::now().time_since_epoch()).count();
}

InfernoScheduler& InfernoScheduler::instance() {
    static InfernoScheduler s;
    return s;
}

void InfernoScheduler::submit(InfernoTask task) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    queue_.push_back(std::move(task));
}

void InfernoScheduler::process_queue() {
    std::vector<InfernoTask> local;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (queue_.empty()) return;
        local.swap(queue_);
    }

    std::sort(local.begin(), local.end(),
              [](const InfernoTask& a, const InfernoTask& b) {
                  return static_cast<uint8_t>(a.priority) >
                         static_cast<uint8_t>(b.priority);
              });

    if (start_time_us_.load(std::memory_order_relaxed) == 0) {
        start_time_us_.store(now_us(), std::memory_order_relaxed);
    }

    for (auto& t : local) {
        if (t.work) t.work();
        total_ops_.fetch_add(t.estimated_ops, std::memory_order_relaxed);
        total_tasks_.fetch_add(1, std::memory_order_relaxed);
    }
}

void InfernoScheduler::clear() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    queue_.clear();
}

double InfernoScheduler::throughput_flops() const {
    int64_t start = start_time_us_.load(std::memory_order_relaxed);
    if (start == 0) return 0.0;
    int64_t elapsed_us = now_us() - start;
    if (elapsed_us <= 0) return 0.0;
    double elapsed_s = (double)elapsed_us / 1e6;
    double ops = (double)total_ops_.load(std::memory_order_relaxed);
    return ops * 2.0 / elapsed_s;
}

double InfernoScheduler::throughput_inferno() const {
    return throughput_flops() / 2.0 / 1e21;
}

size_t InfernoScheduler::queue_size() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return queue_.size();
}

} // namespace quant
