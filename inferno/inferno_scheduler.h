#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>
#include <atomic>
#include <vector>
#include <mutex>

namespace quant {

struct InfernoTask {
    enum class Priority : uint8_t { LOW = 0, NORMAL = 1, HIGH = 2, CRITICAL = 3 };
    std::function<void()> work;
    Priority priority = Priority::NORMAL;
    int64_t estimated_ops = 0;
};

class InfernoScheduler {
public:
    static InfernoScheduler& instance();

    void submit(InfernoTask task);
    void process_queue();
    void clear();

    int64_t total_ops_completed() const { return total_ops_.load(std::memory_order_relaxed); }
    int64_t total_tasks_completed() const { return total_tasks_.load(std::memory_order_relaxed); }
    double throughput_flops() const;
    double throughput_inferno() const;

    size_t queue_size() const;

private:
    InfernoScheduler() = default;
    std::vector<InfernoTask> queue_;
    mutable std::mutex queue_mutex_;
    std::atomic<int64_t> total_ops_{0};
    std::atomic<int64_t> total_tasks_{0};
    std::atomic<int64_t> start_time_us_{0};
};

} // namespace quant
