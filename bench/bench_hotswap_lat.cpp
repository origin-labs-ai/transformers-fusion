// P23 bench — 0ms hot-swap micro-bench (UNVERIFIED until run on shipping HW).
// Measures pointer-swap latency (the swap itself), NOT end-to-end request
// latency. Procedure: build Release, run 7 timed reps after 3 warmup, report
// median ns + p50/p99. Any "0ms" claim requires this output attached.
// Orchestrator wiring: add_executable(bench_p1p73_hotswap p1p73/bench/bench_p1p73_hotswap.cpp)
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

struct Block {
    std::vector<float> w;
    uint64_t version = 0, token = 0;
};

int main(int argc, char** argv) {
    int warmup = 3, reps = 7;
    if (argc > 1) reps = std::max(3, std::atoi(argv[1]));
    auto b0 = std::make_shared<const Block>();
    auto b1 = std::make_shared<const Block>();
    std::shared_ptr<const Block> active = b0;

    auto swap_once = [&]() {
        auto t0 = std::chrono::steady_clock::now();
        active = b1;  // the swap: single pointer store
        std::atomic_thread_fence(std::memory_order_seq_cst);
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    };

    for (int i = 0; i < warmup; ++i) swap_once();
    std::vector<long long> ns;
    for (int i = 0; i < reps; ++i) ns.push_back(swap_once());
    std::sort(ns.begin(), ns.end());
    long long median = ns[ns.size() / 2];
    long long p99 = ns[(ns.size() * 99) / 100 >= ns.size() ? ns.size() - 1 : (ns.size() * 99) / 100];

    std::printf("bench=hotswap_pointer_swap warmup=%d reps=%d median_ns=%lld p99_ns=%lld\n",
                warmup, reps, median, p99);
    std::printf("STATUS=UNVERIFIED (attach commit hash + dataset hash + machine spec to claim)\n");
    return 0;
}
