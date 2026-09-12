#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
// test_expert_prefetch.cpp — Unit test for MoE Expert Prefetching and Memory Offloading
#include "quant/expert_prefetch.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <thread>
#include <chrono>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "  Transcender MoE Expert Prefetching Test    " << std::endl;
    std::cout << "=========================================" << std::endl;

    constexpr int num_experts = 8;
    constexpr int num_layers = 4;
    constexpr size_t expert_size = 1024 * sizeof(float); // 4KB per expert

    std::cout << "[Test 1] Initializing ExpertPrefetcher with 8 experts, 4 layers..." << std::endl;
    quant::ExpertPrefetcher prefetcher(num_experts, num_layers, expert_size, 1, 4);
    prefetcher.initialize();

    std::cout << "[Test 2] Scheduling async prefetch for Layer 0 experts {1, 3}..." << std::endl;
    prefetcher.schedule_prefetch(0, {1, 3});

    // BUGFIX (bug census): fixed 20ms sleep raced the prefetch thread
    // (flaky). Poll stats with a deadline instead — get_expert_weights is
    // synchronous on miss anyway, so this only shortens the happy path.
    for (int i = 0; i < 100; i++) {
        auto s = prefetcher.get_stats();
        if (s.prefetch_hits + s.prefetch_misses > 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    std::cout << "[Test 3] Fetching expert weights for Layer 0, Expert 1..." << std::endl;
    const float* weights_1 = prefetcher.get_expert_weights(0, 1);
    assert(weights_1 != nullptr);

    std::cout << "[Test 4] Releasing expert 1 and checking stats..." << std::endl;
    prefetcher.release_expert(0, 1);

    auto stats = prefetcher.get_stats();
    std::cout << "  -> Prefetch hits: " << stats.prefetch_hits << std::endl;
    std::cout << "  -> Prefetch misses: " << stats.prefetch_misses << std::endl;
    std::cout << "  -> Evictions: " << stats.evictions << std::endl;

    std::cout << "\nEXPERT PREFETCHER TEST PASSED!" << std::endl;
    return 0;
}
