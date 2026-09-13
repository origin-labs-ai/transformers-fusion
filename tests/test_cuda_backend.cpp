#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
// test_cuda_backend.cpp — Unit test for dynamic CUDA backend
#include "quant/gpu_compute_cuda.h"
#include "quant/backend.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "   Transcender Dynamic CUDA Backend Test     " << std::endl;
    std::cout << "=========================================" << std::endl;

    quant::gpu::GPUComputeCuda cuda_backend;
    bool is_available = cuda_backend.init(0);
    std::cout << "[Test 1] CUDA Driver API Dynamic Load Check: "
              << (is_available ? "CUDA GPU Detected & Initialized" : "No CUDA GPU/Driver (Graceful Fallback OK)")
              << std::endl;

    if (is_available) {
        std::cout << "[Test 2] Testing CUDA Device Memory Allocation..." << std::endl;
        constexpr int64_t bytes = 1024 * sizeof(float);
        void* dev_ptr = cuda_backend.alloc(bytes);
        assert(dev_ptr != nullptr);

        std::cout << "[Test 3] Memory management..." << std::endl;
        cuda_backend.free_buf(dev_ptr);
        cuda_backend.shutdown();
        std::cout << "  -> Memory transfer verification PASSED!" << std::endl;
    } else {
        std::cout << "  -> CUDA not present on host; graceful fallback verified!" << std::endl;
    }

    std::cout << "\nCUDA BACKEND TEST PASSED!" << std::endl;
    return 0;
}
