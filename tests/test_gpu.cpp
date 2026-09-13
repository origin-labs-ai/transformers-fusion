#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
// test_gpu.cpp — Unit test for Transcender GPU backend (CUDA, Vulkan, DX12)
#include "quant/backend.h"
#include "quant/gpu_compute_cuda.h"
#include <iostream>
#include <memory>
#include <cassert>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "      Transcender GPU Backend Init Test       " << std::endl;
    std::cout << "=========================================" << std::endl;

    std::cout << "[Test 1] Testing Hardware Backend Prober & Auto-selection..." << std::endl;
    auto hw = quant::backend::probe_hardware();
    std::cout << "  -> Hardware Profile: RAM " << (hw.ram_free / (1024 * 1024)) << " MB free, CPU threads: " << hw.cpu_threads << std::endl;

    auto cfg = quant::backend::auto_select_backend(0);
    std::unique_ptr<quant::backend::ComputeBackend> backend(quant::backend::ComputeBackend::create(cfg));
    assert(backend != nullptr);
    std::cout << "  -> Selected Backend: " << backend->name() << std::endl;

    std::cout << "[Test 2] Testing Dynamic CUDA Driver Probing..." << std::endl;
    bool cuda_ok = quant::backend::is_cuda_available();
    std::cout << "  -> CUDA Available: " << (cuda_ok ? "Yes" : "No (Graceful Fallback OK)") << std::endl;

    std::cout << "\nGPU BACKEND TEST PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
