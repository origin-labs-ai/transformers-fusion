#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
// test_bench.cpp — Benchmark hardware prober and backend micro-benchmarks
#include "quant/backend.h"
#include <iostream>
#include <iomanip>
#include <cassert>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "    Transcender Hardware Probe & Benchmark    " << std::endl;
    std::cout << "=========================================" << std::endl;

    std::cout << "[Test 1] Probing system hardware profile..." << std::endl;
    auto hw = quant::backend::probe_hardware();
    std::cout << "  -> CPU Cores: " << hw.cpu_cores << ", Threads: " << hw.cpu_threads << std::endl;
    std::cout << "  -> System RAM Free: " << (hw.ram_free / (1024 * 1024)) << " MB / "
              << (hw.ram_total / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  -> AVX2 Supported: " << (hw.has_avx2 ? "Yes" : "No") << std::endl;
    std::cout << "  -> CUDA Available: " << (hw.has_cuda ? "Yes" : "No") << std::endl;

    std::cout << "[Test 2] Running CPU backend GEMM micro-benchmark..." << std::endl;
    auto cfg = quant::backend::auto_select_backend(0);
    std::unique_ptr<quant::backend::ComputeBackend> backend(quant::backend::ComputeBackend::create(cfg));
    assert(backend != nullptr);

    double gflops = quant::backend::benchmark_operation(backend.get(), "gemm", 512, 512, 512, 2, 10);
    std::cout << "  -> Performance on 512x512x512 GEMM: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;

    std::cout << "\nHARDWARE BENCHMARK COMPLETED SUCCESSFULLY!" << std::endl;
    return 0;
}
