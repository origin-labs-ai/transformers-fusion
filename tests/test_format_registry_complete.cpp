// test_format_registry_complete.cpp — Complete unit test for FormatRegistry
#include "quant/format_registry.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << " FormatRegistry Complete System Test     " << std::endl;
    std::cout << "=========================================" << std::endl;

    std::cout << "[Test 1] Querying all registered single formats..." << std::endl;
    const auto& singles = quant::FormatRegistry::get_all_singles();
    std::cout << "  -> Total single formats registered: " << singles.size() << std::endl;
    assert(singles.size() >= 19);

    std::cout << "[Test 2] Querying mixes (TWI removed by design; four-tier MXQ)..." << std::endl;
    const auto& twi_mixes = quant::FormatRegistry::get_all_twi_mixes();
    const auto& four_mixes = quant::FormatRegistry::get_all_four_mixes();
    std::cout << "  -> Total TWI_MIX formats: " << twi_mixes.size() << " (removed by design)" << std::endl;
    std::cout << "  -> Total MXQ four-mix formats: " << four_mixes.size() << std::endl;
    assert(twi_mixes.empty());
    assert(four_mixes.size() >= 14);

    std::cout << "[Test 3] Testing BPW lookup for Q1 through Q32..." << std::endl;
    auto q1 = quant::FormatRegistry::get_single_format(1.0f);
    auto q4 = quant::FormatRegistry::get_single_format(4.0f);
    auto q8 = quant::FormatRegistry::get_single_format(8.0f);
    auto q32 = quant::FormatRegistry::get_single_format(32.0f);

    assert(q1.name == "Q1");
    assert(q4.name == "Q4");
    assert(q8.name == "Q8");
    assert(q32.name == "Q32");
    std::cout << "  -> PASSED: Single format lookup by BPW verified!" << std::endl;

    std::cout << "[Test 4] Testing quantize and dequantize roundtrip on single block..." << std::endl;
    constexpr int N = 256;
    std::vector<float> data(N);
    for (int i = 0; i < N; i++) data[i] = (float)i * 0.01f - 1.28f;

    auto qr = quant::FormatRegistry::quantize(data.data(), N, q8);
    assert(qr.success);

    std::vector<float> dequant(N);
    quant::FormatRegistry::dequantize(qr, dequant.data(), N);

    float mse = quant::FormatRegistry::measure_mse(data.data(), dequant.data(), N);
    std::cout << "  -> Q8 reconstruction MSE: " << mse << std::endl;
    assert(mse < 0.01f);
    std::cout << "  -> PASSED: Quantize & dequantize roundtrip verified!" << std::endl;

    std::cout << "[Test 5] Printing complete format table..." << std::endl;
    std::string table = quant::FormatRegistry::get_format_table();
    assert(!table.empty());
    std::cout << table << std::endl;

    std::cout << "\nFORMAT REGISTRY TEST COMPLETED SUCCESSFULLY!" << std::endl;
    return 0;
}
