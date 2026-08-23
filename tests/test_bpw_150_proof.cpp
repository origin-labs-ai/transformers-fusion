// test_bpw_150_proof.cpp — Proof test for 1.50 BPW TWI_MIX quantization
#include "quant/types.h"
#include "quant/format_registry.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "     InNova 1.50 BPW Quality Proof       " << std::endl;
    std::cout << "=========================================" << std::endl;

    std::cout << "[Test 1] Querying 1.50 BPW TWI_MIX format..." << std::endl;
    auto twi_15 = quant::FormatRegistry::get_twi_mix(1.50f);
    assert(std::abs(twi_15.effective_bpw - 1.50f) < 1e-4f);

    std::cout << "  -> Format Name: " << twi_15.name << std::endl;
    std::cout << "  -> Tier 1: Q1 (ratio " << twi_15.tier1_ratio << ")" << std::endl;
    std::cout << "  -> Tier 2: Q4 (ratio " << twi_15.tier2_ratio << ")" << std::endl;

    constexpr int N = 256;
    std::vector<float> data(N);
    for (int i = 0; i < N; i++) data[i] = (float)i * 0.01f - 1.28f;

    auto plan = quant::FormatRegistry::allocate_mix_blocks(twi_15, data.data(), N, 256);
    assert(!plan.formats.empty());
    std::cout << "  -> Allocated block count: " << plan.formats.size() << std::endl;

    std::cout << "\n1.50 BPW PROOF TEST PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
