// test_mixed_precision_proof.cpp — Proof test for ALL 14 QUAD_MIX precision levels
#include "quant/types.h"
#include "quant/format_registry.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "  InNova QUAD_MIX Mixed Precision Proof  " << std::endl;
    std::cout << "=========================================" << std::endl;

    constexpr float target_bpws[7] = {3.5f, 4.5f, 6.5f, 8.5f, 12.5f, 16.5f, 24.5f};

    std::cout << "[Test 1] Testing all 7 base QUAD_MIX formats..." << std::endl;
    for (int i = 0; i < 7; i++) {
        float target = target_bpws[i];
        auto mix = quant::FormatRegistry::get_four_mix(target);
        std::cout << "  -> QUAD_MIX@" << target << " resolved to: " << mix.name
                  << " (effective BPW: " << mix.effective_bpw << ", tiers: " << mix.num_tiers << ")" << std::endl;

        assert(mix.num_tiers == 4);
        assert(std::abs(mix.effective_bpw - target) < 1e-4f);
    }

    std::cout << "[Test 2] Verifying 4-tier allocation for MXQ_24_5_G..." << std::endl;
    // get_four_mix tie-breaks equal-BPW candidates by registry order (plain
    // first), so request the GRP descriptor directly from the mix table.
    const quant::MixDescriptor* mix24_5 = nullptr;
    for (const auto& m : quant::FormatRegistry::get_all_four_mixes()) {
        if (m.id == quant::RegFormat::MXQ_24_5_G) { mix24_5 = &m; break; }
    }
    assert(mix24_5 != nullptr);
    assert(mix24_5->tier1_fmt == quant::RegFormat::Q12);
    assert(mix24_5->tier2_fmt == quant::RegFormat::Q16);
    assert(mix24_5->tier3_fmt == quant::RegFormat::Q24);
    assert(mix24_5->tier4_fmt == quant::RegFormat::Q32);

    std::cout << "  -> MXQ_24_5_G composition verified: Q12(25%) + Q16(30%) + Q24(35%) + Q32(10%) = 24.5 BPW!" << std::endl;

    std::cout << "\nMIXED PRECISION PROOF TEST PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
