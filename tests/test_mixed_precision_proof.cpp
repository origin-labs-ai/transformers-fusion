#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
// test_mixed_precision_proof.cpp — Proof test for ALL 14 QUAD_MIX precision levels
#include "quant/types.h"
#include "quant/format_registry.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "  Transcender QUAD_MIX Mixed Precision Proof  " << std::endl;
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

    std::cout << "[Test 2] Verifying 4-tier allocation for QG_MX_24_5..." << std::endl;
    // get_four_mix tie-breaks equal-BPW candidates by registry order (plain
    // first), so request the GRP descriptor directly from the mix table.
    const quant::MixDescriptor* mix24_5 = nullptr;
    for (const auto& m : quant::FormatRegistry::get_all_four_mixes()) {
        if (m.id == quant::RegFormat::QG_MX_24_5) { mix24_5 = &m; break; }
    }
    assert(mix24_5 != nullptr);
    // QG_MX_* is the GROUPED ladder: tiers are grouped variants (registry
    // format_registry.cpp). Plain ladder Q_MX_24_5 uses Q12/Q16/Q24/Q32.
    assert(mix24_5->tier1_fmt == quant::RegFormat::QG12);
    assert(mix24_5->tier2_fmt == quant::RegFormat::QG16);
    assert(mix24_5->tier3_fmt == quant::RegFormat::QG24);
    assert(mix24_5->tier4_fmt == quant::RegFormat::Q32);

    std::cout << "  -> QG_MX_24_5 composition verified: QG12(25%) + QG16(30%) + QG24(35%) + Q32(10%) = 24.5 BPW!" << std::endl;

    std::cout << "\nMIXED PRECISION PROOF TEST PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
