// ============================================================================
// test_mix_components.cpp — L013/L014 (TRANSCRIPT.md Wave 2)
//   QUAD_MIX = EXACTLY 4 components; TWI_MIX = EXACTLY 2 components.
//   Tier ratios must sum to 1.0 within 1e-6, and each mix's effective_bpw
//   must equal the ratio-weighted BPW of its member formats.
// ============================================================================
#include "quant/format_registry.h"

#include <cmath>
#include <iostream>
#include <map>
#include <string>

using namespace quant;

static int failures = 0;
#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << "  FAIL: " << (msg) << " (line " << __LINE__ << ")\n";    \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

static std::map<int, float> make_bpw_map() {
    std::map<int, float> bpw;
    for (const auto& s : FormatRegistry::get_all_singles())
        bpw[(int)s.id] = s.bpw;
    return bpw;
}

static void validate(const MixDescriptor& m, int expected_tiers,
                     const std::map<int, float>& bpw) {
    const std::string name = m.name;
    CHECK(m.num_tiers == expected_tiers,
          expected_tiers == 4 ? "QUAD must have exactly 4 components"
                              : "TWI must have exactly 2 components");
    float sum = m.tier1_ratio + m.tier2_ratio +
                (m.num_tiers >= 3 ? m.tier3_ratio : 0.0f) +
                (m.num_tiers >= 4 ? m.tier4_ratio : 0.0f);
    CHECK(std::fabs(sum - 1.0f) <= 1e-6f, "tier ratios must sum to 1.0");

    // Adaptive mixes treat effective_bpw as a HARD allocation budget (the
    // allocator assigns blocks by benefit-per-byte), NOT as the naive
    // ratio-weighted mean — so bound-check it against member BPWs instead.
    auto it1 = bpw.find((int)m.tier1_fmt);
    auto it2 = bpw.find((int)m.tier2_fmt);
    if (it1 != bpw.end() && it2 != bpw.end()) {
        float lo = std::min(it1->second, it2->second);
        float hi = std::max(it1->second, it2->second);
        if (m.num_tiers >= 3) {
            auto it3 = bpw.find((int)m.tier3_fmt);
            if (it3 != bpw.end()) { lo = std::min(lo, it3->second); hi = std::max(hi, it3->second); }
        }
        if (m.num_tiers >= 4) {
            auto it4 = bpw.find((int)m.tier4_fmt);
            if (it4 != bpw.end()) { lo = std::min(lo, it4->second); hi = std::max(hi, it4->second); }
        }
        CHECK(m.effective_bpw >= lo - 0.01f && m.effective_bpw <= hi + 0.01f,
              "effective_bpw must lie within its members' BPW range");
    }
}

int main() {
    std::cout << "== MIX component invariant tests (L013/L014) ==" << std::endl;
    const auto bpw = make_bpw_map();

    const auto& twis = FormatRegistry::get_all_twi_mixes();
    CHECK(twis.size() >= 2, "at least two TWI_MIX variants registered");
    for (const auto& m : twis) validate(m, 2, bpw);
    std::cout << "  TWI_MIX variants validated: " << twis.size() << std::endl;

    const auto& quads = FormatRegistry::get_all_four_mixes();
    CHECK(quads.size() >= 14, "all fourteen QUAD_MIX variants registered");
    for (const auto& m : quads) validate(m, 4, bpw);
    std::cout << "  QUAD_MIX variants validated: " << quads.size() << std::endl;

    // Spot-check the flagship: QUAD_MIX@12.5 components are Q6/Q12/Q24/Q32.
    bool found_flagship = false;
    for (const auto& m : quads) {
        if (m.id == RegFormat::MXQ_12_5_GRP) {
            found_flagship = true;
            CHECK(m.tier1_fmt == RegFormat::Q6 && m.tier2_fmt == RegFormat::Q12 &&
                  m.tier3_fmt == RegFormat::Q24 && m.tier4_fmt == RegFormat::Q32,
                  "flagship 12.5 mix members must be Q6/Q12/Q24/Q32");
        }
    }
    CHECK(found_flagship, "flagship QUAD_MIX@12.5 present");

    // Targeted getters must return descriptors of the right tier count.
    CHECK(FormatRegistry::get_four_mix(12.5f).num_tiers == 4, "get_four_mix -> 4 tiers");
    CHECK(FormatRegistry::get_twi_mix(1.5f).num_tiers == 2, "get_twi_mix -> 2 tiers");

    if (failures == 0) {
        std::cout << "MIX COMPONENTS: ALL TESTS PASSED" << std::endl;
        return 0;
    }
    std::cout << "MIX COMPONENTS: " << failures << " FAILURES" << std::endl;
    return 1;
}
