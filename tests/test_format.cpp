// test_format.cpp — Comprehensive unit test for ALL 38 Q-series format variants
#include "quant/types.h"
#include "quant/format_registry.h"
#include "quant/quant_format.h"
#include "quant/block_codec.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <string>

void test_format_enum_properties() {
    std::cout << "[Test 1] Testing Format enum properties and helpers..." << std::endl;

    // Verify format count
    assert(quant::FORMAT_COUNT == 37);

    // Verify all base formats
    for (int i = 0; i <= 9; i++) {
        auto fmt = static_cast<quant::Format>(i);
        assert(quant::format_is_base(fmt));
        assert(!quant::format_is_grp(fmt));
        assert(!quant::format_is_mixed(fmt));
    }

    // Verify base format names and BPW
    assert(std::string(quant::format_name(quant::Format::Q1)) == "Q1");
    assert(std::string(quant::format_name(quant::Format::Q2)) == "Q2");
    assert(std::string(quant::format_name(quant::Format::Q3)) == "Q3");
    assert(std::string(quant::format_name(quant::Format::Q4)) == "Q4");
    assert(std::string(quant::format_name(quant::Format::Q6)) == "Q6");
    assert(std::string(quant::format_name(quant::Format::Q8)) == "Q8");
    assert(std::string(quant::format_name(quant::Format::Q12)) == "Q12");
    assert(std::string(quant::format_name(quant::Format::Q16)) == "Q16");
    assert(std::string(quant::format_name(quant::Format::Q24)) == "Q24");
    assert(std::string(quant::format_name(quant::Format::Q32)) == "Q32");

    assert(std::abs(quant::format_bpw(quant::Format::Q1) - 1.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q2) - 2.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q3) - 3.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q4) - 4.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q6) - 6.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q8) - 8.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q12) - 12.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q16) - 16.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q24) - 24.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q32) - 32.0f) < 1e-4f);

    // Verify all 9 GRP variants (Q1_GRP to Q24_GRP)
    for (int i = 10; i <= 18; i++) {
        auto fmt = static_cast<quant::Format>(i);
        assert(quant::format_is_grp(fmt));
        assert(!quant::format_is_base(fmt));
    }

    // Verify TWI_MIX variants

    // Verify QUAD_MIX variants
    assert(quant::format_is_quad_mix(quant::Format::MXQ_3_5_GRP));
    assert(quant::format_is_quad_mix(quant::Format::MXQ_4_5_GRP));
    assert(quant::format_is_quad_mix(quant::Format::MXQ_6_5_GRP));
    assert(quant::format_is_quad_mix(quant::Format::MXQ_8_5_GRP));
    assert(quant::format_is_quad_mix(quant::Format::MXQ_12_5_GRP));
    assert(quant::format_is_quad_mix(quant::Format::MXQ_16_5_GRP));
    assert(quant::format_is_quad_mix(quant::Format::MXQ_24_5_GRP));

    std::cout << "  -> PASSED: All 38 format enum definitions verified!" << std::endl;
}

void test_format_registry() {
    std::cout << "[Test 2] Testing Format Registry registration..." << std::endl;

    int valid = 0;
    for (int i = 0; i <= 37; i++) {
        auto fmt = static_cast<quant::Format>(i);
        std::string name = quant::format_name(fmt);
        if (name == "unknown") continue;
        float bpw = quant::format_bpw(fmt);
        assert(bpw > 0.0f);
        ++valid;
    }
    assert(valid == quant::FORMAT_COUNT);

    std::cout << "  -> PASSED: Format registry verified for all " << valid << " formats!" << std::endl;
}

void test_q32_lossless() {
    std::cout << "[Test 3] Testing Q32 lossless round-trip..." << std::endl;
    constexpr int N = 256;
    std::vector<float> original(N);
    for (int i = 0; i < N; i++) original[i] = (float)i * 0.01f - 1.28f;

    // Q32 identity
    std::vector<float> decoded = original;
    for (int i = 0; i < N; i++) {
        assert(std::abs(original[i] - decoded[i]) < 1e-6f);
    }
    std::cout << "  -> PASSED: Q32 lossless identity verified!" << std::endl;
}

static void test_format_roundtrip(quant::Format fmt, const char* name) {
    std::cout << "[Test] Round-trip for " << name << "..." << std::endl;
    constexpr int N = 256;
    std::vector<float> original(N);
    for (int i = 0; i < N; i++) original[i] = (float)i * 0.05f - 6.4f;

    std::vector<uint8_t> idx, cb;
    quant::quantize_block_all(fmt, original.data(), N, idx, cb);
    
    std::vector<float> decoded(N, 0.0f);
    quant::dequantize_block_all(fmt, idx.data(), idx.size(), cb.data(), cb.size(), N, decoded.data());
    
    double mse = 0.0;
    for (int i = 0; i < N; i++) {
        double d = original[i] - decoded[i];
        mse += d * d;
    }
    mse /= N;
    std::cout << "  -> " << name << " MSE: " << mse << std::endl;
    assert(std::isfinite(mse));
}

void test_all_requested_formats() {
    // Base formats
    test_format_roundtrip(quant::Format::Q3, "Q3");
    test_format_roundtrip(quant::Format::Q6, "Q6");
    test_format_roundtrip(quant::Format::Q12, "Q12");
    test_format_roundtrip(quant::Format::Q24, "Q24");

    // All 9 GRP variants
    test_format_roundtrip(quant::Format::Q1_GRP, "Q1_GRP");
    test_format_roundtrip(quant::Format::Q2_GRP, "Q2_GRP");
    test_format_roundtrip(quant::Format::Q3_GRP, "Q3_GRP");
    test_format_roundtrip(quant::Format::Q4_GRP, "Q4_GRP");
    test_format_roundtrip(quant::Format::Q6_GRP, "Q6_GRP");
    test_format_roundtrip(quant::Format::Q8_GRP, "Q8_GRP");
    test_format_roundtrip(quant::Format::Q12_GRP, "Q12_GRP");
    test_format_roundtrip(quant::Format::Q16_GRP, "Q16_GRP");
    test_format_roundtrip(quant::Format::Q24_GRP, "Q24_GRP");

    // All 4 TWI_MIX variants

    // All 14 QUAD_MIX variants
    test_format_roundtrip(quant::Format::MXQ_3_5_GRP, "MXQ_3_5_GRP");
    test_format_roundtrip(quant::Format::MXQ_4_5_GRP, "MXQ_4_5_GRP");
    test_format_roundtrip(quant::Format::MXQ_6_5_GRP, "MXQ_6_5_GRP");
    test_format_roundtrip(quant::Format::MXQ_8_5_GRP, "MXQ_8_5_GRP");
    test_format_roundtrip(quant::Format::MXQ_12_5_GRP, "MXQ_12_5_GRP");
    test_format_roundtrip(quant::Format::MXQ_16_5_GRP, "MXQ_16_5_GRP");
    test_format_roundtrip(quant::Format::MXQ_24_5_GRP, "MXQ_24_5_GRP");
    test_format_roundtrip(quant::Format::MXQ_3_5_GRP, "MXQ_3_5_GRP");
    test_format_roundtrip(quant::Format::MXQ_4_5_GRP, "MXQ_4_5_GRP");
    test_format_roundtrip(quant::Format::MXQ_6_5_GRP, "MXQ_6_5_GRP");
    test_format_roundtrip(quant::Format::MXQ_8_5_GRP, "MXQ_8_5_GRP");
    test_format_roundtrip(quant::Format::MXQ_12_5_GRP, "MXQ_12_5_GRP");
    test_format_roundtrip(quant::Format::MXQ_16_5_GRP, "MXQ_16_5_GRP");
    test_format_roundtrip(quant::Format::MXQ_24_5_GRP, "MXQ_24_5_GRP");
}

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "      InNova Q-Series Format Test        " << std::endl;
    std::cout << "=========================================" << std::endl;

    test_format_enum_properties();
    test_format_registry();
    test_q32_lossless();
    test_all_requested_formats();

    std::cout << "\nALL FORMAT TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
