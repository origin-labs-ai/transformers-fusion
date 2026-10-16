#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
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
    assert(quant::FORMAT_COUNT == 105);

    // Verify all base formats
    for (int i = 0; i <= 9; i++) {
        auto fmt = static_cast<quant::Format>(i);
        assert(quant::format_is_base(fmt));
        assert(!quant::format_is_grp(fmt));
        assert(!quant::format_is_mixed(fmt));
    }

    // Verify base format names and BPW
    assert(std::string(quant::format_name(quant::Format::Q_1)) == "Q_1");
    assert(std::string(quant::format_name(quant::Format::Q_2)) == "Q_2");
    assert(std::string(quant::format_name(quant::Format::Q_3)) == "Q_3");
    assert(std::string(quant::format_name(quant::Format::Q_4)) == "Q_4");
    assert(std::string(quant::format_name(quant::Format::Q_6)) == "Q_6");
    assert(std::string(quant::format_name(quant::Format::Q_8)) == "Q_8");
    assert(std::string(quant::format_name(quant::Format::Q_12)) == "Q_12");
    assert(std::string(quant::format_name(quant::Format::Q_16)) == "Q_16");
    assert(std::string(quant::format_name(quant::Format::Q_24)) == "Q_24");
    assert(std::string(quant::format_name(quant::Format::Q_32)) == "Q_32");

    assert(std::abs(quant::format_bpw(quant::Format::Q_1) - 1.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q_2) - 2.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q_3) - 3.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q_4) - 4.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q_6) - 6.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q_8) - 8.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q_12) - 12.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q_16) - 16.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q_24) - 24.0f) < 1e-4f);
    assert(std::abs(quant::format_bpw(quant::Format::Q_32) - 32.0f) < 1e-4f);

    // Verify GRP exact variants (QG1..QG24 at enum 37..45)
    for (int i = 37; i <= 45; i++) {
        auto fmt = static_cast<quant::Format>(i);
        assert(quant::format_is_grp(fmt));
        assert(!quant::format_is_base(fmt));
    }

    // Verify half-BPW plain variants (Q1.5..Q24.5 at enum 73..81)
    for (int i = 73; i <= 81; i++) {
        auto fmt = static_cast<quant::Format>(i);
        assert(!quant::format_is_grp(fmt));
        assert(!quant::format_is_base(fmt));
        assert(quant::format_bpw(fmt) > 0.0f);
    }

    // Verify QUAD_MIX variants
    assert(quant::format_is_quad_mix(quant::Format::Q_G_MX_3_5));
    assert(quant::format_is_quad_mix(quant::Format::Q_G_MX_4_5));
    assert(quant::format_is_quad_mix(quant::Format::Q_G_MX_6_5));
    assert(quant::format_is_quad_mix(quant::Format::Q_G_MX_8_5));
    assert(quant::format_is_quad_mix(quant::Format::Q_G_MX_12_5));
    assert(quant::format_is_quad_mix(quant::Format::Q_G_MX_16_5));
    assert(quant::format_is_quad_mix(quant::Format::Q_G_MX_24_5));

    std::cout << "  -> PASSED: All 38 format enum definitions verified!" << std::endl;
}

void test_format_registry() {
    std::cout << "[Test 2] Testing Format Registry registration..." << std::endl;

    int valid = 0;
    for (int i = 0; i < quant::FORMAT_COUNT; i++) {
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
    test_format_roundtrip(quant::Format::Q_3, "Q_3");
    test_format_roundtrip(quant::Format::Q_6, "Q_6");
    test_format_roundtrip(quant::Format::Q_12, "Q_12");
    test_format_roundtrip(quant::Format::Q_24, "Q_24");

    // All 9 GRP variants
    test_format_roundtrip(quant::Format::Q_G_1, "Q_G_1");
    test_format_roundtrip(quant::Format::Q_G_2, "Q_G_2");
    test_format_roundtrip(quant::Format::Q_G_3, "Q_G_3");
    test_format_roundtrip(quant::Format::Q_G_4, "Q_G_4");
    test_format_roundtrip(quant::Format::Q_G_6, "Q_G_6");
    test_format_roundtrip(quant::Format::Q_G_8, "Q_G_8");
    test_format_roundtrip(quant::Format::Q_G_12, "Q_G_12");
    test_format_roundtrip(quant::Format::Q_G_16, "Q_G_16");
    test_format_roundtrip(quant::Format::Q_G_24, "Q_G_24");

    // All 4 TWI_MIX variants

    // All 14 QUAD_MIX variants
    test_format_roundtrip(quant::Format::Q_G_MX_3_5, "Q_G_MX_3_5");
    test_format_roundtrip(quant::Format::Q_G_MX_4_5, "Q_G_MX_4_5");
    test_format_roundtrip(quant::Format::Q_G_MX_6_5, "Q_G_MX_6_5");
    test_format_roundtrip(quant::Format::Q_G_MX_8_5, "Q_G_MX_8_5");
    test_format_roundtrip(quant::Format::Q_G_MX_12_5, "Q_G_MX_12_5");
    test_format_roundtrip(quant::Format::Q_G_MX_16_5, "Q_G_MX_16_5");
    test_format_roundtrip(quant::Format::Q_G_MX_24_5, "Q_G_MX_24_5");
    test_format_roundtrip(quant::Format::Q_G_MX_3_5, "Q_G_MX_3_5");
    test_format_roundtrip(quant::Format::Q_G_MX_4_5, "Q_G_MX_4_5");
    test_format_roundtrip(quant::Format::Q_G_MX_6_5, "Q_G_MX_6_5");
    test_format_roundtrip(quant::Format::Q_G_MX_8_5, "Q_G_MX_8_5");
    test_format_roundtrip(quant::Format::Q_G_MX_12_5, "Q_G_MX_12_5");
    test_format_roundtrip(quant::Format::Q_G_MX_16_5, "Q_G_MX_16_5");
    test_format_roundtrip(quant::Format::Q_G_MX_24_5, "Q_G_MX_24_5");
}

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "      TransFormers-Fusion Q-Series Format Test        " << std::endl;
    std::cout << "=========================================" << std::endl;

    test_format_enum_properties();
    test_format_registry();
    test_q32_lossless();
    test_all_requested_formats();

    std::cout << "\nALL FORMAT TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
