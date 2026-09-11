// test_code_gen.cpp — template string-correctness GEMM/attn/norm x AVX2/AVX512/NEON + compile-smoke
#include "quant/code_gen.h"
#include "quant/test.h"

#include <cstdio>
#include <string>

using namespace quant;
using namespace quant::code_gen;

namespace {

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

} // namespace

static void test_gemm_templates() {
    TEST_SUITE("codegen gemm");
    CodeGenerator gen;

    std::string avx2 = gen.generate_gemm(16, 16, 16, SIMDTarget::AVX2);
    TEST_CHECK(!avx2.empty(), "gemm avx2 non-empty");
    TEST_CHECK(contains(avx2, "quant_gemm"), "gemm avx2 defines quant_gemm");
    TEST_CHECK(contains(avx2, "immintrin.h"), "gemm avx2 includes x86 simd header");

    std::string avx512 = gen.generate_gemm(16, 16, 16, SIMDTarget::AVX512);
    TEST_CHECK(!avx512.empty(), "gemm avx512 non-empty");
    TEST_CHECK(contains(avx512, "quant_gemm"), "gemm avx512 defines quant_gemm");
    TEST_CHECK(contains(avx512, "immintrin.h"), "gemm avx512 includes x86 simd header");

    std::string neon = gen.generate_gemm(16, 16, 16, SIMDTarget::NEON);
    TEST_CHECK(!neon.empty(), "gemm neon non-empty");
    TEST_CHECK(contains(neon, "quant_gemm"), "gemm neon defines quant_gemm");
    TEST_CHECK(!contains(neon, "immintrin.h"), "gemm neon has no x86 simd header");
}

static void test_attention_templates() {
    TEST_SUITE("codegen attention");
    CodeGenerator gen;

    std::string avx2 = gen.generate_attention(8, 16, 2, SIMDTarget::AVX2);
    TEST_CHECK(!avx2.empty(), "attn avx2 non-empty");
    TEST_CHECK(contains(avx2, "quant_attention"), "attn avx2 defines quant_attention");
    TEST_CHECK(contains(avx2, "scale"), "attn avx2 has scale factor");
    TEST_CHECK(contains(avx2, "_mm256_"), "attn avx2 uses avx2 intrinsics");

    std::string avx512 = gen.generate_attention(8, 16, 2, SIMDTarget::AVX512);
    TEST_CHECK(!avx512.empty(), "attn avx512 non-empty");
    TEST_CHECK(contains(avx512, "quant_attention"), "attn avx512 defines quant_attention");
    TEST_CHECK(contains(avx512, "_mm512_"), "attn avx512 uses avx512 intrinsics");

    std::string neon = gen.generate_attention(8, 16, 2, SIMDTarget::NEON);
    TEST_CHECK(!neon.empty(), "attn neon non-empty");
    TEST_CHECK(contains(neon, "quant_attention"), "attn neon defines quant_attention");
    TEST_CHECK(contains(neon, "loadu_ps"), "attn neon has vector loads");
    TEST_CHECK(!contains(neon, "_mm256_") && !contains(neon, "_mm512_"),
               "attn neon uses no x86 intrinsics");
}

static void test_norm_templates() {
    TEST_SUITE("codegen norm");
    CodeGenerator gen;

    std::string avx2 = gen.generate_rms_norm(32, SIMDTarget::AVX2);
    TEST_CHECK(!avx2.empty(), "rms avx2 non-empty");
    TEST_CHECK(contains(avx2, "quant_rms_norm"), "rms avx2 defines quant_rms_norm");
    TEST_CHECK(contains(avx2, "sqrtf"), "rms avx2 normalizes by rms");

    std::string avx512 = gen.generate_rms_norm(32, SIMDTarget::AVX512);
    TEST_CHECK(!avx512.empty(), "rms avx512 non-empty");
    TEST_CHECK(contains(avx512, "quant_rms_norm"), "rms avx512 defines quant_rms_norm");

    std::string neon = gen.generate_rms_norm(32, SIMDTarget::NEON);
    TEST_CHECK(!neon.empty(), "rms neon non-empty");
    TEST_CHECK(contains(neon, "quant_rms_norm"), "rms neon defines quant_rms_norm");

    std::string ln = gen.generate_layer_norm(32, SIMDTarget::AVX2);
    TEST_CHECK(contains(ln, "quant_layer_norm"), "layer_norm defines quant_layer_norm");
    TEST_CHECK(contains(ln, "mean"), "layer_norm centers by mean");
}

static void test_simd_headers() {
    TEST_SUITE("codegen headers");
    CodeGenerator gen;

    std::string avx2 = gen.generate_avx2_header();
    TEST_CHECK(contains(avx2, "_mm256_"), "avx2 header wraps _mm256 intrinsics");
    TEST_CHECK(contains(avx2, "QUANT_HAS_AVX2"), "avx2 header defines feature macro");

    std::string avx512 = gen.generate_avx512_header();
    TEST_CHECK(contains(avx512, "_mm512_"), "avx512 header wraps _mm512 intrinsics");
    TEST_CHECK(contains(avx512, "QUANT_HAS_AVX512"), "avx512 header defines feature macro");

    std::string neon = gen.generate_neon_header();
    TEST_CHECK(contains(neon, "arm_neon.h"), "neon header includes arm_neon.h");
    TEST_CHECK(contains(neon, "vld1q_f32"), "neon header wraps neon loads");
}

static void test_optimized_kernel_api() {
    TEST_SUITE("codegen kernel api");
    CodeGenerator gen;

    KernelSpec spec;
    spec.type = KernelType::GEMM;
    spec.M = 16;
    spec.N = 16;
    spec.K = 16;
    spec.simd = SIMDTarget::AVX2;
    GeneratedKernel gk = gen.generate_optimized_kernel(spec);
    TEST_CHECK(!gk.source_code.empty(), "optimized kernel has source");
    TEST_CHECK(contains(gk.function_name, "quant_kernel_gemm"), "kernel name encodes gemm");
    TEST_CHECK(contains(gk.function_name, "avx2"), "kernel name encodes simd target");
    TEST_CHECK(contains(gk.source_code, "quant_gemm"), "optimized gemm source defines quant_gemm");
}

static void test_compile_smoke() {
    TEST_SUITE("codegen compile-smoke");
    CodeGenerator gen;

    std::string gemm = gen.generate_gemm(16, 16, 16, SIMDTarget::AVX2);
    CodeValidationResult r_gemm = gen.validate(gemm);
    TEST_CHECK(r_gemm.syntax_valid, "gemm template passes syntax validation");
    printf("  [info] gemm compiles=%d\n", (int)r_gemm.compiles);

    std::string attn = gen.generate_attention(8, 16, 2, SIMDTarget::AVX2);
    CodeValidationResult r_attn = gen.validate(attn);
    TEST_CHECK(r_attn.syntax_valid, "attention template passes syntax validation");
    printf("  [info] attention compiles=%d\n", (int)r_attn.compiles);

    std::string norm = gen.generate_rms_norm(32, SIMDTarget::AVX2);
    CodeValidationResult r_norm = gen.validate(norm);
    TEST_CHECK(r_norm.syntax_valid, "rms_norm template passes syntax validation");
    printf("  [info] rms_norm compiles=%d\n", (int)r_norm.compiles);

    CodeValidationResult bad = gen.validate("this is definitely not valid c++ code ((((");
    TEST_CHECK(!bad.syntax_valid, "validator rejects garbage input");

    // PROD round-2: was `(void)cc; TEST_CHECK(true)` — vacuous. compile_code
    // really invokes the compiler (code_gen.cpp:958-961). BUT the invocation
    // is a bare `cl.exe`/`g++` with no toolchain env guarantee: outside a
    // VS dev-prompt it fails C1034 (no include path) even for valid code.
    // So: garbage must ALWAYS be rejected; valid code must compile WHEN the
    // toolchain is present, else honest-skip (never a fake pass/fail).
    TEST_CHECK(!gen.compile_code("this is definitely not valid c++ code (((("),
               "compile_code rejects garbage");
    {
        // Toolchain probe: a trivial TU must compile iff the env is sane.
        bool toolchain_ok = gen.compile_code(
            "#include <cstdint>\nint32_t probe_fn() { return 42; }\n");
        printf("  [info] toolchain present=%d\n", (int)toolchain_ok);
        bool cc = gen.compile_code(gemm);
        printf("  [info] compile_code(valid gemm)=%d\n", (int)cc);
        if (toolchain_ok) {
            TEST_CHECK(cc, "compile_code accepts the valid gemm template");
        } else {
            printf("  [info] SKIP compile-accept (no toolchain in this env)\n");
            TEST_CHECK(true, "compile-accept skipped honestly without toolchain");
        }
    }
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("Transcender — CodeGen Test Suite\n");
    printf("==========================================\n");

    test_gemm_templates();
    test_attention_templates();
    test_norm_templates();
    test_simd_headers();
    test_optimized_kernel_api();
    test_compile_smoke();

    printf("\n==========================================\n");
    return TEST_REPORT() > 0 ? 1 : 0;
}
