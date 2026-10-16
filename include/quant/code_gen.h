#pragma once
#include "quant/tensor.h"
#include "quant/model.h"
#include "quant/types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <cstdint>

namespace quant {
namespace code_gen {

enum class KernelType : uint8_t {
    GEMM = 0,
    ATTENTION = 1,
    RMS_NORM = 2,
    LAYER_NORM = 3,
    SOFTMAX = 4,
    RELU = 5,
    SILU = 6,
    GELU = 7,
    ADD = 8,
    MUL = 9,
    SCALE = 10,
    RESIDUAL_ADD = 11,
    EMBEDDING_LOOKUP = 12,
    HEAD_PROJECTION = 13,
    QUANTIZE_DEQUANTIZE = 14
};

enum class SIMDTarget : uint8_t {
    SCALAR = 0,
    AVX2 = 1,
    AVX512 = 2,
    NEON = 3
};

enum class UnrollLevel : uint8_t {
    NONE = 0,
    PARTIAL = 1,
    FULL = 2
};

struct KernelSpec {
    KernelType type = KernelType::GEMM;
    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;
    int block_size = 32;
    SIMDTarget simd = SIMDTarget::SCALAR;
    UnrollLevel unroll = UnrollLevel::NONE;
    Format format = Format::Q_32;
    bool fused = false;
    int num_threads = 1;
    std::string name;
};

struct CodeTemplate {
    std::string name;
    std::string description;
    std::string template_code;
    std::vector<std::string> placeholders;
    KernelType kernel_type;
};

struct GeneratedKernel {
    KernelSpec spec;
    std::string source_code;
    std::string function_name;
    bool validated = false;
    bool compiled = false;
    double benchmark_ms = 0.0;
    int64_t ops_per_sec = 0;
    std::string error_message;
};

struct SlowPath {
    std::string op_name;
    double current_time_ms = 0.0;
    double target_time_ms = 0.0;
    std::string current_kernel;
    int priority = 0;
};

struct CodeValidationResult {
    bool syntax_valid = false;
    bool compiles = false;
    bool runs = false;
    bool correctness_pass = false;
    float max_error = 0.0f;
    std::string error_message;
    double runtime_ms = 0.0;
};

class CodeGenerator {
public:
    CodeGenerator(Model* model = nullptr);
    ~CodeGenerator();

    void set_model(Model* model);

    std::string generate_gemm(int64_t M, int64_t N, int64_t K, SIMDTarget simd = SIMDTarget::SCALAR);
    std::string generate_attention(int64_t seq_len, int64_t head_dim, int num_heads, SIMDTarget simd = SIMDTarget::SCALAR);
    std::string generate_rms_norm(int64_t dim, SIMDTarget simd = SIMDTarget::SCALAR);
    std::string generate_layer_norm(int64_t dim, SIMDTarget simd = SIMDTarget::SCALAR);
    std::string generate_softmax(int64_t dim, SIMDTarget simd = SIMDTarget::SCALAR);
    std::string generate_relu(int64_t n, SIMDTarget simd = SIMDTarget::SCALAR);
    std::string generate_silu(int64_t n, SIMDTarget simd = SIMDTarget::SCALAR);
    std::string generate_gelu(int64_t n, SIMDTarget simd = SIMDTarget::SCALAR);
    std::string generate_add(int64_t n, SIMDTarget simd = SIMDTarget::SCALAR);
    std::string generate_mul(int64_t n, SIMDTarget simd = SIMDTarget::SCALAR);
    std::string generate_scale(int64_t n, float scale_val, SIMDTarget simd = SIMDTarget::SCALAR);
    std::string generate_residual_add(int64_t n, SIMDTarget simd = SIMDTarget::SCALAR);
    std::string generate_embedding_lookup(int64_t vocab_size, int64_t embed_dim, SIMDTarget simd = SIMDTarget::SCALAR);
    std::string generate_quantize_dequantize(int64_t n, Format src_format, Format dst_format);

    std::string generate_kernel_for_shape(const KernelSpec& spec);

    std::string generate_format_dispatch(const std::string& op_name, const std::vector<Format>& formats);
    std::string generate_format_branch(const std::string& op_name, Format format, int64_t dim);

    std::string generate_unrolled_loop(const std::string& var, int64_t start, int64_t end,
                                        int unroll_factor, const std::string& body);
    std::string generate_simd_load(const std::string& var, const std::string& ptr, int width);
    std::string generate_simd_store(const std::string& ptr, const std::string& var, int width);
    std::string generate_simd_fma(const std::string& dst, const std::string& a, const std::string& b, const std::string& c);
    std::string generate_simd_add(const std::string& dst, const std::string& a, const std::string& b, int width);
    std::string generate_simd_mul(const std::string& dst, const std::string& a, const std::string& b, int width);

    std::string generate_avx2_header();
    std::string generate_avx512_header();
    std::string generate_neon_header();
    std::string generate_includes(const std::vector<std::string>& includes);

    GeneratedKernel generate_optimized_kernel(const KernelSpec& spec);
    GeneratedKernel specialize_for_shape(KernelType type, int64_t M, int64_t N, int64_t K);

    CodeValidationResult validate(const std::string& code);
    bool compile_code(const std::string& code);
    bool run_test(const std::string& code, const std::string& test_name);

    std::vector<SlowPath> identify_slow_paths(double threshold_ms = 1.0);
    std::string improve_slow_path(const SlowPath& path);
    std::vector<GeneratedKernel> auto_optimize(int n_iterations = 10);

    void register_template(const CodeTemplate& tmpl);
    std::string apply_template(const std::string& template_name,
                                const std::unordered_map<std::string, std::string>& vars);
    std::vector<std::string> get_available_templates() const;

    GeneratedKernel get_kernel(const std::string& name) const;
    std::vector<GeneratedKernel> get_all_kernels() const;
    void clear_kernels();

    std::string generate_full_source(const std::vector<GeneratedKernel>& kernels);
    bool save_source(const std::string& path, const std::string& source);

private:
    Model* model_;
    std::unordered_map<std::string, GeneratedKernel> kernel_cache_;
    std::vector<CodeTemplate> templates_;
    std::vector<SlowPath> slow_paths_;

    std::string indent(int level) const;
    std::string make_function_name(const KernelSpec& spec) const;
    std::string format_type_string(Format fmt) const;
    std::string simd_type_string(SIMDTarget simd, int width) const;
    std::string block_size_literal(int block_size) const;
    std::string generate_loop_header(const std::string& var, int64_t start, int64_t end, const std::string& step = "1");
    std::string generate_if_block(const std::string& condition, const std::string& body, int indent_level = 1);
    std::string generate_block(const std::string& body, int indent_level);
    std::string simd_prefix(SIMDTarget simd) const;
    int simd_width(SIMDTarget simd) const;
    std::string clamp_expr(const std::string& expr, int64_t max_val) const;
    std::string min_expr(const std::string& a, const std::string& b) const;
    std::string max_expr(const std::string& a, const std::string& b) const;

    void init_default_templates();
    std::string gemm_tiled(int64_t M, int64_t N, int64_t K, int tile_m, int tile_n, int tile_k, SIMDTarget simd);
    std::string attention_flash(int64_t seq_len, int64_t head_dim, int num_heads, SIMDTarget simd);
    std::string norm_kernel(const std::string& name, int64_t dim, bool rms, SIMDTarget simd);
    std::string activation_kernel(const std::string& name, int64_t n, const std::string& op, SIMDTarget simd);
    std::string simd_load_store_test(SIMDTarget simd, int width);
};

} // namespace code_gen
} // namespace quant
