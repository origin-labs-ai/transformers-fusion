#include "quant/code_gen.h"
#include "quant/random.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <numeric>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <thread>
#include <chrono>

namespace quant {
namespace code_gen {

namespace {

double now_ms() {
    return (double)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // anonymous namespace

// ========================================================================
// CodeGenerator
// ========================================================================

CodeGenerator::CodeGenerator(Model* model) : model_(model) {
    init_default_templates();
}

CodeGenerator::~CodeGenerator() {}

void CodeGenerator::set_model(Model* model) {
    model_ = model;
}

// ========================================================================
// Utility generators
// ========================================================================

std::string CodeGenerator::indent(int level) const {
    return std::string(level * 4, ' ');
}

std::string CodeGenerator::make_function_name(const KernelSpec& spec) const {
    std::ostringstream oss;
    oss << "quant_kernel_";
    switch (spec.type) {
        case KernelType::GEMM:   oss << "gemm"; break;
        case KernelType::ATTENTION: oss << "attention"; break;
        case KernelType::RMS_NORM: oss << "rms_norm"; break;
        case KernelType::LAYER_NORM: oss << "layer_norm"; break;
        case KernelType::SOFTMAX: oss << "softmax"; break;
        case KernelType::RELU:   oss << "relu"; break;
        case KernelType::SILU:   oss << "silu"; break;
        case KernelType::GELU:   oss << "gelu"; break;
        case KernelType::ADD:    oss << "add"; break;
        case KernelType::MUL:    oss << "mul"; break;
        case KernelType::SCALE:  oss << "scale"; break;
        case KernelType::RESIDUAL_ADD: oss << "residual_add"; break;
        case KernelType::EMBEDDING_LOOKUP: oss << "embed"; break;
        case KernelType::HEAD_PROJECTION: oss << "head_proj"; break;
        case KernelType::QUANTIZE_DEQUANTIZE: oss << "qdq"; break;
    }
    if (spec.M > 0) oss << "_m" << spec.M;
    if (spec.N > 0) oss << "_n" << spec.N;
    if (spec.K > 0) oss << "_k" << spec.K;
    if (spec.simd == SIMDTarget::AVX2) oss << "_avx2";
    else if (spec.simd == SIMDTarget::AVX512) oss << "_avx512";
    else if (spec.simd == SIMDTarget::NEON) oss << "_neon";
    return oss.str();
}

std::string CodeGenerator::format_type_string(Format fmt) const {
    switch (fmt) {
        case Format::Q1:            return "int8_t";
        case Format::Q4:            return "uint8_t";
        case Format::Q8:            return "uint8_t";
        case Format::Q16:           return "float";
        case Format::Q32:           return "float";
        default:                    return "float";
    }
}

std::string CodeGenerator::simd_type_string(SIMDTarget simd, int width) const {
    if (simd == SIMDTarget::AVX2) {
        if (width == 8) return "__m256";
        if (width == 16) return "__m256";
    }
    if (simd == SIMDTarget::AVX512) {
        if (width == 16) return "__m512";
        if (width == 8) return "__m256";
    }
    return "float";
}

std::string CodeGenerator::simd_prefix(SIMDTarget simd) const {
    switch (simd) {
        case SIMDTarget::AVX2:   return "_mm256_";
        case SIMDTarget::AVX512: return "_mm512_";
        case SIMDTarget::NEON:   return "v";
        default:                 return "";
    }
}

int CodeGenerator::simd_width(SIMDTarget simd) const {
    switch (simd) {
        case SIMDTarget::AVX2:   return 8;
        case SIMDTarget::AVX512: return 16;
        case SIMDTarget::NEON:   return 4;
        default:                 return 1;
    }
}

std::string CodeGenerator::generate_loop_header(const std::string& var, int64_t start, int64_t end, const std::string& step) {
    return "for (int64_t " + var + " = " + std::to_string(start) + "; " +
           var + " < " + std::to_string(end) + "; " + var + " += " + step + ")";
}

std::string CodeGenerator::generate_if_block(const std::string& condition, const std::string& body, int indent_level) {
    return indent(indent_level) + "if (" + condition + ") {\n" +
           body + "\n" + indent(indent_level) + "}";
}

std::string CodeGenerator::generate_block(const std::string& body, int indent_level) {
    return indent(indent_level) + "{\n" + body + "\n" + indent(indent_level) + "}";
}

std::string CodeGenerator::clamp_expr(const std::string& expr, int64_t max_val) const {
    return "((" + expr + ") < " + std::to_string(max_val) + " ? (" + expr + ") : " + std::to_string(max_val - 1) + ")";
}

std::string CodeGenerator::min_expr(const std::string& a, const std::string& b) const {
    return "((" + a + ") < (" + b + ") ? (" + a + ") : (" + b + "))";
}

std::string CodeGenerator::max_expr(const std::string& a, const std::string& b) const {
    return "((" + a + ") > (" + b + ") ? (" + a + ") : (" + b + "))";
}

std::string CodeGenerator::block_size_literal(int block_size) const {
    return std::to_string(block_size);
}

// ========================================================================
// GEMM generation
// ========================================================================

std::string CodeGenerator::generate_gemm(int64_t M, int64_t N, int64_t K, SIMDTarget simd) {
    KernelSpec spec;
    spec.type = KernelType::GEMM;
    spec.M = M; spec.N = N; spec.K = K;
    spec.simd = simd;
    return gemm_tiled(M, N, K, 64, 64, 64, simd);
}

std::string CodeGenerator::gemm_tiled(int64_t M, int64_t N, int64_t K, int tile_m, int tile_n, int tile_k, SIMDTarget simd) {
    std::ostringstream code;
    int sw = simd_width(simd);

    code << "#include <cstdint>\n#include <cmath>\n";
    if (simd == SIMDTarget::AVX2) code << "#include <immintrin.h>\n";
    if (simd == SIMDTarget::AVX512) code << "#include <immintrin.h>\n";

    code << "\nextern \"C\" void quant_gemm(const float* __restrict__ A,\n";
    code << "    const float* __restrict__ B, float* __restrict__ C,\n";
    code << "    int64_t M, int64_t N, int64_t K) {\n\n";

    code << indent(1) << "const int64_t TM = " << tile_m << ", TN = " << tile_n << ", TK = " << tile_k << ";\n";
    code << indent(1) << "for (int64_t i0 = 0; i0 < M; i0 += TM) {\n";
    code << indent(2) << "int64_t i_end = " << min_expr("i0 + TM", "M") << ";\n";
    code << indent(2) << "for (int64_t j0 = 0; j0 < N; j0 += TN) {\n";
    code << indent(3) << "int64_t j_end = " << min_expr("j0 + TN", "N") << ";\n";

    code << indent(3) << "for (int64_t ii = i0; ii < i_end; ii++) {\n";
    code << indent(4) << "for (int64_t jj = j0; jj < j_end; jj++) {\n";
    code << indent(5) << "float sum = C[ii * N + jj];\n";

    if (simd == SIMDTarget::SCALAR) {
        code << indent(5) << "for (int64_t k0 = 0; k0 < K; k0 += TK) {\n";
        code << indent(6) << "int64_t k_end = " << min_expr("k0 + TK", "K") << ";\n";
        code << indent(6) << "for (int64_t kk = k0; kk < k_end; kk++) {\n";
        code << indent(7) << "sum += A[ii * K + kk] * B[kk * N + jj];\n";
        code << indent(6) << "}\n";
        code << indent(5) << "}\n";
    } else {
        code << indent(5) << "for (int64_t kk = 0; kk < K; kk++) {\n";
        code << indent(6) << "sum += A[ii * K + kk] * B[kk * N + jj];\n";
        code << indent(5) << "}\n";
    }

    code << indent(5) << "C[ii * N + jj] = sum;\n";
    code << indent(4) << "}\n";
    code << indent(3) << "}\n";
    code << indent(2) << "}\n";
    code << indent(1) << "}\n";
    code << "}\n";

    return code.str();
}

// ========================================================================
// Attention generation
// ========================================================================

std::string CodeGenerator::generate_attention(int64_t seq_len, int64_t head_dim, int num_heads, SIMDTarget simd) {
    KernelSpec spec;
    spec.type = KernelType::ATTENTION;
    spec.M = seq_len; spec.N = head_dim; spec.K = num_heads;
    spec.simd = simd;
    return attention_flash(seq_len, head_dim, num_heads, simd);
}

std::string CodeGenerator::attention_flash(int64_t seq_len, int64_t head_dim, int num_heads, SIMDTarget simd) {
    std::ostringstream code;

    code << "#include <cstdint>\n#include <cmath>\n";
    if (simd != SIMDTarget::SCALAR) code << "#include <immintrin.h>\n";
    code << "\nextern \"C\" void quant_attention(\n";
    code << "    const float* __restrict__ Q, const float* __restrict__ K,\n";
    code << "    const float* __restrict__ V, float* __restrict__ out,\n";
    code << "    int64_t seq_len, int64_t head_dim, int num_heads) {\n\n";

    code << indent(1) << "float scale = 1.0f / sqrtf((float)head_dim);\n";
    code << indent(1) << "int64_t hd = head_dim;\n\n";

    code << indent(1) << "for (int h = 0; h < num_heads; h++) {\n";
    code << indent(2) << "for (int64_t i = 0; i < seq_len; i++) {\n";

    code << indent(3) << "float max_score = -1e30f;\n";
    code << indent(3) << "float scores[" << std::to_string(seq_len > 2048 ? 2048 : seq_len) << "];\n";
    code << indent(3) << "for (int64_t j = 0; j < seq_len; j++) {\n";
    code << indent(4) << "float dot = 0.0f;\n";

    if (simd == SIMDTarget::SCALAR) {
        code << indent(4) << "for (int64_t d = 0; d < hd; d++) {\n";
        code << indent(5) << "dot += Q[i * hd + d] * K[j * hd + d];\n";
        code << indent(4) << "}\n";
    } else {
        int sw = simd_width(simd);
        code << indent(4) << "int64_t d = 0;\n";
        code << indent(4) << simd_prefix(simd) + "float acc = " << simd_prefix(simd) << "set1_ps(0.0f);\n";
        code << indent(4) << "for (; d + " << sw << " <= hd; d += " << sw << ") {\n";
        code << indent(5) << "auto q_vec = " << simd_prefix(simd) << "loadu_ps(&Q[i * hd + d]);\n";
        code << indent(5) << "auto k_vec = " << simd_prefix(simd) << "loadu_ps(&K[j * hd + d]);\n";
        code << indent(5) << "acc = " << simd_prefix(simd) << "fmadd_ps(q_vec, k_vec, acc);\n";
        code << indent(4) << "}\n";
        code << indent(4) << "{ float tmp[" << sw << "]; " << simd_prefix(simd) << "storeu_ps(tmp, acc);\n";
        code << indent(5) << "for (int k = 0; k < " << sw << "; k++) dot += tmp[k]; }\n";
        code << indent(4) << "for (; d < hd; d++) dot += Q[i * hd + d] * K[j * hd + d];\n";
    }

    code << indent(4) << "scores[j] = dot * scale;\n";
    code << indent(4) << "if (scores[j] > max_score) max_score = scores[j];\n";
    code << indent(3) << "}\n";

    code << indent(3) << "float sum = 0.0f;\n";
    code << indent(3) << "for (int64_t j = 0; j < seq_len; j++) {\n";
    code << indent(4) << "scores[j] = expf(scores[j] - max_score);\n";
    code << indent(4) << "sum += scores[j];\n";
    code << indent(3) << "}\n";
    code << indent(3) << "for (int64_t j = 0; j < seq_len; j++) scores[j] /= (sum + 1e-10f);\n\n";

    code << indent(3) << "for (int64_t d = 0; d < hd; d++) {\n";
    code << indent(4) << "float val = 0.0f;\n";
    code << indent(4) << "for (int64_t j = 0; j < seq_len; j++) {\n";
    code << indent(5) << "val += scores[j] * V[j * hd + d];\n";
    code << indent(4) << "}\n";
    code << indent(4) << "out[i * hd + d] = val;\n";
    code << indent(3) << "}\n";

    code << indent(2) << "}\n";
    code << indent(1) << "}\n";
    code << "}\n";

    return code.str();
}

// ========================================================================
// Normalization kernels
// ========================================================================

std::string CodeGenerator::norm_kernel(const std::string& name, int64_t dim, bool rms, SIMDTarget simd) {
    std::ostringstream code;
    code << "#include <cstdint>\n#include <cmath>\n";
    code << "\nextern \"C\" void " << name << "(const float* __restrict__ input,\n";
    code << "    float* __restrict__ output, const float* __restrict__ weight, int64_t dim) {\n\n";

    code << indent(1) << "for (int64_t i = 0; i < dim; i++) {\n";

    if (rms) {
        code << indent(2) << "float sum_sq = 0.0f;\n";
        code << indent(2) << "for (int64_t j = 0; j < dim; j++) {\n";
        code << indent(3) << "sum_sq += input[i * dim + j] * input[i * dim + j];\n";
        code << indent(2) << "}\n";
        code << indent(2) << "float rms = sqrtf(sum_sq / (float)dim + 1e-5f);\n";
        code << indent(2) << "for (int64_t j = 0; j < dim; j++) {\n";
        code << indent(3) << "output[i * dim + j] = (input[i * dim + j] / rms) * weight[j];\n";
        code << indent(2) << "}\n";
    } else {
        code << indent(2) << "float mean = 0.0f;\n";
        code << indent(2) << "for (int64_t j = 0; j < dim; j++) mean += input[i * dim + j];\n";
        code << indent(2) << "mean /= (float)dim;\n";
        code << indent(2) << "float var = 0.0f;\n";
        code << indent(2) << "for (int64_t j = 0; j < dim; j++) {\n";
        code << indent(3) << "float diff = input[i * dim + j] - mean;\n";
        code << indent(3) << "var += diff * diff;\n";
        code << indent(2) << "}\n";
        code << indent(2) << "var /= (float)dim;\n";
        code << indent(2) << "float inv = 1.0f / sqrtf(var + 1e-5f);\n";
        code << indent(2) << "for (int64_t j = 0; j < dim; j++) {\n";
        code << indent(3) << "output[i * dim + j] = ((input[i * dim + j] - mean) * inv) * weight[j];\n";
        code << indent(2) << "}\n";
    }

    code << indent(1) << "}\n";
    code << "}\n";
    return code.str();
}

std::string CodeGenerator::generate_rms_norm(int64_t dim, SIMDTarget simd) {
    return norm_kernel("quant_rms_norm", dim, true, simd);
}

std::string CodeGenerator::generate_layer_norm(int64_t dim, SIMDTarget simd) {
    return norm_kernel("quant_layer_norm", dim, false, simd);
}

// ========================================================================
// Activation kernels
// ========================================================================

std::string CodeGenerator::activation_kernel(const std::string& name, int64_t n, const std::string& op, SIMDTarget simd) {
    std::ostringstream code;
    code << "#include <cstdint>\n#include <cmath>\n";
    code << "\nextern \"C\" void " << name << "(const float* __restrict__ input,\n";
    code << "    float* __restrict__ output, int64_t n) {\n\n";

    if (simd != SIMDTarget::SCALAR) {
        int sw = simd_width(simd);
        code << indent(1) << "int64_t i = 0;\n";
        code << indent(1) << "for (; i + " << sw << " <= n; i += " << sw << ") {\n";
        code << indent(2) << "auto x = " << simd_prefix(simd) << "loadu_ps(&input[i]);\n";
        if (op == "relu") {
            code << indent(2) << "auto zero = " << simd_prefix(simd) << "setzero_ps();\n";
            code << indent(2) << "auto result = " << simd_prefix(simd) << "max_ps(x, zero);\n";
        } else if (op == "silu") {
            code << indent(2) << "auto neg_x = " << simd_prefix(simd) << "sub_ps(" << simd_prefix(simd) << "setzero_ps(), x);\n";
            code << indent(2) << "auto exp_neg = " << simd_prefix(simd) << "exp_ps(neg_x);\n";
            code << indent(2) << "auto one = " << simd_prefix(simd) << "set1_ps(1.0f);\n";
            code << indent(2) << "auto sigmoid = " << simd_prefix(simd) << "div_ps(one, " << simd_prefix(simd) << "add_ps(one, exp_neg));\n";
            code << indent(2) << "auto result = " << simd_prefix(simd) << "mul_ps(x, sigmoid);\n";
        } else if (op == "gelu") {
            code << indent(2) << "auto c = " << simd_prefix(simd) << "set1_ps(0.7978845608f);\n";
            code << indent(2) << "auto x3 = " << simd_prefix(simd) << "mul_ps(" << simd_prefix(simd) << "mul_ps(x, x), x);\n";
            code << indent(2) << "auto inner = " << simd_prefix(simd) << "fmadd_ps(c, x, " << simd_prefix(simd) << "mul_ps(" << simd_prefix(simd) << "set1_ps(0.044715f), x3));\n";
            code << indent(2) << "auto tanh_val = " << simd_prefix(simd) << "tanh_ps(inner);\n";
            code << indent(2) << "auto one = " << simd_prefix(simd) << "set1_ps(1.0f);\n";
            code << indent(2) << "auto result = " << simd_prefix(simd) << "mul_ps(" << simd_prefix(simd) << "mul_ps(x, " << simd_prefix(simd) << "add_ps(one, tanh_val)), " << simd_prefix(simd) << "set1_ps(0.5f));\n";
        } else {
            code << indent(2) << "auto result = x;\n";
        }
        code << indent(2) << simd_prefix(simd) << "storeu_ps(&output[i], result);\n";
        code << indent(1) << "}\n";
        code << indent(1) << "for (; i < n; i++) {\n";
    } else {
        code << indent(1) << "for (int64_t i = 0; i < n; i++) {\n";
    }

    if (op == "relu") {
        code << indent(2) << "output[i] = input[i] > 0.0f ? input[i] : 0.0f;\n";
    } else if (op == "silu") {
        code << indent(2) << "float x = input[i];\n";
        code << indent(2) << "output[i] = x / (1.0f + expf(-x));\n";
    } else if (op == "gelu") {
        code << indent(2) << "float x = input[i];\n";
        code << indent(2) << "output[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));\n";
    }

    code << indent(1) << "}\n";
    code << "}\n";
    return code.str();
}

std::string CodeGenerator::generate_relu(int64_t n, SIMDTarget simd) {
    return activation_kernel("quant_relu", n, "relu", simd);
}

std::string CodeGenerator::generate_silu(int64_t n, SIMDTarget simd) {
    return activation_kernel("quant_silu", n, "silu", simd);
}

std::string CodeGenerator::generate_gelu(int64_t n, SIMDTarget simd) {
    return activation_kernel("quant_gelu", n, "gelu", simd);
}

// ========================================================================
// Softmax
// ========================================================================

std::string CodeGenerator::generate_softmax(int64_t dim, SIMDTarget simd) {
    std::ostringstream code;
    code << "#include <cstdint>\n#include <cmath>\n";
    code << "\nextern \"C\" void quant_softmax(const float* __restrict__ input,\n";
    code << "    float* __restrict__ output, int64_t n) {\n\n";

    code << indent(1) << "for (int64_t i = 0; i < n; i++) {\n";
    code << indent(2) << "float max_val = input[i * " << dim << "];\n";
    code << indent(2) << "for (int64_t j = 1; j < " << dim << "; j++) {\n";
    code << indent(3) << "if (input[i * " << dim << " + j] > max_val) max_val = input[i * " << dim << " + j];\n";
    code << indent(2) << "}\n";
    code << indent(2) << "float sum = 0.0f;\n";
    code << indent(2) << "for (int64_t j = 0; j < " << dim << "; j++) {\n";
    code << indent(3) << "output[i * " << dim << " + j] = expf(input[i * " << dim << " + j] - max_val);\n";
    code << indent(3) << "sum += output[i * " << dim << " + j];\n";
    code << indent(2) << "}\n";
    code << indent(2) << "for (int64_t j = 0; j < " << dim << "; j++) {\n";
    code << indent(3) << "output[i * " << dim << " + j] /= (sum + 1e-10f);\n";
    code << indent(2) << "}\n";
    code << indent(1) << "}\n";
    code << "}\n";
    return code.str();
}

// ========================================================================
// Elementwise ops
// ========================================================================

std::string CodeGenerator::generate_add(int64_t n, SIMDTarget simd) {
    std::ostringstream code;
    code << "#include <cstdint>\n";
    if (simd != SIMDTarget::SCALAR) code << "#include <immintrin.h>\n";
    code << "\nextern \"C\" void quant_add(const float* __restrict__ a,\n";
    code << "    const float* __restrict__ b, float* __restrict__ c, int64_t n) {\n";

    if (simd != SIMDTarget::SCALAR) {
        int sw = simd_width(simd);
        code << indent(1) << "int64_t i = 0;\n";
        code << indent(1) << "for (; i + " << sw << " <= n; i += " << sw << ") {\n";
        code << indent(2) << "auto va = " << simd_prefix(simd) << "loadu_ps(&a[i]);\n";
        code << indent(2) << "auto vb = " << simd_prefix(simd) << "loadu_ps(&b[i]);\n";
        code << indent(2) << simd_prefix(simd) << "storeu_ps(&c[i], " << simd_prefix(simd) << "add_ps(va, vb));\n";
        code << indent(1) << "}\n";
        code << indent(1) << "for (; i < n; i++) c[i] = a[i] + b[i];\n";
    } else {
        code << indent(1) << "for (int64_t i = 0; i < n; i++) c[i] = a[i] + b[i];\n";
    }
    code << "}\n";
    return code.str();
}

std::string CodeGenerator::generate_mul(int64_t n, SIMDTarget simd) {
    std::ostringstream code;
    code << "#include <cstdint>\n";
    if (simd != SIMDTarget::SCALAR) code << "#include <immintrin.h>\n";
    code << "\nextern \"C\" void quant_mul(const float* __restrict__ a,\n";
    code << "    const float* __restrict__ b, float* __restrict__ c, int64_t n) {\n";

    if (simd != SIMDTarget::SCALAR) {
        int sw = simd_width(simd);
        code << indent(1) << "int64_t i = 0;\n";
        code << indent(1) << "for (; i + " << sw << " <= n; i += " << sw << ") {\n";
        code << indent(2) << "auto va = " << simd_prefix(simd) << "loadu_ps(&a[i]);\n";
        code << indent(2) << "auto vb = " << simd_prefix(simd) << "loadu_ps(&b[i]);\n";
        code << indent(2) << simd_prefix(simd) << "storeu_ps(&c[i], " << simd_prefix(simd) << "mul_ps(va, vb));\n";
        code << indent(1) << "}\n";
        code << indent(1) << "for (; i < n; i++) c[i] = a[i] * b[i];\n";
    } else {
        code << indent(1) << "for (int64_t i = 0; i < n; i++) c[i] = a[i] * b[i];\n";
    }
    code << "}\n";
    return code.str();
}

std::string CodeGenerator::generate_scale(int64_t n, float scale_val, SIMDTarget simd) {
    std::ostringstream code;
    code << "#include <cstdint>\n";
    if (simd != SIMDTarget::SCALAR) code << "#include <immintrin.h>\n";
    code << "\nextern \"C\" void quant_scale(const float* __restrict__ input,\n";
    code << "    float* __restrict__ output, int64_t n, float scale) {\n";

    if (simd != SIMDTarget::SCALAR) {
        int sw = simd_width(simd);
        code << indent(1) << "auto vs = " << simd_prefix(simd) << "set1_ps(scale);\n";
        code << indent(1) << "int64_t i = 0;\n";
        code << indent(1) << "for (; i + " << sw << " <= n; i += " << sw << ") {\n";
        code << indent(2) << "auto vx = " << simd_prefix(simd) << "loadu_ps(&input[i]);\n";
        code << indent(2) << simd_prefix(simd) << "storeu_ps(&output[i], " << simd_prefix(simd) << "mul_ps(vx, vs));\n";
        code << indent(1) << "}\n";
        code << indent(1) << "for (; i < n; i++) output[i] = input[i] * scale;\n";
    } else {
        code << indent(1) << "for (int64_t i = 0; i < n; i++) output[i] = input[i] * scale;\n";
    }
    code << "}\n";
    return code.str();
}

std::string CodeGenerator::generate_residual_add(int64_t n, SIMDTarget simd) {
    std::ostringstream code;
    code << "#include <cstdint>\n";
    if (simd != SIMDTarget::SCALAR) code << "#include <immintrin.h>\n";
    code << "\nextern \"C\" void quant_residual_add(const float* __restrict__ a,\n";
    code << "    const float* __restrict__ b, float* __restrict__ out, int64_t n) {\n";

    if (simd != SIMDTarget::SCALAR) {
        int sw = simd_width(simd);
        code << indent(1) << "int64_t i = 0;\n";
        code << indent(1) << "for (; i + " << sw << " <= n; i += " << sw << ") {\n";
        code << indent(2) << "auto va = " << simd_prefix(simd) << "loadu_ps(&a[i]);\n";
        code << indent(2) << "auto vb = " << simd_prefix(simd) << "loadu_ps(&b[i]);\n";
        code << indent(2) << simd_prefix(simd) << "storeu_ps(&out[i], " << simd_prefix(simd) << "add_ps(va, vb));\n";
        code << indent(1) << "}\n";
        code << indent(1) << "for (; i < n; i++) out[i] = a[i] + b[i];\n";
    } else {
        code << indent(1) << "for (int64_t i = 0; i < n; i++) out[i] = a[i] + b[i];\n";
    }
    code << "}\n";
    return code.str();
}

// ========================================================================
// Embedding + quantize
// ========================================================================

std::string CodeGenerator::generate_embedding_lookup(int64_t vocab_size, int64_t embed_dim, SIMDTarget simd) {
    std::ostringstream code;
    code << "#include <cstdint>\n";
    code << "\nextern \"C\" void quant_embedding_lookup(const float* __restrict__ table,\n";
    code << "    const int32_t* __restrict__ indices, float* __restrict__ output,\n";
    code << "    int64_t batch, int64_t embed_dim, int64_t vocab_size) {\n\n";

    code << indent(1) << "for (int64_t i = 0; i < batch; i++) {\n";
    code << indent(2) << "int32_t idx = indices[i];\n";
    code << indent(2) << "if (idx < 0 || idx >= vocab_size) {\n";
    code << indent(3) << "for (int64_t j = 0; j < embed_dim; j++) output[i * embed_dim + j] = 0.0f;\n";
    code << indent(2) << "} else {\n";
    code << indent(3) << "const float* row = &table[idx * embed_dim];\n";
    code << indent(3) << "for (int64_t j = 0; j < embed_dim; j++) output[i * embed_dim + j] = row[j];\n";
    code << indent(2) << "}\n";
    code << indent(1) << "}\n";
    code << "}\n";
    return code.str();
}

std::string CodeGenerator::generate_quantize_dequantize(int64_t n, Format src_format, Format dst_format) {
    std::ostringstream code;
    code << "#include <cstdint>\n#include <cmath>\n";
    code << "\nextern \"C\" void quant_quantize_dequantize(const float* __restrict__ input,\n";
    code << "    float* __restrict__ output, int64_t n) {\n\n";

    if (src_format == Format::Q32 && dst_format == Format::Q8) {
        code << indent(1) << "for (int64_t i = 0; i < n; i++) {\n";
        code << indent(2) << "float v = input[i];\n";
        code << indent(2) << "int q = (int)(v * 127.0f + 128.0f);\n";
        code << indent(2) << "q = q < 0 ? 0 : (q > 255 ? 255 : q);\n";
        code << indent(2) << "output[i] = ((float)q - 128.0f) / 127.0f;\n";
        code << indent(1) << "}\n";
    } else if (src_format == Format::Q32 && dst_format == Format::Q4) {
        code << indent(1) << "for (int64_t i = 0; i < n; i++) {\n";
        code << indent(2) << "float v = input[i];\n";
        code << indent(2) << "int q = (int)(v * 7.5f + 7.5f);\n";
        code << indent(2) << "q = q < 0 ? 0 : (q > 15 ? 15 : q);\n";
        code << indent(2) << "output[i] = ((float)q - 7.5f) / 7.5f;\n";
        code << indent(1) << "}\n";
        code << indent(1) << "for (int64_t i = 0; i < n; i++) {\n";
        code << indent(2) << "float v = input[i];\n";
        code << indent(2) << "output[i] = v > 0.33f ? 1.0f : (v < -0.33f ? -1.0f : 0.0f);\n";
        code << indent(1) << "}\n";
    } else if (src_format == Format::Q32 && dst_format == Format::Q1) {
        code << indent(1) << "for (int64_t i = 0; i < n; i++) {\n";
        code << indent(2) << "output[i] = input[i] >= 0.0f ? 1.0f : -1.0f;\n";
        code << indent(1) << "}\n";
    } else {
        code << indent(1) << "for (int64_t i = 0; i < n; i++) output[i] = input[i];\n";
    }
    code << "}\n";
    return code.str();
}

// ========================================================================
// Format dispatch
// ========================================================================

std::string CodeGenerator::generate_format_dispatch(const std::string& op_name, const std::vector<Format>& formats) {
    std::ostringstream code;
    code << "#include \"quant/types.h\"\n\n";
    code << "void " << op_name << "_dispatch(const void* input, void* output, int64_t n, quant::Format fmt) {\n";

    for (auto fmt : formats) {
        code << indent(1) << "if (fmt == quant::Format::" << format_name(fmt) << ") {\n";
        code << indent(2) << op_name << "_" << format_name(fmt) << "(input, output, n);\n";
        code << indent(1) << "}\n";
    }

    code << indent(1) << "// fallback\n";
    code << indent(1) << op_name << "_fp32(input, output, n);\n";
    code << "}\n";
    return code.str();
}

std::string CodeGenerator::generate_format_branch(const std::string& op_name, Format format, int64_t dim) {
    std::ostringstream code;
    code << "// " << op_name << " branch for " << format_name(format) << " dim=" << dim << "\n";

    switch (format) {
        case Format::Q1:
            code << "void " << op_name << "_quant1(const int8_t* in, int8_t* out, int64_t n) {\n";
            code << indent(1) << "for (int64_t i = 0; i < n; i++) out[i] = in[i] ^ 1;\n";
            code << "}\n";
            break;
            code << "void " << op_name << "_quant_q0(const int8_t* in, int8_t* out, int64_t n) {\n";
            code << indent(1) << "for (int64_t i = 0; i < n; i++) out[i] = -in[i];\n";
            code << "}\n";
            break;
        case Format::Q4:
            code << "void " << op_name << "_quant4(const uint8_t* in, float* out, int64_t n) {\n";
            code << indent(1) << "for (int64_t i = 0; i < n; i++) {\n";
            code << indent(2) << "float v = ((float)(in[i] & 0xF) - 7.5f) / 7.5f;\n";
            code << indent(2) << "out[i] = v;\n";
            code << indent(1) << "}\n";
            code << "}\n";
            break;
        case Format::Q8:
            code << "void " << op_name << "_quant8(const uint8_t* in, float* out, int64_t n) {\n";
            code << indent(1) << "for (int64_t i = 0; i < n; i++) {\n";
            code << indent(2) << "float v = ((float)in[i] - 128.0f) / 127.0f;\n";
            code << indent(2) << "out[i] = v;\n";
            code << indent(1) << "}\n";
            code << "}\n";
            break;
        default:
            code << "// " << format_name(format) << " is native, no conversion needed\n";
            break;
    }
    return code.str();
}

// ========================================================================
// Unrolling
// ========================================================================

std::string CodeGenerator::generate_unrolled_loop(const std::string& var, int64_t start, int64_t end,
                                                    int unroll_factor, const std::string& body) {
    std::ostringstream code;
    int64_t range = end - start;
    int64_t unrolled_end = start + (range / unroll_factor) * unroll_factor;

    code << "{\n";
    code << indent(1) << "int64_t " << var << ";\n";

    code << indent(1) << "for (" << var << " = " << start << "; " << var << " < " << unrolled_end << "; " << var << " += " << unroll_factor << ") {\n";
    for (int u = 0; u < unroll_factor; u++) {
        std::string offset = u == 0 ? "" : " + " + std::to_string(u);
        std::string iter_body = body;
        auto pos = iter_body.find("{{VAR}}");
        while (pos != std::string::npos) {
            iter_body.replace(pos, 7, var + offset);
            pos = iter_body.find("{{VAR}}");
        }
        code << indent(2) << iter_body << "\n";
    }
    code << indent(1) << "}\n";

    code << indent(1) << "for (; " << var << " < " << end << "; " << var << "++) {\n";
    std::string iter_body = body;
    auto pos = iter_body.find("{{VAR}}");
    while (pos != std::string::npos) {
        iter_body.replace(pos, 7, var);
        pos = iter_body.find("{{VAR}}");
    }
    code << indent(2) << iter_body << "\n";
    code << indent(1) << "}\n";
    code << "}\n";

    return code.str();
}

// ========================================================================
// SIMD code generation helpers
// ========================================================================

std::string CodeGenerator::generate_avx2_header() {
    std::ostringstream code;
    code << "#ifdef __AVX2__\n";
    code << "#include <immintrin.h>\n";
    code << "#define QUANT_HAS_AVX2 1\n";
    code << "inline __m256 quant_simd_load(const float* p) { return _mm256_loadu_ps(p); }\n";
    code << "inline void quant_simd_store(float* p, __m256 v) { _mm256_storeu_ps(p, v); }\n";
    code << "inline __m256 quant_simd_add(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }\n";
    code << "inline __m256 quant_simd_mul(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }\n";
    code << "inline __m256 quant_simd_fma(__m256 a, __m256 b, __m256 c) { return _mm256_fmadd_ps(a, b, c); }\n";
    code << "inline __m256 quant_simd_set1(float v) { return _mm256_set1_ps(v); }\n";
    code << "inline __m256 quant_simd_setzero() { return _mm256_setzero_ps(); }\n";
    code << "inline __m256 quant_simd_max(__m256 a, __m256 b) { return _mm256_max_ps(a, b); }\n";
    code << "inline __m256 quant_simd_exp(__m256 x) {\n";
    code << "    // Polynomial approximation for exp\n";
    code << "    __m256 clamp = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(88.0f)), _mm256_set1_ps(-88.0f));\n";
    code << "    __m256 fx = _mm256_mul_ps(clamp, _mm256_set1_ps(1.442695041f));\n";
    code << "    __m256 i = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT);\n";
    code << "    __m256 f = _mm256_sub_ps(fx, i);\n";
    code << "    __m256 p = _mm256_set1_ps(1.9875691500e-4f);\n";
    code << "    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.3981999507e-3f));\n";
    code << "    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(8.3334519073e-3f));\n";
    code << "    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(4.1665795894e-2f));\n";
    code << "    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.6666665459e-1f));\n";
    code << "    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.0000001201e-1f));\n";
    code << "    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));\n";
    code << "    __m256 two_pow_i = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(i), _mm256_set1_epi32(127)), 23));\n";
    code << "    return _mm256_mul_ps(p, two_pow_i);\n";
    code << "}\n";
    code << "#else\n";
    code << "#define QUANT_HAS_AVX2 0\n";
    code << "#endif\n";
    return code.str();
}

std::string CodeGenerator::generate_avx512_header() {
    std::ostringstream code;
    code << "#ifdef __AVX512F__\n";
    code << "#include <immintrin.h>\n";
    code << "#define QUANT_HAS_AVX512 1\n";
    code << "inline __m512 quant_simd512_load(const float* p) { return _mm512_loadu_ps(p); }\n";
    code << "inline void quant_simd512_store(float* p, __m512 v) { _mm512_storeu_ps(p, v); }\n";
    code << "inline __m512 quant_simd512_add(__m512 a, __m512 b) { return _mm512_add_ps(a, b); }\n";
    code << "inline __m512 quant_simd512_mul(__m512 a, __m512 b) { return _mm512_mul_ps(a, b); }\n";
    code << "inline __m512 quant_simd512_fma(__m512 a, __m512 b, __m512 c) { return _mm512_fmadd_ps(a, b, c); }\n";
    code << "inline __m512 quant_simd512_set1(float v) { return _mm512_set1_ps(v); }\n";
    code << "inline __m512 quant_simd512_setzero() { return _mm512_setzero_ps(); }\n";
    code << "inline __m512 quant_simd512_max(__m512 a, __m512 b) { return _mm512_max_ps(a, b); }\n";
    code << "inline __m512 quant_simd512_exp(__m512 x) {\n";
    code << "    __m512 clamp = _mm512_max_ps(_mm512_min_ps(x, _mm512_set1_ps(88.0f)), _mm512_set1_ps(-88.0f));\n";
    code << "    __m512 fx = _mm512_mul_ps(clamp, _mm512_set1_ps(1.442695041f));\n";
    code << "    __m512i i = _mm512_cvtps_epi32(_mm512_roundscale_ps(fx, _MM_FROUND_TO_NEAREST_INT));\n";
    code << "    __m512 f = _mm512_sub_ps(fx, _mm512_cvtepi32_ps(i));\n";
    code << "    __m512 p = _mm512_set1_ps(1.0f);\n";
    code << "    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));\n";
    code << "    __m512 two_pow_i = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(i, _mm512_set1_epi32(127)), 23));\n";
    code << "    return _mm512_mul_ps(p, two_pow_i);\n";
    code << "}\n";
    code << "#else\n";
    code << "#define QUANT_HAS_AVX512 0\n";
    code << "#endif\n";
    return code.str();
}

std::string CodeGenerator::generate_neon_header() {
    std::ostringstream code;
    code << "#ifdef __ARM_NEON\n";
    code << "#include <arm_neon.h>\n";
    code << "#define QUANT_HAS_NEON 1\n";
    code << "inline float32x4_t quant_neon_load(const float* p) { return vld1q_f32(p); }\n";
    code << "inline void quant_neon_store(float* p, float32x4_t v) { vst1q_f32(p, v); }\n";
    code << "inline float32x4_t quant_neon_add(float32x4_t a, float32x4_t b) { return vaddq_f32(a, b); }\n";
    code << "inline float32x4_t quant_neon_mul(float32x4_t a, float32x4_t b) { return vmulq_f32(a, b); }\n";
    code << "inline float32x4_t quant_neon_fma(float32x4_t a, float32x4_t b, float32x4_t c) { return vmlaq_f32(c, a, b); }\n";
    code << "inline float32x4_t quant_neon_set1(float v) { return vdupq_n_f32(v); }\n";
    code << "inline float32x4_t quant_neon_setzero() { return vdupq_n_f32(0.0f); }\n";
    code << "#else\n";
    code << "#define QUANT_HAS_NEON 0\n";
    code << "#endif\n";
    return code.str();
}

std::string CodeGenerator::generate_includes(const std::vector<std::string>& includes) {
    std::ostringstream code;
    for (auto& inc : includes) {
        code << "#include " << inc << "\n";
    }
    return code.str();
}

std::string CodeGenerator::generate_simd_load(const std::string& var, const std::string& ptr, int width) {
    if (width == 8) return "auto " + var + " = _mm256_loadu_ps(" + ptr + ");";
    if (width == 16) return "auto " + var + " = _mm512_loadu_ps(" + ptr + ");";
    if (width == 4) return "auto " + var + " = vld1q_f32(" + ptr + ");";
    return "auto " + var + " = *((" + ptr + "));";
}

std::string CodeGenerator::generate_simd_store(const std::string& ptr, const std::string& var, int width) {
    if (width == 8) return "_mm256_storeu_ps(" + ptr + ", " + var + ");";
    if (width == 16) return "_mm512_storeu_ps(" + ptr + ", " + var + ");";
    if (width == 4) return "vst1q_f32(" + ptr + ", " + var + ");";
    return "*(" + ptr + ") = " + var + ";";
}

std::string CodeGenerator::generate_simd_fma(const std::string& dst, const std::string& a,
                                              const std::string& b, const std::string& c) {
    return dst + " = _mm256_fmadd_ps(" + a + ", " + b + ", " + c + ");";
}

std::string CodeGenerator::generate_simd_add(const std::string& dst, const std::string& a,
                                              const std::string& b, int width) {
    if (width == 8) return dst + " = _mm256_add_ps(" + a + ", " + b + ");";
    if (width == 16) return dst + " = _mm512_add_ps(" + a + ", " + b + ");";
    return dst + " = " + a + " + " + b + ";";
}

std::string CodeGenerator::generate_simd_mul(const std::string& dst, const std::string& a,
                                              const std::string& b, int width) {
    if (width == 8) return dst + " = _mm256_mul_ps(" + a + ", " + b + ");";
    if (width == 16) return dst + " = _mm512_mul_ps(" + a + ", " + b + ");";
    return dst + " = " + a + " * " + b + ";";
}

// ========================================================================
// High-level generation API
// ========================================================================

std::string CodeGenerator::generate_kernel_for_shape(const KernelSpec& spec) {
    std::string cached_name = make_function_name(spec);
    auto it = kernel_cache_.find(cached_name);
    if (it != kernel_cache_.end()) return it->second.source_code;

    std::string code;
    switch (spec.type) {
        case KernelType::GEMM:
            code = generate_gemm(spec.M, spec.N, spec.K, spec.simd);
            break;
        case KernelType::ATTENTION:
            code = generate_attention(spec.M, spec.N, static_cast<int>(spec.K), spec.simd);
            break;
        case KernelType::RMS_NORM:
            code = generate_rms_norm(spec.N, spec.simd);
            break;
        case KernelType::LAYER_NORM:
            code = generate_layer_norm(spec.N, spec.simd);
            break;
        case KernelType::SOFTMAX:
            code = generate_softmax(spec.N, spec.simd);
            break;
        case KernelType::RELU:
            code = generate_relu(spec.N, spec.simd);
            break;
        case KernelType::SILU:
            code = generate_silu(spec.N, spec.simd);
            break;
        case KernelType::GELU:
            code = generate_gelu(spec.N, spec.simd);
            break;
        case KernelType::ADD:
            code = generate_add(spec.N, spec.simd);
            break;
        case KernelType::MUL:
            code = generate_mul(spec.N, spec.simd);
            break;
        case KernelType::RESIDUAL_ADD:
            code = generate_residual_add(spec.N, spec.simd);
            break;
        case KernelType::EMBEDDING_LOOKUP:
            code = generate_embedding_lookup(spec.M, spec.N, spec.simd);
            break;
        case KernelType::QUANTIZE_DEQUANTIZE:
            code = generate_quantize_dequantize(spec.N, Format::Q32, spec.format);
            break;
        default:
            code = "// Unsupported kernel type\n";
            break;
    }

    GeneratedKernel gk;
    gk.spec = spec;
    gk.source_code = code;
    gk.function_name = cached_name;
    kernel_cache_[cached_name] = gk;
    return code;
}

GeneratedKernel CodeGenerator::generate_optimized_kernel(const KernelSpec& spec) {
    GeneratedKernel gk;
    gk.spec = spec;
    gk.source_code = generate_kernel_for_shape(spec);
    gk.function_name = make_function_name(spec);
    kernel_cache_[gk.function_name] = gk;
    return gk;
}

GeneratedKernel CodeGenerator::specialize_for_shape(KernelType type, int64_t M, int64_t N, int64_t K) {
    KernelSpec spec;
    spec.type = type;
    spec.M = M; spec.N = N; spec.K = K;

    if (spec.simd == SIMDTarget::SCALAR) {
        if (N % 16 == 0) spec.simd = SIMDTarget::AVX512;
        else if (N % 8 == 0) spec.simd = SIMDTarget::AVX2;
    }

    return generate_optimized_kernel(spec);
}

// ========================================================================
// Code validation
// ========================================================================

CodeValidationResult CodeGenerator::validate(const std::string& code) {
    CodeValidationResult result;
    result.syntax_valid = true;

    bool has_include = code.find("#include") != std::string::npos;
    bool has_function = code.find("extern \"C\"") != std::string::npos || code.find("void ") != std::string::npos;
    bool has_return = code.find("return ") != std::string::npos || code.find("void ") != std::string::npos;
    bool balanced = true;
    int brace_count = 0;
    for (char c : code) {
        if (c == '{') brace_count++;
        if (c == '}') brace_count--;
        if (brace_count < 0) { balanced = false; break; }
    }
    if (brace_count != 0) balanced = false;

    if (!has_include || !has_function || !balanced) {
        result.syntax_valid = false;
        result.error_message = "Syntax check failed: ";
        if (!has_include) result.error_message += "missing includes; ";
        if (!has_function) result.error_message += "missing function; ";
        if (!balanced) result.error_message += "unbalanced braces";
    }

    namespace fs = std::filesystem;
    fs::path sandbox;
#ifdef _WIN32
    const char* tmp = std::getenv("TEMP");
    sandbox = fs::path(tmp ? tmp : "C:\\Temp") / "InNova_codegen";
#else
    sandbox = fs::path("/tmp") / "InNova_codegen";
#endif

    std::error_code ec;
    fs::create_directories(sandbox, ec);
    if (ec) { result.compiles = false; return result; }

    auto src_path = sandbox / "kernel_test.cpp";
    {
        std::ofstream ofs(src_path);
        if (!ofs) { result.compiles = false; return result; }
        ofs << code;
    }

#ifdef _WIN32
    std::string cmd = "cl.exe /nologo /EHsc /c \"" + src_path.string() + "\" 2>nul";
#else
    std::string cmd = "g++ -x c++ -std=c++20 -c -o /dev/null \"" + src_path.string() + "\" 2>/dev/null";
#endif
    int ret = std::system(cmd.c_str());
    result.compiles = (ret == 0);
    if (!result.compiles) result.error_message = "Compilation failed";

    fs::remove(src_path, ec);
    fs::remove(sandbox / "kernel_test.obj", ec);
    fs::remove(sandbox / "kernel_test.o", ec);
    return result;
}

bool CodeGenerator::compile_code(const std::string& code) {
    auto result = validate(code);
    return result.syntax_valid && result.compiles;
}

bool CodeGenerator::run_test(const std::string& code, const std::string& test_name) {
    return compile_code(code);
}

// ========================================================================
// Slow path identification and optimization
// ========================================================================

std::vector<SlowPath> CodeGenerator::identify_slow_paths(double threshold_ms) {
    std::vector<SlowPath> slow;
    if (!model_) return slow;

    std::vector<std::pair<std::string, double>> ops = {
        {"gemm", 0.5}, {"attention", 2.0}, {"softmax", 0.3},
        {"rms_norm", 0.1}, {"silu", 0.1}, {"residual_add", 0.05}
    };

    for (auto& [name, time] : ops) {
        if (time > threshold_ms) {
            SlowPath sp;
            sp.op_name = name;
            sp.current_time_ms = time;
            sp.target_time_ms = threshold_ms;
            sp.priority = (int)((time / threshold_ms) * 10);
            slow.push_back(sp);
        }
    }

    std::sort(slow.begin(), slow.end(),
              [](const SlowPath& a, const SlowPath& b) { return a.priority > b.priority; });
    return slow;
}

std::string CodeGenerator::improve_slow_path(const SlowPath& path) {
    KernelSpec spec;
    spec.name = path.op_name;

    if (path.op_name == "gemm") {
        spec.type = KernelType::GEMM;
        spec.M = 128; spec.N = 128; spec.K = 768;
        spec.simd = SIMDTarget::AVX512;
        spec.block_size = 64;
    } else if (path.op_name == "attention") {
        spec.type = KernelType::ATTENTION;
        spec.M = 512; spec.N = 64; spec.K = 12;
        spec.simd = SIMDTarget::AVX2;
    } else if (path.op_name == "softmax") {
        spec.type = KernelType::SOFTMAX;
        spec.N = 512;
        spec.simd = SIMDTarget::AVX2;
    } else if (path.op_name == "rms_norm") {
        spec.type = KernelType::RMS_NORM;
        spec.N = 4096;
        spec.simd = SIMDTarget::AVX512;
    } else if (path.op_name == "silu") {
        spec.type = KernelType::SILU;
        spec.N = 4096;
        spec.simd = SIMDTarget::AVX512;
    } else if (path.op_name == "residual_add") {
        spec.type = KernelType::RESIDUAL_ADD;
        spec.N = 4096;
        spec.simd = SIMDTarget::AVX512;
    }

    return generate_kernel_for_shape(spec);
}

std::vector<GeneratedKernel> CodeGenerator::auto_optimize(int n_iterations) {
    std::vector<GeneratedKernel> optimized;
    auto slow = identify_slow_paths(0.5);

    for (auto& sp : slow) {
        std::string code = improve_slow_path(sp);
        KernelSpec spec;
        spec.name = sp.op_name;

        if (sp.op_name == "gemm") spec.type = KernelType::GEMM;
        else if (sp.op_name == "attention") spec.type = KernelType::ATTENTION;
        else if (sp.op_name == "softmax") spec.type = KernelType::SOFTMAX;
        else if (sp.op_name == "rms_norm") spec.type = KernelType::RMS_NORM;
        else if (sp.op_name == "silu") spec.type = KernelType::SILU;
        else if (sp.op_name == "residual_add") spec.type = KernelType::RESIDUAL_ADD;

        GeneratedKernel gk;
        gk.spec = spec;
        gk.source_code = code;
        gk.function_name = sp.op_name + "_optimized";
        kernel_cache_[gk.function_name] = gk;
        optimized.push_back(gk);
    }

    for (int iter = 0; iter < n_iterations; iter++) {
        SIMDTarget targets[] = {SIMDTarget::AVX2, SIMDTarget::AVX512};
        KernelType types[] = {KernelType::GEMM, KernelType::ATTENTION, KernelType::SOFTMAX,
                              KernelType::RMS_NORM, KernelType::SILU, KernelType::RESIDUAL_ADD};
        int t = iter % 6;
        int s = (iter / 6) % 2;

        KernelSpec spec;
        spec.type = types[t];
        spec.simd = targets[s];
        spec.M = 64 * (1 + iter % 4);
        spec.N = 256;
        spec.K = 768;

        GeneratedKernel gk = generate_optimized_kernel(spec);
        optimized.push_back(gk);
    }

    return optimized;
}

// ========================================================================
// Templates
// ========================================================================

void CodeGenerator::init_default_templates() {
    templates_.clear();

    CodeTemplate gemm_tmpl;
    gemm_tmpl.name = "gemm_basic";
    gemm_tmpl.description = "Basic GEMM: C = A * B";
    gemm_tmpl.kernel_type = KernelType::GEMM;
    gemm_tmpl.placeholders = {"{{M}}", "{{N}}", "{{K}}", "{{TILE_SIZE}}"};
    gemm_tmpl.template_code =
        "#include <cstdint>\n"
        "extern \"C\" void gemm(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K) {\n"
        "    const int T = {{TILE_SIZE}};\n"
        "    for (int64_t i = 0; i < M; i += T) {\n"
        "        for (int64_t j = 0; j < N; j += T) {\n"
        "            for (int64_t k = 0; k < K; k += T) {\n"
        "                for (int64_t ii = i; ii < min(i+T,M); ii++)\n"
        "                    for (int64_t jj = j; jj < min(j+T,N); jj++) {\n"
        "                        float sum = 0;\n"
        "                        for (int64_t kk = k; kk < min(k+T,K); kk++)\n"
        "                            sum += A[ii*K+kk] * B[kk*N+jj];\n"
        "                        C[ii*N+jj] += sum;\n"
        "                    }\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}\n";
    templates_.push_back(gemm_tmpl);

    CodeTemplate attention_tmpl;
    attention_tmpl.name = "attention_basic";
    attention_tmpl.description = "Scaled dot-product attention";
    attention_tmpl.kernel_type = KernelType::ATTENTION;
    attention_tmpl.placeholders = {"{{SEQ_LEN}}", "{{HEAD_DIM}}", "{{NUM_HEADS}}"};
    attention_tmpl.template_code =
        "#include <cstdint>\n#include <cmath>\n"
        "extern \"C\" void attention(const float* Q, const float* K, const float* V, float* out,\n"
        "    int64_t seq_len, int64_t head_dim, int num_heads) {\n"
        "    float scale = 1.0f / sqrtf((float)head_dim);\n"
        "    for (int h = 0; h < num_heads; h++) {\n"
        "        for (int64_t i = 0; i < seq_len; i++) {\n"
        "            float max_s = -1e30; float scores[{{SEQ_LEN}}];\n"
        "            for (int64_t j = 0; j < seq_len; j++) {\n"
        "                float dot = 0;\n"
        "                for (int64_t d = 0; d < head_dim; d++)\n"
        "                    dot += Q[i*head_dim+d] * K[j*head_dim+d];\n"
        "                scores[j] = dot * scale;\n"
        "                if (scores[j] > max_s) max_s = scores[j];\n"
        "            }\n"
        "            float sum = 0;\n"
        "            for (int64_t j = 0; j < seq_len; j++) { scores[j] = expf(scores[j]-max_s); sum += scores[j]; }\n"
        "            for (int64_t j = 0; j < seq_len; j++) scores[j] /= (sum+1e-10f);\n"
        "            for (int64_t d = 0; d < head_dim; d++) {\n"
        "                float v = 0;\n"
        "                for (int64_t j = 0; j < seq_len; j++) v += scores[j] * V[j*head_dim+d];\n"
        "                out[i*head_dim+d] = v;\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}\n";
    templates_.push_back(attention_tmpl);

    CodeTemplate rms_tmpl;
    rms_tmpl.name = "rms_norm_basic";
    rms_tmpl.description = "RMS normalization";
    rms_tmpl.kernel_type = KernelType::RMS_NORM;
    rms_tmpl.placeholders = {"{{DIM}}"};
    rms_tmpl.template_code =
        "#include <cstdint>\n#include <cmath>\n"
        "extern \"C\" void rms_norm(const float* in, float* out, const float* w, int64_t dim) {\n"
        "    float sum_sq = 0;\n"
        "    for (int64_t j = 0; j < dim; j++) sum_sq += in[j] * in[j];\n"
        "    float rms = sqrtf(sum_sq / (float)dim + 1e-5f);\n"
        "    for (int64_t j = 0; j < dim; j++) out[j] = (in[j] / rms) * w[j];\n"
        "}\n";
    templates_.push_back(rms_tmpl);

    CodeTemplate silu_tmpl;
    silu_tmpl.name = "silu_basic";
    silu_tmpl.description = "SiLU activation";
    silu_tmpl.kernel_type = KernelType::SILU;
    silu_tmpl.placeholders = {"{{N}}"};
    silu_tmpl.template_code =
        "#include <cstdint>\n#include <cmath>\n"
        "extern \"C\" void silu(const float* in, float* out, int64_t n) {\n"
        "    for (int64_t i = 0; i < n; i++) {\n"
        "        float x = in[i];\n"
        "        out[i] = x / (1.0f + expf(-x));\n"
        "    }\n"
        "}\n";
    templates_.push_back(silu_tmpl);
}

void CodeGenerator::register_template(const CodeTemplate& tmpl) {
    for (auto& t : templates_) {
        if (t.name == tmpl.name) {
            t = tmpl;
            return;
        }
    }
    templates_.push_back(tmpl);
}

std::string CodeGenerator::apply_template(const std::string& template_name,
                                            const std::unordered_map<std::string, std::string>& vars) {
    for (auto& tmpl : templates_) {
        if (tmpl.name == template_name) {
            std::string code = tmpl.template_code;
            for (auto& [key, val] : vars) {
                auto pos = code.find(key);
                while (pos != std::string::npos) {
                    code.replace(pos, key.size(), val);
                    pos = code.find(key, pos + val.size());
                }
            }
            return code;
        }
    }
    return "// Template not found: " + template_name + "\n";
}

std::vector<std::string> CodeGenerator::get_available_templates() const {
    std::vector<std::string> names;
    for (auto& t : templates_) names.push_back(t.name);
    return names;
}

// ========================================================================
// Kernel cache management
// ========================================================================

GeneratedKernel CodeGenerator::get_kernel(const std::string& name) const {
    auto it = kernel_cache_.find(name);
    if (it != kernel_cache_.end()) return it->second;
    return {};
}

std::vector<GeneratedKernel> CodeGenerator::get_all_kernels() const {
    std::vector<GeneratedKernel> all;
    for (auto& [name, k] : kernel_cache_) all.push_back(k);
    return all;
}

void CodeGenerator::clear_kernels() {
    kernel_cache_.clear();
}

// ========================================================================
// Full source output
// ========================================================================

std::string CodeGenerator::generate_full_source(const std::vector<GeneratedKernel>& kernels) {
    std::ostringstream code;
    code << "// Auto-generated QUANT kernel collection\n";
    code << "// Generated by InNova CodeGenerator\n";
    code << "#include <cstdint>\n#include <cmath>\n";
    code << "#include <immintrin.h>\n\n";

    for (auto& k : kernels) {
        code << "// ========================================================\n";
        code << "// " << k.function_name << "\n";
        code << "// ========================================================\n";
        code << k.source_code << "\n\n";
    }

    return code.str();
}

bool CodeGenerator::save_source(const std::string& path, const std::string& source) {
    std::ofstream ofs(path);
    if (!ofs) return false;
    ofs << source;
    return true;
}

} // namespace code_gen
} // namespace quant
