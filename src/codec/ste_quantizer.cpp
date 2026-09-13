#include "quant/ste_quantizer.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace quant {

STEQuantizer::STEQuantizer(Format target_format) : target_format_(target_format) {}

void STEQuantizer::set_target_format(Format fmt) {
    target_format_ = fmt;
}

Format STEQuantizer::target_format() const {
    return target_format_;
}

Tensor STEQuantizer::forward(const Tensor& fp32_weight) {
    // Real STE: quantize → dequantize → return
    // Forward pass sees quantization noise
    // Backward (STE): gradient flows through unchanged
    int64_t n = fp32_weight.numel();
    const float* src = (const float*)fp32_weight.data();
    Tensor result(Shape{n}, DType::F32);
    float* rd = (float*)result.data();

    switch (target_format_) {
        case Format::Q1: {
            std::vector<uint8_t> packed((n + 7) / 8);
            float scale;
            quantize_Q1(src, packed.data(), &scale, n);
            for (int64_t i = 0; i < n; i++) {
                int v = (packed[(size_t)i / 8] >> (i % 8)) & 1;
                rd[i] = (float)(v == 0 ? -1 : 1) * scale;
            }
            break;
        }
        case Format::Q2: {
            std::vector<uint8_t> packed((n + 3) / 4);
            float scale;
            quantize_quant(src, packed.data(), &scale, n);
            for (int64_t i = 0; i < n; i++) {
                int v = (packed[(size_t)i / 4] >> (2 * (i % 4))) & 3;
                rd[i] = (float)(v == 1 ? 1 : v == 2 ? -1 : 0) * scale;
            }
            break;
        }
        case Format::Q3: {
            CodebookQ3 cb3;
            cb3.train(src, (size_t)n);
            for (int64_t i = 0; i < n; i++)
                rd[i] = cb3.dequantize(cb3.quantize(src[i]));
            break;
        }
        case Format::Q8: {
            CodebookQUANT8 cb8;
            cb8.train(src, (size_t)n);
            for (int64_t i = 0; i < n; i++)
                rd[i] = cb8.dequantize(cb8.quantize(src[i]));
            break;
        }
        case Format::Q4: {
            CodebookQUANT4 cb4;
            cb4.train(src, (size_t)n);
            for (int64_t i = 0; i < n; i++)
                rd[i] = cb4.dequantize(cb4.quantize(src[i]));
            break;
        }
        case Format::Q6: {
            CodebookQ6 cb6;
            cb6.train(src, (size_t)n);
            for (int64_t i = 0; i < n; i++)
                rd[i] = cb6.dequantize(cb6.quantize(src[i]));
            break;
        }
        case Format::Q12: {
            CodebookQ12 cb12;
            cb12.train(src, (size_t)n);
            for (int64_t i = 0; i < n; i++)
                rd[i] = cb12.dequantize(cb12.quantize(src[i]));
            break;
        }
        case Format::Q16: {
            // Explicit high-precision passthrough (never falls to default).
            std::memcpy(rd, src, (size_t)n * sizeof(float));
            break;
        }
        case Format::Q24: {
            // Undecided high-BPW passthrough (explicit, not silent default).
            std::memcpy(rd, src, (size_t)n * sizeof(float));
            break;
        }
        case Format::Q32: {
            std::memcpy(rd, src, (size_t)n * sizeof(float));
            break;
        }
        default: {
            // ONLY Q32 (+ undecided high-BPW) may memcpy silently above.
            // K/GRP/half/MX aliases have no STE kernel: log + passthrough.
            std::fprintf(stderr,
                "[STEQuantizer::forward] unhandled format %s (%d) "
                "[k=%d grp=%d half=%d mx=%d]; passthrough memcpy\n",
                format_name(target_format_), (int)target_format_,
                (int)format_is_k(target_format_), (int)format_is_grp(target_format_),
                (int)format_is_half(target_format_), (int)format_is_mx(target_format_));
            std::memcpy(rd, src, (size_t)n * sizeof(float));
            break;
        }
    }
    return result;
}

float STEQuantizer::find_scale(const float* data, int64_t n) {
    float max_abs = 0;
    for (int64_t i = 0; i < n; i++) {
        float a = std::fabs(data[i]);
        if (a > max_abs) max_abs = a;
    }
    return max_abs > 1e-10f ? max_abs : 1.0f;
}

Tensor STEQuantizer::quantize_with_codebook(const Tensor& fp32_weight, CodebookQUANT8& codebook) {
    int64_t n = fp32_weight.numel();
    const float* src = (const float*)fp32_weight.data();

    // Unconditional per-tensor train (no needs_train heuristic).
    codebook.train(src, (size_t)n);

    Tensor quantized(Shape{n}, DType::U8);
    uint8_t* dst = (uint8_t*)quantized.data();

    for (int64_t i = 0; i < n; i++) {
        dst[i] = (uint8_t)codebook.quantize(src[i]);
    }

    // Dequantize back to fp32 for STE (differentiable approximation)
    Tensor result(Shape{n}, DType::F32);
    float* rd = (float*)result.data();
    for (int64_t i = 0; i < n; i++) {
        rd[i] = codebook.dequantize(dst[i]);
    }

    return result;
}

Tensor STEQuantizer::quantize_with_codebook(const Tensor& fp32_weight, CodebookQUANT4& codebook) {
    int64_t n = fp32_weight.numel();
    const float* src = (const float*)fp32_weight.data();

    // Unconditional per-tensor train (no needs_train heuristic).
    codebook.train(src, (size_t)n);

    int packed_bytes = (int)((n + 1) / 2);
    Tensor quantized(Shape{packed_bytes}, DType::U4);
    uint8_t* dst = (uint8_t*)quantized.data();
    std::memset(dst, 0, (size_t)packed_bytes);

    for (int64_t i = 0; i < n; i++) {
        uint8_t idx = (uint8_t)codebook.quantize(src[i]);
        if (i % 2 == 0) {
            dst[i / 2] = (dst[i / 2] & 0xF0) | (idx & 0x0F);
        } else {
            dst[i / 2] = (dst[i / 2] & 0x0F) | ((idx & 0x0F) << 4);
        }
    }

    Tensor result(Shape{n}, DType::F32);
    float* rd = (float*)result.data();
    for (int64_t i = 0; i < n; i++) {
        uint8_t idx;
        if (i % 2 == 0) idx = dst[i / 2] & 0x0F;
        else idx = (dst[i / 2] >> 4) & 0x0F;
        rd[i] = codebook.dequantize(idx);
    }

    return result;
}

Tensor STEQuantizer::forward_mixed(const Tensor& weights, const std::vector<Format>& per_block_formats, int block_size) {
    int64_t n = weights.numel();
    const float* src = (const float*)weights.data();
    Tensor result(Shape{n}, DType::F32);
    float* rd = (float*)result.data();

    int64_t num_blocks = (int64_t)per_block_formats.size();

    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t block_start = b * block_size;
        int64_t block_end = std::min(block_start + block_size, n);
        int64_t block_n = block_end - block_start;
        if (block_n <= 0) break;

        Format fmt = per_block_formats[(size_t)b];

        switch (fmt) {
        case Format::Q2: {
                std::vector<uint8_t> packed((block_n + 3) / 4);
                float scale;
                quantize_quant(src + block_start, packed.data(), &scale, block_n);
                for (int64_t i = 0; i < block_n; i++) {
                    int v = (packed[(size_t)i / 4] >> (2 * (i % 4))) & 3;
                    rd[block_start + i] = (float)(v == 1 ? 1 : v == 2 ? -1 : 0) * scale;
                }
                break;
            }
            case Format::Q1: {
                std::vector<uint8_t> packed((block_n + 7) / 8);
                float scale;
                quantize_Q1(src + block_start, packed.data(), &scale, block_n);
                for (int64_t i = 0; i < block_n; i++) {
                    int v = (packed[(size_t)i / 8] >> (i % 8)) & 1;
                    rd[block_start + i] = (float)(v == 0 ? -1 : 1) * scale;
                }
                break;
            }
            case Format::Q3: {
                CodebookQ3 cb;
                cb.train(src + block_start, (size_t)block_n);
                for (int64_t i = 0; i < block_n; i++)
                    rd[block_start + i] = cb.dequantize(cb.quantize(src[block_start + i]));
                break;
            }
            case Format::Q8: {
                CodebookQUANT8 cb;
                cb.train(src + block_start, (size_t)block_n);
                for (int64_t i = 0; i < block_n; i++)
                    rd[block_start + i] = cb.dequantize(cb.quantize(src[block_start + i]));
                break;
            }
            case Format::Q4: {
                CodebookQUANT4 cb;
                cb.train(src + block_start, (size_t)block_n);
                for (int64_t i = 0; i < block_n; i++)
                    rd[block_start + i] = cb.dequantize(cb.quantize(src[block_start + i]));
                break;
            }
            case Format::Q6: {
                CodebookQ6 cb;
                cb.train(src + block_start, (size_t)block_n);
                for (int64_t i = 0; i < block_n; i++)
                    rd[block_start + i] = cb.dequantize(cb.quantize(src[block_start + i]));
                break;
            }
            case Format::Q12: {
                CodebookQ12 cb;
                cb.train(src + block_start, (size_t)block_n);
                for (int64_t i = 0; i < block_n; i++)
                    rd[block_start + i] = cb.dequantize(cb.quantize(src[block_start + i]));
                break;
            }
            case Format::Q16: {
                // Explicit high-precision passthrough (never falls to default).
                std::memcpy(rd + block_start, src + block_start, (size_t)block_n * sizeof(float));
                break;
            }
            case Format::Q24: {
                // Undecided high-BPW passthrough (explicit, not silent default).
                std::memcpy(rd + block_start, src + block_start, (size_t)block_n * sizeof(float));
                break;
            }
            case Format::Q32: {
                std::memcpy(rd + block_start, src + block_start, (size_t)block_n * sizeof(float));
                break;
            }
            default: {
                // ONLY Q32 (+ undecided high-BPW) may memcpy silently above.
                // K/GRP/half/MX aliases have no per-block STE kernel: log + passthrough.
                std::fprintf(stderr,
                    "[STEQuantizer::forward_mixed] unhandled format %s (%d) "
                    "[k=%d grp=%d half=%d mx=%d]; passthrough memcpy\n",
                    format_name(fmt), (int)fmt,
                    (int)format_is_k(fmt), (int)format_is_grp(fmt),
                    (int)format_is_half(fmt), (int)format_is_mx(fmt));
                std::memcpy(rd + block_start, src + block_start, (size_t)block_n * sizeof(float));
                break;
            }
        }
    }
    return result;
}

void STEQuantizer::quantize_quant(const float* src, uint8_t* dst, float* scale, int64_t n) {
    float s = find_scale(src, n);
    float threshold = s * 0.5f;
    *scale = s;

    int packed_size = (int)((n + 3) / 4);
    std::memset(dst, 0, (size_t)packed_size);

    for (int64_t i = 0; i < n; i++) {
        uint8_t val;
        if (src[i] > threshold) {
            val = 1;
        } else if (src[i] < -threshold) {
            val = 2; // -1 mapped to 2 in 2-bit encoding
        } else {
            val = 0;
        }
        dst[i / 4] |= (val & 0x03) << (2 * (i % 4));
    }
}

void STEQuantizer::quantize_Q1(const float* src, uint8_t* dst, float* scale, int64_t n) {
    float s = find_scale(src, n);
    *scale = s;

    int packed_size = (int)((n + 7) / 8);
    std::memset(dst, 0, (size_t)packed_size);

    for (int64_t i = 0; i < n; i++) {
        if (src[i] > 0) {
            dst[i / 8] |= (1 << (i % 8));
        }
    }
}

int STEQuantizer::bits_for_format(Format f) const {
    switch (f) {
        case Format::Q1: return 1;
        case Format::Q2: return 2;
        case Format::Q3: return 3;
        case Format::Q4: return 4;
        case Format::Q6: return 6;
        case Format::Q8: return 8;
        case Format::Q12: return 12;
        case Format::Q16: return 16;
        case Format::Q24: return 24;
        case Format::Q32: return 32;
        // K variants -> base BPW
        case Format::Q1_K_L: case Format::Q1_K_M: case Format::Q1_K_H: return 1;
        case Format::Q2_K_L: case Format::Q2_K_M: case Format::Q2_K_H: return 2;
        case Format::Q3_K_L: case Format::Q3_K_M: case Format::Q3_K_H: return 3;
        case Format::Q4_K_L: case Format::Q4_K_M: case Format::Q4_K_H: return 4;
        case Format::Q6_K_L: case Format::Q6_K_M: case Format::Q6_K_H: return 6;
        case Format::Q8_K_L: case Format::Q8_K_M: case Format::Q8_K_H: return 8;
        case Format::Q12_K_L: case Format::Q12_K_M: case Format::Q12_K_H: return 12;
        case Format::Q16_K_L: case Format::Q16_K_M: case Format::Q16_K_H: return 16;
        case Format::Q24_K_L: case Format::Q24_K_M: case Format::Q24_K_H: return 24;
        // GRP exact -> base BPW
        case Format::QG1: return 1;
        case Format::QG2: return 2;
        case Format::QG3: return 3;
        case Format::QG4: return 4;
        case Format::QG6: return 6;
        case Format::QG8: return 8;
        case Format::QG12: return 12;
        case Format::QG16: return 16;
        case Format::QG24: return 24;
        // K_G variants -> base BPW
        case Format::QG_1_K_L: case Format::QG_1_K_M: case Format::QG_1_K_H: return 1;
        case Format::QG_2_K_L: case Format::QG_2_K_M: case Format::QG_2_K_H: return 2;
        case Format::QG_3_K_L: case Format::QG_3_K_M: case Format::QG_3_K_H: return 3;
        case Format::QG_4_K_L: case Format::QG_4_K_M: case Format::QG_4_K_H: return 4;
        case Format::QG_6_K_L: case Format::QG_6_K_M: case Format::QG_6_K_H: return 6;
        case Format::QG_8_K_L: case Format::QG_8_K_M: case Format::QG_8_K_H: return 8;
        case Format::QG_12_K_L: case Format::QG_12_K_M: case Format::QG_12_K_H: return 12;
        case Format::QG_16_K_L: case Format::QG_16_K_M: case Format::QG_16_K_H: return 16;
        case Format::QG_24_K_L: case Format::QG_24_K_M: case Format::QG_24_K_H: return 24;
        // half-BPW plain -> base BPW
        case Format::Q1_5: return 1;
        case Format::Q2_5: return 2;
        case Format::Q3_5: return 3;
        case Format::Q4_5: return 4;
        case Format::Q6_5: return 6;
        case Format::Q8_5: return 8;
        case Format::Q12_5: return 12;
        case Format::Q16_5: return 16;
        case Format::Q24_5: return 24;
        // half-BPW GRP -> base BPW
        case Format::QG_1_5: return 1;
        case Format::QG_2_5: return 2;
        case Format::QG_3_5: return 3;
        case Format::QG_4_5: return 4;
        case Format::QG_6_5: return 6;
        case Format::QG_8_5: return 8;
        case Format::QG_12_5: return 12;
        case Format::QG_16_5: return 16;
        case Format::QG_24_5: return 24;
        // MX plain / grouped -> base BPW
        case Format::Q_MX_3_5: case Format::QG_MX_3_5: return 3;
        case Format::Q_MX_4_5: case Format::QG_MX_4_5: return 4;
        case Format::Q_MX_6_5: case Format::QG_MX_6_5: return 6;
        case Format::Q_MX_8_5: case Format::QG_MX_8_5: return 8;
        case Format::Q_MX_12_5: case Format::QG_MX_12_5: return 12;
        case Format::Q_MX_16_5: case Format::QG_MX_16_5: return 16;
        case Format::Q_MX_24_5: case Format::QG_MX_24_5: return 24;
        default: return 8;
    }
}

Tensor STEQuantizer::fake_quantize_qat(const Tensor& fp32_weight, float scale_override) {
    int bits = bits_for_format(target_format_);
    int qmin, qmax;
    qat::get_qrange_bits(bits, true, false, qmin, qmax);
    float scale = scale_override;
    if (scale <= 1e-8f) {
        int64_t n = fp32_weight.numel();
        const float* d = fp32_weight.data<float>();
        float mx = 0;
        for (int64_t i = 0; i < n; ++i) mx = std::max(mx, std::fabs(d[i]));
        if (mx < 1e-8f) mx = 1.0f;
        scale = mx / (float)qmax;
    }
    // STE node: forward = round/clamp, backward = identity
    return qat::fake_quantize(fp32_weight, scale, qmin, qmax);
}

Tensor STEQuantizer::lsq_quantize(const Tensor& fp32_weight, Tensor& scale_param) {
    int bits = bits_for_format(target_format_);
    int qmin, qmax;
    qat::get_qrange_bits(bits, true, false, qmin, qmax);
    return qat::lsq_fake_quantize(fp32_weight, scale_param, qmin, qmax);
}

Tensor STEQuantizer::fake_quantize_with_observer(const Tensor& fp32_weight, qat::Observer& obs, int bits) {
    int b = bits > 0 ? bits : bits_for_format(target_format_);
    return qat::fake_quantize_with_observer(fp32_weight, obs, b, true);
}

} // namespace quant
