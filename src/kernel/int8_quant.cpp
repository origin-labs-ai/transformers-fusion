#include "quant/int8_quant.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace quant {

Int8QuantParams quantize_per_tensor(const float* src, int8_t* dst, int64_t n) {
    Int8QuantParams params;
    float max_abs = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float abs_val = std::fabs(src[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    if (max_abs < 1e-10f) max_abs = 1e-10f;
    params.scale = 127.0f / max_abs;
    params.inv_scale = max_abs / 127.0f;
    for (int64_t i = 0; i < n; i++) {
        dst[i] = (int8_t)std::round(src[i] * params.scale);
    }
    return params;
}

std::vector<Int8QuantParams> quantize_per_token(const float* src, int8_t* dst,
                                                  int64_t rows, int64_t cols) {
    std::vector<Int8QuantParams> params(rows);
    for (int64_t r = 0; r < rows; r++) {
        params[r] = quantize_per_tensor(src + r * cols, dst + r * cols, cols);
    }
    return params;
}

void dequantize_per_tensor(const int8_t* src, float* dst, int64_t n, float inv_scale) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = (float)src[i] * inv_scale;
    }
}

Int8QuantParams quantize_tensor(const Tensor& src, Tensor& dst) {
    return quantize_per_tensor((const float*)src.data(), (int8_t*)dst.data(), src.numel());
}

} // namespace quant
