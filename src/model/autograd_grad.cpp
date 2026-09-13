#include "quant/autograd.h"
#include <cmath>
#include <cstring>
#include <cfloat>

namespace quant {

// ========================================================================
// REAL matmul_grad: dA = grad_output * B^T (using scalar_gemm convention)
// ========================================================================

Tensor matmul_grad(const Tensor& a, const Tensor& b, const Tensor& grad_output) {
    return matmul_grad_wrt_a(grad_output, b);
}

// dA = grad_output * B^T where B is the {K, N} operand of scalar_gemm
// grad_output: {M, N},  b: {K, N}
// dA: {M, K}
// Access pattern matches scalar_gemm: bd[k * N + j] where k in [0,K), j in [0,N)
Tensor matmul_grad_wrt_a(const Tensor& grad_output, const Tensor& b) {
    int64_t M = grad_output.dim(0);
    int64_t N = grad_output.dim(1);
    int64_t K = b.dim(0);
    Tensor dA({M, K});
    const float* gd = grad_output.data<float>();
    const float* bd = b.data<float>();
    float* dad = dA.data<float>();
    for (int64_t i = 0; i < M; ++i)
        for (int64_t k = 0; k < K; ++k) {
            float s = 0.0f;
            for (int64_t j = 0; j < N; ++j)
                s += gd[i * N + j] * bd[k * N + j];
            dad[i * K + k] = s;
        }
    return dA;
}

// dB = A^T * grad_output  where B is stored as {N, K}
// grad_output: {M, N},  a: {M, K}
// Returns: gradient in {K, N} format (matching scalar_gemm access pattern)
Tensor matmul_grad_wrt_b(const Tensor& grad_output, const Tensor& a) {
    int64_t K = a.dim(1);
    int64_t M = a.dim(0);
    int64_t N = grad_output.dim(1);
    Tensor dB({K, N});
    const float* gd = grad_output.data<float>();
    const float* ad = a.data<float>();
    float* dbd = dB.data<float>();
    for (int64_t k = 0; k < K; ++k)
        for (int64_t j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int64_t i = 0; i < M; ++i)
                s += ad[i * K + k] * gd[i * N + j];
            dbd[k * N + j] = s;
        }
    return dB;
}

// Gradient w.r.t. the weight operand of scalar_gemm, stored as {K, N}
// (the same layout matmul_op/Linear feed to the forward pass).
Tensor weight_grad(const Tensor& grad_output, const Tensor& a, const Tensor& weight) {
    // weight: {K, N}, a: {M, K}, grad_output: {M, N}
    Tensor dB = matmul_grad_wrt_b(grad_output, a);  // {K, N} — already matches weight
    (void)weight;
    return dB;
}

// ========================================================================
// Activation gradients
// ========================================================================

Tensor relu_grad(const Tensor& x, const Tensor& grad) {
    Tensor out(x.shape(), DType::F32);
    const float* xd = x.data<float>();
    const float* gd = grad.data<float>();
    float* od = out.data<float>();
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++)
        od[i] = (xd[i] > 0) ? gd[i] : 0.0f;
    return out;
}

Tensor silu_grad(const Tensor& x, const Tensor& grad) {
    Tensor out(x.shape(), DType::F32);
    const float* xd = x.data<float>();
    const float* gd = grad.data<float>();
    float* od = out.data<float>();
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) {
        float sig = 1.0f / (1.0f + std::exp(-xd[i]));
        od[i] = gd[i] * sig * (1.0f + xd[i] * (1.0f - sig));
    }
    return out;
}

Tensor gelu_grad(const Tensor& x, const Tensor& grad) {
    Tensor out(x.shape(), DType::F32);
    const float* xd = x.data<float>();
    const float* gd = grad.data<float>();
    float* od = out.data<float>();
    int64_t n = x.numel();
    float sqrt2_inv = 0.7071067811865475f;
    float sqrt_2pi_inv = 0.3989422804014327f;
    for (int64_t i = 0; i < n; i++) {
        float v = xd[i];
        float cdf = 0.5f * (1.0f + std::erf(v * sqrt2_inv));
        float pdf = std::exp(-0.5f * v * v) * sqrt_2pi_inv;
        od[i] = gd[i] * (cdf + v * pdf);
    }
    return out;
}

Tensor softmax_grad(const Tensor& output, const Tensor& grad) {
    Tensor out(output.shape(), DType::F32);
    const float* sd = output.data<float>();
    const float* gd = grad.data<float>();
    float* od = out.data<float>();
    int64_t rows = output.rank() >= 2 ? output.dim(0) : 1;
    int64_t cols = output.rank() >= 2 ? output.dim(output.rank() - 1) : output.numel();
    for (int64_t r = 0; r < rows; r++) {
        float dot = 0;
        for (int64_t c = 0; c < cols; c++)
            dot += sd[r * cols + c] * gd[r * cols + c];
        for (int64_t c = 0; c < cols; c++) {
            int64_t idx = r * cols + c;
            od[idx] = sd[idx] * (gd[idx] - dot);
        }
    }
    return out;
}

// ========================================================================
// Norm gradients
// ========================================================================

Tensor layer_norm_grad(const Tensor& x, const Tensor& gamma, const Tensor& grad, int N, Tensor* dgamma_out) {
    int64_t D = x.numel() / N;
    Tensor dx(x.shape());
    Tensor dgamma_local({D});
    dx.zero_(); dgamma_local.zero_();

    const float* xd = x.data<float>();
    const float* gd = grad.data<float>();
    const float* gam = gamma.data<float>();
    float* dxd = dx.data<float>();
    float* dgd = dgamma_out ? dgamma_out->data<float>() : dgamma_local.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        double mu = 0, var = 0;
        for (int64_t d = 0; d < D; ++d) mu += xd[n * D + d];
        mu /= D;
        for (int64_t d = 0; d < D; ++d) {
            double diff = xd[n * D + d] - mu;
            var += diff * diff;
        }
        var /= D;
        double inv_std = 1.0 / std::sqrt(var + 1e-5);

        double dnorm = 0, dvar_raw = 0;
        for (int64_t d = 0; d < D; ++d) {
            double diff = xd[n * D + d] - mu;
            double dy = gd[n * D + d];
            dnorm += dy * gam[d];
            dvar_raw += dy * gam[d] * diff;
        }
        double inv_std3 = inv_std * inv_std * inv_std;
        double dmu = -dnorm * inv_std / D;
        for (int64_t d = 0; d < D; ++d) {
            double diff = xd[n * D + d] - mu;
            dxd[n * D + d] = (float)(gd[n * D + d] * inv_std * gam[d] - dnorm * inv_std / D - dvar_raw * inv_std3 * diff / D);
            dgd[d] += (float)(gd[n * D + d] * inv_std * diff);
        }
    }
    return dx;
}

Tensor rms_norm_grad(const Tensor& x, const Tensor& gamma, const Tensor& grad, int N, Tensor* dgamma) {
    int64_t D = x.numel() / N;
    Tensor dx(x.shape());
    dx.zero_();

    const float* xd = x.data<float>();
    const float* gd = grad.data<float>();
    const float* gam = gamma.data<float>();
    float* dxd = dx.data<float>();
    float* dgd = dgamma ? dgamma->data<float>() : nullptr;

    for (int64_t n = 0; n < N; ++n) {
        double ss = 0;
        for (int64_t d = 0; d < D; ++d) ss += (double)xd[n * D + d] * xd[n * D + d];
        ss /= D;
        double inv = 1.0 / std::sqrt(ss + 1e-5);
        double inv3 = inv * inv * inv;
        double sum_dy_gamma_x = 0;
        for (int64_t d = 0; d < D; ++d)
            sum_dy_gamma_x += (double)gd[n * D + d] * gam[d] * xd[n * D + d];
        sum_dy_gamma_x *= inv3 / D;
        for (int64_t d = 0; d < D; ++d) {
            dxd[n * D + d] = (float)((double)gd[n * D + d] * gam[d] * inv -
                                     sum_dy_gamma_x * (double)xd[n * D + d]);
        }
        if (dgd) {
            for (int64_t d = 0; d < D; ++d)
                dgd[d] += (float)((double)gd[n * D + d] * xd[n * D + d] * inv);
        }
    }
    return dx;
}

// ========================================================================
// Loss functions
// ========================================================================

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
    QUANT_CHECK(logits.numel() % targets.numel() == 0, "CE: shape mismatch");
    int64_t batch = targets.numel();
    int64_t C = logits.numel() / batch;
    const float* ld = logits.data<float>();
    const float* td = targets.data<float>();
    Tensor loss(Shape{1});
    float* ld_out = loss.data<float>();
    *ld_out = 0;
    for (int64_t b = 0; b < batch; b++) {
        float max_val = -INFINITY;
        for (int64_t c = 0; c < C; c++)
            if (ld[b * C + c] > max_val) max_val = ld[b * C + c];
        float sum_exp = 0;
        for (int64_t c = 0; c < C; c++)
            sum_exp += std::exp(ld[b * C + c] - max_val);
        float log_sum_exp = max_val + std::log(sum_exp);
        int64_t target = (int64_t)td[b];
        if (target < 0) target = 0;
        if (target >= C) target = C - 1;
        *ld_out += log_sum_exp - ld[b * C + target];
    }
    *ld_out /= (float)batch;
    return loss;
}

Tensor cross_entropy_grad(const Tensor& logits, const Tensor& targets) {
    int64_t batch = targets.numel();
    int64_t C = logits.numel() / batch;
    Tensor grad(logits.shape());
    const float* ld = logits.data<float>();
    const float* td = targets.data<float>();
    float* gd = grad.data<float>();
    for (int64_t b = 0; b < batch; b++) {
        float max_val = -INFINITY;
        for (int64_t c = 0; c < C; c++)
            if (ld[b * C + c] > max_val) max_val = ld[b * C + c];
        float sum_exp = 0;
        for (int64_t c = 0; c < C; c++)
            sum_exp += std::exp(ld[b * C + c] - max_val);
        int64_t target = (int64_t)td[b];
        if (target < 0) target = 0;
        if (target >= C) target = C - 1;
        for (int64_t c = 0; c < C; c++) {
            float soft = std::exp(ld[b * C + c] - max_val) / sum_exp;
            gd[b * C + c] = (soft - (c == target ? 1.0f : 0.0f)) / (float)batch;
        }
    }
    return grad;
}

} // namespace quant
