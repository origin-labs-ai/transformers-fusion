#include "quant/autograd_functions.h"
#include "quant/autograd.h"
#include <cmath>
#include <cstring>
#include <cfloat>

#if defined(QUANT_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace quant {

// ========================================================================
// ScaledDotProductAttentionFunction
// ========================================================================

std::vector<Tensor> ScaledDotProductAttentionFunction::forward(const std::vector<Tensor>& inputs) {
    const Tensor& Q = inputs[0];
    const Tensor& K = inputs[1];
    const Tensor& V = inputs[2];

    B_ = Q.dim(0); H_ = Q.dim(1); S_ = Q.dim(2); D_ = Q.dim(3);
    KV_H_ = K.dim(1); S_full_ = K.dim(2);

    float scale = 1.0f / std::sqrt((float)D_);

    Tensor score({B_, H_, S_, S_full_}, DType::F32);
    Tensor attn_weights({B_, H_, S_, S_full_}, DType::F32);
    Tensor attn_out({B_, H_, S_, D_}, DType::F32);

    const float* qd = Q.data<float>();
    const float* kd = K.data<float>();
    const float* vd = V.data<float>();
    float* sd = score.data<float>();
    float* wd = attn_weights.data<float>();
    float* od = attn_out.data<float>();

    const bool causal = (S_full_ == S_);
    for (int64_t b = 0; b < B_; b++) {
        for (int64_t h = 0; h < H_; h++) {
            int64_t kh = h % KV_H_;
            int64_t h_base = (b * H_ + h) * S_ * S_full_;
            int64_t kh_off = (b * KV_H_ + kh) * S_full_ * D_;
            for (int64_t s = 0; s < S_; s++) {
                int64_t q_off = ((b * H_ + h) * S_ + s) * D_;
                int64_t score_row = h_base + s * S_full_;
                for (int64_t t = 0; t < S_full_; t++) {
                    if (causal && t > s) {
                        sd[score_row + t] = -FLT_MAX / 4.0f;
                        continue;
                    }
                    int64_t k_off = kh_off + t * D_;
                    float sum = 0;
                    for (int64_t d = 0; d < D_; d++)
                        sum += qd[q_off + d] * kd[k_off + d];
                    sd[score_row + t] = sum * scale;
                }
            }
        }
    }

    for (int64_t b = 0; b < B_; b++) {
        for (int64_t h = 0; h < H_; h++) {
            int64_t h_base = (b * H_ + h) * S_ * S_full_;
            for (int64_t s = 0; s < S_; s++) {
                int64_t row = h_base + s * S_full_;
                softmax_forward_avx2(wd + row, sd + row, S_full_);
                if (causal) {
                    for (int64_t t = s + 1; t < S_full_; t++)
                        wd[row + t] = 0.0f;
                    float z = 0.0f;
                    for (int64_t t = 0; t <= s && t < S_full_; t++) z += wd[row + t];
                    if (z > 0.0f) {
                        float inv = 1.0f / z;
                        for (int64_t t = 0; t <= s && t < S_full_; t++)
                            wd[row + t] *= inv;
                    }
                }
            }
        }
    }

    for (int64_t b = 0; b < B_; b++) {
        for (int64_t h = 0; h < H_; h++) {
            int64_t kh = h % KV_H_;
            int64_t h_base = (b * H_ + h) * S_ * S_full_;
            int64_t kh_off = (b * KV_H_ + kh) * S_full_ * D_;
            for (int64_t s = 0; s < S_; s++) {
                int64_t row = h_base + s * S_full_;
                int64_t out_off = ((b * H_ + h) * S_ + s) * D_;
                for (int64_t d = 0; d < D_; d++) {
                    float sum = 0;
                    for (int64_t t = 0; t < S_full_; t++)
                        sum += wd[row + t] * vd[kh_off + t * D_ + d];
                    od[out_off + d] = sum;
                }
            }
        }
    }

    saved = inputs;
    saved.push_back(attn_weights);

    return {attn_out};
}

std::vector<Tensor> ScaledDotProductAttentionFunction::backward(const std::vector<Tensor>& grad_output) {
    const Tensor& d_out = grad_output[0];
    const Tensor& Q = saved[0];
    const Tensor& K = saved[1];
    const Tensor& V = saved[2];
    const Tensor& attn_weights = saved[3];

    const float* qd = Q.data<float>();
    const float* kd = K.data<float>();
    const float* vd = V.data<float>();
    const float* wd = attn_weights.data<float>();
    const float* dd = d_out.data<float>();

    Tensor dQ({B_, H_, S_, D_}, DType::F32);
    Tensor dK({B_, KV_H_, S_full_, D_}, DType::F32);
    Tensor dV({B_, KV_H_, S_full_, D_}, DType::F32);
    dQ.zero_(); dK.zero_(); dV.zero_();

    float scale = 1.0f / std::sqrt((float)D_);
    int64_t H = H_, KV_H = KV_H_, S = S_, S_full = S_full_, D = D_;
    int64_t B = B_;

    Tensor d_aw({B, H, S, S_full}, DType::F32);
    float* daw = d_aw.data<float>();

    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            int64_t kh = h % KV_H;
            int64_t h_base = (b * H + h) * S * S_full;
            int64_t kh_off = (b * KV_H + kh) * S_full * D;
            for (int64_t s = 0; s < S; s++) {
                int64_t row = h_base + s * S_full;
                int64_t d_out_off = ((b * H + h) * S + s) * D;
                for (int64_t t = 0; t < S_full; t++) {
                    float sum = 0;
                    for (int64_t d = 0; d < D; d++)
                        sum += dd[d_out_off + d] * vd[kh_off + t * D + d];
                    daw[row + t] = sum;
                }
            }
        }
    }

    Tensor d_score({B_, H_, S_, S_full_}, DType::F32);
    float* ds = d_score.data<float>();
    for (int64_t b = 0; b < B_; b++) {
        for (int64_t h = 0; h < H_; h++) {
            int64_t h_base = (b * H_ + h) * S_ * S_full_;
            for (int64_t s = 0; s < S_; s++) {
                int64_t row = h_base + s * S_full_;
                float dot = 0;
                for (int64_t t = 0; t < S_full_; t++)
                    dot += wd[row + t] * daw[row + t];
                for (int64_t t = 0; t < S_full_; t++)
                    ds[row + t] = wd[row + t] * (daw[row + t] - dot);
            }
        }
    }

    for (int64_t b = 0; b < B_; b++) {
        for (int64_t h = 0; h < H_; h++) {
            int64_t kh = h % KV_H_;
            int64_t h_base = (b * H_ + h) * S_ * S_full_;
            int64_t kh_off = (b * KV_H_ + kh) * S_full_ * D_;
            for (int64_t s = 0; s < S_; s++) {
                int64_t row = h_base + s * S_full_;
                int64_t dq_off = ((b * H_ + h) * S_ + s) * D_;
                for (int64_t d = 0; d < D_; d++) {
                    float sum = 0;
                    for (int64_t t = 0; t < S_full_; t++)
                        sum += ds[row + t] * kd[kh_off + t * D_ + d];
                    dQ.data<float>()[dq_off + d] = sum * scale;
                }
            }
        }
    }

    for (int64_t b = 0; b < B_; b++) {
        for (int64_t h = 0; h < H_; h++) {
            int64_t kh = h % KV_H_;
            int64_t h_base = (b * H_ + h) * S_ * S_full_;
            int64_t kh_off = (b * KV_H_ + kh) * S_full_ * D_;
            for (int64_t t = 0; t < S_full_; t++) {
                for (int64_t d = 0; d < D_; d++) {
                    float sum = 0;
                    for (int64_t s = 0; s < S_; s++) {
                        int64_t row = h_base + s * S_full_;
                        sum += ds[row + t] * qd[((b * H_ + h) * S_ + s) * D_ + d];
                    }
                    dK.data<float>()[kh_off + t * D_ + d] += sum * scale;
                }
            }
        }
    }

    for (int64_t b = 0; b < B_; b++) {
        for (int64_t h = 0; h < H_; h++) {
            int64_t kh = h % KV_H_;
            int64_t h_base = (b * H_ + h) * S_ * S_full_;
            int64_t kh_off = (b * KV_H_ + kh) * S_full_ * D_;
            for (int64_t t = 0; t < S_full_; t++) {
                for (int64_t d = 0; d < D_; d++) {
                    float sum = 0;
                    for (int64_t s = 0; s < S_; s++) {
                        int64_t row = h_base + s * S_full_;
                        sum += wd[row + t] * dd[((b * H_ + h) * S_ + s) * D_ + d];
                    }
                    dV.data<float>()[kh_off + t * D_ + d] += sum;
                }
            }
        }
    }

    return {dQ, dK, dV};
}

// ========================================================================
// MatMulFunction
// ========================================================================

std::vector<Tensor> MatMulFunction::forward(const std::vector<Tensor>& inputs) {
    saved = inputs;
    const Tensor& a = inputs[0];
    const Tensor& b = inputs[1];
    int64_t M = a.dim(0);
    int64_t K = a.dim(1);
    int64_t N = b.dim(0);  // weights are stored row-major {N, K} = {out, in}
    Tensor out({M, N}, DType::F32);
    kernel::scalar_gemm_bt(
        a.data<float>(), b.data<float>(),
        out.data<float>(), (int)M, (int)N, (int)K
    );
    return {out};
}

std::vector<Tensor> MatMulFunction::backward(const std::vector<Tensor>& grad_output) {
    const Tensor& g_orig = grad_output[0];
    const Tensor& a = saved[0];
    const Tensor& b = saved[1];
    int64_t M = a.dim(0);
    int64_t K = a.dim(1);
    int64_t N = b.dim(0);
    Tensor g = g_orig;
    if (g_orig.rank() > 2) {
        g = g_orig.reshape(Shape{M, N});
    }
    // dA[m,k] = sum_n g[m,n] * B[n,k]   (B is {N, K})
    Tensor ga({M, K});
    {
        const float* gd = g.data<float>();
        const float* bd = b.data<float>();
        float* ad = ga.data<float>();
        for (int64_t m = 0; m < M; m++)
            for (int64_t k = 0; k < K; k++) {
                float s = 0.0f;
                for (int64_t n = 0; n < N; n++)
                    s += gd[m * N + n] * bd[n * K + k];
                ad[m * K + k] = s;
            }
    }
    // dB[n,k] = sum_m g[m,n] * a[m,k]   (B stays {N, K})
    Tensor gb({N, K});
    {
        const float* gd = g.data<float>();
        const float* ad = a.data<float>();
        float* bd = gb.data<float>();
        for (int64_t n = 0; n < N; n++)
            for (int64_t k = 0; k < K; k++) {
                float s = 0.0f;
                for (int64_t m = 0; m < M; m++)
                    s += gd[m * N + n] * ad[m * K + k];
                bd[n * K + k] = s;
            }
    }
    return {ga, gb};
}

// ========================================================================
// AddFunction
// ========================================================================

std::vector<Tensor> AddFunction::forward(const std::vector<Tensor>& inputs) {
    saved = inputs;
    Tensor out(inputs[0].shape(), DType::F32);
    math::add(inputs[0], inputs[1], out);
    return {out};
}

std::vector<Tensor> AddFunction::backward(const std::vector<Tensor>& grad_output) {
    return {grad_output[0], grad_output[0]};
}

// ========================================================================
// SiLUFunction
// ========================================================================

std::vector<Tensor> SiLUFunction::forward(const std::vector<Tensor>& inputs) {
    saved = inputs;
    Tensor out(inputs[0].shape(), DType::F32);
    math::silu(inputs[0], out);
    return {out};
}

std::vector<Tensor> SiLUFunction::backward(const std::vector<Tensor>& grad_output) {
    Tensor g = silu_grad(saved[0], grad_output[0]);
    return {g};
}

// ========================================================================
// MulFunction
// ========================================================================

std::vector<Tensor> MulFunction::forward(const std::vector<Tensor>& inputs) {
    saved = inputs;
    Tensor out(inputs[0].shape(), DType::F32);
    math::mul(inputs[0], inputs[1], out);
    return {out};
}

std::vector<Tensor> MulFunction::backward(const std::vector<Tensor>& grad_output) {
    const Tensor& g = grad_output[0];
    const Tensor& a = saved[0];
    const Tensor& b = saved[1];
    Tensor ga(a.shape(), DType::F32);
    Tensor gb(b.shape(), DType::F32);
    math::mul(g, b, ga);
    math::mul(g, a, gb);
    return {ga, gb};
}

// ========================================================================
// RMSNormFunction
// ========================================================================

std::vector<Tensor> RMSNormFunction::forward(const std::vector<Tensor>& inputs) {
    saved = inputs;
    Tensor out(inputs[0].shape(), DType::F32);
    float eps = saved_eps;
    math::rms_norm(inputs[0], inputs[1], eps, out);
    return {out};
}

std::vector<Tensor> RMSNormFunction::backward(const std::vector<Tensor>& grad_output) {
    const Tensor& x = saved[0];
    const Tensor& gamma = saved[1];
    int64_t N = x.numel() / x.dim(x.rank() - 1);
    Tensor dgamma({gamma.numel()}, DType::F32);
    dgamma.zero_();
    Tensor dx = rms_norm_grad(x, gamma, grad_output[0], (int)N, &dgamma);
    return {dx, dgamma};
}

// ========================================================================
// CrossEntropyFunction
// ========================================================================

std::vector<Tensor> CrossEntropyFunction::forward(const std::vector<Tensor>& inputs) {
    saved = inputs;
    Tensor loss = cross_entropy_loss(inputs[0], inputs[1]);
    return {loss};
}

std::vector<Tensor> CrossEntropyFunction::backward(const std::vector<Tensor>& grad_output) {
    Tensor grad = cross_entropy_grad(saved[0], saved[1]);
    if (grad_output[0].numel() == 1) {
        float scale = grad_output[0].data<float>()[0];
        math::scale(scale, grad, grad);
    }
    return {grad};
}

// ========================================================================
// RotaryFunction
// ========================================================================

RotaryFunction::RotaryFunction(int64_t hd, const Tensor& cos, const Tensor& sin,
                               int64_t ss, int64_t sl)
    : head_dim_(hd), cos_cached_(cos), sin_cached_(sin),
      seq_start_(ss), seq_len_(sl), B_(0), H_(0), S_(0), D_(0) {}

std::vector<Tensor> RotaryFunction::forward(const std::vector<Tensor>& inputs) {
    const Tensor& x = inputs[0];
    B_ = x.dim(0); H_ = x.dim(1); S_ = x.dim(2); D_ = x.dim(3);
    Tensor out(x.shape(), DType::F32);
    const float* xd = x.data<float>();
    float* od = out.data<float>();
    const float* cos_d = cos_cached_.data<float>();
    const float* sin_d = sin_cached_.data<float>();
    int64_t half = D_ / 2;
    for (int64_t b = 0; b < B_; b++)
        for (int64_t h = 0; h < H_; h++)
            for (int64_t s = 0; s < S_; s++) {
                int64_t base = ((b * H_ + h) * S_ + s) * D_;
                int64_t pos = seq_start_ + s;
                for (int64_t d = 0; d < half; d++) {
                    float x1 = xd[base + d];
                    float x2 = xd[base + d + half];
                    float cos_v = cos_d[pos * half + d];
                    float sin_v = sin_d[pos * half + d];
                    od[base + d] = x1 * cos_v - x2 * sin_v;
                    od[base + d + half] = x1 * sin_v + x2 * cos_v;
                }
            }
    return {out};
}

std::vector<Tensor> RotaryFunction::backward(const std::vector<Tensor>& grad_output) {
    const Tensor& grad = grad_output[0];
    Tensor dx(grad.shape(), DType::F32);
    const float* gd = grad.data<float>();
    float* dxd = dx.data<float>();
    const float* cos_d = cos_cached_.data<float>();
    const float* sin_d = sin_cached_.data<float>();
    int64_t half = D_ / 2;
    for (int64_t b = 0; b < B_; b++)
        for (int64_t h = 0; h < H_; h++)
            for (int64_t s = 0; s < S_; s++) {
                int64_t base = ((b * H_ + h) * S_ + s) * D_;
                int64_t pos = seq_start_ + s;
                for (int64_t d = 0; d < half; d++) {
                    float g1 = gd[base + d];
                    float g2 = gd[base + d + half];
                    float cos_v = cos_d[pos * half + d];
                    float sin_v = sin_d[pos * half + d];
                    dxd[base + d] = g1 * cos_v + g2 * sin_v;
                    dxd[base + d + half] = -g1 * sin_v + g2 * cos_v;
                }
            }
    return {dx};
}

// ========================================================================
// BiasAddFunction
// ========================================================================

std::vector<Tensor> BiasAddFunction::forward(const std::vector<Tensor>& inputs) {
    saved = inputs;
    const Tensor& x = inputs[0];
    const Tensor& bias = inputs[1];
    int64_t M = x.dim(0);
    int64_t N = x.dim(1);
    Tensor out(x.shape(), DType::F32);
    const float* xd = x.data<float>();
    const float* bd = bias.data<float>();
    float* od = out.data<float>();
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < N; j++)
            od[i * N + j] = xd[i * N + j] + bd[j];
    return {out};
}

std::vector<Tensor> BiasAddFunction::backward(const std::vector<Tensor>& grad_output) {
    const Tensor& d_out = grad_output[0];
    const Tensor& x = saved[0];
    int64_t N = x.dim(x.rank() - 1);
    int64_t numel = d_out.numel();
    const float* gd = d_out.data<float>();
    Tensor d_x(x.shape(), DType::F32);
    std::memcpy(d_x.data<float>(), gd, numel * sizeof(float));
    Tensor d_bias({N}, DType::F32);
    float* dbd = d_bias.data<float>();
    for (int64_t p = 0; p < numel; p++)
        dbd[p % N] += gd[p];
    return {d_x, d_bias};
}

// ========================================================================
// EmbeddingFunction
// ========================================================================

std::vector<Tensor> EmbeddingFunction::forward(const std::vector<Tensor>& inputs) {
    const Tensor& ids = inputs[0];
    const Tensor& weight = inputs[1];
    int64_t N = ids.numel();
    int64_t D = weight.dim(1);
    int64_t V = weight.dim(0);
    Tensor out({N, D}, DType::F32);
    const float* id_d = ids.data<float>();
    const float* wd = weight.data<float>();
    float* od = out.data<float>();
    saved = {ids.clone(), weight};
    for (int64_t i = 0; i < N; i++) {
        int64_t token = (int64_t)id_d[i];
        if (token < 0 || token >= V) token = 0;
        std::memcpy(od + i * D, wd + token * D, D * sizeof(float));
    }
    return {out};
}

std::vector<Tensor> EmbeddingFunction::backward(const std::vector<Tensor>& grad_output) {
    const Tensor& ids = saved[0];
    const Tensor& weight = saved[1];
    const Tensor& d_out = grad_output[0];
    int64_t N = ids.numel();
    int64_t D = weight.dim(1);
    int64_t V = weight.dim(0);
    const float* id_d = ids.data<float>();
    const float* gd = d_out.data<float>();
    Tensor d_weight(weight.shape(), DType::F32);
    d_weight.zero_();
    float* dwd = d_weight.data<float>();
    for (int64_t i = 0; i < N; i++) {
        int64_t token = (int64_t)id_d[i];
        if (token < 0 || token >= V) token = 0;
        for (int64_t d = 0; d < D; d++)
            dwd[token * D + d] += gd[i * D + d];
    }
    return {Tensor(), d_weight};
}

// ========================================================================
// FlattenAttentionFunction
// ========================================================================

FlattenAttentionFunction::FlattenAttentionFunction(int64_t B, int64_t H, int64_t S, int64_t D)
    : B_(B), H_(H), S_(S), D_(D) {}

std::vector<Tensor> FlattenAttentionFunction::forward(const std::vector<Tensor>& inputs) {
    const Tensor& x = inputs[0];
    Tensor out(Shape{B_ * S_, H_ * D_}, DType::F32);
    const float* xd = x.data<float>();
    float* od = out.data<float>();
    for (int64_t b = 0; b < B_; b++)
        for (int64_t h = 0; h < H_; h++)
            for (int64_t s = 0; s < S_; s++)
                for (int64_t d = 0; d < D_; d++)
                    od[(b * S_ + s) * (H_ * D_) + h * D_ + d] =
                        xd[((b * H_ + h) * S_ + s) * D_ + d];
    return {out};
}

std::vector<Tensor> FlattenAttentionFunction::backward(const std::vector<Tensor>& grad_output) {
    const Tensor& d_out = grad_output[0];
    Tensor d_x(Shape{B_, H_, S_, D_}, DType::F32);
    d_x.zero_();
    const float* dd = d_out.data<float>();
    float* dxd = d_x.data<float>();
    for (int64_t b = 0; b < B_; b++)
        for (int64_t h = 0; h < H_; h++)
            for (int64_t s = 0; s < S_; s++)
                for (int64_t d = 0; d < D_; d++)
                    dxd[((b * H_ + h) * S_ + s) * D_ + d] =
                        dd[(b * S_ + s) * (H_ * D_) + h * D_ + d];
    return {d_x};
}

// ========================================================================
// TransposeFunction
// ========================================================================

TransposeFunction::TransposeFunction(int dim1, int dim2) : dim1_(dim1), dim2_(dim2) {}

std::vector<Tensor> TransposeFunction::forward(const std::vector<Tensor>& inputs) {
    saved = inputs;
    return {inputs[0].transpose(dim1_, dim2_)};
}

std::vector<Tensor> TransposeFunction::backward(const std::vector<Tensor>& grad_output) {
    return {grad_output[0].transpose(dim1_, dim2_)};
}

} // namespace quant
