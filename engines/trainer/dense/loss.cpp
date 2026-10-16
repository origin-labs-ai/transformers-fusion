#include "loss.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace quant {
namespace dense_engine {

static void softmax_rows(const float* lg, float* pr, int64_t rows, int64_t V) {
    for (int64_t r = 0; r < rows; r++) {
        float m = lg[r * V];
        for (int64_t v = 1; v < V; v++) m = std::max(m, lg[r * V + v]);
        double s = 0.0;
        for (int64_t v = 0; v < V; v++) { pr[r * V + v] = std::exp(lg[r * V + v] - m); s += pr[r * V + v]; }
        for (int64_t v = 0; v < V; v++) pr[r * V + v] /= (float)s;
    }
}

static void check_2d(const Tensor& t, const char* what) {
    if (t.rank() != 2) throw std::runtime_error(std::string("[loss] ") + what + " needs 2-D");
}

LossOut ce_loss(const Tensor& logits, const Tensor& labels, Tensor& dlogits,
                int64_t ignore_index) {
    check_2d(logits, "logits");
    int64_t rows = logits.dim(0), V = logits.dim(1);
    if (labels.numel() != rows) throw std::runtime_error("[loss] labels rows mismatch");
    const float* lg = logits.data<float>();
    const int* lb = labels.data<int>();
    std::vector<float> pr((size_t)rows * (size_t)V);
    softmax_rows(lg, pr.data(), rows, V);
    dlogits = Tensor::zeros({rows, V});
    float* dl = dlogits.data<float>();
    double sum = 0.0;
    int64_t n = 0;
    int64_t correct = 0;
    for (int64_t r = 0; r < rows; r++) {
        int t = lb[r];
        if (t == ignore_index || t < 0 || t >= V) continue;
        sum += -std::log(std::max(pr[(size_t)r * V + t], 1e-12f));
        n++;
        int64_t arg = 0;
        for (int64_t v = 1; v < V; v++)
            if (pr[(size_t)r * V + v] > pr[(size_t)r * V + arg]) arg = v;
        if (arg == t) correct++;
        for (int64_t v = 0; v < V; v++) dl[r * V + v] = pr[(size_t)r * V + v];
        dl[r * V + t] -= 1.0f;
    }
    if (n > 0) for (int64_t i = 0; i < rows * V; i++) dl[i] /= (float)n;
    LossOut o;
    o.loss = n > 0 ? (float)(sum / (double)n) : 0.0f;
    o.acc = n > 0 ? (float)correct / (float)n : 0.0f;
    return o;
}

LossOut label_smooth_ce(const Tensor& logits, const Tensor& labels, Tensor& dlogits,
                        float smoothing, int64_t ignore_index) {
    check_2d(logits, "logits");
    int64_t rows = logits.dim(0), V = logits.dim(1);
    const float* lg = logits.data<float>();
    const int* lb = labels.data<int>();
    std::vector<float> pr((size_t)rows * (size_t)V);
    softmax_rows(lg, pr.data(), rows, V);
    dlogits = Tensor::zeros({rows, V});
    float* dl = dlogits.data<float>();
    double sum = 0.0;
    int64_t n = 0;
    for (int64_t r = 0; r < rows; r++) {
        int t = lb[r];
        if (t == ignore_index || t < 0 || t >= V) continue;
        float u = smoothing / (float)V;
        for (int64_t v = 0; v < V; v++) {
            float target = (v == t) ? (1.0f - smoothing + u) : u;
            sum += -target * std::log(std::max(pr[(size_t)r * V + v], 1e-12f));
            dl[r * V + v] = (pr[(size_t)r * V + v] - target);
        }
        n++;
    }
    if (n > 0) for (int64_t i = 0; i < rows * V; i++) dl[i] /= (float)n;
    LossOut o;
    o.loss = n > 0 ? (float)(sum / (double)n) : 0.0f;
    return o;
}

LossOut focal_loss(const Tensor& logits, const Tensor& labels, Tensor& dlogits,
                   float gamma, float alpha, int64_t ignore_index) {
    check_2d(logits, "logits");
    int64_t rows = logits.dim(0), V = logits.dim(1);
    const float* lg = logits.data<float>();
    const int* lb = labels.data<int>();
    std::vector<float> pr((size_t)rows * (size_t)V);
    softmax_rows(lg, pr.data(), rows, V);
    dlogits = Tensor::zeros({rows, V});
    float* dl = dlogits.data<float>();
    double sum = 0.0;
    int64_t n = 0;
    for (int64_t r = 0; r < rows; r++) {
        int t = lb[r];
        if (t == ignore_index || t < 0 || t >= V) continue;
        float pt = std::max(pr[(size_t)r * V + t], 1e-12f);
        float w = alpha * std::pow(1.0f - pt, gamma);
        sum += -w * std::log(pt);
        for (int64_t v = 0; v < V; v++) dl[r * V + v] = w * pr[(size_t)r * V + v];
        dl[r * V + t] -= w;
        dl[r * V + t] += w * pt * gamma * std::log(pt) * (1.0f - pt) / std::max(pt, 1e-6f) * 0.0f;
        n++;
    }
    if (n > 0) for (int64_t i = 0; i < rows * V; i++) dl[i] /= (float)n;
    LossOut o;
    o.loss = n > 0 ? (float)(sum / (double)n) : 0.0f;
    return o;
}

RDropOut rdrop_loss(const Tensor& logits1, const Tensor& logits2, const Tensor& labels,
                    float alpha, int64_t ignore_index) {
    Tensor d1, d2;
    LossOut c1 = ce_loss(logits1, labels, d1, ignore_index);
    LossOut c2 = ce_loss(logits2, labels, d2, ignore_index);
    check_2d(logits1, "logits1");
    int64_t rows = logits1.dim(0), V = logits1.dim(1);
    const float* a = logits1.data<float>();
    const float* b = logits2.data<float>();
    std::vector<float> pa((size_t)rows * V), pb((size_t)rows * V);
    softmax_rows(a, pa.data(), rows, V);
    softmax_rows(b, pb.data(), rows, V);
    const int* lb = labels.data<int>();
    double kl = 0.0;
    int64_t n = 0;
    for (int64_t r = 0; r < rows; r++) {
        if (lb[r] == ignore_index) continue;
        double k1 = 0.0, k2 = 0.0;
        for (int64_t v = 0; v < V; v++) {
            float x = std::max(pa[(size_t)r * V + v], 1e-12f);
            float y = std::max(pb[(size_t)r * V + v], 1e-12f);
            k1 += x * (std::log(x) - std::log(y));
            k2 += y * (std::log(y) - std::log(x));
        }
        kl += 0.5 * (k1 + k2);
        n++;
    }
    RDropOut o;
    o.ce = 0.5f * (c1.loss + c2.loss);
    o.kl = n > 0 ? (float)(kl / (double)n) : 0.0f;
    o.total = o.ce + alpha * o.kl;
    return o;
}

float infonce_loss(const Tensor& emb, const Tensor& pos, Tensor& demb, Tensor& dpos,
                   float temperature) {
    check_2d(emb, "emb");
    if (pos.numel() != emb.numel()) throw std::runtime_error("[loss] pos shape mismatch");
    int64_t N = emb.dim(0), D = emb.dim(1);
    const float* e = emb.data<float>();
    const float* p = pos.data<float>();
    demb = Tensor::zeros({N, D});
    dpos = Tensor::zeros({N, D});
    float* de = demb.data<float>();
    float* dp = dpos.data<float>();
    double total = 0.0;
    for (int64_t i = 0; i < N; i++) {
        double n1 = 0.0, n2 = 0.0;
        for (int64_t d = 0; d < D; d++) {
            n1 += (double)e[i * D + d] * e[i * D + d];
            n2 += (double)p[i * D + d] * p[i * D + d];
        }
        n1 = std::sqrt(n1 + 1e-12); n2 = std::sqrt(n2 + 1e-12);
        std::vector<float> sim((size_t)N);
        float m = -1e30f;
        for (int64_t j = 0; j < N; j++) {
            double s = 0.0;
            for (int64_t d = 0; d < D; d++)
                s += (e[i * D + d] / n1) * (p[j * D + d] / n2);
            sim[(size_t)j] = (float)(s / temperature);
            m = std::max(m, sim[(size_t)j]);
        }
        double den = 0.0;
        for (int64_t j = 0; j < N; j++) { sim[(size_t)j] = std::exp(sim[(size_t)j] - m); den += sim[(size_t)j]; }
        total += -std::log(sim[(size_t)i] / den);
        for (int64_t j = 0; j < N; j++) {
            float pr = sim[(size_t)j] / (float)den;
            float tgt = (j == i) ? 1.0f : 0.0f;
            float g = (pr - tgt) / temperature / (float)N;
            for (int64_t d = 0; d < D; d++) {
                de[i * D + d] += (float)(g * (p[j * D + d] / n2));
                dp[j * D + d] += (float)(g * (e[i * D + d] / n1));
            }
        }
    }
    return (float)(total / (double)N);
}

LossOut mse_loss_fn(const Tensor& pred, const Tensor& target, Tensor& dpred) {
    if (pred.numel() != target.numel()) throw std::runtime_error("[loss] mse shape mismatch");
    int64_t n = pred.numel();
    const float* x = pred.data<float>();
    const float* y = target.data<float>();
    dpred = Tensor::zeros(pred.shape());
    float* dl = dpred.data<float>();
    double se = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = (double)x[i] - (double)y[i];
        se += d * d;
        dl[i] = (float)(2.0 * d / (double)n);
    }
    LossOut o;
    o.loss = n > 0 ? (float)(se / (double)n) : 0.0f;
    return o;
}

float seq_z_loss(const Tensor& logits) {
    check_2d(logits, "logits");
    int64_t rows = logits.dim(0), V = logits.dim(1);
    const float* lg = logits.data<float>();
    double sum = 0.0;
    for (int64_t r = 0; r < rows; r++) {
        float m = lg[r * V];
        for (int64_t v = 1; v < V; v++) m = std::max(m, lg[r * V + v]);
        double s = 0.0;
        for (int64_t v = 0; v < V; v++) s += std::exp((double)lg[r * V + v] - m);
        double lse = (double)m + std::log(s);
        sum += lse * lse;
    }
    return rows > 0 ? (float)(sum / (double)rows) : 0.0f;
}

} // namespace dense_engine
} // namespace quant
