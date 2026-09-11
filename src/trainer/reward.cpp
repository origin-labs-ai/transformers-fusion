#define NOMINMAX
#include "quant/reward.h"
#include "quant/math.h"
#include "quant/kernel.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <random>

namespace quant {

RewardModel::RewardModel(int64_t hidden_size, int64_t num_layers, float importance_factor)
    : hidden_size_(hidden_size), num_layers_(num_layers), importance_factor_(importance_factor) {
    int64_t mid = hidden_size_ * 4;
    fc1_weight_ = Tensor(Shape{hidden_size_, mid});
    fc1_bias_ = Tensor(Shape{mid});
    fc2_weight_ = Tensor(Shape{mid, hidden_size_});
    fc2_bias_ = Tensor(Shape{hidden_size_});
    fc3_weight_ = Tensor(Shape{hidden_size_, 1});
    fc3_bias_ = Tensor(Shape{1});

    float scale1 = 1.0f / std::sqrt((float)hidden_size_);
    float scale2 = 1.0f / std::sqrt((float)mid);
    float* ptr;
    int64_t n;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    ptr = fc1_weight_.data<float>(); n = fc1_weight_.numel();
    for (int64_t i = 0; i < n; i++) ptr[i] = dist(rng) * scale1;
    ptr = fc1_bias_.data<float>(); n = fc1_bias_.numel();
    for (int64_t i = 0; i < n; i++) ptr[i] = 0.0f;

    ptr = fc2_weight_.data<float>(); n = fc2_weight_.numel();
    for (int64_t i = 0; i < n; i++) ptr[i] = dist(rng) * scale2;
    ptr = fc2_bias_.data<float>(); n = fc2_bias_.numel();
    for (int64_t i = 0; i < n; i++) ptr[i] = 0.0f;

    ptr = fc3_weight_.data<float>(); n = fc3_weight_.numel();
    for (int64_t i = 0; i < n; i++) ptr[i] = dist(rng) * 0.01f;
    ptr = fc3_bias_.data<float>();
    ptr[0] = 0.0f;
}

Tensor RewardModel::forward_mlp(const Tensor& x) {
    int64_t B = x.dim(0);
    int64_t D = x.dim(1);

    Tensor h1(Shape{B, fc1_weight_.dim(1)});
    kernel::scalar_gemm(x.data<float>(), fc1_weight_.data<float>(), h1.data<float>(), (int)B, (int)fc1_weight_.dim(1), (int)D);
    const float* b1 = fc1_bias_.data<float>();
    float* h1d = h1.data<float>();
    int64_t M1 = fc1_weight_.dim(1);
    for (int64_t i = 0; i < B; i++)
        for (int64_t j = 0; j < M1; j++)
            h1d[i * M1 + j] += b1[j];

    for (int64_t i = 0; i < B * M1; i++)
        h1d[i] = h1d[i] > 0.0f ? h1d[i] : 0.0f;

    Tensor h2(Shape{B, fc2_weight_.dim(1)});
    kernel::scalar_gemm(h1.data<float>(), fc2_weight_.data<float>(), h2.data<float>(), (int)B, (int)fc2_weight_.dim(1), (int)M1);
    const float* b2 = fc2_bias_.data<float>();
    float* h2d = h2.data<float>();
    int64_t M2 = fc2_weight_.dim(1);
    for (int64_t i = 0; i < B; i++)
        for (int64_t j = 0; j < M2; j++)
            h2d[i * M2 + j] += b2[j];

    for (int64_t i = 0; i < B * M2; i++)
        h2d[i] = h2d[i] > 0.0f ? h2d[i] : 0.0f;

    Tensor out(Shape{B, 1});
    kernel::scalar_gemm(h2.data<float>(), fc3_weight_.data<float>(), out.data<float>(), (int)B, 1, (int)M2);
    const float* b3 = fc3_bias_.data<float>();
    float* od = out.data<float>();
    for (int64_t i = 0; i < B; i++)
        od[i] += b3[0];

    return out;
}

Tensor RewardModel::sigmoid_op(const Tensor& x) {
    Tensor out(x.shape());
    const float* xd = x.data<float>();
    float* od = out.data<float>();
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++)
        od[i] = 1.0f / (1.0f + std::exp(-xd[i]));
    return out;
}

Tensor RewardModel::score(const Tensor& sequence) {
    return forward_mlp(sequence);
}

Tensor RewardModel::score_pair(const Tensor& chosen, const Tensor& rejected) {
    int64_t B = chosen.dim(0);
    Tensor r_chosen = forward_mlp(chosen);
    Tensor r_rejected = forward_mlp(rejected);
    Tensor diff(Shape{B});
    const float* rc = r_chosen.data<float>();
    const float* rr = r_rejected.data<float>();
    float* dd = diff.data<float>();
    for (int64_t i = 0; i < B; i++)
        dd[i] = rc[i] - rr[i];
    return diff;
}

Tensor RewardModel::reward_loss(const Tensor& chosen_rewards, const Tensor& rejected_rewards) {
    int64_t B = chosen_rewards.dim(0);
    Tensor diff(Shape{B});
    const float* c = chosen_rewards.data<float>();
    const float* r = rejected_rewards.data<float>();
    float* d = diff.data<float>();
    for (int64_t i = 0; i < B; i++)
        d[i] = c[i] - r[i];

    float loss_val = 0.0f;
    for (int64_t i = 0; i < B; i++) {
        float sig = 1.0f / (1.0f + std::exp(-d[i]));
        loss_val -= std::log(std::max(sig, 1e-10f));
    }
    loss_val /= (float)B;

    Tensor loss(Shape{1});
    loss.data<float>()[0] = loss_val;
    return loss;
}

float RewardModel::train_step(const Tensor& chosen, const Tensor& rejected, Optimizer* opt, float loss_scale) {
    int64_t B = chosen.dim(0);
    int64_t D = chosen.dim(1);
    int64_t M1 = fc1_weight_.dim(1);
    int64_t M2 = fc2_weight_.dim(1);

    Tensor h1(Shape{B, M1});
    kernel::scalar_gemm(chosen.data<float>(), fc1_weight_.data<float>(), h1.data<float>(), (int)B, (int)M1, (int)D);
    float* h1d = h1.data<float>();
    const float* b1 = fc1_bias_.data<float>();
    for (int64_t i = 0; i < B; i++)
        for (int64_t j = 0; j < M1; j++)
            h1d[i * M1 + j] += b1[j];
    for (int64_t i = 0; i < B * M1; i++)
        if (h1d[i] < 0) h1d[i] = 0.0f;

    Tensor h1_r(Shape{B, M1});
    kernel::scalar_gemm(rejected.data<float>(), fc1_weight_.data<float>(), h1_r.data<float>(), (int)B, (int)M1, (int)D);
    float* h1r = h1_r.data<float>();
    for (int64_t i = 0; i < B; i++)
        for (int64_t j = 0; j < M1; j++)
            h1r[i * M1 + j] += b1[j];
    for (int64_t i = 0; i < B * M1; i++)
        if (h1r[i] < 0) h1r[i] = 0.0f;

    Tensor h2(Shape{B, M2});
    kernel::scalar_gemm(h1.data<float>(), fc2_weight_.data<float>(), h2.data<float>(), (int)B, (int)M2, (int)M1);
    float* h2d = h2.data<float>();
    const float* b2 = fc2_bias_.data<float>();
    for (int64_t i = 0; i < B; i++)
        for (int64_t j = 0; j < M2; j++)
            h2d[i * M2 + j] += b2[j];
    for (int64_t i = 0; i < B * M2; i++)
        if (h2d[i] < 0) h2d[i] = 0.0f;

    Tensor h2_r(Shape{B, M2});
    kernel::scalar_gemm(h1_r.data<float>(), fc2_weight_.data<float>(), h2_r.data<float>(), (int)B, (int)M2, (int)M1);
    float* h2r = h2_r.data<float>();
    for (int64_t i = 0; i < B; i++)
        for (int64_t j = 0; j < M2; j++)
            h2r[i * M2 + j] += b2[j];
    for (int64_t i = 0; i < B * M2; i++)
        if (h2r[i] < 0) h2r[i] = 0.0f;

    Tensor chosen_reward(Shape{B, 1});
    kernel::scalar_gemm(h2.data<float>(), fc3_weight_.data<float>(), chosen_reward.data<float>(), (int)B, 1, (int)M2);
    float* cr = chosen_reward.data<float>();
    const float* b3 = fc3_bias_.data<float>();
    for (int64_t i = 0; i < B; i++) cr[i] += b3[0];

    Tensor rejected_reward(Shape{B, 1});
    kernel::scalar_gemm(h2_r.data<float>(), fc3_weight_.data<float>(), rejected_reward.data<float>(), (int)B, 1, (int)M2);
    float* rr = rejected_reward.data<float>();
    for (int64_t i = 0; i < B; i++) rr[i] += b3[0];

    float loss_val = 0.0f;
    int correct = 0;
    std::vector<float> sig_vals(B);
    for (int64_t i = 0; i < B; i++) {
        float diff = cr[i] - rr[i];
        float sig = 1.0f / (1.0f + std::exp(-diff));
        sig_vals[i] = sig;
        loss_val -= std::log(std::max(sig, 1e-10f));
        if (diff > 0) correct++;
    }
    loss_val /= (float)B;

    if (log_cb_) log_cb_(loss_val, (float)correct / (float)B);

    float inv_batch = 1.0f / (float)B;
    Tensor d_fc3_weight(fc3_weight_.shape());
    Tensor d_fc3_bias(fc3_bias_.shape());
    Tensor d_fc2_weight(fc2_weight_.shape());
    Tensor d_fc2_bias(fc2_bias_.shape());
    Tensor d_fc1_weight(fc1_weight_.shape());
    Tensor d_fc1_bias(fc1_bias_.shape());
    d_fc3_weight.zero_(); d_fc3_bias.zero_();
    d_fc2_weight.zero_(); d_fc2_bias.zero_();
    d_fc1_weight.zero_(); d_fc1_bias.zero_();

    for (int64_t i = 0; i < B; i++) {
        float ds = inv_batch * (sig_vals[i] - 1.0f);
        if (loss_scale != 1.0f) ds *= loss_scale;

        float dcr = ds;
        float drr = -ds;

        for (int64_t j = 0; j < M2; j++) {
            d_fc3_weight.data<float>()[j] += dcr * h2d[i * M2 + j];
        }
        d_fc3_bias.data<float>()[0] += dcr;

        for (int64_t j = 0; j < M2; j++) {
            d_fc3_weight.data<float>()[j] += drr * h2r[i * M2 + j];
        }
        d_fc3_bias.data<float>()[0] += drr;

        for (int64_t j = 0; j < M2; j++) {
            float dh2 = dcr * fc3_weight_.data<float>()[j];
            float dh2r = drr * fc3_weight_.data<float>()[j];
            for (int64_t k = 0; k < M1; k++) {
                float h2a = h2d[i * M2 + j];
                float h2ra = h2r[i * M2 + j];
                if (h2a > 0) {
                    d_fc2_weight.data<float>()[k * M2 + j] += dh2 * h1d[i * M1 + k];
                }
                if (h2ra > 0) {
                    d_fc2_weight.data<float>()[k * M2 + j] += dh2r * h1r[i * M1 + k];
                }
            }
            if (h2d[i * M2 + j] > 0) d_fc2_bias.data<float>()[j] += dh2;
            if (h2r[i * M2 + j] > 0) d_fc2_bias.data<float>()[j] += dh2r;
        }

        for (int64_t j = 0; j < M2; j++) {
            float dh2 = dcr * fc3_weight_.data<float>()[j];
            float dh2r = drr * fc3_weight_.data<float>()[j];
            float dh1_step = (h2d[i * M2 + j] > 0) ? dh2 : 0.0f;
            float dh1r_step = (h2r[i * M2 + j] > 0) ? dh2r : 0.0f;
            for (int64_t k = 0; k < M1; k++) {
                float h1a = h1d[i * M1 + k];
                float h1ra = h1r[i * M1 + k];
                for (int64_t d = 0; d < D; d++) {
                    if (h1a > 0)
                        d_fc1_weight.data<float>()[d * M1 + k] += dh1_step * chosen.data<float>()[i * D + d];
                    if (h1ra > 0)
                        d_fc1_weight.data<float>()[d * M1 + k] += dh1r_step * rejected.data<float>()[i * D + d];
                }
                if (h1a > 0) d_fc1_bias.data<float>()[k] += dh1_step;
                if (h1ra > 0) d_fc1_bias.data<float>()[k] += dh1r_step;
            }
        }
    }

    float inv_scale = 1.0f / loss_scale;
    auto apply_grad = [&](Tensor& param, Tensor& grad) {
        float* pd = param.data<float>();
        const float* gd = grad.data<float>();
        int64_t n = param.numel();
        for (int64_t i = 0; i < n; i++) {
            if (!param.has_grad()) {
                Tensor g(param.shape());
                g.zero_();
                param.set_grad(g);
            }
            param.grad().data<float>()[i] = gd[i] * inv_scale;
        }
    };

    apply_grad(fc1_weight_, d_fc1_weight);
    apply_grad(fc1_bias_, d_fc1_bias);
    apply_grad(fc2_weight_, d_fc2_weight);
    apply_grad(fc2_bias_, d_fc2_bias);
    apply_grad(fc3_weight_, d_fc3_weight);
    apply_grad(fc3_bias_, d_fc3_bias);

    opt->step();
    opt->zero_grad();
    last_loss_ = loss_val;
    return loss_val;
}

std::vector<Tensor*> RewardModel::parameters() {
    return { &fc1_weight_, &fc1_bias_, &fc2_weight_, &fc2_bias_, &fc3_weight_, &fc3_bias_ };
}

void RewardModel::save(const std::string& path) const {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return;
    int32_t h = (int32_t)hidden_size_;
    int32_t nl = (int32_t)num_layers_;
    fwrite(&h, sizeof(h), 1, fp);
    fwrite(&nl, sizeof(nl), 1, fp);
    auto save_tensor = [&](const Tensor& t) {
        int32_t rank = (int32_t)t.rank();
        fwrite(&rank, sizeof(rank), 1, fp);
        for (int32_t i = 0; i < rank; i++) {
            int64_t d = t.dim(i);
            fwrite(&d, sizeof(d), 1, fp);
        }
        fwrite(t.data<float>(), t.numel() * sizeof(float), 1, fp);
    };
    save_tensor(fc1_weight_); save_tensor(fc1_bias_);
    save_tensor(fc2_weight_); save_tensor(fc2_bias_);
    save_tensor(fc3_weight_); save_tensor(fc3_bias_);
    fclose(fp);
}

void RewardModel::load(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return;
    // P0 fix: every fread validated + rank/dims bounded. Old code trusted
    // all bytes: corrupt h/nl/rank/dims => corrupt shapes, unbounded allocs,
    // half-filled tensors. Any short/invalid read => keep in-memory model.
    int32_t h = 0, nl = 0;
    if (fread(&h, sizeof(h), 1, fp) != 1) { fclose(fp); return; }
    if (fread(&nl, sizeof(nl), 1, fp) != 1) { fclose(fp); return; }
    if (h <= 0 || h > 65536 || nl < 0 || nl > 1024) { fclose(fp); return; }
    bool ok = true;
    auto load_tensor = [&](Tensor& t) {
        if (!ok) return;
        int32_t rank = 0;
        if (fread(&rank, sizeof(rank), 1, fp) != 1) { ok = false; return; }
        if (rank < 0 || rank > 8) { ok = false; return; }
        Shape shape;
        shape.rank = rank;
        int64_t numel = 1;
        for (int32_t i = 0; i < rank; i++) {
            int64_t d = 0;
            if (fread(&d, sizeof(d), 1, fp) != 1) { ok = false; return; }
            if (d <= 0 || d > (int64_t)1 << 30) { ok = false; return; }
            shape.dims[i] = d;
            numel *= d;
        }
        t = Tensor(shape);
        if (numel > 0 &&
            fread(t.data<float>(), (size_t)numel * sizeof(float), 1, fp) != 1) {
            ok = false;
            return;
        }
    };
    Tensor w1, b1, w2, b2, w3, b3;
    load_tensor(w1); load_tensor(b1);
    load_tensor(w2); load_tensor(b2);
    load_tensor(w3); load_tensor(b3);
    fclose(fp);
    if (!ok) return; // keep previous in-memory weights on corrupt file
    hidden_size_ = h;
    num_layers_ = nl;
    fc1_weight_ = std::move(w1); fc1_bias_ = std::move(b1);
    fc2_weight_ = std::move(w2); fc2_bias_ = std::move(b2);
    fc3_weight_ = std::move(w3); fc3_bias_ = std::move(b3);
}

float RewardModel::compute_abstention_reward(const std::string& output, float confidence) {
    if (confidence < 0.5f) {
        std::string lower_out = output;
        for (char& c : lower_out) c = std::tolower(c);
        if (lower_out.find("i don't know") != std::string::npos || lower_out.find("i do not know") != std::string::npos) {
            return 1.0f;
        }
        return -2.0f;
    }
    return 0.0f;
}

class AsymmetricReward {
public:
    float score(const std::string& output, float confidence, bool is_fabricated) {
        if (is_fabricated) {
            return -3.0f * (1.0f - confidence);
        }
        std::string lower_out = output;
        for (char& c : lower_out) c = std::tolower(c);
        if (lower_out.find("i don't know") != std::string::npos || lower_out.find("i do not know") != std::string::npos) {
            return 0.5f;
        }
        return 1.0f * confidence;
    }
};

} // namespace quant
