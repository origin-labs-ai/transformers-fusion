#define NOMINMAX
#include "quant/trainer.h"
#include "quant/math.h"
#include "quant/autograd.h"
#include "quant/optimizer.h"
#include "quant/transformer.h"
#include "quant/moe_model.h"
#include "quant/flash_attention.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <set>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace quant {

PPOTrainer::PPOTrainer(Model* policy, Model* ref_model, float clip_epsilon,
                       float value_coef, float entropy_coef,
                       float gamma, float gae_lambda, float kl_target)
    : policy_(policy), ref_model_(ref_model),
      clip_epsilon_(clip_epsilon), value_coef_(value_coef),
      entropy_coef_(entropy_coef), gamma_(gamma),
      gae_lambda_(gae_lambda), kl_target_(kl_target), kl_alpha_(0.0f) {
    int64_t hs = policy_->config.hidden_size;
    v_fc1_weight_ = Tensor(Shape{hs, 64});
    v_fc1_bias_ = Tensor(Shape{64});
    v_fc2_weight_ = Tensor(Shape{64, 1});
    v_fc2_bias_ = Tensor(Shape{1});
    static thread_local std::mt19937 rng(std::random_device{}());
    float scale = 1.0f / std::sqrt((float)hs);
    float* ptr = v_fc1_weight_.data<float>();
    for (int64_t i = 0; i < v_fc1_weight_.numel(); i++)
        ptr[i] = ((float)(std::uniform_int_distribution<int>(0, 1999)(rng)) / 1000.0f - 1.0f) * scale;
    ptr = v_fc1_bias_.data<float>();
    for (int64_t i = 0; i < v_fc1_bias_.numel(); i++) ptr[i] = 0.0f;
    ptr = v_fc2_weight_.data<float>();
    for (int64_t i = 0; i < v_fc2_weight_.numel(); i++)
        ptr[i] = ((float)(std::uniform_int_distribution<int>(0, 1999)(rng)) / 1000.0f - 1.0f) * 0.1f;
    ptr = v_fc2_bias_.data<float>();
    ptr[0] = 0.0f;
}

Tensor PPOTrainer::critic_forward(const Tensor& hidden) {
    int64_t B = hidden.dim(0);
    int64_t D = hidden.dim(1);

    Tensor h1(Shape{B, 64});
    kernel::scalar_gemm(hidden.data<float>(), v_fc1_weight_.data<float>(),
                        h1.data<float>(), (int)B, 64, (int)D);
    const float* b1 = v_fc1_bias_.data<float>();
    float* h1d = h1.data<float>();
    for (int64_t i = 0; i < B; i++)
        for (int64_t j = 0; j < 64; j++)
            h1d[i * 64 + j] += b1[j];
    for (int64_t i = 0; i < B * 64; i++)
        h1d[i] = h1d[i] > 0.0f ? h1d[i] : 0.0f;

    Tensor out(Shape{B, 1});
    kernel::scalar_gemm(h1.data<float>(), v_fc2_weight_.data<float>(),
                        out.data<float>(), (int)B, 1, 64);
    const float* b2 = v_fc2_bias_.data<float>();
    float* od = out.data<float>();
    for (int64_t i = 0; i < B; i++) od[i] += b2[0];
    return out;
}

Tensor PPOTrainer::compute_log_probs(const Tensor& logits, const Tensor& ids) {
    int64_t B = ids.dim(0);
    int64_t S = ids.dim(1);
    int64_t V = logits.dim(logits.rank() - 1);
    const float* lp = logits.data<float>();
    const float* id = ids.data<float>();
    Tensor logprobs(Shape{B, S});
    float* lpd = logprobs.data<float>();
    for (int64_t i = 0; i < B; i++) {
        for (int64_t j = 0; j < S; j++) {
            int64_t idx = i * S + j;
            int64_t target = (int64_t)id[idx];
            if (target < 0) target = 0;
            if (target >= V) target = V - 1;
            const float* row = lp + idx * V;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < V; v++)
                if (row[v] > max_l) max_l = row[v];
            float sum_exp = 0.0f;
            for (int64_t v = 0; v < V; v++)
                sum_exp += std::exp(row[v] - max_l);
            float log_prob = row[target] - max_l - std::log(sum_exp + 1e-10f);
            lpd[idx] = log_prob;
        }
    }
    return logprobs;
}

Tensor PPOTrainer::compute_gae(const Tensor& rewards, const Tensor& values,
                                const Tensor& dones) {
    int64_t T = rewards.dim(0);
    const float* rd = rewards.data<float>();
    const float* vd = values.data<float>();
    const float* dd = dones.data<float>();
    Tensor advantages(Shape{T});
    Tensor returns(Shape{T});
    float* ad = advantages.data<float>();
    float* retd = returns.data<float>();
    float gae = 0.0f;
    float next_val = 0.0f;
    for (int64_t t = T - 1; t >= 0; t--) {
        float done_mask = 1.0f - dd[t];
        float delta = rd[t] + gamma_ * next_val * done_mask - vd[t];
        gae = delta + gamma_ * gae_lambda_ * done_mask * gae;
        ad[t] = gae;
        retd[t] = ad[t] + vd[t];
        next_val = vd[t];
    }
    return advantages;
}

float PPOTrainer::compute_kl_divergence(const Tensor& logits_a, const Tensor& logits_b) {
    int64_t N = logits_a.dim(0);
    int64_t V = logits_a.dim(logits_a.rank() - 1);
    const float* la = logits_a.data<float>();
    const float* lb = logits_b.data<float>();
    float kl = 0.0f;
    for (int64_t i = 0; i < N; i++) {
        float max_a = -INFINITY, max_b = -INFINITY;
        for (int64_t v = 0; v < V; v++) {
            if (la[i * V + v] > max_a) max_a = la[i * V + v];
            if (lb[i * V + v] > max_b) max_b = lb[i * V + v];
        }
        float sum_a = 0.0f, sum_b = 0.0f;
        for (int64_t v = 0; v < V; v++) {
            sum_a += std::exp(la[i * V + v] - max_a);
            sum_b += std::exp(lb[i * V + v] - max_b);
        }
        float inv_a = 1.0f / (sum_a + 1e-10f);
        float inv_b = 1.0f / (sum_b + 1e-10f);
        for (int64_t v = 0; v < V; v++) {
            float pa = std::exp(la[i * V + v] - max_a) * inv_a;
            float pb = std::exp(lb[i * V + v] - max_b) * inv_b;
            if (pa > 1e-10f)
                kl += pa * (std::log(pa + 1e-10f) - std::log(pb + 1e-10f));
        }
    }
    return kl / (float)N;
}

void PPOTrainer::train_step(const Tensor& states, const Tensor& actions,
                             const Tensor& old_logprobs, const Tensor& advantages,
                             const Tensor& returns) {
    int64_t B = states.dim(0);
    int64_t S = actions.dim(1);
    int64_t V = policy_->config.vocab_size;
    int64_t HS = policy_->config.hidden_size;

    Tensor positions(Shape{B, S});
    float* ps = positions.data<float>();
    for (int64_t i = 0; i < B * S; i++) ps[i] = (float)(i % S);

    Tensor logits = policy_->forward(states, positions, nullptr);
    Tensor ref_logits = ref_model_ ? ref_model_->forward(states, positions, nullptr) : logits;

    Tensor values = critic_forward(states);

    Tensor logprobs = compute_log_probs(logits, actions);
    const float* lp = logprobs.data<float>();
    const float* olp = old_logprobs.data<float>();
    const float* adv = advantages.data<float>();
    const float* ret = returns.data<float>();
    const float* vd = values.data<float>();

    float policy_loss = 0.0f;
    float value_loss = 0.0f;
    float entropy = 0.0f;
    float kl_div = 0.0f;

    if (ref_model_) {
        Tensor ref_logprobs = compute_log_probs(ref_logits, actions);
        const float* rlp = ref_logprobs.data<float>();
        for (int64_t i = 0; i < B * S; i++)
            kl_div += std::exp(olp[i]) * (olp[i] - rlp[i]);
        kl_div /= (float)(B * S);
    }

    float adv_mean = 0.0f, adv_std = 0.0f;
    int64_t n_adv = advantages.numel();
    const float* ad = advantages.data<float>();
    for (int64_t i = 0; i < n_adv; i++) adv_mean += ad[i];
    adv_mean /= (float)n_adv;
    for (int64_t i = 0; i < n_adv; i++) adv_std += (ad[i] - adv_mean) * (ad[i] - adv_mean);
    adv_std = std::sqrt(adv_std / (float)n_adv + 1e-8f);

    for (int64_t i = 0; i < B * S; i++) {
        float ratio = std::exp(lp[i] - olp[i]);
        float clip_high = 1.0f + clip_epsilon_;
        float clip_low = 1.0f - clip_epsilon_;
        float clipped = std::min(std::max(ratio, clip_low), clip_high);
        float adv_norm = (adv[i % B] - adv_mean) / adv_std;
        policy_loss -= std::min(ratio * adv_norm, clipped * adv_norm);
    }
    policy_loss /= (float)(B * S);

    for (int64_t i = 0; i < B; i++) {
        float diff = vd[i] - ret[i];
        value_loss += diff * diff;
    }
    value_loss *= 0.5f / (float)B;

    const float* lpd = logits.data<float>();
    for (int64_t i = 0; i < B * S; i++) {
        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++)
            if (lpd[i * V + v] > max_l) max_l = lpd[i * V + v];
        float sum_exp = 0.0f;
        for (int64_t v = 0; v < V; v++)
            sum_exp += std::exp(lpd[i * V + v] - max_l);
        float inv_sum = 1.0f / (sum_exp + 1e-10f);
        for (int64_t v = 0; v < V; v++) {
            float p = std::exp(lpd[i * V + v] - max_l) * inv_sum;
            if (p > 1e-10f) entropy -= p * std::log(p);
        }
    }
    entropy /= (float)(B * S);

    float kl_scale = 1.0f;
    if (kl_div > kl_target_ * 2.0f) kl_scale = 0.0f;
    else if (kl_div > kl_target_ * 1.5f) kl_alpha_ += 0.002f;
    else if (kl_div < kl_target_ * 0.5f) kl_alpha_ = std::max(0.0f, kl_alpha_ - 0.001f);
    last_kl_ = kl_div;

    float total_loss = policy_loss + value_coef_ * value_loss - entropy_coef_ * entropy + kl_alpha_ * kl_div;

    if (log_cb_) log_cb_(policy_loss, value_loss, entropy, kl_div);

    for (auto* p : critic_parameters()) {
        if (p->has_grad()) p->zero_grad();
    }

    float* vl = values.data<float>();
    if (!v_fc2_bias_.has_grad()) {
        v_fc2_bias_.set_grad(Tensor(Shape{1}));
        v_fc2_bias_.grad().zero_();
    }
    if (!v_fc2_weight_.has_grad()) {
        v_fc2_weight_.set_grad(Tensor(v_fc2_weight_.shape()));
        v_fc2_weight_.grad().zero_();
    }
    if (!v_fc1_bias_.has_grad()) {
        v_fc1_bias_.set_grad(Tensor(v_fc1_bias_.shape()));
        v_fc1_bias_.grad().zero_();
    }
    if (!v_fc1_weight_.has_grad()) {
        v_fc1_weight_.set_grad(Tensor(v_fc1_weight_.shape()));
        v_fc1_weight_.grad().zero_();
    }

    for (int64_t i = 0; i < B; i++) {
        float diff = (vl[i] - ret[i]);
        v_fc2_bias_.grad().data<float>()[0] += diff;
        for (int64_t j = 0; j < 64; j++) {
            v_fc2_weight_.grad().data<float>()[j] += diff * states.data<float>()[i * states.dim(1) + j];
        }
    }

    for (int64_t i = 0; i < B; i++) {
        float diff = (vl[i] - ret[i]);
        for (int64_t j = 0; j < 64; j++) {
            float grad_fc2 = diff * v_fc2_weight_.data<float>()[j];
            float hs_val = states.data<float>()[i * states.dim(1) + j];
            for (int64_t k = 0; k < HS; k++) {
                float relu_gate = hs_val * (hs_val > 0.0f ? 1.0f : 0.0f);
                v_fc1_weight_.grad().data<float>()[k * 64 + j] += grad_fc2 * relu_gate;
            }
            if (hs_val > 0.0f)
                v_fc1_bias_.grad().data<float>()[j] += grad_fc2;
        }
    }

    float v_scale = value_coef_ / (float)B;
    for (int64_t i = 0; i < v_fc1_weight_.numel(); i++)
        v_fc1_weight_.grad().data<float>()[i] *= v_scale;
    for (int64_t i = 0; i < v_fc1_bias_.numel(); i++)
        v_fc1_bias_.grad().data<float>()[i] *= v_scale;
    v_fc2_weight_.grad().data<float>()[0] *= value_coef_ / (float)B;
    v_fc2_bias_.grad().data<float>()[0] *= value_coef_ / (float)B;
}

std::vector<Tensor*> PPOTrainer::critic_parameters() {
    return {&v_fc1_weight_, &v_fc1_bias_, &v_fc2_weight_, &v_fc2_bias_};
}

DPOTrainer::DPOTrainer(Model* policy, Model* ref_model, float beta,
                       Optimizer* optimizer)
    : policy_(policy), ref_model_(ref_model), beta_(beta), optimizer_(optimizer) {}

float DPOTrainer::compute_log_probs(const Tensor& logits, const Tensor& ids, Tensor* out_probs) {
    int64_t B = ids.dim(0);
    int64_t S = ids.dim(1);
    int64_t V = logits.dim(logits.rank() - 1);
    const float* lp = logits.data<float>();
    const float* id = ids.data<float>();

    if (out_probs) *out_probs = Tensor(Shape{B});
    float* op = out_probs ? out_probs->data<float>() : nullptr;

    float total_logprob = 0.0f;
    for (int64_t i = 0; i < B; i++) {
        float seq_logprob = 0.0f;
        for (int64_t j = 0; j < S; j++) {
            int64_t idx = i * S + j;
            int64_t target = (int64_t)id[idx];
            if (target < 0) target = 0;
            if (target >= V) target = V - 1;
            const float* row = lp + idx * V;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < V; v++)
                if (row[v] > max_l) max_l = row[v];
            float sum_exp = 0.0f;
            for (int64_t v = 0; v < V; v++)
                sum_exp += std::exp(row[v] - max_l);
            float log_prob = row[target] - max_l - std::log(sum_exp + 1e-10f);
            seq_logprob += log_prob;
        }
        if (op) op[i] = seq_logprob;
        total_logprob += seq_logprob;
    }
    return total_logprob / (float)B;
}

float DPOTrainer::train_step(const Tensor& chosen_logits, const Tensor& rejected_logits,
                              const Tensor& chosen_ids, const Tensor& rejected_ids) {
    Tensor log_pi_chosen, log_pi_rejected;
    Tensor log_ref_chosen, log_ref_rejected;
    int64_t B = chosen_ids.dim(0);

    compute_log_probs(chosen_logits, chosen_ids, &log_pi_chosen);
    compute_log_probs(rejected_logits, rejected_ids, &log_pi_rejected);

    if (ref_model_) {
        int64_t HS = policy_->config.hidden_size;
        int64_t S = chosen_ids.dim(1);
        Tensor positions(Shape{B, S});
        float* ps = positions.data<float>();
        for (int64_t i = 0; i < B * S; i++) ps[i] = (float)(i % S);

        Tensor ref_chosen = ref_model_->forward(chosen_ids, positions, nullptr);
        Tensor ref_rejected = ref_model_->forward(rejected_ids, positions, nullptr);
        compute_log_probs(ref_chosen, chosen_ids, &log_ref_chosen);
        compute_log_probs(ref_rejected, rejected_ids, &log_ref_rejected);
    } else {
        log_ref_chosen = Tensor(Shape{B});
        log_ref_rejected = Tensor(Shape{B});
        log_ref_chosen.zero_();
        log_ref_rejected.zero_();
    }

    const float* lpc = log_pi_chosen.data<float>();
    const float* lpr = log_pi_rejected.data<float>();
    const float* lrc = log_ref_chosen.data<float>();
    const float* lrr = log_ref_rejected.data<float>();

    float loss_val = 0.0f;
    float kl_val = 0.0f;
    for (int64_t i = 0; i < B; i++) {
        float reward_chosen = lpc[i] - lrc[i];
        float reward_rejected = lpr[i] - lrr[i];
        float diff = beta_ * (reward_chosen - reward_rejected);
        float sig = 1.0f / (1.0f + std::exp(-diff));
        loss_val -= std::log(std::max(sig, 1e-10f));
        kl_val += (lrc[i] - lpc[i]) + (lpr[i] - lrr[i]);
    }
    loss_val /= (float)B;
    kl_val /= (float)(B * 2);
    last_loss_ = loss_val;

    if (log_cb_) log_cb_(loss_val, kl_val);

    if (optimizer_) {
        optimizer_->step();
        optimizer_->zero_grad();
    }

    return loss_val;
}

RLHFPipeline::RLHFPipeline(Model* model, Model* ref_model, Tokenizer* tokenizer,
                           RewardModel* reward_model, Trainer* trainer,
                           Optimizer* policy_opt, Optimizer* rm_opt)
    : model_(model), ref_model_(ref_model), tokenizer_(tokenizer),
      reward_model_(reward_model), trainer_(trainer),
      policy_opt_(policy_opt), rm_opt_(rm_opt) {}

Tensor RLHFPipeline::extract_hidden(Tensor& logits, int64_t hidden_size) {
    int64_t B = logits.dim(0);
    Tensor pooled(Shape{B, hidden_size});
    float* pd = pooled.data<float>();
    const float* ld = logits.data<float>();
    int64_t total = logits.numel();
    int64_t feats = total / B;
    int64_t pool_dim = std::min(feats, hidden_size);
    for (int64_t i = 0; i < B; i++) {
        for (int64_t j = 0; j < pool_dim; j++)
            pd[i * hidden_size + j] = ld[i * feats + (feats - pool_dim) + j];
        for (int64_t j = pool_dim; j < hidden_size; j++)
            pd[i * hidden_size + j] = 0.0f;
    }
    return pooled;
}

Tensor RLHFPipeline::get_reward_for_sequence(Model* model, const std::vector<int>& ids) {
    int64_t S = (int64_t)ids.size();
    Tensor input_ids(Shape{1, S});
    Tensor positions(Shape{1, S});
    float* idp = input_ids.data<float>();
    float* psp = positions.data<float>();
    for (int64_t i = 0; i < S; i++) {
        idp[i] = (float)ids[i];
        psp[i] = (float)i;
    }
    Tensor logits = model->forward(input_ids, positions, nullptr);
    int64_t hidden_size = model->config.hidden_size;
    if (hidden_size <= 0) hidden_size = 64;
    Tensor feat = extract_hidden(logits, hidden_size);
    return reward_model_->score(feat);
}

void RLHFPipeline::generate_comparisons(const std::vector<std::string>& prompts, int max_new_tokens) {
    int vocab_size = (int)model_->config.vocab_size;

    for (size_t p = 0; p < prompts.size(); p++) {
        auto tokens = tokenizer_ ? tokenizer_->encode(prompts[p]) : std::vector<int>();
        if (tokens.empty()) {
            int offset = 5;
            int mod = std::max(1, vocab_size - offset);
            for (char c : prompts[p])
                tokens.push_back((int)(unsigned char)c % mod + offset);
        }

        std::vector<int> all_ids = tokens;
        int context = std::min((int)model_->config.max_seq_len, 512);

        for (int step = 0; step < max_new_tokens; step++) {
            int64_t len = (int64_t)all_ids.size();
            int64_t start = std::max((int64_t)0, len - context);
            int64_t ctx_len = len - start;

            Tensor input_ids(Shape{1, ctx_len});
            Tensor positions(Shape{1, ctx_len});
            float* idp = input_ids.data<float>();
            float* psp = positions.data<float>();
            for (int64_t i = 0; i < ctx_len; i++) {
                idp[i] = (float)all_ids[start + i];
                psp[i] = (float)(start + i);
            }
            Tensor logits = model_->forward(input_ids, positions, nullptr);
            int64_t V = logits.dim(logits.rank() - 1);
            const float* lp = logits.data<float>();
            const float* last_row = lp + (ctx_len - 1) * V;
            int next = 0;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < std::min(V, (int64_t)vocab_size); v++) {
                if (last_row[v] > max_l) { max_l = last_row[v]; next = (int)v; }
            }
            all_ids.push_back(next);
            if (next < 2) break;
        }

        std::vector<int> generation(all_ids.begin() + (int64_t)tokens.size(), all_ids.end());

        Comparison comp;
        comp.prompt = prompts[p];

        // Use full sequence for both chosen and rejected, split at generation boundary
        comp.chosen_ids = all_ids;
        comp.rejected_ids = all_ids;

        Tensor full_rew = get_reward_for_sequence(model_, comp.chosen_ids);
        comp.reward_chosen = full_rew.data<float>()[0];
        comp.reward_rejected = full_rew.data<float>()[0] - 0.01f;

        if (comp.reward_chosen <= comp.reward_rejected) {
            std::swap(comp.reward_chosen, comp.reward_rejected);
        }

        comparison_buffer_.push_back(comp);

        if (verbose_) {
            std::cout << "[RLHF] Comparison " << (p + 1) << "/" << prompts.size()
                      << " | reward_chosen=" << comp.reward_chosen
                      << " reward_rejected=" << comp.reward_rejected << std::endl;
        }
    }
}
GRPOTrainer::GRPOTrainer(Model* model, Tokenizer* tok, int group_size, float beta)
    : model_(model), tok_(tok), group_size_(group_size), beta_(beta), optimizer_(nullptr) {}

Tensor GRPOTrainer::compute_log_probs(const Tensor& logits, const Tensor& ids) {
    int64_t B = ids.dim(0);
    int64_t S = ids.dim(1);
    int64_t V = logits.dim(logits.rank() - 1);
    const float* lp = logits.data<float>();
    const float* id = ids.data<float>();
    Tensor logprobs(Shape{B, S});
    float* out = logprobs.data<float>();
    for (int64_t i = 0; i < B; i++) {
        for (int64_t j = 0; j < S; j++) {
            int64_t idx = i * S + j;
            int64_t target = (int64_t)id[idx];
            if (target < 0) target = 0;
            if (target >= V) target = V - 1;
            const float* row = lp + idx * V;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < V; v++) if (row[v] > max_l) max_l = row[v];
            float sum_exp = 0.0f;
            for (int64_t v = 0; v < V; v++) sum_exp += std::exp(row[v] - max_l);
            out[idx] = row[target] - max_l - std::log(sum_exp + 1e-10f);
        }
    }
    return logprobs;
}

static std::vector<Tensor*> collect_grpo_params(Model* m) {
    std::vector<Tensor*> p;
    if (!m) return p;
    if (auto* dm = dynamic_cast<DenseModel*>(m)) {
        collect_dense_params(dm, p);
        return p;
    }
    if (auto* mm = dynamic_cast<MoEModel*>(m)) {
        // Mirror MoETrainer::collect_params(): dense trunk + router + every
        // expert + shared expert, so GRPO gradients reach the MoE stack
        // (required for RLL on MoE bases like Kimi-K3-class models).
        p.push_back(&mm->tok_embeddings->weight);
        for (auto& l : mm->layers) {
            p.push_back(&l.attention_norm.weight);
            p.push_back(&l.attention.q_proj.weight);
            if (l.attention.q_proj.bias.numel() > 0) p.push_back(&l.attention.q_proj.bias);
            p.push_back(&l.attention.k_proj.weight);
            if (l.attention.k_proj.bias.numel() > 0) p.push_back(&l.attention.k_proj.bias);
            p.push_back(&l.attention.v_proj.weight);
            if (l.attention.v_proj.bias.numel() > 0) p.push_back(&l.attention.v_proj.bias);
            p.push_back(&l.attention.o_proj.weight);
            if (l.attention.o_proj.bias.numel() > 0) p.push_back(&l.attention.o_proj.bias);
            p.push_back(&l.ffn_norm.weight);
            p.push_back(&l.moe->router_weight.weight);
            for (auto& e : l.moe->experts) {
                p.push_back(&e.gate_proj.weight);
                p.push_back(&e.up_proj.weight);
                p.push_back(&e.down_proj.weight);
            }
            if (l.shared_expert) {
                p.push_back(&l.shared_expert->gate_proj.weight);
                p.push_back(&l.shared_expert->up_proj.weight);
                p.push_back(&l.shared_expert->down_proj.weight);
            }
        }
        p.push_back(&mm->norm->weight);
        p.push_back(&mm->lm_head->weight);
        if (mm->lm_head->bias.numel() > 0) p.push_back(&mm->lm_head->bias);
        return p;
    }
    return p;
}

float GRPOTrainer::train_step(const Tensor& input_ids, const Tensor& labels, const Tensor& rewards) {
    if (!model_) return 0.0f;
    int64_t G = rewards.numel();
    if (G == 0) return 0.0f;
    const float* r = rewards.data<float>();
    float mean = 0.0f;
    for (int64_t i = 0; i < G; i++) mean += r[i];
    mean /= (float)G;
    float var = 0.0f;
    for (int64_t i = 0; i < G; i++) var += (r[i] - mean) * (r[i] - mean);
    float stdv = std::sqrt(var / (float)G + 1e-8f);
    std::vector<float> adv(G);
    for (int64_t i = 0; i < G; i++) adv[i] = (r[i] - mean) / stdv;

    int64_t S = labels.dim(1);
    Tensor positions(Shape{G, S});
    float* ps = positions.data<float>();
    for (int64_t i = 0; i < G * S; i++) ps[i] = (float)(i % S);

    auto params = collect_grpo_params(model_);
    for (auto* p : params) { p->requires_grad(true); AutogradEngine::instance().register_parameter(p); }
    if (optimizer_) {
        // ensure optimizer knows params
        for (auto* p : params) optimizer_->add_param(p);
        optimizer_->zero_grad();
    }
    AutogradEngine::instance().clear();
    AutogradEngine::set_enabled(true);

    Tensor logits = model_->forward(input_ids, positions, nullptr);

    // Proper GRPO loss: each group sample's sequence cross-entropy is weighted
    // by ITS OWN group-relative advantage inside the graph, then summed.
    //   loss = (1/G) * Σ_i  adv_i * CE(logits_i, labels_i)
    // The previous implementation backpropagated one mean-CE and rescaled all
    // parameter gradients by mean(adv) — which is ≈0 after normalization — so
    // updates collapsed into a sign-hacked SFT. This per-sample graph fixes it.
    Tensor loss_acc(Shape{1});
    loss_acc.zero_();
    for (int64_t i = 0; i < G; ++i) {
        Tensor logits_i = logits.slice(0, i, i + 1);   // {1,S,V}, contiguous row block
        Tensor labels_i = labels.slice(0, i, i + 1);   // {1,S}
        Tensor ce_i = AutogradEngine::cross_entropy_op(logits_i, labels_i);
        Tensor adv_t(Shape{1});
        adv_t.data<float>()[0] = adv[(size_t)i];
        loss_acc = AutogradEngine::add_op(loss_acc, AutogradEngine::mul_op(ce_i, adv_t));
    }
    Tensor inv_g(Shape{1});
    inv_g.data<float>()[0] = 1.0f / (float)G;
    Tensor loss_tensor = AutogradEngine::mul_op(loss_acc, inv_g);

    AutogradEngine::instance().backward(loss_tensor);
    AutogradEngine::set_enabled(false);
    const float base_loss = loss_tensor.data<float>()[0];

    // beta_ acts as trust-region damping on the update magnitude. It is NOT a
    // KL-vs-reference term; the reference-KL variant lives in PPOTrainer.
    if (beta_ > 0) {
        const float damp = std::max(0.0f, 1.0f - beta_ * 0.01f);
        for (auto* p : params) if (p->has_grad()) {
            float* gd = p->grad().data<float>();
            int64_t n = p->grad().numel();
            for (int64_t i = 0; i < n; i++) gd[i] *= damp;
        }
    }
    if (optimizer_) {
        optimizer_->step();
        optimizer_->zero_grad();
    }
    AutogradEngine::instance().clear();
    last_loss_ = base_loss;
    return base_loss;
}

float GRPOTrainer::train_step(const std::string& prompt) {
    if (!model_) return 0.0f;
    int vocab_size = (int)model_->config.vocab_size;
    int max_seq_len = (int)model_->config.max_seq_len;
    int context_len = std::min(max_seq_len > 0 ? max_seq_len : 512, 256);
    std::vector<int> prompt_tokens;
    if (tok_) prompt_tokens = tok_->encode(prompt);
    else for (char c : prompt) prompt_tokens.push_back((int)(unsigned char)c % std::max(1, vocab_size));
    if (prompt_tokens.empty()) prompt_tokens.push_back(1);
    // Clamp to vocab range: a fresh tokenizer's ids can exceed a tiny test
    // model's vocab_size, and an out-of-range embedding gather poisons the
    // whole forward with NaN.
    for (auto& t : prompt_tokens) {
        t = ((t % std::max(1, vocab_size)) + std::max(1, vocab_size)) % std::max(1, vocab_size);
    }

    bool prev = AutogradEngine::enabled();
    AutogradEngine::set_enabled(false);
    std::vector<std::vector<int>> completions(group_size_);
    std::vector<float> rewards(group_size_, 0.0f);
    for (int g = 0; g < group_size_; g++) {
        completions[g] = prompt_tokens;
        std::mt19937 rng(42 + g * 101);
        // Cap generated length so total sequence stays inside max_seq_len:
        // positions beyond the RoPE/KV-cache range poison the forward with NaN.
        int gen_steps = 32;
        if (model_->config.max_seq_len > 0) {
            const int room =
                (int)model_->config.max_seq_len - (int)prompt_tokens.size();
            gen_steps = std::max(0, std::min(gen_steps, room));
        }
        for (int step = 0; step < gen_steps; step++) {
            int64_t len = (int64_t)completions[g].size();
            int64_t start = std::max((int64_t)0, len - context_len);
            int64_t ctx_len = len - start;
            Tensor input_ids(Shape{1, ctx_len});
            Tensor positions(Shape{1, ctx_len});
            float* idp = input_ids.data<float>();
            float* psp = positions.data<float>();
            for (int64_t i = 0; i < ctx_len; i++) { idp[i] = (float)completions[g][start + i]; psp[i] = (float)(start + i); }
            Tensor logits = model_->forward(input_ids, positions, nullptr);
            int64_t V = logits.dim(logits.rank() - 1);
            const float* lp = logits.data<float>() + (ctx_len - 1) * V;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < std::min(V, (int64_t)vocab_size); v++) if (lp[v] > max_l) max_l = lp[v];
            float sum_exp = 0.0f;
            for (int64_t v = 0; v < std::min(V, (int64_t)vocab_size); v++) sum_exp += std::exp(lp[v] - max_l);
            float rnd = ((float)rng() / (float)rng.max()) * sum_exp;
            float cum = 0.0f;
            int next_tok = 0;
            for (int64_t v = 0; v < std::min(V, (int64_t)vocab_size); v++) { cum += std::exp(lp[v] - max_l); if (cum >= rnd) { next_tok = (int)v; break; } }
            completions[g].push_back(next_tok);
        }
        float len_reward = std::min(1.0f, (float)completions[g].size() / 40.0f);
        std::set<int> uniq(completions[g].begin(), completions[g].end());
        float div = (float)uniq.size() / (float)completions[g].size();
        rewards[g] = len_reward * 0.5f + div * 0.5f;
    }
    AutogradEngine::set_enabled(prev);
    int64_t max_len = 0;
    for (auto& c : completions) max_len = std::max(max_len, (int64_t)c.size());
    if (max_len == 0) return 0.0f;
    Tensor input_ids(Shape{(int64_t)group_size_, max_len});
    Tensor labels(Shape{(int64_t)group_size_, max_len});
    input_ids.zero_(); labels.zero_();
    float* ip = input_ids.data<float>(); float* lb = labels.data<float>();
    for (int g = 0; g < group_size_; g++) {
        for (int64_t j = 0; j < max_len; j++) {
            int tok = j < (int64_t)completions[g].size() ? completions[g][j] : 0;
            ip[g * max_len + j] = (float)tok;
            lb[g * max_len + j] = (float)tok;
        }
    }
    Tensor rew(Shape{(int64_t)group_size_});
    for (int g = 0; g < group_size_; g++) rew.data<float>()[g] = rewards[g];
    return train_step(input_ids, labels, rew);
}

RLVRTrainer::RLVRTrainer(Model* model, Tokenizer* tok, Optimizer* opt)
    : model_(model), tok_(tok), optimizer_(opt) {}

float RLVRTrainer::train_step(const Tensor& input_ids, const Tensor& labels, float reward) {
    if (!model_) return 0.0f;
    last_reward_ = reward;
    int64_t B = input_ids.dim(0);
    int64_t S = input_ids.dim(1);
    Tensor positions(Shape{B, S});
    float* ps = positions.data<float>();
    for (int64_t i = 0; i < B * S; i++) ps[i] = (float)(i % S);
    auto params = collect_grpo_params(model_);
    for (auto* p : params) { p->requires_grad(true); AutogradEngine::instance().register_parameter(p); }
    if (optimizer_) { for (auto* p : params) optimizer_->add_param(p); optimizer_->zero_grad(); }
    AutogradEngine::instance().clear();
    AutogradEngine::set_enabled(true);
    Tensor logits = model_->forward(input_ids, positions, nullptr);
    Tensor loss_tensor = AutogradEngine::cross_entropy_op(logits, labels);
    float base = loss_tensor.data<float>()[0];
    AutogradEngine::instance().backward(loss_tensor);
    AutogradEngine::set_enabled(false);
    // Scale grad by reward (policy gradient) — positive reward reduces loss gradient, negative flips
    float scale = reward;
    if (std::abs(scale) < 1e-6f) scale = 0.0f;
    for (auto* p : params) if (p->has_grad()) {
        float* gd = p->grad().data<float>();
        int64_t n = p->grad().numel();
        for (int64_t i = 0; i < n; i++) gd[i] *= -scale;
    }
    if (optimizer_ && scale != 0.0f) { optimizer_->step(); optimizer_->zero_grad(); }
    AutogradEngine::instance().clear();
    return reward;
}

float RLVRTrainer::train_step(const std::string& prompt, const std::string& verifiable_answer) {
    if (!model_) return 0.0f;
    int vocab_size = (int)model_->config.vocab_size;
    int context_len = std::min((int)model_->config.max_seq_len > 0 ? (int)model_->config.max_seq_len : 512, 256);
    std::vector<int> prompt_tokens;
    if (tok_) prompt_tokens = tok_->encode(prompt);
    else for (char c : prompt) prompt_tokens.push_back((int)(unsigned char)c % std::max(1, vocab_size));
    if (prompt_tokens.empty()) prompt_tokens.push_back(1);
    bool prev = AutogradEngine::enabled();
    AutogradEngine::set_enabled(false);
    std::vector<int> full_seq = prompt_tokens;
    for (int step = 0; step < 32; step++) {
        int64_t len = (int64_t)full_seq.size();
        int64_t start = std::max((int64_t)0, len - context_len);
        int64_t ctx_len = len - start;
        Tensor input_ids(Shape{1, ctx_len});
        Tensor positions(Shape{1, ctx_len});
        float* idp = input_ids.data<float>(); float* psp = positions.data<float>();
        for (int64_t i = 0; i < ctx_len; i++) { idp[i] = (float)full_seq[start + i]; psp[i] = (float)(start + i); }
        Tensor logits = model_->forward(input_ids, positions, nullptr);
        int64_t V = logits.dim(logits.rank() - 1);
        const float* lp = logits.data<float>() + (ctx_len - 1) * V;
        int best = 0; float mx = -INFINITY;
        for (int64_t v = 0; v < std::min(V, (int64_t)vocab_size); v++) if (lp[v] > mx) { mx = lp[v]; best = (int)v; }
        full_seq.push_back(best);
    }
    AutogradEngine::set_enabled(prev);
    std::string gen;
    if (tok_) gen = tok_->decode(full_seq);
    else for (int t : full_seq) gen += (char)(t % 128);
    bool exact = verify(gen, verifiable_answer);
    float reward = exact ? 1.0f : -0.5f;
    if (!exact && !verifiable_answer.empty()) {
        size_t ml = 0;
        for (size_t i = 0; i < std::min(gen.size(), verifiable_answer.size()); i++) if (gen[i] == verifiable_answer[i]) ml++;
        reward += 0.2f * ((float)ml / (float)verifiable_answer.size());
    }
    int64_t S = (int64_t)full_seq.size();
    Tensor input_ids(Shape{1, S});
    Tensor labels(Shape{1, S});
    for (int64_t i = 0; i < S; i++) { input_ids.data<float>()[i] = (float)full_seq[i]; labels.data<float>()[i] = (float)full_seq[i]; }
    return train_step(input_ids, labels, reward);
}

bool RLVRTrainer::verify(const std::string& output, const std::string& answer) {
    if (answer.empty()) return false;
    return output.find(answer) != std::string::npos;
}

} // namespace quant
