#define NOMINMAX
#include "quant/trainer.h"
#include "quant/math.h"
#include "quant/autograd.h"
#include "quant/optimizer.h"
#include "quant/transformer.h"
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
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace quant {

void RLHFPipeline::train_reward_model(const std::vector<Comparison>& data, int epochs, int batch_size) {
    if (data.empty()) return;
    int64_t hidden_size = model_->config.hidden_size;
    if (hidden_size <= 0) hidden_size = 64;

    int64_t n = (int64_t)data.size();
    int64_t n_batches = (n + batch_size - 1) / batch_size;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;
        int count = 0;

        for (int64_t b = 0; b < n_batches; b++) {
            int64_t start = b * batch_size;
            int64_t end = std::min(start + batch_size, n);
            int64_t B = end - start;

            Tensor chosen_feat(Shape{B, hidden_size});
            Tensor rejected_feat(Shape{B, hidden_size});
            float* cf = chosen_feat.data<float>();
            float* rf = rejected_feat.data<float>();
            cf[0] = 0.0f; rf[0] = 0.0f;

            for (int64_t i = 0; i < B; i++) {
                const auto& comp = data[(size_t)(start + i)];

                auto fill_feature = [&](Tensor& feat, const std::vector<int>& ids) {
                    int64_t S = (int64_t)ids.size();
                    Tensor inp(Shape{1, std::max((int64_t)1, S)});
                    Tensor pos(Shape{1, std::max((int64_t)1, S)});
                    float* idp = inp.data<float>();
                    float* psp = pos.data<float>();
                    for (int64_t k = 0; k < S; k++) {
                        idp[k] = (float)ids[k];
                        psp[k] = (float)k;
                    }
                    if (S == 0) { idp[0] = 1.0f; psp[0] = 0.0f; }
                    Tensor logits = model_->forward(inp, pos, nullptr);
                    int64_t pool = std::min(logits.numel() / 1, hidden_size);
                    const float* ld = logits.data<float>();
                    int64_t offset = logits.numel() / 1 - pool;
                    for (int64_t k = 0; k < pool; k++)
                        feat.data<float>()[i * hidden_size + k] = ld[offset + k];
                };

                fill_feature(chosen_feat, comp.chosen_ids);
                fill_feature(rejected_feat, comp.rejected_ids);
            }

            reward_model_->train_step(chosen_feat, rejected_feat, rm_opt_);
        }

        if (verbose_) {
            std::cout << "[RLHF] RM epoch " << (epoch + 1) << "/" << epochs << std::endl;
        }
    }
}

void RLHFPipeline::ppo_finetune(const std::vector<std::string>& prompts,
                                 int n_ppo_steps, int ppo_batch_size,
                                 int max_new_tokens) {
    if (!reward_model_ || !model_) return;

    PPOTrainer ppo(model_, ref_model_);
    int vocab_size = (int)model_->config.vocab_size;
    int64_t hidden_size = model_->config.hidden_size;
    if (hidden_size <= 0) hidden_size = 64;

    ppo.set_log_callback([this](float pl, float vl, float ent, float kl) {
        if (verbose_) {
            std::cout << "[PPO] policy_loss=" << pl << " value_loss=" << vl
                      << " entropy=" << ent << " kl=" << kl << std::endl;
        }
        metrics_.ppo_policy_loss = pl;
        metrics_.ppo_value_loss = vl;
        metrics_.ppo_entropy = ent;
        metrics_.ppo_kl = kl;
    });

    for (int step = 0; step < n_ppo_steps; step++) {
        int prompt_idx = step % (int)prompts.size();
        const std::string& prompt = prompts[prompt_idx];
        auto tokens = tokenizer_ ? tokenizer_->encode(prompt) : std::vector<int>();
        if (tokens.empty()) {
            int offset = 5;
            for (char c : prompt)
                tokens.push_back((int)(unsigned char)c % std::max(1, vocab_size - offset) + offset);
        }

        std::vector<int> all_ids = tokens;
        int context = 512;
        for (int t = 0; t < max_new_tokens; t++) {
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
            float max_l_val = -INFINITY;
            for (int64_t v = 0; v < V; v++) {
                if (last_row[v] > max_l_val) { max_l_val = last_row[v]; next = (int)v; }
            }
            all_ids.push_back(next);
            if (next < 2) break;
        }

        std::vector<int> gen_ids(all_ids.begin() + (int64_t)tokens.size(), all_ids.end());
        if (gen_ids.empty()) gen_ids.push_back(1);

        int64_t S = (int64_t)all_ids.size();
        Tensor seq_tensor(Shape{1, S});
        float* seq_d = seq_tensor.data<float>();
        for (int64_t i = 0; i < S; i++) seq_d[i] = (float)all_ids[i];
        Tensor reward = reward_model_->score(seq_tensor);
        float reward_val = reward.data<float>()[0];

        Tensor states(Shape{1, S});
        Tensor actions(Shape{1, S});
        float* sd = states.data<float>();
        float* ad = actions.data<float>();
        for (int64_t i = 0; i < S; i++) {
            sd[i] = (float)all_ids[i];
            ad[i] = (float)all_ids[i];
        }

        Tensor positions(Shape{1, S});
        float* psp = positions.data<float>();
        for (int64_t i = 0; i < S; i++) psp[i] = (float)i;

        Tensor logits = model_->forward(states, positions, nullptr);
        int64_t V = logits.dim(logits.rank() - 1);

        Tensor old_logprobs(Shape{1, S});
        float* ol = old_logprobs.data<float>();
        const float* lpd = logits.data<float>();
        for (int64_t i = 0; i < S; i++) {
            int target = (int)ad[i];
            if (target < 0) target = 0;
            if (target >= V) target = (int)V - 1;
            const float* row = lpd + i * V;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < V; v++) if (row[v] > max_l) max_l = row[v];
            float sum_e = 0.0f;
            for (int64_t v = 0; v < V; v++) sum_e += std::exp(row[v] - max_l);
            ol[i] = row[target] - max_l - std::log(sum_e + 1e-10f);
        }

        Tensor adv(Shape{1});
        adv.data<float>()[0] = reward_val;
        Tensor ret(Shape{1});
        ret.data<float>()[0] = reward_val;

        ppo.train_step(states, actions, old_logprobs, adv, ret);

        metrics_.mean_reward = reward_val;
        metrics_.step++;

        if (log_cb_) log_cb_(metrics_);
    }
}

Tensor RLHFPipeline::compute_gae(const Tensor& rewards, const Tensor& values,
                                  float gamma, float lam) {
    int64_t T = rewards.dim(0);
    Tensor advantages(Shape{T});
    float* adv = advantages.data<float>();
    const float* r = rewards.data<float>();
    const float* v = values.data<float>();
    float gae = 0.0f;
    for (int64_t t = T - 1; t >= 0; t--) {
        float delta = r[t] + gamma * ((t + 1 < T) ? v[t + 1] : 0.0f) - v[t];
        gae = delta + gamma * lam * ((t + 1 < T) ? gae : 0.0f);
        adv[t] = gae;
    }
    return advantages;
}

Tensor RLHFPipeline::compute_returns_discounted(const Tensor& rewards, float gamma) {
    int64_t T = rewards.dim(0);
    Tensor returns(Shape{T});
    float* ret = returns.data<float>();
    const float* r = rewards.data<float>();
    float acc = 0.0f;
    for (int64_t t = T - 1; t >= 0; t--) {
        acc = r[t] + gamma * acc;
        ret[t] = acc;
    }
    return returns;
}

float RLHFPipeline::compute_kl_between_policies(const Tensor& logits_pi, const Tensor& logits_ref,
                                                  const Tensor& actions) {
    int64_t B = actions.dim(0);
    int64_t S = actions.dim(1);
    int64_t V = logits_pi.dim(logits_pi.rank() - 1);
    const float* lp = logits_pi.data<float>();
    const float* lr = logits_ref.data<float>();
    const float* a = actions.data<float>();
    float kl = 0.0f;
    int64_t count = 0;
    for (int64_t i = 0; i < B; i++) {
        for (int64_t j = 0; j < S; j++) {
            int64_t idx = (i * S + j) * V;
            int target = (int)a[i * S + j];
            if (target < 0 || target >= V) continue;
            float max_p = -INFINITY, max_r = -INFINITY;
            for (int64_t v = 0; v < V; v++) {
                if (lp[idx + v] > max_p) max_p = lp[idx + v];
                if (lr[idx + v] > max_r) max_r = lr[idx + v];
            }
            float sum_p = 0.0f, sum_r = 0.0f;
            for (int64_t v = 0; v < V; v++) {
                sum_p += std::exp(lp[idx + v] - max_p);
                sum_r += std::exp(lr[idx + v] - max_r);
            }
            float log_p = lp[idx + target] - max_p - std::log(sum_p + 1e-10f);
            float log_r = lr[idx + target] - max_r - std::log(sum_r + 1e-10f);
            kl += std::exp(log_p) * (log_p - log_r);
            count++;
        }
    }
    return count > 0 ? kl / (float)count : 0.0f;
}

float RLHFPipeline::compute_reward_accuracy(const std::vector<Comparison>& data) {
    if (data.empty()) return 0.0f;
    if (!reward_model_) return 0.0f;
    int correct = 0;
    int64_t hs = model_->config.hidden_size;
    if (hs <= 0) hs = 64;
    for (size_t i = 0; i < data.size(); i++) {
        Tensor chosen_score_t = get_reward_for_sequence(model_, data[i].chosen_ids);
        Tensor rejected_score_t = get_reward_for_sequence(model_, data[i].rejected_ids);
        float chosen_score = chosen_score_t.data<float>()[0];
        float rejected_score = rejected_score_t.data<float>()[0];
        if (chosen_score > rejected_score) correct++;
    }
    return (float)correct / (float)data.size();
}

float RLHFPipeline::train_reward_model_epoch_with_accuracy(
    const std::vector<Comparison>& train_data,
    const std::vector<Comparison>& val_data,
    int epochs, int batch_size, const char* split_name) {
    int64_t hidden_size = model_->config.hidden_size;
    if (hidden_size <= 0) hidden_size = 64;
    int64_t n = (int64_t)train_data.size();
    int64_t n_batches = (n + batch_size - 1) / batch_size;
    float best_val_loss = INFINITY;
    float best_acc = 0.0f;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;
        int count = 0;
        for (int64_t b = 0; b < n_batches; b++) {
            int64_t start = b * batch_size;
            int64_t end = std::min(start + batch_size, n);
            int64_t B = end - start;

            Tensor chosen_feat(Shape{B, hidden_size});
            Tensor rejected_feat(Shape{B, hidden_size});
            Tensor chosen_scores(Shape{B, 1});
            Tensor rejected_scores(Shape{B, 1});

            for (int64_t i = 0; i < B; i++) {
                const auto& comp = train_data[(size_t)(start + i)];
                auto fill = [&](Tensor& feat, const std::vector<int>& ids) {
                    int64_t S = (int64_t)ids.size();
                    if (S == 0) return;
                    Tensor inp(Shape{1, S});
                    Tensor pos(Shape{1, S});
                    float* idp = inp.data<float>();
                    float* psp = pos.data<float>();
                    for (int64_t k = 0; k < S; k++) { idp[k] = (float)ids[k]; psp[k] = (float)k; }
                    Tensor logits = model_->forward(inp, pos, nullptr);
                    int64_t pool = std::min(logits.numel() / 1, hidden_size);
                    const float* ld = logits.data<float>();
                    int64_t offset = logits.numel() / 1 - pool;
                    for (int64_t k = 0; k < pool; k++)
                        feat.data<float>()[i * hidden_size + k] = ld[offset + k];
                };
                fill(chosen_feat, comp.chosen_ids);
                fill(rejected_feat, comp.rejected_ids);
            }
            float loss = reward_model_->train_step(chosen_feat, rejected_feat, rm_opt_);
            epoch_loss += loss;
            count++;
        }

        float val_acc = 0.0f;
        float val_loss = 0.0f;
        if (!val_data.empty()) {
            int val_count = 0;
            for (size_t i = 0; i < val_data.size(); i += (size_t)batch_size) {
                size_t end = std::min(i + (size_t)batch_size, val_data.size());
                size_t B = end - i;
                Tensor v_chosen(Shape{(int64_t)B, hidden_size});
                Tensor v_rejected(Shape{(int64_t)B, hidden_size});
                for (size_t j = i; j < end; j++) {
                    const auto& comp = val_data[j];
                    auto fill = [&](Tensor& feat, const std::vector<int>& ids, size_t idx) {
                        int64_t S = (int64_t)ids.size();
                        if (S == 0) return;
                        Tensor inp(Shape{1, S});
                        Tensor pos(Shape{1, S});
                        float* idp = inp.data<float>();
                        float* psp = pos.data<float>();
                        for (int64_t k = 0; k < S; k++) { idp[k] = (float)ids[k]; psp[k] = (float)k; }
                        Tensor logits = model_->forward(inp, pos, nullptr);
                        int64_t pool = std::min(logits.numel() / 1, hidden_size);
                        const float* ld = logits.data<float>();
                        int64_t offset = logits.numel() / 1 - pool;
                        for (int64_t k = 0; k < pool; k++)
                            feat.data<float>()[idx * hidden_size + k] = ld[offset + k];
                    };
                    fill(v_chosen, comp.chosen_ids, j - i);
                    fill(v_rejected, comp.rejected_ids, j - i);
                }
                val_loss += reward_model_->train_step(v_chosen, v_rejected, nullptr);
                val_count++;
            }
            val_acc = compute_reward_accuracy(val_data);
            if (val_count > 0) val_loss /= (float)val_count;
            if (val_loss < best_val_loss) { best_val_loss = val_loss; best_acc = val_acc; }
        }

        if (verbose_) {
            std::cout << "[" << (split_name ? split_name : "RM") << "] epoch "
                      << (epoch + 1) << "/" << epochs
                      << " loss=" << (count > 0 ? epoch_loss / (float)count : 0.0f);
            if (!val_data.empty())
                std::cout << " val_loss=" << val_loss << " val_acc=" << val_acc;
            std::cout << std::endl;
        }
    }
    return best_acc;
}

void RLHFPipeline::dpo_finetune(const std::vector<Comparison>& comparisons,
                                 int n_steps, int batch_size) {
    if (comparisons.empty() || !model_ || !ref_model_) return;
    DPOTrainer dpo(model_, ref_model_, 0.1f);
    int64_t n = (int64_t)comparisons.size();
    int64_t n_batches = (n + batch_size - 1) / batch_size;

    for (int step = 0; step < n_steps; step++) {
        float total_loss = 0.0f;
        int count = 0;
        for (int64_t b = 0; b < n_batches; b++) {
            int64_t start = b * batch_size;
            int64_t end = std::min(start + batch_size, n);
            int64_t B = end - start;
            int64_t max_S = 0;
            for (int64_t i = 0; i < B; i++) {
                const auto& comp = comparisons[(size_t)(start + i)];
                if ((int64_t)comp.chosen_ids.size() > max_S) max_S = (int64_t)comp.chosen_ids.size();
                if ((int64_t)comp.rejected_ids.size() > max_S) max_S = (int64_t)comp.rejected_ids.size();
            }
            if (max_S == 0) continue;
            int64_t S = std::min(max_S, (int64_t)512);

            Tensor chosen_ids(Shape{B, S});
            Tensor rejected_ids(Shape{B, S});
            chosen_ids.zero_();
            rejected_ids.zero_();

            for (int64_t i = 0; i < B; i++) {
                const auto& comp = comparisons[(size_t)(start + i)];
                for (size_t j = 0; j < comp.chosen_ids.size() && (int64_t)j < S; j++)
                    chosen_ids.data<float>()[i * S + j] = (float)comp.chosen_ids[j];
                for (size_t j = 0; j < comp.rejected_ids.size() && (int64_t)j < S; j++)
                    rejected_ids.data<float>()[i * S + j] = (float)comp.rejected_ids[j];
            }

            Tensor positions(Shape{B, S});
            float* psp = positions.data<float>();
            for (int64_t i = 0; i < B * S; i++) psp[i] = (float)(i % S);

            Tensor chosen_logits = model_->forward(chosen_ids, positions, nullptr);
            Tensor rejected_logits = model_->forward(rejected_ids, positions, nullptr);

            float loss = dpo.train_step(chosen_logits, rejected_logits, chosen_ids, rejected_ids);
            total_loss += loss;
            count++;

            metrics_.dpo_loss = loss;
        }
        if (verbose_ && count > 0) {
            std::cout << "[DPO] step " << (step + 1) << "/" << n_steps
                      << " loss=" << (total_loss / (float)count)
                      << " beta=" << dpo.beta() << std::endl;
        }
    }
}

void RLHFPipeline::grpo_finetune(const std::vector<std::string>& prompts, int n_steps, int max_new_tokens) {
    if (!model_ || prompts.empty()) return;
    GRPOTrainer grpo(model_, tokenizer_, 4, 0.04f);
    if (policy_opt_) grpo.set_optimizer(policy_opt_);
    for (int s = 0; s < n_steps; s++) {
        const std::string& p = prompts[s % prompts.size()];
        float loss = grpo.train_step(p);
        metrics_.dpo_loss = loss;
        if (verbose_) std::cout << "[GRPO] step " << (s+1) << "/" << n_steps << " loss=" << loss << std::endl;
        if (log_cb_) log_cb_(metrics_);
    }
}

void RLHFPipeline::rlvf_finetune(const std::vector<std::pair<std::string,std::string>>& qa_pairs, int n_steps, int max_new_tokens) {
    if (!model_ || qa_pairs.empty()) return;
    RLVRTrainer rlv(model_, tokenizer_, policy_opt_);
    for (int s = 0; s < n_steps; s++) {
        auto& pr = qa_pairs[s % qa_pairs.size()];
        float rew = rlv.train_step(pr.first, pr.second);
        metrics_.mean_reward = rew;
        if (verbose_) std::cout << "[RLVR] step " << (s+1) << "/" << n_steps << " reward=" << rew << std::endl;
        if (log_cb_) log_cb_(metrics_);
    }
}

void RLHFPipeline::run(int n_rounds, int n_prompts, int n_ppo_steps) {
    std::vector<std::string> prompts;
    const char* default_prompts[] = {
        "Explain why the sky is blue.",
        "Write a short poem about AI.",
        "What is the meaning of life?",
        "Describe how a computer works.",
        "What is machine learning?",
        "Tell me a joke.",
        "How do you make a sandwich?",
        "What is the capital of France?",
        "Write a haiku about nature.",
        "Explain gravity in simple terms."
    };
    int np = sizeof(default_prompts) / sizeof(default_prompts[0]);
    for (int i = 0; i < n_prompts; i++)
        prompts.push_back(default_prompts[i % np]);

    std::vector<Comparison> heldout;
    size_t holdout_size = std::max((size_t)1, prompts.size() / 5);
    for (size_t i = prompts.size() - holdout_size; i < prompts.size(); i++)
        heldout.push_back(comparison_buffer_.size() > i ?
                          comparison_buffer_[i] : Comparison{});
    std::vector<std::string> train_prompts(prompts.begin(), prompts.end() - (ptrdiff_t)holdout_size);

    for (int round = 0; round < n_rounds; round++) {
        if (verbose_) {
            std::cout << "\n=== RLHF Round " << (round + 1) << "/" << n_rounds << " ===" << std::endl;
        }

        metrics_.step = round;

        generate_comparisons(train_prompts, 32);

        int rm_epochs = std::max(1, 3 + round);
        train_reward_model_epoch_with_accuracy(comparison_buffer_, heldout, rm_epochs, 8, "RM");

        int ppo_steps = std::max(1, n_ppo_steps + round * 10);
        ppo_finetune(train_prompts, ppo_steps, 8, 32);

        if (comparison_buffer_.size() > 4) {
            dpo_finetune(comparison_buffer_, std::max(1, 2 + round), 4);
        }

        float accuracy = compute_reward_accuracy(comparison_buffer_);
        metrics_.reward_accuracy = accuracy;

        if (verbose_) {
            std::cout << "[RLHF] Round " << (round + 1) << " complete: "
                      << "rm_acc=" << accuracy
                      << " ppo_kl=" << metrics_.ppo_kl
                      << " dpo_loss=" << metrics_.dpo_loss
                      << std::endl;
        }

        if (log_cb_) log_cb_(metrics_);
    }

    if (verbose_) {
        std::cout << "\n=== RLHF Complete ===" << std::endl;
        std::cout << "Total comparisons: " << comparison_buffer_.size() << std::endl;
        std::cout << "Final metrics: reward_loss=" << metrics_.reward_loss
                  << " ppo_kl=" << metrics_.ppo_kl
                  << " dpo_loss=" << metrics_.dpo_loss
                  << " rm_acc=" << metrics_.reward_accuracy << std::endl;
    }
}

} // namespace quant
