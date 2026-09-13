// Phase 5-F6 split of src/inference/inference_opt.cpp (verbatim move, no behavior change).
// This file: SpeculativeDecoder (D2) + TreeDecoder (D6).
// Declarations stay in include/quant/inference_opt.h.
#include "quant/inference_opt.h"
#include "quant/math.h"
#include "quant/int8_quant.h"
#include "quant/random.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <random>
#include <sstream>
#include <cstdio>
namespace quant {

// H3: unified epsilon shared with inference_opt.cpp / inference_paging.cpp
// (same name, same value, file-local so no header/ODR touch). Every
// speculative probability ratio in this TU (verify_tokens + generate,
// target and draft softmaxes plus accept-prob denominator) uses it —
// previously verify_tokens divided by bare `sum`/`d_sum` while generate
// used `sum + 1e-10f`.
namespace { constexpr float kInferenceEps = 1e-10f; }

// ===========================================================================
// D2: Speculative decoding — draft model + verify with top-k/top-p sampling
// ===========================================================================
SpeculativeDecoder::SpeculativeDecoder(Model* draft, Model* target, float gamma,
                                       float min_gamma, float max_gamma)
    : draft_(draft), target_(target), gamma_(gamma),
      min_gamma_(min_gamma), max_gamma_(max_gamma),
      sampler_(make_seed(resolve_base_seed(0), SeedStream::SpeculativeVerify, 0)),
      rng_(make_seed(resolve_base_seed(0), SeedStream::SpeculativeVerify, 1)) {
    sampler_cfg_.temperature = 1.0f;
    sampler_cfg_.top_k = 40;
    sampler_cfg_.top_p = 0.9f;
}

void SpeculativeDecoder::adapt_gamma() {
    acc_ema_ = 0.9f * acc_ema_ + 0.1f * acceptance_rate_;
    float ratio = acc_ema_ / std::max(1.0f - acc_ema_, 1e-6f);
    float new_gamma = std::round(5.0f * std::min(ratio, 5.0f));
    gamma_ = std::max(min_gamma_, std::min(max_gamma_, new_gamma));
}

bool SpeculativeDecoder::verify_tokens(const std::vector<int>& draft_tokens,
                                       const Tensor& target_logits, int vocab_size) {
    const float* logits_base = target_logits.data<float>();
    for (size_t i = 0; i < draft_tokens.size(); i++) {
        const float* logits_row = logits_base + i * vocab_size;
        float p_target = 0, p_draft = 0;
        float max_l = logits_row[0];
        for (int v = 1; v < vocab_size; v++)
            if (logits_row[v] > max_l) max_l = logits_row[v];
        float sum = 0;
        for (int v = 0; v < vocab_size; v++)
            sum += std::exp(logits_row[v] - max_l);
        p_target = std::exp(logits_row[draft_tokens[i]] - max_l) / (sum + kInferenceEps);
        total_count_++;
        if (draft_) {
            Tensor draft_input(Shape{1, 1});
            draft_input.data<float>()[0] = (float)draft_tokens[i];
            Tensor draft_pos(Shape{1, 1});
            draft_pos.data<float>()[0] = (float)i;
            Tensor draft_l = draft_->forward(draft_input, draft_pos);
            const float* dl = draft_l.data<float>();
            float d_max = dl[0];
            for (int v = 1; v < vocab_size; v++)
                if (dl[v] > d_max) d_max = dl[v];
            float d_sum = 0;
            for (int v = 0; v < vocab_size; v++)
                d_sum += std::exp(dl[v] - d_max);
            p_draft = std::exp(dl[draft_tokens[i]] - d_max) / (d_sum + kInferenceEps);
        } else {
            // P13: legit fallback — draft_ absent hai to p_draft uniform
            // (1/vocab) lena fake nahi hai. Metric ke liye flag set karo taaki
            // acceptance_rate_ ko "uniform-fallback" mode me padha ja sake.
            // Code path same rehta hai, todna nahi hai.
            draft_absent_fallback_ = true;
            p_draft = 1.0f / (float)vocab_size;
        }
        float accept_prob = std::min(1.0f, p_target / (p_draft + kInferenceEps));
        bool accepted = rng_.uniform() <= accept_prob;
        if (accepted) accepted_count_++;
        if (!accepted) {
            acceptance_rate_ = total_count_ > 0 ? (float)accepted_count_ / total_count_ : 0.0f;
            return false;
        }
    }
    acceptance_rate_ = total_count_ > 0 ? (float)accepted_count_ / total_count_ : 0.0f;
    return true;
}

std::vector<int> SpeculativeDecoder::generate(const std::vector<int>& prompt, int max_tokens) {
    static thread_local std::mt19937 rng(
        static_cast<std::mt19937::result_type>(
            make_seed(resolve_base_seed(0), SeedStream::SpeculativeFallback, 0)));
    std::vector<int> output = prompt;
    int vocab = draft_ ? (int)draft_->vocab_size() : 32000;
    accepted_count_ = 0;
    total_count_ = 0;
    draft_absent_fallback_ = false; // P13: har generate() run pe metric reset

    while ((int)output.size() < max_tokens) {
        int gamma = (int)gamma_;

        std::vector<int> draft_tokens;
        for (int g = 0; g < gamma && (int)output.size() < max_tokens; g++) {
            Tensor input(Shape{1, 1});
            Tensor pos(Shape{1, 1});
            input.data<float>()[0] = (float)output.back();
            pos.data<float>()[0] = (float)((int)output.size() - 1);

            Tensor logits;
            if (draft_) {
                logits = draft_->forward(input, pos);
                int next = sampler_.sample(logits.data<float>(), vocab, sampler_cfg_);
                draft_tokens.push_back(next);
                output.push_back(next);
            } else {
                // P13: legit fallback — draft_ absent hai to draft tokens ke
                // liye uniform-ish ramp logits use hote hain (fake nahi).
                Tensor logits(Shape{1, 1, vocab});
                float* ld = logits.data<float>();
                for (int v = 0; v < vocab; v++) ld[v] = (float)v / (float)vocab;
                int next = sampler_.sample(logits.data<float>(), vocab, sampler_cfg_);
                draft_tokens.push_back(next);
                output.push_back(next);
            }
        }

        int64_t num_draft = (int64_t)draft_tokens.size();
        Tensor target_input(Shape{1, num_draft});
        Tensor target_pos(Shape{1, num_draft});
        float* tid = target_input.data<float>();
        float* tpd = target_pos.data<float>();
        int64_t base_pos = (int64_t)output.size() - num_draft;
        for (int64_t i = 0; i < num_draft; i++) {
            tid[i] = (float)draft_tokens[(size_t)i];
            tpd[i] = (float)(base_pos + i);
        }

        Tensor target_logits;
        if (target_) {
            target_logits = target_->forward(target_input, target_pos);
        } else {
            target_logits = Tensor(Shape{1, num_draft, vocab});
            float* ld = target_logits.data<float>();
            for (int64_t i = 0; i < num_draft; i++)
                for (int v = 0; v < vocab; v++)
                    ld[i * vocab + v] = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
        }

        bool all_accepted = true;
        const float* tl_base = target_logits.data<float>();
        for (size_t i = 0; i < draft_tokens.size(); i++) {
            const float* logits_row = tl_base + i * vocab;
            float max_l = logits_row[0];
            for (int v = 1; v < vocab; v++)
                if (logits_row[v] > max_l) max_l = logits_row[v];
            float sum = 0;
            for (int v = 0; v < vocab; v++)
                sum += std::exp(logits_row[v] - max_l);
            float p_target = std::exp(logits_row[draft_tokens[i]] - max_l) / (sum + kInferenceEps);
            // D6 W4: verify_tokens path when draft_ present — exact p_draft
            // via draft_->forward with the same formula/epsilon as
            // verify_tokens() above; uniform 1/vocab ONLY when
            // draft_ == nullptr (legit fallback, flagged for metrics).
            // Mirrored inline rather than a literal verify_tokens() call
            // because verify_tokens() returns only bool and cannot report
            // the rejection index needed below for rewind +
            // adjusted-distribution resampling (signature frozen per H3:
            // no header change).
            float p_draft;
            if (draft_) {
                Tensor draft_input(Shape{1, 1});
                draft_input.data<float>()[0] = (float)draft_tokens[i];
                Tensor draft_pos(Shape{1, 1});
                draft_pos.data<float>()[0] = (float)i;
                Tensor draft_l = draft_->forward(draft_input, draft_pos);
                const float* dl = draft_l.data<float>();
                float d_max = dl[0];
                for (int v = 1; v < vocab; v++)
                    if (dl[v] > d_max) d_max = dl[v];
                float d_sum = 0;
                for (int v = 0; v < vocab; v++)
                    d_sum += std::exp(dl[v] - d_max);
                p_draft = std::exp(dl[draft_tokens[i]] - d_max) / (d_sum + kInferenceEps);
            } else {
                draft_absent_fallback_ = true;
                p_draft = 1.0f / (float)vocab;
            }
            total_count_++;
        float accept_prob = std::min(1.0f, p_target / (p_draft + kInferenceEps));
            if (rng_.uniform() <= accept_prob) {
                accepted_count_++;
                continue;
            }
            output.resize(output.size() - (draft_tokens.size() - i));
            std::vector<float> adjusted(vocab);
            float adj_sum = 0;
            for (int v = 0; v < vocab; v++) {
                float t_p = std::exp(logits_row[v] - max_l) / (sum + kInferenceEps);
                adjusted[(size_t)v] = std::max(0.0f, t_p - p_draft);
                adj_sum += adjusted[(size_t)v];
            }
            Tensor adj_logits(Shape{1, 1, vocab});
            float* adj_ld = adj_logits.data<float>();
            if (adj_sum > kInferenceEps) {
                float inv_adj = 1.0f / adj_sum;
                for (int v = 0; v < vocab; v++)
                    adj_ld[v] = adjusted[(size_t)v] * inv_adj;
            } else {
                std::memcpy(adj_ld, logits_row, vocab * sizeof(float));
            }
            int replacement = sampler_.sample(adj_ld, vocab, sampler_cfg_);
            output.push_back(replacement);
            all_accepted = false;
            acceptance_rate_ = total_count_ > 0 ? (float)accepted_count_ / total_count_ : 0.0f;
            break;
        }
        calls_since_adapt_++;
        if (calls_since_adapt_ >= adapt_interval_) {
            adapt_gamma();
            calls_since_adapt_ = 0;
        }
    }
    acceptance_rate_ = total_count_ > 0 ? (float)accepted_count_ / total_count_ : 0.0f;
    return output;
}

// ===========================================================================
// D6: Token tree decoding — beam search with branch-and-verify
// ===========================================================================
TreeDecoder::TreeDecoder(Model* model, int beam_width)
    : model_(model), beam_width_(beam_width) {}

void TreeDecoder::delete_subtree(Node* n) {
    if (!n) return;
    for (Node* child : n->children)
        delete_subtree(child);
    n->children.clear();
    delete n;
}

void TreeDecoder::expand_node(Node* n, int depth, int max_depth) {
    if (!model_ || depth >= max_depth || !n) return;
    if (!n->children.empty()) return;

    int V = (int)model_->vocab_size();
    Tensor input(Shape{1, 1});
    Tensor pos(Shape{1, 1});
    input.data<float>()[0] = (float)n->token;
    pos.data<float>()[0] = (float)depth;

    Tensor logits = model_->forward(input, pos);
    const float* ld = logits.data<float>();

    std::vector<std::pair<float, int>> candidates;
    candidates.reserve((size_t)V);
    for (int v = 0; v < V; v++)
        candidates.push_back({ld[v], v});

    int k = std::min(beam_width_, V);
    if (k > 0) {
        std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
    }

    for (int i = 0; i < k; i++) {
        Node* child = new Node;
        child->token = candidates[(size_t)i].second;
        child->score = n->score + candidates[(size_t)i].first;
        child->parent = n;
        n->children.push_back(child);
    }
}

void TreeDecoder::prune_tree(std::vector<Node*>& candidates) {
    if (candidates.empty()) return;
    std::sort(candidates.begin(), candidates.end(),
              [](Node* a, Node* b) { return a->score > b->score; });
    if ((int)candidates.size() > beam_width_) {
        for (int i = beam_width_; i < (int)candidates.size(); i++)
            delete_subtree(candidates[(size_t)i]);
        candidates.resize(beam_width_);
    }
}

std::vector<int> TreeDecoder::decode(const std::vector<int>& prompt, int max_tokens) {
    std::vector<int> output = prompt;
    Node* best_node = nullptr;

    for (int step = 0; step < max_tokens; step++) {
        Node root{output.back(), 0.0f, nullptr, {}};
        expand_node(&root, step, max_tokens);

        if (root.children.empty()) break;

        std::vector<Node*> candidates = root.children;
        prune_tree(candidates);
        if (candidates.empty()) break;

        Node* best = candidates[0];
        output.push_back(best->token);

        // Delete non-best children of root
        for (Node* child : root.children)
            if (child != best) delete_subtree(child);

        // Clean up previous best_node
        if (best_node) delete_subtree(best_node);

        // Keep best node alive for next iteration
        best_node = best;
        best_node->parent = nullptr;
        root.children.clear();
    }

    if (best_node) delete_subtree(best_node);
    return output;
}

} // namespace quant
