#include "quant/eval.h"
#include "quant/math.h"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cctype>

namespace quant {

static int64_t min_i(int64_t a, int64_t b) { return a < b ? a : b; }
static int64_t max_i(int64_t a, int64_t b) { return a > b ? a : b; }

EvalResult eval_perplexity(Model* model, const std::vector<int>& tokens,
                           int context_size, int stride) {
    EvalResult r;
    if (!model || tokens.empty()) return r;
    int64_t V = model->vocab_size();
    int64_t S = (int64_t)tokens.size();
    if (S <= 1) return r;

    double nll = 0.0;
    int64_t nll_count = 0;

    for (int64_t start = 0; start < S - 1; start += stride) {
        int64_t end = min_i(start + context_size, S - 1);
        int64_t seq_len = end - start;
        if (seq_len < 1) continue;

        Tensor input({1, seq_len});
        Tensor positions({1, seq_len});
        for (int64_t i = 0; i < seq_len; i++) {
            input.data<float>()[i] = (float)tokens[(size_t)(start + i)];
            positions.data<float>()[i] = (float)i;
        }

        Tensor logits = model->forward(input, positions);
        int64_t logit_dim = logits.numel() / seq_len;

        for (int64_t i = 0; i < seq_len - 1; i++) {
            int64_t target = (size_t)(start + i + 1) < tokens.size()
                             ? tokens[(size_t)(start + i + 1)] : 0;
            if (target >= V) target = 0;
            const float* row = logits.data<float>() + i * logit_dim;
            // BUGFIX: max-subtracted softmax — max_l over the row keeps
            // exp(row-max_l) <= 1 so large logits cannot overflow to INF;
            // non-finite logits are skipped so a NaN logit cannot poison
            // the max (NaN max -> every exp NaN).
            float max_l = -INFINITY;
            for (int64_t v = 0; v < min_i(logit_dim, V); v++) {
                float lv = row[v];
                if (std::isfinite(lv)) max_l = std::max(max_l, lv);
            }
            if (!std::isfinite(max_l)) max_l = 0.0f;
            float sum = 0;
            for (int64_t v = 0; v < min_i(logit_dim, V); v++) {
                float lv = row[v];
                sum += std::isfinite(lv) ? std::exp(lv - max_l) : 0.0f;
            }
            // BUGFIX: div-by-zero/log-of-zero guard — fully non-finite rows
            // have sum==0; fall back to uniform so log_prob stays finite.
            if (!(sum > 0.0f) || !std::isfinite(sum))
                sum = (float)max_i(min_i(logit_dim, V), 1);
            float log_prob = (target < logit_dim && std::isfinite(row[target]))
                ? (row[target] - max_l) - std::log(sum)
                : -10.0f;
            // BUGFIX: NaN propagation guard — never let a NaN log_prob
            // poison the running nll average.
            if (std::isfinite(log_prob)) nll += -log_prob;
            nll_count++;
        }
    }

    if (nll_count > 0) {
        r.perplexity = std::exp(nll / nll_count);
        r.total_tokens = nll_count;
    }
    r.task_name = "perplexity";
    return r;
}

EvalResult eval_accuracy(Model* model, const std::vector<int>& tokens,
                         int context_size) {
    EvalResult r;
    if (!model || tokens.empty()) return r;
    int64_t V = model->vocab_size();
    int64_t S = (int64_t)tokens.size();

    for (int64_t i = 0; i + 1 < S; i += context_size) {
        int64_t end = min_i(i + context_size, S - 1);
        int64_t seq_len = end - i;
        if (seq_len < 2) continue;

        Tensor input({1, seq_len});
        Tensor positions({1, seq_len});
        for (int64_t j = 0; j < seq_len; j++) {
            input.data<float>()[j] = (float)tokens[(size_t)(i + j)];
            positions.data<float>()[j] = (float)j;
        }

        Tensor logits = model->forward(input, positions);
        int64_t logit_dim = logits.numel() / seq_len;

        for (int64_t j = 0; j < seq_len - 1; j++) {
            int64_t target = (size_t)(i + j + 1) < tokens.size()
                             ? tokens[(size_t)(i + j + 1)] : 0;
            const float* row = logits.data<float>() + j * logit_dim;
            int best = 0;
            for (int64_t v = 1; v < min_i(logit_dim, V); v++)
                if (row[v] > row[best]) best = static_cast<int>(v);
            r.total++;
            if (best == target) r.correct++;
        }
    }

    if (r.total > 0) r.accuracy = (double)r.correct / r.total;
    r.task_name = "accuracy";
    return r;
}

EvalResult eval_classification(const Tensor& predictions,
                                const std::vector<int>& labels) {
    EvalResult r;
    int64_t n = min_i(predictions.dim(0), (int64_t)labels.size());
    if (n == 0) return r;

    std::vector<int> tp, fp, fn;
    int n_classes = 0;
    for (int l : labels) n_classes = static_cast<int>(max_i(n_classes, l + 1));
    tp.resize((size_t)n_classes, 0);
    fp.resize((size_t)n_classes, 0);
    fn.resize((size_t)n_classes, 0);

    for (int64_t i = 0; i < n; i++) {
        const float* row = predictions.data<float>() + i * predictions.dim(1);
        int pred = 0;
        for (int64_t v = 1; v < predictions.dim(1); v++)
            if (row[v] > row[pred]) pred = static_cast<int>(v);
        int true_label = labels[(size_t)i];
        if (pred == true_label) r.correct++;
        if (true_label < n_classes) {
            if (pred == true_label) tp[(size_t)true_label]++;
            else {
                fp[(size_t)pred]++;
                fn[(size_t)true_label]++;
            }
        }
        r.total++;
    }

    r.accuracy = (double)r.correct / r.total;
    double precision_sum = 0, recall_sum = 0, f1_sum = 0;
    int valid = 0;
    for (int c = 0; c < n_classes; c++) {
        double p = (tp[(size_t)c] + fp[(size_t)c] > 0)
                   ? (double)tp[(size_t)c] / (tp[(size_t)c] + fp[(size_t)c]) : 0;
        double rec = (tp[(size_t)c] + fn[(size_t)c] > 0)
                     ? (double)tp[(size_t)c] / (tp[(size_t)c] + fn[(size_t)c]) : 0;
        if (tp[(size_t)c] > 0) {
            precision_sum += p;
            recall_sum += rec;
            f1_sum += (p + rec > 0) ? 2 * p * rec / (p + rec) : 0;
            valid++;
        }
    }
    if (valid > 0) {
        r.precision = precision_sum / valid;
        r.recall = recall_sum / valid;
        r.f1_score = f1_sum / valid;
    }
    r.task_name = "classification";
    return r;
}

static std::vector<int> get_ngrams(const std::vector<int>& seq, int n) {
    std::vector<int> grams;
    int64_t S = (int64_t)seq.size();
    for (int64_t i = 0; i <= S - n; i++) {
        int hash = 0;
        for (int j = 0; j < n; j++)
            hash = hash * 31 + seq[(size_t)(i + j)];
        grams.push_back(hash);
    }
    return grams;
}

double compute_bleu(const std::vector<int>& candidate,
                    const std::vector<int>& reference, int max_n) {
    if (candidate.empty() || reference.empty()) return 0.0;
    double log_avg = 0.0;
    int clipped = 0, total = 0;
    for (int n = 1; n <= max_n; n++) {
        if ((int)candidate.size() < n || (int)reference.size() < n) continue;
        auto c_grams = get_ngrams(candidate, n);
        auto r_grams = get_ngrams(reference, n);
        std::unordered_map<int, int> r_count;
        for (int g : r_grams) r_count[g]++;
        std::unordered_map<int, int> c_count;
        for (int g : c_grams) c_count[g]++;
        int match = 0, cand_total = 0;
        for (auto& [g, cnt] : c_count) {
            int max_ref = r_count.count(g) ? r_count[g] : 0;
            match += static_cast<int>(min_i(cnt, max_ref));
            cand_total += cnt;
        }
        if (match > 0) {
            // BUGFIX: div-by-zero guard — cand_total>0 whenever match>0
            // (match counts clipped candidate n-grams), but re-check for
            // safety so match/cand_total can never divide by zero.
            if (cand_total > 0) {
                log_avg += std::log((double)match / cand_total);
                clipped += match;
                total += cand_total;
            }
        }
    }
    double bp = (candidate.size() >= reference.size() || candidate.empty())
        ? 1.0 : std::exp(1.0 - (double)reference.size() / candidate.size());
    double avg = (max_n > 0) ? std::exp(log_avg / max_n) : 0.0;
    return bp * avg;
}

double compute_rouge_l(const std::vector<int>& candidate,
                        const std::vector<int>& reference) {
    int64_t m = (int64_t)candidate.size();
    int64_t n = (int64_t)reference.size();
    if (m == 0 || n == 0) return 0.0;
    std::vector<std::vector<int>> dp((size_t)m + 1, std::vector<int>((size_t)n + 1, 0));
    for (int64_t i = 1; i <= m; i++) {
        for (int64_t j = 1; j <= n; j++) {
            if (candidate[(size_t)(i - 1)] == reference[(size_t)(j - 1)])
                dp[(size_t)i][(size_t)j] = dp[(size_t)(i - 1)][(size_t)(j - 1)] + 1;
            else
                dp[(size_t)i][(size_t)j] = static_cast<int>(max_i(dp[(size_t)(i - 1)][(size_t)j],
                                                   dp[(size_t)i][(size_t)(j - 1)]));
        }
    }
    int lcs = dp[(size_t)m][(size_t)n];
    if (lcs == 0) return 0.0;
    double prec = (double)lcs / m;
    double rec = (double)lcs / n;
    return (prec + rec > 0) ? 2.0 * prec * rec / (prec + rec) : 0.0;
}

// ============================================================
// ModelEvaluator
// ============================================================
ModelEvaluator::ModelEvaluator(Model* model) : model_(model) {}

EvalResult ModelEvaluator::evaluate_perplexity(
    const std::vector<int>& tokens, const std::string& name,
    int context, int stride) {
    auto r = eval_perplexity(model_, tokens, context, stride);
    r.task_name = name;
    return r;
}

EvalResult ModelEvaluator::evaluate_accuracy(
    const std::vector<int>& tokens, const std::string& name, int context) {
    auto r = eval_accuracy(model_, tokens, context);
    r.task_name = name;
    return r;
}

EvalResult ModelEvaluator::evaluate_generation(
    int prompt_len, int gen_len, const std::string& name) {
    EvalResult r;
    r.task_name = name;
    if (!model_) return r;
    int64_t V = model_->vocab_size();
    Tensor input({1, prompt_len});
    Tensor positions({1, prompt_len});
    for (int64_t i = 0; i < prompt_len; i++) {
        input.data<float>()[i] = (float)(i % V);
        positions.data<float>()[i] = (float)i;
    }
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < gen_len; i++) {
        Tensor logits = model_->forward(input, positions);
        int64_t logit_dim = logits.numel();
        int best = 0;
        const float* row = logits.data<float>() + (prompt_len - 1) * (logit_dim / prompt_len);
        for (int64_t v = 1; v < logit_dim / prompt_len; v++)
            if (row[v] > row[best]) best = static_cast<int>(v);
        input.data<float>()[(size_t)i] = (float)best;
    }
    auto end = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(end - start).count();
    r.tokens_per_sec = (sec > 0) ? gen_len / sec : 0;
    r.total_tokens = gen_len;
    return r;
}

std::vector<EvalResult> ModelEvaluator::run_all(const std::vector<int>& eval_tokens) {
    std::vector<EvalResult> results;
    results.push_back(evaluate_perplexity(eval_tokens, "perplexity_512", 512, 256));
    results.push_back(evaluate_accuracy(eval_tokens, "accuracy_512", 512));
    results.push_back(evaluate_generation(256, 128, "gen_speed"));
    return results;
}

EvalResult eval_hellaswag(Model* model,
                           const std::vector<std::vector<int>>& contexts,
                           const std::vector<std::vector<int>>& correct_endings,
                           const std::vector<std::vector<int>>& wrong_endings) {
    EvalResult r;
    r.task_name = "hellaswag";
    if (!model || contexts.empty()) return r;
    int64_t V = model->vocab_size();

    for (size_t i = 0; i < contexts.size(); i++) {
        double correct_nll = 0, wrong_nll = 0;
        auto score_ending = [&](const std::vector<int>& ending) -> double {
            std::vector<int> full = contexts[i];
            full.insert(full.end(), ending.begin(), ending.end());
            double nll = 0;
            int64_t S = (int64_t)full.size();
            for (int64_t s = 1; s < S; s += 128) {
                int64_t end = min_i(s + 128, S);
                int64_t len = end - s;
                Tensor input({1, len});
                Tensor pos({1, len});
                for (int64_t j = 0; j < len; j++) {
                    input.data<float>()[j] = (float)full[(size_t)(s + j)];
                    pos.data<float>()[j] = (float)j;
                }
                Tensor logits = model->forward(input, pos);
                int64_t logit_dim = logits.numel() / len;
                for (int64_t j = 0; j < len - 1; j++) {
                    int target = (size_t)(s + j + 1) < full.size() ? full[(size_t)(s + j + 1)] : 0;
                    if (target >= V) target = 0;
                    const float* row = logits.data<float>() + j * logit_dim;
                    // BUGFIX: same max-subtraction + non-finite guards as
                    // eval_perplexity so a NaN row cannot poison the NLL.
                    float max_l = -INFINITY;
                    for (int64_t v = 0; v < min_i(logit_dim, V); v++) {
                        float lv = row[v];
                        if (std::isfinite(lv)) max_l = std::max(max_l, lv);
                    }
                    if (!std::isfinite(max_l)) max_l = 0.0f;
                    float sum = 0;
                    for (int64_t v = 0; v < min_i(logit_dim, V); v++) {
                        float lv = row[v];
                        sum += std::isfinite(lv) ? std::exp(lv - max_l) : 0.0f;
                    }
                    if (!(sum > 0.0f) || !std::isfinite(sum))
                        sum = (float)max_i(min_i(logit_dim, V), 1);
                    float lp = (target < logit_dim && std::isfinite(row[target]))
                        ? (row[target] - max_l) - std::log(sum)
                        : -10.0f;
                    if (std::isfinite(lp)) nll += -lp;
                }
            }
            return nll;
        };

        if (i < correct_endings.size())
            correct_nll = score_ending(correct_endings[i]);
        if (i < wrong_endings.size())
            wrong_nll = score_ending(wrong_endings[i]);

        if (correct_nll < wrong_nll) r.correct++;
        r.total++;
    }

    if (r.total > 0) r.accuracy = (double)r.correct / r.total;
    return r;
}

std::vector<int> load_eval_tokens(const std::string& path, int max_tokens) {
    std::vector<int> tokens;
    std::ifstream f(path);
    if (!f.is_open()) return tokens;
    std::string line;
    while (std::getline(f, line)) {
        line.erase(std::remove_if(line.begin(), line.end(),
                   [](char c) { return !std::isdigit(c) && c != '-'; }), line.end());
        if (line.empty()) continue;
        // BUGFIX: unchecked-stoi guard — a malformed line (huge digits,
        // "-" alone, overflow) must not throw out of eval; skip the line.
        try {
            tokens.push_back(std::stoi(line));
        } catch (...) {
            continue;
        }
        if (max_tokens > 0 && (int)tokens.size() >= max_tokens) break;
    }
    return tokens;
}

} // namespace quant