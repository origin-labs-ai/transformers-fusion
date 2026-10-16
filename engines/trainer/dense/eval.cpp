#include "eval.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace quant {
namespace dense_engine {

int64_t EvalHarness::now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

EvalHarness::EvalHarness(Model* model, Tokenizer* tok)
    : model_(model), tok_(tok), evaluator_(model) {
    if (!model_ || !tok_) throw std::runtime_error("[eval] needs Model*+Tokenizer*");
    t0_us_ = now_us();
}

HarnessOut EvalHarness::run_tokens(const std::vector<int>& tokens, const std::string& name) {
    HarnessOut o;
    o.perplexity = evaluator_.evaluate_perplexity(tokens, name + "_ppl");
    o.accuracy = evaluator_.evaluate_accuracy(tokens, name + "_acc");
    o.tokens = (int64_t)tokens.size();
    int64_t dt = now_us() - t0_us_;
    if (dt > 0) o.tok_per_sec = (double)o.tokens * 1e6 / (double)dt;
    o.perplexity.tokens_per_sec = o.tok_per_sec;
    return o;
}

HarnessOut EvalHarness::run_loader(DataLoader& dl, int64_t max_batches) {
    dl.reset();
    Tensor in, lb;
    std::vector<int> all;
    int64_t n = 0;
    while (n < max_batches && dl.next_batch(in, lb)) {
        const int* d = in.data<int>();
        for (int64_t i = 0; i < in.numel(); i++) all.push_back(d[i]);
        n++;
    }
    if (all.empty()) throw std::runtime_error("[eval] empty eval loader");
    return run_tokens(all, "loader");
}

EvalResult EvalHarness::hellaswag(const std::vector<std::vector<int>>& contexts,
                                  const std::vector<std::vector<int>>& good,
                                  const std::vector<std::vector<int>>& bad) {
    return eval_hellaswag(model_, contexts, good, bad);
}

double EvalHarness::bleu(const std::vector<int>& cand, const std::vector<int>& ref, int max_n) {
    return compute_bleu(cand, ref, max_n);
}

double EvalHarness::rouge_l(const std::vector<int>& cand, const std::vector<int>& ref) {
    return compute_rouge_l(cand, ref);
}

double NgramScore::precision(const std::vector<int>& cand, const std::vector<int>& ref, int n) {
    if (n <= 0 || cand.size() < (size_t)n || ref.empty()) return 0.0;
    std::map<std::vector<int>, int64_t> rc;
    for (size_t i = 0; i + n <= ref.size(); i++) {
        std::vector<int> g(ref.begin() + i, ref.begin() + i + n);
        rc[g]++;
    }
    int64_t hit = 0, total = 0;
    for (size_t i = 0; i + n <= cand.size(); i++) {
        std::vector<int> g(cand.begin() + i, cand.begin() + i + n);
        auto it = rc.find(g);
        if (it != rc.end() && it->second > 0) { hit++; it->second--; }
        total++;
    }
    return total > 0 ? (double)hit / (double)total : 0.0;
}

int64_t NgramScore::lcs_len(const std::vector<int>& a, const std::vector<int>& b) {
    size_t n = a.size(), m = b.size();
    if (n == 0 || m == 0) return 0;
    std::vector<int> prev(m + 1, 0), cur(m + 1, 0);
    for (size_t i = 1; i <= n; i++) {
        for (size_t j = 1; j <= m; j++) {
            if (a[i - 1] == b[j - 1]) cur[j] = prev[j - 1] + 1;
            else cur[j] = std::max(prev[j], cur[j - 1]);
        }
        prev.swap(cur);
    }
    return prev[m];
}

} // namespace dense_engine
} // namespace quant
