#pragma once
#include "quant/eval.h"
#include "quant/trainer.h"
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace quant {
namespace dense_engine {

// Eval harness over ModelEvaluator + DataLoader with real scoring.
// Perplexity/accuracy/BLEU/ROUGE-L/hellaswag, all live tensor+logit math.
struct HarnessOut {
    EvalResult perplexity;
    EvalResult accuracy;
    double bleu = 0.0;
    double rouge_l = 0.0;
    int64_t tokens = 0;
    double tok_per_sec = 0.0;
};

class EvalHarness {
public:
    EvalHarness(Model* model, Tokenizer* tok);
    HarnessOut run_tokens(const std::vector<int>& tokens,
                          const std::string& name = "eval");
    HarnessOut run_loader(DataLoader& dl, int64_t max_batches = 20);
    EvalResult hellaswag(const std::vector<std::vector<int>>& contexts,
                         const std::vector<std::vector<int>>& good,
                         const std::vector<std::vector<int>>& bad);
    double bleu(const std::vector<int>& cand, const std::vector<int>& ref, int max_n = 4);
    double rouge_l(const std::vector<int>& cand, const std::vector<int>& ref);
private:
    Model* model_ = nullptr;
    Tokenizer* tok_ = nullptr;
    ModelEvaluator evaluator_;
    int64_t t0_us_ = 0;
    static int64_t now_us();
};

// N-gram overlap scorer with real counting (backs BLEU/ROUGE-L paths).
struct NgramScore {
    static double precision(const std::vector<int>& cand, const std::vector<int>& ref, int n);
    static int64_t lcs_len(const std::vector<int>& a, const std::vector<int>& b);
};

} // namespace dense_engine
} // namespace quant
