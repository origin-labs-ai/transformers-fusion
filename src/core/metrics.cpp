#define NOMINMAX
#include "quant/metrics.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <limits>

namespace quant {

// ============================================================
// Accuracy
// ============================================================
Accuracy::Accuracy(int top_k) : top_k_(std::max(1, top_k)) {}

bool Accuracy::topk_match(const float* scores, int n_classes, int target) const {
    if (target < 0 || target >= n_classes) return false;
    if (top_k_ == 1) {
        int best = 0;
        for (int i = 1; i < n_classes; i++)
            if (scores[i] > scores[best]) best = i;
        return best == target;
    }
    float target_score = scores[target];
    int count_higher = 0;
    for (int i = 0; i < n_classes; i++) {
        if (scores[i] > target_score) count_higher++;
    }
    return count_higher < top_k_;
}

double Accuracy::update(const Tensor& predictions, const Tensor& targets) {
    int64_t batch = predictions.dim(0);
    int64_t n_classes = predictions.dim(1);
    const float* pred_data = predictions.data<float>();
    const float* tgt_data = targets.data<float>();
    for (int64_t i = 0; i < batch; i++) {
        int target = (int)tgt_data[i];
        if (topk_match(pred_data + i * n_classes, (int)n_classes, target))
            correct_++;
        total_++;
    }
    return value();
}

double Accuracy::value() const {
    return total_ > 0 ? (double)correct_ / total_ : 0.0;
}

void Accuracy::reset() {
    correct_ = 0;
    total_ = 0;
}

// ============================================================
// Perplexity
// ============================================================
double Perplexity::update(const Tensor& logits, const Tensor& targets) {
    int64_t B = logits.dim(0), S = logits.dim(1), V = logits.dim(2);
    const float* logit_data = logits.data<float>();
    const float* tgt_data = targets.data<float>();
    for (int64_t i = 0; i < B * S; i++) {
        const float* row = logit_data + i * V;
        int target = (int)tgt_data[i];
        if (target < 0 || target >= (int)V) continue;
        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t v = 0; v < V; v++) {
            if (row[v] > max_val) max_val = row[v];
        }
        double sum_exp = 0.0;
        for (int64_t v = 0; v < V; v++)
            sum_exp += (double)std::exp(row[v] - max_val);
        double log_prob = (double)(row[target] - max_val) - std::log(sum_exp + 1e-10);
        nll_ += -log_prob;
        count_++;
    }
    return value();
}

double Perplexity::value() const {
    if (count_ <= 0) return 0.0;
    return std::exp(nll_ / (double)count_);
}

void Perplexity::reset() {
    nll_ = 0.0;
    count_ = 0;
}

// ============================================================
// F1Score
// ============================================================
F1Score::F1Score(int num_classes) : num_classes_(num_classes) {
    tp_.resize((size_t)num_classes, 0);
    fp_.resize((size_t)num_classes, 0);
    fn_.resize((size_t)num_classes, 0);
}

void F1Score::update(const Tensor& predictions, const Tensor& targets) {
    int64_t batch = predictions.dim(0);
    int64_t n_classes = predictions.dim(1);
    const float* pred_data = predictions.data<float>();
    const float* tgt_data = targets.data<float>();
    if (tp_.empty()) {
        num_classes_ = (int)n_classes;
        tp_.assign((size_t)n_classes, 0);
        fp_.assign((size_t)n_classes, 0);
        fn_.assign((size_t)n_classes, 0);
    }
    for (int64_t i = 0; i < batch; i++) {
        int pred = 0;
        const float* row = pred_data + i * n_classes;
        for (int64_t v = 1; v < n_classes; v++)
            if (row[v] > row[pred]) pred = static_cast<int>(v);
        int target = (int)tgt_data[i];
        if (target >= 0 && target < num_classes_) {
            if (pred == target) {
                tp_[(size_t)target]++;
            } else {
                fn_[(size_t)target]++;
                if (pred >= 0 && pred < num_classes_)
                    fp_[(size_t)pred]++;
            }
        }
        total_samples_++;
    }
}

double F1Score::precision(int class_idx) const {
    if (class_idx >= 0) {
        int64_t denom = tp_[(size_t)class_idx] + fp_[(size_t)class_idx];
        return denom > 0 ? (double)tp_[(size_t)class_idx] / (double)denom : 0.0;
    }
    double sum = 0.0;
    for (int c = 0; c < num_classes_; c++) {
        int64_t denom = tp_[(size_t)c] + fp_[(size_t)c];
        sum += denom > 0 ? (double)tp_[(size_t)c] / (double)denom : 0.0;
    }
    return num_classes_ > 0 ? sum / num_classes_ : 0.0;
}

double F1Score::recall(int class_idx) const {
    if (class_idx >= 0) {
        int64_t denom = tp_[(size_t)class_idx] + fn_[(size_t)class_idx];
        return denom > 0 ? (double)tp_[(size_t)class_idx] / (double)denom : 0.0;
    }
    double sum = 0.0;
    for (int c = 0; c < num_classes_; c++) {
        int64_t denom = tp_[(size_t)c] + fn_[(size_t)c];
        sum += denom > 0 ? (double)tp_[(size_t)c] / (double)denom : 0.0;
    }
    return num_classes_ > 0 ? sum / num_classes_ : 0.0;
}

double F1Score::f1(int class_idx) const {
    double p = precision(class_idx);
    double r = recall(class_idx);
    return (p + r > 0) ? 2.0 * p * r / (p + r) : 0.0;
}

double F1Score::macro_f1() const {
    double sum = 0.0;
    int valid = 0;
    for (int c = 0; c < num_classes_; c++) {
        double p = precision(c);
        double r = recall(c);
        if (p + r > 0) {
            sum += 2.0 * p * r / (p + r);
            valid++;
        }
    }
    return valid > 0 ? sum / valid : 0.0;
}

double F1Score::weighted_f1() const {
    if (total_samples_ <= 0) return 0.0;
    double sum = 0.0;
    for (int c = 0; c < num_classes_; c++) {
        int64_t support = tp_[(size_t)c] + fn_[(size_t)c];
        if (support > 0) {
            double p = precision(c);
            double r = recall(c);
            double f = (p + r > 0) ? 2.0 * p * r / (p + r) : 0.0;
            sum += f * (double)support;
        }
    }
    return sum / (double)total_samples_;
}

void F1Score::reset() {
    tp_.assign((size_t)num_classes_, 0);
    fp_.assign((size_t)num_classes_, 0);
    fn_.assign((size_t)num_classes_, 0);
    total_samples_ = 0;
}

// ============================================================
// BLEUScore
// ============================================================
BLEUScore::BLEUScore(int max_n) : max_n_(std::max(1, std::min(10, max_n))) {}

std::vector<int> BLEUScore::get_ngrams(const std::vector<int>& seq, int n) {
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

std::unordered_map<int, int> BLEUScore::count_ngrams(
    const std::vector<int>& seq, int n) {
    auto grams = get_ngrams(seq, n);
    std::unordered_map<int, int> counts;
    for (int g : grams) counts[g]++;
    return counts;
}

double BLEUScore::compute(const std::vector<int>& candidate,
                          const std::vector<int>& reference) {
    if (candidate.empty() || reference.empty()) return 0.0;
    double log_avg = 0.0;
    int valid_n = 0;
    for (int n = 1; n <= max_n_; n++) {
        if ((int)candidate.size() < n || (int)reference.size() < n) continue;
        auto c_counts = count_ngrams(candidate, n);

        auto r_counts = count_ngrams(reference, n);
        int match = 0, total = 0;
        for (auto& [g, cnt] : c_counts) {
            auto it = r_counts.find(g);
            int max_ref = (it != r_counts.end()) ? it->second : 0;
            match += std::min(cnt, max_ref);
            total += cnt;
        }
        if (total > 0 && match > 0) {
            log_avg += std::log((double)match / (double)total);
            valid_n++;
        }
    }
    if (valid_n == 0) return 0.0;
    double bp = (candidate.size() >= reference.size())
        ? 1.0 : std::exp(1.0 - (double)reference.size() / (double)candidate.size());
    double avg = std::exp(log_avg / (double)valid_n);
    return bp * avg;
}

double BLEUScore::compute(const Tensor& candidate, const Tensor& reference) {
    int64_t n = candidate.numel();
    std::vector<int> cand_vec((size_t)n);
    std::vector<int> ref_vec((size_t)n);
    const float* cd = candidate.data<float>();
    const float* rd = reference.data<float>();
    for (size_t i = 0; i < (size_t)n; i++) {
        cand_vec[i] = (int)cd[i];
        ref_vec[i] = (int)rd[i];
    }
    return compute(cand_vec, ref_vec);
}

// ============================================================
// ROUGELScore
// ============================================================
double ROUGELScore::compute(const std::vector<int>& candidate,
                            const std::vector<int>& reference) {
    int64_t m = (int64_t)candidate.size();
    int64_t n = (int64_t)reference.size();
    if (m == 0 || n == 0) return 0.0;
    std::vector<std::vector<int>> dp((size_t)m + 1,
                                     std::vector<int>((size_t)n + 1, 0));
    for (int64_t i = 1; i <= m; i++) {
        for (int64_t j = 1; j <= n; j++) {
            if (candidate[(size_t)(i - 1)] == reference[(size_t)(j - 1)])
                dp[(size_t)i][(size_t)j] = dp[(size_t)(i - 1)][(size_t)(j - 1)] + 1;
            else
                dp[(size_t)i][(size_t)j] = std::max(
                    dp[(size_t)(i - 1)][(size_t)j],
                    dp[(size_t)i][(size_t)(j - 1)]);
        }
    }
    int lcs = dp[(size_t)m][(size_t)n];
    if (lcs == 0) return 0.0;
    double prec = (double)lcs / (double)m;
    double rec = (double)lcs / (double)n;
    return (prec + rec > 0) ? 2.0 * prec * rec / (prec + rec) : 0.0;
}

double ROUGELScore::compute(const Tensor& candidate, const Tensor& reference) {
    int64_t cn = candidate.numel();
    int64_t rn = reference.numel();
    std::vector<int> cand_vec((size_t)cn);
    std::vector<int> ref_vec((size_t)rn);
    const float* cd = candidate.data<float>();
    const float* rd = reference.data<float>();
    for (size_t i = 0; i < (size_t)cn; i++) cand_vec[i] = (int)cd[i];
    for (size_t i = 0; i < (size_t)rn; i++) ref_vec[i] = (int)rd[i];
    return compute(cand_vec, ref_vec);
}

// ============================================================
// MeanAveragePrecision
// ============================================================
double MeanAveragePrecision::compute(const Tensor& scores,
                                     const Tensor& labels) {
    int64_t num_q = scores.dim(0);
    int64_t num_i = scores.dim(1);
    const float* score_data = scores.data<float>();
    const float* label_data = labels.data<float>();
    double total_ap = 0.0;
    int count = 0;
    for (int64_t q = 0; q < num_q; q++) {
        const float* q_scores = score_data + q * num_i;
        const float* q_labels = label_data + q * num_i;
        std::vector<int> idx((size_t)num_i);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [q_scores](int a, int b) {
            return q_scores[a] > q_scores[b];
        });
        double ap = 0.0;
        int n_rel = 0;
        for (int64_t k = 0; k < num_i; k++) {
            if (q_labels[idx[(size_t)k]] > 0.5f) {
                n_rel++;
                ap += (double)n_rel / (double)(k + 1);
            }
        }
        if (n_rel > 0) {
            total_ap += ap / (double)n_rel;
            count++;
        }
    }
    double result = count > 0 ? total_ap / (double)count : 0.0;
    ap_scores_.push_back(result);
    return result;
}

double MeanAveragePrecision::value() const {
    if (ap_scores_.empty()) return 0.0;
    double sum = 0.0;
    for (double ap : ap_scores_) sum += ap;
    return sum / (double)ap_scores_.size();
}

void MeanAveragePrecision::reset() {
    ap_scores_.clear();
}

// ============================================================
// ConfusionMatrix
// ============================================================
ConfusionMatrix::ConfusionMatrix(int num_classes)
    : num_classes_(num_classes) {
    matrix_.assign((size_t)num_classes * (size_t)num_classes, 0);
}

void ConfusionMatrix::update(const Tensor& predictions,
                              const Tensor& targets) {
    int64_t batch = predictions.numel();
    const float* pred_data = predictions.data<float>();
    const float* tgt_data = targets.data<float>();
    for (int64_t i = 0; i < batch; i++) {
        int pred = (int)pred_data[i];
        int tgt = (int)tgt_data[i];
        if (pred >= 0 && pred < num_classes_ &&
            tgt >= 0 && tgt < num_classes_) {
            matrix_[(size_t)(tgt * num_classes_ + pred)]++;
            total_++;
        }
    }
}

void ConfusionMatrix::update(const Tensor& logits, const Tensor& targets,
                              bool from_logits) {
    if (!from_logits) {
        update(logits, targets);
        return;
    }
    int64_t batch = logits.dim(0);
    int64_t n_classes = logits.dim(1);
    const float* pred_data = logits.data<float>();
    const float* tgt_data = targets.data<float>();
    for (int64_t i = 0; i < batch; i++) {
        int pred = 0;
        const float* row = pred_data + i * n_classes;
        for (int64_t v = 1; v < n_classes; v++)
            if (row[v] > row[pred]) pred = static_cast<int>(v);
        int tgt = (int)tgt_data[i];
        if (pred >= 0 && pred < num_classes_ &&
            tgt >= 0 && tgt < num_classes_) {
            matrix_[(size_t)(tgt * num_classes_ + pred)]++;
            total_++;
        }
    }
}

int64_t ConfusionMatrix::at(int pred_class, int true_class) const {
    if (pred_class < 0 || pred_class >= num_classes_ ||
        true_class < 0 || true_class >= num_classes_)
        return 0;
    return matrix_[(size_t)(true_class * num_classes_ + pred_class)];
}

int64_t ConfusionMatrix::true_positive(int class_idx) const {
    return at(class_idx, class_idx);
}

int64_t ConfusionMatrix::false_positive(int class_idx) const {
    int64_t sum = 0;
    for (int t = 0; t < num_classes_; t++) {
        if (t != class_idx) sum += at(class_idx, t);
    }
    return sum;
}

int64_t ConfusionMatrix::false_negative(int class_idx) const {
    int64_t sum = 0;
    for (int p = 0; p < num_classes_; p++) {
        if (p != class_idx) sum += at(p, class_idx);
    }
    return sum;
}

int64_t ConfusionMatrix::true_negative(int class_idx) const {
    int64_t sum = 0;
    for (int p = 0; p < num_classes_; p++) {
        for (int t = 0; t < num_classes_; t++) {
            if (p != class_idx && t != class_idx)
                sum += at(p, t);
        }
    }
    return sum;
}

void ConfusionMatrix::reset() {
    matrix_.assign((size_t)num_classes_ * (size_t)num_classes_, 0);
    total_ = 0;
}

} // namespace quant
