#pragma once
#include "quant/tensor.h"
#include "quant/tokenizer.h"
#include <string>
#include <vector>
#include <random>
#include <map>
#include <set>
#include <memory>
#include <functional>

namespace quant {

class DataGenerator {
public:
    DataGenerator(unsigned int seed = 42);

    std::vector<std::string> generate_random_text(size_t num_samples, size_t avg_length,
                                                  size_t vocab_size = 100);
    std::vector<std::string> generate_arithmetic(size_t num_samples,
                                                 const std::string& ops = "+-");
    std::vector<std::string> generate_pattern_recognition(size_t num_samples,
                                                           size_t pattern_length = 4);
    std::vector<std::string> generate_copy_tasks(size_t num_samples,
                                                  size_t max_length = 50);
    std::vector<std::string> generate_reverse_tasks(size_t num_samples,
                                                     size_t max_length = 30);
    std::vector<std::string> generate_sorting_tasks(size_t num_samples,
                                                     size_t max_items = 10);
    std::vector<std::string> generate_ngram_text(size_t num_samples, size_t avg_length,
                                                  int ngram_order = 3);
    std::vector<std::string> generate_masked_prediction(size_t num_samples,
                                                         size_t sentence_length = 10);
    std::vector<std::string> generate_next_token_prediction(size_t num_samples,
                                                             size_t seq_length = 20);
    std::vector<std::string> generate_token_recovery(size_t num_samples,
                                                      size_t seq_length = 15);

    std::vector<std::string> generate_mixed(size_t num_samples,
                                             std::vector<double> task_weights);

private:
    std::mt19937 rng_;
    std::uniform_int_distribution<int> letter_dist_;
    std::uniform_int_distribution<int> digit_dist_;

    std::string random_string(size_t length, bool include_digits = true);
    std::string random_word();
    std::string random_sentence(size_t words);
    std::vector<int> random_sequence(size_t length, int max_val = 100);
    std::string format_arithmetic_problem(int a, int b, char op);
    int compute_arithmetic(int a, int b, char op);
    bool passes_task_weight(double cumulative_weight);
};

class CurriculumGenerator {
public:
    struct Stage {
        std::string name;
        int difficulty;
        int max_samples;
        double accuracy_threshold;
        std::function<std::vector<std::string>(int)> generator_fn;
    };

    CurriculumGenerator(unsigned int seed = 42);

    void add_stage(const Stage& stage);
    std::vector<std::string> generate_next_batch(int batch_size);
    void report_accuracy(double accuracy);
    std::string current_stage_name() const;
    int current_difficulty() const;
    bool is_complete() const;

    void set_progress_fn(std::function<void(const std::string&, int, double)> fn);

private:
    std::mt19937 rng_;
    std::vector<Stage> stages_;
    int current_stage_idx_ = 0;
    int stage_sample_count_ = 0;
    double running_accuracy_ = 0.0;
    int accuracy_samples_ = 0;
    // BUGFIX (bug census): set_progress_fn silently discarded its callback
    // (training progress invisible). Stored + invoked on stage transitions.
    std::function<void(const std::string&, int, double)> progress_fn_;

    bool should_promote() const;
};

class TokenizerEvaluator {
public:
    TokenizerEvaluator(Tokenizer* tokenizer);

    float compression_ratio(const std::string& text) const;
    float re_encoding_accuracy(const std::string& text) const;
    float vocabulary_coverage(const std::string& corpus) const;
    float average_token_length(const std::string& corpus) const;
    float vocabulary_utilization(const std::string& corpus) const;
    float reconstruction_fidelity(const std::string& text) const;
    float character_error_rate(const std::string& original, const std::string& reconstructed) const;
    float byte_efficiency(const std::string& text) const;

    struct EvaluationReport {
        float compression_ratio;
        float re_encoding_accuracy;
        float vocabulary_coverage;
        float average_token_length;
        float vocabulary_utilization;
        float reconstruction_fidelity;
        float byte_efficiency;
        float character_error_rate;
        int total_tokens;
        int unique_tokens;
        int oov_count;
        size_t corpus_size_bytes;
    };

    EvaluationReport evaluate(const std::vector<std::string>& corpus) const;
    std::string report_string(const EvaluationReport& report) const;

    static void compare_tokenizers(const std::vector<Tokenizer*>& tokenizers,
                                    const std::vector<std::string>& corpus,
                                    std::vector<EvaluationReport>& reports);

private:
    Tokenizer* tokenizer_;
    std::vector<int> encode_with_check(const std::string& text) const;
};

} // namespace quant
