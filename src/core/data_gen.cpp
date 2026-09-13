#include "quant/data_gen.h"
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <unordered_set>
#include <cstdio>

namespace quant {

// ==================== DataGenerator ====================

DataGenerator::DataGenerator(unsigned int seed)
    : rng_(seed), letter_dist_(0, 25), digit_dist_(0, 9) {}

std::string DataGenerator::random_string(size_t length, bool include_digits) {
    std::string result;
    for (size_t i = 0; i < length; i++) {
        if (include_digits && rng_() % 4 == 0) {
            result += (char)('0' + digit_dist_(rng_));
        } else {
            result += (char)('a' + letter_dist_(rng_));
        }
    }
    return result;
}

std::string DataGenerator::random_word() {
    size_t len = 2 + (rng_() % 10);
    std::string word;
    word += (char)('a' + letter_dist_(rng_));
    for (size_t i = 1; i < len; i++) {
        word += (char)('a' + letter_dist_(rng_));
    }
    return word;
}

std::string DataGenerator::random_sentence(size_t words) {
    std::string sentence;
    for (size_t i = 0; i < words; i++) {
        if (i > 0) sentence += " ";
        sentence += random_word();
    }
    if (!sentence.empty()) {
        sentence[0] = (char)std::toupper(sentence[0]);
        sentence += ".";
    }
    return sentence;
}

std::vector<int> DataGenerator::random_sequence(size_t length, int max_val) {
    std::vector<int> seq(length);
    for (size_t i = 0; i < length; i++) {
        seq[i] = (int)(rng_() % (max_val + 1));
    }
    return seq;
}

std::string DataGenerator::format_arithmetic_problem(int a, int b, char op) {
    std::ostringstream oss;
    oss << a << " " << op << " " << b << " = ";
    return oss.str();
}

int DataGenerator::compute_arithmetic(int a, int b, char op) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': return (b == 0) ? 0 : a / b;
        default: return 0;
    }
}

bool DataGenerator::passes_task_weight(double cumulative_weight) {
    return (double)(rng_() % 10000) / 10000.0 < cumulative_weight;
}

std::vector<std::string> DataGenerator::generate_random_text(size_t num_samples, size_t avg_length, size_t vocab_size) {
    // BUGFIX (bug census): vocab_size was discarded — output alphabet is
    // a-z(+0-9), i.e. 26/36 symbols, regardless of the parameter. Clamp the
    // alphabet to min(vocab_size, 36) so callers get what they asked for
    // (vocab_size==0 keeps legacy full-alphabet behavior).
    std::vector<std::string> samples;
    for (size_t i = 0; i < num_samples; i++) {
        size_t len = avg_length + (rng_() % (avg_length / 2 + 1));
        std::string s = random_string(len);
        if (vocab_size > 0 && vocab_size < 36) {
            for (char& c : s) {
                if (c >= '0' && c <= '9') {
                    if ((size_t)(c - '0') >= vocab_size) c = (char)('0' + (c - '0') % vocab_size);
                } else if (c >= 'a' && c <= 'z') {
                    size_t alpha = vocab_size > 10 ? vocab_size - 10 : 0;
                    if ((size_t)(c - 'a') >= alpha && alpha > 0) c = (char)('a' + (c - 'a') % alpha);
                }
            }
        }
        samples.push_back(s);
    }
    return samples;
}

std::vector<std::string> DataGenerator::generate_arithmetic(size_t num_samples, const std::string& ops) {
    std::vector<std::string> samples;
    for (size_t i = 0; i < num_samples; i++) {
        char op = ops[rng_() % ops.size()];
        int a, b;
        if (op == '+') {
            a = (int)(rng_() % 1000);
            b = (int)(rng_() % 1000);
        } else if (op == '-') {
            a = (int)(rng_() % 1000);
            b = (int)(rng_() % (a + 1));
        } else if (op == '*') {
            a = (int)(rng_() % 100);
            b = (int)(rng_() % 100);
        } else if (op == '/') {
            b = 1 + (int)(rng_() % 100);
            a = b * (int)(rng_() % 100);
        } else {
            a = (int)(rng_() % 100);
            b = (int)(rng_() % 100);
        }

        int result = compute_arithmetic(a, b, op);
        std::ostringstream oss;
        oss << format_arithmetic_problem(a, b, op) << result;
        samples.push_back(oss.str());
    }
    return samples;
}

std::vector<std::string> DataGenerator::generate_pattern_recognition(size_t num_samples, size_t pattern_length) {
    std::vector<std::string> samples;
    for (size_t i = 0; i < num_samples; i++) {
        std::string pattern;
        bool use_ab = (rng_() % 2 == 0);
        if (use_ab) {
            for (size_t j = 0; j < pattern_length; j++) {
                pattern += (char)('a' + (rng_() % 2));
            }
        } else {
            for (size_t j = 0; j < pattern_length; j++) {
                pattern += (char)('0' + (rng_() % 2));
            }
        }

        size_t repeats = 2 + (rng_() % 4);
        std::string sequence;
        for (size_t r = 0; r < repeats; r++) {
            sequence += pattern;
        }

        size_t noise_len = rng_() % 5;
        for (size_t j = 0; j < noise_len; j++) {
            if (rng_() % 2 == 0) {
                sequence += (char)('a' + (rng_() % 26));
            } else {
                sequence += (char)('0' + (rng_() % 10));
            }
        }

        std::string continuation;
        for (size_t r = 0; r < repeats; r++) {
            continuation += pattern;
        }

        std::ostringstream oss;
        oss << "Pattern: " << sequence << " Continue: " << continuation;
        samples.push_back(oss.str());
    }
    return samples;
}

std::vector<std::string> DataGenerator::generate_copy_tasks(size_t num_samples, size_t max_length) {
    std::vector<std::string> samples;
    for (size_t i = 0; i < num_samples; i++) {
        size_t len = 1 + (rng_() % max_length);
        std::string source;
        for (size_t j = 0; j < len; j++) {
            source += (char)('a' + letter_dist_(rng_));
        }
        std::ostringstream oss;
        oss << "Copy: " << source << " -> " << source;
        samples.push_back(oss.str());
    }
    return samples;
}

std::vector<std::string> DataGenerator::generate_reverse_tasks(size_t num_samples, size_t max_length) {
    std::vector<std::string> samples;
    for (size_t i = 0; i < num_samples; i++) {
        size_t len = 1 + (rng_() % max_length);
        std::string source;
        for (size_t j = 0; j < len; j++) {
            source += (char)('a' + letter_dist_(rng_));
        }
        std::string reversed_src(source.rbegin(), source.rend());
        std::ostringstream oss;
        oss << "Reverse: " << source << " -> " << reversed_src;
        samples.push_back(oss.str());
    }
    return samples;
}

std::vector<std::string> DataGenerator::generate_sorting_tasks(size_t num_samples, size_t max_items) {
    std::vector<std::string> samples;
    for (size_t i = 0; i < num_samples; i++) {
        size_t count = 2 + (rng_() % max_items);
        std::vector<int> numbers;
        for (size_t j = 0; j < count; j++) {
            numbers.push_back((int)(rng_() % 100));
        }
        std::vector<int> sorted = numbers;
        std::sort(sorted.begin(), sorted.end());

        std::ostringstream oss;
        oss << "Sort: ";
        for (size_t j = 0; j < numbers.size(); j++) {
            if (j > 0) oss << ",";
            oss << numbers[j];
        }
        oss << " -> ";
        for (size_t j = 0; j < sorted.size(); j++) {
            if (j > 0) oss << ",";
            oss << sorted[j];
        }
        samples.push_back(oss.str());
    }
    return samples;
}

std::vector<std::string> DataGenerator::generate_ngram_text(size_t num_samples, size_t avg_length, int ngram_order) {
    std::vector<std::string> samples;
    std::vector<std::string> vocab;
    for (int i = 0; i < 50; i++) vocab.push_back(random_word());

    std::map<std::vector<std::string>, std::vector<std::string>> ngram_model;
    std::vector<std::string> seed_ngram;
    for (int i = 0; i < ngram_order; i++) {
        seed_ngram.push_back(vocab[rng_() % vocab.size()]);
    }

    for (size_t i = 0; i < 500; i++) {
        std::vector<std::string> key;
        for (int j = 0; j < ngram_order; j++) {
            key.push_back(vocab[rng_() % vocab.size()]);
        }
        std::string next = vocab[rng_() % vocab.size()];
        ngram_model[key].push_back(next);
    }

    for (size_t i = 0; i < num_samples; i++) {
        std::vector<std::string> current = seed_ngram;
        std::string text;
        for (size_t j = 0; j < ngram_order; j++) {
            if (j > 0) text += " ";
            text += current[j];
        }

        for (size_t j = ngram_order; j < avg_length; j++) {
            std::vector<std::string> key;
            for (int k = (int)j - ngram_order; k < (int)j; k++) {
                key.push_back(current[k]);
            }
            auto it = ngram_model.find(key);
            std::string next_word;
            if (it != ngram_model.end() && !it->second.empty()) {
                next_word = it->second[rng_() % it->second.size()];
            } else {
                next_word = vocab[rng_() % vocab.size()];
            }
            text += " " + next_word;
            current.push_back(next_word);
        }
        samples.push_back(text);
    }
    return samples;
}

std::vector<std::string> DataGenerator::generate_masked_prediction(size_t num_samples, size_t sentence_length) {
    std::vector<std::string> samples;
    for (size_t i = 0; i < num_samples; i++) {
        size_t word_count = 3 + (rng_() % sentence_length);
        std::vector<std::string> sentence_words;
        for (size_t j = 0; j < word_count; j++) {
            sentence_words.push_back(random_word());
        }

        size_t mask_pos = rng_() % word_count;
        std::string target = sentence_words[mask_pos];
        sentence_words[mask_pos] = "[MASK]";

        std::ostringstream oss;
        for (size_t j = 0; j < word_count; j++) {
            if (j > 0) oss << " ";
            oss << sentence_words[j];
        }
        oss << " -> " << target;
        samples.push_back(oss.str());
    }
    return samples;
}

std::vector<std::string> DataGenerator::generate_next_token_prediction(size_t num_samples, size_t seq_length) {
    std::vector<std::string> samples;
    for (size_t i = 0; i < num_samples; i++) {
        std::vector<int> seq = random_sequence(seq_length + 1, 20);
        std::ostringstream oss;
        for (size_t j = 0; j < seq_length; j++) {
            if (j > 0) oss << " ";
            oss << seq[j];
        }
        oss << " -> " << seq[seq_length];
        samples.push_back(oss.str());
    }
    return samples;
}

std::vector<std::string> DataGenerator::generate_token_recovery(size_t num_samples, size_t seq_length) {
    std::vector<std::string> samples;
    for (size_t i = 0; i < num_samples; i++) {
        std::string original = random_string(seq_length);
        size_t corrupt_pos = rng_() % seq_length;
        std::string corrupted = original;
        corrupted[corrupt_pos] = (char)('a' + letter_dist_(rng_));

        std::ostringstream oss;
        oss << corrupted << " -> " << original;
        samples.push_back(oss.str());
    }
    return samples;
}

std::vector<std::string> DataGenerator::generate_mixed(size_t num_samples,
                                                        std::vector<double> task_weights) {
    std::vector<std::string> samples;
    if (task_weights.empty()) {
        task_weights = {0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1};
    }

    double total = std::accumulate(task_weights.begin(), task_weights.end(), 0.0);
    for (auto& w : task_weights) w /= total;

    std::vector<std::function<std::vector<std::string>(size_t)>> generators;
    generators.push_back([this](size_t n) { return generate_random_text(n, 30, 100); });
    generators.push_back([this](size_t n) { return generate_arithmetic(n, "+-"); });
    generators.push_back([this](size_t n) { return generate_pattern_recognition(n, 4); });
    generators.push_back([this](size_t n) { return generate_copy_tasks(n, 30); });
    generators.push_back([this](size_t n) { return generate_reverse_tasks(n, 20); });
    generators.push_back([this](size_t n) { return generate_sorting_tasks(n, 8); });
    generators.push_back([this](size_t n) { return generate_ngram_text(n, 20, 3); });

    for (size_t i = 0; i < num_samples; i++) {
        double r = (double)(rng_() % 10000) / 10000.0;
        double cum = 0.0;
        for (size_t j = 0; j < task_weights.size(); j++) {
            cum += task_weights[j];
            if (r <= cum && j < generators.size()) {
                auto gen_samples = generators[j](1);
                if (!gen_samples.empty()) samples.push_back(gen_samples[0]);
                break;
            }
        }
    }

    return samples;
}

// ==================== CurriculumGenerator ====================

CurriculumGenerator::CurriculumGenerator(unsigned int seed)
    : rng_(seed), running_accuracy_(0.0), accuracy_samples_(0) {}

void CurriculumGenerator::add_stage(const Stage& stage) {
    stages_.push_back(stage);
}

void CurriculumGenerator::report_accuracy(double accuracy) {
    running_accuracy_ = (running_accuracy_ * accuracy_samples_ + accuracy) / (accuracy_samples_ + 1);
    accuracy_samples_++;
    stage_sample_count_++;

    if (should_promote() && current_stage_idx_ + 1 < (int)stages_.size()) {
        current_stage_idx_++;
        stage_sample_count_ = 0;
        running_accuracy_ = 0.0;
        accuracy_samples_ = 0;
        if (progress_fn_) {
            const auto& st = stages_[(size_t)current_stage_idx_];
            progress_fn_(st.name, st.difficulty, st.accuracy_threshold);
        }
    }
}

bool CurriculumGenerator::should_promote() const {
    if (current_stage_idx_ >= (int)stages_.size() - 1) return false;
    if (stage_sample_count_ < 50) return false;
    return running_accuracy_ >= stages_[current_stage_idx_].accuracy_threshold;
}

std::vector<std::string> CurriculumGenerator::generate_next_batch(int batch_size) {
    if (current_stage_idx_ >= (int)stages_.size()) return {};
    if (stages_.empty()) return {};

    const auto& stage = stages_[current_stage_idx_];
    int remaining = stage.max_samples - stage_sample_count_;
    int actual_batch = std::min(batch_size, remaining);
    if (actual_batch <= 0) {
        if (current_stage_idx_ + 1 < (int)stages_.size()) {
            current_stage_idx_++;
            stage_sample_count_ = 0;
            return generate_next_batch(batch_size);
        }
        return {};
    }

    return stage.generator_fn(actual_batch);
}

std::string CurriculumGenerator::current_stage_name() const {
    if (current_stage_idx_ >= (int)stages_.size()) return "complete";
    return stages_[current_stage_idx_].name;
}

int CurriculumGenerator::current_difficulty() const {
    if (current_stage_idx_ >= (int)stages_.size()) return -1;
    return stages_[current_stage_idx_].difficulty;
}

bool CurriculumGenerator::is_complete() const {
    return current_stage_idx_ >= (int)stages_.size();
}

void CurriculumGenerator::set_progress_fn(std::function<void(const std::string&, int, double)> fn) {
    // BUGFIX (bug census): was (void)fn — callback silently dropped.
    progress_fn_ = std::move(fn);
}

// ==================== TokenizerEvaluator ====================

TokenizerEvaluator::TokenizerEvaluator(Tokenizer* tokenizer)
    : tokenizer_(tokenizer) {}

std::vector<int> TokenizerEvaluator::encode_with_check(const std::string& text) const {
    try {
        return tokenizer_->encode(text);
    } catch (...) {
        std::fprintf(stderr, "[WARN] Exception caught: %s (tokenizer encode failed)\n", __func__);
        return {};
    }
}

float TokenizerEvaluator::compression_ratio(const std::string& text) const {
    if (text.empty()) return 0.0f;
    auto ids = encode_with_check(text);
    if (ids.empty()) return 0.0f;
    float bytes_per_token = (float)text.size() / (float)ids.size();
    return bytes_per_token;
}

float TokenizerEvaluator::re_encoding_accuracy(const std::string& text) const {
    if (text.empty()) return 1.0f;
    auto ids = encode_with_check(text);
    if (ids.empty()) return 0.0f;
    std::string reconstructed = tokenizer_->decode(ids);
    size_t min_len = std::min(text.size(), reconstructed.size());
    if (min_len == 0) return 0.0f;
    size_t matches = 0;
    for (size_t i = 0; i < min_len; i++) {
        if (text[i] == reconstructed[i]) matches++;
    }
    return (float)matches / (float)text.size();
}

float TokenizerEvaluator::vocabulary_coverage(const std::string& corpus) const {
    if (corpus.empty()) return 1.0f;
    std::unordered_set<int> vocab_set;
    int total_unique = 0;
    auto ids = encode_with_check(corpus);
    for (int id : ids) {
        if (vocab_set.find(id) == vocab_set.end()) {
            vocab_set.insert(id);
            total_unique++;
        }
    }
    if (total_unique == 0) return 0.0f;
    int vs = tokenizer_->vocab_size();
    if (vs == 0) return 0.0f;
    return (float)vocab_set.size() / (float)vs;
}

float TokenizerEvaluator::average_token_length(const std::string& corpus) const {
    if (corpus.empty()) return 0.0f;
    auto ids = encode_with_check(corpus);
    if (ids.empty()) return 0.0f;
    return (float)ids.size();
}

float TokenizerEvaluator::vocabulary_utilization(const std::string& corpus) const {
    return vocabulary_coverage(corpus);
}

float TokenizerEvaluator::reconstruction_fidelity(const std::string& text) const {
    return re_encoding_accuracy(text);
}

float TokenizerEvaluator::character_error_rate(const std::string& original, const std::string& reconstructed) const {
    if (original.empty() && reconstructed.empty()) return 0.0f;
    if (original.empty()) return 1.0f;

    size_t m = original.size();
    size_t n = reconstructed.size();
    std::vector<size_t> prev(n + 1);
    std::vector<size_t> curr(n + 1);

    for (size_t j = 0; j <= n; j++) prev[j] = j;

    for (size_t i = 1; i <= m; i++) {
        curr[0] = i;
        for (size_t j = 1; j <= n; j++) {
            if (original[i-1] == reconstructed[j-1]) {
                curr[j] = prev[j-1];
            } else {
                curr[j] = 1 + std::min({prev[j], curr[j-1], prev[j-1]});
            }
        }
        std::swap(prev, curr);
    }

    return (float)prev[n] / (float)m;
}

float TokenizerEvaluator::byte_efficiency(const std::string& text) const {
    if (text.empty()) return 0.0f;
    auto ids = encode_with_check(text);
    if (ids.empty()) return 0.0f;

    int bytes_used = 0;
    for (int id : ids) {
        std::string token = tokenizer_->decode({id});
        bytes_used += (int)token.size();
    }

    return (float)bytes_used / (float)text.size();
}

TokenizerEvaluator::EvaluationReport TokenizerEvaluator::evaluate(const std::vector<std::string>& corpus) const {
    EvaluationReport report;
    std::memset(&report, 0, sizeof(report));

    std::string all_text;
    for (const auto& t : corpus) all_text += t + " ";

    report.corpus_size_bytes = all_text.size();

    auto all_ids = encode_with_check(all_text);
    report.total_tokens = (int)all_ids.size();

    std::unordered_set<int> unique_ids(all_ids.begin(), all_ids.end());
    report.unique_tokens = (int)unique_ids.size();

    report.compression_ratio = compression_ratio(all_text);
    report.re_encoding_accuracy = re_encoding_accuracy(all_text);
    report.vocabulary_coverage = vocabulary_coverage(all_text);
    report.average_token_length = average_token_length(all_text);
    report.vocabulary_utilization = report.vocabulary_coverage;
    report.reconstruction_fidelity = report.re_encoding_accuracy;
    report.byte_efficiency = byte_efficiency(all_text);

    if (!all_text.empty()) {
        std::string decoded = tokenizer_->decode(all_ids);
        report.character_error_rate = character_error_rate(all_text, decoded);
    }

    report.oov_count = 0;
    report.unique_tokens = (int)unique_ids.size();

    return report;
}

std::string TokenizerEvaluator::report_string(const EvaluationReport& report) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "Tokenizer Evaluation Report:\n";
    oss << "  Compression Ratio:       " << report.compression_ratio << " bytes/token\n";
    oss << "  Re-encoding Accuracy:    " << report.re_encoding_accuracy * 100.0f << "%\n";
    oss << "  Vocabulary Coverage:     " << report.vocabulary_coverage * 100.0f << "%\n";
    oss << "  Average Token Length:    " << report.average_token_length << " tokens\n";
    oss << "  Vocabulary Utilization:  " << report.vocabulary_utilization * 100.0f << "%\n";
    oss << "  Reconstruction Fidelity: " << report.reconstruction_fidelity * 100.0f << "%\n";
    oss << "  Byte Efficiency:         " << report.byte_efficiency * 100.0f << "%\n";
    oss << "  Character Error Rate:    " << report.character_error_rate * 100.0f << "%\n";
    oss << "  Total Tokens:            " << report.total_tokens << "\n";
    oss << "  Unique Tokens:           " << report.unique_tokens << "\n";
    oss << "  OOV Count:               " << report.oov_count << "\n";
    oss << "  Corpus Size (bytes):     " << report.corpus_size_bytes << "\n";
    return oss.str();
}

void TokenizerEvaluator::compare_tokenizers(const std::vector<Tokenizer*>& tokenizers,
                                             const std::vector<std::string>& corpus,
                                             std::vector<EvaluationReport>& reports) {
    reports.clear();
    for (auto* tok : tokenizers) {
        if (tok) {
            TokenizerEvaluator evaluator(tok);
            reports.push_back(evaluator.evaluate(corpus));
        }
    }
}

} // namespace quant
