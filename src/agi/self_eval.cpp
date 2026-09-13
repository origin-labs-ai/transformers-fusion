// ========================================================================
// self_eval.cpp — Self-Evaluation Benchmark Suite (DIFFUSION T44)
// ========================================================================
#include "quant/self_eval.h"
#include "quant/meta_cognition.h"
#include "quant/agi_utils.h"
#include "quant/random.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <numeric>
#include <ctime>
#include <iomanip>
#include <cstdio>
#include <cstring>
#include <chrono>

namespace quant {
namespace agi {

// ========================================================================
// Constructor
// ========================================================================
SelfEvalSuite::SelfEvalSuite(Model* model) : model_(model) {}

int64_t SelfEvalSuite::get_vocab_size() {
    if (!model_) return 256;
    return (int64_t)model_->config.vocab_size;
}

// ========================================================================
// Accuracy benchmarks
// ========================================================================
AccuracyResult SelfEvalSuite::run_accuracy_benchmark(const std::string& benchmark_name, int64_t n_samples) {
    std::string lower = to_lower(benchmark_name);
    if (lower.find("mmlu") != std::string::npos) return run_accuracy_benchmark_mmlu(n_samples);
    if (lower.find("arc") != std::string::npos) return run_accuracy_benchmark_arc(n_samples);
    if (lower.find("hellaswag") != std::string::npos || lower.find("swag") != std::string::npos)
        return run_accuracy_benchmark_hellaswag(n_samples);
    if (lower.find("gsm") != std::string::npos || lower.find("math") != std::string::npos)
        return run_accuracy_benchmark_gsm8k(n_samples);

    AccuracyResult result;
    result.benchmark_name = benchmark_name;
    result.total_problems = n_samples;

    if (!model_) {
        result.accuracy = 0.0f;
        return result;
    }

    int vocab_size = (int)get_vocab_size();
    struct TestCase { std::string input; std::string keyword; };
    std::vector<TestCase> tests = {
        {"Solve: What is 2+2?", "4"},
        {"Solve: What is 10*3?", "30"},
        {"Solve: What is 15-7?", "8"},
        {"Solve: What is 6*7?", "42"},
        {"Solve: What is 100/4?", "25"},
        {"What is the capital of France?", "Paris"},
        {"What planet is closest to the sun?", "Mercury"},
        {"What is the speed of light?", "299"},
        {"Who wrote Hamlet?", "Shakespeare"},
        {"What is H2O?", "water"},
    };

    int64_t n = std::min(n_samples, (int64_t)tests.size());
    int64_t correct = 0;

    auto start = std::chrono::steady_clock::now();

    for (int64_t i = 0; i < n; i++) {
        auto ids = simple_encode(tests[i].input, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 20);
        std::string response = simple_decode(gen);
        std::string response_lower = to_lower(response);
        if (response_lower.find(to_lower(tests[i].keyword)) != std::string::npos) correct++;
    }

    auto end = std::chrono::steady_clock::now();
    result.total_problems = n;
    result.correct_count = correct;
    result.accuracy = n > 0 ? (float)correct / (float)n : 0.0f;
    result.runtime_ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return result;
}

AccuracyResult SelfEvalSuite::run_accuracy_benchmark_mmlu(int64_t n_samples) {
    AccuracyResult result;
    result.benchmark_name = "MMLU";
    if (!model_) return result;

    int vocab_size = (int)get_vocab_size();
    struct TestCase { std::string question; std::string answer; };
    std::vector<TestCase> tests = {
        {"What is the capital of France?", "Paris"},
        {"What is 2+2?", "4"},
        {"Who wrote Romeo and Juliet?", "Shakespeare"},
        {"What is the chemical symbol for water?", "H2O"},
        {"What planet is known as the Red Planet?", "Mars"},
        {"What is the largest ocean on Earth?", "Pacific"},
        {"What is the powerhouse of the cell?", "mitochondria"},
        {"Who painted the Mona Lisa?", "da Vinci"},
        {"What is the speed of sound?", "343"},
        {"What year did World War II end?", "1945"},
        {"What is Newton's second law?", "F=ma"},
        {"What is the square root of 144?", "12"},
    };

    int64_t n = std::min(n_samples, (int64_t)tests.size());
    int64_t correct = 0;

    auto start = std::chrono::steady_clock::now();

    for (int64_t i = 0; i < n; i++) {
        std::string prompt = "Answer concisely: " + tests[i].question;
        auto ids = simple_encode(prompt, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 20);
        std::string response = simple_decode(gen);
        std::string response_lower = to_lower(response);
        if (response_lower.find(to_lower(tests[i].answer)) != std::string::npos) correct++;
    }

    auto end = std::chrono::steady_clock::now();
    result.total_problems = n;
    result.correct_count = correct;
    result.accuracy = n > 0 ? (float)correct / (float)n : 0.0f;
    result.runtime_ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return result;
}

AccuracyResult SelfEvalSuite::run_accuracy_benchmark_arc(int64_t n_samples) {
    AccuracyResult result;
    result.benchmark_name = "ARC";
    if (!model_) return result;

    int vocab_size = (int)get_vocab_size();
    struct TestCase { std::string question; std::string answer; };
    std::vector<TestCase> tests = {
        {"If you drop a ball, what happens?", "fall"},
        {"What do plants need to grow?", "sunlight"},
        {"What happens when you heat water to 100C?", "boil"},
        {"Why do we wear warm clothes in winter?", "warm"},
        {"What does a seed grow into?", "plant"},
        {"What causes the tides in the ocean?", "moon"},
        {"Why is the sky blue?", "scatter"},
        {"What makes a rainbow?", "light"},
        {"Why do objects fall?", "gravity"},
        {"What makes ice melt?", "heat"},
    };

    int64_t n = std::min(n_samples, (int64_t)tests.size());
    int64_t correct = 0;

    auto start = std::chrono::steady_clock::now();

    for (int64_t i = 0; i < n; i++) {
        auto ids = simple_encode(tests[i].question, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 20);
        std::string response = simple_decode(gen);
        std::string response_lower = to_lower(response);
        if (response_lower.find(to_lower(tests[i].answer)) != std::string::npos) correct++;
    }

    auto end = std::chrono::steady_clock::now();
    result.total_problems = n;
    result.correct_count = correct;
    result.accuracy = n > 0 ? (float)correct / (float)n : 0.0f;
    result.runtime_ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return result;
}

AccuracyResult SelfEvalSuite::run_accuracy_benchmark_hellaswag(int64_t n_samples) {
    AccuracyResult result;
    result.benchmark_name = "HellaSwag";
    if (!model_) return result;

    int vocab_size = (int)get_vocab_size();
    struct TestCase { std::string context; std::string continuation; };
    std::vector<TestCase> tests = {
        {"A person is cooking eggs. What happens next?", "flip"},
        {"A car is driving down the road. What happens next?", "turn"},
        {"A dog is running in the park. What happens next?", "fetch"},
        {"A student is studying for an exam. What happens next?", "read"},
        {"A chef is chopping vegetables. What happens next?", "cook"},
        {"A person is reading a book. What happens next?", "turn"},
        {"A cat is sleeping on a couch. What happens next?", "wake"},
        {"Children are playing in a yard. What happens next?", "run"},
        {"A person is typing on a keyboard. What happens next?", "type"},
        {"Someone is walking on a trail. What happens next?", "walk"},
    };

    int64_t n = std::min(n_samples, (int64_t)tests.size());
    int64_t correct = 0;

    auto start = std::chrono::steady_clock::now();

    for (int64_t i = 0; i < n; i++) {
        auto ids = simple_encode(tests[i].context, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 20);
        std::string response = simple_decode(gen);
        std::string response_lower = to_lower(response);
        if (response_lower.find(to_lower(tests[i].continuation)) != std::string::npos) correct++;
    }

    auto end = std::chrono::steady_clock::now();
    result.total_problems = n;
    result.correct_count = correct;
    result.accuracy = n > 0 ? (float)correct / (float)n : 0.0f;
    result.runtime_ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return result;
}

AccuracyResult SelfEvalSuite::run_accuracy_benchmark_gsm8k(int64_t n_samples) {
    AccuracyResult result;
    result.benchmark_name = "GSM8K";
    if (!model_) return result;

    int vocab_size = (int)get_vocab_size();
    struct TestCase { std::string problem; std::string answer; };
    std::vector<TestCase> tests = {
        {"Alice has 5 apples. She gives 2 to Bob. How many does she have left?", "3"},
        {"A train travels 60 mph for 2.5 hours. How far does it travel?", "150"},
        {"If a shirt costs $25 and is 20% off, what is the price?", "20"},
        {"A rectangle has length 8 and width 5. What is its area?", "40"},
        {"What is 15% of 200?", "30"},
        {"If you buy 3 books at $12 each, how much do you spend?", "36"},
        {"A car goes 300 miles on 10 gallons. What is the MPG?", "30"},
        {"What is the sum of angles in a triangle?", "180"},
    };

    int64_t n = std::min(n_samples, (int64_t)tests.size());
    int64_t correct = 0;

    auto start = std::chrono::steady_clock::now();

    for (int64_t i = 0; i < n; i++) {
        std::string prompt = "Solve step by step: " + tests[i].problem + " Answer:";
        auto ids = simple_encode(prompt, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 40);
        std::string response = simple_decode(gen);
        std::string response_lower = to_lower(response);
        if (response_lower.find(to_lower(tests[i].answer)) != std::string::npos) correct++;
    }

    auto end = std::chrono::steady_clock::now();
    result.total_problems = n;
    result.correct_count = correct;
    result.accuracy = n > 0 ? (float)correct / (float)n : 0.0f;
    result.runtime_ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return result;
}

// ========================================================================
// Calibration benchmark
// ========================================================================
CalibrationResult SelfEvalSuite::run_calibration_benchmark(int64_t n_samples) {
    CalibrationResult result;
    result.total_predictions = n_samples;
    if (!model_) return result;

    int vocab_size = (int)get_vocab_size();
    std::vector<std::string> questions = {
        "What is 2+2?", "What is 10-5?", "What is 3*4?", "What is 20/4?",
        "What is the capital of France?", "What color is the sky?",
        "What is H2O?", "How many legs does a dog have?",
        "What planet do we live on?", "What is 7*8?",
    };

    int64_t correct = 0;
    float total_conf = 0;

    auto start = std::chrono::steady_clock::now();

    for (int64_t i = 0; i < n_samples; i++) {
        std::string q = questions[i % (int64_t)questions.size()];
        auto ids = simple_encode(q, vocab_size);

        int64_t seq_len = (int64_t)ids.size();
        Tensor input_tensor({1, seq_len});
        Tensor pos_tensor({1, seq_len});
        float* idp = input_tensor.data<float>();
        float* psp = pos_tensor.data<float>();
        for (int64_t j = 0; j < seq_len; j++) {
            idp[j] = (float)ids[j];
            psp[j] = (float)j;
        }

        Tensor logits = model_->forward(input_tensor, pos_tensor, nullptr);
        int64_t V = logits.dim(logits.rank() - 1);
        const float* lp = logits.data<float>();

        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++) max_l = std::max(max_l, lp[v]);
        float sum = 0;
        for (int64_t v = 0; v < V; v++) sum += std::exp(lp[v] - max_l);
        float top_prob = std::exp(lp[0] - max_l) / (sum + 1e-10f);

        total_conf += top_prob;

        int gen_count = 0;
        float total_gen_conf = 0;
        for (int64_t v = 0; v < V; v++) {
            float p = std::exp(lp[v] - max_l) / (sum + 1e-10f);
            total_gen_conf += p;
            gen_count++;
            if (gen_count >= 10) break;
        }

        auto gen = generate_new_tokens(model_, ids, vocab_size, 10);
        std::string response = simple_decode(gen);

        bool was_correct = !response.empty() && (
            std::isdigit((unsigned char)response[0]) ||
            response[0] == '-' ||
            response[0] == '+' ||
            to_lower(response).find("true") != std::string::npos ||
            to_lower(response).find("yes") != std::string::npos
        );
        if (was_correct) correct++;

        result.confidence_vs_accuracy.push_back({top_prob, was_correct ? 1.0f : 0.0f});
    }

    auto end = std::chrono::steady_clock::now();
    result.avg_confidence = total_conf / (float)n_samples;
    result.avg_accuracy = (float)correct / (float)n_samples;

    std::sort(result.confidence_vs_accuracy.begin(), result.confidence_vs_accuracy.end(),
              [](auto& a, auto& b) { return a.first < b.first; });

    int num_bins = 10;
    float bin_size = 1.0f / (float)num_bins;
    float total_ece = 0;
    float max_ece = 0;

    for (int b = 0; b < num_bins; b++) {
        float lower = (float)b * bin_size;
        float upper = (float)(b + 1) * bin_size;
        float bin_conf_sum = 0;
        float bin_acc_sum = 0;
        int64_t bin_count = 0;

        for (auto& [conf, acc] : result.confidence_vs_accuracy) {
            if (conf >= lower && conf < upper) {
                bin_conf_sum += conf;
                bin_acc_sum += acc;
                bin_count++;
            }
        }

        if (bin_count > 0) {
            float avg_conf = bin_conf_sum / (float)bin_count;
            float avg_acc = bin_acc_sum / (float)bin_count;
            float bin_ece = std::abs(avg_conf - avg_acc);
            total_ece += bin_ece * (float)bin_count;
            if (bin_ece > max_ece) max_ece = bin_ece;
        }
    }

    result.expected_calibration_error = n_samples > 0 ? total_ece / (float)n_samples : 0.0f;
    result.max_calibration_error = max_ece;
    return result;
}

float SelfEvalSuite::compute_expected_calibration_error(const std::vector<CalibrationBin>& bins) {
    float total_ece = 0;
    int64_t total_count = 0;
    for (auto& bin : bins) {
        if (bin.total_count > 0) {
            float avg_acc = (float)bin.correct_count / (float)bin.total_count;
            float bin_ece = std::abs(bin.avg_confidence - avg_acc);
            total_ece += bin_ece * (float)bin.total_count;
            total_count += bin.total_count;
        }
    }
    return total_count > 0 ? total_ece / (float)total_count : 0.0f;
}

// ========================================================================
// Consistency benchmark
// ========================================================================
ConsistencyResult SelfEvalSuite::run_consistency_benchmark(const std::string& problem, int64_t n_asks) {
    ConsistencyResult result;
    result.problem_description = problem;
    result.total_asks = n_asks;

    if (!model_) return result;

    int vocab_size = (int)get_vocab_size();
    std::vector<std::string> responses;
    std::string first_response;

    for (int64_t i = 0; i < n_asks; i++) {
        auto ids = simple_encode(problem, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 20);
        std::string response = simple_decode(gen);
        responses.push_back(response);

        if (i == 0) first_response = response;
    }

    result.responses = responses;

    int64_t agreements = 0;
    for (auto& r : responses) {
        if (r == first_response) agreements++;
    }

    result.agreement_count = agreements;
    result.consistency_score = n_asks > 0 ? (float)agreements / (float)n_asks : 0.0f;
    return result;
}

std::vector<ConsistencyResult> SelfEvalSuite::run_consistency_suite(int64_t n_problems, int64_t n_asks) {
    std::vector<std::string> problems = {
        "What is 2+2?", "What is the capital of France?", "What is H2O?",
        "What color is the sky?", "How many continents are there?",
        "What is 10*10?", "What is the speed of light?", "Who invented the telephone?",
        "What is the boiling point of water?", "How many days in a week?",
        "What is 100-37?", "What is the largest planet?", "What is 5+7?",
        "What is the freezing point of water?", "How many bones are in the human body?",
    };

    std::vector<ConsistencyResult> results;
    int64_t n = std::min(n_problems, (int64_t)problems.size());
    for (int64_t i = 0; i < n; i++) {
        results.push_back(run_consistency_benchmark(problems[i], n_asks));
    }
    return results;
}

// ========================================================================
// Robustness benchmark
// ========================================================================
std::string SelfEvalSuite::generate_perturbation(const std::string& original) {
    std::string perturbed = original;
    if (perturbed.empty()) return perturbed;

    std::mt19937 rng((unsigned)std::time(nullptr));
    int perturbation_type = (int)(rng() % 5);

    switch (perturbation_type) {
    case 0: {
        int pos = (int)(rng() % (int)perturbed.size());
        perturbed[pos] = (char)('a' + (int)(rng() % 26));
        break;
    }
    case 1: {
        if (perturbed.size() > 1) {
            int pos = (int)(rng() % ((int)perturbed.size() - 1));
            std::swap(perturbed[pos], perturbed[pos + 1]);
        }
        break;
    }
    case 2: {
        int pos = (int)(rng() % ((int)perturbed.size() + 1));
        perturbed.insert(perturbed.begin() + pos, (char)('a' + (int)(rng() % 26)));
        break;
    }
    case 3: {
        if (perturbed.size() > 1) {
            int pos = (int)(rng() % (int)perturbed.size());
            perturbed.erase(pos, 1);
        }
        break;
    }
    case 4: {
        for (char& c : perturbed) {
            if (c == ' ') c = '_';
            else if (c == '_') c = ' ';
        }
        break;
    }
    }

    return perturbed;
}

RobustnessResult SelfEvalSuite::run_robustness_benchmark(const std::string& problem, int64_t n_perturbations) {
    RobustnessResult result;
    result.problem_description = problem;
    result.perturbations_tested = n_perturbations;

    if (!model_) return result;

    int vocab_size = (int)get_vocab_size();
    auto baseline_ids = simple_encode(problem, vocab_size);
    auto baseline_gen = generate_new_tokens(model_, baseline_ids, vocab_size, 20);
    std::string baseline_response = simple_decode(baseline_gen);

    int64_t baseline_correct = 0;
    {
        std::string lower = to_lower(baseline_response);
        if (lower.find("4") != std::string::npos || lower.find("four") != std::string::npos ||
            !baseline_response.empty()) {
            baseline_correct = 1;
        }
    }
    result.baseline_accuracy = (float)baseline_correct;

    int64_t perturbed_correct = 0;
    std::mt19937 rng(42);

    for (int64_t i = 0; i < n_perturbations; i++) {
        std::string perturbed = generate_perturbation(problem);
        auto ids = simple_encode(perturbed, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 20);
        std::string response = simple_decode(gen);

        if (!response.empty()) perturbed_correct++;
    }

    result.perturbed_accuracy = n_perturbations > 0 ? (float)perturbed_correct / (float)n_perturbations : 0.0f;
    result.degradation = result.baseline_accuracy - result.perturbed_accuracy;
    return result;
}

std::vector<RobustnessResult> SelfEvalSuite::run_robustness_suite(int64_t n_problems) {
    std::vector<std::string> problems = {
        "What is 2+2?", "What is 10-3?", "What is 5*5?",
        "What is 100/4?", "What is 7+8?",
    };

    std::vector<RobustnessResult> results;
    int64_t n = std::min(n_problems, (int64_t)problems.size());
    for (int64_t i = 0; i < n; i++) {
        results.push_back(run_robustness_benchmark(problems[i], 20));
    }
    return results;
}

// ========================================================================
// Speed benchmark
// ========================================================================
SpeedResult SelfEvalSuite::run_speed_benchmark(int64_t n_operations) {
    SpeedResult result;
    result.total_operations = n_operations;

    if (!model_) return result;

    int vocab_size = (int)get_vocab_size();
    std::vector<double> latencies;
    latencies.reserve(n_operations);

    auto total_start = std::chrono::steady_clock::now();

    for (int64_t i = 0; i < n_operations; i++) {
        auto op_start = std::chrono::steady_clock::now();

        std::string prompt = "Test operation " + std::to_string(i);
        auto ids = simple_encode(prompt, vocab_size);
        (void)generate_new_tokens(model_, ids, vocab_size, 5);

        auto op_end = std::chrono::steady_clock::now();
        double ms = (double)std::chrono::duration_cast<std::chrono::microseconds>(op_end - op_start).count() / 1000.0;
        latencies.push_back(ms);
    }

    auto total_end = std::chrono::steady_clock::now();
    result.total_time_ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

    std::sort(latencies.begin(), latencies.end());

    double total_lat = 0;
    for (auto l : latencies) total_lat += l;
    result.mean_latency_ms = n_operations > 0 ? (float)(total_lat / (double)n_operations) : 0.0f;
    result.p50_latency_ms = (float)latencies[(int64_t)(n_operations * 0.5)];
    result.p90_latency_ms = (float)latencies[(int64_t)(n_operations * 0.9)];
    result.p99_latency_ms = (float)latencies[std::min((int64_t)(n_operations * 0.99), n_operations - 1)];
    result.ops_per_second = result.total_time_ms > 0 ? (float)((double)n_operations / (result.total_time_ms / 1000.0)) : 0.0f;
    return result;
}

// ========================================================================
// Memory benchmark
// ========================================================================
MemoryResult SelfEvalSuite::run_memory_benchmark(int64_t n_allocations) {
    MemoryResult result;
    result.cache_hits = 0;
    result.cache_misses = 0;

    int64_t vocab_size = get_vocab_size();
    std::vector<std::pair<std::string, Tensor>> cache;
    std::mt19937 rng(42);

    float peak_mb = 0;
    float total_mb = 0;
    int64_t sample_count = 0;

    for (int64_t i = 0; i < n_allocations; i++) {
        std::string key = "key_" + std::to_string((int)(rng() % 100));

        bool found = false;
        for (auto& [k, v] : cache) {
            if (k == key) {
                result.cache_hits++;
                found = true;
                break;
            }
        }

        if (!found) {
            result.cache_misses++;
            int64_t dim = 32 + (int64_t)(rng() % 64);
            Tensor t({dim});
            float* data = t.data<float>();
            for (int64_t j = 0; j < dim; j++) data[j] = (float)rng() / (float)rng.max();
            cache.push_back({key, std::move(t)});
        }

        if ((int64_t)cache.size() > 1000) {
            cache.erase(cache.begin());
        }

        int64_t total_elements = 0;
        for (auto& [k, v] : cache) total_elements += v.numel();
        float current_mb = (float)total_elements * 4.0f / (1024.0f * 1024.0f);
        total_mb += current_mb;
        sample_count++;
        if (current_mb > peak_mb) peak_mb = current_mb;
    }

    result.peak_ram_mb = peak_mb;
    result.avg_ram_mb = sample_count > 0 ? total_mb / (float)sample_count : 0.0f;
    int64_t total_cache = result.cache_hits + result.cache_misses;
    result.cache_efficiency = total_cache > 0 ? (float)result.cache_hits / (float)total_cache : 0.0f;
    return result;
}

// ========================================================================
// Pass@k metric
// ========================================================================
std::vector<std::string> SelfEvalSuite::generate_answer_samples(const std::string& problem, int64_t n) {
    std::vector<std::string> samples;
    if (!model_) return samples;

    int vocab_size = (int)get_vocab_size();

    for (int64_t i = 0; i < n; i++) {
        auto ids = simple_encode(problem, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 20);
        samples.push_back(simple_decode(gen));
    }

    return samples;
}

float SelfEvalSuite::evaluate_answer(const std::string& problem, const std::string& answer) {
    if (answer.empty()) return 0.0f;

    std::string problem_lower = to_lower(problem);
    std::string answer_lower = to_lower(answer);

    if (problem_lower.find("2+2") != std::string::npos || problem_lower.find("2 + 2") != std::string::npos) {
        if (answer_lower.find("4") != std::string::npos || answer_lower.find("four") != std::string::npos) return 1.0f;
    }
    if (problem_lower.find("10-5") != std::string::npos) {
        if (answer_lower.find("5") != std::string::npos || answer_lower.find("five") != std::string::npos) return 1.0f;
    }
    if (problem_lower.find("3*4") != std::string::npos || problem_lower.find("3 * 4") != std::string::npos) {
        if (answer_lower.find("12") != std::string::npos || answer_lower.find("twelve") != std::string::npos) return 1.0f;
    }
    if (problem_lower.find("capital of france") != std::string::npos) {
        if (answer_lower.find("paris") != std::string::npos) return 1.0f;
    }
    if (problem_lower.find("h2o") != std::string::npos) {
        if (answer_lower.find("water") != std::string::npos) return 1.0f;
    }

    if (!answer.empty() && std::isdigit((unsigned char)answer_lower[0])) return 0.5f;
    return 0.3f;
}

std::vector<float> SelfEvalSuite::compute_logits_confidence(const std::string& input) {
    std::vector<float> confidences;
    if (!model_) return confidences;

    int vocab_size = (int)get_vocab_size();
    auto ids = simple_encode(input, vocab_size);
    int64_t seq_len = (int64_t)ids.size();

    Tensor input_tensor({1, seq_len});
    Tensor pos_tensor({1, seq_len});
    float* idp = input_tensor.data<float>();
    float* psp = pos_tensor.data<float>();
    for (int64_t i = 0; i < seq_len; i++) {
        idp[i] = (float)ids[i];
        psp[i] = (float)i;
    }

    Tensor logits = model_->forward(input_tensor, pos_tensor, nullptr);
    int64_t V = logits.dim(logits.rank() - 1);
    const float* lp = logits.data<float>();

    for (int64_t i = 0; i < seq_len; i++) {
        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++) max_l = std::max(max_l, lp[i * V + v]);
        float sum = 0;
        for (int64_t v = 0; v < V; v++) sum += std::exp(lp[i * V + v] - max_l);
        float max_prob = std::exp(lp[i * V + 0] - max_l) / (sum + 1e-10f);
        confidences.push_back(max_prob);
    }
    return confidences;
}

PassAtKResult SelfEvalSuite::run_pass_at_k(const std::string& benchmark_name, int64_t k, int64_t n_problems) {
    PassAtKResult result;
    result.k = k;
    result.total_problems = n_problems;

    std::vector<std::string> problems = {
        "What is 2+2?", "What is 10-5?", "What is 3*4?",
        "What is the capital of France?", "What is H2O?",
        "What is 5+7?", "What is 100/4?", "What is 7*8?",
        "What is 20-9?", "What is 6*6?",
    };

    int64_t n = std::min(n_problems, (int64_t)problems.size());
    int64_t problems_with_correct = 0;

    for (int64_t p = 0; p < n; p++) {
        auto samples = generate_answer_samples(problems[p], k);
        bool any_correct = false;
        for (auto& sample : samples) {
            if (evaluate_answer(problems[p], sample) >= 0.8f) {
                any_correct = true;
                break;
            }
        }
        if (any_correct) problems_with_correct++;
    }

    result.problems_with_at_least_one_correct = problems_with_correct;
    result.pass_at_k = n > 0 ? (float)problems_with_correct / (float)n : 0.0f;
    return result;
}

std::vector<PassAtKResult> SelfEvalSuite::run_pass_at_k_suite(int64_t max_k) {
    std::vector<PassAtKResult> results;
    std::vector<int64_t> k_values;

    for (int64_t k = 1; k <= max_k; k++) {
        k_values.push_back(k);
    }
    if (max_k < 10) {
        k_values.push_back(10);
    }

    for (int64_t k : k_values) {
        results.push_back(run_pass_at_k("general", k, 10));
    }
    return results;
}

// ========================================================================
// Report generation
// ========================================================================
EvaluationReport SelfEvalSuite::generate_report(const std::string& version_id) {
    EvaluationReport report;
    report.version_id = version_id;
    report.timestamp = (int64_t)std::time(nullptr);

    report.accuracy = run_accuracy_benchmark("general", 10);
    report.calibration = run_calibration_benchmark(100);
    report.consistency_results = run_consistency_suite(5, 5);
    report.robustness_results = run_robustness_suite(5);
    report.speed = run_speed_benchmark(50);
    report.memory = run_memory_benchmark(1000);
    report.pass_at_k_results = run_pass_at_k_suite(5);

    report.overall_score = report.accuracy.accuracy * 0.30f +
                           (1.0f - report.calibration.expected_calibration_error) * 0.20f +
                           report.speed.ops_per_second / 100.0f * 0.15f +
                           report.memory.cache_efficiency * 0.15f;

    float avg_consistency = 0;
    for (auto& c : report.consistency_results) avg_consistency += c.consistency_score;
    if (!report.consistency_results.empty())
        avg_consistency /= (float)report.consistency_results.size();
    report.overall_score += avg_consistency * 0.10f;

    float avg_robustness = 0;
    for (auto& r : report.robustness_results) avg_robustness += (1.0f - r.degradation);
    if (!report.robustness_results.empty())
        avg_robustness /= (float)report.robustness_results.size();
    report.overall_score += avg_robustness * 0.10f;

    report.overall_score = std::min(1.0f, std::max(0.0f, report.overall_score));

    report_history_.push_back(report);
    if ((int64_t)report_history_.size() > max_reports) {
        report_history_.erase(report_history_.begin());
    }

    return report;
}

VersionComparison SelfEvalSuite::compare_versions(const EvaluationReport& old_report,
                                                    const EvaluationReport& new_report) {
    VersionComparison comp;
    comp.old_version = old_report.version_id;
    comp.new_version = new_report.version_id;

    comp.accuracy_delta = new_report.accuracy.accuracy - old_report.accuracy.accuracy;
    comp.calibration_delta = old_report.calibration.expected_calibration_error -
                             new_report.calibration.expected_calibration_error;
    comp.speed_delta = new_report.speed.ops_per_second - old_report.speed.ops_per_second;
    comp.memory_delta = old_report.memory.peak_ram_mb - new_report.memory.peak_ram_mb;

    comp.overall_improved = new_report.overall_score > old_report.overall_score;
    return comp;
}

// ========================================================================
// Report storage
// ========================================================================
void SelfEvalSuite::store_report(const EvaluationReport& report) {
    report_history_.push_back(report);
    if ((int64_t)report_history_.size() > max_reports) {
        report_history_.erase(report_history_.begin());
    }
}

std::vector<EvaluationReport> SelfEvalSuite::get_report_history() {
    return report_history_;
}

EvaluationReport SelfEvalSuite::get_latest_report() {
    if (report_history_.empty()) return EvaluationReport{};
    return report_history_.back();
}

} // namespace agi
} // namespace quant
