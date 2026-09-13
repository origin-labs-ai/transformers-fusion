#include "quant/hpo_nas.h"
#include "quant/random.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <numeric>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <fstream>
#include <random>
#include <set>
#include <cstdio>
#include <thread>
#include <functional>

namespace quant {
namespace hpo_nas {

namespace {

std::mt19937& rng() {
    thread_local std::mt19937 gen((unsigned)std::time(nullptr) ^ (unsigned)std::hash<std::thread::id>{}(std::this_thread::get_id()));
    return gen;
}

float uniform_float(float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng());
}

int uniform_int(int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng());
}

double now_ms() {
    return (double)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // anonymous namespace

// ========================================================================
// HPOEngine
// ========================================================================

HPOEngine::HPOEngine(Trainer* trainer, Model* model)
    : trainer_(trainer), model_(model) {}

HPOEngine::~HPOEngine() {}

void HPOEngine::set_search_space(const SearchSpace& space) {
    space_ = space;
}

SearchSpace HPOEngine::get_search_space() const {
    return space_;
}

void HPOEngine::set_hardware_constraints(const HardwareConstraints& hc) {
    hw_constraints_ = hc;
}

void HPOEngine::set_trainer(Trainer* trainer) {
    trainer_ = trainer;
}

void HPOEngine::set_model(Model* model) {
    model_ = model;
}

HPConfig HPOEngine::random_sample() {
    HPConfig c;
    c.id = next_config_id_++;
    c.lr = std::pow(10.0f, uniform_float(std::log10(space_.lr_min), std::log10(space_.lr_max)));
    c.weight_decay = std::pow(10.0f, uniform_float(std::log10(space_.wd_min), std::log10(space_.wd_max)));
    c.beta1 = uniform_float(space_.beta1_min, space_.beta1_max);
    c.beta2 = 0.999f;
    c.warmup_steps = uniform_int((int)space_.warmup_min, (int)space_.warmup_max);
    c.batch_size = (int64_t)1 << uniform_int(0, 7);
    c.batch_size = std::max(space_.batch_size_min, std::min(space_.batch_size_max, c.batch_size));
    c.dropout = uniform_float(space_.dropout_min, space_.dropout_max);
    c.depth = uniform_int(space_.depth_min, space_.depth_max);
    c.width = uniform_int(space_.width_min, space_.width_max);
    c.num_heads = uniform_int(space_.heads_min, std::min(space_.heads_max, c.width));
    while (c.width % c.num_heads != 0 && c.num_heads > 1) c.num_heads--;
    c.ffn_ratio = uniform_float(space_.ffn_ratio_min, space_.ffn_ratio_max);
    c.score = 0.0f;
    c.val_loss = 1e10f;
    c.alive = true;
    return c;
}

std::vector<HPConfig> HPOEngine::random_search(int n_configs) {
    std::vector<HPConfig> configs;
    for (int i = 0; i < n_configs; i++) {
        HPConfig c = random_sample();
        c.score = evaluate_config(c);
        eval_history_.push_back({c, c.score});
        configs.push_back(c);
    }
    return configs;
}

// ========================================================================
// Bayesian Optimization
// ========================================================================

std::vector<float> HPOEngine::config_to_features(const HPConfig& config) const {
    return {
        std::log10(config.lr + 1e-10f),
        std::log10(config.weight_decay + 1e-10f),
        config.beta1,
        (float)config.warmup_steps / 1000.0f,
        std::log2((float)config.batch_size),
        config.dropout,
        (float)config.depth / (float)space_.depth_max,
        (float)config.width / (float)space_.width_max,
        (float)config.num_heads / (float)space_.heads_max,
        config.ffn_ratio / space_.ffn_ratio_max
    };
}

float HPOEngine::rbf_kernel(const float* x1, const float* x2, int n, float length_scale) const {
    float sq_dist = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x1[i] - x2[i];
        sq_dist += d * d;
    }
    return std::exp(-sq_dist / (2.0f * length_scale * length_scale));
}

float HPOEngine::compute_distance(const HPConfig& a, const HPConfig& b) const {
    auto fa = config_to_features(a);
    auto fb = config_to_features(b);
    float sq = 0.0f;
    for (size_t i = 0; i < fa.size(); i++) {
        float d = fa[i] - fb[i];
        sq += d * d;
    }
    return std::sqrt(sq);
}

float HPOEngine::surrogate_predict(const HPConfig& config) {
    if (gp_training_data_.empty()) return 0.0f;

    auto xf = config_to_features(config);
    int n = (int)xf.size();

    float pred = 0.0f;
    float weight_sum = 0.0f;
    float noise = 0.01f;
    float length_scale = 1.0f;

    float k_star[10];
    for (int i = 0; i < n && i < 10; i++) k_star[i] = xf[i];

    for (auto& [train_x, train_y] : gp_training_data_) {
        float k = rbf_kernel(k_star, train_x.data(), std::min(n, (int)train_x.size()), length_scale);
        pred += k * train_y;
        weight_sum += k;
    }

    if (weight_sum > 1e-10f) pred /= weight_sum;
    return pred;
}

void HPOEngine::surrogate_update(const HPConfig& config, float score) {
    gp_training_data_.push_back({config_to_features(config), score});
}

float HPOEngine::acquisition_function(const HPConfig& config, int n_evaluations) const {
    HPOEngine* self = const_cast<HPOEngine*>(this);
    float mean_pred = self->surrogate_predict(config);

    float max_obs = -1e10f;
    for (auto& [_, s] : gp_training_data_) {
        if (s > max_obs) max_obs = s;
    }

    float exploration = 0.0f;
    if (!gp_training_data_.empty()) {
        float min_dist = 1e10f;
        auto xf = config_to_features(config);
        for (auto& [tx, _] : gp_training_data_) {
            float sq = 0.0f;
            for (size_t i = 0; i < xf.size() && i < tx.size(); i++) {
                float d = xf[i] - tx[i];
                sq += d * d;
            }
            float dist = std::sqrt(sq);
            if (dist < min_dist) min_dist = dist;
        }
        exploration = std::min(2.0f, 1.0f / (min_dist + 0.1f));
    }

    float exploration_weight = 2.0f / (1.0f + 0.1f * (float)n_evaluations);
    return mean_pred + exploration_weight * exploration;
}

std::vector<HPConfig> HPOEngine::bayesian_optimization(int n_initial, int n_iterations) {
    gp_training_data_.clear();
    std::vector<HPConfig> all_configs;

    for (int i = 0; i < n_initial; i++) {
        HPConfig c = random_sample();
        c.score = evaluate_config(c);
        eval_history_.push_back({c, c.score});
        surrogate_update(c, c.score);
        all_configs.push_back(c);
    }

    for (int i = 0; i < n_iterations; i++) {
        HPConfig best_candidate = random_sample();
        float best_acq = -1e10f;
        int n_candidates = 200;

        for (int j = 0; j < n_candidates; j++) {
            HPConfig candidate = random_sample();
            float acq = acquisition_function(candidate, (int)all_configs.size());
            if (acq > best_acq) {
                best_acq = acq;
                best_candidate = candidate;
            }
        }

        best_candidate.score = evaluate_config(best_candidate);
        eval_history_.push_back({best_candidate, best_candidate.score});
        surrogate_update(best_candidate, best_candidate.score);
        all_configs.push_back(best_candidate);
    }

    return all_configs;
}

// ========================================================================
// Population-Based Training
// ========================================================================

HPConfig HPOEngine::select_parent(const std::vector<HPConfig>& population) {
    int n = (int)population.size();
    if (n == 0) return random_sample();

    int tournament = std::min(3, n);
    int best_idx = uniform_int(0, n - 1);
    for (int i = 1; i < tournament; i++) {
        int idx = uniform_int(0, n - 1);
        if (population[idx].score > population[best_idx].score) {
            best_idx = idx;
        }
    }
    return population[best_idx];
}

void HPOEngine::exploit_and_explore(HPConfig& child, const HPConfig& parent, float perturbation_factor) {
    child = parent;
    child.id = next_config_id_++;
    child.epoch++;
    child.alive = true;

    float explore = perturbation_factor;
    if (uniform_float(0.0f, 1.0f) < 0.5f) {
        child.lr = parent.lr * std::pow(2.0f, uniform_float(-explore, explore));
        child.lr = std::max(space_.lr_min, std::min(space_.lr_max, child.lr));
    }
    if (uniform_float(0.0f, 1.0f) < 0.5f) {
        child.weight_decay = parent.weight_decay * std::pow(2.0f, uniform_float(-explore, explore));
        child.weight_decay = std::max(space_.wd_min, std::min(space_.wd_max, child.weight_decay));
    }
    if (uniform_float(0.0f, 1.0f) < 0.3f) {
        child.dropout = parent.dropout + uniform_float(-0.1f, 0.1f);
        child.dropout = std::max(space_.dropout_min, std::min(space_.dropout_max, child.dropout));
    }
    if (uniform_float(0.0f, 1.0f) < 0.3f) {
        child.depth = parent.depth + uniform_int(-1, 1);
        child.depth = std::max(space_.depth_min, std::min(space_.depth_max, child.depth));
    }
}

void HPOEngine::population_based_training(int n_population, int n_generations) {
    std::vector<HPConfig> population;
    for (int i = 0; i < n_population; i++) {
        HPConfig c = random_sample();
        c.score = evaluate_config(c);
        eval_history_.push_back({c, c.score});
        population.push_back(c);
    }

    for (int gen = 0; gen < n_generations; gen++) {
        std::sort(population.begin(), population.end(),
                  [](const HPConfig& a, const HPConfig& b) { return a.score > b.score; });

        int keep = std::max(1, n_population / 4);
        int n_kill = n_population - keep;

        for (int i = 0; i < n_kill; i++) {
            HPConfig parent = select_parent(std::vector<HPConfig>(population.begin(), population.begin() + keep));
            float perturbation = 0.5f * (1.0f - (float)gen / (float)n_generations);
            exploit_and_explore(population[keep + i], parent, perturbation);
            population[keep + i].score = evaluate_config(population[keep + i]);
            eval_history_.push_back({population[keep + i], population[keep + i].score});
        }
    }
}

// ========================================================================
// Successive Halving
// ========================================================================

SuccessiveHalvingConfig HPOEngine::setup_successive_halving(int n_configs, int n_rungs) {
    SuccessiveHalvingConfig sh;
    sh.n_configs = n_configs;
    sh.n_rungs = n_rungs;
    sh.reduction_factor = 3.0f;

    sh.configs_per_rung.resize(n_rungs);
    sh.budget_per_rung.resize(n_rungs);

    for (int r = 0; r < n_rungs; r++) {
        sh.configs_per_rung[r] = std::max(1, (int)std::ceil((double)n_configs / std::pow(sh.reduction_factor, r)));
        sh.budget_per_rung[r] = std::pow(2.0, r);
    }

    return sh;
}

std::vector<HPConfig> HPOEngine::successive_halving(int n_configs, int n_rungs) {
    SuccessiveHalvingConfig sh = setup_successive_halving(n_configs, n_rungs);

    std::vector<HPConfig> candidates;
    for (int i = 0; i < sh.configs_per_rung[0]; i++) {
        HPConfig c = random_sample();
        c.score = evaluate_config(c);
        candidates.push_back(c);
    }

    for (int r = 0; r < n_rungs; r++) {
        int n_keep = sh.configs_per_rung[r];
        if (n_keep <= 0) break;
        if (r > 0) {
            int promote = std::min(n_keep, (int)candidates.size());
            std::sort(candidates.begin(), candidates.end(),
                      [](const HPConfig& a, const HPConfig& b) { return a.score > b.score; });
            candidates.resize(promote);
        }

        for (auto& c : candidates) {
            double budget = sh.budget_per_rung[r];
            int evals = std::max(1, (int)budget);
            float total_score = 0.0f;
            for (int e = 0; e < evals; e++) {
                total_score += evaluate_config(c);
            }
            c.score = total_score / (float)evals;
            eval_history_.push_back({c, c.score});
        }
    }

    return candidates;
}

// ========================================================================
// Evaluate
// ========================================================================

float HPOEngine::evaluate_config(const HPConfig& config) {
    float score = 0.0f;

    float lr_score = 1.0f - std::abs(std::log10(config.lr) - std::log10(3e-4f)) / 4.0f;
    lr_score = std::max(0.0f, std::min(1.0f, lr_score));

    float wd_score = 1.0f - std::abs(std::log10(config.weight_decay) - std::log10(1e-2f)) / 3.0f;
    wd_score = std::max(0.0f, std::min(1.0f, wd_score));

    float depth_score = 1.0f - std::abs((float)config.depth - 12.0f) / 24.0f;
    depth_score = std::max(0.0f, std::min(1.0f, depth_score));

    float width_score = 1.0f - std::abs((float)config.width - 768.0f) / 2048.0f;
    width_score = std::max(0.0f, std::min(1.0f, width_score));

    float heads_score = 1.0f - std::abs((float)config.num_heads - 12.0f) / 32.0f;
    heads_score = std::max(0.0f, std::min(1.0f, heads_score));

    float batch_score = 1.0f - std::abs(std::log2((float)config.batch_size) - 5.0f) / 4.0f;
    batch_score = std::max(0.0f, std::min(1.0f, batch_score));

    score = lr_score * 0.25f + wd_score * 0.15f + depth_score * 0.2f +
            width_score * 0.2f + heads_score * 0.1f + batch_score * 0.1f;

    if (hw_constraints_.enabled) {
        int64_t params = estimate_param_count(config.depth, config.width,
                                               (int)((float)config.width * config.ffn_ratio),
                                               (int)config.batch_size * 1000);
        if (params > hw_constraints_.max_params) score *= 0.5f;
    }

    return std::max(0.0f, std::min(1.0f, score));
}

HPConfig HPOEngine::crossover(const HPConfig& a, const HPConfig& b) {
    HPConfig child;
    child.id = next_config_id_++;
    child.lr = uniform_float(0.0f, 1.0f) < 0.5f ? a.lr : b.lr;
    child.weight_decay = uniform_float(0.0f, 1.0f) < 0.5f ? a.weight_decay : b.weight_decay;
    child.beta1 = uniform_float(0.0f, 1.0f) < 0.5f ? a.beta1 : b.beta1;
    child.warmup_steps = uniform_float(0.0f, 1.0f) < 0.5f ? a.warmup_steps : b.warmup_steps;
    child.batch_size = uniform_float(0.0f, 1.0f) < 0.5f ? a.batch_size : b.batch_size;
    child.dropout = uniform_float(0.0f, 1.0f) < 0.5f ? a.dropout : b.dropout;
    child.depth = uniform_float(0.0f, 1.0f) < 0.5f ? a.depth : b.depth;
    child.width = uniform_float(0.0f, 1.0f) < 0.5f ? a.width : b.width;
    child.num_heads = uniform_float(0.0f, 1.0f) < 0.5f ? a.num_heads : b.num_heads;
    child.ffn_ratio = uniform_float(0.0f, 1.0f) < 0.5f ? a.ffn_ratio : b.ffn_ratio;
    child.alive = true;
    return child;
}

HPConfig HPOEngine::mutate_config(const HPConfig& config, float mutation_rate) {
    HPConfig m = config;
    m.id = next_config_id_++;

    auto maybe_mutate = [&](float& val, float lo, float hi, float scale) {
        if (uniform_float(0.0f, 1.0f) < mutation_rate) {
            val *= std::pow(2.0f, uniform_float(-scale, scale));
            val = std::max(lo, std::min(hi, val));
        }
    };

    maybe_mutate(m.lr, space_.lr_min, space_.lr_max, 0.5f);
    maybe_mutate(m.weight_decay, space_.wd_min, space_.wd_max, 0.5f);
    maybe_mutate(m.dropout, space_.dropout_min, space_.dropout_max, 0.2f);
    maybe_mutate(m.ffn_ratio, space_.ffn_ratio_min, space_.ffn_ratio_max, 0.3f);

    if (uniform_float(0.0f, 1.0f) < mutation_rate) {
        m.depth = std::max(space_.depth_min, std::min(space_.depth_max, m.depth + uniform_int(-1, 1)));
    }
    if (uniform_float(0.0f, 1.0f) < mutation_rate) {
        m.width = std::max(space_.width_min, std::min(space_.width_max, m.width + uniform_int(-64, 64)));
        while (m.width % 32 != 0) m.width += (m.width < 768) ? 1 : -1;
    }
    if (uniform_float(0.0f, 1.0f) < mutation_rate) {
        m.num_heads = std::max(space_.heads_min, std::min(space_.heads_max, m.num_heads + uniform_int(-1, 1)));
        while (m.width % m.num_heads != 0 && m.num_heads > 1) m.num_heads--;
    }

    m.alive = true;
    return m;
}

// ========================================================================
// Optimize
// ========================================================================

HPConfig HPOEngine::optimize(int n_budget, const std::string& method) {
    if (method == "bayesian") {
        auto results = bayesian_optimization(5, n_budget - 5);
        return get_best();
    }
    if (method == "halving") {
        auto results = successive_halving(std::min(n_budget, 27), 4);
        return get_best();
    }
    population_based_training(std::min(n_budget, 16), std::max(1, n_budget / 4));
    return get_best();
}

void HPOEngine::record_eval(const HPConfig& config, float score) {
    eval_history_.push_back({config, score});
}

std::vector<std::pair<HPConfig, float>> HPOEngine::get_eval_history() const {
    return eval_history_;
}

HPConfig HPOEngine::get_best() const {
    HPConfig best;
    best.score = -1e10f;
    for (auto& [config, score] : eval_history_) {
        if (score > best.score) {
            best = config;
            best.score = score;
        }
    }
    return best;
}

std::vector<HPConfig> HPOEngine::get_top_k(int k) const {
    std::vector<std::pair<float, int>> sorted;
    for (int i = 0; i < (int)eval_history_.size(); i++) {
        sorted.push_back({eval_history_[i].second, i});
    }
    std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.first > b.first; });

    std::vector<HPConfig> top;
    for (int i = 0; i < std::min(k, (int)sorted.size()); i++) {
        top.push_back(eval_history_[sorted[i].second].first);
    }
    return top;
}

// ========================================================================
// Save/Load
// ========================================================================

bool HPOEngine::save_best_configs(const std::string& path, int top_k) const {
    auto top = const_cast<HPOEngine*>(this)->get_top_k(top_k);
    std::ofstream ofs(path);
    if (!ofs) return false;

    ofs << "{\n  \"configs\": [\n";
    for (size_t i = 0; i < top.size(); i++) {
        auto& c = top[i];
        ofs << "    {\n";
        ofs << "      \"id\": " << c.id << ",\n";
        ofs << "      \"lr\": " << c.lr << ",\n";
        ofs << "      \"weight_decay\": " << c.weight_decay << ",\n";
        ofs << "      \"beta1\": " << c.beta1 << ",\n";
        ofs << "      \"warmup_steps\": " << c.warmup_steps << ",\n";
        ofs << "      \"batch_size\": " << c.batch_size << ",\n";
        ofs << "      \"dropout\": " << c.dropout << ",\n";
        ofs << "      \"depth\": " << c.depth << ",\n";
        ofs << "      \"width\": " << c.width << ",\n";
        ofs << "      \"num_heads\": " << c.num_heads << ",\n";
        ofs << "      \"ffn_ratio\": " << c.ffn_ratio << ",\n";
        ofs << "      \"score\": " << c.score << "\n";
        ofs << "    }" << (i + 1 < top.size() ? "," : "") << "\n";
    }
    ofs << "  ]\n}\n";
    return true;
}

std::vector<HPConfig> HPOEngine::load_configs(const std::string& path) const {
    std::vector<HPConfig> configs;
    std::ifstream ifs(path);
    if (!ifs) return configs;

    std::string line;
    HPConfig current;
    bool in_config = false;

    while (std::getline(ifs, line)) {
        if (line.find("\"id\":") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos) {
                current = HPConfig();
                in_config = true;
            }
        }
        if (!in_config) continue;

        auto extract_float = [&](const std::string& key, float& val) {
            auto pos = line.find("\"" + key + "\":");
            if (pos != std::string::npos) {
                pos = line.find(':', pos) + 1;
                while (pos < line.size() && line[pos] == ' ') pos++;
                try { val = std::stof(line.substr(pos)); } catch (...) { std::fprintf(stderr, "[WARN] Exception caught: %s (stof parse)\n", __func__); }
            }
        };
        auto extract_int = [&](const std::string& key, int& val) {
            auto pos = line.find("\"" + key + "\":");
            if (pos != std::string::npos) {
                pos = line.find(':', pos) + 1;
                while (pos < line.size() && line[pos] == ' ') pos++;
                try { val = std::stoi(line.substr(pos)); } catch (...) { std::fprintf(stderr, "[WARN] Exception caught: %s (stoi parse)\n", __func__); }
            }
        };
        auto extract_int64 = [&](const std::string& key, int64_t& val) {
            auto pos = line.find("\"" + key + "\":");
            if (pos != std::string::npos) {
                pos = line.find(':', pos) + 1;
                while (pos < line.size() && line[pos] == ' ') pos++;
                try { val = (int64_t)std::stoll(line.substr(pos)); } catch (...) { std::fprintf(stderr, "[WARN] Exception caught: %s (stoll parse)\n", __func__); }
            }
        };

        extract_float("lr", current.lr);
        extract_float("weight_decay", current.weight_decay);
        extract_float("beta1", current.beta1);
        extract_int64("warmup_steps", current.warmup_steps);
        extract_int64("batch_size", current.batch_size);
        extract_float("dropout", current.dropout);
        extract_int("depth", current.depth);
        extract_int("width", current.width);
        extract_int("num_heads", current.num_heads);
        extract_float("ffn_ratio", current.ffn_ratio);
        extract_float("score", current.score);

        if (line.find("}") != std::string::npos && in_config) {
            current.alive = true;
            configs.push_back(current);
            in_config = false;
        }
    }

    return configs;
}

int64_t HPOEngine::estimate_param_count(int depth, int width, int ffn_size, int vocab_size) const {
    int64_t attn = (int64_t)depth * 4 * (int64_t)width * (int64_t)width;
    int64_t ffn = (int64_t)depth * 2 * (int64_t)width * (int64_t)ffn_size;
    int64_t embed = (int64_t)vocab_size * (int64_t)width;
    return attn + ffn + embed;
}

int64_t HPOEngine::estimate_memory_bytes(int64_t params) const {
    return params * 4;
}

double HPOEngine::estimate_latency_ms(int depth, int width, int num_heads) const {
    double flops_per_layer = 2.0 * width * width * 4;
    double total_flops = flops_per_layer * depth;
    double giga_flops = 1e9;
    return (total_flops / giga_flops) * 1000.0;
}

// ========================================================================
// NASEngine
// ========================================================================

NASEngine::NASEngine(Model* model) : model_(model) {}

NASEngine::~NASEngine() {}

void NASEngine::set_search_config(const NASConfig& config) {
    config_ = config;
}

void NASEngine::set_hardware_constraints(const HardwareConstraints& hc) {
    hw_constraints_ = hc;
}

void NASEngine::set_model(Model* model) {
    model_ = model;
}

int64_t NASEngine::compute_param_count(const Architecture& arch) const {
    int64_t attn = (int64_t)arch.num_layers * 4 * (int64_t)arch.hidden_size * (int64_t)arch.hidden_size;
    int64_t ffn = (int64_t)arch.num_layers * 2 * (int64_t)arch.hidden_size * (int64_t)arch.ffn_size;
    int64_t embed = (int64_t)arch.vocab_size * (int64_t)arch.hidden_size;
    return attn + ffn + embed;
}

Architecture NASEngine::random_architecture() {
    Architecture a;
    a.id = next_arch_id_++;
    a.num_layers = uniform_int(2, 32);
    a.hidden_size = uniform_int(128, 2048);
    a.hidden_size = (a.hidden_size / 64) * 64;
    a.num_heads = uniform_int(1, std::min(32, a.hidden_size / 64));
    while (a.hidden_size % a.num_heads != 0 && a.num_heads > 1) a.num_heads--;
    a.ffn_size = a.hidden_size * uniform_int(2, 8);
    a.vocab_size = 32000;
    a.dropout = uniform_float(0.0f, 0.3f);
    a.param_count = compute_param_count(a);
    a.memory_bytes = a.param_count * 4;
    a.alive = true;
    return a;
}

std::vector<Architecture> NASEngine::initialize_population(int n) {
    std::vector<Architecture> pop;
    for (int i = 0; i < n; i++) {
        Architecture a = random_architecture();
        a.score = evaluate(a);
        pop.push_back(a);
        history_.push_back(a);
    }
    return pop;
}

bool NASEngine::is_valid(const Architecture& arch) const {
    if (arch.num_layers < 1 || arch.num_layers > 64) return false;
    if (arch.hidden_size < 32 || arch.hidden_size > 8192) return false;
    if (arch.num_heads < 1 || arch.num_heads > arch.hidden_size) return false;
    if (arch.hidden_size % arch.num_heads != 0) return false;
    if (arch.ffn_size < arch.hidden_size || arch.ffn_size > arch.hidden_size * 16) return false;
    if (arch.dropout < 0.0f || arch.dropout > 1.0f) return false;
    return true;
}

NASMutateOp NASEngine::random_mutation_op() const {
    NASMutateOp op;
    op.type = (NASMutateOp::Type)(uniform_int(0, 6));
    op.magnitude = uniform_float(0.05f, 0.5f);
    return op;
}

Architecture NASEngine::mutate(const Architecture& arch) {
    Architecture m = arch;
    m.id = next_arch_id_++;

    NASMutateOp op = random_mutation_op();
    switch (op.type) {
        case NASMutateOp::ADD_LAYER:
            m = mutate_add_layer(arch);
            break;
        case NASMutateOp::REMOVE_LAYER:
            m = mutate_remove_layer(arch);
            break;
        case NASMutateOp::CHANGE_WIDTH:
            m = mutate_change_width(arch);
            break;
        case NASMutateOp::CHANGE_HEADS:
            m = mutate_change_heads(arch);
            break;
        case NASMutateOp::CHANGE_FFN:
            m = mutate_change_ffn(arch);
            break;
        case NASMutateOp::CHANGE_DROPOUT:
            m = mutate_change_dropout(arch);
            break;
        case NASMutateOp::CHANGE_VOCAB:
            m.vocab_size = std::max(1000, std::min(100000, m.vocab_size + uniform_int(-8000, 8000)));
            break;
    }

    if (!is_valid(m)) return arch;
    m.param_count = compute_param_count(m);
    m.memory_bytes = m.param_count * 4;
    return m;
}

Architecture NASEngine::mutate_add_layer(const Architecture& arch) {
    Architecture m = arch;
    m.id = next_arch_id_++;
    m.num_layers = std::min(64, m.num_layers + 1);
    return m;
}

Architecture NASEngine::mutate_remove_layer(const Architecture& arch) {
    Architecture m = arch;
    m.id = next_arch_id_++;
    m.num_layers = std::max(1, m.num_layers - 1);
    return m;
}

Architecture NASEngine::mutate_change_width(const Architecture& arch) {
    Architecture m = arch;
    m.id = next_arch_id_++;
    int delta = uniform_int(-128, 128);
    m.hidden_size = std::max(64, std::min(4096, m.hidden_size + delta));
    m.hidden_size = (m.hidden_size / 64) * 64;
    if (m.hidden_size < 64) m.hidden_size = 64;
    while (m.hidden_size % m.num_heads != 0 && m.num_heads > 1) m.num_heads--;
    m.ffn_size = m.hidden_size * std::max(2, m.ffn_size / std::max(1, arch.hidden_size));
    return m;
}

Architecture NASEngine::mutate_change_heads(const Architecture& arch) {
    Architecture m = arch;
    m.id = next_arch_id_++;
    m.num_heads = std::max(1, std::min(64, m.num_heads + uniform_int(-4, 4)));
    while (m.hidden_size % m.num_heads != 0 && m.num_heads > 1) m.num_heads--;
    return m;
}

Architecture NASEngine::mutate_change_ffn(const Architecture& arch) {
    Architecture m = arch;
    m.id = next_arch_id_++;
    int ratio = std::max(2, std::min(8, m.ffn_size / std::max(1, m.hidden_size) + uniform_int(-1, 1)));
    m.ffn_size = m.hidden_size * ratio;
    return m;
}

Architecture NASEngine::mutate_change_dropout(const Architecture& arch) {
    Architecture m = arch;
    m.id = next_arch_id_++;
    m.dropout = std::max(0.0f, std::min(0.5f, m.dropout + uniform_float(-0.1f, 0.1f)));
    return m;
}

Architecture NASEngine::crossover(const Architecture& a, const Architecture& b) {
    Architecture child;
    child.id = next_arch_id_++;
    child.num_layers = uniform_float(0.0f, 1.0f) < 0.5f ? a.num_layers : b.num_layers;
    child.hidden_size = uniform_float(0.0f, 1.0f) < 0.5f ? a.hidden_size : b.hidden_size;
    child.num_heads = uniform_float(0.0f, 1.0f) < 0.5f ? a.num_heads : b.num_heads;
    child.ffn_size = uniform_float(0.0f, 1.0f) < 0.5f ? a.ffn_size : b.ffn_size;
    child.vocab_size = uniform_float(0.0f, 1.0f) < 0.5f ? a.vocab_size : b.vocab_size;
    child.dropout = uniform_float(0.0f, 1.0f) < 0.5f ? a.dropout : b.dropout;
    child.hidden_size = (child.hidden_size / 64) * 64;
    if (child.hidden_size < 64) child.hidden_size = 64;
    while (child.hidden_size % child.num_heads != 0 && child.num_heads > 1) child.num_heads--;
    child.param_count = compute_param_count(child);
    child.memory_bytes = child.param_count * 4;
    child.alive = true;
    return child;
}

float NASEngine::evaluate(const Architecture& arch) {
    if (!is_valid(arch)) return 0.0f;
    float fitness = compute_fitness(arch);

    if (config_.hardware_aware) {
        float hw_score = hardware_aware_score(arch);
        fitness *= hw_score;
    }

    if (config_.quant_native) {
        float penalty = quant_format_penalty(arch);
        fitness *= penalty;
    }

    return fitness;
}

float NASEngine::compute_fitness(const Architecture& arch) const {
    float param_score = 1.0f / (1.0f + std::log10((float)arch.param_count / 1e6f + 1.0f));
    float depth_score = std::min(1.0f, (float)arch.num_layers / 12.0f);
    float width_score = std::min(1.0f, (float)arch.hidden_size / 768.0f);
    float ratio_score = 1.0f - std::abs((float)arch.ffn_size / (float)arch.hidden_size - 4.0f) / 4.0f;
    ratio_score = std::max(0.0f, ratio_score);
    float dropout_score = arch.dropout < 0.3f ? 1.0f : 1.0f - (arch.dropout - 0.3f);

    return param_score * 0.2f + depth_score * 0.2f + width_score * 0.25f +
           ratio_score * 0.2f + dropout_score * 0.15f;
}

float NASEngine::hardware_aware_score(const Architecture& arch) {
    if (!hw_constraints_.enabled) return 1.0f;

    float score = 1.0f;

    double latency = static_cast<double>(estimate_latency_cycles(arch.num_layers, arch.hidden_size, arch.num_heads));
    double latency_ms = (double)latency / 1e6;
    if (latency_ms > hw_constraints_.target_latency_ms) {
        score *= static_cast<float>(hw_constraints_.target_latency_ms / latency_ms);
    }

    if (arch.memory_bytes > hw_constraints_.target_memory_bytes) {
        score *= (float)hw_constraints_.target_memory_bytes / (float)arch.memory_bytes;
    }

    if (arch.param_count > hw_constraints_.max_params) {
        score *= (float)hw_constraints_.max_params / (float)arch.param_count;
    }

    return std::max(0.01f, score);
}

float NASEngine::quant_format_penalty(const Architecture& arch) const {
    float penalty = 1.0f;
    if (arch.hidden_size % 64 != 0) penalty *= 0.9f;
    if (arch.ffn_size % 64 != 0) penalty *= 0.9f;
    if (arch.num_heads <= 4) penalty *= 1.05f;
    return penalty;
}

int64_t NASEngine::estimate_latency_cycles(int depth, int width, int num_heads) const {
    int64_t per_layer = 2LL * width * width * 4;
    return per_layer * depth;
}

int64_t NASEngine::estimate_memory(int depth, int width, int ffn_size) const {
    int64_t params = (int64_t)depth * 6 * (int64_t)width * (int64_t)width + (int64_t)depth * 2 * (int64_t)width * (int64_t)ffn_size;
    return params * 4;
}

Architecture NASEngine::tournament_select(const std::vector<Architecture>& pop) {
    int best = uniform_int(0, (int)pop.size() - 1);
    for (int i = 1; i < config_.tournament_size && i < (int)pop.size(); i++) {
        int idx = uniform_int(0, (int)pop.size() - 1);
        if (pop[idx].score > pop[best].score) best = idx;
    }
    return pop[best];
}

Architecture NASEngine::search(int population, int generations) {
    std::vector<Architecture> pop = initialize_population(population);

    for (int gen = 0; gen < generations; gen++) {
        std::sort(pop.begin(), pop.end(),
                  [](const Architecture& a, const Architecture& b) { return a.score > b.score; });

        int keep = std::max(2, population / 3);
        std::vector<Architecture> new_pop(pop.begin(), pop.begin() + keep);

        while ((int)new_pop.size() < population) {
            Architecture parent1 = tournament_select(pop);
            Architecture parent2 = tournament_select(pop);
            Architecture child;

            if (uniform_float(0.0f, 1.0f) < config_.crossover_rate) {
                child = crossover(parent1, parent2);
            } else {
                child = parent1;
                child.id = next_arch_id_++;
            }

            if (uniform_float(0.0f, 1.0f) < config_.mutation_rate) {
                child = mutate(child);
            }

            child.score = evaluate(child);
            new_pop.push_back(child);
            history_.push_back(child);
        }

        pop = new_pop;
    }

    std::sort(pop.begin(), pop.end(),
              [](const Architecture& a, const Architecture& b) { return a.score > b.score; });
    return pop.empty() ? random_architecture() : pop[0];
}

// ========================================================================
// DARTS
// ========================================================================

Architecture NASEngine::darts_search(int n_iterations) {
    Architecture best = random_architecture();
    best.score = evaluate(best);

    for (int iter = 0; iter < n_iterations; iter++) {
        Architecture candidate = mutate(best);
        float candidate_score = darts_evaluate(candidate);
        if (candidate_score > best.score) {
            best = candidate;
            best.score = candidate_score;
        }
        best = darts_migrate(best, 0.1f);
        best.score = darts_evaluate(best);
        history_.push_back(best);
    }
    return best;
}

float NASEngine::darts_evaluate(const Architecture& arch) {
    return evaluate(arch);
}

Architecture NASEngine::darts_migrate(const Architecture& arch, float alpha) {
    Architecture m = arch;
    m.id = next_arch_id_++;

    float delta_f = alpha * (uniform_float(0.0f, 1.0f) - 0.5f);
    m.num_layers = std::max(1, std::min(64, (int)((float)m.num_layers * (1.0f + delta_f))));

    delta_f = alpha * (uniform_float(0.0f, 1.0f) - 0.5f);
    m.hidden_size = std::max(64, std::min(4096, (int)((float)m.hidden_size * (1.0f + delta_f))));
    m.hidden_size = (m.hidden_size / 64) * 64;

    delta_f = alpha * (uniform_float(0.0f, 1.0f) - 0.5f);
    m.ffn_size = std::max(m.hidden_size * 2, std::min(m.hidden_size * 16, (int)((float)m.ffn_size * (1.0f + delta_f))));

    while (m.hidden_size % m.num_heads != 0 && m.num_heads > 1) m.num_heads--;

    if (!is_valid(m)) return arch;
    m.param_count = compute_param_count(m);
    m.memory_bytes = m.param_count * 4;
    return m;
}

// ========================================================================
// Pareto / record
// ========================================================================

void NASEngine::record_architecture(const Architecture& arch) {
    history_.push_back(arch);
}

Architecture NASEngine::get_best_architecture() const {
    Architecture best;
    best.score = -1e10f;
    for (auto& a : history_) {
        if (a.score > best.score) best = a;
    }
    return best;
}

std::vector<Architecture> NASEngine::get_pareto_front() const {
    std::vector<Architecture> front;
    for (auto& a : history_) {
        bool dominated = false;
        for (auto& b : history_) {
            if (&a == &b) continue;
            if (b.score >= a.score && b.param_count <= a.param_count &&
                b.memory_bytes <= a.memory_bytes && b.latency_ms <= a.latency_ms) {
                if (b.score > a.score || b.param_count < a.param_count ||
                    b.memory_bytes < a.memory_bytes) {
                    dominated = true;
                    break;
                }
            }
        }
        if (!dominated) front.push_back(a);
    }
    std::sort(front.begin(), front.end(),
              [](const Architecture& a, const Architecture& b) { return a.score > b.score; });
    return front;
}

bool NASEngine::save_architectures(const std::string& path, int top_k) const {
    std::vector<std::pair<float, int>> sorted;
    for (int i = 0; i < (int)history_.size(); i++) {
        sorted.push_back({history_[i].score, i});
    }
    std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.first > b.first; });

    std::ofstream ofs(path);
    if (!ofs) return false;

    int n = std::min(top_k, (int)sorted.size());
    ofs << "{\n  \"architectures\": [\n";
    for (int i = 0; i < n; i++) {
        auto& a = history_[sorted[i].second];
        ofs << "    {\n";
        ofs << "      \"id\": " << a.id << ",\n";
        ofs << "      \"num_layers\": " << a.num_layers << ",\n";
        ofs << "      \"hidden_size\": " << a.hidden_size << ",\n";
        ofs << "      \"num_heads\": " << a.num_heads << ",\n";
        ofs << "      \"ffn_size\": " << a.ffn_size << ",\n";
        ofs << "      \"vocab_size\": " << a.vocab_size << ",\n";
        ofs << "      \"dropout\": " << a.dropout << ",\n";
        ofs << "      \"param_count\": " << a.param_count << ",\n";
        ofs << "      \"score\": " << a.score << "\n";
        ofs << "    }" << (i + 1 < n ? "," : "") << "\n";
    }
    ofs << "  ]\n}\n";
    return true;
}

std::vector<Architecture> NASEngine::load_architectures(const std::string& path) const {
    std::vector<Architecture> archs;
    std::ifstream ifs(path);
    if (!ifs) return archs;

    std::string line;
    Architecture current;
    bool in_arch = false;

    while (std::getline(ifs, line)) {
        if (line.find("\"num_layers\":") != std::string::npos) {
            in_arch = true;
            current = Architecture();
        }
        if (!in_arch) continue;

        auto extract_int = [&](const std::string& key, int& val) {
            auto pos = line.find("\"" + key + "\":");
            if (pos != std::string::npos) {
                pos = line.find(':', pos) + 1;
                while (pos < line.size() && line[pos] == ' ') pos++;
                try { val = std::stoi(line.substr(pos)); } catch (...) { std::fprintf(stderr, "[WARN] Exception caught: %s (stoi parse)\n", __func__); }
            }
        };
        auto extract_int64 = [&](const std::string& key, int64_t& val) {
            auto pos = line.find("\"" + key + "\":");
            if (pos != std::string::npos) {
                pos = line.find(':', pos) + 1;
                while (pos < line.size() && line[pos] == ' ') pos++;
                try { val = (int64_t)std::stoll(line.substr(pos)); } catch (...) { std::fprintf(stderr, "[WARN] Exception caught: %s (stoll parse)\n", __func__); }
            }
        };
        auto extract_float = [&](const std::string& key, float& val) {
            auto pos = line.find("\"" + key + "\":");
            if (pos != std::string::npos) {
                pos = line.find(':', pos) + 1;
                while (pos < line.size() && line[pos] == ' ') pos++;
                try { val = std::stof(line.substr(pos)); } catch (...) { std::fprintf(stderr, "[WARN] Exception caught: %s (stof parse)\n", __func__); }
            }
        };

        extract_int("num_layers", current.num_layers);
        extract_int("hidden_size", current.hidden_size);
        extract_int("num_heads", current.num_heads);
        extract_int("ffn_size", current.ffn_size);
        extract_int("vocab_size", current.vocab_size);
        extract_float("dropout", current.dropout);
        extract_int64("param_count", current.param_count);
        extract_float("score", current.score);

        if (line.find("}") != std::string::npos && in_arch) {
            current.memory_bytes = current.param_count * 4;
            current.alive = true;
            archs.push_back(current);
            in_arch = false;
        }
    }
    return archs;
}

} // namespace hpo_nas
} // namespace quant
