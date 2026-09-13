#include "quant/model.h"
#include "quant/qwen35_tokenizer.h"
#include "quant/generator.h"
#include "quant/random.h"
#include "inference.h"

#include <iostream>
#include <string>
#include <filesystem>
#include <cstring>
#include <cerrno>
#include <cstdlib>
#include <cstdint>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: quant_infer <model.quant> [prompt] [--model-dir DIR]" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];
    std::string prompt = "Hello, ";
    if (argc > 2 && argv[2][0] != '-') prompt = argv[2];

    // Parse optional --model-dir (defaults to parent of model.quant)
    std::string model_dir;
    uint64_t cli_seed = 0;
    for (int i = 2; i < argc; i++) {
        if (std::strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            // BUGFIX (bug census): bare strtoull (garbage → seed) + unchecked
            // _putenv_s/setenv (seed env silently not applied). Validate both.
            const char* sval = argv[++i];
            char* end = nullptr;
            errno = 0;
            unsigned long long v = std::strtoull(sval, &end, 10);
            if (errno != 0 || !end || *end != '\0') {
                std::cerr << "Error: --seed needs a non-negative integer, got '"
                          << sval << "'\n";
                return 2;
            }
            cli_seed = (uint64_t)v;
        }
    }
    if (cli_seed != 0) {
        std::string s = std::to_string(cli_seed);
#ifdef _WIN32
        if (_putenv_s("QUANT_SEED", s.c_str()) != 0) {
            std::cerr << "Error: failed to set QUANT_SEED env\n";
            return 2;
        }
#else
        if (setenv("QUANT_SEED", s.c_str(), 1) != 0) {
            std::cerr << "Error: failed to set QUANT_SEED env\n";
            return 2;
        }
#endif
    }
    {
        uint64_t run_seed = quant::resolve_base_seed(cli_seed);
        const char* env_seed = std::getenv("QUANT_SEED");
        std::cerr << "[seed] base=" << run_seed
                  << " (cli=" << cli_seed
                  << " env=" << (env_seed ? env_seed : "-") << ")" << std::endl;
    }
    if (model_dir.empty()) {
        // BUGFIX (bug census): bare filename → empty parent_path → "" dir →
        // tokenizer silently searched "". Default to "." like evaluate.cpp.
        model_dir = std::filesystem::path(model_path).parent_path().string();
        if (model_dir.empty()) model_dir = ".";
    }

    quant::DenseModel model;
    try {
        model.load(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }

    quant::Qwen35Tokenizer tokenizer;
    try {
        if (!tokenizer.load_from_dir(model_dir)) {
            std::cerr << "Error: failed to load tokenizer from " << model_dir << std::endl;
            return 1;
        }
        std::cerr << "Tokenizer loaded (" << tokenizer.vocab_size() << " vocab)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading tokenizer: " << e.what() << std::endl;
        return 1;
    }

    quant::Generator gen(&model, &tokenizer);

    quant::SamplerConfig cfg;
    cfg.temperature = 0.7f;
    cfg.top_k = 40;
    cfg.top_p = 0.9f;
    cfg.max_tokens = 512;

    auto result = gen.generate_full(prompt, cfg);
    std::cout << result.text << std::endl;
    std::cerr << "Generated " << result.tokens_per_sec << " tok/s"
              << " (" << result.duration_sec << "s)"
              << std::endl;

    return 0;
}
