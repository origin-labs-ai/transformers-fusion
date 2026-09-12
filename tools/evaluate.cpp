#include "quant/model.h"
#include "quant/eval.h"
#include "quant/qwen35_tokenizer.h"
#include "quant/metrics.h"

#include "quant/detail/cli_parse.h"
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <filesystem>

struct EvalArgs {
    std::string model_path;
    std::string model_dir;
    std::string task = "all";
    std::string data_path;
    int batch_size = 1;
    int context_size = 512;
    int max_tokens = 0;
};

static EvalArgs parse_args(int argc, char** argv) {
    EvalArgs args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            args.model_path = argv[++i];
        else if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc)
            args.model_dir = argv[++i];
        else if (strcmp(argv[i], "--task") == 0 && i + 1 < argc)
            args.task = argv[++i];
        else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc)
            args.data_path = argv[++i];
        else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc)
            args.batch_size = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--context") == 0 && i + 1 < argc)
            args.context_size = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc)
            args.max_tokens = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: quant_evaluate --model model.quant --task <task> [options]\n";
            std::cout << "Tasks: perplexity, accuracy, classification, generation, hellaswag, all\n";
            std::cout << "Options:\n";
            std::cout << "  --model-dir <dir>   Model directory with tokenizer.json\n";
            std::cout << "                      (default: directory of --model)\n";
            std::cout << "  --data <path>       Eval data path\n";
            std::cout << "  --batch-size N       Batch size (default: 1)\n";
            std::cout << "  --context N          Context size (default: 512)\n";
            std::cout << "  --max-tokens N       Max eval tokens (default: all)\n";
            exit(0);
        }
    }
    return args;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    if (args.model_path.empty()) {
        std::cerr << "Error: --model required\n";
        return 1;
    }

    quant::DenseModel model;
    try {
        model.load(args.model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }

    if (args.model_dir.empty())
        args.model_dir = std::filesystem::path(args.model_path).parent_path().string();

    // Qwen3.5-family models ship tokenizer.json in the model dir; there is no
    // separate .vocab file. load_from_dir reads tokenizer.json directly.
    quant::Qwen35Tokenizer tokenizer;
    try {
        if (!tokenizer.load_from_dir(args.model_dir)) {
            std::cerr << "Error: failed to load tokenizer from " << args.model_dir
                      << " (expected tokenizer.json)" << std::endl;
            return 1;
        }
        std::cout << "Tokenizer loaded (" << tokenizer.vocab_size() << " vocab)"
                  << " from " << args.model_dir << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error loading tokenizer: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Model loaded: " << model.param_count() << " params\n";

    std::vector<int> eval_tokens;
    if (!args.data_path.empty()) {
        eval_tokens = quant::load_eval_tokens(args.data_path, args.max_tokens);
        std::cout << "Loaded " << eval_tokens.size() << " eval tokens\n";
    }

    quant::ModelEvaluator evaluator(&model);
    evaluator.set_batch_size(args.batch_size);

    if (args.task == "perplexity" || args.task == "all") {
        if (!eval_tokens.empty()) {
            auto r = evaluator.evaluate_perplexity(eval_tokens, "perplexity", args.context_size, args.context_size / 2);
            std::cout << "Perplexity: " << r.perplexity
                      << " (tokens: " << r.total_tokens << ")\n";
        }
    }

    if (args.task == "accuracy" || args.task == "all") {
        if (!eval_tokens.empty()) {
            auto r = evaluator.evaluate_accuracy(eval_tokens, "accuracy", args.context_size);
            std::cout << "Accuracy: " << r.accuracy * 100.0 << "%"
                      << " (" << r.correct << "/" << r.total << ")\n";
        }
    }

    if (args.task == "generation" || args.task == "all") {
        auto r = evaluator.evaluate_generation(args.context_size, 128, "generation");
        std::cout << "Generation speed: " << r.tokens_per_sec << " tok/s\n";
    }

    if (args.task == "classification" || args.task == "all") {
        if (!eval_tokens.empty() && eval_tokens.size() > 1) {
            int64_t V = model.vocab_size();
            int64_t n = (int64_t)eval_tokens.size() - 1;
            quant::Tensor logits(quant::Shape{n, V}, quant::DType::F32);
            std::vector<int> labels((size_t)n);
            for (int64_t i = 0; i < n; i++) {
                labels[(size_t)i] = eval_tokens[(size_t)(i + 1)];
                quant::Tensor input({1, 1});
                quant::Tensor pos({1, 1});
                input.data<float>()[0] = (float)eval_tokens[(size_t)i];
                pos.data<float>()[0] = 0;
                quant::Tensor out = model.forward(input, pos);
                std::memcpy(logits.data<float>() + i * V, out.data<float>(), (size_t)V * sizeof(float));
            }
            auto cr = quant::eval_classification(logits, labels);
            std::cout << "Classification accuracy: " << cr.accuracy * 100.0 << "%"
                      << " f1: " << cr.f1_score
                      << " precision: " << cr.precision
                      << " recall: " << cr.recall << "\n";
        }
    }

    if (args.task == "hellaswag" || args.task == "all") {
        std::cout << "HellaSwag eval requires context/ending data -- skipped\n";
    }

    std::cout << "Evaluation complete.\n";
    return 0;
}
