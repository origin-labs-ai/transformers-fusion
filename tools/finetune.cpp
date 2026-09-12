#include "quant/model.h"
#include "quant/tokenizer.h"
#include "quant/trainer.h"
#include "quant/finetune.h"

#include "quant/detail/cli_parse.h"
#include <iostream>
#include <string>
#include <cstring>

struct FTArgs {
    std::string model_path;
    std::string data_path;
    std::string output_path = "finetuned.quant";
    float learning_rate = 1e-5f;
    int num_epochs = 1;
    int batch_size = 4;
    int seq_length = 512;
    int log_interval = 10;
    int save_interval = 100;
    std::string optimizer_name = "adafactor";
};

static FTArgs parse_args(int argc, char** argv) {
    FTArgs args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            args.model_path = argv[++i];
        else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc)
            args.data_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            args.output_path = argv[++i];
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc)
            args.learning_rate = quant::cli_parse::parse_float(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc)
            args.num_epochs = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc)
            args.batch_size = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--seq-length") == 0 && i + 1 < argc)
            args.seq_length = static_cast<int>(quant::cli_parse::parse_ll(argv[i-1], argv[++i]));
        else if (strcmp(argv[i], "--optimizer") == 0 && i + 1 < argc)
            args.optimizer_name = argv[++i];
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: quant_finetune --model base.quant --data finetune.txt --output finetuned.quant\n";
            std::cout << "  --optimizer NAME  Optimizer: 'adafactor' or 'adamw' (default: adafactor)\n";
            exit(0);
        }
    }
    return args;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    if (args.model_path.empty() || args.data_path.empty()) {
        std::cerr << "Error: --model and --data are required\n";
        return 1;
    }

    std::cout << "=== QUANT Fine-Tuning ===\n";
    std::cout << "Base model: " << args.model_path << "\n";
    std::cout << "Data: " << args.data_path << "\n";
    std::cout << "Output: " << args.output_path << "\n";

    quant::DenseModel model;
    try {
        model.load(args.model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }

    quant::BPETokenizer tokenizer;

    // L073: single entry — finetune CLI delegates to UnifiedTrainer
    // (FineTuner::configure field-maps 1:1 to UnifiedTrainArgs; FineTuner
    // remains for API compat, CLI uses the unified path).
    quant::UnifiedTrainArgs uargs;
    uargs.kind = quant::UnifiedTrainerKind::FineTune;
    uargs.learning_rate = args.learning_rate;
    uargs.num_epochs = args.num_epochs;
    uargs.batch_size = args.batch_size;
    uargs.seq_length = args.seq_length;
    uargs.log_interval = args.log_interval;
    uargs.save_interval = args.save_interval;
    uargs.output_path = args.output_path;
    uargs.data_path = args.data_path;
    uargs.optimizer_name = args.optimizer_name;
    quant::UnifiedTrainer trainer(&model, &tokenizer);
    std::string cfg_err;
    if (!trainer.configure(uargs, &cfg_err)) {
        std::cerr << "Error: " << cfg_err << std::endl;
        return 1;
    }
    std::string run_err;
    if (!trainer.run(&run_err)) {
        std::cerr << "Error: " << run_err << std::endl;
        return 1;
    }

    std::cout << "Fine-tuning complete. Saved to " << args.output_path << std::endl;
    return 0;
}
