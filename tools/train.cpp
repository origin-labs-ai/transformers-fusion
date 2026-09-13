#include "quant/model.h"
#include "quant/transformer.h"
#include "quant/tokenizer.h"
#include "quant/trainer.h"
#include "quant/optimizer.h"
#include "quant/random.h"

#include "quant/detail/cli_parse.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <sstream>

struct TrainArgs {
    std::string model_path = "model.quant";
    std::string data_path = "data.txt";
    std::string config_path;
    int64_t batch_size = 8;
    int64_t seq_length = 512;
    int num_epochs = 3;
    float learning_rate = 3e-4f;
    int vocab_size = 32000;
    int hidden_size = 768;
    int num_layers = 12;
    int num_heads = 12;
    int log_interval = 10;
    int save_interval = 1000;
    std::string optimizer_name = "adafactor";
    uint64_t seed = 0;
};

static TrainArgs parse_args(int argc, char** argv) {
    TrainArgs args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            args.model_path = argv[++i];
        else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc)
            args.data_path = argv[++i];
        else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc)
            args.config_path = argv[++i];
        else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc)
            args.batch_size = quant::cli_parse::parse_ll(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--seq-length") == 0 && i + 1 < argc)
            args.seq_length = quant::cli_parse::parse_ll(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc)
            args.num_epochs = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc)
            args.learning_rate = quant::cli_parse::parse_float(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--vocab-size") == 0 && i + 1 < argc)
            args.vocab_size = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--hidden-size") == 0 && i + 1 < argc)
            args.hidden_size = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--num-layers") == 0 && i + 1 < argc)
            args.num_layers = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--num-heads") == 0 && i + 1 < argc)
            args.num_heads = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--log-interval") == 0 && i + 1 < argc)
            args.log_interval = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--save-interval") == 0 && i + 1 < argc)
            args.save_interval = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--optimizer") == 0 && i + 1 < argc)
            args.optimizer_name = argv[++i];
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            args.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: quant_train --model model.quant --data data.txt [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --batch-size N    Batch size (default: 8)\n";
            std::cout << "  --seq-length N    Sequence length (default: 512)\n";
            std::cout << "  --epochs N        Number of epochs (default: 3)\n";
            std::cout << "  --lr F            Learning rate (default: 3e-4)\n";
            std::cout << "  --vocab-size N    Vocabulary size (default: 32000)\n";
            std::cout << "  --hidden-size N   Hidden size (default: 768)\n";
            std::cout << "  --num-layers N    Number of layers (default: 12)\n";
            std::cout << "  --num-heads N     Number of heads (default: 12)\n";
            std::cout << "  --optimizer NAME  Optimizer: 'adafactor' or 'adamw' (default: adafactor)\n";
            std::cout << "  --seed N          Base seed (0=default 42, or QUANT_SEED env)\n";
            exit(0);
        }
    }
    return args;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    if (args.seed != 0) {
#ifdef _WIN32
        _putenv_s("QUANT_SEED", std::to_string(args.seed).c_str());
#else
        setenv("QUANT_SEED", std::to_string(args.seed).c_str(), 1);
#endif
    }
    {
        uint64_t run_seed = quant::resolve_base_seed(args.seed);
        const char* env_seed = std::getenv("QUANT_SEED");
        std::cout << "[seed] base=" << run_seed
                  << " (cli=" << args.seed
                  << " env=" << (env_seed ? env_seed : "-") << ")" << std::endl;
    }

    std::cout << "=== QUANT Training ===\n";
    std::cout << "Model: " << args.model_path << "\n";
    std::cout << "Data: " << args.data_path << "\n";

    quant::TransformerConfig cfg;
    cfg.vocab_size = args.vocab_size;
    cfg.hidden_size = args.hidden_size;
    cfg.num_layers = args.num_layers;
    cfg.num_heads = args.num_heads;

    quant::DenseModel model(cfg);
    std::cout << "Model created: " << model.param_count() << " params\n";

    quant::BPETokenizer tokenizer;
    std::ifstream data_file(args.data_path, std::ios::binary | std::ios::ate);
    if (!data_file.is_open()) {
        std::cerr << "Error: cannot open " << args.data_path << std::endl;
        return 1;
    }
    // BUGFIX (bug census): whole file slurped into one string (OOM on large
    // corpora). Cap the slurp; bigger corpora need the streaming DataLoader
    // path (UnifiedTrainer reads data_path itself — the slurp below only
    // trains the inline BPE vocab, which needs a sample, not the corpus).
    static constexpr std::streamsize kMaxSlurp = (std::streamsize)64 << 20; // 64 MiB
    std::streamsize fsize = data_file.tellg();
    data_file.seekg(0);
    if (fsize > kMaxSlurp) {
        std::cerr << "[Warning] data file >64MiB: vocab training samples first 64MiB; "
                     "training itself streams from disk.\n";
    }
    std::stringstream ss;
    {
        char chunk[1 << 16];
        std::streamsize left = (fsize < 0 || fsize > kMaxSlurp) ? kMaxSlurp : fsize;
        while (left > 0 && data_file) {
            std::streamsize want = left < (std::streamsize)sizeof(chunk) ? left : (std::streamsize)sizeof(chunk);
            data_file.read(chunk, want);
            std::streamsize got = data_file.gcount();
            if (got <= 0) break;
            ss.write(chunk, got);
            left -= got;
        }
    }
    std::string corpus = ss.str();
    if (!args.config_path.empty()) {
        std::cout << "[Note] --config accepted but not wired into UnifiedTrainer "
                     "(hyperparams come from CLI flags).\n";
    }

    std::vector<std::string> texts = {corpus};
    tokenizer.train(texts, static_cast<int>(cfg.vocab_size));

    // L073: single entry — all training CLI paths go through UnifiedTrainer.
    quant::UnifiedTrainArgs uargs;
    uargs.kind = quant::UnifiedTrainerKind::Dense;
    uargs.batch_size = args.batch_size;
    uargs.seq_length = args.seq_length;
    uargs.num_epochs = args.num_epochs;
    uargs.learning_rate = args.learning_rate;
    uargs.log_interval = args.log_interval;
    uargs.save_interval = args.save_interval;
    uargs.output_path = args.model_path;
    uargs.data_path = args.data_path;
    uargs.optimizer_name = args.optimizer_name;

    quant::UnifiedTrainer trainer(&model, &tokenizer);
    trainer.set_log_callback([](const quant::TrainMetrics& m) {
        std::cout << "Step " << m.step
                  << " | loss: " << m.loss
                  << " | ppl: " << m.perplexity
                  << " | lr: " << m.learning_rate
                  << " | tok/s: " << m.tokens_per_sec
                  << std::endl;
    });
    std::cout << "Using " << (args.optimizer_name == "adamw" ? "AdamW" : "Adafactor")
              << " optimizer via UnifiedTrainer ("
              << quant::unified_trainer_kind_name(uargs.kind) << ").\n";
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
    std::cout << "Training complete. Model saved to " << args.model_path << std::endl;
    return 0;
}
