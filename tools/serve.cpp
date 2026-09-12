#include "quant/production.h"
#include "quant/model.h"
#include "quant/tokenizer.h"
#include "quant/generator.h"

#include "quant/detail/cli_parse.h"
#include <iostream>
#include <string>
#include <cstring>
#include <thread>
#include <atomic>
#include <chrono>
#include <sstream>
#include <fstream>
#include <csignal>

struct ServeArgs {
    std::string model_path;
    int port = 8080;
    int batch_size = 1;
    int num_workers = 4;
    int max_tokens = 512;
};

static ServeArgs parse_args(int argc, char** argv) {
    ServeArgs args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            args.model_path = argv[++i];
        else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            args.port = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc)
            args.batch_size = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc)
            args.num_workers = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc)
            args.max_tokens = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: quant_serve --model model.quant [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --port N           Server port (default: 8080)\n";
            std::cout << "  --batch-size N     Batch size (default: 1)\n";
            std::cout << "  --workers N        Thread pool size (default: 4)\n";
            std::cout << "  --max-tokens N     Max generation tokens (default: 512)\n";
            exit(0);
        }
    }
    return args;
}

static std::atomic<bool> g_running{true};

static void signal_handler(int) {
    g_running = false;
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

    quant::BPETokenizer tokenizer;
    std::string vocab_path = args.model_path;
    size_t dot = vocab_path.rfind('.');
    if (dot != std::string::npos)
        vocab_path = vocab_path.substr(0, dot);
    vocab_path += ".vocab";
    try {
        std::ifstream f(vocab_path);
        if (f.is_open()) {
            f.close();
            tokenizer.load(vocab_path);
        }
    } catch (const std::exception& e) {
        std::cerr << "[Warning] Optional tokenizer load failed: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[Warning] Optional tokenizer load failed with unknown exception." << std::endl;
    }

    std::cout << "Model loaded: " << model.param_count() << " params\n";
    std::cout << "Starting server on port " << args.port << "...\n";

    quant::ModelHTTPServer http_server(&model, args.port);

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    http_server.start();

    std::cout << "Server running on http://localhost:" << args.port << "\n";
    std::cout << "Endpoints:\n";
    std::cout << "  POST /v1/completions  - Generate text\n";
    std::cout << "  GET  /health          - Health check\n";
    std::cout << "  GET  /v1/models       - List models\n";
    std::cout << "Press Ctrl+C to stop.\n";

    while (g_running && http_server.is_running()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    http_server.stop();
    std::cout << "Server stopped.\n";
    return 0;
}
