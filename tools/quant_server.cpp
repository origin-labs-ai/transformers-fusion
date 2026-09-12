#include "quant/model.h"
#include "quant/qwen35_tokenizer.h"
#include "quant/generator.h"
#include "quant/production_socket.h"
#include "quant/http_parse.h"
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
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <filesystem>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#endif

struct QuantServerArgs {
    std::string model_path;
    std::string model_dir;
    int port = 9090;
    int num_workers = 4;
    int max_tokens = 2048;
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
};

static QuantServerArgs parse_args(int argc, char** argv) {
    QuantServerArgs args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            args.model_path = argv[++i];
        else if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc)
            args.model_dir = argv[++i];
        else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            args.port = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc)
            args.num_workers = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc)
            args.max_tokens = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc)
            args.temperature = quant::cli_parse::parse_float(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc)
            args.top_k = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc)
            args.top_p = quant::cli_parse::parse_float(argv[i-1], argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: quant_server --model model.quant [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --port N           Server port (default: 9090)\n";
            std::cout << "  --model-dir DIR    Model directory with tokenizer.json\n";
            std::cout << "                     (default: directory of --model)\n";
            std::cout << "  --workers N        Thread pool size (default: 4)\n";
            std::cout << "  --max-tokens N     Max generation tokens (default: 2048)\n";
            std::cout << "  --temperature F    Sampling temperature (default: 0.7)\n";
            std::cout << "  --top-k N          Top-K sampling (default: 40)\n";
            std::cout << "  --top-p F          Top-P sampling (default: 0.9)\n";
            exit(0);
        }
    }
    return args;
}

static std::atomic<bool> g_running{true};
static void signal_handler(int) { g_running = false; }

// Simple JSON escape
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// Simple JSON object builder
static std::string json_obj(const std::map<std::string, std::string>& fields) {
    std::string out = "{";
    bool first = true;
    for (auto& [k, v] : fields) {
        if (!first) out += ",";
        first = false;
        out += "\"" + k + "\":\"" + json_escape(v) + "\"";
    }
    out += "}";
    return out;
}

static std::string json_error(const std::string& msg) {
    return "{\"error\":\"" + json_escape(msg) + "\"}";
}

class QuantHTTPServer {
public:
    QuantHTTPServer(quant::DenseModel* model, quant::Tokenizer* tokenizer, int port, int workers)
        : model_(model), tokenizer_(tokenizer), port_(port), num_workers_(workers) {}

    void start() {
        running_ = true;
        quant::socket_helpers::platform_init();

        int server_fd = (int)socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) {
            std::cerr << "Failed to create socket\n";
            return;
        }

        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

        sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons((u_short)port_);

        if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
            std::cerr << "Failed to bind to port " << port_ << "\n";
            quant::socket_helpers::close_socket(server_fd);
            quant::socket_helpers::platform_cleanup();
            return;
        }

        if (listen(server_fd, 128) < 0) {
            std::cerr << "Failed to listen\n";
            quant::socket_helpers::close_socket(server_fd);
            quant::socket_helpers::platform_cleanup();
            return;
        }

        std::cout << "quant-server listening on port " << port_ << "\n";
        std::cout << "Model: " << model_->param_count() << " params\n";

        for (int i = 0; i < num_workers_; i++) {
            workers_.emplace_back(&QuantHTTPServer::worker_loop, this, server_fd);
        }

        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }

        quant::socket_helpers::close_socket(server_fd);
        quant::socket_helpers::platform_cleanup();
    }

    void stop() { running_ = false; }

private:
    quant::DenseModel* model_;
    quant::Tokenizer* tokenizer_;
    int port_;
    int num_workers_;
    std::atomic<bool> running_{false};
    std::vector<std::thread> workers_;

    struct HTTPRequest {
        std::string method;
        std::string path;
        std::string body;
    };

    HTTPRequest parse_request(const std::string& raw) {
        HTTPRequest req;
        std::istringstream ss(raw);
        ss >> req.method >> req.path;
        auto header_end = raw.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            req.body = raw.substr(header_end + 4);
        }
        return req;
    }

    void send_response(int fd, int status, const std::string& content_type, const std::string& body, bool keep_alive = false) {
        // C-24 fix: full reason-phrase table (was 200/400/404/500/501 only;
        // 204/413/414 fell through to "Unknown"). Single source of truth:
        // quant::http_parse::status_text (see include/quant/http_parse.h).
        const std::string status_str = quant::http_parse::status_text(status);

        std::string resp = "HTTP/1.1 " + std::to_string(status) + " " + status_str + "\r\n"
            "Content-Type: " + content_type + "\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n"
            "Connection: " + std::string(keep_alive ? "keep-alive" : "close") + "\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type\r\n"
            "\r\n" + body;

        send(fd, resp.c_str(), (int)resp.size(), 0);
    }

    void send_sse_headers(int fd) {
        std::string headers =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n";
        send(fd, headers.c_str(), (int)headers.size(), 0);
    }

    void handle_generate(int fd, const HTTPRequest& req) {
        bool stream = req.body.find("\"stream\":true") != std::string::npos ||
                      req.body.find("\"stream\": true") != std::string::npos;

        std::string prompt = "Hello";
        int max_tokens = 512;
        float temperature = 0.7f;
        int top_k = 40;
        float top_p = 0.9f;

        auto extract_str = [&](const std::string& key) -> std::string {
            auto pos = req.body.find("\"" + key + "\"");
            if (pos == std::string::npos) return "";
            auto start_q = req.body.find('"', pos + key.size() + 3);
            if (start_q == std::string::npos) return "";
            auto end_q = req.body.find('"', start_q + 1);
            if (end_q == std::string::npos) return "";
            std::string val = req.body.substr(start_q + 1, end_q - start_q - 1);
            return val;
        };

        // P0 fix: stoi/stof throw invalid_argument/out_of_range on
        // crafted JSON and would kill the request thread. Clamp to def.
        auto extract_int = [&](const std::string& key, int def) -> int {
            auto pos = req.body.find("\"" + key + "\"");
            if (pos == std::string::npos) return def;
            auto colon = req.body.find(':', pos);
            if (colon == std::string::npos) return def;
            auto num_start = req.body.find_first_of("0123456789-", colon + 1);
            if (num_start == std::string::npos) return def;
            auto num_end = req.body.find_first_not_of("0123456789", num_start);
            if (num_end == std::string::npos) num_end = req.body.size();
            try {
                return std::stoi(req.body.substr(num_start, num_end - num_start));
            } catch (const std::exception&) {
                return def;
            }
        };

        auto extract_float = [&](const std::string& key, float def) -> float {
            auto pos = req.body.find("\"" + key + "\"");
            if (pos == std::string::npos) return def;
            auto colon = req.body.find(':', pos);
            if (colon == std::string::npos) return def;
            auto num_start = req.body.find_first_of("0123456789.-", colon + 1);
            if (num_start == std::string::npos) return def;
            size_t end = colon + 1;
            while (end < req.body.size() && (isdigit(req.body[end]) || req.body[end] == '.' || req.body[end] == '-')) end++;
            try {
                return std::stof(req.body.substr(num_start, end - num_start));
            } catch (const std::exception&) {
                return def;
            }
        };

        std::string p = extract_str("prompt");
        if (!p.empty()) prompt = p;
        max_tokens = extract_int("max_tokens", max_tokens);
        temperature = extract_float("temperature", temperature);
        top_k = extract_int("top_k", top_k);
        top_p = extract_float("top_p", top_p);

        max_tokens = std::min(std::max(1, max_tokens), 4096);

        if (!model_ || !tokenizer_) {
            send_response(fd, 500, "application/json", json_error("model not loaded"));
            return;
        }

        // Tokenize
        std::vector<int> tokens;
        try {
            tokens = tokenizer_->encode(prompt);
        } catch (const std::exception& e) {
            send_response(fd, 500, "application/json", json_error(std::string("tokenize: ") + e.what()));
            return;
        }
        if (tokens.empty()) tokens = {1};

        if (stream) {
            send_sse_headers(fd);

            quant::SamplerConfig scfg;
            scfg.temperature = temperature;
            scfg.top_k = top_k;
            scfg.top_p = top_p;
            scfg.max_tokens = max_tokens;

            quant::KVCache cache((int)model_->config.num_layers,
                               model_->config.max_seq_len,
                               model_->config.num_heads,
                               model_->config.head_dim);

            int count = 0;
            int last_token = 0; // P0: per-request (was racy member)
            for (int i = 0; i < max_tokens; i++) {
                try {
                    int64_t seq_len = (int64_t)(tokens.size() + i);
                    quant::Tensor input({1, seq_len < 2 ? 1 : seq_len});
                    float* id = input.data<float>();
                    if (i == 0) {
                        for (size_t j = 0; j < tokens.size(); j++) id[j] = (float)tokens[j];
                    } else {
                        // P0 fix: was shared member last_token_ (data race
                        // across worker threads). Per-request local instead.
                        id[0] = (float)last_token;
                    }
                    quant::Tensor pos({1, seq_len < 2 ? 1 : seq_len});
                    float* pd = pos.data<float>();
                    if (i == 0) {
                        for (int64_t j = 0; j < (int64_t)seq_len; j++) pd[j] = (float)j;
                    } else {
                        pd[0] = (float)(tokens.size() + i - 1);
                    }

                    quant::Tensor logits = model_->forward(input, pos, &cache);
                    if (logits.numel() == 0) break;

                    int64_t V = logits.dim(logits.rank() - 1);
                    const float* ld = logits.data<float>() + logits.numel() - V;

                    int next = 0;
                    for (int64_t v = 1; v < V; v++)
                        if (ld[v] > ld[next]) next = (int)v;

                    if (next == 2) break;
                    last_token = next;
                    count++;

                    std::string token_text = tokenizer_->decode({next});
                    std::string sse_data = "{\"token\":\"" + json_escape(token_text) + "\",\"index\":" + std::to_string(i) + "}";
                    std::string sse = "data: " + sse_data + "\n\n";
                    if (send(fd, sse.c_str(), (int)sse.size(), 0) < 0) break;
                } catch (...) {
                    std::string err = "data: {\"error\":\"inference failed\"}\n\n";
                    send(fd, err.c_str(), (int)err.size(), 0);
                    break;
                }
            }

            std::string done = "data: {\"done\":true,\"count\":" + std::to_string(count) + "}\n\n";
            send(fd, done.c_str(), (int)done.size(), 0);
        } else {
            quant::SamplerConfig scfg;
            scfg.temperature = temperature;
            scfg.top_k = top_k;
            scfg.top_p = top_p;
            scfg.max_tokens = max_tokens;

            quant::KVCache cache((int)model_->config.num_layers,
                               model_->config.max_seq_len,
                               model_->config.num_heads,
                               model_->config.head_dim);

            std::vector<int> output;
            output.reserve(max_tokens);

            for (int i = 0; i < max_tokens; i++) {
                try {
                    int64_t seq_len = (int64_t)(tokens.size() + output.size());
                    quant::Tensor input({1, seq_len < 2 ? 1 : seq_len});
                    float* id = input.data<float>();
                    if (output.empty()) {
                        for (size_t j = 0; j < tokens.size(); j++) id[j] = (float)tokens[j];
                    } else {
                        id[0] = (float)output.back();
                    }
                    quant::Tensor pos({1, seq_len < 2 ? 1 : seq_len});
                    float* pd = pos.data<float>();
                    if (output.empty()) {
                        for (int64_t j = 0; j < (int64_t)seq_len; j++) pd[j] = (float)j;
                    } else {
                        pd[0] = (float)(tokens.size() + output.size() - 1);
                    }

                    quant::Tensor logits = model_->forward(input, pos, &cache);
                    if (logits.numel() == 0) break;

                    int64_t V = logits.dim(logits.rank() - 1);
                    const float* ld = logits.data<float>() + logits.numel() - V;

                    int next = 0;
                    for (int64_t v = 1; v < V; v++)
                        if (ld[v] > ld[next]) next = (int)v;

                    if (next == 2) break;
                    output.push_back(next);
                } catch (...) {
                    break;
                }
            }

            std::string generated = output.empty() ? "" : tokenizer_->decode(output);

            std::string response = "{\"choices\":[{\"text\":\"" + json_escape(generated) +
                "\"}],\"usage\":{\"prompt_tokens\":" + std::to_string(tokens.size()) +
                ",\"completion_tokens\":" + std::to_string(output.size()) + "}}";

            send_response(fd, 200, "application/json", response);
        }
    }

    void handle_tokenize(int fd, const HTTPRequest& req) {
        if (!tokenizer_) {
            send_response(fd, 500, "application/json", json_error("tokenizer not loaded"));
            return;
        }
        std::string content;
        auto pos = req.body.find("\"content\"");
        if (pos != std::string::npos) {
            auto start_q = req.body.find('"', pos + 9);
            if (start_q != std::string::npos) {
                auto end_q = req.body.find('"', start_q + 1);
                if (end_q != std::string::npos)
                    content = req.body.substr(start_q + 1, end_q - start_q - 1);
            }
        }
        try {
            auto tokens = tokenizer_->encode(content);
            std::string arr = "[";
            for (size_t i = 0; i < tokens.size(); i++) {
                if (i > 0) arr += ",";
                arr += std::to_string(tokens[i]);
            }
            arr += "]";
            send_response(fd, 200, "application/json", "{\"tokens\":" + arr + "}");
        } catch (const std::exception& e) {
            send_response(fd, 500, "application/json", json_error(std::string("tokenize: ") + e.what()));
        }
    }

    void handle_detokenize(int fd, const HTTPRequest& req) {
        if (!tokenizer_) {
            send_response(fd, 500, "application/json", json_error("tokenizer not loaded"));
            return;
        }
        std::vector<int> tokens;
        auto pos = req.body.find("\"tokens\"");
        if (pos != std::string::npos) {
            auto start_b = req.body.find('[', pos);
            if (start_b != std::string::npos) {
                auto end_b = req.body.find(']', start_b);
                if (end_b != std::string::npos) {
                    std::string arr = req.body.substr(start_b + 1, end_b - start_b - 1);
                    std::istringstream ss(arr);
                    std::string item;
                    while (std::getline(ss, item, ',')) {
                        // P0 fix: trim + try/catch — whitespace/empty/overflow
                        // tokens threw and killed the request thread.
                        size_t a = item.find_first_not_of(" \t\r\n");
                        if (a == std::string::npos) continue;
                        size_t b = item.find_last_not_of(" \t\r\n");
                        try {
                            tokens.push_back(std::stoi(item.substr(a, b - a + 1)));
                        } catch (const std::exception&) {
                            send_response(fd, 400, "application/json",
                                          json_error("invalid token id in tokens array"));
                            return;
                        }
                    }
                }
            }
        }
        try {
            std::string text = tokenizer_->decode(tokens);
            send_response(fd, 200, "application/json", "{\"content\":\"" + json_escape(text) + "\"}");
        } catch (const std::exception& e) {
            send_response(fd, 500, "application/json", json_error(std::string("detokenize: ") + e.what()));
        }
    }

    // ---- L017: single timeout style + hard request-size caps ------------
    static constexpr int kClientTimeoutMs = 30000;
    static constexpr size_t kMaxRequestLineBytes = 8192;   // request-line cap

    static void set_client_timeout(int client_fd) {
#ifdef _WIN32
        DWORD t = (DWORD)kClientTimeoutMs;                 // Windows: DWORD ms
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&t, sizeof(t));
#else
        timeval t{kClientTimeoutMs / 1000, (kClientTimeoutMs % 1000) * 1000}; // POSIX: timeval
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &t, sizeof(t));
#endif
    }

    void handle_request(int client_fd) {
        // C-24 fix: Content-Length-aware body read (was single recv — a
        // pipelined/split POST body got truncated to the first segment).
        // Loop until headers + declared body bytes are in, capping the total
        // at 64KB (413) and the request line at 8KB (414).
        static constexpr size_t kCap = 65536;
        std::string raw;
        raw.reserve(16384);
        while (raw.size() < kCap) {
            char tmp[4096];
            int n = recv(client_fd, tmp, (int)sizeof(tmp), 0);
            if (n <= 0) break;
            raw.append(tmp, (size_t)n);
            auto hend = raw.find("\r\n\r\n");
            if (hend != std::string::npos) {
                std::string headers = raw.substr(0, hend);
                int64_t cl = quant::http_parse::parse_content_length(headers, (int64_t)kCap);
                if (cl < 0) {
                    send_response(client_fd, 400, "application/json",
                                  json_error("malformed Content-Length"));
                    return;
                }
                size_t have_body = raw.size() - hend - 4;
                if (cl > (int64_t)kCap || have_body + 4 > kCap) {
                    send_response(client_fd, 413, "application/json",
                                  json_error("request too large (header/body cap 64KB)"));
                    return;
                }
                if (have_body >= (size_t)cl) break; // full body present
                // else: keep reading until declared bytes arrive
            } else if (raw.size() >= kCap) {
                send_response(client_fd, 413, "application/json",
                              json_error("request too large (header/body cap 64KB)"));
                return;
            }
        }
        if (raw.empty()) return;

        // Request-line cap: first CRLF must arrive within 8KB.
        const size_t eol = raw.find("\r\n");
        if (eol != std::string::npos && eol > kMaxRequestLineBytes) {
            send_response(client_fd, 414, "application/json",
                          json_error("request line too long"));
            return;
        }
        auto req = parse_request(raw);

        if (req.method == "OPTIONS") {
            send_response(client_fd, 204, "text/plain", "");
            return;
        }

        if (req.path == "/api/generate") {
            handle_generate(client_fd, req);
        } else if (req.path == "/api/tokenize") {
            handle_tokenize(client_fd, req);
        } else if (req.path == "/api/detokenize") {
            handle_detokenize(client_fd, req);
        } else if (req.path == "/health" || req.path == "/") {
            send_response(client_fd, 200, "application/json",
                "{\"status\":\"ok\",\"model\":\"Transcender-QUANT\"}");
        } else {
            send_response(client_fd, 404, "application/json",
                json_error("not found: " + req.path));
        }
    }

    void worker_loop(int server_fd) {
        fd_set read_fds;
        while (g_running) {
            FD_ZERO(&read_fds);
            FD_SET((unsigned)server_fd, &read_fds);
            struct timeval tv = {1, 0};
            int sel = select(server_fd + 1, &read_fds, nullptr, nullptr, &tv);
            if (sel <= 0) continue;

            sockaddr_in client_addr;
#ifdef _WIN32
            int addr_len = sizeof(client_addr);
#else
            socklen_t addr_len = sizeof(client_addr);
#endif
            int client_fd = (int)accept(server_fd, (sockaddr*)&client_addr, &addr_len);
            if (client_fd < 0) continue;

            set_client_timeout(client_fd);

            handle_request(client_fd);
            quant::socket_helpers::close_socket(client_fd);
        }
    }
    // (P0: racy last_token_ member removed — now a per-request local.)
};

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    if (args.model_path.empty()) {
        std::cerr << "Error: --model required\n";
        return 1;
    }

    // Load model
    quant::DenseModel model;
    try {
        model.load(args.model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return 1;
    }

    // Load tokenizer — Qwen3.5-family models ship tokenizer.json in the model
    // dir; there is no separate .vocab file.
    if (args.model_dir.empty())
        args.model_dir = std::filesystem::path(args.model_path).parent_path().string();
    quant::Qwen35Tokenizer tokenizer;
    try {
        if (!tokenizer.load_from_dir(args.model_dir)) {
            std::cerr << "Error: failed to load tokenizer from " << args.model_dir
                      << " (expected tokenizer.json)\n";
            return 1;
        }
        std::cout << "Tokenizer loaded (" << tokenizer.vocab_size() << " vocab)"
                  << " from " << args.model_dir << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error loading tokenizer: " << e.what() << "\n";
        return 1;
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    QuantHTTPServer server(&model, &tokenizer, args.port, args.num_workers);
    server.start();

    return 0;
}
