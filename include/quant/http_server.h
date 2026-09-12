#pragma once

#include "quant/types.h"
#include "quant/json_parser.h"
#include "quant/socket_common.h"
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <chrono>
#include <unordered_map>

namespace quant {

// ===========================================================================
// RateLimiter — token bucket per-IP rate limiting
// ===========================================================================
class RateLimiter {
public:
    RateLimiter(int64_t max_requests_per_sec = 0, int64_t burst = 0);
    bool allow(const std::string& ip);
    void set_limits(int64_t max_requests_per_sec, int64_t burst);
    int64_t max_requests_per_sec() const { return max_rps_; }
    int64_t burst_size() const { return burst_; }

private:
    struct Bucket {
        int64_t tokens = 0;
        int64_t max_tokens = 0;
        std::chrono::steady_clock::time_point last_refill;
    };
    int64_t max_rps_;
    int64_t burst_;
    mutable std::mutex mtx_;
    std::unordered_map<std::string, Bucket> buckets_;
};

// ===========================================================================
// HTTPRequest — parsed HTTP request
// ===========================================================================
struct HTTPRequest {
    std::string method;
    std::string path;
    std::string body;
    std::unordered_map<std::string, std::string> headers;
    std::unordered_map<std::string, std::string> query_params;
};

// ===========================================================================
// HTTPServer — production-grade HTTP/1.1 server
//
// Features:
//   - Pure C++ raw sockets (Winsock2 / POSIX)
//   - Thread pool for concurrent connections
//   - JSON request/response (hand-written parser, no external libs)
//   - SSE (Server-Sent Events) streaming support
//   - CORS headers on all responses
//   - Per-IP rate limiting (token bucket)
//   - Model load/unload via API
//   - Endpoints: /v1/completions, /v1/chat/completions, /v1/embeddings,
//                /v1/models, /health
// ===========================================================================

class HTTPServer {
public:
    using GenerateCallback = std::function<std::string(const std::string& prompt, int max_tokens, bool stream)>;

    HTTPServer(int port = 8080);
    ~HTTPServer();

    void start();
    void stop();
    bool is_running() const { return running_.load(); }

    void set_thread_pool_size(int n);
    void set_timeout_seconds(int sec);
    void set_max_body_size(int64_t bytes);
    void set_rate_limit(int64_t max_requests_per_sec, int64_t burst);
    // ── L074 server hardening (Phase 16 Wave 7) ──────────────────────────
    // Auth: when non-empty, every /v1/* request must carry
    // "Authorization: Bearer <token>" or it gets 401 (OpenAI error shape).
    // Header cap: total header bytes beyond max_header_bytes_ get 431.
    // Concurrency cap: in-flight /v1/* beyond max_concurrent_ get 503.
    // All three default to permissive (empty/64KB/0=unlimited) so existing
    // callers and tests keep working; quant_serve sets strict values.
    void set_auth_token(const std::string& token);
    void set_max_header_bytes(int64_t bytes);
    void set_max_concurrent(int n);
    const std::string& auth_token() const {
        std::lock_guard<std::mutex> lock(config_mtx_);
        return auth_token_;
    }
    int64_t max_header_bytes() const {
        std::lock_guard<std::mutex> lock(config_mtx_);
        return max_header_bytes_;
    }
    int max_concurrent() const {
        std::lock_guard<std::mutex> lock(config_mtx_);
        return max_concurrent_;
    }
    // ── L075/L076 OpenAI contract helpers (pure, unit-tested) ────────────
    // OpenAI wire shapes shared by the handlers below and by
    // tests/test_server_contract.cpp. SSE uses OpenAI framing
    // ("data: <json>\n\n", terminator "data: [DONE]\n\n") — no "event:" line.
    static std::string openai_error_json(const std::string& message,
                                         const std::string& type,
                                         const std::string& code);
    static std::string sse_data_frame(const std::string& json);
    static std::string sse_done_frame();

    void set_model_name(const std::string& name) { model_name_ = name; }
    const std::string& model_name() const { return model_name_; }

    void set_generate_callback(GenerateCallback cb) { generate_cb_ = std::move(cb); }
    void set_load_model_callback(std::function<bool(const std::string&)> cb) { load_model_cb_ = std::move(cb); }
    void set_unload_model_callback(std::function<void()> cb) { unload_model_cb_ = std::move(cb); }
    void set_embeddings_callback(std::function<std::vector<float>(const std::string&)> cb) { embeddings_cb_ = std::move(cb); }

    int64_t total_requests() const { return total_requests_.load(); }
    int64_t total_errors() const { return total_errors_.load(); }
    int64_t current_requests() const { return current_requests_.load(); }

private:
    int port_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
    int thread_pool_size_ = 4;
    int timeout_seconds_ = 30;
    int64_t max_body_size_ = 4 * 1024 * 1024;
    std::string model_name_ = "Transcender-default";
    std::atomic<int64_t> total_requests_{0};
    std::atomic<int64_t> total_errors_{0};
    // L074 hardening state. auth_token_/max_header_bytes_/max_concurrent_/
    // thread_pool_size_/timeout_seconds_/max_body_size_ are written by
    // setters and read by worker threads — all access via config_mtx_
    // (setters lock, handle_request snapshots once per request).
    // current_requests_ stays atomic (CAS-loop admission in try_acquire_slot).
    mutable std::mutex config_mtx_;
    std::string auth_token_;
    int64_t max_header_bytes_ = 64 * 1024;
    int max_concurrent_ = 0;
    std::atomic<int64_t> current_requests_{0};
    static constexpr int64_t kMaxRequestLineBytes = 8192;

    GenerateCallback generate_cb_;
    std::function<bool(const std::string&)> load_model_cb_;
    std::function<void()> unload_model_cb_;
    std::function<std::vector<float>(const std::string&)> embeddings_cb_;

    RateLimiter rate_limiter_;

    struct ClientConn {
        int fd;
    };
    std::thread server_thread_;
    std::vector<std::thread> worker_threads_;
    std::queue<ClientConn> conn_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    void server_loop();
    void worker_loop();
    void handle_request(int fd);

    bool platform_init();
    void platform_cleanup();
    bool set_socket_timeout(int fd, int seconds);
    void close_fd(int fd);

    void send_response(int fd, int status, const std::string& content_type,
                       const std::string& body);
    void send_sse_headers(int fd);
    void send_sse_event(int fd, const std::string& event, const std::string& data);
    void send_sse_done(int fd);
    void send_error(int fd, int status, const std::string& message);
    // L074/L075 hardening + contract internals (see .cpp)
    void send_openai_error(int fd, int status, const std::string& message,
                           const std::string& type, const std::string& code);
    void send_sse_data(int fd, const std::string& json);
    bool check_auth(int fd, const HTTPRequest& req, const std::string& path);
    bool try_acquire_slot(int fd, const std::string& path);
    void release_slot(const std::string& path);
    static bool is_v1_path(const std::string& path);

    HTTPRequest parse_http_request(const std::string& raw);
    std::unordered_map<std::string, std::string> parse_query_string(const std::string& qs);

    void handle_health(int fd);
    void handle_models(int fd, const HTTPRequest& req);
    void handle_completions(int fd, const HTTPRequest& req);
    void handle_chat_completions(int fd, const HTTPRequest& req);
    void handle_embeddings(int fd, const HTTPRequest& req);
    void handle_options(int fd);
};

} // namespace quant
