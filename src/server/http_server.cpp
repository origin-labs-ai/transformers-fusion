#include "quant/http_server.h"
#include "quant/production_socket.h"
#include <cstring>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <chrono>
#include <atomic>
#include <string_view>
#include <mutex>

namespace quant {

// ===========================================================================
// Platform helpers
// ===========================================================================
namespace {

    bool http_platform_init() {
        return socket_helpers::platform_init();
    }

    void http_platform_cleanup() {
        socket_helpers::platform_cleanup();
    }

    bool set_sock_timeout(int fd, int seconds) {
        return socket_helpers::set_timeout(fd, seconds);
    }

    std::string get_client_ip(int fd) {
        sockaddr_in addr;
#ifdef _WIN32
        int len = sizeof(addr);
#else
        socklen_t len = sizeof(addr);
#endif
        if (getpeername(fd, (sockaddr*)&addr, &len) == 0) {
            char buf[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &addr.sin_addr, buf, sizeof(buf));
            return buf;
        }
        return "unknown";
    }

    std::string status_text(int code) {
        switch (code) {
            case 200: return "OK";
            case 201: return "Created";
            case 204: return "No Content";
            case 301: return "Moved Permanently";
            case 400: return "Bad Request";
            case 401: return "Unauthorized";
            case 403: return "Forbidden";
            case 404: return "Not Found";
            case 405: return "Method Not Allowed";
            case 408: return "Request Timeout";
            case 413: return "Payload Too Large";
            case 414: return "URI Too Long";
            case 415: return "Unsupported Media Type";
            case 429: return "Too Many Requests";
            case 431: return "Request Header Fields Too Large";
            case 500: return "Internal Server Error";
            case 502: return "Bad Gateway";
            case 503: return "Service Unavailable";
            default:  return "Unknown";
        }
    }

} // anonymous namespace

// ===========================================================================
// Response buffer pool — avoids repeated heap allocation for common response sizes
// ===========================================================================
namespace {
    struct ResponsePool {
        static constexpr size_t POOL_SIZE = 64;
        static constexpr size_t BUF_SIZE = 8192;
        std::mutex mtx;
        std::vector<void*> free_bufs;

        ResponsePool() { free_bufs.reserve(POOL_SIZE); }
        ~ResponsePool() {
            for (void* p : free_bufs) ::operator delete(p);
        }

        char* alloc() {
            std::lock_guard<std::mutex> lock(mtx);
            if (!free_bufs.empty()) {
                char* p = (char*)free_bufs.back();
                free_bufs.pop_back();
                return p;
            }
            return (char*)::operator new(BUF_SIZE);
        }

        void release(char* p) {
            std::lock_guard<std::mutex> lock(mtx);
            if (free_bufs.size() < POOL_SIZE) free_bufs.push_back(p);
            else ::operator delete(p);
        }

        static ResponsePool& instance() {
            static ResponsePool pool;
            return pool;
        }
    };
}

// ===========================================================================
// RateLimiter — token bucket
// ===========================================================================

RateLimiter::RateLimiter(int64_t max_requests_per_sec, int64_t burst)
    : max_rps_(max_requests_per_sec), burst_(burst) {}

bool RateLimiter::allow(const std::string& ip) {
    if (max_rps_ <= 0) return true;

    std::lock_guard<std::mutex> lock(mtx_);
    auto now = std::chrono::steady_clock::now();
    auto& b = buckets_[ip];

    if (b.max_tokens == 0) {
        b.max_tokens = burst_ > 0 ? burst_ : max_rps_;
        b.tokens = b.max_tokens;
        b.last_refill = now;
    }

    // Refill tokens
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - b.last_refill).count();
    int64_t refill = elapsed * max_rps_ / 1000;
    if (refill > 0) {
        b.tokens = std::min(b.max_tokens, b.tokens + refill);
        b.last_refill = now;
    }

    if (b.tokens > 0) {
        b.tokens--;
        return true;
    }
    return false;
}

void RateLimiter::set_limits(int64_t max_requests_per_sec, int64_t burst) {
    std::lock_guard<std::mutex> lock(mtx_);
    max_rps_ = max_requests_per_sec;
    burst_ = burst;
}

// ===========================================================================
// HTTPServer
// ===========================================================================

HTTPServer::HTTPServer(int port) : port_(port) {
    rate_limiter_.set_limits(0, 0); // no limit by default
}

HTTPServer::~HTTPServer() {
    stop();
}

void HTTPServer::set_thread_pool_size(int n) {
    std::lock_guard<std::mutex> lock(config_mtx_);
    thread_pool_size_ = (std::max)(1, n);
}

void HTTPServer::set_timeout_seconds(int sec) {
    std::lock_guard<std::mutex> lock(config_mtx_);
    timeout_seconds_ = (std::max)(1, sec);
}

void HTTPServer::set_max_body_size(int64_t bytes) {
    std::lock_guard<std::mutex> lock(config_mtx_);
    max_body_size_ = bytes;
}

void HTTPServer::set_rate_limit(int64_t max_requests_per_sec, int64_t burst) {
    rate_limiter_.set_limits(max_requests_per_sec, burst);
}

// ── L074 hardening setters ─────────────────────────────────────────────────
// BUGFIX (bug census): plain-field writes raced worker-thread reads
// (auth_token_/max_header_bytes_/max_concurrent_ read lock-free in
// check_auth/try_acquire_slot). Config now snapshot under a mutex; the
// admission counter uses a CAS loop so max_concurrent_ is a hard cap
// (fetch_add-then-check overshot under contention).

void HTTPServer::set_auth_token(const std::string& token) {
    std::lock_guard<std::mutex> lock(config_mtx_);
    auth_token_ = token;
}

void HTTPServer::set_max_header_bytes(int64_t bytes) {
    std::lock_guard<std::mutex> lock(config_mtx_);
    max_header_bytes_ = bytes > 0 ? bytes : (64 * 1024);
}

void HTTPServer::set_max_concurrent(int n) {
    std::lock_guard<std::mutex> lock(config_mtx_);
    max_concurrent_ = n > 0 ? n : 0;
    if (max_concurrent_ == 0) current_requests_.store(0);
}

// ── L075/L076 OpenAI contract helpers (pure, unit-tested) ──────────────────

std::string HTTPServer::openai_error_json(const std::string& message,
                                          const std::string& type,
                                          const std::string& code) {
    JsonValue err;
    err.type = JsonValue::OBJECT;
    err.obj["message"] = JsonValue(message);
    err.obj["type"] = JsonValue(type.empty() ? std::string("invalid_request_error") : type);
    err.obj["code"] = JsonValue(code.empty() ? std::string("") : code);
    JsonValue root;
    root.type = JsonValue::OBJECT;
    root.obj["error"] = std::move(err);
    return root.to_string();
}

std::string HTTPServer::sse_data_frame(const std::string& json) {
    return "data: " + json + "\n\n";
}

std::string HTTPServer::sse_done_frame() {
    return "data: [DONE]\n\n";
}

bool HTTPServer::is_v1_path(const std::string& path) {
    return path.size() >= 4 && path.compare(0, 4, "/v1/") == 0;
}

void HTTPServer::send_openai_error(int fd, int status, const std::string& message,
                                   const std::string& type, const std::string& code) {
    send_response(fd, status, "application/json",
                  openai_error_json(message, type, code));
}

void HTTPServer::send_sse_data(int fd, const std::string& json) {
    std::string frame = sse_data_frame(json);
    const char* buf = frame.c_str();
    size_t len = frame.size();
    size_t sent = 0;
    while (sent < len) {
        int n = send(fd, buf + sent, (int)(len - sent), 0);
        if (n <= 0) return;
        sent += n;
    }
}

bool HTTPServer::check_auth(int fd, const HTTPRequest& req, const std::string& path) {
    std::string token;
    {
        std::lock_guard<std::mutex> lock(config_mtx_);
        token = auth_token_;
    }
    if (token.empty()) return true;
    if (!is_v1_path(path)) return true; // /health stays open for probes
    auto it = req.headers.find("authorization");
    if (it == req.headers.end()) {
        send_openai_error(fd, 401, "Missing Authorization header",
                          "invalid_request_error", "invalid_api_key");
        return false;
    }
    const std::string want = "Bearer " + token;
    if (it->second != want) {
        send_openai_error(fd, 401, "Incorrect API key provided",
                          "invalid_request_error", "invalid_api_key");
        return false;
    }
    return true;
}

bool HTTPServer::try_acquire_slot(int fd, const std::string& path) {
    if (!is_v1_path(path)) return true;
    int cap = 0;
    {
        std::lock_guard<std::mutex> lock(config_mtx_);
        cap = max_concurrent_;
    }
    if (cap <= 0) return true;
    // CAS loop: claim only when under cap (fetch_add-then-check overshot).
    int64_t cur = current_requests_.load();
    while (cur < cap) {
        if (current_requests_.compare_exchange_weak(cur, cur + 1))
            return true;
    }
    send_openai_error(fd, 503, "Server is overloaded, try again later",
                      "server_error", "overloaded");
    return false;
}

void HTTPServer::release_slot(const std::string& path) {
    if (!is_v1_path(path)) return;
    int cap = 0;
    {
        std::lock_guard<std::mutex> lock(config_mtx_);
        cap = max_concurrent_;
    }
    if (cap <= 0) return;
    current_requests_.fetch_sub(1);
}

void HTTPServer::start() {
    if (running_.load()) return;
    if (!http_platform_init()) return;
    running_ = true;
    stop_requested_ = false;
    server_thread_ = std::thread(&HTTPServer::server_loop, this);
}

void HTTPServer::stop() {
    if (!running_.load()) return;
    running_ = false;
    stop_requested_ = true;
    queue_cv_.notify_all();
    if (server_thread_.joinable()) server_thread_.join();
    for (auto& t : worker_threads_) {
        if (t.joinable()) t.join();
    }
    worker_threads_.clear();
    http_platform_cleanup();
}

bool HTTPServer::platform_init() {
    return http_platform_init();
}

void HTTPServer::platform_cleanup() {
    http_platform_cleanup();
}

bool HTTPServer::set_socket_timeout(int fd, int seconds) {
    return set_sock_timeout(fd, seconds);
}

void HTTPServer::close_fd(int fd) {
    closesocket(fd);
}

// ===========================================================================
// Server loop — accept connections, dispatch to worker threads
// ===========================================================================

void HTTPServer::server_loop() {
#ifdef _WIN32
    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET) { running_ = false; return; }
#else
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { running_ = false; return; }
#endif

    int opt = 1;
#ifdef _WIN32
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((u_short)port_);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        closesocket(server_fd);
        running_ = false;
        return;
    }

    if (listen(server_fd, 128) < 0) {
        closesocket(server_fd);
        running_ = false;
        return;
    }

    // Spawn worker threads
    int n_workers = (std::max)(1, thread_pool_size_);
    for (int i = 0; i < n_workers; i++) {
        worker_threads_.emplace_back(&HTTPServer::worker_loop, this);
    }

    fd_set read_fds;
    while (running_ && !stop_requested_) {
        FD_ZERO(&read_fds);
#ifdef _WIN32
        FD_SET(server_fd, &read_fds);
#else
        FD_SET(server_fd, &read_fds);
#endif
        struct timeval tv = {1, 0};

        int sel = select((int)server_fd + 1, &read_fds, nullptr, nullptr, &tv);
        if (sel < 0) {
            if (QUANTSOCKET_ERRNO == QUANTSOCKET_EWOULDBLOCK) continue;
            if (!running_) break;
            continue;
        }
        if (sel == 0) continue;

        sockaddr_in client_addr;
#ifdef _WIN32
        int addr_len = sizeof(client_addr);
#else
        socklen_t addr_len = sizeof(client_addr);
#endif
        int client_fd = (int)accept(server_fd, (sockaddr*)&client_addr, &addr_len);
        if (client_fd < 0) continue;

        int nodelay = 1;
#ifdef _WIN32
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&nodelay, sizeof(nodelay));
#else
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
#endif

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            conn_queue_.push({client_fd});
        }
        queue_cv_.notify_one();
    }

    closesocket(server_fd);
    stop_requested_ = true;
    queue_cv_.notify_all();
}

// ===========================================================================
// Worker loop — dequeue and handle connections
// ===========================================================================

void HTTPServer::worker_loop() {
    while (!stop_requested_) {
        ClientConn conn;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::seconds(1), [this] {
                return !conn_queue_.empty() || stop_requested_;
            });
            if (stop_requested_ && conn_queue_.empty()) continue;
            if (conn_queue_.empty()) continue;
            conn = conn_queue_.front();
            conn_queue_.pop();
        }

        int timeout_snap = 0;
        {
            std::lock_guard<std::mutex> lock(config_mtx_);
            timeout_snap = timeout_seconds_;
        }
        set_socket_timeout(conn.fd, timeout_snap);
        handle_request(conn.fd);
        close_fd(conn.fd);
    }
}

// ===========================================================================
// Request parser
// ===========================================================================

HTTPRequest HTTPServer::parse_http_request(const std::string& raw) {
    HTTPRequest req;
    std::string_view sv(raw.data(), raw.size());

    auto header_end = sv.find("\r\n\r\n");
    std::string_view header_section;
    if (header_end != std::string_view::npos) {
        header_section = sv.substr(0, header_end);
        req.body = std::string(sv.substr(header_end + 4));
    } else {
        header_section = sv;
    }

    size_t pos = 0;
    bool first_line = true;
    // BUGFIX (bug census): per-request local (was a member — stale key from
    // a previous request on the same server object folded continuations into
    // the wrong request's headers).
    std::string last_header_key;
    // BUGFIX (bug census): the FINAL header line was silently dropped —
    // header_section excludes the trailing "\r\n\r\n", so the last line has
    // no '\n' and the loop below `break`s before parsing it (every request's
    // last header, e.g. Host, was lost). Iterate lines + tail.
    auto handle_line = [&](std::string_view line) {
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (line.empty()) return;
        if (first_line) {
            first_line = false;
            auto sp1 = line.find(' ');
            // BUGFIX (bug census): `find(' ', sp1+1)` with sp1==npos wraps to
            // a huge pos (UB-ish); malformed request lines also left stale
            // path/method. Validate before splitting.
            if (sp1 == std::string_view::npos) return;
            auto sp2 = line.find(' ', sp1 + 1);
            req.method = std::string(line.substr(0, sp1));
            if (sp2 != std::string_view::npos) {
                req.path = std::string(line.substr(sp1 + 1, sp2 - sp1 - 1));
            } else {
                req.path = std::string(line.substr(sp1 + 1));
            }
            // BUGFIX (bug census): header continuation lines (obs-fold,
            // leading SP/HT) were parsed as new headers (colon search on a
            // continuation). Fold them into the previous value per RFC 7230.
            auto qmark = req.path.find('?');
            if (qmark != std::string::npos) {
                req.query_params = parse_query_string(req.path.substr(qmark + 1));
                req.path = req.path.substr(0, qmark);
            }
        } else if (!line.empty() && (line.front() == ' ' || line.front() == '\t')) {
            // Continuation of the previous header value (obs-fold, RFC 7230).
            if (!last_header_key.empty()) {
                auto jt = req.headers.find(last_header_key);
                if (jt != req.headers.end()) {
                    auto vs = line.find_first_not_of(" \t");
                    if (vs != std::string_view::npos)
                        jt->second += std::string(" ") + std::string(line.substr(vs));
                }
            }
        } else {
            auto colon = line.find(':');
            if (colon != std::string_view::npos) {
                std::string_view key = line.substr(0, colon);
                std::string_view val = line.substr(colon + 1);
                auto ks = key.find_first_not_of(" \t");
                auto ke = key.find_last_not_of(" \t");
                auto vs = val.find_first_not_of(" \t");
                auto ve = val.find_last_not_of(" \t");
                if (ks != std::string_view::npos && vs != std::string_view::npos) {
                    std::string_view k_trimmed = key.substr(ks, ke - ks + 1);
                    std::string_view v_trimmed = val.substr(vs, ve - vs + 1);
                    std::string lower_key(k_trimmed);
                    std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(),
                                   [](unsigned char c) { return (char)std::tolower(c); });
                    req.headers[lower_key] = std::string(v_trimmed);
                    last_header_key = lower_key;
                }
            }
        }
    };
    while (pos < header_section.size()) {
        auto eol = header_section.find('\n', pos);
        if (eol == std::string_view::npos) {
            handle_line(header_section.substr(pos)); // trailing tail (no '\n')
            break;
        }
        handle_line(header_section.substr(pos, eol - pos));
        pos = eol + 1;
    }
    return req;
}

std::unordered_map<std::string, std::string>
HTTPServer::parse_query_string(const std::string& qs) {
    // BUGFIX (bug census): no %XX/+ decoding — clients sending %20 got
    // literal percent-sequences in param values. Decodes per RFC 3986
    // (malformed % sequences pass through literally, never throw).
    auto decode = [](const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (size_t i = 0; i < s.size(); i++) {
            if (s[i] == '+') {
                out += ' ';
            } else if (s[i] == '%' && i + 2 < s.size()) {
                auto hex = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                    return -1;
                };
                int hi = hex(s[i + 1]), lo = hex(s[i + 2]);
                if (hi >= 0 && lo >= 0) {
                    out += (char)(hi * 16 + lo);
                    i += 2;
                    continue;
                }
                out += s[i];
            } else {
                out += s[i];
            }
        }
        return out;
    };
    std::unordered_map<std::string, std::string> result;
    std::istringstream ss(qs);
    std::string pair;
    while (std::getline(ss, pair, '&')) {
        auto eq = pair.find('=');
        if (eq != std::string::npos)
            result[decode(pair.substr(0, eq))] = decode(pair.substr(eq + 1));
        else if (!pair.empty())
            result[decode(pair)] = "";
    }
    return result;
}

// ===========================================================================
// Response helpers
// ===========================================================================

void HTTPServer::send_response(int fd, int status, const std::string& content_type,
                                const std::string& body) {
    std::string resp = "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) + "\r\n"
        "Content-Type: " + content_type + "\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "Connection: close\r\n"
        "Server: Transcender/" + std::to_string(1) + "\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        // L074: baseline security headers on every response.
        "X-Content-Type-Options: nosniff\r\n"
        "X-Frame-Options: DENY\r\n"
        "Referrer-Policy: no-referrer\r\n"
        "\r\n";

    auto send_all = [fd](const char* data, size_t len) -> bool {
        size_t sent = 0;
        while (sent < len) {
            int n = send(fd, data + sent, (int)(len - sent), 0);
            if (n <= 0) return false;
            sent += n;
        }
        return true;
    };

    if (resp.size() + body.size() <= ResponsePool::BUF_SIZE) {
        char* buf = ResponsePool::instance().alloc();
        std::memcpy(buf, resp.data(), resp.size());
        std::memcpy(buf + resp.size(), body.data(), body.size());
        send_all(buf, resp.size() + body.size());
        ResponsePool::instance().release(buf);
    } else {
        if (!send_all(resp.c_str(), resp.size())) return;
        if (!body.empty())
            send_all(body.c_str(), body.size());
    }
}

void HTTPServer::send_sse_headers(int fd) {
    std::string resp =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Server: Transcender\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "\r\n";
    const char* data = resp.c_str();
    size_t len = resp.size();
    size_t sent = 0;
    while (sent < len) {
        int n = send(fd, data + sent, (int)(len - sent), 0);
        if (n <= 0) return;
        sent += n;
    }
}

void HTTPServer::send_sse_event(int fd, const std::string& event,
                                 const std::string& data) {
    // L075: legacy helper kept for API compat. New code must use send_sse_data()
    // (OpenAI framing has no "event:" line; the old prefix breaks openai-python
    // stream parsing). Behavior preserved here so existing callers don't change.
    std::string sse = "event: " + event + "\ndata: " + data + "\n\n";
    const char* buf = sse.c_str();
    size_t len = sse.size();
    size_t sent = 0;
    while (sent < len) {
        int n = send(fd, buf + sent, (int)(len - sent), 0);
        if (n <= 0) return;
        sent += n;
    }
}

void HTTPServer::send_sse_done(int fd) {
    std::string done = "data: [DONE]\n\n";
    const char* data = done.c_str();
    size_t len = done.size();
    size_t sent = 0;
    while (sent < len) {
        int n = send(fd, data + sent, (int)(len - sent), 0);
        if (n <= 0) return;
        sent += n;
    }
}

void HTTPServer::send_error(int fd, int status, const std::string& message) {
    JsonValue err;
    err.type = JsonValue::OBJECT;
    err.obj["error"] = JsonValue(message);
    send_response(fd, status, "application/json", err.to_string());
}

// ===========================================================================
// Request handler — route to endpoint
// ===========================================================================

void HTTPServer::handle_request(int fd) {
    std::vector<char> buf;
    buf.reserve(16384);
    char tmp[4096];

    // BUGFIX (bug census): snapshot hot config once per request — setters
    // take config_mtx_ and workers must not read the plain fields directly.
    int64_t max_body = 0, max_hdr = 0;
    int timeout_s = 0;
    {
        std::lock_guard<std::mutex> lock(config_mtx_);
        max_body = max_body_size_;
        max_hdr = max_header_bytes_;
        timeout_s = timeout_seconds_;
    }

    int n;
    bool timeout = false;
    auto start_time = std::chrono::steady_clock::now();

    while ((int64_t)buf.size() < max_body) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
            >= timeout_s) {
            timeout = true;
            break;
        }

        n = recv(fd, tmp, sizeof(tmp) - 1, 0);
        if (n < 0) {
            int err = QUANTSOCKET_ERRNO;
            if (err == QUANTSOCKET_EWOULDBLOCK || err == QUANTSOCKET_ETIMEDOUT) {
                timeout = true;
            }
            break;
        }
        if (n == 0) break;
        tmp[n] = '\0';
        buf.insert(buf.end(), tmp, tmp + n);

        // Check if we have complete headers
        std::string current(buf.data(), buf.size());
        auto hend = current.find("\r\n\r\n");
        if (hend != std::string::npos) {
            auto cl_pos = current.find("Content-Length:");
            if (cl_pos == std::string::npos)
                cl_pos = current.find("content-length:");
            if (cl_pos != std::string::npos) {
                auto num_start = current.find_first_of("0123456789", cl_pos + 15);
                if (num_start != std::string::npos) {
                    // P0 fix: stoll throws invalid_argument/out_of_range on
                    // attacker-controlled bytes — 400 the request instead of
                    // killing the worker thread.
                    int64_t content_length = -1;
                    try {
                        content_length = std::stoll(current.substr(num_start));
                    } catch (const std::exception&) {
                        send_openai_error(fd, 400, "Malformed Content-Length",
                                          "invalid_request_error", "invalid_content_length");
                        total_errors_.fetch_add(1);
                        return;
                    }
                    if (content_length < 0) {
                        send_openai_error(fd, 400, "Negative Content-Length",
                                          "invalid_request_error", "invalid_content_length");
                        total_errors_.fetch_add(1);
                        return;
                    }
                    if (content_length > max_body) {
                        send_error(fd, 413, "Payload Too Large");
                        total_errors_.fetch_add(1);
                        return;
                    }
                    int64_t body_received = (int64_t)buf.size() - (int64_t)hend - 4;
                    if (body_received >= content_length) break;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    total_requests_.fetch_add(1);

    if (buf.empty()) {
        send_error(fd, 400, "Empty request");
        total_errors_.fetch_add(1);
        return;
    }
    if (timeout) {
        send_error(fd, 408, "Request timeout");
        total_errors_.fetch_add(1);
        return;
    }

    std::string request(buf.data(), buf.size());
    HTTPRequest req = parse_http_request(request);

    // L074: header budget — the raw header section must fit max_header_bytes_.
    {
        auto hend = request.find("\r\n\r\n");
        size_t header_len = (hend == std::string::npos) ? request.size() : hend;
        if ((int64_t)header_len > max_hdr) {
            send_openai_error(fd, 431, "Request header fields too large",
                              "invalid_request_error", "header_too_large");
            total_errors_.fetch_add(1);
            return;
        }
        // L074: request-line cap (first CRLF within 8KB).
        auto eol = request.find("\r\n");
        if (eol != std::string::npos && (int64_t)eol > kMaxRequestLineBytes) {
            send_openai_error(fd, 414, "Request line too long",
                              "invalid_request_error", "request_line_too_long");
            total_errors_.fetch_add(1);
            return;
        }
    }

    // Rate limiting
    if (rate_limiter_.max_requests_per_sec() > 0) {
        std::string client_ip = get_client_ip(fd);
        if (!rate_limiter_.allow(client_ip)) {
            send_openai_error(fd, 429, "Rate limit exceeded",
                              "rate_limit_error", "rate_limit_exceeded");
            return;
        }
    }

    // CORS preflight
    if (req.method == "OPTIONS") {
        handle_options(fd);
        return;
    }

    // Only GET and POST
    if (req.method != "GET" && req.method != "POST") {
        send_openai_error(fd, 405, "Method not allowed",
                          "invalid_request_error", "method_not_allowed");
        return;
    }

    // L074: strict Content-Type for JSON POST bodies. Missing Content-Type is
    // accepted (curl/python defaults vary); a present-but-wrong type gets 415.
    if (req.method == "POST" && !req.body.empty() && is_v1_path(req.path)) {
        auto ct = req.headers.find("content-type");
        if (ct != req.headers.end() &&
            ct->second.find("application/json") == std::string::npos) {
            send_openai_error(fd, 415, "Content-Type must be application/json",
                              "invalid_request_error", "unsupported_media_type");
            return;
        }
    }

    // L074: bearer auth + concurrency admission for /v1/*.
    if (!check_auth(fd, req, req.path)) {
        total_errors_.fetch_add(1);
        return;
    }
    if (!try_acquire_slot(fd, req.path)) {
        total_errors_.fetch_add(1);
        return;
    }

    // Route
    if (req.path == "/health" || req.path == "/") {
        handle_health(fd);
    } else if (req.path == "/v1/models") {
        handle_models(fd, req);
    } else if (req.path == "/v1/completions") {
        handle_completions(fd, req);
    } else if (req.path == "/v1/chat/completions") {
        handle_chat_completions(fd, req);
    } else if (req.path == "/v1/embeddings") {
        handle_embeddings(fd, req);
    } else {
        // L075: unknown /v1/* paths keep the OpenAI error shape; non-API
        // paths keep the legacy plain shape for backward compat.
        if (is_v1_path(req.path))
            send_openai_error(fd, 404, "Not found: " + req.path,
                              "invalid_request_error", "not_found");
        else
            send_error(fd, 404, "Not found: " + req.path);
    }
    release_slot(req.path);
}

// ===========================================================================
// Endpoint handlers
// ===========================================================================

void HTTPServer::handle_options(int fd) {
    std::string resp =
        "HTTP/1.1 204 No Content\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Access-Control-Max-Age: 86400\r\n"
        "Content-Length: 0\r\n"
        "\r\n";
    send(fd, resp.c_str(), (int)resp.size(), 0);
}

void HTTPServer::handle_health(int fd) {
    JsonValue root;
    root.type = JsonValue::OBJECT;
    root.obj["status"] = JsonValue(std::string("ok"));
    root.obj["model"] = JsonValue(model_name_);
    root.obj["version"] = JsonValue(std::string("1.0.0"));
    root.obj["requests"] = JsonValue(total_requests_.load());
    root.obj["errors"] = JsonValue(total_errors_.load());
    send_response(fd, 200, "application/json", root.to_string());
}

void HTTPServer::handle_models(int fd, const HTTPRequest& req) {
    if (req.method == "GET") {
        JsonValue data_arr;
        data_arr.type = JsonValue::ARRAY;

        JsonValue model_obj;
        model_obj.type = JsonValue::OBJECT;
        model_obj.obj["id"] = JsonValue(model_name_);
        model_obj.obj["object"] = JsonValue(std::string("model"));
        model_obj.obj["created"] = JsonValue((int64_t)std::time(nullptr));
        model_obj.obj["owned_by"] = JsonValue(std::string("Transcender"));

        data_arr.arr.push_back(std::move(model_obj));

        JsonValue root;
        root.type = JsonValue::OBJECT;
        root.obj["object"] = JsonValue(std::string("list"));
        root.obj["data"] = std::move(data_arr);

        send_response(fd, 200, "application/json", root.to_string());
    } else {
        send_openai_error(fd, 405, "Use GET for /v1/models",
                          "invalid_request_error", "method_not_allowed");
    }
}

void HTTPServer::handle_completions(int fd, const HTTPRequest& req) {
    if (req.method != "POST") {
        send_openai_error(fd, 405, "POST required for /v1/completions",
                          "invalid_request_error", "method_not_allowed");
        return;
    }

    // Parse request body
    std::string body = req.body;
    JsonValue parsed;
    std::string parse_err;
    if (!body.empty()) {
        parsed = JsonValue::parse(body, &parse_err);
    }

    std::string prompt = "Hello";
    std::string requested_model = model_name_;
    int64_t max_tokens = 64;
    bool stream = false;
    double temperature = 1.0;
    double top_p = 1.0;
    int64_t top_k = 40;

    if (!parse_err.empty() && !body.empty()) {
        send_openai_error(fd, 400, "Invalid JSON: " + parse_err,
                          "invalid_request_error", "invalid_json");
        return;
    }

    if (parsed.is_object()) {
        // L075: prompt accepts string or array-of-strings (first used).
        // BUGFIX (bug census): p.arr[0] bypassed bounds/type checks.
        if (parsed.has("prompt")) {
            const JsonValue& p = parsed["prompt"];
            if (p.is_array() && !p.arr.empty() && p.arr[0].is_string())
                prompt = p.arr[0].as_string_checked();
            else if (p.is_string())
                prompt = p.as_string_checked();
        }
        if (parsed.has("model") && parsed["model"].is_string() &&
            !parsed["model"].as_string().empty())
            requested_model = parsed["model"].as_string();
        if (parsed.has("max_tokens") && parsed["max_tokens"].is_number())
            max_tokens = (int)parsed["max_tokens"].as_float_checked();
        // L075: OpenAI chat alias honored on this endpoint too.
        if (parsed.has("max_completion_tokens") && parsed["max_completion_tokens"].is_number())
            max_tokens = (int)parsed["max_completion_tokens"].as_float_checked();
        if (parsed.has("stream") && parsed["stream"].is_bool())
            stream = parsed["stream"].as_bool_checked();
        if (parsed.has("temperature") && parsed["temperature"].is_number())
            temperature = (float)parsed["temperature"].as_float_checked();
        if (parsed.has("top_p") && parsed["top_p"].is_number())
            top_p = (float)parsed["top_p"].as_float_checked();
        if (parsed.has("top_k") && parsed["top_k"].is_number())
            top_k = (int)parsed["top_k"].as_float_checked();
    }
    (void)temperature; (void)top_p; (void)top_k;

    max_tokens = std::max((int64_t)1, std::min(max_tokens, (int64_t)4096));

    if (stream) {
        send_sse_headers(fd);

        if (!generate_cb_) {
            send_sse_data(fd, openai_error_json("No generate callback registered",
                                                "server_error", "no_generate_callback"));
            send_sse_done(fd);
            return;
        }

        // Stream tokens one by one (OpenAI framing: bare "data:" lines).
        std::string cmpl_id = std::string("cmpl-") + std::to_string(total_requests_.load());
        int64_t created = (int64_t)std::time(nullptr);
        for (int64_t i = 0; i < max_tokens; i++) {
            std::string result = generate_cb_(prompt, 1, true);
            if (result.empty()) break;

            JsonValue chunk;
            chunk.type = JsonValue::OBJECT;
            chunk.obj["id"] = JsonValue(cmpl_id);
            chunk.obj["object"] = JsonValue(std::string("text_completion"));
            chunk.obj["created"] = JsonValue(created);
            chunk.obj["model"] = JsonValue(requested_model);

            JsonValue choices;
            choices.type = JsonValue::ARRAY;
            JsonValue choice;
            choice.type = JsonValue::OBJECT;
            choice.obj["text"] = JsonValue(result);
            choice.obj["index"] = JsonValue((int64_t)0);
            choice.obj["finish_reason"] = JsonValue();
            choices.arr.push_back(std::move(choice));

            chunk.obj["choices"] = std::move(choices);

            send_sse_data(fd, chunk.to_string());
            prompt += result;
        }

        // L075: terminal chunk carries finish_reason so openai-python stops.
        {
            JsonValue chunk;
            chunk.type = JsonValue::OBJECT;
            chunk.obj["id"] = JsonValue(cmpl_id);
            chunk.obj["object"] = JsonValue(std::string("text_completion"));
            chunk.obj["created"] = JsonValue(created);
            chunk.obj["model"] = JsonValue(requested_model);
            JsonValue choices;
            choices.type = JsonValue::ARRAY;
            JsonValue choice;
            choice.type = JsonValue::OBJECT;
            choice.obj["text"] = JsonValue(std::string(""));
            choice.obj["index"] = JsonValue((int64_t)0);
            choice.obj["finish_reason"] = JsonValue(std::string("stop"));
            choices.arr.push_back(std::move(choice));
            chunk.obj["choices"] = std::move(choices);
            send_sse_data(fd, chunk.to_string());
        }
        send_sse_done(fd);
        return;
    }

    // Non-streaming
    if (!generate_cb_) {
        send_openai_error(fd, 503, "No generate callback registered",
                          "server_error", "no_generate_callback");
        return;
    }

    std::string result = generate_cb_(prompt, (int)max_tokens, false);

    JsonValue root;
    root.type = JsonValue::OBJECT;
    root.obj["id"] = JsonValue(std::string("cmpl-") + std::to_string(total_requests_.load()));
    root.obj["object"] = JsonValue(std::string("text_completion"));
    root.obj["created"] = JsonValue((int64_t)std::time(nullptr));
    root.obj["model"] = JsonValue(requested_model);

    JsonValue choices;
    choices.type = JsonValue::ARRAY;
    JsonValue choice;
    choice.type = JsonValue::OBJECT;
    choice.obj["text"] = JsonValue(result);
    choice.obj["index"] = JsonValue((int64_t)0);
    choice.obj["finish_reason"] = JsonValue(std::string("stop"));
    choices.arr.push_back(std::move(choice));
    root.obj["choices"] = std::move(choices);

    JsonValue usage;
    usage.type = JsonValue::OBJECT;
    // Approximate token count: ~4 chars per token for English text.
    // No full tokenizer available in the HTTP handler; this is a rough estimate.
    int64_t approx_prompt_tokens = std::max((int64_t)1, (int64_t)prompt.size() / 4);
    int64_t approx_completion_tokens = std::max((int64_t)1, (int64_t)result.size() / 4);
    usage.obj["prompt_tokens"] = JsonValue(approx_prompt_tokens);
    usage.obj["completion_tokens"] = JsonValue(approx_completion_tokens);
    usage.obj["total_tokens"] = JsonValue(approx_prompt_tokens + approx_completion_tokens);
    root.obj["usage"] = std::move(usage);

    send_response(fd, 200, "application/json", root.to_string());
}

void HTTPServer::handle_chat_completions(int fd, const HTTPRequest& req) {
    if (req.method != "POST") {
        send_openai_error(fd, 405, "POST required for /v1/chat/completions",
                          "invalid_request_error", "method_not_allowed");
        return;
    }

    std::string body = req.body;
    JsonValue parsed;
    std::string parse_err;
    if (!body.empty()) {
        parsed = JsonValue::parse(body, &parse_err);
    }

    std::string requested_model = model_name_;
    int64_t max_tokens = 1024;
    bool stream = false;
    double temperature = 1.0;

    if (!parse_err.empty() && !body.empty()) {
        send_openai_error(fd, 400, "Invalid JSON: " + parse_err,
                          "invalid_request_error", "invalid_json");
        return;
    }

    // L075: full conversation history is folded into the prompt in order
    // (system first, then every user/assistant turn). The old code kept only
    // the last user turn and silently dropped assistant history.
    std::string transcript;
    bool has_messages = false;
    if (parsed.is_object()) {
        if (parsed.has("model") && parsed["model"].is_string() &&
            !parsed["model"].as_string().empty())
            requested_model = parsed["model"].as_string();
        if (parsed.has("max_tokens")) max_tokens = parsed["max_tokens"].as_int();
        if (parsed.has("max_completion_tokens"))
            max_tokens = parsed["max_completion_tokens"].as_int();
        if (parsed.has("stream")) stream = parsed["stream"].as_bool();
        if (parsed.has("temperature")) temperature = parsed["temperature"].as_float();

        if (parsed.has("messages") && parsed["messages"].is_array() &&
            !parsed["messages"].arr.empty()) {
            has_messages = true;
            for (auto& msg : parsed["messages"].arr) {
                std::string role = msg.has("role") ? msg["role"].as_string() : "";
                std::string content = msg.has("content") ? msg["content"].as_string() : "";
                if (role.empty() && content.empty()) continue;
                if (!transcript.empty()) transcript += "\n";
                transcript += role + ": " + content;
            }
        }
    }
    (void)temperature;
    if (!has_messages || transcript.empty()) {
        send_openai_error(fd, 400, "messages array with at least one message is required",
                          "invalid_request_error", "missing_messages");
        return;
    }
    max_tokens = std::max((int64_t)1, std::min(max_tokens, (int64_t)4096));

    std::string full_prompt = transcript;

    if (stream) {
        send_sse_headers(fd);

        if (!generate_cb_) {
            send_sse_data(fd, openai_error_json("No generate callback registered",
                                                "server_error", "no_generate_callback"));
            send_sse_done(fd);
            return;
        }

        std::string chat_id = std::string("chatcmpl-") + std::to_string(total_requests_.load());
        int64_t created = (int64_t)std::time(nullptr);
        // L075: opening chunk announces the assistant role (openai-python
        // expects delta.role on the first chunk).
        {
            JsonValue chunk;
            chunk.type = JsonValue::OBJECT;
            chunk.obj["id"] = JsonValue(chat_id);
            chunk.obj["object"] = JsonValue(std::string("chat.completion.chunk"));
            chunk.obj["created"] = JsonValue(created);
            chunk.obj["model"] = JsonValue(requested_model);
            JsonValue choices;
            choices.type = JsonValue::ARRAY;
            JsonValue choice;
            choice.type = JsonValue::OBJECT;
            JsonValue delta;
            delta.type = JsonValue::OBJECT;
            delta.obj["role"] = JsonValue(std::string("assistant"));
            delta.obj["content"] = JsonValue(std::string(""));
            choice.obj["delta"] = std::move(delta);
            choice.obj["index"] = JsonValue((int64_t)0);
            choice.obj["finish_reason"] = JsonValue();
            choices.arr.push_back(std::move(choice));
            chunk.obj["choices"] = std::move(choices);
            send_sse_data(fd, chunk.to_string());
        }

        for (int64_t i = 0; i < max_tokens; i++) {
            std::string result = generate_cb_(full_prompt, 1, true);
            if (result.empty()) break;

            JsonValue chunk;
            chunk.type = JsonValue::OBJECT;
            chunk.obj["id"] = JsonValue(chat_id);
            chunk.obj["object"] = JsonValue(std::string("chat.completion.chunk"));
            chunk.obj["created"] = JsonValue(created);
            chunk.obj["model"] = JsonValue(requested_model);

            JsonValue choices;
            choices.type = JsonValue::ARRAY;
            JsonValue choice;
            choice.type = JsonValue::OBJECT;
            JsonValue delta;
            delta.type = JsonValue::OBJECT;
            delta.obj["content"] = JsonValue(result);
            choice.obj["delta"] = std::move(delta);
            choice.obj["index"] = JsonValue((int64_t)0);
            choice.obj["finish_reason"] = JsonValue();
            choices.arr.push_back(std::move(choice));
            chunk.obj["choices"] = std::move(choices);

            send_sse_data(fd, chunk.to_string());
            full_prompt += result;
        }

        // L075: terminal chunk with finish_reason "stop".
        {
            JsonValue chunk;
            chunk.type = JsonValue::OBJECT;
            chunk.obj["id"] = JsonValue(chat_id);
            chunk.obj["object"] = JsonValue(std::string("chat.completion.chunk"));
            chunk.obj["created"] = JsonValue(created);
            chunk.obj["model"] = JsonValue(requested_model);
            JsonValue choices;
            choices.type = JsonValue::ARRAY;
            JsonValue choice;
            choice.type = JsonValue::OBJECT;
            JsonValue delta;
            delta.type = JsonValue::OBJECT;
            choice.obj["delta"] = std::move(delta);
            choice.obj["index"] = JsonValue((int64_t)0);
            choice.obj["finish_reason"] = JsonValue(std::string("stop"));
            choices.arr.push_back(std::move(choice));
            chunk.obj["choices"] = std::move(choices);
            send_sse_data(fd, chunk.to_string());
        }
        send_sse_done(fd);
        return;
    }

    if (!generate_cb_) {
        send_openai_error(fd, 503, "No generate callback registered",
                          "server_error", "no_generate_callback");
        return;
    }

    std::string result = generate_cb_(full_prompt, (int)max_tokens, false);

    JsonValue root;
    root.type = JsonValue::OBJECT;
    root.obj["id"] = JsonValue(std::string("chatcmpl-") + std::to_string(total_requests_.load()));
    root.obj["object"] = JsonValue(std::string("chat.completion"));
    root.obj["created"] = JsonValue((int64_t)std::time(nullptr));
    root.obj["model"] = JsonValue(requested_model);

    JsonValue choices;
    choices.type = JsonValue::ARRAY;
    JsonValue choice;
    choice.type = JsonValue::OBJECT;

    JsonValue message;
    message.type = JsonValue::OBJECT;
    message.obj["role"] = JsonValue(std::string("assistant"));
    message.obj["content"] = JsonValue(result);
    choice.obj["message"] = std::move(message);
    choice.obj["index"] = JsonValue((int64_t)0);
    choice.obj["finish_reason"] = JsonValue(std::string("stop"));
    choices.arr.push_back(std::move(choice));
    root.obj["choices"] = std::move(choices);

    JsonValue usage;
    usage.type = JsonValue::OBJECT;
    // Approximate token count: ~4 chars per token for English text.
    // No full tokenizer available in the HTTP handler; this is a rough estimate.
    int64_t approx_prompt_tokens = std::max((int64_t)1, (int64_t)full_prompt.size() / 4);
    int64_t approx_completion_tokens = std::max((int64_t)1, (int64_t)result.size() / 4);
    usage.obj["prompt_tokens"] = JsonValue(approx_prompt_tokens);
    usage.obj["completion_tokens"] = JsonValue(approx_completion_tokens);
    usage.obj["total_tokens"] = JsonValue(approx_prompt_tokens + approx_completion_tokens);
    root.obj["usage"] = std::move(usage);

    send_response(fd, 200, "application/json", root.to_string());
}

void HTTPServer::handle_embeddings(int fd, const HTTPRequest& req) {
    if (req.method != "POST") {
        send_openai_error(fd, 405, "POST required for /v1/embeddings",
                          "invalid_request_error", "method_not_allowed");
        return;
    }

    std::string body = req.body;
    JsonValue parsed;
    std::string parse_err;
    if (!body.empty()) {
        parsed = JsonValue::parse(body, &parse_err);
    }

    if (!parse_err.empty() && !body.empty()) {
        send_openai_error(fd, 400, "Invalid JSON: " + parse_err,
                          "invalid_request_error", "invalid_json");
        return;
    }

    // L075: input accepts a string or an array of strings (OpenAI parity).
    std::string requested_model = model_name_;
    std::vector<std::string> inputs;
    if (parsed.is_object()) {
        if (parsed.has("model") && parsed["model"].is_string() &&
            !parsed["model"].as_string().empty())
            requested_model = parsed["model"].as_string();
        if (parsed.has("input")) {
            const JsonValue& in = parsed["input"];
            if (in.is_string()) {
                inputs.push_back(in.as_string());
            } else if (in.is_array()) {
                for (auto& el : in.arr) {
                    if (el.is_string()) inputs.push_back(el.as_string());
                }
            }
        }
    }
    if (inputs.empty()) {
        send_openai_error(fd, 400, "input must be a string or array of strings",
                          "invalid_request_error", "missing_input");
        return;
    }
    if (inputs.size() > 256) {
        send_openai_error(fd, 400, "too many inputs (max 256)",
                          "invalid_request_error", "too_many_inputs");
        return;
    }

    if (!embeddings_cb_) {
        send_openai_error(fd, 503, "No embeddings callback registered",
                          "server_error", "no_embeddings_callback");
        return;
    }

    JsonValue root;
    root.type = JsonValue::OBJECT;
    root.obj["object"] = JsonValue(std::string("list"));
    root.obj["model"] = JsonValue(requested_model);

    JsonValue data_arr;
    data_arr.type = JsonValue::ARRAY;

    int64_t total_chars = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        std::vector<float> embedding = embeddings_cb_(inputs[i]);
        total_chars += (int64_t)inputs[i].size();

        JsonValue obj;
        obj.type = JsonValue::OBJECT;
        obj.obj["object"] = JsonValue(std::string("embedding"));
        obj.obj["index"] = JsonValue((int64_t)i);

        JsonValue emb_arr;
        emb_arr.type = JsonValue::ARRAY;
        for (float v : embedding)
            emb_arr.arr.push_back(JsonValue((double)v));
        obj.obj["embedding"] = std::move(emb_arr);

        data_arr.arr.push_back(std::move(obj));
    }
    root.obj["data"] = std::move(data_arr);

    JsonValue usage;
    usage.type = JsonValue::OBJECT;
    // Approximate token count: ~4 chars per token for English text.
    int64_t approx_tokens = std::max((int64_t)1, total_chars / 4);
    usage.obj["prompt_tokens"] = JsonValue(approx_tokens);
    usage.obj["total_tokens"] = JsonValue(approx_tokens);
    root.obj["usage"] = std::move(usage);

    send_response(fd, 200, "application/json", root.to_string());
}

} // namespace quant
