#pragma once
#include "quant/model.h"
#include "quant/generator.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <cstdint>
#include <unordered_map>

namespace quant {

// ===========================================================================
// I21: Server metrics
// ===========================================================================
struct ServerMetrics {
    std::atomic<int64_t> total_requests{0};
    std::atomic<int64_t> total_errors{0};
    std::atomic<int64_t> total_tokens_generated{0};
    std::atomic<int64_t> current_requests{0};
    std::atomic<int64_t> max_concurrent{0};
    std::atomic<double> total_latency_ms{0.0};
    // P1 fix: requests_per_sec() needs a start instant. First call stamps it
    // (mutable mutex guards the stamp, not the atomics).
    mutable std::mutex start_mtx;
    mutable std::chrono::steady_clock::time_point start_tp{};
    mutable bool start_set = false;
    double avg_latency_ms() const {
        auto n = total_requests.load();
        return n > 0 ? total_latency_ms.load() / n : 0.0;
    }
    double requests_per_sec() const;
    struct LatencyHistogram {
        std::vector<double> samples;
        mutable std::mutex mtx;
        void record(double ms);
        double percentile(double p) const;
        double p50() const { return percentile(50.0); }
        double p95() const { return percentile(95.0); }
        double p99() const { return percentile(99.0); }
    };
    LatencyHistogram latency_hist;
};

// ===========================================================================
// I22: ModelServer — production-grade serving with request queue, batching,
//      metrics, health, and model lifecycle management
// ===========================================================================
struct ServingRequest {
    int64_t id;
    std::vector<int> prompt_ids;
    std::string prompt_text;
    int max_tokens;
    int priority;               // lower = higher priority
    std::chrono::steady_clock::time_point created_at;
    std::function<void(const std::string&, const std::vector<int>&)> callback;
};

struct BatchConsolidationResult {
    std::vector<int> request_ids;
    Tensor batched_input;
    Tensor batched_positions;
    std::vector<int> seq_lens;
    std::vector<int> offsets;
};

class ModelServer {
public:
    ModelServer(Model* model, Tokenizer* tokenizer, int port = 8080,
                int n_workers = 4);
    ~ModelServer();

    void start();
    void stop();
    bool is_running() const { return running_.load(); }

    // Lifecycle
    bool load_model(const std::string& path);
    void unload_model();
    bool is_model_loaded() const { return model_ != nullptr; }

    // Request submission
    int64_t submit(const ServingRequest& req);
    int64_t submit(const std::string& prompt, int max_tokens,
                    int priority = 0,
                    std::function<void(const std::string&, const std::vector<int>&)> cb = nullptr);

    // Metrics
    const ServerMetrics& metrics() const { return metrics_; }
    ServerMetrics& metrics() { return metrics_; }
    
    // Health
    std::string health_json() const;
    std::string metrics_json() const;

    void set_batch_size(int bs) { max_batch_size_ = bs; }
    void set_queue_wait_ms(int ms) { queue_wait_ms_ = ms; }

private:
    Model* model_;
    Tokenizer* tokenizer_;
    int port_;
    int max_batch_size_ = 8;
    int queue_wait_ms_ = 5;
    std::atomic<bool> running_{false};

    // Request queue (min-heap by priority)
    struct CompareRequest {
        bool operator()(const ServingRequest& a, const ServingRequest& b) {
            if (a.priority != b.priority) return a.priority > b.priority;
            return a.created_at > b.created_at;
        }
    };
    std::priority_queue<ServingRequest, std::vector<ServingRequest>, CompareRequest> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<int64_t> next_request_id_{1};

    // Worker threads
    std::vector<std::thread> workers_;
    std::thread server_thread_;
    std::thread metrics_thread_;

    // Consolidation
    BatchConsolidationResult consolidate_batch(std::vector<ServingRequest>& batch);

    // Loop functions
    void worker_loop();
    void metrics_loop();

    ServerMetrics metrics_;
};

// I3: C API bindings
extern "C" {
    struct QuantModel;
    QuantModel* quant_model_load(const char* path);
    void quant_model_free(QuantModel* model);
    char* quant_generate(QuantModel* model, const char* prompt, int max_tokens);
    void quant_free_string(char* s);
    const char* quant_last_error();
}

// I5: HTTP API server (Model-integrated, direct inference).
// Thin delegate over HTTPServer (forward declaration keeps socket/Windows
// headers out of this header, protecting Logger::Level from the ERROR macro).
class HTTPServer;
class ModelHTTPServer {
public:
    ModelHTTPServer(Model* model, int port = 8080);
    ~ModelHTTPServer();
    void start();
    void stop();
    bool is_running() const;
    void set_thread_pool_size(int n);
    void set_timeout_seconds(int sec);
    void set_max_body_size(size_t bytes);
private:
    Model* model_;
    int port_;
    std::unique_ptr<HTTPServer> inner_;
};

// I6: WebSocket streaming
class WebSocketHandler {
public:
    WebSocketHandler(int port = 8081);
    ~WebSocketHandler();
    void start();
    void stop();
    void broadcast(const std::string& token);
    bool is_running() const { return running_.load(); }
private:
    int port_;
    std::atomic<bool> running_{false};
    std::vector<int> clients_;
    std::mutex clients_mutex_;
    std::thread accept_thread_;
    std::vector<std::thread> client_threads_;

    void accept_loop();
    void client_loop(int client_fd);
    static std::string compute_accept_key(const std::string& client_key);
    static std::string base64_encode(const uint8_t* data, size_t len);
    static void sha1(const uint8_t* data, size_t len, uint8_t out[20]);
    static std::vector<uint8_t> create_frame(const std::string& data, uint8_t opcode);
    static bool parse_frame(const uint8_t* buf, size_t len,
                            std::string& payload, uint8_t& opcode, bool& fin);
    void remove_client(int client_fd);
    void close_socket(int fd);
};

// I11: Error handling system
enum class ErrorCode {
    SUCCESS = 0,
    FILE_NOT_FOUND = -1,
    OUT_OF_MEMORY = -2,
    CUDA_ERROR = -3,
    MODEL_LOAD_FAILED = -4,
    INFERENCE_FAILED = -5,
    INVALID_PARAM = -6,
    NETWORK_ERROR = -7,
    PARSE_ERROR = -8,
    PLUGIN_ERROR = -9,
    TIMEOUT = -10
};

struct Result {
    ErrorCode code = ErrorCode::SUCCESS;
    std::string message;
    bool ok() const { return code == ErrorCode::SUCCESS; }
};

// I12: Logging system
class Logger {
public:
    enum Level : uint8_t { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };
    Logger(Level level = INFO);
    void log(Level level, const std::string& message);
    void set_level(Level level) { level_ = level; }
    void set_file(const std::string& path);
    static Logger& instance();
private:
    Level level_;
    std::string file_path_;
    std::mutex mtx_;
    std::string level_str(Level l);
};

// I13: Configuration (JSON)
class AppConfig {
public:
    AppConfig(const std::string& path = "");
    float get_float(const std::string& key, float def = 0) const;
    int get_int(const std::string& key, int def = 0) const;
    std::string get_string(const std::string& key, const std::string& def = "") const;
    void set(const std::string& key, const std::string& value);
    void save(const std::string& path);
    std::string to_json() const;
    bool validate(std::string* error_out = nullptr) const;
private:
    struct JsonValue {
        enum Type : uint8_t { NULL_VAL = 0, BOOL = 1, INT64 = 2, FLOAT64 = 3, STRING = 4, ARRAY = 5, OBJECT = 6 };
        Type type = NULL_VAL;
        bool bool_val = false;
        int64_t int_val = 0;
        double float_val = 0.0;
        std::string str_val;
        std::vector<JsonValue> arr_val;
        std::unordered_map<std::string, JsonValue> obj_val;

        JsonValue() : type(NULL_VAL) {}
        JsonValue(bool b) : type(BOOL), bool_val(b) {}
        JsonValue(int64_t i) : type(INT64), int_val(i) {}
        JsonValue(double d) : type(FLOAT64), float_val(d) {}
        JsonValue(const std::string& s) : type(STRING), str_val(s) {}
        JsonValue(const char* s) : type(STRING), str_val(s) {}
        JsonValue(std::vector<JsonValue> a) : type(ARRAY), arr_val(std::move(a)) {}
        JsonValue(std::unordered_map<std::string, JsonValue> o) : type(OBJECT), obj_val(std::move(o)) {}
    };

    JsonValue root_;

    JsonValue* resolve_path(const std::string& key);
    const JsonValue* resolve_path(const std::string& key) const;
    static JsonValue parse_json(const std::string& json, size_t& pos, std::string* error);
    static void skip_ws(const std::string& json, size_t& pos);
    static JsonValue parse_value(const std::string& json, size_t& pos, std::string* error);
    static std::string parse_string(const std::string& json, size_t& pos, std::string* error);
    static JsonValue parse_number(const std::string& json, size_t& pos, std::string* error);
    static JsonValue parse_object(const std::string& json, size_t& pos, std::string* error);
    static JsonValue parse_array(const std::string& json, size_t& pos, std::string* error);
    static void serialize_json(const JsonValue& val, std::string& out, int indent);
    static std::string escape_string(const std::string& s);
    static JsonValue auto_type_value(const std::string& val);
};

// I14: Plugin system
class Plugin {
public:
    virtual ~Plugin() = default;
    virtual std::string name() const = 0;
    virtual void on_generate_start(const std::string& prompt) {}
    virtual void on_token_generated(int token) {}
    virtual void on_generate_end(const std::string& output) {}
};

class PluginManager {
public:
    PluginManager() = default;
    ~PluginManager();
    void load(const std::string& path);
    void unload(const std::string& name);
    void unload_all();
    bool hot_reload(const std::string& path);
    void register_plugin(Plugin* plugin);
    void on_generate_start(const std::string& prompt);
    void on_token_generated(int token);
    void on_generate_end(const std::string& output);
private:
    struct PluginEntry {
        Plugin* plugin = nullptr;
        std::string path;
        std::string name;
        void* handle = nullptr;
        std::string load_error;
    };
    std::vector<PluginEntry> entries_;
    std::vector<Plugin*> direct_plugins_;
    std::mutex plugins_mutex_;
    Plugin* load_from_dll(const std::string& path);
};

// I15: Model zoo
class ModelZoo {
public:
    struct ModelInfo { std::string name; std::string path; int64_t params; std::string format; };
    ModelZoo(const std::string& zoo_path = "models/");
    std::vector<ModelInfo> list_models() const;
    Model* load(const std::string& name);
private:
    std::string zoo_path_;
    mutable std::vector<ModelInfo> cache_;
    mutable bool cache_valid_ = false;
    void scan_directory(std::vector<ModelInfo>& out) const;
};

// I16-I18: Language bindings — dynamic library loading for Python/Java/Rust FFI
class PythonBindings { public: static void init(); };
class JavaBindings { public: static void init(); };
class RustBindings { public: static void init(); };

// I19: Mobile deployment — Android/iOS with toolchain detection
class MobileDeploy {
public:
    static bool deploy_android(const std::string& apk_path);
    static bool deploy_ios(const std::string& xcarchive_path);
};

// I20: WebAssembly
class WASMDeploy {
public:
    static bool compile_to_wasm(const std::string& source_path);
};

} // namespace quant
