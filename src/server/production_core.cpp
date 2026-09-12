#include "quant/production_internal.h"
#include "quant/http_server.h"
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <ctime>
#include <thread>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <set>
#include <map>
#include <chrono>

#ifdef ERROR
#undef ERROR
#endif

namespace quant {

namespace {

thread_local std::string g_last_error;

} // anonymous namespace

// ========================================================================
// I3: C API
// ========================================================================
struct QuantModel { Model* model; };

const char* quant_last_error() {
    return g_last_error.c_str();
}

bool file_exists(const char* path) {
#ifdef _WIN32
    DWORD attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
#endif
}

QuantModel* quant_model_load(const char* path) {
    g_last_error.clear();
    if (!path) {
        g_last_error = "path is null";
        errno = EINVAL;
        return nullptr;
    }

    if (!file_exists(path)) {
        g_last_error = std::string("file not found: ") + path;
#ifdef _WIN32
        SetLastError(ERROR_FILE_NOT_FOUND);
#endif
        errno = ENOENT;
        return nullptr;
    }

    auto* om = new(std::nothrow) QuantModel;
    if (!om) {
        g_last_error = "out of memory";
        errno = ENOMEM;
        return nullptr;
    }

    om->model = new(std::nothrow) DenseModel;
    if (!om->model) {
        delete om;
        g_last_error = "out of memory";
        errno = ENOMEM;
        return nullptr;
    }

    try {
        om->model->load(path);
    } catch (const std::exception& e) {
        g_last_error = std::string("model load failed: ") + e.what();
        delete om->model;
        delete om;
#ifdef _WIN32
        SetLastError(ERROR_FILE_NOT_FOUND);
#endif
        errno = ENOENT;
        return nullptr;
    } catch (...) {
        Logger::instance().log(Logger::ERROR, std::string("quant_model_load: unknown exception in ") + __func__);
        g_last_error = "model load failed: unknown error";
        delete om->model;
        delete om;
        errno = ENOENT;
        return nullptr;
    }

    return om;
}

void quant_model_free(QuantModel* model) {
    if (!model) return;
    delete model->model;
    delete model;
}

char* quant_generate(QuantModel* model, const char* prompt, int max_tokens) {
    g_last_error.clear();

    if (!model) {
        g_last_error = "model is null";
        errno = EINVAL;
        char* empty = new char[1];
        empty[0] = '\0';
        return empty;
    }
    if (!model->model) {
        g_last_error = "internal model is null";
        errno = EINVAL;
        char* empty = new char[1];
        empty[0] = '\0';
        return empty;
    }
    if (!prompt) {
        g_last_error = "prompt is null";
        errno = EINVAL;
        char* empty = new char[1];
        empty[0] = '\0';
        return empty;
    }
    if (max_tokens <= 0) {
        max_tokens = 64;
    }
    if (max_tokens > 4096) {
        max_tokens = 4096;
    }

    BPETokenizer bpe;
    std::vector<int> tokens;
    try {
        tokens = bpe.encode(prompt);
    } catch (const std::exception& e) {
        g_last_error = std::string("tokenization failed: ") + e.what();
        errno = EINVAL;
        char* empty = new char[1];
        empty[0] = '\0';
        return empty;
    }

    if (tokens.empty()) tokens = {1};

    std::vector<int> output;
    output.reserve(max_tokens);

    const auto& cfg = model->model->config;
    KVCache cache((int)cfg.num_layers, cfg.max_seq_len, cfg.num_heads, cfg.head_dim);

    auto sample_token = [&](const Tensor& logits) -> int {
        if (logits.numel() == 0) return -1;
        int64_t V = logits.dim(logits.rank() - 1);
        const float* ld = logits.data<float>() + logits.numel() - V;
        int next = 0;
        for (int64_t v = 1; v < V; v++)
            if (ld[v] > ld[next]) next = (int)v;
        return next;
    };

    try {
        int64_t prompt_len = (int64_t)tokens.size();
        Tensor input({1, prompt_len});
        float* id = input.data<float>();
        for (size_t j = 0; j < tokens.size(); j++) id[j] = (float)tokens[j];

        Tensor pos({1, prompt_len});
        float* pd = pos.data<float>();
        for (int64_t j = 0; j < prompt_len; j++) pd[j] = (float)j;

        Tensor logits = model->model->forward(input, pos, &cache);
        int next = sample_token(logits);
        if (next < 0) { /* empty result */ }
        else if (next == 2) { /* EOS */ }
        else output.push_back(next);
    } catch (const std::exception& e) {
        g_last_error = std::string("inference failed: ") + e.what();
        errno = EIO;
    }

    for (int i = 1; i < max_tokens && !output.empty(); i++) {
        try {
            int64_t pos_val = (int64_t)tokens.size() + (int64_t)output.size() - 1;

            Tensor input({1, 1});
            input.data<float>()[0] = (float)output.back();

            Tensor pos({1, 1});
            pos.data<float>()[0] = (float)pos_val;

            Tensor logits = model->model->forward(input, pos, &cache);
            int next = sample_token(logits);
            if (next < 0 || next == 2) break;
            output.push_back(next);
        } catch (const std::exception& e) {
            g_last_error = std::string("inference failed: ") + e.what();
            errno = EIO;
            break;
        }
    }

    std::string result = bpe.decode(output);
    char* cstr = new char[result.size() + 1];
    std::strcpy(cstr, result.c_str());
    return cstr;
}

void quant_free_string(char* s) {
    delete[] s;
}

// ========================================================================
// I5: HTTP Server (Model-integrated)
// ========================================================================
// F5: thin delegate over HTTPServer. All socket/select/worker and route
// handling lives in HTTPServer (src/server/http_server.cpp).
ModelHTTPServer::ModelHTTPServer(Model* model, int port)
    : model_(model), port_(port), inner_(std::make_unique<HTTPServer>(port)) {
    inner_->set_generate_callback(
        [this](const std::string& prompt, int max_tokens, bool /*stream*/) -> std::string {
            Model* m = this->model_;
            if (!m) return "";
            if (max_tokens <= 0) max_tokens = 64;
            if (max_tokens > 4096) max_tokens = 4096;
            try {
                BPETokenizer bpe;
                std::vector<int> tokens;
                try {
                    tokens = bpe.encode(prompt);
                } catch (...) {
                    return "";
                }
                if (tokens.empty()) tokens = {1};
                std::vector<int> output;
                output.reserve(static_cast<size_t>(max_tokens));
                for (int i = 0; i < max_tokens; ++i) {
                    try {
                        int64_t seq_len =
                            static_cast<int64_t>(tokens.size() + output.size());
                        Tensor input({1, seq_len});
                        float* id = input.data<float>();
                        for (size_t j = 0; j < tokens.size(); ++j)
                            id[j] = static_cast<float>(tokens[j]);
                        for (size_t j = 0; j < output.size(); ++j)
                            id[tokens.size() + j] = static_cast<float>(output[j]);
                        Tensor pos({1, seq_len});
                        float* pd = pos.data<float>();
                        for (int64_t j = 0; j < seq_len; ++j)
                            pd[j] = static_cast<float>(j);
                        Tensor logits = m->forward(input, pos);
                        if (logits.numel() == 0) break;
                        int64_t V = logits.dim(logits.rank() - 1);
                        const float* ld = logits.data<float>() + logits.numel() - V;
                        int next = 0;
                        for (int64_t v = 1; v < V; ++v)
                            if (ld[v] > ld[next]) next = static_cast<int>(v);
                        if (next == 2) break;
                        output.push_back(next);
                    } catch (...) {
                        break;
                    }
                }
                try {
                    return bpe.decode(output);
                } catch (...) {
                    return "";
                }
            } catch (...) {
                return "";
            }
        });
}

ModelHTTPServer::~ModelHTTPServer() { stop(); }

void ModelHTTPServer::set_thread_pool_size(int n) { inner_->set_thread_pool_size(n); }

void ModelHTTPServer::set_timeout_seconds(int sec) { inner_->set_timeout_seconds(sec); }

void ModelHTTPServer::set_max_body_size(size_t bytes) {
    inner_->set_max_body_size(static_cast<int64_t>(bytes));
}

void ModelHTTPServer::start() { inner_->start(); }

void ModelHTTPServer::stop() { inner_->stop(); }

bool ModelHTTPServer::is_running() const { return inner_->is_running(); }


// ========================================================================
// I12: Logger
// ========================================================================
Logger::Logger(Level level) : level_(level) {}

std::string Logger::level_str(Level l) {
    switch (l) {
        case DEBUG: return "DEBUG";
        case INFO:  return "INFO";
        case WARN:  return "WARN";
        case ERROR: return "ERROR";
        default:    return "UNKNOWN";
    }
}

void Logger::log(Level level, const std::string& message) {
    // BUGFIX (bug census): std::localtime uses a shared static tm buffer
    // (data race across logging threads) and level_/file_path_ were read
    // without the lock while setters wrote them. Snapshot config under the
    // same mutex; use localtime_r/localtime_s.
    std::lock_guard<std::mutex> lock(mtx_);
    if (level < level_) return;
    std::string path = file_path_;
    auto t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[20];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    std::string line = std::string(buf) + " [" + level_str(level) + "] " + message;
    if (!path.empty()) {
        std::ofstream f(path, std::ios::app);
        if (f) f << line << std::endl;
    }
    std::cout << line << std::endl;
}

void Logger::set_file(const std::string& path) {
    std::lock_guard<std::mutex> lock(mtx_);
    file_path_ = path;
}
Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

// ========================================================================
// I13: Config (JSON parser)
// ========================================================================
AppConfig::AppConfig(const std::string& path) {
    if (path.empty()) return;

    std::ifstream f(path);
    if (!f) return;

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    if (content.empty()) return;

    std::string error;
    size_t pos = 0;
    root_ = parse_json(content, pos, &error);

    if (!error.empty()) {
        Logger::instance().log(Logger::WARN,
            "Config parse error in " + path + ": " + error + " at position " +
            std::to_string(pos));
    }
}

void AppConfig::skip_ws(const std::string& json, size_t& pos) {
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
           json[pos] == '\n' || json[pos] == '\r')) {
        pos++;
    }
}

std::string AppConfig::escape_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (size_t i = 0; i < s.size(); i++) {
        char c = s[i];
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

AppConfig::JsonValue AppConfig::auto_type_value(const std::string& val) {
    char* end = nullptr;
    long long ll = strtoll(val.c_str(), &end, 10);
    if (end && *end == '\0' && end != val.c_str()) {
        return JsonValue((int64_t)ll);
    }

    end = nullptr;
    double d = strtod(val.c_str(), &end);
    if (end && *end == '\0' && end != val.c_str()) {
        return JsonValue(d);
    }

    if (val == "true") return JsonValue(true);
    if (val == "false") return JsonValue(false);

    return JsonValue(val);
}

std::string AppConfig::parse_string(const std::string& json, size_t& pos,
                                     std::string* error) {
    std::string result;
    if (pos >= json.size() || json[pos] != '"') {
        if (error) *error = "expected string";
        return result;
    }
    pos++;

    while (pos < json.size()) {
        char c = json[pos];
        if (c == '"') {
            pos++;
            return result;
        }
        if (c == '\\') {
            pos++;
            if (pos >= json.size()) break;
            switch (json[pos]) {
                case '"':  result += '"'; break;
                case '\\': result += '\\'; break;
                case '/':  result += '/'; break;
                case 'b':  result += '\b'; break;
                case 'f':  result += '\f'; break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                case 'u': {
                    if (pos + 4 < json.size()) {
                        result += "\\u";
                        for (int i = 0; i < 4; i++) {
                            pos++;
                            result += json[pos];
                        }
                    }
                    break;
                }
                default: result += json[pos]; break;
            }
            pos++;
        } else {
            result += c;
            pos++;
        }
    }

    if (error) *error = "unterminated string";
    return result;
}

AppConfig::JsonValue AppConfig::parse_number(const std::string& json,
                                              size_t& pos, std::string* error) {
    size_t start = pos;
    if (pos < json.size() && json[pos] == '-') pos++;

    while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') pos++;

    bool is_float = false;
    if (pos < json.size() && json[pos] == '.') {
        is_float = true;
        pos++;
        while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') pos++;
    }

    if (pos < json.size() && (json[pos] == 'e' || json[pos] == 'E')) {
        is_float = true;
        pos++;
        if (pos < json.size() && (json[pos] == '+' || json[pos] == '-')) pos++;
        while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') pos++;
    }

    std::string num_str = json.substr(start, pos - start);

    if (is_float) {
        char* end = nullptr;
        double d = strtod(num_str.c_str(), &end);
        (void)end;
        return JsonValue(d);
    } else {
        char* end = nullptr;
        long long ll = strtoll(num_str.c_str(), &end, 10);
        (void)end;
        return JsonValue((int64_t)ll);
    }
}

AppConfig::JsonValue AppConfig::parse_value(const std::string& json,
                                             size_t& pos, std::string* error) {
    skip_ws(json, pos);
    if (pos >= json.size()) {
        if (error) *error = "unexpected end of JSON";
        return JsonValue();
    }

    switch (json[pos]) {
        case '{': return parse_object(json, pos, error);
        case '[': return parse_array(json, pos, error);
        case '"': return JsonValue(parse_string(json, pos, error));
        case 't':
            if (json.substr(pos, 4) == "true") { pos += 4; return JsonValue(true); }
            if (error) *error = "expected true";
            return JsonValue();
        case 'f':
            if (json.substr(pos, 5) == "false") { pos += 5; return JsonValue(false); }
            if (error) *error = "expected false";
            return JsonValue();
        case 'n':
            if (json.substr(pos, 4) == "null") { pos += 4; return JsonValue(); }
            if (error) *error = "expected null";
            return JsonValue();
        default:
            if (json[pos] == '-' || (json[pos] >= '0' && json[pos] <= '9'))
                return parse_number(json, pos, error);
            if (error) *error = std::string("unexpected character '") + json[pos] + "'";
            return JsonValue();
    }
}

AppConfig::JsonValue AppConfig::parse_object(const std::string& json,
                                              size_t& pos, std::string* error) {
    std::unordered_map<std::string, JsonValue> obj;
    pos++;

    skip_ws(json, pos);
    if (pos < json.size() && json[pos] == '}') {
        pos++;
        return JsonValue(obj);
    }

    while (pos < json.size()) {
        skip_ws(json, pos);
        if (pos >= json.size()) {
            if (error) *error = "unterminated object";
            break;
        }

        if (json[pos] != '"') {
            if (error) *error = "expected string key in object";
            break;
        }

        std::string key = parse_string(json, pos, error);

        skip_ws(json, pos);
        if (pos >= json.size() || json[pos] != ':') {
            if (error) *error = "expected ':' in object";
            break;
        }
        pos++;

        JsonValue val = parse_value(json, pos, error);
        obj[key] = std::move(val);

        skip_ws(json, pos);
        if (pos >= json.size()) break;

        if (json[pos] == '}') {
            pos++;
            return JsonValue(std::move(obj));
        }

        if (json[pos] != ',') {
            if (error) *error = "expected ',' or '}' in object";
            break;
        }
        pos++;
    }

    return JsonValue(std::move(obj));
}

AppConfig::JsonValue AppConfig::parse_array(const std::string& json,
                                             size_t& pos, std::string* error) {
    std::vector<JsonValue> arr;
    pos++;

    skip_ws(json, pos);
    if (pos < json.size() && json[pos] == ']') {
        pos++;
        return JsonValue(std::move(arr));
    }

    while (pos < json.size()) {
        arr.push_back(parse_value(json, pos, error));

        skip_ws(json, pos);
        if (pos >= json.size()) break;

        if (json[pos] == ']') {
            pos++;
            return JsonValue(std::move(arr));
        }

        if (json[pos] != ',') {
            if (error) *error = "expected ',' or ']' in array";
            break;
        }
        pos++;
    }

    return JsonValue(std::move(arr));
}

AppConfig::JsonValue AppConfig::parse_json(const std::string& json,
                                            size_t& pos, std::string* error) {
    return parse_value(json, pos, error);
}

void AppConfig::serialize_json(const JsonValue& val, std::string& out, int indent) {
    std::string ind(indent, ' ');
    std::string ind_inner(indent + 2, ' ');

    switch (val.type) {
        case JsonValue::NULL_VAL:
            out += "null";
            break;
        case JsonValue::BOOL:
            out += val.bool_val ? "true" : "false";
            break;
        case JsonValue::INT64:
            out += std::to_string(val.int_val);
            break;
        case JsonValue::FLOAT64: {
            char buf[64];
            snprintf(buf, sizeof(buf), "%g", val.float_val);
            out += buf;
            break;
        }
        case JsonValue::STRING:
            out += "\"" + escape_string(val.str_val) + "\"";
            break;
        case JsonValue::ARRAY:
            out += "[";
            for (size_t i = 0; i < val.arr_val.size(); i++) {
                if (i > 0) out += ",";
                serialize_json(val.arr_val[i], out, indent + 2);
            }
            out += "]";
            break;
        case JsonValue::OBJECT:
            out += "{";
            if (!val.obj_val.empty()) {
                bool first = true;
                for (auto& [k, v] : val.obj_val) {
                    if (!first) out += ",";
                    first = false;
                    out += "\"" + escape_string(k) + "\":";
                    serialize_json(v, out, indent + 2);
                }
            }
            out += "}";
            break;
    }
}

AppConfig::JsonValue* AppConfig::resolve_path(const std::string& key) {
    return const_cast<JsonValue*>(const_cast<const AppConfig*>(this)->resolve_path(key));
}

const AppConfig::JsonValue* AppConfig::resolve_path(const std::string& key) const {
    const JsonValue* current = &root_;

    size_t start = 0;
    while (start < key.size()) {
        auto dot = key.find('.', start);
        std::string part = key.substr(start, dot - start);
        start = (dot == std::string::npos) ? key.size() : dot + 1;

        if (current->type != JsonValue::OBJECT) return nullptr;
        auto it = current->obj_val.find(part);
        if (it == current->obj_val.end()) return nullptr;
        current = &it->second;
    }

    return current;
}

float AppConfig::get_float(const std::string& key, float def) const {
    const JsonValue* val = resolve_path(key);
    if (!val) return def;

    switch (val->type) {
        case JsonValue::FLOAT64: return (float)val->float_val;
        case JsonValue::INT64:   return (float)val->int_val;
        case JsonValue::STRING:
            try { return std::stof(val->str_val); } catch (...) { Logger::instance().log(Logger::WARN, std::string("stof parse failed in ") + __func__); return def; }
        case JsonValue::BOOL:    return val->bool_val ? 1.0f : 0.0f;
        default: return def;
    }
}

int AppConfig::get_int(const std::string& key, int def) const {
    const JsonValue* val = resolve_path(key);
    if (!val) return def;

    switch (val->type) {
        case JsonValue::INT64:   return (int)val->int_val;
        case JsonValue::FLOAT64: return (int)val->float_val;
        case JsonValue::STRING:
            try { return std::stoi(val->str_val); } catch (...) { Logger::instance().log(Logger::WARN, std::string("stoi parse failed in ") + __func__); return def; }
        case JsonValue::BOOL:    return val->bool_val ? 1 : 0;
        default: return def;
    }
}

std::string AppConfig::get_string(const std::string& key,
                                   const std::string& def) const {
    const JsonValue* val = resolve_path(key);
    if (!val) return def;

    switch (val->type) {
        case JsonValue::STRING: return val->str_val;
        case JsonValue::INT64:  return std::to_string(val->int_val);
        case JsonValue::FLOAT64: return std::to_string(val->float_val);
        case JsonValue::BOOL:   return val->bool_val ? "true" : "false";
        case JsonValue::NULL_VAL: return "null";
        default: return def;
    }
}

void AppConfig::set(const std::string& key, const std::string& value) {
    if (root_.type != JsonValue::OBJECT && root_.type != JsonValue::NULL_VAL) {
        root_ = JsonValue(std::unordered_map<std::string, JsonValue>());
    }
    if (root_.type == JsonValue::NULL_VAL) {
        root_.type = JsonValue::OBJECT;
    }

    JsonValue* current = &root_;

    size_t start = 0;
    while (start < key.size()) {
        auto dot = key.find('.', start);
        std::string part = key.substr(start, dot - start);
        start = (dot == std::string::npos) ? key.size() : dot + 1;

        if (start >= key.size()) {
            current->obj_val[part] = auto_type_value(value);
        } else {
            auto it = current->obj_val.find(part);
            if (it == current->obj_val.end() ||
                it->second.type != JsonValue::OBJECT) {
                current->obj_val[part] = JsonValue(
                    std::unordered_map<std::string, JsonValue>());
            }
            current = &current->obj_val[part];
        }
    }
}

void AppConfig::save(const std::string& path) {
    std::string json = to_json();
    std::ofstream f(path);
    if (f) f << json << std::endl;
}

std::string AppConfig::to_json() const {
    std::string out;
    serialize_json(root_, out, 0);
    return out;
}

bool AppConfig::validate(std::string* error_out) const {
    std::string error;
    size_t pos = 0;
    std::string json = to_json();
    JsonValue test = parse_json(json, pos, &error);
    if (!error.empty()) {
        if (error_out) *error_out = error;
        return false;
    }
    return pos >= json.size();
}

} // namespace quant
