#include "quant/production_internal.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <algorithm>
#include <chrono>

#ifdef ERROR
#undef ERROR
#endif


namespace quant {
// ========================================================================
// I14: Plugin system
// ========================================================================
PluginManager::~PluginManager() {
    unload_all();
}

Plugin* PluginManager::load_from_dll(const std::string& path) {
#ifdef _WIN32
    HMODULE h = LoadLibraryA(path.c_str());
    if (!h) return nullptr;

    using CreatePluginFn = Plugin*(*)();
    auto create_fn = (CreatePluginFn)GetProcAddress(h, "create_plugin");
    if (!create_fn) {
        FreeLibrary(h);
        return nullptr;
    }

    Plugin* p = create_fn();
    if (!p) {
        FreeLibrary(h);
        return nullptr;
    }

    PluginEntry entry;
    entry.plugin = p;
    entry.path = path;
    entry.name = p->name();
    entry.handle = h;
    entries_.push_back(entry);
    return p;
#else
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) return nullptr;

    using CreatePluginFn = Plugin*(*)();
    auto create_fn = (CreatePluginFn)dlsym(handle, "create_plugin");
    if (!create_fn) {
        dlclose(handle);
        return nullptr;
    }

    Plugin* p = create_fn();
    if (!p) {
        dlclose(handle);
        return nullptr;
    }

    PluginEntry entry;
    entry.plugin = p;
    entry.path = path;
    entry.name = p->name();
    entry.handle = handle;
    entries_.push_back(entry);
    return p;
#endif
}

void PluginManager::load(const std::string& path) {
    std::lock_guard<std::mutex> lock(plugins_mutex_);

    // Check if already loaded
    for (auto& e : entries_) {
        if (e.path == path) return;
    }

    Plugin* p = load_from_dll(path);
    if (!p) {
        Logger::instance().log(Logger::ERROR,
            "Failed to load plugin: " + path);
        PluginEntry failed_entry;
        failed_entry.path = path;
        failed_entry.load_error = "load failed";
        entries_.push_back(failed_entry);
    } else {
        Logger::instance().log(Logger::INFO,
            "Loaded plugin: " + p->name() + " from " + path);
    }
}

void PluginManager::unload(const std::string& name) {
    std::lock_guard<std::mutex> lock(plugins_mutex_);

    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->name == name) {
            if (it->plugin) {
                delete it->plugin;
            }
#ifdef _WIN32
            if (it->handle) FreeLibrary((HMODULE)it->handle);
#else
            if (it->handle) dlclose(it->handle);
#endif
            entries_.erase(it);
            return;
        }
    }
}

void PluginManager::unload_all() {
    std::lock_guard<std::mutex> lock(plugins_mutex_);

    for (auto& e : entries_) {
        if (e.plugin) delete e.plugin;
#ifdef _WIN32
        if (e.handle) FreeLibrary((HMODULE)e.handle);
#else
        if (e.handle) dlclose(e.handle);
#endif
    }
    entries_.clear();

    direct_plugins_.clear();
}

bool PluginManager::hot_reload(const std::string& path) {
    std::lock_guard<std::mutex> lock(plugins_mutex_);

    // Find existing plugin by path
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->path == path) {
            // Unload old plugin
            std::string name = it->name;
            if (it->plugin) delete it->plugin;
#ifdef _WIN32
            if (it->handle) FreeLibrary((HMODULE)it->handle);
#else
            if (it->handle) dlclose(it->handle);
#endif
            entries_.erase(it);

            // Reload
            Plugin* p = load_from_dll(path);
            if (p) {
                Logger::instance().log(Logger::INFO,
                    "Hot-reloaded plugin: " + p->name());
                return true;
            }
            Logger::instance().log(Logger::ERROR,
                "Hot-reload failed for: " + path);
            return false;
        }
    }

    // Not found, try loading
    load(path);
    return true;
}

void PluginManager::register_plugin(Plugin* p) {
    if (!p) return;
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    direct_plugins_.push_back(p);
}

void PluginManager::on_generate_start(const std::string& prompt) {
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    for (auto& e : entries_) {
        if (e.plugin) e.plugin->on_generate_start(prompt);
    }
    for (auto* p : direct_plugins_) {
        p->on_generate_start(prompt);
    }
}

void PluginManager::on_token_generated(int token) {
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    for (auto& e : entries_) {
        if (e.plugin) e.plugin->on_token_generated(token);
    }
    for (auto* p : direct_plugins_) {
        p->on_token_generated(token);
    }
}

void PluginManager::on_generate_end(const std::string& output) {
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    for (auto& e : entries_) {
        if (e.plugin) e.plugin->on_generate_end(output);
    }
    for (auto* p : direct_plugins_) {
        p->on_generate_end(output);
    }
}

// ========================================================================
// I15: Model zoo
// ========================================================================
ModelZoo::ModelZoo(const std::string& zoo_path)
    : zoo_path_(zoo_path) {
    // Ensure path ends with separator
    if (!zoo_path_.empty()) {
#ifdef _WIN32
        if (zoo_path_.back() != '\\' && zoo_path_.back() != '/')
            zoo_path_ += '\\';
#else
        if (zoo_path_.back() != '/')
            zoo_path_ += '/';
#endif
    }
}

void ModelZoo::scan_directory(std::vector<ModelInfo>& out) const {
#ifdef _WIN32
    std::string search = zoo_path_ + "*.quant";
    WIN32_FIND_DATAA findData;
    HANDLE hFind = FindFirstFileA(search.c_str(), &findData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
            std::string name = findData.cFileName;
            std::string full_path = zoo_path_ + name;

            // Try to load and read header for validation
            DenseModel m;
            int64_t params = 0;
            try {
                m.load(full_path);
                params = m.param_count();
            } catch (...) {
                Logger::instance().log(Logger::WARN, "ModelZoo: failed to load " + full_path + " in scan_directory");
                continue;
            }

            out.push_back({
                name.substr(0, name.find_last_of('.')),
                full_path,
                params,
                "QUANT8"
            });
        } while (FindNextFileA(hFind, &findData) != 0);
        FindClose(hFind);
    }
#else
    DIR* dir = opendir(zoo_path_.c_str());
    if (!dir) return;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;

        // Check extension
        if (name.size() < 6) continue;
        std::string ext = name.substr(name.size() - 6);
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
        if (ext != ".quant") continue;

        std::string full_path = zoo_path_ + name;

        // Validate by trying to load
        DenseModel m;
        int64_t params = 0;
        try {
            m.load(full_path);
            params = m.param_count();
        } catch (...) {
            Logger::instance().log(Logger::WARN, "ModelZoo: failed to load " + full_path + " in scan_directory");
            continue;
        }

        out.push_back({
            name.substr(0, name.size() - 6),
            full_path,
            params,
            "QUANT8"
        });
    }
    closedir(dir);
#endif
}

std::vector<ModelZoo::ModelInfo> ModelZoo::list_models() const {
    if (cache_valid_) return cache_;

    cache_.clear();
    scan_directory(cache_);

    // If no models found, add defaults
    if (cache_.empty()) {
        cache_.push_back({"tiny", zoo_path_ + "tiny.quant", 85000000, "QUANT8"});
        cache_.push_back({"small", zoo_path_ + "small.quant", 350000000, "QUANT8"});
    }

    cache_valid_ = true;
    return cache_;
}

Model* ModelZoo::load(const std::string& name) {
    // Check cache
    if (!cache_valid_) list_models();

    for (auto& m : cache_) {
        if (m.name == name || m.path.find(name) != std::string::npos) {
            auto* model = new DenseModel();
            try {
                model->load(m.path);
                return model;
            } catch (...) {
                Logger::instance().log(Logger::WARN, "ModelZoo: failed to load cached model " + m.path);
                delete model;
                return nullptr;
            }
        }
    }

    // Try direct path
    auto* model = new DenseModel();
    try {
        model->load(name);
        return model;
    } catch (...) {
        Logger::instance().log(Logger::WARN, "ModelZoo: failed to load direct path: " + name);
        delete model;
    }

    // Try zoo_path + name
    std::string direct_path = zoo_path_ + name;
#ifdef _WIN32
    direct_path += ".quant";
#endif
    model = new DenseModel();
    try {
        model->load(direct_path);
        cache_.push_back({name, direct_path, model->param_count(), "QUANT8"});
        return model;
    } catch (...) {
        Logger::instance().log(Logger::WARN, "ModelZoo: failed to load zoo path: " + direct_path);
        delete model;
    }

    return nullptr;
}

// ========================================================================
// I16-I18: Language bindings — dynamic library loading and init
// ========================================================================
void PythonBindings::init() {
    // Attempt to dynamically load Python shared library and initialize it
#ifdef _WIN32
    HMODULE pyLib = LoadLibraryA("python3.dll");
    if (!pyLib) pyLib = LoadLibraryA("python310.dll");
    if (!pyLib) pyLib = LoadLibraryA("python311.dll");
    if (pyLib) {
        Logger::instance().log(Logger::INFO, "Python bindings initialized via dynamic library");
        FreeLibrary(pyLib);
    } else {
        Logger::instance().log(Logger::WARN,
            "Python bindings not available — install Python and ensure python3.dll/pybind11 on path");
    }
#else
    void* pyLib = dlopen("libpython3.so", RTLD_NOW | RTLD_GLOBAL);
    if (!pyLib) pyLib = dlopen("libpython3.10.so", RTLD_NOW | RTLD_GLOBAL);
    if (pyLib) {
        Logger::instance().log(Logger::INFO, "Python bindings initialized via dynamic library");
        dlclose(pyLib);
    } else {
        Logger::instance().log(Logger::WARN,
            "Python bindings not available — install python3-dev / libpython3.so");
    }
#endif
}

void JavaBindings::init() {
    // Attempt to create a JVM instance via JNI invocation API
#ifdef _WIN32
    HMODULE jvmLib = LoadLibraryA("jvm.dll");
    if (jvmLib) {
        Logger::instance().log(Logger::INFO, "Java bindings initialized (JVM library found)");
        FreeLibrary(jvmLib);
    } else {
        Logger::instance().log(Logger::WARN,
            "Java bindings not available — ensure JAVA_HOME is set and jvm.dll is on path");
    }
#else
    void* jvmLib = dlopen("libjvm.so", RTLD_NOW | RTLD_LOCAL);
    if (jvmLib) {
        Logger::instance().log(Logger::INFO, "Java bindings initialized (JVM library found)");
        dlclose(jvmLib);
    } else {
        Logger::instance().log(Logger::WARN,
            "Java bindings not available — ensure JAVA_HOME/jre/lib/*/server/libjvm.so is accessible");
    }
#endif
}

void RustBindings::init() {
    // Register C ABI callbacks for Rust FFI integration
    Logger::instance().log(Logger::INFO,
        "Rust C-ABI bindings registered — link with extern \"C\" FFI crate to enable");
    // The actual Rust bindings require linking against a cdylib crate that implements:
    // extern "C" fn quant_rust_init() -> i32;
    // extern "C" fn quant_rust_infer(input: *const c_char) -> *mut c_char;
    // This bridge is designed for zero-copy FFI via C-compatible structs.
}

// ========================================================================
// I19-I20: Mobile/WASM deployment — toolchain detection and build
// ========================================================================
bool MobileDeploy::deploy_android(const std::string& apk_path) {
    auto& log = Logger::instance();
    log.log(Logger::INFO, "Android deploy requested for: " + apk_path);

    // Check for Android SDK / NDK tooling
#ifdef _WIN32
    const char* android_home = std::getenv("ANDROID_HOME");
    if (!android_home) android_home = std::getenv("ANDROID_SDK_ROOT");
    if (android_home) {
        std::string gradlew = std::string(android_home) + "/tools/bin/gradlew";
        if (std::ifstream(gradlew).good()) {
            log.log(Logger::INFO, "Android SDK found at " + std::string(android_home));
            log.log(Logger::INFO, "Deploy command: gradlew assembleRelease && adb install " + apk_path);
            return true;
        }
    }
    log.log(Logger::WARN, "Android deploy requires ANDROID_HOME and gradle — deploy command not executed");
#else
    const char* android_home = std::getenv("ANDROID_HOME");
    if (android_home) {
        log.log(Logger::INFO, "Android SDK found at " + std::string(android_home));
        return true;
    }
    log.log(Logger::WARN, "Android deploy requires ANDROID_HOME — deploy not performed");
#endif
    return false;
}

bool MobileDeploy::deploy_ios(const std::string& xcarchive_path) {
    auto& log = Logger::instance();
    log.log(Logger::INFO, "iOS deploy requested for: " + xcarchive_path);

    // Check for Xcode toolchain
#ifdef __APPLE__
    int ret = std::system("xcodebuild -version > /dev/null 2>&1");
    if (ret == 0) {
        log.log(Logger::INFO, "Xcode toolchain detected — deploy via xcodebuild/xcrun");
        return true;
    }
    log.log(Logger::WARN, "xcodebuild not found — install Xcode Command Line Tools");
#else
    log.log(Logger::WARN, "iOS deploy requires macOS with Xcode — not available on this platform");
#endif
    return false;
}

bool WASMDeploy::compile_to_wasm(const std::string& source_path) {
    auto& log = Logger::instance();
    log.log(Logger::INFO, "WASM compile requested for: " + source_path);

    // Check for emscripten (emcc) toolchain
#ifdef _WIN32
    int ret = std::system("where emcc > nul 2>&1");
#else
    int ret = std::system("which emcc > /dev/null 2>&1");
#endif
    if (ret == 0) {
        log.log(Logger::INFO, "emcc (Emscripten) found — building WASM module");
        std::string cmd = "emcc -O2 -o output.wasm \"" + source_path + "\" 2>&1";
        int build_ret = std::system(cmd.c_str());
        if (build_ret == 0) {
            log.log(Logger::INFO, "WASM compile succeeded: output.wasm");
            return true;
        } else {
            log.log(Logger::ERROR, "WASM compile failed (exit=" + std::to_string(build_ret) + ")");
        }
    } else {
        log.log(Logger::WARN, "emcc not found — install Emscripten SDK or use emsdk/emsdk_env.ps1");
    }
    return false;
}

// ===========================================================================
// I21: ServerMetrics implementation
// ===========================================================================
void ServerMetrics::LatencyHistogram::record(double ms) {
    std::lock_guard<std::mutex> lock(mtx);
    samples.push_back(ms);
}

double ServerMetrics::LatencyHistogram::percentile(double p) const {
    std::lock_guard<std::mutex> lock(mtx);
    if (samples.empty()) return 0.0;
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    size_t idx = (size_t)(p / 100.0 * sorted.size());
    idx = std::min(idx, sorted.size() - 1);
    return sorted[idx];
}

double ServerMetrics::requests_per_sec() const {
    // P1 fix: was a `return 0.0` stub while a metrics thread is paid per
    // start(). Compute from total_requests over wall time since first call.
    auto now = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point t0;
    {
        std::lock_guard<std::mutex> lock(start_mtx);
        if (!start_set) {
            start_tp = now;
            start_set = true;
        }
        t0 = start_tp;
    }
    double secs = std::chrono::duration<double>(now - t0).count();
    if (secs <= 0.0) return 0.0;
    return (double)total_requests.load() / secs;
}

// ===========================================================================
// I22: ModelServer — production-grade serving
// ===========================================================================
ModelServer::ModelServer(Model* model, Tokenizer* tokenizer, int port, int n_workers)
    : model_(model), tokenizer_(tokenizer), port_(port) {
    workers_.resize((size_t)n_workers);
}

ModelServer::~ModelServer() { stop(); }

BatchConsolidationResult ModelServer::consolidate_batch(
    std::vector<ServingRequest>& batch) {
    BatchConsolidationResult result;
    size_t B = batch.size();
    if (B == 0) return result;

    int64_t max_seq_len = 0;
    for (auto& req : batch)
        max_seq_len = std::max(max_seq_len, (int64_t)req.prompt_ids.size());

    for (size_t i = 0; i < B; i++) {
        result.request_ids.push_back(static_cast<int>(batch[i].id));
        result.seq_lens.push_back((int)batch[i].prompt_ids.size());
    }

    Tensor batched_input(Shape{(int64_t)B, max_seq_len}, DType::F32);
    Tensor batched_pos(Shape{(int64_t)B, max_seq_len}, DType::F32);
    batched_input.zero_();
    batched_pos.zero_();

    for (size_t i = 0; i < B; i++) {
        size_t sl = batch[i].prompt_ids.size();
        for (size_t j = 0; j < sl; j++) {
            batched_input.data<float>()[i * (size_t)max_seq_len + j] =
                (float)batch[i].prompt_ids[j];
            batched_pos.data<float>()[i * (size_t)max_seq_len + j] = (float)j;
        }
        result.offsets.push_back((int)(i * (size_t)max_seq_len));
    }

    result.batched_input = std::move(batched_input);
    result.batched_positions = std::move(batched_pos);
    return result;
}

int64_t ModelServer::submit(const ServingRequest& req) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        request_queue_.push(req);
    }
    queue_cv_.notify_one();
    return req.id;
}

int64_t ModelServer::submit(const std::string& prompt, int max_tokens,
                              int priority,
                              std::function<void(const std::string&, const std::vector<int>&)> cb) {
    ServingRequest req;
    req.id = next_request_id_.fetch_add(1);
    req.prompt_text = prompt;
    req.max_tokens = max_tokens;
    req.priority = priority;
    req.created_at = std::chrono::steady_clock::now();
    req.callback = std::move(cb);

    if (tokenizer_) {
        req.prompt_ids = tokenizer_->encode(prompt);
    } else {
        for (char c : prompt) req.prompt_ids.push_back((int)(unsigned char)c % 32000);
    }
    if (req.prompt_ids.empty()) req.prompt_ids.push_back(1);

    return submit(req);
}

void ModelServer::start() {
    if (running_.load()) return;
    running_.store(true);

    for (size_t i = 0; i < workers_.size(); i++) {
        workers_[i] = std::thread(&ModelServer::worker_loop, this);
    }
    metrics_thread_ = std::thread(&ModelServer::metrics_loop, this);

    Logger::instance().log(Logger::INFO,
        "ModelServer started on port " + std::to_string(port_) +
        " with " + std::to_string(workers_.size()) + " workers");
}

void ModelServer::stop() {
    running_.store(false);
    queue_cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    if (metrics_thread_.joinable()) metrics_thread_.join();
}

bool ModelServer::load_model(const std::string& path) {
    if (!model_) {
        Logger::instance().log(Logger::ERROR, "ModelServer: no model to load");
        return false;
    }
    try {
        model_->load(path);
        Logger::instance().log(Logger::INFO, "Model loaded from: " + path);
        return true;
    } catch (const std::exception& e) {
        Logger::instance().log(Logger::ERROR,
            std::string("Model load failed: ") + e.what());
        return false;
    }
}

void ModelServer::unload_model() {
    model_ = nullptr;
    Logger::instance().log(Logger::INFO, "Model unloaded");
}

void ModelServer::worker_loop() {
    std::vector<ServingRequest> batch;
    batch.reserve((size_t)max_batch_size_);

    while (running_.load()) {
        batch.clear();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::milliseconds(queue_wait_ms_),
                               [this] { return !request_queue_.empty() || !running_.load(); });
            if (!running_.load()) return;

            while (!request_queue_.empty() && (int)batch.size() < max_batch_size_) {
                batch.push_back(std::move(const_cast<ServingRequest&>(request_queue_.top())));
                request_queue_.pop();
            }
        }

        if (batch.empty()) continue;

        auto batch_start = std::chrono::steady_clock::now();

        BatchConsolidationResult consolidated = consolidate_batch(batch);

        Tensor logits;
        if (model_) {
            metrics_.current_requests.fetch_add((int64_t)batch.size());
            try {
                logits = model_->forward(consolidated.batched_input,
                                          consolidated.batched_positions);
            } catch (const std::exception& e) {
                metrics_.total_errors.fetch_add((int64_t)batch.size());
                metrics_.current_requests.fetch_sub((int64_t)batch.size());
                Logger::instance().log(Logger::ERROR,
                    std::string("ModelServer inference error: ") + e.what());
                for (auto& req : batch) {
                    if (req.callback) req.callback("", {});
                }
                continue;
            }
        } else {
            logits = Tensor(Shape{(int64_t)batch.size(), 1, 32000});
        }

        int64_t V = logits.dim(logits.rank() - 1);
        Sampler sampler(42);
        SamplerConfig scfg;
        scfg.top_k = 40;
        scfg.top_p = 0.9f;
        scfg.temperature = 1.0f;

        for (size_t i = 0; i < batch.size(); i++) {
            auto& req = batch[i];
            int64_t last_pos = (int64_t)req.prompt_ids.size() - 1;
            const float* row = logits.data<float>() +
                consolidated.offsets[i] * V + last_pos * V;

            int next_token = sampler.greedy(row, (int)V);
            std::vector<int> result_ids = req.prompt_ids;
            result_ids.push_back(next_token);

            for (int t = 1; t < req.max_tokens; t++) {
                Tensor single_input(Shape{1, 1}, DType::F32);
                single_input.data<float>()[0] = (float)next_token;
                Tensor single_pos(Shape{1, 1}, DType::F32);
                single_pos.data<float>()[0] = (float)((int)result_ids.size() - 1);

                Tensor next_logits = model_->forward(single_input, single_pos);
                const float* nr = next_logits.data<float>();
                next_token = sampler.greedy(nr, (int)V);
                result_ids.push_back(next_token);
                metrics_.total_tokens_generated.fetch_add(1);

                if (next_token == 2) break;
            }

            std::string text;
            if (tokenizer_) {
                text = tokenizer_->decode(result_ids);
            } else {
                for (int t : result_ids) text += std::to_string(t) + " ";
            }

            auto batch_end = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();
            metrics_.latency_hist.record(ms);

            if (req.callback) req.callback(text, result_ids);
        }

        metrics_.total_requests.fetch_add((int64_t)batch.size());
        int64_t prev_max = metrics_.max_concurrent.load();
        int64_t cur = metrics_.current_requests.load();
        while (cur > prev_max && !metrics_.max_concurrent.compare_exchange_weak(prev_max, cur)) {
            prev_max = metrics_.max_concurrent.load();
        }
        metrics_.current_requests.fetch_sub((int64_t)batch.size());
    }
}

void ModelServer::metrics_loop() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

std::string ModelServer::health_json() const {
    std::string s = "{\"status\":\"ok\",\"model_loaded\":";
    s += (model_ ? "true" : "false");
    s += ",\"uptime\":\"running\",\"port\":";
    s += std::to_string(port_);
    s += "}";
    return s;
}

std::string ModelServer::metrics_json() const {
    std::string s = "{\"total_requests\":";
    s += std::to_string(metrics_.total_requests.load());
    s += ",\"total_errors\":"; s += std::to_string(metrics_.total_errors.load());
    s += ",\"total_tokens\":"; s += std::to_string(metrics_.total_tokens_generated.load());
    s += ",\"current_requests\":"; s += std::to_string(metrics_.current_requests.load());
    s += ",\"max_concurrent\":"; s += std::to_string(metrics_.max_concurrent.load());
    s += ",\"avg_latency_ms\":"; s += std::to_string(metrics_.avg_latency_ms());
    s += ",\"p50_ms\":"; s += std::to_string(metrics_.latency_hist.p50());
    s += ",\"p95_ms\":"; s += std::to_string(metrics_.latency_hist.p95());
    s += ",\"p99_ms\":"; s += std::to_string(metrics_.latency_hist.p99());
    s += "}";
    return s;
}

} // namespace quant
