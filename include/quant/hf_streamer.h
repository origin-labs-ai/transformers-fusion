// ============================================================================
// hf_streamer.h — HuggingFace dataset streaming via HTTP + Parquet parsing
// ============================================================================
#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>
#endif

namespace quant {

// ============================================================================
// SimpleJSON — lightweight JSON extraction for HF API responses
// ============================================================================
struct SimpleJSON {
    static std::string find_string(const std::string& json, const std::string& key);
    static std::string find_array(const std::string& json, const std::string& key);
    static std::string find_object(const std::string& json, const std::string& key);
    static std::vector<std::string> find_all_strings_in_array(const std::string& json_array);
    static std::vector<std::string> find_all_in_array(const std::string& json_array,
                                                       const std::string& key);
    static std::vector<std::string> find_all_rfilenames(const std::string& json_array);
    static std::string url_encode(const std::string& s);
    static std::string escape(const std::string& s);
};

// ============================================================================
// HTTPClient — HTTP GET via WinHTTP (Windows) or curl (cross-platform)
// ============================================================================
class HTTPClient {
public:
    HTTPClient();
    ~HTTPClient();

    void set_auth_token(const std::string& token) { auth_token_ = token; }

    std::string get(const std::string& host, const std::string& path,
                    bool https = true);
    std::vector<uint8_t> range_request(const std::string& host,
                                        const std::string& path,
                                        int64_t start, int64_t end);
    int64_t get_content_length(const std::string& host, const std::string& path);
    std::vector<uint8_t> get_binary(const std::string& host,
                                     const std::string& path,
                                     bool https = true);

private:
    void* session_ = nullptr;  // HINTERNET on Windows; unused on POSIX
    std::string auth_token_;
};

// ============================================================================
// HFStreamer — stream text data from HuggingFace datasets
// ============================================================================
class HFStreamer {
public:
    HFStreamer() = default;
    ~HFStreamer() = default;

    bool open(const std::string& dataset_name, const std::string& split = "train");
    bool next_sample(std::string& text);
    void close();

    bool is_open() const { return is_open_; }
    int64_t samples_streamed() const { return samples_streamed_; }

private:
    bool is_open_ = false;
    int64_t samples_streamed_ = 0;
    std::string dataset_name_;
    std::string split_;
};

} // namespace quant
