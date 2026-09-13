#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>
#endif
#include "quant/hf_streamer.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <random>
#include <sstream>

#if defined(_WIN32)
#pragma comment(lib, "winhttp.lib")
#endif

namespace quant {

// ════════════════════════════════════════════════════════════════
// SimpleJSON
// ════════════════════════════════════════════════════════════════

std::string SimpleJSON::find_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos += search.size();
    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            if (json[pos + 1] == 'n') result += '\n';
            else if (json[pos + 1] == 't') result += '\t';
            else if (json[pos + 1] == '\\') result += '\\';
            else if (json[pos + 1] == '"') result += '"';
            else result += json[pos + 1];
            pos += 2;
        } else {
            result += json[pos++];
        }
    }
    return result;
}

std::string SimpleJSON::find_array(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":[";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos += search.size();
    int depth = 1;
    size_t start = pos;
    while (pos < json.size() && depth > 0) {
        if (json[pos] == '[') depth++;
        else if (json[pos] == ']') depth--;
        pos++;
    }
    return json.substr(start, pos - start - 1);
}

std::string SimpleJSON::find_object(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":{";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos += search.size() - 1; // point at '{'
    int depth = 0;
    size_t start = pos;
    while (pos < json.size()) {
        if (json[pos] == '{') depth++;
        else if (json[pos] == '}') { depth--; if (depth == 0) break; }
        pos++;
    }
    return json.substr(start, pos - start + 1);
}

std::vector<std::string> SimpleJSON::find_all_strings_in_array(const std::string& json_array) {
    std::vector<std::string> result;
    size_t pos = 0;
    while ((pos = json_array.find("\"", pos)) != std::string::npos) {
        size_t start = pos + 1;
        size_t end = json_array.find("\"", start);
        if (end == std::string::npos) break;
        result.push_back(json_array.substr(start, end - start));
        pos = end + 1;
    }
    return result;
}

std::vector<std::string> SimpleJSON::find_all_in_array(const std::string& json_array,
                                                         const std::string& key) {
    std::vector<std::string> result;
    size_t pos = 0;
    std::string search = "\"" + key + "\":\"";
    while ((pos = json_array.find(search, pos)) != std::string::npos) {
        pos += search.size();
        std::string val;
        while (pos < json_array.size() && json_array[pos] != '"') {
            if (json_array[pos] == '\\' && pos + 1 < json_array.size()) {
                pos += 2;
            } else {
                val += json_array[pos++];
            }
        }
        if (!val.empty()) result.push_back(val);
        pos++;
    }
    return result;
}

std::vector<std::string> SimpleJSON::find_all_rfilenames(const std::string& json_array) {
    return find_all_in_array(json_array, "rfilename");
}

std::string SimpleJSON::url_encode(const std::string& s) {
    std::string result;
    for (char c : s) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            result += c;
        } else if (c == '/') {
            result += '/';
        } else {
            char buf[8];
            snprintf(buf, sizeof(buf), "%%%02X", (unsigned char)c);
            result += buf;
        }
    }
    return result;
}

std::string SimpleJSON::escape(const std::string& s) {
    std::string result;
    for (char c : s) {
        if (c == '"') result += "\\\"";
        else if (c == '\\') result += "\\\\";
        else if (c == '\n') result += "\\n";
        else if (c == '\t') result += "\\t";
        else result += c;
    }
    return result;
}

// ════════════════════════════════════════════════════════════════
// HTTPClient (WinHTTP wrapper)
// ════════════════════════════════════════════════════════════════

HTTPClient::HTTPClient() : session_(nullptr) {
#if defined(_WIN32)
    session_ = WinHttpOpen(L"Transcender/1.0", WINHTTP_ACCESS_TYPE_NO_PROXY,
                           WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
#else
    // POSIX: all transfers go through the curl subprocess (get_binary and
    // friends below); no native session handle needed.
    session_ = nullptr;
#endif
}

HTTPClient::~HTTPClient() {
#if defined(_WIN32)
    if (session_) WinHttpCloseHandle((HINTERNET)session_);
#endif
}

#if defined(_WIN32)
static std::string wide_to_utf8(const wchar_t* wstr) {
    int len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, NULL, 0, NULL, NULL);
    if (len <= 0) return "";
    std::string result((size_t)len - 1, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, &result[0], len, NULL, NULL);
    return result;
}

static std::wstring utf8_to_wide(const std::string& str) {
    int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, NULL, 0);
    if (len <= 0) return L"";
    std::wstring result((size_t)len - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &result[0], len);
    return result;
}
#endif // defined(_WIN32)

std::string HTTPClient::get(const std::string& host, const std::string& path, bool https) {
    // Custom headers are baked into the curl command lines below; no separate
    // header state needed (the WinHTTP session path was already unused here).
    auto binary = get_binary(host, path, https);
    return std::string(binary.begin(), binary.end());
}

// Range request — downloads only byte range [start, end] (inclusive if end >= 0, suffix if start < 0)
std::vector<uint8_t> HTTPClient::range_request(const std::string& host, const std::string& path, int64_t start, int64_t end) {
    std::string url = (host.find("://") != std::string::npos) ? host : "https://" + host + path;
    std::string range_str;
    if (start < 0) {
        range_str = "Range: bytes=" + std::to_string(-start) + "\r\n";
    } else {
        range_str = "Range: bytes=" + std::to_string(start) + "-";
        if (end >= start) range_str += std::to_string(end);
        range_str += "\r\n";
    }
    std::string cmd =
#if defined(_WIN32)
        "curl.exe -s -L -o - -H \"" + range_str + "\"";
#else
        "curl -s -L -o - -H \"" + range_str + "\"";
#endif
    if (!auth_token_.empty()) {
        cmd += " -H \"Authorization: Bearer " + auth_token_ + "\"";
    }
    cmd += " \"" + url + "\"";
    
#if defined(_WIN32)
    FILE* pipe = _popen(cmd.c_str(), "rb");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) return {};
    std::vector<uint8_t> result;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), pipe)) > 0)
        result.insert(result.end(), buf, buf + n);
#if defined(_WIN32)
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return result;
}

int64_t HTTPClient::get_content_length(const std::string& host, const std::string& path) {
    std::string url = "https://" + host + path;
#if defined(_WIN32)
    std::string cmd = "curl.exe -s -I -L";
#else
    std::string cmd = "curl -s -I -L";
#endif
    if (!auth_token_.empty()) {
        cmd += " -H \"Authorization: Bearer " + auth_token_ + "\"";
    }
    cmd += " \"" + url + "\"";
    
#if defined(_WIN32)
    FILE* pipe = _popen(cmd.c_str(), "rb");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) return -1;
    std::string hdr;
    char buf[1024];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf) - 1, pipe)) > 0) {
        buf[n] = 0;
        hdr += buf;
    }
#if defined(_WIN32)
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    // Parse Content-Length from headers
    size_t pos = hdr.find("Content-Length: ");
    if (pos == std::string::npos) pos = hdr.find("content-length: ");
    if (pos == std::string::npos) return -1;
    pos = hdr.find(':', pos) + 1;
    while (pos < hdr.size() && hdr[pos] == ' ') pos++;
    size_t end = pos;
    while (end < hdr.size() && hdr[end] >= '0' && hdr[end] <= '9') end++;
    if (end == pos) return -1;
    return std::stoll(hdr.substr(pos, end - pos));
}

std::vector<uint8_t> HTTPClient::get_binary(const std::string& host, const std::string& path, bool https) {
    // Use curl for downloads (Windows built-in curl.exe; system curl on POSIX).
    std::string url = (https ? "https://" : "http://") + host + path;
#if defined(_WIN32)
    std::string cmd = "curl.exe -s -L -o -";
#else
    std::string cmd = "curl -s -L -o -";
#endif
    if (!auth_token_.empty()) {
        cmd += " -H \"Authorization: Bearer " + auth_token_ + "\"";
    }
    cmd += " \"" + url + "\"";
    
#if defined(_WIN32)
    FILE* pipe = _popen(cmd.c_str(), "rb");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) {
        fprintf(stderr, "    [curl] popen failed\n");
        return {};
    }
    
    std::vector<uint8_t> result;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), pipe)) > 0) {
        result.insert(result.end(), buf, buf + n);
    }
#if defined(_WIN32)
    int rc = _pclose(pipe);
#else
    int rc = pclose(pipe);
#endif
    if (rc != 0) {
        fprintf(stderr, "    [curl] exit %d, got %zu bytes\n", rc, result.size());
    }
    return result;
}

// ════════════════════════════════════════════════════════════════
// ParquetTextReader — minimal parquet reader (text columns only)
// ════════════════════════════════════════════════════════════════

// Minimal Thrift Compact Protocol reader
struct CompactReader {
    const uint8_t* p;
    const uint8_t* end;
    
    CompactReader(const uint8_t* d, size_t sz) : p(d), end(d + sz) {}
    
    int64_t read_varint() {
        int64_t val = 0, shift = 0;
        while (p < end) {
            uint8_t b = *p++;
            val |= (int64_t)(b & 0x7F) << shift;
            shift += 7;
            if (!(b & 0x80)) return val;
        }
        return val;
    }
    
    // Compact protocol I32 is zigzag-encoded varint
    int32_t read_i32() {
        int64_t v = read_varint();
        return (int32_t)((v >> 1) ^ -(v & 1));
    }
    // Compact protocol I64 is zigzag-encoded varint  
    int64_t read_i64() {
        int64_t v = read_varint();
        return (v >> 1) ^ -(v & 1);
    }
    
    // Read zigzag varint (alias)
    int64_t read_zigzag() {
        return read_i64();
    }
    
    // Read Thrift Compact Protocol list header:
    // Single byte: element_type in low 4 bits, count in high 4 bits (0-14)
    // If count == 15, actual count follows as varint
    struct ListHeader { uint8_t elem_type; int64_t count; };
    ListHeader read_list_header() {
        if (p >= end) return {0, 0};
        uint8_t b = *p++;
        uint8_t elem_type = b & 0x0F;
        int64_t count = (b >> 4) & 0x0F;
        if (count == 0x0F) {
            count = read_varint();
        }
        return {elem_type, count};
    }
    
    // Read a fixed-length byte array prefixed by varint length
    std::vector<uint8_t> read_bytes() {
        int64_t len = read_varint();
        if (len <= 0 || p + len > end) return {};
        std::vector<uint8_t> result(p, p + len);
        p += len;
        return result;
    }
    
    // Read binary field (varint length + bytes)
    std::string read_string() {
        auto bytes = read_bytes();
        return std::string(bytes.begin(), bytes.end());
    }
    
    // Skip a field of given type in Compact Protocol
    void skip_field(uint8_t type) {
        switch (type) {
            case 0: break; // STOP
            case 1: case 2: break; // TRUE / FALSE
            case 3: p++; break; // BYTE
            case 4: // I16 — compact: zigzag varint
            case 5: // I32 — compact: zigzag varint
            case 6: // I64 — compact: zigzag varint
                read_varint();
                break;
            case 7: // DOUBLE — 8 bytes
                p += 8; break;
            case 8: { // BINARY — varint length + data
                int64_t len = read_varint();
                if (len > 0 && p + len <= end) p += len;
                break;
            }
            case 9: // LIST — list header + elements
            case 10: { // SET
                auto lh = read_list_header();
                for (int64_t i = 0; i < lh.count; i++) {
                    skip_field(lh.elem_type);
                }
                break;
            }
            case 11: { // MAP (not used in parquet, but handle generically)
                auto lh = read_list_header();
                if (p >= end) break;
                uint8_t val_type = *p++;
                for (int64_t i = 0; i < lh.count; i++) {
                    skip_field(lh.elem_type);
                    skip_field(val_type);
                }
                break;
            }
            case 12: // STRUCT — fields until STOP
            default: {
                while (p < end) {
                    auto sf = read_field();
                    if (sf.type == 0) break;
                    skip_field(sf.type);
                }
                break;
            }
        }
    }
    
    // Read a Thrift struct field header
    // Returns {field_type, field_id} or {0, 0} for STOP
    struct Field { uint8_t type; int16_t id; };
    Field read_field() {
        if (p >= end) return {0, 0};
        uint8_t b = *p++;
        if (b == 0) return {0, 0};
        uint8_t type = b & 0x0F;
        int16_t delta = (b >> 4) & 0x0F;
        if (delta == 0) {
            delta = (int16_t)read_varint();
        }
        return {type, delta};
    }
    
    bool has_more() const { return p < end; }
    size_t remaining() const { return (size_t)(end - p); }
};

struct ParquetMetadata {
    int64_t num_rows = 0;
    std::vector<std::string> schema_cols;
    std::vector<int> col_indices;  // index into row group column chunks
    
    struct RowGroup {
        int64_t num_rows;
        struct ColChunk {
            std::string col_name;
            int64_t file_offset;
            int64_t total_size;
            int64_t num_values;
            int64_t data_page_offset;
            int64_t dict_page_offset;
            int encoding;
        };
        std::vector<ColChunk> columns;
    };
    std::vector<RowGroup> row_groups;
};

static bool parse_parquet_metadata(const uint8_t* data, size_t size, 
                                    const std::string& text_col,
                                    ParquetMetadata& meta) {
    if (size < 12) return false;
    if (memcmp(data + size - 4, "PAR1", 4) != 0) return false;
    
    uint32_t meta_len;
    memcpy(&meta_len, data + size - 8, 4);
    if (meta_len == 0 || meta_len + 8 > size) return false;
    
    const uint8_t* meta_start = data + size - 8 - meta_len;
    CompactReader cr(meta_start, meta_len);
    
    while (cr.has_more()) {
        auto f = cr.read_field();
        if (f.type == 0) break;
        
        if (f.id == 1 && f.type == 5) {
            cr.read_i32();
        } else if (f.id == 2 && f.type == 9) {
            auto lh = cr.read_list_header();
            for (int64_t i = 0; i < lh.count; i++) {
                std::string col_name;
                while (cr.has_more()) {
                    auto sf = cr.read_field();
                    if (sf.type == 0) break;
                    if (sf.id == 2 && sf.type == 8) {
                        col_name = cr.read_string();
                    } else {
                        cr.skip_field(sf.type);
                    }
                }
                meta.schema_cols.push_back(col_name);
            }
        } else if (f.id == 3 && f.type == 6) {
            meta.num_rows = cr.read_i64();
            } else if (f.id == 4 && f.type == 9) {
                auto lh = cr.read_list_header();
                for (int64_t rg = 0; rg < lh.count; rg++) {
                ParquetMetadata::RowGroup rgi;
                while (cr.has_more()) {
                    auto rf = cr.read_field();
                    if (rf.type == 0) break;
                    if (rf.id == 1 && rf.type == 9) {
                        auto clh = cr.read_list_header();
                        for (int64_t c = 0; c < clh.count; c++) {
                            ParquetMetadata::RowGroup::ColChunk ch;
                            while (cr.has_more()) {
                                auto cf = cr.read_field();
                                if (cf.type == 0) break;
                                if (cf.id == 1 && cf.type == 8) cr.read_string();
                                else if (cf.id == 2 && cf.type == 6) ch.file_offset = cr.read_i64();
                                else if (cf.id == 3 && cf.type == 12) {
                                    while (cr.has_more()) {
                                        auto mf = cr.read_field();
                                        if (mf.type == 0) break;
                                        if (mf.id == 1 && mf.type == 5) cr.read_i32();
                                        else if (mf.id == 2 && mf.type == 5) ch.encoding = cr.read_i32();
                                        else if (mf.id == 3 && mf.type == 8) cr.read_string();
                                        else if (mf.id == 4 && mf.type == 6) ch.total_size = cr.read_i64();
                                        else if (mf.id == 5 && mf.type == 6) ch.num_values = cr.read_i64();
                                        else if (mf.id == 6 && mf.type == 6) ch.data_page_offset = cr.read_i64();
                                        else if (mf.id == 7 && mf.type == 6) ch.dict_page_offset = cr.read_i64();
                                        else cr.skip_field(mf.type);
                                    }
                                } else if (cf.id == 4 && cf.type == 5) cr.read_i32();
                                else if (cf.id == 5 && cf.type == 12) cr.skip_field(12);
                                else cr.skip_field(cf.type);
                            }
                            rgi.columns.push_back(ch);
                        }
                    } else if (rf.id == 2 && rf.type == 6) {
                        rgi.num_rows = cr.read_i64();
                    } else if (rf.id == 3 && rf.type == 6) {
                        cr.read_i64();
                    } else cr.skip_field(rf.type);
                }
                meta.row_groups.push_back(rgi);
            }
        } else cr.skip_field(f.type);
    }
    
    int text_col_idx = -1;
    for (size_t i = 1; i < meta.schema_cols.size(); i++) {
        if (meta.schema_cols[i] == text_col) {
            text_col_idx = (int)i - 1;
            break;
        }
    }
    if (text_col_idx < 0) return false;
    
    for (auto& rg : meta.row_groups) {
        if (text_col_idx < (int)rg.columns.size()) {
            meta.col_indices.push_back(text_col_idx);
        }
    }
    
    return !meta.row_groups.empty();
}

} // namespace quant
