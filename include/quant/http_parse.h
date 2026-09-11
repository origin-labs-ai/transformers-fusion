#pragma once
// http_parse.h — header-only pure HTTP helpers shared by the serving tools.
//
// Production-hardening (C-24): centralizes the two things the legacy
// tools/quant_server.cpp got wrong — full reason-phrase table (413/414 were
// "Unknown") and Content-Length-aware body parsing (was single-recv).
// Pure functions => unit-tested in tests/test_server_contract.cpp.
#include <cstdint>
#include <string>
#include <algorithm>
#include <cctype>

namespace quant {
namespace http_parse {

// Full reason-phrase table for every status code the serving tools emit.
// Previously only 200/400/404/500/501 were mapped and 204/413/414 fell
// through to "Unknown" (ledger C-24 PARTIAL).
inline std::string status_text(int code) {
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
        case 501: return "Not Implemented";
        case 502: return "Bad Gateway";
        case 503: return "Service Unavailable";
        default:  return "Unknown";
    }
}

// Parse Content-Length from a raw header section (everything before
// the blank line). Case-insensitive header name, tolerant of OWS.
// Returns: declared body bytes (>=0); 0 when absent; -1 when malformed
// (non-digit value); clamps early past max_body so callers can fail fast
// with 413 without reading the body.
inline int64_t parse_content_length(const std::string& header_section,
                                    int64_t max_body) {
    std::string lower = header_section;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    auto pos = lower.find("content-length:");
    if (pos == std::string::npos) return 0;
    pos += 15; // strlen("content-length:")
    while (pos < lower.size() && (lower[pos] == ' ' || lower[pos] == '\t')) ++pos;
    int64_t v = 0;
    bool any = false;
    while (pos < lower.size() && lower[pos] >= '0' && lower[pos] <= '9') {
        any = true;
        v = v * 10 + (lower[pos] - '0');
        if (v > max_body) return v; // oversize signal; caller sends 413
        ++pos;
    }
    if (!any) return -1; // "Content-Length:" with no digits = malformed
    return v;
}

} // namespace http_parse
} // namespace quant
