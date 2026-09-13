// L076: OpenAI endpoint contract tests (Phase 16 Wave 7).
//
// Pure unit tests over HTTPServer's contract helpers (no sockets): they lock
// the exact wire shapes the handlers emit, so any future handler edit that
// breaks openai-python compatibility fails here first.
// Live SSE/HTTP framing is covered by construction: handlers send frames only
// via sse_data_frame()/sse_done_frame() (OpenAI "data:" framing, no "event:"
// line) and errors only via openai_error_json().
#include "quant/http_server.h"
#include "quant/http_parse.h"
#include "quant/json_parser.h"
#include "quant/test.h"

#include <cstdio>
#include <string>
#include <thread>
#include <chrono>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

using namespace quant;

static void test_openai_error_shape() {
    TEST_SUITE("L076: openai_error_json shape");
    std::string j = HTTPServer::openai_error_json("bad request", "invalid_request_error",
                                                  "invalid_json");
    std::string err;
    JsonValue v = JsonValue::parse(j, &err);
    TEST_CHECK(err.empty(), "error json parses");
    TEST_CHECK(v.is_object() && v.has("error"), "top-level error object");
    const JsonValue& e = v["error"];
    TEST_CHECK(e.is_object(), "error is object");
    TEST_CHECK(e.has("message") && e["message"].as_string() == "bad request",
               "error.message preserved");
    TEST_CHECK(e.has("type") && e["type"].as_string() == "invalid_request_error",
               "error.type preserved");
    TEST_CHECK(e.has("code") && e["code"].as_string() == "invalid_json",
               "error.code preserved");
}

static void test_openai_error_defaults() {
    TEST_SUITE("L076: openai_error_json defaults");
    std::string j = HTTPServer::openai_error_json("x", "", "");
    std::string err;
    JsonValue v = JsonValue::parse(j, &err);
    TEST_CHECK(err.empty(), "default error json parses");
    TEST_CHECK(v["error"]["type"].as_string() == "invalid_request_error",
               "empty type defaults to invalid_request_error");
}

static void test_sse_framing() {
    TEST_SUITE("L076: SSE OpenAI framing");
    std::string f = HTTPServer::sse_data_frame("{\"a\":1}");
    TEST_CHECK(f == "data: {\"a\":1}\n\n", "data frame exact bytes");
    TEST_CHECK(f.find("event:") == std::string::npos, "no event: prefix in data frame");
    std::string d = HTTPServer::sse_done_frame();
    TEST_CHECK(d == "data: [DONE]\n\n", "done frame exact bytes");
}

static void test_hardening_defaults() {
    TEST_SUITE("L074: hardening defaults (permissive)");
    HTTPServer s(0);
    TEST_CHECK(s.auth_token().empty(), "auth empty by default");
    TEST_CHECK(s.max_header_bytes() == 64 * 1024, "header cap 64KB by default");
    TEST_CHECK(s.max_concurrent() == 0, "concurrency unlimited by default");
    TEST_CHECK(!s.is_running(), "not running by default");
    TEST_CHECK(s.current_requests() == 0, "no in-flight requests");
}

static void test_hardening_setters() {
    TEST_SUITE("L074: hardening setters roundtrip");
    HTTPServer s(0);
    s.set_auth_token("sekret");
    s.set_max_header_bytes(8192);
    s.set_max_concurrent(8);
    TEST_CHECK(s.auth_token() == "sekret", "auth token roundtrip");
    TEST_CHECK(s.max_header_bytes() == 8192, "header cap roundtrip");
    TEST_CHECK(s.max_concurrent() == 8, "concurrency roundtrip");
    s.set_max_header_bytes(-1);
    TEST_CHECK(s.max_header_bytes() == 64 * 1024, "negative header cap clamps to default");
    s.set_max_concurrent(-3);
    TEST_CHECK(s.max_concurrent() == 0, "negative concurrency clamps to unlimited");
}

static void test_legacy_status_text() {
    TEST_SUITE("C-24: legacy tool reason phrases (413/414 were Unknown)");
    TEST_CHECK(http_parse::status_text(200) == "OK", "200 OK");
    TEST_CHECK(http_parse::status_text(204) == "No Content", "204 No Content (was Unknown)");
    TEST_CHECK(http_parse::status_text(400) == "Bad Request", "400 Bad Request");
    TEST_CHECK(http_parse::status_text(404) == "Not Found", "404 Not Found");
    TEST_CHECK(http_parse::status_text(413) == "Payload Too Large", "413 Payload Too Large (was Unknown)");
    TEST_CHECK(http_parse::status_text(414) == "URI Too Long", "414 URI Too Long (was Unknown)");
    TEST_CHECK(http_parse::status_text(500) == "Internal Server Error", "500 Internal Server Error");
    TEST_CHECK(http_parse::status_text(501) == "Not Implemented", "501 Not Implemented");
}

static void test_parse_content_length() {
    TEST_SUITE("C-24: Content-Length parsing (was single-recv, no CL)");
    TEST_CHECK(http_parse::parse_content_length("POST /x HTTP/1.1\r\nContent-Length: 42", 65536) == 42,
               "plain CL parsed");
    TEST_CHECK(http_parse::parse_content_length("POST /x HTTP/1.1\r\ncontent-length: 7", 65536) == 7,
               "lowercase CL parsed");
    TEST_CHECK(http_parse::parse_content_length("POST /x HTTP/1.1\r\nContent-Length:   13  ", 65536) == 13,
               "OWS tolerated");
    TEST_CHECK(http_parse::parse_content_length("GET /health HTTP/1.1", 65536) == 0,
               "absent CL means no body");
    TEST_CHECK(http_parse::parse_content_length("POST /x HTTP/1.1\r\nContent-Length: abc", 65536) == -1,
               "malformed CL flagged");
    TEST_CHECK(http_parse::parse_content_length("POST /x HTTP/1.1\r\nContent-Length: 999999", 65536) > 65536,
               "oversize CL signaled for 413");
}

#ifdef _WIN32
static void sock_init() { WSADATA w; WSAStartup(MAKEWORD(2, 2), &w); }
static void sock_cleanup() { WSACleanup(); }
static void sock_close(int fd) { closesocket(fd); }
#else
static void sock_init() {}
static void sock_cleanup() {}
static void sock_close(int fd) { ::close(fd); }
#endif

static std::string http_get(const std::string& host, int port, const std::string& req) {
    int fd = (int)::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return "";
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
#ifdef _WIN32
    InetPtonA(AF_INET, host.c_str(), &addr.sin_addr);
#else
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
#endif
    if (::connect(fd, (sockaddr*)&addr, sizeof(addr)) != 0) { sock_close(fd); return ""; }
    ::send(fd, req.c_str(), (int)req.size(), 0);
    std::string out;
    char buf[4096];
    for (int i = 0; i < 20; i++) {
        int n = ::recv(fd, buf, (int)sizeof(buf) - 1, 0);
        if (n <= 0) break;
        buf[n] = '\0';
        out += buf;
        if (out.find("\r\n\r\n") != std::string::npos) break;
    }
    sock_close(fd);
    return out;
}

static void test_live_server_smoke() {
    TEST_SUITE("C-24: live HTTPServer smoke (real socket, ephemeral port)");
    sock_init();
    const int kPort = 18091; // ephemeral, avoids clashing with dev servers
    HTTPServer srv(kPort);
    srv.set_model_name("smoke-test");
    srv.start();
    // BUGFIX (bug census): fixed 400ms sleep raced server startup
    // (flaky/slow). Poll /health with a deadline instead.
    std::string h;
    for (int i = 0; i < 40; i++) {
        h = http_get("127.0.0.1", kPort, "GET /health HTTP/1.1\r\nHost: x\r\n\r\n");
        if (h.find("200 OK") != std::string::npos) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    TEST_CHECK(srv.is_running(), "server thread running after start");

    // 1. GET /health → 200 OK + {"status":"ok"}
    TEST_CHECK(h.find("200 OK") != std::string::npos, "GET /health returns 200 OK");
    TEST_CHECK(h.find("\"status\"") != std::string::npos, "health body carries status");

    // 2. Oversized request line (>8KB) → 414 with correct reason phrase
    std::string big_path(9000, 'a');
    std::string r = http_get("127.0.0.1", kPort, "GET /" + big_path + " HTTP/1.1\r\nHost: x\r\n\r\n");
    TEST_CHECK(r.find("414") != std::string::npos, "oversized request line returns 414");
    TEST_CHECK(r.find("URI Too Long") != std::string::npos, "414 carries 'URI Too Long' (was Unknown)");

    // 3. Unknown path → 404 (server still alive after the 414)
    std::string n = http_get("127.0.0.1", kPort, "GET /nope HTTP/1.1\r\nHost: x\r\n\r\n");
    TEST_CHECK(n.find("404") != std::string::npos, "unknown path returns 404");

    srv.stop();
    TEST_CHECK(!srv.is_running(), "server stopped cleanly");
    sock_cleanup();
}

static void test_query_decode() {
    TEST_SUITE("bug census: query %XX/+ decoding + obs-fold continuation");
    HTTPServer s(0);
    auto q = s.parse_query_string_public("q=hello%20world&plus=a%2Bb+c&bad=%ZZ&empty=&flag");
    TEST_CHECK(q["q"] == "hello world", "%20 decodes to space");
    TEST_CHECK(q["plus"] == "a+b c", "%2B decodes, + decodes to space");
    TEST_CHECK(q["bad"] == "%ZZ", "malformed % passes through literally");
    TEST_CHECK(q["empty"] == "", "empty value decodes");
    TEST_CHECK(q["flag"] == "", "bare flag decodes to empty");

    HTTPRequest r = s.parse_http_request_public(
        "GET /v1/models?x=1 HTTP/1.1\r\n"
        "X-Long: first\r\n"
        "  second\r\n"
        "\tthird\r\n"
        "Host: y\r\n\r\n");
    TEST_CHECK(r.method == "GET", "method parsed");
    TEST_CHECK(r.path == "/v1/models", "path split from query");
    TEST_CHECK(r.query_params["x"] == "1", "query param parsed");
    auto it = r.headers.find("x-long");
    TEST_CHECK(it != r.headers.end() && it->second == "first second third",
               "obs-fold continuation folded into previous header");

    HTTPRequest bad = s.parse_http_request_public("GARBAGE-NO-SPACES\r\nHost: y\r\n\r\n");
    TEST_CHECK(bad.method.empty(), "malformed request line leaves method empty (no wrap)");

    HTTPRequest nocont = s.parse_http_request_public(
        "GET / HTTP/1.1\r\n"
        "  orphan continuation\r\n"
        "Host: y\r\n\r\n");
    TEST_CHECK(nocont.headers.find("host") != nocont.headers.end(),
               "orphan continuation without prior header is dropped safely");
}

static void test_chunked_rejected_live() {
    TEST_SUITE("bug census: chunked Transfer-Encoding fails closed (501)");
    HTTPServer s(0);
    const int kPort = 18092;
    HTTPServer srv(kPort);
    srv.start();
    std::string h;
    for (int i = 0; i < 40; i++) {
        h = http_get("127.0.0.1", kPort,
                     "GET /health HTTP/1.1\r\nHost: x\r\n\r\n");
        if (h.find("200 OK") != std::string::npos) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    TEST_CHECK(h.find("200 OK") != std::string::npos, "smoke server up");
    std::string r = http_get("127.0.0.1", kPort,
        "POST /v1/completions HTTP/1.1\r\n"
        "Host: x\r\n"
        "Transfer-Encoding: chunked\r\n"
        "\r\n"
        "5\r\nhello\r\n0\r\n\r\n");
    TEST_CHECK(r.find("501") != std::string::npos, "chunked body fails closed with 501");
    TEST_CHECK(r.find("chunked") != std::string::npos, "501 names chunked as the reason");
    srv.stop();
    (void)s;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Transcender - OpenAI Server Contract (L076) Test Suite\n");
    printf("======================================================\n");

    test_openai_error_shape();
    test_openai_error_defaults();
    test_sse_framing();
    test_hardening_defaults();
    test_hardening_setters();
    test_legacy_status_text();
    test_parse_content_length();
    test_query_decode();
    test_chunked_rejected_live();
    test_live_server_smoke();

    printf("\n======================================================\n");
    return TEST_REPORT() > 0 ? 1 : 0;
}
