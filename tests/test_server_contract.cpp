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

    printf("\n======================================================\n");
    return TEST_REPORT() > 0 ? 1 : 0;
}
