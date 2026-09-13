// sandbox.cpp — quant::agi::Sandbox implementation.
//
// HONESTY CONTRACT (R5: no stubs-as-features):
//   * static_analysis()        — REAL. Lexical scan for known-dangerous constructs.
//   * check_resource_limits()  — REAL BUT STATIC. An estimate derived from the source
//                                text, NOT runtime enforcement. It can only reject
//                                code it can see is obviously over budget.
//   * generate_test_program()  — REAL. Pure source-text generation.
//   * count_tests_passed()     — REAL. Parses a report stream.
//   * verify_correctness() / compile_and_test() / benchmark() / run_with_timeout()
//                              — NOT IMPLEMENTED. They fail loud (false / exit_code -1)
//                                with a [REAL-ONLY] marker. Nothing here pretends to
//                                have compiled or executed anything.
//
// Before 2026-09-10 this class was declared in include/quant/agi.h with no definition
// anywhere in the tree or in any commit, which made three tests fail to link.

#include "quant/agi.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace quant {
namespace agi {
namespace {

bool contains_ci(const std::string& hay, const std::string& needle) {
    if (needle.empty()) return false;
    auto it = std::search(hay.begin(), hay.end(), needle.begin(), needle.end(),
                          [](unsigned char a, unsigned char b) {
                              return std::tolower(a) == std::tolower(b);
                          });
    return it != hay.end();
}

// Counts how many times `needle` appears, case-insensitively.
int count_ci(const std::string& hay, const std::string& needle) {
    int n = 0;
    size_t pos = 0;
    while (true) {
        auto it = std::search(hay.begin() + pos, hay.end(), needle.begin(), needle.end(),
                              [](unsigned char a, unsigned char b) {
                                  return std::tolower(a) == std::tolower(b);
                              });
        if (it == hay.end()) break;
        ++n;
        pos = static_cast<size_t>(it - hay.begin()) + needle.size();
        if (pos >= hay.size()) break;
    }
    return n;
}

} // namespace

Sandbox::Sandbox() {
    // Best-effort scratch directory. Failure is not fatal: the analysis paths are
    // pure and never touch the filesystem.
    try {
        auto base = std::filesystem::temp_directory_path() / "quant_sandbox";
        std::filesystem::create_directories(base);
        sandbox_dir_ = base.string();
    } catch (...) {
        sandbox_dir_ = "quant_sandbox";
    }
}

std::string Sandbox::get_sandbox_path() const {
    return sandbox_dir_;
}

std::vector<std::string> Sandbox::static_analysis(const std::string& code) {
    std::vector<std::string> findings;

    struct Rule { const char* token; const char* finding; };
    static const Rule rules[] = {
        { "system(",       "process execution: system()" },
        { "popen(",        "process execution: popen()" },
        { "execv",         "process execution: execve/execvp family" },
        { "execl",         "process execution: execl family" },
        { "spawn",         "process spawning" },
        { "fork(",         "process forking" },
        { ":(){",          "fork-bomb pattern" },
        { ":|:&",          "fork-bomb pattern" },
        { "curl",          "network egress: curl" },
        { "wget",          "network egress: wget" },
        { "socket(",       "network: raw socket" },
        { "connect(",      "network: outbound connect" },
        { "bind(",         "network: bind" },
        { "rm -rf",        "destructive: rm -rf" },
        { "mkfs",          "destructive: mkfs" },
        { "dd if=",        "destructive: dd" },
        { "shutdown",      "destructive: shutdown" },
        { "reboot",        "destructive: reboot" },
        { "/etc/passwd",   "sensitive path: /etc/passwd" },
        { "/etc/shadow",   "sensitive path: /etc/shadow" },
        { "setuid",        "privilege escalation: setuid" },
        { "sudo",          "privilege escalation: sudo" },
        { "dlopen",        "dynamic code loading: dlopen" },
        { "__asm",         "inline assembly" },
        { "asm volatile",  "inline assembly" },
    };

    for (const auto& r : rules) {
        if (contains_ci(code, r.token)) findings.emplace_back(r.finding);
    }

    // Piped-shell pattern: "curl ... | sh" is the classic remote-execution idiom.
    if ((contains_ci(code, "curl") || contains_ci(code, "wget")) &&
        (contains_ci(code, "| sh") || contains_ci(code, "|sh") ||
         contains_ci(code, "| bash") || contains_ci(code, "|bash"))) {
        findings.emplace_back("remote code execution: download piped into a shell");
    }

    return findings;
}

bool Sandbox::check_resource_limits(const std::string& code, int64_t max_memory_mb,
                                    double max_time_sec) {
    // STATIC ESTIMATE ONLY. There is no interpreter here, so this is a conservative
    // guess based on loop nesting and visible allocation sites. It will happily pass
    // code that allocates at runtime; treat a `true` as "nothing obviously over
    // budget", never as a guarantee.
    const double kBytesPerAlloc = 64.0 * 1024.0;   // 64 KiB assumed per allocation site
    const double kOpsPerLoop    = 1.0e6;           // 1M ops assumed per loop
    const double kOpsPerSec     = 1.0e8;           // 100M ops/s assumed throughput

    int alloc_sites = count_ci(code, "new ") + count_ci(code, "malloc(") +
                      count_ci(code, "calloc(") + count_ci(code, "std::vector");

    // Nesting depth: count loop keywords and use the total as a proxy for nesting.
    int loops = count_ci(code, "for") + count_ci(code, "while") + count_ci(code, "do {");
    double est_ops = 1.0;
    for (int i = 0; i < loops; ++i) est_ops *= kOpsPerLoop;
    if (est_ops > 1.0e18) est_ops = 1.0e18;   // clamp; saturates rather than overflows

    double est_bytes = 1024.0 * 1024.0 + static_cast<double>(alloc_sites) * kBytesPerAlloc;
    double est_sec   = est_ops / kOpsPerSec;

    const double max_bytes = static_cast<double>(max_memory_mb) * 1024.0 * 1024.0;
    return est_bytes <= max_bytes && est_sec <= max_time_sec;
}

bool Sandbox::verify_correctness(const std::string& code,
                                 const std::vector<std::string>& test_cases) {
    (void)code;
    (void)test_cases;
    std::fprintf(stderr,
        "[REAL-ONLY] agi::Sandbox::verify_correctness is NOT implemented "
        "(requires compiling and running the candidate) -> reporting false, not pass.\n");
    return false;
}

SandboxResult Sandbox::compile_and_test(const std::string& code, const std::string& task) {
    (void)code;
    (void)task;
    std::fprintf(stderr,
        "[REAL-ONLY] agi::Sandbox::compile_and_test is NOT implemented "
        "(no compile+execute path) -> compiled=false, passed=false, exit_code=-1.\n");
    SandboxResult r;
    r.compiled = false;
    r.passed = false;
    r.score = 0.0f;
    r.stdout_capture.clear();
    r.stderr_capture = "Sandbox::compile_and_test not implemented (no compile+execute path)";
    r.runtime_ms = 0.0;
    r.exit_code = -1;
    return r;
}

bool Sandbox::benchmark(const std::string& code, double& ops_per_sec, double& avg_latency_ms) {
    (void)code;
    ops_per_sec = 0.0;
    avg_latency_ms = 0.0;
    std::fprintf(stderr,
        "[REAL-ONLY] agi::Sandbox::benchmark is NOT implemented "
        "(requires executing the candidate) -> reporting false, not a timing.\n");
    return false;
}

std::string Sandbox::generate_test_program(const std::string& code, const std::string& task) {
    // Pure text generation: wraps the candidate in a main() that prints a parseable
    // PASS/FAIL line per case. Generation is real; running it is not (see above).
    std::string out;
    out += "// Generated by quant::agi::Sandbox::generate_test_program\n";
    out += "// task: ";
    out += task;
    out += "\n#include <cstdio>\n#include <string>\n#include <vector>\n\n";
    out += code;
    out += "\n\nint main() {\n";
    out += "    // No harness is executed by this build; this program is emitted for\n";
    out += "    // an external toolchain to compile.\n";
    out += "    std::printf(\"FAIL sandbox harness not executed\\n\");\n";
    out += "    return 1;\n}\n";
    return out;
}

bool Sandbox::run_with_timeout(const std::string& binary, double timeout_sec,
                               std::string& stdout_out, std::string& stderr_out,
                               int& exit_code) {
    (void)binary;
    (void)timeout_sec;
    stdout_out.clear();
    stderr_out = "Sandbox::run_with_timeout not implemented (no process execution)";
    exit_code = -1;
    std::fprintf(stderr,
        "[REAL-ONLY] agi::Sandbox::run_with_timeout is NOT implemented "
        "(no process execution) -> returning false, exit_code=-1.\n");
    return false;
}

int Sandbox::count_tests_passed(const std::string& stdout_str) {
    // Counts conventional success markers: lines starting with PASS/ok, or [ok].
    int passed = 0;
    size_t line_start = 0;
    while (line_start <= stdout_str.size()) {
        size_t nl = stdout_str.find('\n', line_start);
        std::string line = stdout_str.substr(
            line_start, nl == std::string::npos ? std::string::npos : nl - line_start);

        size_t first = line.find_first_not_of(" \t\r");
        if (first != std::string::npos) {
            std::string trimmed = line.substr(first);
            std::string lower;
            lower.reserve(trimmed.size());
            for (char c : trimmed) lower.push_back(static_cast<char>(std::tolower(
                                       static_cast<unsigned char>(c))));
            if (lower.rfind("pass", 0) == 0 || lower.rfind("ok", 0) == 0 ||
                lower.rfind("[ok]", 0) == 0) {
                ++passed;
            }
        }

        if (nl == std::string::npos) break;
        line_start = nl + 1;
    }
    return passed;
}

} // namespace agi
} // namespace quant
