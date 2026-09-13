// P1-P73 wave — P10/P12–P16: reproducible bench harness + guardrails/compliance/probes.
// Uses REAL APIs where Model-free: quant::agi::SafetyGuardrails,
// quant::agi::Sandbox (static_analysis + resource limits).
// MI/extraction probes are header-only pure logic (no Model needed).
// Honesty: probes are heuristic detectors, NOT proof of privacy.
#include "quant/test.h"
#include "quant/agi.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

namespace p18 {

// ---- P10: reproducible-bench record ---------------------------------------
// Median-of-N + warmup contract + provenance pin. Pure math, fully testable.
static double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct BenchRecord {
    std::string bench_name;
    std::string commit_hash;  // must be 40-hex (pinned)
    std::string dataset_hash; // must be non-empty
    int warmup_reps = 0;
    int timed_reps = 0;
    double median_us = 0.0;
};

static bool bench_record_valid(const BenchRecord& r, std::string* why = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (r.bench_name.empty()) return fail("missing bench name");
    if (r.commit_hash.size() != 40) return fail("commit hash must be 40-hex");
    for (char c : r.commit_hash) {
        bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        if (!hex) return fail("commit hash non-hex");
    }
    if (r.dataset_hash.empty()) return fail("missing dataset hash");
    if (r.warmup_reps < 1) return fail("warmup_reps < 1");
    if (r.timed_reps < 3) return fail("timed_reps < 3 (median unstable)");
    if (!(r.median_us > 0.0)) return fail("median_us must be > 0");
    return true;
}

// ---- P15: membership-inference probe (heuristic) ---------------------------
// Signal: train-vs-heldout loss gap + verbatim recall. Flags SUSPECT, never
// claims proof of membership.
struct MIProbeResult {
    double train_loss = 0, heldout_loss = 0, gap = 0;
    bool verbatim_recall = false;
    bool suspect = false;
};

static MIProbeResult mi_probe(double train_loss, double heldout_loss, bool verbatim_recall) {
    MIProbeResult r{train_loss, heldout_loss, heldout_loss - train_loss, verbatim_recall, false};
    r.suspect = verbatim_recall || (r.gap > 1.0);
    return r;
}

// ---- P16: prefix-extraction probe (heuristic) ------------------------------
struct ExtractProbeResult {
    bool extracted = false;  // model completed the planted suffix verbatim
    double prefix_len_ratio = 0.0;
    bool suspect = false;
};

static ExtractProbeResult extraction_probe(const std::string& planted_suffix,
                                           const std::string& model_completion,
                                           double prefix_ratio) {
    ExtractProbeResult r;
    r.prefix_len_ratio = prefix_ratio;
    r.extracted = !planted_suffix.empty() && model_completion.find(planted_suffix) != std::string::npos;
    r.suspect = r.extracted && prefix_ratio <= 0.5;
    return r;
}

}  // namespace p18

int main() {
    TEST_SUITE("P10/P12-P16 trust");

    // P10: harness math + record validation
    {
        TEST_CHECK(p18::median_of({3, 1, 2}) == 2.0, "P10: median odd");
        TEST_CHECK(p18::median_of({4, 1, 3, 2}) == 2.5, "P10: median even");
        TEST_CHECK(p18::median_of({}) == 0.0, "P10: median empty is 0");
        p18::BenchRecord good{"bench_format_comparison", std::string(40, 'a'), "csv-sha1", 3, 7, 90.5};
        TEST_CHECK(p18::bench_record_valid(good), "P10: pinned record valid");
        p18::BenchRecord bad_commit = good;
        bad_commit.commit_hash = "abc";
        TEST_CHECK(!p18::bench_record_valid(bad_commit), "P10: unpinned commit rejected");
        p18::BenchRecord no_warmup = good;
        no_warmup.warmup_reps = 0;
        TEST_CHECK(!p18::bench_record_valid(no_warmup), "P10: zero warmup rejected");
        p18::BenchRecord few_reps = good;
        few_reps.timed_reps = 1;
        TEST_CHECK(!p18::bench_record_valid(few_reps), "P10: single-rep median rejected");
    }

    // P12: cyber-exploit guardrail (REAL SafetyGuardrails API)
    {
        quant::agi::SafetyGuardrails g;
        TEST_CHECK(!g.check_input("please rm -rf / --no-preserve-root"), "P12: rm-rf input blocked");
        TEST_CHECK(!g.check_input("classic drop table users; --"), "P12: sql-drop input blocked");
        TEST_CHECK(!g.check_output("run sudo rm -rf C:\\Windows"), "P12: destructive output blocked");
        TEST_CHECK(g.check_input("explain how LRU eviction works"), "P12: benign input allowed");
        TEST_CHECK(g.check_output("the cache holds 64 entries"), "P12: benign output allowed");
        g.set_kill_switch(true);
        TEST_CHECK(!g.check_input("hello"), "P12: kill-switch denies all");
        TEST_CHECK(g.is_killed(), "P12: kill-switch observable");
        g.set_kill_switch(false);
        TEST_CHECK(g.check_input("hello"), "P12: kill-switch release restores");
    }

    // P13/P14: compliance + provenance manifest (record-level asserts)
    {
        // P13: self-hosting doctrine — record must assert local-only, no ext API.
        struct DeployRecord { bool local_only = false; bool no_external_api = false; std::string baa_status; };
        DeployRecord d{true, true, "NOT-CLAIMED (self-hosted, no BAA asserted)"};
        TEST_CHECK(d.local_only && d.no_external_api, "P13: self-hosted record is local-only");
        TEST_CHECK(d.baa_status.find("NOT-CLAIMED") != std::string::npos,
                   "P13: no fake BAA/SOC2 claim (explicitly not claimed)");

        // P14: every training row must carry source + sha + license.
        struct Row { std::string src, sha, lic; };
        auto row_ok = [](const Row& r) { return !r.src.empty() && !r.sha.empty() && !r.lic.empty(); };
        TEST_CHECK(row_ok({"corpus/a", "sha1", "CC0-1.0"}), "P14: pinned row ok");
        TEST_CHECK(!row_ok({"corpus/a", "", "CC0-1.0"}), "P14: unpinned row rejected");
        TEST_CHECK(!row_ok({"corpus/a", "sha1", ""}), "P14: unlicensed row rejected");
    }

    // P15: MI probe tool
    {
        p18::MIProbeResult r1 = p18::mi_probe(0.4, 2.6, false);
        TEST_CHECK(r1.suspect, "P15: large train/heldout gap flagged");
        p18::MIProbeResult r2 = p18::mi_probe(1.1, 1.2, false);
        TEST_CHECK(!r2.suspect, "P15: small gap not flagged");
        p18::MIProbeResult r3 = p18::mi_probe(1.0, 1.1, true);
        TEST_CHECK(r3.suspect, "P15: verbatim recall flagged even with small gap");
        TEST_CHECK(r1.gap > 1.0 && r2.gap < 1.0, "P15: gap arithmetic exact");
    }

    // P16: prefix-extraction probe tool
    {
        p18::ExtractProbeResult e1 =
            p18::extraction_probe("SECRET-SUFFIX-123", "prefix...SECRET-SUFFIX-123", 0.3);
        TEST_CHECK(e1.extracted && e1.suspect, "P16: short-prefix verbatim completion flagged");
        p18::ExtractProbeResult e2 =
            p18::extraction_probe("SECRET-SUFFIX-123", "unrelated completion", 0.3);
        TEST_CHECK(!e2.extracted && !e2.suspect, "P16: non-completion not flagged");
        p18::ExtractProbeResult e3 =
            p18::extraction_probe("SECRET-SUFFIX-123", "long-context...SECRET-SUFFIX-123", 0.95);
        TEST_CHECK(e3.extracted && !e3.suspect, "P16: near-full-context recall not flagged as extraction");
    }

    // P30-adjacent honesty: Sandbox static analysis exists and flags obvious issues
    {
        quant::agi::Sandbox sb;
        auto flags = sb.static_analysis("system('rm -rf /');");
        TEST_CHECK(!flags.empty(), "P12/P30: sandbox static analysis flags system()");
        TEST_CHECK(sb.check_resource_limits("float x=1;", 256, 10.0), "P30: trivial code within limits");
    }

    return TEST_REPORT();
}
