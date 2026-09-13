// P1-P73 wave — P2/P3/P5: slop filter / DRM-clean data / Genius-vs-Noise reward.
// Self-contained: only quant/test.h + STL. No Model*, no build side-effects.
// Honesty: this proves the GATE LOGIC, not a trained filter. Integration into
// trainer_data ingest is an orchestrator snippet (see docs/P1P73_REPORT.md).
#include "quant/test.h"

#include <cctype>
#include <cmath>
#include <string>
#include <unordered_set>
#include <vector>

namespace p18 {

// ---- P2: slop filter ------------------------------------------------------
// Deterministic, explainable. Three signals: repetition, keyboard-smash,
// vacuous marketing loops. Score in [0,1]; >=0.5 KEEP, <0.5 REJECT.
static double repetition_ratio(const std::string& s) {
    if (s.size() < 16) return 0.0;
    // fraction of 4-grams that repeat
    std::unordered_set<std::string> seen, dup;
    size_t total = 0;
    for (size_t i = 0; i + 4 <= s.size(); ++i) {
        std::string g = s.substr(i, 4);
        ++total;
        if (!seen.insert(g).second) dup.insert(g);
    }
    return total ? (double)dup.size() / (double)total : 0.0;
}

static bool looks_keyboard_smash(const std::string& s) {
    // long run with <4 distinct alpha chars (asdf/qwer/aaaa) and no spaces
    size_t run = 0, spaces = 0;
    std::unordered_set<char> alpha;
    for (char c : s) {
        if (c == ' ') { ++spaces; continue; }
        if (std::isalpha((unsigned char)c)) alpha.insert((char)std::tolower((unsigned char)c));
        ++run;
    }
    return run >= 24 && spaces == 0 && alpha.size() <= 4;
}

static double slop_score(const std::string& s) {
    if (s.empty()) return 0.0;
    double score = 1.0;
    double rep = repetition_ratio(s);
    if (rep > 0.25) score -= 0.6;               // heavy repetition -> slop
    if (looks_keyboard_smash(s)) score -= 0.7;  // smash -> slop
    std::string low = s;
    for (char& c : low) c = (char)std::tolower((unsigned char)c);
    if (low.find("buy now") != std::string::npos && rep > 0.15) score -= 0.6;
    // 0.6 not 0.3: a 2-char string must land below the 0.5 KEEP threshold, and
    // it previously scored 0.7 and was kept. Too short to be useful => reject.
    if (s.size() < 12) score -= 0.6;
    if (score < 0.0) score = 0.0;
    if (score > 1.0) score = 1.0;
    return score;
}

static bool slop_keep(const std::string& s) { return slop_score(s) >= 0.5; }

// ---- P3: DRM-clean provenance gate ----------------------------------------
struct DataProvenance {
    std::string source;
    std::string license;  // e.g. "CC0-1.0", "MIT", "CC-BY-4.0"
    bool drm_flag = false;
    std::string sha256;   // hex, must be non-empty (pinned bytes)
};

// DRM-clean iff: no DRM flag, license present and allow-listed, sha pinned,
// and no "No AI Training" / "All Rights Reserved" without explicit grant.
static bool drm_clean(const DataProvenance& p, std::string* reason = nullptr) {
    auto fail = [&](const char* r) { if (reason) *reason = r; return false; };
    if (p.drm_flag) return fail("drm_flag set");
    if (p.sha256.empty()) return fail("missing sha256 pin");
    if (p.license.empty()) return fail("missing license");
    std::string low = p.license;
    for (char& c : low) c = (char)std::tolower((unsigned char)c);
    if (low.find("no ai training") != std::string::npos) return fail("license forbids AI training");
    if (low.find("all rights reserved") != std::string::npos) return fail("all-rights-reserved without grant");
    static const std::unordered_set<std::string> allow = {
        "cc0-1.0", "cc-by-4.0", "cc-by-sa-4.0", "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause",
        "public-domain", "synthetic-self-owned"};
    std::string norm;
    for (char c : low) { if (c != ' ' && c != '_') norm += c; }
    if (allow.find(norm) == allow.end()) return fail("license not in allow-list");
    return true;
}

// ---- P5: Genius-vs-Noise deterministic reward ------------------------------
// Three deterministic signals (no learned weights):
//   logic  = balanced delimiters + has predicate structure (verb + terminator)
//   sanity = length sane + no smash + repetition sane
//   novelty= rare-bigram proxy (distinct 3-gram ratio)
// Genius iff weighted sum >= 0.6 AND logic passes.
struct GeniusScore {
    double logic = 0, sanity = 0, novelty = 0, total = 0;
    bool logic_pass = false;
};

static GeniusScore genius_score(const std::string& s) {
    GeniusScore g;
    if (s.empty()) return g;
    int paren = 0, bracket = 0, brace = 0;
    bool balanced = true;
    for (char c : s) {
        if (c == '(') ++paren; else if (c == ')') { --paren; if (paren < 0) balanced = false; }
        if (c == '[') ++bracket; else if (c == ']') { --bracket; if (bracket < 0) balanced = false; }
        if (c == '{') ++brace; else if (c == '}') { --brace; if (brace < 0) balanced = false; }
    }
    if (paren != 0 || bracket != 0 || brace != 0) balanced = false;
    bool has_verb = false, has_end = false;
    std::string low = s;
    for (char& c : low) c = (char)std::tolower((unsigned char)c);
    for (const char* v : {" is ", " are ", " proves ", " implies ", " therefore ", " because "}) {
        if (low.find(v) != std::string::npos) { has_verb = true; break; }
    }
    if (!s.empty() && (s.back() == '.' || s.back() == ';' || s.back() == ')')) has_end = true;
    // Length is NOT a logic signal. The old `|| s.size() > 40` clause let a
    // 52-char keyboard-mash pass `logic_pass` (and therefore genius_keep), which
    // is exactly the case P5 exists to reject. Logic requires predicate structure.
    g.logic_pass = balanced && has_verb;
    g.logic = (balanced ? 0.5 : 0.0) + (has_verb ? 0.3 : 0.0) + (has_end ? 0.2 : 0.0);
    double rep = repetition_ratio(s);
    g.sanity = 1.0;
    if (looks_keyboard_smash(s)) g.sanity -= 0.8;
    if (rep > 0.25) g.sanity -= 0.5;
    if (s.size() < 12) g.sanity -= 0.4;
    if (g.sanity < 0) g.sanity = 0;
    // novelty: distinct 3-gram ratio
    std::unordered_set<std::string> grams;
    size_t total = 0;
    for (size_t i = 0; i + 3 <= s.size(); ++i) { grams.insert(s.substr(i, 3)); ++total; }
    g.novelty = total ? (double)grams.size() / (double)total : 0.0;
    g.total = 0.4 * g.logic + 0.3 * g.sanity + 0.3 * g.novelty;
    return g;
}

static bool genius_keep(const std::string& s) {
    GeniusScore g = genius_score(s);
    return g.logic_pass && g.total >= 0.6;
}

}  // namespace p18

int main() {
    TEST_SUITE("P2/P3/P5 data gates");

    // P2: slop filter
    {
        std::string human = "The KV cache eviction policy is LRU within a 64-entry block window.";
        TEST_CHECK(p18::slop_keep(human), "P2: genuine technical sentence kept");
        TEST_CHECK(p18::slop_score(human) >= 0.5, "P2: human score >= 0.5");

        std::string spam = "Buy now!!! Buy now!!! Buy now!!! Buy now!!! Buy now!!! Deal deal deal.";
        TEST_CHECK(!p18::slop_keep(spam), "P2: marketing loop rejected");
        std::string smash = "asdfasdfasdfasdfasdfasdfasdfasdfasdf";
        TEST_CHECK(!p18::slop_keep(smash), "P2: keyboard smash rejected");
        TEST_CHECK(!p18::slop_keep(""), "P2: empty rejected");
        TEST_CHECK(!p18::slop_keep("hi"), "P2: too-short rejected");
    }

    // P3: DRM-clean gate
    {
        p18::DataProvenance clean{"books/self-owned-corpus", "CC0-1.0", false, "abc123sha"};
        TEST_CHECK(p18::drm_clean(clean), "P3: CC0 pinned source accepted");
        p18::DataProvenance drm{"ebook-store/title", "All Rights Reserved", true, "deadbeef"};
        std::string why;
        TEST_CHECK(!p18::drm_clean(drm, &why), "P3: DRM-flagged source rejected");
        TEST_CHECK(!why.empty(), "P3: rejection reason populated");
        p18::DataProvenance noai{"web/scrape", "CC-BY-4.0 + No AI Training", false, "aa11"};
        TEST_CHECK(!p18::drm_clean(noai), "P3: No-AI-Training license rejected");
        p18::DataProvenance nopin{"web/clean", "MIT", false, ""};
        TEST_CHECK(!p18::drm_clean(nopin), "P3: unpinned bytes rejected");
        p18::DataProvenance nolic{"web/clean", "", false, "bb22"};
        TEST_CHECK(!p18::drm_clean(nolic), "P3: missing license rejected");
    }

    // P5: Genius vs Noise
    {
        std::string kafka = "Gregor Samsa woke to find himself transformed; therefore the trial is unjust because guilt implies a judge.";
        p18::GeniusScore gk = p18::genius_score(kafka);
        TEST_CHECK(gk.logic_pass, "P5: kafka-like passes logic gate");
        TEST_CHECK(p18::genius_keep(kafka), "P5: kafka-like kept as genius");
        std::string monkey = "xkqz wv jjj qqq asdf zxcv mnb vvv bbb qqq www eee rrr";
        p18::GeniusScore gm = p18::genius_score(monkey);
        TEST_CHECK(!p18::genius_keep(monkey), "P5: monkey keyboard rejected as noise");
        TEST_CHECK(gk.total > gm.total, "P5: genius total strictly above noise total");
        TEST_CHECK(!p18::genius_keep(""), "P5: empty never genius");
    }

    return TEST_REPORT();
}
