// ============================================================================
// test_gle_telemetry.cpp — GLE acceptance criteria (TRANSCRIPT.md PART-J.8)
//   1) 1000 events append + read-back integrity == 100%
//   2) hash-chain verifier catches injected tamper (negative test)
//   3) partial last line => parser skips + flags truncated_tail
// ============================================================================
#include "gle_telemetry.h"

#include <cstdio>
#include <fstream>
#include <string>

using namespace gle;

static int failures = 0;
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::printf("  FAIL: %s (line %d)\n", msg, __LINE__);               \
            ++failures;                                                         \
        }                                                                       \
    } while (0)

static const char* kPath = "test_gle_events.jsonl";

int main() {
    std::printf("== GLE telemetry tests ==\n");

    // ---- 1) append 1000 events, read back ---------------------------------
    {
        TelemetryWriter w(kPath);
        for (int i = 0; i < 1000; ++i) {
            Event e;
            e.run_id = "master_plan_v2_20260822";
            e.phase = "0";
            e.task = "L008";
            e.type = (i % 3 == 0) ? EventType::BuildResult : EventType::TestResult;
            e.actor = "ci";
            e.verdict = (i % 7 == 0) ? "FAIL" : "PASS";
            e.evidence = "repo/state/telemetry/runs/r1/artifacts/out_" + std::to_string(i) + ".txt";
            e.numbers = "{\"iter\":" + std::to_string(i) + "}";
            w.append(e);
        }
    }
    {
        TelemetryReader r(kPath);
        CHECK(r.report().events_parsed == 1000, "1000 events parsed");
        CHECK(r.events().size() == 1000, "1000 events stored");
        bool all_fields = true;
        for (size_t i = 0; i < r.events().size(); ++i) {
            const auto& e = r.events()[i];
            if (e.run_id != "master_plan_v2_20260822" || e.task != "L008" ||
                e.numbers.find("\"iter\":" + std::to_string(i)) == std::string::npos) {
                all_fields = false;
                break;
            }
        }
        CHECK(all_fields, "field integrity across all events");
        auto rep = r.verify_chain();
        CHECK(rep.chain_valid, "hash chain valid after clean append");
        CHECK(rep.first_bad_line == 0, "no bad line on clean stream");
    }

    // ---- chain continues correctly when writer re-opens -------------------
    {
        TelemetryWriter w2(kPath);
        Event e;
        e.run_id = "master_plan_v2_20260822";
        e.phase = "0";
        e.task = "L008";
        e.type = EventType::PhaseExit;
        e.actor = "lead";
        e.verdict = "PASS";
        e.evidence = "research/workbench.md";
        w2.append(e);

        TelemetryReader r(kPath);
        CHECK(r.events().size() == 1001, "re-opened writer appends (chain continues)");
        CHECK(r.verify_chain().chain_valid, "chain intact across writer sessions");
    }

    // ---- 2) tamper detection ----------------------------------------------
    {
        // Flip one character inside an early line's verdict.
        std::ifstream in(kPath);
        std::vector<std::string> lines;
        std::string l;
        while (std::getline(in, l)) lines.push_back(l);
        in.close();
        CHECK(lines.size() >= 500, "enough lines before tamper");
        size_t victim = 400;
        size_t pos = lines[victim].find("\"verdict\":\"PASS\"");
        if (pos != std::string::npos)
            lines[victim].replace(pos, 16, "\"verdict\":\"FAIL\"");
        else {
            pos = lines[victim].find("\"verdict\":\"FAIL\"");
            lines[victim].replace(pos, 16, "\"verdict\":\"PASS\"");
        }
        std::ofstream out(kPath);
        for (auto& x : lines) out << x << '\n';
        out.close();

        TelemetryReader r(kPath);
        auto rep = r.verify_chain();
        CHECK(!rep.chain_valid, "tamper detected by chain verifier");
        CHECK(rep.first_bad_line == victim + 1, "tamper localized to correct line");
    }

    // ---- 3) partial tail tolerance ----------------------------------------
    {
        std::ofstream out(kPath, std::ios::app);
        out << "{\"ts\":\"2026-08-22T10:00:00.000+05:30\",\"run_id\":\"x";   // no closing brace
        out.close();

        TelemetryReader r(kPath);
        CHECK(r.report().truncated_tail, "partial last line flagged");
        CHECK(r.verify_chain().events_parsed == r.events().size(), "parser skipped partial tail");
    }

    std::remove(kPath);
    if (failures == 0) {
        std::printf("GLE telemetry: ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("GLE telemetry: %d FAILURES\n", failures);
    return 1;
}
