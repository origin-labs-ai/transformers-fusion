// ============================================================================
// gle_report.cpp — GLE report generator (TRANSCRIPT.md PART-J.5)
// Reads repo/state/telemetry/events.jsonl and prints a markdown summary.
// Zero-dependency CLI: gle_report [events.jsonl]
// ============================================================================
#include "gle_telemetry.h"

#include <cstdio>
#include <map>
#include <string>

int main(int argc, char** argv) {
    // gle_report --init [run_id] : create/append a properly-chained genesis
    // RUN_STARTED event (PART-S agent handoff, step 5).
    std::string flag = argc > 1 ? argv[1] : "";
    if (flag == "--init") {
        const char* init_run_id = argc > 2 ? argv[2] : "master_plan_v2_20260822";
        const char* init_path = "repo/state/telemetry/events.jsonl";
        gle::TelemetryWriter w(init_path);
        gle::Event e;
        e.run_id = init_run_id;
        e.phase = "0";
        e.task = "-";
        e.type = gle::EventType::RunStarted;
        e.actor = "lead";
        e.verdict = "-";
        e.evidence = "TRANSCRIPT.md";
        w.append(e);
        std::printf("GLE stream initialized at %s (run_id=%s)\n", init_path, init_run_id);
        return 0;
    }

    const char* path = argc > 1 ? argv[1] : "repo/state/telemetry/events.jsonl";

    gle::TelemetryReader r(path);
    const auto& rep = r.verify_chain();

    std::printf("=== GLE REPORT ===\n");
    std::printf("Stream: %s\n", path);
    std::printf("Lines read      : %zu\n", rep.lines_read);
    std::printf("Events parsed   : %zu\n", rep.events_parsed);
    std::printf("Skipped lines   : %zu%s\n", rep.skipped_lines,
                rep.truncated_tail ? " (truncated tail flagged)" : "");
    std::printf("Evidence chain  : %s", rep.chain_valid ? "VALID" : "BROKEN");
    if (!rep.chain_valid) std::printf(" (first bad line: %zu)", rep.first_bad_line);
    std::printf("\n");

    if (rep.events_parsed == 0) return 0;

    // Per-event-type counts + verdict split.
    std::map<std::string, int> by_type;
    int pass = 0, fail = 0, other = 0;
    for (const auto& e : r.events()) {
        by_type[gle::event_type_name(e.type)]++;
        if (e.verdict == "PASS") ++pass;
        else if (e.verdict == "FAIL") ++fail;
        else if (!e.verdict.empty() && e.verdict != "-") ++other;
    }

    std::printf("\n| Event | Count |\n|---|---|\n");
    for (const auto& kv : by_type)
        std::printf("| %s | %d |\n", kv.first.c_str(), kv.second);

    std::printf("\nVerdicts: PASS=%d FAIL=%d OTHER=%d\n", pass, fail, other);

    // Orphan check: verdict events must carry evidence.
    size_t orphans = 0;
    for (const auto& e : r.events()) {
        bool is_verdict = (e.type == gle::EventType::CriticVerdict ||
                           e.type == gle::EventType::ClaimVerified);
        if (is_verdict && e.evidence.empty()) ++orphans;
    }
    std::printf("Evidence orphans : %zu %s\n", orphans,
                orphans ? "(RED — verdicts without evidence!)" : "(none)");

    // Latest phase/task snapshot.
    if (!r.events().empty()) {
        const auto& last = r.events().back();
        std::printf("\nLatest event : %s (phase=%s task=%s actor=%s)\n",
                    gle::event_type_name(last.type), last.phase.c_str(),
                    last.task.c_str(), last.actor.c_str());
    }
    return rep.chain_valid && orphans == 0 ? 0 : 1;
}
