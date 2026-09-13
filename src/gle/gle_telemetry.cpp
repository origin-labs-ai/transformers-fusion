// ============================================================================
// gle_telemetry.cpp — GLE telemetry writer/reader (zero-dependency, std only)
// ============================================================================
#include "gle_telemetry.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace gle {

// ---------------------------------------------------------------- FNV-1a 64
static uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

static std::string to_hex(uint64_t v) {
    std::ostringstream os;
    os << std::hex << std::setw(16) << std::setfill('0') << v;
    return os.str();
}

// ------------------------------------------------------------- JSON helpers
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

static std::string json_str(const char* key, const std::string& val) {
    return std::string("\"") + key + "\":\"" + json_escape(val) + "\"";
}

// Extract "key":"value" (string fields only — numbers stay in `numbers` blob).
static bool extract_str(const std::string& line, const char* key, std::string* out) {
    std::string pat = std::string("\"") + key + "\":\"";
    size_t p = line.find(pat);
    if (p == std::string::npos) return false;
    p += pat.size();
    std::string val;
    while (p < line.size()) {
        char c = line[p];
        if (c == '\\' && p + 1 < line.size()) {
            char n = line[p + 1];
            switch (n) {
                case '"':  val += '"';  break;
                case '\\': val += '\\'; break;
                case 'n':  val += '\n'; break;
                case 'r':  val += '\r'; break;
                case 't':  val += '\t'; break;
                default:   val += n;    break;
            }
            p += 2;
            continue;
        }
        if (c == '"') break;
        val += c;
        ++p;
    }
    *out = val;
    return true;
}

// ---------------------------------------------------------------- timestamps
static std::string iso8601_now() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t tt = system_clock::to_time_t(now);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::tm tmv{};
#if defined(_WIN32)
    // BUGFIX (bug census): localtime_s/gmtime_s/mktime return codes were
    // unchecked — failure left tmv/utc zeroed (epoch timestamp, silent).
    // Fail closed to epoch+offset-unknown instead.
    if (localtime_s(&tmv, &tt) != 0) tmv = {};
#else
    if (localtime_r(&tt, &tmv) == nullptr) tmv = {};
#endif
    std::ostringstream os;
    os << std::put_time(&tmv, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(3)
       << std::setfill('0') << (int)ms.count();
    // Offset (minutes east of UTC)
#if defined(_WIN32)
    long off_min = 0;
    static bool got_off = false;
    static long cached = 0;
    if (!got_off) {
        struct tm utc{};
        if (gmtime_s(&utc, &tt) == 0) {
            std::tm tmv_copy = tmv, utc_copy = utc;
            std::time_t t_local = mktime(&tmv_copy), t_utc = mktime(&utc_copy);
            if (t_local != (std::time_t)-1 && t_utc != (std::time_t)-1)
                cached = (long)((t_local - t_utc) / 60);
        }
        got_off = true;
    }
    off_min = cached;
    char sign = off_min >= 0 ? '+' : '-';
    long a = off_min < 0 ? -off_min : off_min;
    os << sign << std::setw(2) << std::setfill('0') << (a / 60) << ':'
       << std::setw(2) << std::setfill('0') << (a % 60);
#endif
    return os.str();
}

// ---------------------------------------------------------------- event name
const char* event_type_name(EventType t) {
    switch (t) {
        case EventType::RunStarted:      return "RUN_STARTED";
        case EventType::BarFrozen:       return "BAR_FROZEN";
        case EventType::UnitSplit:       return "UNIT_SPLIT";
        case EventType::BuildDone:       return "BUILD_DONE";
        case EventType::CriticVerdict:   return "CRITIC_VERDICT";
        case EventType::ArbiterRuling:   return "ARBITER_RULING";
        case EventType::AnticheatSweep:  return "ANTICHEAT_SWEEP";
        case EventType::BuildResult:     return "BUILD_RESULT";
        case EventType::TestResult:      return "TEST_RESULT";
        case EventType::BenchSnapshot:   return "BENCH_SNAPSHOT";
        case EventType::RegressionCheck: return "REGRESSION_CHECK";
        case EventType::ClaimVerified:   return "CLAIM_VERIFIED";
        case EventType::PhaseExit:       return "PHASE_EXIT";
        case EventType::PanicStop:       return "PANIC_STOP";
        case EventType::RunEnded:        return "RUN_ENDED";
        case EventType::AgentStarted:    return "AGENT_STARTED";
        case EventType::InvalidRound:    return "INVALID_ROUND";
    }
    return "UNKNOWN";
}

// ---------------------------------------------------------------- serialize
std::string to_json_line(const Event& e) {
    std::ostringstream os;
    os << "{" << json_str("ts", e.ts)
       << "," << json_str("run_id", e.run_id)
       << "," << json_str("phase", e.phase)
       << "," << json_str("task", e.task)
       << "," << json_str("event", event_type_name(e.type))
       << "," << json_str("actor", e.actor)
       << "," << json_str("verdict", e.verdict)
       << "," << json_str("evidence", e.evidence)
       << "," << json_str("prev_hash", e.prev_hash);
    if (!e.numbers.empty())
        os << ",\"numbers\":" << e.numbers;   // raw object
    else
        os << ",\"numbers\":{}";
    os << "," << json_str("hash", e.hash) << "}";
    return os.str();
}

// ---------------------------------------------------------------- writer
struct TelemetryWriter::Impl {
    std::ofstream file;
    std::string path;
    std::string last_hash = "0";
};

TelemetryWriter::TelemetryWriter(const std::string& path) : impl_(new Impl) {
    impl_->path = path;
    // Continue an existing chain instead of truncating (append-only rule).
    impl_->file.open(path, std::ios::app);
    TelemetryReader r(path);   // find last valid hash
    ChainReport rep = r.verify_chain();
    if (!r.events().empty()) impl_->last_hash = r.events().back().hash;
    else impl_->last_hash = "0";
    // BUGFIX (bug census): chain-broken input was (void)rep-discarded —
    // appending to a corrupt chain silently. Warn loudly instead.
    if (!rep.chain_valid) {
        std::fprintf(stderr, "[telemetry] WARNING: existing chain %s broken at line %llu "
                             "(%llu skipped); appending anyway (audit trail preserved)\n",
                     path.c_str(), (unsigned long long)rep.first_bad_line,
                     (unsigned long long)rep.skipped_lines);
    }
}

TelemetryWriter::~TelemetryWriter() { close(); }

std::string TelemetryWriter::append(const Event& e) {
    Event stored = e;
    stored.ts = iso8601_now();
    stored.prev_hash = impl_->last_hash;

    // Canonical payload = every field except `hash`.
    Event canonical = stored;
    canonical.hash.clear();
    const std::string payload = to_json_line(canonical);
    stored.hash = to_hex(fnv1a(payload));

    impl_->file << to_json_line(stored) << '\n';
    impl_->file.flush();          // crash-safe-ish: each event hits disk
    impl_->last_hash = stored.hash;
    return stored.hash;
}

void TelemetryWriter::close() {
    if (impl_) {
        if (impl_->file.is_open()) impl_->file.close();
        delete impl_;
        impl_ = nullptr;
    }
}

// ---------------------------------------------------------------- reader
static bool parse_line(const std::string& line, Event* e) {
    // Fast structural check.
    if (line.size() < 2 || line.front() != '{' || line.back() != '}') return false;
    bool ok = true;
    std::string ev;
    ok &= extract_str(line, "ts", &e->ts);
    ok &= extract_str(line, "run_id", &e->run_id);
    ok &= extract_str(line, "phase", &e->phase);
    ok &= extract_str(line, "task", &e->task);
    ok &= extract_str(line, "event", &ev);
    ok &= extract_str(line, "actor", &e->actor);
    ok &= extract_str(line, "verdict", &e->verdict);
    ok &= extract_str(line, "evidence", &e->evidence);
    ok &= extract_str(line, "prev_hash", &e->prev_hash);
    ok &= extract_str(line, "hash", &e->hash);
    e->type = EventType::Unknown;
    for (int i = 0; i <= (int)EventType::InvalidRound; ++i) {
        if (ev == event_type_name((EventType)i)) { e->type = (EventType)i; break; }
    }
    if (e->type == EventType::Unknown) return false;
    size_t np = line.find("\"numbers\":");
    if (np != std::string::npos) {
        size_t vstart = np + strlen("\"numbers\":");
        size_t vend = line.rfind(",\"hash\"");
        if (vend != std::string::npos && vend > vstart)
            e->numbers = line.substr(vstart, vend - vstart);
        else
            e->numbers = "{}";
    } else {
        e->numbers = "{}";
    }
    return ok;
}

TelemetryReader::TelemetryReader(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return;
    std::string line;
    size_t lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        if (line.empty()) continue;
        Event e;
        if (line.back() != '}') {           // partial tail (crash mid-write)
            report_.truncated_tail = true;
            report_.skipped_lines++;
            report_.lines_read = lineno;
            continue;
        }
        if (!parse_line(line, &e)) {
            report_.skipped_lines++;
            continue;
        }
        events_.push_back(e);
        report_.events_parsed++;
    }
    report_.lines_read = lineno;
}

ChainReport TelemetryReader::verify_chain() const {
    ChainReport rep = report_;
    rep.chain_valid = true;
    rep.first_bad_line = 0;
    std::string expect_prev = "0";
    size_t line_no = 0;
    for (const auto& e : events_) {
        ++line_no;
        Event canonical = e;
        canonical.hash.clear();
        canonical.prev_hash = e.prev_hash;
        const std::string payload = to_json_line(canonical);
        const std::string want = to_hex(fnv1a(payload));
        if (e.prev_hash != expect_prev || e.hash != want) {
            rep.chain_valid = false;
            rep.first_bad_line = line_no;
            return rep;
        }
        expect_prev = e.hash;
    }
    return rep;
}

} // namespace gle
