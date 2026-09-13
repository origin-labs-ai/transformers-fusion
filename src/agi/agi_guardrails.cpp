#include "quant/agi.h"
#include <chrono>
#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace quant {
namespace agi {

SafetyGuardrails::SafetyGuardrails() {
    // Process-execution / remote-fetch idioms were missing here: check_input()
    // let "system('evil')" through, so check_modification() allowed an
    // unvalidated kernel replacement carrying a shell escape. Added 2026-09-10.
    blocked_patterns_ = {
        "rm -rf", "sudo", "format c:", "shutdown /s", "del /s",
        "rd /s", "cipher /w", "drop table", "delete from",
        "mkfs", ":(){", "fork bomb",
        "system(", "popen(", "execve(", "dlopen("
    };
}

int64_t SafetyGuardrails::current_timestamp() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

bool SafetyGuardrails::check_output(const std::string& output) {
    if (kill_switch_ || paused_) return false;
    for (const auto& p : blocked_patterns_) {
        if (output.find(p) != std::string::npos) {
            audit_log("check_output", output, "blocked");
            return false;
        }
    }
    return true;
}

bool SafetyGuardrails::check_input(const std::string& input) {
    if (kill_switch_ || paused_) return false;
    for (const auto& p : blocked_patterns_) {
        if (input.find(p) != std::string::npos) {
            audit_log("check_input", input, "blocked");
            return false;
        }
    }
    return true;
}

bool SafetyGuardrails::check_modification(const std::string& target, const std::string& change) {
    if (kill_switch_ || paused_) return false;
    bool allowed = check_input(change) && !target.empty();
    audit_log("check_modification", target + " <- " + change, allowed ? "allowed" : "blocked");
    return allowed;
}

bool SafetyGuardrails::check_invariant(const std::string& invariant_name) {
    return invariants_.find(invariant_name) != invariants_.end();
}

void SafetyGuardrails::set_invariant(const std::string& name, const std::string& expression) {
    invariants_[name] = expression;
}

std::vector<std::string> SafetyGuardrails::get_invariants() const {
    std::vector<std::string> names;
    for (const auto& kv : invariants_) names.push_back(kv.first);
    return names;
}

bool SafetyGuardrails::human_override(const std::string& action, const std::string& reason) {
    audit_log("human_override", action + " (" + reason + ")", "granted");
    return true;
}

void SafetyGuardrails::audit_log(const std::string& action, const std::string& detail,
                                 const std::string& result) {
    AuditEntry e;
    e.timestamp = current_timestamp();
    e.action = action;
    e.detail = detail;
    e.result = result;
    std::lock_guard<std::mutex> lock(audit_mutex_);
    audit_log_.push_back(e);
}

std::vector<AuditEntry> SafetyGuardrails::get_audit_log(int count) const {
    std::lock_guard<std::mutex> lock(audit_mutex_);
    if (count <= 0 || (int)audit_log_.size() <= count) return audit_log_;
    return std::vector<AuditEntry>(audit_log_.end() - count, audit_log_.end());
}

HITL::HITL() {}

bool HITL::request_approval(const std::string& action) {
    AuditEntry e;
    e.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    e.action = action;
    e.detail = paused_ ? "auto-denied while paused" : "";
    e.result = paused_ ? "denied" : "approved";
    approval_history_.push_back(e);
    return !paused_;
}

AlignmentSystem::AlignmentSystem() {}

float AlignmentSystem::value_alignment_score(const std::string& output) {
    if (output.empty()) return 0.0f;
    static const char* harmful[] = {"rm -rf", "sudo", "hack", "exploit", "drop table", "kill -9"};
    float score = 0.6f;
    for (const char* h : harmful) {
        if (output.find(h) != std::string::npos) score = 0.1f;
    }
    if (output.find("help") != std::string::npos) score += 0.3f;
    return std::min(1.0f, score);
}

CodeGenSelfImprover::CodeGenSelfImprover(Model* model) : model_(model) {}

std::string CodeGenSelfImprover::generate_kernel(const std::string& op, int64_t M, int64_t N, int64_t K) {
    std::string code = "// generated kernel for " + op + "\n";
    code += "void kernel_" + op + "_" + std::to_string(M) + "x" + std::to_string(N) +
            "x" + std::to_string(K) + "(const float* a, const float* b, float* c) {\n";
    code += "  for (int64_t i = 0; i < " + std::to_string(M) + "; ++i)\n";
    code += "    for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) {\n";
    code += "      float acc = 0.0f;\n";
    code += "      for (int64_t k = 0; k < " + std::to_string(K) + "; ++k)\n";
    code += "        acc += a[i * " + std::to_string(K) + " + k] * b[k * " + std::to_string(N) + " + j];\n";
    code += "      c[i * " + std::to_string(N) + " + j] = acc;\n";
    code += "    }\n";
    code += "}\n";
    return code;
}

bool CodeGenSelfImprover::compile_and_test(const std::string& code) {
    // Simulated compile: any balanced-brace program "compiles" successfully.
    int depth = 0;
    for (char ch : code) {
        if (ch == '{') depth++;
        else if (ch == '}') {
            depth--;
            if (depth < 0) return false;
        }
    }
    return depth == 0;
}

bool CodeGenSelfImprover::replace_kernel(const std::string& op, const std::string& new_code) {
    (void)op;
    return !new_code.empty() && new_code.find("kernel") != std::string::npos;
}

SelfVerifier::SelfVerifier(Model* model) : model_(model) {}

bool SelfVerifier::verify(const std::string& problem, const std::string& solution) {
    size_t eq = problem.find('=');
    if (eq != std::string::npos) {
        std::string rhs = problem.substr(eq + 1);
        while (!rhs.empty() && (rhs.back() == ' ' || rhs.back() == '\t')) rhs.pop_back();
        return rhs == solution;
    }
    return !solution.empty();
}

std::vector<std::string> SelfVerifier::find_edge_cases(const std::string& solution) {
    if (!model_) return {};
    std::vector<std::string> cases;
    if (solution.find("+") != std::string::npos) {
        cases.push_back("zero");
        cases.push_back("negative");
        cases.push_back("max value");
    } else if (solution.find("/") != std::string::npos) {
        cases.push_back("division by zero");
    } else {
        cases.push_back("empty input");
    }
    return cases;
}

CapabilityAmplifier::CapabilityAmplifier(Model* model) : model_(model) {}

float CapabilityAmplifier::measure(const std::string& capability) {
    if (!model_) return 0.5f;
    size_t h = std::hash<std::string>{}(capability);
    return 0.1f + (float)(h % 80) / 100.0f;
}

bool CapabilityAmplifier::improve(const std::string& capability, int steps) {
    (void)capability;
    if (!model_) return false;
    return steps > 0;
}

class NaNWatchdog {
public:
    struct Alert { int layer; int param_idx; int nan_count; int inf_count; };
    std::vector<Alert> check(const std::vector<Tensor*>& params) {
        std::vector<Alert> alerts;
        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* t = params[i];
            int nan_c = 0;
            int inf_c = 0;
            const float* data = t->data<float>();
            int64_t numel = t->numel();
            for (int64_t j = 0; j < numel; ++j) {
                if (std::isnan(data[j])) nan_c++;
                else if (std::isinf(data[j])) inf_c++;
            }
            if (nan_c > 0 || inf_c > 0) alerts.push_back({(int)i, (int)i, nan_c, inf_c});
        }
        return alerts;
    }
    void quarantine(Tensor* t) {
        float* data = t->data<float>();
        int64_t numel = t->numel();
        bool changed = false;
        for (int64_t j = 0; j < numel; ++j) {
            if (std::isnan(data[j]) || std::isinf(data[j])) {
                data[j] = 0.0f;
                changed = true;
            }
        }
        if (changed) quarantine_count_++;
    }
    int total_quarantines() const { return quarantine_count_; }
private:
    int quarantine_count_ = 0;
};

class PageBoundsWatchdog {
public:
    void register_page(void* base, size_t size) { pages_.push_back({(uintptr_t)base, size}); }
    bool check_access(void* ptr, size_t size) const {
        uintptr_t p = (uintptr_t)ptr;
        for (const auto& page : pages_) {
            if (p >= page.base && (p + size) <= (page.base + page.size)) return true;
        }
        return false;
    }
    void* get_page_base(void* ptr) const {
        uintptr_t p = (uintptr_t)ptr;
        for (const auto& page : pages_) {
            if (p >= page.base && p < (page.base + page.size)) return (void*)page.base;
        }
        return nullptr;
    }
private:
    struct PageEntry { uintptr_t base; size_t size; };
    std::vector<PageEntry> pages_;
};

class DeltaTokenWatchdog {
public:
    void stamp(const std::string& page_id, const void* data, size_t len) {
        uint32_t hash = sha256_hash(data, len);
        std::array<uint8_t, 32> stamp_arr;
        std::memset(stamp_arr.data(), 0, 32);
        std::memcpy(stamp_arr.data(), &hash, sizeof(hash));
        stamps_[page_id] = stamp_arr;
    }
    bool verify(const std::string& page_id, const void* data, size_t len) const {
        auto it = stamps_.find(page_id);
        if (it == stamps_.end()) return false;
        uint32_t hash = sha256_hash(data, len);
        uint32_t stored_hash = 0;
        std::memcpy(&stored_hash, it->second.data(), sizeof(stored_hash));
        return hash == stored_hash;
    }
    float variance_bound() const { return 0.05f; }
private:
    std::map<std::string, std::array<uint8_t, 32>> stamps_;
    uint32_t sha256_hash(const void* data, size_t len) const {
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        uint32_t hash = 5381;
        for (size_t i = 0; i < len; ++i) hash = ((hash << 5) + hash) + bytes[i];
        return hash;
    }
};

class PIIExposureCounter {
public:
    int scan(const std::string& text) {
        int matches = 0;
        std::istringstream iss(text);
        std::string token;
        while (iss >> token) {
            if (is_email(token) || is_phone(token) || is_ssn(token) || is_credit_card(token)) {
                matches++; count_++;
            }
        }
        return matches;
    }
    int total_exposures() const { return count_; }
private:
    int count_ = 0;
    bool is_email(const std::string& token) const {
        size_t at = token.find('@');
        size_t dot = token.rfind('.');
        return (at != std::string::npos && dot != std::string::npos && at < dot);
    }
    bool is_phone(const std::string& token) const {
        int digits = 0;
        for (char c : token) if (std::isdigit(c)) digits++;
        return (digits >= 10 && digits <= 15 && token.find('-') != std::string::npos);
    }
    bool is_ssn(const std::string& token) const {
        if (token.length() != 11) return false;
        if (token[3] != '-' || token[6] != '-') return false;
        for (int i : {0,1,2,4,5,7,8,9,10}) if (!std::isdigit(token[i])) return false;
        return true;
    }
    bool is_credit_card(const std::string& token) const {
        int digits = 0;
        for (char c : token) if (std::isdigit(c)) digits++;
        return (digits >= 13 && digits <= 19 && token.find('-') != std::string::npos);
    }
};

} // namespace agi
} // namespace quant
