#include "quant/multi_agent.h"
#include "quant/random.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <numeric>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <thread>
#include <chrono>
#include <functional>
#include <set>

namespace quant {
namespace multi_agent {

namespace {

std::string role_name(AgentRole r) {
    switch (r) {
        case AgentRole::SUPERVISOR: return "Supervisor";
        case AgentRole::PLANNER:    return "Planner";
        case AgentRole::CODER:      return "Coder";
        case AgentRole::REVIEWER:   return "Reviewer";
        case AgentRole::TESTER:     return "Tester";
        case AgentRole::OPTIMIZER:  return "Optimizer";
        default:                    return "Custom";
    }
}

double now_ms() {
    return (double)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int64_t timestamp_now() {
    return (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // anonymous namespace

// ========================================================================
// MultiAgentSystem
// ========================================================================

MultiAgentSystem::MultiAgentSystem(int n_agents, CommTopology topology)
    : topology_(topology) {
    agents_.reserve(n_agents);
    for (int i = 0; i < n_agents; i++) {
        AgentState agent;
        agent.agent_id = i;
        agent.role = (i == 0) ? AgentRole::SUPERVISOR : AgentRole::CUSTOM;
        agent.name = role_name(agent.role) + "_" + std::to_string(i);
        agent.status = "idle";
        agent.active = true;
        agents_.push_back(agent);
    }
}

MultiAgentSystem::~MultiAgentSystem() {}

int MultiAgentSystem::add_agent(AgentRole role, const std::string& name) {
    int id = (int)agents_.size();
    AgentState agent;
    agent.agent_id = id;
    agent.role = role;
    agent.name = name.empty() ? role_name(role) + "_" + std::to_string(id) : name;
    agent.status = "idle";
    agent.active = true;
    agents_.push_back(agent);
    return id;
}

void MultiAgentSystem::remove_agent(int agent_id) {
    if (agent_id < 0 || agent_id >= (int)agents_.size()) return;
    agents_[agent_id].active = false;
    agents_[agent_id].status = "removed";
}

AgentState& MultiAgentSystem::get_agent(int agent_id) {
    return agents_[agent_id];
}

const AgentState& MultiAgentSystem::get_agent(int agent_id) const {
    return agents_[agent_id];
}

int MultiAgentSystem::agent_count() const {
    return (int)agents_.size();
}

// ========================================================================
// Message passing
// ========================================================================

void MultiAgentSystem::send_message(const AgentMessage& msg) {
    AgentMessage m = msg;
    m.id = next_message_id_++;
    if (m.timestamp == 0) m.timestamp = timestamp_now();
    message_queue_.push_back(m);
    if ((int64_t)message_queue_.size() > max_queue_size_) {
        message_queue_.erase(message_queue_.begin());
    }
    route_message(m);
}

std::vector<AgentMessage> MultiAgentSystem::receive_messages(int agent_id) {
    std::vector<AgentMessage> result;
    for (auto& msg : message_queue_) {
        if (msg.to_agent == agent_id || msg.to_agent == -1) {
            result.push_back(msg);
        }
    }
    return result;
}

std::vector<AgentMessage> MultiAgentSystem::broadcast(int from_agent, MessageType type, const std::string& content) {
    std::vector<AgentMessage> sent;
    for (int i = 0; i < (int)agents_.size(); i++) {
        if (i == from_agent || !agents_[i].active) continue;
        if (!is_connected(from_agent, i)) continue;
        AgentMessage msg;
        msg.from_agent = from_agent;
        msg.to_agent = i;
        msg.type = type;
        msg.content = content;
        msg.timestamp = timestamp_now();
        send_message(msg);
        sent.push_back(msg);
    }
    return sent;
}

bool MultiAgentSystem::communicate_direct(int from, int to, const std::string& content) {
    if (from < 0 || from >= (int)agents_.size() || to < 0 || to >= (int)agents_.size()) return false;
    if (!agents_[from].active || !agents_[to].active) return false;
    if (!is_connected(from, to)) return false;

    AgentMessage msg;
    msg.from_agent = from;
    msg.to_agent = to;
    msg.type = MessageType::STATUS_UPDATE;
    msg.content = content;
    msg.timestamp = timestamp_now();
    send_message(msg);

    agents_[from].metrics.messages_sent++;
    agents_[to].metrics.messages_received++;
    update_agent_context(to, "[From " + agents_[from].name + "]: " + content);
    return true;
}

void MultiAgentSystem::route_message(const AgentMessage& msg) {
    agents_[msg.from_agent].history.push_back(
        "[" + std::to_string(msg.timestamp) + "] TO " +
        (msg.to_agent >= 0 ? agents_[msg.to_agent].name : "ALL") +
        ": " + msg.content);
}

bool MultiAgentSystem::is_connected(int from, int to) const {
    if (from == to) return true;
    if (from < 0 || from >= (int)agents_.size() || to < 0 || to >= (int)agents_.size()) return false;

    switch (topology_) {
        case CommTopology::STAR:
            return from == 0 || to == 0;
        case CommTopology::RING: {
            int n = (int)agents_.size();
            return (from + 1) % n == to || (to + 1) % n == from;
        }
        case CommTopology::MESH:
            return true;
        case CommTopology::HIERARCHICAL:
            return from == 0 || to == 0 || std::abs(from - to) <= 1;
        default:
            return true;
    }
}

// ========================================================================
// Task assignment
// ========================================================================

void MultiAgentSystem::assign_task(int agent_id, const std::string& task) {
    if (agent_id < 0 || agent_id >= (int)agents_.size()) return;
    if (!agents_[agent_id].active) return;

    agents_[agent_id].status = "working";
    agents_[agent_id].tasks_assigned++;
    agents_[agent_id].context_window.push_back("TASK: " + task);

    AgentMessage msg;
    msg.from_agent = 0;
    msg.to_agent = agent_id;
    msg.type = MessageType::TASK_ASSIGN;
    msg.content = task;
    msg.timestamp = timestamp_now();
    send_message(msg);
}

void MultiAgentSystem::assign_task_to_role(AgentRole role, const std::string& task) {
    int best = -1;
    float best_score = -1.0f;
    for (int i = 0; i < (int)agents_.size(); i++) {
        if (!agents_[i].active || agents_[i].role != role) continue;
        float score = compute_task_agent_affinity(task, agents_[i]);
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    if (best >= 0) assign_task(best, task);
}

std::string MultiAgentSystem::get_task_status(int64_t task_id) const {
    for (auto& st : subtasks_) {
        if (st.id == task_id) return st.status;
    }
    return "not_found";
}

float MultiAgentSystem::compute_task_agent_affinity(const std::string& task, const AgentState& agent) const {
    float score = 0.5f;
    std::string task_lower = task;
    std::transform(task_lower.begin(), task_lower.end(), task_lower.begin(), ::tolower);

    if (agent.role == AgentRole::PLANNER && task_lower.find("plan") != std::string::npos) score += 0.3f;
    if (agent.role == AgentRole::CODER && (task_lower.find("code") != std::string::npos || task_lower.find("implement") != std::string::npos)) score += 0.3f;
    if (agent.role == AgentRole::REVIEWER && (task_lower.find("review") != std::string::npos || task_lower.find("check") != std::string::npos)) score += 0.3f;
    if (agent.role == AgentRole::TESTER && (task_lower.find("test") != std::string::npos || task_lower.find("verify") != std::string::npos)) score += 0.3f;
    if (agent.role == AgentRole::OPTIMIZER && (task_lower.find("optim") != std::string::npos || task_lower.find("speed") != std::string::npos)) score += 0.3f;

    if (agent.metrics.tasks_completed > 0) {
        score += 0.1f * std::min(1.0f, agent.metrics.avg_quality);
    }
    return score;
}

int MultiAgentSystem::find_best_agent_for_task(const std::string& task_description) const {
    int best = -1;
    float best_score = -1.0f;
    for (int i = 0; i < (int)agents_.size(); i++) {
        if (!agents_[i].active || agents_[i].status == "working") continue;
        float score = compute_task_agent_affinity(task_description, agents_[i]);
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    return best;
}

// ========================================================================
// Task decomposition
// ========================================================================

void MultiAgentSystem::decompose_task(const std::string& complex_task, int max_subtasks) {
    subtasks_.clear();
    std::string task_lower = complex_task;
    std::transform(task_lower.begin(), task_lower.end(), task_lower.begin(), ::tolower);

    std::vector<std::pair<std::string, AgentRole>> decomposed;
    decomposed.push_back({"Analyze requirements: " + complex_task, AgentRole::PLANNER});
    decomposed.push_back({"Design solution for: " + complex_task, AgentRole::PLANNER});

    if (task_lower.find("code") != std::string::npos || task_lower.find("implement") != std::string::npos ||
        task_lower.find("build") != std::string::npos || task_lower.find("write") != std::string::npos) {
        decomposed.push_back({"Implement code for: " + complex_task, AgentRole::CODER});
    }

    if (task_lower.find("optim") != std::string::npos || task_lower.find("performance") != std::string::npos ||
        task_lower.find("speed") != std::string::npos) {
        decomposed.push_back({"Optimize: " + complex_task, AgentRole::OPTIMIZER});
    }

    decomposed.push_back({"Review: " + complex_task, AgentRole::REVIEWER});
    decomposed.push_back({"Test: " + complex_task, AgentRole::TESTER});
    decomposed.push_back({"Final integration: " + complex_task, AgentRole::SUPERVISOR});

    int n = std::min(max_subtasks, (int)decomposed.size());
    for (int i = 0; i < n; i++) {
        Subtask st;
        st.id = next_subtask_id_++;
        st.description = decomposed[i].first;
        st.status = "pending";
        st.progress = 0.0f;
        st.priority = n - i;
        if (i > 0) st.dependencies.push_back(subtasks_.back().id);
        subtasks_.push_back(st);
    }
}

const std::vector<Subtask>& MultiAgentSystem::get_subtasks() const {
    return subtasks_;
}

void MultiAgentSystem::assign_subtask(int64_t subtask_id, int agent_id) {
    for (auto& st : subtasks_) {
        if (st.id == subtask_id) {
            st.assigned_agent = agent_id;
            st.status = "assigned";
            if (agent_id >= 0 && agent_id < (int)agents_.size()) {
                assign_task(agent_id, st.description);
            }
            return;
        }
    }
}

std::vector<Subtask> MultiAgentSystem::get_ready_subtasks() const {
    std::vector<Subtask> ready;
    for (auto& st : subtasks_) {
        if (st.status == "pending" && check_dependency_satisfied(st)) {
            ready.push_back(st);
        }
    }
    return ready;
}

bool MultiAgentSystem::check_dependency_satisfied(const Subtask& st) const {
    for (int64_t dep_id : st.dependencies) {
        bool found_done = false;
        for (auto& other : subtasks_) {
            if (other.id == dep_id && other.status == "completed") {
                found_done = true;
                break;
            }
        }
        if (!found_done) return false;
    }
    return true;
}

void MultiAgentSystem::assign_subtask_by_role(AgentRole role, int64_t subtask_id) {
    for (auto& st : subtasks_) {
        if (st.id == subtask_id) {
            int best = -1;
            float best_score = -1.0f;
            for (int i = 0; i < (int)agents_.size(); i++) {
                if (!agents_[i].active || agents_[i].role != role) continue;
                if (agents_[i].status == "working") continue;
                float score = agents_[i].metrics.avg_quality + 0.1f;
                if (score > best_score) {
                    best_score = score;
                    best = i;
                }
            }
            if (best >= 0) assign_subtask(subtask_id, best);
            return;
        }
    }
}

// ========================================================================
// Consensus / Voting
// ========================================================================

void MultiAgentSystem::submit_vote(int agent_id, const std::string& proposal,
                                    bool approve, float confidence, const std::string& rationale) {
    if (agent_id < 0 || agent_id >= (int)agents_.size()) return;
    Vote v;
    v.voter_agent = agent_id;
    v.proposal = proposal;
    v.approve = approve;
    v.confidence = confidence;
    v.rationale = rationale;
    votes_.push_back(v);
    agents_[agent_id].metrics.votes_cast++;

    AgentMessage msg;
    msg.from_agent = agent_id;
    msg.to_agent = -1;
    msg.type = MessageType::VOTE_CAST;
    msg.content = (approve ? "APPROVE" : "REJECT") + std::string(" ") + proposal +
                  " confidence=" + std::to_string(confidence);
    msg.timestamp = timestamp_now();
    send_message(msg);
}

bool MultiAgentSystem::check_consensus(float threshold) {
    if (votes_.empty()) return false;
    int active_count = 0;
    for (auto& a : agents_) if (a.active) active_count++;
    if (active_count == 0) return false;

    float approve_weight = 0.0f;
    float total_weight = 0.0f;
    for (auto& v : votes_) {
        float w = v.confidence;
        total_weight += w;
        if (v.approve) approve_weight += w;
    }
    if (total_weight < 1e-10f) return false;
    float ratio = approve_weight / total_weight;
    return ratio >= threshold;
}

std::vector<Vote> MultiAgentSystem::get_votes() const {
    return votes_;
}

void MultiAgentSystem::reset_votes() {
    votes_.clear();
}

// ========================================================================
// Blackboard
// ========================================================================

void MultiAgentSystem::write_blackboard(const std::string& key, const std::string& value, int agent_id) {
    for (auto& entry : blackboard_) {
        if (entry.key == key) {
            entry.value = value;
            entry.author_agent = agent_id;
            entry.timestamp = timestamp_now();
            return;
        }
    }
    BlackboardEntry entry;
    entry.key = key;
    entry.value = value;
    entry.author_agent = agent_id;
    entry.timestamp = timestamp_now();
    blackboard_.push_back(entry);
}

std::string MultiAgentSystem::read_blackboard(const std::string& key) const {
    for (auto& entry : blackboard_) {
        if (entry.key == key) return entry.value;
    }
    return "";
}

std::vector<BlackboardEntry> MultiAgentSystem::search_blackboard(const std::string& prefix) const {
    std::vector<BlackboardEntry> results;
    for (auto& entry : blackboard_) {
        if (entry.key.find(prefix) == 0) results.push_back(entry);
    }
    return results;
}

void MultiAgentSystem::cleanup_blackboard() {
    blackboard_.erase(
        std::remove_if(blackboard_.begin(), blackboard_.end(), [](const BlackboardEntry& e) {
            return e.ttl > 0 && (timestamp_now() - e.timestamp) > e.ttl;
        }),
        blackboard_.end());
}

// ========================================================================
// Hierarchical planning
// ========================================================================

HierarchicalPlan MultiAgentSystem::create_plan(const std::string& goal, int max_subtasks) {
    HierarchicalPlan plan;
    plan.goal = goal;
    plan.completed = false;
    plan.overall_progress = 0.0f;

    std::string goal_lower = goal;
    std::transform(goal_lower.begin(), goal_lower.end(), goal_lower.begin(), ::tolower);

    plan.high_level_steps.push_back("Understand goal: " + goal);
    plan.high_level_steps.push_back("Decompose into subtasks");
    plan.high_level_steps.push_back("Assign to specialist agents");
    plan.high_level_steps.push_back("Execute subtasks in parallel where possible");
    plan.high_level_steps.push_back("Review and integrate results");
    plan.high_level_steps.push_back("Verify completion");

    decompose_task(goal, max_subtasks);
    plan.detailed_subtasks = subtasks_;

    current_plan_ = plan;
    return plan;
}

bool MultiAgentSystem::execute_plan(HierarchicalPlan& plan) {
    if (plan.completed) return true;

    bool all_done = true;
    for (auto& st : plan.detailed_subtasks) {
        if (st.status == "completed") continue;
        all_done = false;

        if (st.status == "pending" && check_dependency_satisfied(st)) {
            if (st.assigned_agent < 0) {
                int agent = find_best_agent_for_task(st.description);
                if (agent >= 0) {
                    assign_subtask(st.id, agent);
                } else {
                    continue;
                }
            }
            st.status = "in_progress";
            st.start_time = now_ms();
        }

        if (st.status == "in_progress") {
            st.progress += 0.1f;
            if (st.progress >= 1.0f) {
                st.progress = 1.0f;
                st.status = "completed";
                st.end_time = now_ms();
                if (st.assigned_agent >= 0 && st.assigned_agent < (int)agents_.size()) {
                    agents_[st.assigned_agent].metrics.tasks_completed++;
                    agents_[st.assigned_agent].status = "idle";
                }
            }
        }

        if (st.status == "failed") {
            supervisor_reassign_failed((int)st.id);
            all_done = false;
        }
    }

    float total = 0.0f;
    for (auto& st : plan.detailed_subtasks) total += st.progress;
    plan.overall_progress = plan.detailed_subtasks.empty() ? 0.0f :
        total / (float)plan.detailed_subtasks.size();

    plan.completed = all_done && !plan.detailed_subtasks.empty();
    return plan.completed;
}

float MultiAgentSystem::get_plan_progress(const HierarchicalPlan& plan) const {
    return plan.overall_progress;
}

// ========================================================================
// Supervisor
// ========================================================================

void MultiAgentSystem::supervisor_monitor() {
    for (int i = 0; i < (int)agents_.size(); i++) {
        if (!agents_[i].active) continue;
        if (agents_[i].role == AgentRole::SUPERVISOR && i != 0) continue;

        auto msgs = receive_messages(i);
        for (auto& msg : msgs) {
            if (msg.type == MessageType::ERROR_REPORT) {
                supervisor_resolve_conflicts();
            }
            if (msg.type == MessageType::STATUS_UPDATE) {
                update_agent_context(i, msg.content);
            }
        }
    }

    for (auto& st : subtasks_) {
        if (st.status == "in_progress" && st.start_time > 0) {
            double elapsed = now_ms() - st.start_time;
            if (elapsed > 30000.0) {
                st.status = "failed";
                supervisor_reassign_failed((int)st.id);
            }
        }
    }
}

void MultiAgentSystem::supervisor_resolve_conflicts() {
    reset_votes();
    for (int i = 0; i < (int)agents_.size(); i++) {
        if (!agents_[i].active) continue;
        submit_vote(i, "resolve_conflict", true, 0.7f, "auto-resolve");
    }
}

void MultiAgentSystem::supervisor_reassign_failed(int subtask_id) {
    for (auto& st : subtasks_) {
        if (st.id == subtask_id) {
            int old_agent = st.assigned_agent;
            st.assigned_agent = -1;
            st.status = "pending";
            st.progress = 0.0f;

            if (old_agent >= 0 && old_agent < (int)agents_.size()) {
                agents_[old_agent].metrics.tasks_failed++;
                agents_[old_agent].status = "idle";

                AgentMessage msg;
                msg.from_agent = 0;
                msg.to_agent = old_agent;
                msg.type = MessageType::FEEDBACK;
                msg.content = "Task reassigned due to failure: " + st.description;
                msg.timestamp = timestamp_now();
                send_message(msg);
            }

            int new_agent = find_best_agent_for_task(st.description);
            if (new_agent >= 0) {
                assign_subtask(st.id, new_agent);
            }
            return;
        }
    }
}

void MultiAgentSystem::supervisor_escalate(const std::string& issue) {
    write_blackboard("ESCALATION_" + std::to_string(timestamp_now()), issue, 0);
    broadcast(0, MessageType::CONFLICT_ALERT, "ESCALATION: " + issue);
}

// ========================================================================
// Agent context
// ========================================================================

void MultiAgentSystem::update_agent_context(int agent_id, const std::string& entry) {
    if (agent_id < 0 || agent_id >= (int)agents_.size()) return;
    agents_[agent_id].context_window.push_back(entry);
    int64_t max_context = 100;
    while ((int64_t)agents_[agent_id].context_window.size() > max_context) {
        agents_[agent_id].context_window.erase(agents_[agent_id].context_window.begin());
    }
    agents_[agent_id].history.push_back(entry);
}

AgentRole MultiAgentSystem::string_to_role(const std::string& role_str) const {
    if (role_str == "supervisor") return AgentRole::SUPERVISOR;
    if (role_str == "planner")    return AgentRole::PLANNER;
    if (role_str == "coder")      return AgentRole::CODER;
    if (role_str == "reviewer")   return AgentRole::REVIEWER;
    if (role_str == "tester")     return AgentRole::TESTER;
    if (role_str == "optimizer")  return AgentRole::OPTIMIZER;
    return AgentRole::CUSTOM;
}

std::string MultiAgentSystem::role_to_string(AgentRole role) const {
    return role_name(role);
}

// ========================================================================
// Metrics
// ========================================================================

void MultiAgentSystem::update_metrics(int agent_id) {
    if (agent_id < 0 || agent_id >= (int)agents_.size()) return;
    auto& m = agents_[agent_id].metrics;
    int total = m.tasks_completed + m.tasks_failed;
    if (total > 0) {
        m.avg_quality = (float)m.tasks_completed / (float)total;
    }
}

AgentMetrics MultiAgentSystem::get_metrics(int agent_id) const {
    if (agent_id < 0 || agent_id >= (int)agents_.size()) return {};
    return agents_[agent_id].metrics;
}

std::vector<AgentMetrics> MultiAgentSystem::get_all_metrics() const {
    std::vector<AgentMetrics> all;
    for (auto& a : agents_) all.push_back(a.metrics);
    return all;
}

std::string MultiAgentSystem::metrics_summary() const {
    std::ostringstream oss;
    oss << "=== Multi-Agent System Metrics ===\n";
    oss << "Agents: " << agents_.size() << "\n";
    oss << "Topology: ";
    switch (topology_) {
        case CommTopology::STAR:   oss << "Star"; break;
        case CommTopology::RING:   oss << "Ring"; break;
        case CommTopology::MESH:   oss << "Mesh"; break;
        case CommTopology::HIERARCHICAL: oss << "Hierarchical"; break;
    }
    oss << "\n";
    oss << "Subtasks: " << subtasks_.size() << "\n";
    oss << "Messages in queue: " << message_queue_.size() << "\n";
    oss << "Blackboard entries: " << blackboard_.size() << "\n";
    oss << "Votes cast: " << votes_.size() << "\n\n";

    for (const auto& a : agents_) {
        if (!a.active) continue;
        oss << "Agent " << a.agent_id << " [" << a.name << "]: "
            << "completed=" << a.metrics.tasks_completed
            << " failed=" << a.metrics.tasks_failed
            << " quality=" << std::fixed << std::setprecision(2) << a.metrics.avg_quality
            << " msgs_sent=" << a.metrics.messages_sent
            << " msgs_rcvd=" << a.metrics.messages_received
            << " votes=" << a.metrics.votes_cast
            << " status=" << a.status << "\n";
    }
    return oss.str();
}

// ========================================================================
// Episode / round execution
// ========================================================================

void MultiAgentSystem::run_round(int max_rounds) {
    for (int round = 0; round < max_rounds; round++) {
        supervisor_monitor();

        auto ready = get_ready_subtasks();
        for (auto& st : ready) {
            int agent = find_best_agent_for_task(st.description);
            if (agent >= 0) {
                assign_subtask(st.id, agent);
                st.status = "in_progress";
                st.start_time = now_ms();
            }
        }

        for (auto& st : subtasks_) {
            if (st.status == "in_progress") {
                st.progress += 0.15f + (float)(round % 3) * 0.05f;
                if (st.progress >= 1.0f) {
                    st.progress = 1.0f;
                    st.status = "completed";
                    st.end_time = now_ms();
                    if (st.assigned_agent >= 0 && st.assigned_agent < (int)agents_.size()) {
                        agents_[st.assigned_agent].metrics.tasks_completed++;
                        agents_[st.assigned_agent].status = "idle";
                    }
                }
            }
        }

        write_blackboard("round_" + std::to_string(round) + "_progress",
                         std::to_string((int)(current_plan_.overall_progress * 100)) + "%", 0);

        if (current_plan_.completed) break;
        if (current_plan_.detailed_subtasks.empty()) break;
    }
}

void MultiAgentSystem::run_episode(int steps) {
    for (int step = 0; step < steps; step++) {
        supervisor_monitor();

        std::mt19937 rng((unsigned)(42 + step));
        for (int i = 0; i < (int)agents_.size(); i++) {
            if (!agents_[i].active || agents_[i].status == "working") continue;
            for (int j = 0; j < (int)agents_.size(); j++) {
                if (i == j || !agents_[j].active) continue;
                if (!is_connected(i, j)) continue;
                if ((int)(rng() % 10) < 3) {
                    std::string content = "Step " + std::to_string(step) + " status update from " + agents_[i].name;
                    communicate_direct(i, j, content);
                }
            }
        }

        for (auto& st : subtasks_) {
            if (st.status == "in_progress") {
                st.progress += 0.2f;
                if (st.progress >= 1.0f) {
                    st.progress = 1.0f;
                    st.status = "completed";
                    st.end_time = now_ms();
                    if (st.assigned_agent >= 0 && st.assigned_agent < (int)agents_.size()) {
                        agents_[st.assigned_agent].metrics.tasks_completed++;
                        agents_[st.assigned_agent].status = "idle";
                    }
                }
            }
        }
    }
}

// ========================================================================
// Configuration
// ========================================================================

void MultiAgentSystem::configure(CommTopology topology) {
    topology_ = topology;
}

void MultiAgentSystem::set_max_queue_size(int64_t max_size) {
    max_queue_size_ = max_size;
}

void MultiAgentSystem::set_communication_delay(int64_t delay_ms) {
    communication_delay_ms_ = delay_ms;
}

std::vector<std::string> MultiAgentSystem::get_histories() const {
    std::vector<std::string> histories;
    for (auto& agent : agents_) {
        std::string h;
        for (auto& msg : agent.history) h += msg + "\n";
        histories.push_back(h);
    }
    return histories;
}

} // namespace multi_agent
} // namespace quant
