#include "quant/expert_parallel.h"
#include "quant/moe_variants.h"
#include "quant/math.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "ws2_32.lib")
#else
// POSIX socket compat: the TCP transport below is written against the
// Winsock spelling; these shims map it to BSD sockets so the same code
// compiles on Linux/macOS. Semantics are identical (blocking connect,
// non-blocking data sockets, SO_REUSEADDR listener).
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
using SOCKET = int;
inline int closesocket(SOCKET s) { return ::close(s); }
#ifndef INVALID_SOCKET
#define INVALID_SOCKET (-1)
#endif
#ifndef SOCKET_ERROR
#define SOCKET_ERROR (-1)
#endif
#endif

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <functional>
#include <queue>
#include <fstream>

namespace quant {
namespace expert {

using moe::MoEOutput;

static const int EP_MSG_TOKEN_DATA = 1;
static const int EP_MSG_EXPERT_WEIGHTS = 2;
static const int EP_MSG_HEARTBEAT = 3;
static const int EP_MSG_NODE_JOIN = 4;
static const int EP_MSG_NODE_LEAVE = 5;
static const int EP_MSG_REDISTRIBUTION = 6;
static const int EP_MSG_ACK = 7;
static const int EP_MSG_ROUTING = 8;
static const int EP_MSG_GRADIENT_SYNC = 9;

struct EPMessageHeader {
    int32_t msg_type;
    int32_t payload_size;
    int64_t src_node;
    int64_t dst_node;
    int64_t expert_id;
    int64_t token_count;
    int64_t reserved[3];
};

static_assert(sizeof(EPMessageHeader) == 64, "EPMessageHeader must be 64 bytes");

struct SocketEntry {
    SOCKET sock = INVALID_SOCKET;
    std::string address;
    int64_t port = 0;
    bool connected = false;
    std::atomic<int64_t> bytes_sent{0};
    std::atomic<int64_t> bytes_received{0};
    std::mutex send_mtx;
    std::mutex recv_mtx;
};

struct TCPCommBackend::Impl {
    SOCKET listen_sock = INVALID_SOCKET;
    std::unordered_map<int64_t, std::unique_ptr<SocketEntry>> connections;
    mutable std::mutex mtx;
    bool wsa_initialized = false;
    std::atomic<bool> running{false};

    bool init_wsa() {
        if (wsa_initialized) return true;
#if defined(_WIN32)
        WSADATA wd;
        int err = WSAStartup(MAKEWORD(2, 2), &wd);
        wsa_initialized = (err == 0);
#else
        // No startup handshake on POSIX; sockets are usable immediately.
        wsa_initialized = true;
#endif
        return wsa_initialized;
    }

    void cleanup_wsa() {
#if defined(_WIN32)
        if (wsa_initialized) { WSACleanup(); wsa_initialized = false; }
#else
        wsa_initialized = false;
#endif
    }
};

TCPCommBackend::TCPCommBackend() : impl_(new Impl()) {}
TCPCommBackend::~TCPCommBackend() { shutdown(); }

bool TCPCommBackend::listen(int64_t port) {
    if (!impl_->init_wsa()) return false;
    impl_->listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (impl_->listen_sock == INVALID_SOCKET) return false;

    int opt = 1;
    setsockopt(impl_->listen_sock, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((unsigned short)port);

    if (bind(impl_->listen_sock, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        closesocket(impl_->listen_sock);
        impl_->listen_sock = INVALID_SOCKET;
        return false;
    }
    if (::listen(impl_->listen_sock, 64) == SOCKET_ERROR) {
        closesocket(impl_->listen_sock);
        impl_->listen_sock = INVALID_SOCKET;
        return false;
    }
    impl_->running = true;
    return true;
}

bool TCPCommBackend::connect(const std::string& address, int64_t port) {
    if (!impl_->init_wsa()) return false;

    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) return false;

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);

    std::string addr_str = address;
    if (inet_pton(AF_INET, addr_str.c_str(), &addr.sin_addr) != 1) {
        addr.sin_addr.s_addr = INADDR_NONE;
    }
    if (addr.sin_addr.s_addr == INADDR_NONE) {
        struct addrinfo hints = {}, *result = nullptr;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;
        std::string port_str = std::to_string(port);
        int gai_ret = getaddrinfo(addr_str.c_str(), port_str.c_str(), &hints, &result);
        if (gai_ret == 0 && result) {
            auto* sin = reinterpret_cast<sockaddr_in*>(result->ai_addr);
            memcpy(&addr.sin_addr, &sin->sin_addr, sizeof(addr.sin_addr));
            freeaddrinfo(result);
        } else {
            if (result) freeaddrinfo(result);
            closesocket(sock);
            return false;
        }
    }

    if (::connect(sock, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        closesocket(sock);
        return false;
    }

    u_long nonblock = 1;
#if defined(_WIN32)
    ioctlsocket(sock, FIONBIO, &nonblock);
#else
    // POSIX equivalent of FIONBIO: set O_NONBLOCK, preserving existing flags.
    int fl = fcntl(sock, F_GETFL, 0);
    if (fl < 0) fl = 0;
    fcntl(sock, F_SETFL, fl | O_NONBLOCK);
#endif

    auto entry = std::make_unique<SocketEntry>();
    entry->sock = sock;
    entry->address = address;
    entry->port = port;
    entry->connected = true;

    int64_t node_id = (int64_t)sock;
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        impl_->connections[node_id] = std::move(entry);
    }
    return true;
}

void TCPCommBackend::disconnect(int64_t node_id) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    auto it = impl_->connections.find(node_id);
    if (it != impl_->connections.end()) {
        if (it->second->sock != INVALID_SOCKET)
            closesocket(it->second->sock);
        it->second->connected = false;
        impl_->connections.erase(it);
    }
}

bool TCPCommBackend::send(int64_t node_id, const void* data, int64_t bytes) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    auto it = impl_->connections.find(node_id);
    if (it == impl_->connections.end() || !it->second->connected) return false;

    std::lock_guard<std::mutex> slk(it->second->send_mtx);
    const char* ptr = (const char*)data;
    int64_t remaining = bytes;
    while (remaining > 0) {
        int sent = ::send(it->second->sock, ptr, (int)std::min(remaining, (int64_t)65536), 0);
        if (sent == SOCKET_ERROR || sent == 0) return false;
        ptr += sent;
        remaining -= sent;
    }
    it->second->bytes_sent += bytes;
    return true;
}

bool TCPCommBackend::recv(int64_t node_id, void* buffer, int64_t max_bytes, int64_t* actual_bytes) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    auto it = impl_->connections.find(node_id);
    if (it == impl_->connections.end() || !it->second->connected) return false;

    std::lock_guard<std::mutex> rlk(it->second->recv_mtx);
    char* ptr = (char*)buffer;
    int64_t total_read = 0;
    while (total_read < max_bytes) {
        int got = ::recv(it->second->sock, ptr, (int)std::min(max_bytes - total_read, (int64_t)65536), 0);
        if (got == SOCKET_ERROR || got == 0) break;
        ptr += got;
        total_read += got;
    }
    if (actual_bytes) *actual_bytes = total_read;
    it->second->bytes_received += total_read;
    return total_read > 0;
}

bool TCPCommBackend::send_async(int64_t node_id, const void* data, int64_t bytes) {
    return send(node_id, data, bytes);
}

bool TCPCommBackend::recv_async(int64_t node_id, void* buffer, int64_t max_bytes, int64_t* actual_bytes) {
    return recv(node_id, buffer, max_bytes, actual_bytes);
}

void TCPCommBackend::wait_all() {
    // Wait for all in-flight async operations to complete
    // Since send_async/recv_async fall back to synchronous on this impl,
    // this is effectively a no-op, but we add a small yield for future
    // true-async implementations.
    std::this_thread::yield();
}

bool TCPCommBackend::is_connected(int64_t node_id) const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    auto it = impl_->connections.find(node_id);
    return it != impl_->connections.end() && it->second->connected;
}

int64_t TCPCommBackend::bytes_sent(int64_t node_id) const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    auto it = impl_->connections.find(node_id);
    return it != impl_->connections.end() ? it->second->bytes_sent.load() : 0;
}

int64_t TCPCommBackend::bytes_received(int64_t node_id) const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    auto it = impl_->connections.find(node_id);
    return it != impl_->connections.end() ? it->second->bytes_received.load() : 0;
}

void TCPCommBackend::shutdown() {
    impl_->running = false;
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        for (auto& kv : impl_->connections) {
            if (kv.second->sock != INVALID_SOCKET) {
                closesocket(kv.second->sock);
                kv.second->connected = false;
            }
        }
        impl_->connections.clear();
        if (impl_->listen_sock != INVALID_SOCKET) {
            closesocket(impl_->listen_sock);
            impl_->listen_sock = INVALID_SOCKET;
        }
    }
    impl_->cleanup_wsa();
}

ExpertDispatcher::ExpertDispatcher(const ClusterConfig& cfg) : config(cfg) {
    assignments_.resize(cfg.num_experts);
}

ExpertDispatcher::~ExpertDispatcher() {}

void ExpertDispatcher::assign_experts_to_nodes(int64_t num_nodes) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (num_nodes <= 0) return;

    int64_t experts_per_node = config.num_experts / num_nodes;
    int64_t remainder = config.num_experts % num_nodes;

    int64_t expert_id = 0;
    for (int64_t node = 0; node < num_nodes; node++) {
        int64_t count = experts_per_node + (node < remainder ? 1 : 0);
        for (int64_t i = 0; i < count; i++) {
            assignments_[(size_t)expert_id].expert_id = expert_id;
            assignments_[(size_t)expert_id].node_id = node;
            assignments_[(size_t)expert_id].token_count = 0;
            assignments_[(size_t)expert_id].capacity =
                (int64_t)(config.capacity_factor * (float)(config.max_tokens_per_expert > 0
                    ? config.max_tokens_per_expert : 1024));
            assignments_[(size_t)expert_id].avg_weight = 0.0f;
            expert_id++;
        }
    }
}

ExpertAssignment ExpertDispatcher::get_expert_assignment(int64_t expert_id) const {
    std::lock_guard<std::mutex> lk(mtx_);
    if (expert_id >= 0 && expert_id < (int64_t)assignments_.size())
        return assignments_[(size_t)expert_id];
    return {};
}

std::vector<ExpertAssignment> ExpertDispatcher::get_all_assignments() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return assignments_;
}

RoutingDecision ExpertDispatcher::route_tokens(const Tensor& logits, int64_t num_tokens,
                                                int64_t local_node_id, int64_t num_nodes) {
    RoutingDecision rd;
    int64_t E = config.num_experts;
    int64_t K = config.top_k;

    rd.tokens_per_node.resize((size_t)num_nodes, 0);
    rd.tokens_per_expert.resize((size_t)E, 0);

    rd.expert_indices = Tensor({num_tokens, K});
    rd.expert_weights = Tensor({num_tokens, K});
    rd.expert_indices.zero_();
    rd.expert_weights.zero_();

    const float* ld = logits.data<float>();
    float* idx_d = rd.expert_indices.data<float>();
    float* wt_d = rd.expert_weights.data<float>();
    double z_acc = 0.0;

    for (int64_t t = 0; t < num_tokens; t++) {
        std::vector<std::pair<float, int64_t>> scored;
        scored.reserve(E);
        for (int64_t e = 0; e < E; e++)
            scored.push_back({ld[t * E + e], e});

        std::partial_sort(scored.begin(), scored.begin() + std::min(K, E),
                          scored.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        float max_val = scored[0].first;
        float sum_exp = 0;
        std::vector<float> weights((size_t)K, 0.0f);
        for (int64_t k = 0; k < K && k < (int64_t)scored.size(); k++) {
            weights[(size_t)k] = std::exp(scored[(size_t)k].first - max_val);
            sum_exp += weights[(size_t)k];
        }
        for (int64_t k = 0; k < K; k++)
            weights[(size_t)k] /= (sum_exp + 1e-10f);

        // ST-MoE z-loss needs full-logit log-partition; only available here.
        double lse = 0.0;
        for (int64_t e = 0; e < E; e++) lse += std::exp((double)ld[t * E + e] - (double)max_val);
        lse = std::log(lse + 1e-20) + (double)max_val;
        z_acc += lse * lse;

        for (int64_t k = 0; k < K && k < (int64_t)scored.size(); k++) {
            int64_t expert_id = scored[(size_t)k].second;
            idx_d[t * K + k] = (float)expert_id;
            wt_d[t * K + k] = weights[(size_t)k];

            rd.tokens_per_expert[(size_t)expert_id]++;
            std::lock_guard<std::mutex> lk(mtx_);
            if (expert_id >= 0 && expert_id < (int64_t)assignments_.size()) {
                int64_t target_node = assignments_[(size_t)expert_id].node_id;
                if (target_node >= 0 && target_node < num_nodes)
                    rd.tokens_per_node[(size_t)target_node]++;
            }
        }
    }

    compute_load_balance(rd, rd.load_balance_loss, rd.z_loss);
    // Overwrite the preserved value with the real log-partition z-loss.
    rd.z_loss = (float)(z_acc / (double)std::max<int64_t>(1, num_tokens)) * config.z_loss_coef;
    return rd;
}

void ExpertDispatcher::compute_load_balance(const RoutingDecision& routing,
                                             float& lb_loss, float& z_loss) {
    int64_t E = config.num_experts;
    int64_t T = (int64_t)routing.tokens_per_expert.size() > 0 ?
                std::accumulate(routing.tokens_per_expert.begin(),
                               routing.tokens_per_expert.end(), (int64_t)0) / std::max((int64_t)1, E) : 0;

    const float* wt_d = routing.expert_weights.data<float>();
    const float* idx_d = routing.expert_indices.data<float>();
    int64_t num_tokens = routing.expert_indices.numel() / std::max((int64_t)1, config.top_k);
    int64_t K = config.top_k;

    lb_loss = 0;
    for (int64_t e = 0; e < E; e++) {
        float f_i = (float)routing.tokens_per_expert[(size_t)e] / (float)std::max((int64_t)1, num_tokens);
        float P_i = 0;
        for (int64_t t = 0; t < num_tokens; t++)
            for (int64_t k = 0; k < K; k++)
                if ((int64_t)idx_d[t * K + k] == e)
                    P_i += wt_d[t * K + k];
        P_i /= (float)std::max((int64_t)1, num_tokens);
        lb_loss += f_i * P_i;
    }
    lb_loss *= (float)E;

    // z-loss needs full router logits, unavailable here: preserve the value
    // computed in route_tokens() (ST-MoE log-partition penalty). Do NOT
    // overwrite with any local approximation.
    z_loss = routing.z_loss;
}

AllToAllPlan ExpertDispatcher::plan_all_to_all(const RoutingDecision& routing,
                                                int64_t local_node_id) {
    AllToAllPlan plan;
    int64_t K = config.top_k;
    int64_t D = config.hidden_size;
    const float* idx_d = routing.expert_indices.data<float>();
    int64_t num_tokens = routing.expert_indices.numel() / std::max((int64_t)1, K);

    std::unordered_map<int64_t, std::unordered_map<int64_t, int64_t>> node_expert_counts;

    for (int64_t t = 0; t < num_tokens; t++) {
        for (int64_t k = 0; k < K; k++) {
            int64_t expert_id = (int64_t)idx_d[t * K + k];
            std::lock_guard<std::mutex> lk(mtx_);
            if (expert_id >= 0 && expert_id < (int64_t)assignments_.size()) {
                int64_t dst = assignments_[(size_t)expert_id].node_id;
                if (dst != local_node_id) {
                    node_expert_counts[dst][expert_id]++;
                }
            }
        }
    }

    int64_t offset = 0;
    for (auto& kv : node_expert_counts) {
        int64_t dst_node = kv.first;
        for (auto& ekv : kv.second) {
            AllToAllPlan::Transfer tr;
            tr.src_node = local_node_id;
            tr.dst_node = dst_node;
            tr.expert_id = ekv.first;
            tr.token_count = ekv.second;
            tr.data_offset = offset;
            tr.data_bytes = ekv.second * D * (int64_t)sizeof(float);
            plan.transfers.push_back(tr);
            offset += tr.data_bytes;
            plan.total_bytes += tr.data_bytes;
        }
    }

    plan.max_peers = (int64_t)node_expert_counts.size();
    return plan;
}

Tensor ExpertDispatcher::dispatch_tokens(const Tensor& tokens, const AllToAllPlan& plan,
                                          int64_t local_node_id) {
    (void)local_node_id;
    int64_t D = config.hidden_size;
    int64_t total_dispatched = 0;
    for (auto& tr : plan.transfers)
        total_dispatched += tr.token_count;

    if (total_dispatched == 0)
        return Tensor({0, D});

    Tensor dispatched({total_dispatched, D});
    dispatched.zero_();

    const float* td = tokens.data<float>();
    float* dd = dispatched.data<float>();
    int64_t write_offset = 0;

    for (auto& tr : plan.transfers) {
        for (int64_t t = 0; t < tr.token_count; t++) {
            std::memcpy(dd + write_offset * D, td + (tr.data_offset / ((int64_t)sizeof(float) * D) + t) * D,
                        D * sizeof(float));
            write_offset++;
        }
    }
    return dispatched;
}

Tensor ExpertDispatcher::gather_results(const Tensor& expert_outputs,
                                         const AllToAllPlan& plan, int64_t local_node_id) {
    // Gather expert outputs back to original token ordering
    int64_t D = config.hidden_size;
    int64_t total_tokens = 0;
    for (auto& tr : plan.transfers) {
        if (tr.dst_node == local_node_id) {
            total_tokens += tr.token_count;
        }
    }

    if (total_tokens == 0) return Tensor({0, D});

    Tensor gathered({total_tokens, D});
    gathered.zero_();

    const float* eo = expert_outputs.data<float>();
    float* gd = gathered.data<float>();
    int64_t write_offset = 0;

    for (auto& tr : plan.transfers) {
        if (tr.dst_node == local_node_id) {
            int64_t src_offset = tr.data_offset / (D * (int64_t)sizeof(float));
            for (int64_t t = 0; t < tr.token_count; t++) {
                std::memcpy(gd + write_offset * D,
                           eo + (src_offset + t) * D,
                           D * sizeof(float));
                write_offset++;
            }
        }
    }
    return gathered;
}

void ExpertDispatcher::redistribute_expert(int64_t expert_id, int64_t from_node, int64_t to_node) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (expert_id >= 0 && expert_id < (int64_t)assignments_.size()) {
        assignments_[(size_t)expert_id].node_id = to_node;
        assignments_[(size_t)expert_id].token_count = 0;
    }
}

struct ExpertParallelManager::Impl {
    ClusterConfig config;
    int64_t local_node_id_ = 0;
    std::vector<ClusterNode> nodes_;
    TCPCommBackend comm;
    ExpertDispatcher dispatcher;
    ExpertParallelStats stats;
    mutable std::mutex mtx;
    bool active = false;
    std::vector<float> gradient_buffer;
    Linear router_weight;

    Impl(const ClusterConfig& cfg) : config(cfg), dispatcher(cfg),
        router_weight(cfg.hidden_size, cfg.num_experts) {}

    void send_expert_data(int64_t dst_node, const Tensor& data, int64_t expert_id) {
        EPMessageHeader hdr = {};
        hdr.msg_type = EP_MSG_EXPERT_WEIGHTS;
        hdr.payload_size = (int32_t)data.size_bytes();
        hdr.src_node = local_node_id_;
        hdr.dst_node = dst_node;
        hdr.expert_id = expert_id;
        hdr.token_count = data.numel();

        comm.send(dst_node, &hdr, sizeof(hdr));
        if (data.numel() > 0)
            comm.send(dst_node, data.data(), data.size_bytes());
    }

    Tensor recv_expert_data(int64_t src_node, int64_t* out_expert_id) {
        EPMessageHeader hdr = {};
        int64_t actual = 0;
        if (!comm.recv(src_node, &hdr, sizeof(hdr), &actual) || actual < (int64_t)sizeof(hdr))
            return Tensor({0});

        if (out_expert_id) *out_expert_id = hdr.expert_id;

        if (hdr.payload_size <= 0) return Tensor({0});
        Tensor data({(int64_t)(hdr.payload_size / (int64_t)sizeof(float))});
        comm.recv(src_node, data.data(), hdr.payload_size, nullptr);
        return data;
    }
};

ExpertParallelManager::ExpertParallelManager(const ClusterConfig& cfg)
    : impl_(new Impl(cfg)) {}

ExpertParallelManager::~ExpertParallelManager() { delete impl_; }

bool ExpertParallelManager::init_cluster(int64_t local_node_id, int64_t port) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->local_node_id_ = local_node_id;

    bool ok = impl_->comm.listen(port);
    if (!ok) return false;

    impl_->dispatcher.assign_experts_to_nodes(1);

    ClusterNode self;
    self.node_id = local_node_id;
    self.address = "127.0.0.1";
    self.port = port;
    self.alive = true;
    self.num_experts = impl_->config.num_experts;
    impl_->nodes_.push_back(self);

    impl_->active = true;
    return true;
}

bool ExpertParallelManager::join_cluster(const std::string& coordinator_addr,
                                          int64_t coordinator_port) {
    if (!impl_->comm.connect(coordinator_addr, coordinator_port)) return false;

    EPMessageHeader hdr = {};
    hdr.msg_type = EP_MSG_NODE_JOIN;
    hdr.src_node = impl_->local_node_id_;
    hdr.dst_node = -1;

    int64_t conn_id = -1;
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        conn_id = (int64_t)(uintptr_t)1;
    }
    impl_->comm.send(conn_id, &hdr, sizeof(hdr));
    return true;
}

void ExpertParallelManager::leave_cluster() {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    EPMessageHeader hdr = {};
    hdr.msg_type = EP_MSG_NODE_LEAVE;
    hdr.src_node = impl_->local_node_id_;
    for (auto& node : impl_->nodes_) {
        if (node.node_id != impl_->local_node_id_ && node.alive) {
            impl_->comm.send(node.node_id, &hdr, sizeof(hdr));
        }
    }
    impl_->active = false;
    impl_->comm.shutdown();
}

bool ExpertParallelManager::is_cluster_active() const { return impl_->active; }
int64_t ExpertParallelManager::local_node_id() const { return impl_->local_node_id_; }

int64_t ExpertParallelManager::num_nodes() const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    int64_t count = 0;
    for (auto& n : impl_->nodes_)
        if (n.alive) count++;
    return count;
}

std::vector<ClusterNode> ExpertParallelManager::get_nodes() const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return impl_->nodes_;
}

MoEOutput ExpertParallelManager::forward_expert_parallel(const Tensor& x,
                                                          std::vector<moe::ExpertFFN>& local_experts,
                                                          bool training) {
    (void)training;
    auto start = std::chrono::high_resolution_clock::now();
    MoEOutput output;

    int64_t T = x.dim(0);
    int64_t D = impl_->config.hidden_size;
    int64_t K = impl_->config.top_k;
    int64_t E = impl_->config.num_experts;
    int64_t num_nds = num_nodes();

    if (T == 0 || D == 0) return output;

    Tensor x_flat = x.reshape({T, D});
    Tensor logits = impl_->router_weight.forward(x_flat);

    RoutingDecision routing = impl_->dispatcher.route_tokens(logits, T,
                                                              impl_->local_node_id_, num_nds);

    Tensor result({T, D});
    result.zero_();

    const float* xd = x.data<float>();
    const float* idx_d = routing.expert_indices.data<float>();
    const float* wt_d = routing.expert_weights.data<float>();
    float* rd = result.data<float>();

    int64_t local_expert_count = 0;
    for (auto& a : impl_->dispatcher.get_all_assignments())
        if (a.node_id == impl_->local_node_id_) local_expert_count++;

    for (int64_t t = 0; t < T; t++) {
        for (int64_t k = 0; k < K; k++) {
            int64_t expert_id = (int64_t)idx_d[t * K + k];
            float weight = wt_d[t * K + k];

            bool is_local = false;
            auto assignments = impl_->dispatcher.get_all_assignments();
            if (expert_id >= 0 && expert_id < (int64_t)assignments.size())
                is_local = (assignments[(size_t)expert_id].node_id == impl_->local_node_id_);

            if (is_local && expert_id < (int64_t)local_experts.size()) {
                Tensor token_in({1, D});
                std::memcpy(token_in.data<float>(), xd + t * D, D * sizeof(float));
                Tensor expert_out = local_experts[(size_t)expert_id].forward(token_in);
                const float* eo = expert_out.data<float>();
                int64_t out_d = expert_out.numel();
                for (int64_t d = 0; d < D && d < out_d; d++)
                    rd[t * D + d] += weight * eo[d];
            }
        }
    }

    output.output = result;
    output.router_logits = logits;
    output.expert_weights = routing.expert_weights;
    output.expert_indices = routing.expert_indices;
    output.load_balance_loss = routing.load_balance_loss;
    output.z_loss = routing.z_loss;
    output.tokens_dropped = routing.tokens_dropped;

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        impl_->stats.total_tokens_routed += T;
        impl_->stats.total_tokens_dropped += routing.tokens_dropped;
        impl_->stats.total_expert_forward_calls += local_expert_count;
        impl_->stats.total_forward_time_ms += ms;
    }

    return output;
}

Tensor ExpertParallelManager::expert_parallel_mlp(const Tensor& x,
                                                    std::vector<moe::ExpertFFN>& local_experts) {
    int64_t T = x.dim(0);
    int64_t D = impl_->config.hidden_size;
    if (T == 0 || D == 0) return Tensor({0, D});
    return x.clone();
}

void ExpertParallelManager::sync_gradients(std::vector<moe::ExpertFFN>& local_experts) {
    for (auto& expert : local_experts) {
        Tensor& gw = expert.gate_proj.weight;
        if (gw.numel() > 0) {
            all_reduce_gradients(gw);
        }
    }
}

void ExpertParallelManager::all_reduce_gradients(Tensor& grad) {
    if (grad.numel() == 0) return;

    int64_t n = grad.numel();
    float* gd = grad.data<float>();

    std::vector<float> recv_buf((size_t)n, 0.0f);
    int64_t num_nds = num_nodes();
    if (num_nds <= 1) return;

    for (auto& node : impl_->nodes_) {
        if (node.node_id == impl_->local_node_id_ || !node.alive) continue;
        impl_->comm.send(node.node_id, gd, n * (int64_t)sizeof(float));
    }

    for (auto& node : impl_->nodes_) {
        if (node.node_id == impl_->local_node_id_ || !node.alive) continue;
        std::vector<float> peer_data((size_t)n, 0.0f);
        int64_t got = 0;
        if (impl_->comm.recv(node.node_id, peer_data.data(),
                             n * (int64_t)sizeof(float), &got)) {
            int64_t count = got / (int64_t)sizeof(float);
            for (int64_t i = 0; i < count && i < n; i++)
                recv_buf[(size_t)i] += peer_data[(size_t)i];
        }
    }

    for (int64_t i = 0; i < n; i++)
        gd[i] = (gd[i] + recv_buf[(size_t)i]) / (float)num_nds;
}

void ExpertParallelManager::add_node(const std::string& address, int64_t port) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->comm.connect(address, port);

    ClusterNode node;
    node.node_id = (int64_t)impl_->nodes_.size();
    node.address = address;
    node.port = port;
    node.alive = true;
    node.num_experts = impl_->config.num_experts / ((int64_t)impl_->nodes_.size() + 1);
    impl_->nodes_.push_back(node);

    impl_->dispatcher.assign_experts_to_nodes((int64_t)impl_->nodes_.size());
}

void ExpertParallelManager::remove_node(int64_t node_id) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    for (auto& n : impl_->nodes_) {
        if (n.node_id == node_id) {
            n.alive = false;
            break;
        }
    }
    impl_->comm.disconnect(node_id);
    impl_->dispatcher.assign_experts_to_nodes(num_nodes());
}

void ExpertParallelManager::handle_node_failure(int64_t node_id) {
    redistribute_experts_after_failure(node_id);
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        impl_->stats.node_failures_handled++;
    }
}

void ExpertParallelManager::rebalance_experts() {
    impl_->dispatcher.assign_experts_to_nodes(num_nodes());
}

void ExpertParallelManager::redistribute_experts_after_failure(int64_t failed_node) {
    auto assignments = impl_->dispatcher.get_all_assignments();
    int64_t survivor_idx = 0;
    for (auto& a : assignments) {
        if (a.node_id == failed_node) {
            int64_t target_node = -1;
            for (auto& n : impl_->nodes_) {
                if (n.node_id != failed_node && n.alive) {
                    target_node = n.node_id;
                    break;
                }
            }
            if (target_node < 0) target_node = survivor_idx;
            impl_->dispatcher.redistribute_expert(a.expert_id, failed_node, target_node);
            survivor_idx = (survivor_idx + 1) % std::max((int64_t)1, num_nodes());
        }
    }
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        impl_->stats.expert_redistributions++;
    }
}

bool ExpertParallelManager::checkpoint_experts(const std::vector<moe::ExpertFFN>& experts,
                                                const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) return false;

    // Write header: number of experts
    int64_t num_experts = (int64_t)experts.size();
    ofs.write(reinterpret_cast<const char*>(&num_experts), sizeof(int64_t));

    // Write each expert's parameters (gate_proj, up_proj, down_proj)
    for (int64_t i = 0; i < num_experts; i++) {
        const auto& expert = experts[(size_t)i];

        // gate_proj weights
        int64_t gn = expert.gate_proj.weight.numel();
        ofs.write(reinterpret_cast<const char*>(&gn), sizeof(int64_t));
        if (gn > 0)
            ofs.write(reinterpret_cast<const char*>(expert.gate_proj.weight.data<float>()),
                      gn * sizeof(float));

        // up_proj weights
        int64_t un = expert.up_proj.weight.numel();
        ofs.write(reinterpret_cast<const char*>(&un), sizeof(int64_t));
        if (un > 0)
            ofs.write(reinterpret_cast<const char*>(expert.up_proj.weight.data<float>()),
                      un * sizeof(float));

        // down_proj weights
        int64_t dn = expert.down_proj.weight.numel();
        ofs.write(reinterpret_cast<const char*>(&dn), sizeof(int64_t));
        if (dn > 0)
            ofs.write(reinterpret_cast<const char*>(expert.down_proj.weight.data<float>()),
                      dn * sizeof(float));
    }
    ofs.flush();
    return ofs.good();
}

bool ExpertParallelManager::restore_experts(std::vector<moe::ExpertFFN>& experts,
                                             const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false;

    int64_t num_experts = 0;
    ifs.read(reinterpret_cast<char*>(&num_experts), sizeof(int64_t));
    if (num_experts <= 0 || num_experts > 10000) return false;

    experts.resize((size_t)num_experts);
    for (int64_t i = 0; i < num_experts; i++) {
        auto& expert = experts[(size_t)i];

        // gate_proj weights
        int64_t gn = 0;
        ifs.read(reinterpret_cast<char*>(&gn), sizeof(int64_t));
        if (gn > 0 && gn == expert.gate_proj.weight.numel()) {
            ifs.read(reinterpret_cast<char*>(expert.gate_proj.weight.data<float>()),
                     gn * sizeof(float));
        } else if (gn > 0) {
            ifs.seekg(gn * sizeof(float), std::ios::cur);
        }

        // up_proj weights
        int64_t un = 0;
        ifs.read(reinterpret_cast<char*>(&un), sizeof(int64_t));
        if (un > 0 && un == expert.up_proj.weight.numel()) {
            ifs.read(reinterpret_cast<char*>(expert.up_proj.weight.data<float>()),
                     un * sizeof(float));
        } else if (un > 0) {
            ifs.seekg(un * sizeof(float), std::ios::cur);
        }

        // down_proj weights
        int64_t dn = 0;
        ifs.read(reinterpret_cast<char*>(&dn), sizeof(int64_t));
        if (dn > 0 && dn == expert.down_proj.weight.numel()) {
            ifs.read(reinterpret_cast<char*>(expert.down_proj.weight.data<float>()),
                     dn * sizeof(float));
        } else if (dn > 0) {
            ifs.seekg(dn * sizeof(float), std::ios::cur);
        }
    }
    return ifs.good();
}

ExpertParallelStats ExpertParallelManager::get_stats() const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return impl_->stats;
}

void ExpertParallelManager::reset_stats() {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->stats = {};
}

void ExpertParallelManager::restore_nodes(const std::vector<ClusterNode>& nodes) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->nodes_ = nodes;
    impl_->dispatcher.assign_experts_to_nodes((int64_t)nodes.size());
}

void ExpertParallelManager::restore_stats(const ExpertParallelStats& stats) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->stats = stats;
}

PipelineScheduler::PipelineScheduler(int64_t num_stages, int64_t micro_batch_size)
    : num_stages(num_stages), micro_batch_size(micro_batch_size) {
    stages_.resize((size_t)num_stages);
    for (int64_t i = 0; i < num_stages; i++) {
        stages_[(size_t)i].stage_id = i;
        stages_[(size_t)i].ready = false;
    }
}

PipelineScheduler::~PipelineScheduler() {}

void PipelineScheduler::schedule_forward(const Tensor& input, int64_t stage_id) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (stage_id >= 0 && stage_id < num_stages) {
        stages_[(size_t)stage_id].input_buffer = input;
        stages_[(size_t)stage_id].ready = true;
    }
}

void PipelineScheduler::schedule_backward(const Tensor& grad_output, int64_t stage_id) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (stage_id >= 0 && stage_id < num_stages) {
        stages_[(size_t)stage_id].output_buffer = grad_output;
    }
}

void PipelineScheduler::execute_pipeline(
    std::vector<ExpertParallelManager*>& managers,
    std::vector<std::vector<moe::ExpertFFN>*>& all_experts) {

    if (managers.empty() || all_experts.empty()) return;

    for (int64_t stage = 0; stage < num_stages; stage++) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto& s = stages_[(size_t)stage];
        if (!s.ready || s.input_buffer.numel() == 0) continue;

        if (stage < (int64_t)managers.size() && stage < (int64_t)all_experts.size()) {
            if (managers[(size_t)stage] && all_experts[(size_t)stage]) {
                MoEOutput moe_out = managers[(size_t)stage]->forward_expert_parallel(
                    s.input_buffer, *all_experts[(size_t)stage]);
                s.output_buffer = moe_out.output;
                s.ready = false;

                if (stage + 1 < num_stages) {
                    stages_[(size_t)(stage + 1)].input_buffer = s.output_buffer;
                    stages_[(size_t)(stage + 1)].ready = true;
                }
            }
        }
    }
    current_step_++;
}

PipelineStage PipelineScheduler::get_stage(int64_t stage_id) const {
    std::lock_guard<std::mutex> lk(mtx_);
    if (stage_id >= 0 && stage_id < num_stages)
        return stages_[(size_t)stage_id];
    return {};
}

bool PipelineScheduler::all_stages_done() const {
    std::lock_guard<std::mutex> lk(mtx_);
    for (auto& s : stages_)
        if (s.ready) return false;
    return true;
}

void PipelineScheduler::reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    for (auto& s : stages_) {
        s.ready = false;
        s.input_buffer = Tensor({0});
        s.output_buffer = Tensor({0});
    }
    current_step_ = 0;
}

FaultToleranceManager::FaultToleranceManager(ExpertParallelManager* manager)
    : manager_(manager) {}

FaultToleranceManager::~FaultToleranceManager() { stop_heartbeat(); }

void FaultToleranceManager::start_heartbeat(int64_t interval_ms) {
    running_ = true;
    std::thread([this, interval_ms]() {
        while (running_.load()) {
            for (auto& node : manager_->get_nodes()) {
                if (node.node_id != manager_->local_node_id() && node.alive) {
                    bool healthy = check_node_health(node.node_id);
                    if (!healthy) {
                        on_node_failure(node.node_id);
                    }
                }
            }
            // Portable sleep: Win32 Sleep(ms) vs POSIX thread sleep.
#if defined(_WIN32)
            Sleep((DWORD)interval_ms);
#else
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
#endif
        }
    }).detach();
}

void FaultToleranceManager::stop_heartbeat() { running_ = false; }

bool FaultToleranceManager::check_node_health(int64_t node_id) const {
    return manager_->get_nodes().size() > 0;
}

void FaultToleranceManager::on_node_failure(int64_t node_id) {
    std::lock_guard<std::mutex> lk(mtx_);
    bool already_failed = false;
    for (int64_t fn : failed_nodes_)
        if (fn == node_id) { already_failed = true; break; }
    if (!already_failed) {
        failed_nodes_.push_back(node_id);
        manager_->handle_node_failure(node_id);
    }
}

void FaultToleranceManager::on_node_recovery(int64_t node_id) {
    std::lock_guard<std::mutex> lk(mtx_);
    failed_nodes_.erase(
        std::remove(failed_nodes_.begin(), failed_nodes_.end(), node_id),
        failed_nodes_.end());
    manager_->add_node("127.0.0.1", 0);
}

std::vector<int64_t> FaultToleranceManager::get_failed_nodes() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return failed_nodes_;
}

void FaultToleranceManager::checkpoint_state(const std::string& path) {
    if (path.empty()) return;
    auto nodes = manager_->get_nodes();
    auto stats = manager_->get_stats();
    std::string state_data;
    state_data.reserve(4096);
    state_data += "nodes:" + std::to_string(nodes.size()) + ";";
    state_data += "failed:" + std::to_string(failed_nodes_.size()) + ";";
    for (int64_t fn : failed_nodes_)
        state_data += "failed_node{" + std::to_string(fn) + "};";
    for (auto& n : nodes) {
        state_data += "node{" + std::to_string(n.node_id) + "," +
                      n.address + "," + std::to_string(n.port) + "," +
                      std::to_string(n.num_experts) + "};";
    }
    state_data += "stats{routed=" + std::to_string(stats.total_tokens_routed) +
                  ",dropped=" + std::to_string(stats.total_tokens_dropped) +
                  ",failures=" + std::to_string(stats.node_failures_handled) + "}";
    FILE* f = nullptr;
#if defined(_WIN32)
    fopen_s(&f, path.c_str(), "wb");
#else
    f = std::fopen(path.c_str(), "wb");
#endif
    if (f) {
        uint64_t sz = state_data.size();
        fwrite(&sz, sizeof(sz), 1, f);
        fwrite(state_data.data(), 1, state_data.size(), f);
        fclose(f);
    }
}

void FaultToleranceManager::restore_state(const std::string& path) {
    if (path.empty()) return;
    FILE* f = nullptr;
#if defined(_WIN32)
    fopen_s(&f, path.c_str(), "rb");
#else
    f = std::fopen(path.c_str(), "rb");
#endif
    if (!f) return;
    uint64_t sz = 0;
    if (fread(&sz, sizeof(sz), 1, f) != 1 || sz > 1024 * 1024) { fclose(f); return; }
    std::string data((size_t)sz, '\0');
    if (fread(&data[0], 1, (size_t)sz, f) != sz) { fclose(f); return; }
    fclose(f);

    std::lock_guard<std::mutex> lk(mtx_);
    failed_nodes_.clear();

    // Mirror the format written by checkpoint_state():
    //   nodes:N; failed:N; failed_node{id};* node{id,address,port,num_experts};*
    //   stats{routed=..,dropped=..,failures=..}
    std::vector<ClusterNode> restored_nodes;
    ExpertParallelStats restored_stats;
    bool have_stats = false;
    int64_t failed_count = 0;

    size_t pos = 0;
    while (pos < data.size()) {
        size_t semi = data.find(';', pos);
        std::string token = (semi == std::string::npos)
                                ? data.substr(pos)
                                : data.substr(pos, semi - pos);
        pos = (semi == std::string::npos) ? data.size() : semi + 1;

        if (token.rfind("failed:", 0) == 0) {
            // Total failed count; the actual ids are restored from the
            // per-node "failed_node{id}" records below.
            failed_count = std::stoll(token.substr(7));
            (void)failed_count;
        } else if (token.rfind("failed_node{", 0) == 0) {
            size_t brace = token.find('}');
            if (brace != std::string::npos)
                failed_nodes_.push_back(std::stoll(token.substr(12, brace - 12)));
        } else if (token.rfind("node{", 0) == 0) {
            size_t brace = token.find('}');
            if (brace != std::string::npos) {
                std::string inner = token.substr(5, brace - 5);
                size_t c1 = inner.find(',');
                size_t c2 = c1 == std::string::npos ? std::string::npos
                                                    : inner.find(',', c1 + 1);
                size_t c3 = c2 == std::string::npos ? std::string::npos
                                                    : inner.find(',', c2 + 1);
                if (c1 != std::string::npos && c2 != std::string::npos &&
                    c3 != std::string::npos) {
                    ClusterNode n;
                    n.node_id = std::stoll(inner.substr(0, c1));
                    n.address = inner.substr(c1 + 1, c2 - c1 - 1);
                    n.port = std::stoll(inner.substr(c2 + 1, c3 - c2 - 1));
                    n.num_experts = std::stoll(inner.substr(c3 + 1));
                    n.alive = true;
                    restored_nodes.push_back(n);
                }
            }
        } else if (token.rfind("stats{", 0) == 0) {
            size_t brace = token.find('}');
            if (brace != std::string::npos) {
                std::string inner = token.substr(6, brace - 6);
                auto parse_kv = [&](const std::string& key) -> int64_t {
                    std::string key_eq = key + "=";
                    size_t ks = inner.find(key_eq);
                    if (ks == std::string::npos) return 0;
                    size_t vs = ks + key_eq.size();
                    size_t ve = inner.find(',', vs);
                    std::string val = (ve == std::string::npos)
                                          ? inner.substr(vs)
                                          : inner.substr(vs, ve - vs);
                    return val.empty() ? 0 : std::stoll(val);
                };
                restored_stats.total_tokens_routed = parse_kv("routed");
                restored_stats.total_tokens_dropped = parse_kv("dropped");
                restored_stats.node_failures_handled = parse_kv("failures");
                have_stats = true;
            }
        }
        // "nodes:N" is informational; the node records carry the data.
    }

    // Write the parsed node list and routing stats back into the manager so
    // the checkpoint round-trips (previously the node/stats records were
    // ignored and failed_nodes_ was left cleared).
    if (!restored_nodes.empty()) manager_->restore_nodes(restored_nodes);
    if (have_stats) manager_->restore_stats(restored_stats);
}

// ========================================================================
// Ring AllReduce for gradient synchronization
// ========================================================================

static void ring_allreduce(float* data, int64_t count, int64_t world_size,
                           int64_t rank, TCPCommBackend& comm) {
    if (world_size <= 1) return;

    int64_t chunk_size = (count + world_size - 1) / world_size;
    std::vector<float> recv_buf((size_t)chunk_size, 0.0f);

    for (int64_t step = 0; step < world_size - 1; step++) {
        int64_t send_rank = (rank - step + world_size) % world_size;
        int64_t recv_rank = (rank - step - 1 + world_size) % world_size;
        int64_t chunk_idx = (rank - step + world_size) % world_size;
        int64_t offset = chunk_idx * chunk_size;
        int64_t send_count = std::min(chunk_size, count - offset);

        if (send_count > 0) {
            comm.send(recv_rank, data + offset, (int64_t)(send_count * sizeof(float)));
            comm.recv(send_rank, recv_buf.data(), (int64_t)(send_count * sizeof(float)), nullptr);

            for (int64_t i = 0; i < send_count && offset + i < count; i++)
                data[offset + i] += recv_buf[(size_t)i];
        }
    }

    for (int64_t i = 0; i < count; i++)
        data[i] /= (float)world_size;

    for (int64_t step = 0; step < world_size - 1; step++) {
        int64_t send_rank = (rank + step + 1) % world_size;
        int64_t recv_rank = (rank + step) % world_size;
        int64_t chunk_idx = (rank + step + 1) % world_size;
        int64_t offset = chunk_idx * chunk_size;
        int64_t send_count = std::min(chunk_size, count - offset);

        if (send_count > 0) {
            comm.send(recv_rank, data + offset, (int64_t)(send_count * sizeof(float)));
            comm.recv(send_rank, recv_buf.data(), (int64_t)(send_count * sizeof(float)), nullptr);
            std::memcpy(data + offset, recv_buf.data(), (size_t)(send_count * sizeof(float)));
        }
    }
}

// ========================================================================
// Expert weight serialization helpers
// ========================================================================

static std::vector<uint8_t> serialize_expert_weights(const moe::ExpertFFN& expert) {
    std::vector<uint8_t> data;
    auto append_tensor = [&data](const Tensor& t) {
        int32_t rank = t.rank();
        data.insert(data.end(), (uint8_t*)&rank, (uint8_t*)&rank + sizeof(rank));
        for (int i = 0; i < rank; i++) {
            int64_t d = t.dim(i);
            data.insert(data.end(), (uint8_t*)&d, (uint8_t*)&d + sizeof(d));
        }
        int64_t n = t.numel();
        data.insert(data.end(), (uint8_t*)&n, (uint8_t*)&n + sizeof(n));
        if (n > 0 && t.data()) {
            const uint8_t* ptr = (const uint8_t*)t.data();
            data.insert(data.end(), ptr, ptr + (size_t)(n * sizeof(float)));
        }
    };
    append_tensor(expert.gate_proj.weight);
    append_tensor(expert.up_proj.weight);
    append_tensor(expert.down_proj.weight);
    return data;
}

static bool deserialize_expert_weights(moe::ExpertFFN& expert, const uint8_t* data, size_t size) {
    size_t pos = 0;
    auto read_tensor = [&](Tensor& t) -> bool {
        if (pos + sizeof(int32_t) > size) return false;
        int32_t rank;
        std::memcpy(&rank, data + pos, sizeof(rank));
        pos += sizeof(rank);
        std::vector<int64_t> dims((size_t)rank);
        for (int i = 0; i < rank; i++) {
            if (pos + sizeof(int64_t) > size) return false;
            std::memcpy(&dims[(size_t)i], data + pos, sizeof(int64_t));
            pos += sizeof(int64_t);
        }
        if (pos + sizeof(int64_t) > size) return false;
        int64_t n;
        std::memcpy(&n, data + pos, sizeof(n));
        pos += sizeof(n);
        Shape s;
        s.rank = rank;
        for (int i = 0; i < rank && i < 8; i++) s.dims[i] = dims[(size_t)i];
        t = Tensor(s);
        if (n > 0 && t.data()) {
            if (pos + (size_t)(n * sizeof(float)) > size) return false;
            std::memcpy(t.data(), data + pos, (size_t)(n * sizeof(float)));
            pos += (size_t)(n * sizeof(float));
        }
        return true;
    };
    if (!read_tensor(expert.gate_proj.weight)) return false;
    if (!read_tensor(expert.up_proj.weight)) return false;
    if (!read_tensor(expert.down_proj.weight)) return false;
    return true;
}

// ========================================================================
// Load balancing metrics and analysis
// ========================================================================

struct LoadBalanceMetrics {
    float coefficient_of_variation = 0.0f;
    float max_imbalance_ratio = 0.0f;
    float gini_coefficient = 0.0f;
    float entropy = 0.0f;
    std::vector<float> expert_loads;
    std::vector<float> ideal_loads;
};

static LoadBalanceMetrics compute_load_balance_metrics(const std::vector<int64_t>& tokens_per_expert,
                                                        int64_t total_tokens) {
    LoadBalanceMetrics metrics;
    int64_t E = (int64_t)tokens_per_expert.size();
    if (E == 0 || total_tokens == 0) return metrics;

    float ideal_load = (float)total_tokens / (float)E;
    metrics.expert_loads.resize((size_t)E);
    metrics.ideal_loads.resize((size_t)E, ideal_load);

    float mean = ideal_load;
    float variance = 0;
    float max_load = 0;
    float min_load = (float)total_tokens;

    for (int64_t e = 0; e < E; e++) {
        float load = (float)tokens_per_expert[(size_t)e];
        metrics.expert_loads[(size_t)e] = load;
        if (load > max_load) max_load = load;
        if (load < min_load) min_load = load;
        float diff = load - mean;
        variance += diff * diff;
    }
    variance /= (float)E;
    metrics.coefficient_of_variation = std::sqrt(variance) / (mean + 1e-10f);
    metrics.max_imbalance_ratio = max_load / (min_load + 1e-10f);

    float sum_abs_diff = 0;
    for (int64_t e = 0; e < E; e++)
        sum_abs_diff += std::abs(metrics.expert_loads[(size_t)e] - mean);
    metrics.gini_coefficient = sum_abs_diff / (2.0f * (float)E * mean + 1e-10f);

    metrics.entropy = 0;
    for (int64_t e = 0; e < E; e++) {
        float p = metrics.expert_loads[(size_t)e] / (float)total_tokens;
        if (p > 1e-10f)
            metrics.entropy -= p * std::log(p);
    }
    return metrics;
}

// ========================================================================
// Capacity-aware token routing with overflow handling
// ========================================================================

struct CapacityAwareRouter {
    struct ExpertCapacity {
        int64_t expert_id;
        int64_t capacity;
        int64_t current_count;
        float overflow_ratio;

        bool is_full() const { return current_count >= capacity; }
        float remaining_ratio() const {
            return capacity > 0 ? (float)(capacity - current_count) / (float)capacity : 0.0f;
        }
    };

    std::vector<ExpertCapacity> capacities;
    int64_t total_capacity;
    int64_t overflow_tokens;
    std::vector<int64_t> overflow_mapping;

    CapacityAwareRouter() : total_capacity(0), overflow_tokens(0) {}

    void init(int64_t num_experts, int64_t capacity_factor, int64_t tokens_per_expert) {
        capacities.resize((size_t)num_experts);
        total_capacity = 0;
        overflow_tokens = 0;
        for (int64_t e = 0; e < num_experts; e++) {
            capacities[(size_t)e].expert_id = e;
            capacities[(size_t)e].capacity = (int64_t)(capacity_factor * tokens_per_expert);
            capacities[(size_t)e].current_count = 0;
            capacities[(size_t)e].overflow_ratio = 0.0f;
            total_capacity += capacities[(size_t)e].capacity;
        }
    }

    bool assign_token(int64_t expert_id, int64_t token_idx) {
        if (expert_id < 0 || expert_id >= (int64_t)capacities.size()) return false;
        auto& cap = capacities[(size_t)expert_id];
        if (!cap.is_full()) {
            cap.current_count++;
            return true;
        }
        overflow_tokens++;
        overflow_mapping.push_back(token_idx);
        return false;
    }

    void reset() {
        overflow_tokens = 0;
        overflow_mapping.clear();
        for (auto& c : capacities) {
            c.current_count = 0;
            c.overflow_ratio = 0.0f;
        }
    }

    float compute_overflow_ratio() const {
        int64_t total_assigned = 0;
        for (auto& c : capacities)
            total_assigned += c.current_count;
        return total_capacity > 0 ? (float)overflow_tokens / (float)(total_assigned + overflow_tokens + 1) : 0.0f;
    }
};

// ========================================================================
// Token redistribution after node failure
// ========================================================================

struct RedistributionPlan {
    struct Migration {
        int64_t expert_id;
        int64_t source_node;
        int64_t dest_node;
        int64_t token_count;
        int64_t data_bytes;
    };
    std::vector<Migration> migrations;
    int64_t total_tokens_migrated;
    int64_t total_bytes_transferred;
    std::vector<int64_t> new_expert_counts_per_node;
};

static RedistributionPlan compute_redistribution(
    const std::vector<ExpertAssignment>& assignments,
    int64_t failed_node, int64_t num_nodes,
    int64_t hidden_size) {

    RedistributionPlan plan;
    plan.total_tokens_migrated = 0;
    plan.total_bytes_transferred = 0;
    plan.new_expert_counts_per_node.resize((size_t)num_nodes, 0);

    std::vector<int64_t> survivor_nodes;
    for (int64_t n = 0; n < num_nodes; n++)
        if (n != failed_node)
            survivor_nodes.push_back(n);

    if (survivor_nodes.empty()) return plan;

    int64_t next_survivor = 0;
    for (auto& a : assignments) {
        if (a.node_id == failed_node) {
            RedistributionPlan::Migration mig;
            mig.expert_id = a.expert_id;
            mig.source_node = failed_node;
            mig.dest_node = survivor_nodes[(size_t)(next_survivor % (int64_t)survivor_nodes.size())];
            mig.token_count = a.token_count;
            mig.data_bytes = a.token_count * hidden_size * (int64_t)sizeof(float);
            plan.migrations.push_back(mig);
            plan.total_tokens_migrated += a.token_count;
            plan.total_bytes_transferred += mig.data_bytes;
            plan.new_expert_counts_per_node[(size_t)mig.dest_node] += 1;
            next_survivor++;
        } else {
            if (a.node_id >= 0 && a.node_id < num_nodes)
                plan.new_expert_counts_per_node[(size_t)a.node_id] += 1;
        }
    }
    return plan;
}


} // namespace expert
} // namespace quant
