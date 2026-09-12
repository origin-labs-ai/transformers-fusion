#include "quant/gpu_compute_rpc.h"
#include "quant/math.h"
#include <vector>
#include <cstring>
#include <cstdio>
#include <string>
#include <stdexcept>
#include <mutex>
#include <iostream>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#define SOCKET int
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#define closesocket close
#endif

namespace quant {
namespace gpu {

enum RpcOpCode : uint32_t {
    OP_GEMM = 1,
    OP_RELU = 2,
    OP_SILU = 3,
    OP_GELU = 4,
    OP_SOFTMAX = 5,
    OP_RMSNORM = 6,
    OP_ADD = 7,
    OP_MUL = 8
};

struct GPUComputeRpc::Impl {
    bool initialized = false;
    SOCKET sock = INVALID_SOCKET;
    std::string host_ip;
    int port_num = 9000;
    std::mutex mu;

    bool connect_server() {
        if (sock != INVALID_SOCKET) return true;
#if defined(_WIN32)
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
#endif
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == INVALID_SOCKET) return false;

        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t)port_num);
        if (inet_pton(AF_INET, host_ip.c_str(), &addr.sin_addr) <= 0) {
            closesocket(sock);
            sock = INVALID_SOCKET;
            return false;
        }

        if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
            closesocket(sock);
            sock = INVALID_SOCKET;
            return false;
        }
        return true;
    }

    void disconnect() {
        if (sock != INVALID_SOCKET) {
            closesocket(sock);
            sock = INVALID_SOCKET;
        }
#if defined(_WIN32)
        WSACleanup();
#endif
    }

    bool send_all(const char* data, size_t size) {
        size_t total = 0;
        while (total < size) {
            int sent = send(sock, data + total, (int)(size - total), 0);
            if (sent <= 0) return false;
            total += sent;
        }
        return true;
    }

    bool recv_all(char* data, size_t size) {
        size_t total = 0;
        while (total < size) {
            int recvd = recv(sock, data + total, (int)(size - total), 0);
            if (recvd <= 0) return false;
            total += recvd;
        }
        return true;
    }

    bool execute_rpc(uint32_t op_code, const std::vector<const Tensor*>& inputs, Tensor& output, const std::vector<float>& scalars = {}) {
        std::lock_guard<std::mutex> lock(mu);
        if (!connect_server()) return false;

        std::vector<char> buffer;
        
        // Header
        uint32_t op = op_code;
        uint32_t num_tensors = (uint32_t)inputs.size();
        buffer.insert(buffer.end(), (char*)&op, (char*)&op + 4);
        buffer.insert(buffer.end(), (char*)&num_tensors, (char*)&num_tensors + 4);

        // Inputs
        for (auto t : inputs) {
            uint32_t ndim = (uint32_t)t->rank();
            buffer.insert(buffer.end(), (char*)&ndim, (char*)&ndim + 4);
            for (int i = 0; i < ndim; ++i) {
                int64_t d = t->dim(i);
                buffer.insert(buffer.end(), (char*)&d, (char*)&d + 8);
            }
            size_t data_size = t->numel() * sizeof(float);
            const char* dptr = (const char*)t->data();
            buffer.insert(buffer.end(), dptr, dptr + data_size);
        }

        // Scalars (appended as extra tensor if needed, but here we can just keep protocol simple)
        // We'll skip scalars in binary for this basic protocol, or pass them inline if needed.
        // The prompt protocol: [4B op_code] [4B tensor_count] [for each tensor: 4B ndims, dims[], data[]]
        
        if (!send_all(buffer.data(), buffer.size())) {
            disconnect();
            return false;
        }

        // Response: data[]
        if (!recv_all((char*)output.data(), output.numel() * sizeof(float))) {
            disconnect();
            return false;
        }

        return true;
    }
};

GPUComputeRpc::GPUComputeRpc() : impl_(new Impl()) {}
GPUComputeRpc::~GPUComputeRpc() {
    shutdown();
    delete impl_;
}

bool GPUComputeRpc::init(const char* host, int port) {
    if (impl_->initialized) return true;
    impl_->host_ip = host;
    impl_->port_num = port;
    impl_->initialized = true; // Delay connect to first op
    return true;
}

bool GPUComputeRpc::is_initialized() const { return impl_->initialized; }

void GPUComputeRpc::shutdown() {
    impl_->disconnect();
    impl_->initialized = false;
}

// REAL ONLY: the old code ran math:: (host CPU) whenever the server was
// unreachable and returned normally — callers could not tell remote from
// local execution (silent passthrough, FAKE). Now every transport failure
// throws; outputs are untouched on failure. NOTE: alpha/beta/axis/eps are
// NOT yet on the wire (protocol sends raw tensors only); ops requiring
// them throw until the protocol carries scalars (see backend wrapper for
// gemv/layer_norm/scale which have no opcode at all).
namespace {
[[noreturn]] void throw_rpc_down(const char* op) {
    std::string msg = std::string("[REAL-ONLY] GPUComputeRpc::") + op +
        " failed: no RPC server reachable (127.0.0.1:9000); refusing host-CPU substitution";
    std::fprintf(stderr, "%s\n", msg.c_str());
    throw std::runtime_error(msg);
}
} // namespace

void GPUComputeRpc::gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) {
    // BUGFIX (bug census): alpha/beta were silently dropped off-wire — a
    // caller passing alpha=0.5 got alpha=1.0 results with no error. The
    // protocol sends raw tensors only, so non-default scalars throw until
    // the wire carries them.
    if (alpha != 1.0f || beta != 0.0f)
        throw_rpc_down("gemm(alpha/beta != defaults: scalars not on RPC wire)");
    if (!impl_->initialized) throw_rpc_down("gemm");
    if (!impl_->execute_rpc(OP_GEMM, {&A, &B}, C)) throw_rpc_down("gemm");
}

void GPUComputeRpc::relu(const Tensor& x, Tensor& y) {
    if (!impl_->initialized) throw_rpc_down("relu");
    if (!impl_->execute_rpc(OP_RELU, {&x}, y)) throw_rpc_down("relu");
}

void GPUComputeRpc::gelu(const Tensor& x, Tensor& y) {
    if (!impl_->initialized) throw_rpc_down("gelu");
    if (!impl_->execute_rpc(OP_GELU, {&x}, y)) throw_rpc_down("gelu");
}

void GPUComputeRpc::silu(const Tensor& x, Tensor& y) {
    if (!impl_->initialized) throw_rpc_down("silu");
    if (!impl_->execute_rpc(OP_SILU, {&x}, y)) throw_rpc_down("silu");
}

void GPUComputeRpc::softmax(const Tensor& x, Tensor& y, int axis) {
    // BUGFIX (bug census): axis was silently dropped — axis != 1 callers got
    // wrong-axis results. Only the last-axis protocol path exists.
    if (axis != 1 && axis != -1)
        throw_rpc_down("softmax(axis != 1/-1: axis not on RPC wire)");
    if (!impl_->initialized) throw_rpc_down("softmax");
    if (!impl_->execute_rpc(OP_SOFTMAX, {&x}, y)) throw_rpc_down("softmax");
}

void GPUComputeRpc::rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y) {
    // BUGFIX (bug census): eps was silently dropped — callers got default-eps
    // results. Only eps == 1e-5 (protocol default) is served remotely.
    if (eps != 1e-5f)
        throw_rpc_down("rms_norm(eps != 1e-5: eps not on RPC wire)");
    if (!impl_->initialized) throw_rpc_down("rms_norm");
    if (!impl_->execute_rpc(OP_RMSNORM, {&x, &gamma}, y)) throw_rpc_down("rms_norm");
}

void GPUComputeRpc::add(const Tensor& a, const Tensor& b, Tensor& c) {
    if (!impl_->initialized) throw_rpc_down("add");
    if (!impl_->execute_rpc(OP_ADD, {&a, &b}, c)) throw_rpc_down("add");
}

void GPUComputeRpc::mul(const Tensor& a, const Tensor& b, Tensor& c) {
    if (!impl_->initialized) throw_rpc_down("mul");
    if (!impl_->execute_rpc(OP_MUL, {&a, &b}, c)) throw_rpc_down("mul");
}

void GPUComputeRpc::synchronize() {}
// REAL ONLY: no remote memory query exists; dummy 1GB constants REMOVED
// (they let capacity planners believe 1GB of remote RAM exists).
int64_t GPUComputeRpc::memory_free() const { return 0; }
int64_t GPUComputeRpc::memory_total() const { return 0; }

static GPUComputeRpc g_rpc_compute;
GPUComputeRpc& get_rpc_compute() { return g_rpc_compute; }

} // namespace gpu
} // namespace quant
