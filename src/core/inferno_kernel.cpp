#include "quant/inferno_kernel.h"
#include <chrono>
#include <functional>

namespace quant {
namespace inferno_kernel {

static double now_sec() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

static backend::ComputeBackend* pick(backend::BackendType preferred, bool auto_select) {
    if (!auto_select) {
        backend::BackendConfig cfg;
        cfg.type = preferred;
        backend::ComputeBackend* be = backend::ComputeBackend::create(cfg);
        if (!be->is_available()) { delete be; throw std::runtime_error("[REAL-ONLY] inferno_kernel: preferred backend unavailable"); }
        return be;
    }
    backend::BackendConfig cfg = backend::auto_select_backend();
    backend::ComputeBackend* be = backend::ComputeBackend::create(cfg);
    if (!be->is_available()) { delete be; throw std::runtime_error("[REAL-ONLY] inferno_kernel: auto backend unavailable"); }
    return be;
}

RouteResult gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C,
                 backend::BackendType preferred, bool auto_select) {
    backend::ComputeBackend* be = pick(preferred, auto_select);
    double t0 = now_sec();
    be->gemm(alpha, A, B, beta, C);
    be->synchronize();
    double dt = now_sec() - t0;
    int64_t M = A.dim(0), K = A.dim(1), N = B.dim(1);
    int64_t elements = M * N * K;
    double flops = dt > 0 ? (double)elements * 2.0 / dt : 0.0;
    RouteResult r;
    r.backend = be->type();
    r.elements = elements;
    r.gflops = flops / 1e9;
    r.inferno = flops / 2.0 / 1e21;
    delete be;
    return r;
}

RouteResult gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y,
                 backend::BackendType preferred, bool auto_select) {
    backend::ComputeBackend* be = pick(preferred, auto_select);
    double t0 = now_sec();
    be->gemv(alpha, A, x, beta, y);
    be->synchronize();
    double dt = now_sec() - t0;
    int64_t M = A.dim(0), K = A.dim(1);
    int64_t elements = M * K;
    double flops = dt > 0 ? (double)elements * 2.0 / dt : 0.0;
    RouteResult r;
    r.backend = be->type();
    r.elements = elements;
    r.gflops = flops / 1e9;
    r.inferno = flops / 2.0 / 1e21;
    delete be;
    return r;
}

static RouteResult timed_elt(backend::ComputeBackend* be, int64_t elements,
                             const std::function<void()>& fn) {
    double t0 = now_sec();
    fn();
    be->synchronize();
    double dt = now_sec() - t0;
    double flops = dt > 0 ? (double)elements * 1.0 / dt : 0.0;
    RouteResult r;
    r.backend = be->type();
    r.elements = elements;
    r.gflops = flops / 1e9;
    r.inferno = flops / 2.0 / 1e21;
    return r;
}

RouteResult relu(const Tensor& x, Tensor& y,
                 backend::BackendType preferred, bool auto_select) {
    backend::ComputeBackend* be = pick(preferred, auto_select);
    RouteResult r = timed_elt(be, x.numel(), [&]() { be->relu(x, y); });
    delete be;
    return r;
}

RouteResult gelu(const Tensor& x, Tensor& y,
                 backend::BackendType preferred, bool auto_select) {
    backend::ComputeBackend* be = pick(preferred, auto_select);
    RouteResult r = timed_elt(be, x.numel(), [&]() { be->gelu(x, y); });
    delete be;
    return r;
}

RouteResult silu(const Tensor& x, Tensor& y,
                 backend::BackendType preferred, bool auto_select) {
    backend::ComputeBackend* be = pick(preferred, auto_select);
    RouteResult r = timed_elt(be, x.numel(), [&]() { be->silu(x, y); });
    delete be;
    return r;
}

RouteResult add(const Tensor& a, const Tensor& b, Tensor& c,
                backend::BackendType preferred, bool auto_select) {
    backend::ComputeBackend* be = pick(preferred, auto_select);
    RouteResult r = timed_elt(be, c.numel(), [&]() { be->add(a, b, c); });
    delete be;
    return r;
}

RouteResult mul(const Tensor& a, const Tensor& b, Tensor& c,
                backend::BackendType preferred, bool auto_select) {
    backend::ComputeBackend* be = pick(preferred, auto_select);
    RouteResult r = timed_elt(be, c.numel(), [&]() { be->mul(a, b, c); });
    delete be;
    return r;
}

std::vector<BackendScore> sweep_backends(int64_t M, int64_t N, int64_t K) {
    std::vector<BackendScore> out;
    for (int i = 0; i <= (int)backend::BackendType::CPU_ZENDNN; i++) {
        auto t = (backend::BackendType)i;
        BackendScore s;
        s.type = t;
        backend::BackendConfig cfg;
        cfg.type = t;
        backend::ComputeBackend* be = nullptr;
        try {
            be = backend::ComputeBackend::create(cfg);
        } catch (...) { out.push_back(s); continue; }
        s.available = be->is_available();
        if (!s.available) { delete be; out.push_back(s); continue; }
        double g = backend::benchmark_operation(be, "gemm", M, N, K, 2, 5);
        s.gflops = g;
        s.inferno = g * 1e9 / 2.0 / 1e21;
        delete be;
        out.push_back(s);
    }
    return out;
}

} // namespace inferno_kernel
} // namespace quant
