#pragma once
#include "quant/backend.h"
#include "quant/inferno.h"
#include "quant/tensor.h"

namespace quant {
namespace inferno_kernel {

// Every GEMM/GEMV routed here: best available backend runs the math,
// FLOPS measured first, Inferno derived. No backend is skipped, no
// silent fallback — unavailable backend throws REAL-ONLY.
struct RouteResult {
    backend::BackendType backend = backend::BackendType::CPU_SCALAR;
    double gflops = 0.0;
    double inferno = 0.0;
    int64_t elements = 0;
};

RouteResult gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C,
                 backend::BackendType preferred = backend::BackendType::CPU_SCALAR,
                 bool auto_select = true);
RouteResult gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y,
                 backend::BackendType preferred = backend::BackendType::CPU_SCALAR,
                 bool auto_select = true);
RouteResult relu(const Tensor& x, Tensor& y,
                 backend::BackendType preferred = backend::BackendType::CPU_SCALAR,
                 bool auto_select = true);
RouteResult gelu(const Tensor& x, Tensor& y,
                 backend::BackendType preferred = backend::BackendType::CPU_SCALAR,
                 bool auto_select = true);
RouteResult silu(const Tensor& x, Tensor& y,
                 backend::BackendType preferred = backend::BackendType::CPU_SCALAR,
                 bool auto_select = true);
RouteResult add(const Tensor& a, const Tensor& b, Tensor& c,
                backend::BackendType preferred = backend::BackendType::CPU_SCALAR,
                bool auto_select = true);
RouteResult mul(const Tensor& a, const Tensor& b, Tensor& c,
                backend::BackendType preferred = backend::BackendType::CPU_SCALAR,
                bool auto_select = true);

// Honest sweep: every BackendType created, is_available checked, gemm benched.
// Unavailable/throwing backends report 0.0, never fake numbers.
struct BackendScore {
    backend::BackendType type = backend::BackendType::CPU_SCALAR;
    bool available = false;
    double gflops = 0.0;
    double inferno = 0.0;
};
std::vector<BackendScore> sweep_backends(int64_t M = 512, int64_t N = 512, int64_t K = 512);

} // namespace inferno_kernel
} // namespace quant
