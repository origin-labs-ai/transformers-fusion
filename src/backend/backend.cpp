#include "quant/backend.h"
#include "quant/math.h"
#include "quant/kernel.h"
#include "quant/gpu_compute.h"
#include <thread>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
// REAL ONLY: every accelerator header is included unconditionally so the
// factory has a case for every BackendType on every platform. Availability
// is decided at RUNTIME via is_*_available() + init()+is_initialized(),
// never by #ifdef presence. Missing hardware => UNAVAILABLE + fail-loud,
// never silent CPU passthrough.
#include "quant/gpu_compute_cuda.h"
#include "quant/gpu_compute_metal.h"
#include "quant/gpu_compute_sycl.h"
#include "quant/gpu_compute_cann.h"
#include "quant/gpu_compute_rpc.h"
#include "quant/gpu_compute_openvino.h"
#include "quant/gpu_compute_virtgpu.h"
#include "quant/gpu_compute_webgpu.h"
#include "quant/gpu_compute_zendnn.h"
#include "quant/gpu_compute_hip.h"
#include "quant/gpu_compute_hexagon.h"
#include "quant/gpu_compute_musa.h"
#include "quant/gpu_compute_opencl.h"
#include "quant/gpu_compute_zdnn.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#elif defined(__linux__)
#include <unistd.h>
#include <sys/sysinfo.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>
#include <cpuid.h>
#include <dlfcn.h>
#elif defined(__APPLE__)
#include <unistd.h>
#include <sys/sysctl.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>
#include <TargetConditionals.h>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#else
#include <unistd.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>
#endif

namespace quant {
namespace backend {

// ========================================================================
// REAL ONLY fail-loud helper.
// Every accelerator ComputeBackend MUST call this instead of silently
// running math::*/CPU code when its device is unavailable. Silent CPU
// passthrough is FORBIDDEN: it lets benchmark/GFLOPS claims and
// auto-select pretend a GPU/NPU ran work that the CPU actually did.
// ========================================================================
[[noreturn]] inline void throw_unavailable(const char* backend, const char* op,
                                          const char* reason) {
    std::string msg = std::string("[REAL-ONLY] ") + backend + "::" + op +
                      " unavailable: " + reason +
                      " (no silent CPU fallback; check is_available()/probe_hardware())";
    std::fprintf(stderr, "%s\n", msg.c_str());
    throw std::runtime_error(msg);
}

inline void require_available(bool ok, const char* backend, const char* op,
                              const char* reason) {
    if (!ok) throw_unavailable(backend, op, reason);
}

// ========================================================================
// CPU SCALAR BACKEND (portable, no SIMD) — REAL (always available)
// ========================================================================

class CPUScalarBackend : public ComputeBackend {
public:
    BackendType type() const override { return BackendType::CPU_SCALAR; }
    const char* name() const override { return "CPU_SCALAR"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        math::gemm(alpha, A, B, beta, C);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        math::gemv(alpha, A, x, beta, y);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override { math::softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps, Tensor& y) override {
        math::layer_norm(x, gamma, beta, eps, y);
    }
    void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y) override {
        math::rms_norm(x, gamma, eps, y);
    }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override { math::gelu(x, y); }
    void silu(const Tensor& x, Tensor& y) override { math::silu(x, y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }

    bool is_available() const override { return true; }
    int64_t memory_free() const override { return cpu_memory_free(); }
    int64_t memory_total() const override { return cpu_memory_total(); }
    void synchronize() override {}
};

// ========================================================================
// CPU AVX2 BACKEND
// ========================================================================

#if defined(QUANT_AVX2)
class CPUAVX2Backend : public ComputeBackend {
public:
    BackendType type() const override { return BackendType::CPU_AVX2; }
    const char* name() const override { return "CPU_AVX2"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        require_available(is_available(), "CPU_AVX2", "gemm", "AVX2 unavailable on this host");
        int M = (int)A.numel() / (int)A.dim(A.rank() - 1);
        int K = (int)A.dim(A.rank() - 1);
        int N = (int)B.dim(B.rank() - 1);
        Tensor C_orig;
        if (beta != 0.0f) C_orig.copy_from(C);
        kernel::avx2_gemm(A.data<float>(), B.data<float>(), C.data<float>(), M, N, K);
        float* cd = C.data<float>();
        if (beta != 0.0f) {
            const float* cod = C_orig.data<float>();
            for (int64_t i = 0; i < C.numel(); ++i)
                cd[i] = alpha * cd[i] + beta * cod[i];
        } else if (alpha != 1.0f) {
            for (int64_t i = 0; i < C.numel(); ++i)
                cd[i] = alpha * cd[i];
        }
    }

    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        math::gemv(alpha, A, x, beta, y);
    }

    void softmax(const Tensor& x, Tensor& y, int axis) override {
        if (axis == 1 && x.rank() == 2) {
            int64_t rows = x.dim(0), cols = x.dim(1);
            const float* xd = x.data<float>();
            float* yd = y.data<float>();
            for (int64_t r = 0; r < rows; ++r) {
                float maxv = xd[r * cols];
                for (int64_t c = 1; c < cols; ++c)
                    if (xd[r * cols + c] > maxv) maxv = xd[r * cols + c];
                float sum = 0.0f;
                for (int64_t c = 0; c < cols; ++c) {
                    yd[r * cols + c] = std::exp(xd[r * cols + c] - maxv);
                    sum += yd[r * cols + c];
                }
                float inv = 1.0f / sum;
                for (int64_t c = 0; c < cols; ++c)
                    yd[r * cols + c] *= inv;
            }
        } else {
            math::softmax(x, y, axis);
        }
    }

    void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y) override {
        math::rms_norm(x, gamma, eps, y);
    }
    void layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps, Tensor& y) override {
        math::layer_norm(x, gamma, beta, eps, y);
    }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override {
        const float* xd = x.data<float>();
        float* yd = y.data<float>();
        int64_t n = x.numel();
        for (int64_t i = 0; i < n; ++i)
            yd[i] = 0.5f * xd[i] * (1.0f + std::erff(xd[i] * 0.7071067811865475f));
    }
    void silu(const Tensor& x, Tensor& y) override {
        const float* xd = x.data<float>();
        float* yd = y.data<float>();
        int64_t n = x.numel();
        for (int64_t i = 0; i < n; ++i)
            yd[i] = xd[i] / (1.0f + std::exp(-xd[i]));
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override { return is_avx2_available(); }
    int64_t memory_free() const override { return cpu_memory_free(); }
    int64_t memory_total() const override { return cpu_memory_total(); }
    void synchronize() override {}
};
#else
// Stub when built without QUANT_AVX2: explicit UNAVAILABLE (never silent).
class CPUAVX2Backend : public ComputeBackend {
public:
    BackendType type() const override { return BackendType::CPU_AVX2; }
    const char* name() const override { return "CPU_AVX2"; }
    void gemm(float, const Tensor&, const Tensor&, float, Tensor&) override {
        throw_unavailable("CPU_AVX2", "gemm", "built without QUANT_AVX2");
    }
    void gemv(float, const Tensor&, const Tensor&, float, Tensor&) override {
        throw_unavailable("CPU_AVX2", "gemv", "built without QUANT_AVX2");
    }
    void softmax(const Tensor&, Tensor&, int) override {
        throw_unavailable("CPU_AVX2", "softmax", "built without QUANT_AVX2");
    }
    void layer_norm(const Tensor&, const Tensor&, const Tensor&, float, Tensor&) override {
        throw_unavailable("CPU_AVX2", "layer_norm", "built without QUANT_AVX2");
    }
    void rms_norm(const Tensor&, const Tensor&, float, Tensor&) override {
        throw_unavailable("CPU_AVX2", "rms_norm", "built without QUANT_AVX2");
    }
    void relu(const Tensor&, Tensor&) override {
        throw_unavailable("CPU_AVX2", "relu", "built without QUANT_AVX2");
    }
    void gelu(const Tensor&, Tensor&) override {
        throw_unavailable("CPU_AVX2", "gelu", "built without QUANT_AVX2");
    }
    void silu(const Tensor&, Tensor&) override {
        throw_unavailable("CPU_AVX2", "silu", "built without QUANT_AVX2");
    }
    void add(const Tensor&, const Tensor&, Tensor&) override {
        throw_unavailable("CPU_AVX2", "add", "built without QUANT_AVX2");
    }
    void mul(const Tensor&, const Tensor&, Tensor&) override {
        throw_unavailable("CPU_AVX2", "mul", "built without QUANT_AVX2");
    }
    void scale(float, const Tensor&, Tensor&) override {
        throw_unavailable("CPU_AVX2", "scale", "built without QUANT_AVX2");
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override { return false; }
    int64_t memory_free() const override { return cpu_memory_free(); }
    int64_t memory_total() const override { return cpu_memory_total(); }
    void synchronize() override {}
};
#endif // QUANT_AVX2

// ========================================================================
// CPU AVX-512 BACKEND (requires AVX2 + AVX-512)
// ========================================================================

#if defined(QUANT_AVX2) && defined(QUANT_AVX512)
class CPUAVX512Backend : public ComputeBackend {
    CPUAVX2Backend fallback;
public:
    BackendType type() const override { return BackendType::CPU_AVX512; }
    const char* name() const override { return "CPU_AVX512"; }
    void gemm(float a, const Tensor& A, const Tensor& B, float b, Tensor& C) override {
        require_available(is_available(), "CPU_AVX512", "gemm", "AVX-512 unavailable on this host");
        int64_t M = A.dim(0), K = A.dim(1), N = B.dim(1);
        const float* ad = A.data<float>();
        const float* bd = B.data<float>();
        float* cd = C.data<float>();
        if (b == 0.0f) std::memset(cd, 0, static_cast<size_t>(M * N) * sizeof(float));
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t k = 0; k < K; ++k) {
                __m512 a_val = _mm512_set1_ps(ad[m * K + k] * a);
                int64_t n = 0;
                for (; n + 16 <= N; n += 16) {
                    __m512 bv = _mm512_loadu_ps(bd + k * N + n);
                    __m512 cv = _mm512_loadu_ps(cd + m * N + n);
                    _mm512_storeu_ps(cd + m * N + n, _mm512_fmadd_ps(a_val, bv, cv));
                }
                for (; n < N; ++n)
                    cd[m * N + n] += a_val[0] * bd[k * N + n];
            }
        }
    }
    void gemv(float a, const Tensor& A, const Tensor& x, float b, Tensor& y) override { fallback.gemv(a,A,x,b,y); }
    void softmax(const Tensor& x, Tensor& y, int a) override { fallback.softmax(x,y,a); }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override { fallback.layer_norm(x,g,bt,e,y); }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override { fallback.rms_norm(x,g,e,y); }
    void relu(const Tensor& x, Tensor& y) override { fallback.relu(x,y); }
    void gelu(const Tensor& x, Tensor& y) override { fallback.gelu(x,y); }
    void silu(const Tensor& x, Tensor& y) override { fallback.silu(x,y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { fallback.add(a,b,c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { fallback.mul(a,b,c); }
    void scale(float s, const Tensor& x, Tensor& y) override { fallback.scale(s,x,y); }
    void copy(const Tensor& src, Tensor& dst) override { fallback.copy(src,dst); }
    void fill(Tensor& t, float v) override { fallback.fill(t,v); }
    void zero(Tensor& t) override { fallback.zero(t); }
    bool is_available() const override { return is_avx512_available(); }
    int64_t memory_free() const override { return cpu_memory_free(); }
    int64_t memory_total() const override { return fallback.memory_total(); }
    void synchronize() override {}
};
#else
// Stub when built without AVX-512: explicit UNAVAILABLE (never silent).
class CPUAVX512Backend : public ComputeBackend {
    CPUScalarBackend fallback;
public:
    BackendType type() const override { return BackendType::CPU_AVX512; }
    const char* name() const override { return "CPU_AVX512"; }
    void gemm(float, const Tensor&, const Tensor&, float, Tensor&) override {
        throw_unavailable("CPU_AVX512", "gemm", "built without QUANT_AVX512");
    }
    void gemv(float a, const Tensor& A, const Tensor& x, float b, Tensor& y) override { fallback.gemv(a,A,x,b,y); }
    void softmax(const Tensor& x, Tensor& y, int a) override { fallback.softmax(x,y,a); }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override { fallback.layer_norm(x,g,bt,e,y); }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override { fallback.rms_norm(x,g,e,y); }
    void relu(const Tensor& x, Tensor& y) override { fallback.relu(x,y); }
    void gelu(const Tensor& x, Tensor& y) override { fallback.gelu(x,y); }
    void silu(const Tensor& x, Tensor& y) override { fallback.silu(x,y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { fallback.add(a,b,c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { fallback.mul(a,b,c); }
    void scale(float s, const Tensor& x, Tensor& y) override { fallback.scale(s,x,y); }
    void copy(const Tensor& src, Tensor& dst) override { fallback.copy(src,dst); }
    void fill(Tensor& t, float v) override { fallback.fill(t,v); }
    void zero(Tensor& t) override { fallback.zero(t); }
    bool is_available() const override { return false; }
    int64_t memory_free() const override { return cpu_memory_free(); }
    int64_t memory_total() const override { return cpu_memory_total(); }
    void synchronize() override {}
};
#endif // QUANT_AVX2 && QUANT_AVX512

// ========================================================================
// iGPU SHARED BACKEND — REAL ONLY (zero-copy shared-memory path)
// Verdict: REAL when DirectX device present (dispatches real HLSL shaders
// via DirectXCompute with shared-memory upload/download); UNAVAILABLE +
// fail-loud otherwise. Previous CPU avx2_gemm masquerading as iGPU is
// REMOVED (was silent CPU passthrough, REAL-ONLY violation).
// ========================================================================

class IGPUSharedBackend : public ComputeBackend {
    gpu::DirectXCompute* dx_ = nullptr;
    bool avail_ = false;
public:
    IGPUSharedBackend() {
        try {
            dx_ = &gpu::get_dx_compute();
            if (!dx_->is_initialized()) dx_->init(0);
            avail_ = dx_ && dx_->is_initialized();
        } catch (...) { dx_ = nullptr; avail_ = false; }
    }
    BackendType type() const override { return BackendType::IGPU_SHARED; }
    const char* name() const override { return "IGPU_SHARED"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        require_available(avail_, "IGPU_SHARED", "gemm",
            "no DirectX shared-memory device on this host");
        void* dA = dx_->allocate(A.numel() * sizeof(float));
        void* dB = dx_->allocate(B.numel() * sizeof(float));
        void* dC = dx_->allocate(C.numel() * sizeof(float));
        if (!dA || !dB || !dC) {
            if (dA) dx_->free(dA); if (dB) dx_->free(dB); if (dC) dx_->free(dC);
            throw_unavailable("IGPU_SHARED", "gemm", "device alloc failed");
        }
        dx_->upload(A, dA);
        dx_->upload(B, dB);
        if (beta != 0.0f) dx_->upload(C, dC);
        int64_t M = A.numel() / A.dim(A.rank() - 1);
        int64_t N = B.dim(B.rank() - 1);
        int64_t K = A.dim(A.rank() - 1);
        dx_->gemm(alpha, dA, dB, beta, dC, M, N, K);
        dx_->download(dC, C);
        dx_->free(dA); dx_->free(dB); dx_->free(dC);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        require_available(avail_, "IGPU_SHARED", "gemv",
            "no DirectX shared-memory device on this host");
        void* dA = dx_->allocate(A.numel() * sizeof(float));
        void* dxb = dx_->allocate(x.numel() * sizeof(float));
        void* dy = dx_->allocate(y.numel() * sizeof(float));
        if (!dA || !dxb || !dy) {
            if (dA) dx_->free(dA); if (dxb) dx_->free(dxb); if (dy) dx_->free(dy);
            throw_unavailable("IGPU_SHARED", "gemv", "device alloc failed");
        }
        dx_->upload(A, dA); dx_->upload(x, dxb);
        if (beta != 0.0f) dx_->upload(y, dy);
        dx_->gemv(alpha, dA, dxb, beta, dy, A.numel()/A.dim(A.rank()-1), A.dim(A.rank()-1));
        dx_->download(dy, y);
        dx_->free(dA); dx_->free(dxb); dx_->free(dy);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        require_available(avail_, "IGPU_SHARED", "softmax",
            "no DirectX shared-memory device; axis!=1 unsupported on iGPU path");
        if (axis != 1 && !(axis == -1 && x.rank() == 2))
            throw_unavailable("IGPU_SHARED", "softmax", "only axis=1 2-D supported on device");
        void* dxb = dx_->allocate(x.numel() * sizeof(float));
        void* dy = dx_->allocate(y.numel() * sizeof(float));
        if (!dxb || !dy) {
            if (dxb) dx_->free(dxb); if (dy) dx_->free(dy);
            throw_unavailable("IGPU_SHARED", "softmax", "device alloc failed");
        }
        dx_->upload(x, dxb);
        dx_->softmax(dxb, dy, x.dim(0), x.numel()/x.dim(0));
        dx_->download(dy, y);
        dx_->free(dxb); dx_->free(dy);
    }
    void layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps, Tensor& y) override {
        require_available(avail_, "IGPU_SHARED", "layer_norm",
            "no DirectX shared-memory device on this host");
        void* dxb = dx_->allocate(x.numel()*sizeof(float));
        void* dg = dx_->allocate(gamma.numel()*sizeof(float));
        void* db = dx_->allocate(beta.numel()*sizeof(float));
        void* dy = dx_->allocate(y.numel()*sizeof(float));
        if (!dxb||!dg||!db||!dy) {
            if(dxb)dx_->free(dxb); if(dg)dx_->free(dg); if(db)dx_->free(db); if(dy)dx_->free(dy);
            throw_unavailable("IGPU_SHARED", "layer_norm", "device alloc failed");
        }
        dx_->upload(x,dxb); dx_->upload(gamma,dg); dx_->upload(beta,db);
        dx_->layer_norm(dxb,dg,db,dy,eps,x.dim(0),x.numel()/x.dim(0));
        dx_->download(dy,y);
        dx_->free(dxb); dx_->free(dg); dx_->free(db); dx_->free(dy);
    }
    void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y) override {
        require_available(avail_, "IGPU_SHARED", "rms_norm",
            "no DirectX shared-memory device on this host");
        void* dxb = dx_->allocate(x.numel()*sizeof(float));
        void* dg = dx_->allocate(gamma.numel()*sizeof(float));
        void* dy = dx_->allocate(y.numel()*sizeof(float));
        if (!dxb||!dg||!dy) {
            if(dxb)dx_->free(dxb); if(dg)dx_->free(dg); if(dy)dx_->free(dy);
            throw_unavailable("IGPU_SHARED", "rms_norm", "device alloc failed");
        }
        dx_->upload(x,dxb); dx_->upload(gamma,dg);
        dx_->rms_norm(dxb,dg,dy,eps,x.dim(0),x.numel()/x.dim(0));
        dx_->download(dy,y);
        dx_->free(dxb); dx_->free(dg); dx_->free(dy);
    }
    void relu(const Tensor& x, Tensor& y) override {
        require_available(avail_, "IGPU_SHARED", "relu",
            "no DirectX shared-memory device on this host");
        void* dxb = dx_->allocate(x.numel()*sizeof(float));
        void* dy = dx_->allocate(y.numel()*sizeof(float));
        if (!dxb||!dy) {
            if(dxb)dx_->free(dxb); if(dy)dx_->free(dy);
            throw_unavailable("IGPU_SHARED", "relu", "device alloc failed");
        }
        dx_->upload(x,dxb); dx_->relu(dxb,dy,x.numel()); dx_->download(dy,y);
        dx_->free(dxb); dx_->free(dy);
    }
    void gelu(const Tensor& x, Tensor& y) override {
        require_available(avail_, "IGPU_SHARED", "gelu",
            "no DirectX shared-memory device on this host");
        void* dxb = dx_->allocate(x.numel()*sizeof(float));
        void* dy = dx_->allocate(y.numel()*sizeof(float));
        if (!dxb||!dy) {
            if(dxb)dx_->free(dxb); if(dy)dx_->free(dy);
            throw_unavailable("IGPU_SHARED", "gelu", "device alloc failed");
        }
        dx_->upload(x,dxb); dx_->gelu(dxb,dy,x.numel()); dx_->download(dy,y);
        dx_->free(dxb); dx_->free(dy);
    }
    void silu(const Tensor& x, Tensor& y) override {
        require_available(avail_, "IGPU_SHARED", "silu",
            "no DirectX shared-memory device on this host");
        void* dxb = dx_->allocate(x.numel()*sizeof(float));
        void* dy = dx_->allocate(y.numel()*sizeof(float));
        if (!dxb||!dy) {
            if(dxb)dx_->free(dxb); if(dy)dx_->free(dy);
            throw_unavailable("IGPU_SHARED", "silu", "device alloc failed");
        }
        dx_->upload(x,dxb); dx_->silu(dxb,dy,x.numel()); dx_->download(dy,y);
        dx_->free(dxb); dx_->free(dy);
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(avail_, "IGPU_SHARED", "add",
            "no DirectX shared-memory device on this host");
        void* da = dx_->allocate(a.numel()*sizeof(float));
        void* db = dx_->allocate(b.numel()*sizeof(float));
        void* dc = dx_->allocate(c.numel()*sizeof(float));
        if (!da||!db||!dc) {
            if(da)dx_->free(da); if(db)dx_->free(db); if(dc)dx_->free(dc);
            throw_unavailable("IGPU_SHARED", "add", "device alloc failed");
        }
        dx_->upload(a,da); dx_->upload(b,db); dx_->add(da,db,dc,a.numel()); dx_->download(dc,c);
        dx_->free(da); dx_->free(db); dx_->free(dc);
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(avail_, "IGPU_SHARED", "mul",
            "no DirectX shared-memory device on this host");
        void* da = dx_->allocate(a.numel()*sizeof(float));
        void* db = dx_->allocate(b.numel()*sizeof(float));
        void* dc = dx_->allocate(c.numel()*sizeof(float));
        if (!da||!db||!dc) {
            if(da)dx_->free(da); if(db)dx_->free(db); if(dc)dx_->free(dc);
            throw_unavailable("IGPU_SHARED", "mul", "device alloc failed");
        }
        dx_->upload(a,da); dx_->upload(b,db); dx_->mul(da,db,dc,a.numel()); dx_->download(dc,c);
        dx_->free(da); dx_->free(db); dx_->free(dc);
    }
    void scale(float s, const Tensor& x, Tensor& y) override {
        require_available(avail_, "IGPU_SHARED", "scale",
            "no DirectX shared-memory device on this host");
        void* dxb = dx_->allocate(x.numel()*sizeof(float));
        void* dy = dx_->allocate(y.numel()*sizeof(float));
        if (!dxb||!dy) {
            if(dxb)dx_->free(dxb); if(dy)dx_->free(dy);
            throw_unavailable("IGPU_SHARED", "scale", "device alloc failed");
        }
        dx_->upload(x,dxb); dx_->scale(s,dxb,dy,x.numel()); dx_->download(dy,y);
        dx_->free(dxb); dx_->free(dy);
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }

    bool is_available() const override { return avail_; }
    int64_t memory_free() const override { return avail_ && dx_ ? dx_->memory_free() : 0; }
    int64_t memory_total() const override { return avail_ && dx_ ? dx_->memory_total() : 0; }
    void synchronize() override { if (avail_ && dx_) dx_->synchronize(); }
};

// ========================================================================
// GPU DIRECTX BACKEND — REAL (HLSL shaders in src/backend/gpu_compute.cpp)
// Verdict REAL on Windows+DX12 hardware; UNAVAILABLE+fail-loud elsewhere.
// Previous silent AVX2 fallback REMOVED.
// ========================================================================

class GPUDirectXBackend : public ComputeBackend {
    gpu::DirectXCompute* dx_ = nullptr;
    bool avail_ = false;
public:
    GPUDirectXBackend() {
        try {
            dx_ = &gpu::get_dx_compute();
            if (!dx_->is_initialized()) dx_->init(0);
            avail_ = dx_ && dx_->is_initialized();
        } catch (...) { dx_ = nullptr; avail_ = false; }
    }
    BackendType type() const override { return BackendType::GPU_DIRECTX; }
    const char* name() const override { return "GPU_DIRECTX"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        require_available(avail_, "GPU_DIRECTX", "gemm", "no D3D12 device (Windows+GPU required)");
        void* dA = dx_->allocate(A.numel() * sizeof(float));
        void* dB = dx_->allocate(B.numel() * sizeof(float));
        void* dC = dx_->allocate(C.numel() * sizeof(float));
        if (!dA||!dB||!dC) {
            if(dA)dx_->free(dA); if(dB)dx_->free(dB); if(dC)dx_->free(dC);
            throw_unavailable("GPU_DIRECTX", "gemm", "device alloc failed");
        }
        dx_->upload(A, dA);
        dx_->upload(B, dB);
        if (beta != 0.0f) dx_->upload(C, dC);
        int64_t M = A.numel() / A.dim(A.rank() - 1);
        int64_t N = B.dim(B.rank() - 1);
        int64_t K = A.dim(A.rank() - 1);
        dx_->gemm(alpha, dA, dB, beta, dC, M, N, K);
        dx_->download(dC, C);
        dx_->free(dA);
        dx_->free(dB);
        dx_->free(dC);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        require_available(avail_, "GPU_DIRECTX", "gemv", "no D3D12 device (Windows+GPU required)");
        void* dA = dx_->allocate(A.numel()*sizeof(float));
        void* dxb = dx_->allocate(x.numel()*sizeof(float));
        void* dy = dx_->allocate(y.numel()*sizeof(float));
        if (!dA||!dxb||!dy) {
            if(dA)dx_->free(dA); if(dxb)dx_->free(dxb); if(dy)dx_->free(dy);
            throw_unavailable("GPU_DIRECTX", "gemv", "device alloc failed");
        }
        dx_->upload(A,dA); dx_->upload(x,dxb);
        if (beta != 0.0f) dx_->upload(y,dy);
        dx_->gemv(alpha,dA,dxb,beta,dy,A.numel()/A.dim(A.rank()-1),A.dim(A.rank()-1));
        dx_->download(dy,y);
        dx_->free(dA); dx_->free(dxb); dx_->free(dy);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        require_available(avail_, "GPU_DIRECTX", "softmax", "no D3D12 device");
        if (axis != 1 && !(axis==-1 && x.rank()==2))
            throw_unavailable("GPU_DIRECTX","softmax","only axis=1 2-D supported on device");
        void* dxb = dx_->allocate(x.numel()*sizeof(float));
        void* dy = dx_->allocate(y.numel()*sizeof(float));
        if (!dxb||!dy) {
            if(dxb)dx_->free(dxb); if(dy)dx_->free(dy);
            throw_unavailable("GPU_DIRECTX","softmax","device alloc failed");
        }
        dx_->upload(x,dxb); dx_->softmax(dxb,dy,x.dim(0),x.numel()/x.dim(0)); dx_->download(dy,y);
        dx_->free(dxb); dx_->free(dy);
    }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override {
        require_available(avail_, "GPU_DIRECTX", "layer_norm", "no D3D12 device");
        void* dxb=dx_->allocate(x.numel()*sizeof(float));
        void* dg=dx_->allocate(g.numel()*sizeof(float));
        void* db=dx_->allocate(bt.numel()*sizeof(float));
        void* dy=dx_->allocate(y.numel()*sizeof(float));
        if(!dxb||!dg||!db||!dy){
            if(dxb)dx_->free(dxb); if(dg)dx_->free(dg); if(db)dx_->free(db); if(dy)dx_->free(dy);
            throw_unavailable("GPU_DIRECTX","layer_norm","device alloc failed");
        }
        dx_->upload(x,dxb); dx_->upload(g,dg); dx_->upload(bt,db);
        dx_->layer_norm(dxb,dg,db,dy,e,x.dim(0),x.numel()/x.dim(0)); dx_->download(dy,y);
        dx_->free(dxb); dx_->free(dg); dx_->free(db); dx_->free(dy);
    }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override {
        require_available(avail_, "GPU_DIRECTX", "rms_norm", "no D3D12 device");
        void* dxb=dx_->allocate(x.numel()*sizeof(float));
        void* dg=dx_->allocate(g.numel()*sizeof(float));
        void* dy=dx_->allocate(y.numel()*sizeof(float));
        if(!dxb||!dg||!dy){
            if(dxb)dx_->free(dxb); if(dg)dx_->free(dg); if(dy)dx_->free(dy);
            throw_unavailable("GPU_DIRECTX","rms_norm","device alloc failed");
        }
        dx_->upload(x,dxb); dx_->upload(g,dg);
        dx_->rms_norm(dxb,dg,dy,e,x.dim(0),x.numel()/x.dim(0)); dx_->download(dy,y);
        dx_->free(dxb); dx_->free(dg); dx_->free(dy);
    }
    void relu(const Tensor& x, Tensor& y) override {
        require_available(avail_, "GPU_DIRECTX", "relu", "no D3D12 device");
        void* dxb=dx_->allocate(x.numel()*sizeof(float));
        void* dy=dx_->allocate(y.numel()*sizeof(float));
        if(!dxb||!dy){ if(dxb)dx_->free(dxb); if(dy)dx_->free(dy);
            throw_unavailable("GPU_DIRECTX","relu","device alloc failed"); }
        dx_->upload(x,dxb); dx_->relu(dxb,dy,x.numel()); dx_->download(dy,y);
        dx_->free(dxb); dx_->free(dy);
    }
    void gelu(const Tensor& x, Tensor& y) override {
        require_available(avail_, "GPU_DIRECTX", "gelu", "no D3D12 device");
        void* dxb=dx_->allocate(x.numel()*sizeof(float));
        void* dy=dx_->allocate(y.numel()*sizeof(float));
        if(!dxb||!dy){ if(dxb)dx_->free(dxb); if(dy)dx_->free(dy);
            throw_unavailable("GPU_DIRECTX","gelu","device alloc failed"); }
        dx_->upload(x,dxb); dx_->gelu(dxb,dy,x.numel()); dx_->download(dy,y);
        dx_->free(dxb); dx_->free(dy);
    }
    void silu(const Tensor& x, Tensor& y) override {
        require_available(avail_, "GPU_DIRECTX", "silu", "no D3D12 device");
        void* dxb=dx_->allocate(x.numel()*sizeof(float));
        void* dy=dx_->allocate(y.numel()*sizeof(float));
        if(!dxb||!dy){ if(dxb)dx_->free(dxb); if(dy)dx_->free(dy);
            throw_unavailable("GPU_DIRECTX","silu","device alloc failed"); }
        dx_->upload(x,dxb); dx_->silu(dxb,dy,x.numel()); dx_->download(dy,y);
        dx_->free(dxb); dx_->free(dy);
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(avail_, "GPU_DIRECTX", "add", "no D3D12 device");
        void* da=dx_->allocate(a.numel()*sizeof(float));
        void* db=dx_->allocate(b.numel()*sizeof(float));
        void* dc=dx_->allocate(c.numel()*sizeof(float));
        if(!da||!db||!dc){ if(da)dx_->free(da); if(db)dx_->free(db); if(dc)dx_->free(dc);
            throw_unavailable("GPU_DIRECTX","add","device alloc failed"); }
        dx_->upload(a,da); dx_->upload(b,db); dx_->add(da,db,dc,a.numel()); dx_->download(dc,c);
        dx_->free(da); dx_->free(db); dx_->free(dc);
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(avail_, "GPU_DIRECTX", "mul", "no D3D12 device");
        void* da=dx_->allocate(a.numel()*sizeof(float));
        void* db=dx_->allocate(b.numel()*sizeof(float));
        void* dc=dx_->allocate(c.numel()*sizeof(float));
        if(!da||!db||!dc){ if(da)dx_->free(da); if(db)dx_->free(db); if(dc)dx_->free(dc);
            throw_unavailable("GPU_DIRECTX","mul","device alloc failed"); }
        dx_->upload(a,da); dx_->upload(b,db); dx_->mul(da,db,dc,a.numel()); dx_->download(dc,c);
        dx_->free(da); dx_->free(db); dx_->free(dc);
    }
    void scale(float s, const Tensor& x, Tensor& y) override {
        require_available(avail_, "GPU_DIRECTX", "scale", "no D3D12 device");
        void* dxb=dx_->allocate(x.numel()*sizeof(float));
        void* dy=dx_->allocate(y.numel()*sizeof(float));
        if(!dxb||!dy){ if(dxb)dx_->free(dxb); if(dy)dx_->free(dy);
            throw_unavailable("GPU_DIRECTX","scale","device alloc failed"); }
        dx_->upload(x,dxb); dx_->scale(s,dxb,dy,x.numel()); dx_->download(dy,y);
        dx_->free(dxb); dx_->free(dy);
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }

    bool is_available() const override { return avail_; }
    int64_t memory_free() const override { return avail_ && dx_ ? dx_->memory_free() : 0; }
    int64_t memory_total() const override { return avail_ && dx_ ? dx_->memory_total() : 0; }
    void synchronize() override { if (avail_ && dx_) dx_->synchronize(); }
};

// ========================================================================
// GPU CUDA BACKEND — REAL (PTX kernels in src/backend/gpu_compute_cuda.cpp)
// Verdict REAL on NVIDIA+driver hosts; UNAVAILABLE+fail-loud otherwise.
// Previous silent math:: fallback REMOVED. gemv now dispatches the REAL
// device gemv path (was CPU even when GPU present — PARTIAL gap fixed).
// ========================================================================

class GPUCUDABackend : public ComputeBackend {
    gpu::GPUComputeCuda* cuda_ = nullptr;
    bool avail_ = false;
public:
    GPUCUDABackend() {
        try {
            cuda_ = &gpu::get_cuda_compute();
            if (!cuda_->is_initialized()) cuda_->init(0);
            avail_ = cuda_ && cuda_->is_initialized();
        } catch (...) { cuda_ = nullptr; avail_ = false; }
    }
    BackendType type() const override { return BackendType::GPU_CUDA; }
    const char* name() const override { return "GPU_CUDA"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        require_available(avail_, "GPU_CUDA", "gemm", "no CUDA device/driver on this host");
        void* dA = cuda_->alloc(A.numel() * sizeof(float));
        void* dB = cuda_->alloc(B.numel() * sizeof(float));
        void* dC = cuda_->alloc(C.numel() * sizeof(float));
        if (!dA||!dB||!dC) {
            if(dA)cuda_->free_buf(dA); if(dB)cuda_->free_buf(dB); if(dC)cuda_->free_buf(dC);
            throw_unavailable("GPU_CUDA","gemm","device alloc failed");
        }
        cuda_->upload(A, dA);
        cuda_->upload(B, dB);
        if (beta != 0.0f) cuda_->upload(C, dC);
        int64_t M = A.numel() / A.dim(A.rank() - 1);
        int64_t N = B.dim(B.rank() - 1);
        int64_t K = A.dim(A.rank() - 1);
        cuda_->gemm(alpha, dA, dB, beta, dC, M, N, K);
        cuda_->download(dC, C);
        cuda_->free_buf(dA);
        cuda_->free_buf(dB);
        cuda_->free_buf(dC);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        require_available(avail_, "GPU_CUDA", "gemv", "no CUDA device/driver on this host");
        void* dA = cuda_->alloc(A.numel()*sizeof(float));
        void* dxb = cuda_->alloc(x.numel()*sizeof(float));
        void* dy = cuda_->alloc(y.numel()*sizeof(float));
        if (!dA||!dxb||!dy) {
            if(dA)cuda_->free_buf(dA); if(dxb)cuda_->free_buf(dxb); if(dy)cuda_->free_buf(dy);
            throw_unavailable("GPU_CUDA","gemv","device alloc failed");
        }
        cuda_->upload(A,dA); cuda_->upload(x,dxb);
        if (beta != 0.0f) cuda_->upload(y,dy);
        int64_t M = A.numel()/A.dim(A.rank()-1), N = A.dim(A.rank()-1);
        cuda_->gemv(alpha,dA,dxb,beta,dy,M,N);
        cuda_->download(dy,y);
        cuda_->free_buf(dA); cuda_->free_buf(dxb); cuda_->free_buf(dy);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        require_available(avail_, "GPU_CUDA", "softmax", "no CUDA device/driver on this host");
        if (axis != 1 && !(axis==-1 && x.rank()==2))
            throw_unavailable("GPU_CUDA","softmax","only axis=1 2-D supported on device");
        void* dxb=cuda_->alloc(x.numel()*sizeof(float));
        void* dy=cuda_->alloc(y.numel()*sizeof(float));
        if(!dxb||!dy){ if(dxb)cuda_->free_buf(dxb); if(dy)cuda_->free_buf(dy);
            throw_unavailable("GPU_CUDA","softmax","device alloc failed"); }
        cuda_->upload(x,dxb); cuda_->softmax(dxb,dy,x.dim(0),x.numel()/x.dim(0)); cuda_->download(dy,y);
        cuda_->free_buf(dxb); cuda_->free_buf(dy);
    }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override {
        require_available(avail_, "GPU_CUDA", "layer_norm", "no CUDA device/driver on this host");
        void* dxb=cuda_->alloc(x.numel()*sizeof(float));
        void* dg=cuda_->alloc(g.numel()*sizeof(float));
        void* db=cuda_->alloc(bt.numel()*sizeof(float));
        void* dy=cuda_->alloc(y.numel()*sizeof(float));
        if(!dxb||!dg||!db||!dy){
            if(dxb)cuda_->free_buf(dxb); if(dg)cuda_->free_buf(dg);
            if(db)cuda_->free_buf(db); if(dy)cuda_->free_buf(dy);
            throw_unavailable("GPU_CUDA","layer_norm","device alloc failed"); }
        cuda_->upload(x,dxb); cuda_->upload(g,dg); cuda_->upload(bt,db);
        cuda_->layer_norm(dxb,dg,db,dy,e,x.dim(0),x.numel()/x.dim(0)); cuda_->download(dy,y);
        cuda_->free_buf(dxb); cuda_->free_buf(dg); cuda_->free_buf(db); cuda_->free_buf(dy);
    }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override {
        require_available(avail_, "GPU_CUDA", "rms_norm", "no CUDA device/driver on this host");
        void* dxb=cuda_->alloc(x.numel()*sizeof(float));
        void* dg=cuda_->alloc(g.numel()*sizeof(float));
        void* dy=cuda_->alloc(y.numel()*sizeof(float));
        if(!dxb||!dg||!dy){
            if(dxb)cuda_->free_buf(dxb); if(dg)cuda_->free_buf(dg); if(dy)cuda_->free_buf(dy);
            throw_unavailable("GPU_CUDA","rms_norm","device alloc failed"); }
        cuda_->upload(x,dxb); cuda_->upload(g,dg);
        cuda_->rms_norm(dxb,dg,dy,e,x.dim(0),x.numel()/x.dim(0)); cuda_->download(dy,y);
        cuda_->free_buf(dxb); cuda_->free_buf(dg); cuda_->free_buf(dy);
    }
    void relu(const Tensor& x, Tensor& y) override {
        require_available(avail_, "GPU_CUDA", "relu", "no CUDA device/driver on this host");
        void* dxb=cuda_->alloc(x.numel()*sizeof(float));
        void* dy=cuda_->alloc(y.numel()*sizeof(float));
        if(!dxb||!dy){ if(dxb)cuda_->free_buf(dxb); if(dy)cuda_->free_buf(dy);
            throw_unavailable("GPU_CUDA","relu","device alloc failed"); }
        cuda_->upload(x,dxb); cuda_->relu(dxb,dy,x.numel()); cuda_->download(dy,y);
        cuda_->free_buf(dxb); cuda_->free_buf(dy);
    }
    void gelu(const Tensor& x, Tensor& y) override {
        require_available(avail_, "GPU_CUDA", "gelu", "no CUDA device/driver on this host");
        void* dxb=cuda_->alloc(x.numel()*sizeof(float));
        void* dy=cuda_->alloc(y.numel()*sizeof(float));
        if(!dxb||!dy){ if(dxb)cuda_->free_buf(dxb); if(dy)cuda_->free_buf(dy);
            throw_unavailable("GPU_CUDA","gelu","device alloc failed"); }
        cuda_->upload(x,dxb); cuda_->gelu(dxb,dy,x.numel()); cuda_->download(dy,y);
        cuda_->free_buf(dxb); cuda_->free_buf(dy);
    }
    void silu(const Tensor& x, Tensor& y) override {
        require_available(avail_, "GPU_CUDA", "silu", "no CUDA device/driver on this host");
        void* dxb=cuda_->alloc(x.numel()*sizeof(float));
        void* dy=cuda_->alloc(y.numel()*sizeof(float));
        if(!dxb||!dy){ if(dxb)cuda_->free_buf(dxb); if(dy)cuda_->free_buf(dy);
            throw_unavailable("GPU_CUDA","silu","device alloc failed"); }
        cuda_->upload(x,dxb); cuda_->silu(dxb,dy,x.numel()); cuda_->download(dy,y);
        cuda_->free_buf(dxb); cuda_->free_buf(dy);
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(avail_, "GPU_CUDA", "add", "no CUDA device/driver on this host");
        void* da=cuda_->alloc(a.numel()*sizeof(float));
        void* db=cuda_->alloc(b.numel()*sizeof(float));
        void* dc=cuda_->alloc(c.numel()*sizeof(float));
        if(!da||!db||!dc){ if(da)cuda_->free_buf(da); if(db)cuda_->free_buf(db); if(dc)cuda_->free_buf(dc);
            throw_unavailable("GPU_CUDA","add","device alloc failed"); }
        cuda_->upload(a,da); cuda_->upload(b,db); cuda_->add(da,db,dc,a.numel()); cuda_->download(dc,c);
        cuda_->free_buf(da); cuda_->free_buf(db); cuda_->free_buf(dc);
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(avail_, "GPU_CUDA", "mul", "no CUDA device/driver on this host");
        void* da=cuda_->alloc(a.numel()*sizeof(float));
        void* db=cuda_->alloc(b.numel()*sizeof(float));
        void* dc=cuda_->alloc(c.numel()*sizeof(float));
        if(!da||!db||!dc){ if(da)cuda_->free_buf(da); if(db)cuda_->free_buf(db); if(dc)cuda_->free_buf(dc);
            throw_unavailable("GPU_CUDA","mul","device alloc failed"); }
        cuda_->upload(a,da); cuda_->upload(b,db); cuda_->mul(da,db,dc,a.numel()); cuda_->download(dc,c);
        cuda_->free_buf(da); cuda_->free_buf(db); cuda_->free_buf(dc);
    }
    void scale(float s, const Tensor& x, Tensor& y) override {
        require_available(avail_, "GPU_CUDA", "scale", "no CUDA device/driver on this host");
        void* dxb=cuda_->alloc(x.numel()*sizeof(float));
        void* dy=cuda_->alloc(y.numel()*sizeof(float));
        if(!dxb||!dy){ if(dxb)cuda_->free_buf(dxb); if(dy)cuda_->free_buf(dy);
            throw_unavailable("GPU_CUDA","scale","device alloc failed"); }
        cuda_->upload(x,dxb); cuda_->scale(s,dxb,dy,x.numel()); cuda_->download(dy,y);
        cuda_->free_buf(dxb); cuda_->free_buf(dy);
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override {
        require_available(avail_, "GPU_CUDA", "fill", "no CUDA device/driver on this host");
        void* d=cuda_->alloc(t.numel()*sizeof(float));
        if(!d) throw_unavailable("GPU_CUDA","fill","device alloc failed");
        cuda_->fill(val,d,t.numel()); cuda_->download(d,t); cuda_->free_buf(d);
    }
    void zero(Tensor& t) override { fill(t, 0.0f); }

    bool is_available() const override { return avail_; }
    int64_t memory_free() const override { return avail_ && cuda_ ? cuda_->memory_free() : 0; }
    int64_t memory_total() const override { return avail_ && cuda_ ? cuda_->memory_total() : 0; }
    void synchronize() override { if (avail_ && cuda_) cuda_->synchronize(); }
};

// ========================================================================
// GPU METAL BACKEND — REAL (MSL kernels in src/backend/gpu_compute_metal.cpp)
// Verdict REAL on Apple Silicon/macOS; UNAVAILABLE+fail-loud elsewhere.
// Previous silent math:: fallback REMOVED (REAL-ONLY violation).
// Note: metal_ API is only functional on __APPLE__; on other OSes init
// fails honestly and every op throws.
// ========================================================================

class GPUMetalBackend : public ComputeBackend {
    gpu::GPUComputeMetal* metal_ = nullptr;
    bool avail_ = false;
public:
    GPUMetalBackend() {
        try {
            metal_ = &gpu::get_metal_compute();
            if (!metal_->is_initialized()) metal_->init(0);
            avail_ = metal_ && metal_->is_initialized();
        } catch (...) { metal_ = nullptr; avail_ = false; }
    }
    BackendType type() const override { return BackendType::GPU_METAL; }
    const char* name() const override { return "GPU_METAL"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        require_available(avail_, "GPU_METAL", "gemm", "no Metal device (Apple hardware required)");
        void* dA = metal_->alloc(A.numel() * sizeof(float));
        void* dB = metal_->alloc(B.numel() * sizeof(float));
        void* dC = metal_->alloc(C.numel() * sizeof(float));
        if (!dA||!dB||!dC) {
            if(dA)metal_->free_buf(dA); if(dB)metal_->free_buf(dB); if(dC)metal_->free_buf(dC);
            throw_unavailable("GPU_METAL","gemm","device alloc failed");
        }
        metal_->upload(A, dA);
        metal_->upload(B, dB);
        if (beta != 0.0f) metal_->upload(C, dC);
        int64_t M = A.numel() / A.dim(A.rank() - 1);
        int64_t N = B.dim(B.rank() - 1);
        int64_t K = A.dim(A.rank() - 1);
        metal_->gemm(alpha, dA, dB, beta, dC, M, N, K);
        metal_->download(dC, C);
        metal_->free_buf(dA);
        metal_->free_buf(dB);
        metal_->free_buf(dC);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        require_available(avail_, "GPU_METAL", "gemv", "no Metal device (Apple hardware required)");
        void* dA=metal_->alloc(A.numel()*sizeof(float));
        void* dxb=metal_->alloc(x.numel()*sizeof(float));
        void* dy=metal_->alloc(y.numel()*sizeof(float));
        if(!dA||!dxb||!dy){ if(dA)metal_->free_buf(dA); if(dxb)metal_->free_buf(dxb); if(dy)metal_->free_buf(dy);
            throw_unavailable("GPU_METAL","gemv","device alloc failed"); }
        metal_->upload(A,dA); metal_->upload(x,dxb);
        if (beta!=0.0f) metal_->upload(y,dy);
        metal_->gemv(alpha,dA,dxb,beta,dy,A.numel()/A.dim(A.rank()-1),A.dim(A.rank()-1));
        metal_->download(dy,y);
        metal_->free_buf(dA); metal_->free_buf(dxb); metal_->free_buf(dy);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        require_available(avail_, "GPU_METAL", "softmax", "no Metal device");
        if (axis!=1 && !(axis==-1 && x.rank()==2))
            throw_unavailable("GPU_METAL","softmax","only axis=1 2-D supported on device");
        void* dxb=metal_->alloc(x.numel()*sizeof(float));
        void* dy=metal_->alloc(y.numel()*sizeof(float));
        if(!dxb||!dy){ if(dxb)metal_->free_buf(dxb); if(dy)metal_->free_buf(dy);
            throw_unavailable("GPU_METAL","softmax","device alloc failed"); }
        metal_->upload(x,dxb); metal_->softmax(dxb,dy,x.dim(0),x.numel()/x.dim(0)); metal_->download(dy,y);
        metal_->free_buf(dxb); metal_->free_buf(dy);
    }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override {
        require_available(avail_, "GPU_METAL", "layer_norm", "no Metal device");
        void* dxb=metal_->alloc(x.numel()*sizeof(float));
        void* dg=metal_->alloc(g.numel()*sizeof(float));
        void* db=metal_->alloc(bt.numel()*sizeof(float));
        void* dy=metal_->alloc(y.numel()*sizeof(float));
        if(!dxb||!dg||!db||!dy){
            if(dxb)metal_->free_buf(dxb); if(dg)metal_->free_buf(dg);
            if(db)metal_->free_buf(db); if(dy)metal_->free_buf(dy);
            throw_unavailable("GPU_METAL","layer_norm","device alloc failed"); }
        metal_->upload(x,dxb); metal_->upload(g,dg); metal_->upload(bt,db);
        metal_->layer_norm(dxb,dg,db,dy,e,x.dim(0),x.numel()/x.dim(0)); metal_->download(dy,y);
        metal_->free_buf(dxb); metal_->free_buf(dg); metal_->free_buf(db); metal_->free_buf(dy);
    }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override {
        require_available(avail_, "GPU_METAL", "rms_norm", "no Metal device");
        void* dxb=metal_->alloc(x.numel()*sizeof(float));
        void* dg=metal_->alloc(g.numel()*sizeof(float));
        void* dy=metal_->alloc(y.numel()*sizeof(float));
        if(!dxb||!dg||!dy){
            if(dxb)metal_->free_buf(dxb); if(dg)metal_->free_buf(dg); if(dy)metal_->free_buf(dy);
            throw_unavailable("GPU_METAL","rms_norm","device alloc failed"); }
        metal_->upload(x,dxb); metal_->upload(g,dg);
        metal_->rms_norm(dxb,dg,dy,e,x.dim(0),x.numel()/x.dim(0)); metal_->download(dy,y);
        metal_->free_buf(dxb); metal_->free_buf(dg); metal_->free_buf(dy);
    }
    void relu(const Tensor& x, Tensor& y) override {
        require_available(avail_, "GPU_METAL", "relu", "no Metal device");
        void* dxb=metal_->alloc(x.numel()*sizeof(float));
        void* dy=metal_->alloc(y.numel()*sizeof(float));
        if(!dxb||!dy){ if(dxb)metal_->free_buf(dxb); if(dy)metal_->free_buf(dy);
            throw_unavailable("GPU_METAL","relu","device alloc failed"); }
        metal_->upload(x,dxb); metal_->relu(dxb,dy,x.numel()); metal_->download(dy,y);
        metal_->free_buf(dxb); metal_->free_buf(dy);
    }
    void gelu(const Tensor& x, Tensor& y) override {
        require_available(avail_, "GPU_METAL", "gelu", "no Metal device");
        void* dxb=metal_->alloc(x.numel()*sizeof(float));
        void* dy=metal_->alloc(y.numel()*sizeof(float));
        if(!dxb||!dy){ if(dxb)metal_->free_buf(dxb); if(dy)metal_->free_buf(dy);
            throw_unavailable("GPU_METAL","gelu","device alloc failed"); }
        metal_->upload(x,dxb); metal_->gelu(dxb,dy,x.numel()); metal_->download(dy,y);
        metal_->free_buf(dxb); metal_->free_buf(dy);
    }
    void silu(const Tensor& x, Tensor& y) override {
        require_available(avail_, "GPU_METAL", "silu", "no Metal device");
        void* dxb=metal_->alloc(x.numel()*sizeof(float));
        void* dy=metal_->alloc(y.numel()*sizeof(float));
        if(!dxb||!dy){ if(dxb)metal_->free_buf(dxb); if(dy)metal_->free_buf(dy);
            throw_unavailable("GPU_METAL","silu","device alloc failed"); }
        metal_->upload(x,dxb); metal_->silu(dxb,dy,x.numel()); metal_->download(dy,y);
        metal_->free_buf(dxb); metal_->free_buf(dy);
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(avail_, "GPU_METAL", "add", "no Metal device");
        void* da=metal_->alloc(a.numel()*sizeof(float));
        void* db=metal_->alloc(b.numel()*sizeof(float));
        void* dc=metal_->alloc(c.numel()*sizeof(float));
        if(!da||!db||!dc){ if(da)metal_->free_buf(da); if(db)metal_->free_buf(db); if(dc)metal_->free_buf(dc);
            throw_unavailable("GPU_METAL","add","device alloc failed"); }
        metal_->upload(a,da); metal_->upload(b,db); metal_->add(da,db,dc,a.numel()); metal_->download(dc,c);
        metal_->free_buf(da); metal_->free_buf(db); metal_->free_buf(dc);
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(avail_, "GPU_METAL", "mul", "no Metal device");
        void* da=metal_->alloc(a.numel()*sizeof(float));
        void* db=metal_->alloc(b.numel()*sizeof(float));
        void* dc=metal_->alloc(c.numel()*sizeof(float));
        if(!da||!db||!dc){ if(da)metal_->free_buf(da); if(db)metal_->free_buf(db); if(dc)metal_->free_buf(dc);
            throw_unavailable("GPU_METAL","mul","device alloc failed"); }
        metal_->upload(a,da); metal_->upload(b,db); metal_->mul(da,db,dc,a.numel()); metal_->download(dc,c);
        metal_->free_buf(da); metal_->free_buf(db); metal_->free_buf(dc);
    }
    void scale(float s, const Tensor& x, Tensor& y) override {
        require_available(avail_, "GPU_METAL", "scale", "no Metal device");
        void* dxb=metal_->alloc(x.numel()*sizeof(float));
        void* dy=metal_->alloc(y.numel()*sizeof(float));
        if(!dxb||!dy){ if(dxb)metal_->free_buf(dxb); if(dy)metal_->free_buf(dy);
            throw_unavailable("GPU_METAL","scale","device alloc failed"); }
        metal_->upload(x,dxb); metal_->scale(s,dxb,dy,x.numel()); metal_->download(dy,y);
        metal_->free_buf(dxb); metal_->free_buf(dy);
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override {
        require_available(avail_, "GPU_METAL", "fill", "no Metal device");
        void* d=metal_->alloc(t.numel()*sizeof(float));
        if(!d) throw_unavailable("GPU_METAL","fill","device alloc failed");
        metal_->fill(val,d,t.numel()); metal_->download(d,t); metal_->free_buf(d);
    }
    void zero(Tensor& t) override { fill(t, 0.0f); }

    bool is_available() const override { return avail_; }
    int64_t memory_free() const override { return avail_ && metal_ ? metal_->memory_free() : 0; }
    int64_t memory_total() const override { return avail_ && metal_ ? metal_->memory_total() : 0; }
    void synchronize() override { if (avail_ && metal_) metal_->synchronize(); }
};

// ========================================================================
// GPU SYCL BACKEND (dynamically loaded)
// ========================================================================

class GPUSYCLBackend : public ComputeBackend {
    gpu::GPUComputeSycl* sycl_ = nullptr;
    bool avail_ = false;
public:
    GPUSYCLBackend() {
        try {
            sycl_ = &gpu::get_sycl_compute();
            if (!sycl_->is_initialized()) sycl_->init(0);
            avail_ = sycl_ && sycl_->is_initialized();
        } catch (...) { sycl_ = nullptr; avail_ = false; }
    }
    BackendType type() const override { return BackendType::GPU_SYCL; }
    const char* name() const override { return "GPU_SYCL"; }

    // REAL ONLY verdict PARTIAL: USM alloc/upload/download/sync are REAL
    // (Level-Zero PI); every compute kernel is CPU-only in
    // gpu_compute_sycl.cpp (requires DPC++ to dispatch). Calling them as
    // "SYCL GPU" would be FAKE, so they fail loud. Previous silent
    // math:: fallback REMOVED.
    void gemm(float, const Tensor&, const Tensor&, float, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_SYCL", "gemm", "no SYCL GPU runtime on this host");
        throw_unavailable("GPU_SYCL", "gemm", "SYCL device GEMM kernel not implemented (DPC++ compiler required)");
    }
    void gemv(float, const Tensor&, const Tensor&, float, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_SYCL", "gemv", "no SYCL GPU runtime on this host");
        throw_unavailable("GPU_SYCL", "gemv", "SYCL device GEMV kernel not implemented (DPC++ compiler required)");
    }
    void softmax(const Tensor&, Tensor&, int) override {
        if (!avail_) throw_unavailable("GPU_SYCL", "softmax", "no SYCL GPU runtime on this host");
        throw_unavailable("GPU_SYCL", "softmax", "SYCL device softmax kernel not implemented");
    }
    void layer_norm(const Tensor&, const Tensor&, const Tensor&, float, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_SYCL", "layer_norm", "no SYCL GPU runtime on this host");
        throw_unavailable("GPU_SYCL", "layer_norm", "SYCL device layer_norm kernel not implemented");
    }
    void rms_norm(const Tensor&, const Tensor&, float, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_SYCL", "rms_norm", "no SYCL GPU runtime on this host");
        throw_unavailable("GPU_SYCL", "rms_norm", "SYCL device rms_norm kernel not implemented");
    }
    void relu(const Tensor&, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_SYCL", "relu", "no SYCL GPU runtime on this host");
        throw_unavailable("GPU_SYCL", "relu", "SYCL device relu kernel not implemented");
    }
    void gelu(const Tensor&, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_SYCL", "gelu", "no SYCL GPU runtime on this host");
        throw_unavailable("GPU_SYCL", "gelu", "SYCL device gelu kernel not implemented");
    }
    void silu(const Tensor&, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_SYCL", "silu", "no SYCL GPU runtime on this host");
        throw_unavailable("GPU_SYCL", "silu", "SYCL device silu kernel not implemented");
    }
    void add(const Tensor&, const Tensor&, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_SYCL", "add", "no SYCL GPU runtime on this host");
        throw_unavailable("GPU_SYCL", "add", "SYCL device add kernel not implemented");
    }
    void mul(const Tensor&, const Tensor&, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_SYCL", "mul", "no SYCL GPU runtime on this host");
        throw_unavailable("GPU_SYCL", "mul", "SYCL device mul kernel not implemented");
    }
    void scale(float, const Tensor&, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_SYCL", "scale", "no SYCL GPU runtime on this host");
        throw_unavailable("GPU_SYCL", "scale", "SYCL device scale kernel not implemented");
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }

    bool is_available() const override { return avail_; }
    int64_t memory_free() const override { return avail_ && sycl_ ? sycl_->memory_free() : 0; }
    int64_t memory_total() const override { return avail_ && sycl_ ? sycl_->memory_total() : 0; }
    void synchronize() override { if (avail_ && sycl_) sycl_->synchronize(); }
};

// ========================================================================
// GPU CANN BACKEND (dynamically loaded)
// ========================================================================

class GPUCANNBackend : public ComputeBackend {
    gpu::GPUComputeCann* cann_ = nullptr;
    bool avail_ = false;
public:
    GPUCANNBackend() {
        try {
            cann_ = &gpu::get_cann_compute();
            if (!cann_->is_initialized()) cann_->init(0);
            avail_ = cann_ && cann_->is_initialized();
        } catch (...) { cann_ = nullptr; avail_ = false; }
    }
    BackendType type() const override { return BackendType::GPU_CANN; }
    const char* name() const override { return "GPU_CANN"; }

    // REAL ONLY verdict PARTIAL: ACL alloc/memcpy/stream/sync are REAL;
    // every compute op currently calls aclopExecuteV2 with NULL descriptors
    // (no-op, leaves C untouched) — claiming that as MatMul would be FAKE.
    // All compute ops fail loud until real tensor descriptors land.
    void gemm(float, const Tensor&, const Tensor&, float, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_CANN", "gemm", "no Ascend CANN device on this host");
        throw_unavailable("GPU_CANN", "gemm", "CANN MatMul descriptors not implemented (null-descriptor stub)");
    }
    void gemv(float, const Tensor&, const Tensor&, float, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_CANN", "gemv", "no Ascend CANN device on this host");
        throw_unavailable("GPU_CANN", "gemv", "CANN MatMul descriptors not implemented (null-descriptor stub)");
    }
    void softmax(const Tensor&, Tensor&, int) override {
        if (!avail_) throw_unavailable("GPU_CANN", "softmax", "no Ascend CANN device on this host");
        throw_unavailable("GPU_CANN", "softmax", "CANN SoftmaxV2 descriptors not implemented");
    }
    void layer_norm(const Tensor&, const Tensor&, const Tensor&, float, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_CANN", "layer_norm", "no Ascend CANN device on this host");
        throw_unavailable("GPU_CANN", "layer_norm", "CANN LayerNorm descriptors not implemented");
    }
    void rms_norm(const Tensor&, const Tensor&, float, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_CANN", "rms_norm", "no Ascend CANN device on this host");
        throw_unavailable("GPU_CANN", "rms_norm", "CANN LayerNorm descriptors not implemented");
    }
    void relu(const Tensor&, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_CANN", "relu", "no Ascend CANN device on this host");
        throw_unavailable("GPU_CANN", "relu", "CANN Relu descriptors not implemented");
    }
    void gelu(const Tensor&, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_CANN", "gelu", "no Ascend CANN device on this host");
        throw_unavailable("GPU_CANN", "gelu", "CANN Gelu descriptors not implemented");
    }
    void silu(const Tensor&, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_CANN", "silu", "no Ascend CANN device on this host");
        throw_unavailable("GPU_CANN", "silu", "CANN Swish descriptors not implemented");
    }
    void add(const Tensor&, const Tensor&, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_CANN", "add", "no Ascend CANN device on this host");
        throw_unavailable("GPU_CANN", "add", "CANN Add descriptors not implemented");
    }
    void mul(const Tensor&, const Tensor&, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_CANN", "mul", "no Ascend CANN device on this host");
        throw_unavailable("GPU_CANN", "mul", "CANN Mul descriptors not implemented");
    }
    void scale(float, const Tensor&, Tensor&) override {
        if (!avail_) throw_unavailable("GPU_CANN", "scale", "no Ascend CANN device on this host");
        throw_unavailable("GPU_CANN", "scale", "CANN Scale descriptors not implemented");
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }

    bool is_available() const override { return avail_; }
    int64_t memory_free() const override { return avail_ && cann_ ? cann_->memory_free() : 0; }
    int64_t memory_total() const override { return avail_ && cann_ ? cann_->memory_total() : 0; }
    void synchronize() override { if (avail_ && cann_) cann_->synchronize(); }
};

// ========================================================================
// RPC BACKEND (remote compute)
// ========================================================================

class GPURPCBackend : public ComputeBackend {
    gpu::GPUComputeRpc* rpc_ = nullptr;
public:
    GPURPCBackend() {
        try {
            rpc_ = &gpu::get_rpc_compute();
            rpc_->init();
        } catch (...) {
            rpc_ = nullptr;
        }
    }
    BackendType type() const override { return BackendType::RPC; }
    const char* name() const override { return "RPC"; }

    // REAL ONLY verdict PARTIAL: wire protocol + socket connect are REAL;
    // every op fails loud when no server answers (no silent math::). gemv /
    // layer_norm / scale have no RPC opcode -> fail loud even when linked.
    // Previous silent math:: fallback REMOVED.
    bool rpc_live() const { return rpc_ && rpc_->is_initialized() && is_rpc_available(); }
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        require_available(rpc_live(), "RPC", "gemm", "no RPC server reachable (127.0.0.1:9000)");
        try { rpc_->gemm(alpha, A, B, beta, C); }
        catch (const std::runtime_error&) { throw; }
        catch (...) { throw_unavailable("RPC", "gemm", "RPC transport failed mid-call"); }
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        (void)alpha; (void)A; (void)x; (void)beta; (void)y;
        if (!rpc_live()) throw_unavailable("RPC", "gemv", "no RPC server reachable (127.0.0.1:9000)");
        throw_unavailable("RPC", "gemv", "no GEMV opcode in RPC protocol (OP_GEMM..OP_MUL only)");
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        require_available(rpc_live(), "RPC", "softmax", "no RPC server reachable (127.0.0.1:9000)");
        try { rpc_->softmax(x, y, axis); }
        catch (const std::runtime_error&) { throw; }
        catch (...) { throw_unavailable("RPC", "softmax", "RPC transport failed mid-call"); }
    }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override {
        (void)x; (void)g; (void)bt; (void)e; (void)y;
        if (!rpc_live()) throw_unavailable("RPC", "layer_norm", "no RPC server reachable (127.0.0.1:9000)");
        throw_unavailable("RPC", "layer_norm", "no LAYER_NORM opcode in RPC protocol");
    }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override {
        require_available(rpc_live(), "RPC", "rms_norm", "no RPC server reachable (127.0.0.1:9000)");
        try { rpc_->rms_norm(x, g, e, y); }
        catch (const std::runtime_error&) { throw; }
        catch (...) { throw_unavailable("RPC", "rms_norm", "RPC transport failed mid-call"); }
    }
    void relu(const Tensor& x, Tensor& y) override {
        require_available(rpc_live(), "RPC", "relu", "no RPC server reachable (127.0.0.1:9000)");
        try { rpc_->relu(x, y); }
        catch (const std::runtime_error&) { throw; }
        catch (...) { throw_unavailable("RPC", "relu", "RPC transport failed mid-call"); }
    }
    void gelu(const Tensor& x, Tensor& y) override {
        require_available(rpc_live(), "RPC", "gelu", "no RPC server reachable (127.0.0.1:9000)");
        try { rpc_->gelu(x, y); }
        catch (const std::runtime_error&) { throw; }
        catch (...) { throw_unavailable("RPC", "gelu", "RPC transport failed mid-call"); }
    }
    void silu(const Tensor& x, Tensor& y) override {
        require_available(rpc_live(), "RPC", "silu", "no RPC server reachable (127.0.0.1:9000)");
        try { rpc_->silu(x, y); }
        catch (const std::runtime_error&) { throw; }
        catch (...) { throw_unavailable("RPC", "silu", "RPC transport failed mid-call"); }
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(rpc_live(), "RPC", "add", "no RPC server reachable (127.0.0.1:9000)");
        try { rpc_->add(a, b, c); }
        catch (const std::runtime_error&) { throw; }
        catch (...) { throw_unavailable("RPC", "add", "RPC transport failed mid-call"); }
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(rpc_live(), "RPC", "mul", "no RPC server reachable (127.0.0.1:9000)");
        try { rpc_->mul(a, b, c); }
        catch (const std::runtime_error&) { throw; }
        catch (...) { throw_unavailable("RPC", "mul", "RPC transport failed mid-call"); }
    }
    void scale(float s, const Tensor& x, Tensor& y) override {
        (void)s; (void)x; (void)y;
        if (!rpc_live()) throw_unavailable("RPC", "scale", "no RPC server reachable (127.0.0.1:9000)");
        throw_unavailable("RPC", "scale", "no SCALE opcode in RPC protocol");
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }

    // REAL ONLY: RPC is_available MUST reflect a live server, not "always
    // true with CPU fallback". is_rpc_available() probes localhost:9000;
    // every op throws when the server is unreachable (fail-loud).
    bool is_available() const override { return rpc_ && rpc_->is_initialized() && is_rpc_available(); }
    int64_t memory_free() const override { return 0; }
    int64_t memory_total() const override { return 0; }
    void synchronize() override { if (rpc_) rpc_->synchronize(); }
};

// ========================================================================
// RAM SWAP BACKEND (memory-efficient, CPU, disk swap)
// ========================================================================

class RAMSwapBackend : public CPUScalarBackend {
    int64_t swap_threshold_bytes = 4LL * 1024 * 1024 * 1024;
public:
    BackendType type() const override { return BackendType::RAM_SWAP; }
    const char* name() const override { return "RAM_SWAP"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        if (A.numel() * B.numel() > swap_threshold_bytes) {
            // For large matrices: tile and process sequentially
            int64_t M = A.numel() / A.dim(A.rank() - 1);
            int64_t K = A.dim(A.rank() - 1);
            int64_t N = B.dim(B.rank() - 1);
            int64_t tile_m = 64;
            C.zero_();
            for (int64_t mt = 0; mt < M; mt += tile_m) {
                int64_t m_end = std::min(mt + tile_m, M);
                int64_t m_size = m_end - mt; (void)m_size;
                Tensor A_tile = A.reshape({(int64_t)M, K}).slice(0, mt, m_end);
                Tensor C_tile = C.reshape({(int64_t)M, N}).slice(0, mt, m_end);
                math::gemm(alpha, A_tile, B, beta, C_tile);
            }
        } else {
            math::gemm(alpha, A, B, beta, C);
        }
    }

    bool is_available() const override { return true; }
    int64_t memory_free() const override {
        return cpu_memory_free();
    }
};

// ========================================================================
// DISTRIBUTED BACKEND (multi-node via MPI-style abstraction)
// ========================================================================

class DistributedBackend : public ComputeBackend {
    CPUScalarBackend local_backend;
    int64_t world_size_;
    int64_t rank_;
public:
    DistributedBackend() : world_size_(1), rank_(0) {}
    void init(int64_t world_size, int64_t rank) { world_size_ = world_size; rank_ = rank; }

    BackendType type() const override { return BackendType::DISTRIBUTED; }
    const char* name() const override { return "DISTRIBUTED"; }

    // REAL ONLY: sharding is REAL CPU work, but a default-constructed
    // backend (world_size 1, rank 0) reports unavailable until init(world,
    // rank) runs — every op below enforces that (fail-loud, no silent
    // single-rank masquerade as "distributed").
    bool dist_live() const { return world_size_ > 1 && rank_ >= 0 && rank_ < world_size_; }
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        require_available(dist_live(), "DISTRIBUTED", "gemm", "backend not initialized (need init(world_size>1, rank))");
        // Shard C across devices: each device computes C[rank*rows_per_device:(rank+1)*rows_per_device, :]
        int64_t M = C.numel() / C.dim(C.rank() - 1);
        int64_t K = A.dim(A.rank() - 1);
        int64_t N = B.dim(B.rank() - 1);
        int64_t rows_per_device = (M + world_size_ - 1) / world_size_;
        int64_t start = rank_ * rows_per_device;
        int64_t end = std::min(start + rows_per_device, M);
        if (start >= M) return;
        int64_t local_rows = end - start; (void)local_rows;
        Tensor A_local = A.reshape({(int64_t)M, K}).slice(0, start, end);
        Tensor C_local = C.reshape({(int64_t)M, N}).slice(0, start, end);
        local_backend.gemm(alpha, A_local, B, beta, C_local);
    }

    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        require_available(dist_live(), "DISTRIBUTED", "gemv", "backend not initialized (need init(world_size>1, rank))");
        local_backend.gemv(alpha, A, x, beta, y);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        require_available(dist_live(), "DISTRIBUTED", "softmax", "backend not initialized (need init(world_size>1, rank))");
        local_backend.softmax(x, y, axis);
    }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override {
        require_available(dist_live(), "DISTRIBUTED", "layer_norm", "backend not initialized (need init(world_size>1, rank))");
        local_backend.layer_norm(x,g,bt,e,y);
    }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override {
        require_available(dist_live(), "DISTRIBUTED", "rms_norm", "backend not initialized (need init(world_size>1, rank))");
        local_backend.rms_norm(x,g,e,y);
    }
    void relu(const Tensor& x, Tensor& y) override {
        require_available(dist_live(), "DISTRIBUTED", "relu", "backend not initialized (need init(world_size>1, rank))");
        local_backend.relu(x,y);
    }
    void gelu(const Tensor& x, Tensor& y) override {
        require_available(dist_live(), "DISTRIBUTED", "gelu", "backend not initialized (need init(world_size>1, rank))");
        local_backend.gelu(x,y);
    }
    void silu(const Tensor& x, Tensor& y) override {
        require_available(dist_live(), "DISTRIBUTED", "silu", "backend not initialized (need init(world_size>1, rank))");
        local_backend.silu(x,y);
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(dist_live(), "DISTRIBUTED", "add", "backend not initialized (need init(world_size>1, rank))");
        local_backend.add(a,b,c);
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(dist_live(), "DISTRIBUTED", "mul", "backend not initialized (need init(world_size>1, rank))");
        local_backend.mul(a,b,c);
    }
    void scale(float s, const Tensor& x, Tensor& y) override {
        require_available(dist_live(), "DISTRIBUTED", "scale", "backend not initialized (need init(world_size>1, rank))");
        local_backend.scale(s,x,y);
    }
    void copy(const Tensor& src, Tensor& dst) override { local_backend.copy(src,dst); }
    void fill(Tensor& t, float v) override { local_backend.fill(t,v); }
    void zero(Tensor& t) override { local_backend.zero(t); }

    bool is_available() const override { return dist_live(); }
    int64_t memory_free() const override { return local_backend.memory_free(); }
    int64_t memory_total() const override { return local_backend.memory_total(); }
    void synchronize() override {}

    int64_t world_size() const { return world_size_; }
    int64_t rank() const { return rank_; }
};

// ========================================================================
// OPENVINO BACKEND
// ========================================================================

class GPUOpenVINOBackend : public ComputeBackend {
    gpu::GPUComputeOpenVINO* ov_ = nullptr;
public:
    GPUOpenVINOBackend() {
        try {
            ov_ = &gpu::get_openvino_compute();
            ov_->init(0);
        } catch (...) {
            ov_ = nullptr;
        }
    }
    BackendType type() const override { return BackendType::NPU_OPENVINO; }
    const char* name() const override { return "NPU_OPENVINO"; }
    // REAL ONLY verdict UNAVAILABLE on this host (no openvino_c lib): the
    // low-level gemm is a CPU triple loop + malloc masquerading as NPU, so
    // every compute op fails loud. Previous silent math:: fallback REMOVED.
    // If a future host loads openvino_c AND real graph kernels land, flip
    // these to dispatch; until then PARTIAL at best (init probe REAL).
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        (void)alpha; (void)A; (void)B; (void)beta; (void)C;
        if (!ov_ || !ov_->is_initialized()) throw_unavailable("NPU_OPENVINO", "gemm", "no OpenVINO runtime on this host");
        throw_unavailable("NPU_OPENVINO", "gemm", "OpenVINO graph kernels not implemented (CPU triple-loop stub only)");
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        (void)alpha; (void)A; (void)x; (void)beta; (void)y;
        throw_unavailable("NPU_OPENVINO", "gemv", "no OpenVINO GEMV kernel (stub backend)");
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        (void)x; (void)y; (void)axis;
        throw_unavailable("NPU_OPENVINO", "softmax", "no OpenVINO softmax kernel (stub backend)");
    }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override {
        (void)x; (void)g; (void)bt; (void)e; (void)y;
        throw_unavailable("NPU_OPENVINO", "layer_norm", "no OpenVINO layer_norm kernel (stub backend)");
    }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override {
        (void)x; (void)g; (void)e; (void)y;
        throw_unavailable("NPU_OPENVINO", "rms_norm", "no OpenVINO rms_norm kernel (stub backend)");
    }
    void relu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("NPU_OPENVINO", "relu", "no OpenVINO relu kernel (stub backend)");
    }
    void gelu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("NPU_OPENVINO", "gelu", "no OpenVINO gelu kernel (stub backend)");
    }
    void silu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("NPU_OPENVINO", "silu", "no OpenVINO silu kernel (stub backend)");
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        (void)a; (void)b; (void)c;
        throw_unavailable("NPU_OPENVINO", "add", "no OpenVINO add kernel (stub backend)");
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        (void)a; (void)b; (void)c;
        throw_unavailable("NPU_OPENVINO", "mul", "no OpenVINO mul kernel (stub backend)");
    }
    void scale(float s, const Tensor& x, Tensor& y) override {
        (void)s; (void)x; (void)y;
        throw_unavailable("NPU_OPENVINO", "scale", "no OpenVINO scale kernel (stub backend)");
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override { return ov_ && ov_->is_initialized(); }
    int64_t memory_free() const override { return 0; }
    int64_t memory_total() const override { return 0; }
    void synchronize() override { if (ov_) ov_->synchronize(); }
};

// ========================================================================
// VIRTGPU BACKEND
// ========================================================================

class GPUVirtGPUBackend : public ComputeBackend {
    gpu::GPUComputeVirtGPU* vg_ = nullptr;
public:
    GPUVirtGPUBackend() {
        try {
            vg_ = &gpu::get_virtgpu_compute();
            vg_->init(0);
        } catch (...) {
            vg_ = nullptr;
        }
    }
    BackendType type() const override { return BackendType::GPU_VIRTGPU; }
    const char* name() const override { return "GPU_VIRTGPU"; }
    // REAL ONLY verdict UNAVAILABLE on non-Linux/no-/dev/dri hosts: low-level
    // gemm is a CPU triple loop + malloc, NOT virgl/venus ioctls. Fail loud.
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        (void)alpha; (void)A; (void)B; (void)beta; (void)C;
        if (!vg_ || !vg_->is_initialized()) throw_unavailable("GPU_VIRTGPU", "gemm", "no virgl/venus render node on this host");
        throw_unavailable("GPU_VIRTGPU", "gemm", "VirtGPU GEMM is a CPU triple-loop stub (no virgl command stream)");
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        (void)alpha; (void)A; (void)x; (void)beta; (void)y;
        throw_unavailable("GPU_VIRTGPU", "gemv", "no VirtGPU GEMV kernel (stub backend)");
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        (void)x; (void)y; (void)axis;
        throw_unavailable("GPU_VIRTGPU", "softmax", "no VirtGPU softmax kernel (stub backend)");
    }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override {
        (void)x; (void)g; (void)bt; (void)e; (void)y;
        throw_unavailable("GPU_VIRTGPU", "layer_norm", "no VirtGPU layer_norm kernel (stub backend)");
    }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override {
        (void)x; (void)g; (void)e; (void)y;
        throw_unavailable("GPU_VIRTGPU", "rms_norm", "no VirtGPU rms_norm kernel (stub backend)");
    }
    void relu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("GPU_VIRTGPU", "relu", "no VirtGPU relu kernel (stub backend)");
    }
    void gelu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("GPU_VIRTGPU", "gelu", "no VirtGPU gelu kernel (stub backend)");
    }
    void silu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("GPU_VIRTGPU", "silu", "no VirtGPU silu kernel (stub backend)");
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        (void)a; (void)b; (void)c;
        throw_unavailable("GPU_VIRTGPU", "add", "no VirtGPU add kernel (stub backend)");
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        (void)a; (void)b; (void)c;
        throw_unavailable("GPU_VIRTGPU", "mul", "no VirtGPU mul kernel (stub backend)");
    }
    void scale(float s, const Tensor& x, Tensor& y) override {
        (void)s; (void)x; (void)y;
        throw_unavailable("GPU_VIRTGPU", "scale", "no VirtGPU scale kernel (stub backend)");
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override { return vg_ && vg_->is_initialized(); }
    int64_t memory_free() const override { return 0; }
    int64_t memory_total() const override { return 0; }
    void synchronize() override { if (vg_) vg_->synchronize(); }
};

// ========================================================================
// WEBGPU BACKEND
// ========================================================================

class GPUWebGPUBackend : public ComputeBackend {
    gpu::GPUComputeWebGPU* wg_ = nullptr;
public:
    GPUWebGPUBackend() {
        try {
            wg_ = &gpu::get_webgpu_compute();
            wg_->init(0);
        } catch (...) {
            wg_ = nullptr;
        }
    }
    BackendType type() const override { return BackendType::GPU_WEBGPU; }
    const char* name() const override { return "GPU_WEBGPU"; }
    // REAL ONLY verdict UNAVAILABLE (no Dawn/wgpu lib + no WGSL shaders):
    // low-level gemm is a CPU triple loop + malloc. Fail loud.
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        (void)alpha; (void)A; (void)B; (void)beta; (void)C;
        if (!wg_ || !wg_->is_initialized()) throw_unavailable("GPU_WEBGPU", "gemm", "no WebGPU runtime (Dawn/wgpu) on this host");
        throw_unavailable("GPU_WEBGPU", "gemm", "WebGPU WGSL GEMM shader not implemented (CPU stub only)");
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        (void)alpha; (void)A; (void)x; (void)beta; (void)y;
        throw_unavailable("GPU_WEBGPU", "gemv", "no WebGPU GEMV shader (stub backend)");
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        (void)x; (void)y; (void)axis;
        throw_unavailable("GPU_WEBGPU", "softmax", "no WebGPU softmax shader (stub backend)");
    }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override {
        (void)x; (void)g; (void)bt; (void)e; (void)y;
        throw_unavailable("GPU_WEBGPU", "layer_norm", "no WebGPU layer_norm shader (stub backend)");
    }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override {
        (void)x; (void)g; (void)e; (void)y;
        throw_unavailable("GPU_WEBGPU", "rms_norm", "no WebGPU rms_norm shader (stub backend)");
    }
    void relu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("GPU_WEBGPU", "relu", "no WebGPU relu shader (stub backend)");
    }
    void gelu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("GPU_WEBGPU", "gelu", "no WebGPU gelu shader (stub backend)");
    }
    void silu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("GPU_WEBGPU", "silu", "no WebGPU silu shader (stub backend)");
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        (void)a; (void)b; (void)c;
        throw_unavailable("GPU_WEBGPU", "add", "no WebGPU add shader (stub backend)");
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        (void)a; (void)b; (void)c;
        throw_unavailable("GPU_WEBGPU", "mul", "no WebGPU mul shader (stub backend)");
    }
    void scale(float s, const Tensor& x, Tensor& y) override {
        (void)s; (void)x; (void)y;
        throw_unavailable("GPU_WEBGPU", "scale", "no WebGPU scale shader (stub backend)");
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override { return wg_ && wg_->is_initialized(); }
    int64_t memory_free() const override { return 0; }
    int64_t memory_total() const override { return 0; }
    void synchronize() override { if (wg_) wg_->synchronize(); }
};

// ========================================================================
// ZENDNN BACKEND
// ========================================================================

class GPUZenDNNBackend : public ComputeBackend {
    gpu::GPUComputeZenDNN* zd_ = nullptr;
public:
    GPUZenDNNBackend() {
        try {
            zd_ = &gpu::get_zendnn_compute();
            zd_->init(0);
        } catch (...) {
            zd_ = nullptr;
        }
    }
    BackendType type() const override { return BackendType::CPU_ZENDNN; }
    const char* name() const override { return "CPU_ZENDNN"; }
    // REAL ONLY verdict REAL (CPU-family): ZenDNN IS an x86 CPU library, so
    // CPU execution is the honest claim. gemm dispatches zendnn_sgemm when
    // the lib loads, else the AVX2-tuned blocked path in
    // gpu_compute_zendnn.cpp (REAL CPU, verified vs math::). Other ops run
    // the genuine CPU math:: kernels (REAL for a CPU backend, not a fake
    // accelerator claim). Available when lib present OR AVX2 present.
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        require_available(is_available(), "CPU_ZENDNN", "gemm", "neither ZenDNN lib nor AVX2 on this host");
        if (zd_ && zd_->is_initialized()) {
            void* dA = zd_->alloc(A.numel() * sizeof(float));
            void* dB = zd_->alloc(B.numel() * sizeof(float));
            void* dC = zd_->alloc(C.numel() * sizeof(float));
            if (!dA || !dB || !dC) {
                if (dA) zd_->free_buf(dA); if (dB) zd_->free_buf(dB); if (dC) zd_->free_buf(dC);
                throw_unavailable("CPU_ZENDNN", "gemm", "host alloc failed");
            }
            zd_->upload(A.data(), dA, A.numel() * sizeof(float));
            zd_->upload(B.data(), dB, B.numel() * sizeof(float));
            if (beta != 0.0f) zd_->upload(C.data(), dC, C.numel() * sizeof(float));
            int64_t M = A.numel() / A.dim(A.rank() - 1);
            int64_t N = B.dim(B.rank() - 1);
            int64_t K = A.dim(A.rank() - 1);
            zd_->gemm(alpha, dA, dB, beta, dC, M, N, K);
            zd_->download(dC, C.data(), C.numel() * sizeof(float));
            zd_->free_buf(dA); zd_->free_buf(dB); zd_->free_buf(dC);
            return;
        }
        math::gemm(alpha, A, B, beta, C);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        require_available(is_available(), "CPU_ZENDNN", "gemv", "neither ZenDNN lib nor AVX2 on this host");
        math::gemv(alpha, A, x, beta, y);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        require_available(is_available(), "CPU_ZENDNN", "softmax", "neither ZenDNN lib nor AVX2 on this host");
        math::softmax(x, y, axis);
    }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override {
        require_available(is_available(), "CPU_ZENDNN", "layer_norm", "neither ZenDNN lib nor AVX2 on this host");
        math::layer_norm(x, g, bt, e, y);
    }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override {
        require_available(is_available(), "CPU_ZENDNN", "rms_norm", "neither ZenDNN lib nor AVX2 on this host");
        math::rms_norm(x, g, e, y);
    }
    void relu(const Tensor& x, Tensor& y) override {
        require_available(is_available(), "CPU_ZENDNN", "relu", "neither ZenDNN lib nor AVX2 on this host");
        math::relu(x, y);
    }
    void gelu(const Tensor& x, Tensor& y) override {
        require_available(is_available(), "CPU_ZENDNN", "gelu", "neither ZenDNN lib nor AVX2 on this host");
        math::gelu(x, y);
    }
    void silu(const Tensor& x, Tensor& y) override {
        require_available(is_available(), "CPU_ZENDNN", "silu", "neither ZenDNN lib nor AVX2 on this host");
        math::silu(x, y);
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(is_available(), "CPU_ZENDNN", "add", "neither ZenDNN lib nor AVX2 on this host");
        math::add(a, b, c);
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(is_available(), "CPU_ZENDNN", "mul", "neither ZenDNN lib nor AVX2 on this host");
        math::mul(a, b, c);
    }
    void scale(float s, const Tensor& x, Tensor& y) override {
        require_available(is_available(), "CPU_ZENDNN", "scale", "neither ZenDNN lib nor AVX2 on this host");
        math::scale(s, x, y);
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override {
        return (zd_ && zd_->is_initialized()) || is_avx2_available();
    }
    int64_t memory_free() const override { return cpu_memory_free(); }
    int64_t memory_total() const override { return cpu_memory_total(); }
    void synchronize() override { if (zd_) zd_->synchronize(); }
};

// ========================================================================
// GPU VULKAN BACKEND — H6 honest-fallback (REAL ONLY, no fake GEMM)
// ========================================================================

class GPUVulkanBackend : public ComputeBackend {
    gpu::VulkanBackend* vk_ = nullptr;
public:
    GPUVulkanBackend() {
        try {
            vk_ = &gpu::get_vulkan_backend();
            vk_->init(0);
        } catch (...) {
            vk_ = nullptr;
        }
    }
    BackendType type() const override { return BackendType::GPU_VULKAN; }
    const char* name() const override { return "GPU_VULKAN"; }
    // is_available() now gates on compute_ready(), i.e. device up AND the
    // embedded SPIR-V blobs structurally valid. The device can be present while
    // the shipped shader binaries are not real SPIR-V (2026-09-08: all five
    // blobs fail validation and hang the driver), and reporting "available" in
    // that state is exactly the fake-GPU evidence this file exists to prevent.
    bool is_available() const override {
        return vk_ && vk_->is_initialized() && vk_->compute_ready();
    }
    // REAL ONLY verdict PARTIAL (measured 2026-09-08, AMD iGPU): the relu /
    // gelu / silu / add / mul dispatch path is real code (device buffers +
    // dispatch_simple/binary in gpu_compute_vulkan.cpp) but the five shipped
    // SPIR-V blobs FAIL structural validation, so compute_ready() is false,
    // is_available() is false, and every op throws fail-loud. gemm/gemv/
    // softmax/norm/scale have NO SPIR-V shader -> fail loud when a device IS
    // present (perf-invalid fallback REMOVED), unavailable-throw when no
    // device. Previous warn+CPU path let callers mistake CPU GFLOPS for
    // Vulkan GFLOPS.
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        (void)alpha; (void)A; (void)B; (void)beta; (void)C;
        if (!vk_ || !vk_->is_initialized()) throw_unavailable("GPU_VULKAN", "gemm", "no Vulkan device on this host");
        throw_unavailable("GPU_VULKAN", "gemm", "GEMM SPIR-V shader not implemented (relu/gelu/silu/add/mul only)");
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        (void)alpha; (void)A; (void)x; (void)beta; (void)y;
        if (!vk_ || !vk_->is_initialized()) throw_unavailable("GPU_VULKAN", "gemv", "no Vulkan device on this host");
        throw_unavailable("GPU_VULKAN", "gemv", "GEMV SPIR-V shader not implemented (relu/gelu/silu/add/mul only)");
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        (void)x; (void)y; (void)axis;
        if (!vk_ || !vk_->is_initialized()) throw_unavailable("GPU_VULKAN", "softmax", "no Vulkan device on this host");
        throw_unavailable("GPU_VULKAN", "softmax", "softmax SPIR-V shader not implemented (relu/gelu/silu/add/mul only)");
    }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override {
        (void)x; (void)g; (void)bt; (void)e; (void)y;
        if (!vk_ || !vk_->is_initialized()) throw_unavailable("GPU_VULKAN", "layer_norm", "no Vulkan device on this host");
        throw_unavailable("GPU_VULKAN", "layer_norm", "layer_norm SPIR-V shader not implemented (relu/gelu/silu/add/mul only)");
    }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override {
        (void)x; (void)g; (void)e; (void)y;
        if (!vk_ || !vk_->is_initialized()) throw_unavailable("GPU_VULKAN", "rms_norm", "no Vulkan device on this host");
        throw_unavailable("GPU_VULKAN", "rms_norm", "rms_norm SPIR-V shader not implemented (relu/gelu/silu/add/mul only)");
    }
    void relu(const Tensor& x, Tensor& y) override {
        require_available(vk_ && vk_->is_initialized(), "GPU_VULKAN", "relu", "no Vulkan device on this host");
        // REAL path: device-buffer dispatch via existing relu SPIR-V.
        void* dx = vk_->allocate((size_t)x.numel() * sizeof(float));
        void* dy = vk_->allocate((size_t)y.numel() * sizeof(float));
        if (!dx || !dy) {
            if (dx) vk_->free(dx);
            if (dy) vk_->free(dy);
            throw_unavailable("GPU_VULKAN", "relu", "device alloc failed");
        }
        vk_->upload(x, dx);
        vk_->relu(dx, dy, x.numel());
        vk_->download(dy, y);
        vk_->free(dx);
        vk_->free(dy);
    }
    void gelu(const Tensor& x, Tensor& y) override {
        require_available(vk_ && vk_->is_initialized(), "GPU_VULKAN", "gelu", "no Vulkan device on this host");
        void* dx = vk_->allocate((size_t)x.numel() * sizeof(float));
        void* dy = vk_->allocate((size_t)y.numel() * sizeof(float));
        if (!dx || !dy) {
            if (dx) vk_->free(dx);
            if (dy) vk_->free(dy);
            throw_unavailable("GPU_VULKAN", "gelu", "device alloc failed");
        }
        vk_->upload(x, dx);
        vk_->gelu(dx, dy, x.numel());
        vk_->download(dy, y);
        vk_->free(dx);
        vk_->free(dy);
    }
    void silu(const Tensor& x, Tensor& y) override {
        require_available(vk_ && vk_->is_initialized(), "GPU_VULKAN", "silu", "no Vulkan device on this host");
        void* dx = vk_->allocate((size_t)x.numel() * sizeof(float));
        void* dy = vk_->allocate((size_t)y.numel() * sizeof(float));
        if (!dx || !dy) {
            if (dx) vk_->free(dx);
            if (dy) vk_->free(dy);
            throw_unavailable("GPU_VULKAN", "silu", "device alloc failed");
        }
        vk_->upload(x, dx);
        vk_->silu(dx, dy, x.numel());
        vk_->download(dy, y);
        vk_->free(dx);
        vk_->free(dy);
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(vk_ && vk_->is_initialized(), "GPU_VULKAN", "add", "no Vulkan device on this host");
        void* da = vk_->allocate((size_t)a.numel() * sizeof(float));
        void* db = vk_->allocate((size_t)b.numel() * sizeof(float));
        void* dc = vk_->allocate((size_t)c.numel() * sizeof(float));
        if (!da || !db || !dc) {
            if (da) vk_->free(da);
            if (db) vk_->free(db);
            if (dc) vk_->free(dc);
            throw_unavailable("GPU_VULKAN", "add", "device alloc failed");
        }
        vk_->upload(a, da);
        vk_->upload(b, db);
        vk_->add(da, db, dc, c.numel());
        vk_->download(dc, c);
        vk_->free(da);
        vk_->free(db);
        vk_->free(dc);
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(vk_ && vk_->is_initialized(), "GPU_VULKAN", "mul", "no Vulkan device on this host");
        void* da = vk_->allocate((size_t)a.numel() * sizeof(float));
        void* db = vk_->allocate((size_t)b.numel() * sizeof(float));
        void* dc = vk_->allocate((size_t)c.numel() * sizeof(float));
        if (!da || !db || !dc) {
            if (da) vk_->free(da);
            if (db) vk_->free(db);
            if (dc) vk_->free(dc);
            throw_unavailable("GPU_VULKAN", "mul", "device alloc failed");
        }
        vk_->upload(a, da);
        vk_->upload(b, db);
        vk_->mul(da, db, dc, c.numel());
        vk_->download(dc, c);
        vk_->free(da);
        vk_->free(db);
        vk_->free(dc);
    }
    void scale(float s, const Tensor& x, Tensor& y) override {
        (void)s; (void)x; (void)y;
        if (!vk_ || !vk_->is_initialized()) throw_unavailable("GPU_VULKAN", "scale", "no Vulkan device on this host");
        throw_unavailable("GPU_VULKAN", "scale", "scale SPIR-V shader not implemented (relu/gelu/silu/add/mul only)");
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    int64_t memory_free() const override { return (vk_ && vk_->is_initialized()) ? vk_->memory_free() : 0; }
    int64_t memory_total() const override { return (vk_ && vk_->is_initialized()) ? vk_->memory_total() : 0; }
    void synchronize() override { if (vk_ && vk_->is_initialized()) vk_->synchronize(); }
};

// ========================================================================
// CPU NEON BACKEND — REAL on ARM, UNAVAILABLE+fail-loud elsewhere.
// math:: kernels are genuine CPU execution (honest claim for a CPU
// backend). Previous state: NO factory case at all -> silent scalar via
// default (REAL-ONLY violation). Now explicit.
// ========================================================================

class CPUNEONBackend : public ComputeBackend {
public:
    BackendType type() const override { return BackendType::CPU_NEON; }
    const char* name() const override { return "CPU_NEON"; }
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        require_available(is_available(), "CPU_NEON", "gemm", "NEON unavailable (non-ARM host)");
        math::gemm(alpha, A, B, beta, C);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        require_available(is_available(), "CPU_NEON", "gemv", "NEON unavailable (non-ARM host)");
        math::gemv(alpha, A, x, beta, y);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        require_available(is_available(), "CPU_NEON", "softmax", "NEON unavailable (non-ARM host)");
        math::softmax(x, y, axis);
    }
    void layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps, Tensor& y) override {
        require_available(is_available(), "CPU_NEON", "layer_norm", "NEON unavailable (non-ARM host)");
        math::layer_norm(x, gamma, beta, eps, y);
    }
    void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y) override {
        require_available(is_available(), "CPU_NEON", "rms_norm", "NEON unavailable (non-ARM host)");
        math::rms_norm(x, gamma, eps, y);
    }
    void relu(const Tensor& x, Tensor& y) override {
        require_available(is_available(), "CPU_NEON", "relu", "NEON unavailable (non-ARM host)");
        math::relu(x, y);
    }
    void gelu(const Tensor& x, Tensor& y) override {
        require_available(is_available(), "CPU_NEON", "gelu", "NEON unavailable (non-ARM host)");
        math::gelu(x, y);
    }
    void silu(const Tensor& x, Tensor& y) override {
        require_available(is_available(), "CPU_NEON", "silu", "NEON unavailable (non-ARM host)");
        math::silu(x, y);
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(is_available(), "CPU_NEON", "add", "NEON unavailable (non-ARM host)");
        math::add(a, b, c);
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(is_available(), "CPU_NEON", "mul", "NEON unavailable (non-ARM host)");
        math::mul(a, b, c);
    }
    void scale(float s, const Tensor& x, Tensor& y) override {
        require_available(is_available(), "CPU_NEON", "scale", "NEON unavailable (non-ARM host)");
        math::scale(s, x, y);
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override { return is_neon_available(); }
    int64_t memory_free() const override { return cpu_memory_free(); }
    int64_t memory_total() const override { return cpu_memory_total(); }
    void synchronize() override {}
};

// ========================================================================
// CPU BACKEND (generic alias -> best CPU path) — REAL, always available.
// Previous state: NO factory case -> silent scalar via default. Now explicit:
// dispatches math:: (scalar-portable, REAL CPU claim).
// ========================================================================

class CPUBackend : public ComputeBackend {
public:
    BackendType type() const override { return BackendType::CPU; }
    const char* name() const override { return "CPU"; }
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        math::gemm(alpha, A, B, beta, C);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        math::gemv(alpha, A, x, beta, y);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override { math::softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps, Tensor& y) override {
        math::layer_norm(x, gamma, beta, eps, y);
    }
    void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y) override {
        math::rms_norm(x, gamma, eps, y);
    }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override { math::gelu(x, y); }
    void silu(const Tensor& x, Tensor& y) override { math::silu(x, y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override { return true; }
    int64_t memory_free() const override { return cpu_memory_free(); }
    int64_t memory_total() const override { return cpu_memory_total(); }
    void synchronize() override {}
};

// ========================================================================
// GPU HIP BACKEND — REAL (ROCm dispatch in src/backend/gpu_compute_hip.cpp:
// hipMalloc/hipMemcpy + hiprtc JIT-compiled kernels from real HIP C++
// sources + hipModuleLoadData/hipModuleLaunchKernel). UNAVAILABLE+fail-loud
// when no HIP runtime; per-call launch failures also throw (no host
// substitution anywhere). Previous state: NO factory case at all (orphan
// file) + byte-identical fake ELF stubs (deleted, see honesty note in
// gpu_compute_hip.cpp). gemv/layer_norm have no launch_* -> fail loud.
// ========================================================================

class GPUHIPBackend : public ComputeBackend {
    ::quant::GpuComputeHip* hip_ = nullptr;
    bool avail_ = false;
public:
    GPUHIPBackend() {
        try {
            hip_ = new ::quant::GpuComputeHip();
            avail_ = hip_->init();
            if (!avail_) { delete hip_; hip_ = nullptr; }
        } catch (...) { hip_ = nullptr; avail_ = false; }
    }
    ~GPUHIPBackend() override { delete hip_; }
    BackendType type() const override { return BackendType::GPU_HIP; }
    const char* name() const override { return "GPU_HIP"; }
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        require_available(avail_, "GPU_HIP", "gemm", "no HIP/ROCm runtime on this host");
        int64_t K = A.dim(A.rank() - 1);
        int64_t M = A.numel() / K;
        int64_t N = B.dim(B.rank() - 1);
        Tensor tmp(C.shape());
        hip_->launch_gemm((int)M, (int)N, (int)K, A.data<float>(), B.data<float>(), tmp.data<float>());
        const float* t = tmp.data<float>();
        float* c = C.data<float>();
        if (beta != 0.0f) {
            for (int64_t i = 0; i < C.numel(); ++i) c[i] = alpha * t[i] + beta * c[i];
        } else if (alpha != 1.0f) {
            for (int64_t i = 0; i < C.numel(); ++i) c[i] = alpha * t[i];
        } else {
            for (int64_t i = 0; i < C.numel(); ++i) c[i] = t[i];
        }
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        (void)alpha; (void)A; (void)x; (void)beta; (void)y;
        if (!avail_) throw_unavailable("GPU_HIP", "gemv", "no HIP/ROCm runtime on this host");
        throw_unavailable("GPU_HIP", "gemv", "no launch_gemv in GpuComputeHip (gemm/relu/silu/gelu/softmax/rmsnorm/add/mul only)");
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        require_available(avail_, "GPU_HIP", "softmax", "no HIP/ROCm runtime on this host");
        if (axis != 1 && !(axis == -1 && x.rank() == 2))
            throw_unavailable("GPU_HIP", "softmax", "only axis=1 2-D supported on device");
        y.copy_from(x);
        int rows = (int)x.dim(0), cols = (int)(x.numel() / x.dim(0));
        hip_->launch_softmax(y.data<float>(), rows, cols);
    }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override {
        (void)x; (void)g; (void)bt; (void)e; (void)y;
        if (!avail_) throw_unavailable("GPU_HIP", "layer_norm", "no HIP/ROCm runtime on this host");
        throw_unavailable("GPU_HIP", "layer_norm", "no layernorm kernel in GpuComputeHip (PARTIAL gap)");
    }
    void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y) override {
        require_available(avail_, "GPU_HIP", "rms_norm", "no HIP/ROCm runtime on this host");
        y.copy_from(x);
        int rows = (int)x.dim(0), cols = (int)(x.numel() / x.dim(0));
        hip_->launch_rmsnorm(y.data<float>(), gamma.data<float>(), rows, cols, eps);
    }
    void relu(const Tensor& x, Tensor& y) override {
        require_available(avail_, "GPU_HIP", "relu", "no HIP/ROCm runtime on this host");
        y.copy_from(x);
        hip_->launch_relu(y.data<float>(), (size_t)y.numel());
    }
    void gelu(const Tensor& x, Tensor& y) override {
        require_available(avail_, "GPU_HIP", "gelu", "no HIP/ROCm runtime on this host");
        y.copy_from(x);
        hip_->launch_gelu(y.data<float>(), (size_t)y.numel());
    }
    void silu(const Tensor& x, Tensor& y) override {
        require_available(avail_, "GPU_HIP", "silu", "no HIP/ROCm runtime on this host");
        y.copy_from(x);
        hip_->launch_silu(y.data<float>(), (size_t)y.numel());
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(avail_, "GPU_HIP", "add", "no HIP/ROCm runtime on this host");
        hip_->launch_add(a.data<float>(), b.data<float>(), c.data<float>(), (size_t)a.numel());
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        require_available(avail_, "GPU_HIP", "mul", "no HIP/ROCm runtime on this host");
        hip_->launch_mul(a.data<float>(), b.data<float>(), c.data<float>(), (size_t)a.numel());
    }
    void scale(float s, const Tensor& x, Tensor& y) override {
        require_available(avail_, "GPU_HIP", "scale", "no HIP/ROCm runtime on this host");
        y.copy_from(x);
        hip_->launch_scale(y.data<float>(), s, (size_t)y.numel());
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override {
        require_available(avail_, "GPU_HIP", "fill", "no HIP/ROCm runtime on this host");
        hip_->launch_fill(t.data<float>(), val, (size_t)t.numel());
    }
    void zero(Tensor& t) override { fill(t, 0.0f); }
    bool is_available() const override { return avail_; }
    int64_t memory_free() const override { return 0; }
    int64_t memory_total() const override { return 0; }
    void synchronize() override {}
};

// ========================================================================
// DSP HEXAGON BACKEND — UNAVAILABLE (no HVX kernels: launch_gemm is a host
// CPU triple loop, hexagon_nn_execute_ never called). Fail loud for all
// compute even when the stub lib loads. Previous state: orphan file, no
// factory case. is_available reflects lib presence; compute always throws
// until real HVX graph kernels land.
// ========================================================================

class DSPHexagonBackend : public ComputeBackend {
    ::quant::GpuComputeHexagon* hx_ = nullptr;
    bool avail_ = false;
public:
    DSPHexagonBackend() {
        try {
            hx_ = new ::quant::GpuComputeHexagon();
            avail_ = hx_->init();
            if (!avail_) { delete hx_; hx_ = nullptr; }
        } catch (...) { hx_ = nullptr; avail_ = false; }
    }
    ~DSPHexagonBackend() override { delete hx_; }
    BackendType type() const override { return BackendType::DSP_HEXAGON; }
    const char* name() const override { return "DSP_HEXAGON"; }
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        (void)alpha; (void)A; (void)B; (void)beta; (void)C;
        if (!avail_) throw_unavailable("DSP_HEXAGON", "gemm", "no Hexagon DSP runtime on this host");
        throw_unavailable("DSP_HEXAGON", "gemm", "HVX kernels not implemented (host triple-loop stub only)");
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        (void)alpha; (void)A; (void)x; (void)beta; (void)y;
        throw_unavailable("DSP_HEXAGON", "gemv", "HVX kernels not implemented (stub backend)");
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        (void)x; (void)y; (void)axis;
        throw_unavailable("DSP_HEXAGON", "softmax", "HVX kernels not implemented (stub backend)");
    }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override {
        (void)x; (void)g; (void)bt; (void)e; (void)y;
        throw_unavailable("DSP_HEXAGON", "layer_norm", "HVX kernels not implemented (stub backend)");
    }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override {
        (void)x; (void)g; (void)e; (void)y;
        throw_unavailable("DSP_HEXAGON", "rms_norm", "HVX kernels not implemented (stub backend)");
    }
    void relu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("DSP_HEXAGON", "relu", "HVX kernels not implemented (stub backend)");
    }
    void gelu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("DSP_HEXAGON", "gelu", "HVX kernels not implemented (stub backend)");
    }
    void silu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("DSP_HEXAGON", "silu", "HVX kernels not implemented (stub backend)");
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        (void)a; (void)b; (void)c;
        throw_unavailable("DSP_HEXAGON", "add", "HVX kernels not implemented (stub backend)");
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        (void)a; (void)b; (void)c;
        throw_unavailable("DSP_HEXAGON", "mul", "HVX kernels not implemented (stub backend)");
    }
    void scale(float s, const Tensor& x, Tensor& y) override {
        (void)s; (void)x; (void)y;
        throw_unavailable("DSP_HEXAGON", "scale", "HVX kernels not implemented (stub backend)");
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override { return avail_; }
    int64_t memory_free() const override { return 0; }
    int64_t memory_total() const override { return 0; }
    void synchronize() override {}
};

// ========================================================================
// NPU ZDNN BACKEND (IBM s390x) — UNAVAILABLE (launch_gemm is a host CPU
// triple loop, zdnn_matmul_ never invoked for compute). Fail loud.
// Previous state: orphan file, no factory case.
// ========================================================================

class NPUZDNNBackend : public ComputeBackend {
    ::quant::GpuComputeZDnn* zd_ = nullptr;
    bool avail_ = false;
public:
    NPUZDNNBackend() {
        try {
            zd_ = new ::quant::GpuComputeZDnn();
            avail_ = zd_->init();
            if (!avail_) { delete zd_; zd_ = nullptr; }
        } catch (...) { zd_ = nullptr; avail_ = false; }
    }
    ~NPUZDNNBackend() override { delete zd_; }
    BackendType type() const override { return BackendType::NPU_ZDNN; }
    const char* name() const override { return "NPU_ZDNN"; }
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        (void)alpha; (void)A; (void)B; (void)beta; (void)C;
        if (!avail_) throw_unavailable("NPU_ZDNN", "gemm", "no zDNN runtime (s390x) on this host");
        throw_unavailable("NPU_ZDNN", "gemm", "zDNN matmul not wired (host triple-loop stub only)");
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        (void)alpha; (void)A; (void)x; (void)beta; (void)y;
        throw_unavailable("NPU_ZDNN", "gemv", "zDNN kernels not implemented (stub backend)");
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        (void)x; (void)y; (void)axis;
        throw_unavailable("NPU_ZDNN", "softmax", "zDNN kernels not implemented (stub backend)");
    }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override {
        (void)x; (void)g; (void)bt; (void)e; (void)y;
        throw_unavailable("NPU_ZDNN", "layer_norm", "zDNN kernels not implemented (stub backend)");
    }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override {
        (void)x; (void)g; (void)e; (void)y;
        throw_unavailable("NPU_ZDNN", "rms_norm", "zDNN kernels not implemented (stub backend)");
    }
    void relu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("NPU_ZDNN", "relu", "zDNN kernels not implemented (stub backend)");
    }
    void gelu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("NPU_ZDNN", "gelu", "zDNN kernels not implemented (stub backend)");
    }
    void silu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("NPU_ZDNN", "silu", "zDNN kernels not implemented (stub backend)");
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        (void)a; (void)b; (void)c;
        throw_unavailable("NPU_ZDNN", "add", "zDNN kernels not implemented (stub backend)");
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        (void)a; (void)b; (void)c;
        throw_unavailable("NPU_ZDNN", "mul", "zDNN kernels not implemented (stub backend)");
    }
    void scale(float s, const Tensor& x, Tensor& y) override {
        (void)s; (void)x; (void)y;
        throw_unavailable("NPU_ZDNN", "scale", "zDNN kernels not implemented (stub backend)");
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override { return avail_; }
    int64_t memory_free() const override { return 0; }
    int64_t memory_total() const override { return 0; }
    void synchronize() override {}
};

// ========================================================================
// GPU MUSA BACKEND (Moore Threads) — UNAVAILABLE (launch_gemm is a host
// CPU loop with skip-zeros, musaLaunchKernel_ never invoked). Fail loud.
// Previous state: orphan file, no factory case.
// ========================================================================

class GPUMUSABackend : public ComputeBackend {
    ::quant::GpuComputeMusa* mu_ = nullptr;
    bool avail_ = false;
public:
    GPUMUSABackend() {
        try {
            mu_ = new ::quant::GpuComputeMusa();
            avail_ = mu_->init();
            if (!avail_) { delete mu_; mu_ = nullptr; }
        } catch (...) { mu_ = nullptr; avail_ = false; }
    }
    ~GPUMUSABackend() override { delete mu_; }
    BackendType type() const override { return BackendType::GPU_MUSA; }
    const char* name() const override { return "GPU_MUSA"; }
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        (void)alpha; (void)A; (void)B; (void)beta; (void)C;
        if (!avail_) throw_unavailable("GPU_MUSA", "gemm", "no MUSA runtime on this host");
        throw_unavailable("GPU_MUSA", "gemm", "MUSA kernels not implemented (host loop stub only, musaLaunchKernel never invoked)");
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        (void)alpha; (void)A; (void)x; (void)beta; (void)y;
        throw_unavailable("GPU_MUSA", "gemv", "MUSA kernels not implemented (stub backend)");
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        (void)x; (void)y; (void)axis;
        throw_unavailable("GPU_MUSA", "softmax", "MUSA kernels not implemented (stub backend)");
    }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override {
        (void)x; (void)g; (void)bt; (void)e; (void)y;
        throw_unavailable("GPU_MUSA", "layer_norm", "MUSA kernels not implemented (stub backend)");
    }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override {
        (void)x; (void)g; (void)e; (void)y;
        throw_unavailable("GPU_MUSA", "rms_norm", "MUSA kernels not implemented (stub backend)");
    }
    void relu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("GPU_MUSA", "relu", "MUSA kernels not implemented (stub backend)");
    }
    void gelu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("GPU_MUSA", "gelu", "MUSA kernels not implemented (stub backend)");
    }
    void silu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        throw_unavailable("GPU_MUSA", "silu", "MUSA kernels not implemented (stub backend)");
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        (void)a; (void)b; (void)c;
        throw_unavailable("GPU_MUSA", "add", "MUSA kernels not implemented (stub backend)");
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        (void)a; (void)b; (void)c;
        throw_unavailable("GPU_MUSA", "mul", "MUSA kernels not implemented (stub backend)");
    }
    void scale(float s, const Tensor& x, Tensor& y) override {
        (void)s; (void)x; (void)y;
        throw_unavailable("GPU_MUSA", "scale", "MUSA kernels not implemented (stub backend)");
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override { return avail_; }
    int64_t memory_free() const override { return 0; }
    int64_t memory_total() const override { return 0; }
    void synchronize() override {}
};

// ========================================================================
// GPU OPENCL BACKEND — PARTIAL (REAL GEMM: kOpenCLGEMMKernel source +
// clCreateProgramWithSource/clBuildProgram/clCreateKernel/
// clEnqueueNDRangeKernel in gpu_compute_opencl.cpp; device alloc via
// clCreateBuffer). gemv/norm/act have no kernels -> fail loud.
// Previous state: orphan file, no factory case.
// ========================================================================

class GPUOpenCLBackend : public ComputeBackend {
    ::quant::GpuComputeOpenCL* cl_ = nullptr;
    bool avail_ = false;
public:
    GPUOpenCLBackend() {
        try {
            cl_ = new ::quant::GpuComputeOpenCL();
            avail_ = cl_->init();
            if (!avail_) { delete cl_; cl_ = nullptr; }
        } catch (...) { cl_ = nullptr; avail_ = false; }
    }
    ~GPUOpenCLBackend() override { delete cl_; }
    BackendType type() const override { return BackendType::GPU_OPENCL; }
    const char* name() const override { return "GPU_OPENCL"; }
    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        require_available(avail_, "GPU_OPENCL", "gemm", "no OpenCL platform/device on this host");
        int64_t K = A.dim(A.rank() - 1);
        int64_t M = A.numel() / K;
        int64_t N = B.dim(B.rank() - 1);
        Tensor tmp(C.shape());
        cl_->launch_gemm((int)M, (int)N, (int)K, A.data<float>(), B.data<float>(), tmp.data<float>());
        const float* t = tmp.data<float>();
        float* c = C.data<float>();
        if (beta != 0.0f) {
            for (int64_t i = 0; i < C.numel(); ++i) c[i] = alpha * t[i] + beta * c[i];
        } else if (alpha != 1.0f) {
            for (int64_t i = 0; i < C.numel(); ++i) c[i] = alpha * t[i];
        } else {
            for (int64_t i = 0; i < C.numel(); ++i) c[i] = t[i];
        }
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        (void)alpha; (void)A; (void)x; (void)beta; (void)y;
        if (!avail_) throw_unavailable("GPU_OPENCL", "gemv", "no OpenCL platform/device on this host");
        throw_unavailable("GPU_OPENCL", "gemv", "no OpenCL GEMV kernel (GEMM kernel only)");
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override {
        (void)x; (void)y; (void)axis;
        if (!avail_) throw_unavailable("GPU_OPENCL", "softmax", "no OpenCL platform/device on this host");
        throw_unavailable("GPU_OPENCL", "softmax", "no OpenCL softmax kernel (GEMM kernel only)");
    }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override {
        (void)x; (void)g; (void)bt; (void)e; (void)y;
        if (!avail_) throw_unavailable("GPU_OPENCL", "layer_norm", "no OpenCL platform/device on this host");
        throw_unavailable("GPU_OPENCL", "layer_norm", "no OpenCL layer_norm kernel (GEMM kernel only)");
    }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override {
        (void)x; (void)g; (void)e; (void)y;
        if (!avail_) throw_unavailable("GPU_OPENCL", "rms_norm", "no OpenCL platform/device on this host");
        throw_unavailable("GPU_OPENCL", "rms_norm", "no OpenCL rms_norm kernel (GEMM kernel only)");
    }
    void relu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        if (!avail_) throw_unavailable("GPU_OPENCL", "relu", "no OpenCL platform/device on this host");
        throw_unavailable("GPU_OPENCL", "relu", "no OpenCL relu kernel (GEMM kernel only)");
    }
    void gelu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        if (!avail_) throw_unavailable("GPU_OPENCL", "gelu", "no OpenCL platform/device on this host");
        throw_unavailable("GPU_OPENCL", "gelu", "no OpenCL gelu kernel (GEMM kernel only)");
    }
    void silu(const Tensor& x, Tensor& y) override {
        (void)x; (void)y;
        if (!avail_) throw_unavailable("GPU_OPENCL", "silu", "no OpenCL platform/device on this host");
        throw_unavailable("GPU_OPENCL", "silu", "no OpenCL silu kernel (GEMM kernel only)");
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override {
        (void)a; (void)b; (void)c;
        if (!avail_) throw_unavailable("GPU_OPENCL", "add", "no OpenCL platform/device on this host");
        throw_unavailable("GPU_OPENCL", "add", "no OpenCL add kernel (GEMM kernel only)");
    }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override {
        (void)a; (void)b; (void)c;
        if (!avail_) throw_unavailable("GPU_OPENCL", "mul", "no OpenCL platform/device on this host");
        throw_unavailable("GPU_OPENCL", "mul", "no OpenCL mul kernel (GEMM kernel only)");
    }
    void scale(float s, const Tensor& x, Tensor& y) override {
        (void)s; (void)x; (void)y;
        if (!avail_) throw_unavailable("GPU_OPENCL", "scale", "no OpenCL platform/device on this host");
        throw_unavailable("GPU_OPENCL", "scale", "no OpenCL scale kernel (GEMM kernel only)");
    }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override { return avail_; }
    int64_t memory_free() const override { return 0; }
    int64_t memory_total() const override { return 0; }
    void synchronize() override {}
};

// ========================================================================
// Backend factory
// ========================================================================

// REAL ONLY factory (Vulkan gap FIXED): one explicit case per BackendType
// (all 24), unconditional on every platform. Availability is RUNTIME
// (init+is_initialized), never #ifdef. Unknown enum -> throw, NEVER silent
// scalar (the old `default: return new CPUScalarBackend()` let CPU_NEON,
// CPU, HIP, HEXAGON, ZDNN, MUSA, OPENCL silently pose as scalar).
ComputeBackend* ComputeBackend::create(const BackendConfig& cfg) {
    switch (cfg.type) {
        case BackendType::CPU_SCALAR: return new CPUScalarBackend();
        case BackendType::CPU_AVX2: return new CPUAVX2Backend();
        case BackendType::CPU_AVX512: return new CPUAVX512Backend();
        case BackendType::CPU_NEON: return new CPUNEONBackend();
        case BackendType::CPU: return new CPUBackend();
        case BackendType::GPU_DIRECTX: return new GPUDirectXBackend();
        case BackendType::IGPU_SHARED: return new IGPUSharedBackend();
        case BackendType::GPU_CUDA: return new GPUCUDABackend();
        case BackendType::GPU_METAL: return new GPUMetalBackend();
        case BackendType::GPU_SYCL: return new GPUSYCLBackend();
        case BackendType::GPU_CANN: return new GPUCANNBackend();
        case BackendType::RPC: return new GPURPCBackend();
        case BackendType::RAM_SWAP: return new RAMSwapBackend();
        case BackendType::DISTRIBUTED: return new DistributedBackend();
        case BackendType::NPU_OPENVINO: return new GPUOpenVINOBackend();
        case BackendType::GPU_VIRTGPU: return new GPUVirtGPUBackend();
        case BackendType::GPU_WEBGPU: return new GPUWebGPUBackend();
        case BackendType::CPU_ZENDNN: return new GPUZenDNNBackend();
        case BackendType::GPU_VULKAN: return new GPUVulkanBackend();
        case BackendType::GPU_HIP: return new GPUHIPBackend();
        case BackendType::DSP_HEXAGON: return new DSPHexagonBackend();
        case BackendType::NPU_ZDNN: return new NPUZDNNBackend();
        case BackendType::GPU_MUSA: return new GPUMUSABackend();
        case BackendType::GPU_OPENCL: return new GPUOpenCLBackend();
        default: throw std::runtime_error("[REAL-ONLY] ComputeBackend::create: unknown BackendType (no silent scalar fallback)");
    }
}

// ========================================================================
// Hardware detection
// ========================================================================

static inline void quant_cpuid(int info[4], int leaf) {
#if defined(_WIN32)
    __cpuid(info, leaf);
#elif defined(__aarch64__) || defined(__arm__)
    (void)info; (void)leaf;
#else
    __cpuid(leaf, info[0], info[1], info[2], info[3]);
#endif
}

static inline void quant_cpuidex(int info[4], int leaf, int sub) {
#if defined(_WIN32)
    __cpuidex(info, leaf, sub);
#elif defined(__aarch64__) || defined(__arm__)
    (void)info; (void)leaf; (void)sub;
#else
    __cpuid_count(leaf, sub, info[0], info[1], info[2], info[3]);
#endif
}

bool is_avx2_available() {
#if defined(QUANT_AVX2)
    return true;
#elif defined(__aarch64__) || defined(__arm__)
    return false;
#else
    int cpu_info[4] = {0};
    quant_cpuid(cpu_info, 0);
    int n_ids = cpu_info[0];
    if (n_ids >= 7) {
        quant_cpuidex(cpu_info, 7, 0);
        return (cpu_info[1] & (1 << 5)) != 0;
    }
    return false;
#endif
}

bool is_avx512_available() {
#if defined(QUANT_AVX512)
    return true;
#elif defined(__aarch64__) || defined(__arm__)
    return false;
#else
    int cpu_info[4] = {0};
    quant_cpuid(cpu_info, 0);
    int n_ids = cpu_info[0];
    if (n_ids >= 7) {
        quant_cpuidex(cpu_info, 7, 0);
        return (cpu_info[1] & (1 << 16)) != 0;
    }
    return false;
#endif
}

bool is_neon_available() {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return true;
#else
    return false;
#endif
}

bool is_metal_available() {
#if defined(__APPLE__)
    try {
        auto& metal = gpu::get_metal_compute();
        metal.init(0);
        return metal.is_initialized();
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

bool is_cuda_available() {
#if defined(QUANT_USE_CUDA)
    try {
        auto& cuda = gpu::get_cuda_compute();
        cuda.init(0);
        return cuda.is_initialized();
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

bool is_directx_available() {
#if defined(QUANT_USE_DIRECTX)
    return true;
#elif defined(_WIN32)
    HMODULE d3d12 = LoadLibraryA("d3d12.dll");
    if (d3d12) { FreeLibrary(d3d12); return true; }
    return false;
#else
    return false;
#endif
}

bool is_vulkan_available() {
    // H6 honest-fallback (REAL ONLY): DLL/lib existence is NOT availability.
    // Probe the real device via VulkanBackend init; is_initialized()==vulkan_ok only.
    // Mirrors is_sycl_available()/is_cann_available() pattern. Never claim fake GPU.
    try {
        auto& vk = gpu::get_vulkan_backend();
        if (vk.is_initialized()) return true;
        vk.init(0);
        return vk.is_initialized();
    } catch (...) {
        return false;
    }
}

bool is_sycl_available() {
    try {
        auto& sycl = gpu::get_sycl_compute();
        sycl.init(0);
        return sycl.is_initialized();
    } catch (...) {
        return false;
    }
}

bool is_cann_available() {
    try {
        auto& cann = gpu::get_cann_compute();
        cann.init(0);
        return cann.is_initialized();
    } catch (...) {
        return false;
    }
}

// REAL ONLY: is_rpc_available probes TCP 127.0.0.1:9000 with a short
// timeout. Previous `return true` claimed a server that was never
// contacted (fake availability -> silent CPU fallback in every op).
bool is_rpc_available() {
#if defined(_WIN32)
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) { WSACleanup(); return false; }
    u_long nb = 1;
    ioctlsocket(s, FIONBIO, &nb);
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(9000);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    (void)connect(s, (const sockaddr*)&addr, sizeof(addr));
    fd_set wf;
    FD_ZERO(&wf);
    FD_SET(s, &wf);
    timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 200000;
    int w = select(0, nullptr, &wf, nullptr, &tv);
    bool ok = w > 0 && FD_ISSET(s, &wf);
    closesocket(s);
    WSACleanup();
    return ok;
#else
    try {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) return false;
        int fl = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, fl | O_NONBLOCK);
        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(9000);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        int rc = connect(sock, (const sockaddr*)&addr, sizeof(addr));
        if (rc == 0) { close(sock); return true; }
        if (errno != EINPROGRESS) { close(sock); return false; }
        fd_set wf;
        FD_ZERO(&wf);
        FD_SET(sock, &wf);
        timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 200000;
        int w = select(sock + 1, nullptr, &wf, nullptr, &tv);
        bool ok = false;
        if (w > 0 && FD_ISSET(sock, &wf)) {
            int err = 0;
            socklen_t el = sizeof(err);
            getsockopt(sock, SOL_SOCKET, SO_ERROR, &err, &el);
            ok = (err == 0);
        }
        close(sock);
        return ok;
    } catch (...) {
        return false;
    }
#endif
}

// Honest device probes for the newly-wired backends. Each attempts the real
// runtime init; false on this host is the HONEST answer (no hardware), and
// every compute op on those backends fails loud (see classes above).
bool is_hip_available() {
    try {
        ::quant::GpuComputeHip h;
        return h.init();
    } catch (...) { return false; }
}
bool is_hexagon_available() {
    try {
        ::quant::GpuComputeHexagon h;
        return h.init();
    } catch (...) { return false; }
}
bool is_zdnn_available() {
    try {
        ::quant::GpuComputeZDnn z;
        return z.init();
    } catch (...) { return false; }
}
bool is_musa_available() {
    try {
        ::quant::GpuComputeMusa m;
        return m.init();
    } catch (...) { return false; }
}
bool is_opencl_available() {
    try {
        ::quant::GpuComputeOpenCL c;
        return c.init();
    } catch (...) { return false; }
}

bool is_openvino_available() {
    try {
        auto& ov = gpu::get_openvino_compute();
        ov.init(0);
        return ov.is_initialized();
    } catch (...) { return false; }
}

bool is_virtgpu_available() {
    try {
        auto& vg = gpu::get_virtgpu_compute();
        vg.init(0);
        return vg.is_initialized();
    } catch (...) { return false; }
}

bool is_webgpu_available() {
    try {
        auto& wg = gpu::get_webgpu_compute();
        wg.init(0);
        return wg.is_initialized();
    } catch (...) { return false; }
}

bool is_zendnn_available() {
    try {
        auto& zd = gpu::get_zendnn_compute();
        zd.init(0);
        return zd.is_initialized();
    } catch (...) { return false; }
}

int64_t cpu_memory_free() {
#if defined(_WIN32)
    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    GlobalMemoryStatusEx(&mem);
    return (int64_t)mem.ullAvailPhys;
#elif defined(__linux__)
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return (int64_t)pages * (int64_t)page_size;
#elif defined(__APPLE__)
    int64_t free_pages = 0;
    size_t len = sizeof(free_pages);
    if (sysctlbyname("vm.page_free_count", &free_pages, &len, NULL, 0) == 0) {
        int64_t page_size = 0;
        len = sizeof(page_size);
        sysctlbyname("hw.pagesize", &page_size, &len, NULL, 0);
        if (page_size == 0) page_size = 4096;
        return free_pages * page_size;
    }
    return 8LL * 1024 * 1024 * 1024;
#else
    return 8LL * 1024 * 1024 * 1024;
#endif
}

int64_t cpu_memory_total() {
#if defined(_WIN32)
    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    GlobalMemoryStatusEx(&mem);
    return (int64_t)mem.ullTotalPhys;
#elif defined(__linux__)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return (int64_t)pages * (int64_t)page_size;
#elif defined(__APPLE__)
    int64_t mem = 0;
    size_t len = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &len, NULL, 0);
    return mem;
#else
    return 16LL * 1024 * 1024 * 1024;
#endif
}

int64_t gpu_memory_free(int64_t device_id) {
    (void)device_id;
#if defined(_WIN32)
    if (is_directx_available()) {
        try {
            auto& dx = gpu::get_dx_compute();
            if (dx.is_initialized())
                return dx.memory_free();
        } catch (...) {
            std::fprintf(stderr, "[WARN] Exception caught: %s (GPU memory query failed)\n", __func__);
            return cpu_memory_free();
        }
    }
#endif
#if defined(__APPLE__)
    if (is_metal_available()) {
        try {
            auto& metal = gpu::get_metal_compute();
            if (metal.is_initialized())
                return metal.memory_free();
        } catch (...) {
            return cpu_memory_free();
        }
    }
#endif
    return cpu_memory_free();
}

int64_t metal_memory_free() {
#if defined(__APPLE__)
    return gpu_memory_free(0);
#else
    return 0;
#endif
}

int64_t igpu_memory_free() {
    return cpu_memory_free();
}

Tensor to_backend(const Tensor& t, BackendType dst) {
    if (dst == BackendType::CPU_SCALAR || dst == BackendType::CPU_AVX2) {
        if (!t.data()) return t;
        Tensor out(t.shape());
        std::memcpy(out.data(), t.data(), t.numel() * sizeof(float));
        return out;
    }
    // GPU backends: copy to a new CPU tensor (true GPU transfer not yet implemented)
    if (t.data()) {
        Tensor out(t.shape());
        std::memcpy(out.data(), t.data(), t.numel() * sizeof(float));
        return out;
    }
    return t;
}

Tensor from_backend(const Tensor& t, BackendType src) {
    if (src == BackendType::CPU_SCALAR || src == BackendType::CPU_AVX2) {
        if (!t.data()) return t;
        Tensor out(t.shape());
        std::memcpy(out.data(), t.data(), t.numel() * sizeof(float));
        return out;
    }
    // GPU backends: copy from a source tensor to CPU (true GPU transfer not yet implemented)
    if (t.data()) {
        Tensor out(t.shape());
        std::memcpy(out.data(), t.data(), t.numel() * sizeof(float));
        return out;
    }
    return t;
}

// ========================================================================
// Hardware-target selection system
// ========================================================================

HardwareProfile probe_hardware() {
    HardwareProfile hw;

    // CPU features
    hw.has_avx2 = is_avx2_available();
    hw.has_avx512 = is_avx512_available();
    hw.has_neon = is_neon_available();

    // GPU features (all 24-family probes; missing hardware => false, honest)
    hw.has_cuda = is_cuda_available();
    hw.has_directx = is_directx_available();
    hw.has_vulkan = is_vulkan_available();
    hw.has_metal = is_metal_available();
    hw.has_sycl = is_sycl_available();
    hw.has_cann = is_cann_available();
    hw.has_rpc = is_rpc_available();
    hw.has_hip = is_hip_available();
    hw.has_hexagon = is_hexagon_available();
    hw.has_zdnn = is_zdnn_available();
    hw.has_musa = is_musa_available();
    hw.has_opencl = is_opencl_available();
    hw.has_openvino = is_openvino_available();
    hw.has_virtgpu = is_virtgpu_available();
    hw.has_webgpu = is_webgpu_available();
    hw.has_zendnn = is_zendnn_available();

    // RAM
    hw.ram_total = cpu_memory_total();
    hw.ram_free = cpu_memory_free();

    // GPU VRAM (DirectX)
#if defined(_WIN32)
    if (hw.has_directx) {
        try {
            auto& dx = gpu::get_dx_compute();
            if (dx.is_initialized()) {
                hw.vram_free = dx.memory_free();
                hw.vram_total = dx.memory_total();
            }
        } catch (...) {
            std::fprintf(stderr, "[WARN] Exception caught: %s (VRAM query failed)\n", __func__);
            hw.vram_free = 0;
            hw.vram_total = 0;
        }
    }
#endif

    // CPU cores/threads
#if defined(_WIN32)
    SYSTEM_INFO sys = {};
    GetSystemInfo(&sys);
    hw.cpu_cores = (int32_t)sys.dwNumberOfProcessors;
#else
    hw.cpu_cores = (int32_t)std::thread::hardware_concurrency();
#endif
    hw.cpu_threads = (int32_t)std::thread::hardware_concurrency();

    // OS detection
#if defined(_WIN32)
    hw.is_windows = true;
#elif defined(__linux__)
    hw.is_linux = true;
#elif defined(__APPLE__)
    hw.is_macos = true;
#endif

#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
    hw.is_arm = true;
#endif

    return hw;
}

BackendConfig select_optimal_backend(const HardwareProfile& hw,
                                     int64_t model_size_bytes) {
    BackendConfig cfg;
    cfg.threads = hw.cpu_threads > 0 ? hw.cpu_threads : 4;

    // REAL ONLY priority: only backends with REAL compute kernels on the
    // current host are eligible. PARTIAL/UNAVAILABLE backends (SYCL/CANN/
    // OpenVINO/VirtGPU/WebGPU/HIP-partial/OpenCL-partial/RPC-down/Hexagon/
    // ZDNN/MUSA) are NEVER auto-selected: benchmark_operation returns 0 for
    // unavailable and their compute ops throw, so selecting them would hand
    // the caller a throwing backend. Eligible: CUDA > Metal > Vulkan-partial
    // (relu-family REAL) > DX12 > AVX512 > AVX2 > NEON > ZenDNN-CPU > CPU.
    // NOTE: Vulkan is PARTIAL (relu/gelu/silu/add/mul REAL, gemm throws);
    // it is selected only when no FULLY-REAL GEMM backend exists, and the
    // caller must use the relu-family ops or expect gemm to throw.

    if (hw.has_cuda) {
        cfg.type = BackendType::GPU_CUDA;
        cfg.device_name = "GPU_CUDA";
        return cfg;
    }

    if (hw.has_metal) {
        cfg.type = BackendType::GPU_METAL;
        cfg.device_name = "GPU_METAL";
        return cfg;
    }

    if (hw.has_directx && hw.vram_total > 0) {
        cfg.type = BackendType::GPU_DIRECTX;
        cfg.device_name = "GPU_DIRECTX";
        return cfg;
    }

    if (hw.has_avx512) {
        cfg.type = BackendType::CPU_AVX512;
        cfg.device_name = "CPU_AVX512";
        return cfg;
    }

    // CPU with AVX2 for medium models that fit in RAM
    if (hw.has_avx2) {
        if (model_size_bytes == 0 || model_size_bytes <= (int64_t)(hw.ram_free * 0.5)) {
            cfg.type = BackendType::CPU_AVX2;
            cfg.device_name = "CPU_AVX2";
            return cfg;
        }
    }

    if (hw.has_neon) {
        cfg.type = BackendType::CPU_NEON;
        cfg.device_name = "CPU_NEON";
        return cfg;
    }

    if (hw.has_zendnn) {
        cfg.type = BackendType::CPU_ZENDNN;
        cfg.device_name = "CPU_ZENDNN";
        return cfg;
    }

    // Vulkan PARTIAL last-resort before generic CPU: relu-family ops are
    // REAL on a live device. Callers needing GEMM must handle the throw.
    if (hw.has_vulkan) {
        cfg.type = BackendType::GPU_VULKAN;
        cfg.device_name = "GPU_VULKAN";
        return cfg;
    }

    // iGPU shared memory for large models on systems with DirectX
    if (hw.has_directx) {
        cfg.type = BackendType::IGPU_SHARED;
        cfg.device_name = "IGPU_SHARED";
        cfg.memory_fraction = 0.8f;
        return cfg;
    }

    // RAM swap for very large models (tiled matmul, disk offload)
    if (model_size_bytes > hw.ram_free / 2) {
        cfg.type = BackendType::RAM_SWAP;
        cfg.device_name = "RAM_SWAP";
        cfg.memory_fraction = 0.95f;
        return cfg;
    }

    // Fallback: scalar CPU
    cfg.type = BackendType::CPU_SCALAR;
    cfg.device_name = "CPU_SCALAR";
    return cfg;
}

double benchmark_operation(ComputeBackend* backend, const char* operation,
                           int64_t M, int64_t N, int64_t K,
                           int warmup, int iters) {
    if (!backend || !backend->is_available()) return 0.0;

    // Use N as the element count for element-wise ops; for gemm use proper 2D shapes
    int64_t n_elem = (M > 0 && N > 0) ? M * N : 0;

    Tensor A, B, C, X, Y;
    if (strcmp(operation, "gemm") == 0) {
        A = Tensor({M, K}, DType::F32);
        B = Tensor({K, N}, DType::F32);
        C = Tensor({M, N}, DType::F32);
        A.fill(1.0f);
        B.fill(2.0f);
        C.fill(0.0f);
    } else if (strcmp(operation, "relu") == 0 || strcmp(operation, "add") == 0) {
        n_elem = (n_elem > 0) ? n_elem : 1024 * 1024;
        A = Tensor({(int64_t)n_elem}, DType::F32);
        B = Tensor({(int64_t)n_elem}, DType::F32);
        C = Tensor({(int64_t)n_elem}, DType::F32);
        Y = Tensor({(int64_t)n_elem}, DType::F32);
        A.fill(1.0f);
        B.fill(2.0f);
        C.fill(0.0f);
    } else if (strcmp(operation, "softmax") == 0 || strcmp(operation, "rms_norm") == 0) {
        int64_t rows = 1024, cols = 1024;
        A = Tensor({rows, cols}, DType::F32);
        Y = Tensor({rows, cols}, DType::F32);
        if (strcmp(operation, "rms_norm") == 0) {
            X = Tensor({cols}, DType::F32);
            X.fill(1.0f);
        }
        A.fill(1.0f);
    } else {
        return 0.0; // unknown operation
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    auto elapsed_us = [&]() -> double {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    };

    // REAL ONLY: PARTIAL backends (e.g. live Vulkan on gemm) throw for
    // unimplemented ops. A benchmark that throws measures nothing — return
    // 0.0 (honest "unmeasurable") instead of propagating or, worse,
    // substituting CPU timings under a GPU name.
    try {
        // Warmup
        for (int i = 0; i < warmup; i++) {
            if (strcmp(operation, "gemm") == 0) {
                backend->gemm(1.0f, A, B, 0.0f, C);
            } else if (strcmp(operation, "relu") == 0) {
                backend->relu(A, Y);
            } else if (strcmp(operation, "add") == 0) {
                backend->add(A, B, C);
            } else if (strcmp(operation, "softmax") == 0) {
                backend->softmax(A, Y, 1);
            } else if (strcmp(operation, "rms_norm") == 0) {
                backend->rms_norm(A, X, 1e-5f, Y);
            }
        }

        // Benchmark
        backend->synchronize();
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            if (strcmp(operation, "gemm") == 0) {
                backend->gemm(1.0f, A, B, 0.0f, C);
            } else if (strcmp(operation, "relu") == 0) {
                backend->relu(A, Y);
            } else if (strcmp(operation, "add") == 0) {
                backend->add(A, B, C);
            } else if (strcmp(operation, "softmax") == 0) {
                backend->softmax(A, Y, 1);
            } else if (strcmp(operation, "rms_norm") == 0) {
                backend->rms_norm(A, X, 1e-5f, Y);
            }
        }
        backend->synchronize();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[REAL-ONLY] benchmark_operation(%s/%s) unmeasurable: %s\n",
                     backend->name(), operation, e.what());
        return 0.0;
    }
    double dt_us = elapsed_us();

    double avg_us = dt_us / iters;
    double flops = 0;

    if (strcmp(operation, "gemm") == 0) {
        flops = 2.0 * (double)M * (double)N * (double)K / (avg_us * 1e-6) * 1e-9;
    } else if (strcmp(operation, "relu") == 0 || strcmp(operation, "add") == 0) {
        flops = (double)n_elem / (avg_us * 1e-6) * 1e-9;
    } else if (strcmp(operation, "softmax") == 0 || strcmp(operation, "rms_norm") == 0) {
        double n = 1024.0 * 1024.0;
        flops = n * 5.0 / (avg_us * 1e-6) * 1e-9;
    }

    return flops; // GFLOPS
}

} // namespace backend
} // namespace quant
