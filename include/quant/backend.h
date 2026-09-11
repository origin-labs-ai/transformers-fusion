#pragma once
#include "quant/types.h"
#include "quant/tensor.h"
#include <string>

namespace quant {
namespace backend {

enum class BackendType {
    CPU_SCALAR = 0,
    CPU_AVX2,
    CPU_AVX512,
    CPU_NEON,
    GPU_CUDA,
    GPU_DIRECTX,
    GPU_VULKAN,
    GPU_SYCL,
    GPU_CANN,
    GPU_METAL,
    RPC,
    IGPU_SHARED,
    RAM_SWAP,
    DISTRIBUTED,
    CPU,
    GPU_HIP,
    DSP_HEXAGON,
    NPU_ZDNN,
    GPU_MUSA,
    GPU_OPENCL,
    NPU_OPENVINO,
    GPU_VIRTGPU,
    GPU_WEBGPU,
    CPU_ZENDNN
};

inline const char* backend_name(BackendType t) {
    switch (t) {
        case BackendType::CPU_SCALAR: return "CPU_SCALAR";
        case BackendType::CPU_AVX2: return "CPU_AVX2";
        case BackendType::CPU_AVX512: return "CPU_AVX512";
        case BackendType::CPU_NEON: return "CPU_NEON";
        case BackendType::GPU_CUDA: return "GPU_CUDA";
        case BackendType::GPU_DIRECTX: return "GPU_DIRECTX";
        case BackendType::GPU_VULKAN: return "GPU_VULKAN";
        case BackendType::GPU_SYCL: return "GPU_SYCL";
        case BackendType::GPU_CANN: return "GPU_CANN";
        case BackendType::GPU_METAL: return "GPU_METAL";
        case BackendType::RPC: return "RPC";
        case BackendType::IGPU_SHARED: return "IGPU_SHARED";
        case BackendType::RAM_SWAP: return "RAM_SWAP";
        case BackendType::DISTRIBUTED: return "DISTRIBUTED";
        case BackendType::CPU: return "CPU";
        case BackendType::GPU_HIP: return "GPU_HIP";
        case BackendType::DSP_HEXAGON: return "DSP_HEXAGON";
        case BackendType::NPU_ZDNN: return "NPU_ZDNN";
        case BackendType::GPU_MUSA: return "GPU_MUSA";
        case BackendType::GPU_OPENCL: return "GPU_OPENCL";
        case BackendType::NPU_OPENVINO: return "NPU_OPENVINO";
        case BackendType::GPU_VIRTGPU: return "GPU_VIRTGPU";
        case BackendType::GPU_WEBGPU: return "GPU_WEBGPU";
        case BackendType::CPU_ZENDNN: return "CPU_ZENDNN";
        default: return "UNKNOWN";
    }
}

struct BackendConfig {
    BackendType type = BackendType::CPU_SCALAR;
    int64_t device_id = 0;
    int64_t num_devices = 1;
    int64_t threads = 0;
    bool enable_fp16 = false;
    bool enable_mixed_precision = false;
    float memory_fraction = 0.9f;
    std::string device_name;
};

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;
    virtual BackendType type() const = 0;
    virtual const char* name() const = 0;

    virtual void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) = 0;
    virtual void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) = 0;

    virtual void softmax(const Tensor& x, Tensor& y, int axis) = 0;
    virtual void layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps, Tensor& y) = 0;
    virtual void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y) = 0;

    virtual void relu(const Tensor& x, Tensor& y) = 0;
    virtual void gelu(const Tensor& x, Tensor& y) = 0;
    virtual void silu(const Tensor& x, Tensor& y) = 0;

    virtual void add(const Tensor& a, const Tensor& b, Tensor& c) = 0;
    virtual void mul(const Tensor& a, const Tensor& b, Tensor& c) = 0;
    virtual void scale(float s, const Tensor& x, Tensor& y) = 0;

    virtual void copy(const Tensor& src, Tensor& dst) = 0;
    virtual void fill(Tensor& t, float val) = 0;
    virtual void zero(Tensor& t) = 0;

    virtual bool is_available() const = 0;
    virtual int64_t memory_free() const = 0;
    virtual int64_t memory_total() const = 0;
    virtual void synchronize() = 0;

    static ComputeBackend* create(const BackendConfig& cfg);
};

bool is_avx2_available();
bool is_avx512_available();
bool is_neon_available();
bool is_cuda_available();
bool is_directx_available();
bool is_vulkan_available();
bool is_metal_available();
bool is_sycl_available();
bool is_cann_available();
bool is_rpc_available();
bool is_hip_available();
bool is_hexagon_available();
bool is_zdnn_available();
bool is_musa_available();
bool is_opencl_available();
bool is_openvino_available();
bool is_virtgpu_available();
bool is_webgpu_available();
bool is_zendnn_available();

int64_t cpu_memory_free();
int64_t cpu_memory_total();
int64_t gpu_memory_free(int64_t device_id);
int64_t metal_memory_free();
int64_t igpu_memory_free();

Tensor to_backend(const Tensor& t, BackendType dst);
Tensor from_backend(const Tensor& t, BackendType src);

// ========================================================================
// Hardware-target selection system
// ========================================================================

struct HardwareProfile {
    bool has_avx2 = false;
    bool has_avx512 = false;
    bool has_neon = false;
    bool has_cuda = false;
    bool has_directx = false;
    bool has_vulkan = false;
    bool has_metal = false;
    bool has_sycl = false;
    bool has_cann = false;
    bool has_rpc = false;
    bool has_hip = false;
    bool has_hexagon = false;
    bool has_zdnn = false;
    bool has_musa = false;
    bool has_opencl = false;
    bool has_openvino = false;
    bool has_virtgpu = false;
    bool has_webgpu = false;
    bool has_zendnn = false;
    int64_t ram_total = 0;   // bytes
    int64_t ram_free = 0;    // bytes
    int64_t vram_total = 0;  // bytes (GPU dedicated)
    int64_t vram_free = 0;   // bytes
    int32_t cpu_cores = 0;
    int32_t cpu_threads = 0;
    bool is_windows = false;
    bool is_linux = false;
    bool is_macos = false;
    bool is_arm = false;
};

// Probes all available hardware and returns a complete profile.
HardwareProfile probe_hardware();

// Selects the optimal backend given hardware capabilities and model size.
// model_size_bytes = number of parameters * bytes_per_param (e.g., 4 for FP32).
// Returns a BackendConfig with the recommended settings.
BackendConfig select_optimal_backend(const HardwareProfile& hw,
                                     int64_t model_size_bytes = 0);

// Convenience: probe once, then auto-select.
inline BackendConfig auto_select_backend(int64_t model_size_bytes = 0) {
    return select_optimal_backend(probe_hardware(), model_size_bytes);
}

// Micro-benchmark a single operation on a backend. Returns GFLOPS or bandwidth.
// operation: "gemm", "relu", "add", "softmax", "rms_norm"
// Returns measured throughput in GFLOPS (always > 0 on success), or exactly
// 0.0 as the documented UNMEASURABLE sentinel: null/dead backend, unknown op,
// or the device threw mid-benchmark (a real throw is logged to stderr).
// Callers must treat 0.0 as "no measurement", never as a slow device —
// a live benchmark cannot return 0.0 since any completed timing is finite.
// (Deliberately NOT NaN: callers and T5 pin `== 0.0`; NaN would silently
// pass/fail those equality checks. See backend.cpp:3043.)
double benchmark_operation(ComputeBackend* backend, const char* operation,
                           int64_t M = 1024, int64_t N = 1024, int64_t K = 1024,
                           int warmup = 5, int iters = 30);

} // namespace backend
} // namespace quant
