#include "quant/gpu_compute_metal.h"
#include <string>
#include <vector>
#include <mutex>
#include <cstring>
#include <cmath>

#if defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace quant {
namespace gpu {

static const char* METAL_MSL_CODE = R"(
#include <metal_stdlib>
using namespace metal;

kernel void gemm_kernel(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]], constant float& alpha [[buffer(3)]], constant float& beta [[buffer(4)]], constant uint& M [[buffer(5)]], constant uint& N [[buffer(6)]], constant uint& K [[buffer(7)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= N || gid.y >= M) return;
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[gid.y * K + k] * B[k * N + gid.x];
    }
    if (beta == 0.0) {
        C[gid.y * N + gid.x] = alpha * sum;
    } else {
        C[gid.y * N + gid.x] = alpha * sum + beta * C[gid.y * N + gid.x];
    }
}

kernel void relu_kernel(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) y[gid] = max(x[gid], 0.0f);
}

kernel void silu_kernel(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) { float val = x[gid]; y[gid] = val / (1.0f + exp(-val)); }
}

kernel void gelu_kernel(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) { float val = x[gid]; y[gid] = 0.5f * val * (1.0f + tanh(sqrt(2.0f / 3.1415926535f) * (val + 0.044715f * val * val * val))); }
}

kernel void add_kernel(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* c [[buffer(2)]], constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) c[gid] = a[gid] + b[gid];
}

kernel void mul_kernel(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* c [[buffer(2)]], constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) c[gid] = a[gid] * b[gid];
}

kernel void scale_kernel(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant float& s [[buffer(2)]], constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) y[gid] = s * x[gid];
}

kernel void fill_kernel(device float* x [[buffer(0)]], constant float& val [[buffer(1)]], constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) x[gid] = val;
}

kernel void copy_kernel(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) dst[gid] = src[gid];
}

kernel void softmax_kernel(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= rows) return;
    uint row = gid;
    float max_val = -1e20;
    for (uint i = 0; i < cols; i++) max_val = max(max_val, x[row * cols + i]);
    float sum = 0.0;
    for (uint i = 0; i < cols; i++) {
        float e = exp(x[row * cols + i] - max_val);
        y[row * cols + i] = e;
        sum += e;
    }
    for (uint i = 0; i < cols; i++) y[row * cols + i] /= sum;
}

kernel void rmsnorm_kernel(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device float* y [[buffer(2)]], constant uint& rows [[buffer(3)]], constant uint& cols [[buffer(4)]], constant float& eps [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= rows) return;
    uint row = gid;
    float sum = 0.0;
    for (uint i = 0; i < cols; i++) { float val = x[row * cols + i]; sum += val * val; }
    float inv_rms = 1.0f / sqrt(sum / (float)cols + eps);
    for (uint i = 0; i < cols; i++) y[row * cols + i] = x[row * cols + i] * inv_rms * w[i];
}

kernel void rope_kernel(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {}
kernel void attention_kernel(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {}
kernel void reduce_sum_kernel(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {}
kernel void reduce_max_kernel(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {}
)";

struct MTLSize {
    size_t width, height, depth;
};

using id = void*;
using Class = void*;
using SEL = void*;

#if defined(__APPLE__)
extern "C" {
    id objc_getClass(const char *name);
    SEL sel_registerName(const char *str);
    id objc_msgSend(id self, SEL op, ...);
}

typedef id (*PFN_MTLCreateSystemDefaultDevice)();

template<typename Ret = id, typename... Args>
Ret msgSend(id obj, SEL op, Args... args) {
    using FuncType = Ret(*)(id, SEL, Args...);
    FuncType func = (FuncType)objc_msgSend;
    return func(obj, op, args...);
}

#endif // __APPLE__

struct GPUComputeMetal::Impl {
    bool ok = false;
    std::mutex mtx;
    
#if defined(__APPLE__)
    void* libobjc = nullptr;
    void* libmetal = nullptr;
    
    id device = nullptr;
    id command_queue = nullptr;
    id library = nullptr;
    
    id f_gemm = nullptr, f_relu = nullptr, f_silu = nullptr, f_gelu = nullptr;
    id f_add = nullptr, f_mul = nullptr, f_scale = nullptr, f_fill = nullptr, f_copy = nullptr;
    id f_softmax = nullptr, f_rmsnorm = nullptr;
    
    std::vector<id> bufs;
    std::vector<id> active_cmd_bufs;
    
    SEL sel_stringWithUTF8String;
    SEL sel_newLibraryWithSource;
    SEL sel_newFunctionWithName;
    SEL sel_newComputePipelineStateWithFunction;
    SEL sel_newBufferWithLength;
    SEL sel_commandBuffer;
    SEL sel_computeCommandEncoder;
    SEL sel_setComputePipelineState;
    SEL sel_setBuffer;
    SEL sel_setBytes;
    SEL sel_dispatchThreads;
    SEL sel_endEncoding;
    SEL sel_commit;
    SEL sel_waitUntilCompleted;
    SEL sel_contents;
    SEL sel_didModifyRange;
    
    Class cls_NSString;

    id get_function(const char* name) {
        id str = msgSend(cls_NSString, sel_stringWithUTF8String, name);
        id func = msgSend(library, sel_newFunctionWithName, str);
        if (!func) return nullptr;
        id err = nullptr;
        id pipeline = msgSend(device, sel_newComputePipelineStateWithFunction, func, &err);
        return pipeline;
    }

    bool init(int64_t device_id) {
        libobjc = dlopen("/usr/lib/libobjc.A.dylib", RTLD_NOW);
        libmetal = dlopen("/System/Library/Frameworks/Metal.framework/Metal", RTLD_NOW);
        if (!libobjc || !libmetal) return false;
        
        PFN_MTLCreateSystemDefaultDevice createDevice = (PFN_MTLCreateSystemDefaultDevice)dlsym(libmetal, "MTLCreateSystemDefaultDevice");
        if (!createDevice) return false;
        
        device = createDevice();
        if (!device) return false;
        
        sel_stringWithUTF8String = sel_registerName("stringWithUTF8String:");
        sel_newLibraryWithSource = sel_registerName("newLibraryWithSource:options:error:");
        sel_newFunctionWithName = sel_registerName("newFunctionWithName:");
        sel_newComputePipelineStateWithFunction = sel_registerName("newComputePipelineStateWithFunction:error:");
        sel_newBufferWithLength = sel_registerName("newBufferWithLength:options:");
        sel_commandBuffer = sel_registerName("commandBuffer");
        sel_computeCommandEncoder = sel_registerName("computeCommandEncoder");
        sel_setComputePipelineState = sel_registerName("setComputePipelineState:");
        sel_setBuffer = sel_registerName("setBuffer:offset:atIndex:");
        sel_setBytes = sel_registerName("setBytes:length:atIndex:");
        sel_dispatchThreads = sel_registerName("dispatchThreads:threadsPerThreadgroup:");
        sel_endEncoding = sel_registerName("endEncoding");
        sel_commit = sel_registerName("commit");
        sel_waitUntilCompleted = sel_registerName("waitUntilCompleted");
        sel_contents = sel_registerName("contents");
        sel_didModifyRange = sel_registerName("didModifyRange:");
        
        cls_NSString = objc_getClass("NSString");
        
        id src_str = msgSend(cls_NSString, sel_stringWithUTF8String, METAL_MSL_CODE);
        id err = nullptr;
        library = msgSend(device, sel_newLibraryWithSource, src_str, nullptr, &err);
        if (!library) return false;
        
        command_queue = msgSend(device, sel_registerName("newCommandQueue"));
        if (!command_queue) return false;
        
        f_gemm = get_function("gemm_kernel");
        f_relu = get_function("relu_kernel");
        f_silu = get_function("silu_kernel");
        f_gelu = get_function("gelu_kernel");
        f_add = get_function("add_kernel");
        f_mul = get_function("mul_kernel");
        f_scale = get_function("scale_kernel");
        f_fill = get_function("fill_kernel");
        f_copy = get_function("copy_kernel");
        f_softmax = get_function("softmax_kernel");
        f_rmsnorm = get_function("rmsnorm_kernel");
        
        ok = true;
        return true;
    }
    
    void shutdown() {
        if (!ok) return;
        ok = false;
        bufs.clear();
        if (libmetal) dlclose(libmetal);
        if (libobjc) dlclose(libobjc);
    }
#else
    bool init(int64_t) { return false; }
    void shutdown() {}
#endif
};

GPUComputeMetal::GPUComputeMetal() : impl_(new Impl()) {}
GPUComputeMetal::~GPUComputeMetal() { shutdown(); delete impl_; }

bool GPUComputeMetal::init(int64_t device_id) {
    if (impl_->ok) return true;
    return impl_->init(device_id);
}

bool GPUComputeMetal::is_initialized() const { return impl_->ok; }
void GPUComputeMetal::shutdown() { impl_->shutdown(); }

void* GPUComputeMetal::alloc(int64_t bytes) {
    if (!impl_->ok || bytes <= 0) return nullptr;
#if defined(__APPLE__)
    id buf = msgSend(impl_->device, impl_->sel_newBufferWithLength, (size_t)bytes, (size_t)0);
    if (buf) {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        impl_->bufs.push_back(buf);
    }
    return buf;
#else
    return nullptr;
#endif
}

void GPUComputeMetal::free_buf(void* ptr) {
    if (!impl_->ok || !ptr) return;
#if defined(__APPLE__)
    id buf = (id)ptr;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    for (auto it = impl_->bufs.begin(); it != impl_->bufs.end(); ++it) {
        if (*it == buf) {
            impl_->bufs.erase(it);
            break;
        }
    }
#endif
}

void GPUComputeMetal::upload(const Tensor& src, void* dst) {
    if (!impl_->ok || !dst || src.numel() == 0) return;
#if defined(__APPLE__)
    id buf = (id)dst;
    void* contents = msgSend<void*>(buf, impl_->sel_contents);
    if (contents) {
        std::memcpy(contents, src.data(), src.size_bytes());
    }
#endif
}

void GPUComputeMetal::download(void* src, Tensor& dst) {
    if (!impl_->ok || !src || dst.numel() == 0) return;
#if defined(__APPLE__)
    synchronize();
    id buf = (id)src;
    void* contents = msgSend<void*>(buf, impl_->sel_contents);
    if (contents) {
        std::memcpy(dst.data(), contents, dst.size_bytes());
    }
#endif
}

#if defined(__APPLE__)
static void dispatch_kernel(GPUComputeMetal::Impl* impl, id pipeline, MTLSize grid, MTLSize block,
    const std::vector<std::pair<id, size_t>>& buffers,
    const std::vector<std::pair<const void*, size_t>>& bytes_args)
{
    id cmd_buf = msgSend(impl->command_queue, impl->sel_commandBuffer);
    id encoder = msgSend(cmd_buf, impl->sel_computeCommandEncoder);
    
    msgSend<void>(encoder, impl->sel_setComputePipelineState, pipeline);
    
    size_t idx = 0;
    for (auto& buf : buffers) {
        msgSend<void>(encoder, impl->sel_setBuffer, buf.first, buf.second, idx++);
    }
    for (auto& bytes : bytes_args) {
        msgSend<void>(encoder, impl->sel_setBytes, bytes.first, bytes.second, idx++);
    }
    
    msgSend<void>(encoder, impl->sel_dispatchThreads, grid, block);
    msgSend<void>(encoder, impl->sel_endEncoding);
    
    msgSend<void>(cmd_buf, impl->sel_commit);
    
    std::lock_guard<std::mutex> lk(impl->mtx);
    impl->active_cmd_bufs.push_back(cmd_buf);
}
#endif

void GPUComputeMetal::gemm(float alpha, const void* A, const void* B, float beta, void* C, int64_t M, int64_t N, int64_t K) {
    if (!impl_->ok || !A || !B || !C) return;
#if defined(__APPLE__)
    MTLSize grid = { (size_t)N, (size_t)M, 1 };
    MTLSize block = { 16, 16, 1 };
    uint32_t uM = M, uN = N, uK = K;
    dispatch_kernel(impl_, impl_->f_gemm, grid, block,
        {{(id)A, 0}, {(id)B, 0}, {(id)C, 0}},
        {{&alpha, sizeof(alpha)}, {&beta, sizeof(beta)},
         {&uM, sizeof(uM)}, {&uN, sizeof(uN)}, {&uK, sizeof(uK)}});
#endif
}

void GPUComputeMetal::gemv(float alpha, const void* A, const void* x, float beta, void* y, int64_t M, int64_t N) {
    gemm(alpha, A, x, beta, y, M, 1, N);
}

void GPUComputeMetal::relu(const void* x, void* y, int64_t n) {
    if (!impl_->ok || !x || !y) return;
#if defined(__APPLE__)
    MTLSize grid = { (size_t)n, 1, 1 };
    MTLSize block = { 256, 1, 1 };
    uint32_t uN = n;
    dispatch_kernel(impl_, impl_->f_relu, grid, block,
        {{(id)x, 0}, {(id)y, 0}},
        {{&uN, sizeof(uN)}});
#endif
}

void GPUComputeMetal::gelu(const void* x, void* y, int64_t n) {
    if (!impl_->ok || !x || !y) return;
#if defined(__APPLE__)
    MTLSize grid = { (size_t)n, 1, 1 };
    MTLSize block = { 256, 1, 1 };
    uint32_t uN = n;
    dispatch_kernel(impl_, impl_->f_gelu, grid, block,
        {{(id)x, 0}, {(id)y, 0}},
        {{&uN, sizeof(uN)}});
#endif
}

void GPUComputeMetal::silu(const void* x, void* y, int64_t n) {
    if (!impl_->ok || !x || !y) return;
#if defined(__APPLE__)
    MTLSize grid = { (size_t)n, 1, 1 };
    MTLSize block = { 256, 1, 1 };
    uint32_t uN = n;
    dispatch_kernel(impl_, impl_->f_silu, grid, block,
        {{(id)x, 0}, {(id)y, 0}},
        {{&uN, sizeof(uN)}});
#endif
}

void GPUComputeMetal::softmax(const void* x, void* y, int64_t rows, int64_t cols) {
    if (!impl_->ok || !x || !y) return;
#if defined(__APPLE__)
    MTLSize grid = { (size_t)rows, 1, 1 };
    MTLSize block = { 256, 1, 1 };
    uint32_t r = rows, c = cols;
    dispatch_kernel(impl_, impl_->f_softmax, grid, block,
        {{(id)x, 0}, {(id)y, 0}},
        {{&r, sizeof(r)}, {&c, sizeof(c)}});
#endif
}

void GPUComputeMetal::rms_norm(const void* x, const void* weight, void* y, int64_t rows, int64_t cols, float eps) {
    if (!impl_->ok || !x || !weight || !y) return;
#if defined(__APPLE__)
    MTLSize grid = { (size_t)rows, 1, 1 };
    MTLSize block = { 256, 1, 1 };
    uint32_t r = rows, c = cols;
    dispatch_kernel(impl_, impl_->f_rmsnorm, grid, block,
        {{(id)x, 0}, {(id)weight, 0}, {(id)y, 0}},
        {{&r, sizeof(r)}, {&c, sizeof(c)}, {&eps, sizeof(eps)}});
#endif
}

void GPUComputeMetal::layer_norm(const void* x, const void* gamma, const void* beta, void* y, int64_t rows, int64_t cols, float eps) {
    rms_norm(x, gamma, y, rows, cols, eps);
}

void GPUComputeMetal::add(const void* a, const void* b, void* c, int64_t n) {
    if (!impl_->ok || !a || !b || !c) return;
#if defined(__APPLE__)
    MTLSize grid = { (size_t)n, 1, 1 };
    MTLSize block = { 256, 1, 1 };
    uint32_t uN = n;
    dispatch_kernel(impl_, impl_->f_add, grid, block,
        {{(id)a, 0}, {(id)b, 0}, {(id)c, 0}},
        {{&uN, sizeof(uN)}});
#endif
}

void GPUComputeMetal::mul(const void* a, const void* b, void* c, int64_t n) {
    if (!impl_->ok || !a || !b || !c) return;
#if defined(__APPLE__)
    MTLSize grid = { (size_t)n, 1, 1 };
    MTLSize block = { 256, 1, 1 };
    uint32_t uN = n;
    dispatch_kernel(impl_, impl_->f_mul, grid, block,
        {{(id)a, 0}, {(id)b, 0}, {(id)c, 0}},
        {{&uN, sizeof(uN)}});
#endif
}

void GPUComputeMetal::scale(float s, const void* x, void* y, int64_t n) {
    if (!impl_->ok || !x || !y) return;
#if defined(__APPLE__)
    MTLSize grid = { (size_t)n, 1, 1 };
    MTLSize block = { 256, 1, 1 };
    uint32_t uN = n;
    dispatch_kernel(impl_, impl_->f_scale, grid, block,
        {{(id)x, 0}, {(id)y, 0}},
        {{&s, sizeof(s)}, {&uN, sizeof(uN)}});
#endif
}

void GPUComputeMetal::fill(float val, void* x, int64_t n) {
    if (!impl_->ok || !x) return;
#if defined(__APPLE__)
    MTLSize grid = { (size_t)n, 1, 1 };
    MTLSize block = { 256, 1, 1 };
    uint32_t uN = n;
    dispatch_kernel(impl_, impl_->f_fill, grid, block,
        {{(id)x, 0}},
        {{&val, sizeof(val)}, {&uN, sizeof(uN)}});
#endif
}

void GPUComputeMetal::copy_buf(const void* src, void* dst, int64_t n) {
    if (!impl_->ok || !src || !dst) return;
#if defined(__APPLE__)
    MTLSize grid = { (size_t)n, 1, 1 };
    MTLSize block = { 256, 1, 1 };
    uint32_t uN = n;
    dispatch_kernel(impl_, impl_->f_copy, grid, block,
        {{(id)src, 0}, {(id)dst, 0}},
        {{&uN, sizeof(uN)}});
#endif
}

void GPUComputeMetal::synchronize() {
    if (!impl_->ok) return;
#if defined(__APPLE__)
    std::vector<id> to_wait;
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        to_wait = impl_->active_cmd_bufs;
        impl_->active_cmd_bufs.clear();
    }
    for (id cmd : to_wait) {
        msgSend<void>(cmd, impl_->sel_waitUntilCompleted);
    }
#endif
}

int64_t GPUComputeMetal::memory_free() const {
    return 1LL * 1024 * 1024 * 1024;
}

int64_t GPUComputeMetal::memory_total() const {
    return 1LL * 1024 * 1024 * 1024;
}

static GPUComputeMetal g_metal_compute;
GPUComputeMetal& get_metal_compute() { return g_metal_compute; }

} // namespace gpu
} // namespace quant
