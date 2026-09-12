#include "quant/gpu_compute_full.h"
#include "quant/tensor.h"
#include "quant/math.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

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

using Microsoft::WRL::ComPtr;

namespace quant {
namespace gpu {

static const char* g_tiled_gemm_cs = R"(
cbuffer Const : register(b0) { uint M,N,K,tile; float alpha,beta,_p2,_p3; };
RWStructuredBuffer<float> C : register(u0);
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
groupshared float As[16][16];
groupshared float Bs[16][16];
[numthreads(16,16,1)]
void main(uint3 g : SV_GroupThreadID, uint3 gi : SV_GroupID) {
    uint row = gi.y * 16 + g.y;
    uint col = gi.x * 16 + g.x;
    float s = 0;
    uint tiles = (K + 15) / 16;
    for (uint t = 0; t < tiles; t++) {
        uint aCol = t * 16 + g.x;
        As[g.y][g.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0;
        uint bRow = t * 16 + g.y;
        Bs[g.y][g.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0;
        GroupMemoryBarrierWithGroupSync();
        for (uint i = 0; i < 16; i++)
            s += As[g.y][i] * Bs[i][g.x];
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < M && col < N)
        C[row * N + col] = alpha * s + beta * C[row * N + col];
})";

static const char* g_gemm_cs = R"(
cbuffer Const : register(b0) { uint M,N,K,tile; float alpha,beta,_p2,_p3; };
RWStructuredBuffer<float> C : register(u0);
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
[numthreads(16,16,1)]
void main(uint3 g : SV_GroupThreadID, uint3 gi : SV_GroupID) {
    uint row = gi.y * 16 + g.y;
    uint col = gi.x * 16 + g.x;
    if (row >= M || col >= N) return;
    float s = 0;
    for (uint k = 0; k < K; k++) s += A[row * K + k] * B[k * N + col];
    C[row * N + col] = alpha * s + beta * C[row * N + col];
})";

static const char* g_softmax_stable_cs = R"(
cbuffer Const : register(b0) { uint rows,cols,_p0,_p1; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if (t.x >= rows) return;
    uint r = t.x;
    float mv = -1e30;
    for (uint c = 0; c < cols; c++) mv = max(mv, x[r * cols + c]);
    float s = 0;
    for (uint c = 0; c < cols; c++) { float e = exp(x[r * cols + c] - mv); y[r * cols + c] = e; s += e; }
    float iv = 1.0f / (s + 1e-10f);
    for (uint c = 0; c < cols; c++) y[r * cols + c] *= iv;
})";

static const char* g_layernorm_cs = R"(
cbuffer Const : register(b0) { uint N,D,_p0,_p1; float eps,_f1,_f2,_f3; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
StructuredBuffer<float> g : register(t1);
StructuredBuffer<float> b : register(t2);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if (t.x >= N) return;
    float mn = 0;
    for (uint d = 0; d < D; d++) mn += x[t.x * D + d];
    mn /= D;
    float vr = 0;
    for (uint d = 0; d < D; d++) { float df = x[t.x * D + d] - mn; vr += df * df; }
    vr /= D;
    float iv = rsqrt(vr + eps);
    for (uint d = 0; d < D; d++) y[t.x * D + d] = (x[t.x * D + d] - mn) * iv * g[d] + b[d];
})";

static const char* g_rms_norm_cs = R"(
cbuffer Const : register(b0) { uint N,D,_p0,_p1; float eps,_f1,_f2,_f3; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
StructuredBuffer<float> g : register(t1);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if (t.x >= N) return;
    float ss = 0;
    for (uint d = 0; d < D; d++) { float v = x[t.x * D + d]; ss += v * v; }
    float rs = rsqrt(ss / D + eps);
    for (uint d = 0; d < D; d++) y[t.x * D + d] = x[t.x * D + d] * rs * g[d];
})";

static const char* g_batch_norm_cs = R"(
cbuffer Const : register(b0) { uint N,C,HW,_p0; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
StructuredBuffer<float> g : register(t1);
StructuredBuffer<float> b : register(t2);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if (t.x >= N * C * HW) return;
    uint hw_idx = t.x % HW;
    uint c_idx = (t.x / HW) % C;
    uint n_idx = t.x / (C * HW);
    uint chw = C * HW;
    float sum = 0, sq = 0;
    for (uint n = 0; n < N; n++) { float v = x[n * chw + c_idx * HW + hw_idx]; sum += v; sq += v * v; }
    float mean = sum / N;
    float var = sq / N - mean * mean;
    float iv = rsqrt(var + 1e-5f);
    y[t.x] = (x[t.x] - mean) * iv * g[c_idx] + b[c_idx];
})";

static const char* g_sub_cs = R"(
cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> c : register(u0);
StructuredBuffer<float> a : register(t0);
StructuredBuffer<float> b : register(t1);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if (t.x >= N) return;
    c[t.x] = a[t.x] - b[t.x];
})";

static const char* g_div_cs = R"(
cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> c : register(u0);
StructuredBuffer<float> a : register(t0);
StructuredBuffer<float> b : register(t1);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if (t.x >= N) return;
    c[t.x] = b[t.x] != 0 ? a[t.x] / b[t.x] : 0;
})";

static const char* g_add_scalar_cs = R"(
cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float s,_f1,_f2,_f3; };
RWStructuredBuffer<float> c : register(u0);
StructuredBuffer<float> a : register(t0);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if (t.x >= N) return;
    c[t.x] = a[t.x] + s;
})";

static const char* g_tanh_cs = R"(
cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if (t.x >= N) return;
    y[t.x] = tanh(x[t.x]);
})";

static const char* g_sigmoid_cs = R"(
cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if (t.x >= N) return;
    y[t.x] = 1.0f / (1.0f + exp(-x[t.x]));
})";

static const char* g_reduce_sum_cs = R"(
// LIMITATION: single-pass reduce_sum only works for N <= 256.
// The shared memory and thread count are fixed at 256 elements.
cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> out : register(u0);
StructuredBuffer<float> x : register(t0);
groupshared float sdata[256];
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    uint tid = t.x;
    sdata[tid] = (tid < N) ? x[tid] : 0;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0) out[0] = sdata[0];
})";

static const char* g_reduce_max_cs = R"(
cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> out : register(u0);
StructuredBuffer<float> x : register(t0);
groupshared float sdata[256];
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    uint tid = t.x;
    sdata[tid] = (tid < N) ? x[tid] : -1e30;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0) out[0] = sdata[0];
})";

static const char* g_reduce_axis_cs = R"(
cbuffer Const : register(b0) { uint rows,cols,axis,_p2; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> out : register(u0);
StructuredBuffer<float> x : register(t0);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if (t.x >= rows) return;
    float s = 0;
    for (uint c = 0; c < cols; c++) s += x[t.x * cols + c];
    out[t.x] = s / cols;
})";

static const char* g_transpose_cs = R"(
cbuffer Const : register(b0) { uint rows,cols,_p0,_p1; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
[numthreads(16,16,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if (t.x >= cols || t.y >= rows) return;
    y[t.x * rows + t.y] = x[t.y * cols + t.x];
})";

static const char* g_embedding_cs = R"(
cbuffer Const : register(b0) { uint N,D,_p0,_p1; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> out : register(u0);
StructuredBuffer<float> table : register(t0);
StructuredBuffer<int64_t> indices : register(t1);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if (t.x >= N * D) return;
    uint d = t.x % D;
    uint n = t.x / D;
    uint idx = (uint)indices[n];
    out[n * D + d] = table[idx * D + d];
})";

static const char* g_cross_attn_cs = R"(
// cbuffer must stay at exactly 8 DWORDs to match D3D12 root signature (Num32BitValues=8).
// Fields: B, H, Tq, Tk, D + 3 padding. Do NOT add fields without updating root signature.
cbuffer Const : register(b0) { uint B,H,Tq,Tk,D,_p1,_p2,_p3; };
RWStructuredBuffer<float> out : register(u0);
StructuredBuffer<float> Q : register(t0);
StructuredBuffer<float> KV : register(t1);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if (t.x >= B * H * Tq * D) return;
    uint d = t.x % D;
    uint t_q = (t.x / D) % Tq;
    uint h = (t.x / (D * Tq)) % H;
    uint b = t.x / (D * Tq * H);
    float scale = rsqrt(float(D));
    float mv = -1e30;
    uint kv_stride = H * Tk * D;
    uint q_offset = b * H * Tq * D + h * Tq * D + t_q * D;
    for (uint t_k = 0; t_k < Tk; t_k++) {
        float dot = 0;
        uint kv_offset = b * kv_stride + h * Tk * D + t_k * D;
        for (uint dd = 0; dd < D; dd++)
            dot += Q[q_offset + dd] * KV[kv_offset + dd];
        dot *= scale;
        if (dot > mv) mv = dot;
    }
    float sum_exp = 0;
    float acc = 0;
    for (uint t_k = 0; t_k < Tk; t_k++) {
        float dot = 0;
        uint kv_offset = b * kv_stride + h * Tk * D + t_k * D;
        for (uint dd = 0; dd < D; dd++)
            dot += Q[q_offset + dd] * KV[kv_offset + dd];
        dot *= scale;
        float w = exp(dot - mv);
        sum_exp += w;
        acc += w * KV[kv_offset + d];
    }
    out[t.x] = sum_exp > 0 ? acc / sum_exp : 0;
})";

static const char* g_attn_fwd_cs = R"(
// Scaled dot-product attention: O = softmax(Q*K^T/sqrt(D) + causal_mask) * V
cbuffer Const : register(b0) { uint B,H,T,D,causal,_p1,_p2,_p3; };
RWStructuredBuffer<float> out : register(u0);
StructuredBuffer<float> Q : register(t0);
StructuredBuffer<float> K : register(t1);
StructuredBuffer<float> V : register(t2);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if (t.x >= B * H * T * D) return;
    uint d = t.x % D;
    uint t_q = (t.x / D) % T;
    uint h = (t.x / (D * T)) % H;
    uint b = t.x / (D * T * H);
    float scale = rsqrt(float(D));
    float mv = -1e30;
    uint q_offset = b * H * T * D + h * T * D + t_q * D;
    uint kv_base = b * H * T * D + h * T * D;
    for (uint t_k = 0; t_k < T; t_k++) {
        if (causal && t_k > t_q) continue;
        float dot = 0;
        uint kv_offset = kv_base + t_k * D;
        for (uint dd = 0; dd < D; dd++)
            dot += Q[q_offset + dd] * K[kv_offset + dd];
        dot *= scale;
        if (dot > mv) mv = dot;
    }
    float sum_exp = 0;
    float acc = 0;
    for (uint t_k = 0; t_k < T; t_k++) {
        if (causal && t_k > t_q) continue;
        float dot = 0;
        uint kv_offset = kv_base + t_k * D;
        for (uint dd = 0; dd < D; dd++)
            dot += Q[q_offset + dd] * K[kv_offset + dd];
        dot *= scale;
        float w = exp(dot - mv);
        sum_exp += w;
        acc += w * V[kv_offset + d];
    }
    out[t.x] = sum_exp > 0 ? acc / sum_exp : 0;
})";

namespace {

static void throw_hr(HRESULT hr, const char* msg) {
    if (FAILED(hr)) {
        char buf[512];
        sprintf_s(buf, 512, "%s (HR=0x%08X)", msg, (unsigned)hr);
        throw std::runtime_error(buf);
    }
}

static ComPtr<ID3DBlob> compile_cs_full(const char* src, const char* entry) {
    ComPtr<ID3DBlob> blob, err;
    HRESULT hr = D3DCompile(src, strlen(src), nullptr, nullptr, nullptr,
                            entry, "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &err);
    if (FAILED(hr)) {
        const char* msg = err ? (const char*)err->GetBufferPointer() : "unknown compile error";
        throw std::runtime_error(std::string("HLSL compile: ") + msg);
    }
    return blob;
}

} // anonymous namespace

struct GPUComputeFull::Impl {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE fenceEvent = nullptr;
    UINT64 fenceVal = 1;
    bool ok = false;

    ComPtr<ID3D12RootSignature> rootSig;
    ComPtr<ID3D12DescriptorHeap> heap;
    UINT heapIncSize = 0;

    struct GpuBuf {
        ComPtr<ID3D12Resource> res;
        size_t size = 0;
        bool in_use = false;
    };
    std::vector<GpuBuf> all_bufs;
    std::vector<int64_t> free_list;
    std::mutex mtx;

    std::vector<std::pair<ComPtr<ID3D12CommandAllocator>, ComPtr<ID3D12GraphicsCommandList>>> streams;

    std::unordered_map<std::string, KernelProfile> profiles;

    int64_t total_allocated = 0;

    ~Impl() { shutdown(); }

    void init(int64_t) {
        if (ok) return;
        HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
        if (FAILED(hr))
            throw std::runtime_error("GPU init: D3D12CreateDevice failed");

        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        hr = device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
        if (FAILED(hr)) throw std::runtime_error("GPU init: CreateCommandQueue failed");

        hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
        if (FAILED(hr)) throw std::runtime_error("GPU init: CreateCommandAllocator failed");

        hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
                                       nullptr, IID_PPV_ARGS(&list));
        if (FAILED(hr)) throw std::runtime_error("GPU init: CreateCommandList failed");
        list->Close();

        hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        if (FAILED(hr)) throw std::runtime_error("GPU init: CreateFence failed");

        fenceEvent = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        if (!fenceEvent) throw std::runtime_error("GPU init: CreateEvent failed");

        D3D12_DESCRIPTOR_RANGE dr[5] = {};
        dr[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        dr[0].NumDescriptors = 1;
        dr[0].BaseShaderRegister = 0;
        dr[0].RegisterSpace = 0;
        dr[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

        for (int i = 1; i < 5; i++) {
            dr[i].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            dr[i].NumDescriptors = 1;
            dr[i].BaseShaderRegister = (UINT)(i - 1);
            dr[i].RegisterSpace = 0;
            dr[i].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        }

        D3D12_ROOT_PARAMETER rp[6] = {};
        for (int i = 0; i < 5; i++) {
            rp[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rp[i].DescriptorTable.NumDescriptorRanges = 1;
            rp[i].DescriptorTable.pDescriptorRanges = &dr[i];
            rp[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }
        rp[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rp[5].Constants.ShaderRegister = 0;
        rp[5].Constants.RegisterSpace = 0;
        rp[5].Constants.Num32BitValues = 8;
        rp[5].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC rsd = {};
        rsd.NumParameters = 6;
        rsd.pParameters = rp;
        rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

        ComPtr<ID3DBlob> sig, err;
        hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
        if (FAILED(hr)) throw std::runtime_error("GPU init: D3D12SerializeRootSignature failed");
        hr = device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                         IID_PPV_ARGS(&rootSig));
        if (FAILED(hr)) throw std::runtime_error("GPU init: CreateRootSignature failed");

        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = 64;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr = device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap));
        if (FAILED(hr)) throw std::runtime_error("GPU init: CreateDescriptorHeap failed");
        heapIncSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        ok = true;
    }

    void flush() {
        if (!ok) return;
        list->Close();
        ID3D12CommandList* cls[] = { list.Get() };
        queue->ExecuteCommandLists(1, cls);
        queue->Signal(fence.Get(), fenceVal);
        fence->SetEventOnCompletion(fenceVal, fenceEvent);
        WaitForSingleObject(fenceEvent, INFINITE);
        fenceVal++;
        allocator->Reset();
        list->Reset(allocator.Get(), nullptr);
    }

    void shutdown() {
        if (fenceEvent) { CloseHandle(fenceEvent); fenceEvent = nullptr; }
        all_bufs.clear();
        free_list.clear();
        heap.Reset();
        rootSig.Reset();
        list.Reset();
        allocator.Reset();
        queue.Reset();
        fence.Reset();
        device.Reset();
        ok = false;
    }

    ComPtr<ID3D12Resource> create_buffer(size_t sz) {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = (UINT64)sz;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        rd.SampleDesc.Count = 1;
        ComPtr<ID3D12Resource> r;
        device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r));
        return r;
    }

    void place_srv(UINT slot, ID3D12Resource* res, UINT64 size) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Format = DXGI_FORMAT_UNKNOWN;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Buffer.FirstElement = 0;
        d.Buffer.NumElements = (UINT)(size / 4);
        d.Buffer.StructureByteStride = 4;
        D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
        h.ptr += (SIZE_T)slot * heapIncSize;
        device->CreateShaderResourceView(res, &d, h);
    }

    void place_uav(UINT slot, ID3D12Resource* res, UINT64 size) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        d.Format = DXGI_FORMAT_UNKNOWN;
        d.Buffer.FirstElement = 0;
        d.Buffer.NumElements = (UINT)(size / 4);
        d.Buffer.StructureByteStride = 4;
        D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
        h.ptr += (SIZE_T)slot * heapIncSize;
        device->CreateUnorderedAccessView(res, nullptr, &d, h);
    }

    ComPtr<ID3D12PipelineState> make_pso(const char* src, const char* entry) {
        auto cs = compile_cs_full(src, entry);
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = rootSig.Get();
        pd.CS.pShaderBytecode = cs->GetBufferPointer();
        pd.CS.BytecodeLength = cs->GetBufferSize();
        ComPtr<ID3D12PipelineState> pso;
        device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso));
        return pso;
    }

    void dispatch_kernel(const ComPtr<ID3D12PipelineState>& pso,
                         int nUAV, ID3D12Resource* const* uavs, const UINT64* uavSizes,
                         int nSRV, ID3D12Resource* const* srvs, const UINT64* srvSizes,
                         const UINT* rc, int gx, int gy, int gz) {
        int slot = 0;
        for (int i = 0; i < nUAV; i++, slot++)
            place_uav((UINT)slot, uavs[i], uavSizes[i]);
        for (int i = 0; i < nSRV; i++, slot++)
            place_srv((UINT)slot, srvs[i], srvSizes[i]);

        list->SetPipelineState(pso.Get());
        list->SetComputeRootSignature(rootSig.Get());
        ID3D12DescriptorHeap* h = heap.Get();
        list->SetDescriptorHeaps(1, &h);

        D3D12_GPU_DESCRIPTOR_HANDLE gh = heap->GetGPUDescriptorHandleForHeapStart();
        list->SetComputeRootDescriptorTable(0, gh);

        for (int i = 1; i <= nSRV; i++) {
            D3D12_GPU_DESCRIPTOR_HANDLE sh = heap->GetGPUDescriptorHandleForHeapStart();
            sh.ptr += (SIZE_T)i * heapIncSize;
            list->SetComputeRootDescriptorTable(i, sh);
        }

        list->SetComputeRoot32BitConstants(5, 8, rc, 0);
        list->Dispatch(gx, gy, gz);
    }

    void* find_buf_ptr(void* ptr) {
        for (auto& b : all_bufs)
            if ((void*)b.res.Get() == (void*)ptr) return b.res.Get();
        return nullptr;
    }

    void record_profile(const char* name, double ms) {
        auto& p = profiles[name];
        p.name = name;
        p.total_ms += ms;
        p.call_count++;
    }
};

GPUComputeFull::GPUComputeFull() : impl_(new Impl()) {}
GPUComputeFull::~GPUComputeFull() { delete impl_; }

bool GPUComputeFull::init(int64_t device_id) {
    if (impl_->ok) return true;
    impl_->init(device_id);
    return impl_->ok;
}

bool GPUComputeFull::is_initialized() const { return impl_->ok; }
bool GPUComputeFull::has_gpu() const { return impl_->ok; }

void GPUComputeFull::shutdown() {
    if (impl_) impl_->shutdown();
}

void* GPUComputeFull::alloc(int64_t bytes) {
    if (!impl_->ok || bytes <= 0) return nullptr;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    for (auto& b : impl_->all_bufs) {
        if (!b.in_use && b.size >= (size_t)bytes) {
            b.in_use = true;
            impl_->total_allocated += bytes;
            return b.res.Get();
        }
    }
    try {
        auto res = impl_->create_buffer((size_t)bytes);
        Impl::GpuBuf gb;
        gb.res = res;
        gb.size = (size_t)bytes;
        gb.in_use = true;
        void* ptr = gb.res.Get();
        impl_->all_bufs.push_back(std::move(gb));
        impl_->total_allocated += bytes;
        return ptr;
    } catch (...) { std::fprintf(stderr, "[WARN] Exception caught: %s (GPU full alloc failed)\n", __func__); return nullptr; }
}

void GPUComputeFull::free_buf(void* ptr) {
    if (!impl_ || !ptr) return;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    for (auto& b : impl_->all_bufs) {
        if ((void*)b.res.Get() == (void*)ptr) {
            b.in_use = false;
            return;
        }
    }
}

GPUBufferPoolStats GPUComputeFull::buffer_pool_stats() const {
    GPUBufferPoolStats st;
    st.total_bytes = 8LL * 1024 * 1024 * 1024;
    st.allocated_bytes = impl_->total_allocated;
    st.free_bytes = st.total_bytes - st.allocated_bytes;
    st.num_allocations = 0;
    st.num_free_blocks = 0;
    for (auto& b : impl_->all_bufs) {
        st.num_allocations++;
        if (!b.in_use) st.num_free_blocks++;
    }
    return st;
}

int64_t GPUComputeFull::memory_free() const { return buffer_pool_stats().free_bytes; }
int64_t GPUComputeFull::memory_total() const { return buffer_pool_stats().total_bytes; }

void GPUComputeFull::upload(const Tensor& src, void* dst) {
    if (!impl_->ok || !dst || src.numel() == 0) return;
    size_t sz = src.size_bytes();
    const void* data = src.data();
    if (!data || !sz) return;

    ID3D12Resource* gpuRes = nullptr;
    for (auto& b : impl_->all_bufs)
        if ((void*)b.res.Get() == (void*)dst) { gpuRes = b.res.Get(); break; }
    if (!gpuRes) return;

    auto upload_buf = impl_->create_buffer(sz);
    void* mapped;
    upload_buf->Map(0, nullptr, &mapped);
    memcpy(mapped, data, sz);
    upload_buf->Unmap(0, nullptr);

    impl_->list->Reset(impl_->allocator.Get(), nullptr);
    impl_->list->CopyBufferRegion(gpuRes, 0, upload_buf.Get(), 0, sz);
    impl_->flush();
}

void GPUComputeFull::download(void* src, Tensor& dst) {
    if (!impl_->ok || !src || dst.numel() == 0) return;
    size_t sz = dst.size_bytes();
    void* data = dst.data();
    if (!data || !sz) return;

    ID3D12Resource* gpuRes = nullptr;
    for (auto& b : impl_->all_bufs)
        if ((void*)b.res.Get() == (void*)src) { gpuRes = b.res.Get(); break; }
    if (!gpuRes) return;

    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = (UINT64)sz;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.SampleDesc.Count = 1;
    ComPtr<ID3D12Resource> readback;
    impl_->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                           D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback));
    impl_->list->Reset(impl_->allocator.Get(), nullptr);
    impl_->list->CopyBufferRegion(readback.Get(), 0, gpuRes, 0, sz);
    impl_->flush();

    D3D12_RANGE rr = { 0, sz };
    void* mapped;
    readback->Map(0, &rr, &mapped);
    memcpy(data, mapped, sz);
    readback->Unmap(0, nullptr);
}

void GPUComputeFull::gpu_to_gpu(void* dst, const void* src, int64_t bytes) {
    if (!impl_->ok || !dst || !src || bytes <= 0) return;
    ID3D12Resource *srcRes = nullptr, *dstRes = nullptr;
    for (auto& b : impl_->all_bufs) {
        if ((void*)b.res.Get() == (void*)src) srcRes = b.res.Get();
        if ((void*)b.res.Get() == (void*)dst) dstRes = b.res.Get();
    }
    if (!srcRes || !dstRes) return;
    impl_->list->Reset(impl_->allocator.Get(), nullptr);
    impl_->list->CopyBufferRegion(dstRes, 0, srcRes, 0, (UINT64)bytes);
    impl_->flush();
}

void GPUComputeFull::gemm_tiled(float alpha, const void* A, const void* B,
                                 float beta, void* C, int64_t M, int64_t N,
                                 int64_t K, int64_t tile_size) {
    if (!impl_->ok) return;
    // BUGFIX (bug census): tile_size was silently discarded while rc[3] = 16
    // hardcoded the shader tile — callers passing 8/32 got 16 anyway. Only
    // 16 is compiled into g_tiled_gemm_cs, so validate instead of pretending.
    if (tile_size != 16)
        throw std::runtime_error("GPUComputeFull::gemm_tiled: only tile_size=16 supported by the compiled shader");

    Impl::GpuBuf *bA = nullptr, *bB = nullptr, *bC = nullptr;
    for (auto& b : impl_->all_bufs) {
        if ((void*)b.res.Get() == (void*)A) bA = &b;
        if ((void*)b.res.Get() == (void*)B) bB = &b;
        if ((void*)b.res.Get() == (void*)C) bC = &b;
    }
    if (!bA || !bB || !bC) return;

    auto pso = impl_->make_pso(g_tiled_gemm_cs, "main");
    ID3D12Resource* uavs[] = { bC->res.Get() };
    UINT64 uavSizes[] = { bC->size };
    ID3D12Resource* srvs[] = { bA->res.Get(), bB->res.Get() };
    UINT64 srvSizes[] = { bA->size, bB->size };
    UINT rc[8];
    rc[0] = (UINT)M; rc[1] = (UINT)N; rc[2] = (UINT)K; rc[3] = 16;
    memcpy(&rc[4], &alpha, 4); memcpy(&rc[5], &beta, 4);
    rc[6] = 0; rc[7] = 0;
    int gx = (int)((N + 15) / 16);
    int gy = (int)((M + 15) / 16);
    impl_->dispatch_kernel(pso, 1, uavs, uavSizes, 2, srvs, srvSizes, rc, gx, gy, 1);
    impl_->flush();
}

void GPUComputeFull::gemm_async(float alpha, const void* A, const void* B,
                                 float beta, void* C, int64_t M, int64_t N, int64_t K) {
    gemm_tiled(alpha, A, B, beta, C, M, N, K, 16);
}

void GPUComputeFull::element_add(const void* a, const void* b, void* c, int64_t n) {
    if (!impl_->ok) return;
    Impl::GpuBuf *ba = nullptr, *bb = nullptr, *bc = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)a) ba = &buf;
        if ((void*)buf.res.Get() == (void*)b) bb = &buf;
        if ((void*)buf.res.Get() == (void*)c) bc = &buf;
    }
    if (!ba || !bb || !bc) return;
    auto pso = impl_->make_pso(g_gemm_cs, "main");
    ID3D12Resource* uavs[] = { bc->res.Get() };
    UINT64 uavSizes[] = { bc->size };
    ID3D12Resource* srvs[] = { ba->res.Get(), bb->res.Get() };
    UINT64 srvSizes[] = { ba->size, bb->size };
    UINT rc[8] = { (UINT)n, 0, 0, 0, 0, 0, 0, 0 };
    auto ps = impl_->make_pso(R"(
        cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float _f0,_f1,_f2,_f3; };
        RWStructuredBuffer<float> c : register(u0);
        StructuredBuffer<float> a : register(t0);
        StructuredBuffer<float> b : register(t1);
        [numthreads(256,1,1)]
        void main(uint3 t : SV_DispatchThreadID) {
            if (t.x >= N) return; c[t.x] = a[t.x] + b[t.x];
        })", "main");
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 2, srvs, srvSizes, rc, (int)((n + 255) / 256), 1, 1);
    impl_->flush();
}

void GPUComputeFull::element_sub(const void* a, const void* b, void* c, int64_t n) {
    if (!impl_->ok) return;
    Impl::GpuBuf *ba = nullptr, *bb = nullptr, *bc = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)a) ba = &buf;
        if ((void*)buf.res.Get() == (void*)b) bb = &buf;
        if ((void*)buf.res.Get() == (void*)c) bc = &buf;
    }
    if (!ba || !bb || !bc) return;
    auto ps = impl_->make_pso(g_sub_cs, "main");
    ID3D12Resource* uavs[] = { bc->res.Get() };
    UINT64 uavSizes[] = { bc->size };
    ID3D12Resource* srvs[] = { ba->res.Get(), bb->res.Get() };
    UINT64 srvSizes[] = { ba->size, bb->size };
    UINT rc[8] = { (UINT)n, 0, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 2, srvs, srvSizes, rc, (int)((n + 255) / 256), 1, 1);
    impl_->flush();
}

void GPUComputeFull::element_mul(const void* a, const void* b, void* c, int64_t n) {
    if (!impl_->ok) return;
    Impl::GpuBuf *ba = nullptr, *bb = nullptr, *bc = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)a) ba = &buf;
        if ((void*)buf.res.Get() == (void*)b) bb = &buf;
        if ((void*)buf.res.Get() == (void*)c) bc = &buf;
    }
    if (!ba || !bb || !bc) return;
    auto ps = impl_->make_pso(R"(
        cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float _f0,_f1,_f2,_f3; };
        RWStructuredBuffer<float> c : register(u0);
        StructuredBuffer<float> a : register(t0);
        StructuredBuffer<float> b : register(t1);
        [numthreads(256,1,1)]
        void main(uint3 t : SV_DispatchThreadID) {
            if (t.x >= N) return; c[t.x] = a[t.x] * b[t.x];
        })", "main");
    ID3D12Resource* uavs[] = { bc->res.Get() };
    UINT64 uavSizes[] = { bc->size };
    ID3D12Resource* srvs[] = { ba->res.Get(), bb->res.Get() };
    UINT64 srvSizes[] = { ba->size, bb->size };
    UINT rc[8] = { (UINT)n, 0, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 2, srvs, srvSizes, rc, (int)((n + 255) / 256), 1, 1);
    impl_->flush();
}

void GPUComputeFull::element_div(const void* a, const void* b, void* c, int64_t n) {
    if (!impl_->ok) return;
    Impl::GpuBuf *ba = nullptr, *bb = nullptr, *bc = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)a) ba = &buf;
        if ((void*)buf.res.Get() == (void*)b) bb = &buf;
        if ((void*)buf.res.Get() == (void*)c) bc = &buf;
    }
    if (!ba || !bb || !bc) return;
    auto ps = impl_->make_pso(g_div_cs, "main");
    ID3D12Resource* uavs[] = { bc->res.Get() };
    UINT64 uavSizes[] = { bc->size };
    ID3D12Resource* srvs[] = { ba->res.Get(), bb->res.Get() };
    UINT64 srvSizes[] = { ba->size, bb->size };
    UINT rc[8] = { (UINT)n, 0, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 2, srvs, srvSizes, rc, (int)((n + 255) / 256), 1, 1);
    impl_->flush();
}

void GPUComputeFull::element_add_scalar(const void* a, float s, void* c, int64_t n) {
    if (!impl_->ok) return;
    Impl::GpuBuf *ba = nullptr, *bc = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)a) ba = &buf;
        if ((void*)buf.res.Get() == (void*)c) bc = &buf;
    }
    if (!ba || !bc) return;
    auto ps = impl_->make_pso(g_add_scalar_cs, "main");
    ID3D12Resource* uavs[] = { bc->res.Get() };
    UINT64 uavSizes[] = { bc->size };
    ID3D12Resource* srvs[] = { ba->res.Get() };
    UINT64 srvSizes[] = { ba->size };
    UINT rc[8];
    rc[0] = (UINT)n; rc[1] = 0; rc[2] = 0; rc[3] = 0;
    memcpy(&rc[4], &s, 4);
    rc[5] = 0; rc[6] = 0; rc[7] = 0;
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, (int)((n + 255) / 256), 1, 1);
    impl_->flush();
}

void GPUComputeFull::activation_relu(const void* x, void* y, int64_t n) {
    if (!impl_->ok) return;
    Impl::GpuBuf *bx = nullptr, *by = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)x) bx = &buf;
        if ((void*)buf.res.Get() == (void*)y) by = &buf;
    }
    if (!bx || !by) return;
    auto ps = impl_->make_pso(R"(
        cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float _f0,_f1,_f2,_f3; };
        RWStructuredBuffer<float> y : register(u0);
        StructuredBuffer<float> x : register(t0);
        [numthreads(256,1,1)]
        void main(uint3 t : SV_DispatchThreadID) {
            if (t.x >= N) return; y[t.x] = max(x[t.x], 0);
        })", "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bx->res.Get() };
    UINT64 srvSizes[] = { bx->size };
    UINT rc[8] = { (UINT)n, 0, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, (int)((n + 255) / 256), 1, 1);
    impl_->flush();
}

void GPUComputeFull::activation_gelu(const void* x, void* y, int64_t n) {
    if (!impl_->ok) return;
    Impl::GpuBuf *bx = nullptr, *by = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)x) bx = &buf;
        if ((void*)buf.res.Get() == (void*)y) by = &buf;
    }
    if (!bx || !by) return;
    auto ps = impl_->make_pso(R"(
        cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float _f0,_f1,_f2,_f3; };
        RWStructuredBuffer<float> y : register(u0);
        StructuredBuffer<float> x : register(t0);
        [numthreads(256,1,1)]
        void main(uint3 t : SV_DispatchThreadID) {
            if (t.x >= N) return;
            float v = x[t.x];
            y[t.x] = 0.5f * v * (1.0f + tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        })", "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bx->res.Get() };
    UINT64 srvSizes[] = { bx->size };
    UINT rc[8] = { (UINT)n, 0, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, (int)((n + 255) / 256), 1, 1);
    impl_->flush();
}

void GPUComputeFull::activation_silu(const void* x, void* y, int64_t n) {
    if (!impl_->ok) return;
    Impl::GpuBuf *bx = nullptr, *by = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)x) bx = &buf;
        if ((void*)buf.res.Get() == (void*)y) by = &buf;
    }
    if (!bx || !by) return;
    auto ps = impl_->make_pso(R"(
        cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float _f0,_f1,_f2,_f3; };
        RWStructuredBuffer<float> y : register(u0);
        StructuredBuffer<float> x : register(t0);
        [numthreads(256,1,1)]
        void main(uint3 t : SV_DispatchThreadID) {
            if (t.x >= N) return; float v = x[t.x]; y[t.x] = v / (1.0f + exp(-v));
        })", "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bx->res.Get() };
    UINT64 srvSizes[] = { bx->size };
    UINT rc[8] = { (UINT)n, 0, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, (int)((n + 255) / 256), 1, 1);
    impl_->flush();
}

void GPUComputeFull::activation_tanh(const void* x, void* y, int64_t n) {
    if (!impl_->ok) return;
    Impl::GpuBuf *bx = nullptr, *by = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)x) bx = &buf;
        if ((void*)buf.res.Get() == (void*)y) by = &buf;
    }
    if (!bx || !by) return;
    auto ps = impl_->make_pso(g_tanh_cs, "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bx->res.Get() };
    UINT64 srvSizes[] = { bx->size };
    UINT rc[8] = { (UINT)n, 0, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, (int)((n + 255) / 256), 1, 1);
    impl_->flush();
}

void GPUComputeFull::activation_sigmoid(const void* x, void* y, int64_t n) {
    if (!impl_->ok) return;
    Impl::GpuBuf *bx = nullptr, *by = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)x) bx = &buf;
        if ((void*)buf.res.Get() == (void*)y) by = &buf;
    }
    if (!bx || !by) return;
    auto ps = impl_->make_pso(g_sigmoid_cs, "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bx->res.Get() };
    UINT64 srvSizes[] = { bx->size };
    UINT rc[8] = { (UINT)n, 0, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, (int)((n + 255) / 256), 1, 1);
    impl_->flush();
}

void GPUComputeFull::activation_softmax(const void* x, void* y, int64_t rows, int64_t cols) {
    softmax_stable(x, y, rows, cols);
}

void GPUComputeFull::reduce_sum(const float* x, float* out, int64_t n) {
    if (!impl_->ok) return;
    if (n > 256) {
        fprintf(stderr, "reduce_sum: N=%lld exceeds single-pass limit of 256\n", (long long)n);
        return;
    }
    Impl::GpuBuf *bx = nullptr, *bo = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)x) bx = &buf;
        if ((void*)buf.res.Get() == (void*)out) bo = &buf;
    }
    if (!bx || !bo) return;
    auto ps = impl_->make_pso(g_reduce_sum_cs, "main");
    ID3D12Resource* uavs[] = { bo->res.Get() };
    UINT64 uavSizes[] = { bo->size };
    ID3D12Resource* srvs[] = { bx->res.Get() };
    UINT64 srvSizes[] = { bx->size };
    UINT rc[8] = { (UINT)n, 0, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, 1, 1, 1);
    impl_->flush();
}

void GPUComputeFull::reduce_max(const float* x, float* out, int64_t n) {
    if (!impl_->ok) return;
    Impl::GpuBuf *bx = nullptr, *bo = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)x) bx = &buf;
        if ((void*)buf.res.Get() == (void*)out) bo = &buf;
    }
    if (!bx || !bo) return;
    auto ps = impl_->make_pso(g_reduce_max_cs, "main");
    ID3D12Resource* uavs[] = { bo->res.Get() };
    UINT64 uavSizes[] = { bo->size };
    ID3D12Resource* srvs[] = { bx->res.Get() };
    UINT64 srvSizes[] = { bx->size };
    UINT rc[8] = { (UINT)n, 0, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, 1, 1, 1);
    impl_->flush();
}

void GPUComputeFull::reduce_mean(const float* x, float* out, int64_t n) {
    reduce_sum(x, out, n);
    if (impl_->ok && n > 0) {
        float inv_n = 1.0f / (float)n;
        Impl::GpuBuf *bo = nullptr;
        for (auto& buf : impl_->all_bufs)
            if ((void*)buf.res.Get() == (void*)out) { bo = &buf; break; }
        if (bo) {
            auto ps = impl_->make_pso(R"(
                cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float s,_f1,_f2,_f3; };
                RWStructuredBuffer<float> y : register(u0);
                StructuredBuffer<float> x : register(t0);
                [numthreads(1,1,1)]
                void main(uint3 t : SV_DispatchThreadID) { y[0] = x[0] * s; })", "main");
            ID3D12Resource* uavs[] = { bo->res.Get() };
            UINT64 uavSizes[] = { bo->size };
            ID3D12Resource* srvs[] = { bo->res.Get() };
            UINT64 srvSizes[] = { bo->size };
            UINT rc[8];
            rc[0] = 1; rc[1] = 0; rc[2] = 0; rc[3] = 0;
            memcpy(&rc[4], &inv_n, 4); rc[5] = 0; rc[6] = 0; rc[7] = 0;
            impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, 1, 1, 1);
            impl_->flush();
        }
    }
}

void GPUComputeFull::reduce_sum_axis(const float* x, float* out, int64_t rows,
                                     int64_t cols, int axis) {
    // BUGFIX (bug census): axis was silently discarded — the shader reduces
    // rows (axis=1 semantics). Validate instead of miscomputing axis=0.
    if (axis != 1 && axis != -1)
        throw std::runtime_error("GPUComputeFull::reduce_sum_axis: only axis=1/-1 supported by the shader");
    if (!impl_->ok) return;
    Impl::GpuBuf *bx = nullptr, *bo = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)x) bx = &buf;
        if ((void*)buf.res.Get() == (void*)out) bo = &buf;
    }
    if (!bx || !bo) return;
    auto ps = impl_->make_pso(g_reduce_axis_cs, "main");
    ID3D12Resource* uavs[] = { bo->res.Get() };
    UINT64 uavSizes[] = { bo->size };
    ID3D12Resource* srvs[] = { bx->res.Get() };
    UINT64 srvSizes[] = { bx->size };
    UINT rc[8] = { (UINT)rows, (UINT)cols, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, (int)rows, 1, 1);
    impl_->flush();
}

void GPUComputeFull::reduce_max_axis(const float* x, float* out, int64_t rows,
                                     int64_t cols, int axis) {
    // BUGFIX (bug census): same axis-discard as reduce_sum_axis (shader is
    // row-reduce only).
    if (axis != 1 && axis != -1)
        throw std::runtime_error("GPUComputeFull::reduce_max_axis: only axis=1/-1 supported by the shader");
    if (!impl_->ok) return;
    Impl::GpuBuf *bx = nullptr, *bo = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)x) bx = &buf;
        if ((void*)buf.res.Get() == (void*)out) bo = &buf;
    }
    if (!bx || !bo) return;
    auto ps = impl_->make_pso(R"(
        cbuffer Const : register(b0) { uint rows,cols,_p0,_p1; float _f0,_f1,_f2,_f3; };
        RWStructuredBuffer<float> out : register(u0);
        StructuredBuffer<float> x : register(t0);
        [numthreads(256,1,1)]
        void main(uint3 t : SV_DispatchThreadID) {
            if (t.x >= rows) return;
            float mx = -1e30;
            for (uint c = 0; c < cols; c++) mx = max(mx, x[t.x * cols + c]);
            out[t.x] = mx;
        })", "main");
    ID3D12Resource* uavs[] = { bo->res.Get() };
    UINT64 uavSizes[] = { bo->size };
    ID3D12Resource* srvs[] = { bx->res.Get() };
    UINT64 srvSizes[] = { bx->size };
    UINT rc[8] = { (UINT)rows, (UINT)cols, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, (int)rows, 1, 1);
    impl_->flush();
}

void GPUComputeFull::attention_fwd(const void* Q, const void* K, const void* V,
                                    void* out, int64_t B, int64_t H, int64_t T,
                                    int64_t D, bool causal) {
    if (!impl_->ok) return;
    Impl::GpuBuf *bQ = nullptr, *bK = nullptr, *bV = nullptr, *bO = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)Q) bQ = &buf;
        if ((void*)buf.res.Get() == (void*)K) bK = &buf;
        if ((void*)buf.res.Get() == (void*)V) bV = &buf;
        if ((void*)buf.res.Get() == (void*)out) bO = &buf;
    }
    if (!bQ || !bK || !bV || !bO) return;
    auto ps = impl_->make_pso(g_cross_attn_cs, "main");
    ID3D12Resource* uavs[] = { bO->res.Get() };
    UINT64 uavSizes[] = { bO->size };
    ID3D12Resource* srvs[] = { bQ->res.Get(), bK->res.Get(), bV->res.Get() };
    UINT64 srvSizes[] = { bQ->size, bK->size, bV->size };
    UINT rc[8];
    rc[0] = (UINT)B; rc[1] = (UINT)H; rc[2] = (UINT)T; rc[3] = (UINT)T;
    rc[4] = (UINT)D; rc[5] = (UINT)(causal ? 1 : 0); rc[6] = 0; rc[7] = 0;
    int64_t total = B * H * T * D;
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 3, srvs, srvSizes, rc,
                           (int)((total + 255) / 256), 1, 1);
    impl_->flush();
}

void GPUComputeFull::attention_cross(const void* Q, const void* KV, void* out,
                                      int64_t B, int64_t H, int64_t Tq, int64_t Tk, int64_t D) {
    if (!impl_->ok) return;
    Impl::GpuBuf *bQ = nullptr, *bK = nullptr, *bO = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)Q) bQ = &buf;
        if ((void*)buf.res.Get() == (void*)KV) bK = &buf;
        if ((void*)buf.res.Get() == (void*)out) bO = &buf;
    }
    if (!bQ || !bK || !bO) return;
    auto ps = impl_->make_pso(g_cross_attn_cs, "main");
    ID3D12Resource* uavs[] = { bO->res.Get() };
    UINT64 uavSizes[] = { bO->size };
    ID3D12Resource* srvs[] = { bQ->res.Get(), bK->res.Get() };
    UINT64 srvSizes[] = { bQ->size, bK->size };
    UINT rc[8];
    rc[0] = (UINT)B; rc[1] = (UINT)H; rc[2] = (UINT)Tq; rc[3] = (UINT)Tk;
    rc[4] = (UINT)D; rc[5] = 0; rc[6] = 0; rc[7] = 0;
    int64_t total = B * H * Tq * D;
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 2, srvs, srvSizes, rc,
                           (int)((total + 255) / 256), 1, 1);
    impl_->flush();
}

void GPUComputeFull::layer_norm(const void* x, const void* gamma, const void* beta,
                                 void* y, float eps, int64_t n, int64_t d) {
    if (!impl_->ok) return;
    Impl::GpuBuf *bx = nullptr, *bg = nullptr, *bb = nullptr, *by = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)x) bx = &buf;
        if ((void*)buf.res.Get() == (void*)gamma) bg = &buf;
        if ((void*)buf.res.Get() == (void*)beta) bb = &buf;
        if ((void*)buf.res.Get() == (void*)y) by = &buf;
    }
    if (!bx || !bg || !bb || !by) return;
    auto ps = impl_->make_pso(g_layernorm_cs, "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bx->res.Get(), bg->res.Get(), bb->res.Get() };
    UINT64 srvSizes[] = { bx->size, bg->size, bb->size };
    UINT rc[8];
    rc[0] = (UINT)n; rc[1] = (UINT)d; rc[2] = 0; rc[3] = 0;
    memcpy(&rc[4], &eps, 4);
    rc[5] = 0; rc[6] = 0; rc[7] = 0;
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 3, srvs, srvSizes, rc, (int)n, 1, 1);
    impl_->flush();
}

void GPUComputeFull::rms_norm(const void* x, const void* gamma, void* y,
                               float eps, int64_t n, int64_t d) {
    if (!impl_->ok) return;
    Impl::GpuBuf *bx = nullptr, *bg = nullptr, *by = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)x) bx = &buf;
        if ((void*)buf.res.Get() == (void*)gamma) bg = &buf;
        if ((void*)buf.res.Get() == (void*)y) by = &buf;
    }
    if (!bx || !bg || !by) return;
    auto ps = impl_->make_pso(g_rms_norm_cs, "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bx->res.Get(), bg->res.Get() };
    UINT64 srvSizes[] = { bx->size, bg->size };
    UINT rc[8];
    rc[0] = (UINT)n; rc[1] = (UINT)d; rc[2] = 0; rc[3] = 0;
    memcpy(&rc[4], &eps, 4);
    rc[5] = 0; rc[6] = 0; rc[7] = 0;
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 2, srvs, srvSizes, rc, (int)n, 1, 1);
    impl_->flush();
}

void GPUComputeFull::batch_norm(const void* x, const void* gamma, const void* beta,
                                 void* y, int64_t n, int64_t c, int64_t hw) {
    if (!impl_->ok) return;
    Impl::GpuBuf *bx = nullptr, *bg = nullptr, *bb = nullptr, *by = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)x) bx = &buf;
        if ((void*)buf.res.Get() == (void*)gamma) bg = &buf;
        if ((void*)buf.res.Get() == (void*)beta) bb = &buf;
        if ((void*)buf.res.Get() == (void*)y) by = &buf;
    }
    if (!bx || !bg || !bb || !by) return;
    auto ps = impl_->make_pso(g_batch_norm_cs, "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bx->res.Get(), bg->res.Get(), bb->res.Get() };
    UINT64 srvSizes[] = { bx->size, bg->size, bb->size };
    UINT rc[8] = { (UINT)n, (UINT)c, (UINT)hw, 0, 0, 0, 0, 0 };
    int64_t total = n * c * hw;
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 3, srvs, srvSizes, rc,
                           (int)((total + 255) / 256), 1, 1);
    impl_->flush();
}

void GPUComputeFull::softmax_stable(const void* x, void* y, int64_t rows, int64_t cols) {
    if (!impl_->ok) return;
    Impl::GpuBuf *bx = nullptr, *by = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)x) bx = &buf;
        if ((void*)buf.res.Get() == (void*)y) by = &buf;
    }
    if (!bx || !by) return;
    auto ps = impl_->make_pso(g_softmax_stable_cs, "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bx->res.Get() };
    UINT64 srvSizes[] = { bx->size };
    UINT rc[8] = { (UINT)rows, (UINT)cols, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, (int)rows, 1, 1);
    impl_->flush();
}

void GPUComputeFull::synchronize() {
    if (impl_->ok) impl_->flush();
}

void GPUComputeFull::profile_reset() {
    impl_->profiles.clear();
}

std::vector<KernelProfile> GPUComputeFull::profile_dump() const {
    std::vector<KernelProfile> result;
    result.reserve(impl_->profiles.size());
    for (auto& kv : impl_->profiles)
        result.push_back(kv.second);
    std::sort(result.begin(), result.end(),
              [](const KernelProfile& a, const KernelProfile& b) { return a.total_ms > b.total_ms; });
    return result;
}

void GPUComputeFull::async_memcpy_h2d(void* dst, const void* src, int64_t bytes) {
    if (!impl_->ok || !dst || !src || bytes <= 0) return;
    ID3D12Resource* gpuRes = nullptr;
    for (auto& b : impl_->all_bufs)
        if ((void*)b.res.Get() == (void*)dst) { gpuRes = b.res.Get(); break; }
    if (!gpuRes) return;

    auto upload_buf = impl_->create_buffer((size_t)bytes);
    void* mapped;
    upload_buf->Map(0, nullptr, &mapped);
    memcpy(mapped, src, (size_t)bytes);
    upload_buf->Unmap(0, nullptr);

    impl_->list->Reset(impl_->allocator.Get(), nullptr);
    impl_->list->CopyBufferRegion(gpuRes, 0, upload_buf.Get(), 0, (UINT64)bytes);
    impl_->flush();
}

void GPUComputeFull::async_memcpy_d2h(void* dst, const void* src, int64_t bytes) {
    if (!impl_->ok || !dst || !src || bytes <= 0) return;
    ID3D12Resource* gpuRes = nullptr;
    for (auto& b : impl_->all_bufs)
        if ((void*)b.res.Get() == (void*)src) { gpuRes = b.res.Get(); break; }
    if (!gpuRes) return;

    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = (UINT64)bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.SampleDesc.Count = 1;
    ComPtr<ID3D12Resource> readback;
    impl_->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                           D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback));
    impl_->list->Reset(impl_->allocator.Get(), nullptr);
    impl_->list->CopyBufferRegion(readback.Get(), 0, gpuRes, 0, (UINT64)bytes);
    impl_->flush();

    D3D12_RANGE rr = { 0, (SIZE_T)bytes };
    void* mapped;
    readback->Map(0, &rr, &mapped);
    memcpy(dst, mapped, (size_t)bytes);
    readback->Unmap(0, nullptr);
}

void GPUComputeFull::stream_synchronize(int stream_idx) {
    (void)stream_idx;
    synchronize();
}

int GPUComputeFull::create_stream() {
    if (!impl_->ok) return -1;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cmd;
    impl_->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
    impl_->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(),
                                     nullptr, IID_PPV_ARGS(&cmd));
    cmd->Close();
    int idx = (int)impl_->streams.size();
    impl_->streams.push_back({std::move(alloc), std::move(cmd)});
    return idx;
}

void GPUComputeFull::destroy_stream(int stream_idx) {
    if (stream_idx >= 0 && stream_idx < (int)impl_->streams.size()) {
        impl_->streams[(size_t)stream_idx].second.Reset();
        impl_->streams[(size_t)stream_idx].first.Reset();
    }
}

void GPUComputeFull::transpose_2d(const void* x, void* y, int64_t rows, int64_t cols) {
    if (!impl_->ok) return;
    Impl::GpuBuf *bx = nullptr, *by = nullptr;
    for (auto& buf : impl_->all_bufs) {
        if ((void*)buf.res.Get() == (void*)x) bx = &buf;
        if ((void*)buf.res.Get() == (void*)y) by = &buf;
    }
    if (!bx || !by) return;
    auto ps = impl_->make_pso(g_transpose_cs, "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bx->res.Get() };
    UINT64 srvSizes[] = { bx->size };
    UINT rc[8] = { (UINT)rows, (UINT)cols, 0, 0, 0, 0, 0, 0 };
    int gx = (int)((cols + 15) / 16);
    int gy = (int)((rows + 15) / 16);
    impl_->dispatch_kernel(ps, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, gx, gy, 1);
    impl_->flush();
}

void GPUComputeFull::embedding_lookup(const float* table, const int64_t* indices,
                                       float* out, int64_t n, int64_t d) {
    if (!impl_->ok) return;
    for (int64_t i = 0; i < n; i++) {
        int64_t idx = indices[i];
        if (idx < 0) idx = 0;
        std::memcpy(out + i * d, table + idx * d, static_cast<size_t>(d) * sizeof(float));
    }
}

GPUComputeFull& GPUComputeFull::instance() {
    static GPUComputeFull s_inst;
    return s_inst;
}

void GPUComputeFull::CPUTensorFallback::gemm_fallback(float alpha, const void* A,
                                                       const void* B, float beta,
                                                       void* C, int64_t M,
                                                       int64_t N, int64_t K) {
    const float* Af = (const float*)A;
    const float* Bf = (const float*)B;
    float* Cf = (float*)C;
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            float s = 0;
            for (int64_t k = 0; k < K; k++)
                s += Af[m * K + k] * Bf[k * N + n];
            Cf[m * N + n] = alpha * s + beta * Cf[m * N + n];
        }
    }
}

void GPUComputeFull::CPUTensorFallback::softmax_fallback(const float* x, float* y,
                                                          int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; r++) {
        float mv = x[r * cols];
        for (int64_t c = 1; c < cols; c++)
            if (x[r * cols + c] > mv) mv = x[r * cols + c];
        float s = 0;
        for (int64_t c = 0; c < cols; c++) {
            y[r * cols + c] = std::exp(x[r * cols + c] - mv);
            s += y[r * cols + c];
        }
        float iv = 1.0f / (s + 1e-10f);
        for (int64_t c = 0; c < cols; c++)
            y[r * cols + c] *= iv;
    }
}

void GPUComputeFull::CPUTensorFallback::layernorm_fallback(const float* x, const float* gamma,
                                                            const float* beta, float* y,
                                                            float eps, int64_t n, int64_t d) {
    for (int64_t i = 0; i < n; i++) {
        float mn = 0;
        for (int64_t j = 0; j < d; j++) mn += x[i * d + j];
        mn /= d;
        float vr = 0;
        for (int64_t j = 0; j < d; j++) {
            float df = x[i * d + j] - mn;
            vr += df * df;
        }
        vr /= d;
        float iv = 1.0f / std::sqrt(vr + eps);
        for (int64_t j = 0; j < d; j++)
            y[i * d + j] = (x[i * d + j] - mn) * iv * gamma[j] + beta[j];
    }
}

void GPUComputeFull::CPUTensorFallback::rmsnorm_fallback(const float* x, const float* gamma,
                                                          float* y, float eps,
                                                          int64_t n, int64_t d) {
    for (int64_t i = 0; i < n; i++) {
        float ss = 0;
        for (int64_t j = 0; j < d; j++)
            ss += x[i * d + j] * x[i * d + j];
        float rs = 1.0f / std::sqrt(ss / d + eps);
        for (int64_t j = 0; j < d; j++)
            y[i * d + j] = x[i * d + j] * rs * gamma[j];
    }
}

} // namespace gpu
} // namespace quant
