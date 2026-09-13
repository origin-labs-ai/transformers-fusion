#include "quant/gpu_compute.h"
#include "quant/tensor.h"

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
#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <new>



using namespace Microsoft::WRL;

// ========================================================================
// HLSL compute shader sources (compiled at runtime via D3DCompile)
// ========================================================================

static const char* g_gemm_cs = R"(
cbuffer Const : register(b0) { uint M,N,K,_p0; float alpha,beta,_p1,_p2; };
RWStructuredBuffer<float> C : register(u0);
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
[numthreads(16,16,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if(t.x>=N||t.y>=M)return; float s=0;
    for(uint i=0;i<K;i++) s+=A[t.y*K+i]*B[i*N+t.x];
    C[t.y*N+t.x]=alpha*s+beta*C[t.y*N+t.x];
})";

static const char* g_gemv_cs = R"(
cbuffer Const : register(b0) { uint M,N,_p0,_p1; float alpha,beta,_p2,_p3; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> x : register(t1);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if(t.x>=M)return; float s=0;
    for(uint i=0;i<N;i++) s+=A[t.x*N+i]*x[i];
    y[t.x]=alpha*s+beta*y[t.x];
})";

static const char* g_relu_cs = R"(
cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if(t.x>=N)return; y[t.x]=max(x[t.x],0);
})";

static const char* g_gelu_cs = R"(
cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if(t.x>=N)return;
    float v=x[t.x];
    y[t.x]=0.5f*v*(1.0f+tanh(0.7978845608028654f*(v+0.044715f*v*v*v)));
})";

static const char* g_silu_cs = R"(
cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if(t.x>=N)return; float v=x[t.x]; y[t.x]=v/(1.0f+exp(-v));
})";

static const char* g_add_cs = R"(
cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> c : register(u0);
StructuredBuffer<float> a : register(t0);
StructuredBuffer<float> b : register(t1);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if(t.x>=N)return; c[t.x]=a[t.x]+b[t.x];
})";

static const char* g_mul_cs = R"(
cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> c : register(u0);
StructuredBuffer<float> a : register(t0);
StructuredBuffer<float> b : register(t1);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if(t.x>=N)return; c[t.x]=a[t.x]*b[t.x];
})";

static const char* g_scale_cs = R"(
cbuffer Const : register(b0) { uint N,_p0,_p1,_p2; float s,_f1,_f2,_f3; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if(t.x>=N)return; y[t.x]=s*x[t.x];
})";

static const char* g_softmax_cs = R"(
cbuffer Const : register(b0) { uint rows,cols,_p0,_p1; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if(t.x>=rows)return; uint r=t.x;
    float mv=x[r*cols]; for(uint c=1;c<cols;c++) mv=max(mv,x[r*cols+c]);
    float s=0; for(uint c=0;c<cols;c++){float e=exp(x[r*cols+c]-mv); y[r*cols+c]=e; s+=e;}
    float iv=1.0f/s; for(uint c=0;c<cols;c++) y[r*cols+c]*=iv;
})";

static const char* g_rms_norm_cs = R"(
cbuffer Const : register(b0) { uint N,D,_p0,_p1; float eps,_f1,_f2,_f3; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
StructuredBuffer<float> g : register(t1);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if(t.x>=N)return; float ss=0;
    for(uint d=0;d<D;d++){float v=x[t.x*D+d]; ss+=v*v;}
    float rs=rsqrt(ss/D+eps);
    for(uint d=0;d<D;d++) y[t.x*D+d]=x[t.x*D+d]*rs*g[d];
})";

static const char* g_layer_norm_cs = R"(
cbuffer Const : register(b0) { uint N,D,_p0,_p1; float eps,_f1,_f2,_f3; };
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
StructuredBuffer<float> g : register(t1);
StructuredBuffer<float> b : register(t2);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    if(t.x>=N)return; float mn=0;
    for(uint d=0;d<D;d++) mn+=x[t.x*D+d]; mn/=D;
    float vr=0; for(uint d=0;d<D;d++){float df=x[t.x*D+d]-mn; vr+=df*df;}
    vr/=D; float iv=rsqrt(vr+eps);
    for(uint d=0;d<D;d++) y[t.x*D+d]=(x[t.x*D+d]-mn)*iv*g[d]+b[d];
})";

static const char* g_moe_gather_cs = R"(
cbuffer Const : register(b0) { uint T,K,D,_p0; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> o : register(u0);
StructuredBuffer<float> x : register(t0);
StructuredBuffer<int64_t> idx : register(t1);
StructuredBuffer<float> w : register(t2);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    uint i=t.x; if(i>=T*K*D)return;
    uint d=i%D,k=(i/D)%K,tk=i/(K*D);
    uint ix=(uint)idx[tk*K+k];
    o[tk*K*D+k*D+d]=w[tk*K+k]*x[ix*D+d];
})";

static const char* g_moe_scatter_cs = R"(
cbuffer Const : register(b0) { uint T,K,D,_p0; float _f0,_f1,_f2,_f3; };
RWStructuredBuffer<float> o : register(u0);
StructuredBuffer<int64_t> idx : register(t1);
StructuredBuffer<float> w : register(t2);
StructuredBuffer<float> eo : register(t3);
[numthreads(256,1,1)]
void main(uint3 t : SV_DispatchThreadID) {
    uint i=t.x; if(i>=T*K*D)return;
    uint d=i%D,k=(i/D)%K,tk=i/(K*D);
    uint ix=(uint)idx[tk*K+k];
    float val=w[tk*K+k]*eo[tk*K*D+k*D+d];
    int old,comp,newv;
    comp=((int*)&o[ix*D+d])[0];
    do{old=comp;newv=asint(asfloat(old)+val);
       InterlockedCompareExchange((int*)&o[ix*D+d],old,newv,comp);
    }while(comp!=old);
})";

// ========================================================================
// D3D12 helpers
// ========================================================================

namespace {

static void throw_hr(HRESULT hr, const char* msg) {
    if (FAILED(hr)) {
        char buf[512];
        sprintf_s(buf, 512, "%s (HR=0x%08X)", msg, (unsigned)hr);
        throw std::runtime_error(buf);
    }
}

static ComPtr<ID3DBlob> compile_cs(const char* src, const char* entry) {
    ComPtr<ID3DBlob> blob, err;
    HRESULT hr = D3DCompile(src, strlen(src), nullptr, nullptr, nullptr,
                            entry, "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &err);
    if (FAILED(hr)) {
        const char* msg = err ? (const char*)err->GetBufferPointer() : "unknown";
        throw std::runtime_error(std::string("CS compile error: ") + msg);
    }
    return blob;
}

} // anonymous namespace

namespace quant {
namespace gpu {

// ========================================================================
// DirectXCompute::Impl
// ========================================================================

struct DirectXCompute::Impl {
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
    };
    std::vector<GpuBuf> bufs;
    std::mutex mtx;

    ~Impl() { shutdown(); }

    static Impl* create() { return new Impl(); }

    void init(int64_t) {
        if (ok) return;

        throw_hr(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)),
                 "D3D12CreateDevice");

        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        throw_hr(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "CreateCommandQueue");

        throw_hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
                 "CreateCommandAllocator");

        throw_hr(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
                                           nullptr, IID_PPV_ARGS(&list)),
                 "CreateCommandList");
        list->Close();

        throw_hr(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)),
                 "CreateFence");

        fenceEvent = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        if (!fenceEvent) throw std::runtime_error("CreateEvent failed");

        // Root signature: 5 descriptor tables (u0, t0, t1, t2, t3) + root constants (b0, 8 DWORDs)
        {
            D3D12_DESCRIPTOR_RANGE dr[5] = {};
            dr[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; dr[0].NumDescriptors = 1;
            dr[0].BaseShaderRegister = 0; dr[0].RegisterSpace = 0;
            dr[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

            dr[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; dr[1].NumDescriptors = 1;
            dr[1].BaseShaderRegister = 0; dr[1].RegisterSpace = 0;
            dr[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

            dr[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; dr[2].NumDescriptors = 1;
            dr[2].BaseShaderRegister = 1; dr[2].RegisterSpace = 0;
            dr[2].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

            dr[3].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; dr[3].NumDescriptors = 1;
            dr[3].BaseShaderRegister = 2; dr[3].RegisterSpace = 0;
            dr[3].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

            dr[4].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; dr[4].NumDescriptors = 1;
            dr[4].BaseShaderRegister = 3; dr[4].RegisterSpace = 0;
            dr[4].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

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
            throw_hr(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err),
                     "SerializeRootSignature");
            throw_hr(device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                                  IID_PPV_ARGS(&rootSig)),
                     "CreateRootSignature");
        }

        // Descriptor heap (64 entries)
        {
            D3D12_DESCRIPTOR_HEAP_DESC hd = {};
            hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            hd.NumDescriptors = 64;
            hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            throw_hr(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "CreateDescriptorHeap");
            heapIncSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }

        ok = true;
    }

    void flush() {
        if (!ok) return;
        throw_hr(list->Close(), "Close");
        ID3D12CommandList* cls[] = { list.Get() };
        queue->ExecuteCommandLists(1, cls);
        throw_hr(queue->Signal(fence.Get(), fenceVal), "Signal");
        throw_hr(fence->SetEventOnCompletion(fenceVal, fenceEvent), "SetEvent");
        WaitForSingleObject(fenceEvent, INFINITE);
        fenceVal++;
        throw_hr(allocator->Reset(), "ResetAlloc");
        throw_hr(list->Reset(allocator.Get(), nullptr), "ResetList");
    }

    void shutdown() {
        if (fenceEvent) { CloseHandle(fenceEvent); fenceEvent = nullptr; }
        bufs.clear();
        heap.Reset();
        rootSig.Reset();
        list.Reset();
        allocator.Reset();
        queue.Reset();
        fence.Reset();
        device.Reset();
        ok = false;
    }

    ComPtr<ID3D12Resource> create_buffer(size_t sz, D3D12_HEAP_TYPE ht, D3D12_RESOURCE_STATES initSt) {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = ht;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = (UINT64)sz;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = (ht == D3D12_HEAP_TYPE_DEFAULT) ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        rd.SampleDesc.Count = 1;
        ComPtr<ID3D12Resource> r;
        throw_hr(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, initSt, nullptr, IID_PPV_ARGS(&r)),
                 "CreateCommittedResource");
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
        auto cs = compile_cs(src, entry);
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = rootSig.Get();
        pd.CS.pShaderBytecode = cs->GetBufferPointer();
        pd.CS.BytecodeLength = cs->GetBufferSize();
        ComPtr<ID3D12PipelineState> pso;
        throw_hr(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)), "CreatePSO");
        return pso;
    }

    void dispatch(const ComPtr<ID3D12PipelineState>& pso,
                  int nUAV, ID3D12Resource* const* uavs, const UINT64* uavSizes,
                  int nSRV, ID3D12Resource* const* srvs, const UINT64* srvSizes,
                  const UINT* rootConsts,
                  int gx, int gy, int gz)
    {
        // Place descriptors in heap
        int slot = 0;
        for (int i = 0; i < nUAV; i++, slot++)
            place_uav((UINT)slot, uavs[i], uavSizes[i]);
        for (int i = 0; i < nSRV; i++, slot++)
            place_srv((UINT)slot, srvs[i], srvSizes[i]);

        list->SetPipelineState(pso.Get());
        list->SetComputeRootSignature(rootSig.Get());

        ID3D12DescriptorHeap* h = heap.Get();
        list->SetDescriptorHeaps(1, &h);

        // UAV table at root slot 0
        D3D12_GPU_DESCRIPTOR_HANDLE gh = heap->GetGPUDescriptorHandleForHeapStart();
        list->SetComputeRootDescriptorTable(0, gh);

        // SRV tables at root slots 1..nSRV
        for (int i = 1; i <= nSRV; i++) {
            D3D12_GPU_DESCRIPTOR_HANDLE sh = heap->GetGPUDescriptorHandleForHeapStart();
            sh.ptr += (SIZE_T)i * heapIncSize;
            list->SetComputeRootDescriptorTable(i, sh);
        }

        // Root constants at root slot 5
        list->SetComputeRoot32BitConstants(5, 8, rootConsts, 0);

        list->Dispatch(gx, gy, gz);
    }
};

// ========================================================================
// DirectXCompute public methods
// ========================================================================

DirectXCompute::DirectXCompute() : impl_(nullptr) {}
DirectXCompute::~DirectXCompute() { delete impl_; }

bool DirectXCompute::init(int64_t device_id) {
    if (impl_) return true;
    try {
        impl_ = Impl::create();
        impl_->init(device_id);
        return true;
    } catch (...) {
        std::fprintf(stderr, "[WARN] Exception caught: %s (DirectX init failed)\n", __func__);
        delete impl_;
        impl_ = nullptr;
        return false;
    }
}

bool DirectXCompute::is_initialized() const { return impl_ && impl_->ok; }

void DirectXCompute::shutdown() {
    if (impl_) { impl_->shutdown(); delete impl_; impl_ = nullptr; }
}

void* DirectXCompute::allocate(size_t bytes) {
    if (!impl_) return nullptr;
    try {
        auto res = impl_->create_buffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
        Impl::GpuBuf gb;
        gb.res = res;
        gb.size = bytes;
        void* ptr = gb.res.Get();
        { std::lock_guard<std::mutex> lk(impl_->mtx);
          impl_->bufs.push_back(std::move(gb));
        }
        return ptr;
    } catch (...) { std::fprintf(stderr, "[WARN] Exception caught: %s (GPU alloc failed)\n", __func__); return nullptr; }
}

void DirectXCompute::free(void* ptr) {
    if (!impl_ || !ptr) return;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    for (size_t i = 0; i < impl_->bufs.size(); i++) {
        if (impl_->bufs[i].res.Get() == ptr) {
            impl_->bufs.erase(impl_->bufs.begin() + (int)i);
            return;
        }
    }
}

void DirectXCompute::upload(const Tensor& src, void* dst) {
    if (!impl_ || !dst) return;
    size_t sz = src.size_bytes();
    const void* data = src.data();
    if (!data || !sz) return;

    ID3D12Resource* gpuRes = nullptr;
    { std::lock_guard<std::mutex> lk(impl_->mtx);
      for (auto& b : impl_->bufs)
          if (b.res.Get() == dst) { gpuRes = b.res.Get(); break; }
    }
    if (!gpuRes) return;

    auto upload = impl_->create_buffer(sz, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped;
    upload->Map(0, nullptr, &mapped);
    memcpy(mapped, data, sz);
    upload->Unmap(0, nullptr);

    impl_->list->Reset(impl_->allocator.Get(), nullptr);
    impl_->list->CopyBufferRegion(gpuRes, 0, upload.Get(), 0, sz);
    impl_->flush();
}

void DirectXCompute::download(void* src, Tensor& dst) {
    if (!impl_ || !src) return;
    size_t sz = dst.size_bytes();
    void* data = dst.data();
    if (!data || !sz) return;

    ID3D12Resource* gpuRes = nullptr;
    { std::lock_guard<std::mutex> lk(impl_->mtx);
      for (auto& b : impl_->bufs)
          if (b.res.Get() == src) { gpuRes = b.res.Get(); break; }
    }
    if (!gpuRes) return;

    auto readback = impl_->create_buffer(sz, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    impl_->list->Reset(impl_->allocator.Get(), nullptr);
    impl_->list->CopyBufferRegion(readback.Get(), 0, gpuRes, 0, sz);
    impl_->flush();

    D3D12_RANGE rr = { 0, sz };
    void* mapped;
    readback->Map(0, &rr, &mapped);
    memcpy(data, mapped, sz);
    readback->Unmap(0, nullptr);
}

void DirectXCompute::copy(void* dst, const void* src, size_t bytes) {
    if (!impl_ || !dst || !src || !bytes) return;

    ID3D12Resource *srcRes = nullptr, *dstRes = nullptr;
    { std::lock_guard<std::mutex> lk(impl_->mtx);
      for (auto& b : impl_->bufs) {
          if (b.res.Get() == src) srcRes = b.res.Get();
          if (b.res.Get() == dst) dstRes = b.res.Get();
      }
    }
    if (!srcRes || !dstRes) return;

    impl_->list->Reset(impl_->allocator.Get(), nullptr);
    impl_->list->CopyBufferRegion(dstRes, 0, srcRes, 0, bytes);
    impl_->flush();
}

void DirectXCompute::gemm(float alpha, const void* A, const void* B, float beta, void* C,
                          int64_t M, int64_t N, int64_t K)
{
    if (!impl_) return;
    Impl::GpuBuf *bA = nullptr, *bB = nullptr, *bC = nullptr;
    { std::lock_guard<std::mutex> lk(impl_->mtx);
      for (auto& b : impl_->bufs) {
          if (b.res.Get() == A) bA = &b;
          if (b.res.Get() == B) bB = &b;
          if (b.res.Get() == C) bC = &b;
      }
    }
    if (!bA || !bB || !bC) return;

    auto pso = impl_->make_pso(g_gemm_cs, "main");
    ID3D12Resource* uavs[] = { bC->res.Get() };
    UINT64 uavSizes[] = { bC->size };
    ID3D12Resource* srvs[] = { bA->res.Get(), bB->res.Get() };
    UINT64 srvSizes[] = { bA->size, bB->size };
    UINT rc[8];
    rc[0] = (UINT)M; rc[1] = (UINT)N; rc[2] = (UINT)K; rc[3] = 0;
    memcpy(&rc[4], &alpha, 4); memcpy(&rc[5], &beta, 4);
    rc[6] = 0; rc[7] = 0;
    int gx = (int)((N + 15) / 16);
    int gy = (int)((M + 15) / 16);
    impl_->dispatch(pso, 1, uavs, uavSizes, 2, srvs, srvSizes, rc, gx, gy, 1);
    impl_->flush();
}

void DirectXCompute::gemv(float alpha, const void* A, const void* x, float beta, void* y,
                          int64_t M, int64_t N)
{
    if (!impl_) return;
    Impl::GpuBuf *bA = nullptr, *bx = nullptr, *by = nullptr;
    { std::lock_guard<std::mutex> lk(impl_->mtx);
      for (auto& b : impl_->bufs) {
          if (b.res.Get() == A) bA = &b;
          if (b.res.Get() == x) bx = &b;
          if (b.res.Get() == y) by = &b;
      }
    }
    if (!bA || !bx || !by) return;

    auto pso = impl_->make_pso(g_gemv_cs, "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bA->res.Get(), bx->res.Get() };
    UINT64 srvSizes[] = { bA->size, bx->size };
    UINT rc[8];
    rc[0] = (UINT)M; rc[1] = (UINT)N; rc[2] = 0; rc[3] = 0;
    memcpy(&rc[4], &alpha, 4); memcpy(&rc[5], &beta, 4);
    rc[6] = 0; rc[7] = 0;
    int gx = (int)((M + 255) / 256);
    impl_->dispatch(pso, 1, uavs, uavSizes, 2, srvs, srvSizes, rc, gx, 1, 1);
    impl_->flush();
}

void DirectXCompute::relu(const void* x, void* y, int64_t n) {
    if (!impl_) return;
    Impl::GpuBuf *bx = nullptr, *by = nullptr;
    { std::lock_guard<std::mutex> lk(impl_->mtx);
      for (auto& b : impl_->bufs) { if (b.res.Get() == x) bx = &b; if (b.res.Get() == y) by = &b; }
    }
    if (!bx || !by) return;
    auto pso = impl_->make_pso(g_relu_cs, "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bx->res.Get() };
    UINT64 srvSizes[] = { bx->size };
    UINT rc[8] = { (UINT)n, 0, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch(pso, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, (int)((n + 255) / 256), 1, 1);
    impl_->flush();
}

void DirectXCompute::gelu(const void* x, void* y, int64_t n) {
    if (!impl_) return;
    Impl::GpuBuf *bx = nullptr, *by = nullptr;
    { std::lock_guard<std::mutex> lk(impl_->mtx);
      for (auto& b : impl_->bufs) { if (b.res.Get() == x) bx = &b; if (b.res.Get() == y) by = &b; }
    }
    if (!bx || !by) return;
    auto pso = impl_->make_pso(g_gelu_cs, "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bx->res.Get() };
    UINT64 srvSizes[] = { bx->size };
    UINT rc[8] = { (UINT)n, 0, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch(pso, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, (int)((n + 255) / 256), 1, 1);
    impl_->flush();
}

void DirectXCompute::silu(const void* x, void* y, int64_t n) {
    if (!impl_) return;
    Impl::GpuBuf *bx = nullptr, *by = nullptr;
    { std::lock_guard<std::mutex> lk(impl_->mtx);
      for (auto& b : impl_->bufs) { if (b.res.Get() == x) bx = &b; if (b.res.Get() == y) by = &b; }
    }
    if (!bx || !by) return;
    auto pso = impl_->make_pso(g_silu_cs, "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bx->res.Get() };
    UINT64 srvSizes[] = { bx->size };
    UINT rc[8] = { (UINT)n, 0, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch(pso, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, (int)((n + 255) / 256), 1, 1);
    impl_->flush();
}

void DirectXCompute::add(const void* a, const void* b, void* c, int64_t n) {
    if (!impl_) return;
    Impl::GpuBuf *ba = nullptr, *bb = nullptr, *bc = nullptr;
    { std::lock_guard<std::mutex> lk(impl_->mtx);
      for (auto& buf : impl_->bufs) {
          if (buf.res.Get() == a) ba = &buf;
          if (buf.res.Get() == b) bb = &buf;
          if (buf.res.Get() == c) bc = &buf;
      }
    }
    if (!ba || !bb || !bc) return;
    auto pso = impl_->make_pso(g_add_cs, "main");
    ID3D12Resource* uavs[] = { bc->res.Get() };
    UINT64 uavSizes[] = { bc->size };
    ID3D12Resource* srvs[] = { ba->res.Get(), bb->res.Get() };
    UINT64 srvSizes[] = { ba->size, bb->size };
    UINT rc[8] = { (UINT)n, 0, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch(pso, 1, uavs, uavSizes, 2, srvs, srvSizes, rc, (int)((n + 255) / 256), 1, 1);
    impl_->flush();
}

void DirectXCompute::mul(const void* a, const void* b, void* c, int64_t n) {
    if (!impl_) return;
    Impl::GpuBuf *ba = nullptr, *bb = nullptr, *bc = nullptr;
    { std::lock_guard<std::mutex> lk(impl_->mtx);
      for (auto& buf : impl_->bufs) {
          if (buf.res.Get() == a) ba = &buf;
          if (buf.res.Get() == b) bb = &buf;
          if (buf.res.Get() == c) bc = &buf;
      }
    }
    if (!ba || !bb || !bc) return;
    auto pso = impl_->make_pso(g_mul_cs, "main");
    ID3D12Resource* uavs[] = { bc->res.Get() };
    UINT64 uavSizes[] = { bc->size };
    ID3D12Resource* srvs[] = { ba->res.Get(), bb->res.Get() };
    UINT64 srvSizes[] = { ba->size, bb->size };
    UINT rc[8] = { (UINT)n, 0, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch(pso, 1, uavs, uavSizes, 2, srvs, srvSizes, rc, (int)((n + 255) / 256), 1, 1);
    impl_->flush();
}

void DirectXCompute::scale(float s, const void* x, void* y, int64_t n) {
    if (!impl_) return;
    Impl::GpuBuf *bx = nullptr, *by = nullptr;
    { std::lock_guard<std::mutex> lk(impl_->mtx);
      for (auto& buf : impl_->bufs) { if (buf.res.Get() == x) bx = &buf; if (buf.res.Get() == y) by = &buf; }
    }
    if (!bx || !by) return;
    auto pso = impl_->make_pso(g_scale_cs, "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bx->res.Get() };
    UINT64 srvSizes[] = { bx->size };
    UINT rc[8] = { (UINT)n, 0, 0, 0, 0, 0, 0, 0 };
    memcpy(&rc[4], &s, 4);
    impl_->dispatch(pso, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, (int)((n + 255) / 256), 1, 1);
    impl_->flush();
}

void DirectXCompute::softmax(const void* x, void* y, int64_t rows, int64_t cols) {
    if (!impl_) return;
    Impl::GpuBuf *bx = nullptr, *by = nullptr;
    { std::lock_guard<std::mutex> lk(impl_->mtx);
      for (auto& buf : impl_->bufs) { if (buf.res.Get() == x) bx = &buf; if (buf.res.Get() == y) by = &buf; }
    }
    if (!bx || !by) return;
    auto pso = impl_->make_pso(g_softmax_cs, "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bx->res.Get() };
    UINT64 srvSizes[] = { bx->size };
    UINT rc[8] = { (UINT)rows, (UINT)cols, 0, 0, 0, 0, 0, 0 };
    impl_->dispatch(pso, 1, uavs, uavSizes, 1, srvs, srvSizes, rc, (int)rows, 1, 1);
    impl_->flush();
}

void DirectXCompute::rms_norm(const void* x, const void* gamma, void* y, float eps,
                              int64_t n, int64_t d)
{
    if (!impl_) return;
    Impl::GpuBuf *bx = nullptr, *bg = nullptr, *by = nullptr;
    { std::lock_guard<std::mutex> lk(impl_->mtx);
      for (auto& buf : impl_->bufs) {
          if (buf.res.Get() == x) bx = &buf;
          if (buf.res.Get() == gamma) bg = &buf;
          if (buf.res.Get() == y) by = &buf;
      }
    }
    if (!bx || !bg || !by) return;
    auto pso = impl_->make_pso(g_rms_norm_cs, "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bx->res.Get(), bg->res.Get() };
    UINT64 srvSizes[] = { bx->size, bg->size };
    UINT rc[8] = { (UINT)n, (UINT)d, 0, 0, 0, 0, 0, 0 };
    memcpy(&rc[4], &eps, 4);
    impl_->dispatch(pso, 1, uavs, uavSizes, 2, srvs, srvSizes, rc, (int)n, 1, 1);
    impl_->flush();
}

void DirectXCompute::layer_norm(const void* x, const void* gamma, const void* beta, void* y,
                                float eps, int64_t n, int64_t d)
{
    if (!impl_) return;
    Impl::GpuBuf *bx = nullptr, *bg = nullptr, *bb = nullptr, *by = nullptr;
    { std::lock_guard<std::mutex> lk(impl_->mtx);
      for (auto& buf : impl_->bufs) {
          if (buf.res.Get() == x) bx = &buf;
          if (buf.res.Get() == gamma) bg = &buf;
          if (buf.res.Get() == beta) bb = &buf;
          if (buf.res.Get() == y) by = &buf;
      }
    }
    if (!bx || !bg || !bb || !by) return;
    auto pso = impl_->make_pso(g_layer_norm_cs, "main");
    ID3D12Resource* uavs[] = { by->res.Get() };
    UINT64 uavSizes[] = { by->size };
    ID3D12Resource* srvs[] = { bx->res.Get(), bg->res.Get(), bb->res.Get() };
    UINT64 srvSizes[] = { bx->size, bg->size, bb->size };
    UINT rc[8] = { (UINT)n, (UINT)d, 0, 0, 0, 0, 0, 0 };
    memcpy(&rc[4], &eps, 4);
    impl_->dispatch(pso, 1, uavs, uavSizes, 3, srvs, srvSizes, rc, (int)n, 1, 1);
    impl_->flush();
}

void DirectXCompute::moe_gather(const void* x, const int64_t* indices, const float* weights,
                                void* out, int64_t T, int64_t K, int64_t D)
{
    if (!impl_) return;
    Impl::GpuBuf *bx = nullptr, *bout = nullptr;
    { std::lock_guard<std::mutex> lk(impl_->mtx);
      for (auto& buf : impl_->bufs) {
          if (buf.res.Get() == x) bx = &buf;
          if (buf.res.Get() == out) bout = &buf;
      }
    }
    if (!bx || !bout) return;

    size_t idxSz = (size_t)(T * K) * sizeof(int64_t);
    size_t wgtSz = (size_t)(T * K) * sizeof(float);
    auto idxBuf = impl_->create_buffer(idxSz, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto wgtBuf = impl_->create_buffer(wgtSz, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    { void* mp; idxBuf->Map(0, nullptr, &mp); memcpy(mp, indices, idxSz); idxBuf->Unmap(0, nullptr); }
    { void* mp; wgtBuf->Map(0, nullptr, &mp); memcpy(mp, weights, wgtSz); wgtBuf->Unmap(0, nullptr); }

    auto idxGPU = impl_->create_buffer(idxSz, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    auto wgtGPU = impl_->create_buffer(wgtSz, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    impl_->list->Reset(impl_->allocator.Get(), nullptr);
    impl_->list->CopyBufferRegion(idxGPU.Get(), 0, idxBuf.Get(), 0, idxSz);
    impl_->list->CopyBufferRegion(wgtGPU.Get(), 0, wgtBuf.Get(), 0, wgtSz);
    impl_->flush();

    auto pso = impl_->make_pso(g_moe_gather_cs, "main");
    ID3D12Resource* uavs[] = { bout->res.Get() };
    UINT64 uavSizes[] = { bout->size };
    ID3D12Resource* srvs[] = { bx->res.Get(), idxGPU.Get(), wgtGPU.Get() };
    UINT64 srvSizes[] = { bx->size, idxSz, wgtSz };
    UINT rc[8] = { (UINT)T, (UINT)K, (UINT)D, 0, 0, 0, 0, 0 };
    int64_t total = T * K * D;
    impl_->dispatch(pso, 1, uavs, uavSizes, 3, srvs, srvSizes, rc, (int)((total + 255) / 256), 1, 1);
    impl_->flush();
}

void DirectXCompute::moe_scatter_add(void* out, const int64_t* indices, const float* weights,
                                     const void* expert_out, int64_t T, int64_t K, int64_t D)
{
    if (!impl_) return;
    Impl::GpuBuf *bout = nullptr, *beo = nullptr;
    { std::lock_guard<std::mutex> lk(impl_->mtx);
      for (auto& buf : impl_->bufs) {
          if (buf.res.Get() == out) bout = &buf;
          if (buf.res.Get() == expert_out) beo = &buf;
      }
    }
    if (!bout || !beo) return;

    size_t idxSz = (size_t)(T * K) * sizeof(int64_t);
    size_t wgtSz = (size_t)(T * K) * sizeof(float);
    auto idxBuf = impl_->create_buffer(idxSz, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto wgtBuf = impl_->create_buffer(wgtSz, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    { void* mp; idxBuf->Map(0, nullptr, &mp); memcpy(mp, indices, idxSz); idxBuf->Unmap(0, nullptr); }
    { void* mp; wgtBuf->Map(0, nullptr, &mp); memcpy(mp, weights, wgtSz); wgtBuf->Unmap(0, nullptr); }

    auto idxGPU = impl_->create_buffer(idxSz, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    auto wgtGPU = impl_->create_buffer(wgtSz, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    impl_->list->Reset(impl_->allocator.Get(), nullptr);
    impl_->list->CopyBufferRegion(idxGPU.Get(), 0, idxBuf.Get(), 0, idxSz);
    impl_->list->CopyBufferRegion(wgtGPU.Get(), 0, wgtBuf.Get(), 0, wgtSz);
    impl_->flush();

    auto pso = impl_->make_pso(g_moe_scatter_cs, "main");
    ID3D12Resource* uavs[] = { bout->res.Get() };
    UINT64 uavSizes[] = { bout->size };
    ID3D12Resource* srvs[] = { idxGPU.Get(), wgtGPU.Get(), beo->res.Get() };
    UINT64 srvSizes[] = { idxSz, wgtSz, beo->size };
    UINT rc[8] = { (UINT)T, (UINT)K, (UINT)D, 0, 0, 0, 0, 0 };
    int64_t total = T * K * D;
    impl_->dispatch(pso, 1, uavs, uavSizes, 3, srvs, srvSizes, rc, (int)((total + 255) / 256), 1, 1);
    impl_->flush();
}

void DirectXCompute::synchronize() {
    if (impl_) impl_->flush();
}

int64_t DirectXCompute::memory_free() const {
    if (!impl_) return 0;
    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
    ComPtr<IDXGIFactory4> factory;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        ComPtr<IDXGIAdapter1> adapter1;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter1) != DXGI_ERROR_NOT_FOUND; i++) {
            DXGI_ADAPTER_DESC1 desc;
            if (SUCCEEDED(adapter1->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                ComPtr<IDXGIAdapter3> adapter3;
                if (SUCCEEDED(adapter1.As(&adapter3))) {
                    adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info);
                    return (int64_t)info.Budget;
                }
            }
        }
    }
    return 8LL * 1024 * 1024 * 1024;
}

int64_t DirectXCompute::memory_total() const {
    if (!impl_) return 0;
    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
    ComPtr<IDXGIFactory4> factory;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        ComPtr<IDXGIAdapter1> adapter1;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter1) != DXGI_ERROR_NOT_FOUND; i++) {
            DXGI_ADAPTER_DESC1 desc;
            if (SUCCEEDED(adapter1->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                ComPtr<IDXGIAdapter3> adapter3;
                if (SUCCEEDED(adapter1.As(&adapter3))) {
                    adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info);
                    return (int64_t)info.Budget;
                }
            }
        }
    }
    return 16LL * 1024 * 1024 * 1024;
}

} // namespace gpu
} // namespace quant
