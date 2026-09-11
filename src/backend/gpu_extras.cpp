#include "quant/gpu_extras.h"
#include "quant/math.h"
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_OSX || TARGET_OS_IPHONE
#include <dlfcn.h>
#define QUANT_HAVE_METAL 1
#endif
#endif

namespace quant {
namespace gpu {

// ========================================================================
// Vulkan minimal type forward-declarations (no SDK headers)
// ========================================================================
#ifdef _WIN32
using VkInstance = uint64_t;
using VkPhysicalDevice = uint64_t;
using VkDevice = uint64_t;
using VkQueue = uint64_t;
using VkCommandPool = uint64_t;
using VkCommandBuffer = uint64_t;
using VkBuffer = uint64_t;
using VkDeviceMemory = uint64_t;
using VkFence = uint64_t;
using VkShaderModule = uint64_t;

using PFN_vkCreateInstance = uint64_t(__stdcall*)(const void*, const void*, VkInstance*);
using PFN_vkDestroyInstance = void(__stdcall*)(VkInstance, const void*);
using PFN_vkEnumeratePhysicalDevices = uint64_t(__stdcall*)(VkInstance, uint32_t*, VkPhysicalDevice*);
using PFN_vkGetPhysicalDeviceProperties = void(__stdcall*)(VkPhysicalDevice, void*);
using PFN_vkGetPhysicalDeviceMemoryProperties = void(__stdcall*)(VkPhysicalDevice, void*);
using PFN_vkGetPhysicalDeviceQueueFamilyProperties = void(__stdcall*)(VkPhysicalDevice, uint32_t*, void*);
using PFN_vkCreateDevice = uint64_t(__stdcall*)(VkPhysicalDevice, const void*, const void*, VkDevice*);
using PFN_vkDestroyDevice = void(__stdcall*)(VkDevice, const void*);
using PFN_vkGetDeviceQueue = void(__stdcall*)(VkDevice, uint32_t, uint32_t, VkQueue*);
using PFN_vkCreateCommandPool = uint64_t(__stdcall*)(VkDevice, const void*, const void*, VkCommandPool*);
using PFN_vkDestroyCommandPool = void(__stdcall*)(VkDevice, VkCommandPool, const void*);
using PFN_vkAllocateCommandBuffers = uint64_t(__stdcall*)(VkDevice, const void*, VkCommandBuffer*);
using PFN_vkBeginCommandBuffer = uint64_t(__stdcall*)(VkCommandBuffer, const void*);
using PFN_vkEndCommandBuffer = uint64_t(__stdcall*)(VkCommandBuffer);
using PFN_vkQueueSubmit = uint64_t(__stdcall*)(VkQueue, uint32_t, const void*, VkFence);
using PFN_vkQueueWaitIdle = uint64_t(__stdcall*)(VkQueue);
using PFN_vkDeviceWaitIdle = uint64_t(__stdcall*)(VkDevice);
using PFN_vkAllocateMemory = uint64_t(__stdcall*)(VkDevice, const void*, const void*, VkDeviceMemory*);
using PFN_vkFreeMemory = void(__stdcall*)(VkDevice, VkDeviceMemory, const void*);
using PFN_vkMapMemory = uint64_t(__stdcall*)(VkDevice, VkDeviceMemory, uint64_t, uint64_t, uint32_t, void**);
using PFN_vkUnmapMemory = void(__stdcall*)(VkDevice, VkDeviceMemory);
using PFN_vkCreateFence = uint64_t(__stdcall*)(VkDevice, const void*, const void*, VkFence*);
using PFN_vkDestroyFence = void(__stdcall*)(VkDevice, VkFence, const void*);
using PFN_vkWaitForFences = uint64_t(__stdcall*)(VkDevice, uint32_t, const VkFence*, uint32_t, uint64_t);
using PFN_vkResetFences = uint64_t(__stdcall*)(VkDevice, uint32_t, const VkFence*);

#define VK_API_VERSION_1_0 ((1u << 22) | (0u << 12) | 0u)
#endif

struct VulkanCompute::Impl {
    bool initialized = false;

#ifdef _WIN32
    HMODULE vulkan_lib = nullptr;

    PFN_vkCreateInstance vkCreateInstance = nullptr;
    PFN_vkDestroyInstance vkDestroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkCreateDevice vkCreateDevice = nullptr;
    PFN_vkDestroyDevice vkDestroyDevice = nullptr;
    PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;
    PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
    PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
    PFN_vkDeviceWaitIdle vkDeviceWaitIdle = nullptr;
    PFN_vkAllocateMemory vkAllocateMemory = nullptr;
    PFN_vkFreeMemory vkFreeMemory = nullptr;
    PFN_vkMapMemory vkMapMemory = nullptr;
    PFN_vkUnmapMemory vkUnmapMemory = nullptr;
    PFN_vkCreateFence vkCreateFence = nullptr;
    PFN_vkDestroyFence vkDestroyFence = nullptr;
    PFN_vkWaitForFences vkWaitForFences = nullptr;
    PFN_vkResetFences vkResetFences = nullptr;

    VkInstance instance = 0;
    VkPhysicalDevice phys_device = 0;
    VkDevice device = 0;
    VkQueue queue = 0;
    uint32_t queue_family = 0;
    VkCommandPool cmd_pool = 0;
    VkCommandBuffer cmd_buf = 0;
    VkFence fence = 0;
    int64_t total_mem = 0;

    template<typename T>
    T load(const char* name) {
        return reinterpret_cast<T>(GetProcAddress(vulkan_lib, name));
    }

    bool load_functions() {
        if (!vulkan_lib) return false;
        vkCreateInstance = load<PFN_vkCreateInstance>("vkCreateInstance");
        vkDestroyInstance = load<PFN_vkDestroyInstance>("vkDestroyInstance");
        vkEnumeratePhysicalDevices = load<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
        vkGetPhysicalDeviceProperties = load<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
        vkGetPhysicalDeviceMemoryProperties = load<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties");
        vkGetPhysicalDeviceQueueFamilyProperties = load<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties");
        vkCreateDevice = load<PFN_vkCreateDevice>("vkCreateDevice");
        vkDestroyDevice = load<PFN_vkDestroyDevice>("vkDestroyDevice");
        vkGetDeviceQueue = load<PFN_vkGetDeviceQueue>("vkGetDeviceQueue");
        vkCreateCommandPool = load<PFN_vkCreateCommandPool>("vkCreateCommandPool");
        vkDestroyCommandPool = load<PFN_vkDestroyCommandPool>("vkDestroyCommandPool");
        vkAllocateCommandBuffers = load<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers");
        vkDeviceWaitIdle = load<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle");
        vkAllocateMemory = load<PFN_vkAllocateMemory>("vkAllocateMemory");
        vkFreeMemory = load<PFN_vkFreeMemory>("vkFreeMemory");
        vkMapMemory = load<PFN_vkMapMemory>("vkMapMemory");
        vkUnmapMemory = load<PFN_vkUnmapMemory>("vkUnmapMemory");
        vkCreateFence = load<PFN_vkCreateFence>("vkCreateFence");
        vkDestroyFence = load<PFN_vkDestroyFence>("vkDestroyFence");
        vkWaitForFences = load<PFN_vkWaitForFences>("vkWaitForFences");
        vkResetFences = load<PFN_vkResetFences>("vkResetFences");
        return vkCreateInstance && vkCreateDevice && vkAllocateMemory && vkMapMemory;
    }
#endif
};

VulkanCompute::VulkanCompute() : impl_(new Impl) {}
VulkanCompute::~VulkanCompute() {
#ifdef _WIN32
    if (impl_->vulkan_lib) {
        if (impl_->device && impl_->vkDeviceWaitIdle) impl_->vkDeviceWaitIdle(impl_->device);
        if (impl_->fence && impl_->vkDestroyFence) impl_->vkDestroyFence(impl_->device, impl_->fence, nullptr);
        if (impl_->cmd_pool && impl_->vkDestroyCommandPool) impl_->vkDestroyCommandPool(impl_->device, impl_->cmd_pool, nullptr);
        if (impl_->device && impl_->vkDestroyDevice) impl_->vkDestroyDevice(impl_->device, nullptr);
        if (impl_->instance && impl_->vkDestroyInstance) impl_->vkDestroyInstance(impl_->instance, nullptr);
        FreeLibrary(impl_->vulkan_lib);
    }
#endif
    delete impl_;
}

bool VulkanCompute::init(int64_t device_id) {
#ifdef _WIN32
    if (impl_->initialized) return true;

    impl_->vulkan_lib = LoadLibraryA("vulkan-1.dll");
    if (!impl_->vulkan_lib) return false;
    if (!impl_->load_functions()) { FreeLibrary(impl_->vulkan_lib); impl_->vulkan_lib = nullptr; return false; }

    // VkApplicationInfo
    struct VkApplicationInfo {
        uint32_t sType; const void* pNext; const char* pApplicationName;
        uint32_t applicationVersion; const char* pEngineName; uint32_t engineVersion; uint32_t apiVersion;
    } appInfo = {12, nullptr, "Transcender", 1, "Transcender", 1, VK_API_VERSION_1_0};
    // VkInstanceCreateInfo
    struct VkInstanceCreateInfo {
        uint32_t sType; const void* pNext; uint32_t flags;
        const VkApplicationInfo* pApplicationInfo; uint32_t enabledLayerCount;
        const char* const* ppEnabledLayerNames; uint32_t enabledExtensionCount;
        const char* const* ppEnabledExtensionNames;
    } instInfo = {13, nullptr, 0, &appInfo, 0, nullptr, 0, nullptr};

    if (impl_->vkCreateInstance(&instInfo, nullptr, &impl_->instance) != 0) {
        FreeLibrary(impl_->vulkan_lib); impl_->vulkan_lib = nullptr; return false;
    }

    uint32_t gpu_count = 0;
    impl_->vkEnumeratePhysicalDevices(impl_->instance, &gpu_count, nullptr);
    if (gpu_count == 0) { impl_->vkDestroyInstance(impl_->instance, nullptr); FreeLibrary(impl_->vulkan_lib); impl_->vulkan_lib = nullptr; return false; }
    std::vector<VkPhysicalDevice> phys(gpu_count);
    impl_->vkEnumeratePhysicalDevices(impl_->instance, &gpu_count, phys.data());
    impl_->phys_device = phys[device_id < (int64_t)gpu_count ? (uint32_t)device_id : 0];

    // Query memory
    struct VkPhysicalDeviceMemoryProperties {
        uint32_t memoryTypeCount; uint32_t memoryTypes[32]; uint32_t memoryHeapCount; uint64_t memoryHeaps[16];
    } memProps = {};
    impl_->vkGetPhysicalDeviceMemoryProperties(impl_->phys_device, &memProps);
    for (uint32_t h = 0; h < memProps.memoryHeapCount && h < 16; h++)
        impl_->total_mem += (int64_t)memProps.memoryHeaps[h];

    // Find first compute-capable queue family
    struct VkQueueFamilyProperties { uint32_t queueFlags; uint32_t queueCount; uint32_t timestampValidBits; uint64_t minImageTransferGranularity; };
    uint32_t family_count = 0;
    if (impl_->vkGetPhysicalDeviceQueueFamilyProperties) {
        impl_->vkGetPhysicalDeviceQueueFamilyProperties(impl_->phys_device, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        impl_->vkGetPhysicalDeviceQueueFamilyProperties(impl_->phys_device, &family_count, families.data());
        for (uint32_t i = 0; i < family_count; i++) {
            if (families[i].queueFlags & 0x01) { impl_->queue_family = i; break; }
        }
    }

    float queue_priority = 1.0f;
    struct VkDeviceQueueCreateInfo {
        uint32_t sType; const void* pNext; uint32_t queueFamilyIndex;
        uint32_t queueCount; const float* pQueuePriorities;
    } queueInfo = {18, nullptr, impl_->queue_family, 1, &queue_priority};
    struct VkDeviceCreateInfo {
        uint32_t sType; const void* pNext; uint32_t flags;
        uint32_t queueCreateInfoCount; const VkDeviceQueueCreateInfo* pQueueCreateInfos;
        uint32_t enabledLayerCount; const char* const* ppEnabledLayerNames;
        uint32_t enabledExtensionCount; const char* const* ppEnabledExtensionNames;
        const void* pEnabledFeatures;
    } devInfo = {19, nullptr, 0, 1, &queueInfo, 0, nullptr, 0, nullptr, nullptr};

    if (impl_->vkCreateDevice(impl_->phys_device, &devInfo, nullptr, &impl_->device) != 0) {
        impl_->vkDestroyInstance(impl_->instance, nullptr); FreeLibrary(impl_->vulkan_lib); impl_->vulkan_lib = nullptr; return false;
    }

    impl_->vkGetDeviceQueue(impl_->device, impl_->queue_family, 0, &impl_->queue);

    // Command pool
    struct VkCommandPoolCreateInfo {
        uint32_t sType; const void* pNext; uint32_t flags; uint32_t queueFamilyIndex;
    } poolInfo = {20, nullptr, 0x01, impl_->queue_family};
    if (impl_->vkCreateCommandPool(impl_->device, &poolInfo, nullptr, &impl_->cmd_pool) != 0) {
        impl_->vkDestroyDevice(impl_->device, nullptr); impl_->vkDestroyInstance(impl_->instance, nullptr);
        FreeLibrary(impl_->vulkan_lib); impl_->vulkan_lib = nullptr; return false;
    }

    // Command buffer
    struct VkCommandBufferAllocateInfo {
        uint32_t sType; const void* pNext; VkCommandPool commandPool;
        uint32_t level; uint32_t commandBufferCount;
    } cmdAllocInfo = {22, nullptr, impl_->cmd_pool, 0, 1};
    impl_->vkAllocateCommandBuffers(impl_->device, &cmdAllocInfo, &impl_->cmd_buf);

    // Fence
    struct VkFenceCreateInfo { uint32_t sType; const void* pNext; uint32_t flags; } fenceInfo = {24, nullptr, 0};
    impl_->vkCreateFence(impl_->device, &fenceInfo, nullptr, &impl_->fence);

    impl_->initialized = true;
    return true;
#else
    (void)device_id;
    return false;
#endif
}

bool VulkanCompute::is_initialized() const { return impl_->initialized; }

void* VulkanCompute::allocate(size_t bytes) {
    if (!impl_->initialized) return nullptr;
    return std::malloc(bytes);
}

void VulkanCompute::free(void* ptr) { std::free(ptr); }

void VulkanCompute::upload(const Tensor& src, void* dst) {
    if (!impl_->initialized || !dst || src.numel() == 0) return;
    std::memcpy(dst, src.data<float>(), src.size_bytes());
}

void VulkanCompute::download(void* src, Tensor& dst) {
    if (!impl_->initialized || !src || dst.numel() == 0) return;
    std::memcpy(dst.data<float>(), src, dst.size_bytes());
}

void VulkanCompute::synchronize() {
#ifdef _WIN32
    if (impl_->initialized && impl_->vkDeviceWaitIdle)
        impl_->vkDeviceWaitIdle(impl_->device);
#endif
}

int64_t VulkanCompute::memory_free() const {
    return impl_->total_mem > 0 ? impl_->total_mem : 8LL * 1024 * 1024 * 1024;
}

// ========================================================================
// MetalCompute (E11) — macOS only
// ========================================================================
struct MetalCompute::Impl {
    bool initialized = false;
#ifdef QUANT_HAVE_METAL
    void* device = nullptr;
    void* command_queue = nullptr;
#endif
};

MetalCompute::MetalCompute() : impl_(new Impl) {}

MetalCompute::~MetalCompute() {
#ifdef QUANT_HAVE_METAL
    if (impl_->command_queue && impl_->device) {
        using objc_msgSend_t = void (*)(void*, const char*);
        auto msg = reinterpret_cast<objc_msgSend_t>(dlsym(RTLD_DEFAULT, "objc_msgSend"));
        (void)msg;
    }
#endif
    delete impl_;
}

bool MetalCompute::init(int64_t device_id) {
#ifdef QUANT_HAVE_METAL
    void* metal_lib = dlopen("/System/Library/Frameworks/Metal.framework/Metal", RTLD_LAZY);
    if (!metal_lib) return false;

    using MTLCreateSystemDefaultDevice_t = void* (*)();
    auto MTLCreateSystemDefaultDevice = (MTLCreateSystemDefaultDevice_t)dlsym(metal_lib, "MTLCreateSystemDefaultDevice");
    if (!MTLCreateSystemDefaultDevice) { dlclose(metal_lib); return false; }

    impl_->device = MTLCreateSystemDefaultDevice();
    if (!impl_->device) { dlclose(metal_lib); return false; }

    using objc_msgSend_t = void* (*)(void*, const char*);
    auto msg = reinterpret_cast<objc_msgSend_t>(dlsym(RTLD_DEFAULT, "objc_msgSend"));
    if (msg)
        impl_->command_queue = msg(impl_->device, "newCommandQueue");

    impl_->initialized = impl_->command_queue != nullptr;
    dlclose(metal_lib);
    return impl_->initialized;
#else
    (void)device_id;
    return false;
#endif
}

bool MetalCompute::is_initialized() const { return impl_->initialized; }
void MetalCompute::synchronize() {}

// ========================================================================
// IGPUSharedBackend (E3)
// ========================================================================
struct IGPUSharedBackend::Impl {
    bool initialized = false;
    void* shared_heap = nullptr;
    size_t heap_size = 0;
    size_t heap_used = 0; // P0: bump cursor (allocate() was aliasing)
};

IGPUSharedBackend::IGPUSharedBackend() : impl_(new Impl) {}
IGPUSharedBackend::~IGPUSharedBackend() {
    if (impl_->shared_heap) { std::free(impl_->shared_heap); impl_->shared_heap = nullptr; }
    delete impl_;
}

bool IGPUSharedBackend::init(int64_t) {
    impl_->heap_size = 512 * 1024 * 1024;
    impl_->heap_used = 0;
    impl_->shared_heap = std::malloc(impl_->heap_size);
    impl_->initialized = impl_->shared_heap != nullptr;
    return impl_->initialized;
}

void* IGPUSharedBackend::allocate(size_t bytes) {
    // P0 fix: old code returned the same shared_heap base for EVERY
    // allocation (aliasing corruption — two tensors, one address).
    // Bump-allocator with 64B alignment; nullptr when the heap is exhausted.
    constexpr size_t kAlign = 64;
    if (!impl_->shared_heap || bytes == 0) return nullptr;
    size_t aligned = (bytes + kAlign - 1) & ~(kAlign - 1);
    if (impl_->heap_used + aligned > (size_t)impl_->heap_size) return nullptr;
    char* p = (char*)impl_->shared_heap + impl_->heap_used;
    impl_->heap_used += aligned;
    return p;
}
void IGPUSharedBackend::free(void*) {}
void IGPUSharedBackend::upload(const Tensor& src, void* dst) {
    if (!dst || src.numel() == 0) return;
    std::memcpy(dst, src.data<float>(), src.size_bytes());
}
void IGPUSharedBackend::download(void* src, Tensor& dst) {
    if (!src || dst.numel() == 0) return;
    std::memcpy(dst.data<float>(), src, dst.size_bytes());
}
void IGPUSharedBackend::synchronize() {}
// P0 fix: old code reported the whole heap as free even after bump
// allocations — now reports the true remainder.
int64_t IGPUSharedBackend::memory_free() const {
    if (!impl_->shared_heap || impl_->heap_used >= (size_t)impl_->heap_size) return 0;
    return (int64_t)((size_t)impl_->heap_size - impl_->heap_used);
}

Tensor IGPUSharedBackend::shared_tensor(const Shape& shape) {
    size_t bytes = shape.numel() * sizeof(float);
    void* ptr = allocate(bytes);
    if (!ptr) return Tensor(shape);
    Tensor t(shape);
    std::memcpy(t.data<float>(), ptr, bytes);
    return t;
}

// ========================================================================
// MultiGPUManager (E12)
// ========================================================================
MultiGPUManager::MultiGPUManager() {}
MultiGPUManager::~MultiGPUManager() {
    for (auto& d : devices_)
        d.initialized = false;
}

int MultiGPUManager::detect_devices() {
    devices_.clear();
    devices_.push_back({true, "CPU", 0});

    // P0 fix: old code pushed 8 phantom "GPU_i" entries with a hardcoded
    // 8 GiB each whenever DXGI merely loaded (EnumAdapters never called) —
    // users saw 8 fake GPUs. Now: only entries we actually enumerate.
    // Full per-adapter VRAM query is future work; until then, report exactly
    // what we verified and nothing more.
#ifdef _WIN32
    HMODULE dxgi = LoadLibraryA("dxgi.dll");
    if (dxgi) {
        using PFN_CreateDXGIFactory1 = uint64_t(__stdcall*)(const uint64_t&, void**);
        auto factory_fn = (PFN_CreateDXGIFactory1)GetProcAddress(dxgi, "CreateDXGIFactory1");
        // NOTE: factory handle intentionally unused until the real
        // EnumAdapters path lands — we must not fabricate devices.
        (void)factory_fn;
        FreeLibrary(dxgi);
    }
#endif

    return (int)devices_.size();
}

bool MultiGPUManager::init_device(int idx) {
    if (idx < 0 || idx >= (int)devices_.size()) return false;
    devices_[idx].initialized = true;
    return true;
}

void* MultiGPUManager::allocate_on(int idx, size_t bytes) {
    if (idx == 0) return std::malloc(bytes);
    return nullptr;
}

void MultiGPUManager::free_on(int idx, void* ptr) {
    if (idx == 0) std::free(ptr);
}

void MultiGPUManager::transfer(int, int dst_idx, void* src, void* dst, size_t bytes) {
    if (dst_idx == 0) std::memcpy(dst, src, bytes);
}

std::string MultiGPUManager::device_name(int idx) const {
    if (idx < 0 || idx >= (int)devices_.size()) return "unknown";
    return devices_[idx].name;
}

// ========================================================================
// Auto-tuning (E14)
// ========================================================================
TunedParams auto_tune_gemm(int64_t M, int64_t N, int64_t K) {
    TunedParams p;
    if (M >= 1024 && N >= 1024) { p.block_m = 128; p.block_n = 128; p.block_k = 32; }
    else if (M >= 256) { p.block_m = 64; p.block_n = 64; p.block_k = 16; }
    return p;
}

// ========================================================================
// GPU Fallback (E15)
// ========================================================================
GPUFallback::GPUFallback() {}

} // namespace gpu
} // namespace quant
