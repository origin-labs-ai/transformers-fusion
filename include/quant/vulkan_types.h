#pragma once
#ifndef QUANT_VULKAN_TYPES_H
#define QUANT_VULKAN_TYPES_H

#include <cstdint>
#include <cstddef>

namespace quant {
namespace gpu {

using PFN_vkVoidFunction = void(*)();

struct VkInstance_T { char _opaque; };
struct VkPhysicalDevice_T { char _opaque; };
struct VkDevice_T { char _opaque; };
struct VkQueue_T { char _opaque; };
struct VkCommandPool_T { char _opaque; };
struct VkCommandBuffer_T { char _opaque; };
struct VkBuffer_T { char _opaque; };
struct VkDeviceMemory_T { char _opaque; };
struct VkPipeline_T { char _opaque; };
struct VkPipelineLayout_T { char _opaque; };
struct VkShaderModule_T { char _opaque; };
struct VkDescriptorSetLayout_T { char _opaque; };
struct VkDescriptorPool_T { char _opaque; };
struct VkDescriptorSet_T { char _opaque; };
struct VkFence_T { char _opaque; };
struct VkRenderPass_T { char _opaque; };
struct VkFramebuffer_T { char _opaque; };
struct VkSemaphore_T { char _opaque; };

using VkInstance = VkInstance_T*;
using VkPhysicalDevice = VkPhysicalDevice_T*;
using VkDevice = VkDevice_T*;
using VkQueue = VkQueue_T*;
using VkCommandPool = VkCommandPool_T*;
using VkCommandBuffer = VkCommandBuffer_T*;
using VkBuffer = VkBuffer_T*;
using VkDeviceMemory = VkDeviceMemory_T*;
using VkPipeline = VkPipeline_T*;
using VkPipelineLayout = VkPipelineLayout_T*;
using VkShaderModule = VkShaderModule_T*;
using VkDescriptorSetLayout = VkDescriptorSetLayout_T*;
using VkDescriptorPool = VkDescriptorPool_T*;
using VkDescriptorSet = VkDescriptorSet_T*;
using VkFence = VkFence_T*;
using VkRenderPass = VkRenderPass_T*;
using VkFramebuffer = VkFramebuffer_T*;
using VkSemaphore = VkSemaphore_T*;

using VkFlags = uint32_t;
using VkDeviceSize = uint64_t;
using VkResult = int;
using VkPhysicalDeviceType = int;
using VkSharingMode = int;
using VkBufferUsageFlagBits = int;
using VkMemoryPropertyFlagBits = int;
using VkPipelineBindPoint = int;
using VkStructureType = int;
using VkBool32 = uint32_t;

#ifndef VK_MAKE_VERSION
#define VK_MAKE_VERSION(major, minor, patch) \
    (((uint32_t)(major) << 22) | ((uint32_t)(minor) << 12) | (uint32_t)(patch))
#endif

constexpr int VK_SUCCESS = 0;
constexpr int VK_FALSE = 0;
constexpr int VK_TRUE = 1;
// These enum values must match the real Vulkan ABI. Several were wrong
// (PIPELINE_BIND_POINT_COMPUTE was 0 == GRAPHICS, and eight sType values
// were off), which made the driver misinterpret structs and crash inside
// vkCmdBindPipeline. Corrected 2026-09-12 against vulkan_core.h.
constexpr int VK_SHARING_MODE_EXCLUSIVE = 0;
constexpr int VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x20;
constexpr int VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x1;
constexpr int VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x2;
constexpr int VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001;
constexpr int VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002;
constexpr int VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004;
constexpr int VK_PIPELINE_BIND_POINT_COMPUTE = 1;
constexpr int VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
constexpr int VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
constexpr int VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2;
constexpr int VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3;
constexpr int VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12;
constexpr int VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5;
constexpr int VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
constexpr int VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30;
constexpr int VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16;
constexpr int VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29;
constexpr int VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
constexpr int VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
constexpr int VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18;
constexpr int VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES = 16;
constexpr int VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39;
constexpr int VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40;
constexpr int VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42;
constexpr int VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8;
constexpr int VK_STRUCTURE_TYPE_SUBMIT_INFO = 4;
constexpr int VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35;
constexpr int VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 1;
constexpr int VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 2;
constexpr uint64_t VK_QUEUE_FAMILY_IGNORED = 0xFFFFFFFF;
constexpr uint64_t VK_WHOLE_SIZE = 0xFFFFFFFFFFFFFFFFULL;
constexpr VkDeviceSize VK_MIN_MEMORY_MAP_ALIGNMENT = 64;

struct VkApplicationInfo {
    VkStructureType sType; const void* pNext; const char* pApplicationName;
    uint32_t applicationVersion; const char* pEngineName; uint32_t engineVersion; uint32_t apiVersion;
};

struct VkInstanceCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags;
    const VkApplicationInfo* pApplicationInfo;
    uint32_t enabledLayerCount; const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount; const char* const* ppEnabledExtensionNames;
};

struct VkDeviceQueueCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags;
    uint32_t queueFamilyIndex; uint32_t queueCount; const float* pQueuePriorities;
};

struct VkPhysicalDeviceFeatures { uint32_t robustBufferAccess; char _rest[496]; };

struct VkDeviceCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags;
    uint32_t queueCreateInfoCount; const VkDeviceQueueCreateInfo* pQueueCreateInfos;
    uint32_t enabledLayerCount; const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount; const char* const* ppEnabledExtensionNames;
    const VkPhysicalDeviceFeatures* pEnabledFeatures;
};

struct VkPhysicalDeviceMemoryProperties {
    uint32_t memoryTypeCount;
    struct { uint32_t heapIndex; VkFlags propertyFlags; } memoryTypes[32];
    uint32_t memoryHeapCount;
    struct { VkDeviceSize size; VkFlags flags; } memoryHeaps[16];
};

struct VkPhysicalDeviceProperties {
    // REAL ONLY: matches Vulkan spec prefix — NO sType/pNext.
    // apiVersion@0, driverVersion@4, vendorID@8, deviceID@12, deviceType@16, deviceName@20.
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    VkPhysicalDeviceType deviceType;
    char deviceName[256];
    uint8_t _rest[512];
};

// REAL VkQueueFamilyProperties: 24B total, queueFlags@0.
struct VkQueueFamilyProperties {
    VkFlags queueFlags;                        // @0
    uint32_t queueCount;                       // @4
    uint32_t timestampValidBits;               // @8
    uint32_t minImageTransferGranularity[3];   // @12 (12B)
};

static_assert(offsetof(VkPhysicalDeviceProperties, apiVersion) == 0, "REAL VkPhysicalDeviceProperties: apiVersion@0");
static_assert(offsetof(VkPhysicalDeviceProperties, driverVersion) == 4, "REAL VkPhysicalDeviceProperties: driverVersion@4");
static_assert(offsetof(VkPhysicalDeviceProperties, vendorID) == 8, "REAL VkPhysicalDeviceProperties: vendorID@8");
static_assert(offsetof(VkPhysicalDeviceProperties, deviceID) == 12, "REAL VkPhysicalDeviceProperties: deviceID@12");
static_assert(offsetof(VkPhysicalDeviceProperties, deviceType) == 16, "REAL VkPhysicalDeviceProperties: deviceType@16");
static_assert(offsetof(VkPhysicalDeviceProperties, deviceName) == 20, "REAL VkPhysicalDeviceProperties: deviceName@20");
static_assert(sizeof(VkQueueFamilyProperties) == 24, "REAL VkQueueFamilyProperties must be 24B");
static_assert(offsetof(VkQueueFamilyProperties, queueFlags) == 0, "REAL VkQueueFamilyProperties: flags@0");

struct VkBufferCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags; VkDeviceSize size;
    VkBufferUsageFlagBits usage; VkSharingMode sharingMode;
    uint32_t queueFamilyIndexCount; const uint32_t* pQueueFamilyIndices;
};

struct VkMemoryAllocateInfo {
    VkStructureType sType; const void* pNext; VkDeviceSize allocationSize; uint32_t memoryTypeIndex;
};

struct VkDescriptorSetLayoutBinding {
    uint32_t binding; int descriptorType; uint32_t descriptorCount; VkFlags stageFlags;
    const void* pImmutableSamplers;
};

struct VkDescriptorSetLayoutCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags;
    uint32_t bindingCount; const VkDescriptorSetLayoutBinding* pBindings;
};

struct VkPushConstantRange { VkFlags stageFlags; uint32_t offset; uint32_t size; };

struct VkPipelineLayoutCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags;
    uint32_t setLayoutCount; const VkDescriptorSetLayout* pSetLayouts;
    uint32_t pushConstantRangeCount; const VkPushConstantRange* pPushConstantRanges;
};

struct VkShaderModuleCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags;
    size_t codeSize; const uint32_t* pCode;
};

struct VkSpecializationMapEntry { uint32_t constantID; uint32_t offset; size_t size; };
struct VkSpecializationInfo { uint32_t mapEntryCount; const VkSpecializationMapEntry* pMapEntries; size_t dataSize; const void* pData; };

struct VkPipelineShaderStageCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags;
    VkFlags stage; VkShaderModule module; const char* pName; const VkSpecializationInfo* pSpecializationInfo;
};

struct VkComputePipelineCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags;
    VkPipelineShaderStageCreateInfo stage; VkPipelineLayout layout;
    VkPipeline basePipelineHandle; int32_t basePipelineIndex;
};

struct VkDescriptorPoolSize { int type; uint32_t descriptorCount; };

struct VkDescriptorPoolCreateInfo {
    VkStructureType sType; const void* pNext; VkFlags flags;
    uint32_t maxSets; uint32_t poolSizeCount; const VkDescriptorPoolSize* pPoolSizes;
};

struct VkDescriptorSetAllocateInfo {
    VkStructureType sType; const void* pNext; VkDescriptorPool descriptorPool;
    uint32_t descriptorSetCount; const VkDescriptorSetLayout* pSetLayouts;
};

struct VkDescriptorBufferInfo { VkBuffer buffer; VkDeviceSize offset; VkDeviceSize range; };

// ABI NOTE: must match the real VkWriteDescriptorSet exactly (64 bytes on x64).
// This struct previously omitted pImageInfo, so the driver read pImageInfo from
// offset 40 (where pBufferInfo actually sits) and pBufferInfo from offset 48 —
// past the end of the object. That produced garbage pointers inside the driver
// and a segfault that moved around between runs. The tail field is never read
// for STORAGE_BUFFER writes but must exist so the offsets line up.
struct VkWriteDescriptorSet {
    VkStructureType sType; const void* pNext; VkDescriptorSet dstSet;
    uint32_t dstBinding; uint32_t dstArrayElement; uint32_t descriptorCount;
    int descriptorType;
    const void* pImageInfo;
    const VkDescriptorBufferInfo* pBufferInfo;
    const void* pTexelBufferView;
};

struct VkCommandPoolCreateInfo { VkStructureType sType; const void* pNext; VkFlags flags; uint32_t queueFamilyIndex; };
struct VkCommandBufferAllocateInfo { VkStructureType sType; const void* pNext; VkCommandPool commandPool; int level; uint32_t commandBufferCount; };
struct VkCommandBufferBeginInfo { VkStructureType sType; const void* pNext; VkFlags flags; };
struct VkBufferCopy { VkDeviceSize srcOffset; VkDeviceSize dstOffset; VkDeviceSize size; };
struct VkMemoryRequirements { VkDeviceSize size; VkDeviceSize alignment; uint32_t memoryTypeBits; };
struct VkFenceCreateInfo { VkStructureType sType; const void* pNext; VkFlags flags; };

struct VkSubmitInfo {
    VkStructureType sType;
    const void* pNext;
    uint32_t waitSemaphoreCount;
    const VkSemaphore* pWaitSemaphores;
    const VkFlags* pWaitDstStageMask;
    uint32_t commandBufferCount;
    const VkCommandBuffer* pCommandBuffers;
    uint32_t signalSemaphoreCount;
    const VkSemaphore* pSignalSemaphores;
};

} // namespace gpu
} // namespace quant

#endif // QUANT_VULKAN_TYPES_H
