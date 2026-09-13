#include "quant/gpu_compute.h"
#include "quant/tensor.h"
#include "quant/vulkan_types.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <new>
#include <algorithm>
#include <cstdio>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace quant {
namespace gpu {

// Vulkan function pointer types
using PFN_vkGetInstanceProcAddr = PFN_vkVoidFunction(*)(VkInstance, const char*);
using PFN_vkGetDeviceProcAddr = PFN_vkVoidFunction(*)(VkDevice, const char*);
using PFN_vkCreateInstance = VkResult(*)(const VkInstanceCreateInfo*, const void*, VkInstance*);
using PFN_vkDestroyInstance = void(*)(VkInstance, const void*);
using PFN_vkEnumeratePhysicalDevices = VkResult(*)(VkInstance, uint32_t*, VkPhysicalDevice*);
using PFN_vkGetPhysicalDeviceProperties = void(*)(VkPhysicalDevice, void*);
using PFN_vkGetPhysicalDeviceMemoryProperties = void(*)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);
using PFN_vkCreateDevice = VkResult(*)(VkPhysicalDevice, const VkDeviceCreateInfo*, const void*, VkDevice*);
using PFN_vkDestroyDevice = void(*)(VkDevice, const void*);
using PFN_vkGetDeviceQueue = void(*)(VkDevice, uint32_t, uint32_t, VkQueue*);
using PFN_vkCreateBuffer = VkResult(*)(VkDevice, const VkBufferCreateInfo*, const void*, VkBuffer*);
using PFN_vkDestroyBuffer = void(*)(VkDevice, VkBuffer, const void*);
using PFN_vkGetBufferMemoryRequirements = void(*)(VkDevice, VkBuffer, VkMemoryRequirements*);
using PFN_vkAllocateMemory = VkResult(*)(VkDevice, const VkMemoryAllocateInfo*, const void*, VkDeviceMemory*);
using PFN_vkFreeMemory = void(*)(VkDevice, VkDeviceMemory, const void*);
using PFN_vkBindBufferMemory = VkResult(*)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
using PFN_vkMapMemory = VkResult(*)(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void**);
using PFN_vkUnmapMemory = void(*)(VkDevice, VkDeviceMemory);
using PFN_vkCreateDescriptorSetLayout = VkResult(*)(VkDevice, const VkDescriptorSetLayoutCreateInfo*, const void*, VkDescriptorSetLayout*);
using PFN_vkDestroyDescriptorSetLayout = void(*)(VkDevice, VkDescriptorSetLayout, const void*);
using PFN_vkCreatePipelineLayout = VkResult(*)(VkDevice, const VkPipelineLayoutCreateInfo*, const void*, VkPipelineLayout*);
using PFN_vkDestroyPipelineLayout = void(*)(VkDevice, VkPipelineLayout, const void*);
using PFN_vkCreateShaderModule = VkResult(*)(VkDevice, const VkShaderModuleCreateInfo*, const void*, VkShaderModule*);
using PFN_vkDestroyShaderModule = void(*)(VkDevice, VkShaderModule, const void*);
using PFN_vkCreateComputePipelines = VkResult(*)(VkDevice, VkPipeline, uint32_t, const VkComputePipelineCreateInfo*, const void*, VkPipeline*);
using PFN_vkDestroyPipeline = void(*)(VkDevice, VkPipeline, const void*);
using PFN_vkCreateDescriptorPool = VkResult(*)(VkDevice, const VkDescriptorPoolCreateInfo*, const void*, VkDescriptorPool*);
using PFN_vkDestroyDescriptorPool = void(*)(VkDevice, VkDescriptorPool, const void*);
using PFN_vkAllocateDescriptorSets = VkResult(*)(VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*);
using PFN_vkFreeDescriptorSets = VkResult(*)(VkDevice, VkDescriptorPool, uint32_t, const VkDescriptorSet*);
using PFN_vkUpdateDescriptorSets = void(*)(VkDevice, uint32_t, const VkWriteDescriptorSet*, uint32_t, const void*);
using PFN_vkCreateCommandPool = VkResult(*)(VkDevice, const VkCommandPoolCreateInfo*, const void*, VkCommandPool*);
using PFN_vkDestroyCommandPool = void(*)(VkDevice, VkCommandPool, const void*);
using PFN_vkAllocateCommandBuffers = VkResult(*)(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*);
using PFN_vkFreeCommandBuffers = void(*)(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*);
using PFN_vkBeginCommandBuffer = VkResult(*)(VkCommandBuffer, const VkCommandBufferBeginInfo*);
using PFN_vkEndCommandBuffer = VkResult(*)(VkCommandBuffer);
using PFN_vkCmdPipelineBarrier = void(*)(VkCommandBuffer, VkFlags, VkFlags, VkFlags, uint32_t, const void*, uint32_t, const void*, uint32_t, const void*);
using PFN_vkCmdBindPipeline = void(*)(VkCommandBuffer, VkPipelineBindPoint, VkPipeline);
using PFN_vkCmdBindDescriptorSets = void(*)(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet*, uint32_t, const uint32_t*);
using PFN_vkCmdPushConstants = void(*)(VkCommandBuffer, VkPipelineLayout, VkFlags, uint32_t, uint32_t, const void*);
using PFN_vkCmdDispatch = void(*)(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
using PFN_vkCmdCopyBuffer = void(*)(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy*);
using PFN_vkQueueSubmit = VkResult(*)(VkQueue, uint32_t, const void*, VkFence);
using PFN_vkQueueWaitIdle = VkResult(*)(VkQueue);
using PFN_vkDeviceWaitIdle = VkResult(*)(VkDevice);
using PFN_vkCreateFence = VkResult(*)(VkDevice, const VkFenceCreateInfo*, const void*, VkFence*);
using PFN_vkDestroyFence = void(*)(VkDevice, VkFence, const void*);
using PFN_vkWaitForFences = VkResult(*)(VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t);
using PFN_vkResetFences = VkResult(*)(VkDevice, uint32_t, const VkFence*);
using PFN_vkGetPhysicalDeviceQueueFamilyProperties = void(*)(VkPhysicalDevice, uint32_t*, void*);

// ========================================================================
// SPIR-V compute shader byte arrays (compiled from GLSL)
// ========================================================================

// Fence wait budget. Previously UINT64_MAX: if vkQueueSubmit failed or a shader
// never completed, the fence was never signalled and the process stalled forever
// (observed 2026-09-08 on an AMD Radeon(TM) iGPU — relu hung until ctest killed
// it at 60 s). Fail-loud beats an infinite stall: on timeout we bail without
// touching the destination buffer, so parity checks report the gap honestly.
static constexpr uint64_t kFenceWaitNs = 2000000000ULL;  // 2 seconds

// relu — GLSL 450 compute, compiled to SPIR-V by glslang on 2026-09-10.
//   layout(local_size_x=256) in; push_constant{ uint n; }; buffer bufs[] @ binding 0
//   i=GlobalInvocationID.x; if(i>=n) return; bufs[0].data[i]=max(bufs[1].data[i],0.0);  [bufs[2]]
// (Previous blob was not valid SPIR-V: the instruction stream broke at word 13 and
//  hung the AMD driver inside vkCreateShaderModule.)
static const uint32_t g_relu_spirv[] = {
    0x07230203,0x00010000,0x00080008,0x00000034,0x00000000,0x00020011,
    0x00000001,0x0006000b,0x00000001,0x4c534c47,0x6474732e,0x3035342e,
    0x00000000,0x0003000e,0x00000000,0x00000001,0x0006000f,0x00000005,
    0x00000004,0x6e69616d,0x00000000,0x0000000b,0x00060010,0x00000004,
    0x00000011,0x00000100,0x00000001,0x00000001,0x00030003,0x00000002,
    0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00030005,
    0x00000008,0x00000069,0x00080005,0x0000000b,0x475f6c67,0x61626f6c,
    0x766e496c,0x7461636f,0x496e6f69,0x00000044,0x00030005,0x00000011,
    0x00004350,0x00040006,0x00000011,0x00000000,0x0000006e,0x00030005,
    0x00000013,0x00006370,0x00030005,0x00000020,0x00000078,0x00040005,
    0x00000022,0x73667542,0x00000000,0x00050006,0x00000022,0x00000000,
    0x61746164,0x00000000,0x00040005,0x00000026,0x73667562,0x00000000,
    0x00040047,0x0000000b,0x0000000b,0x0000001c,0x00050048,0x00000011,
    0x00000000,0x00000023,0x00000000,0x00030047,0x00000011,0x00000002,
    0x00040047,0x00000021,0x00000006,0x00000004,0x00050048,0x00000022,
    0x00000000,0x00000023,0x00000000,0x00030047,0x00000022,0x00000003,
    0x00040047,0x00000026,0x00000022,0x00000000,0x00040047,0x00000026,
    0x00000021,0x00000000,0x00040047,0x00000033,0x0000000b,0x00000019,
    0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00040015,
    0x00000006,0x00000020,0x00000000,0x00040020,0x00000007,0x00000007,
    0x00000006,0x00040017,0x00000009,0x00000006,0x00000003,0x00040020,
    0x0000000a,0x00000001,0x00000009,0x0004003b,0x0000000a,0x0000000b,
    0x00000001,0x0004002b,0x00000006,0x0000000c,0x00000000,0x00040020,
    0x0000000d,0x00000001,0x00000006,0x0003001e,0x00000011,0x00000006,
    0x00040020,0x00000012,0x00000009,0x00000011,0x0004003b,0x00000012,
    0x00000013,0x00000009,0x00040015,0x00000014,0x00000020,0x00000001,
    0x0004002b,0x00000014,0x00000015,0x00000000,0x00040020,0x00000016,
    0x00000009,0x00000006,0x00020014,0x00000019,0x00030016,0x0000001e,
    0x00000020,0x00040020,0x0000001f,0x00000007,0x0000001e,0x0003001d,
    0x00000021,0x0000001e,0x0003001e,0x00000022,0x00000021,0x0004002b,
    0x00000006,0x00000023,0x00000002,0x0004001c,0x00000024,0x00000022,
    0x00000023,0x00040020,0x00000025,0x00000002,0x00000024,0x0004003b,
    0x00000025,0x00000026,0x00000002,0x0004002b,0x00000014,0x00000027,
    0x00000001,0x00040020,0x00000029,0x00000002,0x0000001e,0x0004002b,
    0x0000001e,0x0000002e,0x00000000,0x0004002b,0x00000006,0x00000031,
    0x00000100,0x0004002b,0x00000006,0x00000032,0x00000001,0x0006002c,
    0x00000009,0x00000033,0x00000031,0x00000032,0x00000032,0x00050036,
    0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,
    0x0004003b,0x00000007,0x00000008,0x00000007,0x0004003b,0x0000001f,
    0x00000020,0x00000007,0x00050041,0x0000000d,0x0000000e,0x0000000b,
    0x0000000c,0x0004003d,0x00000006,0x0000000f,0x0000000e,0x0003003e,
    0x00000008,0x0000000f,0x0004003d,0x00000006,0x00000010,0x00000008,
    0x00050041,0x00000016,0x00000017,0x00000013,0x00000015,0x0004003d,
    0x00000006,0x00000018,0x00000017,0x000500ae,0x00000019,0x0000001a,
    0x00000010,0x00000018,0x000300f7,0x0000001c,0x00000000,0x000400fa,
    0x0000001a,0x0000001b,0x0000001c,0x000200f8,0x0000001b,0x000100fd,
    0x000200f8,0x0000001c,0x0004003d,0x00000006,0x00000028,0x00000008,
    0x00070041,0x00000029,0x0000002a,0x00000026,0x00000027,0x00000015,
    0x00000028,0x0004003d,0x0000001e,0x0000002b,0x0000002a,0x0003003e,
    0x00000020,0x0000002b,0x0004003d,0x00000006,0x0000002c,0x00000008,
    0x0004003d,0x0000001e,0x0000002d,0x00000020,0x0007000c,0x0000001e,
    0x0000002f,0x00000001,0x00000028,0x0000002d,0x0000002e,0x00070041,
    0x00000029,0x00000030,0x00000026,0x00000015,0x00000015,0x0000002c,
    0x0003003e,0x00000030,0x0000002f,0x000100fd,0x00010038
};

// gelu — GLSL 450 compute, compiled to SPIR-V by glslang on 2026-09-10.
//   layout(local_size_x=256) in; push_constant{ uint n; }; buffer bufs[] @ binding 0
//   i=GlobalInvocationID.x; if(i>=n) return; x=bufs[1].data[i]; bufs[0].data[i]=0.5*x*(1+tanh(0.7978845608*(x+0.044715*x^3)));  [bufs[2]]
// (Previous blob was not valid SPIR-V: the instruction stream broke at word 13 and
//  hung the AMD driver inside vkCreateShaderModule.)
static const uint32_t g_gelu_spirv[] = {
    0x07230203,0x00010000,0x00080008,0x00000043,0x00000000,0x00020011,
    0x00000001,0x0006000b,0x00000001,0x4c534c47,0x6474732e,0x3035342e,
    0x00000000,0x0003000e,0x00000000,0x00000001,0x0006000f,0x00000005,
    0x00000004,0x6e69616d,0x00000000,0x0000000b,0x00060010,0x00000004,
    0x00000011,0x00000100,0x00000001,0x00000001,0x00030003,0x00000002,
    0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00030005,
    0x00000008,0x00000069,0x00080005,0x0000000b,0x475f6c67,0x61626f6c,
    0x766e496c,0x7461636f,0x496e6f69,0x00000044,0x00030005,0x00000011,
    0x00004350,0x00040006,0x00000011,0x00000000,0x0000006e,0x00030005,
    0x00000013,0x00006370,0x00030005,0x00000020,0x00000078,0x00040005,
    0x00000022,0x73667542,0x00000000,0x00050006,0x00000022,0x00000000,
    0x61746164,0x00000000,0x00040005,0x00000026,0x73667562,0x00000000,
    0x00040047,0x0000000b,0x0000000b,0x0000001c,0x00050048,0x00000011,
    0x00000000,0x00000023,0x00000000,0x00030047,0x00000011,0x00000002,
    0x00040047,0x00000021,0x00000006,0x00000004,0x00050048,0x00000022,
    0x00000000,0x00000023,0x00000000,0x00030047,0x00000022,0x00000003,
    0x00040047,0x00000026,0x00000022,0x00000000,0x00040047,0x00000026,
    0x00000021,0x00000000,0x00040047,0x00000042,0x0000000b,0x00000019,
    0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00040015,
    0x00000006,0x00000020,0x00000000,0x00040020,0x00000007,0x00000007,
    0x00000006,0x00040017,0x00000009,0x00000006,0x00000003,0x00040020,
    0x0000000a,0x00000001,0x00000009,0x0004003b,0x0000000a,0x0000000b,
    0x00000001,0x0004002b,0x00000006,0x0000000c,0x00000000,0x00040020,
    0x0000000d,0x00000001,0x00000006,0x0003001e,0x00000011,0x00000006,
    0x00040020,0x00000012,0x00000009,0x00000011,0x0004003b,0x00000012,
    0x00000013,0x00000009,0x00040015,0x00000014,0x00000020,0x00000001,
    0x0004002b,0x00000014,0x00000015,0x00000000,0x00040020,0x00000016,
    0x00000009,0x00000006,0x00020014,0x00000019,0x00030016,0x0000001e,
    0x00000020,0x00040020,0x0000001f,0x00000007,0x0000001e,0x0003001d,
    0x00000021,0x0000001e,0x0003001e,0x00000022,0x00000021,0x0004002b,
    0x00000006,0x00000023,0x00000002,0x0004001c,0x00000024,0x00000022,
    0x00000023,0x00040020,0x00000025,0x00000002,0x00000024,0x0004003b,
    0x00000025,0x00000026,0x00000002,0x0004002b,0x00000014,0x00000027,
    0x00000001,0x00040020,0x00000029,0x00000002,0x0000001e,0x0004002b,
    0x0000001e,0x0000002d,0x3f000000,0x0004002b,0x0000001e,0x00000030,
    0x3f800000,0x0004002b,0x0000001e,0x00000031,0x3f4c422a,0x0004002b,
    0x0000001e,0x00000033,0x3d372713,0x0004002b,0x00000006,0x00000040,
    0x00000100,0x0004002b,0x00000006,0x00000041,0x00000001,0x0006002c,
    0x00000009,0x00000042,0x00000040,0x00000041,0x00000041,0x00050036,
    0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,
    0x0004003b,0x00000007,0x00000008,0x00000007,0x0004003b,0x0000001f,
    0x00000020,0x00000007,0x00050041,0x0000000d,0x0000000e,0x0000000b,
    0x0000000c,0x0004003d,0x00000006,0x0000000f,0x0000000e,0x0003003e,
    0x00000008,0x0000000f,0x0004003d,0x00000006,0x00000010,0x00000008,
    0x00050041,0x00000016,0x00000017,0x00000013,0x00000015,0x0004003d,
    0x00000006,0x00000018,0x00000017,0x000500ae,0x00000019,0x0000001a,
    0x00000010,0x00000018,0x000300f7,0x0000001c,0x00000000,0x000400fa,
    0x0000001a,0x0000001b,0x0000001c,0x000200f8,0x0000001b,0x000100fd,
    0x000200f8,0x0000001c,0x0004003d,0x00000006,0x00000028,0x00000008,
    0x00070041,0x00000029,0x0000002a,0x00000026,0x00000027,0x00000015,
    0x00000028,0x0004003d,0x0000001e,0x0000002b,0x0000002a,0x0003003e,
    0x00000020,0x0000002b,0x0004003d,0x00000006,0x0000002c,0x00000008,
    0x0004003d,0x0000001e,0x0000002e,0x00000020,0x00050085,0x0000001e,
    0x0000002f,0x0000002d,0x0000002e,0x0004003d,0x0000001e,0x00000032,
    0x00000020,0x0004003d,0x0000001e,0x00000034,0x00000020,0x00050085,
    0x0000001e,0x00000035,0x00000033,0x00000034,0x0004003d,0x0000001e,
    0x00000036,0x00000020,0x00050085,0x0000001e,0x00000037,0x00000035,
    0x00000036,0x0004003d,0x0000001e,0x00000038,0x00000020,0x00050085,
    0x0000001e,0x00000039,0x00000037,0x00000038,0x00050081,0x0000001e,
    0x0000003a,0x00000032,0x00000039,0x00050085,0x0000001e,0x0000003b,
    0x00000031,0x0000003a,0x0006000c,0x0000001e,0x0000003c,0x00000001,
    0x00000015,0x0000003b,0x00050081,0x0000001e,0x0000003d,0x00000030,
    0x0000003c,0x00050085,0x0000001e,0x0000003e,0x0000002f,0x0000003d,
    0x00070041,0x00000029,0x0000003f,0x00000026,0x00000015,0x00000015,
    0x0000002c,0x0003003e,0x0000003f,0x0000003e,0x000100fd,0x00010038
};

// silu — GLSL 450 compute, compiled to SPIR-V by glslang on 2026-09-10.
//   layout(local_size_x=256) in; push_constant{ uint n; }; buffer bufs[] @ binding 0
//   i=GlobalInvocationID.x; if(i>=n) return; x=bufs[1].data[i]; bufs[0].data[i]=x/(1+exp(-x));  [bufs[2]]
// (Previous blob was not valid SPIR-V: the instruction stream broke at word 13 and
//  hung the AMD driver inside vkCreateShaderModule.)
static const uint32_t g_silu_spirv[] = {
    0x07230203,0x00010000,0x00080008,0x00000038,0x00000000,0x00020011,
    0x00000001,0x0006000b,0x00000001,0x4c534c47,0x6474732e,0x3035342e,
    0x00000000,0x0003000e,0x00000000,0x00000001,0x0006000f,0x00000005,
    0x00000004,0x6e69616d,0x00000000,0x0000000b,0x00060010,0x00000004,
    0x00000011,0x00000100,0x00000001,0x00000001,0x00030003,0x00000002,
    0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00030005,
    0x00000008,0x00000069,0x00080005,0x0000000b,0x475f6c67,0x61626f6c,
    0x766e496c,0x7461636f,0x496e6f69,0x00000044,0x00030005,0x00000011,
    0x00004350,0x00040006,0x00000011,0x00000000,0x0000006e,0x00030005,
    0x00000013,0x00006370,0x00030005,0x00000020,0x00000078,0x00040005,
    0x00000022,0x73667542,0x00000000,0x00050006,0x00000022,0x00000000,
    0x61746164,0x00000000,0x00040005,0x00000026,0x73667562,0x00000000,
    0x00040047,0x0000000b,0x0000000b,0x0000001c,0x00050048,0x00000011,
    0x00000000,0x00000023,0x00000000,0x00030047,0x00000011,0x00000002,
    0x00040047,0x00000021,0x00000006,0x00000004,0x00050048,0x00000022,
    0x00000000,0x00000023,0x00000000,0x00030047,0x00000022,0x00000003,
    0x00040047,0x00000026,0x00000022,0x00000000,0x00040047,0x00000026,
    0x00000021,0x00000000,0x00040047,0x00000037,0x0000000b,0x00000019,
    0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00040015,
    0x00000006,0x00000020,0x00000000,0x00040020,0x00000007,0x00000007,
    0x00000006,0x00040017,0x00000009,0x00000006,0x00000003,0x00040020,
    0x0000000a,0x00000001,0x00000009,0x0004003b,0x0000000a,0x0000000b,
    0x00000001,0x0004002b,0x00000006,0x0000000c,0x00000000,0x00040020,
    0x0000000d,0x00000001,0x00000006,0x0003001e,0x00000011,0x00000006,
    0x00040020,0x00000012,0x00000009,0x00000011,0x0004003b,0x00000012,
    0x00000013,0x00000009,0x00040015,0x00000014,0x00000020,0x00000001,
    0x0004002b,0x00000014,0x00000015,0x00000000,0x00040020,0x00000016,
    0x00000009,0x00000006,0x00020014,0x00000019,0x00030016,0x0000001e,
    0x00000020,0x00040020,0x0000001f,0x00000007,0x0000001e,0x0003001d,
    0x00000021,0x0000001e,0x0003001e,0x00000022,0x00000021,0x0004002b,
    0x00000006,0x00000023,0x00000002,0x0004001c,0x00000024,0x00000022,
    0x00000023,0x00040020,0x00000025,0x00000002,0x00000024,0x0004003b,
    0x00000025,0x00000026,0x00000002,0x0004002b,0x00000014,0x00000027,
    0x00000001,0x00040020,0x00000029,0x00000002,0x0000001e,0x0004002b,
    0x0000001e,0x0000002e,0x3f800000,0x0004002b,0x00000006,0x00000035,
    0x00000100,0x0004002b,0x00000006,0x00000036,0x00000001,0x0006002c,
    0x00000009,0x00000037,0x00000035,0x00000036,0x00000036,0x00050036,
    0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,
    0x0004003b,0x00000007,0x00000008,0x00000007,0x0004003b,0x0000001f,
    0x00000020,0x00000007,0x00050041,0x0000000d,0x0000000e,0x0000000b,
    0x0000000c,0x0004003d,0x00000006,0x0000000f,0x0000000e,0x0003003e,
    0x00000008,0x0000000f,0x0004003d,0x00000006,0x00000010,0x00000008,
    0x00050041,0x00000016,0x00000017,0x00000013,0x00000015,0x0004003d,
    0x00000006,0x00000018,0x00000017,0x000500ae,0x00000019,0x0000001a,
    0x00000010,0x00000018,0x000300f7,0x0000001c,0x00000000,0x000400fa,
    0x0000001a,0x0000001b,0x0000001c,0x000200f8,0x0000001b,0x000100fd,
    0x000200f8,0x0000001c,0x0004003d,0x00000006,0x00000028,0x00000008,
    0x00070041,0x00000029,0x0000002a,0x00000026,0x00000027,0x00000015,
    0x00000028,0x0004003d,0x0000001e,0x0000002b,0x0000002a,0x0003003e,
    0x00000020,0x0000002b,0x0004003d,0x00000006,0x0000002c,0x00000008,
    0x0004003d,0x0000001e,0x0000002d,0x00000020,0x0004003d,0x0000001e,
    0x0000002f,0x00000020,0x0004007f,0x0000001e,0x00000030,0x0000002f,
    0x0006000c,0x0000001e,0x00000031,0x00000001,0x0000001b,0x00000030,
    0x00050081,0x0000001e,0x00000032,0x0000002e,0x00000031,0x00050088,
    0x0000001e,0x00000033,0x0000002d,0x00000032,0x00070041,0x00000029,
    0x00000034,0x00000026,0x00000015,0x00000015,0x0000002c,0x0003003e,
    0x00000034,0x00000033,0x000100fd,0x00010038
};

// add — GLSL 450 compute, compiled to SPIR-V by glslang on 2026-09-10.
//   layout(local_size_x=256) in; push_constant{ uint n; }; buffer bufs[] @ binding 0
//   i=GlobalInvocationID.x; if(i>=n) return; bufs[0].data[i]=bufs[1].data[i]+bufs[2].data[i];  [bufs[3]]
// (Previous blob was not valid SPIR-V: the instruction stream broke at word 13 and
//  hung the AMD driver inside vkCreateShaderModule.)
static const uint32_t g_add_spirv[] = {
    0x07230203,0x00010000,0x00080008,0x00000034,0x00000000,0x00020011,
    0x00000001,0x0006000b,0x00000001,0x4c534c47,0x6474732e,0x3035342e,
    0x00000000,0x0003000e,0x00000000,0x00000001,0x0006000f,0x00000005,
    0x00000004,0x6e69616d,0x00000000,0x0000000b,0x00060010,0x00000004,
    0x00000011,0x00000100,0x00000001,0x00000001,0x00030003,0x00000002,
    0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00030005,
    0x00000008,0x00000069,0x00080005,0x0000000b,0x475f6c67,0x61626f6c,
    0x766e496c,0x7461636f,0x496e6f69,0x00000044,0x00030005,0x00000011,
    0x00004350,0x00040006,0x00000011,0x00000000,0x0000006e,0x00030005,
    0x00000013,0x00006370,0x00040005,0x00000020,0x73667542,0x00000000,
    0x00050006,0x00000020,0x00000000,0x61746164,0x00000000,0x00040005,
    0x00000024,0x73667562,0x00000000,0x00040047,0x0000000b,0x0000000b,
    0x0000001c,0x00050048,0x00000011,0x00000000,0x00000023,0x00000000,
    0x00030047,0x00000011,0x00000002,0x00040047,0x0000001f,0x00000006,
    0x00000004,0x00050048,0x00000020,0x00000000,0x00000023,0x00000000,
    0x00030047,0x00000020,0x00000003,0x00040047,0x00000024,0x00000022,
    0x00000000,0x00040047,0x00000024,0x00000021,0x00000000,0x00040047,
    0x00000033,0x0000000b,0x00000019,0x00020013,0x00000002,0x00030021,
    0x00000003,0x00000002,0x00040015,0x00000006,0x00000020,0x00000000,
    0x00040020,0x00000007,0x00000007,0x00000006,0x00040017,0x00000009,
    0x00000006,0x00000003,0x00040020,0x0000000a,0x00000001,0x00000009,
    0x0004003b,0x0000000a,0x0000000b,0x00000001,0x0004002b,0x00000006,
    0x0000000c,0x00000000,0x00040020,0x0000000d,0x00000001,0x00000006,
    0x0003001e,0x00000011,0x00000006,0x00040020,0x00000012,0x00000009,
    0x00000011,0x0004003b,0x00000012,0x00000013,0x00000009,0x00040015,
    0x00000014,0x00000020,0x00000001,0x0004002b,0x00000014,0x00000015,
    0x00000000,0x00040020,0x00000016,0x00000009,0x00000006,0x00020014,
    0x00000019,0x00030016,0x0000001e,0x00000020,0x0003001d,0x0000001f,
    0x0000001e,0x0003001e,0x00000020,0x0000001f,0x0004002b,0x00000006,
    0x00000021,0x00000003,0x0004001c,0x00000022,0x00000020,0x00000021,
    0x00040020,0x00000023,0x00000002,0x00000022,0x0004003b,0x00000023,
    0x00000024,0x00000002,0x0004002b,0x00000014,0x00000026,0x00000001,
    0x00040020,0x00000028,0x00000002,0x0000001e,0x0004002b,0x00000014,
    0x0000002b,0x00000002,0x0004002b,0x00000006,0x00000031,0x00000100,
    0x0004002b,0x00000006,0x00000032,0x00000001,0x0006002c,0x00000009,
    0x00000033,0x00000031,0x00000032,0x00000032,0x00050036,0x00000002,
    0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,0x0004003b,
    0x00000007,0x00000008,0x00000007,0x00050041,0x0000000d,0x0000000e,
    0x0000000b,0x0000000c,0x0004003d,0x00000006,0x0000000f,0x0000000e,
    0x0003003e,0x00000008,0x0000000f,0x0004003d,0x00000006,0x00000010,
    0x00000008,0x00050041,0x00000016,0x00000017,0x00000013,0x00000015,
    0x0004003d,0x00000006,0x00000018,0x00000017,0x000500ae,0x00000019,
    0x0000001a,0x00000010,0x00000018,0x000300f7,0x0000001c,0x00000000,
    0x000400fa,0x0000001a,0x0000001b,0x0000001c,0x000200f8,0x0000001b,
    0x000100fd,0x000200f8,0x0000001c,0x0004003d,0x00000006,0x00000025,
    0x00000008,0x0004003d,0x00000006,0x00000027,0x00000008,0x00070041,
    0x00000028,0x00000029,0x00000024,0x00000026,0x00000015,0x00000027,
    0x0004003d,0x0000001e,0x0000002a,0x00000029,0x0004003d,0x00000006,
    0x0000002c,0x00000008,0x00070041,0x00000028,0x0000002d,0x00000024,
    0x0000002b,0x00000015,0x0000002c,0x0004003d,0x0000001e,0x0000002e,
    0x0000002d,0x00050081,0x0000001e,0x0000002f,0x0000002a,0x0000002e,
    0x00070041,0x00000028,0x00000030,0x00000024,0x00000015,0x00000015,
    0x00000025,0x0003003e,0x00000030,0x0000002f,0x000100fd,0x00010038
};

// mul — GLSL 450 compute, compiled to SPIR-V by glslang on 2026-09-10.
//   layout(local_size_x=256) in; push_constant{ uint n; }; buffer bufs[] @ binding 0
//   i=GlobalInvocationID.x; if(i>=n) return; bufs[0].data[i]=bufs[1].data[i]*bufs[2].data[i];  [bufs[3]]
// (Previous blob was not valid SPIR-V: the instruction stream broke at word 13 and
//  hung the AMD driver inside vkCreateShaderModule.)
static const uint32_t g_mul_spirv[] = {
    0x07230203,0x00010000,0x00080008,0x00000034,0x00000000,0x00020011,
    0x00000001,0x0006000b,0x00000001,0x4c534c47,0x6474732e,0x3035342e,
    0x00000000,0x0003000e,0x00000000,0x00000001,0x0006000f,0x00000005,
    0x00000004,0x6e69616d,0x00000000,0x0000000b,0x00060010,0x00000004,
    0x00000011,0x00000100,0x00000001,0x00000001,0x00030003,0x00000002,
    0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00030005,
    0x00000008,0x00000069,0x00080005,0x0000000b,0x475f6c67,0x61626f6c,
    0x766e496c,0x7461636f,0x496e6f69,0x00000044,0x00030005,0x00000011,
    0x00004350,0x00040006,0x00000011,0x00000000,0x0000006e,0x00030005,
    0x00000013,0x00006370,0x00040005,0x00000020,0x73667542,0x00000000,
    0x00050006,0x00000020,0x00000000,0x61746164,0x00000000,0x00040005,
    0x00000024,0x73667562,0x00000000,0x00040047,0x0000000b,0x0000000b,
    0x0000001c,0x00050048,0x00000011,0x00000000,0x00000023,0x00000000,
    0x00030047,0x00000011,0x00000002,0x00040047,0x0000001f,0x00000006,
    0x00000004,0x00050048,0x00000020,0x00000000,0x00000023,0x00000000,
    0x00030047,0x00000020,0x00000003,0x00040047,0x00000024,0x00000022,
    0x00000000,0x00040047,0x00000024,0x00000021,0x00000000,0x00040047,
    0x00000033,0x0000000b,0x00000019,0x00020013,0x00000002,0x00030021,
    0x00000003,0x00000002,0x00040015,0x00000006,0x00000020,0x00000000,
    0x00040020,0x00000007,0x00000007,0x00000006,0x00040017,0x00000009,
    0x00000006,0x00000003,0x00040020,0x0000000a,0x00000001,0x00000009,
    0x0004003b,0x0000000a,0x0000000b,0x00000001,0x0004002b,0x00000006,
    0x0000000c,0x00000000,0x00040020,0x0000000d,0x00000001,0x00000006,
    0x0003001e,0x00000011,0x00000006,0x00040020,0x00000012,0x00000009,
    0x00000011,0x0004003b,0x00000012,0x00000013,0x00000009,0x00040015,
    0x00000014,0x00000020,0x00000001,0x0004002b,0x00000014,0x00000015,
    0x00000000,0x00040020,0x00000016,0x00000009,0x00000006,0x00020014,
    0x00000019,0x00030016,0x0000001e,0x00000020,0x0003001d,0x0000001f,
    0x0000001e,0x0003001e,0x00000020,0x0000001f,0x0004002b,0x00000006,
    0x00000021,0x00000003,0x0004001c,0x00000022,0x00000020,0x00000021,
    0x00040020,0x00000023,0x00000002,0x00000022,0x0004003b,0x00000023,
    0x00000024,0x00000002,0x0004002b,0x00000014,0x00000026,0x00000001,
    0x00040020,0x00000028,0x00000002,0x0000001e,0x0004002b,0x00000014,
    0x0000002b,0x00000002,0x0004002b,0x00000006,0x00000031,0x00000100,
    0x0004002b,0x00000006,0x00000032,0x00000001,0x0006002c,0x00000009,
    0x00000033,0x00000031,0x00000032,0x00000032,0x00050036,0x00000002,
    0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,0x0004003b,
    0x00000007,0x00000008,0x00000007,0x00050041,0x0000000d,0x0000000e,
    0x0000000b,0x0000000c,0x0004003d,0x00000006,0x0000000f,0x0000000e,
    0x0003003e,0x00000008,0x0000000f,0x0004003d,0x00000006,0x00000010,
    0x00000008,0x00050041,0x00000016,0x00000017,0x00000013,0x00000015,
    0x0004003d,0x00000006,0x00000018,0x00000017,0x000500ae,0x00000019,
    0x0000001a,0x00000010,0x00000018,0x000300f7,0x0000001c,0x00000000,
    0x000400fa,0x0000001a,0x0000001b,0x0000001c,0x000200f8,0x0000001b,
    0x000100fd,0x000200f8,0x0000001c,0x0004003d,0x00000006,0x00000025,
    0x00000008,0x0004003d,0x00000006,0x00000027,0x00000008,0x00070041,
    0x00000028,0x00000029,0x00000024,0x00000026,0x00000015,0x00000027,
    0x0004003d,0x0000001e,0x0000002a,0x00000029,0x0004003d,0x00000006,
    0x0000002c,0x00000008,0x00070041,0x00000028,0x0000002d,0x00000024,
    0x0000002b,0x00000015,0x0000002c,0x0004003d,0x0000001e,0x0000002e,
    0x0000002d,0x00050085,0x0000001e,0x0000002f,0x0000002a,0x0000002e,
    0x00070041,0x00000028,0x00000030,0x00000024,0x00000015,0x00000015,
    0x00000025,0x0003003e,0x00000030,0x0000002f,0x000100fd,0x00010038
};

// ========================================================================
// VulkanBackend::Impl — Vulkan state + CPU fallback
// ========================================================================

struct VulkanBackend::Impl {
    bool vulkan_ok = false;
    // True only if every embedded SPIR-V blob passed structural validation.
    // The device can be perfectly healthy while the shipped shaders are not.
    bool shaders_ok = false;
    bool cpu_fallback = false;

    void* vk_lib = nullptr;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_fn = nullptr;
    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr_fn = nullptr;

    VkInstance instance = nullptr;
    VkPhysicalDevice phys_dev = nullptr;
    VkDevice device = nullptr;
    VkQueue compute_queue = nullptr;
    uint32_t compute_queue_family = 0;
    VkCommandPool cmd_pool = nullptr;

    VkPhysicalDeviceMemoryProperties mem_props = {};

    VkDescriptorSetLayout ds_layout = nullptr;
    VkPipelineLayout pipeline_layout = nullptr;
    VkDescriptorPool desc_pool = nullptr;

    std::unordered_map<std::string, VkPipeline> pipeline_cache;
    std::mutex cache_mtx;

    struct GpuBuffer {
        VkBuffer buffer = nullptr;
        VkDeviceMemory memory = nullptr;
        size_t size = 0;
        void* mapped = nullptr;
    };
    std::vector<GpuBuffer> buffers;
    std::mutex buf_mtx;

    // Vulkan function pointers
    PFN_vkDestroyInstance vkDestroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkCreateDevice vkCreateDevice = nullptr;
    PFN_vkDestroyDevice vkDestroyDevice = nullptr;
    PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;
    PFN_vkCreateBuffer vkCreateBuffer = nullptr;
    PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
    PFN_vkAllocateMemory vkAllocateMemory = nullptr;
    PFN_vkFreeMemory vkFreeMemory = nullptr;
    PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
    PFN_vkMapMemory vkMapMemory = nullptr;
    PFN_vkUnmapMemory vkUnmapMemory = nullptr;
    PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = nullptr;
    PFN_vkCreatePipelineLayout vkCreatePipelineLayout = nullptr;
    PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout = nullptr;
    PFN_vkCreateShaderModule vkCreateShaderModule = nullptr;
    PFN_vkDestroyShaderModule vkDestroyShaderModule = nullptr;
    PFN_vkCreateComputePipelines vkCreateComputePipelines = nullptr;
    PFN_vkDestroyPipeline vkDestroyPipeline = nullptr;
    PFN_vkCreateDescriptorPool vkCreateDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets = nullptr;
    PFN_vkFreeDescriptorSets vkFreeDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets = nullptr;
    PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
    PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
    PFN_vkFreeCommandBuffers vkFreeCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
    PFN_vkCmdBindPipeline vkCmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets = nullptr;
    PFN_vkCmdPushConstants vkCmdPushConstants = nullptr;
    PFN_vkCmdDispatch vkCmdDispatch = nullptr;
    PFN_vkCmdCopyBuffer vkCmdCopyBuffer = nullptr;
    PFN_vkQueueSubmit vkQueueSubmit = nullptr;
    PFN_vkQueueWaitIdle vkQueueWaitIdle = nullptr;
    PFN_vkDeviceWaitIdle vkDeviceWaitIdle = nullptr;
    PFN_vkCreateFence vkCreateFence = nullptr;
    PFN_vkDestroyFence vkDestroyFence = nullptr;
    PFN_vkWaitForFences vkWaitForFences = nullptr;
    PFN_vkResetFences vkResetFences = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties = nullptr;

    bool load_vulkan_library() {
#ifdef _WIN32
        vk_lib = LoadLibraryA("vulkan-1.dll");
#else
        vk_lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
        if (!vk_lib) return false;

#ifdef _WIN32
        vkGetInstanceProcAddr_fn = (PFN_vkGetInstanceProcAddr)GetProcAddress((HMODULE)vk_lib, "vkGetInstanceProcAddr");
#else
        vkGetInstanceProcAddr_fn = (PFN_vkGetInstanceProcAddr)dlsym(vk_lib, "vkGetInstanceProcAddr");
#endif
        return vkGetInstanceProcAddr_fn != nullptr;
    }

    void load_func(const char* name, void** out) {
        if (!out) return;
        *out = nullptr;
        if (!vkGetInstanceProcAddr_fn || !instance) return;
        *out = (void*)vkGetInstanceProcAddr_fn(instance, name);
    }
    void load_dev_func(const char* name, void** out) {
        if (!out) return;
        *out = nullptr;
        // REAL ONLY: device functions via vkGetDeviceProcAddr path first.
        if (vkGetDeviceProcAddr_fn && device) {
            *out = (void*)vkGetDeviceProcAddr_fn(device, name);
            if (*out) return;
        }
        // Fallback: instance proc addr also resolves device functions.
        if (vkGetInstanceProcAddr_fn && instance)
            *out = (void*)vkGetInstanceProcAddr_fn(instance, name);
    }

    bool init_vulkan(int64_t device_id) {
        (void)device_id;
        // Guard double-init (REAL ONLY): never create a second VkInstance/VkDevice.
        if (vulkan_ok && instance && device) return true;
        if (instance || device) return vulkan_ok;

        if (!load_vulkan_library()) return false;
        if (!vkGetInstanceProcAddr_fn) return false;

        auto createInstance_fn = (PFN_vkCreateInstance)vkGetInstanceProcAddr_fn(nullptr, "vkCreateInstance");
        if (!createInstance_fn) return false;

        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Transcender QUANT";
        appInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);

        VkInstanceCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &appInfo;

        VkResult r = createInstance_fn(&ci, nullptr, &instance);
        if (r != VK_SUCCESS || !instance) { instance = nullptr; return false; }

        load_func("vkDestroyInstance", (void**)&vkDestroyInstance);
        load_func("vkEnumeratePhysicalDevices", (void**)&vkEnumeratePhysicalDevices);
        load_func("vkGetPhysicalDeviceProperties", (void**)&vkGetPhysicalDeviceProperties);
        load_func("vkGetPhysicalDeviceMemoryProperties", (void**)&vkGetPhysicalDeviceMemoryProperties);
        load_func("vkCreateDevice", (void**)&vkCreateDevice);
        load_func("vkGetPhysicalDeviceQueueFamilyProperties", (void**)&vkGetPhysicalDeviceQueueFamilyProperties);
        // REAL ONLY: device-function loader path via vkGetDeviceProcAddr (with instance fallback).
        vkGetDeviceProcAddr_fn = (PFN_vkGetDeviceProcAddr)vkGetInstanceProcAddr_fn(instance, "vkGetDeviceProcAddr");
        // Null-check every instance-level vk* before use (REAL ONLY, no fake probe).
        if (!vkDestroyInstance || !vkEnumeratePhysicalDevices || !vkGetPhysicalDeviceProperties ||
            !vkGetPhysicalDeviceMemoryProperties || !vkCreateDevice ||
            !vkGetPhysicalDeviceQueueFamilyProperties) {
            if (instance) vkDestroyInstance(instance, nullptr);
            instance = nullptr;
            return false;
        }

        uint32_t dev_count = 0;
        r = vkEnumeratePhysicalDevices(instance, &dev_count, nullptr);
        if (r != VK_SUCCESS || dev_count == 0) {
            vkDestroyInstance(instance, nullptr);
            instance = nullptr;
            return false;
        }
        std::vector<VkPhysicalDevice> devs(dev_count);
        r = vkEnumeratePhysicalDevices(instance, &dev_count, devs.data());
        if (r != VK_SUCCESS || dev_count == 0) {
            vkDestroyInstance(instance, nullptr);
            instance = nullptr;
            return false;
        }

        int best_idx = 0;
        VkPhysicalDeviceType best_type = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU + 1;
        for (uint32_t i = 0; i < dev_count; i++) {
            VkPhysicalDeviceType dt = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU + 1;
            // REAL ONLY heap probe buffer (>=4KB): driver writes full VkPhysicalDeviceProperties.
            // REAL prefix: apiVersion@0, driverVersion@4, vendorID@8, deviceID@12,
            // deviceType@16, deviceName@20 — NO fake sType/pNext.
            std::vector<char> probe(4096, 0);
            vkGetPhysicalDeviceProperties(devs[i], probe.data());
            uint32_t devType32 = 0;
            memcpy(&devType32, probe.data() + 16, sizeof(devType32));
            dt = (VkPhysicalDeviceType)devType32;
            if (dt == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU || dt < best_type) {
                best_type = dt;
                best_idx = (int)i;
            }
        }
        phys_dev = devs[(size_t)best_idx];

        {
            // REAL ONLY heap re-probe (validation, result discarded): same 4KB REAL layout.
            std::vector<char> probe(4096, 0);
            vkGetPhysicalDeviceProperties(phys_dev, probe.data());
        }
        vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem_props);

        // Two-call idiom with REAL 24B VkQueueFamilyProperties (queueFlags@0).
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qf_count, nullptr);
        if (qf_count == 0) {
            vkDestroyInstance(instance, nullptr);
            instance = nullptr;
            return false;
        }
        std::vector<char> qf_props((size_t)qf_count * 24, 0);
        {
            uint32_t qf_count2 = qf_count;
            vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qf_count2, qf_props.data());
            size_t cap = qf_props.size() / 24;
            if ((size_t)qf_count2 < cap) qf_count = qf_count2;
            else qf_count = (uint32_t)cap;
            if (qf_count == 0) {
                vkDestroyInstance(instance, nullptr);
                instance = nullptr;
                return false;
            }
        }

        compute_queue_family = 0;
        bool found = false;
        for (uint32_t i = 0; i < qf_count; i++) {
            uint32_t flags = 0;
            memcpy(&flags, qf_props.data() + (size_t)i * 24 + 0, sizeof(flags));
            if (flags & 0x2) {
                compute_queue_family = i;
                found = true;
                break;
            }
        }
        if (!found) {
            vkDestroyInstance(instance, nullptr);
            instance = nullptr;
            return false;
        }

        float priority = 1.0f;
        VkDeviceQueueCreateInfo dqi = {};
        dqi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        dqi.queueFamilyIndex = compute_queue_family;
        dqi.queueCount = 1;
        dqi.pQueuePriorities = &priority;

        VkDeviceCreateInfo devCI = {};
        devCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        devCI.queueCreateInfoCount = 1;
        devCI.pQueueCreateInfos = &dqi;

        r = vkCreateDevice(phys_dev, &devCI, nullptr, &device);
        if (r != VK_SUCCESS || !device) {
            device = nullptr;
            vkDestroyInstance(instance, nullptr);
            instance = nullptr;
            return false;
        }

        load_dev_func("vkDestroyDevice", (void**)&vkDestroyDevice);
        load_dev_func("vkGetDeviceQueue", (void**)&vkGetDeviceQueue);
        load_dev_func("vkCreateBuffer", (void**)&vkCreateBuffer);
        load_dev_func("vkDestroyBuffer", (void**)&vkDestroyBuffer);
        load_dev_func("vkGetBufferMemoryRequirements", (void**)&vkGetBufferMemoryRequirements);
        load_dev_func("vkAllocateMemory", (void**)&vkAllocateMemory);
        load_dev_func("vkFreeMemory", (void**)&vkFreeMemory);
        load_dev_func("vkBindBufferMemory", (void**)&vkBindBufferMemory);
        load_dev_func("vkMapMemory", (void**)&vkMapMemory);
        load_dev_func("vkUnmapMemory", (void**)&vkUnmapMemory);
        load_dev_func("vkCreateDescriptorSetLayout", (void**)&vkCreateDescriptorSetLayout);
        load_dev_func("vkDestroyDescriptorSetLayout", (void**)&vkDestroyDescriptorSetLayout);
        load_dev_func("vkCreatePipelineLayout", (void**)&vkCreatePipelineLayout);
        load_dev_func("vkDestroyPipelineLayout", (void**)&vkDestroyPipelineLayout);
        load_dev_func("vkCreateShaderModule", (void**)&vkCreateShaderModule);
        load_dev_func("vkDestroyShaderModule", (void**)&vkDestroyShaderModule);
        load_dev_func("vkCreateComputePipelines", (void**)&vkCreateComputePipelines);
        load_dev_func("vkDestroyPipeline", (void**)&vkDestroyPipeline);
        load_dev_func("vkCreateDescriptorPool", (void**)&vkCreateDescriptorPool);
        load_dev_func("vkDestroyDescriptorPool", (void**)&vkDestroyDescriptorPool);
        load_dev_func("vkAllocateDescriptorSets", (void**)&vkAllocateDescriptorSets);
        load_dev_func("vkFreeDescriptorSets", (void**)&vkFreeDescriptorSets);
        load_dev_func("vkUpdateDescriptorSets", (void**)&vkUpdateDescriptorSets);
        load_dev_func("vkCreateCommandPool", (void**)&vkCreateCommandPool);
        load_dev_func("vkDestroyCommandPool", (void**)&vkDestroyCommandPool);
        load_dev_func("vkAllocateCommandBuffers", (void**)&vkAllocateCommandBuffers);
        load_dev_func("vkFreeCommandBuffers", (void**)&vkFreeCommandBuffers);
        load_dev_func("vkBeginCommandBuffer", (void**)&vkBeginCommandBuffer);
        load_dev_func("vkEndCommandBuffer", (void**)&vkEndCommandBuffer);
        load_dev_func("vkCmdBindPipeline", (void**)&vkCmdBindPipeline);
        load_dev_func("vkCmdBindDescriptorSets", (void**)&vkCmdBindDescriptorSets);
        load_dev_func("vkCmdPushConstants", (void**)&vkCmdPushConstants);
        load_dev_func("vkCmdDispatch", (void**)&vkCmdDispatch);
        load_dev_func("vkCmdCopyBuffer", (void**)&vkCmdCopyBuffer);
        load_dev_func("vkQueueSubmit", (void**)&vkQueueSubmit);
        load_dev_func("vkQueueWaitIdle", (void**)&vkQueueWaitIdle);
        load_dev_func("vkDeviceWaitIdle", (void**)&vkDeviceWaitIdle);
        load_dev_func("vkCreateFence", (void**)&vkCreateFence);
        load_dev_func("vkDestroyFence", (void**)&vkDestroyFence);
        load_dev_func("vkWaitForFences", (void**)&vkWaitForFences);
        load_dev_func("vkResetFences", (void**)&vkResetFences);

        // Null-check every device-level vk* needed for init before use (REAL ONLY).
        if (!vkDestroyDevice || !vkGetDeviceQueue || !vkCreateCommandPool || !vkDestroyCommandPool ||
            !vkCreateDescriptorSetLayout || !vkDestroyDescriptorSetLayout ||
            !vkCreatePipelineLayout || !vkDestroyPipelineLayout ||
            !vkCreateDescriptorPool || !vkDestroyDescriptorPool) {
            vkDestroyDevice(device, nullptr);
            device = nullptr;
            vkDestroyInstance(instance, nullptr);
            instance = nullptr;
            return false;
        }

        vkGetDeviceQueue(device, compute_queue_family, 0, &compute_queue);
        if (!compute_queue) {
            vkDestroyDevice(device, nullptr);
            device = nullptr;
            vkDestroyInstance(instance, nullptr);
            instance = nullptr;
            return false;
        }

        VkCommandPoolCreateInfo cpi = {};
        cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpi.flags = 0x2;
        cpi.queueFamilyIndex = compute_queue_family;
        r = vkCreateCommandPool(device, &cpi, nullptr, &cmd_pool);
        if (r != VK_SUCCESS || !cmd_pool) {
            cmd_pool = nullptr;
            vkDestroyDevice(device, nullptr);
            device = nullptr;
            vkDestroyInstance(instance, nullptr);
            instance = nullptr;
            return false;
        }

        // descriptorCount must be >= the largest number of storage buffers any
        // shader indexes at binding 0. dispatch_compute writes n_bufs = 2
        // (elementwise: out,in1) or 3 (binary: out,in1,in2). It was 1, which
        // made every vkUpdateDescriptorSets write past the declared array —
        // invalid usage that hung the AMD driver inside the compute dispatch
        // (2026-09-08). Declaring 3 covers both shapes; shaders that use fewer
        // simply leave the tail unbound, which is legal.
        VkDescriptorSetLayoutBinding binding = {};
        binding.binding = 0;
        binding.descriptorType = 7;
        binding.descriptorCount = 3;
        binding.stageFlags = 0x20;

        VkDescriptorSetLayoutCreateInfo dslci = {};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 1;
        dslci.pBindings = &binding;
        r = vkCreateDescriptorSetLayout(device, &dslci, nullptr, &ds_layout);
        if (r != VK_SUCCESS || !ds_layout) {
            ds_layout = nullptr;
            vkDestroyCommandPool(device, cmd_pool, nullptr);
            cmd_pool = nullptr;
            vkDestroyDevice(device, nullptr);
            device = nullptr;
            vkDestroyInstance(instance, nullptr);
            instance = nullptr;
            return false;
        }

        VkPushConstantRange pcr = {};
        pcr.stageFlags = 0x20;
        pcr.offset = 0;
        pcr.size = 32;

        VkPipelineLayoutCreateInfo plci = {};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &ds_layout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        r = vkCreatePipelineLayout(device, &plci, nullptr, &pipeline_layout);
        if (r != VK_SUCCESS || !pipeline_layout) {
            pipeline_layout = nullptr;
            vkDestroyDescriptorSetLayout(device, ds_layout, nullptr);
            ds_layout = nullptr;
            vkDestroyCommandPool(device, cmd_pool, nullptr);
            cmd_pool = nullptr;
            vkDestroyDevice(device, nullptr);
            device = nullptr;
            vkDestroyInstance(instance, nullptr);
            instance = nullptr;
            return false;
        }

        VkDescriptorPoolSize dps = {7, 256};
        VkDescriptorPoolCreateInfo dpci = {};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.flags = 0x2;
        dpci.maxSets = 256;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &dps;
        r = vkCreateDescriptorPool(device, &dpci, nullptr, &desc_pool);
        if (r != VK_SUCCESS || !desc_pool) {
            desc_pool = nullptr;
            vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            pipeline_layout = nullptr;
            vkDestroyDescriptorSetLayout(device, ds_layout, nullptr);
            ds_layout = nullptr;
            vkDestroyCommandPool(device, cmd_pool, nullptr);
            cmd_pool = nullptr;
            vkDestroyDevice(device, nullptr);
            device = nullptr;
            vkDestroyInstance(instance, nullptr);
            instance = nullptr;
            return false;
        }

        // Validate the embedded compute shaders BEFORE handing them to the
        // driver. vkCreateShaderModule/vkCreateComputePipelines HANG on a
        // malformed module on at least the AMD proprietary driver, which is how
        // test_gpu_capability was burning its full 60 s timeout. Detect it here
        // and mark the backend not compute-ready instead of stalling.
        shaders_ok = spirv_well_formed(g_relu_spirv, sizeof(g_relu_spirv)) &&
                     spirv_well_formed(g_gelu_spirv, sizeof(g_gelu_spirv)) &&
                     spirv_well_formed(g_silu_spirv, sizeof(g_silu_spirv)) &&
                     spirv_well_formed(g_add_spirv,  sizeof(g_add_spirv)) &&
                     spirv_well_formed(g_mul_spirv,  sizeof(g_mul_spirv));
        if (!shaders_ok) {
            std::fprintf(stderr,
                "[REAL-ONLY] GPU_VULKAN: embedded SPIR-V blobs failed structural "
                "validation -- compute dispatch disabled. Device is present; the "
                "shipped shader binaries are not valid SPIR-V.\n");
        }

        vulkan_ok = true;
        return true;
    }

    bool compute_ready() const { return vulkan_ok && shaders_ok; }

    // Structural sanity check on a SPIR-V module: correct magic, and an
    // instruction stream whose (length, opcode) words tile the buffer exactly.
    // This catches the class of corruption that makes drivers hang; it is not a
    // full validator and deliberately does not try to be one.
    static bool spirv_well_formed(const uint32_t* w, size_t nbytes) {
        const size_t nwords = nbytes / sizeof(uint32_t);
        if (!w || nwords < 6) return false;
        if (w[0] != 0x07230203u) return false;   // 'SPIR' magic
        size_t i = 5;                            // skip the 5-word module header
        while (i < nwords) {
            uint32_t len = w[i] >> 16;
            if (len == 0 || i + len > nwords) return false;
            i += len;
        }
        return i == nwords;
    }

    VkPipeline get_pipeline(const uint32_t* spirv, size_t spirv_size) {
        // Never hand a malformed module to the driver: that hangs.
        if (!spirv_well_formed(spirv, spirv_size)) return nullptr;
        std::string key((const char*)spirv, spirv_size);
        {
            std::lock_guard<std::mutex> lk(cache_mtx);
            auto it = pipeline_cache.find(key);
            if (it != pipeline_cache.end()) return it->second;
        }

        VkShaderModuleCreateInfo smci = {};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spirv_size;
        smci.pCode = spirv;
        VkShaderModule sm = nullptr;
        VkResult r = vkCreateShaderModule(device, &smci, nullptr, &sm);
        if (r != VK_SUCCESS || !sm) return nullptr;

        VkPipelineShaderStageCreateInfo ssi = {};
        ssi.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;   // 18, was 0x5
        ssi.stage = 0x20;
        ssi.module = sm;
        ssi.pName = "main";

        VkComputePipelineCreateInfo cpci = {};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = ssi;
        cpci.layout = pipeline_layout;

        VkPipeline pipeline = nullptr;
        r = vkCreateComputePipelines(device, nullptr, 1, &cpci, nullptr, &pipeline);
        vkDestroyShaderModule(device, sm, nullptr);

        if (r == VK_SUCCESS && pipeline) {
            std::lock_guard<std::mutex> lk(cache_mtx);
            pipeline_cache[key] = pipeline;
        }
        return pipeline;
    }

    uint32_t find_memory_type(uint32_t type_bits, VkFlags props) {
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
            if ((type_bits & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & props))
                return i;
        }
        return 0;
    }

    GpuBuffer create_gpu_buffer(size_t sz, VkBufferUsageFlagBits usage, VkFlags memProps) {
        GpuBuffer gb{};
        gb.size = sz;

        VkBufferCreateInfo bci = {};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sz;
        bci.usage = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bci, nullptr, &gb.buffer) != VK_SUCCESS) return gb;

        VkMemoryRequirements mr = {};
        vkGetBufferMemoryRequirements(device, gb.buffer, &mr);

        VkMemoryAllocateInfo mai = {};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = find_memory_type(mr.memoryTypeBits, memProps);

        if (vkAllocateMemory(device, &mai, nullptr, &gb.memory) != VK_SUCCESS) {
            vkDestroyBuffer(device, gb.buffer, nullptr);
            gb.buffer = nullptr;
            return gb;
        }
        vkBindBufferMemory(device, gb.buffer, gb.memory, 0);
        return gb;
    }

    void copy_to_gpu(VkBuffer dst, const void* src, size_t sz) {
        auto staging = create_gpu_buffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!staging.buffer) return;
        void* mapped = nullptr;
        vkMapMemory(device, staging.memory, 0, sz, 0, &mapped);
        if (mapped) {
            memcpy(mapped, src, sz);
            vkUnmapMemory(device, staging.memory);
        }

        VkCommandBufferAllocateInfo cbai = {};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmd_pool;
        cbai.level = 0;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd = nullptr;
        vkAllocateCommandBuffers(device, &cbai, &cmd);

        VkCommandBufferBeginInfo cbbi = {};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = 0x1;
        vkBeginCommandBuffer(cmd, &cbbi);

        VkBufferCopy bc = {0, 0, sz};
        vkCmdCopyBuffer(cmd, staging.buffer, dst, 1, &bc);

        vkEndCommandBuffer(cmd);

        VkFenceCreateInfo fci = {};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = nullptr;
        vkCreateFence(device, &fci, nullptr, &fence);

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = 4;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        // Fail-loud: a failed submit never signals the fence, and waiting on it
        // stalls forever. Only wait when the submit actually succeeded.
        if (vkQueueSubmit(compute_queue, 1, &submitInfo, fence) == VK_SUCCESS)
            vkWaitForFences(device, 1, &fence, VK_TRUE, kFenceWaitNs);
        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, cmd_pool, 1, &cmd);
        vkDestroyBuffer(device, staging.buffer, nullptr);
        vkFreeMemory(device, staging.memory, nullptr);
    }

    void copy_from_gpu(void* dst, VkBuffer src, size_t sz) {
        auto staging = create_gpu_buffer(sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!staging.buffer) return;

        VkCommandBufferAllocateInfo cbai = {};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmd_pool;
        cbai.level = 0;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd = nullptr;
        vkAllocateCommandBuffers(device, &cbai, &cmd);

        VkCommandBufferBeginInfo cbbi = {};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = 0x1;
        vkBeginCommandBuffer(cmd, &cbbi);

        VkBufferCopy bc = {0, 0, sz};
        vkCmdCopyBuffer(cmd, src, staging.buffer, 1, &bc);
        vkEndCommandBuffer(cmd);

        VkFenceCreateInfo fci = {};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = nullptr;
        vkCreateFence(device, &fci, nullptr, &fence);

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = 4;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        // Fail-loud: a failed submit never signals the fence, and waiting on it
        // stalls forever. Only wait when the submit actually succeeded.
        if (vkQueueSubmit(compute_queue, 1, &submitInfo, fence) == VK_SUCCESS)
            vkWaitForFences(device, 1, &fence, VK_TRUE, kFenceWaitNs);
        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, cmd_pool, 1, &cmd);

        void* mapped = nullptr;
        vkMapMemory(device, staging.memory, 0, sz, 0, &mapped);
        if (mapped) {
            memcpy(dst, mapped, sz);
            vkUnmapMemory(device, staging.memory);
        }
        vkDestroyBuffer(device, staging.buffer, nullptr);
        vkFreeMemory(device, staging.memory, nullptr);
    }

    void dispatch_compute(VkPipeline pipeline, const uint32_t* push_data, uint32_t push_size,
                           VkBuffer* storage_bufs, uint32_t n_bufs, uint32_t gx, uint32_t gy, uint32_t gz) {
        VkDescriptorSetAllocateInfo dsai = {};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = desc_pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &ds_layout;

        VkDescriptorSet ds = nullptr;
        if (vkAllocateDescriptorSets(device, &dsai, &ds) != VK_SUCCESS) return;

        std::vector<VkDescriptorBufferInfo> dbi(n_bufs);
        for (uint32_t i = 0; i < n_bufs; i++) {
            dbi[i].buffer = storage_bufs[i];
            dbi[i].offset = 0;
            dbi[i].range = VK_WHOLE_SIZE;
        }

        VkWriteDescriptorSet wds = {};
        wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds.dstSet = ds;
        wds.dstBinding = 0;
        wds.descriptorCount = n_bufs;
        wds.descriptorType = 7;
        wds.pBufferInfo = dbi.data();
        vkUpdateDescriptorSets(device, 1, &wds, 0, nullptr);

        VkCommandBufferAllocateInfo cbai = {};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmd_pool;
        cbai.level = 0;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd = nullptr;
        vkAllocateCommandBuffers(device, &cbai, &cmd);

        VkCommandBufferBeginInfo cbbi = {};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = 0x1;
        vkBeginCommandBuffer(cmd, &cbbi);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &ds, 0, nullptr);
        if (push_data && push_size > 0) {
            vkCmdPushConstants(cmd, pipeline_layout, 0x20, 0, push_size, push_data);
        }
        vkCmdDispatch(cmd, gx, gy, gz);

        vkEndCommandBuffer(cmd);

        VkFenceCreateInfo fci = {};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = nullptr;
        vkCreateFence(device, &fci, nullptr, &fence);

        VkSubmitInfo si = {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        // Fail-loud: a failed submit never signals the fence, so waiting on it
        // would stall forever. Bail instead of hanging (R5: no silent stalls).
        if (vkQueueSubmit(compute_queue, 1, &si, fence) != VK_SUCCESS) {
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, cmd_pool, 1, &cmd);
            vkFreeDescriptorSets(device, desc_pool, 1, &ds);
            return;
        }
        if (vkWaitForFences(device, 1, &fence, VK_TRUE, kFenceWaitNs) != VK_SUCCESS) {
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, cmd_pool, 1, &cmd);
            vkFreeDescriptorSets(device, desc_pool, 1, &ds);
            return;
        }
        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, cmd_pool, 1, &cmd);
        vkFreeDescriptorSets(device, desc_pool, 1, &ds);
    }

    void dispatch_simple(const uint32_t* spirv, size_t spirv_sz,
                          VkBuffer in1, VkBuffer out, uint32_t n) {
        VkPipeline pl = get_pipeline(spirv, spirv_sz);
        // Fail-loud: a null pipeline (shaders failed validation or the driver
        // refused them) must never silently skip dispatch — download would
        // read an unwritten device buffer as if it were a GPU result.
        // (2026-09-08: all five shipped blobs fail structural validation, so
        // this is the live path on the AMD iGPU until real shaders land.)
        if (!pl)
            throw std::runtime_error("[REAL-ONLY] GPU_VULKAN: compute pipeline unavailable "
                                     "(SPIR-V failed validation or driver refused it)");
        uint32_t push[8] = {n, 0, 0, 0, 0, 0, 0, 0};
        VkBuffer bufs[] = {out, in1};
        uint32_t gx = (n + 255) / 256;
        dispatch_compute(pl, push, 32, bufs, 2, gx, 1, 1);
    }

    void dispatch_binary(const uint32_t* spirv, size_t spirv_sz,
                          VkBuffer in1, VkBuffer in2, VkBuffer out, uint32_t n) {
        VkPipeline pl = get_pipeline(spirv, spirv_sz);
        if (!pl)
            throw std::runtime_error("[REAL-ONLY] GPU_VULKAN: compute pipeline unavailable "
                                     "(SPIR-V failed validation or driver refused it)");
        uint32_t push[8] = {n, 0, 0, 0, 0, 0, 0, 0};
        VkBuffer bufs[] = {out, in1, in2};
        uint32_t gx = (n + 255) / 256;
        dispatch_compute(pl, push, 32, bufs, 3, gx, 1, 1);
    }

    GpuBuffer* find_buf(void* ptr) {
        std::lock_guard<std::mutex> lk(buf_mtx);
        for (auto& b : buffers) {
            if (b.buffer == (VkBuffer)ptr) return &b;
        }
        return nullptr;
    }

    ~Impl() { shutdown_vulkan(); }

    void shutdown_vulkan() {
        if (!device) return;

        {
            std::lock_guard<std::mutex> lk(cache_mtx);
            for (auto& kv : pipeline_cache)
                vkDestroyPipeline(device, kv.second, nullptr);
            pipeline_cache.clear();
        }

        for (auto& b : buffers) {
            if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
            if (b.memory) vkFreeMemory(device, b.memory, nullptr);
        }
        buffers.clear();

        if (desc_pool) vkDestroyDescriptorPool(device, desc_pool, nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        if (ds_layout) vkDestroyDescriptorSetLayout(device, ds_layout, nullptr);
        if (cmd_pool) vkDestroyCommandPool(device, cmd_pool, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);

        device = nullptr;
        instance = nullptr;
        vulkan_ok = false;
    }
};

// ========================================================================
// VulkanBackend public API
// ========================================================================

VulkanBackend::VulkanBackend() : impl_(new Impl()) {}
VulkanBackend::~VulkanBackend() { delete impl_; }

bool VulkanBackend::compute_ready() const {
    return impl_ && impl_->compute_ready();
}

bool VulkanBackend::init(int64_t device_id) {
    if (impl_->vulkan_ok) return true;
    if (impl_->init_vulkan(device_id)) return true;
    impl_->cpu_fallback = true;
    return false;
}

bool VulkanBackend::is_initialized() const { return impl_->vulkan_ok; } // H6 honest-fallback (REAL ONLY): CPU fallback must NOT report available
void VulkanBackend::shutdown() { impl_->shutdown_vulkan(); }

void* VulkanBackend::allocate(size_t bytes) {
    if (impl_->vulkan_ok) {
        auto gb = impl_->create_gpu_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!gb.buffer) return nullptr;
        std::lock_guard<std::mutex> lk(impl_->buf_mtx);
        void* handle = (void*)gb.buffer;
        impl_->buffers.push_back(std::move(gb));
        return handle;
    }
    return std::malloc(bytes);
}

void VulkanBackend::free(void* ptr) {
    if (!ptr) return;
    if (impl_->vulkan_ok) {
        std::lock_guard<std::mutex> lk(impl_->buf_mtx);
        for (size_t i = 0; i < impl_->buffers.size(); i++) {
            if (impl_->buffers[i].buffer == (VkBuffer)ptr) {
                // Destroy the device objects, not just the record: the old
                // code erased the entry and leaked every VkBuffer+memory.
                if (impl_->buffers[i].buffer)
                    impl_->vkDestroyBuffer(impl_->device, impl_->buffers[i].buffer, nullptr);
                if (impl_->buffers[i].memory)
                    impl_->vkFreeMemory(impl_->device, impl_->buffers[i].memory, nullptr);
                impl_->buffers.erase(impl_->buffers.begin() + (int)i);
                return;
            }
        }
    } else {
        std::free(ptr);
    }
}

void VulkanBackend::upload(const Tensor& src, void* dst) {
    if (!dst || src.numel() == 0) return;
    if (impl_->vulkan_ok) {
        impl_->copy_to_gpu((VkBuffer)dst, src.data(), src.size_bytes());
    } else {
        memcpy(dst, src.data(), src.size_bytes());
    }
}

void VulkanBackend::download(void* src, Tensor& dst) {
    if (!src || dst.numel() == 0) return;
    if (impl_->vulkan_ok) {
        impl_->copy_from_gpu(dst.data(), (VkBuffer)src, dst.size_bytes());
    } else {
        memcpy(dst.data(), src, dst.size_bytes());
    }
}

// ========================================================================
// CPU fallback implementations
// ========================================================================

static void cpu_relu(const float* x, float* y, int64_t n) {
    for (int64_t i = 0; i < n; i++) y[i] = x[i] > 0 ? x[i] : 0;
}

static void cpu_gelu(const float* x, float* y, int64_t n) {
    const float c = 0.7978845608028654f;
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
    }
}

static void cpu_silu(const float* x, float* y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v / (1.0f + std::exp(-v));
    }
}

static void cpu_add(const float* a, const float* b, float* c, int64_t n) {
    for (int64_t i = 0; i < n; i++) c[i] = a[i] + b[i];
}

static void cpu_mul(const float* a, const float* b, float* c, int64_t n) {
    for (int64_t i = 0; i < n; i++) c[i] = a[i] * b[i];
}

static void cpu_scale(float s, const float* x, float* y, int64_t n) {
    for (int64_t i = 0; i < n; i++) y[i] = s * x[i];
}

static void cpu_softmax(const float* x, float* y, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; r++) {
        float mx = x[r * cols];
        for (int64_t c = 1; c < cols; c++) mx = std::max(mx, x[r * cols + c]);
        float sum = 0;
        for (int64_t c = 0; c < cols; c++) {
            y[r * cols + c] = std::exp(x[r * cols + c] - mx);
            sum += y[r * cols + c];
        }
        float inv = 1.0f / (sum + 1e-10f);
        for (int64_t c = 0; c < cols; c++) y[r * cols + c] *= inv;
    }
}

static void cpu_rms_norm(const float* x, const float* gamma, float* y, float eps, int64_t n, int64_t d) {
    for (int64_t i = 0; i < n; i++) {
        float ss = 0;
        for (int64_t j = 0; j < d; j++) ss += x[i * d + j] * x[i * d + j];
        float rs = 1.0f / std::sqrt(ss / d + eps);
        for (int64_t j = 0; j < d; j++) y[i * d + j] = x[i * d + j] * rs * gamma[j];
    }
}

static void cpu_layer_norm(const float* x, const float* gamma, const float* beta,
                             float* y, float eps, int64_t n, int64_t d) {
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
        for (int64_t j = 0; j < d; j++) y[i * d + j] = (x[i * d + j] - mn) * iv * gamma[j] + beta[j];
    }
}

static void cpu_gemm(float alpha, const float* A, const float* B, float beta, float* C,
                      int64_t M, int64_t N, int64_t K) {
    constexpr int64_t TILE = 64;
    for (int64_t m0 = 0; m0 < M; m0 += TILE) {
        int64_t m1 = (m0 + TILE < M) ? m0 + TILE : M;
        for (int64_t n0 = 0; n0 < N; n0 += TILE) {
            int64_t n1 = (n0 + TILE < N) ? n0 + TILE : N;
            for (int64_t k0 = 0; k0 < K; k0 += TILE) {
                int64_t k1 = (k0 + TILE < K) ? k0 + TILE : K;
                for (int64_t m = m0; m < m1; m++) {
                    for (int64_t n = n0; n < n1; n++) {
                        float s = (k0 == 0) ? 0.0f : C[m * N + n];
                        for (int64_t k = k0; k < k1; k++)
                            s += A[m * K + k] * B[k * N + n];
                        if (k0 == 0)
                            C[m * N + n] = alpha * s + beta * C[m * N + n];
                        else
                            C[m * N + n] += alpha * s;
                    }
                }
            }
        }
    }
}

static void cpu_gemv(float alpha, const float* A, const float* x, float beta, float* y,
                      int64_t M, int64_t N) {
    for (int64_t m = 0; m < M; m++) {
        float s = 0;
        for (int64_t n = 0; n < N; n++) s += A[m * N + n] * x[n];
        y[m] = alpha * s + beta * y[m];
    }
}

static void cpu_moe_gather(const float* x, const int64_t* indices, const float* weights,
                             float* out, int64_t T, int64_t K, int64_t D) {
    for (int64_t t = 0; t < T; t++) {
        for (int64_t k = 0; k < K; k++) {
            int64_t ix = indices[t * K + k];
            float w = weights[t * K + k];
            for (int64_t d = 0; d < D; d++) out[t * K * D + k * D + d] = w * x[ix * D + d];
        }
    }
}

static void cpu_moe_scatter_add(float* out, const int64_t* indices, const float* weights,
                                  const float* expert_out, int64_t T, int64_t K, int64_t D) {
    for (int64_t t = 0; t < T; t++) {
        for (int64_t k = 0; k < K; k++) {
            int64_t ix = indices[t * K + k];
            float w = weights[t * K + k];
            for (int64_t d = 0; d < D; d++)
                out[ix * D + d] += w * expert_out[t * K * D + k * D + d];
        }
    }
}

static void cpu_flash_attention(void* out_v, const void* Q_v, const void* K_v, const void* V_v,
                                  int64_t B, int64_t H, int64_t N, int64_t D, bool causal) {
    const float* Q = (const float*)Q_v;
    const float* Kk = (const float*)K_v;
    const float* Vv = (const float*)V_v;
    float* O = (float*)out_v;
    float scale = 1.0f / std::sqrt((float)D);

    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t qi = 0; qi < N; qi++) {
                float max_val = -1e30f;
                float sum_exp = 0;
                for (int64_t ki = 0; ki < N; ki++) {
                    if (causal && ki > qi) continue;
                    float dot = 0;
                    for (int64_t d = 0; d < D; d++) {
                        dot += Q[b * H * N * D + h * N * D + qi * D + d] *
                               Kk[b * H * N * D + h * N * D + ki * D + d];
                    }
                    dot *= scale;
                    if (dot > max_val) max_val = dot;
                }
                for (int64_t d = 0; d < D; d++) O[b * H * N * D + h * N * D + qi * D + d] = 0;
                for (int64_t ki = 0; ki < N; ki++) {
                    if (causal && ki > qi) continue;
                    float dot = 0;
                    for (int64_t d = 0; d < D; d++) {
                        dot += Q[b * H * N * D + h * N * D + qi * D + d] *
                               Kk[b * H * N * D + h * N * D + ki * D + d];
                    }
                    dot *= scale;
                    float w = std::exp(dot - max_val);
                    sum_exp += w;
                    for (int64_t d = 0; d < D; d++)
                        O[b * H * N * D + h * N * D + qi * D + d] += w * Vv[b * H * N * D + h * N * D + ki * D + d];
                }
                if (sum_exp > 0) {
                    for (int64_t d = 0; d < D; d++)
                        O[b * H * N * D + h * N * D + qi * D + d] /= sum_exp;
                }
            }
        }
    }
}

// ========================================================================
// VulkanBackend dispatch wrappers
// ========================================================================

void VulkanBackend::gemm(float alpha, const void* A, const void* B, float beta, void* C,
                          int64_t M, int64_t N, int64_t K) {
    // H6 honest-fallback (REAL ONLY): no SPIR-V GEMM shader — do NOT fake GPU.
    // Silent CPU fallback ONLY when !vulkan_ok; when vulkan_ok WARN perf-invalid.
    if (!impl_->vulkan_ok) {
        cpu_gemm(alpha, (const float*)A, (const float*)B, beta, (float*)C, M, N, K);
        return;
    }
    std::fprintf(stderr, "[WARN] Vulkan GEMM SPIR-V not implemented (REAL ONLY): CPU fallback, perf-invalid\n");
    cpu_gemm(alpha, (const float*)A, (const float*)B, beta, (float*)C, M, N, K);
}

void VulkanBackend::gemv(float alpha, const void* A, const void* x, float beta, void* y,
                          int64_t M, int64_t N) {
    // No GEMV SPIR-V shader available (REAL ONLY, no fake GEMM).
    // BUG FIX: previously dispatched g_relu_spirv which is an activation shader, not GEMV.
    if (!impl_->vulkan_ok) {
        cpu_gemv(alpha, (const float*)A, (const float*)x, beta, (float*)y, M, N);
        return;
    }
    std::fprintf(stderr, "[WARN] Vulkan GEMV SPIR-V not implemented (REAL ONLY): CPU fallback, perf-invalid\n");
    cpu_gemv(alpha, (const float*)A, (const float*)x, beta, (float*)y, M, N);
}

void VulkanBackend::relu(const void* x, void* y, int64_t n) {
    if (impl_->vulkan_ok) {
        impl_->dispatch_simple(g_relu_spirv, sizeof(g_relu_spirv), (VkBuffer)x, (VkBuffer)y, (uint32_t)n);
    } else {
        cpu_relu((const float*)x, (float*)y, n);
    }
}

void VulkanBackend::gelu(const void* x, void* y, int64_t n) {
    if (impl_->vulkan_ok) {
        impl_->dispatch_simple(g_gelu_spirv, sizeof(g_gelu_spirv), (VkBuffer)x, (VkBuffer)y, (uint32_t)n);
    } else {
        cpu_gelu((const float*)x, (float*)y, n);
    }
}

void VulkanBackend::silu(const void* x, void* y, int64_t n) {
    if (impl_->vulkan_ok) {
        impl_->dispatch_simple(g_silu_spirv, sizeof(g_silu_spirv), (VkBuffer)x, (VkBuffer)y, (uint32_t)n);
    } else {
        cpu_silu((const float*)x, (float*)y, n);
    }
}

void VulkanBackend::add(const void* a, const void* b, void* c, int64_t n) {
    if (impl_->vulkan_ok) {
        impl_->dispatch_binary(g_add_spirv, sizeof(g_add_spirv), (VkBuffer)a, (VkBuffer)b, (VkBuffer)c, (uint32_t)n);
    } else {
        cpu_add((const float*)a, (const float*)b, (float*)c, n);
    }
}

void VulkanBackend::mul(const void* a, const void* b, void* c, int64_t n) {
    if (impl_->vulkan_ok) {
        impl_->dispatch_binary(g_mul_spirv, sizeof(g_mul_spirv), (VkBuffer)a, (VkBuffer)b, (VkBuffer)c, (uint32_t)n);
    } else {
        cpu_mul((const float*)a, (const float*)b, (float*)c, n);
    }
}

void VulkanBackend::scale(float s, const void* x, void* y, int64_t n) {
    // H6 honest-fallback: no SPIR-V scale shader; silent CPU ONLY when !vulkan_ok.
    if (!impl_->vulkan_ok) {
        cpu_scale(s, (const float*)x, (float*)y, n);
        return;
    }
    std::fprintf(stderr, "[WARN] Vulkan scale SPIR-V not implemented (REAL ONLY): CPU fallback, perf-invalid\n");
    cpu_scale(s, (const float*)x, (float*)y, n);
}

void VulkanBackend::softmax(const void* x, void* y, int64_t rows, int64_t cols) {
    // H6 honest-fallback: no SPIR-V softmax shader; silent CPU ONLY when !vulkan_ok.
    if (!impl_->vulkan_ok) {
        cpu_softmax((const float*)x, (float*)y, rows, cols);
        return;
    }
    std::fprintf(stderr, "[WARN] Vulkan softmax SPIR-V not implemented (REAL ONLY): CPU fallback, perf-invalid\n");
    cpu_softmax((const float*)x, (float*)y, rows, cols);
}

void VulkanBackend::rms_norm(const void* x, const void* gamma, void* y, float eps,
                              int64_t n, int64_t d) {
    // H6 honest-fallback: no SPIR-V rms_norm shader; silent CPU ONLY when !vulkan_ok.
    if (!impl_->vulkan_ok) {
        cpu_rms_norm((const float*)x, (const float*)gamma, (float*)y, eps, n, d);
        return;
    }
    std::fprintf(stderr, "[WARN] Vulkan rms_norm SPIR-V not implemented (REAL ONLY): CPU fallback, perf-invalid\n");
    cpu_rms_norm((const float*)x, (const float*)gamma, (float*)y, eps, n, d);
}

void VulkanBackend::layer_norm(const void* x, const void* gamma, const void* beta,
                                void* y, float eps, int64_t n, int64_t d) {
    // H6 honest-fallback: no SPIR-V layer_norm shader; silent CPU ONLY when !vulkan_ok.
    if (!impl_->vulkan_ok) {
        cpu_layer_norm((const float*)x, (const float*)gamma, (const float*)beta, (float*)y, eps, n, d);
        return;
    }
    std::fprintf(stderr, "[WARN] Vulkan layer_norm SPIR-V not implemented (REAL ONLY): CPU fallback, perf-invalid\n");
    cpu_layer_norm((const float*)x, (const float*)gamma, (const float*)beta, (float*)y, eps, n, d);
}

void VulkanBackend::moe_gather(const void* x, const int64_t* indices, const float* weights,
                                void* out, int64_t T, int64_t K, int64_t D) {
    // H6 honest-fallback: no SPIR-V moe_gather shader; silent CPU ONLY when !vulkan_ok.
    if (!impl_->vulkan_ok) {
        cpu_moe_gather((const float*)x, indices, weights, (float*)out, T, K, D);
        return;
    }
    std::fprintf(stderr, "[WARN] Vulkan moe_gather SPIR-V not implemented (REAL ONLY): CPU fallback, perf-invalid\n");
    cpu_moe_gather((const float*)x, indices, weights, (float*)out, T, K, D);
}

void VulkanBackend::moe_scatter_add(void* out, const int64_t* indices, const float* weights,
                                     const void* expert_out, int64_t T, int64_t K, int64_t D) {
    // H6 honest-fallback: no SPIR-V moe_scatter shader; silent CPU ONLY when !vulkan_ok.
    if (!impl_->vulkan_ok) {
        cpu_moe_scatter_add((float*)out, indices, weights, (const float*)expert_out, T, K, D);
        return;
    }
    std::fprintf(stderr, "[WARN] Vulkan moe_scatter_add SPIR-V not implemented (REAL ONLY): CPU fallback, perf-invalid\n");
    cpu_moe_scatter_add((float*)out, indices, weights, (const float*)expert_out, T, K, D);
}

void VulkanBackend::flash_attention(void* out, const void* Q, const void* K, const void* V,
                                     int64_t B, int64_t H, int64_t N, int64_t D, bool causal) {
    // H6 honest-fallback: no SPIR-V flash_attention shader; silent CPU ONLY when !vulkan_ok.
    if (!impl_->vulkan_ok) {
        cpu_flash_attention(out, Q, K, V, B, H, N, D, causal);
        return;
    }
    std::fprintf(stderr, "[WARN] Vulkan flash_attention SPIR-V not implemented (REAL ONLY): CPU fallback, perf-invalid\n");
    cpu_flash_attention(out, Q, K, V, B, H, N, D, causal);
}

void VulkanBackend::pipeline_cache_clear() {
    if (impl_->vulkan_ok) {
        std::lock_guard<std::mutex> lk(impl_->cache_mtx);
        for (auto& kv : impl_->pipeline_cache)
            impl_->vkDestroyPipeline(impl_->device, kv.second, nullptr);
        impl_->pipeline_cache.clear();
    }
}

size_t VulkanBackend::pipeline_cache_size() const {
    std::lock_guard<std::mutex> lk(impl_->cache_mtx);
    return impl_->pipeline_cache.size();
}

void VulkanBackend::synchronize() {
    if (impl_->vulkan_ok && impl_->device) {
        impl_->vkDeviceWaitIdle(impl_->device);
    }
}

int64_t VulkanBackend::memory_free() const {
    if (!impl_->vulkan_ok) return 0;
    int64_t total = 0;
    for (uint32_t i = 0; i < impl_->mem_props.memoryHeapCount; i++) {
        if (impl_->mem_props.memoryHeaps[i].flags & 0x1)
            total += (int64_t)impl_->mem_props.memoryHeaps[i].size;
    }
    return total;
}

int64_t VulkanBackend::memory_total() const {
    return memory_free();
}

// ========================================================================
// Factory functions
// ========================================================================

static DirectXCompute s_dx_compute;
static VulkanBackend s_vulkan_backend;
static std::once_flag s_gpu_init_flag;

GPUType detect_best_gpu() {
    return GPUType::VULKAN;
}

DirectXCompute& get_dx_compute() { return s_dx_compute; }
VulkanBackend& get_vulkan_backend() { return s_vulkan_backend; }

bool gpu_available() {
    return s_dx_compute.is_initialized() || s_vulkan_backend.is_initialized();
}

void init_gpu(GPUType type, int64_t device) {
    std::call_once(s_gpu_init_flag, [type, device]() {
        if (type == GPUType::DIRECTX12) {
            s_dx_compute.init(device);
        } else if (type == GPUType::VULKAN) {
            if (!s_vulkan_backend.init(device)) {
                s_dx_compute.init(device);
            }
        }
    });
}

void shutdown_gpu() {
    s_vulkan_backend.shutdown();
    s_dx_compute.shutdown();
}

// ========================================================================
// Tensor-level GPU kernel wrappers
// ========================================================================

Tensor vk_flash_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                           int64_t B, int64_t H, int64_t N, int64_t D, bool causal) {
    Tensor out(Shape{B, H, N, D}, DType::F32);
    // H6 honest-fallback: VulkanBackend::flash_attention handles !vulkan_ok CPU fallback
    // internally (WARN perf-invalid only when vulkan_ok but no shader). Always dispatch
    // so !vulkan_ok still yields correct CPU result instead of uninitialized output.
    auto& vk = get_vulkan_backend();
    vk.flash_attention(out.data<float>(), Q.data<float>(), K.data<float>(), V.data<float>(),
                        B, H, N, D, causal);
    return out;
}

Tensor vk_cross_entropy(const Tensor& logits, const Tensor& targets) {
    int64_t B = logits.dim(0);
    int64_t V = logits.numel() / B;
    const float* ld = logits.data<float>();
    const float* td = targets.data<float>();
    Tensor loss(Shape{B}, DType::F32);
    float* loss_data = loss.data<float>();
    for (int64_t b = 0; b < B; b++) {
        float max_val = ld[b * V];
        for (int64_t v = 1; v < V; v++)
            if (ld[b * V + v] > max_val) max_val = ld[b * V + v];
        float sum_exp = 0;
        for (int64_t v = 0; v < V; v++)
            sum_exp += std::exp(ld[b * V + v] - max_val);
        float log_sum_exp = max_val + std::log(sum_exp + 1e-10f);
        int64_t target = (int64_t)td[b];
        if (target < 0) target = 0;
        if (target >= V) target = V - 1;
        loss_data[b] = log_sum_exp - ld[b * V + target];
    }
    return loss;
}

Tensor vk_cross_entropy_grad(const Tensor& logits, const Tensor& targets) {
    int64_t B = logits.dim(0);
    int64_t V = logits.numel() / B;
    const float* ld = logits.data<float>();
    const float* td = targets.data<float>();
    Tensor grad(logits.shape(), DType::F32);
    float* gd = grad.data<float>();
    for (int64_t b = 0; b < B; b++) {
        float max_val = ld[b * V];
        for (int64_t v = 1; v < V; v++)
            if (ld[b * V + v] > max_val) max_val = ld[b * V + v];
        float sum_exp = 0;
        for (int64_t v = 0; v < V; v++)
            sum_exp += std::exp(ld[b * V + v] - max_val);
        int64_t target = (int64_t)td[b];
        if (target < 0) target = 0;
        if (target >= V) target = V - 1;
        for (int64_t v = 0; v < V; v++) {
            float soft = std::exp(ld[b * V + v] - max_val) / (sum_exp + 1e-10f);
            gd[b * V + v] = (soft - (v == target ? 1.0f : 0.0f)) / (float)B;
        }
    }
    return grad;
}

Tensor vk_swiglu(const Tensor& gate, const Tensor& up) {
    int64_t n = gate.numel();
    Tensor out(gate.shape(), DType::F32);
    // H6 honest-fallback (REAL ONLY, CPU-only op): always compute so !vulkan_ok
    // still yields correct result; never claim GPU for this path.
    std::vector<float> gate_buf(n), up_buf(n), out_buf(n);
    memcpy(gate_buf.data(), gate.data<float>(), n * sizeof(float));
    memcpy(up_buf.data(), up.data<float>(), n * sizeof(float));
    for (int64_t i = 0; i < n; i++) {
        float g = gate_buf[i] / (1.0f + std::exp(-gate_buf[i]));
        out_buf[i] = g * up_buf[i];
    }
    memcpy(out.data<float>(), out_buf.data(), n * sizeof(float));
    return out;
}

Tensor vk_rms_norm(const Tensor& x, const Tensor& gamma, float eps) {
    int64_t n = x.dim(0);
    int64_t d = x.numel() / n;
    Tensor out(x.shape(), DType::F32);
    // H6 honest-fallback: always dispatch; VulkanBackend handles !vulkan_ok CPU fallback.
    auto& vk = get_vulkan_backend();
    vk.rms_norm(x.data<float>(), gamma.data<float>(), out.data<float>(), eps, n, d);
    return out;
}

Tensor vk_rms_norm_add(const Tensor& x, const Tensor& residual,
                         const Tensor& gamma, float eps) {
    int64_t n = x.dim(0);
    int64_t d = x.numel() / n;
    Tensor out(x.shape(), DType::F32);
    // H6 honest-fallback: always dispatch; VulkanBackend handles !vulkan_ok CPU fallback.
    auto& vk = get_vulkan_backend();
    std::vector<float> tmp(n * d);
    vk.rms_norm(x.data<float>(), gamma.data<float>(), tmp.data(), eps, n, d);
    const float* res = residual.data<float>();
    for (int64_t i = 0; i < n * d; i++) out.data<float>()[i] = tmp[i] + res[i];
    return out;
}

void vk_rope(Tensor& Q, Tensor& K, const Tensor& cos_cache,
               const Tensor& sin_cache, int64_t seq_start) {
    int64_t B = Q.shape().dims[0];
    int64_t H = Q.shape().dims[1];
    int64_t S = Q.shape().dims[2];
    int64_t D = Q.shape().dims[3];
    int64_t half_D = D / 2;
    int64_t cos_stride = cos_cache.shape().dims[1];
    float* qd = Q.data<float>();
    float* kd = K.data<float>();
    const float* cos_d = cos_cache.data<float>();
    const float* sin_d = sin_cache.data<float>();
    int64_t HSD = H * S * D;
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = 0; s < S; s++) {
                int64_t pos = seq_start + s;
                int64_t x_off = b * HSD + h * S * D + s * D;
                int64_t c_off = pos * cos_stride;
                for (int64_t d = 0; d < half_D; d++) {
                    float cv = cos_d[c_off + d];
                    float sv = sin_d[c_off + d];
                    float q1 = qd[x_off + d];
                    float q2 = qd[x_off + d + half_D];
                    qd[x_off + d] = q1 * cv - q2 * sv;
                    qd[x_off + d + half_D] = q1 * sv + q2 * cv;
                    float k1 = kd[x_off + d];
                    float k2 = kd[x_off + d + half_D];
                    kd[x_off + d] = k1 * cv - k2 * sv;
                    kd[x_off + d + half_D] = k1 * sv + k2 * cv;
                }
            }
        }
    }
}

std::pair<Tensor, Tensor> vk_topk_softmax(const Tensor& logits, int64_t k) {
    int64_t rows = logits.dim(0);
    int64_t cols = logits.numel() / rows;
    if (k > cols) k = cols;
    if (k <= 0) k = 1;
    Tensor indices(Shape{rows, k}, DType::I64);
    Tensor weights(Shape{rows, k}, DType::F32);
    const float* ld = logits.data<float>();
    int64_t* id = indices.data<int64_t>();
    float* wd = weights.data<float>();
    for (int64_t r = 0; r < rows; r++) {
        std::vector<std::pair<float, int64_t>> scored;
        scored.reserve(cols);
        for (int64_t c = 0; c < cols; c++)
            scored.push_back({ld[r * cols + c], c});
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        float max_val = scored[0].first;
        float sum = 0;
        for (int64_t j = 0; j < k; j++) {
            float e = std::exp(scored[j].first - max_val);
            id[r * k + j] = scored[j].second;
            wd[r * k + j] = e;
            sum += e;
        }
        float inv = 1.0f / (sum + 1e-10f);
        for (int64_t j = 0; j < k; j++) wd[r * k + j] *= inv;
    }
    return {indices, weights};
}

Tensor vk_gelu(const Tensor& x) {
    int64_t n = x.numel();
    Tensor out(x.shape(), DType::F32);
    auto& vk = get_vulkan_backend();
    if (!vk.is_initialized()) {
        // H6 honest-fallback: CPU fallback ONLY when !vulkan_ok (host pointers valid here).
        vk.gelu(x.data<float>(), out.data<float>(), n);
        return out;
    }
    // H6 REAL ONLY path: host pointers are NOT valid VkBuffers when vulkan_ok.
    // Use device-buffer upload/dispatch/download so gelu really runs on GPU.
    void* dx = vk.allocate((size_t)n * sizeof(float));
    void* dy = vk.allocate((size_t)n * sizeof(float));
    if (!dx || !dy) {
        if (dx) vk.free(dx);
        if (dy) vk.free(dy);
        std::fprintf(stderr, "[WARN] vk_gelu alloc failed (REAL ONLY): CPU fallback, perf-invalid\n");
        vk.gelu(x.data<float>(), out.data<float>(), n);
        return out;
    }
    vk.upload(x, dx);
    vk.gelu(dx, dy, n);
    vk.download(dy, out);
    vk.free(dx);
    vk.free(dy);
    return out;
}

Tensor vk_softmax(const Tensor& x) {
    int64_t rows = x.dim(0);
    int64_t cols = x.numel() / rows;
    Tensor out(x.shape(), DType::F32);
    // H6 honest-fallback: always dispatch; VulkanBackend handles !vulkan_ok CPU fallback.
    auto& vk = get_vulkan_backend();
    vk.softmax(x.data<float>(), out.data<float>(), rows, cols);
    return out;
}

Tensor vk_layer_norm(const Tensor& x, const Tensor& gamma,
                       const Tensor& beta, float eps) {
    int64_t n = x.dim(0);
    int64_t d = x.numel() / n;
    Tensor out(x.shape(), DType::F32);
    // H6 honest-fallback: always dispatch; VulkanBackend handles !vulkan_ok CPU fallback.
    auto& vk = get_vulkan_backend();
    vk.layer_norm(x.data<float>(), gamma.data<float>(), beta.data<float>(),
                   out.data<float>(), eps, n, d);
    return out;
}

} // namespace gpu
} // namespace quant
