#include "quant/igpu_zero_copy.h"
#include "quant/tensor.h"
#include "quant/vulkan_types.h"
#include <cstdint>
#include <cstring>
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

using PFN_vkGetInstanceProcAddr = PFN_vkVoidFunction(*)(VkInstance, const char*);
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

// Fence wait budget. Was UINT64_MAX: a failed submit or non-completing shader
// meant the fence never signalled and the process stalled forever. Fail loud
// and bail instead (see gpu_compute_vulkan.cpp for the observed hang).
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

struct IGPUZeroCopyAllocator::Impl {
    bool vulkan_ok = false;
    bool cpu_fallback = false;
    IGPUDeviceInfo info;

    void* vk_lib = nullptr;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_fn = nullptr;

    VkInstance instance = nullptr;
    VkPhysicalDevice phys_dev = nullptr;
    VkDevice device = nullptr;
    VkQueue compute_queue = nullptr;
    uint32_t compute_queue_family_idx = 0;
    VkCommandPool cmd_pool = nullptr;

    VkPhysicalDeviceMemoryProperties mem_props = {};

    VkDescriptorSetLayout ds_layout = nullptr;
    VkPipelineLayout pipeline_layout = nullptr;
    VkDescriptorPool desc_pool = nullptr;

    std::unordered_map<std::string, VkPipeline> pipeline_cache;
    std::mutex cache_mtx;

    struct UnifiedAllocation {
        VkBuffer buffer = nullptr;
        VkDeviceMemory memory = nullptr;
        void* mapped = nullptr;
        size_t size = 0;
    };
    std::vector<UnifiedAllocation> allocations;
    std::mutex alloc_mtx;

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
        *out = (void*)vkGetInstanceProcAddr_fn(instance, name);
    }
    void load_dev_func(const char* name, void** out) {
        *out = (void*)vkGetInstanceProcAddr_fn((VkInstance)device, name);
    }

    bool init_vulkan_igpu(int64_t device_id) {
        if (!load_vulkan_library()) return false;

        auto createInstance_fn = (PFN_vkCreateInstance)vkGetInstanceProcAddr_fn(nullptr, "vkCreateInstance");
        if (!createInstance_fn) return false;

        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Transcender iGPU Zero-Copy";
        appInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);

        VkInstanceCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &appInfo;

        VkResult r = createInstance_fn(&ci, nullptr, &instance);
        if (r != VK_SUCCESS || !instance) return false;

        load_func("vkDestroyInstance", (void**)&vkDestroyInstance);
        load_func("vkEnumeratePhysicalDevices", (void**)&vkEnumeratePhysicalDevices);
        load_func("vkGetPhysicalDeviceProperties", (void**)&vkGetPhysicalDeviceProperties);
        load_func("vkGetPhysicalDeviceMemoryProperties", (void**)&vkGetPhysicalDeviceMemoryProperties);
        load_func("vkCreateDevice", (void**)&vkCreateDevice);
        load_func("vkGetPhysicalDeviceQueueFamilyProperties", (void**)&vkGetPhysicalDeviceQueueFamilyProperties);

        uint32_t dev_count = 0;
        vkEnumeratePhysicalDevices(instance, &dev_count, nullptr);
        if (dev_count == 0) return false;
        std::vector<VkPhysicalDevice> devs(dev_count);
        vkEnumeratePhysicalDevices(instance, &dev_count, devs.data());

        int igpu_idx = -1;
        int any_idx = 0;
        for (uint32_t i = 0; i < dev_count; i++) {
            char buf[sizeof(VkPhysicalDeviceProperties)] = {};
            ((uint32_t*)buf)[0] = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES;
            vkGetPhysicalDeviceProperties(devs[i], buf);
            VkPhysicalDeviceType dt = (VkPhysicalDeviceType)((uint32_t*)buf)[7];
            if (dt == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                igpu_idx = (int)i;
                break;
            }
            if (igpu_idx < 0) any_idx = (int)i;
        }

        if (igpu_idx >= 0) {
            phys_dev = devs[igpu_idx];
        } else {
            phys_dev = devs[any_idx];
        }

        {
            char buf[sizeof(VkPhysicalDeviceProperties)] = {};
            ((uint32_t*)buf)[0] = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES;
            vkGetPhysicalDeviceProperties(phys_dev, buf);
            info.device_name = ((const VkPhysicalDeviceProperties*)buf)->deviceName;
            info.device_type = ((const VkPhysicalDeviceProperties*)buf)->deviceType;
            if (info.device_type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                info.vendor = "Integrated GPU";
            } else {
                info.vendor = "Discrete GPU";
            }
        }

        vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem_props);

        info.dedicated_vram = 0;
        info.shared_memory = 0;
        for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
            if (mem_props.memoryHeaps[i].flags & 0x1) {
                info.dedicated_vram += mem_props.memoryHeaps[i].size;
            } else {
                info.shared_memory += mem_props.memoryHeaps[i].size;
            }
        }

        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qf_count, nullptr);
        std::vector<char> qf_props(qf_count * 48);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qf_count, qf_props.data());

        compute_queue_family_idx = 0;
        bool found = false;
        for (uint32_t i = 0; i < qf_count; i++) {
            uint32_t flags = ((uint32_t*)qf_props.data())[i * 12 + 2];
            if (flags & 0x2) {
                compute_queue_family_idx = i;
                found = true;
                break;
            }
        }
        if (!found) return false;
        info.compute_queue_family = compute_queue_family_idx;

        float priority = 1.0f;
        VkDeviceQueueCreateInfo dqi = {};
        dqi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        dqi.queueFamilyIndex = compute_queue_family_idx;
        dqi.queueCount = 1;
        dqi.pQueuePriorities = &priority;

        VkDeviceCreateInfo devCI = {};
        devCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        devCI.queueCreateInfoCount = 1;
        devCI.pQueueCreateInfos = &dqi;

        r = vkCreateDevice(phys_dev, &devCI, nullptr, &device);
        if (r != VK_SUCCESS || !device) return false;

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

        vkGetDeviceQueue(device, compute_queue_family_idx, 0, &compute_queue);

        VkCommandPoolCreateInfo cpi = {};
        cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpi.flags = 0x2;
        cpi.queueFamilyIndex = compute_queue_family_idx;
        vkCreateCommandPool(device, &cpi, nullptr, &cmd_pool);

        VkDescriptorSetLayoutBinding binding = {};
        binding.binding = 0;
        binding.descriptorType = 7;
        binding.descriptorCount = 1;
        binding.stageFlags = 0x20;

        VkDescriptorSetLayoutCreateInfo dslci = {};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 1;
        dslci.pBindings = &binding;
        vkCreateDescriptorSetLayout(device, &dslci, nullptr, &ds_layout);

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
        vkCreatePipelineLayout(device, &plci, nullptr, &pipeline_layout);

        VkDescriptorPoolSize dps = {7, 256};
        VkDescriptorPoolCreateInfo dpci = {};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.flags = 0x2;
        dpci.maxSets = 256;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &dps;
        vkCreateDescriptorPool(device, &dpci, nullptr, &desc_pool);

        vulkan_ok = true;
        return true;
    }

    uint32_t find_memory_type(uint32_t type_bits, VkFlags props) {
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
            if ((type_bits & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & props))
                return i;
        }
        return 0;
    }

    VkPipeline get_pipeline(const uint32_t* spirv, size_t spirv_size) {
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
        ssi.sType = 0x00000005;
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
        if (push_data && push_size > 0)
            vkCmdPushConstants(cmd, pipeline_layout, 0x20, 0, push_size, push_data);
        vkCmdDispatch(cmd, gx, gy, gz);

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
        vkDestroyDescriptorPool(device, desc_pool, nullptr);

        VkDescriptorPoolSize dps2 = {7, 256};
        VkDescriptorPoolCreateInfo dpci2 = {};
        dpci2.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci2.flags = 0x2;
        dpci2.maxSets = 256;
        dpci2.poolSizeCount = 1;
        dpci2.pPoolSizes = &dps2;
        vkCreateDescriptorPool(device, &dpci2, nullptr, &desc_pool);
    }

    void dispatch_simple(const uint32_t* spirv, size_t spirv_sz,
                          VkBuffer in1, VkBuffer out, uint32_t n) {
        VkPipeline pl = get_pipeline(spirv, spirv_sz);
        if (!pl) return;
        uint32_t push[8] = {n, 0, 0, 0, 0, 0, 0, 0};
        VkBuffer bufs[] = {out, in1};
        uint32_t gx = (n + 255) / 256;
        dispatch_compute(pl, push, 32, bufs, 2, gx, 1, 1);
    }

    void dispatch_binary(const uint32_t* spirv, size_t spirv_sz,
                          VkBuffer in1, VkBuffer in2, VkBuffer out, uint32_t n) {
        VkPipeline pl = get_pipeline(spirv, spirv_sz);
        if (!pl) return;
        uint32_t push[8] = {n, 0, 0, 0, 0, 0, 0, 0};
        VkBuffer bufs[] = {out, in1, in2};
        uint32_t gx = (n + 255) / 256;
        dispatch_compute(pl, push, 32, bufs, 3, gx, 1, 1);
    }

    bool has_unified_memory() {
        if (!vulkan_ok) return false;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
            VkFlags props = mem_props.memoryTypes[i].propertyFlags;
            if ((props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
                (props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                return true;
            }
        }
        return false;
    }

    ~Impl() { shutdown(); }

    void shutdown() {
        if (!device) return;

        {
            std::lock_guard<std::mutex> lk(cache_mtx);
            for (auto& kv : pipeline_cache)
                vkDestroyPipeline(device, kv.second, nullptr);
            pipeline_cache.clear();
        }

        {
            std::lock_guard<std::mutex> lk(alloc_mtx);
            for (auto& a : allocations) {
                if (a.mapped) vkUnmapMemory(device, a.memory);
                if (a.buffer) vkDestroyBuffer(device, a.buffer, nullptr);
                if (a.memory) vkFreeMemory(device, a.memory, nullptr);
            }
            allocations.clear();
        }

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

IGPUZeroCopyAllocator::IGPUZeroCopyAllocator() : impl_(new Impl()) {}
IGPUZeroCopyAllocator::~IGPUZeroCopyAllocator() { delete impl_; }

bool IGPUZeroCopyAllocator::init(int64_t device_id) {
    if (impl_->vulkan_ok) return true;
    if (impl_->init_vulkan_igpu(device_id)) {
        if (!impl_->has_unified_memory()) {
            impl_->cpu_fallback = true;
        }
        return true;
    }
    impl_->cpu_fallback = true;
    return false;
}

void IGPUZeroCopyAllocator::shutdown() { impl_->shutdown(); }

bool IGPUZeroCopyAllocator::is_initialized() const { return impl_->vulkan_ok; }

const IGPUDeviceInfo& IGPUZeroCopyAllocator::device_info() const { return impl_->info; }

void* IGPUZeroCopyAllocator::allocate_unified(size_t bytes) {
    if (!impl_->vulkan_ok || bytes == 0) return nullptr;

    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = nullptr;
    if (impl_->vkCreateBuffer(impl_->device, &bci, nullptr, &buffer) != VK_SUCCESS) return nullptr;

    VkMemoryRequirements mr = {};
    impl_->vkGetBufferMemoryRequirements(impl_->device, buffer, &mr);

    VkFlags unified_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    uint32_t mem_type = impl_->find_memory_type(mr.memoryTypeBits, unified_props);
    if (mem_type == 0) {
        VkFlags fallback_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        mem_type = impl_->find_memory_type(mr.memoryTypeBits, fallback_props);
    }

    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = mem_type;

    VkDeviceMemory memory = nullptr;
    if (impl_->vkAllocateMemory(impl_->device, &mai, nullptr, &memory) != VK_SUCCESS) {
        impl_->vkDestroyBuffer(impl_->device, buffer, nullptr);
        return nullptr;
    }

    impl_->vkBindBufferMemory(impl_->device, buffer, memory, 0);

    void* mapped = nullptr;
    impl_->vkMapMemory(impl_->device, memory, 0, bytes, 0, &mapped);
    if (!mapped) {
        impl_->vkDestroyBuffer(impl_->device, buffer, nullptr);
        impl_->vkFreeMemory(impl_->device, memory, nullptr);
        return nullptr;
    }

    std::lock_guard<std::mutex> lk(impl_->alloc_mtx);
    impl_->allocations.push_back({buffer, memory, mapped, bytes});
    return mapped;
}

void IGPUZeroCopyAllocator::free_unified(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lk(impl_->alloc_mtx);
    for (size_t i = 0; i < impl_->allocations.size(); i++) {
        if (impl_->allocations[i].mapped == ptr) {
            auto& a = impl_->allocations[i];
            if (a.mapped) impl_->vkUnmapMemory(impl_->device, a.memory);
            if (a.buffer) impl_->vkDestroyBuffer(impl_->device, a.buffer, nullptr);
            if (a.memory) impl_->vkFreeMemory(impl_->device, a.memory, nullptr);
            impl_->allocations.erase(impl_->allocations.begin() + (int)i);
            return;
        }
    }
}

bool IGPUZeroCopyAllocator::upload_direct(const void* src, void* dst, size_t bytes) {
    if (!src || !dst || bytes == 0) return false;
    if (impl_->cpu_fallback) {
        memcpy(dst, src, bytes);
        return true;
    }
    memcpy(dst, src, bytes);
    return true;
}

bool IGPUZeroCopyAllocator::download_direct(void* src, void* dst, size_t bytes) {
    if (!src || !dst || bytes == 0) return false;
    if (impl_->cpu_fallback) {
        memcpy(dst, src, bytes);
        return true;
    }
    memcpy(dst, src, bytes);
    return true;
}

size_t IGPUZeroCopyAllocator::available_unified() const {
    if (!impl_->vulkan_ok) return 0;
    size_t total = 0;
    for (uint32_t i = 0; i < impl_->mem_props.memoryHeapCount; i++) {
        if (impl_->mem_props.memoryHeaps[i].flags & 0x1)
            total += impl_->mem_props.memoryHeaps[i].size;
    }
    return total;
}

size_t IGPUZeroCopyAllocator::total_unified() const { return available_unified(); }

static void cpu_zc_gemm(const float* A, const float* B, float* C,
                         int64_t M, int64_t N, int64_t K) {
    for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++) {
            float s = 0;
            for (int64_t k = 0; k < K; k++) s += A[m * K + k] * B[k * N + n];
            C[m * N + n] = s;
        }
}

static void cpu_zc_softmax(const float* x, float* y, int64_t rows, int64_t cols) {
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

static void cpu_zc_rms_norm(const float* x, const float* gamma, float* y,
                             float eps, int64_t n, int64_t d) {
    for (int64_t i = 0; i < n; i++) {
        float ss = 0;
        for (int64_t j = 0; j < d; j++) ss += x[i * d + j] * x[i * d + j];
        float rs = 1.0f / std::sqrt(ss / d + eps);
        for (int64_t j = 0; j < d; j++) y[i * d + j] = x[i * d + j] * rs * gamma[j];
    }
}

static void cpu_zc_silu(const float* x, float* y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v / (1.0f + std::exp(-v));
    }
}

static float cpu_zc_cross_entropy(const float* logits, const int64_t* targets,
                                   int64_t B, int64_t V) {
    float total = 0;
    for (int64_t b = 0; b < B; b++) {
        float mx = logits[b * V];
        for (int64_t v = 1; v < V; v++) mx = std::max(mx, logits[b * V + v]);
        float sum = 0;
        for (int64_t v = 0; v < V; v++) sum += std::exp(logits[b * V + v] - mx);
        float log_sum = std::log(sum + 1e-10f) + mx;
        total += log_sum - logits[b * V + targets[b]];
    }
    return total / B;
}

struct IGPUZeroCopyTrainer::Impl {
    IGPUZeroCopyAllocator allocator;
    bool available = false;

    bool init() {
        if (!allocator.init()) return false;
        available = allocator.is_initialized();
        return available;
    }

    Tensor forward_linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
        int64_t M = input.dim(0);
        int64_t K = input.numel() / M;
        int64_t N = weight.numel() / K;
        Tensor out(Shape{M, N}, DType::F32);

        if (available) {
            void* in_zc = allocator.allocate_unified(input.size_bytes());
            void* w_zc = allocator.allocate_unified(weight.size_bytes());
            void* out_zc = allocator.allocate_unified(out.size_bytes());
            void* b_zc = allocator.allocate_unified(bias.size_bytes());

            if (in_zc && w_zc && out_zc && b_zc) {
                allocator.upload_direct(input.data(), in_zc, input.size_bytes());
                allocator.upload_direct(weight.data(), w_zc, weight.size_bytes());
                allocator.upload_direct(bias.data(), b_zc, bias.size_bytes());

                cpu_zc_gemm((const float*)in_zc, (const float*)w_zc, (float*)out_zc, M, N, K);

                const float* bd = (const float*)b_zc;
                float* od = (float*)out_zc;
                for (int64_t m = 0; m < M; m++)
                    for (int64_t n = 0; n < N; n++)
                        od[m * N + n] += bd[n];

                allocator.download_direct(out_zc, out.data(), out.size_bytes());

                allocator.free_unified(in_zc);
                allocator.free_unified(w_zc);
                allocator.free_unified(out_zc);
                allocator.free_unified(b_zc);
            } else {
                cpu_zc_gemm(input.data<float>(), weight.data<float>(), out.data<float>(), M, N, K);
                const float* bd = bias.data<float>();
                float* od = out.data<float>();
                for (int64_t m = 0; m < M; m++)
                    for (int64_t n = 0; n < N; n++)
                        od[m * N + n] += bd[n];
                if (in_zc) allocator.free_unified(in_zc);
                if (w_zc) allocator.free_unified(w_zc);
                if (out_zc) allocator.free_unified(out_zc);
                if (b_zc) allocator.free_unified(b_zc);
            }
        } else {
            cpu_zc_gemm(input.data<float>(), weight.data<float>(), out.data<float>(), M, N, K);
            const float* bd = bias.data<float>();
            float* od = out.data<float>();
            for (int64_t m = 0; m < M; m++)
                for (int64_t n = 0; n < N; n++)
                    od[m * N + n] += bd[n];
        }
        return out;
    }

    Tensor forward_attention(const Tensor& Q, const Tensor& K, const Tensor& V, bool causal) {
        int64_t B = Q.dim(0);
        int64_t H = Q.dim(1);
        int64_t N = Q.dim(2);
        int64_t D = Q.dim(3);
        Tensor out(Shape{B, H, N, D}, DType::F32);
        float scale = 1.0f / std::sqrt((float)D);

        const float* Qd = Q.data<float>();
        const float* Kd = K.data<float>();
        const float* Vd = V.data<float>();
        float* Od = out.data<float>();

        for (int64_t b = 0; b < B; b++) {
            for (int64_t h = 0; h < H; h++) {
                int64_t off = b * H * N * D + h * N * D;
                for (int64_t qi = 0; qi < N; qi++) {
                    float max_val = -1e30f;
                    for (int64_t ki = 0; ki <= (causal ? qi : N - 1); ki++) {
                        float dot = 0;
                        for (int64_t d = 0; d < D; d++)
                            dot += Qd[off + qi * D + d] * Kd[off + ki * D + d];
                        dot *= scale;
                        if (dot > max_val) max_val = dot;
                    }
                    float sum_exp = 0;
                    for (int64_t d = 0; d < D; d++) Od[off + qi * D + d] = 0;
                    for (int64_t ki = 0; ki <= (causal ? qi : N - 1); ki++) {
                        float dot = 0;
                        for (int64_t d = 0; d < D; d++)
                            dot += Qd[off + qi * D + d] * Kd[off + ki * D + d];
                        dot *= scale;
                        float w = std::exp(dot - max_val);
                        sum_exp += w;
                        for (int64_t d = 0; d < D; d++)
                            Od[off + qi * D + d] += w * Vd[off + ki * D + d];
                    }
                    if (sum_exp > 0)
                        for (int64_t d = 0; d < D; d++)
                            Od[off + qi * D + d] /= sum_exp;
                }
            }
        }
        return out;
    }

    Tensor forward_rms_norm(const Tensor& x, const Tensor& gamma, float eps) {
        int64_t n = x.dim(0);
        int64_t d = x.numel() / n;
        Tensor out(x.shape(), DType::F32);
        cpu_zc_rms_norm(x.data<float>(), gamma.data<float>(), out.data<float>(), eps, n, d);
        return out;
    }

    Tensor forward_swiglu(const Tensor& gate, const Tensor& up) {
        int64_t n = gate.numel();
        Tensor out(gate.shape(), DType::F32);
        const float* g = gate.data<float>();
        const float* u = up.data<float>();
        float* o = out.data<float>();
        for (int64_t i = 0; i < n; i++) {
            float s = g[i] / (1.0f + std::exp(-g[i]));
            o[i] = s * u[i];
        }
        return out;
    }

    float compute_loss(const Tensor& logits, const Tensor& targets) {
        int64_t B = logits.dim(0);
        int64_t V = logits.numel() / B;
        return cpu_zc_cross_entropy(logits.data<float>(), targets.data<int64_t>(), B, V);
    }

    bool train_step(const Tensor& input_ids, const Tensor& labels, float& loss_out) {
        loss_out = compute_loss(input_ids, labels);
        return true;
    }
};

IGPUZeroCopyTrainer::IGPUZeroCopyTrainer() : impl_(new Impl()) {}
IGPUZeroCopyTrainer::~IGPUZeroCopyTrainer() { delete impl_; }

bool IGPUZeroCopyTrainer::init() { return impl_->init(); }
bool IGPUZeroCopyTrainer::is_available() const { return impl_->available; }

Tensor IGPUZeroCopyTrainer::forward_linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    return impl_->forward_linear(input, weight, bias);
}

Tensor IGPUZeroCopyTrainer::forward_attention(const Tensor& Q, const Tensor& K, const Tensor& V, bool causal) {
    return impl_->forward_attention(Q, K, V, causal);
}

Tensor IGPUZeroCopyTrainer::forward_rms_norm(const Tensor& x, const Tensor& gamma, float eps) {
    return impl_->forward_rms_norm(x, gamma, eps);
}

Tensor IGPUZeroCopyTrainer::forward_swiglu(const Tensor& gate, const Tensor& up) {
    return impl_->forward_swiglu(gate, up);
}

float IGPUZeroCopyTrainer::compute_loss(const Tensor& logits, const Tensor& targets) {
    return impl_->compute_loss(logits, targets);
}

bool IGPUZeroCopyTrainer::train_step(const Tensor& input_ids, const Tensor& labels, float& loss_out) {
    return impl_->train_step(input_ids, labels, loss_out);
}

void IGPUZeroCopyTrainer::synchronize() {
    if (impl_->available && impl_->allocator.is_initialized()) {
        impl_->allocator.shutdown();
        impl_->allocator.init();
    }
}

static IGPUZeroCopyTrainer s_igpu_trainer;

IGPUZeroCopyTrainer& get_igpu_trainer() { return s_igpu_trainer; }

} // namespace gpu
} // namespace quant
