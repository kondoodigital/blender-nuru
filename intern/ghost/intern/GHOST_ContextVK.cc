/* SPDX-FileCopyrightText: 2022-2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup GHOST
 */

#include "GHOST_ContextVK.hh"

#ifdef _WIN32
#  include <vulkan/vulkan_win32.h>
#elif defined(__APPLE__)
#  include <vulkan/vulkan_metal.h>
#else /* X11/WAYLAND. */
#  ifdef WITH_GHOST_X11
#    include <vulkan/vulkan_xlib.h>
#  endif
#  ifdef WITH_GHOST_WAYLAND
#    include <vulkan/vulkan_wayland.h>
#  endif
#endif

#include "vulkan/vk_ghost_api.hh"

#if !defined(_WIN32) or defined(_M_ARM64)
/* Silence compilation warning on non-windows x64 systems. */
#  define VMA_EXTERNAL_MEMORY_WIN32 0
#endif
#include "vk_mem_alloc.h"

#include "CLG_log.h"

#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include <algorithm>
#include <array>
#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <vector>

#include <sys/stat.h>

using namespace std;

static CLG_LogRef LOG = {"ghost.context"};

#define __STR(A) "" #A
#define VK_CHECK(__expression, fail_value) \
  do { \
    VkResult r = (__expression); \
    if (r != VK_SUCCESS) { \
      CLOG_ERROR(&LOG, \
                 "Vulkan: %s resulted in code %s.", \
                 __STR(__expression), \
                 blender::gpu::to_string(r)); \
      return fail_value; \
    } \
  } while (0)

/* -------------------------------------------------------------------- */
/** \name Nuru: NVIDIA Nsight Aftermath GPU crash dumps (EMERALD)
 *
 * Env-gated (`BLENDER_NVIDIA_AFTERMATH=1`) post-mortem GPU crash analysis for the Windows
 * Vulkan HWRT bring-up (async external-batch VK_ERROR_DEVICE_LOST root-cause hunt; see
 * docs/agent_handoff.md). The SDK is loaded at runtime from
 * `BLENDER_AFTERMATH_DLL`, next to the executable, or `%USERPROFILE%/.aftermath/lib/x64/`,
 * so the build has no compile- or link-time dependency on the Nsight Aftermath SDK and the
 * minimal C ABI below is declared locally (SDK 2025.5, API version 0x21A).
 *
 * On a GPU crash the driver invokes our callback with an `.nv-gpudmp` blob; we write it to
 * `%TEMP%/nuru-aftermath/`, decode it to JSON in-place, and log both paths. Open the binary
 * dump in Nsight Graphics for the full inspector.
 * \{ */

#ifdef _WIN32

namespace nuru_aftermath {

/* Local declarations of the stable Nsight Aftermath C ABI (subset). */
using AftermathResult = int32_t;
using AftermathDecoder = void *;
using PFN_AddDescription = void(__cdecl *)(uint32_t key, const char *value);
using PFN_GpuCrashDumpCb = void(__cdecl *)(const void *dump, uint32_t size, void *user);
using PFN_ShaderDebugInfoCb = void(__cdecl *)(const void *data, uint32_t size, void *user);
using PFN_DescriptionCb = void(__cdecl *)(PFN_AddDescription add, void *user);
using PFN_ResolveMarkerCb = void(__cdecl *)(
    const void *marker, uint32_t size, void *user, void **resolved, uint32_t *resolved_size);

constexpr uint32_t AFTERMATH_API_VERSION = 0x21A; /* SDK 2025.5. */
constexpr uint32_t WATCHED_API_VULKAN = 0x2;
constexpr uint32_t FEATURE_DEFER_DEBUG_INFO = 0x1;
constexpr uint32_t DECODER_ALL_INFO = 0x3FFF;
constexpr uint32_t FORMATTER_UTF8 = 0x2;
constexpr int32_t STATUS_COLLECTING_FAILED = 2;
constexpr int32_t STATUS_FINISHED = 4;

using PFN_EnableGpuCrashDumps = AftermathResult(__cdecl *)(uint32_t api_version,
                                                           uint32_t watched_apis,
                                                           uint32_t flags,
                                                           PFN_GpuCrashDumpCb dump_cb,
                                                           PFN_ShaderDebugInfoCb debug_info_cb,
                                                           PFN_DescriptionCb description_cb,
                                                           PFN_ResolveMarkerCb resolve_marker_cb,
                                                           void *user);
using PFN_GetCrashDumpStatus = AftermathResult(__cdecl *)(int32_t *r_status);
using PFN_CreateDecoder = AftermathResult(__cdecl *)(uint32_t api_version,
                                                     const void *dump,
                                                     uint32_t size,
                                                     AftermathDecoder *r_decoder);
using PFN_GenerateJSON = AftermathResult(__cdecl *)(AftermathDecoder decoder,
                                                    uint32_t decoder_flags,
                                                    uint32_t format_flags,
                                                    void *shader_debug_info_lookup_cb,
                                                    void *shader_lookup_cb,
                                                    void *shader_source_lookup_cb,
                                                    void *user,
                                                    uint32_t *r_json_size);
using PFN_GetJSON = AftermathResult(__cdecl *)(AftermathDecoder decoder,
                                               uint32_t size,
                                               char *r_json);
using PFN_DestroyDecoder = AftermathResult(__cdecl *)(AftermathDecoder decoder);

static PFN_GetCrashDumpStatus fn_get_status = nullptr;
static PFN_CreateDecoder fn_create_decoder = nullptr;
static PFN_GenerateJSON fn_generate_json = nullptr;
static PFN_GetJSON fn_get_json = nullptr;
static PFN_DestroyDecoder fn_destroy_decoder = nullptr;
static std::mutex dump_mutex;
static int dump_counter = 0;
static bool enabled = false;

/* 0 = off, 1 = full diagnostics config (resource tracking + shader error reporting +
 * automatic checkpoints), 2 = watcher only (no VK_NV_device_diagnostics_config; least
 * timing-intrusive mode for reproducing races that the full config perturbs away). */
static int config_level()
{
  const char *value = getenv("BLENDER_NVIDIA_AFTERMATH");
  if (value == nullptr || value[0] == '\0' || (value[0] == '0' && value[1] == '\0')) {
    return 0;
  }
  return (value[0] == '2' && value[1] == '\0') ? 2 : 1;
}

static bool requested()
{
  return config_level() != 0;
}

static std::string dump_directory()
{
  const char *temp = getenv("TEMP");
  std::string dir = (temp != nullptr) ? std::string(temp) : std::string(".");
  dir += "\\nuru-aftermath";
  CreateDirectoryA(dir.c_str(), nullptr);
  return dir;
}

static void write_file(const std::string &path, const void *data, const uint32_t size)
{
  FILE *file = fopen(path.c_str(), "wb");
  if (file == nullptr) {
    fprintf(stderr, "[NURU_AFTERMATH] failed to open %s for writing\n", path.c_str());
    return;
  }
  fwrite(data, 1, size, file);
  fclose(file);
}

static void __cdecl gpu_crash_dump_callback(const void *dump, const uint32_t size, void * /*user*/)
{
  std::scoped_lock lock(dump_mutex);
  const std::string dir = dump_directory();
  const int index = dump_counter++;
  const std::string base = dir + "\\blender-" + std::to_string(GetCurrentProcessId()) + "-" +
                           std::to_string(index);

  const std::string dump_path = base + ".nv-gpudmp";
  write_file(dump_path, dump, size);
  fprintf(stderr, "[NURU_AFTERMATH] GPU crash dump written: %s\n", dump_path.c_str());

  /* Decode to JSON in-place so the root cause is readable without Nsight Graphics. */
  if (fn_create_decoder && fn_generate_json && fn_get_json && fn_destroy_decoder) {
    AftermathDecoder decoder = nullptr;
    if (fn_create_decoder(AFTERMATH_API_VERSION, dump, size, &decoder) >= 0 && decoder) {
      uint32_t json_size = 0;
      if (fn_generate_json(decoder,
                           DECODER_ALL_INFO,
                           FORMATTER_UTF8,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           &json_size) >= 0 &&
          json_size > 0)
      {
        std::vector<char> json(json_size);
        if (fn_get_json(decoder, json_size, json.data()) >= 0) {
          const std::string json_path = base + ".json";
          write_file(json_path, json.data(), json_size - 1);
          fprintf(stderr, "[NURU_AFTERMATH] decoded crash dump JSON: %s\n", json_path.c_str());
        }
      }
      fn_destroy_decoder(decoder);
    }
  }
}

static void __cdecl shader_debug_info_callback(const void *data, const uint32_t size, void * /*user*/)
{
  /* Only invoked for shaders referenced by a crash dump (deferred callbacks). */
  std::scoped_lock lock(dump_mutex);
  const std::string dir = dump_directory();
  static int debug_info_counter = 0;
  const std::string path = dir + "\\shader-" + std::to_string(GetCurrentProcessId()) + "-" +
                           std::to_string(debug_info_counter++) + ".nvdbg";
  write_file(path, data, size);
  fprintf(stderr, "[NURU_AFTERMATH] shader debug info written: %s\n", path.c_str());
}

static void __cdecl description_callback(PFN_AddDescription add, void * /*user*/)
{
  /* 0x1 = ApplicationName key. */
  add(0x1, "Blender-Nuru windows-vulkan HWRT");
}

static void wait_for_pending_dump_at_exit()
{
  if (!enabled || fn_get_status == nullptr) {
    return;
  }
  /* The driver collects the dump asynchronously after a device loss; give it time to finish
   * before the process exits or the dump is lost. No-op when no crash happened. */
  int32_t status = -1;
  if (fn_get_status(&status) < 0 || status == 0) {
    return;
  }
  const ULONGLONG deadline = GetTickCount64() + 10000;
  while (status != STATUS_FINISHED && status != STATUS_COLLECTING_FAILED &&
         GetTickCount64() < deadline)
  {
    Sleep(50);
    if (fn_get_status(&status) < 0) {
      return;
    }
  }
}

static void ensure_enabled()
{
  static bool attempted = false;
  if (attempted) {
    return;
  }
  attempted = true;
  if (!requested()) {
    return;
  }

  HMODULE lib = nullptr;
  const char *override_path = getenv("BLENDER_AFTERMATH_DLL");
  if (override_path != nullptr && override_path[0] != '\0') {
    lib = LoadLibraryA(override_path);
  }
  if (lib == nullptr) {
    lib = LoadLibraryA("GFSDK_Aftermath_Lib.x64.dll");
  }
  if (lib == nullptr) {
    const char *profile = getenv("USERPROFILE");
    if (profile != nullptr) {
      const std::string sdk_path = std::string(profile) +
                                   "\\.aftermath\\lib\\x64\\GFSDK_Aftermath_Lib.x64.dll";
      lib = LoadLibraryA(sdk_path.c_str());
    }
  }
  if (lib == nullptr) {
    fprintf(stderr,
            "[NURU_AFTERMATH] BLENDER_NVIDIA_AFTERMATH is set but GFSDK_Aftermath_Lib.x64.dll "
            "was not found (set BLENDER_AFTERMATH_DLL or install to %%USERPROFILE%%\\.aftermath)\n");
    return;
  }

  const auto fn_enable = reinterpret_cast<PFN_EnableGpuCrashDumps>(
      GetProcAddress(lib, "GFSDK_Aftermath_EnableGpuCrashDumps"));
  fn_get_status = reinterpret_cast<PFN_GetCrashDumpStatus>(
      GetProcAddress(lib, "GFSDK_Aftermath_GetCrashDumpStatus"));
  fn_create_decoder = reinterpret_cast<PFN_CreateDecoder>(
      GetProcAddress(lib, "GFSDK_Aftermath_GpuCrashDump_CreateDecoder"));
  fn_generate_json = reinterpret_cast<PFN_GenerateJSON>(
      GetProcAddress(lib, "GFSDK_Aftermath_GpuCrashDump_GenerateJSON"));
  fn_get_json = reinterpret_cast<PFN_GetJSON>(
      GetProcAddress(lib, "GFSDK_Aftermath_GpuCrashDump_GetJSON"));
  fn_destroy_decoder = reinterpret_cast<PFN_DestroyDecoder>(
      GetProcAddress(lib, "GFSDK_Aftermath_GpuCrashDump_DestroyDecoder"));
  if (fn_enable == nullptr) {
    fprintf(stderr, "[NURU_AFTERMATH] GFSDK_Aftermath_EnableGpuCrashDumps export missing\n");
    return;
  }

  const AftermathResult result = fn_enable(AFTERMATH_API_VERSION,
                                           WATCHED_API_VULKAN,
                                           FEATURE_DEFER_DEBUG_INFO,
                                           gpu_crash_dump_callback,
                                           shader_debug_info_callback,
                                           description_callback,
                                           nullptr,
                                           nullptr);
  if (result < 0) {
    fprintf(stderr, "[NURU_AFTERMATH] GFSDK_Aftermath_EnableGpuCrashDumps failed: 0x%x\n",
            uint32_t(result));
    return;
  }
  enabled = true;
  atexit(wait_for_pending_dump_at_exit);
  fprintf(stderr,
          "[NURU_AFTERMATH] GPU crash dumps enabled (dumps in %s)\n",
          dump_directory().c_str());
}

static bool is_enabled()
{
  return enabled;
}

}  // namespace nuru_aftermath

#endif /* _WIN32 */

/** \} */

/* -------------------------------------------------------------------- */
/** \name Swap-chain resources
 * \{ */

void GHOST_SwapchainImage::destroy(VkDevice vk_device)
{
  vkDestroySemaphore(vk_device, present_semaphore, nullptr);
  present_semaphore = VK_NULL_HANDLE;
  vk_image = VK_NULL_HANDLE;
}

void GHOST_FrameDiscard::destroy(VkDevice vk_device)
{
  while (!swapchains.empty()) {
    VkSwapchainKHR vk_swapchain = swapchains.back();
    swapchains.pop_back();
    vkDestroySwapchainKHR(vk_device, vk_swapchain, nullptr);
  }
  while (!semaphores.empty()) {
    VkSemaphore vk_semaphore = semaphores.back();
    semaphores.pop_back();
    vkDestroySemaphore(vk_device, vk_semaphore, nullptr);
  }
}

void GHOST_Frame::destroy(VkDevice vk_device)
{
  vkDestroyFence(vk_device, submission_fence, nullptr);
  submission_fence = VK_NULL_HANDLE;
  vkDestroySemaphore(vk_device, acquire_semaphore, nullptr);
  acquire_semaphore = VK_NULL_HANDLE;
  discard_pile.destroy(vk_device);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Extension list
 * \{ */

struct GHOST_ExtensionsVK {
  blender::Vector<VkExtensionProperties> extensions;
  blender::Vector<const char *> enabled;

  bool is_supported(const char *extension_name) const
  {
    for (const VkExtensionProperties &extension : extensions) {
      if (STREQ(extension.extensionName, extension_name)) {
        return true;
      }
    }
    return false;
  }

  bool is_supported(blender::Span<const char *> extension_names)
  {
    for (const char *extension_name : extension_names) {
      if (!is_supported(extension_name)) {
        return false;
      }
    }
    return true;
  }

  bool enable(const char *extension_name, bool optional = false)
  {
    bool supported = is_supported(extension_name);
    if (supported) {
      CLOG_TRACE(&LOG,
                 "Vulkan: %s extension enabled: name=%s",
                 optional ? "optional" : "required",
                 extension_name);
      enabled.append(extension_name);
      return true;
    }

    CLOG_AT_LEVEL(&LOG,
                  (optional ? CLG_LEVEL_TRACE : CLG_LEVEL_ERROR),
                  "Vulkan: %s extension not available: name=%s",
                  optional ? "optional" : "required",
                  extension_name);

    return false;
  }

  bool disable(const char *extension_name)
  {
    bool is_extension_enabled = is_enabled(extension_name);
    if (is_extension_enabled) {
      CLOG_TRACE(
          &LOG, "Vulkan: extension disabled for compatibility reasons: name=%s", extension_name);
      enabled.remove(enabled.first_index_of(extension_name));
      return true;
    }

    return false;
  }

  bool enable(const blender::Span<const char *> &extension_names, bool optional = false)
  {
    bool failure = false;
    for (const char *extension_name : extension_names) {
      failure |= !enable(extension_name, optional);
    }
    return !failure;
  }

  bool is_enabled(const char *extension_name) const
  {
    for (const char *enabled_extension_name : enabled) {
      if (STREQ(enabled_extension_name, extension_name)) {
        return true;
      }
    }
    return false;
  }
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Vulkan Device
 * \{ */

class GHOST_DeviceVK {
 public:
  VkPhysicalDevice vk_physical_device = VK_NULL_HANDLE;
  GHOST_ExtensionsVK extensions;

  VkDevice vk_device = VK_NULL_HANDLE;

  uint32_t generic_queue_family = 0;
  VkQueue generic_queue = VK_NULL_HANDLE;
  VmaAllocator vma_allocator = VK_NULL_HANDLE;

  VkPhysicalDeviceProperties2 properties = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
  };
  VkPhysicalDeviceVulkan12Properties properties_12 = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES,
  };
  VkPhysicalDeviceFeatures2 features = {};
  VkPhysicalDeviceVulkan11Features features_11 = {};
  VkPhysicalDeviceVulkan12Features features_12 = {};
  VkPhysicalDeviceRobustness2FeaturesEXT features_robustness2 = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};

  int users = 0;

  /** Mutex to externally synchronize access to queue. */
  std::mutex queue_mutex;

  bool use_vk_ext_swapchain_maintenance_1 = false;
  bool use_vk_ext_swapchain_colorspace = false;

 public:
  GHOST_DeviceVK(VkPhysicalDevice vk_physical_device, const bool use_vk_ext_swapchain_colorspace)
      : vk_physical_device(vk_physical_device),
        use_vk_ext_swapchain_colorspace(use_vk_ext_swapchain_colorspace)
  {
    properties.pNext = &properties_12;
    vkGetPhysicalDeviceProperties2(vk_physical_device, &properties);

    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features_11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features.pNext = &features_11;
    features_11.pNext = &features_12;
    features_12.pNext = &features_robustness2;

    vkGetPhysicalDeviceFeatures2(vk_physical_device, &features);
    init_extensions();
  }

  ~GHOST_DeviceVK()
  {
    if (vma_allocator != VK_NULL_HANDLE) {
      vmaDestroyAllocator(vma_allocator);
      vma_allocator = VK_NULL_HANDLE;
    }
    if (vk_device != VK_NULL_HANDLE) {
      vkDestroyDevice(vk_device, nullptr);
      vk_device = VK_NULL_HANDLE;
    }
  }

  bool init_extensions()
  {
    uint32_t extensions_count;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(
                 vk_physical_device, nullptr, &extensions_count, nullptr),
             false);
    extensions.extensions.resize(extensions_count);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(
                 vk_physical_device, nullptr, &extensions_count, extensions.extensions.data()),
             false);
    return true;
  }

  void wait_idle()
  {
    if (vk_device) {
      std::scoped_lock lock(queue_mutex);
      vkDeviceWaitIdle(vk_device);
    }
  }

  void init_generic_queue_family()
  {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_device, &queue_family_count, nullptr);

    vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        vk_physical_device, &queue_family_count, queue_families.data());

    generic_queue_family = 0;
    for (const auto &queue_family : queue_families) {
      /* Every VULKAN implementation by spec must have one queue family that support both graphics
       * and compute pipelines. We select this one; compute only queue family hints at asynchronous
       * compute implementations. */
      if ((queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
          (queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT))
      {
        return;
      }
      generic_queue_family++;
    }
  }

  void init_generic_queue()
  {
    vkGetDeviceQueue(vk_device, generic_queue_family, 0, &generic_queue);
  }

  void init_memory_allocator(VkInstance vk_instance)
  {
    VmaAllocatorCreateInfo vma_allocator_create_info = {};
    vma_allocator_create_info.vulkanApiVersion = VK_API_VERSION_1_2;
    vma_allocator_create_info.physicalDevice = vk_physical_device;
    vma_allocator_create_info.device = vk_device;
    vma_allocator_create_info.instance = vk_instance;
    vma_allocator_create_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    if (extensions.is_enabled(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME)) {
      vma_allocator_create_info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
    }
    if (extensions.is_enabled(VK_KHR_MAINTENANCE_4_EXTENSION_NAME)) {
      vma_allocator_create_info.flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT;
    }
    vmaCreateAllocator(&vma_allocator_create_info, &vma_allocator);
  }
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Vulkan Instance
 * \{ */

struct GHOST_InstanceVK {
  VkInstance vk_instance = VK_NULL_HANDLE;
  VkPhysicalDevice vk_physical_device = VK_NULL_HANDLE;

  GHOST_ExtensionsVK extensions;

  std::optional<GHOST_DeviceVK> device;

  GHOST_InstanceVK()
  {
    init_extensions();
  }

  ~GHOST_InstanceVK()
  {
    device.reset();
    vkDestroyInstance(vk_instance, nullptr);
    vk_physical_device = VK_NULL_HANDLE;
    vk_instance = VK_NULL_HANDLE;
  }

  bool init_extensions()
  {
    uint32_t extension_count = 0;
    VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr), false);
    extensions.extensions.resize(extension_count);
    VK_CHECK(vkEnumerateInstanceExtensionProperties(
                 nullptr, &extension_count, extensions.extensions.data()),
             false);
    return true;
  }

  bool create_instance(uint32_t vulkan_api_version)
  {
#ifdef _WIN32
    /* Nuru: must be called before any Vulkan device exists to watch it for GPU crashes. */
    nuru_aftermath::ensure_enabled();
#endif
    VkApplicationInfo vk_application_info = {VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                             nullptr,
                                             "Blender",
                                             VK_MAKE_VERSION(1, 0, 0),
                                             "Blender",
                                             VK_MAKE_VERSION(1, 0, 0),
                                             vulkan_api_version};
    VkInstanceCreateInfo vk_instance_create_info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                                    nullptr,
                                                    0,
                                                    &vk_application_info,
                                                    0,
                                                    nullptr,
                                                    uint32_t(extensions.enabled.size()),
                                                    extensions.enabled.data()

    };

    VK_CHECK(vkCreateInstance(&vk_instance_create_info, nullptr, &vk_instance), false);
    return true;
  }

  bool select_physical_device(const GHOST_GPUDevice &preferred_device,
                              const blender::Span<const char *> required_extensions)
  {
    VkPhysicalDevice best_physical_device = VK_NULL_HANDLE;

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(vk_instance, &device_count, nullptr);

    vector<VkPhysicalDevice> physical_devices(device_count);
    vkEnumeratePhysicalDevices(vk_instance, &device_count, physical_devices.data());

    int best_device_score = -1;
    int device_index = -1;
    for (const auto &physical_device : physical_devices) {
      GHOST_DeviceVK device_vk(physical_device, false);
      device_index++;

      if (!device_vk.extensions.is_supported(required_extensions)) {
        continue;
      }
      if (!blender::gpu::GPU_vulkan_is_supported_driver(physical_device)) {
        continue;
      }

      if (
#ifndef __APPLE__
          !device_vk.features.features.geometryShader ||
#endif
          !device_vk.features.features.vertexPipelineStoresAndAtomics ||
          !device_vk.features.features.multiViewport ||
          !device_vk.features.features.shaderClipDistance ||
          !device_vk.features.features.fragmentStoresAndAtomics ||
          !device_vk.features.features.multiDrawIndirect ||
          !device_vk.features.features.imageCubeArray ||
          !device_vk.features.features.dualSrcBlend || !device_vk.features.features.logicOp ||
          !device_vk.features.features.imageCubeArray)
      {
        continue;
      }

      int device_score = 0;
      switch (device_vk.properties.properties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
          device_score = 400;
          break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
          device_score = 300;
          break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
          device_score = 200;
          break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
          device_score = 100;
          break;
        default:
          break;
      }

      /* User has configured a preferred device. Add bonus score when vendor and device match.
       * Driver id isn't considered as drivers update more frequently and can break the device
       * selection. */
      if (device_vk.properties.properties.deviceID == preferred_device.device_id &&
          device_vk.properties.properties.vendorID == preferred_device.vendor_id)
      {
        device_score += 500;
        if (preferred_device.index == device_index) {
          device_score += 10;
        }
      }
      if (device_score > best_device_score) {
        best_physical_device = physical_device;
        best_device_score = device_score;
      }
    }

    if (best_physical_device == VK_NULL_HANDLE) {
      CLOG_ERROR(&LOG, "No suitable Vulkan Device found!");
      return GHOST_kFailure;
    }

    vk_physical_device = best_physical_device;

    return GHOST_kSuccess;
  }

  bool create_device(const bool use_vk_ext_swapchain_maintenance1,
                     const bool is_debug,
                     blender::Span<const char *> required_device_extensions,
                     blender::Span<const char *> optional_device_extensions)
  {
    device.emplace(vk_physical_device, use_vk_ext_swapchain_maintenance1);
    GHOST_DeviceVK &device = *this->device;

    device.extensions.enable(required_device_extensions);
    device.extensions.enable(optional_device_extensions, true);

    /* Disabling pipeline libraries and dynamic vertex input on AMD drivers due to random crashes
     * that are also happening when enabling the extension, but not using it at all. This needs
     * more investigation as it could be related to development workflows.
     *
     * This seems to affect the pro drivers more than the `Adrenalin` ones.
     * But as both share the same code-base it is better to disable them until
     * it is clear what causes the crashes and when these were fixed.
     *
     * Ref #151103
     */
    /* Nuru: VK_KHR_acceleration_structure requires deferred host operations and buffer device
     * address. Keep the trio consistent: without the prerequisites, drop the ray tracing
     * extensions instead of creating an invalid device. */
    if (device.extensions.is_enabled(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
        (!device.extensions.is_enabled(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) ||
         device.features_12.bufferDeviceAddress == VK_FALSE))
    {
      device.extensions.disable(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
      device.extensions.disable(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    }
    if (device.extensions.is_enabled(VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
        !device.extensions.is_enabled(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME))
    {
      device.extensions.disable(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    }

    const bool is_amd_driver = device.properties_12.driverID == VK_DRIVER_ID_AMD_PROPRIETARY ||
                               device.properties_12.driverID == VK_DRIVER_ID_AMD_OPEN_SOURCE;
    if (is_amd_driver && is_debug) {
      device.extensions.disable(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
      device.extensions.disable(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
      device.extensions.disable(VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME);
    }

#ifdef _WIN32
    /* Intel 7th to 10th Gen Processor iGPUs show a black screen at application startup when using
     * VK_EXT_vertex_input_dynamic_state. The used driver version for these iGPUs is 101.2xxx or
     * older.
     *
     * Ref: #147721
     */
    if (device.properties_12.driverID == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS &&
        device.properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
    {
      const uint32_t driver_version = device.properties.properties.driverVersion;
      uint32_t driver_version_major = driver_version >> 14u;
      uint32_t driver_version_minor = driver_version & 0x3fffu;
      if (driver_version_major < 101 ||
          (driver_version_major == 101 && driver_version_minor < 3000))
      {
        device.extensions.disable(VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME);
      }
    }
#endif

    device.init_generic_queue_family();

    float queue_priorities[] = {1.0f};
    vector<VkDeviceQueueCreateInfo> queue_create_infos;
    VkDeviceQueueCreateInfo graphic_queue_create_info = {};
    graphic_queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    graphic_queue_create_info.queueFamilyIndex = device.generic_queue_family;
    graphic_queue_create_info.queueCount = 1;
    graphic_queue_create_info.pQueuePriorities = queue_priorities;
    queue_create_infos.push_back(graphic_queue_create_info);

    VkPhysicalDeviceFeatures device_features = {};
#ifndef __APPLE__
    device_features.geometryShader = VK_TRUE;
#endif
    device_features.vertexPipelineStoresAndAtomics = VK_TRUE;
    device_features.multiViewport = VK_TRUE;
    device_features.shaderClipDistance = VK_TRUE;
    device_features.fragmentStoresAndAtomics = VK_TRUE;
    device_features.logicOp = VK_TRUE;
    device_features.dualSrcBlend = VK_TRUE;
    device_features.imageCubeArray = VK_TRUE;
    device_features.multiDrawIndirect = VK_TRUE;
    device_features.drawIndirectFirstInstance = VK_TRUE;
    device_features.samplerAnisotropy = device.features.features.samplerAnisotropy;
    device_features.wideLines = device.features.features.wideLines;
    /* Nuru: hardware ray tracing kernels rely on 64-bit integers and format-less storage image
     * access. Enable them when the device offers them (always present on NVIDIA RTX). */
    device_features.shaderInt64 = device.features.features.shaderInt64;
    device_features.shaderStorageImageReadWithoutFormat =
        device.features.features.shaderStorageImageReadWithoutFormat;
    device_features.shaderStorageImageWriteWithoutFormat =
        device.features.features.shaderStorageImageWriteWithoutFormat;

    VkDeviceCreateInfo device_create_info = {};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount = uint32_t(queue_create_infos.size());
    device_create_info.pQueueCreateInfos = queue_create_infos.data();
    device_create_info.enabledExtensionCount = uint32_t(device.extensions.enabled.size());
    device_create_info.ppEnabledExtensionNames = device.extensions.enabled.data();
    device_create_info.pEnabledFeatures = &device_features;

    std::vector<void *> feature_struct_ptr;

    /* Enable vulkan 11 features when supported on physical device. */
    VkPhysicalDeviceVulkan11Features vulkan_11_features = {};
    vulkan_11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vulkan_11_features.shaderDrawParameters = VK_TRUE;
    feature_struct_ptr.push_back(&vulkan_11_features);

    /* Enable optional vulkan 12 features when supported on physical device. */
    VkPhysicalDeviceVulkan12Features vulkan_12_features = {};
    vulkan_12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan_12_features.shaderOutputLayer = device.features_12.shaderOutputLayer;
    vulkan_12_features.shaderOutputViewportIndex = device.features_12.shaderOutputViewportIndex;
    vulkan_12_features.bufferDeviceAddress = device.features_12.bufferDeviceAddress;
    vulkan_12_features.timelineSemaphore = VK_TRUE;
    /* Nuru: hardware ray-tracing kernels use scalar block layout so GLSL uniform/storage blocks
     * byte-match the tightly packed C++ uniform structs. */
    vulkan_12_features.scalarBlockLayout = device.features_12.scalarBlockLayout;
    feature_struct_ptr.push_back(&vulkan_12_features);

#ifndef __APPLE__
    /* Enable provoking vertex. */
    VkPhysicalDeviceProvokingVertexFeaturesEXT provoking_vertex_features = {};
    provoking_vertex_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT;
    provoking_vertex_features.provokingVertexLast = VK_TRUE;
    feature_struct_ptr.push_back(&provoking_vertex_features);
#endif

    /* Enable dynamic rendering. */
    VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering = {};
    dynamic_rendering.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
    dynamic_rendering.dynamicRendering = VK_TRUE;
    feature_struct_ptr.push_back(&dynamic_rendering);

    VkPhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT
        dynamic_rendering_unused_attachments = {};
    dynamic_rendering_unused_attachments.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_FEATURES_EXT;
    dynamic_rendering_unused_attachments.dynamicRenderingUnusedAttachments = VK_TRUE;
    if (device.extensions.is_enabled(VK_EXT_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_EXTENSION_NAME)) {
      feature_struct_ptr.push_back(&dynamic_rendering_unused_attachments);
    }

    VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR dynamic_rendering_local_read = {};
    dynamic_rendering_local_read.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR;
    dynamic_rendering_local_read.dynamicRenderingLocalRead = VK_TRUE;
    if (device.extensions.is_enabled(VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME)) {
      feature_struct_ptr.push_back(&dynamic_rendering_local_read);
    }

    /* VK_EXT_robustness2 */
    VkPhysicalDeviceRobustness2FeaturesEXT robustness_2_features = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
    if (device.extensions.is_enabled(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME)) {
      robustness_2_features.nullDescriptor = device.features_robustness2.nullDescriptor;
      feature_struct_ptr.push_back(&robustness_2_features);
    }

    /* Query for Mainenance4 (core in Vulkan 1.3). */
    VkPhysicalDeviceMaintenance4FeaturesKHR maintenance_4 = {};
    maintenance_4.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES_KHR;
    maintenance_4.maintenance4 = VK_TRUE;
    if (device.extensions.is_enabled(VK_KHR_MAINTENANCE_4_EXTENSION_NAME)) {
      feature_struct_ptr.push_back(&maintenance_4);
    }

    /* Swap-chain maintenance 1 is optional. */
    VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT swapchain_maintenance_1 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT, nullptr, VK_TRUE};
    if (device.extensions.is_enabled(VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME)) {
      feature_struct_ptr.push_back(&swapchain_maintenance_1);
      device.use_vk_ext_swapchain_maintenance_1 = true;
    }

    /* Query and enable Fragment Shader Barycentrics. */
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR fragment_shader_barycentric = {};
    fragment_shader_barycentric.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR;
    fragment_shader_barycentric.fragmentShaderBarycentric = VK_TRUE;
    if (device.extensions.is_enabled(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME)) {
      feature_struct_ptr.push_back(&fragment_shader_barycentric);
    }

    /* VK_EXT_memory_priority */
    VkPhysicalDeviceMemoryPriorityFeaturesEXT memory_priority = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT, nullptr, VK_TRUE};
    if (device.extensions.is_enabled(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME)) {
      feature_struct_ptr.push_back(&memory_priority);
    }

    /* VK_EXT_pageable_device_local_memory */
    VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT pageable_device_local_memory = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES_EXT,
        nullptr,
        VK_TRUE};
    if (device.extensions.is_enabled(VK_EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_EXTENSION_NAME)) {
      feature_struct_ptr.push_back(&pageable_device_local_memory);
    }

    /* VK_EXT_graphics_pipeline_library */
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT graphics_pipeline_library = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT,
        nullptr,
        VK_TRUE};
    if (device.extensions.is_enabled(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME)) {
      feature_struct_ptr.push_back(&graphics_pipeline_library);
    }

    /* VK_EXT_line_rasterization */
    VkPhysicalDeviceLineRasterizationFeaturesKHR line_rasterization_features = {};
    line_rasterization_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT;
    line_rasterization_features.bresenhamLines = VK_TRUE;
    if (device.extensions.is_enabled(VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME)) {
      feature_struct_ptr.push_back(&line_rasterization_features);
    }

    /* VK_EXT_extended_dynamic_state */
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT extended_dynamic_state = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT, nullptr, VK_TRUE};
    if (device.extensions.is_enabled(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME)) {
      feature_struct_ptr.push_back(&extended_dynamic_state);
    }

    /* VK_EXT_vertex_input_dynamic_state */
    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT vertex_input_dynamic_state = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT,
        nullptr,
        VK_TRUE};
    if (device.extensions.is_enabled(VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME)) {
      feature_struct_ptr.push_back(&vertex_input_dynamic_state);
    }

    /* VK_EXT_host_image_copy */
    VkPhysicalDeviceHostImageCopyFeaturesEXT host_image_copy = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT, nullptr, VK_TRUE};
    if (device.extensions.is_enabled(VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME)) {
      feature_struct_ptr.push_back(&host_image_copy);
    }

    /* Nuru: VK_KHR_acceleration_structure */
    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features = {};
    acceleration_structure_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    acceleration_structure_features.accelerationStructure = VK_TRUE;
    if (device.extensions.is_enabled(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) {
      feature_struct_ptr.push_back(&acceleration_structure_features);
    }

    /* Nuru: VK_KHR_ray_query */
    VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features = {};
    ray_query_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    ray_query_features.rayQuery = VK_TRUE;
    if (device.extensions.is_enabled(VK_KHR_RAY_QUERY_EXTENSION_NAME)) {
      feature_struct_ptr.push_back(&ray_query_features);
    }

#ifdef _WIN32
    /* Nuru: VK_NV_device_diagnostics_config (Nsight Aftermath, BLENDER_NVIDIA_AFTERMATH=1).
     * Resource tracking names page-fault addresses, shader error reporting surfaces silent
     * OOB/misaligned accesses, automatic checkpoints record per-command call sites (the
     * checkpoints extension is appended above). Shader debug info is intentionally off:
     * our SPIR-V is built without debug info, and it adds compile overhead. */
    VkDeviceDiagnosticsConfigCreateInfoNV aftermath_diagnostics = {};
    aftermath_diagnostics.sType = VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV;
    aftermath_diagnostics.flags = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV |
                                  VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV |
                                  VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_ERROR_REPORTING_BIT_NV;
    if (nuru_aftermath::is_enabled() && nuru_aftermath::config_level() == 1 &&
        device.extensions.is_enabled(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME))
    {
      feature_struct_ptr.push_back(&aftermath_diagnostics);
    }
#endif

    /* Link all registered feature structs. */
    for (int i = 1; i < feature_struct_ptr.size(); i++) {
      ((VkBaseInStructure *)(feature_struct_ptr[i - 1]))->pNext =
          (VkBaseInStructure *)(feature_struct_ptr[i]);
    }

    device_create_info.pNext = feature_struct_ptr[0];
    VK_CHECK(vkCreateDevice(vk_physical_device, &device_create_info, nullptr, &device.vk_device),
             GHOST_kFailure);
    device.init_generic_queue();
    device.init_memory_allocator(vk_instance);
    return true;
  }
};

/** \} */

/**
 * A shared device between multiple contexts.
 *
 * The logical device needs to be shared as multiple contexts can be created and the logical
 * vulkan device they share should be the same otherwise memory operations might be done on the
 * incorrect device.
 */
static std::optional<GHOST_InstanceVK> vulkan_instance;

bool GHOST_ContextVK::is_instance_extension_enabled(blender::StringRefNull extension_name)
{
  if (!vulkan_instance.has_value()) {
    return false;
  }
  return vulkan_instance->extensions.is_enabled(extension_name.c_str());
}

bool GHOST_ContextVK::is_device_extension_enabled(blender::StringRefNull extension_name)
{
  if (!vulkan_instance.has_value()) {
    return false;
  }
  if (!vulkan_instance->device.has_value()) {
    return false;
  }
  return vulkan_instance->device->extensions.is_enabled(extension_name.c_str());
}

/** \} */

GHOST_ContextVK::GHOST_ContextVK(const GHOST_ContextParams &context_params,
#ifdef _WIN32
                                 HWND hwnd,
#elif defined(__APPLE__)
                                 void *metal_layer,
#else
                                 GHOST_TVulkanPlatformType platform,
                                 /* X11 */
                                 Window window,
                                 Display *display,
                                 /* Wayland */
                                 wl_surface *wayland_surface,
                                 wl_display *wayland_display,
                                 const GHOST_ContextVK_WindowInfo *wayland_window_info,
#endif
                                 int contextMajorVersion,
                                 int contextMinorVersion,
                                 const GHOST_GPUDevice &preferred_device,
                                 const GHOST_WindowHDRInfo *hdr_info)
    : GHOST_Context(context_params),
#ifdef _WIN32
      hwnd_(hwnd),
#elif defined(__APPLE__)
      metal_layer_(metal_layer),
#else
      platform_(platform),
      /* X11 */
      display_(display),
      window_(window),
      /* Wayland */
      wayland_surface_(wayland_surface),
      wayland_display_(wayland_display),
      wayland_window_info_(wayland_window_info),
#endif
      context_major_version_(contextMajorVersion),
      context_minor_version_(contextMinorVersion),
      preferred_device_(preferred_device),
      hdr_info_(hdr_info),
      surface_(VK_NULL_HANDLE),
      swapchain_(VK_NULL_HANDLE),
      frame_data_(2),
      render_frame_(0),
      use_hdr_swapchain_(false)
{
  frame_data_.reserve(5);
}

GHOST_ContextVK::~GHOST_ContextVK()
{
  if (vulkan_instance.has_value()) {
    GHOST_InstanceVK &instance_vk = vulkan_instance.value();
    GHOST_DeviceVK &device_vk = instance_vk.device.value();
    device_vk.wait_idle();
    for (VkFence fence : fence_pile_) {
      vkDestroyFence(device_vk.vk_device, fence, nullptr);
    }
    fence_pile_.clear();
    destroySwapchain();

    if (surface_ != VK_NULL_HANDLE) {
      vkDestroySurfaceKHR(instance_vk.vk_instance, surface_, nullptr);
    }

    device_vk.users--;
    if (device_vk.users == 0) {
      vulkan_instance.reset();
    }
  }
}

GHOST_TSuccess GHOST_ContextVK::swapBufferAcquire()
{
  if (acquired_swapchain_image_index_.has_value()) {
    assert(false);
    return GHOST_kFailure;
  }

  GHOST_DeviceVK &device_vk = vulkan_instance->device.value();
  VkDevice vk_device = device_vk.vk_device;

  /* This method is called after all the draw calls in the application, and it signals that
   * we are ready to both (1) submit commands for those draw calls to the device and
   * (2) begin building the next frame. It is assumed as an invariant that the submission fence
   * in the current GHOST_Frame has been signaled. So, we wait for the *next* GHOST_Frame's
   * submission fence to be signaled, to ensure the invariant holds for the next call to
   * `swapBuffers`.
   *
   * We will pass the current GHOST_Frame to the swap_buffer_draw_callback_ for command buffer
   * submission, and it is the responsibility of that callback to use the current GHOST_Frame's
   * fence for it's submission fence. Since the callback is called after we wait for the next frame
   * to be complete, it is also safe in the callback to clean up resources associated with the next
   * frame.
   */
  render_frame_ = (render_frame_ + 1) % frame_data_.size();
  GHOST_Frame &submission_frame_data = frame_data_[render_frame_];
  /* Wait for previous time that the frame was used to finish rendering. Presenting can
   * still happen in parallel, but acquiring needs can only happen when the frame acquire semaphore
   * has been signaled and waited for. */
  if (submission_frame_data.submission_fence) {
    vkWaitForFences(vk_device, 1, &submission_frame_data.submission_fence, true, UINT64_MAX);
  }
  for (VkSwapchainKHR swapchain : submission_frame_data.discard_pile.swapchains) {
    this->destroySwapchainPresentFences(swapchain);
  }
  submission_frame_data.discard_pile.destroy(vk_device);

  const bool use_hdr_swapchain = hdr_info_ &&
                                 (hdr_info_->wide_gamut_enabled || hdr_info_->hdr_enabled) &&
                                 device_vk.use_vk_ext_swapchain_colorspace;
  if (use_hdr_swapchain != use_hdr_swapchain_) {
    /* Re-create swapchain if HDR mode was toggled in the system settings. */
    recreateSwapchain(use_hdr_swapchain);
  }
  else {
#ifdef WITH_GHOST_WAYLAND
    /* Wayland doesn't provide a WSI with windowing capabilities, therefore cannot detect whether
     * the swap-chain needs to be recreated. But as a side effect we can recreate the swap-chain
     * before presenting. */
    if (wayland_window_info_) {
      const bool recreate_swapchain =
          ((wayland_window_info_->size[0] !=
            std::max(render_extent_.width, render_extent_min_.width)) ||
           (wayland_window_info_->size[1] !=
            std::max(render_extent_.height, render_extent_min_.height)));

      if (recreate_swapchain) {
        /* Swap-chain is out of date. Recreate swap-chain. */
        recreateSwapchain(use_hdr_swapchain);
      }
    }
#endif
  }
  /* there is no valid swapchain when the previous window was minimized. User can have maximized
   * the window so we need to check if the swapchain has to be created. */
  if (swapchain_ == VK_NULL_HANDLE) {
    recreateSwapchain(use_hdr_swapchain);
  }

  /* Acquiree next image, swapchain can be (or become) invalid when minimizing window.*/
  uint32_t image_index = 0;
  if (swapchain_ != VK_NULL_HANDLE) {
    /* Some platforms (NVIDIA/Wayland) can receive an out of date swapchain when acquiring the next
     * swapchain image. Other do it when calling vkQueuePresent. */
    VkResult acquire_result = VK_ERROR_OUT_OF_DATE_KHR;
    while (swapchain_ != VK_NULL_HANDLE &&
           (ELEM(acquire_result, VK_ERROR_OUT_OF_DATE_KHR, VK_SUBOPTIMAL_KHR)))
    {
      acquire_result = vkAcquireNextImageKHR(vk_device,
                                             swapchain_,
                                             UINT64_MAX,
                                             submission_frame_data.acquire_semaphore,
                                             VK_NULL_HANDLE,
                                             &image_index);
      if (ELEM(acquire_result, VK_ERROR_OUT_OF_DATE_KHR, VK_SUBOPTIMAL_KHR)) {
        recreateSwapchain(use_hdr_swapchain);
      }
    }
  }

  /* Acquired callback is also called when there is no swapchain.
   *
   * When acquiring swap chain (image) and the swap chain is discarded (window has been minimized).
   * We have trigger a last acquired callback to reduce the attachments of the GPUFramebuffer.
   * Vulkan backend will retrieve the data (getVulkanSwapChainFormat) containing a render extent of
   * 0,0.
   *
   * The next frame window manager will detect that the window is minimized and doesn't draw the
   * window at all.
   */
  if (swap_buffer_acquired_callback_) {
    swap_buffer_acquired_callback_();
  }

  if (swapchain_ == VK_NULL_HANDLE) {
    CLOG_TRACE(&LOG, "Swap-chain unavailable (minimized window).");
    return GHOST_kSuccess;
  }

  CLOG_DEBUG(&LOG,
             "Acquired swap-chain image (render_frame=%" PRIu64 ", image_index=%u)",
             render_frame_,
             image_index);
  acquired_swapchain_image_index_ = image_index;

  return GHOST_kSuccess;
}
VkFence GHOST_ContextVK::getFence()
{
  if (!fence_pile_.empty()) {
    VkFence fence = fence_pile_.back();
    fence_pile_.pop_back();
    return fence;
  }
  GHOST_DeviceVK &device_vk = vulkan_instance->device.value();
  VkFence fence = VK_NULL_HANDLE;
  const VkFenceCreateInfo fence_create_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  vkCreateFence(device_vk.vk_device, &fence_create_info, nullptr, &fence);
  return fence;
}

void GHOST_ContextVK::setPresentFence(VkSwapchainKHR swapchain, VkFence present_fence)
{
  if (present_fence == VK_NULL_HANDLE) {
    return;
  }
  present_fences_[swapchain].push_back(present_fence);
  GHOST_DeviceVK &device_vk = vulkan_instance->device.value();
  /** Recycle signaled fences. */
  for (std::pair<const VkSwapchainKHR, std::vector<VkFence>> &item : present_fences_) {
    std::vector<VkFence>::iterator end = item.second.end();
    std::vector<VkFence>::iterator it = std::remove_if(
        item.second.begin(), item.second.end(), [&](const VkFence fence) {
          if (vkGetFenceStatus(device_vk.vk_device, fence) == VK_NOT_READY) {
            return false;
          }
          vkResetFences(device_vk.vk_device, 1, &fence);
          fence_pile_.push_back(fence);
          return true;
        });
    item.second.erase(it, end);
  }
}

GHOST_TSuccess GHOST_ContextVK::swapBufferRelease()
{
  /* Minimized windows don't have a swapchain and swapchain image. In this case we perform the draw
   * to release render graph and discarded resources. */
  if (swapchain_ == VK_NULL_HANDLE) {
    GHOST_VulkanSwapChainData swap_chain_data = {};
    if (swap_buffer_draw_callback_) {
      swap_buffer_draw_callback_(&swap_chain_data);
    }
    return GHOST_kSuccess;
  }

  if (!acquired_swapchain_image_index_.has_value()) {
    assert(false);
    return GHOST_kFailure;
  }
  GHOST_DeviceVK &device_vk = vulkan_instance->device.value();
  VkDevice vk_device = device_vk.vk_device;

  uint32_t image_index = acquired_swapchain_image_index_.value();
  GHOST_SwapchainImage &swapchain_image = swapchain_images_[image_index];
  GHOST_Frame &submission_frame_data = frame_data_[render_frame_];
  const bool use_hdr_swapchain = hdr_info_ && hdr_info_->hdr_enabled &&
                                 device_vk.use_vk_ext_swapchain_colorspace;

  GHOST_VulkanSwapChainData swap_chain_data;
  swap_chain_data.image = swapchain_image.vk_image;
  swap_chain_data.surface_format = surface_format_;
  swap_chain_data.extent = render_extent_;
  swap_chain_data.submission_fence = submission_frame_data.submission_fence;
  swap_chain_data.acquire_semaphore = submission_frame_data.acquire_semaphore;
  swap_chain_data.present_semaphore = swapchain_image.present_semaphore;
  swap_chain_data.sdr_scale = (hdr_info_) ? hdr_info_->sdr_white_level : 1.0f;

  vkResetFences(vk_device, 1, &submission_frame_data.submission_fence);
  if (swap_buffer_draw_callback_) {
    swap_buffer_draw_callback_(&swap_chain_data);
  }

  VkPresentInfoKHR present_info = {};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = &swapchain_image.present_semaphore;
  present_info.swapchainCount = 1;
  present_info.pSwapchains = &swapchain_;
  present_info.pImageIndices = &image_index;
  present_info.pResults = nullptr;

  VkResult present_result = VK_SUCCESS;
  {
    std::scoped_lock lock(device_vk.queue_mutex);
    VkSwapchainPresentFenceInfoEXT fence_info{VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT};
    VkFence present_fence = VK_NULL_HANDLE;
    if (device_vk.use_vk_ext_swapchain_maintenance_1) {
      present_fence = this->getFence();

      fence_info.swapchainCount = 1;
      fence_info.pFences = &present_fence;

      present_info.pNext = &fence_info;
    }
    present_result = vkQueuePresentKHR(device_vk.generic_queue, &present_info);
    this->setPresentFence(swapchain_, present_fence);
  }
  acquired_swapchain_image_index_.reset();

  if (ELEM(present_result, VK_ERROR_OUT_OF_DATE_KHR, VK_SUBOPTIMAL_KHR)) {
    recreateSwapchain(use_hdr_swapchain);
    return GHOST_kSuccess;
  }
  if (present_result != VK_SUCCESS) {
    CLOG_ERROR(&LOG,
               "Vulkan: failed to present swap-chain image : %s",
               blender::gpu::to_string(present_result));
    return GHOST_kFailure;
  }

  return GHOST_kSuccess;
}

GHOST_TSuccess GHOST_ContextVK::getVulkanSwapChainFormat(
    GHOST_VulkanSwapChainData *r_swap_chain_data)
{
  r_swap_chain_data->image = VK_NULL_HANDLE;
  r_swap_chain_data->surface_format = surface_format_;
  r_swap_chain_data->extent = render_extent_;
  r_swap_chain_data->sdr_scale = (hdr_info_) ? hdr_info_->sdr_white_level : 1.0f;

  return GHOST_kSuccess;
}

GHOST_TSuccess GHOST_ContextVK::getVulkanHandles(GHOST_VulkanHandles &r_handles)
{
  r_handles = {
      VK_NULL_HANDLE, /* instance */
      VK_NULL_HANDLE, /* physical_device */
      VK_NULL_HANDLE, /* device */
      0,              /* queue_family */
      VK_NULL_HANDLE, /* queue */
      nullptr,        /* queue_mutex */
      VK_NULL_HANDLE, /* vma_allocator */
  };

  if (vulkan_instance.has_value() && vulkan_instance.value().device.has_value()) {
    GHOST_InstanceVK &instance_vk = vulkan_instance.value();
    GHOST_DeviceVK &device_vk = instance_vk.device.value();
    r_handles = {
        instance_vk.vk_instance,
        device_vk.vk_physical_device,
        device_vk.vk_device,
        device_vk.generic_queue_family,
        device_vk.generic_queue,
        &device_vk.queue_mutex,
        device_vk.vma_allocator,
    };
  }

  return GHOST_kSuccess;
}

GHOST_TSuccess GHOST_ContextVK::setVulkanSwapBuffersCallbacks(
    std::function<void(const GHOST_VulkanSwapChainData *)> swap_buffer_draw_callback,
    std::function<void(void)> swap_buffer_acquired_callback,
    std::function<void(GHOST_VulkanOpenXRData *)> openxr_acquire_framebuffer_image_callback,
    std::function<void(GHOST_VulkanOpenXRData *)> openxr_release_framebuffer_image_callback)
{
  swap_buffer_draw_callback_ = swap_buffer_draw_callback;
  swap_buffer_acquired_callback_ = swap_buffer_acquired_callback;
  openxr_acquire_framebuffer_image_callback_ = openxr_acquire_framebuffer_image_callback;
  openxr_release_framebuffer_image_callback_ = openxr_release_framebuffer_image_callback;
  return GHOST_kSuccess;
}

GHOST_TSuccess GHOST_ContextVK::activateDrawingContext()
{
  active_context_ = this;
  return GHOST_kSuccess;
}

GHOST_TSuccess GHOST_ContextVK::releaseDrawingContext()
{
  active_context_ = nullptr;
  return GHOST_kSuccess;
}

static GHOST_TSuccess selectPresentMode(const GHOST_TVSyncModes vsync,
                                        VkPhysicalDevice device,
                                        VkSurfaceKHR surface,
                                        VkPresentModeKHR *r_presentMode)
{
  uint32_t present_count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_count, nullptr);
  vector<VkPresentModeKHR> presents(present_count);
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_count, presents.data());

  if (vsync != GHOST_kVSyncModeUnset) {
    const bool vsync_off = (vsync == GHOST_kVSyncModeOff);
    if (vsync_off) {
      for (auto present_mode : presents) {
        if (present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
          *r_presentMode = present_mode;
          return GHOST_kSuccess;
        }
      }
      CLOG_WARN(&LOG,
                "Vulkan: VSync off was requested via --gpu-vsync, "
                "but VK_PRESENT_MODE_IMMEDIATE_KHR is not supported.");
    }
  }

  /* MAILBOX is the lowest latency V-Sync enabled mode. We will use it if available as it fixes
   * some lag on NVIDIA/Intel GPUs. */
  /* TODO: select the correct presentation mode based on the actual being performed by the user.
   * When low latency is required (paint cursor) we should select mailbox, otherwise we can do FIFO
   * to reduce CPU/GPU usage. */
  for (auto present_mode : presents) {
    if (present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
      *r_presentMode = present_mode;
      return GHOST_kSuccess;
    }
  }

  /* FIFO present mode is always available and we (should) prefer it as it will keep the main loop
   * running along the monitor refresh rate. Mailbox and FIFO relaxed can generate a lot of frames
   * that will never be displayed. */
  *r_presentMode = VK_PRESENT_MODE_FIFO_KHR;
  return GHOST_kSuccess;
}

/**
 * Select the surface format that we will use.
 *
 * We will select any 8bit UNORM surface.
 */
static bool selectSurfaceFormat(const VkPhysicalDevice physical_device,
                                const VkSurfaceKHR surface,
                                bool use_hdr_swapchain,
                                VkSurfaceFormatKHR &r_surfaceFormat)
{
  uint32_t format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, nullptr);
  vector<VkSurfaceFormatKHR> formats(format_count);
  vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, formats.data());

  array<pair<VkColorSpaceKHR, VkFormat>, 4> selection_order = {
      make_pair(VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT, VK_FORMAT_R16G16B16A16_SFLOAT),
      make_pair(VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, VK_FORMAT_R8G8B8A8_UNORM),
      make_pair(VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, VK_FORMAT_B8G8R8A8_UNORM),
  };

  for (pair<VkColorSpaceKHR, VkFormat> &pair : selection_order) {
    if (pair.second == VK_FORMAT_R16G16B16A16_SFLOAT && !use_hdr_swapchain) {
      continue;
    }
    for (const VkSurfaceFormatKHR &format : formats) {
      if (format.colorSpace == pair.first && format.format == pair.second) {
        r_surfaceFormat = format;
        return true;
      }
    }
  }

  return false;
}

GHOST_TSuccess GHOST_ContextVK::initializeFrameData()
{
  VkDevice device = vulkan_instance.value().device.value().vk_device;

  const VkSemaphoreCreateInfo vk_semaphore_create_info = {
      VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, nullptr, 0};
  const VkFenceCreateInfo vk_fence_create_info = {
      VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT};
  for (GHOST_SwapchainImage &swapchain_image : swapchain_images_) {
    /* VK_EXT_swapchain_maintenance1 reuses present semaphores. */
    if (swapchain_image.present_semaphore == VK_NULL_HANDLE) {
      VK_CHECK(vkCreateSemaphore(
                   device, &vk_semaphore_create_info, nullptr, &swapchain_image.present_semaphore),
               GHOST_kFailure);
    }
  }

  for (int index = 0; index < frame_data_.size(); index++) {
    GHOST_Frame &frame_data = frame_data_[index];
    /* VK_EXT_swapchain_maintenance1 reuses acquire semaphores. */
    if (frame_data.acquire_semaphore == VK_NULL_HANDLE) {
      VK_CHECK(vkCreateSemaphore(
                   device, &vk_semaphore_create_info, nullptr, &frame_data.acquire_semaphore),
               GHOST_kFailure);
    }
    if (frame_data.submission_fence == VK_NULL_HANDLE) {
      VK_CHECK(vkCreateFence(device, &vk_fence_create_info, nullptr, &frame_data.submission_fence),
               GHOST_kFailure);
    }
  }

  return GHOST_kSuccess;
}

GHOST_TSuccess GHOST_ContextVK::recreateSwapchain(bool use_hdr_swapchain)
{
  GHOST_InstanceVK &instance_vk = vulkan_instance.value();
  GHOST_DeviceVK &device_vk = instance_vk.device.value();

  surface_format_ = {};
  if (!selectSurfaceFormat(
          device_vk.vk_physical_device, surface_, use_hdr_swapchain, surface_format_))
  {
    return GHOST_kFailure;
  }

  VkPresentModeKHR present_mode;
  if (!selectPresentMode(getVSync(), device_vk.vk_physical_device, surface_, &present_mode)) {
    return GHOST_kFailure;
  }

  /* Query the surface capabilities for the given present mode on the surface. */
  VkSurfacePresentScalingCapabilitiesEXT vk_surface_present_scaling_capabilities = {
      VK_STRUCTURE_TYPE_SURFACE_PRESENT_SCALING_CAPABILITIES_EXT,
  };
  VkSurfaceCapabilities2KHR vk_surface_capabilities = {
      VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR,
      &vk_surface_present_scaling_capabilities,
  };
  VkSurfacePresentModeEXT vk_surface_present_mode = {
      VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_EXT, nullptr, present_mode};
  VkPhysicalDeviceSurfaceInfo2KHR vk_physical_device_surface_info = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR, &vk_surface_present_mode, surface_};
  VkSurfaceCapabilitiesKHR capabilities = {};

  if (device_vk.use_vk_ext_swapchain_maintenance_1) {
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilities2KHR(device_vk.vk_physical_device,
                                                        &vk_physical_device_surface_info,
                                                        &vk_surface_capabilities),
             GHOST_kFailure);
    capabilities = vk_surface_capabilities.surfaceCapabilities;
  }
  else {
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                 device_vk.vk_physical_device, surface_, &capabilities),
             GHOST_kFailure);
  }

  use_hdr_swapchain_ = use_hdr_swapchain;
  render_extent_ = capabilities.currentExtent;
  render_extent_min_ = capabilities.minImageExtent;
  if (render_extent_.width == UINT32_MAX) {
    /* Window Manager is going to set the surface size based on the given size.
     * Choose something between minImageExtent and maxImageExtent. */
    int width = 0;
    int height = 0;

#ifdef WITH_GHOST_WAYLAND
    /* Wayland doesn't provide a windowing API via WSI. */
    if (wayland_window_info_) {
      width = wayland_window_info_->size[0];
      height = wayland_window_info_->size[1];
    }
#endif

    if (width == 0 || height == 0) {
      width = 1280;
      height = 720;
    }

    render_extent_.width = width;
    render_extent_.height = height;

    if (capabilities.minImageExtent.width > render_extent_.width) {
      render_extent_.width = capabilities.minImageExtent.width;
    }
    if (capabilities.minImageExtent.height > render_extent_.height) {
      render_extent_.height = capabilities.minImageExtent.height;
    }
  }

  if (device_vk.use_vk_ext_swapchain_maintenance_1) {
    if (vk_surface_present_scaling_capabilities.minScaledImageExtent.width > render_extent_.width)
    {
      render_extent_.width = vk_surface_present_scaling_capabilities.minScaledImageExtent.width;
    }
    if (vk_surface_present_scaling_capabilities.minScaledImageExtent.height >
        render_extent_.height)
    {
      render_extent_.height = vk_surface_present_scaling_capabilities.minScaledImageExtent.height;
    }
  }

  /* Discard swapchain resources of current swapchain. */
  GHOST_FrameDiscard &discard_pile = frame_data_[render_frame_].discard_pile;
  for (GHOST_SwapchainImage &swapchain_image : swapchain_images_) {
    swapchain_image.vk_image = VK_NULL_HANDLE;
    if (swapchain_image.present_semaphore != VK_NULL_HANDLE) {
      discard_pile.semaphores.push_back(swapchain_image.present_semaphore);
      swapchain_image.present_semaphore = VK_NULL_HANDLE;
    }
  }

  /* Swap-chains with out any resolution should not be created. In the case the render extent is
   * zero we should not use the swap-chain.
   *
   * VUID-VkSwapchainCreateInfoKHR-imageExtent-01689
   */
  if (render_extent_.width == 0 || render_extent_.height == 0) {
    if (swapchain_) {
      discard_pile.swapchains.push_back(swapchain_);
      swapchain_ = VK_NULL_HANDLE;
    }
    return GHOST_kFailure;
  }

  /* Use double buffering when using FIFO. Increasing the number of images could stall when doing
   * actions that require low latency (paint cursor, UI resizing). MAILBOX prefers triple
   * buffering. */
  uint32_t image_count_requested = present_mode == VK_PRESENT_MODE_MAILBOX_KHR ? 3 : 2;
  /* NOTE: maxImageCount == 0 means no limit. */
  if (capabilities.minImageCount != 0 && image_count_requested < capabilities.minImageCount) {
    image_count_requested = capabilities.minImageCount;
  }
  if (capabilities.maxImageCount != 0 && image_count_requested > capabilities.maxImageCount) {
    image_count_requested = capabilities.maxImageCount;
  }

  VkSwapchainKHR old_swapchain = swapchain_;

  /* First time we stretch the swapchain image as it can happen that the first frame size isn't
   * correctly reported by the initial swapchain. All subsequent creations will use one to one as
   * that can reduce resizing artifacts. */
  VkPresentScalingFlagBitsEXT vk_present_scaling = old_swapchain == VK_NULL_HANDLE ?
                                                       VK_PRESENT_SCALING_STRETCH_BIT_EXT :
                                                       VK_PRESENT_SCALING_ONE_TO_ONE_BIT_EXT;

  VkSwapchainPresentModesCreateInfoEXT vk_swapchain_present_modes = {
      VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT, nullptr, 1, &present_mode};
  VkSwapchainPresentScalingCreateInfoEXT vk_swapchain_present_scaling = {
      VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_EXT,
      &vk_swapchain_present_modes,
      vk_surface_present_scaling_capabilities.supportedPresentScaling & vk_present_scaling,
      vk_surface_present_scaling_capabilities.supportedPresentGravityX &
          VK_PRESENT_GRAVITY_MIN_BIT_EXT,
      vk_surface_present_scaling_capabilities.supportedPresentGravityY &
          VK_PRESENT_GRAVITY_MAX_BIT_EXT,
  };

  VkSwapchainCreateInfoKHR create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  if (device_vk.use_vk_ext_swapchain_maintenance_1) {
    create_info.pNext = &vk_swapchain_present_scaling;
  }
  create_info.surface = surface_;
  create_info.minImageCount = image_count_requested;
  create_info.imageFormat = surface_format_.format;
  create_info.imageColorSpace = surface_format_.colorSpace;
  create_info.imageExtent = render_extent_;
  create_info.imageArrayLayers = 1;
  create_info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                           (use_hdr_swapchain ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
  create_info.preTransform = capabilities.currentTransform;
  create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  create_info.presentMode = present_mode;
  create_info.clipped = VK_TRUE;
  create_info.oldSwapchain = old_swapchain;
  create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  create_info.queueFamilyIndexCount = 0;
  create_info.pQueueFamilyIndices = nullptr;

  VK_CHECK(vkCreateSwapchainKHR(device_vk.vk_device, &create_info, nullptr, &swapchain_),
           GHOST_kFailure);

  /* image_count may not be what we requested! Getter for final value. */
  uint32_t actual_image_count = 0;
  vkGetSwapchainImagesKHR(device_vk.vk_device, swapchain_, &actual_image_count, nullptr);
  /* Some platforms require a minimum amount of render frames that is larger than we expect. When
   * that happens we should increase the number of frames in flight. We could also consider
   * splitting the frame in flight and image specific data. */
  if (actual_image_count > frame_data_.size()) {
    CLOG_TRACE(&LOG, "Vulkan: Increasing frame data to %u frames", actual_image_count);
    assert(actual_image_count <= frame_data_.capacity());
    frame_data_.resize(actual_image_count);
  }
  swapchain_images_.resize(actual_image_count);
  std::vector<VkImage> swapchain_images(actual_image_count);
  vkGetSwapchainImagesKHR(
      device_vk.vk_device, swapchain_, &actual_image_count, swapchain_images.data());
  for (int index = 0; index < actual_image_count; index++) {
    swapchain_images_[index].vk_image = swapchain_images[index];
  }
  CLOG_DEBUG(&LOG,
             "Vulkan: recreating swapchain: width=%u, height=%u, format=%d, colorSpace=%d, "
             "present_mode=%d, image_count_requested=%u, image_count_acquired=%u, "
             "swapchain=%" PRIx64 ", old_swapchain=%" PRIx64 "",
             render_extent_.width,
             render_extent_.height,
             surface_format_.format,
             surface_format_.colorSpace,
             present_mode,
             image_count_requested,
             actual_image_count,
             uint64_t(swapchain_),
             uint64_t(old_swapchain));
  /* Construct new semaphores. It can be that image_count is larger than previously. We only need
   * to fill in where the handle is `VK_NULL_HANDLE`. */
  /* Previous handles from the frame data cannot be used and should be discarded. */
  for (GHOST_Frame &frame : frame_data_) {
    if (frame.acquire_semaphore != VK_NULL_HANDLE) {
      discard_pile.semaphores.push_back(frame.acquire_semaphore);
    }
    frame.acquire_semaphore = VK_NULL_HANDLE;
  }
  if (old_swapchain) {
    discard_pile.swapchains.push_back(old_swapchain);
  }
  initializeFrameData();

  image_count_ = actual_image_count;

  return GHOST_kSuccess;
}

void GHOST_ContextVK::destroySwapchainPresentFences(VkSwapchainKHR swapchain)
{
  GHOST_DeviceVK &device_vk = vulkan_instance.value().device.value();
  const std::vector<VkFence> &fences = present_fences_[swapchain];
  if (!fences.empty()) {
    vkWaitForFences(device_vk.vk_device, fences.size(), fences.data(), VK_TRUE, UINT64_MAX);
    for (VkFence fence : fences) {
      vkDestroyFence(device_vk.vk_device, fence, nullptr);
    }
  }
  present_fences_.erase(swapchain);
}

GHOST_TSuccess GHOST_ContextVK::destroySwapchain()
{
  GHOST_DeviceVK &device_vk = vulkan_instance.value().device.value();

  if (swapchain_ != VK_NULL_HANDLE) {
    this->destroySwapchainPresentFences(swapchain_);
    vkDestroySwapchainKHR(device_vk.vk_device, swapchain_, nullptr);
  }
  device_vk.wait_idle();
  for (GHOST_SwapchainImage &swapchain_image : swapchain_images_) {
    swapchain_image.destroy(device_vk.vk_device);
  }
  swapchain_images_.clear();
  for (GHOST_Frame &frame_data : frame_data_) {
    for (VkSwapchainKHR swapchain : frame_data.discard_pile.swapchains) {
      this->destroySwapchainPresentFences(swapchain);
    }
    frame_data.destroy(device_vk.vk_device);
  }
  frame_data_.clear();

  return GHOST_kSuccess;
}

const char *GHOST_ContextVK::getPlatformSpecificSurfaceExtension() const
{
#ifdef _WIN32
  return VK_KHR_WIN32_SURFACE_EXTENSION_NAME;
#elif defined(__APPLE__)
  return VK_EXT_METAL_SURFACE_EXTENSION_NAME;
#else /* UNIX/Linux */
  switch (platform_) {
#  ifdef WITH_GHOST_X11
    case GHOST_kVulkanPlatformX11:
      return VK_KHR_XLIB_SURFACE_EXTENSION_NAME;
      break;
#  endif
#  ifdef WITH_GHOST_WAYLAND
    case GHOST_kVulkanPlatformWayland:
      return VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME;
      break;
#  endif
    case GHOST_kVulkanPlatformHeadless:
      break;
  }
#endif
  return nullptr;
}

GHOST_TSuccess GHOST_ContextVK::initializeDrawingContext()
{
  bool use_vk_ext_swapchain_colorspace = false;
#ifdef _WIN32
  const bool use_window_surface = (hwnd_ != nullptr);
#elif defined(__APPLE__)
  const bool use_window_surface = (metal_layer_ != nullptr);
#else /* UNIX/Linux */
  bool use_window_surface = false;
  switch (platform_) {
#  ifdef WITH_GHOST_X11
    case GHOST_kVulkanPlatformX11:
      use_window_surface = (display_ != nullptr) && (window_ != (Window) nullptr);
      break;
#  endif
#  ifdef WITH_GHOST_WAYLAND
    case GHOST_kVulkanPlatformWayland:
      use_window_surface = (wayland_display_ != nullptr) && (wayland_surface_ != nullptr);
      break;
#  endif
    case GHOST_kVulkanPlatformHeadless:
      use_window_surface = false;
      break;
  }
#endif

  blender::Vector<const char *> required_device_extensions;
  blender::Vector<const char *> optional_device_extensions;

  /* Initialize VkInstance */
  if (!vulkan_instance.has_value()) {
    vulkan_instance.emplace();
    GHOST_InstanceVK &instance_vk = vulkan_instance.value();
    instance_vk.extensions.enable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, true);

    /* Some XR platforms load functions without knowing if they were replaced by a core
     * function. Monado for example always uses the extension functions. Due to maintenance changes
     * drivers now only return the function pointer when the extension is enabled.
     *
     * We work around this by requesting Vulkan promoted extensions.
     */
#ifdef WITH_XR_OPENXR
    /* Vulkan 1.1 promoted instance extensions, enabled for OpenXR usage.*/
    instance_vk.extensions.enable(VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME);
    instance_vk.extensions.enable(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    instance_vk.extensions.enable(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
    instance_vk.extensions.enable(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    /* SteamVR requests both NVIDIA and KHR rectified extension. */
    instance_vk.extensions.enable(VK_NV_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME, true);

    /* Has been promoted to VK_EXT_debug_utils. */
    instance_vk.extensions.enable(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, true);
#endif

    if (use_window_surface) {
      const char *native_surface_extension_name = getPlatformSpecificSurfaceExtension();
      instance_vk.extensions.enable(VK_KHR_SURFACE_EXTENSION_NAME);
      instance_vk.extensions.enable(native_surface_extension_name);
      /* X11 doesn't use the correct swapchain offset, flipping can squash the first frames. */
      const bool use_vk_ext_swapchain_maintenance1 =
#ifdef WITH_GHOST_X11
          platform_ != GHOST_kVulkanPlatformX11 &&
#endif
          instance_vk.extensions.is_supported(VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME) &&
          instance_vk.extensions.is_supported(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
      if (use_vk_ext_swapchain_maintenance1) {
        instance_vk.extensions.enable(VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
        instance_vk.extensions.enable(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
        optional_device_extensions.append(VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
      }

      use_vk_ext_swapchain_colorspace = instance_vk.extensions.enable(
          VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME, true);

      required_device_extensions.append(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    if (!instance_vk.create_instance(
            VK_MAKE_VERSION(context_major_version_, context_minor_version_, 0)))
    {
      vulkan_instance.reset();
      return GHOST_kFailure;
    }
  }
  GHOST_InstanceVK &instance_vk = vulkan_instance.value();

  /* Initialize VkSurface */
  if (use_window_surface) {
#ifdef _WIN32
    VkWin32SurfaceCreateInfoKHR surface_create_info = {};
    surface_create_info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    surface_create_info.hinstance = GetModuleHandle(nullptr);
    surface_create_info.hwnd = hwnd_;
    VK_CHECK(
        vkCreateWin32SurfaceKHR(instance_vk.vk_instance, &surface_create_info, nullptr, &surface_),
        GHOST_kFailure);
#elif defined(__APPLE__)
    VkMetalSurfaceCreateInfoEXT info = {};
    info.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
    info.pNext = nullptr;
    info.flags = 0;
    info.pLayer = static_cast<CAMetalLayer *>(metal_layer_);
    VK_CHECK(vkCreateMetalSurfaceEXT(instance_vk.vk_instance, &info, nullptr, &surface_),
             GHOST_kFailure);
#else
    switch (platform_) {
#  ifdef WITH_GHOST_X11
      case GHOST_kVulkanPlatformX11: {
        VkXlibSurfaceCreateInfoKHR surface_create_info = {};
        surface_create_info.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
        surface_create_info.dpy = display_;
        surface_create_info.window = window_;
        VK_CHECK(vkCreateXlibSurfaceKHR(
                     instance_vk.vk_instance, &surface_create_info, nullptr, &surface_),
                 GHOST_kFailure);
        break;
      }
#  endif
#  ifdef WITH_GHOST_WAYLAND
      case GHOST_kVulkanPlatformWayland: {
        VkWaylandSurfaceCreateInfoKHR surface_create_info = {};
        surface_create_info.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
        surface_create_info.display = wayland_display_;
        surface_create_info.surface = wayland_surface_;
        VK_CHECK(vkCreateWaylandSurfaceKHR(
                     instance_vk.vk_instance, &surface_create_info, nullptr, &surface_),
                 GHOST_kFailure);
        break;
      }
#  endif
      case GHOST_kVulkanPlatformHeadless: {
        surface_ = VK_NULL_HANDLE;
        break;
      }
    }

#endif
  }

  /* Initialize VkDevice */
  if (!vulkan_instance->device.has_value()) {
    /* External memory extensions. */
#ifdef _WIN32
    optional_device_extensions.append(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#elif defined(__APPLE__)
#else /* Linux */
    optional_device_extensions.append(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif

#ifndef __APPLE__
    required_device_extensions.append(VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME);
#endif
    required_device_extensions.append(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    optional_device_extensions.append(VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME);
    optional_device_extensions.append(VK_EXT_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_EXTENSION_NAME);
    optional_device_extensions.append(VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME);
    optional_device_extensions.append(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    optional_device_extensions.append(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
    optional_device_extensions.append(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
    optional_device_extensions.append(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    optional_device_extensions.append(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME);
    optional_device_extensions.append(VK_EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_EXTENSION_NAME);
    optional_device_extensions.append(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
    optional_device_extensions.append(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
    optional_device_extensions.append(VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME);
    /* Nuru: hardware ray tracing (acceleration structures + ray queries in compute). */
    optional_device_extensions.append(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    optional_device_extensions.append(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    optional_device_extensions.append(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    /* Nuru: GPU-wedge diagnostic markers; recording is env-gated in the GPU backend
     * (BLENDER_VULKAN_CHECKPOINTS=1), enabling the extension alone is inert. */
    optional_device_extensions.append(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
#ifdef _WIN32
    /* Nuru: Nsight Aftermath feature configuration (BLENDER_NVIDIA_AFTERMATH=1; level 2 keeps
     * the watcher without the diagnostics config to preserve race timing). */
    if (nuru_aftermath::is_enabled() && nuru_aftermath::config_level() == 1) {
      optional_device_extensions.append(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME);
    }
#endif
    /* Disabled as the extension is available, but without any features set. */
#ifndef __APPLE__
    optional_device_extensions.append(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME);
#endif
    optional_device_extensions.append(VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME);
    optional_device_extensions.append(VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME);
    optional_device_extensions.append(VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME);
#if 0
    /* VK_EXT_host_image_copy isn't supported by Renderdoc and also isn't working as expected. */
    optional_device_extensions.append(VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME);
#endif

#ifdef WITH_XR_OPENXR
    optional_device_extensions.extend({
#  ifdef _WIN32
        VK_KHR_EXTERNAL_FENCE_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,

        VK_KHR_WIN32_KEYED_MUTEX_EXTENSION_NAME,
#  elif defined(__APPLE__)
#  else
        VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#  endif
        /* Vulkan 1.1 promoted device extensions, enabled for OpenXR usage. */
        VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
        VK_KHR_BIND_MEMORY_2_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
        VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME,
        VK_KHR_MULTIVIEW_EXTENSION_NAME,
        VK_KHR_MAINTENANCE_1_EXTENSION_NAME,
        VK_KHR_MAINTENANCE_2_EXTENSION_NAME,

        /* Vulkan 1.2 promoted device extensions, enabled for OpenXR usage. */
        VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,

        /* Vulkan 1.3 promoted device extensions, enabled for OpenXR usage. */
        VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME,

        /* Vulkan 1.4 promoted device extensions, enabled for OpenXR usage. */
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,

        /* Has been promoted to VK_EXT_debug_utils */
        VK_EXT_DEBUG_MARKER_EXTENSION_NAME});
#endif

    if (!instance_vk.select_physical_device(preferred_device_, required_device_extensions)) {
      return GHOST_kFailure;
    }

    if (!instance_vk.create_device(use_vk_ext_swapchain_colorspace,
                                   context_params_.is_debug,
                                   required_device_extensions,
                                   optional_device_extensions))
    {
      return GHOST_kFailure;
    }
  }
  GHOST_DeviceVK &device_vk = instance_vk.device.value();

  device_vk.users++;

  render_extent_ = {0, 0};
  render_extent_min_ = {0, 0};
  surface_format_ = {VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};

  active_context_ = this;
  return GHOST_kSuccess;
}

GHOST_TSuccess GHOST_ContextVK::releaseNativeHandles()
{
  return GHOST_kSuccess;
}
