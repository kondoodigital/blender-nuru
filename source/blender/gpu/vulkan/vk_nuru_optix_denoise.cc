/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * OptiX denoiser backend for the Nuru HWRT denoise step. See vk_nuru_optix_denoise.hh.
 *
 * CUDA is loaded at runtime through cuew (no build-time CUDA dependency); the OptiX entry
 * points come from the driver's nvoptix.dll through the SDK stub loader. The CUDA device is
 * matched to the active Vulkan device by UUID so the imported allocations alias the exact
 * memory the Vulkan pack kernel wrote.
 */

#ifdef WITH_NURU_OPTIX_DENOISER

#  include "vk_nuru_optix_denoise.hh"

#  include <algorithm>
#  include <cstdio>
#  include <cstring>

#  include "vk_backend.hh"
#  include "vk_device.hh"

/* cuew drags in windows.h; keep its min/max macros away from the STL/BLI templates. */
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <cuew.h>
#  ifdef min
#    undef min
#  endif
#  ifdef max
#    undef max
#  endif

/* cuew already declares the CUDA driver types the OptiX headers need.
 * The function table symbol (g_optixFunctionTable_<ABI>) is intentionally NOT defined here:
 * Cycles' OptiX device defines it (same SDK headers), and the build gate requires
 * WITH_CYCLES_DEVICE_OPTIX, so both share one table. optixInit() is idempotent. */
#  define OPTIX_DONT_INCLUDE_CUDA
#  include <optix.h>
#  include <optix_stubs.h>

namespace blender::gpu::nuru_optix {

/* cuew does not expose this flag; value from the CUDA driver API headers. */
#  ifndef CUDA_EXTERNAL_MEMORY_DEDICATED
#    define CUDA_EXTERNAL_MEMORY_DEDICATED 0x1
#  endif

struct OptixDenoiseCache {
  /* -1 = untried, 0 = unavailable/failed (disabled for the session), 1 = ready. */
  int init_state = -1;

  CUdevice cu_device = 0;
  CUcontext cu_context = nullptr; /* Primary context, retained. */
  CUstream cu_stream = nullptr;
  OptixDeviceContext optix_context = nullptr;

  OptixDenoiser denoiser = nullptr;
  bool denoiser_guide_albedo = false;
  bool denoiser_guide_normal = false;
  int setup_width = 0;
  int setup_height = 0;

  CUdeviceptr state_ptr = 0;
  size_t state_size = 0;
  CUdeviceptr scratch_ptr = 0;
  size_t scratch_size = 0;
  CUdeviceptr intensity_ptr = 0;
};

static OptixDenoiseCache &cache_get()
{
  static OptixDenoiseCache cache;
  return cache;
}

static void disable_with_notice(const char *site, const int code)
{
  OptixDenoiseCache &cache = cache_get();
  cache.init_state = 0;
  std::fprintf(stderr,
               "Nuru OptiX denoiser unavailable (%s failed, status=%d); "
               "falling back to OpenImageDenoise.\n",
               site,
               code);
}

#  define OPTIX_DENOISE_CHECK_CU(call) \
    { \
      const CUresult result_ = (call); \
      if (result_ != CUDA_SUCCESS) { \
        disable_with_notice(#call, int(result_)); \
        return false; \
      } \
    } \
    (void)0

#  define OPTIX_DENOISE_CHECK(call) \
    { \
      const OptixResult result_ = (call); \
      if (result_ != OPTIX_SUCCESS) { \
        disable_with_notice(#call, int(result_)); \
        return false; \
      } \
    } \
    (void)0

/* Pick the CUDA device whose UUID matches the active Vulkan physical device. */
static bool select_cuda_device(CUdevice &r_device)
{
  const VkPhysicalDeviceIDProperties &id_props =
      VKBackend::get().device.physical_device_id_properties_get();

  int device_count = 0;
  OPTIX_DENOISE_CHECK_CU(cuDeviceGetCount(&device_count));
  if (device_count <= 0) {
    disable_with_notice("cuDeviceGetCount", 0);
    return false;
  }
  for (int i = 0; i < device_count; i++) {
    CUdevice device = 0;
    if (cuDeviceGet(&device, i) != CUDA_SUCCESS) {
      continue;
    }
    CUuuid uuid = {};
    if (cuDeviceGetUuid != nullptr && cuDeviceGetUuid(&uuid, device) == CUDA_SUCCESS) {
      if (std::memcmp(uuid.bytes, id_props.deviceUUID, VK_UUID_SIZE) == 0) {
        r_device = device;
        return true;
      }
    }
  }
  /* No UUID match (or no UUID API): use the first device. Single-GPU systems are the common
   * case; on mismatch the import below fails cleanly and disables the path. */
  OPTIX_DENOISE_CHECK_CU(cuDeviceGet(&r_device, 0));
  return true;
}

static bool ensure_initialized()
{
  OptixDenoiseCache &cache = cache_get();
  if (cache.init_state == 1) {
    return true;
  }
  if (cache.init_state == 0) {
    return false;
  }

  if (cuewInit(CUEW_INIT_CUDA) != CUEW_SUCCESS) {
    disable_with_notice("cuewInit", 0);
    return false;
  }
  OPTIX_DENOISE_CHECK_CU(cuInit(0));
  if (!select_cuda_device(cache.cu_device)) {
    return false;
  }
  OPTIX_DENOISE_CHECK_CU(cuDevicePrimaryCtxRetain(&cache.cu_context, cache.cu_device));
  OPTIX_DENOISE_CHECK_CU(cuCtxPushCurrent(cache.cu_context));
  const CUresult stream_result = cuStreamCreate(&cache.cu_stream, CU_STREAM_NON_BLOCKING);
  cuCtxPopCurrent(nullptr);
  if (stream_result != CUDA_SUCCESS) {
    disable_with_notice("cuStreamCreate", int(stream_result));
    return false;
  }

  OPTIX_DENOISE_CHECK(optixInit());
  OptixDeviceContextOptions options = {};
  OPTIX_DENOISE_CHECK(optixDeviceContextCreate(cache.cu_context, &options, &cache.optix_context));

  cache.init_state = 1;
  return true;
}

static bool ensure_denoiser(const bool guide_albedo,
                            const bool guide_normal,
                            const int width,
                            const int height)
{
  OptixDenoiseCache &cache = cache_get();
  if (cache.denoiser != nullptr &&
      (cache.denoiser_guide_albedo != guide_albedo || cache.denoiser_guide_normal != guide_normal))
  {
    optixDenoiserDestroy(cache.denoiser);
    cache.denoiser = nullptr;
  }
  if (cache.denoiser == nullptr) {
    OptixDenoiserOptions options = {};
    options.guideAlbedo = guide_albedo ? 1u : 0u;
    options.guideNormal = guide_normal ? 1u : 0u;
    options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY; /* Packed planes carry no alpha. */
    OPTIX_DENOISE_CHECK(optixDenoiserCreate(
        cache.optix_context, OPTIX_DENOISER_MODEL_KIND_HDR, &options, &cache.denoiser));
    cache.denoiser_guide_albedo = guide_albedo;
    cache.denoiser_guide_normal = guide_normal;
    cache.setup_width = 0;
    cache.setup_height = 0;
  }

  if (cache.setup_width == width && cache.setup_height == height) {
    return true;
  }

  OptixDenoiserSizes sizes = {};
  OPTIX_DENOISE_CHECK(
      optixDenoiserComputeMemoryResources(cache.denoiser, unsigned(width), unsigned(height), &sizes));
  /* No tiling: the whole plane is denoised in one launch. The scratch also serves
   * `optixDenoiserComputeIntensity`. */
  const size_t scratch_size = std::max(sizes.withoutOverlapScratchSizeInBytes,
                                       sizes.computeIntensitySizeInBytes);

  if (cache.state_size < sizes.stateSizeInBytes) {
    if (cache.state_ptr != 0) {
      cuMemFree(cache.state_ptr);
      cache.state_ptr = 0;
    }
    OPTIX_DENOISE_CHECK_CU(cuMemAlloc(&cache.state_ptr, sizes.stateSizeInBytes));
    cache.state_size = sizes.stateSizeInBytes;
  }
  if (cache.scratch_size < scratch_size) {
    if (cache.scratch_ptr != 0) {
      cuMemFree(cache.scratch_ptr);
      cache.scratch_ptr = 0;
    }
    OPTIX_DENOISE_CHECK_CU(cuMemAlloc(&cache.scratch_ptr, scratch_size));
    cache.scratch_size = scratch_size;
  }
  if (cache.intensity_ptr == 0) {
    OPTIX_DENOISE_CHECK_CU(cuMemAlloc(&cache.intensity_ptr, sizeof(float)));
  }

  OPTIX_DENOISE_CHECK(optixDenoiserSetup(cache.denoiser,
                                         cache.cu_stream,
                                         unsigned(width),
                                         unsigned(height),
                                         cache.state_ptr,
                                         cache.state_size,
                                         cache.scratch_ptr,
                                         cache.scratch_size));
  cache.setup_width = width;
  cache.setup_height = height;
  return true;
}

/* Per-call zero-copy import of one exportable Vulkan allocation. Imports reference the
 * allocation independently of the OIDN import of the same NT handle; freeing the mapped
 * pointer and the import releases only this reference. */
struct ImportedPlane {
  CUexternalMemory memory = nullptr;
  CUdeviceptr ptr = 0;

  bool import(const DenoisePlane &plane)
  {
    if (plane.win32_handle == nullptr) {
      return false;
    }
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC handle_desc = {};
    handle_desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
    handle_desc.handle.win32.handle = plane.win32_handle;
    handle_desc.size = plane.alloc_size;
    /* The Vulkan allocation uses VkMemoryDedicatedAllocateInfo; CUDA requires the matching
     * dedicated flag on import. */
    handle_desc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
    if (cuImportExternalMemory(&memory, &handle_desc) != CUDA_SUCCESS) {
      memory = nullptr;
      return false;
    }
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC buffer_desc = {};
    buffer_desc.offset = 0;
    buffer_desc.size = plane.alloc_size;
    if (cuExternalMemoryGetMappedBuffer(&ptr, memory, &buffer_desc) != CUDA_SUCCESS) {
      ptr = 0;
      return false;
    }
    return true;
  }

  ~ImportedPlane()
  {
    if (ptr != 0) {
      cuMemFree(ptr);
    }
    if (memory != nullptr) {
      cuDestroyExternalMemory(memory);
    }
  }
};

static OptixImage2D plane_image(const CUdeviceptr ptr, const int width, const int height)
{
  OptixImage2D image = {};
  image.data = ptr;
  image.width = unsigned(width);
  image.height = unsigned(height);
  image.pixelStrideInBytes = sizeof(float) * 3;
  image.rowStrideInBytes = unsigned(width) * sizeof(float) * 3;
  image.format = OPTIX_PIXEL_FORMAT_FLOAT3;
  return image;
}

bool denoise(const DenoiseParams &params)
{
  if (params.color.win32_handle == nullptr || params.output.win32_handle == nullptr ||
      params.width <= 0 || params.height <= 0)
  {
    return false;
  }
  if (!ensure_initialized()) {
    return false;
  }
  OptixDenoiseCache &cache = cache_get();

  const bool guide_albedo = params.albedo.win32_handle != nullptr;
  const bool guide_normal = params.normal.win32_handle != nullptr;

  OPTIX_DENOISE_CHECK_CU(cuCtxPushCurrent(cache.cu_context));
  bool success = false;
  {
    if (!ensure_denoiser(guide_albedo, guide_normal, params.width, params.height)) {
      cuCtxPopCurrent(nullptr);
      return false;
    }

    ImportedPlane color, output, albedo, normal;
    if (!color.import(params.color) || !output.import(params.output) ||
        (guide_albedo && !albedo.import(params.albedo)) ||
        (guide_normal && !normal.import(params.normal)))
    {
      disable_with_notice("cuImportExternalMemory", 0);
      cuCtxPopCurrent(nullptr);
      return false;
    }

    OptixDenoiserLayer layer = {};
    layer.input = plane_image(color.ptr, params.width, params.height);
    layer.output = plane_image(output.ptr, params.width, params.height);

    OptixDenoiserGuideLayer guide_layer = {};
    if (guide_albedo) {
      guide_layer.albedo = plane_image(albedo.ptr, params.width, params.height);
    }
    if (guide_normal) {
      guide_layer.normal = plane_image(normal.ptr, params.width, params.height);
    }

    OptixDenoiserParams denoiser_params = {};
    denoiser_params.hdrIntensity = cache.intensity_ptr;
    denoiser_params.blendFactor = 0.0f;

    const OptixResult intensity_result = optixDenoiserComputeIntensity(cache.denoiser,
                                                                       cache.cu_stream,
                                                                       &layer.input,
                                                                       cache.intensity_ptr,
                                                                       cache.scratch_ptr,
                                                                       cache.scratch_size);
    if (intensity_result != OPTIX_SUCCESS) {
      disable_with_notice("optixDenoiserComputeIntensity", int(intensity_result));
      cuCtxPopCurrent(nullptr);
      return false;
    }
    const OptixResult invoke_result = optixDenoiserInvoke(cache.denoiser,
                                                          cache.cu_stream,
                                                          &denoiser_params,
                                                          cache.state_ptr,
                                                          cache.state_size,
                                                          &guide_layer,
                                                          &layer,
                                                          1,
                                                          0,
                                                          0,
                                                          cache.scratch_ptr,
                                                          cache.scratch_size);
    if (invoke_result != OPTIX_SUCCESS) {
      disable_with_notice("optixDenoiserInvoke", int(invoke_result));
      cuCtxPopCurrent(nullptr);
      return false;
    }
    /* The Vulkan unpack kernel reads the output plane right after; full CPU sync mirrors the
     * `oidnSyncDevice` contract of the OIDN path. The imported planes are released on scope
     * exit only after the stream has drained. */
    const CUresult sync_result = cuStreamSynchronize(cache.cu_stream);
    if (sync_result != CUDA_SUCCESS) {
      disable_with_notice("cuStreamSynchronize", int(sync_result));
      cuCtxPopCurrent(nullptr);
      return false;
    }
    success = true;
  }
  cuCtxPopCurrent(nullptr);
  return success;
}

void free_resources()
{
  OptixDenoiseCache &cache = cache_get();
  if (cache.init_state != 1) {
    return;
  }
  cuCtxPushCurrent(cache.cu_context);
  if (cache.denoiser != nullptr) {
    optixDenoiserDestroy(cache.denoiser);
    cache.denoiser = nullptr;
  }
  for (CUdeviceptr *ptr : {&cache.state_ptr, &cache.scratch_ptr, &cache.intensity_ptr}) {
    if (*ptr != 0) {
      cuMemFree(*ptr);
      *ptr = 0;
    }
  }
  cache.state_size = 0;
  cache.scratch_size = 0;
  if (cache.optix_context != nullptr) {
    optixDeviceContextDestroy(cache.optix_context);
    cache.optix_context = nullptr;
  }
  if (cache.cu_stream != nullptr) {
    cuStreamDestroy(cache.cu_stream);
    cache.cu_stream = nullptr;
  }
  cuCtxPopCurrent(nullptr);
  cuDevicePrimaryCtxRelease(cache.cu_device);
  cache.cu_context = nullptr;
  cache.init_state = -1;
  cache.setup_width = 0;
  cache.setup_height = 0;
}

}  // namespace blender::gpu::nuru_optix

#endif /* WITH_NURU_OPTIX_DENOISER */
