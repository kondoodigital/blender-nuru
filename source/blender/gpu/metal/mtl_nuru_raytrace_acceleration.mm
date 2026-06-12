/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include "mtl_nuru_raytrace_acceleration.hh"

#include "GPU_batch.hh"
#include "GPU_capabilities.hh"
#include "GPU_state.hh"
#include "GPU_vertex_format.hh"

#include "mtl_batch.hh"
#include "mtl_context.hh"
#include "mtl_index_buffer.hh"
#include "mtl_shader.hh"
#include "mtl_storage_buffer.hh"
#include "mtl_texture.hh"
#include "mtl_vertex_buffer.hh"

#include "BLI_math_matrix.hh"
#include "BLI_time.h"
#include "BLI_math_vector.hh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef WITH_OPENIMAGEDENOISE
#  include <OpenImageDenoise/oidn.h>
#  if OIDN_VERSION_MAJOR < 2
#    define oidnExecuteFilterAsync oidnExecuteFilter
#  endif
#endif

namespace blender {

static int gpu_metal_shadow_transparency_bits(const float value)
{
  int bits = 0;
  const float clamped = std::clamp(value, 0.0f, 1.0f);
  memcpy(&bits, &clamped, sizeof(bits));
  return bits;
}

struct GPUHardwareRaytraceScene {
  id<MTLAccelerationStructure> top_level_acceleration_structure = nil;
  id<MTLBuffer> emissive_radiance_buffer = nil;
  id<MTLBuffer> emissive_light_buffer = nil;
  id<MTLBuffer> diffuse_albedo_buffer = nil;
  id<MTLBuffer> material_proxy_buffer = nil;
  id<MTLBuffer> triangle_normal_buffer = nil;
  id<MTLBuffer> triangle_smooth_normal_buffer = nil;
  id<MTLBuffer> triangle_local_position_buffer = nil;
  id<MTLBuffer> triangle_normal_range_buffer = nil;
  std::vector<id<MTLAccelerationStructure>> bottom_level_acceleration_structures;
  std::vector<id<MTLBuffer>> geometry_buffers;
  std::vector<std::vector<float4>> local_triangle_normals;
  std::vector<std::vector<float4>> local_triangle_smooth_normals;
  std::vector<std::vector<float4>> local_triangle_positions;
  int geometry_count = 0;
  int instance_count = 0;
  int emissive_light_count = 0;
  id<MTLCommandBuffer> shadow_batch_command_buffer = nil;
  NSMutableArray *shadow_batch_retained_resources = nil;
  bool shadow_batch_has_work = false;

  ~GPUHardwareRaytraceScene()
  {
    if (shadow_batch_retained_resources != nil) {
      [shadow_batch_retained_resources release];
    }
    if (top_level_acceleration_structure != nil) {
      [top_level_acceleration_structure release];
    }
    if (emissive_radiance_buffer != nil) {
      [emissive_radiance_buffer release];
    }
    if (emissive_light_buffer != nil) {
      [emissive_light_buffer release];
    }
    if (diffuse_albedo_buffer != nil) {
      [diffuse_albedo_buffer release];
    }
    if (material_proxy_buffer != nil) {
      [material_proxy_buffer release];
    }
    if (triangle_normal_buffer != nil) {
      [triangle_normal_buffer release];
    }
    if (triangle_smooth_normal_buffer != nil) {
      [triangle_smooth_normal_buffer release];
    }
    if (triangle_local_position_buffer != nil) {
      [triangle_local_position_buffer release];
    }
    if (triangle_normal_range_buffer != nil) {
      [triangle_normal_range_buffer release];
    }
    for (id<MTLAccelerationStructure> blas : bottom_level_acceleration_structures) {
      if (blas != nil) {
        [blas release];
      }
    }
    for (id<MTLBuffer> geometry_buffer : geometry_buffers) {
      if (geometry_buffer != nil) {
        [geometry_buffer release];
      }
    }
  }
};

}  // namespace blender

namespace blender::gpu::metal {

struct SceneGeometryBuild {
  id<MTLAccelerationStructure> acceleration_structure = nil;
  id<MTLBuffer> vertex_buffer = nil;
  id<MTLBuffer> index_buffer = nil;
  float4x4 object_to_world = float4x4::identity();
  uint32_t instance_count = 1;
  uint32_t user_id = 0;
  float3 emissive_radiance = float3(0.0f);
  float3 diffuse_albedo = float3(0.8f);
  float3 reflection_color = float3(0.8f);
  float reflection_roughness = 1.0f;
  float3 transmission_color = float3(0.8f);
  float transmission_roughness = 1.0f;
  float reflection_ior = 1.45f;
  float refraction_ior = 1.45f;
  float packed_thickness = 0.0f;
  float alpha = 1.0f;
  float reflection_layer_coverage = 0.0f;
  uint32_t closure_type = 1u;
  uint32_t proxy_flags = 0u;
  std::vector<float4> triangle_normals;
  std::vector<float4> triangle_smooth_normals;
  std::vector<float4> triangle_local_positions;
};

static float scene_emissive_energy_sum(Span<SceneGeometryBuild> geometry)
{
  float energy_sum = 0.0f;
  for (const SceneGeometryBuild &entry : geometry) {
    const float emissive_max = std::max(
        std::max(entry.emissive_radiance.x, entry.emissive_radiance.y), entry.emissive_radiance.z);
    if (emissive_max <= 0.0f) {
      continue;
    }
    energy_sum += emissive_max * float(std::max(entry.instance_count, 1u));
  }
  return energy_sum;
}

static bool env_flag_enabled(const char *name)
{
  const char *value = std::getenv(name);
  return (value != nullptr) && (value[0] != '\0') && !(value[0] == '0' && value[1] == '\0');
}

static bool metal_raytrace_perf_logging_enabled()
{
  return env_flag_enabled("BLENDER_EEVEE_HWRT_PERF");
}

static void retain_resource(NSMutableArray *resources, id resource)
{
  if (resources != nil && resource != nil) {
    [resources addObject:resource];
  }
}

static NSMutableArray *retained_resources_for_command_buffer(id<MTLCommandBuffer> command_buffer,
                                                             const char *label)
{
  NSMutableArray *resources = [[NSMutableArray alloc] init];
  [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed_buffer) {
    if (completed_buffer.status != MTLCommandBufferStatusCompleted) {
      std::fprintf(stderr,
                   "%s failed with status=%ld\n",
                   label,
                   long(completed_buffer.status));
    }
    [resources release];
  }];
  return resources;
}

static void retain_scene_resources(GPUHardwareRaytraceScene *scene, NSMutableArray *resources)
{
  if (scene == nullptr || resources == nil) {
    return;
  }
  retain_resource(resources, scene->top_level_acceleration_structure);
  retain_resource(resources, scene->emissive_radiance_buffer);
  retain_resource(resources, scene->emissive_light_buffer);
  retain_resource(resources, scene->diffuse_albedo_buffer);
  retain_resource(resources, scene->material_proxy_buffer);
  retain_resource(resources, scene->triangle_normal_buffer);
  retain_resource(resources, scene->triangle_smooth_normal_buffer);
  retain_resource(resources, scene->triangle_local_position_buffer);
  retain_resource(resources, scene->triangle_normal_range_buffer);
  for (id<MTLAccelerationStructure> blas : scene->bottom_level_acceleration_structures) {
    retain_resource(resources, blas);
  }
  for (id<MTLBuffer> geometry_buffer : scene->geometry_buffers) {
    retain_resource(resources, geometry_buffer);
  }
}

static void encoder_use_buffer_vector(id<MTLComputeCommandEncoder> encoder,
                                      const std::vector<id<MTLBuffer>> &buffers,
                                      const MTLResourceUsage usage)
{
  if (encoder == nil || buffers.empty()) {
    return;
  }
  const id<MTLResource> __unsafe_unretained *resources =
      reinterpret_cast<const id<MTLResource> __unsafe_unretained *>(buffers.data());
  [encoder useResources:resources count:buffers.size() usage:usage];
}

static void encoder_use_scene_geometry_resources(id<MTLComputeCommandEncoder> encoder,
                                                 GPUHardwareRaytraceScene *scene)
{
  if (encoder == nil || scene == nullptr) {
    return;
  }
  [encoder useResource:scene->top_level_acceleration_structure usage:MTLResourceUsageRead];
  for (id<MTLAccelerationStructure> blas : scene->bottom_level_acceleration_structures) {
    [encoder useResource:blas usage:MTLResourceUsageRead];
  }
  encoder_use_buffer_vector(encoder, scene->geometry_buffers, MTLResourceUsageRead);
}

static void encoder_use_scene_shading_resources(id<MTLComputeCommandEncoder> encoder,
                                                GPUHardwareRaytraceScene *scene)
{
  if (encoder == nil || scene == nullptr) {
    return;
  }
  id<MTLResource> __unsafe_unretained resources[] = {
      scene->emissive_radiance_buffer,
      scene->emissive_light_buffer,
      scene->diffuse_albedo_buffer,
      scene->material_proxy_buffer,
      scene->triangle_normal_buffer,
      scene->triangle_smooth_normal_buffer,
      scene->triangle_local_position_buffer,
      scene->triangle_normal_range_buffer,
  };
  [encoder useResources:resources count:8 usage:MTLResourceUsageRead];
}

struct AccelerationStructureBuildBatch {
  id<MTLCommandBuffer> command_buffer = nil;
  id<MTLAccelerationStructureCommandEncoder> encoder = nil;
  NSMutableArray *retained_resources = nil;
};

static bool begin_acceleration_structure_build_batch(id<MTLCommandQueue> queue,
                                                     const char *label,
                                                     AccelerationStructureBuildBatch &r_batch)
{
  r_batch.command_buffer = [queue commandBuffer];
  if (r_batch.command_buffer == nil) {
    return false;
  }
  r_batch.retained_resources = retained_resources_for_command_buffer(r_batch.command_buffer, label);
  r_batch.encoder = [r_batch.command_buffer accelerationStructureCommandEncoder];
  if (r_batch.encoder == nil) {
    [r_batch.retained_resources release];
    r_batch.retained_resources = nil;
    r_batch.command_buffer = nil;
    return false;
  }
  return true;
}

static void commit_acceleration_structure_build_batch(AccelerationStructureBuildBatch &batch)
{
  if (batch.encoder != nil) {
    [batch.encoder endEncoding];
  }
  if (batch.command_buffer != nil) {
    [batch.command_buffer commit];
  }
  batch.encoder = nil;
  batch.command_buffer = nil;
  batch.retained_resources = nil;
}

static bool begin_shadow_trace_batch(id<MTLCommandQueue> queue,
                                     GPUHardwareRaytraceScene *scene,
                                     const char *label)
{
  if (scene == nullptr) {
    return false;
  }
  if (scene->shadow_batch_command_buffer != nil) {
    return true;
  }
  scene->shadow_batch_command_buffer = [queue commandBuffer];
  if (scene->shadow_batch_command_buffer == nil) {
    return false;
  }
  scene->shadow_batch_retained_resources = retained_resources_for_command_buffer(
      scene->shadow_batch_command_buffer, label);
  retain_scene_resources(scene, scene->shadow_batch_retained_resources);
  scene->shadow_batch_has_work = false;
  return true;
}

static void cancel_shadow_trace_resources_if_needed(const bool uses_batch,
                                                    NSMutableArray *retained_resources)
{
  if (!uses_batch && retained_resources != nil) {
    [retained_resources release];
  }
}

static id<MTLCommandBuffer> trace_command_buffer_for_shadow(GPUHardwareRaytraceScene *scene,
                                                            id<MTLCommandQueue> queue,
                                                            const char *label,
                                                            NSMutableArray **r_retained_resources,
                                                            bool &r_uses_batch)
{
  r_uses_batch = (scene != nullptr && scene->shadow_batch_command_buffer != nil);
  if (r_uses_batch) {
    *r_retained_resources = scene->shadow_batch_retained_resources;
    return scene->shadow_batch_command_buffer;
  }
  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  if (command_buffer == nil) {
    *r_retained_resources = nil;
    return nil;
  }
  *r_retained_resources = retained_resources_for_command_buffer(command_buffer, label);
  retain_scene_resources(scene, *r_retained_resources);
  return command_buffer;
}

static bool finish_shadow_trace_command_buffer(GPUHardwareRaytraceScene *scene,
                                               id<MTLCommandBuffer> command_buffer,
                                               const bool uses_batch)
{
  if (uses_batch) {
    if (scene != nullptr) {
      scene->shadow_batch_has_work = true;
    }
    return true;
  }
  if (command_buffer == nil) {
    return false;
  }
  [command_buffer commit];
  const bool wait_for_completion = env_flag_enabled("BLENDER_EEVEE_HWRT_FORCE_SYNC");
  if (wait_for_completion) {
    [command_buffer waitUntilCompleted];
  }

  const bool success = wait_for_completion ?
                           (command_buffer.status == MTLCommandBufferStatusCompleted) :
                           true;
  if (success && wait_for_completion) {
    GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH | GPU_BARRIER_SHADER_IMAGE_ACCESS);
  }
  return success;
}

static bool commit_shadow_trace_batch(GPUHardwareRaytraceScene *scene)
{
  if (scene == nullptr) {
    return false;
  }
  if (scene->shadow_batch_command_buffer == nil) {
    return true;
  }
  id<MTLCommandBuffer> command_buffer = scene->shadow_batch_command_buffer;
  const bool has_work = scene->shadow_batch_has_work;
  scene->shadow_batch_command_buffer = nil;
  scene->shadow_batch_retained_resources = nil;
  scene->shadow_batch_has_work = false;
  /* Commit even empty batches so the command buffer completion handler remains the single owner
   * of retained-resource cleanup. Leaving an uncommitted batch to deallocate during
   * `MTLBackend::render_end()` can surface as a failed shadow batch during autorelease-pool
   * teardown. */
  UNUSED_VARS(has_work);
  return finish_shadow_trace_command_buffer(scene, command_buffer, false);
}

struct HardwareTraceUniforms {
  float4x4 viewinv;
  float4x4 wininv;
  int2 full_resolution;
  int resolution_scale;
  int resolution_scale_denominator;
  int closure_index;
  uint32_t feature_mask;
  int hardware_trace_phase;
  int reflection_bounces;
  int refraction_bounces;
  int _pad0;
  int2 resolution_bias;
  float clamp_indirect;
  float4 world_probe_atlas_coord;
  /* x: specular/refraction world probe, y: emissive count, z: GI sample count,
   * w: diffuse GI world probe. */
  int4 use_environment_pad;
  int4 light_count_pad;
  float4 sampling_rand;
  /* x: secondary GI receiver gather enabled, y: gather sample count. */
  int4 secondary_gi_pad;
};

struct HardwareReflectedReceiverGIUniformsHost {
  int4 resolution_samples;
  float4 normal_bias_pad;
  int4 environment_pad;
  int4 light_count_pad;
  float4 sampling_rand;
  float4 world_probe_atlas_coord;
};

struct HardwareShadowUniforms {
  float4x4 viewinv;
  float4x4 wininv;
  int4 resolution_layer;
  float4 light_direction_bias;
  float4 shadow_params;
  int4 world_sun_slot_pad;
  float4 sampling_rand;
};

struct HardwareLocalShadowUniforms {
  float4x4 viewinv;
  float4x4 wininv;
  int4 resolution_layer_type;
  float4 light_position_radius;
  float4 light_x_axis_size_x;
  float4 light_y_axis_size_y;
  float4 shadow_offset_scale;
  float4 normal_bias_pad;
  float4 sampling_rand;
  /* Nuru: caustic-shadow extras. .x = Photons intensity (0..10). Remaining slots reserved. */
  float4 caustic_params;
};

struct HardwareEnvironmentVisibilityUniforms {
  float4x4 viewinv;
  float4x4 wininv;
  int4 resolution_samples;
  float4 normal_bias_pad;
  float4 sampling_rand;
};

struct EmissiveLightRecord {
  float4 center_radius;
};

static MTLAttributeFormat to_acceleration_vertex_format(MTLVertexFormat format)
{
  switch (format) {
    case MTLVertexFormatFloat2:
      return MTLAttributeFormatFloat2;
    case MTLVertexFormatFloat3:
      return MTLAttributeFormatFloat3;
    case MTLVertexFormatFloat4:
      return MTLAttributeFormatFloat4;
    default:
      return MTLAttributeFormatInvalid;
  }
}

static void copy_transform_to_metal(const float4x4 &transform,
                                    MTLPackedFloat4x3 &r_transform) API_AVAILABLE(macos(12.0))
{
  std::memset(&r_transform, 0, sizeof(r_transform));

  const float *src = transform.base_ptr();
  float *dst = reinterpret_cast<float *>(&r_transform);
  /* Blender matrices are already column-major in memory, and Metal expects the top three rows of
   * that 4x4 transform packed as four float3 columns. Preserve the translation that lives in the
   * fourth Blender column instead of reusing Cycles' 4x3 Transform transpose path verbatim. */
  for (int column = 0; column < 4; column++) {
    for (int row = 0; row < 3; row++) {
      dst[column * 3 + row] = src[column * 4 + row];
    }
  }
}

static id<MTLAccelerationStructure> build_acceleration_structure(
    id<MTLDevice> device,
    id<MTLAccelerationStructureCommandEncoder> encoder,
    NSMutableArray *retained_resources,
    MTLAccelerationStructureDescriptor *descriptor,
    NSArray *additional_resources = nil) API_AVAILABLE(macos(12.0))
{
  MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:descriptor];
  if (sizes.accelerationStructureSize == 0 || encoder == nil) {
    return nil;
  }

  id<MTLAccelerationStructure> acceleration_structure = [device
      newAccelerationStructureWithSize:sizes.accelerationStructureSize];
  if (acceleration_structure == nil) {
    return nil;
  }

  const NSUInteger scratch_size = (sizes.buildScratchBufferSize == 0) ?
                                      1 :
                                      sizes.buildScratchBufferSize;
  id<MTLBuffer> scratch_buffer = [device newBufferWithLength:scratch_size
                                                     options:MTLResourceStorageModePrivate];
  if (scratch_buffer == nil) {
    [acceleration_structure release];
    return nil;
  }

  retain_resource(retained_resources, acceleration_structure);
  retain_resource(retained_resources, scratch_buffer);
  retain_resource(retained_resources, additional_resources);
  [encoder buildAccelerationStructure:acceleration_structure
                           descriptor:descriptor
                        scratchBuffer:scratch_buffer
                  scratchBufferOffset:0];
  [scratch_buffer release];
  return acceleration_structure;
}

static id<MTLAccelerationStructure> build_acceleration_structure(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    MTLAccelerationStructureDescriptor *descriptor,
    NSArray *additional_resources = nil) API_AVAILABLE(macos(12.0))
{
  AccelerationStructureBuildBatch build_batch;
  if (!begin_acceleration_structure_build_batch(queue, "Metal RT AS build", build_batch)) {
    return nil;
  }
  id<MTLAccelerationStructure> acceleration_structure = build_acceleration_structure(
      device, build_batch.encoder, build_batch.retained_resources, descriptor, additional_resources);
  if (acceleration_structure == nil) {
    [build_batch.retained_resources release];
    return nil;
  }
  commit_acceleration_structure_build_batch(build_batch);
  return acceleration_structure;
}

static bool resolve_position_input(Batch *batch,
                                   id<MTLBuffer> &r_vertex_buffer,
                                   NSUInteger &r_vertex_buffer_offset,
                                   NSUInteger &r_vertex_stride,
                                   uint &r_vertex_count,
                                   MTLVertexFormat &r_vertex_format)
{
  for (VertBuf *vert_buf : Span<VertBuf *>(batch->verts, GPU_BATCH_VBO_MAX_LEN)) {
    if (vert_buf == nullptr) {
      continue;
    }

    const int attr_id = GPU_vertformat_attr_id_get(&vert_buf->format, "pos");
    if (attr_id < 0) {
      continue;
    }

    MTLVertBuf *metal_vert_buf = static_cast<MTLVertBuf *>(vert_buf);
    metal_vert_buf->bind();

    const GPUVertAttr &attr = vert_buf->format.attrs[attr_id];
    const MTLVertexFormat vertex_format = to_mtl(
        attr.type.comp_type(), attr.type.fetch_mode(), attr.type.comp_len());
    if (vertex_format == MTLVertexFormatInvalid) {
      continue;
    }

    r_vertex_buffer = metal_vert_buf->get_metal_buffer_for_raytracing();
    r_vertex_buffer_offset = attr.offset;
    r_vertex_stride = vert_buf->format.stride;
    r_vertex_count = vert_buf->vertex_len;
    r_vertex_format = vertex_format;
    return (r_vertex_buffer != nil) && (r_vertex_stride != 0);
  }

  return false;
}

static bool resolve_normal_input(Batch *batch,
                                 id<MTLBuffer> &r_vertex_buffer,
                                 NSUInteger &r_vertex_buffer_offset,
                                 NSUInteger &r_vertex_stride,
                                 uint &r_vertex_count,
                                 MTLVertexFormat &r_vertex_format)
{
  for (VertBuf *vert_buf : Span<VertBuf *>(batch->verts, GPU_BATCH_VBO_MAX_LEN)) {
    if (vert_buf == nullptr) {
      continue;
    }

    const int attr_id = GPU_vertformat_attr_id_get(&vert_buf->format, "nor");
    if (attr_id < 0) {
      continue;
    }

    MTLVertBuf *metal_vert_buf = static_cast<MTLVertBuf *>(vert_buf);
    metal_vert_buf->bind();

    const GPUVertAttr &attr = vert_buf->format.attrs[attr_id];
    const MTLVertexFormat vertex_format = to_mtl(
        attr.type.comp_type(), attr.type.fetch_mode(), attr.type.comp_len());
    if (vertex_format == MTLVertexFormatInvalid) {
      continue;
    }

    r_vertex_buffer = metal_vert_buf->get_metal_buffer_for_raytracing();
    r_vertex_buffer_offset = attr.offset;
    r_vertex_stride = vert_buf->format.stride;
    r_vertex_count = vert_buf->vertex_len;
    r_vertex_format = vertex_format;
    return (r_vertex_buffer != nil) && (r_vertex_stride != 0);
  }

  return false;
}

static float3 read_vertex_position(const void *vertex_base,
                                   NSUInteger vertex_buffer_offset,
                                   NSUInteger vertex_stride,
                                   MTLVertexFormat vertex_format,
                                   uint vertex_index)
{
  const char *vertex_ptr = static_cast<const char *>(vertex_base) + vertex_buffer_offset +
                           NSUInteger(vertex_index) * vertex_stride;
  switch (vertex_format) {
    case MTLVertexFormatFloat2: {
      const float2 &co = *reinterpret_cast<const float2 *>(vertex_ptr);
      return float3(co.x, co.y, 0.0f);
    }
    case MTLVertexFormatFloat3:
      return *reinterpret_cast<const float3 *>(vertex_ptr);
    case MTLVertexFormatFloat4: {
      const float4 &co = *reinterpret_cast<const float4 *>(vertex_ptr);
      return float3(co.x, co.y, co.z);
    }
    default:
      return float3(0.0f);
  }
}

static float snorm10_to_float(const uint32_t value)
{
  int v = int(value & 0x3FFu);
  if ((v & 0x200) != 0) {
    v |= ~0x3FF;
  }
  return std::max(float(v) / 511.0f, -1.0f);
}

static float snorm16_to_float(const int16_t value)
{
  return std::max(float(value) / 32767.0f, -1.0f);
}

static float half_to_float(const uint16_t bits)
{
  /* IEEE 754 half-precision -> single-precision conversion. */
  const uint32_t sign = uint32_t(bits & 0x8000u) << 16;
  uint32_t exponent = uint32_t(bits & 0x7C00u) >> 10;
  uint32_t mantissa = uint32_t(bits & 0x03FFu);
  uint32_t result;
  if (exponent == 0u) {
    if (mantissa == 0u) {
      result = sign;
    }
    else {
      /* Subnormal: normalize. */
      while ((mantissa & 0x0400u) == 0u) {
        mantissa <<= 1;
        exponent -= 1u;
      }
      exponent += 1u;
      mantissa &= 0x03FFu;
      result = sign | ((exponent + (127u - 15u)) << 23) | (mantissa << 13);
    }
  }
  else if (exponent == 0x1Fu) {
    result = sign | 0x7F800000u | (mantissa << 13);
  }
  else {
    result = sign | ((exponent + (127u - 15u)) << 23) | (mantissa << 13);
  }
  float f;
  std::memcpy(&f, &result, sizeof(f));
  return f;
}

static float3 read_vertex_normal(const void *vertex_base,
                                 NSUInteger vertex_buffer_offset,
                                 NSUInteger vertex_stride,
                                 MTLVertexFormat vertex_format,
                                 uint vertex_index)
{
  const char *vertex_ptr = static_cast<const char *>(vertex_base) + vertex_buffer_offset +
                           NSUInteger(vertex_index) * vertex_stride;
  switch (vertex_format) {
    case MTLVertexFormatFloat3:
      return *reinterpret_cast<const float3 *>(vertex_ptr);
    case MTLVertexFormatFloat4: {
      const float4 &nor = *reinterpret_cast<const float4 *>(vertex_ptr);
      return float3(nor.x, nor.y, nor.z);
    }
    case MTLVertexFormatInt1010102Normalized: {
      const uint32_t packed = *reinterpret_cast<const uint32_t *>(vertex_ptr);
      return float3(snorm10_to_float(packed >> 0),
                    snorm10_to_float(packed >> 10),
                    snorm10_to_float(packed >> 20));
    }
    /* Nuru: SNORM_16_16_16_16 is the high-quality loop-normal format Blender's mesh extractor
     * uploads (see `extract_mesh_vbo_lnor.cc`). Without this case the HWRT acceleration build
     * silently zeroed every per-corner smooth normal, and the per-triangle fallback flat normal
     * was used at refraction/reflection hits, producing visibly faceted glass even on smooth-
     * shaded meshes. Honoring this format is what makes HWRT respect Blender's smooth / auto-
     * smooth / flat shading choice end-to-end. */
    case MTLVertexFormatShort4Normalized: {
      const int16_t *nor = reinterpret_cast<const int16_t *>(vertex_ptr);
      return float3(snorm16_to_float(nor[0]), snorm16_to_float(nor[1]), snorm16_to_float(nor[2]));
    }
    case MTLVertexFormatShort3Normalized: {
      const int16_t *nor = reinterpret_cast<const int16_t *>(vertex_ptr);
      return float3(snorm16_to_float(nor[0]), snorm16_to_float(nor[1]), snorm16_to_float(nor[2]));
    }
    case MTLVertexFormatShort2Normalized: {
      const int16_t *nor = reinterpret_cast<const int16_t *>(vertex_ptr);
      return float3(snorm16_to_float(nor[0]), snorm16_to_float(nor[1]), 0.0f);
    }
    case MTLVertexFormatHalf4: {
      const uint16_t *nor = reinterpret_cast<const uint16_t *>(vertex_ptr);
      return float3(half_to_float(nor[0]), half_to_float(nor[1]), half_to_float(nor[2]));
    }
    case MTLVertexFormatHalf3: {
      const uint16_t *nor = reinterpret_cast<const uint16_t *>(vertex_ptr);
      return float3(half_to_float(nor[0]), half_to_float(nor[1]), half_to_float(nor[2]));
    }
    default:
      return float3(0.0f);
  }
}

static uint read_triangle_index(const void *index_base, MTLIndexType index_type, uint index)
{
  if (index_type == MTLIndexTypeUInt16) {
    return reinterpret_cast<const uint16_t *>(index_base)[index];
  }
  return reinterpret_cast<const uint32_t *>(index_base)[index];
}

static std::vector<float4> build_triangle_normal_data(id<MTLBuffer> vertex_buffer,
                                                      NSUInteger vertex_buffer_offset,
                                                      NSUInteger vertex_stride,
                                                      MTLVertexFormat vertex_format,
                                                      id<MTLBuffer> index_buffer,
                                                      NSUInteger index_buffer_offset,
                                                      MTLIndexType index_type,
                                                      NSUInteger triangle_count)
{
  std::vector<float4> triangle_normals(triangle_count, float4(0.0f));
  if (triangle_count == 0 || vertex_buffer == nil || [vertex_buffer contents] == nil) {
    return triangle_normals;
  }

  const void *vertex_base = [vertex_buffer contents];
  const void *index_base = (index_buffer != nil && [index_buffer contents] != nil) ?
                               (static_cast<const char *>([index_buffer contents]) + index_buffer_offset) :
                               nullptr;

  for (NSUInteger tri = 0; tri < triangle_count; tri++) {
    const uint i0 = (index_base != nullptr) ? read_triangle_index(index_base, index_type, uint(tri * 3 + 0)) :
                                              uint(tri * 3 + 0);
    const uint i1 = (index_base != nullptr) ? read_triangle_index(index_base, index_type, uint(tri * 3 + 1)) :
                                              uint(tri * 3 + 1);
    const uint i2 = (index_base != nullptr) ? read_triangle_index(index_base, index_type, uint(tri * 3 + 2)) :
                                              uint(tri * 3 + 2);

    const float3 p0 = read_vertex_position(
        vertex_base, vertex_buffer_offset, vertex_stride, vertex_format, i0);
    const float3 p1 = read_vertex_position(
        vertex_base, vertex_buffer_offset, vertex_stride, vertex_format, i1);
    const float3 p2 = read_vertex_position(
        vertex_base, vertex_buffer_offset, vertex_stride, vertex_format, i2);

    float3 N = math::cross(p1 - p0, p2 - p0);
    const float len_sq = math::length_squared(N);
    if (len_sq > 1.0e-20f) {
      N /= std::sqrt(len_sq);
    }
    else {
      N = float3(0.0f, 0.0f, 1.0f);
    }
    triangle_normals[tri] = float4(N, 0.0f);
  }

  return triangle_normals;
}

static std::vector<float4> build_triangle_smooth_normal_data(id<MTLBuffer> normal_buffer,
                                                             NSUInteger normal_buffer_offset,
                                                             NSUInteger normal_stride,
                                                             MTLVertexFormat normal_format,
                                                             id<MTLBuffer> index_buffer,
                                                             NSUInteger index_buffer_offset,
                                                             MTLIndexType index_type,
                                                             NSUInteger triangle_count,
                                                             const std::vector<float4> &fallback_normals)
{
  std::vector<float4> smooth_normals(triangle_count * 3, float4(0.0f));
  if (triangle_count == 0) {
    return smooth_normals;
  }

  const void *normal_base = (normal_buffer != nil && [normal_buffer contents] != nil) ?
                                [normal_buffer contents] :
                                nullptr;
  const void *index_base = (index_buffer != nil && [index_buffer contents] != nil) ?
                               (static_cast<const char *>([index_buffer contents]) + index_buffer_offset) :
                               nullptr;

  for (NSUInteger tri = 0; tri < triangle_count; tri++) {
    const uint i0 = (index_base != nullptr) ? read_triangle_index(index_base, index_type, uint(tri * 3 + 0)) :
                                              uint(tri * 3 + 0);
    const uint i1 = (index_base != nullptr) ? read_triangle_index(index_base, index_type, uint(tri * 3 + 1)) :
                                              uint(tri * 3 + 1);
    const uint i2 = (index_base != nullptr) ? read_triangle_index(index_base, index_type, uint(tri * 3 + 2)) :
                                              uint(tri * 3 + 2);

    const float3 fallback = (tri < fallback_normals.size()) ?
                                float3(fallback_normals[tri].x,
                                       fallback_normals[tri].y,
                                       fallback_normals[tri].z) :
                                float3(0.0f, 0.0f, 1.0f);
    float3 normals[3] = {fallback, fallback, fallback};
    if (normal_base != nullptr) {
      normals[0] = read_vertex_normal(
          normal_base, normal_buffer_offset, normal_stride, normal_format, i0);
      normals[1] = read_vertex_normal(
          normal_base, normal_buffer_offset, normal_stride, normal_format, i1);
      normals[2] = read_vertex_normal(
          normal_base, normal_buffer_offset, normal_stride, normal_format, i2);
    }

    for (int corner = 0; corner < 3; corner++) {
      float3 N = normals[corner];
      const float len_sq = math::length_squared(N);
      if (len_sq > 1.0e-20f) {
        N /= std::sqrt(len_sq);
      }
      else {
        N = fallback;
      }
      smooth_normals[tri * 3 + corner] = float4(N, 0.0f);
    }
  }

  return smooth_normals;
}

static std::vector<float4> build_triangle_local_position_data(id<MTLBuffer> vertex_buffer,
                                                              NSUInteger vertex_buffer_offset,
                                                              NSUInteger vertex_stride,
                                                              MTLVertexFormat vertex_format,
                                                              id<MTLBuffer> index_buffer,
                                                              NSUInteger index_buffer_offset,
                                                              MTLIndexType index_type,
                                                              NSUInteger triangle_count)
{
  std::vector<float4> local_positions(triangle_count * 3, float4(0.0f));
  if (triangle_count == 0 || vertex_buffer == nil || [vertex_buffer contents] == nil) {
    return local_positions;
  }

  const void *vertex_base = [vertex_buffer contents];
  const void *index_base = (index_buffer != nil && [index_buffer contents] != nil) ?
                               (static_cast<const char *>([index_buffer contents]) + index_buffer_offset) :
                               nullptr;

  for (NSUInteger tri = 0; tri < triangle_count; tri++) {
    const uint i0 = (index_base != nullptr) ? read_triangle_index(index_base, index_type, uint(tri * 3 + 0)) :
                                              uint(tri * 3 + 0);
    const uint i1 = (index_base != nullptr) ? read_triangle_index(index_base, index_type, uint(tri * 3 + 1)) :
                                              uint(tri * 3 + 1);
    const uint i2 = (index_base != nullptr) ? read_triangle_index(index_base, index_type, uint(tri * 3 + 2)) :
                                              uint(tri * 3 + 2);

    local_positions[tri * 3 + 0] = float4(
        read_vertex_position(vertex_base, vertex_buffer_offset, vertex_stride, vertex_format, i0), 0.0f);
    local_positions[tri * 3 + 1] = float4(
        read_vertex_position(vertex_base, vertex_buffer_offset, vertex_stride, vertex_format, i1), 0.0f);
    local_positions[tri * 3 + 2] = float4(
        read_vertex_position(vertex_base, vertex_buffer_offset, vertex_stride, vertex_format, i2), 0.0f);
  }

  return local_positions;
}

static bool build_entry_blas(MTLContext *ctx,
                             const GPUHardwareRaytraceSceneEntry &entry,
                             SceneGeometryBuild &r_geometry,
                             AccelerationStructureBuildBatch *build_batch = nullptr)
    API_AVAILABLE(macos(12.0))
{
  if (entry.batch == nullptr) {
    return false;
  }

  Batch *batch = entry.batch;

  id<MTLBuffer> vertex_buffer = nil;
  NSUInteger vertex_buffer_offset = 0;
  NSUInteger vertex_stride = 0;
  uint vertex_count = 0;
  MTLVertexFormat vertex_format = MTLVertexFormatInvalid;
  if (!resolve_position_input(
          batch, vertex_buffer, vertex_buffer_offset, vertex_stride, vertex_count, vertex_format))
  {
    return false;
  }

  id<MTLBuffer> normal_buffer = nil;
  NSUInteger normal_buffer_offset = 0;
  NSUInteger normal_stride = 0;
  uint normal_count = 0;
  MTLVertexFormat normal_format = MTLVertexFormatInvalid;
  const bool has_normal_input = resolve_normal_input(
      batch, normal_buffer, normal_buffer_offset, normal_stride, normal_count, normal_format);
  UNUSED_VARS(has_normal_input, normal_count);
  const MTLAttributeFormat acceleration_vertex_format = to_acceleration_vertex_format(vertex_format);
  if (acceleration_vertex_format == MTLAttributeFormatInvalid) {
    return false;
  }

  GPUPrimType final_primitive_type = batch->prim_type;
  id<MTLBuffer> index_buffer = nil;
  NSUInteger index_buffer_offset = 0;
  MTLIndexType index_type = MTLIndexTypeUInt32;
  NSUInteger triangle_count = 0;

  if (batch->elem != nullptr) {
    MTLIndexBuf *metal_index_buf = static_cast<MTLIndexBuf *>(batch->elem);
    uint index_count = metal_index_buf->index_len_get();
    if (index_count == 0) {
      return false;
    }

    metal_index_buf->upload_data();
    index_buffer = metal_index_buf->get_index_buffer(final_primitive_type, index_count);
    if (index_buffer == nil || final_primitive_type != GPU_PRIM_TRIS || (index_count % 3) != 0) {
      return false;
    }

    index_type = metal_index_buf->is_32bit() ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
    index_buffer_offset = metal_index_buf->index_start_get() *
                          (metal_index_buf->is_32bit() ? sizeof(uint32_t) :
                                                         sizeof(uint16_t));
    vertex_buffer_offset += metal_index_buf->index_base_get() * vertex_stride;
    triangle_count = index_count / 3;
  }
  else {
    if (final_primitive_type != GPU_PRIM_TRIS || (vertex_count % 3) != 0) {
      return false;
    }
    triangle_count = vertex_count / 3;
  }

  if (triangle_count == 0) {
    return false;
  }

  MTLAccelerationStructureTriangleGeometryDescriptor *geometry_descriptor =
      [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
  if (@available(macos 13.0, *)) {
    geometry_descriptor.vertexFormat = acceleration_vertex_format;
  }
  geometry_descriptor.vertexBuffer = vertex_buffer;
  geometry_descriptor.vertexBufferOffset = vertex_buffer_offset;
  geometry_descriptor.vertexStride = vertex_stride;
  geometry_descriptor.triangleCount = triangle_count;
  geometry_descriptor.intersectionFunctionTableOffset = 0;
  geometry_descriptor.allowDuplicateIntersectionFunctionInvocation = false;
  geometry_descriptor.opaque = true;

  if (index_buffer != nil) {
    geometry_descriptor.indexBuffer = index_buffer;
    geometry_descriptor.indexBufferOffset = index_buffer_offset;
    geometry_descriptor.indexType = index_type;
  }

  MTLPrimitiveAccelerationStructureDescriptor *acceleration_descriptor =
      [MTLPrimitiveAccelerationStructureDescriptor descriptor];
  acceleration_descriptor.geometryDescriptors = @[ geometry_descriptor ];

  NSMutableArray *build_resources = [[NSMutableArray alloc] init];
  retain_resource(build_resources, vertex_buffer);
  retain_resource(build_resources, index_buffer);
  id<MTLAccelerationStructure> acceleration_structure =
      (build_batch != nullptr) ?
          build_acceleration_structure(ctx->device,
                                       build_batch->encoder,
                                       build_batch->retained_resources,
                                       acceleration_descriptor,
                                       build_resources) :
          build_acceleration_structure(
              ctx->device, ctx->queue, acceleration_descriptor, build_resources);
  [build_resources release];
  if (acceleration_structure == nil) {
    return false;
  }

  r_geometry.acceleration_structure = acceleration_structure;
  if (vertex_buffer != nil) {
    [vertex_buffer retain];
    r_geometry.vertex_buffer = vertex_buffer;
  }
  if (index_buffer != nil) {
    [index_buffer retain];
    r_geometry.index_buffer = index_buffer;
  }
  r_geometry.object_to_world = entry.object_to_world;
  r_geometry.instance_count = std::max(entry.instance_count, uint32_t(1));
  r_geometry.user_id = entry.user_id;
  r_geometry.emissive_radiance = entry.emissive_radiance;
  r_geometry.diffuse_albedo = entry.diffuse_albedo;
  r_geometry.reflection_color = entry.reflection_color;
  r_geometry.reflection_roughness = entry.reflection_roughness;
  r_geometry.transmission_color = entry.transmission_color;
  r_geometry.transmission_roughness = entry.transmission_roughness;
  r_geometry.reflection_ior = entry.reflection_ior;
  r_geometry.refraction_ior = entry.refraction_ior;
  r_geometry.packed_thickness = entry.packed_thickness;
  r_geometry.alpha = entry.alpha;
  r_geometry.reflection_layer_coverage = entry.reflection_layer_coverage;
  r_geometry.closure_type = entry.closure_type;
  r_geometry.proxy_flags = entry.proxy_flags;
  r_geometry.triangle_normals = build_triangle_normal_data(vertex_buffer,
                                                           vertex_buffer_offset,
                                                           vertex_stride,
                                                           vertex_format,
                                                           index_buffer,
                                                           index_buffer_offset,
                                                           index_type,
                                                           triangle_count);
  r_geometry.triangle_smooth_normals = build_triangle_smooth_normal_data(normal_buffer,
                                                                         normal_buffer_offset,
                                                                         normal_stride,
                                                                         normal_format,
                                                                         index_buffer,
                                                                         index_buffer_offset,
                                                                         index_type,
                                                                         triangle_count,
                                                                         r_geometry.triangle_normals);
  r_geometry.triangle_local_positions = build_triangle_local_position_data(vertex_buffer,
                                                                           vertex_buffer_offset,
                                                                           vertex_stride,
                                                                           vertex_format,
                                                                           index_buffer,
                                                                           index_buffer_offset,
                                                                           index_type,
                                                                           triangle_count);
  return true;
}

static id<MTLAccelerationStructure> build_top_level_acceleration_structure(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    const std::vector<SceneGeometryBuild> &geometry) API_AVAILABLE(macos(12.0))
{
  if (geometry.empty()) {
    return nil;
  }

  NSUInteger instance_count = 0;
  for (const SceneGeometryBuild &entry : geometry) {
    instance_count += entry.instance_count;
  }
  if (instance_count == 0) {
    return nil;
  }

  id<MTLBuffer> instance_buffer = [device
      newBufferWithLength:instance_count * sizeof(MTLAccelerationStructureUserIDInstanceDescriptor)
                  options:MTLResourceStorageModeShared];
  if (instance_buffer == nil) {
    return nil;
  }

  auto *instances = reinterpret_cast<MTLAccelerationStructureUserIDInstanceDescriptor *>(
      instance_buffer.contents);
  std::vector<id<MTLAccelerationStructure>> blas_handles;
  blas_handles.reserve(geometry.size());

  NSUInteger write_index = 0;
  for (uint32_t geometry_index = 0; geometry_index < geometry.size(); geometry_index++) {
    const SceneGeometryBuild &entry = geometry[geometry_index];
    blas_handles.push_back(entry.acceleration_structure);

    for (uint32_t instance_index = 0; instance_index < entry.instance_count; instance_index++) {
      MTLAccelerationStructureUserIDInstanceDescriptor &descriptor = instances[write_index++];
      std::memset(&descriptor, 0, sizeof(descriptor));
      descriptor.accelerationStructureIndex = geometry_index;
      descriptor.userID = entry.user_id;
      descriptor.mask = 0xFF;
      descriptor.intersectionFunctionTableOffset = 0;
      descriptor.options = MTLAccelerationStructureInstanceOptionOpaque;
      copy_transform_to_metal(entry.object_to_world, descriptor.transformationMatrix);
    }
  }

  NSArray *all_blas = [NSArray arrayWithObjects:blas_handles.data() count:blas_handles.size()];

  MTLInstanceAccelerationStructureDescriptor *acceleration_descriptor =
      [MTLInstanceAccelerationStructureDescriptor descriptor];
  acceleration_descriptor.instanceCount = instance_count;
  acceleration_descriptor.instanceDescriptorType =
      MTLAccelerationStructureInstanceDescriptorTypeUserID;
  acceleration_descriptor.instanceDescriptorBuffer = instance_buffer;
  acceleration_descriptor.instanceDescriptorBufferOffset = 0;
  acceleration_descriptor.instanceDescriptorStride =
      sizeof(MTLAccelerationStructureUserIDInstanceDescriptor);
  acceleration_descriptor.instancedAccelerationStructures = all_blas;

  NSMutableArray *build_resources = [[NSMutableArray alloc] init];
  retain_resource(build_resources, instance_buffer);
  retain_resource(build_resources, all_blas);
  id<MTLAccelerationStructure> acceleration_structure = build_acceleration_structure(
      device, queue, acceleration_descriptor, build_resources);
  [build_resources release];

  [instance_buffer release];
  return acceleration_structure;
}

static id<MTLBuffer> build_emissive_radiance_buffer(id<MTLDevice> device,
                                                    const std::vector<SceneGeometryBuild> &geometry)
    API_AVAILABLE(macos(14.0))
{
  uint32_t max_user_id = 0;
  for (const SceneGeometryBuild &entry : geometry) {
    max_user_id = std::max(max_user_id, entry.user_id);
  }

  const NSUInteger color_count = geometry.empty() ? 1 : NSUInteger(max_user_id) + 1;
  id<MTLBuffer> buffer = [device newBufferWithLength:color_count * sizeof(float4)
                                             options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    return nil;
  }

  auto *emissive_radiance = reinterpret_cast<float4 *>(buffer.contents);
  for (NSUInteger i = 0; i < color_count; i++) {
    emissive_radiance[i] = float4(0.0f);
  }

  for (const SceneGeometryBuild &entry : geometry) {
    emissive_radiance[entry.user_id] = float4(entry.emissive_radiance, 0.0f);
  }

  return buffer;
}

static float4 compute_world_bounding_sphere(const SceneGeometryBuild &entry)
{
  if (entry.triangle_local_positions.empty()) {
    return float4(entry.object_to_world.location(), 1.0f);
  }

  float3 bounds_min = math::transform_point(
      entry.object_to_world, float3(entry.triangle_local_positions[0].x,
                                    entry.triangle_local_positions[0].y,
                                    entry.triangle_local_positions[0].z));
  float3 bounds_max = bounds_min;
  for (const float4 &local_position : entry.triangle_local_positions) {
    const float3 world_position = math::transform_point(
        entry.object_to_world, float3(local_position.x, local_position.y, local_position.z));
    bounds_min = math::min(bounds_min, world_position);
    bounds_max = math::max(bounds_max, world_position);
  }

  const float3 center = (bounds_min + bounds_max) * 0.5f;
  float radius_sq = 1.0e-6f;
  for (const float4 &local_position : entry.triangle_local_positions) {
    const float3 world_position = math::transform_point(
        entry.object_to_world, float3(local_position.x, local_position.y, local_position.z));
    radius_sq = std::max(radius_sq, math::distance_squared(center, world_position));
  }
  return float4(center, std::sqrt(radius_sq));
}

static id<MTLBuffer> build_emissive_light_buffer(id<MTLDevice> device,
                                                 const std::vector<SceneGeometryBuild> &geometry,
                                                 int &r_light_count) API_AVAILABLE(macos(14.0))
{
  std::vector<EmissiveLightRecord> emissive_lights;
  emissive_lights.reserve(geometry.size());
  for (const SceneGeometryBuild &entry : geometry) {
    const float emissive_peak = std::max(
        entry.emissive_radiance.x, std::max(entry.emissive_radiance.y, entry.emissive_radiance.z));
    if (!(emissive_peak > 0.0f)) {
      continue;
    }
    emissive_lights.push_back({compute_world_bounding_sphere(entry)});
  }

  r_light_count = int(emissive_lights.size());
  const NSUInteger light_count = emissive_lights.empty() ? 1 : emissive_lights.size();
  id<MTLBuffer> buffer = [device newBufferWithLength:light_count * sizeof(EmissiveLightRecord)
                                             options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    return nil;
  }

  auto *lights = reinterpret_cast<EmissiveLightRecord *>(buffer.contents);
  lights[0].center_radius = float4(0.0f, 0.0f, 0.0f, 1.0f);
  for (NSUInteger i = 0; i < emissive_lights.size(); i++) {
    lights[i] = emissive_lights[i];
  }
  return buffer;
}

static id<MTLBuffer> build_diffuse_albedo_buffer(id<MTLDevice> device,
                                                 const std::vector<SceneGeometryBuild> &geometry)
    API_AVAILABLE(macos(14.0))
{
  /* Indirect diffuse GI intentionally consumes the lean proxy set only:
   * emissive radiance comes from the separate emissive buffer, and diffuse transport only needs
   * this coarse albedo field instead of the specular/direct continuation payload. */
  uint32_t max_user_id = 0;
  for (const SceneGeometryBuild &entry : geometry) {
    max_user_id = std::max(max_user_id, entry.user_id);
  }

  const NSUInteger color_count = geometry.empty() ? 1 : NSUInteger(max_user_id) + 1;
  id<MTLBuffer> buffer = [device newBufferWithLength:color_count * sizeof(float4)
                                             options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    return nil;
  }

  auto *diffuse_albedo = reinterpret_cast<float4 *>(buffer.contents);
  for (NSUInteger i = 0; i < color_count; i++) {
    diffuse_albedo[i] = float4(0.8f, 0.8f, 0.8f, 0.0f);
  }

  for (const SceneGeometryBuild &entry : geometry) {
    diffuse_albedo[entry.user_id] = float4(entry.diffuse_albedo, 0.0f);
  }

  return buffer;
}

struct HardwareMaterialProxyRecord {
  float4 reflection_color_roughness;
  float4 transmission_color_roughness;
  float4 ior_closure_type;
  float4 packed_thickness;
};

struct TriangleNormalRangeRecord {
  uint32_t offset;
  uint32_t count;
};

static id<MTLBuffer> build_material_proxy_buffer(id<MTLDevice> device,
                                                 const std::vector<SceneGeometryBuild> &geometry)
    API_AVAILABLE(macos(14.0))
{
  /* Direct/specular fallback keeps the bounded continuation proxy separate from the diffuse GI
   * buffer: one dominant closure family plus tint, roughness, IOR, and the dielectric hint. */
  uint32_t max_user_id = 0;
  for (const SceneGeometryBuild &entry : geometry) {
    max_user_id = std::max(max_user_id, entry.user_id);
  }

  const NSUInteger proxy_count = geometry.empty() ? 1 : NSUInteger(max_user_id) + 1;
  id<MTLBuffer> buffer = [device newBufferWithLength:proxy_count * sizeof(HardwareMaterialProxyRecord)
                                             options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    return nil;
  }

  auto *proxies = reinterpret_cast<HardwareMaterialProxyRecord *>(buffer.contents);
  for (NSUInteger i = 0; i < proxy_count; i++) {
    proxies[i].reflection_color_roughness = float4(0.8f, 0.8f, 0.8f, 1.0f);
    proxies[i].transmission_color_roughness = float4(0.8f, 0.8f, 0.8f, 1.0f);
    proxies[i].ior_closure_type = float4(1.45f, 1.45f, 1.0f, 0.0f);
    proxies[i].packed_thickness = float4(0.0f);
  }

  for (const SceneGeometryBuild &entry : geometry) {
    proxies[entry.user_id].reflection_color_roughness = float4(entry.reflection_color,
                                                               entry.reflection_roughness);
    proxies[entry.user_id].transmission_color_roughness = float4(entry.transmission_color,
                                                                 entry.transmission_roughness);
    proxies[entry.user_id].ior_closure_type = float4(
        entry.reflection_ior, entry.refraction_ior, float(entry.closure_type), float(entry.proxy_flags));
    proxies[entry.user_id].packed_thickness = float4(
        entry.packed_thickness, entry.alpha, entry.reflection_layer_coverage, 0.0f);
  }

  return buffer;
}

static id<MTLBuffer> build_triangle_normal_buffer(id<MTLDevice> device,
                                                  const std::vector<SceneGeometryBuild> &geometry,
                                                  std::vector<TriangleNormalRangeRecord> &r_ranges)
    API_AVAILABLE(macos(14.0))
{
  uint32_t max_user_id = 0;
  for (const SceneGeometryBuild &entry : geometry) {
    max_user_id = std::max(max_user_id, entry.user_id);
  }

  r_ranges.assign(geometry.empty() ? 1 : max_user_id + 1, {0u, 0u});
  std::vector<float4> triangle_normals;
  for (const SceneGeometryBuild &entry : geometry) {
    TriangleNormalRangeRecord range = {};
    range.offset = uint32_t(triangle_normals.size());
    range.count = uint32_t(entry.triangle_normals.size());
    if (range.count > 0) {
      for (const float4 &normal_local : entry.triangle_normals) {
        float3 normal_world = math::transform_direction(
            entry.object_to_world, float3(normal_local.x, normal_local.y, normal_local.z));
        const float len_sq = math::length_squared(normal_world);
        if (len_sq > 1.0e-20f) {
          normal_world /= std::sqrt(len_sq);
        }
        else {
          normal_world = float3(0.0f, 0.0f, 1.0f);
        }
        triangle_normals.emplace_back(normal_world, 0.0f);
      }
    }
    r_ranges[entry.user_id] = range;
  }

  const NSUInteger normal_count = triangle_normals.empty() ? 1 : triangle_normals.size();
  id<MTLBuffer> buffer = [device newBufferWithLength:normal_count * sizeof(float4)
                                             options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    return nil;
  }

  auto *out_normals = reinterpret_cast<float4 *>(buffer.contents);
  out_normals[0] = float4(0.0f);
  for (NSUInteger i = 0; i < triangle_normals.size(); i++) {
    out_normals[i] = triangle_normals[i];
  }
  return buffer;
}

static id<MTLBuffer> build_triangle_smooth_normal_buffer(id<MTLDevice> device,
                                                         const std::vector<SceneGeometryBuild> &geometry,
                                                         const std::vector<TriangleNormalRangeRecord> &ranges)
    API_AVAILABLE(macos(14.0))
{
  std::vector<float4> triangle_smooth_normals;
  for (const SceneGeometryBuild &entry : geometry) {
    if (entry.triangle_smooth_normals.empty()) {
      continue;
    }
    for (const float4 &normal_local : entry.triangle_smooth_normals) {
      float3 normal_world = math::transform_direction(
          entry.object_to_world, float3(normal_local.x, normal_local.y, normal_local.z));
      const float len_sq = math::length_squared(normal_world);
      if (len_sq > 1.0e-20f) {
        normal_world /= std::sqrt(len_sq);
      }
      else {
        normal_world = float3(0.0f, 0.0f, 1.0f);
      }
      triangle_smooth_normals.emplace_back(normal_world, 0.0f);
    }
  }

  const NSUInteger normal_count = triangle_smooth_normals.empty() ? 1 : triangle_smooth_normals.size();
  id<MTLBuffer> buffer = [device newBufferWithLength:normal_count * sizeof(float4)
                                             options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    return nil;
  }

  auto *out_normals = reinterpret_cast<float4 *>(buffer.contents);
  out_normals[0] = float4(0.0f);
  for (NSUInteger i = 0; i < triangle_smooth_normals.size(); i++) {
    out_normals[i] = triangle_smooth_normals[i];
  }
  UNUSED_VARS(ranges);
  return buffer;
}

static id<MTLBuffer> build_triangle_local_position_buffer(
    id<MTLDevice> device, const std::vector<SceneGeometryBuild> &geometry)
{
  std::vector<float4> triangle_local_positions;
  for (const SceneGeometryBuild &entry : geometry) {
    if (entry.triangle_local_positions.empty()) {
      continue;
    }
    triangle_local_positions.insert(triangle_local_positions.end(),
                                    entry.triangle_local_positions.begin(),
                                    entry.triangle_local_positions.end());
  }

  const NSUInteger position_count = triangle_local_positions.empty() ? 1 :
                                                                     triangle_local_positions.size();
  id<MTLBuffer> buffer = [device newBufferWithLength:position_count * sizeof(float4)
                                             options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    return nil;
  }

  auto *out_positions = reinterpret_cast<float4 *>(buffer.contents);
  out_positions[0] = float4(0.0f);
  for (NSUInteger i = 0; i < triangle_local_positions.size(); i++) {
    out_positions[i] = triangle_local_positions[i];
  }
  return buffer;
}

static id<MTLBuffer> build_triangle_normal_range_buffer(id<MTLDevice> device,
                                                        const std::vector<TriangleNormalRangeRecord> &ranges)
    API_AVAILABLE(macos(14.0))
{
  const NSUInteger range_count = ranges.empty() ? 1 : ranges.size();
  id<MTLBuffer> buffer = [device newBufferWithLength:range_count * sizeof(TriangleNormalRangeRecord)
                                             options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    return nil;
  }

  auto *out_ranges = reinterpret_cast<TriangleNormalRangeRecord *>(buffer.contents);
  out_ranges[0] = {0u, 0u};
  for (NSUInteger i = 0; i < ranges.size(); i++) {
    out_ranges[i] = ranges[i];
  }
  return buffer;
}

static bool begin_hardware_trace_capture(id<MTLCommandQueue> queue) API_AVAILABLE(macos(10.15))
{
  const char *capture_path = std::getenv("BLENDER_EEVEE_METAL_RT_CAPTURE_PATH");
  if (capture_path == nullptr || capture_path[0] == '\0') {
    return false;
  }

  static std::atomic<bool> capture_consumed = false;
  if (capture_consumed.exchange(true)) {
    return false;
  }

  MTLCaptureManager *capture_manager = [MTLCaptureManager sharedCaptureManager];
  if (![capture_manager supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
    std::fprintf(stderr,
                 "Metal RT capture unsupported; launch with METAL_CAPTURE_ENABLED=1 or enable "
                 "Metal GPU capture in Xcode.\n");
    return false;
  }

  MTLCaptureDescriptor *capture_descriptor = [[MTLCaptureDescriptor alloc] init];
  capture_descriptor.captureObject = queue;
  capture_descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
  capture_descriptor.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:capture_path]];

  NSError *error = nil;
  const bool started = [capture_manager startCaptureWithDescriptor:capture_descriptor error:&error];
  [capture_descriptor release];
  if (!started) {
    std::fprintf(stderr,
                 "Metal RT capture start failed: %s\n",
                 (error != nil) ? error.localizedDescription.UTF8String : "unknown error");
    return false;
  }

  std::fprintf(stderr, "Metal RT capture started: %s\n", capture_path);
  return true;
}

static void end_hardware_trace_capture(const bool capture_started) API_AVAILABLE(macos(10.15))
{
  if (!capture_started) {
    return;
  }

  [[MTLCaptureManager sharedCaptureManager] stopCapture];
  std::fprintf(stderr, "Metal RT capture stopped\n");
}

static NSString *hardware_trace_shader_source() API_AVAILABLE(macos(14.0))
{
  return @"#include <metal_stdlib>\n"
         "#include <metal_raytracing>\n"
         "using namespace metal;\n"
         "using namespace metal::raytracing;\n"
         "constant int GBUF_NONE = 0;\n"
         "constant int GBUF_DIFFUSE = 1;\n"
         "constant int GBUF_REFLECTION = 2;\n"
         "constant int GBUF_REFLECTION_COLORLESS = 3;\n"
         "constant int GBUF_REFRACTION = 8;\n"
         "constant int GBUF_REFRACTION_COLORLESS = 9;\n"
         "constant int GBUF_SUBSURFACE = 11;\n"
         "constant int GBUFFER_HEADER_BITS_PER_BIN = 4;\n"
"constant uint GBUF_TRANSMISSION_BIT = 1u << 3u;\n"
         "constant uint FEATURE_HARDWARE_GI = 1u << 0u;\n"
         "constant uint FEATURE_HARDWARE_REFLECTIONS = 1u << 2u;\n"
         "constant uint FEATURE_HARDWARE_REFRACTIONS = 1u << 3u;\n"
         "constant uint HWRT_CLOSURE_DIFFUSE = 1u;\n"
         "constant uint HWRT_CLOSURE_REFLECTION = 7u;\n"
         "constant uint HWRT_CLOSURE_REFRACTION = 12u;\n"
         "constant uint HWRT_PROXY_FLAG_DIELECTRIC_REFLECTION = 1u << 0u;\n"
         "constant uint HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL = 1u << 1u;\n"
         "constant uint HWRT_PROXY_FLAG_ALPHA_BLEND_TRANSPARENT = 1u << 2u;\n"
         "constant uint HWRT_PROXY_FLAG_PRINCIPLED_TRANSMISSION_LAYER = 1u << 3u;\n"
         "constant uint HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL = 1u << 4u;\n"
         "constant uint HWRT_PROXY_FLAG_METALLIC_BSDF_SCENE_FINAL = 1u << 6u;\n"
         "constant uint HWRT_PROXY_FLAG_THIN_GLASS = 1u << 7u;\n"
         "constant uint HWRT_HIT_IDENTITY_PRINCIPLED_LAYERED_SCENE_FINAL = 1u << 5u;\n"
         "constant uint HWRT_HIT_IDENTITY_METALLIC_BSDF_SCENE_FINAL = 1u << 6u;\n"
         "inline uint2 unpackUvec2x16(uint packed)\n"
         "{\n"
         "  return uint2(packed & 0xFFFFu, packed >> 16u);\n"
         "}\n"
         "inline float3 barycentric_expand(float2 barycentric)\n"
         "{\n"
         "  return float3(max(0.0f, 1.0f - barycentric.x - barycentric.y), barycentric.x, barycentric.y);\n"
         "}\n"
         "/* Nuru: PCG hash for stochastic perturbation in the caustic trace. Used to give each\n"
         "  * receiver pixel + sample + bounce a tiny independent rotation of the metal mirror\n"
         "  * direction so adjacent receivers' bent rays diverge slightly, distributing what would\n"
         "  * otherwise be a binary-aligned/misaligned outcome across the receiver buffer. With\n"
         "  * higher HWRT Shadow Samples the per-pixel variance averages out as 1/sqrt(N). */\n"
         "inline uint pcg_hash(uint v)\n"
         "{\n"
         "  uint state = v * 747796405u + 2891336453u;\n"
         "  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;\n"
         "  return (word >> 22u) ^ word;\n"
         "}\n"
         "inline float2 hash_to_unit_disk(uint seed)\n"
         "{\n"
         "  const uint h1 = pcg_hash(seed);\n"
         "  const uint h2 = pcg_hash(h1);\n"
         "  const float u = float(h1) * (1.0f / 4294967296.0f);\n"
         "  const float v = float(h2) * (1.0f / 4294967296.0f);\n"
         "  const float r = sqrt(u);\n"
         "  const float theta = v * 6.283185307f;\n"
         "  return float2(r * cos(theta), r * sin(theta));\n"
         "}\n"
         "inline void orthonormal_basis(float3 N, thread float3 &T, thread float3 &B)\n"
         "{\n"
         "  const float3 up = (abs(N.z) < 0.9f) ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);\n"
         "  T = normalize(cross(N, up));\n"
         "  B = cross(N, T);\n"
         "}\n"
"struct HardwareMaterialProxy {\n"
"  float4 reflection_color_roughness;\n"
"  float4 transmission_color_roughness;\n"
"  float4 ior_closure_type;\n"
"  float4 packed_thickness;\n"
"};\n"
"struct TriangleNormalRange {\n"
"  uint offset;\n"
"  uint count;\n"
"};\n"
         "struct HardwareTraceUniforms {\n"
         "  float4x4 viewinv;\n"
         "  float4x4 wininv;\n"
         "  int2 full_resolution;\n"
         "  int resolution_scale;\n"
         "  int resolution_scale_denominator;\n"
         "  int closure_index;\n"
         "  uint feature_mask;\n"
         "  int hardware_trace_phase;\n"
         "  int reflection_bounces;\n"
         "  int refraction_bounces;\n"
         "  int _pad0;\n"
         "  int2 resolution_bias;\n"
         "  float clamp_indirect;\n"
         "  float4 world_probe_atlas_coord;\n"
"  int4 use_environment_pad; /* x: specular/refraction world probe, y: emissive count,\n"
"                             * z: GI sample count, w: diffuse GI world probe. */\n"
"  int4 light_count_pad;\n"
         "  float4 sampling_rand;\n"
"  int4 secondary_gi_pad;\n"
         "};\n"
         "struct HardwareShadowUniforms {\n"
         "  float4x4 viewinv;\n"
         "  float4x4 wininv;\n"
         "  int4 resolution_layer;\n"
         "  float4 light_direction_bias;\n"
         "  float4 shadow_params;\n"
"  int4 world_sun_slot_pad;\n"
         "  float4 sampling_rand;\n"
         "};\n"
         "struct HardwareLocalShadowUniforms {\n"
         "  float4x4 viewinv;\n"
         "  float4x4 wininv;\n"
         "  int4 resolution_layer_type;\n"
         "  float4 light_position_radius;\n"
         "  float4 light_x_axis_size_x;\n"
         "  float4 light_y_axis_size_y;\n"
         "  float4 shadow_offset_scale;\n"
         "  float4 normal_bias_pad;\n"
         "  float4 sampling_rand;\n"
         "  float4 caustic_params;\n"
         "};\n"
         "struct HardwareEnvironmentVisibilityUniforms {\n"
         "  float4x4 viewinv;\n"
         "  float4x4 wininv;\n"
         "  int4 resolution_samples;\n"
         "  float4 normal_bias_pad;\n"
         "  float4 sampling_rand;\n"
         "};\n"
"struct HardwareReflectedReceiverGIUniforms {\n"
"  int4 resolution_samples;\n"
"  float4 normal_bias_pad;\n"
"  int4 environment_pad;\n"
"  int4 light_count_pad;\n"
"  float4 sampling_rand;\n"
"  float4 world_probe_atlas_coord;\n"
"};\n"
"struct HardwareReceiverCausticUniforms {\n"
"  float4x4 viewinv;\n"
"  float4x4 wininv;\n"
"  int4 resolution_samples;\n"
"  float4 normal_bias_photons;\n"
"  int4 light_count_pad;\n"
"  float4 sampling_rand;\n"
"  float4 world_probe_atlas_coord;\n"
"};\n"
"struct EmissiveLightRecord {\n"
"  float4 center_radius;\n"
"};\n"
"struct FastGILightRecord {\n"
"  float4 object_to_world_x;\n"
"  float4 object_to_world_y;\n"
"  float4 object_to_world_z;\n"
"  float4 color_diffuse_power;\n"
"  float4 direction_type;\n"
"  float4 attenuation_spot;\n"
"  float4 spot_size_inv;\n"
"};\n"
         "constant uint LIGHT_SUN = 0u;\n"
         "constant uint LIGHT_SUN_ORTHO = 1u;\n"
         "constant uint LIGHT_OMNI_SPHERE = 10u;\n"
         "constant uint LIGHT_OMNI_DISK = 11u;\n"
         "constant uint LIGHT_SPOT_SPHERE = 12u;\n"
         "constant uint LIGHT_SPOT_DISK = 13u;\n"
         "constant uint LIGHT_RECT = 20u;\n"
         "constant uint LIGHT_ELLIPSE = 21u;\n"
         "inline float3 point_screen_to_world(float2 uv, float depth, float4x4 wininv, float4x4 viewinv)\n"
         "{\n"
         "  float3 ssP = float3(uv, depth);\n"
         "  float3 ndc = ssP * 2.0f - 1.0f;\n"
         "  float4 viewP = wininv * float4(ndc, 1.0f);\n"
         "  float3 vP = viewP.xyz / viewP.w;\n"
         "  return (viewinv * float4(vP, 1.0f)).xyz;\n"
         "}\n"
         "inline float3 point_screen_to_world(float2 uv, float depth, constant HardwareTraceUniforms &u)\n"
         "{\n"
         "  return point_screen_to_world(uv, depth, u.wininv, u.viewinv);\n"
         "}\n"
         "inline float3 point_screen_to_world(int2 texel, float depth, constant HardwareShadowUniforms &u)\n"
         "{\n"
         "  const float2 uv = (float2(texel) + 0.5f) / float2(u.resolution_layer.xy);\n"
         "  return point_screen_to_world(uv, depth, u.wininv, u.viewinv);\n"
         "}\n"
         "inline float3 point_screen_to_world(int2 texel, float depth, constant HardwareLocalShadowUniforms &u)\n"
         "{\n"
         "  const float2 uv = (float2(texel) + 0.5f) / float2(u.resolution_layer_type.xy);\n"
         "  return point_screen_to_world(uv, depth, u.wininv, u.viewinv);\n"
         "}\n"
         "inline float3 point_screen_to_world(int2 texel, float depth, constant HardwareEnvironmentVisibilityUniforms &u)\n"
         "{\n"
         "  const float2 uv = (float2(texel) + 0.5f) / float2(u.resolution_samples.xy);\n"
         "  return point_screen_to_world(uv, depth, u.wininv, u.viewinv);\n"
         "}\n"
"inline float3 point_screen_to_world(int2 texel, float depth, constant HardwareReceiverCausticUniforms &u)\n"
"{\n"
"  const float2 uv = (float2(texel) + 0.5f) / float2(u.resolution_samples.xy);\n"
"  return point_screen_to_world(uv, depth, u.wininv, u.viewinv);\n"
"}\n"
         "inline bool depth_is_valid(float depth)\n"
         "{\n"
         "  return depth > 0.0f && depth < 1.0f;\n"
         "}\n"
         "inline float sample_depth_clamped(int2 texel, depth2d<float, access::sample> depth_tx, constant HardwareShadowUniforms &u)\n"
         "{\n"
         "  constexpr sampler depth_sampler(coord::normalized, address::clamp_to_edge, filter::nearest);\n"
         "  const int2 clamped = clamp(texel, int2(0), int2(u.resolution_layer.xy) - int2(1));\n"
         "  const float2 uv = (float2(clamped) + 0.5f) / float2(u.resolution_layer.xy);\n"
         "  return 1.0f - depth_tx.sample(depth_sampler, uv);\n"
         "}\n"
         "inline float sample_depth_clamped(int2 texel, depth2d<float, access::sample> depth_tx, constant HardwareLocalShadowUniforms &u)\n"
         "{\n"
         "  constexpr sampler depth_sampler(coord::normalized, address::clamp_to_edge, filter::nearest);\n"
         "  const int2 clamped = clamp(texel, int2(0), int2(u.resolution_layer_type.xy) - int2(1));\n"
         "  const float2 uv = (float2(clamped) + 0.5f) / float2(u.resolution_layer_type.xy);\n"
         "  return 1.0f - depth_tx.sample(depth_sampler, uv);\n"
         "}\n"
         "inline float sample_depth_clamped(int2 texel, depth2d<float, access::sample> depth_tx, constant HardwareEnvironmentVisibilityUniforms &u)\n"
         "{\n"
         "  constexpr sampler depth_sampler(coord::normalized, address::clamp_to_edge, filter::nearest);\n"
         "  const int2 clamped = clamp(texel, int2(0), int2(u.resolution_samples.xy) - int2(1));\n"
         "  const float2 uv = (float2(clamped) + 0.5f) / float2(u.resolution_samples.xy);\n"
         "  return 1.0f - depth_tx.sample(depth_sampler, uv);\n"
         "}\n"
"inline float sample_depth_clamped(int2 texel, depth2d<float, access::sample> depth_tx, constant HardwareReceiverCausticUniforms &u)\n"
"{\n"
"  constexpr sampler depth_sampler(coord::normalized, address::clamp_to_edge, filter::nearest);\n"
"  const int2 clamped = clamp(texel, int2(0), int2(u.resolution_samples.xy) - int2(1));\n"
"  const float2 uv = (float2(clamped) + 0.5f) / float2(u.resolution_samples.xy);\n"
"  return 1.0f - depth_tx.sample(depth_sampler, uv);\n"
"}\n"
         "inline float3 normal_unpack(float2 N_packed)\n"
         "{\n"
         "  N_packed = N_packed * 2.0f - 1.0f;\n"
         "  float3 N = float3(N_packed.x, N_packed.y, 1.0f - fabs(N_packed.x) - fabs(N_packed.y));\n"
         "  const float t = clamp(-N.z, 0.0f, 1.0f);\n"
         "  N.x += (N.x >= 0.0f) ? -t : t;\n"
         "  N.y += (N.y >= 0.0f) ? -t : t;\n"
         "  return normalize(N);\n"
         "}\n"
         "inline float3 geometry_normal_unpack(uint data, float3 N)\n"
         "{\n"
         "  if ((data & (63u << 20u)) == 0u) {\n"
         "    return N;\n"
         "  }\n"
         "  float3 Ng = float3((uint3(data) >> (uint3(0, 1, 2) + 20u)) & 1u) -\n"
         "              float3((uint3(data) >> (uint3(3, 4, 5) + 20u)) & 1u);\n"
         "  return normalize(Ng);\n"
         "}\n"
         "inline bool load_gbuffer_receiver_normal(int2 texel,\n"
         "                                         texture2d_array<uint, access::read> gbuf_header_tx,\n"
         "                                         texture2d_array<float, access::read> gbuf_normal_tx,\n"
         "                                         thread float3 &r_N)\n"
         "{\n"
         "  const uint header = gbuf_header_tx.read(uint2(texel), 0).x;\n"
         "  if (header == 0u) {\n"
         "    return false;\n"
         "  }\n"
         "  const float2 packed_N = gbuf_normal_tx.read(uint2(texel), 0).xy;\n"
         "  const float3 surface_N = normal_unpack(packed_N);\n"
         "  r_N = geometry_normal_unpack(header, surface_N);\n"
         "  return all(isfinite(r_N)) && (dot(r_N, r_N) > 1.0e-10f);\n"
         "}\n"
"struct ThicknessData {\n"
"  float value;\n"
"  bool sphere_mode;\n"
"};\n"
"inline uint gbuffer_bin_to_layer(uint header, uint bin_id)\n"
"{\n"
"  const uint type0 = (header >> (GBUFFER_HEADER_BITS_PER_BIN * 0u)) & 15u;\n"
"  const uint type1 = (header >> (GBUFFER_HEADER_BITS_PER_BIN * 1u)) & 15u;\n"
"  switch (bin_id) {\n"
"    case 2u:\n"
"      return uint(type0 != 0u) + uint(type1 != 0u);\n"
"    case 1u:\n"
"      return uint(type0 != 0u);\n"
"    default:\n"
"      return 0u;\n"
"  }\n"
"}\n"
"inline uint gbuffer_tangent_space_id(uint header, uint layer_id)\n"
"{\n"
"  if (layer_id == 0u) {\n"
"    return 0u;\n"
"  }\n"
"  return 3u & (header >> ((12u - 2u) + layer_id * 2u));\n"
"}\n"
"inline bool gbuffer_has_additional_data(uint header)\n"
"{\n"
"  const uint transmission_mask = (GBUF_TRANSMISSION_BIT << (GBUFFER_HEADER_BITS_PER_BIN * 0u)) |\n"
"                                 (GBUF_TRANSMISSION_BIT << (GBUFFER_HEADER_BITS_PER_BIN * 1u)) |\n"
"                                 (GBUF_TRANSMISSION_BIT << (GBUFFER_HEADER_BITS_PER_BIN * 2u));\n"
"  return (header & transmission_mask) != 0u;\n"
"}\n"
"inline ThicknessData thickness_unpack(float thickness_packed)\n"
"{\n"
"  ThicknessData thickness;\n"
"  float value = (thickness_packed > 0.5f) ? (1.0f - thickness_packed) : thickness_packed;\n"
"  value = value / max(1.0f - 2.0f * value, 1.0e-8f);\n"
"  thickness.value = value;\n"
"  thickness.sphere_mode = (thickness_packed <= 0.5f);\n"
"  return thickness;\n"
"}\n"
"inline bool load_gbuffer_surface_normal(int2 texel,\n"
"                                        uint header,\n"
"                                        uint closure_index,\n"
"                                        texture2d_array<float, access::read> gbuf_normal_tx,\n"
"                                        thread float3 &r_N)\n"
"{\n"
"  if (header == 0u) {\n"
"    return false;\n"
"  }\n"
"  const uint layer_id = gbuffer_bin_to_layer(header, closure_index);\n"
"  const uint normal_id = gbuffer_tangent_space_id(header, layer_id);\n"
"  const float2 packed_N = gbuf_normal_tx.read(uint2(texel), normal_id).xy;\n"
"  r_N = normal_unpack(packed_N);\n"
"  return all(isfinite(r_N)) && (dot(r_N, r_N) > 1.0e-10f);\n"
"}\n"
"inline bool load_gbuffer_thickness(int2 texel,\n"
"                                   uint header,\n"
"                                   texture2d_array<float, access::read> gbuf_normal_tx,\n"
"                                   thread ThicknessData &r_thickness)\n"
"{\n"
"  if (!gbuffer_has_additional_data(header) || gbuf_normal_tx.get_array_size() == 0) {\n"
"    return false;\n"
"  }\n"
"  const uint additional_layer = gbuf_normal_tx.get_array_size() - 1;\n"
"  r_thickness = thickness_unpack(gbuf_normal_tx.read(uint2(texel), additional_layer).x);\n"
"  return r_thickness.value > 0.0f;\n"
"}\n"
"inline float3 thickness_intersection_offset(ThicknessData thickness, float3 N, float3 L)\n"
"{\n"
"  const float cos_alpha = dot(L, -N);\n"
"  if (!(cos_alpha > 1.0e-5f)) {\n"
"    return float3(0.0f);\n"
"  }\n"
"  if (thickness.sphere_mode) {\n"
"    return L * (cos_alpha * thickness.value);\n"
"  }\n"
"  return L * (thickness.value / cos_alpha);\n"
"}\n"
"inline float hwrt_specular_ray_epsilon(bool thin_refraction)\n"
"{\n"
"  /* Thin glass shells in asset space can be smaller than the historical 1e-3 launch bias after\n"
"   * object scaling. Keep the larger guard for mirror-only traces, but let refraction traverse\n"
"   * real exit faces instead of stepping over them. */\n"
"  return thin_refraction ? 1.0e-5f : 1.0e-3f;\n"
"}\n"
"inline float hwrt_specular_ray_tmin(bool thin_refraction)\n"
"{\n"
"  return thin_refraction ? 1.0e-5f : 5.0e-4f;\n"
"}\n"
"inline float hwrt_gi_ray_epsilon(float reference_distance)\n"
"{\n"
"  /* Nuru: GI rays need a scale-relative epsilon. Fixed world-unit offsets become huge when an\n"
"   * asset is scaled down and can skip sealed wall blockers at edges. Keep this far below typical\n"
"   * wall thickness while still large enough to avoid exact self-intersection. */\n"
"  return clamp(max(reference_distance, 1.0e-4f) * 1.25e-6f, 1.0e-8f, 2.0e-5f);\n"
"}\n"
"inline float hwrt_gi_self_hit_distance(float epsilon)\n"
"{\n"
"  return max(epsilon * 16.0f, 1.0e-7f);\n"
"}\n"
"inline float2 direction_pack(float3 dir)\n"
"{\n"
"  const float dir_len_sq = dot(dir, dir);\n"
"  if (!(dir_len_sq > 1.0e-10f)) {\n"
"    return float2(0.5f, 0.5f);\n"
"  }\n"
"  dir *= rsqrt(dir_len_sq);\n"
"  dir /= max(fabs(dir.x) + fabs(dir.y) + fabs(dir.z), 1.0e-8f);\n"
"  float2 packed = dir.xy;\n"
"  if (dir.z < 0.0f) {\n"
"    const float2 sign_dir = float2((packed.x >= 0.0f) ? 1.0f : -1.0f,\n"
"                                   (packed.y >= 0.0f) ? 1.0f : -1.0f);\n"
"    packed = (1.0f - abs(float2(packed.y, packed.x))) * sign_dir;\n"
"  }\n"
"  return packed * 0.5f + 0.5f;\n"
"}\n"
         "inline float3 estimate_world_normal(int2 texel, float depth, depth2d<float, access::sample> depth_tx, constant HardwareShadowUniforms &u)\n"
         "{\n"
         "  const float3 P = point_screen_to_world(texel, depth, u);\n"
         "  const float depth_px = sample_depth_clamped(texel + int2(1, 0), depth_tx, u);\n"
         "  const float depth_nx = sample_depth_clamped(texel + int2(-1, 0), depth_tx, u);\n"
         "  const float depth_py = sample_depth_clamped(texel + int2(0, 1), depth_tx, u);\n"
         "  const float depth_ny = sample_depth_clamped(texel + int2(0, -1), depth_tx, u);\n"
         "  float3 dPdx = float3(0.0f);\n"
         "  float3 dPdy = float3(0.0f);\n"
         "  if (depth_is_valid(depth_px) && depth_is_valid(depth_nx)) {\n"
         "    const bool use_pos = fabs(depth_px - depth) < fabs(depth_nx - depth);\n"
         "    const float3 Pn = point_screen_to_world(texel + (use_pos ? int2(1, 0) : int2(-1, 0)), use_pos ? depth_px : depth_nx, u);\n"
         "    dPdx = use_pos ? (Pn - P) : (P - Pn);\n"
         "  }\n"
         "  else if (depth_is_valid(depth_px)) {\n"
         "    dPdx = point_screen_to_world(texel + int2(1, 0), depth_px, u) - P;\n"
         "  }\n"
         "  else if (depth_is_valid(depth_nx)) {\n"
         "    dPdx = P - point_screen_to_world(texel + int2(-1, 0), depth_nx, u);\n"
         "  }\n"
         "  if (depth_is_valid(depth_py) && depth_is_valid(depth_ny)) {\n"
         "    const bool use_pos = fabs(depth_py - depth) < fabs(depth_ny - depth);\n"
         "    const float3 Pn = point_screen_to_world(texel + (use_pos ? int2(0, 1) : int2(0, -1)), use_pos ? depth_py : depth_ny, u);\n"
         "    dPdy = use_pos ? (Pn - P) : (P - Pn);\n"
         "  }\n"
         "  else if (depth_is_valid(depth_py)) {\n"
         "    dPdy = point_screen_to_world(texel + int2(0, 1), depth_py, u) - P;\n"
         "  }\n"
         "  else if (depth_is_valid(depth_ny)) {\n"
         "    dPdy = P - point_screen_to_world(texel + int2(0, -1), depth_ny, u);\n"
         "  }\n"
         "  if (dot(dPdx, dPdx) <= 1.0e-16f || dot(dPdy, dPdy) <= 1.0e-16f) {\n"
         "    return float3(0.0f);\n"
         "  }\n"
         "  float3 N = cross(dPdx, dPdy);\n"
         "  const float len_sq = dot(N, N);\n"
         "  if (!(len_sq > 1.0e-16f)) {\n"
         "    return float3(0.0f);\n"
         "  }\n"
         "  N *= rsqrt(len_sq);\n"
         "  if (dot(N, u.light_direction_bias.xyz) < 0.0f) {\n"
         "    N = -N;\n"
         "  }\n"
         "  return N;\n"
         "}\n"
         "inline float3 estimate_world_normal(int2 texel, float depth, depth2d<float, access::sample> depth_tx, constant HardwareLocalShadowUniforms &u)\n"
         "{\n"
         "  const float3 P = point_screen_to_world(texel, depth, u);\n"
         "  const float depth_px = sample_depth_clamped(texel + int2(1, 0), depth_tx, u);\n"
         "  const float depth_nx = sample_depth_clamped(texel + int2(-1, 0), depth_tx, u);\n"
         "  const float depth_py = sample_depth_clamped(texel + int2(0, 1), depth_tx, u);\n"
         "  const float depth_ny = sample_depth_clamped(texel + int2(0, -1), depth_tx, u);\n"
         "  float3 dPdx = float3(0.0f);\n"
         "  float3 dPdy = float3(0.0f);\n"
         "  if (depth_is_valid(depth_px) && depth_is_valid(depth_nx)) {\n"
         "    const bool use_pos = fabs(depth_px - depth) < fabs(depth_nx - depth);\n"
         "    const float3 Pn = point_screen_to_world(texel + (use_pos ? int2(1, 0) : int2(-1, 0)), use_pos ? depth_px : depth_nx, u);\n"
         "    dPdx = use_pos ? (Pn - P) : (P - Pn);\n"
         "  }\n"
         "  else if (depth_is_valid(depth_px)) {\n"
         "    dPdx = point_screen_to_world(texel + int2(1, 0), depth_px, u) - P;\n"
         "  }\n"
         "  else if (depth_is_valid(depth_nx)) {\n"
         "    dPdx = P - point_screen_to_world(texel + int2(-1, 0), depth_nx, u);\n"
         "  }\n"
         "  if (depth_is_valid(depth_py) && depth_is_valid(depth_ny)) {\n"
         "    const bool use_pos = fabs(depth_py - depth) < fabs(depth_ny - depth);\n"
         "    const float3 Pn = point_screen_to_world(texel + (use_pos ? int2(0, 1) : int2(0, -1)), use_pos ? depth_py : depth_ny, u);\n"
         "    dPdy = use_pos ? (Pn - P) : (P - Pn);\n"
         "  }\n"
         "  else if (depth_is_valid(depth_py)) {\n"
         "    dPdy = point_screen_to_world(texel + int2(0, 1), depth_py, u) - P;\n"
         "  }\n"
         "  else if (depth_is_valid(depth_ny)) {\n"
         "    dPdy = P - point_screen_to_world(texel + int2(0, -1), depth_ny, u);\n"
         "  }\n"
         "  if (dot(dPdx, dPdx) <= 1.0e-16f || dot(dPdy, dPdy) <= 1.0e-16f) {\n"
         "    return float3(0.0f);\n"
         "  }\n"
         "  float3 N = cross(dPdx, dPdy);\n"
         "  const float len_sq = dot(N, N);\n"
         "  if (!(len_sq > 1.0e-16f)) {\n"
         "    return float3(0.0f);\n"
         "  }\n"
         "  N *= rsqrt(len_sq);\n"
         "  if (dot(N, (u.light_position_radius.xyz + u.shadow_offset_scale.xyz) - P) < 0.0f) {\n"
         "    N = -N;\n"
         "  }\n"
         "  return N;\n"
         "}\n"
         "inline float3 estimate_world_normal(int2 texel, float depth, depth2d<float, access::sample> depth_tx, constant HardwareEnvironmentVisibilityUniforms &u)\n"
         "{\n"
         "  const float3 P = point_screen_to_world(texel, depth, u);\n"
         "  const float depth_px = sample_depth_clamped(texel + int2(1, 0), depth_tx, u);\n"
         "  const float depth_nx = sample_depth_clamped(texel + int2(-1, 0), depth_tx, u);\n"
         "  const float depth_py = sample_depth_clamped(texel + int2(0, 1), depth_tx, u);\n"
         "  const float depth_ny = sample_depth_clamped(texel + int2(0, -1), depth_tx, u);\n"
         "  float3 dPdx = float3(0.0f);\n"
         "  float3 dPdy = float3(0.0f);\n"
         "  if (depth_is_valid(depth_px) && depth_is_valid(depth_nx)) {\n"
         "    const bool use_pos = fabs(depth_px - depth) < fabs(depth_nx - depth);\n"
         "    const float3 Pn = point_screen_to_world(texel + (use_pos ? int2(1, 0) : int2(-1, 0)), use_pos ? depth_px : depth_nx, u);\n"
         "    dPdx = use_pos ? (Pn - P) : (P - Pn);\n"
         "  }\n"
         "  else if (depth_is_valid(depth_px)) {\n"
         "    dPdx = point_screen_to_world(texel + int2(1, 0), depth_px, u) - P;\n"
         "  }\n"
         "  else if (depth_is_valid(depth_nx)) {\n"
         "    dPdx = P - point_screen_to_world(texel + int2(-1, 0), depth_nx, u);\n"
         "  }\n"
         "  if (depth_is_valid(depth_py) && depth_is_valid(depth_ny)) {\n"
         "    const bool use_pos = fabs(depth_py - depth) < fabs(depth_ny - depth);\n"
         "    const float3 Pn = point_screen_to_world(texel + (use_pos ? int2(0, 1) : int2(0, -1)), use_pos ? depth_py : depth_ny, u);\n"
         "    dPdy = use_pos ? (Pn - P) : (P - Pn);\n"
         "  }\n"
         "  else if (depth_is_valid(depth_py)) {\n"
         "    dPdy = point_screen_to_world(texel + int2(0, 1), depth_py, u) - P;\n"
         "  }\n"
         "  else if (depth_is_valid(depth_ny)) {\n"
         "    dPdy = P - point_screen_to_world(texel + int2(0, -1), depth_ny, u);\n"
         "  }\n"
         "  if (dot(dPdx, dPdx) <= 1.0e-16f || dot(dPdy, dPdy) <= 1.0e-16f) {\n"
         "    return float3(0.0f);\n"
         "  }\n"
         "  float3 N = cross(dPdx, dPdy);\n"
         "  const float len_sq = dot(N, N);\n"
         "  if (!(len_sq > 1.0e-16f)) {\n"
         "    return float3(0.0f);\n"
         "  }\n"
         "  N *= rsqrt(len_sq);\n"
         "  const float3 camera_pos = u.viewinv[3].xyz;\n"
         "  if (dot(N, camera_pos - P) < 0.0f) {\n"
         "    N = -N;\n"
         "  }\n"
         "  return N;\n"
         "}\n"
         "inline bool is_area_light(uint type)\n"
         "{\n"
         "  return type >= LIGHT_RECT;\n"
         "}\n"
         "inline bool is_sphere_light(uint type)\n"
         "{\n"
         "  return type == LIGHT_OMNI_SPHERE || type == LIGHT_SPOT_SPHERE;\n"
         "}\n"
         "inline float hash12(float2 p)\n"
         "{\n"
         "  return fract(sin(dot(p, float2(127.1f, 311.7f))) * 43758.5453123f);\n"
         "}\n"
         "inline float2 rand2(uint2 tid, int sample_index, int layer)\n"
         "{\n"
         "  const float2 base = float2(float(tid.x), float(tid.y)) + float2(float(layer) * 13.17f, float(sample_index) * 19.31f);\n"
         "  return float2(hash12(base + float2(0.17f, 0.31f)), hash12(base.yx + float2(0.73f, 0.53f)));\n"
         "}\n"
         "inline float2 rand2_shadow(uint2 tid, int sample_index, int layer, float4 sampling_rand)\n"
         "{\n"
         "  const float2 seed = float2(sampling_rand.x * 23.47f + sampling_rand.z * 11.13f,\n"
         "                              sampling_rand.y * 29.59f + sampling_rand.w * 7.71f);\n"
         "  const float2 base = float2(float(tid.x), float(tid.y)) + float2(float(layer) * 13.17f, float(sample_index) * 19.31f) + seed;\n"
         "  return float2(hash12(base + float2(0.17f, 0.31f)), hash12(base.yx + float2(0.73f, 0.53f)));\n"
         "}\n"
         "inline float2 rand2_trace(uint2 tid, int sample_index, int layer, constant HardwareTraceUniforms &u)\n"
         "{\n"
         "  const float2 seed = float2(u.sampling_rand.x * 23.47f + u.sampling_rand.z * 11.13f,\n"
         "                              u.sampling_rand.y * 29.59f + u.sampling_rand.w * 7.71f);\n"
         "  const float2 base = float2(float(tid.x), float(tid.y)) + float2(float(layer) * 13.17f, float(sample_index) * 19.31f) + seed;\n"
         "  return float2(hash12(base + float2(0.17f, 0.31f)), hash12(base.yx + float2(0.73f, 0.53f)));\n"
         "}\n"
         "inline float2 sample_circle(float rand)\n"
         "{\n"
         "  const float phi = (rand - 0.5f) * 6.28318530718f;\n"
         "  return float2(cos(phi), sin(phi));\n"
         "}\n"
         "inline float2 sample_disk(float2 rand)\n"
         "{\n"
         "  return sample_circle(rand.y) * sqrt(rand.x);\n"
         "}\n"
         "inline float3 sample_cylinder(float2 rand)\n"
         "{\n"
         "  return float3(rand.x, sample_circle(rand.y));\n"
         "}\n"
         "inline float3 ggx_sample_vndf(float3 rand, float3 Vt, float alpha)\n"
         "{\n"
         "  const float3 Vh = normalize(float3(alpha * Vt.xy, Vt.z));\n"
         "  const float cos_theta = mix(-Vh.z, 1.0f, rand.x);\n"
         "  const float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));\n"
         "  const float3 Lh = float3(sin_theta * rand.yz, cos_theta);\n"
         "  const float3 Hh = Vh + Lh;\n"
         "  return normalize(float3(alpha * Hh.xy, max(0.0f, Hh.z)));\n"
         "}\n"
         "inline void make_orthonormal_basis(float3 n, thread float3 &right, thread float3 &up)\n"
         "{\n"
         "  const float3 helper = (fabs(n.z) < 0.999f) ? float3(0.0f, 0.0f, 1.0f) : float3(0.0f, 1.0f, 0.0f);\n"
         "  right = normalize(cross(helper, n));\n"
         "  up = normalize(cross(n, right));\n"
         "}\n"
"inline float2 octahedral_uv_from_direction(float3 co);\n"
"inline float receiver_caustic_hash(uint2 tid, int sample_index, float2 offset, constant HardwareReceiverCausticUniforms &u)\n"
"{\n"
"  const float2 seed = float2(u.sampling_rand.x * 23.47f + u.sampling_rand.z * 11.13f,\n"
"                              u.sampling_rand.y * 29.59f + u.sampling_rand.w * 7.71f);\n"
"  const float2 base = float2(tid) + float2(float(sample_index) * 19.31f) + offset + seed;\n"
"  return hash12(base);\n"
"}\n"
"inline float3 sample_receiver_caustic_world_direction(uint2 tid,\n"
"                                                      int sample_index,\n"
"                                                      float3 N,\n"
"                                                      constant HardwareReceiverCausticUniforms &u)\n"
"{\n"
"  float3 right, up;\n"
"  make_orthonormal_basis(N, right, up);\n"
"  const float2 xi = float2(receiver_caustic_hash(tid, sample_index, float2(0.21f, 0.79f), u),\n"
"                           receiver_caustic_hash(tid, sample_index, float2(0.57f, 0.33f), u));\n"
"  const float2 disk = sample_disk(xi);\n"
"  const float z = sqrt(max(0.0f, 1.0f - dot(disk, disk)));\n"
"  return normalize(right * disk.x + up * disk.y + N * z);\n"
"}\n"
"inline float3 sample_receiver_caustic_world_radiance(texture2d_array<float, access::sample> world_probe_tx,\n"
"                                                     float3 direction,\n"
"                                                     constant HardwareReceiverCausticUniforms &u)\n"
"{\n"
"  if (u.world_probe_atlas_coord.w < 0.0f) {\n"
"    return float3(0.0f);\n"
"  }\n"
"  constexpr sampler linear_sampler(coord::normalized, address::clamp_to_edge, filter::linear);\n"
"  const float2 oct = octahedral_uv_from_direction(direction);\n"
"  const float mip_level = 0.0f;\n"
"  const float2 atlas_uv = u.world_probe_atlas_coord.xy + oct * u.world_probe_atlas_coord.z;\n"
"  return max(world_probe_tx.sample(linear_sampler, atlas_uv, uint(u.world_probe_atlas_coord.w), level(mip_level)).rgb,\n"
"             float3(0.0f));\n"
"}\n"
         "inline float3 sample_trace_diffuse_direction(uint2 tid,\n"
         "                                             int sample_index,\n"
         "                                             int layer,\n"
         "                                             float3 N,\n"
         "                                             constant HardwareTraceUniforms &u)\n"
         "{\n"
         "  float3 right, up;\n"
         "  make_orthonormal_basis(N, right, up);\n"
         "  const float2 disk = sample_disk(rand2_trace(tid, sample_index, layer, u));\n"
         "  const float z = sqrt(max(0.0f, 1.0f - dot(disk, disk)));\n"
         "  return normalize(right * disk.x + up * disk.y + N * z);\n"
         "}\n"
         "inline float3 sample_rough_specular_direction(uint2 tid,\n"
         "                                             int sample_index,\n"
         "                                             int layer,\n"
         "                                             float3 ray_direction,\n"
         "                                             float3 surface_N,\n"
         "                                             float roughness,\n"
         "                                             bool refract_mode,\n"
         "                                             float eta,\n"
         "                                             constant HardwareTraceUniforms &u)\n"
         "{\n"
         "  const float alpha = roughness * roughness;\n"
         "  float3 sharp_dir = refract_mode ? refract(ray_direction, surface_N, eta) :\n"
         "                                    reflect(ray_direction, surface_N);\n"
         "  if (refract_mode && !(dot(sharp_dir, sharp_dir) > 1.0e-10f)) {\n"
         "    sharp_dir = reflect(ray_direction, surface_N);\n"
         "  }\n"
         "  if (!(alpha > 4.0e-4f)) {\n"
         "    return sharp_dir;\n"
         "  }\n"
         "  float3 right, up;\n"
         "  make_orthonormal_basis(surface_N, right, up);\n"
         "  const float3 V = -ray_direction;\n"
         "  const float3 Vt = float3(dot(V, right), dot(V, up), dot(V, surface_N));\n"
         "  if (!(Vt.z > 1.0e-5f)) {\n"
         "    return sharp_dir;\n"
         "  }\n"
         "  const float3 Ht = ggx_sample_vndf(sample_cylinder(rand2_trace(tid, sample_index, layer, u)), Vt, alpha);\n"
         "  const float3 H = normalize(right * Ht.x + up * Ht.y + surface_N * Ht.z);\n"
         "  float3 sampled_dir = refract_mode ? refract(ray_direction, H, eta) : reflect(ray_direction, H);\n"
         "  if (refract_mode && !(dot(sampled_dir, sampled_dir) > 1.0e-10f)) {\n"
         "    sampled_dir = reflect(ray_direction, H);\n"
         "  }\n"
         "  return sampled_dir;\n"
         "}\n"
         "inline float dielectric_fresnel_reflectance(float3 ray_direction, float3 surface_N, float ior)\n"
         "{\n"
         "  const float f0 = pow((ior - 1.0f) / (ior + 1.0f), 2.0f);\n"
         "  const float cos_theta = clamp(dot(-ray_direction, surface_N), 0.0f, 1.0f);\n"
         "  const float f = pow(1.0f - cos_theta, 5.0f);\n"
         "  return clamp(f0 + (1.0f - f0) * f, 0.0f, 1.0f);\n"
         "}\n"
         "inline float projected_sphere_disk_radius(float sphere_radius, float distance_to_sphere)\n"
         "{\n"
         "  return sphere_radius * rsqrt(max(1.0e-8f, 1.0f - (sphere_radius * sphere_radius) / max(distance_to_sphere * distance_to_sphere, 1.0e-8f)));\n"
         "}\n"
         "inline float2 sample_directional_shadow_disk(uint2 tid, int sample_index, int sample_count, int layer, float4 sampling_rand)\n"
         "{\n"
         "  const int safe_sample_count = max(sample_count, 1);\n"
         "  const float2 rand = rand2_shadow(tid, sample_index, layer, sampling_rand);\n"
         "  const float radius = sqrt((float(sample_index) + rand.x) / float(safe_sample_count));\n"
         "  const float angle = 6.28318530718f * fract(rand.y + 0.61803398875f * float(sample_index));\n"
         "  return float2(cos(angle), sin(angle)) * radius;\n"
         "}\n"
"inline float3 directional_shadow_light_direction(constant HardwareShadowUniforms &u,\n"
"                                                 constant float4 *world_sunlight_direction)\n"
"{\n"
"  if (u.world_sun_slot_pad.x >= 0) {\n"
"    const float4 packed_direction = world_sunlight_direction[u.world_sun_slot_pad.x];\n"
"    if (packed_direction.w > 0.0f && all(isfinite(packed_direction.xyz)) &&\n"
"        dot(packed_direction.xyz, packed_direction.xyz) > 1.0e-10f)\n"
"    {\n"
"      return normalize(packed_direction.xyz);\n"
"    }\n"
"  }\n"
"  return u.light_direction_bias.xyz;\n"
"}\n"
"inline float3 sample_directional_shadow_direction(uint2 tid,\n"
"                                                  int sample_index,\n"
"                                                  constant HardwareShadowUniforms &u,\n"
"                                                  constant float4 *world_sunlight_direction)\n"
         "{\n"
"  const float3 light_direction = directional_shadow_light_direction(u, world_sunlight_direction);\n"
         "  if (!(u.shadow_params.x > 1.0e-6f)) {\n"
"    return light_direction;\n"
         "  }\n"
         "  float3 right, up;\n"
"  make_orthonormal_basis(light_direction, right, up);\n"
         "  const int sample_count = max(int(u.shadow_params.y), 1);\n"
         "  const float2 disk = sample_directional_shadow_disk(\n"
         "      tid, sample_index, sample_count, u.resolution_layer.z, u.sampling_rand) *\n"
         "                      tan(u.shadow_params.x);\n"
"  return normalize(light_direction + right * disk.x + up * disk.y);\n"
         "}\n"
         "inline float3 sample_local_shadow_target(uint2 tid, int sample_index, float3 P, constant HardwareLocalShadowUniforms &u)\n"
         "{\n"
         "  const float3 center = u.light_position_radius.xyz + u.shadow_offset_scale.xyz;\n"
         "  if (is_area_light(uint(u.resolution_layer_type.w))) {\n"
         "    float2 rand = rand2_shadow(tid, sample_index, u.resolution_layer_type.z, u.sampling_rand);\n"
         "    if (uint(u.resolution_layer_type.w) == LIGHT_RECT) {\n"
         "      rand = rand * 2.0f - 1.0f;\n"
         "    }\n"
         "    else {\n"
         "      rand = sample_disk(rand);\n"
         "    }\n"
         "    rand *= float2(u.light_x_axis_size_x.w, u.light_y_axis_size_y.w) * u.shadow_offset_scale.w;\n"
         "    return center + u.light_x_axis_size_x.xyz * rand.x + u.light_y_axis_size_y.xyz * rand.y;\n"
         "  }\n"
         "  float3 L = center - P;\n"
         "  const float distance_to_light = length(L);\n"
         "  if (!(distance_to_light > 1.0e-5f)) {\n"
         "    return center;\n"
         "  }\n"
         "  L /= distance_to_light;\n"
         "  float radius = u.light_position_radius.w;\n"
         "  if (is_sphere_light(uint(u.resolution_layer_type.w))) {\n"
         "    radius = projected_sphere_disk_radius(radius, distance_to_light);\n"
         "  }\n"
         "  if (!(radius > 1.0e-6f)) {\n"
         "    return center;\n"
         "  }\n"
         "  float3 right, up;\n"
         "  make_orthonormal_basis(L, right, up);\n"
         "  const float2 disk = sample_disk(rand2_shadow(tid, sample_index, u.resolution_layer_type.z, u.sampling_rand)) * radius;\n"
         "  return center + right * disk.x + up * disk.y;\n"
         "}\n"
         "inline float3 sample_environment_visibility_direction(uint2 tid, int sample_index, float3 N, constant HardwareEnvironmentVisibilityUniforms &u)\n"
         "{\n"
         "  float3 right, up;\n"
         "  make_orthonormal_basis(N, right, up);\n"
         "  const float2 disk = sample_disk(rand2_shadow(tid, sample_index, u.resolution_samples.w, u.sampling_rand));\n"
         "  const float z = sqrt(max(0.0f, 1.0f - dot(disk, disk)));\n"
         "  return normalize(right * disk.x + up * disk.y + N * z);\n"
         "}\n"
"inline bool fast_gi_is_sun(uint type)\n"
"{\n"
"  return type <= LIGHT_SUN_ORTHO;\n"
"}\n"
"inline bool fast_gi_is_spot(uint type)\n"
"{\n"
"  return type == LIGHT_SPOT_SPHERE || type == LIGHT_SPOT_DISK;\n"
"}\n"
"inline bool fast_gi_is_area(uint type)\n"
"{\n"
"  return type >= LIGHT_RECT;\n"
"}\n"
"inline bool fast_gi_is_sphere(uint type)\n"
"{\n"
"  return type == LIGHT_OMNI_SPHERE || type == LIGHT_SPOT_SPHERE;\n"
"}\n"
"inline float3 fast_gi_transform_location(FastGILightRecord light)\n"
"{\n"
"  return float3(light.object_to_world_x.w, light.object_to_world_y.w, light.object_to_world_z.w);\n"
"}\n"
"inline float3 fast_gi_transform_z_axis(FastGILightRecord light)\n"
"{\n"
"  return float3(light.object_to_world_x.z, light.object_to_world_y.z, light.object_to_world_z.z);\n"
"}\n"
"inline float3 fast_gi_transform_direction_transposed(FastGILightRecord light, float3 direction)\n"
"{\n"
"  return float3x3(float3(light.object_to_world_x.x, light.object_to_world_x.y, light.object_to_world_x.z),\n"
"                  float3(light.object_to_world_y.x, light.object_to_world_y.y, light.object_to_world_y.z),\n"
"                  float3(light.object_to_world_z.x, light.object_to_world_z.y, light.object_to_world_z.z)) * direction;\n"
"}\n"
"inline float fast_gi_light_influence_attenuation(float dist, float inv_sqr_influence)\n"
"{\n"
"  const float factor = dist * dist * inv_sqr_influence;\n"
"  const float fac = saturate(1.0f - factor * factor);\n"
"  return fac * fac;\n"
"}\n"
"inline float fast_gi_light_spot_attenuation(FastGILightRecord light, float3 L)\n"
"{\n"
"  const float3 lL = fast_gi_transform_direction_transposed(light, L);\n"
"  if (!(lL.z > 0.0f)) {\n"
"    return 0.0f;\n"
"  }\n"
"  const float inv_z = 1.0f / max(lL.z, 1.0e-6f);\n"
"  const float2 scaled = lL.xy * light.spot_size_inv.xy * inv_z;\n"
"  const float ellipse = rsqrt(1.0f + dot(scaled, scaled));\n"
"  return smoothstep(0.0f, 1.0f, ellipse * light.attenuation_spot.z + light.attenuation_spot.w);\n"
"}\n"
"inline float fast_gi_light_surface_attenuation(FastGILightRecord light, uint type, float3 L, float dist)\n"
"{\n"
"  if (fast_gi_is_sun(type)) {\n"
"    return 1.0f;\n"
"  }\n"
"  float attenuation = fast_gi_is_spot(type) ? fast_gi_light_spot_attenuation(light, L) : 1.0f;\n"
"  attenuation *= fast_gi_light_influence_attenuation(dist, light.attenuation_spot.y);\n"
"  if (fast_gi_is_area(type)) {\n"
"    attenuation *= float(dot(L, fast_gi_transform_z_axis(light)) > 0.0f);\n"
"  }\n"
"  return attenuation;\n"
"}\n"
"inline float fast_gi_light_point_power(FastGILightRecord light, uint type, float dist, float3 L)\n"
"{\n"
"  if (fast_gi_is_sun(type)) {\n"
"    return 1.0f;\n"
"  }\n"
"  /* Nuru: Eevee local power[LIGHT_DIFFUSE] bakes a 1/r^2 shape normalization that the analytic\n"
"   * LTC eval cancels with its solid-angle term (~ r^2/d^2 for small sources). The previous\n"
"   * 1/d^2-only form left 1/r^2 uncancelled and exploded small-radius lights (~100x for the\n"
"   * default 0.1 m point) the moment locals joined the gather NEE. Use the Lambert small-source\n"
"   * approximation solid_angle/pi = 2*(1 - sqrt(1 - (r/d)^2)), capped inside the source. */\n"
"  float radius = max(light.attenuation_spot.x, 1.0e-4f);\n"
"  if (fast_gi_is_sphere(type) && dist > 1.0e-5f) {\n"
"    radius = projected_sphere_disk_radius(radius, dist);\n"
"  }\n"
"  const float x = clamp(radius / max(dist, 1.0e-5f), 0.0f, 1.0f);\n"
"  float power = 2.0f * (1.0f - sqrt(max(1.0f - x * x, 0.0f)));\n"
"  if (fast_gi_is_area(type)) {\n"
"    power *= saturate(dot(fast_gi_transform_z_axis(light), L));\n"
"  }\n"
"  return power;\n"
"}\n"
"/* --- Nuru light tree (Stage A many-light sampling). Classic published light-hierarchy\n"
" * importance sampling (Conty&Kulla 2018 / Yuksel stochastic lightcuts descent). NO reservoirs,\n"
" * NO spatiotemporal sample reuse (patent fence). Node encoding documented in\n"
" * GPU_nuru_hardware_raytrace.hh; nodes ride in light_buf after the light records. --- */\n"
"struct LightTreeNode {\n"
"  float3 center; float radius;\n"
"  float3 axis; float cos_theta; /* -2 = omnidirectional */\n"
"  float power; int left; int right; int light; /* light >= 0 -> leaf record index */\n"
"  int cluster;\n"
"};\n"
"/* --- Nuru NIS stage G1: trained cluster-multiplier network, MSL mirror of\n"
" * eevee_nuru_nis_mlp_lib.glsl. Weights layout and architecture MUST stay in sync:\n"
" * 24 (frequency-encoded position) -> 32 ReLU -> 32 ReLU -> 32 logits;\n"
" * [W1 24x32][b1 32][W2 32x32][b2 32][W3 32x32][b3 32], row-major out*in+in.\n"
" * Multipliers are exp-clamped strictly positive: learned shaping can only redistribute\n"
" * samples between clusters, never zero the PMF support (unbiasedness guard). --- */\n"
"constant constexpr int NIS_IN = 24;\n"
"constant constexpr int NIS_HIDDEN = 32;\n"
"constant constexpr int NIS_OUT = 32;\n"
"constant constexpr int NIS_W1 = 0;\n"
"constant constexpr int NIS_B1 = NIS_W1 + NIS_IN * NIS_HIDDEN;\n"
"constant constexpr int NIS_W2 = NIS_B1 + NIS_HIDDEN;\n"
"constant constexpr int NIS_B2 = NIS_W2 + NIS_HIDDEN * NIS_HIDDEN;\n"
"constant constexpr int NIS_W3 = NIS_B2 + NIS_HIDDEN;\n"
"constant constexpr int NIS_B3 = NIS_W3 + NIS_HIDDEN * NIS_OUT;\n"
"inline void nis_cluster_multipliers(constant float *nis_weights,\n"
"                                    bool nis_enabled,\n"
"                                    float3 P,\n"
"                                    thread float *r_m)\n"
"{\n"
"  if (!nis_enabled) {\n"
"    for (int c = 0; c < NIS_OUT; c++) {\n"
"      r_m[c] = 1.0f;\n"
"    }\n"
"    return;\n"
"  }\n"
"  float encoded[NIS_IN];\n"
"  int write_index = 0;\n"
"  float frequency = 0.05f;\n"
"  for (int octave = 0; octave < 4; octave++) {\n"
"    const float3 phase = P * frequency;\n"
"    encoded[write_index++] = sin(phase.x);\n"
"    encoded[write_index++] = sin(phase.y);\n"
"    encoded[write_index++] = sin(phase.z);\n"
"    encoded[write_index++] = cos(phase.x);\n"
"    encoded[write_index++] = cos(phase.y);\n"
"    encoded[write_index++] = cos(phase.z);\n"
"    frequency *= 4.0f;\n"
"  }\n"
"  float h1[NIS_HIDDEN];\n"
"  for (int j = 0; j < NIS_HIDDEN; j++) {\n"
"    float acc = nis_weights[NIS_B1 + j];\n"
"    for (int i = 0; i < NIS_IN; i++) {\n"
"      acc += nis_weights[NIS_W1 + j * NIS_IN + i] * encoded[i];\n"
"    }\n"
"    h1[j] = max(acc, 0.0f);\n"
"  }\n"
"  float h2[NIS_HIDDEN];\n"
"  for (int j = 0; j < NIS_HIDDEN; j++) {\n"
"    float acc = nis_weights[NIS_B2 + j];\n"
"    for (int i = 0; i < NIS_HIDDEN; i++) {\n"
"      acc += nis_weights[NIS_W2 + j * NIS_HIDDEN + i] * h1[i];\n"
"    }\n"
"    h2[j] = max(acc, 0.0f);\n"
"  }\n"
"  for (int j = 0; j < NIS_OUT; j++) {\n"
"    float acc = nis_weights[NIS_B3 + j];\n"
"    for (int i = 0; i < NIS_HIDDEN; i++) {\n"
"      acc += nis_weights[NIS_W3 + j * NIS_HIDDEN + i] * h2[i];\n"
"    }\n"
"    r_m[j] = exp(clamp(acc, -4.0f, 4.0f));\n"
"  }\n"
"}\n"
"inline LightTreeNode light_tree_node_load(constant FastGILightRecord *light_buf,\n"
"                                          int light_count,\n"
"                                          int node_index)\n"
"{\n"
"  const FastGILightRecord rec = light_buf[light_count + node_index];\n"
"  LightTreeNode node;\n"
"  node.center = rec.object_to_world_x.xyz;\n"
"  node.radius = rec.object_to_world_x.w;\n"
"  node.axis = rec.object_to_world_y.xyz;\n"
"  node.cos_theta = rec.object_to_world_y.w;\n"
"  node.power = rec.object_to_world_z.x;\n"
"  node.left = int(rec.object_to_world_z.y);\n"
"  node.right = int(rec.object_to_world_z.z);\n"
"  node.light = int(rec.object_to_world_z.w);\n"
"  node.cluster = clamp(int(rec.spot_size_inv.z), 0, NIS_OUT - 1);\n"
"  return node;\n"
"}\n"
"inline float light_tree_cluster_importance(float3 P, float3 N, LightTreeNode node,\n"
"                                            thread const float *nis_m)\n"
"{\n"
"  const float3 to_c = node.center - P;\n"
"  const float d2 = max(dot(to_c, to_c), 1.0e-8f);\n"
"  const float d = sqrt(d2);\n"
"  const float3 dir = to_c / d;\n"
"  const float sin_u = clamp(node.radius / d, 0.0f, 1.0f);\n"
"  /* Receiver facing widened by the cluster's angular extent. Floored so the descent PDF stays\n"
"   * non-zero wherever a leaf could still contribute (unbiasedness guard). */\n"
"  const float facing = max(saturate(dot(N, dir) + sin_u), 0.01f);\n"
"  float emitter = 1.0f;\n"
"  if (node.cos_theta > -1.5f) {\n"
"    const float cos_e = dot(-dir, node.axis);\n"
"    emitter = max(saturate((cos_e - node.cos_theta) / max(1.0f - node.cos_theta, 1.0e-4f) + sin_u),\n"
"                  0.01f);\n"
"  }\n"
"  const float dist_sq = max(d2, 0.25f * node.radius * node.radius);\n"
"  return nis_m[node.cluster] * node.power * facing * emitter / max(dist_sq, 1.0e-6f);\n"
"}\n"
"/* Pick one light record: stochastic tree descent over the locals plus a top-level sun group.\n"
" * Returns the record index, or -1 when nothing is pickable; r_pick_pdf is the discrete pick\n"
" * probability for the 1/pdf compensation. One random pair, rescaled per level. */\n"
"inline int light_tree_sample_record(constant FastGILightRecord *light_buf,\n"
"                                    int light_count,\n"
"                                    int local_count,\n"
"                                    float3 P,\n"
"                                    float3 N,\n"
"                                    float2 rand,\n"
"                                    thread const float *nis_m,\n"
"                                    thread float &r_pick_pdf)\n"
"{\n"
"  r_pick_pdf = 1.0f;\n"
"  local_count = clamp(local_count, 0, light_count);\n"
"  const int sun_count = light_count - local_count;\n"
"  float sun_importance = 0.0f;\n"
"  const int sun_scan = min(sun_count, 8);\n"
"  for (int k = 0; k < sun_scan; k++) {\n"
"    const FastGILightRecord rec = light_buf[local_count + k];\n"
"    const float3 sun_L = normalize(-rec.direction_type.xyz);\n"
"    const float color_max = max(rec.color_diffuse_power.x,\n"
"                                max(rec.color_diffuse_power.y, rec.color_diffuse_power.z));\n"
"    sun_importance += rec.color_diffuse_power.w * max(color_max, 1.0e-4f) *\n"
"                      max(saturate(dot(N, sun_L)), 0.01f);\n"
"  }\n"
"  if (sun_count > sun_scan) {\n"
"    sun_importance *= float(sun_count) / float(max(sun_scan, 1));\n"
"  }\n"
"  float local_importance = 0.0f;\n"
"  if (local_count > 0) {\n"
"    local_importance = light_tree_cluster_importance(\n"
"        P, N, light_tree_node_load(light_buf, light_count, 0), nis_m);\n"
"  }\n"
"  const float group_sum = sun_importance + local_importance;\n"
"  if (!(group_sum > 1.0e-12f)) {\n"
"    return -1;\n"
"  }\n"
"  const float p_local = local_importance / group_sum;\n"
"  float r = clamp(rand.x, 0.0f, 0.999999f);\n"
"  if (r >= p_local) {\n"
"    if (sun_count <= 0) {\n"
"      return -1;\n"
"    }\n"
"    const float r_sun = (r - p_local) / max(1.0f - p_local, 1.0e-6f);\n"
"    const int sun_index = min(int(r_sun * float(sun_count)), sun_count - 1);\n"
"    r_pick_pdf = max(1.0f - p_local, 1.0e-6f) / float(sun_count);\n"
"    return local_count + sun_index;\n"
"  }\n"
"  r_pick_pdf = max(p_local, 1.0e-6f);\n"
"  float r_descend = clamp(rand.y, 0.0f, 0.999999f);\n"
"  int node_index = 0;\n"
"  for (int depth = 0; depth < 32; depth++) {\n"
"    const LightTreeNode node = light_tree_node_load(light_buf, light_count, node_index);\n"
"    if (node.light >= 0) {\n"
"      return node.light;\n"
"    }\n"
"    if (node.left < 0 || node.right < 0) {\n"
"      return -1;\n"
"    }\n"
"    const float imp_l = light_tree_cluster_importance(\n"
"        P, N, light_tree_node_load(light_buf, light_count, node.left), nis_m);\n"
"    const float imp_r = light_tree_cluster_importance(\n"
"        P, N, light_tree_node_load(light_buf, light_count, node.right), nis_m);\n"
"    const float pair_sum = imp_l + imp_r;\n"
"    const float p_l = (pair_sum > 1.0e-12f) ? (imp_l / pair_sum) : 0.5f;\n"
"    if (r_descend < p_l) {\n"
"      r_descend = r_descend / max(p_l, 1.0e-6f);\n"
"      r_pick_pdf *= max(p_l, 1.0e-6f);\n"
"      node_index = node.left;\n"
"    }\n"
"    else {\n"
"      r_descend = (r_descend - p_l) / max(1.0f - p_l, 1.0e-6f);\n"
"      r_pick_pdf *= max(1.0f - p_l, 1.0e-6f);\n"
"      node_index = node.right;\n"
"    }\n"
"  }\n"
"  return -1;\n"
"}\n"
"inline float3 sample_reflected_receiver_gi_direction(uint2 tid,\n"
"                                                    int sample_index,\n"
"                                                    float3 N,\n"
"                                                    constant HardwareReflectedReceiverGIUniforms &u)\n"
"{\n"
"  const float sample_count = float(max(u.resolution_samples.w, 1));\n"
"  const float2 seed = float2(float(tid.x), float(tid.y * 17u)) +\n"
"                      u.sampling_rand.xy * 37.0f + u.sampling_rand.zw * 11.0f;\n"
"  const float phi_offset = hash12(seed + float2(0.37f, 0.61f));\n"
"  const float sample_u = (float(sample_index) + 0.5f) / sample_count;\n"
"  const float z = mix(sqrt(max(0.0f, 1.0f - sample_u)), 1.0f - sample_u, 0.45f);\n"
"  const float phi = 2.39996322973f * (float(sample_index) + phi_offset * sample_count + 0.5f);\n"
"  const float r = sqrt(max(0.0f, 1.0f - z * z));\n"
"  float3 right, up;\n"
"  make_orthonormal_basis(N, right, up);\n"
"  return normalize(right * (cos(phi) * r) + up * (sin(phi) * r) + N * z);\n"
"}\n"
"inline float reflected_receiver_gi_hash(uint2 tid,\n"
"                                        int sample_index,\n"
"                                        float2 offset,\n"
"                                        constant HardwareReflectedReceiverGIUniforms &u)\n"
"{\n"
"  const float2 seed = float2(u.sampling_rand.x * 23.47f + u.sampling_rand.z * 11.13f,\n"
"                            u.sampling_rand.y * 29.59f + u.sampling_rand.w * 7.71f);\n"
"  const float2 base = float2(float(tid.x), float(tid.y * 17u));\n"
"  return hash12(base + offset + seed + float2(float(sample_index) * 0.07f));\n"
"}\n"
"inline float3 fast_gi_hit_normal(uint user_id,\n"
"                                 uint primitive_id,\n"
"                                 float3 sample_dir,\n"
"                                 constant float4 *triangle_normals,\n"
"                                 constant TriangleNormalRange *triangle_normal_ranges)\n"
"{\n"
"  float3 hit_normal = -sample_dir;\n"
"  const TriangleNormalRange normal_range = triangle_normal_ranges[user_id];\n"
"  if (primitive_id < normal_range.count) {\n"
"    hit_normal = triangle_normals[normal_range.offset + primitive_id].xyz;\n"
"  }\n"
"  const float len_sq = dot(hit_normal, hit_normal);\n"
"  if (!(len_sq > 1.0e-10f)) {\n"
"    return -sample_dir;\n"
"  }\n"
"  hit_normal *= rsqrt(len_sq);\n"
"  return (dot(hit_normal, sample_dir) < 0.0f) ? hit_normal : -hit_normal;\n"
"}\n"
"inline float3 hit_shadow_receiver_normal(uint2 tid,\n"
"                                         float3 fallback_normal,\n"
"                                         texture2d<uint, access::read> hit_identity_img,\n"
"                                         constant float4 *triangle_normals,\n"
"                                         constant TriangleNormalRange *triangle_normal_ranges)\n"
"{\n"
"  float3 receiver_normal = normalize(fallback_normal);\n"
"  const uint4 hit_identity = hit_identity_img.read(tid);\n"
"  const uint user_id = hit_identity.x;\n"
"  const uint primitive_id = hit_identity.y;\n"
"  const uint identity_flags = hit_identity.z;\n"
"  if (user_id == 0xFFFFFFFFu) {\n"
"    return receiver_normal;\n"
"  }\n"
"  const TriangleNormalRange normal_range = triangle_normal_ranges[user_id];\n"
"  if (primitive_id < normal_range.count) {\n"
"    receiver_normal = triangle_normals[normal_range.offset + primitive_id].xyz;\n"
"  }\n"
"  const float len_sq = dot(receiver_normal, receiver_normal);\n"
"  if (!(len_sq > 1.0e-10f)) {\n"
"    receiver_normal = normalize(fallback_normal);\n"
"  }\n"
"  else {\n"
"    receiver_normal *= rsqrt(len_sq);\n"
"  }\n"
"  if ((identity_flags & 1u) == 0u) {\n"
"    receiver_normal = -receiver_normal;\n"
"  }\n"
"  return (dot(receiver_normal, fallback_normal) >= 0.0f) ? receiver_normal : -receiver_normal;\n"
"}\n"
"inline float2 octahedral_uv_from_direction(float3 co)\n"
"{\n"
"  co /= max(dot(float3(1.0f), abs(co)), 1.0e-8f);\n"
"  if (co.z < 0.0f) {\n"
"    const float2 sign_xy = float2((co.x >= 0.0f) ? 1.0f : -1.0f,\n"
"                                  (co.y >= 0.0f) ? 1.0f : -1.0f);\n"
"    co.xy = (1.0f - abs(co.yx)) * sign_xy;\n"
"  }\n"
"  return co.xy * 0.5f + 0.5f;\n"
"}\n"
"inline float3 sample_reflected_receiver_gi_world_radiance(\n"
"    texture2d_array<float, access::sample> world_probe_tx,\n"
"    float3 direction,\n"
"    constant HardwareReflectedReceiverGIUniforms &u)\n"
"{\n"
"  if (u.environment_pad.x == 0) {\n"
"    return float3(0.0f);\n"
"  }\n"
"  const float4 atlas_coord = u.world_probe_atlas_coord;\n"
"  if (!(atlas_coord.z > 0.0f) || !(atlas_coord.w >= 0.0f)) {\n"
"    return float3(0.0f);\n"
"  }\n"
"  const float3 sample_dir = normalize(direction);\n"
"  const float2 octahedral_uv = octahedral_uv_from_direction(sample_dir);\n"
"  const float mip_0_res = max(atlas_coord.z * 4096.0f, 1.0f);\n"
"  const float2 local_uv = octahedral_uv * ((mip_0_res - 2.0f) / mip_0_res) + 0.5f / mip_0_res;\n"
"  const float2 atlas_uv = local_uv * atlas_coord.z + atlas_coord.xy;\n"
"  constexpr sampler linear_sampler(coord::normalized, address::clamp_to_edge, filter::linear);\n"
"  return world_probe_tx.sample(linear_sampler, atlas_uv, uint(max(int(atlas_coord.w), 0)), level(0.0f)).xyz;\n"
"}\n"
"inline float3 reflected_receiver_gi_direction_unpack(float2 packed_dir)\n"
"{\n"
"  packed_dir = packed_dir * 2.0f - 1.0f;\n"
"  float3 dir = float3(packed_dir.x, packed_dir.y, 1.0f - fabs(packed_dir.x) - fabs(packed_dir.y));\n"
"  const float t = clamp(-dir.z, 0.0f, 1.0f);\n"
"  dir.x += (dir.x >= 0.0f) ? -t : t;\n"
"  dir.y += (dir.y >= 0.0f) ? -t : t;\n"
"  const float len_sq = dot(dir, dir);\n"
"  return (len_sq > 1.0e-10f) ? dir * rsqrt(len_sq) : float3(0.0f, 0.0f, 1.0f);\n"
"}\n"
"inline float3 reflected_receiver_gi_luma_clamp(float3 radiance, float max_luma)\n"
"{\n"
"  radiance = max(radiance, float3(0.0f));\n"
"  const float luma = dot(radiance, float3(0.2126f, 0.7152f, 0.0722f));\n"
"  if (luma > max_luma) {\n"
"    radiance *= max_luma / max(luma, 1.0e-4f);\n"
"  }\n"
"  return radiance;\n"
"}\n"
"inline float3 reflected_receiver_gi_cone_direction(uint2 tid,\n"
"                                                   int sample_index,\n"
"                                                   float3 reflection_dir,\n"
"                                                   float roughness,\n"
"                                                   constant HardwareReflectedReceiverGIUniforms &u)\n"
"{\n"
"  const float cone_roughness = clamp(max(roughness, 0.18f), 0.0f, 0.75f);\n"
"  float3 tangent = (fabs(reflection_dir.z) < 0.999f) ? normalize(cross(reflection_dir, float3(0.0f, 0.0f, 1.0f))) :\n"
"                                                        float3(1.0f, 0.0f, 0.0f);\n"
"  float3 bitangent = normalize(cross(tangent, reflection_dir));\n"
"  const float2 r = float2(reflected_receiver_gi_hash(tid, sample_index, float2(0.21f, 0.79f), u),\n"
"                         reflected_receiver_gi_hash(tid, sample_index, float2(0.57f, 0.33f), u));\n"
"  const float phi = 6.28318530718f * r.x;\n"
"  const float sin_theta = sqrt(max(0.0f, r.y)) * cone_roughness;\n"
"  const float cos_theta = sqrt(max(0.0f, 1.0f - sin_theta * sin_theta));\n"
"  return normalize(reflection_dir * cos_theta + tangent * (cos(phi) * sin_theta) + bitangent * (sin(phi) * sin_theta));\n"
"}\n"
"inline float3 sample_trace_world_radiance(texture2d_array<float, access::sample> world_probe_tx,\n"
"                                          float3 direction,\n"
"                                          bool use_environment,\n"
"                                          constant HardwareTraceUniforms &u)\n"
"{\n"
"  if (!use_environment) {\n"
"    return float3(0.0f);\n"
"  }\n"
"  const float4 atlas_coord = u.world_probe_atlas_coord;\n"
"  if (!(atlas_coord.z > 0.0f) || !(atlas_coord.w >= 0.0f)) {\n"
"    return float3(0.0f);\n"
"  }\n"
"  const float3 sample_dir = normalize(direction);\n"
"  const float2 octahedral_uv = octahedral_uv_from_direction(sample_dir);\n"
"  const float mip_0_res = max(atlas_coord.z * 4096.0f, 1.0f);\n"
"  const float2 local_uv = octahedral_uv * ((mip_0_res - 2.0f) / mip_0_res) + 0.5f / mip_0_res;\n"
"  const float2 atlas_uv = local_uv * atlas_coord.z + atlas_coord.xy;\n"
"  constexpr sampler linear_sampler(coord::normalized, address::clamp_to_edge, filter::linear);\n"
"  return world_probe_tx.sample(linear_sampler, atlas_uv, uint(max(int(atlas_coord.w), 0)), level(0.0f)).xyz;\n"
"}\n"
"inline float3 sample_reflected_receiver_gi_direct_light(\n"
"    uint2 tid,\n"
"    int sample_index,\n"
"    int sample_count,\n"
"    float3 P,\n"
"    float3 N,\n"
"    instance_acceleration_structure scene,\n"
"    constant FastGILightRecord *light_buf,\n"
"    constant HardwareMaterialProxy *material_proxies,\n"
"    constant float *nis_weights,\n"
"    constant HardwareReflectedReceiverGIUniforms &u,\n"
"    thread int &r_feedback_cluster,\n"
"    thread float &r_feedback_weight)\n"
"{\n"
"  r_feedback_cluster = -1;\n"
"  r_feedback_weight = 0.0f;\n"
"  const int light_count = max(u.light_count_pad.x, 0);\n"
"  const int light_sample_count = min(max(u.light_count_pad.y, 0), sample_count);\n"
"  if (light_count <= 0 || light_sample_count <= 0 || sample_index >= light_sample_count) {\n"
"    return float3(0.0f);\n"
"  }\n"
"  /* Nuru light tree (Stage A): importance-weighted pick over suns + local lights. The shadow\n"
"   * ray below provides the wall-tight visibility that the old center-ray estimate lacked, so\n"
"   * locals can finally light mirror-visible interiors (lamp-lit rooms). */\n"
"  const int local_count = clamp(u.light_count_pad.z, 0, light_count);\n"
"  const float2 pick_rand = float2(\n"
"      reflected_receiver_gi_hash(tid, sample_index, float2(0.41f, 0.67f), u),\n"
"      reflected_receiver_gi_hash(tid, sample_index, float2(0.83f, 0.29f), u));\n"
"  float pick_pdf = 1.0f;\n"
"  /* Nuru NIS G1: learned cluster multipliers shape the descent at this gather position. */\n"
"  float nis_m[NIS_OUT];\n"
"  nis_cluster_multipliers(nis_weights, (u.light_count_pad.w != 0), P, nis_m);\n"
"  const int light_index = light_tree_sample_record(\n"
"      light_buf, light_count, local_count, P, N, pick_rand, nis_m, pick_pdf);\n"
"  if (light_index < 0) {\n"
"    return float3(0.0f);\n"
"  }\n"
"  r_feedback_cluster = clamp(int(light_buf[light_index].spot_size_inv.z), 0, NIS_OUT - 1);\n"
"  const FastGILightRecord light = light_buf[light_index];\n"
"  const uint type = uint(light.direction_type.w + 0.5f);\n"
"  float3 L = float3(0.0f, 0.0f, 1.0f);\n"
"  float light_distance = 100000.0f;\n"
"  if (fast_gi_is_sun(type)) {\n"
"    L = normalize(-light.direction_type.xyz);\n"
"  }\n"
"  else {\n"
"    const float3 to_light = fast_gi_transform_location(light) - P;\n"
"    const float dist_sqr = dot(to_light, to_light);\n"
"    if (!(dist_sqr > 1.0e-10f)) {\n"
"      return float3(0.0f);\n"
"    }\n"
"    light_distance = sqrt(dist_sqr);\n"
"    L = to_light / light_distance;\n"
"  }\n"
"  const float attenuation = fast_gi_light_surface_attenuation(light, type, L, light_distance);\n"
"  const float facing = saturate(dot(N, L));\n"
"  if (!(attenuation > 1.0e-6f) || !(facing > 1.0e-4f)) {\n"
"    return float3(0.0f);\n"
"  }\n"
"  const float occlusion_bias = max(u.normal_bias_pad.x, 1.0e-4f);\n"
"  const float occlusion_tmin = max(5.0e-4f, occlusion_bias * 0.5f);\n"
"  const float occlusion_tmax = fast_gi_is_sun(type) ?\n"
"                                   100000.0f :\n"
"                                   max(light_distance - occlusion_bias, occlusion_tmin);\n"
"  /* Transparent punch-through occlusion, mirroring hardware_shadow_visibility: skip thin\n"
"   * glass, tint through refraction, attenuate through alpha-blend cutouts, terminate on\n"
"   * opaque. Sun through window glass is the dominant interior light; an opaque-only ray\n"
"   * killed every such sample (classroom black-mirror regression). */\n"
"  intersector<triangle_data, instancing, max_levels<2>> occ;\n"
"  occ.assume_geometry_type(geometry_type::triangle);\n"
"  occ.force_opacity(forced_opacity::opaque);\n"
"  float3 occlusion_throughput = float3(1.0f);\n"
"  float occlusion_current_tmin = occlusion_tmin;\n"
"  const float3 occlusion_origin = P + N * occlusion_bias;\n"
"  for (int occlusion_bounce = 0; occlusion_bounce < 4; occlusion_bounce++) {\n"
"    const intersection_result<triangle_data, instancing, max_levels<2>> occlusion =\n"
"        occ.intersect(ray(occlusion_origin, L, occlusion_current_tmin, occlusion_tmax), scene);\n"
"    if (occlusion.type != intersection_type::triangle) {\n"
"      break;\n"
"    }\n"
"    const HardwareMaterialProxy occluder = material_proxies[occlusion.user_instance_id[0]];\n"
"    const uint occluder_flags = uint(occluder.ior_closure_type.w);\n"
"    if ((occluder_flags & HWRT_PROXY_FLAG_THIN_GLASS) == 0u) {\n"
"      const uint occluder_closure = uint(occluder.ior_closure_type.z);\n"
"      if (occluder_closure == HWRT_CLOSURE_REFRACTION) {\n"
"        occlusion_throughput *= clamp(occluder.transmission_color_roughness.rgb,\n"
"                                      float3(0.0f),\n"
"                                      float3(1.0f));\n"
"      }\n"
"      else if ((occluder_flags & HWRT_PROXY_FLAG_ALPHA_BLEND_TRANSPARENT) != 0u) {\n"
"        occlusion_throughput *= saturate(1.0f - occluder.packed_thickness.y);\n"
"      }\n"
"      else {\n"
"        return float3(0.0f);\n"
"      }\n"
"      if (max(occlusion_throughput.r,\n"
"              max(occlusion_throughput.g, occlusion_throughput.b)) < 1.0e-3f) {\n"
"        return float3(0.0f);\n"
"      }\n"
"    }\n"
"    occlusion_current_tmin = occlusion.distance + 1.0e-4f;\n"
"    if (occlusion_current_tmin >= occlusion_tmax) {\n"
"      break;\n"
"    }\n"
"  }\n"
"  const float direct_scale = float(sample_count) / float(light_sample_count);\n"
"  const float power = light.color_diffuse_power.w *\n"
"                      fast_gi_light_point_power(light, type, light_distance, L) * attenuation *\n"
"                      facing * direct_scale / max(pick_pdf, 1.0e-6f);\n"
"  const float3 contribution = occlusion_throughput * light.color_diffuse_power.xyz * power;\n"
"  r_feedback_weight = dot(max(contribution, float3(0.0f)),\n"
"                          float3(0.2126f, 0.7152f, 0.0722f));\n"
"  return contribution;\n"
"}\n"
"inline float4 sample_trace_emissive_direction(uint2 tid,\n"
"                                             int sample_index,\n"
"                                             float3 P,\n"
"                                             constant EmissiveLightRecord *emissive_lights,\n"
"                                             constant HardwareTraceUniforms &u)\n"
"{\n"
"  const int light_count = max(u.use_environment_pad.y, 0);\n"
"  if (light_count <= 0) {\n"
"    return float4(0.0f);\n"
"  }\n"
"  const float2 select_rand = rand2_trace(tid, sample_index, 101 + sample_index * 17, u);\n"
"  const int light_index = min(int(select_rand.x * float(light_count)), light_count - 1);\n"
"  const float4 light = emissive_lights[light_index].center_radius;\n"
"  float3 L = light.xyz - P;\n"
"  const float distance_to_light = length(L);\n"
"  if (!(distance_to_light > 1.0e-5f)) {\n"
"    return float4(0.0f);\n"
"  }\n"
"  L /= distance_to_light;\n"
"  const float aperture = min(light.w / distance_to_light, 0.95f);\n"
"  const float cos_theta_max = sqrt(max(1.0f - aperture * aperture, 0.0f));\n"
"  const float cone_solid_angle = max(6.28318530718f * (1.0f - cos_theta_max), 1.0e-4f);\n"
"  float3 right, up;\n"
"  make_orthonormal_basis(L, right, up);\n"
"  const float2 rand = rand2_trace(tid, sample_index, 173 + light_index * 23, u);\n"
"  const float cos_theta = mix(1.0f, cos_theta_max, rand.x);\n"
"  const float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));\n"
"  const float phi = 6.28318530718f * rand.y;\n"
"  const float3 dir = normalize(L * cos_theta + right * (cos(phi) * sin_theta) + up * (sin(phi) * sin_theta));\n"
"  const float pdf = (1.0f / float(light_count)) / cone_solid_angle;\n"
"  return float4(dir, pdf);\n"
"}\n"
"inline float3 sample_trace_direct_light(uint2 tid,\n"
"                                        int sample_index,\n"
"                                        int sample_count,\n"
"                                        float3 P,\n"
"                                        float3 N,\n"
"                                        uint source_user_id,\n"
"                                        uint source_primitive_id,\n"
"                                        bool trace_visibility,\n"
"                                        bool include_locals,\n"
"                                        instance_acceleration_structure scene,\n"
"                                        constant FastGILightRecord *light_buf,\n"
"                                        constant float *nis_weights,\n"
"                                        constant HardwareTraceUniforms &u)\n"
"{\n"
"  const int light_count = max(u.light_count_pad.x, 0);\n"
"  const int light_sample_count = min(max(u.light_count_pad.y, 0), sample_count);\n"
"  if (light_count <= 0 || light_sample_count <= 0 || sample_index >= light_sample_count) {\n"
"    return float3(0.0f);\n"
"  }\n"
"  int light_index = -1;\n"
"  float select_weight = 0.0f;\n"
"  if (include_locals) {\n"
"    /* Nuru light tree: importance-weighted pick over suns + locals. Used by the secondary GI\n"
"     * receiver gather, whose dome hits have NO hit-lighting transport: NEE here is the only\n"
"     * light estimator, so no double counting. */\n"
"    const float2 pick_rand = rand2_trace(tid, sample_index, 211 + sample_index * 19, u);\n"
"    float pick_pdf = 1.0f;\n"
"    float nis_m[NIS_OUT];\n"
"    nis_cluster_multipliers(nis_weights, (u.secondary_gi_pad.z != 0), P, nis_m);\n"
"    light_index = light_tree_sample_record(\n"
"        light_buf, light_count, max(u.light_count_pad.z, 0), P, N, pick_rand, nis_m, pick_pdf);\n"
"    select_weight = 1.0f / max(pick_pdf, 1.0e-6f);\n"
"  }\n"
"  else {\n"
"    light_index = min(int(rand2_trace(tid,\n"
"                                      sample_index,\n"
"                                      211 + sample_index * 19,\n"
"                                      u).x * float(light_count)),\n"
"                      light_count - 1);\n"
"    select_weight = float(light_count);\n"
"  }\n"
"  if (light_index < 0) {\n"
"    return float3(0.0f);\n"
"  }\n"
"  const FastGILightRecord light = light_buf[light_index];\n"
"  const uint type = uint(light.direction_type.w + 0.5f);\n"
"  if (!include_locals && !fast_gi_is_sun(type)) {\n"
"    /* Nuru: the primary gather transport is a calibrated DISJOINT split: the hit-lighting\n"
"     * kernel evaluates LOCAL lights analytically at GI hits (with hit-shadow visibility),\n"
"     * while this in-kernel NEE owns SUNS for the dome rays. Cycles-matched on the closed-room\n"
"     * repro (June 12 2026): adding locals here double-counts their bounce. */\n"
"    return float3(0.0f);\n"
"  }\n"
"  float3 L = float3(0.0f, 0.0f, 1.0f);\n"
"  float light_distance = 100000.0f;\n"
"  if (fast_gi_is_sun(type)) {\n"
"    L = normalize(-light.direction_type.xyz);\n"
"  }\n"
"  else {\n"
"    const float3 to_light = fast_gi_transform_location(light) - P;\n"
"    const float dist_sqr = dot(to_light, to_light);\n"
"    if (!(dist_sqr > 1.0e-10f)) {\n"
"      return float3(0.0f);\n"
"    }\n"
"    light_distance = sqrt(dist_sqr);\n"
"    L = to_light / light_distance;\n"
"  }\n"
"  const float attenuation = fast_gi_light_surface_attenuation(light, type, L, light_distance);\n"
"  const float facing = saturate(dot(N, L));\n"
"  if (!(attenuation > 1.0e-6f) || !(facing > 1.0e-4f)) {\n"
"    return float3(0.0f);\n"
"  }\n"
"  if (trace_visibility) {\n"
"    const float visibility_origin_epsilon = hwrt_gi_ray_epsilon(light_distance);\n"
"    const float visibility_ray_tmin = visibility_origin_epsilon;\n"
"    const float visibility_self_hit_tmax = hwrt_gi_self_hit_distance(visibility_origin_epsilon);\n"
"    const float3 origin = P + N * visibility_origin_epsilon;\n"
"    const float ray_tmax = fast_gi_is_sun(type) ? 100000.0f : max(light_distance - visibility_origin_epsilon, visibility_ray_tmin);\n"
"    intersector<triangle_data, instancing, max_levels<2>> i;\n"
"    i.assume_geometry_type(geometry_type::triangle);\n"
"    i.force_opacity(forced_opacity::opaque);\n"
"    intersection_result<triangle_data, instancing, max_levels<2>> intersection = i.intersect(\n"
"        ray(origin, L, visibility_ray_tmin, ray_tmax), scene);\n"
"    if (intersection.type == intersection_type::triangle &&\n"
"        intersection.distance <= visibility_self_hit_tmax &&\n"
"        intersection.user_instance_id[0] == source_user_id &&\n"
"        intersection.primitive_id == source_primitive_id)\n"
"    {\n"
"      /* Nuru: visibility to local lights must be as tight as the diffuse gather itself. Ignore\n"
"       * only the exact source-triangle self hit; do not use a distance-scaled bias that can skip\n"
"       * sealed wall blockers. */\n"
"      const float3 retry_origin = origin + L * (intersection.distance + visibility_ray_tmin);\n"
"      intersection = i.intersect(ray(retry_origin, L, visibility_ray_tmin, ray_tmax), scene);\n"
"    }\n"
"    if (intersection.type == intersection_type::triangle) {\n"
"      return float3(0.0f);\n"
"    }\n"
"  }\n"
"  const float direct_scale = float(sample_count) / float(light_sample_count);\n"
"  const float power = light.color_diffuse_power.w *\n"
"                      fast_gi_light_point_power(light, type, light_distance, L) * attenuation *\n"
"                      facing * select_weight * direct_scale;\n"
"  return light.color_diffuse_power.xyz * power;\n"
"}\n"
"inline bool fast_gi_skip_stable_space(bool reuse_history,\n"
"                                      float history_error,\n"
"                                      float4 history_visibility)\n"
"{\n"
"  if (!reuse_history) {\n"
"    return false;\n"
"  }\n"
"  const float occupancy = saturate(history_visibility.x);\n"
"  const float thickness = saturate(history_visibility.y);\n"
"  const float openness = saturate(history_visibility.z);\n"
"  const bool stable_empty = occupancy < 0.04f && thickness < 0.04f && openness > 0.92f && history_error < 0.08f;\n"
"  const bool stable_occluded = occupancy > 0.96f && thickness > 0.85f && openness < 0.12f && history_error < 0.12f;\n"
"  return stable_empty || stable_occluded;\n"
"}\n"
"inline int fast_gi_adaptive_sample_count(int base_sample_count,\n"
"                                         bool reuse_history,\n"
"                                         float history_error,\n"
"                                         float4 history_visibility)\n"
"{\n"
"  if (!reuse_history) {\n"
"    return max(base_sample_count, 1);\n"
"  }\n"
"  const float occupancy = saturate(history_visibility.x);\n"
"  const float thickness = saturate(history_visibility.y);\n"
"  const float openness = saturate(history_visibility.z);\n"
"  const float error_factor = saturate(history_error * 0.8f);\n"
"  float sample_scale = 0.35f + 0.65f * error_factor;\n"
"  if (occupancy < 0.08f && thickness < 0.08f && openness > 0.80f) {\n"
"    sample_scale *= 0.75f;\n"
"  }\n"
"  if (occupancy > 0.90f && thickness > 0.75f && openness < 0.25f) {\n"
"    sample_scale *= 0.65f;\n"
"  }\n"
"  return clamp(int(round(float(base_sample_count) * sample_scale)), 1, max(base_sample_count, 1));\n"
"}\n"
         "kernel void eevee_hardware_trace_override(\n"
         "    uint3 threadgroup_id [[threadgroup_position_in_grid]],\n"
         "    uint3 local_id [[thread_position_in_threadgroup]],\n"
         "    instance_acceleration_structure scene [[buffer(0)]],\n"
         "    constant HardwareTraceUniforms &uniforms [[buffer(1)]],\n"
         "    constant float4 *emissive_radiance [[buffer(2)]],\n"
         "    constant float4 *diffuse_albedo [[buffer(3)]],\n"
"    constant HardwareMaterialProxy *material_proxy [[buffer(4)]],\n"
"    constant float4 *triangle_normals [[buffer(5)]],\n"
"    constant TriangleNormalRange *triangle_normal_ranges [[buffer(6)]],\n"
         "    constant uint *tiles_coord_buf [[buffer(7)]],\n"
         "    constant float4 *triangle_smooth_normals [[buffer(8)]],\n"
         "    constant float4 *triangle_local_positions [[buffer(9)]],\n"
"    constant EmissiveLightRecord *emissive_lights [[buffer(10)]],\n"
"    constant FastGILightRecord *trace_lights [[buffer(11)]],\n"
"    constant float *nis_weights [[buffer(12)]],\n"
         "    texture2d<half, access::read> ray_data_tx [[texture(0)]],\n"
         "    depth2d<float, access::sample> depth_tx [[texture(1)]],\n"
"    texture2d_array<uint, access::read> gbuf_header_tx [[texture(2)]],\n"
"    texture2d_array<float, access::read> gbuf_normal_tx [[texture(3)]],\n"
"    texture2d<float, access::read> screen_continuation_img [[texture(4)]],\n"
"    texture2d<float, access::read_write> ray_time_img [[texture(5)]],\n"
"    texture2d<float, access::read_write> ray_radiance_img [[texture(6)]],\n"
"    texture2d<float, access::write> hit_albedo_img [[texture(7)]],\n"
"    texture2d<float, access::write> hit_material_img [[texture(8)]],\n"
"    texture2d<float, access::write> hit_normal_img [[texture(9)]],\n"
"    texture2d<float, access::write> hit_position_img [[texture(10)]],\n"
"    texture2d<uint, access::write> hit_identity_img [[texture(11)]],\n"
"    texture2d<float, access::write> hit_barycentric_img [[texture(12)]],\n"
"    texture2d<float, access::write> hit_world_position_img [[texture(13)]],\n"
"    texture2d<float, access::write> hit_throughput_img [[texture(14)]],\n"
"    texture2d<float, access::write> layered_receiver_ray_time_img [[texture(15)]],\n"
"    texture2d<float, access::write> layered_receiver_ray_radiance_img [[texture(16)]],\n"
"    texture2d<float, access::write> layered_receiver_albedo_img [[texture(17)]],\n"
"    texture2d<float, access::write> layered_receiver_material_img [[texture(18)]],\n"
"    texture2d<float, access::write> layered_receiver_normal_img [[texture(19)]],\n"
"    texture2d<float, access::write> layered_receiver_position_img [[texture(20)]],\n"
"    texture2d<uint, access::write> layered_receiver_identity_img [[texture(21)]],\n"
"    texture2d<float, access::write> layered_receiver_barycentric_img [[texture(22)]],\n"
"    texture2d<float, access::write> layered_receiver_world_position_img [[texture(23)]],\n"
"    texture2d<float, access::write> layered_receiver_throughput_img [[texture(24)]],\n"
"    texture2d<float, access::write> transmission_receiver_ray_time_img [[texture(25)]],\n"
"    texture2d<float, access::write> transmission_receiver_ray_radiance_img [[texture(26)]],\n"
"    texture2d<float, access::write> transmission_receiver_albedo_img [[texture(27)]],\n"
"    texture2d<float, access::write> transmission_receiver_material_img [[texture(28)]],\n"
"    texture2d<float, access::write> transmission_receiver_normal_img [[texture(29)]],\n"
"    texture2d<float, access::write> transmission_receiver_position_img [[texture(30)]],\n"
"    texture2d<uint, access::write> transmission_receiver_identity_img [[texture(31)]],\n"
"    texture2d<float, access::write> transmission_receiver_barycentric_img [[texture(32)]],\n"
"    texture2d<float, access::write> transmission_receiver_world_position_img [[texture(33)]],\n"
"    texture2d<float, access::write> transmission_receiver_throughput_img [[texture(34)]],\n"
"    texture2d_array<float, access::sample> world_probe_tx [[texture(35)]])\n"
         "{\n"
         "  const uint2 tile_coord = unpackUvec2x16(tiles_coord_buf[threadgroup_id.x]);\n"
         "  const uint2 tid = uint2(local_id.xy) + tile_coord * 8u;\n"
         "  if (tid.x >= ray_data_tx.get_width() || tid.y >= ray_data_tx.get_height()) {\n"
         "    return;\n"
         "  }\n"
         "  const half4 packed_ray = ray_data_tx.read(tid);\n"
         "  const float preserved_screen_time = ray_time_img.read(tid).x;\n"
         "  const float4 preserved_radiance = ray_radiance_img.read(tid);\n"
         "  const float4 screen_continuation = screen_continuation_img.read(tid);\n"
"  layered_receiver_ray_time_img.write(float4(0.0f), tid);\n"
"  layered_receiver_ray_radiance_img.write(float4(0.0f), tid);\n"
"  layered_receiver_albedo_img.write(float4(0.0f), tid);\n"
"  layered_receiver_material_img.write(float4(0.0f), tid);\n"
"  layered_receiver_normal_img.write(float4(0.0f), tid);\n"
"  layered_receiver_position_img.write(float4(0.0f), tid);\n"
"  layered_receiver_world_position_img.write(float4(0.0f), tid);\n"
"  layered_receiver_throughput_img.write(float4(0.0f), tid);\n"
"  layered_receiver_identity_img.write(uint4(0u, 0u, 0u, 0xFFFFFFFFu), tid);\n"
"  layered_receiver_barycentric_img.write(float4(0.0f), tid);\n"
"  transmission_receiver_ray_time_img.write(float4(0.0f), tid);\n"
"  transmission_receiver_ray_radiance_img.write(float4(0.0f), tid);\n"
"  transmission_receiver_albedo_img.write(float4(0.0f), tid);\n"
"  transmission_receiver_material_img.write(float4(0.0f), tid);\n"
"  transmission_receiver_normal_img.write(float4(0.0f), tid);\n"
"  transmission_receiver_position_img.write(float4(0.0f), tid);\n"
"  transmission_receiver_world_position_img.write(float4(0.0f), tid);\n"
"  transmission_receiver_throughput_img.write(float4(0.0f), tid);\n"
"  transmission_receiver_identity_img.write(uint4(0u, 0u, 0u, 0xFFFFFFFFu), tid);\n"
"  transmission_receiver_barycentric_img.write(float4(0.0f), tid);\n"
         "  if (packed_ray.w == 0.0h) {\n"
         "    ray_time_img.write(float4(-1.0f, 0.0f, 0.0f, 0.0f), tid);\n"
         "    ray_radiance_img.write(preserved_radiance, tid);\n"
         "    hit_albedo_img.write(float4(0.0f), tid);\n"
"    hit_material_img.write(float4(0.0f), tid);\n"
"    hit_normal_img.write(float4(0.0f), tid);\n"
"    hit_position_img.write(float4(0.0f), tid);\n"
"    hit_world_position_img.write(float4(0.0f), tid);\n"
"    hit_throughput_img.write(float4(0.0f), tid);\n"
"    hit_identity_img.write(uint4(0u, 0u, 0u, 0xFFFFFFFFu), tid);\n"
"    hit_barycentric_img.write(float4(0.0f), tid);\n"
         "    return;\n"
         "  }\n"
"  const int scale = max(uniforms.resolution_scale, 1);\n"
"  const int denominator = max(uniforms.resolution_scale_denominator, 1);\n"
"  const int2 cell_min = (int2(tid) * scale + int2(denominator - 1)) / denominator;\n"
"  const int2 cell_max = ((int2(tid) + int2(1)) * scale + int2(denominator - 1)) / denominator - int2(1);\n"
"  const int2 cell_extent = max(cell_max - cell_min + int2(1), int2(1));\n"
"  const int2 local_offset = min((max(uniforms.resolution_bias, int2(0)) * cell_extent) / scale, cell_extent - int2(1));\n"
"  const int2 texel_fullres = cell_min + local_offset;\n"
         "  if (texel_fullres.x < 0 || texel_fullres.y < 0 || texel_fullres.x >= uniforms.full_resolution.x || texel_fullres.y >= uniforms.full_resolution.y) {\n"
         "    ray_time_img.write(float4(-1.0f, 0.0f, 0.0f, 0.0f), tid);\n"
         "    ray_radiance_img.write(preserved_radiance, tid);\n"
         "    hit_albedo_img.write(float4(0.0f), tid);\n"
"    hit_material_img.write(float4(0.0f), tid);\n"
"    hit_normal_img.write(float4(0.0f), tid);\n"
"    hit_position_img.write(float4(0.0f), tid);\n"
"    hit_world_position_img.write(float4(0.0f), tid);\n"
"    hit_throughput_img.write(float4(0.0f), tid);\n"
"    hit_identity_img.write(uint4(0u, 0u, 0u, 0xFFFFFFFFu), tid);\n"
"    hit_barycentric_img.write(float4(0.0f), tid);\n"
         "    return;\n"
         "  }\n"
         "  const uint gbuf_header = gbuf_header_tx.read(uint2(texel_fullres), 0).x;\n"
         "  const uint gbuf_mode = (gbuf_header >> (uniforms.closure_index * GBUFFER_HEADER_BITS_PER_BIN)) & 15u;\n"
         "  const bool supports_hardware_gi = ((uniforms.feature_mask & FEATURE_HARDWARE_GI) != 0u) && ((gbuf_mode == GBUF_DIFFUSE) || (gbuf_mode == GBUF_SUBSURFACE));\n"
         "  const bool supports_hardware_reflection = ((uniforms.feature_mask & FEATURE_HARDWARE_REFLECTIONS) != 0u) && ((gbuf_mode == GBUF_REFLECTION) || (gbuf_mode == GBUF_REFLECTION_COLORLESS));\n"
         "  const bool supports_hardware_refraction = ((uniforms.feature_mask & FEATURE_HARDWARE_REFRACTIONS) != 0u) && ((gbuf_mode == GBUF_REFRACTION) || (gbuf_mode == GBUF_REFRACTION_COLORLESS));\n"
         "  const bool continuation_required = (supports_hardware_reflection && (uniforms.reflection_bounces > 1)) ||\n"
         "                                   (supports_hardware_refraction && (uniforms.refraction_bounces > 1));\n"
         "  const bool has_screen_continuation = screen_continuation.w > 0.0f;\n"
         "  const bool scene_final_specular_phase = (uniforms.hardware_trace_phase == 2);\n"
         "  const bool preserved_screen_hit = !scene_final_specular_phase &&\n"
         "                                    (supports_hardware_reflection || supports_hardware_refraction) &&\n"
         "                                    (preserved_screen_time > 0.0f) &&\n"
         "                                    (preserved_screen_time < 10000.0f);\n"
         "  const bool use_preserved_screen_hit = preserved_screen_hit &&\n"
         "                                        (!continuation_required || has_screen_continuation);\n"
         "  if (!(supports_hardware_gi || supports_hardware_reflection || supports_hardware_refraction) || gbuf_mode == GBUF_NONE) {\n"
         "    ray_time_img.write(float4(-1.0f, 0.0f, 0.0f, 0.0f), tid);\n"
         "    ray_radiance_img.write(preserved_radiance, tid);\n"
         "    hit_albedo_img.write(float4(0.0f), tid);\n"
         "    hit_material_img.write(float4(0.0f), tid);\n"
         "    hit_normal_img.write(float4(0.0f), tid);\n"
         "    hit_position_img.write(float4(0.0f), tid);\n"
         "    hit_world_position_img.write(float4(0.0f), tid);\n"
         "    hit_throughput_img.write(float4(0.0f), tid);\n"
         "    hit_identity_img.write(uint4(0u, 0u, 0u, 0xFFFFFFFFu), tid);\n"
         "    hit_barycentric_img.write(float4(0.0f), tid);\n"
         "    return;\n"
         "  }\n"
         "  const float2 uv = (float2(texel_fullres) + 0.5f) / float2(uniforms.full_resolution);\n"
         "  constexpr sampler depth_sampler(coord::normalized, address::clamp_to_edge, filter::nearest);\n"
         "  const float depth = 1.0f - depth_tx.sample(depth_sampler, uv);\n"
         "  if (depth <= 0.0f || depth >= 1.0f) {\n"
         "    ray_time_img.write(float4(-2.0f, 0.0f, 0.0f, 0.0f), tid);\n"
         "    ray_radiance_img.write(preserved_radiance, tid);\n"
         "    hit_albedo_img.write(float4(0.0f), tid);\n"
"    hit_material_img.write(float4(0.0f), tid);\n"
"    hit_normal_img.write(float4(0.0f), tid);\n"
"    hit_position_img.write(float4(0.0f), tid);\n"
"    hit_world_position_img.write(float4(0.0f), tid);\n"
"    hit_throughput_img.write(float4(0.0f), tid);\n"
"    hit_identity_img.write(uint4(0u, 0u, 0u, 0xFFFFFFFFu), tid);\n"
"    hit_barycentric_img.write(float4(0.0f), tid);\n"
         "    return;\n"
         "  }\n"
         "  float3 ray_direction = normalize(float3(packed_ray.xyz));\n"
"  if (preserved_screen_hit && !continuation_required) {\n"
"    ray_time_img.write(float4(max(preserved_screen_time, 1.0e-4f), 0.0f, 0.0f, 0.0f), tid);\n"
"    ray_radiance_img.write(preserved_radiance, tid);\n"
"    hit_albedo_img.write(float4(0.0f), tid);\n"
"    hit_material_img.write(float4(0.0f), tid);\n"
"    hit_normal_img.write(float4(0.0f), tid);\n"
"    hit_position_img.write(float4(0.0f), tid);\n"
"    hit_world_position_img.write(float4(0.0f), tid);\n"
"    hit_throughput_img.write(float4(0.0f), tid);\n"
"    hit_identity_img.write(uint4(0u, 0u, 0u, 0xFFFFFFFFu), tid);\n"
"    hit_barycentric_img.write(float4(0.0f), tid);\n"
"    return;\n"
"  }\n"
"  float3 ray_origin = point_screen_to_world(uv, depth, uniforms);\n"
"  int start_bounce = 0;\n"
"  if (use_preserved_screen_hit && continuation_required && has_screen_continuation) {\n"
"    ray_origin = screen_continuation.xyz;\n"
"    start_bounce = 1;\n"
"  }\n"
"  if (scene_final_specular_phase && !use_preserved_screen_hit && supports_hardware_reflection) {\n"
"    float3 surface_N;\n"
"    if (load_gbuffer_surface_normal(\n"
"            texel_fullres, gbuf_header, uint(uniforms.closure_index), gbuf_normal_tx, surface_N))\n"
"    {\n"
"      /* Keep the late mirror/reflection launch epsilon small enough that enclosed receivers such\n"
"       * as nearby room walls are not skipped and replaced by the world/HDRI miss path. */\n"
"      ray_origin += surface_N * ((dot(surface_N, ray_direction) >= 0.0f) ? 1.0e-3f : -1.0e-3f);\n"
"    }\n"
"  }\n"
"  /* Full HWRT can already traverse the real back-face of the refractive object on the first\n"
"   * bounce. Do not analytically skip through thickness here or we will overrun nearby receivers\n"
"   * and distort the apparent IOR. */\n"
"  ray_origin += ray_direction * hwrt_specular_ray_epsilon(supports_hardware_refraction);\n"
         "  intersector<triangle_data, instancing, max_levels<2>> i;\n"
         "  i.assume_geometry_type(geometry_type::triangle);\n"
         "  i.force_opacity(forced_opacity::opaque);\n"
         "  int max_bounces = 1;\n"
"  if (supports_hardware_reflection) {\n"
"    max_bounces = max(uniforms.reflection_bounces, 1);\n"
"  }\n"
"  else if (supports_hardware_refraction) {\n"
"    max_bounces = max(uniforms.refraction_bounces, 1);\n"
"  }\n"
"  float3 radiance = use_preserved_screen_hit ? preserved_radiance.xyz : float3(0.0f);\n"
"  float3 throughput = float3(1.0f);\n"
"  float total_distance = (use_preserved_screen_hit && continuation_required && has_screen_continuation) ?\n"
"                             max(screen_continuation.w, 0.0f) :\n"
"                             0.0f;\n"
"  float3 final_position = ray_origin;\n"
"  float3 final_local_position = float3(0.0f);\n"
"  float3 final_direction = ray_direction;\n"
"  float final_segment_distance = 0.0f;\n"
"  float3 final_normal = float3(0.0f);\n"
"  float2 final_barycentric = float2(0.0f);\n"
"  float3 carried_scene_final_throughput = float3(1.0f);\n"
"  bool apply_scene_final_throughput = false;\n"
"  float3 preserved_output_throughput = float3(1.0f);\n"
"  uint final_user_id = 0u;\n"
"  uint final_primitive_id = 0u;\n"
"  uint final_front_facing = 1u;\n"
"  bool preserved_scene_final_reflective_hit = false;\n"
"  bool preserved_transparent_scene_final_hit = false;\n"
"  bool preserved_layered_scene_final_hit = false;\n"
"  float3 preserved_position = float3(0.0f);\n"
"  float3 preserved_local_position = float3(0.0f);\n"
"  float3 preserved_direction = ray_direction;\n"
"  float preserved_total_distance = 0.0f;\n"
"  float preserved_segment_distance = 0.0f;\n"
"  float3 preserved_normal = float3(0.0f);\n"
"  float2 preserved_barycentric = float2(0.0f);\n"
"  uint preserved_user_id = 0u;\n"
"  uint preserved_primitive_id = 0u;\n"
"  uint preserved_front_facing = 1u;\n"
"  bool final_layered_principled_scene_final_hit = false;\n"
"  bool final_transparent_scene_final_hit = false;\n"
"  bool final_refracted_textured_receiver_hit = false;\n"
"  HardwareMaterialProxy preserved_proxy;\n"
"  preserved_proxy.reflection_color_roughness = float4(0.0f);\n"
"  preserved_proxy.transmission_color_roughness = float4(0.0f);\n"
"  preserved_proxy.ior_closure_type = float4(0.0f);\n"
"  preserved_proxy.packed_thickness = float4(0.0f);\n"
"  uint preserved_proxy_closure = 0u;\n"
"  bool layered_receiver_valid = false;\n"
"  float3 layered_receiver_position = float3(0.0f);\n"
"  float3 layered_receiver_local_position = float3(0.0f);\n"
"  float3 layered_receiver_direction = ray_direction;\n"
"  float layered_receiver_total_distance = 0.0f;\n"
"  float layered_receiver_segment_distance = 0.0f;\n"
"  float3 layered_receiver_normal = float3(0.0f);\n"
"  float2 layered_receiver_barycentric = float2(0.0f);\n"
"  uint layered_receiver_user_id = 0u;\n"
"  uint layered_receiver_primitive_id = 0u;\n"
"  uint layered_receiver_front_facing = 1u;\n"
"  float3 layered_receiver_carried_throughput = float3(1.0f);\n"
"  HardwareMaterialProxy layered_receiver_proxy;\n"
"  layered_receiver_proxy.reflection_color_roughness = float4(0.0f);\n"
"  layered_receiver_proxy.transmission_color_roughness = float4(0.0f);\n"
"  layered_receiver_proxy.ior_closure_type = float4(0.0f);\n"
"  layered_receiver_proxy.packed_thickness = float4(0.0f);\n"
"  uint layered_receiver_proxy_closure = 0u;\n"
"  float3 layered_receiver_continued_radiance = float3(0.0f);\n"
"  bool transmission_receiver_valid = false;\n"
"  float3 transmission_receiver_position = float3(0.0f);\n"
"  float3 transmission_receiver_local_position = float3(0.0f);\n"
"  float3 transmission_receiver_direction = ray_direction;\n"
"  float transmission_receiver_total_distance = 0.0f;\n"
"  float transmission_receiver_segment_distance = 0.0f;\n"
"  float3 transmission_receiver_normal = float3(0.0f);\n"
"  float2 transmission_receiver_barycentric = float2(0.0f);\n"
"  uint transmission_receiver_user_id = 0u;\n"
"  uint transmission_receiver_primitive_id = 0u;\n"
"  uint transmission_receiver_front_facing = 1u;\n"
"  float3 transmission_receiver_carried_throughput = float3(1.0f);\n"
"  bool transmission_receiver_apply_throughput = false;\n"
"  bool transmission_receiver_direct_lit_reflective = false;\n"
"  bool transmission_receiver_lock_surface = false;\n"
"  HardwareMaterialProxy transmission_receiver_proxy;\n"
"  transmission_receiver_proxy.reflection_color_roughness = float4(0.0f);\n"
"  transmission_receiver_proxy.transmission_color_roughness = float4(0.0f);\n"
"  transmission_receiver_proxy.ior_closure_type = float4(0.0f);\n"
"  transmission_receiver_proxy.packed_thickness = float4(0.0f);\n"
"  uint transmission_receiver_proxy_closure = 0u;\n"
"  float3 transmission_receiver_continued_radiance = float3(0.0f);\n"
"  const float ray_tmin = (scene_final_specular_phase && !use_preserved_screen_hit) ?\n"
"                         hwrt_specular_ray_tmin(supports_hardware_refraction) :\n"
"                         0.0f;\n"
"  HardwareMaterialProxy final_proxy;\n"
"  final_proxy.reflection_color_roughness = float4(0.0f);\n"
"  final_proxy.transmission_color_roughness = float4(0.0f);\n"
"  final_proxy.ior_closure_type = float4(0.0f);\n"
"  final_proxy.packed_thickness = float4(0.0f);\n"
"  uint final_proxy_closure = 0u;\n"
"  int thin_glass_passthrough_count = 0;\n"
"  for (int bounce = start_bounce; bounce < max_bounces; bounce++) {\n"
"    intersection_result<triangle_data, instancing, max_levels<2>> intersection = "
"i.intersect(ray(ray_origin, ray_direction, ray_tmin, 10000.0f), scene);\n"
"    if (intersection.type != intersection_type::triangle) {\n"
"      const float2 packed_direction = direction_pack(ray_direction);\n"
"      const bool has_specular_throughput =\n"
"          ((final_proxy_closure == HWRT_CLOSURE_REFLECTION) ||\n"
"           (final_proxy_closure == HWRT_CLOSURE_REFRACTION)) &&\n"
"          (dot(throughput, throughput) > 1.0e-10f);\n"
"      const float3 miss_proxy_color = (final_proxy_closure == HWRT_CLOSURE_REFLECTION) ?\n"
"                                     final_proxy.reflection_color_roughness.xyz :\n"
"                                     final_proxy.transmission_color_roughness.xyz;\n"
"      const float3 miss_tint = has_specular_throughput ?\n"
"                                   clamp(throughput * miss_proxy_color,\n"
"                                         float3(0.0f),\n"
"                                         float3(uniforms.clamp_indirect)) :\n"
"                                   float3(0.0f);\n"
"      const float3 miss_origin = has_specular_throughput ? final_position : ray_origin;\n"
"      const float3 miss_normal = has_specular_throughput ? final_normal : float3(0.0f);\n"
"      if (preserved_layered_scene_final_hit || preserved_scene_final_reflective_hit ||\n"
"          preserved_transparent_scene_final_hit) {\n"
"        break;\n"
"      }\n"
"      ray_time_img.write(float4(-3.0f, 0.0f, 0.0f, 0.0f), tid);\n"
"      ray_radiance_img.write(float4(radiance, 0.0f), tid);\n"
"      hit_albedo_img.write(float4(miss_tint, 0.0f), tid);\n"
"      hit_material_img.write(float4(0.0f, 0.0f, float(final_proxy_closure), packed_direction.x), tid);\n"
"      hit_normal_img.write(float4(miss_normal, packed_direction.y), tid);\n"
"      hit_position_img.write(float4(miss_origin, total_distance), tid);\n"
"      hit_world_position_img.write(float4(miss_origin, total_distance), tid);\n"
"      hit_throughput_img.write(float4(0.0f), tid);\n"
"      hit_identity_img.write(uint4(0u, 0u, 0u, 0xFFFFFFFFu), tid);\n"
"      hit_barycentric_img.write(float4(0.0f), tid);\n"
"      return;\n"
"    }\n"
"    const float hit_time = intersection.distance;\n"
"    total_distance += hit_time;\n"
"    final_position = ray_origin + ray_direction * hit_time;\n"
"    final_direction = ray_direction;\n"
"    final_segment_distance = hit_time;\n"
"    const uint user_id = intersection.user_instance_id[0];\n"
"    final_user_id = user_id;\n"
"    final_primitive_id = intersection.primitive_id;\n"
"    final_barycentric = intersection.triangle_barycentric_coord;\n"
"    if (!use_preserved_screen_hit) {\n"
"      radiance += throughput * min(emissive_radiance[user_id].xyz, float3(uniforms.clamp_indirect));\n"
"    }\n"
"    final_proxy = material_proxy[user_id];\n"
"    float3 raw_hit_normal = float3(0.0f);\n"
"    float3 smooth_hit_normal = float3(0.0f);\n"
"    const TriangleNormalRange normal_range = triangle_normal_ranges[user_id];\n"
"    if (intersection.primitive_id < normal_range.count) {\n"
"      raw_hit_normal = triangle_normals[normal_range.offset + intersection.primitive_id].xyz;\n"
"      const uint smooth_offset = (normal_range.offset + intersection.primitive_id) * 3u;\n"
"      const float3 bary = barycentric_expand(final_barycentric);\n"
"      final_local_position = triangle_local_positions[smooth_offset + 0u].xyz * bary.x +\n"
"                             triangle_local_positions[smooth_offset + 1u].xyz * bary.y +\n"
"                             triangle_local_positions[smooth_offset + 2u].xyz * bary.z;\n"
"      smooth_hit_normal = triangle_smooth_normals[smooth_offset + 0u].xyz * bary.x +\n"
"                          triangle_smooth_normals[smooth_offset + 1u].xyz * bary.y +\n"
"                          triangle_smooth_normals[smooth_offset + 2u].xyz * bary.z;\n"
"    }\n"
"    bool entering = true;\n"
"    float3 hit_normal = smooth_hit_normal;\n"
"    if (!(dot(hit_normal, hit_normal) > 1.0e-10f)) {\n"
"      hit_normal = raw_hit_normal;\n"
"    }\n"
"    if (!(dot(hit_normal, hit_normal) > 1.0e-10f)) {\n"
"      final_normal = -ray_direction;\n"
"    }\n"
"    else {\n"
"      entering = dot(hit_normal, ray_direction) < 0.0f;\n"
"      final_front_facing = entering ? 1u : 0u;\n"
"      final_normal = entering ? hit_normal : -hit_normal;\n"
"    }\n"
"    const uint proxy_closure = uint(final_proxy.ior_closure_type.z + 0.5f);\n"
"    const uint proxy_flags = uint(final_proxy.ior_closure_type.w + 0.5f);\n"
"    const float reflection_roughness = clamp(final_proxy.reflection_color_roughness.w, 0.0f, 1.0f);\n"
"    const float transmission_roughness = clamp(final_proxy.transmission_color_roughness.w, 0.0f, 1.0f);\n"
"    const float transparent_alpha = clamp(final_proxy.packed_thickness.y, 0.0f, 1.0f);\n"
"    const float refraction_ior = max(final_proxy.ior_closure_type.y, 1.0e-3f);\n"
"    const float eta = entering ? (1.0f / refraction_ior) : refraction_ior;\n"
"    const bool resolving_preserved_scene_final_receiver =\n"
"        preserved_layered_scene_final_hit || preserved_scene_final_reflective_hit ||\n"
"        preserved_transparent_scene_final_hit;\n"
"    const bool supports_hardware_specular_receiver = supports_hardware_reflection ||\n"
"                                                     supports_hardware_refraction;\n"
"    const bool thin_glass_passthrough =\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_THIN_GLASS) != 0u) &&\n"
"        (!scene_final_specular_phase || supports_hardware_gi || (bounce > start_bounce) ||\n"
"         resolving_preserved_scene_final_receiver);\n"
"    if (thin_glass_passthrough && (thin_glass_passthrough_count < 8)) {\n"
"      thin_glass_passthrough_count++;\n"
"      ray_origin = final_position + ray_direction * hwrt_specular_ray_epsilon(false);\n"
"      bounce -= 1;\n"
"      continue;\n"
"    }\n"
"    /* Alpha-cutout pass-through (Thin Glass contract for partial-coverage proxies): outside the\n"
"     * scene-final phase, an alpha-transparent hit must not be accepted and lit as a solid\n"
"     * surface. Recording it fed the cutout's support quad into diffuse-GI hit lighting as a\n"
"     * dark occluder (black blotches around spider webs / leaves on nearby walls, independent of\n"
"     * the HWRT shadow toggle). Attenuate by coverage and keep flying without consuming the\n"
"     * bounce, like the Thin Glass branch above. The scene-final phase keeps its dedicated\n"
"     * preserve_transparent_scene_final compositing below. */\n"
"    const bool alpha_cutout_passthrough =\n"
"        !scene_final_specular_phase &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_ALPHA_BLEND_TRANSPARENT) != 0u) &&\n"
"        (transparent_alpha < 1.0f - 1.0e-3f);\n"
"    if (alpha_cutout_passthrough && (thin_glass_passthrough_count < 8)) {\n"
"      thin_glass_passthrough_count++;\n"
"      throughput *= max(float3(1.0f - transparent_alpha), float3(0.0f));\n"
"      if (!(dot(throughput, throughput) > 1.0e-10f)) {\n"
"        break;\n"
"      }\n"
"      ray_origin = final_position + ray_direction * hwrt_specular_ray_epsilon(false);\n"
"      bounce -= 1;\n"
"      continue;\n"
"    }\n"
"    const bool preserve_layered_principled_scene_final =\n"
"        scene_final_specular_phase && supports_hardware_specular_receiver &&\n"
"        !resolving_preserved_scene_final_receiver &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL) != 0u) &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_ALPHA_BLEND_TRANSPARENT) == 0u);\n"
"    const bool preserve_textured_specular_scene_final =\n"
"        scene_final_specular_phase && supports_hardware_specular_receiver &&\n"
"        !preserved_layered_scene_final_hit && !preserved_scene_final_reflective_hit &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL) != 0u) &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_ALPHA_BLEND_TRANSPARENT) == 0u);\n"
"    const bool preserve_transparent_scene_final =\n"
"        scene_final_specular_phase && !preserved_transparent_scene_final_hit &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_ALPHA_BLEND_TRANSPARENT) != 0u) &&\n"
"        (transparent_alpha < 1.0f - 1.0e-3f);\n"
"    const bool preserve_scene_final_transmission_layer =\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_PRINCIPLED_TRANSMISSION_LAYER) != 0u) &&\n"
"        (preserve_layered_principled_scene_final || preserve_textured_specular_scene_final ||\n"
"         preserve_transparent_scene_final);\n"
"    const bool preserve_scene_final_transparent_layer = preserve_transparent_scene_final;\n"
"    const uint scene_final_proxy_carrier =\n"
"        (preserve_scene_final_transmission_layer || preserve_scene_final_transparent_layer) ?\n"
"                                                HWRT_CLOSURE_DIFFUSE :\n"
"                                                proxy_closure;\n"
"    uint resolved_proxy_closure = proxy_closure;\n"
"    if ((proxy_closure == HWRT_CLOSURE_REFRACTION) &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_DIELECTRIC_REFLECTION) != 0u)) {\n"
"      const float3 refracted = refract(ray_direction, final_normal, eta);\n"
"      const bool has_refraction = dot(refracted, refracted) > 1.0e-10f;\n"
"      const float fresnel = dielectric_fresnel_reflectance(ray_direction, final_normal, refraction_ior);\n"
"      const float branch_rand = rand2_trace(\n"
"          tid,\n"
"          bounce + 1,\n"
"          uniforms.closure_index + int(HWRT_CLOSURE_REFLECTION + HWRT_CLOSURE_REFRACTION),\n"
"          uniforms).x;\n"
"      if (!has_refraction || (branch_rand < fresnel)) {\n"
"        resolved_proxy_closure = HWRT_CLOSURE_REFLECTION;\n"
"      }\n"
"    }\n"
"    if (preserve_textured_specular_scene_final &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_DIELECTRIC_REFLECTION) != 0u)) {\n"
"      resolved_proxy_closure = HWRT_CLOSURE_REFLECTION;\n"
"    }\n"
"    const bool replay_textured_specular_receiver =\n"
"        scene_final_specular_phase && (bounce > start_bounce) &&\n"
"        (supports_hardware_reflection || supports_hardware_refraction) &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL) != 0u) &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_THIN_GLASS) == 0u) &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_ALPHA_BLEND_TRANSPARENT) == 0u) &&\n"
"        ((resolved_proxy_closure == HWRT_CLOSURE_REFLECTION) ||\n"
"         (resolved_proxy_closure == HWRT_CLOSURE_REFRACTION));\n"
"    const bool preserved_material_scene_final =\n"
"        preserve_layered_principled_scene_final || preserve_textured_specular_scene_final;\n"
"    const bool layered_receiver_continuation = preserved_material_scene_final &&\n"
"                                             (scene_final_proxy_carrier == HWRT_CLOSURE_DIFFUSE) &&\n"
"                                             (reflection_roughness <= 1.0f);\n"
"    const uint continuation_proxy_closure = layered_receiver_continuation ?\n"
"                                               HWRT_CLOSURE_REFLECTION :\n"
"                                               resolved_proxy_closure;\n"
"    const uint preserved_scene_final_proxy_closure = preserve_textured_specular_scene_final ?\n"
"                                                     resolved_proxy_closure :\n"
"                                                     scene_final_proxy_carrier;\n"
"    final_proxy_closure =\n"
"        (preserved_material_scene_final || preserve_scene_final_transmission_layer ||\n"
"         preserve_scene_final_transparent_layer) ?\n"
"            preserved_scene_final_proxy_closure :\n"
"            resolved_proxy_closure;\n"
"    if (preserved_material_scene_final) {\n"
"      preserved_layered_scene_final_hit = true;\n"
"      preserved_position = final_position;\n"
"      preserved_local_position = final_local_position;\n"
"      preserved_direction = final_direction;\n"
"      preserved_total_distance = total_distance;\n"
"      preserved_segment_distance = final_segment_distance;\n"
"      preserved_normal = final_normal;\n"
"      preserved_barycentric = final_barycentric;\n"
"      preserved_user_id = final_user_id;\n"
"      preserved_primitive_id = final_primitive_id;\n"
"      preserved_front_facing = final_front_facing;\n"
"      preserved_proxy = final_proxy;\n"
"      preserved_proxy_closure = final_proxy_closure;\n"
"      final_layered_principled_scene_final_hit = true;\n"
"    }\n"
"    if (preserve_transparent_scene_final) {\n"
"      preserved_transparent_scene_final_hit = true;\n"
"      preserved_position = final_position;\n"
"      preserved_local_position = final_local_position;\n"
"      preserved_direction = final_direction;\n"
"      preserved_total_distance = total_distance;\n"
"      preserved_segment_distance = final_segment_distance;\n"
"      preserved_normal = final_normal;\n"
"      preserved_barycentric = final_barycentric;\n"
"      preserved_user_id = final_user_id;\n"
"      preserved_primitive_id = final_primitive_id;\n"
"      preserved_front_facing = final_front_facing;\n"
"      preserved_proxy = final_proxy;\n"
"      preserved_proxy_closure = final_proxy_closure;\n"
"      preserved_output_throughput = throughput;\n"
"      final_transparent_scene_final_hit = true;\n"
"    }\n"
"    if (scene_final_specular_phase && supports_hardware_reflection && !preserved_material_scene_final && !preserve_transparent_scene_final && !preserve_scene_final_transmission_layer && !preserved_layered_scene_final_hit && !preserved_scene_final_reflective_hit &&\n"
"        (resolved_proxy_closure == HWRT_CLOSURE_REFLECTION)) {\n"
"      preserved_scene_final_reflective_hit = true;\n"
"      preserved_position = final_position;\n"
"      preserved_local_position = final_local_position;\n"
"      preserved_direction = final_direction;\n"
"      preserved_total_distance = total_distance;\n"
"      preserved_segment_distance = final_segment_distance;\n"
"      preserved_normal = final_normal;\n"
"      preserved_barycentric = final_barycentric;\n"
"      preserved_user_id = final_user_id;\n"
"      preserved_primitive_id = final_primitive_id;\n"
"      preserved_front_facing = final_front_facing;\n"
"      preserved_proxy = final_proxy;\n"
"      preserved_proxy_closure = final_proxy_closure;\n"
"    }\n"
"    /* The scene-final reflective early-out is only valid for the single-bounce shortcut. Once the\n"
"     * user requests deeper reflection continuation, do not clamp the late path back to the first\n"
"     * reflective secondary or nested glossy reflections disappear. */\n"
"    if (scene_final_specular_phase && supports_hardware_reflection && !continuation_required && !preserved_material_scene_final && !preserve_transparent_scene_final && !preserve_scene_final_transmission_layer &&\n"
"        (bounce == start_bounce) && (resolved_proxy_closure == HWRT_CLOSURE_REFLECTION)) {\n"
"      break;\n"
"    }\n"
"    const bool next_hit_is_specular_continuation =\n"
"        ((resolved_proxy_closure == HWRT_CLOSURE_REFLECTION) ||\n"
"         (resolved_proxy_closure == HWRT_CLOSURE_REFRACTION)) &&\n"
"        (bounce + 1 < max_bounces);\n"
"    if (preserved_scene_final_reflective_hit && next_hit_is_specular_continuation) {\n"
"      preserved_scene_final_reflective_hit = false;\n"
"    }\n"
"    if (resolving_preserved_scene_final_receiver && (bounce > start_bounce)) {\n"
"      layered_receiver_valid = true;\n"
"      layered_receiver_position = final_position;\n"
"      layered_receiver_local_position = final_local_position;\n"
"      layered_receiver_direction = final_direction;\n"
"      layered_receiver_total_distance = total_distance;\n"
"      layered_receiver_segment_distance = final_segment_distance;\n"
"      layered_receiver_normal = final_normal;\n"
"      layered_receiver_barycentric = final_barycentric;\n"
"      layered_receiver_user_id = final_user_id;\n"
"      layered_receiver_primitive_id = final_primitive_id;\n"
"      layered_receiver_front_facing = final_front_facing;\n"
"      layered_receiver_proxy = final_proxy;\n"
"      layered_receiver_proxy_closure = resolved_proxy_closure;\n"
"      layered_receiver_carried_throughput = throughput;\n"
"    }\n"
"    if (preserve_transparent_scene_final) {\n"
"      throughput *= max(float3(1.0f - transparent_alpha), float3(0.0f));\n"
"      if (!(dot(throughput, throughput) > 1.0e-10f)) {\n"
"        break;\n"
"      }\n"
"      ray_origin = final_position + ray_direction * hwrt_specular_ray_epsilon(supports_hardware_refraction);\n"
"      continue;\n"
"    }\n"
"    const bool can_continue = (bounce + 1 < max_bounces) &&\n"
"                              ((continuation_proxy_closure == HWRT_CLOSURE_REFLECTION) ||\n"
"                               (continuation_proxy_closure == HWRT_CLOSURE_REFRACTION));\n"
"    if (replay_textured_specular_receiver && !can_continue) {\n"
"      final_refracted_textured_receiver_hit = true;\n"
"    }\n"
"    if (!can_continue) {\n"
"      break;\n"
"    }\n"
"    float3 next_direction = ray_direction;\n"
"    const bool skip_textured_proxy_throughput_tint =\n"
"        scene_final_specular_phase &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL) != 0u) &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_THIN_GLASS) == 0u);\n"
"    if (continuation_proxy_closure == HWRT_CLOSURE_REFLECTION) {\n"
"      const float3 reflection_tint = clamp(final_proxy.reflection_color_roughness.xyz,\n"
"                                           float3(0.0f),\n"
"                                           float3(uniforms.clamp_indirect));\n"
"      if (!skip_textured_proxy_throughput_tint) {\n"
"        throughput *= reflection_tint;\n"
"      }\n"
"      if ((proxy_flags & HWRT_PROXY_FLAG_THIN_GLASS) != 0u) {\n"
"        const float thin_glass_ior = max(final_proxy.ior_closure_type.x, 1.0e-3f);\n"
"        throughput *= dielectric_fresnel_reflectance(ray_direction, final_normal, thin_glass_ior);\n"
"      }\n"
"      next_direction = sample_rough_specular_direction(\n"
"          tid,\n"
"          bounce + 1,\n"
"          uniforms.closure_index + int(HWRT_CLOSURE_REFLECTION),\n"
"          ray_direction,\n"
"          final_normal,\n"
"          reflection_roughness,\n"
"          false,\n"
"          1.0f,\n"
"          uniforms);\n"
"    }\n"
"    else {\n"
"      if (!skip_textured_proxy_throughput_tint) {\n"
"        throughput *= clamp(final_proxy.transmission_color_roughness.xyz,\n"
"                             float3(0.0f),\n"
"                             float3(uniforms.clamp_indirect));\n"
"      }\n"
"      next_direction = sample_rough_specular_direction(\n"
"          tid,\n"
"          bounce + 1,\n"
"          uniforms.closure_index + int(HWRT_CLOSURE_REFRACTION),\n"
"          ray_direction,\n"
"          final_normal,\n"
"          transmission_roughness,\n"
"          true,\n"
"          eta,\n"
"          uniforms);\n"
"    }\n"
"    if (scene_final_specular_phase) {\n"
"      carried_scene_final_throughput = clamp(\n"
"          throughput, float3(0.0f), float3(uniforms.clamp_indirect));\n"
"      apply_scene_final_throughput = true;\n"
"    }\n"
"    if (!(dot(next_direction, next_direction) > 1.0e-10f)) {\n"
"      break;\n"
"    }\n"
"    ray_direction = normalize(next_direction);\n"
"    ray_origin = final_position + ray_direction *\n"
"                 hwrt_specular_ray_epsilon(continuation_proxy_closure == HWRT_CLOSURE_REFRACTION);\n"
"    if ((continuation_proxy_closure == HWRT_CLOSURE_REFRACTION) && entering) {\n"
"      const ThicknessData proxy_thickness = thickness_unpack(final_proxy.packed_thickness.x);\n"
"      if (proxy_thickness.value > 0.0f) {\n"
"        const float3 thickness_offset = thickness_intersection_offset(proxy_thickness, final_normal, ray_direction);\n"
"        const float thickness_distance = length(thickness_offset);\n"
"        if (thickness_distance > 1.0e-4f) {\n"
"          intersection_result<triangle_data, instancing, max_levels<2>> thickness_intersection =\n"
"              i.intersect(ray(ray_origin,\n"
"                              ray_direction,\n"
"                              hwrt_specular_ray_tmin(true),\n"
"                              thickness_distance),\n"
"                          scene);\n"
"          if (thickness_intersection.type != intersection_type::triangle) {\n"
"            ray_origin += thickness_offset;\n"
"            total_distance += thickness_distance;\n"
"          }\n"
"        }\n"
"      }\n"
"    }\n"
"  }\n"
"  if (preserved_layered_scene_final_hit || preserved_transparent_scene_final_hit) {\n"
"    const uint preserved_proxy_flags = uint(preserved_proxy.ior_closure_type.w + 0.5f);\n"
"    const bool preserved_has_transmission_layer =\n"
"        ((preserved_proxy_flags & HWRT_PROXY_FLAG_PRINCIPLED_TRANSMISSION_LAYER) != 0u);\n"
"    if (preserved_has_transmission_layer) {\n"
"      const bool preserved_transmission_needs_real_exit_hit =\n"
"          ((preserved_proxy_flags & HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL) != 0u) ||\n"
"          ((preserved_proxy_flags & HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL) != 0u);\n"
"      const float preserved_transmission_roughness =\n"
"          clamp(preserved_proxy.transmission_color_roughness.w, 0.0f, 1.0f);\n"
"      const float preserved_refraction_ior = max(preserved_proxy.ior_closure_type.y, 1.0e-3f);\n"
"      const bool preserved_entering = (preserved_front_facing != 0u);\n"
"      const float preserved_eta = preserved_entering ? (1.0f / preserved_refraction_ior) :\n"
"                                                     preserved_refraction_ior;\n"
"      float3 transmission_ray_direction = sample_rough_specular_direction(\n"
"          tid,\n"
"          start_bounce + 1,\n"
"          uniforms.closure_index + int(HWRT_CLOSURE_REFRACTION),\n"
"          preserved_direction,\n"
"          preserved_normal,\n"
"          preserved_transmission_roughness,\n"
"          true,\n"
"          preserved_eta,\n"
"          uniforms);\n"
"      if (dot(transmission_ray_direction, transmission_ray_direction) > 1.0e-10f) {\n"
"        float3 transmission_ray_origin = preserved_position +\n"
"                                        normalize(transmission_ray_direction) *\n"
"                                            hwrt_specular_ray_epsilon(true);\n"
"        transmission_ray_direction = normalize(transmission_ray_direction);\n"
"        float transmission_total_distance = preserved_total_distance;\n"
"        float3 transmission_throughput = float3(1.0f);\n"
"        if (preserved_entering && !preserved_transmission_needs_real_exit_hit) {\n"
"          const ThicknessData preserved_thickness = thickness_unpack(preserved_proxy.packed_thickness.x);\n"
"          if (preserved_thickness.value > 0.0f) {\n"
"            const float3 thickness_offset = thickness_intersection_offset(\n"
"                preserved_thickness, preserved_normal, transmission_ray_direction);\n"
"            const float thickness_distance = length(thickness_offset);\n"
"            if (thickness_distance > 1.0e-4f) {\n"
"              intersection_result<triangle_data, instancing, max_levels<2>> thickness_intersection =\n"
"                  i.intersect(ray(transmission_ray_origin,\n"
"                                  transmission_ray_direction,\n"
"                                  hwrt_specular_ray_tmin(true),\n"
"                                  thickness_distance),\n"
"                              scene);\n"
"              if (thickness_intersection.type != intersection_type::triangle) {\n"
"                transmission_ray_origin += thickness_offset;\n"
"                transmission_total_distance += thickness_distance;\n"
"              }\n"
"            }\n"
"          }\n"
"        }\n"
"        const int transmission_max_bounces = max(uniforms.refraction_bounces, 1);\n"
"        int transmission_thin_glass_passthrough_count = 0;\n"
"        for (int transmission_bounce = start_bounce + 1;\n"
"             transmission_bounce < transmission_max_bounces;\n"
"             transmission_bounce++) {\n"
"          intersection_result<triangle_data, instancing, max_levels<2>> transmission_intersection =\n"
"              i.intersect(ray(transmission_ray_origin,\n"
"                              transmission_ray_direction,\n"
"                              hwrt_specular_ray_tmin(true),\n"
"                              10000.0f),\n"
"                          scene);\n"
"          if (transmission_intersection.type != intersection_type::triangle) {\n"
"            if (transmission_receiver_valid) {\n"
"              transmission_receiver_continued_radiance += transmission_throughput *\n"
"                  min(sample_trace_world_radiance(world_probe_tx,\n"
"                                                 transmission_ray_direction,\n"
"                                                 uniforms.use_environment_pad.x != 0,\n"
"                                                 uniforms),\n"
"                      float3(uniforms.clamp_indirect));\n"
"              break;\n"
"            }\n"
"            transmission_receiver_valid = true;\n"
"            transmission_receiver_position = transmission_ray_origin;\n"
"            transmission_receiver_local_position = float3(0.0f);\n"
"            transmission_receiver_direction = transmission_ray_direction;\n"
"            transmission_receiver_total_distance = transmission_total_distance;\n"
"            transmission_receiver_segment_distance = 0.0f;\n"
"            transmission_receiver_normal = -transmission_ray_direction;\n"
"            transmission_receiver_barycentric = float2(0.0f);\n"
"            transmission_receiver_user_id = 0u;\n"
"            transmission_receiver_primitive_id = 0u;\n"
"            transmission_receiver_front_facing = 1u;\n"
"            transmission_receiver_proxy = preserved_proxy;\n"
"            transmission_receiver_proxy_closure = HWRT_CLOSURE_REFRACTION;\n"
"            transmission_receiver_carried_throughput = transmission_throughput;\n"
"            transmission_receiver_apply_throughput = true;\n"
"            transmission_receiver_direct_lit_reflective = false;\n"
"            break;\n"
"          }\n"
"          const float transmission_hit_time = transmission_intersection.distance;\n"
"          transmission_total_distance += transmission_hit_time;\n"
"          const float3 transmission_position = transmission_ray_origin +\n"
"                                              transmission_ray_direction * transmission_hit_time;\n"
"          const uint transmission_user_id = transmission_intersection.user_instance_id[0];\n"
"          const uint transmission_primitive_id = transmission_intersection.primitive_id;\n"
"          const float2 transmission_bary = transmission_intersection.triangle_barycentric_coord;\n"
"          if (transmission_receiver_valid) {\n"
"            transmission_receiver_continued_radiance += transmission_throughput *\n"
"                min(emissive_radiance[transmission_user_id].xyz, float3(uniforms.clamp_indirect));\n"
"          }\n"
"          HardwareMaterialProxy transmission_proxy = material_proxy[transmission_user_id];\n"
"          float3 transmission_raw_hit_normal = float3(0.0f);\n"
"          float3 transmission_smooth_hit_normal = float3(0.0f);\n"
"          float3 transmission_local_position = float3(0.0f);\n"
"          const TriangleNormalRange transmission_normal_range =\n"
"              triangle_normal_ranges[transmission_user_id];\n"
"          if (transmission_primitive_id < transmission_normal_range.count) {\n"
"            transmission_raw_hit_normal =\n"
"                triangle_normals[transmission_normal_range.offset + transmission_primitive_id].xyz;\n"
"            const uint transmission_smooth_offset =\n"
"                (transmission_normal_range.offset + transmission_primitive_id) * 3u;\n"
"            const float3 transmission_bary3 = barycentric_expand(transmission_bary);\n"
"            transmission_local_position =\n"
"                triangle_local_positions[transmission_smooth_offset + 0u].xyz * transmission_bary3.x +\n"
"                triangle_local_positions[transmission_smooth_offset + 1u].xyz * transmission_bary3.y +\n"
"                triangle_local_positions[transmission_smooth_offset + 2u].xyz * transmission_bary3.z;\n"
"            transmission_smooth_hit_normal =\n"
"                triangle_smooth_normals[transmission_smooth_offset + 0u].xyz * transmission_bary3.x +\n"
"                triangle_smooth_normals[transmission_smooth_offset + 1u].xyz * transmission_bary3.y +\n"
"                triangle_smooth_normals[transmission_smooth_offset + 2u].xyz * transmission_bary3.z;\n"
"          }\n"
"          bool transmission_entering = true;\n"
"          float3 transmission_hit_normal = transmission_smooth_hit_normal;\n"
"          if (!(dot(transmission_hit_normal, transmission_hit_normal) > 1.0e-10f)) {\n"
"            transmission_hit_normal = transmission_raw_hit_normal;\n"
"          }\n"
"          uint transmission_front_facing = 1u;\n"
"          float3 transmission_normal = -transmission_ray_direction;\n"
"          if (dot(transmission_hit_normal, transmission_hit_normal) > 1.0e-10f) {\n"
"            transmission_entering = dot(transmission_hit_normal, transmission_ray_direction) < 0.0f;\n"
"            transmission_front_facing = transmission_entering ? 1u : 0u;\n"
"            transmission_normal = transmission_entering ? transmission_hit_normal :\n"
"                                                        -transmission_hit_normal;\n"
"          }\n"
"          const uint transmission_proxy_closure = uint(transmission_proxy.ior_closure_type.z + 0.5f);\n"
"          const uint transmission_proxy_flags = uint(transmission_proxy.ior_closure_type.w + 0.5f);\n"
"          const float transmission_reflection_roughness =\n"
"              clamp(transmission_proxy.reflection_color_roughness.w, 0.0f, 1.0f);\n"
"          const float transmission_refraction_roughness =\n"
"              clamp(transmission_proxy.transmission_color_roughness.w, 0.0f, 1.0f);\n"
"          const float transmission_proxy_ior =\n"
"              max(transmission_proxy.ior_closure_type.y, 1.0e-3f);\n"
"          const float transmission_eta = transmission_entering ? (1.0f / transmission_proxy_ior) :\n"
"                                                              transmission_proxy_ior;\n"
"          if (((transmission_proxy_flags & HWRT_PROXY_FLAG_THIN_GLASS) != 0u) &&\n"
"              (transmission_thin_glass_passthrough_count < 8)) {\n"
"            transmission_thin_glass_passthrough_count++;\n"
"            transmission_ray_origin = transmission_position + transmission_ray_direction *\n"
"                                      hwrt_specular_ray_epsilon(false);\n"
"            transmission_bounce -= 1;\n"
"            continue;\n"
"          }\n"
"          uint transmission_resolved_proxy_closure = transmission_proxy_closure;\n"
"          if ((transmission_proxy_closure == HWRT_CLOSURE_REFRACTION) &&\n"
"              ((transmission_proxy_flags & HWRT_PROXY_FLAG_DIELECTRIC_REFLECTION) != 0u)) {\n"
"            const float3 refracted = refract(\n"
"                transmission_ray_direction, transmission_normal, transmission_eta);\n"
"            const bool has_refraction = dot(refracted, refracted) > 1.0e-10f;\n"
"            const float fresnel = dielectric_fresnel_reflectance(\n"
"                transmission_ray_direction, transmission_normal, transmission_proxy_ior);\n"
"            const float branch_rand = rand2_trace(\n"
"                tid,\n"
"                transmission_bounce + 1,\n"
"                uniforms.closure_index + int(HWRT_CLOSURE_REFLECTION + HWRT_CLOSURE_REFRACTION),\n"
"                uniforms).x;\n"
"            if (!has_refraction || (branch_rand < fresnel)) {\n"
"              transmission_resolved_proxy_closure = HWRT_CLOSURE_REFLECTION;\n"
"            }\n"
"          }\n"
"          const bool transmission_replay_reflective_receiver =\n"
"              ((transmission_proxy_flags & HWRT_PROXY_FLAG_ALPHA_BLEND_TRANSPARENT) == 0u) &&\n"
"              (transmission_resolved_proxy_closure == HWRT_CLOSURE_REFLECTION);\n"
"          const bool transmission_replay_textured_reflective_receiver =\n"
"              transmission_replay_reflective_receiver &&\n"
"              (((transmission_proxy_flags & HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL) != 0u) ||\n"
"               ((transmission_proxy_flags & HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL) != 0u));\n"
"          const bool transmission_replay_textured_refractive_receiver =\n"
"              !transmission_entering &&\n"
"              (transmission_resolved_proxy_closure == HWRT_CLOSURE_REFRACTION) &&\n"
"              (((transmission_proxy_flags & HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL) != 0u) ||\n"
"               ((transmission_proxy_flags & HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL) != 0u));\n"
"          const bool transmission_can_continue =\n"
"              (transmission_bounce + 1 < transmission_max_bounces) &&\n"
"              ((transmission_resolved_proxy_closure == HWRT_CLOSURE_REFLECTION) ||\n"
"               (transmission_resolved_proxy_closure == HWRT_CLOSURE_REFRACTION));\n"
"          if ((transmission_replay_reflective_receiver ||\n"
"               transmission_replay_textured_refractive_receiver) &&\n"
"              !transmission_receiver_valid) {\n"
"            transmission_receiver_valid = true;\n"
"            transmission_receiver_position = transmission_position;\n"
"            transmission_receiver_local_position = transmission_local_position;\n"
"            transmission_receiver_direction = transmission_ray_direction;\n"
"            transmission_receiver_total_distance = transmission_total_distance;\n"
"            transmission_receiver_segment_distance = transmission_hit_time;\n"
"            transmission_receiver_normal = transmission_normal;\n"
"            transmission_receiver_barycentric = transmission_bary;\n"
"            transmission_receiver_user_id = transmission_user_id;\n"
"            transmission_receiver_primitive_id = transmission_primitive_id;\n"
"            transmission_receiver_front_facing = transmission_front_facing;\n"
"            transmission_receiver_proxy = transmission_proxy;\n"
"            transmission_receiver_proxy_closure = transmission_resolved_proxy_closure;\n"
"            transmission_receiver_carried_throughput = transmission_throughput;\n"
"            transmission_receiver_direct_lit_reflective = transmission_replay_reflective_receiver;\n"
"            transmission_receiver_lock_surface = transmission_replay_textured_reflective_receiver ||\n"
"                                                 transmission_replay_textured_refractive_receiver;\n"
"          }\n"
"          if (!transmission_can_continue) {\n"
"            if (transmission_receiver_valid) {\n"
"              if (transmission_receiver_lock_surface) {\n"
"                break;\n"
"              }\n"
"              transmission_receiver_position = transmission_position;\n"
"              transmission_receiver_local_position = transmission_local_position;\n"
"              transmission_receiver_direction = transmission_ray_direction;\n"
"              transmission_receiver_total_distance = transmission_total_distance;\n"
"              transmission_receiver_segment_distance = transmission_hit_time;\n"
"              transmission_receiver_normal = transmission_normal;\n"
"              transmission_receiver_barycentric = transmission_bary;\n"
"              transmission_receiver_user_id = transmission_user_id;\n"
"              transmission_receiver_primitive_id = transmission_primitive_id;\n"
"              transmission_receiver_front_facing = transmission_front_facing;\n"
"              transmission_receiver_proxy = transmission_proxy;\n"
"              transmission_receiver_proxy_closure = transmission_resolved_proxy_closure;\n"
"              transmission_receiver_carried_throughput = transmission_throughput;\n"
"              transmission_receiver_direct_lit_reflective = transmission_replay_reflective_receiver;\n"
"              break;\n"
"            }\n"
"            transmission_receiver_valid = true;\n"
"            transmission_receiver_position = transmission_position;\n"
"            transmission_receiver_local_position = transmission_local_position;\n"
"            transmission_receiver_direction = transmission_ray_direction;\n"
"            transmission_receiver_total_distance = transmission_total_distance;\n"
"            transmission_receiver_segment_distance = transmission_hit_time;\n"
"            transmission_receiver_normal = transmission_normal;\n"
"            transmission_receiver_barycentric = transmission_bary;\n"
"            transmission_receiver_user_id = transmission_user_id;\n"
"            transmission_receiver_primitive_id = transmission_primitive_id;\n"
"            transmission_receiver_front_facing = transmission_front_facing;\n"
"            transmission_receiver_proxy = transmission_proxy;\n"
"            transmission_receiver_proxy_closure = transmission_resolved_proxy_closure;\n"
"            transmission_receiver_carried_throughput = transmission_throughput;\n"
"            transmission_receiver_direct_lit_reflective = transmission_replay_reflective_receiver;\n"
"            break;\n"
"          }\n"
"          float3 transmission_next_direction = transmission_ray_direction;\n"
"          if (transmission_resolved_proxy_closure == HWRT_CLOSURE_REFLECTION) {\n"
"            if (!transmission_replay_reflective_receiver) {\n"
"              transmission_throughput *= clamp(\n"
"                  transmission_proxy.reflection_color_roughness.xyz,\n"
"                  float3(0.0f),\n"
"                  float3(uniforms.clamp_indirect));\n"
"            }\n"
"            transmission_receiver_apply_throughput = true;\n"
"            transmission_next_direction = sample_rough_specular_direction(\n"
"                tid,\n"
"                transmission_bounce + 1,\n"
"                uniforms.closure_index + int(HWRT_CLOSURE_REFLECTION),\n"
"                transmission_ray_direction,\n"
"                transmission_normal,\n"
"                transmission_reflection_roughness,\n"
"                false,\n"
"                1.0f,\n"
"                uniforms);\n"
"          }\n"
"          else {\n"
"            transmission_throughput *= clamp(\n"
"                transmission_proxy.transmission_color_roughness.xyz,\n"
"                float3(0.0f),\n"
"                float3(uniforms.clamp_indirect));\n"
"            transmission_receiver_apply_throughput = true;\n"
"            transmission_next_direction = sample_rough_specular_direction(\n"
"                tid,\n"
"                transmission_bounce + 1,\n"
"                uniforms.closure_index + int(HWRT_CLOSURE_REFRACTION),\n"
"                transmission_ray_direction,\n"
"                transmission_normal,\n"
"                transmission_refraction_roughness,\n"
"                true,\n"
"                transmission_eta,\n"
"                uniforms);\n"
"          }\n"
"          if (!(dot(transmission_next_direction, transmission_next_direction) > 1.0e-10f)) {\n"
"            break;\n"
"          }\n"
"          transmission_ray_direction = normalize(transmission_next_direction);\n"
"          transmission_ray_origin = transmission_position + transmission_ray_direction *\n"
"              hwrt_specular_ray_epsilon(transmission_resolved_proxy_closure == HWRT_CLOSURE_REFRACTION);\n"
"          const bool transmission_needs_real_exit_hit =\n"
"              ((transmission_proxy_flags & HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL) != 0u) ||\n"
"              ((transmission_proxy_flags & HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL) != 0u);\n"
"          if ((transmission_resolved_proxy_closure == HWRT_CLOSURE_REFRACTION) &&\n"
"              transmission_entering && !transmission_needs_real_exit_hit) {\n"
"            const ThicknessData transmission_thickness = thickness_unpack(\n"
"                transmission_proxy.packed_thickness.x);\n"
"            if (transmission_thickness.value > 0.0f) {\n"
"              const float3 thickness_offset = thickness_intersection_offset(\n"
"                  transmission_thickness, transmission_normal, transmission_ray_direction);\n"
"              const float thickness_distance = length(thickness_offset);\n"
"              if (thickness_distance > 1.0e-4f) {\n"
"                intersection_result<triangle_data, instancing, max_levels<2>> thickness_intersection =\n"
"                    i.intersect(ray(transmission_ray_origin,\n"
"                                    transmission_ray_direction,\n"
"                                    hwrt_specular_ray_tmin(true),\n"
"                                    thickness_distance),\n"
"                                scene);\n"
"                if (thickness_intersection.type != intersection_type::triangle) {\n"
"                  transmission_ray_origin += thickness_offset;\n"
"                  transmission_total_distance += thickness_distance;\n"
"                }\n"
"              }\n"
"            }\n"
"          }\n"
"        }\n"
"      }\n"
"    }\n"
"  }\n"
"  if (preserved_layered_scene_final_hit || preserved_scene_final_reflective_hit ||\n"
"      preserved_transparent_scene_final_hit) {\n"
"    total_distance = preserved_total_distance;\n"
"    final_position = preserved_position;\n"
"    final_local_position = preserved_local_position;\n"
"    final_direction = preserved_direction;\n"
"    final_segment_distance = preserved_segment_distance;\n"
"    final_normal = preserved_normal;\n"
"    final_barycentric = preserved_barycentric;\n"
"    final_user_id = preserved_user_id;\n"
"    final_primitive_id = preserved_primitive_id;\n"
"    final_front_facing = preserved_front_facing;\n"
"    final_proxy = preserved_proxy;\n"
"    final_proxy_closure = preserved_proxy_closure;\n"
"  }\n"
"  float3 final_output_throughput = throughput;\n"
"  if (preserved_transparent_scene_final_hit) {\n"
"    final_output_throughput = preserved_output_throughput;\n"
"  }\n"
"  /* Keep this scene-final trace responsible for exporting the hit payload only.\n"
"   * Secondary GI receiver injection is removed from the active Nuru runtime. */\n"
"  const bool primary_diffuse_transport_receiver = supports_hardware_gi &&\n"
"                                                  !scene_final_specular_phase &&\n"
"                                                  (final_proxy_closure == HWRT_CLOSURE_DIFFUSE);\n"
"  if (primary_diffuse_transport_receiver) {\n"
"    const int diffuse_sample_count = max(uniforms.use_environment_pad.z, 1);\n"
"    const bool use_emissive_mixture = (uniforms.use_environment_pad.y > 0) &&\n"
"                                      (diffuse_sample_count > 1);\n"
"    const float3 diffuse_gather_N = fast_gi_hit_normal(\n"
"        final_user_id,\n"
"        final_primitive_id,\n"
"        final_direction,\n"
"        triangle_normals,\n"
"        triangle_normal_ranges);\n"
"    const float diffuse_origin_epsilon = hwrt_gi_ray_epsilon(max(final_segment_distance, total_distance));\n"
"    const float diffuse_ray_tmin = diffuse_origin_epsilon;\n"
"    const float diffuse_self_hit_tmax = hwrt_gi_self_hit_distance(diffuse_origin_epsilon);\n"
"    const float3 diffuse_origin = final_position + diffuse_gather_N * diffuse_origin_epsilon;\n"
"    float3 incoming = float3(0.0f);\n"
"    for (int diffuse_sample = 0; diffuse_sample < diffuse_sample_count; diffuse_sample++) {\n"
"      const bool use_emissive_guiding = use_emissive_mixture &&\n"
"                                        ((diffuse_sample & 1) == 0);\n"
"      float4 guided_sample = float4(0.0f);\n"
"      float3 diffuse_dir;\n"
"      float diffuse_weight = 1.0f;\n"
"      if (use_emissive_guiding) {\n"
"        guided_sample = sample_trace_emissive_direction(\n"
"            tid, diffuse_sample, diffuse_origin, emissive_lights, uniforms);\n"
"        diffuse_dir = guided_sample.xyz;\n"
"        const float cosine_pdf = saturate(dot(diffuse_gather_N, diffuse_dir)) * 0.31830988618f;\n"
"        const float mixture_pdf = max(0.5f * guided_sample.w + 0.5f * cosine_pdf, 1.0e-6f);\n"
"        diffuse_weight = cosine_pdf / mixture_pdf;\n"
"      }\n"
"      else {\n"
"        diffuse_dir = sample_trace_diffuse_direction(\n"
"            tid, diffuse_sample, uniforms.closure_index + 37 + diffuse_sample * 13, diffuse_gather_N, uniforms);\n"
"        diffuse_weight = use_emissive_mixture ? 2.0f : 1.0f;\n"
"      }\n"
"      if (!(dot(diffuse_dir, diffuse_dir) > 1.0e-10f) || diffuse_weight <= 0.0f) {\n"
"        continue;\n"
"      }\n"
"      intersection_result<triangle_data, instancing, max_levels<2>> diffuse_intersection =\n"
"          i.intersect(ray(diffuse_origin, diffuse_dir, diffuse_ray_tmin, 10000.0f), scene);\n"
"      if (diffuse_intersection.type == intersection_type::triangle &&\n"
"          diffuse_intersection.distance <= diffuse_self_hit_tmax &&\n"
"          diffuse_intersection.user_instance_id[0] == final_user_id &&\n"
"          diffuse_intersection.primitive_id == final_primitive_id)\n"
"      {\n"
"        /* Nuru: ignore only the exact source triangle self-hit. Do not hide nearby blockers with\n"
"         * a large launch bias; sealed wall shells must remain visible to the dome ray. */\n"
"        const float3 retry_origin = diffuse_origin + diffuse_dir * (diffuse_intersection.distance +\n"
"                                                                   diffuse_ray_tmin);\n"
"        diffuse_intersection = i.intersect(ray(retry_origin, diffuse_dir, diffuse_ray_tmin, 10000.0f), scene);\n"
"      }\n"
"      if (diffuse_intersection.type == intersection_type::triangle) {\n"
"        const uint diffuse_user_id = diffuse_intersection.user_instance_id[0];\n"
"        incoming += min(max(emissive_radiance[diffuse_user_id].xyz, float3(0.0f)),\n"
"                        float3(uniforms.clamp_indirect)) * diffuse_weight;\n"
"        const float3 diffuse_hit_P = diffuse_origin + diffuse_dir * diffuse_intersection.distance;\n"
"        const float3 diffuse_hit_N = fast_gi_hit_normal(\n"
"            diffuse_user_id,\n"
"            diffuse_intersection.primitive_id,\n"
"            diffuse_dir,\n"
"            triangle_normals,\n"
"            triangle_normal_ranges);\n"
"        incoming += sample_trace_direct_light(tid,\n"
"                                              diffuse_sample,\n"
"                                              diffuse_sample_count,\n"
"                                              diffuse_hit_P,\n"
"                                              diffuse_hit_N,\n"
"                                              diffuse_user_id,\n"
"                                              diffuse_intersection.primitive_id,\n"
"                                              true,\n"
"                                              false,\n"
"                                              scene,\n"
"                                              trace_lights,\n"
"                                              nis_weights,\n"
"                                              uniforms) * diffuse_weight;\n"
"      }\n"
"      else if (!use_emissive_guiding) {\n"
"        incoming += min(sample_trace_world_radiance(\n"
"                            world_probe_tx, diffuse_dir, uniforms.use_environment_pad.w != 0, uniforms),\n"
"                        float3(uniforms.clamp_indirect)) * diffuse_weight;\n"
"      }\n"
"    }\n"
"    incoming /= float(diffuse_sample_count);\n"
"    /* Keep diffuse GI demodulated here; the full-resolution G-buffer color is applied later. */\n"
"    radiance += min(final_output_throughput * incoming,\n"
"                    float3(uniforms.clamp_indirect));\n"
"  }\n"
"  /* Nuru Secondary GI (user architecture): per-pixel diffuse final gather at scene-final\n"
"   * receiver hits. Mirror-visible diffuse surfaces get their one-bounce indirect from real\n"
"   * rays; the noisy term rides the traced radiance into the shared scene-final OIDN denoise\n"
"   * and temporal accumulation. Direct light at the receiver stays owned by the hit-lighting\n"
"   * kernel; the gather's dome hits have no other transport, so their NEE samples the full\n"
"   * light tree (suns + locals) without double counting. */\n"
"  const float2 packed_direction = direction_pack(final_direction);\n"
"  ray_time_img.write(float4(max(total_distance, 1.0e-4f), 0.0f, 0.0f, 0.0f), tid);\n"
"  ray_radiance_img.write(float4(radiance, 0.0f), tid);\n"
"  float3 final_proxy_color = (final_proxy_closure == HWRT_CLOSURE_REFLECTION) ?\n"
"                                 final_proxy.reflection_color_roughness.xyz :\n"
"                             (final_proxy_closure == HWRT_CLOSURE_REFRACTION) ?\n"
"                                 final_proxy.transmission_color_roughness.xyz :\n"
"                                 diffuse_albedo[final_user_id].xyz;\n"
"  const uint final_proxy_flags = uint(final_proxy.ior_closure_type.w + 0.5f);\n"
"  const bool final_textured_specular_scene_final_hit = final_layered_principled_scene_final_hit &&\n"
"      (final_proxy_closure == HWRT_CLOSURE_REFLECTION) &&\n"
"      ((final_proxy_flags & HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL) != 0u);\n"
"  const bool final_metallic_bsdf_scene_final_hit = final_textured_specular_scene_final_hit &&\n"
"      ((final_proxy_flags & HWRT_PROXY_FLAG_METALLIC_BSDF_SCENE_FINAL) != 0u);\n"
"  if ((final_proxy_flags & HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL) != 0u) {\n"
"    final_proxy_color = float3(1.0f);\n"
"  }\n"
"  float final_proxy_roughness = (final_proxy_closure == HWRT_CLOSURE_REFLECTION) ?\n"
"                                    final_proxy.reflection_color_roughness.w :\n"
"                                    final_proxy.transmission_color_roughness.w;\n"
"  float final_proxy_ior = (final_proxy_closure == HWRT_CLOSURE_REFLECTION) ?\n"
"                              final_proxy.ior_closure_type.x :\n"
"                              final_proxy.ior_closure_type.y;\n"
"  hit_albedo_img.write(float4(clamp(final_output_throughput * final_proxy_color,\n"
"                                float3(0.0f),\n"
"                                float3(uniforms.clamp_indirect)),\n"
"                              -1.0f),\n"
"                        tid);\n"
"  hit_material_img.write(float4(final_proxy_roughness,\n"
"                                 final_proxy_ior,\n"
"                                 float(final_proxy_closure),\n"
"                                 packed_direction.x),\n"
"                         tid);\n"
"  hit_normal_img.write(float4(final_normal, packed_direction.y), tid);\n"
"  hit_position_img.write(float4(final_local_position, total_distance), tid);\n"
"  hit_world_position_img.write(float4(final_position, total_distance), tid);\n"
"  hit_throughput_img.write(float4(apply_scene_final_throughput ? carried_scene_final_throughput :\n"
"                                       float3(1.0f),\n"
"                                   apply_scene_final_throughput ? 1.0f : 0.0f),\n"
"                           tid);\n"
"  const uint identity_flags = final_front_facing |\n"
"                              ((final_layered_principled_scene_final_hit ||\n"
"                                preserved_scene_final_reflective_hit) ? 2u : 0u) |\n"
"                              (final_transparent_scene_final_hit ? 4u : 0u) |\n"
"                              ((final_refracted_textured_receiver_hit ||\n"
"                                final_textured_specular_scene_final_hit) ? 16u : 0u) |\n"
"                              (((final_proxy_flags & HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL) != 0u) ?\n"
"                                   HWRT_HIT_IDENTITY_PRINCIPLED_LAYERED_SCENE_FINAL :\n"
"                                   0u) |\n"
"                              (final_metallic_bsdf_scene_final_hit ?\n"
"                                   HWRT_HIT_IDENTITY_METALLIC_BSDF_SCENE_FINAL :\n"
"                                   0u);\n"
"  hit_identity_img.write(uint4(final_user_id, final_primitive_id, identity_flags, 0xFFFFFFFFu), tid);\n"
"  const float final_reflection_layer_coverage =\n"
"      ((final_proxy_flags & HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL) != 0u) ?\n"
"          clamp(final_proxy.packed_thickness.z, 0.0f, 1.0f) :\n"
"          1.0f;\n"
"  hit_barycentric_img.write(float4(final_barycentric.x,\n"
"                                  final_barycentric.y,\n"
"                                  final_segment_distance,\n"
"                                  final_reflection_layer_coverage),\n"
"                           tid);\n"
"  if (layered_receiver_valid) {\n"
"    const float2 layered_receiver_packed_direction = direction_pack(layered_receiver_direction);\n"
"    const uint layered_receiver_proxy_flags = uint(layered_receiver_proxy.ior_closure_type.w + 0.5f);\n"
"    float3 layered_receiver_proxy_color =\n"
"        (layered_receiver_proxy_closure == HWRT_CLOSURE_REFLECTION) ?\n"
"            layered_receiver_proxy.reflection_color_roughness.xyz :\n"
"        (layered_receiver_proxy_closure == HWRT_CLOSURE_REFRACTION) ?\n"
"            layered_receiver_proxy.transmission_color_roughness.xyz :\n"
"            diffuse_albedo[layered_receiver_user_id].xyz;\n"
"    if ((layered_receiver_proxy_flags & HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL) != 0u) {\n"
"      layered_receiver_proxy_color = float3(1.0f);\n"
"    }\n"
"    const float layered_receiver_proxy_roughness =\n"
"        (layered_receiver_proxy_closure == HWRT_CLOSURE_REFLECTION) ?\n"
"            layered_receiver_proxy.reflection_color_roughness.w :\n"
"            layered_receiver_proxy.transmission_color_roughness.w;\n"
"    const float layered_receiver_proxy_ior =\n"
"        (layered_receiver_proxy_closure == HWRT_CLOSURE_REFLECTION) ?\n"
"            layered_receiver_proxy.ior_closure_type.x :\n"
"            layered_receiver_proxy.ior_closure_type.y;\n"
"    const float layered_receiver_reflection_layer_coverage =\n"
"        ((layered_receiver_proxy_flags & HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL) != 0u) ?\n"
"            clamp(layered_receiver_proxy.packed_thickness.z, 0.0f, 1.0f) :\n"
"            1.0f;\n"
"    layered_receiver_ray_time_img.write(\n"
"        float4(max(layered_receiver_total_distance, 1.0e-4f), 0.0f, 0.0f, 0.0f), tid);\n"
"    layered_receiver_ray_radiance_img.write(float4(layered_receiver_continued_radiance, 0.0f), tid);\n"
"    layered_receiver_albedo_img.write(\n"
"        float4(clamp(layered_receiver_proxy_color,\n"
"                     float3(0.0f),\n"
"                     float3(uniforms.clamp_indirect)),\n"
"               -1.0f),\n"
"        tid);\n"
"    layered_receiver_material_img.write(float4(layered_receiver_proxy_roughness,\n"
"                                               layered_receiver_proxy_ior,\n"
"                                               float(layered_receiver_proxy_closure),\n"
"                                               layered_receiver_packed_direction.x),\n"
"                                       tid);\n"
"    layered_receiver_normal_img.write(float4(layered_receiver_normal, layered_receiver_packed_direction.y), tid);\n"
"    layered_receiver_position_img.write(\n"
"        float4(layered_receiver_local_position, layered_receiver_total_distance), tid);\n"
"    layered_receiver_world_position_img.write(\n"
"        float4(layered_receiver_position, layered_receiver_total_distance), tid);\n"
"    const float layered_receiver_throughput_alpha =\n"
"        (preserved_transparent_scene_final_hit ||\n"
"         (scene_final_specular_phase && continuation_required)) ?\n"
"            1.0f :\n"
"            0.0f;\n"
"    layered_receiver_throughput_img.write(\n"
"        float4(layered_receiver_carried_throughput, layered_receiver_throughput_alpha),\n"
"        tid);\n"
"    const uint layered_receiver_identity_flags =\n"
"        layered_receiver_front_facing | 8u |\n"
"        (((layered_receiver_proxy_closure == HWRT_CLOSURE_REFLECTION) &&\n"
"          ((layered_receiver_proxy_flags & HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL) != 0u)) ?\n"
"             16u :\n"
"             0u) |\n"
"        (((layered_receiver_proxy_flags & HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL) != 0u) ?\n"
"             2u :\n"
"             0u) |\n"
"        (((layered_receiver_proxy_flags & HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL) != 0u) ?\n"
"             HWRT_HIT_IDENTITY_PRINCIPLED_LAYERED_SCENE_FINAL :\n"
"             0u);\n"
"    layered_receiver_identity_img.write(uint4(layered_receiver_user_id,\n"
"                                              layered_receiver_primitive_id,\n"
"                                              layered_receiver_identity_flags,\n"
"                                              0xFFFFFFFFu),\n"
"                                      tid);\n"
"    layered_receiver_barycentric_img.write(\n"
"        float4(layered_receiver_barycentric.x,\n"
"               layered_receiver_barycentric.y,\n"
"               layered_receiver_segment_distance,\n"
"               layered_receiver_reflection_layer_coverage),\n"
"        tid);\n"
"  }\n"
"  if (transmission_receiver_valid) {\n"
"    const float2 transmission_receiver_packed_direction = direction_pack(\n"
"        transmission_receiver_direction);\n"
"    const uint transmission_receiver_proxy_flags = uint(\n"
"        transmission_receiver_proxy.ior_closure_type.w + 0.5f);\n"
"    float3 transmission_receiver_proxy_color =\n"
"        (transmission_receiver_proxy_closure == HWRT_CLOSURE_REFLECTION) ?\n"
"            transmission_receiver_proxy.reflection_color_roughness.xyz :\n"
"        (transmission_receiver_proxy_closure == HWRT_CLOSURE_REFRACTION) ?\n"
"            transmission_receiver_proxy.transmission_color_roughness.xyz :\n"
"            diffuse_albedo[transmission_receiver_user_id].xyz;\n"
"    if ((transmission_receiver_proxy_flags & HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL) != 0u) {\n"
"      transmission_receiver_proxy_color = float3(1.0f);\n"
"    }\n"
"    const float transmission_receiver_proxy_roughness =\n"
"        (transmission_receiver_proxy_closure == HWRT_CLOSURE_REFLECTION) ?\n"
"            transmission_receiver_proxy.reflection_color_roughness.w :\n"
"            transmission_receiver_proxy.transmission_color_roughness.w;\n"
"    const float transmission_receiver_proxy_ior =\n"
"        (transmission_receiver_proxy_closure == HWRT_CLOSURE_REFLECTION) ?\n"
"            transmission_receiver_proxy.ior_closure_type.x :\n"
"            transmission_receiver_proxy.ior_closure_type.y;\n"
"    const float transmission_receiver_reflection_layer_coverage =\n"
"        ((transmission_receiver_proxy_flags & HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL) != 0u) ?\n"
"            clamp(transmission_receiver_proxy.packed_thickness.z, 0.0f, 1.0f) :\n"
"            1.0f;\n"
"    transmission_receiver_ray_time_img.write(\n"
"        float4(max(transmission_receiver_total_distance, 1.0e-4f), 0.0f, 0.0f, 0.0f), tid);\n"
"    transmission_receiver_ray_radiance_img.write(float4(transmission_receiver_continued_radiance, 0.0f), tid);\n"
"    transmission_receiver_albedo_img.write(\n"
"        float4(clamp(transmission_receiver_proxy_color,\n"
"                     float3(0.0f),\n"
"                     float3(uniforms.clamp_indirect)),\n"
"               -1.0f),\n"
"        tid);\n"
"    transmission_receiver_material_img.write(\n"
"        float4(transmission_receiver_proxy_roughness,\n"
"               transmission_receiver_proxy_ior,\n"
"               float(transmission_receiver_proxy_closure),\n"
"               transmission_receiver_packed_direction.x),\n"
"        tid);\n"
"    transmission_receiver_normal_img.write(\n"
"        float4(transmission_receiver_normal, transmission_receiver_packed_direction.y), tid);\n"
"    transmission_receiver_position_img.write(\n"
"        float4(transmission_receiver_local_position, transmission_receiver_total_distance), tid);\n"
"    transmission_receiver_world_position_img.write(\n"
"        float4(transmission_receiver_position, transmission_receiver_total_distance), tid);\n"
"    transmission_receiver_throughput_img.write(\n"
"        float4(transmission_receiver_carried_throughput,\n"
"               transmission_receiver_apply_throughput ? 1.0f : 0.0f),\n"
"        tid);\n"
"    const bool transmission_receiver_textured_replay =\n"
"        transmission_receiver_direct_lit_reflective ||\n"
"        ((transmission_receiver_proxy_flags & HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL) != 0u) ||\n"
"        ((transmission_receiver_proxy_flags & HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL) != 0u);\n"
"    const uint transmission_receiver_identity_flags =\n"
"        transmission_receiver_front_facing | 8u |\n"
"        (transmission_receiver_textured_replay ? 16u : 0u) |\n"
"        (((transmission_receiver_proxy_flags & HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL) != 0u) ?\n"
"             2u :\n"
"             0u) |\n"
"        (((transmission_receiver_proxy_flags & HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL) != 0u) ?\n"
"             HWRT_HIT_IDENTITY_PRINCIPLED_LAYERED_SCENE_FINAL :\n"
"             0u);\n"
"    transmission_receiver_identity_img.write(uint4(transmission_receiver_user_id,\n"
"                                                   transmission_receiver_primitive_id,\n"
"                                                   transmission_receiver_identity_flags,\n"
"                                                   0xFFFFFFFFu),\n"
"                                           tid);\n"
"    transmission_receiver_barycentric_img.write(\n"
"        float4(transmission_receiver_barycentric.x,\n"
"               transmission_receiver_barycentric.y,\n"
"               transmission_receiver_segment_distance,\n"
"               transmission_receiver_reflection_layer_coverage),\n"
"        tid);\n"
"  }\n"
         "}\n"
         /* Nuru: HWRT transparent-shadow accumulation.
          *
          * The vanilla shadow kernels use `force_opacity::opaque` + a single `intersect()` and
          * treat any triangle hit as a fully occluded sample. That makes glass/refraction cast a
          * pitch-black shadow, which is wrong for the Nuru workflow: the user explicitly wants
          * glass to attenuate light by its transmission color instead of blocking it.
          *
          * `hardware_shadow_visibility` performs one material-aware shadow trace. Opaque mode
          * (transparency = 0) is a single any-hit test. Transparent mode walks glass/alpha with
          * Beer-Lambert tint and Color Transmission saturation. Mid values reuse the same walk
          * and scale the tinted result: any geometry along the ray blocks the opaque component,
          * so mix(opaque, tinted, t) = tinted * t when geometry was hit. */
"/* Nuru: classic opaque HWRT shadow. Thin Glass is pass-through; any other triangle blocks. */\n"
"inline float3 opaque_shadow_visibility(\n"
"    instance_acceleration_structure scene,\n"
"    float3 origin,\n"
"    float3 dir,\n"
"    float t_min,\n"
"    float t_max,\n"
"    constant HardwareMaterialProxy *material_proxies)\n"
"{\n"
"  intersector<triangle_data, instancing, max_levels<2>> ix;\n"
"  ix.assume_geometry_type(geometry_type::triangle);\n"
"  ix.force_opacity(forced_opacity::opaque);\n"
"  float current_tmin = t_min;\n"
"  for (int b = 0; b < 8; b++) {\n"
"    const intersection_result<triangle_data, instancing, max_levels<2>> hit =\n"
"        ix.intersect(ray(origin, dir, current_tmin, t_max), scene);\n"
"    if (hit.type != intersection_type::triangle) {\n"
"      return float3(1.0f);\n"
"    }\n"
"    const HardwareMaterialProxy proxy = material_proxies[hit.user_instance_id[0]];\n"
"    const uint proxy_flags = uint(proxy.ior_closure_type.w);\n"
"    if ((proxy_flags & HWRT_PROXY_FLAG_THIN_GLASS) == 0u) {\n"
"      return float3(0.0f);\n"
"    }\n"
"    current_tmin = hit.distance + 1.0e-4f;\n"
"    if (current_tmin >= t_max) {\n"
"      return float3(1.0f);\n"
"    }\n"
"  }\n"
"  return float3(0.0f);\n"
"}\n"
"inline float3 hardware_shadow_apply_color_transmission(float3 tint, float color_intensity)\n"
"{\n"
"  const float luma = dot(tint, float3(0.2126f, 0.7152f, 0.0722f));\n"
"  return mix(float3(luma), tint, saturate(color_intensity));\n"
"}\n"
"/* Nuru: Snell-bent caustic trace. Returns ONLY the caustic contribution (tinted throughput\n"
" * along the refracted path, modulated by a tight alignment cone against the light's\n"
" * direction at the exit, scaled by Photons intensity). Returns float3(0) if the ray never\n"
" * actually bent (no refractive surface hit) so it contributes nothing to non-glass paths. */\n"
"inline float3 transparent_shadow_caustic_only(\n"
"    instance_acceleration_structure scene,\n"
"    float3 origin,\n"
"    float3 dir,\n"
"    float t_min,\n"
"    float t_max,\n"
"    constant HardwareMaterialProxy *material_proxies,\n"
"    constant float4 *triangle_normals,\n"
"    constant float4 *triangle_smooth_normals,\n"
"    constant TriangleNormalRange *triangle_normal_ranges,\n"
"    bool is_directional,\n"
"    float3 light_pos_or_dir,\n"
"    float photons_intensity)\n"
"{\n"
"  intersector<triangle_data, instancing, max_levels<2>> ix;\n"
"  ix.assume_geometry_type(geometry_type::triangle);\n"
"  ix.force_opacity(forced_opacity::opaque);\n"
"  constexpr int max_transparent_bounces = 8;\n"
"  float3 throughput = float3(1.0f);\n"
"  float current_tmin = t_min;\n"
"  float current_tmax = t_max;\n"
"  bool ray_was_bent = false;\n"
"  for (int b = 0; b < max_transparent_bounces; b++) {\n"
"    intersection_result<triangle_data, instancing, max_levels<2>> hit =\n"
"        ix.intersect(ray(origin, dir, current_tmin, current_tmax), scene);\n"
"    if (hit.type != intersection_type::triangle) {\n"
"      if (!ray_was_bent) {\n"
"        return float3(0.0f);\n"
"      }\n"
"      const float3 to_light = is_directional ? light_pos_or_dir\n"
"                                             : normalize(light_pos_or_dir - origin);\n"
"      const float align = saturate(dot(dir, to_light));\n"
"      /* Nuru: smooth cone falloff. `pow(align, 128)` was too binary -> every receiver pixel\n"
"       * either fired at full brightness or got nothing, producing speckled fireflies. The\n"
"       * smoothstep gives partial credit across a ~10 degree band so adjacent pixels'\n"
"       * contributions vary smoothly. Caustic outline is wider but visibly less noisy. */\n"
"      const float caustic_peak = photons_intensity *\n"
"                                 smoothstep(0.985f, 0.999f, align);\n"
"      return throughput * caustic_peak;\n"
"    }\n"
"    const uint user_id = hit.user_instance_id[0];\n"
"    const HardwareMaterialProxy proxy = material_proxies[user_id];\n"
"    const uint proxy_flags = uint(proxy.ior_closure_type.w);\n"
"    if ((proxy_flags & HWRT_PROXY_FLAG_THIN_GLASS) != 0u) {\n"
"      current_tmin = hit.distance + 1.0e-4f;\n"
"      if (current_tmin >= current_tmax) {\n"
"        return ray_was_bent ? throughput : float3(0.0f);\n"
"      }\n"
"      continue;\n"
"    }\n"
"    const uint closure_type = uint(proxy.ior_closure_type.z);\n"
"    const float alpha = proxy.packed_thickness.y;\n"
"    const float refraction_roughness = proxy.transmission_color_roughness.w;\n"
"    /* Nuru: caustic-bendable only for low-roughness REFRACTION (clear glass / refractive\n"
"     * BSDF). Reflection / metal is intentionally NOT handled here: the per-pixel bent-shadow\n"
"     * approach is geometrically sparse for mirrors and produces noisy fireflies that don't\n"
"     * converge cleanly. Metal caustics are best done as a separate forward photon-mapping\n"
"     * stage and that is left for a future change; for now metal is opaque in the caustic\n"
"     * trace (same as DIAMOND 19 behavior: metal hard-shadows correctly in `tint_only`). */\n"
"    constexpr float caustic_roughness_threshold = 0.15f;\n"
"    const bool is_caustic_refraction = (closure_type == HWRT_CLOSURE_REFRACTION) &&\n"
"                                       (refraction_roughness < caustic_roughness_threshold);\n"
"    const bool is_alpha_blend = ((proxy_flags & HWRT_PROXY_FLAG_ALPHA_BLEND_TRANSPARENT) != 0u);\n"
"    if (is_caustic_refraction) {\n"
"      throughput *= proxy.transmission_color_roughness.rgb;\n"
"    }\n"
"    else if (is_alpha_blend) {\n"
"      throughput *= float3(saturate(1.0f - alpha));\n"
"    }\n"
"    else {\n"
"      /* Metal (any reflection closure), rough glass, diffuse, or any other surface\n"
"       * terminates the bent caustic ray. The tinted shadow path is independent. */\n"
"      return float3(0.0f);\n"
"    }\n"
"    const float max_through = max(throughput.r, max(throughput.g, throughput.b));\n"
"    if (max_through < 1.0e-3f) {\n"
"      return float3(0.0f);\n"
"    }\n"
"    if (is_caustic_refraction) {\n"
"      /* Snell refraction via smooth normal. `eta = n_from / n_to`. */\n"
"      const TriangleNormalRange normal_range = triangle_normal_ranges[user_id];\n"
"      const uint primitive_offset = normal_range.offset + hit.primitive_id;\n"
"      const float3 flat_N = triangle_normals[primitive_offset].xyz;\n"
"      const uint smooth_offset = primitive_offset * 3u;\n"
"      const float3 bary = barycentric_expand(hit.triangle_barycentric_coord);\n"
"      float3 smooth_N_raw =\n"
"          triangle_smooth_normals[smooth_offset + 0u].xyz * bary.x +\n"
"          triangle_smooth_normals[smooth_offset + 1u].xyz * bary.y +\n"
"          triangle_smooth_normals[smooth_offset + 2u].xyz * bary.z;\n"
"      const float smooth_len_sq = dot(smooth_N_raw, smooth_N_raw);\n"
"      const float3 N_smooth = (smooth_len_sq > 1.0e-10f) ? (smooth_N_raw * rsqrt(smooth_len_sq))\n"
"                                                          : flat_N;\n"
"      const float cos_i_signed = dot(dir, N_smooth);\n"
"      const bool entering = (cos_i_signed < 0.0f);\n"
"      const float3 N_oriented = entering ? N_smooth : -N_smooth;\n"
"      const float cos_i = -dot(dir, N_oriented);\n"
"      const float ior = max(proxy.ior_closure_type.y, 1.0f);\n"
"      const float eta = entering ? (1.0f / ior) : ior;\n"
"      const float sin2_t = eta * eta * (1.0f - cos_i * cos_i);\n"
"      if (sin2_t >= 1.0f) {\n"
"        /* Total internal reflection - no caustic path through this hit. */\n"
"        return float3(0.0f);\n"
"      }\n"
"      const float cos_t = sqrt(1.0f - sin2_t);\n"
"      const float3 refracted = normalize(eta * dir + (eta * cos_i - cos_t) * N_oriented);\n"
"      const float3 hit_P = origin + dir * hit.distance;\n"
"      origin = hit_P;\n"
"      dir = refracted;\n"
"      current_tmin = 1.0e-3f;\n"
"      current_tmax = is_directional ? 100000.0f\n"
"                                    : max(length(light_pos_or_dir - origin), 1.0e-3f);\n"
"      ray_was_bent = true;\n"
"    }\n"
"    else {\n"
"      current_tmin = hit.distance + 1.0e-4f;\n"
"      if (current_tmin >= current_tmax) {\n"
"        if (!ray_was_bent) {\n"
"          return float3(0.0f);\n"
"        }\n"
"        const float3 to_light = is_directional ? light_pos_or_dir\n"
"                                               : normalize(light_pos_or_dir - origin);\n"
"        const float align = saturate(dot(dir, to_light));\n"
"        const float caustic_peak = photons_intensity *\n"
"                                   smoothstep(0.985f, 0.999f, align);\n"
"        return throughput * caustic_peak;\n"
"      }\n"
"    }\n"
"  }\n"
"  return float3(0.0f);\n"
"}\n"
"/* Nuru: single shadow trace with opacity/transmission blend and Color Transmission tint.\n"
" * transparency = 0 uses one any-hit opaque test. Otherwise one material-aware walk:\n"
" * glass/alpha attenuates throughput, opaque surfaces block. With any hit along the ray the\n"
" * opaque component is fully blocked, so mix(opaque, tinted, t) reduces to tinted * t. */\n"
"inline float3 hardware_shadow_visibility(\n"
"    instance_acceleration_structure scene,\n"
"    float3 origin,\n"
"    float3 dir,\n"
"    float t_min,\n"
"    float t_max,\n"
"    constant HardwareMaterialProxy *material_proxies,\n"
"    constant float4 *triangle_normals,\n"
"    constant float4 *triangle_smooth_normals,\n"
"    constant TriangleNormalRange *triangle_normal_ranges,\n"
"    float transparency,\n"
"    bool enable_caustics,\n"
"    bool is_directional,\n"
"    float3 light_pos_or_dir,\n"
"    float color_intensity,\n"
"    float photons_intensity)\n"
"{\n"
"  const float t = saturate(transparency);\n"
"  if (t <= 0.0f) {\n"
"    return opaque_shadow_visibility(scene, origin, dir, t_min, t_max, material_proxies);\n"
"  }\n"
"  intersector<triangle_data, instancing, max_levels<2>> ix;\n"
"  ix.assume_geometry_type(geometry_type::triangle);\n"
"  ix.force_opacity(forced_opacity::opaque);\n"
"  constexpr int max_transparent_bounces = 8;\n"
"  float3 throughput = float3(1.0f);\n"
"  float current_tmin = t_min;\n"
"  bool any_hit = false;\n"
"  const float vis_scale = min(t, 1.0f);\n"
"  for (int b = 0; b < max_transparent_bounces; b++) {\n"
"    intersection_result<triangle_data, instancing, max_levels<2>> hit =\n"
"        ix.intersect(ray(origin, dir, current_tmin, t_max), scene);\n"
"    if (hit.type != intersection_type::triangle) {\n"
"      if (!any_hit) {\n"
"        return float3(1.0f);\n"
"      }\n"
"      const float3 tinted = hardware_shadow_apply_color_transmission(throughput, color_intensity);\n"
"      float3 visibility = tinted * vis_scale;\n"
"      if (enable_caustics && vis_scale > 0.0f) {\n"
"        const float3 caustic = transparent_shadow_caustic_only(\n"
"            scene, origin, dir, t_min, t_max,\n"
"            material_proxies, triangle_normals, triangle_smooth_normals, triangle_normal_ranges,\n"
"            is_directional, light_pos_or_dir, photons_intensity);\n"
"        const float unoccluded = saturate(max(tinted.r, max(tinted.g, tinted.b)));\n"
"        visibility += caustic * saturate(1.0f - unoccluded) * vis_scale;\n"
"      }\n"
"      return visibility;\n"
"    }\n"
"    const uint user_id = hit.user_instance_id[0];\n"
"    const HardwareMaterialProxy proxy = material_proxies[user_id];\n"
"    const uint proxy_flags = uint(proxy.ior_closure_type.w);\n"
"    /* Thin Glass is fully shadow-exempt by contract. It must NOT mark the path as occluded:\n"
"     * a pane as the only surface between receiver and light would otherwise return\n"
"     * `vis_scale` instead of full visibility and cast a phantom transparent shadow. */\n"
"    if ((proxy_flags & HWRT_PROXY_FLAG_THIN_GLASS) != 0u) {\n"
"      current_tmin = hit.distance + 1.0e-4f;\n"
"      if (current_tmin >= t_max) {\n"
"        return any_hit ? hardware_shadow_apply_color_transmission(throughput, color_intensity) *\n"
"                             vis_scale :\n"
"                         float3(1.0f);\n"
"      }\n"
"      continue;\n"
"    }\n"
"    any_hit = true;\n"
"    const uint closure_type = uint(proxy.ior_closure_type.z);\n"
"    const float alpha = proxy.packed_thickness.y;\n"
"    const bool is_refraction = (closure_type == HWRT_CLOSURE_REFRACTION);\n"
"    const bool is_alpha_blend = ((proxy_flags & HWRT_PROXY_FLAG_ALPHA_BLEND_TRANSPARENT) != 0u);\n"
"    if (is_refraction) {\n"
"      throughput *= proxy.transmission_color_roughness.rgb;\n"
"    }\n"
"    else if (is_alpha_blend) {\n"
"      throughput *= float3(saturate(1.0f - alpha));\n"
"    }\n"
"    else {\n"
"      return float3(0.0f);\n"
"    }\n"
"    const float max_through = max(throughput.r, max(throughput.g, throughput.b));\n"
"    if (max_through < 1.0e-3f) {\n"
"      return float3(0.0f);\n"
"    }\n"
"    current_tmin = hit.distance + 1.0e-4f;\n"
"    if (current_tmin >= t_max) {\n"
"      const float3 tinted = hardware_shadow_apply_color_transmission(throughput, color_intensity);\n"
"      float3 visibility = tinted * vis_scale;\n"
"      if (enable_caustics && vis_scale > 0.0f) {\n"
"        const float3 caustic = transparent_shadow_caustic_only(\n"
"            scene, origin, dir, t_min, t_max,\n"
"            material_proxies, triangle_normals, triangle_smooth_normals, triangle_normal_ranges,\n"
"            is_directional, light_pos_or_dir, photons_intensity);\n"
"        const float unoccluded = saturate(max(tinted.r, max(tinted.g, tinted.b)));\n"
"        visibility += caustic * saturate(1.0f - unoccluded) * vis_scale;\n"
"      }\n"
"      return visibility;\n"
"    }\n"
"  }\n"
"  return float3(0.0f);\n"
"}\n"
         "kernel void eevee_hardware_trace_directional_shadow(\n"
         "    uint2 tid [[thread_position_in_grid]],\n"
         "    instance_acceleration_structure scene [[buffer(0)]],\n"
         "    constant HardwareShadowUniforms &uniforms [[buffer(1)]],\n"
         "    constant float4 *world_sunlight_direction [[buffer(2)]],\n"
         "    constant HardwareMaterialProxy *material_proxies [[buffer(3)]],\n"
         "    constant float4 *triangle_normals [[buffer(4)]],\n"
         "    constant float4 *triangle_smooth_normals [[buffer(5)]],\n"
         "    constant TriangleNormalRange *triangle_normal_ranges [[buffer(6)]],\n"
         "    depth2d<float, access::sample> depth_tx [[texture(0)]],\n"
         "    texture2d_array<uint, access::read> gbuf_header_tx [[texture(1)]],\n"
         "    texture2d_array<float, access::read> gbuf_normal_tx [[texture(2)]],\n"
         "    texture2d_array<float, access::write> shadow_visibility_img [[texture(3)]])\n"
         "{\n"
         "  if (tid.x >= uint(uniforms.resolution_layer.x) || tid.y >= uint(uniforms.resolution_layer.y)) {\n"
         "    return;\n"
         "  }\n"
         "  constexpr sampler depth_sampler(coord::normalized, address::clamp_to_edge, filter::nearest);\n"
         "  const float2 uv = (float2(tid) + 0.5f) / float2(uniforms.resolution_layer.xy);\n"
         "  const float depth = 1.0f - depth_tx.sample(depth_sampler, uv);\n"
         "  if (!depth_is_valid(depth)) {\n"
         "    shadow_visibility_img.write(float4(1.0f), tid, uint(uniforms.resolution_layer.z));\n"
         "    return;\n"
         "  }\n"
         "  const float3 P = point_screen_to_world(int2(tid), depth, uniforms);\n"
         "  float3 N = float3(0.0f);\n"
         "  if (!load_gbuffer_receiver_normal(int2(tid), gbuf_header_tx, gbuf_normal_tx, N)) {\n"
         "    N = estimate_world_normal(int2(tid), depth, depth_tx, uniforms);\n"
         "  }\n"
         "  if (dot(N, N) < 1.0e-10f) {\n"
         "    N = uniforms.light_direction_bias.xyz;\n"
         "  }\n"
         "  const float normal_bias = max(5.0e-3f, uniforms.light_direction_bias.w);\n"
         "  const float ray_tmin = max(5.0e-4f, normal_bias * 0.5f);\n"
         "  const int sample_count = (uniforms.shadow_params.x > 1.0e-6f) ? max(int(uniforms.shadow_params.y), 1) : 1;\n"
         "  const bool enable_caustics = (uniforms.world_sun_slot_pad.y != 0);\n"
         "  const float color_intensity = saturate(uniforms.shadow_params.z);\n"
         "  const float photons_intensity = max(uniforms.shadow_params.w, 0.0f);\n"
         "  const float transparent_shadows = as_type<float>(uint(uniforms.world_sun_slot_pad.w));\n"
         "  float3 visibility = float3(0.0f);\n"
         "  for (int sample_index = 0; sample_index < sample_count; sample_index++) {\n"
         "    const float3 sample_dir = sample_directional_shadow_direction(\n"
         "        tid, sample_index, uniforms, world_sunlight_direction);\n"
"    const float3 origin = P + N * normal_bias;\n"
         "    visibility += hardware_shadow_visibility(scene, origin, sample_dir, ray_tmin, 100000.0f, material_proxies, triangle_normals, triangle_smooth_normals, triangle_normal_ranges, transparent_shadows, enable_caustics, true, sample_dir, color_intensity, photons_intensity);\n"
         "  }\n"
         "  visibility /= float(sample_count);\n"
         "  shadow_visibility_img.write(float4(visibility, 1.0f), tid, uint(uniforms.resolution_layer.z));\n"
         "}\n"
"kernel void eevee_hardware_trace_directional_hit_shadow(\n"
"    uint3 threadgroup_id [[threadgroup_position_in_grid]],\n"
"    uint3 local_id [[thread_position_in_threadgroup]],\n"
"    instance_acceleration_structure scene [[buffer(0)]],\n"
"    constant HardwareShadowUniforms &uniforms [[buffer(1)]],\n"
"    constant float4 *world_sunlight_direction [[buffer(2)]],\n"
"    constant uint *tiles_coord_buf [[buffer(3)]],\n"
"    constant float4 *triangle_normals [[buffer(4)]],\n"
"    constant TriangleNormalRange *triangle_normal_ranges [[buffer(5)]],\n"
"    constant HardwareMaterialProxy *material_proxies [[buffer(6)]],\n"
"    constant float4 *triangle_smooth_normals [[buffer(7)]],\n"
"    texture2d<float, access::read> hit_normal_img [[texture(0)]],\n"
"    texture2d<float, access::read> hit_world_position_img [[texture(1)]],\n"
"    texture2d<uint, access::read> hit_identity_img [[texture(2)]],\n"
"    texture2d_array<float, access::write> shadow_visibility_img [[texture(3)]])\n"
"{\n"
"  const uint2 tile_coord = unpackUvec2x16(tiles_coord_buf[threadgroup_id.x]);\n"
"  const uint2 tid = uint2(local_id.xy) + tile_coord * 8u;\n"
"  if (tid.x >= hit_world_position_img.get_width() || tid.y >= hit_world_position_img.get_height()) {\n"
"    return;\n"
"  }\n"
"  const float3 P = hit_world_position_img.read(tid).xyz;\n"
"  float3 N = hit_normal_img.read(tid).xyz;\n"
"  if (!all(isfinite(P)) || !all(isfinite(N)) || dot(N, N) < 1.0e-10f) {\n"
"    return;\n"
"  }\n"
"  N = normalize(N);\n"
"  const float3 shadow_N = hit_shadow_receiver_normal(\n"
"      tid, N, hit_identity_img, triangle_normals, triangle_normal_ranges);\n"
"  const float normal_bias = max(5.0e-3f, uniforms.light_direction_bias.w);\n"
"  const float ray_tmin = max(5.0e-4f, normal_bias * 0.5f);\n"
"  const int sample_count = (uniforms.shadow_params.x > 1.0e-6f) ? max(int(uniforms.shadow_params.y), 1) : 1;\n"
"  const bool enable_caustics = (uniforms.world_sun_slot_pad.y != 0);\n"
"  const float color_intensity = saturate(uniforms.shadow_params.z);\n"
"  const float photons_intensity = max(uniforms.shadow_params.w, 0.0f);\n"
"  const float transparent_shadows = as_type<float>(uint(uniforms.world_sun_slot_pad.w));\n"
"  float3 visibility = float3(0.0f);\n"
"  for (int sample_index = 0; sample_index < sample_count; sample_index++) {\n"
"    const float3 sample_dir = sample_directional_shadow_direction(\n"
"        tid, sample_index, uniforms, world_sunlight_direction);\n"
"    const float3 origin = P + shadow_N * normal_bias;\n"
"    visibility += hardware_shadow_visibility(scene, origin, sample_dir, ray_tmin, 100000.0f, material_proxies, triangle_normals, triangle_smooth_normals, triangle_normal_ranges, transparent_shadows, enable_caustics, true, sample_dir, color_intensity, photons_intensity);\n"
"  }\n"
"  visibility /= float(sample_count);\n"
"  shadow_visibility_img.write(float4(visibility, 1.0f), tid, uint(uniforms.resolution_layer.z));\n"
"}\n"
         "kernel void eevee_hardware_trace_environment_visibility(\n"
         "    uint2 tid [[thread_position_in_grid]],\n"
         "    instance_acceleration_structure scene [[buffer(0)]],\n"
         "    constant HardwareEnvironmentVisibilityUniforms &uniforms [[buffer(1)]],\n"
         "    depth2d<float, access::sample> depth_tx [[texture(0)]],\n"
         "    texture2d_array<uint, access::read> gbuf_header_tx [[texture(1)]],\n"
         "    texture2d_array<float, access::read> gbuf_normal_tx [[texture(2)]],\n"
         "    texture2d<float, access::write> environment_visibility_img [[texture(3)]])\n"
         "{\n"
         "  if (tid.x >= uint(uniforms.resolution_samples.x) || tid.y >= uint(uniforms.resolution_samples.y)) {\n"
         "    return;\n"
         "  }\n"
         "  constexpr sampler depth_sampler(coord::normalized, address::clamp_to_edge, filter::nearest);\n"
         "  const float2 uv = (float2(tid) + 0.5f) / float2(uniforms.resolution_samples.xy);\n"
         "  const float depth = 1.0f - depth_tx.sample(depth_sampler, uv);\n"
         "  if (!depth_is_valid(depth)) {\n"
         "    environment_visibility_img.write(float4(0.0f, 0.0f, 0.0f, 1.0f), tid);\n"
         "    return;\n"
         "  }\n"
         "  const float3 P = point_screen_to_world(int2(tid), depth, uniforms);\n"
         "  float3 N = float3(0.0f);\n"
         "  if (!load_gbuffer_receiver_normal(int2(tid), gbuf_header_tx, gbuf_normal_tx, N)) {\n"
         "    N = estimate_world_normal(int2(tid), depth, depth_tx, uniforms);\n"
         "  }\n"
         "  if (dot(N, N) < 1.0e-10f) {\n"
         "    environment_visibility_img.write(float4(0.0f, 0.0f, 0.0f, 1.0f), tid);\n"
         "    return;\n"
         "  }\n"
         "  N = normalize(N);\n"
         "  const float normal_bias = max(4.0e-3f, uniforms.normal_bias_pad.x);\n"
         "  const float ray_tmin = max(5.0e-4f, normal_bias * 0.25f);\n"
         "  const int sample_count = max(uniforms.resolution_samples.z, 1);\n"
         "  intersector<triangle_data, instancing, max_levels<2>> i;\n"
         "  i.assume_geometry_type(geometry_type::triangle);\n"
         "  i.force_opacity(forced_opacity::opaque);\n"
         "  float visibility = 0.0f;\n"
         "  const float3 origin = P + N * normal_bias;\n"
         "  float3 average_direction = float3(0.0f);\n"
         "  for (int sample_index = 0; sample_index < sample_count; sample_index++) {\n"
         "    const float3 sample_dir = sample_environment_visibility_direction(tid, sample_index, N, uniforms);\n"
         "    intersection_result<triangle_data, instancing, max_levels<2>> intersection = i.intersect(ray(origin, sample_dir, ray_tmin, 100000.0f), scene);\n"
         "    const float sample_visibility = (intersection.type == intersection_type::triangle) ? 0.0f : 1.0f;\n"
         "    visibility += sample_visibility;\n"
         "    average_direction += sample_dir * sample_visibility;\n"
         "  }\n"
         "  visibility /= float(sample_count);\n"
         "  average_direction /= float(sample_count);\n"
         "  environment_visibility_img.write(float4(average_direction, visibility), tid);\n"
         "}\n"
"kernel void eevee_hardware_trace_hit_environment_visibility(\n"
"    uint3 threadgroup_id [[threadgroup_position_in_grid]],\n"
"    uint3 local_id [[thread_position_in_threadgroup]],\n"
"    instance_acceleration_structure scene [[buffer(0)]],\n"
"    constant HardwareEnvironmentVisibilityUniforms &uniforms [[buffer(1)]],\n"
"    constant uint *tiles_coord_buf [[buffer(2)]],\n"
"    texture2d<float, access::read> hit_normal_img [[texture(0)]],\n"
"    texture2d<float, access::read> hit_world_position_img [[texture(1)]],\n"
"    texture2d<float, access::write> environment_visibility_img [[texture(2)]])\n"
"{\n"
"  const uint2 tile_coord = unpackUvec2x16(tiles_coord_buf[threadgroup_id.x]);\n"
"  const uint2 tid = uint2(local_id.xy) + tile_coord * 8u;\n"
"  if (tid.x >= hit_world_position_img.get_width() || tid.y >= hit_world_position_img.get_height()) {\n"
"    return;\n"
"  }\n"
"  const float3 P = hit_world_position_img.read(tid).xyz;\n"
"  float3 N = hit_normal_img.read(tid).xyz;\n"
"  if (!all(isfinite(P)) || !all(isfinite(N)) || dot(N, N) < 1.0e-10f) {\n"
"    return;\n"
"  }\n"
"  N = normalize(N);\n"
"  const float normal_bias = max(4.0e-3f, uniforms.normal_bias_pad.x);\n"
"  const float ray_tmin = max(5.0e-4f, normal_bias * 0.25f);\n"
"  const int sample_count = max(uniforms.resolution_samples.z, 1);\n"
"  intersector<triangle_data, instancing, max_levels<2>> i;\n"
"  i.assume_geometry_type(geometry_type::triangle);\n"
"  i.force_opacity(forced_opacity::opaque);\n"
"  float visibility = 0.0f;\n"
"  const float3 origin = P + N * normal_bias;\n"
"  float3 average_direction = float3(0.0f);\n"
"  for (int sample_index = 0; sample_index < sample_count; sample_index++) {\n"
"    const float3 sample_dir = sample_environment_visibility_direction(tid, sample_index, N, uniforms);\n"
"    intersection_result<triangle_data, instancing, max_levels<2>> intersection = i.intersect(ray(origin, sample_dir, ray_tmin, 100000.0f), scene);\n"
"    const float sample_visibility = (intersection.type == intersection_type::triangle) ? 0.0f : 1.0f;\n"
"    visibility += sample_visibility;\n"
"    average_direction += sample_dir * sample_visibility;\n"
"  }\n"
"  visibility /= float(sample_count);\n"
"  average_direction /= float(sample_count);\n"
"  environment_visibility_img.write(float4(average_direction, visibility), tid);\n"
"}\n"
"kernel void eevee_hardware_trace_reflected_receiver_gi(\n"
"    uint3 threadgroup_id [[threadgroup_position_in_grid]],\n"
"    uint3 local_id [[thread_position_in_threadgroup]],\n"
"    instance_acceleration_structure scene [[buffer(0)]],\n"
"    constant HardwareReflectedReceiverGIUniforms &uniforms [[buffer(1)]],\n"
"    constant float4 *emissive_radiance [[buffer(2)]],\n"
"    constant float4 *diffuse_albedo [[buffer(3)]],\n"
"    constant float4 *triangle_normals [[buffer(4)]],\n"
"    constant TriangleNormalRange *triangle_normal_ranges [[buffer(5)]],\n"
"    constant FastGILightRecord *receiver_gi_lights [[buffer(6)]],\n"
"    constant uint *tiles_coord_buf [[buffer(7)]],\n"
"    constant HardwareMaterialProxy *material_proxy [[buffer(8)]],\n"
"    constant float *nis_weights [[buffer(9)]],\n"
"    device uint *nis_feedback [[buffer(10)]],\n"
"    texture2d<float, access::write> receiver_gi_img [[texture(0)]],\n"
"    texture2d_array<float, access::sample> world_probe_tx [[texture(1)]],\n"
"    texture2d<float, access::read> ray_time_img [[texture(2)]],\n"
"    texture2d<float, access::read> hit_albedo_img [[texture(3)]],\n"
"    texture2d<float, access::read> hit_normal_img [[texture(4)]],\n"
"    texture2d<float, access::read> hit_world_position_img [[texture(5)]],\n"
"    texture2d<float, access::read> hit_material_img [[texture(6)]])\n"
"{\n"
"  const uint2 tile_coord = unpackUvec2x16(tiles_coord_buf[threadgroup_id.x]);\n"
"  const uint2 tid = uint2(local_id.xy) + tile_coord * 8u;\n"
"  if (tid.x >= uint(uniforms.resolution_samples.x) || tid.y >= uint(uniforms.resolution_samples.y) ||\n"
"      tid.x >= receiver_gi_img.get_width() || tid.y >= receiver_gi_img.get_height()) {\n"
"    return;\n"
"  }\n"
"  const uint resolution_divisor = uint(max(uniforms.resolution_samples.z, 1));\n"
"  const uint2 anchor_tid = min((tid / resolution_divisor) * resolution_divisor + resolution_divisor / 2u,\n"
"                               uint2(uint(uniforms.resolution_samples.x - 1),\n"
"                                     uint(uniforms.resolution_samples.y - 1)));\n"
"  if (any(tid != anchor_tid)) {\n"
"    return;\n"
"  }\n"
"  /* Early-outs must NOT write: the trace dispatch runs once per closure bin, and a later\n"
"   * bin's inactive rays at this texel would otherwise clobber the receiver GI computed for\n"
"   * an earlier bin (classroom regression: glass refraction bins zeroed the mirror's data).\n"
"   * The host clears the texture once per submit, so unwritten texels stay invalid. */\n"
"  if (!(ray_time_img.read(tid).x > 0.0f)) {\n"
"    return;\n"
"  }\n"
"  const float3 P = hit_world_position_img.read(tid).xyz;\n"
"  float3 N = hit_normal_img.read(tid).xyz;\n"
"  const float3 receiver_albedo = max(hit_albedo_img.read(tid).xyz, float3(0.0f));\n"
"  const float4 hit_material = hit_material_img.read(tid);\n"
"  const uint receiver_closure = uint(max(hit_material.z, 0.0f) + 0.5f);\n"
"  const bool receiver_is_reflection = (receiver_closure == HWRT_CLOSURE_REFLECTION);\n"
"  if (!all(isfinite(P)) || !all(isfinite(N)) || dot(N, N) < 1.0e-10f ||\n"
"      max(receiver_albedo.x, max(receiver_albedo.y, receiver_albedo.z)) <= 1.0e-5f) {\n"
"    return;\n"
"  }\n"
"  N = normalize(N);\n"
"  const int sample_count = max(uniforms.resolution_samples.w, 1);\n"
"  const float normal_bias = max(uniforms.normal_bias_pad.x, 1.0e-4f);\n"
"  const float ray_tmin = max(5.0e-4f, normal_bias * 0.5f);\n"
"  const float ray_tmax = 1000.0f;\n"
"  intersector<triangle_data, instancing, max_levels<2>> i;\n"
"  i.assume_geometry_type(geometry_type::triangle);\n"
"  i.force_opacity(forced_opacity::opaque);\n"
"  float3 accum = float3(0.0f);\n"
"  const float3 incoming_ray_dir = reflected_receiver_gi_direction_unpack(\n"
"      float2(hit_material.w, hit_normal_img.read(tid).w));\n"
"  const float3 reflection_dir = normalize(reflect(incoming_ray_dir, N));\n"
"  for (int sample_index = 0; sample_index < sample_count; sample_index++) {\n"
"    const float3 sample_dir = receiver_is_reflection ?\n"
"                                  reflected_receiver_gi_cone_direction(\n"
"                                      tid, sample_index, reflection_dir, hit_material.x, uniforms) :\n"
"                                  sample_reflected_receiver_gi_direction(tid, sample_index, N, uniforms);\n"
"    const float3 origin = P + N * normal_bias;\n"
"    intersection_result<triangle_data, instancing, max_levels<2>> intersection = i.intersect(\n"
"        ray(origin, sample_dir, ray_tmin, ray_tmax), scene);\n"
"    float3 incoming = float3(0.0f);\n"
"    if (intersection.type == intersection_type::triangle) {\n"
"      const uint user_id = intersection.user_instance_id[0];\n"
"      /* Do NOT pick up emissive surface radiance here. At 2-4 dome samples per receiver,\n"
"       * bright emissive fixtures turn the lamp-visibility solid angle into giant halo rings\n"
"       * on mirror-visible walls/ceilings (classroom F12 regression). Direct lighting of\n"
"       * receivers, including from light objects in the fixtures, is owned by the light-tree\n"
"       * NEE below, which is smooth and shadowed. */\n"
"      const float3 hit_albedo = max(diffuse_albedo[user_id].xyz, float3(0.0f));\n"
"      if (max(hit_albedo.x, max(hit_albedo.y, hit_albedo.z)) > 1.0e-4f) {\n"
"        const float3 hit_N = fast_gi_hit_normal(\n"
"            user_id, intersection.primitive_id, sample_dir, triangle_normals, triangle_normal_ranges);\n"
"        const float3 hit_P = origin + sample_dir * intersection.distance;\n"
"        /* No desaturation (user directive): keep the bounce surface's true color so sun-warmed\n"
"         * floors tint the receiver GI like real bounce light. Firefly control stays with the\n"
"         * luma clamps below, which preserve hue. */\n"
"        const float3 soft_hit_albedo = hit_albedo;\n"
"        /* Pre-shadow-ray era damping removed: tree NEE with occlusion rays is physically\n"
"         * scaled, so the bounce lands at its real energy (DIAMOND 25 calibration). */\n"
"        int feedback_cluster = -1;\n"
"        float feedback_weight = 0.0f;\n"
"        incoming += soft_hit_albedo * sample_reflected_receiver_gi_direct_light(tid,\n"
"                                                                          sample_index,\n"
"                                                                          sample_count,\n"
"                                                                          hit_P,\n"
"                                                                          hit_N,\n"
"                                                                          scene,\n"
"                                                                          receiver_gi_lights,\n"
"                                                                          material_proxy,\n"
"                                                                          nis_weights,\n"
"                                                                          uniforms,\n"
"                                                                          feedback_cluster,\n"
"                                                                          feedback_weight) * (receiver_is_reflection ? 0.6f : 1.0f);\n"
"        /* Nuru NIS G3: off-screen training feedback - mirror-interior positions never appear\n"
"         * in the camera-tile trainer, so a subsampled ring of receiver NEE outcomes teaches\n"
"         * the network those regions too. Entry 0 of the buffer is the atomic counter. */\n"
"        if (feedback_cluster >= 0 && sample_index == 0 && ((tid.x & 7u) == 0u) &&\n"
"            ((tid.y & 7u) == 0u) && (uniforms.environment_pad.y != 0))\n"
"        {\n"
"          const uint slot = atomic_fetch_add_explicit(\n"
"              (device atomic_uint *)&nis_feedback[0], 1u, memory_order_relaxed);\n"
"          if (slot < 4096u) {\n"
"            device float *entry = (device float *)&nis_feedback[8u + slot * 8u];\n"
"            entry[0] = hit_P.x;\n"
"            entry[1] = hit_P.y;\n"
"            entry[2] = hit_P.z;\n"
"            entry[3] = min(feedback_weight, 32.0f);\n"
"            entry[4] = float(feedback_cluster);\n"
"          }\n"
"          else {\n"
"            atomic_fetch_sub_explicit((device atomic_uint *)&nis_feedback[0], 1u,\n"
"                                      memory_order_relaxed);\n"
"          }\n"
"        }\n"
"        incoming += soft_hit_albedo * sample_reflected_receiver_gi_world_radiance(\n"
"                                         world_probe_tx, hit_N, uniforms) * (receiver_is_reflection ? 0.15f : 0.25f);\n"
"      }\n"
"    }\n"
"    else {\n"
"      incoming = sample_reflected_receiver_gi_world_radiance(world_probe_tx, sample_dir, uniforms) *\n"
"                 (receiver_is_reflection ? 0.5f : 1.0f);\n"
"    }\n"
"    accum += reflected_receiver_gi_luma_clamp(incoming, receiver_is_reflection ? 1.5f : 2.5f);\n"
"  }\n"
"  receiver_gi_img.write(float4(reflected_receiver_gi_luma_clamp(accum / float(sample_count),\n"
"                                                                receiver_is_reflection ? 0.9f : 1.8f),\n"
"                               1.0f),\n"
"                        tid);\n"
"}\n"
"inline bool secondary_photon_gi_initial_ray(uint2 tid,\n"
"                                            int sample_index,\n"
"                                            float3 incoming_ray_dir,\n"
"                                            float3 N,\n"
"                                            float4 hit_material,\n"
"                                            constant HardwareReflectedReceiverGIUniforms &u,\n"
"                                            thread float3 &ray_dir)\n"
"{\n"
"  const uint closure = uint(max(hit_material.z, 0.0f) + 0.5f);\n"
"  if (closure == HWRT_CLOSURE_REFLECTION) {\n"
"    const float3 reflection_dir = normalize(reflect(incoming_ray_dir, N));\n"
"    ray_dir = reflected_receiver_gi_cone_direction(tid, sample_index, reflection_dir, hit_material.x, u);\n"
"    return dot(ray_dir, ray_dir) > 1.0e-10f;\n"
"  }\n"
"  if (closure == HWRT_CLOSURE_REFRACTION) {\n"
"    const float eta = 1.0f / max(hit_material.y, 1.0e-3f);\n"
"    float3 refracted_dir = refract(incoming_ray_dir, N, eta);\n"
"    if (!(dot(refracted_dir, refracted_dir) > 1.0e-10f)) {\n"
"      refracted_dir = reflect(incoming_ray_dir, N);\n"
"    }\n"
"    ray_dir = reflected_receiver_gi_cone_direction(\n"
"        tid, sample_index, normalize(refracted_dir), hit_material.x, u);\n"
"    return dot(ray_dir, ray_dir) > 1.0e-10f;\n"
"  }\n"
"  return false;\n"
"}\n"
"inline bool secondary_photon_gi_continue_ray(uint2 tid,\n"
"                                             int sample_index,\n"
"                                             int redirect_index,\n"
"                                             float3 incoming_ray_dir,\n"
"                                             float3 hit_N,\n"
"                                             HardwareMaterialProxy proxy,\n"
"                                             constant HardwareReflectedReceiverGIUniforms &u,\n"
"                                             thread float3 &throughput,\n"
"                                             thread float3 &ray_dir)\n"
"{\n"
"  const uint closure = uint(proxy.ior_closure_type.z + 0.5f);\n"
"  if (closure == HWRT_CLOSURE_REFLECTION) {\n"
"    const float3 tint = clamp(proxy.reflection_color_roughness.xyz, float3(0.0f), float3(4.0f));\n"
"    if (!(max(tint.x, max(tint.y, tint.z)) > 1.0e-4f)) {\n"
"      return false;\n"
"    }\n"
"    throughput *= tint;\n"
"    ray_dir = reflected_receiver_gi_cone_direction(\n"
"        tid, sample_index + redirect_index * 17, normalize(reflect(incoming_ray_dir, hit_N)), proxy.reflection_color_roughness.w, u);\n"
"    return dot(ray_dir, ray_dir) > 1.0e-10f && dot(throughput, throughput) > 1.0e-10f;\n"
"  }\n"
"  if (closure == HWRT_CLOSURE_REFRACTION) {\n"
"    const float3 tint = clamp(proxy.transmission_color_roughness.xyz, float3(0.0f), float3(4.0f));\n"
"    if (!(max(tint.x, max(tint.y, tint.z)) > 1.0e-4f)) {\n"
"      return false;\n"
"    }\n"
"    const float eta = dot(incoming_ray_dir, hit_N) < 0.0f ? (1.0f / max(proxy.ior_closure_type.y, 1.0e-3f)) :\n"
"                                                          max(proxy.ior_closure_type.y, 1.0e-3f);\n"
"    float3 refracted_dir = refract(incoming_ray_dir, hit_N, eta);\n"
"    if (!(dot(refracted_dir, refracted_dir) > 1.0e-10f)) {\n"
"      refracted_dir = reflect(incoming_ray_dir, hit_N);\n"
"    }\n"
"    throughput *= tint;\n"
"    ray_dir = reflected_receiver_gi_cone_direction(\n"
"        tid, sample_index + redirect_index * 19, normalize(refracted_dir), proxy.transmission_color_roughness.w, u);\n"
"    return dot(ray_dir, ray_dir) > 1.0e-10f && dot(throughput, throughput) > 1.0e-10f;\n"
"  }\n"
"  return false;\n"
"}\n"
"kernel void eevee_hardware_trace_secondary_photon_gi(\n"
"    uint3 threadgroup_id [[threadgroup_position_in_grid]],\n"
"    uint3 local_id [[thread_position_in_threadgroup]],\n"
"    instance_acceleration_structure scene [[buffer(0)]],\n"
"    constant HardwareReflectedReceiverGIUniforms &uniforms [[buffer(1)]],\n"
"    constant float4 *emissive_radiance [[buffer(2)]],\n"
"    constant float4 *diffuse_albedo [[buffer(3)]],\n"
"    constant float4 *triangle_normals [[buffer(4)]],\n"
"    constant TriangleNormalRange *triangle_normal_ranges [[buffer(5)]],\n"
"    constant FastGILightRecord *secondary_photon_lights [[buffer(6)]],\n"
"    constant uint *tiles_coord_buf [[buffer(7)]],\n"
"    constant HardwareMaterialProxy *material_proxy [[buffer(8)]],\n"
"    constant float *nis_weights [[buffer(9)]],\n"
"    texture2d<float, access::write> secondary_photon_gi_img [[texture(0)]],\n"
"    texture2d_array<float, access::sample> world_probe_tx [[texture(1)]],\n"
"    texture2d<float, access::read> ray_time_img [[texture(2)]],\n"
"    texture2d<float, access::read> hit_albedo_img [[texture(3)]],\n"
"    texture2d<float, access::read> hit_normal_img [[texture(4)]],\n"
"    texture2d<float, access::read> hit_world_position_img [[texture(5)]],\n"
"    texture2d<float, access::read> hit_material_img [[texture(6)]])\n"
"{\n"
"  const uint2 tile_coord = unpackUvec2x16(tiles_coord_buf[threadgroup_id.x]);\n"
"  const uint2 tid = uint2(local_id.xy) + tile_coord * 8u;\n"
"  if (tid.x >= uint(uniforms.resolution_samples.x) || tid.y >= uint(uniforms.resolution_samples.y) ||\n"
"      tid.x >= secondary_photon_gi_img.get_width() || tid.y >= secondary_photon_gi_img.get_height()) {\n"
"    return;\n"
"  }\n"
"  const uint resolution_divisor = uint(max(uniforms.resolution_samples.z, 1));\n"
"  const uint2 anchor_tid = min((tid / resolution_divisor) * resolution_divisor + resolution_divisor / 2u,\n"
"                               uint2(uint(uniforms.resolution_samples.x - 1),\n"
"                                     uint(uniforms.resolution_samples.y - 1)));\n"
"  if (any(tid != anchor_tid)) {\n"
"    return;\n"
"  }\n"
"  if (!(ray_time_img.read(tid).x > 0.0f)) {\n"
"    secondary_photon_gi_img.write(float4(0.0f), tid);\n"
"    return;\n"
"  }\n"
"  const float3 P = hit_world_position_img.read(tid).xyz;\n"
"  float3 N = hit_normal_img.read(tid).xyz;\n"
"  const float4 hit_material = hit_material_img.read(tid);\n"
"  const float3 receiver_tint = max(hit_albedo_img.read(tid).xyz, float3(0.0f));\n"
"  const uint receiver_closure = uint(max(hit_material.z, 0.0f) + 0.5f);\n"
"  if (!all(isfinite(P)) || !all(isfinite(N)) || dot(N, N) < 1.0e-10f ||\n"
"      ((receiver_closure != HWRT_CLOSURE_REFLECTION) && (receiver_closure != HWRT_CLOSURE_REFRACTION)))\n"
"  {\n"
"    secondary_photon_gi_img.write(float4(0.0f), tid);\n"
"    return;\n"
"  }\n"
"  N = normalize(N);\n"
"  const float receiver_energy = max(receiver_tint.x, max(receiver_tint.y, receiver_tint.z));\n"
"  const float3 receiver_throughput = (receiver_energy > 1.0e-5f) ? receiver_tint : float3(1.0f);\n"
"  const int sample_count = max(uniforms.resolution_samples.w, 1);\n"
"  const float normal_bias = max(uniforms.normal_bias_pad.x, 1.0e-4f);\n"
"  const float ray_tmin = max(5.0e-4f, normal_bias * 0.5f);\n"
"  const float ray_tmax = 1000.0f;\n"
"  const float3 incoming_ray_dir = reflected_receiver_gi_direction_unpack(\n"
"      float2(hit_material.w, hit_normal_img.read(tid).w));\n"
"  intersector<triangle_data, instancing, max_levels<2>> i;\n"
"  i.assume_geometry_type(geometry_type::triangle);\n"
"  i.force_opacity(forced_opacity::opaque);\n"
"  float3 accum = float3(0.0f);\n"
"  for (int sample_index = 0; sample_index < sample_count; sample_index++) {\n"
"    float3 trace_dir = float3(0.0f);\n"
"    if (!secondary_photon_gi_initial_ray(tid, sample_index, incoming_ray_dir, N, hit_material, uniforms, trace_dir)) {\n"
"      continue;\n"
"    }\n"
"    float3 trace_origin = P + ((receiver_closure == HWRT_CLOSURE_REFRACTION) ? trace_dir : N) * normal_bias;\n"
"    float3 throughput = receiver_throughput;\n"
"    float3 sample_radiance = float3(0.0f);\n"
"    for (int redirect = 0; redirect < 3; redirect++) {\n"
"      intersection_result<triangle_data, instancing, max_levels<2>> intersection = i.intersect(\n"
"          ray(trace_origin, trace_dir, ray_tmin, ray_tmax), scene);\n"
"      if (intersection.type != intersection_type::triangle) {\n"
"        sample_radiance += throughput * sample_reflected_receiver_gi_world_radiance(\n"
"                                         world_probe_tx, trace_dir, uniforms) * 0.18f;\n"
"        break;\n"
"      }\n"
"      const uint user_id = intersection.user_instance_id[0];\n"
"      const float3 hit_P = trace_origin + trace_dir * intersection.distance;\n"
"      const float3 hit_emissive = max(emissive_radiance[user_id].xyz, float3(0.0f));\n"
"      if (max(hit_emissive.x, max(hit_emissive.y, hit_emissive.z)) > 1.0e-5f) {\n"
"        sample_radiance += throughput * hit_emissive;\n"
"      }\n"
"      const HardwareMaterialProxy proxy = material_proxy[user_id];\n"
"      const uint proxy_closure = uint(proxy.ior_closure_type.z + 0.5f);\n"
"      const float3 hit_N = fast_gi_hit_normal(\n"
"          user_id, intersection.primitive_id, trace_dir, triangle_normals, triangle_normal_ranges);\n"
"      if ((redirect < 2) && dot(hit_N, hit_N) > 1.0e-10f &&\n"
"          ((proxy_closure == HWRT_CLOSURE_REFLECTION) || (proxy_closure == HWRT_CLOSURE_REFRACTION)))\n"
"      {\n"
"        float3 next_dir = float3(0.0f);\n"
"        if (secondary_photon_gi_continue_ray(\n"
"                tid, sample_index, redirect + 1, trace_dir, normalize(hit_N), proxy, uniforms, throughput, next_dir))\n"
"        {\n"
"          const float3 offset_N = (proxy_closure == HWRT_CLOSURE_REFRACTION) ? next_dir : normalize(hit_N);\n"
"          trace_origin = hit_P + offset_N * normal_bias;\n"
"          trace_dir = next_dir;\n"
"          continue;\n"
"        }\n"
"      }\n"
"      const float3 hit_albedo = max(diffuse_albedo[user_id].xyz, float3(0.0f));\n"
"      if (max(hit_albedo.x, max(hit_albedo.y, hit_albedo.z)) > 1.0e-4f && dot(hit_N, hit_N) > 1.0e-10f) {\n"
"        const float hit_luma = dot(hit_albedo, float3(0.2126f, 0.7152f, 0.0722f));\n"
"        const float3 soft_hit_albedo = mix(hit_albedo, float3(hit_luma), 0.25f);\n"
"        int photon_feedback_cluster = -1;\n"
"        float photon_feedback_weight = 0.0f;\n"
"        sample_radiance += throughput * soft_hit_albedo * sample_reflected_receiver_gi_direct_light(\n"
"                               tid,\n"
"                               sample_index,\n"
"                               sample_count,\n"
"                               hit_P,\n"
"                               normalize(hit_N),\n"
"                               scene,\n"
"                               secondary_photon_lights,\n"
"                               material_proxy,\n"
"                               nis_weights,\n"
"                               uniforms,\n"
"                               photon_feedback_cluster,\n"
"                               photon_feedback_weight) * 0.12f;\n"
"        sample_radiance += throughput * soft_hit_albedo * sample_reflected_receiver_gi_world_radiance(\n"
"                               world_probe_tx, normalize(hit_N), uniforms) * 0.06f;\n"
"      }\n"
"      break;\n"
"    }\n"
"    accum += reflected_receiver_gi_luma_clamp(sample_radiance, 1.25f);\n"
"  }\n"
"  secondary_photon_gi_img.write(float4(reflected_receiver_gi_luma_clamp(accum / float(sample_count), 0.85f),\n"
"                                      1.0f),\n"
"                               tid);\n"
"}\n"
"kernel void eevee_hardware_trace_receiver_caustics(\n"
"    uint2 tid [[thread_position_in_grid]],\n"
"    instance_acceleration_structure scene [[buffer(0)]],\n"
"    constant HardwareReceiverCausticUniforms &uniforms [[buffer(1)]],\n"
"    constant HardwareMaterialProxy *material_proxies [[buffer(2)]],\n"
"    constant float4 *triangle_normals [[buffer(3)]],\n"
"    constant TriangleNormalRange *triangle_normal_ranges [[buffer(4)]],\n"
"    constant FastGILightRecord *receiver_lights [[buffer(5)]],\n"
"    depth2d<float, access::sample> depth_tx [[texture(0)]],\n"
"    texture2d_array<uint, access::read> gbuf_header_tx [[texture(1)]],\n"
"    texture2d_array<float, access::read> gbuf_normal_tx [[texture(2)]],\n"
"    texture2d<float, access::write> caustics_img [[texture(3)]],\n"
"    texture2d_array<float, access::sample> world_probe_tx [[texture(4)]])\n"
"{\n"
"  if (tid.x >= uint(uniforms.resolution_samples.x) ||\n"
"      tid.y >= uint(uniforms.resolution_samples.y))\n"
"  {\n"
"    return;\n"
"  }\n"
"  caustics_img.write(float4(0.0f), tid);\n"
"  const int light_count = max(uniforms.light_count_pad.x, 0);\n"
"  const int sample_count = max(uniforms.light_count_pad.y, 1);\n"
"  const bool use_world = uniforms.world_probe_atlas_coord.w >= 0.0f;\n"
"  const float photons_intensity = max(uniforms.normal_bias_photons.y, 0.0f);\n"
"  if ((light_count <= 0 && !use_world) || !(photons_intensity > 1.0e-6f)) {\n"
"    return;\n"
"  }\n"
"  constexpr sampler depth_sampler(coord::normalized, address::clamp_to_edge, filter::nearest);\n"
"  const float2 uv = (float2(tid) + 0.5f) / float2(uniforms.resolution_samples.xy);\n"
"  const float depth = 1.0f - depth_tx.sample(depth_sampler, uv);\n"
"  if (!depth_is_valid(depth)) {\n"
"    return;\n"
"  }\n"
"  const uint header = gbuf_header_tx.read(tid, 0).x;\n"
"  const uint mode0 = (header >> (GBUFFER_HEADER_BITS_PER_BIN * 0u)) & 15u;\n"
"  const bool diffuse_receiver = (mode0 == uint(GBUF_DIFFUSE)) ||\n"
"                                (mode0 == uint(GBUF_SUBSURFACE));\n"
"  if (!diffuse_receiver) {\n"
"    return;\n"
"  }\n"
"  float3 N = float3(0.0f);\n"
"  if (!load_gbuffer_receiver_normal(int2(tid), gbuf_header_tx, gbuf_normal_tx, N)) {\n"
"    return;\n"
"  }\n"
"  const float3 P = point_screen_to_world(int2(tid), depth, uniforms);\n"
"  const float normal_bias = max(uniforms.normal_bias_photons.x, 1.0e-4f);\n"
"  const float ray_tmin = max(5.0e-4f, normal_bias * 0.5f);\n"
"  intersector<triangle_data, instancing, max_levels<2>> ix;\n"
"  ix.assume_geometry_type(geometry_type::triangle);\n"
"  ix.force_opacity(forced_opacity::opaque);\n"
"  float3 accum = float3(0.0f);\n"
"  for (int sample_index = 0; sample_index < sample_count; sample_index++) {\n"
"    const float2 r = rand2_shadow(tid, sample_index, 0, uniforms.sampling_rand);\n"
"    const bool sample_world = use_world && (light_count <= 0 || r.x > 0.55f);\n"
"    FastGILightRecord light;\n"
"    uint type = LIGHT_SUN;\n"
"    float3 L = float3(0.0f, 0.0f, 1.0f);\n"
"    float light_distance = 100000.0f;\n"
"    float3 light_radiance = float3(1.0f);\n"
"    float light_power = 1.0f;\n"
"    float attenuation = 1.0f;\n"
"    if (sample_world) {\n"
"      L = sample_receiver_caustic_world_direction(tid, sample_index, N, uniforms);\n"
"      light_radiance = sample_receiver_caustic_world_radiance(world_probe_tx, L, uniforms);\n"
"    }\n"
"    else {\n"
"      const int light_index = min(int(r.x * float(light_count)), light_count - 1);\n"
"      light = receiver_lights[light_index];\n"
"      type = uint(light.direction_type.w + 0.5f);\n"
"      if (fast_gi_is_sun(type)) {\n"
"        L = normalize(-light.direction_type.xyz);\n"
"      }\n"
"      else {\n"
"        const float3 to_light = fast_gi_transform_location(light) - P;\n"
"        const float dist_sqr = dot(to_light, to_light);\n"
"        if (!(dist_sqr > 1.0e-10f)) {\n"
"          continue;\n"
"        }\n"
"        light_distance = sqrt(dist_sqr);\n"
"        L = to_light / light_distance;\n"
"      }\n"
"      attenuation = fast_gi_light_surface_attenuation(light, type, L, light_distance);\n"
"      if (!(attenuation > 1.0e-6f)) {\n"
"        continue;\n"
"      }\n"
"      light_power = light.color_diffuse_power.w *\n"
"                    fast_gi_light_point_power(light, type, light_distance, L) *\n"
"                    attenuation * float(light_count);\n"
"      light_radiance = light.color_diffuse_power.xyz;\n"
"    }\n"
"    const float facing = saturate(dot(N, L));\n"
"    if (!(facing > 1.0e-4f)) {\n"
"      continue;\n"
"    }\n"
"    float3 origin = P + N * normal_bias;\n"
"    float3 dir = L;\n"
"    float tmax = fast_gi_is_sun(type) ? 100000.0f : max(light_distance - normal_bias, ray_tmin);\n"
"    float3 throughput = float3(1.0f);\n"
"    bool crossed_refraction = false;\n"
"    for (int bounce = 0; bounce < 8; bounce++) {\n"
"      intersection_result<triangle_data, instancing, max_levels<2>> hit =\n"
"          ix.intersect(ray(origin, dir, ray_tmin, tmax), scene);\n"
"      if (hit.type != intersection_type::triangle) {\n"
"        if (!crossed_refraction) {\n"
"          break;\n"
"        }\n"
"        const float3 to_light = fast_gi_is_sun(type) ? L :\n"
"                                normalize(fast_gi_transform_location(light) - origin);\n"
"        const float align = smoothstep(0.94f, 0.995f, saturate(dot(dir, to_light)));\n"
"        accum += throughput * light_radiance * light_power * facing * align;\n"
"        break;\n"
"      }\n"
"      const uint user_id = hit.user_instance_id[0];\n"
"      const HardwareMaterialProxy proxy = material_proxies[user_id];\n"
"      const uint proxy_closure = uint(proxy.ior_closure_type.z + 0.5f);\n"
"      const uint proxy_flags = uint(proxy.ior_closure_type.w + 0.5f);\n"
"      const bool is_alpha_blend = ((proxy_flags & HWRT_PROXY_FLAG_ALPHA_BLEND_TRANSPARENT) != 0u);\n"
"      if (proxy_closure == HWRT_CLOSURE_REFRACTION) {\n"
"        throughput *= max(proxy.transmission_color_roughness.rgb, float3(0.0f));\n"
"        if (!(dot(throughput, throughput) > 1.0e-10f)) {\n"
"          break;\n"
"        }\n"
"        float3 hit_N = fast_gi_hit_normal(\n"
"            user_id, hit.primitive_id, dir, triangle_normals, triangle_normal_ranges);\n"
"        if (!(dot(hit_N, hit_N) > 1.0e-10f)) {\n"
"          break;\n"
"        }\n"
"        hit_N = normalize(hit_N);\n"
"        const float cos_i_signed = dot(dir, hit_N);\n"
"        const bool entering = (cos_i_signed < 0.0f);\n"
"        const float3 N_oriented = entering ? hit_N : -hit_N;\n"
"        const float cos_i = -dot(dir, N_oriented);\n"
"        const float ior = max(proxy.ior_closure_type.y, 1.0f);\n"
"        const float eta = entering ? (1.0f / ior) : ior;\n"
"        const float sin2_t = eta * eta * (1.0f - cos_i * cos_i);\n"
"        if (sin2_t >= 1.0f) {\n"
"          break;\n"
"        }\n"
"        const float cos_t = sqrt(1.0f - sin2_t);\n"
"        const float3 refracted = normalize(eta * dir + (eta * cos_i - cos_t) * N_oriented);\n"
"        origin = origin + dir * hit.distance + refracted * normal_bias;\n"
"        dir = refracted;\n"
"        if (!fast_gi_is_sun(type)) {\n"
"          tmax = max(length(fast_gi_transform_location(light) - origin), ray_tmin);\n"
"        }\n"
"        crossed_refraction = true;\n"
"        continue;\n"
"      }\n"
"      if (is_alpha_blend) {\n"
"        throughput *= float3(saturate(1.0f - proxy.packed_thickness.y));\n"
"        origin = origin + dir * hit.distance + dir * normal_bias;\n"
"        continue;\n"
"      }\n"
"      break;\n"
"    }\n"
"  }\n"
"  const float3 result = reflected_receiver_gi_luma_clamp(accum / float(sample_count), 2.0f) *\n"
"                        photons_intensity;\n"
"  caustics_img.write(float4(result, 1.0f), tid);\n"
"}\n"
         "kernel void eevee_hardware_trace_local_shadow(\n"
         "    uint2 tid [[thread_position_in_grid]],\n"
         "    instance_acceleration_structure scene [[buffer(0)]],\n"
         "    constant HardwareLocalShadowUniforms &uniforms [[buffer(1)]],\n"
         "    constant HardwareMaterialProxy *material_proxies [[buffer(2)]],\n"
         "    constant float4 *triangle_normals [[buffer(3)]],\n"
         "    constant float4 *triangle_smooth_normals [[buffer(4)]],\n"
         "    constant TriangleNormalRange *triangle_normal_ranges [[buffer(5)]],\n"
         "    depth2d<float, access::sample> depth_tx [[texture(0)]],\n"
         "    texture2d_array<uint, access::read> gbuf_header_tx [[texture(1)]],\n"
         "    texture2d_array<float, access::read> gbuf_normal_tx [[texture(2)]],\n"
         "    texture2d_array<float, access::write> shadow_visibility_img [[texture(3)]])\n"
         "{\n"
         "  if (tid.x >= uint(uniforms.resolution_layer_type.x) || tid.y >= uint(uniforms.resolution_layer_type.y)) {\n"
         "    return;\n"
         "  }\n"
         "  constexpr sampler depth_sampler(coord::normalized, address::clamp_to_edge, filter::nearest);\n"
         "  const float2 uv = (float2(tid) + 0.5f) / float2(uniforms.resolution_layer_type.xy);\n"
         "  const float depth = 1.0f - depth_tx.sample(depth_sampler, uv);\n"
         "  if (!depth_is_valid(depth)) {\n"
         "    shadow_visibility_img.write(float4(1.0f), tid, uint(uniforms.resolution_layer_type.z));\n"
         "    return;\n"
         "  }\n"
         "  const float3 P = point_screen_to_world(int2(tid), depth, uniforms);\n"
         "  float3 center = uniforms.light_position_radius.xyz + uniforms.shadow_offset_scale.xyz;\n"
         "  float3 L = center - P;\n"
         "  const float light_distance = length(L);\n"
         "  if (!(light_distance > 1.0e-5f)) {\n"
         "    shadow_visibility_img.write(float4(1.0f), tid, uint(uniforms.resolution_layer_type.z));\n"
         "    return;\n"
         "  }\n"
         "  L /= light_distance;\n"
         "  float3 N = float3(0.0f);\n"
         "  if (!load_gbuffer_receiver_normal(int2(tid), gbuf_header_tx, gbuf_normal_tx, N)) {\n"
         "    N = estimate_world_normal(int2(tid), depth, depth_tx, uniforms);\n"
         "  }\n"
         "  if (dot(N, N) < 1.0e-10f) {\n"
         "    N = L;\n"
         "  }\n"
         "  const float normal_bias = max(4.0e-3f, uniforms.normal_bias_pad.x);\n"
         "  const float ray_tmin = max(5.0e-4f, normal_bias * 0.25f);\n"
         "  const bool area_soft = is_area_light(uint(uniforms.resolution_layer_type.w)) && (max(uniforms.light_x_axis_size_x.w, uniforms.light_y_axis_size_y.w) * uniforms.shadow_offset_scale.w > 1.0e-6f);\n"
         "  const bool local_soft = (!is_area_light(uint(uniforms.resolution_layer_type.w))) && (uniforms.light_position_radius.w > 1.0e-6f);\n"
         "  const int sample_count = (area_soft || local_soft) ? max(int(uniforms.normal_bias_pad.y), 1) : 1;\n"
         "  const bool enable_caustics = (uniforms.normal_bias_pad.z > 0.5f);\n"
         "  const float color_intensity = saturate(uniforms.normal_bias_pad.w);\n"
         "  const float photons_intensity = max(uniforms.caustic_params.x, 0.0f);\n"
         "  const float transparent_shadows = uniforms.caustic_params.y;\n"
         "  float3 visibility = float3(0.0f);\n"
         "  for (int sample_index = 0; sample_index < sample_count; sample_index++) {\n"
         "    const float3 target = sample_local_shadow_target(tid, sample_index, P, uniforms);\n"
         "    float3 sample_L = target - P;\n"
         "    const float sample_distance = length(sample_L);\n"
         "    if (!(sample_distance > 1.0e-5f)) {\n"
         "      visibility += float3(1.0f);\n"
         "      continue;\n"
         "    }\n"
         "    sample_L /= sample_distance;\n"
"    const float ray_tmax = max(ray_tmin, sample_distance);\n"
"    const float3 origin = P + N * normal_bias;\n"
         "    visibility += hardware_shadow_visibility(scene, origin, sample_L, ray_tmin, ray_tmax, material_proxies, triangle_normals, triangle_smooth_normals, triangle_normal_ranges, transparent_shadows, enable_caustics, false, target, color_intensity, photons_intensity);\n"
         "  }\n"
         "  visibility /= float(sample_count);\n"
         "  shadow_visibility_img.write(float4(visibility, 1.0f), tid, uint(uniforms.resolution_layer_type.z));\n"
         "}\n"
         "kernel void eevee_hardware_trace_local_hit_shadow(\n"
         "    uint3 threadgroup_id [[threadgroup_position_in_grid]],\n"
         "    uint3 local_id [[thread_position_in_threadgroup]],\n"
         "    instance_acceleration_structure scene [[buffer(0)]],\n"
         "    constant HardwareLocalShadowUniforms &uniforms [[buffer(1)]],\n"
         "    constant uint *tiles_coord_buf [[buffer(2)]],\n"
"    constant float4 *triangle_normals [[buffer(3)]],\n"
"    constant TriangleNormalRange *triangle_normal_ranges [[buffer(4)]],\n"
         "    constant HardwareMaterialProxy *material_proxies [[buffer(5)]],\n"
         "    constant float4 *triangle_smooth_normals [[buffer(6)]],\n"
         "    texture2d<float, access::read> hit_normal_img [[texture(0)]],\n"
         "    texture2d<float, access::read> hit_world_position_img [[texture(1)]],\n"
"    texture2d<uint, access::read> hit_identity_img [[texture(2)]],\n"
"    texture2d_array<float, access::write> shadow_visibility_img [[texture(3)]])\n"
         "{\n"
         "  const uint2 tile_coord = unpackUvec2x16(tiles_coord_buf[threadgroup_id.x]);\n"
         "  const uint2 tid = uint2(local_id.xy) + tile_coord * 8u;\n"
         "  if (tid.x >= hit_world_position_img.get_width() || tid.y >= hit_world_position_img.get_height()) {\n"
         "    return;\n"
         "  }\n"
         "  const float3 P = hit_world_position_img.read(tid).xyz;\n"
         "  float3 N = hit_normal_img.read(tid).xyz;\n"
         "  if (!all(isfinite(P)) || !all(isfinite(N)) || dot(N, N) < 1.0e-10f) {\n"
         "    return;\n"
         "  }\n"
         "  N = normalize(N);\n"
"  const float3 shadow_N = hit_shadow_receiver_normal(\n"
"      tid, N, hit_identity_img, triangle_normals, triangle_normal_ranges);\n"
         "  float3 center = uniforms.light_position_radius.xyz + uniforms.shadow_offset_scale.xyz;\n"
         "  float3 L = center - P;\n"
         "  const float light_distance = length(L);\n"
         "  if (!(light_distance > 1.0e-5f)) {\n"
         "    shadow_visibility_img.write(float4(1.0f), tid, uint(uniforms.resolution_layer_type.z));\n"
         "    return;\n"
         "  }\n"
         "  L /= light_distance;\n"
         "  const float normal_bias = max(4.0e-3f, uniforms.normal_bias_pad.x);\n"
         "  const float ray_tmin = max(5.0e-4f, normal_bias * 0.25f);\n"
         "  const bool area_soft = is_area_light(uint(uniforms.resolution_layer_type.w)) && (max(uniforms.light_x_axis_size_x.w, uniforms.light_y_axis_size_y.w) * uniforms.shadow_offset_scale.w > 1.0e-6f);\n"
         "  const bool local_soft = (!is_area_light(uint(uniforms.resolution_layer_type.w))) && (uniforms.light_position_radius.w > 1.0e-6f);\n"
         "  const int sample_count = (area_soft || local_soft) ? max(int(uniforms.normal_bias_pad.y), 1) : 1;\n"
         "  const bool enable_caustics = (uniforms.normal_bias_pad.z > 0.5f);\n"
         "  const float color_intensity = saturate(uniforms.normal_bias_pad.w);\n"
         "  const float photons_intensity = max(uniforms.caustic_params.x, 0.0f);\n"
         "  const float transparent_shadows = uniforms.caustic_params.y;\n"
         "  float3 visibility = float3(0.0f);\n"
         "  for (int sample_index = 0; sample_index < sample_count; sample_index++) {\n"
         "    const float3 target = sample_local_shadow_target(tid, sample_index, P, uniforms);\n"
         "    float3 sample_L = target - P;\n"
         "    const float sample_distance = length(sample_L);\n"
         "    if (!(sample_distance > 1.0e-5f)) {\n"
         "      visibility += float3(1.0f);\n"
         "      continue;\n"
         "    }\n"
         "    sample_L /= sample_distance;\n"
         "    const float ray_tmax = max(ray_tmin, sample_distance);\n"
"    const float3 origin = P + shadow_N * normal_bias;\n"
         "    visibility += hardware_shadow_visibility(scene, origin, sample_L, ray_tmin, ray_tmax, material_proxies, triangle_normals, triangle_smooth_normals, triangle_normal_ranges, transparent_shadows, enable_caustics, false, target, color_intensity, photons_intensity);\n"
         "  }\n"
         "  visibility /= float(sample_count);\n"
         "  shadow_visibility_img.write(float4(visibility, 1.0f), tid, uint(uniforms.resolution_layer_type.z));\n"
         "}\n";
}

static id<MTLLibrary> get_hardware_trace_library(id<MTLDevice> device) API_AVAILABLE(macos(14.0))
{
  static id<MTLLibrary> library = nil;
  if (library != nil) {
    return library;
  }

  MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
  options.fastMathEnabled = YES;
  options.preserveInvariance = YES;
  options.languageVersion = MTLLanguageVersion2_2;
#if defined(MAC_OS_VERSION_14_0)
  if (@available(macos 14.0, *)) {
    options.languageVersion = MTLLanguageVersion3_1;
  }
#endif

  NSError *error = nil;
  library = [device newLibraryWithSource:hardware_trace_shader_source() options:options error:&error];
  [options release];
  if (library == nil) {
    if (error != nil) {
      fprintf(stderr, "Metal RT shader library compile failed: %s\n",
              error.localizedDescription.UTF8String);
    }
    return nil;
  }

  return library;
}

#ifdef WITH_OPENIMAGEDENOISE
static NSString *oidn_interop_shader_source()
{
  return @"#include <metal_stdlib>\n"
          "using namespace metal;\n"
          "kernel void eevee_oidn_pack(texture2d<float, access::read> color_tx [[texture(0)]],\n"
          "                           texture2d<float, access::read> albedo_tx [[texture(1)]],\n"
          "                           texture2d<float, access::read> normal_tx [[texture(2)]],\n"
          "                           device packed_float3 *color_buf [[buffer(0)]],\n"
          "                           device packed_float3 *albedo_buf [[buffer(1)]],\n"
          "                           device packed_float3 *normal_buf [[buffer(2)]],\n"
          "                           constant uint2 &extent [[buffer(3)]],\n"
          "                           constant uint2 &use_aux [[buffer(4)]],\n"
          "                           uint2 tid [[thread_position_in_grid]])\n"
          "{\n"
          "  if (tid.x >= extent.x || tid.y >= extent.y) {\n"
          "    return;\n"
          "  }\n"
          "  const uint index = tid.y * extent.x + tid.x;\n"
          "  color_buf[index] = packed_float3(max(color_tx.read(tid).xyz, float3(0.0f)));\n"
          "  if (use_aux.x != 0u) {\n"
          "    albedo_buf[index] = packed_float3(max(albedo_tx.read(tid).xyz, float3(0.0f)));\n"
          "  }\n"
          "  if (use_aux.y != 0u) {\n"
          "    float3 normal = normal_tx.read(tid).xyz;\n"
          "    normal_buf[index] = packed_float3((dot(normal, normal) > 1.0e-10f) ? normalize(normal) : float3(0.0f, 0.0f, 1.0f));\n"
          "  }\n"
          "}\n"
          "kernel void eevee_oidn_unpack(texture2d<float, access::read> input_tx [[texture(0)]],\n"
          "                             texture2d<float, access::write> output_tx [[texture(1)]],\n"
          "                             device const packed_float3 *output_buf [[buffer(0)]],\n"
          "                             constant uint2 &extent [[buffer(1)]],\n"
          "                             uint2 tid [[thread_position_in_grid]])\n"
          "{\n"
          "  if (tid.x >= extent.x || tid.y >= extent.y) {\n"
          "    return;\n"
          "  }\n"
          "  const uint index = tid.y * extent.x + tid.x;\n"
          "  const float alpha = input_tx.read(tid).w;\n"
          "  output_tx.write(float4(float3(output_buf[index]), alpha), tid);\n"
          "}\n";
}

static id<MTLLibrary> get_oidn_interop_library(id<MTLDevice> device)
{
  static id<MTLLibrary> library = nil;
  if (library != nil) {
    return library;
  }

  MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
  options.fastMathEnabled = YES;
  options.preserveInvariance = YES;
  options.languageVersion = MTLLanguageVersion2_2;

  NSError *error = nil;
  library = [device newLibraryWithSource:oidn_interop_shader_source() options:options error:&error];
  [options release];
  if (library == nil) {
    if (error != nil) {
      fprintf(stderr, "Metal OIDN interop shader compile failed: %s\n",
              error.localizedDescription.UTF8String);
    }
    return nil;
  }

  return library;
}

static id<MTLComputePipelineState> get_oidn_interop_pipeline(id<MTLDevice> device,
                                                             NSString *function_name)
{
  id<MTLLibrary> library = get_oidn_interop_library(device);
  if (library == nil) {
    return nil;
  }

  NSError *error = nil;
  id<MTLFunction> function = [library newFunctionWithName:function_name];
  if (function == nil) {
    return nil;
  }
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                               error:&error];
  [function release];
  if (pipeline == nil && error != nil) {
    fprintf(stderr, "Metal OIDN interop pipeline creation failed: %s\n",
            error.localizedDescription.UTF8String);
  }
  return pipeline;
}

static id<MTLComputePipelineState> get_oidn_pack_pipeline(id<MTLDevice> device)
{
  static id<MTLComputePipelineState> pipeline = nil;
  if (pipeline == nil) {
    pipeline = get_oidn_interop_pipeline(device, @"eevee_oidn_pack");
  }
  return pipeline;
}

static id<MTLComputePipelineState> get_oidn_unpack_pipeline(id<MTLDevice> device)
{
  static id<MTLComputePipelineState> pipeline = nil;
  if (pipeline == nil) {
    pipeline = get_oidn_interop_pipeline(device, @"eevee_oidn_unpack");
  }
  return pipeline;
}

struct OIDNInteropCache {
  OIDNDevice device = nullptr;
  OIDNFilter filter = nullptr;
  id<MTLCommandQueue> queue = nil;
  bool device_uses_gpu = true;
  bool filter_uses_aux = false;
  id<MTLBuffer> color_buffer = nil;
  id<MTLBuffer> albedo_buffer = nil;
  id<MTLBuffer> normal_buffer = nil;
  id<MTLBuffer> output_buffer = nil;
  bool filter_use_albedo = false;
  bool filter_use_normal = false;
  int filter_quality = 0;
  int filter_prefilter = 0;

  ~OIDNInteropCache()
  {
    release_oidn();
    release_buffers();
  }

  void release_oidn()
  {
    if (device != nullptr) {
      oidnSyncDevice(device);
    }
    if (filter != nullptr) {
      oidnReleaseFilter(filter);
      filter = nullptr;
    }
    if (device != nullptr) {
      oidnReleaseDevice(device);
      device = nullptr;
    }
    if (queue != nil) {
      [queue release];
      queue = nil;
    }
  }

  void release_buffers()
  {
    if (color_buffer != nil) {
      [color_buffer release];
      color_buffer = nil;
    }
    if (albedo_buffer != nil) {
      [albedo_buffer release];
      albedo_buffer = nil;
    }
    if (normal_buffer != nil) {
      [normal_buffer release];
      normal_buffer = nil;
    }
    if (output_buffer != nil) {
      [output_buffer release];
      output_buffer = nil;
    }
  }
};

static OIDNInteropCache &oidn_interop_cache()
{
  static OIDNInteropCache cache;
  return cache;
}

static void oidn_sync_device(OIDNInteropCache &cache)
{
  if (cache.device != nullptr) {
    oidnSyncDevice(cache.device);
  }
}

static bool oidn_report_error(OIDNDevice device, const char *context)
{
  const char *error_message = nullptr;
  OIDNError error = oidnGetDeviceError(device, &error_message);
  if (error == OIDN_ERROR_NONE) {
    return false;
  }

  fprintf(stderr, "Eevee OIDN %s failed: %s\n", context, error_message ? error_message : "unknown");
  return true;
}

static bool oidn_perf_logging_enabled()
{
  const char *value = std::getenv("BLENDER_EEVEE_HWRT_PERF");
  return (value != nullptr) && (value[0] != '\0') && !(value[0] == '0' && value[1] == '\0');
}

static bool ensure_oidn_buffer(id<MTLDevice> device,
                               id<MTLBuffer> &buffer,
                               NSUInteger byte_size,
                               const char *label,
                               bool shared)
{
  const MTLStorageMode storage_mode = shared ? MTLStorageModeShared : MTLStorageModePrivate;
  if (buffer != nil && [buffer length] >= byte_size && [buffer storageMode] == storage_mode) {
    return true;
  }
  if (buffer != nil) {
    [buffer release];
    buffer = nil;
  }

  buffer = [device newBufferWithLength:byte_size
                               options:shared ? MTLResourceStorageModeShared :
                                                MTLResourceStorageModePrivate];
  if (buffer == nil) {
    return false;
  }
  buffer.label = [NSString stringWithUTF8String:label];
  return true;
}

static bool ensure_oidn_device(OIDNInteropCache &cache,
                               id<MTLDevice> device,
                               id<MTLCommandQueue> queue,
                               bool use_gpu)
{
  if (cache.device != nullptr && cache.device_uses_gpu == use_gpu &&
      (!use_gpu || cache.queue == queue))
  {
    return true;
  }

  cache.release_oidn();
  cache.device_uses_gpu = use_gpu;
  if (use_gpu) {
    if (!oidnIsMetalDeviceSupported(device)) {
      return false;
    }

    cache.queue = [queue retain];
    MTLCommandQueue_id oidn_queue = cache.queue;
    cache.device = oidnNewMetalDevice(&oidn_queue, 1);
  }
  else {
    cache.device = oidnNewDevice(OIDN_DEVICE_TYPE_CPU);
  }

  if (cache.device == nullptr) {
    return false;
  }
  oidnCommitDevice(cache.device);
  return !oidn_report_error(cache.device, "device commit");
}

static int oidn_quality_from_nuru(const int quality)
{
  switch (quality) {
    case 1:
      return OIDN_QUALITY_HIGH;
    case 3:
      return OIDN_QUALITY_FAST;
    case 2:
    default:
      return OIDN_QUALITY_BALANCED;
  }
}

static bool ensure_oidn_filter(OIDNInteropCache &cache,
                               bool use_albedo,
                               bool use_normal,
                               int quality,
                               int prefilter)
{
  if (cache.filter != nullptr && cache.filter_use_albedo == use_albedo &&
      cache.filter_use_normal == use_normal && cache.filter_quality == quality &&
      cache.filter_prefilter == prefilter)
  {
    return true;
  }
  if (cache.filter != nullptr) {
    oidn_sync_device(cache);
    oidnReleaseFilter(cache.filter);
    cache.filter = nullptr;
  }

  cache.filter = oidnNewFilter(cache.device, "RT");
  if (cache.filter == nullptr) {
    return !oidn_report_error(cache.device, "filter creation");
  }

  oidnSetFilterBool(cache.filter, "hdr", true);
  oidnSetFilterBool(cache.filter, "srgb", false);
  oidnSetFilterInt(cache.filter, "quality", oidn_quality_from_nuru(quality));
  oidnSetFilterBool(cache.filter, "cleanAux", prefilter != 2);
  cache.filter_uses_aux = use_albedo || use_normal;
  cache.filter_use_albedo = use_albedo;
  cache.filter_use_normal = use_normal;
  cache.filter_quality = quality;
  cache.filter_prefilter = prefilter;
  return true;
}

static bool run_oidn_interop_pack(id<MTLCommandQueue> queue,
                                  id<MTLComputePipelineState> pipeline,
                                  id<MTLTexture> input,
                                  id<MTLTexture> albedo,
                                  id<MTLTexture> normal,
                                  OIDNInteropCache &cache,
                                  int2 extent,
                                  bool use_albedo,
                                  bool use_normal,
                                  bool wait_until_completed)
{
  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  if (command_buffer == nil || encoder == nil) {
    return false;
  }

  uint2 mtl_extent = uint2(uint(extent.x), uint(extent.y));
  uint2 use_aux = uint2(use_albedo ? 1u : 0u, use_normal ? 1u : 0u);
  [encoder setComputePipelineState:pipeline];
  [encoder setTexture:input atIndex:0];
  [encoder setTexture:use_albedo ? albedo : input atIndex:1];
  [encoder setTexture:use_normal ? normal : input atIndex:2];
  [encoder setBuffer:cache.color_buffer offset:0 atIndex:0];
  [encoder setBuffer:use_albedo ? cache.albedo_buffer : cache.color_buffer offset:0 atIndex:1];
  [encoder setBuffer:use_normal ? cache.normal_buffer : cache.color_buffer offset:0 atIndex:2];
  [encoder setBytes:&mtl_extent length:sizeof(mtl_extent) atIndex:3];
  [encoder setBytes:&use_aux length:sizeof(use_aux) atIndex:4];
  [encoder dispatchThreads:MTLSizeMake(extent.x, extent.y, 1)
     threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
  [encoder endEncoding];
  [command_buffer commit];
  if (wait_until_completed) {
    [command_buffer waitUntilCompleted];
    return command_buffer.status == MTLCommandBufferStatusCompleted;
  }
  return true;
}

static bool run_oidn_interop_unpack(id<MTLCommandQueue> queue,
                                    id<MTLComputePipelineState> pipeline,
                                    id<MTLTexture> input,
                                    id<MTLTexture> output,
                                    OIDNInteropCache &cache,
                                    int2 extent,
                                    bool wait_until_completed)
{
  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  if (command_buffer == nil || encoder == nil) {
    return false;
  }

  uint2 mtl_extent = uint2(uint(extent.x), uint(extent.y));
  [encoder setComputePipelineState:pipeline];
  [encoder setTexture:input atIndex:0];
  [encoder setTexture:output atIndex:1];
  [encoder setBuffer:cache.output_buffer offset:0 atIndex:0];
  [encoder setBytes:&mtl_extent length:sizeof(mtl_extent) atIndex:1];
  [encoder dispatchThreads:MTLSizeMake(extent.x, extent.y, 1)
     threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
  [encoder endEncoding];
  [command_buffer commit];
  if (wait_until_completed) {
    [command_buffer waitUntilCompleted];
    return command_buffer.status == MTLCommandBufferStatusCompleted;
  }
  return true;
}

static bool set_oidn_image(OIDNDevice device,
                           OIDNFilter filter,
                           const char *name,
                           id<MTLBuffer> buffer,
                           size_t width,
                           size_t height)
{
  OIDNBuffer oidn_buffer = oidnNewSharedBufferFromMetal(device, buffer);
  if (oidn_buffer == nullptr) {
    return !oidn_report_error(device, name);
  }

  const size_t pixel_stride = sizeof(float) * 3;
  oidnSetFilterImage(
      filter, name, oidn_buffer, OIDN_FORMAT_FLOAT3, width, height, 0, pixel_stride, width * pixel_stride);
  oidnReleaseBuffer(oidn_buffer);
  return true;
}

static bool set_oidn_cpu_image(OIDNFilter filter,
                               const char *name,
                               id<MTLBuffer> buffer,
                               size_t width,
                               size_t height)
{
  void *data = [buffer contents];
  if (data == nullptr) {
    return false;
  }

  const size_t pixel_stride = sizeof(float) * 3;
  oidnSetSharedFilterImage(
      filter, name, data, OIDN_FORMAT_FLOAT3, width, height, 0, pixel_stride, width * pixel_stride);
  return true;
}
#endif

static id<MTLComputePipelineState> get_hardware_trace_pipeline(id<MTLDevice> device)
    API_AVAILABLE(macos(14.0))
{
  static id<MTLComputePipelineState> pipeline = nil;
  if (pipeline != nil) {
    return pipeline;
  }

  id<MTLLibrary> library = get_hardware_trace_library(device);
  if (library == nil) {
    return nil;
  }

  id<MTLFunction> function = [library newFunctionWithName:@"eevee_hardware_trace_override"];
  if (function == nil) {
    return nil;
  }

  NSError *error = nil;
  pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  [function release];
  if (pipeline == nil && error != nil) {
    fprintf(stderr, "Metal RT pipeline creation failed: %s\n",
            error.localizedDescription.UTF8String);
  }
  return pipeline;
}

static id<MTLComputePipelineState> get_hardware_reflected_receiver_gi_pipeline(
    id<MTLDevice> device) API_AVAILABLE(macos(14.0))
{
  static id<MTLComputePipelineState> pipeline = nil;
  if (pipeline != nil) {
    return pipeline;
  }

  id<MTLLibrary> library = get_hardware_trace_library(device);
  if (library == nil) {
    return nil;
  }

  id<MTLFunction> function = [library
      newFunctionWithName:@"eevee_hardware_trace_reflected_receiver_gi"];
  if (function == nil) {
    return nil;
  }

  NSError *error = nil;
  pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  [function release];
  if (pipeline == nil && error != nil) {
    fprintf(stderr, "Metal RT reflected receiver GI pipeline creation failed: %s\n",
            error.localizedDescription.UTF8String);
  }
  return pipeline;
}

static id<MTLComputePipelineState> get_hardware_directional_shadow_pipeline(id<MTLDevice> device)
    API_AVAILABLE(macos(14.0))
{
  static id<MTLComputePipelineState> pipeline = nil;
  if (pipeline != nil) {
    return pipeline;
  }

  id<MTLLibrary> library = get_hardware_trace_library(device);
  if (library == nil) {
    return nil;
  }

  NSError *error = nil;
  id<MTLFunction> function = [library newFunctionWithName:@"eevee_hardware_trace_directional_shadow"];
  if (function == nil) {
    return nil;
  }

  pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  [function release];
  if (pipeline == nil && error != nil) {
    fprintf(stderr, "Metal RT shadow pipeline creation failed: %s\n",
            error.localizedDescription.UTF8String);
  }
  return pipeline;
}

static id<MTLComputePipelineState> get_hardware_directional_hit_shadow_pipeline(id<MTLDevice> device)
    API_AVAILABLE(macos(14.0))
{
  static id<MTLComputePipelineState> pipeline = nil;
  if (pipeline != nil) {
    return pipeline;
  }

  id<MTLLibrary> library = get_hardware_trace_library(device);
  if (library == nil) {
    return nil;
  }

  NSError *error = nil;
  id<MTLFunction> function = [library newFunctionWithName:@"eevee_hardware_trace_directional_hit_shadow"];
  if (function == nil) {
    return nil;
  }

  pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  [function release];
  if (pipeline == nil && error != nil) {
    fprintf(stderr, "Metal RT hit-shadow pipeline creation failed: %s\n",
            error.localizedDescription.UTF8String);
  }
  return pipeline;
}

static id<MTLComputePipelineState> get_hardware_environment_visibility_pipeline(
    id<MTLDevice> device) API_AVAILABLE(macos(14.0))
{
  static id<MTLComputePipelineState> pipeline = nil;
  if (pipeline != nil) {
    return pipeline;
  }

  id<MTLLibrary> library = get_hardware_trace_library(device);
  if (library == nil) {
    return nil;
  }

  NSError *error = nil;
  id<MTLFunction> function = [library
      newFunctionWithName:@"eevee_hardware_trace_environment_visibility"];
  if (function == nil) {
    return nil;
  }

  pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  [function release];
  if (pipeline == nil && error != nil) {
    fprintf(stderr,
            "Metal RT environment visibility pipeline creation failed: %s\n",
            error.localizedDescription.UTF8String);
  }
  return pipeline;
}

static id<MTLComputePipelineState> get_hardware_hit_environment_visibility_pipeline(
    id<MTLDevice> device) API_AVAILABLE(macos(14.0))
{
  static id<MTLComputePipelineState> pipeline = nil;
  if (pipeline != nil) {
    return pipeline;
  }

  id<MTLLibrary> library = get_hardware_trace_library(device);
  if (library == nil) {
    return nil;
  }

  NSError *error = nil;
  id<MTLFunction> function = [library
      newFunctionWithName:@"eevee_hardware_trace_hit_environment_visibility"];
  if (function == nil) {
    return nil;
  }

  pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  [function release];
  if (pipeline == nil && error != nil) {
    fprintf(stderr,
            "Metal RT hit environment visibility pipeline creation failed: %s\n",
            error.localizedDescription.UTF8String);
  }
  return pipeline;
}

static id<MTLComputePipelineState> get_hardware_local_hit_shadow_pipeline(id<MTLDevice> device)
    API_AVAILABLE(macos(14.0))
{
  static id<MTLComputePipelineState> pipeline = nil;
  if (pipeline != nil) {
    return pipeline;
  }

  id<MTLLibrary> library = get_hardware_trace_library(device);
  if (library == nil) {
    return nil;
  }

  NSError *error = nil;
  id<MTLFunction> function = [library newFunctionWithName:@"eevee_hardware_trace_local_hit_shadow"];
  if (function == nil) {
    return nil;
  }

  pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  [function release];
  if (pipeline == nil && error != nil) {
    fprintf(stderr, "Metal RT local hit-shadow pipeline creation failed: %s\n",
            error.localizedDescription.UTF8String);
  }
  return pipeline;
}

static id<MTLComputePipelineState> get_hardware_local_shadow_pipeline(id<MTLDevice> device)
    API_AVAILABLE(macos(14.0))
{
  static id<MTLComputePipelineState> pipeline = nil;
  if (pipeline != nil) {
    return pipeline;
  }

  id<MTLLibrary> library = get_hardware_trace_library(device);
  if (library == nil) {
    return nil;
  }

  NSError *error = nil;
  id<MTLFunction> function = [library newFunctionWithName:@"eevee_hardware_trace_local_shadow"];
  if (function == nil) {
    return nil;
  }

  pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  [function release];
  if (pipeline == nil && error != nil) {
    fprintf(stderr, "Metal RT local shadow pipeline creation failed: %s\n",
            error.localizedDescription.UTF8String);
  }
  return pipeline;
}

GPUHardwareRaytraceScene *raytrace_scene_build(Span<GPUHardwareRaytraceSceneEntry> entries,
                                            GPUHardwareRaytraceSceneStats *r_stats)
{
  if (r_stats != nullptr) {
    *r_stats = {};
  }

  if (!GPU_hardware_raytracing_support()) {
    return nullptr;
  }

#if defined(MAC_OS_VERSION_14_0)
  if (@available(macos 14.0, *)) {
    const bool perf_logging_enabled = metal_raytrace_perf_logging_enabled();
    const double build_start_time = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
    MTLContext *ctx = MTLContext::get();
    if (ctx == nullptr || ctx->device == nil || ctx->queue == nil) {
      return nullptr;
    }

    std::vector<SceneGeometryBuild> built_geometry;
    built_geometry.reserve(entries.size());
    AccelerationStructureBuildBatch blas_build_batch;
    if (!begin_acceleration_structure_build_batch(ctx->queue,
                                                  "Metal RT BLAS build",
                                                  blas_build_batch))
    {
      return nullptr;
    }

    GPUHardwareRaytraceScene *scene = new GPUHardwareRaytraceScene();
    for (const GPUHardwareRaytraceSceneEntry &entry : entries) {
      SceneGeometryBuild geometry;
      if (!build_entry_blas(ctx, entry, geometry, &blas_build_batch)) {
        continue;
      }

      built_geometry.push_back(geometry);
      scene->bottom_level_acceleration_structures.push_back(geometry.acceleration_structure);
      scene->local_triangle_normals.push_back(geometry.triangle_normals);
      scene->local_triangle_smooth_normals.push_back(geometry.triangle_smooth_normals);
      scene->local_triangle_positions.push_back(geometry.triangle_local_positions);
      if (geometry.vertex_buffer != nil) {
        scene->geometry_buffers.push_back(geometry.vertex_buffer);
      }
      if (geometry.index_buffer != nil) {
        scene->geometry_buffers.push_back(geometry.index_buffer);
      }
      scene->geometry_count++;
      scene->instance_count += int(geometry.instance_count);
    }
    commit_acceleration_structure_build_batch(blas_build_batch);

    if (r_stats != nullptr) {
      r_stats->geometry_count = scene->geometry_count;
      r_stats->instance_count = scene->instance_count;
      r_stats->built_blas_count = int(scene->bottom_level_acceleration_structures.size());
    }

    if (scene->bottom_level_acceleration_structures.empty() || scene->instance_count == 0) {
      delete scene;
      return nullptr;
    }

    scene->top_level_acceleration_structure = build_top_level_acceleration_structure(
        ctx->device, ctx->queue, built_geometry);
    if (scene->top_level_acceleration_structure == nil) {
      delete scene;
      return nullptr;
    }

    std::vector<TriangleNormalRangeRecord> triangle_normal_ranges;
    scene->emissive_radiance_buffer = build_emissive_radiance_buffer(ctx->device, built_geometry);
    scene->emissive_light_buffer = build_emissive_light_buffer(
        ctx->device, built_geometry, scene->emissive_light_count);
    scene->diffuse_albedo_buffer = build_diffuse_albedo_buffer(ctx->device, built_geometry);
    scene->material_proxy_buffer = build_material_proxy_buffer(ctx->device, built_geometry);
    scene->triangle_normal_buffer = build_triangle_normal_buffer(
        ctx->device, built_geometry, triangle_normal_ranges);
    scene->triangle_smooth_normal_buffer = build_triangle_smooth_normal_buffer(
        ctx->device, built_geometry, triangle_normal_ranges);
    scene->triangle_local_position_buffer = build_triangle_local_position_buffer(ctx->device,
                                                                                 built_geometry);
    scene->triangle_normal_range_buffer = build_triangle_normal_range_buffer(
        ctx->device, triangle_normal_ranges);
    if (scene->emissive_radiance_buffer == nil || scene->emissive_light_buffer == nil ||
        scene->diffuse_albedo_buffer == nil ||
        scene->material_proxy_buffer == nil || scene->triangle_normal_buffer == nil ||
        scene->triangle_smooth_normal_buffer == nil || scene->triangle_local_position_buffer == nil ||
        scene->triangle_normal_range_buffer == nil)
    {
      delete scene;
      return nullptr;
    }

    if (r_stats != nullptr) {
      r_stats->emissive_light_count = scene->emissive_light_count;
      r_stats->emissive_energy_sum = scene_emissive_energy_sum(built_geometry);
      r_stats->built_scene = true;
    }
    if (perf_logging_enabled) {
      const double elapsed_ms = (BLI_time_now_seconds() - build_start_time) * 1000.0;
      std::fprintf(stderr,
                   "EEVEE HWRT perf rt_scene_build geometries=%d instances=%d built_blas=%d emissive_lights=%d elapsed_ms=%.2f\n",
                   scene->geometry_count,
                   scene->instance_count,
                   int(scene->bottom_level_acceleration_structures.size()),
                   scene->emissive_light_count,
                   elapsed_ms);
    }
    return scene;
  }
#endif

  return nullptr;
}

bool raytrace_scene_update(GPUHardwareRaytraceScene *scene,
                           Span<GPUHardwareRaytraceSceneEntry> entries,
                           const GPUHardwareRaytraceSceneUpdateParams &update_params,
                           GPUHardwareRaytraceSceneStats *r_stats)
{
  if (r_stats != nullptr) {
    *r_stats = {};
  }
  if (scene == nullptr || scene->bottom_level_acceleration_structures.size() != entries.size() ||
      scene->local_triangle_normals.size() != entries.size() ||
      scene->local_triangle_smooth_normals.size() != entries.size() ||
      scene->local_triangle_positions.size() != entries.size())
  {
    return false;
  }

  /* Selective per-entry BLAS rebuild is not implemented on Metal yet. Returning false routes
   * the caller (`RayTraceModule::acquire_hardware_rt_scene`) through the existing full scene
   * rebuild, which is exactly the established Metal behavior for geometry recalcs. Do not
   * silently ignore the span: the flagged entries' geometry changed, and a partial update
   * without rebuilding their BLAS would trace stale geometry. */
  if (!update_params.rebuild_blas_indices.is_empty()) {
    return false;
  }

  if (!GPU_hardware_raytracing_support()) {
    return false;
  }

#if defined(MAC_OS_VERSION_14_0)
  if (@available(macos 14.0, *)) {
    const bool perf_logging_enabled = metal_raytrace_perf_logging_enabled();
    const double update_start_time = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
    MTLContext *ctx = MTLContext::get();
    if (ctx == nullptr || ctx->device == nil || ctx->queue == nil) {
      return false;
    }

    if (!update_params.update_tlas && !update_params.update_emissive_data &&
        !update_params.update_material_data && !update_params.update_world_geometry_data)
    {
      if (r_stats != nullptr) {
        r_stats->geometry_count = scene->geometry_count;
        r_stats->instance_count = scene->instance_count;
        r_stats->built_blas_count = 0;
        r_stats->emissive_light_count = scene->emissive_light_count;
        r_stats->built_scene = true;
      }
      return true;
    }

    std::vector<SceneGeometryBuild> updated_geometry;
    updated_geometry.reserve(entries.size());
    int instance_count = 0;
    for (const int i : entries.index_range()) {
      const GPUHardwareRaytraceSceneEntry &entry = entries[i];
      SceneGeometryBuild geometry;
      geometry.acceleration_structure = scene->bottom_level_acceleration_structures[i];
      geometry.object_to_world = entry.object_to_world;
      geometry.instance_count = std::max(entry.instance_count, uint32_t(1));
      geometry.user_id = uint32_t(i);
      geometry.emissive_radiance = entry.emissive_radiance;
      geometry.diffuse_albedo = entry.diffuse_albedo;
      geometry.reflection_color = entry.reflection_color;
      geometry.reflection_roughness = entry.reflection_roughness;
      geometry.transmission_color = entry.transmission_color;
      geometry.transmission_roughness = entry.transmission_roughness;
      geometry.reflection_ior = entry.reflection_ior;
      geometry.refraction_ior = entry.refraction_ior;
      geometry.packed_thickness = entry.packed_thickness;
      geometry.alpha = entry.alpha;
      geometry.reflection_layer_coverage = entry.reflection_layer_coverage;
      geometry.closure_type = entry.closure_type;
      geometry.proxy_flags = entry.proxy_flags;
      geometry.triangle_normals = scene->local_triangle_normals[i];
      geometry.triangle_smooth_normals = scene->local_triangle_smooth_normals[i];
      geometry.triangle_local_positions = scene->local_triangle_positions[i];
      updated_geometry.push_back(std::move(geometry));
      instance_count += int(updated_geometry.back().instance_count);
    }

    if (updated_geometry.empty() || instance_count == 0) {
      return false;
    }

    id<MTLAccelerationStructure> new_tlas = nil;
    if (update_params.update_tlas) {
      new_tlas = build_top_level_acceleration_structure(ctx->device, ctx->queue, updated_geometry);
      if (new_tlas == nil) {
        return false;
      }
    }

    int new_emissive_light_count = scene->emissive_light_count;
    id<MTLBuffer> new_emissive = nil;
    id<MTLBuffer> new_emissive_lights = nil;
    id<MTLBuffer> new_diffuse = nil;
    id<MTLBuffer> new_proxy = nil;
    id<MTLBuffer> new_triangle_normals = nil;
    id<MTLBuffer> new_triangle_smooth_normals = nil;
    if (update_params.update_emissive_data) {
      new_emissive = build_emissive_radiance_buffer(ctx->device, updated_geometry);
      new_emissive_lights = build_emissive_light_buffer(
          ctx->device, updated_geometry, new_emissive_light_count);
      if (new_emissive == nil || new_emissive_lights == nil) {
        if (new_tlas != nil) {
          [new_tlas release];
        }
        if (new_emissive != nil) {
          [new_emissive release];
        }
        if (new_emissive_lights != nil) {
          [new_emissive_lights release];
        }
        return false;
      }
    }
    if (update_params.update_material_data) {
      new_diffuse = build_diffuse_albedo_buffer(ctx->device, updated_geometry);
      new_proxy = build_material_proxy_buffer(ctx->device, updated_geometry);
      if (new_diffuse == nil || new_proxy == nil)
      {
        if (new_tlas != nil) {
          [new_tlas release];
        }
        if (new_emissive != nil) {
          [new_emissive release];
        }
        if (new_emissive_lights != nil) {
          [new_emissive_lights release];
        }
        if (new_diffuse != nil) {
          [new_diffuse release];
        }
        if (new_proxy != nil) {
          [new_proxy release];
        }
        return false;
      }
    }
    if (update_params.update_world_geometry_data) {
      std::vector<TriangleNormalRangeRecord> triangle_normal_ranges;
      new_triangle_normals = build_triangle_normal_buffer(
          ctx->device, updated_geometry, triangle_normal_ranges);
      new_triangle_smooth_normals = build_triangle_smooth_normal_buffer(
          ctx->device, updated_geometry, triangle_normal_ranges);
      if (new_triangle_normals == nil || new_triangle_smooth_normals == nil) {
        if (new_tlas != nil) {
          [new_tlas release];
        }
        if (new_emissive != nil) {
          [new_emissive release];
        }
        if (new_emissive_lights != nil) {
          [new_emissive_lights release];
        }
        if (new_diffuse != nil) {
          [new_diffuse release];
        }
        if (new_proxy != nil) {
          [new_proxy release];
        }
        if (new_triangle_normals != nil) {
          [new_triangle_normals release];
        }
        if (new_triangle_smooth_normals != nil) {
          [new_triangle_smooth_normals release];
        }
        return false;
      }
    }

    if (update_params.update_tlas) {
      if (scene->top_level_acceleration_structure != nil) {
        [scene->top_level_acceleration_structure release];
      }
      scene->top_level_acceleration_structure = new_tlas;
    }
    if (update_params.update_emissive_data) {
      if (scene->emissive_radiance_buffer != nil) {
        [scene->emissive_radiance_buffer release];
      }
      if (scene->emissive_light_buffer != nil) {
        [scene->emissive_light_buffer release];
      }
      scene->emissive_radiance_buffer = new_emissive;
      scene->emissive_light_buffer = new_emissive_lights;
      scene->emissive_light_count = new_emissive_light_count;
    }
    if (update_params.update_material_data) {
      if (scene->diffuse_albedo_buffer != nil) {
        [scene->diffuse_albedo_buffer release];
      }
      if (scene->material_proxy_buffer != nil) {
        [scene->material_proxy_buffer release];
      }
      scene->diffuse_albedo_buffer = new_diffuse;
      scene->material_proxy_buffer = new_proxy;
    }
    if (update_params.update_world_geometry_data) {
      if (scene->triangle_normal_buffer != nil) {
        [scene->triangle_normal_buffer release];
      }
      if (scene->triangle_smooth_normal_buffer != nil) {
        [scene->triangle_smooth_normal_buffer release];
      }
      scene->triangle_normal_buffer = new_triangle_normals;
      scene->triangle_smooth_normal_buffer = new_triangle_smooth_normals;
    }
    scene->geometry_count = int(updated_geometry.size());
    scene->instance_count = instance_count;

    if (r_stats != nullptr) {
      r_stats->geometry_count = scene->geometry_count;
      r_stats->instance_count = scene->instance_count;
      r_stats->built_blas_count = 0;
      r_stats->emissive_light_count = scene->emissive_light_count;
      r_stats->emissive_energy_sum = scene_emissive_energy_sum(updated_geometry);
      r_stats->built_scene = true;
    }
    if (perf_logging_enabled) {
      const double elapsed_ms = (BLI_time_now_seconds() - update_start_time) * 1000.0;
      std::fprintf(stderr,
                   "EEVEE HWRT perf rt_scene_update tlas=%d emissive=%d material=%d world_geom=%d geometries=%d instances=%d elapsed_ms=%.2f\n",
                   update_params.update_tlas ? 1 : 0,
                   update_params.update_emissive_data ? 1 : 0,
                   update_params.update_material_data ? 1 : 0,
                   update_params.update_world_geometry_data ? 1 : 0,
                   scene->geometry_count,
                   scene->instance_count,
                   elapsed_ms);
    }
    return true;
  }
#endif

  return false;
}

bool raytrace_scene_trace(GPUHardwareRaytraceScene *scene, const GPUHardwareRaytraceTraceParams &params)
{
  if (scene == nullptr || scene->top_level_acceleration_structure == nil || params.ray_data_tx == nullptr ||
      params.depth_tx == nullptr || params.gbuf_header_tx == nullptr ||
      params.gbuf_normal_tx == nullptr || params.screen_continuation_tx == nullptr ||
      params.ray_time_tx == nullptr ||
      params.ray_radiance_tx == nullptr || params.hit_albedo_tx == nullptr ||
      params.hit_throughput_tx == nullptr ||
      params.hit_material_tx == nullptr || params.hit_normal_tx == nullptr ||
      params.hit_position_tx == nullptr || params.hit_world_position_tx == nullptr ||
      params.hit_identity_tx == nullptr || params.hit_barycentric_tx == nullptr ||
      params.layered_receiver_ray_time_tx == nullptr ||
      params.layered_receiver_ray_radiance_tx == nullptr ||
      params.layered_receiver_albedo_tx == nullptr ||
      params.layered_receiver_throughput_tx == nullptr ||
      params.layered_receiver_material_tx == nullptr ||
      params.layered_receiver_normal_tx == nullptr ||
      params.layered_receiver_position_tx == nullptr ||
      params.layered_receiver_world_position_tx == nullptr ||
      params.layered_receiver_identity_tx == nullptr ||
      params.layered_receiver_barycentric_tx == nullptr ||
      params.transmission_receiver_ray_time_tx == nullptr ||
      params.transmission_receiver_ray_radiance_tx == nullptr ||
      params.transmission_receiver_albedo_tx == nullptr ||
      params.transmission_receiver_throughput_tx == nullptr ||
      params.transmission_receiver_material_tx == nullptr ||
      params.transmission_receiver_normal_tx == nullptr ||
      params.transmission_receiver_position_tx == nullptr ||
      params.transmission_receiver_world_position_tx == nullptr ||
      params.transmission_receiver_identity_tx == nullptr ||
      params.transmission_receiver_barycentric_tx == nullptr)
  {
    return false;
  }

  if (!GPU_hardware_raytracing_support()) {
    return false;
  }

#if defined(MAC_OS_VERSION_14_0)
  if (@available(macos 14.0, *)) {
    MTLContext *ctx = MTLContext::get();
    if (ctx == nullptr || ctx->device == nil || ctx->queue == nil) {
      return false;
    }

    id<MTLComputePipelineState> pipeline = get_hardware_trace_pipeline(ctx->device);
    if (pipeline == nil) {
      return false;
    }

    MTLTexture *ray_data_tx = unwrap(params.ray_data_tx);
    MTLTexture *depth_tx = unwrap(params.depth_tx);
    MTLTexture *gbuf_header_tx = unwrap(params.gbuf_header_tx);
    MTLTexture *gbuf_normal_tx = unwrap(params.gbuf_normal_tx);
    MTLTexture *screen_continuation_tx = unwrap(params.screen_continuation_tx);
    MTLTexture *world_probe_tx = unwrap(params.world_probe_tx);
    MTLTexture *ray_time_tx = unwrap(params.ray_time_tx);
    MTLTexture *ray_radiance_tx = unwrap(params.ray_radiance_tx);
    MTLTexture *hit_albedo_tx = unwrap(params.hit_albedo_tx);
    MTLTexture *hit_throughput_tx = unwrap(params.hit_throughput_tx);
    MTLTexture *hit_material_tx = unwrap(params.hit_material_tx);
    MTLTexture *hit_normal_tx = unwrap(params.hit_normal_tx);
    MTLTexture *hit_position_tx = unwrap(params.hit_position_tx);
    MTLTexture *hit_world_position_tx = unwrap(params.hit_world_position_tx);
    MTLTexture *hit_identity_tx = unwrap(params.hit_identity_tx);
    MTLTexture *hit_barycentric_tx = unwrap(params.hit_barycentric_tx);
    MTLTexture *layered_receiver_ray_time_tx = unwrap(params.layered_receiver_ray_time_tx);
    MTLTexture *layered_receiver_ray_radiance_tx = unwrap(params.layered_receiver_ray_radiance_tx);
    MTLTexture *layered_receiver_albedo_tx = unwrap(params.layered_receiver_albedo_tx);
    MTLTexture *layered_receiver_throughput_tx = unwrap(params.layered_receiver_throughput_tx);
    MTLTexture *layered_receiver_material_tx = unwrap(params.layered_receiver_material_tx);
    MTLTexture *layered_receiver_normal_tx = unwrap(params.layered_receiver_normal_tx);
    MTLTexture *layered_receiver_position_tx = unwrap(params.layered_receiver_position_tx);
    MTLTexture *layered_receiver_world_position_tx = unwrap(params.layered_receiver_world_position_tx);
    MTLTexture *layered_receiver_identity_tx = unwrap(params.layered_receiver_identity_tx);
    MTLTexture *layered_receiver_barycentric_tx = unwrap(params.layered_receiver_barycentric_tx);
    MTLTexture *transmission_receiver_ray_time_tx = unwrap(params.transmission_receiver_ray_time_tx);
    MTLTexture *transmission_receiver_ray_radiance_tx = unwrap(
        params.transmission_receiver_ray_radiance_tx);
    MTLTexture *transmission_receiver_albedo_tx = unwrap(params.transmission_receiver_albedo_tx);
    MTLTexture *transmission_receiver_throughput_tx = unwrap(
        params.transmission_receiver_throughput_tx);
    MTLTexture *transmission_receiver_material_tx = unwrap(params.transmission_receiver_material_tx);
    MTLTexture *transmission_receiver_normal_tx = unwrap(params.transmission_receiver_normal_tx);
    MTLTexture *transmission_receiver_position_tx = unwrap(params.transmission_receiver_position_tx);
    MTLTexture *transmission_receiver_world_position_tx = unwrap(
        params.transmission_receiver_world_position_tx);
    MTLTexture *transmission_receiver_identity_tx = unwrap(params.transmission_receiver_identity_tx);
    MTLTexture *transmission_receiver_barycentric_tx = unwrap(
        params.transmission_receiver_barycentric_tx);
    MTLStorageBuf *dispatch_ssbo = static_cast<MTLStorageBuf *>(params.dispatch_buf);
    MTLStorageBuf *tiles_coord_ssbo = static_cast<MTLStorageBuf *>(params.tiles_coord_buf);
    if (ray_data_tx == nullptr || depth_tx == nullptr || gbuf_header_tx == nullptr ||
        gbuf_normal_tx == nullptr || screen_continuation_tx == nullptr ||
        ray_time_tx == nullptr || ray_radiance_tx == nullptr || hit_albedo_tx == nullptr ||
        hit_throughput_tx == nullptr ||
        hit_material_tx == nullptr || hit_normal_tx == nullptr || hit_position_tx == nullptr ||
        hit_world_position_tx == nullptr || hit_identity_tx == nullptr ||
        hit_barycentric_tx == nullptr || layered_receiver_ray_time_tx == nullptr ||
        layered_receiver_ray_radiance_tx == nullptr || layered_receiver_albedo_tx == nullptr ||
        layered_receiver_throughput_tx == nullptr || layered_receiver_material_tx == nullptr ||
        layered_receiver_normal_tx == nullptr || layered_receiver_position_tx == nullptr ||
        layered_receiver_world_position_tx == nullptr || layered_receiver_identity_tx == nullptr ||
        layered_receiver_barycentric_tx == nullptr ||
        transmission_receiver_ray_time_tx == nullptr ||
        transmission_receiver_ray_radiance_tx == nullptr ||
        transmission_receiver_albedo_tx == nullptr ||
        transmission_receiver_throughput_tx == nullptr ||
        transmission_receiver_material_tx == nullptr ||
        transmission_receiver_normal_tx == nullptr ||
        transmission_receiver_position_tx == nullptr ||
        transmission_receiver_world_position_tx == nullptr ||
        transmission_receiver_identity_tx == nullptr ||
        transmission_receiver_barycentric_tx == nullptr || dispatch_ssbo == nullptr ||
        tiles_coord_ssbo == nullptr)
    {
      return false;
    }

    HardwareTraceUniforms uniforms = {};
    uniforms.viewinv = params.viewinv;
    uniforms.wininv = params.wininv;
    uniforms.full_resolution = params.full_resolution;
    uniforms.resolution_scale = std::max(params.resolution_scale, 1);
    uniforms.resolution_scale_denominator = std::max(params.resolution_scale_denominator, 1);
    uniforms.closure_index = std::max(params.closure_index, 0);
    uniforms.feature_mask = params.feature_mask;
    uniforms.hardware_trace_phase = params.hardware_trace_phase;
    uniforms.reflection_bounces = std::clamp(
        params.reflection_bounces, 1, GPU_HARDWARE_RAYTRACE_SPECULAR_MAX_BOUNCES);
    uniforms.refraction_bounces = std::clamp(
        params.refraction_bounces, 1, GPU_HARDWARE_RAYTRACE_SPECULAR_MAX_BOUNCES);
    uniforms._pad0 = 0;
    uniforms.resolution_bias = params.resolution_bias;
    uniforms.clamp_indirect = std::max(params.clamp_indirect, 0.0f);
    uniforms.world_probe_atlas_coord = params.world_probe_atlas_coord;
    uniforms.use_environment_pad = int4((params.use_environment && world_probe_tx != nullptr) ? 1 : 0,
                                        std::max(scene->emissive_light_count, 0),
                                        std::max(params.gi_diffuse_sample_count, 1),
                                        (params.use_diffuse_environment && world_probe_tx != nullptr) ?
                                            1 :
                                            0);
    const int trace_local_lights = std::clamp(params.local_light_count, 0, params.light_count);
    uniforms.light_count_pad = int4(std::max(params.light_count, 0),
                                    std::max(params.light_sample_count, 0),
                                    trace_local_lights,
                                    (trace_local_lights > 0) ? (2 * trace_local_lights - 1) : 0);
    uniforms.sampling_rand = params.sampling_rand;
    uniforms.secondary_gi_pad = int4(params.secondary_gi ? 1 : 0,
                                     std::max(params.secondary_gi_samples, 1),
                                     (params.nis_enable && params.nis_weights_buf != nullptr) ? 1 : 0,
                                     0);

    MTLStorageBuf *light_ssbo = (params.light_buf != nullptr) ?
                                    static_cast<MTLStorageBuf *>(params.light_buf) :
                                    nullptr;
    id<MTLBuffer> light_handle = (light_ssbo != nullptr) ? light_ssbo->get_metal_buffer() : nil;

    const bool capture_started = begin_hardware_trace_capture(ctx->queue);

    id<MTLCommandBuffer> command_buffer = [ctx->queue commandBuffer];
    NSMutableArray *retained_resources = retained_resources_for_command_buffer(command_buffer,
                                                                               "Metal RT trace");
    retain_scene_resources(scene, retained_resources);
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      end_hardware_trace_capture(capture_started);
      return false;
    }

    [encoder setComputePipelineState:pipeline];
    encoder_use_scene_geometry_resources(encoder, scene);
    [encoder setAccelerationStructure:scene->top_level_acceleration_structure atBufferIndex:0];
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:1];
    encoder_use_scene_shading_resources(encoder, scene);
    id<MTLBuffer> dispatch_buf_handle = dispatch_ssbo->get_metal_buffer();
    id<MTLBuffer> tiles_coord_handle = tiles_coord_ssbo->get_metal_buffer();
    if (dispatch_buf_handle == nil || tiles_coord_handle == nil) {
      [encoder endEncoding];
      end_hardware_trace_capture(capture_started);
      return false;
    }
    [encoder useResource:dispatch_buf_handle usage:MTLResourceUsageRead];
    [encoder useResource:tiles_coord_handle usage:MTLResourceUsageRead];
    if (light_handle != nil) {
      [encoder useResource:light_handle usage:MTLResourceUsageRead];
    }
    [encoder setBuffer:scene->emissive_radiance_buffer offset:0 atIndex:2];
    [encoder setBuffer:scene->diffuse_albedo_buffer offset:0 atIndex:3];
    [encoder setBuffer:scene->material_proxy_buffer offset:0 atIndex:4];
    [encoder setBuffer:scene->triangle_normal_buffer offset:0 atIndex:5];
    [encoder setBuffer:scene->triangle_normal_range_buffer offset:0 atIndex:6];
    [encoder setBuffer:tiles_coord_handle offset:0 atIndex:7];
    [encoder setBuffer:scene->triangle_smooth_normal_buffer offset:0 atIndex:8];
    [encoder setBuffer:scene->triangle_local_position_buffer offset:0 atIndex:9];
    [encoder setBuffer:scene->emissive_light_buffer offset:0 atIndex:10];
    [encoder setBuffer:light_handle offset:0 atIndex:11];
    id<MTLBuffer> nis_weights_handle = nil;
    if (params.nis_weights_buf != nullptr) {
      nis_weights_handle =
          static_cast<MTLStorageBuf *>(params.nis_weights_buf)->get_metal_buffer();
    }
    if (nis_weights_handle == nil) {
      /* The kernels guard on the enable lane, but Metal requires a binding. */
      nis_weights_handle = scene->emissive_radiance_buffer;
    }
    [encoder useResource:nis_weights_handle usage:MTLResourceUsageRead];
    [encoder setBuffer:nis_weights_handle offset:0 atIndex:12];
    id<MTLTexture> ray_data_handle = ray_data_tx->get_metal_handle();
    id<MTLTexture> depth_handle = depth_tx->get_metal_handle();
    id<MTLTexture> gbuf_header_handle = gbuf_header_tx->get_metal_handle();
    id<MTLTexture> gbuf_normal_handle = gbuf_normal_tx->get_metal_handle();
    id<MTLTexture> screen_continuation_handle = screen_continuation_tx->get_metal_handle();
    id<MTLTexture> world_probe_handle = world_probe_tx != nullptr ?
                                             world_probe_tx->get_metal_handle() :
                                             nil;
    id<MTLTexture> ray_time_handle = ray_time_tx->get_metal_handle();
    id<MTLTexture> ray_radiance_handle = ray_radiance_tx->get_metal_handle();
    id<MTLTexture> hit_albedo_handle = hit_albedo_tx->get_metal_handle();
    id<MTLTexture> hit_throughput_handle = hit_throughput_tx->get_metal_handle();
    id<MTLTexture> hit_material_handle = hit_material_tx->get_metal_handle();
    id<MTLTexture> hit_normal_handle = hit_normal_tx->get_metal_handle();
    id<MTLTexture> hit_position_handle = hit_position_tx->get_metal_handle();
    id<MTLTexture> hit_world_position_handle = hit_world_position_tx->get_metal_handle();
    id<MTLTexture> hit_identity_handle = hit_identity_tx->get_metal_handle();
    id<MTLTexture> hit_barycentric_handle = hit_barycentric_tx->get_metal_handle();
    id<MTLTexture> layered_receiver_ray_time_handle =
        layered_receiver_ray_time_tx->get_metal_handle();
    id<MTLTexture> layered_receiver_ray_radiance_handle =
        layered_receiver_ray_radiance_tx->get_metal_handle();
    id<MTLTexture> layered_receiver_albedo_handle =
        layered_receiver_albedo_tx->get_metal_handle();
    id<MTLTexture> layered_receiver_throughput_handle =
        layered_receiver_throughput_tx->get_metal_handle();
    id<MTLTexture> layered_receiver_material_handle =
        layered_receiver_material_tx->get_metal_handle();
    id<MTLTexture> layered_receiver_normal_handle =
        layered_receiver_normal_tx->get_metal_handle();
    id<MTLTexture> layered_receiver_position_handle =
        layered_receiver_position_tx->get_metal_handle();
    id<MTLTexture> layered_receiver_world_position_handle =
        layered_receiver_world_position_tx->get_metal_handle();
    id<MTLTexture> layered_receiver_identity_handle =
        layered_receiver_identity_tx->get_metal_handle();
    id<MTLTexture> layered_receiver_barycentric_handle =
        layered_receiver_barycentric_tx->get_metal_handle();
    id<MTLTexture> transmission_receiver_ray_time_handle =
        transmission_receiver_ray_time_tx->get_metal_handle();
    id<MTLTexture> transmission_receiver_ray_radiance_handle =
        transmission_receiver_ray_radiance_tx->get_metal_handle();
    id<MTLTexture> transmission_receiver_albedo_handle =
        transmission_receiver_albedo_tx->get_metal_handle();
    id<MTLTexture> transmission_receiver_throughput_handle =
        transmission_receiver_throughput_tx->get_metal_handle();
    id<MTLTexture> transmission_receiver_material_handle =
        transmission_receiver_material_tx->get_metal_handle();
    id<MTLTexture> transmission_receiver_normal_handle =
        transmission_receiver_normal_tx->get_metal_handle();
    id<MTLTexture> transmission_receiver_position_handle =
        transmission_receiver_position_tx->get_metal_handle();
    id<MTLTexture> transmission_receiver_world_position_handle =
        transmission_receiver_world_position_tx->get_metal_handle();
    id<MTLTexture> transmission_receiver_identity_handle =
        transmission_receiver_identity_tx->get_metal_handle();
    id<MTLTexture> transmission_receiver_barycentric_handle =
        transmission_receiver_barycentric_tx->get_metal_handle();
    if (ray_data_handle == nil || depth_handle == nil || gbuf_header_handle == nil ||
        gbuf_normal_handle == nil || screen_continuation_handle == nil ||
        ray_time_handle == nil || ray_radiance_handle == nil || hit_albedo_handle == nil ||
        hit_throughput_handle == nil ||
        hit_material_handle == nil || hit_normal_handle == nil || hit_position_handle == nil ||
        hit_world_position_handle == nil || hit_identity_handle == nil ||
        hit_barycentric_handle == nil || layered_receiver_ray_time_handle == nil ||
        layered_receiver_ray_radiance_handle == nil || layered_receiver_albedo_handle == nil ||
        layered_receiver_throughput_handle == nil || layered_receiver_material_handle == nil ||
        layered_receiver_normal_handle == nil || layered_receiver_position_handle == nil ||
        layered_receiver_world_position_handle == nil || layered_receiver_identity_handle == nil ||
        layered_receiver_barycentric_handle == nil ||
        transmission_receiver_ray_time_handle == nil ||
        transmission_receiver_ray_radiance_handle == nil ||
        transmission_receiver_albedo_handle == nil ||
        transmission_receiver_throughput_handle == nil ||
        transmission_receiver_material_handle == nil ||
        transmission_receiver_normal_handle == nil ||
        transmission_receiver_position_handle == nil ||
        transmission_receiver_world_position_handle == nil ||
        transmission_receiver_identity_handle == nil ||
        transmission_receiver_barycentric_handle == nil)
    {
      [encoder endEncoding];
      end_hardware_trace_capture(capture_started);
      return false;
    }
    [encoder setTexture:ray_data_handle atIndex:0];
    [encoder setTexture:depth_handle atIndex:1];
    [encoder setTexture:gbuf_header_handle atIndex:2];
    [encoder setTexture:gbuf_normal_handle atIndex:3];
    [encoder setTexture:screen_continuation_handle atIndex:4];
    [encoder setTexture:ray_time_handle atIndex:5];
    [encoder setTexture:ray_radiance_handle atIndex:6];
    [encoder setTexture:hit_albedo_handle atIndex:7];
    [encoder setTexture:hit_material_handle atIndex:8];
    [encoder setTexture:hit_normal_handle atIndex:9];
    [encoder setTexture:hit_position_handle atIndex:10];
    [encoder setTexture:hit_identity_handle atIndex:11];
    [encoder setTexture:hit_barycentric_handle atIndex:12];
    [encoder setTexture:hit_world_position_handle atIndex:13];
    [encoder setTexture:hit_throughput_handle atIndex:14];
    [encoder setTexture:layered_receiver_ray_time_handle atIndex:15];
    [encoder setTexture:layered_receiver_ray_radiance_handle atIndex:16];
    [encoder setTexture:layered_receiver_albedo_handle atIndex:17];
    [encoder setTexture:layered_receiver_material_handle atIndex:18];
    [encoder setTexture:layered_receiver_normal_handle atIndex:19];
    [encoder setTexture:layered_receiver_position_handle atIndex:20];
    [encoder setTexture:layered_receiver_identity_handle atIndex:21];
    [encoder setTexture:layered_receiver_barycentric_handle atIndex:22];
    [encoder setTexture:layered_receiver_world_position_handle atIndex:23];
    [encoder setTexture:layered_receiver_throughput_handle atIndex:24];
    [encoder setTexture:transmission_receiver_ray_time_handle atIndex:25];
    [encoder setTexture:transmission_receiver_ray_radiance_handle atIndex:26];
    [encoder setTexture:transmission_receiver_albedo_handle atIndex:27];
    [encoder setTexture:transmission_receiver_material_handle atIndex:28];
    [encoder setTexture:transmission_receiver_normal_handle atIndex:29];
    [encoder setTexture:transmission_receiver_position_handle atIndex:30];
    [encoder setTexture:transmission_receiver_identity_handle atIndex:31];
    [encoder setTexture:transmission_receiver_barycentric_handle atIndex:32];
    [encoder setTexture:transmission_receiver_world_position_handle atIndex:33];
    [encoder setTexture:transmission_receiver_throughput_handle atIndex:34];
    [encoder setTexture:world_probe_handle atIndex:35];

    const MTLSize group_size = MTLSizeMake(8, 8, 1);
    [encoder dispatchThreadgroupsWithIndirectBuffer:dispatch_buf_handle
                               indirectBufferOffset:0
                              threadsPerThreadgroup:group_size];

    /* Nuru Secondary GI (Stage A revival): per-pixel receiver GI for mirror-visible diffuse
     * surfaces. Reads the hit payload the main kernel just wrote, traces a small dome per
     * receiver texel, and writes the dedicated receiver-GI texture consumed by the hit-lighting
     * pass. Dispatched over the same trace tiles. */
    if (params.secondary_gi) {
      id<MTLComputePipelineState> receiver_gi_pipeline = get_hardware_reflected_receiver_gi_pipeline(
          ctx->device);
      /* The receiver-GI kernel is payload-agnostic: run it once per receiver payload set
       * (main hit, layered/Principled-metallic, transmission). */
      struct ReceiverGIVariant {
        gpu::Texture *out_tx;
        id<MTLTexture> ray_time;
        id<MTLTexture> albedo;
        id<MTLTexture> normal;
        id<MTLTexture> world_position;
        id<MTLTexture> material;
      };
      const ReceiverGIVariant receiver_gi_variants[3] = {
          {params.reflected_receiver_gi_tx,
           ray_time_handle,
           hit_albedo_handle,
           hit_normal_handle,
           hit_world_position_handle,
           hit_material_handle},
          {params.layered_receiver_gi_tx,
           layered_receiver_ray_time_handle,
           layered_receiver_albedo_handle,
           layered_receiver_normal_handle,
           layered_receiver_world_position_handle,
           layered_receiver_material_handle},
          {params.transmission_receiver_gi_tx,
           transmission_receiver_ray_time_handle,
           transmission_receiver_albedo_handle,
           transmission_receiver_normal_handle,
           transmission_receiver_world_position_handle,
           transmission_receiver_material_handle},
      };
      bool receiver_gi_barrier_done = false;
      for (const ReceiverGIVariant &variant : receiver_gi_variants) {
        if (receiver_gi_pipeline == nil || variant.out_tx == nullptr) {
          continue;
        }
        MTLTexture *receiver_gi_tx = unwrap(variant.out_tx);
        id<MTLTexture> receiver_gi_handle = receiver_gi_tx != nullptr ?
                                                receiver_gi_tx->get_metal_handle() :
                                                nil;
        if (receiver_gi_handle == nil) {
          continue;
        }
        if (!receiver_gi_barrier_done) {
          [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];
          receiver_gi_barrier_done = true;
        }
        HardwareReflectedReceiverGIUniformsHost receiver_uniforms = {};
        receiver_uniforms.resolution_samples = int4(int([receiver_gi_handle width]),
                                                    int([receiver_gi_handle height]),
                                                    1,
                                                    std::max(params.secondary_gi_samples, 1));
        receiver_uniforms.normal_bias_pad = float4(5.0e-3f, 0.0f, 0.0f, 0.0f);
        receiver_uniforms.environment_pad = int4(
            (params.use_diffuse_environment && params.world_probe_tx != nullptr) ? 1 : 0,
            (params.nis_enable && params.nis_weights_buf != nullptr &&
             params.nis_feedback_buf != nullptr) ?
                1 :
                0,
            0,
            0);
        receiver_uniforms.light_count_pad = int4(
            std::max(params.light_count, 0),
            std::min(params.light_count > 0 ? 4 : 0, std::max(params.secondary_gi_samples, 1)),
            std::clamp(params.local_light_count, 0, params.light_count),
            (params.nis_enable && params.nis_weights_buf != nullptr) ? 1 : 0);
        receiver_uniforms.sampling_rand = params.sampling_rand;
        receiver_uniforms.world_probe_atlas_coord = params.world_probe_atlas_coord;
        [encoder setComputePipelineState:receiver_gi_pipeline];
        [encoder setAccelerationStructure:scene->top_level_acceleration_structure
                            atBufferIndex:0];
        [encoder setBytes:&receiver_uniforms length:sizeof(receiver_uniforms) atIndex:1];
        [encoder setBuffer:scene->emissive_radiance_buffer offset:0 atIndex:2];
        [encoder setBuffer:scene->diffuse_albedo_buffer offset:0 atIndex:3];
        [encoder setBuffer:scene->triangle_normal_buffer offset:0 atIndex:4];
        [encoder setBuffer:scene->triangle_normal_range_buffer offset:0 atIndex:5];
        [encoder setBuffer:light_handle offset:0 atIndex:6];
        [encoder setBuffer:tiles_coord_handle offset:0 atIndex:7];
        [encoder setBuffer:scene->material_proxy_buffer offset:0 atIndex:8];
        [encoder setBuffer:nis_weights_handle offset:0 atIndex:9];
        id<MTLBuffer> nis_feedback_handle = nil;
        if (params.nis_feedback_buf != nullptr) {
          nis_feedback_handle =
              static_cast<MTLStorageBuf *>(params.nis_feedback_buf)->get_metal_buffer();
        }
        if (nis_feedback_handle == nil) {
          nis_feedback_handle = scene->emissive_radiance_buffer;
        }
        [encoder useResource:nis_feedback_handle usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [encoder setBuffer:nis_feedback_handle offset:0 atIndex:10];
        [encoder setTexture:receiver_gi_handle atIndex:0];
        [encoder setTexture:world_probe_handle atIndex:1];
        [encoder setTexture:variant.ray_time atIndex:2];
        [encoder setTexture:variant.albedo atIndex:3];
        [encoder setTexture:variant.normal atIndex:4];
        [encoder setTexture:variant.world_position atIndex:5];
        [encoder setTexture:variant.material atIndex:6];
        [encoder dispatchThreadgroupsWithIndirectBuffer:dispatch_buf_handle
                                   indirectBufferOffset:0
                                  threadsPerThreadgroup:group_size];
      }
    }
    [encoder endEncoding];

    [command_buffer commit];
    const bool wait_for_completion = capture_started || env_flag_enabled("BLENDER_EEVEE_HWRT_FORCE_SYNC");
    if (wait_for_completion) {
      [command_buffer waitUntilCompleted];
      end_hardware_trace_capture(capture_started);
    }

    const bool success = wait_for_completion ?
                             (command_buffer.status == MTLCommandBufferStatusCompleted) :
                             true;
    if (wait_for_completion && !success) {
      fprintf(stderr, "Metal RT trace command failed with status=%ld\n", long(command_buffer.status));
    }
    if (success && wait_for_completion) {
      GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH | GPU_BARRIER_SHADER_IMAGE_ACCESS);
    }
    return success;
  }
#endif

  return false;
}

bool raytrace_scene_shadow_batch_begin(GPUHardwareRaytraceScene *scene)
{
  if (scene == nullptr) {
    return false;
  }
#if defined(MAC_OS_VERSION_14_0)
  if (@available(macos 14.0, *)) {
    MTLContext *ctx = MTLContext::get();
    if (ctx == nullptr || ctx->queue == nil) {
      return false;
    }
    return begin_shadow_trace_batch(ctx->queue, scene, "Metal RT shadow batch");
  }
#endif
  return false;
}

bool raytrace_scene_shadow_batch_end(GPUHardwareRaytraceScene *scene)
{
  if (scene == nullptr) {
    return false;
  }
#if defined(MAC_OS_VERSION_14_0)
  if (@available(macos 14.0, *)) {
    return commit_shadow_trace_batch(scene);
  }
#endif
  return false;
}

bool raytrace_scene_trace_directional_shadow(GPUHardwareRaytraceScene *scene,
                                             const GPUHardwareRaytraceDirectionalShadowParams &params)
{
  if (scene == nullptr || scene->top_level_acceleration_structure == nil || params.depth_tx == nullptr ||
      params.gbuf_header_tx == nullptr || params.gbuf_normal_tx == nullptr ||
      params.shadow_visibility_tx == nullptr)
  {
    return false;
  }

  if (!GPU_hardware_raytracing_support()) {
    return false;
  }

#if defined(MAC_OS_VERSION_14_0)
  if (@available(macos 14.0, *)) {
    MTLContext *ctx = MTLContext::get();
    if (ctx == nullptr || ctx->device == nil || ctx->queue == nil) {
      return false;
    }

    id<MTLComputePipelineState> pipeline = get_hardware_directional_shadow_pipeline(ctx->device);
    if (pipeline == nil) {
      return false;
    }

    MTLTexture *depth_tx = unwrap(params.depth_tx);
    MTLTexture *gbuf_header_tx = unwrap(params.gbuf_header_tx);
    MTLTexture *gbuf_normal_tx = unwrap(params.gbuf_normal_tx);
    MTLTexture *shadow_visibility_tx = unwrap(params.shadow_visibility_tx);
    MTLStorageBuf *world_sunlight_ssbo = static_cast<MTLStorageBuf *>(
        params.world_sunlight_direction_buf);
    if (depth_tx == nullptr || gbuf_header_tx == nullptr || gbuf_normal_tx == nullptr ||
        shadow_visibility_tx == nullptr)
    {
      return false;
    }

    HardwareShadowUniforms uniforms = {};
    uniforms.viewinv = params.viewinv;
    uniforms.wininv = params.wininv;
    uniforms.resolution_layer = int4(params.full_resolution.x,
                                     params.full_resolution.y,
                                     std::max(params.shadow_layer, 0),
                                     0);
    uniforms.light_direction_bias = float4(params.light_direction, std::max(params.normal_bias, 0.0f));
    /* Nuru: `shadow_params.z` = Color Intensity (0..1), `.w` = Photons intensity (0..10). */
    uniforms.shadow_params = float4(std::max(params.shadow_angle, 0.0f),
                                    float(std::max(params.sample_count, 1)),
                                    std::clamp(params.color_intensity, 0.0f, 1.0f),
                                    std::max(params.photons_intensity, 0.0f));
    /* Nuru: `.y` = caustic toggle, `.w` = transparent-shadow blend (float bits). */
    uniforms.world_sun_slot_pad = int4(params.world_sun_slot,
                                       params.use_caustics ? 1 : 0,
                                       0,
                                       gpu_metal_shadow_transparency_bits(params.shadow_transparency));
    uniforms.sampling_rand = params.sampling_rand;

    NSMutableArray *retained_resources = nil;
    bool uses_batch = false;
    id<MTLCommandBuffer> command_buffer = trace_command_buffer_for_shadow(
        scene, ctx->queue, "Metal RT directional shadow", &retained_resources, uses_batch);
    if (command_buffer == nil) {
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      cancel_shadow_trace_resources_if_needed(uses_batch, retained_resources);
      return false;
    }

    [encoder setComputePipelineState:pipeline];
    encoder_use_scene_geometry_resources(encoder, scene);
    /* Nuru: HWRT transparent shadows need the per-instance material proxy buffer at trace time
     * so refraction / alpha-blend hits can attenuate the ray instead of fully blocking it.
     * The normal buffers are bound for the optional caustic-shadow focus heuristic. */
    [encoder useResource:scene->material_proxy_buffer usage:MTLResourceUsageRead];
    [encoder useResource:scene->triangle_normal_buffer usage:MTLResourceUsageRead];
    [encoder useResource:scene->triangle_smooth_normal_buffer usage:MTLResourceUsageRead];
    [encoder useResource:scene->triangle_normal_range_buffer usage:MTLResourceUsageRead];
    [encoder setAccelerationStructure:scene->top_level_acceleration_structure atBufferIndex:0];
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:1];
    id<MTLBuffer> world_sunlight_handle = (world_sunlight_ssbo != nullptr) ?
                                              world_sunlight_ssbo->get_metal_buffer() :
                                              nil;
    if (params.world_sun_slot >= 0 && world_sunlight_handle == nil) {
      [encoder endEncoding];
      cancel_shadow_trace_resources_if_needed(uses_batch, retained_resources);
      return false;
    }
    if (world_sunlight_handle != nil) {
      [encoder useResource:world_sunlight_handle usage:MTLResourceUsageRead];
      retain_resource(retained_resources, world_sunlight_handle);
    }
    [encoder setBuffer:world_sunlight_handle offset:0 atIndex:2];
    [encoder setBuffer:scene->material_proxy_buffer offset:0 atIndex:3];
    [encoder setBuffer:scene->triangle_normal_buffer offset:0 atIndex:4];
    [encoder setBuffer:scene->triangle_smooth_normal_buffer offset:0 atIndex:5];
    [encoder setBuffer:scene->triangle_normal_range_buffer offset:0 atIndex:6];
    id<MTLTexture> depth_handle = depth_tx->get_metal_handle();
    id<MTLTexture> gbuf_header_handle = gbuf_header_tx->get_metal_handle();
    id<MTLTexture> gbuf_normal_handle = gbuf_normal_tx->get_metal_handle();
    id<MTLTexture> shadow_visibility_handle = shadow_visibility_tx->get_metal_handle();
    if (depth_handle == nil || gbuf_header_handle == nil || gbuf_normal_handle == nil ||
        shadow_visibility_handle == nil)
    {
      [encoder endEncoding];
      cancel_shadow_trace_resources_if_needed(uses_batch, retained_resources);
      return false;
    }
    [encoder setTexture:depth_handle atIndex:0];
    [encoder setTexture:gbuf_header_handle atIndex:1];
    [encoder setTexture:gbuf_normal_handle atIndex:2];
    [encoder setTexture:shadow_visibility_handle atIndex:3];

    const NSUInteger width = std::max<NSUInteger>(1, params.full_resolution.x);
    const NSUInteger height = std::max<NSUInteger>(1, params.full_resolution.y);
    const NSUInteger threads_x = 8;
    const NSUInteger threads_y = std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup /
                                                             threads_x);
    const MTLSize grid_size = MTLSizeMake(width, height, 1);
    const MTLSize group_size = MTLSizeMake(threads_x, std::min<NSUInteger>(8, threads_y), 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];
    return finish_shadow_trace_command_buffer(scene, command_buffer, uses_batch);
  }
#endif

  return false;
}

bool raytrace_scene_trace_directional_hit_shadow(
    GPUHardwareRaytraceScene *scene, const GPUHardwareRaytraceDirectionalHitShadowParams &params)
{
  if (scene == nullptr || scene->top_level_acceleration_structure == nil ||
      params.hit_normal_tx == nullptr || params.hit_world_position_tx == nullptr ||
      params.hit_identity_tx == nullptr ||
      params.shadow_visibility_tx == nullptr || params.dispatch_buf == nullptr ||
      params.tiles_coord_buf == nullptr)
  {
    return false;
  }

  if (!GPU_hardware_raytracing_support()) {
    return false;
  }

#if defined(MAC_OS_VERSION_14_0)
  if (@available(macos 14.0, *)) {
    MTLContext *ctx = MTLContext::get();
    if (ctx == nullptr || ctx->device == nil || ctx->queue == nil) {
      return false;
    }

    id<MTLComputePipelineState> pipeline = get_hardware_directional_hit_shadow_pipeline(
        ctx->device);
    if (pipeline == nil) {
      return false;
    }

    MTLTexture *hit_normal_tx = unwrap(params.hit_normal_tx);
    MTLTexture *hit_world_position_tx = unwrap(params.hit_world_position_tx);
    MTLTexture *hit_identity_tx = unwrap(params.hit_identity_tx);
    MTLTexture *shadow_visibility_tx = unwrap(params.shadow_visibility_tx);
    MTLStorageBuf *dispatch_ssbo = static_cast<MTLStorageBuf *>(params.dispatch_buf);
    MTLStorageBuf *tiles_coord_ssbo = static_cast<MTLStorageBuf *>(params.tiles_coord_buf);
    MTLStorageBuf *world_sunlight_ssbo = static_cast<MTLStorageBuf *>(
        params.world_sunlight_direction_buf);
    if (hit_normal_tx == nullptr || hit_world_position_tx == nullptr || hit_identity_tx == nullptr ||
        shadow_visibility_tx == nullptr ||
        dispatch_ssbo == nullptr || tiles_coord_ssbo == nullptr)
    {
      return false;
    }

    HardwareShadowUniforms uniforms = {};
    uniforms.resolution_layer = int4(params.tracing_resolution.x,
                                     params.tracing_resolution.y,
                                     std::max(params.shadow_layer, 0),
                                     0);
    uniforms.light_direction_bias = float4(params.light_direction, std::max(params.normal_bias, 0.0f));
    uniforms.shadow_params = float4(std::max(params.shadow_angle, 0.0f),
                                    float(std::max(params.sample_count, 1)),
                                    std::clamp(params.color_intensity, 0.0f, 1.0f),
                                    std::max(params.photons_intensity, 0.0f));
    uniforms.world_sun_slot_pad = int4(params.world_sun_slot,
                                       params.use_caustics ? 1 : 0,
                                       0,
                                       gpu_metal_shadow_transparency_bits(params.shadow_transparency));
    uniforms.sampling_rand = params.sampling_rand;

    NSMutableArray *retained_resources = nil;
    bool uses_batch = false;
    id<MTLCommandBuffer> command_buffer = trace_command_buffer_for_shadow(
        scene, ctx->queue, "Metal RT directional hit shadow", &retained_resources, uses_batch);
    if (command_buffer == nil) {
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      cancel_shadow_trace_resources_if_needed(uses_batch, retained_resources);
      return false;
    }

    [encoder setComputePipelineState:pipeline];
    encoder_use_scene_geometry_resources(encoder, scene);
    [encoder useResource:scene->triangle_normal_buffer usage:MTLResourceUsageRead];
    [encoder useResource:scene->triangle_smooth_normal_buffer usage:MTLResourceUsageRead];
    [encoder useResource:scene->triangle_normal_range_buffer usage:MTLResourceUsageRead];
    /* Nuru: HWRT transparent shadows (smooth normal buffer also needed for caustic focus). */
    [encoder useResource:scene->material_proxy_buffer usage:MTLResourceUsageRead];
    [encoder setAccelerationStructure:scene->top_level_acceleration_structure atBufferIndex:0];
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:1];
    id<MTLBuffer> world_sunlight_handle = (world_sunlight_ssbo != nullptr) ?
                                              world_sunlight_ssbo->get_metal_buffer() :
                                              nil;
    if (params.world_sun_slot >= 0 && world_sunlight_handle == nil) {
      [encoder endEncoding];
      cancel_shadow_trace_resources_if_needed(uses_batch, retained_resources);
      return false;
    }
    if (world_sunlight_handle != nil) {
      [encoder useResource:world_sunlight_handle usage:MTLResourceUsageRead];
      retain_resource(retained_resources, world_sunlight_handle);
    }
    [encoder setBuffer:world_sunlight_handle offset:0 atIndex:2];
    id<MTLBuffer> dispatch_buf_handle = dispatch_ssbo->get_metal_buffer();
    id<MTLBuffer> tiles_coord_handle = tiles_coord_ssbo->get_metal_buffer();
    if (dispatch_buf_handle == nil || tiles_coord_handle == nil) {
      [encoder endEncoding];
      cancel_shadow_trace_resources_if_needed(uses_batch, retained_resources);
      return false;
    }
    [encoder useResource:dispatch_buf_handle usage:MTLResourceUsageRead];
    [encoder useResource:tiles_coord_handle usage:MTLResourceUsageRead];
    [encoder setBuffer:tiles_coord_handle offset:0 atIndex:3];
    [encoder setBuffer:scene->triangle_normal_buffer offset:0 atIndex:4];
    [encoder setBuffer:scene->triangle_normal_range_buffer offset:0 atIndex:5];
    [encoder setBuffer:scene->material_proxy_buffer offset:0 atIndex:6];
    [encoder setBuffer:scene->triangle_smooth_normal_buffer offset:0 atIndex:7];
    id<MTLTexture> hit_normal_handle = hit_normal_tx->get_metal_handle();
    id<MTLTexture> hit_world_position_handle = hit_world_position_tx->get_metal_handle();
    id<MTLTexture> hit_identity_handle = hit_identity_tx->get_metal_handle();
    id<MTLTexture> shadow_visibility_handle = shadow_visibility_tx->get_metal_handle();
    if (hit_normal_handle == nil || hit_world_position_handle == nil || hit_identity_handle == nil ||
        shadow_visibility_handle == nil)
    {
      [encoder endEncoding];
      cancel_shadow_trace_resources_if_needed(uses_batch, retained_resources);
      return false;
    }
    [encoder setTexture:hit_normal_handle atIndex:0];
    [encoder setTexture:hit_world_position_handle atIndex:1];
    [encoder setTexture:hit_identity_handle atIndex:2];
    [encoder setTexture:shadow_visibility_handle atIndex:3];

    const MTLSize group_size = MTLSizeMake(8, 8, 1);
    [encoder dispatchThreadgroupsWithIndirectBuffer:dispatch_buf_handle
                               indirectBufferOffset:0
                              threadsPerThreadgroup:group_size];
    [encoder endEncoding];
    return finish_shadow_trace_command_buffer(scene, command_buffer, uses_batch);
  }
#endif

  return false;
}

bool raytrace_scene_trace_environment_visibility(
    GPUHardwareRaytraceScene *scene, const GPUHardwareRaytraceEnvironmentVisibilityParams &params)
{
  if (scene == nullptr || scene->top_level_acceleration_structure == nil || params.depth_tx == nullptr ||
      params.gbuf_header_tx == nullptr || params.gbuf_normal_tx == nullptr ||
      params.environment_visibility_tx == nullptr)
  {
    return false;
  }

  if (!GPU_hardware_raytracing_support()) {
    return false;
  }

#if defined(MAC_OS_VERSION_14_0)
  if (@available(macos 14.0, *)) {
    MTLContext *ctx = MTLContext::get();
    if (ctx == nullptr || ctx->device == nil || ctx->queue == nil) {
      return false;
    }

    id<MTLComputePipelineState> pipeline = get_hardware_environment_visibility_pipeline(
        ctx->device);
    if (pipeline == nil) {
      return false;
    }

    MTLTexture *depth_tx = unwrap(params.depth_tx);
    MTLTexture *gbuf_header_tx = unwrap(params.gbuf_header_tx);
    MTLTexture *gbuf_normal_tx = unwrap(params.gbuf_normal_tx);
    MTLTexture *environment_visibility_tx = unwrap(params.environment_visibility_tx);
    if (depth_tx == nullptr || gbuf_header_tx == nullptr || gbuf_normal_tx == nullptr ||
        environment_visibility_tx == nullptr)
    {
      return false;
    }

    HardwareEnvironmentVisibilityUniforms uniforms = {};
    uniforms.viewinv = params.viewinv;
    uniforms.wininv = params.wininv;
    uniforms.resolution_samples = int4(params.full_resolution.x,
                                       params.full_resolution.y,
                                       std::max(params.sample_count, 1),
                                       0);
    uniforms.normal_bias_pad = float4(
        std::max(params.normal_bias, 0.0f), float(std::max(params.sample_count, 1)), 0.0f, 0.0f);
    uniforms.sampling_rand = params.sampling_rand;

    id<MTLCommandBuffer> command_buffer = [ctx->queue commandBuffer];
    NSMutableArray *retained_resources = retained_resources_for_command_buffer(
        command_buffer, "Metal RT environment visibility");
    retain_scene_resources(scene, retained_resources);
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return false;
    }

    [encoder setComputePipelineState:pipeline];
    encoder_use_scene_geometry_resources(encoder, scene);
    [encoder setAccelerationStructure:scene->top_level_acceleration_structure atBufferIndex:0];
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:1];
    id<MTLTexture> depth_handle = depth_tx->get_metal_handle();
    id<MTLTexture> gbuf_header_handle = gbuf_header_tx->get_metal_handle();
    id<MTLTexture> gbuf_normal_handle = gbuf_normal_tx->get_metal_handle();
    id<MTLTexture> environment_visibility_handle = environment_visibility_tx->get_metal_handle();
    if (depth_handle == nil || gbuf_header_handle == nil || gbuf_normal_handle == nil ||
        environment_visibility_handle == nil)
    {
      [encoder endEncoding];
      return false;
    }
    [encoder setTexture:depth_handle atIndex:0];
    [encoder setTexture:gbuf_header_handle atIndex:1];
    [encoder setTexture:gbuf_normal_handle atIndex:2];
    [encoder setTexture:environment_visibility_handle atIndex:3];

    const NSUInteger width = std::max<NSUInteger>(1, params.full_resolution.x);
    const NSUInteger height = std::max<NSUInteger>(1, params.full_resolution.y);
    const NSUInteger threads_x = 8;
    const NSUInteger threads_y = std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup /
                                                             threads_x);
    const MTLSize grid_size = MTLSizeMake(width, height, 1);
    const MTLSize group_size = MTLSizeMake(threads_x, std::min<NSUInteger>(8, threads_y), 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];

    [command_buffer commit];
    const bool wait_for_completion = env_flag_enabled("BLENDER_EEVEE_HWRT_FORCE_SYNC");
    if (wait_for_completion) {
      [command_buffer waitUntilCompleted];
    }

    const bool success = wait_for_completion ?
                             (command_buffer.status == MTLCommandBufferStatusCompleted) :
                             true;
    if (success && wait_for_completion) {
      GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH | GPU_BARRIER_SHADER_IMAGE_ACCESS);
    }
    return success;
  }
#endif

  return false;
}

bool raytrace_scene_trace_hit_environment_visibility(
    GPUHardwareRaytraceScene *scene, const GPUHardwareRaytraceHitEnvironmentVisibilityParams &params)
{
  if (scene == nullptr || scene->top_level_acceleration_structure == nil ||
      params.hit_normal_tx == nullptr || params.hit_world_position_tx == nullptr ||
      params.environment_visibility_tx == nullptr || params.dispatch_buf == nullptr ||
      params.tiles_coord_buf == nullptr)
  {
    return false;
  }

  if (!GPU_hardware_raytracing_support()) {
    return false;
  }

#if defined(MAC_OS_VERSION_14_0)
  if (@available(macos 14.0, *)) {
    MTLContext *ctx = MTLContext::get();
    if (ctx == nullptr || ctx->device == nil || ctx->queue == nil) {
      return false;
    }

    id<MTLComputePipelineState> pipeline = get_hardware_hit_environment_visibility_pipeline(
        ctx->device);
    if (pipeline == nil) {
      return false;
    }

    MTLTexture *hit_normal_tx = unwrap(params.hit_normal_tx);
    MTLTexture *hit_world_position_tx = unwrap(params.hit_world_position_tx);
    MTLTexture *environment_visibility_tx = unwrap(params.environment_visibility_tx);
    MTLStorageBuf *dispatch_ssbo = static_cast<MTLStorageBuf *>(params.dispatch_buf);
    MTLStorageBuf *tiles_coord_ssbo = static_cast<MTLStorageBuf *>(params.tiles_coord_buf);
    if (hit_normal_tx == nullptr || hit_world_position_tx == nullptr ||
        environment_visibility_tx == nullptr || dispatch_ssbo == nullptr ||
        tiles_coord_ssbo == nullptr)
    {
      return false;
    }

    HardwareEnvironmentVisibilityUniforms uniforms = {};
    uniforms.resolution_samples = int4(params.tracing_resolution.x,
                                       params.tracing_resolution.y,
                                       std::max(params.sample_count, 1),
                                       0);
    uniforms.normal_bias_pad = float4(
        std::max(params.normal_bias, 0.0f), float(std::max(params.sample_count, 1)), 0.0f, 0.0f);
    uniforms.sampling_rand = params.sampling_rand;

    id<MTLCommandBuffer> command_buffer = [ctx->queue commandBuffer];
    NSMutableArray *retained_resources = retained_resources_for_command_buffer(
        command_buffer, "Metal RT hit environment visibility");
    retain_scene_resources(scene, retained_resources);
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return false;
    }

    [encoder setComputePipelineState:pipeline];
    encoder_use_scene_geometry_resources(encoder, scene);
    [encoder setAccelerationStructure:scene->top_level_acceleration_structure atBufferIndex:0];
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:1];
    id<MTLBuffer> dispatch_buf_handle = dispatch_ssbo->get_metal_buffer();
    id<MTLBuffer> tiles_coord_handle = tiles_coord_ssbo->get_metal_buffer();
    if (dispatch_buf_handle == nil || tiles_coord_handle == nil) {
      [encoder endEncoding];
      return false;
    }
    [encoder useResource:dispatch_buf_handle usage:MTLResourceUsageRead];
    [encoder useResource:tiles_coord_handle usage:MTLResourceUsageRead];
    [encoder setBuffer:tiles_coord_handle offset:0 atIndex:2];
    id<MTLTexture> hit_normal_handle = hit_normal_tx->get_metal_handle();
    id<MTLTexture> hit_world_position_handle = hit_world_position_tx->get_metal_handle();
    id<MTLTexture> environment_visibility_handle = environment_visibility_tx->get_metal_handle();
    if (hit_normal_handle == nil || hit_world_position_handle == nil ||
        environment_visibility_handle == nil)
    {
      [encoder endEncoding];
      return false;
    }
    [encoder setTexture:hit_normal_handle atIndex:0];
    [encoder setTexture:hit_world_position_handle atIndex:1];
    [encoder setTexture:environment_visibility_handle atIndex:2];

    const MTLSize group_size = MTLSizeMake(8, 8, 1);
    [encoder dispatchThreadgroupsWithIndirectBuffer:dispatch_buf_handle
                               indirectBufferOffset:0
                              threadsPerThreadgroup:group_size];
    [encoder endEncoding];

    [command_buffer commit];
    retain_resource(retained_resources, command_buffer);
    return true;
  }
#endif

  return false;
}

bool raytrace_scene_trace_local_shadow(GPUHardwareRaytraceScene *scene,
                                       const GPUHardwareRaytraceLocalShadowParams &params)
{
  if (scene == nullptr || scene->top_level_acceleration_structure == nil || params.depth_tx == nullptr ||
      params.gbuf_header_tx == nullptr || params.gbuf_normal_tx == nullptr ||
      params.shadow_visibility_tx == nullptr)
  {
    return false;
  }

  if (!GPU_hardware_raytracing_support()) {
    return false;
  }

#if defined(MAC_OS_VERSION_14_0)
  if (@available(macos 14.0, *)) {
    MTLContext *ctx = MTLContext::get();
    if (ctx == nullptr || ctx->device == nil || ctx->queue == nil) {
      return false;
    }

    id<MTLComputePipelineState> pipeline = get_hardware_local_shadow_pipeline(ctx->device);
    if (pipeline == nil) {
      return false;
    }

    MTLTexture *depth_tx = unwrap(params.depth_tx);
    MTLTexture *gbuf_header_tx = unwrap(params.gbuf_header_tx);
    MTLTexture *gbuf_normal_tx = unwrap(params.gbuf_normal_tx);
    MTLTexture *shadow_visibility_tx = unwrap(params.shadow_visibility_tx);
    if (depth_tx == nullptr || gbuf_header_tx == nullptr || gbuf_normal_tx == nullptr ||
        shadow_visibility_tx == nullptr)
    {
      return false;
    }

    HardwareLocalShadowUniforms uniforms = {};
    uniforms.viewinv = params.viewinv;
    uniforms.wininv = params.wininv;
    uniforms.resolution_layer_type = int4(params.full_resolution.x,
                                          params.full_resolution.y,
                                          std::max(params.shadow_layer, 0),
                                          int(params.light_type));
    uniforms.light_position_radius = float4(params.light_position, std::max(params.shadow_radius, 0.0f));
    uniforms.light_x_axis_size_x = float4(params.light_x_axis, std::max(params.area_size_x, 0.0f));
    uniforms.light_y_axis_size_y = float4(params.light_y_axis, std::max(params.area_size_y, 0.0f));
    uniforms.shadow_offset_scale = float4(params.shadow_offset, std::max(params.area_shadow_scale, 0.0f));
    /* Nuru: `normal_bias_pad.z` carries the caustic-shadow toggle (0/1), `.w` the Color
     * Intensity slider (0..1). `caustic_params.x` carries the Photons intensity (0..10). */
    uniforms.normal_bias_pad = float4(std::max(params.normal_bias, 0.0f),
                                      float(std::max(params.sample_count, 1)),
                                      params.use_caustics ? 1.0f : 0.0f,
                                      std::clamp(params.color_intensity, 0.0f, 1.0f));
    uniforms.sampling_rand = params.sampling_rand;
    uniforms.caustic_params = float4(std::max(params.photons_intensity, 0.0f),
                                     std::clamp(params.shadow_transparency, 0.0f, 1.0f),
                                     0.0f,
                                     0.0f);

    NSMutableArray *retained_resources = nil;
    bool uses_batch = false;
    id<MTLCommandBuffer> command_buffer = trace_command_buffer_for_shadow(
        scene, ctx->queue, "Metal RT local shadow", &retained_resources, uses_batch);
    if (command_buffer == nil) {
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      cancel_shadow_trace_resources_if_needed(uses_batch, retained_resources);
      return false;
    }

    [encoder setComputePipelineState:pipeline];
    encoder_use_scene_geometry_resources(encoder, scene);
    /* Nuru: HWRT transparent shadows + caustic focus need the proxy + normal buffers. */
    [encoder useResource:scene->material_proxy_buffer usage:MTLResourceUsageRead];
    [encoder useResource:scene->triangle_normal_buffer usage:MTLResourceUsageRead];
    [encoder useResource:scene->triangle_smooth_normal_buffer usage:MTLResourceUsageRead];
    [encoder useResource:scene->triangle_normal_range_buffer usage:MTLResourceUsageRead];
    [encoder setAccelerationStructure:scene->top_level_acceleration_structure atBufferIndex:0];
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:1];
    [encoder setBuffer:scene->material_proxy_buffer offset:0 atIndex:2];
    [encoder setBuffer:scene->triangle_normal_buffer offset:0 atIndex:3];
    [encoder setBuffer:scene->triangle_smooth_normal_buffer offset:0 atIndex:4];
    [encoder setBuffer:scene->triangle_normal_range_buffer offset:0 atIndex:5];
    id<MTLTexture> depth_handle = depth_tx->get_metal_handle();
    id<MTLTexture> gbuf_header_handle = gbuf_header_tx->get_metal_handle();
    id<MTLTexture> gbuf_normal_handle = gbuf_normal_tx->get_metal_handle();
    id<MTLTexture> shadow_visibility_handle = shadow_visibility_tx->get_metal_handle();
    if (depth_handle == nil || gbuf_header_handle == nil || gbuf_normal_handle == nil ||
        shadow_visibility_handle == nil)
    {
      [encoder endEncoding];
      cancel_shadow_trace_resources_if_needed(uses_batch, retained_resources);
      return false;
    }
    [encoder setTexture:depth_handle atIndex:0];
    [encoder setTexture:gbuf_header_handle atIndex:1];
    [encoder setTexture:gbuf_normal_handle atIndex:2];
    [encoder setTexture:shadow_visibility_handle atIndex:3];

    const NSUInteger width = std::max<NSUInteger>(1, params.full_resolution.x);
    const NSUInteger height = std::max<NSUInteger>(1, params.full_resolution.y);
    const NSUInteger threads_x = 8;
    const NSUInteger threads_y = std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup /
                                                             threads_x);
    const MTLSize grid_size = MTLSizeMake(width, height, 1);
    const MTLSize group_size = MTLSizeMake(threads_x, std::min<NSUInteger>(8, threads_y), 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];
    return finish_shadow_trace_command_buffer(scene, command_buffer, uses_batch);
  }
#endif

  return false;
}

bool raytrace_scene_trace_local_hit_shadow(GPUHardwareRaytraceScene *scene,
                                           const GPUHardwareRaytraceLocalHitShadowParams &params)
{
  if (scene == nullptr || scene->top_level_acceleration_structure == nil ||
      params.hit_normal_tx == nullptr || params.hit_world_position_tx == nullptr ||
      params.hit_identity_tx == nullptr ||
      params.shadow_visibility_tx == nullptr || params.dispatch_buf == nullptr ||
      params.tiles_coord_buf == nullptr)
  {
    return false;
  }

  if (!GPU_hardware_raytracing_support()) {
    return false;
  }

#if defined(MAC_OS_VERSION_14_0)
  if (@available(macos 14.0, *)) {
    MTLContext *ctx = MTLContext::get();
    if (ctx == nullptr || ctx->device == nil || ctx->queue == nil) {
      return false;
    }

    id<MTLComputePipelineState> pipeline = get_hardware_local_hit_shadow_pipeline(ctx->device);
    if (pipeline == nil) {
      return false;
    }

    MTLTexture *hit_normal_tx = unwrap(params.hit_normal_tx);
    MTLTexture *hit_world_position_tx = unwrap(params.hit_world_position_tx);
    MTLTexture *hit_identity_tx = unwrap(params.hit_identity_tx);
    MTLTexture *shadow_visibility_tx = unwrap(params.shadow_visibility_tx);
    MTLStorageBuf *dispatch_ssbo = static_cast<MTLStorageBuf *>(params.dispatch_buf);
    MTLStorageBuf *tiles_coord_ssbo = static_cast<MTLStorageBuf *>(params.tiles_coord_buf);
    if (hit_normal_tx == nullptr || hit_world_position_tx == nullptr || hit_identity_tx == nullptr ||
        shadow_visibility_tx == nullptr ||
        dispatch_ssbo == nullptr || tiles_coord_ssbo == nullptr)
    {
      return false;
    }

    HardwareLocalShadowUniforms uniforms = {};
    uniforms.resolution_layer_type = int4(params.tracing_resolution.x,
                                          params.tracing_resolution.y,
                                          std::max(params.shadow_layer, 0),
                                          int(params.light_type));
    uniforms.light_position_radius = float4(params.light_position, std::max(params.shadow_radius, 0.0f));
    uniforms.light_x_axis_size_x = float4(params.light_x_axis, std::max(params.area_size_x, 0.0f));
    uniforms.light_y_axis_size_y = float4(params.light_y_axis, std::max(params.area_size_y, 0.0f));
    uniforms.shadow_offset_scale = float4(params.shadow_offset, std::max(params.area_shadow_scale, 0.0f));
    uniforms.normal_bias_pad = float4(std::max(params.normal_bias, 0.0f),
                                      float(std::max(params.sample_count, 1)),
                                      params.use_caustics ? 1.0f : 0.0f,
                                      std::clamp(params.color_intensity, 0.0f, 1.0f));
    uniforms.sampling_rand = params.sampling_rand;
    uniforms.caustic_params = float4(std::max(params.photons_intensity, 0.0f),
                                     std::clamp(params.shadow_transparency, 0.0f, 1.0f),
                                     0.0f,
                                     0.0f);

    NSMutableArray *retained_resources = nil;
    bool uses_batch = false;
    id<MTLCommandBuffer> command_buffer = trace_command_buffer_for_shadow(
        scene, ctx->queue, "Metal RT local hit shadow", &retained_resources, uses_batch);
    if (command_buffer == nil) {
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      cancel_shadow_trace_resources_if_needed(uses_batch, retained_resources);
      return false;
    }

    [encoder setComputePipelineState:pipeline];
    encoder_use_scene_geometry_resources(encoder, scene);
    [encoder useResource:scene->triangle_normal_buffer usage:MTLResourceUsageRead];
    [encoder useResource:scene->triangle_smooth_normal_buffer usage:MTLResourceUsageRead];
    [encoder useResource:scene->triangle_normal_range_buffer usage:MTLResourceUsageRead];
    /* Nuru: HWRT transparent shadows (smooth normals also needed for caustic focus). */
    [encoder useResource:scene->material_proxy_buffer usage:MTLResourceUsageRead];
    [encoder setAccelerationStructure:scene->top_level_acceleration_structure atBufferIndex:0];
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:1];
    id<MTLBuffer> dispatch_buf_handle = dispatch_ssbo->get_metal_buffer();
    id<MTLBuffer> tiles_coord_handle = tiles_coord_ssbo->get_metal_buffer();
    if (dispatch_buf_handle == nil || tiles_coord_handle == nil) {
      [encoder endEncoding];
      cancel_shadow_trace_resources_if_needed(uses_batch, retained_resources);
      return false;
    }
    [encoder useResource:dispatch_buf_handle usage:MTLResourceUsageRead];
    [encoder useResource:tiles_coord_handle usage:MTLResourceUsageRead];
    [encoder setBuffer:tiles_coord_handle offset:0 atIndex:2];
    [encoder setBuffer:scene->triangle_normal_buffer offset:0 atIndex:3];
    [encoder setBuffer:scene->triangle_normal_range_buffer offset:0 atIndex:4];
    [encoder setBuffer:scene->material_proxy_buffer offset:0 atIndex:5];
    [encoder setBuffer:scene->triangle_smooth_normal_buffer offset:0 atIndex:6];
    id<MTLTexture> hit_normal_handle = hit_normal_tx->get_metal_handle();
    id<MTLTexture> hit_world_position_handle = hit_world_position_tx->get_metal_handle();
    id<MTLTexture> hit_identity_handle = hit_identity_tx->get_metal_handle();
    id<MTLTexture> shadow_visibility_handle = shadow_visibility_tx->get_metal_handle();
    if (hit_normal_handle == nil || hit_world_position_handle == nil || hit_identity_handle == nil ||
        shadow_visibility_handle == nil)
    {
      [encoder endEncoding];
      cancel_shadow_trace_resources_if_needed(uses_batch, retained_resources);
      return false;
    }
    [encoder setTexture:hit_normal_handle atIndex:0];
    [encoder setTexture:hit_world_position_handle atIndex:1];
    [encoder setTexture:hit_identity_handle atIndex:2];
    [encoder setTexture:shadow_visibility_handle atIndex:3];

    const MTLSize group_size = MTLSizeMake(8, 8, 1);
    [encoder dispatchThreadgroupsWithIndirectBuffer:dispatch_buf_handle
                               indirectBufferOffset:0
                              threadsPerThreadgroup:group_size];
    [encoder endEncoding];
    return finish_shadow_trace_command_buffer(scene, command_buffer, uses_batch);
  }
#endif

  return false;
}

bool raytrace_denoise_oidn(const GPUHardwareRaytraceOIDNDenoiseParams &params)
{
#ifndef WITH_OPENIMAGEDENOISE
  (void)params;
  return false;
#else
  if (params.input_radiance_tx == nullptr || params.output_radiance_tx == nullptr ||
      params.extent.x <= 0 || params.extent.y <= 0)
  {
    return false;
  }

  MTLContext *ctx = MTLContext::get();
  if (ctx == nullptr || ctx->device == nil || ctx->queue == nil) {
    return false;
  }

  MTLTexture *input_tx = unwrap(params.input_radiance_tx);
  MTLTexture *output_tx = unwrap(params.output_radiance_tx);
  MTLTexture *albedo_tx = unwrap(params.albedo_tx);
  MTLTexture *normal_tx = unwrap(params.normal_tx);
  if (input_tx == nullptr || output_tx == nullptr) {
    return false;
  }

  id<MTLTexture> input_handle = input_tx->get_metal_handle();
  id<MTLTexture> output_handle = output_tx->get_metal_handle();
  id<MTLTexture> albedo_handle = albedo_tx != nullptr ? albedo_tx->get_metal_handle() : nil;
  id<MTLTexture> normal_handle = normal_tx != nullptr ? normal_tx->get_metal_handle() : nil;
  const bool use_albedo = params.use_albedo && albedo_handle != nil;
  const bool use_normal = params.use_normal && normal_handle != nil;
  if (input_handle == nil || output_handle == nil) {
    return false;
  }
  if ([input_handle width] != NSUInteger(params.extent.x) ||
      [input_handle height] != NSUInteger(params.extent.y) ||
      [output_handle width] != NSUInteger(params.extent.x) ||
      [output_handle height] != NSUInteger(params.extent.y))
  {
    return false;
  }
  if (use_albedo && ([albedo_handle width] != NSUInteger(params.extent.x) ||
                     [albedo_handle height] != NSUInteger(params.extent.y)))
  {
    return false;
  }
  if (use_normal && ([normal_handle width] != NSUInteger(params.extent.x) ||
                     [normal_handle height] != NSUInteger(params.extent.y)))
  {
    return false;
  }

  id<MTLComputePipelineState> pack_pipeline = get_oidn_pack_pipeline(ctx->device);
  id<MTLComputePipelineState> unpack_pipeline = get_oidn_unpack_pipeline(ctx->device);
  if (pack_pipeline == nil || unpack_pipeline == nil) {
    return false;
  }

  const bool use_gpu = params.use_gpu;
  OIDNInteropCache &cache = oidn_interop_cache();
  if (!ensure_oidn_device(cache, ctx->device, ctx->queue, use_gpu) ||
      !ensure_oidn_filter(
          cache, use_albedo, use_normal, params.quality, params.prefilter))
  {
    return false;
  }

  const NSUInteger buffer_size = NSUInteger(params.extent.x) * NSUInteger(params.extent.y) *
                                 NSUInteger(sizeof(float) * 3);
  if (!ensure_oidn_buffer(
          ctx->device, cache.color_buffer, buffer_size, "Eevee OIDN color", !use_gpu) ||
      !ensure_oidn_buffer(
          ctx->device, cache.output_buffer, buffer_size, "Eevee OIDN output", !use_gpu))
  {
    return false;
  }
  if (use_albedo &&
      !ensure_oidn_buffer(
          ctx->device, cache.albedo_buffer, buffer_size, "Eevee OIDN albedo", !use_gpu))
  {
    return false;
  }
  if (use_normal &&
      !ensure_oidn_buffer(
          ctx->device, cache.normal_buffer, buffer_size, "Eevee OIDN normal", !use_gpu))
  {
    return false;
  }

  const bool wait_for_pack = true;
  const bool wait_for_unpack = true;

  const bool perf_logging_enabled = oidn_perf_logging_enabled();
  const double pack_start_time = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  if (!run_oidn_interop_pack(ctx->queue,
                             pack_pipeline,
                             input_handle,
                             albedo_handle,
                             normal_handle,
                             cache,
                             params.extent,
                             use_albedo,
                             use_normal,
                             wait_for_pack))
  {
    return false;
  }
  const double pack_ms = perf_logging_enabled ?
                             (BLI_time_now_seconds() - pack_start_time) * 1000.0 :
                             0.0;

  const size_t width = size_t(params.extent.x);
  const size_t height = size_t(params.extent.y);
  if (use_gpu) {
    if (!set_oidn_image(cache.device, cache.filter, "color", cache.color_buffer, width, height) ||
        !set_oidn_image(cache.device, cache.filter, "output", cache.output_buffer, width, height))
    {
      return false;
    }
    if (use_albedo &&
        !set_oidn_image(cache.device, cache.filter, "albedo", cache.albedo_buffer, width, height))
    {
      return false;
    }
    if (use_normal &&
        !set_oidn_image(cache.device, cache.filter, "normal", cache.normal_buffer, width, height))
    {
      return false;
    }
  }
  else {
    if (!set_oidn_cpu_image(cache.filter, "color", cache.color_buffer, width, height) ||
        !set_oidn_cpu_image(cache.filter, "output", cache.output_buffer, width, height))
    {
      return false;
    }
    if (use_albedo &&
        !set_oidn_cpu_image(cache.filter, "albedo", cache.albedo_buffer, width, height))
    {
      return false;
    }
    if (use_normal &&
        !set_oidn_cpu_image(cache.filter, "normal", cache.normal_buffer, width, height))
    {
      return false;
    }
  }

  oidnCommitFilter(cache.filter);
  if (oidn_report_error(cache.device, "filter commit")) {
    return false;
  }
  const double filter_start_time = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  oidnExecuteFilter(cache.filter);
  if (oidn_report_error(cache.device, "filter execution")) {
    return false;
  }
  oidn_sync_device(cache);
  if (oidn_report_error(cache.device, "filter sync")) {
    return false;
  }
  const double filter_ms = perf_logging_enabled ?
                               (BLI_time_now_seconds() - filter_start_time) * 1000.0 :
                               0.0;

  const double unpack_start_time = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  const bool unpack_submitted = run_oidn_interop_unpack(
      ctx->queue, unpack_pipeline, input_handle, output_handle, cache, params.extent, wait_for_unpack);
  if (perf_logging_enabled) {
    const double unpack_ms = (BLI_time_now_seconds() - unpack_start_time) * 1000.0;
    std::fprintf(stderr,
                 "EEVEE HWRT perf oidn_backend gpu=%d extent=%dx%d albedo=%d normal=%d "
                 "pack_ms=%.2f filter_submit_ms=%.2f unpack_ms=%.2f unpacked=%d\n",
                 use_gpu ? 1 : 0,
                 params.extent.x,
                 params.extent.y,
                 use_albedo ? 1 : 0,
                 use_normal ? 1 : 0,
                 pack_ms,
                 filter_ms,
                 unpack_ms,
                 unpack_submitted ? 1 : 0);
  }
  return unpack_submitted;
#endif
}

void raytrace_scene_free(GPUHardwareRaytraceScene *scene)
{
  delete scene;
}

}  // namespace blender::gpu::metal
