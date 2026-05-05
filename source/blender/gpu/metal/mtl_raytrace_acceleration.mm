/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include "mtl_raytrace_acceleration.hh"

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

namespace blender {

struct GPUMetalRaytraceScene {
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

  ~GPUMetalRaytraceScene()
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

static void retain_scene_resources(GPUMetalRaytraceScene *scene, NSMutableArray *resources)
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
                                                 GPUMetalRaytraceScene *scene)
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
                                                GPUMetalRaytraceScene *scene)
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
                                     GPUMetalRaytraceScene *scene,
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

static id<MTLCommandBuffer> trace_command_buffer_for_shadow(GPUMetalRaytraceScene *scene,
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

static bool finish_shadow_trace_command_buffer(GPUMetalRaytraceScene *scene,
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

static bool commit_shadow_trace_batch(GPUMetalRaytraceScene *scene)
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
  int closure_index;
  uint32_t feature_mask;
  int hardware_trace_phase;
  int reflection_bounces;
  int refraction_bounces;
  int2 resolution_bias;
  float clamp_indirect;
  float4 world_probe_atlas_coord;
  int4 use_environment_pad;
  float4 sampling_rand;
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
};

struct HardwareEnvironmentVisibilityUniforms {
  float4x4 viewinv;
  float4x4 wininv;
  int4 resolution_samples;
  float4 normal_bias_pad;
  float4 sampling_rand;
};

struct HardwareFastGIUniforms {
  float4 cascade_config[3];
  int4 grid_cascade_samples;
  int4 brick_origin_pad;
  int4 brick_extent_pad;
  float4 normal_bias_pad;
  int4 reuse_history_pad;
  float4 sampling_rand;
  int4 emissive_light_count_pad;
  float4 world_probe_atlas_coord;
  int4 gi_environment_pad;
};

struct HardwareReflectedReceiverGIUniforms {
  int4 resolution_samples;
  float4 normal_bias_pad;
  int4 environment_pad;
  int4 light_count_pad;
  float4 sampling_rand;
  float4 world_probe_atlas_coord;
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
                             const GPUMetalRaytraceSceneEntry &entry,
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
         "inline uint2 unpackUvec2x16(uint packed)\n"
         "{\n"
         "  return uint2(packed & 0xFFFFu, packed >> 16u);\n"
         "}\n"
         "inline float3 barycentric_expand(float2 barycentric)\n"
         "{\n"
         "  return float3(max(0.0f, 1.0f - barycentric.x - barycentric.y), barycentric.x, barycentric.y);\n"
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
         "  int closure_index;\n"
         "  uint feature_mask;\n"
         "  int hardware_trace_phase;\n"
         "  int reflection_bounces;\n"
         "  int refraction_bounces;\n"
         "  int2 resolution_bias;\n"
         "  float clamp_indirect;\n"
         "  float4 world_probe_atlas_coord;\n"
         "  int4 use_environment_pad;\n"
         "  float4 sampling_rand;\n"
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
         "};\n"
         "struct HardwareEnvironmentVisibilityUniforms {\n"
         "  float4x4 viewinv;\n"
         "  float4x4 wininv;\n"
         "  int4 resolution_samples;\n"
         "  float4 normal_bias_pad;\n"
         "  float4 sampling_rand;\n"
         "};\n"
"struct HardwareFastGIUniforms {\n"
"  float4 cascade_config[3];\n"
"  int4 grid_cascade_samples;\n"
"  int4 brick_origin_pad;\n"
"  int4 brick_extent_pad;\n"
"  float4 normal_bias_pad;\n"
"  int4 reuse_history_pad;\n"
"  float4 sampling_rand;\n"
"  int4 emissive_light_count_pad;\n"
"  float4 world_probe_atlas_coord;\n"
"  int4 gi_environment_pad;\n"
"};\n"
"struct HardwareReflectedReceiverGIUniforms {\n"
"  int4 resolution_samples;\n"
"  float4 normal_bias_pad;\n"
"  int4 environment_pad;\n"
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
"inline float fast_gi_hash(uint3 tid, int cascade_index, float2 offset, constant HardwareFastGIUniforms &u);\n"
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
"  const float d_sqr = dist * dist;\n"
"  float radius = light.attenuation_spot.x;\n"
"  if (fast_gi_is_sphere(type) && dist > 1.0e-5f) {\n"
"    radius = projected_sphere_disk_radius(radius, dist);\n"
"  }\n"
"  const float r_sqr = radius * radius;\n"
"  float power = 2.0f / (d_sqr + r_sqr + dist * sqrt(max(d_sqr + r_sqr, 1.0e-8f)));\n"
"  if (fast_gi_is_area(type)) {\n"
"    power *= saturate(dot(fast_gi_transform_z_axis(light), L));\n"
"  }\n"
"  return power;\n"
"}\n"
"inline float3 sample_fast_gi_direct_light(uint3 tid,\n"
"                                          int sample_index,\n"
"                                          int sample_count,\n"
"                                          float3 P,\n"
"                                          float3 N,\n"
"                                          float normal_bias,\n"
"                                          instance_acceleration_structure scene,\n"
"                                          constant FastGILightRecord *light_buf,\n"
"                                          constant HardwareFastGIUniforms &u)\n"
"{\n"
"  const int light_count = max(u.emissive_light_count_pad.y, 0);\n"
"  const int light_sample_count = min(max(u.emissive_light_count_pad.z, 0), sample_count);\n"
"  if (light_count <= 0 || light_sample_count <= 0 || sample_index >= light_sample_count) {\n"
"    return float3(0.0f);\n"
"  }\n"
"  const int cascade_index = u.grid_cascade_samples.y;\n"
"  const int light_index = min(int(fast_gi_hash(tid,\n"
"                                               cascade_index + sample_index * 29,\n"
"                                               float2(0.41f, 0.67f),\n"
"                                               u) * float(light_count)),\n"
"                              light_count - 1);\n"
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
"  if (!(attenuation > 1.0e-6f)) {\n"
"    return float3(0.0f);\n"
"  }\n"
"  const float facing = saturate(dot(N, L));\n"
"  if (!(facing > 1.0e-4f)) {\n"
"    return float3(0.0f);\n"
"  }\n"
"  const float3 origin = P + N * normal_bias;\n"
"  const float ray_tmin = max(5.0e-4f, normal_bias * 0.5f);\n"
"  const float ray_tmax = fast_gi_is_sun(type) ? 100000.0f : max(light_distance - normal_bias, ray_tmin);\n"
"  intersector<triangle_data, instancing, max_levels<2>> i;\n"
"  i.assume_geometry_type(geometry_type::triangle);\n"
"  i.force_opacity(forced_opacity::opaque);\n"
"  intersection_result<triangle_data, instancing, max_levels<2>> intersection = i.intersect(ray(origin, L, ray_tmin, ray_tmax), scene);\n"
"  if (intersection.type == intersection_type::triangle) {\n"
"    return float3(0.0f);\n"
"  }\n"
"  const float direct_scale = float(sample_count) / float(light_sample_count);\n"
"  const float power = light.color_diffuse_power.w *\n"
"                      fast_gi_light_point_power(light, type, light_distance, L) *\n"
"                      attenuation * facing * float(light_count) * direct_scale;\n"
"  return light.color_diffuse_power.xyz * power;\n"
"}\n"
"inline float fast_gi_hash(uint3 tid, int cascade_index, float2 offset, constant HardwareFastGIUniforms &u)\n"
"{\n"
"  const float2 seed = float2(u.sampling_rand.x * 23.47f + u.sampling_rand.z * 11.13f,\n"
"                              u.sampling_rand.y * 29.59f + u.sampling_rand.w * 7.71f);\n"
"  const uint3 brick_voxel = uint3(max(u.brick_origin_pad.x, 0),\n"
"                                  max(u.brick_origin_pad.y, 0),\n"
"                                  max(u.brick_origin_pad.z, 0)) + tid;\n"
"  const float2 base = float2(float(brick_voxel.x + brick_voxel.z * 31u),\n"
"                              float(brick_voxel.y + uint(cascade_index) * 17u));\n"
"  return hash12(base + offset + seed);\n"
"}\n"
"inline float3 sample_fast_gi_direction(uint3 tid, int sample_index, constant HardwareFastGIUniforms &u)\n"
"{\n"
"  const int cascade_index = u.grid_cascade_samples.y;\n"
"  const float sample_count = float(max(u.grid_cascade_samples.w, 1));\n"
"  const float phi_offset = fast_gi_hash(tid, cascade_index, float2(0.73f, 0.53f), u);\n"
"  const float sample_u = (float(sample_index) + 0.5f) / sample_count;\n"
"  const float z = 1.0f - 2.0f * sample_u;\n"
"  const float phi = 2.39996322973f * (float(sample_index) + phi_offset * sample_count + 0.5f);\n"
"  const float r = sqrt(max(0.0f, 1.0f - z * z));\n"
"  return float3(cos(phi) * r, sin(phi) * r, z);\n"
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
"inline float fast_gi_average_radiance_weight(float proposal_pdf)\n"
"{\n"
"  const float uniform_sphere_pdf = 0.07957747154f;\n"
"  return uniform_sphere_pdf / max(proposal_pdf, 1.0e-6f);\n"
"}\n"
"inline float fast_gi_balanced_average_radiance_weight(float proposal_pdf, float proposal_fraction)\n"
"{\n"
"  const float uniform_pdf = 0.07957747154f;\n"
"  const float mix_fraction = saturate(proposal_fraction);\n"
"  const float mixture_pdf = mix(uniform_pdf, proposal_pdf, mix_fraction);\n"
"  return uniform_pdf / max(mixture_pdf, 1.0e-6f);\n"
"}\n"
"inline float4 sample_fast_gi_emissive_direction(uint3 tid,\n"
"                                                int sample_index,\n"
"                                                float3 P,\n"
"                                                constant EmissiveLightRecord *emissive_lights,\n"
"                                                constant HardwareFastGIUniforms &u)\n"
"{\n"
"  const int light_count = max(u.emissive_light_count_pad.x, 0);\n"
"  if (light_count <= 0) {\n"
"    return float4(sample_fast_gi_direction(tid, sample_index, u), 0.07957747154f);\n"
"  }\n"
"  const float select = fast_gi_hash(tid,\n"
"                                    u.grid_cascade_samples.y + sample_index * 7,\n"
"                                    float2(0.11f + float(sample_index) * 0.37f, 0.89f),\n"
"                                    u);\n"
"  const int light_index = min(int(select * float(light_count)), light_count - 1);\n"
"  const float light_select_pdf = 1.0f / float(light_count);\n"
"  const float4 light = emissive_lights[light_index].center_radius;\n"
"  float3 L = light.xyz - P;\n"
"  const float distance_to_light = length(L);\n"
"  if (!(distance_to_light > 1.0e-5f)) {\n"
"    return float4(sample_fast_gi_direction(tid, sample_index, u), 0.07957747154f);\n"
"  }\n"
"  L /= distance_to_light;\n"
"  const float aperture = min(light.w / distance_to_light, 0.9f);\n"
"  const float cos_theta = sqrt(max(1.0f - aperture * aperture, 0.0f));\n"
"  const float cone_solid_angle = max(6.28318530718f * (1.0f - cos_theta), 1.0e-4f);\n"
"  const float emissive_pdf = light_select_pdf / cone_solid_angle;\n"
"  if (!(aperture > 1.0e-5f)) {\n"
"    return float4(L, emissive_pdf);\n"
"  }\n"
"  float3 right, up;\n"
"  make_orthonormal_basis(L, right, up);\n"
"  const float2 jitter = float2(fast_gi_hash(tid,\n"
"                                            u.grid_cascade_samples.y + sample_index * 11,\n"
"                                            float2(0.31f, 0.17f),\n"
"                                            u),\n"
"                              fast_gi_hash(tid,\n"
"                                            u.grid_cascade_samples.y + sample_index * 13,\n"
"                                            float2(0.59f, 0.41f),\n"
"                                            u));\n"
"  const float2 disk = sample_disk(jitter) * aperture;\n"
"  return float4(normalize(L + right * disk.x + up * disk.y), emissive_pdf);\n"
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
"inline float3 sample_fast_gi_diffuse_direction(uint3 tid,\n"
"                                               int sample_index,\n"
"                                               int bounce_index,\n"
"                                               float3 N,\n"
"                                               constant HardwareFastGIUniforms &u)\n"
"{\n"
"  float3 right, up;\n"
"  make_orthonormal_basis(N, right, up);\n"
"  const int hash_cascade = u.grid_cascade_samples.y + bounce_index * 23;\n"
"  const float2 xi = float2(fast_gi_hash(tid,\n"
"                                        hash_cascade + sample_index * 17,\n"
"                                        float2(0.19f, 0.83f),\n"
"                                        u),\n"
"                          fast_gi_hash(tid,\n"
"                                        hash_cascade + sample_index * 19,\n"
"                                        float2(0.67f, 0.29f),\n"
"                                        u));\n"
"  const float2 disk = sample_disk(xi);\n"
"  const float z = sqrt(max(0.0f, 1.0f - dot(disk, disk)));\n"
"  return normalize(right * disk.x + up * disk.y + N * z);\n"
"}\n"
"inline float3 sample_fast_gi_reflection_direction(uint3 tid,\n"
"                                                  int sample_index,\n"
"                                                  int bounce_index,\n"
"                                                  float3 ray_direction,\n"
"                                                  float3 surface_N,\n"
"                                                  float roughness,\n"
"                                                  constant HardwareFastGIUniforms &u)\n"
"{\n"
"  const float alpha = roughness * roughness;\n"
"  const float3 sharp_dir = reflect(ray_direction, surface_N);\n"
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
"  const int hash_cascade = u.grid_cascade_samples.y + bounce_index * 29;\n"
"  const float2 xi = float2(fast_gi_hash(tid,\n"
"                                        hash_cascade + sample_index * 23,\n"
"                                        float2(0.43f, 0.17f + 0.11f * float(bounce_index)),\n"
"                                        u),\n"
"                          fast_gi_hash(tid,\n"
"                                        hash_cascade + sample_index * 31,\n"
"                                        float2(0.71f, 0.29f + 0.07f * float(bounce_index)),\n"
"                                        u));\n"
"  const float3 Ht = ggx_sample_vndf(sample_cylinder(xi), Vt, alpha);\n"
"  const float3 H = normalize(right * Ht.x + up * Ht.y + surface_N * Ht.z);\n"
"  const float3 sampled_dir = reflect(ray_direction, H);\n"
"  return (dot(sampled_dir, sampled_dir) > 1.0e-10f) ? normalize(sampled_dir) : sharp_dir;\n"
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
"inline float3 sample_fast_gi_world_radiance(texture2d_array<float, access::sample> world_probe_tx,\n"
"                                            float3 direction,\n"
"                                            constant HardwareFastGIUniforms &u)\n"
"{\n"
"  if (u.gi_environment_pad.y == 0) {\n"
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
"                                          constant HardwareTraceUniforms &u)\n"
"{\n"
"  if (u.use_environment_pad.x == 0) {\n"
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
"    constant FastGILightRecord *light_buf,\n"
"    constant HardwareReflectedReceiverGIUniforms &u)\n"
"{\n"
"  const int light_count = max(u.light_count_pad.x, 0);\n"
"  const int light_sample_count = min(max(u.light_count_pad.y, 0), sample_count);\n"
"  if (light_count <= 0 || light_sample_count <= 0 || sample_index >= light_sample_count) {\n"
"    return float3(0.0f);\n"
"  }\n"
"  const int light_index = min(int(reflected_receiver_gi_hash(\n"
"                                      tid, sample_index, float2(0.41f, 0.67f), u) * float(light_count)),\n"
"                              light_count - 1);\n"
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
"  const float direct_scale = float(sample_count) / float(light_sample_count);\n"
"  const float power = light.color_diffuse_power.w *\n"
"                      fast_gi_light_point_power(light, type, light_distance, L) * attenuation *\n"
"                      facing * float(light_count) * direct_scale;\n"
"  return light.color_diffuse_power.xyz * power;\n"
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
"inline float4 sample_fast_gi_cascade(texture3d<float, access::sample> fast_gi_history_tx,\n"
"                                     float3 P,\n"
"                                     int cascade_index,\n"
"                                     constant HardwareFastGIUniforms &u)\n"
"{\n"
"  if (cascade_index < 0 || cascade_index >= u.grid_cascade_samples.z) {\n"
"    return float4(0.0f);\n"
"  }\n"
"  const int grid_resolution = max(u.grid_cascade_samples.x, 1);\n"
"  const float4 cascade_cfg = u.cascade_config[cascade_index];\n"
"  const float voxel_size = cascade_cfg.w;\n"
"  if (!(voxel_size > 0.0f)) {\n"
"    return float4(0.0f);\n"
"  }\n"
"  const float3 cascade_min = cascade_cfg.xyz - 0.5f * float(grid_resolution) * voxel_size;\n"
"  const float3 uvw = (P - cascade_min) / (float(grid_resolution) * voxel_size);\n"
"  if (any(uvw < float3(0.0f)) || any(uvw >= float3(1.0f))) {\n"
"    return float4(0.0f);\n"
"  }\n"
"  constexpr sampler linear_sampler(coord::normalized, address::clamp_to_zero, filter::linear);\n"
"  const float3 atlas_uvw = float3(uvw.xy, (uvw.z + float(cascade_index)) / float(max(u.grid_cascade_samples.z, 1)));\n"
"  return fast_gi_history_tx.sample(linear_sampler, atlas_uvw);\n"
"}\n"
"inline float4 sample_fast_gi_next_cascade_continuation(\n"
"    texture3d<float, access::sample> fast_gi_history_tx,\n"
"    float3 P,\n"
"    int cascade_index,\n"
"    constant HardwareFastGIUniforms &u)\n"
"{\n"
"  const int next_cascade_index = cascade_index + 1;\n"
"  if (next_cascade_index >= u.grid_cascade_samples.z) {\n"
"    return float4(0.0f);\n"
"  }\n"
"  const float4 coarse = sample_fast_gi_cascade(fast_gi_history_tx, P, next_cascade_index, u);\n"
"  if (coarse.w <= 1.0e-4f) {\n"
"    return float4(0.0f);\n"
"  }\n"
"  const float confidence = saturate(coarse.w);\n"
"  return float4(coarse.xyz / coarse.w, confidence);\n"
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
         "  const int2 texel_fullres = int2(tid) * uniforms.resolution_scale + uniforms.resolution_bias;\n"
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
"    max_bounces = 1 + max(uniforms.reflection_bounces, 1);\n"
"  }\n"
"  else if (supports_hardware_refraction) {\n"
"    max_bounces = 1 + max(uniforms.refraction_bounces, 1);\n"
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
"    const float reflection_layer_coverage = clamp(final_proxy.packed_thickness.z, 0.0f, 1.0f);\n"
"    const float refraction_ior = max(final_proxy.ior_closure_type.y, 1.0e-3f);\n"
"    const float eta = entering ? (1.0f / refraction_ior) : refraction_ior;\n"
"    const bool full_reflection_coverage = (proxy_closure == HWRT_CLOSURE_REFLECTION) &&\n"
"                                        (reflection_layer_coverage >= 1.0f - 1.0e-3f);\n"
"    const bool preserve_layered_principled_scene_final =\n"
"        scene_final_specular_phase && supports_hardware_reflection &&\n"
"        (bounce == start_bounce) &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL) != 0u) &&\n"
"        !full_reflection_coverage &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_ALPHA_BLEND_TRANSPARENT) == 0u);\n"
"    const bool preserve_textured_specular_scene_final =\n"
"        scene_final_specular_phase && supports_hardware_reflection &&\n"
"        (bounce == start_bounce) &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL) != 0u) &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_ALPHA_BLEND_TRANSPARENT) == 0u);\n"
"    const bool preserve_transparent_scene_final =\n"
"        scene_final_specular_phase && (bounce == start_bounce) &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_ALPHA_BLEND_TRANSPARENT) != 0u) &&\n"
"        (transparent_alpha < 1.0f - 1.0e-3f);\n"
"    const bool replay_refracted_textured_receiver =\n"
"        scene_final_specular_phase && supports_hardware_refraction && (bounce > start_bounce) &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL) != 0u) &&\n"
"        ((proxy_flags & HWRT_PROXY_FLAG_ALPHA_BLEND_TRANSPARENT) == 0u);\n"
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
"    if (scene_final_specular_phase && supports_hardware_reflection && !preserved_material_scene_final && !preserve_transparent_scene_final && !preserve_scene_final_transmission_layer && (bounce == start_bounce) &&\n"
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
"    if ((preserved_layered_scene_final_hit || preserved_scene_final_reflective_hit ||\n"
"         preserved_transparent_scene_final_hit) &&\n"
"        (bounce > start_bounce)) {\n"
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
"      break;\n"
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
"                              !replay_refracted_textured_receiver &&\n"
"                              ((continuation_proxy_closure == HWRT_CLOSURE_REFLECTION) ||\n"
"                               (continuation_proxy_closure == HWRT_CLOSURE_REFRACTION));\n"
"    if (replay_refracted_textured_receiver) {\n"
"      final_refracted_textured_receiver_hit = true;\n"
"    }\n"
"    if (!can_continue) {\n"
"      break;\n"
"    }\n"
"    float3 next_direction = ray_direction;\n"
"    if (continuation_proxy_closure == HWRT_CLOSURE_REFLECTION) {\n"
"      const float3 reflection_tint = clamp(final_proxy.reflection_color_roughness.xyz,\n"
"                                           float3(0.0f),\n"
"                                           float3(uniforms.clamp_indirect));\n"
"      throughput *= reflection_tint;\n"
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
"      throughput *= clamp(final_proxy.transmission_color_roughness.xyz,\n"
"                           float3(0.0f),\n"
"                           float3(uniforms.clamp_indirect));\n"
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
"        if (preserved_entering) {\n"
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
"        const int transmission_max_bounces = 1 + max(uniforms.refraction_bounces, 1);\n"
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
"                  min(sample_trace_world_radiance(world_probe_tx, transmission_ray_direction, uniforms),\n"
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
"          const bool transmission_can_continue =\n"
"              (transmission_bounce + 1 < transmission_max_bounces) &&\n"
"              ((transmission_resolved_proxy_closure == HWRT_CLOSURE_REFLECTION) ||\n"
"               (transmission_resolved_proxy_closure == HWRT_CLOSURE_REFRACTION));\n"
"          if (transmission_replay_reflective_receiver && !transmission_receiver_valid) {\n"
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
"          }\n"
"          if (!transmission_can_continue) {\n"
"            if (transmission_receiver_valid) {\n"
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
"          if ((transmission_resolved_proxy_closure == HWRT_CLOSURE_REFRACTION) &&\n"
"              transmission_entering) {\n"
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
"  /* Reflected receiver Secondary GI is owned by eevee_hardware_trace_reflected_receiver_gi.\n"
"   * Keep this scene-final trace responsible for exporting the hit payload only; otherwise the\n"
"   * mirror receives the old inline continuation plus the new blurred receiver GI. */\n"
"  const bool scene_final_diffuse_receiver = false;\n"
"  if (scene_final_diffuse_receiver) {\n"
"    const int diffuse_sample_count = 8;\n"
"    const float3 diffuse_origin = final_position + final_normal * 2.0e-3f;\n"
"    float3 incoming = float3(0.0f);\n"
"    for (int diffuse_sample = 0; diffuse_sample < diffuse_sample_count; diffuse_sample++) {\n"
"      const bool use_emissive_guiding = (uniforms.use_environment_pad.y > 0) &&\n"
"                                        ((diffuse_sample & 1) == 0);\n"
"      float4 guided_sample = float4(0.0f);\n"
"      float3 diffuse_dir;\n"
"      float diffuse_weight = 1.0f;\n"
"      if (use_emissive_guiding) {\n"
"        guided_sample = sample_trace_emissive_direction(\n"
"            tid, diffuse_sample, diffuse_origin, emissive_lights, uniforms);\n"
"        diffuse_dir = guided_sample.xyz;\n"
"        const float cosine_pdf = saturate(dot(final_normal, diffuse_dir)) * 0.31830988618f;\n"
"        const float mixture_pdf = max(0.5f * guided_sample.w + 0.5f * cosine_pdf, 1.0e-6f);\n"
"        diffuse_weight = cosine_pdf / mixture_pdf;\n"
"      }\n"
"      else {\n"
"        diffuse_dir = sample_trace_diffuse_direction(\n"
"            tid, diffuse_sample, uniforms.closure_index + 37 + diffuse_sample * 13, final_normal, uniforms);\n"
"        diffuse_weight = (uniforms.use_environment_pad.y > 0) ? 2.0f : 1.0f;\n"
"      }\n"
"      if (!(dot(diffuse_dir, diffuse_dir) > 1.0e-10f) || diffuse_weight <= 0.0f) {\n"
"        continue;\n"
"      }\n"
"      intersection_result<triangle_data, instancing, max_levels<2>> diffuse_intersection =\n"
"          i.intersect(ray(diffuse_origin, diffuse_dir, 5.0e-4f, 10000.0f), scene);\n"
"      if (diffuse_intersection.type == intersection_type::triangle) {\n"
"        const uint diffuse_user_id = diffuse_intersection.user_instance_id[0];\n"
"        incoming += min(max(emissive_radiance[diffuse_user_id].xyz, float3(0.0f)),\n"
"                        float3(uniforms.clamp_indirect)) * diffuse_weight;\n"
"      }\n"
"      else if (!use_emissive_guiding) {\n"
"        incoming += min(sample_trace_world_radiance(world_probe_tx, diffuse_dir, uniforms),\n"
"                        float3(uniforms.clamp_indirect)) * diffuse_weight;\n"
"      }\n"
"    }\n"
"    incoming /= float(diffuse_sample_count);\n"
"    const float3 receiver_albedo = min(max(diffuse_albedo[final_user_id].xyz, float3(0.0f)),\n"
"                                       float3(uniforms.clamp_indirect));\n"
"    radiance += min(final_output_throughput * receiver_albedo * incoming,\n"
"                    float3(uniforms.clamp_indirect));\n"
"  }\n"
"  const float2 packed_direction = direction_pack(final_direction);\n"
"  ray_time_img.write(float4(max(total_distance, 1.0e-4f), 0.0f, 0.0f, 0.0f), tid);\n"
"  ray_radiance_img.write(float4(radiance, 0.0f), tid);\n"
"  float3 final_proxy_color = (final_proxy_closure == HWRT_CLOSURE_REFLECTION) ?\n"
"                                 final_proxy.reflection_color_roughness.xyz :\n"
"                                 final_proxy.transmission_color_roughness.xyz;\n"
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
"                              (final_refracted_textured_receiver_hit ? 16u : 0u);\n"
"  hit_identity_img.write(uint4(final_user_id, final_primitive_id, identity_flags, 0xFFFFFFFFu), tid);\n"
"  hit_barycentric_img.write(float4(final_barycentric.x, final_barycentric.y, final_segment_distance, 0.0f), tid);\n"
"  if (layered_receiver_valid) {\n"
"    const float2 layered_receiver_packed_direction = direction_pack(layered_receiver_direction);\n"
"    const float3 layered_receiver_proxy_color =\n"
"        (layered_receiver_proxy_closure == HWRT_CLOSURE_REFLECTION) ?\n"
"            layered_receiver_proxy.reflection_color_roughness.xyz :\n"
"            layered_receiver_proxy.transmission_color_roughness.xyz;\n"
"    const float layered_receiver_proxy_roughness =\n"
"        (layered_receiver_proxy_closure == HWRT_CLOSURE_REFLECTION) ?\n"
"            layered_receiver_proxy.reflection_color_roughness.w :\n"
"            layered_receiver_proxy.transmission_color_roughness.w;\n"
"    const float layered_receiver_proxy_ior =\n"
"        (layered_receiver_proxy_closure == HWRT_CLOSURE_REFLECTION) ?\n"
"            layered_receiver_proxy.ior_closure_type.x :\n"
"            layered_receiver_proxy.ior_closure_type.y;\n"
"    layered_receiver_ray_time_img.write(\n"
"        float4(max(layered_receiver_total_distance, 1.0e-4f), 0.0f, 0.0f, 0.0f), tid);\n"
"    layered_receiver_ray_radiance_img.write(float4(0.0f), tid);\n"
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
"    layered_receiver_throughput_img.write(\n"
"        float4(layered_receiver_carried_throughput,\n"
"               preserved_transparent_scene_final_hit ? 1.0f : 0.0f),\n"
"        tid);\n"
"    layered_receiver_identity_img.write(uint4(layered_receiver_user_id,\n"
"                                              layered_receiver_primitive_id,\n"
"                                              layered_receiver_front_facing | 8u,\n"
"                                              0xFFFFFFFFu),\n"
"                                      tid);\n"
"    layered_receiver_barycentric_img.write(\n"
"        float4(layered_receiver_barycentric.x,\n"
"               layered_receiver_barycentric.y,\n"
"               layered_receiver_segment_distance,\n"
"               0.0f),\n"
"        tid);\n"
"  }\n"
"  if (transmission_receiver_valid) {\n"
"    const float2 transmission_receiver_packed_direction = direction_pack(\n"
"        transmission_receiver_direction);\n"
"    const float3 transmission_receiver_proxy_color =\n"
"        (transmission_receiver_proxy_closure == HWRT_CLOSURE_REFLECTION) ?\n"
"            transmission_receiver_proxy.reflection_color_roughness.xyz :\n"
"            transmission_receiver_proxy.transmission_color_roughness.xyz;\n"
"    const float transmission_receiver_proxy_roughness =\n"
"        (transmission_receiver_proxy_closure == HWRT_CLOSURE_REFLECTION) ?\n"
"            transmission_receiver_proxy.reflection_color_roughness.w :\n"
"            transmission_receiver_proxy.transmission_color_roughness.w;\n"
"    const float transmission_receiver_proxy_ior =\n"
"        (transmission_receiver_proxy_closure == HWRT_CLOSURE_REFLECTION) ?\n"
"            transmission_receiver_proxy.ior_closure_type.x :\n"
"            transmission_receiver_proxy.ior_closure_type.y;\n"
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
"    const uint transmission_receiver_identity_flags =\n"
"        transmission_receiver_front_facing | 8u |\n"
"        ((transmission_receiver_proxy_closure == HWRT_CLOSURE_REFLECTION) ? 16u : 0u);\n"
"    transmission_receiver_identity_img.write(uint4(transmission_receiver_user_id,\n"
"                                                   transmission_receiver_primitive_id,\n"
"                                                   transmission_receiver_identity_flags,\n"
"                                                   0xFFFFFFFFu),\n"
"                                           tid);\n"
"    transmission_receiver_barycentric_img.write(\n"
"        float4(transmission_receiver_barycentric.x,\n"
"               transmission_receiver_barycentric.y,\n"
"               transmission_receiver_segment_distance,\n"
"               0.0f),\n"
"        tid);\n"
"  }\n"
         "}\n"
         "kernel void eevee_hardware_trace_directional_shadow(\n"
         "    uint2 tid [[thread_position_in_grid]],\n"
         "    instance_acceleration_structure scene [[buffer(0)]],\n"
         "    constant HardwareShadowUniforms &uniforms [[buffer(1)]],\n"
         "    constant float4 *world_sunlight_direction [[buffer(2)]],\n"
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
         "  intersector<triangle_data, instancing, max_levels<2>> i;\n"
         "  i.assume_geometry_type(geometry_type::triangle);\n"
         "  i.force_opacity(forced_opacity::opaque);\n"
         "  const int sample_count = (uniforms.shadow_params.x > 1.0e-6f) ? max(int(uniforms.shadow_params.y), 1) : 1;\n"
         "  float visibility = 0.0f;\n"
         "  for (int sample_index = 0; sample_index < sample_count; sample_index++) {\n"
         "    const float3 sample_dir = sample_directional_shadow_direction(\n"
         "        tid, sample_index, uniforms, world_sunlight_direction);\n"
"    const float3 origin = P + N * normal_bias;\n"
         "    intersection_result<triangle_data, instancing, max_levels<2>> intersection = i.intersect(ray(origin, sample_dir, ray_tmin, 100000.0f), scene);\n"
         "    visibility += (intersection.type == intersection_type::triangle) ? 0.0f : 1.0f;\n"
         "  }\n"
         "  visibility /= float(sample_count);\n"
         "  shadow_visibility_img.write(float4(visibility), tid, uint(uniforms.resolution_layer.z));\n"
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
"  intersector<triangle_data, instancing, max_levels<2>> i;\n"
"  i.assume_geometry_type(geometry_type::triangle);\n"
"  i.force_opacity(forced_opacity::opaque);\n"
"  const int sample_count = (uniforms.shadow_params.x > 1.0e-6f) ? max(int(uniforms.shadow_params.y), 1) : 1;\n"
"  float visibility = 0.0f;\n"
"  for (int sample_index = 0; sample_index < sample_count; sample_index++) {\n"
"    const float3 sample_dir = sample_directional_shadow_direction(\n"
"        tid, sample_index, uniforms, world_sunlight_direction);\n"
"    const float3 origin = P + shadow_N * normal_bias;\n"
"    intersection_result<triangle_data, instancing, max_levels<2>> intersection = i.intersect(ray(origin, sample_dir, ray_tmin, 100000.0f), scene);\n"
"    visibility += (intersection.type == intersection_type::triangle) ? 0.0f : 1.0f;\n"
"  }\n"
"  visibility /= float(sample_count);\n"
"  shadow_visibility_img.write(float4(visibility), tid, uint(uniforms.resolution_layer.z));\n"
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
"kernel void eevee_hardware_trace_fast_gi(\n"
"    uint3 tid [[thread_position_in_grid]],\n"
"    instance_acceleration_structure scene [[buffer(0)]],\n"
"    constant HardwareFastGIUniforms &uniforms [[buffer(1)]],\n"
"    constant float4 *emissive_radiance [[buffer(2)]],\n"
"    constant float4 *diffuse_albedo [[buffer(3)]],\n"
"    constant HardwareMaterialProxy *material_proxy [[buffer(4)]],\n"
"    constant EmissiveLightRecord *emissive_lights [[buffer(5)]],\n"
"    constant float4 *triangle_normals [[buffer(6)]],\n"
"    constant TriangleNormalRange *triangle_normal_ranges [[buffer(7)]],\n"
"    constant FastGILightRecord *fast_gi_lights [[buffer(8)]],\n"
"    texture3d<float, access::sample> fast_gi_history_tx [[texture(0)]],\n"
"    texture3d<float, access::read_write> fast_gi_img [[texture(1)]],\n"
"    texture3d<float, access::read_write> fast_gi_error_img [[texture(2)]],\n"
"    texture3d<float, access::read_write> fast_gi_visibility_img [[texture(3)]],\n"
"    texture2d_array<float, access::sample> world_probe_tx [[texture(4)]])\n"
"{\n"
"  const uint grid_resolution_u = uint(max(uniforms.grid_cascade_samples.x, 1));\n"
"  const uint3 brick_extent_u = uint3(max(uniforms.brick_extent_pad.x, 1),\n"
"                                     max(uniforms.brick_extent_pad.y, 1),\n"
"                                     max(uniforms.brick_extent_pad.z, 1));\n"
"  if (any(tid >= brick_extent_u)) {\n"
"    return;\n"
"  }\n"
"  const uint3 brick_origin_u = uint3(max(uniforms.brick_origin_pad.x, 0),\n"
"                                     max(uniforms.brick_origin_pad.y, 0),\n"
"                                     max(uniforms.brick_origin_pad.z, 0));\n"
"  const uint3 cascade_voxel = brick_origin_u + tid;\n"
"  if (any(cascade_voxel >= uint3(grid_resolution_u))) {\n"
"    return;\n"
"  }\n"
"  const int cascade_index = uniforms.grid_cascade_samples.y;\n"
"  const int cascade_count = max(uniforms.grid_cascade_samples.z, 1);\n"
"  const int base_sample_count = max(uniforms.grid_cascade_samples.w, 1);\n"
"  const int gi_bounces = max(uniforms.gi_environment_pad.x, 1);\n"
"  const uint3 atlas_voxel = uint3(\n"
"      cascade_voxel.xy, cascade_voxel.z + uint(cascade_index) * grid_resolution_u);\n"
"  const bool reuse_history = uniforms.reuse_history_pad.x != 0;\n"
"  const float4 history = reuse_history ? fast_gi_img.read(atlas_voxel) : float4(0.0f);\n"
"  const float history_error = reuse_history ? fast_gi_error_img.read(atlas_voxel).x : 1.0f;\n"
"  const float4 history_visibility = reuse_history ? fast_gi_visibility_img.read(atlas_voxel) : float4(0.0f);\n"
"  if (fast_gi_skip_stable_space(reuse_history, history_error, history_visibility)) {\n"
"    fast_gi_img.write(history, atlas_voxel);\n"
"    fast_gi_error_img.write(history_error * 0.99f, atlas_voxel);\n"
"    fast_gi_visibility_img.write(history_visibility, atlas_voxel);\n"
"    return;\n"
"  }\n"
"  const int sample_count = fast_gi_adaptive_sample_count(\n"
"      base_sample_count, reuse_history, history_error, history_visibility);\n"
"  const float4 cascade_cfg = uniforms.cascade_config[cascade_index];\n"
"  const float voxel_size = cascade_cfg.w;\n"
"  if (!(voxel_size > 0.0f)) {\n"
"    fast_gi_img.write(reuse_history ? history * 0.98f : float4(0.0f), atlas_voxel);\n"
"    fast_gi_error_img.write(reuse_history ? history_error * 0.98f : 1.0f, atlas_voxel);\n"
"    fast_gi_visibility_img.write(reuse_history ? history_visibility * 0.98f : float4(0.0f), atlas_voxel);\n"
"    return;\n"
"  }\n"
"  const float3 cascade_min = cascade_cfg.xyz - 0.5f * float(grid_resolution_u) * voxel_size;\n"
"  const float3 P = cascade_min + (float3(cascade_voxel) + 0.5f) * voxel_size;\n"
"  const float shell_distance = voxel_size * float(grid_resolution_u);\n"
"  const float normal_bias = max(uniforms.normal_bias_pad.x, voxel_size * 0.05f);\n"
"  const float ray_tmin = max(5.0e-4f, normal_bias * 0.25f);\n"
"  const float ray_tmax = max(ray_tmin * 2.0f, shell_distance);\n"
"  const int emissive_importance_count = (uniforms.emissive_light_count_pad.x > 0) ?\n"
"                                          min(3, max(sample_count / 6, 1)) :\n"
"                                          0;\n"
"  const bool allow_environment_diffuse_continuation = (uniforms.gi_environment_pad.y != 0);\n"
"  const int diffuse_continuation_count = ((gi_bounces > 1 && cascade_index + 1 < cascade_count) ||\n"
"                                           allow_environment_diffuse_continuation) ?\n"
"                                           min((cascade_index == 0) ? 3 : 2,\n"
"                                               max(sample_count / 6, 1)) :\n"
"                                           0;\n"
"  const float emissive_proposal_fraction = float(emissive_importance_count) / float(max(sample_count, 1));\n"
"  intersector<triangle_data, instancing, max_levels<2>> i;\n"
"  i.assume_geometry_type(geometry_type::triangle);\n"
"  i.force_opacity(forced_opacity::opaque);\n"
"  float3 accum = float3(0.0f);\n"
"  float accum_luma = 0.0f;\n"
"  float accum_luma_sq = 0.0f;\n"
"  float accum_weight = 0.0f;\n"
"  float occupancy_accum = 0.0f;\n"
"  float thickness_accum = 0.0f;\n"
"  float openness_accum = 0.0f;\n"
"  for (int sample_index = 0; sample_index < sample_count; sample_index++) {\n"
"    const bool use_emissive_importance = sample_index < emissive_importance_count;\n"
"    const float4 emissive_sample = use_emissive_importance ?\n"
"                                      sample_fast_gi_emissive_direction(\n"
"                                          tid, sample_index, P, emissive_lights, uniforms) :\n"
"                                      float4(0.0f, 0.0f, 0.0f, 0.07957747154f);\n"
"    float sample_weight = 1.0f;\n"
"    const float3 sample_dir = use_emissive_importance ?\n"
"                                  emissive_sample.xyz :\n"
"                                  sample_fast_gi_direction(tid, sample_index, uniforms);\n"
"    const float emissive_balance_weight = use_emissive_importance ?\n"
"                                            fast_gi_balanced_average_radiance_weight(\n"
"                                                emissive_sample.w, emissive_proposal_fraction) :\n"
"                                            1.0f;\n"
"    if (use_emissive_importance) {\n"
"      sample_weight = fast_gi_average_radiance_weight(emissive_sample.w);\n"
"    }\n"
"    float3 sample_radiance = float3(0.0f);\n"
"    float3 sample_throughput = float3(1.0f);\n"
"    float3 trace_origin = P + sample_dir * normal_bias;\n"
"    float3 trace_dir = sample_dir;\n"
"    float trace_tmax = ray_tmax;\n"
"    const int max_specular_redirects = max(gi_bounces, 1);\n"
"    int specular_redirects = 0;\n"
"    while (true) {\n"
"      intersection_result<triangle_data, instancing, max_levels<2>> intersection = i.intersect(\n"
"          ray(trace_origin, trace_dir, ray_tmin, trace_tmax), scene);\n"
"      if (intersection.type != intersection_type::triangle) {\n"
"        if (specular_redirects == 0) {\n"
"          openness_accum += 1.0f;\n"
"        }\n"
"        if ((gi_bounces > specular_redirects + 1) && (cascade_index + 1 < cascade_count)) {\n"
"          const float3 far_P = trace_origin + trace_dir * trace_tmax;\n"
"          const float4 miss_continuation = sample_fast_gi_next_cascade_continuation(\n"
"              fast_gi_history_tx, far_P, cascade_index, uniforms);\n"
"          if (miss_continuation.w > 1.0e-4f) {\n"
"            sample_radiance = sample_throughput * miss_continuation.xyz;\n"
"            sample_weight *= 0.7f * miss_continuation.w;\n"
"          }\n"
"          else if (use_emissive_importance) {\n"
"            sample_weight = 0.0f;\n"
"          }\n"
"          else {\n"
"            sample_radiance = sample_throughput * sample_fast_gi_world_radiance(\n"
"                world_probe_tx, trace_dir, uniforms);\n"
"          }\n"
"        }\n"
"        else if (use_emissive_importance) {\n"
"          sample_weight = 0.0f;\n"
"        }\n"
"        else {\n"
"          sample_radiance = sample_throughput * sample_fast_gi_world_radiance(\n"
"              world_probe_tx, trace_dir, uniforms);\n"
"        }\n"
"        break;\n"
"      }\n"
"      if (specular_redirects == 0) {\n"
"        const float near_hit = saturate(1.0f - intersection.distance / max(voxel_size * 2.0f, 1.0e-4f));\n"
"        openness_accum += saturate(intersection.distance / max(ray_tmax, 1.0e-4f));\n"
"        occupancy_accum += 1.0f;\n"
"        thickness_accum += near_hit;\n"
"      }\n"
"      const uint user_id = intersection.user_instance_id[0];\n"
"      const float3 hit_P = trace_origin + trace_dir * intersection.distance;\n"
"      const float3 hit_albedo = max(diffuse_albedo[user_id].xyz, float3(0.0f));\n"
"      const float3 hit_emissive = max(emissive_radiance[user_id].xyz, float3(0.0f));\n"
"      const bool hit_has_diffuse = max(hit_albedo.x, max(hit_albedo.y, hit_albedo.z)) > 1.0e-4f;\n"
"      const HardwareMaterialProxy hit_proxy = material_proxy[user_id];\n"
"      const uint proxy_closure = uint(hit_proxy.ior_closure_type.z + 0.5f);\n"
"      const uint proxy_flags = uint(hit_proxy.ior_closure_type.w + 0.5f);\n"
"      const float3 reflection_tint = max(hit_proxy.reflection_color_roughness.xyz, float3(0.0f));\n"
"      const float reflection_roughness = clamp(hit_proxy.reflection_color_roughness.w, 0.0f, 1.0f);\n"
"      const bool reflector_eligible = (proxy_closure == HWRT_CLOSURE_REFLECTION) &&\n"
"                                      /* The Fast GI path only has the coarse dominant proxy, so\n"
"                                       * keep layered Principled materials on the old diffuse-owned\n"
"                                       * path instead of collapsing them into mirror-only redirectors. */\n"
"                                      ((proxy_flags & HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL) == 0u) &&\n"
"                                      (specular_redirects < max_specular_redirects) &&\n"
"                                      (max(reflection_tint.x,\n"
"                                           max(reflection_tint.y, reflection_tint.z)) > 1.0e-4f);\n"
"      sample_radiance += sample_throughput * hit_emissive;\n"
"      if (use_emissive_importance &&\n"
"          max(hit_emissive.x, max(hit_emissive.y, hit_emissive.z)) > 1.0e-4f)\n"
"      {\n"
"        /* Only strongly guided emissive hits get the extra rebalance. */\n"
"        const float emissive_pdf_gate_bias = 1.2f;\n"
"        const float emissive_pdf_gate_scale = 1.0f;\n"
"        const float emissive_hit_rebalance = 7.0f;\n"
"        const float emissive_pdf_gate = saturate((emissive_sample.w - emissive_pdf_gate_bias) /\n"
"                                                emissive_pdf_gate_scale);\n"
"        sample_weight = mix(sample_weight,\n"
"                            emissive_balance_weight * emissive_hit_rebalance,\n"
"                            emissive_pdf_gate);\n"
"      }\n"
"      float3 hit_N = float3(0.0f);\n"
"      const bool need_hit_normal = reflector_eligible ||\n"
"                                   (hit_has_diffuse &&\n"
"                                    ((uniforms.emissive_light_count_pad.y > 0 &&\n"
"                                      uniforms.emissive_light_count_pad.z > 0) ||\n"
"                                     sample_index < diffuse_continuation_count));\n"
"      if (need_hit_normal) {\n"
"        hit_N = fast_gi_hit_normal(\n"
"            user_id, intersection.primitive_id, trace_dir, triangle_normals, triangle_normal_ranges);\n"
"      }\n"
"      if (reflector_eligible && dot(hit_N, hit_N) > 1.0e-10f) {\n"
"        sample_throughput *= reflection_tint;\n"
"        if (!(dot(sample_throughput, sample_throughput) > 1.0e-10f)) {\n"
"          sample_weight = 0.0f;\n"
"          break;\n"
"        }\n"
"        const float3 reflected_dir = sample_fast_gi_reflection_direction(\n"
"            tid, sample_index, specular_redirects + 1, trace_dir, hit_N, reflection_roughness, uniforms);\n"
"        if (!(dot(reflected_dir, reflected_dir) > 1.0e-10f)) {\n"
"          sample_weight = 0.0f;\n"
"          break;\n"
"        }\n"
"        trace_origin = hit_P + hit_N * max(normal_bias, ray_tmin);\n"
"        trace_dir = reflected_dir;\n"
"        trace_tmax = max(ray_tmax, shell_distance * 1.5f);\n"
"        specular_redirects++;\n"
"        continue;\n"
"      }\n"
"      const int remaining_gi_bounces = max(gi_bounces - specular_redirects, 0);\n"
"      if (hit_has_diffuse && uniforms.emissive_light_count_pad.y > 0 &&\n"
"          uniforms.emissive_light_count_pad.z > 0)\n"
"      {\n"
"        sample_radiance += sample_throughput * hit_albedo * sample_fast_gi_direct_light(\n"
"            tid, sample_index, sample_count, hit_P, hit_N, normal_bias, scene, fast_gi_lights, uniforms);\n"
"      }\n"
"      if (remaining_gi_bounces > 1) {\n"
"        const float4 hit_continuation = sample_fast_gi_next_cascade_continuation(\n"
"            fast_gi_history_tx, hit_P, cascade_index, uniforms);\n"
"        sample_radiance += sample_throughput * hit_albedo * hit_continuation.xyz *\n"
"                          (0.95f * hit_continuation.w);\n"
"      }\n"
"      const bool allow_receiver_continuation = (sample_index < diffuse_continuation_count) &&\n"
"                                             max(emissive_radiance[user_id].x,\n"
"                                                 max(emissive_radiance[user_id].y,\n"
"                                                     emissive_radiance[user_id].z)) <= 1.0e-4f &&\n"
"                                             hit_has_diffuse &&\n"
"                                             (((remaining_gi_bounces > 1) &&\n"
"                                               (cascade_index + 1 < cascade_count)) ||\n"
"                                              allow_environment_diffuse_continuation);\n"
"      if (allow_receiver_continuation) {\n"
"        const float3 bounce_dir = sample_fast_gi_diffuse_direction(\n"
"            tid, sample_index, specular_redirects + 1, hit_N, uniforms);\n"
"        const float3 bounce_origin = hit_P + hit_N * normal_bias;\n"
"        const float bounce_tmin = max(5.0e-4f, normal_bias * 0.5f);\n"
"        const float bounce_tmax = max(trace_tmax, shell_distance * 1.5f);\n"
"        intersection_result<triangle_data, instancing, max_levels<2>> bounce_intersection = i.intersect(\n"
"            ray(bounce_origin, bounce_dir, bounce_tmin, bounce_tmax), scene);\n"
"        if (bounce_intersection.type == intersection_type::triangle) {\n"
"          if (remaining_gi_bounces > 1) {\n"
"            const uint bounce_user_id = bounce_intersection.user_instance_id[0];\n"
"            const float3 bounce_hit_P = bounce_origin + bounce_dir * bounce_intersection.distance;\n"
"            float3 continuation_radiance = max(emissive_radiance[bounce_user_id].xyz, float3(0.0f));\n"
"            const float3 bounce_albedo = max(diffuse_albedo[bounce_user_id].xyz, float3(0.0f));\n"
"            const float4 bounce_continuation = sample_fast_gi_next_cascade_continuation(\n"
"                fast_gi_history_tx, bounce_hit_P, cascade_index, uniforms);\n"
"            continuation_radiance += bounce_albedo * bounce_continuation.xyz *\n"
"                                     (0.7f * bounce_continuation.w);\n"
"            sample_radiance += sample_throughput * hit_albedo * continuation_radiance * 0.35f;\n"
"          }\n"
"        }\n"
"        else if (cascade_index + 1 < cascade_count) {\n"
"          const float3 bounce_far_P = bounce_origin + bounce_dir * bounce_tmax;\n"
"          const float4 bounce_escape = sample_fast_gi_next_cascade_continuation(\n"
"              fast_gi_history_tx, bounce_far_P, cascade_index, uniforms);\n"
"          if (bounce_escape.w > 1.0e-4f) {\n"
"            sample_radiance += sample_throughput * hit_albedo * bounce_escape.xyz *\n"
"                              (0.18f * bounce_escape.w);\n"
"          }\n"
"          else {\n"
"            sample_radiance += sample_throughput * hit_albedo *\n"
"                              sample_fast_gi_world_radiance(world_probe_tx, bounce_dir, uniforms) *\n"
"                              0.18f;\n"
"          }\n"
"        }\n"
"        else {\n"
"          sample_radiance += sample_throughput * hit_albedo *\n"
"                            sample_fast_gi_world_radiance(world_probe_tx, bounce_dir, uniforms) *\n"
"                            0.18f;\n"
"        }\n"
"      }\n"
"      break;\n"
"    }\n"
"    accum += sample_radiance * sample_weight;\n"
"    const float sample_luma = dot(sample_radiance, float3(0.2126f, 0.7152f, 0.0722f));\n"
"    accum_luma += sample_luma * sample_weight;\n"
"    accum_luma_sq += sample_luma * sample_luma * sample_weight;\n"
"    accum_weight += sample_weight;\n"
"  }\n"
"  const float occupancy = saturate(occupancy_accum / float(sample_count));\n"
"  const float thickness = saturate(thickness_accum / max(occupancy_accum, 1.0e-4f));\n"
"  const float openness = saturate(openness_accum / float(sample_count));\n"
"  const float4 visibility_target = float4(occupancy, thickness, openness, 0.0f);\n"
"  if (accum_weight <= 1.0e-4f) {\n"
"    fast_gi_img.write(reuse_history ? history * 0.985f : float4(0.0f), atlas_voxel);\n"
"    fast_gi_error_img.write(reuse_history ? history_error * 0.985f : 1.0f, atlas_voxel);\n"
"    fast_gi_visibility_img.write(mix(history_visibility, visibility_target, reuse_history ? 0.25f : 1.0f), atlas_voxel);\n"
"    return;\n"
"  }\n"
"  const float confidence = saturate(accum_weight / float(sample_count));\n"
"  const float base_history_blend = (cascade_index == 0) ? 0.28f : ((cascade_index == 1) ? 0.20f : 0.12f);\n"
"  const float sample_bonus = min(0.08f, max(float(sample_count - 4), 0.0f) * 0.004f);\n"
"  const float history_blend = reuse_history ? min(base_history_blend + sample_bonus, 0.42f) : 1.0f;\n"
"  const float3 averaged_radiance = accum / accum_weight;\n"
"  const float mean_luma = accum_luma / accum_weight;\n"
"  const float variance_luma = max(accum_luma_sq / accum_weight - mean_luma * mean_luma, 0.0f);\n"
"  const float normalized_error = min(sqrt(variance_luma) / max(mean_luma, 1.0e-3f), 8.0f);\n"
"  const float4 target = float4(averaged_radiance * confidence, confidence);\n"
"  fast_gi_img.write(mix(history, target, history_blend), atlas_voxel);\n"
"  fast_gi_error_img.write(mix(history_error, normalized_error, history_blend), atlas_voxel);\n"
"  fast_gi_visibility_img.write(mix(history_visibility, visibility_target, history_blend), atlas_voxel);\n"
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
"  if (!(ray_time_img.read(tid).x > 0.0f)) {\n"
"    receiver_gi_img.write(float4(0.0f), tid);\n"
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
"    receiver_gi_img.write(float4(0.0f), tid);\n"
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
"      incoming += max(emissive_radiance[user_id].xyz, float3(0.0f));\n"
"      const float3 hit_albedo = max(diffuse_albedo[user_id].xyz, float3(0.0f));\n"
"      if (max(hit_albedo.x, max(hit_albedo.y, hit_albedo.z)) > 1.0e-4f) {\n"
"        const float3 hit_N = fast_gi_hit_normal(\n"
"            user_id, intersection.primitive_id, sample_dir, triangle_normals, triangle_normal_ranges);\n"
"        const float3 hit_P = origin + sample_dir * intersection.distance;\n"
"        const float hit_luma = dot(hit_albedo, float3(0.2126f, 0.7152f, 0.0722f));\n"
"        const float3 soft_hit_albedo = mix(hit_albedo, float3(hit_luma), 0.35f);\n"
"        incoming += soft_hit_albedo * sample_reflected_receiver_gi_direct_light(tid,\n"
"                                                                          sample_index,\n"
"                                                                          sample_count,\n"
"                                                                          hit_P,\n"
"                                                                          hit_N,\n"
"                                                                          receiver_gi_lights,\n"
"                                                                          uniforms) * (receiver_is_reflection ? 0.10f : 0.16f);\n"
"        incoming += soft_hit_albedo * sample_reflected_receiver_gi_world_radiance(\n"
"                                         world_probe_tx, hit_N, uniforms) * (receiver_is_reflection ? 0.05f : 0.07f);\n"
"      }\n"
"    }\n"
"    else {\n"
"      incoming = sample_reflected_receiver_gi_world_radiance(world_probe_tx, sample_dir, uniforms) *\n"
"                 (receiver_is_reflection ? 0.10f : 0.16f);\n"
"    }\n"
"    accum += reflected_receiver_gi_luma_clamp(incoming, receiver_is_reflection ? 0.6f : 0.9f);\n"
"  }\n"
"  receiver_gi_img.write(float4(reflected_receiver_gi_luma_clamp(accum / float(sample_count),\n"
"                                                                receiver_is_reflection ? 0.35f : 0.65f),\n"
"                               1.0f),\n"
"                        tid);\n"
"}\n"
         "kernel void eevee_hardware_trace_local_shadow(\n"
         "    uint2 tid [[thread_position_in_grid]],\n"
         "    instance_acceleration_structure scene [[buffer(0)]],\n"
         "    constant HardwareLocalShadowUniforms &uniforms [[buffer(1)]],\n"
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
         "  intersector<triangle_data, instancing, max_levels<2>> i;\n"
         "  i.assume_geometry_type(geometry_type::triangle);\n"
         "  i.force_opacity(forced_opacity::opaque);\n"
         "  const bool area_soft = is_area_light(uint(uniforms.resolution_layer_type.w)) && (max(uniforms.light_x_axis_size_x.w, uniforms.light_y_axis_size_y.w) * uniforms.shadow_offset_scale.w > 1.0e-6f);\n"
         "  const bool local_soft = (!is_area_light(uint(uniforms.resolution_layer_type.w))) && (uniforms.light_position_radius.w > 1.0e-6f);\n"
         "  const int sample_count = (area_soft || local_soft) ? max(int(uniforms.normal_bias_pad.y), 1) : 1;\n"
         "  float visibility = 0.0f;\n"
         "  for (int sample_index = 0; sample_index < sample_count; sample_index++) {\n"
         "    const float3 target = sample_local_shadow_target(tid, sample_index, P, uniforms);\n"
         "    float3 sample_L = target - P;\n"
         "    const float sample_distance = length(sample_L);\n"
         "    if (!(sample_distance > 1.0e-5f)) {\n"
         "      visibility += 1.0f;\n"
         "      continue;\n"
         "    }\n"
         "    sample_L /= sample_distance;\n"
"    const float ray_tmax = max(ray_tmin, sample_distance);\n"
"    const float3 origin = P + N * normal_bias;\n"
         "    intersection_result<triangle_data, instancing, max_levels<2>> intersection = i.intersect(ray(origin, sample_L, ray_tmin, ray_tmax), scene);\n"
         "    visibility += (intersection.type == intersection_type::triangle) ? 0.0f : 1.0f;\n"
         "  }\n"
         "  visibility /= float(sample_count);\n"
         "  shadow_visibility_img.write(float4(visibility), tid, uint(uniforms.resolution_layer_type.z));\n"
         "}\n"
         "kernel void eevee_hardware_trace_local_hit_shadow(\n"
         "    uint3 threadgroup_id [[threadgroup_position_in_grid]],\n"
         "    uint3 local_id [[thread_position_in_threadgroup]],\n"
         "    instance_acceleration_structure scene [[buffer(0)]],\n"
         "    constant HardwareLocalShadowUniforms &uniforms [[buffer(1)]],\n"
         "    constant uint *tiles_coord_buf [[buffer(2)]],\n"
"    constant float4 *triangle_normals [[buffer(3)]],\n"
"    constant TriangleNormalRange *triangle_normal_ranges [[buffer(4)]],\n"
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
         "  intersector<triangle_data, instancing, max_levels<2>> i;\n"
         "  i.assume_geometry_type(geometry_type::triangle);\n"
         "  i.force_opacity(forced_opacity::opaque);\n"
         "  const bool area_soft = is_area_light(uint(uniforms.resolution_layer_type.w)) && (max(uniforms.light_x_axis_size_x.w, uniforms.light_y_axis_size_y.w) * uniforms.shadow_offset_scale.w > 1.0e-6f);\n"
         "  const bool local_soft = (!is_area_light(uint(uniforms.resolution_layer_type.w))) && (uniforms.light_position_radius.w > 1.0e-6f);\n"
         "  const int sample_count = (area_soft || local_soft) ? max(int(uniforms.normal_bias_pad.y), 1) : 1;\n"
         "  float visibility = 0.0f;\n"
         "  for (int sample_index = 0; sample_index < sample_count; sample_index++) {\n"
         "    const float3 target = sample_local_shadow_target(tid, sample_index, P, uniforms);\n"
         "    float3 sample_L = target - P;\n"
         "    const float sample_distance = length(sample_L);\n"
         "    if (!(sample_distance > 1.0e-5f)) {\n"
         "      visibility += 1.0f;\n"
         "      continue;\n"
         "    }\n"
         "    sample_L /= sample_distance;\n"
         "    const float ray_tmax = max(ray_tmin, sample_distance);\n"
"    const float3 origin = P + shadow_N * normal_bias;\n"
         "    intersection_result<triangle_data, instancing, max_levels<2>> intersection = i.intersect(ray(origin, sample_L, ray_tmin, ray_tmax), scene);\n"
         "    visibility += (intersection.type == intersection_type::triangle) ? 0.0f : 1.0f;\n"
         "  }\n"
         "  visibility /= float(sample_count);\n"
         "  shadow_visibility_img.write(float4(visibility), tid, uint(uniforms.resolution_layer_type.z));\n"
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

static id<MTLComputePipelineState> get_hardware_fast_gi_pipeline(id<MTLDevice> device)
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
  id<MTLFunction> function = [library newFunctionWithName:@"eevee_hardware_trace_fast_gi"];
  if (function == nil) {
    return nil;
  }

  pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  [function release];
  if (pipeline == nil && error != nil) {
    fprintf(stderr, "Metal RT Fast GI pipeline creation failed: %s\n",
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

  NSError *error = nil;
  id<MTLFunction> function = [library
      newFunctionWithName:@"eevee_hardware_trace_reflected_receiver_gi"];
  if (function == nil) {
    return nil;
  }

  pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  [function release];
  if (pipeline == nil && error != nil) {
    fprintf(stderr,
            "Metal RT reflected receiver GI pipeline creation failed: %s\n",
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

GPUMetalRaytraceScene *raytrace_scene_build(Span<GPUMetalRaytraceSceneEntry> entries,
                                            GPUMetalRaytraceSceneStats *r_stats)
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

    GPUMetalRaytraceScene *scene = new GPUMetalRaytraceScene();
    for (const GPUMetalRaytraceSceneEntry &entry : entries) {
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
                   "EEVEE HWRT perf metal_scene_build geometries=%d instances=%d built_blas=%d emissive_lights=%d elapsed_ms=%.2f\n",
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

bool raytrace_scene_update(GPUMetalRaytraceScene *scene,
                           Span<GPUMetalRaytraceSceneEntry> entries,
                           const GPUMetalRaytraceSceneUpdateParams &update_params,
                           GPUMetalRaytraceSceneStats *r_stats)
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
      const GPUMetalRaytraceSceneEntry &entry = entries[i];
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
                   "EEVEE HWRT perf metal_scene_update tlas=%d emissive=%d material=%d world_geom=%d geometries=%d instances=%d elapsed_ms=%.2f\n",
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

bool raytrace_scene_trace(GPUMetalRaytraceScene *scene, const GPUMetalRaytraceTraceParams &params)
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
    uniforms.closure_index = std::max(params.closure_index, 0);
    uniforms.feature_mask = params.feature_mask;
    uniforms.hardware_trace_phase = params.hardware_trace_phase;
    uniforms.reflection_bounces = std::max(params.reflection_bounces, 1);
    uniforms.refraction_bounces = std::max(params.refraction_bounces, 1);
    uniforms.resolution_bias = params.resolution_bias;
    uniforms.clamp_indirect = std::max(params.clamp_indirect, 0.0f);
    uniforms.world_probe_atlas_coord = params.world_probe_atlas_coord;
    uniforms.use_environment_pad = int4((params.use_environment && world_probe_tx != nullptr) ? 1 : 0,
                                        std::max(scene->emissive_light_count, 0),
                                        0,
                                        0);
    uniforms.sampling_rand = params.sampling_rand;

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
    [encoder setBuffer:scene->emissive_radiance_buffer offset:0 atIndex:2];
    [encoder setBuffer:scene->diffuse_albedo_buffer offset:0 atIndex:3];
    [encoder setBuffer:scene->material_proxy_buffer offset:0 atIndex:4];
    [encoder setBuffer:scene->triangle_normal_buffer offset:0 atIndex:5];
    [encoder setBuffer:scene->triangle_normal_range_buffer offset:0 atIndex:6];
    [encoder setBuffer:tiles_coord_handle offset:0 atIndex:7];
    [encoder setBuffer:scene->triangle_smooth_normal_buffer offset:0 atIndex:8];
    [encoder setBuffer:scene->triangle_local_position_buffer offset:0 atIndex:9];
    [encoder setBuffer:scene->emissive_light_buffer offset:0 atIndex:10];
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

bool raytrace_scene_shadow_batch_begin(GPUMetalRaytraceScene *scene)
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

bool raytrace_scene_shadow_batch_end(GPUMetalRaytraceScene *scene)
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

bool raytrace_scene_trace_directional_shadow(GPUMetalRaytraceScene *scene,
                                             const GPUMetalRaytraceDirectionalShadowParams &params)
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
    uniforms.shadow_params = float4(
        std::max(params.shadow_angle, 0.0f), float(std::max(params.sample_count, 1)), 0.0f, 0.0f);
    uniforms.world_sun_slot_pad = int4(params.world_sun_slot, 0, 0, 0);
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
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceDirectionalHitShadowParams &params)
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
    uniforms.shadow_params = float4(
        std::max(params.shadow_angle, 0.0f), float(std::max(params.sample_count, 1)), 0.0f, 0.0f);
    uniforms.world_sun_slot_pad = int4(params.world_sun_slot, 0, 0, 0);
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
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceEnvironmentVisibilityParams &params)
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
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceHitEnvironmentVisibilityParams &params)
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

bool raytrace_scene_trace_fast_gi(GPUMetalRaytraceScene *scene,
                                  const GPUMetalRaytraceFastGIParams &params)
{
  if (scene == nullptr || scene->top_level_acceleration_structure == nil ||
      params.fast_gi_history_tx == nullptr || params.fast_gi_tx == nullptr ||
      params.fast_gi_error_tx == nullptr || params.fast_gi_visibility_tx == nullptr)
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

    id<MTLComputePipelineState> pipeline = get_hardware_fast_gi_pipeline(ctx->device);
    if (pipeline == nil) {
      return false;
    }

    MTLTexture *fast_gi_history_tx = unwrap(params.fast_gi_history_tx);
    MTLTexture *fast_gi_tx = unwrap(params.fast_gi_tx);
    MTLTexture *fast_gi_error_tx = unwrap(params.fast_gi_error_tx);
    MTLTexture *fast_gi_visibility_tx = unwrap(params.fast_gi_visibility_tx);
    MTLTexture *world_probe_tx = unwrap(params.world_probe_tx);
    if (fast_gi_history_tx == nullptr || fast_gi_tx == nullptr || fast_gi_error_tx == nullptr ||
        fast_gi_visibility_tx == nullptr)
    {
      return false;
    }

    MTLStorageBuf *fast_gi_light_ssbo = (params.light_buf != nullptr) ?
                                            static_cast<MTLStorageBuf *>(params.light_buf) :
                                            nullptr;
    id<MTLBuffer> fast_gi_light_handle = (fast_gi_light_ssbo != nullptr) ?
                                             fast_gi_light_ssbo->get_metal_buffer() :
                                             nil;

    HardwareFastGIUniforms uniforms = {};
    for (int i = 0; i < 3; i++) {
      uniforms.cascade_config[i] = params.cascade_config[i];
    }
    uniforms.grid_cascade_samples = int4(std::max(params.grid_resolution, 1),
                                         std::max(params.cascade_index, 0),
                                         std::max(params.cascade_count, 1),
                                         std::max(params.sample_count, 1));
    uniforms.brick_origin_pad = int4(
        std::max(params.brick_origin.x, 0),
        std::max(params.brick_origin.y, 0),
        std::max(params.brick_origin.z, 0),
        0);
    uniforms.brick_extent_pad = int4(
        std::max(params.brick_extent.x, 1),
        std::max(params.brick_extent.y, 1),
        std::max(params.brick_extent.z, 1),
        0);
    uniforms.normal_bias_pad = float4(std::max(params.normal_bias, 1.0e-4f), 0.0f, 0.0f, 0.0f);
    uniforms.reuse_history_pad = int4(params.reuse_history ? 1 : 0, 0, 0, 0);
    uniforms.sampling_rand = params.sampling_rand;
    uniforms.emissive_light_count_pad = int4(std::max(scene->emissive_light_count, 0),
                                             std::max(params.light_count, 0),
                                             std::max(params.light_sample_count, 0),
                                             0);
    uniforms.world_probe_atlas_coord = params.world_probe_atlas_coord;
    uniforms.gi_environment_pad = int4(std::max(params.gi_bounces, 1),
                                       params.use_environment ? 1 : 0,
                                       0,
                                       0);

    id<MTLCommandBuffer> command_buffer = [ctx->queue commandBuffer];
    NSMutableArray *retained_resources = retained_resources_for_command_buffer(command_buffer,
                                                                               "Metal RT fast GI");
    retain_scene_resources(scene, retained_resources);
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return false;
    }

    [encoder setComputePipelineState:pipeline];
    encoder_use_scene_geometry_resources(encoder, scene);
    encoder_use_scene_shading_resources(encoder, scene);
    if (fast_gi_light_handle != nil) {
      [encoder useResource:fast_gi_light_handle usage:MTLResourceUsageRead];
    }
    [encoder setAccelerationStructure:scene->top_level_acceleration_structure atBufferIndex:0];
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:1];
    [encoder setBuffer:scene->emissive_radiance_buffer offset:0 atIndex:2];
    [encoder setBuffer:scene->diffuse_albedo_buffer offset:0 atIndex:3];
    [encoder setBuffer:scene->material_proxy_buffer offset:0 atIndex:4];
    [encoder setBuffer:scene->emissive_light_buffer offset:0 atIndex:5];
    [encoder setBuffer:scene->triangle_normal_buffer offset:0 atIndex:6];
    [encoder setBuffer:scene->triangle_normal_range_buffer offset:0 atIndex:7];
    [encoder setBuffer:fast_gi_light_handle offset:0 atIndex:8];
    id<MTLTexture> history_handle = fast_gi_history_tx->get_metal_handle();
    id<MTLTexture> output_handle = fast_gi_tx->get_metal_handle();
    id<MTLTexture> error_handle = fast_gi_error_tx->get_metal_handle();
    id<MTLTexture> visibility_handle = fast_gi_visibility_tx->get_metal_handle();
    id<MTLTexture> world_probe_handle = world_probe_tx ? world_probe_tx->get_metal_handle() : nil;
    if (history_handle == nil || output_handle == nil || error_handle == nil ||
        visibility_handle == nil)
    {
      [encoder endEncoding];
      return false;
    }
    [encoder setTexture:history_handle atIndex:0];
    [encoder setTexture:output_handle atIndex:1];
    [encoder setTexture:error_handle atIndex:2];
    [encoder setTexture:visibility_handle atIndex:3];
    [encoder setTexture:world_probe_handle atIndex:4];

    const MTLSize grid_size = MTLSizeMake(std::max<NSUInteger>(1, params.brick_extent.x),
                                          std::max<NSUInteger>(1, params.brick_extent.y),
                                          std::max<NSUInteger>(1, params.brick_extent.z));
    const MTLSize group_size = MTLSizeMake(4, 4, 4);
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

bool raytrace_scene_trace_reflected_receiver_gi(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceReflectedReceiverGIParams &params)
{
  if (scene == nullptr || scene->top_level_acceleration_structure == nil ||
      params.receiver_gi_tx == nullptr || params.dispatch_buf == nullptr ||
      params.tiles_coord_buf == nullptr || params.ray_time_tx == nullptr ||
      params.hit_albedo_tx == nullptr || params.hit_material_tx == nullptr ||
      params.hit_normal_tx == nullptr || params.hit_world_position_tx == nullptr)
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

    id<MTLComputePipelineState> pipeline = get_hardware_reflected_receiver_gi_pipeline(
        ctx->device);
    if (pipeline == nil) {
      return false;
    }

    MTLTexture *receiver_gi_tx = unwrap(params.receiver_gi_tx);
    MTLTexture *world_probe_tx = unwrap(params.world_probe_tx);
    MTLTexture *ray_time_tx = unwrap(params.ray_time_tx);
    MTLTexture *hit_albedo_tx = unwrap(params.hit_albedo_tx);
    MTLTexture *hit_material_tx = unwrap(params.hit_material_tx);
    MTLTexture *hit_normal_tx = unwrap(params.hit_normal_tx);
    MTLTexture *hit_world_position_tx = unwrap(params.hit_world_position_tx);
    MTLStorageBuf *dispatch_ssbo = static_cast<MTLStorageBuf *>(params.dispatch_buf);
    MTLStorageBuf *tiles_coord_ssbo = static_cast<MTLStorageBuf *>(params.tiles_coord_buf);
    if (receiver_gi_tx == nullptr || ray_time_tx == nullptr || hit_albedo_tx == nullptr ||
        hit_material_tx == nullptr || hit_normal_tx == nullptr ||
        hit_world_position_tx == nullptr || dispatch_ssbo == nullptr || tiles_coord_ssbo == nullptr)
    {
      return false;
    }

    HardwareReflectedReceiverGIUniforms uniforms = {};
    uniforms.resolution_samples = int4(params.tracing_resolution.x,
                                       params.tracing_resolution.y,
                                       std::max(params.resolution_divisor, 1),
                                       std::max(params.sample_count, 1));
    uniforms.normal_bias_pad = float4(std::max(params.normal_bias, 1.0e-4f), 0.0f, 0.0f, 0.0f);
    uniforms.environment_pad = int4(params.use_environment ? 1 : 0, 0, 0, 0);
    uniforms.light_count_pad = int4(std::max(params.light_count, 0),
                                    std::max(params.light_sample_count, 0),
                                    0,
                                    0);
    uniforms.sampling_rand = params.sampling_rand;
    uniforms.world_probe_atlas_coord = params.world_probe_atlas_coord;

    MTLStorageBuf *light_ssbo = (params.light_buf != nullptr) ?
                                    static_cast<MTLStorageBuf *>(params.light_buf) :
                                    nullptr;
    id<MTLBuffer> light_handle = (light_ssbo != nullptr) ? light_ssbo->get_metal_buffer() : nil;
    id<MTLBuffer> dispatch_buf_handle = dispatch_ssbo->get_metal_buffer();
    id<MTLBuffer> tiles_coord_handle = tiles_coord_ssbo->get_metal_buffer();
    if (dispatch_buf_handle == nil || tiles_coord_handle == nil) {
      return false;
    }

    id<MTLCommandBuffer> command_buffer = [ctx->queue commandBuffer];
    NSMutableArray *retained_resources = retained_resources_for_command_buffer(
        command_buffer, "Metal RT reflected receiver GI");
    retain_scene_resources(scene, retained_resources);
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return false;
    }

    [encoder setComputePipelineState:pipeline];
    encoder_use_scene_geometry_resources(encoder, scene);
    encoder_use_scene_shading_resources(encoder, scene);
    if (light_handle != nil) {
      [encoder useResource:light_handle usage:MTLResourceUsageRead];
    }
    [encoder useResource:dispatch_buf_handle usage:MTLResourceUsageRead];
    [encoder useResource:tiles_coord_handle usage:MTLResourceUsageRead];
    [encoder setAccelerationStructure:scene->top_level_acceleration_structure atBufferIndex:0];
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:1];
    [encoder setBuffer:scene->emissive_radiance_buffer offset:0 atIndex:2];
    [encoder setBuffer:scene->diffuse_albedo_buffer offset:0 atIndex:3];
    [encoder setBuffer:scene->triangle_normal_buffer offset:0 atIndex:4];
    [encoder setBuffer:scene->triangle_normal_range_buffer offset:0 atIndex:5];
    [encoder setBuffer:light_handle offset:0 atIndex:6];
    [encoder setBuffer:tiles_coord_handle offset:0 atIndex:7];
    id<MTLTexture> receiver_gi_handle = receiver_gi_tx->get_metal_handle();
    id<MTLTexture> world_probe_handle = world_probe_tx ? world_probe_tx->get_metal_handle() : nil;
    id<MTLTexture> ray_time_handle = ray_time_tx->get_metal_handle();
    id<MTLTexture> hit_albedo_handle = hit_albedo_tx->get_metal_handle();
    id<MTLTexture> hit_material_handle = hit_material_tx->get_metal_handle();
    id<MTLTexture> hit_normal_handle = hit_normal_tx->get_metal_handle();
    id<MTLTexture> hit_world_position_handle = hit_world_position_tx->get_metal_handle();
    if (receiver_gi_handle == nil || ray_time_handle == nil || hit_albedo_handle == nil ||
        hit_material_handle == nil || hit_normal_handle == nil || hit_world_position_handle == nil)
    {
      [encoder endEncoding];
      return false;
    }
    [encoder setTexture:receiver_gi_handle atIndex:0];
    [encoder setTexture:world_probe_handle atIndex:1];
    [encoder setTexture:ray_time_handle atIndex:2];
    [encoder setTexture:hit_albedo_handle atIndex:3];
    [encoder setTexture:hit_normal_handle atIndex:4];
    [encoder setTexture:hit_world_position_handle atIndex:5];
    [encoder setTexture:hit_material_handle atIndex:6];

    const MTLSize group_size = MTLSizeMake(8, 8, 1);
    [encoder dispatchThreadgroupsWithIndirectBuffer:dispatch_buf_handle
                               indirectBufferOffset:0
                              threadsPerThreadgroup:group_size];
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

bool raytrace_scene_trace_local_shadow(GPUMetalRaytraceScene *scene,
                                       const GPUMetalRaytraceLocalShadowParams &params)
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
    uniforms.normal_bias_pad = float4(
        std::max(params.normal_bias, 0.0f), float(std::max(params.sample_count, 1)), 0.0f, 0.0f);
    uniforms.sampling_rand = params.sampling_rand;

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
    [encoder setAccelerationStructure:scene->top_level_acceleration_structure atBufferIndex:0];
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:1];
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

bool raytrace_scene_trace_local_hit_shadow(GPUMetalRaytraceScene *scene,
                                           const GPUMetalRaytraceLocalHitShadowParams &params)
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
    uniforms.normal_bias_pad = float4(
        std::max(params.normal_bias, 0.0f), float(std::max(params.sample_count, 1)), 0.0f, 0.0f);
    uniforms.sampling_rand = params.sampling_rand;

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
    [encoder useResource:scene->triangle_normal_range_buffer usage:MTLResourceUsageRead];
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

void raytrace_scene_free(GPUMetalRaytraceScene *scene)
{
  delete scene;
}

}  // namespace blender::gpu::metal
