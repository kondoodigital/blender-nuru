/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup eevee
 *
 * The ray-tracing module class handles ray generation, scheduling, tracing and denoising.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>

#include "BLI_listbase.h"
#include "BLI_time.h"
#include "MEM_guardedalloc.h"

#include "GPU_batch.hh"
#include "GPU_debug.hh"
#include "GPU_material.hh"
#include "GPU_metal_raytrace.hh"
#include "GPU_storage_buffer.hh"
#include "GPU_state.hh"
#include "GPU_texture.hh"
#include "GPU_vertex_buffer.hh"

#include "DNA_ID.h"

#include "gpu_shader_private.hh"

#include "DEG_depsgraph_query.hh"

#include "eevee_camera.hh"
#include "eevee_instance.hh"
#include "eevee_raytrace.hh"

namespace blender::eevee {

static int hardware_indirect_gi_resolution_sanitize(const int value);

static RaytraceEEVEE_SpecularMode sanitize_specular_mode(const int value)
{
  return (value >= RAYTRACE_EEVEE_SPECULAR_MODE_OFF &&
          value <= RAYTRACE_EEVEE_SPECULAR_MODE_AUTO) ?
             RaytraceEEVEE_SpecularMode(value) :
             RAYTRACE_EEVEE_SPECULAR_MODE_OFF;
}

static RaytraceEEVEE_SpecularMode sanitize_reflection_mode(const int value)
{
  const RaytraceEEVEE_SpecularMode mode = sanitize_specular_mode(value);
  return (mode == RAYTRACE_EEVEE_SPECULAR_MODE_OFF) ? RAYTRACE_EEVEE_SPECULAR_MODE_OFF :
                                                      RAYTRACE_EEVEE_SPECULAR_MODE_FULL_RT;
}

static bool hardware_fast_gi_stats_logging_enabled()
{
  const char *value = std::getenv("BLENDER_EEVEE_HWRT_FAST_GI_STATS");
  return (value != nullptr) && (value[0] != '\0') && !(value[0] == '0' && value[1] == '\0');
}

static bool hardware_fast_gi_debug_overlay_enabled()
{
  const char *value = std::getenv("BLENDER_EEVEE_HWRT_FAST_GI_DEBUG_VIEW");
  return (value != nullptr) && (value[0] != '\0') && !(value[0] == '0' && value[1] == '\0');
}

static bool hardware_fast_gi_freeze_updates_enabled()
{
  const char *value = std::getenv("BLENDER_EEVEE_HWRT_FAST_GI_FREEZE_UPDATES");
  return (value != nullptr) && (value[0] != '\0') && !(value[0] == '0' && value[1] == '\0');
}

static int hardware_debug_view_mode()
{
  const char *value = std::getenv("BLENDER_EEVEE_HWRT_DEBUG_VIEW_MODE");
  if (value == nullptr || value[0] == '\0' || (value[0] == '0' && value[1] == '\0')) {
    return HWRT_DEBUG_VIEW_NONE;
  }
  if (std::strcmp(value, "radiance") == 0 || std::strcmp(value, "1") == 0) {
    return HWRT_DEBUG_VIEW_RADIANCE;
  }
  if (std::strcmp(value, "occupancy") == 0 || std::strcmp(value, "visibility") == 0 ||
      std::strcmp(value, "2") == 0)
  {
    return HWRT_DEBUG_VIEW_OCCUPANCY_THICKNESS;
  }
  if (std::strcmp(value, "confidence") == 0 || std::strcmp(value, "3") == 0) {
    return HWRT_DEBUG_VIEW_CONFIDENCE;
  }
  if (std::strcmp(value, "invalid") == 0 || std::strcmp(value, "4") == 0) {
    return HWRT_DEBUG_VIEW_INVALID_BRICKS;
  }
  if (std::strcmp(value, "leak") == 0 || std::strcmp(value, "5") == 0) {
    return HWRT_DEBUG_VIEW_LEAK_RISK;
  }
  if (std::strcmp(value, "direct") == 0 || std::strcmp(value, "6") == 0) {
    return HWRT_DEBUG_VIEW_DIRECT_LIGHT;
  }
  return HWRT_DEBUG_VIEW_NONE;
}

static int hardware_debug_isolate_mode()
{
  const char *value = std::getenv("BLENDER_EEVEE_HWRT_DEBUG_ISOLATE");
  if (value == nullptr || value[0] == '\0' || (value[0] == '0' && value[1] == '\0')) {
    return HWRT_DEBUG_ISOLATE_NONE;
  }
  if (std::strcmp(value, "direct") == 0 || std::strcmp(value, "1") == 0) {
    return HWRT_DEBUG_ISOLATE_DIRECT;
  }
  if (std::strcmp(value, "indirect") == 0 || std::strcmp(value, "2") == 0) {
    return HWRT_DEBUG_ISOLATE_INDIRECT;
  }
  return HWRT_DEBUG_ISOLATE_NONE;
}

static bool hardware_perf_logging_enabled()
{
  const char *value = std::getenv("BLENDER_EEVEE_HWRT_PERF");
  return (value != nullptr) && (value[0] != '\0') && !(value[0] == '0' && value[1] == '\0');
}

static bool batch_has_ssbo_attribute(gpu::Batch *batch, const char *shader_attr_name)
{
  const bool allow_default_uv_alias = std::strcmp(shader_attr_name, "eevee_default_uv_attr") == 0;
  for (int v = GPU_BATCH_VBO_MAX_LEN - 1; v > -1; v--) {
    gpu::VertBuf *vbo = batch->verts[v];
    if (vbo == nullptr) {
      continue;
    }
    const GPUVertFormat *format = GPU_vertbuf_get_format(vbo);
    for (uint attr_index = 0; attr_index < format->attr_len; attr_index++) {
      const GPUVertAttr *attr = &format->attrs[attr_index];
      for (uint name_index = 0; name_index < attr->name_len; name_index++) {
        const char *attr_name = GPU_vertformat_attr_name_get(format, attr, name_index);
        if (std::strcmp(attr_name, shader_attr_name) == 0) {
          return true;
        }
        if (allow_default_uv_alias && std::strcmp(attr_name, "a") == 0) {
          return true;
        }
      }
    }
  }
  return false;
}

static bool hardware_hit_eval_batch_compatible(gpu::Batch *batch, gpu::Shader *shader)
{
  if ((batch == nullptr) || (shader == nullptr) || (shader->interface == nullptr)) {
    return false;
  }

  const gpu::ShaderInterface *interface = shader->interface;
  if (interface->ssbo_attr_mask_ == 0) {
    return true;
  }

  uint16_t missing_attributes = interface->ssbo_attr_mask_;
  if (missing_attributes & (1 << GPU_SSBO_INDEX_BUF_SLOT)) {
    if (batch->elem != nullptr) {
      /* `GPU_batch_bind_as_resources()` will bind the element buffer as an SSBO whenever the batch
       * owns one. Require it to be initialized and non-empty here so sparse hit-eval replay fails
       * closed instead of walking into backend asserts such as Metal's `bind_as_ssbo()` guard. */
      if (batch->elem->is_init() && batch->elem->index_len_get() > 0) {
        missing_attributes &= ~(1 << GPU_SSBO_INDEX_BUF_SLOT);
      }
    }
    else if (batch->verts[0] != nullptr) {
      /* Procedural batches without an element list still bind the first vertex buffer to satisfy
       * the index SSBO slot and set `gpu_index_no_buffer = true`. */
      missing_attributes &= ~(1 << GPU_SSBO_INDEX_BUF_SLOT);
    }
  }

  const gpu::ShaderInput *ssbo_inputs = interface->inputs_ + interface->attr_len_ +
                                        interface->ubo_len_ + interface->uniform_len_;
  for (uint input_index = 0; input_index < interface->ssbo_len_; input_index++) {
    const gpu::ShaderInput &input = ssbo_inputs[input_index];
    if ((input.location < 0) || ((missing_attributes & (1 << input.location)) == 0)) {
      continue;
    }
    if (batch_has_ssbo_attribute(batch, interface->input_name_get(&input))) {
      missing_attributes &= ~(1 << input.location);
      if (missing_attributes == 0) {
        return true;
      }
    }
  }

  return missing_attributes == 0;
}

static bool hardware_fast_gi_screen_seed_allowed()
{
  const char *value = std::getenv("BLENDER_EEVEE_HWRT_FAST_GI_ALLOW_SCREEN_SEED");
  if (value != nullptr && value[0] != '\0') {
    return !((value[0] == '0') && (value[1] == '\0'));
  }
  /* Screen-seeded resolved radiance is only a support/debug fallback now. The traced Fast GI
   * producer owns light-object, emissive, and world transport directly and must stay independent
   * from RT shadow outputs and other resolved screen-space feature state by default. */
  return false;
}

static GPUMetalRaytraceFastGILightRecord hardware_fast_gi_light_record_from_light(
    const LightData &light)
{
  GPUMetalRaytraceFastGILightRecord record;
  record.object_to_world_x = light.object_to_world.x;
  record.object_to_world_y = light.object_to_world.y;
  record.object_to_world_z = light.object_to_world.z;
  record.color_diffuse_power = float4(
      std::abs(light.color.x), std::abs(light.color.y), std::abs(light.color.z), light.power[LIGHT_VOLUME]);
  record.direction_type = float4(0.0f, 0.0f, 1.0f, float(light.type));
  if (is_sun_light(light.type)) {
    record.direction_type = float4(light.sun().direction, float(light.type));
    record.attenuation_spot.x = light.sun().shape_radius;
  }
  else {
    record.attenuation_spot.x = light.local().local.shape_radius;
    record.attenuation_spot.y = light.local().local.influence_radius_invsqr_surface;
    if (is_spot_light(light.type)) {
      record.attenuation_spot.z = light.spot().spot_mul;
      record.attenuation_spot.w = light.spot().spot_bias;
      record.spot_size_inv = float4(
          light.spot().spot_size_inv.x, light.spot().spot_size_inv.y, 0.0f, 0.0f);
    }
  }
  return record;
}

static int hardware_fast_gi_direct_light_sample_count(const int light_count,
                                                      const bool is_viewport,
                                                      const int quality_tier)
{
  if (light_count <= 0) {
    return 0;
  }
  int sample_count = is_viewport ? 4 : 8;
  if (quality_tier >= 2) {
    sample_count += 4;
  }
  if (quality_tier >= 3 && !is_viewport) {
    sample_count += 4;
  }
  return clamp_i(sample_count, 1, is_viewport ? 8 : 16);
}

static bool hardware_viewport_interactive(const Instance &inst)
{
  return inst.is_viewport() && (inst.sampling.interactive_mode() || inst.is_transforming ||
                                inst.is_navigating || inst.is_painting || inst.is_playback);
}

static int hardware_interactive_resolution_scale(const Instance & /*inst*/,
                                                 const uint32_t /*feature_mask*/,
                                                 const int resolution_scale)
{
  /* Honor the configured ray-tracing resolution in every mode. Users can still opt into 1:2, 1:4,
   * and lower quality explicitly through the resolution setting. */
  return resolution_scale;
}

static constexpr int hardware_visibility_temporal_sample_count = 1;

static float4 hardware_shadow_sampling_rand(const Instance &inst)
{
  const float3 shadow_rng = inst.sampling.rng_3d_get(eSamplingDimension::SAMPLING_SHADOW_U);
  return float4(shadow_rng.x,
                shadow_rng.y,
                shadow_rng.z,
                inst.sampling.rng_get(eSamplingDimension::SAMPLING_SHADOW_X));
}

enum eHardwareGIProducerBackend {
  HWRT_GI_PRODUCER_BACKEND_METAL_RT = 0,
  HWRT_GI_PRODUCER_BACKEND_COMPUTE = 1,
};

static eHardwareGIProducerBackend hardware_gi_producer_backend(
    const GPUMetalRaytraceScene *metal_scene)
{
  return (metal_scene != nullptr) ? HWRT_GI_PRODUCER_BACKEND_METAL_RT :
                                    HWRT_GI_PRODUCER_BACKEND_COMPUTE;
}

static bool hardware_gi_trace_current_field(const eHardwareGIProducerBackend backend,
                                            GPUMetalRaytraceScene *metal_scene,
                                            const GPUMetalRaytraceFastGIParams &fast_gi_params)
{
  switch (backend) {
    case HWRT_GI_PRODUCER_BACKEND_METAL_RT: {
      if (metal_scene == nullptr) {
        return false;
      }
      return GPU_metal_raytrace_scene_trace_fast_gi(metal_scene, fast_gi_params);
    }
    case HWRT_GI_PRODUCER_BACKEND_COMPUTE:
      /* Reserved seam for a future GI-only compute producer path. */
      return false;
  }

  return false;
}

enum eHardwareAdaptiveQualityTier {
  HWRT_QUALITY_PERFORMANCE = 0,
  HWRT_QUALITY_BALANCED = 1,
  HWRT_QUALITY_HIGH = 2,
  HWRT_QUALITY_REFERENCE = 3,
};

enum eHardwareScenePriority {
  HWRT_SCENE_INTERIOR = 0,
  HWRT_SCENE_MIXED = 1,
  HWRT_SCENE_OPEN = 2,
};

enum eHardwareBudgetRebalance {
  HWRT_BUDGET_FAVOR_DIRECT = 0,
  HWRT_BUDGET_BALANCED = 1,
  HWRT_BUDGET_FAVOR_INDIRECT = 2,
};

static const char *hardware_quality_tier_name(const int tier)
{
  switch (tier) {
    case HWRT_QUALITY_PERFORMANCE:
      return "perf";
    case HWRT_QUALITY_HIGH:
      return "high";
    case HWRT_QUALITY_REFERENCE:
      return "ref";
    case HWRT_QUALITY_BALANCED:
    default:
      return "balanced";
  }
}

static const char *hardware_scene_priority_name(const int priority)
{
  switch (priority) {
    case HWRT_SCENE_INTERIOR:
      return "interior";
    case HWRT_SCENE_OPEN:
      return "open";
    case HWRT_SCENE_MIXED:
    default:
      return "mixed";
  }
}

static const char *hardware_budget_rebalance_name(const int mode)
{
  switch (mode) {
    case HWRT_BUDGET_FAVOR_DIRECT:
      return "direct";
    case HWRT_BUDGET_FAVOR_INDIRECT:
      return "indirect";
    case HWRT_BUDGET_BALANCED:
    default:
      return "balanced";
  }
}

static const char *hardware_debug_view_mode_name(const int mode)
{
  switch (mode) {
    case HWRT_DEBUG_VIEW_RADIANCE:
      return "radiance";
    case HWRT_DEBUG_VIEW_OCCUPANCY_THICKNESS:
      return "occupancy";
    case HWRT_DEBUG_VIEW_CONFIDENCE:
      return "confidence";
    case HWRT_DEBUG_VIEW_INVALID_BRICKS:
      return "invalid";
    case HWRT_DEBUG_VIEW_LEAK_RISK:
      return "leak";
    case HWRT_DEBUG_VIEW_DIRECT_LIGHT:
      return "direct";
    case HWRT_DEBUG_VIEW_NONE:
    default:
      return "off";
  }
}

static const char *hardware_debug_isolate_mode_name(const int mode)
{
  switch (mode) {
    case HWRT_DEBUG_ISOLATE_DIRECT:
      return "direct";
    case HWRT_DEBUG_ISOLATE_INDIRECT:
      return "indirect";
    case HWRT_DEBUG_ISOLATE_NONE:
    default:
      return "none";
  }
}

static int hardware_fast_gi_cascade_sample_count(const int cascade_index,
                                                 const bool is_viewport,
                                                 const bool interactive,
                                                 const int quality_tier)
{
  int sample_count = 0;
  if (is_viewport) {
    if (interactive) {
      switch (cascade_index) {
        case 0:
          sample_count = 4;
          break;
        case 1:
          sample_count = 8;
          break;
        default:
          sample_count = 12;
          break;
      }
    }
    else {
      switch (cascade_index) {
        case 0:
          sample_count = 8;
          break;
        case 1:
          sample_count = 16;
          break;
        default:
          sample_count = 24;
          break;
      }
    }
  }
  else {
    switch (cascade_index) {
      case 0:
        sample_count = 10;
        break;
      case 1:
        sample_count = 20;
        break;
      default:
        sample_count = 32;
        break;
    }
  }

  switch (quality_tier) {
    case HWRT_QUALITY_PERFORMANCE:
      sample_count = max_ii(2, int(std::ceil(float(sample_count) * 0.75f)));
      break;
    case HWRT_QUALITY_HIGH:
      sample_count = int(std::ceil(float(sample_count) * 1.25f));
      break;
    case HWRT_QUALITY_REFERENCE:
      sample_count = int(std::ceil(float(sample_count) * 1.5f));
      break;
    case HWRT_QUALITY_BALANCED:
    default:
      break;
  }
  return sample_count;
}

struct HardwareFastGIMemoryLayout {
  int grid_resolution = 1;
  int cascade_count = 1;
  int64_t budget_bytes = 0;
  int64_t requested_bytes = 0;
  int64_t allocated_bytes = 0;
  bool memory_limited = false;
};

struct HardwareFastGISceneScaleAnalysis {
  float3 scene_center = float3(0.0f);
  float scene_bounds_radius = 12.0f;
  float scene_radius = 12.0f;
  float forward_extent = 12.0f;
  float lateral_extent = 6.0f;
  float density = 0.0f;
  int active_entry_count = 0;
};

static int64_t hardware_fast_gi_memory_usage_bytes(const int grid_resolution, const int cascade_count)
{
  constexpr int64_t bytes_per_voxel = 8 + 2 + 8;
  return int64_t(max_ii(grid_resolution, 1)) * int64_t(max_ii(grid_resolution, 1)) *
         int64_t(max_ii(grid_resolution, 1)) * int64_t(max_ii(cascade_count, 1)) * bytes_per_voxel;
}

static int64_t hardware_fast_gi_memory_budget_bytes(const bool is_viewport)
{
  const char *override_mb = std::getenv("BLENDER_EEVEE_HWRT_FAST_GI_BUDGET_MB");
  if (override_mb != nullptr && override_mb[0] != '\0') {
    char *end = nullptr;
    const long parsed_mb = std::strtol(override_mb, &end, 10);
    if (end != override_mb && parsed_mb > 0) {
      return int64_t(parsed_mb) * 1024 * 1024;
    }
  }
  return is_viewport ? (5 * 1024 * 1024) / 4 : (2 * 1024 * 1024);
}

static HardwareFastGIMemoryLayout hardware_fast_gi_fit_memory_budget(
    int requested_grid_resolution, int requested_cascade_count, const bool is_viewport)
{
  HardwareFastGIMemoryLayout layout;
  layout.grid_resolution = max_ii(requested_grid_resolution, 4);
  layout.cascade_count = max_ii(requested_cascade_count, 1);
  layout.budget_bytes = hardware_fast_gi_memory_budget_bytes(is_viewport);
  layout.requested_bytes = hardware_fast_gi_memory_usage_bytes(
      layout.grid_resolution, layout.cascade_count);
  layout.allocated_bytes = layout.requested_bytes;
  while (layout.allocated_bytes > layout.budget_bytes && layout.cascade_count > 1) {
    layout.cascade_count--;
    layout.memory_limited = true;
    layout.allocated_bytes = hardware_fast_gi_memory_usage_bytes(
        layout.grid_resolution, layout.cascade_count);
  }
  while (layout.allocated_bytes > layout.budget_bytes && layout.grid_resolution > 4) {
    layout.grid_resolution = max_ii(4, layout.grid_resolution - 4);
    layout.memory_limited = true;
    layout.allocated_bytes = hardware_fast_gi_memory_usage_bytes(
        layout.grid_resolution, layout.cascade_count);
  }
  return layout;
}

static float hardware_fast_gi_scene_entry_radius(const HardwareRaytraceSceneEntry &entry)
{
  const float3 x_axis = float3(entry.object_to_world.x_axis());
  const float3 y_axis = float3(entry.object_to_world.y_axis());
  const float3 z_axis = float3(entry.object_to_world.z_axis());
  return max_ff(
      0.25f,
      0.5f * max_ff(math::length(x_axis), max_ff(math::length(y_axis), math::length(z_axis))));
}

static HardwareFastGISceneScaleAnalysis hardware_fast_gi_scene_scale_analysis(
    Span<HardwareRaytraceSceneEntry> scene_entries,
    const float3 &camera_position,
    const float3 &camera_forward,
    const float camera_clip_far)
{
  HardwareFastGISceneScaleAnalysis analysis;
  const float clip_far = max_ff(camera_clip_far, 12.0f);
  float3 safe_camera_forward = camera_forward;
  if (dot(safe_camera_forward, safe_camera_forward) <= 1.0e-8f) {
    safe_camera_forward = float3(0.0f, 0.0f, -1.0f);
  }
  else {
    safe_camera_forward = math::normalize(safe_camera_forward);
  }

  float scene_radius = 0.0f;
  float forward_extent = 0.0f;
  float lateral_extent = 0.0f;
  float density_accum = 0.0f;
  float3 bounds_min = float3(0.0f);
  float3 bounds_max = float3(0.0f);
  bool has_bounds = false;

  for (const HardwareRaytraceSceneEntry &entry : scene_entries) {
    if (entry.batch == nullptr) {
      continue;
    }

    const float radius = hardware_fast_gi_scene_entry_radius(entry);
    const float3 center = entry.object_to_world.location();
    const float3 to_entry = center - camera_position;
    const float distance = math::length(to_entry);
    const float forward = dot(to_entry, safe_camera_forward);
    const float lateral = math::length(to_entry - safe_camera_forward * forward);
    const float instance_factor = 1.0f + 0.25f * min_ff(float(entry.instance_count - 1), 3.0f);
    const float3 entry_min = center - float3(radius);
    const float3 entry_max = center + float3(radius);
    if (!has_bounds) {
      bounds_min = entry_min;
      bounds_max = entry_max;
      has_bounds = true;
    }
    else {
      bounds_min = float3(min_ff(bounds_min.x, entry_min.x),
                          min_ff(bounds_min.y, entry_min.y),
                          min_ff(bounds_min.z, entry_min.z));
      bounds_max = float3(max_ff(bounds_max.x, entry_max.x),
                          max_ff(bounds_max.y, entry_max.y),
                          max_ff(bounds_max.z, entry_max.z));
    }

    scene_radius = max_ff(scene_radius, min_ff(distance + radius, clip_far));
    forward_extent = max_ff(forward_extent, min_ff(max_ff(forward, 0.0f) + radius, clip_far));
    lateral_extent = max_ff(lateral_extent, min_ff(lateral + radius, clip_far));
    density_accum += (radius * instance_factor) / max_ff(distance + radius, 1.0f);
    analysis.active_entry_count++;
  }

  if (analysis.active_entry_count == 0) {
    return analysis;
  }

  const float3 scene_center = (bounds_min + bounds_max) * 0.5f;
  const float scene_bounds_radius = math::length(bounds_max - bounds_min) * 0.5f;
  analysis.scene_center = scene_center;
  analysis.scene_bounds_radius = clamp_f(scene_bounds_radius, 4.0f, clip_far);
  analysis.scene_radius = clamp_f(max_ff(max_ff(scene_radius, lateral_extent * 1.25f),
                                         scene_bounds_radius),
                                  4.0f,
                                  clip_far);
  analysis.forward_extent = clamp_f(max_ff(forward_extent, 4.0f), 4.0f, clip_far);
  analysis.lateral_extent = clamp_f(max_ff(lateral_extent, 2.0f), 2.0f, clip_far);
  analysis.density = clamp_f(density_accum / float(analysis.active_entry_count), 0.0f, 1.0f);
  return analysis;
}

static int hardware_fast_gi_scene_priority(const HardwareFastGISceneScaleAnalysis &analysis,
                                           const LightCullingData &culling_data)
{
  if (analysis.active_entry_count == 0) {
    return HWRT_SCENE_OPEN;
  }

  const bool compact_scene = analysis.scene_bounds_radius <= 18.0f;
  const bool emissive_room_scene = compact_scene && culling_data.local_lights_len <= 4u &&
                                   analysis.density >= 0.14f;
  const bool dense_scene = analysis.density >= 0.16f ||
                           (analysis.density >= 0.10f && culling_data.local_lights_len >= 12u);
  if ((compact_scene && dense_scene) || emissive_room_scene) {
    return HWRT_SCENE_INTERIOR;
  }

  const bool large_scene = analysis.scene_radius >= 28.0f || analysis.forward_extent >= 24.0f;
  const bool sparse_scene = analysis.density <= 0.08f && culling_data.local_lights_len <= 8u;
  if (large_scene && sparse_scene) {
    return HWRT_SCENE_OPEN;
  }

  return HWRT_SCENE_MIXED;
}

static int hardware_fast_gi_budget_rebalance(const int quality_tier,
                                             const int scene_priority,
                                             const HardwareFastGISceneScaleAnalysis &analysis,
                                             const LightCullingData &culling_data)
{
  const uint total_light_count = culling_data.local_lights_len + culling_data.sun_lights_len;
  if (total_light_count == 0u) {
    return HWRT_BUDGET_FAVOR_INDIRECT;
  }
  if (scene_priority == HWRT_SCENE_INTERIOR) {
    return HWRT_BUDGET_FAVOR_INDIRECT;
  }
  if (scene_priority == HWRT_SCENE_OPEN &&
      (culling_data.local_lights_len >= 12u || quality_tier == HWRT_QUALITY_PERFORMANCE))
  {
    return HWRT_BUDGET_FAVOR_DIRECT;
  }
  if (analysis.density >= 0.22f) {
    return HWRT_BUDGET_FAVOR_INDIRECT;
  }
  return HWRT_BUDGET_BALANCED;
}

static int hardware_fast_gi_quality_tier(const bool is_viewport,
                                         const float smoothed_traced_ms,
                                         const HardwareFastGISceneScaleAnalysis &analysis,
                                         const int scene_priority)
{
  if (!is_viewport) {
    return HWRT_QUALITY_REFERENCE;
  }
  if (smoothed_traced_ms > 18.0f) {
    return HWRT_QUALITY_PERFORMANCE;
  }
  if (smoothed_traced_ms > 11.0f) {
    return HWRT_QUALITY_BALANCED;
  }
  if (smoothed_traced_ms > 0.0f) {
    return HWRT_QUALITY_HIGH;
  }
  if (analysis.scene_radius <= 12.0f && analysis.density <= 0.08f) {
    return HWRT_QUALITY_HIGH;
  }
  if (scene_priority == HWRT_SCENE_OPEN) {
    return HWRT_QUALITY_PERFORMANCE;
  }
  return HWRT_QUALITY_BALANCED;
}

static int hardware_fast_gi_requested_grid_resolution(const int base_resolution_setting,
                                                      const HardwareFastGISceneScaleAnalysis &analysis,
                                                      const int quality_tier,
                                                      const int budget_rebalance_mode)
{
  int resolution = clamp_i(32 / max_ii(base_resolution_setting, 1), 4, 32);
  if (analysis.active_entry_count == 0) {
    return resolution;
  }

  if (analysis.scene_bounds_radius <= 12.0f) {
    resolution += 4;
  }
  else if (analysis.scene_bounds_radius >= 24.0f) {
    resolution -= 4;
  }

  if (analysis.density >= 0.30f) {
    resolution += 4;
  }
  else if (analysis.density <= 0.10f) {
    resolution -= 4;
  }

  if (budget_rebalance_mode == HWRT_BUDGET_FAVOR_INDIRECT) {
    resolution += 4;
  }
  else if (budget_rebalance_mode == HWRT_BUDGET_FAVOR_DIRECT) {
    resolution -= 4;
  }
  switch (quality_tier) {
    case HWRT_QUALITY_PERFORMANCE:
      resolution -= 4;
      break;
    case HWRT_QUALITY_HIGH:
      resolution += 4;
      break;
    case HWRT_QUALITY_REFERENCE:
      resolution += 8;
      break;
    case HWRT_QUALITY_BALANCED:
    default:
      break;
  }

  return clamp_i(int(4.0f * std::round(float(resolution) * 0.25f)), 4, 32);
}

static bool hardware_fast_gi_use_scene_stable_anchor(const HardwareFastGISceneScaleAnalysis &analysis,
                                                     const int scene_priority)
{
  if (analysis.active_entry_count == 0) {
    return false;
  }
  return scene_priority == HWRT_SCENE_INTERIOR || analysis.scene_bounds_radius <= 24.0f ||
         analysis.density >= 0.10f;
}

static float hardware_fast_gi_requested_distance(const float user_distance,
                                                 const HardwareFastGISceneScaleAnalysis &analysis,
                                                 const float camera_clip_far,
                                                 const int scene_priority,
                                                 const bool is_viewport)
{
  if (user_distance > 0.0f) {
    return user_distance;
  }

  float distance = 12.0f;
  if (analysis.active_entry_count > 0) {
    if (hardware_fast_gi_use_scene_stable_anchor(analysis, scene_priority)) {
      distance = analysis.scene_bounds_radius * 1.35f;
    }
    else {
      distance = max_ff(analysis.scene_radius * 1.15f,
                        max_ff(analysis.forward_extent * 1.25f, analysis.lateral_extent * 1.75f));
    }
    if (analysis.density >= 0.30f) {
      distance *= 0.85f;
    }
    else if (analysis.density <= 0.10f) {
      distance *= 1.15f;
    }
  }

  const float distance_cap = min_ff(
      is_viewport ? 48.0f : 96.0f,
      max_ff(camera_clip_far * (is_viewport ? 0.35f : 0.60f), 12.0f));
  return clamp_f(distance, 6.0f, distance_cap);
}

static float3 hardware_fast_gi_field_center(const HardwareFastGISceneScaleAnalysis &analysis,
                                            const int scene_priority,
                                            const float3 &view_location)
{
  return hardware_fast_gi_use_scene_stable_anchor(analysis, scene_priority) ?
             analysis.scene_center :
             view_location;
}

static uint hardware_direct_light_sample_count(const LightCullingData &culling_data,
                                               const bool is_viewport,
                                               const int quality_tier,
                                               const int budget_rebalance_mode)
{
  const uint total_light_count = culling_data.local_lights_len + culling_data.sun_lights_len;
  int sample_count = is_viewport ? 2 : 4;
  if (budget_rebalance_mode == HWRT_BUDGET_FAVOR_DIRECT) {
    sample_count += 1;
  }
  else if (budget_rebalance_mode == HWRT_BUDGET_FAVOR_INDIRECT) {
    sample_count -= 1;
  }

  switch (quality_tier) {
    case HWRT_QUALITY_PERFORMANCE:
      sample_count -= 1;
      break;
    case HWRT_QUALITY_HIGH:
    case HWRT_QUALITY_REFERENCE:
      sample_count += 1;
      break;
    case HWRT_QUALITY_BALANCED:
    default:
      break;
  }
  if (total_light_count <= 2u) {
    sample_count = min_ii(sample_count, is_viewport ? 2 : 4);
  }
  return uint(clamp_i(sample_count, 1, is_viewport ? 4 : 6));
}

static uint hardware_world_sun_light_count(Instance &inst, const LightCullingData &culling_data)
{
  float3 sky_sun_direction;
  if (inst.world.has_volume_absorption() ||
      !inst.world.sky_sun_shadow_direction_get(sky_sun_direction))
  {
    return 0u;
  }
  const uint configured_world_suns = inst.pipelines.world.use_lightpath_node() ? WORLD_SUN_MAX : 1u;
  return std::min(configured_world_suns, uint(culling_data.sun_lights_len));
}

static HardwareDirectLightData hardware_direct_light_data(const LightCullingData &culling_data,
                                                          const uint world_sun_lights_len,
                                                          const bool is_viewport,
                                                          const int quality_tier,
                                                          const int budget_rebalance_mode)
{
  HardwareDirectLightData data = {};
  data.selection_mode = HWRT_DIRECT_LIGHT_SELECTION_TILE;
  data.tile_size_px = uint(max_ff(culling_data.tile_size, 1.0f));
  data.tile_word_len = culling_data.tile_word_len;
  data.candidate_local_lights_len = culling_data.local_lights_len;
  data.local_lights_len = culling_data.local_lights_len;
  data.sun_lights_len = culling_data.sun_lights_len;
  data.light_samples_per_shading_point = hardware_direct_light_sample_count(
      culling_data, is_viewport, quality_tier, budget_rebalance_mode);
  data.trace_sun_lights_separately = (culling_data.sun_lights_len > 0);
  data.sample_emissive_meshes = false;
  data.local_light_importance_scale = 1.0f;
  data.area_light_importance_scale = 1.35f;
  data.textured_light_importance_scale = 1.5f;
  data.sun_light_importance_scale = 2.0f;
  data.world_sun_lights_len = world_sun_lights_len;
  return data;
}

static int hardware_fast_gi_brick_resolution(const int grid_resolution)
{
  return max_ii(1, min_ii(grid_resolution, 8));
}

static int hardware_fast_gi_brick_update_period(const int cascade_index,
                                                const bool is_viewport,
                                                const bool field_was_valid,
                                                const bool interactive)
{
  if (!is_viewport || !field_was_valid || !interactive) {
    return 1;
  }
  switch (cascade_index) {
    case 0:
      return 1;
    case 1:
      return 2;
    default:
      return 4;
  }
}

static bool hardware_fast_gi_should_update_brick(const uint64_t sample_index,
                                                 const int brick_index,
                                                 const int update_period)
{
  if (update_period <= 1) {
    return true;
  }
  return (brick_index % update_period) == int(sample_index % uint64_t(update_period));
}

static int3 hardware_fast_gi_camera_brick_shift(const float4 &previous_cascade_cfg,
                                                const float4 &current_cascade_cfg,
                                                const int brick_resolution)
{
  const float voxel_size = current_cascade_cfg.w;
  if (!(voxel_size > 0.0f) || brick_resolution <= 0) {
    return int3(0);
  }
  const float inv_voxel_size = 1.0f / voxel_size;
  const int3 voxel_shift = int3(int(std::round((current_cascade_cfg.x - previous_cascade_cfg.x) *
                                               inv_voxel_size)),
                                int(std::round((current_cascade_cfg.y - previous_cascade_cfg.y) *
                                               inv_voxel_size)),
                                int(std::round((current_cascade_cfg.z - previous_cascade_cfg.z) *
                                               inv_voxel_size)));
  return int3(voxel_shift.x / brick_resolution,
              voxel_shift.y / brick_resolution,
              voxel_shift.z / brick_resolution);
}

static bool hardware_fast_gi_brick_invalidated_by_camera_motion(const int3 &brick_coord,
                                                                const int3 &brick_grid_extent,
                                                                const int3 &brick_shift)
{
  if (brick_shift == int3(0)) {
    return false;
  }
  const int3 previous_brick_coord = brick_coord - brick_shift;
  return (previous_brick_coord.x < 0 || previous_brick_coord.y < 0 || previous_brick_coord.z < 0 ||
          previous_brick_coord.x >= brick_grid_extent.x ||
          previous_brick_coord.y >= brick_grid_extent.y ||
          previous_brick_coord.z >= brick_grid_extent.z);
}

struct HardwareFastGIBrickCandidate {
  int3 brick_coord;
  int brick_index = 0;
  float priority = 0.0f;
  bool camera_invalidated = false;
  bool reuse_compatible = false;
};

static int hardware_fast_gi_brick_sample_count(const int cascade_index,
                                               const bool is_viewport,
                                               const bool interactive,
                                               const int quality_tier,
                                               const int budget_rebalance_mode,
                                               const HardwareFastGIBrickCandidate &candidate)
{
  const int base_count = hardware_fast_gi_cascade_sample_count(
      cascade_index, is_viewport, interactive, quality_tier);
  const float priority_norm = std::clamp(candidate.priority / 24.0f, 0.0f, 1.0f);
  float budget_scale = 0.65f + priority_norm * 0.85f;
  if (budget_rebalance_mode == HWRT_BUDGET_FAVOR_INDIRECT) {
    budget_scale *= 1.15f;
  }
  else if (budget_rebalance_mode == HWRT_BUDGET_FAVOR_DIRECT) {
    budget_scale *= 0.85f;
  }
  if (candidate.camera_invalidated) {
    budget_scale = max_ff(budget_scale, 1.45f);
  }
  if (!candidate.reuse_compatible) {
    budget_scale = max_ff(budget_scale, 1.15f);
  }
  return max_ii(2, int(std::round(float(base_count) * budget_scale)));
}

static bool hardware_fast_gi_reuse_compatible(const bool field_was_valid,
                                              const bool field_config_valid,
                                              const bool force_full_refresh,
                                              const bool camera_invalidated)
{
  return field_was_valid && field_config_valid && !force_full_refresh && !camera_invalidated;
}

static float hardware_fast_gi_brick_visibility_score(const float3 &camera_position,
                                                     const float3 &camera_forward,
                                                     const float4 &cascade_config,
                                                     const int3 &brick_coord,
                                                     const int brick_resolution)
{
  const float voxel_size = cascade_config.w;
  if (!(voxel_size > 0.0f)) {
    return 0.0f;
  }
  const float3 brick_center = float3(cascade_config.x, cascade_config.y, cascade_config.z) +
                              (float3(brick_coord * brick_resolution) +
                               float3(brick_resolution * 0.5f)) *
                                  voxel_size;
  const float3 view_to_brick = math::normalize(brick_center - camera_position);
  return max_ff(0.0f, math::dot(view_to_brick, camera_forward));
}

static float3 hardware_fast_gi_brick_center(const float4 &cascade_config,
                                            const int3 &brick_coord,
                                            const int brick_resolution)
{
  return float3(cascade_config.x, cascade_config.y, cascade_config.z) +
         (float3(brick_coord * brick_resolution) + float3(brick_resolution * 0.5f)) *
             cascade_config.w;
}

static float hardware_fast_gi_energy_falloff(const float3 &brick_center,
                                             const float3 &source_position,
                                             const float source_energy,
                                             const float source_radius)
{
  if (!(source_energy > 0.0f)) {
    return 0.0f;
  }
  const float safe_radius = max_ff(source_radius, 1.0e-3f);
  const float distance = math::distance(brick_center, source_position);
  const float normalized_distance = distance / safe_radius;
  return std::log1p(source_energy) / (1.0f + normalized_distance * normalized_distance);
}

static float hardware_fast_gi_brick_high_energy_score(
    const float3 &brick_center,
    const float voxel_size,
    const int brick_resolution,
    Span<HardwareRaytraceSceneEntry> scene_entries,
    Span<LightData> local_lights)
{
  float score = 0.0f;
  const float brick_world_size = max_ff(voxel_size * brick_resolution, 1.0e-3f);
  for (const HardwareRaytraceSceneEntry &entry : scene_entries) {
    const float emissive_energy = math::reduce_max(entry.emissive_radiance);
    if (!(emissive_energy > 0.0f)) {
      continue;
    }
    score += hardware_fast_gi_energy_falloff(
        brick_center, entry.object_to_world.location(), emissive_energy, brick_world_size * 2.0f);
  }
  for (const LightData &light : local_lights) {
    const float light_energy = max_ffff(light.power[LIGHT_DIFFUSE],
                                        light.power[LIGHT_SPECULAR],
                                        light.power[LIGHT_TRANSMISSION],
                                        light.power[LIGHT_VOLUME]) *
                               math::reduce_max(float3(light.color));
    if (!(light_energy > 0.0f)) {
      continue;
    }
    score += hardware_fast_gi_energy_falloff(brick_center,
                                             light_position_get(light),
                                             light_energy,
                                             light.local().local.influence_radius_max);
  }
  return score;
}

static Vector<HardwareFastGIBrickCandidate> hardware_fast_gi_ranked_bricks(
    const int3 &brick_grid_extent,
    const int brick_resolution,
    const float4 &cascade_config,
    const int3 &camera_brick_shift,
    const bool field_was_valid,
    const bool field_config_valid,
    const bool force_full_refresh,
    const bool prioritize_dirty,
    const float3 &camera_position,
    const float3 &camera_forward,
    Span<HardwareRaytraceSceneEntry> scene_entries,
    Span<LightData> local_lights)
{
  Vector<HardwareFastGIBrickCandidate> candidates;
  candidates.reserve(brick_grid_extent.x * brick_grid_extent.y * brick_grid_extent.z);
  bool has_emissive_entries = false;
  for (const HardwareRaytraceSceneEntry &entry : scene_entries) {
    if (math::reduce_max(entry.emissive_radiance) > 0.0f) {
      has_emissive_entries = true;
      break;
    }
  }
  const bool has_high_energy_sources = has_emissive_entries || !local_lights.is_empty();
  const bool emissive_only_scene = has_emissive_entries && local_lights.is_empty();
  const float high_energy_weight = emissive_only_scene ? 5.0f : 3.0f;
  for (int brick_z = 0; brick_z < brick_grid_extent.z; brick_z++) {
    for (int brick_y = 0; brick_y < brick_grid_extent.y; brick_y++) {
      for (int brick_x = 0; brick_x < brick_grid_extent.x; brick_x++) {
        HardwareFastGIBrickCandidate candidate;
        candidate.brick_coord = int3(brick_x, brick_y, brick_z);
        candidate.brick_index = brick_x + brick_y * brick_grid_extent.x +
                                brick_z * brick_grid_extent.x * brick_grid_extent.y;
        candidate.camera_invalidated = hardware_fast_gi_brick_invalidated_by_camera_motion(
            candidate.brick_coord, brick_grid_extent, camera_brick_shift);
        candidate.reuse_compatible = hardware_fast_gi_reuse_compatible(
            field_was_valid, field_config_valid, force_full_refresh, candidate.camera_invalidated);
        const float3 brick_center = hardware_fast_gi_brick_center(
            cascade_config, candidate.brick_coord, brick_resolution);
        float visibility_score = hardware_fast_gi_brick_visibility_score(
            camera_position, camera_forward, cascade_config, candidate.brick_coord, brick_resolution);
        if (emissive_only_scene) {
          /* Emissive-only rooms still need rear-hemisphere bricks to stay warm when the source sits
           * behind the camera, so do not collapse their scheduling score fully to zero. */
          visibility_score = max_ff(visibility_score, 0.35f);
        }
        candidate.priority = visibility_score * 4.0f;
        if (has_high_energy_sources) {
          candidate.priority += hardware_fast_gi_brick_high_energy_score(
                                    brick_center,
                                    cascade_config.w,
                                    brick_resolution,
                                    scene_entries,
                                    local_lights) *
                                high_energy_weight;
        }
        if (prioritize_dirty) {
          candidate.priority += 8.0f;
        }
        if (candidate.camera_invalidated) {
          candidate.priority += 16.0f;
        }
        candidates.append(candidate);
      }
    }
  }
  std::sort(candidates.begin(),
            candidates.end(),
            [](const HardwareFastGIBrickCandidate &a, const HardwareFastGIBrickCandidate &b) {
              if (a.priority != b.priority) {
                return a.priority > b.priority;
              }
              return a.brick_index < b.brick_index;
            });
  return candidates;
}

static float3 hardware_fast_gi_snapped_center(const float3 &view_location, const float voxel_size)
{
  if (!(voxel_size > 0.0f)) {
    return view_location;
  }
  const float inv_voxel_size = 1.0f / voxel_size;
  return float3(math::floor(view_location.x * inv_voxel_size + 0.5f) * voxel_size,
                math::floor(view_location.y * inv_voxel_size + 0.5f) * voxel_size,
                math::floor(view_location.z * inv_voxel_size + 0.5f) * voxel_size);
}

static float3 hardware_fast_gi_hysteresis_center(const float3 &view_location,
                                                 const float4 &previous_cascade_cfg,
                                                 const float voxel_size,
                                                 const int brick_resolution)
{
  const float3 snapped_center = hardware_fast_gi_snapped_center(view_location, voxel_size);
  if (!(voxel_size > 0.0f) || brick_resolution <= 0 || previous_cascade_cfg.w != voxel_size) {
    return snapped_center;
  }
  const float hysteresis_distance = voxel_size * brick_resolution * 0.5f;
  float3 center = previous_cascade_cfg.xyz();
  if (std::abs(view_location.x - center.x) > hysteresis_distance) {
    center.x = snapped_center.x;
  }
  if (std::abs(view_location.y - center.y) > hysteresis_distance) {
    center.y = snapped_center.y;
  }
  if (std::abs(view_location.z - center.z) > hysteresis_distance) {
    center.z = snapped_center.z;
  }
  return center;
}

static bool hardware_fast_gi_cascade_config_changed(const float4 &a, const float4 &b)
{
  const float eps = 1.0e-6f * max_ff(max_ff(std::abs(a.w), std::abs(b.w)), 1.0f);
  return std::abs(a.x - b.x) > eps || std::abs(a.y - b.y) > eps ||
         std::abs(a.z - b.z) > eps || std::abs(a.w - b.w) > eps;
}

static void hardware_fast_gi_cascade_config_fill(float4 r_cascade_config[3],
                                                 const int cascade_count,
                                                 const float3 &fast_gi_center,
                                                 const float base_cell_size,
                                                 const int grid_resolution,
                                                 const bool field_config_valid,
                                                 const float4 field_cascade_config[3])
{
  const int brick_resolution = hardware_fast_gi_brick_resolution(grid_resolution);
  for (const int cascade_index : IndexRange(cascade_count)) {
    const float voxel_size = base_cell_size * float(1 << cascade_index);
    const float3 current_field_cascade_center =
        field_config_valid ?
            hardware_fast_gi_hysteresis_center(fast_gi_center,
                                               field_cascade_config[cascade_index],
                                               voxel_size,
                                               brick_resolution) :
            hardware_fast_gi_snapped_center(fast_gi_center, voxel_size);
    r_cascade_config[cascade_index] = float4(current_field_cascade_center, voxel_size);
  }
  for (const int cascade_index : IndexRange(cascade_count, 3 - cascade_count)) {
    r_cascade_config[cascade_index] = float4(0.0f);
  }
}

static int effective_hardware_specular_bounces(const int user_bounces,
                                               const RaytraceEEVEE_SpecularMode mode)
{
  const int clamped_user_bounces = max_ii(1, user_bounces);
  switch (mode) {
    case RAYTRACE_EEVEE_SPECULAR_MODE_AUTO:
    case RAYTRACE_EEVEE_SPECULAR_MODE_FULL_RT:
      return clamped_user_bounces;
    case RAYTRACE_EEVEE_SPECULAR_MODE_HYBRID:
      /* Hybrid already keeps the first visible segment on the screen path when trustworthy, so cap
       * the continuation budget to a small fixed count instead of inheriting arbitrarily high
       * multi-bounce costs from the Full RT path. */
      return min_ii(clamped_user_bounces, 2);
    case RAYTRACE_EEVEE_SPECULAR_MODE_OFF:
    default:
      return 1;
  }
}

static constexpr int hardware_gi_fixed_bounces = 1;

RayTraceModule::~RayTraceModule()
{
  free_hardware_metal_scene_cache();
}

/* -------------------------------------------------------------------- */
/** \name Raytracing
 *
 * \{ */

void RayTraceModule::init()
{
  const SceneEEVEE &sce_eevee = inst_.scene->eevee;

  ray_tracing_options_ = sce_eevee.ray_tracing_options;
  if ((sce_eevee.flag & SCE_EEVEE_FAST_GI_ENABLED) == 0) {
    ray_tracing_options_.trace_max_roughness = 1.0f;
  }
  /* Always initialize thickness, for the ray-cast node. */
  data_.thickness = ray_tracing_options_.screen_trace_thickness;

  use_raytracing_ = (sce_eevee.flag & SCE_EEVEE_SSR_ENABLED) != 0;
  tracing_method_ = RaytraceEEVEE_Method(sce_eevee.ray_tracing_method);
  if (tracing_method_ == RAYTRACE_EEVEE_METHOD_HARDWARE && inst_.is_viewport() &&
      inst_.v3d != nullptr && inst_.v3d->shading.type != OB_RENDER)
  {
    /* Nuru / Hardware RT is a Rendered-viewport feature only. Material Preview should stay on the
     * classic Eevee viewport path instead of partially activating the hardware stack. */
    tracing_method_ = RAYTRACE_EEVEE_METHOD_SCREEN;
  }
  hardware_gi_mode_ = RaytraceEEVEE_GIMode(sce_eevee.hardware_raytracing_gi_mode);
  if (!ELEM(hardware_gi_mode_, RAYTRACE_EEVEE_GI_MODE_ACCURATE, RAYTRACE_EEVEE_GI_MODE_OFF)) {
    hardware_gi_mode_ = RAYTRACE_EEVEE_GI_MODE_ACCURATE;
  }
  hardware_gi_enabled_ = use_hardware_tracing() && (hardware_gi_mode_ == RAYTRACE_EEVEE_GI_MODE_ACCURATE);
  hardware_fast_gi_valid_ = false;
  hardware_fast_gi_field_config_valid_ = false;
  hardware_fast_gi_depsgraph_update_count_valid_ = false;
  hardware_caustics_enabled_ = false;
  hardware_shadow_enabled_ = use_hardware_tracing() &&
                             (sce_eevee.hardware_raytracing_features &
                              RAYTRACE_EEVEE_HARDWARE_SHADOWS);
  hardware_lighting_use_hardware_rt_shadows_ = hardware_shadow_enabled_;
  hardware_reflection_mode_ = use_hardware_tracing() ?
                                  sanitize_reflection_mode(
                                      sce_eevee.hardware_raytracing_reflection_mode) :
                                  RAYTRACE_EEVEE_SPECULAR_MODE_OFF;
  hardware_refraction_mode_ = use_hardware_tracing() ?
                                  sanitize_specular_mode(
                                      sce_eevee.hardware_raytracing_refraction_mode) :
                                  RAYTRACE_EEVEE_SPECULAR_MODE_OFF;
  hardware_environment_enabled_ = use_hardware_tracing() &&
                                  (sce_eevee.hardware_raytracing_features &
                                   RAYTRACE_EEVEE_HARDWARE_ENVIRONMENT);
  hardware_lighting_use_hardware_rt_environment_visibility_ = hardware_environment_enabled_;
  /* Do not expose the cascaded Fast GI brick field as final lighting truth. The regular Hardware
   * RT GI path remains enabled through `hardware_gi_enabled_`; the brick cache was too visibly
   * camera/local-grid dependent in mirrors and final renders. */
  hardware_fast_gi_enabled_ = false;
  fast_gi_ray_count_ = sce_eevee.fast_gi_ray_count;
  fast_gi_step_count_ = sce_eevee.fast_gi_step_count;
  fast_gi_ao_only_ = (sce_eevee.fast_gi_method == FAST_GI_AO_ONLY);
  if (!use_hardware_tracing() || (active_hardware_feature_mask() == 0 && !use_hardware_fast_gi())) {
    free_hardware_metal_scene_cache();
  }

  float4 data(0.0f);
  radiance_dummy_black_tx_.ensure_2d(
      gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, int2(1), GPU_TEXTURE_USAGE_SHADER_READ, data);
  const float visibility = 1.0f;
  const float environment_visibility[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  hardware_shadow_visibility_tx_.ensure_2d_array(
      gpu::TextureFormat::SFLOAT_16, int2(1), 1, GPU_TEXTURE_USAGE_SHADER_READ, &visibility);
  hardware_secondary_shadow_visibility_tx_.ensure_2d_array(
      gpu::TextureFormat::SFLOAT_16, int2(1), 1, GPU_TEXTURE_USAGE_SHADER_READ, &visibility);
  hardware_environment_visibility_tx_.ensure_2d(
      gpu::TextureFormat::SFLOAT_16_16_16_16,
      int2(1),
      GPU_TEXTURE_USAGE_SHADER_READ,
      environment_visibility);
  const float4 zero(0.0f);
  hardware_caustics_history_tx_.ensure_2d(
      gpu::TextureFormat::SFLOAT_16_16_16_16,
      int2(1),
      GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE,
      zero);
  const eGPUTextureUsage cache_usage = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;
  hardware_indirect_gi_radiance_cache_tx_.ensure_2d_array(
      gpu::TextureFormat::SFLOAT_16_16_16_16, int2(1), 6, cache_usage);
  hardware_indirect_gi_position_cache_tx_.ensure_2d_array(
      gpu::TextureFormat::SFLOAT_16_16_16_16, int2(1), 6, cache_usage);
  hardware_indirect_gi_normal_cache_tx_.ensure_2d_array(
      gpu::TextureFormat::SFLOAT_16_16_16_16, int2(1), 6, cache_usage);
  hardware_reflected_receiver_gi_tx_.ensure_2d(
      gpu::TextureFormat::SFLOAT_16_16_16_16, int2(1), cache_usage);
  hardware_reflected_receiver_gi_blur_tx_.ensure_2d(
      gpu::TextureFormat::SFLOAT_16_16_16_16, int2(1), cache_usage);
  GPU_texture_clear(hardware_indirect_gi_radiance_cache_tx_, GPU_DATA_FLOAT, &zero);
  GPU_texture_clear(hardware_indirect_gi_position_cache_tx_, GPU_DATA_FLOAT, &zero);
  GPU_texture_clear(hardware_indirect_gi_normal_cache_tx_, GPU_DATA_FLOAT, &zero);
  GPU_texture_clear(hardware_reflected_receiver_gi_tx_, GPU_DATA_FLOAT, &zero);
  GPU_texture_clear(hardware_reflected_receiver_gi_blur_tx_, GPU_DATA_FLOAT, &zero);
  if (!hardware_fast_gi_tx_.is_valid()) {
    hardware_fast_gi_tx_.ensure_3d(gpu::TextureFormat::SFLOAT_16_16_16_16,
                                   int3(1),
                                   GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE,
                                   zero);
  }
  if (!hardware_fast_gi_error_tx_.is_valid()) {
    hardware_fast_gi_error_tx_.ensure_3d(gpu::TextureFormat::SFLOAT_16,
                                         int3(1),
                                         GPU_TEXTURE_USAGE_SHADER_READ |
                                             GPU_TEXTURE_USAGE_SHADER_WRITE,
                                         &zero.x);
  }
  if (!hardware_fast_gi_visibility_tx_.is_valid()) {
    hardware_fast_gi_visibility_tx_.ensure_3d(gpu::TextureFormat::SFLOAT_16_16_16_16,
                                              int3(1),
                                              GPU_TEXTURE_USAGE_SHADER_READ |
                                                  GPU_TEXTURE_USAGE_SHADER_WRITE,
                                              zero);
  }
}

void RayTraceModule::warm_tracing_backend()
{
  auto warm_screen_backend = [&]() {
    if (inst_.planar_probes.enabled()) {
      inst_.manager->warm_shader_specialization(trace_planar_ps_);
    }
    for (int j : IndexRange(2)) {
      data_.trace_refraction = bool(j);
      inst_.manager->warm_shader_specialization(trace_screen_ps_);
    }
  };

  if (use_screen_tracing()) {
    warm_screen_backend();
    return;
  }

  if (use_hardware_tracing()) {
    warm_hardware_tracing_backend();
    return;
  }

  inst_.manager->warm_shader_specialization(trace_fallback_ps_);
}

void RayTraceModule::submit_tracing_backend(View &render_view)
{
  use_hardware_specular_scene_ = false;
  use_hardware_hybrid_retrace_ = false;

  auto submit_screen_backend = [&]() {
    if (inst_.planar_probes.enabled()) {
      inst_.manager->submit(trace_planar_ps_, render_view);
    }
    inst_.manager->submit(trace_screen_ps_, render_view);
  };

  if (use_screen_tracing()) {
    submit_screen_backend();
    return;
  }

  if (use_hardware_tracing()) {
    submit_hardware_tracing_backend(render_view);
    return;
  }

  inst_.manager->submit(trace_fallback_ps_, render_view);
}

void RayTraceModule::warm_hardware_tracing_backend()
{
  /* Hardware GI currently overrides the classic screen/probe result for supported diffuse-like
   * closures only, so warm the classic passes as the baseline. */
  if (inst_.planar_probes.enabled()) {
    inst_.manager->warm_shader_specialization(trace_planar_ps_);
  }
  for (int j : IndexRange(2)) {
    data_.trace_refraction = bool(j);
    inst_.manager->warm_shader_specialization(trace_screen_ps_);
  }
  inst_.manager->warm_shader_specialization(trace_hardware_lighting_ps_);
  inst_.manager->warm_shader_specialization(hardware_reflected_receiver_gi_blur_ps_);
  inst_.manager->warm_shader_specialization(hardware_indirect_gi_cache_store_ps_);
  inst_.manager->warm_shader_specialization(scene_final_specular_resolve_ps_);
}

void RayTraceModule::update_hardware_tracing_scene_state()
{
  hardware_scene_entry_count_ = 0;
  hardware_scene_instance_count_ = 0;

  if (!use_hardware_tracing()) {
    return;
  }

  const Vector<HardwareRaytraceSceneEntry> &scene_entries = inst_.sync.hardware_raytrace_scene_entries();
  hardware_scene_entry_count_ = int(scene_entries.size());
  for (const HardwareRaytraceSceneEntry &entry : scene_entries) {
    hardware_scene_instance_count_ += int(entry.resource_handle.id_range().size());
  }
}

static Vector<GPUMetalRaytraceSceneEntry> build_hardware_metal_scene_entries(
    Span<HardwareRaytraceSceneEntry> scene_entries,
    int *r_emissive_entry_count = nullptr,
    float *r_emissive_peak = nullptr)
{
  if (r_emissive_entry_count != nullptr) {
    *r_emissive_entry_count = 0;
  }
  if (r_emissive_peak != nullptr) {
    *r_emissive_peak = 0.0f;
  }

  Vector<GPUMetalRaytraceSceneEntry> metal_scene_entries;
  metal_scene_entries.reserve(scene_entries.size());

  uint32_t user_id = 0;
  int emissive_entry_count = 0;
  float emissive_peak = 0.0f;
  for (const HardwareRaytraceSceneEntry &entry : scene_entries) {
    if (entry.batch == nullptr) {
      continue;
    }

    GPUMetalRaytraceSceneEntry metal_entry;
    metal_entry.batch = entry.batch;
    metal_entry.object_to_world = entry.object_to_world;
    metal_entry.instance_count = std::max(1u, entry.instance_count);
    metal_entry.user_id = user_id++;
    metal_entry.emissive_radiance = entry.emissive_radiance;
    metal_entry.diffuse_albedo = entry.diffuse_albedo;
    metal_entry.reflection_color = entry.reflection_color;
    metal_entry.reflection_roughness = entry.reflection_roughness;
    metal_entry.transmission_color = entry.transmission_color;
    metal_entry.transmission_roughness = entry.transmission_roughness;
    metal_entry.reflection_ior = entry.reflection_ior;
    metal_entry.refraction_ior = entry.refraction_ior;
    metal_entry.packed_thickness = entry.packed_thickness;
    metal_entry.alpha = entry.alpha;
    metal_entry.closure_type = entry.closure_type;
    metal_entry.proxy_flags = entry.proxy_flags;
    emissive_peak = std::max(emissive_peak, math::reduce_max(metal_entry.emissive_radiance));
    if (math::reduce_max(metal_entry.emissive_radiance) > 0.0f) {
      emissive_entry_count++;
    }
    metal_entry.material_slot = entry.material_slot;
    metal_entry.is_sculpt = entry.is_sculpt;
    metal_scene_entries.append(metal_entry);
  }

  if (r_emissive_entry_count != nullptr) {
    *r_emissive_entry_count = emissive_entry_count;
  }
  if (r_emissive_peak != nullptr) {
    *r_emissive_peak = emissive_peak;
  }

  return metal_scene_entries;
}

static Vector<HardwareRaytraceSceneEntry> sorted_hardware_scene_entries(
    const Vector<HardwareRaytraceSceneEntry> &scene_entries)
{
  Vector<HardwareRaytraceSceneEntry> sorted_entries = scene_entries;
  std::sort(sorted_entries.begin(),
            sorted_entries.end(),
            [](const HardwareRaytraceSceneEntry &a, const HardwareRaytraceSceneEntry &b) {
              if (a.object_key.hash() != b.object_key.hash()) {
                return a.object_key.hash() < b.object_key.hash();
              }
              if (a.material_slot != b.material_slot) {
                return a.material_slot < b.material_slot;
              }
              if (a.is_sculpt != b.is_sculpt) {
                return int(a.is_sculpt) < int(b.is_sculpt);
              }
              return uintptr_t(a.batch) < uintptr_t(b.batch);
            });
  return sorted_entries;
}

static GPUMetalRaytraceScene *build_hardware_metal_scene(
    Span<HardwareRaytraceSceneEntry> scene_entries,
                                                         GPUMetalRaytraceSceneStats *r_stats,
                                                         int *r_emissive_entry_count = nullptr,
                                                         float *r_emissive_peak = nullptr)
{
  if (r_stats != nullptr) {
    *r_stats = {};
  }
  Vector<GPUMetalRaytraceSceneEntry> metal_scene_entries = build_hardware_metal_scene_entries(
      scene_entries, r_emissive_entry_count, r_emissive_peak);
  return GPU_metal_raytrace_scene_build(metal_scene_entries.as_span(), r_stats);
}

static const char *hardware_scene_entries_geometry_mismatch_reason(
    const Vector<HardwareRaytraceSceneEntry> &entries,
    const Vector<HardwareRaytraceSceneEntry> &cached_entries,
    int *r_index = nullptr)
{
  if (entries.size() != cached_entries.size()) {
    if (r_index != nullptr) {
      *r_index = -1;
    }
    return "count";
  }
  for (const int i : entries.index_range()) {
    const HardwareRaytraceSceneEntry &entry = entries[i];
    const HardwareRaytraceSceneEntry &cached = cached_entries[i];
    if (!(entry.object_key == cached.object_key)) {
      if (r_index != nullptr) {
        *r_index = i;
      }
      return "object_key";
    }
    if (entry.material_slot != cached.material_slot) {
      if (r_index != nullptr) {
        *r_index = i;
      }
      return "material_slot";
    }
    if (entry.is_sculpt != cached.is_sculpt) {
      if (r_index != nullptr) {
        *r_index = i;
      }
      return "is_sculpt";
    }
    if (entry.batch != cached.batch) {
      if (r_index != nullptr) {
        *r_index = i;
      }
      return "batch";
    }
    if (entry.resource_handle.id_range() != cached.resource_handle.id_range()) {
      if (r_index != nullptr) {
        *r_index = i;
      }
      return "resource_range";
    }
  }
  return nullptr;
}

static bool hardware_scene_entries_match_geometry(
    const Vector<HardwareRaytraceSceneEntry> &entries,
    const Vector<HardwareRaytraceSceneEntry> &cached_entries)
{
  return hardware_scene_entries_geometry_mismatch_reason(entries, cached_entries) == nullptr;
}

static bool hardware_scene_entries_emissive_changed(
    const Vector<HardwareRaytraceSceneEntry> &entries,
    const Vector<HardwareRaytraceSceneEntry> &cached_entries)
{
  if (entries.size() != cached_entries.size()) {
    return false;
  }
  for (const int i : entries.index_range()) {
    const HardwareRaytraceSceneEntry &entry = entries[i];
    const HardwareRaytraceSceneEntry &cached = cached_entries[i];
    if (entry.emissive_radiance != cached.emissive_radiance) {
      return true;
    }
  }
  return false;
}

static bool hardware_scene_entries_transform_changed(
    const Vector<HardwareRaytraceSceneEntry> &entries,
    const Vector<HardwareRaytraceSceneEntry> &cached_entries)
{
  if (entries.size() != cached_entries.size()) {
    return false;
  }
  for (const int i : entries.index_range()) {
    const HardwareRaytraceSceneEntry &entry = entries[i];
    const HardwareRaytraceSceneEntry &cached = cached_entries[i];
    if (entry.object_to_world != cached.object_to_world) {
      return true;
    }
  }
  return false;
}

static bool hardware_scene_entries_instance_count_changed(
    const Vector<HardwareRaytraceSceneEntry> &entries,
    const Vector<HardwareRaytraceSceneEntry> &cached_entries)
{
  if (entries.size() != cached_entries.size()) {
    return false;
  }
  for (const int i : entries.index_range()) {
    const HardwareRaytraceSceneEntry &entry = entries[i];
    const HardwareRaytraceSceneEntry &cached = cached_entries[i];
    if (entry.instance_count != cached.instance_count) {
      return true;
    }
  }
  return false;
}

static bool hardware_scene_entries_material_proxy_changed(
    const Vector<HardwareRaytraceSceneEntry> &entries,
    const Vector<HardwareRaytraceSceneEntry> &cached_entries)
{
  if (entries.size() != cached_entries.size()) {
    return false;
  }
  for (const int i : entries.index_range()) {
    const HardwareRaytraceSceneEntry &entry = entries[i];
    const HardwareRaytraceSceneEntry &cached = cached_entries[i];
    if (entry.diffuse_albedo != cached.diffuse_albedo ||
        entry.reflection_color != cached.reflection_color ||
        entry.reflection_roughness != cached.reflection_roughness ||
        entry.transmission_color != cached.transmission_color ||
        entry.transmission_roughness != cached.transmission_roughness ||
        entry.reflection_ior != cached.reflection_ior ||
        entry.refraction_ior != cached.refraction_ior ||
        entry.packed_thickness != cached.packed_thickness ||
        entry.alpha != cached.alpha ||
        entry.reflection_layer_coverage != cached.reflection_layer_coverage ||
        entry.closure_type != cached.closure_type || entry.proxy_flags != cached.proxy_flags ||
        entry.material_runtime_hash != cached.material_runtime_hash)
    {
      return true;
    }
  }
  return false;
}

static bool hardware_scene_entries_require_blas_rebuild(
    const Vector<HardwareRaytraceSceneEntry> &entries)
{
  for (const HardwareRaytraceSceneEntry &entry : entries) {
    if ((entry.recalc & ID_RECALC_GEOMETRY) != 0) {
      return true;
    }
  }
  return false;
}

static uint32_t filtered_hardware_feature_mask(const RayTraceModule &raytracing,
                                               const eClosureBits active_closures)
{
  const uint32_t enabled_mask = raytracing.active_hardware_feature_mask();
  uint32_t mask = 0;
  if ((active_closures & (CLOSURE_DIFFUSE | CLOSURE_SSS)) != 0 &&
      (enabled_mask & RAYTRACE_EEVEE_HARDWARE_GI) != 0)
  {
    mask |= RAYTRACE_EEVEE_HARDWARE_GI;
  }
  if ((active_closures & CLOSURE_REFLECTION) != 0 &&
      (enabled_mask & RAYTRACE_EEVEE_HARDWARE_REFLECTIONS) != 0)
  {
    mask |= RAYTRACE_EEVEE_HARDWARE_REFLECTIONS;
  }
  if ((active_closures & CLOSURE_REFRACTION) != 0 &&
      (enabled_mask & RAYTRACE_EEVEE_HARDWARE_REFRACTIONS) != 0)
  {
    mask |= RAYTRACE_EEVEE_HARDWARE_REFRACTIONS;
  }
  if (mask != 0 && (enabled_mask & RAYTRACE_EEVEE_HARDWARE_ENVIRONMENT) != 0) {
    mask |= RAYTRACE_EEVEE_HARDWARE_ENVIRONMENT;
  }
  return mask;
}

static int effective_hardware_resolution_scale(const uint32_t feature_mask,
                                               const int base_scale,
                                               const RaytraceEEVEE_SpecularMode reflection_mode,
                                               const RaytraceEEVEE_SpecularMode refraction_mode)
{
  const bool has_hardware_gi = (feature_mask & RAYTRACE_EEVEE_HARDWARE_GI) != 0;
  const bool has_hardware_specular = (feature_mask &
                                      (RAYTRACE_EEVEE_HARDWARE_REFLECTIONS |
                                       RAYTRACE_EEVEE_HARDWARE_REFRACTIONS)) != 0;
  const bool has_full_rt_specular =
      has_hardware_specular &&
      (ELEM(reflection_mode,
            RAYTRACE_EEVEE_SPECULAR_MODE_FULL_RT,
            RAYTRACE_EEVEE_SPECULAR_MODE_AUTO) ||
       ELEM(refraction_mode,
            RAYTRACE_EEVEE_SPECULAR_MODE_FULL_RT,
            RAYTRACE_EEVEE_SPECULAR_MODE_AUTO));
  /* Keep Full RT-owned sharp specular paths at full resolution. Auto can still route rougher
   * pixels through the Hybrid screen path, but it should not downscale the sharp subset it
   * promotes to Hardware ownership. */
  if (has_full_rt_specular) {
    return 1;
  }
  /* Full Hardware GI converges poorly once the trace gets too coarse. Keep diffuse RT at no worse
   * than half resolution so temporal accumulation integrates real lighting samples instead of large
   * blocks, while still allowing the user control to reduce cost compared with full resolution. */
  if (has_hardware_gi) {
    return min_ii(2, max_ii(1, power_of_2_max_i(base_scale)));
  }
  return max_ii(1, power_of_2_max_i(base_scale));
}

void RayTraceModule::free_hardware_metal_scene_cache()
{
  GPU_metal_raytrace_scene_free(hardware_metal_scene_cache_);
  hardware_metal_scene_cache_ = nullptr;
  hardware_metal_scene_stats_cache_ = {};
  hardware_metal_scene_entries_cache_.clear();
  hardware_metal_scene_update_count_ = 0;
  hardware_metal_scene_update_count_valid_ = false;
  hardware_metal_scene_signature_ = 0;
  hardware_metal_scene_signature_valid_ = false;
  invalidate_sorted_hardware_scene_entries_cache();
}

void RayTraceModule::invalidate_sorted_hardware_scene_entries_cache()
{
  hardware_sorted_scene_entries_cache_.clear();
  hardware_sorted_scene_entries_update_count_ = 0;
  hardware_sorted_scene_entries_update_count_valid_ = false;
}

void RayTraceModule::invalidate_viewport_hardware_visibility_cache()
{
  hardware_primary_environment_visibility_ready_ = false;
  hardware_primary_environment_visibility_depth_tx_ = nullptr;
  hardware_primary_environment_visibility_normal_tx_ = nullptr;
  hardware_primary_environment_visibility_extent_ = int2(0);
  hardware_primary_environment_enabled_ = false;
  hardware_primary_shadow_visibility_ready_ = false;
  hardware_primary_shadow_visibility_depth_tx_ = nullptr;
  hardware_primary_shadow_visibility_normal_tx_ = nullptr;
  hardware_primary_shadow_visibility_extent_ = int2(0);
  hardware_primary_shadow_direct_enabled_ = false;
  hardware_primary_shadow_world_enabled_ = false;
}

const Vector<HardwareRaytraceSceneEntry> &RayTraceModule::current_sorted_hardware_scene_entries(
    const uint64_t depsgraph_update_count)
{
  if (!hardware_sorted_scene_entries_update_count_valid_ ||
      hardware_sorted_scene_entries_update_count_ != depsgraph_update_count)
  {
    hardware_sorted_scene_entries_cache_ = sorted_hardware_scene_entries(
        inst_.sync.hardware_raytrace_scene_entries());
    hardware_sorted_scene_entries_update_count_ = depsgraph_update_count;
    hardware_sorted_scene_entries_update_count_valid_ = true;
  }
  return hardware_sorted_scene_entries_cache_;
}

GPUMetalRaytraceScene *RayTraceModule::acquire_hardware_metal_scene(
    GPUMetalRaytraceSceneStats *r_stats, const bool require_current_feature_mask)
{
  if (r_stats != nullptr) {
    *r_stats = {};
  }
  const bool perf_logging_enabled = hardware_perf_logging_enabled();
  const double perf_start_time = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  const uint64_t current_scene_signature = inst_.sync.hardware_raytrace_scene_signature();

  if (!use_hardware_tracing() || (active_hardware_feature_mask() == 0 && !use_hardware_fast_gi())) {
    free_hardware_metal_scene_cache();
    return nullptr;
  }
  if (require_current_feature_mask && current_hardware_feature_mask_ == 0) {
    return nullptr;
  }

  const uint64_t depsgraph_update_count = (inst_.depsgraph != nullptr) ?
                                              DEG_get_update_count(inst_.depsgraph) :
                                              0;
  if (hardware_metal_scene_cache_ != nullptr && hardware_metal_scene_update_count_valid_ &&
      hardware_metal_scene_update_count_ == depsgraph_update_count &&
      hardware_metal_scene_signature_valid_ &&
      hardware_metal_scene_signature_ == current_scene_signature)
  {
    if (std::getenv("BLENDER_EEVEE_HWRT_CACHE_LOG") != nullptr) {
      std::fprintf(stderr,
                   "EEVEE HWRT scene cache hit update=%llu entries=%d instances=%d\n",
                   (unsigned long long)depsgraph_update_count,
                   hardware_scene_entry_count_,
                   hardware_scene_instance_count_);
    }
    if (perf_logging_enabled) {
      const double elapsed_ms = (BLI_time_now_seconds() - perf_start_time) * 1000.0;
      std::fprintf(stderr,
                   "EEVEE HWRT perf scene_cache=hit entries=%d instances=%d elapsed_ms=%.2f\n",
                   hardware_scene_entry_count_,
                   hardware_scene_instance_count_,
                   elapsed_ms);
    }
    if (r_stats != nullptr) {
      *r_stats = hardware_metal_scene_stats_cache_;
    }
    return hardware_metal_scene_cache_;
  }

  const Vector<HardwareRaytraceSceneEntry> sorted_scene_entries =
      current_sorted_hardware_scene_entries(depsgraph_update_count);
  int geometry_mismatch_index = -1;
  const char *geometry_mismatch_reason = hardware_scene_entries_geometry_mismatch_reason(
      sorted_scene_entries, hardware_metal_scene_entries_cache_, &geometry_mismatch_index);
  const bool geometry_matches_cache = geometry_mismatch_reason == nullptr;
  const bool blas_rebuild_required = hardware_scene_entries_require_blas_rebuild(
      sorted_scene_entries);
  const bool transform_changed = geometry_matches_cache &&
                                 hardware_scene_entries_transform_changed(
                                     sorted_scene_entries, hardware_metal_scene_entries_cache_);
  const bool instance_count_changed = geometry_matches_cache &&
                                      hardware_scene_entries_instance_count_changed(
                                          sorted_scene_entries,
                                          hardware_metal_scene_entries_cache_);
  const bool animation_changed = transform_changed || instance_count_changed;
  const bool emissive_changed = geometry_matches_cache &&
                                hardware_scene_entries_emissive_changed(
                                    sorted_scene_entries, hardware_metal_scene_entries_cache_);
  const bool material_changed = geometry_matches_cache &&
                                hardware_scene_entries_material_proxy_changed(
                                    sorted_scene_entries, hardware_metal_scene_entries_cache_);
  const bool needs_full_rebuild = !geometry_matches_cache || blas_rebuild_required;
  const bool cache_logging_enabled = std::getenv("BLENDER_EEVEE_HWRT_CACHE_LOG") != nullptr;

  if (hardware_metal_scene_cache_ != nullptr && !needs_full_rebuild) {
    if (!animation_changed && !emissive_changed && !material_changed) {
      hardware_metal_scene_entries_cache_ = sorted_scene_entries;
      hardware_metal_scene_update_count_ = depsgraph_update_count;
      hardware_metal_scene_update_count_valid_ = true;
      hardware_metal_scene_signature_ = current_scene_signature;
      hardware_metal_scene_signature_valid_ = true;
      if (cache_logging_enabled) {
        std::fprintf(stderr,
                     "EEVEE HWRT scene cache reuse update=%llu entries=%d instances=%d reason=depsgraph_only\n",
                     (unsigned long long)depsgraph_update_count,
                     hardware_scene_entry_count_,
                     hardware_scene_instance_count_);
      }
      if (perf_logging_enabled) {
        const double elapsed_ms = (BLI_time_now_seconds() - perf_start_time) * 1000.0;
        std::fprintf(stderr,
                     "EEVEE HWRT perf scene_cache=reuse entries=%d instances=%d elapsed_ms=%.2f\n",
                     hardware_scene_entry_count_,
                     hardware_scene_instance_count_,
                     elapsed_ms);
      }
      if (r_stats != nullptr) {
        *r_stats = hardware_metal_scene_stats_cache_;
      }
      return hardware_metal_scene_cache_;
    }
    Vector<GPUMetalRaytraceSceneEntry> metal_scene_entries = build_hardware_metal_scene_entries(
        sorted_scene_entries);
    GPUMetalRaytraceSceneUpdateParams update_params;
    update_params.update_tlas = animation_changed;
    update_params.update_emissive_data = emissive_changed || animation_changed;
    update_params.update_material_data = material_changed;
    update_params.update_world_geometry_data = animation_changed;
    if (GPU_metal_raytrace_scene_update(
            hardware_metal_scene_cache_,
            metal_scene_entries.as_span(),
            update_params,
            &hardware_metal_scene_stats_cache_))
    {
      hardware_metal_scene_entries_cache_ = sorted_scene_entries;
      hardware_metal_scene_update_count_ = depsgraph_update_count;
      hardware_metal_scene_update_count_valid_ = true;
      hardware_metal_scene_signature_ = current_scene_signature;
      hardware_metal_scene_signature_valid_ = true;
      if (cache_logging_enabled) {
        std::fprintf(stderr,
                     "EEVEE HWRT scene cache update update=%llu entries=%d instances=%d tlas=%d emissive=%d material=%d world_geom=%d\n",
                     (unsigned long long)depsgraph_update_count,
                     hardware_scene_entry_count_,
                     hardware_scene_instance_count_,
                     update_params.update_tlas ? 1 : 0,
                     update_params.update_emissive_data ? 1 : 0,
                     update_params.update_material_data ? 1 : 0,
                     update_params.update_world_geometry_data ? 1 : 0);
      }
      if (perf_logging_enabled) {
        const double elapsed_ms = (BLI_time_now_seconds() - perf_start_time) * 1000.0;
        std::fprintf(stderr,
                     "EEVEE HWRT perf scene_cache=update entries=%d instances=%d tlas=%d emissive=%d material=%d world_geom=%d elapsed_ms=%.2f\n",
                     hardware_scene_entry_count_,
                     hardware_scene_instance_count_,
                     update_params.update_tlas ? 1 : 0,
                     update_params.update_emissive_data ? 1 : 0,
                     update_params.update_material_data ? 1 : 0,
                     update_params.update_world_geometry_data ? 1 : 0,
                     elapsed_ms);
      }
      if (r_stats != nullptr) {
        *r_stats = hardware_metal_scene_stats_cache_;
      }
      return hardware_metal_scene_cache_;
    }
  }

  free_hardware_metal_scene_cache();
  hardware_metal_scene_cache_ = build_hardware_metal_scene(
      sorted_scene_entries, &hardware_metal_scene_stats_cache_);
  hardware_metal_scene_entries_cache_ = sorted_scene_entries;
  hardware_metal_scene_update_count_ = depsgraph_update_count;
  hardware_metal_scene_update_count_valid_ = true;
  hardware_metal_scene_signature_ = current_scene_signature;
  hardware_metal_scene_signature_valid_ = true;
  if (cache_logging_enabled) {
    std::fprintf(stderr,
                 "EEVEE HWRT scene cache miss update=%llu entries=%d instances=%d built=%d reason=%s reason_index=%d geometry_match=%d blas_rebuild=%d animation=%d shading=%d\n",
                 (unsigned long long)depsgraph_update_count,
                 hardware_scene_entry_count_,
                 hardware_scene_instance_count_,
                 hardware_metal_scene_stats_cache_.built_scene ? 1 : 0,
                 geometry_mismatch_reason != nullptr ?
                     geometry_mismatch_reason :
                     (blas_rebuild_required ? "geometry_recalc" : "cold_start"),
                 geometry_mismatch_index,
                 geometry_matches_cache ? 1 : 0,
                 blas_rebuild_required ? 1 : 0,
                 animation_changed ? 1 : 0,
                 (emissive_changed || material_changed) ? 1 : 0);
  }
  if (perf_logging_enabled) {
    const double elapsed_ms = (BLI_time_now_seconds() - perf_start_time) * 1000.0;
    std::fprintf(stderr,
                 "EEVEE HWRT perf scene_cache=miss entries=%d instances=%d built=%d elapsed_ms=%.2f\n",
                 hardware_scene_entry_count_,
                 hardware_scene_instance_count_,
                 hardware_metal_scene_stats_cache_.built_scene ? 1 : 0,
                 elapsed_ms);
  }
  if (r_stats != nullptr) {
    *r_stats = hardware_metal_scene_stats_cache_;
  }
  return hardware_metal_scene_cache_;
}

gpu::Texture **RayTraceModule::directional_shadow_visibility_tx()
{
  return &hardware_shadow_visibility_tx_;
}

gpu::Texture **RayTraceModule::direct_light_accum_tx()
{
  return &hardware_direct_light_denoised_tx_;
}

gpu::Texture **RayTraceModule::environment_visibility_tx()
{
  return &hardware_environment_visibility_tx_;
}

gpu::Texture **RayTraceModule::caustics_tx()
{
  return &hardware_caustics_history_tx_;
}

gpu::Texture **RayTraceModule::fast_gi_tx()
{
  return &hardware_fast_gi_tx_;
}

gpu::Texture **RayTraceModule::fast_gi_visibility_tx()
{
  return &hardware_fast_gi_visibility_tx_;
}

void RayTraceModule::update_hardware_fast_gi_field(View &render_view,
                                                   gpu::Texture *depth_tx,
                                                   gpu::Texture *input_radiance_tx,
                                                   int2 /*extent*/)
{
  if (!use_hardware_tracing_method() || !use_hardware_fast_gi()) {
    hardware_fast_gi_valid_ = false;
    hardware_fast_gi_depsgraph_update_count_valid_ = false;
    hardware_fast_gi_light_invalidation_pending_ = false;
    hardware_fast_gi_world_invalidation_pending_ = false;
    hardware_fast_gi_emissive_invalidation_pending_ = false;
    hardware_fast_gi_material_invalidation_pending_ = false;
    hardware_fast_gi_transform_invalidation_pending_ = false;
    hardware_fast_gi_geometry_invalidation_pending_ = false;
    hardware_fast_gi_animation_invalidation_pending_ = false;
    hardware_fast_gi_field_config_valid_ = false;
    hardware_fast_gi_smoothed_traced_ms_ = 0.0f;
    return;
  }
  const uint64_t depsgraph_update_count = (inst_.depsgraph != nullptr) ?
                                              DEG_get_update_count(inst_.depsgraph) :
                                              0;
  const bool current_field_depsgraph_update_changed =
      !hardware_fast_gi_depsgraph_update_count_valid_ ||
      hardware_fast_gi_depsgraph_update_count_ != depsgraph_update_count;
  const bool depsgraph_update_changed = current_field_depsgraph_update_changed;
  const bool field_was_valid = hardware_fast_gi_valid_;
  const bool field_config_was_valid = hardware_fast_gi_field_config_valid_;
  if (depsgraph_update_changed) {
    hardware_fast_gi_light_invalidation_pending_ |= inst_.lights.fast_gi_lighting_changed();
    hardware_fast_gi_world_invalidation_pending_ |= inst_.world.fast_gi_changed();
  }
  const Vector<HardwareRaytraceSceneEntry> sorted_scene_entries =
      current_sorted_hardware_scene_entries(depsgraph_update_count);
  const bool geometry_matches_cache = hardware_scene_entries_match_geometry(
      sorted_scene_entries, hardware_metal_scene_entries_cache_);
  const bool has_hardware_scene_cache = hardware_metal_scene_cache_ != nullptr ||
                                        !hardware_metal_scene_entries_cache_.is_empty();
  if (has_hardware_scene_cache && !geometry_matches_cache) {
    hardware_fast_gi_geometry_invalidation_pending_ = true;
  }
  if (geometry_matches_cache) {
    hardware_fast_gi_emissive_invalidation_pending_ |= hardware_scene_entries_emissive_changed(
        sorted_scene_entries, hardware_metal_scene_entries_cache_);
    hardware_fast_gi_material_invalidation_pending_ |= hardware_scene_entries_material_proxy_changed(
        sorted_scene_entries, hardware_metal_scene_entries_cache_);
    hardware_fast_gi_transform_invalidation_pending_ |= hardware_scene_entries_transform_changed(
        sorted_scene_entries, hardware_metal_scene_entries_cache_);
    hardware_fast_gi_animation_invalidation_pending_ |= hardware_scene_entries_instance_count_changed(
        sorted_scene_entries, hardware_metal_scene_entries_cache_);
  }
  const bool light_invalidation_pending = hardware_fast_gi_light_invalidation_pending_;
  const bool world_invalidation_pending = hardware_fast_gi_world_invalidation_pending_;
  const bool emissive_invalidation_pending = hardware_fast_gi_emissive_invalidation_pending_;
  const bool material_invalidation_pending = hardware_fast_gi_material_invalidation_pending_;
  const bool transform_invalidation_pending = hardware_fast_gi_transform_invalidation_pending_;
  const bool geometry_invalidation_pending = hardware_fast_gi_geometry_invalidation_pending_;
  const bool animation_invalidation_pending = hardware_fast_gi_animation_invalidation_pending_;
  const bool soft_invalidation_pending = light_invalidation_pending || world_invalidation_pending ||
                                         emissive_invalidation_pending ||
                                         material_invalidation_pending ||
                                         transform_invalidation_pending;
  const bool hard_invalidation_pending = geometry_invalidation_pending ||
                                         animation_invalidation_pending;
  const bool any_invalidation_pending = soft_invalidation_pending || hard_invalidation_pending;
  if (any_invalidation_pending)
  {
    hardware_fast_gi_valid_ = false;
    hardware_fast_gi_field_config_valid_ = false;
  }

  const int grid_resolution = max_ii(data_.hardware_fast_gi_grid_resolution, 1);
  const int cascade_count = max_ii(data_.hardware_fast_gi_cascade_count, 1);
  const int3 grid_extent(grid_resolution);
  const float base_cell_size = max_ff(hardware_fast_gi_requested_distance_ /
                                          float(2 * grid_resolution),
                                      0.25f);
  const int brick_resolution = hardware_fast_gi_brick_resolution(grid_resolution);
  const int3 brick_grid_extent = math::divide_ceil(grid_extent, int3(brick_resolution));
  const int bricks_per_cascade = brick_grid_extent.x * brick_grid_extent.y * brick_grid_extent.z;
  const int3 dispatch_size = math::divide_ceil(grid_extent, int3(4));
  const bool log_fast_gi_stats = hardware_fast_gi_stats_logging_enabled();
  const double update_start_time = log_fast_gi_stats ? BLI_time_now_seconds() : 0.0;
  const double fast_gi_mem_mib = double(hardware_fast_gi_allocated_bytes_) / (1024.0 * 1024.0);
  const double fast_gi_budget_mib = double(hardware_fast_gi_budget_bytes_) / (1024.0 * 1024.0);
  double scene_acquire_ms = 0.0;
  double traced_ms = 0.0;
  double screen_seed_ms = 0.0;
  int emissive_light_count = 0;
  float emissive_energy_sum = 0.0f;
  const bool interactive_viewport = hardware_viewport_interactive(inst_);
  if (hardware_fast_gi_freeze_updates_) {
    hardware_fast_gi_valid_ = field_was_valid;
    if (log_fast_gi_stats) {
      GPU_flush();
      const double elapsed_ms = (BLI_time_now_seconds() - update_start_time) * 1000.0;
      std::fprintf(stderr,
                   "EEVEE HWRT FastGI stats viewport=%d interactive=%d sample=%llu "
                   "tier=%s scene=%s budget=%s debug=%s isolate=%s freeze=%d smoothed_traced_ms=%.2f direct_samples=%d grid=%d/%d brick_res=%d active_bricks=%d reused_bricks=%d light_invalidated=%d world_invalidated=%d emissive_invalidated=%d material_invalidated=%d geometry_invalidated=%d animation_invalidated=%d total_bricks=%d active_cascades=%d/%d memory_limited=%d fast_gi_mem_mib=%.2f budget_mib=%.2f scene_radius=%.2f density=%.3f field_distance=%.2f updated_range=[%d,%d) dispatch=%dx%dx%d traced=%d screen_seed=%d field_valid=%d skipped=%d scene_acquire_ms=%.2f traced_ms=%.2f screen_seed_ms=%.2f elapsed_ms=%.2f\n",
                   inst_.is_viewport() ? 1 : 0,
                   interactive_viewport ? 1 : 0,
                   (unsigned long long)inst_.sampling.sample_index(),
                   hardware_quality_tier_name(hardware_fast_gi_quality_tier_),
                   hardware_scene_priority_name(hardware_fast_gi_scene_priority_),
                   hardware_budget_rebalance_name(hardware_fast_gi_budget_rebalance_),
                   hardware_debug_view_mode_name(hardware_debug_view_mode_),
                   hardware_debug_isolate_mode_name(hardware_debug_isolate_mode_),
                   1,
                   hardware_fast_gi_smoothed_traced_ms_,
                   hardware_direct_light_sample_count_,
                   grid_resolution,
                   hardware_fast_gi_requested_grid_resolution_,
                   brick_resolution,
                   0,
                   0,
                   light_invalidation_pending ? 1 : 0,
                   world_invalidation_pending ? 1 : 0,
                   emissive_invalidation_pending ? 1 : 0,
                   material_invalidation_pending ? 1 : 0,
                   geometry_invalidation_pending ? 1 : 0,
                   animation_invalidation_pending ? 1 : 0,
                   bricks_per_cascade * cascade_count,
                   field_was_valid ? cascade_count : 0,
                   cascade_count,
                   hardware_fast_gi_memory_limited_ ? 1 : 0,
                   fast_gi_mem_mib,
                   fast_gi_budget_mib,
                   hardware_fast_gi_scene_radius_,
                   hardware_fast_gi_scene_density_,
                   hardware_fast_gi_requested_distance_,
                   0,
                   0,
                   dispatch_size.x,
                   dispatch_size.y,
                   dispatch_size.z,
                   0,
                   0,
                   field_was_valid ? 1 : 0,
                   1,
                   scene_acquire_ms,
                   traced_ms,
                   screen_seed_ms,
                   elapsed_ms);
    }
    if (inst_.is_viewport() && hardware_fast_gi_debug_overlay_enabled()) {
      inst_.info_append(
          "Fast GI Debug: valid={} producer=freeze tier={} scene={} budget={} debug={} isolate={} freeze={} traced_ms={:.2f} direct_samples={} camera_dependent={} bricks={}/{} reused_bricks={} resident_cascades={}/{} memory_limited={} mem_mib={:.2f}/{:.2f} scene_radius={:.2f} density={:.3f} field_distance={:.2f} light_invalidated={} world_invalidated={} emissive_invalidated={} material_invalidated={} geometry_invalidated={} animation_invalidated={} emissive_lights={} emissive_energy={:.3f}",
          field_was_valid ? 1 : 0,
          hardware_quality_tier_name(hardware_fast_gi_quality_tier_),
          hardware_scene_priority_name(hardware_fast_gi_scene_priority_),
          hardware_budget_rebalance_name(hardware_fast_gi_budget_rebalance_),
          hardware_debug_view_mode_name(hardware_debug_view_mode_),
          hardware_debug_isolate_mode_name(hardware_debug_isolate_mode_),
          1,
          hardware_fast_gi_smoothed_traced_ms_,
          hardware_direct_light_sample_count_,
          0,
          0,
          bricks_per_cascade * cascade_count,
          0,
          field_was_valid ? cascade_count : 0,
          cascade_count,
          hardware_fast_gi_memory_limited_ ? 1 : 0,
          fast_gi_mem_mib,
          fast_gi_budget_mib,
          hardware_fast_gi_scene_radius_,
          hardware_fast_gi_scene_density_,
          hardware_fast_gi_requested_distance_,
          light_invalidation_pending ? 1 : 0,
          world_invalidation_pending ? 1 : 0,
          emissive_invalidation_pending ? 1 : 0,
          material_invalidation_pending ? 1 : 0,
          geometry_invalidation_pending ? 1 : 0,
          animation_invalidation_pending ? 1 : 0,
          emissive_light_count,
          emissive_energy_sum);
    }
    return;
  }
  if (interactive_viewport) {
    /* Live viewport motion should stay responsive. Reuse the last settled field if we have one,
     * otherwise leave diffuse ownership on the classic path until the viewport settles and a real
     * Fast GI update can complete. */
    hardware_fast_gi_valid_ = field_was_valid;
    if (log_fast_gi_stats) {
      GPU_flush();
      const double elapsed_ms = (BLI_time_now_seconds() - update_start_time) * 1000.0;
      std::fprintf(stderr,
                   "EEVEE HWRT FastGI stats viewport=%d interactive=%d sample=%llu "
                   "tier=%s scene=%s budget=%s debug=%s isolate=%s freeze=%d smoothed_traced_ms=%.2f direct_samples=%d grid=%d/%d brick_res=%d active_bricks=%d reused_bricks=%d light_invalidated=%d world_invalidated=%d emissive_invalidated=%d material_invalidated=%d geometry_invalidated=%d animation_invalidated=%d total_bricks=%d active_cascades=%d/%d memory_limited=%d fast_gi_mem_mib=%.2f budget_mib=%.2f scene_radius=%.2f density=%.3f field_distance=%.2f updated_range=[%d,%d) dispatch=%dx%dx%d traced=%d screen_seed=%d field_valid=%d skipped=%d scene_acquire_ms=%.2f traced_ms=%.2f screen_seed_ms=%.2f elapsed_ms=%.2f\n",
                   inst_.is_viewport() ? 1 : 0,
                   1,
                   (unsigned long long)inst_.sampling.sample_index(),
                   hardware_quality_tier_name(hardware_fast_gi_quality_tier_),
                   hardware_scene_priority_name(hardware_fast_gi_scene_priority_),
                   hardware_budget_rebalance_name(hardware_fast_gi_budget_rebalance_),
                   hardware_debug_view_mode_name(hardware_debug_view_mode_),
                   hardware_debug_isolate_mode_name(hardware_debug_isolate_mode_),
                   hardware_fast_gi_freeze_updates_ ? 1 : 0,
                   hardware_fast_gi_smoothed_traced_ms_,
                   hardware_direct_light_sample_count_,
                   grid_resolution,
                   hardware_fast_gi_requested_grid_resolution_,
                   brick_resolution,
                   0,
                   0,
                   light_invalidation_pending ? 1 : 0,
                   world_invalidation_pending ? 1 : 0,
                   emissive_invalidation_pending ? 1 : 0,
                   material_invalidation_pending ? 1 : 0,
                   geometry_invalidation_pending ? 1 : 0,
                   animation_invalidation_pending ? 1 : 0,
                   bricks_per_cascade * cascade_count,
                   cascade_count,
                   hardware_fast_gi_requested_cascade_count_,
                   hardware_fast_gi_memory_limited_ ? 1 : 0,
                   fast_gi_mem_mib,
                   fast_gi_budget_mib,
                   hardware_fast_gi_scene_radius_,
                   hardware_fast_gi_scene_density_,
                   hardware_fast_gi_requested_distance_,
                   0,
                   0,
                   dispatch_size.x,
                   dispatch_size.y,
                   dispatch_size.z,
                   0,
                   0,
                   hardware_fast_gi_valid_ ? 1 : 0,
                   1,
                   scene_acquire_ms,
                   traced_ms,
                   screen_seed_ms,
                   elapsed_ms);
    }
    if (inst_.is_viewport() && hardware_fast_gi_debug_overlay_enabled()) {
      inst_.info_append(
          "Fast GI Debug: valid={} producer=hold tier={} scene={} budget={} debug={} isolate={} freeze={} traced_ms={:.2f} direct_samples={} camera_dependent={} bricks={}/{} reused_bricks={} resident_cascades={}/{} memory_limited={} mem_mib={:.2f}/{:.2f} scene_radius={:.2f} density={:.3f} field_distance={:.2f} light_invalidated={} world_invalidated={} emissive_invalidated={} material_invalidated={} geometry_invalidated={} animation_invalidated={} emissive_lights={} emissive_energy={:.3f}",
          hardware_fast_gi_valid_ ? 1 : 0,
          hardware_quality_tier_name(hardware_fast_gi_quality_tier_),
          hardware_scene_priority_name(hardware_fast_gi_scene_priority_),
          hardware_budget_rebalance_name(hardware_fast_gi_budget_rebalance_),
          hardware_debug_view_mode_name(hardware_debug_view_mode_),
          hardware_debug_isolate_mode_name(hardware_debug_isolate_mode_),
          hardware_fast_gi_freeze_updates_ ? 1 : 0,
          hardware_fast_gi_smoothed_traced_ms_,
          hardware_direct_light_sample_count_,
          0,
          bricks_per_cascade * cascade_count,
          0,
          cascade_count,
          hardware_fast_gi_requested_cascade_count_,
          hardware_fast_gi_memory_limited_ ? 1 : 0,
          fast_gi_mem_mib,
          fast_gi_budget_mib,
          hardware_fast_gi_scene_radius_,
          hardware_fast_gi_scene_density_,
          hardware_fast_gi_requested_distance_,
          light_invalidation_pending ? 1 : 0,
          world_invalidation_pending ? 1 : 0,
          emissive_invalidation_pending ? 1 : 0,
          material_invalidation_pending ? 1 : 0,
          geometry_invalidation_pending ? 1 : 0,
          animation_invalidation_pending ? 1 : 0,
          emissive_light_count,
          emissive_energy_sum);
    }
    return;
  }
  hardware_fast_gi_cascade_config_fill(data_.hardware_fast_gi_cascade_config,
                                       cascade_count,
                                       hardware_fast_gi_field_center_,
                                       base_cell_size,
                                       grid_resolution,
                                       hardware_fast_gi_field_config_valid_,
                                       hardware_fast_gi_field_cascade_config_);
  inst_.uniform_data.push_update();

  const bool amortize_viewport_update = interactive_viewport;
  const int cascade_begin = amortize_viewport_update ?
                                int(inst_.sampling.sample_index() % uint64_t(cascade_count)) :
                                0;
  const int cascade_end = amortize_viewport_update ? cascade_begin + 1 : cascade_count;
  bool used_traced_fast_gi = false;
  bool used_screen_seed_fast_gi = false;
  const double scene_acquire_start_time = log_fast_gi_stats ? BLI_time_now_seconds() : 0.0;
  GPUMetalRaytraceSceneStats metal_scene_stats;
  GPUMetalRaytraceScene *metal_scene = acquire_hardware_metal_scene(&metal_scene_stats, false);
  if (log_fast_gi_stats) {
    GPU_flush();
    scene_acquire_ms = (BLI_time_now_seconds() - scene_acquire_start_time) * 1000.0;
  }
  emissive_light_count = metal_scene_stats.emissive_light_count;
  emissive_energy_sum = metal_scene_stats.emissive_energy_sum;
  const eHardwareGIProducerBackend gi_producer_backend = hardware_gi_producer_backend(metal_scene);
  const bool traced_fast_gi_available = (gi_producer_backend == HWRT_GI_PRODUCER_BACKEND_METAL_RT);
  const bool use_screen_seed_fast_gi = false && (input_radiance_tx != nullptr) &&
                                       hardware_fast_gi_screen_seed_allowed();
  const bool use_traced_fast_gi = traced_fast_gi_available;
  int active_brick_count = 0;
  int total_brick_count = 0;
  int camera_invalidated_brick_count = 0;
  int reused_brick_count = 0;
  if (use_traced_fast_gi) {
    auto trace_ranked_bricks = [&]() -> bool {
      const int producer_grid_resolution = max_ii(data_.hardware_fast_gi_grid_resolution, 1);
      const int producer_cascade_count = max_ii(data_.hardware_fast_gi_cascade_count, 1);
      const float4 *previous_cascade_config = hardware_fast_gi_field_cascade_config_;
      const float4 *target_cascade_config = data_.hardware_fast_gi_cascade_config;
      const float3 raytrace_rng = inst_.sampling.rng_3d_get(eSamplingDimension::SAMPLING_RAYTRACE_U);
      const float4 sampling_rand = float4(
          raytrace_rng.x,
          raytrace_rng.y,
          raytrace_rng.z,
          inst_.sampling.rng_get(eSamplingDimension::SAMPLING_CLOSURE));
      const bool force_full_refresh = !field_was_valid || geometry_invalidation_pending ||
                                      animation_invalidation_pending;
      const bool prioritize_dirty = force_full_refresh || light_invalidation_pending ||
                                    world_invalidation_pending || emissive_invalidation_pending ||
                                    material_invalidation_pending ||
                                    transform_invalidation_pending;
      const float3 camera_position = inst_.camera.position();
      const float3 camera_forward = math::normalize(inst_.camera.forward());
      Vector<LightData> local_lights;
      /* The diffuse GI producer stays world/probe-owned here; explicit direct-light re-injection
       * remains out of scope for the current traced field. */
      const int fast_gi_light_count = 0;
      const int fast_gi_light_sample_count = 0;
      gpu::StorageBuf *fast_gi_light_buf = nullptr;
      const SphereProbe &world_probe = inst_.sphere_probes.world_sphere_probe();
      const bool world_probe_available = world_probe.atlas_coord.atlas_layer >= 0 &&
                                        world_probe.atlas_coord.subdivision_lvl >= 0;
      const SphereProbeUvArea world_probe_atlas_coord = world_probe_available ?
                                                            world_probe.atlas_coord.
                                                                as_sampling_coord() :
                                                            SphereProbeUvArea{
                                                                float2(0.0f), 0.0f, -1.0f};
      for (int cascade_index = cascade_end - 1; cascade_index >= cascade_begin; cascade_index--) {
        const int update_period = hardware_fast_gi_brick_update_period(
            cascade_index, inst_.is_viewport(), field_was_valid, interactive_viewport);
        const int3 camera_brick_shift = (field_was_valid && field_config_was_valid) ?
                                            hardware_fast_gi_camera_brick_shift(
                                                previous_cascade_config[cascade_index],
                                                target_cascade_config[cascade_index],
                                                brick_resolution) :
                                            int3(0);
        const bool field_mapping_changed =
            field_was_valid && field_config_was_valid &&
            hardware_fast_gi_cascade_config_changed(previous_cascade_config[cascade_index],
                                                    target_cascade_config[cascade_index]);
        const bool cascade_force_full_refresh = force_full_refresh || field_mapping_changed;
        Vector<HardwareFastGIBrickCandidate> ranked_bricks = hardware_fast_gi_ranked_bricks(
            brick_grid_extent,
            brick_resolution,
            target_cascade_config[cascade_index],
            camera_brick_shift,
            field_was_valid,
            field_config_was_valid,
            cascade_force_full_refresh,
            prioritize_dirty,
            camera_position,
            camera_forward,
            sorted_scene_entries,
            local_lights);
        total_brick_count += int(ranked_bricks.size());
        for (const int ranked_index : ranked_bricks.index_range()) {
          const HardwareFastGIBrickCandidate &candidate = ranked_bricks[ranked_index];
          if (candidate.reuse_compatible &&
              !hardware_fast_gi_should_update_brick(
                  inst_.sampling.sample_index(), ranked_index, update_period))
          {
            reused_brick_count++;
            continue;
          }
          active_brick_count++;
          if (candidate.camera_invalidated) {
            camera_invalidated_brick_count++;
          }

          GPUMetalRaytraceFastGIParams fast_gi_params;
          fast_gi_params.fast_gi_history_tx = hardware_fast_gi_tx_;
          fast_gi_params.fast_gi_tx = hardware_fast_gi_tx_;
          fast_gi_params.fast_gi_error_tx = hardware_fast_gi_error_tx_;
          fast_gi_params.fast_gi_visibility_tx = hardware_fast_gi_visibility_tx_;
          fast_gi_params.world_probe_tx = inst_.sphere_probes.octahedral_probes_texture();
          fast_gi_params.light_buf = fast_gi_light_buf;
          fast_gi_params.grid_resolution = producer_grid_resolution;
          fast_gi_params.brick_origin = candidate.brick_coord * brick_resolution;
          fast_gi_params.brick_extent = int3(min_ii(brick_resolution,
                                                    producer_grid_resolution -
                                                        fast_gi_params.brick_origin.x),
                                              min_ii(brick_resolution,
                                                    producer_grid_resolution -
                                                        fast_gi_params.brick_origin.y),
                                              min_ii(brick_resolution,
                                                    producer_grid_resolution -
                                                        fast_gi_params.brick_origin.z));
          fast_gi_params.cascade_index = cascade_index;
          fast_gi_params.cascade_count = producer_cascade_count;
          fast_gi_params.sample_count = hardware_visibility_temporal_sample_count;
          fast_gi_params.gi_bounces = hardware_gi_fixed_bounces;
          fast_gi_params.light_count = fast_gi_light_count;
          fast_gi_params.light_sample_count = fast_gi_light_sample_count;
          fast_gi_params.normal_bias = max_ff(
              target_cascade_config[cascade_index].w * 0.05f, 1.0e-3f);
          fast_gi_params.reuse_history = candidate.reuse_compatible;
          fast_gi_params.use_environment = use_hardware_gi() || use_hardware_environment();
          fast_gi_params.sampling_rand = sampling_rand;
          fast_gi_params.world_probe_atlas_coord = float4(world_probe_atlas_coord.offset.x,
                                                          world_probe_atlas_coord.offset.y,
                                                          world_probe_atlas_coord.scale,
                                                          world_probe_atlas_coord.layer);
          for (const int config_index : IndexRange(3)) {
            fast_gi_params.cascade_config[config_index] = target_cascade_config[config_index];
          }
          if (!hardware_gi_trace_current_field(gi_producer_backend, metal_scene, fast_gi_params)) {
            return false;
          }
        }
      }
      return true;
    };

    const double traced_start_time = BLI_time_now_seconds();
    const bool current_field_update_succeeded = trace_ranked_bricks();
    traced_ms = (BLI_time_now_seconds() - traced_start_time) * 1000.0;
    used_traced_fast_gi = current_field_update_succeeded;
  }

  if (use_screen_seed_fast_gi) {
    const double screen_seed_start_time = log_fast_gi_stats ? BLI_time_now_seconds() : 0.0;
    used_screen_seed_fast_gi = true;
    for (const int cascade_index : IndexRange(cascade_begin, cascade_end - cascade_begin)) {
      PassSimple &pass = hardware_fast_gi_update_ps_[cascade_index];
      pass.init();
      gpu::Shader *sh = inst_.shaders.static_shader_get(RAY_HARDWARE_FAST_GI_UPDATE);
      pass.specialize_constant(sh, "cascade_index", &hardware_fast_gi_cascade_index_[cascade_index]);
      pass.shader_set(sh);
      pass.bind_texture("depth_tx", depth_tx);
      pass.bind_texture("input_radiance_tx", input_radiance_tx);
      pass.bind_image("out_fast_gi_img", &hardware_fast_gi_tx_);
      pass.bind_image("out_fast_gi_visibility_img", &hardware_fast_gi_visibility_tx_);
      pass.bind_resources(inst_.uniform_data);
      pass.dispatch(dispatch_size);
      inst_.manager->submit(pass, render_view);
    }
    GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH | GPU_BARRIER_SHADER_IMAGE_ACCESS);
    if (log_fast_gi_stats) {
      GPU_flush();
      screen_seed_ms = (BLI_time_now_seconds() - screen_seed_start_time) * 1000.0;
    }
  }

  bool next_fast_gi_valid = false;
  if (used_traced_fast_gi || used_screen_seed_fast_gi) {
    /* A screen-seeded refresh may preserve a previously warm traced field for continuity, but it
     * never upgrades a cold field to authoritative truth on its own. */
    next_fast_gi_valid = used_traced_fast_gi || (used_screen_seed_fast_gi && field_was_valid);
  }
  else if (interactive_viewport) {
    /* Keep the last warm field alive during a bad live update instead of immediately failing the
     * diffuse owner closed to black while the viewport is moving. */
    next_fast_gi_valid = field_was_valid;
  }
  hardware_fast_gi_valid_ = next_fast_gi_valid;
  if (used_traced_fast_gi) {
    if (used_traced_fast_gi) {
      hardware_fast_gi_depsgraph_update_count_ = depsgraph_update_count;
      hardware_fast_gi_depsgraph_update_count_valid_ = true;
    }
    else if (!hardware_fast_gi_valid_) {
      hardware_fast_gi_depsgraph_update_count_valid_ = false;
    }
    for (const int cascade_index : IndexRange(cascade_count)) {
      if (used_traced_fast_gi) {
        hardware_fast_gi_field_cascade_config_[cascade_index] =
            data_.hardware_fast_gi_cascade_config[cascade_index];
      }
    }
    if (used_traced_fast_gi) {
      hardware_fast_gi_field_config_valid_ = true;
    }
    else if (!hardware_fast_gi_valid_) {
      hardware_fast_gi_field_config_valid_ = false;
    }
    hardware_fast_gi_light_invalidation_pending_ = false;
    hardware_fast_gi_world_invalidation_pending_ = false;
    hardware_fast_gi_emissive_invalidation_pending_ = false;
    hardware_fast_gi_material_invalidation_pending_ = false;
    hardware_fast_gi_transform_invalidation_pending_ = false;
    hardware_fast_gi_geometry_invalidation_pending_ = false;
    hardware_fast_gi_animation_invalidation_pending_ = false;
  }
  else {
    if (!hardware_fast_gi_valid_) {
      hardware_fast_gi_depsgraph_update_count_valid_ = false;
      hardware_fast_gi_field_config_valid_ = false;
    }
  }
  if (used_traced_fast_gi && traced_ms > 0.0) {
    hardware_fast_gi_smoothed_traced_ms_ = (hardware_fast_gi_smoothed_traced_ms_ > 0.0f) ?
                                               (hardware_fast_gi_smoothed_traced_ms_ * 0.75f +
                                                float(traced_ms) * 0.25f) :
                                               float(traced_ms);
  }

  if (log_fast_gi_stats) {
    GPU_flush();
    const double elapsed_ms = (BLI_time_now_seconds() - update_start_time) * 1000.0;
    std::fprintf(stderr,
                 "EEVEE HWRT FastGI stats viewport=%d interactive=%d sample=%llu "
                 "tier=%s scene=%s budget=%s debug=%s isolate=%s freeze=%d smoothed_traced_ms=%.2f direct_samples=%d grid=%d/%d brick_res=%d active_bricks=%d reused_bricks=%d camera_invalidated=%d light_invalidated=%d world_invalidated=%d emissive_invalidated=%d material_invalidated=%d geometry_invalidated=%d animation_invalidated=%d total_bricks=%d active_cascades=%d/%d memory_limited=%d fast_gi_mem_mib=%.2f budget_mib=%.2f scene_radius=%.2f density=%.3f field_distance=%.2f updated_range=[%d,%d) dispatch=%dx%dx%d traced=%d screen_seed=%d field_valid=%d scene_acquire_ms=%.2f traced_ms=%.2f screen_seed_ms=%.2f elapsed_ms=%.2f\n",
                 inst_.is_viewport() ? 1 : 0,
                 interactive_viewport ? 1 : 0,
                 (unsigned long long)inst_.sampling.sample_index(),
                 hardware_quality_tier_name(hardware_fast_gi_quality_tier_),
                 hardware_scene_priority_name(hardware_fast_gi_scene_priority_),
                 hardware_budget_rebalance_name(hardware_fast_gi_budget_rebalance_),
                 hardware_debug_view_mode_name(hardware_debug_view_mode_),
                 hardware_debug_isolate_mode_name(hardware_debug_isolate_mode_),
                 hardware_fast_gi_freeze_updates_ ? 1 : 0,
                 hardware_fast_gi_smoothed_traced_ms_,
                 hardware_direct_light_sample_count_,
                 grid_resolution,
                 hardware_fast_gi_requested_grid_resolution_,
                 brick_resolution,
                 active_brick_count,
                 reused_brick_count,
                 camera_invalidated_brick_count,
                 light_invalidation_pending ? 1 : 0,
                 world_invalidation_pending ? 1 : 0,
                 emissive_invalidation_pending ? 1 : 0,
                 material_invalidation_pending ? 1 : 0,
                 geometry_invalidation_pending ? 1 : 0,
                 animation_invalidation_pending ? 1 : 0,
                 use_traced_fast_gi ? total_brick_count : 0,
                 cascade_count,
                 hardware_fast_gi_requested_cascade_count_,
                 hardware_fast_gi_memory_limited_ ? 1 : 0,
                 fast_gi_mem_mib,
                 fast_gi_budget_mib,
                 hardware_fast_gi_scene_radius_,
                 hardware_fast_gi_scene_density_,
                 hardware_fast_gi_requested_distance_,
                 cascade_begin,
                 cascade_end,
                 dispatch_size.x,
                 dispatch_size.y,
                 dispatch_size.z,
                 used_traced_fast_gi ? 1 : 0,
                 used_screen_seed_fast_gi ? 1 : 0,
                 hardware_fast_gi_valid_ ? 1 : 0,
                 scene_acquire_ms,
                 traced_ms,
                 screen_seed_ms,
                 elapsed_ms);
  }
  if (inst_.is_viewport() && hardware_fast_gi_debug_overlay_enabled()) {
    const char *producer = used_traced_fast_gi ? "traced" :
                           (used_screen_seed_fast_gi ? "screen" : "none");
    inst_.info_append(
        "Fast GI Debug: valid={} producer={} tier={} scene={} budget={} debug={} isolate={} freeze={} traced_ms={:.2f} direct_samples={} camera_dependent={} bricks={}/{} reused_bricks={} resident_cascades={}/{} memory_limited={} mem_mib={:.2f}/{:.2f} scene_radius={:.2f} density={:.3f} field_distance={:.2f} camera_invalidated={} light_invalidated={} world_invalidated={} emissive_invalidated={} material_invalidated={} geometry_invalidated={} animation_invalidated={} emissive_lights={} emissive_energy={:.3f}",
        hardware_fast_gi_valid_ ? 1 : 0,
        producer,
        hardware_quality_tier_name(hardware_fast_gi_quality_tier_),
        hardware_scene_priority_name(hardware_fast_gi_scene_priority_),
        hardware_budget_rebalance_name(hardware_fast_gi_budget_rebalance_),
        hardware_debug_view_mode_name(hardware_debug_view_mode_),
        hardware_debug_isolate_mode_name(hardware_debug_isolate_mode_),
        hardware_fast_gi_freeze_updates_ ? 1 : 0,
        hardware_fast_gi_smoothed_traced_ms_,
        hardware_direct_light_sample_count_,
        used_screen_seed_fast_gi ? 1 : 0,
        active_brick_count,
        use_traced_fast_gi ? total_brick_count : 0,
        reused_brick_count,
        cascade_count,
        hardware_fast_gi_requested_cascade_count_,
        hardware_fast_gi_memory_limited_ ? 1 : 0,
        fast_gi_mem_mib,
        fast_gi_budget_mib,
        hardware_fast_gi_scene_radius_,
        hardware_fast_gi_scene_density_,
        hardware_fast_gi_requested_distance_,
        camera_invalidated_brick_count,
        light_invalidation_pending ? 1 : 0,
        world_invalidation_pending ? 1 : 0,
        emissive_invalidation_pending ? 1 : 0,
        material_invalidation_pending ? 1 : 0,
        geometry_invalidation_pending ? 1 : 0,
        animation_invalidation_pending ? 1 : 0,
        emissive_light_count,
        emissive_energy_sum);
  }
}

void RayTraceModule::render_directional_shadow_visibility(View &render_view,
                                                          gpu::Texture *depth_tx,
                                                          gpu::Texture *gbuf_normal_tx,
                                                          int2 extent)
{
  const bool perf_logging_enabled = hardware_perf_logging_enabled();
  const double perf_start_time = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  const float visibility = 1.0f;
  const bool use_world_rt_shadows = use_hardware_environment();
  const bool use_direct_rt_shadows = use_hardware_shadows();
  auto mark_shadow_visibility_ready = [&]() {
    hardware_primary_shadow_visibility_ready_ = true;
    hardware_primary_shadow_visibility_depth_tx_ = depth_tx;
    hardware_primary_shadow_visibility_normal_tx_ = gbuf_normal_tx;
    hardware_primary_shadow_visibility_extent_ = extent;
    hardware_primary_shadow_direct_enabled_ = use_direct_rt_shadows;
    hardware_primary_shadow_world_enabled_ = use_world_rt_shadows;
  };
  const bool reuse_shadow_visibility = hardware_primary_shadow_visibility_ready_ &&
                                       hardware_primary_shadow_visibility_depth_tx_ == depth_tx &&
                                       hardware_primary_shadow_visibility_normal_tx_ ==
                                           gbuf_normal_tx &&
                                       hardware_primary_shadow_visibility_extent_ == extent &&
                                       hardware_primary_shadow_direct_enabled_ ==
                                           use_direct_rt_shadows &&
                                       hardware_primary_shadow_world_enabled_ ==
                                           use_world_rt_shadows;
  if (reuse_shadow_visibility) {
    if (perf_logging_enabled) {
      const double elapsed_ms = (BLI_time_now_seconds() - perf_start_time) * 1000.0;
      std::fprintf(stderr,
                   "EEVEE HWRT perf primary_shadows reused=1 direct=%d world=%d elapsed_ms=%.2f\n",
                   use_direct_rt_shadows ? 1 : 0,
                   use_world_rt_shadows ? 1 : 0,
                   elapsed_ms);
    }
    return;
  }

  if (!use_hardware_tracing() || (!use_direct_rt_shadows && !use_world_rt_shadows) ||
      depth_tx == nullptr ||
      gbuf_normal_tx == nullptr ||
      (inst_.lights.sun_lights_len() + inst_.lights.local_lights_len()) == 0)
  {
    const float4 zero = float4(0.0f);
    hardware_direct_light_accum_tx_.ensure_2d(
        gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, int2(1), GPU_TEXTURE_USAGE_SHADER_READ, zero);
    hardware_direct_light_denoised_tx_.ensure_2d(
        gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, int2(1), GPU_TEXTURE_USAGE_SHADER_READ, zero);
    hardware_shadow_visibility_tx_.ensure_2d_array(
        gpu::TextureFormat::SFLOAT_16, int2(1), 1, GPU_TEXTURE_USAGE_SHADER_READ, &visibility);
    mark_shadow_visibility_ready();
    return;
  }

  update_hardware_tracing_scene_state();
  if (hardware_scene_entry_count_ == 0) {
    const float4 zero = float4(0.0f);
    hardware_direct_light_accum_tx_.ensure_2d(
        gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, int2(1), GPU_TEXTURE_USAGE_SHADER_READ, zero);
    hardware_direct_light_denoised_tx_.ensure_2d(
        gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, int2(1), GPU_TEXTURE_USAGE_SHADER_READ, zero);
    hardware_shadow_visibility_tx_.ensure_2d_array(
        gpu::TextureFormat::SFLOAT_16, int2(1), 1, GPU_TEXTURE_USAGE_SHADER_READ, &visibility);
    mark_shadow_visibility_ready();
    return;
  }

  constexpr eGPUTextureUsage usage_rw = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;
  const int local_light_count = inst_.lights.local_lights_len();
  const int sun_light_count = inst_.lights.sun_lights_len();
  const int total_light_count = local_light_count + sun_light_count;
  const int hwrt_shadow_sample_count = hardware_visibility_temporal_sample_count;
  const int hwrt_world_shadow_sample_count = hardware_visibility_temporal_sample_count;
  eGPUTextureUsage direct_light_output_usage = usage_rw;
  if (total_light_count == 0) {
    const float4 zero = float4(0.0f);
    hardware_direct_light_accum_tx_.ensure_2d(
        gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, int2(1), GPU_TEXTURE_USAGE_SHADER_READ, zero);
    hardware_direct_light_denoised_tx_.ensure_2d(
        gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, int2(1), GPU_TEXTURE_USAGE_SHADER_READ, zero);
    hardware_shadow_visibility_tx_.ensure_2d_array(
        gpu::TextureFormat::SFLOAT_16, int2(1), 1, GPU_TEXTURE_USAGE_SHADER_READ, &visibility);
    mark_shadow_visibility_ready();
    return;
  }
  const LightCullingData &light_culling_data = inst_.lights.culling_data();
  const int2 direct_light_tile_extent = int2(light_culling_data.tile_x_len, light_culling_data.tile_y_len);
  hardware_shadow_visibility_tx_.ensure_2d_array(
      gpu::TextureFormat::SFLOAT_16, extent, total_light_count, usage_rw);
  hardware_direct_light_accum_tx_.ensure_2d(
      gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, extent, direct_light_output_usage);
  hardware_direct_light_denoised_tx_.ensure_2d(
      gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, extent, direct_light_output_usage);
  hardware_direct_light_depth_tx_.ensure_2d(gpu::TextureFormat::SFLOAT_32, extent, usage_rw);
  hardware_direct_light_tilemask_tx_.ensure_2d(
      gpu::TextureFormat::RAYTRACE_TILEMASK_FORMAT, direct_light_tile_extent, usage_rw);
  hardware_shadow_visibility_tx_.clear(float4(1.0f));
  hardware_direct_light_accum_tx_.clear(float4(0.0f));
  hardware_direct_light_denoised_tx_.clear(float4(0.0f));
  hardware_direct_light_depth_tx_.clear(float4(0.0f));
  hardware_direct_light_tilemask_tx_.clear(uint4(0u));
  GPU_flush();

  Vector<LightData> local_lights;
  Vector<LightData> sun_lights;
  Vector<int> sun_light_world_slots;
  inst_.lights.append_sync_local_lights(local_lights);
  inst_.lights.append_sync_sun_lights(sun_lights, &sun_light_world_slots);

  const double scene_acquire_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  GPUMetalRaytraceSceneStats metal_scene_stats;
  GPUMetalRaytraceScene *metal_scene = acquire_hardware_metal_scene(&metal_scene_stats, false);
  const double scene_acquire_ms = perf_logging_enabled ?
                                      (BLI_time_now_seconds() - scene_acquire_start) * 1000.0 :
                                      0.0;
  if (metal_scene == nullptr || !metal_scene_stats.built_scene) {
    return;
  }

  int traced_local_lights = 0;
  int traced_sun_lights = 0;
  bool shadow_trace_failed = false;
  const double trace_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  const float4 shadow_sampling_rand = hardware_shadow_sampling_rand(inst_);
  GPU_debug_group_begin("Hardware RT Shadows");
  const bool shadow_batch_active = GPU_metal_raytrace_scene_shadow_batch_begin(metal_scene);
  if (use_direct_rt_shadows) {
    for (const int local_index : local_lights.index_range()) {
      const LightData &light = local_lights[local_index];
      if (light.tilemap_index == LIGHT_NO_SHADOW || light.color.x < 0.0f) {
        continue;
      }

      GPUMetalRaytraceLocalShadowParams shadow_params;
      shadow_params.depth_tx = depth_tx;
      shadow_params.gbuf_header_tx = inst_.gbuffer.header_tx;
      shadow_params.gbuf_normal_tx = gbuf_normal_tx;
      shadow_params.shadow_visibility_tx = hardware_shadow_visibility_tx_;
      shadow_params.viewinv = render_view.viewinv();
      shadow_params.wininv = render_view.wininv();
      shadow_params.full_resolution = extent;
      shadow_params.shadow_layer = local_index;
      shadow_params.sample_count = hwrt_shadow_sample_count;
      shadow_params.light_type = uint32_t(light.type);
      shadow_params.light_position = light_position_get(light);
      shadow_params.shadow_radius = light.local().local.shadow_radius;
      shadow_params.light_x_axis = light_x_axis(light);
      shadow_params.area_size_x = 0.0f;
      shadow_params.light_y_axis = light_y_axis(light);
      shadow_params.area_size_y = 0.0f;
      shadow_params.shadow_offset = light_x_axis(light) * light.local().local.shadow_position.x +
                                    light_y_axis(light) * light.local().local.shadow_position.y +
                                    light_z_axis(light) * light.local().local.shadow_position.z;
      shadow_params.area_shadow_scale = 1.0f;
      if (is_area_light(light.type)) {
        shadow_params.area_size_x = light.area().size.x;
        shadow_params.area_size_y = light.area().size.y;
        shadow_params.area_shadow_scale = light.area().shadow_scale;
      }
      shadow_params.normal_bias = std::max(4.0e-3f, light.filter_radius * 4.0e-3f);
      shadow_params.sampling_rand = shadow_sampling_rand;
      const bool trace_submitted = GPU_metal_raytrace_scene_trace_local_shadow(metal_scene,
                                                                               shadow_params);
      traced_local_lights += trace_submitted ? 1 : 0;
      shadow_trace_failed |= !trace_submitted;
    }
  }
  for (const int sun_index : sun_lights.index_range()) {
    const LightData &light = sun_lights[sun_index];
    if (!is_sun_light(light.type) ||
        (light.tilemap_index == LIGHT_NO_SHADOW && light.color.x >= 0.0f))
    {
      continue;
    }
    const bool is_world_sun = light.color.x < 0.0f;
    if ((is_world_sun && !use_world_rt_shadows) || (!is_world_sun && !use_direct_rt_shadows)) {
      continue;
    }

    float3 light_direction = normalize(light_z_axis(light));
    int world_sun_slot = -1;
    if (light.color.x < 0.0f) {
      world_sun_slot = (sun_index < sun_light_world_slots.size()) ? sun_light_world_slots[sun_index] :
                                                                 -1;
    }

    GPUMetalRaytraceDirectionalShadowParams shadow_params;
    shadow_params.depth_tx = depth_tx;
    shadow_params.gbuf_header_tx = inst_.gbuffer.header_tx;
    shadow_params.gbuf_normal_tx = gbuf_normal_tx;
    shadow_params.shadow_visibility_tx = hardware_shadow_visibility_tx_;
    shadow_params.world_sunlight_direction_buf = inst_.world.sunlight_rt_direction;
    shadow_params.viewinv = render_view.viewinv();
    shadow_params.wininv = render_view.wininv();
    shadow_params.full_resolution = extent;
    shadow_params.shadow_layer = local_light_count + sun_index;
    shadow_params.world_sun_slot = world_sun_slot;
    shadow_params.sample_count = is_world_sun ? hwrt_world_shadow_sample_count :
                                                hwrt_shadow_sample_count;
    shadow_params.light_direction = light_direction;
    shadow_params.normal_bias = std::max(5.0e-3f, light.filter_radius * 5.0e-3f);
    shadow_params.shadow_angle = light.sun().shadow_angle;
    shadow_params.sampling_rand = shadow_sampling_rand;
    const bool trace_submitted = GPU_metal_raytrace_scene_trace_directional_shadow(metal_scene,
                                                                                    shadow_params);
    traced_sun_lights += trace_submitted ? 1 : 0;
    shadow_trace_failed |= !trace_submitted;
  }
  const bool shadow_batch_committed = shadow_batch_active ?
                                          GPU_metal_raytrace_scene_shadow_batch_end(metal_scene) :
                                          true;
  GPU_debug_group_end();
  if (!shadow_trace_failed && shadow_batch_committed) {
    mark_shadow_visibility_ready();
  }
  if (perf_logging_enabled) {
    const double trace_ms = (BLI_time_now_seconds() - trace_start) * 1000.0;
    const double elapsed_ms = (BLI_time_now_seconds() - perf_start_time) * 1000.0;
    std::fprintf(stderr,
                 "EEVEE HWRT perf primary_shadows reused=0 local=%d sun=%d batched=%d committed=%d scene_acquire_ms=%.2f trace_submit_ms=%.2f elapsed_ms=%.2f\n",
                 traced_local_lights,
                 traced_sun_lights,
                 shadow_batch_active ? 1 : 0,
                 shadow_batch_committed ? 1 : 0,
                 scene_acquire_ms,
                 trace_ms,
                 elapsed_ms);
  }
  if (!inst_.is_viewport() || !use_direct_rt_shadows) {
    return;
  }
  inst_.manager->submit(hardware_direct_light_visibility_ps_);
  inst_.manager->submit(hardware_direct_light_accum_ps_, render_view);
  inst_.manager->submit(hardware_direct_light_denoise_ps_, render_view);
}

void RayTraceModule::render_environment_visibility(View &render_view,
                                                   gpu::Texture *depth_tx,
                                                   gpu::Texture *gbuf_normal_tx,
                                                   int2 extent)
{
  const bool perf_logging_enabled = hardware_perf_logging_enabled();
  const double perf_start_time = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  const float visibility[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  const bool use_hw_environment = use_hardware_environment();
  auto mark_environment_visibility_ready = [&]() {
    hardware_primary_environment_visibility_ready_ = true;
    hardware_primary_environment_visibility_depth_tx_ = depth_tx;
    hardware_primary_environment_visibility_normal_tx_ = gbuf_normal_tx;
    hardware_primary_environment_visibility_extent_ = extent;
    hardware_primary_environment_enabled_ = use_hw_environment;
  };
  const bool reuse_environment_visibility =
      hardware_primary_environment_visibility_ready_ &&
      hardware_primary_environment_visibility_depth_tx_ == depth_tx &&
      hardware_primary_environment_visibility_normal_tx_ == gbuf_normal_tx &&
      hardware_primary_environment_visibility_extent_ == extent &&
      hardware_primary_environment_enabled_ == use_hw_environment;
  if (reuse_environment_visibility) {
    if (perf_logging_enabled) {
      const double elapsed_ms = (BLI_time_now_seconds() - perf_start_time) * 1000.0;
      std::fprintf(stderr,
                   "EEVEE HWRT perf primary_environment reused=1 enabled=%d elapsed_ms=%.2f\n",
                   use_hw_environment ? 1 : 0,
                   elapsed_ms);
    }
    return;
  }
  if (!use_hardware_tracing() || !use_hardware_environment() || depth_tx == nullptr ||
      gbuf_normal_tx == nullptr)
  {
    hardware_environment_visibility_tx_.ensure_2d(
        gpu::TextureFormat::SFLOAT_16_16_16_16,
        int2(1),
        GPU_TEXTURE_USAGE_SHADER_READ,
        visibility);
    mark_environment_visibility_ready();
    return;
  }

  update_hardware_tracing_scene_state();
  if (hardware_scene_entry_count_ == 0) {
    hardware_environment_visibility_tx_.ensure_2d(
        gpu::TextureFormat::SFLOAT_16_16_16_16,
        int2(1),
        GPU_TEXTURE_USAGE_SHADER_READ,
        visibility);
    mark_environment_visibility_ready();
    return;
  }

  constexpr eGPUTextureUsage usage_rw = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;
  hardware_environment_visibility_tx_.ensure_2d(
      gpu::TextureFormat::SFLOAT_16_16_16_16, extent, usage_rw);
  hardware_environment_visibility_tx_.clear(float4(0.0f, 0.0f, 0.0f, 1.0f));
  GPU_flush();

  const double scene_acquire_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  GPUMetalRaytraceSceneStats metal_scene_stats;
  GPUMetalRaytraceScene *metal_scene = acquire_hardware_metal_scene(
      &metal_scene_stats, false);
  const double scene_acquire_ms = perf_logging_enabled ?
                                      (BLI_time_now_seconds() - scene_acquire_start) * 1000.0 :
                                      0.0;
  if (metal_scene == nullptr || !metal_scene_stats.built_scene) {
    return;
  }

  GPUMetalRaytraceEnvironmentVisibilityParams env_params;
  env_params.depth_tx = depth_tx;
  env_params.gbuf_header_tx = inst_.gbuffer.header_tx;
  env_params.gbuf_normal_tx = gbuf_normal_tx;
  env_params.environment_visibility_tx = hardware_environment_visibility_tx_;
  env_params.viewinv = render_view.viewinv();
  env_params.wininv = render_view.wininv();
  env_params.full_resolution = extent;
  env_params.sample_count = hardware_visibility_temporal_sample_count;
  env_params.normal_bias = 5.0e-3f;
  env_params.sampling_rand = hardware_shadow_sampling_rand(inst_);
  const double trace_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  const bool trace_submitted = GPU_metal_raytrace_scene_trace_environment_visibility(metal_scene,
                                                                                     env_params);
  if (trace_submitted) {
    GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH | GPU_BARRIER_SHADER_IMAGE_ACCESS);
    mark_environment_visibility_ready();
  }
  if (perf_logging_enabled) {
    const double trace_ms = (BLI_time_now_seconds() - trace_start) * 1000.0;
    const double elapsed_ms = (BLI_time_now_seconds() - perf_start_time) * 1000.0;
    std::fprintf(stderr,
                 "EEVEE HWRT perf primary_environment reused=0 committed=%d scene_acquire_ms=%.2f trace_submit_ms=%.2f elapsed_ms=%.2f\n",
                 trace_submitted ? 1 : 0,
                 scene_acquire_ms,
                 trace_ms,
                 elapsed_ms);
  }
}

void RayTraceModule::render_secondary_environment_visibility(GPUMetalRaytraceScene *metal_scene,
                                                             int2 tracing_extent)
{
  const float visibility[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  if (!use_hardware_tracing() || !use_hardware_environment() || metal_scene == nullptr ||
      tracing_extent.x <= 0 || tracing_extent.y <= 0)
  {
    hardware_secondary_environment_visibility_tx_.ensure_2d(
        gpu::TextureFormat::SFLOAT_16_16_16_16,
        int2(1),
        GPU_TEXTURE_USAGE_SHADER_READ,
        visibility);
    return;
  }

  constexpr eGPUTextureUsage usage_rw = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;
  hardware_secondary_environment_visibility_tx_.ensure_2d(
      gpu::TextureFormat::SFLOAT_16_16_16_16, tracing_extent, usage_rw);
  hardware_secondary_environment_visibility_tx_.clear(float4(0.0f, 0.0f, 0.0f, 1.0f));
  GPU_flush();

  GPUMetalRaytraceHitEnvironmentVisibilityParams env_params;
  env_params.hit_normal_tx = hit_normal_tx_;
  env_params.hit_world_position_tx = hit_world_position_tx_;
  env_params.environment_visibility_tx = hardware_secondary_environment_visibility_tx_;
  env_params.dispatch_buf = hardware_resolve_dispatch_buf_;
  env_params.tiles_coord_buf = hardware_resolve_tiles_buf_;
  env_params.tracing_resolution = tracing_extent;
  env_params.sample_count = hardware_visibility_temporal_sample_count;
  env_params.normal_bias = 5.0e-3f;
  env_params.sampling_rand = hardware_shadow_sampling_rand(inst_);
  if (GPU_metal_raytrace_scene_trace_hit_environment_visibility(metal_scene, env_params)) {
    GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH | GPU_BARRIER_SHADER_IMAGE_ACCESS);
  }
}

void RayTraceModule::render_secondary_shadow_visibility(GPUMetalRaytraceScene *metal_scene,
                                                        int2 tracing_extent)
{
  const bool perf_logging_enabled = hardware_perf_logging_enabled();
  const double perf_start_time = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  const float visibility = 1.0f;
  const bool use_world_rt_shadows = use_hardware_environment();
  const bool use_direct_rt_shadows = use_hardware_shadows();
  if (!use_hardware_tracing() || (!use_direct_rt_shadows && !use_world_rt_shadows) ||
      metal_scene == nullptr ||
      tracing_extent.x <= 0 || tracing_extent.y <= 0)
  {
    hardware_secondary_shadow_visibility_tx_.ensure_2d_array(
        gpu::TextureFormat::SFLOAT_16, int2(1), 1, GPU_TEXTURE_USAGE_SHADER_READ, &visibility);
    return;
  }

  const int local_light_count = inst_.lights.local_lights_len();
  const int sun_light_count = inst_.lights.sun_lights_len();
  const int total_light_count = local_light_count + sun_light_count;
  const int hwrt_shadow_sample_count = hardware_visibility_temporal_sample_count;
  const int hwrt_world_shadow_sample_count = hardware_visibility_temporal_sample_count;
  if (total_light_count == 0) {
    hardware_secondary_shadow_visibility_tx_.ensure_2d_array(
        gpu::TextureFormat::SFLOAT_16, int2(1), 1, GPU_TEXTURE_USAGE_SHADER_READ, &visibility);
    return;
  }

  constexpr eGPUTextureUsage usage_rw = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;
  hardware_secondary_shadow_visibility_tx_.ensure_2d_array(
      gpu::TextureFormat::SFLOAT_16, tracing_extent, total_light_count, usage_rw);
  hardware_secondary_shadow_visibility_tx_.clear(float4(1.0f));
  GPU_flush();

  Vector<LightData> local_lights;
  Vector<LightData> sun_lights;
  Vector<int> sun_light_world_slots;
  inst_.lights.append_sync_local_lights(local_lights);
  inst_.lights.append_sync_sun_lights(sun_lights, &sun_light_world_slots);

  int traced_local_lights = 0;
  int traced_sun_lights = 0;
  const float4 shadow_sampling_rand = hardware_shadow_sampling_rand(inst_);
  GPU_debug_group_begin("Hardware RT Hit Shadows");
  const bool shadow_batch_active = GPU_metal_raytrace_scene_shadow_batch_begin(metal_scene);
  if (use_direct_rt_shadows) {
    for (const int local_index : local_lights.index_range()) {
      const LightData &light = local_lights[local_index];
      if (light.tilemap_index == LIGHT_NO_SHADOW || light.color.x < 0.0f) {
        continue;
      }

      GPUMetalRaytraceLocalHitShadowParams shadow_params;
      shadow_params.hit_normal_tx = hit_normal_tx_;
      shadow_params.hit_world_position_tx = hit_world_position_tx_;
      shadow_params.hit_identity_tx = hit_identity_tx_;
      shadow_params.shadow_visibility_tx = hardware_secondary_shadow_visibility_tx_;
      shadow_params.dispatch_buf = hardware_resolve_dispatch_buf_;
      shadow_params.tiles_coord_buf = hardware_resolve_tiles_buf_;
      shadow_params.tracing_resolution = tracing_extent;
      shadow_params.shadow_layer = local_index;
      shadow_params.sample_count = hwrt_shadow_sample_count;
      shadow_params.light_type = uint32_t(light.type);
      shadow_params.light_position = light_position_get(light);
      shadow_params.shadow_radius = light.local().local.shadow_radius;
      shadow_params.light_x_axis = light_x_axis(light);
      shadow_params.area_size_x = 0.0f;
      shadow_params.light_y_axis = light_y_axis(light);
      shadow_params.area_size_y = 0.0f;
      shadow_params.shadow_offset = light_x_axis(light) * light.local().local.shadow_position.x +
                                    light_y_axis(light) * light.local().local.shadow_position.y +
                                    light_z_axis(light) * light.local().local.shadow_position.z;
      shadow_params.area_shadow_scale = 1.0f;
      if (is_area_light(light.type)) {
        shadow_params.area_size_x = light.area().size.x;
        shadow_params.area_size_y = light.area().size.y;
        shadow_params.area_shadow_scale = light.area().shadow_scale;
      }
      shadow_params.normal_bias = std::max(4.0e-3f, light.filter_radius * 4.0e-3f);
      shadow_params.sampling_rand = shadow_sampling_rand;
      traced_local_lights +=
          GPU_metal_raytrace_scene_trace_local_hit_shadow(metal_scene, shadow_params) ? 1 : 0;
    }
  }

  for (const int sun_index : sun_lights.index_range()) {
    const LightData &light = sun_lights[sun_index];
    if (!is_sun_light(light.type) ||
        (light.tilemap_index == LIGHT_NO_SHADOW && light.color.x >= 0.0f))
    {
      continue;
    }
    const bool is_world_sun = light.color.x < 0.0f;
    if ((is_world_sun && !use_world_rt_shadows) || (!is_world_sun && !use_direct_rt_shadows)) {
      continue;
    }

    float3 light_direction = normalize(light_z_axis(light));
    int world_sun_slot = -1;
    if (light.color.x < 0.0f) {
      world_sun_slot = (sun_index < sun_light_world_slots.size()) ? sun_light_world_slots[sun_index] :
                                                                 -1;
    }

    GPUMetalRaytraceDirectionalHitShadowParams shadow_params;
    shadow_params.hit_normal_tx = hit_normal_tx_;
    shadow_params.hit_world_position_tx = hit_world_position_tx_;
    shadow_params.hit_identity_tx = hit_identity_tx_;
    shadow_params.shadow_visibility_tx = hardware_secondary_shadow_visibility_tx_;
    shadow_params.dispatch_buf = hardware_resolve_dispatch_buf_;
    shadow_params.tiles_coord_buf = hardware_resolve_tiles_buf_;
    shadow_params.world_sunlight_direction_buf = inst_.world.sunlight_rt_direction;
    shadow_params.tracing_resolution = tracing_extent;
    shadow_params.shadow_layer = local_light_count + sun_index;
    shadow_params.world_sun_slot = world_sun_slot;
    shadow_params.sample_count = is_world_sun ? hwrt_world_shadow_sample_count :
                                                hwrt_shadow_sample_count;
    shadow_params.light_direction = light_direction;
    shadow_params.normal_bias = std::max(5.0e-3f, light.filter_radius * 5.0e-3f);
    shadow_params.shadow_angle = light.sun().shadow_angle;
    shadow_params.sampling_rand = shadow_sampling_rand;
    traced_sun_lights += GPU_metal_raytrace_scene_trace_directional_hit_shadow(metal_scene,
                                                                               shadow_params) ?
                             1 :
                             0;
  }
  const bool shadow_batch_committed = shadow_batch_active ?
                                          GPU_metal_raytrace_scene_shadow_batch_end(metal_scene) :
                                          true;
  GPU_debug_group_end();
  if (perf_logging_enabled) {
    const double elapsed_ms = (BLI_time_now_seconds() - perf_start_time) * 1000.0;
    std::fprintf(stderr,
                 "EEVEE HWRT perf secondary_shadows local=%d sun=%d batched=%d committed=%d elapsed_ms=%.2f\n",
                 traced_local_lights,
                 traced_sun_lights,
                 shadow_batch_active ? 1 : 0,
                 shadow_batch_committed ? 1 : 0,
                 elapsed_ms);
  }
}

void RayTraceModule::submit_hardware_tracing_backend(View &render_view)
{
  const bool perf_logging_enabled = hardware_perf_logging_enabled();
  const double perf_start_time = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  update_hardware_tracing_scene_state();
  const uint32_t specular_feature_mask = RAYTRACE_EEVEE_HARDWARE_REFLECTIONS |
                                         RAYTRACE_EEVEE_HARDWARE_REFRACTIONS;
  use_hardware_specular_scene_ = (current_hardware_feature_mask_ & specular_feature_mask) != 0 &&
                                 hardware_scene_entry_count_ > 0;
  use_hardware_hybrid_retrace_ =
      use_hardware_specular_scene_ &&
      (ELEM(hardware_reflection_mode_,
            RAYTRACE_EEVEE_SPECULAR_MODE_HYBRID,
            RAYTRACE_EEVEE_SPECULAR_MODE_AUTO) ||
       ELEM(hardware_refraction_mode_,
            RAYTRACE_EEVEE_SPECULAR_MODE_HYBRID,
            RAYTRACE_EEVEE_SPECULAR_MODE_AUTO));
  if (std::getenv("BLENDER_EEVEE_HWRT_CACHE_LOG") != nullptr) {
    std::fprintf(stderr,
                 "EEVEE HWRT trace closure=%d features=0x%x entries=%d instances=%d\n",
                 data_.closure_index,
                 unsigned(current_hardware_feature_mask_),
                 hardware_scene_entry_count_,
                 hardware_scene_instance_count_);
  }

  GPU_debug_group_begin("Hardware RT");

  auto submit_screen_baseline = [&]() {
    if (inst_.planar_probes.enabled()) {
      inst_.manager->submit(trace_planar_ps_, render_view);
    }
    inst_.manager->submit(trace_screen_ps_, render_view);
  };

  const bool use_hardware_closure_override = current_hardware_feature_mask_ != 0;
  const bool has_classic_specular_fallback =
      (current_trace_active_closures_ & (CLOSURE_REFLECTION | CLOSURE_REFRACTION)) != 0;
  ray_time_tx_.clear(float4(0.0f));
  ray_radiance_tx_.clear(float4(0.0f));

  if (!use_hardware_closure_override || hardware_scene_entry_count_ == 0) {
    if (has_classic_specular_fallback) {
      submit_screen_baseline();
    }
    GPU_debug_group_end();
    return;
  }

  submit_screen_baseline();

  hardware_trace_dispatch_buf_.clear_to_zero();
  inst_.manager->submit(hardware_trace_tile_compact_ps_);

  hit_albedo_tx_.clear(float4(0.0f));
  hit_throughput_tx_.clear(float4(0.0f));
  hit_material_tx_.clear(float4(0.0f));
  hit_normal_tx_.clear(float4(0.0f));
  hit_position_tx_.clear(float4(0.0f));
  hit_world_position_tx_.clear(float4(0.0f));
  hit_identity_tx_.clear(uint4(0u));
  hit_barycentric_tx_.clear(float4(0.0f));
  layered_receiver_ray_time_tx_.clear(float4(0.0f));
  layered_receiver_ray_radiance_tx_.clear(float4(0.0f));
  layered_receiver_albedo_tx_.clear(float4(0.0f));
  layered_receiver_throughput_tx_.clear(float4(0.0f));
  layered_receiver_material_tx_.clear(float4(0.0f));
  layered_receiver_normal_tx_.clear(float4(0.0f));
  layered_receiver_position_tx_.clear(float4(0.0f));
  layered_receiver_world_position_tx_.clear(float4(0.0f));
  layered_receiver_identity_tx_.clear(uint4(0u));
  layered_receiver_barycentric_tx_.clear(float4(0.0f));
  hardware_reflected_receiver_gi_tx_.clear(float4(0.0f));
  hardware_reflected_receiver_gi_blur_tx_.clear(float4(0.0f));
  transmission_receiver_ray_time_tx_.clear(float4(0.0f));
  transmission_receiver_ray_radiance_tx_.clear(float4(0.0f));
  transmission_receiver_albedo_tx_.clear(float4(0.0f));
  transmission_receiver_throughput_tx_.clear(float4(0.0f));
  transmission_receiver_material_tx_.clear(float4(0.0f));
  transmission_receiver_normal_tx_.clear(float4(0.0f));
  transmission_receiver_position_tx_.clear(float4(0.0f));
  transmission_receiver_world_position_tx_.clear(float4(0.0f));
  transmission_receiver_identity_tx_.clear(uint4(0u));
  transmission_receiver_barycentric_tx_.clear(float4(0.0f));
  GPU_flush();

  const double scene_acquire_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  GPUMetalRaytraceSceneStats metal_scene_stats;
  GPUMetalRaytraceScene *metal_scene = acquire_hardware_metal_scene(&metal_scene_stats);
  const double scene_acquire_ms = perf_logging_enabled ?
                                      (BLI_time_now_seconds() - scene_acquire_start) * 1000.0 :
                                      0.0;

  /* Hardware RT owns this closure. Only explicit Hardware hits or Hardware miss-resolve passes
   * should contribute to the result. */
  if (metal_scene != nullptr && metal_scene_stats.built_scene) {
    GPU_debug_group_begin("Metal Scene");
    hardware_resolve_dispatch_buf_.clear_to_zero();
    const int2 tracing_res = math::divide_ceil(data_.full_resolution, int2(data_.resolution_scale));
    GPUMetalRaytraceTraceParams trace_params;
    trace_params.ray_data_tx = ray_data_tx_;
    trace_params.depth_tx = renderbuf_depth_view_;
    trace_params.gbuf_header_tx = inst_.gbuffer.header_tx;
    trace_params.gbuf_normal_tx = inst_.gbuffer.normal_tx;
    trace_params.screen_continuation_tx = screen_continuation_tx_;
    trace_params.world_probe_tx = inst_.sphere_probes.octahedral_probes_texture();
    trace_params.ray_time_tx = ray_time_tx_;
    trace_params.ray_radiance_tx = ray_radiance_tx_;
    trace_params.hit_albedo_tx = hit_albedo_tx_;
    trace_params.hit_throughput_tx = hit_throughput_tx_;
    trace_params.hit_material_tx = hit_material_tx_;
    trace_params.hit_normal_tx = hit_normal_tx_;
    trace_params.hit_position_tx = hit_position_tx_;
    trace_params.hit_world_position_tx = hit_world_position_tx_;
    trace_params.hit_identity_tx = hit_identity_tx_;
    trace_params.hit_barycentric_tx = hit_barycentric_tx_;
    trace_params.layered_receiver_ray_time_tx = layered_receiver_ray_time_tx_;
    trace_params.layered_receiver_ray_radiance_tx = layered_receiver_ray_radiance_tx_;
    trace_params.layered_receiver_albedo_tx = layered_receiver_albedo_tx_;
    trace_params.layered_receiver_throughput_tx = layered_receiver_throughput_tx_;
    trace_params.layered_receiver_material_tx = layered_receiver_material_tx_;
    trace_params.layered_receiver_normal_tx = layered_receiver_normal_tx_;
    trace_params.layered_receiver_position_tx = layered_receiver_position_tx_;
    trace_params.layered_receiver_world_position_tx = layered_receiver_world_position_tx_;
    trace_params.layered_receiver_identity_tx = layered_receiver_identity_tx_;
    trace_params.layered_receiver_barycentric_tx = layered_receiver_barycentric_tx_;
    trace_params.transmission_receiver_ray_time_tx = transmission_receiver_ray_time_tx_;
    trace_params.transmission_receiver_ray_radiance_tx = transmission_receiver_ray_radiance_tx_;
    trace_params.transmission_receiver_albedo_tx = transmission_receiver_albedo_tx_;
    trace_params.transmission_receiver_throughput_tx = transmission_receiver_throughput_tx_;
    trace_params.transmission_receiver_material_tx = transmission_receiver_material_tx_;
    trace_params.transmission_receiver_normal_tx = transmission_receiver_normal_tx_;
    trace_params.transmission_receiver_position_tx = transmission_receiver_position_tx_;
    trace_params.transmission_receiver_world_position_tx = transmission_receiver_world_position_tx_;
    trace_params.transmission_receiver_identity_tx = transmission_receiver_identity_tx_;
    trace_params.transmission_receiver_barycentric_tx = transmission_receiver_barycentric_tx_;
    trace_params.viewinv = render_view.viewinv();
    trace_params.wininv = render_view.wininv();
    trace_params.full_resolution = data_.full_resolution;
    trace_params.resolution_scale = data_.resolution_scale;
    trace_params.closure_index = data_.closure_index;
    trace_params.feature_mask = current_hardware_feature_mask_;
    trace_params.hardware_trace_phase = data_.hardware_trace_phase;
    trace_params.reflection_bounces = data_.hardware_reflection_bounces;
    trace_params.refraction_bounces = data_.hardware_refraction_bounces;
    trace_params.resolution_bias = data_.resolution_bias;
    trace_params.clamp_indirect = 1.0e10f;
    const SphereProbe &world_probe = inst_.sphere_probes.world_sphere_probe();
    const bool world_probe_available = world_probe.atlas_coord.atlas_layer >= 0 &&
                                       world_probe.atlas_coord.subdivision_lvl >= 0;
    const SphereProbeUvArea world_probe_atlas_coord = world_probe_available ?
                                                          world_probe.atlas_coord.
                                                              as_sampling_coord() :
                                                          SphereProbeUvArea{
                                                              float2(0.0f), 0.0f, -1.0f};
    trace_params.world_probe_atlas_coord = float4(world_probe_atlas_coord.offset.x,
                                                  world_probe_atlas_coord.offset.y,
                                                  world_probe_atlas_coord.scale,
                                                  world_probe_atlas_coord.layer);
    trace_params.use_environment = use_hardware_gi() || use_hardware_environment();
    const float3 raytrace_rng = inst_.sampling.rng_3d_get(eSamplingDimension::SAMPLING_RAYTRACE_U);
    trace_params.sampling_rand = float4(
        raytrace_rng.x,
        raytrace_rng.y,
        raytrace_rng.z,
        inst_.sampling.rng_get(eSamplingDimension::SAMPLING_CLOSURE));
    trace_params.dispatch_buf = hardware_trace_dispatch_buf_;
    trace_params.tiles_coord_buf = hardware_trace_tiles_buf_;
    const double trace_submit_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
    GPU_metal_raytrace_scene_trace(metal_scene, trace_params);
    const double trace_submit_ms = perf_logging_enabled ?
                                       (BLI_time_now_seconds() - trace_submit_start) * 1000.0 :
                                       0.0;
    inst_.manager->submit(hardware_tile_compact_ps_);
    submit_hardware_hit_evaluation_backend(render_view);
    const double secondary_environment_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
    render_secondary_environment_visibility(metal_scene, tracing_res);
    const double secondary_environment_ms =
        perf_logging_enabled ?
            (BLI_time_now_seconds() - secondary_environment_start) * 1000.0 :
            0.0;
    const double secondary_shadow_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
    render_secondary_shadow_visibility(
        hardware_indirect_gi_cache_rendering_ ? nullptr : metal_scene, tracing_res);
    const double secondary_shadow_ms = perf_logging_enabled ?
                                           (BLI_time_now_seconds() - secondary_shadow_start) * 1000.0 :
                                           0.0;
    render_reflected_receiver_gi(metal_scene, tracing_res);
    inst_.manager->submit(trace_hardware_lighting_ps_, render_view);
    if (perf_logging_enabled) {
      const double elapsed_ms = (BLI_time_now_seconds() - perf_start_time) * 1000.0;
      std::fprintf(stderr,
                   "EEVEE HWRT perf trace closure=%d features=0x%x scene_acquire_ms=%.2f trace_submit_ms=%.2f secondary_env_ms=%.2f secondary_shadow_ms=%.2f elapsed_ms=%.2f\n",
                   data_.closure_index,
                   unsigned(current_hardware_feature_mask_),
                   scene_acquire_ms,
                   trace_submit_ms,
                   secondary_environment_ms,
                   secondary_shadow_ms,
                   elapsed_ms);
    }
    GPU_debug_group_end();
  }

  GPU_debug_group_end();
}

bool RayTraceModule::submit_hardware_hit_evaluation_backend(View &render_view)
{
  const uint32_t replay_feature_mask = RAYTRACE_EEVEE_HARDWARE_GI |
                                       RAYTRACE_EEVEE_HARDWARE_REFLECTIONS |
                                       RAYTRACE_EEVEE_HARDWARE_REFRACTIONS;
  if ((current_hardware_feature_mask_ & replay_feature_mask) == 0) {
    return false;
  }

  const Span<HardwareRaytraceSceneEntry> all_entries = hardware_metal_scene_entries_cache_;
  if (all_entries.is_empty()) {
    return false;
  }

  Texture &depth_tx = inst_.render_buffers.depth_tx;
  GPU_debug_group_begin("Hardware RT Hit Eval");
  const int2 tracing_extent = ray_data_tx_.size().xy();
  const int entry_count = all_entries.size();
  const int max_hit_records = max_ii(1, tracing_extent.x * tracing_extent.y);

  struct HitEvalPayload {
    gpu::Texture *ray_time_tx;
    gpu::Texture *ray_radiance_tx;
    gpu::Texture *hit_albedo_tx;
    gpu::Texture *hit_throughput_tx;
    gpu::Texture *hit_material_tx;
    gpu::Texture *hit_normal_tx;
    gpu::Texture *hit_position_tx;
    gpu::Texture *hit_world_position_tx;
    gpu::Texture *hit_identity_tx;
    gpu::Texture *hit_barycentric_tx;
  };

  auto submit_payload_hit_eval = [&](const HitEvalPayload &payload) -> bool {
    hit_eval_count_buf_.clear_to_zero();
    hit_eval_offset_buf_.clear_to_zero();
    hit_eval_cursor_buf_.clear_to_zero();
    hit_eval_indirect_buf_.clear_to_zero();

    hit_eval_count_ps_.init();
    hit_eval_count_ps_.shader_set(inst_.shaders.static_shader_get(RAY_HIT_EVAL_COUNT));
    hit_eval_count_ps_.push_constant("scene_entry_count", entry_count);
    hit_eval_count_ps_.bind_image("ray_time_img", payload.ray_time_tx);
    hit_eval_count_ps_.bind_image("hit_identity_img", payload.hit_identity_tx);
    hit_eval_count_ps_.bind_texture("depth_tx", &depth_tx);
    hit_eval_count_ps_.bind_ssbo("hit_eval_count_buf", &hit_eval_count_buf_);
    hit_eval_count_ps_.bind_ssbo("tiles_coord_buf", &hardware_resolve_tiles_buf_);
    hit_eval_count_ps_.bind_resources(inst_.uniform_data);
    hit_eval_count_ps_.dispatch(hardware_resolve_dispatch_buf_);
    hit_eval_count_ps_.barrier(GPU_BARRIER_SHADER_STORAGE);
    inst_.manager->submit(hit_eval_count_ps_);

    hit_eval_prefix_ps_.init();
    hit_eval_prefix_ps_.shader_set(inst_.shaders.static_shader_get(RAY_HIT_EVAL_PREFIX));
    hit_eval_prefix_ps_.push_constant("scene_entry_count", entry_count);
    hit_eval_prefix_ps_.bind_ssbo("hit_eval_count_buf", &hit_eval_count_buf_);
    hit_eval_prefix_ps_.bind_ssbo("hit_eval_offset_buf", &hit_eval_offset_buf_);
    hit_eval_prefix_ps_.bind_ssbo("hit_eval_cursor_buf", &hit_eval_cursor_buf_);
    hit_eval_prefix_ps_.bind_ssbo("hit_eval_indirect_draw_buf", &hit_eval_indirect_buf_);
    hit_eval_prefix_ps_.dispatch(int3((entry_count + 63) / 64, 1, 1));
    hit_eval_prefix_ps_.barrier(GPU_BARRIER_SHADER_STORAGE | GPU_BARRIER_COMMAND);
    inst_.manager->submit(hit_eval_prefix_ps_);

    hit_eval_compact_ps_.init();
    hit_eval_compact_ps_.shader_set(inst_.shaders.static_shader_get(RAY_HIT_EVAL_COMPACT));
    hit_eval_compact_ps_.push_constant("scene_entry_count", entry_count);
    hit_eval_compact_ps_.bind_image("ray_time_img", payload.ray_time_tx);
    hit_eval_compact_ps_.bind_image("hit_identity_img", payload.hit_identity_tx);
    hit_eval_compact_ps_.bind_image("hit_material_img", payload.hit_material_tx);
    hit_eval_compact_ps_.bind_image("hit_normal_img", payload.hit_normal_tx);
    hit_eval_compact_ps_.bind_image("hit_barycentric_img", payload.hit_barycentric_tx);
    hit_eval_compact_ps_.bind_texture("depth_tx", &depth_tx);
    hit_eval_compact_ps_.bind_texture("hit_world_position_tx", payload.hit_world_position_tx);
    hit_eval_compact_ps_.bind_ssbo("hit_eval_offset_buf", &hit_eval_offset_buf_);
    hit_eval_compact_ps_.bind_ssbo("hit_eval_cursor_buf", &hit_eval_cursor_buf_);
    hit_eval_compact_ps_.bind_ssbo("hit_eval_resource_id_buf", &hit_eval_resource_id_buf_);
    hit_eval_compact_ps_.bind_ssbo("hit_eval_list_buf", &hit_eval_records_buf_);
    hit_eval_compact_ps_.bind_ssbo("tiles_coord_buf", &hardware_resolve_tiles_buf_);
    hit_eval_compact_ps_.bind_resources(inst_.uniform_data);
    hit_eval_compact_ps_.dispatch(hardware_resolve_dispatch_buf_);
    hit_eval_compact_ps_.barrier(GPU_BARRIER_SHADER_STORAGE);
    inst_.manager->submit(hit_eval_compact_ps_);
    GPU_storagebuf_sync_as_indirect_buffer(hit_eval_indirect_buf_);

    hit_eval_ps_.init();
    hit_eval_ps_.state_set(DRW_STATE_WRITE_COLOR | DRW_STATE_DEPTH_ALWAYS);
    hit_eval_ps_.framebuffer_set(&hit_eval_fb_);

    auto bind_hit_eval_resources = [&](draw::PassSimple &pass) {
      pass.bind_texture(RBUFS_UTILITY_TEX_SLOT, inst_.pipelines.utility_tx);
      pass.bind_texture("ray_data_tx", &ray_data_tx_);
      pass.bind_texture("ray_time_tx", payload.ray_time_tx);
      pass.bind_texture("hit_identity_tx", payload.hit_identity_tx);
      pass.bind_texture("hit_barycentric_tx", payload.hit_barycentric_tx);
      pass.bind_image("hit_albedo_img", payload.hit_albedo_tx);
      pass.bind_image("hit_throughput_img", payload.hit_throughput_tx);
      pass.bind_image("hit_material_img", payload.hit_material_tx);
      pass.bind_image("hit_normal_img", payload.hit_normal_tx);
      pass.bind_image("hit_position_img", payload.hit_position_tx);
      pass.bind_texture("hit_world_position_tx", payload.hit_world_position_tx);
      pass.bind_image("ray_radiance_img", payload.ray_radiance_tx);
      pass.bind_ssbo("hit_eval_list_buf", &hit_eval_records_buf_);
      pass.bind_resources(inst_.uniform_data);
      pass.bind_resources(inst_.sampling);
    };

    hit_eval_fb_.ensure(GPU_ATTACHMENT_TEXTURE(renderbuf_depth_view_));
    hit_eval_fb_.bind();
    GPU_framebuffer_viewport_set(hit_eval_fb_, 0, 0, UNPACK2(tracing_extent));

    bool submitted_any = false;
    for (const int entry_index : all_entries.index_range()) {
      const HardwareRaytraceSceneEntry &entry = all_entries[entry_index];
      if (entry.batch == nullptr || entry.hit_eval_object == nullptr ||
          entry.hit_eval_object->id.name[2] == '\0')
      {
        /* Converted legacy wrappers stay on the bounded proxy payload instead of sparse replay. */
        continue;
      }
      if (entry.hit_eval_object->type != OB_MESH || entry.hit_eval_object->data == nullptr ||
          GS(static_cast<ID *>(entry.hit_eval_object->data)->name) != ID_ME)
      {
        continue;
      }
      if (!DEG_is_evaluated(entry.hit_eval_object)) {
        /* Sparse hit-eval replay expects the evaluated object/material state that draw sync
         * compiled against. Falling back to the proxy payload is safer than tripping eval-only
         * material queries on original objects in viewport paths. */
        continue;
      }

      Material &material = inst_.materials.material_get(
          entry.hit_eval_object, false, entry.material_slot, MAT_GEOM_MESH);
      GPUMaterial *gpumat = material.hit_eval.gpumat;
      if ((gpumat == nullptr) || (GPU_material_status(gpumat) != GPU_MAT_SUCCESS)) {
        /* Fail closed: keep the proxy payload from the Hardware trace and let the later lighting
         * resolve use that simplification instead of attempting sparse material replay. */
        continue;
      }
      gpu::Shader *shader = GPU_material_get_shader(gpumat);
      if (!hardware_hit_eval_batch_compatible(entry.batch, shader)) {
        /* Production scenes can contain batches whose material replay requests vertex attributes
         * the cached draw batch does not expose as SSBO inputs. Fail closed to the proxy payload
         * rather than asserting in GPU_batch_bind_as_resources(). */
        continue;
      }

      hit_eval_ps_.material_set(*inst_.manager, gpumat, true, inst_.anisotropic_filtering);
      bind_hit_eval_resources(hit_eval_ps_);
      hit_eval_ps_.draw_expand_indirect(entry.batch,
                                        GPU_PRIM_TRIS,
                                        1,
                                        &hit_eval_indirect_buf_,
                                        uint32_t(sizeof(DrawCommand) * entry_index),
                                        ResourceIDRange(entry.resource_handle).first);
      submitted_any = true;
    }

    if (!submitted_any || hit_eval_ps_.is_empty()) {
      return false;
    }

    hit_eval_ps_.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS | GPU_BARRIER_SHADER_STORAGE);
    inst_.manager->submit(hit_eval_ps_, render_view);
    return true;
  };

  hit_eval_count_buf_.resize(entry_count);
  hit_eval_offset_buf_.resize(entry_count);
  hit_eval_cursor_buf_.resize(entry_count);
  hit_eval_resource_id_buf_.resize(entry_count);
  hit_eval_indirect_buf_.resize(entry_count);
  hit_eval_records_buf_.resize(max_hit_records);

  for (const int entry_index : all_entries.index_range()) {
    const HardwareRaytraceSceneEntry &entry = all_entries[entry_index];
    const ResourceIDRange resource_range = entry.resource_handle;
    hit_eval_resource_id_buf_.get_or_resize(entry_index) = uint(resource_range.first.raw);
  }
  hit_eval_resource_id_buf_.push_update();
  const bool submitted_primary = submit_payload_hit_eval({ray_time_tx_,
                                                          ray_radiance_tx_,
                                                          hit_albedo_tx_,
                                                          hit_throughput_tx_,
                                                          hit_material_tx_,
                                                          hit_normal_tx_,
                                                          hit_position_tx_,
                                                          hit_world_position_tx_,
                                                          hit_identity_tx_,
                                                          hit_barycentric_tx_});
  const bool submitted_receiver = submit_payload_hit_eval({layered_receiver_ray_time_tx_,
                                                           layered_receiver_ray_radiance_tx_,
                                                           layered_receiver_albedo_tx_,
                                                           layered_receiver_throughput_tx_,
                                                           layered_receiver_material_tx_,
                                                           layered_receiver_normal_tx_,
                                                           layered_receiver_position_tx_,
                                                           layered_receiver_world_position_tx_,
                                                           layered_receiver_identity_tx_,
                                                           layered_receiver_barycentric_tx_});
  const bool submitted_transmission_receiver = submit_payload_hit_eval(
      {transmission_receiver_ray_time_tx_,
       transmission_receiver_ray_radiance_tx_,
       transmission_receiver_albedo_tx_,
       transmission_receiver_throughput_tx_,
       transmission_receiver_material_tx_,
       transmission_receiver_normal_tx_,
       transmission_receiver_position_tx_,
       transmission_receiver_world_position_tx_,
       transmission_receiver_identity_tx_,
       transmission_receiver_barycentric_tx_});
  GPU_debug_group_end();
  return submitted_primary || submitted_receiver || submitted_transmission_receiver;
}

void RayTraceModule::sync()
{
  Texture &depth_tx = inst_.render_buffers.depth_tx;
  constexpr GPUSamplerState fast_gi_field_sampler = {GPU_SAMPLER_FILTERING_LINEAR};
  viewport_history_reset_ = inst_.is_viewport() && inst_.sampling.is_reset();
  invalidate_sorted_hardware_scene_entries_cache();
  invalidate_viewport_hardware_visibility_cache();

  if (!use_raytracing_) {
    /* Do not request raytracing shaders if not needed. */
    return;
  }

#define PASS_VARIATION(_pass_name, _index, _suffix) \
  ((_index == 0) ? _pass_name##reflect##_suffix : \
   (_index == 1) ? _pass_name##refract##_suffix : \
                   _pass_name##diffuse##_suffix)

  /* Setup. */
  {
    PassSimple &pass = tile_classify_ps_;
    pass.init();
    pass.shader_set(inst_.shaders.static_shader_get(RAY_TILE_CLASSIFY));
    pass.bind_image("tile_raytrace_denoise_img", &tile_raytrace_denoise_tx_);
    pass.bind_image("tile_raytrace_tracing_img", &tile_raytrace_tracing_tx_);
    pass.bind_image("tile_horizon_denoise_img", &tile_horizon_denoise_tx_);
    pass.bind_image("tile_horizon_tracing_img", &tile_horizon_tracing_tx_);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.gbuffer);
    pass.dispatch(&tile_classify_dispatch_size_);
    pass.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS | GPU_BARRIER_SHADER_STORAGE);
  }
  {
    PassSimple &pass = tile_compact_ps_;
    gpu::Shader *sh = inst_.shaders.static_shader_get(RAY_TILE_COMPACT);
    pass.init();
    pass.specialize_constant(sh, "closure_index", &data_.closure_index);
    pass.specialize_constant(sh, "resolution_scale", &data_.resolution_scale);
    pass.shader_set(sh);
    pass.bind_image("tile_raytrace_denoise_img", &tile_raytrace_denoise_tx_);
    pass.bind_image("tile_raytrace_tracing_img", &tile_raytrace_tracing_tx_);
    pass.bind_ssbo("raytrace_tracing_dispatch_buf", &raytrace_tracing_dispatch_buf_);
    pass.bind_ssbo("raytrace_denoise_dispatch_buf", &raytrace_denoise_dispatch_buf_);
    pass.bind_ssbo("raytrace_tracing_tiles_buf", &raytrace_tracing_tiles_buf_);
    pass.bind_ssbo("raytrace_denoise_tiles_buf", &raytrace_denoise_tiles_buf_);
    pass.bind_resources(inst_.uniform_data);
    pass.dispatch(&tile_compact_dispatch_size_);
    pass.barrier(GPU_BARRIER_SHADER_STORAGE);
  }
  {
    PassSimple &pass = hardware_direct_light_tile_compact_ps_;
    pass.init();
    pass.shader_set(inst_.shaders.static_shader_get(RAY_HARDWARE_DIRECT_LIGHT_TILE_COMPACT));
    pass.bind_ssbo("hardware_direct_light_dispatch_buf", &hardware_direct_light_dispatch_buf_);
    pass.bind_ssbo("hardware_direct_light_work_tiles_buf", &hardware_direct_light_work_tiles_buf_);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.lights);
    pass.dispatch(&hardware_direct_light_tile_compact_dispatch_size_);
    pass.barrier(GPU_BARRIER_SHADER_STORAGE);
  }
  {
    PassSimple &pass = hardware_direct_light_visibility_ps_;
    pass.init();
    pass.shader_set(inst_.shaders.static_shader_get(RAY_HARDWARE_DIRECT_LIGHT_VISIBILITY));
    pass.bind_texture("hardware_rt_shadow_visibility_tx", &hardware_shadow_visibility_tx_);
    pass.bind_ssbo("hardware_direct_light_work_tiles_buf", &hardware_direct_light_work_tiles_buf_);
    pass.bind_ssbo("hardware_direct_light_visibility_samples_buf",
                   &hardware_direct_light_visibility_samples_buf_);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.lights);
    pass.bind_resources(inst_.sampling);
    pass.dispatch(hardware_direct_light_dispatch_buf_);
    pass.barrier(GPU_BARRIER_SHADER_STORAGE | GPU_BARRIER_TEXTURE_FETCH);
  }
  {
    PassSimple &pass = hardware_direct_light_accum_ps_;
    pass.init();
    pass.shader_set(inst_.shaders.static_shader_get(RAY_HARDWARE_DIRECT_LIGHT_ACCUM));
    pass.bind_image("out_direct_light_accum_img", &hardware_direct_light_accum_tx_);
    pass.bind_texture("depth_tx", &renderbuf_depth_view_);
    pass.bind_ssbo("hardware_direct_light_work_tiles_buf", &hardware_direct_light_work_tiles_buf_);
    pass.bind_ssbo("hardware_direct_light_visibility_samples_buf",
                   &hardware_direct_light_visibility_samples_buf_);
    pass.bind_texture(RBUFS_UTILITY_TEX_SLOT, inst_.pipelines.utility_tx);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.sampling);
    pass.bind_resources(inst_.lights);
    pass.bind_resources(inst_.gbuffer);
    pass.dispatch(hardware_direct_light_dispatch_buf_);
    pass.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS);
  }
  {
    PassSimple &pass = hardware_direct_light_denoise_ps_;
    pass.init();
    pass.shader_set(inst_.shaders.static_shader_get(RAY_HARDWARE_DIRECT_LIGHT_DENOISE));
    pass.bind_texture("depth_tx", &renderbuf_depth_view_);
    pass.bind_texture("hardware_rt_shadow_visibility_tx", &hardware_shadow_visibility_tx_);
    pass.bind_image("in_direct_light_accum_img", &hardware_direct_light_accum_tx_);
    pass.bind_image("out_direct_light_denoised_img", &hardware_direct_light_denoised_tx_);
    pass.bind_image("out_direct_light_depth_img", &hardware_direct_light_depth_tx_);
    pass.bind_image("direct_light_tilemask_img", &hardware_direct_light_tilemask_tx_);
    pass.bind_ssbo("hardware_direct_light_work_tiles_buf", &hardware_direct_light_work_tiles_buf_);
    pass.bind_ssbo("hardware_direct_light_visibility_samples_buf",
                   &hardware_direct_light_visibility_samples_buf_);
    pass.bind_texture(RBUFS_UTILITY_TEX_SLOT, inst_.pipelines.utility_tx);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.lights);
    inst_.lights.bind_no_cull_light_resources(pass);
    pass.bind_resources(inst_.gbuffer);
    pass.dispatch(hardware_direct_light_dispatch_buf_);
    pass.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS | GPU_BARRIER_TEXTURE_FETCH);
  }
  {
    PassSimple &pass = hardware_indirect_gi_cache_store_ps_;
    pass.init();
    pass.shader_set(inst_.shaders.static_shader_get(RAY_HARDWARE_INDIRECT_GI_CACHE_STORE));
    pass.bind_texture("depth_tx", &renderbuf_depth_view_);
    pass.bind_texture("combined_tx", &inst_.render_buffers.combined_tx);
    pass.bind_image("out_indirect_gi_radiance_cache_img",
                    &hardware_indirect_gi_radiance_cache_tx_);
    pass.bind_image("out_indirect_gi_position_cache_img",
                    &hardware_indirect_gi_position_cache_tx_);
    pass.bind_image("out_indirect_gi_normal_cache_img", &hardware_indirect_gi_normal_cache_tx_);
    pass.push_constant("cache_face_index", &hardware_indirect_gi_cache_face_index_);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.gbuffer);
    pass.dispatch(&hardware_indirect_gi_cache_dispatch_size_);
    pass.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS | GPU_BARRIER_TEXTURE_FETCH);
  }
  {
    PassSimple &pass = generate_ps_;
    pass.init();
    gpu::Shader *sh = inst_.shaders.static_shader_get(RAY_GENERATE);
    pass.specialize_constant(sh, "closure_index", &data_.closure_index);
    pass.shader_set(sh);
    pass.bind_texture(RBUFS_UTILITY_TEX_SLOT, inst_.pipelines.utility_tx);
    pass.bind_image("out_ray_data_img", &ray_data_tx_);
    pass.bind_ssbo("tiles_coord_buf", &raytrace_tracing_tiles_buf_);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.sampling);
    pass.bind_resources(inst_.gbuffer);
    pass.dispatch(raytrace_tracing_dispatch_buf_);
    pass.barrier(GPU_BARRIER_SHADER_STORAGE | GPU_BARRIER_TEXTURE_FETCH |
                 GPU_BARRIER_SHADER_IMAGE_ACCESS);
  }
  /* Tracing. */
  {
    PassSimple &pass = trace_planar_ps_;
    pass.init();
    gpu::Shader *sh = inst_.shaders.static_shader_get(RAY_TRACE_PLANAR);
    pass.specialize_constant(
        sh, "use_hardware_specular_scene", reinterpret_cast<bool *>(&use_hardware_specular_scene_));
    pass.specialize_constant(sh, "closure_index", &data_.closure_index);
    pass.shader_set(sh);
    pass.bind_ssbo("tiles_coord_buf", &raytrace_tracing_tiles_buf_);
    pass.bind_image("ray_data_img", &ray_data_tx_);
    pass.bind_image("ray_time_img", &ray_time_tx_);
    pass.bind_image("ray_radiance_img", &ray_radiance_tx_);
    pass.bind_texture("depth_tx", &depth_tx);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.sampling);
    pass.bind_resources(inst_.planar_probes);
    pass.bind_resources(inst_.volume_probes);
    pass.bind_resources(inst_.sphere_probes);
    pass.bind_resources(inst_.gbuffer);
    /* TODO(@fclem): Use another dispatch with only tiles that touches planar captures. */
    pass.dispatch(raytrace_tracing_dispatch_buf_);
    pass.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS | GPU_BARRIER_TEXTURE_FETCH);
  }
  {
    PassSimple &pass = trace_screen_ps_;
    pass.init();
    gpu::Shader *sh = inst_.shaders.static_shader_get(RAY_TRACE_SCREEN);
    pass.specialize_constant(
        sh, "trace_refraction", reinterpret_cast<bool *>(&data_.trace_refraction));
    pass.specialize_constant(sh,
                             "use_hardware_rt_environment_visibility",
                             reinterpret_cast<bool *>(&hardware_environment_enabled_));
    pass.specialize_constant(
        sh, "use_hardware_specular_scene", reinterpret_cast<bool *>(&use_hardware_specular_scene_));
    pass.specialize_constant(
        sh, "use_hardware_hybrid_retrace", reinterpret_cast<bool *>(&use_hardware_hybrid_retrace_));
    pass.specialize_constant(
        sh, "use_screen_ownership_history", reinterpret_cast<bool *>(&use_screen_ownership_history_));
    pass.specialize_constant(sh, "closure_index", &data_.closure_index);
    pass.shader_set(sh);
    pass.bind_ssbo("tiles_coord_buf", &raytrace_tracing_tiles_buf_);
    pass.bind_image("ray_data_img", &ray_data_tx_);
    pass.bind_image("ray_time_img", &ray_time_tx_);
    pass.bind_image("screen_continuation_img", &screen_continuation_tx_);
    pass.bind_image("screen_ownership_img", &screen_ownership_tx_);
    pass.bind_texture("radiance_front_tx", &screen_radiance_front_tx_);
    pass.bind_texture("radiance_back_tx", &screen_radiance_back_tx_);
    pass.bind_texture("ownership_history_tx", &screen_ownership_history_tx_);
    pass.bind_texture("hiz_front_tx", &inst_.hiz_buffer.front.ref_tx_);
    pass.bind_texture("hiz_back_tx", &inst_.hiz_buffer.back.ref_tx_);
    /* Still bind front to hiz_tx for validation layers. */
    pass.bind_resources(inst_.hiz_buffer.front);
    pass.bind_texture("depth_tx", &depth_tx);
    pass.bind_texture("hardware_rt_environment_visibility_tx", &hardware_environment_visibility_tx_);
    pass.bind_texture("hardware_fast_gi_tx", &hardware_fast_gi_tx_, fast_gi_field_sampler);
    pass.bind_texture(
        "hardware_fast_gi_visibility_tx", &hardware_fast_gi_visibility_tx_, fast_gi_field_sampler);
    pass.bind_image("ray_radiance_img", &ray_radiance_tx_);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.sampling);
    pass.bind_resources(inst_.volume_probes);
    pass.bind_resources(inst_.sphere_probes);
    pass.bind_resources(inst_.gbuffer);
    pass.dispatch(raytrace_tracing_dispatch_buf_);
    pass.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS);
  }
  {
    PassSimple &pass = trace_fallback_ps_;
    pass.init();
    gpu::Shader *sh = inst_.shaders.static_shader_get(RAY_TRACE_FALLBACK);
    pass.specialize_constant(sh,
                             "use_hardware_rt_environment_visibility",
                             reinterpret_cast<bool *>(&hardware_environment_enabled_));
    pass.specialize_constant(sh, "closure_index", &data_.closure_index);
    pass.shader_set(sh);
    pass.bind_ssbo("tiles_coord_buf", &raytrace_tracing_tiles_buf_);
    pass.bind_image("ray_data_img", &ray_data_tx_);
    pass.bind_image("ray_time_img", &ray_time_tx_);
    pass.bind_image("ray_radiance_img", &ray_radiance_tx_);
    pass.bind_texture("depth_tx", &depth_tx);
    pass.bind_texture("hardware_rt_environment_visibility_tx", &hardware_environment_visibility_tx_);
    pass.bind_texture("hardware_fast_gi_tx", &hardware_fast_gi_tx_, fast_gi_field_sampler);
    pass.bind_texture(
        "hardware_fast_gi_visibility_tx", &hardware_fast_gi_visibility_tx_, fast_gi_field_sampler);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.volume_probes);
    pass.bind_resources(inst_.sphere_probes);
    pass.bind_resources(inst_.sampling);
    pass.bind_resources(inst_.gbuffer);
    pass.dispatch(raytrace_tracing_dispatch_buf_);
    pass.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS);
  }
  {
    PassSimple &pass = hardware_trace_tile_compact_ps_;
    pass.init();
    pass.shader_set(inst_.shaders.static_shader_get(RAY_HARDWARE_TRACE_TILE_COMPACT));
    pass.bind_image("ray_data_img", &ray_data_tx_);
    pass.bind_image("ray_time_img", &ray_time_tx_);
    pass.bind_ssbo("hardware_trace_dispatch_buf", &hardware_trace_dispatch_buf_);
    pass.bind_ssbo("hardware_trace_tiles_buf", &hardware_trace_tiles_buf_);
    pass.bind_ssbo("tiles_coord_buf", &raytrace_tracing_tiles_buf_);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.gbuffer);
    pass.dispatch(raytrace_tracing_dispatch_buf_);
    pass.barrier(GPU_BARRIER_SHADER_STORAGE);
  }
  {
    PassSimple &pass = hardware_tile_compact_ps_;
    pass.init();
    pass.shader_set(inst_.shaders.static_shader_get(RAY_HARDWARE_TILE_COMPACT));
    pass.bind_image("ray_time_img", &ray_time_tx_);
    pass.bind_image("hit_normal_img", &hit_normal_tx_);
    pass.bind_ssbo("hardware_resolve_dispatch_buf", &hardware_resolve_dispatch_buf_);
    pass.bind_ssbo("hardware_resolve_tiles_buf", &hardware_resolve_tiles_buf_);
    pass.bind_ssbo("tiles_coord_buf", &hardware_trace_tiles_buf_);
    pass.dispatch(hardware_trace_dispatch_buf_);
    pass.barrier(GPU_BARRIER_SHADER_STORAGE);
  }
  {
    PassSimple &pass = trace_hardware_lighting_ps_;
    pass.init();
    gpu::Shader *sh = inst_.shaders.static_shader_get(RAY_TRACE_HARDWARE_LIGHTING);
    pass.specialize_constant(
        sh, "use_hardware_environment", reinterpret_cast<bool *>(&hardware_environment_enabled_));
    pass.specialize_constant(
        sh,
        "use_hardware_rt_shadows",
        reinterpret_cast<bool *>(&hardware_lighting_use_hardware_rt_shadows_));
    pass.specialize_constant(sh,
                             "use_hardware_rt_environment_visibility",
                             reinterpret_cast<bool *>(
                                 &hardware_lighting_use_hardware_rt_environment_visibility_));
    pass.specialize_constant(sh, "closure_index", &data_.closure_index);
    pass.shader_set(sh);
    pass.bind_ssbo("tiles_coord_buf", &hardware_resolve_tiles_buf_);
    pass.bind_texture(RBUFS_UTILITY_TEX_SLOT, inst_.pipelines.utility_tx);
    pass.bind_texture("depth_tx", &depth_tx);
    pass.bind_texture("hardware_rt_shadow_visibility_tx", &hardware_secondary_shadow_visibility_tx_);
    pass.bind_texture("radiance_front_tx", &screen_radiance_front_tx_);
    pass.bind_texture("radiance_back_tx", &screen_radiance_back_tx_);
    pass.bind_texture("hit_world_position_tx", &hit_world_position_tx_);
    pass.bind_texture("hit_transmission_layer_tx", &hit_throughput_tx_);
    pass.bind_texture("layered_receiver_throughput_tx", &layered_receiver_throughput_tx_);
    pass.bind_texture("layered_receiver_ray_time_tx", &layered_receiver_ray_time_tx_);
    pass.bind_texture("layered_receiver_ray_radiance_tx", &layered_receiver_ray_radiance_tx_);
    pass.bind_texture("layered_receiver_hit_albedo_tx", &layered_receiver_albedo_tx_);
    pass.bind_texture("layered_receiver_hit_material_tx", &layered_receiver_material_tx_);
    pass.bind_texture("layered_receiver_hit_normal_tx", &layered_receiver_normal_tx_);
    pass.bind_texture("layered_receiver_hit_position_tx", &layered_receiver_position_tx_);
    pass.bind_texture("layered_receiver_hit_identity_tx", &layered_receiver_identity_tx_);
    pass.bind_texture("layered_receiver_world_position_tx", &layered_receiver_world_position_tx_);
    pass.bind_texture("transmission_receiver_throughput_tx", &transmission_receiver_throughput_tx_);
    pass.bind_texture("transmission_receiver_ray_time_tx", &transmission_receiver_ray_time_tx_);
    pass.bind_texture("transmission_receiver_ray_radiance_tx", &transmission_receiver_ray_radiance_tx_);
    pass.bind_texture("transmission_receiver_hit_albedo_tx", &transmission_receiver_albedo_tx_);
    pass.bind_texture("transmission_receiver_hit_material_tx", &transmission_receiver_material_tx_);
    pass.bind_texture("transmission_receiver_hit_normal_tx", &transmission_receiver_normal_tx_);
    pass.bind_texture("transmission_receiver_hit_position_tx", &transmission_receiver_position_tx_);
    pass.bind_texture("transmission_receiver_hit_identity_tx", &transmission_receiver_identity_tx_);
    pass.bind_texture("transmission_receiver_world_position_tx", &transmission_receiver_world_position_tx_);
    pass.bind_texture("hardware_rt_environment_visibility_tx", &hardware_environment_visibility_tx_);
    pass.bind_texture("hardware_rt_hit_environment_visibility_tx",
                      &hardware_secondary_environment_visibility_tx_);
    pass.bind_texture("hardware_indirect_gi_radiance_cache_tx",
                      &hardware_indirect_gi_radiance_cache_tx_);
    pass.bind_texture("hardware_indirect_gi_position_cache_tx",
                      &hardware_indirect_gi_position_cache_tx_);
    pass.bind_texture("hardware_indirect_gi_normal_cache_tx",
                      &hardware_indirect_gi_normal_cache_tx_);
    pass.bind_texture("hardware_reflected_receiver_gi_tx", &hardware_reflected_receiver_gi_blur_tx_);
    pass.bind_texture("hardware_fast_gi_tx", &hardware_fast_gi_tx_, fast_gi_field_sampler);
    pass.bind_texture(
        "hardware_fast_gi_visibility_tx", &hardware_fast_gi_visibility_tx_, fast_gi_field_sampler);
    pass.bind_image("ray_data_img", &ray_data_tx_);
    pass.bind_image("ray_time_img", &ray_time_tx_);
    pass.bind_image("hit_albedo_img", &hit_albedo_tx_);
    pass.bind_image("hit_material_img", &hit_material_tx_);
    pass.bind_image("hit_normal_img", &hit_normal_tx_);
    pass.bind_image("hit_position_img", &hit_position_tx_);
    pass.bind_image("hit_identity_img", &hit_identity_tx_);
    pass.bind_image("hardware_caustics_img", &hardware_caustics_history_tx_);
    pass.bind_image("ray_radiance_img", &ray_radiance_tx_);
    inst_.lights.bind_no_cull_light_resources(pass);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.sampling);
    pass.bind_resources(inst_.volume_probes);
    pass.bind_resources(inst_.sphere_probes);
    pass.bind_resources(inst_.gbuffer);
    pass.bind_resources(inst_.lights);
    pass.bind_resources(inst_.shadows);
    pass.dispatch(hardware_resolve_dispatch_buf_);
    pass.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS);
  }
  {
    PassSimple &pass = hardware_reflected_receiver_gi_blur_ps_;
    pass.init();
    pass.shader_set(inst_.shaders.static_shader_get(RAY_HARDWARE_REFLECTED_RECEIVER_GI_BLUR));
    pass.bind_texture("reflected_receiver_gi_tx", &hardware_reflected_receiver_gi_tx_);
    pass.bind_texture("hit_normal_tx", &hit_normal_tx_);
    pass.bind_texture("hit_world_position_tx", &hit_world_position_tx_);
    pass.bind_texture("ray_time_tx", &ray_time_tx_);
    pass.bind_image("out_reflected_receiver_gi_img", &hardware_reflected_receiver_gi_blur_tx_);
    pass.bind_ssbo("tiles_coord_buf", &hardware_resolve_tiles_buf_);
    pass.push_constant("reflected_receiver_gi_resolution_divisor",
                       &hardware_reflected_receiver_gi_resolution_divisor_);
    pass.dispatch(hardware_resolve_dispatch_buf_);
    pass.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS | GPU_BARRIER_TEXTURE_FETCH);
  }
  /* Denoise. */
  {
    PassSimple &pass = denoise_spatial_ps_;
    gpu::Shader *sh = inst_.shaders.static_shader_get(RAY_DENOISE_SPATIAL);
    pass.init();
    pass.specialize_constant(sh, "closure_index", &data_.closure_index);
    pass.specialize_constant(sh, "raytrace_resolution_scale", &data_.resolution_scale);
    pass.specialize_constant(sh, "skip_denoise", reinterpret_cast<bool *>(&data_.skip_denoise));
    pass.shader_set(sh);
    pass.bind_ssbo("tiles_coord_buf", &raytrace_denoise_tiles_buf_);
    pass.bind_texture(RBUFS_UTILITY_TEX_SLOT, inst_.pipelines.utility_tx);
    pass.bind_texture("depth_tx", &depth_tx);
    pass.bind_image("ray_data_img", &ray_data_tx_);
    pass.bind_image("ray_time_img", &ray_time_tx_);
    pass.bind_image("ray_radiance_img", &ray_radiance_tx_);
    pass.bind_image("out_radiance_img", &denoised_spatial_tx_);
    pass.bind_image("out_variance_img", &hit_variance_tx_);
    pass.bind_image("out_hit_depth_img", &hit_depth_tx_);
    pass.bind_image("hit_position_img", &hit_position_tx_);
    pass.bind_image("tile_mask_img", &tile_raytrace_denoise_tx_);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.sampling);
    pass.bind_resources(inst_.gbuffer);
    pass.dispatch(raytrace_denoise_dispatch_buf_);
    /* Can either be loaded by next denoise pass as image or by combined pass as texture if this is
     * the lass stage. */
    pass.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS | GPU_BARRIER_TEXTURE_FETCH);
  }
  {
    PassSimple &pass = denoise_temporal_ps_;
    gpu::Shader *sh = inst_.shaders.static_shader_get(RAY_DENOISE_TEMPORAL);
    pass.init();
    pass.specialize_constant(sh, "closure_index", &data_.closure_index);
    pass.shader_set(sh);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_texture("radiance_history_tx", &radiance_history_tx_);
    pass.bind_texture("variance_history_tx", &variance_history_tx_);
    pass.bind_texture("tilemask_history_tx", &tilemask_history_tx_);
    pass.bind_texture("depth_tx", &depth_tx);
    pass.bind_image("hit_depth_img", &hit_depth_tx_);
    pass.bind_image("in_radiance_img", &denoised_spatial_tx_);
    pass.bind_image("out_radiance_img", &denoised_temporal_tx_);
    pass.bind_image("in_variance_img", &hit_variance_tx_);
    pass.bind_image("out_variance_img", &denoise_variance_tx_);
    pass.bind_ssbo("tiles_coord_buf", &raytrace_denoise_tiles_buf_);
    pass.bind_resources(inst_.sampling);
    pass.dispatch(raytrace_denoise_dispatch_buf_);
    /* Can either be loaded by next denoise pass as image or by combined pass as texture if this is
     * the lass stage. */
    pass.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS | GPU_BARRIER_TEXTURE_FETCH);
  }
  {
    PassSimple &pass = denoise_bilateral_ps_;
    pass.init();
    gpu::Shader *sh = inst_.shaders.static_shader_get(RAY_DENOISE_BILATERAL);
    pass.specialize_constant(sh, "closure_index", &data_.closure_index);
    pass.shader_set(sh);
    pass.bind_texture("depth_tx", &depth_tx);
    pass.bind_image("in_radiance_img", &denoised_temporal_tx_);
    pass.bind_image("out_radiance_img", &denoised_bilateral_tx_);
    pass.bind_image("in_variance_img", &denoise_variance_tx_);
    pass.bind_image("tile_mask_img", &tile_raytrace_denoise_tx_);
    pass.bind_ssbo("tiles_coord_buf", &raytrace_denoise_tiles_buf_);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.sampling);
    pass.bind_resources(inst_.gbuffer);
    pass.dispatch(raytrace_denoise_dispatch_buf_);
    /* Can either be loaded and written by horizon scan as image or by combined pass as texture. */
    pass.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS | GPU_BARRIER_TEXTURE_FETCH);
  }
  {
    PassSimple &pass = horizon_schedule_ps_;
    /* Reuse tile compaction shader but feed it with horizon scan specific buffers. */
    gpu::Shader *sh = inst_.shaders.static_shader_get(RAY_TILE_COMPACT);
    pass.init();
    pass.specialize_constant(sh, "closure_index", 0);
    pass.specialize_constant(sh, "resolution_scale", &data_.horizon_resolution_scale);
    pass.shader_set(sh);
    pass.bind_image("tile_raytrace_denoise_img", &tile_horizon_denoise_tx_);
    pass.bind_image("tile_raytrace_tracing_img", &tile_horizon_tracing_tx_);
    pass.bind_ssbo("raytrace_tracing_dispatch_buf", &horizon_tracing_dispatch_buf_);
    pass.bind_ssbo("raytrace_denoise_dispatch_buf", &horizon_denoise_dispatch_buf_);
    pass.bind_ssbo("raytrace_tracing_tiles_buf", &horizon_tracing_tiles_buf_);
    pass.bind_ssbo("raytrace_denoise_tiles_buf", &horizon_denoise_tiles_buf_);
    pass.bind_resources(inst_.uniform_data);
    pass.dispatch(&horizon_schedule_dispatch_size_);
    pass.barrier(GPU_BARRIER_SHADER_STORAGE);
  }
  {
    PassSimple &pass = horizon_setup_ps_;
    pass.init();
    pass.shader_set(inst_.shaders.static_shader_get(HORIZON_SETUP));
    pass.bind_resources(inst_.uniform_data);
    pass.bind_texture("depth_tx", &depth_tx);
    pass.bind_texture(
        "in_radiance_tx", &screen_radiance_front_tx_, GPUSamplerState::default_sampler());
    pass.bind_image("out_radiance_img", &downsampled_in_radiance_tx_);
    pass.bind_image("out_normal_img", &downsampled_in_normal_tx_);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.gbuffer);
    pass.dispatch(&horizon_tracing_dispatch_size_);
    /* Result loaded by the next stage using samplers. */
    pass.barrier(GPU_BARRIER_TEXTURE_FETCH);
  }
  {
    PassSimple &pass = horizon_scan_ps_;
    pass.init();
    gpu::Shader *sh = inst_.shaders.static_shader_get(HORIZON_SCAN);
    pass.specialize_constant(sh, "slice_count", fast_gi_ray_count_);
    pass.specialize_constant(sh, "step_count", fast_gi_step_count_);
    pass.specialize_constant(sh, "ao_only", fast_gi_ao_only_);
    pass.shader_set(sh);
    pass.bind_texture("screen_radiance_tx", &downsampled_in_radiance_tx_);
    pass.bind_texture("screen_normal_tx", &downsampled_in_normal_tx_);
    pass.bind_image("sh_0_img", &horizon_radiance_tx_[0]);
    pass.bind_image("sh_1_img", &horizon_radiance_tx_[1]);
    pass.bind_image("sh_2_img", &horizon_radiance_tx_[2]);
    pass.bind_image("sh_3_img", &horizon_radiance_tx_[3]);
    pass.bind_ssbo("tiles_coord_buf", &horizon_tracing_tiles_buf_);
    pass.bind_texture(RBUFS_UTILITY_TEX_SLOT, inst_.pipelines.utility_tx);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.hiz_buffer.front);
    pass.bind_resources(inst_.sampling);
    pass.bind_resources(inst_.gbuffer);
    pass.dispatch(horizon_tracing_dispatch_buf_);
    /* Result loaded by the next stage using samplers. */
    pass.barrier(GPU_BARRIER_TEXTURE_FETCH);
  }
  {
    PassSimple &pass = horizon_denoise_ps_;
    pass.init();
    gpu::Shader *sh = inst_.shaders.static_shader_get(HORIZON_DENOISE);
    pass.shader_set(sh);
    pass.bind_texture("depth_tx", &depth_tx);
    pass.bind_texture("horizon_radiance_0_tx", &horizon_radiance_tx_[0]);
    pass.bind_texture("horizon_radiance_1_tx", &horizon_radiance_tx_[1]);
    pass.bind_texture("horizon_radiance_2_tx", &horizon_radiance_tx_[2]);
    pass.bind_texture("horizon_radiance_3_tx", &horizon_radiance_tx_[3]);
    pass.bind_texture("screen_normal_tx", &downsampled_in_normal_tx_);
    pass.bind_image("sh_0_img", &horizon_radiance_denoised_tx_[0]);
    pass.bind_image("sh_1_img", &horizon_radiance_denoised_tx_[1]);
    pass.bind_image("sh_2_img", &horizon_radiance_denoised_tx_[2]);
    pass.bind_image("sh_3_img", &horizon_radiance_denoised_tx_[3]);
    pass.bind_ssbo("tiles_coord_buf", &horizon_tracing_tiles_buf_);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.sampling);
    pass.bind_resources(inst_.hiz_buffer.front);
    pass.dispatch(horizon_tracing_dispatch_buf_);
    /* Result loaded by the next stage using samplers. */
    pass.barrier(GPU_BARRIER_TEXTURE_FETCH);
  }
  {
    PassSimple &pass = horizon_resolve_ps_;
    pass.init();
    gpu::Shader *sh = inst_.shaders.static_shader_get(HORIZON_RESOLVE);
    pass.shader_set(sh);
    pass.bind_texture("depth_tx", &depth_tx);
    pass.bind_texture("horizon_radiance_0_tx", &horizon_radiance_denoised_tx_[0]);
    pass.bind_texture("horizon_radiance_1_tx", &horizon_radiance_denoised_tx_[1]);
    pass.bind_texture("horizon_radiance_2_tx", &horizon_radiance_denoised_tx_[2]);
    pass.bind_texture("horizon_radiance_3_tx", &horizon_radiance_denoised_tx_[3]);
    pass.bind_texture("screen_normal_tx", &downsampled_in_normal_tx_);
    pass.bind_image("closure0_img", &horizon_scan_output_tx_[0]);
    pass.bind_image("closure1_img", &horizon_scan_output_tx_[1]);
    pass.bind_image("closure2_img", &horizon_scan_output_tx_[2]);
    pass.bind_ssbo("tiles_coord_buf", &horizon_denoise_tiles_buf_);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.sampling);
    pass.bind_resources(inst_.gbuffer);
    pass.bind_resources(inst_.volume_probes);
    pass.bind_resources(inst_.sphere_probes);
    pass.dispatch(horizon_denoise_dispatch_buf_);
    /* Can either be loaded by another denoising stage or by combined pass as texture. */
    pass.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS | GPU_BARRIER_TEXTURE_FETCH);
  }

  for (int i : IndexRange(3)) {
    const bool use_denoise = (ray_tracing_options_.flag & RAYTRACE_EEVEE_USE_DENOISE);
    const bool use_spatial_denoise = (ray_tracing_options_.denoise_stages &
                                      RAYTRACE_EEVEE_DENOISE_SPATIAL) &&
                                     use_denoise;
    const bool use_temporal_denoise = (ray_tracing_options_.denoise_stages &
                                       RAYTRACE_EEVEE_DENOISE_TEMPORAL) &&
                                      use_spatial_denoise;
    const bool use_bilateral_denoise = (ray_tracing_options_.denoise_stages &
                                        RAYTRACE_EEVEE_DENOISE_BILATERAL) &&
                                       use_temporal_denoise;

    data_.closure_index = i;
    data_.resolution_scale = hardware_interactive_resolution_scale(
        inst_,
        active_hardware_feature_mask(),
        effective_hardware_resolution_scale(active_hardware_feature_mask(),
                                            ray_tracing_options_.resolution_scale,
                                            hardware_reflection_mode_,
                                            hardware_refraction_mode_));
    data_.skip_denoise = !use_spatial_denoise;
    data_.use_hardware_ign_sampling = use_hardware_rt_gi();
    inst_.manager->warm_shader_specialization(tile_classify_ps_);
    inst_.manager->warm_shader_specialization(tile_compact_ps_);
    inst_.manager->warm_shader_specialization(hardware_direct_light_tile_compact_ps_);
    inst_.manager->warm_shader_specialization(hardware_direct_light_visibility_ps_);
    inst_.manager->warm_shader_specialization(hardware_direct_light_accum_ps_);
    inst_.manager->warm_shader_specialization(hardware_direct_light_denoise_ps_);
    inst_.manager->warm_shader_specialization(generate_ps_);
    warm_tracing_backend();
    if (use_spatial_denoise) {
      inst_.manager->warm_shader_specialization(denoise_spatial_ps_);
    }
    if (use_temporal_denoise) {
      inst_.manager->warm_shader_specialization(denoise_temporal_ps_);
    }
    if (use_bilateral_denoise) {
      inst_.manager->warm_shader_specialization(denoise_bilateral_ps_);
    }
    bool use_horizon_scan = this->use_horizon_scan(ray_tracing_options_);
    if (use_horizon_scan) {
      inst_.manager->warm_shader_specialization(horizon_schedule_ps_);
      inst_.manager->warm_shader_specialization(horizon_setup_ps_);
      inst_.manager->warm_shader_specialization(horizon_scan_ps_);
      inst_.manager->warm_shader_specialization(horizon_denoise_ps_);
      inst_.manager->warm_shader_specialization(horizon_resolve_ps_);
    }
  }
}

void RayTraceModule::debug_pass_sync() {}

void RayTraceModule::debug_draw(View & /*view*/, gpu::FrameBuffer * /*view_fb*/) {}

static void raytrace_history_invalidate_on_viewport_reset(RayTraceBuffer &rt_buffer)
{
  rt_buffer.history_persmat = float4x4::zero();
  if (rt_buffer.radiance_feedback_tx.is_valid()) {
    rt_buffer.radiance_feedback_tx.clear(float4(0.0f));
  }

  for (RayTraceBuffer::DenoiseBuffer &denoise_buf : rt_buffer.closures) {
    denoise_buf.history_persmat = float4x4::zero();
    denoise_buf.valid_history = false;
    denoise_buf.valid_screen_ownership_history = false;
  }
}

RayTraceResult RayTraceModule::render(RayTraceBuffer &rt_buffer,
                                      gpu::Texture *screen_radiance_back_tx,
                                      eClosureBits active_closures,
                                      /* TODO(fclem): Maybe wrap these two in some other class. */
                                      View &main_view,
                                      View &render_view)
{
  return render_phase(rt_buffer,
                      nullptr,
                      screen_radiance_back_tx,
                      active_closures,
                      main_view,
                      render_view,
                      HWRT_TRACE_PHASE_FULL,
                      UINT32_MAX,
                      true);
}

RayTraceResult RayTraceModule::render_phase(RayTraceBuffer &rt_buffer,
                                            gpu::Texture *screen_radiance_front_tx,
                                            gpu::Texture *screen_radiance_back_tx,
                                            eClosureBits active_closures,
                                            View &main_view,
                                            View &render_view,
                                            eHardwareTracePhase trace_phase,
                                            uint32_t feature_mask_override,
                                            bool enable_horizon_scan)
{
  using namespace blender::math;
  BLI_assert(use_raytracing_);

  const bool history_reset = viewport_history_reset_ && !hardware_indirect_gi_cache_rendering_;
  if (!hardware_indirect_gi_cache_rendering_) {
    viewport_history_reset_ = false;
  }
  if (history_reset) {
    /* Viewport resets already invalidate accumulation, but the Hardware RT path also carries
     * per-closure radiance/ownership histories and screen-feedback across frames. If those remain
     * valid after a scene edit, the next traced frame can blend pre-edit state back into updated
     * geometry or lighting. */
    raytrace_history_invalidate_on_viewport_reset(rt_buffer);
    /* Keep the world-space diffuse GI field warm across viewport resets. Camera motion and other
     * live viewport resets should not force the current traced field to restart from black every
     * frame. */
    if (!inst_.is_viewport() || !use_hardware_fast_gi()) {
      hardware_fast_gi_valid_ = false;
      hardware_fast_gi_tx_.clear(float4(0.0f));
    }
  }

  screen_radiance_front_tx_ = screen_radiance_front_tx ? screen_radiance_front_tx :
                                                        (rt_buffer.radiance_feedback_tx.is_valid() ?
                                                             rt_buffer.radiance_feedback_tx :
                                                             radiance_dummy_black_tx_);
  screen_radiance_back_tx_ = screen_radiance_back_tx ? screen_radiance_back_tx :
                                                       screen_radiance_front_tx_;

  RaytraceEEVEE options = ray_tracing_options_;
  const bool needs_diffuse_rt_path = use_hardware_rt_gi() &&
                                     (trace_phase != HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR);
  if (needs_diffuse_rt_path) {
    /* Traced Hardware GI still needs the diffuse ray path enabled locally, but keep the shared
     * scene options unchanged so feature-disabled cases still preserve the classic screen/probe
     * behavior. */
    options.trace_max_roughness = 1.0f;
  }
  const uint32_t enabled_hardware_specular_mask = active_hardware_feature_mask() &
                                                  (RAYTRACE_EEVEE_HARDWARE_REFLECTIONS |
                                                   RAYTRACE_EEVEE_HARDWARE_REFRACTIONS);
  if (enabled_hardware_specular_mask != 0 && !needs_diffuse_rt_path) {
    /* Keep Principled dielectric diffuse-reflection replay on through the full roughness range in
     * direct view instead of fading it out through the shared roughness-mask band. */
    options.trace_max_roughness = 1.0f;
  }

  bool use_horizon_scan = enable_horizon_scan && this->use_horizon_scan(options);
  current_hardware_feature_mask_ = (feature_mask_override == UINT32_MAX) ?
                                       filtered_hardware_feature_mask(*this, active_closures) :
                                       feature_mask_override;
  current_trace_active_closures_ = active_closures;

  const int resolution_scale = hardware_interactive_resolution_scale(
      inst_,
      current_hardware_feature_mask_,
      effective_hardware_resolution_scale(current_hardware_feature_mask_,
                                          options.resolution_scale,
                                          hardware_reflection_mode_,
                                          hardware_refraction_mode_));
  const HardwareFastGISceneScaleAnalysis fast_gi_scene_analysis = hardware_fast_gi_scene_scale_analysis(
      inst_.sync.hardware_raytrace_scene_entries(),
      render_view.viewinv().location(),
      inst_.camera.forward(),
      inst_.camera.data_get().clip_far);
  const int adaptive_scene_priority = hardware_fast_gi_scene_priority(
      fast_gi_scene_analysis, inst_.lights.culling_data());
  const float3 fast_gi_field_center = hardware_fast_gi_field_center(
      fast_gi_scene_analysis, adaptive_scene_priority, render_view.viewinv().location());
  const bool use_viewport_sized_fast_gi = use_hardware_fast_gi() || inst_.is_viewport();
  const int adaptive_quality_tier = hardware_fast_gi_quality_tier(
      use_viewport_sized_fast_gi,
      hardware_fast_gi_smoothed_traced_ms_,
      fast_gi_scene_analysis,
      adaptive_scene_priority);
  const int adaptive_budget_rebalance = hardware_fast_gi_budget_rebalance(
      adaptive_quality_tier, adaptive_scene_priority, fast_gi_scene_analysis, inst_.lights.culling_data());
  hardware_debug_view_mode_ = hardware_debug_view_mode();
  hardware_debug_isolate_mode_ = hardware_debug_isolate_mode();
  if (hardware_indirect_gi_cache_rendering_) {
    hardware_debug_isolate_mode_ = HWRT_DEBUG_ISOLATE_INDIRECT;
  }
  hardware_fast_gi_freeze_updates_ = hardware_fast_gi_freeze_updates_enabled();
  const int requested_hardware_fast_gi_grid_resolution = hardware_fast_gi_requested_grid_resolution(
      inst_.scene->eevee.fast_gi_resolution,
      fast_gi_scene_analysis,
      adaptive_quality_tier,
      adaptive_budget_rebalance);
  const int requested_hardware_fast_gi_cascade_count = use_viewport_sized_fast_gi ? 2 : 3;
  const HardwareFastGIMemoryLayout fast_gi_memory_layout = hardware_fast_gi_fit_memory_budget(
      requested_hardware_fast_gi_grid_resolution,
      requested_hardware_fast_gi_cascade_count,
      use_viewport_sized_fast_gi);
  const int hardware_fast_gi_grid_resolution = fast_gi_memory_layout.grid_resolution;
  const int hardware_fast_gi_cascade_count = fast_gi_memory_layout.cascade_count;
  const float hardware_fast_gi_distance = hardware_fast_gi_requested_distance(
      inst_.scene->eevee.fast_gi_distance,
      fast_gi_scene_analysis,
      inst_.camera.data_get().clip_far,
      adaptive_scene_priority,
      use_viewport_sized_fast_gi);
  const float hardware_fast_gi_base_cell_size = max_ff(
      hardware_fast_gi_distance / float(2 * hardware_fast_gi_grid_resolution), 0.25f);
  const int horizon_resolution_scale = max_ii(
      1, power_of_2_max_i(inst_.scene->eevee.fast_gi_resolution));

  const int2 extent = inst_.render_buffers.extent_get();
  const int2 tracing_res = math::divide_ceil(extent, int2(resolution_scale));
  const int2 tracing_res_horizon = math::divide_ceil(extent, int2(horizon_resolution_scale));
  const int2 group_size(RAYTRACE_GROUP_SIZE);

  const int2 denoise_tiles = divide_ceil(extent, group_size);
  const int2 raytrace_tiles = divide_ceil(tracing_res, group_size);
  const int2 raytrace_tiles_horizon = divide_ceil(tracing_res_horizon, group_size);
  const int denoise_tile_count = denoise_tiles.x * denoise_tiles.y;
  const int raytrace_tile_count = raytrace_tiles.x * raytrace_tiles.y;
  const int raytrace_tile_count_horizon = raytrace_tiles_horizon.x * raytrace_tiles_horizon.y;
  const LightCullingData &light_culling_data = inst_.lights.culling_data();
  const int direct_light_tile_count = int(light_culling_data.tile_x_len *
                                          light_culling_data.tile_y_len);
  tile_classify_dispatch_size_ = int3(denoise_tiles, 1);
  horizon_schedule_dispatch_size_ = int3(divide_ceil(raytrace_tiles_horizon, group_size), 1);
  tile_compact_dispatch_size_ = int3(divide_ceil(raytrace_tiles, group_size), 1);
  hardware_direct_light_tile_compact_dispatch_size_ = int3(
      max_ii((direct_light_tile_count + 63) / 64, 1), 1, 1);
  tracing_dispatch_size_ = int3(raytrace_tiles, 1);
  horizon_tracing_dispatch_size_ = int3(raytrace_tiles_horizon, 1);

  /* TODO(fclem): Use real max closure count from shader. */
  const int closure_count = 3;
  gpu::TextureFormat format = gpu::TextureFormat::RAYTRACE_TILEMASK_FORMAT;
  eGPUTextureUsage usage_rw = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;
  const bool use_receiver_caustics = use_hardware_tracing() && use_hardware_caustics();
  const bool caustics_recreated = hardware_caustics_history_tx_.ensure_2d(
      gpu::TextureFormat::SFLOAT_16_16_16_16,
      use_receiver_caustics ? extent : int2(1),
      usage_rw);
  if (history_reset || caustics_recreated || !use_receiver_caustics) {
    hardware_caustics_history_tx_.clear(float4(0.0f));
  }
  const bool use_fast_gi_field = use_hardware_tracing() && use_hardware_fast_gi();
  const bool fast_gi_recreated = hardware_fast_gi_tx_.ensure_3d(
      gpu::TextureFormat::SFLOAT_16_16_16_16,
      use_fast_gi_field ?
          int3(hardware_fast_gi_grid_resolution,
               hardware_fast_gi_grid_resolution,
               hardware_fast_gi_grid_resolution * hardware_fast_gi_cascade_count) :
          int3(1),
      usage_rw);
  const bool fast_gi_error_recreated = hardware_fast_gi_error_tx_.ensure_3d(
      gpu::TextureFormat::SFLOAT_16,
      use_fast_gi_field ?
          int3(hardware_fast_gi_grid_resolution,
               hardware_fast_gi_grid_resolution,
               hardware_fast_gi_grid_resolution * hardware_fast_gi_cascade_count) :
          int3(1),
      usage_rw);
  const bool fast_gi_visibility_recreated = hardware_fast_gi_visibility_tx_.ensure_3d(
      gpu::TextureFormat::SFLOAT_16_16_16_16,
      use_fast_gi_field ?
          int3(hardware_fast_gi_grid_resolution,
               hardware_fast_gi_grid_resolution,
               hardware_fast_gi_grid_resolution * hardware_fast_gi_cascade_count) :
          int3(1),
      usage_rw);
  const bool invalidate_fast_gi_for_reset = history_reset && !inst_.is_viewport();
  if (invalidate_fast_gi_for_reset || fast_gi_recreated || fast_gi_error_recreated ||
      fast_gi_visibility_recreated || !use_fast_gi_field)
  {
    hardware_fast_gi_tx_.clear(float4(0.0f));
    hardware_fast_gi_error_tx_.clear(float4(0.0f));
    hardware_fast_gi_visibility_tx_.clear(float4(0.0f));
    hardware_fast_gi_valid_ = false;
    hardware_fast_gi_depsgraph_update_count_valid_ = false;
    hardware_fast_gi_field_config_valid_ = false;
    if (!use_fast_gi_field) {
      hardware_fast_gi_light_invalidation_pending_ = false;
      hardware_fast_gi_world_invalidation_pending_ = false;
      hardware_fast_gi_emissive_invalidation_pending_ = false;
      hardware_fast_gi_material_invalidation_pending_ = false;
      hardware_fast_gi_transform_invalidation_pending_ = false;
      hardware_fast_gi_geometry_invalidation_pending_ = false;
      hardware_fast_gi_animation_invalidation_pending_ = false;
    }
  }

  tile_raytrace_denoise_tx_.ensure_2d_array(format, denoise_tiles, closure_count, usage_rw);
  tile_raytrace_tracing_tx_.ensure_2d_array(format, raytrace_tiles, closure_count, usage_rw);
  /* Kept as 2D array for compatibility with the tile compaction shader. */
  tile_horizon_denoise_tx_.ensure_2d_array(format, denoise_tiles, 1, usage_rw);
  tile_horizon_tracing_tx_.ensure_2d_array(format, raytrace_tiles_horizon, 1, usage_rw);

  tile_raytrace_denoise_tx_.clear(uint4(0u));
  tile_raytrace_tracing_tx_.clear(uint4(0u));
  tile_horizon_denoise_tx_.clear(uint4(0u));
  tile_horizon_tracing_tx_.clear(uint4(0u));

  horizon_tracing_tiles_buf_.resize(ceil_to_multiple_u(raytrace_tile_count_horizon, 512));
  horizon_denoise_tiles_buf_.resize(ceil_to_multiple_u(denoise_tile_count, 512));
  raytrace_tracing_tiles_buf_.resize(ceil_to_multiple_u(raytrace_tile_count, 512));
  raytrace_denoise_tiles_buf_.resize(ceil_to_multiple_u(denoise_tile_count, 512));
  hardware_direct_light_work_tiles_buf_.resize(
      ceil_to_multiple_u(max_ii(direct_light_tile_count, 1), 512));
  hardware_direct_light_visibility_samples_buf_.resize(
      ceil_to_multiple_u(max_ii(direct_light_tile_count, 1), 512));
  hardware_trace_tiles_buf_.resize(ceil_to_multiple_u(raytrace_tile_count, 512));
  hardware_resolve_tiles_buf_.resize(ceil_to_multiple_u(raytrace_tile_count, 512));

  /* Data for tile classification. */
  float roughness_mask_start = options.trace_max_roughness;
  float roughness_mask_fade = 0.2f;
  if (enabled_hardware_specular_mask != 0) {
    roughness_mask_fade = 0.5f;
  }
  data_.roughness_mask_scale = 1.0 / roughness_mask_fade;
  data_.roughness_mask_bias = data_.roughness_mask_scale * roughness_mask_start;

  /* Data for the radiance setup. */
  data_.resolution_scale = resolution_scale;
  data_.resolution_bias = int2(inst_.sampling.rng_2d_get(SAMPLING_RAYTRACE_V) * resolution_scale);
  data_.history_persmat = rt_buffer.history_persmat;
  data_.radiance_persmat = render_view.persmat();
  data_.full_resolution = extent;
  data_.full_resolution_inv = 1.0f / float2(extent);
  data_.hardware_gi_bounces = hardware_gi_fixed_bounces;
  data_.hardware_gi_mode = int(hardware_gi_mode_);
  data_.hardware_reflection_bounces = effective_hardware_specular_bounces(
      inst_.scene->eevee.ray_tracing_reflection_bounces, hardware_reflection_mode_);
  data_.hardware_refraction_bounces = effective_hardware_specular_bounces(
      inst_.scene->eevee.ray_tracing_refraction_bounces, hardware_refraction_mode_);
  data_.hardware_caustics_samples = max_ii(1, inst_.scene->eevee.ray_tracing_caustics_samples);
  data_.hardware_reflection_mode = int(hardware_reflection_mode_);
  data_.hardware_refraction_mode = int(hardware_refraction_mode_);
  data_.use_hardware_fast_gi = use_hardware_fast_gi();
  data_.use_hardware_fast_gi_field = has_hardware_fast_gi_field();
  data_.use_hardware_caustics = use_hardware_caustics();
  data_.use_hardware_ign_sampling = use_hardware_rt_gi();
  data_.hardware_feature_mask = current_hardware_feature_mask_;
  data_.use_hardware_tracing_method = use_hardware_tracing_method();
  data_.hardware_trace_phase = int(trace_phase);
  data_.hardware_fast_gi_grid_resolution = hardware_fast_gi_grid_resolution;
  data_.hardware_fast_gi_cascade_count = hardware_fast_gi_cascade_count;
  data_.hardware_debug_view_mode = hardware_debug_view_mode_;
  data_.hardware_debug_isolate_mode = hardware_debug_isolate_mode_;
  data_.hardware_debug_freeze_updates = hardware_fast_gi_freeze_updates_ ? 1 : 0;
  data_.hardware_direct_light = hardware_direct_light_data(
      inst_.lights.culling_data(),
      hardware_world_sun_light_count(inst_, inst_.lights.culling_data()),
      inst_.is_viewport(),
      adaptive_quality_tier,
      adaptive_budget_rebalance);
  hardware_fast_gi_memory_limited_ = use_fast_gi_field && fast_gi_memory_layout.memory_limited;
  hardware_fast_gi_budget_bytes_ = use_fast_gi_field ? fast_gi_memory_layout.budget_bytes : 0;
  hardware_fast_gi_requested_bytes_ = use_fast_gi_field ? fast_gi_memory_layout.requested_bytes : 0;
  hardware_fast_gi_allocated_bytes_ = use_fast_gi_field ? fast_gi_memory_layout.allocated_bytes : 0;
  hardware_fast_gi_requested_grid_resolution_ = requested_hardware_fast_gi_grid_resolution;
  hardware_fast_gi_requested_cascade_count_ = requested_hardware_fast_gi_cascade_count;
  hardware_fast_gi_quality_tier_ = adaptive_quality_tier;
  hardware_fast_gi_scene_priority_ = adaptive_scene_priority;
  hardware_fast_gi_budget_rebalance_ = adaptive_budget_rebalance;
  hardware_direct_light_sample_count_ = data_.hardware_direct_light.light_samples_per_shading_point;
  hardware_fast_gi_requested_distance_ = hardware_fast_gi_distance;
  hardware_fast_gi_field_center_ = fast_gi_field_center;
  hardware_fast_gi_scene_radius_ = fast_gi_scene_analysis.scene_radius;
  hardware_fast_gi_scene_density_ = fast_gi_scene_analysis.density;

  float4 target_fast_gi_cascade_config[3];
  hardware_fast_gi_cascade_config_fill(target_fast_gi_cascade_config,
                                       hardware_fast_gi_cascade_count,
                                       fast_gi_field_center,
                                       hardware_fast_gi_base_cell_size,
                                       hardware_fast_gi_grid_resolution,
                                       hardware_fast_gi_field_config_valid_,
                                       hardware_fast_gi_field_cascade_config_);

  /* The visible frame samples the field that is already resident. If the field producer later
   * recenters or snaps the cascades, that new mapping becomes visible only with the matching
   * freshly-written texture on the next frame. */
  const bool sample_existing_fast_gi_field = use_fast_gi_field && hardware_fast_gi_valid_ &&
                                            hardware_fast_gi_field_config_valid_;
  const float4 *sample_fast_gi_cascade_config = sample_existing_fast_gi_field ?
                                                    hardware_fast_gi_field_cascade_config_ :
                                                    target_fast_gi_cascade_config;
  for (const int cascade_index : IndexRange(3)) {
    data_.hardware_fast_gi_cascade_config[cascade_index] =
        sample_fast_gi_cascade_config[cascade_index];
  }

  data_.horizon_resolution_scale = horizon_resolution_scale;
  data_.horizon_resolution_bias = int2(inst_.sampling.rng_2d_get(SAMPLING_RAYTRACE_V) *
                                       horizon_resolution_scale);
  /* TODO(fclem): Eventually all uniform data is setup here. */

  inst_.uniform_data.push_update();

  RayTraceResult result;

  GPU_debug_group_begin("Raytracing");

  const bool has_active_closure = active_closures != CLOSURE_NONE;

  if (has_active_closure) {
    inst_.manager->submit(tile_classify_ps_);
  }
  if (use_hardware_tracing()) {
    hardware_direct_light_dispatch_buf_.clear_to_zero();
    inst_.manager->submit(hardware_direct_light_tile_compact_ps_);
  }

  data_.trace_refraction = screen_radiance_back_tx != nullptr;

  for (int i = 0; i < 3; i++) {
    result.closures[i] = trace(i, (closure_count > i), options, rt_buffer, main_view, render_view);
  }

  if (has_active_closure) {
    if (use_horizon_scan) {
      GPU_debug_group_begin("Horizon Scan");

      downsampled_in_radiance_tx_.acquire(
          tracing_res_horizon, gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, usage_rw);
      downsampled_in_normal_tx_.acquire(
          tracing_res_horizon, gpu::TextureFormat::UNORM_10_10_10_2, usage_rw);

      horizon_radiance_tx_[0].acquire(
          tracing_res_horizon, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
      horizon_radiance_denoised_tx_[0].acquire(
          tracing_res_horizon, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
      for (int i : IndexRange(1, 3)) {
        horizon_radiance_tx_[i].acquire(
            tracing_res_horizon, gpu::TextureFormat::UNORM_8_8_8_8, usage_rw);
        horizon_radiance_denoised_tx_[i].acquire(
            tracing_res_horizon, gpu::TextureFormat::UNORM_8_8_8_8, usage_rw);
      }
      for (int i : IndexRange(3)) {
        horizon_scan_output_tx_[i] = result.closures[i];
      }

      horizon_tracing_dispatch_buf_.clear_to_zero();
      horizon_denoise_dispatch_buf_.clear_to_zero();
      inst_.manager->submit(horizon_schedule_ps_);

      inst_.manager->submit(horizon_setup_ps_, render_view);
      inst_.manager->submit(horizon_scan_ps_, render_view);
      inst_.manager->submit(horizon_denoise_ps_, render_view);
      inst_.manager->submit(horizon_resolve_ps_, render_view);

      for (int i : IndexRange(4)) {
        horizon_radiance_tx_[i].release();
        horizon_radiance_denoised_tx_[i].release();
      }
      downsampled_in_radiance_tx_.release();
      downsampled_in_normal_tx_.release();

      GPU_debug_group_end();
    }
  }

  GPU_debug_group_end();

  rt_buffer.history_persmat = render_view.persmat();
  current_hardware_feature_mask_ = 0;
  current_trace_active_closures_ = CLOSURE_NONE;
  data_.hardware_trace_phase = int(HWRT_TRACE_PHASE_FULL);

  return result;
}

static int hardware_indirect_gi_resolution_sanitize(const int value)
{
  switch (value) {
    case 1:
    case 2:
    case 4:
      return value;
    default:
      return 4;
  }
}

void RayTraceModule::render_reflected_receiver_gi(GPUMetalRaytraceScene *metal_scene,
                                                  int2 tracing_extent)
{
  if (metal_scene == nullptr ||
      data_.hardware_trace_phase != int(HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR) ||
      !inst_.scene->eevee.use_hardware_raytracing_indirect_gi_cache)
  {
    return;
  }

  const SphereProbe &world_probe = inst_.sphere_probes.world_sphere_probe();
  const bool world_probe_available = world_probe.atlas_coord.atlas_layer >= 0 &&
                                     world_probe.atlas_coord.subdivision_lvl >= 0;
  const SphereProbeUvArea world_probe_atlas_coord = world_probe_available ?
                                                       world_probe.atlas_coord.as_sampling_coord() :
                                                       SphereProbeUvArea{
                                                           float2(0.0f), 0.0f, -1.0f};
  Vector<LightData> local_lights;
  Vector<LightData> sun_lights;
  Vector<int> sun_light_world_slots;
  inst_.lights.append_sync_local_lights(local_lights);
  inst_.lights.append_sync_sun_lights(sun_lights, &sun_light_world_slots);
  const int light_count = min_ii(256, int(local_lights.size() + sun_lights.size()));
  for (int light_index = 0; light_index < light_count; light_index++) {
    const LightData &light = (light_index < local_lights.size()) ?
                                 local_lights[light_index] :
                                 sun_lights[light_index - local_lights.size()];
    hardware_fast_gi_light_buf_.get_or_resize(light_index) =
        hardware_fast_gi_light_record_from_light(light);
  }
  if (light_count > 0) {
    hardware_fast_gi_light_buf_.resize(light_count);
    hardware_fast_gi_light_buf_.push_update();
  }
  GPUMetalRaytraceReflectedReceiverGIParams params;
  params.receiver_gi_tx = hardware_reflected_receiver_gi_tx_;
  params.world_probe_tx = inst_.sphere_probes.octahedral_probes_texture();
  params.light_buf = (light_count > 0) ?
                         static_cast<gpu::StorageBuf *>(hardware_fast_gi_light_buf_) :
                         nullptr;
  params.dispatch_buf = hardware_resolve_dispatch_buf_;
  params.tiles_coord_buf = hardware_resolve_tiles_buf_;
  params.ray_time_tx = ray_time_tx_;
  params.hit_albedo_tx = hit_albedo_tx_;
  params.hit_normal_tx = hit_normal_tx_;
  params.hit_world_position_tx = hit_world_position_tx_;
  params.tracing_resolution = tracing_extent;
  hardware_reflected_receiver_gi_resolution_divisor_ = hardware_indirect_gi_resolution_sanitize(
      inst_.scene->eevee.hardware_raytracing_indirect_gi_resolution);
  params.resolution_divisor = hardware_reflected_receiver_gi_resolution_divisor_;
  params.sample_count = inst_.is_viewport() ? 2 : 4;
  params.light_count = light_count;
  params.light_sample_count = min_ii(inst_.is_viewport() ? 1 : 2,
                                     hardware_fast_gi_direct_light_sample_count(
                                         light_count, inst_.is_viewport(), hardware_fast_gi_quality_tier_));
  params.normal_bias = 1.0e-3f;
  params.use_environment = use_hardware_gi() || use_hardware_environment();
  const float3 raytrace_rng = inst_.sampling.rng_3d_get(eSamplingDimension::SAMPLING_RAYTRACE_U);
  params.sampling_rand = float4(
      raytrace_rng.x,
      raytrace_rng.y,
      raytrace_rng.z,
      inst_.sampling.rng_get(eSamplingDimension::SAMPLING_CLOSURE));
  params.world_probe_atlas_coord = float4(world_probe_atlas_coord.offset.x,
                                          world_probe_atlas_coord.offset.y,
                                          world_probe_atlas_coord.scale,
                                          world_probe_atlas_coord.layer);
  if (!GPU_metal_raytrace_scene_trace_reflected_receiver_gi(metal_scene, params)) {
    return;
  }

  GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH | GPU_BARRIER_SHADER_IMAGE_ACCESS);
  inst_.manager->submit(hardware_reflected_receiver_gi_blur_ps_);
}

void RayTraceModule::render_hardware_indirect_gi_cache(View &main_view)
{
  hardware_indirect_gi_cache_valid_ = false;
  const float4 zero(0.0f);
  GPU_texture_clear(hardware_indirect_gi_radiance_cache_tx_, GPU_DATA_FLOAT, &zero);
  GPU_texture_clear(hardware_indirect_gi_position_cache_tx_, GPU_DATA_FLOAT, &zero);
  GPU_texture_clear(hardware_indirect_gi_normal_cache_tx_, GPU_DATA_FLOAT, &zero);
  if (hardware_indirect_gi_cache_rendering_ ||
      !inst_.scene->eevee.use_hardware_raytracing_indirect_gi_cache || !use_hardware_tracing() ||
      !use_hardware_rt_gi() || !use_hardware_reflections())
  {
    return;
  }

  const eClosureBits active_closures = inst_.pipelines.deferred.closure_bits_get();
  if ((active_closures & (CLOSURE_DIFFUSE | CLOSURE_SSS)) == 0) {
    return;
  }

  /* The cache renders before the main view's light-culling submit. Prime the world-sun HWRT
   * buffers here because secondary hit shadows fetch their Metal resources immediately. */
  inst_.world.sync_sunlight_rt_direction_overrides();

  const int requested_resolution_scale = hardware_indirect_gi_resolution_sanitize(
      inst_.scene->eevee.hardware_raytracing_indirect_gi_resolution);
  const int resolution_scale = requested_resolution_scale;
  const int2 main_extent = inst_.film.render_extent_get();
  const int main_max_extent = max_ii(main_extent.x, main_extent.y);
  const int face_extent = max_ii(1, (main_max_extent + resolution_scale - 1) / resolution_scale);
  const int2 extent(face_extent);
  hardware_indirect_gi_cache_resolution_ = face_extent;
  hardware_indirect_gi_cache_dispatch_size_ = int3(
      math::divide_ceil(extent, int2(RAYTRACE_GROUP_SIZE)), 1);

  const eGPUTextureUsage cache_usage = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;
  hardware_indirect_gi_radiance_cache_tx_.ensure_2d_array(
      gpu::TextureFormat::SFLOAT_16_16_16_16, extent, 6, cache_usage);
  hardware_indirect_gi_position_cache_tx_.ensure_2d_array(
      gpu::TextureFormat::SFLOAT_16_16_16_16, extent, 6, cache_usage);
  hardware_indirect_gi_normal_cache_tx_.ensure_2d_array(
      gpu::TextureFormat::SFLOAT_16_16_16_16, extent, 6, cache_usage);

  RenderBuffers &rbufs = inst_.render_buffers;
  GBuffer &gbuf = inst_.gbuffer;
  rbufs.acquire(extent);
  gbuf.acquire(extent,
               inst_.pipelines.deferred.header_layer_count(),
               inst_.pipelines.deferred.closure_layer_count(),
               inst_.pipelines.deferred.normal_layer_count());

  hardware_indirect_gi_combined_fb_.ensure(GPU_ATTACHMENT_TEXTURE(rbufs.depth_tx),
                                           GPU_ATTACHMENT_TEXTURE(rbufs.combined_tx));

  const bool with_raycast = inst_.pipelines.has_raycast;
  hardware_indirect_gi_prepass_fb_.ensure(
      GPU_ATTACHMENT_TEXTURE(rbufs.depth_tx),
      with_raycast ? GPU_ATTACHMENT_TEXTURE(rbufs.prepass_normal_tx) : GPU_ATTACHMENT_NONE,
      with_raycast ? GPU_ATTACHMENT_TEXTURE(rbufs.object_id_tx) : GPU_ATTACHMENT_NONE,
      GPU_ATTACHMENT_TEXTURE(rbufs.vector_tx));
  hardware_indirect_gi_gbuffer_fb_.ensure(
      GPU_ATTACHMENT_TEXTURE(rbufs.depth_tx),
      GPU_ATTACHMENT_TEXTURE(rbufs.combined_tx),
      GPU_ATTACHMENT_TEXTURE_LAYER(gbuf.header_tx.layer_view(0), 0),
      GPU_ATTACHMENT_TEXTURE_LAYER(gbuf.normal_tx.layer_view(0), 0),
      GPU_ATTACHMENT_TEXTURE_LAYER(gbuf.closure_tx.layer_view(0), 0),
      GPU_ATTACHMENT_TEXTURE_LAYER(gbuf.closure_tx.layer_view(1), 0));

  View cache_view = {"Trace.HardwareIndirectGICacheView"};
  const CameraData &camera_data = inst_.camera.data_get();
  float4x4 winmat;
  cubeface_winmat_get(winmat, camera_data.clip_near, camera_data.clip_far);
  if (viewport_history_reset_) {
    for (int face : IndexRange(6)) {
      raytrace_history_invalidate_on_viewport_reset(hardware_indirect_gi_cache_rt_buffer_[face]);
      raytrace_history_invalidate_on_viewport_reset(
          hardware_indirect_gi_cache_refract_rt_buffer_[face]);
    }
  }

  const RayPipelineType previous_ray_type = inst_.pipelines.data.ray_type;
  const bool previous_hardware_lighting_use_hardware_rt_shadows =
      hardware_lighting_use_hardware_rt_shadows_;
  const bool previous_hardware_lighting_use_hardware_rt_environment_visibility =
      hardware_lighting_use_hardware_rt_environment_visibility_;
  const int previous_hardware_debug_isolate_mode = hardware_debug_isolate_mode_;
  if (inst_.pipelines.data.ray_type != RAY_TYPE_DIFFUSE) {
    inst_.pipelines.data.ray_type = RAY_TYPE_DIFFUSE;
    inst_.uniform_data.push_update();
  }
  hardware_lighting_use_hardware_rt_shadows_ = false;
  hardware_lighting_use_hardware_rt_environment_visibility_ = false;
  hardware_debug_isolate_mode_ = HWRT_DEBUG_ISOLATE_INDIRECT;
  hardware_indirect_gi_cache_rendering_ = true;
  for (int face : IndexRange(6)) {
    float4x4 viewmat = cubeface_mat(face);
    viewmat = math::translate(viewmat, -main_view.location());
    cache_view.sync(viewmat, winmat);

    inst_.lights.set_view(cache_view, extent);

    GPU_framebuffer_bind(hardware_indirect_gi_combined_fb_);
    GPU_framebuffer_clear_color_depth(
        hardware_indirect_gi_combined_fb_, {0.0, 0.0, 0.0, 1.0}, inst_.film.depth.clear_value);
    float4 clear_velocity = float4(0.0f);
    GPU_texture_clear(rbufs.vector_tx, GPU_DATA_FLOAT, &clear_velocity);
    if (with_raycast) {
      rbufs.object_id_tx.clear(uint4(0));
      rbufs.prepass_normal_tx.clear(float4(0.0f));
    }

    inst_.hiz_buffer.set_source(&rbufs.depth_tx);
    inst_.pipelines.deferred.render(cache_view,
                                    cache_view,
                                    hardware_indirect_gi_prepass_fb_,
                                    hardware_indirect_gi_combined_fb_,
                                    hardware_indirect_gi_gbuffer_fb_,
                                    extent,
                                    hardware_indirect_gi_cache_rt_buffer_[face],
                                    hardware_indirect_gi_cache_refract_rt_buffer_[face]);

    renderbuf_depth_view_ = rbufs.depth_tx;
    hardware_indirect_gi_cache_face_index_ = face;
    inst_.manager->submit(hardware_indirect_gi_cache_store_ps_, cache_view);
    if (inst_.is_viewport()) {
      /* Each face is a complete low-res diffuse combine capture. Drain between viewport faces to
       * bound Metal command-buffer pressure without dropping any cubemap directions. */
      GPU_finish();
    }
  }
  hardware_indirect_gi_cache_rendering_ = false;
  if (inst_.pipelines.data.ray_type != previous_ray_type) {
    inst_.pipelines.data.ray_type = previous_ray_type;
    inst_.uniform_data.push_update();
  }
  hardware_lighting_use_hardware_rt_shadows_ = previous_hardware_lighting_use_hardware_rt_shadows;
  hardware_lighting_use_hardware_rt_environment_visibility_ =
      previous_hardware_lighting_use_hardware_rt_environment_visibility;
  hardware_debug_isolate_mode_ = previous_hardware_debug_isolate_mode;

  gbuf.release();
  rbufs.release();
  hardware_indirect_gi_cache_valid_ = true;
}

void RayTraceModule::render_scene_final_specular(RayTraceBuffer &rt_buffer,
                                                 gpu::Texture *scene_radiance_tx,
                                                 eClosureBits active_closures,
                                                 View &main_view,
                                                 View &render_view)
{
  const bool perf_logging_enabled = hardware_perf_logging_enabled();
  const double perf_start_time = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  if (!use_hardware_tracing()) {
    return;
  }

  const uint32_t feature_mask = filtered_hardware_feature_mask(*this, active_closures) &
                                (RAYTRACE_EEVEE_HARDWARE_REFLECTIONS |
                                 RAYTRACE_EEVEE_HARDWARE_REFRACTIONS |
                                 RAYTRACE_EEVEE_HARDWARE_ENVIRONMENT);
  if ((feature_mask & (RAYTRACE_EEVEE_HARDWARE_REFLECTIONS |
                       RAYTRACE_EEVEE_HARDWARE_REFRACTIONS)) == 0)
  {
    return;
  }

  gpu::Texture *depth_tx = inst_.render_buffers.depth_tx;
  const int2 extent = inst_.render_buffers.extent_get();
  render_environment_visibility(render_view, depth_tx, inst_.gbuffer.normal_tx, extent);
  render_directional_shadow_visibility(render_view, depth_tx, inst_.gbuffer.normal_tx, extent);

  RayTraceResult result = render_phase(rt_buffer,
                                       scene_radiance_tx,
                                       (active_closures & CLOSURE_REFRACTION) ? scene_radiance_tx :
                                                                                nullptr,
                                       active_closures,
                                       main_view,
                                       render_view,
                                       HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR,
                                       feature_mask,
                                       false);

  scene_final_specular_resolve_ps_.init();
  scene_final_specular_resolve_ps_.shader_set(
      inst_.shaders.static_shader_get(RAY_TRACE_SCENE_FINAL_SPECULAR_RESOLVE));
  scene_final_specular_resolve_ps_.bind_texture("indirect_radiance_1_tx", &result.closures[0]);
  scene_final_specular_resolve_ps_.bind_texture("indirect_radiance_2_tx", &result.closures[1]);
  scene_final_specular_resolve_ps_.bind_texture("indirect_radiance_3_tx", &result.closures[2]);
  scene_final_specular_resolve_ps_.bind_image("combined_img", &inst_.render_buffers.combined_tx);
  scene_final_specular_resolve_ps_.bind_image("rp_color_img", &inst_.render_buffers.rp_color_tx);
  scene_final_specular_resolve_ps_.bind_resources(inst_.gbuffer);
  scene_final_specular_resolve_ps_.bind_resources(inst_.uniform_data);
  scene_final_specular_resolve_ps_.dispatch(
      int3(math::divide_ceil(extent, int2(RAYTRACE_GROUP_SIZE)), 1));
  scene_final_specular_resolve_ps_.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS | GPU_BARRIER_TEXTURE_FETCH);
  inst_.manager->submit(scene_final_specular_resolve_ps_, render_view);

  result.release();
  if (perf_logging_enabled) {
    const double elapsed_ms = (BLI_time_now_seconds() - perf_start_time) * 1000.0;
    std::fprintf(stderr,
                 "EEVEE HWRT perf scene_final_specular features=0x%x elapsed_ms=%.2f\n",
                 unsigned(feature_mask),
                 elapsed_ms);
  }
}

RayTraceResultTexture RayTraceModule::trace(
    int closure_index,
    bool active_layer,
    RaytraceEEVEE options,
    RayTraceBuffer &rt_buffer,
    /* TODO(fclem): Maybe wrap these two in some other class. */
    View &main_view,
    View &render_view)
{
  RayTraceBuffer::DenoiseBuffer *denoise_buf = &rt_buffer.closures[closure_index];

  if (!active_layer) {
    /* Early out. Release persistent buffers. Still acquire one dummy resource for validation. */
    denoise_buf->denoised_spatial_tx.acquire(int2(1),
                                             gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT);
    denoise_buf->radiance_history_tx.release();
    denoise_buf->variance_history_tx.release();
    denoise_buf->screen_ownership_history_tx.release();
    denoise_buf->valid_screen_ownership_history = false;
    denoise_buf->tilemask_history_tx.free();
    return {denoise_buf->denoised_spatial_tx};
  }

  const int resolution_scale = hardware_interactive_resolution_scale(
      inst_,
      current_hardware_feature_mask_,
      effective_hardware_resolution_scale(current_hardware_feature_mask_,
                                          options.resolution_scale,
                                          hardware_reflection_mode_,
                                          hardware_refraction_mode_));

  const int2 extent = inst_.film.render_extent_get();
  const int2 tracing_res = math::divide_ceil(extent, int2(resolution_scale));

  renderbuf_depth_view_ = inst_.render_buffers.depth_tx;

  const bool scene_final_specular_phase = (data_.hardware_trace_phase ==
                                           int(HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR));
  const bool use_denoise = (options.flag & RAYTRACE_EEVEE_USE_DENOISE);
  const bool allow_scene_final_spatial_denoise = !scene_final_specular_phase || (resolution_scale > 1);
  const bool use_spatial_denoise = allow_scene_final_spatial_denoise &&
                                   (options.denoise_stages & RAYTRACE_EEVEE_DENOISE_SPATIAL) &&
                                   use_denoise;
  const bool use_temporal_denoise = !scene_final_specular_phase &&
                                    (options.denoise_stages & RAYTRACE_EEVEE_DENOISE_TEMPORAL) &&
                                    use_spatial_denoise;
  const bool use_bilateral_denoise = !scene_final_specular_phase &&
                                     (options.denoise_stages &
                                      RAYTRACE_EEVEE_DENOISE_BILATERAL) &&
                                     use_temporal_denoise;

  eGPUTextureUsage usage_rw = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;

  GPU_debug_group_begin("Raytracing");

  data_.thickness = options.screen_trace_thickness;
  data_.quality = 1.0f - 0.95f * options.screen_trace_quality;

  float roughness_mask_start = options.trace_max_roughness;
  float roughness_mask_fade = 0.2f;
  if ((current_hardware_feature_mask_ & (RAYTRACE_EEVEE_HARDWARE_REFLECTIONS |
                                         RAYTRACE_EEVEE_HARDWARE_REFRACTIONS)) != 0)
  {
    roughness_mask_fade = 0.5f;
  }
  data_.roughness_mask_scale = 1.0 / roughness_mask_fade;
  data_.roughness_mask_bias = data_.roughness_mask_scale * roughness_mask_start;

  data_.resolution_scale = resolution_scale;
  data_.resolution_bias = int2(inst_.sampling.rng_2d_get(SAMPLING_RAYTRACE_V) * resolution_scale);
  data_.denoise_history_persmat = denoise_buf->history_persmat;
  data_.radiance_persmat = scene_final_specular_phase ? main_view.persmat() : render_view.persmat();
  data_.full_resolution = extent;
  data_.full_resolution_inv = 1.0f / float2(extent);
  data_.skip_denoise = !use_spatial_denoise;
  data_.closure_index = closure_index;
  data_.hardware_gi_bounces = hardware_gi_fixed_bounces;
  data_.hardware_gi_mode = int(hardware_gi_mode_);
  data_.hardware_reflection_bounces = effective_hardware_specular_bounces(
      inst_.scene->eevee.ray_tracing_reflection_bounces, hardware_reflection_mode_);
  data_.hardware_refraction_bounces = effective_hardware_specular_bounces(
      inst_.scene->eevee.ray_tracing_refraction_bounces, hardware_refraction_mode_);
  data_.hardware_caustics_samples = max_ii(1, inst_.scene->eevee.ray_tracing_caustics_samples);
  data_.hardware_reflection_mode = int(hardware_reflection_mode_);
  data_.hardware_refraction_mode = int(hardware_refraction_mode_);
  data_.use_hardware_fast_gi = use_hardware_fast_gi();
  data_.use_hardware_fast_gi_field = has_hardware_fast_gi_field();
  data_.use_hardware_caustics = use_hardware_caustics();
  data_.use_hardware_ign_sampling = use_hardware_rt_gi();
  data_.hardware_feature_mask = current_hardware_feature_mask_;
  data_.use_hardware_tracing_method = use_hardware_tracing_method();
  inst_.uniform_data.push_update();

  if (denoise_buf->screen_ownership_history_tx.acquire(
          tracing_res, gpu::TextureFormat::RAYTRACE_VARIANCE_FORMAT, usage_rw) ||
      denoise_buf->valid_screen_ownership_history == false)
  {
    denoise_buf->screen_ownership_history_tx.clear(float4(0.0f));
  }
  screen_ownership_history_tx_ = denoise_buf->screen_ownership_history_tx;
  use_screen_ownership_history_ = denoise_buf->valid_screen_ownership_history;

  /* Ray setup. */
  raytrace_tracing_dispatch_buf_.clear_to_zero();
  raytrace_denoise_dispatch_buf_.clear_to_zero();
  inst_.manager->submit(tile_compact_ps_);

  {
    /* Tracing rays. */
    ray_data_tx_.acquire(tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    ray_time_tx_.acquire(tracing_res, gpu::TextureFormat::RAYTRACE_RAYTIME_FORMAT, usage_rw);
    ray_radiance_tx_.acquire(tracing_res, gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, usage_rw);
    screen_continuation_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    screen_ownership_tx_.acquire(tracing_res, gpu::TextureFormat::RAYTRACE_VARIANCE_FORMAT, usage_rw);
    hit_albedo_tx_.acquire(tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    hit_throughput_tx_.acquire(tracing_res,
                               gpu::TextureFormat::SFLOAT_16_16_16_16,
                               usage_rw);
    hit_material_tx_.acquire(tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    hit_normal_tx_.acquire(tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    hit_position_tx_.acquire(tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    hit_world_position_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    hit_identity_tx_.acquire(tracing_res, gpu::TextureFormat::UINT_32_32_32_32, usage_rw);
    hit_barycentric_tx_.acquire(tracing_res,
                                gpu::TextureFormat::SFLOAT_16_16_16_16,
                                usage_rw);
    layered_receiver_ray_time_tx_.acquire(
        tracing_res, gpu::TextureFormat::RAYTRACE_RAYTIME_FORMAT, usage_rw);
    layered_receiver_ray_radiance_tx_.acquire(
        tracing_res, gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, usage_rw);
    layered_receiver_albedo_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    layered_receiver_throughput_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    layered_receiver_material_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    layered_receiver_normal_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    layered_receiver_position_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    layered_receiver_world_position_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    layered_receiver_identity_tx_.acquire(
        tracing_res, gpu::TextureFormat::UINT_32_32_32_32, usage_rw);
    layered_receiver_barycentric_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    hardware_reflected_receiver_gi_tx_.ensure_2d(
        gpu::TextureFormat::SFLOAT_16_16_16_16, tracing_res, usage_rw);
    hardware_reflected_receiver_gi_blur_tx_.ensure_2d(
        gpu::TextureFormat::SFLOAT_16_16_16_16, tracing_res, usage_rw);
    transmission_receiver_ray_time_tx_.acquire(
        tracing_res, gpu::TextureFormat::RAYTRACE_RAYTIME_FORMAT, usage_rw);
    transmission_receiver_ray_radiance_tx_.acquire(
        tracing_res, gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, usage_rw);
    transmission_receiver_albedo_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    transmission_receiver_throughput_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    transmission_receiver_material_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    transmission_receiver_normal_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    transmission_receiver_position_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    transmission_receiver_world_position_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    transmission_receiver_identity_tx_.acquire(
        tracing_res, gpu::TextureFormat::UINT_32_32_32_32, usage_rw);
    transmission_receiver_barycentric_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    screen_continuation_tx_.clear(float4(0.0f));
    screen_ownership_tx_.clear(float4(0.0f));

    inst_.manager->submit(generate_ps_, render_view);
    submit_tracing_backend(render_view);
  }

  RayTraceResultTexture result;

  /* Spatial denoise pass is required to resolve at least one ray per pixel. */
  {
    denoise_buf->denoised_spatial_tx.acquire(
        extent, gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, usage_rw);
    hit_variance_tx_.acquire(use_temporal_denoise ? extent : int2(1),
                             gpu::TextureFormat::RAYTRACE_VARIANCE_FORMAT);
    hit_depth_tx_.acquire(use_temporal_denoise ? extent : int2(1), gpu::TextureFormat::SFLOAT_32);
    denoised_spatial_tx_ = denoise_buf->denoised_spatial_tx;

    inst_.manager->submit(denoise_spatial_ps_, render_view);

    result = {denoise_buf->denoised_spatial_tx};
  }

  ray_data_tx_.release();
  ray_time_tx_.release();
  ray_radiance_tx_.release();
  screen_continuation_tx_.release();
  hit_material_tx_.release();
  hit_normal_tx_.release();
  hit_position_tx_.release();
  hit_world_position_tx_.release();
  hit_identity_tx_.release();
  hit_barycentric_tx_.release();
  layered_receiver_ray_time_tx_.release();
  layered_receiver_ray_radiance_tx_.release();
  layered_receiver_albedo_tx_.release();
  layered_receiver_throughput_tx_.release();
  layered_receiver_material_tx_.release();
  layered_receiver_normal_tx_.release();
  layered_receiver_position_tx_.release();
  layered_receiver_world_position_tx_.release();
  layered_receiver_identity_tx_.release();
  layered_receiver_barycentric_tx_.release();
  transmission_receiver_ray_time_tx_.release();
  transmission_receiver_ray_radiance_tx_.release();
  transmission_receiver_albedo_tx_.release();
  transmission_receiver_throughput_tx_.release();
  transmission_receiver_material_tx_.release();
  transmission_receiver_normal_tx_.release();
  transmission_receiver_position_tx_.release();
  transmission_receiver_world_position_tx_.release();
  transmission_receiver_identity_tx_.release();
  transmission_receiver_barycentric_tx_.release();

  if (use_temporal_denoise) {
    denoise_buf->denoised_temporal_tx.acquire(
        extent, gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, usage_rw);
    denoise_variance_tx_.acquire(use_bilateral_denoise ? extent : int2(1),
                                 gpu::TextureFormat::RAYTRACE_VARIANCE_FORMAT,
                                 usage_rw);
    denoise_buf->variance_history_tx.acquire(use_bilateral_denoise ? extent : int2(1),
                                             gpu::TextureFormat::RAYTRACE_VARIANCE_FORMAT,
                                             usage_rw);
    denoise_buf->tilemask_history_tx.ensure_2d_array(gpu::TextureFormat::RAYTRACE_TILEMASK_FORMAT,
                                                     tile_raytrace_denoise_tx_.size().xy(),
                                                     tile_raytrace_denoise_tx_.size().z,
                                                     usage_rw);

    if (denoise_buf->radiance_history_tx.acquire(
            extent, gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, usage_rw) ||
        denoise_buf->valid_history == false)
    {
      /* If viewport resolution changes, do not try to use history. */
      denoise_buf->tilemask_history_tx.clear(uint4(0u));
    }
    radiance_history_tx_ = denoise_buf->radiance_history_tx;
    variance_history_tx_ = denoise_buf->variance_history_tx;
    tilemask_history_tx_ = denoise_buf->tilemask_history_tx;
    denoised_temporal_tx_ = denoise_buf->denoised_temporal_tx;

    inst_.manager->submit(denoise_temporal_ps_, render_view);

    /* Radiance will be swapped with history in #RayTraceResult::release().
     * Variance is swapped with history after bilateral denoise.
     * It keeps data-flow easier to follow. */
    result = {denoise_buf->denoised_temporal_tx, denoise_buf->radiance_history_tx};
    /* Not referenced by result anymore. */
    denoise_buf->denoised_spatial_tx.release();

    GPU_texture_copy(denoise_buf->tilemask_history_tx, tile_raytrace_denoise_tx_);
  }

  /* Only use history buffer for the next frame if temporal denoise was used by the current one. */
  denoise_buf->valid_history = use_temporal_denoise;
  denoise_buf->valid_screen_ownership_history = use_hardware_hybrid_retrace_;
  if (use_temporal_denoise || use_hardware_hybrid_retrace_) {
    /* Radiance and Hybrid ownership reproject from the same primary view. */
    denoise_buf->history_persmat = main_view.persmat();
  }

  hit_variance_tx_.release();
  hit_depth_tx_.release();

  if (use_bilateral_denoise) {
    denoise_buf->denoised_bilateral_tx.acquire(
        extent, gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, usage_rw);
    denoised_bilateral_tx_ = denoise_buf->denoised_bilateral_tx;

    inst_.manager->submit(denoise_bilateral_ps_, render_view);

    /* Swap after last use, retain history buffers until next cycle. */
    TextureFromPool::swap(denoise_buf->denoised_temporal_tx, denoise_buf->radiance_history_tx);
    TextureFromPool::swap(denoise_variance_tx_, denoise_buf->variance_history_tx);
    denoise_buf->radiance_history_tx.retain();
    denoise_buf->variance_history_tx.retain();

    result = {denoise_buf->denoised_bilateral_tx};
    /* Not referenced by result anymore. */
    denoise_buf->denoised_temporal_tx.release();
  }
  else if (use_temporal_denoise) {
    /* Not referenced by result anymore. */
    denoise_buf->variance_history_tx.retain();
  }

  if (use_hardware_hybrid_retrace_) {
    TextureFromPool::swap(screen_ownership_tx_, denoise_buf->screen_ownership_history_tx);
    denoise_buf->screen_ownership_history_tx.retain();
  }
  else {
    denoise_buf->screen_ownership_history_tx.release();
  }
  screen_ownership_tx_.release();
  screen_ownership_history_tx_ = nullptr;
  use_screen_ownership_history_ = false;
  hit_albedo_tx_.release();
  hit_throughput_tx_.release();

  denoise_variance_tx_.release();

  GPU_debug_group_end();

  return result;
}

RayTraceResult RayTraceModule::alloc_only(RayTraceBuffer &rt_buffer)
{
  const int2 extent = inst_.film.render_extent_get();
  eGPUTextureUsage usage_rw = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;

  RayTraceResult result;
  for (int i = 0; i < 3; i++) {
    RayTraceBuffer::DenoiseBuffer *denoise_buf = &rt_buffer.closures[i];
    denoise_buf->denoised_bilateral_tx.acquire(
        extent, gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, usage_rw);
    result.closures[i] = {denoise_buf->denoised_bilateral_tx};
  }
  return result;
}

RayTraceResult RayTraceModule::alloc_dummy(RayTraceBuffer &rt_buffer)
{
  eGPUTextureUsage usage_rw = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;

  RayTraceResult result;
  for (int i = 0; i < 3; i++) {
    RayTraceBuffer::DenoiseBuffer *denoise_buf = &rt_buffer.closures[i];
    denoise_buf->denoised_bilateral_tx.acquire(
        int2(1), gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, usage_rw);
    result.closures[i] = {denoise_buf->denoised_bilateral_tx};
  }
  return result;
}
/** \} */

}  // namespace blender::eevee
