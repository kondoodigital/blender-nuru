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
#include "GPU_nuru_hardware_raytrace.hh"
#include "GPU_storage_buffer.hh"
#include "GPU_state.hh"
#include "GPU_texture.hh"
#include "GPU_vertex_buffer.hh"

#include "DNA_ID.h"
#include "DNA_scene_types.h"

#include "gpu_shader_private.hh"

#include "BKE_scene.hh"

#include "DEG_depsgraph_query.hh"

#include "eevee_camera.hh"
#include "eevee_instance.hh"
#include "eevee_raytrace.hh"

namespace blender::eevee {

static RaytraceEEVEE_SpecularMode sanitize_specular_mode(const int value)
{
  return (value >= RAYTRACE_EEVEE_SPECULAR_MODE_OFF &&
          value <= RAYTRACE_EEVEE_SPECULAR_MODE_AUTO) ?
             RaytraceEEVEE_SpecularMode(value) :
             RAYTRACE_EEVEE_SPECULAR_MODE_OFF;
}

static bool instance_is_shader_preview(const Instance &inst)
{
  return BKE_scene_is_eevee_shader_preview(inst.scene);
}

static bool instance_is_material_preview(const Instance &inst)
{
  return inst.is_viewport() && inst.v3d != nullptr && inst.v3d->shading.type == OB_MATERIAL;
}

static RaytraceEEVEE_SpecularMode sanitize_reflection_mode(const int value)
{
  const RaytraceEEVEE_SpecularMode mode = sanitize_specular_mode(value);
  return (mode == RAYTRACE_EEVEE_SPECULAR_MODE_OFF) ? RAYTRACE_EEVEE_SPECULAR_MODE_OFF :
                                                      RAYTRACE_EEVEE_SPECULAR_MODE_FULL_RT;
}

static int hardware_debug_view_mode()
{
  const char *value = std::getenv("BLENDER_EEVEE_HWRT_DEBUG_VIEW_MODE");
  if (value == nullptr || value[0] == '\0' || (value[0] == '0' && value[1] == '\0')) {
    return HWRT_DEBUG_VIEW_NONE;
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

struct HardwareHitEvalBatchSupport {
  /** Geometry SSBOs (positions, normals, index buffer) the replay cannot run without. */
  bool geometry_compatible = false;
  /** Material-attribute SSBO slots the batch cannot fulfill. Replay still runs with these bound
   * to a dummy buffer and zeroed `gpu_attr_N`/`gpu_attr_N_meta` descriptors, making every fetch
   * return zero exactly like the raster pipeline's missing-attribute fallback. */
  uint16_t missing_attr_mask = 0;
};

static HardwareHitEvalBatchSupport hardware_hit_eval_batch_support(gpu::Batch *batch,
                                                                   gpu::Shader *shader)
{
  HardwareHitEvalBatchSupport support;
  if ((batch == nullptr) || (shader == nullptr) || (shader->interface == nullptr)) {
    return support;
  }

  const gpu::ShaderInterface *interface = shader->interface;
  if (interface->ssbo_attr_mask_ == 0) {
    support.geometry_compatible = true;
    return support;
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

  uint16_t missing_material_attributes = 0;
  const gpu::ShaderInput *ssbo_inputs = interface->inputs_ + interface->attr_len_ +
                                        interface->ubo_len_ + interface->uniform_len_;
  for (uint input_index = 0; input_index < interface->ssbo_len_; input_index++) {
    const gpu::ShaderInput &input = ssbo_inputs[input_index];
    if ((input.location < 0) || ((missing_attributes & (1 << input.location)) == 0)) {
      continue;
    }
    const char *input_name = interface->input_name_get(&input);
    if (batch_has_ssbo_attribute(batch, input_name)) {
      missing_attributes &= ~(1 << input.location);
      continue;
    }
    const bool is_required_geometry = STREQ(input_name, "pos") || STREQ(input_name, "nor") ||
                                      STREQ(input_name, "gpu_index_buf");
    if (!is_required_geometry) {
      /* Material node trees can reference attribute layers (UV maps, color attributes, tangents)
       * that the evaluated mesh does not carry. Direct-view rendering tolerates those by reading
       * zeros, so sparse replay must not drop the whole material (and all of its valid textures)
       * over them. Track the slot for a dummy binding instead. */
      missing_material_attributes |= uint16_t(1 << input.location);
      missing_attributes &= ~(1 << input.location);
    }
  }

  support.geometry_compatible = (missing_attributes == 0);
  support.missing_attr_mask = missing_material_attributes;
  return support;
}

static GPUHardwareRaytraceFastGILightRecord hardware_fast_gi_light_record_from_light(
    const LightData &light)
{
  GPUHardwareRaytraceFastGILightRecord record;
  record.object_to_world_x = light.object_to_world.x;
  record.object_to_world_y = light.object_to_world.y;
  record.object_to_world_z = light.object_to_world.z;
  record.color_diffuse_power = float4(
      std::abs(light.color.x),
      std::abs(light.color.y),
      std::abs(light.color.z),
      light.power[LIGHT_DIFFUSE]);
  record.direction_type = float4(0.0f, 0.0f, 1.0f, float(light.type));
  /* Nuru NIS: stable cluster id rides the spare spot lane for kernel-side multiplier lookup
   * (tree nodes carry their representative leaf's cluster in the same lane). */
  record.spot_size_inv.z = float(light.cluster_id);
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

/* ---------------------------------------------------------------------------------------------
 * Nuru light tree (many-light sampling, Stage A).
 *
 * CPU median-split BVH over the LOCAL lights packed into the Fast GI light record buffer,
 * appended to the same buffer as record-encoded nodes (see GPU_nuru_hardware_raytrace.hh for
 * the slot encoding). Kernels replace uniform light picking with stochastic descent weighted by
 * the classic cluster importance (power, distance, receiver facing, emitter cone); published
 * unpatented techniques only - no reservoirs, no spatiotemporal sample reuse.
 * --------------------------------------------------------------------------------------------- */

struct HardwareLightTreeBuildLight {
  float3 position = float3(0.0f);
  /* Scalar importance power: diffuse power scaled by the dominant color channel. */
  float power = 0.0f;
  /* Main emission axis (world -Z of the light) and emitter cone: cos_half_angle, with -2.0f as
   * the omnidirectional sentinel (point lights). */
  float3 axis = float3(0.0f, 0.0f, -1.0f);
  float cos_theta = -2.0f;
  float radius = 0.0f;
  int record_index = 0;
  /* Nuru NIS cluster id (LightData.cluster_id) for the kernel multiplier lookup. */
  int cluster_id = 0;
};

static HardwareLightTreeBuildLight hardware_light_tree_build_light_from_light(
    const LightData &light, const int record_index)
{
  HardwareLightTreeBuildLight build_light;
  build_light.position = float3(light.object_to_world.x.w,
                                light.object_to_world.y.w,
                                light.object_to_world.z.w);
  const float3 color_abs = math::abs(float3(light.color.x, light.color.y, light.color.z));
  build_light.power = std::max(light.power[LIGHT_DIFFUSE], 0.0f) *
                      std::max(math::reduce_max(color_abs), 1.0e-4f);
  build_light.record_index = record_index;
  build_light.radius = is_local_light(light.type) ? std::max(light.local().local.shape_radius, 0.0f) :
                                                    0.0f;
  if (is_spot_light(light.type) || is_area_light(light.type)) {
    /* World emission axis is the light's -Z. Treat spots/areas as hemisphere emitters for the
     * cluster cone: cheap, conservative, and correct for the descent floor. */
    const float3 z_axis = float3(light.object_to_world.x.z,
                                 light.object_to_world.y.z,
                                 light.object_to_world.z.z);
    const float len = math::length(z_axis);
    build_light.axis = (len > 1.0e-6f) ? (-z_axis / len) : float3(0.0f, 0.0f, -1.0f);
    build_light.cos_theta = 0.0f;
  }
  build_light.cluster_id = light.cluster_id;
  return build_light;
}

/* Aggregate two cluster orientation cones conservatively (axis blend, widened half-angle). */
static void hardware_light_tree_cone_union(const float3 &axis_a,
                                           const float cos_a,
                                           const float3 &axis_b,
                                           const float cos_b,
                                           float3 &r_axis,
                                           float &r_cos)
{
  if (cos_a <= -1.5f || cos_b <= -1.5f) {
    /* Any omnidirectional child makes the cluster omnidirectional. */
    r_axis = float3(0.0f, 0.0f, -1.0f);
    r_cos = -2.0f;
    return;
  }
  const float3 axis_sum = axis_a + axis_b;
  const float len = math::length(axis_sum);
  if (len < 1.0e-6f) {
    r_axis = float3(0.0f, 0.0f, -1.0f);
    r_cos = -2.0f;
    return;
  }
  r_axis = axis_sum / len;
  const float theta_a = std::acos(std::clamp(cos_a, -1.0f, 1.0f));
  const float theta_b = std::acos(std::clamp(cos_b, -1.0f, 1.0f));
  const float divergence_a = std::acos(std::clamp(math::dot(r_axis, axis_a), -1.0f, 1.0f));
  const float divergence_b = std::acos(std::clamp(math::dot(r_axis, axis_b), -1.0f, 1.0f));
  const float theta = std::max(theta_a + divergence_a, theta_b + divergence_b);
  if (theta >= float(M_PI) * 0.5f) {
    r_cos = -2.0f;
    return;
  }
  r_cos = std::cos(theta);
}

static int hardware_light_tree_build_recursive(
    MutableSpan<HardwareLightTreeBuildLight> lights,
    Vector<GPUHardwareRaytraceFastGILightRecord> &nodes)
{
  const int node_index = int(nodes.size());
  nodes.append(GPUHardwareRaytraceFastGILightRecord());
  if (lights.size() == 1) {
    const HardwareLightTreeBuildLight &light = lights[0];
    GPUHardwareRaytraceFastGILightRecord &node = nodes[node_index];
    node.object_to_world_x = float4(light.position, std::max(light.radius, 1.0e-3f));
    node.object_to_world_y = float4(light.axis, light.cos_theta);
    node.object_to_world_z = float4(light.power, -1.0f, -1.0f, float(light.record_index));
    node.spot_size_inv.z = float(light.cluster_id);
    return node_index;
  }
  /* Median split along the longest centroid-bounds axis. */
  float3 bounds_min = lights[0].position;
  float3 bounds_max = lights[0].position;
  for (const HardwareLightTreeBuildLight &light : lights) {
    bounds_min = math::min(bounds_min, light.position);
    bounds_max = math::max(bounds_max, light.position);
  }
  const float3 extent = bounds_max - bounds_min;
  const int split_axis = (extent.x >= extent.y && extent.x >= extent.z) ? 0 :
                         (extent.y >= extent.z)                         ? 1 :
                                                                          2;
  std::sort(lights.begin(),
            lights.end(),
            [split_axis](const HardwareLightTreeBuildLight &a,
                         const HardwareLightTreeBuildLight &b) {
              return a.position[split_axis] < b.position[split_axis];
            });
  const int64_t half = lights.size() / 2;
  const int left = hardware_light_tree_build_recursive(lights.slice(0, half), nodes);
  const int right = hardware_light_tree_build_recursive(
      lights.slice(half, lights.size() - half), nodes);
  GPUHardwareRaytraceFastGILightRecord &node = nodes[node_index];
  const float4 left_sphere = nodes[left].object_to_world_x;
  const float4 right_sphere = nodes[right].object_to_world_x;
  /* Bounding sphere of the two child spheres. */
  const float3 left_center = left_sphere.xyz();
  const float3 right_center = right_sphere.xyz();
  const float3 span = right_center - left_center;
  const float span_len = math::length(span);
  float3 center;
  float radius;
  if (span_len + right_sphere.w <= left_sphere.w) {
    center = left_center;
    radius = left_sphere.w;
  }
  else if (span_len + left_sphere.w <= right_sphere.w) {
    center = right_center;
    radius = right_sphere.w;
  }
  else {
    radius = 0.5f * (span_len + left_sphere.w + right_sphere.w);
    const float3 dir = (span_len > 1.0e-6f) ? span / span_len : float3(0.0f);
    center = left_center + dir * (radius - left_sphere.w);
  }
  float3 cone_axis;
  float cone_cos;
  hardware_light_tree_cone_union(nodes[left].object_to_world_y.xyz(),
                                 nodes[left].object_to_world_y.w,
                                 nodes[right].object_to_world_y.xyz(),
                                 nodes[right].object_to_world_y.w,
                                 cone_axis,
                                 cone_cos);
  node.object_to_world_x = float4(center, radius);
  node.object_to_world_y = float4(cone_axis, cone_cos);
  node.object_to_world_z = float4(nodes[left].object_to_world_z.x +
                                      nodes[right].object_to_world_z.x,
                                  float(left),
                                  float(right),
                                  -1.0f);
  /* Representative cluster for the NIS multiplier: the higher-power child's (sampling shaping
   * only; the descent PDF uses the same weighted importances, so any choice stays unbiased). */
  node.spot_size_inv.z = (nodes[left].object_to_world_z.x >= nodes[right].object_to_world_z.x) ?
                             nodes[left].spot_size_inv.z :
                             nodes[right].spot_size_inv.z;
  return node_index;
}

/* Build the light tree over the local lights and return the encoded node records. */
static Vector<GPUHardwareRaytraceFastGILightRecord> hardware_light_tree_build(
    const Span<LightData> local_lights, const int record_offset)
{
  Vector<GPUHardwareRaytraceFastGILightRecord> nodes;
  if (local_lights.is_empty()) {
    return nodes;
  }
  Vector<HardwareLightTreeBuildLight> build_lights;
  build_lights.reserve(local_lights.size());
  for (const int i : local_lights.index_range()) {
    build_lights.append(
        hardware_light_tree_build_light_from_light(local_lights[i], record_offset + i));
  }
  nodes.reserve(2 * build_lights.size() - 1);
  hardware_light_tree_build_recursive(build_lights.as_mutable_span(), nodes);
  return nodes;
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

static bool hardware_viewport_interactive(const Instance &inst)
{
  return inst.is_viewport() && (inst.sampling.interactive_mode() || inst.is_transforming ||
                                inst.is_navigating || inst.is_painting || inst.is_playback);
}

static bool hardware_uses_viewport_reference(const Instance &inst)
{
  return inst.is_viewport() || inst.is_image_render;
}

static float hardware_interactive_resolution_scale(const Instance & /*inst*/,
                                                   const uint32_t /*feature_mask*/,
                                                   const float resolution_scale)
{
  /* Honor the configured ray-tracing resolution percentage in every mode. */
  return resolution_scale;
}

static float raytrace_resolution_percentage_sanitize(const float value)
{
  if (!std::isfinite(value)) {
    return 50.0f;
  }
  switch (int(std::lround(value))) {
    case RAYTRACE_EEVEE_RESOLUTION_SCALE_100:
    case RAYTRACE_EEVEE_RESOLUTION_SCALE_75:
    case RAYTRACE_EEVEE_RESOLUTION_SCALE_50:
    case RAYTRACE_EEVEE_RESOLUTION_SCALE_25:
      return float(int(std::lround(value)));
    default:
      return 50.0f;
  }
}

/* Tracing resolution = `screen * denominator / numerator`. 75% needs numerator=4 / denominator=3
 * so the 1/N integer divisor model still works without going to floats. */
static int raytrace_resolution_scale_numerator(const float value)
{
  switch (int(raytrace_resolution_percentage_sanitize(value))) {
    case RAYTRACE_EEVEE_RESOLUTION_SCALE_100:
      return 1;
    case RAYTRACE_EEVEE_RESOLUTION_SCALE_75:
      return 4;
    case RAYTRACE_EEVEE_RESOLUTION_SCALE_25:
      return 4;
    case RAYTRACE_EEVEE_RESOLUTION_SCALE_50:
    default:
      return 2;
  }
}

static int raytrace_resolution_scale_denominator(const float value)
{
  switch (int(raytrace_resolution_percentage_sanitize(value))) {
    case RAYTRACE_EEVEE_RESOLUTION_SCALE_75:
      return 3;
    case RAYTRACE_EEVEE_RESOLUTION_SCALE_100:
    case RAYTRACE_EEVEE_RESOLUTION_SCALE_50:
    case RAYTRACE_EEVEE_RESOLUTION_SCALE_25:
    default:
      return 1;
  }
}

static int2 raytrace_tracing_resolution(int2 extent, const float resolution_scale)
{
  const int numerator = raytrace_resolution_scale_numerator(resolution_scale);
  const int denominator = raytrace_resolution_scale_denominator(resolution_scale);
  return math::divide_ceil(extent * denominator, int2(numerator));
}

static int hardware_gi_spatial_sample_count_sanitize(const int value)
{
  switch (value) {
    case RAYTRACE_EEVEE_GI_SPATIAL_8:
    case RAYTRACE_EEVEE_GI_SPATIAL_16:
    case RAYTRACE_EEVEE_GI_SPATIAL_32:
      return value;
    default:
      return RAYTRACE_EEVEE_GI_SPATIAL_8;
  }
}

static int hardware_shadow_sample_count_sanitize(const int value)
{
  return clamp_i(value, 1, 16);
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

struct HardwareFastGISceneScaleAnalysis {
  float3 scene_center = float3(0.0f);
  float scene_bounds_radius = 12.0f;
  float scene_radius = 12.0f;
  float forward_extent = 12.0f;
  float lateral_extent = 6.0f;
  float density = 0.0f;
  int active_entry_count = 0;
};

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

static int effective_hardware_specular_bounces(const int user_bounces,
                                               const RaytraceEEVEE_SpecularMode mode)
{
  const int clamped_user_bounces = clamp_i(user_bounces, 1, HWRT_SPECULAR_MAX_BOUNCES);
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
  free_hardware_rt_scene_cache();
}

/* -------------------------------------------------------------------- */
/** \name Raytracing
 *
 * \{ */

void RayTraceModule::apply_shader_preview_nuru_disable()
{
  shader_preview_disable_nuru_ = true;
  use_raytracing_ = false;
  tracing_method_ = RAYTRACE_EEVEE_METHOD_SCREEN;
  hardware_gi_mode_ = RAYTRACE_EEVEE_GI_MODE_OFF;
  hardware_gi_enabled_ = false;
  hardware_shadow_enabled_ = false;
  hardware_lighting_use_hardware_rt_shadows_ = false;
  hardware_reflection_mode_ = RAYTRACE_EEVEE_SPECULAR_MODE_OFF;
  hardware_refraction_mode_ = RAYTRACE_EEVEE_SPECULAR_MODE_OFF;
  hardware_environment_enabled_ = false;
  hardware_lighting_use_hardware_rt_environment_visibility_ = false;
  hardware_secondary_gi_enabled_ = false;
  ray_tracing_options_.flag &= ~RAYTRACE_EEVEE_USE_DENOISE;
  free_hardware_rt_scene_cache();
}

void RayTraceModule::init()
{
  const bool is_shader_preview = instance_is_shader_preview(inst_);
  shader_preview_disable_nuru_ = is_shader_preview || instance_is_material_preview(inst_);
  if (is_shader_preview) {
    BKE_scene_eevee_force_classic_raytracing(inst_.scene);
  }

  const SceneEEVEE &sce_eevee = inst_.scene->eevee;

  ray_tracing_options_ = sce_eevee.ray_tracing_options;
  /* Nuru: OIDN is mandatory for HWRT indirect; scene DNA may still carry user toggles. */
  if (!shader_preview_disable_nuru_) {
    ray_tracing_options_.flag |= RAYTRACE_EEVEE_USE_DENOISE;
    ray_tracing_options_.denoise_filter = RAYTRACE_EEVEE_DENOISE_FILTER_OIDN;
  }
  if ((sce_eevee.flag & SCE_EEVEE_FAST_GI_ENABLED) == 0) {
    ray_tracing_options_.trace_max_roughness = 1.0f;
  }
  /* Always initialize thickness, for the ray-cast node. */
  data_.thickness = ray_tracing_options_.screen_trace_thickness;

  use_raytracing_ = (sce_eevee.flag & SCE_EEVEE_SSR_ENABLED) != 0;
  tracing_method_ = RaytraceEEVEE_Method(sce_eevee.ray_tracing_method);
  /* Nuru final Combined renders must match the Rendered viewport reference. Keep HWRT enabled
   * for image renders; only non-rendered viewport pass inspection falls back to classic paths. */
  if (tracing_method_ == RAYTRACE_EEVEE_METHOD_HARDWARE &&
      !inst_.is_rendered_viewport() && !inst_.is_image_render)
  {
    tracing_method_ = RAYTRACE_EEVEE_METHOD_SCREEN;
  }
  if (tracing_method_ == RAYTRACE_EEVEE_METHOD_HARDWARE && inst_.is_viewport()) {
    const eViewLayerEEVEEPassType viewport_pass = eViewLayerEEVEEPassType(
        inst_.v3d->shading.render_pass);
    if (!ELEM(viewport_pass,
              EEVEE_RENDER_PASS_COMBINED,
              EEVEE_RENDER_PASS_DIFFUSE_LIGHT,
              EEVEE_RENDER_PASS_SPECULAR_LIGHT,
              EEVEE_RENDER_PASS_SHADOW))
    {
      tracing_method_ = RAYTRACE_EEVEE_METHOD_SCREEN;
    }
  }
  hardware_gi_mode_ = RaytraceEEVEE_GIMode(sce_eevee.hardware_raytracing_gi_mode);
  if (!ELEM(hardware_gi_mode_, RAYTRACE_EEVEE_GI_MODE_ACCURATE, RAYTRACE_EEVEE_GI_MODE_OFF)) {
    hardware_gi_mode_ = RAYTRACE_EEVEE_GI_MODE_ACCURATE;
  }
  hardware_gi_enabled_ = use_hardware_tracing() && (hardware_gi_mode_ == RAYTRACE_EEVEE_GI_MODE_ACCURATE);
  hardware_shadow_enabled_ = use_hardware_tracing() &&
                             (sce_eevee.hardware_raytracing_features &
                              RAYTRACE_EEVEE_HARDWARE_SHADOWS);
  hardware_lighting_use_hardware_rt_shadows_ = hardware_shadow_enabled_;
  hardware_shadow_color_intensity_ = std::clamp(
      sce_eevee.hardware_raytracing_shadow_color_intensity, 0.0f, 1.0f);
  hardware_shadow_transparency_ = std::clamp(
      sce_eevee.hardware_raytracing_shadow_transparency, 0.0f, 1.0f);
  hardware_reflection_mode_ = use_hardware_tracing() ?
                                  sanitize_reflection_mode(
                                      sce_eevee.hardware_raytracing_reflection_mode) :
                                  RAYTRACE_EEVEE_SPECULAR_MODE_OFF;
  hardware_refraction_mode_ = use_hardware_tracing() ?
                                  sanitize_specular_mode(
                                      sce_eevee.hardware_raytracing_refraction_mode) :
                                  RAYTRACE_EEVEE_SPECULAR_MODE_OFF;
  hardware_environment_enabled_ = use_hardware_tracing() &&
                                  (hardware_reflection_mode_ != RAYTRACE_EEVEE_SPECULAR_MODE_OFF ||
                                   hardware_refraction_mode_ != RAYTRACE_EEVEE_SPECULAR_MODE_OFF);
  hardware_lighting_use_hardware_rt_environment_visibility_ = hardware_environment_enabled_;
  /* Secondary GI (per-pixel receiver dome gather) is fully deactivated: in production interiors
   * the 2-4 sample gather contributes mostly variance artifacts in mirrors, not usable bounce
   * light. The kernels and host plumbing stay parked for a future higher-sample/NIS-guided
   * revival; the quick-settings toggle is removed with it. NIS many-light sampling stays active
   * on the direct-light and gather NEE paths independently of this gate. */
  hardware_secondary_gi_enabled_ = false;
  fast_gi_ray_count_ = sce_eevee.fast_gi_ray_count;
  fast_gi_step_count_ = sce_eevee.fast_gi_step_count;
  fast_gi_ao_only_ = (sce_eevee.fast_gi_method == FAST_GI_AO_ONLY);
  hardware_shadow_sample_count_ = hardware_shadow_sample_count_sanitize(
      sce_eevee.hardware_raytracing_shadow_samples);
  if (!use_hardware_tracing() || active_hardware_feature_mask() == 0) {
    free_hardware_rt_scene_cache();
  }

  if (shader_preview_disable_nuru_) {
    apply_shader_preview_nuru_disable();
  }

  float4 data(0.0f);
  radiance_dummy_black_tx_.ensure_2d(
      gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, int2(1), GPU_TEXTURE_USAGE_SHADER_READ, data);
  /* Nuru: HWRT shadow visibility carries RGB attenuation so colored/transparent shadows through
   * refractive materials (e.g. glass) tint the lit area per channel instead of just scaling its
   * brightness uniformly. Fully-lit init is `(1,1,1,1)`. */
  const float visibility[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  const float environment_visibility[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  hardware_shadow_visibility_tx_.ensure_2d_array(
      gpu::TextureFormat::SFLOAT_16_16_16_16, int2(1), 1, GPU_TEXTURE_USAGE_SHADER_READ, visibility);
  hardware_secondary_shadow_visibility_tx_.ensure_2d_array(
      gpu::TextureFormat::SFLOAT_16_16_16_16, int2(1), 1, GPU_TEXTURE_USAGE_SHADER_READ, visibility);
  hardware_layered_receiver_shadow_visibility_tx_.ensure_2d_array(
      gpu::TextureFormat::SFLOAT_16_16_16_16, int2(1), 1, GPU_TEXTURE_USAGE_SHADER_READ, visibility);
  hardware_transmission_receiver_shadow_visibility_tx_.ensure_2d_array(
      gpu::TextureFormat::SFLOAT_16_16_16_16, int2(1), 1, GPU_TEXTURE_USAGE_SHADER_READ, visibility);
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
    hardware_scene_instance_count_ += int(entry.resource_handle.index_range().size());
  }
}

static Vector<GPUHardwareRaytraceSceneEntry> build_hardware_rt_scene_entries(
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

  Vector<GPUHardwareRaytraceSceneEntry> rt_scene_entries;
  rt_scene_entries.reserve(scene_entries.size());

  uint32_t user_id = 0;
  int emissive_entry_count = 0;
  float emissive_peak = 0.0f;
  for (const HardwareRaytraceSceneEntry &entry : scene_entries) {
    if (entry.batch == nullptr) {
      continue;
    }

    GPUHardwareRaytraceSceneEntry rt_entry;
    rt_entry.batch = entry.batch;
    rt_entry.object_to_world = entry.object_to_world;
    rt_entry.instance_count = std::max(1u, entry.instance_count);
    rt_entry.user_id = user_id++;
    rt_entry.emissive_radiance = entry.emissive_radiance;
    rt_entry.diffuse_albedo = entry.diffuse_albedo;
    rt_entry.reflection_color = entry.reflection_color;
    rt_entry.reflection_roughness = entry.reflection_roughness;
    rt_entry.transmission_color = entry.transmission_color;
    rt_entry.transmission_roughness = entry.transmission_roughness;
    rt_entry.reflection_ior = entry.reflection_ior;
    rt_entry.refraction_ior = entry.refraction_ior;
    rt_entry.packed_thickness = entry.packed_thickness;
    rt_entry.alpha = entry.alpha;
    rt_entry.reflection_layer_coverage = entry.reflection_layer_coverage;
    rt_entry.closure_type = entry.closure_type;
    rt_entry.proxy_flags = entry.proxy_flags;
    emissive_peak = std::max(emissive_peak, math::reduce_max(rt_entry.emissive_radiance));
    if (math::reduce_max(rt_entry.emissive_radiance) > 0.0f) {
      emissive_entry_count++;
    }
    rt_entry.material_slot = entry.material_slot;
    rt_entry.is_sculpt = entry.is_sculpt;
    rt_scene_entries.append(rt_entry);
  }

  if (r_emissive_entry_count != nullptr) {
    *r_emissive_entry_count = emissive_entry_count;
  }
  if (r_emissive_peak != nullptr) {
    *r_emissive_peak = emissive_peak;
  }

  return rt_scene_entries;
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

static GPUHardwareRaytraceScene *build_hardware_rt_scene(
    Span<HardwareRaytraceSceneEntry> scene_entries,
                                                         GPUHardwareRaytraceSceneStats *r_stats,
                                                         int *r_emissive_entry_count = nullptr,
                                                         float *r_emissive_peak = nullptr)
{
  if (r_stats != nullptr) {
    *r_stats = {};
  }
  Vector<GPUHardwareRaytraceSceneEntry> rt_scene_entries = build_hardware_rt_scene_entries(
      scene_entries, r_emissive_entry_count, r_emissive_peak);
  return GPU_hardware_raytrace_scene_build(rt_scene_entries.as_span(), r_stats);
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
    /* Entries with a geometry recalc are rebuilt selectively from their fresh batch; a changed
     * batch pointer or resource range is expected there and must not force a full scene
     * rebuild. */
    const bool geometry_recalc = (entry.recalc & ID_RECALC_GEOMETRY) != 0;
    if (geometry_recalc) {
      continue;
    }
    if (entry.batch != cached.batch) {
      if (r_index != nullptr) {
        *r_index = i;
      }
      return "batch";
    }
    if (entry.resource_handle.index_range() != cached.resource_handle.index_range()) {
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

/* Indices are in the GPU entry space: `build_hardware_rt_scene_entries` skips null-batch
 * entries, so count only batched entries while walking the host list. */
static Vector<int> hardware_scene_entries_blas_rebuild_indices(
    const Vector<HardwareRaytraceSceneEntry> &entries)
{
  Vector<int> indices;
  int gpu_index = 0;
  for (const int i : entries.index_range()) {
    if (entries[i].batch == nullptr) {
      continue;
    }
    if ((entries[i].recalc & ID_RECALC_GEOMETRY) != 0) {
      indices.append(gpu_index);
    }
    gpu_index++;
  }
  return indices;
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
  return mask;
}

static float effective_hardware_resolution_scale(const uint32_t feature_mask,
                                                 const float base_scale,
                                                 const RaytraceEEVEE_SpecularMode /*reflection_mode*/,
                                                 const RaytraceEEVEE_SpecularMode /*refraction_mode*/)
{
  UNUSED_VARS(feature_mask);
  return raytrace_resolution_percentage_sanitize(base_scale);
}

void RayTraceModule::free_hardware_rt_scene_cache()
{
  GPU_hardware_raytrace_scene_free(hardware_rt_scene_cache_);
  hardware_rt_scene_cache_ = nullptr;
  hardware_rt_scene_stats_cache_ = {};
  hardware_rt_scene_entries_cache_.clear();
  hardware_rt_scene_update_count_ = 0;
  hardware_rt_scene_update_count_valid_ = false;
  hardware_rt_scene_signature_ = 0;
  hardware_rt_scene_signature_valid_ = false;
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
  hardware_primary_shadow_visibility_sample_index_ = 0;
  hardware_primary_shadow_visibility_sample_count_ = 1;
  hardware_primary_shadow_direct_enabled_ = false;
  hardware_primary_shadow_world_enabled_ = false;
  hardware_primary_shadow_color_intensity_ = 0.5f;
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

GPUHardwareRaytraceScene *RayTraceModule::acquire_hardware_rt_scene(
    GPUHardwareRaytraceSceneStats *r_stats, const bool require_current_feature_mask)
{
  if (r_stats != nullptr) {
    *r_stats = {};
  }
  const bool perf_logging_enabled = hardware_perf_logging_enabled();
  const double perf_start_time = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  const uint64_t current_scene_signature = inst_.sync.hardware_raytrace_scene_signature();

  if (!use_hardware_tracing() || active_hardware_feature_mask() == 0) {
    free_hardware_rt_scene_cache();
    return nullptr;
  }
  if (require_current_feature_mask && current_hardware_feature_mask_ == 0) {
    return nullptr;
  }

  const uint64_t depsgraph_update_count = (inst_.depsgraph != nullptr) ?
                                              DEG_get_update_count(inst_.depsgraph) :
                                              0;
  if (hardware_rt_scene_cache_ != nullptr && hardware_rt_scene_update_count_valid_ &&
      hardware_rt_scene_update_count_ == depsgraph_update_count &&
      hardware_rt_scene_signature_valid_ &&
      hardware_rt_scene_signature_ == current_scene_signature)
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
      *r_stats = hardware_rt_scene_stats_cache_;
    }
    return hardware_rt_scene_cache_;
  }

  const Vector<HardwareRaytraceSceneEntry> sorted_scene_entries =
      current_sorted_hardware_scene_entries(depsgraph_update_count);
  int geometry_mismatch_index = -1;
  const char *geometry_mismatch_reason = hardware_scene_entries_geometry_mismatch_reason(
      sorted_scene_entries, hardware_rt_scene_entries_cache_, &geometry_mismatch_index);
  const bool geometry_matches_cache = geometry_mismatch_reason == nullptr;
  const Vector<int> blas_rebuild_indices = geometry_matches_cache ?
                                               hardware_scene_entries_blas_rebuild_indices(
                                                   sorted_scene_entries) :
                                               Vector<int>();
  const bool blas_rebuild_requested = !blas_rebuild_indices.is_empty();
  const bool transform_changed = geometry_matches_cache &&
                                 hardware_scene_entries_transform_changed(
                                     sorted_scene_entries, hardware_rt_scene_entries_cache_);
  const bool instance_count_changed = geometry_matches_cache &&
                                      hardware_scene_entries_instance_count_changed(
                                          sorted_scene_entries,
                                          hardware_rt_scene_entries_cache_);
  const bool animation_changed = transform_changed || instance_count_changed;
  const bool emissive_changed = geometry_matches_cache &&
                                hardware_scene_entries_emissive_changed(
                                    sorted_scene_entries, hardware_rt_scene_entries_cache_);
  const bool material_changed = geometry_matches_cache &&
                                hardware_scene_entries_material_proxy_changed(
                                    sorted_scene_entries, hardware_rt_scene_entries_cache_);
  const bool needs_full_rebuild = !geometry_matches_cache;
  const bool cache_logging_enabled = std::getenv("BLENDER_EEVEE_HWRT_CACHE_LOG") != nullptr;

  if (hardware_rt_scene_cache_ != nullptr && !needs_full_rebuild) {
    if (!animation_changed && !emissive_changed && !material_changed && !blas_rebuild_requested) {
      hardware_rt_scene_entries_cache_ = sorted_scene_entries;
      hardware_rt_scene_update_count_ = depsgraph_update_count;
      hardware_rt_scene_update_count_valid_ = true;
      hardware_rt_scene_signature_ = current_scene_signature;
      hardware_rt_scene_signature_valid_ = true;
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
        *r_stats = hardware_rt_scene_stats_cache_;
      }
      return hardware_rt_scene_cache_;
    }
    Vector<GPUHardwareRaytraceSceneEntry> rt_scene_entries = build_hardware_rt_scene_entries(
        sorted_scene_entries);
    GPUHardwareRaytraceSceneUpdateParams update_params;
    /* A selective BLAS rebuild moves geometry: the TLAS, the packed world-geometry streams, and
     * the emissive bounding spheres all reference it and must refresh together. */
    update_params.update_tlas = animation_changed || blas_rebuild_requested;
    update_params.update_emissive_data = emissive_changed || animation_changed ||
                                         blas_rebuild_requested;
    update_params.update_material_data = material_changed;
    update_params.update_world_geometry_data = animation_changed || blas_rebuild_requested;
    update_params.rebuild_blas_indices = blas_rebuild_indices;
    if (GPU_hardware_raytrace_scene_update(
            hardware_rt_scene_cache_,
            rt_scene_entries.as_span(),
            update_params,
            &hardware_rt_scene_stats_cache_))
    {
      hardware_rt_scene_entries_cache_ = sorted_scene_entries;
      hardware_rt_scene_update_count_ = depsgraph_update_count;
      hardware_rt_scene_update_count_valid_ = true;
      hardware_rt_scene_signature_ = current_scene_signature;
      hardware_rt_scene_signature_valid_ = true;
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
        *r_stats = hardware_rt_scene_stats_cache_;
      }
      return hardware_rt_scene_cache_;
    }
  }

  free_hardware_rt_scene_cache();
  hardware_rt_scene_cache_ = build_hardware_rt_scene(
      sorted_scene_entries, &hardware_rt_scene_stats_cache_);
  hardware_rt_scene_entries_cache_ = sorted_scene_entries;
  hardware_rt_scene_update_count_ = depsgraph_update_count;
  hardware_rt_scene_update_count_valid_ = true;
  hardware_rt_scene_signature_ = current_scene_signature;
  hardware_rt_scene_signature_valid_ = true;
  if (cache_logging_enabled) {
    std::fprintf(stderr,
                 "EEVEE HWRT scene cache miss update=%llu entries=%d instances=%d built=%d reason=%s reason_index=%d geometry_match=%d blas_rebuild=%d animation=%d shading=%d\n",
                 (unsigned long long)depsgraph_update_count,
                 hardware_scene_entry_count_,
                 hardware_scene_instance_count_,
                 hardware_rt_scene_stats_cache_.built_scene ? 1 : 0,
                 geometry_mismatch_reason != nullptr ?
                     geometry_mismatch_reason :
                     (blas_rebuild_requested ? "geometry_recalc" : "cold_start"),
                 geometry_mismatch_index,
                 geometry_matches_cache ? 1 : 0,
                 blas_rebuild_requested ? 1 : 0,
                 animation_changed ? 1 : 0,
                 (emissive_changed || material_changed) ? 1 : 0);
  }
  if (perf_logging_enabled) {
    const double elapsed_ms = (BLI_time_now_seconds() - perf_start_time) * 1000.0;
    std::fprintf(stderr,
                 "EEVEE HWRT perf scene_cache=miss entries=%d instances=%d built=%d elapsed_ms=%.2f\n",
                 hardware_scene_entry_count_,
                 hardware_scene_instance_count_,
                 hardware_rt_scene_stats_cache_.built_scene ? 1 : 0,
                 elapsed_ms);
  }
  if (r_stats != nullptr) {
    *r_stats = hardware_rt_scene_stats_cache_;
  }
  return hardware_rt_scene_cache_;
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


void RayTraceModule::render_directional_shadow_visibility(View &render_view,
                                                          gpu::Texture *depth_tx,
                                                          gpu::Texture *gbuf_normal_tx,
                                                          int2 extent)
{
  const bool perf_logging_enabled = hardware_perf_logging_enabled();
  const double perf_start_time = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  /* Nuru: RGBA visibility, fully lit init. See top of file for rationale. */
  const float visibility[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  const bool use_world_rt_shadows = use_hardware_environment();
  const bool use_direct_rt_shadows = use_hardware_shadows();
  const float shadow_color_intensity = hardware_shadow_color_intensity();
  auto mark_shadow_visibility_ready = [&]() {
    hardware_primary_shadow_visibility_ready_ = true;
    hardware_primary_shadow_visibility_depth_tx_ = depth_tx;
    hardware_primary_shadow_visibility_normal_tx_ = gbuf_normal_tx;
    hardware_primary_shadow_visibility_extent_ = extent;
    hardware_primary_shadow_visibility_sample_index_ = inst_.sampling.sample_index();
    hardware_primary_shadow_visibility_sample_count_ = hardware_shadow_sample_count_;
    hardware_primary_shadow_direct_enabled_ = use_direct_rt_shadows;
    hardware_primary_shadow_world_enabled_ = use_world_rt_shadows;
    hardware_primary_shadow_color_intensity_ = shadow_color_intensity;
  };
  const bool reuse_shadow_visibility = hardware_primary_shadow_visibility_ready_ &&
                                       hardware_primary_shadow_visibility_depth_tx_ == depth_tx &&
                                       hardware_primary_shadow_visibility_normal_tx_ ==
                                           gbuf_normal_tx &&
                                       hardware_primary_shadow_visibility_extent_ == extent &&
                                       hardware_primary_shadow_visibility_sample_index_ ==
                                           inst_.sampling.sample_index() &&
                                       hardware_primary_shadow_visibility_sample_count_ ==
                                           hardware_shadow_sample_count_ &&
                                       hardware_primary_shadow_direct_enabled_ ==
                                           use_direct_rt_shadows &&
                                       hardware_primary_shadow_world_enabled_ ==
                                           use_world_rt_shadows &&
                                       hardware_primary_shadow_color_intensity_ ==
                                           shadow_color_intensity;
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
        gpu::TextureFormat::SFLOAT_16_16_16_16, int2(1), 1, GPU_TEXTURE_USAGE_SHADER_READ, visibility);
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
        gpu::TextureFormat::SFLOAT_16_16_16_16, int2(1), 1, GPU_TEXTURE_USAGE_SHADER_READ, visibility);
    mark_shadow_visibility_ready();
    return;
  }

  constexpr eGPUTextureUsage usage_rw = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;
  const int local_light_count = inst_.lights.local_lights_len();
  const int sun_light_count = inst_.lights.sun_lights_len();
  const int total_light_count = local_light_count + sun_light_count;
  const int hwrt_shadow_sample_count = hardware_shadow_sample_count_;
  const int hwrt_world_shadow_sample_count = hardware_shadow_sample_count_;
  eGPUTextureUsage direct_light_output_usage = usage_rw;
  if (total_light_count == 0) {
    const float4 zero = float4(0.0f);
    hardware_direct_light_accum_tx_.ensure_2d(
        gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, int2(1), GPU_TEXTURE_USAGE_SHADER_READ, zero);
    hardware_direct_light_denoised_tx_.ensure_2d(
        gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, int2(1), GPU_TEXTURE_USAGE_SHADER_READ, zero);
    hardware_shadow_visibility_tx_.ensure_2d_array(
        gpu::TextureFormat::SFLOAT_16_16_16_16, int2(1), 1, GPU_TEXTURE_USAGE_SHADER_READ, visibility);
    mark_shadow_visibility_ready();
    return;
  }
  const LightCullingData &light_culling_data = inst_.lights.culling_data();
  const int2 direct_light_tile_extent = int2(light_culling_data.tile_x_len, light_culling_data.tile_y_len);
  hardware_shadow_visibility_tx_.ensure_2d_array(
      gpu::TextureFormat::SFLOAT_16_16_16_16, extent, total_light_count, usage_rw);
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
  /* No flush needed: the Vulkan backend flushes the render graph and waits for queue
   * submission before every kernel dispatch, and Metal commits in CPU order. The extra
   * flush only split the frame into more queue submissions. */

  Vector<LightData> local_lights;
  Vector<LightData> sun_lights;
  Vector<int> sun_light_world_slots;
  inst_.lights.append_sync_local_lights(local_lights);
  inst_.lights.append_sync_sun_lights(sun_lights, &sun_light_world_slots);

  const double scene_acquire_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  GPUHardwareRaytraceSceneStats rt_scene_stats;
  GPUHardwareRaytraceScene *rt_scene = acquire_hardware_rt_scene(&rt_scene_stats, false);
  const double scene_acquire_ms = perf_logging_enabled ?
                                      (BLI_time_now_seconds() - scene_acquire_start) * 1000.0 :
                                      0.0;
  if (rt_scene == nullptr || !rt_scene_stats.built_scene) {
    return;
  }

  int traced_local_lights = 0;
  int traced_sun_lights = 0;
  bool shadow_trace_failed = false;
  const double trace_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  const float4 shadow_sampling_rand = hardware_shadow_sampling_rand(inst_);
  GPU_debug_group_begin("Hardware RT Shadows");
  const bool shadow_batch_active = GPU_hardware_raytrace_scene_shadow_batch_begin(rt_scene);
  if (use_direct_rt_shadows) {
    for (const int local_index : local_lights.index_range()) {
      const LightData &light = local_lights[local_index];
      /* Use `cast_shadow` rather than `tilemap_index` so HWRT keeps firing even when the VSM
       * tilemap has been intentionally skipped (Nuru: no VSM for local lights when HWRT on). */
      if (!light.cast_shadow || light.color.x < 0.0f) {
        continue;
      }

      GPUHardwareRaytraceLocalShadowParams shadow_params;
      shadow_params.depth_tx = depth_tx;
      shadow_params.gbuf_header_tx = inst_.gbuffer.header_tx;
      shadow_params.gbuf_normal_tx = gbuf_normal_tx;
      shadow_params.shadow_visibility_tx = hardware_shadow_visibility_tx_;
      shadow_params.viewinv = render_view.viewinv();
      shadow_params.wininv = render_view.wininv();
      shadow_params.full_resolution = extent;
      shadow_params.shadow_layer = light.shadow_layer;
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
      shadow_params.use_caustics = false;
      shadow_params.color_intensity = shadow_color_intensity;
      shadow_params.photons_intensity = 0.0f;
      shadow_params.shadow_transparency = hardware_shadow_transparency();
      shadow_params.sampling_rand = shadow_sampling_rand;
      const bool trace_submitted = GPU_hardware_raytrace_scene_trace_local_shadow(rt_scene,
                                                                               shadow_params);
      traced_local_lights += trace_submitted ? 1 : 0;
      shadow_trace_failed |= !trace_submitted;
    }
  }
  for (const int sun_index : sun_lights.index_range()) {
    const LightData &light = sun_lights[sun_index];
    /* HWRT uses `cast_shadow`, not the VSM tilemap presence. Sun-VSM is always skipped in Nuru. */
    if (!is_sun_light(light.type) || (!light.cast_shadow && light.color.x >= 0.0f)) {
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

    GPUHardwareRaytraceDirectionalShadowParams shadow_params;
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
    shadow_params.use_caustics = false;
    shadow_params.color_intensity = shadow_color_intensity;
    shadow_params.photons_intensity = 0.0f;
    shadow_params.shadow_transparency = hardware_shadow_transparency();
    shadow_params.sampling_rand = shadow_sampling_rand;
    const bool trace_submitted = GPU_hardware_raytrace_scene_trace_directional_shadow(rt_scene,
                                                                                    shadow_params);
    traced_sun_lights += trace_submitted ? 1 : 0;
    shadow_trace_failed |= !trace_submitted;
  }
  const bool shadow_batch_committed = shadow_batch_active ?
                                          GPU_hardware_raytrace_scene_shadow_batch_end(rt_scene) :
                                          true;
  GPU_debug_group_end();
  if (shadow_batch_committed) {
    GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH | GPU_BARRIER_SHADER_IMAGE_ACCESS);
  }
  if (!shadow_trace_failed && shadow_batch_committed) {
    mark_shadow_visibility_ready();
  }
  if (perf_logging_enabled) {
    const double trace_ms = (BLI_time_now_seconds() - trace_start) * 1000.0;
    const double elapsed_ms = (BLI_time_now_seconds() - perf_start_time) * 1000.0;
    std::fprintf(stderr,
                 "EEVEE HWRT perf primary_shadows reused=0 samples=%d local=%d sun=%d batched=%d committed=%d scene_acquire_ms=%.2f trace_submit_ms=%.2f elapsed_ms=%.2f\n",
                 hwrt_shadow_sample_count,
                 traced_local_lights,
                 traced_sun_lights,
                 shadow_batch_active ? 1 : 0,
                 shadow_batch_committed ? 1 : 0,
                 scene_acquire_ms,
                 trace_ms,
                 elapsed_ms);
  }
  if (!hardware_uses_viewport_reference(inst_) || !use_direct_rt_shadows ||
      !hardware_direct_light_dispatch_ready_ || hardware_direct_light_dispatch_extent_ != extent ||
      hardware_direct_light_dispatch_viewinv_ != render_view.viewinv() ||
      hardware_direct_light_dispatch_wininv_ != render_view.wininv())
  {
    return;
  }
  inst_.manager->submit(hardware_direct_light_visibility_ps_);
  inst_.manager->submit(hardware_direct_light_accum_ps_, render_view);
  hardware_nis_train_offset_ = (hardware_nis_train_offset_ + 1) % 16384;
  inst_.manager->submit(hardware_nis_train_ps_, render_view);
  inst_.manager->submit(hardware_nis_train_feedback_ps_, render_view);
  inst_.manager->submit(hardware_nis_adam_ps_);
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
  /* No flush needed: the Vulkan backend flushes the render graph and waits for queue
   * submission before every kernel dispatch, and Metal commits in CPU order. The extra
   * flush only split the frame into more queue submissions. */

  const double scene_acquire_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  GPUHardwareRaytraceSceneStats rt_scene_stats;
  GPUHardwareRaytraceScene *rt_scene = acquire_hardware_rt_scene(
      &rt_scene_stats, false);
  const double scene_acquire_ms = perf_logging_enabled ?
                                      (BLI_time_now_seconds() - scene_acquire_start) * 1000.0 :
                                      0.0;
  if (rt_scene == nullptr || !rt_scene_stats.built_scene) {
    return;
  }

  GPUHardwareRaytraceEnvironmentVisibilityParams env_params;
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
  const bool trace_submitted = GPU_hardware_raytrace_scene_trace_environment_visibility(rt_scene,
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

void RayTraceModule::render_secondary_environment_visibility(GPUHardwareRaytraceScene *rt_scene,
                                                             int2 tracing_extent)
{
  const float visibility[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  if (!use_hardware_tracing() || !use_hardware_environment() || rt_scene == nullptr ||
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
  /* No flush: the clear and the visibility kernel execute in queue order on the same GPU queue;
   * the submission break only added CPU cost per closure call. */

  GPUHardwareRaytraceHitEnvironmentVisibilityParams env_params;
  env_params.hit_normal_tx = hit_normal_tx_;
  env_params.hit_world_position_tx = hit_world_position_tx_;
  env_params.environment_visibility_tx = hardware_secondary_environment_visibility_tx_;
  env_params.dispatch_buf = hardware_resolve_dispatch_buf_;
  env_params.tiles_coord_buf = hardware_resolve_tiles_buf_;
  env_params.tracing_resolution = tracing_extent;
  env_params.sample_count = hardware_visibility_temporal_sample_count;
  env_params.normal_bias = 5.0e-3f;
  env_params.sampling_rand = hardware_shadow_sampling_rand(inst_);
  if (GPU_hardware_raytrace_scene_trace_hit_environment_visibility(rt_scene, env_params)) {
    GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH | GPU_BARRIER_SHADER_IMAGE_ACCESS);
  }
}

void RayTraceModule::render_hit_shadow_visibility(GPUHardwareRaytraceScene *rt_scene,
                                                  int2 tracing_extent,
                                                  gpu::Texture *hit_normal_tx,
                                                  gpu::Texture *hit_world_position_tx,
                                                  gpu::Texture *hit_identity_tx,
                                                  Texture &shadow_visibility_tx)
{
  const bool perf_logging_enabled = hardware_perf_logging_enabled();
  const double perf_start_time = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  /* Nuru: RGBA visibility, fully lit init. */
  const float visibility[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  const bool use_world_rt_shadows = use_hardware_environment();
  const bool use_direct_rt_shadows = use_hardware_shadows();
  if (!use_hardware_tracing() || (!use_direct_rt_shadows && !use_world_rt_shadows) ||
      rt_scene == nullptr ||
      hit_normal_tx == nullptr || hit_world_position_tx == nullptr || hit_identity_tx == nullptr ||
      tracing_extent.x <= 0 || tracing_extent.y <= 0)
  {
    shadow_visibility_tx.ensure_2d_array(
        gpu::TextureFormat::SFLOAT_16_16_16_16, int2(1), 1, GPU_TEXTURE_USAGE_SHADER_READ, visibility);
    return;
  }

  const int local_light_count = inst_.lights.local_lights_len();
  const int sun_light_count = inst_.lights.sun_lights_len();
  const int total_light_count = local_light_count + sun_light_count;
  const int hwrt_shadow_sample_count = hardware_shadow_sample_count_;
  const int hwrt_world_shadow_sample_count = hardware_shadow_sample_count_;
  if (total_light_count == 0) {
    shadow_visibility_tx.ensure_2d_array(
        gpu::TextureFormat::SFLOAT_16_16_16_16, int2(1), 1, GPU_TEXTURE_USAGE_SHADER_READ, visibility);
    return;
  }

  constexpr eGPUTextureUsage usage_rw = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;
  shadow_visibility_tx.ensure_2d_array(
      gpu::TextureFormat::SFLOAT_16_16_16_16, tracing_extent, total_light_count, usage_rw);
  shadow_visibility_tx.clear(float4(1.0f));
  /* No flush needed: the Vulkan backend flushes the render graph and waits for queue
   * submission before every kernel dispatch, and Metal commits in CPU order. The extra
   * flush only split the frame into more queue submissions. */

  Vector<LightData> local_lights;
  Vector<LightData> sun_lights;
  Vector<int> sun_light_world_slots;
  inst_.lights.append_sync_local_lights(local_lights);
  inst_.lights.append_sync_sun_lights(sun_lights, &sun_light_world_slots);

  int traced_local_lights = 0;
  int traced_sun_lights = 0;
  const float4 shadow_sampling_rand = hardware_shadow_sampling_rand(inst_);
  GPU_debug_group_begin("Hardware RT Hit Shadows");
  const bool shadow_batch_active = GPU_hardware_raytrace_scene_shadow_batch_begin(rt_scene);
  if (use_direct_rt_shadows) {
    for (const int local_index : local_lights.index_range()) {
      const LightData &light = local_lights[local_index];
      /* See note in `render_directional_shadow_visibility`. */
      if (!light.cast_shadow || light.color.x < 0.0f) {
        continue;
      }

      GPUHardwareRaytraceLocalHitShadowParams shadow_params;
      shadow_params.hit_normal_tx = hit_normal_tx;
      shadow_params.hit_world_position_tx = hit_world_position_tx;
      shadow_params.hit_identity_tx = hit_identity_tx;
      shadow_params.shadow_visibility_tx = shadow_visibility_tx;
      shadow_params.dispatch_buf = hardware_resolve_dispatch_buf_;
      shadow_params.tiles_coord_buf = hardware_resolve_tiles_buf_;
      shadow_params.tracing_resolution = tracing_extent;
      shadow_params.shadow_layer = light.shadow_layer;
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
      shadow_params.use_caustics = false;
      shadow_params.color_intensity = hardware_shadow_color_intensity();
      shadow_params.photons_intensity = 0.0f;
      shadow_params.shadow_transparency = hardware_shadow_transparency();
      shadow_params.sampling_rand = shadow_sampling_rand;
      traced_local_lights +=
          GPU_hardware_raytrace_scene_trace_local_hit_shadow(rt_scene, shadow_params) ? 1 : 0;
    }
  }

  for (const int sun_index : sun_lights.index_range()) {
    const LightData &light = sun_lights[sun_index];
    /* See note in `render_directional_shadow_visibility`. */
    if (!is_sun_light(light.type) || (!light.cast_shadow && light.color.x >= 0.0f)) {
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

    GPUHardwareRaytraceDirectionalHitShadowParams shadow_params;
    shadow_params.hit_normal_tx = hit_normal_tx;
    shadow_params.hit_world_position_tx = hit_world_position_tx;
    shadow_params.hit_identity_tx = hit_identity_tx;
    shadow_params.shadow_visibility_tx = shadow_visibility_tx;
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
    shadow_params.use_caustics = false;
    shadow_params.color_intensity = hardware_shadow_color_intensity();
    shadow_params.photons_intensity = 0.0f;
    shadow_params.shadow_transparency = hardware_shadow_transparency();
    shadow_params.sampling_rand = shadow_sampling_rand;
    traced_sun_lights += GPU_hardware_raytrace_scene_trace_directional_hit_shadow(rt_scene,
                                                                               shadow_params) ?
                             1 :
                             0;
  }
  const bool shadow_batch_committed = shadow_batch_active ?
                                          GPU_hardware_raytrace_scene_shadow_batch_end(rt_scene) :
                                          true;
  GPU_debug_group_end();
  if (shadow_batch_committed) {
    GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH | GPU_BARRIER_SHADER_IMAGE_ACCESS);
  }
  if (perf_logging_enabled) {
    const double elapsed_ms = (BLI_time_now_seconds() - perf_start_time) * 1000.0;
    std::fprintf(stderr,
                 "EEVEE HWRT perf secondary_shadows samples=%d local=%d sun=%d batched=%d committed=%d elapsed_ms=%.2f\n",
                 hwrt_shadow_sample_count,
                 traced_local_lights,
                 traced_sun_lights,
                 shadow_batch_active ? 1 : 0,
                 shadow_batch_committed ? 1 : 0,
                 elapsed_ms);
  }
}

void RayTraceModule::render_secondary_shadow_visibility(GPUHardwareRaytraceScene *rt_scene,
                                                        int2 tracing_extent)
{
  render_hit_shadow_visibility(rt_scene,
                               tracing_extent,
                               hit_normal_tx_,
                               hit_world_position_tx_,
                               hit_identity_tx_,
                               hardware_secondary_shadow_visibility_tx_);
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

  const double clears_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  hit_albedo_tx_.clear(float4(0.0f));
  reflected_receiver_gi_tx_.clear(float4(0.0f));
  hardware_nis_feedback_buf_.clear_to_zero();
  layered_receiver_gi_tx_.clear(float4(0.0f));
  transmission_receiver_gi_tx_.clear(float4(0.0f));
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
  /* No flush needed: the Vulkan backend flushes the render graph and waits for queue
   * submission before every kernel dispatch, and Metal commits in CPU order. The extra
   * flush only split the frame into more queue submissions. */
  const double clears_ms = perf_logging_enabled ?
                               (BLI_time_now_seconds() - clears_start) * 1000.0 :
                               0.0;

  const double scene_acquire_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  GPUHardwareRaytraceSceneStats rt_scene_stats;
  GPUHardwareRaytraceScene *rt_scene = acquire_hardware_rt_scene(&rt_scene_stats);
  const double scene_acquire_ms = perf_logging_enabled ?
                                      (BLI_time_now_seconds() - scene_acquire_start) * 1000.0 :
                                      0.0;

  /* Hardware RT owns this closure. Only explicit Hardware hits or Hardware miss-resolve passes
   * should contribute to the result. */
  if (rt_scene != nullptr && rt_scene_stats.built_scene) {
    GPU_debug_group_begin("Hardware RT Scene");
    hardware_resolve_dispatch_buf_.clear_to_zero();
    const int2 tracing_res = math::divide_ceil(
        data_.full_resolution * data_.resolution_scale_denominator, int2(data_.resolution_scale));
    GPUHardwareRaytraceTraceParams trace_params;
    trace_params.ray_data_tx = ray_data_tx_;
    trace_params.depth_tx = renderbuf_depth_view_;
    trace_params.gbuf_header_tx = inst_.gbuffer.header_tx;
    trace_params.gbuf_normal_tx = inst_.gbuffer.normal_tx;
    trace_params.screen_continuation_tx = screen_continuation_tx_;
    trace_params.world_probe_tx = inst_.sphere_probes.octahedral_probes_texture();
    trace_params.ray_time_tx = ray_time_tx_;
    trace_params.ray_radiance_tx = ray_radiance_tx_;
    trace_params.hit_albedo_tx = hit_albedo_tx_;
    trace_params.reflected_receiver_gi_tx = use_hardware_fast_gi_secondary() ?
                                                static_cast<gpu::Texture *>(
                                                    reflected_receiver_gi_tx_) :
                                                nullptr;
    trace_params.layered_receiver_gi_tx = use_hardware_fast_gi_secondary() ?
                                              static_cast<gpu::Texture *>(
                                                  layered_receiver_gi_tx_) :
                                              nullptr;
    trace_params.transmission_receiver_gi_tx = use_hardware_fast_gi_secondary() ?
                                                   static_cast<gpu::Texture *>(
                                                       transmission_receiver_gi_tx_) :
                                                   nullptr;
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
    trace_params.resolution_scale_denominator = data_.resolution_scale_denominator;
    trace_params.closure_index = data_.closure_index;
    trace_params.feature_mask = current_hardware_feature_mask_;
    trace_params.hardware_trace_phase = data_.hardware_trace_phase;
    trace_params.reflection_bounces = data_.hardware_reflection_bounces;
    trace_params.refraction_bounces = data_.hardware_refraction_bounces;
    trace_params.resolution_bias = data_.resolution_bias;
    trace_params.clamp_indirect = 1.0e10f;
    /* Nuru light tree (Stage A many-light sampling / NIS): the CPU tree build and the light
     * record upload only depend on the synced lights. Rebuild once per depsgraph update instead
     * of once per closure/phase trace call; light edits invalidate the depsgraph, so
     * interactivity is unaffected. */
    const uint64_t light_records_update_count = (inst_.depsgraph != nullptr) ?
                                                    DEG_get_update_count(inst_.depsgraph) :
                                                    0;
    if (!hardware_light_records_update_count_valid_ ||
        hardware_light_records_update_count_ != light_records_update_count)
    {
      Vector<LightData> local_lights;
      Vector<LightData> sun_lights;
      Vector<int> sun_light_world_slots;
      inst_.lights.append_sync_local_lights(local_lights);
      inst_.lights.append_sync_sun_lights(sun_lights, &sun_light_world_slots);
      const int local_light_count = min_ii(256, int(local_lights.size()));
      const int light_count = min_ii(256, int(local_lights.size() + sun_lights.size()));
      for (int light_index = 0; light_index < light_count; light_index++) {
        const LightData &light = (light_index < local_lights.size()) ?
                                     local_lights[light_index] :
                                     sun_lights[light_index - local_lights.size()];
        hardware_fast_gi_light_buf_.get_or_resize(light_index) =
            hardware_fast_gi_light_record_from_light(light);
      }
      /* Encoded tree nodes ride in the same buffer after the light records, so every existing
       * binding carries the tree. */
      const Vector<GPUHardwareRaytraceFastGILightRecord> tree_nodes = hardware_light_tree_build(
          Span<LightData>(local_lights).take_front(local_light_count), 0);
      for (const int node_index : tree_nodes.index_range()) {
        hardware_fast_gi_light_buf_.get_or_resize(light_count + node_index) =
            tree_nodes[node_index];
      }
      if (light_count > 0) {
        hardware_fast_gi_light_buf_.resize(light_count + int(tree_nodes.size()));
        hardware_fast_gi_light_buf_.push_update();
      }
      hardware_light_records_light_count_ = light_count;
      hardware_light_records_local_light_count_ = local_light_count;
      hardware_light_records_update_count_ = light_records_update_count;
      hardware_light_records_update_count_valid_ = true;
    }
    const int trace_local_light_count = hardware_light_records_local_light_count_;
    const int trace_light_count = hardware_light_records_light_count_;
    trace_params.light_buf = (trace_light_count > 0) ?
                                 static_cast<gpu::StorageBuf *>(hardware_fast_gi_light_buf_) :
                                 nullptr;
    trace_params.light_count = trace_light_count;
    trace_params.local_light_count = trace_local_light_count;
    trace_params.nis_weights_buf = hardware_nis_weights_buf_;
    trace_params.nis_enable = use_hardware_direct_light();
    trace_params.nis_feedback_buf = hardware_nis_feedback_buf_;
    trace_params.secondary_gi = use_hardware_fast_gi_secondary();
    trace_params.secondary_gi_samples = inst_.is_viewport() ? 2 : 4;
    const bool viewport_reference = hardware_uses_viewport_reference(inst_);
    trace_params.light_sample_count = hardware_fast_gi_direct_light_sample_count(
        trace_light_count, viewport_reference, hardware_fast_gi_quality_tier_);
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
    trace_params.use_environment = use_hardware_reflections() || use_hardware_refractions();
    trace_params.use_diffuse_environment = use_hardware_rt_gi();
    trace_params.gi_diffuse_sample_count = hardware_gi_spatial_sample_count_sanitize(
        ray_tracing_options_.gi_spatial_samples);
    const float3 raytrace_rng = inst_.sampling.rng_3d_get(eSamplingDimension::SAMPLING_RAYTRACE_U);
    trace_params.sampling_rand = float4(
        raytrace_rng.x,
        raytrace_rng.y,
        raytrace_rng.z,
        inst_.sampling.rng_get(eSamplingDimension::SAMPLING_CLOSURE));
    trace_params.dispatch_buf = hardware_trace_dispatch_buf_;
    trace_params.tiles_coord_buf = hardware_trace_tiles_buf_;
    const double trace_submit_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
    GPU_hardware_raytrace_scene_trace(rt_scene, trace_params);
    const double trace_submit_ms = perf_logging_enabled ?
                                       (BLI_time_now_seconds() - trace_submit_start) * 1000.0 :
                                       0.0;
    inst_.manager->submit(hardware_tile_compact_ps_);
    const double hit_eval_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
    submit_hardware_hit_evaluation_backend(render_view);
    const double hit_eval_ms = perf_logging_enabled ?
                                   (BLI_time_now_seconds() - hit_eval_start) * 1000.0 :
                                   0.0;
    const double secondary_environment_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
    render_secondary_environment_visibility(rt_scene, tracing_res);
    const double secondary_environment_ms =
        perf_logging_enabled ?
            (BLI_time_now_seconds() - secondary_environment_start) * 1000.0 :
            0.0;
    const double secondary_shadow_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
    render_secondary_shadow_visibility(rt_scene, tracing_res);
    render_hit_shadow_visibility(rt_scene,
                                 tracing_res,
                                 layered_receiver_normal_tx_,
                                 layered_receiver_world_position_tx_,
                                 layered_receiver_identity_tx_,
                                 hardware_layered_receiver_shadow_visibility_tx_);
    render_hit_shadow_visibility(rt_scene,
                                 tracing_res,
                                 transmission_receiver_normal_tx_,
                                 transmission_receiver_world_position_tx_,
                                 transmission_receiver_identity_tx_,
                                 hardware_transmission_receiver_shadow_visibility_tx_);
    const double secondary_shadow_ms = perf_logging_enabled ?
                                           (BLI_time_now_seconds() - secondary_shadow_start) * 1000.0 :
                                           0.0;
    const double lighting_start = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
    inst_.manager->submit(trace_hardware_lighting_ps_, render_view);
    if (perf_logging_enabled) {
      const double lighting_ms = (BLI_time_now_seconds() - lighting_start) * 1000.0;
      const double elapsed_ms = (BLI_time_now_seconds() - perf_start_time) * 1000.0;
      std::fprintf(stderr,
                   "EEVEE HWRT perf trace closure=%d features=0x%x clears_ms=%.2f scene_acquire_ms=%.2f trace_submit_ms=%.2f hit_eval_ms=%.2f secondary_env_ms=%.2f secondary_shadow_ms=%.2f lighting_ms=%.2f elapsed_ms=%.2f\n",
                   data_.closure_index,
                   unsigned(current_hardware_feature_mask_),
                   clears_ms,
                   scene_acquire_ms,
                   trace_submit_ms,
                   hit_eval_ms,
                   secondary_environment_ms,
                   secondary_shadow_ms,
                   lighting_ms,
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

  const Span<HardwareRaytraceSceneEntry> all_entries = hardware_rt_scene_entries_cache_;
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
    /* Submit with the view: the shader uses `drw_point_screen_to_world()` which reads the view
     * UBO (slot 11); submitting without a view leaves that slot unbound (garbage matrices on
     * Vulkan, assert under --debug-gpu). */
    inst_.manager->submit(hit_eval_compact_ps_, render_view);
    GPU_storagebuf_sync_as_indirect_buffer(hit_eval_indirect_buf_);

    hit_eval_ps_.init();
    hit_eval_ps_.state_set(DRW_STATE_WRITE_COLOR | DRW_STATE_DEPTH_ALWAYS);
    hit_eval_ps_.framebuffer_set(&hit_eval_fb_);

    auto bind_hit_eval_resources = [&](draw::PassSimple &pass) {
      pass.bind_texture(RBUFS_UTILITY_TEX_SLOT, inst_.pipelines.utility_tx);
      /* Nuru: replayed material graphs can sample HiZ (slot 3) like every other material
       * pass; without this bind Vulkan logs a missing-bind error per hit-eval draw. */
      pass.bind_resources(inst_.hiz_buffer.front);
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
      const HardwareHitEvalBatchSupport support = hardware_hit_eval_batch_support(entry.batch,
                                                                                  shader);
      if (!support.geometry_compatible) {
        /* Position/normal/index SSBOs are non-negotiable for replaying the traced triangle. Fail
         * closed to the proxy payload rather than asserting in GPU_batch_bind_as_resources(). */
        continue;
      }

      hit_eval_ps_.material_set(*inst_.manager, gpumat, true);
      bind_hit_eval_resources(hit_eval_ps_);
      /* Per-material batches index into a subrange of the mesh index buffer. The BLAS applies
       * this offset at build time, so the replayed primitive ids are subrange-relative while
       * `GPU_batch_bind_as_resources()` binds the whole buffer. */
      hit_eval_ps_.push_constant("hit_eval_index_start",
                                 (entry.batch->elem != nullptr) ?
                                     int(entry.batch->elem->index_start_get()) :
                                     0);
      if (support.missing_attr_mask != 0) {
        /* Material attribute layers the mesh does not carry: bind a dummy buffer to satisfy the
         * SSBO slot and zero the attribute descriptors. A zero `gpu_attr_N_meta` makes every
         * `hit_attr_fetch_*` word-load short-circuit to zero before touching the buffer, matching
         * the raster pipeline's missing-attribute behavior (reads as zero). The recorded zeros
         * stay authoritative because `GPU_batch_bind_as_resources()` only overwrites descriptors
         * for attributes it actually finds in the batch. */
        for (int location = 0; location < 16; location++) {
          if ((support.missing_attr_mask & (1 << location)) == 0) {
            continue;
          }
          hit_eval_ps_.bind_ssbo(location, &hit_eval_dummy_attr_buf_);
          char gpu_attr_name[32];
          SNPRINTF(gpu_attr_name, "gpu_attr_%d", location);
          hit_eval_ps_.push_constant(gpu_attr_name, int2(0));
          char gpu_attr_meta_name[32];
          SNPRINTF(gpu_attr_meta_name, "gpu_attr_%d_meta", location);
          hit_eval_ps_.push_constant(gpu_attr_meta_name, 0);
        }
      }
      hit_eval_ps_.draw_expand_indirect(entry.batch,
                                        GPU_PRIM_TRIS,
                                        1,
                                        &hit_eval_indirect_buf_,
                                        uint32_t(sizeof(DrawCommand) * entry_index),
                                        ResourceIndexRange(entry.resource_handle).first);
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
  /* Grow-only: the record capacity follows the per-phase tracing resolution, which alternates
   * between the GI scale and the full-resolution scene-final specular phase within one frame. */
  hit_eval_records_buf_.resize(max_ii(int(hit_eval_records_buf_.size()), max_hit_records));

  /* The resource-id table only depends on the sorted scene entries; skip the rebuild + GPU
   * upload for the repeated per-closure/per-phase calls within one synced frame. */
  if (!hit_eval_resource_ids_update_count_valid_ ||
      hit_eval_resource_ids_update_count_ != hardware_sorted_scene_entries_update_count_ ||
      hit_eval_resource_ids_entry_count_ != entry_count)
  {
    for (const int entry_index : all_entries.index_range()) {
      const HardwareRaytraceSceneEntry &entry = all_entries[entry_index];
      const ResourceIndexRange resource_range = entry.resource_handle;
      hit_eval_resource_id_buf_.get_or_resize(entry_index) = uint(resource_range.first.raw);
    }
    hit_eval_resource_id_buf_.push_update();
    hit_eval_resource_ids_update_count_ = hardware_sorted_scene_entries_update_count_;
    hit_eval_resource_ids_entry_count_ = entry_count;
    hit_eval_resource_ids_update_count_valid_ = true;
  }
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
  /* The layered/transmission receiver payloads are only produced by the scene-final specular
   * preservation paths in the trace kernels; during the GI phase they are cleared textures.
   * Recording ~entry_count replay draws for three payload sets per call was the dominant CPU
   * cost of this function, so skip the receiver payload passes outside the scene-final phase. */
  const bool scene_final_specular_phase = (data_.hardware_trace_phase ==
                                           int(HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR));
  const bool submitted_receiver = scene_final_specular_phase &&
                                  submit_payload_hit_eval({layered_receiver_ray_time_tx_,
                                                           layered_receiver_ray_radiance_tx_,
                                                           layered_receiver_albedo_tx_,
                                                           layered_receiver_throughput_tx_,
                                                           layered_receiver_material_tx_,
                                                           layered_receiver_normal_tx_,
                                                           layered_receiver_position_tx_,
                                                           layered_receiver_world_position_tx_,
                                                           layered_receiver_identity_tx_,
                                                           layered_receiver_barycentric_tx_});
  const bool submitted_transmission_receiver =
      scene_final_specular_phase &&
      submit_payload_hit_eval({transmission_receiver_ray_time_tx_,
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
    pass.specialize_constant(
        sh, "resolution_scale_denominator", &data_.resolution_scale_denominator);
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
    pass.push_constant("hardware_direct_light_tile_capacity", &hardware_direct_light_tile_capacity_);
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
    pass.push_constant("hardware_direct_light_tile_capacity", &hardware_direct_light_tile_capacity_);
    pass.bind_texture("hardware_rt_shadow_visibility_tx", &hardware_shadow_visibility_tx_);
    /* Nuru N0: position-aware light importance reconstructs P at the sampled texel. */
    pass.bind_texture("depth_tx", &inst_.render_buffers.depth_tx);
    pass.bind_ssbo("hardware_light_cluster_weight_buf", &hardware_light_cluster_weight_buf_);
    pass.bind_ssbo("hardware_nis_weights_buf", &hardware_nis_weights_buf_);
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
    pass.push_constant("hardware_direct_light_tile_capacity", &hardware_direct_light_tile_capacity_);
    pass.bind_image("out_direct_light_accum_img", &hardware_direct_light_accum_tx_);
    pass.bind_texture("depth_tx", &renderbuf_depth_view_);
    pass.bind_ssbo("hardware_light_cluster_weight_buf", &hardware_light_cluster_weight_buf_);
    pass.bind_ssbo("hardware_nis_weights_buf", &hardware_nis_weights_buf_);
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
    /* Nuru NIS stage N2: online training of the cluster-multiplier network. The trainer reads
     * the realized contribution feedback written by the accumulation kernel; Adam applies the
     * averaged step. Both run every frame; the trainer strides the tiles for budget. */
    PassSimple &pass = hardware_nis_train_ps_;
    pass.init();
    pass.shader_set(inst_.shaders.static_shader_get(NIS_TRAIN));
    pass.push_constant("hardware_direct_light_tile_capacity", &hardware_direct_light_tile_capacity_);
    pass.push_constant("hardware_nis_train_stride", &hardware_nis_train_stride_);
    pass.push_constant("hardware_nis_train_offset", &hardware_nis_train_offset_);
    pass.bind_texture("depth_tx", &renderbuf_depth_view_);
    pass.bind_ssbo("hardware_light_cluster_weight_buf", &hardware_light_cluster_weight_buf_);
    pass.bind_ssbo("hardware_nis_weights_buf", &hardware_nis_weights_buf_);
    pass.bind_ssbo("hardware_nis_grads_buf", &hardware_nis_grads_buf_);
    pass.bind_ssbo("hardware_nis_train_count_buf", &hardware_nis_train_count_buf_);
    pass.bind_ssbo("hardware_direct_light_work_tiles_buf", &hardware_direct_light_work_tiles_buf_);
    pass.bind_ssbo("hardware_direct_light_visibility_samples_buf",
                   &hardware_direct_light_visibility_samples_buf_);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.lights);
    pass.bind_resources(inst_.sampling);
    pass.dispatch(int3(divide_ceil(int2(hardware_direct_light_tile_capacity_, 1), int2(64, 1)), 1));
    pass.barrier(GPU_BARRIER_SHADER_STORAGE);
  }
  {
    PassSimple &pass = hardware_nis_train_feedback_ps_;
    pass.init();
    pass.shader_set(inst_.shaders.static_shader_get(NIS_TRAIN_FEEDBACK));
    pass.bind_ssbo("hardware_light_cluster_weight_buf", &hardware_light_cluster_weight_buf_);
    pass.bind_ssbo("hardware_nis_weights_buf", &hardware_nis_weights_buf_);
    pass.bind_ssbo("hardware_nis_grads_buf", &hardware_nis_grads_buf_);
    pass.bind_ssbo("hardware_nis_train_count_buf", &hardware_nis_train_count_buf_);
    pass.bind_ssbo("hardware_nis_feedback_buf", &hardware_nis_feedback_buf_);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.lights);
    pass.dispatch(int3(4096 / 64, 1, 1));
    pass.barrier(GPU_BARRIER_SHADER_STORAGE);
  }
  {
    PassSimple &pass = hardware_nis_adam_ps_;
    pass.init();
    pass.shader_set(inst_.shaders.static_shader_get(NIS_ADAM));
    pass.push_constant("hardware_nis_param_count", &hardware_nis_param_count_);
    pass.bind_ssbo("hardware_nis_weights_buf", &hardware_nis_weights_buf_);
    pass.bind_ssbo("hardware_nis_grads_buf", &hardware_nis_grads_buf_);
    pass.bind_ssbo("hardware_nis_train_count_buf", &hardware_nis_train_count_buf_);
    pass.bind_ssbo("hardware_nis_adam_m_buf", &hardware_nis_adam_m_buf_);
    pass.bind_ssbo("hardware_nis_adam_v_buf", &hardware_nis_adam_v_buf_);
    pass.dispatch(int3((hardware_nis_param_count_ + 63) / 64, 1, 1));
    pass.barrier(GPU_BARRIER_SHADER_STORAGE);
  }
  {
    PassSimple &pass = hardware_direct_light_denoise_ps_;
    pass.init();
    pass.shader_set(inst_.shaders.static_shader_get(RAY_HARDWARE_DIRECT_LIGHT_DENOISE));
    pass.push_constant("hardware_direct_light_tile_capacity", &hardware_direct_light_tile_capacity_);
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
    pass.bind_ssbo("hardware_nis_weights_buf", &hardware_nis_weights_buf_);
    pass.bind_ssbo("hardware_light_cluster_weight_buf", &hardware_light_cluster_weight_buf_);
    pass.bind_ssbo("tiles_coord_buf", &hardware_resolve_tiles_buf_);
    pass.bind_texture(RBUFS_UTILITY_TEX_SLOT, inst_.pipelines.utility_tx);
    pass.bind_texture("depth_tx", &depth_tx);
    pass.bind_texture("hardware_rt_shadow_visibility_tx", &hardware_secondary_shadow_visibility_tx_);
    pass.bind_texture("hardware_layered_receiver_rt_shadow_visibility_tx",
                      &hardware_layered_receiver_shadow_visibility_tx_);
    pass.bind_texture("hardware_transmission_receiver_rt_shadow_visibility_tx",
                      &hardware_transmission_receiver_shadow_visibility_tx_);
    pass.bind_texture("radiance_front_tx", &screen_radiance_front_tx_);
    pass.bind_texture("radiance_back_tx", &screen_radiance_back_tx_);
    pass.bind_texture("hit_world_position_tx", &hit_world_position_tx_);
    pass.bind_texture("hit_transmission_layer_tx", &hit_throughput_tx_);
    /* Nuru: declared in the create info (slots 22/23) and read for the Principled metallic
     * coverage; they were never bound here, which Metal masks silently while Vulkan logs a
     * missing-bind error every frame and samples a dummy (coverage 0). Core-promotion
     * candidate: validate the Metal mirror-metal matrix after picking this up. */
    pass.bind_texture("hit_barycentric_tx", &hit_barycentric_tx_);
    pass.bind_texture("layered_receiver_barycentric_tx", &layered_receiver_barycentric_tx_);
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
    pass.bind_texture("hardware_reflected_receiver_gi_tx", &reflected_receiver_gi_tx_);
    pass.bind_texture("hardware_layered_receiver_gi_tx", &layered_receiver_gi_tx_);
    pass.bind_texture("hardware_transmission_receiver_gi_tx", &transmission_receiver_gi_tx_);
    pass.bind_texture("hardware_secondary_photon_gi_tx", &radiance_dummy_black_tx_);
    pass.bind_texture("hardware_layered_secondary_photon_gi_tx", &radiance_dummy_black_tx_);
    pass.bind_texture("hardware_transmission_secondary_photon_gi_tx", &radiance_dummy_black_tx_);
    pass.bind_texture("hardware_direct_light_tx", &hardware_direct_light_denoised_tx_);
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
    PassSimple &pass = shared_indirect_accum_ps_;
    pass.init();
    gpu::Shader *sh = inst_.shaders.static_shader_get(RAY_SHARED_INDIRECT_ACCUM);
    pass.specialize_constant(sh, "closure_index", &data_.closure_index);
    pass.shader_set(sh);
    pass.bind_image("ray_radiance_img", &ray_radiance_tx_);
    pass.bind_image("shared_radiance_img", &shared_indirect_radiance_tx_);
    pass.bind_image("shared_albedo_img", &shared_indirect_albedo_tx_);
    pass.bind_image("shared_normal_img", &shared_indirect_normal_tx_);
    pass.bind_ssbo("tiles_coord_buf", &raytrace_tracing_tiles_buf_);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.gbuffer);
    pass.dispatch(raytrace_tracing_dispatch_buf_);
    pass.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS | GPU_BARRIER_TEXTURE_FETCH);
  }
  {
    PassSimple &pass = shared_indirect_reconstruct_ps_;
    pass.init();
    gpu::Shader *sh = inst_.shaders.static_shader_get(RAY_SHARED_INDIRECT_RECONSTRUCT);
    pass.specialize_constant(sh, "raytrace_resolution_scale", &data_.resolution_scale);
    pass.specialize_constant(sh, "source_is_oidn", &shared_indirect_reconstruct_source_is_oidn_);
    pass.specialize_constant(sh, "active_closure_count", &shared_indirect_active_closure_count_);
    pass.shader_set(sh);
    pass.bind_texture("depth_tx", &depth_tx);
    pass.bind_image("shared_radiance_img", &shared_indirect_reconstruct_source_tx_);
    pass.bind_image("out_radiance_img", &shared_indirect_reconstructed_tx_);
    pass.bind_image("shared_albedo_img", &shared_indirect_albedo_tx_);
    pass.bind_resources(inst_.uniform_data);
    pass.bind_resources(inst_.gbuffer);
    pass.dispatch(&shared_indirect_reconstruct_dispatch_size_);
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
    pass.bind_image("ray_radiance_img", &ray_radiance_denoise_source_tx_);
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
    pass.specialize_constant(sh, "fast_gi_slice_count", fast_gi_ray_count_);
    pass.specialize_constant(sh, "fast_gi_step_count", fast_gi_step_count_);
    pass.specialize_constant(sh, "fast_gi_ao_only", fast_gi_ao_only_);
    pass.shader_set(sh);
    pass.bind_texture("screen_radiance_tx", &downsampled_in_radiance_tx_);
    pass.bind_texture("screen_normal_tx", &downsampled_in_normal_tx_);
    /* Names must match `eevee_horizon_scan` create info exactly: a mismatched name is silently
     * dropped and the shader then writes through unbound image handles (pool garbage composited
     * by the resolve pass; full-frame noise in SCREEN method, green contours with GI off). */
    pass.bind_image("horizon_radiance_0_img", &horizon_radiance_tx_[0]);
    pass.bind_image("horizon_radiance_1_img", &horizon_radiance_tx_[1]);
    pass.bind_image("horizon_radiance_2_img", &horizon_radiance_tx_[2]);
    pass.bind_image("horizon_radiance_3_img", &horizon_radiance_tx_[3]);
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
    /* Names must match `eevee_horizon_denoise` create info exactly (see scan pass note). */
    pass.bind_texture("in_sh_0_tx", &horizon_radiance_tx_[0]);
    pass.bind_texture("in_sh_1_tx", &horizon_radiance_tx_[1]);
    pass.bind_texture("in_sh_2_tx", &horizon_radiance_tx_[2]);
    pass.bind_texture("in_sh_3_tx", &horizon_radiance_tx_[3]);
    pass.bind_texture("screen_normal_tx", &downsampled_in_normal_tx_);
    pass.bind_image("out_sh_0_img", &horizon_radiance_denoised_tx_[0]);
    pass.bind_image("out_sh_1_img", &horizon_radiance_denoised_tx_[1]);
    pass.bind_image("out_sh_2_img", &horizon_radiance_denoised_tx_[2]);
    pass.bind_image("out_sh_3_img", &horizon_radiance_denoised_tx_[3]);
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
    const bool use_bilateral_denoise = !use_hardware_tracing_method() && use_temporal_denoise &&
                                       (ray_tracing_options_.denoise_stages &
                                        RAYTRACE_EEVEE_DENOISE_BILATERAL);

    data_.closure_index = i;
    const float resolution_scale = hardware_interactive_resolution_scale(
        inst_,
        active_hardware_feature_mask(),
        effective_hardware_resolution_scale(active_hardware_feature_mask(),
                                            ray_tracing_options_.resolution_scale,
                                            hardware_reflection_mode_,
                                            hardware_refraction_mode_));
    data_.resolution_scale = raytrace_resolution_scale_numerator(resolution_scale);
    data_.resolution_scale_denominator = raytrace_resolution_scale_denominator(resolution_scale);
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
    const bool warm_scene_final_skip_denoise =
        (active_hardware_feature_mask() & (RAYTRACE_EEVEE_HARDWARE_REFLECTIONS |
                                           RAYTRACE_EEVEE_HARDWARE_REFRACTIONS)) != 0 &&
        use_spatial_denoise && data_.resolution_scale == data_.resolution_scale_denominator;
    if (warm_scene_final_skip_denoise) {
      /* At a 100% trace resolution the scene-final phase disables the spatial blur
       * (`allow_scene_final_spatial_denoise`), so it runs the skip-denoise variant of the
       * spatial pass. */
      const auto gi_skip_denoise = data_.skip_denoise;
      data_.skip_denoise = true;
      inst_.manager->warm_shader_specialization(denoise_spatial_ps_);
      data_.skip_denoise = gi_skip_denoise;
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
  rt_buffer.shared_indirect_oidn_history_tx.release();
  rt_buffer.valid_shared_indirect_oidn_history = false;

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

  const bool history_reset = viewport_history_reset_;
  viewport_history_reset_ = false;
  if (history_reset) {
    /* Viewport resets already invalidate accumulation, but the Hardware RT path also carries
     * per-closure radiance/ownership histories and screen-feedback across frames. If those remain
     * valid after a scene edit, the next traced frame can blend pre-edit state back into updated
     * geometry or lighting. */
    raytrace_history_invalidate_on_viewport_reset(rt_buffer);
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
  if (trace_phase == HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR) {
    /* The scene-final resolve composites the traced radiance at every specular texel and no
     * horizon-scan companion pass runs in this phase. Tile classification must therefore cover
     * the full roughness range; otherwise (e.g. the screen-trace fallback with all HWRT specular
     * features disabled) rough-specular tiles are never traced nor denoise-written and the
     * resolve reads uninitialized pool memory (white tile artifacts). */
    options.trace_max_roughness = 1.0f;
    /* The scene-final phase follows the user's raytracing resolution scale like every other
     * phase. Mirror texture detail is capped at the ray grid below 100%, but that is the user's
     * explicit speed/sharpness trade: a forced full-resolution override here made reflections
     * and refractions (rays + per-sample OIDN at full extent) the dominant frame cost no matter
     * what the resolution setting said. */
  }

  bool use_horizon_scan = enable_horizon_scan && this->use_horizon_scan(options);
  current_hardware_feature_mask_ = (feature_mask_override == UINT32_MAX) ?
                                       filtered_hardware_feature_mask(*this, active_closures) :
                                       feature_mask_override;
  current_trace_active_closures_ = active_closures;

  const float resolution_scale = hardware_interactive_resolution_scale(
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
  const int adaptive_quality_tier = hardware_fast_gi_quality_tier(
      inst_.is_viewport(), 0.0f, fast_gi_scene_analysis, adaptive_scene_priority);
  const int adaptive_budget_rebalance = hardware_fast_gi_budget_rebalance(
      adaptive_quality_tier, adaptive_scene_priority, fast_gi_scene_analysis, inst_.lights.culling_data());
  hardware_debug_view_mode_ = hardware_debug_view_mode();
  hardware_debug_isolate_mode_ = hardware_debug_isolate_mode();
  const int horizon_resolution_scale = max_ii(
      1, power_of_2_max_i(inst_.scene->eevee.fast_gi_resolution));

  const int2 extent = inst_.render_buffers.extent_get();
  const int2 tracing_res = raytrace_tracing_resolution(extent, resolution_scale);
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

  const int active_closure_count = max_ii(1, min_ii(3, to_gbuffer_bin_count(active_closures)));
  /* The raytrace tile masks are bound and copied as three-closure resources in the pass graph and
   * temporal history. Keep their storage shape stable even when only one closure is currently
   * active. */
  constexpr int tile_closure_count = 3;
  gpu::TextureFormat format = gpu::TextureFormat::RAYTRACE_TILEMASK_FORMAT;
  eGPUTextureUsage usage_rw = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;
  const bool caustics_recreated = hardware_caustics_history_tx_.ensure_2d(
      gpu::TextureFormat::SFLOAT_16_16_16_16,
      int2(1),
      usage_rw);
  if (history_reset || caustics_recreated) {
    hardware_caustics_history_tx_.clear(float4(0.0f));
  }
  tile_raytrace_denoise_tx_.ensure_2d_array(format, denoise_tiles, tile_closure_count, usage_rw);
  tile_raytrace_tracing_tx_.ensure_2d_array(format, raytrace_tiles, tile_closure_count, usage_rw);
  /* Kept as 2D array for compatibility with the tile compaction shader. */
  tile_horizon_denoise_tx_.ensure_2d_array(format, denoise_tiles, 1, usage_rw);
  tile_horizon_tracing_tx_.ensure_2d_array(format, raytrace_tiles_horizon, 1, usage_rw);

  tile_raytrace_denoise_tx_.clear(uint4(0u));
  tile_raytrace_tracing_tx_.clear(uint4(0u));
  tile_horizon_denoise_tx_.clear(uint4(0u));
  tile_horizon_tracing_tx_.clear(uint4(0u));

  horizon_tracing_tiles_buf_.resize(ceil_to_multiple_u(raytrace_tile_count_horizon, 512));
  horizon_denoise_tiles_buf_.resize(ceil_to_multiple_u(denoise_tile_count, 512));
  /* Grow-only: the scene-final specular phase traces at full resolution while diffuse GI keeps
   * the configured scale, so the tracing tile count alternates within a frame. Reallocating the
   * capacity buffers twice per frame would churn GPU allocations for no benefit. */
  raytrace_tracing_tiles_buf_.resize(
      max_ii(int(raytrace_tracing_tiles_buf_.size()), int(ceil_to_multiple_u(raytrace_tile_count, 512))));
  raytrace_denoise_tiles_buf_.resize(ceil_to_multiple_u(denoise_tile_count, 512));
  hardware_direct_light_tile_capacity_ = int(
      ceil_to_multiple_u(max_ii(direct_light_tile_count, 1), 512));
  if (!hardware_nis_initialized_) {
    /* Nuru NIS network: sizes from eevee_nuru_nis_mlp_lib.glsl. */
    constexpr int nis_in = 24;
    constexpr int nis_hidden = 32;
    constexpr int nis_out = 32;
    constexpr int w1 = nis_in * nis_hidden;
    constexpr int w2 = nis_hidden * nis_hidden;
    constexpr int w3 = nis_hidden * nis_out;
    hardware_nis_param_count_ = w1 + nis_hidden + w2 + nis_hidden + w3 + nis_out;
    hardware_nis_weights_buf_.resize(ceil_to_multiple_u(hardware_nis_param_count_, 4));
    hardware_nis_grads_buf_.resize(ceil_to_multiple_u(hardware_nis_param_count_, 4));
    hardware_nis_adam_m_buf_.resize(ceil_to_multiple_u(hardware_nis_param_count_, 4));
    hardware_nis_adam_v_buf_.resize(ceil_to_multiple_u(hardware_nis_param_count_, 4));
    /* Deterministic Xavier-ish init for the hidden layers; ZERO output layer so the untrained
     * residual is exactly neutral (m = exp(0) = 1 -> stage-N1 estimator). */
    uint rng_state = 0x9E3779B9u;
    auto rng_signed = [&rng_state]() {
      rng_state = rng_state * 1664525u + 1013904223u;
      return (float(rng_state >> 8u) / float(1u << 24u)) * 2.0f - 1.0f;
    };
    int cursor = 0;
    for (int i = 0; i < w1; i++) {
      hardware_nis_weights_buf_[cursor++] = rng_signed() * 0.25f;
    }
    for (int i = 0; i < nis_hidden; i++) {
      hardware_nis_weights_buf_[cursor++] = 0.0f;
    }
    for (int i = 0; i < w2; i++) {
      hardware_nis_weights_buf_[cursor++] = rng_signed() * 0.18f;
    }
    for (int i = 0; i < nis_hidden; i++) {
      hardware_nis_weights_buf_[cursor++] = 0.0f;
    }
    for (int i = 0; i < w3 + nis_out; i++) {
      hardware_nis_weights_buf_[cursor++] = 0.0f;
    }
    for (int i = 0; i < int(hardware_nis_grads_buf_.size()); i++) {
      hardware_nis_grads_buf_[i] = 0;
    }
    hardware_nis_train_count_buf_.resize(4);
    for (int i = 0; i < 4; i++) {
      hardware_nis_train_count_buf_[i] = 0u;
    }
    for (int i = 0; i < int(hardware_nis_adam_m_buf_.size()); i++) {
      hardware_nis_adam_m_buf_[i] = 0.0f;
      hardware_nis_adam_v_buf_[i] = 0.0f;
    }
    hardware_nis_weights_buf_.push_update();
    hardware_nis_grads_buf_.push_update();
    hardware_nis_train_count_buf_.push_update();
    hardware_nis_adam_m_buf_.push_update();
    hardware_nis_adam_v_buf_.push_update();
    hardware_nis_initialized_ = true;
  }
  if (!hardware_light_cluster_weights_initialized_) {
    /* Nuru NIS: neutral multipliers until the online training (stage N2) writes them. */
    for (int c = 0; c < 32; c++) {
      hardware_light_cluster_weight_buf_[c] = 1.0f;
    }
    hardware_light_cluster_weight_buf_.push_update();
    hardware_light_cluster_weights_initialized_ = true;
  }
  hardware_direct_light_work_tiles_buf_.resize(hardware_direct_light_tile_capacity_);
  hardware_direct_light_visibility_samples_buf_.resize(hardware_direct_light_tile_capacity_);
  /* Grow-only: see `raytrace_tracing_tiles_buf_` above. */
  hardware_trace_tiles_buf_.resize(
      max_ii(int(hardware_trace_tiles_buf_.size()), int(ceil_to_multiple_u(raytrace_tile_count, 512))));
  hardware_resolve_tiles_buf_.resize(
      max_ii(int(hardware_resolve_tiles_buf_.size()), int(ceil_to_multiple_u(raytrace_tile_count, 512))));

  /* Data for tile classification. */
  float roughness_mask_start = options.trace_max_roughness;
  float roughness_mask_fade = 0.2f;
  if (enabled_hardware_specular_mask != 0) {
    roughness_mask_fade = 0.5f;
  }
  data_.roughness_mask_scale = 1.0 / roughness_mask_fade;
  data_.roughness_mask_bias = data_.roughness_mask_scale * roughness_mask_start;

  /* Data for the radiance setup. */
  data_.resolution_scale = raytrace_resolution_scale_numerator(resolution_scale);
  data_.resolution_scale_denominator = raytrace_resolution_scale_denominator(resolution_scale);
  data_.resolution_bias = int2(inst_.sampling.rng_2d_get(SAMPLING_RAYTRACE_V) *
                               data_.resolution_scale);
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
  data_.use_hardware_fast_gi_secondary = use_hardware_fast_gi_secondary();
  data_.hardware_nis_enable = use_hardware_direct_light() ? 1 : 0;
  data_.use_hardware_caustics = false;
  data_.use_hardware_ign_sampling = use_hardware_rt_gi();
  data_.hardware_feature_mask = current_hardware_feature_mask_;
  data_.use_hardware_tracing_method = use_hardware_tracing_method();
  data_.hardware_trace_phase = int(trace_phase);
  data_.hardware_debug_view_mode = hardware_debug_view_mode_;
  data_.hardware_debug_isolate_mode = hardware_debug_isolate_mode_;
  data_.hardware_debug_freeze_updates = 0;
  data_.hardware_direct_light = hardware_direct_light_data(
      inst_.lights.culling_data(),
      hardware_world_sun_light_count(inst_, inst_.lights.culling_data()),
      hardware_uses_viewport_reference(inst_),
      adaptive_quality_tier,
      adaptive_budget_rebalance);
  hardware_fast_gi_quality_tier_ = adaptive_quality_tier;
  hardware_direct_light_sample_count_ = data_.hardware_direct_light.light_samples_per_shading_point;

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
  hardware_direct_light_dispatch_ready_ = false;
  hardware_direct_light_dispatch_extent_ = int2(0);
  hardware_direct_light_dispatch_viewinv_ = float4x4::zero();
  hardware_direct_light_dispatch_wininv_ = float4x4::zero();
  if (use_hardware_tracing()) {
    hardware_direct_light_dispatch_buf_.clear_to_zero();
    inst_.manager->submit(hardware_direct_light_tile_compact_ps_);
    hardware_direct_light_dispatch_ready_ = true;
    hardware_direct_light_dispatch_extent_ = extent;
    hardware_direct_light_dispatch_viewinv_ = render_view.viewinv();
    hardware_direct_light_dispatch_wininv_ = render_view.wininv();
  }

  data_.trace_refraction = screen_radiance_back_tx != nullptr;

  if (has_active_closure && use_shared_oidn_denoise(options)) {
    result = render_shared_oidn(rt_buffer, active_closure_count, options, main_view, render_view);
  }
  else {
    for (int i = 0; i < 3; i++) {
      result.closures[i] = trace(i,
                                 has_active_closure && (active_closure_count > i),
                                 options,
                                 rt_buffer,
                                 main_view,
                                 render_view);
    }
  }

  if (has_active_closure && !result.use_shared_indirect) {
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

void RayTraceModule::render_scene_final_specular(RayTraceBuffer &rt_buffer,
                                                 gpu::Texture *scene_radiance_tx,
                                                 eClosureBits active_closures,
                                                 View &main_view,
                                                 View &render_view)
{
  if (shader_preview_disable_nuru_) {
    return;
  }

  const bool perf_logging_enabled = hardware_perf_logging_enabled();
  const double perf_start_time = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
  if (!use_hardware_tracing()) {
    return;
  }

  const uint32_t feature_mask = filtered_hardware_feature_mask(*this, active_closures) &
                                (RAYTRACE_EEVEE_HARDWARE_REFLECTIONS |
                                 RAYTRACE_EEVEE_HARDWARE_REFRACTIONS);
  if ((feature_mask & (RAYTRACE_EEVEE_HARDWARE_REFLECTIONS |
                       RAYTRACE_EEVEE_HARDWARE_REFRACTIONS)) == 0)
  {
    /* Nuru: no kernel-less fallback here; scene-final specular (including Thin Glass
     * reflections) only resolves with the HWRT specular features enabled. The classic
     * screen-trace fallback was evaluated and removed: with per-sample per-closure OIDN it was
     * sluggish next to the hardware path, and with spatial-only denoising it was too noisy to
     * ship. Re-adding it viably requires routing the kernel-less screen baseline through the
     * shared-OIDN chain (which expects kernel-written guide textures) at hardware-path cost. */
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
  gpu::Shader *sh = inst_.shaders.static_shader_get(RAY_TRACE_SCENE_FINAL_SPECULAR_RESOLVE);
  /* Pass by value: `result` is a stack local and this pass is also replayed by
   * `warm_hardware_tracing_backend()` long after this frame returned. */
  scene_final_specular_resolve_ps_.specialize_constant(
      sh, "use_shared_indirect", result.use_shared_indirect);
  scene_final_specular_resolve_ps_.specialize_constant(
      sh, "scene_final_feature_mask", int(feature_mask));
  scene_final_specular_resolve_ps_.shader_set(sh);
  scene_final_specular_resolve_ps_.bind_texture("indirect_radiance_1_tx", &result.closures[0]);
  scene_final_specular_resolve_ps_.bind_texture("indirect_radiance_2_tx", &result.closures[1]);
  scene_final_specular_resolve_ps_.bind_texture("indirect_radiance_3_tx", &result.closures[2]);
  if (result.use_shared_indirect) {
    scene_final_specular_resolve_ps_.bind_texture("shared_indirect_tx", &result.shared_indirect);
  }
  else {
    scene_final_specular_resolve_ps_.bind_texture("shared_indirect_tx", &result.closures[0]);
  }
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

bool RayTraceModule::use_shared_oidn_denoise(const RaytraceEEVEE &options) const
{
  const bool use_denoise = (options.flag & RAYTRACE_EEVEE_USE_DENOISE);
  if (inst_.is_viewport()) {
    const eViewLayerEEVEEPassType viewport_pass = eViewLayerEEVEEPassType(
        inst_.v3d->shading.render_pass);
    if (!ELEM(viewport_pass,
              EEVEE_RENDER_PASS_COMBINED,
              EEVEE_RENDER_PASS_DIFFUSE_LIGHT,
              EEVEE_RENDER_PASS_SPECULAR_LIGHT))
    {
      return false;
    }
  }
  return use_hardware_tracing() && use_denoise &&
         (options.denoise_filter == RAYTRACE_EEVEE_DENOISE_FILTER_OIDN) &&
         (current_hardware_feature_mask_ != 0);
}

static int denoise_sample_interval_sanitize(const int value)
{
  switch (value) {
    case RAYTRACE_EEVEE_DENOISE_SAMPLE_INTERVAL_1:
    case RAYTRACE_EEVEE_DENOISE_SAMPLE_INTERVAL_2:
    case RAYTRACE_EEVEE_DENOISE_SAMPLE_INTERVAL_4:
    case RAYTRACE_EEVEE_DENOISE_SAMPLE_INTERVAL_8:
    case RAYTRACE_EEVEE_DENOISE_SAMPLE_INTERVAL_16:
      return value;
    default:
      return RAYTRACE_EEVEE_DENOISE_SAMPLE_INTERVAL_1;
  }
}

bool RayTraceModule::use_shared_oidn_this_sample(const RaytraceEEVEE &options) const
{
  const int interval = denoise_sample_interval_sanitize(options.denoise_sample_interval);
  if (interval <= 1) {
    return true;
  }

  const uint64_t sample_number = inst_.is_viewport() ?
                                     inst_.sampling.viewport_sample_index() :
                                     inst_.sampling.sample_index();
  const uint64_t sample_count = inst_.is_viewport() ?
                                    inst_.sampling.viewport_sample_count() :
                                    inst_.sampling.sample_count();
  const bool first_sample = sample_number <= 1 || viewport_history_reset_;
  const bool final_sample = (sample_count > 0) && (sample_number == sample_count);
  return first_sample || final_sample || ((sample_number % uint64_t(interval)) == 0);
}

void RayTraceModule::trace_shared_oidn_closure(int closure_index,
                                               RaytraceEEVEE /*options*/,
                                               RayTraceBuffer &rt_buffer,
                                               View & /*main_view*/,
                                               View &render_view)
{
  eGPUTextureUsage usage_rw = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;
  const int2 tracing_res = shared_indirect_radiance_tx_.size().xy();
  RayTraceBuffer::DenoiseBuffer *denoise_buf = &rt_buffer.closures[closure_index];

  data_.closure_index = closure_index;
  inst_.uniform_data.push_update();

  if (denoise_buf->screen_ownership_history_tx.acquire(
          tracing_res, gpu::TextureFormat::RAYTRACE_VARIANCE_FORMAT, usage_rw) ||
      denoise_buf->valid_screen_ownership_history == false)
  {
    denoise_buf->screen_ownership_history_tx.clear(float4(0.0f));
  }
  screen_ownership_history_tx_ = denoise_buf->screen_ownership_history_tx;
  use_screen_ownership_history_ = false;

  raytrace_tracing_dispatch_buf_.clear_to_zero();
  raytrace_denoise_dispatch_buf_.clear_to_zero();
  inst_.manager->submit(tile_compact_ps_);

  ray_data_tx_.acquire(tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
  ray_time_tx_.acquire(tracing_res, gpu::TextureFormat::RAYTRACE_RAYTIME_FORMAT, usage_rw);
  ray_radiance_tx_.acquire(tracing_res, gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, usage_rw);
  screen_continuation_tx_.acquire(
      tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
  screen_ownership_tx_.acquire(tracing_res, gpu::TextureFormat::RAYTRACE_VARIANCE_FORMAT, usage_rw);
  hit_albedo_tx_.acquire(tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
  reflected_receiver_gi_tx_.acquire(
      tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
  layered_receiver_gi_tx_.acquire(
      tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
  transmission_receiver_gi_tx_.acquire(
      tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
  hit_throughput_tx_.acquire(tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
  hit_material_tx_.acquire(tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
  hit_normal_tx_.acquire(tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
  hit_position_tx_.acquire(tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
  hit_world_position_tx_.acquire(
      tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
  hit_identity_tx_.acquire(tracing_res, gpu::TextureFormat::UINT_32_32_32_32, usage_rw);
  hit_barycentric_tx_.acquire(tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
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
  inst_.manager->submit(shared_indirect_accum_ps_, render_view);

  ray_data_tx_.release();
  ray_time_tx_.release();
  ray_radiance_tx_.release();
  screen_continuation_tx_.release();
  screen_ownership_tx_.release();
  screen_ownership_history_tx_ = nullptr;
  use_screen_ownership_history_ = false;
  denoise_buf->screen_ownership_history_tx.release();
  denoise_buf->valid_screen_ownership_history = false;
  hit_albedo_tx_.release();
  reflected_receiver_gi_tx_.release();
  layered_receiver_gi_tx_.release();
  transmission_receiver_gi_tx_.release();
  hit_throughput_tx_.release();
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
}

RayTraceResult RayTraceModule::render_shared_oidn(RayTraceBuffer &rt_buffer,
                                                  const int active_closure_count,
                                                  RaytraceEEVEE options,
                                                  View &main_view,
                                                  View &render_view)
{
  eGPUTextureUsage usage_rw = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;
  const float resolution_scale = hardware_interactive_resolution_scale(
      inst_,
      current_hardware_feature_mask_,
      effective_hardware_resolution_scale(current_hardware_feature_mask_,
                                          options.resolution_scale,
                                          hardware_reflection_mode_,
                                          hardware_refraction_mode_));
  const int2 extent = inst_.film.render_extent_get();
  const int2 tracing_res = raytrace_tracing_resolution(extent, resolution_scale);

  renderbuf_depth_view_ = inst_.render_buffers.depth_tx;

  RayTraceBuffer::DenoiseBuffer *shared_denoise_buf = &rt_buffer.closures[0];
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
  data_.resolution_scale = raytrace_resolution_scale_numerator(resolution_scale);
  data_.resolution_scale_denominator = raytrace_resolution_scale_denominator(resolution_scale);
  data_.resolution_bias = int2(inst_.sampling.rng_2d_get(SAMPLING_RAYTRACE_V) *
                               data_.resolution_scale);
  data_.denoise_history_persmat = shared_denoise_buf->history_persmat;
  data_.radiance_persmat = (data_.hardware_trace_phase == int(HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR)) ?
                               main_view.persmat() :
                               render_view.persmat();
  data_.full_resolution = extent;
  data_.full_resolution_inv = 1.0f / float2(extent);
  data_.skip_denoise = false;
  data_.closure_index = 0;
  data_.hardware_gi_bounces = hardware_gi_fixed_bounces;
  data_.hardware_gi_mode = int(hardware_gi_mode_);
  data_.hardware_reflection_bounces = effective_hardware_specular_bounces(
      inst_.scene->eevee.ray_tracing_reflection_bounces, hardware_reflection_mode_);
  data_.hardware_refraction_bounces = effective_hardware_specular_bounces(
      inst_.scene->eevee.ray_tracing_refraction_bounces, hardware_refraction_mode_);
  data_.hardware_caustics_samples = max_ii(1, inst_.scene->eevee.ray_tracing_caustics_samples);
  data_.hardware_reflection_mode = int(hardware_reflection_mode_);
  data_.hardware_refraction_mode = int(hardware_refraction_mode_);
  data_.use_hardware_fast_gi_secondary = use_hardware_fast_gi_secondary();
  data_.hardware_nis_enable = use_hardware_direct_light() ? 1 : 0;
  data_.use_hardware_caustics = false;
  data_.use_hardware_ign_sampling = use_hardware_rt_gi();
  data_.hardware_feature_mask = current_hardware_feature_mask_;
  data_.use_hardware_tracing_method = use_hardware_tracing_method();
  inst_.uniform_data.push_update();

  for (int i = 0; i < 3; i++) {
    RayTraceBuffer::DenoiseBuffer &denoise_buf = rt_buffer.closures[i];
    denoise_buf.radiance_history_tx.release();
    denoise_buf.variance_history_tx.release();
    denoise_buf.tilemask_history_tx.free();
    denoise_buf.valid_history = false;
  }

  RayTraceResult result = alloc_only(rt_buffer);
  result.use_shared_indirect = true;

  shared_indirect_radiance_tx_.acquire(
      tracing_res, gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, usage_rw);
  shared_indirect_albedo_tx_.acquire(
      tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
  shared_indirect_normal_tx_.acquire(
      tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
  shared_indirect_radiance_tx_.clear(float4(0.0f));
  shared_indirect_albedo_tx_.clear(float4(0.0f));
  shared_indirect_normal_tx_.clear(float4(0.0f, 0.0f, 1.0f, 0.0f));

  for (int i = 0; i < active_closure_count; i++) {
    trace_shared_oidn_closure(i, options, rt_buffer, main_view, render_view);
  }

  if (rt_buffer.valid_shared_indirect_oidn_history &&
      (!rt_buffer.shared_indirect_oidn_history_tx.is_valid() ||
       rt_buffer.shared_indirect_oidn_history_tx.width() != tracing_res.x ||
       rt_buffer.shared_indirect_oidn_history_tx.height() != tracing_res.y))
  {
    rt_buffer.shared_indirect_oidn_history_tx.release();
    rt_buffer.valid_shared_indirect_oidn_history = false;
  }

  const bool perf_logging_enabled = hardware_perf_logging_enabled();
  const bool scheduled_oidn = use_shared_oidn_this_sample(options);
  const bool run_oidn = scheduled_oidn || !rt_buffer.valid_shared_indirect_oidn_history;
  bool oidn_submitted = false;
  if (run_oidn) {
    shared_indirect_oidn_tx_.acquire(
        tracing_res, gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, usage_rw);
    GPUHardwareRaytraceOIDNDenoiseParams oidn_params;
    oidn_params.input_radiance_tx = shared_indirect_radiance_tx_;
    oidn_params.output_radiance_tx = shared_indirect_oidn_tx_;
    oidn_params.albedo_tx = shared_indirect_albedo_tx_;
    oidn_params.normal_tx = shared_indirect_normal_tx_;
    oidn_params.extent = tracing_res;
    oidn_params.use_albedo = options.denoise_input_passes >=
                             RAYTRACE_EEVEE_DENOISE_INPUT_RGB_ALBEDO;
    oidn_params.use_normal = options.denoise_input_passes >=
                             RAYTRACE_EEVEE_DENOISE_INPUT_RGB_ALBEDO_NORMAL;
    oidn_params.use_gpu = options.denoise_use_gpu != 0;
    oidn_params.use_optix_denoiser = options.denoise_backend ==
                                     RAYTRACE_EEVEE_DENOISE_BACKEND_OPTIX;
    oidn_params.quality = options.denoise_quality;
    oidn_params.prefilter = options.denoise_prefilter;

    const double oidn_start_time = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
    GPU_flush();
    oidn_submitted = GPU_hardware_raytrace_denoise_oidn(oidn_params);
    if (perf_logging_enabled) {
      const double oidn_elapsed_ms = (BLI_time_now_seconds() - oidn_start_time) * 1000.0;
      std::fprintf(stderr,
                   "EEVEE HWRT perf oidn shared=1 closures=%d sample=%llu viewport_sample=%llu "
                   "interval=%d scheduled=%d gpu=%d "
                   "rgb=1 albedo=%d normal=%d prefilter=%d quality=%d tracing_res=%dx%d "
                   "submitted=%d cpu_submit_ms=%.2f\n",
                   active_closure_count,
                   static_cast<unsigned long long>(inst_.sampling.sample_index()),
                   static_cast<unsigned long long>(inst_.sampling.viewport_sample_index()),
                   denoise_sample_interval_sanitize(options.denoise_sample_interval),
                   scheduled_oidn ? 1 : 0,
                   oidn_params.use_gpu ? 1 : 0,
                   oidn_params.use_albedo ? 1 : 0,
                   oidn_params.use_normal ? 1 : 0,
                   oidn_params.prefilter,
                   oidn_params.quality,
                   tracing_res.x,
                   tracing_res.y,
                   oidn_submitted ? 1 : 0,
                   oidn_elapsed_ms);
    }
    if (oidn_submitted) {
      TextureFromPool::swap(shared_indirect_oidn_tx_, rt_buffer.shared_indirect_oidn_history_tx);
      rt_buffer.shared_indirect_oidn_history_tx.retain();
      rt_buffer.valid_shared_indirect_oidn_history = true;
      shared_indirect_oidn_tx_.release();
    }
  }
  else if (perf_logging_enabled) {
    std::fprintf(stderr,
                 "EEVEE HWRT perf oidn_skip shared=1 closures=%d sample=%llu "
                 "viewport_sample=%llu interval=%d history=%d tracing_res=%dx%d\n",
                 active_closure_count,
                 static_cast<unsigned long long>(inst_.sampling.sample_index()),
                 static_cast<unsigned long long>(inst_.sampling.viewport_sample_index()),
                 denoise_sample_interval_sanitize(options.denoise_sample_interval),
                 rt_buffer.valid_shared_indirect_oidn_history ? 1 : 0,
                 tracing_res.x,
                 tracing_res.y);
  }

  shared_indirect_reconstructed_tx_.acquire(
      extent, gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, usage_rw);
  const bool use_oidn_history = rt_buffer.valid_shared_indirect_oidn_history;
  shared_indirect_reconstruct_source_tx_ = use_oidn_history ?
                                              static_cast<gpu::Texture *>(
                                                  rt_buffer.shared_indirect_oidn_history_tx) :
                                              static_cast<gpu::Texture *>(shared_indirect_radiance_tx_);
  shared_indirect_reconstruct_source_is_oidn_ = use_oidn_history;
  shared_indirect_active_closure_count_ = active_closure_count;
  shared_indirect_reconstruct_dispatch_size_ = int3(
      math::divide_ceil(extent, int2(RAYTRACE_GROUP_SIZE)), 1);
  inst_.manager->submit(shared_indirect_reconstruct_ps_, render_view);
  shared_indirect_reconstruct_source_tx_ = nullptr;
  shared_indirect_reconstruct_source_is_oidn_ = false;
  shared_indirect_active_closure_count_ = 0;

  result.shared_indirect = {shared_indirect_reconstructed_tx_};
  shared_indirect_radiance_tx_.release();
  shared_indirect_oidn_tx_.release();
  shared_indirect_albedo_tx_.release();
  shared_indirect_normal_tx_.release();

  return result;
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

  const float resolution_scale = hardware_interactive_resolution_scale(
      inst_,
      current_hardware_feature_mask_,
      effective_hardware_resolution_scale(current_hardware_feature_mask_,
                                          options.resolution_scale,
                                          hardware_reflection_mode_,
                                          hardware_refraction_mode_));

  const int2 extent = inst_.film.render_extent_get();
  const int2 tracing_res = raytrace_tracing_resolution(extent, resolution_scale);

  renderbuf_depth_view_ = inst_.render_buffers.depth_tx;

  const bool scene_final_specular_phase = (data_.hardware_trace_phase ==
                                           int(HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR));
  const bool use_denoise = (options.flag & RAYTRACE_EEVEE_USE_DENOISE);
  const bool allow_scene_final_spatial_denoise = !scene_final_specular_phase ||
                                                (raytrace_resolution_scale_numerator(
                                                     resolution_scale) >
                                                 raytrace_resolution_scale_denominator(
                                                     resolution_scale));
  /* OIDN guide textures (`hit_albedo_tx_`, `hit_normal_tx_`) are only cleared and written when
   * the hardware trace kernel actually runs. With an empty feature mask (e.g. the PRECOMBINE
   * phase with hardware GI disabled, or the Thin Glass screen-trace fallback in the scene-final
   * phase) `submit_hardware_tracing_backend()` early-outs right after the screen baseline, so
   * the guides keep stale texture-pool contents; feeding those to OIDN corrupts the whole
   * closure radiance. Such phases still get denoised: OIDN runs guide-less (RGB only, see
   * `use_oidn_guides_` below) and the classic spatial denoiser is force-allowed so rough
   * screen-traced reflections do not ship raw GGX noise at 100% trace resolution. */
  const bool hardware_kernel_will_run = (current_hardware_feature_mask_ != 0) &&
                                        (hardware_scene_entry_count_ > 0);
  const bool use_spatial_denoise = (allow_scene_final_spatial_denoise ||
                                    !hardware_kernel_will_run) &&
                                   (options.denoise_stages & RAYTRACE_EEVEE_DENOISE_SPATIAL) &&
                                   use_denoise;
  const bool use_temporal_denoise = !scene_final_specular_phase &&
                                    (options.denoise_stages & RAYTRACE_EEVEE_DENOISE_TEMPORAL) &&
                                    use_spatial_denoise;
  /* OIDN only when the hardware kernel actually ran. Kernel-less phases (GI off, the
   * scene-final screen-trace fallback) use the classic spatial denoise chain instead:
   * per-closure guide-less OIDN costs roughly 12 ms/sample and made the fallback path feel
   * sluggish next to the hardware path, while the force-allowed spatial denoiser above keeps
   * rough screen-traced reflections acceptable at interactive cost. */
  const bool use_oidn_denoise = use_denoise && use_hardware_tracing_method() &&
                                hardware_kernel_will_run;
  use_oidn_guides_ = hardware_kernel_will_run;
  const bool use_bilateral_denoise = !use_hardware_tracing_method() && !scene_final_specular_phase &&
                                     use_temporal_denoise &&
                                     (options.denoise_stages &
                                      RAYTRACE_EEVEE_DENOISE_BILATERAL);

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

  data_.resolution_scale = raytrace_resolution_scale_numerator(resolution_scale);
  data_.resolution_scale_denominator = raytrace_resolution_scale_denominator(resolution_scale);
  data_.resolution_bias = int2(inst_.sampling.rng_2d_get(SAMPLING_RAYTRACE_V) *
                               data_.resolution_scale);
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
  data_.use_hardware_fast_gi_secondary = use_hardware_fast_gi_secondary();
  data_.hardware_nis_enable = use_hardware_direct_light() ? 1 : 0;
  data_.use_hardware_caustics = false;
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
    reflected_receiver_gi_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    layered_receiver_gi_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
    transmission_receiver_gi_tx_.acquire(
        tracing_res, gpu::TextureFormat::SFLOAT_16_16_16_16, usage_rw);
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

  if (use_oidn_denoise) {
    ray_radiance_oidn_tx_.acquire(
        tracing_res, gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, usage_rw);
    /* Guide-less when the hardware kernel did not run this phase: `hit_albedo_tx_` /
     * `hit_normal_tx_` only contain valid data when the kernel writes them. */
    const bool use_oidn_albedo = use_oidn_guides_ &&
                                 options.denoise_input_passes >=
                                     RAYTRACE_EEVEE_DENOISE_INPUT_RGB_ALBEDO;
    const bool use_oidn_normal = use_oidn_guides_ &&
                                 options.denoise_input_passes >=
                                     RAYTRACE_EEVEE_DENOISE_INPUT_RGB_ALBEDO_NORMAL;

    GPUHardwareRaytraceOIDNDenoiseParams oidn_params;
    oidn_params.input_radiance_tx = ray_radiance_tx_;
    oidn_params.output_radiance_tx = ray_radiance_oidn_tx_;
    oidn_params.albedo_tx = hit_albedo_tx_;
    oidn_params.normal_tx = hit_normal_tx_;
    oidn_params.extent = tracing_res;
    oidn_params.use_albedo = use_oidn_albedo;
    oidn_params.use_normal = use_oidn_normal;
    oidn_params.use_gpu = options.denoise_use_gpu != 0;
    oidn_params.use_optix_denoiser = options.denoise_backend ==
                                     RAYTRACE_EEVEE_DENOISE_BACKEND_OPTIX;
    oidn_params.quality = options.denoise_quality;
    oidn_params.prefilter = options.denoise_prefilter;

    const bool perf_logging_enabled = hardware_perf_logging_enabled();
    const double oidn_start_time = perf_logging_enabled ? BLI_time_now_seconds() : 0.0;
    GPU_flush();
    const bool oidn_submitted = GPU_hardware_raytrace_denoise_oidn(oidn_params);
    if (perf_logging_enabled) {
      const double oidn_elapsed_ms = (BLI_time_now_seconds() - oidn_start_time) * 1000.0;
      std::fprintf(stderr,
                   "EEVEE HWRT perf oidn closure=%d gpu=%d rgb=%d albedo=%d normal=%d "
                   "prefilter=%d quality=%d tracing_res=%dx%d submitted=%d cpu_submit_ms=%.2f\n",
                   closure_index,
                   oidn_params.use_gpu ? 1 : 0,
                   1,
                   use_oidn_albedo ? 1 : 0,
                   use_oidn_normal ? 1 : 0,
                   oidn_params.prefilter,
                   oidn_params.quality,
                   tracing_res.x,
                   tracing_res.y,
                   oidn_submitted ? 1 : 0,
                   oidn_elapsed_ms);
    }
    if (oidn_submitted) {
      ray_radiance_denoise_source_tx_ = ray_radiance_oidn_tx_;
    }
  }

  RayTraceResultTexture result;

  /* Spatial denoise pass is required to resolve at least one ray per pixel. */
  {
    if (ray_radiance_denoise_source_tx_ == nullptr) {
      ray_radiance_denoise_source_tx_ = ray_radiance_tx_;
    }
    denoise_buf->denoised_spatial_tx.acquire(
        extent, gpu::TextureFormat::RAYTRACE_RADIANCE_FORMAT, usage_rw);
    hit_variance_tx_.acquire(use_temporal_denoise ? extent : int2(1),
                             gpu::TextureFormat::RAYTRACE_VARIANCE_FORMAT);
    hit_depth_tx_.acquire(use_temporal_denoise ? extent : int2(1), gpu::TextureFormat::SFLOAT_32);
    denoised_spatial_tx_ = denoise_buf->denoised_spatial_tx;

    inst_.manager->submit(denoise_spatial_ps_, render_view);

    result = {denoise_buf->denoised_spatial_tx};
  }
  ray_radiance_denoise_source_tx_ = nullptr;
  ray_radiance_oidn_tx_.release();

  ray_data_tx_.release();
  ray_time_tx_.release();
  ray_radiance_tx_.release();
  screen_continuation_tx_.release();
  hit_material_tx_.release();
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
  reflected_receiver_gi_tx_.release();
  layered_receiver_gi_tx_.release();
  transmission_receiver_gi_tx_.release();
  hit_throughput_tx_.release();
  hit_normal_tx_.release();

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
    denoise_buf->denoised_bilateral_tx.clear(float4(0.0f));
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
