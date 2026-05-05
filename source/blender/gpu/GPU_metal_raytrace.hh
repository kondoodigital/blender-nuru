/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#pragma once

#include "BLI_math_matrix_types.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"
#include "BLI_sys_types.h"

namespace blender {

namespace gpu {
class Batch;
class StorageBuf;
class Texture;
}

enum GPUMetalRaytraceMaterialEvalPolicy : uint32_t {
  /** Direct-light visibility and shadow rays never replay materials at the occluder hit. */
  GPU_METAL_RAYTRACE_MATERIAL_EVAL_DIRECT_VISIBILITY_ONLY = 0u,
  /** Traversal and continuation use only the sync-time material proxy stored per scene entry. */
  GPU_METAL_RAYTRACE_MATERIAL_EVAL_PROXY_CONTINUATION = 1u,
  /** Full Eevee material replay is reserved for compacted final hit records only. */
  GPU_METAL_RAYTRACE_MATERIAL_EVAL_COMPACT_HIT_REPLAY = 2u,
};

enum GPUMetalRaytraceMaterialProxySet : uint32_t {
  /** Indirect diffuse GI only needs emissive radiance plus coarse diffuse albedo. */
  GPU_METAL_RAYTRACE_PROXY_INDIRECT_DIFFUSE = 0u,
  /** Direct/specular fallback needs one dominant closure family plus tint/roughness/IOR. */
  GPU_METAL_RAYTRACE_PROXY_DIRECT_AND_SPECULAR = 1u,
};

struct GPUMetalRaytraceSceneEntry {
  gpu::Batch *batch = nullptr;
  float4x4 object_to_world = float4x4::identity();
  uint32_t instance_count = 1;
  uint32_t user_id = 0;
  /** Indirect diffuse proxy set. See #GPUMetalRaytraceMaterialProxySet. */
  float3 emissive_radiance = float3(0.0f);
  float3 diffuse_albedo = float3(0.8f);
  /** Direct/specular proxy set. See #GPUMetalRaytraceMaterialProxySet. */
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
  int material_slot = -1;
  bool is_sculpt = false;
};

struct GPUMetalRaytraceTraceParams {
  /** Full replay is a later sparse stage; traversal itself stays on proxy-only material data. */
  gpu::Texture *ray_data_tx = nullptr;
  gpu::Texture *depth_tx = nullptr;
  gpu::Texture *gbuf_header_tx = nullptr;
  gpu::Texture *gbuf_normal_tx = nullptr;
  gpu::Texture *screen_continuation_tx = nullptr;
  gpu::Texture *world_probe_tx = nullptr;
  gpu::Texture *ray_time_tx = nullptr;
  gpu::Texture *ray_radiance_tx = nullptr;
  gpu::Texture *hit_albedo_tx = nullptr;
  gpu::Texture *hit_throughput_tx = nullptr;
  gpu::Texture *hit_material_tx = nullptr;
  gpu::Texture *hit_normal_tx = nullptr;
  gpu::Texture *hit_position_tx = nullptr;
  gpu::Texture *hit_world_position_tx = nullptr;
  gpu::Texture *hit_identity_tx = nullptr;
  gpu::Texture *hit_barycentric_tx = nullptr;
  gpu::Texture *layered_receiver_ray_time_tx = nullptr;
  gpu::Texture *layered_receiver_ray_radiance_tx = nullptr;
  gpu::Texture *layered_receiver_albedo_tx = nullptr;
  gpu::Texture *layered_receiver_throughput_tx = nullptr;
  gpu::Texture *layered_receiver_material_tx = nullptr;
  gpu::Texture *layered_receiver_normal_tx = nullptr;
  gpu::Texture *layered_receiver_position_tx = nullptr;
  gpu::Texture *layered_receiver_world_position_tx = nullptr;
  gpu::Texture *layered_receiver_identity_tx = nullptr;
  gpu::Texture *layered_receiver_barycentric_tx = nullptr;
  gpu::Texture *transmission_receiver_ray_time_tx = nullptr;
  gpu::Texture *transmission_receiver_ray_radiance_tx = nullptr;
  gpu::Texture *transmission_receiver_albedo_tx = nullptr;
  gpu::Texture *transmission_receiver_throughput_tx = nullptr;
  gpu::Texture *transmission_receiver_material_tx = nullptr;
  gpu::Texture *transmission_receiver_normal_tx = nullptr;
  gpu::Texture *transmission_receiver_position_tx = nullptr;
  gpu::Texture *transmission_receiver_world_position_tx = nullptr;
  gpu::Texture *transmission_receiver_identity_tx = nullptr;
  gpu::Texture *transmission_receiver_barycentric_tx = nullptr;
  gpu::StorageBuf *dispatch_buf = nullptr;
  gpu::StorageBuf *tiles_coord_buf = nullptr;
  float4x4 viewinv = float4x4::identity();
  float4x4 wininv = float4x4::identity();
  int2 full_resolution = int2(1);
  int resolution_scale = 1;
  int closure_index = 0;
  uint32_t feature_mask = 0;
  int hardware_trace_phase = 0;
  int reflection_bounces = 1;
  int refraction_bounces = 1;
  int2 resolution_bias = int2(0);
  float clamp_indirect = 1.0e10f;
  float4 world_probe_atlas_coord = float4(0.0f, 0.0f, 0.0f, -1.0f);
  bool use_environment = false;
  float4 sampling_rand = float4(0.0f);
};

struct GPUMetalRaytraceDirectionalShadowParams {
  gpu::Texture *depth_tx = nullptr;
  gpu::Texture *gbuf_header_tx = nullptr;
  gpu::Texture *gbuf_normal_tx = nullptr;
  gpu::Texture *shadow_visibility_tx = nullptr;
  gpu::StorageBuf *world_sunlight_direction_buf = nullptr;
  float4x4 viewinv = float4x4::identity();
  float4x4 wininv = float4x4::identity();
  int2 full_resolution = int2(1);
  int shadow_layer = 0;
  int world_sun_slot = -1;
  float3 light_direction = float3(0.0f, 0.0f, 1.0f);
  float normal_bias = 1.0e-3f;
  float shadow_angle = 0.0f;
  int sample_count = 1;
  float4 sampling_rand = float4(0.0f);
};

struct GPUMetalRaytraceDirectionalHitShadowParams {
  gpu::Texture *hit_normal_tx = nullptr;
  gpu::Texture *hit_world_position_tx = nullptr;
  gpu::Texture *hit_identity_tx = nullptr;
  gpu::Texture *shadow_visibility_tx = nullptr;
  gpu::StorageBuf *dispatch_buf = nullptr;
  gpu::StorageBuf *tiles_coord_buf = nullptr;
  gpu::StorageBuf *world_sunlight_direction_buf = nullptr;
  int2 tracing_resolution = int2(1);
  int shadow_layer = 0;
  int world_sun_slot = -1;
  float3 light_direction = float3(0.0f, 0.0f, 1.0f);
  float normal_bias = 1.0e-3f;
  float shadow_angle = 0.0f;
  int sample_count = 1;
  float4 sampling_rand = float4(0.0f);
};

struct GPUMetalRaytraceLocalShadowParams {
  gpu::Texture *depth_tx = nullptr;
  gpu::Texture *gbuf_header_tx = nullptr;
  gpu::Texture *gbuf_normal_tx = nullptr;
  gpu::Texture *shadow_visibility_tx = nullptr;
  float4x4 viewinv = float4x4::identity();
  float4x4 wininv = float4x4::identity();
  int2 full_resolution = int2(1);
  int shadow_layer = 0;
  uint32_t light_type = 0;
  float3 light_position = float3(0.0f);
  float shadow_radius = 0.0f;
  float3 light_x_axis = float3(1.0f, 0.0f, 0.0f);
  float area_size_x = 0.0f;
  float3 light_y_axis = float3(0.0f, 1.0f, 0.0f);
  float area_size_y = 0.0f;
  float3 shadow_offset = float3(0.0f);
  float area_shadow_scale = 1.0f;
  float normal_bias = 1.0e-3f;
  int sample_count = 1;
  float4 sampling_rand = float4(0.0f);
};

struct GPUMetalRaytraceLocalHitShadowParams {
  gpu::Texture *hit_normal_tx = nullptr;
  gpu::Texture *hit_world_position_tx = nullptr;
  gpu::Texture *hit_identity_tx = nullptr;
  gpu::Texture *shadow_visibility_tx = nullptr;
  gpu::StorageBuf *dispatch_buf = nullptr;
  gpu::StorageBuf *tiles_coord_buf = nullptr;
  int2 tracing_resolution = int2(1);
  int shadow_layer = 0;
  uint32_t light_type = 0;
  float3 light_position = float3(0.0f);
  float shadow_radius = 0.0f;
  float3 light_x_axis = float3(1.0f, 0.0f, 0.0f);
  float area_size_x = 0.0f;
  float3 light_y_axis = float3(0.0f, 1.0f, 0.0f);
  float area_size_y = 0.0f;
  float3 shadow_offset = float3(0.0f);
  float area_shadow_scale = 1.0f;
  float normal_bias = 1.0e-3f;
  int sample_count = 1;
  float4 sampling_rand = float4(0.0f);
};

struct GPUMetalRaytraceEnvironmentVisibilityParams {
  gpu::Texture *depth_tx = nullptr;
  gpu::Texture *gbuf_header_tx = nullptr;
  gpu::Texture *gbuf_normal_tx = nullptr;
  gpu::Texture *environment_visibility_tx = nullptr;
  float4x4 viewinv = float4x4::identity();
  float4x4 wininv = float4x4::identity();
  int2 full_resolution = int2(1);
  int sample_count = 1;
  float normal_bias = 1.0e-3f;
  float4 sampling_rand = float4(0.0f);
};

struct GPUMetalRaytraceHitEnvironmentVisibilityParams {
  gpu::Texture *hit_normal_tx = nullptr;
  gpu::Texture *hit_world_position_tx = nullptr;
  gpu::Texture *environment_visibility_tx = nullptr;
  gpu::StorageBuf *dispatch_buf = nullptr;
  gpu::StorageBuf *tiles_coord_buf = nullptr;
  int2 tracing_resolution = int2(1);
  int sample_count = 1;
  float normal_bias = 1.0e-3f;
  float4 sampling_rand = float4(0.0f);
};

/* Compact light payload for Fast GI direct-light estimation.
 * Keep the layout float4-only so the CPU writer and Metal kernel can share it without depending on
 * Eevee light headers in the GPU module. */
struct GPUMetalRaytraceFastGILightRecord {
  float4 object_to_world_x = float4(1.0f, 0.0f, 0.0f, 0.0f);
  float4 object_to_world_y = float4(0.0f, 1.0f, 0.0f, 0.0f);
  float4 object_to_world_z = float4(0.0f, 0.0f, 1.0f, 0.0f);
  float4 color_diffuse_power = float4(0.0f);
  float4 direction_type = float4(0.0f, 0.0f, 1.0f, 0.0f);
  float4 attenuation_spot = float4(0.0f);
  float4 spot_size_inv = float4(0.0f);
};

struct GPUMetalRaytraceFastGIParams {
  gpu::Texture *fast_gi_history_tx = nullptr;
  gpu::Texture *fast_gi_tx = nullptr;
  gpu::Texture *fast_gi_error_tx = nullptr;
  gpu::Texture *fast_gi_visibility_tx = nullptr;
  gpu::Texture *world_probe_tx = nullptr;
  gpu::StorageBuf *light_buf = nullptr;
  int grid_resolution = 1;
  int3 brick_origin = int3(0);
  int3 brick_extent = int3(1);
  int cascade_index = 0;
  int cascade_count = 1;
  int sample_count = 6;
  int gi_bounces = 1;
  int light_count = 0;
  int light_sample_count = 1;
  float normal_bias = 1.0e-3f;
  bool reuse_history = true;
  bool use_environment = true;
  float4 sampling_rand = float4(0.0f);
  float4 world_probe_atlas_coord = float4(0.0f, 0.0f, 0.0f, -1.0f);
  float4 cascade_config[3] = {float4(0.0f), float4(0.0f), float4(0.0f)};
};

struct GPUMetalRaytraceReflectedReceiverGIParams {
  gpu::Texture *receiver_gi_tx = nullptr;
  gpu::Texture *world_probe_tx = nullptr;
  gpu::StorageBuf *light_buf = nullptr;
  gpu::StorageBuf *dispatch_buf = nullptr;
  gpu::StorageBuf *tiles_coord_buf = nullptr;
  gpu::Texture *ray_time_tx = nullptr;
  gpu::Texture *hit_albedo_tx = nullptr;
  gpu::Texture *hit_material_tx = nullptr;
  gpu::Texture *hit_normal_tx = nullptr;
  gpu::Texture *hit_world_position_tx = nullptr;
  int2 tracing_resolution = int2(1);
  int resolution_divisor = 4;
  int sample_count = 8;
  int light_count = 0;
  int light_sample_count = 2;
  float normal_bias = 1.0e-3f;
  bool use_environment = true;
  float4 sampling_rand = float4(0.0f);
  float4 world_probe_atlas_coord = float4(0.0f, 0.0f, 0.0f, -1.0f);
};

struct GPUMetalRaytraceSceneStats {
  int geometry_count = 0;
  int instance_count = 0;
  int built_blas_count = 0;
  int emissive_light_count = 0;
  float emissive_energy_sum = 0.0f;
  bool built_scene = false;
};

struct GPUMetalRaytraceSceneUpdateParams {
  bool update_tlas = true;
  bool update_emissive_data = true;
  bool update_material_data = true;
  bool update_world_geometry_data = true;
};

struct GPUMetalRaytraceScene;

GPUMetalRaytraceScene *GPU_metal_raytrace_scene_build(
    Span<GPUMetalRaytraceSceneEntry> entries, GPUMetalRaytraceSceneStats *r_stats = nullptr);
bool GPU_metal_raytrace_scene_update(GPUMetalRaytraceScene *scene,
                                     Span<GPUMetalRaytraceSceneEntry> entries,
                                     const GPUMetalRaytraceSceneUpdateParams &update_params,
                                     GPUMetalRaytraceSceneStats *r_stats = nullptr);
bool GPU_metal_raytrace_scene_trace(GPUMetalRaytraceScene *scene,
                                    const GPUMetalRaytraceTraceParams &params);
bool GPU_metal_raytrace_scene_trace_directional_shadow(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceDirectionalShadowParams &params);
bool GPU_metal_raytrace_scene_trace_local_shadow(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceLocalShadowParams &params);
bool GPU_metal_raytrace_scene_trace_directional_hit_shadow(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceDirectionalHitShadowParams &params);
bool GPU_metal_raytrace_scene_trace_local_hit_shadow(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceLocalHitShadowParams &params);
bool GPU_metal_raytrace_scene_shadow_batch_begin(GPUMetalRaytraceScene *scene);
bool GPU_metal_raytrace_scene_shadow_batch_end(GPUMetalRaytraceScene *scene);
bool GPU_metal_raytrace_scene_trace_environment_visibility(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceEnvironmentVisibilityParams &params);
bool GPU_metal_raytrace_scene_trace_hit_environment_visibility(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceHitEnvironmentVisibilityParams &params);
bool GPU_metal_raytrace_scene_trace_fast_gi(GPUMetalRaytraceScene *scene,
                                            const GPUMetalRaytraceFastGIParams &params);
bool GPU_metal_raytrace_scene_trace_reflected_receiver_gi(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceReflectedReceiverGIParams &params);
void GPU_metal_raytrace_scene_free(GPUMetalRaytraceScene *scene);

}  // namespace blender
