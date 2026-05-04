/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/**
 * Shared code between host and client code-bases.
 */

#pragma once

#include "GPU_shader_shared_utils.hh"

#ifndef GPU_SHADER
namespace blender::eevee {
#endif

enum [[host_shared]] eHardwareDirectLightSelectionMode : uint32_t {
  HWRT_DIRECT_LIGHT_SELECTION_TILE = 0u,
  HWRT_DIRECT_LIGHT_SELECTION_PIXEL = 1u,
};

enum [[host_shared]] eHardwareDebugViewMode : uint32_t {
  HWRT_DEBUG_VIEW_NONE = 0u,
  HWRT_DEBUG_VIEW_RADIANCE = 1u,
  HWRT_DEBUG_VIEW_OCCUPANCY_THICKNESS = 2u,
  HWRT_DEBUG_VIEW_CONFIDENCE = 3u,
  HWRT_DEBUG_VIEW_INVALID_BRICKS = 4u,
  HWRT_DEBUG_VIEW_LEAK_RISK = 5u,
  HWRT_DEBUG_VIEW_DIRECT_LIGHT = 6u,
};

enum [[host_shared]] eHardwareDebugIsolateMode : uint32_t {
  HWRT_DEBUG_ISOLATE_NONE = 0u,
  HWRT_DEBUG_ISOLATE_DIRECT = 1u,
  HWRT_DEBUG_ISOLATE_INDIRECT = 2u,
};

enum [[host_shared]] eHardwareTracePhase : uint32_t {
  HWRT_TRACE_PHASE_FULL = 0u,
  HWRT_TRACE_PHASE_PRECOMBINE = 1u,
  HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR = 2u,
};

struct [[host_shared]] HardwareDirectLightData {
  /** Candidate lights are selected from the existing Eevee light-culling tiles. */
  uint selection_mode;
  /** Tile size inherited from Eevee light culling. */
  uint tile_size_px;
  /** Number of bitmap words describing the local-light subset for one tile. */
  uint tile_word_len;
  /** Number of local lights considered by the tile candidate stage before stochastic picking. */
  uint candidate_local_lights_len;
  /** Total local lights in the current culling input. */
  uint local_lights_len;
  /** Directional / sun lights that bypass the local tile candidate bitmap. */
  uint sun_lights_len;
  /** Fixed stochastic light samples evaluated per shading point before denoise. */
  uint light_samples_per_shading_point;
  /** Directional/sun lighting is handled outside the bounded local-light tile budget. */
  bool32_t trace_sun_lights_separately;
  /** Mesh emissives are excluded until they have an explicit sampled-emitter representation. */
  bool32_t sample_emissive_meshes;
  /** Baseline importance weight for analytic local lights. */
  float local_light_importance_scale;
  /** Extra importance weight reserved for area-light candidates. */
  float area_light_importance_scale;
  /** Reserved extra importance weight for textured/light-profile candidates. */
  float textured_light_importance_scale;
  /** Separate directional/sun importance scale for the dedicated sun path. */
  float sun_light_importance_scale;
  /** Leading directional entries reserved for extracted world suns. */
  uint world_sun_lights_len;
  int _pad1;
  int _pad2;
};

struct [[host_shared]] HardwareDirectLightWorkTile {
  /** Tile coordinate in light-culling space packed as `packUvec2x16(tile_coord)`. */
  uint packed_tile_coord;
  /** Offset into Eevee's existing `light_tile_buf` bitmap for this tile. */
  uint candidate_word_offset;
  /** Number of bitmap words that describe this tile's local-light candidate subset. */
  uint candidate_word_count;
  /** Bounded stochastic light samples reserved for this tile. */
  uint sample_budget;
};

struct [[host_shared]] HardwareDirectLightVisibilitySample {
  /** Source tile in light-culling space packed as `packUvec2x16(tile_coord)`. */
  uint packed_tile_coord;
  /** Full-resolution texel used to read RT visibility for this tile sample. */
  uint packed_sample_texel;
  /** Selected local-light index in Eevee light-buffer order, or `0xFFFFFFFFu` if unused. */
  uint local_light_index;
  /** Selected sun-light index relative to the sun range, or `0xFFFFFFFFu` if unused. */
  uint sun_light_index;
  /** RT visibility for the selected local-light sample. */
  float local_visibility;
  /** Importance used when selecting the local-light sample. */
  float local_importance;
  /** RT visibility for the selected sun-light sample. */
  float sun_visibility;
  /** Importance used when selecting the sun-light sample. */
  float sun_importance;
};

struct [[host_shared]] RayTraceData {
  /** ViewProjection matrix used to render the previous frame. */
  float4x4 history_persmat;
  /** ViewProjection matrix used to render the radiance texture. */
  float4x4 radiance_persmat;
  /** ViewProjection matrix used to denoise the previous frame. */
  float4x4 denoise_history_persmat;
  /** Input resolution. */
  int2 full_resolution;
  /** Inverse of input resolution to get screen UVs. */
  float2 full_resolution_inv;
  /** Scale and bias to go from ray-trace resolution to input resolution. */
  int2 resolution_bias;
  int resolution_scale;
  /** View space thickness the objects. */
  float thickness;
  /** Scale and bias to go from horizon-trace resolution to input resolution. */
  int2 horizon_resolution_bias;
  int horizon_resolution_scale;
  /** Determine how fast the sample steps are getting bigger. */
  float quality;
  /** Maximum roughness for which we will trace a ray. */
  float roughness_mask_scale;
  float roughness_mask_bias;
  /** If set to true will bypass spatial denoising. */
  bool32_t skip_denoise;
  /** If set to false will bypass tracing for refractive closures. */
  bool32_t trace_refraction;
  /** Closure being ray-traced. */
  int closure_index;
  /** Bounce limit for the experimental hardware GI path. */
  int hardware_gi_bounces;
  /** Layering mode for the experimental hardware GI path. */
  int hardware_gi_mode;
  /** Bounce limit for the experimental hardware reflection path. */
  int hardware_reflection_bounces;
  /** Bounce limit for the experimental hardware refraction path. */
  int hardware_refraction_bounces;
  /** Adaptive sample budget for receiver-side caustics refinement. */
  int hardware_caustics_samples;
  /** User-selected ownership mode for specular reflections. */
  int hardware_reflection_mode;
  /** User-selected ownership mode for specular refractions. */
  int hardware_refraction_mode;
  /** Use the dedicated cascaded Hardware RT fast indirect-light field. */
  bool32_t use_hardware_fast_gi;
  /** Dedicated Hardware RT fast-GI field contains valid world-space data this frame. */
  bool32_t use_hardware_fast_gi_field;
  /** Add the adaptive receiver-side caustics term in Hardware RT lighting. */
  bool32_t use_hardware_caustics;
  /** Use screen-trace-like IGN sampling for hardware GI ray generation. */
  bool32_t use_hardware_ign_sampling;
  /** Active Hardware RT feature bits for the current deferred layer. */
  uint hardware_feature_mask;
  /** The current tracing method is the Hardware RT path. */
  bool32_t use_hardware_tracing_method;
  /** Resolution of each world-space Fast GI cascade grid. */
  int hardware_fast_gi_grid_resolution;
  /** Number of active world-space Fast GI cascades. */
  int hardware_fast_gi_cascade_count;
  /** Developer-only full-screen debug visualization mode. */
  int hardware_debug_view_mode;
  /** Developer-only direct/indirect isolation mode. */
  int hardware_debug_isolate_mode;
  /** Freeze Fast GI field updates while keeping the previous field visible. */
  int hardware_debug_freeze_updates;
  /** Selects whether the current tracing pass is full, pre-combine GI, or late specular only. */
  int hardware_trace_phase;
  /** xyz = cascade center in world space, w = voxel size. */
  float4 hardware_fast_gi_cascade_config[3];
  /** Shared contract for the future many-light Hardware RT direct path. */
  struct HardwareDirectLightData hardware_direct_light;
};

enum [[host_shared]] eHardwareHitEvalFlag : uint32_t {
  HIT_EVAL_FLAG_FRONT_FACING = 1u << 0,
};

struct [[host_shared]] HardwareHitEvalRecord {
  uint packed_texel;
  uint resource_id_raw;
  uint primitive_id;
  uint flags;

  float2 barycentric_coords;
  float2 _pad0;

  packed_float3 view_origin;
  float _pad1;
};

struct [[host_shared]] HardwareTraceDebugCounters {
  uint hybrid_retrace_reject_count;
  uint _pad0;
  uint _pad1;
  uint _pad2;
};

struct [[host_shared]] AOData {
  float2 pixel_size;
  float distance;
  float lod_factor;

  float thickness_near;
  float thickness_far;
  float angle_bias;
  float gi_distance;

  float lod_factor_ao;
  float _pad0;
  float _pad1;
  float _pad2;
};

#ifndef GPU_SHADER
}  // namespace blender::eevee
#endif
