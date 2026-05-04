/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/**
 * Compact the original tracing tiles down to only the ones that still need the Metal Hardware
 * trace after screen / planar ownership has already been resolved.
 */

#include "infos/eevee_tracing_infos.hh"

COMPUTE_SHADER_CREATE_INFO(eevee_ray_hardware_trace_tile_compact)

#include "eevee_gbuffer_read_lib.glsl"
#include "eevee_sampling_lib.glsl"
#include "gpu_shader_codegen_lib.glsl"
#include "gpu_shader_math_vector_lib.glsl"
#include "gpu_shader_utildefines_lib.glsl"

#ifdef GLSL_CPP_STUBS
constexpr uint GBUF_NONE = gbuffer::GBUF_NONE;
constexpr uint GBUF_DIFFUSE = gbuffer::GBUF_DIFFUSE;
constexpr uint GBUF_SUBSURFACE = gbuffer::GBUF_SUBSURFACE;
constexpr uint GBUF_REFLECTION = gbuffer::GBUF_REFLECTION;
constexpr uint GBUF_REFLECTION_COLORLESS = gbuffer::GBUF_REFLECTION_COLORLESS;
constexpr uint GBUF_REFRACTION = gbuffer::GBUF_REFRACTION;
constexpr uint GBUF_REFRACTION_COLORLESS = gbuffer::GBUF_REFRACTION_COLORLESS;
#endif

#define FEATURE_HARDWARE_GI (1u << 0u)
#define FEATURE_HARDWARE_REFLECTIONS (1u << 2u)
#define FEATURE_HARDWARE_REFRACTIONS (1u << 3u)

shared uint tile_needs_hardware_trace;

bool texel_requires_hardware_trace(int2 texel)
{
  if (!in_image_range(texel, ray_data_img)) {
    return false;
  }

  const float4 packed_ray = imageLoadFast(ray_data_img, texel);
  if (packed_ray.w == 0.0f) {
    return false;
  }

  int2 texel_fullres = texel * uniform_buf.raytrace.resolution_scale +
                       uniform_buf.raytrace.resolution_bias;
  if (uniform_buf.raytrace.use_hardware_ign_sampling && (uniform_buf.raytrace.resolution_scale > 1)) {
    texel_fullres = raytrace_representative_fullres_texel(
        texel, uniform_buf.raytrace.resolution_scale, uniform_buf.raytrace.resolution_bias);
  }
  if (any(lessThan(texel_fullres, int2(0))) ||
      any(greaterThanEqual(texel_fullres, textureSize(gbuf_header_tx, 0).xy)))
  {
    return false;
  }

  const gbuffer::Header gbuf_header = gbuffer::read_header(texel_fullres);
  const uint gbuf_mode = gbuf_header.bin_type(uniform_buf.raytrace.closure_index);
  const uint hardware_feature_mask = uniform_buf.raytrace.hardware_feature_mask;
  const bool supports_hardware_gi = ((hardware_feature_mask & FEATURE_HARDWARE_GI) != 0u) &&
                                    ((gbuf_mode == GBUF_DIFFUSE) || (gbuf_mode == GBUF_SUBSURFACE));
  const bool supports_hardware_reflection =
      ((hardware_feature_mask & FEATURE_HARDWARE_REFLECTIONS) != 0u) &&
      ((gbuf_mode == GBUF_REFLECTION) || (gbuf_mode == GBUF_REFLECTION_COLORLESS));
  const bool supports_hardware_refraction =
      ((hardware_feature_mask & FEATURE_HARDWARE_REFRACTIONS) != 0u) &&
      ((gbuf_mode == GBUF_REFRACTION) || (gbuf_mode == GBUF_REFRACTION_COLORLESS));

  if (!(supports_hardware_gi || supports_hardware_reflection || supports_hardware_refraction) ||
      (gbuf_mode == GBUF_NONE))
  {
    return false;
  }

  const bool continuation_required =
      (supports_hardware_reflection && (uniform_buf.raytrace.hardware_reflection_bounces > 1)) ||
      (supports_hardware_refraction && (uniform_buf.raytrace.hardware_refraction_bounces > 1));
  const float preserved_screen_time = imageLoadFast(ray_time_img, texel).x;
  const bool scene_final_specular_phase =
      (uniform_buf.raytrace.hardware_trace_phase == HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR);
  const bool preserved_screen_hit =
      (supports_hardware_reflection || supports_hardware_refraction) &&
      (preserved_screen_time > 0.0f) && (preserved_screen_time < 10000.0f);

  return !((preserved_screen_hit && !continuation_required) && !scene_final_specular_phase);
}

void main()
{
  if (gl_LocalInvocationIndex == 0u) {
    tile_needs_hardware_trace = 0u;

    if (gl_WorkGroupID.x == 0u) {
      /* Match the tile compaction initialization pattern for indirect dispatch buffers. */
#if defined(GPU_INTEL) && defined(OS_WIN)
      atomicExchange(hardware_trace_dispatch_buf.num_groups_y, 1u);
      atomicExchange(hardware_trace_dispatch_buf.num_groups_z, 1u);
#else
      hardware_trace_dispatch_buf.num_groups_y = 1u;
      hardware_trace_dispatch_buf.num_groups_z = 1u;
#endif
    }
  }

  barrier();

  constexpr uint tile_size = RAYTRACE_GROUP_SIZE;
  const uint2 tile_coord = unpackUvec2x16(tiles_coord_buf[gl_WorkGroupID.x]);
  const int2 texel = int2(gl_LocalInvocationID.xy + tile_coord * tile_size);

  if (texel_requires_hardware_trace(texel)) {
    tile_needs_hardware_trace = 1u;
  }

  barrier();

  if (gl_LocalInvocationIndex == 0u && tile_needs_hardware_trace > 0u) {
    const uint tile_index = atomicAdd(hardware_trace_dispatch_buf.num_groups_x, 1u);
    hardware_trace_tiles_buf[tile_index] = packUvec2x16(tile_coord);
  }
}
