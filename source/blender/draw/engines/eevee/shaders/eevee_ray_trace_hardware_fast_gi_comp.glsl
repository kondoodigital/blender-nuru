/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "infos/eevee_tracing_infos.hh"

COMPUTE_SHADER_CREATE_INFO(eevee_ray_hardware_fast_gi_update)

#include "draw_view_lib.glsl"
#include "eevee_reverse_z_lib.bsl.hh"
#include "gpu_shader_utildefines_lib.glsl"

float hardware_fast_gi_history_blend(int cascade_index)
{
  return (cascade_index == 0) ? 0.28f : (cascade_index == 1) ? 0.18f : 0.10f;
}

void main()
{
  int3 voxel = int3(gl_GlobalInvocationID.xyz);
  int grid_resolution = max(uniform_buf.raytrace.hardware_fast_gi_grid_resolution, 1);
  if (any(greaterThanEqual(voxel, int3(grid_resolution)))) {
    return;
  }

  int3 atlas_voxel = int3(voxel.xy, voxel.z + cascade_index * grid_resolution);
  float4 history = imageLoadFast(out_fast_gi_img, atlas_voxel);

  float4 cascade_cfg = uniform_buf.raytrace.hardware_fast_gi_cascade_config[cascade_index];
  float voxel_size = cascade_cfg.w;
  float3 cascade_min = cascade_cfg.xyz - 0.5f * float(grid_resolution) * voxel_size;
  float3 P = cascade_min + (float3(voxel) + 0.5f) * voxel_size;

  float3 screen_P = drw_point_world_to_screen(P);
  if (any(lessThan(screen_P.xy, float2(0.0f))) || any(greaterThanEqual(screen_P.xy, float2(1.0f)))) {
    imageStoreFast(out_fast_gi_img, atlas_voxel, history * 0.995f);
    return;
  }

  int2 depth_extent = textureSize(depth_tx, 0);
  int2 texel = clamp(int2(screen_P.xy * float2(depth_extent)), int2(0), depth_extent - 1);
  float depth = reverse_z::read(texelFetch(depth_tx, texel, 0).r);
  if (!(depth > 0.0f && depth < 1.0f)) {
    imageStoreFast(out_fast_gi_img, atlas_voxel, history * 0.995f);
    return;
  }

  float2 uv = (float2(texel) + 0.5f) / float2(depth_extent);
  float3 surface_P = drw_point_screen_to_world(float3(uv, depth));
  float fit_weight = 1.0f - saturate(distance(surface_P, P) / (voxel_size * 1.75f));
  fit_weight *= fit_weight;
  if (fit_weight <= 1.0e-4f) {
    imageStoreFast(out_fast_gi_img, atlas_voxel, history * 0.99f);
    return;
  }

  float3 radiance = max(texelFetch(input_radiance_tx, texel, 0).rgb, float3(0.0f));
  if (max(max(radiance.x, radiance.y), radiance.z) <= 1.0e-5f) {
    imageStoreFast(out_fast_gi_img, atlas_voxel, history * 0.99f);
    return;
  }

  float edge_dist = min(min(screen_P.x, screen_P.y), min(1.0f - screen_P.x, 1.0f - screen_P.y));
  float border_weight = saturate(edge_dist * 6.0f);
  float sample_weight = fit_weight * border_weight;
  float4 target = float4(radiance * sample_weight, sample_weight);
  float blend = hardware_fast_gi_history_blend(cascade_index);
  imageStoreFast(out_fast_gi_img, atlas_voxel, mix(history, target, blend));
}
