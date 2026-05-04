/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "infos/eevee_tracing_infos.hh"

COMPUTE_SHADER_CREATE_INFO(eevee_ray_hardware_reflected_receiver_gi_blur)

#include "gpu_shader_utildefines_lib.glsl"

void main()
{
  constexpr uint tile_size = RAYTRACE_GROUP_SIZE;
  uint2 tile_coord = unpackUvec2x16(tiles_coord_buf[gl_WorkGroupID.x]);
  int2 texel = int2(gl_LocalInvocationID.xy + tile_coord * tile_size);
  int2 extent = textureSize(reflected_receiver_gi_tx, 0);

  if (any(greaterThanEqual(texel, extent))) {
    return;
  }

  if (!(texelFetch(ray_time_tx, texel, 0).x > 0.0f)) {
    imageStore(out_reflected_receiver_gi_img, texel, float4(0.0f));
    return;
  }

  float3 center_N = texelFetch(hit_normal_tx, texel, 0).xyz;
  float3 center_P = texelFetch(hit_world_position_tx, texel, 0).xyz;
  if (dot(center_N, center_N) <= 1.0e-10f || any(notEqual(center_P, center_P))) {
    imageStore(out_reflected_receiver_gi_img, texel, float4(0.0f));
    return;
  }
  center_N = normalize(center_N);

  int divisor = max(reflected_receiver_gi_resolution_divisor, 1);
  float2 grid_coord = (float2(texel) - float(divisor) * 0.5f) / float(divisor);
  int2 base_cell = int2(floor(grid_coord));
  float3 accum = float3(0.0f);
  float weight_sum = 0.0f;

  for (int y = -3; y <= 3; y++) {
    for (int x = -3; x <= 3; x++) {
      int2 sample_texel = clamp((base_cell + int2(x, y)) * divisor + int2(divisor / 2),
                                int2(0),
                                extent - int2(1));
      float4 sample_gi = texelFetch(reflected_receiver_gi_tx, sample_texel, 0);
      if (sample_gi.a <= 0.5f || !(texelFetch(ray_time_tx, sample_texel, 0).x > 0.0f)) {
        continue;
      }

      float3 sample_N = texelFetch(hit_normal_tx, sample_texel, 0).xyz;
      float3 sample_P = texelFetch(hit_world_position_tx, sample_texel, 0).xyz;
      if (dot(sample_N, sample_N) <= 1.0e-10f || any(notEqual(sample_P, sample_P))) {
        continue;
      }
      sample_N = normalize(sample_N);

      float normal_weight = smoothstep(0.35f, 0.9f, dot(center_N, sample_N));
      float distance_weight = exp2(-length(sample_P - center_P) * 0.75f);
      float2 texel_delta = (float2(sample_texel) - float2(texel)) / float(divisor);
      float spatial_weight = exp2(-dot(texel_delta, texel_delta) * 0.45f);
      float weight = normal_weight * distance_weight * spatial_weight;
      accum += max(sample_gi.rgb, float3(0.0f)) * weight;
      weight_sum += weight;
    }
  }

  if (weight_sum <= 1.0e-4f) {
    imageStore(out_reflected_receiver_gi_img, texel, float4(0.0f));
    return;
  }
  imageStore(out_reflected_receiver_gi_img, texel, float4(accum / max(weight_sum, 1.0e-4f), 1.0f));
}
