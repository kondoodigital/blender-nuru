/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "infos/eevee_tracing_infos.hh"

COMPUTE_SHADER_CREATE_INFO(eevee_ray_hardware_indirect_gi_cache_store)

#include "draw_view_lib.glsl"
#include "eevee_gbuffer_read_lib.glsl"
#include "eevee_reverse_z_lib.bsl.hh"
#include "gpu_shader_math_vector_lib.glsl"

void main()
{
  int2 texel = int2(gl_GlobalInvocationID.xy);
  int2 extent = imageSize(out_indirect_gi_radiance_cache_img).xy;
  if (any(greaterThanEqual(texel, extent))) {
    return;
  }

  int3 cache_texel = int3(texel, cache_face_index);
  float depth = reverse_z::read(texelFetch(depth_tx, texel, 0).r);
  if (!(depth > 0.0f && depth < 1.0f)) {
    imageStore(out_indirect_gi_radiance_cache_img, cache_texel, float4(0.0f));
    imageStore(out_indirect_gi_position_cache_img, cache_texel, float4(0.0f));
    imageStore(out_indirect_gi_normal_cache_img, cache_texel, float4(0.0f));
    return;
  }

  float2 uv = (float2(texel) + 0.5f) / float2(extent);
  float3 P = drw_point_screen_to_world(float3(uv, depth));
  gbuffer::Layers gbuf = gbuffer::read_layers(texel);
  float3 N = gbuf.surface_N();
  if (dot(N, N) <= 1.0e-8f) {
    imageStore(out_indirect_gi_radiance_cache_img, cache_texel, float4(0.0f));
    imageStore(out_indirect_gi_position_cache_img, cache_texel, float4(0.0f));
    imageStore(out_indirect_gi_normal_cache_img, cache_texel, float4(0.0f));
    return;
  }
  N = normalize(N);

  float3 radiance = max(texelFetch(combined_tx, texel, 0).rgb, float3(0.0f));
  imageStore(out_indirect_gi_radiance_cache_img, cache_texel, float4(radiance, 1.0f));
  imageStore(out_indirect_gi_position_cache_img, cache_texel, float4(P, 1.0f));
  imageStore(out_indirect_gi_normal_cache_img, cache_texel, float4(N, 1.0f));
}
