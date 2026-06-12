/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/**
 * Accumulate all active HWRT closure radiance into a single colored ray-grid signal for OIDN.
 */

#include "infos/eevee_tracing_infos.hh"

COMPUTE_SHADER_CREATE_INFO(eevee_ray_shared_indirect_accum)

#include "eevee_gbuffer_read_lib.glsl"
#include "eevee_sampling_lib.glsl"
#include "gpu_shader_codegen_lib.glsl"

void main()
{
  constexpr uint tile_size = RAYTRACE_GROUP_SIZE;
  uint2 tile_coord = unpackUvec2x16(tiles_coord_buf[gl_WorkGroupID.x]);
  int2 texel = int2(gl_LocalInvocationID.xy + tile_coord * tile_size);

  if (any(greaterThanEqual(texel, imageSize(shared_radiance_img).xy))) {
    return;
  }

  int2 texel_fullres = raytrace_texel_to_fullres(texel,
                                                 uniform_buf.raytrace.resolution_scale,
                                                 uniform_buf.raytrace.resolution_scale_denominator,
                                                 uniform_buf.raytrace.resolution_bias);
  if (uniform_buf.raytrace.use_hardware_ign_sampling &&
      (uniform_buf.raytrace.resolution_scale > 1))
  {
    texel_fullres = raytrace_representative_fullres_texel(
        texel,
        uniform_buf.raytrace.resolution_scale,
        uniform_buf.raytrace.resolution_scale_denominator,
        uniform_buf.raytrace.resolution_bias);
  }

  const bool valid_fullres = all(greaterThanEqual(texel_fullres, int2(0))) &&
                             all(lessThan(texel_fullres, textureSize(gbuf_header_tx, 0).xy));

  if (closure_index == 0) {
    imageStoreFast(shared_radiance_img, texel, float4(0.0f));
    imageStoreFast(shared_albedo_img, texel, float4(0.0f));
    imageStoreFast(shared_normal_img, texel, float4(0.0f, 0.0f, 1.0f, 0.0f));
  }

  if (!valid_fullres) {
    return;
  }

  ClosureUndetermined closure = gbuffer::read_bin(texel_fullres, closure_index);
  if (closure.type == CLOSURE_NONE_ID) {
    return;
  }

  float3 closure_color = max(closure.color, float3(0.0f));
  if ((closure.type == CLOSURE_BSDF_TRANSLUCENT_ID ||
       closure.type == CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID) &&
      (gbuffer::read_thickness(gbuffer::read_header(texel_fullres), texel_fullres) != 0.0f))
  {
    closure_color *= closure_color;
  }
  float3 radiance = max(imageLoadFast(ray_radiance_img, texel).rgb, float3(0.0f));
  float3 colored_radiance = radiance * closure_color;

  float3 accum = imageLoadFast(shared_radiance_img, texel).rgb + colored_radiance;
  imageStoreFast(shared_radiance_img, texel, float4(accum, 1.0f));

  float4 albedo_accum = imageLoadFast(shared_albedo_img, texel);
  albedo_accum.rgb += closure_color;
  albedo_accum.a += 1.0f;
  imageStoreFast(shared_albedo_img, texel, albedo_accum);

  if (closure_index == 0) {
    imageStoreFast(shared_normal_img, texel, float4(closure.N, 1.0f));
  }
}
