/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/**
 * Generate Ray direction along with other data that are then used
 * by the next pass to trace the rays.
 */

#include "infos/eevee_tracing_infos.hh"

#ifdef SCENE_FINAL_SPECULAR_RESOLVE_PASS
COMPUTE_SHADER_CREATE_INFO(eevee_ray_trace_scene_final_specular_resolve)

#include "eevee_colorspace_lib.bsl.hh"
#include "eevee_gbuffer_read_lib.glsl"

void main()
{
  int2 texel = int2(gl_GlobalInvocationID.xy);
  if (any(greaterThanEqual(texel, imageSize(combined_img).xy))) {
    return;
  }

  const gbuffer::Layers gbuf = gbuffer::read_layers(texel);
  const uchar closure_count = gbuf.header.closure_len();
  const uint3 bin_indices = gbuf.header.bin_index_per_layer();

  float3 out_indirect = float3(0.0f);
  float3 specular_indirect = float3(0.0f);

  for (uchar i = 0; i < GBUFFER_LAYER_MAX && i < closure_count; i++) {
    ClosureUndetermined cl = gbuf.layer_get(i);
    if ((cl.type != CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID) &&
        (cl.type != CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID))
    {
      continue;
    }

    const uchar layer_index = bin_indices[i];
    float3 closure_indirect_light = float3(0.0f);
    switch (layer_index) {
      case 0:
        closure_indirect_light = texelFetch(indirect_radiance_1_tx, texel, 0).rgb;
        break;
      case 1:
        closure_indirect_light = texelFetch(indirect_radiance_2_tx, texel, 0).rgb;
        break;
      case 2:
        closure_indirect_light = texelFetch(indirect_radiance_3_tx, texel, 0).rgb;
        break;
      default:
        break;
    }

    if ((cl.type == CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID) &&
        (gbuffer::read_thickness(gbuf.header, texel).value() != 0.0f))
    {
      /* Match deferred combine's double application for thickness-aware transmission. */
      cl.color *= cl.color;
    }

    out_indirect += closure_indirect_light * cl.color;
    specular_indirect += closure_indirect_light;
  }

  const float clamp_indirect = uniform_buf.clamp.surface_indirect;
  out_indirect = colorspace_brightness_clamp_max(out_indirect, clamp_indirect);
  specular_indirect = colorspace_brightness_clamp_max(specular_indirect, clamp_indirect);

  out_indirect *= uniform_buf.clamp.indirect_scale;
  specular_indirect *= uniform_buf.clamp.indirect_scale;

  if (dot(out_indirect, out_indirect) > 1.0e-10f) {
    float4 combined = imageLoadFast(combined_img, texel);
    combined.rgb += out_indirect;
    imageStoreFast(combined_img, texel, combined);
  }

  const int specular_light_id = uniform_buf.render_pass.specular_light_id;
  if ((specular_light_id >= 0) && (dot(specular_indirect, specular_indirect) > 1.0e-10f)) {
    int3 rp_texel = int3(texel, specular_light_id);
    float4 specular_light = imageLoadFast(rp_color_img, rp_texel);
    specular_light.rgb += specular_indirect;
    imageStoreFast(rp_color_img, rp_texel, specular_light);
  }
}
#else
COMPUTE_SHADER_CREATE_INFO(eevee_ray_generate)

#include "eevee_gbuffer_read_lib.glsl"
#include "eevee_ray_generate_lib.glsl"
#include "eevee_raytrace_shared.hh"
#include "eevee_sampling_lib.glsl"
#include "gpu_shader_codegen_lib.glsl"

void main()
{
  constexpr uint tile_size = RAYTRACE_GROUP_SIZE;
  uint2 tile_coord = unpackUvec2x16(tiles_coord_buf[gl_WorkGroupID.x]);
  int2 texel = int2(gl_LocalInvocationID.xy + tile_coord * tile_size);

  int2 texel_fullres = texel * uniform_buf.raytrace.resolution_scale +
                       uniform_buf.raytrace.resolution_bias;
  if (uniform_buf.raytrace.use_hardware_ign_sampling && (uniform_buf.raytrace.resolution_scale > 1)) {
    texel_fullres = raytrace_representative_fullres_texel(
        texel, uniform_buf.raytrace.resolution_scale, uniform_buf.raytrace.resolution_bias);
  }

  gbuffer::Header gbuf_header = gbuffer::read_header(texel_fullres);
  ClosureUndetermined closure = gbuffer::read_bin(texel_fullres, closure_index);

  if (closure.type == CLOSURE_NONE_ID) {
    imageStore(out_ray_data_img, texel, float4(0.0f));
    return;
  }

  if (uniform_buf.raytrace.use_hardware_tracing_method) {
    const bool is_diffuse_like = (closure.type == CLOSURE_BSDF_DIFFUSE_ID) ||
                                 (closure.type == CLOSURE_BSSRDF_BURLEY_ID);
    const bool is_specular_like = (closure.type == CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID) ||
                                  (closure.type == CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID);
    const bool precombine_specular_caustics = (uniform_buf.raytrace.hardware_trace_phase ==
                                               HWRT_TRACE_PHASE_PRECOMBINE) &&
                                              uniform_buf.raytrace.use_hardware_caustics;
    if ((uniform_buf.raytrace.hardware_trace_phase == HWRT_TRACE_PHASE_PRECOMBINE &&
         is_specular_like && !precombine_specular_caustics) ||
        (uniform_buf.raytrace.hardware_trace_phase == HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR &&
         !is_specular_like))
    {
      imageStore(out_ray_data_img, texel, float4(0.0f));
      return;
    }
    if (uniform_buf.raytrace.hardware_trace_phase == HWRT_TRACE_PHASE_PRECOMBINE &&
        !is_diffuse_like && !is_specular_like)
    {
      imageStore(out_ray_data_img, texel, float4(0.0f));
      return;
    }
  }

  float2 uv = (float2(texel_fullres) + 0.5f) / float2(textureSize(gbuf_header_tx, 0).xy);
  float3 P = drw_point_screen_to_world(float3(uv, 0.5f));
  float3 V = drw_world_incident_vector(P);
  float2 noise;
  const bool use_diffuse_ign = uniform_buf.raytrace.use_hardware_ign_sampling &&
                               ((closure.type == CLOSURE_BSDF_DIFFUSE_ID) ||
                                (closure.type == CLOSURE_BSSRDF_BURLEY_ID));
  if (use_diffuse_ign) {
    /* Keep the finer RT diffuse progression from IGN, but reintroduce blue-noise spatial
     * decorrelation so indirect light does not bunch into broad screen-space clumps. */
    float2 ign_noise = interleaved_gradient_noise(float2(texel_fullres) + 0.5f,
                                                  float2(5.0f + float(closure_index) * 13.0f,
                                                         11.0f + float(closure_index) * 17.0f),
                                                  sampling_rng_2D_get(SAMPLING_RAYTRACE_W));
    float2 blue_noise = utility_tx_fetch(utility_tx, float2(texel_fullres), UTIL_BLUE_NOISE_LAYER).rg;
    noise = fract(float2(ign_noise.x, blue_noise.y) + sampling_rng_2D_get(SAMPLING_RAYTRACE_U));
  }
  else {
    int2 noise_offset = int2(sampling_rng_2D_get(SAMPLING_RAYTRACE_V) * float(UTIL_TEX_SIZE));
    int2 noise_texel = int2(texel_fullres.x * 73 + texel_fullres.y * 19,
                            texel_fullres.x * 29 + texel_fullres.y * 71) +
                       noise_offset;
    float2 blue_noise = utility_tx_fetch(utility_tx, float2(noise_texel), UTIL_BLUE_NOISE_LAYER).rg;
    float2 ig_noise = interleaved_gradient_noise(float2(texel_fullres) + 0.5f,
                                                 float2(5.0f, 11.0f),
                                                 sampling_rng_2D_get(SAMPLING_RAYTRACE_W));
    noise = fract(blue_noise + ig_noise + sampling_rng_2D_get(SAMPLING_RAYTRACE_U));
  }

  Thickness thickness = gbuffer::read_thickness(gbuf_header, texel_fullres);

  BsdfSample samp = ray_generate_direction(noise.xy, closure, V, thickness);

  /* Store inverse pdf to speedup denoising.
   * Limit to the smallest non-0 value that the format can encode.
   * Strangely it does not correspond to the IEEE spec. */
  float inv_pdf = (samp.pdf == 0.0f) ? 0.0f : max(6e-8f, 1.0f / samp.pdf);
  imageStoreFast(out_ray_data_img, texel, float4(samp.direction, inv_pdf));
}
#endif
