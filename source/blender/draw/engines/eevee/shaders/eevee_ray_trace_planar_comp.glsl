/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/**
 * Use screen space tracing against depth buffer of recorded planar capture to find intersection
 * with the scene and its radiance.
 * This pass runs before the screen trace and evaluates valid rays for planar probes. These rays
 * are then tagged to avoid re-evaluation by screen trace.
 */

#include "infos/eevee_tracing_infos.hh"

COMPUTE_SHADER_CREATE_INFO(eevee_ray_trace_planar)

#include "eevee_bxdf_sampling_lib.glsl"
#include "eevee_closure_lib.glsl"
#include "eevee_colorspace_lib.bsl.hh"
#include "eevee_gbuffer_read_lib.glsl"
#define RAYTRACE_GI_MODE_OFF 2
#include "eevee_lightprobe_eval_lib.glsl"
#include "eevee_ray_trace_screen_lib.glsl"
#include "eevee_ray_types_lib.bsl.hh"
#include "eevee_reverse_z_lib.bsl.hh"
#include "eevee_sampling_lib.glsl"

#define RAYTRACE_SPECULAR_MODE_AUTO 3
#define RAYTRACE_SPECULAR_MODE_FULL_RT 2
#define AUTO_FULL_RT_REFLECTION_MAX_ROUGHNESS 0.999f
#define HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR 2

bool planar_hit_requests_full_rt(ClosureType type, float roughness)
{
  if (!use_hardware_specular_scene || (type != CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID)) {
    return false;
  }

  const int mode = uniform_buf.raytrace.hardware_reflection_mode;
  if (mode == RAYTRACE_SPECULAR_MODE_FULL_RT) {
    return true;
  }
  if (mode != RAYTRACE_SPECULAR_MODE_AUTO) {
    return false;
  }

  return roughness <= AUTO_FULL_RT_REFLECTION_MAX_ROUGHNESS;
}

bool planar_hit_closure_is_diffuse_gi(ClosureType type)
{
  return (type == CLOSURE_BSDF_DIFFUSE_ID) || (type == CLOSURE_BSSRDF_BURLEY_ID);
}

bool planar_hit_closure_is_specular(ClosureType type)
{
  return (type == CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID) ||
         (type == CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID);
}

void main()
{
  constexpr uint tile_size = RAYTRACE_GROUP_SIZE;
  uint2 tile_coord = unpackUvec2x16(tiles_coord_buf[gl_WorkGroupID.x]);
  int2 texel = int2(gl_LocalInvocationID.xy + tile_coord * tile_size);

  /* Check if texel is out of bounds,
   * so we can utilize fast texture functions and early-out if not. */
  if (any(greaterThanEqual(texel, imageSize(ray_time_img).xy))) {
    return;
  }

  float4 ray_data_im = imageLoadFast(ray_data_img, texel);
  float ray_pdf_inv = ray_data_im.w;

  if (ray_pdf_inv == 0.0f) {
    /* Invalid ray or pixels without ray. Do not trace. */
    imageStoreFast(ray_time_img, texel, float4(0.0f));
    imageStoreFast(ray_radiance_img, texel, float4(0.0f));
    return;
  }

  int2 texel_fullres = texel * uniform_buf.raytrace.resolution_scale +
                       uniform_buf.raytrace.resolution_bias;

  gbuffer::Header gbuf_header = gbuffer::read_header(texel_fullres);
  ClosureUndetermined cl = gbuffer::read_bin(texel_fullres, closure_index);
  ClosureType closure_type = cl.type;
  float roughness = closure_apparent_roughness_get(cl);
  const bool precombine_specular_caustics = uniform_buf.raytrace.use_hardware_tracing_method &&
                                            (uniform_buf.raytrace.hardware_trace_phase !=
                                             HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR) &&
                                            uniform_buf.raytrace.use_hardware_caustics &&
                                            planar_hit_closure_is_specular(closure_type);
  if (precombine_specular_caustics) {
    imageStoreFast(ray_time_img, texel, float4(0.0f));
    imageStoreFast(ray_radiance_img, texel, float4(0.0f));
    return;
  }
  if (planar_hit_requests_full_rt(closure_type, roughness)) {
    return;
  }

  if ((closure_type == CLOSURE_BSDF_TRANSLUCENT_ID) ||
      (closure_type == CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID))
  {
    /* Planar light-probes cannot trace refraction yet. */
    return;
  }

  float depth = reverse_z::read(texelFetch(depth_tx, texel_fullres, 0).r);
  float2 uv = (float2(texel_fullres) + 0.5f) * uniform_buf.raytrace.full_resolution_inv;

  float3 P = drw_point_screen_to_world(float3(uv, depth));
  float3 V = drw_world_incident_vector(P);

  int planar_id = lightprobe_planar_select(P, V, ray_data_im.xyz);
  if (planar_id == -1) {
    return;
  }

  PlanarProbeData planar = probe_planar_buf[planar_id];

  /* Tag the ray data so that screen trace will not try to evaluate it and override the result. */
  imageStoreFast(ray_data_img, texel, float4(ray_data_im.xyz, -ray_data_im.w));

  Ray ray;
  ray.origin = P;
  ray.direction = ray_data_im.xyz;

  if (uniform_buf.raytrace.use_hardware_tracing_method &&
      planar_hit_closure_is_diffuse_gi(closure_type) &&
      (uniform_buf.raytrace.hardware_gi_mode == RAYTRACE_GI_MODE_OFF))
  {
    imageStoreFast(ray_time_img, texel, float4(10000.0f));
    imageStoreFast(ray_radiance_img, texel, float4(0.0f));
    return;
  }

  float3 radiance = float3(0.0f);
  float noise_offset = sampling_rng_1D_get(SAMPLING_RAYTRACE_W);
  float rand_trace = interleaved_gradient_noise(float2(texel), 5.0f, noise_offset);

  /* TODO(fclem): Take IOR into account in the roughness LOD bias. */
  /* TODO(fclem): pdf to roughness mapping is a crude approximation. Find something better. */
  // float roughness = saturate(ray_pdf_inv);

  /* Transform the ray into planar view-space. */
  Ray ray_view;
  ray_view.origin = transform_point(planar.viewmat, ray.origin);
  ray_view.direction = transform_direction(planar.viewmat, ray.direction);
  /* Extend the ray to cover the whole view. */
  ray_view.max_time = 1000.0f;

  ScreenTraceHitData hit = raytrace_planar(
      uniform_buf.raytrace, planar_depth_tx, planar, rand_trace, ray_view);

  if (hit.valid) {
    /* Evaluate radiance at hit-point. */
    radiance = textureLod(planar_radiance_tx, float3(hit.ss_hit_P.xy, planar_id), 0.0f).rgb;
  }
  else {
    /* Using ray direction as geometric normal to bias the sampling position.
     * This is faster than loading the gbuffer again and averages between reflected and normal
     * direction over many rays. */
    float3 Ng = ray.direction;
    /* Fall back to nearest light-probe. */
    LightProbeSample samp = lightprobe_load(float2(texel), P, Ng, V);
    radiance = lightprobe_eval_direction(samp, P, ray.direction, ray_pdf_inv);
    /* Set point really far for correct reprojection of background. */
    hit.time = 10000.0f;
  }

  radiance = colorspace_brightness_clamp_max(radiance, uniform_buf.clamp.surface_indirect);

  imageStoreFast(ray_time_img, texel, float4(hit.time));
  imageStoreFast(ray_radiance_img, texel, float4(radiance, 0.0f));
}
