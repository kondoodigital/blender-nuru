/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/**
 * Use screen space tracing against depth buffer to find intersection with the scene.
 */

#include "infos/eevee_tracing_infos.hh"

COMPUTE_SHADER_CREATE_INFO(eevee_ray_trace_screen)

#include "eevee_bxdf_sampling_lib.glsl"
#include "eevee_closure_lib.glsl"
#include "eevee_colorspace_lib.bsl.hh"
#include "eevee_gbuffer_read_lib.glsl"
#include "eevee_hardware_environment_visibility_lib.glsl"
#define RAYTRACE_GI_MODE_OFF 2
#include "eevee_lightprobe_eval_lib.glsl"
#include "eevee_ray_trace_screen_lib.glsl"
#include "eevee_ray_types_lib.bsl.hh"
#include "eevee_reverse_z_lib.bsl.hh"
#include "eevee_sampling_lib.glsl"
#include "eevee_spherical_harmonics.bsl.hh"

#define RAYTRACE_SPECULAR_MODE_OFF 0
#define RAYTRACE_SPECULAR_MODE_AUTO 3
#define RAYTRACE_SPECULAR_MODE_HYBRID 1
#define RAYTRACE_SPECULAR_MODE_FULL_RT 2
#define AUTO_FULL_RT_REFLECTION_MAX_ROUGHNESS 0.999f
#define AUTO_FULL_RT_REFRACTION_MAX_ROUGHNESS 0.10f
#define HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR 2

bool screen_hit_closure_is_specular(ClosureType type)
{
  return (type == CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID) ||
         (type == CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID);
}

bool screen_hit_closure_is_diffuse_gi(ClosureType type)
{
  return (type == CLOSURE_BSDF_DIFFUSE_ID) || (type == CLOSURE_BSSRDF_BURLEY_ID);
}

int screen_hit_specular_mode(ClosureType type, float roughness)
{
  int mode = RAYTRACE_SPECULAR_MODE_OFF;
  switch (type) {
    case CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID:
      mode = uniform_buf.raytrace.hardware_reflection_mode;
      break;
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
      mode = uniform_buf.raytrace.hardware_refraction_mode;
      break;
    default:
      return RAYTRACE_SPECULAR_MODE_OFF;
  }

  if (mode != RAYTRACE_SPECULAR_MODE_AUTO) {
    return mode;
  }

  if (type == CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID) {
    return (roughness <= AUTO_FULL_RT_REFLECTION_MAX_ROUGHNESS) ?
               RAYTRACE_SPECULAR_MODE_FULL_RT :
               RAYTRACE_SPECULAR_MODE_HYBRID;
  }

  return (roughness <= AUTO_FULL_RT_REFRACTION_MAX_ROUGHNESS) ?
             RAYTRACE_SPECULAR_MODE_FULL_RT :
             RAYTRACE_SPECULAR_MODE_HYBRID;
}

bool screen_hit_requests_full_rt(ClosureType type, float roughness)
{
  return use_hardware_specular_scene &&
         (screen_hit_specular_mode(type, roughness) == RAYTRACE_SPECULAR_MODE_FULL_RT);
}

bool screen_hit_requests_hybrid_retrace(ClosureType type, float roughness)
{
  return use_hardware_hybrid_retrace &&
         (screen_hit_specular_mode(type, roughness) == RAYTRACE_SPECULAR_MODE_HYBRID);
}

bool screen_hit_tracks_ownership_history(ClosureType type, float roughness)
{
  return use_screen_ownership_history &&
         (screen_hit_specular_mode(type, roughness) == RAYTRACE_SPECULAR_MODE_HYBRID);
}

bool screen_hit_continuation_required(bool is_reflection)
{
  return is_reflection ? (uniform_buf.raytrace.hardware_reflection_bounces > 1) :
                         (uniform_buf.raytrace.hardware_refraction_bounces > 1);
}

bool screen_hit_continuation_closure_load(int2 hit_texel,
                                          ClosureUndetermined &hit_cl,
                                          Thickness &hit_thickness)
{
  gbuffer::Header hit_header = gbuffer::read_header(hit_texel);
  hit_cl = closure_new(CLOSURE_NONE_ID);
  hit_cl.N = float3(0.0f);
  hit_thickness = gbuffer::read_thickness(hit_header, hit_texel);

  float best_score = 0.0f;
  for (int i = 0; i < GBUFFER_LAYER_MAX; i++) {
    ClosureUndetermined candidate = gbuffer::read_bin(hit_header, hit_texel, i);
    if (!screen_hit_closure_is_specular(candidate.type)) {
      continue;
    }
    float score = average(abs(candidate.color));
    if (score >= best_score) {
      best_score = score;
      hit_cl = candidate;
    }
  }

  return (hit_cl.type != CLOSURE_NONE_ID) && (dot(hit_cl.N, hit_cl.N) > 1.0e-10f);
}

bool screen_hit_continuation_store(int2 texel, ScreenTraceHitData hit, Ray ray)
{
  const int2 gbuf_extent = textureSize(gbuf_header_tx, 0).xy;
  const int2 hit_texel = clamp(int2(hit.ss_hit_P.xy * float2(gbuf_extent)), int2(0), gbuf_extent - 1);

  ClosureUndetermined hit_cl;
  Thickness hit_thickness;
  if (!screen_hit_continuation_closure_load(hit_texel, hit_cl, hit_thickness)) {
    return false;
  }

  float3 hit_N = safe_normalize(hit_cl.N);
  if (!(dot(hit_N, hit_N) > 1.0e-10f)) {
    return false;
  }

  float3 next_direction = ray.direction;
  switch (hit_cl.type) {
    case CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID:
      next_direction = reflect(ray.direction, hit_N);
      break;
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID: {
      float ior = max(to_closure_refraction(hit_cl).ior, 1.0e-3f);
      float eta = (dot(hit_N, ray.direction) < 0.0f) ? (1.0f / ior) : ior;
      float3 refracted = refract(ray.direction, hit_N, eta);
      next_direction = (dot(refracted, refracted) > 1.0e-10f) ? refracted :
                                                               reflect(ray.direction, hit_N);
      break;
    }
    case CLOSURE_BSDF_TRANSLUCENT_ID:
    case CLOSURE_BSSRDF_BURLEY_ID:
    case CLOSURE_BSDF_DIFFUSE_ID:
    case CLOSURE_NONE_ID:
      return false;
  }

  if (!(dot(next_direction, next_direction) > 1.0e-10f)) {
    return false;
  }

  Ray continuation_ray;
  continuation_ray.origin = transform_point(drw_view().viewinv, hit.v_hit_P);
  continuation_ray.direction = normalize(next_direction);
  if (hit_cl.type == CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID) {
    continuation_ray = raytrace_thickness_ray_amend(
        continuation_ray, hit_cl, -ray.direction, hit_thickness);
  }

  float ray_pdf_inv = imageLoadFast(ray_data_img, texel).w;
  imageStoreFast(ray_data_img, texel, float4(continuation_ray.direction, ray_pdf_inv));
  imageStoreFast(screen_continuation_img, texel, float4(continuation_ray.origin, hit.time));
  return true;
}

bool screen_hit_near_border(int2 hit_texel, int2 extent)
{
  return any(lessThanEqual(hit_texel, int2(1))) ||
         any(greaterThanEqual(hit_texel, max(extent - int2(2), int2(0))));
}

bool screen_hit_depth_load(int2 texel, float &depth)
{
  depth = reverse_z::read(texelFetch(depth_tx, texel, 0).r);
  return depth > 0.0f && depth < 1.0f;
}

bool screen_hit_surface_normal_load(int2 texel, float3 &surface_N)
{
  gbuffer::Layers layers = gbuffer::read_layers(texel);
  if (layers.has_no_closure()) {
    surface_N = float3(0.0f);
    return false;
  }
  surface_N = safe_normalize(layers.surface_N());
  return dot(surface_N, surface_N) > 1.0e-10f;
}

bool screen_hit_neighbor_unstable(int2 neighbor_texel, float hit_depth, float3 hit_N)
{
  float neighbor_depth;
  if (!screen_hit_depth_load(neighbor_texel, neighbor_depth)) {
    return true;
  }

  float3 neighbor_N;
  if (!screen_hit_surface_normal_load(neighbor_texel, neighbor_N)) {
    return true;
  }

  const float normal_alignment = dot(hit_N, neighbor_N);
  const float depth_delta = abs(neighbor_depth - hit_depth);
  return (normal_alignment < 0.25f) || ((depth_delta > 2.0e-2f) && (normal_alignment < 0.95f));
}

bool screen_hit_neighborhood_stable(int2 hit_texel, float hit_depth, float3 hit_N)
{
  int unstable_neighbor_count = 0;
  unstable_neighbor_count += int(screen_hit_neighbor_unstable(hit_texel + int2(-1, 0), hit_depth, hit_N));
  unstable_neighbor_count += int(screen_hit_neighbor_unstable(hit_texel + int2(1, 0), hit_depth, hit_N));
  unstable_neighbor_count += int(screen_hit_neighbor_unstable(hit_texel + int2(0, -1), hit_depth, hit_N));
  unstable_neighbor_count += int(screen_hit_neighbor_unstable(hit_texel + int2(0, 1), hit_depth, hit_N));
  return unstable_neighbor_count < 2;
}

float screen_hit_confidence_score(ScreenTraceHitData hit, Ray ray)
{
  const int2 gbuf_extent = textureSize(gbuf_header_tx, 0).xy;
  const int2 hit_texel = clamp(int2(hit.ss_hit_P.xy * float2(gbuf_extent)), int2(0), gbuf_extent - 1);
  if (screen_hit_near_border(hit_texel, gbuf_extent)) {
    return 0.0f;
  }

  float hit_depth;
  if (!screen_hit_depth_load(hit_texel, hit_depth)) {
    return 0.0f;
  }

  float3 hit_N;
  if (!screen_hit_surface_normal_load(hit_texel, hit_N)) {
    return 0.0f;
  }

  float confidence = 1.0f;
  /* Prefer a clean RT retrace over weak screen ownership on grazing / silhouette-style hits. */
  const float view_alignment = abs(dot(hit_N, -normalize(ray.direction)));
  if (view_alignment < 0.125f) {
    return 0.0f;
  }
  if (view_alignment < 0.25f) {
    confidence = min(confidence, 0.45f);
  }

  /* Reject local disocclusion / missing-data neighborhoods before they split ownership. */
  if (!screen_hit_neighborhood_stable(hit_texel, hit_depth, hit_N)) {
    confidence = min(confidence, 0.35f);
  }

  return confidence;
}

float screen_hit_previous_ownership(ClosureType type, float roughness, float3 hit_P)
{
  if (!screen_hit_tracks_ownership_history(type, roughness)) {
    return 0.0f;
  }

  float2 uv = project_point(uniform_buf.raytrace.denoise_history_persmat, hit_P).xy * 0.5f + 0.5f;
  if (!in_range_exclusive(uv, float2(0.0f), float2(1.0f))) {
    return 0.0f;
  }
  return texture(ownership_history_tx, uv).r;
}

bool screen_hit_confidence_ok(ClosureType type, float roughness, ScreenTraceHitData hit, Ray ray)
{
  const float confidence = screen_hit_confidence_score(hit, ray);
  if (confidence >= 0.75f) {
    return true;
  }
  if (confidence < 0.35f) {
    return false;
  }

  float3 hit_P = transform_point(drw_view().viewinv, hit.v_hit_P);
  return screen_hit_previous_ownership(type, roughness, hit_P) > 0.5f;
}

void main()
{
  constexpr uint tile_size = RAYTRACE_GROUP_SIZE;
  uint2 tile_coord = unpackUvec2x16(tiles_coord_buf[gl_WorkGroupID.x]);
  int2 texel = int2(gl_LocalInvocationID.xy + tile_coord * tile_size);

  /* Check whether texel is out of bounds for all cases, so we can utilize fast
   * texture functions and early exit if not. */
  if (any(greaterThanEqual(texel, imageSize(ray_data_img).xy)) || any(lessThan(texel, int2(0)))) {
    return;
  }

  imageStoreFast(screen_continuation_img, texel, float4(0.0f));
  imageStoreFast(screen_ownership_img, texel, float4(0.0f));

  float4 ray_data_im = imageLoadFast(ray_data_img, texel);
  float ray_pdf_inv = ray_data_im.w;

  if (ray_pdf_inv < 0.0f) {
    /* Ray destined to planar trace. */
    return;
  }

  if (ray_pdf_inv == 0.0f) {
    /* Invalid ray or pixels without ray. Do not trace. */
    imageStoreFast(ray_time_img, texel, float4(0.0f));
    imageStoreFast(ray_radiance_img, texel, float4(0.0f));
    return;
  }

  int2 texel_fullres = texel * uniform_buf.raytrace.resolution_scale +
                       uniform_buf.raytrace.resolution_bias;
  if (uniform_buf.raytrace.use_hardware_ign_sampling && (uniform_buf.raytrace.resolution_scale > 1)) {
    texel_fullres = raytrace_representative_fullres_texel(
        texel, uniform_buf.raytrace.resolution_scale, uniform_buf.raytrace.resolution_bias);
  }

  gbuffer::Header gbuf_header = gbuffer::read_header(texel_fullres);
  ClosureUndetermined cl = gbuffer::read_bin(texel_fullres, closure_index);
  ClosureType closure_type = cl.type;
  float roughness = closure_apparent_roughness_get(cl);
  const bool precombine_specular_caustics = uniform_buf.raytrace.use_hardware_tracing_method &&
                                            (uniform_buf.raytrace.hardware_trace_phase !=
                                             HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR) &&
                                            uniform_buf.raytrace.use_hardware_caustics &&
                                            screen_hit_closure_is_specular(closure_type);
  if (precombine_specular_caustics) {
    imageStoreFast(ray_time_img, texel, float4(0.0f));
    imageStoreFast(ray_radiance_img, texel, float4(0.0f));
    return;
  }
  if (screen_hit_requests_full_rt(closure_type, roughness))
  {
    imageStoreFast(ray_time_img, texel, float4(0.0f));
    imageStoreFast(ray_radiance_img, texel, float4(0.0f));
    return;
  }
  if (uniform_buf.raytrace.use_hardware_tracing_method &&
      screen_hit_closure_is_diffuse_gi(closure_type) &&
      (uniform_buf.raytrace.hardware_gi_mode != RAYTRACE_GI_MODE_OFF))
  {
    /* HWRT-owned diffuse GI must not sample camera-visible radiance buffers. Let the scene-space
     * backend handle emissive geometry so rotating or hiding an emitter from the camera cannot
     * change the GI distribution. */
    imageStoreFast(ray_time_img, texel, float4(0.0f));
    imageStoreFast(ray_radiance_img, texel, float4(0.0f));
    return;
  }

  bool is_reflection = true;
  if ((closure_type == CLOSURE_BSDF_TRANSLUCENT_ID) ||
      (closure_type == CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID))
  {
    is_reflection = false;
  }

  float depth = reverse_z::read(texelFetch(depth_tx, texel_fullres, 0).r);
  float2 uv = (float2(texel_fullres) + 0.5f) * uniform_buf.raytrace.full_resolution_inv;

  float3 P = drw_point_screen_to_world(float3(uv, depth));
  float3 V = drw_world_incident_vector(P);
  Ray ray;
  ray.origin = P;
  ray.direction = ray_data_im.xyz;

  if (uniform_buf.raytrace.use_hardware_tracing_method &&
      screen_hit_closure_is_diffuse_gi(closure_type) &&
      (uniform_buf.raytrace.hardware_gi_mode == RAYTRACE_GI_MODE_OFF))
  {
    imageStoreFast(ray_time_img, texel, float4(10000.0f));
    imageStoreFast(ray_radiance_img, texel, float4(0.0f));
    return;
  }

  /* Only closure 0 can be a transmission closure. */
  if (closure_index == 0) {
    const Thickness thickness = gbuffer::read_thickness(gbuf_header, texel_fullres);
    if (thickness.value() != 0.0f) {
      ray = raytrace_thickness_ray_amend(ray, cl, V, thickness);
    }
  }

  float3 radiance = float3(0.0f);
  float noise_offset = sampling_rng_1D_get(SAMPLING_RAYTRACE_W);
  float rand_trace = interleaved_gradient_noise(float2(texel), 5.0f, noise_offset);

  /* Transform the ray into view-space. */
  Ray ray_view;
  ray_view.origin = transform_point(drw_view().viewmat, ray.origin);
  ray_view.direction = transform_direction(drw_view().viewmat, ray.direction);
  /* Extend the ray to cover the whole view. */
  ray_view.max_time = 1000.0f;

  ScreenTraceHitData hit;
  hit.valid = false;
  /* This huge branch is likely to be a huge issue for performance.
   * We could split the shader but that would mean to dispatch some area twice for the same closure
   * index. Another idea is to put both HiZ buffer int he same texture and dynamically access one
   * or the other. But that might also impact performance. */
  if (is_reflection) {
    hit = raytrace_screen(uniform_buf.raytrace,
                          uniform_buf.hiz,
                          hiz_front_tx,
                          rand_trace,
                          roughness,
                          true,  /* discard_backface */
                          false, /* allow_self_intersection */
                          ray_view);

    if (hit.valid) {
      float3 hit_P = transform_point(drw_view().viewinv, hit.v_hit_P);
      /* TODO(@fclem): Split matrix multiply for precision. */
      float3 history_ndc_hit_P = project_point(uniform_buf.raytrace.radiance_persmat, hit_P);
      float3 history_ss_hit_P = history_ndc_hit_P * 0.5f + 0.5f;
      /* Fetch radiance at hit-point. */
      radiance = textureLod(radiance_front_tx, history_ss_hit_P.xy, 0.0f).rgb;
    }
  }
  else if (trace_refraction) {
    hit = raytrace_screen(uniform_buf.raytrace,
                          uniform_buf.hiz,
                          hiz_back_tx,
                          rand_trace,
                          roughness,
                          false, /* discard_backface */
                          true,  /* allow_self_intersection */
                          ray_view);

    if (hit.valid) {
      radiance = textureLod(radiance_back_tx, hit.ss_hit_P.xy, 0.0f).rgb;
    }
  }

  if (hit.valid && screen_hit_requests_hybrid_retrace(closure_type, roughness) &&
      !screen_hit_confidence_ok(closure_type, roughness, hit, ray))
  {
    hit.valid = false;
    radiance = float3(0.0f);
  }
  if (hit.valid && screen_hit_tracks_ownership_history(closure_type, roughness)) {
    imageStoreFast(screen_ownership_img, texel, float4(1.0f, 0.0f, 0.0f, 0.0f));
  }

  if (!hit.valid) {
    {
    /* Using ray direction as geometric normal to bias the sampling position.
     * This is faster than loading the gbuffer again and averages between reflected and normal
     * direction over many rays. */
      float3 Ng = ray.direction;
      /* Fall back to nearest light-probe. */
      LightProbeSample samp = lightprobe_load(float2(texel), ray.origin, Ng, V);
      const bool use_dome_world_probe =
          uniform_buf.raytrace.use_hardware_fast_gi &&
          uniform_buf.raytrace.use_hardware_fast_gi_field &&
          screen_hit_closure_is_diffuse_gi(closure_type) && lightprobe_uses_world(samp);
      if (use_dome_world_probe) {
        radiance = float3(0.0f);
        hit.time = 10000.0f;
      }
      else if (!use_hardware_rt_environment_visibility && lightprobe_uses_world(samp)) {
        radiance = float3(0.0f);
        hit.time = 10000.0f;
      }
      else {
        /* Clamp SH to have parity with forward evaluation. */
        float clamp_indirect = uniform_buf.clamp.surface_indirect;
        samp.volume_irradiance = spherical_harmonics::clamp_energy(samp.volume_irradiance,
                                                                   clamp_indirect);

        float3 world_direction = ray.direction;
        float environment_visibility = 1.0f;
        if (use_hardware_rt_environment_visibility && lightprobe_uses_world(samp) &&
            screen_hit_closure_is_diffuse_gi(closure_type))
        {
          HardwareEnvironmentVisibilityData env_visibility = hardware_environment_visibility_load(
              texel_fullres, Ng);
          /* The environment visibility buffer is a diffuse dome-occlusion approximation. Keep
           * specular world misses on their traced direction instead of blurring Hybrid env
           * reflections through that bent-dome signal. */
          world_direction = hardware_environment_visibility_direction(
              env_visibility, ray.direction, Ng);
          environment_visibility = env_visibility.visibility;
        }

        radiance = lightprobe_eval_direction(samp, ray.origin, world_direction, ray_pdf_inv);
        /* Set point really far for correct reprojection of background. */
        hit.time = 10000.0f;
        radiance *= environment_visibility;
      }
    }
  }
  else if (screen_hit_continuation_required(is_reflection)) {
    screen_hit_continuation_store(texel, hit, ray);
  }
  radiance = colorspace_brightness_clamp_max(radiance, uniform_buf.clamp.surface_indirect);

  imageStoreFast(ray_time_img, texel, float4(hit.time));
  imageStoreFast(ray_radiance_img, texel, float4(radiance, 0.0f));
}
