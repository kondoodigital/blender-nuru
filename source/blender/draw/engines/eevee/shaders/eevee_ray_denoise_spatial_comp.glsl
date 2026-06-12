/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/* Nuru: Contains Nuru-specific changes relative to the Blender parent. */

/**
 * Spatial ray reuse. Denoise raytrace result using ratio estimator.
 *
 * Input: Ray direction * hit time, Ray radiance, Ray hit depth
 * Output: Ray radiance reconstructed, Mean Ray hit depth, Radiance Variance
 *
 * Shader is specialized depending on the type of ray to denoise.
 *
 * Following "Stochastic All The Things: Raytracing in Hybrid Real-Time Rendering"
 * by Tomasz Stachowiak
 * https://www.ea.com/seed/news/seed-dd18-presentation-slides-raytracing
 */

#include "infos/eevee_tracing_infos.hh"

COMPUTE_SHADER_CREATE_INFO(eevee_ray_denoise_spatial)

#include "draw_view_lib.glsl"
#include "eevee_closure_lib.glsl"
#include "eevee_gbuffer_read_lib.glsl"
#include "eevee_reverse_z_lib.glsl"
#include "eevee_sampling_lib.glsl"
#include "gpu_shader_codegen_lib.glsl"
#include "gpu_shader_math_base_lib.glsl"
#include "gpu_shader_utildefines_lib.glsl"

void transmission_thickness_amend_closure(ClosureUndetermined &cl, float3 &V, float thickness)
{
  switch (cl.type) {
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
      bxdf_ggx_context_amend_transmission(cl, V, thickness);
      break;
    case CLOSURE_NONE_ID:
    case CLOSURE_BSDF_DIFFUSE_ID:
    case CLOSURE_BSDF_TRANSLUCENT_ID:
    case CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID:
    case CLOSURE_BSSRDF_BURLEY_ID:
      break;
  }
}

/* Tag pixel radiance as invalid. */
void invalid_pixel_write(int2 texel)
{
  imageStoreFast(out_radiance_img, texel, float4(FLT_11_11_10_MAX, 0.0f));
  imageStoreFast(out_variance_img, texel, float4(0.0f));
  imageStoreFast(out_hit_depth_img, texel, float4(0.0f));
}

float bicubic_bspline_weight(float x)
{
  x = abs(x);
  if (x < 1.0f) {
    return (4.0f + x * x * (-6.0f + 3.0f * x)) / 6.0f;
  }
  if (x < 2.0f) {
    float tail = 2.0f - x;
    return (tail * tail * tail) / 6.0f;
  }
  return 0.0f;
}

bool specular_bicubic_sample_valid(int2 sample_texel,
                                   float center_depth,
                                   ClosureUndetermined closure)
{
  if (any(lessThan(sample_texel, int2(0))) ||
      any(greaterThanEqual(sample_texel, imageSize(ray_radiance_img).xy)))
  {
    return false;
  }

  float4 ray_data = imageLoad(ray_data_img, sample_texel);
  if (abs(ray_data.w) == 0.0f) {
    return false;
  }

  int2 sample_texel_fullres = raytrace_texel_to_fullres(
      sample_texel,
      raytrace_resolution_scale,
      uniform_buf.raytrace.resolution_scale_denominator,
      uniform_buf.raytrace.resolution_bias);
  if (uniform_buf.raytrace.use_hardware_ign_sampling) {
    sample_texel_fullres = raytrace_representative_fullres_texel(
        sample_texel,
        raytrace_resolution_scale,
        uniform_buf.raytrace.resolution_scale_denominator,
        uniform_buf.raytrace.resolution_bias);
  }
  if (!in_texture_range(sample_texel_fullres, gbuf_header_tx)) {
    return false;
  }

  float sample_depth = reverse_z::read(texelFetch(depth_tx, sample_texel_fullres, 0).r);
  if (sample_depth <= 0.0f || sample_depth >= 1.0f ||
      abs(sample_depth - center_depth) > 2.0e-3f)
  {
    return false;
  }

  ClosureUndetermined sample_closure = gbuffer::read_bin(sample_texel_fullres, closure_index);
  return sample_closure.type == closure.type && dot(sample_closure.N, closure.N) >= 0.5f;
}

bool specular_bicubic_reconstruct(int2 texel_fullres,
                                  float center_depth,
                                  ClosureUndetermined closure,
                                  out float3 radiance,
                                  out float hit_variance,
                                  out float closest_hit_time)
{
  const bool specular_closure = (closure.type == CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID) ||
                                (closure.type == CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID);
  if (!specular_closure ||
      raytrace_resolution_scale <= uniform_buf.raytrace.resolution_scale_denominator)
  {
    return false;
  }

  float scale = float(max(raytrace_resolution_scale, 1));
  float denominator = float(max(uniform_buf.raytrace.resolution_scale_denominator, 1));
  float2 lowres_coord = (float2(texel_fullres - uniform_buf.raytrace.resolution_bias) *
                         denominator) /
                        scale;
  float2 base_coord = floor(lowres_coord);
  float2 interp = lowres_coord - base_coord;
  int2 base_texel = int2(base_coord);

  float3 radiance_accum = float3(0.0f);
  float3 rgb_moment = float3(0.0f);
  float weight_accum = 0.0f;
  float min_hit_time = 1.0e10f;

  for (int y = -1; y <= 2; y++) {
    float weight_y = bicubic_bspline_weight(float(y) - interp.y);
    for (int x = -1; x <= 2; x++) {
      float weight = bicubic_bspline_weight(float(x) - interp.x) * weight_y;
      if (weight <= 0.0f) {
        continue;
      }

      int2 sample_texel = base_texel + int2(x, y);
      if (!specular_bicubic_sample_valid(sample_texel, center_depth, closure)) {
        continue;
      }

      float3 sample_radiance = imageLoad(ray_radiance_img, sample_texel).rgb;
      float sample_hit_time = imageLoad(ray_time_img, sample_texel).r;
      if (sample_hit_time > 0.0f) {
        min_hit_time = min(min_hit_time, sample_hit_time);
      }

      radiance_accum += sample_radiance * weight;
      rgb_moment += square(sample_radiance) * weight;
      weight_accum += weight;
    }
  }

  if (weight_accum <= 0.0f) {
    return false;
  }

  float inv_weight = safe_rcp(weight_accum);
  radiance = radiance_accum * inv_weight;
  float3 rgb_variance = abs(rgb_moment * inv_weight - square(radiance));
  hit_variance = reduce_max(rgb_variance);
  closest_hit_time = min_hit_time;
  return true;
}

void main()
{
  constexpr uint tile_size = RAYTRACE_GROUP_SIZE;
  uint2 tile_coord = unpackUvec2x16(tiles_coord_buf[gl_WorkGroupID.x]);

  int2 texel_fullres = int2(gl_LocalInvocationID.xy + tile_coord * tile_size);
  int2 texel = raytrace_fullres_to_texel(texel_fullres,
                                         raytrace_resolution_scale,
                                         uniform_buf.raytrace.resolution_scale_denominator,
                                         uniform_buf.raytrace.resolution_bias);

  /* Clear neighbor tiles that will not be processed. */
  /* TODO(fclem): Optimize this. We don't need to clear the whole ring. */
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      if (x == 0 && y == 0) {
        continue;
      }

      int2 tile_coord_neighbor = int2(tile_coord) + int2(x, y);
      if (!in_image_range(tile_coord_neighbor, tile_mask_img)) {
        continue;
      }

      int3 sample_tile = int3(tile_coord_neighbor, closure_index);

      uint tile_mask = imageLoadFast(tile_mask_img, sample_tile).r;
      bool tile_is_unused = !flag_test(tile_mask, 1u << 0u);
      if (tile_is_unused) {
        int2 texel_fullres_neighbor = texel_fullres + int2(x, y) * int(tile_size);
        invalid_pixel_write(texel_fullres_neighbor);
      }
    }
  }

  bool valid_texel = in_texture_range(texel_fullres, gbuf_header_tx);
  if (!valid_texel) {
    invalid_pixel_write(texel_fullres);
    return;
  }

  gbuffer::Header gbuf_header = gbuffer::read_header(texel_fullres);

  ClosureUndetermined closure = gbuffer::read_bin(texel_fullres, closure_index);

  if (closure.type == CLOSURE_NONE_ID) {
    invalid_pixel_write(texel_fullres);
    return;
  }

  float2 uv = (float2(texel_fullres) + 0.5f) * uniform_buf.raytrace.full_resolution_inv;
  float3 P = drw_point_screen_to_world(float3(uv, 0.5f));
  float3 V = drw_world_incident_vector(P);
  float center_depth = reverse_z::read(texelFetch(depth_tx, texel_fullres, 0).r);

  float thickness = gbuffer::read_thickness(gbuf_header, texel_fullres);
  if (thickness != 0.0f) {
    transmission_thickness_amend_closure(closure, V, thickness);
  }

  float3 bicubic_radiance;
  float bicubic_hit_variance;
  float bicubic_closest_hit_time;
  if (specular_bicubic_reconstruct(
          texel_fullres, center_depth, closure, bicubic_radiance, bicubic_hit_variance, bicubic_closest_hit_time))
  {
    float hit_depth = center_depth;
    if (bicubic_closest_hit_time < 1.0e9f) {
      float scene_z = drw_depth_screen_to_view(center_depth);
      hit_depth = drw_depth_view_to_screen(scene_z - bicubic_closest_hit_time);
    }
    imageStoreFast(out_radiance_img, texel_fullres, float4(bicubic_radiance, 0.0f));
    imageStoreFast(out_variance_img, texel_fullres, float4(bicubic_hit_variance));
    imageStoreFast(out_hit_depth_img, texel_fullres, float4(hit_depth));
    return;
  }

  if (skip_denoise) {
    imageStore(out_radiance_img, texel_fullres, imageLoad(ray_radiance_img, texel));
    return;
  }

  /* Compute filter size and needed sample count */
  float apparent_roughness = closure_apparent_roughness_get(closure);
  float filter_size_factor = saturate(apparent_roughness * 8.0f);
  uint sample_count = 1u + uint(15.0f * filter_size_factor + 0.5f);
  /* NOTE: filter_size should never be greater than twice RAYTRACE_GROUP_SIZE. Otherwise, the
   * reconstruction can becomes ill defined since we don't know if further tiles are valid. */
  float filter_size = 12.0f * sqrt(filter_size_factor);
  if (raytrace_resolution_scale > 1) {
    /* OIDN already denoised in ray-grid space. Reconstruction only picks same-surface neighbors. */
    filter_size = (raytrace_resolution_scale <= 2) ? 1.5f : 2.5f;
    sample_count = (raytrace_resolution_scale <= 2) ? 4u : 9u;
  }

  float2 noise = utility_tx_fetch(utility_tx, float2(texel_fullres), UTIL_BLUE_NOISE_LAYER).ba;
  noise += sampling_rng_1D_get(SAMPLING_CLOSURE);

  float3 rgb_moment = float3(0.0f);
  float3 radiance_accum = float3(0.0f);
  float weight_accum = 0.0f;
  float closest_hit_time = 1.0e10f;

  for (uint i = 0u; i < sample_count; i++) {
    float2 offset_f = (fract(hammersley_2d(i, sample_count) + noise) - 0.5f) * filter_size;
    int2 offset = int2(floor(offset_f + 0.5f));
    int2 sample_texel = texel + offset;

    if (raytrace_resolution_scale > 1) {
      int2 sample_texel_fullres = raytrace_texel_to_fullres(
          sample_texel,
          raytrace_resolution_scale,
          uniform_buf.raytrace.resolution_scale_denominator,
          uniform_buf.raytrace.resolution_bias);
      if (uniform_buf.raytrace.use_hardware_ign_sampling) {
        sample_texel_fullres = raytrace_representative_fullres_texel(
            sample_texel,
            raytrace_resolution_scale,
            uniform_buf.raytrace.resolution_scale_denominator,
            uniform_buf.raytrace.resolution_bias);
      }
      if (!in_texture_range(sample_texel_fullres, gbuf_header_tx)) {
        continue;
      }

      float sample_depth = reverse_z::read(texelFetch(depth_tx, sample_texel_fullres, 0).r);
      if (sample_depth <= 0.0f || sample_depth >= 1.0f ||
          abs(sample_depth - center_depth) > 2.0e-3f)
      {
        continue;
      }

      ClosureUndetermined sample_closure = gbuffer::read_bin(sample_texel_fullres, closure_index);
      if (sample_closure.type != closure.type || dot(sample_closure.N, closure.N) < 0.5f) {
        continue;
      }
    }

    float4 ray_data = imageLoad(ray_data_img, sample_texel);
    float ray_time = imageLoad(ray_time_img, sample_texel).r;
    float4 ray_radiance = imageLoad(ray_radiance_img, sample_texel);

    float3 ray_direction = ray_data.xyz;
    float ray_pdf_inv = abs(ray_data.w);
    /* Skip invalid pixels. */
    if (ray_pdf_inv == 0.0f) {
      continue;
    }

    closest_hit_time = min(closest_hit_time, ray_time);

    /* Slide 54. */
    /* The reference is wrong.
     * The ratio estimator is `pdf_local / pdf_ray` instead of `bsdf_local / pdf_ray`. */
    float pdf = closure_evaluate_pdf(closure, ray_direction, V, thickness);
    float weight = pdf * ray_pdf_inv;

    radiance_accum += ray_radiance.rgb * weight;
    weight_accum += weight;

    rgb_moment += square(ray_radiance.rgb) * weight;
  }
  float inv_weight = safe_rcp(weight_accum);

  radiance_accum *= inv_weight;
  /* Use radiance sum as signal mean. */
  float3 rgb_mean = radiance_accum;
  rgb_moment *= inv_weight;

  float3 rgb_variance = abs(rgb_moment - square(rgb_mean));
  float hit_variance = reduce_max(rgb_variance);

  float scene_z = drw_depth_screen_to_view(center_depth);
  float hit_depth = drw_depth_view_to_screen(scene_z - closest_hit_time);

  imageStoreFast(out_radiance_img, texel_fullres, float4(radiance_accum, 0.0f));
  imageStoreFast(out_variance_img, texel_fullres, float4(hit_variance));
  imageStoreFast(out_hit_depth_img, texel_fullres, float4(hit_depth));
}
