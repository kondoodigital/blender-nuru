/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/**
 * Does not use any tracing method. Only rely on local light probes to get the incoming radiance.
 */

#include "infos/eevee_tracing_infos.hh"

COMPUTE_SHADER_CREATE_INFO(eevee_ray_trace_fallback)

#include "eevee_bxdf_sampling_lib.glsl"
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

  /* Check if texel is out of bounds,
   * so we can utilize fast texture functions and early-out if not. */
  if (any(greaterThanEqual(texel, imageSize(ray_time_img).xy))) {
    return;
  }

  float depth = reverse_z::read(texelFetch(depth_tx, texel_fullres, 0).r);
  float2 uv = (float2(texel_fullres) + 0.5f) * uniform_buf.raytrace.full_resolution_inv;

  float4 ray_data_im = imageLoadFast(ray_data_img, texel);
  float ray_pdf_inv = ray_data_im.w;

  if (ray_pdf_inv == 0.0f) {
    /* Invalid ray or pixels without ray. Do not trace. */
    imageStoreFast(ray_time_img, texel, float4(0.0f));
    imageStoreFast(ray_radiance_img, texel, float4(0.0f));
    return;
  }

  float3 P = drw_point_screen_to_world(float3(uv, depth));
  float3 V = drw_world_incident_vector(P);

  Ray ray;
  ray.origin = P;
  ray.direction = ray_data_im.xyz;
  ClosureUndetermined surface_closure = gbuffer::read_bin(texel_fullres, closure_index);
  const bool surface_is_specular = (surface_closure.type ==
                                    CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID) ||
                                   (surface_closure.type ==
                                    CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID);
  const bool precombine_specular_caustics = uniform_buf.raytrace.use_hardware_tracing_method &&
                                            (uniform_buf.raytrace.hardware_trace_phase !=
                                             HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR) &&
                                            uniform_buf.raytrace.use_hardware_caustics &&
                                            surface_is_specular;
  if (precombine_specular_caustics) {
    /* Fail closed for the non-hardware fallback, but leave the ray unresolved so the later
     * hardware-trace compaction still dispatches the real caustics-owned HWRT path. */
    imageStoreFast(ray_time_img, texel, float4(0.0f));
    imageStoreFast(ray_radiance_img, texel, float4(0.0f));
    return;
  }

  /* Only closure 0 can be a transmission closure. */
  if (closure_index == 0) {
    const gbuffer::Header gbuf_header = gbuffer::read_header(texel_fullres);
    const Thickness thickness = gbuffer::read_thickness(gbuf_header, texel_fullres);
    if (thickness.value() != 0.0f) {
      ClosureUndetermined cl = gbuffer::read_bin(texel_fullres, closure_index);
      ray = raytrace_thickness_ray_amend(ray, cl, V, thickness);
    }
  }

  /* Using ray direction as geometric normal to bias the sampling position.
   * This is faster than loading the gbuffer again and averages between reflected and normal
   * direction over many rays. */
  float3 radiance = float3(0.0f);
  if (uniform_buf.raytrace.use_hardware_tracing_method &&
      ((surface_closure.type == CLOSURE_BSDF_DIFFUSE_ID) ||
       (surface_closure.type == CLOSURE_BSSRDF_BURLEY_ID)) &&
      (uniform_buf.raytrace.hardware_gi_mode == RAYTRACE_GI_MODE_OFF))
  {
    radiance = float3(0.0f);
  }
  else {
    float3 Ng = ray.direction;
    LightProbeSample samp = lightprobe_load(float2(texel), ray.origin, Ng, V);
    const bool use_dome_world_probe =
        uniform_buf.raytrace.use_hardware_fast_gi &&
        uniform_buf.raytrace.use_hardware_fast_gi_field &&
        ((surface_closure.type == CLOSURE_BSDF_DIFFUSE_ID) ||
         (surface_closure.type == CLOSURE_BSSRDF_BURLEY_ID)) &&
        lightprobe_uses_world(samp);
    if (use_dome_world_probe) {
      radiance = float3(0.0f);
    }
    else if (!use_hardware_rt_environment_visibility && lightprobe_uses_world(samp)) {
      radiance = float3(0.0f);
    }
    else {
    /* Clamp SH to have parity with forward evaluation. */
      float clamp_indirect = uniform_buf.clamp.surface_indirect;
      samp.volume_irradiance = spherical_harmonics::clamp_energy(samp.volume_irradiance,
                                                                 clamp_indirect);

      float3 world_direction = ray.direction;
      float environment_visibility = 1.0f;
      const bool surface_is_diffuse_gi = (surface_closure.type == CLOSURE_BSDF_DIFFUSE_ID) ||
                                         (surface_closure.type == CLOSURE_BSSRDF_BURLEY_ID);
      if (use_hardware_rt_environment_visibility && lightprobe_uses_world(samp) &&
          surface_is_diffuse_gi)
      {
        HardwareEnvironmentVisibilityData env_visibility = hardware_environment_visibility_load(
            texel_fullres, Ng);
        /* The environment visibility buffer is a diffuse dome-occlusion approximation. Keep
         * specular fallback world misses on their traced direction so mirrors/glass do not inherit
         * diffuse dome blur from the primary surface. */
        world_direction = hardware_environment_visibility_direction(
            env_visibility, ray.direction, Ng);
        environment_visibility = env_visibility.visibility;
      }
      radiance = lightprobe_eval_direction(samp, ray.origin, world_direction, ray_pdf_inv);
      radiance *= environment_visibility;
    }
  }
  /* Set point really far for correct reprojection of background. */
  float hit_time = 1000.0f;

  radiance = colorspace_brightness_clamp_max(radiance, uniform_buf.clamp.surface_indirect);

  imageStoreFast(ray_time_img, texel, float4(hit_time));
  imageStoreFast(ray_radiance_img, texel, float4(radiance, 0.0f));
}
