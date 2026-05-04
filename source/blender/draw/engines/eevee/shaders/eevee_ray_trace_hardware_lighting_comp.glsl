/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/**
 * Evaluate secondary-hit lighting for the experimental Hardware GI path.
 *
 * The Metal trace currently provides hit distance plus coarse material proxies. This pass rebuilds
 * the hit point in Eevee space and reuses the existing light/shadow evaluation code so sun, point,
 * and spot lights can contribute indirect bounce without re-implementing Eevee lighting in Metal.
 */

#include "infos/eevee_tracing_infos.hh"

#define SHADOW_DISPATCH_USE_GLOBAL_TEXEL
#define SHADOW_DISPATCH_USE_GLOBAL_HARDWARE_RT
#define LIGHT_ITER_FORCE_NO_CULLING
#define LIGHT_CLOSURE_EVAL_COUNT 1
COMPUTE_SHADER_CREATE_INFO(eevee_ray_trace_hardware_lighting)

#include "eevee_closure_lib.glsl"
#include "eevee_colorspace_lib.bsl.hh"
#include "eevee_gbuffer_read_lib.glsl"
#include "eevee_hardware_fast_gi_lib.glsl"
#include "eevee_hardware_environment_visibility_lib.glsl"
int2 shadow_dispatch_texel_fullres = int2(0);
bool shadow_dispatch_use_hardware_rt = false;
#include "eevee_light_eval_lib.glsl"
#include "eevee_lightprobe_eval_lib.glsl"
#include "eevee_sampling_lib.glsl"
#include "eevee_ray_trace_screen_lib.glsl"
#include "eevee_reverse_z_lib.bsl.hh"
#include "eevee_spherical_harmonics.bsl.hh"
#include "gpu_shader_codegen_lib.glsl"

#define RAYTRACE_SPECULAR_MODE_OFF 0
#define RAYTRACE_SPECULAR_MODE_AUTO 3
#define RAYTRACE_SPECULAR_MODE_HYBRID 1
#define RAYTRACE_SPECULAR_MODE_FULL_RT 2
#define AUTO_FULL_RT_REFLECTION_MAX_ROUGHNESS 0.999f
#define AUTO_FULL_RT_REFRACTION_MAX_ROUGHNESS 0.10f
#define HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR 2
#define PRINCIPLED_DIFFUSE_REFLECTION_FADE_START 0.5f
#define PRINCIPLED_DIFFUSE_REFLECTION_FADE_END 1.0f

int hardware_hit_specular_mode(ClosureUndetermined cl)
{
  int mode = RAYTRACE_SPECULAR_MODE_OFF;
  switch (cl.type) {
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

  const float roughness = closure_apparent_roughness_get(cl);
  if (cl.type == CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID) {
    return (roughness <= AUTO_FULL_RT_REFLECTION_MAX_ROUGHNESS) ?
               RAYTRACE_SPECULAR_MODE_FULL_RT :
               RAYTRACE_SPECULAR_MODE_HYBRID;
  }

  return (roughness <= AUTO_FULL_RT_REFRACTION_MAX_ROUGHNESS) ?
             RAYTRACE_SPECULAR_MODE_FULL_RT :
             RAYTRACE_SPECULAR_MODE_HYBRID;
}

float hardware_principled_diffuse_reflection_fade(ClosureUndetermined base_cl,
                                                  ClosureUndetermined specular_cl)
{
  const float base_strength = average(abs(base_cl.color));
  const float specular_strength = average(abs(specular_cl.color));
  const float total_strength = base_strength + specular_strength;
  return (total_strength > 1.0e-6f) ? saturate(base_strength / total_strength) : 1.0f;
}

float hardware_hit_reflection_layer_opacity(ClosureUndetermined specular_cl)
{
  return (specular_cl.type == CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID) ?
             saturate(specular_cl.data.y) :
             0.0f;
}

ClosureUndetermined hardware_hit_refracted_metal_direct_closure(ClosureUndetermined cl,
                                                                bool refracted_textured_receiver)
{
  if (refracted_textured_receiver && cl.type == CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID) {
    /* Analytic lights are not geometry in the Metal acceleration structure. Keep the closure on
     * the metal/specular path, but avoid a delta-like lobe becoming black except at the tiny
     * reflected-light highlight. */
    cl.data.x = max(cl.data.x, 0.25f);
  }
  return cl;
}

bool hardware_hit_preserves_screen_baseline(ClosureUndetermined primary_closure)
{
  return hardware_hit_specular_mode(primary_closure) != RAYTRACE_SPECULAR_MODE_FULL_RT;
}

bool hardware_primary_surface_has_full_rt_specular(int2 texel_fullres)
{
  const gbuffer::Layers gbuf = gbuffer::read_layers(texel_fullres);
  const uchar closure_count = gbuf.header.closure_len();
  for (uchar i = 0; i < GBUFFER_LAYER_MAX && i < closure_count; i++) {
    const ClosureUndetermined cl = gbuf.layer_get(i);
    if (hardware_hit_closure_is_specular_family(cl.type) &&
        (hardware_hit_specular_mode(cl) == RAYTRACE_SPECULAR_MODE_FULL_RT))
    {
      return true;
    }
  }
  return false;
}

bool hardware_hit_allows_scene_final_raster_reuse(int2 texel_fullres)
{
  if (uniform_buf.raytrace.hardware_trace_phase != HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR) {
    return true;
  }
  /* Fail closed for the whole primary surface, not only the currently resolved closure bin.
   * Full RT specular pixels can still carry extra base/specular bins in the GBuffer, and letting
   * one of those bins opt back into raster reuse reintroduces camera-relative reflected/refracted
   * patterns even though the visible primary closure is already in Full RT mode. */
  return !hardware_primary_surface_has_full_rt_specular(texel_fullres);
}

bool hardware_hit_uses_caustics()
{
  return uniform_buf.raytrace.use_hardware_caustics;
}

float3 hardware_caustics_load(int2 texel_fullres)
{
  return imageLoadFast(hardware_caustics_img, texel_fullres).rgb;
}

bool hardware_ray_load(int2 texel,
                       int2 &texel_fullres,
                       float4 &ray_data_im,
                       float &ray_time)
{
  if (any(lessThan(texel, int2(0))) || any(greaterThanEqual(texel, imageSize(ray_data_img).xy))) {
    return false;
  }

  ray_data_im = imageLoadFast(ray_data_img, texel);
  ray_time = imageLoadFast(ray_time_img, texel).r;
  if (ray_data_im.w == 0.0f) {
    return false;
  }

  texel_fullres = texel * uniform_buf.raytrace.resolution_scale + uniform_buf.raytrace.resolution_bias;
  if (uniform_buf.raytrace.use_hardware_ign_sampling && (uniform_buf.raytrace.resolution_scale > 1)) {
    texel_fullres = raytrace_representative_fullres_texel(
        texel, uniform_buf.raytrace.resolution_scale, uniform_buf.raytrace.resolution_bias);
  }
  if (any(lessThan(texel_fullres, int2(0))) ||
      any(greaterThanEqual(texel_fullres, textureSize(depth_tx, 0))))
  {
    return false;
  }

  return true;
}

float3 hardware_direction_unpack(float2 packed_dir)
{
  packed_dir = packed_dir * 2.0f - 1.0f;
  float3 dir = float3(
      packed_dir.x, packed_dir.y, 1.0f - abs(packed_dir.x) - abs(packed_dir.y));
  float t = clamp(-dir.z, 0.0f, 1.0f);
  dir.x += (dir.x >= 0.0f) ? -t : t;
  dir.y += (dir.y >= 0.0f) ? -t : t;
  return normalize(dir);
}

bool hardware_hit_direction_load(int2 texel, float3 &ray_direction)
{
  float2 packed_dir = float2(imageLoadFast(hit_material_img, texel).w,
                             imageLoadFast(hit_normal_img, texel).w);
  if (all(equal(packed_dir, float2(0.0f)))) {
    return false;
  }
  ray_direction = hardware_direction_unpack(packed_dir);
  return isfinite(ray_direction.x) && isfinite(ray_direction.y) && isfinite(ray_direction.z) &&
         dot(ray_direction, ray_direction) > 1.0e-10f;
}

ClosureType hardware_hit_closure_type_unpack(float packed_type)
{
  return ClosureType(uint(max(packed_type, 0.0f) + 0.5f));
}

bool hardware_hit_closure_is_specular_family(ClosureType type)
{
  return (type == CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID) ||
         (type == CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID);
}

bool hardware_hit_closure_is_base_family(ClosureType type)
{
  return (type == CLOSURE_BSDF_DIFFUSE_ID) || (type == CLOSURE_BSDF_TRANSLUCENT_ID) ||
         (type == CLOSURE_BSSRDF_BURLEY_ID);
}

bool hardware_hit_uses_proxy_payload(int2 texel)
{
  return imageLoadFast(hit_albedo_img, texel).a < 0.0f;
}

bool hardware_hit_load(int2 texel, float3 &P_hit, float3 &V)
{
  int2 texel_fullres;
  float4 ray_data_im;
  float ray_time;
  if (!hardware_ray_load(texel, texel_fullres, ray_data_im, ray_time) || ray_time <= 0.0f) {
    return false;
  }

  float depth = reverse_z::read(texelFetch(depth_tx, texel_fullres, 0).r);
  if (!(depth > 0.0f && depth < 1.0f)) {
    return false;
  }

  float2 uv = (float2(texel_fullres) + 0.5f) * uniform_buf.raytrace.full_resolution_inv;
  float3 ray_direction = normalize(ray_data_im.xyz);
  if (!hardware_hit_direction_load(texel, ray_direction)) {
    ray_direction = normalize(ray_data_im.xyz);
  }
  P_hit = texelFetch(hit_world_position_tx, texel, 0).xyz;
  if (!(dot(P_hit, P_hit) > 1.0e-10f)) {
    float3 P = drw_point_screen_to_world(float3(uv, depth));
    P_hit = P + ray_direction * ray_time;
  }
  V = -ray_direction;
  return true;
}

bool hardware_hit_normal_load(int2 texel, float3 &N)
{
  N = imageLoadFast(hit_normal_img, texel).rgb;
  return dot(N, N) > 1.0e-10f;
}

bool hardware_hit_shadow_payload_valid(int2 texel)
{
  float3 shadow_N = imageLoadFast(hit_normal_img, texel).rgb;
  if (!(isfinite(shadow_N.x) && isfinite(shadow_N.y) && isfinite(shadow_N.z)) ||
      dot(shadow_N, shadow_N) <= 1.0e-10f)
  {
    return false;
  }

  float3 shadow_P = texelFetch(hit_world_position_tx, texel, 0).xyz;
  return isfinite(shadow_P.x) && isfinite(shadow_P.y) && isfinite(shadow_P.z) &&
         dot(shadow_P, shadow_P) > 1.0e-10f;
}

bool hardware_hit_is_preserved_layered_scene_final(int2 texel)
{
  return (imageLoadFast(hit_identity_img, texel).z & 2u) != 0u;
}

bool hardware_hit_is_preserved_transparent_scene_final(int2 texel)
{
  return (imageLoadFast(hit_identity_img, texel).z & 4u) != 0u;
}

float4 hardware_hit_transmission_layer_load(int2 texel)
{
  return texelFetch(hit_transmission_layer_tx, texel, 0);
}

bool hardware_hit_object_infos_load(int2 texel, ObjectInfos &object_infos)
{
  const uint resource_id = imageLoadFast(hit_identity_img, texel).w;
  if (resource_id == 0xFFFFFFFFu) {
    object_infos = ObjectInfos();
    return false;
  }
  object_infos = drw_infos[resource_id];
  return true;
}

bool hardware_hit_visible_surface_position_matches(float3 P_hit,
                                                   int2 lookup_texel,
                                                   float lookup_depth)
{
  const int2 extent = textureSize(depth_tx, 0);
  const int2 lookup_texel_x = min(lookup_texel + int2(1, 0), extent - 1);
  const int2 lookup_texel_y = min(lookup_texel + int2(0, 1), extent - 1);

  const float2 lookup_uv = (float2(lookup_texel) + 0.5f) / float2(extent);
  const float2 lookup_uv_x = (float2(lookup_texel_x) + 0.5f) / float2(extent);
  const float2 lookup_uv_y = (float2(lookup_texel_y) + 0.5f) / float2(extent);

  const float3 lookup_P = drw_point_screen_to_world(float3(lookup_uv, lookup_depth));
  const float3 lookup_Px = drw_point_screen_to_world(float3(lookup_uv_x, lookup_depth));
  const float3 lookup_Py = drw_point_screen_to_world(float3(lookup_uv_y, lookup_depth));

  const float footprint = max(length(lookup_Px - lookup_P), length(lookup_Py - lookup_P));
  const float position_tolerance_scale =
      (uniform_buf.raytrace.hardware_trace_phase == HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR) ? 3.0f :
                                                                                            1.5f;
  const float position_tolerance = max(footprint * position_tolerance_scale, 1.0e-3f);
  return distance(lookup_P, P_hit) <= position_tolerance;
}

bool hardware_hit_visible_surface_lookup_matches(int2 texel,
                                                 int2 lookup_texel,
                                                 float3 P_hit,
                                                 float3 N_hit,
                                                 bool allow_opposite_normals)
{
  if (any(lessThan(lookup_texel, int2(0))) ||
      any(greaterThanEqual(lookup_texel, textureSize(depth_tx, 0))))
  {
    return false;
  }

  float lookup_depth = reverse_z::read(texelFetch(depth_tx, lookup_texel, 0).r);
  if (!(lookup_depth > 0.0f && lookup_depth < 1.0f)) {
    return false;
  }

  const uint hit_object_id = imageLoadFast(hit_identity_img, texel).w;
  const gbuffer::Header gbuf_header = gbuffer::read_header(lookup_texel);
  const bool position_matches = hardware_hit_visible_surface_position_matches(
      P_hit, lookup_texel, lookup_depth);
  if (gbuf_header.use_object_id()) {
    const uint visible_object_id = gbuffer::read_object_id(lookup_texel);
    /* Sparse replay resolves real hits to a concrete object id in `hit_identity.w`, but miss
     * payloads only preserve the last specular surface point/normal. Let those miss payloads fall
     * back to a normal-only match so scene-final specular can still reuse the directly visible
     * raster sample of that last metal/glass surface. */
    if ((visible_object_id == 0u) ||
        ((hit_object_id != 0xFFFFFFFFu) && (visible_object_id != hit_object_id)))
    {
      return false;
    }
  }
  else if (!position_matches) {
    /* Ordinary opaque surfaces often skip the optional object-id payload entirely. Use a tight
     * position check in that case so visible floors and walls can still seed scene-final specular
     * from the already rendered raster sample instead of falling back to coarse replay lighting. */
    return false;
  }

  if (uniform_buf.raytrace.hardware_trace_phase == HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR) {
    /* The late mirror/refraction resolve should prefer the already composed visible surface once
     * the projected hit lands on the right object/position. Requiring close normal agreement here
     * rejects valid curved-surface matches and pushes the pixel back into the coarse hit-lighting
     * fallback, which is exactly the dark/noisy artifact seen on reflected spheres and floor
     * patches. */
    return position_matches;
  }

  const gbuffer::Layers gbuf = gbuffer::read_layers(lookup_texel);
  const float normal_alignment = dot(gbuf.surface_N(), N_hit);
  return allow_opposite_normals ? abs(normal_alignment) > 0.8f : normal_alignment > 0.8f;
}

bool hardware_hit_visible_surface_lookup_refine(int2 texel,
                                                int2 lookup_texel,
                                                float3 P_hit,
                                                float3 N_hit,
                                                bool allow_opposite_normals,
                                                int2 &refined_lookup_texel)
{
  const bool scene_final_specular_phase =
      (uniform_buf.raytrace.hardware_trace_phase == HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR);
  const int search_radius = scene_final_specular_phase ? (allow_opposite_normals ? 8 : 6) :
                                                         (allow_opposite_normals ? 2 : 1);
  for (int y = -search_radius; y <= search_radius; y++) {
    for (int x = -search_radius; x <= search_radius; x++) {
      const int2 candidate = lookup_texel + int2(x, y);
      if (hardware_hit_visible_surface_lookup_matches(
              texel, candidate, P_hit, N_hit, allow_opposite_normals))
      {
        refined_lookup_texel = candidate;
        return true;
      }
    }
  }
  return false;
}

bool hardware_hit_visible_surface_lookup_texel_load(int2 texel,
                                                    float3 P_hit,
                                                    float3 N_hit,
                                                    bool allow_opposite_normals,
                                                    float2 &lookup_uv,
                                                    int2 &lookup_texel)
{
  lookup_uv = float2(0.0f);
  lookup_texel = int2(0);

  float4 ray_rad = imageLoadFast(ray_radiance_img, texel);
  uint packed_uv = floatBitsToUint(ray_rad.a);
  if (packed_uv != 0u) {
    lookup_uv = float2(float(packed_uv & 0xFFFFu) / 65535.0f,
                       float((packed_uv >> 16u) & 0xFFFFu) / 65535.0f);
    if (all(greaterThanEqual(lookup_uv, float2(0.0f))) && all(lessThan(lookup_uv, float2(1.0f)))) {
      lookup_texel = clamp(int2(lookup_uv * float2(textureSize(depth_tx, 0))),
                           int2(0),
                           textureSize(depth_tx, 0) - 1);
      if (hardware_hit_visible_surface_lookup_matches(
              texel, lookup_texel, P_hit, N_hit, allow_opposite_normals) ||
          hardware_hit_visible_surface_lookup_refine(
              texel, lookup_texel, P_hit, N_hit, allow_opposite_normals, lookup_texel))
      {
        /* Scene-final mirror/refraction is only allowed to replace the pixel from raster when the
         * projected hit still validates against the visible resolved surface. */
        lookup_uv = (float2(lookup_texel) + 0.5f) / float2(textureSize(depth_tx, 0));
        return true;
      }
    }
  }

  float3 screen_P = drw_point_world_to_screen(P_hit);
  lookup_uv = screen_P.xy;
  if (any(lessThan(lookup_uv, float2(0.0f))) || any(greaterThanEqual(lookup_uv, float2(1.0f)))) {
    return false;
  }

  lookup_texel = clamp(int2(lookup_uv * float2(textureSize(depth_tx, 0))),
                       int2(0),
                       textureSize(depth_tx, 0) - 1);
  if (hardware_hit_visible_surface_lookup_matches(
          texel, lookup_texel, P_hit, N_hit, allow_opposite_normals) ||
      hardware_hit_visible_surface_lookup_refine(
          texel, lookup_texel, P_hit, N_hit, allow_opposite_normals, lookup_texel))
  {
    lookup_uv = (float2(lookup_texel) + 0.5f) / float2(textureSize(depth_tx, 0));
    return true;
  }
  return false;
}

bool hardware_hit_environment_visibility_load(int2 texel,
                                              float3 fallback_N,
                                              HardwareEnvironmentVisibilityData &data)
{
  if (any(lessThan(texel, int2(0))) ||
      any(greaterThanEqual(texel, textureSize(hardware_rt_hit_environment_visibility_tx, 0))))
  {
    data.average_direction = float3(0.0f);
    data.visibility = 1.0f;
    data.validity = 0.0f;
    return false;
  }

  float4 visibility_data = texelFetch(hardware_rt_hit_environment_visibility_tx, texel, 0);
  data.average_direction = visibility_data.xyz;
  data.visibility = saturate(visibility_data.w);
  data.validity = hardware_environment_visibility_validity(data.average_direction, data.visibility);
  if (data.validity < 0.5f) {
    data.visibility = 1.0f;
  }
  if (dot(data.average_direction, data.average_direction) <= 1.0e-8f) {
    data.average_direction = safe_normalize(fallback_N) * (2.0f / 3.0f);
  }
  return hardware_environment_visibility_is_valid(data);
}

bool hardware_hit_visible_surface_uses_back_radiance(int2 lookup_texel)
{
  const gbuffer::Layers gbuf = gbuffer::read_layers(lookup_texel);
  const uchar closure_count = gbuf.header.closure_len();
  for (uchar i = 0; i < GBUFFER_LAYER_MAX && i < closure_count; i++) {
    const ClosureUndetermined cl = gbuf.layer_get(i);
    if (closure_has_transmission(cl.type)) {
      return true;
    }
  }
  return false;
}

bool hardware_hit_is_visible_to_main_camera(float3 P_hit)
{
  float3 screen_P = drw_point_world_to_screen(P_hit);
  float2 lookup_uv = screen_P.xy;
  if (any(lessThan(lookup_uv, float2(0.0f))) || any(greaterThanEqual(lookup_uv, float2(1.0f)))) {
    return false;
  }

  int2 lookup_texel = clamp(int2(lookup_uv * float2(textureSize(depth_tx, 0))),
                            int2(0),
                            textureSize(depth_tx, 0) - 1);
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      int2 candidate = clamp(lookup_texel + int2(x, y), int2(0), textureSize(depth_tx, 0) - 1);
      float lookup_depth = reverse_z::read(texelFetch(depth_tx, candidate, 0).r);
      if ((lookup_depth > 0.0f && lookup_depth < 1.0f) &&
          hardware_hit_visible_surface_position_matches(P_hit, candidate, lookup_depth))
      {
        return true;
      }
    }
  }
  return false;
}

bool hardware_hit_raster_radiance_load(int2 texel,
                                       float3 P_hit,
                                       float3 N_hit,
                                       bool allow_opposite_normals,
                                       bool strip_receiver_caustics,
                                       float3 &radiance)
{
  radiance = float3(0.0f);

  float2 lookup_uv;
  int2 lookup_texel;
  if (!hardware_hit_visible_surface_lookup_texel_load(
          texel, P_hit, N_hit, allow_opposite_normals, lookup_uv, lookup_texel))
  {
    return false;
  }

  if (uniform_buf.raytrace.hardware_trace_phase == HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR) {
    radiance = hardware_hit_visible_surface_uses_back_radiance(lookup_texel) ?
                   texelFetch(radiance_back_tx, lookup_texel, 0).rgb :
                   texelFetch(radiance_front_tx, lookup_texel, 0).rgb;
  }
  else {
    radiance = hardware_hit_visible_surface_uses_back_radiance(lookup_texel) ?
                   textureLod(radiance_back_tx, lookup_uv, 0.0f).rgb :
                   textureLod(radiance_front_tx, lookup_uv, 0.0f).rgb;
  }
  if (strip_receiver_caustics) {
    /* Keep sharp visible-surface replay for layered Principled reflections, but do not fold the
     * receiver-only caustics buffer back into that rough diffuse-reflection handoff. */
    radiance = max(radiance - hardware_caustics_load(lookup_texel), float3(0.0f));
  }
  return true;
}

bool hardware_hit_raster_back_radiance_load(int2 texel,
                                            float3 P_hit,
                                            float3 N_hit,
                                            bool allow_opposite_normals,
                                            bool strip_receiver_caustics,
                                            float3 &radiance)
{
  radiance = float3(0.0f);

  float2 lookup_uv;
  int2 lookup_texel;
  if (!hardware_hit_visible_surface_lookup_texel_load(
          texel, P_hit, N_hit, allow_opposite_normals, lookup_uv, lookup_texel))
  {
    return false;
  }

  radiance = (uniform_buf.raytrace.hardware_trace_phase == HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR) ?
                 texelFetch(radiance_back_tx, lookup_texel, 0).rgb :
                 textureLod(radiance_back_tx, lookup_uv, 0.0f).rgb;
  if (strip_receiver_caustics) {
    /* Keep the transmission fallback replay-owned, but do not feed the late receiver-only
     * caustics buffer back through `radiance_back_tx` when caustics are enabled. */
    radiance = max(radiance - hardware_caustics_load(lookup_texel), float3(0.0f));
  }
  return true;
}

float3 hardware_indirect_gi_cache_coord(float3 direction)
{
  float3 abs_direction = abs(direction);
  float2 face_uv;
  float face_index;
  float major_axis;
  if (abs_direction.x >= abs_direction.y && abs_direction.x >= abs_direction.z) {
    major_axis = max(abs_direction.x, 1.0e-8f);
    if (direction.x > 0.0f) {
      face_index = 0.0f;
      face_uv = float2(-direction.z, -direction.y) / major_axis;
    }
    else {
      face_index = 1.0f;
      face_uv = float2(direction.z, -direction.y) / major_axis;
    }
  }
  else if (abs_direction.y >= abs_direction.z) {
    major_axis = max(abs_direction.y, 1.0e-8f);
    if (direction.y > 0.0f) {
      face_index = 2.0f;
      face_uv = float2(direction.x, direction.z) / major_axis;
    }
    else {
      face_index = 3.0f;
      face_uv = float2(direction.x, -direction.z) / major_axis;
    }
  }
  else {
    major_axis = max(abs_direction.z, 1.0e-8f);
    if (direction.z > 0.0f) {
      face_index = 4.0f;
      face_uv = float2(direction.x, -direction.y) / major_axis;
    }
    else {
      face_index = 5.0f;
      face_uv = float2(-direction.x, -direction.y) / major_axis;
    }
  }
  return float3(face_uv * 0.5f + 0.5f, face_index);
}

bool hardware_hit_indirect_gi_cache_radiance_load(float3 P_hit, float3 N_hit, float3 &radiance)
{
  radiance = float3(0.0f);
  if (hardware_hit_is_visible_to_main_camera(P_hit)) {
    return false;
  }

  float3 camera_to_hit = P_hit - drw_view_position();
  float hit_distance = length(camera_to_hit);
  if (!(hit_distance > 1.0e-4f)) {
    return false;
  }

  float3 cache_direction = camera_to_hit / hit_distance;
  float3 cache_coord = hardware_indirect_gi_cache_coord(cache_direction);
  float4 cached_position = textureLod(hardware_indirect_gi_position_cache_tx, cache_coord, 0.0f);
  if (cached_position.a < 0.5f) {
    return false;
  }

  float3 cached_normal = textureLod(hardware_indirect_gi_normal_cache_tx, cache_coord, 0.0f).xyz;
  if (dot(cached_normal, cached_normal) <= 1.0e-8f) {
    return false;
  }
  cached_normal = normalize(cached_normal);

  int cache_extent = max(textureSize(hardware_indirect_gi_radiance_cache_tx, 0).x, 1);
  float distance_tolerance = max(hit_distance / float(cache_extent), 0.05f) * 2.5f;
  if (distance(P_hit, cached_position.xyz) > distance_tolerance) {
    return false;
  }
  if (dot(cached_normal, N_hit) < 0.25f) {
    return false;
  }

  radiance = max(textureLod(hardware_indirect_gi_radiance_cache_tx, cache_coord, 0.0f).rgb,
                 float3(0.0f));
  return true;
}

/* Indirect/base-family simplification:
 * - proxy payloads keep at most one base-family lobe,
 * - subsurface collapses to diffuse,
 * - specular-family proxy closures are dropped from the base/indirect path. */
ClosureUndetermined hardware_hit_base_closure_load(int2 texel, float3 N)
{
  float4 hit_base = imageLoadFast(hit_albedo_img, texel);
  float4 hit_material = imageLoadFast(hit_material_img, texel);
  const bool proxy_payload = hardware_hit_uses_proxy_payload(texel);
  ClosureType type = proxy_payload ? hardware_hit_closure_type_unpack(hit_material.z) :
                                     hardware_hit_closure_type_unpack(hit_base.a);
  if (proxy_payload && !hardware_hit_closure_is_base_family(type)) {
    type = CLOSURE_NONE_ID;
  }
  if (type == CLOSURE_BSSRDF_BURLEY_ID) {
    type = CLOSURE_BSDF_DIFFUSE_ID;
  }

  ClosureUndetermined cl = closure_new(type);
  cl.weight = 1.0f;
  cl.color = hit_base.rgb;
  cl.N = N;
  cl.data = float4(0.0f);

  switch (cl.type) {
    case CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID:
      cl.data.x = hit_material.x;
      cl.data.y = hit_material.y;
      break;
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
      cl.data.x = hit_material.x;
      cl.data.y = hit_material.y;
      break;
    case CLOSURE_BSDF_TRANSLUCENT_ID:
    case CLOSURE_BSDF_DIFFUSE_ID:
    case CLOSURE_BSSRDF_BURLEY_ID:
    case CLOSURE_NONE_ID:
      break;
  }
  return cl;
}

/* Direct/specular simplification:
 * - replay or proxy fallback keeps at most one dominant specular-family lobe,
 * - proxy payloads reuse the coarse base tint as the bounded fallback color,
 * - proxy-only hits do not carry thickness. */
ClosureUndetermined hardware_hit_specular_closure_load(int2 texel, float3 N)
{
  float4 hit_base = imageLoadFast(hit_albedo_img, texel);
  float4 hit_material = imageLoadFast(hit_material_img, texel);
  float4 hit_specular = imageLoadFast(hit_position_img, texel);
  const bool proxy_payload = hardware_hit_uses_proxy_payload(texel);
  const ClosureType type = hardware_hit_closure_type_unpack(hit_material.z);

  ClosureUndetermined cl = closure_new(
      hardware_hit_closure_is_specular_family(type) ? type : CLOSURE_NONE_ID);
  cl.weight = 1.0f;
  cl.color = proxy_payload ? hit_base.rgb : hit_specular.rgb;
  cl.N = N;
  cl.data = float4(0.0f);

  switch (cl.type) {
    case CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID:
      cl.data.x = hit_material.x;
      cl.data.y = hit_material.y;
      break;
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
      cl.data.x = hit_material.x;
      cl.data.y = hit_material.y;
      break;
    case CLOSURE_BSDF_TRANSLUCENT_ID:
    case CLOSURE_BSDF_DIFFUSE_ID:
    case CLOSURE_BSSRDF_BURLEY_ID:
    case CLOSURE_NONE_ID:
      break;
  }

  return cl;
}

Thickness hardware_hit_thickness_load(int2 texel)
{
  if (hardware_hit_uses_proxy_payload(texel)) {
    return Thickness::zero();
  }
  return gbuffer::thickness_unpack(imageLoadFast(hit_position_img, texel).w);
}

bool layered_receiver_hit_exists(int2 texel)
{
  return texelFetch(layered_receiver_ray_time_tx, texel, 0).r > 0.0f;
}

float4 layered_receiver_hit_throughput_load(int2 texel)
{
  return texelFetch(layered_receiver_throughput_tx, texel, 0);
}

bool layered_receiver_hit_uses_proxy_payload(int2 texel)
{
  return texelFetch(layered_receiver_hit_albedo_tx, texel, 0).a < 0.0f;
}

bool layered_receiver_hit_direction_load(int2 texel, float3 &ray_direction)
{
  float2 packed_dir = float2(texelFetch(layered_receiver_hit_material_tx, texel, 0).w,
                             texelFetch(layered_receiver_hit_normal_tx, texel, 0).w);
  if (all(equal(packed_dir, float2(0.0f)))) {
    return false;
  }
  ray_direction = hardware_direction_unpack(packed_dir);
  return isfinite(ray_direction.x) && isfinite(ray_direction.y) && isfinite(ray_direction.z) &&
         dot(ray_direction, ray_direction) > 1.0e-10f;
}

bool layered_receiver_hit_load(int2 texel, float3 &P_hit, float3 &V)
{
  if (!layered_receiver_hit_exists(texel)) {
    return false;
  }
  float3 ray_direction;
  if (!layered_receiver_hit_direction_load(texel, ray_direction)) {
    return false;
  }
  P_hit = texelFetch(layered_receiver_world_position_tx, texel, 0).xyz;
  if (!(isfinite(P_hit.x) && isfinite(P_hit.y) && isfinite(P_hit.z)) ||
      dot(P_hit, P_hit) <= 1.0e-10f)
  {
    return false;
  }
  V = -ray_direction;
  return true;
}

bool layered_receiver_hit_normal_load(int2 texel, float3 &N)
{
  N = texelFetch(layered_receiver_hit_normal_tx, texel, 0).rgb;
  return dot(N, N) > 1.0e-10f;
}

bool layered_receiver_hit_shadow_payload_valid(int2 texel)
{
  float3 shadow_N = texelFetch(layered_receiver_hit_normal_tx, texel, 0).rgb;
  if (!(isfinite(shadow_N.x) && isfinite(shadow_N.y) && isfinite(shadow_N.z)) ||
      dot(shadow_N, shadow_N) <= 1.0e-10f)
  {
    return false;
  }

  float3 shadow_P = texelFetch(layered_receiver_world_position_tx, texel, 0).xyz;
  return isfinite(shadow_P.x) && isfinite(shadow_P.y) && isfinite(shadow_P.z) &&
         dot(shadow_P, shadow_P) > 1.0e-10f;
}

bool layered_receiver_hit_object_infos_load(int2 texel, ObjectInfos &object_infos)
{
  const uint resource_id = texelFetch(layered_receiver_hit_identity_tx, texel, 0).w;
  if (resource_id == 0xFFFFFFFFu) {
    object_infos = ObjectInfos();
    return false;
  }
  object_infos = drw_infos[resource_id];
  return true;
}

ClosureUndetermined layered_receiver_hit_base_closure_load(int2 texel, float3 N)
{
  float4 hit_base = texelFetch(layered_receiver_hit_albedo_tx, texel, 0);
  float4 hit_material = texelFetch(layered_receiver_hit_material_tx, texel, 0);
  const bool proxy_payload = layered_receiver_hit_uses_proxy_payload(texel);
  ClosureType type = proxy_payload ? hardware_hit_closure_type_unpack(hit_material.z) :
                                     hardware_hit_closure_type_unpack(hit_base.a);
  if (proxy_payload && !hardware_hit_closure_is_base_family(type)) {
    type = CLOSURE_NONE_ID;
  }
  if (type == CLOSURE_BSSRDF_BURLEY_ID) {
    type = CLOSURE_BSDF_DIFFUSE_ID;
  }

  ClosureUndetermined cl = closure_new(type);
  cl.weight = 1.0f;
  cl.color = hit_base.rgb;
  cl.N = N;
  cl.data = float4(0.0f);

  switch (cl.type) {
    case CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID:
      cl.data.x = hit_material.x;
      cl.data.y = hit_material.y;
      break;
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
      cl.data.x = hit_material.x;
      cl.data.y = hit_material.y;
      break;
    case CLOSURE_BSDF_TRANSLUCENT_ID:
    case CLOSURE_BSDF_DIFFUSE_ID:
    case CLOSURE_BSSRDF_BURLEY_ID:
    case CLOSURE_NONE_ID:
      break;
  }
  return cl;
}

ClosureUndetermined layered_receiver_hit_specular_closure_load(int2 texel, float3 N)
{
  float4 hit_base = texelFetch(layered_receiver_hit_albedo_tx, texel, 0);
  float4 hit_material = texelFetch(layered_receiver_hit_material_tx, texel, 0);
  float4 hit_specular = texelFetch(layered_receiver_hit_position_tx, texel, 0);
  const bool proxy_payload = layered_receiver_hit_uses_proxy_payload(texel);
  const ClosureType type = hardware_hit_closure_type_unpack(hit_material.z);

  ClosureUndetermined cl = closure_new(
      hardware_hit_closure_is_specular_family(type) ? type : CLOSURE_NONE_ID);
  cl.weight = 1.0f;
  cl.color = proxy_payload ? hit_base.rgb : hit_specular.rgb;
  cl.N = N;
  cl.data = float4(0.0f);

  switch (cl.type) {
    case CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID:
      cl.data.x = hit_material.x;
      break;
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
      cl.data.x = hit_material.x;
      cl.data.y = hit_material.y;
      break;
    case CLOSURE_BSDF_TRANSLUCENT_ID:
    case CLOSURE_BSDF_DIFFUSE_ID:
    case CLOSURE_BSSRDF_BURLEY_ID:
    case CLOSURE_NONE_ID:
      break;
  }

  return cl;
}

Thickness layered_receiver_hit_thickness_load(int2 texel)
{
  if (layered_receiver_hit_uses_proxy_payload(texel)) {
    return Thickness::zero();
  }
  return gbuffer::thickness_unpack(texelFetch(layered_receiver_hit_position_tx, texel, 0).w);
}

bool hardware_hit_closure_has_energy(ClosureUndetermined cl)
{
  return (cl.type != CLOSURE_NONE_ID) && (dot(cl.color, cl.color) > 1.0e-10f);
}

float hardware_hit_closure_color_strength(ClosureUndetermined cl)
{
  float3 color = abs(cl.color);
  return max(color.x, max(color.y, color.z));
}

bool hardware_hit_use_exact_local_lights()
{
  return (light_cull_buf.local_lights_len > 0u) && (light_cull_buf.local_lights_len <= 8u);
}

LightData hardware_hit_exact_local_light(uint local_light_index)
{
  return light_buf_no_cull[light_cull_buf.sun_lights_len + local_light_index];
}

void hardware_hit_light_eval_exact_local(uint local_light_index,
                                         const bool is_transmission,
                                         ClosureLightStack &stack,
                                         float3 P,
                                         float3 Ng,
                                         float3 V,
                                         Thickness thickness,
                                         uchar receiver_light_set,
                                         float terminator_normal_offset,
                                         float terminator_geometry_offset)
{
  LightData light = hardware_hit_exact_local_light(local_light_index);

  if (!light_linking_affects_receiver(light.light_set_membership, receiver_light_set)) {
    return;
  }

#if defined(SPECIALIZED_SHADOW_PARAMS)
  int ray_count = shadow_ray_count;
  int ray_step_count = shadow_ray_step_count;
#else
  int ray_count = uniform_buf.shadow.ray_count;
  int ray_step_count = uniform_buf.shadow.step_count;
#endif

  LightVector lv = light_vector_get(light, false, P);
  bool is_translucent_with_thickness = is_transmission &&
                                       (stack.cl[0].type == LIGHT_TRANSLUCENT_WITH_THICKNESS);
  float attenuation = light_attenuation_surface(light, false, lv);

  if (!is_translucent_with_thickness) {
    attenuation *= light_attenuation_facing(light, lv.L, lv.dist, stack.cl[0].N, is_transmission);
  }

  if (attenuation < LIGHT_ATTENUATION_THRESHOLD) {
    return;
  }

  float shadow = 1.0f;
  if (light.tilemap_index != LIGHT_NO_SHADOW) {
    shadow = shadow_eval_dispatch(local_light_index,
                                  light,
                                  false,
                                  is_transmission,
                                  is_translucent_with_thickness,
                                  thickness,
                                  P,
                                  Ng,
                                  stack.cl[0].N,
                                  terminator_normal_offset,
                                  terminator_geometry_offset,
                                  ray_count,
                                  ray_step_count);
  }

  if (is_translucent_with_thickness) {
    stack.cl[0].N = lv.L;
    attenuation *= M_1_PI;
  }

  light_eval_single_closure(light, lv, stack.cl[0], V, attenuation, shadow);
  if (!is_transmission) {
#if LIGHT_CLOSURE_EVAL_COUNT > 1
    light_eval_single_closure(light, lv, stack.cl[1], V, attenuation, shadow);
#endif
#if LIGHT_CLOSURE_EVAL_COUNT > 2
    light_eval_single_closure(light, lv, stack.cl[2], V, attenuation, shadow);
#endif
#if LIGHT_CLOSURE_EVAL_COUNT > 3
#  error
#endif
  }
}

bool hardware_hit_allows_raster_reuse(int2 texel,
                                      bool preserve_screen_baseline,
                                      bool has_replayed_material,
                                      float3 existing_radiance,
                                      ClosureUndetermined base_cl,
                                      ClosureUndetermined specular_cl)
{
  if (preserve_screen_baseline) {
    return false;
  }
  if (has_replayed_material) {
    return false;
  }
  if (dot(existing_radiance, existing_radiance) > 1.0e-10f) {
    return false;
  }
  return hardware_hit_closure_has_energy(base_cl) || hardware_hit_closure_has_energy(specular_cl);
}

bool hardware_hit_closure_uses_environment_visibility(ClosureUndetermined cl,
                                                      bool primary_is_diffuse_gi)
{
  if (!use_hardware_rt_environment_visibility || primary_is_diffuse_gi) {
    return false;
  }
  return true;
}

void hardware_hit_closure_light_terms(int2 texel_fullres,
                                      int2 texel,
                                      float3 P_hit,
                                      float3 N,
                                      float3 V,
                                      ClosureUndetermined cl,
                                      Thickness thickness,
                                      bool primary_is_diffuse_gi,
                                      float3 &direct_radiance,
                                      float3 &probe_radiance,
                                      bool &probe_uses_world)
{
  direct_radiance = float3(0.0f);
  probe_radiance = float3(0.0f);
  probe_uses_world = false;
  if (!hardware_hit_closure_has_energy(cl)) {
    return;
  }

  const bool scene_final_specular_phase =
      (uniform_buf.raytrace.hardware_trace_phase == HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR);
  const bool is_transmission = closure_has_transmission(cl.type);
  const bool is_diffuse_family = (cl.type == CLOSURE_BSDF_DIFFUSE_ID) ||
                                 (cl.type == CLOSURE_BSSRDF_BURLEY_ID);
  const bool scene_final_reflected_diffuse = scene_final_specular_phase && is_diffuse_family;
  const bool direct_lit_refracted_textured_receiver =
      scene_final_specular_phase && !primary_is_diffuse_gi &&
      ((imageLoadFast(hit_identity_img, texel).z & 16u) != 0u);
  const bool suppress_scene_final_direct_hit_light = scene_final_specular_phase &&
                                                     !primary_is_diffuse_gi &&
                                                     !direct_lit_refracted_textured_receiver &&
                                                     (is_transmission ||
                                                      hardware_hit_closure_is_specular_family(cl.type));
  const Thickness cl_thickness = is_transmission ? thickness : Thickness::zero();
  uchar receiver_light_set = 0u;
  float normal_offset = 0.0f;
  float geometry_offset = 0.0f;
  ObjectInfos object_infos;
  if (hardware_hit_object_infos_load(texel, object_infos)) {
    receiver_light_set = receiver_light_set_get(object_infos);
    normal_offset = object_infos.shadow_terminator_normal_offset;
    geometry_offset = object_infos.shadow_terminator_geometry_offset;
  }
  shadow_dispatch_texel_fullres = texel;
  shadow_dispatch_use_hardware_rt = false;
  if ((use_hardware_rt_shadows || use_hardware_rt_environment_visibility) &&
      !direct_lit_refracted_textured_receiver)
  {
    shadow_dispatch_use_hardware_rt = hardware_hit_shadow_payload_valid(texel);
  }
  ClosureLightStack stack;
  ClosureUndetermined light_cl = hardware_hit_refracted_metal_direct_closure(
      cl, direct_lit_refracted_textured_receiver);
  stack.cl[0] = is_transmission ? closure_light_new(light_cl, V, cl_thickness) :
                                  closure_light_new(light_cl, V);
  LIGHT_FOREACH_BEGIN_DIRECTIONAL (light_cull_buf, l_idx) {
    light_eval_single(
        l_idx,
        true,
        is_transmission,
        stack,
        P_hit,
        N,
        V,
        cl_thickness,
        receiver_light_set,
        normal_offset,
        geometry_offset);
  }
  LIGHT_FOREACH_END

  if (hardware_hit_use_exact_local_lights()) {
    for (uint local_light_index = 0u; local_light_index < light_cull_buf.local_lights_len;
         local_light_index++)
    {
      hardware_hit_light_eval_exact_local(local_light_index,
                                          is_transmission,
                                          stack,
                                          P_hit,
                                          N,
                                          V,
                                          cl_thickness,
                                          receiver_light_set,
                                          normal_offset,
                                          geometry_offset);
    }
  }
  else {
    LIGHT_FOREACH_BEGIN_LOCAL_NO_CULL(light_cull_buf, l_idx) {
      light_eval_single(
          l_idx,
          false,
          is_transmission,
          stack,
          P_hit,
          N,
          V,
          cl_thickness,
          receiver_light_set,
          normal_offset,
          geometry_offset);
    }
    LIGHT_FOREACH_END
  }

  LightProbeSample samp = lightprobe_load(float2(texel_fullres), P_hit, N, V);
  probe_uses_world = lightprobe_uses_world(samp);
  float3 probe_light = lightprobe_eval(samp, cl, P_hit, V, cl_thickness);
  const bool use_dome_world_probe = hardware_fast_gi_enabled_for_diffuse() &&
                                    is_diffuse_family &&
                                    probe_uses_world &&
                                    (primary_is_diffuse_gi || !use_hardware_environment);
  if (!use_hardware_environment && probe_uses_world && !primary_is_diffuse_gi &&
      !scene_final_reflected_diffuse)
  {
    probe_light = float3(0.0f);
  }
  else if (hardware_hit_closure_uses_environment_visibility(cl, primary_is_diffuse_gi) &&
           probe_uses_world)
  {
    HardwareEnvironmentVisibilityData env_visibility;
    if (hardware_hit_environment_visibility_load(texel, N, env_visibility)) {
      LightProbeRay probe_ray = bxdf_lightprobe_ray(cl, P_hit, V, cl_thickness);
      float3 world_direction = is_diffuse_family ?
                                   hardware_environment_visibility_direction(
                                       env_visibility, probe_ray.dominant_direction, N) :
                                   probe_ray.dominant_direction;
      probe_light = lightprobe_eval_with_direction(
          samp, cl, P_hit, V, cl_thickness, world_direction);
      if (is_diffuse_family) {
        float diffuse_world_visibility = square(saturate((env_visibility.visibility - 0.05f) / 0.95f));
        probe_light *= diffuse_world_visibility;
      }
      else {
        probe_light *= env_visibility.visibility;
      }
    }
  }
  if (use_dome_world_probe) {
    /* Let GI fully own world transport for diffuse-primary and GI-only cases. */
    probe_light = hardware_fast_gi_sample(P_hit);
  }
  else if (scene_final_reflected_diffuse && hardware_fast_gi_enabled_for_diffuse()) {
    /* Scene-final reflected diffuse must match the visible deferred surface path: TFC/Fast GI is
     * the diffuse GI owner even when the reflected receiver resolves to a local probe. */
    probe_light += hardware_fast_gi_sample(P_hit);
  }

  float3 shading_color = max(cl.color, float3(0.0f));
  direct_radiance = suppress_scene_final_direct_hit_light ?
                        float3(0.0f) :
                        (stack.cl[0].light_shadowed * shading_color);
  if (is_diffuse_family) {
    probe_radiance = probe_light * shading_color;
    return;
  }
  probe_radiance = probe_light * shading_color;
}

void layered_receiver_hit_closure_light_terms(int2 texel_fullres,
                                              int2 texel,
                                              float3 P_hit,
                                              float3 N,
                                              float3 V,
                                              ClosureUndetermined cl,
                                              Thickness thickness,
                                              bool primary_is_diffuse_gi,
                                              float3 &direct_radiance,
                                              float3 &probe_radiance)
{
  direct_radiance = float3(0.0f);
  probe_radiance = float3(0.0f);
  if (!hardware_hit_closure_has_energy(cl)) {
    return;
  }

  const bool scene_final_specular_phase =
      (uniform_buf.raytrace.hardware_trace_phase == HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR);
  const bool is_transmission = closure_has_transmission(cl.type);
  const bool is_diffuse_family = (cl.type == CLOSURE_BSDF_DIFFUSE_ID) ||
                                 (cl.type == CLOSURE_BSSRDF_BURLEY_ID);
  const bool scene_final_reflected_diffuse = scene_final_specular_phase && is_diffuse_family;
  const bool direct_lit_refracted_textured_receiver =
      scene_final_specular_phase && !primary_is_diffuse_gi &&
      ((texelFetch(layered_receiver_hit_identity_tx, texel, 0).z & 16u) != 0u);
  const bool suppress_scene_final_direct_hit_light = scene_final_specular_phase &&
                                                     !primary_is_diffuse_gi &&
                                                     !direct_lit_refracted_textured_receiver &&
                                                     (is_transmission ||
                                                      hardware_hit_closure_is_specular_family(cl.type));
  const Thickness cl_thickness = is_transmission ? thickness : Thickness::zero();
  uchar receiver_light_set = 0u;
  float normal_offset = 0.0f;
  float geometry_offset = 0.0f;
  ObjectInfos object_infos;
  if (layered_receiver_hit_object_infos_load(texel, object_infos)) {
    receiver_light_set = receiver_light_set_get(object_infos);
    normal_offset = object_infos.shadow_terminator_normal_offset;
    geometry_offset = object_infos.shadow_terminator_geometry_offset;
  }
  shadow_dispatch_texel_fullres = texel;
  shadow_dispatch_use_hardware_rt = false;
  if ((use_hardware_rt_shadows || use_hardware_rt_environment_visibility) &&
      !direct_lit_refracted_textured_receiver)
  {
    shadow_dispatch_use_hardware_rt = layered_receiver_hit_shadow_payload_valid(texel);
  }
  ClosureLightStack stack;
  ClosureUndetermined light_cl = hardware_hit_refracted_metal_direct_closure(
      cl, direct_lit_refracted_textured_receiver);
  stack.cl[0] = is_transmission ? closure_light_new(light_cl, V, cl_thickness) :
                                  closure_light_new(light_cl, V);
  LIGHT_FOREACH_BEGIN_DIRECTIONAL (light_cull_buf, l_idx) {
    light_eval_single(
        l_idx,
        true,
        is_transmission,
        stack,
        P_hit,
        N,
        V,
        cl_thickness,
        receiver_light_set,
        normal_offset,
        geometry_offset);
  }
  LIGHT_FOREACH_END

  if (hardware_hit_use_exact_local_lights()) {
    for (uint local_light_index = 0u; local_light_index < light_cull_buf.local_lights_len;
         local_light_index++)
    {
      hardware_hit_light_eval_exact_local(local_light_index,
                                          is_transmission,
                                          stack,
                                          P_hit,
                                          N,
                                          V,
                                          cl_thickness,
                                          receiver_light_set,
                                          normal_offset,
                                          geometry_offset);
    }
  }
  else {
    LIGHT_FOREACH_BEGIN_LOCAL_NO_CULL(light_cull_buf, l_idx) {
      light_eval_single(
          l_idx,
          false,
          is_transmission,
          stack,
          P_hit,
          N,
          V,
          cl_thickness,
          receiver_light_set,
          normal_offset,
          geometry_offset);
    }
    LIGHT_FOREACH_END
  }

  LightProbeSample samp = lightprobe_load(float2(texel_fullres), P_hit, N, V);
  float3 probe_light = lightprobe_eval(samp, cl, P_hit, V, cl_thickness);
  const bool use_dome_world_probe = hardware_fast_gi_enabled_for_diffuse() &&
                                    is_diffuse_family &&
                                    lightprobe_uses_world(samp) &&
                                    (primary_is_diffuse_gi || !use_hardware_environment);
  if (!use_hardware_environment && lightprobe_uses_world(samp) && !primary_is_diffuse_gi &&
      !scene_final_reflected_diffuse)
  {
    probe_light = float3(0.0f);
  }
  else if (hardware_hit_closure_uses_environment_visibility(cl, primary_is_diffuse_gi) &&
           lightprobe_uses_world(samp))
  {
    /* Receiver shading does not have its own environment-visibility buffer yet, but transmission
     * receivers still need the traced transmitted/world direction rather than the front mirror texel
     * fallback or they lose the HDRI/world contribution entirely on miss. */
    if (hardware_hit_closure_is_specular_family(cl.type)) {
      probe_light = lightprobe_eval_with_direction(samp, cl, P_hit, V, cl_thickness, -V);
    }
  }
  if (use_dome_world_probe) {
    probe_light = hardware_fast_gi_sample(P_hit);
  }
  else if (scene_final_reflected_diffuse && hardware_fast_gi_enabled_for_diffuse()) {
    probe_light += hardware_fast_gi_sample(P_hit);
  }

  float3 shading_color = max(cl.color, float3(0.0f));
  direct_radiance = suppress_scene_final_direct_hit_light ?
                        float3(0.0f) :
                        (stack.cl[0].light_shadowed * shading_color);
  probe_radiance = probe_light * shading_color;
}

float3 layered_receiver_hit_radiance_resolve(int2 texel, int2 texel_fullres, bool primary_is_diffuse_gi)
{
  float3 P_hit, V;
  if (!layered_receiver_hit_load(texel, P_hit, V)) {
    return float3(0.0f);
  }

  float3 N;
  if (!layered_receiver_hit_normal_load(texel, N)) {
    return float3(0.0f);
  }

  Thickness thickness = layered_receiver_hit_thickness_load(texel);
  ClosureUndetermined base_cl = layered_receiver_hit_base_closure_load(texel, N);
  ClosureUndetermined specular_cl = layered_receiver_hit_specular_closure_load(texel, N);

  float3 radiance = texelFetch(layered_receiver_ray_radiance_tx, texel, 0).rgb;
  float3 base_direct = float3(0.0f);
  float3 base_probe = float3(0.0f);
  float3 specular_direct = float3(0.0f);
  float3 specular_probe = float3(0.0f);
  layered_receiver_hit_closure_light_terms(
      texel_fullres, texel, P_hit, N, V, base_cl, thickness, primary_is_diffuse_gi, base_direct, base_probe);
  layered_receiver_hit_closure_light_terms(texel_fullres,
                                           texel,
                                           P_hit,
                                           N,
                                           V,
                                           specular_cl,
                                           thickness,
                                           primary_is_diffuse_gi,
                                           specular_direct,
                                           specular_probe);
  radiance += base_direct + specular_direct;
  if (primary_is_diffuse_gi ||
      (uniform_buf.raytrace.hardware_trace_phase == HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR))
  {
    radiance += base_probe + specular_probe;
  }
  float4 carried_throughput = layered_receiver_hit_throughput_load(texel);
  if (carried_throughput.a > 0.5f) {
    radiance *= max(carried_throughput.rgb, float3(0.0f));
  }
  return radiance;
}

bool transmission_receiver_hit_exists(int2 texel)
{
  return texelFetch(transmission_receiver_ray_time_tx, texel, 0).r > 0.0f;
}

float4 transmission_receiver_hit_throughput_load(int2 texel)
{
  return texelFetch(transmission_receiver_throughput_tx, texel, 0);
}

bool transmission_receiver_hit_uses_proxy_payload(int2 texel)
{
  return texelFetch(transmission_receiver_hit_albedo_tx, texel, 0).a < 0.0f;
}

bool transmission_receiver_hit_direction_load(int2 texel, float3 &ray_direction)
{
  float2 packed_dir = float2(texelFetch(transmission_receiver_hit_material_tx, texel, 0).w,
                             texelFetch(transmission_receiver_hit_normal_tx, texel, 0).w);
  if (all(equal(packed_dir, float2(0.0f)))) {
    return false;
  }
  ray_direction = hardware_direction_unpack(packed_dir);
  return isfinite(ray_direction.x) && isfinite(ray_direction.y) && isfinite(ray_direction.z) &&
         dot(ray_direction, ray_direction) > 1.0e-10f;
}

bool transmission_receiver_hit_load(int2 texel, float3 &P_hit, float3 &V)
{
  if (!transmission_receiver_hit_exists(texel)) {
    return false;
  }
  float3 ray_direction;
  if (!transmission_receiver_hit_direction_load(texel, ray_direction)) {
    return false;
  }
  P_hit = texelFetch(transmission_receiver_world_position_tx, texel, 0).xyz;
  if (!(isfinite(P_hit.x) && isfinite(P_hit.y) && isfinite(P_hit.z)) ||
      dot(P_hit, P_hit) <= 1.0e-10f)
  {
    return false;
  }
  V = -ray_direction;
  return true;
}

bool transmission_receiver_hit_normal_load(int2 texel, float3 &N)
{
  N = texelFetch(transmission_receiver_hit_normal_tx, texel, 0).rgb;
  return dot(N, N) > 1.0e-10f;
}

bool transmission_receiver_hit_shadow_payload_valid(int2 texel)
{
  float3 shadow_N = texelFetch(transmission_receiver_hit_normal_tx, texel, 0).rgb;
  if (!(isfinite(shadow_N.x) && isfinite(shadow_N.y) && isfinite(shadow_N.z)) ||
      dot(shadow_N, shadow_N) <= 1.0e-10f)
  {
    return false;
  }

  float3 shadow_P = texelFetch(transmission_receiver_world_position_tx, texel, 0).xyz;
  return isfinite(shadow_P.x) && isfinite(shadow_P.y) && isfinite(shadow_P.z) &&
         dot(shadow_P, shadow_P) > 1.0e-10f;
}

bool transmission_receiver_hit_object_infos_load(int2 texel, ObjectInfos &object_infos)
{
  const uint resource_id = texelFetch(transmission_receiver_hit_identity_tx, texel, 0).w;
  if (resource_id == 0xFFFFFFFFu) {
    object_infos = ObjectInfos();
    return false;
  }
  object_infos = drw_infos[resource_id];
  return true;
}

ClosureUndetermined transmission_receiver_hit_base_closure_load(int2 texel, float3 N)
{
  float4 hit_base = texelFetch(transmission_receiver_hit_albedo_tx, texel, 0);
  float4 hit_material = texelFetch(transmission_receiver_hit_material_tx, texel, 0);
  const bool proxy_payload = transmission_receiver_hit_uses_proxy_payload(texel);
  ClosureType type = proxy_payload ? hardware_hit_closure_type_unpack(hit_material.z) :
                                     hardware_hit_closure_type_unpack(hit_base.a);
  if (proxy_payload && !hardware_hit_closure_is_base_family(type)) {
    type = CLOSURE_NONE_ID;
  }
  if (type == CLOSURE_BSSRDF_BURLEY_ID) {
    type = CLOSURE_BSDF_DIFFUSE_ID;
  }

  ClosureUndetermined cl = closure_new(type);
  cl.weight = 1.0f;
  cl.color = hit_base.rgb;
  cl.N = N;
  cl.data = float4(0.0f);

  switch (cl.type) {
    case CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID:
      cl.data.x = hit_material.x;
      break;
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
      cl.data.x = hit_material.x;
      cl.data.y = hit_material.y;
      break;
    case CLOSURE_BSDF_TRANSLUCENT_ID:
    case CLOSURE_BSDF_DIFFUSE_ID:
    case CLOSURE_BSSRDF_BURLEY_ID:
    case CLOSURE_NONE_ID:
      break;
  }
  return cl;
}

ClosureUndetermined transmission_receiver_hit_specular_closure_load(int2 texel, float3 N)
{
  float4 hit_base = texelFetch(transmission_receiver_hit_albedo_tx, texel, 0);
  float4 hit_material = texelFetch(transmission_receiver_hit_material_tx, texel, 0);
  float4 hit_specular = texelFetch(transmission_receiver_hit_position_tx, texel, 0);
  const bool proxy_payload = transmission_receiver_hit_uses_proxy_payload(texel);
  const ClosureType type = hardware_hit_closure_type_unpack(hit_material.z);

  ClosureUndetermined cl = closure_new(
      hardware_hit_closure_is_specular_family(type) ? type : CLOSURE_NONE_ID);
  cl.weight = 1.0f;
  cl.color = proxy_payload ? hit_base.rgb : hit_specular.rgb;
  cl.N = N;
  cl.data = float4(0.0f);

  switch (cl.type) {
    case CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID:
      cl.data.x = hit_material.x;
      break;
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
      cl.data.x = hit_material.x;
      cl.data.y = hit_material.y;
      break;
    case CLOSURE_BSDF_TRANSLUCENT_ID:
    case CLOSURE_BSDF_DIFFUSE_ID:
    case CLOSURE_BSSRDF_BURLEY_ID:
    case CLOSURE_NONE_ID:
      break;
  }

  return cl;
}

Thickness transmission_receiver_hit_thickness_load(int2 texel)
{
  if (transmission_receiver_hit_uses_proxy_payload(texel)) {
    return Thickness::zero();
  }
  return gbuffer::thickness_unpack(texelFetch(transmission_receiver_hit_position_tx, texel, 0).w);
}

void transmission_receiver_hit_closure_light_terms(int2 texel_fullres,
                                                   int2 texel,
                                                   float3 P_hit,
                                                   float3 N,
                                                   float3 V,
                                                   ClosureUndetermined cl,
                                                   Thickness thickness,
                                                   bool primary_is_diffuse_gi,
                                                   float3 &direct_radiance,
                                                   float3 &probe_radiance)
{
  direct_radiance = float3(0.0f);
  probe_radiance = float3(0.0f);
  if (!hardware_hit_closure_has_energy(cl)) {
    return;
  }

  const bool scene_final_specular_phase =
      (uniform_buf.raytrace.hardware_trace_phase == HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR);
  const bool is_transmission = closure_has_transmission(cl.type);
  const bool is_diffuse_family = (cl.type == CLOSURE_BSDF_DIFFUSE_ID) ||
                                 (cl.type == CLOSURE_BSSRDF_BURLEY_ID);
  const bool scene_final_reflected_diffuse = scene_final_specular_phase && is_diffuse_family;
  const bool direct_lit_refracted_textured_receiver =
      scene_final_specular_phase && !primary_is_diffuse_gi &&
      ((texelFetch(transmission_receiver_hit_identity_tx, texel, 0).z & 16u) != 0u);
  const bool suppress_scene_final_direct_hit_light = scene_final_specular_phase &&
                                                     !primary_is_diffuse_gi &&
                                                     !direct_lit_refracted_textured_receiver &&
                                                     (is_transmission ||
                                                      hardware_hit_closure_is_specular_family(cl.type));
  const Thickness cl_thickness = is_transmission ? thickness : Thickness::zero();
  uchar receiver_light_set = 0u;
  float normal_offset = 0.0f;
  float geometry_offset = 0.0f;
  ObjectInfos object_infos;
  if (transmission_receiver_hit_object_infos_load(texel, object_infos)) {
    receiver_light_set = receiver_light_set_get(object_infos);
    normal_offset = object_infos.shadow_terminator_normal_offset;
    geometry_offset = object_infos.shadow_terminator_geometry_offset;
  }
  shadow_dispatch_texel_fullres = texel;
  shadow_dispatch_use_hardware_rt = false;
  if ((use_hardware_rt_shadows || use_hardware_rt_environment_visibility) &&
      !direct_lit_refracted_textured_receiver)
  {
    shadow_dispatch_use_hardware_rt = transmission_receiver_hit_shadow_payload_valid(texel);
  }
  ClosureLightStack stack;
  ClosureUndetermined light_cl = hardware_hit_refracted_metal_direct_closure(
      cl, direct_lit_refracted_textured_receiver);
  stack.cl[0] = is_transmission ? closure_light_new(light_cl, V, cl_thickness) :
                                  closure_light_new(light_cl, V);
  LIGHT_FOREACH_BEGIN_DIRECTIONAL (light_cull_buf, l_idx) {
    light_eval_single(
        l_idx,
        true,
        is_transmission,
        stack,
        P_hit,
        N,
        V,
        cl_thickness,
        receiver_light_set,
        normal_offset,
        geometry_offset);
  }
  LIGHT_FOREACH_END

  if (hardware_hit_use_exact_local_lights()) {
    for (uint local_light_index = 0u; local_light_index < light_cull_buf.local_lights_len;
         local_light_index++)
    {
      hardware_hit_light_eval_exact_local(local_light_index,
                                          is_transmission,
                                          stack,
                                          P_hit,
                                          N,
                                          V,
                                          cl_thickness,
                                          receiver_light_set,
                                          normal_offset,
                                          geometry_offset);
    }
  }
  else {
    LIGHT_FOREACH_BEGIN_LOCAL_NO_CULL(light_cull_buf, l_idx) {
      light_eval_single(
          l_idx,
          false,
          is_transmission,
          stack,
          P_hit,
          N,
          V,
          cl_thickness,
          receiver_light_set,
          normal_offset,
          geometry_offset);
    }
    LIGHT_FOREACH_END
  }

  LightProbeSample samp = lightprobe_load(float2(texel_fullres), P_hit, N, V);
  float3 probe_light = lightprobe_eval(samp, cl, P_hit, V, cl_thickness);
  const bool use_dome_world_probe = hardware_fast_gi_enabled_for_diffuse() &&
                                    is_diffuse_family &&
                                    lightprobe_uses_world(samp) &&
                                    (primary_is_diffuse_gi || !use_hardware_environment);
  if (!use_hardware_environment && lightprobe_uses_world(samp) && !primary_is_diffuse_gi &&
      !scene_final_reflected_diffuse)
  {
    probe_light = float3(0.0f);
  }
  else if (hardware_hit_closure_uses_environment_visibility(cl, primary_is_diffuse_gi) &&
           lightprobe_uses_world(samp))
  {
    /* Keep receiver ownership consistent with the existing layered-reflection resolve. */
    probe_light *= 1.0f;
  }
  if (use_dome_world_probe) {
    probe_light = hardware_fast_gi_sample(P_hit);
  }
  else if (scene_final_reflected_diffuse && hardware_fast_gi_enabled_for_diffuse()) {
    probe_light += hardware_fast_gi_sample(P_hit);
  }

  float3 shading_color = max(cl.color, float3(0.0f));
  direct_radiance = suppress_scene_final_direct_hit_light ?
                        float3(0.0f) :
                        (stack.cl[0].light_shadowed * shading_color);
  probe_radiance = probe_light * shading_color;
}

float3 transmission_receiver_hit_radiance_resolve(int2 texel,
                                                  int2 texel_fullres,
                                                  bool primary_is_diffuse_gi)
{
  float3 P_hit, V;
  if (!transmission_receiver_hit_load(texel, P_hit, V)) {
    return float3(0.0f);
  }

  float3 N;
  if (!transmission_receiver_hit_normal_load(texel, N)) {
    return float3(0.0f);
  }

  Thickness thickness = transmission_receiver_hit_thickness_load(texel);
  ClosureUndetermined base_cl = transmission_receiver_hit_base_closure_load(texel, N);
  ClosureUndetermined specular_cl = transmission_receiver_hit_specular_closure_load(texel, N);

  float3 radiance = texelFetch(transmission_receiver_ray_radiance_tx, texel, 0).rgb;
  const bool direct_lit_refracted_textured_receiver =
      (uniform_buf.raytrace.hardware_trace_phase == HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR) &&
      !primary_is_diffuse_gi &&
      ((texelFetch(transmission_receiver_hit_identity_tx, texel, 0).z & 16u) != 0u);
  const bool replayed_reflective_receiver =
      direct_lit_refracted_textured_receiver &&
      hardware_hit_closure_is_specular_family(specular_cl.type);
  if (replayed_reflective_receiver) {
    float3 metal_color = max(specular_cl.color, float3(0.0f));
    if (!(dot(metal_color, metal_color) > 1.0e-10f)) {
      metal_color = max(base_cl.color, float3(0.0f));
    }
    if (dot(radiance, radiance) > 1.0e-10f) {
      radiance *= metal_color;
    }
  }

  float3 base_direct = float3(0.0f);
  float3 base_probe = float3(0.0f);
  float3 specular_direct = float3(0.0f);
  float3 specular_probe = float3(0.0f);
  transmission_receiver_hit_closure_light_terms(
      texel_fullres, texel, P_hit, N, V, base_cl, thickness, primary_is_diffuse_gi, base_direct, base_probe);
  transmission_receiver_hit_closure_light_terms(texel_fullres,
                                                texel,
                                                P_hit,
                                                N,
                                                V,
                                                specular_cl,
                                                thickness,
                                                primary_is_diffuse_gi,
                                                specular_direct,
                                                specular_probe);
  radiance += base_direct + specular_direct;
  if (primary_is_diffuse_gi ||
      (uniform_buf.raytrace.hardware_trace_phase == HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR))
  {
    radiance += base_probe + specular_probe;
  }
  float4 carried_throughput = transmission_receiver_hit_throughput_load(texel);
  if (carried_throughput.a > 0.5f) {
    radiance *= max(carried_throughput.rgb, float3(0.0f));
  }
  return radiance;
}

bool hardware_primary_surface_position_load(int2 texel_fullres, float3 &P)
{
  if (any(lessThan(texel_fullres, int2(0))) ||
      any(greaterThanEqual(texel_fullres, textureSize(depth_tx, 0))))
  {
    return false;
  }
  float depth = reverse_z::read(texelFetch(depth_tx, texel_fullres, 0).r);
  if (!(depth > 0.0f && depth < 1.0f)) {
    return false;
  }
  float2 uv = (float2(texel_fullres) + 0.5f) * uniform_buf.raytrace.full_resolution_inv;
  P = drw_point_screen_to_world(float3(uv, depth));
  return true;
}

float hardware_hit_caustic_focus(int2 texel, int2 texel_fullres, float3 P_hit, float3 V)
{
  float compression = 0.0f;
  float3 hit_px, hit_py, dummy_V;
  float3 primary_P;
  if (hardware_hit_load(texel + int2(1, 0), hit_px, dummy_V) &&
      hardware_hit_load(texel + int2(0, 1), hit_py, dummy_V))
  {
    float3 primary_px;
    float3 primary_py;
    const int step = max(uniform_buf.raytrace.resolution_scale, 1);
    if (hardware_primary_surface_position_load(texel_fullres, primary_P) &&
        hardware_primary_surface_position_load(texel_fullres + int2(step, 0), primary_px) &&
        hardware_primary_surface_position_load(texel_fullres + int2(0, step), primary_py))
    {
      float primary_area = length(primary_px - primary_P) * length(primary_py - primary_P);
      float receiver_area = length(hit_px - P_hit) * length(hit_py - P_hit);
      if ((primary_area > 1.0e-6f) && (receiver_area > 1.0e-6f)) {
        compression = saturate((primary_area / receiver_area - 1.0f) * 0.03f);
      }
    }
  }

  float bending = 0.0f;
  if (hardware_primary_surface_position_load(texel_fullres, primary_P)) {
    float3 primary_V = drw_world_incident_vector(primary_P);
    bending = saturate((1.0f - abs(dot(normalize(primary_V), normalize(V)))) * 4.0f);
  }

  float focus_seed = max(compression, bending * 0.5f);
  if (!(focus_seed > 1.0e-6f)) {
    focus_seed = 0.25f;
  }

  float sharpness = 1.0f + log2(float(max(uniform_buf.raytrace.hardware_caustics_samples, 1))) *
                               0.5f;
  float gain = 1.0f + log2(float(max(uniform_buf.raytrace.hardware_caustics_samples, 1))) * 0.2f;
  return pow(focus_seed, sharpness) * gain;
}

bool hardware_hit_caustics_eligible(ClosureUndetermined primary_closure, ClosureUndetermined base_cl)
{
  if (!hardware_hit_uses_caustics() || !hardware_hit_closure_has_energy(base_cl) ||
      !hardware_hit_closure_has_energy(primary_closure))
  {
    return false;
  }
  if ((base_cl.type != CLOSURE_BSDF_DIFFUSE_ID) && (base_cl.type != CLOSURE_BSSRDF_BURLEY_ID)) {
    return false;
  }
  /* Caustics-only ownership is limited to reflective/refractive transport. Sharp non-diffuse
   * closures like translucent fallbacks should not broaden this receiver buffer into a generic
   * indirect-light path. */
  if (!hardware_hit_closure_is_specular_family(primary_closure.type)) {
    return false;
  }

  return closure_apparent_roughness_get(primary_closure) < 0.2f;
}

bool hardware_hit_reflected_receiver_caustics_eligible(bool scene_final_specular_phase,
                                                       ClosureUndetermined primary_closure,
                                                       ClosureUndetermined base_cl)
{
  if (!hardware_hit_uses_caustics() || !scene_final_specular_phase ||
      !hardware_hit_closure_has_energy(primary_closure) || !hardware_hit_closure_has_energy(base_cl))
  {
    return false;
  }
  if (primary_closure.type != CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID) {
    return false;
  }
  if ((base_cl.type != CLOSURE_BSDF_DIFFUSE_ID) && (base_cl.type != CLOSURE_BSSRDF_BURLEY_ID)) {
    return false;
  }
  return closure_apparent_roughness_get(primary_closure) < 0.2f;
}

float3 hardware_hit_caustics_target(int2 texel,
                                    int2 texel_fullres,
                                    float3 P_hit,
                                    float3 V,
                                    float3 direct_light,
                                    float3 probe_light,
                                    bool probe_uses_world,
                                    float3 transport_seed)
{
  float focus = hardware_hit_caustic_focus(texel, texel_fullres, P_hit, V);
  const float sample_gain = 0.15f +
                            log2(float(max(uniform_buf.raytrace.hardware_caustics_samples, 1))) *
                                0.05f;
  /* Keep receiver caustics focused. Reuse direct lighting plus the world/HDRI-side probe term,
   * but do not let generic local diffuse probe energy broaden the caustics buffer. */
  float3 caustic_source = max(
      direct_light + (probe_uses_world ? (probe_light * 0.25f) : float3(0.0f)), float3(0.0f));
  if (dot(caustic_source, caustic_source) <= 1.0e-10f) {
    /* When the broad precombine baseline is fail-closed, black-world local-light scenes can lose
     * the old internal caustic seed entirely. Borrow a small amount of resolved transport only for
     * the receiver-caustics buffer so the focused late pass stays alive without restoring visible
     * precombine radiance. */
    caustic_source = max(transport_seed * 0.25f, float3(0.0f));
  }
  if (dot(caustic_source, caustic_source) <= 1.0e-10f) {
    return float3(0.0f);
  }
  const float3 caustic_seed = max(caustic_source, float3(0.35f));
  return caustic_seed * max(focus, 0.20f) * sample_gain;
}

float3 hardware_hit_caustics_resolve(int2 texel,
                                     int2 texel_fullres,
                                     float3 P_hit,
                                     float3 N_hit,
                                     float3 V,
                                     float3 base_direct,
                                     float3 base_probe,
                                     bool base_probe_uses_world,
                                     float3 transport_seed)
{
  float2 receiver_uv;
  int2 receiver_texel;
  if (!hardware_hit_visible_surface_lookup_texel_load(
          texel, P_hit, N_hit, true, receiver_uv, receiver_texel))
  {
    return float3(0.0f);
  }
  const float3 target = hardware_hit_caustics_target(texel,
                                                     texel_fullres,
                                                     P_hit,
                                                     V,
                                                     base_direct,
                                                     base_probe,
                                                     base_probe_uses_world,
                                                     transport_seed);
  if (dot(target, target) <= 1.0e-10f) {
    return float3(0.0f);
  }
  const float history_blend = clamp(
      4.0f / float(max(uniform_buf.raytrace.hardware_caustics_samples, 1)), 0.05f, 0.5f);
  const float3 resolved = mix(hardware_caustics_load(receiver_texel), target, history_blend);
  imageStoreFast(hardware_caustics_img, receiver_texel, float4(resolved, 1.0f));
  return resolved;
}

bool hardware_environment_miss_load(int2 texel,
                                    int2 &texel_fullres,
                                    Ray &ray,
                                    float3 &V,
                                    float &ray_pdf_inv,
                                    bool &preserve_existing_radiance)
{
  float4 ray_data_im;
  float ray_time;
  if (!hardware_ray_load(texel, texel_fullres, ray_data_im, ray_time) || ray_time != -3.0f)
  {
    return false;
  }

  ClosureUndetermined miss_closure = gbuffer::read_bin(texel_fullres, closure_index);
  preserve_existing_radiance = hardware_hit_preserves_screen_baseline(miss_closure);
  float depth = reverse_z::read(texelFetch(depth_tx, texel_fullres, 0).r);
  if (!(depth > 0.0f && depth < 1.0f)) {
    return false;
  }

  float2 uv = (float2(texel_fullres) + 0.5f) * uniform_buf.raytrace.full_resolution_inv;
  float3 P = drw_point_screen_to_world(float3(uv, depth));
  float3 stored_origin = imageLoadFast(hit_position_img, texel).xyz;
  if (dot(stored_origin, stored_origin) > 1.0e-10f) {
    P = stored_origin;
  }
  float3 miss_direction = ray_data_im.xyz;
  if (!hardware_hit_direction_load(texel, miss_direction)) {
    miss_direction = normalize(miss_direction);
  }
  ray.direction = miss_direction;
  V = -ray.direction;
  ray.origin = P;
  ray_pdf_inv = ray_data_im.w;

  if (closure_index == 0) {
    const gbuffer::Header gbuf_header = gbuffer::read_header(texel_fullres);
    const Thickness thickness = gbuffer::read_thickness(gbuf_header, texel_fullres);
    if (thickness.value() != 0.0f) {
      ClosureUndetermined cl = gbuffer::read_bin(texel_fullres, closure_index);
      ray = raytrace_thickness_ray_amend(ray, cl, V, thickness);
    }
  }

  return true;
}

float3 hardware_environment_miss_tint_load(int2 texel)
{
  ClosureType miss_type = hardware_hit_closure_type_unpack(imageLoadFast(hit_material_img, texel).z);
  if (!hardware_hit_closure_is_specular_family(miss_type)) {
    return float3(1.0f);
  }
  float3 miss_tint = max(imageLoadFast(hit_albedo_img, texel).rgb, float3(0.0f));
  return (dot(miss_tint, miss_tint) > 1.0e-10f) ? miss_tint : float3(1.0f);
}

float3 hardware_hit_normal_estimate(int2 texel, float3 P_hit, float3 V)
{
  float3 Px, Vx;
  float3 Py, Vy;
  bool valid_x = hardware_hit_load(texel + int2(1, 0), Px, Vx);
  bool valid_y = hardware_hit_load(texel + int2(0, 1), Py, Vy);

  float3 N = V;
  if (valid_x && valid_y) {
    float3 candidate = cross(Px - P_hit, Py - P_hit);
    float len_sq = dot(candidate, candidate);
    if (len_sq > 1.0e-10f) {
      N = candidate * inversesqrt(len_sq);
    }
  }

  if (dot(N, V) < 0.0f) {
    N = -N;
  }
  return N;
}

bool hardware_hit_payload_exists(int2 texel)
{
  float3 hit_normal = imageLoadFast(hit_normal_img, texel).rgb;
  return dot(hit_normal, hit_normal) > 1.0e-10f;
}

bool hardware_hit_has_sparse_replay_radiance(int2 texel)
{
  return imageLoadFast(hit_identity_img, texel).w != 0xFFFFFFFFu;
}

void main()
{
  constexpr uint tile_size = RAYTRACE_GROUP_SIZE;
  uint2 tile_coord = unpackUvec2x16(tiles_coord_buf[gl_WorkGroupID.x]);
  int2 texel = int2(gl_LocalInvocationID.xy + tile_coord * tile_size);

  if (any(greaterThanEqual(texel, imageSize(ray_data_img).xy))) {
    return;
  }
  Ray miss_ray;
  int2 miss_texel_fullres;
  float3 miss_V;
  float ray_pdf_inv;
  bool preserve_existing_radiance = true;
  if (hardware_environment_miss_load(
          texel, miss_texel_fullres, miss_ray, miss_V, ray_pdf_inv, preserve_existing_radiance))
  {
    ClosureUndetermined miss_closure = gbuffer::read_bin(miss_texel_fullres, closure_index);
    const bool miss_is_diffuse_gi = (miss_closure.type == CLOSURE_BSDF_DIFFUSE_ID) ||
                                    (miss_closure.type == CLOSURE_BSSRDF_BURLEY_ID);
    const bool scene_final_specular_phase =
        (uniform_buf.raytrace.hardware_trace_phase == HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR);
    const bool precombine_specular_caustics_phase = !scene_final_specular_phase &&
                                                    !miss_is_diffuse_gi &&
                                                    hardware_hit_uses_caustics();
    float3 radiance = (!precombine_specular_caustics_phase && preserve_existing_radiance) ?
                          imageLoadFast(ray_radiance_img, texel).rgb :
                          float3(0.0f);
    float3 Ng = miss_ray.direction;
    LightProbeSample samp = lightprobe_load(float2(miss_texel_fullres), miss_ray.origin, Ng, miss_V);
    const bool use_dome_world_probe = hardware_fast_gi_enabled_for_diffuse() &&
                                      miss_is_diffuse_gi && lightprobe_uses_world(samp);
    if (use_dome_world_probe) {
      radiance = colorspace_brightness_clamp_max(radiance, uniform_buf.clamp.surface_indirect);
      imageStoreFast(ray_time_img, texel, float4(10000.0f));
      imageStoreFast(ray_radiance_img, texel, float4(radiance, 0.0f));
      return;
    }
    if (!use_hardware_rt_environment_visibility && lightprobe_uses_world(samp) &&
        !miss_is_diffuse_gi)
    {
      radiance = colorspace_brightness_clamp_max(radiance, uniform_buf.clamp.surface_indirect);
      imageStoreFast(ray_time_img, texel, float4(10000.0f));
      imageStoreFast(ray_radiance_img, texel, float4(radiance, 0.0f));
      return;
    }
    ClosureType miss_proxy_type = hardware_hit_closure_type_unpack(imageLoadFast(hit_material_img, texel).z);
    float3 miss_N;
    float3 raster_radiance;
    if (scene_final_specular_phase &&
        hardware_hit_allows_scene_final_raster_reuse(miss_texel_fullres) &&
        hardware_hit_normal_load(texel, miss_N) &&
        hardware_hit_raster_radiance_load(
            texel,
            miss_ray.origin,
            miss_N,
            closure_has_transmission(miss_proxy_type),
            false,
            raster_radiance))
    {
      radiance = colorspace_brightness_clamp_max(raster_radiance, uniform_buf.clamp.surface_indirect);
      imageStoreFast(ray_time_img, texel, float4(10000.0f));
      imageStoreFast(ray_radiance_img, texel, float4(radiance, 0.0f));
      return;
    }

    samp.volume_irradiance = spherical_harmonics::clamp_energy(samp.volume_irradiance,
                                                               uniform_buf.clamp.surface_indirect);
    float3 world_direction = miss_ray.direction;
    float environment_visibility = 1.0f;
    if (use_hardware_rt_environment_visibility && lightprobe_uses_world(samp)) {
      HardwareEnvironmentVisibilityData env_visibility = hardware_environment_visibility_load(
          miss_texel_fullres, Ng);
      if (miss_is_diffuse_gi) {
        /* The env visibility dome is a diffuse world-transport approximation. Keep specular world
         * misses on their traced direction instead of blurring them through the primary-surface
         * dome visibility, which is what makes reflected env look smeared in mirror paths. */
        world_direction = hardware_environment_visibility_direction(
            env_visibility, miss_ray.direction, Ng);
        environment_visibility = env_visibility.visibility;
      }
    }
    float3 incoming_radiance = lightprobe_eval_direction(
        samp, miss_ray.origin, world_direction, ray_pdf_inv);
    incoming_radiance *= hardware_environment_miss_tint_load(texel);
    if (!precombine_specular_caustics_phase) {
      radiance += incoming_radiance * environment_visibility;
    }
    radiance = colorspace_brightness_clamp_max(radiance, uniform_buf.clamp.surface_indirect);
    imageStoreFast(ray_time_img, texel, float4(10000.0f));
    imageStoreFast(ray_radiance_img, texel, float4(radiance, 0.0f));
    return;
  }

  /* Only Hybrid-style ownership may preserve a screen-space first hit baseline. If screen-owned
   * radiance leaks into a Full RT specular pixel, fail closed to the Hardware result instead of
   * mixing both paths. */
  int2 primary_texel_fullres = texel * uniform_buf.raytrace.resolution_scale +
                               uniform_buf.raytrace.resolution_bias;
  if (uniform_buf.raytrace.use_hardware_ign_sampling && (uniform_buf.raytrace.resolution_scale > 1)) {
    primary_texel_fullres = raytrace_representative_fullres_texel(
        texel, uniform_buf.raytrace.resolution_scale, uniform_buf.raytrace.resolution_bias);
  }
  ClosureUndetermined primary_closure = gbuffer::read_bin(primary_texel_fullres, closure_index);
  const bool primary_is_diffuse_gi = (primary_closure.type == CLOSURE_BSDF_DIFFUSE_ID) ||
                                     (primary_closure.type == CLOSURE_BSSRDF_BURLEY_ID);
  const bool scene_final_specular_phase =
      (uniform_buf.raytrace.hardware_trace_phase == HWRT_TRACE_PHASE_SCENE_FINAL_SPECULAR);
  const bool precombine_specular_caustics_phase = !scene_final_specular_phase &&
                                                  !primary_is_diffuse_gi &&
                                                  hardware_hit_uses_caustics();
  const bool allow_scene_final_raster_reuse = hardware_hit_allows_scene_final_raster_reuse(
      primary_texel_fullres);
  const bool preserve_screen_baseline = hardware_hit_preserves_screen_baseline(primary_closure);
  float3 screen_radiance = imageLoadFast(ray_radiance_img, texel).rgb;
  float screen_ray_time = imageLoadFast(ray_time_img, texel).x;
  if (!precombine_specular_caustics_phase && preserve_screen_baseline &&
      !hardware_hit_payload_exists(texel) &&
      dot(screen_radiance, screen_radiance) > 1.0e-10f &&
      screen_ray_time > 0.0f &&
      screen_ray_time < 9999.0f)
  {
    return;
  }

  float3 P_hit, V;
  if (!hardware_hit_load(texel, P_hit, V)) {
    if (precombine_specular_caustics_phase) {
      imageStoreFast(ray_time_img, texel, float4(10000.0f));
      imageStoreFast(ray_radiance_img, texel, float4(0.0f));
    }
    return;
  }

  int2 texel_fullres = texel * uniform_buf.raytrace.resolution_scale +
                       uniform_buf.raytrace.resolution_bias;
  if (uniform_buf.raytrace.use_hardware_ign_sampling && (uniform_buf.raytrace.resolution_scale > 1)) {
    texel_fullres = raytrace_representative_fullres_texel(
        texel, uniform_buf.raytrace.resolution_scale, uniform_buf.raytrace.resolution_bias);
  }

  float3 N;
  if (!hardware_hit_normal_load(texel, N)) {
    N = hardware_hit_normal_estimate(texel, P_hit, V);
  }
  Thickness thickness = hardware_hit_thickness_load(texel);
  ClosureUndetermined base_cl = hardware_hit_base_closure_load(texel, N);
  ClosureUndetermined specular_cl = hardware_hit_specular_closure_load(texel, N);
  const bool hit_has_replayed_material = !hardware_hit_uses_proxy_payload(texel);
  const bool preserved_layered_scene_final = hardware_hit_is_preserved_layered_scene_final(texel);
  const bool preserved_transparent_scene_final = hardware_hit_is_preserved_transparent_scene_final(
      texel);
  const bool preserved_scene_final_composite = preserved_layered_scene_final ||
                                               preserved_transparent_scene_final;
  const bool direct_lit_refracted_textured_receiver =
      scene_final_specular_phase && !primary_is_diffuse_gi &&
      ((imageLoadFast(hit_identity_img, texel).z & 16u) != 0u);
  const bool primary_requests_resolved_surface =
      !primary_is_diffuse_gi &&
      (hardware_hit_specular_mode(primary_closure) == RAYTRACE_SPECULAR_MODE_FULL_RT);
  const bool hit_prefers_back_radiance = closure_has_transmission(specular_cl.type);
  const bool allow_diffuse_world_seed = use_hardware_environment || primary_is_diffuse_gi;
  const bool allow_sparse_replay_seed = allow_diffuse_world_seed ||
                                        preserved_transparent_scene_final ||
                                        (scene_final_specular_phase &&
                                         primary_requests_resolved_surface);
  const float4 transmission_layer = hardware_hit_transmission_layer_load(texel);
  const bool has_transmission_layer = preserved_scene_final_composite &&
                                      (transmission_layer.a > 0.5f);
  const float3 transmission_layer_color = max(transmission_layer.rgb, float3(0.0f));
  const float reflection_layer_opacity = preserved_layered_scene_final ?
                                             hardware_hit_reflection_layer_opacity(specular_cl) :
                                             0.0f;
  const float principled_reflection_layer_visibility = preserved_layered_scene_final ?
                                                           reflection_layer_opacity :
                                                           1.0f;
  float3 caustic_transport_seed = float3(0.0f);

  float3 radiance = (!precombine_specular_caustics_phase && preserve_screen_baseline) ?
                        imageLoadFast(ray_radiance_img, texel).rgb :
                        float3(0.0f);
  if (!precombine_specular_caustics_phase && !preserve_screen_baseline && allow_sparse_replay_seed &&
      hardware_hit_has_sparse_replay_radiance(texel))
  {
    radiance = imageLoadFast(ray_radiance_img, texel).rgb;
  }
  if (direct_lit_refracted_textured_receiver &&
      hardware_hit_closure_is_specular_family(specular_cl.type) &&
      dot(radiance, radiance) > 1.0e-10f)
  {
    float3 metal_color = max(specular_cl.color, float3(0.0f));
    if (!(dot(metal_color, metal_color) > 1.0e-10f)) {
      metal_color = max(base_cl.color, float3(0.0f));
    }
    radiance *= metal_color;
  }
  float3 raster_radiance;
  if (precombine_specular_caustics_phase &&
      primary_requests_resolved_surface &&
      hardware_hit_raster_radiance_load(
          texel,
          P_hit,
          N,
          hit_prefers_back_radiance,
          hardware_hit_uses_caustics(),
          raster_radiance))
  {
    /* Keep the visible result fail-closed to RT. Raster may only seed the later receiver-caustics
     * buffer when the preferred direct/world caustic seed is black. */
    caustic_transport_seed = raster_radiance;
  }
  if (!precombine_specular_caustics_phase && !scene_final_specular_phase &&
      allow_scene_final_raster_reuse && !primary_requests_resolved_surface &&
      !hardware_hit_uses_caustics() &&
      hardware_hit_allows_raster_reuse(
          texel, preserve_screen_baseline, hit_has_replayed_material, radiance, base_cl, specular_cl) &&
      hardware_hit_raster_radiance_load(
          texel, P_hit, N, hit_prefers_back_radiance, false, raster_radiance))
  {
    radiance += raster_radiance;
  }
  else {
    float3 base_direct = float3(0.0f);
    float3 base_probe = float3(0.0f);
    bool base_probe_uses_world = false;
    float3 specular_direct = float3(0.0f);
    float3 specular_probe = float3(0.0f);
    bool specular_probe_uses_world = false;
    float3 layered_receiver_radiance = float3(0.0f);
    float3 transmission_layer_radiance = float3(0.0f);
    hardware_hit_closure_light_terms(
        texel_fullres,
        texel,
        P_hit,
        N,
        V,
        base_cl,
        thickness,
        primary_is_diffuse_gi,
        base_direct,
        base_probe,
        base_probe_uses_world);
    hardware_hit_closure_light_terms(texel_fullres,
                                     texel,
                                     P_hit,
                                     N,
                                     V,
                                     specular_cl,
                                     thickness,
                                     primary_is_diffuse_gi,
                                     specular_direct,
                                     specular_probe,
                                     specular_probe_uses_world);
    if (preserved_scene_final_composite && layered_receiver_hit_exists(texel)) {
      layered_receiver_radiance = layered_receiver_hit_radiance_resolve(
          texel, texel_fullres, primary_is_diffuse_gi);
    }
    if (has_transmission_layer && transmission_receiver_hit_exists(texel)) {
      transmission_layer_radiance = transmission_receiver_hit_radiance_resolve(
          texel, texel_fullres, primary_is_diffuse_gi);
    }
    else if (has_transmission_layer) {
      hardware_hit_raster_back_radiance_load(
          texel, P_hit, N, true, hardware_hit_uses_caustics(), transmission_layer_radiance);
    }

    float3 visible_receiver_radiance = float3(0.0f);
    const bool scene_final_visible_diffuse_receiver =
        scene_final_specular_phase &&
        ((base_cl.type == CLOSURE_BSDF_DIFFUSE_ID) || (base_cl.type == CLOSURE_BSSRDF_BURLEY_ID)) &&
        hardware_hit_raster_radiance_load(
            texel, P_hit, N, hit_prefers_back_radiance, false, visible_receiver_radiance);

    if (!precombine_specular_caustics_phase) {
      if (scene_final_visible_diffuse_receiver) {
        /* Mirrors can reflect the resolved GI that is already visible for the receiver in the
         * main view. This is only used after the traced hit validates against the visible surface,
         * so off-camera receivers still stay owned by the traced/probe fallback paths. The
         * scene-final resolve applies the indirect scale later, while the visible combined buffer
         * is already scaled. Undo that scale here to avoid brightening reflected screen GI twice. */
        radiance += visible_receiver_radiance / max(uniform_buf.clamp.indirect_scale, 1.0e-4f);
      }
      else {
        radiance += base_direct;
        if (primary_is_diffuse_gi || scene_final_specular_phase) {
          radiance += base_probe;
        }
        radiance += specular_direct * principled_reflection_layer_visibility;
        if (primary_is_diffuse_gi || scene_final_specular_phase) {
          radiance += specular_probe * principled_reflection_layer_visibility;
        }
      }
      if (preserved_layered_scene_final &&
          dot(layered_receiver_radiance, layered_receiver_radiance) > 1.0e-10f)
      {
        radiance += layered_receiver_radiance * max(specular_cl.color, float3(0.0f)) *
                    principled_reflection_layer_visibility;
      }
      if (has_transmission_layer &&
          dot(transmission_layer_radiance, transmission_layer_radiance) > 1.0e-10f)
      {
        radiance += transmission_layer_radiance * transmission_layer_color;
      }
      if (preserved_transparent_scene_final &&
          dot(layered_receiver_radiance, layered_receiver_radiance) > 1.0e-10f)
      {
        radiance += layered_receiver_radiance;
      }
    }

    if (!scene_final_specular_phase && hardware_hit_caustics_eligible(primary_closure, base_cl))
    {
      hardware_hit_caustics_resolve(
          texel,
          texel_fullres,
          P_hit,
          N,
          V,
          base_direct,
          base_probe,
          base_probe_uses_world,
          caustic_transport_seed);
    }
    if (hardware_hit_reflected_receiver_caustics_eligible(
            scene_final_specular_phase, primary_closure, base_cl))
    {
      float3 scene_final_caustics = hardware_hit_caustics_target(texel,
                                                                 texel_fullres,
                                                                 P_hit,
                                                                 V,
                                                                 base_direct,
                                                                 base_probe,
                                                                 base_probe_uses_world,
                                                                 caustic_transport_seed);
      if (has_transmission_layer) {
        float3 transmission_P;
        float3 transmission_V;
        if (transmission_receiver_hit_load(texel, transmission_P, transmission_V)) {
          scene_final_caustics = hardware_hit_caustics_target(texel,
                                                              texel_fullres,
                                                              transmission_P,
                                                              transmission_V,
                                                              base_direct,
                                                              base_probe,
                                                              base_probe_uses_world,
                                                              caustic_transport_seed);
        }
        scene_final_caustics *= max(transmission_layer_color, float3(0.0f));
      }
      else {
        scene_final_caustics *= 2.5f;
      }
      radiance += scene_final_caustics;
    }
  }
  radiance = colorspace_brightness_clamp_max(radiance, uniform_buf.clamp.surface_indirect);
  imageStoreFast(ray_radiance_img, texel, float4(radiance, 0.0f));
}
