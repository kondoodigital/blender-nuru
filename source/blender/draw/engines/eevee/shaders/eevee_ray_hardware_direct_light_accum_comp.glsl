/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/**
 * Convert sampled many-light RT visibility records into a sparse combined direct-light buffer.
 *
 * This first slice intentionally targets opaque/specular direct lighting only. Transmission-style
 * closures remain on the existing deferred path until the material-policy todos land.
 */

#include "infos/eevee_tracing_infos.hh"

#define SHADOW_DISPATCH_USE_GLOBAL_TEXEL
#define LIGHT_ITER_FORCE_NO_CULLING
#define LIGHT_CLOSURE_EVAL_COUNT 1
COMPUTE_SHADER_CREATE_INFO(eevee_ray_hardware_direct_light_accum)

#include "eevee_closure_lib.glsl"
#include "eevee_colorspace_lib.bsl.hh"
#include "eevee_gbuffer_read_lib.glsl"
int2 shadow_dispatch_texel_fullres = int2(0);
#include "eevee_light_eval_lib.glsl"
#include "eevee_reverse_z_lib.bsl.hh"
#include "gpu_shader_codegen_lib.glsl"

float light_color_importance(LightData light)
{
  return max(dot(abs(light.color), float3(0.2126f, 0.7152f, 0.0722f)), 1.0e-4f);
}

float local_light_importance(uint light_index)
{
  LightData light = light_buf[light_index];
  float importance = light_color_importance(light) *
                     uniform_buf.raytrace.hardware_direct_light.local_light_importance_scale;
  if (is_area_light(light.type)) {
    importance *= uniform_buf.raytrace.hardware_direct_light.area_light_importance_scale;
  }
  return importance;
}

float local_total_importance(HardwareDirectLightWorkTile work_tile)
{
  float total_importance = 0.0f;
  for (uint word_index = 0u; word_index < work_tile.candidate_word_count; word_index++) {
    uint word = light_tile_buf[work_tile.candidate_word_offset + word_index];
    int bit_index;
    while ((bit_index = findLSB(word)) != -1) {
      word &= ~(1u << uint(bit_index));
      const uint light_index = word_index * 32u + uint(bit_index);
      if (light_index < uniform_buf.raytrace.hardware_direct_light.local_lights_len) {
        total_importance += local_light_importance(light_index);
      }
    }
  }
  return total_importance;
}

float sun_total_importance()
{
  float total_importance = 0.0f;
  for (uint sun_index = 0u; sun_index < uniform_buf.raytrace.hardware_direct_light.sun_lights_len;
       sun_index++)
  {
    const uint light_index = uniform_buf.raytrace.hardware_direct_light.local_lights_len + sun_index;
    total_importance += light_color_importance(light_buf[light_index]) *
                        uniform_buf.raytrace.hardware_direct_light.sun_light_importance_scale;
  }
  return total_importance;
}

bool hardware_direct_light_supported_closure(ClosureUndetermined cl)
{
  return (cl.type != CLOSURE_NONE_ID) && (cl.type != CLOSURE_BSDF_TRANSLUCENT_ID) &&
         (cl.type != CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID);
}

LightData hardware_direct_light_resolved(uint light_index)
{
  LightData light = light_buf[light_index];
  light.color = abs(light.color);
  const uint local_lights_len = uniform_buf.raytrace.hardware_direct_light.local_lights_len;
  const uint sun_lights_len = uniform_buf.raytrace.hardware_direct_light.sun_lights_len;
  if (sun_lights_len < 2u || light_index < local_lights_len || light_index >= local_lights_len + 2u) {
    return light;
  }

  const uint first_index = local_lights_len;
  const uint second_index = local_lights_len + 1u;
  LightData first = light_buf[first_index];
  LightData second = light_buf[second_index];
  first.color = abs(first.color);
  second.color = abs(second.color);

  const bool split_pair = dot(first.color - second.color, first.color - second.color) <= 1.0e-6f &&
                          dot(first.sun().direction, second.sun().direction) >= 0.9999f &&
                          ((first.power[LIGHT_DIFFUSE] == 0.0f && second.power[LIGHT_SPECULAR] == 0.0f) ||
                           (first.power[LIGHT_SPECULAR] == 0.0f && second.power[LIGHT_DIFFUSE] == 0.0f));
  if (!split_pair) {
    return light;
  }

  const uint peer_index = (light_index == first_index) ? second_index : first_index;
  const LightData peer = light_buf[peer_index];
  light.power[LIGHT_DIFFUSE] = max(light.power[LIGHT_DIFFUSE], peer.power[LIGHT_DIFFUSE]);
  light.power[LIGHT_SPECULAR] = max(light.power[LIGHT_SPECULAR], peer.power[LIGHT_SPECULAR]);
  light.power[LIGHT_TRANSMISSION] = max(light.power[LIGHT_TRANSMISSION], peer.power[LIGHT_TRANSMISSION]);
  light.power[LIGHT_VOLUME] = max(light.power[LIGHT_VOLUME], peer.power[LIGHT_VOLUME]);
  return light;
}

float3 hardware_direct_light_evaluate_single(uint light_index,
                                             bool is_directional,
                                             float shadow,
                                             ClosureUndetermined cl,
                                             float3 P,
                                             float3 Ng,
                                             float3 V,
                                             uchar receiver_light_set)
{
  LightData light = hardware_direct_light_resolved(light_index);
  if (!light_linking_affects_receiver(light.light_set_membership, receiver_light_set)) {
    return float3(0.0f);
  }
  if (!hardware_direct_light_supported_closure(cl)) {
    return float3(0.0f);
  }

  ClosureLight cl_light = closure_light_new(cl, V);
  LightVector lv = light_vector_get(light, is_directional, P);
  float attenuation = light_attenuation_surface(light, is_directional, lv);
  attenuation *= light_attenuation_facing(light, lv.L, lv.dist, cl_light.N, false);
  if (attenuation < LIGHT_ATTENUATION_THRESHOLD) {
    return float3(0.0f);
  }

  light_eval_single_closure(light, lv, cl_light, V, attenuation, shadow);
  return cl_light.light_shadowed * cl.color;
}

void main()
{
  const uint queue_index = gl_GlobalInvocationID.x;
  const HardwareDirectLightWorkTile work_tile = hardware_direct_light_work_tiles_buf[queue_index];
  const HardwareDirectLightVisibilitySample visibility_record =
      hardware_direct_light_visibility_samples_buf[queue_index];
  const int2 texel = int2(unpackUvec2x16(visibility_record.packed_sample_texel));

  if (any(lessThan(texel, int2(0))) || any(greaterThanEqual(texel, textureSize(depth_tx, 0)))) {
    return;
  }

  float depth = reverse_z::read(texelFetch(depth_tx, texel, 0).r);
  if (!(depth > 0.0f && depth < 1.0f)) {
    return;
  }

  float2 uv = (float2(texel) + 0.5f) * uniform_buf.raytrace.full_resolution_inv;
  const gbuffer::Layers gbuf = gbuffer::read_layers(texel);
  const float3 P = drw_point_screen_to_world(float3(uv, depth));
  const float3 Ng = gbuf.header.geometry_normal(gbuf.surface_N());
  const float3 V = drw_world_incident_vector(P);
  shadow_dispatch_texel_fullres = texel;

  uchar receiver_light_set = 0u;
  const uint object_id = gbuffer::read_object_id(texel);
  ObjectInfos object_infos = drw_infos[object_id];
  receiver_light_set = receiver_light_set_get(object_infos);

  float3 radiance = float3(0.0f);

  if (visibility_record.local_light_index != 0xFFFFFFFFu &&
      visibility_record.local_importance > 0.0f)
  {
    const float sample_weight = local_total_importance(work_tile) /
                                max(visibility_record.local_importance, 1.0e-6f);
    for (uchar closure_index = 0u; closure_index < gbuf.header.closure_len(); closure_index++) {
      radiance += hardware_direct_light_evaluate_single(visibility_record.local_light_index,
                                                        false,
                                                        visibility_record.local_visibility,
                                                        gbuf.layer[closure_index],
                                                        P,
                                                        Ng,
                                                        V,
                                                        receiver_light_set) *
                  sample_weight;
    }
  }

  if (visibility_record.sun_light_index != 0xFFFFFFFFu &&
      visibility_record.sun_importance > 0.0f)
  {
    const uint sun_light_index = uniform_buf.raytrace.hardware_direct_light.local_lights_len +
                                 visibility_record.sun_light_index;
    const float sample_weight = sun_total_importance() /
                                max(visibility_record.sun_importance, 1.0e-6f);
    for (uchar closure_index = 0u; closure_index < gbuf.header.closure_len(); closure_index++) {
      radiance += hardware_direct_light_evaluate_single(sun_light_index,
                                                        true,
                                                        visibility_record.sun_visibility,
                                                        gbuf.layer[closure_index],
                                                        P,
                                                        Ng,
                                                        V,
                                                        receiver_light_set) *
                  sample_weight;
    }
  }

  radiance = colorspace_brightness_clamp_max(radiance, uniform_buf.clamp.surface_direct);
  imageStore(out_direct_light_accum_img, texel, float4(radiance, 1.0f));
}
