/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/**
 * Sample a bounded many-light direct subset from the queued tiles and resolve their RT visibility
 * from the already generated primary-surface Hardware RT shadow atlas.
 */

#include "infos/eevee_tracing_infos.hh"

COMPUTE_SHADER_CREATE_INFO(eevee_ray_hardware_direct_light_visibility)

#include "eevee_sampling_lib.glsl"
#include "gpu_shader_codegen_lib.glsl"
#include "gpu_shader_math_vector_lib.glsl"
#include "gpu_shader_utildefines_lib.glsl"

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

float sun_light_importance(uint sun_index)
{
  const uint light_index = uniform_buf.raytrace.hardware_direct_light.local_lights_len + sun_index;
  LightData light = light_buf[light_index];
  return light_color_importance(light) *
         uniform_buf.raytrace.hardware_direct_light.sun_light_importance_scale;
}

bool select_local_light(HardwareDirectLightWorkTile work_tile,
                        float selector,
                        uint &r_light_index,
                        float &r_importance)
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

  if (!(total_importance > 0.0f)) {
    r_light_index = 0xFFFFFFFFu;
    r_importance = 0.0f;
    return false;
  }

  const float target = selector * total_importance;
  float accum_importance = 0.0f;
  for (uint word_index = 0u; word_index < work_tile.candidate_word_count; word_index++) {
    uint word = light_tile_buf[work_tile.candidate_word_offset + word_index];
    int bit_index;
    while ((bit_index = findLSB(word)) != -1) {
      word &= ~(1u << uint(bit_index));
      const uint light_index = word_index * 32u + uint(bit_index);
      if (light_index >= uniform_buf.raytrace.hardware_direct_light.local_lights_len) {
        continue;
      }
      const float importance = local_light_importance(light_index);
      accum_importance += importance;
      if (accum_importance >= target) {
        r_light_index = light_index;
        r_importance = importance;
        return true;
      }
    }
  }

  r_light_index = 0xFFFFFFFFu;
  r_importance = 0.0f;
  return false;
}

bool select_sun_light(float selector, uint &r_sun_index, float &r_importance)
{
  const uint sun_lights_len = uniform_buf.raytrace.hardware_direct_light.sun_lights_len;
  if (sun_lights_len == 0u ||
      !uniform_buf.raytrace.hardware_direct_light.trace_sun_lights_separately)
  {
    r_sun_index = 0xFFFFFFFFu;
    r_importance = 0.0f;
    return false;
  }

  float total_importance = 0.0f;
  for (uint sun_index = 0u; sun_index < sun_lights_len; sun_index++) {
    total_importance += sun_light_importance(sun_index);
  }

  if (!(total_importance > 0.0f)) {
    r_sun_index = 0xFFFFFFFFu;
    r_importance = 0.0f;
    return false;
  }

  const float target = selector * total_importance;
  float accum_importance = 0.0f;
  for (uint sun_index = 0u; sun_index < sun_lights_len; sun_index++) {
    const float importance = sun_light_importance(sun_index);
    accum_importance += importance;
    if (accum_importance >= target) {
      r_sun_index = sun_index;
      r_importance = importance;
      return true;
    }
  }

  r_sun_index = 0xFFFFFFFFu;
  r_importance = 0.0f;
  return false;
}

void main()
{
  const uint queue_index = gl_GlobalInvocationID.x;
  const HardwareDirectLightWorkTile work_tile = hardware_direct_light_work_tiles_buf[queue_index];
  const uint2 tile_coord = unpackUvec2x16(work_tile.packed_tile_coord);
  const uint tile_size_px = max(uniform_buf.raytrace.hardware_direct_light.tile_size_px, 1u);

  const float2 noise = interleaved_gradient_noise(
      float2(tile_coord * tile_size_px) + 0.5f,
      float2(0.0f, 1.0f),
      float2(sampling_rng_1D_get(SAMPLING_SHADOW_U), sampling_rng_1D_get(SAMPLING_SHADOW_X)));
  const uint2 sample_offset = min(uint2(noise * float(tile_size_px)), uint2(tile_size_px - 1u));
  uint2 sample_texel = tile_coord * tile_size_px + sample_offset;
  const int2 visibility_extent = textureSize(hardware_rt_shadow_visibility_tx, 0).xy;
  sample_texel = clamp(sample_texel,
                       uint2(0u),
                       uint2(max(visibility_extent.x - 1, 0), max(visibility_extent.y - 1, 0)));

  HardwareDirectLightVisibilitySample visibility_record;
  visibility_record.packed_tile_coord = work_tile.packed_tile_coord;
  visibility_record.packed_sample_texel = packUvec2x16(sample_texel);
  visibility_record.local_light_index = 0xFFFFFFFFu;
  visibility_record.sun_light_index = 0xFFFFFFFFu;
  visibility_record.local_visibility = 0.0f;
  visibility_record.local_importance = 0.0f;
  visibility_record.sun_visibility = 0.0f;
  visibility_record.sun_importance = 0.0f;

  const float local_selector = interleaved_gradient_noise(
      float2(sample_texel) + 0.5f, 2.0f, sampling_rng_1D_get(SAMPLING_RAYTRACE_U));
  if (select_local_light(
          work_tile,
          local_selector,
          visibility_record.local_light_index,
          visibility_record.local_importance))
  {
    visibility_record.local_visibility =
        texelFetch(hardware_rt_shadow_visibility_tx,
                   int3(int2(sample_texel), int(visibility_record.local_light_index)),
                   0)
            .r;
  }

  const float sun_selector = interleaved_gradient_noise(
      float2(sample_texel) + 0.5f, 3.0f, sampling_rng_1D_get(SAMPLING_RAYTRACE_X));
  if (select_sun_light(
          sun_selector, visibility_record.sun_light_index, visibility_record.sun_importance))
  {
    const uint shadow_layer = uniform_buf.raytrace.hardware_direct_light.local_lights_len +
                              visibility_record.sun_light_index;
    visibility_record.sun_visibility =
        texelFetch(hardware_rt_shadow_visibility_tx, int3(int2(sample_texel), int(shadow_layer)), 0)
            .r;
  }

  hardware_direct_light_visibility_samples_buf[queue_index] = visibility_record;
}
