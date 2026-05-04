/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#define HWRT_FAST_GI_CASCADE_MAX 3
#define HWRT_FAST_GI_DIFFUSE_SCALE 0.636619772f
#define HWRT_FAST_GI_MIN_SAFE_CONFIDENCE 0.08f
#define HWRT_FAST_GI_MIN_SAFE_ACCUM_WEIGHT 0.06f
bool hardware_fast_gi_enabled_for_diffuse()
{
  /* The traced Fast GI field is a shared diffuse transport cache. Scene-final Hardware reflection /
   * refraction passes intentionally narrow `hardware_feature_mask` down to specular ownership, but
   * they still need to consume the already-built GI field on diffuse receivers seen through those
   * paths. Gate on the GI field itself rather than the per-phase feature mask. */
  return uniform_buf.raytrace.use_hardware_fast_gi &&
         uniform_buf.raytrace.use_hardware_fast_gi_field;
}

float4 hardware_fast_gi_cascade_visibility_support(int cascade_index, float3 P)
{
  float4 cascade_cfg = uniform_buf.raytrace.hardware_fast_gi_cascade_config[cascade_index];
  float voxel_size = cascade_cfg.w;
  int grid_res = max(uniform_buf.raytrace.hardware_fast_gi_grid_resolution, 1);
  float3 cascade_min = cascade_cfg.xyz - 0.5f * float(grid_res) * voxel_size;
  float3 uvw = (P - cascade_min) / (float(grid_res) * voxel_size);
  if (any(lessThan(uvw, float3(0.0f))) || any(greaterThanEqual(uvw, float3(1.0f)))) {
    return float4(0.0f);
  }

  float cascade_count = float(max(uniform_buf.raytrace.hardware_fast_gi_cascade_count, 1));
  float3 atlas_uvw = float3(uvw.xy, (uvw.z + float(cascade_index)) / cascade_count);
  return textureLod(hardware_fast_gi_visibility_tx, atlas_uvw, 0.0f);
}

float2 hardware_fast_gi_cascade_visibility(int cascade_index, float3 P)
{
  return hardware_fast_gi_cascade_visibility_support(cascade_index, P).xy;
}

float hardware_fast_gi_visibility_gate(float2 occupancy_thickness)
{
  float occupancy = saturate(occupancy_thickness.x);
  float thickness = saturate(occupancy_thickness.y);
  return 1.0f - saturate(max(occupancy * 0.85f, thickness));
}

float hardware_fast_gi_confidence_gate(float confidence)
{
  return saturate((confidence - HWRT_FAST_GI_MIN_SAFE_CONFIDENCE) /
                  (1.0f - HWRT_FAST_GI_MIN_SAFE_CONFIDENCE));
}

float hardware_fast_gi_leak_risk(float2 occupancy_thickness, float invalid)
{
  float occupancy = saturate(occupancy_thickness.x);
  float thickness = saturate(occupancy_thickness.y);
  float local_invalid = saturate((saturate(invalid) - 0.18f) / (1.0f - 0.18f));
  return saturate(occupancy * (1.0f - thickness) * local_invalid * 2.0f);
}

float3 hardware_fast_gi_cascade_sample(int cascade_index, float3 P, float &weight)
{
  float4 cascade_cfg = uniform_buf.raytrace.hardware_fast_gi_cascade_config[cascade_index];
  float voxel_size = cascade_cfg.w;
  int grid_res = max(uniform_buf.raytrace.hardware_fast_gi_grid_resolution, 1);
  float3 cascade_min = cascade_cfg.xyz - 0.5f * float(grid_res) * voxel_size;
  float3 uvw = (P - cascade_min) / (float(grid_res) * voxel_size);
  if (any(lessThan(uvw, float3(0.0f))) || any(greaterThanEqual(uvw, float3(1.0f)))) {
    weight = 0.0f;
    return float3(0.0f);
  }

  float cascade_count = float(max(uniform_buf.raytrace.hardware_fast_gi_cascade_count, 1));
  float3 atlas_uvw = float3(uvw.xy, (uvw.z + float(cascade_index)) / cascade_count);
  float4 sample_value = textureLod(hardware_fast_gi_tx, atlas_uvw, 0.0f);
  float raw_weight = sample_value.a;
  if (raw_weight <= 1.0e-4f) {
    weight = 0.0f;
    return float3(0.0f);
  }
  weight = hardware_fast_gi_confidence_gate(raw_weight);
  if (weight <= 1.0e-4f) {
    return float3(0.0f);
  }
  return sample_value.rgb / raw_weight;
}

float3 hardware_fast_gi_sample(float3 P)
{
  if (!hardware_fast_gi_enabled_for_diffuse()) {
    return float3(0.0f);
  }

  float3 accum = float3(0.0f);
  float accum_weight = 0.0f;
  int cascade_count = max(uniform_buf.raytrace.hardware_fast_gi_cascade_count, 1);
  float coarse_gate = 1.0f;

  float cascade_weight;
  float3 cascade_radiance = float3(0.0f);
  float4 cascade_support = float4(0.0f);
  float support_gate = 0.0f;
  if (cascade_count > 0) {
    cascade_radiance = hardware_fast_gi_cascade_sample(0, P, cascade_weight);
    cascade_support = hardware_fast_gi_cascade_visibility_support(0, P);
    support_gate = max(hardware_fast_gi_visibility_gate(cascade_support.xy), cascade_support.z);
    if (cascade_weight < 0.25f && support_gate <= 0.05f) {
      cascade_weight = 0.0f;
    }
    /* Leak-prone samples can still carry visible radiance even when their confidence stays above
     * the hard fail-closed threshold. Attenuate only the radiance, not the confidence weight, so
     * risky local samples contribute less light instead of merely reweighting the same leaked
     * value against neighboring cascades. */
    cascade_radiance *= 1.0f - 1.45f * hardware_fast_gi_leak_risk(cascade_support.xy,
                                                                   1.0f - cascade_weight);
    accum += cascade_radiance * cascade_weight * coarse_gate;
    accum_weight += cascade_weight * coarse_gate;
    coarse_gate *= hardware_fast_gi_visibility_gate(cascade_support.xy);
  }

  if (cascade_count > 1) {
    cascade_radiance = hardware_fast_gi_cascade_sample(1, P, cascade_weight);
    cascade_support = hardware_fast_gi_cascade_visibility_support(1, P);
    support_gate = max(hardware_fast_gi_visibility_gate(cascade_support.xy), cascade_support.z);
    if (cascade_weight < 0.25f && support_gate <= 0.05f) {
      cascade_weight = 0.0f;
    }
    cascade_radiance *= 1.0f - 1.45f * hardware_fast_gi_leak_risk(cascade_support.xy,
                                                                   1.0f - cascade_weight);
    accum += cascade_radiance * cascade_weight * coarse_gate;
    accum_weight += cascade_weight * coarse_gate;
    coarse_gate *= hardware_fast_gi_visibility_gate(cascade_support.xy);
  }

  if (cascade_count > 2) {
    cascade_radiance = hardware_fast_gi_cascade_sample(2, P, cascade_weight);
    cascade_support = hardware_fast_gi_cascade_visibility_support(2, P);
    support_gate = max(hardware_fast_gi_visibility_gate(cascade_support.xy), cascade_support.z);
    if (cascade_weight < 0.25f && support_gate <= 0.05f) {
      cascade_weight = 0.0f;
    }
    cascade_radiance *= 1.0f - 1.45f * hardware_fast_gi_leak_risk(cascade_support.xy,
                                                                   1.0f - cascade_weight);
    accum += cascade_radiance * cascade_weight * coarse_gate;
    accum_weight += cascade_weight * coarse_gate;
  }

  if (accum_weight <= HWRT_FAST_GI_MIN_SAFE_ACCUM_WEIGHT) {
    return float3(0.0f);
  }
  /* The traced field stores a sphere-average radiance estimate. Convert it to an upper-hemisphere
   * diffuse bounce proxy before feeding it into the layered GI stack; using only `1/pi` leaves
   * closed-room emissive transport materially under-lit because the consumer is otherwise averaging
   * over directions that cannot contribute on the back hemisphere of an opaque surface. */
  return (accum / accum_weight) * HWRT_FAST_GI_DIFFUSE_SCALE;
}
