/* SPDX-FileCopyrightText: 2019-2022 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/* Nuru: Contains Nuru-specific changes relative to the Blender parent. */

#include "gpu_shader_math_vector_safe_lib.glsl"
#include "gpu_shader_utildefines_lib.glsl"

[[node]]
void node_bsdf_translucent(float4 color, float3 N, float weight, Closure &result)
{
  color = max(color, float4(0.0f));
  float roughness = 1.0f;
  float ior = 50.0f;
  N = safe_normalize(N);

  ClosureRefraction refraction_data;
  refraction_data.weight = weight;
  refraction_data.color = color.rgb;
  refraction_data.N = N;
  refraction_data.roughness = saturate(roughness);
  refraction_data.ior = max(ior, 1e-5f);

  result = closure_eval(refraction_data);
}
