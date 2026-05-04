/* SPDX-FileCopyrightText: 2019-2022 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "gpu_shader_math_vector_safe_lib.glsl"

[[node]]
void node_bsdf_transparent(float4 color, float weight, Closure &result)
{
  color = max(color, float4(0.0f));

  ClosureRefraction refraction_data;
  refraction_data.weight = weight;
  refraction_data.color = color.rgb;
  refraction_data.N = safe_normalize(g_data.N);
  refraction_data.roughness = 0.0f;
  refraction_data.ior = 1.0f;
  result = closure_eval(refraction_data);
}
