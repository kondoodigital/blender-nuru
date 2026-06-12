/* SPDX-FileCopyrightText: 2019-2022 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "gpu_shader_math_vector_safe_lib.glsl"
#include "gpu_shader_utildefines_lib.glsl"

[[node]]
void node_bsdf_thin_glass(float4 color,
                          float film_alpha,
                          float roughness,
                          float ior,
                          float3 N,
                          float weight,
                          const float do_multiscatter,
                          const float use_fresnel,
                          Closure &result)
{
  color = max(color, float4(0.0f));
  film_alpha = saturate(film_alpha);
  roughness = saturate(roughness);
  ior = max(ior, 1.0e-5f);
  N = safe_normalize(N);

  float3 V = coordinate_incoming(g_data.P);
  float NV = saturate(abs(dot(N, V)));

  float2 bsdf = bsdf_lut(NV, roughness, ior, do_multiscatter != 0.0f);
  float f0 = square((ior - 1.0f) / (ior + 1.0f));
  float one_minus_nv = 1.0f - NV;
  float fresnel = f0 + (1.0f - f0) * square(square(one_minus_nv)) * one_minus_nv;

  ClosureReflection reflection_data;
  reflection_data.weight = weight;
  reflection_data.color = color.rgb;
  reflection_data.N = N;
  reflection_data.roughness = roughness;

  g_thin_glass_color = color.rgb;
  g_thin_glass_film_alpha = film_alpha;
  g_thin_glass_reflection_weight = (use_fresnel != 0.0f) ? fresnel : 1.0f;

  ClosureTransparency transparency_data;
#if defined(MAT_SHADOW)
  transparency_data.weight = weight;
  transparency_data.transmittance = float3(1.0f);
#elif defined(MAT_HIT_EVAL)
  transparency_data.weight = weight;
  transparency_data.transmittance = float3(1.0f);
#elif defined(MAT_FORWARD)
  transparency_data.weight = bsdf.y * weight;
  transparency_data.transmittance = color.rgb;
#else
  transparency_data.weight = 0.0f;
  transparency_data.transmittance = float3(0.0f);
#endif
  transparency_data.holdout = 0.0f;

  result = closure_eval(reflection_data, transparency_data);
}
