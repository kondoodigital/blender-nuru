/* SPDX-FileCopyrightText: 2019 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

float checker_sample_fac(float3 p)
{
  /* Prevent precision issues on unit coordinates. */
  p = (p + 0.000001f) * 0.999999f;

  int xi = int(abs(floor(p.x)));
  int yi = int(abs(floor(p.y)));
  int zi = int(abs(floor(p.z)));

  bool check = ((mod(xi, 2) == mod(yi, 2)) == bool(mod(zi, 2)));
  return check ? 1.0f : 0.0f;
}

float checker_axis_filtered_signal(float p, float width)
{
  float wave = sin(p * 3.14159265359f);
  float softness = max(width * 3.14159265359f, 1.0e-5f);
  return wave * inversesqrt(wave * wave + softness * softness);
}

float checker_filtered_fac(float3 p, float width)
{
  width = max(width, 1.0e-5f);
  p = (p + 0.000001f) * 0.999999f;
  width *= 0.999999f;

  float sx = checker_axis_filtered_signal(p.x, width);
  float sy = checker_axis_filtered_signal(p.y, width);
  float sz = checker_axis_filtered_signal(p.z, width);

  return saturate(0.5f - 0.5f * sx * sy * sz);
}

[[node]]
void node_tex_checker(
    float3 co, float4 color1, float4 color2, float scale, float4 &color, float &fac)
{
  float3 p = co * scale;

  fac = checker_sample_fac(p);

  if (material_texture_filter_active()) {
    fac = checker_filtered_fac(p, material_texture_filter_checker_width_get());
  }

  color = mix(color2, color1, fac);
}
