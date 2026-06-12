/* SPDX-FileCopyrightText: 2019 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/* Nuru: Contains Nuru-specific changes relative to the Blender parent. */

float node_tex_checker_axis_soft(float value, float width)
{
  float p = abs(value);
  float wave = sin(p * M_PI);
  return wave * inversesqrt(wave * wave + max(width * width, 1.0e-6f));
}

float node_tex_checker_filtered_factor(float3 p, float width)
{
  float sx = node_tex_checker_axis_soft(p.x, width);
  float sy = node_tex_checker_axis_soft(p.y, width);
  float sz = node_tex_checker_axis_soft(p.z, width);
  return saturate(0.5f - 0.5f * sx * sy * sz);
}

[[node]]
void node_tex_checker(
    float3 co, float4 color1, float4 color2, float scale, float4 &color, float &fac)
{
  float3 p = co * scale;

  /* Prevent precision issues on unit coordinates. */
  p = (p + 0.000001f) * 0.999999f;

#ifdef EEVEE_MATERIAL_TEXTURE_FILTER
  if (material_texture_filter_active()) {
    float width = material_texture_filter_procedural_width_get();
    fac = node_tex_checker_filtered_factor(p, width);
    color = mix(color2, color1, fac);
    return;
  }
#endif

  int xi = int(abs(floor(p.x)));
  int yi = int(abs(floor(p.y)));
  int zi = int(abs(floor(p.z)));

  bool check = ((mod(xi, 2) == mod(yi, 2)) == bool(mod(zi, 2)));

  color = check ? color1 : color2;
  fac = check ? 1.0f : 0.0f;
}
