/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "infos/eevee_tracing_infos.hh"

COMPUTE_SHADER_CREATE_INFO(eevee_ray_hit_eval_count)

#include "eevee_sampling_lib.glsl"
#include "eevee_reverse_z_lib.bsl.hh"
#include "gpu_shader_codegen_lib.glsl"
#include "gpu_shader_utildefines_lib.glsl"

bool hit_eval_primary_depth_valid(int2 texel)
{
  int2 texel_fullres = texel * uniform_buf.raytrace.resolution_scale +
                       uniform_buf.raytrace.resolution_bias;
  if (uniform_buf.raytrace.use_hardware_ign_sampling && (uniform_buf.raytrace.resolution_scale > 1)) {
    texel_fullres = raytrace_representative_fullres_texel(
        texel, uniform_buf.raytrace.resolution_scale, uniform_buf.raytrace.resolution_bias);
  }
  if (any(greaterThanEqual(texel_fullres, textureSize(depth_tx, 0))) ||
      any(lessThan(texel_fullres, int2(0))))
  {
    return false;
  }
  const float depth = reverse_z::read(texelFetch(depth_tx, texel_fullres, 0).r);
  return depth > 0.0f && depth < 1.0f;
}

void main()
{
  constexpr uint tile_size = RAYTRACE_GROUP_SIZE;
  const uint2 tile_coord = unpackUvec2x16(tiles_coord_buf[gl_WorkGroupID.x]);
  const int2 texel = int2(gl_LocalInvocationID.xy + tile_coord * tile_size);

  if (any(greaterThanEqual(texel, imageSize(ray_time_img).xy)) ||
      any(lessThan(texel, int2(0))))
  {
    return;
  }

  const float ray_time = imageLoadFast(ray_time_img, texel).x;
  /* Sparse hit-eval only accepts real Hardware hits with a valid scene identity. */
  if (!(ray_time > 0.0f) || !hit_eval_primary_depth_valid(texel)) {
    return;
  }

  const uint user_id = imageLoadFast(hit_identity_img, texel).x;
  if (user_id >= uint(scene_entry_count)) {
    return;
  }

  atomicAdd(hit_eval_count_buf[user_id], 1u);
}
