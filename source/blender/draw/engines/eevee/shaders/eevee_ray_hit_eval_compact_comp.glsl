/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "infos/eevee_tracing_infos.hh"

COMPUTE_SHADER_CREATE_INFO(eevee_ray_hit_eval_compact)

#include "eevee_sampling_lib.glsl"
#include "draw_view_lib.glsl"
#include "eevee_reverse_z_lib.bsl.hh"
#include "gpu_shader_codegen_lib.glsl"
#include "gpu_shader_utildefines_lib.glsl"

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

bool hit_eval_primary_depth_load(int2 texel, float3 &primary_P)
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
    primary_P = float3(0.0f);
    return false;
  }

  const float depth = reverse_z::read(texelFetch(depth_tx, texel_fullres, 0).r);
  if (!(depth > 0.0f && depth < 1.0f)) {
    primary_P = float3(0.0f);
    return false;
  }

  const float2 uv = (float2(texel_fullres) + 0.5f) * uniform_buf.raytrace.full_resolution_inv;
  primary_P = drw_point_screen_to_world(float3(uv, depth));
  return true;
}

float3 hit_eval_view_origin_load(int2 texel, float3 primary_P)
{
  const float3 hit_P = texelFetch(hit_world_position_tx, texel, 0).xyz;
  const float final_segment_distance = imageLoadFast(hit_barycentric_img, texel).z;
  if (!(final_segment_distance > 0.0f) || !(dot(hit_P, hit_P) > 1.0e-10f)) {
    return primary_P;
  }

  const float2 packed_direction = float2(imageLoadFast(hit_material_img, texel).w,
                                         imageLoadFast(hit_normal_img, texel).w);
  if (all(equal(packed_direction, float2(0.0f)))) {
    return primary_P;
  }

  const float3 ray_direction = hardware_direction_unpack(packed_direction);
  if (!(isfinite(ray_direction.x) && isfinite(ray_direction.y) && isfinite(ray_direction.z)) ||
      !(dot(ray_direction, ray_direction) > 1.0e-10f))
  {
    return primary_P;
  }

  /* Replay the material from the actual incoming segment origin. Using the primary surface point
   * for multi-bounce hits warps view-dependent reflection/Fresnel evaluation once continuation
   * leaves the first visible surface. */
  return hit_P - ray_direction * final_segment_distance;
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
  if (!(ray_time > 0.0f)) {
    return;
  }

  const uint4 hit_identity = imageLoadFast(hit_identity_img, texel);
  const uint user_id = hit_identity.x;
  if (user_id >= uint(scene_entry_count)) {
    return;
  }

  float3 primary_P;
  if (!hit_eval_primary_depth_load(texel, primary_P)) {
    return;
  }

  const uint record_index = hit_eval_offset_buf[user_id] +
                            atomicAdd(hit_eval_cursor_buf[user_id], 1u);
  const bool front_facing = (hit_identity.z & 1u) != 0u;

  HardwareHitEvalRecord record;
  record.packed_texel = packUvec2x16(uint2(texel));
  record.resource_id_raw = hit_eval_resource_id_buf[user_id];
  record.primitive_id = hit_identity.y;
  /* `hit_normal_img` stores the ray-oriented final shading normal, so deriving facing from it
   * collapses all hits to "front-facing". Trust the explicit side bit exported by Metal instead. */
  record.flags = front_facing ? HIT_EVAL_FLAG_FRONT_FACING : 0u;
  record.barycentric_coords = imageLoadFast(hit_barycentric_img, texel).xy;
  record._pad0 = float2(0.0f);
  record.view_origin = hit_eval_view_origin_load(texel, primary_P);
  record._pad1 = 0.0f;

  imageStoreFast(hit_identity_img, texel, uint4(hit_identity.xyz, record.resource_id_raw));
  hit_eval_list_buf[record_index] = record;
}
