/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/**
 * Compact the original tracing tiles down to only the ones that still need downstream
 * Hardware-only resolve work after the Metal pass. This keeps preserved pure-screen tiles
 * out of sparse hit-eval and hardware-lighting dispatch.
 */

#include "infos/eevee_tracing_infos.hh"

COMPUTE_SHADER_CREATE_INFO(eevee_ray_hardware_tile_compact)

#include "gpu_shader_codegen_lib.glsl"
#include "gpu_shader_math_vector_lib.glsl"
#include "gpu_shader_utildefines_lib.glsl"

shared uint tile_needs_hardware_resolve;

void main()
{
  if (gl_LocalInvocationIndex == 0u) {
    tile_needs_hardware_resolve = 0u;

    if (gl_WorkGroupID.x == 0u) {
      /* Match the tile compaction initialization pattern for indirect dispatch buffers. */
#if defined(GPU_INTEL) && defined(OS_WIN)
      atomicExchange(hardware_resolve_dispatch_buf.num_groups_y, 1u);
      atomicExchange(hardware_resolve_dispatch_buf.num_groups_z, 1u);
#else
      hardware_resolve_dispatch_buf.num_groups_y = 1u;
      hardware_resolve_dispatch_buf.num_groups_z = 1u;
#endif
    }
  }

  barrier();

  constexpr uint tile_size = RAYTRACE_GROUP_SIZE;
  const uint2 tile_coord = unpackUvec2x16(tiles_coord_buf[gl_WorkGroupID.x]);
  const int2 texel = int2(gl_LocalInvocationID.xy + tile_coord * tile_size);

  if (in_image_range(texel, ray_time_img)) {
    const float ray_time = imageLoadFast(ray_time_img, texel).x;
    const float3 hit_normal = imageLoadFast(hit_normal_img, texel).rgb;
    const bool has_payload = dot(hit_normal, hit_normal) > 1.0e-10f;
    if (has_payload || ray_time == -3.0f) {
      tile_needs_hardware_resolve = 1u;
    }
  }

  barrier();

  if (gl_LocalInvocationIndex == 0u && tile_needs_hardware_resolve > 0u) {
    const uint tile_index = atomicAdd(hardware_resolve_dispatch_buf.num_groups_x, 1u);
    hardware_resolve_tiles_buf[tile_index] = packUvec2x16(tile_coord);
  }
}
