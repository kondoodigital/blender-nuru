/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/**
 * Virtual shadow-mapping: propagate LOD0 update tags to lower LOD tiles.
 */

#include "infos/eevee_shadow_pipeline_infos.hh"

COMPUTE_SHADER_CREATE_INFO(eevee_shadow_tag_update_propagate)

#include "eevee_shadow_tilemap_lib.glsl"

void main()
{
  ShadowTileMapData tilemap = tilemaps_buf[gl_GlobalInvocationID.z];

  const uint2 texel = gl_GlobalInvocationID.xy;
  const int tile_index_lod0 = shadow_tile_offset(texel, tilemap.tiles_index, 0);
  uint tile_data = tiles_buf[tile_index_lod0];
  const bool do_update = (tile_data & uint(SHADOW_TAG_UPDATE)) != 0u;
  if (do_update) {
    /* Remove the transient flag. */
    tile_data &= ~uint(SHADOW_TAG_UPDATE);
    /* Assign persistent flag. */
    tile_data |= uint(SHADOW_DO_UPDATE);
    tiles_buf[tile_index_lod0] = tile_data;
  }

  if (do_update) {
    for (int lod = 1; lod <= SHADOW_TILEMAP_LOD; lod++) {
      const int tile_index = shadow_tile_offset(texel >> lod, tilemap.tiles_index, lod);
      atomicOr(tiles_buf[tile_index], uint(SHADOW_DO_UPDATE));
    }
  }
}
