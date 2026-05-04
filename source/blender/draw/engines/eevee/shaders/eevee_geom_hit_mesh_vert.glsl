/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "infos/eevee_geom_infos.hh"
#include "infos/eevee_nodetree_infos.hh"
#include "infos/eevee_surf_hit_eval_infos.hh"

VERTEX_SHADER_CREATE_INFO(eevee_nodetree)
VERTEX_SHADER_CREATE_INFO(eevee_geom_hit_mesh)
VERTEX_SHADER_CREATE_INFO(eevee_surf_hit_eval)

#define DRW_CUSTOM_RESOURCE_ID
#define DRW_CUSTOM_VIEW_POSITION
#define DRW_CUSTOM_WORLD_INCIDENT_VECTOR
#define EEVEE_HIT_EVAL_GENERATED_ORCO

float3 g_hit_eval_object_P;
#ifdef GLSL_CPP_STUBS
float3 drw_object_orco(float3 lP);
#endif

float3 eevee_hit_eval_generated_orco()
{
  return drw_object_orco(g_hit_eval_object_P);
}

HardwareHitEvalRecord hit_eval_record_get()
{
  return hit_eval_list_buf[uint(gl_VertexID) / 3u];
}

uint drw_custom_resource_id_raw()
{
  return hit_eval_record_get().resource_id_raw;
}

float3 drw_custom_view_position()
{
  return hit_eval_record_get().view_origin;
}

float3 drw_custom_world_incident_vector(float3 P)
{
  return normalize(drw_custom_view_position() - P);
}

#include "draw_model_lib.glsl"
#include "eevee_attributes_mesh_lib.glsl"
#include "eevee_hit_attr_lib.glsl"
#include "eevee_nodetree_vert_lib.glsl"
#include "eevee_surf_lib.glsl"
#include "gpu_shader_index_load_lib.glsl"
#include "gpu_shader_utildefines_lib.glsl"

bool hit_eval_front_facing(HardwareHitEvalRecord record)
{
  return (record.flags & HIT_EVAL_FLAG_FRONT_FACING) != 0u;
}

float3 hit_eval_barycentric_expand(float2 barycentric_coords)
{
  return float3(max(0.0f, 1.0f - barycentric_coords.x - barycentric_coords.y),
                barycentric_coords.x,
                barycentric_coords.y);
}

float2 hit_eval_triangle_offset(uint corner)
{
  switch (corner) {
    case 0u:
      return float2(-0.5f, -0.5f);
    case 1u:
      return float2(1.5f, -0.5f);
    default:
      return float2(-0.5f, 1.5f);
  }
}

uint hit_eval_corner_index(bool front_facing, uint local_vertex)
{
  if (front_facing) {
    return local_vertex;
  }
  switch (local_vertex) {
    case 1u:
      return 2u;
    case 2u:
      return 1u;
    default:
      return 0u;
  }
}

void main()
{
  HardwareHitEvalRecord record = hit_eval_record_get();
  const uint primitive_vertex = uint(gl_VertexID) % 3u;
  const uint primitive_index = record.primitive_id * 3u;

  const uint i0 = gpu_index_load(primitive_index + 0u);
  const uint i1 = gpu_index_load(primitive_index + 1u);
  const uint i2 = gpu_index_load(primitive_index + 2u);

  const float3 barycentric = hit_eval_barycentric_expand(record.barycentric_coords);

  const float3 pos0 = hit_attr_fetch_float3(pos, gpu_attr_0, gpu_attr_0_meta, i0);
  const float3 pos1 = hit_attr_fetch_float3(pos, gpu_attr_0, gpu_attr_0_meta, i1);
  const float3 pos2 = hit_attr_fetch_float3(pos, gpu_attr_0, gpu_attr_0_meta, i2);

  const float3 nor0 = hit_attr_fetch_float3(nor, gpu_attr_1, gpu_attr_1_meta, i0);
  const float3 nor1 = hit_attr_fetch_float3(nor, gpu_attr_1, gpu_attr_1_meta, i1);
  const float3 nor2 = hit_attr_fetch_float3(nor, gpu_attr_1, gpu_attr_1_meta, i2);

  const float3 object_P = pos0 * barycentric.x + pos1 * barycentric.y + pos2 * barycentric.z;
  float3 object_N = nor0 * barycentric.x + nor1 * barycentric.y + nor2 * barycentric.z;

  if (!(dot(object_N, object_N) > 1.0e-10f)) {
    object_N = cross(pos1 - pos0, pos2 - pos0);
  }

  init_interface();

  interp.P = drw_point_object_to_world(object_P);
  interp.N = safe_normalize(drw_normal_object_to_world(object_N));

  init_globals();
  MeshVertex domain = MeshVertex();
  domain._pad = 0;
  domain.vertex_indices = int3(i0, i1, i2);
  domain.barycentric_weights = barycentric;
  g_hit_eval_object_P = object_P;
  attrib_load(domain);

  interp.P += nodetree_displacement();

  hit_flat.packed_texel = record.packed_texel;
  hit_flat.resource_id_raw = record.resource_id_raw;
  hit_flat.front_facing = hit_eval_front_facing(record) ? 1 : 0;
  hit_flat.view_origin = record.view_origin;

  const uint2 texel = unpackUvec2x16(record.packed_texel);
  const int2 extent = textureSize(ray_data_tx, 0).xy;
  const uint corner = hit_eval_corner_index(hit_eval_front_facing(record), primitive_vertex);
  const float2 ndc_center = (float2(texel) + 0.5f) * (2.0f / float2(extent)) - 1.0f;
  const float2 ndc_offset = hit_eval_triangle_offset(corner) * (2.0f / float2(extent));
  gl_Position = float4(ndc_center + ndc_offset, 0.0f, 1.0f);
}
