/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#ifdef GPU_SHADER
#  pragma once
#  include "BLI_utildefines_variadic.h"

#  include "gpu_shader_compat.hh"

#  include "draw_object_infos_infos.hh"
#  include "draw_view_infos.hh"
#  include "gpu_index_load_infos.hh"

#  include "eevee_common_infos.hh"
#  include "eevee_fullscreen_infos.hh"
#  include "eevee_light_infos.hh"
#  include "eevee_sampling_infos.hh"
#  include "eevee_shadow_infos.hh"
#  include "eevee_shadow_shared.hh"
#  include "eevee_uniform_infos.hh"
#  include "eevee_volume_infos.hh"
#endif

#ifdef GLSL_CPP_STUBS
#  define CURVES_SHADER
#  define DRW_HAIR_INFO

#  define POINTCLOUD_SHADER
#  define DRW_POINTCLOUD_INFO

#  define SHADOW_UPDATE_ATOMIC_RASTER
#  define MAT_TRANSPARENT
#endif

#include "eevee_defines.hh"
#include "gpu_shader_create_info.hh"

/* Common interface */
GPU_SHADER_NAMED_INTERFACE_INFO(eevee_surf_iface, interp)
/* World Position. */
SMOOTH(float3, P)
/* World Normal. */
SMOOTH(float3, N)
GPU_SHADER_NAMED_INTERFACE_END(interp)

GPU_SHADER_NAMED_INTERFACE_INFO(eevee_surf_hit_eval_flat_iface, hit_flat)
FLAT(uint, packed_texel)
FLAT(uint, resource_id_raw)
FLAT(int, front_facing)
FLAT(float3, view_origin)
GPU_SHADER_NAMED_INTERFACE_END(hit_flat)

GPU_SHADER_CREATE_INFO(eevee_geom_mesh)
TYPEDEF_SOURCE("eevee_defines.hh")
DEFINE("MAT_GEOM_MESH")
VERTEX_IN(0, float3, pos)
VERTEX_IN(1, float3, nor)
VERTEX_SOURCE("eevee_geom_mesh_vert.glsl")
VERTEX_OUT(eevee_surf_iface)
ADDITIONAL_INFO(draw_modelmat)
ADDITIONAL_INFO(draw_object_infos)
ADDITIONAL_INFO(draw_resource_id_varying)
ADDITIONAL_INFO(draw_view)
GPU_SHADER_CREATE_END()

GPU_SHADER_CREATE_INFO(eevee_geom_hit_fullscreen)
TYPEDEF_SOURCE("eevee_defines.hh")
DEFINE("MAT_GEOM_MESH")
BUILTINS(BuiltinBits::VERTEX_ID)
VERTEX_SOURCE("eevee_geom_hit_fullscreen_vert.glsl")
VERTEX_OUT(eevee_surf_iface)
VERTEX_OUT(eevee_fullscreen_iface)
ADDITIONAL_INFO(draw_modelmat)
ADDITIONAL_INFO(draw_object_infos)
ADDITIONAL_INFO(draw_view)
ADDITIONAL_INFO(gpu_index_buffer_load)
STORAGE_BUF_FREQ(0, read, uint, pos[], GEOMETRY)
STORAGE_BUF_FREQ(1, read, uint, nor[], GEOMETRY)
PUSH_CONSTANT(int2, gpu_attr_0)
PUSH_CONSTANT(int, gpu_attr_0_meta)
PUSH_CONSTANT(int2, gpu_attr_1)
PUSH_CONSTANT(int, gpu_attr_1_meta)
GPU_SHADER_CREATE_END()

GPU_SHADER_CREATE_INFO(eevee_geom_hit_mesh)
TYPEDEF_SOURCE("eevee_defines.hh")
DEFINE("MAT_GEOM_MESH")
DEFINE("DRW_CUSTOM_RESOURCE_ID")
DEFINE("DRW_CUSTOM_VIEW_POSITION")
DEFINE("DRW_CUSTOM_WORLD_INCIDENT_VECTOR")
DEFINE("EEVEE_CUSTOM_FRONT_FACING")
DEFINE("EEVEE_HIT_EVAL_GENERATED_ORCO")
BUILTINS(BuiltinBits::VERTEX_ID)
VERTEX_SOURCE("eevee_geom_hit_mesh_vert.glsl")
VERTEX_OUT(eevee_surf_iface)
VERTEX_OUT(eevee_surf_hit_eval_flat_iface)
ADDITIONAL_INFO(draw_modelmat_common)
ADDITIONAL_INFO(draw_object_infos)
ADDITIONAL_INFO(draw_view)
ADDITIONAL_INFO(gpu_index_buffer_load)
STORAGE_BUF_FREQ(0, read, uint, pos[], GEOMETRY)
STORAGE_BUF_FREQ(1, read, uint, nor[], GEOMETRY)
PUSH_CONSTANT(int2, gpu_attr_0)
PUSH_CONSTANT(int, gpu_attr_0_meta)
PUSH_CONSTANT(int2, gpu_attr_1)
PUSH_CONSTANT(int, gpu_attr_1_meta)
GPU_SHADER_CREATE_END()

GPU_SHADER_NAMED_INTERFACE_INFO(eevee_surf_pointcloud_iface, pointcloud_interp)
SMOOTH(float, radius)
SMOOTH(float3, position)
GPU_SHADER_NAMED_INTERFACE_END(pointcloud_interp)
GPU_SHADER_NAMED_INTERFACE_INFO(eevee_surf_pointcloud_flat_iface, pointcloud_interp_flat)
FLAT(int, id)
GPU_SHADER_NAMED_INTERFACE_END(pointcloud_interp_flat)

GPU_SHADER_CREATE_INFO(eevee_geom_pointcloud)
TYPEDEF_SOURCE("eevee_defines.hh")
PUSH_CONSTANT(bool, ptcloud_backface)
DEFINE("MAT_GEOM_POINTCLOUD")
VERTEX_SOURCE("eevee_geom_pointcloud_vert.glsl")
VERTEX_OUT(eevee_surf_iface)
VERTEX_OUT(eevee_surf_pointcloud_iface)
VERTEX_OUT(eevee_surf_pointcloud_flat_iface)
ADDITIONAL_INFO(draw_pointcloud)
ADDITIONAL_INFO(draw_modelmat)
ADDITIONAL_INFO(draw_object_infos)
ADDITIONAL_INFO(draw_resource_id_varying)
ADDITIONAL_INFO(draw_view)
GPU_SHADER_CREATE_END()

GPU_SHADER_CREATE_INFO(eevee_geom_volume)
TYPEDEF_SOURCE("eevee_defines.hh")
DEFINE("MAT_GEOM_VOLUME")
VERTEX_IN(0, float3, pos)
VERTEX_OUT(eevee_surf_iface)
VERTEX_SOURCE("eevee_geom_volume_vert.glsl")
ADDITIONAL_INFO(draw_modelmat)
ADDITIONAL_INFO(draw_object_infos)
ADDITIONAL_INFO(draw_resource_id_varying)
ADDITIONAL_INFO(draw_volume_infos)
ADDITIONAL_INFO(draw_view)
GPU_SHADER_CREATE_END()

GPU_SHADER_NAMED_INTERFACE_INFO(eevee_surf_curve_iface, curve_interp)
SMOOTH(float3, tangent)
SMOOTH(float3, binormal)
SMOOTH(float, time)
SMOOTH(float, time_width)
SMOOTH(float, radius)
SMOOTH(float, point_id) /* Smooth to be used for barycentric. */
GPU_SHADER_NAMED_INTERFACE_END(curve_interp)
GPU_SHADER_NAMED_INTERFACE_INFO(eevee_surf_curve_flat_iface, curve_interp_flat)
FLAT(int, strand_id)
GPU_SHADER_NAMED_INTERFACE_END(curve_interp_flat)

GPU_SHADER_CREATE_INFO(eevee_geom_curves)
TYPEDEF_SOURCE("eevee_defines.hh")
DEFINE("MAT_GEOM_CURVES")
VERTEX_SOURCE("eevee_geom_curves_vert.glsl")
VERTEX_OUT(eevee_surf_iface)
VERTEX_OUT(eevee_surf_curve_iface)
VERTEX_OUT(eevee_surf_curve_flat_iface)
ADDITIONAL_INFO(draw_modelmat)
ADDITIONAL_INFO(draw_object_infos)
ADDITIONAL_INFO(draw_resource_id_varying)
ADDITIONAL_INFO(draw_view)
ADDITIONAL_INFO(draw_curves)
ADDITIONAL_INFO(draw_curves_infos)
GPU_SHADER_CREATE_END()

GPU_SHADER_CREATE_INFO(eevee_geom_world)
TYPEDEF_SOURCE("eevee_defines.hh")
DEFINE("MAT_GEOM_WORLD")
BUILTINS(BuiltinBits::VERTEX_ID)
VERTEX_SOURCE("eevee_geom_world_vert.glsl")
VERTEX_OUT(eevee_surf_iface)
ADDITIONAL_INFO(draw_modelmat)
ADDITIONAL_INFO(draw_object_infos) /* Unused, but allow debug compilation. */
ADDITIONAL_INFO(draw_resource_id_varying)
ADDITIONAL_INFO(draw_view)
GPU_SHADER_CREATE_END()
