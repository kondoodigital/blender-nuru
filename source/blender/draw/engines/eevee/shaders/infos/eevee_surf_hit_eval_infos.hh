#ifdef GPU_SHADER
#  pragma once
#  include "gpu_shader_compat.hh"

#  include "draw_view_infos.hh"
#  include "eevee_common_infos.hh"
#  include "eevee_sampling_infos.hh"
#  include "eevee_uniform_infos.hh"
#endif

#include "eevee_defines.hh"
#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(eevee_surf_hit_eval)
DEFINE("MAT_HIT_EVAL")
TYPEDEF_SOURCE("eevee_raytrace_shared.hh")
TYPEDEF_SOURCE("eevee_defines.hh")
IMAGE(0, SFLOAT_16_16_16_16, read_write, image2D, hit_albedo_img)
IMAGE(1, SFLOAT_16_16_16_16, read_write, image2D, hit_material_img)
IMAGE(2, SFLOAT_16_16_16_16, read_write, image2D, hit_normal_img)
IMAGE(3, SFLOAT_16_16_16_16, read_write, image2D, hit_position_img)
IMAGE(4, RAYTRACE_RADIANCE_FORMAT, read_write, image2D, ray_radiance_img)
IMAGE(5, SFLOAT_16_16_16_16, read_write, image2D, hit_throughput_img)
SAMPLER(4, sampler2D, ray_data_tx)
SAMPLER(5, sampler2D, ray_time_tx)
SAMPLER(6, usampler2D, hit_identity_tx)
SAMPLER(7, sampler2D, hit_barycentric_tx)
SAMPLER(8, sampler2D, hit_world_position_tx)
STORAGE_BUF(11, read, HardwareHitEvalRecord, hit_eval_list_buf[])
FRAGMENT_SOURCE("eevee_surf_hit_eval_frag.glsl")
ADDITIONAL_INFO(eevee_global_ubo)
ADDITIONAL_INFO(eevee_hiz_data)
ADDITIONAL_INFO(eevee_sampling_data)
ADDITIONAL_INFO(eevee_utility_texture)
GPU_SHADER_CREATE_END()
