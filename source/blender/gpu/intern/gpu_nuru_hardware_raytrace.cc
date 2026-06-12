/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include "GPU_context.hh"
#include "GPU_nuru_hardware_raytrace.hh"

#ifdef WITH_METAL_BACKEND
#  include "metal/mtl_nuru_raytrace_acceleration.hh"
#endif

namespace blender {

GPUHardwareRaytraceScene *GPU_hardware_raytrace_scene_build(Span<GPUHardwareRaytraceSceneEntry> entries,
                                                      GPUHardwareRaytraceSceneStats *r_stats)
{
  if (r_stats != nullptr) {
    *r_stats = {};
  }

#ifdef WITH_METAL_BACKEND
  if (GPU_backend_get_type() == GPU_BACKEND_METAL) {
    return gpu::metal::raytrace_scene_build(entries, r_stats);
  }
#endif

  (void)entries;
  return nullptr;
}

bool GPU_hardware_raytrace_scene_update(GPUHardwareRaytraceScene *scene,
                                     Span<GPUHardwareRaytraceSceneEntry> entries,
                                     const GPUHardwareRaytraceSceneUpdateParams &update_params,
                                     GPUHardwareRaytraceSceneStats *r_stats)
{
  if (r_stats != nullptr) {
    *r_stats = {};
  }
#ifdef WITH_METAL_BACKEND
  if (scene != nullptr && GPU_backend_get_type() == GPU_BACKEND_METAL) {
    return gpu::metal::raytrace_scene_update(scene, entries, update_params, r_stats);
  }
#else
  (void)scene;
  (void)entries;
#endif
  (void)update_params;
  return false;
}

bool GPU_hardware_raytrace_scene_trace(GPUHardwareRaytraceScene *scene,
                                    const GPUHardwareRaytraceTraceParams &params)
{
#ifdef WITH_METAL_BACKEND
  if (scene != nullptr && GPU_backend_get_type() == GPU_BACKEND_METAL) {
    return gpu::metal::raytrace_scene_trace(scene, params);
  }
#else
  (void)scene;
  (void)params;
#endif
  return false;
}

bool GPU_hardware_raytrace_scene_trace_directional_shadow(
    GPUHardwareRaytraceScene *scene, const GPUHardwareRaytraceDirectionalShadowParams &params)
{
#ifdef WITH_METAL_BACKEND
  if (scene != nullptr && GPU_backend_get_type() == GPU_BACKEND_METAL) {
    return gpu::metal::raytrace_scene_trace_directional_shadow(scene, params);
  }
#else
  (void)scene;
  (void)params;
#endif
  return false;
}

bool GPU_hardware_raytrace_scene_trace_directional_hit_shadow(
    GPUHardwareRaytraceScene *scene, const GPUHardwareRaytraceDirectionalHitShadowParams &params)
{
#ifdef WITH_METAL_BACKEND
  if (scene != nullptr && GPU_backend_get_type() == GPU_BACKEND_METAL) {
    return gpu::metal::raytrace_scene_trace_directional_hit_shadow(scene, params);
  }
#else
  (void)scene;
  (void)params;
#endif
  return false;
}

bool GPU_hardware_raytrace_scene_trace_local_shadow(GPUHardwareRaytraceScene *scene,
                                                 const GPUHardwareRaytraceLocalShadowParams &params)
{
#ifdef WITH_METAL_BACKEND
  if (scene != nullptr && GPU_backend_get_type() == GPU_BACKEND_METAL) {
    return gpu::metal::raytrace_scene_trace_local_shadow(scene, params);
  }
#else
  (void)scene;
  (void)params;
#endif
  return false;
}

bool GPU_hardware_raytrace_scene_trace_local_hit_shadow(
    GPUHardwareRaytraceScene *scene, const GPUHardwareRaytraceLocalHitShadowParams &params)
{
#ifdef WITH_METAL_BACKEND
  if (scene != nullptr && GPU_backend_get_type() == GPU_BACKEND_METAL) {
    return gpu::metal::raytrace_scene_trace_local_hit_shadow(scene, params);
  }
#else
  (void)scene;
  (void)params;
#endif
  return false;
}

bool GPU_hardware_raytrace_scene_shadow_batch_begin(GPUHardwareRaytraceScene *scene)
{
#ifdef WITH_METAL_BACKEND
  if (scene != nullptr && GPU_backend_get_type() == GPU_BACKEND_METAL) {
    return gpu::metal::raytrace_scene_shadow_batch_begin(scene);
  }
#else
  (void)scene;
#endif
  return false;
}

bool GPU_hardware_raytrace_scene_shadow_batch_end(GPUHardwareRaytraceScene *scene)
{
#ifdef WITH_METAL_BACKEND
  if (scene != nullptr && GPU_backend_get_type() == GPU_BACKEND_METAL) {
    return gpu::metal::raytrace_scene_shadow_batch_end(scene);
  }
#else
  (void)scene;
#endif
  return false;
}

bool GPU_hardware_raytrace_scene_trace_environment_visibility(
    GPUHardwareRaytraceScene *scene, const GPUHardwareRaytraceEnvironmentVisibilityParams &params)
{
#ifdef WITH_METAL_BACKEND
  if (scene != nullptr && GPU_backend_get_type() == GPU_BACKEND_METAL) {
    return gpu::metal::raytrace_scene_trace_environment_visibility(scene, params);
  }
#else
  (void)scene;
  (void)params;
#endif
  return false;
}

bool GPU_hardware_raytrace_scene_trace_hit_environment_visibility(
    GPUHardwareRaytraceScene *scene, const GPUHardwareRaytraceHitEnvironmentVisibilityParams &params)
{
#ifdef WITH_METAL_BACKEND
  if (scene != nullptr && GPU_backend_get_type() == GPU_BACKEND_METAL) {
    return gpu::metal::raytrace_scene_trace_hit_environment_visibility(scene, params);
  }
#else
  (void)scene;
  (void)params;
#endif
  return false;
}

bool GPU_hardware_raytrace_denoise_oidn(const GPUHardwareRaytraceOIDNDenoiseParams &params)
{
#ifdef WITH_METAL_BACKEND
  if (GPU_backend_get_type() == GPU_BACKEND_METAL) {
    return gpu::metal::raytrace_denoise_oidn(params);
  }
#else
  (void)params;
#endif
  return false;
}

void GPU_hardware_raytrace_scene_free(GPUHardwareRaytraceScene *scene)
{
#ifdef WITH_METAL_BACKEND
  if (scene != nullptr) {
    gpu::metal::raytrace_scene_free(scene);
  }
#else
  (void)scene;
#endif
}

}  // namespace blender
