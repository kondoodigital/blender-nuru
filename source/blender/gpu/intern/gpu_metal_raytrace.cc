/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include "GPU_context.hh"
#include "GPU_metal_raytrace.hh"

#ifdef WITH_METAL_BACKEND
#  include "metal/mtl_raytrace_acceleration.hh"
#endif

namespace blender {

GPUMetalRaytraceScene *GPU_metal_raytrace_scene_build(Span<GPUMetalRaytraceSceneEntry> entries,
                                                      GPUMetalRaytraceSceneStats *r_stats)
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

bool GPU_metal_raytrace_scene_update(GPUMetalRaytraceScene *scene,
                                     Span<GPUMetalRaytraceSceneEntry> entries,
                                     const GPUMetalRaytraceSceneUpdateParams &update_params,
                                     GPUMetalRaytraceSceneStats *r_stats)
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

bool GPU_metal_raytrace_scene_trace(GPUMetalRaytraceScene *scene,
                                    const GPUMetalRaytraceTraceParams &params)
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

bool GPU_metal_raytrace_scene_trace_directional_shadow(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceDirectionalShadowParams &params)
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

bool GPU_metal_raytrace_scene_trace_directional_hit_shadow(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceDirectionalHitShadowParams &params)
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

bool GPU_metal_raytrace_scene_trace_local_shadow(GPUMetalRaytraceScene *scene,
                                                 const GPUMetalRaytraceLocalShadowParams &params)
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

bool GPU_metal_raytrace_scene_trace_local_hit_shadow(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceLocalHitShadowParams &params)
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

bool GPU_metal_raytrace_scene_shadow_batch_begin(GPUMetalRaytraceScene *scene)
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

bool GPU_metal_raytrace_scene_shadow_batch_end(GPUMetalRaytraceScene *scene)
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

bool GPU_metal_raytrace_scene_trace_environment_visibility(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceEnvironmentVisibilityParams &params)
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

bool GPU_metal_raytrace_scene_trace_hit_environment_visibility(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceHitEnvironmentVisibilityParams &params)
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

bool GPU_metal_raytrace_scene_trace_fast_gi(GPUMetalRaytraceScene *scene,
                                            const GPUMetalRaytraceFastGIParams &params)
{
#ifdef WITH_METAL_BACKEND
  if (scene != nullptr && GPU_backend_get_type() == GPU_BACKEND_METAL) {
    return gpu::metal::raytrace_scene_trace_fast_gi(scene, params);
  }
#else
  (void)scene;
  (void)params;
#endif
  return false;
}

bool GPU_metal_raytrace_scene_trace_reflected_receiver_gi(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceReflectedReceiverGIParams &params)
{
#ifdef WITH_METAL_BACKEND
  if (scene != nullptr && GPU_backend_get_type() == GPU_BACKEND_METAL) {
    return gpu::metal::raytrace_scene_trace_reflected_receiver_gi(scene, params);
  }
#else
  (void)scene;
  (void)params;
#endif
  return false;
}

void GPU_metal_raytrace_scene_free(GPUMetalRaytraceScene *scene)
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
