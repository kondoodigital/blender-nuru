/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#pragma once

#include "GPU_metal_raytrace.hh"

namespace blender::gpu::metal {

GPUMetalRaytraceScene *raytrace_scene_build(Span<GPUMetalRaytraceSceneEntry> entries,
                                            GPUMetalRaytraceSceneStats *r_stats);
bool raytrace_scene_update(GPUMetalRaytraceScene *scene,
                           Span<GPUMetalRaytraceSceneEntry> entries,
                           const GPUMetalRaytraceSceneUpdateParams &update_params,
                           GPUMetalRaytraceSceneStats *r_stats);
bool raytrace_scene_trace(GPUMetalRaytraceScene *scene, const GPUMetalRaytraceTraceParams &params);
bool raytrace_scene_trace_directional_shadow(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceDirectionalShadowParams &params);
bool raytrace_scene_trace_directional_hit_shadow(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceDirectionalHitShadowParams &params);
bool raytrace_scene_trace_local_shadow(GPUMetalRaytraceScene *scene,
                                       const GPUMetalRaytraceLocalShadowParams &params);
bool raytrace_scene_trace_local_hit_shadow(GPUMetalRaytraceScene *scene,
                                           const GPUMetalRaytraceLocalHitShadowParams &params);
bool raytrace_scene_shadow_batch_begin(GPUMetalRaytraceScene *scene);
bool raytrace_scene_shadow_batch_end(GPUMetalRaytraceScene *scene);
bool raytrace_scene_trace_environment_visibility(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceEnvironmentVisibilityParams &params);
bool raytrace_scene_trace_hit_environment_visibility(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceHitEnvironmentVisibilityParams &params);
bool raytrace_scene_trace_fast_gi(GPUMetalRaytraceScene *scene,
                                  const GPUMetalRaytraceFastGIParams &params);
bool raytrace_scene_trace_reflected_receiver_gi(
    GPUMetalRaytraceScene *scene, const GPUMetalRaytraceReflectedReceiverGIParams &params);
void raytrace_scene_free(GPUMetalRaytraceScene *scene);

}  // namespace blender::gpu::metal
