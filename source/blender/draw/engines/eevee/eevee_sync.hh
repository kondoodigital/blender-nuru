/* SPDX-FileCopyrightText: 2021 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup eevee
 *
 * Structures to identify unique data blocks. The keys are unique so we are able to
 * match ids across frame updates.
 */

#pragma once

#include "BKE_duplilist.hh"
#include "BLI_math_matrix_types.hh"
#include "BLI_map.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_vector.hh"
#include "DNA_modifier_types.h"
#include "DNA_object_types.h"
#include "DRW_render.hh"

#include "draw_handle.hh"

namespace blender::eevee {

using namespace draw;

class Instance;

/* -------------------------------------------------------------------- */
/** \name Sync Module
 *
 * \{ */

struct BaseHandle {
  unsigned int recalc;
};

struct ObjectHandle : BaseHandle {
  ObjectKey object_key;
};

struct WorldHandle : public BaseHandle {};

struct SceneHandle : public BaseHandle {};

struct HardwareRaytraceSceneEntry {
  ObjectKey object_key;
  /* Replay-safe evaluated object used for sparse hit-eval material replay. */
  Object *hit_eval_object = nullptr;
  blender::gpu::Batch *batch = nullptr;
  int recalc = 0;
  ResourceHandleRange resource_handle = {};
  float4x4 object_to_world = float4x4::identity();
  uint32_t instance_count = 1;
  int material_slot = -1;
  bool is_sculpt = false;
  float3 emissive_radiance = float3(0.0f);
  float3 diffuse_albedo = float3(0.8f);
  float3 reflection_color = float3(0.8f);
  float reflection_roughness = 1.0f;
  float3 transmission_color = float3(0.8f);
  float transmission_roughness = 1.0f;
  float reflection_ior = 1.45f;
  float refraction_ior = 1.45f;
  float packed_thickness = 0.0f;
  float alpha = 1.0f;
  float reflection_layer_coverage = 0.0f;
  uint32_t closure_type = 1u;
  uint32_t proxy_flags = 0u;
  uint64_t material_runtime_hash = 0;
};

class SyncModule {
 private:
  Instance &inst_;

  Map<ObjectKey, ObjectHandle> ob_handles = {};
  Vector<HardwareRaytraceSceneEntry> hardware_raytrace_scene_entries_;
  uint64_t hardware_raytrace_scene_signature_ = 0;

 public:
  SyncModule(Instance &inst) : inst_(inst) {};
  ~SyncModule() {};

  void begin_sync();

  ObjectHandle &sync_object(const ObjectRef &ob_ref);
  WorldHandle sync_world(const blender::World &world);

  void sync_mesh(Object *ob, ObjectHandle &ob_handle, const ObjectRef &ob_ref);
  bool sync_sculpt(Object *ob, ObjectHandle &ob_handle, const ObjectRef &ob_ref);
  void sync_pointcloud(Object *ob, ObjectHandle &ob_handle, const ObjectRef &ob_ref);
  void sync_volume(Object *ob, ObjectHandle &ob_handle, const ObjectRef &ob_ref);
  void sync_curves(Object *ob,
                   ObjectHandle &ob_handle,
                   const ObjectRef &ob_ref,
                   ResourceHandleRange res_handle = {},
                   ModifierData *modifier_data = nullptr,
                   ParticleSystem *particle_sys = nullptr);

  const Vector<HardwareRaytraceSceneEntry> &hardware_raytrace_scene_entries() const
  {
    return hardware_raytrace_scene_entries_;
  }

  uint64_t hardware_raytrace_scene_signature() const
  {
    return hardware_raytrace_scene_signature_;
  }

  void append_hardware_raytrace_scene_entry(Object *ob,
                                            Object *hit_eval_object,
                                            const ObjectKey &object_key,
                                            blender::gpu::Batch *geom,
                                            int recalc,
                                            ResourceHandleRange res_handle,
                                            int material_slot,
                                            bool is_sculpt,
                                            /* Traversal/continuation stay on this bounded proxy set.
                                             * Full material replay is deferred to sparse hit-eval. */
                                            const float3 &emissive_radiance,
                                            const float3 &diffuse_albedo,
                                            const float3 &reflection_color,
                                            float reflection_roughness,
                                            const float3 &transmission_color,
                                            float transmission_roughness,
                                            float reflection_ior,
                                            float refraction_ior,
                                            float packed_thickness,
                                            float alpha,
                                            float reflection_layer_coverage,
                                            uint32_t closure_type,
                                            uint32_t proxy_flags,
                                            uint64_t material_runtime_hash);
};

using HairHandleCallback = FunctionRef<void(ObjectHandle, ModifierData &, ParticleSystem &)>;
void foreach_hair_particle_handle(Instance &inst,
                                  ObjectRef &ob_ref,
                                  ObjectHandle ob_handle,
                                  HairHandleCallback callback);

/** \} */

}  // namespace blender::eevee
