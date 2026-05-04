/* SPDX-FileCopyrightText: 2021 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup eevee
 *
 * World rendering with material handling. Also take care of lookdev
 * HDRI and default material.
 */

#pragma once

#include "DNA_world_types.h"

#include "eevee_lookdev.hh"
#include "eevee_sync.hh"

namespace blender {

struct bNodeTree;
struct bNodeSocketValueRGBA;
struct UniformBuffer;

namespace eevee {

class Instance;

/** \} */

/* -------------------------------------------------------------------- */
/** \name World
 *
 * \{ */

class World {
 public:
  /**
   * Buffer containing the sun light for the world.
   * Filled by #LightProbeModule and read by #LightModule.
   */
  UniformArrayBuffer<LightData, 2> sunlight = {"sunlight"};
  draw::StorageArrayBuffer<LightData, WORLD_SUN_MAX> sunlight_rt = {"sunlight_rt"};
  draw::StorageArrayBuffer<float4, WORLD_SUN_MAX> sunlight_rt_direction = {
      "sunlight_rt_direction"};

 private:
  Instance &inst_;

  /* Used to detect if world change. */
  blender::World *prev_original_world = nullptr;
  bNodeTree *prev_fast_gi_ntree_ = nullptr;
  GPUMaterial *prev_fast_gi_gpumat_ = nullptr;
  bool prev_fast_gi_has_sky_sun_shadow_ = false;
  float3 prev_fast_gi_sky_sun_shadow_direction_ = float3(0.0f, 0.0f, 1.0f);

  /* Used when the scene doesn't have a world. */
  blender::World *default_world_ = nullptr;

  /* Is true if world as a valid volume shader compiled. */
  bool has_volume_ = false;
  /* Is true if the volume shader has absorption. Disables distant lights. */
  bool has_volume_absorption_ = false;
  /* Is true if the volume shader has scattering. */
  bool has_volume_scatter_ = false;
  /* Is true if the surface shader is compiled and ready. */
  bool is_ready_ = false;
  /* Latched once per sync so world-space GI can invalidate on HDRI/sky changes. */
  bool fast_gi_changed_ = false;
  /* CPU-side fallback used only for Hardware RT world-sun shadow tracing. */
  bool has_sky_sun_shadow_ = false;
  float3 sky_sun_shadow_direction_ = float3(0.0f, 0.0f, 1.0f);

  LookdevWorld lookdev_world_;

 public:
  World(Instance &inst) : inst_(inst) {};
  ~World();

  /* Setup and request the background shader. */
  void sync();

  bool has_volume() const
  {
    return has_volume_;
  }

  bool has_volume_absorption() const
  {
    return has_volume_absorption_;
  }

  bool has_volume_scatter() const
  {
    return has_volume_scatter_;
  }

  bool is_ready() const
  {
    return is_ready_;
  }

  bool fast_gi_changed() const
  {
    return fast_gi_changed_;
  }

  float sun_threshold();

  float sun_angle()
  {
    return scene_world_get()->sun_angle;
  }

  float sun_shadow_max_resolution()
  {
    return scene_world_get()->sun_shadow_maximum_resolution;
  }

  float sun_shadow_filter_radius()
  {
    return scene_world_get()->sun_shadow_filter_radius;
  }

  float sun_shadow_jitter_overblur()
  {
    return scene_world_get()->sun_shadow_jitter_overblur;
  }

  bool use_sun_shadow()
  {
    return scene_world_get()->flag & WO_USE_SUN_SHADOW;
  }

  bool use_sun_shadow_jitter()
  {
    return scene_world_get()->flag & WO_USE_SUN_SHADOW_JITTER;
  }

  bool sky_sun_shadow_direction_get(float3 &r_direction) const
  {
    if (!has_sky_sun_shadow_) {
      return false;
    }
    r_direction = sky_sun_shadow_direction_;
    return true;
  }

  void sync_sunlight_rt_direction_overrides();

 private:
  void sync_volume(const WorldHandle &world_handle, bool wait_ready);

  /* Returns a dummy black world for when a valid world isn't present or when we want to suppress
   * any light coming from the world. */
  blender::World *default_world_get();

  /* Returns either the scene world or the default world if scene has no world. */
  blender::World *scene_world_get();
};

/** \} */

}  // namespace eevee
}  // namespace blender
