/* SPDX-FileCopyrightText: 2021 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup eevee
 */

#include "BKE_lib_id.hh"
#include "BKE_node.hh"
#include "BKE_node_legacy_types.hh"
#include "BLI_math_vector.hh"
#include "DEG_depsgraph_query.hh"
#include "NOD_shader.h"

#include "eevee_instance.hh"

namespace blender::eevee {

namespace {

static void sky_simplify_multiscatter_elevation_rotation(float &sun_elevation, float &sun_rotation)
{
  float new_sun_elevation = fmodf(sun_elevation, 2.0f * M_PI);
  if (fabsf(new_sun_elevation) >= M_PI) {
    new_sun_elevation -= copysignf(2.0f, new_sun_elevation) * M_PI;
  }
  if (new_sun_elevation >= M_PI_2 || new_sun_elevation <= -M_PI_2) {
    new_sun_elevation = copysignf(M_PI, new_sun_elevation) - new_sun_elevation;
    sun_rotation += M_PI;
  }

  float new_sun_rotation = fmodf(sun_rotation, 2.0f * M_PI);
  if (new_sun_rotation < 0.0f) {
    new_sun_rotation += 2.0f * M_PI;
  }
  new_sun_rotation = 2.0f * M_PI - new_sun_rotation;

  sun_elevation = new_sun_elevation;
  sun_rotation = new_sun_rotation;
}

static bool node_visited(const Vector<const bNode *> &visited, const bNode *node)
{
  for (const bNode *entry : visited) {
    if (entry == node) {
      return true;
    }
  }
  return false;
}

static const NodeTexSky *world_linked_sky_texture(const bNodeTree &ntree)
{
  bNode *output_node = ntreeShaderOutputNode(const_cast<bNodeTree *>(&ntree), SHD_OUTPUT_EEVEE);
  if (output_node == nullptr) {
    return nullptr;
  }

  const bNodeSocket *surface_socket = bke::node_find_socket(*output_node, SOCK_IN, "Surface");
  if (surface_socket == nullptr || surface_socket->link == nullptr) {
    return nullptr;
  }

  Vector<const bNode *> stack;
  Vector<const bNode *> visited;
  stack.append(surface_socket->link->fromnode);

  while (!stack.is_empty()) {
    const bNode *node = stack.pop_last();
    if (node == nullptr || node_visited(visited, node)) {
      continue;
    }
    visited.append(node);

    if (node->type_legacy == SH_NODE_TEX_SKY && node->storage != nullptr) {
      return static_cast<const NodeTexSky *>(node->storage);
    }

    for (const bNodeSocket &input : node->inputs) {
      if (input.link != nullptr) {
        stack.append(input.link->fromnode);
      }
    }
  }

  return nullptr;
}

static bool world_sky_shadow_direction(const bNodeTree *ntree, float3 &r_direction)
{
  if (ntree == nullptr) {
    return false;
  }

  const NodeTexSky *sky = world_linked_sky_texture(*ntree);
  if (sky == nullptr || sky->sun_disc == 0) {
    return false;
  }

  if (ELEM(sky->sky_model, SHD_SKY_PREETHAM, SHD_SKY_HOSEK)) {
    const float3 dir = float3(sky->sun_direction[0], sky->sun_direction[1], sky->sun_direction[2]);
    if (math::length_squared(dir) <= 1.0e-8f) {
      return false;
    }
    r_direction = math::normalize(dir);
    return true;
  }

  float sun_elevation = sky->sun_elevation;
  float sun_rotation = sky->sun_rotation;
  if (sky->sky_model == SHD_SKY_SINGLE_SCATTERING) {
    sun_rotation = fmodf(sun_rotation, 2.0f * M_PI);
    if (sun_rotation < 0.0f) {
      sun_rotation += 2.0f * M_PI;
    }
    sun_rotation = 2.0f * M_PI - sun_rotation;
  }
  else {
    sky_simplify_multiscatter_elevation_rotation(sun_elevation, sun_rotation);
  }

  const float longitude = sun_rotation - M_PI_2;
  r_direction = float3(cosf(sun_elevation) * cosf(longitude),
                       cosf(sun_elevation) * sinf(longitude),
                       sinf(sun_elevation));
  return math::length_squared(r_direction) > 1.0e-8f;
}

}  // namespace

/* -------------------------------------------------------------------- */
/** \name World
 *
 * \{ */

World::~World()
{
  if (default_world_ != nullptr) {
    BKE_id_free(nullptr, default_world_);
  }
}

blender::World *World::default_world_get()
{
  if (default_world_ == nullptr) {
    default_world_ = BKE_id_new_nomain<blender::World>("EEVEE default world");

    BLI_listbase_clear(&default_world_->gpumaterial);
  }
  return default_world_;
}

blender::World *World::scene_world_get()
{
  return (inst_.scene->world != nullptr) ? inst_.scene->world : default_world_get();
}

float World::sun_threshold()
{
  /* No sun extraction during baking. */
  if (inst_.is_baking()) {
    return 0.0;
  }

  float sun_threshold = scene_world_get()->sun_threshold;
  if (inst_.use_studio_light()) {
    /* Do not call `lookdev_world_.intensity_get()` as it might not be initialized yet. */
    sun_threshold *= inst_.v3d->shading.studiolight_intensity;
  }
  return sun_threshold;
}

void World::sync_sunlight_rt_direction_overrides()
{
  for (const int i : IndexRange(WORLD_SUN_MAX)) {
    sunlight_rt[i] = LightData{};
    sunlight_rt_direction[i] = float4(0.0f);
  }

  float3 exact_sky_direction;
  if (sky_sun_shadow_direction_get(exact_sky_direction)) {
    LightData override_light = {};
    override_light.object_to_world.x.z = exact_sky_direction.x;
    override_light.object_to_world.y.z = exact_sky_direction.y;
    override_light.object_to_world.z.z = exact_sky_direction.z;

    sunlight_rt[WORLD_SUN_DIFFUSE] = override_light;
    sunlight_rt[WORLD_SUN_GLOSSY] = override_light;
    const float4 packed_direction = float4(exact_sky_direction, 1.0f);
    sunlight_rt_direction[WORLD_SUN_DIFFUSE] = packed_direction;
    sunlight_rt_direction[WORLD_SUN_GLOSSY] = packed_direction;
  }

  sunlight_rt.push_update();
  sunlight_rt_direction.push_update();
}

void World::sync()
{
  bool has_update = false;
  fast_gi_changed_ = false;

  WorldHandle wo_handle = {0};
  if (inst_.scene->world != nullptr) {
    /* Detect world update before overriding it. */
    wo_handle = inst_.sync.sync_world(*inst_.scene->world);
    has_update = wo_handle.recalc != 0;
  }

  bool wait_ready = true;  // TODO !inst_.is_image_render;

  /* Sync volume first since its result can override the surface world. */
  sync_volume(wo_handle, wait_ready);

  blender::World *bl_world;
  if (inst_.use_studio_light()) {
    has_update |= lookdev_world_.sync(LookdevParameters(inst_.v3d));
    bl_world = lookdev_world_.world_get();
  }
  else if ((inst_.view_layer->layflag & SCE_LAY_SKY) == 0) {
    bl_world = default_world_get();
  }
  else if (has_volume_absorption_) {
    bl_world = default_world_get();
  }
  else {
    bl_world = scene_world_get();
  }

  blender::World *world_override = DEG_get_evaluated(inst_.depsgraph,
                                                     inst_.view_layer->world_override);
  if (world_override) {
    bl_world = world_override;
  }

  bNodeTree *ntree = (bl_world->nodetree) ? bl_world->nodetree : default_world_get()->nodetree;

  {
    if (has_volume_absorption_) {
      /* Replace world by black world. */
      bl_world = default_world_get();
    }
  }

  /* We have to manually test here because we have overrides. */
  blender::World *orig_world = DEG_get_original(bl_world);
  const bool original_world_changed = assign_if_different(prev_original_world, orig_world);
  if (original_world_changed) {
    has_update = true;
  }

  has_sky_sun_shadow_ = world_sky_shadow_direction(
      bl_world->nodetree ? bl_world->nodetree : default_world_get()->nodetree,
      sky_sun_shadow_direction_);

  inst_.light_probes.sync_world(bl_world, has_update);
  fast_gi_changed_ = has_update;

  if (inst_.is_viewport() && has_update) {
    /* Catch lookdev viewport properties updates. */
    inst_.sampling.reset();
  }

  GPUMaterial *gpumat = inst_.shaders.world_shader_get(
      bl_world, ntree, MAT_PIPE_DEFERRED, !wait_ready);

  if (GPU_material_status(gpumat) == GPU_MAT_FAILED) {
    bl_world = default_world_get();
    ntree = bl_world->nodetree;
    gpumat = inst_.shaders.world_shader_get(bl_world, ntree, MAT_PIPE_DEFERRED, !wait_ready);
  }
  if (GPU_material_status(gpumat) == GPU_MAT_QUEUED) {
    is_ready_ = false;
    return;
  }
  is_ready_ = true;

  fast_gi_changed_ = original_world_changed;
  fast_gi_changed_ |= assign_if_different(prev_fast_gi_ntree_, ntree);
  fast_gi_changed_ |= assign_if_different(prev_fast_gi_gpumat_, gpumat);
  fast_gi_changed_ |= assign_if_different(prev_fast_gi_has_sky_sun_shadow_, has_sky_sun_shadow_);
  if (has_sky_sun_shadow_) {
    fast_gi_changed_ |= assign_if_different(prev_fast_gi_sky_sun_shadow_direction_,
                                            sky_sun_shadow_direction_);
  }
  else {
    prev_fast_gi_sky_sun_shadow_direction_ = float3(0.0f, 0.0f, 1.0f);
  }

  inst_.manager->register_layer_attributes(gpumat);

  float opacity = inst_.use_studio_light() ? lookdev_world_.background_opacity_get() :
                                             inst_.film.background_opacity_get();
  float background_blur = inst_.use_studio_light() ? lookdev_world_.background_blur_get() : 0.0;

  inst_.pipelines.background.sync(gpumat, opacity, background_blur);
  inst_.pipelines.world.sync(gpumat);
}

void World::sync_volume(const WorldHandle &world_handle, bool wait_ready)
{
  /* Studio lights have no volume shader. */
  blender::World *world = inst_.use_studio_light() ? nullptr : inst_.scene->world;

  GPUMaterial *gpumat = nullptr;

  /* Only the scene world nodetree can have volume shader. */
  if (world && world->nodetree) {
    gpumat = inst_.shaders.world_shader_get(
        world, world->nodetree, MAT_PIPE_VOLUME_MATERIAL, !wait_ready);
  }

  bool had_volume = has_volume_;

  if (gpumat && (GPU_material_status(gpumat) == GPU_MAT_SUCCESS)) {
    has_volume_ = GPU_material_has_volume_output(gpumat);
    has_volume_scatter_ = GPU_material_flag_get(gpumat, GPU_MATFLAG_VOLUME_SCATTER);
    has_volume_absorption_ = GPU_material_flag_get(gpumat, GPU_MATFLAG_VOLUME_ABSORPTION);
  }
  else {
    has_volume_ = has_volume_absorption_ = has_volume_scatter_ = false;
  }

  /* World volume needs to be always synced for correct clearing of parameter buffers. */
  inst_.pipelines.world_volume.sync(gpumat);

  if (has_volume_ || had_volume) {
    inst_.volume.world_sync(world_handle);
  }
}

/** \} */

}  // namespace blender::eevee
