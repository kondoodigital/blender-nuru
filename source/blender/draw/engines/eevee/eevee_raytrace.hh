/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup eevee
 *
 * The ray-tracing module class handles ray generation, scheduling, tracing and denoising.
 */

#pragma once

#include "GPU_capabilities.hh"
#include "GPU_metal_raytrace.hh"

#include "DNA_scene_types.h"

#include "DRW_gpu_wrapper.hh"
#include "DRW_render.hh"

#include "eevee_raytrace_shared.hh"
#include "eevee_sync.hh"

namespace blender::eevee {

class Instance;

using RayTraceTileBuf = draw::StorageArrayBuffer<uint, 1024, true>;

/* -------------------------------------------------------------------- */
/** \name Ray-tracing Buffers
 *
 * Contain persistent data used for temporal denoising. Similar to \class GBuffer but only contains
 * persistent data.
 * \{ */

/**
 * Contain persistent buffer that need to be stored per view, per deferred layer.
 */
struct RayTraceBuffer {
  /** Set of buffers that need to be allocated for each ray type. */
  struct DenoiseBuffer {
    /* Persistent history buffers. */
    TextureFromPool radiance_history_tx = {"radiance_tx"};
    TextureFromPool variance_history_tx = {"variance_tx"};
    TextureFromPool screen_ownership_history_tx = {"screen_ownership_history_tx"};
    /* Map of tiles that were processed inside the history buffer. */
    Texture tilemask_history_tx = {"tilemask_tx"};
    /** Perspective matrix for which the history buffers were recorded. */
    float4x4 history_persmat;
    /** True if history buffer was used last frame and can be re-projected. */
    bool valid_history = false;
    /** True if screen-ownership history was written last frame and can be re-projected. */
    bool valid_screen_ownership_history = false;
    /**
     * Textures containing the ray hit radiance denoised (full-res). One of them is result_tx.
     * One might become result buffer so it need instantiation by closure type to avoid reuse.
     */
    TextureFromPool denoised_spatial_tx = {"denoised_spatial_tx"};
    TextureFromPool denoised_temporal_tx = {"denoised_temporal_tx"};
    TextureFromPool denoised_bilateral_tx = {"denoised_bilateral_tx"};
  };
  /**
   * One for each closure. Not to be mistaken with deferred layer type.
   */
  DenoiseBuffer closures[3];

  /**
   * Radiance feedback of the deferred layer for next sample's reflection or next layer's
   * transmission.
   */
  Texture radiance_feedback_tx = {"radiance_feedback_tx"};
  /**
   * Perspective matrix for which the radiance feedback buffer was recorded.
   * Can be different from de-noise buffer's history matrix.
   */
  float4x4 history_persmat = float4x4::zero();

  gpu::Texture *feedback_ensure(bool is_dummy, int2 extent)
  {
    eGPUTextureUsage usage_rw = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;
    if (radiance_feedback_tx.ensure_2d(
            gpu::TextureFormat::SFLOAT_16_16_16_16, is_dummy ? int2(1) : extent, usage_rw))
    {
      radiance_feedback_tx.clear(float4(0.0f));
    }
    return radiance_feedback_tx;
  }
};

/**
 * Contains the result texture.
 * The result buffer is usually short lived and is kept in a TextureFromPool managed by the mode.
 * This structure contains a reference to it so that it can be freed after use by the caller.
 */
class RayTraceResultTexture {
 private:
  /** Result is in a temporary texture that needs to be released. */
  TextureFromPool *result_ = nullptr;
  /** Value of `result_->tx_` that can be referenced in advance. */
  gpu::Texture *tx_ = nullptr;
  /** History buffer to swap the temporary texture that does not need to be released. */
  TextureFromPool *history_ = nullptr;

 public:
  RayTraceResultTexture() = default;
  RayTraceResultTexture(TextureFromPool &result) : result_(result.ptr()), tx_(result) {};
  RayTraceResultTexture(TextureFromPool &result, TextureFromPool &history)
      : result_(result.ptr()), tx_(result), history_(history.ptr()) {};

  operator gpu::Texture *() const
  {
    BLI_assert(tx_ != nullptr);
    return tx_;
  }

  gpu::Texture **operator&()
  {
    return &tx_;
  }

  void release()
  {
    if (history_) {
      /* Swap after last use, retain history until next cycle. */
      TextureFromPool::swap(*result_, *history_);
      history_->retain();
    }
    /* Release previous history. */
    result_->release();
  }
};

struct RayTraceResult {
  RayTraceResultTexture closures[3];

  void release()
  {
    for (int i = 0; i < 3; i++) {
      closures[i].release();
    }
  }
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Ray-tracing
 * \{ */

class RayTraceModule {
 private:
  Instance &inst_;

  draw::PassSimple tile_classify_ps_ = {"TileClassify"};
  draw::PassSimple tile_compact_ps_ = {"TileCompact"};
  draw::PassSimple hardware_direct_light_tile_compact_ps_ = {"HardwareDirectLightTileCompact"};
  draw::PassSimple hardware_direct_light_visibility_ps_ = {"HardwareDirectLightVisibility"};
  draw::PassSimple hardware_direct_light_accum_ps_ = {"HardwareDirectLightAccum"};
  draw::PassSimple hardware_direct_light_denoise_ps_ = {"HardwareDirectLightDenoise"};
  draw::PassSimple hardware_trace_tile_compact_ps_ = {"HardwareTraceTileCompact"};
  draw::PassSimple hardware_tile_compact_ps_ = {"HardwareTileCompact"};
  draw::PassSimple generate_ps_ = {"RayGenerate"};
  draw::PassSimple trace_planar_ps_ = {"Trace.Planar"};
  draw::PassSimple trace_screen_ps_ = {"Trace.Screen"};
  draw::PassSimple trace_fallback_ps_ = {"Trace.Fallback"};
  draw::PassSimple trace_hardware_lighting_ps_ = {"Trace.HardwareLighting"};
  draw::PassSimple hardware_reflected_receiver_gi_blur_ps_ = {"Trace.HardwareReflectedReceiverGIBlur"};
  draw::PassSimple hardware_layered_receiver_gi_blur_ps_ = {"Trace.HardwareLayeredReceiverGIBlur"};
  draw::PassSimple hardware_transmission_receiver_gi_blur_ps_ = {
      "Trace.HardwareTransmissionReceiverGIBlur"};
  draw::PassSimple hardware_secondary_photon_gi_blur_ps_ = {"Trace.HardwareSecondaryPhotonGIBlur"};
  draw::PassSimple hardware_layered_secondary_photon_gi_blur_ps_ = {
      "Trace.HardwareLayeredSecondaryPhotonGIBlur"};
  draw::PassSimple hardware_transmission_secondary_photon_gi_blur_ps_ = {
      "Trace.HardwareTransmissionSecondaryPhotonGIBlur"};
  draw::PassSimple scene_final_specular_resolve_ps_ = {"Trace.SceneFinalSpecularResolve"};
  draw::PassSimple hardware_indirect_gi_cache_store_ps_ = {"Trace.HardwareIndirectGICacheStore"};
  draw::PassSimple hardware_fast_gi_update_ps_[3] = {{"Trace.HardwareFastGIUpdate0"},
                                                     {"Trace.HardwareFastGIUpdate1"},
                                                     {"Trace.HardwareFastGIUpdate2"}};
  int hardware_fast_gi_cascade_index_[3] = {0, 1, 2};
  draw::PassSimple hit_eval_count_ps_ = {"Trace.HitEvalCount"};
  draw::PassSimple hit_eval_prefix_ps_ = {"Trace.HitEvalPrefix"};
  draw::PassSimple hit_eval_compact_ps_ = {"Trace.HitEvalCompact"};
  draw::PassSimple hit_eval_ps_ = {"Trace.HitEval"};
  draw::PassSimple denoise_spatial_ps_ = {"DenoiseSpatial"};
  draw::PassSimple denoise_temporal_ps_ = {"DenoiseTemporal"};
  draw::PassSimple denoise_bilateral_ps_ = {"DenoiseBilateral"};
  draw::PassSimple horizon_schedule_ps_ = {"HorizonScan.Schedule"};
  draw::PassSimple horizon_setup_ps_ = {"HorizonScan.Setup"};
  draw::PassSimple horizon_scan_ps_ = {"HorizonScan.Trace"};
  draw::PassSimple horizon_denoise_ps_ = {"HorizonScan.Denoise"};
  draw::PassSimple horizon_resolve_ps_ = {"HorizonScan.Resolve"};

  /** Dispatch with enough tiles for the whole screen. */
  int3 tile_classify_dispatch_size_ = int3(1);
  /** Dispatch with enough tiles for the tile mask. */
  int3 tile_compact_dispatch_size_ = int3(1);
  /** Dispatch with enough tiles for the direct-light work queue. */
  int3 hardware_direct_light_tile_compact_dispatch_size_ = int3(1);
  int3 horizon_schedule_dispatch_size_ = int3(1);
  /** Dispatch with enough tiles for the tracing resolution. */
  int3 tracing_dispatch_size_ = int3(1);
  int3 horizon_tracing_dispatch_size_ = int3(1);
  /** 2D tile mask to check which unused adjacent tile we need to clear and which tile we need to
   * dispatch for each work type. */
  Texture tile_raytrace_denoise_tx_ = {"tile_raytrace_denoise_tx_"};
  Texture tile_raytrace_tracing_tx_ = {"tile_raytrace_tracing_tx_"};
  Texture tile_horizon_denoise_tx_ = {"tile_horizon_denoise_tx_"};
  Texture tile_horizon_tracing_tx_ = {"tile_horizon_tracing_tx_"};
  /** Indirect dispatch rays. Avoid dispatching work-groups that will not trace anything. */
  DispatchIndirectBuf raytrace_tracing_dispatch_buf_ = {"raytrace_tracing_dispatch_buf_"};
  /** Indirect dispatch denoise full-resolution tiles. */
  DispatchIndirectBuf raytrace_denoise_dispatch_buf_ = {"raytrace_denoise_dispatch_buf_"};
  /** Indirect dispatch for the Metal Hardware trace kernel itself. */
  DispatchIndirectBuf hardware_trace_dispatch_buf_ = {"hardware_trace_dispatch_buf_"};
  /** Indirect dispatch for direct-light tile work generation. */
  DispatchIndirectBuf hardware_direct_light_dispatch_buf_ = {"hardware_direct_light_dispatch_buf_"};
  /** Indirect dispatch for downstream Hardware-only resolve work. */
  DispatchIndirectBuf hardware_resolve_dispatch_buf_ = {"hardware_resolve_dispatch_buf_"};
  /** Indirect dispatch horizon scan. Avoid dispatching work-groups that will not scan anything. */
  DispatchIndirectBuf horizon_tracing_dispatch_buf_ = {"horizon_tracing_dispatch_buf_"};
  /** Indirect dispatch denoise full-resolution tiles. */
  DispatchIndirectBuf horizon_denoise_dispatch_buf_ = {"horizon_denoise_dispatch_buf_"};
  /** Pointer to the texture to store the result of horizon scan in. */
  gpu::Texture *horizon_scan_output_tx_[3] = {nullptr};
  /** Tile buffer that contains tile coordinates. */
  RayTraceTileBuf raytrace_tracing_tiles_buf_ = {"raytrace_tracing_tiles_buf_"};
  RayTraceTileBuf raytrace_denoise_tiles_buf_ = {"raytrace_denoise_tiles_buf_"};
  draw::StorageArrayBuffer<HardwareDirectLightWorkTile, 1024, true>
      hardware_direct_light_work_tiles_buf_ = {"hardware_direct_light_work_tiles_buf_"};
  draw::StorageArrayBuffer<HardwareDirectLightVisibilitySample, 1024, true>
      hardware_direct_light_visibility_samples_buf_ = {"hardware_direct_light_visibility_samples_buf_"};
  draw::StorageArrayBuffer<GPUMetalRaytraceFastGILightRecord, 256, true> hardware_fast_gi_light_buf_ = {
      "hardware_fast_gi_light_buf_"};
  RayTraceTileBuf hardware_trace_tiles_buf_ = {"hardware_trace_tiles_buf_"};
  RayTraceTileBuf hardware_resolve_tiles_buf_ = {"hardware_resolve_tiles_buf_"};
  RayTraceTileBuf horizon_tracing_tiles_buf_ = {"horizon_tracing_tiles_buf_"};
  RayTraceTileBuf horizon_denoise_tiles_buf_ = {"horizon_denoise_tiles_buf_"};
  /** Texture containing the ray direction and PDF. */
  TextureFromPool ray_data_tx_ = {"ray_data_tx"};
  /** Texture containing the ray hit time. */
  TextureFromPool ray_time_tx_ = {"ray_data_tx"};
  /** Texture containing the ray hit radiance (tracing-res). */
  TextureFromPool ray_radiance_tx_ = {"ray_radiance_tx"};
  /** Hybrid screen-hit continuation origin/time for bounce 2+ handoff into Hardware RT. */
  TextureFromPool screen_continuation_tx_ = {"screen_continuation_tx_"};
  /** Current-frame Hybrid screen ownership written by the screen trace pass. */
  TextureFromPool screen_ownership_tx_ = {"screen_ownership_tx_"};
  /** Approximate hit albedo exported by the hardware trace. */
  TextureFromPool hit_albedo_tx_ = {"hit_albedo_tx_"};
  /** Specular path-throughput tint carried from earlier continuation bounces. */
  TextureFromPool hit_throughput_tx_ = {"hit_throughput_tx_"};
  /** Approximate hit material parameters exported by the hardware trace. */
  TextureFromPool hit_material_tx_ = {"hit_material_tx_"};
  /** Geometric hit normal exported by the hardware trace. */
  TextureFromPool hit_normal_tx_ = {"hit_normal_tx_"};
  /** World-space hit position or final miss origin exported by the hardware trace. */
  TextureFromPool hit_position_tx_ = {"hit_position_tx_"};
  /** Exact world-space hit position exported by the hardware trace. */
  TextureFromPool hit_world_position_tx_ = {"hit_world_position_tx_"};
  /** Stable secondary-hit identifiers exported by the hardware trace. */
  TextureFromPool hit_identity_tx_ = {"hit_identity_tx_"};
  /** Secondary-hit barycentric coordinates exported by the hardware trace. */
  TextureFromPool hit_barycentric_tx_ = {"hit_barycentric_tx_"};
  /** Optional later receiver payload for layered scene-final Principled reflection. */
  TextureFromPool layered_receiver_ray_time_tx_ = {"layered_receiver_ray_time_tx_"};
  TextureFromPool layered_receiver_ray_radiance_tx_ = {"layered_receiver_ray_radiance_tx_"};
  TextureFromPool layered_receiver_albedo_tx_ = {"layered_receiver_albedo_tx_"};
  TextureFromPool layered_receiver_throughput_tx_ = {"layered_receiver_throughput_tx_"};
  TextureFromPool layered_receiver_material_tx_ = {"layered_receiver_material_tx_"};
  TextureFromPool layered_receiver_normal_tx_ = {"layered_receiver_normal_tx_"};
  TextureFromPool layered_receiver_position_tx_ = {"layered_receiver_position_tx_"};
  TextureFromPool layered_receiver_world_position_tx_ = {"layered_receiver_world_position_tx_"};
  TextureFromPool layered_receiver_identity_tx_ = {"layered_receiver_identity_tx_"};
  TextureFromPool layered_receiver_barycentric_tx_ = {"layered_receiver_barycentric_tx_"};
  /** Optional later receiver payload for layered scene-final Principled transmission. */
  TextureFromPool transmission_receiver_ray_time_tx_ = {"transmission_receiver_ray_time_tx_"};
  TextureFromPool transmission_receiver_ray_radiance_tx_ = {"transmission_receiver_ray_radiance_tx_"};
  TextureFromPool transmission_receiver_albedo_tx_ = {"transmission_receiver_albedo_tx_"};
  TextureFromPool transmission_receiver_throughput_tx_ = {"transmission_receiver_throughput_tx_"};
  TextureFromPool transmission_receiver_material_tx_ = {"transmission_receiver_material_tx_"};
  TextureFromPool transmission_receiver_normal_tx_ = {"transmission_receiver_normal_tx_"};
  TextureFromPool transmission_receiver_position_tx_ = {"transmission_receiver_position_tx_"};
  TextureFromPool transmission_receiver_world_position_tx_ = {"transmission_receiver_world_position_tx_"};
  TextureFromPool transmission_receiver_identity_tx_ = {"transmission_receiver_identity_tx_"};
  TextureFromPool transmission_receiver_barycentric_tx_ = {"transmission_receiver_barycentric_tx_"};
  draw::StorageArrayBuffer<uint, 64, true> hit_eval_count_buf_ = {"hit_eval_count_buf_"};
  draw::StorageArrayBuffer<uint, 64, true> hit_eval_offset_buf_ = {"hit_eval_offset_buf_"};
  draw::StorageArrayBuffer<uint, 64, true> hit_eval_cursor_buf_ = {"hit_eval_cursor_buf_"};
  draw::StorageArrayBuffer<uint, 64> hit_eval_resource_id_buf_ = {"hit_eval_resource_id_buf_"};
  draw::StorageArrayBuffer<DrawCommand, 16, true> hit_eval_indirect_buf_ = {
      "hit_eval_indirect_buf_"};
  draw::StorageArrayBuffer<HardwareHitEvalRecord, 1024, true> hit_eval_records_buf_ = {
      "hit_eval_records_buf_"};
  draw::Framebuffer hit_eval_fb_ = {"Trace.HitEvalFB"};
  /** Texture containing the horizon local radiance. */
  TextureFromPool horizon_radiance_tx_[4] = {{"horizon_radiance_tx_"}};
  TextureFromPool horizon_radiance_denoised_tx_[4] = {{"horizon_radiance_denoised_tx_"}};
  /** Texture containing the input screen radiance but re-projected. */
  TextureFromPool downsampled_in_radiance_tx_ = {"downsampled_in_radiance_tx_"};
  /** Texture containing the view space normal. The BSDF normal is arbitrarily chosen. */
  TextureFromPool downsampled_in_normal_tx_ = {"downsampled_in_normal_tx_"};
  /** Textures containing the ray hit radiance denoised (full-res). One of them is result_tx. */
  gpu::Texture *denoised_spatial_tx_ = nullptr;
  gpu::Texture *denoised_temporal_tx_ = nullptr;
  gpu::Texture *denoised_bilateral_tx_ = nullptr;
  /** Ray hit depth for temporal denoising. Output of spatial denoise. */
  TextureFromPool hit_depth_tx_ = {"hit_depth_tx_"};
  /** Ray hit variance for temporal denoising. Output of spatial denoise. */
  TextureFromPool hit_variance_tx_ = {"hit_variance_tx_"};
  /** Temporally stable variance for temporal denoising. Output of temporal denoise. */
  TextureFromPool denoise_variance_tx_ = {"denoise_variance_tx_"};
  /** Persistent texture reference for temporal denoising input. */
  gpu::Texture *radiance_history_tx_ = nullptr;
  gpu::Texture *variance_history_tx_ = nullptr;
  gpu::Texture *tilemask_history_tx_ = nullptr;
  gpu::Texture *screen_ownership_history_tx_ = nullptr;
  /** Radiance input for screen space tracing. */
  gpu::Texture *screen_radiance_front_tx_ = nullptr;
  gpu::Texture *screen_radiance_back_tx_ = nullptr;

  Texture radiance_dummy_black_tx_ = {"radiance_dummy_black_tx"};
  Texture hardware_shadow_visibility_tx_ = {"hardware_shadow_visibility_tx_"};
  Texture hardware_direct_light_accum_tx_ = {"hardware_direct_light_accum_tx_"};
  Texture hardware_direct_light_denoised_tx_ = {"hardware_direct_light_denoised_tx_"};
  Texture hardware_direct_light_depth_tx_ = {"hardware_direct_light_depth_tx_"};
  Texture hardware_direct_light_tilemask_tx_ = {"hardware_direct_light_tilemask_tx_"};
  Texture hardware_secondary_shadow_visibility_tx_ = {"hardware_secondary_shadow_visibility_tx_"};
  Texture hardware_layered_receiver_shadow_visibility_tx_ = {
      "hardware_layered_receiver_shadow_visibility_tx_"};
  Texture hardware_transmission_receiver_shadow_visibility_tx_ = {
      "hardware_transmission_receiver_shadow_visibility_tx_"};
  Texture hardware_secondary_environment_visibility_tx_ = {
      "hardware_secondary_environment_visibility_tx_"};
  Texture hardware_environment_visibility_tx_ = {"hardware_environment_visibility_tx_"};
  Texture hardware_caustics_history_tx_ = {"hardware_caustics_history_tx_"};
  Texture hardware_indirect_gi_radiance_cache_tx_ = {"hardware_indirect_gi_radiance_cache_tx_"};
  Texture hardware_indirect_gi_position_cache_tx_ = {"hardware_indirect_gi_position_cache_tx_"};
  Texture hardware_indirect_gi_normal_cache_tx_ = {"hardware_indirect_gi_normal_cache_tx_"};
  Texture hardware_reflected_receiver_gi_tx_ = {"hardware_reflected_receiver_gi_tx_"};
  Texture hardware_reflected_receiver_gi_blur_tx_ = {"hardware_reflected_receiver_gi_blur_tx_"};
  Texture hardware_layered_receiver_gi_tx_ = {"hardware_layered_receiver_gi_tx_"};
  Texture hardware_layered_receiver_gi_blur_tx_ = {"hardware_layered_receiver_gi_blur_tx_"};
  Texture hardware_transmission_receiver_gi_tx_ = {"hardware_transmission_receiver_gi_tx_"};
  Texture hardware_transmission_receiver_gi_blur_tx_ = {
      "hardware_transmission_receiver_gi_blur_tx_"};
  Texture hardware_secondary_photon_gi_tx_ = {"hardware_secondary_photon_gi_tx_"};
  Texture hardware_secondary_photon_gi_blur_tx_ = {"hardware_secondary_photon_gi_blur_tx_"};
  Texture hardware_layered_secondary_photon_gi_tx_ = {"hardware_layered_secondary_photon_gi_tx_"};
  Texture hardware_layered_secondary_photon_gi_blur_tx_ = {
      "hardware_layered_secondary_photon_gi_blur_tx_"};
  Texture hardware_transmission_secondary_photon_gi_tx_ = {
      "hardware_transmission_secondary_photon_gi_tx_"};
  Texture hardware_transmission_secondary_photon_gi_blur_tx_ = {
      "hardware_transmission_secondary_photon_gi_blur_tx_"};
  Texture hardware_fast_gi_tx_ = {"hardware_fast_gi_tx_"};
  Texture hardware_fast_gi_error_tx_ = {"hardware_fast_gi_error_tx_"};
  Texture hardware_fast_gi_visibility_tx_ = {"hardware_fast_gi_visibility_tx_"};
  GPUMetalRaytraceScene *hardware_metal_scene_cache_ = nullptr;
  GPUMetalRaytraceSceneStats hardware_metal_scene_stats_cache_ = {};
  Vector<HardwareRaytraceSceneEntry> hardware_metal_scene_entries_cache_;
  uint64_t hardware_metal_scene_update_count_ = 0;
  bool hardware_metal_scene_update_count_valid_ = false;
  uint64_t hardware_metal_scene_signature_ = 0;
  bool hardware_metal_scene_signature_valid_ = false;
  Vector<HardwareRaytraceSceneEntry> hardware_sorted_scene_entries_cache_;
  uint64_t hardware_sorted_scene_entries_update_count_ = 0;
  bool hardware_sorted_scene_entries_update_count_valid_ = false;
  /** Dummy texture when the tracing is disabled. */
  TextureFromPool dummy_result_tx_ = {"dummy_result_tx"};
  /** Pointer to `inst_.render_buffers.depth_tx` updated before submission. */
  gpu::Texture *renderbuf_depth_view_ = nullptr;

  /** Copy of the scene options to avoid changing parameters during motion blur. */
  RaytraceEEVEE ray_tracing_options_;
  int fast_gi_ray_count_ = 0;
  int fast_gi_step_count_ = 0;
  bool fast_gi_ao_only_ = false;

  bool use_raytracing_ = false;
  bool hardware_gi_enabled_ = false;
  RaytraceEEVEE_GIMode hardware_gi_mode_ = RAYTRACE_EEVEE_GI_MODE_ACCURATE;
  bool hardware_fast_gi_enabled_ = false;
  bool hardware_caustics_enabled_ = false;
  bool hardware_shadow_enabled_ = false;
  bool hardware_lighting_use_hardware_rt_shadows_ = false;
  RaytraceEEVEE_SpecularMode hardware_reflection_mode_ = RAYTRACE_EEVEE_SPECULAR_MODE_OFF;
  RaytraceEEVEE_SpecularMode hardware_refraction_mode_ = RAYTRACE_EEVEE_SPECULAR_MODE_OFF;
  bool hardware_environment_enabled_ = false;
  bool hardware_lighting_use_hardware_rt_environment_visibility_ = false;
  eClosureBits current_trace_active_closures_ = CLOSURE_NONE;
  uint32_t current_hardware_feature_mask_ = 0;
  bool use_hardware_specular_scene_ = false;
  bool use_hardware_hybrid_retrace_ = false;
  bool use_screen_ownership_history_ = false;
  bool hardware_fast_gi_valid_ = false;
  bool hardware_indirect_gi_cache_valid_ = false;
  bool hardware_indirect_gi_cache_rendering_ = false;
  int hardware_indirect_gi_cache_resolution_ = 1;
  int hardware_indirect_gi_cache_face_index_ = 0;
  int3 hardware_indirect_gi_cache_dispatch_size_ = int3(1);
  int hardware_reflected_receiver_gi_resolution_divisor_ = 4;
  Framebuffer hardware_indirect_gi_prepass_fb_ = {"Trace.HardwareIndirectGICachePrepass"};
  Framebuffer hardware_indirect_gi_combined_fb_ = {"Trace.HardwareIndirectGICacheCombined"};
  Framebuffer hardware_indirect_gi_gbuffer_fb_ = {"Trace.HardwareIndirectGICacheGBuffer"};
  RayTraceBuffer hardware_indirect_gi_cache_rt_buffer_[6];
  RayTraceBuffer hardware_indirect_gi_cache_refract_rt_buffer_[6];
  uint64_t hardware_fast_gi_depsgraph_update_count_ = 0;
  bool hardware_fast_gi_depsgraph_update_count_valid_ = false;
  bool hardware_fast_gi_light_invalidation_pending_ = false;
  bool hardware_fast_gi_world_invalidation_pending_ = false;
  bool hardware_fast_gi_emissive_invalidation_pending_ = false;
  bool hardware_fast_gi_material_invalidation_pending_ = false;
  bool hardware_fast_gi_transform_invalidation_pending_ = false;
  bool hardware_fast_gi_geometry_invalidation_pending_ = false;
  bool hardware_fast_gi_animation_invalidation_pending_ = false;
  bool hardware_fast_gi_field_config_valid_ = false;
  bool hardware_fast_gi_memory_limited_ = false;
  int64_t hardware_fast_gi_budget_bytes_ = 0;
  int64_t hardware_fast_gi_requested_bytes_ = 0;
  int64_t hardware_fast_gi_allocated_bytes_ = 0;
  int hardware_fast_gi_requested_grid_resolution_ = 1;
  int hardware_fast_gi_requested_cascade_count_ = 1;
  float hardware_fast_gi_requested_distance_ = 0.0f;
  float3 hardware_fast_gi_field_center_ = float3(0.0f);
  float hardware_fast_gi_scene_radius_ = 0.0f;
  float hardware_fast_gi_scene_density_ = 0.0f;
  float hardware_fast_gi_smoothed_traced_ms_ = 0.0f;
  int hardware_fast_gi_quality_tier_ = 1;
  int hardware_fast_gi_scene_priority_ = 1;
  int hardware_fast_gi_budget_rebalance_ = 1;
  int hardware_debug_view_mode_ = 0;
  int hardware_debug_isolate_mode_ = 0;
  bool hardware_fast_gi_freeze_updates_ = false;
  int hardware_direct_light_sample_count_ = 0;
  float4 hardware_fast_gi_field_cascade_config_[3] = {
      float4(0.0f), float4(0.0f), float4(0.0f)};
  /** Latched during sync before `sampling.step()` clears viewport reset state. */
  bool viewport_history_reset_ = false;
  bool hardware_primary_environment_visibility_ready_ = false;
  gpu::Texture *hardware_primary_environment_visibility_depth_tx_ = nullptr;
  gpu::Texture *hardware_primary_environment_visibility_normal_tx_ = nullptr;
  int2 hardware_primary_environment_visibility_extent_ = int2(0);
  bool hardware_primary_environment_enabled_ = false;
  bool hardware_primary_shadow_visibility_ready_ = false;
  gpu::Texture *hardware_primary_shadow_visibility_depth_tx_ = nullptr;
  gpu::Texture *hardware_primary_shadow_visibility_normal_tx_ = nullptr;
  int2 hardware_primary_shadow_visibility_extent_ = int2(0);
  uint64_t hardware_primary_shadow_visibility_sample_index_ = 0;
  bool hardware_primary_shadow_direct_enabled_ = false;
  bool hardware_primary_shadow_world_enabled_ = false;
  bool hardware_direct_light_dispatch_ready_ = false;

  RaytraceEEVEE_Method tracing_method_ = RAYTRACE_EEVEE_METHOD_PROBE;
  int hardware_scene_entry_count_ = 0;
  int hardware_scene_instance_count_ = 0;

  RayTraceData &data_;

  const Vector<HardwareRaytraceSceneEntry> &current_sorted_hardware_scene_entries(
      uint64_t depsgraph_update_count);
  void invalidate_sorted_hardware_scene_entries_cache();
  void invalidate_viewport_hardware_visibility_cache();

 public:
  RayTraceModule(Instance &inst, RayTraceData &data) : inst_(inst), data_(data) {};
  ~RayTraceModule();

  void init();

  void sync();

  /**
   * RayTrace the scene and resolve radiance buffer for the corresponding `closure_bit`.
   *
   * IMPORTANT: Should not be conditionally executed as it manages the RayTraceResult.
   * IMPORTANT: The screen tracing will be using the front and back Hierarchical-Z Buffer in its
   * current state.
   *
   * \arg rt_buffer is the layer's permanent storage.
   * \arg screen_radiance_back_tx is the texture used for screen space transmission rays.
   * \arg screen_radiance_front_tx is the texture used for screen space reflection rays.
   * \arg screen_radiance_persmat is the view projection matrix used for screen_radiance_front_tx.
   * \arg active_closures is a mask of all active closures in a deferred layer.
   * \arg main_view is the un-jittered view.
   * \arg render_view is the TAA jittered view.
   * \arg force_no_tracing will run the pipeline without any tracing, relying only on local probes.
   */
  RayTraceResult render(RayTraceBuffer &rt_buffer,
                        gpu::Texture *screen_radiance_back_tx,
                        eClosureBits active_closures,
                        /* TODO(fclem): Maybe wrap these two in some other class. */
                        View &main_view,
                        View &render_view);

  RayTraceResult render_phase(RayTraceBuffer &rt_buffer,
                              gpu::Texture *screen_radiance_front_tx,
                              gpu::Texture *screen_radiance_back_tx,
                              eClosureBits active_closures,
                              View &main_view,
                              View &render_view,
                              eHardwareTracePhase trace_phase,
                              uint32_t feature_mask_override,
                              bool enable_horizon_scan);

  void render_scene_final_specular(RayTraceBuffer &rt_buffer,
                                   gpu::Texture *scene_radiance_tx,
                                   eClosureBits active_closures,
                                   View &main_view,
                                   View &render_view);

  /**
   * Only allocate the RayTraceResult results buffers to be used by other passes.
   */
  RayTraceResult alloc_only(RayTraceBuffer &rt_buffer);

  /**
   * Only allocate the RayTraceResult results buffers as dummy texture to ensure correct bindings.
   */
  RayTraceResult alloc_dummy(RayTraceBuffer &rt_buffer);

  void debug_pass_sync();
  void debug_draw(View &view, gpu::FrameBuffer *view_fb);
  void render_directional_shadow_visibility(
      View &render_view, gpu::Texture *depth_tx, gpu::Texture *gbuf_normal_tx, int2 extent);
  void render_environment_visibility(
      View &render_view, gpu::Texture *depth_tx, gpu::Texture *gbuf_normal_tx, int2 extent);
  void render_secondary_environment_visibility(GPUMetalRaytraceScene *metal_scene,
                                               int2 tracing_extent);
  void render_secondary_shadow_visibility(GPUMetalRaytraceScene *metal_scene, int2 tracing_extent);
  void render_hit_shadow_visibility(GPUMetalRaytraceScene *metal_scene,
                                    int2 tracing_extent,
                                    gpu::Texture *hit_normal_tx,
                                    gpu::Texture *hit_world_position_tx,
                                    gpu::Texture *hit_identity_tx,
                                    Texture &shadow_visibility_tx);
  void render_reflected_receiver_gi(GPUMetalRaytraceScene *metal_scene, int2 tracing_extent);
  void render_secondary_photon_gi(GPUMetalRaytraceScene *metal_scene, int2 tracing_extent);
  void render_hardware_indirect_gi_cache(View &main_view);
  void update_hardware_fast_gi_field(
      View &render_view, gpu::Texture *depth_tx, gpu::Texture *input_radiance_tx, int2 extent);
  gpu::Texture **directional_shadow_visibility_tx();
  gpu::Texture **direct_light_accum_tx();
  gpu::Texture **environment_visibility_tx();
  gpu::Texture **caustics_tx();
  gpu::Texture **fast_gi_tx();
  gpu::Texture **fast_gi_visibility_tx();

  bool use_raytracing() const
  {
    return use_raytracing_;
  }

  bool use_fast_gi() const
  {
    return use_horizon_scan(ray_tracing_options_);
  }

  bool use_hardware_tracing_method() const
  {
    return tracing_method_ == RAYTRACE_EEVEE_METHOD_HARDWARE;
  }

  bool use_hardware_shadows() const
  {
    return hardware_shadow_enabled_;
  }

  bool use_hardware_fast_gi() const
  {
    return hardware_fast_gi_enabled_;
  }

  bool use_hardware_direct_light() const
  {
    return use_hardware_tracing_method() && use_hardware_shadows() &&
           hardware_direct_light_denoised_tx_.is_valid();
  }

  bool use_hardware_caustics() const
  {
    return hardware_caustics_enabled_;
  }

  bool use_hardware_gi_refine() const
  {
    return false;
  }

  bool use_hardware_rt_gi() const
  {
    return hardware_gi_enabled_;
  }

  bool use_hardware_reflections() const
  {
    return hardware_reflection_mode_ != RAYTRACE_EEVEE_SPECULAR_MODE_OFF;
  }

  bool use_hardware_refractions() const
  {
    return hardware_refraction_mode_ != RAYTRACE_EEVEE_SPECULAR_MODE_OFF;
  }

  bool use_hardware_environment() const
  {
    return hardware_environment_enabled_;
  }

  uint32_t active_hardware_feature_mask() const
  {
    return (use_hardware_rt_gi() ? RAYTRACE_EEVEE_HARDWARE_GI : 0) |
           (use_hardware_shadows() ? RAYTRACE_EEVEE_HARDWARE_SHADOWS : 0) |
           (use_hardware_reflections() ? RAYTRACE_EEVEE_HARDWARE_REFLECTIONS : 0) |
           (use_hardware_refractions() ? RAYTRACE_EEVEE_HARDWARE_REFRACTIONS : 0) |
           (use_hardware_environment() ? RAYTRACE_EEVEE_HARDWARE_ENVIRONMENT : 0);
  }

  bool has_hardware_fast_gi_field() const
  {
    if (!hardware_fast_gi_enabled_) {
      return false;
    }
    return hardware_fast_gi_valid_;
  }

  bool is_hardware_indirect_gi_cache_rendering() const
  {
    return hardware_indirect_gi_cache_rendering_;
  }

 private:
  bool use_screen_tracing() const
  {
    return tracing_method_ == RAYTRACE_EEVEE_METHOD_SCREEN;
  }

  bool use_hardware_tracing() const
  {
    return use_raytracing_ && tracing_method_ == RAYTRACE_EEVEE_METHOD_HARDWARE &&
           GPU_viewport_hardware_raytracing_support();
  }

  bool use_hardware_gi() const
  {
    return hardware_gi_enabled_;
  }

  bool use_hardware_tracing_method_for_gi() const
  {
    return use_hardware_tracing_method() && use_hardware_rt_gi();
  }

  bool use_horizon_scan(const RaytraceEEVEE &options) const
  {
    return use_raytracing() && !use_hardware_tracing_method() && !use_hardware_gi() &&
           options.trace_max_roughness < 1.0f;
  }

  void warm_tracing_backend();
  void warm_hardware_tracing_backend();

  void submit_tracing_backend(View &render_view);
  void submit_hardware_tracing_backend(View &render_view);
  bool submit_hardware_hit_evaluation_backend(View &render_view);
  void update_hardware_tracing_scene_state();
  void free_hardware_metal_scene_cache();
  GPUMetalRaytraceScene *acquire_hardware_metal_scene(GPUMetalRaytraceSceneStats *r_stats,
                                                      bool require_current_feature_mask = true);

  RayTraceResultTexture trace(int closure_index,
                              bool active_layer,
                              RaytraceEEVEE options,
                              RayTraceBuffer &rt_buffer,
                              /* TODO(fclem): Maybe wrap these two in some other class. */
                              View &main_view,
                              View &render_view);
};

/** \} */

}  // namespace blender::eevee
