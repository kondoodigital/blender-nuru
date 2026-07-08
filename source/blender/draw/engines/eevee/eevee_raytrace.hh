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
#include "GPU_nuru_hardware_raytrace.hh"

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
  /* Last valid shared OIDN result, reused on interval-skipped samples. */
  TextureFromPool shared_indirect_oidn_history_tx = {"shared_indirect_oidn_history_tx"};
  bool valid_shared_indirect_oidn_history = false;
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
    if (result_ == nullptr) {
      /* Default-constructed (e.g. a held result that was never produced). */
      return;
    }
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
  RayTraceResultTexture shared_indirect;
  bool use_shared_indirect = false;

  void release()
  {
    for (int i = 0; i < 3; i++) {
      closures[i].release();
    }
    if (use_shared_indirect) {
      shared_indirect.release();
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
  draw::PassSimple scene_final_specular_resolve_ps_ = {"Trace.SceneFinalSpecularResolve"};
  draw::PassSimple hit_eval_count_ps_ = {"Trace.HitEvalCount"};
  draw::PassSimple hit_eval_prefix_ps_ = {"Trace.HitEvalPrefix"};
  draw::PassSimple hit_eval_compact_ps_ = {"Trace.HitEvalCompact"};
  draw::PassSimple hit_eval_ps_ = {"Trace.HitEval"};
  draw::PassSimple shared_indirect_accum_ps_ = {"Trace.SharedIndirectAccum"};
  draw::PassSimple shared_indirect_reconstruct_ps_ = {"Trace.SharedIndirectReconstruct"};
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
  int3 shared_indirect_reconstruct_dispatch_size_ = int3(1);
  int shared_indirect_active_closure_count_ = 0;
  /** False when the hardware trace kernel did not run in the current `trace()` phase: the OIDN
   * albedo/normal guide textures then hold stale pool memory and OIDN must run RGB-only. */
  bool use_oidn_guides_ = true;
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
  /** Exact allocation size (in tiles) of the two buffers above; GPU-side bound for all
   * stores/reads indexed by the compacted tile count. */
  int hardware_direct_light_tile_capacity_ = 1;
  draw::StorageArrayBuffer<GPUHardwareRaytraceFastGILightRecord, 256> hardware_fast_gi_light_buf_ = {
      "hardware_fast_gi_light_buf_"};
  /** Light record + tree buffer cache: lights only change with a depsgraph update, so the CPU
   * tree build + GPU upload run once per update instead of once per trace call. */
  uint64_t hardware_light_records_update_count_ = 0;
  bool hardware_light_records_update_count_valid_ = false;
  int hardware_light_records_light_count_ = 0;
  int hardware_light_records_local_light_count_ = 0;
  /** Nuru NIS: learned per-cluster sampling multipliers (ones until the network trains). */
  draw::StorageArrayBuffer<float, 32> hardware_light_cluster_weight_buf_ = {
      "hardware_light_cluster_weight_buf_"};
  bool hardware_light_cluster_weights_initialized_ = false;
  /** Nuru NIS stage N2: tiny-MLP parameters, gradient accumulator (float bits in uint for CAS
   * atomics), trained-sample counter + step counter, and Adam moments. */
  draw::StorageArrayBuffer<float, 4> hardware_nis_weights_buf_ = {"hardware_nis_weights_buf_"};
  draw::StorageArrayBuffer<int, 4> hardware_nis_grads_buf_ = {"hardware_nis_grads_buf_"};
  draw::StorageArrayBuffer<uint, 4> hardware_nis_train_count_buf_ = {
      "hardware_nis_train_count_buf_"};
  draw::StorageArrayBuffer<float, 4> hardware_nis_adam_m_buf_ = {"hardware_nis_adam_m_buf_"};
  draw::StorageArrayBuffer<float, 4> hardware_nis_adam_v_buf_ = {"hardware_nis_adam_v_buf_"};
  bool hardware_nis_initialized_ = false;
  int hardware_nis_train_stride_ = 4;
  int hardware_nis_train_offset_ = 0;
  int hardware_nis_param_count_ = 0;
  /** Nuru NIS G3: receiver feedback ring ([0] counter + 4096 entries x 8 words). */
  draw::StorageArrayBuffer<uint, 8 + 4096 * 8> hardware_nis_feedback_buf_ = {
      "hardware_nis_feedback_buf_"};
  draw::PassSimple hardware_nis_train_ps_ = {"Trace.NISTrain"};
  draw::PassSimple hardware_nis_train_feedback_ps_ = {"Trace.NISTrainFeedback"};
  draw::PassSimple hardware_nis_adam_ps_ = {"Trace.NISAdam"};
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
  /** OIDN-cleaned raw ray hit radiance before Eevee spatial/temporal reuse. */
  TextureFromPool ray_radiance_oidn_tx_ = {"ray_radiance_oidn_tx"};
  /** Ray radiance image consumed by the spatial resolve pass. */
  gpu::Texture *ray_radiance_denoise_source_tx_ = nullptr;
  /** Shared OIDN indirect path: one accumulated ray-grid radiance and matching guides. */
  TextureFromPool shared_indirect_radiance_tx_ = {"shared_indirect_radiance_tx"};
  TextureFromPool shared_indirect_oidn_tx_ = {"shared_indirect_oidn_tx"};
  TextureFromPool shared_indirect_albedo_tx_ = {"shared_indirect_albedo_tx"};
  TextureFromPool shared_indirect_normal_tx_ = {"shared_indirect_normal_tx"};
  TextureFromPool shared_indirect_reconstructed_tx_ = {"shared_indirect_reconstructed_tx"};
  gpu::Texture *shared_indirect_reconstruct_source_tx_ = nullptr;
  bool shared_indirect_reconstruct_source_is_oidn_ = false;
  /** Hybrid screen-hit continuation origin/time for bounce 2+ handoff into Hardware RT. */
  TextureFromPool screen_continuation_tx_ = {"screen_continuation_tx_"};
  /** Current-frame Hybrid screen ownership written by the screen trace pass. */
  TextureFromPool screen_ownership_tx_ = {"screen_ownership_tx_"};
  /** Approximate hit albedo exported by the hardware trace. */
  TextureFromPool hit_albedo_tx_ = {"hit_albedo_tx_"};
  /** Nuru Secondary GI: per-pixel receiver GI for mirror-visible diffuse surfaces. */
  TextureFromPool reflected_receiver_gi_tx_ = {"reflected_receiver_gi_tx_"};
  TextureFromPool layered_receiver_gi_tx_ = {"layered_receiver_gi_tx_"};
  TextureFromPool transmission_receiver_gi_tx_ = {"transmission_receiver_gi_tx_"};
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
  /** Cache key for `hit_eval_resource_id_buf_`: the table only changes with the sorted scene
   * entries, not per closure/phase call. */
  uint64_t hit_eval_resource_ids_update_count_ = 0;
  bool hit_eval_resource_ids_update_count_valid_ = false;
  int hit_eval_resource_ids_entry_count_ = 0;
  draw::StorageArrayBuffer<DrawCommand, 16, true> hit_eval_indirect_buf_ = {
      "hit_eval_indirect_buf_"};
  draw::StorageArrayBuffer<HardwareHitEvalRecord, 1024, true> hit_eval_records_buf_ = {
      "hit_eval_records_buf_"};
  /** Bound to material-attribute SSBO slots the draw batch cannot fulfill. Never read by the
   * shader (the zeroed `gpu_attr_N_meta` short-circuits every fetch) but required so the slot has
   * a valid buffer binding. */
  draw::StorageArrayBuffer<uint4, 1, true> hit_eval_dummy_attr_buf_ = {
      "hit_eval_dummy_attr_buf_"};
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
  GPUHardwareRaytraceScene *hardware_rt_scene_cache_ = nullptr;
  GPUHardwareRaytraceSceneStats hardware_rt_scene_stats_cache_ = {};
  Vector<HardwareRaytraceSceneEntry> hardware_rt_scene_entries_cache_;
  uint64_t hardware_rt_scene_update_count_ = 0;
  bool hardware_rt_scene_update_count_valid_ = false;
  uint64_t hardware_rt_scene_signature_ = 0;
  bool hardware_rt_scene_signature_valid_ = false;
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
  /** Nuru: user toggle for Secondary GI (mirror receiver GI). */
  bool hardware_secondary_gi_enabled_ = false;
  bool hardware_shadow_enabled_ = false;
  /* Nuru: latched UI slider `Color Transmission` (0..1) for tinted shadow saturation. */
  float hardware_shadow_color_intensity_ = 0.5f;
  /* Nuru: latched UI slider `Transparent Shadows` (0 opaque .. 1 transparent). */
  float hardware_shadow_transparency_ = 1.0f;
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
  int hardware_fast_gi_quality_tier_ = 1;
  int hardware_debug_view_mode_ = 0;
  int hardware_debug_isolate_mode_ = 0;
  int hardware_direct_light_sample_count_ = 0;
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
  int hardware_primary_shadow_visibility_sample_count_ = 1;
  bool hardware_primary_shadow_direct_enabled_ = false;
  bool hardware_primary_shadow_world_enabled_ = false;
  float hardware_primary_shadow_color_intensity_ = 0.5f;
  int hardware_shadow_sample_count_ = 1;
  bool hardware_direct_light_dispatch_ready_ = false;
  int2 hardware_direct_light_dispatch_extent_ = int2(0);
  float4x4 hardware_direct_light_dispatch_viewinv_ = float4x4::zero();
  float4x4 hardware_direct_light_dispatch_wininv_ = float4x4::zero();

  RaytraceEEVEE_Method tracing_method_ = RAYTRACE_EEVEE_METHOD_PROBE;
  int hardware_scene_entry_count_ = 0;
  int hardware_scene_instance_count_ = 0;

  RayTraceData &data_;

  const Vector<HardwareRaytraceSceneEntry> &current_sorted_hardware_scene_entries(
      uint64_t depsgraph_update_count);
  void invalidate_sorted_hardware_scene_entries_cache();
  void invalidate_viewport_hardware_visibility_cache();

  /** Hard-disable Nuru HWRT for shader/material preview renders. */
  void apply_shader_preview_nuru_disable();

  /** Shader/material preview must never run Nuru HWRT (see `BKE_scene_is_eevee_shader_preview`). */
  bool shader_preview_disable_nuru_ = false;

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
  void render_secondary_environment_visibility(GPUHardwareRaytraceScene *rt_scene,
                                               int2 tracing_extent);
  void render_secondary_shadow_visibility(GPUHardwareRaytraceScene *rt_scene, int2 tracing_extent);
  void render_hit_shadow_visibility(GPUHardwareRaytraceScene *rt_scene,
                                    int2 tracing_extent,
                                    gpu::Texture *hit_normal_tx,
                                    gpu::Texture *hit_world_position_tx,
                                    gpu::Texture *hit_identity_tx,
                                    Texture &shadow_visibility_tx);
  gpu::Texture **directional_shadow_visibility_tx();
  gpu::Texture **direct_light_accum_tx();
  gpu::Texture **environment_visibility_tx();
  gpu::Texture **caustics_tx();

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
    return !shader_preview_disable_nuru_ && tracing_method_ == RAYTRACE_EEVEE_METHOD_HARDWARE;
  }

  bool is_shader_preview_nuru_disabled() const
  {
    return shader_preview_disable_nuru_;
  }

  bool use_hardware_shadows() const
  {
    return hardware_shadow_enabled_;
  }

  float hardware_shadow_color_intensity() const
  {
    return hardware_shadow_color_intensity_;
  }

  float hardware_shadow_transparency() const
  {
    return hardware_shadow_transparency_;
  }

  /* Nuru: Secondary GI (mirror-interior receiver GI), stage N3 of the NIS workstream. The
   * per-pixel receiver dome kernels (tree NEE + transparent punch-through, RUBY 24
   * calibration) light diffuse surfaces seen through scene-final reflections/refractions.
   * Cache-free by construction: every term is traced per pixel per frame. */
  bool use_hardware_fast_gi_secondary() const
  {
    return hardware_secondary_gi_enabled_ && use_hardware_tracing() &&
           ((hardware_reflection_mode_ != RAYTRACE_EEVEE_SPECULAR_MODE_OFF) ||
            (hardware_refraction_mode_ != RAYTRACE_EEVEE_SPECULAR_MODE_OFF));
  }

  bool use_hardware_direct_light() const
  {
    return use_hardware_tracing_method() && use_hardware_shadows() &&
           hardware_direct_light_denoised_tx_.is_valid();
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
           (use_hardware_refractions() ? RAYTRACE_EEVEE_HARDWARE_REFRACTIONS : 0);
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
  void free_hardware_rt_scene_cache();
  GPUHardwareRaytraceScene *acquire_hardware_rt_scene(GPUHardwareRaytraceSceneStats *r_stats,
                                                      bool require_current_feature_mask = true);

  RayTraceResultTexture trace(int closure_index,
                              bool active_layer,
                              RaytraceEEVEE options,
                              RayTraceBuffer &rt_buffer,
                              /* TODO(fclem): Maybe wrap these two in some other class. */
                              View &main_view,
                              View &render_view);
  bool use_shared_oidn_denoise(const RaytraceEEVEE &options) const;
  bool use_shared_oidn_this_sample(const RaytraceEEVEE &options) const;
  void trace_shared_oidn_closure(int closure_index,
                                 RaytraceEEVEE options,
                                 RayTraceBuffer &rt_buffer,
                                 View &main_view,
                                 View &render_view);
  RayTraceResult render_shared_oidn(RayTraceBuffer &rt_buffer,
                                    const int active_closure_count,
                                    RaytraceEEVEE options,
                                    View &main_view,
                                    View &render_view);
};

/** \} */

}  // namespace blender::eevee
