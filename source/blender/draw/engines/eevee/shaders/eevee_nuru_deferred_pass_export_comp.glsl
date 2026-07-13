/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/**
 * Post-combine pass export.
 *
 * Reconstructs the EEVEE COLOR-storage viewport render passes (Normal, Position,
 * Diffuse Color/Light, Specular Color/Light) into `rp_color_tx` *from compute*.
 *
 * The deferred combine fragment shader's `output_renderpass_color` writes to `rp_color_tx`
 * do not reliably reach the array texture under the Nuru pipeline on Metal, so the per-pass
 * data path stays gray when a user selects one of those passes in the viewport. Compute
 * shader image stores into `rp_color_tx` *do* reach, so the per-pass values are recomputed
 * here using the same closure/radiance inputs as the combine pass and written from a compute
 * dispatch after the combine has run.
 *
 * The shader is gated by C++ on `DeferredLayer::any_pass_export_enabled_`; when no
 * COLOR-storage pass is requested (typical Combined-only viewport) the dispatch is skipped
 * entirely, and even when it runs the per-pass `imageStore` calls are gated by the
 * specialization-constant pass IDs so unused slots cost nothing.
 *
 * Diffuse Light receives `diffuse_direct + diffuse_indirect` where `diffuse_indirect`
 * picks up the Nuru OIDN-denoised shared diffuse GI (via `shared_indirect_tx`) when
 * `use_shared_indirect` is set, matching what combine writes into the Combined pass.
 *
 * Specular Light receives `specular_direct + specular_indirect`. When
 * `defer_hardware_specular_indirect` is set the indirect term is intentionally omitted
 * here so the existing `scene_final_specular` pass can add the HWRT reflection/refraction
 * contribution directly into the same `specular_light_id` slot.
 */

#include "infos/eevee_deferred_infos.hh"

COMPUTE_SHADER_CREATE_INFO(eevee_deferred_pass_export)

#include "draw_view_lib.glsl"
#include "eevee_colorspace_lib.glsl"
#include "eevee_gbuffer_read_lib.glsl"
#include "gpu_shader_math_vector_reduce_lib.glsl"
#include "gpu_shader_shared_exponent_lib.glsl"

float3 load_radiance_direct(int2 texel, uchar i)
{
  uint data = 0u;
  switch (i) {
    case 0:
      data = texelFetch(direct_radiance_1_tx, texel, 0).r;
      break;
    case 1:
      data = texelFetch(direct_radiance_2_tx, texel, 0).r;
      break;
    case 2:
      data = texelFetch(direct_radiance_3_tx, texel, 0).r;
      break;
    default:
      break;
  }
  return rgb9e5_decode(data);
}

float3 load_radiance_indirect(int2 texel, uchar i)
{
  switch (i) {
    case 0:
      return texelFetch(indirect_radiance_1_tx, texel, 0).rgb;
    case 1:
      return texelFetch(indirect_radiance_2_tx, texel, 0).rgb;
    case 2:
      return texelFetch(indirect_radiance_3_tx, texel, 0).rgb;
    default:
      return float3(0.0f);
  }
}

void main()
{
  int2 texel = int2(gl_GlobalInvocationID.xy);
  if (any(greaterThanEqual(texel, imageSize(rp_color_img).xy))) {
    return;
  }

  /* Cheap depth early-out for background pixels. `hiz_tx` LOD 0 stores the regular window-
   * space depth (1.0 at the far plane / unfilled background), matching what
   * `eevee_deferred_light_frag.glsl` and `eevee_deferred_combine_frag.glsl` consume for the
   * position pass. */
  float window_depth = texelFetch(hiz_tx, texel, 0).r;
  if (window_depth >= 1.0f) {
    return;
  }

  const gbuffer::Layers gbuf = gbuffer::read_layers(texel);
  if (gbuf.header.is_shadow_catcher()) {
    return;
  }
  const uchar closure_count = gbuf.header.closure_len();
  const uint3 bin_indices = gbuf.header.bin_index_per_layer();

  float3 diffuse_color = float3(0.0f);
  float3 diffuse_direct = float3(0.0f);
  float3 diffuse_indirect = float3(0.0f);
  float3 specular_color = float3(0.0f);
  float3 specular_direct = float3(0.0f);
  float3 specular_indirect = float3(0.0f);
  float3 average_normal = float3(0.0f);

  for (uchar i = 0; i < GBUFFER_LAYER_MAX && i < closure_count; i++) {
    ClosureUndetermined cl = gbuf.layer_get(i);
    if (cl.type == CLOSURE_NONE_ID) {
      continue;
    }
    uchar layer_index = bin_indices[i];
    float3 closure_direct_light = load_radiance_direct(texel, layer_index);
    float3 closure_indirect_light = float3(0.0f);

    if (use_split_radiance && !use_shared_indirect) {
      closure_indirect_light = load_radiance_indirect(texel, layer_index);
    }

    average_normal += cl.N * reduce_add(cl.color);

    switch (cl.type) {
      case CLOSURE_BSDF_TRANSLUCENT_ID:
      case CLOSURE_BSSRDF_BURLEY_ID:
      case CLOSURE_BSDF_DIFFUSE_ID:
        diffuse_color += cl.color;
        diffuse_direct += closure_direct_light;
        diffuse_indirect += closure_indirect_light;
        break;
      case CLOSURE_BSDF_MICROFACET_GGX_REFLECTION_ID:
      case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
        specular_color += cl.color;
        specular_direct += closure_direct_light;
        if (!defer_hardware_specular_indirect) {
          specular_indirect += closure_indirect_light;
        }
        break;
      case CLOSURE_NONE_ID:
        break;
    }
  }

  if (use_shared_indirect) {
    /* OIDN-denoised shared diffuse GI from the Nuru raytracing path. */
    float3 shared_indirect = texelFetch(shared_indirect_tx, texel, 0).rgb;
    diffuse_indirect += shared_indirect;
  }

  /* Match combine's clamping + scaling so the per-pass values are in the same space as
   * what gets added into `Combined`. */
  float clamp_direct = uniform_buf.clamp.surface_direct;
  float clamp_indirect = uniform_buf.clamp.surface_indirect;
  diffuse_direct = colorspace_brightness_clamp_max(diffuse_direct, clamp_direct);
  diffuse_indirect = colorspace_brightness_clamp_max(diffuse_indirect, clamp_indirect);
  specular_direct = colorspace_brightness_clamp_max(specular_direct, clamp_direct);
  specular_indirect = colorspace_brightness_clamp_max(specular_indirect, clamp_indirect);

  diffuse_direct *= uniform_buf.clamp.direct_scale;
  diffuse_indirect *= uniform_buf.clamp.indirect_scale;
  specular_direct *= uniform_buf.clamp.direct_scale;
  specular_indirect *= uniform_buf.clamp.indirect_scale;

  /* Per-pass writes. The pass IDs are specialization constants so unused slots compile
   * away entirely. `imageStore` from compute is what reaches `rp_color_tx` under Nuru on
   * Metal (fragment writes from combine do not), which is why this pass exists at all. */
  if (pass_diffuse_color_id != -1) {
    imageStore(rp_color_img, int3(texel, pass_diffuse_color_id), float4(diffuse_color, 1.0f));
  }
  if (pass_diffuse_light_id != -1) {
    float3 diffuse_light = diffuse_direct + diffuse_indirect;
    imageStore(rp_color_img, int3(texel, pass_diffuse_light_id), float4(diffuse_light, 1.0f));
  }
  if (pass_specular_color_id != -1) {
    imageStore(
        rp_color_img, int3(texel, pass_specular_color_id), float4(specular_color, 1.0f));
  }
  if (pass_specular_light_id != -1) {
    float3 specular_light = specular_direct + specular_indirect;
    imageStore(
        rp_color_img, int3(texel, pass_specular_light_id), float4(specular_light, 1.0f));
  }
  if (pass_normal_id != -1) {
    float normal_len = length(average_normal);
    /* Normalize or fall back to the gbuffer surface normal, same as combine does. */
    average_normal = (normal_len < 1e-5f) ? gbuf.surface_N() : (average_normal / normal_len);
    imageStore(rp_color_img, int3(texel, pass_normal_id), float4(average_normal, 1.0f));
  }
  if (pass_position_id != -1) {
    /* Match combine fragment: `screen_uv` there is the smooth interpolant from the
     * fullscreen vertex shader which equals `gl_FragCoord.xy / framebuffer_size`. From
     * compute that is `(texel + 0.5) / hiz_size`. Using `hiz_tx`'s size keeps it aligned
     * with the depth source. */
    float2 uv = (float2(texel) + 0.5f) / float2(textureSize(hiz_tx, 0).xy);
    float3 P = drw_point_screen_to_world(float3(uv, window_depth));
    imageStore(rp_color_img, int3(texel, pass_position_id), float4(P, 1.0f));
  }
}
