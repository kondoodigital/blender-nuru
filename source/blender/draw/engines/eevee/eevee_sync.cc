/* SPDX-FileCopyrightText: 2021 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup eevee
 *
 * Converts the different renderable object types to draw-calls.
 */

#include "BKE_compute_context_cache.hh"
#include "BKE_node.hh"
#include "BKE_node_legacy_types.hh"
#include "BKE_object.hh"
#include "BKE_paint.hh"
#include "BKE_paint_bvh.hh"
#include "BLI_hash.hh"
#include "DNA_node_types.h"
#include "DNA_curves_types.h"
#include "DNA_modifier_types.h"
#include "DNA_particle_types.h"
#include "DNA_pointcloud_types.h"
#include "DNA_volume_types.h"

#include "GPU_capabilities.hh"
#include "GPU_context.hh"

#include "draw_cache.hh"
#include "draw_common.hh"
#include "draw_sculpt.hh"

#include "NOD_shader.h"
#include "NOD_socket_value_inference.hh"

#include "eevee_instance.hh"

#include <algorithm>
#include <limits>

namespace blender::eevee {

enum HardwareMaterialProxyClosureType : uint32_t {
  HWRT_CLOSURE_DIFFUSE = 1u,
  HWRT_CLOSURE_TRANSLUCENT = 6u,
  HWRT_CLOSURE_REFLECTION = 7u,
  HWRT_CLOSURE_REFRACTION = 12u,
};

enum HardwareMaterialProxyFlags : uint32_t {
  HWRT_PROXY_FLAG_DIELECTRIC_REFLECTION = 1u << 0,
  HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL = 1u << 1,
  HWRT_PROXY_FLAG_ALPHA_BLEND_TRANSPARENT = 1u << 2,
  HWRT_PROXY_FLAG_PRINCIPLED_TRANSMISSION_LAYER = 1u << 3,
  HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL = 1u << 4,
};

/* Bounded RT material policy:
 * - direct-light and shadow rays never evaluate hit materials,
 * - traversal / continuation run on this cheap sync-time proxy only,
 * - sparse MAT_PIPE_HIT_EVAL replay is reserved for compacted resolved hits,
 * - indirect diffuse GI only consumes emissive + diffuse-albedo data, while direct/specular
 *   fallback keeps one dominant closure family with tint/roughness/IOR. */
struct HardwareMaterialProxy {
  float3 reflection_color = float3(0.8f);
  float reflection_roughness = 1.0f;
  float3 transmission_color = float3(0.8f);
  float transmission_roughness = 1.0f;
  float reflection_ior = 1.45f;
  float refraction_ior = 1.45f;
  float packed_thickness = 0.0f;
  float alpha = 1.0f;
  float reflection_layer_coverage = 0.0f;
  uint32_t closure_type = HWRT_CLOSURE_DIFFUSE;
  uint32_t flags = 0u;
};

static float3 socket_color_value(const bNodeSocket *socket)
{
  if (socket == nullptr || socket->default_value == nullptr) {
    return float3(0.0f);
  }
  const float *rgba = static_cast<const bNodeSocketValueRGBA *>(socket->default_value)->value;
  return float3(rgba[0], rgba[1], rgba[2]);
}

static float socket_float_value(const bNodeSocket *socket)
{
  if (socket == nullptr || socket->default_value == nullptr) {
    return 0.0f;
  }
  return static_cast<const bNodeSocketValueFloat *>(socket->default_value)->value;
}

static float3 inferred_socket_color_value(const bNodeSocket *socket,
                                          nodes::SocketValueInferencer &inferencer)
{
  if (socket == nullptr) {
    return float3(0.0f);
  }
  if (const std::optional<float3> value = inferencer.get_socket_value({nullptr, socket})
                                                .get_if_primitive<float3>())
  {
    return *value;
  }
  return socket_color_value(socket);
}

static bool inferred_socket_color_is_dynamic(const bNodeSocket *socket,
                                             nodes::SocketValueInferencer &inferencer)
{
  if (socket == nullptr) {
    return false;
  }
  if (socket->link != nullptr) {
    return true;
  }
  return !inferencer.get_socket_value({nullptr, socket}).get_if_primitive<float3>().has_value();
}

static float inferred_socket_float_value(const bNodeSocket *socket,
                                         nodes::SocketValueInferencer &inferencer)
{
  if (socket == nullptr) {
    return 0.0f;
  }
  if (const std::optional<float> value = inferencer.get_socket_value({nullptr, socket})
                                               .get_if_primitive<float>())
  {
    return *value;
  }
  return socket_float_value(socket);
}

static float proxy_thickness_pack(const float thickness, const bool slab_mode)
{
  const float clamped_thickness = std::max(0.0f, thickness);
  const float thickness_packed = clamped_thickness / (1.0f + 2.0f * clamped_thickness);
  return slab_mode ? 1.0f - thickness_packed : thickness_packed;
}

static float material_default_thickness(const Object *ob)
{
  if (ob == nullptr) {
    return 0.0f;
  }
  float dimensions[3];
  BKE_object_dimensions_get(ob, dimensions);
  return std::max(0.0f, math::reduce_min(float3(dimensions[0], dimensions[1], dimensions[2])));
}

static float material_output_thickness(const blender::Material *blender_mat,
                                       nodes::SocketValueInferencer &inferencer)
{
  if (blender_mat == nullptr || blender_mat->nodetree == nullptr) {
    return -1.0f;
  }
  const bNode *output_node = ntreeShaderOutputNode(
      const_cast<bNodeTree *>(blender_mat->nodetree), SHD_OUTPUT_EEVEE);
  if (output_node == nullptr) {
    return -1.0f;
  }
  const bNodeSocket *thickness_socket = bke::node_find_socket(*output_node, SOCK_IN, "Thickness");
  if (thickness_socket == nullptr) {
    return -1.0f;
  }
  if (const std::optional<float> value = inferencer.get_socket_value({nullptr, thickness_socket})
                                             .get_if_primitive<float>())
  {
    return std::max(0.0f, *value);
  }
  if (thickness_socket->link == nullptr) {
    return std::max(0.0f, socket_float_value(thickness_socket));
  }
  return -1.0f;
}

static float3 material_surface_emissive_radiance(const bNodeSocket *shader_socket,
                                                 nodes::SocketValueInferencer &inferencer,
                                                 ResourceScope &scope,
                                                 bke::ComputeContextCache &compute_context_cache,
                                                 int depth);

static float3 material_shader_node_emissive_radiance(const bNode &node,
                                                     int output_index,
                                                     nodes::SocketValueInferencer &inferencer,
                                                     ResourceScope &scope,
                                                     bke::ComputeContextCache &compute_context_cache,
                                                     int depth)
{
  if (depth > 16) {
    return float3(0.0f);
  }

  switch (node.type_legacy) {
    case SH_NODE_EMISSION: {
      const bNodeSocket *color_socket = bke::node_find_socket(node, SOCK_IN, "Color");
      const bNodeSocket *strength_socket = bke::node_find_socket(node, SOCK_IN, "Strength");
      return inferred_socket_color_value(color_socket, inferencer) *
             std::max(0.0f, inferred_socket_float_value(strength_socket, inferencer));
    }
    case SH_NODE_BSDF_PRINCIPLED: {
      const bNodeSocket *color_socket = bke::node_find_socket(node, SOCK_IN, "Emission Color");
      const bNodeSocket *strength_socket = bke::node_find_socket(node, SOCK_IN, "Emission Strength");
      return inferred_socket_color_value(color_socket, inferencer) *
             std::max(0.0f, inferred_socket_float_value(strength_socket, inferencer));
    }
    case SH_NODE_ADD_SHADER: {
      const Span<const bNodeSocket *> inputs = node.input_sockets();
      if (inputs.size() < 2) {
        return float3(0.0f);
      }
      return material_surface_emissive_radiance(
                 inputs[0], inferencer, scope, compute_context_cache, depth + 1) +
             material_surface_emissive_radiance(
                 inputs[1], inferencer, scope, compute_context_cache, depth + 1);
    }
    case SH_NODE_MIX_SHADER: {
      const Span<const bNodeSocket *> inputs = node.input_sockets();
      if (inputs.size() < 3) {
        return float3(0.0f);
      }
      const float fac = math::clamp(
          inferred_socket_float_value(inputs[0], inferencer), 0.0f, 1.0f);
      const float3 a = material_surface_emissive_radiance(
          inputs[1], inferencer, scope, compute_context_cache, depth + 1);
      const float3 b = material_surface_emissive_radiance(
          inputs[2], inferencer, scope, compute_context_cache, depth + 1);
      return a * (1.0f - fac) + b * fac;
    }
    case NODE_REROUTE: {
      const Span<const bNodeSocket *> inputs = node.input_sockets();
      return inputs.is_empty() ?
                 float3(0.0f) :
                 material_surface_emissive_radiance(
                     inputs[0], inferencer, scope, compute_context_cache, depth + 1);
    }
    case NODE_GROUP:
    case NODE_CUSTOM_GROUP: {
      const bNodeTree *group = reinterpret_cast<const bNodeTree *>(node.id);
      if (group == nullptr) {
        return float3(0.0f);
      }
      group->ensure_topology_cache();
      group->ensure_interface_cache();
      const bNode *group_output_node = group->group_output_node();
      if (group_output_node == nullptr) {
        return float3(0.0f);
      }
      const Span<const bNodeSocket *> group_inputs = node.input_sockets();
      const Span<const bNodeSocket *> group_outputs = group_output_node->input_sockets();
      if (output_index < 0 || output_index >= group_outputs.size()) {
        return float3(0.0f);
      }
      auto get_group_input_value = [&](const int group_input_i) -> nodes::InferenceValue {
        if (group_input_i < 0 || group_input_i >= group_inputs.size()) {
          return nodes::InferenceValue::Unknown();
        }
        return inferencer.get_socket_value({nullptr, group_inputs[group_input_i]});
      };
      nodes::SocketValueInferencer group_inferencer{
          *group, scope, compute_context_cache, get_group_input_value};
      return material_surface_emissive_radiance(
          group_outputs[output_index], group_inferencer, scope, compute_context_cache, depth + 1);
    }
    default:
      return float3(0.0f);
  }
}

static float3 material_surface_emissive_radiance(const bNodeSocket *shader_socket,
                                                 nodes::SocketValueInferencer &inferencer,
                                                 ResourceScope &scope,
                                                 bke::ComputeContextCache &compute_context_cache,
                                                 int depth)
{
  if (depth > 16 || shader_socket == nullptr || shader_socket->link == nullptr) {
    return float3(0.0f);
  }
  const bNodeLink *link = shader_socket->link;
  return material_shader_node_emissive_radiance(
      *link->fromnode, link->fromsock->index(), inferencer, scope, compute_context_cache, depth);
}

static float3 material_emissive_radiance(const blender::Material *blender_mat)
{
  if (blender_mat == nullptr || blender_mat->nodetree == nullptr) {
    return float3(0.0f);
  }
  blender_mat->nodetree->ensure_topology_cache();
  blender_mat->nodetree->ensure_interface_cache();
  const bNode *output_node = ntreeShaderOutputNode(
      const_cast<bNodeTree *>(blender_mat->nodetree), SHD_OUTPUT_EEVEE);
  if (output_node == nullptr) {
    return float3(0.0f);
  }
  const bNodeSocket *surface_socket = bke::node_find_socket(*output_node, SOCK_IN, "Surface");
  if (surface_socket == nullptr) {
    return float3(0.0f);
  }

  ResourceScope scope;
  bke::ComputeContextCache compute_context_cache;
  nodes::SocketValueInferencer inferencer{
      *blender_mat->nodetree, scope, compute_context_cache};
  return material_surface_emissive_radiance(
      surface_socket, inferencer, scope, compute_context_cache, 0);
}

static float3 material_diffuse_albedo(const blender::Material *blender_mat)
{
  if (blender_mat == nullptr || blender_mat->nodetree == nullptr) {
    return float3(0.8f);
  }

  blender_mat->nodetree->ensure_topology_cache();
  blender_mat->nodetree->ensure_interface_cache();
  ResourceScope scope;
  bke::ComputeContextCache compute_context_cache;
  nodes::SocketValueInferencer inferencer{
      *blender_mat->nodetree, scope, compute_context_cache};
  float3 albedo(0.8f);
  for (const bNode *node = static_cast<const bNode *>(blender_mat->nodetree->nodes.first);
       node != nullptr;
       node = node->next)
  {
    switch (node->type_legacy) {
      case SH_NODE_BSDF_PRINCIPLED: {
        const bNodeSocket *color_socket = bke::node_find_socket(*node, SOCK_IN, "Base Color");
        return inferred_socket_color_value(color_socket, inferencer);
      }
      case SH_NODE_BSDF_DIFFUSE: {
        const bNodeSocket *color_socket = bke::node_find_socket(*node, SOCK_IN, "Color");
        return inferred_socket_color_value(color_socket, inferencer);
      }
      default:
        break;
    }
  }

  return albedo;
}

static HardwareMaterialProxy material_hardware_proxy(const blender::Material *blender_mat,
                                                     const Object *ob)
{
  HardwareMaterialProxy proxy;
  if (blender_mat == nullptr || blender_mat->nodetree == nullptr) {
    return proxy;
  }

  blender_mat->nodetree->ensure_topology_cache();
  blender_mat->nodetree->ensure_interface_cache();
  ResourceScope scope;
  bke::ComputeContextCache compute_context_cache;
  nodes::SocketValueInferencer inferencer{
      *blender_mat->nodetree, scope, compute_context_cache};
  proxy.reflection_color = material_diffuse_albedo(blender_mat);
  proxy.transmission_color = proxy.reflection_color;
  auto finalize_proxy = [&]() -> HardwareMaterialProxy {
    if ((proxy.closure_type == HWRT_CLOSURE_REFRACTION) ||
        ((proxy.flags & HWRT_PROXY_FLAG_PRINCIPLED_TRANSMISSION_LAYER) != 0u))
    {
      const float thickness = material_output_thickness(blender_mat, inferencer);
      const bool slab_mode = (blender_mat->thickness_mode == MA_THICKNESS_SLAB);
      proxy.packed_thickness = proxy_thickness_pack(
          (thickness >= 0.0f) ? thickness : material_default_thickness(ob), slab_mode);
    }
    return proxy;
  };
  /* Keep the sync-time proxy bounded: one dominant base/specular family with coarse tint,
   * roughness, IOR, and the dielectric-reflection hint for continuation branching. */

  for (const bNode *node = static_cast<const bNode *>(blender_mat->nodetree->nodes.first);
       node != nullptr;
       node = node->next)
  {
    switch (node->type_legacy) {
      case SH_NODE_BSDF_PRINCIPLED: {
        const float transmission = inferred_socket_float_value(
            bke::node_find_socket(*node, SOCK_IN, "Transmission Weight"), inferencer);
        const float alpha = clamp_f(inferred_socket_float_value(
                                        bke::node_find_socket(*node, SOCK_IN, "Alpha"), inferencer),
                                    0.0f,
                                    1.0f);
        const float metallic = inferred_socket_float_value(
            bke::node_find_socket(*node, SOCK_IN, "Metallic"), inferencer);
        const float coat_weight = std::max(
            0.0f,
            inferred_socket_float_value(bke::node_find_socket(*node, SOCK_IN, "Coat Weight"),
                                        inferencer));
        const float coat_roughness = inferred_socket_float_value(
            bke::node_find_socket(*node, SOCK_IN, "Coat Roughness"), inferencer);
        const float coat_ior = inferred_socket_float_value(
            bke::node_find_socket(*node, SOCK_IN, "Coat IOR"), inferencer);
        const bNodeSocket *coat_tint_socket = bke::node_find_socket(*node, SOCK_IN, "Coat Tint");
        const float3 coat_tint = (coat_tint_socket != nullptr) ?
                                     inferred_socket_color_value(coat_tint_socket, inferencer) :
                                     float3(1.0f);
        const bNodeSocket *base_color_socket = bke::node_find_socket(*node, SOCK_IN, "Base Color");
        const float3 base_color = inferred_socket_color_value(base_color_socket, inferencer);
        const bool dynamic_base_color = inferred_socket_color_is_dynamic(base_color_socket, inferencer);
        const float roughness = inferred_socket_float_value(
            bke::node_find_socket(*node, SOCK_IN, "Roughness"), inferencer);
        const float ior = std::max(1.0e-3f,
                                   inferred_socket_float_value(
                                       bke::node_find_socket(*node, SOCK_IN, "IOR"), inferencer));
        const bool transmissive_proxy = (transmission > 1.0e-3f) && (metallic <= 0.5f);
        const bool metallic_proxy = (metallic > 1.0e-3f) && (transmission <= 1.0e-3f);
        const bool dielectric_reflection_scene_final_proxy =
            (coat_weight <= 1.0e-3f) && (metallic < 1.0f - 1.0e-3f) &&
            (transmission <= 1.0e-3f) &&
            /* Keep the layered Principled replay eligible through the full roughness range. */
            (roughness <= 1.0f) && (ior > 1.0f + 1.0e-3f);
        const bool layered_scene_final_proxy =
            (coat_weight > 1.0e-3f) ||
            /* Keep the mirrored Principled handoff continuous through metallic=1.0. The replayed
             * first-hit shader naturally fades the diffuse/base response to zero there, while the
             * coarse pure-reflection proxy branch creates a visible hard switch exactly at 1.0. */
            ((metallic > 1.0e-3f) && (metallic < 1.0f - 1.0e-3f) &&
             (transmission <= 1.0e-3f)) ||
            /* Preserve the same replay handoff through the full Principled transmission range so
             * pure transmissive hits do not snap back to the coarse glass proxy in mirrors. */
            (transmission > 1.0e-3f) ||
            dielectric_reflection_scene_final_proxy;
        /* Sparse hit replay can only preserve one dominant specular-family lobe on the fallback
         * path. Keep transmissive Principled materials tinted on that reflection branch too so
         * mirrored colored glass/coat cases do not bleach back to untinted dielectric F0. */
        proxy.reflection_color = base_color;
        proxy.transmission_color = base_color;
        proxy.reflection_roughness = (coat_weight > 1.0e-3f) ? coat_roughness : roughness;
        proxy.transmission_roughness = roughness;
        if (coat_weight > 1.0e-3f) {
          proxy.reflection_color *= coat_tint;
        }
        proxy.reflection_ior = (coat_weight > 1.0e-3f) ? std::max(1.0e-3f, coat_ior) : ior;
        proxy.refraction_ior = ior;
        proxy.alpha = alpha;
        proxy.reflection_layer_coverage = clamp_f(metallic, 0.0f, 1.0f);
        proxy.closure_type = metallic_proxy   ? HWRT_CLOSURE_REFLECTION :
                             transmissive_proxy ? HWRT_CLOSURE_REFRACTION :
                                                  HWRT_CLOSURE_DIFFUSE;
        if (layered_scene_final_proxy) {
          /* Mirror-style scene-final reflection should replay the first layered Principled surface
           * so the combined shader stack survives secondary hits instead of collapsing to the coarse
           * pure metal / glass proxy branch. */
          proxy.flags |= HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL;
        }
        if (transmission > 1.0e-3f) {
          /* Mixed Principled reflection+transmission needs a dedicated transmitted receiver path in
           * scene-final replay instead of collapsing both branches into one dominant proxy lobe. */
          proxy.flags |= HWRT_PROXY_FLAG_PRINCIPLED_TRANSMISSION_LAYER;
        }
        if (transmissive_proxy) {
          proxy.flags |= HWRT_PROXY_FLAG_DIELECTRIC_REFLECTION;
        }
        if (dynamic_base_color && metallic >= 1.0f - 1.0e-3f &&
            transmission <= 1.0e-3f)
        {
          proxy.flags |= HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL;
        }
        return finalize_proxy();
      }
      case SH_NODE_BSDF_GLASS:
      case SH_NODE_BSDF_REFRACTION: {
        const float3 color = inferred_socket_color_value(
            bke::node_find_socket(*node, SOCK_IN, "Color"), inferencer);
        const float roughness = inferred_socket_float_value(
            bke::node_find_socket(*node, SOCK_IN, "Roughness"), inferencer);
        const float ior = std::max(1.0e-3f,
                                   inferred_socket_float_value(
                                       bke::node_find_socket(*node, SOCK_IN, "IOR"), inferencer));
        /* Keep reflected colored glass on the material tint instead of flattening the proxy
         * reflection branch back to white. Colorless glass is unchanged because `color == 1`. */
        proxy.reflection_color = color;
        proxy.transmission_color = color;
        proxy.reflection_roughness = roughness;
        proxy.transmission_roughness = roughness;
        proxy.reflection_ior = ior;
        proxy.refraction_ior = ior;
        proxy.reflection_layer_coverage = 1.0f;
        proxy.closure_type = HWRT_CLOSURE_REFRACTION;
        if (node->type_legacy == SH_NODE_BSDF_GLASS) {
          proxy.flags |= HWRT_PROXY_FLAG_DIELECTRIC_REFLECTION;
          proxy.flags |= HWRT_PROXY_FLAG_PRINCIPLED_TRANSMISSION_LAYER;
        }
        proxy.flags |= HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL;
        return finalize_proxy();
      }
      case SH_NODE_BSDF_METALLIC: {
        const float3 color = inferred_socket_color_value(
            bke::node_find_socket(*node, SOCK_IN, "Base Color"), inferencer);
        const float roughness = inferred_socket_float_value(
            bke::node_find_socket(*node, SOCK_IN, "Roughness"), inferencer);
        proxy.reflection_color = color;
        proxy.transmission_color = color;
        proxy.reflection_roughness = roughness;
        proxy.transmission_roughness = roughness;
        proxy.closure_type = HWRT_CLOSURE_REFLECTION;
        proxy.flags |= HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL;
        return finalize_proxy();
      }
      case SH_NODE_BSDF_GLOSSY: {
        const float3 color = inferred_socket_color_value(
            bke::node_find_socket(*node, SOCK_IN, "Color"), inferencer);
        const float roughness = inferred_socket_float_value(
            bke::node_find_socket(*node, SOCK_IN, "Roughness"), inferencer);
        proxy.reflection_color = color;
        proxy.transmission_color = color;
        proxy.reflection_roughness = roughness;
        proxy.transmission_roughness = roughness;
        proxy.closure_type = HWRT_CLOSURE_REFLECTION;
        proxy.flags |= HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL;
        return finalize_proxy();
      }
      case SH_NODE_BSDF_TRANSLUCENT: {
        const float3 color = inferred_socket_color_value(
            bke::node_find_socket(*node, SOCK_IN, "Color"), inferencer);
        proxy.reflection_color = color;
        proxy.transmission_color = color;
        proxy.closure_type = HWRT_CLOSURE_TRANSLUCENT;
        return finalize_proxy();
      }
      case SH_NODE_BSDF_DIFFUSE: {
        const float3 color = inferred_socket_color_value(
            bke::node_find_socket(*node, SOCK_IN, "Color"), inferencer);
        const float roughness = inferred_socket_float_value(
            bke::node_find_socket(*node, SOCK_IN, "Roughness"), inferencer);
        proxy.reflection_color = color;
        proxy.transmission_color = color;
        proxy.reflection_roughness = roughness;
        proxy.transmission_roughness = roughness;
        proxy.closure_type = HWRT_CLOSURE_DIFFUSE;
        return finalize_proxy();
      }
      default:
        break;
    }
  }

  return finalize_proxy();
}

static float3 gpu_material_emissive_radiance(GPUMaterial *gpu_material)
{
  return material_emissive_radiance((gpu_material != nullptr) ? GPU_material_get_material(gpu_material) :
                                                               nullptr);
}

static HardwareMaterialProxy gpu_material_hardware_proxy(GPUMaterial *gpu_material, const Object *ob)
{
  return material_hardware_proxy(
      (gpu_material != nullptr) ? GPU_material_get_material(gpu_material) : nullptr, ob);
}

static void hardware_proxy_make_clear_refraction(HardwareMaterialProxy &proxy)
{
  proxy.reflection_color = float3(1.0f);
  proxy.reflection_roughness = 0.0f;
  proxy.transmission_color = float3(1.0f);
  proxy.transmission_roughness = 0.0f;
  proxy.reflection_ior = 1.0f;
  proxy.refraction_ior = 1.0f;
  proxy.packed_thickness = 0.0f;
  proxy.alpha = 1.0f;
  proxy.reflection_layer_coverage = 0.0f;
  proxy.closure_type = HWRT_CLOSURE_REFRACTION;
  proxy.flags &= ~(HWRT_PROXY_FLAG_DIELECTRIC_REFLECTION |
                   HWRT_PROXY_FLAG_PRINCIPLED_LAYERED_SCENE_FINAL |
                   HWRT_PROXY_FLAG_ALPHA_BLEND_TRANSPARENT |
                   HWRT_PROXY_FLAG_PRINCIPLED_TRANSMISSION_LAYER |
                   HWRT_PROXY_FLAG_TEXTURED_SPECULAR_SCENE_FINAL);
}

static Vector<const GPUMaterial *> material_array_hit_eval_gpu_materials(
    const MaterialArray &material_array)
{
  Vector<const GPUMaterial *> materials;
  materials.reserve(material_array.materials.size());
  for (const int material_index : material_array.materials.index_range()) {
    const Material &material = material_array.materials[material_index];
    const GPUMaterial *shading_gpumat = material_array.gpu_materials[material_index];
    materials.append((material.hit_eval.gpumat != nullptr) ? material.hit_eval.gpumat :
                                                         shading_gpumat);
  }
  return materials;
}

static Vector<const GPUMaterial *> material_array_shading_gpu_materials(
    const MaterialArray &material_array)
{
  Vector<const GPUMaterial *> materials;
  materials.reserve(material_array.gpu_materials.size());
  for (GPUMaterial *shading_gpumat : material_array.gpu_materials) {
    materials.append(shading_gpumat);
  }
  return materials;
}

static Vector<GPUMaterial *> material_array_replay_attribute_materials(
    const MaterialArray &material_array)
{
  Vector<GPUMaterial *> materials;
  materials.reserve(material_array.gpu_materials.size() * 2);
  for (const int material_index : material_array.materials.index_range()) {
    GPUMaterial *shading_gpumat = material_array.gpu_materials[material_index];
    GPUMaterial *hit_eval_gpumat = material_array.materials[material_index].hit_eval.gpumat;
    if (shading_gpumat != nullptr) {
      materials.append(shading_gpumat);
    }
    if ((hit_eval_gpumat != nullptr) && (hit_eval_gpumat != shading_gpumat)) {
      materials.append(hit_eval_gpumat);
    }
  }
  return materials;
}

static uint32_t hardware_raytrace_instance_count(const ResourceHandleRange &res_handle)
{
  const int64_t raw_instance_count = std::max<int64_t>(1, res_handle.id_range().size());
  return uint32_t(std::min<int64_t>(raw_instance_count, std::numeric_limits<uint32_t>::max()));
}

static uint64_t gpu_material_runtime_hash(GPUMaterial *gpu_material)
{
  if (gpu_material == nullptr) {
    return 0;
  }

  uint64_t hash = get_default_hash(
      GPU_material_uuid_get(gpu_material),
      GPU_material_compilation_timestamp(gpu_material),
      uint64_t(GPU_material_flag(gpu_material)));
  const ListBaseT<GPUMaterialTexture> textures = GPU_material_textures(gpu_material);
  for (const GPUMaterialTexture *texture = static_cast<const GPUMaterialTexture *>(textures.first);
       texture != nullptr;
       texture = texture->next)
  {
    hash = get_default_hash(hash,
                            texture->ima,
                            texture->iuser_available ? texture->iuser.framenr : 0,
                            texture->iuser_available ? texture->iuser.tile : 0,
                            texture->iuser_available ? texture->iuser.layer : 0);
    hash = get_default_hash(hash,
                            texture->iuser_available ? texture->iuser.pass : 0,
                            texture->iuser_available ? texture->iuser.multi_index : 0,
                            texture->iuser_available ? texture->iuser.view : 0);
  }
  return hash;
}

static uint64_t hardware_material_runtime_hash(GPUMaterial *gpu_material, GPUMaterial *hit_eval_gpumat)
{
  uint64_t hash = gpu_material_runtime_hash(gpu_material);
  if (hit_eval_gpumat != nullptr && hit_eval_gpumat != gpu_material) {
    hash = get_default_hash(hash, gpu_material_runtime_hash(hit_eval_gpumat));
  }
  return hash;
}

/* -------------------------------------------------------------------- */
/** \name Recalc
 *
 * \{ */

static bool use_hardware_raytrace_scene_capture(const Instance &inst)
{
  return (inst.scene->eevee.flag & SCE_EEVEE_SSR_ENABLED) != 0 &&
         inst.scene->eevee.ray_tracing_method == RAYTRACE_EEVEE_METHOD_HARDWARE &&
         GPU_viewport_hardware_raytracing_support();
}

void SyncModule::begin_sync()
{
  hardware_raytrace_scene_entries_.clear();
  hardware_raytrace_scene_signature_ = 0;
}

ObjectHandle &SyncModule::sync_object(const ObjectRef &ob_ref)
{
  ObjectKey key(ob_ref);

  ObjectHandle &handle = ob_handles.lookup_or_add_cb(key, [&]() {
    ObjectHandle new_handle;
    new_handle.object_key = key;
    return new_handle;
  });

  handle.recalc = inst_.get_recalc_flags(ob_ref);

  return handle;
}

WorldHandle SyncModule::sync_world(const blender::World &world)
{
  WorldHandle handle;
  handle.recalc = inst_.get_recalc_flags(world);
  return handle;
}

void SyncModule::append_hardware_raytrace_scene_entry(Object *ob,
                                                      Object *hit_eval_object,
                                                      const ObjectKey &object_key,
                                                      gpu::Batch *geom,
                                                      int recalc,
                                                      ResourceHandleRange res_handle,
                                                      int material_slot,
                                                      bool is_sculpt,
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
                                                      uint64_t material_runtime_hash)
{
  if ((ob == nullptr) || (geom == nullptr) || !res_handle.is_valid()) {
    return;
  }

  hardware_raytrace_scene_entries_.append(
      {object_key,
       hit_eval_object,
       geom,
       recalc,
       res_handle,
       float4x4(ob->object_to_world().ptr()),
       hardware_raytrace_instance_count(res_handle),
       material_slot,
       is_sculpt,
       emissive_radiance,
       diffuse_albedo,
       reflection_color,
       reflection_roughness,
       transmission_color,
       transmission_roughness,
       reflection_ior,
       refraction_ior,
       packed_thickness,
       alpha,
       reflection_layer_coverage,
       closure_type,
       proxy_flags,
       material_runtime_hash});
  hardware_raytrace_scene_signature_ = get_default_hash(
      hardware_raytrace_scene_signature_,
      ob,
      geom,
      material_slot,
      is_sculpt,
      material_runtime_hash);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Common
 * \{ */

static inline void geometry_call(PassMain::Sub *sub_pass,
                                 gpu::Batch *geom,
                                 ResourceHandleRange resource_handle)
{
  if (sub_pass != nullptr) {
    sub_pass->draw(geom, resource_handle);
  }
}

static inline void volume_call(MaterialPass &matpass,
                               Scene *scene,
                               Object *ob,
                               gpu::Batch *geom,
                               ResourceHandleRange res_handle)
{
  if (matpass.sub_pass != nullptr) {
    PassMain::Sub *object_pass = volume_sub_pass(*matpass.sub_pass, scene, ob, matpass.gpumat);
    if (object_pass != nullptr) {
      object_pass->draw(geom, res_handle);
    }
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Mesh
 * \{ */

void SyncModule::sync_mesh(Object *ob, ObjectHandle &ob_handle, const ObjectRef &ob_ref)
{
  if (!inst_.use_surfaces) {
    return;
  }

  if ((ob->dt < OB_SOLID) && (inst_.is_viewport() && inst_.v3d->shading.type != OB_RENDER)) {
    /** Do not render objects with display type lower than solid when in material preview mode. */
    return;
  }

  ResourceHandleRange res_handle = inst_.manager->unique_handle(ob_ref);

  bool has_motion = inst_.velocity.step_object_sync(
      ob_handle.object_key, ob_ref, ob_handle.recalc, res_handle);

  MaterialArray &material_array = inst_.materials.material_array_get(ob, has_motion);
  const bool capture_hardware_scene = use_hardware_raytrace_scene_capture(inst_);
  const Vector<const GPUMaterial *> hit_eval_gpu_materials = capture_hardware_scene ?
                                                                 material_array_hit_eval_gpu_materials(
                                                                     material_array) :
                                                                 Vector<const GPUMaterial *>();
  const Vector<GPUMaterial *> replay_attribute_materials = capture_hardware_scene ?
                                                               material_array_replay_attribute_materials(
                                                                   material_array) :
                                                               Vector<GPUMaterial *>();

  Span<gpu::Batch *> mat_geom = DRW_cache_object_surface_material_get(
      ob, material_array.gpu_materials);
  if (mat_geom.is_empty()) {
    return;
  }
  const Span<gpu::Batch *> hit_eval_geom = capture_hardware_scene ?
                                               DRW_cache_object_surface_material_get(
                                                   ob, hit_eval_gpu_materials.as_span()) :
                                               Span<gpu::Batch *>();
  Object *const hit_eval_object = ob_ref.stable_hit_eval_object();

  bool is_alpha_blend = false;
  bool has_transparent_shadows = false;
  bool has_volume = false;
  float inflate_bounds = 0.0f;
  for (auto i : material_array.gpu_materials.index_range()) {
    gpu::Batch *geom = mat_geom[i];
    if (geom == nullptr) {
      continue;
    }
    gpu::Batch *hwrt_geom = (!hit_eval_geom.is_empty() && (hit_eval_geom[i] != nullptr)) ?
                                hit_eval_geom[i] :
                                geom;

    Material &material = material_array.materials[i];
    GPUMaterial *gpu_material = material_array.gpu_materials[i];

    if (material.has_volume) {
      volume_call(material.volume_occupancy, inst_.scene, ob, geom, res_handle);
      volume_call(material.volume_material, inst_.scene, ob, geom, res_handle);
      has_volume = true;
      /* Do not render surface if we are rendering a volume object
       * and do not have a surface closure. */
      if (!material.has_surface) {
        continue;
      }
    }

    if (capture_hardware_scene) {
      HardwareMaterialProxy material_proxy = gpu_material_hardware_proxy(gpu_material, ob);
      const uint64_t material_runtime_hash = hardware_material_runtime_hash(
          gpu_material, material.hit_eval.gpumat);
      if (material.is_alpha_blend_transparent &&
          GPU_material_flag_get(gpu_material, GPU_MATFLAG_TRANSPARENT))
      {
        hardware_proxy_make_clear_refraction(material_proxy);
      }
      append_hardware_raytrace_scene_entry(ob,
                                           hit_eval_object,
                                           ob_handle.object_key,
                                           hwrt_geom,
                                           ob_handle.recalc,
                                           res_handle,
                                           i,
                                           false,
                                           gpu_material_emissive_radiance(gpu_material),
                                           material_diffuse_albedo(GPU_material_get_material(gpu_material)),
                                           material_proxy.reflection_color,
                                           material_proxy.reflection_roughness,
                                           material_proxy.transmission_color,
                                           material_proxy.transmission_roughness,
                                           material_proxy.reflection_ior,
                                           material_proxy.refraction_ior,
                                           material_proxy.packed_thickness,
                                           material_proxy.alpha,
                                           material_proxy.reflection_layer_coverage,
                                           material_proxy.closure_type,
                                           material_proxy.flags,
                                           material_runtime_hash);
    }

    geometry_call(material.capture.sub_pass, geom, res_handle);
    geometry_call(material.overlap_masking.sub_pass, geom, res_handle);
    geometry_call(material.prepass.sub_pass, geom, res_handle);
    geometry_call(material.shading.sub_pass, geom, res_handle);
    geometry_call(material.shadow.sub_pass, geom, res_handle);

    geometry_call(material.planar_probe_prepass.sub_pass, geom, res_handle);
    geometry_call(material.planar_probe_shading.sub_pass, geom, res_handle);
    geometry_call(material.lightprobe_sphere_prepass.sub_pass, geom, res_handle);
    geometry_call(material.lightprobe_sphere_shading.sub_pass, geom, res_handle);

    is_alpha_blend = is_alpha_blend || material.is_alpha_blend_transparent;
    has_transparent_shadows = has_transparent_shadows || material.has_transparent_shadows;

    blender::Material *mat = GPU_material_get_material(gpu_material);
    inst_.cryptomatte.sync_material(mat);

    if (GPU_material_has_displacement_output(gpu_material)) {
      inflate_bounds = math::max(inflate_bounds, mat->inflate_bounds);
    }
  }

  if (has_volume) {
    inst_.volume.object_sync(ob_handle);
  }

  if (inflate_bounds != 0.0f) {
    inst_.manager->update_handle_bounds(res_handle, ob_ref, inflate_bounds);
  }

  inst_.manager->extract_object_attributes(res_handle,
                                           ob_ref,
                                           capture_hardware_scene ? replay_attribute_materials.as_span() :
                                                                    material_array.gpu_materials.as_span());

  inst_.shadows.sync_object(ob, ob_handle, res_handle, is_alpha_blend, has_transparent_shadows);
  inst_.cryptomatte.sync_object(ob, res_handle);
}

bool SyncModule::sync_sculpt(Object *ob, ObjectHandle &ob_handle, const ObjectRef &ob_ref)
{
  if (!inst_.use_surfaces) {
    return false;
  }

  bool pbvh_draw = BKE_sculptsession_use_pbvh_draw(ob, inst_.rv3d) && !inst_.is_image_render;
  if (!pbvh_draw) {
    return false;
  }

  ResourceHandleRange res_handle = inst_.manager->unique_handle_for_sculpt(ob_ref);
  Object *const hit_eval_object = ob_ref.stable_hit_eval_object();

  bool has_motion = false;
  MaterialArray &material_array = inst_.materials.material_array_get(ob, has_motion);
  const bool capture_hardware_scene = use_hardware_raytrace_scene_capture(inst_);
  const Vector<const GPUMaterial *> sculpt_gpu_materials = capture_hardware_scene ?
                                                               material_array_hit_eval_gpu_materials(
                                                                   material_array) :
                                                               material_array_shading_gpu_materials(
                                                                   material_array);
  const Vector<GPUMaterial *> replay_attribute_materials = capture_hardware_scene ?
                                                               material_array_replay_attribute_materials(
                                                                   material_array) :
                                                               Vector<GPUMaterial *>();

  bool is_alpha_blend = false;
  bool has_transparent_shadows = false;
  bool has_volume = false;
  float inflate_bounds = 0.0f;
  for (SculptBatch &batch : sculpt_batches_per_material_get(ob_ref.object, sculpt_gpu_materials))
  {
    gpu::Batch *geom = batch.batch;
    if (geom == nullptr) {
      continue;
    }

    Material &material = material_array.materials[batch.material_slot];
    GPUMaterial *gpu_material = material_array.gpu_materials[batch.material_slot];

    if (material.has_volume) {
      volume_call(material.volume_occupancy, inst_.scene, ob, geom, res_handle);
      volume_call(material.volume_material, inst_.scene, ob, geom, res_handle);
      has_volume = true;
      /* Do not render surface if we are rendering a volume object
       * and do not have a surface closure. */
      if (material.has_surface == false) {
        continue;
      }
    }

    if (capture_hardware_scene) {
      HardwareMaterialProxy material_proxy = gpu_material_hardware_proxy(gpu_material, ob);
      const uint64_t material_runtime_hash = hardware_material_runtime_hash(
          gpu_material, material.hit_eval.gpumat);
      if (material.is_alpha_blend_transparent &&
          GPU_material_flag_get(gpu_material, GPU_MATFLAG_TRANSPARENT))
      {
        hardware_proxy_make_clear_refraction(material_proxy);
      }
      append_hardware_raytrace_scene_entry(ob,
                                           hit_eval_object,
                                           ob_handle.object_key,
                                           geom,
                                           ob_handle.recalc,
                                           res_handle,
                                           batch.material_slot,
                                           true,
                                           gpu_material_emissive_radiance(gpu_material),
                                           material_diffuse_albedo(GPU_material_get_material(gpu_material)),
                                           material_proxy.reflection_color,
                                           material_proxy.reflection_roughness,
                                           material_proxy.transmission_color,
                                           material_proxy.transmission_roughness,
                                           material_proxy.reflection_ior,
                                           material_proxy.refraction_ior,
                                           material_proxy.packed_thickness,
                                           material_proxy.alpha,
                                           material_proxy.reflection_layer_coverage,
                                           material_proxy.closure_type,
                                           material_proxy.flags,
                                           material_runtime_hash);
    }

    geometry_call(material.capture.sub_pass, geom, res_handle);
    geometry_call(material.overlap_masking.sub_pass, geom, res_handle);
    geometry_call(material.prepass.sub_pass, geom, res_handle);
    geometry_call(material.shading.sub_pass, geom, res_handle);
    geometry_call(material.shadow.sub_pass, geom, res_handle);

    geometry_call(material.planar_probe_prepass.sub_pass, geom, res_handle);
    geometry_call(material.planar_probe_shading.sub_pass, geom, res_handle);
    geometry_call(material.lightprobe_sphere_prepass.sub_pass, geom, res_handle);
    geometry_call(material.lightprobe_sphere_shading.sub_pass, geom, res_handle);

    is_alpha_blend = is_alpha_blend || material.is_alpha_blend_transparent;
    has_transparent_shadows = has_transparent_shadows || material.has_transparent_shadows;

    blender::Material *mat = GPU_material_get_material(gpu_material);
    inst_.cryptomatte.sync_material(mat);

    if (GPU_material_has_displacement_output(gpu_material)) {
      inflate_bounds = math::max(inflate_bounds, mat->inflate_bounds);
    }
  }

  if (has_volume) {
    inst_.volume.object_sync(ob_handle);
  }

  inst_.manager->extract_object_attributes(res_handle,
                                           ob_ref,
                                           capture_hardware_scene ? replay_attribute_materials.as_span() :
                                                                    material_array.gpu_materials.as_span());

  inst_.shadows.sync_object(ob, ob_handle, res_handle, is_alpha_blend, has_transparent_shadows);
  inst_.cryptomatte.sync_object(ob, res_handle);

  return true;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Point Cloud
 * \{ */

void SyncModule::sync_pointcloud(Object *ob, ObjectHandle &ob_handle, const ObjectRef &ob_ref)
{
  const int material_slot = POINTCLOUD_MATERIAL_NR;

  ResourceHandleRange res_handle = inst_.manager->unique_handle(ob_ref);

  bool has_motion = inst_.velocity.step_object_sync(
      ob_handle.object_key, ob_ref, ob_handle.recalc, res_handle);

  Material &material = inst_.materials.material_get(
      ob, has_motion, material_slot - 1, MAT_GEOM_POINTCLOUD);

  auto drawcall_add = [&](MaterialPass &matpass, bool dual_sided = false) {
    if (matpass.sub_pass == nullptr) {
      return;
    }
    PassMain::Sub &object_pass = matpass.sub_pass->sub("Point Cloud Sub Pass");
    gpu::Batch *geometry = pointcloud_sub_pass_setup(object_pass, ob, matpass.gpumat);
    if (dual_sided) {
      /* WORKAROUND: Hack to generate backfaces. Should also be baked into the Index Buf too at
       * some point in the future. */
      object_pass.push_constant("ptcloud_backface", false);
      object_pass.draw(geometry, res_handle);
      object_pass.push_constant("ptcloud_backface", true);
      object_pass.draw(geometry, res_handle);
    }
    else {
      object_pass.push_constant("ptcloud_backface", false);
      object_pass.draw(geometry, res_handle);
    }
  };

  if (material.has_volume) {
    /* Only support single volume material for now. */
    drawcall_add(material.volume_occupancy, true);
    drawcall_add(material.volume_material);
    inst_.volume.object_sync(ob_handle);

    /* Do not render surface if we are rendering a volume object
     * and do not have a surface closure. */
    if (material.has_surface == false) {
      return;
    }
  }

  drawcall_add(material.capture);
  drawcall_add(material.overlap_masking);
  drawcall_add(material.prepass);
  drawcall_add(material.shading);
  drawcall_add(material.shadow);

  drawcall_add(material.planar_probe_prepass);
  drawcall_add(material.planar_probe_shading);
  drawcall_add(material.lightprobe_sphere_prepass);
  drawcall_add(material.lightprobe_sphere_shading);

  inst_.cryptomatte.sync_object(ob, res_handle);
  GPUMaterial *gpu_material = material.shading.gpumat;
  blender::Material *mat = GPU_material_get_material(gpu_material);
  inst_.cryptomatte.sync_material(mat);

  if (GPU_material_has_displacement_output(gpu_material) && mat->inflate_bounds != 0.0f) {
    inst_.manager->update_handle_bounds(res_handle, ob_ref, mat->inflate_bounds);
  }

  inst_.manager->extract_object_attributes(res_handle, ob_ref, material.shading.gpumat);

  inst_.shadows.sync_object(ob,
                            ob_handle,
                            res_handle,
                            material.is_alpha_blend_transparent,
                            material.has_transparent_shadows);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Volume Objects
 * \{ */

void SyncModule::sync_volume(Object *ob, ObjectHandle &ob_handle, const ObjectRef &ob_ref)
{
  if (!inst_.use_volumes) {
    return;
  }

  ResourceHandleRange res_handle = inst_.manager->unique_handle(ob_ref);

  const int material_slot = VOLUME_MATERIAL_NR;

  /* Motion is not supported on volumes yet. */
  const bool has_motion = false;

  Material &material = inst_.materials.material_get(
      ob, has_motion, material_slot - 1, MAT_GEOM_VOLUME);

  if (!GPU_material_has_volume_output(material.volume_material.gpumat)) {
    return;
  }

  /* Do not render the object if there is no attribute used in the volume.
   * This mimic Cycles behavior (see #124061). */
  ListBaseT<GPUMaterialAttribute> attr_list = GPU_material_attributes(
      material.volume_material.gpumat);
  if (BLI_listbase_is_empty(&attr_list)) {
    return;
  }

  auto drawcall_add =
      [&](MaterialPass &matpass, gpu::Batch *geom, ResourceHandleRange res_handle) {
        if (matpass.sub_pass == nullptr) {
          return false;
        }
        PassMain::Sub *object_pass = volume_sub_pass(
            *matpass.sub_pass, inst_.scene, ob, matpass.gpumat);
        if (object_pass != nullptr) {
          object_pass->draw(geom, res_handle);
          return true;
        }
        return false;
      };

  /* Use bounding box tag empty spaces. */
  gpu::Batch *geom = inst_.volume.unit_cube_batch_get();

  bool is_rendered = false;
  is_rendered |= drawcall_add(material.volume_occupancy, geom, res_handle);
  is_rendered |= drawcall_add(material.volume_material, geom, res_handle);

  if (!is_rendered) {
    return;
  }

  inst_.manager->extract_object_attributes(res_handle, ob_ref, material.volume_material.gpumat);

  inst_.volume.object_sync(ob_handle);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Hair
 * \{ */

void SyncModule::sync_curves(Object *ob,
                             ObjectHandle &ob_handle,
                             const ObjectRef &ob_ref,
                             ResourceHandleRange res_handle,
                             ModifierData *modifier_data,
                             ParticleSystem *particle_sys)
{
  if (!inst_.use_curves) {
    return;
  }

  int mat_nr = CURVES_MATERIAL_NR;
  if (particle_sys != nullptr) {
    mat_nr = particle_sys->part->omat;
  }

  if (!res_handle.is_valid()) {
    /* For curve objects. */
    res_handle = inst_.manager->unique_handle(ob_ref);
  }

  bool has_motion = inst_.velocity.step_object_sync(
      ob_handle.object_key, ob_ref, ob_handle.recalc, res_handle, modifier_data, particle_sys);
  Material &material = inst_.materials.material_get(ob, has_motion, mat_nr - 1, MAT_GEOM_CURVES);

  auto drawcall_add = [&](MaterialPass &matpass) {
    if (matpass.sub_pass == nullptr) {
      return;
    }
    if (particle_sys != nullptr) {
      PassMain::Sub &sub_pass = matpass.sub_pass->sub("Hair SubPass");
      gpu::Batch *geometry = hair_sub_pass_setup(
          sub_pass, inst_.scene, ob_ref, particle_sys, modifier_data, matpass.gpumat);
      sub_pass.draw(geometry, res_handle);
    }
    else {
      PassMain::Sub &sub_pass = matpass.sub_pass->sub("Curves SubPass");
      const char *error = nullptr;
      gpu::Batch *geometry = curves_sub_pass_setup(
          sub_pass, inst_.scene, ob, error, matpass.gpumat);
      if (error) {
        inst_.info_append(error);
      }
      sub_pass.draw(geometry, res_handle);
    }
  };

  if (material.has_volume) {
    /* Only support single volume material for now. */
    drawcall_add(material.volume_occupancy);
    drawcall_add(material.volume_material);
    inst_.volume.object_sync(ob_handle);
    /* Do not render surface if we are rendering a volume object
     * and do not have a surface closure. */
    if (material.has_surface == false) {
      return;
    }
  }

  drawcall_add(material.capture);
  drawcall_add(material.overlap_masking);
  drawcall_add(material.prepass);
  drawcall_add(material.shading);
  drawcall_add(material.shadow);

  drawcall_add(material.planar_probe_prepass);
  drawcall_add(material.planar_probe_shading);
  drawcall_add(material.lightprobe_sphere_prepass);
  drawcall_add(material.lightprobe_sphere_shading);

  inst_.cryptomatte.sync_object(ob, res_handle);
  GPUMaterial *gpu_material = material.shading.gpumat;
  blender::Material *mat = GPU_material_get_material(gpu_material);
  inst_.cryptomatte.sync_material(mat);

  if (GPU_material_has_displacement_output(gpu_material) && mat->inflate_bounds != 0.0f) {
    inst_.manager->update_handle_bounds(res_handle, ob_ref, mat->inflate_bounds);
  }

  inst_.manager->extract_object_attributes(res_handle, ob_ref, material.shading.gpumat);

  inst_.shadows.sync_object(ob,
                            ob_handle,
                            res_handle,
                            material.is_alpha_blend_transparent,
                            material.has_transparent_shadows);
}

/** \} */

void foreach_hair_particle_handle(Instance &inst,
                                  ObjectRef &ob_ref,
                                  ObjectHandle ob_handle,
                                  HairHandleCallback callback)
{
  int sub_key = 1;

  for (ModifierData &md : ob_ref.object->modifiers) {
    if (md.type == eModifierType_ParticleSystem) {
      ParticleSystem *particle_sys = reinterpret_cast<ParticleSystemModifierData *>(&md)->psys;
      ParticleSettings *part_settings = particle_sys->part;
      /* Only use the viewport drawing mode for material preview. */
      const int draw_as = (part_settings->draw_as == PART_DRAW_REND || !inst.is_viewport()) ?
                              part_settings->ren_as :
                              part_settings->draw_as;
      if (draw_as != PART_DRAW_PATH ||
          !DRW_object_is_visible_psys_in_active_context(ob_ref.object, particle_sys))
      {
        continue;
      }

      ObjectHandle particle_sys_handle = ob_handle;
      particle_sys_handle.object_key = ObjectKey(ob_ref, sub_key++);
      particle_sys_handle.recalc = particle_sys->recalc;

      callback(particle_sys_handle, md, *particle_sys);
    }
  }
}

}  // namespace blender::eevee
