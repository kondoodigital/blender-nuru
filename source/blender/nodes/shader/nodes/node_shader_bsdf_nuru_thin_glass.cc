/* SPDX-FileCopyrightText: 2005 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "BKE_context.hh"

#include "UI_interface_layout.hh"
#include "UI_resources.hh"

#include "node_shader_util.hh"

namespace blender {

namespace nodes::node_shader_bsdf_nuru_thin_glass_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.use_custom_socket_order();

  b.add_output<decl::Shader>("BSDF");

  b.add_default_layout();

  b.add_input<decl::Color>("Color").default_value({1.0f, 1.0f, 1.0f, 1.0f});
  b.add_input<decl::Float>("Film Alpha")
      .default_value(0.0f)
      .min(0.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR)
      .description(
          "Lerps the pass-through contribution toward a film hole while preserving reflections. "
          "Requires the Film > Transparent render setting");
  b.add_input<decl::Float>("Roughness")
      .default_value(0.0f)
      .min(0.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR);
  b.add_input<decl::Float>("IOR").default_value(1.5f).min(0.0f).max(1000.0f);
  b.add_input<decl::Vector>("Normal").hide_value();
  b.add_input<decl::Float>("Weight").available(false);
}

static void node_shader_buts_thin_glass(ui::Layout &layout, bContext *C, PointerRNA *ptr)
{
  /* Film Alpha composites a film hole; without a transparent film the render keeps an opaque
   * background and the slider appears to do nothing. Warn only while it actually matters so the
   * node stays clean in correctly configured scenes. */
  const bNode *node = static_cast<const bNode *>(ptr->data);
  const bNodeSocket *film_alpha = bke::node_find_socket(*node, SOCK_IN, "Film Alpha");
  if (film_alpha != nullptr) {
    const auto *film_alpha_value = static_cast<const bNodeSocketValueFloat *>(
        film_alpha->default_value);
    const bool film_alpha_used = (film_alpha->flag & SOCK_IS_LINKED) ||
                                 film_alpha_value->value > 0.0f;
    const Scene *scene = CTX_data_scene(C);
    const bool film_transparent = scene != nullptr && scene->r.alphamode == R_ALPHAPREMUL;
    if (film_alpha_used && !film_transparent) {
      layout.label(RPT_("Film Alpha requires Film > Transparent"), ICON_INFO);
    }
  }
  layout.prop(ptr, "distribution", ui::ITEM_R_SPLIT_EMPTY_NAME, "", ICON_NONE);
  layout.prop(ptr, "use_fresnel", ui::ITEM_R_SPLIT_EMPTY_NAME, "Use Fresnel", ICON_NONE);
}

static void node_shader_init_thin_glass(bNodeTree * /*ntree*/, bNode *node)
{
  node->custom1 = SHD_GLOSSY_MULTI_GGX;
  /* custom2 bit 0 disables Fresnel, so zero keeps existing and new nodes Fresnel-enabled. */
  node->custom2 = 0;
}

static int node_shader_gpu_bsdf_thin_glass(GPUMaterial *mat,
                                           bNode *node,
                                           bNodeExecData * /*execdata*/,
                                           GPUNodeStack *in,
                                           GPUNodeStack *out)
{
  if (!in[4].link) {
    GPU_link(mat, "world_normals_get", &in[4].link);
  }

  eGPUMaterialFlag flag = GPU_MATFLAG_GLOSSY | GPU_MATFLAG_TRANSPARENT |
                          GPU_MATFLAG_THIN_GLASS;

  if (in[0].might_be_tinted()) {
    flag |= GPU_MATFLAG_REFLECTION_MAYBE_COLORED | GPU_MATFLAG_TRANSPARENT_MAYBE_COLORED;
  }

  GPU_material_flag_set(mat, flag);

  float use_multi_scatter = (node->custom1 == SHD_GLOSSY_MULTI_GGX) ? 1.0f : 0.0f;
  float use_fresnel = (node->custom2 & 1) ? 0.0f : 1.0f;

  return GPU_stack_link(
      mat,
      node,
      "node_bsdf_thin_glass",
      in,
      out,
      GPU_constant(&use_multi_scatter),
      GPU_constant(&use_fresnel));
}

}  // namespace nodes::node_shader_bsdf_nuru_thin_glass_cc

void register_node_type_sh_bsdf_thin_glass()
{
  namespace file_ns = nodes::node_shader_bsdf_nuru_thin_glass_cc;

  static bke::bNodeType ntype;

  sh_node_type_base(&ntype, "ShaderNodeBsdfThinGlass", SH_NODE_BSDF_THIN_GLASS);
  ntype.ui_name = "Thin Glass BSDF";
  ntype.ui_description =
      "Glass-like reflections with a fully transparent transmitted lobe that does not trace "
      "refraction, shadows, or hardware refraction";
  ntype.enum_name_legacy = "BSDF_FAKE_GLASS";
  ntype.nclass = NODE_CLASS_SHADER;
  ntype.declare = file_ns::node_declare;
  ntype.gather_link_search_ops = search_link_ops_for_shader_bsdf_node;
  ntype.add_ui_poll = object_shader_nodes_poll;
  bke::node_type_size_preset(ntype, bke::eNodeSizePreset::Middle);
  ntype.draw_buttons = file_ns::node_shader_buts_thin_glass;
  ntype.initfunc = file_ns::node_shader_init_thin_glass;
  ntype.gpu_fn = file_ns::node_shader_gpu_bsdf_thin_glass;

  bke::node_register_type(ntype);
}

}  // namespace blender
