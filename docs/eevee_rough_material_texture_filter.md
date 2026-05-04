<!-- SPDX-FileCopyrightText: 2026 Kondoo Digital GmbH -->
<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

# Eevee Rough Material Texture Filter

This document describes the rough material texture filtering implemented in
Diamond 1.

## Implemented Behavior

Rough GGX reflection and GGX refraction closures can request roughness-aware
filtering of material texture inputs during material evaluation.

This is not a late blur of reflected or refracted radiance. The material graph is
evaluated with a filtering roughness value before GBuffer packing or HWRT
hit-eval output is consumed by lighting.

User-visible behavior:

- diffuse-only materials do not blur texture inputs from roughness;
- GGX reflection can request filtering;
- GGX refraction can request filtering;
- visible surface evaluation and HWRT hit-eval replay use the same filtered
  material helper;
- a texture node participates only if the node implements a filtering path.

## Owning Files

The implemented ownership is in:

- `source/blender/draw/engines/eevee/shaders/eevee_surf_lib.glsl`
- `source/blender/draw/engines/eevee/shaders/eevee_nodetree_lib.glsl`
- `source/blender/gpu/shaders/material/gpu_shader_material_tex_checker.glsl`

The surface shader entry points that use filtered material evaluation include:

- deferred surfaces;
- hybrid surfaces;
- forward surfaces;
- HWRT hit-eval surfaces.

These paths call `nodetree_surface_material_filter_eval(...)` instead of calling
the plain nodetree surface function directly.

## Evaluation Model

The helper performs two material evaluations:

1. Evaluate with filtering disabled.
2. Capture roughness from weighted GGX reflection/refraction closures.
3. Re-evaluate the nodetree with `g_material_texture_filter_roughness` active.

The captured value is a material-input filtering hint. It is not a physically
exact footprint and it is not a replacement for ray differentials.

## Checker Texture

The checker texture is the implemented participating procedural node.

Behavior:

- unfiltered materials keep the legacy checker sample;
- filtered materials use a continuous roughness-filtered checker factor;
- the filter preserves checker topology;
- the filter avoids fixed-count supersampling bands;
- the filter avoids fake cross-shaped bands at checker cell intersections.

The filter smooths a signed checker square wave per axis with a continuous
periodic soft-sign approximation, then reconstructs the checker factor from the
smoothed axis signals.

The filter width comes from `material_texture_filter_checker_width_get()`. It
uses squared material roughness as a conservative proxy for checker-cell
footprint size.

## What Is Not Implemented

Diamond 1 does not document additional texture-node filtering as current
behavior. Image textures and other procedural texture nodes should not be listed
as participating unless their shader code has an implemented filtering path.

The rough material texture filter also does not change:

- Principled extra specular layer ownership;
- HWRT proxy classification;
- diffuse GI ownership;
- late scene-final radiance blur.

## Validation Focus

When validating this feature, check:

- diffuse-only material texture output remains unfiltered;
- rough reflective checker textures soften continuously;
- rough refractive checker textures soften continuously;
- visible surface and HWRT hit-eval replay agree for the same material;
- changing roughness does not introduce discrete value bands.
