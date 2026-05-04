<!-- SPDX-FileCopyrightText: 2026 Kondoo Digital GmbH -->
<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

# Principled HWRT Audit

This audit records the implemented Diamond 1 Principled BSDF and HWRT material
contract. It separates active support from simplification and unsupported
behavior.

## Current Architecture

Principled behavior in Nuru is split across visible Eevee shading, sparse HWRT
hit-eval replay, and proxy fallback.

Important paths:

- `source/blender/gpu/shaders/material/gpu_shader_material_principled.glsl`
  emits the visible Principled closure stack.
- `source/blender/draw/engines/eevee/shaders/eevee_nodetree_lib.glsl` routes
  closures into Eevee surface evaluation.
- `source/blender/draw/engines/eevee/eevee_gbuffer.hh` and
  `source/blender/draw/engines/eevee/shaders/eevee_gbuffer_lib.glsl` define
  compact deferred storage.
- `source/blender/draw/engines/eevee/shaders/eevee_surf_hit_eval_frag.glsl`
  replays material evaluation for traced HWRT hits when possible.
- `source/blender/draw/engines/eevee/shaders/eevee_ray_trace_hardware_lighting_comp.glsl`
  consumes replay/proxy closures in late HWRT lighting.
- `source/blender/draw/engines/eevee/eevee_sync.cc` creates the coarse
  `HardwareMaterialProxy`.
- `source/blender/gpu/metal/mtl_raytrace_acceleration.mm` exports proxy and hit
  payloads from Metal tracing.

## Replay Versus Proxy

Sparse replay and proxy fallback are not equivalent.

Sparse replay:

- can evaluate the real material graph at a traced hit;
- can preserve texture, normal, object, and closure information when the hit is
  compatible;
- still collapses the result into a limited base/specular closure contract for
  late lighting.

Proxy fallback:

- stores one dominant material family;
- stores coarse tint, roughness, and IOR-like parameters;
- does not store the complete Principled lobe stack;
- does not preserve all texture/attribute/normal behavior;
- must be treated as a bounded fallback.

Changing proxy classification changes shared HWRT fallback behavior, not only
Principled behavior.

## Primary Eevee Path Matrix

| Parameter group | Current status |
| --- | --- |
| Base color | Supported through visible closure color and storage collapse. |
| Roughness | Supported for isotropic reflection/refraction roughness. |
| Metallic | Simplified into reflection color/energy, not stored as a separate late HWRT channel. |
| IOR | Supported for refraction closure behavior; reflection-side effects are baked/simplified. |
| Specular IOR level / specular tint | Simplified into visible reflection color/Fresnel response. |
| Transmission weight | Supported at closure level, simplified by later storage and replay collapse. |
| Coat | Present as visible layered reflection behavior; compact late HWRT representation remains simplified. |
| Sheen | Simplified into diffuse/subsurface energy rather than a separate late HWRT closure. |
| Subsurface | Supported for the current Burley path, then collapsed as a base-family result in late HWRT lighting. |
| Subsurface IOR / anisotropy | Unsupported in active Eevee Principled shading. |
| Anisotropy / anisotropic rotation / tangent | Unsupported in active Eevee/Nuru reflection. |
| Emission | Present through separate emission paths, not as a normal deferred closure mode. |
| Alpha | Present through transparency/holdout behavior, not as a normal deferred closure mode. |
| Thin film thickness / IOR | Unsupported in active Eevee/Nuru Principled shading. |
| Diffuse roughness | Unsupported as a meaningful deferred/HWRT-resolved quantity. |

## HWRT Secondary-Hit Matrix

| Parameter group | Sparse replay | Proxy fallback | Current contract |
| --- | --- | --- | --- |
| Base color | Yes | Coarse tint | Supported when replay is available; simplified on proxy. |
| Roughness | Yes | Coarse roughness | Supported for simple isotropic cases. |
| Metallic | Present in replayed result | Threshold/family approximation | Simplified and family-dependent. |
| IOR | Present | Coarse parameter | Supported for simple refraction; simplified for layered cases. |
| Transmission | Present in replayed result | On/off family approximation | Simplified by closure collapse. |
| Coat | Present in replayed result | Coarse reflection approximation | Simplified in late HWRT lighting. |
| Specular IOR level / tint | Present only if replay selection keeps the effect | Not represented | Simplified or missing in fallback. |
| Sheen | May survive only as baked base energy | Not represented | Simplified. |
| Subsurface | May replay, then collapses to base-family lighting | Not represented | Simplified. |
| Thickness | Present in replay | Proxy treats thickness as zero | Replay-only for accurate thickness. |
| Emission | Separate/inferred path | Not full graph representation | Limited for non-constant graphs. |
| Normal / bump / attributes | Present when replay-compatible | Not represented | Replay-only. |
| Alpha / holdout | Partial through replay/composite paths | Not represented | Limited. |
| Thin film / anisotropy / diffuse roughness | Not meaningfully supported | Not represented | Unsupported. |

## Implemented Scene-Final Behavior

Diamond 1 includes explicit scene-final handling for reflected and refracted
material views.

- Full RT specular paths avoid preserving a hidden screen-space first-hit
  baseline unless a specific reuse path validates it.
- Refracted textured receivers can keep direct-light tint from the replayed
  metal/reflective material.
- Preserved layered scene-final hits can carry transmission-layer and layered
  receiver payloads.
- Rough secondary GI receivers may reuse validated raster/cache radiance.
- Pathtracer-equivalent sharp mirror correctness for Primary and Secondary GI is
  a current limit.

## Implemented Rough Texture Interaction

Rough material texture filtering applies before GBuffer packing and HWRT
hit-eval consumption.

- GGX reflection/refraction roughness can request filtering.
- Diffuse-only Principled usage does not blur texture inputs.
- The checker texture is the implemented participating node.
- Filtering does not change Principled lobe support or proxy family
  classification.
- EEVEE shaders are usually working; however, as Nuru is raytracing, some
  shaders and textures may need adjustments.
- Extended Shading and EEVEE/Cycles shader parity are Work in Progress.

## Current Limitations

These are Diamond 1 contract limits, not proposed work:

- no separate late HWRT anisotropic reflection model;
- no thin-film support;
- no diffuse roughness transport;
- no complete separate sheen closure in late HWRT lighting;
- no complete separate clearcoat storage contract for all layered cases;
- no full proxy representation of complex node graphs;
- Extended Shading and EEVEE/Cycles shader parity are Work in Progress;
- replay still collapses rich material graphs to limited late-lighting closure
  families;
- proxy fallback can lose texture, normal, thickness, alpha, and layered lobe
  information.

## Validation Coverage

Existing probe coverage relevant to this audit:

- `tests/python/eevee_hwrt_specular_parity.py`
  covers roughness, IOR, modifier-driven replay, and texture-binding parity.
- `tests/python/eevee_hwrt_specular_material_color_probe.py`
  covers metallic tint and textured metallic secondary-hit reflection behavior.
- `tests/python/eevee_hwrt_reflected_texture_probe.py`
  covers textured reflected hits.
- `tests/python/eevee_hwrt_refracted_texture_probe.py`
  covers textured refracted hits.
- `tests/python/eevee_hwrt_principled_dielectric_mirror_probe.py`
  covers current Principled dielectric mirror behavior.
- `tests/python/eevee_hwrt_principled_coat_probe.py`
  covers current Principled coat behavior.
- `tests/python/eevee_hwrt_validation_corpus.py`
  records the current validation envelope.

## Review Rule

When changing Principled/HWRT behavior, identify which implemented boundary is
being changed:

- visible Principled shader;
- GBuffer storage;
- sparse hit-eval replay;
- late HWRT lighting;
- proxy fallback;
- Metal payload export.

Do not describe a change as "Principled support" unless the affected boundary
and its current simplifications are explicit.
