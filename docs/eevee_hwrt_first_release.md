<!-- SPDX-FileCopyrightText: 2026 Kondoo Digital GmbH -->
<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

# Eevee HWRT First Release Policy

This document describes the implemented Diamond 1 release envelope for Nuru
Eevee Hardware RT.

## Supported Scope

- Active viewport backend: Metal.
- Practical device target: Apple M3+ on macOS 14+.
- OptiX and CUDA Nuru backend support are Work in Progress.
- Apple M1/M2 and non-RTX-class cards cannot be supported due to lack of RT
  cores.
- User-facing tracing method: `Nuru Raytracing`.
- Material Preview uses the classic Eevee screen path instead of partial Nuru
  activation.

## Artist-Facing Controls

Implemented Nuru controls are intentionally per-feature:

- `Global Illumination`
- `GI Resolution`
- `Indirect GI`
- `Indirect GI Resolution`
- `Raytrace Reflections`
- reflection `Bounces`
- `Raytrace Refractions`
- refraction `Bounces`
- `Raytrace Environment`
- `Raytrace Shadows`
- surface `Indirect Light` clamp

Implemented modes:

- GI: Off or On.
- Reflections: Off or On.
- Refractions: Off or On.

When a Hardware RT feature is off, classic Eevee keeps ownership of that feature.

## Active GI Policy

- Regular Hardware RT GI owns active Diamond 1 diffuse Nuru GI.
- Screen-space or already-composed radiance must not replace the visible color
  of Full RT reflection/refraction paths.
- `Indirect GI` enables Primary and Secondary GI.
- Hardware GI and Hardware Environment are separate feature owners. Hardware GI
  keeps diffuse world transport behavior even when explicit Hardware Environment
  visibility is off.

## Shadows And Environment

- Hardware Shadows are a separate feature flag.
- Primary and secondary shadow visibility are explicit Hardware RT passes.
- Hardware Environment controls explicit environment visibility and world-shadow
  transport.
- Secondary GI capture disables HWRT shadow/environment visibility so stored
  radiance is GI rather than direct-lit beauty lighting.

## Reflections

Reflections are classic Eevee or Full RT.

Implemented Hardware RT reflection behavior includes:

- Metal scene tracing;
- hit position, normal, identity, material, and barycentric payloads;
- sparse material replay when a hit can be evaluated;
- proxy fallback when replay is unavailable;
- scene-final specular resolve;
- Primary and Secondary GI receiver handling.

Pathtracer-equivalent sharp mirror correctness for Primary and Secondary GI is a
current limit.

## Refractions

Refractions are classic Eevee or Full RT.

- Full RT makes Hardware RT own the path.
- Refracted hits that land on reflective or metallic receivers keep replayed
  material tint for continuation radiance.

## Material Policy

Diamond 1 material behavior is Eevee closure behavior with HWRT replay support.

- Sparse hit-eval replay can evaluate the real material graph at a traced hit.
- Late HWRT lighting still consumes collapsed base/specular closure families.
- Proxy fallback is coarse and not a complete Principled representation.
- Principled anisotropy, thin film, diffuse roughness, and some layered
  interactions remain unsupported or simplified.
- EEVEE shaders are usually working; however, as Nuru is raytracing, some
  shaders and textures may need adjustments.
- Extended Shading and EEVEE/Cycles shader parity are Work in Progress.

## Rough Texture Filtering

- GGX reflection and GGX refraction can request roughness filtering of material
  texture inputs.
- Diffuse-only materials do not blur texture inputs from roughness.
- The checker texture is the implemented participating procedural texture.
- Filtering happens during material evaluation, not as a late radiance blur.

## Work In Progress / Current Limits

- OptiX and CUDA Nuru backend support are Work in Progress;
- Apple M1/M2 and non-RTX-class cards cannot be supported due to lack of RT
  cores;
- Extended Shading and EEVEE/Cycles shader parity are Work in Progress;
- EEVEE shaders are usually working, but some shaders and textures may need
  adjustments for raytracing;
- screen-space radiance as diffuse GI correctness truth;
- anisotropic Principled reflection;
- thin-film Principled behavior;
- diffuse roughness as meaningful deferred/HWRT-resolved transport;
- full diffuse GI transport through refractive/translucent materials;
- volume multiple-scattering parity through the Nuru GI path;
- Pathtracer-equivalent sharp mirror correctness for Primary and Secondary GI.

## Debug Surface

Active/backend controls:

- `BLENDER_EEVEE_HWRT_PERF`
- `BLENDER_EEVEE_HWRT_FORCE_SYNC`
- `BLENDER_EEVEE_METAL_RT_CAPTURE_PATH`
- `BLENDER_GPU_METAL_IGNORE_TEXTURE_POOL_ASSERT`
- `BLENDER_EEVEE_HWRT_CACHE_LOG`
- `BLENDER_EEVEE_HWRT_DEBUG_VIEW_MODE`
- `BLENDER_EEVEE_HWRT_DEBUG_ISOLATE`
