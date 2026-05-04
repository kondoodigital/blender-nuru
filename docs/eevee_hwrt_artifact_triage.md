<!-- SPDX-FileCopyrightText: 2026 Kondoo Digital GmbH -->
<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

# Eevee HWRT Artifact Triage

This guide covers active Diamond 1 Nuru HWRT artifacts only.

## Useful Debug Controls

Active runtime controls include:

- `BLENDER_EEVEE_HWRT_PERF=1`
- `BLENDER_EEVEE_HWRT_FORCE_SYNC=1`
- `BLENDER_GPU_METAL_IGNORE_TEXTURE_POOL_ASSERT=1`
- `BLENDER_EEVEE_HWRT_CACHE_LOG=1`
- `BLENDER_EEVEE_HWRT_DEBUG_VIEW_MODE=direct`
- `BLENDER_EEVEE_HWRT_DEBUG_ISOLATE={direct|indirect}`

## GI Issues

Start with the feature state:

- confirm `Nuru Raytracing` is selected;
- confirm Hardware GI is On;
- compare with Hardware GI Off to separate direct lighting from diffuse GI;
- check whether Primary and Secondary GI data is involved;
- compare the probe result against a Cycles diffuse-only reference when the issue is GI truth.

Common failure classes:

- world transport missing from diffuse GI;
- Primary/Secondary GI receiver reuse outside its validation envelope;
- stale secondary GI data after scene, material, or transform edits;
- direct-light contribution being mistaken for diffuse GI.

## Shadow And Environment Issues

For shadow visibility, compare Hardware RT Shadows On and Off. For world visibility, compare Hardware RT Environment On and Off. If the artifact only appears with a feature enabled, inspect the matching visibility texture and direct-light accumulation path before changing GI ownership.

## Reflection And Refraction Issues

Scene-final specular traces are separate from the pre-combine GI pass. Sharp scene-final mirrors and refractions should stay replay-owned where possible; rough secondary GI receivers may reuse validated raster/cache radiance. If a mirror shows stale diffuse energy, check receiver identity, normal, position, and secondary GI validity.

## Material Issues

Material replay is richer than proxy fallback but still limited. EEVEE shaders are usually working; however, as Nuru is raytracing, some shaders and textures may need adjustments. When secondary hits look wrong, check whether the hit used replay data, proxy material data, or fallback radiance. Texture filtering for rough materials is documented separately in `eevee_rough_material_texture_filter.md`.
