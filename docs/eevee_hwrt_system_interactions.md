<!-- SPDX-FileCopyrightText: 2026 Kondoo Digital GmbH -->
<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

# Eevee HWRT System Interactions

This document defines the implemented ownership boundaries between Nuru Hardware
RT, classic Eevee, GI, shadows, environment, reflections, refractions, material
replay, and caches in Diamond 1.

## Top-Level Ownership

- If the method is not `Nuru Raytracing`, classic Eevee tracing behavior owns the
  ray tracing features.
- If the method is `Nuru Raytracing` but the active viewport mode is Material
  Preview, the runtime uses classic screen tracing.
- If a Nuru feature toggle is disabled, that feature keeps classic Eevee
  ownership.
- If the backend is unsupported, the Hardware RT path must fail closed.

## Feature Mask

The active Hardware RT feature mask contains:

- `RAYTRACE_EEVEE_HARDWARE_GI`
- `RAYTRACE_EEVEE_HARDWARE_SHADOWS`
- `RAYTRACE_EEVEE_HARDWARE_REFLECTIONS`
- `RAYTRACE_EEVEE_HARDWARE_REFRACTIONS`
- `RAYTRACE_EEVEE_HARDWARE_ENVIRONMENT`

## GI Rules

- Hardware GI is enabled only when the method is Hardware RT and GI mode is On.
- Regular Hardware RT GI owns active diffuse Nuru GI.
- `Indirect GI` enables Primary and Secondary GI.
- Diffuse GI must not be inferred from scene-final screen radiance.
- Hardware GI can use world transport through explicit Hardware RT GI and miss
  handling.

## Shadow Rules

- Hardware Shadows are separate from Hardware GI.
- Primary and secondary shadow visibility use explicit Hardware RT passes.
- World/environment visibility and analytic direct-light shadow visibility are
  separate decisions.
- If Hardware Shadows are off, classic Eevee shadowing remains responsible.
- Secondary GI capture disables HWRT shadow visibility.

## Environment Rules

- Hardware Environment owns explicit environment visibility and world-shadow
  transport when enabled.
- Hardware GI keeps its own diffuse world transport behavior when Hardware
  Environment is off.
- Specular world misses keep direction-specific environment evaluation rather
  than using diffuse dome visibility as sharp specular truth.

## Reflection Rules

- Reflections are classic Eevee or Full RT.
- Full RT reflection avoids hidden screen ownership.
- Full RT scene-final specular receivers skip raster/cache reuse for diffuse
  receivers so screen/cache seams are not reflected into RT.

## Refraction Rules

- Refractions can be classic Eevee or Full RT.
- Full RT refraction makes Hardware RT own the path.
- Refracted reflective/metal receivers preserve replayed material tint.
- Transmission-layer payloads can preserve scene-final composite information.

## Material Replay Rules

- Sparse hit-eval replay is preferred when the traced hit has enough identity,
  barycentric, object, and batch compatibility data.
- Replay can evaluate the real material graph at the hit.
- Late HWRT lighting consumes collapsed base-family and specular-family
  closures.
- Proxy fallback is one dominant family with coarse parameters.
- Proxy fallback is not a full material model.

## Primary And Secondary GI

Primary and Secondary GI support data is used only when:

- Hardware RT is active;
- Hardware GI is on;
- Hardware reflections are on;
- `use_hardware_raytracing_indirect_gi_cache` is enabled;
- position and normal validation pass.

Secondary GI stores radiance, world position, and world normal. Pathtracer-equivalent
sharp mirror correctness for Primary and Secondary GI is a current limit.

## Review Checklist

- GI change: verify diffuse behavior and one specular guardrail.
- Shadow change: verify Hardware Shadows on/off and classic shadow fallback.
- Environment change: verify Hardware Environment on/off and GI world transport.
- Reflection change: verify Full RT reflection and Primary/Secondary GI receiver
  behavior.
- Refraction change: verify Off/Full RT ownership and textured refracted hits.
- Material replay change: verify replay and proxy fallback non-regression.
- Primary/Secondary GI change: verify gating, validation, and sharp mirror
  exclusion.
