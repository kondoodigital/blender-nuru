<!-- SPDX-FileCopyrightText: 2026 Kondoo Digital GmbH -->
<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

# Eevee HWRT Ship Criteria

This document defines the Diamond 1 release gate for implemented Nuru Eevee
Hardware RT behavior.

## Supported Validation Target

- Backend: Metal.
- Hardware class: Apple M3+.
- OS class: macOS with Metal ray tracing support.
- OptiX and CUDA Nuru backend support are Work in Progress.
- Apple M1/M2 and non-RTX-class cards cannot be supported due to lack of RT
  cores.

## 1. Backend And Startup

Required:

- Blender starts with the intended macOS build.
- The UI exposes `Nuru Raytracing`.
- The support label reports a detected hardware ray tracing backend on the
  validation device.
- Material Preview remains on the classic screen path.

Fail if:

- Nuru appears supported on an unsupported backend.
- Hardware RT activates partially in Material Preview.
- Startup or shader warmup asserts before a scene can render.

## 2. Per-Feature Ownership

Required:

- GI On/Off preserves diffuse ownership.
- Reflections Off/On preserve classic versus Full RT behavior.
- Refractions Off/On preserve classic versus Full RT behavior.
- Hardware Environment affects explicit environment visibility without
  redefining diffuse GI.
- Hardware Shadows affect Hardware RT shadow visibility while classic shadows
  remain available when off.

Useful probes:

- `tests/python/eevee_hwrt_diffuse_only_gi_probe.py`
- `tests/python/eevee_hwrt_gi_off_environment_probe.py`
- `tests/python/eevee_hwrt_environment_interior_probe.py`
- `tests/python/eevee_hwrt_environment_dome_probe.py`
- `tests/python/eevee_hwrt_specular_parity.py`

Fail if a disabled Hardware RT feature still behaves as active, or if GI
brightens specular-only paths that should not receive diffuse transport.

## 3. Reflection And Refraction

Required:

- Full RT reflection traces secondary geometry and preserves material color.
- Full RT refraction sees textured hits behind refractive surfaces when replay
  data exists.
- Refracted metal/reflective receivers preserve replayed material tint.
- Full RT scene-final specular receivers do not reuse raster/cache radiance as
  visible color.
- Pathtracer-equivalent sharp mirror correctness for Primary and Secondary GI is
  a current limit.

Useful probes:

- `tests/python/eevee_hwrt_reflected_texture_probe.py`
- `tests/python/eevee_hwrt_refracted_texture_probe.py`
- `tests/python/eevee_hwrt_specular_material_color_probe.py`
- `tests/python/eevee_hwrt_principled_dielectric_mirror_probe.py`
- `tests/python/eevee_hwrt_principled_coat_probe.py`
- `tests/python/eevee_hwrt_reflected_gi_probe.py`

Fail if Full RT specular preserves hidden screen ownership, replay data is
ignored, or sharp mirrors reflect cache/screen seams.

## 4. Material Support

Required:

- Diffuse-only material textures are not filtered by roughness.
- Rough GGX reflection/refraction can request material texture filtering.
- Checker texture filtering is continuous when active.
- Proxy fallback remains coarse and fail-closed.

Useful probes:

- `tests/python/eevee_hwrt_specular_parity.py`
- `tests/python/eevee_hwrt_reflected_texture_probe.py`
- `tests/python/eevee_hwrt_refracted_texture_probe.py`
- `tests/python/eevee_hwrt_live_texture_probe.py`

Fail if rough texture filtering is implemented as late radiance blur, or if a
Principled/proxy change breaks unrelated diffuse or composite materials.

## 5. Cache And History

Required:

- Primary and Secondary GI data is used only when Hardware RT, GI, reflections,
  and the matching option are enabled.
- Secondary GI capture stores radiance, position, and normal.
- Secondary GI reuse validates position and normal.
- Secondary GI capture disables HWRT shadow/environment visibility.
- History reuse does not hide real scene edits.

Useful probes:

- `tests/python/eevee_hwrt_camera_rotation_fresh_probe.py`
- `tests/python/eevee_hwrt_animation_response_probe.py`
- `tests/python/eevee_hwrt_render_frame_live_fresh_probe.py`

Fail if secondary GI radiance is used without validation or contains direct-lit
beauty lighting.

## 6. Production Runtime

Required:

- User-selected production scenes render without crashes under the intended Nuru
  feature configuration.
- Effectively black output is classified with evidence as genuinely dark,
  unsupported, or a real regression.
- Runtime blockers are captured in repeatable command output or probe output.

Useful probe:

- `tests/python/eevee_hwrt_production_scene_probe.py`

Fail if a production scene asserts, aborts, or renders unexplained black output.

## 7. Documentation

Required:

- Docs describe implemented Diamond 1 behavior only.
- Probe names in docs correspond to files under `tests/python/`.
- Unsupported cases are explicit.
- Debug variables listed in docs exist in code.

Fail if docs claim a Work in Progress backend as active, describe an
unimplemented control surface as shipped, or cite stale file paths.

## Go / No-Go Rule

Use `Go` only when the required categories pass on the supported Metal backend
and the docs match the implemented runtime.

Use `No-Go` when any required category fails, a production-scene crash is
unexplained, or signoff depends on undocumented caveats.
