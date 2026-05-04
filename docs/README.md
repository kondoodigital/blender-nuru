<!-- SPDX-FileCopyrightText: 2026 Kondoo Digital GmbH -->
<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

# Documentation

These documents describe the implemented Nuru state in the `DIAMOND 1` branch.
They are current-state notes for the Blender Eevee Hardware RT work in this
repository, with Work in Progress items labeled as such.

## Current Implemented Scope

- Nuru is an Eevee Hardware RT branch focused on Apple Metal ray tracing.
- The supported viewport backend gate is Metal on macOS with hardware ray
tracing support on Apple M3+ hardware.
- OptiX and CUDA Nuru backend support are Work in Progress.
- Apple M1/M2 and non-RTX-class cards cannot be supported due to lack of RT
  cores.
- The user-facing ray tracing method is `Nuru Raytracing`.
- Hardware RT is per-feature: GI, shadows, reflections, refractions, and
environment visibility can be owned independently.
- Classic Eevee behavior remains the fallback when Nuru Hardware RT is disabled,
unsupported, or a specific feature toggle is off.

## Document Map

- `eevee_hwrt_first_release.md` - implemented Diamond 1 runtime contract and
first-release support envelope.
- `eevee_hwrt_ship_criteria.md` - current release gate and validation checklist
grounded in implemented probes.
- `eevee_hwrt_system_interactions.md` - ownership rules between GI, shadows,
environment, reflections, refractions, scene-final replay, and classic Eevee.
- `eevee_hwrt_visibility_support.md` - implemented visibility-support data for
the current Hardware RT paths.
- `eevee_hwrt_artifact_triage.md` - current debug variables, probe categories,
and evidence-first triage workflow.
- `eevee_rough_material_texture_filter.md` - implemented rough material texture
filtering for visible and HWRT hit-eval material replay.
- `principled_hwrt_audit.md` - current Principled BSDF support, simplifications,
and HWRT replay/proxy limits.
- `shared_repo_workflow.md` - safe shared-source workflow and per-OS build-root
convention for this repository.
