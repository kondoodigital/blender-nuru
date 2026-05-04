<!-- SPDX-FileCopyrightText: 2026 Kondoo Digital GmbH -->
<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

# Eevee HWRT Visibility Support

Diamond 1 visibility support is scoped to the active Nuru Hardware RT paths.

Active visibility support is limited to the implemented Hardware RT paths:

- direct and secondary shadow visibility textures;
- environment visibility textures;
- hit, receiver, and secondary GI textures used by reflection/refraction and
  Primary and Secondary GI;
- classic Eevee visibility and light-probe data when a feature remains on the classic path.

## Ownership

`source/blender/draw/engines/eevee/eevee_raytrace.cc` owns the active HWRT visibility orchestration. The active Metal backend remains under `source/blender/gpu/metal/` and provides the ray queries used by shadows, environment visibility, scene-final reflection/refraction, and Primary and Secondary GI support.

## Documentation Boundary

New visibility documentation should be added only for active runtime data that is still allocated, bound, and consumed by the Diamond 1 renderer.
