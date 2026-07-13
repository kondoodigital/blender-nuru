# Nuru Materials

Nuru works with Eevee materials and adds hardware ray tracing for selected lighting paths. Most ordinary materials work as expected, but complex material node setups should be checked in **Rendered** viewport.

## Reflections

Enable **Raytrace Reflections** when reflective materials need hardware ray tracing.

Reflection quality depends on the material roughness, scene content, and the **Bounces** value. Higher bounce counts can show deeper mirror chains, but they cost more time.

## Materials Seen In Reflections

Surfaces visible inside mirrors and refractive objects are shaded with their real material:

- Image textures and UV maps stay intact in the reflection.
- Color-processing nodes (Mix, Overlay, and similar) between a texture and the shader are respected.
- Mixed shader setups such as **Mix Shader (Diffuse, Glossy)** show their textured diffuse appearance in mirrors.
- Objects with multiple material slots reflect each material correctly.

The reflection's sharpness follows the trace **Resolution** setting: 100% gives the sharpest mirror detail, lower values trade detail for speed.

## Refractions And Transmission

Enable **Raytrace Refractions** for glass, refractive materials, and transmission effects.

The **Bounces** value controls how far refraction can continue through layered transparent or refractive surfaces. If glass looks incomplete, try increasing the bounce count before changing the material.

## Transparent Shadows

Transparent and tinted materials can affect shadows when **Raytrace Shadows** is enabled.

| Control | Material effect |
| --- | --- |
| **Transparent Shadows** | Controls how much transparency changes the shadow strength. |
| **Color Transmission** | Controls how much material color appears in the shadow tint. |

These controls are especially useful for colored glass, transparent textures, and alpha materials.

## Thin Glass

The **Thin Glass BSDF** renders single-pane glass (windows, display cases) without refraction offsets. Thin Glass never casts ray-traced shadows: light passes through panes at full strength, so window glass does not darken a sun-lit room.

## Practical Material Expectations

- Nuru is based on Eevee material behavior, not Cycles material parity.
- Simple Principled, glass, glossy, transparent, and textured materials are the best starting point.
- Very complex layered node graphs may need simplification for predictable RT results.
- Some advanced material features can look different from Cycles.
- Check reflective and refractive materials in **Rendered** viewport, not **Material Preview**.

## Rough Materials

Rough reflective and refractive materials may soften texture detail in the RT result. This is expected: rough surfaces spread reflected or transmitted detail instead of showing a sharp texture copy.

## Volumes

Volumetric materials and world volumes render in Nuru: object fog, world atmosphere, lit fog with shadowed light shafts, and volume absorption all work in the Rendered viewport and final frames.

## Related Pages

- [UI Reference](nuru_ui_reference.html)
- [Nuru Overview](nuru_overview.html)
- [Shadows And Lights](nuru_shadows_and_lights.html)
