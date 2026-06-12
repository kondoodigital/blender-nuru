# Nuru Dos And Don'ts

A practical field guide for getting the most out of Nuru Raytracing: how to build materials
that trace beautifully, how to use the new denoiser choice, and which global settings to reach
for first. Everything here reflects how Nuru actually traces your scene.

## Materials

### Glass and windows

- **Do** use the **Thin Glass** BSDF for window panes, glass doors, picture frames, and any
  flat sheet of glass. Thin Glass passes light at full strength, casts no darkening ray-traced
  shadow, and takes its reflection tint directly from its **Color**. Keep **Use Fresnel**
  enabled for natural grazing-angle reflections.
- **Don't** build flat panes from the thick **Glass** or **Refraction** BSDF. A sun-lit room
  behind a refractive pane pays full refraction bounces for geometry that has no thickness to
  refract, and the pane darkens the room for no visual gain.
- **Do** reserve real refraction (**Raytrace Refractions** + a Glass BSDF) for objects with
  body: bottles, lenses, ice cubes, thick table tops. Raise the refraction **Bounces** only
  when light has to pass through several layers, for example a bottle inside a display case.

### Mirrors and metals

- **Do** make dedicated mirror materials: a Glossy BSDF, or a Principled BSDF with
  **Metallic 1.0** and low **Roughness**. Smooth surfaces are traced as true mirrors and show
  the real materials of what they reflect, including image textures and multi-material objects.
- **Don't** mix a faint Glossy lobe into a mostly diffuse material and expect a mirror. When
  diffuse and glossy shading are mixed on one surface, Nuru shades the surface diffuse-first;
  the dedicated mirror path is for surfaces that are genuinely specular.
- **Do** use the roughness dial intentionally: low roughness gives sharp, fast mirror images;
  higher roughness costs more cleanup work from the denoiser. If a surface only needs a soft
  sheen, classic Eevee shading is often all you need - save Full RT reflections for surfaces
  where the mirror image matters.

### Transparency and cutouts

- **Do** use **Transparent BSDF** or Principled **Alpha** for cutout cards: leaves, fences,
  wire mesh, spider webs, decals. Cutouts pass GI and continuation rays with partial coverage
  instead of acting as solid dark blockers, and their shadows soften accordingly.
- **Don't** expect per-texel alpha detail inside reflections and ray-traced shadows. Nuru
  estimates one coverage value per material, so a leaf texture reads as "about half open" to
  the rays rather than leaf-shaped. Keep cutout cards small relative to the frame and nobody
  will ever notice.
- **Do** split heavy alpha work into separate materials: one material for the dense foliage
  card, another for the solid trunk, so the trunk keeps crisp shadows.

### Emission and volumes

- **Do** let emissive materials carry your scene lighting. Emissive surfaces feed Nuru's GI
  and light selection, so neon signs, screens, and light panels genuinely illuminate.
- **Do** use Principled Volume for fog and atmosphere: lit fog, shadowed sun shafts through
  blinds, and world volumes all render in viewport and final frames.
- **Don't** stack many large overlapping volume objects when one world volume would do - every
  overlapping volume step costs froxel work.

## The Denoiser Choice

The **Denoiser** list at the top of the Denoise settings selects the engine for the final
Hardware RT denoise. The pipeline is identical for both; switching is instant and safe.

- **Do** try both on your scene. **OpenImageDenoise** (default) preserves fine texture detail
  and is extremely fast with Nuru's zero-copy path. **OptiX Denoiser** (NVIDIA GPUs) has a
  smoother, softer character that some interiors and volumetric shots prefer.
- **Do** keep **Passes** at **Albedo and Normal**: the guides let either denoiser separate
  noise from texture, which keeps detail and stabilizes animation.
- **Don't** judge a denoiser on the first frame. Pipelines warm up on the first render after
  launch; compare settled viewport or second-render results.
- **Do** raise **Denoise Sample Interval** (2-4) on heavy scenes: samples keep accumulating
  every frame while the full denoise runs only on the interval, which smooths interactivity.
- **Don't** disable **Use GPU** except to diagnose a problem - the CPU path is a fallback,
  not a quality upgrade.

If OptiX is not available on your system, Nuru prints a single console notice and continues
with OpenImageDenoise - your render never stops.

## Global Settings

### Resolution is your biggest lever

- **Do** work the viewport at **50%** Resolution and switch to **75%** or **100%** for finals.
  The Resolution setting scales every traced effect - GI, reflections, refractions - so it is
  the single most effective speed control in Nuru.
- **Don't** push 100% Resolution during lookdev. You are paying full-rate reflections to
  evaluate a material you are still changing.

### Light and GI

- **Do** trust the defaults first: **GI Spatial 16** suits most scenes. Drop to **8** for
  fast exteriors, raise to **32** for smooth dim interiors.
- **Do** use **Indirect Clamp** to tame fireflies from small bright sources reflected in
  glossy corners. A moderate clamp keeps energy believable while killing sparkle.
- **Don't** crank **Indirect Light** strength to rescue a dark room. Add or brighten real
  lights (or emissive surfaces) instead - boosted indirect multiplies noise along with light.
- **Do** use many lights freely. Nuru picks lights intelligently per region, so a bar full of
  practicals or a hall of ceiling fixtures stays efficient.

### Shadows

- **Do** raise shadow **Samples** for large area lights and soft sun - that is exactly what
  the control is for.
- **Do** use **Transparent Shadows** and **Color Transmission** together for tinted glass:
  the first controls how much light passes, the second how much color it carries.
- **Don't** leave Raytrace Shadows on for stylized scenes that want classic shadow maps -
  Nuru features are toggles, not obligations; mix hardware and classic paths per scene.

### Workflow

- **Do** let the viewport settle. Nuru accumulates samples the moment you release the mouse;
  the image refines in place within moments.
- **Don't** benchmark the first F12 after launching Blender. Shader and pipeline warm-up makes
  the first frame slower; every later frame is the real speed.
- **Do** render animation with **Temporal Accumulation** enabled (it is by default) - it is
  the difference between shimmering noise and a calm sequence.

## Related Pages

- [UI Reference](nuru_ui_reference.html)
- [Materials](nuru_materials.html)
- [Shadows And Lights](nuru_shadows_and_lights.html)
- [Nuru Overview](nuru_overview.html)
