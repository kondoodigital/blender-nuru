# Nuru UI Reference

This page explains the Nuru controls shown in Blender's **Render Properties**.

## Where To Find Nuru

1. Open **Render Properties**.
2. Set the render engine to **Eevee**.
3. Enable **Raytracing**.
4. Set **Method** to **Nuru Raytracing**.

When the current device supports Nuru, the panel shows the Nuru logo, hardware support status, and **Quick Settings**.

## Raytracing

| Control | What it does |
| --- | --- |
| **Raytracing** | Turns the Eevee ray tracing panel on or off. |
| **Method** | Choose **Nuru Raytracing** to use Nuru. Other methods use Blender's standard Eevee paths. |
| **Resolution** | Sets the trace resolution for Nuru work: 100%, 75%, 50%, or 25%. Lower values render faster with less detail. The scale applies to all traced effects, including reflections and refractions. |

## Global Illumination

| Control | What it does |
| --- | --- |
| **Global Illumination** | Enables hardware-traced diffuse indirect light. Turn it off when you want classic Eevee behavior for indirect lighting. |
| **GI Spatial** | Controls spatial reuse for diffuse GI. Higher values can look smoother, but cost more time. |

## Reflections

| Control | What it does |
| --- | --- |
| **Raytrace Reflections** | Enables Full RT reflections. Use it for mirrors, glossy surfaces, and reflective materials that need hardware ray tracing. |
| **Bounces** | Sets how many reflection events can continue. Higher values can show deeper mirror chains at higher cost. |

## Refractions

| Control | What it does |
| --- | --- |
| **Raytrace Refractions** | Enables Full RT refractions and transmission. Use it for glass and transparent refractive materials. |
| **Bounces** | Sets how many refraction or transmission events can continue. Higher values can help layered glass, but cost more time. |

## Shadows

| Control | What it does |
| --- | --- |
| **Raytrace Shadows** | Enables hardware ray-traced shadow visibility for lights. |
| **Samples** | Controls soft shadow quality. Higher values reduce noise and cost more time. |
| **Transparent Shadows** | Controls how much transparent materials affect shadow strength. At 0, shadows behave more opaque. At 1, transparency is fully considered. |
| **Color Transmission** | Controls how strongly transparent or tinted materials color their shadows. At 0, tint is reduced. At 1, tint is preserved. |

## Shadow Catcher

Select a mesh receiver, open **Object Properties → Visibility → Mask**, and enable
**Shadow Catcher**. Nuru hides the receiver surface and keeps only direct sun and local-light
shadows in the Combined RGBA result. Unshadowed receiver pixels are transparent; soft,
alpha-cutout, and glass-tinted shadows retain their partial opacity and visible tint.

The Nuru catcher is receiver-only. It does not cast shadows, appear in reflections or
refractions, or contribute material shading, emission, ambient occlusion, caustics, or indirect
light. Enable **Film → Transparent** for a transparent render intended for compositing.

The refreshed macOS, Windows, and Linux 1.0 packages include this control.

## Indirect Light

| Control | What it does |
| --- | --- |
| **Indirect Light** | Adjusts the strength of indirect lighting in the scene. |
| **Indirect Clamp** | Limits very bright indirect lighting to reduce fireflies and unstable highlights. |

## Denoise

| Control | What it does |
| --- | --- |
| **Denoiser** | Chooses the denoise engine: **OpenImageDenoise** (default) or **OptiX Denoiser** (NVIDIA GPUs). Both run in the same pipeline; OptiX falls back to OpenImageDenoise when unavailable. |
| **Passes** | Chooses which extra information the denoiser can use. More guidance can improve stability. |
| **Prefilter** | Sets how the denoiser prepares its inputs before denoising. |
| **Quality** | Balances denoising quality and performance. |
| **Denoise Sample Interval** | Controls how often denoising updates while samples accumulate. |
| **Use GPU** | Uses GPU acceleration for denoising when available. |
| **Spatial Reuse** | Reuses nearby image information to reduce noise. |
| **Temporal Accumulation** | Reuses information across samples over time for a steadier viewport. |

## Volumes

The standard Eevee volume controls work with Nuru: object and world volumes, lit fog, and volume shadows render in viewport and final frames.

## Related Pages

- [Nuru Overview](nuru_overview.html)
- [Shadows And Lights](nuru_shadows_and_lights.html)
- [Materials](nuru_materials.html)
