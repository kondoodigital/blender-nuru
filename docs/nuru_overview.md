# Nuru Overview

Nuru is Kondoo Digital's hardware ray tracing mode for Blender **Eevee**. It adds **Nuru Raytracing** as a ray tracing method and gives artists direct controls for global illumination, shadows, reflections, and refractions.

Nuru is built for interactive real-time work. It is not a path tracer, and it is not intended to match Cycles feature-for-feature.

## Platform

| Item | Status |
| --- | --- |
| Packaged backends | Metal on macOS (`macos-metal`), Vulkan on Windows (`windows-vulkan`), and Vulkan on Linux (`linux-vulkan`) |
| Shared branch | `nuru-core` |
| macOS branch | `macos-metal` |
| Windows branch | `windows-vulkan` |
| Linux branch | `linux-vulkan` |
| Recommended macOS hardware | Apple M3 or newer |
| Recommended macOS version | macOS 14 or newer |
| Unsupported Apple GPUs | M1 and M2 do not provide the required hardware RT support |
| Recommended Windows hardware | NVIDIA GeForce RTX (RTX 20 series or newer) with a current driver |
| Recommended Windows version | Windows 10 or 11, 64-bit; see [Nuru On Windows](nuru_windows.html) |
| Recommended Linux hardware | NVIDIA GeForce RTX (RTX 20 series or newer) with the 595-series or newer proprietary driver |
| Recommended Linux version | 64-bit Linux; Ubuntu 22.04+ / Debian 12+ for the `.deb`; see [Nuru On Linux](nuru_linux.html) |

When Nuru is off, or when a specific Nuru feature is off, Blender falls back to the available Eevee behavior for that feature.

## Multi-OS Branch Model

Nuru source is organized so shared behavior and platform-specific backend work do not mix:

| Branch | Contents |
| --- | --- |
| `nuru-core` | Shared Nuru UI, RNA/DNA, materials, Eevee orchestration, docs, and backend-neutral hardware ray tracing API. |
| `macos-metal` | Metal hardware ray tracing implementation, macOS app bundle, and macOS build/test helper changes. |
| `windows-vulkan` | Windows Vulkan implementation branch, based on the shared core. |
| `linux-vulkan` | Linux Vulkan implementation branch, based on the shared core. |

Packaged builds are available for macOS (Metal), Windows (Vulkan, NVIDIA RTX), and Linux (Vulkan, NVIDIA RTX). All three inherit the same shared Nuru UI and renderer behavior from `nuru-core`.

The refreshed macOS, Windows, and Linux 1.0 packages include **Direct Shadow Catcher**.

## Advantages

Nuru is designed for scenes where speed, interactivity, and animation stability matter.

| Advantage | What it means for users |
| --- | --- |
| High-resolution rendering | For 4K and larger output, Nuru can use reduced internal RT resolution while preserving the final image size. The **Resolution** setting applies to every traced effect, including reflections and refractions. |
| Direct Shadow Catcher | Mark a mesh as a receiver-only compositing matte. Its surface disappears while direct sun and local-light shadows remain in transparent Combined RGBA, including soft, alpha-cutout, and tinted-glass attenuation. |
| True materials in reflections | Mirrors and refractive surfaces show the real material of what they reflect: image textures, UV maps, and mixed shader setups stay intact in the reflection. |
| Stable animation frames | Denoising and temporal accumulation can reduce frame-to-frame shimmer compared with low-sample path-traced animation. |
| Fast warmed-up playback and rendering | After the first warm-up, repeated frames can progress quickly because Nuru traces selected effects instead of every possible light path. |
| Viewport-to-final consistency | The Rendered viewport and final render use the same Nuru lighting path, so what you tune in the viewport carries into the saved frame. |
| Selective ray tracing | GI, shadows, reflections, and refractions can be enabled independently, so you only pay for what the scene needs. |
| Volumetrics | Object and world volumes render in viewport and final frames, including lit fog and sun shafts. |

## Basic Workflow

1. Open **Render Properties**.
2. Set the render engine to **Eevee**.
3. Enable **Raytracing**.
4. Set **Method** to **Nuru Raytracing**.
5. Use **Quick Settings** to control the RT features.

Nuru is intended for **Rendered** viewport and final render. **Material Preview** uses Blender's standard preview path.

## Quick Settings Summary

| Control | Role |
| --- | --- |
| **Resolution** | Sets Nuru trace resolution: 100%, 75%, 50%, or 25%. |
| **Global Illumination** | Enables diffuse hardware-traced indirect light. |
| **GI Spatial** | Smooths diffuse GI by reusing nearby information. |
| **Raytrace Reflections** | Enables Full RT reflections. |
| **Reflection Bounces** | Sets how many reflection events can continue. |
| **Raytrace Refractions** | Enables Full RT refractions and transmission. |
| **Refraction Bounces** | Sets how many refraction events can continue. |
| **Raytrace Shadows** | Enables hardware ray-traced shadows. |
| **Samples** | Controls soft shadow quality. |
| **Transparent Shadows** | Controls how transparency affects shadow strength. |
| **Color Transmission** | Controls how strongly transparent materials tint shadows. |
| **Indirect Light** | Adjusts indirect light strength. |
| **Indirect Clamp** | Limits very bright indirect lighting. |
| **Denoise** | Controls the denoising options shown in Quick Settings. |

See [UI Reference](nuru_ui_reference.html) for the full control list.

## Current Limits

- Nuru does not run in **Material Preview**.
- Some complex material graphs may need adjustment for stable RT results.
- Rough reflections soften texture detail by design; lower the trace **Resolution** for speed or keep it at 100% for the sharpest mirrors.

## Related

- [UI Reference](nuru_ui_reference.html)
- [Shadows And Lights](nuru_shadows_and_lights.html)
- [Materials](nuru_materials.html)
- [Nuru On Windows](nuru_windows.html)
