# Nuru User Documentation

These documents explain how to use Nuru in Blender Eevee. They focus on visible controls, expected results, and current user-facing limits.

## Start Here

| Document | What it covers |
| --- | --- |
| [Nuru Overview](nuru_overview.html) | What Nuru is, supported hardware, the multi-OS branch model, main workflow, and limits. |
| [UI Reference](nuru_ui_reference.html) | Every visible Nuru Quick Settings control and what it changes. |
| [Shadows And Lights](nuru_shadows_and_lights.html) | Raytrace Shadows, shadow samples, Transparent Shadows, and Color Transmission. |
| [Materials](nuru_materials.html) | Material behavior, reflections, refractions, transparent materials, and practical limits. |
| [Dos And Don'ts](nuru_dos_and_donts.html) | Practical guide: material recipes, the denoiser choice, and the global settings that matter most. |
| [Nuru On macOS](nuru_macos.html) | macOS requirements, installation, Gatekeeper notes, and Apple Silicon support. |
| [Nuru On Windows](nuru_windows.html) | Windows requirements, installation, SmartScreen notes, GPU denoising, and Cycles CUDA/OptiX support. |

## Quick Path

Open **Render Properties**, set the engine to **Eevee**, enable **Raytracing**, then choose **Method → Nuru Raytracing**. The Nuru **Quick Settings** controls appear when the current hardware supports the Nuru backend.

Volumetrics, ray-traced GI, shadows, reflections, and refractions all render in the **Rendered** viewport and final frames.

## Multi-OS Source Branches

Nuru development is split across shared and platform-specific branches:

| Branch | Role |
| --- | --- |
| `nuru-core` | Shared Nuru renderer, UI, material, documentation, and backend-neutral hardware ray tracing API. |
| `macos-metal` | Current macOS Metal implementation and macOS packaging. |
| `windows-vulkan` | Windows Vulkan implementation branch. |
| `linux-vulkan` | Linux Vulkan implementation branch. |

The visible Nuru controls documented here are shared user-facing behavior. Platform-specific renderer code belongs on the matching OS branch.
