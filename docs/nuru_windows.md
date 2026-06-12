# Nuru On Windows

Blender-Nuru for Windows runs Eevee on the Vulkan graphics backend with Nuru hardware ray tracing on NVIDIA RTX GPUs. This page covers requirements, installation, and Windows-specific notes.

## Requirements

| Item | Requirement |
| --- | --- |
| Operating system | Windows 10 or Windows 11, 64-bit |
| GPU | NVIDIA GeForce RTX class with hardware ray tracing (RTX 20 series or newer); validated on RTX 50 series |
| GPU driver | A current NVIDIA Game Ready or Studio driver |
| Graphics backend | Vulkan, built into the package; no extra setup is needed |
| Other GPUs | AMD and Intel GPUs are not yet validated for Nuru hardware ray tracing |

## Install

1. Download the Windows zip from the [Releases page](https://github.com/kondoodigital/blender-nuru/releases).
2. Unzip it anywhere you like, for example `C:\Programs\Blender-Nuru`. No installer is needed.
3. Start **`Blender-Nuru.exe`**. A console window opens alongside the app and shows log output. To start without the console window, use `Blender-Nuru-launcher.exe` instead.

### Windows SmartScreen

The build is not code-signed, so the first start may show **"Windows protected your PC"**. Click **More info**, then **Run anyway**.

## Enable Nuru

1. Open **Render Properties**.
2. Set the render engine to **Eevee**.
3. Enable **Raytracing**.
4. Set **Method** to **Nuru Raytracing**.
5. Use **Quick Settings** to enable the RT features you need.

The Nuru **Quick Settings** appear when the active GPU supports the Nuru backend. If they stay hidden, update the NVIDIA driver and confirm the RTX GPU is the active render device.

## GPU Denoising

Denoising on Windows runs on the GPU, and the denoiser shares its working memory directly with the renderer (zero-copy). The **Denoise** option is fast at every trace resolution and is safe to keep enabled for interactive work and final renders.

## Cycles On Windows

This package also includes Cycles GPU kernels for NVIDIA **CUDA** and **OptiX**. Enable them under **Edit > Preferences > System > Cycles Render Devices**.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| Nuru Quick Settings do not appear | Update the NVIDIA driver and confirm an RTX GPU is active. |
| The app does not start, or reports a device error | Update the NVIDIA driver and install pending Windows updates, then try again. |
| The first rendered frame is slow | Shaders and pipelines warm up on the first render after installation; later frames are much faster. |
| SmartScreen blocks the app | Click **More info**, then **Run anyway**. The build is unsigned. |

## Related

- [Nuru Overview](nuru_overview.html)
- [UI Reference](nuru_ui_reference.html)
- [Shadows And Lights](nuru_shadows_and_lights.html)
- [Materials](nuru_materials.html)
