<!-- SPDX-FileCopyrightText: 2026 Kondoo Digital GmbH -->
<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

<div align="center">
  <img src="images/nuru_logo_512.png" alt="Nuru, hardware ray tracing for Blender Eevee by Kondoo Digital" width="512">

  <p>
    <a href="https://vimeo.com/1195782891" title="Blender Nuru Demo"><img src="https://i.vimeocdn.com/video/2161734263-565930f305f1aeb3a15054953f9fa840c701b51988f221ead59ff318abd881f6-d_1280" alt="Blender Nuru Demo video" width="720"></a>
    <br>
    <a href="https://vimeo.com/1195782891"><strong>&#9654; Watch the Blender Nuru demo on Vimeo</strong></a>
  </p>

  <p><strong>Hardware ray tracing for Blender Eevee, built for responsive real-time work.</strong></p>
  <p><em>Nuru</em> means <strong>light</strong> in Swahili.</p>
  <p>A Blender branch by <strong>Kondoo Digital GmbH</strong>.</p>

  <p>
    <a href="https://kondoodigital.github.io/blender-nuru/"><strong>Documentation</strong></a>
  </p>

  <table>
    <tr>
      <th align="center" width="33%">Windows</th>
      <th align="center" width="33%">macOS</th>
      <th align="center" width="33%">Linux</th>
    </tr>
    <tr>
      <td align="center">NVIDIA RTX &middot; Vulkan</td>
      <td align="center">Apple Silicon M3+ &middot; Metal</td>
      <td align="center">NVIDIA RTX &middot; Vulkan</td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/kondoodigital/blender-nuru/releases/download/v5.1.1-0.9.8/Blender-Nuru-windows-5.1.1-0.9.8.exe"><strong>Installer 5.1.1-0.9.8</strong></a><br/><a href="https://github.com/kondoodigital/blender-nuru/releases/download/v5.1.1-0.9.8/Blender-Nuru-windows-5.1.1-0.9.8.zip">Portable Zip</a></td>
      <td align="center"><a href="https://github.com/kondoodigital/blender-nuru/releases/download/v5.1.1-0.9.8/Blender-Nuru-macos-5.1.1-0.9.8.zip"><strong>Download 5.1.1-0.9.8</strong></a></td>
      <td align="center"><a href="https://github.com/kondoodigital/blender-nuru/releases/download/v5.1.1-0.9.8/Blender-Nuru-linux-5.1.1-0.9.8.deb"><strong>Deb Package 5.1.1-0.9.8</strong></a><br/><a href="https://github.com/kondoodigital/blender-nuru/releases/download/v5.1.1-0.9.8/Blender-Nuru-linux-5.1.1-0.9.8.tar.gz">Portable Tar</a></td>
    </tr>
  </table>

  <p>The builds are unsigned: see <a href="#install-on-windows">Install On Windows</a> and <a href="#install-on-macos">Install On macOS</a> below, or the platform pages on the documentation site.</p>
</div>

## Nuru

**Nuru** adds a **Nuru Raytracing** method to Blender Eevee. Artists choose which parts of the render use hardware ray tracing — global illumination, shadows, reflections, and refractions — and pay only for what they enable.

Nuru is a hybrid real-time renderer built for interactive viewport work and fast, stable animation frames. It is not a path tracer and does not replace Cycles; the **Rendered** viewport and the final render share the same Nuru lighting path, so what you see while working is what you get.

## Features

- **Selective hardware ray tracing** — enable GI, shadows, reflections, and refractions independently.
- **True materials in reflections** — mirrors and refractive surfaces show the real material of what they reflect: image textures, UV maps, mixed shaders, and multi-material objects stay intact.
- **Thin Glass** — window panes pass light at full strength and never darken a sun-lit room.
- **One Resolution control** — the trace resolution (100/75/50/25%) applies to every traced effect; 50% is fast for lookdev, 100% gives the sharpest mirrors.
- **Volumetrics** — object and world volumes with lit fog and sun shafts, in viewport and final frames.
- **Animation stability** — integrated denoising and temporal accumulation keep sequences calm at low sample counts.
- **Denoiser choice** — OpenImageDenoise or the NVIDIA OptiX denoiser (Windows), both zero-copy in the same pipeline.
- **Many lights welcome** — light selection scales to scenes full of practicals and fixtures.

## Install On Windows

**Installer (recommended):** download and run `Blender-Nuru-windows-<version>.exe`. The wizard shows the GPL license, lets you pick the install folder, and creates Start Menu and Desktop shortcuts. Uninstall any time from **Settings > Apps**. If SmartScreen warns about an unrecognized app, choose **More info > Run anyway**.

**Portable zip:**

1. Download the Windows zip and unzip it anywhere you like, for example `C:\Programs\Blender-Nuru`.
2. Start `Blender-Nuru.exe`, or `Blender-Nuru-launcher.exe` to start without a console window.

Full Windows notes: [Nuru On Windows](https://kondoodigital.github.io/blender-nuru/docs/nuru_windows.html).

## Install On macOS

1. Download the macOS zip, unzip it, and move `Blender-Nuru.app` to `/Applications`.
2. macOS marks downloaded apps with a quarantine flag, and Blender-Nuru is not distributed through Apple notarization. Clear the flag once in **Terminal**:

```sh
xattr -dr com.apple.quarantine /Applications/Blender-Nuru.app
```

3. Start **Blender-Nuru** from Applications or Launchpad.

Full macOS notes: [Nuru On macOS](https://kondoodigital.github.io/blender-nuru/docs/nuru_macos.html).

## Documentation

Guides, the UI reference, materials and Dos And Don'ts, and the platform pages for Windows, macOS, and Linux live on the documentation site:

**[kondoodigital.github.io/blender-nuru](https://kondoodigital.github.io/blender-nuru/)**

The same pages ship inside every download under `docs/`.

## Upstream Blender Links

- [Blender Website](https://www.blender.org)
- [Blender Manual](https://docs.blender.org/manual/en/latest/index.html)
- [Blender Developer Documentation](https://developer.blender.org/docs/)

## License

Nuru documentation and Kondoo Digital additions are provided under Blender's GPL license. Blender as a whole is licensed under the GNU General Public License. See [blender.org/about/license](https://www.blender.org/about/license) for details.
