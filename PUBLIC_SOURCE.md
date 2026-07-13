# Blender-Nuru Public Source

This repository publishes the complete corresponding source for the Blender-Nuru public source snapshot.

- Exact Nuru source commit (macOS binary): `0e9b94f620d` (`RUBY 49`, release `v5.1.1-1.0`)
- Exact Nuru source commit (Windows binary): `98366ee9445` (`EMERALD 40`, release `v5.1.1-1.0`)
- Exact Nuru source commit (Linux binary): `965b15b7896` (`SAPPHIRE 61`, release `v5.1.1-1.0`)
- Public source snapshot: `Blender-Nuru 5.1.1-0.9.8 public source snapshot` plus the
  `Blender-Nuru 5.1.1-0.9.8 Windows public source update` (tag `v5.1.1-0.9.8-windows`) and the
  `Blender-Nuru 5.1.1-0.9.8 Linux public source update` (tag `v5.1.1-0.9.8-linux`), updated by
  the `Blender-Nuru 5.1.1-1.0 Linux public source update` (tag `v5.1.1-1.0-linux`)
  the `Blender-Nuru 5.1.1-1.0 Windows public source update` (tag `v5.1.1-1.0-windows`),
  and the `Blender-Nuru 5.1.1-1.0 macOS public source update`, refreshed by
  `Blender-Nuru 5.1.1-1.0 macOS Shadow Catcher source refresh`
  (tag `v5.1.1-1.0-macos`), then the
  `Blender-Nuru 5.1.1-1.0 Windows Shadow Catcher source refresh`
  (tag `v5.1.1-1.0-windows`), and the
  `Blender-Nuru 5.1.1-1.0 Linux Shadow Catcher source refresh`
  (tag `v5.1.1-1.0-linux`).
- Runtime and build-required source assets, including the macOS prebuilt
  libraries under `lib/macos_arm64/`, are stored directly in Git.
- The Windows prebuilt libraries are not stored in this repository; they are fetched from the
  upstream Blender mirror during setup (see Windows Developer Build below).
- No Git LFS objects are required for the normal public clone, build, install, or runtime path.
- The upstream Blender developer test corpus under `tests/files/**` is not part of the default public source snapshot.
- The `images/` folder, including the Nuru logo PNG files, is stored as normal Git blobs.

## Recommended Clone

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/kondoodigital/blender-nuru.git Nuru
cd Nuru
git checkout v5.1.1-1.0-macos
git lfs install --local --skip-smudge
```

Use `git checkout v5.1.1-1.0-macos` for the exact source of the current macOS binary,
`git checkout v5.1.1-1.0-windows` for the exact source of the current Windows binary, or
`git checkout v5.1.1-1.0-linux` for the exact source of the current Linux binary
(`v5.1.1-0.9.8-windows` and `v5.1.1-0.9.8-linux` remain for previous binaries).

For branch-based development instead of the pinned source tags:

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/kondoodigital/blender-nuru.git Nuru
cd Nuru
git checkout main
git lfs install --local --skip-smudge
```

The `GIT_LFS_SKIP_SMUDGE=1` and `--skip-smudge` settings are kept in the instructions so accidental LFS downloads do not become part of the default public workflow.

## Nuru Images And Logo

The public source snapshot includes the Nuru image assets as normal Git files:

```sh
ls images/
```

Expected logo files include:

```text
images/nuru_logo.png
images/nuru_logo_256.png
images/nuru_logo_4x_upscale_4x.png
images/nuru_logo_512.png
images/nuru_logo_wide.png
```

These files are not Git LFS pointers and do not require a Git LFS download.

## Developer Build

On macOS, use a repo-local development build directory:

```sh
cd Nuru
make developer ninja BUILD_DIR=builds/macos-dev
```

Launch the built application from the build tree:

```sh
builds/macos-dev/bin/Blender-Nuru.app/Contents/MacOS/Blender-Nuru
```

For direct CMake usage, configure and build the same development tree explicitly:

```sh
cmake -S . -B builds/macos-dev -G Ninja -C build_files/cmake/config/blender_developer.cmake
cmake --build builds/macos-dev --target install
```

## Windows Developer Build

Prerequisites: Visual Studio 2022 Build Tools (MSVC v143 and a Windows 10/11 SDK), CMake, Ninja, and Git.

Fetch the upstream Windows prebuilt libraries into the source tree:

```sh
cd Nuru
git clone --depth 1 https://projects.blender.org/blender/lib-windows_x64.git lib/windows_x64
```

Configure and build from a Visual Studio x64 developer prompt. The Windows build uses the Vulkan backend only, so `WITH_OPENGL_BACKEND=OFF` is required:

```sh
cmake -S . -B builds/dev -G Ninja -C build_files/cmake/config/blender_developer.cmake -DWITH_OPENGL_BACKEND=OFF
cmake --build builds/dev --target install
```

Launch the built application from the build tree:

```sh
builds/dev/bin/Blender-Nuru.exe
```

Optional: Cycles GPU kernels for NVIDIA need the CUDA Toolkit (`WITH_CYCLES_CUDA_BINARIES=ON`) and the OptiX SDK (`OPTIX_ROOT_DIR`). They are not required for Nuru hardware ray tracing in Eevee.

## Linux Developer Build

Prerequisites: GCC or Clang, CMake, Ninja, and Git on a 64-bit Linux distribution.

Fetch the upstream Linux prebuilt libraries into the source tree:

```sh
cd Nuru
git clone --depth 1 https://projects.blender.org/blender/lib-linux_x64.git lib/linux_x64
```

Configure and build. The Linux build uses the Vulkan backend only, so `WITH_OPENGL_BACKEND=OFF` is required:

```sh
cmake -S . -B builds/dev -G Ninja -C build_files/cmake/config/blender_developer.cmake -DWITH_OPENGL_BACKEND=OFF
cmake --build builds/dev --target install
```

Launch the built application from the build tree:

```sh
builds/dev/bin/blender-nuru
```

Nuru hardware ray tracing at runtime needs an NVIDIA RTX GPU with the proprietary driver, 595 series or newer. Optional: Cycles GPU kernels for NVIDIA need the CUDA Toolkit (`WITH_CYCLES_CUDA_BINARIES=ON`) and the OptiX SDK headers (`OPTIX_ROOT_DIR`, for example from the redistributable `github.com/NVIDIA/optix-dev` checkout). They are not required for Nuru hardware ray tracing in Eevee.

## Developer Test Assets

The `tests/files/**` corpus is for Blender developer validation parity. It is not required to build, install, or run the shipped Blender-Nuru binary and is intentionally excluded from the default public source snapshot. Normal developers can compile Blender-Nuru without these assets.

Developers who want to run the Blender test suite should fetch test assets intentionally into a separate checkout, then copy only `tests/files/` into the Nuru source tree:

```sh
cd ..
GIT_LFS_SKIP_SMUDGE=1 git clone https://projects.blender.org/blender/blender.git blender-test-assets
cd blender-test-assets
git lfs install --local
git lfs pull --include="tests/files/**" --exclude=""
rsync -a tests/files/ ../Nuru/tests/files/
```

If you maintain a dedicated Nuru or Blender test-data mirror, use that mirror URL instead of the Blender source URL above. Do not fetch these assets from GitHub LFS on this public repository.

After copying the optional test assets, run tests from the Nuru checkout:

```sh
cd ../Nuru
ctest --test-dir builds/macos-dev
```

## Binary Distribution

Do not use Git LFS as the public distribution path for Blender-Nuru binaries. Publish binaries through GitHub Releases or another artifact host, and pin each binary release to the exact source tag or commit used to build it. The macOS DMG and zip, the Windows installer and zip, and the Linux `.deb` and `.tar.gz` are all published on the `v5.1.1-1.0` release and correspond to the `v5.1.1-1.0-macos`, `v5.1.1-1.0-windows`, and `v5.1.1-1.0-linux` source tags respectively.

The refreshed macOS, Windows, and Linux 1.0 packages include Direct Shadow Catcher.
