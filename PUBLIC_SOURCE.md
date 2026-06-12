# Blender-Nuru Public Source

This repository publishes the complete corresponding source for the Blender-Nuru public source snapshot.

- Exact Nuru source commit: `be4bb28fa8ba1855f02d67018c3fe63b9b9b67e1`
- Public source snapshot: `Blender-Nuru 5.1.1-0.9.8 public source snapshot`
- Runtime and build-required source assets, including the macOS prebuilt
  libraries under `lib/macos_arm64/`, are stored directly in Git.
- No Git LFS objects are required for the normal public clone, build, install, or runtime path.
- The upstream Blender developer test corpus under `tests/files/**` is not part of the default public source snapshot.
- The `images/` folder, including the Nuru logo PNG files, is stored as normal Git blobs.

## Recommended Clone

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/kondoodigital/blender-nuru.git Nuru
cd Nuru
git checkout v5.1.1-0.9.8
git lfs install --local --skip-smudge
```

For branch-based development instead of the pinned source tag:

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
Nuru Logo Wide.png
Nuru Logo-4x-upscale-4x.png
Nuru Logo.png
nuru_logo_256.png
nuru_logo_512.png
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

Do not use Git LFS as the public distribution path for Blender-Nuru binaries. Publish binaries through GitHub Releases or another artifact host, and pin each binary release to the exact source tag or commit used to build it. The macOS binary for this snapshot is published as a GitHub Release asset on the `v5.1.1-0.9.8` release.
