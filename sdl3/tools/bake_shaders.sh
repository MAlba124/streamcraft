#!/usr/bin/env bash
# Bake the owned video render pipeline's GLSL shaders to SPIR-V blobs.
#
# The runtime uses `include_bytes!` on the committed .spv files (no runtime shader
# compiler ships), exactly like the scope UI's committed font atlas. Regenerate the
# blobs whenever `shaders/video.vert` / `shaders/video.frag` change:
#
#     nix develop --command sdl3/tools/bake_shaders.sh
#
# (glslang / glslangValidator is provided by the devshell — see flake.nix.) The .spv
# outputs land next to the sources in sdl3/shaders/ and MUST be committed.
#
# -V selects Vulkan SPIR-V (the format SDL_GPU consumes via SDL_GPU_SHADERFORMAT_SPIRV
# on the Vulkan backend). We pass the stage explicitly (-S) since the sources have
# non-standard extensions in some checkouts; here the .vert/.frag extensions already
# imply the stage, but -S keeps it deterministic.
set -euo pipefail

# Resolve the shaders dir relative to this script (repo-root-independent).
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
shaders="$here/../shaders"

glslangValidator -V -S vert "$shaders/video.vert" -o "$shaders/video.vert.spv"
glslangValidator -V -S frag "$shaders/video.frag" -o "$shaders/video.frag.spv"

echo "baked: video.vert.spv video.frag.spv"
