#!/usr/bin/env bash
# Compile a .hip file into a build/ folder next to it, then run it on one GPU.
#
# Usage: common/run.sh FILE.hip [DEVICE] [PROGRAM_ARGS...]
#   DEVICE defaults to 0:
#     0 = Radeon AI PRO R9700 (gfx1201)
#     1 = Radeon RX 7900 XTX  (gfx1100)
#     2 = Radeon RX 6900 XT   (gfx1030)
set -euo pipefail

# Full path on purpose: the hipcc in .venv can't find the ROCm device library.
HIPCC=/opt/rocm/bin/hipcc
ARCHES=(gfx1201 gfx1100 gfx1030)

if [[ $# -lt 1 ]]; then
    echo "usage: $0 FILE.hip [DEVICE] [PROGRAM_ARGS...]" >&2
    exit 2
fi

src=$1
device=${2:-0}

if [[ ! -f $src ]]; then
    echo "error: no such file: $src" >&2
    exit 2
fi
if [[ ! $device =~ ^[0-2]$ ]]; then
    echo "error: DEVICE must be 0, 1 or 2, got: $device" >&2
    exit 2
fi

build_dir=$(dirname "$src")/build
out=$build_dir/$(basename "$src" .hip)
mkdir -p "$build_dir"

arch_flags=()
for arch in "${ARCHES[@]}"; do
    arch_flags+=("--offload-arch=$arch")
done

"$HIPCC" "${arch_flags[@]}" "$src" -o "$out"
HIP_VISIBLE_DEVICES=$device "$out" "${@:3}"
