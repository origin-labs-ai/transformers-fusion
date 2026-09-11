#!/usr/bin/env bash
# Transcender distribution builder
# Usage: bash scripts/make_dist.sh [version]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# P2 fix: default version was stale "v0.1.0-engine-prod". Derive the one-truth
# from CMakeLists.txt (project(Transcender VERSION x.y.z)) unless overridden.
DEFAULT_VERSION="$(grep -m1 -oE 'VERSION [0-9]+\.[0-9]+\.[0-9]+' CMakeLists.txt | awk '{print $2}')"
VERSION="${1:-v${DEFAULT_VERSION:-1.1.0}}"
OUTDIR="dist"
mkdir -p "$OUTDIR/source"

echo "=== Building distribution: $VERSION ==="

# --- Platform detection ---
case "$(uname -s)" in
    Linux*)  PLAT="linux-x86_64" ;;
    Darwin*) PLAT="macos-arm64"  ;;
    CYGWIN*|MINGW*|MSYS*) PLAT="windows-x64" ;;
    *)       echo "Unknown OS"; exit 1 ;;
esac

echo "Platform: $PLAT"

# --- Build ---
cmake -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DQUANT_BUILD_TESTS=ON -DQUANT_BUILD_BENCHMARKS=ON
cmake --build build-release --parallel

# --- Copy binaries ---
mkdir -p "$OUTDIR/$PLAT"
if [ "$PLAT" = "windows-x64" ]; then
    find build-release -maxdepth 2 -name "*.exe" -exec cp {} "$OUTDIR/$PLAT/" \;
else
    find build-release -maxdepth 2 -type f -executable -exec cp {} "$OUTDIR/$PLAT/" \;
fi

# --- SHA256SUMS ---
cd "$OUTDIR/$PLAT"
rm -f SHA256SUMS
sha256sum * > SHA256SUMS
cd "$ROOT"

# --- Source tarball ---
# P2 fix: must include cmake/ (root CMakeLists does include(arch)/include(compiler)),
# quant_config.h.in (configure_file template), and sops/ (built unconditionally).
tar --exclude='.git' --exclude='build*' --exclude='dist' --exclude='.kilo' \
    --exclude='.research' --exclude='.github' \
    -czf "$OUTDIR/Transcender-$VERSION-source.tar.gz" \
    CMakeLists.txt quant_config.h.in LICENSE README.md \
    cmake/ src/ include/ engines/ tests/ bench/ tools/ sops/

cd "$OUTDIR"
sha256sum "Transcender-$VERSION-source.tar.gz" > "Transcender-$VERSION-source.tar.gz.sha256"
cd "$ROOT"

echo "=== Distribution built at $OUTDIR/ ==="
echo "Binaries: $OUTDIR/$PLAT/"
echo "Source:   $OUTDIR/Transcender-$VERSION-source.tar.gz"
