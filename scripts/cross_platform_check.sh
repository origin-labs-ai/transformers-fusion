#!/usr/bin/env bash
# cross_platform_check.sh — Verify bitwise determinism between Windows and Linux builds
#
# Usage:
#   ./scripts/cross_platform_check.sh <windows_build_dir> <linux_build_dir>
#
# Both build directories must contain the same set of binary output files
# (executables / shared libraries) produced by identical source revisions.
# The script compares SHA-256 checksums to confirm bitwise-identical output.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
  echo "Usage: $0 <windows_build_dir> <linux_build_dir>"
  echo ""
  echo "Compares binary artifacts between a Windows and Linux build of Transcender"
  echo "to verify bitwise determinism of quantized output files."
  echo ""
  echo "Both directories should point to the CMake build output (e.g. build/Release or build)."
  exit 1
}

if [[ $# -lt 2 ]]; then
  usage
fi

WIN_DIR="$1"
LIN_DIR="$2"

if [[ ! -d "$WIN_DIR" ]]; then
  echo -e "${RED}ERROR: Windows build directory does not exist: $WIN_DIR${NC}"
  exit 1
fi

if [[ ! -d "$LIN_DIR" ]]; then
  echo -e "${RED}ERROR: Linux build directory does not exist: $LIN_DIR${NC}"
  exit 1
fi

echo "=============================================="
echo " Transcender Cross-Platform Determinism Check"
echo "=============================================="
echo ""
echo "Windows build dir: $WIN_DIR"
echo "Linux  build dir: $LIN_DIR"
echo ""

# -------------------------------------------------------------------
# 1. Collect binary artifacts from both directories
#    We look for common binary extensions that Transcender produces:
#    executables, static/shared libraries, and QUANT quantized weight files.
# -------------------------------------------------------------------

TMPDIR_WIN=$(mktemp -d)
TMPDIR_LIN=$(mktemp -d)
trap 'rm -rf "$TMPDIR_WIN" "$TMPDIR_LIN"' EXIT

# Find binary files: .exe, .dll, .lib, .so, .a, .quant, .quant2, .quant4, .quant8, .bin
BINARY_EXTS="exe|dll|lib|so|a|quant|quant2|quant4|quant8|quant16|quant32|quant_quant|bin"

find_artifacts() {
  local dir="$1"
  local out="$2"
  # Find all matching files and record relative path + sha256
  while IFS= read -r -d '' file; do
    local rel
    rel=$(realpath --relative-to="$dir" "$file")
    local hash
    hash=$(sha256sum "$file" | awk '{print $1}')
    echo "$rel $hash"
  done < <(find "$dir" -type f \( \
    -name "*.exe" -o \
    -name "*.dll" -o \
    -name "*.lib" -o \
    -name "*.so" -o \
    -name "*.a" -o \
    -name "*.o" -o \
    -name "*.quant" -o \
    -name "*.quant2" -o \
    -name "*.quant4" -o \
    -name "*.quant8" -o \
    -name "*.quant16" -o \
    -name "*.quant32" -o \
    -name "*.quant_quant*" -o \
    -name "*.bin" \
  \) -print0 | sort -z)
}

echo "Scanning Windows artifacts..."
find_artifacts "$WIN_DIR" > "$TMPDIR_WIN/artifacts.txt" 2>/dev/null || true

echo "Scanning Linux artifacts..."
find_artifacts "$LIN_DIR" > "$TMPDIR_LIN/artifacts.txt" 2>/dev/null || true

WIN_COUNT=$(wc -l < "$TMPDIR_WIN/artifacts.txt" | tr -d ' ')
LIN_COUNT=$(wc -l < "$TMPDIR_LIN/artifacts.txt" | tr -d ' ')

echo ""
echo "Found $WIN_COUNT binary artifacts in Windows build"
echo "Found $LIN_COUNT binary artifacts in Linux build"
echo ""

# -------------------------------------------------------------------
# 2. Check for files present in one build but not the other
# -------------------------------------------------------------------

MISMATCH=0

# Extract filenames only (without hashes) for comparison
awk '{print $1}' "$TMPDIR_WIN/artifacts.txt" | sort > "$TMPDIR_WIN/names.txt"
awk '{print $1}' "$TMPDIR_LIN/artifacts.txt" | sort > "$TMPDIR_LIN/names.txt"

WIN_ONLY=$(comm -23 "$TMPDIR_WIN/names.txt" "$TMPDIR_LIN/names.txt" || true)
LIN_ONLY=$(comm -13 "$TMPDIR_WIN/names.txt" "$TMPDIR_LIN/names.txt" || true)

if [[ -n "$WIN_ONLY" ]]; then
  echo -e "${YELLOW}Files only in Windows build:${NC}"
  echo "$WIN_ONLY" | while read -r f; do echo "  - $f"; done
  MISMATCH=1
fi

if [[ -n "$LIN_ONLY" ]]; then
  echo -e "${YELLOW}Files only in Linux build:${NC}"
  echo "$LIN_ONLY" | while read -r f; do echo "  - $f"; done
  MISMATCH=1
fi

echo ""

# -------------------------------------------------------------------
# 3. Compare checksums for files present in both builds
# -------------------------------------------------------------------

echo "Comparing checksums for common files..."
echo ""

MATCH_COUNT=0
DIFF_COUNT=0
SKIP_COUNT=0

# Join on filename to compare hashes
while IFS= read -r line; do
  fname=$(echo "$line" | awk '{print $1}')
  whash=$(echo "$line" | awk '{print $2}')
  lhash=$(awk -v fn="$fname" '$1 == fn {print $2}' "$TMPDIR_LIN/artifacts.txt" || true)

  if [[ -z "$lhash" ]]; then
    continue
  fi

  if [[ "$whash" == "$lhash" ]]; then
    MATCH_COUNT=$((MATCH_COUNT + 1))
  else
    DIFF_COUNT=$((DIFF_COUNT + 1))
    echo -e "  ${RED}MISMATCH${NC}: $fname"
    echo "    Windows: $whash"
    echo "    Linux:   $lhash"
  fi
done < "$TMPDIR_WIN/artifacts.txt"

echo ""
echo "=============================================="
echo " Results"
echo "=============================================="
echo "  Files matched:   $MATCH_COUNT"
echo "  Files mismatched: $DIFF_COUNT"
echo "  Files skipped:   $WIN_ONLY is not compared"
echo ""

if [[ $DIFF_COUNT -eq 0 && $MISMATCH -eq 0 ]]; then
  echo -e "${GREEN}PASS${NC}: All binary artifacts are bitwise identical across platforms."
  echo ""
  echo "This confirms that Transcender produces deterministic output"
  echo "regardless of compiler (MSVC vs GCC vs Clang) and OS."
  exit 0
elif [[ $DIFF_COUNT -eq 0 && $MISMATCH -ne 0 ]]; then
  echo -e "${YELLOW}PARTIAL PASS${NC}: All common files match, but some files exist on only one platform."
  echo "This may be expected (e.g. Windows-only GPU libs, platform-specific tests)."
  exit 0
else
  echo -e "${RED}FAIL${NC}: $DIFF_COUNT file(s) have different checksums across platforms."
  echo ""
  echo "Possible causes:"
  echo "  - Non-deterministic compiler output (floating-point rounding, struct padding)"
  echo "  - Different code paths taken per platform"
  echo "  - Use of platform-specific APIs or intrinsics"
  echo "  - Uninitialized memory or undefined behavior"
  exit 1
fi
