#!/usr/bin/env bash
# scripts/coverage.sh — Build with coverage flags, run all tests, generate lcov+HTML report
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build_coverage"
REPORT_DIR="${BUILD_DIR}/coverage"
HTML_DIR="${REPORT_DIR}/html"

echo "=== Transcender Code Coverage ==="
echo "Project : ${PROJECT_DIR}"
echo "Build   : ${BUILD_DIR}"
echo "Report  : ${HTML_DIR}"
echo ""

# ── 1. Configure with coverage flags ──────────────────────────────
echo "[1/4] Configuring CMake with -DQUANT_COVERAGE=ON ..."
cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DQUANT_COVERAGE=ON \
  -DQUANT_BUILD_TESTS=ON \
  -DQUANT_BUILD_BENCHMARKS=OFF \
  -DQUANT_BUILD_TOOLS=OFF

# ── 2. Build ─────────────────────────────────────────────────────
echo "[2/4] Building ..."
cmake --build "${BUILD_DIR}" -j "$(nproc 2>/dev/null || echo 4)"

# ── 3. Run all tests ─────────────────────────────────────────────
echo "[3/4] Running tests ..."
cd "${BUILD_DIR}"
ctest --output-on-failure -j "$(nproc 2>/dev/null || echo 4)"

# ── 4. Generate coverage report ──────────────────────────────────
echo "[4/4] Generating coverage report ..."

LCOV="$(command -v lcov || true)"
GENHTML="$(command -v genhtml || true)"

if [ -z "$LCOV" ] || [ -z "$GENHTML" ]; then
  echo "ERROR: lcov and/or genhtml not found. Install lcov:"
  echo "  Ubuntu/Debian: sudo apt install lcov"
  echo "  macOS: brew install lcov"
  exit 1
fi

mkdir -p "${REPORT_DIR}"

# Zero previous counters
${LCOV} --directory "${BUILD_DIR}" --zerocounters 2>/dev/null || true

# Capture fresh counters
${LCOV} --directory "${BUILD_DIR}" \
  --capture \
  --output-file "${REPORT_DIR}/coverage_raw.info"

# Remove unwanted paths (system headers, deps, test/bench/tool sources)
${LCOV} --remove "${REPORT_DIR}/coverage_raw.info" \
  '/usr/*' '/opt/*' \
  "${BUILD_DIR}/_deps/*" \
  '*/tests/*' '*/bench/*' '*/tools/*' \
  --output-file "${REPORT_DIR}/coverage.info"

# Generate HTML
${GENHTML} "${REPORT_DIR}/coverage.info" \
  --output-directory "${HTML_DIR}" \
  --title "Transcender Code Coverage" \
  --legend \
  --quiet

# Summary
echo ""
echo "=== Coverage Summary ==="
${LCOV} --summary "${REPORT_DIR}/coverage.info" 2>&1 || true
echo ""
echo "HTML report: ${HTML_DIR}/index.html"
echo "Raw info   : ${REPORT_DIR}/coverage.info"
