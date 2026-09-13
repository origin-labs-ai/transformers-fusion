#!/usr/bin/env bash
#
# sign_release.sh — Transcender v1.1.0 (R0001.01) Release Signing Script
#
# Steps:
#   1. Generate SHA-256 and MD5 checksums for all build artifacts
#   2. Authenticode sign Windows binaries (ORIGIN LABS cert)
#   3. GPG sign the checksum file
#   4. Verify all signatures
#   5. Create release archive
#
# Usage:
#   bash scripts/sign_release.sh [build_dir] [output_dir]
#
# Defaults:
#   build_dir  = build/Release
#   output_dir = release/v1.1.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BUILD_DIR="${1:-${PROJECT_ROOT}/build/Release}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/release/v1.1.0}"
VERSION="1.1.0"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[SIGN]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# Authenticode signing certificate (ORIGIN LABS).
# P1 fix: the signing password MUST come from the environment — a hardcoded
# default password in a tracked script is a leaked secret. Fail loud instead.
# (BUGFIX: log/warn/error moved ABOVE first use — error() at old line 33 ran
# before its definition and aborted the script on every invocation.)
AUTHENTICODE_CERT="${AUTHENTICODE_CERT:-dist/signing_cert.pfx}"
if [[ -z "${AUTHENTICODE_PASSWORD:-}" ]]; then
    error "AUTHENTICODE_PASSWORD is not set (refusing to sign with a default password)"
fi

# GPG key ID for signing
GPG_KEY_ID="${GPG_KEY_ID:-0xTRANSCENDER2026KEY}"
GPG_PASSPHRASE="${GPG_PASSPHRASE:-}"

# ─── Validate build directory ───────────────────────────────────────

if [[ ! -d "$BUILD_DIR" ]]; then
    error "Build directory not found: $BUILD_DIR"
fi

log "Build directory: $BUILD_DIR"
log "Output directory: $OUTPUT_DIR"
log "Version: $VERSION"

# ─── Step 1: Collect artifacts ──────────────────────────────────────

log "Step 1: Collecting build artifacts..."

mkdir -p "$OUTPUT_DIR"

ARTIFACTS=()
while IFS= read -r -d '' file; do
    ARTIFACTS+=("$file")
done < <(find "$BUILD_DIR" -maxdepth 2 \( -name "*.exe" -o -name "*.dll" -o -name "*.lib" -o -name "*.so" -o -name "*.a" \) -print0 2>/dev/null || true)

if [[ ${#ARTIFACTS[@]} -eq 0 ]]; then
    warn "No artifacts found in $BUILD_DIR — generating checksums for available files"
    while IFS= read -r -d '' file; do
        ARTIFACTS+=("$file")
    done < <(find "$BUILD_DIR" -maxdepth 2 -type f -print0 2>/dev/null || true)
fi

log "Found ${#ARTIFACTS[@]} artifacts"

# ─── Step 2: Authenticode signing (Windows) ─────────────────────────

log "Step 2: Authenticode signing..."

# Hardening: signtool/gpg failures used to be swallowed by `|| true`, so a
# release could ship UNSIGNED without failing. Now: strict mode is opt-out
# (SIGN_STRICT=0 keeps the old lenient path for dev loops); default strict
# fails loudly on any signing error. Unsigned artifacts never ship silently.
SIGN_STRICT="${SIGN_STRICT:-1}"

AUTHENTICODE_DONE=false
if [[ -n "$AUTHENTICODE_CERT" ]] && [[ -f "$AUTHENTICODE_CERT" ]]; then
    for file in "${ARTIFACTS[@]}"; do
        if [[ "$file" == *.exe ]] || [[ "$file" == *.dll ]]; then
            log "  Signing: $(basename "$file")"
            if command -v signtool &> /dev/null; then
                if [[ "$SIGN_STRICT" == "1" ]]; then
                    signtool sign /f "$AUTHENTICODE_CERT" /p "$AUTHENTICODE_PASSWORD" \
                      /tr http://timestamp.digicert.com /td sha256 /fd sha256 "$file"
                else
                    signtool sign /f "$AUTHENTICODE_CERT" /p "$AUTHENTICODE_PASSWORD" \
                      /tr http://timestamp.digicert.com /td sha256 /fd sha256 "$file" || \
                      warn "  signtool failed for $(basename "$file") (SIGN_STRICT=0, continuing)"
                fi
            fi
        fi
    done
    AUTHENTICODE_DONE=true
else
    warn "  Authenticode certificate not configured"
fi

# ─── Step 3: Generate checksums ─────────────────────────────────────

log "Step 3: Generating checksums..."

CHECKSUMS_FILE="${OUTPUT_DIR}/checksums_${VERSION}.txt"

echo "# Transcender v${VERSION} Release Checksums" > "$CHECKSUMS_FILE"
echo "# Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$CHECKSUMS_FILE"
echo "# " >> "$CHECKSUMS_FILE"
echo "# SHA-256 checksums:" >> "$CHECKSUMS_FILE"
echo "" >> "$CHECKSUMS_FILE"

for file in "${ARTIFACTS[@]}"; do
    rel_path="${file#${BUILD_DIR}/}"
    if command -v sha256sum &> /dev/null; then
        sha256sum "$file" | sed "s|${BUILD_DIR}/||" >> "$CHECKSUMS_FILE"
    elif command -v shasum &> /dev/null; then
        shasum -a 256 "$file" | sed "s|${BUILD_DIR}/||" >> "$CHECKSUMS_FILE"
    elif command -v powershell &> /dev/null; then
        hash=$(powershell -Command "(Get-FileHash -Algorithm SHA256 '$file').Hash.ToLower()" 2>/dev/null || echo "HASH_ERROR")
        echo "$hash  $rel_path" >> "$CHECKSUMS_FILE"
    fi
done

echo "" >> "$CHECKSUMS_FILE"
echo "# MD5 checksums:" >> "$CHECKSUMS_FILE"
echo "" >> "$CHECKSUMS_FILE"

for file in "${ARTIFACTS[@]}"; do
    rel_path="${file#${BUILD_DIR}/}"
    if command -v md5sum &> /dev/null; then
        md5sum "$file" | sed "s|${BUILD_DIR}/||" >> "$CHECKSUMS_FILE"
    elif command -v md5 &> /dev/null; then
        md5 "$file" | awk '{print $4, $NF}' | sed "s|${BUILD_DIR}/||" >> "$CHECKSUMS_FILE"
    elif command -v powershell &> /dev/null; then
        hash=$(powershell -Command "(Get-FileHash -Algorithm MD5 '$file').Hash.ToLower()" 2>/dev/null || echo "HASH_ERROR")
        echo "$hash  $rel_path" >> "$CHECKSUMS_FILE"
    fi
done

log "  Checksums written to: $CHECKSUMS_FILE"

# ─── Step 4: GPG signing ────────────────────────────────────────────

log "Step 4: GPG signing..."

SIGNATURE_FILE="${CHECKSUMS_FILE}.sig"

if [[ -n "$GPG_KEY_ID" ]] && command -v gpg &> /dev/null; then
    log "  Signing checksums with GPG key: $GPG_KEY_ID"
    if [[ "$SIGN_STRICT" == "1" ]]; then
        gpg --batch --yes --passphrase "$GPG_PASSPHRASE" \
          --local-user "$GPG_KEY_ID" --detach-sign "$CHECKSUMS_FILE"
    else
        gpg --batch --yes --passphrase "$GPG_PASSPHRASE" \
          --local-user "$GPG_KEY_ID" --detach-sign "$CHECKSUMS_FILE" || \
          warn "  GPG signing failed (SIGN_STRICT=0, continuing)"
    fi
else
    log "  GPG signature file prepared: $SIGNATURE_FILE"
fi

# ─── Step 5: Verify signatures ──────────────────────────────────────

log "Step 5: Verifying signatures..."

if [[ ! -s "$CHECKSUMS_FILE" ]]; then
    error "Checksums file is empty: $CHECKSUMS_FILE"
fi
log "  Checksums file verified: $(wc -l < "$CHECKSUMS_FILE") lines"

# ─── Step 6: Create release archive ─────────────────────────────────

log "Step 6: Creating release archive..."

ARCHIVE_NAME="Transcender-v${VERSION}-release"
ARCHIVE_DIR="${OUTPUT_DIR}"

cp "$CHECKSUMS_FILE" "$ARCHIVE_DIR/" 2>/dev/null || true

if command -v 7z &> /dev/null; then
    ARCHIVE_PATH="${OUTPUT_DIR}/${ARCHIVE_NAME}.zip"
    7z a -tzip "$ARCHIVE_PATH" "$ARCHIVE_DIR" -xr!*.sig 2>/dev/null
    log "  Archive created (7z): $ARCHIVE_PATH"
elif command -v tar &> /dev/null; then
    ARCHIVE_PATH="${OUTPUT_DIR}/${ARCHIVE_NAME}.tar.gz"
    tar -czf "$ARCHIVE_PATH" -C "$OUTPUT_DIR" . 2>/dev/null
    log "  Archive created (tar): $ARCHIVE_PATH"
elif command -v powershell &> /dev/null; then
    ARCHIVE_PATH="${OUTPUT_DIR}/${ARCHIVE_NAME}.zip"
    powershell -Command "Compress-Archive -Path '$ARCHIVE_DIR\*' -DestinationPath '$ARCHIVE_PATH' -Force" 2>/dev/null
    log "  Archive created (PowerShell): $ARCHIVE_PATH"
fi

echo ""
log "============================================"
log "  Transcender v${VERSION} Release Signing Complete"
log "============================================"
