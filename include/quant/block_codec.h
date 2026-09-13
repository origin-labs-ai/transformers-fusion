#pragma once

#include "quant/types.h"
#include <cstdint>
#include <cstddef>
#include <vector>

namespace quant {

// ============================================================================
// Canonical QUANT block codec — single source of truth for on-disk block payloads.
//
// Every format's payload is self-contained in `indices` (and optionally
// `codebook` for QUANT1 block means) and occupies AT MOST the claimed BPW:
//
//   Format               Payload layout (n = block weights)
//   -----                ------------------------------------
//   QUANT32                n*4 bytes raw FP32                  -> 32.00 BPW
//   QUANT16 / QUANT16_G    n*2 bytes raw FP16                  -> 16.00 BPW
//   QUANT8  / QUANT8_G     8b*(n) + per-16 7b sc + FP16 d      ->  8.00/ 8.50 BPW
//   QUANT4  / QUANT4_G     4b*(n) + per-32 6b sc + per-32 6b min + FP16 d/dm -> 4.00/ 4.50 BPW
//   QUANT2  / QG2     2b*(n) + per-16 4b sc + per-16 4b min + FP16 d/dm -> 2.00/ 2.625 BPW
//   QUANT1                 ceil(n/32)*2 bytes FP16 block means ->  0.50 BPW
//   QG1             16b FP16 scale + 16 slots + 1b*(n-16)-> 1.00 BPW
//   Q1_5             per 32: 16b FP16 scale + 32 signs   ->  1.50 BPW
//   QG_1_5         16b scale + n signs + (n/2-16) ref. ->  1.50 BPW
//   QUANT_Q1         4B FP32 scale + k*3 records         ->  2.00 BPW
//   QUANT_Q1_G     4B header (2xFP16) + k*3 records    ->  2.00 BPW
//   QG_6_K_L        6b*(n) + per-16 int8 scales + FP16 d->  6.5625 BPW
//                    (Q6_K scheme: full 256-block = 210 B exact;
//                     per-16 scales only when they fit the tail budget)
//
// Among the GRP variants, QG2 / QUANT4_G carry REAL per-group AFFINE
// state (scale + min): QG2 stores a 4-bit scale and a 4-bit min per
// 16-weight group plus FP16 d/dm (64 + 8 + 8 + 4 = 84 B per 256-block =
// 2.625 BPW, exactly the Q2_K budget); QUANT4_G stores a 6-bit scale and a
// 6-bit min per 32-weight group plus FP16 d/dm (128 + 6 + 6 + 4 = 144 B per
// 256-block = 4.5 BPW, exactly the Q4_K budget). QUANT8_G keeps per-16-group
// 7-bit scales + FP16 d (14 + 2 = 16 B per 256-block = 0.5 BPW). The overhead
// is included in the honest claimed BPW (QG2=2.625, QUANT4_G=4.5,
// QUANT8_G=8.5; full blocks land exactly at 84/144/272 bytes). This is our
// own native layout with a Lloyd-refined plain least-squares fitter (not
// GGML's weighted heuristic). A tail that cannot afford the group header
// inside ceil(bpw*n/8)+1 degrades deterministically to the plain grouped
// codec (quant_grp16/dequant_grp16), which encoder and decoder compute the
// same way.
// FP16 d. QG1 and QG_1_5 store a single block-level FP16 scale
// (funded by slots / refinement bits), QUANT_Q1_G a 4-byte per-half-block
// FP16 header, and QUANT16_G is plain FP16 with no group state at all.
// QG_6_K_L mirrors the GGML Q6_K scheme: per-16-element int8 scales + one
// FP16 global scale d (16 groups cover a full 256-weight block). Payload size
// obeys the budget contract below (full blocks exact, tail blocks +1 byte of
// container alignment), so a stored block never exceeds format_bpw() except
// for that documented per-tail-block allowance.
//
// Non-GRP lattice formats fund their single FP16 scale with "slots": a few
// weights per block that decode as 0.0 (skipped on encode too). All bit
// streams are LSB-first, matching FormatRegistry's tensor-level layouts so
// claimed BPW equals stored bytes everywhere (block and tensor level).
// ============================================================================

// Quantize one block into its canonical payload. `codebook` is used only by
// QUANT1 (FP16 block means); every other format leaves it empty.
//
// Budget contract (enforced by tests/test_block_codec.cpp):
//   full blocks (n = 256): stored bytes == ceil(claimed_bpw * n / 8) exactly;
//   tail blocks:           stored bytes <= ceil(claimed_bpw * n / 8) + 1
//                          (one byte of container alignment per tail block).
bool quantize_block_all(Format fmt, const float* w, int n,
                        std::vector<uint8_t>& indices, std::vector<uint8_t>& codebook);

// Decode one block payload into `out` (nw floats). Bounds-safe: reads past
// the payload return 0, unhandled formats produce all-zeros.
void dequantize_block_all(Format fmt,
                          const uint8_t* indices, size_t idx_bytes,
                          const uint8_t* codebook, size_t cb_bytes,
                          uint32_t nw, float* out);

// Actual bits-per-weight of a stored payload (0.0 if nw == 0).
float block_actual_bpw(uint32_t nw, size_t idx_bytes, size_t cb_bytes);

// Max payload bytes for the claimed BPW of `fmt` over n weights.
size_t block_claimed_bytes(Format fmt, uint32_t n);

} // namespace quant
