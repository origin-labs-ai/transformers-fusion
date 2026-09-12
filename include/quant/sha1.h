#pragma once

#include <cstdint>
#include <cstring>

namespace quant {
namespace sha1 {

struct SHA1_CTX {
    uint32_t state[5];
    uint64_t count;
    uint8_t buffer[64];

    static void init(SHA1_CTX* ctx) {
        ctx->state[0] = 0x67452301;
        ctx->state[1] = 0xEFCDAB89;
        ctx->state[2] = 0x98BADCFE;
        ctx->state[3] = 0x10325476;
        ctx->state[4] = 0xC3D2E1F0;
        ctx->count = 0;
    }

    static uint32_t rotl32(uint32_t x, int n) {
        return (x << n) | (x >> (32 - n));
    }

    static void process_block(SHA1_CTX* ctx, const uint8_t block[64]) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t)block[i*4] << 24) |
                   ((uint32_t)block[i*4+1] << 16) |
                   ((uint32_t)block[i*4+2] << 8) |
                   ((uint32_t)block[i*4+3]);
        }
        for (int i = 16; i < 80; i++) {
            w[i] = rotl32(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
        }

        uint32_t a = ctx->state[0];
        uint32_t b = ctx->state[1];
        uint32_t c = ctx->state[2];
        uint32_t d = ctx->state[3];
        uint32_t e = ctx->state[4];

        auto round = [&](int t) {
            uint32_t f, k;
            if (t < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
            else if (t < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
            else if (t < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else { f = b ^ c ^ d; k = 0xCA62C1D6; }

            uint32_t temp = rotl32(a, 5) + f + e + k + w[t];
            e = d; d = c; c = rotl32(b, 30); b = a; a = temp;
        };

        for (int t = 0; t < 80; t++) round(t);

        ctx->state[0] += a;
        ctx->state[1] += b;
        ctx->state[2] += c;
        ctx->state[3] += d;
        ctx->state[4] += e;
    }

    static void update(SHA1_CTX* ctx, const uint8_t* data, size_t len) {
        // BUGFIX (bug census): short updates (len < 64-idx) fell through to
        // the tail memcpy with a WRONG source offset: `data + (len-(len-idx)%64)`
        // pointed past the input when len < part, causing OOB read + wrong
        // digest. Correct incremental logic: fill the partial block, and only
        // copy the remainder (len % 64) from the tail.
        if (!ctx || (!data && len > 0)) return;
        size_t idx = ctx->count & 63;
        ctx->count += len;
        size_t part = 64 - idx;
        size_t i = 0;
        if (len >= part) {
            memcpy(ctx->buffer + idx, data, part);
            process_block(ctx, ctx->buffer);
            i = part;
            for (; i + 63 < len; i += 64)
                process_block(ctx, data + i);
            idx = 0;
        }
        size_t rem = len - i;
        if (rem > 0)
            memcpy(ctx->buffer + idx, data + i, rem);
    }

    static void final(SHA1_CTX* ctx, uint8_t out[20]) {
        uint64_t bits = ctx->count * 8;
        uint8_t pad = 0x80;
        update(ctx, &pad, 1);
        while ((ctx->count & 63) != 56) {
            uint8_t zero = 0;
            update(ctx, &zero, 1);
        }
        for (int i = 7; i >= 0; i--) {
            uint8_t byte = (uint8_t)(bits >> (i * 8));
            update(ctx, &byte, 1);
        }
        for (int i = 0; i < 5; i++) {
            out[i*4]   = (uint8_t)(ctx->state[i] >> 24);
            out[i*4+1] = (uint8_t)(ctx->state[i] >> 16);
            out[i*4+2] = (uint8_t)(ctx->state[i] >> 8);
            out[i*4+3] = (uint8_t)(ctx->state[i]);
        }
    }
};

} // namespace sha1
} // namespace quant
