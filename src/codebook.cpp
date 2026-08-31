#include "quant/codebook.h"
#include "quant/random.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

namespace quant {

// ===========================================================================
// IEEE 754 FP16 <-> FP32 conversion
// ===========================================================================

uint16_t CodebookQUANT4::float_to_half(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint16_t sign = (uint16_t)((u >> 16) & 0x8000);
    int32_t exp = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x007FFFFF;

    if (exp <= 0) {
        // Denormal or zero
        if (exp < -10) return sign;
        mant = (mant | 0x00800000) >> (1 - exp);
        return sign | (uint16_t)(mant >> 13);
    }
    if (exp >= 31) {
        // Infinity or NaN
        return sign | 0x7C00 | (mant != 0 ? 0x0200 : 0);
    }
    return sign | ((uint16_t)exp << 10) | (uint16_t)(mant >> 13);
}

float CodebookQUANT4::half_to_float(uint16_t h) {
    uint16_t sign = h & 0x8000;
    int32_t exp = (h >> 10) & 0x1F;
    uint16_t mant = h & 0x03FF;

    uint32_t f;
    if (exp == 0) {
        // Denormal or zero
        if (mant == 0) {
            f = sign << 16;
        } else {
            // Normalize
            int shift = 10;
            while ((mant & 0x0400) == 0) { mant <<= 1; shift--; }
            exp = 1 - shift + 127 - 15;
            mant = (mant & 0x03FF) << (23 - 10);
            f = ((uint32_t)sign << 16) | ((uint32_t)exp << 23) | mant;
        }
    } else if (exp == 31) {
        // Infinity or NaN
        f = ((uint32_t)sign << 16) | 0x7F800000 | ((uint32_t)mant << 13);
    } else {
        f = ((uint32_t)sign << 16) | (((uint32_t)(exp - 15 + 127)) << 23) | ((uint32_t)mant << 13);
    }

    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

// ===========================================================================
// CodebookQUANT8
// ===========================================================================

void CodebookQUANT8::train(const float* data, size_t count) {
    if (count == 0) return;

    RNG rng;

    // Initialize centroids from random data points
    size_t init_count = std::min((size_t)SIZE, count);
    std::vector<size_t> indices(count);
    for (size_t i = 0; i < count; i++) indices[i] = i;

    for (size_t i = count; i > 1 && init_count < count; i--) {
        size_t j = (size_t)rng.uniform_int(0, (int)i - 1);
        std::swap(indices[i - 1], indices[j]);
    }

    for (int c = 0; c < (int)init_count; c++) centroids[c] = data[indices[c]];
    for (int c = (int)init_count; c < SIZE; c++) centroids[c] = data[0];

    std::vector<uint8_t> assign(count);
    std::vector<double> sums(SIZE);
    std::vector<int> counts(SIZE);

    for (int iter = 0; iter < 20; iter++) {
        // Assignment step
        for (size_t i = 0; i < count; i++) {
            float best_dist = std::numeric_limits<float>::max();
            uint8_t best_c = 0;
            for (int c = 0; c < SIZE; c++) {
                float d = data[i] - centroids[c];
                float dist = d * d;
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = (uint8_t)c;
                }
            }
            assign[i] = best_c;
        }

        // Update step
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t i = 0; i < count; i++) {
            uint8_t c = assign[i];
            sums[c] += data[i];
            counts[c]++;
        }

        bool changed = false;
        for (int c = 0; c < SIZE; c++) {
            if (counts[c] > 0) {
                float new_val = (float)(sums[c] / (double)counts[c]);
                if (std::abs(new_val - centroids[c]) > 1e-7f) changed = true;
                centroids[c] = new_val;
            }
        }

        if (!changed) break;
    }
}

void CodebookQUANT8::ema_update(const float* data, const uint8_t* assign, size_t count, float lr) {
    if (count == 0 || lr <= 0.0f) return;

    std::vector<double> sums(SIZE, 0.0);
    std::vector<int> counts(SIZE, 0);

    for (size_t i = 0; i < count; i++) {
        uint8_t c = assign[i];
        sums[c] += data[i];
        counts[c]++;
    }

    for (int c = 0; c < SIZE; c++) {
        if (counts[c] > 0) {
            float mean = (float)(sums[c] / (double)counts[c]);
            centroids[c] = (1.0f - lr) * centroids[c] + lr * mean;
        }
    }
}

void CodebookQUANT8::ema_update(float decay) {
    QUANT_CHECK(decay > 0.0f && decay < 1.0f, "EMA decay must be in (0,1)");
    for (int c = 0; c < SIZE; c++) {
        double bc = batch_counts_[c];
        double bs = batch_sums_[c];
        if (bc > 0.0) {
            running_counts_[c] = running_counts_[c] * decay + bc * (1.0 - decay);
            running_sums_[c] = running_sums_[c] * decay + bs * (1.0 - decay);
            centroids[c] = (float)(running_sums_[c] / running_counts_[c]);
        }
        batch_counts_[c] = 0.0;
        batch_sums_[c] = 0.0;
    }
}

uint16_t CodebookQUANT8::quantize(float val) const {
    uint16_t best = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < SIZE; i++) {
        float d = val - centroids[i];
        float dist = d * d;
        if (dist < best_dist) {
            best_dist = dist;
            best = (uint16_t)i;
        }
    }
    batch_counts_[best] += 1.0;
    batch_sums_[best] += val;
    return best;
}

float CodebookQUANT8::dequantize(uint16_t idx) const {
    return centroids[idx];
}

size_t CodebookQUANT8::serialized_size() const {
    return sizeof(uint32_t) + SIZE * sizeof(float);
}

size_t CodebookQUANT8::serialize(uint8_t* dst) const {
    uint32_t magic = 0x51554138; // "QUA8"
    std::memcpy(dst, &magic, sizeof(magic));
    std::memcpy(dst + sizeof(magic), centroids, SIZE * sizeof(float));
    return serialized_size();
}

CodebookQUANT8 CodebookQUANT8::deserialize(const uint8_t* src, size_t& offset, size_t size) {
    CodebookQUANT8 cb;
    if (offset + sizeof(uint32_t) + SIZE * sizeof(float) > size)
        throw Error("CodebookQUANT8::deserialize: buffer overflow");
    uint32_t magic;
    std::memcpy(&magic, src + offset, sizeof(magic));
    offset += sizeof(magic);
    if (magic != 0x51554138) throw Error("CodebookQUANT8::deserialize: invalid magic");
    std::memcpy(cb.centroids, src + offset, SIZE * sizeof(float));
    offset += SIZE * sizeof(float);
    return cb;
}

// ===========================================================================
// CodebookQUANT4
// ===========================================================================

void CodebookQUANT4::train(const float* data, size_t count) {
    if (count == 0) return;

    RNG rng;

    size_t init_count = std::min((size_t)SIZE, count);
    std::vector<size_t> indices(count);
    for (size_t i = 0; i < count; i++) indices[i] = i;

    for (size_t i = count; i > 1 && init_count < count; i--) {
        size_t j = (size_t)rng.uniform_int(0, (int)i - 1);
        std::swap(indices[i - 1], indices[j]);
    }

    std::vector<float> tmp_centroids(SIZE);
    for (int c = 0; c < (int)init_count; c++) tmp_centroids[c] = data[indices[c]];
    for (int c = (int)init_count; c < SIZE; c++) tmp_centroids[c] = data[0];

    std::vector<uint8_t> assign(count);
    std::vector<double> sums(SIZE);
    std::vector<int> counts(SIZE);

    for (int iter = 0; iter < 20; iter++) {
        for (size_t i = 0; i < count; i++) {
            float best_dist = std::numeric_limits<float>::max();
            uint8_t best_c = 0;
            for (int c = 0; c < SIZE; c++) {
                float d = data[i] - tmp_centroids[c];
                float dist = d * d;
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = (uint8_t)c;
                }
            }
            assign[i] = best_c;
        }

        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t i = 0; i < count; i++) {
            uint8_t c = assign[i];
            sums[c] += data[i];
            counts[c]++;
        }

        bool changed = false;
        for (int c = 0; c < SIZE; c++) {
            if (counts[c] > 0) {
                float new_val = (float)(sums[c] / (double)counts[c]);
                if (std::abs(new_val - tmp_centroids[c]) > 1e-7f) changed = true;
                tmp_centroids[c] = new_val;
            }
        }

        if (!changed) break;
    }

    // Store as FP16
    for (int c = 0; c < SIZE; c++) {
        centroids[c] = float_to_half(tmp_centroids[c]);
    }
}

void CodebookQUANT4::ema_update(const float* data, const uint8_t* assign, size_t count, float lr) {
    if (count == 0 || lr <= 0.0f) return;

    std::vector<double> sums(SIZE, 0.0);
    std::vector<int> counts(SIZE, 0);

    for (size_t i = 0; i < count; i++) {
        uint8_t c = assign[i];
        sums[c] += data[i];
        counts[c]++;
    }

    for (int c = 0; c < SIZE; c++) {
        if (counts[c] > 0) {
            float mean = (float)(sums[c] / (double)counts[c]);
            float cur = half_to_float(centroids[c]);
            float updated = (1.0f - lr) * cur + lr * mean;
            centroids[c] = float_to_half(updated);
        }
    }
}

void CodebookQUANT4::ema_update(float decay) {
    QUANT_CHECK(decay > 0.0f && decay < 1.0f, "EMA decay must be in (0,1)");
    for (int c = 0; c < SIZE; c++) {
        double bc = batch_counts_[c];
        double bs = batch_sums_[c];
        if (bc > 0.0) {
            running_counts_[c] = running_counts_[c] * decay + bc * (1.0 - decay);
            running_sums_[c] = running_sums_[c] * decay + bs * (1.0 - decay);
            centroids[c] = float_to_half((float)(running_sums_[c] / running_counts_[c]));
        }
        batch_counts_[c] = 0.0;
        batch_sums_[c] = 0.0;
    }
}

uint16_t CodebookQUANT4::quantize(float val) const {
    uint16_t best = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < SIZE; i++) {
        float d = val - half_to_float(centroids[i]);
        float dist = d * d;
        if (dist < best_dist) {
            best_dist = dist;
            best = (uint16_t)i;
        }
    }
    batch_counts_[best] += 1.0;
    batch_sums_[best] += val;
    return best;
}

float CodebookQUANT4::dequantize(uint16_t idx) const {
    return half_to_float(centroids[idx]);
}

size_t CodebookQUANT4::serialized_size() const {
    return sizeof(uint32_t) + SIZE * sizeof(uint16_t);
}

size_t CodebookQUANT4::serialize(uint8_t* dst) const {
    uint32_t magic = 0x51554134; // "QUA4"
    std::memcpy(dst, &magic, sizeof(magic));
    std::memcpy(dst + sizeof(magic), centroids, SIZE * sizeof(uint16_t));
    return serialized_size();
}

CodebookQUANT4 CodebookQUANT4::deserialize(const uint8_t* src, size_t& offset, size_t size) {
    CodebookQUANT4 cb;
    if (offset + sizeof(uint32_t) + SIZE * sizeof(uint16_t) > size)
        throw Error("CodebookQUANT4::deserialize: buffer overflow");
    uint32_t magic;
    std::memcpy(&magic, src + offset, sizeof(magic));
    offset += sizeof(magic);
    if (magic != 0x51554134) throw Error("CodebookQUANT4::deserialize: invalid magic");
    std::memcpy(cb.centroids, src + offset, SIZE * sizeof(uint16_t));
    offset += SIZE * sizeof(uint16_t);
    return cb;
}

// ===========================================================================
// CodebookQ3, CodebookQ6, CodebookQ12
// ===========================================================================

template<typename T>
static void k_means_train(T* centroids, int size, const float* data, size_t count, bool f16) {
    if (count == 0) return;
    RNG rng;
    size_t init_count = std::min((size_t)size, count);
    std::vector<size_t> indices(count);
    for (size_t i = 0; i < count; i++) indices[i] = i;
    for (size_t i = count; i > 1 && init_count < count; i--) {
        size_t j = (size_t)rng.uniform_int(0, (int)i - 1);
        std::swap(indices[i - 1], indices[j]);
    }
    std::vector<float> fp_centroids(size);
    for (int c = 0; c < (int)init_count; c++) fp_centroids[c] = data[indices[c]];
    for (int c = (int)init_count; c < size; c++) fp_centroids[c] = data[0];

    std::vector<int> assign(count);
    std::vector<double> sums(size);
    std::vector<int> counts(size);

    for (int iter = 0; iter < 20; iter++) {
        for (size_t i = 0; i < count; i++) {
            float best_dist = std::numeric_limits<float>::max();
            int best_c = 0;
            for (int c = 0; c < size; c++) {
                float d = data[i] - fp_centroids[c];
                float dist = d * d;
                if (dist < best_dist) { best_dist = dist; best_c = c; }
            }
            assign[i] = best_c;
        }
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = 0; i < count; i++) {
            int c = assign[i];
            sums[c] += data[i];
            counts[c]++;
        }
        bool changed = false;
        for (int c = 0; c < size; c++) {
            if (counts[c] > 0) {
                float new_val = (float)(sums[c] / (double)counts[c]);
                if (std::abs(new_val - fp_centroids[c]) > 1e-7f) changed = true;
                fp_centroids[c] = new_val;
            }
        }
        if (!changed) break;
    }
    for (int c = 0; c < size; c++) {
        if constexpr (std::is_same_v<T, uint16_t>) {
            centroids[c] = CodebookQUANT4::float_to_half(fp_centroids[c]);
        } else {
            centroids[c] = fp_centroids[c];
        }
    }
}

void CodebookQ3::train(const float* data, size_t count) { k_means_train(centroids, SIZE, data, count, false); }
uint16_t CodebookQ3::quantize(float val) const {
    uint16_t best = 0; float best_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < SIZE; i++) {
        float d = val - centroids[i]; float dist = d * d;
        if (dist < best_dist) { best_dist = dist; best = (uint16_t)i; }
    }
    return best;
}
float CodebookQ3::dequantize(uint16_t idx) const { return centroids[idx]; }

void CodebookQ6::train(const float* data, size_t count) { k_means_train(centroids, SIZE, data, count, false); }
uint16_t CodebookQ6::quantize(float val) const {
    uint16_t best = 0; float best_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < SIZE; i++) {
        float d = val - centroids[i]; float dist = d * d;
        if (dist < best_dist) { best_dist = dist; best = (uint16_t)i; }
    }
    return best;
}
float CodebookQ6::dequantize(uint16_t idx) const { return centroids[idx]; }

void CodebookQ12::train(const float* data, size_t count) { k_means_train(centroids, SIZE, data, count, true); }
uint16_t CodebookQ12::quantize(float val) const {
    uint16_t best = 0; float best_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < SIZE; i++) {
        float d = val - CodebookQUANT4::half_to_float(centroids[i]); float dist = d * d;
        if (dist < best_dist) { best_dist = dist; best = (uint16_t)i; }
    }
    return best;
}
float CodebookQ12::dequantize(uint16_t idx) const { return CodebookQUANT4::half_to_float(centroids[idx]); }

} // namespace quant
