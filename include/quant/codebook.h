#pragma once

#include "quant/types.h"

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <limits>

namespace quant {

struct CodebookEntry {
    float val;
    float scale;
};

class Codebook {
public:
    virtual ~Codebook() = default;
    virtual void ema_update(float decay) = 0;
    virtual uint16_t quantize(float val) const = 0;
    virtual float dequantize(uint16_t idx) const = 0;
};

class CodebookQUANT8 : public Codebook {
public:
    static constexpr int SIZE = 256;
    float centroids[SIZE];

    CodebookQUANT8() {
        for (int i = 0; i < SIZE; i++) centroids[i] = 0.0f;
        for (int i = 0; i < SIZE; i++) batch_counts_[i] = 0.0;
        for (int i = 0; i < SIZE; i++) batch_sums_[i] = 0.0;
        for (int i = 0; i < SIZE; i++) running_counts_[i] = 0.0;
        for (int i = 0; i < SIZE; i++) running_sums_[i] = 0.0;
    }

    void train(const float* data, size_t count);
    void ema_update(const float* data, const uint8_t* assign, size_t count, float lr);
    void ema_update(float decay) override;
    uint16_t quantize(float val) const override;
    float dequantize(uint16_t idx) const override;
    size_t serialized_size() const;
    size_t serialize(uint8_t* dst) const;
    static CodebookQUANT8 deserialize(const uint8_t* src, size_t& offset, size_t size);

private:
    mutable double batch_counts_[SIZE];
    mutable double batch_sums_[SIZE];
    double running_counts_[SIZE];
    double running_sums_[SIZE];
};

class CodebookQUANT4 : public Codebook {
public:
    static constexpr int SIZE = 16;
    uint16_t centroids[SIZE];  // F16 storage

    CodebookQUANT4() {
        for (int i = 0; i < SIZE; i++) centroids[i] = 0;
        for (int i = 0; i < SIZE; i++) batch_counts_[i] = 0.0;
        for (int i = 0; i < SIZE; i++) batch_sums_[i] = 0.0;
        for (int i = 0; i < SIZE; i++) running_counts_[i] = 0.0;
        for (int i = 0; i < SIZE; i++) running_sums_[i] = 0.0;
    }

    void train(const float* data, size_t count);
    void ema_update(const float* data, const uint8_t* assign, size_t count, float lr);
    void ema_update(float decay) override;
    uint16_t quantize(float val) const override;
    float dequantize(uint16_t idx) const override;
    size_t serialized_size() const;
    size_t serialize(uint8_t* dst) const;
    static CodebookQUANT4 deserialize(const uint8_t* src, size_t& offset, size_t size);

    static uint16_t float_to_half(float f);
    static float half_to_float(uint16_t h);

private:
    mutable double batch_counts_[SIZE];
    mutable double batch_sums_[SIZE];
    double running_counts_[SIZE];
    double running_sums_[SIZE];
};

struct QuantScale {
    float scale;
};

struct Quant1Scale {
    float scale;
};

class CodebookQ3 : public Codebook {
public:
    static constexpr int SIZE = 8;
    float centroids[SIZE];

    CodebookQ3() {
        for (int i = 0; i < SIZE; i++) centroids[i] = 0.0f;
    }
    void train(const float* data, size_t count);
    // No batch accumulators (unlike QUANT8/QUANT4): k-means retrains whole
    // centroids via train(), so decay-only EMA is intentionally a no-op.
    // Validates decay range so misuse fails loud instead of silently.
    void ema_update(float decay) override {
        if (!(decay > 0.0f && decay < 1.0f))
            throw Error("CodebookQ3::ema_update: decay must be in (0,1)");
    }
    uint16_t quantize(float val) const override;
    float dequantize(uint16_t idx) const override;
};

class CodebookQ6 : public Codebook {
public:
    static constexpr int SIZE = 64;
    float centroids[SIZE];

    CodebookQ6() {
        for (int i = 0; i < SIZE; i++) centroids[i] = 0.0f;
    }
    void train(const float* data, size_t count);
    // Same no-op-by-design contract as CodebookQ3 (k-means via train()).
    void ema_update(float decay) override {
        if (!(decay > 0.0f && decay < 1.0f))
            throw Error("CodebookQ6::ema_update: decay must be in (0,1)");
    }
    uint16_t quantize(float val) const override;
    float dequantize(uint16_t idx) const override;
};

class CodebookQ12 : public Codebook {
public:
    static constexpr int SIZE = 4096;
    uint16_t centroids[SIZE]; // F16

    CodebookQ12() {
        for (int i = 0; i < SIZE; i++) centroids[i] = 0;
    }
    void train(const float* data, size_t count);
    // Same no-op-by-design contract as CodebookQ3 (k-means via train()).
    void ema_update(float decay) override {
        if (!(decay > 0.0f && decay < 1.0f))
            throw Error("CodebookQ12::ema_update: decay must be in (0,1)");
    }
    uint16_t quantize(float val) const override;
    float dequantize(uint16_t idx) const override;
};

} // namespace quant
