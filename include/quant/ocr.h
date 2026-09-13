// ============================================================================
// ocr.h — OCR (Optical Character Recognition) types for Transcender
// ============================================================================
#pragma once
#include "quant/tensor.h"
#include "quant/image.h"
#include <string>
#include <vector>

namespace quant {

// ========================================================================
// OCR Result — recognized text region with bounding box
// ========================================================================
struct OCRBox {
    float x0, y0, x1, y1;  // Normalized bounding box coordinates
    std::string text;
    float confidence = 0.0f;
};

struct OCRResult {
    std::vector<OCRBox> boxes;
    std::string full_text;
};

// ========================================================================
// OCREngine — text recognition from images
// ========================================================================
class OCREngine {
public:
    OCREngine() = default;
    ~OCREngine() = default;

    struct Config {
        int64_t hidden_size = 512;
        int64_t num_layers = 6;
        int64_t num_heads = 8;
    };

    void init(const Config& cfg);
    OCRResult recognize(const Image& image);
    OCRResult recognize(const Tensor& image_tensor);

    bool is_initialized() const { return initialized_; }

private:
    Config config_;
    bool initialized_ = false;
};

} // namespace quant
