// ============================================================================
// vision.h — Vision processing types for Transcender
// ============================================================================
#pragma once
#include "quant/tensor.h"
#include "quant/image.h"
#include <vector>
#include <string>

namespace quant {

// ========================================================================
// VisionEncoder — encode images to latent representations
// ========================================================================
class VisionEncoder {
public:
    VisionEncoder() = default;
    ~VisionEncoder() = default;

    struct Config {
        int64_t img_size = 224;
        int64_t patch_size = 16;
        int64_t hidden_size = 768;
        int64_t num_layers = 12;
        int64_t num_heads = 12;
    };

    void init(const Config& cfg);
    Tensor encode(const Image& image);
    Tensor encode(const Tensor& image_tensor);

    int64_t hidden_size() const { return hidden_size_; }
    bool is_initialized() const { return initialized_; }

private:
    int64_t hidden_size_ = 0;
    bool initialized_ = false;
};

// ========================================================================
// ImageClassifier — classify images into categories
// ========================================================================
struct ClassificationResult {
    int32_t top_class = 0;
    float top_confidence = 0.0f;
    std::vector<float> logits;
};

class ImageClassifier {
public:
    ImageClassifier() = default;
    ~ImageClassifier() = default;

    ClassificationResult classify(const Image& image);
    ClassificationResult classify(const Tensor& image_tensor);

private:
    VisionEncoder encoder_;
    bool initialized_ = false;
};

} // namespace quant
