// ============================================================================
// image.h — Image processing types for Transcender
// ============================================================================
#pragma once
#include "quant/tensor.h"
#include <cstdint>
#include <vector>
#include <string>

namespace quant {

// ========================================================================
// Image — simple image container (HWC layout, float32 pixels)
// ========================================================================
class Image {
public:
    Image() = default;
    Image(int width, int height, int channels);
    ~Image() = default;

    static Image load(const std::string& path);
    void save(const std::string& path) const;

    int width() const { return width_; }
    int height() const { return height_; }
    int channels() const { return channels_; }
    int64_t numel() const { return static_cast<int64_t>(width_) * height_ * channels_; }

    float* data() { return pixels_.data(); }
    const float* data() const { return pixels_.data(); }

    Tensor to_tensor() const;

private:
    int width_ = 0;
    int height_ = 0;
    int channels_ = 0;
    std::vector<float> pixels_;
};

} // namespace quant
