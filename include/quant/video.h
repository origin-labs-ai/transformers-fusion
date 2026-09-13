// ============================================================================
// video.h — Video processing types for Transcender
// ============================================================================
#pragma once
#include "quant/tensor.h"
#include "quant/image.h"
#include <vector>
#include <string>

namespace quant {

// ========================================================================
// Video — simple video container (sequence of frames)
// ========================================================================
class Video {
public:
    Video() = default;
    ~Video() = default;

    static Video load(const std::string& path);

    int num_frames() const { return static_cast<int>(frames_.size()); }
    int width() const { return width_; }
    int height() const { return height_; }
    float fps() const { return fps_; }

    const Image& frame(int index) const { return frames_[index]; }
    std::vector<Image>& frames() { return frames_; }

private:
    std::vector<Image> frames_;
    int width_ = 0;
    int height_ = 0;
    float fps_ = 30.0f;
};

// ========================================================================
// VideoEncoder — encode video frames to latent representations
// ========================================================================
class VideoEncoder {
public:
    VideoEncoder() = default;
    ~VideoEncoder() = default;

    struct Config {
        int64_t tube_size = 2;
        int64_t img_size = 224;
        int64_t patch_size = 16;
        int64_t hidden_size = 768;
        int64_t num_layers = 12;
        int64_t num_heads = 12;
        int64_t max_frames = 32;
    };

    void init(const Config& cfg);
    Tensor encode(const Video& video);
    Tensor encode(const std::vector<Tensor>& frames);

    int64_t hidden_size() const { return hidden_size_; }
    bool is_initialized() const { return initialized_; }

private:
    int64_t hidden_size_ = 0;
    bool initialized_ = false;
};

} // namespace quant
