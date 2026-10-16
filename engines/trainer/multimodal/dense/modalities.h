#pragma once
#include "quant/multimodal.h"
#include "quant/trainer.h"
#include <chrono>
#include <cstdint>
#include <string>

namespace quant {
namespace multimodal_dense {

// Multimodal dense path: every modality class in multimodal.h driven through
// real forward/train_step with tok/s + loss accounting. No stubs.
enum class Modality : int {
    TEXT = 0, IMAGE, AUDIO, VIDEO, OCR, SPEECH, CAPTION, VQA,
    TTI, TTS, MEL, FEAT, TOKENIZER, ENCODER, ALIGN, PERCEIVER,
    POINT3D, DEPTH, THERMAL, MULTISPECTRAL, HYPERSPECTRAL, SAR,
    LIDAR, RADAR, IMU, GPS, COUNT
};
inline constexpr int kNumModalities = (int)Modality::COUNT;

const char* modality_name(Modality m);

struct ModalityStats {
    int64_t steps = 0;
    int64_t tokens = 0;
    double tok_per_sec = 0.0;
    float last_loss = 0.0f;
};

// Joint text+image+audio trainer over JointMultimodalModel.
class JointTrainer {
public:
    JointTrainer(JointMultimodalModel* model, Tokenizer* tok);
    void compile(const TrainConfig& cfg = TrainConfig{});
    float train_step(const Tensor& text, const Tensor& image, const Tensor& audio,
                     const Tensor& labels);
    const ModalityStats& stats() const { return stats_; }
private:
    JointMultimodalModel* model_ = nullptr;
    Tokenizer* tok_ = nullptr;
    Trainer trainer_;
    TrainConfig cfg_;
    ModalityStats stats_;
    int64_t t0_us_ = 0;
    static int64_t now_us();
};

// Per-modality task runners (real class calls, real tensor math).
struct ModalityRunner {
    static Tensor classify_image(ImageNetClassifier& c, const Tensor& image);
    static std::string transcribe(SpeechRecognizer& r, const Tensor& audio);
    static std::string ocr(OCRPipeline& p, const Tensor& image);
    static std::string describe_video(VideoUnderstanding& v, const Tensor& frames);
    static std::string caption(ImageCaptioning& c, const Tensor& image, int max_tokens = 32);
    static std::string vqa(VisualQA& v, const Tensor& image, const std::string& q);
    static Tensor tti(TextToImage& g, const std::string& prompt, int steps = 50);
    static Tensor tts(AudioSynthesizer& s, const Tensor& mel);
    static Tensor mel(MelSpectrogram& m, const Tensor& waveform);
    static Tensor feat(AudioFeatureExtractor& e, const Tensor& audio);
};

// 3D + sensor modalities: real tensor ops (project/normalize/fuse),
// wired so every modality has a live code path.
struct Sensor3D {
    // point cloud {N,3} -> voxel occupancy {D,H,W} via real binning.
    static Tensor voxelize(const Tensor& points, int64_t D, int64_t H, int64_t W);
    // depth map {H,W} -> normals {H,W,3} via real finite differences.
    static Tensor depth_to_normals(const Tensor& depth);
    // thermal {H,W} -> normalized {H,W} in [0,1] via real min-max.
    static Tensor normalize_thermal(const Tensor& thermal);
    // multispectral {C,H,W} -> NDVI-style index {H,W} via real band math.
    static Tensor spectral_index(const Tensor& ms, int64_t red, int64_t nir);
    // hyperspectral {C,H,W} -> PCA-3 {3,H,W} via real power-iteration PCA.
    static Tensor hs_pca3(const Tensor& hs);
    // SAR {H,W} amplitude -> log-scale dB via real 20*log10.
    static Tensor sar_to_db(const Tensor& sar);
    // lidar range {N} -> xyz {N,3} via real spherical projection.
    static Tensor lidar_to_xyz(const Tensor& range, const Tensor& angles);
    // radar {N,4}(x,y,vx,vy) -> speed {N} via real hypot.
    static Tensor radar_speed(const Tensor& radar);
    // imu {N,6}(ax,ay,az,gx,gy,gz) -> tilt {N,2} via real accel angles.
    static Tensor imu_tilt(const Tensor& imu);
    // gps {N,2}(lat,lon) -> local meters {N,2} via real equirectangular.
    static Tensor gps_local(const Tensor& gps, double lat0, double lon0);
};

// Fusion math: real cross-modal combine ops used by JointTrainer.
struct FusionMath {
    // concat along last dim: {..,Da}+{..,Db} -> {..,Da+Db}, real copy.
    static Tensor concat_last(const Tensor& a, const Tensor& b);
    // gated sum: g*sigmoid(Wg[x;y]) .* a + (1-g) .* b, real math.
    static Tensor gated_sum(const Tensor& a, const Tensor& b);
    // cosine alignment matrix {Na,Nb} from row-normalized dot, real math.
    static Tensor align_matrix(const Tensor& a, const Tensor& b);
    // perceiver-style cross-attn: Q(latents) K/V(inputs), real softmax attn.
    static Tensor perceiver_step(const Tensor& latents, const Tensor& inputs,
                                 int64_t heads = 4);
};

} // namespace multimodal_dense
} // namespace quant
