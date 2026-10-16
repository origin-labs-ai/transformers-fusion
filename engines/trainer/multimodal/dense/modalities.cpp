#include "modalities.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace quant {
namespace multimodal_dense {

const char* modality_name(Modality m) {
    switch (m) {
        case Modality::TEXT: return "text";
        case Modality::IMAGE: return "image";
        case Modality::AUDIO: return "audio";
        case Modality::VIDEO: return "video";
        case Modality::OCR: return "ocr";
        case Modality::SPEECH: return "speech";
        case Modality::CAPTION: return "caption";
        case Modality::VQA: return "vqa";
        case Modality::TTI: return "text-to-image";
        case Modality::TTS: return "text-to-speech";
        case Modality::MEL: return "mel";
        case Modality::FEAT: return "audio-feat";
        case Modality::TOKENIZER: return "multimodal-tokenizer";
        case Modality::ENCODER: return "modality-encoder";
        case Modality::ALIGN: return "cross-align";
        case Modality::PERCEIVER: return "perceiver";
        case Modality::POINT3D: return "point3d";
        case Modality::DEPTH: return "depth";
        case Modality::THERMAL: return "thermal";
        case Modality::MULTISPECTRAL: return "multispectral";
        case Modality::HYPERSPECTRAL: return "hyperspectral";
        case Modality::SAR: return "sar";
        case Modality::LIDAR: return "lidar";
        case Modality::RADAR: return "radar";
        case Modality::IMU: return "imu";
        case Modality::GPS: return "gps";
        default: return "unknown";
    }
}

int64_t JointTrainer::now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

JointTrainer::JointTrainer(JointMultimodalModel* model, Tokenizer* tok)
    : model_(model), tok_(tok), trainer_(nullptr, tok) {
    if (!model_ || !tok_) throw std::runtime_error("[multimodal-dense] needs JointMultimodalModel*+Tokenizer*");
    t0_us_ = now_us();
}

void JointTrainer::compile(const TrainConfig& cfg) { cfg_ = cfg; }

float JointTrainer::train_step(const Tensor& text, const Tensor& image, const Tensor& audio,
                               const Tensor& labels) {
    Tensor fused = model_->forward(text, image, audio);
    const float* f = fused.data<float>();
    const float* l = labels.data<float>();
    int64_t n = std::min(fused.numel(), labels.numel());
    double se = 0.0;
    for (int64_t i = 0; i < n; i++) { double d = (double)f[i] - (double)l[i]; se += d * d; }
    float loss = n > 0 ? (float)(se / (double)n) : 0.0f;
    stats_.steps++;
    stats_.tokens += text.numel();
    stats_.last_loss = loss;
    int64_t dt = now_us() - t0_us_;
    if (dt > 0) stats_.tok_per_sec = (double)stats_.tokens * 1e6 / (double)dt;
    (void)trainer_;
    (void)cfg_;
    return loss;
}

Tensor ModalityRunner::classify_image(ImageNetClassifier& c, const Tensor& image) { return c.classify(image); }
std::string ModalityRunner::transcribe(SpeechRecognizer& r, const Tensor& audio) { return r.transcribe(audio); }
std::string ModalityRunner::ocr(OCRPipeline& p, const Tensor& image) { return p.recognize(image); }
std::string ModalityRunner::describe_video(VideoUnderstanding& v, const Tensor& frames) { return v.describe(frames); }
std::string ModalityRunner::caption(ImageCaptioning& c, const Tensor& image, int max_tokens) { return c.caption(image, max_tokens); }
std::string ModalityRunner::vqa(VisualQA& v, const Tensor& image, const std::string& q) { return v.answer(image, q); }
Tensor ModalityRunner::tti(TextToImage& g, const std::string& prompt, int steps) { return g.generate(prompt, steps); }
Tensor ModalityRunner::tts(AudioSynthesizer& s, const Tensor& mel) { return s.synthesize(mel); }
Tensor ModalityRunner::mel(MelSpectrogram& m, const Tensor& waveform) { return m.compute(waveform); }
Tensor ModalityRunner::feat(AudioFeatureExtractor& e, const Tensor& audio) { return e.extract(audio); }

Tensor Sensor3D::voxelize(const Tensor& points, int64_t D, int64_t H, int64_t W) {
    if (points.rank() != 2 || points.dim(1) != 3) throw std::runtime_error("[sensor3d] voxelize needs {N,3}");
    int64_t N = points.dim(0);
    const float* p = points.data<float>();
    float mnx = p[0], mxx = p[0], mny = p[1], mxx2 = p[1], mnz = p[2], mxz = p[2];
    (void)mxx2;
    for (int64_t i = 0; i < N; i++) {
        mnx = std::min(mnx, p[3 * i]); mxx = std::max(mxx, p[3 * i]);
        mny = std::min(mny, p[3 * i + 1]); mxz = std::max(mxz, p[3 * i + 2]);
        mnz = std::min(mnz, p[3 * i + 2]);
    }
    Tensor vox = Tensor::zeros({D, H, W});
    float* v = vox.data<float>();
    for (int64_t i = 0; i < N; i++) {
        int64_t x = (int64_t)((p[3 * i] - mnx) / (mxx - mnx + 1e-6f) * (W - 1));
        int64_t y = (int64_t)((p[3 * i + 1] - mny) / (mxx - mnx + 1e-6f) * (H - 1));
        int64_t z = (int64_t)((p[3 * i + 2] - mnz) / (mxz - mnz + 1e-6f) * (D - 1));
        x = std::max<int64_t>(0, std::min<int64_t>(W - 1, x));
        y = std::max<int64_t>(0, std::min<int64_t>(H - 1, y));
        z = std::max<int64_t>(0, std::min<int64_t>(D - 1, z));
        v[z * H * W + y * W + x] = 1.0f;
    }
    return vox;
}

Tensor Sensor3D::depth_to_normals(const Tensor& depth) {
    if (depth.rank() != 2) throw std::runtime_error("[sensor3d] depth needs {H,W}");
    int64_t H = depth.dim(0), W = depth.dim(1);
    const float* d = depth.data<float>();
    Tensor nrm = Tensor::zeros({H, W, 3});
    float* o = nrm.data<float>();
    for (int64_t y = 0; y < H; y++) {
        for (int64_t x = 0; x < W; x++) {
            float dx = d[y * W + std::min<int64_t>(W - 1, x + 1)] - d[y * W + std::max<int64_t>(0, x - 1)];
            float dy = d[std::min<int64_t>(H - 1, y + 1) * W + x] - d[std::max<int64_t>(0, y - 1) * W + x];
            float inv = 1.0f / std::sqrt(dx * dx + dy * dy + 1.0f);
            o[(y * W + x) * 3 + 0] = -dx * inv;
            o[(y * W + x) * 3 + 1] = -dy * inv;
            o[(y * W + x) * 3 + 2] = inv;
        }
    }
    return nrm;
}

Tensor Sensor3D::normalize_thermal(const Tensor& thermal) {
    int64_t n = thermal.numel();
    const float* t = thermal.data<float>();
    float mn = t[0], mx = t[0];
    for (int64_t i = 1; i < n; i++) { mn = std::min(mn, t[i]); mx = std::max(mx, t[i]); }
    Tensor o = Tensor::zeros(thermal.shape());
    float* oo = o.data<float>();
    for (int64_t i = 0; i < n; i++) oo[i] = (t[i] - mn) / (mx - mn + 1e-6f);
    return o;
}

Tensor Sensor3D::spectral_index(const Tensor& ms, int64_t red, int64_t nir) {
    if (ms.rank() != 3) throw std::runtime_error("[sensor3d] ms needs {C,H,W}");
    int64_t C = ms.dim(0), H = ms.dim(1), W = ms.dim(2);
    if (red < 0 || red >= C || nir < 0 || nir >= C) throw std::runtime_error("[sensor3d] bad band");
    const float* m = ms.data<float>();
    Tensor o = Tensor::zeros({H, W});
    float* oo = o.data<float>();
    for (int64_t i = 0; i < H * W; i++) {
        float r = m[red * H * W + i], q = m[nir * H * W + i];
        oo[i] = (q - r) / (q + r + 1e-6f);
    }
    return o;
}

Tensor Sensor3D::sar_to_db(const Tensor& sar) {
    int64_t n = sar.numel();
    const float* s = sar.data<float>();
    Tensor o = Tensor::zeros(sar.shape());
    float* oo = o.data<float>();
    for (int64_t i = 0; i < n; i++) oo[i] = 20.0f * std::log10(std::fabs(s[i]) + 1e-6f);
    return o;
}

Tensor Sensor3D::lidar_to_xyz(const Tensor& range, const Tensor& angles) {
    if (range.numel() != angles.numel()) throw std::runtime_error("[sensor3d] range/angles mismatch");
    int64_t N = range.numel();
    const float* r = range.data<float>();
    const float* a = angles.data<float>();
    Tensor o = Tensor::zeros({N, 3});
    float* oo = o.data<float>();
    for (int64_t i = 0; i < N; i++) {
        oo[3 * i] = r[i] * std::cos(a[i]);
        oo[3 * i + 1] = r[i] * std::sin(a[i]);
        oo[3 * i + 2] = 0.0f;
    }
    return o;
}

Tensor Sensor3D::radar_speed(const Tensor& radar) {
    if (radar.rank() != 2 || radar.dim(1) != 4) throw std::runtime_error("[sensor3d] radar needs {N,4}");
    int64_t N = radar.dim(0);
    const float* q = radar.data<float>();
    Tensor o = Tensor::zeros({N});
    float* oo = o.data<float>();
    for (int64_t i = 0; i < N; i++)
        oo[i] = std::sqrt(q[4 * i + 2] * q[4 * i + 2] + q[4 * i + 3] * q[4 * i + 3]);
    return o;
}

Tensor Sensor3D::imu_tilt(const Tensor& imu) {
    if (imu.rank() != 2 || imu.dim(1) != 6) throw std::runtime_error("[sensor3d] imu needs {N,6}");
    int64_t N = imu.dim(0);
    const float* q = imu.data<float>();
    Tensor o = Tensor::zeros({N, 2});
    float* oo = o.data<float>();
    for (int64_t i = 0; i < N; i++) {
        float ax = q[6 * i], ay = q[6 * i + 1], az = q[6 * i + 2];
        oo[2 * i] = std::atan2(ax, std::sqrt(ay * ay + az * az));
        oo[2 * i + 1] = std::atan2(ay, std::sqrt(ax * ax + az * az));
    }
    return o;
}

Tensor Sensor3D::gps_local(const Tensor& gps, double lat0, double lon0) {
    if (gps.rank() != 2 || gps.dim(1) != 2) throw std::runtime_error("[sensor3d] gps needs {N,2}");
    int64_t N = gps.dim(0);
    const float* g = gps.data<float>();
    Tensor o = Tensor::zeros({N, 2});
    float* oo = o.data<float>();
    const double kR = 6371000.0;
    const double kD = 3.141592653589793 / 180.0;
    for (int64_t i = 0; i < N; i++) {
        double la = g[2 * i], lo = g[2 * i + 1];
        oo[2 * i] = (float)((lo - lon0) * kD * kR * std::cos(lat0 * kD));
        oo[2 * i + 1] = (float)((la - lat0) * kD * kR);
    }
    return o;
}

Tensor Sensor3D::hs_pca3(const Tensor& hs) {
    if (hs.rank() != 3) throw std::runtime_error("[sensor3d] hs needs {C,H,W}");
    int64_t C = hs.dim(0), HW = hs.dim(1) * hs.dim(2);
    const float* m = hs.data<float>();
    std::vector<double> mean((size_t)C, 0.0);
    for (int64_t c = 0; c < C; c++) {
        double s = 0.0;
        for (int64_t i = 0; i < HW; i++) s += m[c * HW + i];
        mean[(size_t)c] = s / (double)HW;
    }
    std::vector<std::vector<double>> comps;
    std::vector<double> resid((size_t)C * (size_t)HW);
    for (int64_t c = 0; c < C; c++)
        for (int64_t i = 0; i < HW; i++) resid[(size_t)c * (size_t)HW + i] = m[c * HW + i] - mean[(size_t)c];
    for (int k = 0; k < 3 && k < C; k++) {
        std::vector<double> v((size_t)C, 1.0 / std::sqrt((double)C));
        for (int it = 0; it < 25; it++) {
            std::vector<double> w((size_t)C, 0.0);
            for (int64_t c = 0; c < C; c++) {
                double s = 0.0;
                for (int64_t d = 0; d < C; d++) {
                    double cov = 0.0;
                    for (int64_t i = 0; i < HW; i++)
                        cov += resid[(size_t)c * (size_t)HW + i] * resid[(size_t)d * (size_t)HW + i];
                    s += cov * v[(size_t)d];
                }
                w[(size_t)c] = s;
            }
            double nrm = 0.0;
            for (double x : w) nrm += x * x;
            nrm = std::sqrt(nrm) + 1e-12;
            for (size_t c = 0; c < v.size(); c++) v[c] = w[c] / nrm;
        }
        comps.push_back(v);
        for (int64_t i = 0; i < HW; i++) {
            double proj = 0.0;
            for (int64_t c = 0; c < C; c++) proj += resid[(size_t)c * (size_t)HW + i] * v[(size_t)c];
            for (int64_t c = 0; c < C; c++) resid[(size_t)c * (size_t)HW + i] -= proj * v[(size_t)c];
        }
    }
    int64_t K = (int64_t)comps.size();
    Tensor o = Tensor::zeros({K, hs.dim(1), hs.dim(2)});
    float* oo = o.data<float>();
    for (int64_t k = 0; k < K; k++)
        for (int64_t i = 0; i < HW; i++) {
            double s = mean[0] * 0.0;
            for (int64_t c = 0; c < C; c++) s += (m[c * HW + i]) * comps[(size_t)k][(size_t)c];
            oo[k * HW + i] = (float)s;
        }
    return o;
}

Tensor FusionMath::concat_last(const Tensor& a, const Tensor& b) {
    if (a.rank() < 1 || b.rank() < 1) throw std::runtime_error("[fusion] rank>=1 needed");
    if (a.rank() != b.rank()) throw std::runtime_error("[fusion] rank mismatch");
    for (int r = 0; r < a.rank() - 1; r++)
        if (a.dim(r) != b.dim(r)) throw std::runtime_error("[fusion] outer shape mismatch");
    int64_t Da = a.dim(a.rank() - 1), Db = b.dim(b.rank() - 1);
    int64_t outer = a.numel() / Da;
    Shape sh;
    sh.rank = a.rank();
    for (int r = 0; r < a.rank() - 1; r++) sh.dims[r] = a.dim(r);
    sh.dims[a.rank() - 1] = Da + Db;
    Tensor o = Tensor::zeros(sh);
    const float* x = a.data<float>();
    const float* y = b.data<float>();
    float* oo = o.data<float>();
    for (int64_t i = 0; i < outer; i++) {
        std::memcpy(oo + i * (Da + Db), x + i * Da, (size_t)Da * sizeof(float));
        std::memcpy(oo + i * (Da + Db) + Da, y + i * Db, (size_t)Db * sizeof(float));
    }
    return o;
}

Tensor FusionMath::gated_sum(const Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel()) throw std::runtime_error("[fusion] gated_sum shape mismatch");
    int64_t n = a.numel();
    const float* x = a.data<float>();
    const float* y = b.data<float>();
    Tensor o = Tensor::zeros(a.shape());
    float* oo = o.data<float>();
    for (int64_t i = 0; i < n; i++) {
        float g = 1.0f / (1.0f + std::exp(-(x[i] + y[i])));
        oo[i] = g * x[i] + (1.0f - g) * y[i];
    }
    return o;
}

Tensor FusionMath::align_matrix(const Tensor& a, const Tensor& b) {
    if (a.rank() != 2 || b.rank() != 2) throw std::runtime_error("[fusion] align needs 2-D");
    if (a.dim(1) != b.dim(1)) throw std::runtime_error("[fusion] align dim mismatch");
    int64_t Na = a.dim(0), Nb = b.dim(0), D = a.dim(1);
    const float* x = a.data<float>();
    const float* y = b.data<float>();
    Tensor o = Tensor::zeros({Na, Nb});
    float* oo = o.data<float>();
    std::vector<float> nx((size_t)Na, 1e-6f), ny((size_t)Nb, 1e-6f);
    for (int64_t i = 0; i < Na; i++) {
        double s = 0.0;
        for (int64_t d = 0; d < D; d++) s += (double)x[i * D + d] * x[i * D + d];
        nx[(size_t)i] = (float)std::sqrt(s + 1e-12);
    }
    for (int64_t j = 0; j < Nb; j++) {
        double s = 0.0;
        for (int64_t d = 0; d < D; d++) s += (double)y[j * D + d] * y[j * D + d];
        ny[(size_t)j] = (float)std::sqrt(s + 1e-12);
    }
    for (int64_t i = 0; i < Na; i++)
        for (int64_t j = 0; j < Nb; j++) {
            double s = 0.0;
            for (int64_t d = 0; d < D; d++) s += (double)x[i * D + d] * y[j * D + d];
            oo[i * Nb + j] = (float)(s / nx[(size_t)i] / ny[(size_t)j]);
        }
    return o;
}

Tensor FusionMath::perceiver_step(const Tensor& latents, const Tensor& inputs, int64_t heads) {
    if (latents.rank() != 2 || inputs.rank() != 2) throw std::runtime_error("[fusion] perceiver needs 2-D");
    if (latents.dim(1) != inputs.dim(1)) throw std::runtime_error("[fusion] perceiver dim mismatch");
    if (heads <= 0) throw std::runtime_error("[fusion] bad heads");
    int64_t L = latents.dim(0), N = inputs.dim(0), D = latents.dim(1);
    int64_t Hd = D / heads;
    if (Hd * heads != D) throw std::runtime_error("[fusion] D must split across heads");
    const float* Q = latents.data<float>();
    const float* K = inputs.data<float>();
    Tensor o = Tensor::zeros({L, D});
    float* oo = o.data<float>();
    std::vector<float> scores((size_t)N);
    for (int64_t h = 0; h < heads; h++) {
        for (int64_t l = 0; l < L; l++) {
            float m = -1e30f;
            for (int64_t n = 0; n < N; n++) {
                double s = 0.0;
                for (int64_t d = 0; d < Hd; d++)
                    s += (double)Q[l * D + h * Hd + d] * K[n * D + h * Hd + d];
                s /= std::sqrt((double)Hd);
                scores[(size_t)n] = (float)s;
                m = std::max(m, (float)s);
            }
            double sum = 0.0;
            for (int64_t n = 0; n < N; n++) { scores[(size_t)n] = std::exp(scores[(size_t)n] - m); sum += scores[(size_t)n]; }
            for (int64_t d = 0; d < Hd; d++) {
                double acc = 0.0;
                for (int64_t n = 0; n < N; n++) acc += (scores[(size_t)n] / sum) * K[n * D + h * Hd + d];
                oo[l * D + h * Hd + d] = (float)acc + Q[l * D + h * Hd + d];
            }
        }
    }
    return o;
}

} // namespace multimodal_dense
} // namespace quant
