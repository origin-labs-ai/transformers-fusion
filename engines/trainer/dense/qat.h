#pragma once
#include "quant/ste_quantizer.h"
#include "quant/trainer.h"
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace quant {
namespace dense_engine {

struct QATStats {
    int64_t steps = 0;
    float mean_scale = 0.0f;
    float mean_sqnr_db = 0.0f;
};

class QATEngine {
public:
    QATEngine(int bits = 8, bool symmetric = true, bool use_lsq = false,
              float init_scale = 0.1f);
    void attach(Trainer* trainer);
    Tensor fake_quantize(const Tensor& weight);
    float step_scales(float lr = 1e-4f);
    void calibrate(const std::vector<Tensor*>& params, int64_t batches = 8);
    float sqnr_db(const Tensor& fp32, const Tensor& fq);
    void set_bits(int bits);
    void set_schedule(int start_step, int end_step);
    bool active_at(int step) const;
    const QATStats& stats() const { return stats_; }
    STEQuantizer& inner() { return ste_; }
private:
    STEQuantizer ste_;
    Trainer* trainer_ = nullptr;
    QATStats stats_;
    int start_ = 0;
    int end_ = 1 << 30;
};

struct MPStats {
    float loss_scale = 128.0f;
    int64_t overflows = 0;
    int64_t steps = 0;
};

class MixedPrecision {
public:
    MixedPrecision(float init_scale = 128.0f, int interval = 2000);
    void unscale(std::vector<Tensor*>& params);
    bool check_overflow(const std::vector<Tensor*>& params);
    void step(const std::vector<Tensor*>& params);
    float scale() const { return scale_; }
    const MPStats& stats() const { return stats_; }
private:
    float scale_ = 128.0f;
    int interval_ = 2000;
    int since_grow_ = 0;
    MPStats stats_;
};

} // namespace dense_engine
} // namespace quant
