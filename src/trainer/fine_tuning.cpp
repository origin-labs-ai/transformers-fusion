// ============================================================================
// Transcender Fine-Tuning Engine — implementation of the three native strategies
// declared in quant/fine_tuning.h. See the header for the design notes.
// ============================================================================

#include "quant/fine_tuning.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <random>
#include <algorithm>
#include <memory>
#include <functional>

namespace quant {

// ============================================================================
// Shared internals
// ============================================================================

namespace {

// Causal attention mask identical to DenseModel::forward.
Tensor make_causal_mask(int64_t B, int64_t S) {
    // BUGFIX (bug census): B was (void)-discarded — mask is {1,1,S,S} (batch
    // broadcast), so B is intentionally unused. Unnamed param states it.
    (void)B; // documented-unused: batch broadcasts from {1,1,S,S}
    Tensor mask(Shape{1, 1, S, S}, DType::F32);
    float* md = mask.data<float>();
    for (int64_t s = 0; s < S; s++)
        for (int64_t t = 0; t < S; t++)
            md[s * S + t] = (t > s) ? -INFINITY : 0.0f;
    return mask;
}

// Slot gating: y = x * tanh(g). The gate is a FIXED architectural scale
// (init 1.0 => tanh ~ 0.76), NOT a trained parameter: it bounds the slot's
// contribution (|y| <= |x|) so the slot can never blow up the logits, while
// the slot factors receive a CONSTANT-scale gradient (dL/dx = yg * tanh(g))
// from the very first step — no dead start, no gate-vs-slot deadlock. The
// guard's "gate open" test reads |tanh(g)|, which is fixed at 0.76, so the
// consult decision is driven purely by the base-confidence gate.
class GatedMulFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        const Tensor& x = inputs[0];
        const float g = inputs[1].data<float>()[0];
        const float tg = std::tanh(g);
        Tensor y(x.shape(), DType::F32);
        const float* xd = x.data<float>();
        float* yd = y.data<float>();
        int64_t n = x.numel();
        for (int64_t i = 0; i < n; ++i) yd[i] = xd[i] * tg;
        return {y};
    }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override {
        const Tensor& yg = grad_output[0];
        const Tensor& x = saved[0];
        const float g = saved[1].data<float>()[0];
        const float tg = std::tanh(g);
        Tensor gx(x.shape(), DType::F32);
        const float* ygd = yg.data<float>();
        float* gxd = gx.data<float>();
        int64_t n = x.numel();
        for (int64_t i = 0; i < n; ++i) gxd[i] = ygd[i] * tg;
        // The gate is a fixed constant: no gradient to it.
        Tensor gg(Shape{1}, DType::F32);
        gg.data<float>()[0] = 0.0f;
        return {gx, gg};
    }
};

Tensor gated_mul_op(const Tensor& x, const Tensor& g) {
    auto fn = std::make_shared<GatedMulFunction>();
    auto outputs = fn->forward({x, g});
    Tensor& y = outputs[0];
    y.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {x, g};
    node->outputs = {y};
    AutogradEngine::instance().register_node(node);
    return y;
}

// Serialization helpers for the quantized adapter store (.nrad).
void write_u32(std::ofstream& f, uint32_t v) { f.write((const char*)&v, sizeof(v)); }
void write_u64(std::ofstream& f, uint64_t v) { f.write((const char*)&v, sizeof(v)); }
void write_f32(std::ofstream& f, float v)    { f.write((const char*)&v, sizeof(v)); }

bool read_u32(std::ifstream& f, uint32_t& v) { return (bool)f.read((char*)&v, sizeof(v)); }
bool read_u64(std::ifstream& f, uint64_t& v) { return (bool)f.read((char*)&v, sizeof(v)); }
bool read_f32(std::ifstream& f, float& v)    { return (bool)f.read((char*)&v, sizeof(v)); }

void write_quant_tensor(std::ofstream& f, const float* data, int64_t n,
                        const FormatDescriptor& desc) {
    QuantResult qr = FormatRegistry::quantize(data, n, desc);
    // Store v2 layout: [ver=2][n][ok][nidx][indices][cb_bytes][codebook]
    // [ncb_fp32][codebook_fp32][ngs][gscales][nzp][gzp][gscale][bs]
    // [nblk][block_idx_bytes...][block_cb_bytes...].
    // The wire payload (indices + codebook + per-block byte layout) is fully
    // self-describing, so dequantize() rebuilds the EXACT stored bytes.
    write_u32(f, 2);
    write_u64(f, (uint64_t)n);
    uint8_t ok = qr.success ? 1 : 0;
    f.write((const char*)&ok, 1);
    write_u32(f, (uint32_t)qr.indices.size());
    if (!qr.indices.empty())
        f.write((const char*)qr.indices.data(), (std::streamsize)qr.indices.size());
    write_u32(f, (uint32_t)qr.codebook.size());
    if (!qr.codebook.empty())
        f.write((const char*)qr.codebook.data(), (std::streamsize)qr.codebook.size());
    write_u32(f, (uint32_t)qr.codebook_fp32.size());
    if (!qr.codebook_fp32.empty())
        f.write((const char*)qr.codebook_fp32.data(),
                (std::streamsize)qr.codebook_fp32.size() * (std::streamsize)sizeof(float));
    write_u32(f, (uint32_t)qr.group_scales.size());
    if (!qr.group_scales.empty())
        f.write((const char*)qr.group_scales.data(),
                (std::streamsize)qr.group_scales.size() * (std::streamsize)sizeof(float));
    write_u32(f, (uint32_t)qr.group_zero_points.size());
    if (!qr.group_zero_points.empty())
        f.write((const char*)qr.group_zero_points.data(),
                (std::streamsize)qr.group_zero_points.size() * (std::streamsize)sizeof(float));
    write_f32(f, qr.global_scale);
    write_u32(f, (uint32_t)qr.block_size);
    write_u32(f, (uint32_t)qr.format.id);  // format identity for decode
    write_u32(f, (uint32_t)qr.block_idx_bytes.size());
    for (uint32_t v : qr.block_idx_bytes) write_u32(f, v);
    for (uint32_t v : qr.block_cb_bytes) write_u32(f, v);
}

// Reads a stored quantized tensor back to fp32. Returns false on EOF/corruption.
bool read_quant_tensor(std::ifstream& f, std::vector<float>& out, int64_t* n_out = nullptr) {
    uint32_t first = 0;
    if (!read_u32(f, first)) return false;
    uint64_t n = 0;
    if (first == 2) {
        if (!read_u64(f, n)) return false;
    } else {
        // v1 store: the first 4 bytes were the low half of the u64 count.
        uint32_t hi = 0;
        if (!read_u32(f, hi)) return false;
        n = ((uint64_t)hi << 32) | first;
    }
    uint8_t ok = 0;
    f.read((char*)&ok, 1);
    uint32_t nidx = 0, ncb = 0, ngs = 0, nzp = 0, bs = 0;
    float gscale = 0.0f;
    if (!read_u32(f, nidx)) return false;
    std::vector<uint8_t> indices(nidx);
    if (nidx) f.read((char*)indices.data(), (std::streamsize)nidx);
    // v2 wire codebook channel (byte count); absent in v1.
    std::vector<uint8_t> wire_codebook;
    if (first == 2) {
        uint32_t ncbb = 0;
        if (!read_u32(f, ncbb)) return false;
        wire_codebook.resize(ncbb);
        if (ncbb) f.read((char*)wire_codebook.data(), (std::streamsize)ncbb);
    }
    if (!read_u32(f, ncb)) return false;
    std::vector<float> codebook(ncb);
    if (ncb) f.read((char*)codebook.data(), (std::streamsize)ncb * (std::streamsize)sizeof(float));
    if (!read_u32(f, ngs)) return false;
    std::vector<float> gscales(ngs);
    if (ngs) f.read((char*)gscales.data(), (std::streamsize)ngs * (std::streamsize)sizeof(float));
    if (!read_u32(f, nzp)) return false;
    std::vector<float> gzp(nzp);
    if (nzp) f.read((char*)gzp.data(), (std::streamsize)nzp * (std::streamsize)sizeof(float));
    if (!read_f32(f, gscale)) return false;
    if (!read_u32(f, bs)) return false;

    uint32_t fmtid = 0;
    if (first == 2) {
        if (!read_u32(f, fmtid)) return false;  // format identity for decode
    }

    // v2 per-block wire layout (enables exact multi-block decode).
    std::vector<uint32_t> block_idx, block_cb;
    if (first == 2) {
        uint32_t nblk = 0;
        if (!read_u32(f, nblk)) return false;
        block_idx.resize(nblk);
        block_cb.resize(nblk);
        for (uint32_t i = 0; i < nblk; i++) if (!read_u32(f, block_idx[i])) return false;
        for (uint32_t i = 0; i < nblk; i++) if (!read_u32(f, block_cb[i])) return false;
    }

    if (!ok || n == 0) {
        out.assign((size_t)n, 0.0f);
        if (n_out) *n_out = (int64_t)n;
        return true;
    }
    QuantResult qr;
    qr.indices = std::move(indices);
    qr.codebook = std::move(wire_codebook);
    qr.block_idx_bytes = std::move(block_idx);
    qr.block_cb_bytes = std::move(block_cb);
    qr.codebook_fp32 = std::move(codebook);
    qr.group_scales = std::move(gscales);
    qr.group_zero_points = std::move(gzp);
    qr.global_scale = gscale;
    qr.block_size = (int)bs;
    qr.num_elements = (int64_t)n;
    qr.success = true;

    // Restore the stored format so dequantize decodes the right codec.
    if (first == 2 && fmtid != 0) {
        const auto& singles = FormatRegistry::get_all_singles();
        for (const auto& s : singles)
            if (s.id == (RegFormat)fmtid) { qr.format = s; break; }
        if (qr.format.name.empty()) return false;
    } else {
        // v1 stores carried no format identity; default to the saved fp32 copy.
        qr.format = FormatRegistry::parse_format_name("Q32");
    }
    out.resize((size_t)n);
    FormatRegistry::dequantize(qr, out.data(), (int64_t)n);
    if (n_out) *n_out = (int64_t)n;
    return true;
}

void write_raw_tensor(std::ofstream& f, const float* data, int64_t n) {
    write_u64(f, (uint64_t)n);
    f.write((const char*)data, (std::streamsize)n * (std::streamsize)sizeof(float));
}

bool read_raw_tensor(std::ifstream& f, std::vector<float>& out) {
    uint64_t n = 0;
    if (!read_u64(f, n)) return false;
    out.resize((size_t)n);
    if (n) f.read((char*)out.data(), (std::streamsize)n * (std::streamsize)sizeof(float));
    return n == 0 || (bool)f;
}

} // namespace

// ============================================================================
// METHOD 1 — SelectiveFineTuner
// ============================================================================

SelectiveFineTuner::SelectiveFineTuner(DenseModel* model) : model_(model) {}

void SelectiveFineTuner::configure(const SelectiveTunerConfig& cfg) {
    cfg_ = cfg;
    if (!model_) return;
    model_->get_parameters(params_);
    fisher_.clear();

    optimizer_ = Adafactor(cfg_.learning_rate, 0.999f, 1e-30f, 0.0f);
    // Register every model weight with Adafactor. Adafactor factorizes a 2-D
    // parameter, so each weight is exposed through a 2-D view aliasing its
    // buffer (a 1-D weight such as an RMSNorm scale is flattened to 1×N).
    opt_views_.clear();
    opt_views_.reserve(params_.size());
    for (auto* p : params_) {
        int64_t n = p->numel();
        fisher_.emplace_back((size_t)n, 0.0f);
        p->requires_grad(true);
        Shape vs = (p->rank() >= 2 && p->dim(0) * p->dim(1) == n)
                       ? Shape{p->dim(0), p->dim(1)}
                       : Shape{1, n};
        opt_views_.emplace_back(p->view(vs));
        opt_views_.back().requires_grad(true);
    }
    for (auto& v : opt_views_)
        if (v.numel() > 0) optimizer_.add_param(&v);
}

void SelectiveFineTuner::freeze_all(bool freeze) {
    for (auto* p : params_) p->requires_grad(!freeze);
}

float SelectiveFineTuner::lr_at(int step) const {
    if (cfg_.warmup_steps > 0 && step <= cfg_.warmup_steps)
        return cfg_.learning_rate * (float)step / (float)cfg_.warmup_steps;
    return cfg_.learning_rate;
}

void SelectiveFineTuner::ste_roundtrip_block(float* data, int64_t n, Format fmt) {
    if (n <= 0 || fmt == Format::Q32) return;
    // Fresh codebook per block: FormatRegistry trains and stores the codec for
    // this exact block, then dequantize returns the straight-through values.
    // (STEQuantizer::forward trains a fresh per-tensor codebook per call —
    // correct per-tensor but coarser than per-block — so it is not used here.)
    FormatDescriptor desc = FormatRegistry::parse_format_name(format_name(fmt));
    if (desc.name.empty()) return;
    QuantResult qr = FormatRegistry::quantize(data, n, desc);
    if (!qr.success) return;
    std::vector<float> rt((size_t)n);
    FormatRegistry::dequantize(qr, rt.data(), n);
    std::memcpy(data, rt.data(), (size_t)n * sizeof(float));
}

std::vector<bool> SelectiveFineTuner::select_blocks(
    const float* grad, const float* fisher, int64_t n,
    int block_size, float sigma, float fisher_bias,
    float min_frac, float* threshold_out) {
    std::vector<bool> mask;
    if (!grad || !fisher || n <= 0 || block_size <= 0) return mask;
    int64_t nb = (n + block_size - 1) / block_size;
    mask.assign((size_t)nb, false);
    if (nb == 0) return mask;

    std::vector<float> sal((size_t)nb, 0.0f);
    double sum = 0.0, sum2 = 0.0;
    for (int64_t b = 0; b < nb; ++b) {
        int64_t start = b * block_size;
        int64_t end = std::min(n, start + block_size);
        double g2 = 0.0, f2 = 0.0;
        int64_t cnt = end - start;
        for (int64_t i = start; i < end; ++i) {
            g2 += (double)grad[i] * (double)grad[i];
            f2 += (double)std::max(0.0f, fisher[i]);
        }
        float gnorm = (cnt > 0) ? (float)std::sqrt(g2 / (double)cnt) : 0.0f;
        float fmean = (cnt > 0) ? (float)(f2 / (double)cnt) : 0.0f;
        sal[(size_t)b] = gnorm * (1.0f + fisher_bias * std::log1p(std::max(0.0f, fmean)));
        sum += (double)sal[(size_t)b];
        sum2 += (double)sal[(size_t)b] * (double)sal[(size_t)b];
    }

    double mean = sum / (double)nb;
    double var = std::max(0.0, sum2 / (double)nb - mean * mean);
    float thr = (float)(mean + (double)sigma * std::sqrt(var));
    int64_t sel = 0;
    for (int64_t b = 0; b < nb; ++b)
        if (sal[(size_t)b] > thr) { mask[(size_t)b] = true; ++sel; }

    int64_t floor_sel = (int64_t)(min_frac * (double)nb);
    floor_sel = std::max<int64_t>(0, std::min(floor_sel, nb));
    if (sel < floor_sel && floor_sel > 0) {
        std::vector<int64_t> order((size_t)nb);
        for (int64_t b = 0; b < nb; ++b) order[(size_t)b] = b;
        std::partial_sort(order.begin(), order.begin() + (size_t)floor_sel, order.end(),
                          [&](int64_t a, int64_t b) { return sal[(size_t)a] > sal[(size_t)b]; });
        std::vector<bool> nmask((size_t)nb, false);
        for (int64_t k = 0; k < floor_sel; ++k) nmask[(size_t)order[(size_t)k]] = true;
        mask = std::move(nmask);
        sel = floor_sel;
    }

    if (threshold_out) *threshold_out = thr;
    return mask;
}

void SelectiveFineTuner::accumulate_fisher(const Tensor& input_ids,
                                           const Tensor& positions,
                                           const Tensor& target_ids) {
    if (!model_ || input_ids.numel() == 0) return;
    auto& engine = AutogradEngine::instance();
    for (auto* p : params_)
        if (p->has_grad()) p->zero_grad();

    AutogradEngine::set_enabled(true);
    for (auto* p : params_) engine.register_parameter(p);
    Tensor logits = model_->forward(input_ids, positions);
    Tensor loss_t = AutogradEngine::cross_entropy_op(logits, target_ids);
    engine.backward(loss_t);
    engine.clear();
    AutogradEngine::set_enabled(false);

    const float decay = cfg_.fisher_decay;
    for (size_t pi = 0; pi < params_.size(); ++pi) {
        Tensor* p = params_[pi];
        if (!p->has_grad()) continue;
        const float* g = p->grad().data<float>();
        auto& f = fisher_[pi];
        int64_t n = p->numel();
        for (int64_t i = 0; i < n; ++i)
            f[(size_t)i] = decay * f[(size_t)i] + (1.0f - decay) * g[i] * g[i];
    }
}

void SelectiveFineTuner::accumulate_fisher(DataLoader& dl, int max_batches) {
    Tensor in(Shape{dl.batch_size(), dl.seq_length()}, DType::F32);
    Tensor tgt(Shape{dl.batch_size(), dl.seq_length()}, DType::F32);
    int done = 0;
    while (dl.next_batch(in, tgt) && (max_batches <= 0 || done < max_batches)) {
        Tensor positions(in.shape(), DType::F32);
        float* pd = positions.data<float>();
        for (int64_t i = 0; i < in.numel(); ++i) pd[i] = (float)(i % dl.seq_length());
        accumulate_fisher(in, positions, tgt);
        ++done;
    }
}

void SelectiveFineTuner::fine_tune(const Tensor& input_ids, const Tensor& positions,
                                   const Tensor& target_ids, int steps) {
    if (!model_ || steps <= 0) return;
    auto& engine = AutogradEngine::instance();

    for (int s = 0; s < steps; ++s) {
        ++step_;
        float lr = lr_at(step_);
        optimizer_.set_lr(lr);

        for (auto* p : params_)
            if (p->has_grad()) p->zero_grad();
        AutogradEngine::set_enabled(true);
        for (auto* p : params_) engine.register_parameter(p);

        Tensor logits = model_->forward(input_ids, positions);
        Tensor loss_t = AutogradEngine::cross_entropy_op(logits, target_ids);
        float loss_val = *(const float*)loss_t.data();
        engine.backward(loss_t);
        engine.clear();
        AutogradEngine::set_enabled(false);

        // Global gradient norm + clip.
        double gnorm2 = 0.0;
        for (auto* p : params_)
            if (p->has_grad()) {
                const float* g = p->grad().data<float>();
                int64_t n = p->numel();
                for (int64_t i = 0; i < n; ++i) gnorm2 += (double)g[i] * (double)g[i];
            }
        float gnorm = (float)std::sqrt(gnorm2);
        float clip = (cfg_.max_grad_norm > 0.0f && gnorm > cfg_.max_grad_norm)
                         ? (cfg_.max_grad_norm / (gnorm + 1e-8f))
                         : 1.0f;

        int total_blocks = 0, selected_blocks = 0;
        // Per-param block masks, retained for the post-step STE roundtrip.
        std::vector<std::vector<bool>> block_masks(params_.size());

        for (size_t pi = 0; pi < params_.size(); ++pi) {
            Tensor* p = params_[pi];
            if (!p->has_grad()) {
                // No fresh gradient this step: hand Adafactor a zeroed view
                // gradient so it never applies a stale update to this weight.
                Tensor zg(Shape{p->numel()});
                zg.zero_();
                opt_views_[pi].set_grad(zg);
                continue;
            }
            int64_t n = p->numel();
            const float* g = p->grad().data<float>();
            const float* f = fisher_[pi].data();

            std::vector<float> gv((size_t)n);
            for (int64_t i = 0; i < n; ++i) gv[(size_t)i] = g[i] * clip;

            float thr = 0.0f;
            std::vector<bool> mask = select_blocks(
                gv.data(), f, n, cfg_.block_size, cfg_.saliency_sigma,
                cfg_.fisher_bias, cfg_.min_select_fraction, &thr);
            total_blocks += (int)mask.size();
            for (bool b : mask)
                if (b) ++selected_blocks;
            block_masks[pi] = mask;

            // select_blocks returns one flag per BLOCK; the Adafactor update
            // is per-weight, so expand the block mask to a per-weight mask (a
            // weight is updated iff its block is selected) and zero the
            // gradients of every non-selected block: Adafactor then moves
            // ONLY the selected blocks' weights.
            std::vector<bool> wmask((size_t)n, false);
            for (size_t b = 0; b < mask.size(); ++b) {
                if (!mask[b]) continue;
                const int64_t start = (int64_t)b * cfg_.block_size;
                const int64_t end = std::min(n, start + cfg_.block_size);
                for (int64_t i = start; i < end; ++i) wmask[(size_t)i] = true;
            }
            Tensor grad_t(Shape{n});
            float* gd = grad_t.data<float>();
            for (int64_t i = 0; i < n; ++i)
                gd[i] = wmask[(size_t)i] ? gv[(size_t)i] : 0.0f;
            opt_views_[pi].set_grad(grad_t);
        }

        // Adafactor steps every registered weight view using the masked
        // gradients (zeroed outside the selected blocks).
        optimizer_.step();

        // STE quant roundtrip: requantize the just-updated selected blocks.
        if (cfg_.ste_roundtrip) {
            for (size_t pi = 0; pi < params_.size(); ++pi) {
                Tensor* p = params_[pi];
                if (!p->has_grad() || block_masks[pi].empty()) continue;
                const std::vector<bool>& mask = block_masks[pi];
                int64_t n = p->numel();
                int64_t nb = (n + cfg_.block_size - 1) / cfg_.block_size;
                for (int64_t b = 0; b < nb; ++b) {
                    if (!mask[(size_t)b]) continue;
                    int64_t start = b * cfg_.block_size;
                    int64_t end = std::min(n, start + cfg_.block_size);
                    ste_roundtrip_block(p->data<float>() + start, end - start, cfg_.target_format);
                }
            }
        }

        stats_.selected_blocks = selected_blocks;
        stats_.total_blocks = total_blocks;
        stats_.selected_fraction = total_blocks > 0
            ? (float)selected_blocks / (float)total_blocks : 0.0f;
        stats_.last_loss = loss_val;
        stats_.steps = step_;

        if (cfg_.save_interval > 0 && (step_ % cfg_.save_interval == 0))
            save(cfg_.output_path);
    }
}

void SelectiveFineTuner::fine_tune(DataLoader& dl, int max_batches) {
    Tensor in(Shape{dl.batch_size(), dl.seq_length()}, DType::F32);
    Tensor tgt(Shape{dl.batch_size(), dl.seq_length()}, DType::F32);
    int done = 0;
    while (dl.next_batch(in, tgt) && (max_batches <= 0 || done < max_batches)) {
        Tensor positions(in.shape(), DType::F32);
        float* pd = positions.data<float>();
        for (int64_t i = 0; i < in.numel(); ++i) pd[i] = (float)(i % dl.seq_length());
        fine_tune(in, positions, tgt, 1);
        ++done;
    }
}

void SelectiveFineTuner::save(const std::string& path) const {
    if (model_) model_->save_quantized(path, cfg_.target_format);
}

// ============================================================================
// METHOD 2 — RankAdapterEngine
// ============================================================================

RankAdapterEngine::RankAdapterEngine(DenseModel* model)
    : model_(model), optimizer_(3e-4f, 0.999f, 1e-30f, 0.0f) {}

void RankAdapterEngine::configure(const RankAdapterConfig& cfg) {
    cfg_ = cfg;
    optimizer_ = Adafactor(cfg_.learning_rate, 0.999f, 1e-30f, cfg_.weight_decay);
}

void RankAdapterEngine::freeze_base(bool freeze) {
    std::vector<Tensor*> bp;
    if (model_) model_->get_parameters(bp);
    for (auto* p : bp) p->requires_grad(!freeze);
}

void RankAdapterEngine::init_adapters() {
    if (!model_) return;
    const TransformerConfig& cfg = model_->config;
    adapters_.clear();
    adapters_.reserve((size_t)cfg.num_layers);
    optimizer_ = Adafactor(cfg_.learning_rate, 0.999f, 1e-30f, cfg_.weight_decay);

    std::mt19937 rng(1337);
    float scale = (float)cfg_.alpha / (float)std::max(1, cfg_.rank);
    float init_std = std::sqrt(1.0f / (float)std::max<int64_t>(1, cfg.ffn_hidden_size));
    std::normal_distribution<float> nd(0.0f, init_std);

    for (int64_t i = 0; i < cfg.num_layers; ++i) {
        LayerAdapter ad;
        ad.layer_index = i;
        ad.rank = cfg_.rank;
        ad.in_dim = cfg.ffn_hidden_size;
        ad.out_dim = cfg.hidden_size;
        ad.name = "layers." + std::to_string(i) + ".ffn.down_proj";
        ad.A = Tensor(Shape{ad.rank, ad.in_dim}, DType::F32);
        ad.B = Tensor(Shape{ad.out_dim, ad.rank}, DType::F32);
        // A ~ N(0, 1/sqrt(in)) pre-scaled by alpha/rank; B = 0 => zero delta.
        float* ad_ = ad.A.data<float>();
        for (int64_t k = 0; k < ad.A.numel(); ++k) ad_[k] = (float)nd(rng) * scale;
        ad.B.fill(0.0f);
        if (cfg_.use_dora) {
            // m init = ||W0[:, o]||_c, the column norms of the frozen base in the
            // engine's {in, out} memory layout (position k*out + o). At this
            // point the delta is exactly zero, so this also makes W' == W0.
            ad.magnitude = Tensor(Shape{ad.out_dim, 1}, DType::F32);
            const Linear& base = model_->layers[(size_t)i]->ffn.down_proj;
            const float* w0 = base.weight.data<float>();
            float* mag = ad.magnitude.data<float>();
            for (int64_t o = 0; o < ad.out_dim; ++o) {
                double s = 0.0;
                for (int64_t k = 0; k < ad.in_dim; ++k) {
                    double d = (double)w0[k * ad.out_dim + o];
                    s += d * d;
                }
                mag[o] = (float)std::sqrt(s);
            }
            ad.magnitude.requires_grad(true);
        }
        ad.A.requires_grad(true);
        ad.B.requires_grad(true);
        adapters_.push_back(std::move(ad));
    }
    for (auto& ad : adapters_) {
        optimizer_.add_param(&ad.A);
        optimizer_.add_param(&ad.B);
    }
    freeze_base(true);
}

Tensor RankAdapterEngine::ffn_forward(int64_t layer_idx, const Tensor& x) const {
    const FFN& f = model_->layers[(size_t)layer_idx]->ffn;
    Tensor g = f.gate_proj.forward(x);
    Tensor u = f.up_proj.forward(x);
    Tensor a = AutogradEngine::silu_op(g);
    Tensor inter = AutogradEngine::mul_op(a, u);   // {..., ffn_hidden}
    Tensor out = f.down_proj.forward(inter);       // {..., hidden}

    const LayerAdapter* ad = nullptr;
    for (const auto& cand : adapters_)
        if (cand.layer_index == layer_idx) { ad = &cand; break; }

    if (ad && ad->A.numel() > 0) {
        // Linear::forward restores leading dims for rank>2 inputs, so flatten
        // the intermediate to 2-D for the low-rank delta path (gradients stay
        // correct: reshape is a view, and the autograd graph is elementwise
        // between the relevant nodes).
        int64_t M = inter.numel() / ad->in_dim;
        Tensor inter2 = (inter.rank() > 2) ? inter.reshape(Shape{M, ad->in_dim}) : inter;
        Tensor mid = AutogradEngine::matmul_op(inter2, ad->A, M, ad->rank, ad->in_dim);
        Tensor delta = AutogradEngine::matmul_op(mid, ad->B, M, ad->out_dim, ad->rank);
        if (delta.rank() < out.rank()) delta = delta.reshape(out.shape());
        out = AutogradEngine::add_op(out, delta);
    }
    return out;
}

Tensor RankAdapterEngine::block_forward(int64_t layer_idx, const Tensor& x,
                                        const Tensor& positions, const Tensor& mask,
                                        KVCache& cache) const {
    TransformerBlock& l = *model_->layers[(size_t)layer_idx];
    if (l.use_parallel_residual) {
        Tensor normed_attn = l.attention_norm.forward(x);
        Tensor normed_ffn = l.ffn_norm.forward(x);
        Tensor attn_out = l.attention.forward(normed_attn, positions, mask, cache, (int)layer_idx);
        Tensor ffn_out = ffn_forward(layer_idx, normed_ffn);
        Tensor combined = AutogradEngine::add_op(attn_out, ffn_out);
        return AutogradEngine::add_op(combined, x);
    }
    Tensor attn_input = l.attention_norm.forward(x);
    Tensor attn_out = l.attention.forward(attn_input, positions, mask, cache, (int)layer_idx);
    attn_out = AutogradEngine::add_op(attn_out, x);
    Tensor ffn_input = l.ffn_norm.forward(attn_out);
    Tensor ffn_out = ffn_forward(layer_idx, ffn_input);
    return AutogradEngine::add_op(ffn_out, attn_out);
}

Tensor RankAdapterEngine::forward_with_adapters(const Tensor& input_ids,
                                                const Tensor& positions,
                                                KVCache* cache) {
    if (!model_) return Tensor();
    const TransformerConfig& cfg = model_->config;
    int64_t B = input_ids.dim(0), S = input_ids.dim(1);

    // Save/restore the engine's enabled state so a bare forward (e.g. an
    // eval call) never leaks graph building into later inference passes.
    const bool prev_enabled = AutogradEngine::enabled();
    AutogradEngine::set_enabled(true);
    Tensor h = model_->tok_embeddings->forward(input_ids.reshape(Shape{B * S}));
    h = h.reshape(Shape{B, S, cfg.hidden_size});

    KVCache local_cache;
    KVCache* active = cache;
    if (!active) {
        local_cache.init((int)cfg.num_layers, cfg.max_seq_len, cfg.num_heads, cfg.head_dim);
        active = &local_cache;
    }
    Tensor mask = make_causal_mask(B, S);
    for (int64_t i = 0; i < cfg.num_layers; ++i)
        h = block_forward(i, h, positions, mask, *active);

    h = model_->norm->forward(h);
    AutogradEngine::set_enabled(prev_enabled);
    return model_->lm_head->forward(h);
}

void RankAdapterEngine::clear_grads() {
    for (auto& ad : adapters_) {
        if (ad.A.has_grad()) ad.A.zero_grad();
        if (ad.B.has_grad()) ad.B.zero_grad();
    }
}

void RankAdapterEngine::train_step(const Tensor& input_ids, const Tensor& positions,
                                   const Tensor& target_ids) {
    if (adapters_.empty()) init_adapters();
    ++step_;
    auto& engine = AutogradEngine::instance();
    clear_grads();
    AutogradEngine::set_enabled(true);
    for (auto& ad : adapters_) {
        engine.register_parameter(&ad.A);
        engine.register_parameter(&ad.B);
    }
    Tensor logits = forward_with_adapters(input_ids, positions);
    Tensor loss_t = AutogradEngine::cross_entropy_op(logits, target_ids);
    last_loss_ = *(const float*)loss_t.data();
    engine.backward(loss_t);
    engine.clear();
    AutogradEngine::set_enabled(false);

    float lr = cfg_.learning_rate;
    if (cfg_.warmup_steps > 0 && step_ <= cfg_.warmup_steps)
        lr = cfg_.learning_rate * (float)step_ / (float)cfg_.warmup_steps;
    optimizer_.set_lr(lr);
    optimizer_.set_grad_clip_norm(1.0f);
    optimizer_.step();
}

void RankAdapterEngine::merge_into_base() {
    if (!model_) return;
    for (const auto& ad : adapters_) {
        Linear& proj = model_->layers[(size_t)ad.layer_index]->ffn.down_proj;
        float* w = proj.weight.data<float>();
        const float* bb = ad.B.data<float>();  // {out, rank}
        const float* aa = ad.A.data<float>();  // {rank, in}
        int64_t out = ad.out_dim, r = ad.rank, in = ad.in_dim;
        // The engine's GEMM reads the down-proj weight at buffer position
        // k*out+o (weights stored {in, out} in memory), and the adapter's
        // forward delta is  delta[m][o] = sum_k inter[m][k] * (sum_r A[k*rank+r]
        // * B[r*out+o]). So the merged ΔW lives at position k*out+o — writing
        // it at o*in+i (the naive textbook layout) would corrupt the weights.
        if (!cfg_.use_dora) {
            for (int64_t k = 0; k < in; ++k)
                for (int64_t o = 0; o < out; ++o) {
                    double acc = 0.0;
                    for (int64_t t = 0; t < r; ++t)
                        acc += (double)aa[k * r + t] * (double)bb[t * out + o];
                    w[k * out + o] += (float)acc;
                }
            continue;
        }

        // DoRA: W' = m ⊙ V / ||V||_c with V = W0 + B·A. Two passes are required —
        // the column norm needs the whole merged column before any of it can be
        // written back. A zero-norm column (or a missing magnitude vector) is
        // left as the plain merged value rather than dividing by zero.
        const bool have_mag = (ad.magnitude.numel() == out);
        const float* mag = have_mag ? ad.magnitude.data<float>() : nullptr;
        std::vector<float> v((size_t)in * (size_t)out);
        std::vector<double> nrm((size_t)out, 0.0);
        for (int64_t k = 0; k < in; ++k)
            for (int64_t o = 0; o < out; ++o) {
                double acc = 0.0;
                for (int64_t t = 0; t < r; ++t)
                    acc += (double)aa[k * r + t] * (double)bb[t * out + o];
                float merged = w[k * out + o] + (float)acc;
                v[(size_t)k * out + o] = merged;
                nrm[(size_t)o] += (double)merged * (double)merged;
            }
        for (int64_t o = 0; o < out; ++o) nrm[(size_t)o] = std::sqrt(nrm[(size_t)o]);
        for (int64_t k = 0; k < in; ++k)
            for (int64_t o = 0; o < out; ++o) {
                const double n = nrm[(size_t)o];
                if (!have_mag || !(n > 0.0)) {
                    w[k * out + o] = v[(size_t)k * out + o];
                    continue;
                }
                w[k * out + o] = (float)((double)mag[o] * (double)v[(size_t)k * out + o] / n);
            }
    }
}

int64_t RankAdapterEngine::dora_param_count() const {
    if (!cfg_.use_dora) return 0;
    int64_t n = 0;
    for (const auto& ad : adapters_) n += ad.magnitude.numel();
    return n;
}

void RankAdapterEngine::magnitude_step(const std::vector<Tensor>& dL_dW, float lr) {
    if (!cfg_.use_dora || !model_) return;
    for (size_t li = 0; li < adapters_.size(); ++li) {
        LayerAdapter& ad = adapters_[li];
        if (li >= dL_dW.size()) break;
        const Tensor& dw = dL_dW[li];
        const int64_t out = ad.out_dim, r = ad.rank, in = ad.in_dim;
        if (dw.numel() != in * out || ad.magnitude.numel() != out) continue;

        const Linear& base = model_->layers[(size_t)ad.layer_index]->ffn.down_proj;
        const float* w0 = base.weight.data<float>();
        const float* aa = ad.A.data<float>();
        const float* bb = ad.B.data<float>();
        const float* g = dw.data<float>();
        float* mag = ad.magnitude.data<float>();

        // Rebuild V and its column norms at the current factors. This is the same
        // quantity merge_into_base() renormalises to, so the gradient is taken at
        // the point the forward actually used.
        std::vector<double> v((size_t)in * (size_t)out), nrm((size_t)out, 0.0);
        for (int64_t k = 0; k < in; ++k)
            for (int64_t o = 0; o < out; ++o) {
                double acc = 0.0;
                for (int64_t t = 0; t < r; ++t)
                    acc += (double)aa[k * r + t] * (double)bb[t * out + o];
                double merged = (double)w0[k * out + o] + acc;
                v[(size_t)k * out + o] = merged;
                nrm[(size_t)o] += merged * merged;
            }
        for (int64_t o = 0; o < out; ++o) nrm[(size_t)o] = std::sqrt(nrm[(size_t)o]);

        for (int64_t o = 0; o < out; ++o) {
            const double n = nrm[(size_t)o];
            if (!(n > 0.0)) continue;   // no direction to scale — leave m alone
            double gm = 0.0;
            for (int64_t k = 0; k < in; ++k)
                gm += (double)g[k * out + o] * v[(size_t)k * out + o] / n;
            mag[o] -= lr * (float)gm;
        }
    }
}

int64_t RankAdapterEngine::adapter_param_count() const {
    int64_t n = 0;
    for (const auto& ad : adapters_) n += ad.A.numel() + ad.B.numel();
    return n;
}

void RankAdapterEngine::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return;
    f.write("NRAD", 4);
    // v1 = LoRA only (A,B). v2 adds the DoRA magnitude vector per layer. v1 files
    // stay loadable; a DoRA engine writes v2.
    write_u32(f, cfg_.use_dora ? 2u : 1u);
    write_u32(f, (uint32_t)adapters_.size());
    FormatDescriptor desc = FormatRegistry::parse_format_name(format_name(cfg_.factor_format));
    if (desc.name.empty()) desc = FormatRegistry::get_single_format(8.0f);
    for (const auto& ad : adapters_) {
        int64_t li = ad.layer_index;
        f.write((const char*)&li, sizeof(li));
        write_u64(f, (uint64_t)ad.rank);
        write_u64(f, (uint64_t)ad.out_dim);
        write_u64(f, (uint64_t)ad.in_dim);
        write_quant_tensor(f, ad.A.data<float>(), ad.A.numel(), desc);
        write_quant_tensor(f, ad.B.data<float>(), ad.B.numel(), desc);
        if (cfg_.use_dora) {
            // Magnitude is a norm-like quantity and must survive exactly, so it
            // is stored as raw fp32 rather than through the factor quantizer.
            if (ad.magnitude.numel() > 0)
                f.write((const char*)ad.magnitude.data<float>(),
                        (std::streamsize)(ad.magnitude.numel() * sizeof(float)));
        }
    }
}

bool RankAdapterEngine::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4] = {0, 0, 0, 0};
    f.read(magic, 4);
    if (std::memcmp(magic, "NRAD", 4) != 0) return false;
    uint32_t version = 0, count = 0;
    if (!read_u32(f, version) || (version != 1 && version != 2)) return false;
    if (!read_u32(f, count) || count == 0) return false;
    const bool has_dora = (version >= 2);

    adapters_.clear();
    adapters_.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        LayerAdapter ad;
        int64_t li = 0;
        uint64_t rk = 0, od = 0, id = 0;
        if (!f.read((char*)&li, sizeof(li))) return false;
        if (!read_u64(f, rk) || !read_u64(f, od) || !read_u64(f, id)) return false;
        ad.layer_index = li;
        ad.rank = (int64_t)rk;
        ad.out_dim = (int64_t)od;
        ad.in_dim = (int64_t)id;
        ad.name = "layers." + std::to_string(li) + ".ffn.down_proj";
        ad.A = Tensor(Shape{ad.rank, ad.in_dim}, DType::F32);
        ad.B = Tensor(Shape{ad.out_dim, ad.rank}, DType::F32);
        std::vector<float> av, bv;
        if (!read_quant_tensor(f, av) || !read_quant_tensor(f, bv)) return false;
        if ((int64_t)av.size() != ad.A.numel() || (int64_t)bv.size() != ad.B.numel())
            return false;
        std::memcpy(ad.A.data<float>(), av.data(), (size_t)av.size() * sizeof(float));
        std::memcpy(ad.B.data<float>(), bv.data(), (size_t)bv.size() * sizeof(float));
        if (has_dora) {
            ad.magnitude = Tensor(Shape{ad.out_dim, 1}, DType::F32);
            if (!f.read((char*)ad.magnitude.data<float>(),
                        (std::streamsize)(ad.magnitude.numel() * sizeof(float))))
                return false;
            ad.magnitude.requires_grad(true);
        }
        ad.A.requires_grad(true);
        ad.B.requires_grad(true);
        adapters_.push_back(std::move(ad));
    }
    // A v2 file carries DoRA magnitudes, so the engine must treat the loaded
    // adapters as DoRA or merge_into_base() would silently drop the renormalise.
    if (has_dora) cfg_.use_dora = true;

    optimizer_ = Adafactor(cfg_.learning_rate, 0.999f, 1e-30f, cfg_.weight_decay);
    for (auto& ad : adapters_) {
        optimizer_.add_param(&ad.A);
        optimizer_.add_param(&ad.B);
    }
    freeze_base(true);
    return true;
}

// ============================================================================
// METHOD 3 — KnowledgeExpansionEngine
// ============================================================================

KnowledgeExpansionEngine::KnowledgeExpansionEngine(DenseModel* model)
    : model_(model), optimizer_(3e-4f, 0.999f, 1e-30f, 0.01f) {
    slots_.reserve(64);
}

void KnowledgeExpansionEngine::configure(const KnowledgeExpansionConfig& cfg) {
    cfg_ = cfg;
    // Small weight decay bounds slot-factor growth (the fixed gate already
    // bounds the contribution magnitude; decay prevents the factors from
    // drifting to pathological values over long training).
    optimizer_ = Adafactor(cfg_.learning_rate, 0.999f, 1e-30f, 0.01f);
}

void KnowledgeExpansionEngine::freeze_base(bool freeze) {
    std::vector<Tensor*> bp;
    if (model_) model_->get_parameters(bp);
    for (auto* p : bp) p->requires_grad(!freeze);
}

int64_t KnowledgeExpansionEngine::add_slot(const std::string& name, int64_t width) {
    if (!model_) return -1;
    int64_t D = model_->config.hidden_size;
    int64_t W = (width > 0) ? width : cfg_.slot_width;

    KnowledgeSlot s;
    s.name = name;
    s.hidden_size = D;
    s.width = W;
    s.w_in = Tensor(Shape{W, D}, DType::F32);
    s.w_out = Tensor(Shape{D, W}, DType::F32);
    s.gate = Tensor(Shape{1}, DType::F32);

    std::mt19937 rng((unsigned)std::hash<std::string>{}(name.empty() ? "slot" : name));
    std::normal_distribution<float> nd(0.0f, std::sqrt(2.0f / (float)std::max<int64_t>(1, D)));
    std::normal_distribution<float> ndo(0.0f, std::sqrt(1.0f / (float)std::max<int64_t>(1, W)));
    float* wd = s.w_in.data<float>();
    for (int64_t i = 0; i < s.w_in.numel(); ++i) wd[i] = (float)nd(rng);
    // w_out random, gate FIXED at 1.0 (tanh(1) ~ 0.76 — a constant
    // architectural scale, not a trained parameter): the slot factors get a
    // constant-strength gradient from the very first step, so they learn
    // immediately and deterministically. Weight decay (set on the optimizer)
    // bounds their growth; the fixed gate bounds the contribution magnitude.
    float* wod = s.w_out.data<float>();
    for (int64_t i = 0; i < s.w_out.numel(); ++i) wod[i] = (float)ndo(rng);
    s.gate.fill(1.0f);
    s.w_in.requires_grad(true);
    s.w_out.requires_grad(true);
    s.gate.requires_grad(false);

    slots_.push_back(std::move(s));
    return (int64_t)slots_.size() - 1;
}

Tensor KnowledgeExpansionEngine::extract_hidden(const Tensor& input_ids,
                                                const Tensor& positions,
                                                KVCache* cache) const {
    const TransformerConfig& cfg = model_->config;
    int64_t B = input_ids.dim(0), S = input_ids.dim(1);

    // Save/restore the engine's enabled state (see forward_with_adapters).
    const bool prev_enabled = AutogradEngine::enabled();
    AutogradEngine::set_enabled(true);
    Tensor h = model_->tok_embeddings->forward(input_ids.reshape(Shape{B * S}));
    h = h.reshape(Shape{B, S, cfg.hidden_size});

    KVCache local_cache;
    KVCache* active = cache;
    if (!active) {
        local_cache.init((int)cfg.num_layers, cfg.max_seq_len, cfg.num_heads, cfg.head_dim);
        active = &local_cache;
    }
    Tensor mask = make_causal_mask(B, S);
    for (int64_t i = 0; i < cfg.num_layers; ++i) {
        TransformerBlock& l = *model_->layers[(size_t)i];
        if (l.use_parallel_residual) {
            Tensor na = l.attention_norm.forward(h);
            Tensor nf = l.ffn_norm.forward(h);
            Tensor ao = l.attention.forward(na, positions, mask, *active, (int)i);
            Tensor fo = l.ffn.forward(nf);
            h = AutogradEngine::add_op(AutogradEngine::add_op(ao, fo), h);
        } else {
            Tensor ai = l.attention_norm.forward(h);
            Tensor ao = l.attention.forward(ai, positions, mask, *active, (int)i);
            ao = AutogradEngine::add_op(ao, h);
            Tensor fi = l.ffn_norm.forward(ao);
            Tensor fo = l.ffn.forward(fi);
            h = AutogradEngine::add_op(fo, ao);
        }
    }
    Tensor out = model_->norm->forward(h);
    AutogradEngine::set_enabled(prev_enabled);
    return out;
}

Tensor KnowledgeExpansionEngine::apply_slots(const Tensor& h) const {
    if (slots_.empty()) return h;
    int64_t B = h.dim(0), S = h.dim(1), D = h.dim(2);
    Tensor h2 = h.reshape(Shape{B * S, D});
    Tensor acc = h2;
    for (const auto& s : slots_) {
        if (s.w_in.numel() == 0) continue;
        Tensor proj = AutogradEngine::matmul_op(h2, s.w_in, B * S, s.width, D);
        Tensor act = AutogradEngine::silu_op(proj);
        Tensor mid = AutogradEngine::matmul_op(act, s.w_out, B * S, D, s.width);
        // Per-token RMS normalization of the slot output: the slot's
        // contribution is bounded to O(1) regardless of how large the slot
        // factors grow, so training is stable and the correction never blows
        // up the logits. (gamma = ones, differentiable via rms_norm_op.)
        Tensor ones(Shape{D}, DType::F32);
        ones.fill(1.0f);
        Tensor mid_norm = AutogradEngine::rms_norm_op(mid, ones, 1e-6f);
        Tensor gated = gated_mul_op(mid_norm, s.gate);
        acc = AutogradEngine::add_op(acc, gated);
    }
    return acc.reshape(Shape{B, S, D});
}

Tensor KnowledgeExpansionEngine::forward_with_slots(const Tensor& input_ids,
                                                    const Tensor& positions,
                                                    KVCache* cache) {
    if (!model_) return Tensor();
    Tensor h = extract_hidden(input_ids, positions, cache);
    Tensor hc = apply_slots(h);
    return model_->lm_head->forward(hc);
}

Tensor KnowledgeExpansionEngine::forward_guarded(const Tensor& input_ids,
                                                 const Tensor& positions,
                                                 KVCache* cache,
                                                 Tensor* confidence_out,
                                                 Tensor* slot_used_out) {
    if (!model_) return Tensor();
    Tensor h = extract_hidden(input_ids, positions, cache);
    int64_t B = h.dim(0), S = h.dim(1);
    int64_t V = model_->vocab_size();

    Tensor base_logits = model_->lm_head->forward(h);
    Tensor hc = apply_slots(h);
    Tensor corr_logits = model_->lm_head->forward(hc);

    Tensor out_logits(base_logits.shape(), DType::F32);
    Tensor conf(Shape{B, S}, DType::F32);
    Tensor used(Shape{B, S}, DType::F32);

    const float* bl = base_logits.data<float>();
    const float* cl = corr_logits.data<float>();
    float* ol = out_logits.data<float>();
    float* cd = conf.data<float>();
    float* ud = used.data<float>();

    // Average per-slot tanh gate activation (0 = silent, |tanh| <= 1). The
    // stored gate is a plain scalar; tanh bounds it for comparison.
    float gate_act = 0.0f;
    for (const auto& s : slots_) {
        const float g = s.gate.data<float>()[0];
        gate_act += (float)std::fabs(std::tanh(g));
    }
    gate_act /= (float)std::max<size_t>(1, slots_.size());

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < S; ++s) {
            int64_t row = b * S + s;
            const float* rb = bl + row * V;
            // argmax + max for numerical stability
            int64_t am = 0;
            float mx = rb[0];
            for (int64_t v = 1; v < V; ++v)
                if (rb[v] > mx) { mx = rb[v]; am = v; }
            double sum = 0.0;
            for (int64_t v = 0; v < V; ++v)
                sum += std::exp((double)rb[v] - (double)mx);
            float p_max = (float)(std::exp((double)rb[am] - (double)mx) / sum);

            bool consult = (p_max < cfg_.confidence_threshold) &&
                           (gate_act >= cfg_.slot_gate_threshold);
            const float* src = consult ? (cl + row * V) : (bl + row * V);
            std::memcpy(ol + row * V, src, (size_t)V * sizeof(float));
            cd[b * S + s] = p_max;
            ud[b * S + s] = consult ? 1.0f : 0.0f;
        }
    }

    if (confidence_out) *confidence_out = conf;
    if (slot_used_out) *slot_used_out = used;
    return out_logits;
}

void KnowledgeExpansionEngine::clear_grads() {
    for (auto& s : slots_) {
        if (s.w_in.has_grad()) s.w_in.zero_grad();
        if (s.w_out.has_grad()) s.w_out.zero_grad();
        if (s.gate.has_grad()) s.gate.zero_grad();
    }
}

void KnowledgeExpansionEngine::train_step(const Tensor& input_ids,
                                          const Tensor& positions,
                                          const Tensor& target_ids) {
    if (slots_.empty()) return;
    ++step_;
    auto& engine = AutogradEngine::instance();

    // Register any slots added since the last optimizer setup. The gate is a
    // fixed architectural constant (not a parameter), so it is not optimized.
    for (size_t i = opt_registered_; i < slots_.size(); ++i) {
        optimizer_.add_param(&slots_[i].w_in);
        optimizer_.add_param(&slots_[i].w_out);
        opt_registered_ = i + 1;
    }

    clear_grads();
    AutogradEngine::set_enabled(true);
    for (auto& s : slots_) {
        engine.register_parameter(&s.w_in);
        engine.register_parameter(&s.w_out);
    }
    Tensor logits = forward_with_slots(input_ids, positions);
    Tensor loss_t = AutogradEngine::cross_entropy_op(logits, target_ids);
    last_loss_ = *(const float*)loss_t.data();
    engine.backward(loss_t);
    engine.clear();
    AutogradEngine::set_enabled(false);

    float lr = cfg_.learning_rate;
    if (cfg_.warmup_steps > 0 && step_ <= cfg_.warmup_steps)
        lr = cfg_.learning_rate * (float)step_ / (float)cfg_.warmup_steps;
    optimizer_.set_lr(lr);
    optimizer_.set_grad_clip_norm(1.0f);
    optimizer_.step();
}

int64_t KnowledgeExpansionEngine::slot_param_count() const {
    int64_t n = 0;
    for (const auto& s : slots_) n += s.w_in.numel() + s.w_out.numel() + s.gate.numel();
    return n;
}

void KnowledgeExpansionEngine::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return;
    f.write("KEXP", 4);
    write_u32(f, 1);
    write_u32(f, (uint32_t)slots_.size());
    for (const auto& s : slots_) {
        write_u32(f, (uint32_t)s.name.size());
        if (!s.name.empty()) f.write(s.name.data(), (std::streamsize)s.name.size());
        int64_t hd = s.hidden_size;
        f.write((const char*)&hd, sizeof(hd));
        int64_t wd = s.width;
        f.write((const char*)&wd, sizeof(wd));
        write_raw_tensor(f, s.w_in.data<float>(), s.w_in.numel());
        write_raw_tensor(f, s.w_out.data<float>(), s.w_out.numel());
        write_raw_tensor(f, s.gate.data<float>(), s.gate.numel());
        write_f32(f, s.usage);
        write_f32(f, s.avg_confidence);
    }
}

bool KnowledgeExpansionEngine::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4] = {0, 0, 0, 0};
    f.read(magic, 4);
    if (std::memcmp(magic, "KEXP", 4) != 0) return false;
    uint32_t version = 0, count = 0;
    if (!read_u32(f, version) || version != 1) return false;
    if (!read_u32(f, count)) return false;

    slots_.clear();
    slots_.reserve(count ? count : 64);
    for (uint32_t i = 0; i < count; ++i) {
        KnowledgeSlot s;
        uint32_t nl = 0;
        if (!read_u32(f, nl)) return false;
        s.name.resize(nl);
        if (nl) f.read(&s.name[0], (std::streamsize)nl);
        int64_t hd = 0, wd = 0;
        if (!f.read((char*)&hd, sizeof(hd))) return false;
        if (!f.read((char*)&wd, sizeof(wd))) return false;
        s.hidden_size = hd;
        s.width = wd;
        s.w_in = Tensor(Shape{s.width, s.hidden_size}, DType::F32);
        s.w_out = Tensor(Shape{s.hidden_size, s.width}, DType::F32);
        s.gate = Tensor(Shape{1}, DType::F32);
        std::vector<float> wi, wo, ga;
        if (!read_raw_tensor(f, wi) || !read_raw_tensor(f, wo) || !read_raw_tensor(f, ga))
            return false;
        if ((int64_t)wi.size() != s.w_in.numel() || (int64_t)wo.size() != s.w_out.numel() ||
            (int64_t)ga.size() != s.gate.numel())
            return false;
        std::memcpy(s.w_in.data<float>(), wi.data(), (size_t)wi.size() * sizeof(float));
        std::memcpy(s.w_out.data<float>(), wo.data(), (size_t)wo.size() * sizeof(float));
        std::memcpy(s.gate.data<float>(), ga.data(), (size_t)ga.size() * sizeof(float));
        read_f32(f, s.usage);
        read_f32(f, s.avg_confidence);
        s.w_in.requires_grad(true);
        s.w_out.requires_grad(true);
        s.gate.requires_grad(false);
        slots_.push_back(std::move(s));
    }

    optimizer_ = Adafactor(cfg_.learning_rate, 0.999f, 1e-30f, 0.01f);
    for (auto& s : slots_) {
        optimizer_.add_param(&s.w_in);
        optimizer_.add_param(&s.w_out);
    }
    opt_registered_ = slots_.size();
    freeze_base(true);
    return true;
}

} // namespace quant
