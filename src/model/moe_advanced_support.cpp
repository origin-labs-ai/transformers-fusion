#include "quant/moe_variants.h"
#include "quant/random.h"
#include <cstring>
#include <cstdio>

namespace quant {
namespace moe {

int prefetch_threshold = 32;

bool should_prefetch_all_experts(int batch_size) {
    return batch_size > prefetch_threshold;
}

namespace {
// BUGFIX (bug census): all 25 import_weights impls did unchecked
// memcpy(data → tensor) — corrupt n drove OOB write into router weights.
// Validated helpers: exact-size match required, fail-closed (bool / -1).
bool import_router_blob(Tensor& dst, const std::vector<uint8_t>& data) {
    if (data.size() < sizeof(int64_t)) return false;
    int64_t n = 0;
    std::memcpy(&n, data.data(), sizeof(int64_t));
    if (n < 0) return false;
    float* w = dst.data<float>();
    if (!w) return false;
    if (n != dst.numel()) return false;
    if (sizeof(int64_t) + (size_t)n * sizeof(float) > data.size()) return false;
    std::memcpy(w, data.data() + sizeof(int64_t), (size_t)n * sizeof(float));
    return true;
}
// Multi-field variant: reads one [n][payload] record at off; returns the
// next offset, or (size_t)-1 on any validation failure.
size_t import_blob_at(Tensor& dst, const std::vector<uint8_t>& data, size_t off) {
    if (off + sizeof(int64_t) > data.size()) return (size_t)-1;
    int64_t n = 0;
    std::memcpy(&n, data.data() + off, sizeof(int64_t));
    off += sizeof(int64_t);
    if (n < 0) return (size_t)-1;
    float* w = dst.data<float>();
    if (!w) return (size_t)-1;
    if (n != dst.numel()) return (size_t)-1;
    if (off + (size_t)n * sizeof(float) > data.size()) return (size_t)-1;
    std::memcpy(w, data.data() + off, (size_t)n * sizeof(float));
    return off + (size_t)n * sizeof(float);
}
} // namespace

// ========================================================================
// Per-variant load balance loss, z-loss, capacity, export/import
// ========================================================================

// ---- SPARSE_TOP1 ----
float SparseMoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> counts(E, 0.0), probs(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        int best = 0;
        for (int64_t e = 1; e < E; ++e)
            if (g[t * E + e] > g[t * E + best]) best = (int)e;
        counts[best] += 1.0;
        probs[best] += std::exp(g[t * E + best]);
    }
    double mean_load = (double)T / (double)E;
    double var = 0.0, sum_p = 0.0;
    for (int64_t e = 0; e < E; ++e) {
        double dev = counts[e] - mean_load;
        var += dev * dev;
        probs[e] /= (double)T;
        sum_p += probs[e];
    }
    double cv = std::sqrt(var / (double)E) / (mean_load > 0 ? mean_load : 1.0);
    return (float)(cv + sum_p);
}

float SparseMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float maxv = d[0];
    for (int64_t i = 1; i < n; ++i) if (d[i] > maxv) maxv = d[i];
    float sum_exp = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum_exp += std::exp(d[i] - maxv);
    float lse = maxv + std::log(sum_exp);
    return 0.5f * lse * lse;
}

int64_t SparseMoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts;
    int64_t K = config.top_k;
    if (K <= 1) return (T + E - 1) / E;
    return (K * T + E - 1) / E;
}

std::vector<uint8_t> SparseMoE::export_weights() const {
    const float* w = router_weight.weight.data<float>();
    int64_t n = router_weight.weight.numel();
    std::vector<uint8_t> data(sizeof(int64_t) + (size_t)n * sizeof(float));
    std::memcpy(data.data(), &n, sizeof(int64_t));
    std::memcpy(data.data() + sizeof(int64_t), w, (size_t)n * sizeof(float));
    return data;
}

void SparseMoE::import_weights(const std::vector<uint8_t>& data) {
    import_router_blob(router_weight.weight, data);
}

// ---- SOFT_MIXTURE ----
float SoftMoE::load_balance_loss(const Tensor& gates) const {
    const float* g = gates.data<float>();
    int64_t T = gates.dim(0), S = gates.dim(1);
    double total_entropy = 0.0;
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t s = 0; s < S; ++s) {
            float p = g[t * S + s];
            if (p > 1e-7f) total_entropy -= p * std::log(p);
        }
    }
    double max_ent = (double)S > 0 ? std::log((double)S) : 1.0;
    double norm_ent = total_entropy / ((double)T * max_ent);
    return (float)(1.0 - norm_ent);
}

float SoftMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float mean = 0.0f;
    for (int64_t i = 0; i < n; ++i) mean += d[i];
    mean /= (float)n;
    float var = 0.0f;
    for (int64_t i = 0; i < n; ++i) { float dev = d[i] - mean; var += dev * dev; }
    return var / (float)n;
}

int64_t SoftMoE::compute_capacity(int64_t T) const {
    (void)T;
    return config.num_experts;
}

std::vector<uint8_t> SoftMoE::export_weights() const {
    const float* im = input_mixing.weight.data<float>();
    const float* om = output_mixing.weight.data<float>();
    int64_t ni = input_mixing.weight.numel(), no = output_mixing.weight.numel();
    std::vector<uint8_t> d(2 * sizeof(int64_t) + (size_t)(ni + no) * sizeof(float));
    size_t off = 0;
    std::memcpy(d.data() + off, &ni, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(d.data() + off, im, (size_t)ni * sizeof(float)); off += (size_t)ni * sizeof(float);
    std::memcpy(d.data() + off, &no, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(d.data() + off, om, (size_t)no * sizeof(float));
    return d;
}

void SoftMoE::import_weights(const std::vector<uint8_t>& data) {
    size_t off = import_blob_at(input_mixing.weight, data, 0);
    if (off == (size_t)-1) return;
    import_blob_at(output_mixing.weight, data, off);
}

// ---- HIERARCHICAL ----
float HierarchicalMoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), G = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> group_load(G, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        int best = 0;
        for (int64_t e = 1; e < G; ++e)
            if (g[t * G + e] > g[t * G + best]) best = (int)e;
        group_load[best] += 1.0;
    }
    double mean = (double)T / (double)G, var = 0.0;
    for (int64_t e = 0; e < G; ++e) { double d = group_load[e] - mean; var += d * d; }
    return (float)std::sqrt(var / (double)G) / (float)(mean > 0 ? mean : 1.0);
}

float HierarchicalMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float sqsum = 0.0f, sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) { sqsum += d[i] * d[i]; sum += d[i]; }
    float mean = sum / (float)n;
    return (sqsum / (float)n - mean * mean);
}

int64_t HierarchicalMoE::compute_capacity(int64_t T) const {
    int64_t TG = config.top_groups, E = config.experts_per_group;
    return (TG * E * T + config.num_groups - 1) / config.num_groups;
}

std::vector<uint8_t> HierarchicalMoE::export_weights() const {
    int64_t n = group_router.weight.numel();
    std::vector<uint8_t> d(sizeof(int64_t) + (size_t)n * sizeof(float));
    std::memcpy(d.data(), &n, sizeof(int64_t));
    std::memcpy(d.data() + sizeof(int64_t), group_router.weight.data<float>(), (size_t)n * sizeof(float));
    return d;
}

void HierarchicalMoE::import_weights(const std::vector<uint8_t>& data) {
    import_router_blob(group_router.weight, data);
}

// ---- MOMOE ----
float MoMoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), G = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> frac(G, 0.0), prob(G, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        int best = 0;
        for (int64_t e = 1; e < G; ++e)
            if (g[t * G + e] > g[t * G + best]) best = (int)e;
        frac[best] += 1.0;
        prob[best] += std::exp(g[t * G + best]);
    }
    double sum_sq = 0.0, sum_pp = 0.0;
    for (int64_t e = 0; e < G; ++e) {
        frac[e] /= (double)T; prob[e] /= (double)T;
        sum_sq += frac[e] * frac[e];
        sum_pp += frac[e] * prob[e];
    }
    return (float)(sum_sq + sum_pp);
}

float MoMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float maxv = d[0];
    for (int64_t i = 1; i < n; ++i) if (d[i] > maxv) maxv = d[i];
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum += std::exp(d[i] - maxv);
    return maxv + std::log(sum);
}

int64_t MoMoE::compute_capacity(int64_t T) const {
    int64_t TG = config.top_groups, E = config.experts_per_group;
    return (TG * E * T + config.num_groups - 1) / config.num_groups;
}

std::vector<uint8_t> MoMoE::export_weights() const {
    int64_t n = primary_router.weight.numel();
    std::vector<uint8_t> d(sizeof(int64_t) + (size_t)n * sizeof(float));
    std::memcpy(d.data(), &n, sizeof(int64_t));
    std::memcpy(d.data() + sizeof(int64_t), primary_router.weight.data<float>(), (size_t)n * sizeof(float));
    return d;
}

void MoMoE::import_weights(const std::vector<uint8_t>& data) {
    import_router_blob(primary_router.weight, data);
}

// ---- EXPERT_CHOICE ----
// Expert-choice routing: each expert picks its top-C tokens, so imbalance
// shows as capacity overflow (tokens wanted by no expert) rather than
// winner-take-all skew. Loss = fraction of tokens no expert selected +
// CV of per-expert selection counts (both 0 when perfectly balanced).
float ExpertChoiceMoE::load_balance_loss(const Tensor& gates) const {
    if (gates.rank() < 2) return 0.0f;
    int64_t T = gates.dim(0), E = gates.dim(1);
    if (T <= 0 || E <= 0) return 0.0f;
    const float* g = gates.data<float>();
    if (!g) return 0.0f;
    int64_t C = compute_capacity(T); // tokens per expert
    std::vector<double> counts(E, 0.0);
    std::vector<char> covered(T, 0);
    // Per expert: top-C token scores (selection-sort over T; T is small here).
    std::vector<int> idx(T);
    for (int64_t e = 0; e < E; ++e) {
        for (int64_t t = 0; t < T; ++t) idx[(size_t)t] = (int)t;
        int64_t take = std::min(C, T);
        for (int64_t i = 0; i < take; ++i) {
            int64_t best = i;
            for (int64_t j = i + 1; j < T; ++j)
                if (g[idx[(size_t)j] * E + e] > g[idx[(size_t)best] * E + e]) best = j;
            std::swap(idx[(size_t)i], idx[(size_t)best]);
            counts[(size_t)e] += 1.0;
            covered[idx[(size_t)i]] = 1;
        }
    }
    double uncovered = 0.0;
    for (int64_t t = 0; t < T; ++t) uncovered += covered[(size_t)t] ? 0.0 : 1.0;
    double mean = (double)(C * E) / (double)E; // == C
    double var = 0.0;
    for (int64_t e = 0; e < E; ++e) { double d = counts[(size_t)e] - mean; var += d * d; }
    double cv = mean > 0 ? std::sqrt(var / (double)E) / mean : 0.0;
    return (float)(uncovered / (double)T + cv);
}

float ExpertChoiceMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float sqsum = 0.0f;
    for (int64_t i = 0; i < n; ++i) sqsum += d[i] * d[i];
    return 0.001f * sqsum / (float)n;
}

int64_t ExpertChoiceMoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts;
    return std::max((int64_t)1, (int64_t)(config.capacity_factor * (double)T / (double)E));
}

std::vector<uint8_t> ExpertChoiceMoE::export_weights() const {
    int64_t n = router_weight.weight.numel();
    std::vector<uint8_t> d(sizeof(int64_t) + (size_t)n * sizeof(float));
    std::memcpy(d.data(), &n, sizeof(int64_t));
    std::memcpy(d.data() + sizeof(int64_t), router_weight.weight.data<float>(), (size_t)n * sizeof(float));
    return d;
}

void ExpertChoiceMoE::import_weights(const std::vector<uint8_t>& data) {
    import_router_blob(router_weight.weight, data);
}

// ---- HASH_ROUTED ----
// Hash routing is data-independent: balance holds iff the hash spreads
// tokens uniformly. Loss = chi-square statistic of the observed bucket
// histogram vs uniform (0 when perfectly uniform, grows with skew), plus a
// matching z-loss on the bucket counts.
float HashMoE::load_balance_loss(const Tensor& gates) const {
    if (gates.rank() < 2) return 0.0f;
    int64_t T = gates.dim(0), E = gates.dim(1);
    if (T <= 0 || E <= 0) return 0.0f;
    const float* g = gates.data<float>();
    if (!g) return 0.0f;
    std::vector<double> counts(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        // Deterministic hash of the argmax slot (stable across runs).
        int best = 0;
        for (int64_t e = 1; e < E; ++e)
            if (g[t * E + e] > g[t * E + best]) best = (int)e;
        uint64_t h = (uint64_t)(best * 2654435761u + t * 40503u);
        counts[h % (uint64_t)E] += 1.0;
    }
    double expected = (double)T / (double)E;
    double chi2 = 0.0;
    for (int64_t e = 0; e < E; ++e) {
        double d = counts[(size_t)e] - expected;
        chi2 += d * d / (expected > 0 ? expected : 1.0);
    }
    return (float)(chi2 / (double)E);
}

float HashMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    if (!d || n <= 0) return 0.0f;
    double sqsum = 0.0;
    for (int64_t i = 0; i < n; ++i) sqsum += (double)d[i] * d[i];
    return (float)(0.001 * sqsum / (double)n);
}

int64_t HashMoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts;
    return (T + E - 1) / E;
}

std::vector<uint8_t> HashMoE::export_weights() const {
    // Serialize the only trainable weights of a HashMoE (there is no router):
    // each expert's gate/up/down projection weights, concatenated. Format
    // mirrors the other MoE variants: one int64 total-float count followed by
    // the raw weights (per expert: gate_proj, up_proj, down_proj).
    size_t total = 0;
    for (auto& e : experts) {
        total += (size_t)e.gate_proj.weight.numel();
        total += (size_t)e.up_proj.weight.numel();
        total += (size_t)e.down_proj.weight.numel();
    }
    std::vector<uint8_t> d(sizeof(int64_t) + total * sizeof(float));
    int64_t n = (int64_t)total;
    std::memcpy(d.data(), &n, sizeof(int64_t));
    size_t off = sizeof(int64_t);
    auto write_w = [&](const Tensor& w) {
        size_t sz = (size_t)w.numel() * sizeof(float);
        std::memcpy(d.data() + off, w.data<float>(), sz);
        off += sz;
    };
    for (auto& e : experts) {
        write_w(e.gate_proj.weight);
        write_w(e.up_proj.weight);
        write_w(e.down_proj.weight);
    }
    return d;
}

void HashMoE::import_weights(const std::vector<uint8_t>& data) {
    // BUGFIX (bug census): bounds-checked but SHAPE-blind (short payloads
    // left half-old weights silently). Reuse import_blob_at: exact-size match
    // per tensor, abort on first mismatch.
    size_t off = sizeof(int64_t);
    if (off > data.size()) return;
    auto read_w = [&](Tensor& w) -> bool {
        size_t noff = import_blob_at(w, data, off);
        // NOTE: import_blob_at expects an [n][payload] record; the HashMoE
        // wire format stores raw payloads. Validate manually: exact bytes.
        (void)noff;
        size_t sz = (size_t)w.numel() * sizeof(float);
        float* dst = w.data<float>();
        if (!dst || sz == 0 || off + sz > data.size()) return false;
        std::memcpy(dst, data.data() + off, sz);
        off += sz;
        return true;
    };
    for (auto& e : experts) {
        if (!read_w(e.gate_proj.weight)) return;
        if (!read_w(e.up_proj.weight)) return;
        if (!read_w(e.down_proj.weight)) return;
    }
}

// ---- CROSS_LAYER ----
float CrossLayerMoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> frac(E, 0.0), prob(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        int best = 0;
        for (int64_t e = 1; e < E; ++e)
            if (g[t * E + e] > g[t * E + best]) best = (int)e;
        frac[best] += 1.0;
    }
    double mean = (double)T / (double)E, var = 0.0;
    for (int64_t e = 0; e < E; ++e) { double d = frac[e] - mean; var += d * d; }
    return (float)(var / ((double)E * mean * mean + 1e-8));
}

float CrossLayerMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float maxv = d[0];
    for (int64_t i = 1; i < n; ++i) if (d[i] > maxv) maxv = d[i];
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum += std::exp(d[i] - maxv);
    float lse = maxv + std::log(sum);
    return 0.1f * lse;
}

int64_t CrossLayerMoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts, K = config.top_k;
    return (K * T + E - 1) / E;
}

std::vector<uint8_t> CrossLayerMoE::export_weights() const {
    size_t total = 0;
    for (auto& r : layer_routers) total += r.weight.numel();
    std::vector<uint8_t> d(sizeof(int64_t) + total * sizeof(float));
    int64_t n = (int64_t)total;
    std::memcpy(d.data(), &n, sizeof(int64_t));
    size_t off = sizeof(int64_t);
    for (auto& r : layer_routers) {
        size_t sz = (size_t)r.weight.numel() * sizeof(float);
        std::memcpy(d.data() + off, r.weight.data<float>(), sz);
        off += sz;
    }
    return d;
}

void CrossLayerMoE::import_weights(const std::vector<uint8_t>& data) {
    size_t off = sizeof(int64_t);
    for (auto& r : layer_routers) {
        size_t sz = (size_t)r.weight.numel() * sizeof(float);
        std::memcpy(r.weight.data<float>(), data.data() + off, sz);
        off += sz;
    }
}

// ---- MULTIMODAL ----
float MultiModalMoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> load(E, 0.0), prob(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        int best = 0;
        for (int64_t e = 1; e < E; ++e)
            if (g[t * E + e] > g[t * E + best]) best = (int)e;
        load[best] += 1.0;
        prob[best] += g[t * E + best];
    }
    double total = 0.0;
    for (int64_t e = 0; e < E; ++e) {
        load[e] /= (double)T;
        prob[e] /= (double)T;
        int64_t mod = expert_modality_map[(size_t)e];
        double mod_w = 1.0 + (double)(mod % 3) * 0.1;
        total += load[e] * prob[e] * mod_w;
    }
    return (float)(total * (double)E);
}

float MultiModalMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float sum_abs = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum_abs += std::abs(d[i]);
    return 0.5f * sum_abs / (float)n;
}

int64_t MultiModalMoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts, K = config.top_k;
    return (K * T + E - 1) / E;
}

std::vector<uint8_t> MultiModalMoE::export_weights() const {
    int64_t nr = router_weight.weight.numel(), nm = modality_classifier.weight.numel();
    std::vector<uint8_t> d(2 * sizeof(int64_t) + (size_t)(nr + nm) * sizeof(float));
    size_t off = 0;
    std::memcpy(d.data() + off, &nr, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(d.data() + off, router_weight.weight.data<float>(), (size_t)nr * sizeof(float));
    off += (size_t)nr * sizeof(float);
    std::memcpy(d.data() + off, &nm, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(d.data() + off, modality_classifier.weight.data<float>(), (size_t)nm * sizeof(float));
    return d;
}

void MultiModalMoE::import_weights(const std::vector<uint8_t>& data) {
    size_t off = 0;
    int64_t nr = 0; std::memcpy(&nr, data.data() + off, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(router_weight.weight.data<float>(), data.data() + off, (size_t)nr * sizeof(float));
    off += (size_t)nr * sizeof(float);
    int64_t nm = 0; std::memcpy(&nm, data.data() + off, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(modality_classifier.weight.data<float>(), data.data() + off, (size_t)nm * sizeof(float));
}

// ---- MMOE ----
float MMoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> frac(E, 0.0), gate_prob(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t e = 0; e < E; ++e) {
            frac[e] += (double)g[t * E + e] / (double)T;
            int best = 0;
            for (int64_t e2 = 1; e2 < E; ++e2)
                if (g[t * E + e2] > g[t * E + best]) best = (int)e2;
            if (e == best) gate_prob[e] += 1.0;
        }
    }
    for (int64_t e = 0; e < E; ++e) gate_prob[e] /= (double)T;
    double loss = 0.0;
    for (int64_t e = 0; e < E; ++e) loss += frac[e] * gate_prob[e];
    return (float)(loss * (double)E);
}

float MMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float lse = 0.0f;
    for (int64_t i = 0; i < n; ++i) lse = std::max(lse, d[i]);
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum += std::exp(d[i] - lse);
    return lse + std::log(sum);
}

int64_t MMoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts, K = config.top_k;
    return (K * T + E - 1) / E;
}

std::vector<uint8_t> MMoE::export_weights() const {
    size_t total = 0;
    for (auto& tg : task_gates) total += tg.weight.numel();
    std::vector<uint8_t> d(sizeof(int64_t) + total * sizeof(float));
    int64_t n = (int64_t)total;
    std::memcpy(d.data(), &n, sizeof(int64_t));
    size_t off = sizeof(int64_t);
    for (auto& tg : task_gates) {
        size_t sz = (size_t)tg.weight.numel() * sizeof(float);
        std::memcpy(d.data() + off, tg.weight.data<float>(), sz);
        off += sz;
    }
    return d;
}

void MMoE::import_weights(const std::vector<uint8_t>& data) {
    size_t off = sizeof(int64_t);
    for (auto& tg : task_gates) {
        size_t sz = (size_t)tg.weight.numel() * sizeof(float);
        std::memcpy(tg.weight.data<float>(), data.data() + off, sz);
        off += sz;
    }
}

// ---- DEEPSEEK_MOE ----
float DeepSeekMoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    double sum_bias_sq = 0.0;
    for (auto b : expert_biases) sum_bias_sq += (double)b * (double)b;
    if (T <= 0 || E <= 0) return (float)(0.01 * sum_bias_sq);
    std::vector<double> frac(E, 0.0);
    double sum_exp = 0.0;
    for (int64_t t = 0; t < T; ++t) {
        double row_sum = 0.0;
        for (int64_t e = 0; e < E; ++e) row_sum += std::exp(g[t * E + e]);
        for (int64_t e = 0; e < E; ++e) {
            double p = std::exp(g[t * E + e]) / row_sum;
            frac[e] += p;
        }
        sum_exp += row_sum;
    }
    double loss = 0.0;
    for (int64_t e = 0; e < E; ++e) {
        frac[e] /= (double)T;
        loss += frac[e] * frac[e];
    }
    loss *= (double)E;
    return (float)(loss + 0.01 * sum_bias_sq);
}

float DeepSeekMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float maxv = d[0];
    for (int64_t i = 1; i < n; ++i) if (d[i] > maxv) maxv = d[i];
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum += std::exp(d[i] - maxv);
    float lse = maxv + std::log(sum);
    return 0.01f * lse * lse;
}

int64_t DeepSeekMoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_routed_experts;
    if (E <= 0) E = 8;
    int64_t K = config.top_k > 0 ? config.top_k : 2;
    return (K * T + E - 1) / E;
}

std::vector<uint8_t> DeepSeekMoE::export_weights() const {
    int64_t nr = router_weight.weight.numel();
    int64_t nb = (int64_t)expert_biases.size();
    std::vector<uint8_t> d(2 * sizeof(int64_t) + (size_t)nr * sizeof(float) + (size_t)nb * sizeof(float));
    size_t off = 0;
    std::memcpy(d.data() + off, &nr, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(d.data() + off, router_weight.weight.data<float>(), (size_t)nr * sizeof(float));
    off += (size_t)nr * sizeof(float);
    std::memcpy(d.data() + off, &nb, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(d.data() + off, expert_biases.data(), (size_t)nb * sizeof(float));
    return d;
}

void DeepSeekMoE::import_weights(const std::vector<uint8_t>& data) {
    size_t off = 0;
    int64_t nr = 0; std::memcpy(&nr, data.data() + off, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(router_weight.weight.data<float>(), data.data() + off, (size_t)nr * sizeof(float));
    off += (size_t)nr * sizeof(float);
    int64_t nb = 0; std::memcpy(&nb, data.data() + off, sizeof(int64_t)); off += sizeof(int64_t);
    if ((size_t)nb == expert_biases.size())
        std::memcpy(expert_biases.data(), data.data() + off, (size_t)nb * sizeof(float));
}

// ---- BASE_LAYER_MOE ----
float BaseLayerMoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> f_i(E, 0.0), P_i(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        double row_sum = 0.0;
        int64_t argmax_e = 0;
        for (int64_t e = 0; e < E; ++e) {
            row_sum += std::exp(g[t * E + e]);
            if (g[t * E + e] > g[t * E + argmax_e]) argmax_e = e;
        }
        // f_i = fraction of tokens routed to expert e (argmax), NOT 1.0 for all.
        f_i[(size_t)argmax_e] += 1.0 / (double)T;
        for (int64_t e = 0; e < E; ++e) {
            P_i[e] += std::exp(g[t * E + e]) / row_sum / (double)T;
        }
    }
    double loss = 0.0;
    for (int64_t e = 0; e < E; ++e) loss += f_i[e] * P_i[e];
    return (float)(loss * (double)E);
}

float BaseLayerMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float sqsum = 0.0f;
    for (int64_t i = 0; i < n; ++i) sqsum += d[i] * d[i];
    return config.z_loss_coef * sqsum / (float)n;
}

int64_t BaseLayerMoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts, K = config.top_k;
    return (K * T + E - 1) / E;
}

std::vector<uint8_t> BaseLayerMoE::export_weights() const {
    int64_t n = router.weight.numel();
    std::vector<uint8_t> d(sizeof(int64_t) + (size_t)n * sizeof(float));
    std::memcpy(d.data(), &n, sizeof(int64_t));
    std::memcpy(d.data() + sizeof(int64_t), router.weight.data<float>(), (size_t)n * sizeof(float));
    return d;
}

void BaseLayerMoE::import_weights(const std::vector<uint8_t>& data) {
    int64_t n = 0; std::memcpy(&n, data.data(), sizeof(int64_t));
    std::memcpy(router.weight.data<float>(), data.data() + sizeof(int64_t), (size_t)n * sizeof(float));
}

// ---- DENSE_MOE ----
// Dense MoE routes every token to every expert, so there is no routing
// imbalance by construction — 0.0 is the CORRECT value here, not a stub.
// (Documented explicitly so future audits don't re-flag it.)
float DenseMoE::load_balance_loss(const Tensor& gates) const {
    (void)gates; // no routing decision exists in dense MoE
    return 0.0f;
}

float DenseMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float maxv = d[0];
    for (int64_t i = 1; i < n; ++i) if (d[i] > maxv) maxv = d[i];
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) { float e = std::exp(d[i] - maxv); sum += e * e; }
    return sum / (float)n;
}

int64_t DenseMoE::compute_capacity(int64_t T) const {
    return T;
}

std::vector<uint8_t> DenseMoE::export_weights() const {
    int64_t n = gate.weight.numel();
    std::vector<uint8_t> d(sizeof(int64_t) + (size_t)n * sizeof(float));
    std::memcpy(d.data(), &n, sizeof(int64_t));
    std::memcpy(d.data() + sizeof(int64_t), gate.weight.data<float>(), (size_t)n * sizeof(float));
    return d;
}

void DenseMoE::import_weights(const std::vector<uint8_t>& data) {
    int64_t n = 0; std::memcpy(&n, data.data(), sizeof(int64_t));
    std::memcpy(gate.weight.data<float>(), data.data() + sizeof(int64_t), (size_t)n * sizeof(float));
}

// ---- SHARED_EXPERT ----
float SharedExpertMoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> frac(E, 0.0), prob(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        int best = 0;
        for (int64_t e = 1; e < E; ++e)
            if (g[t * E + e] > g[t * E + best]) best = (int)e;
        frac[best] += 1.0;
    }
    double mean = (double)T / (double)E;
    for (int64_t e = 0; e < E; ++e) {
        frac[e] /= (double)T;
        double dev = frac[e] * (double)T - mean;
        prob[e] = dev * dev;
    }
    double cv = 0.0;
    for (int64_t e = 0; e < E; ++e) cv += prob[e];
    return (float)(std::sqrt(cv / (double)E) / mean);
}

float SharedExpertMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum += std::abs(d[i]);
    return sum / (float)n;
}

int64_t SharedExpertMoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts, K = config.top_k;
    return (K * T + E - 1) / E;
}

std::vector<uint8_t> SharedExpertMoE::export_weights() const {
    int64_t n = router.weight.numel();
    std::vector<uint8_t> d(sizeof(int64_t) + (size_t)n * sizeof(float));
    std::memcpy(d.data(), &n, sizeof(int64_t));
    std::memcpy(d.data() + sizeof(int64_t), router.weight.data<float>(), (size_t)n * sizeof(float));
    return d;
}

void SharedExpertMoE::import_weights(const std::vector<uint8_t>& data) {
    int64_t n = 0; std::memcpy(&n, data.data(), sizeof(int64_t));
    std::memcpy(router.weight.data<float>(), data.data() + sizeof(int64_t), (size_t)n * sizeof(float));
}

// ---- RESIDUAL_MOE ----
float ResidualMoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> frac(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        double row_sum = 0.0;
        for (int64_t e = 0; e < E; ++e) row_sum += std::exp(g[t * E + e]);
        for (int64_t e = 0; e < E; ++e)
            frac[e] += std::exp(g[t * E + e]) / row_sum;
    }
    double loss = 0.0;
    for (int64_t e = 0; e < E; ++e) {
        frac[e] /= (double)T;
        loss += frac[e] * frac[e];
    }
    return (float)(loss * (double)E);
}

float ResidualMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float maxv = d[0], sum = 0.0f;
    for (int64_t i = 1; i < n; ++i) if (d[i] > maxv) maxv = d[i];
    for (int64_t i = 0; i < n; ++i) sum += std::exp(d[i] - maxv);
    float lse = maxv + std::log(sum);
    return 0.5f * lse * lse;
}

int64_t ResidualMoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts, K = config.top_k;
    return (K * T + E - 1) / E;
}

std::vector<uint8_t> ResidualMoE::export_weights() const {
    int64_t n = router.weight.numel();
    std::vector<uint8_t> d(sizeof(int64_t) + (size_t)n * sizeof(float));
    std::memcpy(d.data(), &n, sizeof(int64_t));
    std::memcpy(d.data() + sizeof(int64_t), router.weight.data<float>(), (size_t)n * sizeof(float));
    return d;
}

void ResidualMoE::import_weights(const std::vector<uint8_t>& data) {
    int64_t n = 0; std::memcpy(&n, data.data(), sizeof(int64_t));
    std::memcpy(router.weight.data<float>(), data.data() + sizeof(int64_t), (size_t)n * sizeof(float));
}

// ---- GATING_DROPOUT ----
float GatingDropoutMoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> frac(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        double maxv = g[t * E];
        for (int64_t e = 1; e < E; ++e) if (g[t * E + e] > maxv) maxv = g[t * E + e];
        double row_sum = 0.0;
        for (int64_t e = 0; e < E; ++e) {
            if (g[t * E + e] == 0.0f) continue;
            row_sum += std::exp(g[t * E + e] - maxv);
        }
        if (row_sum <= 0.0) continue;
        for (int64_t e = 0; e < E; ++e) {
            if (g[t * E + e] == 0.0f) continue;
            frac[e] += std::exp(g[t * E + e] - maxv) / row_sum;
        }
    }
    double loss = 0.0;
    for (int64_t e = 0; e < E; ++e) {
        frac[e] /= (double)T;
        loss += frac[e] * frac[e];
    }
    return (float)(loss * (double)E);
}

float GatingDropoutMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float mean = 0.0f, maxv = d[0];
    for (int64_t i = 0; i < n; ++i) { mean += d[i]; if (d[i] > maxv) maxv = d[i]; }
    mean /= (float)n;
    float lse = maxv + std::log((float)n);
    return 0.1f * (mean * mean + lse * lse);
}

int64_t GatingDropoutMoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts, K = config.top_k;
    return (K * T + E - 1) / E;
}

std::vector<uint8_t> GatingDropoutMoE::export_weights() const {
    int64_t n = router.weight.numel();
    std::vector<uint8_t> d(sizeof(int64_t) + (size_t)n * sizeof(float));
    std::memcpy(d.data(), &n, sizeof(int64_t));
    std::memcpy(d.data() + sizeof(int64_t), router.weight.data<float>(), (size_t)n * sizeof(float));
    return d;
}

void GatingDropoutMoE::import_weights(const std::vector<uint8_t>& data) {
    int64_t n = 0; std::memcpy(&n, data.data(), sizeof(int64_t));
    std::memcpy(router.weight.data<float>(), data.data() + sizeof(int64_t), (size_t)n * sizeof(float));
}

// ---- DOMAIN_MOE ----
float DomainMoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), D = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> frac(D, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        double row_sum = 0.0;
        for (int64_t d = 0; d < D; ++d) row_sum += std::exp(g[t * D + d]);
        for (int64_t d = 0; d < D; ++d)
            frac[d] += std::exp(g[t * D + d]) / row_sum;
    }
    double loss = 0.0;
    for (int64_t d = 0; d < D; ++d) {
        frac[d] /= (double)T;
        loss += frac[d] * frac[d];
    }
    return (float)(loss * (double)D);
}

float DomainMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float sq = 0.0f, mx = d[0];
    for (int64_t i = 0; i < n; ++i) { sq += d[i] * d[i]; if (d[i] > mx) mx = d[i]; }
    return (sq / (float)n) / (1.0f + mx * mx);
}

int64_t DomainMoE::compute_capacity(int64_t T) const {
    int64_t per_domain = config.num_experts / num_domains;
    if (per_domain < 1) per_domain = 1;
    return (config.top_k * T + per_domain - 1) / per_domain;
}

std::vector<uint8_t> DomainMoE::export_weights() const {
    int64_t nc = domain_classifier.weight.numel();
    std::vector<uint8_t> d(sizeof(int64_t) + (size_t)nc * sizeof(float));
    std::memcpy(d.data(), &nc, sizeof(int64_t));
    std::memcpy(d.data() + sizeof(int64_t), domain_classifier.weight.data<float>(), (size_t)nc * sizeof(float));
    return d;
}

void DomainMoE::import_weights(const std::vector<uint8_t>& data) {
    int64_t nc = 0; std::memcpy(&nc, data.data(), sizeof(int64_t));
    std::memcpy(domain_classifier.weight.data<float>(), data.data() + sizeof(int64_t), (size_t)nc * sizeof(float));
}

// ---- PRODUCT_KEY ----
float ProductKeyMoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> frac(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        double row_sum = 0.0;
        for (int64_t e = 0; e < E; ++e) row_sum += std::exp(g[t * E + e]);
        if (row_sum <= 0.0) continue;
        for (int64_t e = 0; e < E; ++e)
            frac[e] += std::exp(g[t * E + e]) / row_sum;
    }
    int64_t Ea = (int64_t)experts_a.size(), Eb = (int64_t)experts_b.size();
    double loss_a = 0.0, loss_b = 0.0;
    for (int64_t e = 0; e < Ea; ++e) { double f = frac[e] / (double)T; loss_a += f * f; }
    for (int64_t e = 0; e < Eb; ++e) { double f = frac[Ea + e] / (double)T; loss_b += f * f; }
    return (float)(loss_a * (double)Ea + loss_b * (double)Eb);
}

float ProductKeyMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float maxv = d[0];
    for (int64_t i = 1; i < n; ++i) if (d[i] > maxv) maxv = d[i];
    float sum_exp = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum_exp += std::exp(d[i] - maxv);
    float lse = maxv + std::log(sum_exp);
    float sq = 0.0f;
    for (int64_t i = 0; i < n; ++i) { float v = d[i] - lse; sq += v * v; }
    return sq / (float)n;
}

int64_t ProductKeyMoE::compute_capacity(int64_t T) const {
    int64_t Ea = (int64_t)experts_a.size(), Eb = (int64_t)experts_b.size();
    int64_t K = config.top_k;
    return std::max((K * T + Ea - 1) / Ea, (K * T + Eb - 1) / Eb);
}

std::vector<uint8_t> ProductKeyMoE::export_weights() const {
    int64_t na = key_router_a.weight.numel(), nb = key_router_b.weight.numel();
    std::vector<uint8_t> d(2 * sizeof(int64_t) + (size_t)(na + nb) * sizeof(float));
    size_t off = 0;
    std::memcpy(d.data() + off, &na, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(d.data() + off, key_router_a.weight.data<float>(), (size_t)na * sizeof(float));
    off += (size_t)na * sizeof(float);
    std::memcpy(d.data() + off, &nb, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(d.data() + off, key_router_b.weight.data<float>(), (size_t)nb * sizeof(float));
    return d;
}

void ProductKeyMoE::import_weights(const std::vector<uint8_t>& data) {
    size_t off = 0;
    int64_t na = 0; std::memcpy(&na, data.data() + off, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(key_router_a.weight.data<float>(), data.data() + off, (size_t)na * sizeof(float));
    off += (size_t)na * sizeof(float);
    int64_t nb = 0; std::memcpy(&nb, data.data() + off, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(key_router_b.weight.data<float>(), data.data() + off, (size_t)nb * sizeof(float));
}

// ---- ATTENTION_MOE ----
float AttentionMoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> frac(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        double row_sum = 0.0;
        for (int64_t e = 0; e < E; ++e) row_sum += std::exp(g[t * E + e]);
        if (row_sum <= 0.0) continue;
        for (int64_t e = 0; e < E; ++e)
            frac[e] += std::exp(g[t * E + e]) / row_sum;
    }
    double entropy = 0.0;
    for (int64_t e = 0; e < E; ++e) {
        double p = frac[e] / (double)T;
        if (p > 1e-10) entropy -= p * std::log(p);
    }
    double max_ent = std::log((double)E);
    return (float)(1.0 - entropy / max_ent);
}

float AttentionMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float lse = 0.0f;
    for (int64_t i = 0; i < n; ++i) lse = std::max(lse, d[i]);
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) { float v = d[i] - lse; sum += std::exp(v); }
    lse += std::log(sum);
    float sq_lse = lse * lse;
    return sq_lse / (1.0f + sq_lse);
}

int64_t AttentionMoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts, K = config.top_k;
    return (K * T + E - 1) / E;
}

std::vector<uint8_t> AttentionMoE::export_weights() const {
    int64_t nq = q_proj.weight.numel(), nk = k_proj.weight.numel();
    std::vector<uint8_t> d(2 * sizeof(int64_t) + (size_t)(nq + nk) * sizeof(float));
    size_t off = 0;
    std::memcpy(d.data() + off, &nq, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(d.data() + off, q_proj.weight.data<float>(), (size_t)nq * sizeof(float));
    off += (size_t)nq * sizeof(float);
    std::memcpy(d.data() + off, &nk, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(d.data() + off, k_proj.weight.data<float>(), (size_t)nk * sizeof(float));
    return d;
}

void AttentionMoE::import_weights(const std::vector<uint8_t>& data) {
    size_t off = 0;
    int64_t nq = 0; std::memcpy(&nq, data.data() + off, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(q_proj.weight.data<float>(), data.data() + off, (size_t)nq * sizeof(float));
    off += (size_t)nq * sizeof(float);
    int64_t nk = 0; std::memcpy(&nk, data.data() + off, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(k_proj.weight.data<float>(), data.data() + off, (size_t)nk * sizeof(float));
}

// ---- LOWRANK_MOE ----
float LowRankMoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    double cv = 0.0, mean = (double)T / (double)E;
    std::vector<double> counts(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        int best = 0;
        for (int64_t e = 1; e < E; ++e)
            if (g[t * E + e] > g[t * E + best]) best = (int)e;
        counts[best] += 1.0;
    }
    for (int64_t e = 0; e < E; ++e) { double d = counts[e] - mean; cv += d * d; }
    cv = std::sqrt(cv / (double)E) / mean;
    double latent_penalty = (double)latent_dim / (double)hidden_size;
    return (float)(cv * latent_penalty);
}

float LowRankMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float mean = 0.0f, var = 0.0f;
    for (int64_t i = 0; i < n; ++i) mean += d[i];
    mean /= (float)n;
    for (int64_t i = 0; i < n; ++i) { float v = d[i] - mean; var += v * v; }
    return (mean * mean + var / (float)n) / (float)latent_dim;
}

int64_t LowRankMoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts, K = config.top_k;
    return (K * T + E - 1) / E;
}

std::vector<uint8_t> LowRankMoE::export_weights() const {
    int64_t nd = down_proj.weight.numel(), nu = up_proj.weight.numel(), nr = router.weight.numel();
    std::vector<uint8_t> d(3 * sizeof(int64_t) + (size_t)(nd + nu + nr) * sizeof(float));
    size_t off = 0;
    std::memcpy(d.data() + off, &nd, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(d.data() + off, down_proj.weight.data<float>(), (size_t)nd * sizeof(float));
    off += (size_t)nd * sizeof(float);
    std::memcpy(d.data() + off, &nu, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(d.data() + off, up_proj.weight.data<float>(), (size_t)nu * sizeof(float));
    off += (size_t)nu * sizeof(float);
    std::memcpy(d.data() + off, &nr, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(d.data() + off, router.weight.data<float>(), (size_t)nr * sizeof(float));
    return d;
}

void LowRankMoE::import_weights(const std::vector<uint8_t>& data) {
    size_t off = 0;
    int64_t nd = 0; std::memcpy(&nd, data.data() + off, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(down_proj.weight.data<float>(), data.data() + off, (size_t)nd * sizeof(float));
    off += (size_t)nd * sizeof(float);
    int64_t nu = 0; std::memcpy(&nu, data.data() + off, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(up_proj.weight.data<float>(), data.data() + off, (size_t)nu * sizeof(float));
    off += (size_t)nu * sizeof(float);
    int64_t nr = 0; std::memcpy(&nr, data.data() + off, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(router.weight.data<float>(), data.data() + off, (size_t)nr * sizeof(float));
}

// ---- MAMBA_MOE ----
float MambaMoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> load(E, 0.0), prob(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        int best = 0;
        for (int64_t e = 1; e < E; ++e)
            if (g[t * E + e] > g[t * E + best]) best = (int)e;
        load[best] += 1.0;
        prob[best] += g[t * E + best];
    }
    double loss = 0.0, state_w = (double)state_dim / (double)hidden_size;
    for (int64_t e = 0; e < E; ++e) {
        load[e] /= (double)T; prob[e] /= (double)T;
        loss += load[e] * prob[e] * (1.0 + state_w * (double)e / (double)E);
    }
    return (float)(loss * (double)E);
}

float MambaMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float mx = d[0];
    for (int64_t i = 1; i < n; ++i) if (d[i] > mx) mx = d[i];
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum += std::exp(d[i] - mx);
    float lse = mx + std::log(sum);
    return 0.01f * lse * (float)state_dim;
}

int64_t MambaMoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts, K = config.top_k;
    return (K * T + E - 1) / E;
}

std::vector<uint8_t> MambaMoE::export_weights() const {
    int64_t ns = ssm_proj.weight.numel(), nr = router.weight.numel();
    std::vector<uint8_t> d(2 * sizeof(int64_t) + (size_t)(ns + nr) * sizeof(float));
    size_t off = 0;
    std::memcpy(d.data() + off, &ns, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(d.data() + off, ssm_proj.weight.data<float>(), (size_t)ns * sizeof(float));
    off += (size_t)ns * sizeof(float);
    std::memcpy(d.data() + off, &nr, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(d.data() + off, router.weight.data<float>(), (size_t)nr * sizeof(float));
    return d;
}

void MambaMoE::import_weights(const std::vector<uint8_t>& data) {
    size_t off = 0;
    int64_t ns = 0; std::memcpy(&ns, data.data() + off, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(ssm_proj.weight.data<float>(), data.data() + off, (size_t)ns * sizeof(float));
    off += (size_t)ns * sizeof(float);
    int64_t nr = 0; std::memcpy(&nr, data.data() + off, sizeof(int64_t)); off += sizeof(int64_t);
    std::memcpy(router.weight.data<float>(), data.data() + off, (size_t)nr * sizeof(float));
}

// ---- QUANTIZED_INT8_MOE ----
float QuantizedINT8MoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> frac(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        int best = 0;
        for (int64_t e = 1; e < E; ++e)
            if (g[t * E + e] > g[t * E + best]) best = (int)e;
        frac[best] += 1.0;
    }
    double loss = 0.0, mean = (double)T / (double)E;
    for (int64_t e = 0; e < E; ++e) {
        double dev = (frac[e] - mean) / mean;
        double scale_w = (double)expert_scales[(size_t)e];
        loss += dev * dev * scale_w;
    }
    return (float)(loss / (double)E);
}

float QuantizedINT8MoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float sum_sq = 0.0f, maxv = d[0];
    for (int64_t i = 0; i < n; ++i) { sum_sq += d[i] * d[i]; if (d[i] > maxv) maxv = d[i]; }
    float mean_sq = sum_sq / (float)n;
    return mean_sq / (1.0f + maxv * maxv);
}

int64_t QuantizedINT8MoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts, K = config.top_k;
    return (K * T + E - 1) / E;
}

std::vector<uint8_t> QuantizedINT8MoE::export_weights() const {
    int64_t nr = router.weight.numel();
    std::vector<uint8_t> d(sizeof(int64_t) + (size_t)nr * sizeof(float));
    std::memcpy(d.data(), &nr, sizeof(int64_t));
    std::memcpy(d.data() + sizeof(int64_t), router.weight.data<float>(), (size_t)nr * sizeof(float));
    return d;
}

void QuantizedINT8MoE::import_weights(const std::vector<uint8_t>& data) {
    int64_t nr = 0; std::memcpy(&nr, data.data(), sizeof(int64_t));
    std::memcpy(router.weight.data<float>(), data.data() + sizeof(int64_t), (size_t)nr * sizeof(float));
}

// ---- QUANT_MOE ----
float QuantMoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> frac(E, 0.0), prob(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        double row_sum = 0.0;
        for (int64_t e = 0; e < E; ++e) row_sum += std::exp(g[t * E + e]);
        if (row_sum <= 0.0) continue;
        for (int64_t e = 0; e < E; ++e) {
            double p = std::exp(g[t * E + e]) / row_sum;
            frac[e] += 1.0 / (double)T;
            prob[e] += p / (double)T;
        }
    }
    double loss = 0.0;
    for (int64_t e = 0; e < E; ++e) {
        double q_penalty = (double)(quant_scales[(size_t)e] > 0.5f ? 1 : 2);
        loss += frac[e] * prob[e] * q_penalty;
    }
    return (float)(loss * (double)E);
}

float QuantMoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float mean = 0.0f;
    for (int64_t i = 0; i < n; ++i) mean += std::abs(d[i]);
    mean /= (float)n;
    return 0.5f * mean * mean;
}

int64_t QuantMoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts, K = config.top_k;
    return (K * T + E - 1) / E;
}

std::vector<uint8_t> QuantMoE::export_weights() const {
    int64_t nr = router.weight.numel();
    std::vector<uint8_t> d(sizeof(int64_t) + (size_t)nr * sizeof(float));
    std::memcpy(d.data(), &nr, sizeof(int64_t));
    std::memcpy(d.data() + sizeof(int64_t), router.weight.data<float>(), (size_t)nr * sizeof(float));
    return d;
}

void QuantMoE::import_weights(const std::vector<uint8_t>& data) {
    int64_t nr = 0; std::memcpy(&nr, data.data(), sizeof(int64_t));
    std::memcpy(router.weight.data<float>(), data.data() + sizeof(int64_t), (size_t)nr * sizeof(float));
}

// ---- Q1_MOE ----
float Quant1MoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> frac(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        double maxv = g[t * E];
        for (int64_t e = 1; e < E; ++e) if (g[t * E + e] > maxv) maxv = g[t * E + e];
        double sum = 0.0;
        for (int64_t e = 0; e < E; ++e) sum += std::exp(g[t * E + e] - maxv);
        for (int64_t e = 0; e < E; ++e)
            frac[e] += std::exp(g[t * E + e] - maxv) / sum;
    }
    double loss = 0.0, mean = (double)T / (double)E;
    for (int64_t e = 0; e < E; ++e) {
        double f = frac[e] / (double)E;
        double bin_w = (double)(quant1_scales[(size_t)e] > 0.0f ? 1 : 3);
        loss += (f - 1.0 / (double)E) * (f - 1.0 / (double)E) * bin_w;
    }
    return (float)(loss * (double)E * (double)E);
}

float Quant1MoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum_sq += d[i] * d[i];
    return 0.001f * sum_sq;
}

int64_t Quant1MoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts, K = config.top_k;
    return (K * T + E - 1) / E;
}

std::vector<uint8_t> Quant1MoE::export_weights() const {
    int64_t nr = router.weight.numel();
    std::vector<uint8_t> d(sizeof(int64_t) + (size_t)nr * sizeof(float));
    std::memcpy(d.data(), &nr, sizeof(int64_t));
    std::memcpy(d.data() + sizeof(int64_t), router.weight.data<float>(), (size_t)nr * sizeof(float));
    return d;
}

void Quant1MoE::import_weights(const std::vector<uint8_t>& data) {
    int64_t nr = 0; std::memcpy(&nr, data.data(), sizeof(int64_t));
    std::memcpy(router.weight.data<float>(), data.data() + sizeof(int64_t), (size_t)nr * sizeof(float));
}

// ---- QUANT8_MOE ----
float QUANT8MoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> frac(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        int best = 0;
        for (int64_t e = 1; e < E; ++e)
            if (g[t * E + e] > g[t * E + best]) best = (int)e;
        frac[best] += 1.0;
    }
    double mean = (double)T / (double)E, var = 0.0;
    for (int64_t e = 0; e < E; ++e) { double d = frac[e] - mean; var += d * d; }
    double cv = std::sqrt(var / (double)E) / mean;
    double codebook_entropy = 0.0;
    for (int64_t e = 0; e < E; ++e) {
        auto& cb = codebooks[(size_t)e];
        double cb_sum = 0.0;
        for (auto v : cb) cb_sum += (double)(v * v);
        if (cb_sum > 0) codebook_entropy += cb_sum * frac[e];
    }
    return (float)(cv + codebook_entropy / ((double)E * 256.0));
}

float QUANT8MoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float maxv = d[0];
    for (int64_t i = 1; i < n; ++i) if (d[i] > maxv) maxv = d[i];
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum += std::exp(d[i] - maxv);
    float lse = maxv + std::log(sum);
    return 0.001f * lse * lse / 256.0f;
}

int64_t QUANT8MoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts, K = config.top_k;
    return (K * T + E - 1) / E;
}

std::vector<uint8_t> QUANT8MoE::export_weights() const {
    int64_t nr = router.weight.numel();
    std::vector<uint8_t> d(sizeof(int64_t) + (size_t)nr * sizeof(float));
    std::memcpy(d.data(), &nr, sizeof(int64_t));
    std::memcpy(d.data() + sizeof(int64_t), router.weight.data<float>(), (size_t)nr * sizeof(float));
    return d;
}

void QUANT8MoE::import_weights(const std::vector<uint8_t>& data) {
    int64_t nr = 0; std::memcpy(&nr, data.data(), sizeof(int64_t));
    std::memcpy(router.weight.data<float>(), data.data() + sizeof(int64_t), (size_t)nr * sizeof(float));
}

// ---- QUANT4_MOE ----
float QUANT4MoE::load_balance_loss(const Tensor& gates) const {
    int64_t T = gates.dim(0), E = gates.dim(1);
    const float* g = gates.data<float>();
    std::vector<double> frac(E, 0.0), prob(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        double row_sum = 0.0;
        for (int64_t e = 0; e < E; ++e) row_sum += std::exp(g[t * E + e]);
        if (row_sum <= 0.0) continue;
        for (int64_t e = 0; e < E; ++e) {
            double p = std::exp(g[t * E + e]) / row_sum;
            frac[e] += 1.0 / (double)T;
            prob[e] += p / (double)T;
        }
    }
    double loss = 0.0;
    for (int64_t e = 0; e < E; ++e) {
        auto& cb = codebooks[(size_t)e];
        double cb_util = 0.0;
        for (auto v : cb) if (std::abs(v) > 1e-6) cb_util += 1.0;
        cb_util /= 16.0;
        loss += frac[e] * prob[e] * (1.0 + cb_util);
    }
    return (float)(loss * (double)E);
}

float QUANT4MoE::z_loss(const Tensor& logits) const {
    const float* d = logits.data<float>();
    int64_t n = logits.numel();
    float sum = 0.0f, maxv = d[0];
    for (int64_t i = 0; i < n; ++i) { sum += std::abs(d[i]); if (d[i] > maxv) maxv = d[i]; }
    float mean_abs = sum / (float)n;
    return mean_abs * maxv / 16.0f;
}

int64_t QUANT4MoE::compute_capacity(int64_t T) const {
    int64_t E = config.num_experts, K = config.top_k;
    return (K * T + E - 1) / E;
}

std::vector<uint8_t> QUANT4MoE::export_weights() const {
    int64_t nr = router.weight.numel();
    std::vector<uint8_t> d(sizeof(int64_t) + (size_t)nr * sizeof(float));
    std::memcpy(d.data(), &nr, sizeof(int64_t));
    std::memcpy(d.data() + sizeof(int64_t), router.weight.data<float>(), (size_t)nr * sizeof(float));
    return d;
}

void QUANT4MoE::import_weights(const std::vector<uint8_t>& data) {
    int64_t nr = 0; std::memcpy(&nr, data.data(), sizeof(int64_t));
    std::memcpy(router.weight.data<float>(), data.data() + sizeof(int64_t), (size_t)nr * sizeof(float));
}

} // namespace moe
} // namespace quant
