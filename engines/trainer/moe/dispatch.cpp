#include "dispatch.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace quant {
namespace moe_engine {

DispatchPlan plan_dispatch(const Tensor& indices, const Tensor& weights,
                           int64_t num_experts, int64_t slots_per_expert) {
    if (indices.rank() != 2 || weights.rank() != 2)
        throw std::runtime_error("[dispatch] indices/weights need 2-D");
    if (indices.dim(0) != weights.dim(0) || indices.dim(1) != weights.dim(1))
        throw std::runtime_error("[dispatch] indices/weights shape mismatch");
    if (num_experts <= 0 || slots_per_expert <= 0)
        throw std::runtime_error("[dispatch] bad experts/slots");
    int64_t tokens = indices.dim(0), top_k = indices.dim(1);
    const int* ix = indices.data<int>();
    const float* w = weights.data<float>();
    DispatchPlan p;
    p.slots_per_expert = slots_per_expert;
    p.per_expert.assign((size_t)num_experts, {});
    for (int64_t t = 0; t < tokens; t++) {
        for (int64_t k = 0; k < top_k; k++) {
            int e = ix[t * top_k + k];
            if (e < 0 || e >= num_experts) throw std::runtime_error("[dispatch] expert OOB");
            DispatchItem it;
            it.token = t;
            it.expert = e;
            it.weight = w[t * top_k + k];
            if ((int64_t)p.per_expert[(size_t)e].size() < slots_per_expert)
                p.per_expert[(size_t)e].push_back(it);
            else
                p.dropped.push_back(it);
        }
    }
    return p;
}

ExpertInputs gather_expert_inputs(const Tensor& hidden, const DispatchPlan& plan,
                                  int64_t num_experts) {
    if (hidden.rank() != 2) throw std::runtime_error("[dispatch] hidden needs 2-D");
    int64_t dim = hidden.dim(1);
    const float* h = hidden.data<float>();
    ExpertInputs o;
    o.bufs.reserve((size_t)num_experts);
    o.counts.assign((size_t)num_experts, 0);
    for (int64_t e = 0; e < num_experts; e++) {
        const auto& items = plan.per_expert[(size_t)e];
        Tensor b = Tensor::zeros({(int64_t)items.size() > 0 ? (int64_t)items.size() : 1, dim});
        float* d = b.data<float>();
        for (size_t i = 0; i < items.size(); i++)
            std::memcpy(d + i * (size_t)dim, h + items[i].token * dim, (size_t)dim * sizeof(float));
        o.counts[(size_t)e] = (int64_t)items.size();
        o.bufs.push_back(b);
    }
    return o;
}

CombineOut combine_expert_outputs(const std::vector<Tensor>& expert_outs,
                                  const DispatchPlan& plan,
                                  int64_t tokens, int64_t dim) {
    if ((int64_t)expert_outs.size() != (int64_t)plan.per_expert.size())
        throw std::runtime_error("[dispatch] expert_outs/plan mismatch");
    CombineOut o;
    o.hidden = Tensor::zeros({tokens, dim});
    float* h = o.hidden.data<float>();
    for (size_t e = 0; e < plan.per_expert.size(); e++) {
        const float* y = expert_outs[e].data<float>();
        for (size_t i = 0; i < plan.per_expert[e].size(); i++) {
            int64_t t = plan.per_expert[e][i].token;
            float w = plan.per_expert[e][i].weight;
            for (int64_t d = 0; d < dim; d++) h[t * dim + d] += w * y[i * (size_t)dim + d];
        }
    }
    o.dropped = (int64_t)plan.dropped.size();
    return o;
}

DropStats drop_stats(const DispatchPlan& plan, int64_t num_experts) {
    DropStats s;
    s.per_expert_kept.assign((size_t)num_experts, 0);
    for (int64_t e = 0; e < num_experts; e++)
        s.per_expert_kept[(size_t)e] = (int64_t)plan.per_expert[(size_t)e].size();
    for (auto v : s.per_expert_kept) s.kept += v;
    s.dropped = (int64_t)plan.dropped.size();
    s.assigned = s.kept + s.dropped;
    s.drop_rate = s.assigned > 0 ? (double)s.dropped / (double)s.assigned : 0.0;
    return s;
}

} // namespace moe_engine
} // namespace quant
