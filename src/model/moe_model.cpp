#include "quant/moe_model.h"
#include "quant/math.h"
#include "quant/autograd.h"
#include "quant/quant_format.h"
#include "quant/expert_prefetch.h"
#include <cstring>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace quant {

// ==================================================================
// MoEBlock
// ==================================================================

MoEBlock::MoEBlock(const TransformerConfig& cfg, const moe::MoEAllConfig& moe_cfg)
    : moe_config(moe_cfg)
{
    attention_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps);
    attention = Attention(cfg);
    ffn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps);

    moe = std::make_unique<moe::SparseMoE>(cfg.hidden_size, moe_cfg);

    if (moe_cfg.use_shared_expert && moe_cfg.num_shared_experts > 0) {
        shared_expert = std::make_unique<moe::ExpertFFN>(
            cfg.hidden_size, moe_cfg.expert_hidden_size, cfg.activation);
    }
}

Tensor MoEBlock::forward(const Tensor& x, const Tensor& positions,
                          const Tensor& mask, KVCache& cache, int layer_idx,
                          bool training)
{
    Tensor attn_input = attention_norm.forward(x);
    Tensor attn_out = attention.forward(attn_input, positions, mask, cache, layer_idx);
    attn_out = AutogradEngine::add_op(attn_out, x);

    Tensor ffn_input = ffn_norm.forward(attn_out);
    moe::MoEOutput moe_out = moe->forward(ffn_input, training);

    load_balance_loss = moe_out.load_balance_loss;
    z_loss = moe_out.z_loss;
    tokens_dropped = moe_out.tokens_dropped;
    last_expert_indices = moe_out.expert_indices;
    last_router_logits = moe_out.router_logits;

    Tensor ffn_output = moe_out.output;
    if (shared_expert) {
        Tensor shared_out = shared_expert->forward(ffn_input);
        ffn_output = AutogradEngine::add_op(ffn_output, shared_out);
    }

    return AutogradEngine::add_op(ffn_output, attn_out);
}

int64_t MoEBlock::param_count() const {
    int64_t count = 0;
    count += attention.q_proj.param_count();
    count += attention.k_proj.param_count();
    count += attention.v_proj.param_count();
    count += attention.o_proj.param_count();
    count += attention_norm.weight.numel();
    count += ffn_norm.weight.numel();
    count += moe->router_weight.param_count();

    // ALL stored expert parameters (all experts in the pool)
    if (!moe->experts.empty()) {
        int64_t exp = moe->experts[0].gate_proj.param_count() +
                      moe->experts[0].up_proj.param_count() +
                      moe->experts[0].down_proj.param_count();
        count += (int64_t)moe->experts.size() * exp;
    }

    if (shared_expert) {
        count += shared_expert->gate_proj.param_count();
        count += shared_expert->up_proj.param_count();
        count += shared_expert->down_proj.param_count();
    }
    return count;
}

int64_t MoEBlock::activated_param_count() const {
    int64_t count = 0;
    count += attention.q_proj.param_count();
    count += attention.k_proj.param_count();
    count += attention.v_proj.param_count();
    count += attention.o_proj.param_count();
    count += attention_norm.weight.numel();
    count += ffn_norm.weight.numel();
    count += moe->router_weight.param_count();

    // Only top-k expert parameters active per token
    if (!moe->experts.empty()) {
        int64_t exp = moe->experts[0].gate_proj.param_count() +
                      moe->experts[0].up_proj.param_count() +
                      moe->experts[0].down_proj.param_count();
        count += moe_config.top_k * exp;
    }

    if (shared_expert) {
        count += shared_expert->gate_proj.param_count();
        count += shared_expert->up_proj.param_count();
        count += shared_expert->down_proj.param_count();
    }
    return count;
}

int64_t MoEBlock::num_stored_experts() const {
    return (int64_t)moe->experts.size();
}

// ==================================================================
// MoEModel
// ==================================================================

MoEModel::MoEModel(const TransformerConfig& cfg, const moe::MoEAllConfig& moe_cfg)
    : moe_config(moe_cfg)
{
    config = cfg;
    build_layers();
}

void MoEModel::build_layers() {
    tok_embeddings = std::make_unique<Embedding>(config.vocab_size, config.hidden_size);
    layers.clear();
    for (int64_t i = 0; i < config.num_layers; i++) {
        layers.emplace_back(config, moe_config);
    }
    norm = std::make_unique<RMSNorm>(config.hidden_size, config.norm_eps);
    lm_head = std::make_unique<Linear>(config.hidden_size, config.vocab_size);
}

int64_t MoEModel::vocab_size() const { return config.vocab_size; }

int64_t MoEModel::param_count() const {
    int64_t count = tok_embeddings->param_count();
    count += norm->weight.numel();
    count += lm_head->param_count();

    for (auto& l : layers) {
        count += l.activated_param_count();
    }
    return count;
}

int64_t MoEModel::stored_param_count() const {
    int64_t count = tok_embeddings->param_count();
    count += norm->weight.numel();
    count += lm_head->param_count();

    int64_t expert_ffn_params = 0;
    if (!layers.empty() && !layers[0].moe->experts.empty()) {
        auto& e = layers[0].moe->experts[0];
        expert_ffn_params = e.gate_proj.param_count() +
                            e.up_proj.param_count() +
                            e.down_proj.param_count();
    }

    for (auto& l : layers) {
        count += l.attention.q_proj.param_count();
        count += l.attention.k_proj.param_count();
        count += l.attention.v_proj.param_count();
        count += l.attention.o_proj.param_count();
        count += l.attention_norm.weight.numel();
        count += l.ffn_norm.weight.numel();
        count += l.moe->router_weight.param_count();
        count += (int64_t)l.moe->experts.size() * expert_ffn_params;
        if (l.shared_expert) {
            count += l.shared_expert->gate_proj.param_count();
            count += l.shared_expert->up_proj.param_count();
            count += l.shared_expert->down_proj.param_count();
        }
    }
    return count;
}

Tensor MoEModel::forward(const Tensor& input_ids, const Tensor& positions,
                          KVCache* cache)
{
    int64_t B = input_ids.dim(0);
    int64_t S = input_ids.dim(1);

    Tensor h = tok_embeddings->forward(input_ids.reshape(Shape{B * S}));
    h = h.reshape(Shape{B, S, config.hidden_size});

    KVCache local_cache;
    KVCache* active_cache = cache;
    if (!active_cache) {
        local_cache.init((int)config.num_layers, config.max_seq_len,
                         config.num_heads, config.head_dim);
        active_cache = &local_cache;
    }

    Tensor causal_mask(Shape{1, 1, S, S});
    float* md = causal_mask.data<float>();
    for (int64_t s = 0; s < S; s++) {
        for (int64_t t = 0; t < S; t++) {
            md[s * S + t] = (t > s) ? -INFINITY : 0.0f;
        }
    }

    total_load_balance_loss = 0.0f;
    total_z_loss = 0.0f;

    for (int64_t i = 0; i < config.num_layers; i++) {
        h = layers[i].forward(h, positions, causal_mask, *active_cache, (int)i);
        total_load_balance_loss += layers[i].load_balance_loss;
        total_z_loss += layers[i].z_loss;
        
        if (i + 1 < config.num_layers && prefetcher) {
            if (moe::should_prefetch_all_experts((int)B)) {
                std::vector<int> exp_ids(layers[i+1].num_stored_experts());
                std::iota(exp_ids.begin(), exp_ids.end(), 0);
                prefetcher->schedule_prefetch(i + 1, exp_ids);
            } else {
                Tensor indices = layers[i].last_expert_indices;
                if (indices.numel() > 0) {
                    std::vector<int> exp_ids;
                    const int64_t* ptr = indices.data<int64_t>();
                    for (int64_t k = 0; k < indices.numel(); ++k) {
                        exp_ids.push_back((int)ptr[k]);
                    }
                    std::sort(exp_ids.begin(), exp_ids.end());
                    exp_ids.erase(std::unique(exp_ids.begin(), exp_ids.end()), exp_ids.end());
                    prefetcher->schedule_prefetch(i + 1, exp_ids);
                }
            }
        }
    }

    h = norm->forward(h);
    return lm_head->forward(h);
}

// ==================================================================
// 48T scaling config presets
// ==================================================================

TransformerConfig MoEModel::config_48T() {
    TransformerConfig cfg;
    cfg.vocab_size = 320000;
    cfg.hidden_size = 24576;
    cfg.num_layers = 128;
    cfg.num_heads = 128;
    cfg.head_dim = 192;
    cfg.ffn_hidden_size = 98304;
    cfg.norm_eps = 1e-5f;
    cfg.rope_theta = 10000.0f;
    cfg.max_seq_len = 16384;
    cfg.activation = Activation::SiLU;
    cfg.num_kv_heads = 8;
    return cfg;
}

TransformerConfig MoEModel::config_1T() {
    TransformerConfig cfg;
    cfg.vocab_size = 128000;
    cfg.hidden_size = 8192;
    cfg.num_layers = 64;
    cfg.num_heads = 64;
    cfg.head_dim = 128;
    cfg.ffn_hidden_size = 16384;
    cfg.norm_eps = 1e-5f;
    cfg.rope_theta = 10000.0f;
    cfg.max_seq_len = 8192;
    cfg.activation = Activation::SiLU;
    cfg.num_kv_heads = 8;
    return cfg;
}

TransformerConfig MoEModel::config_100B() {
    TransformerConfig cfg;
    cfg.vocab_size = 32000;
    cfg.hidden_size = 4096;
    cfg.num_layers = 32;
    cfg.num_heads = 32;
    cfg.head_dim = 128;
    cfg.ffn_hidden_size = 8192;
    cfg.norm_eps = 1e-5f;
    cfg.rope_theta = 10000.0f;
    cfg.max_seq_len = 4096;
    cfg.activation = Activation::SiLU;
    cfg.num_kv_heads = 4;
    return cfg;
}

moe::MoEAllConfig MoEModel::moe_config_48T() {
    moe::MoEAllConfig mc;
    mc.variant = moe::MoEVariant::DEEPSEEK_MOE;
    mc.num_experts = 8192;
    mc.top_k = 8;
    mc.expert_hidden_size = 2048;
    mc.load_balance_coef = 0.01f;
    mc.z_loss_coef = 0.001f;
    mc.use_shared_expert = true;
    mc.num_shared_experts = 1;
    mc.num_routed_experts = 8192;
    mc.capacity_strategy = moe::CapacityStrategy::TOKEN_DROP;
    mc.capacity_factor = 1.25f;
    return mc;
}

moe::MoEAllConfig MoEModel::moe_config_1T() {
    moe::MoEAllConfig mc;
    mc.variant = moe::MoEVariant::SPARSE_TOPK;
    mc.num_experts = 256;
    mc.top_k = 6;
    mc.expert_hidden_size = 4096;
    mc.load_balance_coef = 0.01f;
    mc.z_loss_coef = 0.001f;
    mc.use_shared_expert = false;
    mc.capacity_strategy = moe::CapacityStrategy::TOKEN_DROP;
    mc.capacity_factor = 1.25f;
    return mc;
}

moe::MoEAllConfig MoEModel::moe_config_100B() {
    moe::MoEAllConfig mc;
    mc.variant = moe::MoEVariant::SPARSE_TOPK;
    mc.num_experts = 64;
    mc.top_k = 4;
    mc.expert_hidden_size = 4096;
    mc.load_balance_coef = 0.01f;
    mc.z_loss_coef = 0.001f;
    mc.use_shared_expert = false;
    mc.capacity_strategy = moe::CapacityStrategy::TOKEN_DROP;
    mc.capacity_factor = 1.5f;
    return mc;
}

// ==================================================================
// Save — QUANT format with MoE expert tensors
// ==================================================================

static std::vector<std::pair<std::string, Tensor>> collect_moe_named_tensors(const MoEModel& m) {
    std::vector<std::pair<std::string, Tensor>> tensors;
    tensors.emplace_back("tok_embeddings.weight", m.tok_embeddings->weight);

    for (size_t i = 0; i < m.layers.size(); i++) {
        std::string p = "layers." + std::to_string(i) + ".";
        const auto& l = m.layers[i];
        tensors.emplace_back(p + "attention_norm.weight", l.attention_norm.weight);
        tensors.emplace_back(p + "ffn_norm.weight", l.ffn_norm.weight);
        tensors.emplace_back(p + "attention.q_proj.weight", l.attention.q_proj.weight);
        if (l.attention.q_proj.bias.numel() > 0)
            tensors.emplace_back(p + "attention.q_proj.bias", l.attention.q_proj.bias);
        tensors.emplace_back(p + "attention.k_proj.weight", l.attention.k_proj.weight);
        if (l.attention.k_proj.bias.numel() > 0)
            tensors.emplace_back(p + "attention.k_proj.bias", l.attention.k_proj.bias);
        tensors.emplace_back(p + "attention.v_proj.weight", l.attention.v_proj.weight);
        if (l.attention.v_proj.bias.numel() > 0)
            tensors.emplace_back(p + "attention.v_proj.bias", l.attention.v_proj.bias);
        tensors.emplace_back(p + "attention.o_proj.weight", l.attention.o_proj.weight);
        if (l.attention.o_proj.bias.numel() > 0)
            tensors.emplace_back(p + "attention.o_proj.bias", l.attention.o_proj.bias);

        tensors.emplace_back(p + "moe.router.weight", l.moe->router_weight.weight);

        for (size_t e = 0; e < l.moe->experts.size(); e++) {
            std::string ep = p + "moe.expert." + std::to_string(e) + ".";
            tensors.emplace_back(ep + "gate_proj.weight", l.moe->experts[e].gate_proj.weight);
            tensors.emplace_back(ep + "up_proj.weight", l.moe->experts[e].up_proj.weight);
            tensors.emplace_back(ep + "down_proj.weight", l.moe->experts[e].down_proj.weight);
        }

        if (l.shared_expert) {
            tensors.emplace_back(p + "shared_expert.gate_proj.weight", l.shared_expert->gate_proj.weight);
            tensors.emplace_back(p + "shared_expert.up_proj.weight", l.shared_expert->up_proj.weight);
            tensors.emplace_back(p + "shared_expert.down_proj.weight", l.shared_expert->down_proj.weight);
        }
    }

    tensors.emplace_back("norm.weight", m.norm->weight);
    tensors.emplace_back("lm_head.weight", m.lm_head->weight);
    if (m.lm_head->bias.numel() > 0)
        tensors.emplace_back("lm_head.bias", m.lm_head->bias);

    return tensors;
}

void MoEModel::save(const std::string& quant_path) const {
    auto named = collect_moe_named_tensors(*this);

    std::vector<FormatBlockEntry> ft;
    std::vector<TensorEntry> te;
    std::vector<std::string> names;
    std::vector<BlockData> blocks;
    uint32_t block_id = 0;
    uint32_t block_start = 0;

    for (auto& [name, t] : named) {
        names.push_back(name);
        int64_t numel = t.numel();
        if (numel == 0) continue;

        const float* td = t.data<float>();
        uint32_t block_count = 0;
        uint32_t bs = block_start;

        for (int64_t offset = 0; offset < numel; ) {
            uint32_t blk_size = (uint32_t)std::min((int64_t)32768, numel - offset);
            BlockData bd;
            bd.format = Format::Q32;
            bd.num_weights = blk_size;
            bd.indices.resize(blk_size * 4);
            std::memcpy(bd.indices.data(), td + offset, blk_size * 4);
            blocks.push_back(bd);

            FormatBlockEntry fbe;
            fbe.block_id = block_id++;
            fbe.format = 5;
            fbe.cb_bytes = 0;
            ft.push_back(fbe);
            block_count++;
            offset += blk_size;
        }

        TensorEntry entry;
        entry.block_start = bs;
        entry.num_blocks = block_count;
        te.push_back(entry);
        block_start = block_id;
    }

    QUANTWriter writer(quant_path);
    QUANTHeader hdr;
    std::memcpy(hdr.magic, "QUA1", 4);
    hdr.version = 1;
    hdr.flags = 0;
    hdr.config_size = sizeof(TransformerConfig);
    writer.write_header(hdr, (const uint8_t*)&config);
    writer.write_format_table(ft);
    writer.write_tensor_table(te, names);
    for (auto& bd : blocks) writer.write_block(bd);
    writer.close();
}

void MoEModel::load(const std::string& quant_path) {
    if (quant_path.size() < 6 ||
        quant_path.substr(quant_path.size() - 6) != ".quant") {
        throw std::runtime_error(
            "QUANT format required: only .quant files are supported. "
            "External format adapters removed. QUANT is ~40-45% more compute-efficient "
            "than standard formats — contact owner for commercial deployment");
    }
    QUANTReader reader(quant_path);
    if (!reader.valid()) throw std::runtime_error("Cannot open: " + quant_path);
    if (reader.header().config_size >= sizeof(TransformerConfig)) {
        auto cfg_data = reader.read_config();
        if (cfg_data.size() >= sizeof(TransformerConfig)) {
            std::memcpy(&config, cfg_data.data(), sizeof(TransformerConfig));
        }
    }
    build_layers();

    auto tensor_names = reader.tensor_names();
    std::unordered_map<std::string, Tensor> loaded_weights;
    for (const auto& n : tensor_names) {
        loaded_weights[n] = reader.read_tensor(n);
    }

    auto assign = [&](const std::string& name, Tensor& dst) {
        auto it = loaded_weights.find(name);
        if (it != loaded_weights.end() && it->second.numel() == dst.numel()) {
            dst.copy_from(it->second);
        }
    };

    assign("tok_embeddings.weight", tok_embeddings->weight);

    for (size_t i = 0; i < layers.size(); i++) {
        std::string p = "layers." + std::to_string(i) + ".";
        auto& l = layers[i];
        assign(p + "attention_norm.weight", l.attention_norm.weight);
        assign(p + "ffn_norm.weight", l.ffn_norm.weight);
        assign(p + "attention.q_proj.weight", l.attention.q_proj.weight);
        assign(p + "attention.k_proj.weight", l.attention.k_proj.weight);
        assign(p + "attention.v_proj.weight", l.attention.v_proj.weight);
        assign(p + "attention.o_proj.weight", l.attention.o_proj.weight);

        assign(p + "moe.router.weight", l.moe->router_weight.weight);

        for (size_t e = 0; e < l.moe->experts.size(); e++) {
            std::string ep = p + "moe.expert." + std::to_string(e) + ".";
            assign(ep + "gate_proj.weight", l.moe->experts[e].gate_proj.weight);
            assign(ep + "up_proj.weight", l.moe->experts[e].up_proj.weight);
            assign(ep + "down_proj.weight", l.moe->experts[e].down_proj.weight);
        }

        if (l.shared_expert) {
            assign(p + "shared_expert.gate_proj.weight", l.shared_expert->gate_proj.weight);
            assign(p + "shared_expert.up_proj.weight", l.shared_expert->up_proj.weight);
            assign(p + "shared_expert.down_proj.weight", l.shared_expert->down_proj.weight);
        }
    }

    assign("norm.weight", norm->weight);
    assign("lm_head.weight", lm_head->weight);
}

// ==================================================================
// ExpertParallel
// ==================================================================

ExpertParallel::ExpertParallel(int num_experts, int num_ranks, int rank,
                                DistributedContext* ctx)
    : num_experts_(num_experts), num_ranks_(num_ranks), rank_(rank)
{
    if (ctx) {
        ctx_ = ctx;
        owns_ctx_ = false;
    } else {
        ctx_ = new DistributedContext(num_ranks, rank, DistributedContext::Mode::DDP);
        owns_ctx_ = true;
    }
    build_expert_distribution();
}

ExpertParallel::~ExpertParallel() {
    if (owns_ctx_) delete ctx_;
}

void ExpertParallel::build_expert_distribution() {
    local_experts_.clear();
    for (int e = rank_; e < num_experts_; e += num_ranks_) {
        local_experts_.push_back(e);
    }
}

std::vector<int> ExpertParallel::local_experts() const {
    return local_experts_;
}

void ExpertParallel::alltoall_experts(const Tensor& input, Tensor& output) const {
    int64_t T = input.dim(0);
    int64_t D = input.dim(1);

    int64_t chunk_size = T * D;
    std::vector<float> flat(chunk_size * num_ranks_, 0.0f);
    std::memcpy(flat.data() + rank_ * chunk_size, input.data<float>(), chunk_size * sizeof(float));

    ctx_->all_gather(input, output);

    (void)output;
}

Tensor ExpertParallel::forward_with_parallel(MoEModel* model, const Tensor& input,
                                              const Tensor& positions, KVCache* cache) const
{
    int64_t B = input.dim(0);
    int64_t S = input.dim(1);
    int64_t D = model->config.hidden_size;

    Tensor h = model->tok_embeddings->forward(input.reshape(Shape{B * S}));
    h = h.reshape(Shape{B, S, D});

    KVCache local_cache;
    KVCache* active_cache = cache;
    if (!active_cache) {
        local_cache.init((int)model->config.num_layers, model->config.max_seq_len,
                         model->config.num_heads, model->config.head_dim);
        active_cache = &local_cache;
    }

    Tensor causal_mask(Shape{1, 1, S, S});
    float* md = causal_mask.data<float>();
    for (int64_t s = 0; s < S; s++)
        for (int64_t t = 0; t < S; t++)
            md[s * S + t] = (t > s) ? -INFINITY : 0.0f;

    for (int64_t i = 0; i < model->config.num_layers; i++) {
        auto& block = model->layers[i];
        Tensor attn_input = block.attention_norm.forward(h);
        Tensor attn_out = block.attention.forward(attn_input, positions, causal_mask, *active_cache, (int)i);
        h = AutogradEngine::add_op(attn_out, h);

        Tensor ffn_input = block.ffn_norm.forward(h);

        int64_t T = B * S;
        Tensor router_logits = block.moe->router_weight.forward(ffn_input.reshape(Shape{T, D}));
        Tensor expert_weights, expert_indices;
        Tensor probs = moe::softmax_with_topk(router_logits, block.moe_config.top_k,
                                               expert_indices, expert_weights);

        int64_t K = block.moe_config.top_k;
        int64_t E = block.moe_config.num_experts;
        Tensor flat_input = ffn_input.reshape(Shape{T, D});

        std::vector<std::vector<float>> expert_buffers(E);
        std::vector<std::vector<int64_t>> expert_token_ids(E);
        std::vector<std::vector<float>> expert_token_weights(E);

        const float* x_data = flat_input.data<float>();
        const int64_t* idx_data = expert_indices.data<int64_t>();
        const float* w_data = expert_weights.data<float>();

        for (int64_t t = 0; t < T; t++) {
            for (int64_t k = 0; k < K; k++) {
                int64_t e = idx_data[t * K + k];
                if (e >= 0 && e < E) {
                    size_t pos = expert_buffers[e].size();
                    expert_buffers[e].resize(pos + D);
                    std::memcpy(&expert_buffers[e][pos], x_data + t * D, D * sizeof(float));
                    expert_token_ids[e].push_back(t);
                    expert_token_weights[e].push_back(w_data[t * K + k]);
                }
            }
        }

        Tensor output_flat = Tensor::zeros(Shape{T, D});
        float* out_data = output_flat.data<float>();

        for (int64_t e = 0; e < E; e++) {
            if (expert_buffers[e].empty()) continue;
            bool is_local = false;
            for (int le : local_experts_) {
                if (le == (int)e) { is_local = true; break; }
            }
            if (!is_local) continue;

            int64_t nt = (int64_t)expert_token_ids[e].size();
            Tensor ex_input(Shape{nt, D}, DType::F32);
            std::memcpy(ex_input.data<float>(), expert_buffers[e].data(), nt * D * sizeof(float));

            Tensor ex_out;
            auto& expert = block.moe->experts[e];
            Tensor gate = expert.gate_proj.forward(ex_input);
            Tensor up = expert.up_proj.forward(ex_input);
            Tensor act_out({gate.shape()});
            if (expert.activation == Activation::SiLU)
                math::silu(gate, act_out);
            else if (expert.activation == Activation::GELU)
                math::gelu(gate, act_out);
            else
                math::relu(gate, act_out);
            Tensor gated({gate.shape()});
            math::mul(act_out, up, gated);
            ex_out = expert.down_proj.forward(gated);

            const float* eo_data = ex_out.data<float>();
            for (int64_t t = 0; t < nt; t++) {
                int64_t token_id = expert_token_ids[e][t];
                float weight = expert_token_weights[e][t];
                for (int64_t d = 0; d < D; d++) {
                    out_data[token_id * D + d] += weight * eo_data[t * D + d];
                }
            }
        }

        ctx_->all_reduce(output_flat);

        if (block.shared_expert) {
            Tensor shared_out = block.shared_expert->forward(ffn_input);
            math::add(output_flat, shared_out.reshape(Shape{T, D}), output_flat);
        }

        h = AutogradEngine::add_op(output_flat.reshape(Shape{B, S, D}), h);
    }

    h = model->norm->forward(h);
    return model->lm_head->forward(h);
}

void ExpertParallel::sync_gradients(MoEModel* model) const {
    if (num_ranks_ <= 1) return;

    auto sync_param = [&](Tensor& t) {
        if (t.has_grad()) {
            ctx_->all_reduce(t.grad());
        }
    };

    sync_param(model->tok_embeddings->weight);
    for (auto& l : model->layers) {
        sync_param(l.attention_norm.weight);
        sync_param(l.ffn_norm.weight);
        sync_param(l.attention.q_proj.weight);
        sync_param(l.attention.k_proj.weight);
        sync_param(l.attention.v_proj.weight);
        sync_param(l.attention.o_proj.weight);
        sync_param(l.moe->router_weight.weight);

        for (auto& e : l.moe->experts) {
            sync_param(e.gate_proj.weight);
            sync_param(e.up_proj.weight);
            sync_param(e.down_proj.weight);
        }

        if (l.shared_expert) {
            sync_param(l.shared_expert->gate_proj.weight);
            sync_param(l.shared_expert->up_proj.weight);
            sync_param(l.shared_expert->down_proj.weight);
        }
    }
    sync_param(model->norm->weight);
    sync_param(model->lm_head->weight);
}

void MoEModel::init_prefetcher(int prefetch_ahead, int max_resident) {
    int num_layers = (int)layers.size();
    int num_experts = 0;
    size_t expert_bytes = 0;
    for (auto& l : layers) {
        if (l.moe && !l.moe->experts.empty()) {
            num_experts = std::max(num_experts, (int)l.moe->experts.size());
            auto& e = l.moe->experts[0];
            expert_bytes = (size_t)(e.gate_proj.weight.numel() + e.up_proj.weight.numel() +
                e.down_proj.weight.numel()) * sizeof(float);
        }
    }
    if (num_layers <= 0 || num_experts <= 0 || expert_bytes == 0) return;
    prefetcher = std::make_unique<ExpertPrefetcher>(num_experts, num_layers,
        expert_bytes, prefetch_ahead, max_resident);
    prefetcher->initialize();
    wire_prefetcher_sources();
}

void MoEModel::wire_prefetcher_sources() {
    if (!prefetcher) return;
    for (size_t li = 0; li < layers.size(); ++li) {
        auto& l = layers[li];
        if (!l.moe) continue;
        for (size_t ei = 0; ei < l.moe->experts.size(); ++ei) {
            auto& e = l.moe->experts[ei];
            std::vector<SourceSegment> segs;
            segs.push_back({e.gate_proj.weight.data(), (size_t)e.gate_proj.weight.numel() * sizeof(float)});
            segs.push_back({e.up_proj.weight.data(), (size_t)e.up_proj.weight.numel() * sizeof(float)});
            segs.push_back({e.down_proj.weight.data(), (size_t)e.down_proj.weight.numel() * sizeof(float)});
            prefetcher->set_expert_segments((int)li, (int)ei, segs);
        }
    }
}

} // namespace quant
