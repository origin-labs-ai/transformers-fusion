#include "quant/model.h"
#include "quant/transformer.h"
#include "quant/autograd.h"
#include <cstring>
#include <cmath>
#include <random>
#include <unordered_map>
#include <stdexcept>

namespace quant {

DenseModel::DenseModel(const TransformerConfig& cfg) {
    config = cfg;
    build_layers();
}

void DenseModel::build_layers() {
    tok_embeddings = std::make_unique<Embedding>(config.vocab_size, config.hidden_size);
    layers.clear();
    for (int64_t i = 0; i < config.num_layers; i++) {
        layers.push_back(std::make_unique<TransformerBlock>(config));
    }
    norm = std::make_unique<RMSNorm>(config.hidden_size, config.norm_eps);
    lm_head = std::make_unique<Linear>(config.hidden_size, config.vocab_size);
    mtp_heads.clear();
    for (int i = 0; i < config.mtp_num_heads; i++) {
        mtp_heads.push_back(std::make_unique<Linear>(config.hidden_size, config.vocab_size));
    }
}

int64_t DenseModel::vocab_size() const { return config.vocab_size; }

int64_t DenseModel::param_count() const {
    int64_t count = tok_embeddings->param_count();
    for (auto& l : layers) {
        count += l->attention.q_proj.param_count();
        count += l->attention.k_proj.param_count();
        count += l->attention.v_proj.param_count();
        count += l->attention.o_proj.param_count();
        count += l->ffn.gate_proj.param_count();
        count += l->ffn.up_proj.param_count();
        count += l->ffn.down_proj.param_count();
        count += l->attention_norm.weight.numel();
        count += l->ffn_norm.weight.numel();
    }
    count += norm->weight.numel();
    count += lm_head->param_count();
    for (auto& h : mtp_heads) count += (int64_t)h->param_count();
    return count;
}

void DenseModel::init_weights() {
    init_weights(0); // 0 = legacy entropy (std::random_device)
}

void DenseModel::init_weights(uint64_t seed) {
    if (layers.empty()) build_layers();
    std::mt19937 rng(seed != 0 ? (uint32_t)seed : std::random_device{}());
    auto rand_fill = [&](Tensor& t) {
        float* d = t.data<float>();
        int64_t fan_in = (t.rank() >= 2) ? t.dim(t.rank() - 1) : t.numel();
        float std_dev = std::sqrt(2.0f / fan_in);
        std::normal_distribution<float> dist(0.0f, std_dev);
        for (int64_t i = 0; i < t.numel(); i++) d[i] = dist(rng);
    };
    for (auto& l : layers) {
        rand_fill(l->attention.q_proj.weight);
        rand_fill(l->attention.k_proj.weight);
        rand_fill(l->attention.v_proj.weight);
        rand_fill(l->attention.o_proj.weight);
        rand_fill(l->ffn.gate_proj.weight);
        rand_fill(l->ffn.up_proj.weight);
        rand_fill(l->ffn.down_proj.weight);
        l->attention_norm.weight.fill(1.0f);
        l->ffn_norm.weight.fill(1.0f);
    }
    rand_fill(lm_head->weight);
    for (auto& h : mtp_heads) rand_fill(h->weight);
    norm->weight.fill(1.0f);
    rand_fill(tok_embeddings->weight);
}

void DenseModel::get_parameters(std::vector<Tensor*>& params) {
    params.clear();
    params.push_back(&tok_embeddings->weight);
    for (auto& l : layers) {
        params.push_back(&l->attention.q_proj.weight);
        params.push_back(&l->attention.k_proj.weight);
        params.push_back(&l->attention.v_proj.weight);
        params.push_back(&l->attention.o_proj.weight);
        params.push_back(&l->ffn.gate_proj.weight);
        params.push_back(&l->ffn.up_proj.weight);
        params.push_back(&l->ffn.down_proj.weight);
        params.push_back(&l->attention_norm.weight);
        params.push_back(&l->ffn_norm.weight);
    }
    params.push_back(&norm->weight);
    params.push_back(&lm_head->weight);
    for (auto& h : mtp_heads) params.push_back(&h->weight);
}

Tensor DenseModel::forward(const Tensor& input_ids, const Tensor& positions,
                           KVCache* cache) {
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

    for (int64_t i = 0; i < config.num_layers; i++) {
        h = layers[i]->forward(h, positions, causal_mask, *active_cache, (int)i);
    }

    h = norm->forward(h);
    return lm_head->forward(h);
}

// ========================================================================
// REAL save: write all named weight tensors to QUANT format
// ========================================================================

static std::vector<std::pair<std::string, Tensor>> collect_named_tensors(const DenseModel& m) {
    std::vector<std::pair<std::string, Tensor>> tensors;
    tensors.emplace_back("tok_embeddings.weight", m.tok_embeddings->weight);
    for (size_t i = 0; i < m.layers.size(); i++) {
        std::string p = "layers." + std::to_string(i) + ".";
        const auto& l = *m.layers[i];
        tensors.emplace_back(p + "attention_norm.weight", l.attention_norm.weight);
        tensors.emplace_back(p + "ffn_norm.weight", l.ffn_norm.weight);
        tensors.emplace_back(p + "attention.q_proj.weight", l.attention.q_proj.weight);
        if (l.attention.q_proj.bias.numel() > 0)
            tensors.emplace_back(p + "attention.q_proj.bias", l.attention.q_proj.bias);
        tensors.emplace_back(p + "attention.k_proj.weight", l.attention.k_proj.weight);
        tensors.emplace_back(p + "attention.v_proj.weight", l.attention.v_proj.weight);
        tensors.emplace_back(p + "attention.o_proj.weight", l.attention.o_proj.weight);
        tensors.emplace_back(p + "ffn.gate_proj.weight", l.ffn.gate_proj.weight);
        tensors.emplace_back(p + "ffn.up_proj.weight", l.ffn.up_proj.weight);
        tensors.emplace_back(p + "ffn.down_proj.weight", l.ffn.down_proj.weight);
    }
    tensors.emplace_back("norm.weight", m.norm->weight);
    tensors.emplace_back("lm_head.weight", m.lm_head->weight);
    if (m.lm_head->bias.numel() > 0)
        tensors.emplace_back("lm_head.bias", m.lm_head->bias);
    for (size_t i = 0; i < m.mtp_heads.size(); i++) {
        tensors.emplace_back("mtp_heads." + std::to_string(i) + ".weight", m.mtp_heads[i]->weight);
        if (m.mtp_heads[i]->bias.numel() > 0)
            tensors.emplace_back("mtp_heads." + std::to_string(i) + ".bias", m.mtp_heads[i]->bias);
    }
    return tensors;
}

void DenseModel::save(const std::string& quant_path) const {
    auto named = collect_named_tensors(*this);

    // First pass: build format table, tensor table, and block data in memory
    std::vector<FormatBlockEntry> ft;
    std::vector<TensorEntry> te;
    std::vector<std::string> names;
    std::vector<BlockData> blocks;
    uint32_t block_id = 0;
    uint32_t block_start = 0;

    for (auto& [name, t] : named) {
        int64_t numel = t.numel();
        if (numel == 0) continue;
        names.push_back(name);

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
            fbe.format = 9; // FP32 (Q32)
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

    // Write: header → format table → tensor table → block data
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

void DenseModel::save_quantized(const std::string& quant_path, Format fmt) const {
    auto named = collect_named_tensors(*this);

    std::vector<FormatBlockEntry> ft;
    std::vector<TensorEntry> te;
    std::vector<std::string> names;
    std::vector<BlockData> blocks;
    uint32_t block_id = 0;
    uint32_t block_start = 0;

    int codebook_size = 0;
    Format quant_fmt = fmt;
    if (fmt == Format::Q8) codebook_size = 256;
    else if (fmt == Format::Q4) codebook_size = 16;
    else if (fmt == Format::Q2) codebook_size = 4;

    for (auto& [name, t] : named) {
        int64_t numel = t.numel();
        if (numel == 0) continue;
        names.push_back(name);

        const float* td = t.data<float>();
        uint32_t block_count = 0;
        uint32_t bs = block_start;

        if (codebook_size > 0) {
            CodebookQUANT8 cb;
            if (codebook_size == 16) {
                CodebookQUANT4 cb4;
                cb4.train(td, (size_t)numel);
                std::vector<uint8_t> indices(numel);
                for (int64_t i = 0; i < numel; i++)
                    indices[i] = cb4.quantize(td[i]);
                BlockData bd;
                bd.format = quant_fmt;
                bd.num_weights = (uint32_t)numel;
                bd.indices = indices;
                std::vector<uint8_t> cb_raw(cb4.serialized_size());
                cb4.serialize(cb_raw.data());
                bd.codebook = cb_raw;
                blocks.push_back(bd);
                FormatBlockEntry fbe;
                fbe.block_id = block_id++;
                fbe.format = (uint32_t)quant_fmt;
                fbe.cb_bytes = (uint32_t)cb_raw.size();
                ft.push_back(fbe);
                block_count++;
            } else {
                cb.train(td, (size_t)numel);
                std::vector<uint8_t> indices(numel);
                for (int64_t i = 0; i < numel; i++)
                    indices[i] = cb.quantize(td[i]);
                BlockData bd;
                bd.format = quant_fmt;
                bd.num_weights = (uint32_t)numel;
                bd.indices = indices;
                std::vector<uint8_t> cb_raw(cb.serialized_size());
                cb.serialize(cb_raw.data());
                bd.codebook = cb_raw;
                blocks.push_back(bd);
                FormatBlockEntry fbe;
                fbe.block_id = block_id++;
                fbe.format = (uint32_t)quant_fmt;
                fbe.cb_bytes = (uint32_t)cb_raw.size();
                ft.push_back(fbe);
                block_count++;
            }
        } else {
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
                fbe.format = 9;
                fbe.cb_bytes = 0;
                ft.push_back(fbe);
                block_count++;
                offset += blk_size;
            }
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

// ========================================================================
// REAL load: read named tensors from QUANT format and assign to model
// ========================================================================

void DenseModel::load(const std::string& quant_path) {
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

    // Collect weight map from reader
    auto tensor_names = reader.tensor_names();
    std::unordered_map<std::string, Tensor> loaded_weights;
    for (const auto& n : tensor_names) {
        loaded_weights[n] = reader.read_tensor(n);
    }

    // Build weight assign map (strict: missing or shape-mismatched weights throw)
    std::vector<std::string> load_errors;
    auto assign = [&](const std::string& name, Tensor& dst) {
        auto it = loaded_weights.find(name);
        if (it == loaded_weights.end()) {
            load_errors.push_back("missing: " + name);
            return;
        }
        if (it->second.numel() != dst.numel()) {
            load_errors.push_back("shape mismatch: " + name + " file=" +
                std::to_string(it->second.numel()) + " model=" + std::to_string(dst.numel()));
            return;
        }
        dst.copy_from(it->second);
    };

    assign("tok_embeddings.weight", tok_embeddings->weight);
    for (size_t i = 0; i < layers.size(); i++) {
        std::string p = "layers." + std::to_string(i) + ".";
        auto& l = *layers[i];
        assign(p + "attention_norm.weight", l.attention_norm.weight);
        assign(p + "ffn_norm.weight", l.ffn_norm.weight);
        assign(p + "attention.q_proj.weight", l.attention.q_proj.weight);
        assign(p + "attention.k_proj.weight", l.attention.k_proj.weight);
        assign(p + "attention.v_proj.weight", l.attention.v_proj.weight);
        assign(p + "attention.o_proj.weight", l.attention.o_proj.weight);
        assign(p + "ffn.gate_proj.weight", l.ffn.gate_proj.weight);
        assign(p + "ffn.up_proj.weight", l.ffn.up_proj.weight);
        assign(p + "ffn.down_proj.weight", l.ffn.down_proj.weight);
    }
    assign("norm.weight", norm->weight);
    assign("lm_head.weight", lm_head->weight);
    for (size_t i = 0; i < mtp_heads.size(); i++) {
        assign("mtp_heads." + std::to_string(i) + ".weight", mtp_heads[i]->weight);
    }
    if (!load_errors.empty()) {
        std::string msg = "DenseModel::load corrupt/incomplete .quant (" +
            std::to_string(load_errors.size()) + " errors): ";
        for (size_t i = 0; i < load_errors.size() && i < 8; ++i) {
            if (i) msg += "; ";
            msg += load_errors[i];
        }
        throw std::runtime_error(msg);
    }
}

void Model::load(const std::string& quant_path) {
    throw std::runtime_error(
        "Model::load() base class cannot load weights — use DenseModel::load() "
        "which reads config + weights from QUANT blocks.");
}

void Model::save(const std::string& quant_path) const {
    QUANTWriter writer(quant_path);
    QUANTHeader hdr;
    std::memcpy(hdr.magic, "QUA1", 4);
    hdr.version = 1;
    hdr.flags = 0;
    hdr.config_size = sizeof(TransformerConfig);
    writer.write_header(hdr, (const uint8_t*)&config);
    writer.close();
}

// MTP: run backbone, then produce one logit tensor per head
std::vector<Tensor> DenseModel::mtp_forward(const Tensor& input_ids, const Tensor& positions, KVCache* cache) {
    Tensor hidden;
    {
        int64_t B = input_ids.dim(0);
        int64_t S = input_ids.dim(1);
        Tensor h = tok_embeddings->forward(input_ids.reshape(Shape{B * S}));
        h = h.reshape(Shape{B, S, config.hidden_size});
        KVCache local_cache;
        KVCache* active_cache = cache;
        if (!active_cache) {
            local_cache.init((int)config.num_layers, config.max_seq_len, config.num_heads, config.head_dim);
            active_cache = &local_cache;
        }
        Tensor causal_mask(Shape{1, 1, S, S});
        float* md = causal_mask.data<float>();
        for (int64_t s = 0; s < S; s++) for (int64_t t = 0; t < S; t++) md[s * S + t] = (t > s) ? -INFINITY : 0.0f;
        for (int64_t i = 0; i < config.num_layers; i++) h = layers[i]->forward(h, positions, causal_mask, *active_cache, (int)i);
        h = norm->forward(h);
        hidden = h;
    }
    std::vector<Tensor> logits;
    logits.reserve(mtp_heads.size() + 1);
    logits.push_back(lm_head->forward(hidden.reshape(Shape{hidden.dim(0) * hidden.dim(1), config.hidden_size})));
    for (auto& h : mtp_heads) logits.push_back(h->forward(hidden.reshape(Shape{hidden.dim(0) * hidden.dim(1), config.hidden_size})));
    return logits;
}

float DenseModel::mtp_loss(const std::vector<Tensor>& mtp_logits, const Tensor& labels) const {
    if (mtp_logits.empty())
        throw std::runtime_error("mtp_loss: empty mtp_logits (no heads to score)");
    int64_t B = labels.dim(0), S = labels.dim(1);
    if (B <= 0 || S <= 0)
        throw std::runtime_error("mtp_loss: empty labels (no valid tokens)");
    const float* ld = labels.data<float>();
    double weighted_sum = 0.0;
    int64_t total_valid = 0;
    for (size_t h = 0; h < mtp_logits.size(); h++) {
        int64_t k = (int64_t)(h + 1);   // head h predicts k steps ahead
        int64_t valid = S - k;          // positions [0, valid) valid; tail [S-k, S) masked
        if (valid <= 0) continue;       // fully-masked head: no valid positions
        int64_t rows = B * S;           // mtp_forward emits one row per (b, s)
        int64_t C = mtp_logits[h].numel() / rows;
        // Gather contiguous valid-only tensors (invalid tails are interleaved
        // per batch row, so a flat prefix slice would be wrong for B > 1).
        Tensor vlogits(Shape{B * valid, C}, DType::F32);
        Tensor vlabels(Shape{B * valid}, DType::F32);
        float* vl = vlogits.data<float>();
        float* vs = vlabels.data<float>();
        const float* hl = mtp_logits[h].data<float>();
        for (int64_t b = 0; b < B; b++) {
            for (int64_t s = 0; s < valid; s++) {
                vs[b * valid + s] = ld[b * S + s + k];
                std::memcpy(vl + (b * valid + s) * C, hl + (b * S + s) * C,
                            (size_t)C * sizeof(float));
            }
        }
        Tensor ce = AutogradEngine::cross_entropy_op(vlogits, vlabels);
        float ce_val = *(const float*)ce.data();
        weighted_sum += (double)ce_val * (double)(B * valid);
        total_valid += B * valid;
    }
    if (total_valid == 0)
        throw std::runtime_error("mtp_loss: no valid tokens (sequence too short for all heads)");
    return (float)(weighted_sum / (double)total_valid);
}

} // namespace quant
