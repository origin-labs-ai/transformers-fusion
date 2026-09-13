// Phase 5-F6 split of src/inference/inference_opt.cpp (verbatim move, no behavior change).
// This file: DynamicBatcher (D11) + EmbeddingEndpoint (D18) + Reranker (D19)
// + GrammarDecoder (D20), plus the file-local static helpers text_to_token_ids
// and extract_hidden_state (used only by EmbeddingEndpoint/Reranker, moved with them).
// NOTE: MultiQueryAttention / GroupedQueryAttention / SlidingWindowAttention / ALiBi
// are declared in include/quant/inference_opt.h but have zero definitions anywhere
// in the repo (verified by grep for "MultiQueryAttention::" etc. under src/),
// so there are no bodies to move for them. Declarations stay in the header.
#include "quant/inference_opt.h"
#include "quant/math.h"
#include "quant/int8_quant.h"
#include "quant/random.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <random>
#include <sstream>
#include <cstdio>
namespace quant {

// ===========================================================================
// D11: Dynamic batching — variable batch sizes
// ===========================================================================
DynamicBatcher::DynamicBatcher(Model* model) : model_(model) {}

std::vector<std::string> DynamicBatcher::batch_generate(
    const std::vector<std::string>& prompts, int max_tokens) {

    if (!model_) {
        std::vector<std::string> fallback;
        for (auto& p : prompts) fallback.push_back("output");
        return fallback;
    }

    std::vector<std::vector<int>> token_ids;
    size_t max_len = 0;
    for (auto& p : prompts) {
        std::vector<int> ids;
        for (char c : p) ids.push_back((int)(unsigned char)c);
        if (ids.empty()) ids.push_back(0);
        token_ids.push_back(ids);
        max_len = std::max(max_len, ids.size());
    }

    int64_t B = (int64_t)prompts.size();
    int64_t S = (int64_t)max_len;

    // Pad sequences to max length
    Tensor batch_input(Shape{B, S});
    batch_input.zero_();
    Tensor batch_pos(Shape{B, S});
    for (int64_t b = 0; b < B; b++) {
        for (size_t s = 0; s < token_ids[(size_t)b].size(); s++) {
            batch_input.data<float>()[b * S + s] = (float)token_ids[(size_t)b][s];
            batch_pos.data<float>()[b * S + s] = (float)s;
        }
        for (size_t s = token_ids[(size_t)b].size(); s < (size_t)S; s++) {
            batch_pos.data<float>()[b * S + s] = (float)s;
        }
    }

    Tensor logits = model_->forward(batch_input, batch_pos);
    int64_t V = logits.dim(logits.rank() - 1);

    // Generate tokens for each prompt
    std::vector<std::string> results;
    for (int64_t b = 0; b < B; b++) {
        std::vector<int> gen = token_ids[(size_t)b];

        for (int t = 0; t < max_tokens; t++) {
            int64_t pos = (int64_t)gen.size() - 1;
            Tensor single_input(Shape{1, 1});
            Tensor single_pos(Shape{1, 1});
            single_input.data<float>()[0] = (float)gen.back();
            single_pos.data<float>()[0] = (float)pos;

            Tensor single_logits = model_->forward(single_input, single_pos);
            const float* row = single_logits.data<float>();

            int best = 0;
            for (int v = 1; v < V; v++)
                if (row[v] > row[best]) best = v;
            gen.push_back(best);
        }

        std::string result;
        for (int id : gen) result += std::to_string(id) + " ";
        results.push_back(result);
    }

    return results;
}

// ===========================================================================
// D18: Embedding endpoint — last hidden state extraction
// ===========================================================================
EmbeddingEndpoint::EmbeddingEndpoint(Model* model) : model_(model) {}

static std::vector<int> text_to_token_ids(const std::string& text, int vocab_size) {
    std::vector<int> ids;
    for (char c : text)
        ids.push_back((int)(unsigned char)c % vocab_size);
    if (ids.empty()) ids.push_back(0);
    return ids;
}

static Tensor extract_hidden_state(DenseModel* dm, const std::vector<int>& token_ids) {
    int64_t S = (int64_t)token_ids.size();
    Tensor input(Shape{1, S});
    for (int64_t i = 0; i < S; i++)
        input.data<float>()[i] = (float)token_ids[(size_t)i];

    Tensor positions(Shape{1, S});
    for (int64_t i = 0; i < S; i++)
        positions.data<float>()[i] = (float)i;

    Tensor mask(Shape{1, 1, S, S});
    mask.fill(-INFINITY);
    for (int64_t i = 0; i < S; i++)
        for (int64_t j = 0; j <= i; j++)
            mask.data<float>()[i * S + j] = 0.0f;

    Tensor h = dm->tok_embeddings->forward(input);
    KVCache cache((int)dm->layers.size(), dm->config.max_seq_len,
                  dm->config.num_heads, dm->config.head_dim);
    for (size_t l = 0; l < dm->layers.size(); l++)
        h = dm->layers[l]->forward(h, positions, mask, cache, (int)l);
    h = dm->norm->forward(h);

    return h;
}

Tensor EmbeddingEndpoint::embed(const std::string& text) {
    auto* dm = dynamic_cast<DenseModel*>(model_);
    if (!dm || !model_) {
        int64_t h = model_ ? model_->config.hidden_size : 768;
        Tensor fallback({h});
        fallback.zero_();
        return fallback;
    }

    auto ids = text_to_token_ids(text, (int)model_->vocab_size());
    Tensor hidden = extract_hidden_state(dm, ids);
    int64_t S = hidden.dim(1);
    int64_t D = hidden.dim(2);

    // Mean pool over sequence dimension
    Tensor embedding({D});
    embedding.zero_();
    const float* hd = hidden.data<float>();
    float* ed = embedding.data<float>();
    for (int64_t s = 0; s < S; s++)
        for (int64_t d = 0; d < D; d++)
            ed[d] += hd[s * D + d];
    float inv = 1.0f / (float)S;
    for (int64_t d = 0; d < D; d++)
        ed[d] *= inv;

    return embedding;
}

Tensor EmbeddingEndpoint::embed_batch(const std::vector<std::string>& texts) {
    auto* dm = dynamic_cast<DenseModel*>(model_);
    if (!dm || texts.empty()) {
        return Tensor({(int64_t)texts.size(), model_->config.hidden_size});
    }

    int64_t D = dm->config.hidden_size;
    int64_t B = (int64_t)texts.size();
    Tensor embeddings({B, D});
    embeddings.zero_();

    for (int64_t b = 0; b < B; b++) {
        Tensor single_emb = embed(texts[(size_t)b]);
        float* ed = embeddings.data<float>() + b * D;
        const float* sd = single_emb.data<float>();
        std::memcpy(ed, sd, (size_t)D * sizeof(float));
    }

    return embeddings;
}

// ===========================================================================
// D19: Reranking — cross-encoder scoring
// ===========================================================================
Reranker::Reranker(Model* model) : model_(model) {}

float Reranker::score(const std::string& query, const std::string& document) {
    auto* dm = dynamic_cast<DenseModel*>(model_);
    if (!dm) return 0.0f;

    // Cross-encoder: concatenate query + document tokens
    auto q_ids = text_to_token_ids(query, (int)model_->vocab_size());
    auto d_ids = text_to_token_ids(document, (int)model_->vocab_size());

    std::vector<int> combined = q_ids;
    combined.push_back(1); // [SEP]
    combined.insert(combined.end(), d_ids.begin(), d_ids.end());

    Tensor hidden = extract_hidden_state(dm, combined);
    Tensor logits = dm->lm_head->forward(hidden);

    int64_t S = logits.dim(1);
    int64_t V = logits.dim(2);

    // Use mean of last token's logits as relevance score
    const float* last_row = logits.data<float>() + (S - 1) * V;
    float score_val = 0;
    for (int64_t v = 0; v < V; v++)
        score_val += last_row[v];
    score_val = std::tanh(score_val / (float)V);

    return score_val;
}

std::vector<float> Reranker::score_batch(const std::string& query,
                                          const std::vector<std::string>& documents) {
    std::vector<float> scores;
    scores.reserve(documents.size());
    for (auto& d : documents)
        scores.push_back(score(query, d));
    return scores;
}

// ===========================================================================
// D20: Grammar decoding — enforce JSON/regex token-level constraints
// ===========================================================================
GrammarDecoder::GrammarDecoder(const std::string& grammar_file, int vocab_size) {
    parse_grammar(grammar_file, vocab_size);
}

void GrammarDecoder::parse_grammar(const std::string& grammar_file, int vocab_size) {
    if (vocab_size > 0) vocab_size_ = vocab_size;
    allowed_tokens_.resize(1024, std::vector<bool>((size_t)vocab_size_, true));

    // Determine grammar mode from filename
    bool json_mode = grammar_file.find("json") != std::string::npos ||
                     grammar_file.find("JSON") != std::string::npos;
    bool regex_mode = grammar_file.find("regex") != std::string::npos ||
                       grammar_file.find("re") != std::string::npos;

    if (json_mode) {
        // In JSON mode, constrain: must start with { or [, must contain valid JSON structure
        for (size_t state = 0; state < allowed_tokens_.size(); state++) {
            for (int v = 0; v < vocab_size_; v++) {
                bool allowed = true;
                if (state == 0) {
                    // First token must be { or [ for JSON
                    allowed = (v == 0 || v == 1); // simplified: 0={, 1=[
                }
                allowed_tokens_[state][(size_t)v] = allowed;
            }
        }
        // JSON structural tokens always allowed
        for (size_t state = 0; state < allowed_tokens_.size(); state++) {
            for (int v = 0; v < 256; v++) {
                if (v == '{' || v == '}' || v == '[' || v == ']' ||
                    v == '"' || v == ':' || v == ',' || v == ' ' ||
                    v == '\n' || v == '\t' || v == '0' || v == '1' ||
                    v == '2' || v == '3' || v == '4' || v == '5' ||
                    v == '6' || v == '7' || v == '8' || v == '9' ||
                    v == '-' || v == '.' || v == 'e' || v == 'E' ||
                    v == 't' || v == 'r' || v == 'u' || v == 'e' ||
                    v == 'f' || v == 'a' || v == 'l' || v == 's' ||
                    v == 'n' || v == 'u' || v == 'l') {
                    allowed_tokens_[state][(size_t)v] = true;
                }
            }
        }
    } else if (regex_mode) {
        // Regex mode: allow alphanumeric and regex special chars
        for (size_t state = 0; state < allowed_tokens_.size(); state++) {
            for (int v = 0; v < vocab_size_; v++) {
                allowed_tokens_[state][(size_t)v] = true;
            }
        }
    } else {
        // Default: all tokens allowed
        for (size_t state = 0; state < allowed_tokens_.size(); state++) {
            allowed_tokens_[state].assign((size_t)vocab_size_, true);
        }
    }
}

bool GrammarDecoder::matches_prefix(const std::vector<int>& prefix) const {
    // Check if prefix is valid according to grammar constraints
    for (int token : prefix) {
        if (token < 0 || token >= vocab_size_) return false;
    }
    return true;
}

std::vector<int> GrammarDecoder::constrain(const std::vector<float>& logits,
                                            const std::vector<int>& prefix) {
    std::vector<int> result;
    size_t state = prefix.size();

    // Determine which tokens are allowed at this state
    const auto& allowed = allowed_tokens_[std::min(state, allowed_tokens_.size() - 1)];

    // Find best token among allowed ones
    int best_idx = -1;
    float best_val = -INFINITY;
    int n_logits = std::min((int)logits.size(), vocab_size_);

    for (int v = 0; v < n_logits; v++) {
        if (allowed[(size_t)v] && logits[(size_t)v] > best_val) {
            best_val = logits[(size_t)v];
            best_idx = v;
        }
    }

    // If no allowed token found, fallback to argmax
    if (best_idx < 0) {
        for (int v = 0; v < n_logits; v++)
            if (logits[(size_t)v] > best_val) {
                best_val = logits[(size_t)v];
                best_idx = v;
            }
    }

    result.push_back(best_idx);
    return result;
}

} // namespace quant
