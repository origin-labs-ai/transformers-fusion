#include "dataloader.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

namespace quant {
namespace dense_engine {

DenseDataLoader::DenseDataLoader(Tokenizer* tok, const std::string& data_path,
                                 int64_t batch_size, int64_t seq_length)
    : tok_(tok), batch_size_(batch_size), seq_length_(seq_length) {
    if (!tok_) throw std::runtime_error("[dense-engine] DenseDataLoader needs Tokenizer*");
    if (batch_size_ <= 0 || seq_length_ <= 0)
        throw std::runtime_error("[dense-engine] bad batch/seq slice");
    load_file(data_path);
}

void DenseDataLoader::load_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("[dense-engine] cannot open data file: " + path);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (text.empty()) throw std::runtime_error("[dense-engine] empty data file: " + path);
    tokens_ = tok_->encode(text);
    if ((int64_t)tokens_.size() < batch_size_ * (seq_length_ + 1) + 1)
        throw std::runtime_error("[dense-engine] data file too small for one batch");
}

bool DenseDataLoader::next_batch(Tensor& input_ids, Tensor& labels) {
    int64_t need = batch_size_ * (seq_length_ + 1);
    if (pos_ + need > (int64_t)tokens_.size()) return false;
    input_ids = Tensor({batch_size_, seq_length_}, DType::I32);
    labels = Tensor({batch_size_, seq_length_}, DType::I32);
    int* ii = input_ids.data<int>();
    int* ll = labels.data<int>();
    for (int64_t b = 0; b < batch_size_; b++) {
        for (int64_t s = 0; s < seq_length_; s++) {
            ii[b * seq_length_ + s] = tokens_[(size_t)(pos_ + b * (seq_length_ + 1) + s)];
            ll[b * seq_length_ + s] = tokens_[(size_t)(pos_ + b * (seq_length_ + 1) + s + 1)];
        }
    }
    pos_ += need;
    return true;
}

void DenseDataLoader::shuffle(int epoch) {
    int64_t stride = seq_length_ + 1;
    int64_t nseq = (int64_t)tokens_.size() / stride;
    std::vector<int> idx((size_t)nseq);
    for (int64_t i = 0; i < nseq; i++) idx[(size_t)i] = (int)i;
    std::mt19937 rng(1234u + (uint32_t)epoch);
    std::shuffle(idx.begin(), idx.end(), rng);
    std::vector<int> out;
    out.reserve(tokens_.size());
    for (int v : idx)
        for (int64_t s = 0; s < stride; s++) out.push_back(tokens_[(size_t)v * (size_t)stride + (size_t)s]);
    tokens_.swap(out);
    pos_ = 0;
}

void DenseDataLoader::reset() { pos_ = 0; }

int64_t DenseDataLoader::num_batches() const {
    int64_t need = batch_size_ * (seq_length_ + 1);
    if (need <= 0) return 0;
    return (int64_t)tokens_.size() / need;
}

static const char kMagic[8] = {'D', 'E', 'N', 'S', 'E', 'C', 'K', '1'};

void DenseCheckpoint::save(const std::string& path, const std::vector<Tensor*>& params, int64_t step) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("[dense-engine] cannot write checkpoint: " + path);
    f.write(kMagic, 8);
    f.write(reinterpret_cast<const char*>(&step), sizeof(step));
    int64_t n = (int64_t)params.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (auto* p : params) {
        int64_t ne = p->numel();
        f.write(reinterpret_cast<const char*>(&ne), sizeof(ne));
        f.write(reinterpret_cast<const char*>(p->data<float>()), ne * sizeof(float));
    }
    if (!f) throw std::runtime_error("[dense-engine] checkpoint write failed: " + path);
}

int64_t DenseCheckpoint::load(const std::string& path, const std::vector<Tensor*>& params) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("[dense-engine] cannot read checkpoint: " + path);
    char magic[8] = {};
    f.read(magic, 8);
    if (std::memcmp(magic, kMagic, 8) != 0)
        throw std::runtime_error("[dense-engine] bad checkpoint magic: " + path);
    int64_t step = 0;
    f.read(reinterpret_cast<char*>(&step), sizeof(step));
    int64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (n != (int64_t)params.size())
        throw std::runtime_error("[dense-engine] checkpoint param count mismatch");
    for (auto* p : params) {
        int64_t ne = 0;
        f.read(reinterpret_cast<char*>(&ne), sizeof(ne));
        if (ne != p->numel())
            throw std::runtime_error("[dense-engine] checkpoint shape mismatch");
        f.read(reinterpret_cast<char*>(p->data<float>()), ne * sizeof(float));
    }
    if (!f) throw std::runtime_error("[dense-engine] checkpoint read failed: " + path);
    return step;
}

StreamDataLoader::StreamDataLoader(Tokenizer* tok, const std::string& data_path,
                                   int64_t batch_size, int64_t seq_length,
                                   int64_t chunk_bytes)
    : tok_(tok), path_(data_path), batch_size_(batch_size),
      seq_length_(seq_length), chunk_bytes_(chunk_bytes) {
    if (!tok_) throw std::runtime_error("[dense-engine] StreamDataLoader needs Tokenizer*");
    if (batch_size_ <= 0 || seq_length_ <= 0 || chunk_bytes_ <= 0)
        throw std::runtime_error("[dense-engine] bad stream slice");
    file_.open(path_, std::ios::binary);
    if (!file_) throw std::runtime_error("[dense-engine] cannot open: " + path_);
}

StreamDataLoader::~StreamDataLoader() { if (file_.is_open()) file_.close(); }

void StreamDataLoader::reset() {
    file_.clear();
    file_.seekg(0, std::ios::beg);
    buf_.clear();
    batches_ = 0;
    eof_ = false;
}

void StreamDataLoader::refill() {
    if (eof_) return;
    std::string chunk((size_t)chunk_bytes_, '\0');
    file_.read(chunk.data(), chunk_bytes_);
    std::streamsize got = file_.gcount();
    if (got <= 0) { eof_ = true; return; }
    chunk.resize((size_t)got);
    std::vector<int> ids = tok_->encode(chunk);
    buf_.insert(buf_.end(), ids.begin(), ids.end());
}

bool StreamDataLoader::take_batch(Tensor& input_ids, Tensor& labels) {
    int64_t need = batch_size_ * (seq_length_ + 1);
    if ((int64_t)buf_.size() < need) return false;
    input_ids = Tensor({batch_size_, seq_length_}, DType::I32);
    labels = Tensor({batch_size_, seq_length_}, DType::I32);
    int* ii = input_ids.data<int>();
    int* ll = labels.data<int>();
    for (int64_t b = 0; b < batch_size_; b++) {
        for (int64_t s = 0; s < seq_length_; s++) {
            ii[b * seq_length_ + s] = buf_[(size_t)(b * (seq_length_ + 1) + s)];
            ll[b * seq_length_ + s] = buf_[(size_t)(b * (seq_length_ + 1) + s + 1)];
        }
    }
    buf_.erase(buf_.begin(), buf_.begin() + (size_t)need);
    batches_++;
    return true;
}

bool StreamDataLoader::next_batch(Tensor& input_ids, Tensor& labels) {
    int64_t need = batch_size_ * (seq_length_ + 1);
    while ((int64_t)buf_.size() < need && !eof_) refill();
    return take_batch(input_ids, labels);
}

static uint64_t xorshift64(uint64_t& s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return s;
}

void AugmentPolicy::apply_ids(Tensor& ids) {
    if (token_dropout <= 0 && span_mask_prob <= 0) return;
    int64_t n = ids.numel();
    int* d = ids.data<int>();
    uint64_t s = seed ? seed : 0x9E3779B97F4A7C15ull;
    for (int64_t i = 0; i < n; i++) {
        double u = (double)(xorshift64(s) >> 11) / (double)0x1FFFFF;
        if (u < token_dropout) { d[i] = pad_id; continue; }
        double v = (double)(xorshift64(s) >> 11) / (double)0x1FFFFF;
        if (v < span_mask_prob) {
            for (int64_t k = 0; k < span_len && i + k < n; k++) d[i + k] = pad_id;
            i += span_len - 1;
        }
    }
}

void AugmentPolicy::apply_logits(Tensor& logits) {
    if (logit_noise_std <= 0) return;
    int64_t n = logits.numel();
    float* d = logits.data<float>();
    uint64_t s = seed ^ 0xD1B54A32ull;
    for (int64_t i = 0; i < n; i++) {
        double u = (double)(xorshift64(s) >> 11) / (double)0x1FFFFF - 0.5;
        d[i] += (float)(u * 3.4641016 * logit_noise_std);
    }
}

int64_t Curriculum::effective_len() const {
    if (total_epochs <= 0) return max_len;
    float t = (float)current_epoch / (float)total_epochs;
    t = std::max(0.0f, std::min(1.0f, t));
    return min_len + (int64_t)((max_len - min_len) * t * t * (3.0f - 2.0f * t));
}

float Curriculum::difficulty_mix() const {
    if (total_epochs <= 0) return 1.0f;
    float t = (float)current_epoch / (float)total_epochs;
    return std::max(0.0f, std::min(1.0f, t));
}

} // namespace dense_engine
} // namespace quant
