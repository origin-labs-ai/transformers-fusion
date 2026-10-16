#pragma once
#include "quant/trainer.h"
#include "quant/random.h"
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace quant {
namespace dense_engine {

// Real dataloader: tokenizes a text file once, yields (input_ids, labels)
// shifted pairs, with shuffle/reset/num_batches. Backed by Tokenizer encode.
// No silent empty batches — exhausted loader returns false.
class DenseDataLoader {
public:
    DenseDataLoader(Tokenizer* tok, const std::string& data_path,
                    int64_t batch_size, int64_t seq_length);
    bool next_batch(Tensor& input_ids, Tensor& labels);
    void shuffle(int epoch = 0);
    void reset();
    int64_t num_batches() const;
    int64_t batch_size() const { return batch_size_; }
    int64_t seq_length() const { return seq_length_; }
private:
    Tokenizer* tok_ = nullptr;
    std::vector<int> tokens_;
    int64_t batch_size_ = 8;
    int64_t seq_length_ = 512;
    int64_t pos_ = 0;
    void load_file(const std::string& path);
};

// Streaming loader: reads fixed-size text chunks, tokenizes on demand,
// never holds the whole file. Real rolling buffer with refill.
class StreamDataLoader {
public:
    StreamDataLoader(Tokenizer* tok, const std::string& data_path,
                     int64_t batch_size, int64_t seq_length,
                     int64_t chunk_bytes = 1 << 20);
    ~StreamDataLoader();
    bool next_batch(Tensor& input_ids, Tensor& labels);
    void reset();
    int64_t batches_served() const { return batches_; }
private:
    Tokenizer* tok_ = nullptr;
    std::string path_;
    int64_t batch_size_ = 8;
    int64_t seq_length_ = 512;
    int64_t chunk_bytes_ = 1 << 20;
    std::ifstream file_;
    std::vector<int> buf_;
    int64_t batches_ = 0;
    bool eof_ = false;
    void refill();
    bool take_batch(Tensor& input_ids, Tensor& labels);
};

// Augment policy: token dropout + span mask + gaussian logit noise,
// real in-place tensor math on int ids / float logits.
struct AugmentPolicy {
    float token_dropout = 0.0f;
    float span_mask_prob = 0.0f;
    int64_t span_len = 8;
    float logit_noise_std = 0.0f;
    int pad_id = 0;
    uint64_t seed = 1234;
    void apply_ids(Tensor& ids);
    void apply_logits(Tensor& logits);
};

// Curriculum scheduler: seq-len ramp + difficulty mix readout, real math.
struct Curriculum {
    int64_t min_len = 64;
    int64_t max_len = 512;
    int total_epochs = 3;
    int current_epoch = 0;
    int64_t effective_len() const;
    float difficulty_mix() const;
};

// Real checkpoint: writes/reads raw param tensors + step count with magic.
class DenseCheckpoint {
public:
    static void save(const std::string& path, const std::vector<Tensor*>& params, int64_t step);
    static int64_t load(const std::string& path, const std::vector<Tensor*>& params);
};

} // namespace dense_engine
} // namespace quant
