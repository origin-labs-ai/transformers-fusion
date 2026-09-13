#include "quant/parquet_streamer.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <random>

namespace quant {

// ========================================================================
// Minimal Parquet-like JSONL reader (parquet files via HuggingFace API
// often come as .jsonl.zst or .jsonl — this reads raw JSONL text)
// ========================================================================

ParquetReader::ParquetReader()
    : fp_(nullptr), total_tokens_(0), rows_read_(0), file_size_(0) {}

ParquetReader::~ParquetReader() { close(); }

bool ParquetReader::open(const std::string& path) {
    close();
    fp_ = fopen(path.c_str(), "rb");
    if (!fp_) return false;
    fseek(fp_, 0, SEEK_END);
    file_size_ = ftell(fp_);
    fseek(fp_, 0, SEEK_SET);
    total_tokens_ = 0;
    rows_read_ = 0;
    return true;
}

bool ParquetReader::next_batch(float* tokens, int64_t max_tokens, int64_t& actual_tokens) {
    if (!fp_) return false;
    actual_tokens = 0;

    char line[8192];
    while (actual_tokens < max_tokens && fgets(line, sizeof(line), fp_)) {
        rows_read_++;
        // Simple extraction: find "value" fields in JSON and extract text
        // This handles ShareGPT-style conversation format
        char* p = strstr(line, "\"value\"");
        if (!p) continue;
        p = strchr(p + 7, ':');
        if (!p) continue;
        p = strchr(p + 1, '"');
        if (!p) continue;
        p++;
        char* end = strchr(p, '"');
        if (!end) continue;

        // Convert text chars to token IDs (simple ASCII mapping for now)
        for (char* c = p; c < end && actual_tokens < max_tokens; c++) {
            tokens[actual_tokens++] = (float)(unsigned char)*c;
        }
        total_tokens_++;
    }
    return actual_tokens > 0;
}

void ParquetReader::reset() {
    if (fp_) fseek(fp_, 0, SEEK_SET);
    total_tokens_ = 0;
    rows_read_ = 0;
}

void ParquetReader::close() {
    if (fp_) { fclose(fp_); fp_ = nullptr; }
}

// ========================================================================
// Streaming Dataset — manages multiple shards
// ========================================================================

StreamingDataset::StreamingDataset(const std::string& data_dir, int64_t vocab_size)
    : vocab_size_(vocab_size), total_tokens_(0), current_shard_(0),
      epochs_(0), buffer_pos_(0), buffer_len_(0) {
    (void)data_dir;
    text_buffer_.resize(1 << 20); // 1M float buffer
}

StreamingDataset::~StreamingDataset() {}

void StreamingDataset::add_shard(const std::string& parquet_path) {
    ParquetShard s;
    s.path = parquet_path;
    s.num_rows = 0;
    s.num_tokens = 0;
    shards_.push_back(s);
    total_tokens_ += s.num_tokens;
}

bool StreamingDataset::next_batch(float* input_ids, float* labels,
                                    int64_t batch_size, int64_t seq_len) {
    int64_t needed = batch_size * seq_len;
    int64_t filled = 0;

    while (filled < needed) {
        // Try to get tokens from buffer
        while (buffer_pos_ < buffer_len_ && filled < needed) {
            input_ids[filled] = text_buffer_[buffer_pos_];
            if (filled + 1 < needed)
                labels[filled] = text_buffer_[buffer_pos_ + 1];
            else
                labels[filled] = text_buffer_[buffer_pos_];
            buffer_pos_++;
            filled++;
        }

        // Need more data
        if (filled < needed) {
            if (!fill_buffer()) {
                // All shards exhausted
                if (filled == 0) return false;
                // Pad remaining with zeros
                while (filled < needed) {
                    input_ids[filled] = 0.0f;
                    labels[filled] = 0.0f;
                    filled++;
                }
                break;
            }
        }
    }
    return filled > 0;
}

void StreamingDataset::reset() {
    current_shard_ = 0;
    buffer_pos_ = 0;
    buffer_len_ = 0;
    epochs_ = 0;
}

void StreamingDataset::shuffle_shards() {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(shards_.begin(), shards_.end(), g);
}

bool StreamingDataset::fill_buffer() {
    while (current_shard_ < (int64_t)shards_.size()) {
        ParquetReader reader;
        if (reader.open(shards_[(size_t)current_shard_].path)) {
            int64_t got = 0;
            reader.next_batch(text_buffer_.data(), (int64_t)text_buffer_.size(), got);
            buffer_len_ = got;
            buffer_pos_ = 0;
            reader.close();
            current_shard_++;
            if (buffer_len_ > 0) return true;
        } else {
            current_shard_++;
        }
    }
    // All shards done — wrap around for next epoch
    if (!shards_.empty()) {
        current_shard_ = 0;
        epochs_++;
        shuffle_shards();
        return fill_buffer();
    }
    return false;
}

} // namespace quant
