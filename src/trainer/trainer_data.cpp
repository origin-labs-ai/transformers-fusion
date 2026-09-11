#define NOMINMAX
#include "quant/trainer.h"
#include "quant/math.h"
#include "quant/autograd.h"
#include "quant/optimizer.h"
#include "quant/transformer.h"
#include "quant/flash_attention.h"
#include "quant/detail/fp16.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace quant {

static inline uint16_t float_to_fp16(float v) {
    uint32_t u32;
    memcpy(&u32, &v, sizeof(u32));
    uint32_t sign = (u32 >> 31) & 1;
    uint32_t exp = (u32 >> 23) & 0xFF;
    uint32_t mant = u32 & 0x7FFFFF;
    if (exp == 0) return (uint16_t)(sign << 15);
    if (exp >= 0x8E) return (uint16_t)((sign << 15) | 0x7C00 | (mant ? 0x200 : 0));
    uint16_t f16 = (uint16_t)((sign << 15) | ((exp - 112) << 10) | (mant >> 13));
    return f16;
}


// FP16 bits -> float — see quant/detail/fp16.h (DEDUP).
using quant::detail::fp16_to_float;

DataLoader::DataLoader(Tokenizer* tokenizer, const std::string& data_path,
                       int64_t batch_size, int64_t seq_length, bool stream_from_disk)
    : tokenizer_(tokenizer), batch_size_(batch_size), seq_length_(seq_length),
      current_pos_(0), streaming_(stream_from_disk), data_path_(data_path) {
    if (streaming_) {
        file_stream_.open(data_path, std::ios::binary);
        if (!file_stream_.is_open()) {
            num_batches_ = 0;
            return;
        }
        file_stream_.seekg(0, std::ios::end);
        int64_t file_size = (int64_t)file_stream_.tellg();
        file_stream_.seekg(0);
        int64_t approx_tokens = file_size / 4;
        int64_t tokens_per_batch = batch_size_ * seq_length_;
        num_batches_ = (std::max)(approx_tokens / tokens_per_batch, (int64_t)1);
        tokenize_chunk();
    } else {
        std::ifstream f(data_path, std::ios::binary | std::ios::ate);
        if (!f.is_open()) {
            num_batches_ = 0;
            return;
        }
        size_t size = (size_t)f.tellg();
        f.seekg(0);
        std::string text((size_t)size, '\0');
        f.read(&text[0], size);
        tokenized_data_ = tokenizer_->encode(text);
        int64_t total_tokens = (int64_t)tokenized_data_.size();
        int64_t tokens_per_batch = batch_size_ * seq_length_;
        num_batches_ = total_tokens / tokens_per_batch;
    }
}

DataLoader::DataLoader(Tokenizer* tokenizer, const std::string& data_path,
                       int64_t batch_size, int64_t seq_length,
                       bool stream_from_disk, int num_workers,
                       int64_t prefetch_capacity, bool use_mmap)
    : tokenizer_(tokenizer), batch_size_(batch_size), seq_length_(seq_length),
      current_pos_(0), streaming_(stream_from_disk), data_path_(data_path),
      num_workers_(num_workers), prefetch_capacity_(prefetch_capacity),
      use_mmap_(use_mmap) {
    if (use_mmap_) {
        open_mmap(data_path_);
        if (mmap_ptr_) {
            std::string text((const char*)mmap_ptr_, mmap_size_);
            tokenized_data_ = tokenizer_->encode(text);
            int64_t total_tokens = (int64_t)tokenized_data_.size();
            int64_t tokens_per_batch = batch_size_ * seq_length_;
            num_batches_ = total_tokens / tokens_per_batch;
        }
    } else if (streaming_) {
        file_stream_.open(data_path, std::ios::binary);
        if (!file_stream_.is_open()) { num_batches_ = 0; return; }
        file_stream_.seekg(0, std::ios::end);
        int64_t file_size = (int64_t)file_stream_.tellg();
        file_stream_.seekg(0);
        int64_t approx_tokens = file_size / 4;
        int64_t tokens_per_batch = batch_size_ * seq_length_;
        num_batches_ = (std::max)(approx_tokens / tokens_per_batch, (int64_t)1);
        tokenize_chunk();
    } else {
        std::ifstream f(data_path, std::ios::binary | std::ios::ate);
        if (!f.is_open()) { num_batches_ = 0; return; }
        size_t size = (size_t)f.tellg();
        f.seekg(0);
        std::string text((size_t)size, '\0');
        f.read(&text[0], size);
        tokenized_data_ = tokenizer_->encode(text);
        int64_t total_tokens = (int64_t)tokenized_data_.size();
        int64_t tokens_per_batch = batch_size_ * seq_length_;
        num_batches_ = total_tokens / tokens_per_batch;
    }
    if (num_workers_ > 0 && prefetch_capacity_ > 0)
        start_prefetch();
}

void DataLoader::open_mmap(const std::string& path) {
#ifdef _WIN32
    file_handle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file_handle_ == INVALID_HANDLE_VALUE) return;
    LARGE_INTEGER li;
    GetFileSizeEx(file_handle_, &li);
    mmap_size_ = (size_t)li.QuadPart;
    mmap_handle_ = CreateFileMappingA(file_handle_, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mmap_handle_) { CloseHandle(file_handle_); file_handle_ = nullptr; return; }
    mmap_ptr_ = MapViewOfFile(mmap_handle_, FILE_MAP_READ, 0, 0, 0);
#else
    mmap_fd_ = open(path.c_str(), O_RDONLY);
    if (mmap_fd_ < 0) return;
    struct stat st;
    fstat(mmap_fd_, &st);
    mmap_size_ = (size_t)st.st_size;
    mmap_ptr_ = mmap(NULL, mmap_size_, PROT_READ, MAP_PRIVATE, mmap_fd_, 0);
    if (mmap_ptr_ == MAP_FAILED) { mmap_ptr_ = nullptr; close(mmap_fd_); mmap_fd_ = -1; }
#endif
}

void DataLoader::close_mmap() {
    if (!mmap_ptr_) return;
#ifdef _WIN32
    UnmapViewOfFile(mmap_ptr_);
    if (mmap_handle_) CloseHandle(mmap_handle_);
    if (file_handle_) CloseHandle(file_handle_);
#else
    munmap(mmap_ptr_, mmap_size_);
    if (mmap_fd_ >= 0) close(mmap_fd_);
#endif
    mmap_ptr_ = nullptr;
    mmap_size_ = 0;
}

DataLoader::~DataLoader() {
    stop_prefetch();
    close_mmap();
    if (file_stream_.is_open()) file_stream_.close();
}

void DataLoader::prefetch_worker() {
    while (prefetch_running_) {
        Tensor input_ids(Shape{batch_size_, seq_length_}, DType::F32);
        Tensor labels(Shape{batch_size_, seq_length_}, DType::F32);
        bool ok;
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex_);
            ok = next_batch(input_ids, labels);
        }
        if (!ok) break;
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex_);
            if (prefetch_queue_.size() < (size_t)prefetch_capacity_)
                prefetch_queue_.push({input_ids.clone(), labels.clone()});
        }
    }
}

void DataLoader::start_prefetch() {
    prefetch_running_ = true;
    prefetch_thread_ = std::thread(&DataLoader::prefetch_worker, this);
}

void DataLoader::stop_prefetch() {
    prefetch_running_ = false;
    if (prefetch_thread_.joinable()) prefetch_thread_.join();
}

void DataLoader::apply_augmentation(Tensor& input_ids, Tensor& labels) {
    if (!aug_cfg_.enabled) return;
    float* id = input_ids.data<float>();
    float* ld = labels.data<float>();
    int64_t n = input_ids.numel();
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, aug_cfg_.noise_std);
    for (int64_t i = 0; i < n; i++) {
        if (aug_cfg_.mask_prob > 0 && dist(rng) < aug_cfg_.mask_prob)
            id[i] = 0;
        if (aug_cfg_.replace_prob > 0 && dist(rng) < aug_cfg_.replace_prob)
            id[i] = (float)(rng() % 32000);
        if (aug_cfg_.noise_std > 0)
            id[i] += noise(rng);
        if (aug_cfg_.mask_prob > 0 && dist(rng) < aug_cfg_.mask_prob)
            ld[i] = 0;
    }
}

void DataLoader::tokenize_chunk() {
    if (!file_stream_.is_open()) return;
    stream_chunk_.clear();
    std::string buffer;
    buffer.resize(STREAM_CHUNK_TOKENS * 4);
    file_stream_.read(&buffer[0], STREAM_CHUNK_TOKENS * 4);
    size_t bytes_read = (size_t)file_stream_.gcount();
    if (bytes_read == 0) return;
    buffer.resize(bytes_read);
    tokenized_data_ = tokenizer_->encode(buffer);
    stream_file_offset_ = 0;
}

bool DataLoader::next_batch(Tensor& input_ids, Tensor& labels) {
    if (current_pos_ >= num_batches_) return false;

    if (streaming_) {
        int64_t tokens_needed = batch_size_ * seq_length_;
        while ((int64_t)(stream_chunk_.size()) - stream_file_offset_ < tokens_needed + 1) {
            if (file_stream_.eof()) return false;
            tokenize_chunk();
            if (tokenized_data_.empty()) return false;
            stream_chunk_.insert(stream_chunk_.end(),
                                 tokenized_data_.begin(), tokenized_data_.end());
        }
        float* id = (float*)input_ids.data();
        float* ld = (float*)labels.data();
        for (int64_t i = 0; i < tokens_needed; i++) {
            id[i] = (float)stream_chunk_[stream_file_offset_ + i];
            ld[i] = (float)stream_chunk_[stream_file_offset_ + i + 1];
        }
        stream_file_offset_ += tokens_needed;
        current_pos_++;
        return true;
    }

    int64_t start = current_pos_ * batch_size_ * seq_length_;
    int64_t end = start + batch_size_ * seq_length_;
    if (end + 1 > (int64_t)tokenized_data_.size()) return false;

    float* id = (float*)input_ids.data();
    float* ld = (float*)labels.data();

    for (int64_t i = 0; i < batch_size_ * seq_length_; i++) {
        int64_t src_idx = start + i;
        int64_t label_idx = src_idx + 1;
        if (label_idx >= (int64_t)tokenized_data_.size()) {
            label_idx = (int64_t)tokenized_data_.size() - 1;
        }
        id[i] = (float)tokenized_data_[src_idx];
        ld[i] = (float)tokenized_data_[label_idx];
    }

    current_pos_++;
    return true;
}

void DataLoader::shuffle(int epoch) {
    if (tokenized_data_.empty()) return;
    std::mt19937 g(42 + epoch);
    int64_t seq_len = seq_length_;
    if (seq_len <= 0) seq_len = 512;
    int64_t num_seqs = (int64_t)tokenized_data_.size() / seq_len;
    std::vector<std::vector<int>> sequences(num_seqs);
    for (int64_t i = 0; i < num_seqs; i++) {
        int64_t start = i * seq_len;
        int64_t end = std::min(start + seq_len, (int64_t)tokenized_data_.size());
        sequences[i].assign(tokenized_data_.begin() + start, tokenized_data_.begin() + end);
    }
    std::shuffle(sequences.begin(), sequences.end(), g);
    tokenized_data_.clear();
    for (auto& seq : sequences) {
        tokenized_data_.insert(tokenized_data_.end(), seq.begin(), seq.end());
    }
    int64_t remaining = (int64_t)tokenized_data_.size() % seq_len;
    if (remaining > 0 && num_seqs > 0) {
        std::mt19937 g2(42 + epoch + 1000);
        auto last_start = tokenized_data_.end() - remaining;
        std::shuffle(last_start, tokenized_data_.end(), g2);
    }
    current_pos_ = 0;
}

void DataLoader::reset() {
    current_pos_ = 0;
    if (streaming_ && file_stream_.is_open()) {
        file_stream_.clear();
        file_stream_.seekg(0);
        stream_chunk_.clear();
        stream_file_offset_ = 0;
        tokenize_chunk();
    }
}

int64_t DataLoader::num_batches() const {
    return num_batches_;
}

StreamingDataLoader::StreamingDataLoader(Tokenizer* tokenizer, const std::string& data_path,
                                         int64_t batch_size, int64_t seq_length)
    : tokenizer_(tokenizer), batch_size_(batch_size), seq_length_(seq_length) {
    file_.open(data_path, std::ios::binary);
    if (!file_.is_open()) { num_batches_ = 0; return; }
    file_.seekg(0, std::ios::end);
    int64_t file_size = (int64_t)file_.tellg();
    file_.seekg(0);
    int64_t approx_tokens = file_size / 4;
    int64_t tokens_per_batch = batch_size_ * seq_length_;
    num_batches_ = (std::max)(approx_tokens / tokens_per_batch, (int64_t)1);
    fill_buffer();
}

void StreamingDataLoader::fill_buffer() {
    if (eof_) return;
    std::string chunk;
    chunk.resize(chunk_bytes_);
    file_.read(&chunk[0], chunk_bytes_);
    size_t got = (size_t)file_.gcount();
    if (got == 0) { eof_ = true; return; }
    chunk.resize(got);
    auto toks = tokenizer_->encode(chunk);
    buffer_.insert(buffer_.end(), toks.begin(), toks.end());
}

bool StreamingDataLoader::next_batch(Tensor& input_ids, Tensor& labels) {
    int64_t need = batch_size_ * seq_length_;
    while ((int64_t)buffer_.size() < need + 1) {
        if (eof_) return false;
        fill_buffer();
        if (eof_ && (int64_t)buffer_.size() < need + 1) return false;
    }
    float* id = input_ids.data<float>();
    float* ld = labels.data<float>();
    for (int64_t i = 0; i < need; i++) {
        id[i] = (float)buffer_[i];
        ld[i] = (float)buffer_[i + 1];
    }
    buffer_.erase(buffer_.begin(), buffer_.begin() + need);
    return true;
}

void StreamingDataLoader::reset() {
    if (file_.is_open()) {
        file_.clear();
        file_.seekg(0);
    }
    buffer_.clear();
    eof_ = false;
    fill_buffer();
}

} // namespace quant
