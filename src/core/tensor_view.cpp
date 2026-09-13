// ============================================================================
// PILLAR 1: TensorView + MmapDataLoader + MinHash Implementation
// ============================================================================

#include "quant/tensor_view.h"
#include "quant/mmap_dataloader.h"
#include <algorithm>
#include <cstring>
#include <numeric>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#undef min
#undef max
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace quant {

// ============================================================================
// MinHash Deduplication Filter
// ============================================================================

MinHashDedup::MinHashDedup(int64_t ngram_size, int num_hashes, int64_t max_capacity)
    : ngram_size_(ngram_size), num_hashes_(num_hashes), max_capacity_(max_capacity) {
    fingerprints_.resize(max_capacity, 0);
}

uint64_t MinHashDedup::compute_fingerprint(const int64_t* tokens, int64_t n) const {
    // Use a simple polynomial hash: h = sum(tokens[i] * base^i) mod large_prime
    // This is fast and has good distribution for n-grams
    uint64_t hash = 0;
    uint64_t base = 1315423911u; // Large prime for mixing
    for (int64_t i = 0; i < n; ++i) {
        hash ^= static_cast<uint64_t>(tokens[i]) * base;
        hash = (hash << 13) | (hash >> 19); // Rotate bits
        base *= 2654435761u; // Golden ratio hash
    }
    return hash;
}

bool MinHashDedup::check_and_insert(uint64_t fp) {
    int64_t idx = fp % max_capacity_;
    if (fingerprints_[idx] == fp) {
        return true; // Duplicate found
    }
    fingerprints_[idx] = fp;
    return false;
}

bool MinHashDedup::is_duplicate(const int64_t* tokens, int64_t n) {
    if (n < ngram_size_) return false;
    total_checked_++;
    uint64_t fp = compute_fingerprint(tokens, ngram_size_);
    if (check_and_insert(fp)) {
        duplicates_found_++;
        return true;
    }
    return false;
}

void MinHashDedup::insert(const int64_t* tokens, int64_t n) {
    if (n < ngram_size_) return;
    uint64_t fp = compute_fingerprint(tokens, ngram_size_);
    check_and_insert(fp);
}

void MinHashDedup::clear() {
    std::fill(fingerprints_.begin(), fingerprints_.end(), 0);
    total_checked_ = 0;
    duplicates_found_ = 0;
}

// ============================================================================
// MmapDataLoader — Memory-mapped streaming DataLoader
// ============================================================================

MmapDataLoader::MmapDataLoader(const std::string& file_path,
                               BPETokenizer* tokenizer,
                               const MmapDataLoaderConfig& cfg)
    : cfg_(cfg), tokenizer_(tokenizer), dedup_(cfg.ngram_size, 128, 1000000),
      rng_(static_cast<unsigned int>(cfg.seed)) {
    mmap_file(file_path);
}

MmapDataLoader::~MmapDataLoader() {
    unmmap_file();
}

void MmapDataLoader::mmap_file(const std::string& path) {
#ifdef _WIN32
    file_handle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_handle_ == INVALID_HANDLE_VALUE)
        throw std::runtime_error("MmapDataLoader: Cannot open " + path);

    LARGE_INTEGER file_size;
    GetFileSizeEx(file_handle_, &file_size);
    mapped_size_ = file_size.QuadPart;

    mapping_handle_ = CreateFileMappingA(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping_handle_) {
        CloseHandle(file_handle_);
        throw std::runtime_error("MmapDataLoader: CreateFileMapping failed");
    }

    mapped_data_ = (char*)MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);
    if (!mapped_data_) {
        CloseHandle(mapping_handle_);
        CloseHandle(file_handle_);
        throw std::runtime_error("MmapDataLoader: MapViewOfFile failed");
    }
#else
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
        throw std::runtime_error("MmapDataLoader: Cannot open " + path);

    struct stat st;
    if (fstat(fd_, &st) < 0) {
        close(fd_);
        throw std::runtime_error("MmapDataLoader: fstat failed");
    }
    mapped_size_ = st.st_size;

    mapped_data_ = (char*)mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_data_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("MmapDataLoader: mmap failed");
    }
#endif
}

void MmapDataLoader::unmmap_file() {
#ifdef _WIN32
    if (mapped_data_) UnmapViewOfFile(mapped_data_);
    if (mapping_handle_) CloseHandle(mapping_handle_);
    if (file_handle_ != INVALID_HANDLE_VALUE) CloseHandle(file_handle_);
#else
    if (mapped_data_ && mapped_data_ != MAP_FAILED) munmap(mapped_data_, mapped_size_);
    if (fd_ >= 0) close(fd_);
#endif
    mapped_data_ = nullptr;
}

std::vector<int64_t> MmapDataLoader::tokenize_window(int64_t start, int64_t length) const {
    std::string text(mapped_data_ + start, length);
    auto ids = tokenizer_->encode(text);
    std::vector<int64_t> tokens(ids.begin(), ids.end());
    return tokens;
}

void MmapDataLoader::refill_buffer() {
    if (file_offset_ >= mapped_size_) return;

    int64_t window_size = std::min(cfg_.mmap_window, mapped_size_ - file_offset_);
    auto tokens = tokenize_window(file_offset_, window_size);
    file_offset_ += window_size;

    // MinHash deduplication
    if (cfg_.dedup_enabled && tokens.size() >= static_cast<size_t>(cfg_.ngram_size)) {
        std::vector<int64_t> filtered;
        filtered.reserve(tokens.size());
        for (size_t i = 0; i + cfg_.ngram_size <= tokens.size(); ++i) {
            if (!dedup_.is_duplicate(tokens.data() + i, cfg_.ngram_size)) {
                filtered.push_back(tokens[i]);
            }
        }
        // Add remaining tokens
        for (size_t i = std::max((int64_t)0, (int64_t)tokens.size() - cfg_.ngram_size + 1);
             i < tokens.size(); ++i) {
            filtered.push_back(tokens[i]);
        }
        tokens = std::move(filtered);
    }

    for (auto t : tokens) {
        token_buffer_.push_back(t);
        total_tokens_++;
    }
}

bool MmapDataLoader::has_next() const {
    int64_t needed = cfg_.batch_size * cfg_.seq_length + 1;
    return (int64_t)token_buffer_.size() >= needed || file_offset_ < mapped_size_;
}

int64_t MmapDataLoader::num_batches() const {
    return total_tokens_ / (cfg_.batch_size * cfg_.seq_length);
}

void MmapDataLoader::reset() {
    file_offset_ = 0;
    token_buffer_.clear();
    total_tokens_ = 0;
    dedup_.clear();
}

Tensor MmapDataLoader::next_batch() {
    int64_t needed = cfg_.batch_size * cfg_.seq_length;

    while ((int64_t)token_buffer_.size() < needed) {
        refill_buffer();
        if (file_offset_ >= mapped_size_ && token_buffer_.size() < static_cast<size_t>(needed))
            break;
    }

    Tensor batch(Shape{cfg_.batch_size, cfg_.seq_length}, DType::I64);
    batch.zero_();
    int64_t* data = batch.data<int64_t>();

    int64_t filled = std::min((int64_t)token_buffer_.size(), needed);
    for (int64_t i = 0; i < filled; ++i) {
        data[i] = token_buffer_.front();
        token_buffer_.pop_front();
    }

    return batch;
}

std::pair<Tensor, Tensor> MmapDataLoader::next_batch_with_labels() {
    int64_t needed = cfg_.batch_size * (cfg_.seq_length + 1);

    while ((int64_t)token_buffer_.size() < needed) {
        refill_buffer();
        if (file_offset_ >= mapped_size_ && token_buffer_.size() < static_cast<size_t>(needed))
            break;
    }

    Tensor input(Shape{cfg_.batch_size, cfg_.seq_length}, DType::I64);
    Tensor target(Shape{cfg_.batch_size, cfg_.seq_length}, DType::I64);
    input.zero_();
    target.zero_();

    int64_t* inp_data = input.data<int64_t>();
    int64_t* tgt_data = target.data<int64_t>();

    for (int64_t s = 0; s < cfg_.batch_size; ++s) {
        for (int64_t j = 0; j < cfg_.seq_length; ++j) {
            if (token_buffer_.empty()) break;
            int64_t tok = token_buffer_.front();
            token_buffer_.pop_front();
            inp_data[s * cfg_.seq_length + j] = tok;
            if (!token_buffer_.empty()) {
                tgt_data[s * cfg_.seq_length + j] = token_buffer_.front();
            }
        }
    }

    return {input, target};
}

} // namespace quant
