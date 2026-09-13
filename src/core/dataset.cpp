#include "quant/dataset.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <numeric>
#include <cstring>

namespace quant {

// ==================== InMemoryDataset ====================

InMemoryDataset::InMemoryDataset() {}

std::pair<Tensor, Tensor> InMemoryDataset::get(size_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (index >= inputs_.size()) {
        throw std::out_of_range("InMemoryDataset::get: index " + std::to_string(index) +
                                " out of range (size " + std::to_string(inputs_.size()) + ")");
    }
    Tensor input = inputs_[index];
    Tensor target;
    if (index < targets_.size()) {
        target = targets_[index];
    } else {
        target = input.clone();
    }
    return {input, target};
}

void InMemoryDataset::add_sample(const Tensor& input, const Tensor& target) {
    std::lock_guard<std::mutex> lock(mutex_);
    inputs_.push_back(input.clone());
    targets_.push_back(target.clone());
}

void InMemoryDataset::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    inputs_.clear();
    targets_.clear();
}

std::unique_ptr<InMemoryDataset> InMemoryDataset::from_text_file(
    const std::string& path, Tokenizer* tok, int64_t seq_len,
    int64_t stride, int64_t max_samples) {
    auto dataset = std::make_unique<InMemoryDataset>();
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    if (stride < 0) stride = seq_len;

    std::vector<int> all_tokens;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (tok) {
            auto ids = tok->encode(line);
            all_tokens.insert(all_tokens.end(), ids.begin(), ids.end());
        } else {
            for (char c : line) all_tokens.push_back((unsigned char)c);
        }
    }
    file.close();

    if (all_tokens.empty()) {
        throw std::runtime_error("No tokens found in file: " + path);
    }

    int64_t samples_added = 0;
    size_t pos = 0;
    while (pos + (size_t)seq_len <= all_tokens.size()) {
        std::vector<int64_t> input_ids(seq_len);
        std::vector<int64_t> target_ids(seq_len);
        for (int64_t j = 0; j < seq_len; j++) {
            input_ids[j] = all_tokens[pos + j];
            if (pos + j + 1 < all_tokens.size()) {
                target_ids[j] = all_tokens[pos + j + 1];
            } else {
                target_ids[j] = all_tokens[pos + j];
            }
        }

        Tensor input_tensor(Shape{1, seq_len}, DType::I64);
        Tensor target_tensor(Shape{1, seq_len}, DType::I64);
        std::memcpy(input_tensor.data<int64_t>(), input_ids.data(), seq_len * sizeof(int64_t));
        std::memcpy(target_tensor.data<int64_t>(), target_ids.data(), seq_len * sizeof(int64_t));
        dataset->add_sample(input_tensor, target_tensor);
        samples_added++;

        if (max_samples > 0 && samples_added >= max_samples) break;
        pos += stride;
    }

    return dataset;
}

std::unique_ptr<InMemoryDataset> InMemoryDataset::from_tokenized_file(
    const std::string& path, int64_t seq_len, int64_t max_samples) {
    auto dataset = std::make_unique<InMemoryDataset>();
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open tokenized file: " + path);
    }

    file.seekg(0, std::ios::end);
    int64_t file_size = (int64_t)file.tellg();
    file.seekg(0, std::ios::beg);

    int64_t num_tokens = file_size / sizeof(int64_t);
    int64_t samples_count = num_tokens / seq_len;
    if (max_samples > 0 && samples_count > max_samples) samples_count = max_samples;

    for (int64_t i = 0; i < samples_count; i++) {
        std::vector<int64_t> tokens(seq_len + 1);
        file.read((char*)tokens.data(), (seq_len + 1) * sizeof(int64_t));
        if ((size_t)file.gcount() < (size_t)((seq_len + 1) * sizeof(int64_t))) break;

        Tensor input(Shape{1, seq_len}, DType::I64);
        Tensor target(Shape{1, seq_len}, DType::I64);
        int64_t* inp_data = input.data<int64_t>();
        int64_t* tgt_data = target.data<int64_t>();
        for (int64_t j = 0; j < seq_len; j++) {
            inp_data[j] = tokens[j];
            tgt_data[j] = tokens[j + 1];
        }
        dataset->add_sample(input, target);
    }

    return dataset;
}

std::unique_ptr<InMemoryDataset> InMemoryDataset::from_directory(
    const std::string& dir, const std::string& ext,
    Tokenizer* tok, int64_t seq_len) {
    auto dataset = std::make_unique<InMemoryDataset>();

    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string path = entry.path().string();
        if (!ext.empty() && path.size() >= ext.size() &&
            path.substr(path.size() - ext.size()) != ext) continue;

        try {
            auto partial = from_text_file(path, tok, seq_len, -1, -1);
            for (size_t i = 0; i < partial->size(); i++) {
                auto sample = partial->get(i);
                dataset->add_sample(sample.first, sample.second);
            }
        } catch (const std::exception& e) {
            // NOTE (bug census): swallow-and-continue is INTENTIONAL here
            // (one bad file must not kill a directory ingest), but the old
            // (void)e hid which file failed. Log it.
            std::fprintf(stderr, "[dataset] skipping unreadable file: %s\n", e.what());
            continue;
        }
    }

    return dataset;
}

// ==================== StreamingDataset ====================

StreamingDataset::StreamingDataset(const std::string& pattern, int64_t seq_len,
                                    int64_t buffer_size, int64_t seed)
    : seq_len_(seq_len), buffer_size_(buffer_size), est_size_(1000000),
      rng_((unsigned int)seed), current_shard_(0),
      prefetch_active_(false) {
    std::string base = pattern;
    size_t asterisk = pattern.find('*');
    if (asterisk != std::string::npos) {
        // NOTE (bug census): dir-part computation was dead ((void)dir) —
        // shard open below uses pattern+".shard.N" literally. Documented:
        // glob expansion is future work; '*' patterns currently resolve to
        // the literal pattern path (callers: pass concrete prefixes).
        std::string dir_part = pattern.substr(0, asterisk);
        size_t slash_pos = dir_part.rfind('/');
        std::string dir;
        if (slash_pos != std::string::npos) {
            dir = pattern.substr(0, slash_pos);
        } else {
            dir = ".";
        }
        (void)dir; // documented-unused: glob expansion not implemented yet
    }

    for (int i = 0; i < 8; i++) {
        Shard shard;
        shard.path = pattern + ".shard." + std::to_string(i);
        shard.file = fopen(shard.path.c_str(), "rb");
        shard.eof = (shard.file == nullptr);
        shards_.push_back(shard);
    }

    if (shards_.empty()) {
        Shard shard;
        shard.path = pattern;
        shard.file = fopen(shard.path.c_str(), "rb");
        shard.eof = (shard.file == nullptr);
        shards_.push_back(shard);
    }

    if (!shards_.empty() && shards_[0].file) {
        est_size_ = 1000000;
        refill();
    }
}

StreamingDataset::~StreamingDataset() {
    prefetch_active_ = false;
    if (prefetch_thread_.joinable()) prefetch_thread_.join();
    for (auto& shard : shards_) {
        if (shard.file) fclose(shard.file);
    }
}

void StreamingDataset::open_shard(size_t idx) {
    if (idx >= shards_.size()) return;
    if (shards_[idx].file) return;
    shards_[idx].file = fopen(shards_[idx].path.c_str(), "rb");
    shards_[idx].eof = (shards_[idx].file == nullptr);
}

void StreamingDataset::close_shard(size_t idx) {
    if (idx >= shards_.size()) return;
    if (shards_[idx].file) {
        fclose(shards_[idx].file);
        shards_[idx].file = nullptr;
    }
    shards_[idx].eof = true;
}

int64_t StreamingDataset::read_tokens_from_shard(size_t idx, std::vector<int64_t>& out, int64_t max_count) {
    if (idx >= shards_.size() || !shards_[idx].file || shards_[idx].eof) return 0;
    FILE* f = shards_[idx].file;
    int64_t count = 0;
    char buf[65536];
    bool in_token = false;
    bool negative = false;
    int64_t accum = 0;

    while (count < max_count) {
        size_t nread = fread(buf, 1, sizeof(buf), f);
        if (nread == 0) {
            shards_[idx].eof = true;
            break;
        }

        for (size_t i = 0; i < nread; i++) {
            char c = buf[i];
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
                if (in_token) {
                    out.push_back(negative ? -accum : accum);
                    count++;
                    in_token = false;
                    accum = 0;
                    negative = false;
                }
            } else if (c == '-' && !in_token) {
                negative = true;
                in_token = true;
            } else if (c >= '0' && c <= '9') {
                in_token = true;
                accum = accum * 10 + (c - '0');
            } else {
                if (in_token) {
                    out.push_back(negative ? -accum : accum);
                    count++;
                    in_token = false;
                    accum = 0;
                    negative = false;
                }
                out.push_back((int64_t)(unsigned char)c);
                count++;
            }
        }
    }

    if (in_token) {
        out.push_back(negative ? -accum : accum);
        count++;
    }

    return count;
}

void StreamingDataset::refill() {
    std::lock_guard<std::mutex> lock(mutex_);
    refill_locked();
}

// Internal: caller MUST hold mutex_ (get() holds it across the drain loop;
// re-locking the non-recursive mutex would self-deadlock).
void StreamingDataset::refill_locked() {

    std::vector<int64_t> tokens;
    int64_t needed = buffer_size_;

    if (!shards_.empty() && shards_[current_shard_].file && !shards_[current_shard_].eof) {
        int64_t got = read_tokens_from_shard(current_shard_, tokens, needed);
        if (got < needed) {
            current_shard_ = (current_shard_ + 1) % (int)shards_.size();
            open_shard(current_shard_);
            if (shards_[current_shard_].file && !shards_[current_shard_].eof) {
                read_tokens_from_shard(current_shard_, tokens, needed - got);
            }
        }
    }

    for (int64_t t : tokens) {
        buffer_.push_back(t);
    }
}

void StreamingDataset::prefetch_worker() {
    while (prefetch_active_) {
        std::unique_lock<std::mutex> lock(prefetch_mutex_);
        prefetch_cv_.wait_for(lock, std::chrono::milliseconds(100),
                              [this]() { return (int64_t)prefetch_buffer_.size() < buffer_size_ / 2; });
        if (!prefetch_active_) break;
        lock.unlock();

        std::vector<int64_t> tokens;
        {
            std::lock_guard<std::mutex> shard_lock(mutex_);
            if (!shards_.empty() && shards_[current_shard_].file && !shards_[current_shard_].eof) {
                read_tokens_from_shard(current_shard_, tokens, buffer_size_ / 4);
                if (tokens.empty()) {
                    current_shard_ = (current_shard_ + 1) % (int)shards_.size();
                    open_shard(current_shard_);
                }
            }
        }

        if (!tokens.empty()) {
            std::lock_guard<std::mutex> pb_lock(prefetch_mutex_);
            for (int64_t t : tokens) prefetch_buffer_.push_back(t);
            prefetch_cv_.notify_one();
        }
    }
}

bool StreamingDataset::has_next() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!buffer_.empty()) return true;

    if (!prefetch_active_ && prefetch_thread_.get_id() == std::thread::id()) {
        prefetch_active_ = true;
        prefetch_thread_ = std::thread(&StreamingDataset::prefetch_worker, this);
    }

    {
        std::lock_guard<std::mutex> pb_lock(prefetch_mutex_);
        while (!prefetch_buffer_.empty()) {
            buffer_.push_back(prefetch_buffer_.front());
            prefetch_buffer_.pop_front();
        }
    }

    return !buffer_.empty();
}

std::pair<Tensor, Tensor> StreamingDataset::get(size_t index) {
    // BUGFIX (bug census): touched buffer_/shards_/current_shard_ with no
    // mutex_ while prefetch_worker mutates shards (race/lost tokens), and
    // buffer_ paths were inconsistently locked for concurrent consumers.
    // Whole body serialized on mutex_ now (prefetch handoff still via
    // prefetch_mutex_ inside the same critical section — fixed lock order:
    // mutex_ before prefetch_mutex_, matching has_next/shuffle/reset).
    std::lock_guard<std::mutex> lock(mutex_);
    // NOTE (bug census): `index` intentionally unused — streaming datasets
    // serve FIFO from the buffer, not random access (matches base API).
    (void)index; // documented-unused: FIFO streaming, no random access
    int64_t needed = seq_len_ + 1;

    while ((int64_t)buffer_.size() < needed) {
        if (!prefetch_active_ && prefetch_thread_.get_id() == std::thread::id()) {
            prefetch_active_ = true;
            prefetch_thread_ = std::thread(&StreamingDataset::prefetch_worker, this);
        }
        {
            std::lock_guard<std::mutex> pb_lock(prefetch_mutex_);
            while (!prefetch_buffer_.empty()) {
                buffer_.push_back(prefetch_buffer_.front());
                prefetch_buffer_.pop_front();
            }
        }
        if ((int64_t)buffer_.size() < needed) {
            refill_locked();
        }
        if ((int64_t)buffer_.size() < needed) break;
    }

    int64_t actual = std::min((int64_t)buffer_.size(), needed);
    std::vector<int64_t> tokens;
    for (int64_t i = 0; i < actual; i++) {
        tokens.push_back(buffer_.front());
        buffer_.pop_front();
    }

    Tensor input(Shape{1, seq_len_}, DType::I64);
    Tensor target(Shape{1, seq_len_}, DType::I64);
    input.zero_();
    target.zero_();

    int64_t* inp_data = input.data<int64_t>();
    int64_t* tgt_data = target.data<int64_t>();

    for (int64_t i = 0; i < seq_len_ && i < (int64_t)tokens.size() - 1; i++) {
        inp_data[i] = tokens[i];
        tgt_data[i] = tokens[i + 1];
    }

    return {input, target};
}

void StreamingDataset::shuffle() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_.empty()) return;
    std::vector<int64_t> vec(buffer_.begin(), buffer_.end());
    std::shuffle(vec.begin(), vec.end(), rng_);
    buffer_.assign(vec.begin(), vec.end());
}

void StreamingDataset::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.clear();
    current_shard_ = 0;
    for (auto& shard : shards_) {
        if (shard.file) {
            fclose(shard.file);
            shard.file = nullptr;
        }
    }
    for (auto& shard : shards_) {
        shard.file = fopen(shard.path.c_str(), "rb");
        shard.eof = (shard.file == nullptr);
    }
    {
        std::lock_guard<std::mutex> pb_lock(prefetch_mutex_);
        prefetch_buffer_.clear();
    }
    refill();
}

// ==================== ChunkedDataset ====================

ChunkedDataset::ChunkedDataset(const std::string& path, int64_t chunk_size,
                                int64_t overlap, int64_t seq_len)
    : path_(path), file_(nullptr), file_size_(0),
      chunk_size_(chunk_size), overlap_(overlap),
      seq_len_(seq_len), num_chunks_(0) {
    file_ = fopen(path.c_str(), "rb");
    if (!file_) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    fseek(file_, 0, SEEK_END);
    file_size_ = ftell(file_);
    fseek(file_, 0, SEEK_SET);
    build_chunk_index();
}

void ChunkedDataset::build_chunk_index() {
    num_chunks_ = 0;
    chunk_offsets_.clear();
    int64_t pos = 0;
    int64_t stride = chunk_size_ - overlap_;
    if (stride <= 0) stride = chunk_size_ / 2;
    if (stride <= 0) stride = 1;

    while (pos < file_size_) {
        chunk_offsets_.push_back(pos);
        num_chunks_++;
        pos += stride;
    }
}

size_t ChunkedDataset::size() const {
    return (size_t)num_chunks_;
}

std::vector<int64_t> ChunkedDataset::read_chunk(int64_t chunk_idx) const {
    if (chunk_idx < 0 || chunk_idx >= num_chunks_) return {};

    int64_t offset = chunk_offsets_[chunk_idx];
    int64_t to_read = std::min(chunk_size_, file_size_ - offset);
    if (to_read <= 0) return {};

    FILE* f = fopen(path_.c_str(), "rb");
    if (!f) return {};
    fseek(f, static_cast<long>(offset), SEEK_SET);

    std::vector<int64_t> result;
    std::vector<char> raw(to_read);
    size_t nread = fread(raw.data(), 1, to_read, f);
    fclose(f);

    if (nread == 0) return result;

    int64_t accum = 0;
    bool in_token = false;
    bool negative = false;
    for (size_t i = 0; i < nread; i++) {
        char c = raw[i];
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
            if (in_token) {
                result.push_back(negative ? -accum : accum);
                in_token = false;
                accum = 0;
                negative = false;
            }
        } else if (c == '-' && !in_token) {
            negative = true;
            in_token = true;
        } else if (c >= '0' && c <= '9') {
            in_token = true;
            accum = accum * 10 + (c - '0');
        } else {
            if (in_token) {
                result.push_back(negative ? -accum : accum);
                in_token = false;
                accum = 0;
                negative = false;
            }
            result.push_back((int64_t)(unsigned char)c);
        }
    }
    if (in_token) {
        result.push_back(negative ? -accum : accum);
    }

    return result;
}

std::pair<Tensor, Tensor> ChunkedDataset::get(size_t index) {
    auto tokens = read_chunk((int64_t)index);
    if (tokens.empty()) {
        return {Tensor(Shape{1, seq_len_}, DType::I64), Tensor(Shape{1, seq_len_}, DType::I64)};
    }

    int64_t actual = std::min((int64_t)tokens.size(), seq_len_ + 1);
    Tensor input(Shape{1, seq_len_}, DType::I64);
    Tensor target(Shape{1, seq_len_}, DType::I64);
    input.zero_();
    target.zero_();
    int64_t* inp = input.data<int64_t>();
    int64_t* tgt = target.data<int64_t>();

    for (int64_t i = 0; i < actual - 1; i++) {
        inp[i] = tokens[i];
        tgt[i] = tokens[i + 1];
    }

    return {input, target};
}

// ==================== FilterDataset ====================

FilterDataset::FilterDataset(std::shared_ptr<Dataset> base, FilterFn fn)
    : base_(base), fn_(fn), filtered_(false) {
    apply_filter();
}

void FilterDataset::apply_filter() {
    indices_.clear();
    for (size_t i = 0; i < base_->size(); i++) {
        auto sample = base_->get(i);
        if (fn_(sample.first, sample.second)) {
            indices_.push_back(i);
        }
    }
    filtered_ = true;
}

std::pair<Tensor, Tensor> FilterDataset::get(size_t index) {
    if (!filtered_) apply_filter();
    if (index >= indices_.size()) {
        throw std::out_of_range("FilterDataset::get: index out of range");
    }
    return base_->get(indices_[index]);
}

// ==================== MapDataset ====================

MapDataset::MapDataset(std::shared_ptr<Dataset> base, MapFn fn)
    : base_(base), fn_(fn) {}

std::pair<Tensor, Tensor> MapDataset::get(size_t index) {
    auto sample = base_->get(index);
    if (fn_) return fn_(sample.first, sample.second);
    return sample;
}

// ==================== ConcatDataset ====================

void ConcatDataset::add_dataset(std::shared_ptr<Dataset> ds) {
    datasets_.push_back(ds);
    rebuild_cumulative();
}

void ConcatDataset::rebuild_cumulative() {
    cumulative_sizes_.clear();
    size_t cum = 0;
    for (const auto& ds : datasets_) {
        cum += ds->size();
        cumulative_sizes_.push_back(cum);
    }
}

size_t ConcatDataset::size() const {
    if (cumulative_sizes_.empty()) return 0;
    return cumulative_sizes_.back();
}

std::pair<Tensor, Tensor> ConcatDataset::get(size_t index) {
    for (size_t i = 0; i < cumulative_sizes_.size(); i++) {
        if (index < cumulative_sizes_[i]) {
            size_t prev = (i == 0) ? 0 : cumulative_sizes_[i - 1];
            return datasets_[i]->get(index - prev);
        }
    }
    throw std::out_of_range("ConcatDataset::get: index out of range");
}

// ==================== ShuffleDataset ====================

ShuffleDataset::ShuffleDataset(std::shared_ptr<Dataset> base, size_t buffer_size, unsigned int seed)
    : base_(base), buffer_size_(buffer_size), rng_(seed) {
    reshuffle();
}

void ShuffleDataset::reshuffle() {
    size_t n = base_->size();
    perm_.resize(n);
    std::iota(perm_.begin(), perm_.end(), 0);
    std::shuffle(perm_.begin(), perm_.end(), rng_);
}

std::pair<Tensor, Tensor> ShuffleDataset::get(size_t index) {
    if (index >= perm_.size()) {
        throw std::out_of_range("ShuffleDataset::get: index out of range");
    }
    return base_->get(perm_[index]);
}

// ==================== SplitDataset ====================

DatasetSplit split_dataset(std::shared_ptr<Dataset> base,
                            double train_ratio, double val_ratio,
                            unsigned int seed) {
    size_t n = base->size();
    size_t train_size = (size_t)(n * train_ratio);
    size_t val_size = (size_t)(n * val_ratio);
    size_t test_size = n - train_size - val_size;

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    auto make_split = [&](size_t start, size_t count) -> std::shared_ptr<Dataset> {
        struct SliceDataset : public Dataset {
            std::shared_ptr<Dataset> base;
            std::vector<size_t> idx;
            SliceDataset(std::shared_ptr<Dataset> b, std::vector<size_t> i)
                : base(b), idx(std::move(i)) {}
            size_t size() const override { return idx.size(); }
            std::pair<Tensor, Tensor> get(size_t index) override {
                return base->get(idx[index]);
            }
        };
        std::vector<size_t> slice(indices.begin() + start, indices.begin() + start + count);
        return std::make_shared<SliceDataset>(base, slice);
    };

    DatasetSplit split;
    split.train = make_split(0, train_size);
    split.val = make_split(train_size, val_size);
    split.test = make_split(train_size + val_size, test_size);
    return split;
}

} // namespace quant
