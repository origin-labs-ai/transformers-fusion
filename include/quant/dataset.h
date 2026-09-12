#pragma once
#include "quant/tensor.h"
#include "quant/tokenizer.h"
#include <string>
#include <vector>
#include <deque>
#include <random>
#include <memory>
#include <functional>
#include <future>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace quant {

class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;
    virtual std::pair<Tensor, Tensor> get(size_t index) = 0;
};

class InMemoryDataset : public Dataset {
public:
    InMemoryDataset();

    size_t size() const override { return inputs_.size(); }
    std::pair<Tensor, Tensor> get(size_t index) override;

    void add_sample(const Tensor& input, const Tensor& target);
    void clear();

    static std::unique_ptr<InMemoryDataset> from_text_file(
        const std::string& path, Tokenizer* tok, int64_t seq_len,
        int64_t stride = -1, int64_t max_samples = -1);

    static std::unique_ptr<InMemoryDataset> from_tokenized_file(
        const std::string& path, int64_t seq_len,
        int64_t max_samples = -1);

    static std::unique_ptr<InMemoryDataset> from_directory(
        const std::string& dir, const std::string& ext,
        Tokenizer* tok, int64_t seq_len);

    const std::vector<Tensor>& inputs() const { return inputs_; }
    const std::vector<Tensor>& targets() const { return targets_; }

private:
    // BUGFIX (bug census): concurrent train/loader threads raced on these
    // vectors. All access serialized on mutex_.
    mutable std::mutex mutex_;
    std::vector<Tensor> inputs_;
    std::vector<Tensor> targets_;
};

class StreamingDataset : public Dataset {
public:
    StreamingDataset(const std::string& pattern, int64_t seq_len,
                     int64_t buffer_size = 10000, int64_t seed = 42);
    ~StreamingDataset() override;

    size_t size() const override { return est_size_; }
    std::pair<Tensor, Tensor> get(size_t index) override;

    bool has_next();
    void shuffle();
    void reset();
    std::vector<int64_t> next_chunk(int64_t chunk_size);

    int64_t num_chunks() const { return est_size_ / std::max((int64_t)1, seq_len_); }

private:
    struct Shard {
        std::string path;
        FILE* file = nullptr;
        bool eof = false;
    };

    std::vector<Shard> shards_;
    int64_t seq_len_;
    int64_t buffer_size_;
    int64_t est_size_;
    std::mt19937 rng_;
    std::deque<int64_t> buffer_;
    int current_shard_;
    std::mutex mutex_;
    std::thread prefetch_thread_;
    std::atomic<bool> prefetch_active_;
    std::deque<int64_t> prefetch_buffer_;
    std::mutex prefetch_mutex_;
    std::condition_variable prefetch_cv_;

    void refill();
    // Internal: caller MUST hold mutex_ (get() holds it; re-locking the
    // non-recursive mutex would self-deadlock — same split as kv_cache).
    void refill_locked();
    void prefetch_worker();
    void open_shard(size_t idx);
    void close_shard(size_t idx);
    int64_t read_tokens_from_shard(size_t idx, std::vector<int64_t>& out, int64_t max_count);
};

class ChunkedDataset : public Dataset {
public:
    ChunkedDataset(const std::string& path, int64_t chunk_size,
                   int64_t overlap = 0, int64_t seq_len = 2048);

    size_t size() const override;
    std::pair<Tensor, Tensor> get(size_t index) override;

    int64_t num_chunks() const { return num_chunks_; }
    void set_overlap(int64_t overlap) { overlap_ = overlap; }

private:
    std::string path_;
    FILE* file_;
    int64_t file_size_;
    int64_t chunk_size_;
    int64_t overlap_;
    int64_t seq_len_;
    int64_t num_chunks_;
    std::vector<int64_t> chunk_offsets_;

    std::vector<int64_t> read_chunk(int64_t chunk_idx) const;
    void build_chunk_index();
};

class FilterDataset : public Dataset {
public:
    using FilterFn = std::function<bool(const Tensor&, const Tensor&)>;

    FilterDataset(std::shared_ptr<Dataset> base, FilterFn fn);

    size_t size() const override { return indices_.size(); }
    std::pair<Tensor, Tensor> get(size_t index) override;

    void apply_filter();
    std::vector<size_t> filtered_indices() const { return indices_; }

private:
    std::shared_ptr<Dataset> base_;
    FilterFn fn_;
    std::vector<size_t> indices_;
    bool filtered_;
};

class MapDataset : public Dataset {
public:
    using MapFn = std::function<std::pair<Tensor, Tensor>(const Tensor&, const Tensor&)>;

    MapDataset(std::shared_ptr<Dataset> base, MapFn fn);

    size_t size() const override { return base_->size(); }
    std::pair<Tensor, Tensor> get(size_t index) override;

    void set_map_fn(MapFn fn) { fn_ = fn; }

private:
    std::shared_ptr<Dataset> base_;
    MapFn fn_;
};

class ConcatDataset : public Dataset {
public:
    void add_dataset(std::shared_ptr<Dataset> ds);

    size_t size() const override;
    std::pair<Tensor, Tensor> get(size_t index) override;

    size_t num_datasets() const { return datasets_.size(); }

private:
    std::vector<std::shared_ptr<Dataset>> datasets_;
    std::vector<size_t> cumulative_sizes_;
    void rebuild_cumulative();
};

class ShuffleDataset : public Dataset {
public:
    ShuffleDataset(std::shared_ptr<Dataset> base, size_t buffer_size,
                   unsigned int seed = 42);

    size_t size() const override { return base_->size(); }
    std::pair<Tensor, Tensor> get(size_t index) override;

    void reshuffle();
    std::vector<size_t> permutation() const { return perm_; }

private:
    std::shared_ptr<Dataset> base_;
    std::vector<size_t> perm_;
    size_t buffer_size_;
    std::mt19937 rng_;
};

struct DatasetSplit {
    std::shared_ptr<Dataset> train;
    std::shared_ptr<Dataset> val;
    std::shared_ptr<Dataset> test;
};

DatasetSplit split_dataset(std::shared_ptr<Dataset> base,
                           double train_ratio, double val_ratio,
                           unsigned int seed = 42);

} // namespace quant
