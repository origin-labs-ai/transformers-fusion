#pragma once
#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/codebook.h"
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <unordered_map>

namespace quant {

class MappedFile;

#pragma pack(push, 1)
struct QUANTHeader {
    char magic[4];      // "QUA1"
    uint32_t version;   // packed: major<<22 | minor<<12 | patch
    uint32_t flags;
    uint32_t config_size;
};

struct FormatBlockEntry {
    uint32_t block_id;
    uint8_t format;    // Format enum value (0..14, see quant/types.h Format)
    uint32_t cb_bytes; // codebook size in bytes
};

struct TensorEntry {
    uint16_t name_len;
    uint32_t block_start;
    uint32_t num_blocks;
};
#pragma pack(pop)

struct BlockData {
    Format format;
    std::vector<uint8_t> codebook;
    std::vector<uint8_t> indices;
    uint32_t num_weights;
};

class QUANTWriter {
public:
    explicit QUANTWriter(const std::string& path);
    ~QUANTWriter();
    
    void write_header(const QUANTHeader& hdr, const uint8_t* config_data);
    void write_format_table(const std::vector<FormatBlockEntry>& entries);
    void write_block(const BlockData& block);
    void write_raw(const char* data, size_t size);
    // Content-addressed dedup write: returns offset, skips duplicate blobs
    size_t write_dedup(const uint8_t* data, size_t size);
    void write_tensor_table(const std::vector<TensorEntry>& entries,
                            const std::vector<std::string>& names);
    void close();
    
private:
    std::ofstream file_;
    size_t data_start_;
    struct BlobIndex { size_t offset; size_t size; };
    std::unordered_map<std::string, BlobIndex> blob_index_; // hex_sha256 -> {offset, size}
};

class QUANTReader {
public:
    explicit QUANTReader(const std::string& path);
    ~QUANTReader();
    QUANTReader(const QUANTReader&) = delete;
    QUANTReader& operator=(const QUANTReader&) = delete;
    QUANTReader(QUANTReader&&) noexcept = default;
    QUANTReader& operator=(QUANTReader&&) noexcept = default;
    
    const QUANTHeader& header() const;
    std::vector<uint8_t> read_config() const;
    std::vector<FormatBlockEntry> read_format_table() const;
    BlockData read_block(uint32_t block_id) const;
    size_t num_blocks() const;
    
    // Read a named tensor's blocks and reconstruct
    Tensor read_tensor(const std::string& name) const;
    std::vector<std::string> tensor_names() const;
    
    // Random-access block reader using the prebuilt offset index (O(1)).
    const uint8_t* block_ptr(uint32_t block_id) const;
    size_t block_offset(uint32_t block_id) const { return block_offsets_[block_id]; }
    const FormatBlockEntry& format_entry(uint32_t block_id) const { return cached_ft_[block_id]; }
    bool tensor_blocks(const std::string& name, uint32_t& start, uint32_t& count) const;
    
    // Check if file was successfully opened AND fully validated.
    // P0 fix: old valid() returned true whenever data_ != nullptr, so a
    // corrupt-magic file that bailed mid-parse still looked valid. valid_
    // flips true only after the whole header walk succeeds.
    bool valid() const { return valid_; }

    // Get format info for a tensor
    std::vector<Format> tensor_formats(const std::string& name) const;

private:
    std::unique_ptr<MappedFile> mapped_file_;
    const uint8_t* data_;
    size_t file_size_;
    QUANTHeader header_;
    size_t format_table_offset_;
    size_t tensor_table_offset_;
    size_t data_offset_;
    uint32_t num_format_blocks_;
    uint32_t num_tensors_;
    mutable std::vector<FormatBlockEntry> cached_ft_;
    std::vector<size_t> block_offsets_;   // per-block byte offset from file start
    bool valid_ = false; // P0: set true only after full header walk succeeds
};

// ===========================================================================
// QUANT Idx — SHA256 integrity-checked index file format
// Header: magic "TranscenderIDX" | version | num_tensors
// Then for each tensor: name_len | name bytes | sha256(name) [32 bytes]
// On read, each tensor name is re-hashed and compared fail-fast; the first
// corrupt name is reported by name.
// ===========================================================================

struct SHA256Hash {
    uint8_t bytes[32];
};

struct IdxTensorEntry {
    std::string name;
    SHA256Hash name_hash;   // sha256(name) stored in file
};

class QUANTIdxWriter {
public:
    explicit QUANTIdxWriter(const std::string& path);
    ~QUANTIdxWriter();

    // Writes the full idx file: header magic "TranscenderIDX", version,
    // num_tensors, then per-tensor name + computed sha256(name).
    void write_idx(uint32_t version, const std::vector<std::string>& tensor_names);

    void close();

private:
    std::ofstream file_;
};

class QUANTIdxReader {
public:
    explicit QUANTIdxReader(const std::string& path);
    ~QUANTIdxReader();

    // Reads the idx file and recomputes sha256 for every stored tensor name.
    // On the first mismatch throws quant::Error naming the corrupt tensor.
    // Returns the verified list of tensor names on success.
    std::vector<std::string> read_idx();

    bool valid() const { return data_ != nullptr; }
    uint32_t version() const { return version_; }
    uint32_t num_tensors() const { return num_tensors_; }

private:
    MappedFile* mapped_file_;
    const uint8_t* data_;
    size_t file_size_;
    uint32_t version_;
    uint32_t num_tensors_;
    bool checked_;
    size_t magic_size_ = 15; // 15 = TranscenderIDX, 10 = legacy InNovaIDX
    std::vector<std::string> names_;
};

} // namespace quant
