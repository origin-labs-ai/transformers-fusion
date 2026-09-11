#include "quant/quant_format.h"
#include "quant/block_codec.h"
#include <cstring>
#include <cstdio>
#include <vector>
#include <string>
#include <new>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace quant {

// ===========================================================================
// SHA-256 — pure C++, no external dependencies (FIPS 180-4)
// ===========================================================================

namespace {

struct SHA256Ctx {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t buffer[64];
    size_t buflen;
};

inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

const uint32_t SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

void sha256_init(SHA256Ctx& c) {
    static const uint32_t H0[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };
    for (int i = 0; i < 8; i++) c.state[i] = H0[i];
    c.bitlen = 0;
    c.buflen = 0;
}

void sha256_block(SHA256Ctx& c, const uint8_t* block) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i*4] << 24) | ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8) | ((uint32_t)block[i*4+3]);
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=c.state[0],b=c.state[1],cc=c.state[2],d=c.state[3];
    uint32_t e=c.state[4],f=c.state[5],g=c.state[6],h=c.state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + SHA256_K[i] + w[i];
        uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        uint32_t maj = (a & b) ^ (a & cc) ^ (b & cc);
        uint32_t t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=cc; cc=b; b=a; a=t1+t2;
    }
    c.state[0]+=a; c.state[1]+=b; c.state[2]+=cc; c.state[3]+=d;
    c.state[4]+=e; c.state[5]+=f; c.state[6]+=g; c.state[7]+=h;
}

void sha256_update(SHA256Ctx& c, const uint8_t* data, size_t len) {
    c.bitlen += (uint64_t)len * 8;
    while (len > 0) {
        size_t take = 64 - c.buflen;
        if (take > len) take = len;
        memcpy(c.buffer + c.buflen, data, take);
        c.buflen += take; data += take; len -= take;
        if (c.buflen == 64) { sha256_block(c, c.buffer); c.buflen = 0; }
    }
}

void sha256_final(SHA256Ctx& c, uint8_t out[32]) {
    c.buffer[c.buflen++] = 0x80;
    if (c.buflen > 56) {
        while (c.buflen < 64) c.buffer[c.buflen++] = 0;
        sha256_block(c, c.buffer);
        c.buflen = 0;
    }
    while (c.buflen < 56) c.buffer[c.buflen++] = 0;
    for (int i = 7; i >= 0; i--) c.buffer[c.buflen++] = (uint8_t)(c.bitlen >> (i*8));
    sha256_block(c, c.buffer);
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(c.state[i] >> 24);
        out[i*4+1] = (uint8_t)(c.state[i] >> 16);
        out[i*4+2] = (uint8_t)(c.state[i] >> 8);
        out[i*4+3] = (uint8_t)(c.state[i]);
    }
}

// Compute SHA-256 of a byte range into SHA256Hash.
SHA256Hash sha256(const uint8_t* data, size_t len) {
    SHA256Ctx c;
    sha256_init(c);
    sha256_update(c, data, len);
    SHA256Hash h;
    sha256_final(c, h.bytes);
    return h;
}

} // namespace

#if defined(_WIN32)
class MappedFile {
public:
    MappedFile() : hFile(INVALID_HANDLE_VALUE), hMap(nullptr), data(nullptr), size_(0) {}

    ~MappedFile() { close(); }

    bool open(const char* path) {
        hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) return false;

        LARGE_INTEGER li;
        GetFileSizeEx(hFile, &li);
        size_ = (size_t)li.QuadPart;

        hMap = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; return false; }

        data = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!data) { CloseHandle(hMap); hMap = NULL; CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; return false; }

        return true;
    }

    void close() {
        if (data) { UnmapViewOfFile(data); data = nullptr; }
        if (hMap) { CloseHandle(hMap); hMap = NULL; }
        if (hFile != INVALID_HANDLE_VALUE) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; }
    }

    const uint8_t* ptr() const { return (const uint8_t*)data; }
    size_t size() const { return size_; }
    bool is_open() const { return data != nullptr; }

private:
    HANDLE hFile;
    HANDLE hMap;
    void* data;
    size_t size_;
};
#else
class MappedFile {
public:
    MappedFile() : data_(nullptr), size_(0) {}
    ~MappedFile() { delete[] data_; }

    // P0 fix: untrusted-file trust boundary. Old code ignored fseek/ftell
    // errors (ftell -1L became a huge size_t), allocated new[] from that
    // size unchecked, and swallowed fread short-reads while returning true.
    // Cap: no single .quant file may exceed 64 GiB in the mmap shim.
    static constexpr size_t kMaxMapBytes = (size_t)64 << 30;

    bool open(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) return false;
        if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
        long sz = ftell(f);
        if (sz < 0) { fclose(f); return false; }
        if ((size_t)sz > kMaxMapBytes) { fclose(f); return false; }
        if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return false; }
        size_ = (size_t)sz;
        if (size_ == 0) { fclose(f); return false; }
        uint8_t* buf = nullptr;
        try {
            buf = new uint8_t[size_];
        } catch (const std::bad_alloc&) {
            fclose(f);
            return false;
        }
        if (fread(buf, 1, size_, f) != size_) {
            delete[] buf;
            fclose(f);
            size_ = 0;
            return false;
        }
        data_ = buf;
        fclose(f);
        return true;
    }

    const uint8_t* ptr() const { return data_; }
    size_t size() const { return size_; }
    bool is_open() const { return data_ != nullptr; }

private:
    uint8_t* data_;
    size_t size_;
};
#endif

// ===========================================================================
// QUANTWriter
// ===========================================================================

QUANTWriter::QUANTWriter(const std::string& path)
    : file_(path, std::ios::binary), data_start_(0) {}

QUANTWriter::~QUANTWriter() {
    if (file_.is_open()) close();
}

void QUANTWriter::write_header(const QUANTHeader& hdr, const uint8_t* config_data) {
    if (!file_.is_open()) return;
    file_.write(hdr.magic, 4);
    file_.write((const char*)&hdr.version, sizeof(hdr.version));
    file_.write((const char*)&hdr.flags, sizeof(hdr.flags));
    file_.write((const char*)&hdr.config_size, sizeof(hdr.config_size));
    if (config_data && hdr.config_size > 0) {
        file_.write((const char*)config_data, hdr.config_size);
    }
    data_start_ = (size_t)file_.tellp();
}

void QUANTWriter::write_format_table(const std::vector<FormatBlockEntry>& entries) {
    if (!file_.is_open()) return;
    uint32_t num = (uint32_t)entries.size();
    file_.write((const char*)&num, sizeof(num));
    for (const auto& e : entries) {
        file_.write((const char*)&e.block_id, sizeof(e.block_id));
        file_.write((const char*)&e.format, sizeof(e.format));
        file_.write((const char*)&e.cb_bytes, sizeof(e.cb_bytes));
    }
}

void QUANTWriter::write_block(const BlockData& block) {
    if (!file_.is_open()) return;
    uint32_t nw = block.num_weights;
    file_.write((const char*)&nw, sizeof(nw));
    uint32_t cb_bytes = (uint32_t)block.codebook.size();
    file_.write((const char*)&cb_bytes, sizeof(cb_bytes));
    if (cb_bytes > 0) {
        file_.write((const char*)block.codebook.data(), cb_bytes);
    }
    uint32_t idx_bytes = (uint32_t)block.indices.size();
    file_.write((const char*)&idx_bytes, sizeof(idx_bytes));
    if (idx_bytes > 0) {
        file_.write((const char*)block.indices.data(), idx_bytes);
    }
}

void QUANTWriter::write_raw(const char* data, size_t size) {
    if (!file_.is_open() || !data || size == 0) return;
    file_.write(data, (std::streamsize)size);
}

size_t QUANTWriter::write_dedup(const uint8_t* data, size_t size) {
    if (!file_.is_open() || !data || size == 0) return (size_t)-1;
    SHA256Hash h = sha256(data, size);
    static const char hexmap[] = "0123456789abcdef";
    char hex[65];
    for (int i = 0; i < 32; i++) {
        hex[i*2]   = hexmap[(h.bytes[i] >> 4) & 0xf];
        hex[i*2+1] = hexmap[h.bytes[i] & 0xf];
    }
    hex[64] = '\0';
    std::string key(hex);
    auto it = blob_index_.find(key);
    if (it != blob_index_.end()) return it->second.offset;
    size_t offset = (size_t)file_.tellp();
    file_.write((const char*)data, size);
    BlobIndex bi;
    bi.offset = offset;
    bi.size = size;
    blob_index_[key] = bi;
    return offset;
}

void QUANTWriter::write_tensor_table(const std::vector<TensorEntry>& entries,
                                    const std::vector<std::string>& names) {
    if (!file_.is_open()) return;
    uint32_t num = (uint32_t)entries.size();
    file_.write((const char*)&num, sizeof(num));
    for (size_t i = 0; i < entries.size(); i++) {
        const auto& e = entries[i];
        uint16_t name_len = (uint16_t)(i < names.size() ? names[i].size() : 0);
        file_.write((const char*)&name_len, sizeof(name_len));
        if (name_len > 0) {
            file_.write(names[i].data(), name_len);
        }
        file_.write((const char*)&e.block_start, sizeof(e.block_start));
        file_.write((const char*)&e.num_blocks, sizeof(e.num_blocks));
    }
}

void QUANTWriter::close() {
    if (file_.is_open()) {
        file_.close();
    }
}

// ===========================================================================
// QUANTReader
// ===========================================================================

QUANTReader::QUANTReader(const std::string& path)
    : mapped_file_(nullptr), data_(nullptr), file_size_(0),
      format_table_offset_(0), tensor_table_offset_(0), data_offset_(0),
      num_format_blocks_(0), num_tensors_(0), valid_(false)
{
    // P0 fix: unique_ptr ownership (was raw new, leaked on every early
    // return) + valid_ gate (was valid() true on corrupt-magic bail) +
    // sizeof(QUANTHeader) (was hardcoded 16).
    auto mf = std::make_unique<MappedFile>();
    if (!mf->open(path.c_str())) return;
    mapped_file_ = std::move(mf);
    data_ = mapped_file_->ptr();
    file_size_ = mapped_file_->size();

    if (file_size_ < sizeof(QUANTHeader)) return;
    memcpy(&header_, data_, sizeof(QUANTHeader));

    if (memcmp(header_.magic, "QUA1", 4) != 0) return;

    format_table_offset_ = sizeof(QUANTHeader) + header_.config_size;
    if (format_table_offset_ + sizeof(uint32_t) > file_size_) return;

    const uint8_t* p = data_ + format_table_offset_;
    memcpy(&num_format_blocks_, p, sizeof(num_format_blocks_)); p += sizeof(uint32_t);

    size_t ft_bytes = sizeof(uint32_t) + (size_t)num_format_blocks_ * sizeof(FormatBlockEntry);
    tensor_table_offset_ = format_table_offset_ + ft_bytes;

    if (tensor_table_offset_ + sizeof(uint32_t) > file_size_) return;
    p = data_ + tensor_table_offset_;
    memcpy(&num_tensors_, p, sizeof(num_tensors_)); p += sizeof(uint32_t);

    const uint8_t* tt_start = p;
    const uint8_t* tt_end = data_ + file_size_;
    for (uint32_t i = 0; i < num_tensors_; i++) {
        if (tt_start + sizeof(uint16_t) > tt_end) { num_tensors_ = i; break; }
        uint16_t name_len; memcpy(&name_len, tt_start, sizeof(name_len)); tt_start += sizeof(name_len);
        if (tt_start + name_len + sizeof(uint32_t) + sizeof(uint32_t) > tt_end) { num_tensors_ = i; break; }
        tt_start += name_len;
        tt_start += sizeof(uint32_t) + sizeof(uint32_t);
    }
    data_offset_ = (size_t)(tt_start - data_);

    p = data_ + format_table_offset_;
    memcpy(&num_format_blocks_, p, sizeof(num_format_blocks_)); p += sizeof(uint32_t);
    cached_ft_.resize(num_format_blocks_);
    for (uint32_t i = 0; i < num_format_blocks_; i++) {
        memcpy(&cached_ft_[i].block_id, p, sizeof(uint32_t)); p += sizeof(uint32_t);
        memcpy(&cached_ft_[i].format, p, sizeof(uint8_t)); p += sizeof(uint8_t);
        memcpy(&cached_ft_[i].cb_bytes, p, sizeof(uint32_t)); p += sizeof(uint32_t);
    }

    block_offsets_.resize(num_format_blocks_);
    size_t off = data_offset_;
    const uint8_t* end = data_ + file_size_;
    for (uint32_t i = 0; i < num_format_blocks_; i++) {
        block_offsets_[i] = off;
        const uint8_t* q = data_ + off;
        if (q + sizeof(uint32_t) * 2 > end) break;
        uint32_t nw; memcpy(&nw, q, sizeof(nw));
        uint32_t cb; memcpy(&cb, q + sizeof(nw), sizeof(cb));
        if (q + sizeof(nw) + sizeof(cb) + cb + sizeof(uint32_t) > end) break;
        uint32_t idx; memcpy(&idx, q + sizeof(nw) + sizeof(cb) + cb, sizeof(idx));
        off += sizeof(nw) + sizeof(cb) + cb + sizeof(idx) + idx;
    }
    valid_ = true; // P0: whole header walk succeeded — file is valid
}

QUANTReader::~QUANTReader() = default; // P0: mapped_file_ is unique_ptr now

const QUANTHeader& QUANTReader::header() const {
    return header_;
}

std::vector<uint8_t> QUANTReader::read_config() const {
    // Guard against a corrupt/truncated header: config must fit in the file.
    if (header_.config_size == 0 || !data_) return {};
    if (sizeof(QUANTHeader) + header_.config_size > file_size_) return {};
    std::vector<uint8_t> cfg(header_.config_size);
    memcpy(cfg.data(), data_ + sizeof(QUANTHeader), header_.config_size);
    return cfg;
}

std::vector<FormatBlockEntry> QUANTReader::read_format_table() const {
    return cached_ft_;
}

size_t QUANTReader::num_blocks() const {
    return num_format_blocks_;
}

BlockData QUANTReader::read_block(uint32_t block_id) const {
    BlockData bd;
    bd.format = Format::Q32;
    if (block_id >= (uint32_t)block_offsets_.size()) return bd;

    // The first block of a valid file starts after the header (offset 0 means
    // an uninitialized/truncated entry). Check with pure size_t arithmetic
    // before forming the pointer, so a corrupt offset cannot cause an
    // out-of-bounds read.
    if (block_offsets_[block_id] < sizeof(QUANTHeader)) return bd;
    if ((size_t)block_offsets_[block_id] + sizeof(uint32_t) * 2 > file_size_) return bd;

    const uint8_t* p = data_ + block_offsets_[block_id];
    const uint8_t* file_end = data_ + file_size_;
    uint32_t nw; memcpy(&nw, p, sizeof(nw)); p += sizeof(nw);
    bd.num_weights = nw;
    uint32_t cb_bytes; memcpy(&cb_bytes, p, sizeof(cb_bytes)); p += sizeof(cb_bytes);
    if (cb_bytes > 0) {
        if (p + cb_bytes > file_end) return bd;
        bd.codebook.resize(cb_bytes);
        memcpy(bd.codebook.data(), p, cb_bytes); p += cb_bytes;
    }
    if (p + sizeof(uint32_t) > file_end) return bd;
    uint32_t idx_bytes; memcpy(&idx_bytes, p, sizeof(idx_bytes)); p += sizeof(idx_bytes);
    if (idx_bytes > 0) {
        if (p + idx_bytes > file_end) return bd;
        bd.indices.resize(idx_bytes);
        memcpy(bd.indices.data(), p, idx_bytes);
    }

    if (block_id < (uint32_t)cached_ft_.size()) {
        const uint8_t wire_fmt = cached_ft_[block_id].format;
        // v3 enum spans 105 ids (hole at 19 by design); anything outside the
        // valid set falls back to Q32 so a corrupt table cannot misdecode.
        bd.format = (wire_fmt < FORMAT_COUNT && wire_fmt != 19)
                        ? (Format)wire_fmt : Format::Q32;
    }

    return bd;
}

const uint8_t* QUANTReader::block_ptr(uint32_t block_id) const {
    if (block_id >= (uint32_t)block_offsets_.size()) return nullptr;
    if (block_offsets_[block_id] < sizeof(QUANTHeader)) return nullptr;
    if ((size_t)block_offsets_[block_id] + sizeof(uint32_t) * 2 > file_size_) return nullptr;
    return data_ + block_offsets_[block_id];
}

bool QUANTReader::tensor_blocks(const std::string& name, uint32_t& start, uint32_t& count) const {
    if (!data_) return false;
    const uint8_t* end = data_ + file_size_;
    const uint8_t* p = data_ + tensor_table_offset_;
    if (p + sizeof(uint32_t) > end) return false;
    uint32_t num_tensors; memcpy(&num_tensors, p, sizeof(num_tensors)); p += sizeof(uint32_t);
    for (uint32_t i = 0; i < num_tensors; i++) {
        if (p + sizeof(uint16_t) > end) return false;
        uint16_t name_len; memcpy(&name_len, p, sizeof(name_len)); p += sizeof(name_len);
        if (p + name_len + sizeof(uint32_t) * 2 > end) return false;
        std::string tensor_name((const char*)p, name_len); p += name_len;
        uint32_t block_start; memcpy(&block_start, p, sizeof(block_start)); p += sizeof(block_start);
        uint32_t num_blocks; memcpy(&num_blocks, p, sizeof(num_blocks)); p += sizeof(num_blocks);
        if (tensor_name == name) {
            start = block_start;
            count = num_blocks;
            return true;
        }
    }
    return false;
}

Tensor QUANTReader::read_tensor(const std::string& name) const {
    if (!data_) return Tensor();
    const uint8_t* end = data_ + file_size_;
    const uint8_t* p = data_ + tensor_table_offset_;
    if (p + sizeof(uint32_t) > end) return Tensor();
    uint32_t num_tensors; memcpy(&num_tensors, p, sizeof(num_tensors)); p += sizeof(uint32_t);

    for (uint32_t i = 0; i < num_tensors; i++) {
        if (p + sizeof(uint16_t) > end) return Tensor();
        uint16_t name_len; memcpy(&name_len, p, sizeof(name_len)); p += sizeof(name_len);
        if (p + name_len + sizeof(uint32_t) * 2 > end) return Tensor();
        std::string tensor_name((const char*)p, name_len); p += name_len;
        uint32_t block_start; memcpy(&block_start, p, sizeof(block_start)); p += sizeof(block_start);
        uint32_t num_blocks; memcpy(&num_blocks, p, sizeof(num_blocks)); p += sizeof(num_blocks);

        if (tensor_name == name) {
            int64_t total_weights = 0;
            for (uint32_t b = 0; b < num_blocks; b++) {
                BlockData bd = read_block(block_start + b);
                total_weights += bd.num_weights;
            }
            if (total_weights == 0) return Tensor();

            Tensor t(Shape{total_weights}, DType::F32);
            float* td = (float*)t.data();
            for (uint32_t b = 0; b < num_blocks; b++) {
                uint32_t blk_id = block_start + b;
                BlockData bd = read_block(blk_id);
                dequantize_block_all(bd.format, bd.indices.data(), bd.indices.size(),
                                     bd.codebook.data(), bd.codebook.size(),
                                     bd.num_weights, td);
                td += bd.num_weights;
            }
            return t;
        }
    }
    return Tensor();
}

std::vector<std::string> QUANTReader::tensor_names() const {
    if (!data_) return {};
    std::vector<std::string> names;
    const uint8_t* p = data_ + tensor_table_offset_;
    const uint8_t* end = data_ + file_size_;
    if ((size_t)tensor_table_offset_ + sizeof(uint32_t) > file_size_) return {};
    uint32_t num_tensors; memcpy(&num_tensors, p, sizeof(num_tensors)); p += sizeof(uint32_t);
    for (uint32_t i = 0; i < num_tensors; i++) {
        if (p + sizeof(uint16_t) > end) break;
        uint16_t name_len; memcpy(&name_len, p, sizeof(name_len)); p += sizeof(name_len);
        if (p + name_len + sizeof(uint32_t) + sizeof(uint32_t) > end) break;
        std::string tensor_name((const char*)p, name_len); p += name_len;
        names.push_back(tensor_name);
        p += sizeof(uint32_t) + sizeof(uint32_t);
    }
    return names;
}

std::vector<Format> QUANTReader::tensor_formats(const std::string& name) const {
    if (!data_) return {};
    std::vector<Format> fmts;
    const uint8_t* p = data_ + tensor_table_offset_;
    const uint8_t* end = data_ + file_size_;
    if ((size_t)tensor_table_offset_ + sizeof(uint32_t) > file_size_) return {};
    uint32_t num_tensors; memcpy(&num_tensors, p, sizeof(num_tensors)); p += sizeof(uint32_t);
    for (uint32_t i = 0; i < num_tensors; i++) {
        if (p + sizeof(uint16_t) > end) break;
        uint16_t name_len; memcpy(&name_len, p, sizeof(name_len)); p += sizeof(name_len);
        if (p + name_len + sizeof(uint32_t) * 2 > end) break;
        std::string tensor_name((const char*)p, name_len); p += name_len;
        uint32_t block_start; memcpy(&block_start, p, sizeof(block_start)); p += sizeof(block_start);
        uint32_t num_blocks; memcpy(&num_blocks, p, sizeof(num_blocks)); p += sizeof(num_blocks);

        if (tensor_name == name) {
            for (uint32_t b = 0; b < num_blocks && (block_start + b) < (uint32_t)cached_ft_.size(); b++) {
                uint32_t bid = block_start + b;
                const uint8_t wire_fmt = cached_ft_[bid].format;
                fmts.push_back((wire_fmt < FORMAT_COUNT && wire_fmt != 19)
                                   ? (Format)wire_fmt : Format::Q32);
            }
            return fmts;
        }
    }
    return fmts;
}

// ===========================================================================
// QUANTIdxWriter
// ===========================================================================

QUANTIdxWriter::QUANTIdxWriter(const std::string& path)
    : file_(path, std::ios::binary) {}

QUANTIdxWriter::~QUANTIdxWriter() {
    if (file_.is_open()) close();
}

void QUANTIdxWriter::write_idx(uint32_t version, const std::vector<std::string>& tensor_names) {
    if (!file_.is_open()) return;
    // Header: magic "TranscenderIDX" | version | num_tensors
    static const char MAGIC[15] = {'T','r','a','n','s','c','e','n','d','e','r','I','D','X','\0'};
    file_.write(MAGIC, 15);
    file_.write((const char*)&version, sizeof(version));
    uint32_t num = (uint32_t)tensor_names.size();
    file_.write((const char*)&num, sizeof(num));

    // Per tensor: name_len | name | sha256(name)
    for (const auto& name : tensor_names) {
        uint16_t name_len = (uint16_t)name.size();
        file_.write((const char*)&name_len, sizeof(name_len));
        if (name_len > 0) {
            file_.write(name.data(), name_len);
            SHA256Hash h = sha256((const uint8_t*)name.data(), name.size());
            file_.write((const char*)h.bytes, 32);
        } else {
            SHA256Hash h = sha256(nullptr, 0);
            file_.write((const char*)h.bytes, 32);
        }
    }
}

void QUANTIdxWriter::close() {
    if (file_.is_open()) file_.close();
}

// ===========================================================================
// QUANTIdxReader — recompute + fail-fast on corrupt
// ===========================================================================

QUANTIdxReader::QUANTIdxReader(const std::string& path)
    : mapped_file_(nullptr), data_(nullptr), file_size_(0),
      version_(0), num_tensors_(0), checked_(false)
{
    MappedFile* mf = new MappedFile();
    if (!mf->open(path.c_str())) { delete mf; return; }
    mapped_file_ = mf;
    data_ = mapped_file_->ptr();
    file_size_ = mapped_file_->size();
    // Magic: "TranscenderIDX" (15B, current) or legacy "InNovaIDX" (10B).
    // Reader accepts both; writer always emits current. Minimum size uses legacy.
    if (file_size_ < 18) { return; }
    if (memcmp(data_, "TranscenderIDX", 14) == 0) { magic_size_ = 15; }
    else if (memcmp(data_, "InNovaIDX", 9) == 0) { magic_size_ = 10; }
    else { data_ = nullptr; return; }
    size_t off = magic_size_;
    memcpy(&version_, data_ + off, sizeof(version_)); off += sizeof(version_);
    memcpy(&num_tensors_, data_ + off, sizeof(num_tensors_)); off += sizeof(num_tensors_);
}

QUANTIdxReader::~QUANTIdxReader() {
    delete mapped_file_;
}

std::vector<std::string> QUANTIdxReader::read_idx() {
    if (!data_) throw Error("QUANTIdxReader: invalid idx file");
    if (checked_) return names_;
    size_t off = magic_size_ + sizeof(version_) + sizeof(num_tensors_);
    for (uint32_t i = 0; i < num_tensors_; i++) {
        // Fail fast if file is truncated/corrupt structurally
        if (off + sizeof(uint16_t) > file_size_)
            throw Error("QUANTIdxReader: corrupt idx — truncated before tensor name length at index " + std::to_string(i));
        uint16_t name_len;
        memcpy(&name_len, data_ + off, sizeof(name_len)); off += sizeof(name_len);
        if (off + name_len + 32 > file_size_)
            throw Error("QUANTIdxReader: corrupt idx — truncated tensor name/hash at index " + std::to_string(i));
        std::string name((const char*)(data_ + off), name_len);
        off += name_len;
        SHA256Hash stored;
        memcpy(stored.bytes, data_ + off, 32); off += 32;

        // Recompute SHA-256 of the stored name and compare fail-fast
        SHA256Hash recomputed = sha256((const uint8_t*)name.data(), name.size());
        if (memcmp(recomputed.bytes, stored.bytes, 32) != 0) {
            // Report exactly which tensor name is corrupt
            std::string disp = name.empty() ? std::string("<empty>") : name;
            throw Error("QUANTIdxReader: corrupt idx — name hash mismatch for tensor \"" + disp +
                        "\" at index " + std::to_string(i));
        }
        names_.push_back(name);
    }
    checked_ = true;
    return names_;
}

} // namespace quant
