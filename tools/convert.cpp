#include "quant/quant_format.h"
#include "quant/tensor.h"
#include "quant/types.h"
#include "quant/format_planner.h"
#include "quant/codebook.h"
#include "quant/block_codec.h"

#include "quant/detail/cli_parse.h"
#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <vector>
#include <cstdint>
#include <map>
#include <algorithm>
#include <cmath>

struct ConvertArgs {
    std::string input_path;
    std::string output_path;
    std::string input_fmt = "rawfp32";
    float target_bpw = 0.0f; // 0 = no compression (FP32 passthrough)
    bool verbose = false;
};

static ConvertArgs parse_args(int argc, char** argv) {
    ConvertArgs args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            args.input_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            args.output_path = argv[++i];
        else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc)
            args.input_fmt = argv[++i];
        else if (strcmp(argv[i], "--bpw") == 0 && i + 1 < argc) {
            // BUGFIX (bug census): bare atof fail-open (garbage → 0.0 =
            // silent no-compression). Validated + range-checked.
            args.target_bpw = quant::cli_parse::parse_float(argv[i-1], argv[++i]);
            if (!(args.target_bpw >= 0.0f) || args.target_bpw > 32.0f) {
                std::cerr << "Error: --bpw needs 0..32\n";
                exit(2);
            }
        } else if (strcmp(argv[i], "--verbose") == 0)
            args.verbose = true;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: quant_convert --input <file> --output <model.quant> [options]\n";
            std::cout << "  --format fmt   Input format: rawfp32 (default), gguf\n";
            std::cout << "  --bpw N        Target bits-per-weight for FormatPlanner (0=no compression)\n";
            std::cout << "  --verbose      Verbose output\n";
            std::cout << "\nFormats: 1.0(quant1) 1.50(quant) 4.0(quant4) 8.0(quant8) 16.0(fp16) 32.0(fp32)\n";
            exit(0);
        }
    }
    return args;
}

// GGUF v1/v2/v3 header parsing and Q4_0/Q4_1/Q8_0/F16 dequantization
struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims;
    uint64_t ne[4];
    uint32_t ggml_type;
    uint64_t offset;
};

static const uint32_t GGML_TYPE_Q4_0 = 2;
static const uint32_t GGML_TYPE_Q4_1 = 3;
static const uint32_t GGML_TYPE_Q8_0 = 8;
static const uint32_t GGML_TYPE_F16  = 1;
static const uint32_t GGML_TYPE_F32  = 0;

static uint64_t read_le_u64(std::istream& is, bool& ok) {
    uint64_t v = 0;
    is.read(reinterpret_cast<char*>(&v), 8);
    if (!is) { ok = false; return 0; }
    return v;
}
static uint64_t read_le_u64(std::istream& is) {
    bool ok = true;
    return read_le_u64(is, ok);
}
static uint32_t read_le_u32(std::istream& is, bool& ok) {
    uint32_t v = 0;
    is.read(reinterpret_cast<char*>(&v), 4);
    if (!is) { ok = false; return 0; }
    return v;
}
static uint32_t read_le_u32(std::istream& is) {
    bool ok = true;
    return read_le_u32(is, ok);
}
[[maybe_unused]] static uint16_t read_le_u16(std::istream& is) {
    uint16_t v = 0;
    is.read(reinterpret_cast<char*>(&v), 2);
    return v;
}
[[maybe_unused]] static uint8_t read_u8(std::istream& is) {
    uint8_t v = 0;
    is.read(reinterpret_cast<char*>(&v), 1);
    return v;
}
static std::string read_string(std::istream& is) {
    // BUGFIX (bug census): trusted 8-byte len for string(len) + short read —
    // corrupt length = OOM, and &s[0] on empty string is UB.
    bool ok = true;
    uint64_t len = read_le_u64(is, ok);
    if (!ok) return "";
    static constexpr uint64_t kMaxNameLen = 1 << 20; // 1 MiB: names are tiny
    if (len > kMaxNameLen || !is) return "";
    std::string s;
    if (len == 0) return s;
    s.resize((size_t)len);
    is.read(&s[0], (std::streamsize)len);
    if (!is) return "";
    return s;
}

// Dequantization helpers — all take (block, out, offset, count) so the tail
// block writes only its valid remainder (BUGFIX: always wrote 32 floats,
// overflowing all_weights when numel%32 != 0).
static void dequantize_q4_0(const uint8_t* block, float* out, int64_t offset, int count) {
    // Q4_0 block: 1x fp16 scale (2 bytes) + 32x 4-bit values (16 bytes) = 18 bytes
    int16_t scale_half = *(const int16_t*)(block);
    float scale = (float)scale_half;
    // Convert from IEEE 754 half to float manually
    if (scale_half) {
        int sign = (scale_half >> 15) & 1;
        int exp  = (scale_half >> 10) & 0x1F;
        int mant = scale_half & 0x3FF;
        if (exp == 0) {
            // subnormal
            scale = (float)(mant) / 16384.0f * 2.0f;
            if (sign) scale = -scale;
        } else if (exp == 31) {
            scale = mant ? NAN : INFINITY;
        } else {
            scale = (float)((mant | 0x400) << 13) / 8388608.0f * (1 << (exp - 15));
            if (sign) scale = -scale;
        }
    }
    for (int i = 0; i < count; i++) {
        uint8_t q = block[2 + i / 2];
        if (i % 2 == 0) q &= 0x0F;
        else            q >>= 4;
        out[offset + i] = ((float)q - 8.0f) * scale;
    }
}

static void dequantize_q4_1(const uint8_t* block, float* out, int64_t offset, int count) {
    // Q4_1 block: 1x fp16 scale (2 bytes) + 1x fp16 min (2 bytes) + 32x 4-bit values (16 bytes) = 20 bytes
    int16_t scale_half = *(const int16_t*)(block);
    int16_t min_half   = *(const int16_t*)(block + 2);
    auto half_to_float = [](int16_t h) -> float {
        int sign = (h >> 15) & 1;
        int exp  = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0) {
            float v = (float)(mant) / 16384.0f * 2.0f;
            return sign ? -v : v;
        } else if (exp == 31) {
            return mant ? NAN : INFINITY;
        } else {
            float v = (float)((mant | 0x400) << 13) / 8388608.0f * (1 << (exp - 15));
            return sign ? -v : v;
        }
    };
    float scale = half_to_float(scale_half);
    float min   = half_to_float(min_half);
    for (int i = 0; i < count; i++) {
        uint8_t q = block[4 + i / 2];
        if (i % 2 == 0) q &= 0x0F;
        else            q >>= 4;
        out[offset + i] = (float)q * scale + min;
    }
}

static void dequantize_q8_0(const uint8_t* block, float* out, int64_t offset, int count) {
    // Q8_0 block: 1x fp16 scale (2 bytes) + 32x int8 values (32 bytes) = 34 bytes
    int16_t scale_half = *(const int16_t*)(block);
    auto half_to_float = [](int16_t h) -> float {
        int sign = (h >> 15) & 1;
        int exp  = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0) {
            float v = (float)(mant) / 16384.0f * 2.0f;
            return sign ? -v : v;
        } else if (exp == 31) {
            return mant ? NAN : INFINITY;
        } else {
            float v = (float)((mant | 0x400) << 13) / 8388608.0f * (1 << (exp - 15));
            return sign ? -v : v;
        }
    };
    float scale = half_to_float(scale_half);
    for (int i = 0; i < count; i++) {
        int8_t q = (int8_t)block[2 + i];
        out[offset + i] = (float)q * scale;
    }
}

// Read GGUF model from file and convert to FP32
static bool read_gguf(const std::string& path, std::vector<float>& all_weights,
                      std::vector<int64_t>& all_shapes, std::vector<std::string>& all_names,
                      bool verbose) {
    std::ifstream is(path, std::ios::binary);
    if (!is) { std::cerr << "Cannot open " << path << "\n"; return false; }

    // Read magic: "GGUF" v1/v2/v3
    char magic[4];
    is.read(magic, 4);
    if (memcmp(magic, "GGUF", 4) != 0) {
        std::cerr << "Not a GGUF file (magic: " << magic[0] << magic[1] << magic[2] << magic[3] << ")\n";
        return false;
    }
    uint32_t version = read_le_u32(is);
    if (version < 1 || version > 3) {
        std::cerr << "Unsupported GGUF version: " << version << "\n";
        return false;
    }
    uint64_t n_tensors   = read_le_u64(is);
    uint64_t n_kv        = read_le_u64(is);
    if (verbose)
        std::cout << "GGUF v" << version << " tensors=" << n_tensors << " metadata=" << n_kv << "\n";

    // Skip metadata key-value pairs
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = read_string(is);
        uint32_t val_type = read_le_u32(is);
        switch (val_type) {
            case 0: is.ignore(1); break;                             // uint8
            case 1: is.ignore(8); break;                             // int8
            case 2: is.ignore(4); break;                             // uint16
            case 3: is.ignore(2); break;                             // int16
            case 4: is.ignore(8); break;                             // uint32
            case 5: is.ignore(4); break;                             // int32
            case 6: is.ignore(8); break;                             // float32
            case 7: is.ignore(4); break;                             // bool
            case 8: read_string(is); break;                          // string
            case 9: {                                                // array
                uint32_t atype = read_le_u32(is);
                uint64_t alen  = read_le_u64(is);
                for (uint64_t j = 0; j < alen; j++) {
                    switch (atype) {
                        case 8: read_string(is); break;
                        default: is.ignore(4); break;
                    }
                }
                break;
            }
            case 10: is.ignore(8); break;                            // uint64
            case 11: is.ignore(8); break;                            // int64
            case 12: is.ignore(8); break;                            // float64
            default: is.ignore(4); break;
        }
    }

    // Read tensor info
    struct GGUFTensorInfo info;
    std::vector<GGUFTensorInfo> tensor_infos;
    tensor_infos.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        info.name = read_string(is);
        if (!is) { std::cerr << "Truncated tensor header\n"; return false; }
        info.n_dims = read_le_u32(is);
        // BUGFIX (bug census): attacker n_dims wrote past ne[4] stack array.
        if (!is || info.n_dims > 4) { std::cerr << "Bad n_dims\n"; return false; }
        info.ggml_type = read_le_u32(is);
        info.offset = read_le_u64(is);
        // v3 adds file offset alignment
        if (version >= 3)
            info.offset = read_le_u64(is);
        if (!is) { std::cerr << "Truncated tensor header\n"; return false; }
        for (uint32_t d = 0; d < info.n_dims; d++)
            info.ne[d] = read_le_u64(is);
        if (!is) { std::cerr << "Truncated tensor dims\n"; return false; }
        for (uint32_t d = info.n_dims; d < 4; d++)
            info.ne[d] = 1;
        tensor_infos.push_back(info);
    }

    // Read and dequantize each tensor
    for (auto& ti : tensor_infos) {
        int64_t num_el = 1;
        for (uint32_t d = 0; d < ti.n_dims; d++)
            num_el *= (int64_t)ti.ne[d];

        all_names.push_back(ti.name);
        all_shapes.push_back(num_el);
        size_t start = all_weights.size();
        all_weights.resize(start + num_el);

        is.seekg(ti.offset, std::ios::beg);
        if (verbose)
            std::cout << "  tensor " << ti.name << " type=" << ti.ggml_type << " n_el=" << num_el << " offset=" << ti.offset << "\n";

        if (ti.ggml_type == GGML_TYPE_F32) {
            // Direct FP32 read
            is.read(reinterpret_cast<char*>(&all_weights[start]), (std::streamsize)(num_el * 4));
        } else if (ti.ggml_type == GGML_TYPE_F16) {
            for (int64_t j = 0; j < num_el; j++) {
                int16_t h = 0;
                is.read(reinterpret_cast<char*>(&h), 2);
                int sign = (h >> 15) & 1;
                int exp  = (h >> 10) & 0x1F;
                int mant = h & 0x3FF;
                float f;
                if (exp == 0)
                    f = (float)(mant) / 16384.0f * 2.0f;
                else if (exp == 31)
                    f = mant ? NAN : INFINITY;
                else
                    f = (float)((mant | 0x400) << 13) / 8388608.0f * (1 << (exp - 15));
                all_weights[start + j] = sign ? -f : f;
            }
        } else if (ti.ggml_type == GGML_TYPE_Q4_0) {
            int64_t n_blocks = (num_el + 31) / 32;
            for (int64_t b = 0; b < n_blocks; b++) {
                uint8_t block[18];
                is.read(reinterpret_cast<char*>(block), 18);
                if (!is) { std::cerr << "Truncated Q4_0 block\n"; return false; }
                int64_t done = std::min<int64_t>(32, num_el - b * 32);
                dequantize_q4_0(block, all_weights.data(), start + b * 32, (int)done);
            }
        } else if (ti.ggml_type == GGML_TYPE_Q4_1) {
            int64_t n_blocks = (num_el + 31) / 32;
            for (int64_t b = 0; b < n_blocks; b++) {
                uint8_t block[20];
                is.read(reinterpret_cast<char*>(block), 20);
                if (!is) { std::cerr << "Truncated Q4_1 block\n"; return false; }
                int64_t done = std::min<int64_t>(32, num_el - b * 32);
                dequantize_q4_1(block, all_weights.data(), start + b * 32, (int)done);
            }
        } else if (ti.ggml_type == GGML_TYPE_Q8_0) {
            int64_t n_blocks = (num_el + 31) / 32;
            for (int64_t b = 0; b < n_blocks; b++) {
                uint8_t block[34];
                is.read(reinterpret_cast<char*>(block), 34);
                if (!is) { std::cerr << "Truncated Q8_0 block\n"; return false; }
                int64_t done = std::min<int64_t>(32, num_el - b * 32);
                dequantize_q8_0(block, all_weights.data(), start + b * 32, (int)done);
            }
        } else {
            std::cerr << "Unsupported GGML type " << ti.ggml_type << " for tensor " << ti.name << "\n";
            return false;
        }
    }
    return true;
}

// Quantize one tensor with the canonical block codec: per 256-weight block,
// the planner-assigned format is encoded with quantize_block_all() so the
// output is byte-identical to what the engine's dequantize_block_all()
// decodes (single encoding per format ID — no hand-rolled lookalikes).
struct TensorBlockData {
    quant::Format format;
    std::vector<uint8_t> indices;
    std::vector<uint8_t> codebook;
};

static bool quantize_tensor_blocks(const float* data, int64_t numel, float target_bpw,
                                   quant::FormatPlan* plan_out,
                                   std::vector<TensorBlockData>& blocks,
                                   std::vector<quant::Format>& formats) {
    if (!data || numel <= 0 || target_bpw <= 0.0f) return false;

    quant::FormatPlanner planner(target_bpw);
    int64_t num_blocks = (int64_t)((numel + 255) / 256);
    quant::FormatPlan plan = planner.plan_for_model(num_blocks * 256);
    if (plan_out) *plan_out = plan;

    blocks.clear();
    formats.clear();
    for (int64_t b = 0; b < num_blocks; b++) {
        const int64_t start = b * 256;
        const int64_t blen = std::min<int64_t>(256, numel - start);
        quant::Format fmt = (b < (int64_t)plan.blocks.size())
                              ? plan.blocks[(size_t)b].assigned_format
                              : quant::Format::Q8;
        TensorBlockData blk;
        blk.format = fmt;
        if (!quant::quantize_block_all(fmt, data + start, (int)blen, blk.indices, blk.codebook))
            return false;
        formats.push_back(fmt);
        blocks.push_back(std::move(blk));
    }
    return true;
}

static bool convert_raw_fp32(const std::string& in_path, const std::string& out_path) {
    std::ifstream in(in_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: cannot open input: " << in_path << std::endl;
        return false;
    }
    in.seekg(0, std::ios::end);
    size_t file_size = (size_t)in.tellg();
    in.seekg(0, std::ios::beg);

    if (file_size == 0 || file_size % 4 != 0) {
        std::cerr << "Error: raw FP32 file size must be multiple of 4\n";
        return false;
    }

    size_t num_floats = file_size / 4;
    std::vector<float> data(num_floats);
    in.read(reinterpret_cast<char*>(data.data()), (std::streamsize)file_size);
    in.close();

    quant::Tensor weights(quant::Shape{1, (int64_t)num_floats}, quant::DType::F32);
    memcpy(weights.data(), data.data(), file_size);

    quant::QUANTWriter writer(out_path);
    quant::QUANTHeader hdr;
    memcpy(hdr.magic, "QUA1", 4);
    hdr.version = 1;
    hdr.flags = 0;
    hdr.config_size = 0;
    writer.write_header(hdr, nullptr);

    quant::FormatBlockEntry ft_entry;
    ft_entry.block_id = 0;
    ft_entry.format = (uint8_t)quant::Format::Q32;
    ft_entry.cb_bytes = 0;
    writer.write_format_table({ft_entry});

    quant::TensorEntry t_entry;
    t_entry.name_len = 0;
    t_entry.block_start = 0;
    t_entry.num_blocks = 1;
    writer.write_tensor_table({t_entry}, {"weights"});

    quant::BlockData block;
    block.format = quant::Format::Q32;
    block.num_weights = (uint32_t)num_floats;
    block.indices.resize(file_size);
    memcpy(block.indices.data(), data.data(), file_size);
    writer.write_block(block);

    writer.close();
    return true;
}

// High-level GGUF conversion: reads via read_gguf(), optionally compresses with FormatPlanner, writes QUANT
static bool convert_gguf(const std::string& in_path, const std::string& out_path, float target_bpw) {
    std::vector<float> all_weights;
    std::vector<int64_t> all_shapes;
    std::vector<std::string> all_names;
    if (!read_gguf(in_path, all_weights, all_shapes, all_names, false))
        return false;

    quant::QUANTWriter writer(out_path);
    quant::QUANTHeader hdr;
    memcpy(hdr.magic, "QUA1", 4);
    hdr.version = 1;
    hdr.flags = 0;
    hdr.config_size = 0;
    writer.write_header(hdr, nullptr);

    const bool use_compression = (target_bpw > 0.0f && !all_weights.empty());

    std::vector<quant::FormatBlockEntry> fmt_entries;
    std::vector<quant::TensorEntry> t_entries;
    std::vector<std::string> t_names;
    std::vector<quant::BlockData> block_data;
    quant::Format primary_format = quant::Format::Q32;

    // Real stored bytes / weights across all written blocks — used to print
    // the ACTUAL achieved BPW (not the requested target) after conversion.
    double total_qbytes = 0.0;
    int64_t total_qweights = 0;

    size_t cursor = 0;
    for (size_t i = 0; i < all_names.size(); i++) {
        const int64_t numel = all_shapes[i];
        if (numel < 0) { std::cerr << "Error: negative tensor size for " << all_names[i] << "\n"; return false; }

        quant::TensorEntry te;
        te.name_len = (uint32_t)all_names[i].size();
        te.block_start = (uint32_t)fmt_entries.size();
        te.num_blocks = 0;
        t_entries.push_back(te);
        t_names.push_back(all_names[i]);

        if (use_compression) {
            // Canonical per-block encoding for THIS tensor (its own plan), so
            // the written payload is byte-identical to the engine decode path.
            std::vector<TensorBlockData> blocks;
            std::vector<quant::Format> formats;
            quant::FormatPlan plan;
            if (!quantize_tensor_blocks(all_weights.data() + cursor, numel, target_bpw,
                                        &plan, blocks, formats)) {
                std::cerr << "Error: block quantization failed for " << all_names[i] << "\n";
                return false;
            }
            for (size_t b = 0; b < blocks.size(); b++) {
                total_qbytes += (double)(blocks[b].indices.size() +
                                         blocks[b].codebook.size());
                total_qweights += (int64_t)std::min<int64_t>(
                    256, numel - (int64_t)b * 256);

                quant::FormatBlockEntry fe;
                fe.block_id = (uint32_t)fmt_entries.size();
                fe.format = (uint8_t)blocks[b].format;
                fe.cb_bytes = (uint32_t)blocks[b].codebook.size();
                fmt_entries.push_back(fe);

                quant::BlockData block;
                block.format = blocks[b].format;
                block.num_weights = (uint32_t)std::min<int64_t>(256, numel - (int64_t)b * 256);
                block.indices = std::move(blocks[b].indices);
                block.codebook = std::move(blocks[b].codebook);
                block_data.push_back(std::move(block));
                te.num_blocks++;
            }
            if (primary_format == quant::Format::Q32 && !formats.empty())
                primary_format = formats[0];
        } else {
            // FP32: one raw block per tensor.
            quant::FormatBlockEntry fe;
            fe.block_id = (uint32_t)fmt_entries.size();
            fe.format = (uint8_t)quant::Format::Q32;
            fe.cb_bytes = 0;
            fmt_entries.push_back(fe);

            quant::BlockData block;
            block.format = quant::Format::Q32;
            block.num_weights = (uint32_t)numel;
            const size_t byte_size = (size_t)numel * sizeof(float);
            block.indices.resize(byte_size);
            if (byte_size > 0) memcpy(block.indices.data(), &all_weights[cursor], byte_size);
            block_data.push_back(std::move(block));
            te.num_blocks = 1;
        }
        cursor += (size_t)numel;
    }

    writer.write_format_table(fmt_entries);
    writer.write_tensor_table(t_entries, t_names);
    for (auto& block : block_data) writer.write_block(block);

    writer.close();
    std::cout << "Converted " << all_names.size() << " tensors to QUANT: " << out_path;
    if (use_compression) {
        const double achieved = (total_qweights > 0)
            ? total_qbytes * 8.0 / (double)total_qweights : 0.0;
        std::cout << " (requested " << target_bpw << " BPW; achieved "
                  << achieved << " BPW; primary format "
                  << quant::format_name(primary_format) << ")";
    }
    std::cout << std::endl;
    return true;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    std::cout << "QUANT Converter" << std::endl;

    if (args.input_path.empty() || args.output_path.empty()) {
        std::cerr << "Error: --input and --output are required\n";
        return 1;
    }

    bool ok = false;
    if (args.input_fmt == "rawfp32") {
        ok = convert_raw_fp32(args.input_path, args.output_path);
    } else if (args.input_fmt == "gguf") {
        ok = convert_gguf(args.input_path, args.output_path, args.target_bpw);
    } else {
        std::cerr << "Unknown format: " << args.input_fmt << std::endl;
        return 1;
    }

    if (!ok) return 1;
    if (args.verbose) {
        std::cout << "Written: " << args.output_path << std::endl;
    }
    return 0;
}
