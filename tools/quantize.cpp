#include "quant/quant_format.h"
#include "quant/quant_engines.h"
#include "quant/codebook.h"
#include "quant/kernel.h"
#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/block_codec.h"

#include "quant/detail/cli_parse.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>

using namespace quant;

static Format parse_format(const std::string& s) {
    // BUGFIX (bug census): duplicated QG alternatives, lowercase qg* never
    // matched, quant1/quant*/quad names from --help rejected, and unknown
    // fail-open defaulted to Q8 with a warning. Now: canonical v3 names +
    // common aliases; unknown is a hard error (no silent misquantize).
    // NOTE: throws on unknown — parse_args lets it propagate as a usage error.
    auto lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    if (lower == "q1") return Format::Q1;
    if (lower == "q2") return Format::Q2;
    if (lower == "q3") return Format::Q3;
    if (lower == "q4") return Format::Q4;
    if (lower == "q6") return Format::Q6;
    if (lower == "q8" || lower == "quant8") return Format::Q8;
    if (lower == "q12") return Format::Q12;
    if (lower == "q16" || lower == "fp16") return Format::Q16;
    if (lower == "q24") return Format::Q24;
    if (lower == "q32" || lower == "fp32") return Format::Q32;
    if (lower == "qg1" || lower == "q1_g" || lower == "q1_grp") return Format::QG1;
    if (lower == "qg2" || lower == "q2_g" || lower == "q2_grp") return Format::QG2;
    if (lower == "qg3" || lower == "q3_g" || lower == "q3_grp") return Format::QG3;
    if (lower == "qg4" || lower == "q4_g" || lower == "q4_grp") return Format::QG4;
    if (lower == "qg6" || lower == "q6_g" || lower == "q6_grp") return Format::QG6;
    if (lower == "qg8" || lower == "q8_g" || lower == "q8_grp") return Format::QG8;
    if (lower == "qg12" || lower == "q12_g" || lower == "q12_grp") return Format::QG12;
    if (lower == "qg16" || lower == "q16_g" || lower == "q16_grp") return Format::QG16;
    if (lower == "qg24" || lower == "q24_g" || lower == "q24_grp") return Format::QG24;
    throw std::runtime_error("unknown format '" + s + "'");
}

static Format s_default_format = Format::Q8;

struct QuantArgs {
    std::string input_path;
    std::string output_path;
    std::string format = "q8"; // BUGFIX: was "quant8" which parse_format rejected (always warned + fell back)
    std::string per_layer_format;
    int num_bits = 8; // BUGFIX: parsed but never read downstream — kept for compat, validated
};

static QuantArgs parse_args(int argc, char** argv) {
    QuantArgs args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            args.input_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            args.output_path = argv[++i];
        else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            args.format = argv[++i];
            try {
                s_default_format = parse_format(args.format);
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
                exit(2);
            }
        } else if (strcmp(argv[i], "--per-layer") == 0 && i + 1 < argc)
            args.per_layer_format = argv[++i];
        else if (strcmp(argv[i], "--num-bits") == 0 && i + 1 < argc) {
            args.num_bits = quant::cli_parse::parse_int(argv[i-1], argv[++i]);
            if (args.num_bits <= 0 || args.num_bits > 32) {
                std::cerr << "Error: --num-bits needs 1..32\n";
                exit(2);
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: quant_quantize --input model.quant --output quantized.quant [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --format <f>       Target format (default: q8)\n";
            std::cout << "                     Formats: q1 q2 q3 q4 q6 q8 q12 q16 q24 q32/fp32,\n";
            std::cout << "                     qg1 qg2 qg3 qg4 qg6 qg8 qg12 qg16 qg24 (aliases:\n";
            std::cout << "                     qN_g, qN_grp, any case; quant8 = q8, fp16 = q16)\n";
            std::cout << "  --per-layer <csv>  Per-layer formats (name=fmt,name=fmt,...)\n";
            std::cout << "  --num-bits N       Bits hint 1..32 (default: 8; informational)\n";
            exit(0);
        }
    }
    return args;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    if (args.input_path.empty() || args.output_path.empty()) {
        std::cerr << "Error: --input and --output required\n";
        return 1;
    }

    std::cout << "QUANT Quantization Tool\n";
    std::cout << "Input: " << args.input_path << "\n";
    std::cout << "Output: " << args.output_path << "\n";
    std::cout << "Format: " << args.format << "\n";

    std::unordered_map<std::string, Format> per_layer;
    if (!args.per_layer_format.empty()) {
        std::stringstream ss(args.per_layer_format);
        std::string item;
        while (std::getline(ss, item, ',')) {
            auto eq = item.find('=');
            if (eq != std::string::npos) {
                std::string name = item.substr(0, eq);
                std::string fmt = item.substr(eq + 1);
                try {
                    per_layer[name] = parse_format(fmt);
                } catch (const std::exception& e) {
                    std::cerr << "Error: layer '" << name << "': " << e.what() << "\n";
                    return 2;
                }
                std::cout << "  Layer '" << name << "' -> " << fmt << "\n";
            } else if (!item.empty()) {
                std::cerr << "Error: --per-layer entry '" << item << "' needs name=fmt\n";
                return 2;
            }
        }
    }

    QUANTReader reader(args.input_path);
    if (!reader.valid()) {
        std::cerr << "Error: cannot open " << args.input_path << "\n";
        return 1;
    }

    auto tensor_names = reader.tensor_names();
    std::cout << "Found " << tensor_names.size() << " tensors\n";

    QUANTWriter writer(args.output_path);
    QUANTHeader hdr;
    std::memcpy(hdr.magic, "QUA1", 4);
    hdr.version = 1;
    hdr.flags = 0;
    hdr.config_size = 0;
    writer.write_header(hdr, nullptr);

    std::vector<FormatBlockEntry> ft_entries;
    std::vector<TensorEntry> tensor_entries;
    std::vector<std::string> names;
    uint32_t block_id = 0;

    for (const auto& name : tensor_names) {
        Tensor tensor = reader.read_tensor(name);
        // BUGFIX (bug census): empty tensors silently `continue`d yet the
        // tool exited 0 (silent skip). Count + warn instead.
        if (tensor.numel() == 0) {
            std::cerr << "Warning: skipping empty tensor '" << name << "'\n";
            continue;
        }

        Format fmt = s_default_format;
        auto it = per_layer.find(name);
        if (it != per_layer.end())
            fmt = it->second;

        int64_t num_blocks = (tensor.numel() + 255) / 256;

        const float* data = tensor.data<float>();
        int64_t total_weights = tensor.numel();

        // Real stored bytes produced by the canonical codec for this tensor.
        size_t qbytes = 0;

        for (int64_t b = 0; b < num_blocks; b++) {
            int64_t block_start = b * 256;
            int64_t block_end = std::min(block_start + 256, total_weights);
            int64_t block_size = block_end - block_start;

            BlockData block;
            block.format = fmt;
            block.num_weights = (uint32_t)block_size;

            // Encode with the CANONICAL block codec — the single source of
            // truth for on-disk payloads. This keeps the tool's output
            // byte-identical in layout to everything the reader decodes
            // (quantize_block_all/dequantize_block_all), for every format.
            // BUGFIX (bug census): the Q32 fallback return was unchecked —
            // double-failure wrote a corrupt block. Abort loudly instead.
            if (!quantize_block_all(fmt, data + block_start, (int)block_size,
                                    block.indices, block.codebook)) {
                std::cerr << "Warning: " << name << " block " << b
                          << ": format unsupported by canonical codec, "
                             "writing raw Q32\n";
                block.format = Format::Q32;
                if (!quantize_block_all(Format::Q32, data + block_start,
                                        (int)block_size, block.indices, block.codebook)) {
                    std::cerr << "Error: " << name << " block " << b
                              << ": Q32 fallback encode failed — aborting\n";
                    return 1;
                }
            }

            writer.write_block(block);
            qbytes += block.indices.size() + block.codebook.size();

            FormatBlockEntry entry;
            entry.block_id = block_id;
            entry.format = (uint8_t)block.format;
            entry.cb_bytes = (uint32_t)block.codebook.size();
            ft_entries.push_back(entry);

            block_id++;
        }

        TensorEntry te;
        te.name_len = (uint16_t)name.size();
        te.block_start = block_id - (uint32_t)num_blocks;
        te.num_blocks = (uint32_t)num_blocks;
        tensor_entries.push_back(te);
        names.push_back(name);

        size_t mb = (tensor.size_bytes() + 1048575) / 1048576;
        size_t qmb = (qbytes + 1048575) / 1048576;
        std::cout << "  " << name << ": " << mb << "MB -> " << qmb << "MB ("
                  << args.format << ")\n";
    }

    writer.write_format_table(ft_entries);
    writer.write_tensor_table(tensor_entries, names);
    writer.close();

    std::cout << "Quantization complete. Output: " << args.output_path << "\n";
    return 0;
}
