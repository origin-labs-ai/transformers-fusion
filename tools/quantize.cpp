#include "quant/quant_format.h"
#include "quant/quant_engines.h"
#include "quant/codebook.h"
#include "quant/kernel.h"
#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/block_codec.h"

#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>

using namespace quant;

static Format parse_format(const std::string& s) {
    if (s == "q1" || s == "Q1") return Format::Q1;
    if (s == "q2" || s == "Q2") return Format::Q2;
    if (s == "q3" || s == "Q3") return Format::Q3;
    if (s == "q4" || s == "Q4") return Format::Q4;
    if (s == "q6" || s == "Q6") return Format::Q6;
    if (s == "q8" || s == "Q8") return Format::Q8;
    if (s == "q12" || s == "Q12") return Format::Q12;
    if (s == "q16" || s == "Q16") return Format::Q16;
    if (s == "q24" || s == "Q24") return Format::Q24;
    if (s == "q32" || s == "Q32" || s == "fp32" || s == "FP32") return Format::Q32;
    if (s == "q1_grp" || s == "Q1_GRP") return Format::Q1_GRP;
    if (s == "q2_grp" || s == "Q2_GRP") return Format::Q2_GRP;
    if (s == "q3_grp" || s == "Q3_GRP") return Format::Q3_GRP;
    if (s == "q4_grp" || s == "Q4_GRP") return Format::Q4_GRP;
    if (s == "q6_grp" || s == "Q6_GRP") return Format::Q6_GRP;
    if (s == "q8_grp" || s == "Q8_GRP") return Format::Q8_GRP;
    if (s == "q12_grp" || s == "Q12_GRP") return Format::Q12_GRP;
    if (s == "q16_grp" || s == "Q16_GRP") return Format::Q16_GRP;
    if (s == "q24_grp" || s == "Q24_GRP") return Format::Q24_GRP;
    if (s == "quad_3_5" || s == "QUAD_MIX_3_5") return Format::MXQ_3_5_GRP;
    std::cerr << "Warning: unknown format '" << s << "' — defaulting to Q8\n";
    return Format::Q8;
}

static Format s_default_format = Format::Q8;

struct QuantArgs {
    std::string input_path;
    std::string output_path;
    std::string format = "quant8";
    std::string per_layer_format;
    int num_bits = 8;
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
            s_default_format = parse_format(args.format);
        } else if (strcmp(argv[i], "--per-layer") == 0 && i + 1 < argc)
            args.per_layer_format = argv[++i];
        else if (strcmp(argv[i], "--num-bits") == 0 && i + 1 < argc)
            args.num_bits = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: quant_quantize --input model.quant --output quantized.quant [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --format <f>       Target format (default: quant8)\n";
            std::cout << "                     Formats: fp32, fp16, quant1, quant1_grp, quant2,\n";
            std::cout << "                     quant2_grp, quant4, quant4_grp, quant8, quant8_grp,\n";
            std::cout << "                     quant (1.50 bpw), quant_q0_grp,\n";
            std::cout << "                     quant_sparse, quant_sparse_grp\n";
            std::cout << "  --per-layer <csv>  Per-layer formats (name=fmt,name=fmt,...)\n";
            std::cout << "  --num-bits N       Number of bits (default: 8)\n";
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
                per_layer[name] = parse_format(fmt);
                std::cout << "  Layer '" << name << "' -> " << fmt << "\n";
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
        if (tensor.numel() == 0) continue;

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
            if (!quantize_block_all(fmt, data + block_start, (int)block_size,
                                    block.indices, block.codebook)) {
                std::cerr << "Warning: " << name << " block " << b
                          << ": format unsupported by canonical codec, "
                             "writing raw Q32\n";
                block.format = Format::Q32;
                quantize_block_all(Format::Q32, data + block_start,
                                   (int)block_size, block.indices, block.codebook);
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
