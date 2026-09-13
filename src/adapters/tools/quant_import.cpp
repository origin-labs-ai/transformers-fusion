// ============================================================================
// quant_import.cpp — CLI: auto-detect format & import -> single-format QUANT file
// ============================================================================
// Usage:
//   quant_import --input <path> --output <out.quant> [--format Q2]
//              [--bpw 2.0] [--block-size 256] [--verbose]
//
// Detects input format by magic bytes / extension, then dispatches to the
// appropriate bridge (GGUF, Safetensors, raw FP32/FP16/FP8, or re-quant QUANT).
//
// Sharded safetensors models (directory with model.safetensors.index.json, or
// the index file itself) are imported shard-by-shard into ONE .quant file.
// ============================================================================
#include "adapters/adapter_core.h"
#include "adapters/ptq_bridge.h"
#include "adapters/gguf_bridge.h"
#include "adapters/safetensors_bridge.h"
#include "quant/quant_format.h"
#include "quant/block_codec.h"
#include "quant/detail/cli_parse.h"

#include <iostream>
#include <cstring>
#include <cstdio>
#include <cctype>
#include <fstream>
#include <filesystem>
#include <system_error>
#include <map>
#include <algorithm>

using namespace quant::adapters;
using namespace quant;

static bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

static std::string to_upper(std::string s) {
    for (auto& c : s) c = (char)std::toupper((unsigned char)c);
    return s;
}

// Nearest single format for a requested --bpw.
static quant::Format nearest_format_for_bpw(float bpw) {
    if (bpw <= 1.25f) return quant::Format::Q1;
    if (bpw <= 1.75f) return quant::Format::Q1_5;
    if (bpw <= 3.0f)  return quant::Format::Q2;
    if (bpw <= 6.0f)  return quant::Format::QG4;
    if (bpw <= 12.0f) return quant::Format::QG8;
    if (bpw <= 24.0f) return quant::Format::QG16;
    return quant::Format::Q32;
}

// Parse every Q-series single format plus the Q_MX/QG_MX compound formats.
// Compounds set `out` to the mix wire format and `out_compound` to the
// compound RegFormat id.
static bool parse_format_name(const char* name, quant::Format& out, quant::RegFormat& out_compound) {
    std::string s = to_upper(name);
    bool is_compound = false;

    // ---- Q-series singles (Q2 covers the two Q1 aliases) ----
    if (s == "Q1")              { out = quant::Format::Q1; }
    else if (s == "Q2")         { out = quant::Format::Q2; }
    else if (s == "Q4")         { out = quant::Format::Q4; }
    else if (s == "Q8")         { out = quant::Format::Q8; }
    else if (s == "Q16")        { out = quant::Format::Q16; }
    else if (s == "Q32")        { out = quant::Format::Q32; }
    else if (s == "QG1")     { out = quant::Format::QG1; }
    else if (s == "QG2")     { out = quant::Format::QG2; }
    else if (s == "QG4")     { out = quant::Format::QG4; }
    else if (s == "QG8")     { out = quant::Format::QG8; }
    else if (s == "QG16")    { out = quant::Format::QG16; }
    else if (s == "Q1_5")     { out = quant::Format::Q1_5; }
    else if (s == "QG_1_5") { out = quant::Format::QG_1_5; }
    // ---- QG_MX compounds (adaptive, four-tier) ----
    else if (s == "QG_MX_3_5" || s == "QG_MX_3.5")   { out = quant::Format::QG_MX_3_5;   out_compound = quant::RegFormat::QG_MX_3_5;   is_compound = true; }
    else if (s == "QG_MX_4_5" || s == "QG_MX_4.5")   { out = quant::Format::QG_MX_4_5;   out_compound = quant::RegFormat::QG_MX_4_5;   is_compound = true; }
    else if (s == "QG_MX_6_5" || s == "QG_MX_6.5")   { out = quant::Format::QG_MX_6_5;   out_compound = quant::RegFormat::QG_MX_6_5;   is_compound = true; }
    else if (s == "QG_MX_8_5" || s == "QG_MX_8.5")   { out = quant::Format::QG_MX_8_5;   out_compound = quant::RegFormat::QG_MX_8_5;   is_compound = true; }
    else if (s == "QG_MX_12_5" || s == "QG_MX_12.5") { out = quant::Format::QG_MX_12_5;  out_compound = quant::RegFormat::QG_MX_12_5;  is_compound = true; }
    else if (s == "QG_MX_16_5" || s == "QG_MX_16.5") { out = quant::Format::QG_MX_16_5;  out_compound = quant::RegFormat::QG_MX_16_5;  is_compound = true; }
    else if (s == "QG_MX_24_5" || s == "QG_MX_24.5") { out = quant::Format::QG_MX_24_5;  out_compound = quant::RegFormat::QG_MX_24_5;  is_compound = true; }
    else return false;

    if (!is_compound) out_compound = quant::format_to_regformat(out);
    return true;
}

// Parse safetensors index.json "weight_map" into ordered (tensor_name, shard)
// pairs. Minimal parser: expects {"weight_map": {"name": "shard", ...}}.
static std::vector<std::pair<std::string, std::string>> parse_weight_map(const std::string& json) {
    std::vector<std::pair<std::string, std::string>> out;
    size_t p = json.find("\"weight_map\"");
    if (p == std::string::npos) return out;
    p = json.find('{', p);
    if (p == std::string::npos) return out;
    p++;

    while (p < json.size()) {
        while (p < json.size() && json[p] != '"') p++;
        if (p >= json.size()) break;
        p++;
        std::string key;
        while (p < json.size() && json[p] != '"') {
            if (json[p] == '\\' && p + 1 < json.size()) { key += json[p + 1]; p += 2; continue; }
            key += json[p++];
        }
        p++; // closing quote
        while (p < json.size() && json[p] != ':') p++;
        p++; // colon
        while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n' || json[p] == '\r')) p++;
        if (p >= json.size() || json[p] != '"') break;
        p++;
        std::string val;
        while (p < json.size() && json[p] != '"') {
            if (json[p] == '\\' && p + 1 < json.size()) { val += json[p + 1]; p += 2; continue; }
            val += json[p++];
        }
        p++; // closing quote
        out.push_back({ key, val });
        while (p < json.size() && json[p] != ',') {
            if (json[p] == '}') break;
            p++;
        }
        p++; // comma (or brace, then loop exits on next '"' scan)
    }
    return out;
}

static std::string read_text_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return s;
}

// Import a sharded safetensors model into a single .quant file, processing one
// shard at a time to bound peak memory (one shard FP32 + compressed blocks).
static bool import_sharded(const std::string& input, const BridgeConfig& cfg) {
    std::fprintf(stderr, "[0] import_sharded entered, input=%s\n", input.c_str());
    std::string dir = input;
    std::string index_path;
    if (ends_with(input, ".json")) {
        index_path = input;
        size_t slash = input.find_last_of("/\\");
        dir = (slash == std::string::npos) ? "." : input.substr(0, slash);
    } else {
        while (dir.size() > 1 && (dir.back() == '/' || dir.back() == '\\')) dir.pop_back();
        index_path = dir + "/model.safetensors.index.json";
    }

    std::fprintf(stderr, "[0.5] reading index %s\n", index_path.c_str());
    std::string json = read_text_file(index_path);
    std::fprintf(stderr, "[1] index read %zu bytes\n", json.size());
    if (json.empty()) {
        std::fprintf(stderr, "Error: cannot read index file %s\n", index_path.c_str());
        return false;
    }
    auto wm = parse_weight_map(json);
    std::fprintf(stderr, "[2] weight map %zu tensors\n", wm.size());
    std::fprintf(stdout, "Sharded model: %zu tensors across %s\n", wm.size(), index_path.c_str());

    // Ordered unique shards.
    std::vector<std::string> shards;
    for (const auto& e : wm)
        if (std::find(shards.begin(), shards.end(), e.second) == shards.end())
            shards.push_back(e.second);

    QUANTWriter writer(cfg.output_path);
    std::fprintf(stderr, "[3] writer created\n");
    QUANTHeader hdr;
    std::memcpy(hdr.magic, "QUA1", 4);
    hdr.version = 1;
    hdr.flags = 0;
    hdr.config_size = 0;
    writer.write_header(hdr, nullptr);

    const int bs = cfg.block_size > 0 ? cfg.block_size : 256;
    const quant::Format fmt = cfg.format;
    const quant::MixDescriptor* mix = find_mix_descriptor(cfg.compound);
    const float eff_bpw = mix ? mix->effective_bpw : quant::format_bpw(fmt);

    std::vector<FormatBlockEntry> ft_entries;
    std::vector<TensorEntry> tensor_entries;
    std::vector<std::string> names;
    std::string tmp_path = cfg.output_path + ".tmp";
    std::ofstream block_tmp(tmp_path, std::ios::binary | std::ios::trunc);
    if (!block_tmp) {
        std::fprintf(stderr, "Error: cannot create temp block file %s\n", tmp_path.c_str());
        return false;
    }
    uint32_t block_id = 0;
    double total_bytes = 0.0;
    int64_t total_weights = 0;
    ft_entries.reserve(40000000);

    auto write_block_to_tmp = [&](const BlockData& b) {
        uint32_t nw = b.num_weights;
        block_tmp.write((const char*)&nw, sizeof(nw));
        uint32_t cb = (uint32_t)b.codebook.size();
        block_tmp.write((const char*)&cb, sizeof(cb));
        if (cb > 0) block_tmp.write((const char*)b.codebook.data(), cb);
        uint32_t idx = (uint32_t)b.indices.size();
        block_tmp.write((const char*)&idx, sizeof(idx));
        if (idx > 0) block_tmp.write((const char*)b.indices.data(), idx);
    };

    for (const auto& shard : shards) {
        std::string shard_path = dir + "/" + shard;
        auto tensors = load_safetensors(shard_path, cfg.verbose);
        if (tensors.empty()) {
            std::fprintf(stderr, "Error: shard %s loaded no tensors\n", shard_path.c_str());
            return false;
        }
        std::map<std::string, AdapterTensor> by_name;
        for (auto& t : tensors) by_name[t.name] = std::move(t);
        tensors.clear();

        for (const auto& e : wm) {
            if (e.second != shard) continue;
            auto it = by_name.find(e.first);
            if (it == by_name.end()) continue;
            AdapterTensor t = std::move(it->second);

            names.push_back(t.name);
            int64_t numel = (int64_t)t.data.size();
            if (numel == 0) {
                TensorEntry te; te.name_len = 0; te.block_start = block_id; te.num_blocks = 0;
                tensor_entries.push_back(te);
                continue;
            }
            int num_blocks = (int)((numel + bs - 1) / bs);
            if (num_blocks == 0) continue;
            quant::FormatRegistry::MixBlockPlan plan;
            const std::vector<quant::Format> fmts =
                quant::adapters::allocate_tensor_formats(t.name, numel, t.data.data(),
                                                       bs, fmt, mix, &plan, &t.shape);
            num_blocks = (int)plan.block_starts.size();
            if (num_blocks == 0) continue;

            uint32_t block_start = block_id;
            for (int b = 0; b < num_blocks; b++) {
                int64_t start = plan.block_starts[(size_t)b];
                int n = (int)plan.block_lens[(size_t)b];
                const quant::Format bfmt = fmts[(size_t)b];
                BlockData block;
                block.format = bfmt;
                block.num_weights = (uint32_t)n;
                quantize_block(bfmt, t.data.data() + start, n, block.indices, block.codebook);
                write_block_to_tmp(block);

                FormatBlockEntry fe;
                fe.block_id = block_id++;
                fe.format = (uint8_t)bfmt;
                fe.cb_bytes = (uint32_t)block.codebook.size();
                ft_entries.push_back(fe);
                total_bytes += (double)quant::block_claimed_bytes(bfmt, (uint32_t)n);
                total_weights += n;
            }
            TensorEntry te;
            te.name_len = (uint16_t)t.name.size();
            te.block_start = block_start;
            te.num_blocks = (uint32_t)num_blocks;
            tensor_entries.push_back(te);

            if (cfg.verbose)
                std::printf("  %-48s blocks=%-5d bpw~%.2f raw=%lldKB\n",
                            t.name.c_str(), num_blocks,
                            estimate_mixed_bpw(numel, bs, eff_bpw),
                            (long long)((numel * 4) / 1024));
        }
    }

    block_tmp.flush();
    block_tmp.close();

    writer.write_format_table(ft_entries);
    writer.write_tensor_table(tensor_entries, names);
    {
        std::ifstream bt(tmp_path, std::ios::binary);
        std::vector<char> buf(1 << 16);
        while (bt) {
            bt.read(buf.data(), (std::streamsize)buf.size());
            std::streamsize got = bt.gcount();
            if (got > 0) writer.write_raw(buf.data(), (size_t)got);
        }
        bt.close();
    }
    writer.close();
    std::remove(tmp_path.c_str());

    if (cfg.verbose && total_weights > 0) {
        std::printf("  achieved avg bpw = %.3f (actual stored bytes across %lld weights, base %s)\n",
                    total_bytes * 8.0 / (double)total_weights, (long long)total_weights,
                    mix ? mix->name.c_str() : quant::format_name(fmt));
        std::uintmax_t file_bytes = 0;
        std::error_code ec;
        file_bytes = std::filesystem::file_size(cfg.output_path, ec);
        if (!ec && file_bytes > 0)
            std::printf("  disk bpw = %.3f (file %s, %.1f MB incl. format table + block headers)\n",
                        (double)file_bytes * 8.0 / (double)total_weights,
                        cfg.output_path.c_str(), (double)file_bytes / (1024.0 * 1024.0));
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "QUANT IMPORT — Auto-detect format & import to a single-format QUANT file\n\n"
            "Usage:\n"
            "  quant_import --input <path> --output <out.quant> [options]\n\n"
            "Options:\n"
            "  --input <path>        Input model path (any supported format, or a\n"
            "                        sharded safetensors directory / index.json)\n"
            "  --output <path>       Output QUANT file\n"
            "  --format <name>       Quantization format for ALL blocks\n"
            "                        (default: QG_MX_4_5, adaptive QG_MX)\n"
            "                        Singles: Q1/Q2/Q4/Q8/Q16/Q32, QG1/QG2/QG4/QG8/QG16,\n"
            "                        Q1_5, QG_1_5, Q2. Compounds: QG_MX_3_5/4_5/6_5/8_5/\n"
            "                        12_5/16_5/24_5 (adaptive QG_MX), QG_MX_3_5 (3.5 BPW)\n"
            "                        and QG_MX_4_5 (4.5 BPW) mix member\n"
            "                        formats adaptively by measured benefit per byte\n"
            "                        under a hard BPW budget (never exceeded).\n"
            "  --bpw <float>         Named MIX profile selector (legacy shorthand):\n"
            "                        1.75 -> QG_MX_3_5 (wire spend 3.78125 BPW),\n"
            "                        2.0 -> QG_MX_4_5 (wire spend 4.5 BPW).\n"
            "                        Other values map to the nearest single\n"
            "                        format at its wire BPW. NOTE: the MIX profiles\n"
            "                        spend their wire BPW, not the shorthand number\n"
            "                        (target_bpw is set to the wire spend below).\n"
            "                        (default: 2.0 -> QG_MX_4_5)\n"
            "  --block-size <N>      Block size (default: 256)\n"
            "  --verbose             Print per-tensor stats\n"
            "  -h, --help            Show this help\n\n"
            "Formats: QG_MX_4_5 (4.5, adaptive QG_MX), QG_MX_3_5 (3.5,\n"
            "         adaptive QG_MX), Q2 (2.0), Q1_5 (1.5),\n"
            "         Q1 (1.0), QG2 (2.625), QG4 (4.5), QG8 (8.5),\n"
            "         QG16 (16.0), Q32 (32.0), + all GRP/single variants,\n"
            "         QG_MX_* (QG_MX) compounds\n"
            "Supported input formats (auto-detected):\n"
            "  GGUF, Safetensors (single file OR sharded dir/index.json),\n"
            "  raw FP32/FP16, raw FP8 (E4M3/E5M2), existing .quant\n");
        return 0;
    }

    BridgeConfig cfg;
    std::fprintf(stderr, "[A] cfg constructed\n");
    // Default format: QG_MX_4_5 — adaptive QG_MX.
    // --bpw 1.75 selects QG_MX_3_5 (adaptive QG_MX).
    cfg.format = quant::Format::Q4;
    cfg.compound = quant::RegFormat::QG_MX_4_5;
    cfg.target_bpw = 2.0f;
    cfg.block_size = 256;
    std::string input_path;
    bool bpw_given = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)   input_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) cfg.output_path = argv[++i];
        else if (strcmp(argv[i], "--bpw") == 0 && i + 1 < argc) {
            float v = quant::cli_parse::parse_float("--bpw", argv[++i]);
            if (!(v >= 0.5f) || !(v <= 32.0f)) {
                std::fprintf(stderr, "Error: --bpw must be in [0.5, 32], got '%s'\n", argv[i]);
                return 2;
            }
            cfg.target_bpw = v; bpw_given = true;
        }
        else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            quant::Format f;
            quant::RegFormat comp = quant::format_to_regformat(quant::Format::Q2);
            if (!parse_format_name(argv[++i], f, comp)) {
                std::fprintf(stderr, "Error: unknown format '%s'\n", argv[i]);
                return 1;
            }
            cfg.format = f;
            cfg.compound = comp;
            cfg.target_bpw = quant::format_bpw(f);
            bpw_given = false;
        }
        else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            int bs = quant::cli_parse::parse_int("--block-size", argv[++i]);
            if (bs < 32 || bs > 8192) {
                std::fprintf(stderr, "Error: --block-size must be in [32, 8192], got '%s'\n", argv[i]);
                return 2;
            }
            cfg.block_size = bs;
        }
        else if (strcmp(argv[i], "--verbose") == 0) cfg.verbose = true;
    }

    if (bpw_given) {
        if (std::fabs(cfg.target_bpw - 1.75f) < 0.01f) {
            cfg.format = quant::Format::Q2;
            cfg.compound = quant::RegFormat::QG_MX_3_5;   // QG_MX_3_5 via --bpw 1.75
            cfg.target_bpw = quant::format_bpw(quant::Format::QG_MX_3_5);  // wire 3.78125
        } else if (std::fabs(cfg.target_bpw - 2.0f) < 0.01f) {
            cfg.format = quant::Format::Q4;
            cfg.compound = quant::RegFormat::QG_MX_4_5;  // QG_MX_4_5 via --bpw 2.0
            cfg.target_bpw = quant::format_bpw(quant::Format::QG_MX_4_5);  // wire 4.5
        } else {
            cfg.format = nearest_format_for_bpw(cfg.target_bpw);
            cfg.compound = quant::format_to_regformat(cfg.format);
            cfg.target_bpw = quant::format_bpw(cfg.format);
        }
    }

    if (input_path.empty() || cfg.output_path.empty()) {
        std::fprintf(stderr, "Error: --input and --output are required.\n");
        return 1;
    }

    // Sharded safetensors model: directory or index.json.
    std::fprintf(stderr, "[B] args parsed, input=%s\n", input_path.c_str());
    bool sharded = false;
    if (ends_with(input_path, ".json")) {
        sharded = true;
    } else {
        std::ifstream probe(input_path, std::ios::binary);
        if (!probe) {
            // Not a readable file -> assume a model directory with an index.
            sharded = true;
        }
    }
    std::fprintf(stderr, "[C] sharded=%d\n", (int)sharded);

    bool ok = false;
    if (sharded) {
        ok = import_sharded(input_path, cfg);
    } else {
        ExternalFormat fmt = detect_format(input_path);
        std::fprintf(stdout, "Detected format: %s\n", external_format_name(fmt));
        switch (fmt) {
            case ExternalFormat::GGUF:         ok = gguf_to_quant(input_path, cfg);    break;
            case ExternalFormat::SAFETENSORS:  ok = safetensors_to_quant(input_path, cfg); break;
            case ExternalFormat::QUANT:          ok = ptq_requant_quant(input_path, cfg); break;
            case ExternalFormat::RAW_FP16:
            case ExternalFormat::RAW_FP8_E4M3:
            case ExternalFormat::RAW_FP8_E5M2:
            case ExternalFormat::RAW_FP32:
            default:                           ok = ptq_raw(input_path, fmt, cfg);   break;
        }
    }

    if (!ok) {
        std::fprintf(stderr, "Error: import failed for %s\n", input_path.c_str());
        return 1;
    }
    const quant::MixDescriptor* mix = find_mix_descriptor(cfg.compound);
    const float eff_bpw = mix ? mix->effective_bpw : quant::format_bpw(cfg.format);
    std::fprintf(stdout, "Success: imported %s (%s, %.2f BPW) -> %s\n",
                 input_path.c_str(),
                 mix ? mix->name.c_str() : quant::format_name(cfg.format),
                 eff_bpw, cfg.output_path.c_str());
    return 0;
}
