// ============================================================================
// quant_import.cpp — CLI: auto-detect format & import -> single-format QUANT file
// ============================================================================
// Usage:
//   quant_import --input <path> --output <out.quant> [--format QUANT_Q1_G]
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
    if (bpw <= 1.25f) return quant::Format::QUANT1;
    if (bpw <= 1.75f) return quant::Format::QUANT_Q0;
    if (bpw <= 3.0f)  return quant::Format::QUANT_Q1_G;
    if (bpw <= 6.0f)  return quant::Format::QUANT4_G;
    if (bpw <= 12.0f) return quant::Format::QUANT8_G;
    if (bpw <= 24.0f) return quant::Format::QUANT16_G;
    return quant::Format::QUANT32;
}

// Parse every QUANT/QUANT single format plus the TWI_MIX (MIX_*) and QUAD_MIX
// (QUAD_*) compound formats. Compounds set `out` to the member wire format
// (highest tier) and `out_compound` to the compound RegFormat id.
static bool parse_format_name(const char* name, quant::Format& out, quant::RegFormat& out_compound) {
    std::string s = to_upper(name);
    bool is_compound = false;

    // ---- QUANT / QUANT singles ----
    if (s == "QUANT1")              { out = quant::Format::QUANT1; }
    else if (s == "QUANT2")         { out = quant::Format::QUANT2; }
    else if (s == "QUANT4")         { out = quant::Format::QUANT4; }
    else if (s == "QUANT8")         { out = quant::Format::QUANT8; }
    else if (s == "QUANT16")        { out = quant::Format::QUANT16; }
    else if (s == "QUANT32")        { out = quant::Format::QUANT32; }
    else if (s == "QUANT1_G")     { out = quant::Format::QUANT1_G; }
    else if (s == "QUANT2_G")     { out = quant::Format::QUANT2_G; }
    else if (s == "QUANT4_G")     { out = quant::Format::QUANT4_G; }
    else if (s == "QUANT8_G")     { out = quant::Format::QUANT8_G; }
    else if (s == "QUANT16_G")    { out = quant::Format::QUANT16_G; }
    else if (s == "QUANT_Q1") { out = quant::Format::QUANT_Q1; }
    else if (s == "QUANT_Q1_G") { out = quant::Format::QUANT_Q1_G; }
    else if (s == "QUANT_Q0")     { out = quant::Format::QUANT_Q0; }
    else if (s == "QUANT_Q0_G") { out = quant::Format::QUANT_Q0_G; }
    // ---- TWI_MIX (two-tier) compounds ---- 
    else if (s == "MIX_QUANT8_QUANT2_01_99")  { out = quant::Format::QUANT8;   out_compound = quant::RegFormat::MIX_QUANT8_QUANT2_01_99; is_compound = true; }
    else if (s == "MIX_QUANT8_QUANT4_05_95")  { out = quant::Format::QUANT8;   out_compound = quant::RegFormat::MIX_QUANT8_QUANT4_05_95; is_compound = true; }
    else if (s == "MIX_QUANT4_QUANT2_10_90")  { out = quant::Format::QUANT4;   out_compound = quant::RegFormat::MIX_QUANT4_QUANT2_10_90; is_compound = true; }
    else if (s == "MIX_QUANT8_QUANT2_10_90")  { out = quant::Format::QUANT8;   out_compound = quant::RegFormat::MIX_QUANT8_QUANT2_10_90; is_compound = true; }
    else if (s == "MIX_QUANT_Q0_QUANT8_05_95") { out = quant::Format::QUANT8;   out_compound = quant::RegFormat::MIX_QUANT_Q0_QUANT8_05_95; is_compound = true; }
    else if (s == "MIX_QUANT16_QUANT4_01_99") { out = quant::Format::QUANT16;  out_compound = quant::RegFormat::MIX_QUANT16_QUANT4_01_99; is_compound = true; }
    else if (s == "MIX_QUANT16_QUANT8_05_95") { out = quant::Format::QUANT16;  out_compound = quant::RegFormat::MIX_QUANT16_QUANT8_05_95; is_compound = true; }
    else if (s == "MIX_QUANT32_QUANT8_01_99") { out = quant::Format::QUANT32;  out_compound = quant::RegFormat::MIX_QUANT32_QUANT8_01_99; is_compound = true; }
    else if (s == "QUANT_MIX_Q0") { out = quant::Format::QUANT2; out_compound = quant::RegFormat::QUANT_TWI_MIX_Q0; is_compound = true; }
    // ---- QUAD_MIX (four-tier) compounds ----
    else if (s == "QUAD_QUANT2_QUANT4_QUANT8_QUANT16") { out = quant::Format::QUANT16; out_compound = quant::RegFormat::QUAD_QUANT2_QUANT4_QUANT8_QUANT16; is_compound = true; }
    else if (s == "QUAD_QUANT4_QUANT8_QUANT16_QUANT32") { out = quant::Format::QUANT32; out_compound = quant::RegFormat::QUAD_QUANT4_QUANT8_QUANT16_QUANT32; is_compound = true; }
    else if (s == "QUANT_MIX_Q1") { out = quant::Format::QUANT4; out_compound = quant::RegFormat::QUANT_QUAD_MIX_Q0; is_compound = true; }
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
            "                        (default: QUANT_MIX_Q1, exactly 2.0 BPW,\n"
            "                        adaptive QUAD_MIX)\n"
            "                        Singles: QUANT1/2/4/8/16/32[_G], QUANT_Q0[_G],\n"
            "                        QUANT_Q1[_G]. Compounds: MIX_* (TWI_MIX),\n"
            "                        QUAD_* (QUAD_MIX), QUANT_MIX_Q0 (1.75 BPW exact)\n"
            "                        and QUANT_MIX_Q1 (2.0 BPW exact) mix member\n"
            "                        formats adaptively by measured benefit per byte\n"
            "                        under a hard BPW budget (never exceeded).\n"
            "  --bpw <float>         Target bits-per-weight. 1.75 -> QUANT_MIX_Q0,\n"
            "                        2.0 -> QUANT_MIX_Q1, other values map to the\n"
            "                        nearest single format (default: 2.0 -> QUANT_MIX_Q1)\n"
            "  --block-size <N>      Block size (default: 256)\n"
            "  --verbose             Print per-tensor stats\n"
            "  -h, --help            Show this help\n\n"
            "Formats: QUANT_MIX_Q1 (2.0, adaptive QUAD_MIX), QUANT_MIX_Q0 (1.75,\n"
            "         adaptive TWI_MIX), QUANT_Q1_G (2.0), QUANT_Q0 (1.5),\n"
            "         QUANT1 (1.0), QUANT2_G (2.625), QUANT4_G (4.5), QUANT8_G (8.5),\n"
            "         QUANT16_G (16.0), QUANT32 (32.0), + all GRP/single variants,\n"
            "         MIX_* (TWI_MIX) and QUAD_* (QUAD_MIX) compounds\n"
            "Supported input formats (auto-detected):\n"
            "  GGUF, Safetensors (single file OR sharded dir/index.json),\n"
            "  raw FP32/FP16, raw FP8 (E4M3/E5M2), existing .quant\n");
        return 0;
    }

    BridgeConfig cfg;
    std::fprintf(stderr, "[A] cfg constructed\n");
    // Default format: QUANT_MIX_Q1 — exactly 2.0 BPW, adaptive QUAD_MIX.
    // --bpw 1.75 selects QUANT_MIX_Q0 (exactly 1.75 BPW, adaptive TWI_MIX).
    cfg.format = quant::Format::QUANT4;
    cfg.compound = quant::RegFormat::QUANT_QUAD_MIX_Q0;
    cfg.target_bpw = 2.0f;
    cfg.block_size = 256;
    std::string input_path;
    bool bpw_given = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)   input_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) cfg.output_path = argv[++i];
        else if (strcmp(argv[i], "--bpw") == 0 && i + 1 < argc) { cfg.target_bpw = (float)std::atof(argv[++i]); bpw_given = true; }
        else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            quant::Format f;
            quant::RegFormat comp = quant::format_to_regformat(quant::Format::QUANT_Q1_G);
            if (!parse_format_name(argv[++i], f, comp)) {
                std::fprintf(stderr, "Error: unknown format '%s'\n", argv[i]);
                return 1;
            }
            cfg.format = f;
            cfg.compound = comp;
            cfg.target_bpw = quant::format_bpw(f);
            bpw_given = false;
        }
        else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) cfg.block_size = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--verbose") == 0) cfg.verbose = true;
    }

    if (bpw_given) {
        if (std::fabs(cfg.target_bpw - 1.75f) < 0.01f) {
            cfg.format = quant::Format::QUANT2;
            cfg.compound = quant::RegFormat::QUANT_TWI_MIX_Q0;   // QUANT_MIX_Q0, 1.75 exact
        } else if (std::fabs(cfg.target_bpw - 2.0f) < 0.01f) {
            cfg.format = quant::Format::QUANT4;
            cfg.compound = quant::RegFormat::QUANT_QUAD_MIX_Q0;  // QUANT_MIX_Q1, 2.0 exact
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
