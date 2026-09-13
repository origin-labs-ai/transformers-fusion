// generate_comparison_visuals.cpp — reads bench_format_comparison.csv (REAL
// measured data) and generates docs/COMPARISON_CHARTS.md with embedded SVGs.
// No hardcoded numbers: every value plotted comes from the CSV produced by
// bench_format_comparison.exe.
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

struct Row {
    std::string dataset, format;
    double bpw = 0, mse = 0, psnr = 0, enc = 0, dec = 0;
};

static bool is_ref(const std::string& f) {
    return f.rfind("[ref]", 0) == 0;
}

static std::string svg_color(const Row& r) {
    if (is_ref(r.format)) return "#d33";
    if (r.format.rfind("MXQ", 0) == 0 || r.format.rfind("Q_MX", 0) == 0 || r.format.rfind("QG_MX", 0) == 0) return "#c6a700";
    if (r.format.find("_G") != std::string::npos) return "#083";
    return "#06c";
}

int main() {
    std::ifstream in("bench_format_comparison.csv");
    if (!in.is_open()) { std::cerr << "ERROR: run bench_format_comparison.exe first (CSV missing)\n"; return 1; }
    std::string line;
    std::getline(in, line); // header
    std::vector<Row> rows;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell; std::vector<std::string> c;
        while (std::getline(ss, cell, ',')) c.push_back(cell);
        if (c.size() < 5) continue;
        Row r; r.dataset = c[0]; r.format = c[1];
        try {
            r.bpw = std::stod(c[2]); r.mse = std::stod(c[3]); r.psnr = std::stod(c[4]);
            if (c.size() > 7) { r.enc = std::stod(c[5]); r.dec = std::stod(c[6]); }
        } catch (...) { continue; }
        rows.push_back(r);
    }
    if (rows.empty()) { std::cerr << "ERROR: CSV has no rows\n"; return 2; }

    std::ofstream f("docs/COMPARISON_CHARTS.md");
    if (!f.is_open()) { std::cerr << "ERROR: cannot open docs/COMPARISON_CHARTS.md\n"; return 3; }

    f << "# Transcender vs Industrial Baselines — REAL Measured Charts\n\n";
    f << "> Generated from `bench_format_comparison.csv` by `tools/generate_comparison_visuals.cpp`.\n";
    f << "> Every number below comes from a measured round-trip of the production codec.\n";
    f << "> BPW ironclad: each format stores EXACTLY the BPW in its name.\n\n";

    const char* dss[2] = { "gaussian", "real" };
    for (int di = 0; di < 2; ++di) {
        const std::string ds = dss[di];
        std::vector<Row> rs;
        for (auto& r : rows) if (r.dataset == ds) rs.push_back(r);
        double minb = 1e9, maxb = 0, minp = 1e9, maxp = -1e9;
        for (auto& r : rs) {
            minb = std::min(minb, r.bpw); maxb = std::max(maxb, r.bpw);
            minp = std::min(minp, r.psnr); maxp = std::max(maxp, r.psnr);
        }
        const int W = 900, H = 460, M = 56;
        auto X = [&](double b) { return M + (b - minb) / std::max(1e-9, maxb - minb) * (W - 2 * M); };
        auto Y = [&](double p) { return H - M - (p - minp) / std::max(1e-9, maxp - minp) * (H - 2 * M); };

        f << "## " << (di == 0 ? "Gaussian weights" : "Real trained weights") << " — PSNR vs BPW\n\n";
        f << "<svg xmlns='http://www.w3.org/2000/svg' width='" << W << "' height='" << H << "' font-family='monospace' font-size='10'>\n";
        f << "<rect width='100%' height='100%' fill='#fafafa'/>\n";
        for (int g = 0; g <= 4; ++g) {
            double p = minp + (maxp - minp) * g / 4.0;
            int y = (int)Y(p);
            f << "<line x1='" << M << "' y1='" << y << "' x2='" << W - M << "' y2='" << y << "' stroke='#ddd'/>\n";
            f << "<text x='4' y='" << y + 3 << "'>" << (int)p << " dB</text>\n";
            double b = minb + (maxb - minb) * g / 4.0;
            int x = (int)X(b);
            f << "<text x='" << x - 12 << "' y='" << H - M + 14 << "'>" << b << " bpw</text>\n";
        }
        f << "<line x1='" << M << "' y1='" << H - M << "' x2='" << W - M << "' y2='" << H - M << "' stroke='#333'/>\n";
        f << "<line x1='" << M << "' y1='" << M / 2 << "' x2='" << M << "' y2='" << H - M << "' stroke='#333'/>\n";
        for (auto& r : rs) {
            const std::string col = svg_color(r);
            const bool ref = is_ref(r.format);
            const int rad = ref ? 4 : 3;
            f << "<circle cx='" << (int)X(r.bpw) << "' cy='" << (int)Y(r.psnr) << "' r='" << rad << "' fill='" << col << "'" << (ref ? " stroke='#900'" : "") << "/>\n";
            if (ref || r.psnr > maxp * 0.55)
                f << "<text x='" << (int)X(r.bpw) + 5 << "' y='" << (int)Y(r.psnr) - 4 << "' fill='" << col << "'>" << r.format << "</text>\n";
        }
        f << "<rect x='" << W - 260 << "' y='" << 8 << "' width='252' height='52' fill='#fff' stroke='#ccc'/>\n";
        f << "<circle cx='" << W - 246 << "' cy='20' r='3' fill='#06c'/><text x='" << W - 238 << "' y='23'>Transcender plain</text>\n";
        f << "<circle cx='" << W - 246 << "' cy='34' r='3' fill='#083'/><text x='" << W - 238 << "' y='37'>Transcender GRP/K_G</text>\n";
        f << "<circle cx='" << W - 246 << "' cy='48' r='3' fill='#c6a700'/><text x='" << W - 238 << "' y='51'>Transcender MXQ mix</text>\n";
        f << "<circle cx='" << W - 140 << "' cy='20' r='4' fill='#d33' stroke='#900'/><text x='" << W - 132 << "' y='23'>[ref] industrial</text>\n";
        f << "</svg>\n\n";

        // Same-BPW industrial head-to-head table computed from CSV.
        f << "| Transcender | BPW | PSNR dB | Competitor | BPW | PSNR dB | Delta | Verdict |\n|---|---|---|---|---|---|---|---|\n";
        struct PairDef { const char* a; const char* b; };
        PairDef pairs[] = {
            {"Q16", "[ref] IEEE FP16"},
            {"QG_MX_16.5", "[ref] IEEE FP16"},
            {"QG_8.5", "[ref] GGUF Q8_0"},
            {"Q_MX_8.5", "[ref] GGUF Q8_0"},
            {"QG_6.5", "[ref] GGUF Q6_K"},
            {"QG_4.5", "[ref] GGUF Q4_K"},
            {"QG_MX_4.5", "[ref] GGUF Q4_K"},
            {"QG1", "[ref] BitNet b1.58"},
            {"QG1", "[ref] Binary 1-bit"},
            {"Q8_K_M", "[ref] INT8 uniform"},
        };
        auto find_row = [&](const std::string& n) -> const Row* {
            for (auto& r : rs) if (r.format == n) return &r;
            return nullptr;
        };
        for (auto& pd : pairs) {
            const Row* a = find_row(pd.a);
            const Row* b = find_row(pd.b);
            if (!a || !b) continue;
            double dlt = a->psnr - b->psnr;
            f << "|" << a->format << "|" << a->bpw << "|" << a->psnr << "|"
              << b->format << "|" << b->bpw << "|" << b->psnr << "|"
              << (dlt >= 0 ? "+" : "") << dlt << " dB|"
              << (dlt >= 0 ? "**WIN**" : "LOSS") << "|\n";
        }
        f << "\n";
    }

    f << "## Competitor landscape\n\nSee `docs/COMPETITOR_ANALYSIS.md` for the researched mapping\n";
    f << "(GGUF K-quants/IQ, GPTQ, AWQ, SmoothQuant, SpQR, SqueezeLLM, AQLM,\n";
    f << "QuIP#, EXL2/EXL3, BitNet b1.58, BinaryNet) with their published metrics\n";
    f << "and honest notes on metric differences (weight-PSNR vs perplexity).\n";
    std::cout << "docs/COMPARISON_CHARTS.md generated (" << rows.size() << " CSV rows read)\n";
    return 0;
}
