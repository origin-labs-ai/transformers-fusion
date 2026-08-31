// quant_chat.cpp — terminal chat with Ornith-1.0-9B @ 2 BPW (QUANT_Q1_G) QUANT model
#include "quant/qwen35_engine.h"
#include "quant/qwen35_tokenizer.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>

static double now_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: quant_chat <model_dir> <model.quant> [--temp T] [--topk K] [--max-new N]\n");
        return 1;
    }
    std::string model_dir = argv[1];
    std::string quant_path = argv[2];
    float temp = 0.7f;
    int topk = 50;
    int max_new = 512;
    bool raw = false;
    const char* trace_out = nullptr;
    const char* h_out = nullptr;
    for (int i = 3; i < argc; i++) {
        if (std::strcmp(argv[i], "--temp") == 0 && i + 1 < argc) temp = (float)std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--topk") == 0 && i + 1 < argc) topk = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--max-new") == 0 && i + 1 < argc) max_new = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--raw") == 0) raw = true;
        else if (std::strcmp(argv[i], "--trace-out") == 0 && i + 1 < argc) trace_out = argv[++i];
        else if (std::strcmp(argv[i], "--h-out") == 0 && i + 1 < argc) h_out = argv[++i];
    }

    quant::Qwen35Tokenizer tok;
    if (!tok.load_from_dir(model_dir)) {
        std::fprintf(stderr, "failed to load tokenizer from %s\n", model_dir.c_str());
        return 1;
    }
    std::fprintf(stderr, "tokenizer loaded (%d vocab)\n", tok.vocab_size());

    quant::Qwen35Engine eng;
    double t0 = now_s();
    if (!eng.load(quant_path)) {
        std::fprintf(stderr, "failed to load engine from %s\n", quant_path.c_str());
        return 1;
    }
    std::fprintf(stderr, "engine loaded in %.1fs\n", now_s() - t0);

    std::vector<quant::Qwen35Message> history;
    history.push_back({"system", "You are Ornith, a helpful assistant created by Alibaba Cloud. "
                                "You are a helpful, harmless and faithful assistant.", ""});

    std::vector<int> all_ids;
    size_t shown = 0;
    std::string line;
    std::printf("> ");
    std::fflush(stdout);

    while (std::getline(std::cin, line)) {
        if (line == "exit" || line == "quit") break;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) { std::printf("> "); std::fflush(stdout); continue; }

        history.push_back({"user", line, ""});
        std::vector<int> ids;
        if (raw) {
            ids = tok.encode(line);
        } else {
            ids = tok.apply_chat_template(history, true);
        }
        history.push_back({"assistant", "", ""}); // will be filled below

        double t1 = now_s();
        eng.reset();
        eng.prefill(ids);
        std::fprintf(stderr, "\n[prefill %zu tok in %.1fs]\n", ids.size(), now_s() - t1);

        if (trace_out && ids.size() == 1) {
            FILE* f = std::fopen(trace_out, "wb");
            if (f) {
                std::fwrite(eng.trace(0), 4, 4096, f);
                std::fclose(f);
                std::fprintf(stderr, "[trace0 written: %s]\n", trace_out);
            }
        }
        if (h_out) {
            FILE* f = std::fopen(h_out, "wb");
            if (f) {
                std::fwrite(eng.last_h().data(), 4, 4096, f);
                std::fclose(f);
                std::fprintf(stderr, "[h written: %s]\n", h_out);
            }
        }

        {
            const float* L = eng.logits();
            float mx = -1e30f; int mxi = 0;
            double s = 0; int nan = 0;
            for (int i = 0; i < eng.vocab(); i++) {
                float v = L[i];
                if (v != v) nan++;
                if (v > mx) { mx = v; mxi = i; }
                s += v * v;
            }
            std::fprintf(stderr, "[logits] argmax=%d mx=%.3f rms=%.3f nan=%d\n", mxi, mx,
                         (float)std::sqrt(s / eng.vocab()), nan);
            std::fprintf(stderr, "[argmax text] %s\n",
                         tok.decode(std::vector<int>{mxi}, true).c_str());
            std::vector<int> top5i;
            for (int t = 0; t < 5; t++) {
                int bi = -1; float bv = -1e30f;
                for (int i = 0; i < eng.vocab(); i++) {
                    bool dup = false;
                    for (int q = 0; q < (int)top5i.size(); q++) if (top5i[q] == i) dup = true;
                    if (!dup && L[i] > bv) { bv = L[i]; bi = i; }
                }
                if (bi >= 0) top5i.push_back(bi);
            }
            for (int q = 0; q < (int)top5i.size(); q++)
                std::fprintf(stderr, "[top%d] %d %s\n", q + 1, top5i[q],
                             tok.decode(std::vector<int>{top5i[q]}, true).c_str());
        }

        all_ids = ids;
        shown = 0;
        int generated = 0;
        bool stop = false;
        while (generated < max_new) {
            int id = eng.sample(temp, topk);
            if (id == 248046 || id == 248044 || id == 248045) { stop = true; break; }
            all_ids.push_back(id);
            eng.append_token(id);
            generated++;

            std::string dec = tok.decode(all_ids, true);
            if (dec.size() > shown) {
                std::fwrite(dec.data() + shown, 1, dec.size() - shown, stdout);
                std::fflush(stdout);
                shown = dec.size();
            }
        }
        if (!stop) { /* max_new reached */ }
        std::fprintf(stderr, "\n[%d tokens in %.1fs]\n", generated, now_s() - t1);

        std::string reply = tok.decode(all_ids, true);
        size_t tp = reply.find("</think>");
        if (tp != std::string::npos) {
            history.back().reasoning_content = reply.substr(0, tp);
            history.back().content = reply.substr(tp + 8);
        } else {
            history.back().content = reply;
        }
        std::printf("\n> ");
        std::fflush(stdout);
    }
    return 0;
}
