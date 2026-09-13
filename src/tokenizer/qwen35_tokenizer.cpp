#include "quant/qwen35_tokenizer.h"
#include "quant/json_parser.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cstdio>

namespace quant {

int Qwen35Tokenizer::utf8_decode(const std::string& s, size_t& pos) {
    unsigned char c = (unsigned char)s[pos];
    if (c < 0x80) { pos++; return c; }
    if ((c & 0xE0) == 0xC0 && pos + 1 < s.size()) {
        unsigned char c2 = (unsigned char)s[pos + 1];
        if ((c2 & 0xC0) == 0x80) { pos += 2; return ((c & 0x1F) << 6) | (c2 & 0x3F); }
    }
    if ((c & 0xF0) == 0xE0 && pos + 2 < s.size()) {
        unsigned char c2 = (unsigned char)s[pos + 1], c3 = (unsigned char)s[pos + 2];
        if ((c2 & 0xC0) == 0x80 && (c3 & 0xC0) == 0x80) {
            pos += 3;
            return ((c & 0x0F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
        }
    }
    if ((c & 0xF8) == 0xF0 && pos + 3 < s.size()) {
        unsigned char c2 = (unsigned char)s[pos + 1], c3 = (unsigned char)s[pos + 2], c4 = (unsigned char)s[pos + 3];
        if ((c2 & 0xC0) == 0x80 && (c3 & 0xC0) == 0x80 && (c4 & 0xC0) == 0x80) {
            pos += 4;
            return ((c & 0x07) << 18) | ((c2 & 0x3F) << 12) | ((c3 & 0x3F) << 6) | (c4 & 0x3F);
        }
    }
    pos++;
    return 0xFFFD;
}

void Qwen35Tokenizer::utf8_encode(std::string& out, int cp) {
    if (cp < 0x80) {
        out += (char)cp;
    } else if (cp < 0x800) {
        out += (char)(0xC0 | (cp >> 6));
        out += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char)(0xE0 | (cp >> 12));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    } else {
        out += (char)(0xF0 | (cp >> 18));
        out += (char)(0x80 | ((cp >> 12) & 0x3F));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    }
}

bool Qwen35Tokenizer::is_letter(int cp) {
    if (cp >= 0x41 && cp <= 0x5A) return true;
    if (cp >= 0x61 && cp <= 0x7A) return true;
    if (cp >= 0x300 && cp <= 0x36F) return true;
    if (cp >= 0x370 && cp <= 0x3FF) return true;
    if (cp >= 0x400 && cp <= 0x52F) return true;
    if (cp >= 0x590 && cp <= 0x5FF) return true;
    if (cp >= 0x600 && cp <= 0x6FF) return true;
    if (cp >= 0x750 && cp <= 0x77F) return true;
    if (cp >= 0x900 && cp <= 0x9FF) return true;
    if (cp >= 0xA00 && cp <= 0xAFF) return true;
    if (cp >= 0xB00 && cp <= 0xBFF) return true;
    if (cp >= 0xC00 && cp <= 0xCFF) return true;
    if (cp >= 0xD00 && cp <= 0xDFF) return true;
    if (cp >= 0xE00 && cp <= 0xEFF) return true;
    if (cp >= 0x1000 && cp <= 0x109F) return true;
    if (cp >= 0x10A0 && cp <= 0x10FF) return true;
    if (cp >= 0x1100 && cp <= 0x11FF) return true;
    if (cp >= 0x1200 && cp <= 0x137F) return true;
    if (cp >= 0x13A0 && cp <= 0x13FF) return true;
    if (cp >= 0x1DC0 && cp <= 0x1DFF) return true;
    if (cp >= 0x1E00 && cp <= 0x1EFF) return true;
    if (cp >= 0x1F00 && cp <= 0x1FFF) return true;
    if (cp >= 0x20D0 && cp <= 0x20FF) return true;
    if (cp >= 0x2C60 && cp <= 0x2C7F) return true;
    if (cp >= 0x2E80 && cp <= 0x2EFF) return true;
    if (cp >= 0x3040 && cp <= 0x30FF) return true;
    if (cp >= 0x3130 && cp <= 0x318F) return true;
    if (cp >= 0x31A0 && cp <= 0x31BF) return true;
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    if (cp >= 0xA000 && cp <= 0xA48F) return true;
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    if (cp >= 0xFB00 && cp <= 0xFB4F) return true;
    if (cp >= 0xFB50 && cp <= 0xFDFF) return true;
    if (cp >= 0xFE20 && cp <= 0xFE2F) return true;
    if (cp >= 0xFE70 && cp <= 0xFEFF) return true;
    if (cp >= 0xFF21 && cp <= 0xFF3A) return true;
    if (cp >= 0xFF41 && cp <= 0xFF5A) return true;
    if (cp >= 0x10400 && cp <= 0x1044F) return true;
    if (cp >= 0x1D400 && cp <= 0x1D7FF) return true;
    if (cp >= 0x2F800 && cp <= 0x2FA1F) return true;
    return false;
}

bool Qwen35Tokenizer::is_digit(int cp) {
    if (cp >= 0x30 && cp <= 0x39) return true;
    if (cp >= 0x660 && cp <= 0x669) return true;
    if (cp >= 0x6F0 && cp <= 0x6F9) return true;
    if (cp >= 0x966 && cp <= 0x96F) return true;
    if (cp >= 0x9E6 && cp <= 0x9EF) return true;
    if (cp >= 0xA66 && cp <= 0xA6F) return true;
    if (cp >= 0xAE6 && cp <= 0xAEF) return true;
    if (cp >= 0xB66 && cp <= 0xB6F) return true;
    if (cp >= 0xBE6 && cp <= 0xBEF) return true;
    if (cp >= 0xC66 && cp <= 0xC6F) return true;
    if (cp >= 0xCE6 && cp <= 0xCEF) return true;
    if (cp >= 0xD66 && cp <= 0xD6F) return true;
    if (cp >= 0xE50 && cp <= 0xE59) return true;
    if (cp >= 0xED0 && cp <= 0xED9) return true;
    if (cp >= 0xF20 && cp <= 0xF29) return true;
    if (cp >= 0xFF10 && cp <= 0xFF19) return true;
    if (cp >= 0x104A0 && cp <= 0x104A9) return true;
    return false;
}

bool Qwen35Tokenizer::is_ws(int cp) {
    if (cp >= 0x09 && cp <= 0x0D) return true;
    if (cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680) return true;
    if (cp >= 0x2000 && cp <= 0x200A) return true;
    if (cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000) return true;
    return false;
}

void Qwen35Tokenizer::build_byte_map() {
    byte_chars_.resize(256);
    for (int b = 0; b < 256; b++) {
        if (b >= 33 && b <= 126) {
            byte_chars_[(size_t)b] = std::string(1, (char)b);
        } else {
            std::string tmp;
            utf8_encode(tmp, 256 + b);
            byte_chars_[(size_t)b] = tmp;
        }
    }
}

static std::string json_str_value(const JsonValue& v) {
    if (v.is_string()) return v.as_string();
    return "";
}

bool Qwen35Tokenizer::load_from_dir(const std::string& model_dir) {
    build_byte_map();

    std::ifstream f(model_dir + "/tokenizer.json", std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();

    std::string err;
    JsonValue root = JsonValue::parse(text, &err);
    if (!err.empty() || root.is_null()) return false;

    const JsonValue& added = root["added_tokens"];
    for (size_t i = 0; i < added.arr.size(); i++) {
        int id = (int)added[i]["id"].as_int();
        std::string content = json_str_value(added[i]["content"]);
        bool special = added[i]["special"].as_bool();
        if (content.empty()) continue;
        if (special) {
            special_ids_.push_back(id);
            special_tokens_.push_back(content);
            special_by_id_[id] = content;
        }
        if (!token_to_id_.count(content)) token_to_id_[content] = id;
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
    std::sort(special_ids_.begin(), special_ids_.end());
    special_ids_.erase(std::unique(special_ids_.begin(), special_ids_.end()), special_ids_.end());

    const JsonValue& model = root["model"];
    const JsonValue& vocab = model["vocab"];
    id_to_token_.resize(vocab.obj.size());
    for (const auto& kv : vocab.obj) {
        int id = (int)kv.second.as_int();
        if (id >= 0 && id < (int)id_to_token_.size()) id_to_token_[(size_t)id] = kv.first;
        token_to_id_[kv.first] = id;
    }

    const JsonValue& merges = model["merges"];
    for (size_t i = 0; i < merges.arr.size(); i++) {
        std::string a, b;
        const JsonValue& mv = merges[i];
        if (mv.is_array() && mv.arr.size() == 2) {
            a = mv.arr[0].as_string();
            b = mv.arr[1].as_string();
        } else {
            std::string m = mv.as_string();
            size_t sp = m.find(' ');
            if (sp == std::string::npos || sp == 0 || sp + 1 >= m.size()) continue;
            a = m.substr(0, sp);
            b = m.substr(sp + 1);
        }
        auto ia = token_to_id_.find(a);
        auto ib = token_to_id_.find(b);
        if (ia == token_to_id_.end() || ib == token_to_id_.end()) continue;
        merges_.push_back({ ia->second, ib->second });
    }

    std::fprintf(stderr, "[tok] vocab=%zu merges=%zu specials=%zu\n",
                 id_to_token_.size(), merges_.size(), special_ids_.size());
    return !id_to_token_.empty() && !merges_.empty();
}

bool Qwen35Tokenizer::is_special(int id) const {
    return special_by_id_.count(id) != 0;
}

static bool starts_with_ci(const std::string& s, size_t pos, const char* t) {
    for (size_t i = 0; t[i]; i++) {
        if (pos + i >= s.size()) return false;
        char a = s[pos + i], b = t[i];
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
        if (a != b) return false;
    }
    return true;
}

std::vector<std::string> Qwen35Tokenizer::pretokenize(const std::string& text) const {
    std::vector<std::string> out;
    size_t pos = 0;
    const size_t n = text.size();

    auto skip_letters = [&](size_t p) {
        while (p < n) {
            size_t q = p;
            int cp = utf8_decode(text, q);
            if (!is_letter(cp)) break;
            p = q;
        }
        return p;
    };
    auto skip_crlf = [&](size_t p) {
        while (p < n) {
            size_t q = p;
            int cp = utf8_decode(text, q);
            if (cp != '\r' && cp != '\n') break;
            p = q;
        }
        return p;
    };

    while (pos < n) {
        size_t p2 = pos;
        int cp0 = utf8_decode(text, p2);
        int tok_start = -1, tok_end = -1;

        if (tok_start < 0) {
            const char* ctr[7] = { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };
            int clen[7] = { 2, 2, 3, 3, 2, 3, 2 };
            for (int k = 0; k < 7; k++) {
                if (starts_with_ci(text, pos, ctr[k])) {
                    tok_start = (int)pos;
                    tok_end = (int)(pos + clen[k]);
                    break;
                }
            }
        }

        if (tok_start < 0) {
            if (is_letter(cp0)) {
                tok_start = (int)pos;
                tok_end = (int)skip_letters(pos);
            } else if (cp0 != '\r' && cp0 != '\n' && !is_digit(cp0)) {
                size_t after = p2;
                int cp1 = (after < n) ? utf8_decode(text, after) : -1;
                if (cp1 >= 0 && is_letter(cp1)) {
                    tok_start = (int)pos;
                    tok_end = (int)skip_letters(after);
                }
            }
        }

        if (tok_start < 0 && is_digit(cp0)) {
            tok_start = (int)pos;
            tok_end = (int)p2;
        }

        if (tok_start < 0) {
            size_t q = pos;
            if (cp0 == ' ') q = p2;
            size_t start = q;
            while (q < n) {
                size_t qq = q;
                int cp = utf8_decode(text, qq);
                if (is_ws(cp) || is_letter(cp) || is_digit(cp)) break;
                q = qq;
            }
            if (q > start) {
                tok_start = (int)pos;
                tok_end = (int)skip_crlf(q);
            }
        }

        if (tok_start < 0) {
            size_t q = pos;
            while (q < n) {
                size_t qq = q;
                int cp = utf8_decode(text, qq);
                if (!is_ws(cp)) break;
                q = qq;
            }
            if (q > pos) {
                tok_start = (int)pos;
                tok_end = (int)q;
            }
        }

        if (tok_start < 0) {
            tok_start = (int)pos;
            tok_end = (int)p2;
        }

        out.push_back(text.substr((size_t)tok_start, (size_t)(tok_end - tok_start)));
        pos = (size_t)tok_end;
    }
    return out;
}

std::vector<int> Qwen35Tokenizer::bpe_encode(const std::string& word) const {
    std::vector<int> seq;
    for (size_t i = 0; i < word.size(); i++) {
        unsigned char b = (unsigned char)word[i];
        const std::string& t = byte_chars_[(size_t)b];
        auto it = token_to_id_.find(t);
        if (it != token_to_id_.end()) {
            seq.push_back(it->second);
        } else {
            seq.push_back((int)b);
        }
    }
    std::unordered_map<uint64_t, int> rank;
    for (size_t i = 0; i < merges_.size(); i++) {
        rank[((uint64_t)(uint32_t)merges_[i].first << 32) | (uint32_t)merges_[i].second] = (int)i;
    }
    std::unordered_map<uint64_t, int> target;
    for (const auto& m : merges_) {
        const std::string& ta = id_to_token_[(size_t)m.first];
        const std::string& tb = id_to_token_[(size_t)m.second];
        auto it = token_to_id_.find(ta + tb);
        if (it != token_to_id_.end()) {
            target[((uint64_t)(uint32_t)m.first << 32) | (uint32_t)m.second] = it->second;
        }
    }
    while (seq.size() > 1) {
        int best_i = -1, best_r = INT32_MAX;
        for (size_t i = 0; i + 1 < seq.size(); i++) {
            uint64_t key = ((uint64_t)(uint32_t)seq[i] << 32) | (uint32_t)seq[i + 1];
            auto it = rank.find(key);
            if (it != rank.end() && it->second < best_r) {
                best_r = it->second;
                best_i = (int)i;
            }
        }
        if (best_i < 0) break;
        uint64_t key = ((uint64_t)(uint32_t)seq[(size_t)best_i] << 32) | (uint32_t)seq[(size_t)best_i + 1];
        seq[(size_t)best_i] = target.at(key);
        seq.erase(seq.begin() + best_i + 1);
    }
    return seq;
}

std::vector<int> Qwen35Tokenizer::encode(const std::string& text) {
    std::vector<int> ids;
    size_t pos = 0;
    const size_t n = text.size();
    while (pos < n) {
        bool matched = false;
        for (const auto& st : special_tokens_) {
            if (st.size() > n - pos) continue;
            if (text.compare(pos, st.size(), st) == 0) {
                ids.push_back(token_to_id_.at(st));
                pos += st.size();
                matched = true;
                break;
            }
        }
        if (matched) continue;

        // find the next special token boundary so pretokenize never swallows one
        size_t end = n;
        for (const auto& st : special_tokens_) {
            size_t p = text.find(st, pos);
            if (p != std::string::npos && p < end) end = p;
        }
        if (end > pos) {
            std::vector<std::string> toks = pretokenize(text.substr(pos, end - pos));
            for (const auto& t : toks) {
                std::vector<int> sub = bpe_encode(t);
                ids.insert(ids.end(), sub.begin(), sub.end());
            }
            pos = end;
        } else {
            pos++;
        }
    }
    return ids;
}

std::string Qwen35Tokenizer::token_to_bytes(const std::string& token) const {
    std::unordered_map<std::string, unsigned char> rev;
    for (int b = 0; b < 256; b++) rev[byte_chars_[(size_t)b]] = (unsigned char)b;
    std::string out;
    size_t pos = 0;
    while (pos < token.size()) {
        bool found = false;
        // Try multi-byte UTF-8 sequences first (2-byte matches from byte_chars_)
        for (int len = 2; len >= 1; len--) {
            if (pos + (size_t)len > token.size()) continue;
            auto it = rev.find(token.substr(pos, (size_t)len));
            if (it != rev.end()) {
                out += (char)it->second;
                pos += (size_t)len;
                found = true;
                break;
            }
        }
        if (found) continue;
        // Handle <0xHH> hex escape sequences
        if (token.compare(pos, 3, "<0x") == 0 && pos + 5 <= token.size() &&
            token[pos + 5] == '>') {
            unsigned int v;
            if (sscanf(token.c_str() + pos + 3, "%2x", &v) == 1) {
                out += (char)v;
                pos += 6;
                continue;
            }
        }
        // Single character not in byte_chars_ — use raw byte value
        // This handles Qwen's vocab where printable ASCII chars are stored as-is
        unsigned char c = (unsigned char)token[pos];
        if (c >= 33 && c <= 126) {
            out += (char)c;
            pos++;
            continue;
        }
        // Fallback: emit raw bytes of the token string
        out += token.substr(pos);
        break;
    }
    return out;
}

std::string Qwen35Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= (int)id_to_token_.size()) continue;
        const std::string& tok = id_to_token_[(size_t)id];
        auto sp = special_by_id_.find(id);
        if (sp != special_by_id_.end()) {
            if (!skip_special) out += sp->second;
            continue;
        }
        out += token_to_bytes(tok);
    }
    return out;
}

static std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\n' || s[a] == '\r' || s[a] == '\t')) a++;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\n' || s[b - 1] == '\r' || s[b - 1] == '\t')) b--;
    return s.substr(a, b - a);
}

std::vector<int> Qwen35Tokenizer::apply_chat_template(
    const std::vector<Qwen35Message>& messages, bool add_generation_prompt) {
    std::string out;
    for (size_t i = 0; i < messages.size(); i++) {
        const Qwen35Message& m = messages[i];
        std::string content = trim(m.content);
        if (m.role == "system") {
            if (i == 0) out += "<|im_start|>system\n" + content + "<|im_end|>\n";
        } else if (m.role == "user") {
            out += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (m.role == "assistant") {
            std::string reasoning = m.reasoning_content;
            std::string body = content;
            if (reasoning.empty() && content.find("</think>") != std::string::npos &&
                content.find("<think>") != std::string::npos) {
                size_t a = content.find("<think>") + 7;
                size_t b = content.find("</think>");
                reasoning = trim(content.substr(a, b - a));
                body = trim(content.substr(b + 8));
            }
            out += "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + body + "<|im_end|>\n";
        }
    }
    if (add_generation_prompt) out += "<|im_start|>assistant\n<think>\n";
    return encode(out);
}

} // namespace quant
