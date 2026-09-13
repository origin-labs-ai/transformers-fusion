#include "quant/bpe_tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <climits>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <unordered_set>
#include <queue>

namespace quant {

std::string ByteLevelBPETokenizer::byte_map_;
int ByteLevelBPETokenizer::byte_map_index[256];
std::once_flag ByteLevelBPETokenizer::byte_map_flag_;

void ByteLevelBPETokenizer::init_byte_map() {
    std::call_once(byte_map_flag_, []() {
        byte_map_ = build_byte_map();
        for (int i = 0; i < 256; i++) {
            char c = (char)i;
            std::string key(1, c);
            auto pos = byte_map_.find(key);
            if (pos != std::string::npos) {
                byte_map_index[i] = (int)pos;
            } else {
                byte_map_index[i] = i + 256;
            }
        }
    });
}

std::string ByteLevelBPETokenizer::build_byte_map() {
    std::string map;
    for (int i = 33; i <= 126; i++) {
        map += (char)i;
    }
    for (int i = 0; i < 256; i++) {
        if ((i < 33 || i > 126) && i != 32) {
            map += (char)(192 + (i >> 6));
            map += (char)(128 + (i & 63));
        }
    }
    map += ' ';
    return map;
}

ByteLevelBPETokenizer::ByteLevelBPETokenizer() {
    init_byte_map();
    vocab_.push_back("<unk>");
    token_to_id_["<unk>"] = 0;
    id_to_token_.push_back("<unk>");

    for (int i = 0; i < 256; i++) {
        std::string t = byte_to_token((unsigned char)i);
        vocab_.push_back(t);
        token_to_id_[t] = (int)vocab_.size() - 1;
        id_to_token_.push_back(t);
    }
    bos_id_ = (int)vocab_.size();
    vocab_.push_back("<bos>");
    token_to_id_["<bos>"] = bos_id_;
    id_to_token_.push_back("<bos>");

    eos_id_ = (int)vocab_.size();
    vocab_.push_back("<eos>");
    token_to_id_["<eos>"] = eos_id_;
    id_to_token_.push_back("<eos>");
}

int ByteLevelBPETokenizer::vocab_size() const {
    return (int)vocab_.size();
}

std::string ByteLevelBPETokenizer::byte_to_token(unsigned char b) const {
    if (b >= 33 && b <= 126 && b != 32) {
        return std::string(1, (char)b);
    }
    if (b == 32) return "Ġ";
    char buf[4];
    if (b < 192) {
        buf[0] = (char)(192 + (b >> 6));
        buf[1] = (char)(128 + (b & 63));
        buf[2] = 0;
    } else {
        buf[0] = (char)(b);
        buf[1] = 0;
    }
    return std::string(buf);
}

unsigned char ByteLevelBPETokenizer::token_to_byte(const std::string& t) const {
    if (t.size() == 1) return (unsigned char)t[0];
    if (t == "Ġ") return 32;
    if (t.size() >= 2) {
        unsigned char b1 = (unsigned char)t[0];
        unsigned char b2 = (unsigned char)t[1];
        if ((b1 & 0xE0) == 0xC0 && (b2 & 0xC0) == 0x80) {
            int val = ((b1 & 0x1F) << 6) | (b2 & 0x3F);
            if (val >= 0 && val < 256) return (unsigned char)val;
        }
    }
    return 0;
}

std::vector<int> ByteLevelBPETokenizer::byte_encode(const std::string& text) const {
    std::vector<int> ids;
    for (size_t i = 0; i < text.size(); i++) {
        unsigned char c = (unsigned char)text[i];
        std::string t = byte_to_token(c);
        auto it = token_to_id_.find(t);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            ids.push_back(0);
        }
    }
    return ids;
}

std::string ByteLevelBPETokenizer::byte_decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        if (id >= 0 && id < (int)id_to_token_.size()) {
            std::string t = id_to_token_[id];
            if (t == "<unk>" || t == "<bos>" || t == "<eos>") continue;
            result += token_to_byte(t);
        }
    }
    return result;
}

std::vector<int> ByteLevelBPETokenizer::bpe_merge_tokens(const std::vector<int>& ids) const {
    std::vector<int> result = ids;
    while (true) {
        int best_left = -1, best_right = -1, best_id = -1;
        for (size_t i = 0; i + 1 < result.size(); i++) {
            auto it = merges_.find({result[i], result[i+1]});
            if (it != merges_.end()) {
                best_left = result[i];
                best_right = result[i+1];
                best_id = it->second;
                break;
            }
        }
        if (best_id == -1) break;
        std::vector<int> next;
        for (size_t i = 0; i < result.size(); i++) {
            if (i + 1 < result.size() && result[i] == best_left && result[i+1] == best_right) {
                next.push_back(best_id);
                i++;
            } else {
                next.push_back(result[i]);
            }
        }
        result = next;
    }
    return result;
}

void ByteLevelBPETokenizer::add_token(const std::string& token) {
    if (token_to_id_.find(token) == token_to_id_.end()) {
        int id = (int)vocab_.size();
        vocab_.push_back(token);
        token_to_id_[token] = id;
        id_to_token_.push_back(token);
    }
}

int ByteLevelBPETokenizer::get_token_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) return it->second;
    return 0;
}

void ByteLevelBPETokenizer::train(const std::vector<std::string>& corpus, int vocab_size) {
    std::vector<int> all_ids;
    for (const auto& text : corpus) {
        auto ids = byte_encode(text);
        all_ids.insert(all_ids.end(), ids.begin(), ids.end());
    }

    int start_id = (int)vocab_.size();
    int current_size = 256;
    int max_merges = std::min(vocab_size - start_id, 50000);

    for (int m = 0; m < max_merges; m++) {
        std::map<std::pair<int,int>, int> pair_counts;
        for (size_t i = 0; i + 1 < all_ids.size(); i++) {
            if (all_ids[i] >= 0 && all_ids[i+1] >= 0) {
                pair_counts[{all_ids[i], all_ids[i+1]}]++;
            }
        }
        if (pair_counts.empty()) break;

        auto best = pair_counts.begin();
        for (auto it = pair_counts.begin(); it != pair_counts.end(); ++it) {
            if (it->second > best->second) best = it;
        }

        int new_id = current_size++;
        if (new_id < (int)vocab_.size()) new_id = (int)vocab_.size();
        merges_[best->first] = new_id;

        auto left_it = token_to_id_.find(byte_to_token((unsigned char)best->first.first));
        auto right_it = token_to_id_.find(byte_to_token((unsigned char)best->first.second));
        if (left_it == token_to_id_.end() || right_it == token_to_id_.end()) continue;

        std::string merged_token = id_to_token_[best->first.first] + id_to_token_[best->first.second];
        add_token(merged_token);

        std::vector<int> merged;
        for (size_t i = 0; i < all_ids.size(); i++) {
            if (i + 1 < all_ids.size() && all_ids[i] == best->first.first && all_ids[i+1] == best->first.second) {
                merged.push_back(new_id);
                i++;
            } else {
                merged.push_back(all_ids[i]);
            }
        }
        all_ids = merged;
    }
}

std::vector<int> ByteLevelBPETokenizer::encode(const std::string& text) {
    auto ids = byte_encode(text);
    return bpe_merge_tokens(ids);
}

std::string ByteLevelBPETokenizer::decode(const std::vector<int>& ids) {
    std::string result;
    for (int id : ids) {
        if (id >= 0 && id < (int)id_to_token_.size()) {
            std::string t = id_to_token_[id];
            if (t == "<unk>" || t == "<bos>" || t == "<eos>") continue;
            result += token_to_byte(t);
        }
    }
    return result;
}

void ByteLevelBPETokenizer::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    int vs = (int)vocab_.size();
    f.write((char*)&vs, sizeof(vs));
    for (const auto& v : vocab_) {
        int len = (int)v.size();
        f.write((char*)&len, sizeof(len));
        f.write(v.data(), len);
    }
    int ms = (int)merges_.size();
    f.write((char*)&ms, sizeof(ms));
    for (const auto& m : merges_) {
        f.write((char*)&m.first.first, sizeof(int));
        f.write((char*)&m.first.second, sizeof(int));
        f.write((char*)&m.second, sizeof(int));
    }
}

void ByteLevelBPETokenizer::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    int vs; f.read((char*)&vs, sizeof(vs));
    vocab_.resize(vs);
    token_to_id_.clear();
    id_to_token_.resize(vs);
    for (int i = 0; i < vs; i++) {
        int len; f.read((char*)&len, sizeof(len));
        vocab_[i].resize(len);
        f.read(&vocab_[i][0], len);
        token_to_id_[vocab_[i]] = i;
        id_to_token_[i] = vocab_[i];
    }
    int ms; f.read((char*)&ms, sizeof(ms));
    merges_.clear();
    for (int i = 0; i < ms; i++) {
        int a, b, c; f.read((char*)&a, sizeof(a));
        f.read((char*)&b, sizeof(b));
        f.read((char*)&c, sizeof(c));
        merges_[{a,b}] = c;
    }
}

MultiLingualTokenizer::MultiLingualTokenizer() {
    for (int i = 0; i < 256; i++) {
        std::string t(1, (char)i);
        vocab_.push_back(t);
        token_to_id_[t] = i;
    }
    vocab_.push_back("<unk>");
    token_to_id_["<unk>"] = 256;
    vocab_.push_back("<bos>");
    token_to_id_["<bos>"] = 257;
    vocab_.push_back("<eos>");
    token_to_id_["<eos>"] = 258;
    unk_id_ = 256;
    bos_id_ = 257;
    eos_id_ = 258;
}

int MultiLingualTokenizer::vocab_size() const {
    return (int)vocab_.size() + (int)lang_tags_.size();
}

void MultiLingualTokenizer::add_language_token(const std::string& lang) {
    lang_tags_.push_back(lang);
    std::string tag = "<" + lang + ">";
    token_to_id_[tag] = (int)vocab_.size();
    vocab_.push_back(tag);
    special_ids_.push_back((int)vocab_.size() - 1);
}

void MultiLingualTokenizer::set_fallback_strategy([[maybe_unused]] const std::string& strategy) {
}

std::string MultiLingualTokenizer::detect_language_segment(const std::string& segment) const {
    return UnicodeUtil::detect_language(segment);
}

std::vector<std::string> MultiLingualTokenizer::split_by_language(const std::string& text) const {
    std::vector<std::string> segments;
    std::string current;
    std::string current_lang = "unknown";
    size_t pos = 0;

    while (pos < text.size()) {
        int cp = UnicodeUtil::utf8_decode(text, pos);
        if (cp < 0) break;

        std::string seg_lang = "unknown";
        if (UnicodeUtil::is_cjk(cp)) seg_lang = "cjk";
        else if (UnicodeUtil::is_letter(cp) || UnicodeUtil::is_digit(cp)) seg_lang = "text";
        else seg_lang = "other";

        if (seg_lang != current_lang && !current.empty() && seg_lang == "cjk") {
            segments.push_back(current);
            current.clear();
        } else if (seg_lang != current_lang && !current.empty() && current_lang == "cjk") {
            segments.push_back(current);
            current.clear();
        }

        current += UnicodeUtil::utf8_encode(cp);
        current_lang = seg_lang;
    }
    if (!current.empty()) segments.push_back(current);
    return segments;
}

std::vector<int> MultiLingualTokenizer::encode_utf8_bytes(const std::string& text) const {
    std::vector<int> ids;
    for (size_t i = 0; i < text.size(); i++) {
        unsigned char c = (unsigned char)text[i];
        ids.push_back(c);
    }
    return ids;
}

std::string MultiLingualTokenizer::decode_utf8_bytes(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        if (id >= 0 && id < 256) {
            result += (char)(unsigned char)id;
        } else if (id < (int)vocab_.size()) {
            result += vocab_[id];
        }
    }
    return result;
}

std::vector<int> MultiLingualTokenizer::encode_segment(const std::string& segment) const {
    std::vector<int> ids;
    for (size_t i = 0; i < segment.size(); i++) {
        ids.push_back((unsigned char)segment[i]);
    }
    std::vector<int> result = ids;
    while (true) {
        auto best = merges_.end();
        for (size_t i = 0; i + 1 < result.size(); i++) {
            auto it = merges_.find({result[i], result[i+1]});
            if (it != merges_.end()) {
                best = it;
                break;
            }
        }
        if (best == merges_.end()) break;
        std::vector<int> next;
        for (size_t i = 0; i < result.size(); i++) {
            if (i + 1 < result.size() && result[i] == best->first.first && result[i+1] == best->first.second) {
                next.push_back(best->second);
                i++;
            } else {
                next.push_back(result[i]);
            }
        }
        result = next;
    }
    return result;
}

std::vector<int> MultiLingualTokenizer::encode(const std::string& text) {
    auto segments = split_by_language(text);
    std::vector<int> ids;
    for (const auto& seg : segments) {
        auto seg_ids = encode_segment(seg);
        ids.insert(ids.end(), seg_ids.begin(), seg_ids.end());
    }
    return ids;
}

std::string MultiLingualTokenizer::decode(const std::vector<int>& ids) {
    return decode_utf8_bytes(ids);
}

std::vector<int> MultiLingualTokenizer::encode_with_lang_tag(const std::string& text, const std::string& lang) {
    std::vector<int> ids;
    std::string tag = "<" + lang + ">";
    auto it = token_to_id_.find(tag);
    if (it != token_to_id_.end()) ids.push_back(it->second);

    auto regular_ids = encode(text);
    ids.insert(ids.end(), regular_ids.begin(), regular_ids.end());

    it = token_to_id_.find(tag);
    if (it != token_to_id_.end()) ids.push_back(it->second);

    return ids;
}

void MultiLingualTokenizer::train(const std::vector<std::string>& corpus, int vocab_size) {
    std::vector<int> all_bytes;
    for (const auto& text : corpus) {
        for (size_t i = 0; i < text.size(); i++) {
            all_bytes.push_back((unsigned char)text[i]);
        }
    }

    int current_size = 259;
    while (current_size < vocab_size) {
        std::map<std::pair<int,int>, int> pair_counts;
        for (size_t i = 0; i + 1 < all_bytes.size(); i++) {
            pair_counts[{all_bytes[i], all_bytes[i+1]}]++;
        }
        if (pair_counts.empty()) break;

        auto best = pair_counts.begin();
        for (auto it = pair_counts.begin(); it != pair_counts.end(); ++it) {
            if (it->second > best->second) best = it;
        }

        int new_id = current_size++;
        merges_[best->first] = new_id;
        std::string merged_tok;
        if (best->first.first >= 0 && best->first.first < (int)vocab_.size()) {
            merged_tok += vocab_[best->first.first];
        } else {
            merged_tok += (char)(unsigned char)best->first.first;
        }
        if (best->first.second >= 0 && best->first.second < (int)vocab_.size()) {
            merged_tok += vocab_[best->first.second];
        } else {
            merged_tok += (char)(unsigned char)best->first.second;
        }
        vocab_.push_back(merged_tok);
        token_to_id_[merged_tok] = new_id;

        std::vector<int> merged;
        for (size_t i = 0; i < all_bytes.size(); i++) {
            if (i + 1 < all_bytes.size() && all_bytes[i] == best->first.first && all_bytes[i+1] == best->first.second) {
                merged.push_back(new_id);
                i++;
            } else {
                merged.push_back(all_bytes[i]);
            }
        }
        all_bytes = merged;
    }
}

void MultiLingualTokenizer::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    int vs = (int)vocab_.size();
    f.write((char*)&vs, sizeof(vs));
    for (const auto& v : vocab_) {
        int len = (int)v.size();
        f.write((char*)&len, sizeof(len));
        f.write(v.data(), len);
    }
    int ms = (int)merges_.size();
    f.write((char*)&ms, sizeof(ms));
    for (const auto& m : merges_) {
        f.write((char*)&m.first.first, sizeof(int));
        f.write((char*)&m.first.second, sizeof(int));
        f.write((char*)&m.second, sizeof(int));
    }
}

void MultiLingualTokenizer::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    int vs; f.read((char*)&vs, sizeof(vs));
    vocab_.resize(vs);
    token_to_id_.clear();
    for (int i = 0; i < vs; i++) {
        int len; f.read((char*)&len, sizeof(len));
        vocab_[i].resize(len);
        f.read(&vocab_[i][0], len);
        token_to_id_[vocab_[i]] = i;
    }
    int ms; f.read((char*)&ms, sizeof(ms));
    merges_.clear();
    for (int i = 0; i < ms; i++) {
        int a, b, c; f.read((char*)&a, sizeof(a));
        f.read((char*)&b, sizeof(b));
        f.read((char*)&c, sizeof(c));
        merges_[{a,b}] = c;
    }
}

} // namespace quant
