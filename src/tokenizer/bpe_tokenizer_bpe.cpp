#include "quant/bpe_tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <unordered_set>
#include <queue>

namespace quant {

BPETokenizer::BPETokenizer() {
    for (int i = 0; i < 256; i++) {
        vocab_.push_back(std::string(1, (char)i));
    }
    bos_id_ = 1;
    eos_id_ = 2;
    unk_id_ = 0;
}

void BPETokenizer::add_token(const std::string& token) {
    vocab_.push_back(token);
}

int BPETokenizer::get_token_id(const std::string& token) const {
    for (size_t i = 0; i < vocab_.size(); i++) {
        if (vocab_[i] == token) return (int)i;
    }
    return unk_id_;
}

std::vector<int> BPETokenizer::bpe_merge(const std::vector<int>& ids) const {
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

void BPETokenizer::train(const std::vector<std::string>& texts, int vocab_size) {
    std::vector<int> all_bytes;
    for (const auto& t : texts) {
        for (char c : t) all_bytes.push_back((unsigned char)c);
    }

    int current_size = 256;
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
        add_token(vocab_[best->first.first] + vocab_[best->first.second]);

        std::vector<int> merged;
        for (size_t i = 0; i < all_bytes.size(); i++) {
            if (i + 1 < all_bytes.size() &&
                all_bytes[i] == best->first.first &&
                all_bytes[i+1] == best->first.second) {
                merged.push_back(new_id);
                i++;
            } else {
                merged.push_back(all_bytes[i]);
            }
        }
        all_bytes = merged;
    }
}

std::vector<int> BPETokenizer::encode(const std::string& text) {
    std::vector<int> ids;
    for (char c : text) ids.push_back((unsigned char)c);
    return bpe_merge(ids);
}

std::string BPETokenizer::decode(const std::vector<int>& ids) {
    std::string result;
    for (int id : ids) {
        if (id >= 0 && id < (int)vocab_.size()) {
            result += vocab_[id];
        }
    }
    return result;
}

void BPETokenizer::save(const std::string& path) const {
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

void BPETokenizer::load(const std::string& path) {
    // BUGFIX (bug census): unvalidated vs/len from file (negative → huge
    // resize/OOM; short reads → corrupt vocab). Bounds + stream checks.
    static constexpr int kMaxVocab = 1 << 20, kMaxTokLen = 1 << 16, kMaxMerges = 1 << 24;
    std::ifstream f(path, std::ios::binary);
    if (!f) return;
    int vs = 0;
    f.read((char*)&vs, sizeof(vs));
    if (!f || vs < 0 || vs > kMaxVocab) return;
    std::vector<std::string> vocab((size_t)vs);
    for (int i = 0; i < vs; i++) {
        int len = 0;
        f.read((char*)&len, sizeof(len));
        if (!f || len < 0 || len > kMaxTokLen) return;
        vocab[(size_t)i].resize((size_t)len);
        if (len > 0) {
            f.read(&vocab[(size_t)i][0], len);
            if (!f) return;
        }
    }
    int ms = 0;
    f.read((char*)&ms, sizeof(ms));
    if (!f || ms < 0 || ms > kMaxMerges) return;
    std::map<std::pair<int,int>, int> merges;
    for (int i = 0; i < ms; i++) {
        int a = 0, b = 0, c = 0;
        f.read((char*)&a, sizeof(a));
        f.read((char*)&b, sizeof(b));
        f.read((char*)&c, sizeof(c));
        if (!f) return;
        merges[{a,b}] = c;
    }
    vocab_ = std::move(vocab);
    merges_ = std::move(merges);
}

UnigramTokenizer::UnigramTokenizer() {
    Piece unk;
    unk.token = "<unk>";
    unk.log_prob = -100.0;
    unk.id = 0;
    pieces_.push_back(unk);
    token_to_id_["<unk>"] = 0;

    Piece bos;
    bos.token = "<bos>";
    bos.log_prob = -100.0;
    bos.id = 1;
    pieces_.push_back(bos);
    token_to_id_["<bos>"] = 1;

    Piece eos;
    eos.token = "<eos>";
    eos.log_prob = -100.0;
    eos.id = 2;
    pieces_.push_back(eos);
    token_to_id_["<eos>"] = 2;

    bos_id_ = 1;
    eos_id_ = 2;
    unk_id_ = 0;
}

std::vector<int> UnigramTokenizer::text_to_chars(const std::string& text) const {
    return UnicodeUtil::utf8_to_codepoints(text);
}

std::string UnigramTokenizer::chars_to_text(const std::vector<int>& chars) const {
    return UnicodeUtil::codepoints_to_utf8(chars);
}

void UnigramTokenizer::initialize_vocab(const std::vector<std::vector<int>>& corpus, int target_size) {
    std::unordered_map<std::string, int> char_counts;
    std::set<std::string> unique_pieces;

    unique_pieces.insert("<unk>");
    unique_pieces.insert("<bos>");
    unique_pieces.insert("<eos>");

    for (const auto& seq : corpus) {
        for (int cp : seq) {
            std::string piece = UnicodeUtil::utf8_encode(cp);
            unique_pieces.insert(piece);
            char_counts[piece]++;
        }
    }

    int max_char_pieces = target_size - 3;
    if ((int)unique_pieces.size() > max_char_pieces) {
        std::vector<std::pair<int, std::string>> sorted;
        for (const auto& p : unique_pieces) {
            if (p == "<unk>" || p == "<bos>" || p == "<eos>") continue;
            sorted.push_back({char_counts[p], p});
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        unique_pieces.clear();
        unique_pieces.insert("<unk>");
        unique_pieces.insert("<bos>");
        unique_pieces.insert("<eos>");
        for (int i = 0; i < max_char_pieces && i < (int)sorted.size(); i++) {
            unique_pieces.insert(sorted[i].second);
        }
    }

    for (auto it = unique_pieces.begin(); it != unique_pieces.end(); ++it) {
        if (token_to_id_.find(*it) == token_to_id_.end()) {
            Piece p;
            p.token = *it;
            p.id = (int)pieces_.size();
            p.log_prob = 0.0;
            pieces_.push_back(p);
            token_to_id_[*it] = p.id;
        }
    }

    int total_count = 0;
    for (const auto& seq : corpus) total_count += (int)seq.size();
    if (total_count == 0) total_count = 1;

    for (auto& p : pieces_) {
        int count = char_counts[p.token];
        if (count == 0) count = 1;
        p.log_prob = std::log((double)count / total_count);
    }
}

std::vector<int> UnigramTokenizer::viterbi_decode(const std::vector<int>& chars, double& score) const {
    int n = (int)chars.size();
    std::vector<double> dp(n + 1, -1e100);
    std::vector<int> back(n + 1, -1);
    std::vector<int> back_len(n + 1, 0);
    dp[0] = 0.0;

    for (int i = 0; i < n; i++) {
        if (dp[i] < -1e99) continue;
        std::string prefix;
        for (int j = i; j < n; j++) {
            prefix += UnicodeUtil::utf8_encode(chars[j]);
            auto it = token_to_id_.find(prefix);
            if (it != token_to_id_.end()) {
                int id = it->second;
                double logp = 0.0;
                if (id >= 0 && id < (int)pieces_.size()) {
                    logp = pieces_[id].log_prob;
                }
                double new_score = dp[i] + logp;
                if (new_score > dp[j + 1]) {
                    dp[j + 1] = new_score;
                    back[j + 1] = i;
                    back_len[j + 1] = j - i + 1;
                }
            }
        }
    }

    score = dp[n];
    std::vector<int> result;
    int pos = n;
    while (pos > 0) {
        int start = back[pos];
        if (start < 0) {
            result.push_back(unk_id_);
            pos--;
            continue;
        }
        std::string piece;
        for (int k = start; k < pos; k++) {
            piece += UnicodeUtil::utf8_encode(chars[k]);
        }
        auto it = token_to_id_.find(piece);
        if (it != token_to_id_.end()) {
            result.push_back(it->second);
        } else {
            result.push_back(unk_id_);
        }
        pos = start;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

void UnigramTokenizer::em_step(const std::vector<std::vector<int>>& corpus, std::vector<double>& probs) {
    std::vector<double> expected_counts(pieces_.size(), 1e-10);

    for (const auto& seq : corpus) {
        double total_score = 0.0;
        std::vector<int> seg = viterbi_decode(seq, total_score);
        for (int id : seg) {
            if (id >= 0 && id < (int)expected_counts.size()) {
                expected_counts[id] += 1.0;
            }
        }
    }

    double total = 0.0;
    for (double c : expected_counts) total += c;
    if (total <= 0) total = 1.0;

    for (size_t i = 0; i < pieces_.size(); i++) {
        probs[i] = expected_counts[i] / total;
    }

    double min_prob = 1e-8;
    for (auto& p : probs) {
        if (p < min_prob) p = min_prob;
    }
}

std::vector<int> UnigramTokenizer::best_segmentation(const std::vector<int>& chars) const {
    double score = 0.0;
    return viterbi_decode(chars, score);
}

void UnigramTokenizer::train(const std::vector<std::string>& corpus, int vocab_size) {
    std::vector<std::vector<int>> corpus_seqs;
    for (const auto& text : corpus) {
        corpus_seqs.push_back(text_to_chars(text));
    }

    initialize_vocab(corpus_seqs, vocab_size);

    std::vector<double> probs(pieces_.size(), 0.0);
    for (size_t i = 0; i < pieces_.size(); i++) {
        probs[i] = std::exp(pieces_[i].log_prob);
    }

    for (int iter = 0; iter < 10; iter++) {
        em_step(corpus_seqs, probs);

        for (size_t i = 0; i < pieces_.size(); i++) {
            pieces_[i].log_prob = std::log(probs[i]);
        }

        while ((int)pieces_.size() > vocab_size) {
            int worst_idx = -1;
            double worst_prob = 1.0;
            for (size_t i = 3; i < pieces_.size(); i++) {
                if (probs[i] < worst_prob) {
                    worst_prob = probs[i];
                    worst_idx = (int)i;
                }
            }
            if (worst_idx < 0 || worst_idx >= (int)pieces_.size()) break;
            pieces_.erase(pieces_.begin() + worst_idx);
            probs.erase(probs.begin() + worst_idx);
        }

        token_to_id_.clear();
        for (size_t i = 0; i < pieces_.size(); i++) {
            token_to_id_[pieces_[i].token] = (int)i;
        }
    }
}

std::vector<int> UnigramTokenizer::encode(const std::string& text) {
    std::vector<int> chars = text_to_chars(text);
    double score = 0.0;
    return viterbi_decode(chars, score);
}

std::string UnigramTokenizer::decode(const std::vector<int>& ids) {
    std::string result;
    for (int id : ids) {
        if (id >= 0 && id < (int)pieces_.size()) {
            if (pieces_[id].token == "<unk>" || pieces_[id].token == "<bos>" || pieces_[id].token == "<eos>") {
                continue;
            }
            result += pieces_[id].token;
        }
    }
    return result;
}

std::vector<int> UnigramTokenizer::encode_with_scores(const std::string& text, std::vector<double>& scores) {
    std::vector<int> chars = text_to_chars(text);
    double score = 0.0;
    scores.clear();
    auto ids = viterbi_decode(chars, score);
    for (int id : ids) {
        if (id >= 0 && id < (int)pieces_.size()) {
            scores.push_back(pieces_[id].log_prob);
        } else {
            scores.push_back(-100.0);
        }
    }
    return ids;
}

std::vector<std::vector<int>> UnigramTokenizer::encode_nbest(const std::string& text, int n) {
    std::vector<int> chars = text_to_chars(text);
    int len = (int)chars.size();
    struct DPNode {
        double score;
        int prev_pos;
        int prev_id;
    };
    std::vector<std::vector<DPNode>> dp(len + 1);
    dp[0].push_back({0.0, -1, -1});

    for (int i = 0; i < len; i++) {
        if (dp[i].empty()) continue;
        std::string prefix;
        for (int j = i; j < len; j++) {
            prefix += UnicodeUtil::utf8_encode(chars[j]);
            auto it = token_to_id_.find(prefix);
            if (it != token_to_id_.end()) {
                int id = it->second;
                double logp = (id >= 0 && id < (int)pieces_.size()) ? pieces_[id].log_prob : -100.0;
                for (const auto& node : dp[i]) {
                    DPNode new_node;
                    new_node.score = node.score + logp;
                    new_node.prev_pos = i;
                    new_node.prev_id = id;
                    dp[j + 1].push_back(new_node);
                }
            }
        }
        if ((int)dp[i].size() > n) {
            std::partial_sort(dp[i].begin(), dp[i].begin() + n, dp[i].end(),
                              [](const DPNode& a, const DPNode& b) { return a.score > b.score; });
            dp[i].resize(n);
        }
    }

    if (dp[len].empty()) {
        return {{unk_id_}};
    }
    if ((int)dp[len].size() > n) {
        std::partial_sort(dp[len].begin(), dp[len].begin() + n, dp[len].end(),
                          [](const DPNode& a, const DPNode& b) { return a.score > b.score; });
        dp[len].resize(n);
    }

    std::vector<std::vector<int>> results;
    for (const auto& node : dp[len]) {
        std::vector<int> seq;
        int pos = len;
        int cur_id = node.prev_id;
        int prev = node.prev_pos;
        while (pos > 0 && prev >= 0) {
            seq.push_back(cur_id);
            for (const auto& pn : dp[prev]) {
                if (std::abs(pn.score - (node.score - pieces_[cur_id].log_prob)) < 1e-6) {
                    cur_id = pn.prev_id;
                    pos = prev;
                    prev = pn.prev_pos;
                    break;
                }
            }
            if (prev == pos) break;
        }
        std::reverse(seq.begin(), seq.end());
        results.push_back(seq);
    }
    return results;
}

void UnigramTokenizer::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    int vs = (int)pieces_.size();
    f.write((char*)&vs, sizeof(vs));
    for (const auto& p : pieces_) {
        int len = (int)p.token.size();
        f.write((char*)&len, sizeof(len));
        f.write(p.token.data(), len);
        f.write((char*)&p.log_prob, sizeof(double));
        f.write((char*)&p.id, sizeof(int));
    }
}

void UnigramTokenizer::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    int vs; f.read((char*)&vs, sizeof(vs));
    pieces_.resize(vs);
    token_to_id_.clear();
    for (int i = 0; i < vs; i++) {
        int len; f.read((char*)&len, sizeof(len));
        pieces_[i].token.resize(len);
        f.read(&pieces_[i].token[0], len);
        f.read((char*)&pieces_[i].log_prob, sizeof(double));
        f.read((char*)&pieces_[i].id, sizeof(int));
        token_to_id_[pieces_[i].token] = pieces_[i].id;
    }
}

WordPieceTokenizer::WordPieceTokenizer() {
    token_to_id_[unk_token_] = 0;
    token_to_id_[cls_token_] = 101;
    token_to_id_[sep_token_] = 102;
    token_to_id_[pad_token_] = 0;
    token_to_id_[mask_token_] = 103;
    vocab_.push_back(unk_token_);
    for (int i = 1; i <= 100; i++) vocab_.push_back("");
    vocab_[101] = cls_token_;
    vocab_[102] = sep_token_;
    vocab_[103] = mask_token_;
    bos_id_ = 101;
    eos_id_ = 102;
    unk_id_ = 100;
}

void WordPieceTokenizer::set_special_tokens(const std::string& unk, const std::string& cls,
                                            const std::string& sep, const std::string& pad,
                                            const std::string& mask) {
    unk_token_ = unk;
    cls_token_ = cls;
    sep_token_ = sep;
    pad_token_ = pad;
    mask_token_ = mask;
}

double WordPieceTokenizer::compute_pair_score(
    const std::string& a, const std::string& b,
    const std::map<std::pair<std::string,std::string>, int>& pair_counts,
    const std::map<std::string, int>& symbol_counts) const {
    auto pit = pair_counts.find({a, b});
    if (pit == pair_counts.end()) return 0.0;
    auto ait = symbol_counts.find(a);
    auto bit = symbol_counts.find(b);
    if (ait == symbol_counts.end() || bit == symbol_counts.end()) return 0.0;
    if (ait->second == 0 || bit->second == 0) return 0.0;
    return (double)pit->second / ((double)ait->second * (double)bit->second);
}

std::vector<std::string> WordPieceTokenizer::split_word_for_wp(const std::string& word) const {
    std::vector<std::string> result;
    size_t pos = 0;
    while (pos < word.size()) {
        size_t before = pos;
        int cp = UnicodeUtil::utf8_decode(word, pos);
        if (cp >= 0) {
            result.push_back(UnicodeUtil::utf8_encode(cp));
        } else {
            result.push_back(std::string(1, word[pos]));
            pos = before + 1;
        }
    }
    return result;
}

std::vector<std::string> WordPieceTokenizer::wordpiece_tokenize(const std::string& word) const {
    std::vector<std::string> tokens;
    size_t pos = 0;
    while (pos < word.size()) {
        std::string best_match;
        int best_end = -1;
        for (int len = (int)word.size() - (int)pos; len >= 1; len--) {
            std::string sub = word.substr(pos, len);
            if (pos > 0) sub = "##" + sub;
            if (token_to_id_.count(sub)) {
                best_match = sub;
                best_end = (int)pos + len;
                break;
            }
        }
        if (best_end < 0) {
            tokens.push_back(unk_token_);
            size_t before = pos;
            UnicodeUtil::utf8_decode(word, pos);
            if (pos == before) pos++;
            continue;
        }
        tokens.push_back(best_match);
        pos = best_end;
    }
    return tokens;
}

std::vector<int> WordPieceTokenizer::encode_word(const std::string& word) const {
    std::vector<int> ids;
    auto pieces = wordpiece_tokenize(word);
    for (const auto& p : pieces) {
        auto it = token_to_id_.find(p);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            ids.push_back(unk_id_);
        }
    }
    return ids;
}

void WordPieceTokenizer::train(const std::vector<std::string>& corpus, int vocab_size) {
    std::map<std::pair<std::string,std::string>, int> pair_counts;
    std::map<std::string, int> symbol_counts;
    std::vector<std::vector<std::string>> word_pieces;

    for (const auto& text : corpus) {
        std::vector<std::string> words = UnicodeUtil::split_on_whitespace(text);
        for (const auto& word : words) {
            auto pieces = split_word_for_wp(word);
            word_pieces.push_back(pieces);
            if (token_to_id_.find(word) == token_to_id_.end()) {
                int id = (int)vocab_.size();
                token_to_id_[word] = id;
                vocab_.push_back(word);
            }
            for (const auto& p : pieces) {
                symbol_counts[p]++;
            }
        }
    }

    int current_size = (int)vocab_.size();
    int max_iterations = vocab_size - current_size - 4;

    for (int iter = 0; iter < max_iterations && current_size < vocab_size; iter++) {
        pair_counts.clear();
        for (const auto& pieces : word_pieces) {
            for (size_t i = 0; i + 1 < pieces.size(); i++) {
                pair_counts[{pieces[i], pieces[i+1]}]++;
            }
        }

        std::string best_pair_a, best_pair_b;
        double best_score = -1.0;
        for (const auto& pc : pair_counts) {
            double score = compute_pair_score(pc.first.first, pc.first.second, pair_counts, symbol_counts);
            if (score > best_score) {
                best_score = score;
                best_pair_a = pc.first.first;
                best_pair_b = pc.first.second;
            }
        }

        if (best_pair_a.empty()) break;

        std::string merged = best_pair_a + best_pair_b;
        token_to_id_[merged] = current_size;
        if ((int)vocab_.size() <= current_size) vocab_.resize(current_size + 1);
        vocab_[current_size] = merged;
        current_size++;

        symbol_counts[merged] = 0;
        auto ait = symbol_counts.find(best_pair_a);
        auto bit = symbol_counts.find(best_pair_b);
        if (ait != symbol_counts.end()) symbol_counts[merged] += ait->second;
        if (bit != symbol_counts.end()) symbol_counts[merged] += bit->second;

        for (auto& pieces : word_pieces) {
            std::vector<std::string> new_pieces;
            for (size_t i = 0; i < pieces.size(); i++) {
                if (i + 1 < pieces.size() && pieces[i] == best_pair_a && pieces[i+1] == best_pair_b) {
                    new_pieces.push_back(merged);
                    i++;
                } else {
                    new_pieces.push_back(pieces[i]);
                }
            }
            pieces = new_pieces;
        }
    }
}

std::vector<int> WordPieceTokenizer::encode(const std::string& text) {
    std::vector<int> ids;
    std::vector<std::string> words = UnicodeUtil::split_on_whitespace(text);

    auto cls_it = token_to_id_.find(cls_token_);
    if (cls_it != token_to_id_.end()) ids.push_back(cls_it->second);

    for (const auto& word : words) {
        auto word_ids = encode_word(word);
        ids.insert(ids.end(), word_ids.begin(), word_ids.end());
    }

    auto sep_it = token_to_id_.find(sep_token_);
    if (sep_it != token_to_id_.end()) ids.push_back(sep_it->second);

    return ids;
}

std::string WordPieceTokenizer::decode(const std::vector<int>& ids) {
    std::string result;
    for (int id : ids) {
        if (id >= 0 && id < (int)vocab_.size()) {
            std::string token = vocab_[id];
            if (token.size() >= 2 && token.substr(0, 2) == "##") {
                result += token.substr(2);
            } else if (token == cls_token_ || token == sep_token_ ||
                       token == pad_token_ || token == mask_token_ || token == unk_token_) {
                if (token == sep_token_ || token == cls_token_) {
                    result += " ";
                }
            } else {
                if (!result.empty() && result.back() != ' ') result += " ";
                result += token;
            }
        }
    }
    return result;
}

void WordPieceTokenizer::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    int vs = (int)vocab_.size();
    f.write((char*)&vs, sizeof(vs));
    for (const auto& v : vocab_) {
        int len = (int)v.size();
        f.write((char*)&len, sizeof(len));
        f.write(v.data(), len);
    }
    int st = (int)unk_token_.size();
    f.write((char*)&st, sizeof(st));
    f.write(unk_token_.data(), st);
}

void WordPieceTokenizer::load(const std::string& path) {
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
    int st; f.read((char*)&st, sizeof(st));
    unk_token_.resize(st);
    f.read(&unk_token_[0], st);
}

// ── L072 SentencePiece interchange (UnigramTokenizer) ───────────────────────
// U+2581 (LOWER ONE EIGHTH BLOCK) UTF-8 bytes: E2 96 81. SentencePiece uses it
// as the word-boundary marker; decoding maps it back to ASCII space.
namespace {
constexpr char kSpMarker[] = "\xE2\x96\x81";
} // namespace

std::string UnigramTokenizer::sentencepiece_normalize(const std::string& text) {
    // Minimal NFKC-ish normalization without ICU: collapse ASCII whitespace
    // runs to a single space, trim ends, then mark word starts with U+2581.
    // Full-width / compatibility forms are passed through unchanged
    // (documented limitation; use Python `sentencepiece` export for training).
    std::string collapsed;
    collapsed.reserve(text.size());
    bool in_ws = true; // trim leading
    for (unsigned char c : text) {
        bool ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                   c == '\v' || c == '\f');
        if (ws) {
            in_ws = true;
        } else {
            if (!collapsed.empty() && in_ws) collapsed += ' ';
            collapsed += (char)c;
            in_ws = false;
        }
    }
    if (collapsed.empty()) return collapsed;
    // Map spaces to U+2581, keeping a leading marker on non-space start.
    std::string out;
    out.reserve(collapsed.size() + 4);
    for (size_t i = 0; i < collapsed.size(); i++) {
        if (collapsed[i] == ' ') {
            out += kSpMarker;
        } else {
            if (i == 0) out += kSpMarker;
            out += collapsed[i];
        }
    }
    return out;
}

std::string UnigramTokenizer::sentencepiece_join(const std::string& text) {
    // Inverse of the marker mapping above: U+2581 -> space, trim one leading
    // space that corresponds to the start-of-sentence marker.
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        if (i + 3 <= text.size() && text.compare(i, 3, kSpMarker) == 0) {
            out += ' ';
            i += 3;
        } else {
            out += text[i];
            i++;
        }
    }
    if (!out.empty() && out[0] == ' ') out.erase(out.begin());
    return out;
}

bool UnigramTokenizer::load_sentencepiece_tsv(const std::string& path,
                                              std::string* err_out) {
    auto fail = [&](const std::string& m) {
        if (err_out) *err_out = m;
        return false;
    };
    std::ifstream f(path);
    if (!f.is_open()) return fail("sentencepiece: cannot open " + path);
    std::vector<Piece> pieces;
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        lineno++;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        auto tab = line.find('\t');
        if (tab == std::string::npos)
            return fail("sentencepiece: line " + std::to_string(lineno) +
                        " missing TAB separator");
        std::string tok = line.substr(0, tab);
        std::string score_s = line.substr(tab + 1);
        if (tok.empty())
            return fail("sentencepiece: line " + std::to_string(lineno) +
                        " empty piece");
        char* endp = nullptr;
        double lp = std::strtod(score_s.c_str(), &endp);
        if (endp == score_s.c_str())
            return fail("sentencepiece: line " + std::to_string(lineno) +
                        " bad log_prob '" + score_s + "'");
        Piece p;
        p.token = tok;
        p.log_prob = lp;
        p.id = (int)pieces.size();
        pieces.push_back(std::move(p));
    }
    if (pieces.empty()) return fail("sentencepiece: no pieces in " + path);
    pieces_ = std::move(pieces);
    token_to_id_.clear();
    for (size_t i = 0; i < pieces_.size(); i++) {
        pieces_[i].id = (int)i;
        token_to_id_[pieces_[i].token] = (int)i;
    }
    // Keep conventional ids when the file carries them positionally:
    // callers may remap bos/eos/unk via set_*_id().
    return true;
}

bool UnigramTokenizer::save_sentencepiece_tsv(const std::string& path) const {
    std::ofstream f(path);
    if (!f.is_open()) return false;
    for (const auto& p : pieces_) {
        f << p.token << '\t' << p.log_prob << '\n';
        if (!f.good()) return false;
    }
    return true;
}

std::vector<int> UnigramTokenizer::encode_sentencepiece(const std::string& text) {
    std::string norm = sentencepiece_normalize(text);
    if (norm.empty()) return {};
    std::vector<int> chars = text_to_chars(norm);
    double score = 0.0;
    return viterbi_decode(chars, score);
}

std::string UnigramTokenizer::decode_sentencepiece(const std::vector<int>& ids) {
    std::string raw = decode(ids);
    return sentencepiece_join(raw);
}

} // namespace quant
