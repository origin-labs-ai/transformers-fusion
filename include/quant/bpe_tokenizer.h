#pragma once
#include "quant/tokenizer.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <cmath>
#include <functional>
#include <mutex>
#include <thread>
#include <queue>
#include <random>

namespace quant {

struct UnicodeUtil {
    static bool is_cjk(int cp);
    static bool is_whitespace(int cp);
    static bool is_punctuation(int cp);
    static bool is_letter(int cp);
    static bool is_digit(int cp);
    static bool is_control(int cp);
    static bool is_math_symbol(int cp);
    static bool is_currency(int cp);
    static std::string utf8_encode(int cp);
    static int utf8_decode(const std::string& s, size_t& pos);
    static std::vector<int> utf8_to_codepoints(const std::string& s);
    static std::string codepoints_to_utf8(const std::vector<int>& cps);
    static std::string nfkc_normalize(const std::string& s);
    static std::vector<std::string> split_on_whitespace(const std::string& s);
    static std::vector<std::string> pretokenize(const std::string& s);
    static std::string detect_language(const std::string& segment);
    static int compose_pair(int a, int b);
    static bool is_regional_indicator(int cp);
    static bool is_variation_selector(int cp);
    static bool is_combining_mark(int cp);
    static bool is_emoji(int cp);
    static bool is_emoji_modifier(int cp);
    static int width(int cp);
};

class UnigramTokenizer : public Tokenizer {
public:
    UnigramTokenizer();

    void train(const std::vector<std::string>& corpus, int vocab_size = 16000);
    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& ids) override;

    int vocab_size() const override { return (int)pieces_.size(); }
    int bos_id() const override { return bos_id_; }
    int eos_id() const override { return eos_id_; }

    void set_bos_id(int id) { bos_id_ = id; }
    void set_eos_id(int id) { eos_id_ = id; }
    void set_unk_id(int id) { unk_id_ = id; }

    void save(const std::string& path) const;
    void load(const std::string& path);

    // ── L072 SentencePiece interchange (Phase 16 Wave 7) ───────────────────
    // Binary protobuf .model is NOT parsed (no protobuf dep by design).
    // Supported: TSV interchange "<piece>\t<log_prob>" per line (export via
    // save_sentencepiece_tsv, e.g. from a Python `sentencepiece` export), plus
    // SentencePiece encode/decode semantics (U+2581 word marker, NFKC-ish
    // whitespace normalization). encode_sentencepiece() normalizes then runs
    // the same Viterbi unigram segmentation as encode(); decode_sentencepiece()
    // joins then maps U+2581 back to space.
    static std::string sentencepiece_normalize(const std::string& text);
    static std::string sentencepiece_join(const std::string& text);
    bool load_sentencepiece_tsv(const std::string& path, std::string* err_out);
    bool save_sentencepiece_tsv(const std::string& path) const;
    std::vector<int> encode_sentencepiece(const std::string& text);
    std::string decode_sentencepiece(const std::vector<int>& ids);

    std::vector<int> encode_with_scores(const std::string& text, std::vector<double>& scores);
    std::vector<std::vector<int>> encode_nbest(const std::string& text, int n = 5);

private:
    struct Piece {
        std::string token;
        double log_prob;
        int id;
    };
    std::vector<Piece> pieces_;
    std::unordered_map<std::string, int> token_to_id_;
    int bos_id_ = 1;
    int eos_id_ = 2;
    int unk_id_ = 0;

    std::vector<int> viterbi_decode(const std::vector<int>& chars, double& score) const;
    void em_step(const std::vector<std::vector<int>>& corpus, std::vector<double>& probs);
    std::vector<int> best_segmentation(const std::vector<int>& chars) const;
    void initialize_vocab(const std::vector<std::vector<int>>& corpus, int target_size);
    std::vector<int> text_to_chars(const std::string& text) const;
    std::string chars_to_text(const std::vector<int>& chars) const;
};

class WordPieceTokenizer : public Tokenizer {
public:
    WordPieceTokenizer();

    void train(const std::vector<std::string>& corpus, int vocab_size = 30000);
    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& ids) override;

    int vocab_size() const override { return (int)vocab_.size(); }
    int bos_id() const override { return bos_id_; }
    int eos_id() const override { return eos_id_; }

    void set_special_tokens(const std::string& unk, const std::string& cls,
                            const std::string& sep, const std::string& pad,
                            const std::string& mask);

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> token_to_id_;
    std::string unk_token_ = "[UNK]";
    std::string cls_token_ = "[CLS]";
    std::string sep_token_ = "[SEP]";
    std::string pad_token_ = "[PAD]";
    std::string mask_token_ = "[MASK]";
    int bos_id_ = 101;
    int eos_id_ = 102;
    int unk_id_ = 100;

    std::vector<int> encode_word(const std::string& word) const;
    double compute_pair_score(const std::string& a, const std::string& b,
                              const std::map<std::pair<std::string,std::string>, int>& pair_counts,
                              const std::map<std::string, int>& symbol_counts) const;
    std::vector<std::string> wordpiece_tokenize(const std::string& word) const;
    std::vector<std::string> split_word_for_wp(const std::string& word) const;
};

class ByteLevelBPETokenizer : public Tokenizer {
public:
    ByteLevelBPETokenizer();

    void train(const std::vector<std::string>& corpus, int vocab_size = 50000);
    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& ids) override;

    int vocab_size() const override;
    int bos_id() const override { return bos_id_; }
    int eos_id() const override { return eos_id_; }

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::vector<std::string> vocab_;
    std::map<std::pair<int, int>, int> merges_;
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    int bos_id_ = 3;
    int eos_id_ = 4;
    int unk_id_ = 0;

    std::string byte_to_token(unsigned char b) const;
    unsigned char token_to_byte(const std::string& t) const;
    std::vector<int> byte_encode(const std::string& text) const;
    std::string byte_decode(const std::vector<int>& ids) const;
    std::vector<int> bpe_merge_tokens(const std::vector<int>& ids) const;
    void add_token(const std::string& token);
    int get_token_id(const std::string& token) const;

    static std::string build_byte_map();
    static std::string byte_map_;
    static int byte_map_index[256];
    static std::once_flag byte_map_flag_;
    static void init_byte_map();
};

class SentencePieceTrainer {
public:
    enum ModelType : uint8_t { BPE = 0, UNIGRAM = 1, WORDPIECE = 2 };

    static void train(const std::vector<std::string>& corpus,
                      const std::string& model_prefix,
                      ModelType model_type = BPE,
                      int vocab_size = 32000,
                      float character_coverage = 0.9995f,
                      int num_threads = 4,
                      unsigned int seed = 42);

    static std::string normalize_nfkc(const std::string& text);
    static std::vector<std::string> pretokenize_default(const std::string& text);
    static std::vector<std::string> split_on_whitespace_and_punctuation(const std::string& text);
};

class MultiLingualTokenizer : public Tokenizer {
public:
    MultiLingualTokenizer();

    void train(const std::vector<std::string>& corpus, int vocab_size = 50000);
    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& ids) override;

    int vocab_size() const override;
    int bos_id() const override { return bos_id_; }
    int eos_id() const override { return eos_id_; }

    void save(const std::string& path) const;
    void load(const std::string& path);

    void set_fallback_strategy(const std::string& strategy);
    void add_language_token(const std::string& lang);
    std::vector<int> encode_with_lang_tag(const std::string& text, const std::string& lang);

private:
    std::vector<std::string> vocab_;
    std::map<std::pair<int, int>, int> merges_;
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<int> special_ids_;
    std::vector<std::string> lang_tags_;
    int bos_id_ = 1;
    int eos_id_ = 2;
    int unk_id_ = 0;

    std::string detect_language_segment(const std::string& segment) const;
    std::vector<int> encode_segment(const std::string& segment) const;
    std::vector<int> encode_utf8_bytes(const std::string& text) const;
    std::string decode_utf8_bytes(const std::vector<int>& ids) const;
    std::vector<std::string> split_by_language(const std::string& text) const;
};

} // namespace quant
