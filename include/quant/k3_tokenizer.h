#pragma once
#include "quant/tokenizer.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace quant {

class K3Tokenizer : public Tokenizer {
public:
    K3Tokenizer();

    bool load_from_dir(const std::string& model_dir);

    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& ids) override { return decode(ids, true); }
    std::string decode(const std::vector<int>& ids, bool skip_special) const;

    int vocab_size() const override { return vocab_size_; }
    int bos_id() const override { return bos_id_; }
    int eos_id() const override { return eos_id_; }
    int im_start_id() const { return im_start_id_; }
    int im_end_id() const { return im_end_id_; }
    bool is_special(int id) const;

    // Chat template helper (Kimi style)
    std::vector<int> apply_chat_template(
        const std::vector<std::pair<std::string,std::string>>& messages,
        bool add_generation_prompt);

private:
    BPETokenizer bpe_;
    int vocab_size_ = 152064;
    int bos_id_ = 1;
    int eos_id_ = 2;
    int im_start_id_ = 151644;
    int im_end_id_ = 151645;
    std::unordered_map<int, std::string> special_map_;
};

} // namespace quant
