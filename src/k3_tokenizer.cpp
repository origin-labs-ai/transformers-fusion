#include "quant/k3_tokenizer.h"
#include <fstream>
#include <filesystem>

namespace quant {

K3Tokenizer::K3Tokenizer() {
    special_map_[im_start_id_] = "<|im_start|>";
    special_map_[im_end_id_] = "<|im_end|>";
    special_map_[eos_id_] = "<|im_end|>";
}

bool K3Tokenizer::load_from_dir(const std::string& model_dir) {
    namespace fs = std::filesystem;
    std::string vocab_path = model_dir + "/vocab.json";
    std::string merges_path = model_dir + "/merges.txt";
    if (!fs::exists(vocab_path) || !fs::exists(merges_path)) return false;
    // delegate to BPE loader if files exist
    bpe_.load(model_dir + "/tokenizer.json");
    return true;
}

std::vector<int> K3Tokenizer::encode(const std::string& text) {
    // simple delegation to BPE with K3 vocab clamp
    auto ids = bpe_.encode(text);
    for (auto &id : ids) id = id % vocab_size_;
    return ids;
}

std::string K3Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
    std::vector<int> filtered;
    for (int id : ids) {
        if (skip_special && is_special(id)) continue;
        filtered.push_back(id);
    }
    // reuse BPE decode via temporary copy (BPE decode is non-const, so cast)
    return const_cast<BPETokenizer&>(bpe_).decode(filtered);
}

bool K3Tokenizer::is_special(int id) const {
    return special_map_.find(id) != special_map_.end();
}

std::vector<int> K3Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string,std::string>>& messages,
    bool add_generation_prompt) {
    std::string prompt;
    for (auto &m : messages) {
        prompt += "<|im_start|>" + m.first + "\n" + m.second + "<|im_end|>\n";
    }
    if (add_generation_prompt) prompt += "<|im_start|>assistant\n";
    return encode(prompt);
}

} // namespace quant
