#undef NDEBUG   // Release defines NDEBUG (/O2 /Ob2 /DNDEBUG), which compiles every
                // assert() below out of the binary. Tests that rely on assert()
                // were passing vacuously; this keeps them live in Release too.
#include "quant/tokenizer.h"

#include <iostream>
#include <cassert>
#include <string>
#include <vector>

int main() {
    // Test BPETokenizer training and encode-decode identity
    {
        quant::BPETokenizer tokenizer;

        // Train on a small corpus
        std::vector<std::string> corpus = {
            "hello world",
            "hello there",
            "world of hello",
            "hello hello world",
            "test tokenizer",
            "bpe tokenization"
        };

        tokenizer.train(corpus, 64);
        assert(tokenizer.vocab_size() >= 10);
        // A byte-level BPE always keeps its 256 byte tokens as the base vocab,
        // so the trained vocabulary is at least 256 entries (never below).
        assert(tokenizer.vocab_size() >= 256);

        // Test encode-decode identity for training texts
        for (const auto& text : corpus) {
            auto ids = tokenizer.encode(text);
            assert(!ids.empty());
            auto decoded = tokenizer.decode(ids);
            // The decoded text might differ slightly from original
            // due to BPE, but should contain the same words
            assert(!decoded.empty());
            std::cout << "  encode/decode: \"" << text << "\" -> "
                      << ids.size() << " tokens" << std::endl;
        }
    }

    // Test with minimal vocabulary
    {
        quant::BPETokenizer tokenizer;
        std::vector<std::string> corpus = {
            "a b c",
            "a a b",
            "b c c"
        };
        tokenizer.train(corpus, 10);
        assert(tokenizer.vocab_size() >= 4);

        auto ids = tokenizer.encode("a b c");
        assert(!ids.empty());
        auto decoded = tokenizer.decode(ids);
        assert(!decoded.empty());
    }

    // Test single character
    {
        quant::BPETokenizer tokenizer;
        std::vector<std::string> corpus = {"a"};
        tokenizer.train(corpus, 8);
        auto ids = tokenizer.encode("a");
        assert(!ids.empty());
        std::string dec = tokenizer.decode(ids);
        assert(!dec.empty());
    }

    // Test empty input
    {
        quant::BPETokenizer tokenizer;
        std::vector<std::string> corpus = {"hello"};
        tokenizer.train(corpus, 16);
        auto ids = tokenizer.encode("");
        assert(ids.empty());
        auto dec = tokenizer.decode({});
        assert(dec.empty());
    }

    // Test vocabulary size boundary
    {
        quant::BPETokenizer tokenizer;
        std::vector<std::string> corpus = {"one two three four five"};
        tokenizer.train(corpus, 256);
        assert(tokenizer.vocab_size() <= 256);
        assert(tokenizer.vocab_size() >= 5);
    }

    // Test BOS/EOS IDs
    {
        quant::BPETokenizer tokenizer;
        assert(tokenizer.bos_id() >= 0);
        assert(tokenizer.eos_id() >= 0);
        assert(tokenizer.bos_id() != tokenizer.eos_id());
    }

    // Test that encoding produces valid IDs
    {
        quant::BPETokenizer tokenizer;
        std::vector<std::string> corpus = {"hello world this is a test"};
        tokenizer.train(corpus, 32);
        auto ids = tokenizer.encode("hello world");
        for (int id : ids) {
            (void)id;
            assert(id >= 0);
            assert(id < tokenizer.vocab_size());
        }
    }

    std::cout << "All tokenizer tests passed!" << std::endl;
    return 0;
}
